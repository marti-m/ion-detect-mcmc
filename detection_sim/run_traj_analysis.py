from PyonDetect.ion.ion import Ion, BerylliumIon
from PyonDetect.estimator.estimator import EntropyGainEstimator, ThresholdingEstimator, AdaptiveMLEstimator, NonPoissonianAdaptiveMLEstimator, MinExpectedBinsEstimator, HMMMinExpectedBinsEstimator
from PyonDetect.environment.environment import SimulationEnvironment

import numpy as np
import pandas as pd
import multiprocessing as mp
import time

def run_environment_traj_analysis(estimator, initial_state, path_estimator):
    path_root = './outputs/'
    if initial_state == 0:
        path_test = 'traj_analysis/bright/'
    else:
        path_test = 'traj_analysis/dark/'
    n_bins = 20

    R_B = 25.0 / 200 # per us
    R_D = 0.0 / 200 # per us
    e_c = 1e-4
    t_detection = 200 # us

    ions = [Ion(initial_state, R_dark=R_D, R_bright=R_B) for i in range(1)]
    estimators = [estimator]
    env = SimulationEnvironment(ions, estimators, detection_time=t_detection, n_subbins=n_bins, n_repetition=20000)
    env.run()
    env.save_to_csv(path_root + path_estimator + path_test)
    env.save_to_json(path_root + path_estimator + path_test)



if __name__ == "__main__":
    mp.freeze_support() # add this line on PCs running a Windows OS

    sec = time.time()
    print(time.ctime(sec))

    # INIT
    n_bins = 20
    p_success = 1

    R_B = 25.0 / 200 # per us
    R_D = 0.0 / 200 # per us
    e_c = 1e-4
    t_detection = 200 # us

    path_est = ['entropy_gain/', 'aml/', 'minbins/', 'hmm/']


    estimators = [EntropyGainEstimator(R_D=R_D, R_B=R_B, e_c=e_c, n_subbins_max=n_bins, n_counts_max=10, save_trajectory=True),
                AdaptiveMLEstimator(R_D=R_D, R_B=R_B, e_c=e_c, n_subbins_max=n_bins, save_trajectory=True),
                MinExpectedBinsEstimator(R_D=R_D, R_B=R_B, e_c=e_c, n_subbins_max=n_bins, save_trajectory=True),
                HMMMinExpectedBinsEstimator(R_D=R_D, R_B=R_B, tau_db = 1e30, tau_bd = 1e30, e_c=e_c, t_det=t_detection, n_subbins_max=n_bins, save_trajectory=True)]

    # bright runs
    print("bright, R_D = 0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(estimators))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_traj_analysis, zip(estimators, initial_state, path_est))


    # dark runs
    print("dark, R_D = 0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(estimators))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_traj_analysis, zip(estimators, initial_state, path_est))

    