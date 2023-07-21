from PyonDetect.ion.ion import Ion, BerylliumIon
from PyonDetect.estimator.estimator import EntropyGainEstimator, ThresholdingEstimator, AdaptiveMLEstimator, NonPoissonianAdaptiveMLEstimator, MinExpectedBinsEstimator, HMMMinExpectedBinsEstimator
from PyonDetect.environment.environment import SimulationEnvironment

import numpy as np
import pandas as pd
import multiprocessing as mp
import time

def run_environment(ion, estimator, path_est, initial_state, p_pi_success):
    path_root = './outputs/' + path_est
    if initial_state == 0:
        path_test = 'MAP_adv/bright_R_D_0/'
    else:
        path_test = 'MAP_adv/dark_R_D_0/'
    path = path_root + path_test
    
    n_bins = 5
    t_detection = 200 # us
    ions = [ion]
    estimators = [estimator]
    env = SimulationEnvironment(ions, estimators, detection_time=t_detection, n_subbins=n_bins, p_pi_success=p_pi_success, n_repetition=20000)
    env.run()
    env.save_to_csv(path)

if __name__ == "__main__":
    mp.freeze_support() # add this line on PCs running a Windows OS

    sec = time.time()
    print(time.ctime(sec))

    # INIT
    n_bins = 5


    R_B = 50.0 / 200 # per us
    R_D = 0.0 / 200 # per us
    e_c = 0.0
    t_detection = 200 # us
    p_pi_success = 0.9

    path_est = ['entropy_gain/', 'aml/', 'minbins/', 'hmm_no_exp/']


    estimators = [EntropyGainEstimator(R_D=R_D, R_B=R_B, e_c=e_c, n_subbins_max=n_bins, n_counts_max=10),
                AdaptiveMLEstimator(R_D=R_D, R_B=R_B, e_c=e_c, n_subbins_max=n_bins),
                MinExpectedBinsEstimator(R_D=R_D, R_B=R_B, e_c=e_c, n_subbins_max=n_bins),
                HMMMinExpectedBinsEstimator(R_D=R_D, R_B=R_B, tau_db = 1.1687 * 1e6, tau_bd = (200 / 0.01) * 1e6, e_c=e_c, t_det=t_detection, n_subbins_max=n_bins, exp_threshold=False, p_pi_success=p_pi_success)]

    p_success = np.repeat(p_pi_success, len(estimators))

     # bright runs
    print("bright, R_D = 0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(estimators))
    ions = [BerylliumIon(0, R_dark=R_D) for i in range(len(estimators))]
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(ions, estimators, path_est, initial_state, p_success))


    # dark runs
    print("dark, R_D = 0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(estimators))
    ions = [BerylliumIon(1, R_dark=R_D) for i in range(len(estimators))]
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(ions, estimators, path_est, initial_state, p_success))