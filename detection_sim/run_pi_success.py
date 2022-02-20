from PyonDetect.ion.ion import Ion
from PyonDetect.estimator.estimator import EntropyGainEstimator, ThresholdingEstimator, AdaptiveMLEstimator, NonPoissonianAdaptiveMLEstimator, MinExpectedBinsEstimator
from PyonDetect.environment.environment import SimulationEnvironment

import numpy as np
import pandas as pd
import multiprocessing as mp
import time


def run_environment_EG(initial_state, p_success, path):
    n_bins = 20
    R_B = 25 / 200 # per us
    R_D = 0.1 / 200 # per us
    e_c = 1e-4
    t_detection = 200
    ion = Ion(initial_state, R_dark=R_D, R_bright=R_B)
    estimator = EntropyGainEstimator(R_D=R_D, R_B=R_B, e_c=e_c, n_subbins_max=n_bins, n_counts_max=10, save_trajectory=True)
    env = SimulationEnvironment([ion], [estimator], detection_time=t_detection, n_subbins=n_bins, p_pi_success=p_success, n_repetition=100000)
    env.run()
    env.save_to_csv(path)
    env.save_to_json(path)

def run_environment_MB(initial_state, p_success, path):
    n_bins = 20
    R_B = 25 / 200 # per us
    R_D = 0.1 / 200 # per us
    e_c = 1e-4
    t_detection = 200
    ion = Ion(initial_state, R_dark=R_D, R_bright=R_B)
    estimator = MinExpectedBinsEstimator(R_D=R_D, R_B=R_B, e_c=e_c, n_subbins_max=n_bins, save_trajectory=True)
    env = SimulationEnvironment([ion], [estimator], detection_time=t_detection, n_subbins=n_bins, p_pi_success=p_success, n_repetition=100000)
    env.run()
    env.save_to_csv(path)
    env.save_to_json(path)

if __name__ == "__main__":
    mp.freeze_support() # add this line on PCs running a Windows OS

    sec = time.time()
    print(time.ctime(sec))

    # INIT
    p_success = np.arange(0.995, 1, 0.0005)


    # dark runs
    print("dark entropy gain")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(p_success))
    path = np.repeat('./outputs/EG/dark_success_test/', len(p_success))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_EG, zip(initial_state, p_success, path))


    #bright runs
    print("bright entropy gain")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(p_success))
    path = np.repeat('./outputs/EG/bright_success_test/', len(p_success))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_EG, zip(initial_state, p_success, path))

    # dark runs
    print("dark min bins")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(p_success))
    path = np.repeat('./outputs/MB/dark_success_test/', len(p_success))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_MB, zip(initial_state, p_success, path))


    #bright runs
    print("bright min bins")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(p_success))
    path = np.repeat('./outputs/MB/bright_success_test/', len(p_success))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_MB, zip(initial_state, p_success, path))