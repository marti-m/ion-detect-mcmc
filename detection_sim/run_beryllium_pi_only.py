from PyonDetect.ion.ion import BerylliumIon
from PyonDetect.estimator.estimator import EntropyGainEstimator, ThresholdingEstimator, AdaptiveMLEstimator, NonPoissonianAdaptiveMLEstimator, MinExpectedBinsEstimator, HMMMinExpectedBinsEstimator
from PyonDetect.environment.environment import SimulationEnvironment

import numpy as np
import pandas as pd
import multiprocessing as mp
import time

def run_environment(n_bins, initial_state, R_D, path):
    e_c = 1e-4
    t_detection = 200
    R_B = 26.4 / 200
    ions = [BerylliumIon(initial_state, R_dark=R_D) for i in range(3)]
    estimator1 = MinExpectedBinsEstimator(R_D=R_D, R_B=R_B, e_c=e_c, n_subbins_max=n_bins, save_trajectory=False)
    estimator2 = EntropyGainEstimator(R_D=R_D, R_B=R_B, e_c=e_c, n_subbins_max=n_bins, n_counts_max=10, save_trajectory=False)
    estimator3 = HMMMinExpectedBinsEstimator(R_D=R_D, R_B=R_B, tau_db = (1 / 78.46) * 1e6, tau_bd = (1 / 3.43e2) * 1e6, e_c=(1 - 0.9965), t_det=t_detection, n_subbins_max=n_bins, save_trajectory=False)
    estimators = [estimator1, estimator2, estimator3]
    env = SimulationEnvironment(ions, estimators, detection_time=t_detection, n_subbins=n_bins, n_repetition=20000)
    env.run()
    env.save_to_csv(path)

if __name__ == "__main__":
    mp.freeze_support() # add this line on PCs running a Windows OS

    sec = time.time()
    print(time.ctime(sec))

    # INIT
    max_bins = 36
    min_bins = 5

    n_bins = np.arange(min_bins, max_bins)


    # dark runs
    print("dark, R_D = 0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.0, len(n_bins))
    path = np.repeat('./outputs/beryllium_test_pi_only/dark_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.0, len(n_bins))
    path = np.repeat('./outputs/beryllium_test_pi_only/bright_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))


    # dark runs
    print("dark, R_D = 0.1")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.1/200, len(n_bins))
    path = np.repeat('./outputs/beryllium_test_pi_only/dark_R_D_0_1/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.1/200, len(n_bins))
    path = np.repeat('./outputs/beryllium_test_pi_only/bright_R_D_0_1/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))


    # dark runs
    print("dark, R_D = 0.5")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/beryllium_test_pi_only/dark_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/beryllium_test_pi_only/bright_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))
