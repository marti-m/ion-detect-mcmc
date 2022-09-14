from PyonDetect.ion.ion import Ion
from PyonDetect.estimator.estimator import EntropyGainEstimator, ThresholdingEstimator, AdaptiveMLEstimator, NonPoissonianAdaptiveMLEstimator, MinExpectedBinsEstimator
from PyonDetect.environment.environment import SimulationEnvironment

import numpy as np
import pandas as pd
import multiprocessing as mp
import time


def run_environment(n_bins, initial_state, R_D, path):
    R_B = 25 / 200 # per us
    e_c = 1e-4
    t_detection = 200
    ions = [Ion(initial_state, R_dark=R_D, R_bright=R_B) for i in range(1)]
    estimator = EntropyGainEstimator(R_D=R_D, R_B=R_B, e_c=e_c, n_subbins_max=n_bins, n_counts_max=10, save_trajectory=False)
    estimators = [estimator]
    env = SimulationEnvironment(ions, estimators, detection_time=t_detection, n_subbins=n_bins, n_repetition=10000)
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
    print("dark, R_D = 0.0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.0/200, len(n_bins))
    path = np.repeat('./outputs/entropy_gain/dark_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.0/200, len(n_bins))
    path = np.repeat('./outputs/entropy_gain/bright_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))


    # dark runs
    print("dark, R_D = 0.5")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/entropy_gain/dark_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/entropy_gain/bright_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))