from PyonDetect.ion.ion import BerylliumIon
from PyonDetect.estimator.estimator import EntropyGainEstimator, ThresholdingEstimator, AdaptiveMLEstimator, NonPoissonianAdaptiveMLEstimator, MinExpectedBinsEstimator, HMMMinExpectedBinsEstimator
from PyonDetect.environment.environment import SimulationEnvironment

import numpy as np
import pandas as pd
import multiprocessing as mp
import time

def run_environment(n_bins, initial_state, R_D, path):
    e_c = 1 - 0.996
    t_detection = 200
    R_B = 26.4 / 200
    ion = BerylliumIon(initial_state, R_dark=R_D)
    estimator = HMMMinExpectedBinsEstimator(R_D=R_D, R_B=R_B, tau_db = (1 / 78.46) * 1e6, tau_bd = (1 / 3.43e2) * 1e6, e_c=e_c, t_det=t_detection, n_subbins_max=n_bins, save_trajectory=False)
    env = SimulationEnvironment([ion], [estimator], detection_time=t_detection, n_subbins=n_bins, n_repetition=100)
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
    path = np.repeat('./outputs/msg_pass/dark_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.0, len(n_bins))
    path = np.repeat('./outputs/msg_pass/bright_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))


    # dark runs
    print("dark, R_D = 0.5")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/msg_pass/dark_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/msg_pass/bright_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment, zip(n_bins, initial_state, R_dark, path))
