from PyonDetect.ion.ion import Ion, BerylliumIon
from PyonDetect.estimator.estimator import EntropyGainEstimator, ThresholdingEstimator, AdaptiveMLEstimator, NonPoissonianAdaptiveMLEstimator, MinExpectedBinsEstimator, HMMMinExpectedBinsEstimator
from PyonDetect.environment.environment import SimulationEnvironment

import numpy as np
import pandas as pd
import multiprocessing as mp
import time

def run_environment_ideal_hmm(n_bins, initial_state, R_D, path):
    R_B = 25 / 200 # per us
    e_c = 1e-4
    t_detection = 200
    ions = [Ion(initial_state, R_dark=R_D, R_bright=R_B) for i in range(1)]
    estimator = HMMMinExpectedBinsEstimator(R_D=R_D, R_B=R_B, tau_db = 1e30, tau_bd = 1e30, e_c=e_c, t_det=t_detection, n_subbins_max=n_bins, save_trajectory=False)
    estimators = [estimator]
    env = SimulationEnvironment(ions, estimators, detection_time=t_detection, n_subbins=n_bins, n_repetition=20000)
    env.run()
    env.save_to_csv(path)

def run_environment_beryllium_hmm(n_bins, initial_state, R_D, path):
    e_c = 1 - 0.996
    t_detection = 200
    R_B = 26.4 / 200
    ions = [BerylliumIon(initial_state, R_dark=R_D) for i in range(1)]
    estimator = HMMMinExpectedBinsEstimator(R_D=R_D, R_B=R_B, tau_db = (1 / 78.46) * 1e6, tau_bd = (1 / 3.43e2) * 1e6, e_c=e_c, t_det=t_detection, n_subbins_max=n_bins, save_trajectory=False)
    estimators = [estimator]
    env = SimulationEnvironment(ions, estimators, detection_time=t_detection, n_subbins=n_bins, n_repetition=20000)
    env.run()
    env.save_to_csv(path)

def run_environment_pi_success_hmm(initial_state, p_success, R_D, path):
    n_bins = 20
    R_B = 25 / 200 # per us
    e_c = 1e-4
    t_detection = 200
    ion = Ion(initial_state, R_dark=R_D, R_bright=R_B)
    estimator = HMMMinExpectedBinsEstimator(R_D=R_D, R_B=R_B, tau_db = 1e30, tau_bd = 1e30, e_c=e_c, p_pi_success=p_success, t_det=t_detection, n_subbins_max=n_bins, save_trajectory=False)
    env = SimulationEnvironment([ion], [estimator], detection_time=t_detection, n_subbins=n_bins, p_pi_success=p_success, n_repetition=100000)
    env.run()
    env.save_to_csv(path)
    #env.save_to_json(path)

def run_environment_ideal_hmm_no_exp(n_bins, initial_state, R_D, path):
    R_B = 25 / 200 # per us
    e_c = 1e-4
    t_detection = 200
    ions = [Ion(initial_state, R_dark=R_D, R_bright=R_B) for i in range(1)]
    estimator = HMMMinExpectedBinsEstimator(R_D=R_D, R_B=R_B, tau_db = 1e30, tau_bd = 1e30, e_c=e_c, t_det=t_detection, n_subbins_max=n_bins, exp_threshold=False, save_trajectory=False)
    estimators = [estimator]
    env = SimulationEnvironment(ions, estimators, detection_time=t_detection, n_subbins=n_bins, n_repetition=20000)
    env.run()
    env.save_to_csv(path)

def run_environment_beryllium_hmm_no_exp(n_bins, initial_state, R_D, path):
    e_c = 1e-4
    t_detection = 200
    R_B = 26.4 / 200
    ions = [BerylliumIon(initial_state, R_dark=R_D) for i in range(1)]
    estimator = HMMMinExpectedBinsEstimator(R_D=R_D, R_B=R_B, tau_db = (1 / 78.46) * 1e6, tau_bd = (1 / 3.43e2) * 1e6, e_c=e_c, t_det=t_detection, n_subbins_max=n_bins, exp_threshold=False, save_trajectory=False)
    estimators = [estimator]
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
    p_success = np.arange(0.995, 1, 0.0005)
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------HMM---------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # ---------------------------------------------ideal--------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # dark runs
    """
    print("ideal, HMM, dark, R_D = 0.0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.0/200, len(n_bins))
    path = np.repeat('./outputs/hmm/ideal_bk/dark_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_ideal_hmm, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("ideal, HMM, bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.0/200, len(n_bins))
    path = np.repeat('./outputs/hmm/ideal_bk/bright_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_ideal_hmm, zip(n_bins, initial_state, R_dark, path))


    # dark runs
    print("ideal, HMM, dark, R_D = 0.5")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/hmm/ideal_bk/dark_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_ideal_hmm, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("ideal, HMM, bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/hmm/ideal_bk/bright_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_ideal_hmm, zip(n_bins, initial_state, R_dark, path))
    """
    # ----------------------------------------------------------------------------------------------------
    # ---------------------------------------------Beryllium----------------------------------------------
    # ----------------------------------------------------------------------------------------------------

    # dark runs
    print("Beryllium, HMM, dark, R_D = 0.0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.0/200, len(n_bins))
    path = np.repeat('./outputs/hmm/beryllium/dark_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_beryllium_hmm, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("Beryllium, HMM, bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.0/200, len(n_bins))
    path = np.repeat('./outputs/hmm/beryllium/bright_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_beryllium_hmm, zip(n_bins, initial_state, R_dark, path))


    # dark runs
    print("Beryllium, HMM, dark, R_D = 0.5")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/hmm/beryllium/dark_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_beryllium_hmm, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("Beryllium, hmm, bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/hmm/beryllium/bright_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_beryllium_hmm, zip(n_bins, initial_state, R_dark, path))


    # ----------------------------------------------------------------------------------------------------
    # ---------------------------------------------Pi success --------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    """
    # dark runs
    print("pi_success, HMM, dark, R_D = 0.0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(p_success))
    R_dark = np.repeat(0.0/200, len(p_success))
    path = np.repeat('./outputs/hmm/pi_success/dark_R_D_0/', len(p_success))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_pi_success_hmm, zip(initial_state, p_success, R_dark, path))

    # bright runs
    print("pi_success, HMM, bright, R_D = 0.0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(p_success))
    R_dark = np.repeat(0.0/200, len(p_success))
    path = np.repeat('./outputs/hmm/pi_success/bright_R_D_0/', len(p_success))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_pi_success_hmm, zip(initial_state, p_success, R_dark, path))

    # dark runs
    print("pi_success, HMM, dark, R_D = 0.5")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(p_success))
    R_dark = np.repeat(0.5/200, len(p_success))
    path = np.repeat('./outputs/hmm/pi_success/dark_R_D_0_5/', len(p_success))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_pi_success_hmm, zip(initial_state, p_success, R_dark, path))

    # bright runs
    print("pi_success, HMM, bright, R_D = 0.5")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(p_success))
    R_dark = np.repeat(0.5/200, len(p_success))
    path = np.repeat('./outputs/hmm/pi_success/bright_R_D_0_5/', len(p_success))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_pi_success_hmm, zip(initial_state, p_success, R_dark, path))

    """
    # ----------------------------------------------------------------------------------------------------
    # -------------------------------------------HMM no_exp-----------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # ---------------------------------------------ideal--------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # dark runs
    """
    print("ideal, HMM, dark, R_D = 0.0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.0/200, len(n_bins))
    path = np.repeat('./outputs/hmm_no_exp/ideal_bk/dark_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_ideal_hmm_no_exp, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("ideal, HMM, bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.0/200, len(n_bins))
    path = np.repeat('./outputs/hmm_no_exp/ideal_bk/bright_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_ideal_hmm_no_exp, zip(n_bins, initial_state, R_dark, path))


    # dark runs
    print("ideal, HMM, dark, R_D = 0.5")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/hmm_no_exp/ideal_bk/dark_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_ideal_hmm_no_exp, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("ideal, HMM, bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/hmm_no_exp/ideal_bk/bright_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_ideal_hmm_no_exp, zip(n_bins, initial_state, R_dark, path))

    # ----------------------------------------------------------------------------------------------------
    # ---------------------------------------------Beryllium----------------------------------------------
    # ----------------------------------------------------------------------------------------------------

    # dark runs
    print("Beryllium, HMM, dark, R_D = 0.0")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.0/200, len(n_bins))
    path = np.repeat('./outputs/hmm_no_exp/beryllium/dark_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_beryllium_hmm_no_exp, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("Beryllium, HMM, bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.0/200, len(n_bins))
    path = np.repeat('./outputs/hmm_no_exp/beryllium/bright_R_D_0/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_beryllium_hmm_no_exp, zip(n_bins, initial_state, R_dark, path))


    # dark runs
    print("Beryllium, HMM, dark, R_D = 0.5")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/hmm_no_exp/beryllium/dark_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_beryllium_hmm_no_exp, zip(n_bins, initial_state, R_dark, path))


    #bright runs
    print("Beryllium, hmm, bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.5/200, len(n_bins))
    path = np.repeat('./outputs/hmm_no_exp/beryllium/bright_R_D_0_5/', len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_environment_beryllium_hmm_no_exp, zip(n_bins, initial_state, R_dark, path))
"""