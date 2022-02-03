from PyonDetect.ion.ion import BerylliumIon, CalciumIon
from PyonDetect.estimator.estimator import EntropyGainEstimator, ThresholdingEstimator, AdaptiveMLEstimator, NonPoissonianAdaptiveMLEstimator
from PyonDetect.environment.environment import SimulationEnvironment

import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import time


def run_multiple_environments(n_bins, initial_state, R_dark = 0):
    ion1 = CalciumIon(initial_state = initial_state, R_dark = R_dark / 200, R_bright=25/200, tau_bd=np.finfo(float).max, tau_db=np.finfo(float).max)
    ion2 = CalciumIon(initial_state = initial_state, R_dark = R_dark / 200, R_bright=25/200, tau_bd=np.finfo(float).max, tau_db=np.finfo(float).max)
    estimator1 = AdaptiveMLEstimator(R_D = R_dark / 200, R_B = 25/200, e_c=0.0001, n_subbins_max=n_bins)
    estimator2 = EntropyGainEstimator(R_D = R_dark / 200, R_B = 25/200, e_c= 0.0001, n_subbins_max=n_bins, n_counts_max=10)
    env = SimulationEnvironment([ion1, ion2], [estimator1, estimator2], detection_time=200, n_subbins = n_bins, n_repetition=10000)
    env.repeat_single_shots()
    return env.get_stats()


if __name__ == "__main__":
    mp.freeze_support() # add this line on PCs running a Windows OS

    sec = time.time()
    print(time.ctime(sec))

    # INIT
    max_bins = 50
    min_bins = 5

    n_bins = np.arange(min_bins, max_bins)

    det_mean = np.zeros(max_bins-min_bins)
    det_t_mean = np.zeros(max_bins-min_bins)
    det_n_mean = np.zeros(max_bins-min_bins)
    det_mean_std = np.zeros(max_bins-min_bins)
    det_t_mean_std = np.zeros(max_bins-min_bins)

    det_mean_noPi= np.zeros(max_bins-min_bins)
    det_t_mean_noPi = np.zeros(max_bins-min_bins)
    det_n_mean_noPi = np.zeros(max_bins-min_bins)
    det_mean_std_noPi = np.zeros(max_bins-min_bins)
    det_t_mean_std_noPi = np.zeros(max_bins-min_bins)

    det_mean_dark = np.zeros(max_bins-min_bins)
    det_t_mean_dark = np.zeros(max_bins-min_bins)
    det_n_mean_dark = np.zeros(max_bins-min_bins)
    det_mean_std_dark = np.zeros(max_bins-min_bins)
    det_t_mean_std_dark = np.zeros(max_bins-min_bins)

    det_mean_noPi_dark= np.zeros(max_bins-min_bins)
    det_t_mean_noPi_dark = np.zeros(max_bins-min_bins)
    det_n_mean_noPi_dark = np.zeros(max_bins-min_bins)
    det_mean_std_noPi_dark = np.zeros(max_bins-min_bins)
    det_t_mean_std_noPi_dark = np.zeros(max_bins-min_bins)

    # dark runs
    print("dark")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0, len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_multiple_environments, zip(n_bins, initial_state, R_dark))


    for bins in n_bins:
        index = bins - min_bins
        result = res[index]
        det_mean_noPi_dark[index] = np.mean(result['prediction_0'])
        det_t_mean_noPi_dark[index] = np.mean(result['n_subbins_0']) * (200/bins)
        det_n_mean_noPi_dark[index] = 0 # no pulses can be applied

        det_mean_dark[index] =  np.mean(result['prediction_1'])
        det_t_mean_dark[index] = np.mean(result['n_subbins_1']) * (200/bins)
        det_n_mean_dark[index] = np.mean(result['n_pi_pulses_1'])

    #bright runs
    print("bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0, len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_multiple_environments, zip(n_bins, initial_state, R_dark))

    for bins in n_bins:
        index = bins - min_bins
        result = res[index]
        det_mean_noPi[bins-min_bins] = np.mean(result['prediction_0'])
        det_t_mean_noPi[bins-min_bins] = np.mean(result['n_subbins_0']) * (200/bins)
        det_n_mean_noPi[bins-min_bins] = 0 # no pulses can be applied

        det_mean[bins-min_bins] =  np.mean(result['prediction_1'])
        det_t_mean[bins-min_bins] = np.mean(result['n_subbins_1']) * (200/bins)
        det_n_mean[bins-min_bins] = np.mean(result['n_pi_pulses_1'])

    
    result = {  'n_bins': n_bins,
                'det_mean_noPi_dark': det_mean_noPi_dark,
                'det_t_mean_noPi_dark': det_t_mean_noPi_dark,
                'det_mean_dark': det_mean_dark,
                'det_t_mean_dark': det_t_mean_dark,
                'det_n_mean_dark': det_n_mean_dark,
                'det_mean_noPi': det_mean_noPi,
                'det_t_mean_noPi': det_t_mean_noPi,
                'det_mean': det_mean,
                'det_t_mean': det_t_mean,
                'det_n_mean': det_n_mean
                }
    df = pd.DataFrame(data = result)
    df.to_csv("./output/R_D=0_AMLE_EntropyGain.csv")


     # INIT
    max_bins = 50
    min_bins = 5

    n_bins = np.arange(min_bins, max_bins)

    det_mean = np.zeros(max_bins-min_bins)
    det_t_mean = np.zeros(max_bins-min_bins)
    det_n_mean = np.zeros(max_bins-min_bins)
    det_mean_std = np.zeros(max_bins-min_bins)
    det_t_mean_std = np.zeros(max_bins-min_bins)

    det_mean_noPi= np.zeros(max_bins-min_bins)
    det_t_mean_noPi = np.zeros(max_bins-min_bins)
    det_n_mean_noPi = np.zeros(max_bins-min_bins)
    det_mean_std_noPi = np.zeros(max_bins-min_bins)
    det_t_mean_std_noPi = np.zeros(max_bins-min_bins)

    det_mean_dark = np.zeros(max_bins-min_bins)
    det_t_mean_dark = np.zeros(max_bins-min_bins)
    det_n_mean_dark = np.zeros(max_bins-min_bins)
    det_mean_std_dark = np.zeros(max_bins-min_bins)
    det_t_mean_std_dark = np.zeros(max_bins-min_bins)

    det_mean_noPi_dark= np.zeros(max_bins-min_bins)
    det_t_mean_noPi_dark = np.zeros(max_bins-min_bins)
    det_n_mean_noPi_dark = np.zeros(max_bins-min_bins)
    det_mean_std_noPi_dark = np.zeros(max_bins-min_bins)
    det_t_mean_std_noPi_dark = np.zeros(max_bins-min_bins)

    # dark runs
    print("dark")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(1, len(n_bins))
    R_dark = np.repeat(0.1, len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_multiple_environments, zip(n_bins, initial_state, R_dark))


    for bins in n_bins:
        index = bins - min_bins
        result = res[index]
        det_mean_noPi_dark[index] = np.mean(result['prediction_0'])
        det_t_mean_noPi_dark[index] = np.mean(result['n_subbins_0']) * (200/bins)
        det_n_mean_noPi_dark[index] = 0 # no pulses can be applied

        det_mean_dark[index] =  np.mean(result['prediction_1'])
        det_t_mean_dark[index] = np.mean(result['n_subbins_1']) * (200/bins)
        det_n_mean_dark[index] = np.mean(result['n_pi_pulses_1'])

    #bright runs
    print("bright")
    sec = time.time()
    print(time.ctime(sec))

    initial_state = np.repeat(0, len(n_bins))
    R_dark = np.repeat(0.1, len(n_bins))
    with mp.Pool(mp.cpu_count()) as p:
        res = p.starmap(run_multiple_environments, zip(n_bins, initial_state, R_dark))

    for bins in n_bins:
        index = bins - min_bins
        result = res[index]
        det_mean_noPi[bins-min_bins] = np.mean(result['prediction_0'])
        det_t_mean_noPi[bins-min_bins] = np.mean(result['n_subbins_0']) * (200/bins)
        det_n_mean_noPi[bins-min_bins] = 0 # no pulses can be applied

        det_mean[bins-min_bins] =  np.mean(result['prediction_1'])
        det_t_mean[bins-min_bins] = np.mean(result['n_subbins_1']) * (200/bins)
        det_n_mean[bins-min_bins] = np.mean(result['n_pi_pulses_1'])

    
    result = {  'n_bins': n_bins,
                'det_mean_noPi_dark': det_mean_noPi_dark,
                'det_t_mean_noPi_dark': det_t_mean_noPi_dark,
                'det_mean_dark': det_mean_dark,
                'det_t_mean_dark': det_t_mean_dark,
                'det_n_mean_dark': det_n_mean_dark,
                'det_mean_noPi': det_mean_noPi,
                'det_t_mean_noPi': det_t_mean_noPi,
                'det_mean': det_mean,
                'det_t_mean': det_t_mean,
                'det_n_mean': det_n_mean
                }
    df = pd.DataFrame(data = result)
    df.to_csv("./output/R_D=0e-1_AMLE_EntropyGain.csv")
    
    plt.plot(range(min_bins, max_bins), det_t_mean, label='EntropyGain bright')
    plt.plot(range(min_bins, max_bins), det_t_mean_noPi, label='AMLE bright')
    plt.plot(range(min_bins, max_bins), det_t_mean_dark, label = 'EntropyGain dark')
    plt.plot(range(min_bins, max_bins), det_t_mean_noPi_dark, label = 'AMLE dark')
    plt.legend()
    plt.show()