import numpy as np
import pandas as pd
import time
import json

class SimulationEnvironment():
    
    def __init__(self, ion_list, estimator_list, detection_time=200, n_subbins = 5, n_repetition=100, p_pi_success=1, verbose=False):
        assert len(ion_list) != 0, "Need to pass more than one ion to the environment"
        assert len(estimator_list) != 0, "Need to pass more than estimator to the environment"
        assert len(ion_list) == len(estimator_list), "Ions and estimators need to be paired in the list!"
        
        self.__ions = ion_list
        self.__estimators = estimator_list
        
        self.__ions_unique = self.__unique_instances(self.__ions)
        self.__estimators_unique = self.__unique_instances(self.__estimators)
        
        self.__verbose = verbose
        
        if self.__verbose:
            print(self.__ions, self.__ions_unique, self.__estimators, self.__estimators_unique)
            
        # build result dictionary
        self.__stats = {}
        self.__trajectories = {}
        
        # histogram array
        self._hist_arr = None
        
        for i, estimator in enumerate(self.__estimators_unique):
            stats_i = estimator.get_stats()
            if self.__verbose:
                print(stats_i)
            for key in stats_i.keys():
                self.__stats[key + "_" + str(i)] = np.array([])
                
            trajectories_i = estimator.get_trajectory()
            if trajectories_i is not None:
                for key in trajectories_i.keys():
                    self.__trajectories[key + "_" + str(i)] = []
        
        self.n_repetition = n_repetition
        
        self.subbin_time = detection_time / n_subbins # in us
        self.n_subbins = n_subbins
        
        self.ready = False
        self.__p_pi_pulse = p_pi_success
        
    def __unique_instances(self, list_instances):
        unique_instances = []
        for inst in list_instances:
            if inst in unique_instances:
                continue
            else:
                unique_instances.append(inst)
            
        return unique_instances
    
    def get_if_ready(self):
        return self.ready
        
    def single_shot(self):
        # go over all subbins
        for i in range(self.n_subbins):
            # update ions (only unique instances, as ions and estimators can be paired arbitrarily)
            for ion in self.__ions_unique:
                ion.update_ion(self.subbin_time, self.subbin_time / 1000)
            
            # distrubite photon counts among an estimator/ion pair
            for ion, estimator in zip (self.__ions, self.__estimators):
                if not estimator.get_if_ready():
                    n_photons = ion.get_photon_counts()
                    estimator.update_subbin_photon_counts(n_photons)
            
            # update esimators (only unique estimators need to be updated!) and check if they are done when using an adaptive detection scheme
            all_ready = True
            for estimator in self.__estimators_unique:
                estimator.update_prediction(self.subbin_time)
                if estimator.get_if_ready() == False:
                    all_ready = False
                    
            # if all estimators are ready, readback results and reset estimator
            if all_ready:
                break
                    
            # check if control action needs to be done
            # if multiple estimators request a Pi pulse, all will be applied on all ions!!!
            # TO DO: change how to control ions if multiple estimators want to do so.
            for ion, estimator in zip (self.__ions, self.__estimators):
                pi_pulse = estimator.get_is_pi_pulse_needed()
                if pi_pulse:
                    success = ion.pi_pulse(p_success = self.__p_pi_pulse)
                    estimator.pi_pulse_applied(success)
        
        # readout estimators and reset
        for i, estimator in enumerate(self.__estimators_unique):
            stats_i = estimator.get_stats()
            for key in stats_i.keys():
                self.__stats[key + "_" + str(i)] = np.append(self.__stats[key + "_" + str(i)], stats_i[key])
                
                
            trajectories_i = estimator.get_trajectory()
            if trajectories_i is not None:
                for key in trajectories_i.keys():
                    self.__trajectories[key + "_" + str(i)].append(trajectories_i[key])
                    
            estimator.reset()
        # also reset the ion (re-initialization)    
        for ion in self.__ions_unique:
            ion.reset()
            
    def run(self):
        for i in range(self.n_repetition):
            self.single_shot()
        
        self._bin_trajectories()
        self.ready = True
        
    def _bin_trajectories(self):
        if 'n_photons_subbin_0' in self.__trajectories.keys():
            # the subbin photon count and Pi pulse success indicator should uniquely determine trajectories,
            # as the estimator is deterministic with respect to the photon counts
            trj_photons = self.__trajectories['n_photons_subbin_0']
            if 'pulse_success_0' in self.__trajectories.keys():
                trj_pi_success = self.__trajectories['pulse_success_0']
                # hash the numpy arrays of the subbin photon count to get a unique identifier
                trj_hash = np.array([hash(np.append(trj_photon_i, trj_pi_success_i).tobytes()) for trj_photon_i, trj_pi_success_i in zip(trj_photons, trj_pi_success)])
            else:
                trj_hash = np.array([hash(trj_photon_i) for trj_photon_i in trj_photons])


            # get unique hashes and counts, and sort them
            unique, counts = np.unique(trj_hash, return_counts=True)
            # get example index of trajectory for each unique value
            ex_idx = np.array([np.min(np.argwhere(trj_hash == hash_i)) for hash_i in unique])
            hist_arr = np.column_stack((counts, unique, ex_idx))
            # sort by frequency
            arg_sort = np.flip(np.argsort(hist_arr[:, 0]))
            self._hist_arr = hist_arr[arg_sort, :]
        
    def get_stats(self):
        return self.__stats
    
    def get_trajectories(self):
        return self.__trajectories
    
    def save_to_csv(self, path=None):
        if self.ready:
            df = pd.DataFrame(data = self.__stats)
            if self.__verbose:
                print(df)
            
            if path is None:
                # save to local directory
                path = "./"
                
            filename = f"n_subbins={self.n_subbins}_p_pi_suc={self.__p_pi_pulse:.5f}.csv"
            df.to_csv(path + filename)        
        else:
            print("The simulation environment has not terminated")
            
    def save_to_json(self, path=None):
        assert self._hist_arr is not None, "Environment has not finished yet"
        dict_tot = {}
        for i, idx in enumerate(self._hist_arr[:, 2]):
            dict_i = {}
            dict_i['probability'] = self._hist_arr[i, 0] / self.n_repetition
            for key in self.__trajectories.keys():
                dict_i[key] = self.__trajectories[key][idx].tolist()
            dict_tot[str(i)] = dict_i
            
        if path is None:
                # save to local directory
                path = "./"
        filename = f"n_subbins={self.n_subbins}_p_pi_suc={self.__p_pi_pulse:.5f}.json"

        with open(path + filename, 'w') as f:
            json.dump(json.dumps(dict_tot, indent=1), f)
        
            
    def set_p_pi_pulse(self, p_pi_pulse):
        self.__p_pi_pulse = p_pi_pulse
        
        
            
        
            
        
        
        
    
