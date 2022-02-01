import numpy as np
import pandas as pd

class SimulationEnvironment():
    
    def __init__(self, ion_list, estimator_list, detection_time=200, n_subbins = 5, n_repetition=100, verbose=False):
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
        
        for i, estimator in enumerate(self.__estimators_unique):
            stats_i = estimator.get_stats()
            if verbose:
                print(stats_i)
            for key in stats_i.keys():
                self.__stats[key + "_" + str(i)] = np.array([])
        
        self.n_repetition = n_repetition
        
        self.subbin_time = detection_time / n_subbins # in us
        self.n_subbins = n_subbins
        
        self.ready = False
        self.__p_pi_pulse = 1
        
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
                    ion.pi_pulse(p_success = self.__p_pi_pulse)
                    estimator.pi_pulse_applied()
        
        # readout estimators and reset
        for i, estimator in enumerate(self.__estimators_unique):
            stats_i = estimator.get_stats()
            for key in stats_i.keys():
                self.__stats[key + "_" + str(i)] = np.append(self.__stats[key + "_" + str(i)], stats_i[key])
                estimator.reset()
                
        # also reset the ion (re-initialization)    
        for ion in self.__ions_unique:
            ion.reset()
            
    def repeat_single_shots(self):
        for i in range(self.n_repetition):
            self.single_shot()
        
        self.ready = True
        
    def get_stats(self):
        return self.__stats
    
    def save_to_csv(self, path=None):
        if self.ready:
            df = pd.DataFrame(data = self.__stats)
            if self.__verbose:
                print(df)
            
            if path is None:
                # save to local directory
                path = "./"
                
            filename = f"t_subbin={self.subbin_time}_n_subbins={self.n_subbins}.csv"
            df.to_csv(path + filename)
                
            
            
        else:
            print("The simulation environment has not terminated")
            
    def set_p_pi_pulse(self, p_pi_pulse):
        self.__p_pi_pulse = p_pi_pulse
        
        
            
        
            
        
        
        
    
