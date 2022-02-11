import random
import numpy as np

class Ion:
    def __init__(self, initial_state, R_dark, R_bright):
        self.state = initial_state
        self.init_state = initial_state
        self.R_dark = R_dark
        self.R_bright= R_bright
        self.poisson_mu = 0   
        self.n_photons_after_update = 0
        

    def get_photon_counts(self, verbose=False):
        if verbose:
            print(f"Photons to estimator: {self.n_photons_after_update}")
        return self.n_photons_after_update


    def pi_pulse(self, p_success=1):
        # check if pulse is on resonance with transition
        if self.state in (0, 1):
            rand = random.random()
            if (rand < p_success):
                self.state = 1 - self.state


    def update_ion(self, bin_time, dt, verbose=False):
        # simulate ion dynamics using Markov Chain Monte Carlo
        self.poisson_mu = 0
        t = 0
        while(t < bin_time):
            if verbose:
                print(f"State at time {t} us: {self.state}")

            # backround photon counts that are always present
            self.poisson_mu += self.R_dark * dt
            # check if in fluorescing state
            if self.state == 0:
                self.poisson_mu+= self.R_bright * dt
            t+=dt
            
        self.n_photons_after_update = np.random.poisson(self.poisson_mu)
            
        
    def reset(self):
        self.poisson_mu = 0
        self.state = self.init_state
        
class CalciumIon(Ion):
    def __init__(self, initial_state=0, R_dark= 0.1 / 200, R_bright = 20 / 200, tau_db = 1.1687 * 1e6, tau_bd = (200 / 0.01) * 1e6):
        super().__init__(initial_state, R_dark, R_bright)
        self.tau_db = tau_db
        self.tau_bd = tau_bd


    def update_ion(self, bin_time, dt, verbose=False):
        # simulate ion dynamics using rates
        self.poisson_mu = 0
        t = 0
        while(t < bin_time):
            if verbose:
                pass
                #print(f"State at time {t} us: {self.state}")

            # backround photon counts that are always present
            self.poisson_mu += self.R_dark * dt
            # check if in fluorescing state
            if self.state == 0:
                self.poisson_mu+= self.R_bright * dt # add to mean photon count
            
            # can add additional dynamics :-)

            # dark-to-bright leakage
            if self.state == 1:
                p_decay = 1 - np.exp(-dt / self.tau_db) # chance of decaying into a bright state
                ## check if decayed
                rand = random.random()
                if (rand < p_decay):
                    self.state = 1 # change to bright state

            # bright to dark leakage
            if self.state == 0:
                p_decay = 1 - np.exp(-dt / self.tau_bd) # chance of decaying into a shelved dark state
                ## check if decayed
                rand = random.random()
                if (rand < p_decay):
                    self.state = 3 # shelving state is assumed to be state 3, not taking part in the cycling
            t+=dt
            

        n = np.random.poisson(self.poisson_mu)
        self.n_photons_after_update = n
        if verbose:
            print(f"mean: {self.poisson_mu}, n photons: {self.photons}")
            
class BerylliumIon(Ion):
    def __init__(self, initial_state=0, R_dark= 0.1 / 200, R_bright = 24 / 200, tau_db = (1 / 78.46) * 1e6, tau_bd = (1 / 3.43e2) * 1e6):
        super().__init__(initial_state, R_dark, R_bright)
        self.tau_db = tau_db
        self.tau_bd = tau_bd


    def update_ion(self, bin_time, dt, verbose=False):
        # simulate ion dynamics using rates
        self.poisson_mu = 0
        t = 0
        while(t < bin_time):
            if verbose:
                print(f"State at time {t} us: {self.state}")

            # backround photon counts that are always present
            self.poisson_mu += self.R_dark * dt
            # check if in fluorescing state
            if self.state == 0:
                self.poisson_mu+= self.R_bright * dt # add to mean photon count
            
            # can add additional dynamics :-)

            # dark-to-bright leakage
            if self.state == 1:
                p_decay = 1 - np.exp(-dt / self.tau_db) # chance of decaying into a bright state
                ## check if decayed
                rand = random.random()
                if (rand < p_decay):
                    self.state = 1 # change to bright state

            # bright to dark leakage
            if self.state == 0:
                p_decay = 1 - np.exp(-dt / self.tau_bd) # chance of decaying into a shelved dark state
                ## check if decayed
                rand = random.random()
                if (rand < p_decay):
                    self.state = 3 # shelving state is assumed to be state 3, not taking part in the cycling
            t+=dt
            
        if verbose:
            print(self.poisson_mu)
        self.n_photons_after_update = np.random.poisson(self.poisson_mu)
        
