from scipy.stats import poisson
import numpy as np

class Estimator:
    
    def __init__(self):
        self.prediction = None
        self.ready = False
        self.n_photons_tot = 0
        self.n_photons_subbin = 0
        
        self.pi_pulse = False
        
        self.estimator_type = "PrototypeEstimator"

    def predict(self):
        return self.prediction
    
    def update_subbin_photon_counts(self, n_photons):
        self.n_photons_subbin += n_photons
        

    def update_prediction(self, dt_subbin):
        self.n_photons_tot += self.n_photons_subbin
        self.n_photons_subbin = 0
        self.ready = True
        
    def get_if_ready(self):
        return self.ready
    
    def get_is_pi_pulse_needed(self):
        return self.pi_pulse
    
    def pi_pulse_applied(self):
        pass
    
        
    def get_stats(self):
        stats = {'prediction' : self.prediction,
                 'n_photons_tot' : self.n_photons_tot}
        return stats
    
    def get_estimator_type(self):
        return self.estimator_type
    
    def reset(self):
        self.n_photons_tot = 0
        self.prediction = None
        self.ready = False
        
class ThresholdingEstimator(Estimator):
    def __init__(self, threshold, t_bin):
        super().__init__()
        self.threshold = threshold
        
        self.t = 0
        self.t_bin = t_bin
        
        self.estimator_type = "Thresholdig"

    def update_prediction(self, dt_subbin):
        self.t += dt_subbin
        
        self.n_photons_tot += self.n_photons_subbin
        
        
        if (self.t >= self.t_bin):
            if (self.n_photons_tot > self.threshold):
                self.prediction = 0
            else:
                self.prediction = 1
            self.ready = True
            
        self.n_photons_subbin = 0
        
    def get_stats(self):
        stats = {'prediction' : self.prediction,
                 'n_photons_tot' : self.n_photons_tot}
        return stats
        
    def reset(self):
        self.t = 0
        self.total_count = 0
        self.n_photons_tot = 0
        
        self.prediction = None
        self.ready = False
        
        
        
class AdaptiveMLEstimator(Estimator):
    
    def __init__(self, R_D, R_B, e_c, n_subbins_max):
        super().__init__()
        self.R_D = R_D
        self.R_B = R_B
        self.p_b = 1
        self.p_d = 1
        self.e_c = e_c
        self.n_subbins_max = n_subbins_max
        self.n_subbins = 0
        
        self.estimator_type = "AdaptiveML"
        
        
    def update_likelihoods(self, n_photons, dt):
        mean_b = (self.R_D + self.R_B) * dt
        mean_d = self.R_D * dt
        self.p_b *= poisson.pmf(n_photons, mean_b)
        self.p_d *= poisson.pmf(n_photons, mean_d)
        
    def update_prediction(self, dt_subbin):
        if not self.ready:
            # count bin
            self.n_subbins+=1
            # update photon tally
            self.n_photons_tot += self.n_photons_subbin
            # update likelihoods
            self.update_likelihoods(self.n_photons_subbin, dt_subbin)
            #evaluate errors
            sum_p = self.p_b + self.p_d
            e_d = self.p_b / sum_p
            e_b = self.p_d / sum_p
            # check if threshold met
            if min(e_d, e_b) < self.e_c:
                self.ready = True
                if (self.p_d > self.p_b):
                    self.prediction = 1
                else:
                    self.prediction = 0
            if (self.n_subbins >= self.n_subbins_max):
                self.ready = True
                if (self.p_d > self.p_b):
                    self.prediction = 1
                else:
                    self.prediction = 0
            
            self.n_photons_subbin = 0
                
    def get_stats(self):
        stats = {'prediction': self.prediction,
                    'n_subbins': self.n_subbins,
                    'n_photons_tot': self.n_photons_tot}
        return stats
    
    def reset(self):
        self.ready = False
        self.prediction = None
        self.pi_pulse = False
        
        self.p_b = 1
        self.p_d = 1
        
        self.n_subbins = 0
        self.n_photons_tot = 0
        
class NonPoissonianAdaptiveMLEstimator(AdaptiveMLEstimator):
    
    def __init__(self, R_D, R_B, tau_bd, tau_db, e_c, n_subbins_max):
        super().__init__(R_D, R_B, e_c, n_subbins_max)
        
        self.tau_bd = tau_bd
        self.tau_db = tau_db
        
        self.M_dark = 1
        self.S_dark = 0
        
        self.M_bright = 1
        self.S_bright = 0
        
        self.estimator_type = "NonPoissonianAdaptiveML"
        
    
    def update_likelihoods(self, n_photons, dt):
        mean_b = (self.R_D + self.R_B) * dt
        mean_d = self.R_D * dt
        
        # update dark state likelihood
        ## first update S as we need old M!
        self.S_dark = (self.S_dark + self.M_dark) * poisson.pmf(n_photons, mean_b)
        ## update M
        self.M_dark *= poisson.pmf(n_photons, mean_d)
        ## new likelihood for dark state
        self.p_d = (1 - (self.n_subbins * dt / self.tau_db)) * self.M_dark + (dt / self.tau_db) * self.S_dark
        
        # update bright state likelihood
        ## first update S as we need old M!
        self.S_bright = (self.S_bright + self.M_bright) * poisson.pmf(n_photons, mean_d)
        ## update M
        self.M_bright *= poisson.pmf(n_photons, mean_b)
        ## new likelihood for dark state
        self.p_b = (1 - (self.n_subbins * dt / self.tau_bd)) * self.M_bright + (dt / self.tau_bd) * self.S_bright
        
        
    def reset(self):
        self.ready = False
        self.prediction = None
        self.pi_pulse = False
        
        self.p_b = 1
        self.p_d = 1
        self.n_subbins = 0
        self.n_photons_tot = 0
        
        self.M_dark = 1
        self.S_dark = 0
        
        self.M_bright = 1
        self.S_bright = 0
        
class EntropyGainEstimator(Estimator):
    
    def __init__(self, R_D, R_B, e_c, n_subbins_max, n_counts_max):
        super().__init__()
        self.R_D = R_D
        self.R_B = R_B
        self.p_b = 1
        self.p_d = 1
        self.e_c = e_c
        
        self.n_subbins_max = n_subbins_max
        self.n_subbins = 0
        
        self.n_counts_max = n_counts_max
        self.state_flipped = False
        
        self.pi_pulse = False
        self.n_pi_pulses = 0
        
        self.estimator_type = "EntropyGain"
        
    def update_likelihoods(self, n_photons, dt):
        mean_b = (self.R_D + self.R_B) * dt
        mean_d = self.R_D * dt
        if not self.state_flipped:
            self.p_b *= poisson.pmf(n_photons, mean_b)
            self.p_d *= poisson.pmf(n_photons, mean_d)
        else:
            self.p_b *= poisson.pmf(n_photons, mean_d)
            self.p_d *= poisson.pmf(n_photons, mean_b)
            
    def get_entropy(self, p_1, p_2):
        # the entropy is a measure for the randomness/uncertainty of an outcome/observation
        # For a discrete probability distribution with m outcomes, the entropy is maximized for a uniform distribution (e.g P(x) = 1 / m for all x in (1, ..., m))

        eps = np.finfo('float').eps
        return -(p_1 * np.log(p_1 + eps) + p_2 * np.log(p_2 + eps))
    
    def get_probabilities(self, p_1, p_2):
        s = p_1 + p_2
        return p_1 / s, p_2 / s
            
        
    def is_pi_pulse_needed(self, dt):
        """rho_smin_B, rho_smin_D are the likelihood at the previous step"""
        H_diff_tot = 0
        for i in range(self.n_counts_max):
            # calculate poisson means for subbin assuming state does not change during bin time
            mean_b = (self.R_D + self.R_B) * dt
            mean_d = self.R_D * dt
            # calculate the possible likelihoods for every photon count if we do not apply a Pi pulse
            rho_b_i = self.p_b * poisson.pmf(i, mean_b)
            rho_d_i = self.p_d * poisson.pmf(i, mean_d)
            
            # calculate the possible likelihoods for every photon count if we DO apply a Pi pulse
            rho_b_pi_i = self.p_b * poisson.pmf(i, mean_d)
            rho_d_pi_i = self.p_d * poisson.pmf(i, mean_b)
            
            # calculate the associated probabilities (normalize such that probabilities add to one)
            p_d_i, p_b_i = self.get_probabilities(rho_d_i, rho_b_i)
            p_d_pi_i, p_b_pi_i = self.get_probabilities(rho_d_pi_i, rho_b_pi_i)
            
            #calculate the entropies:
            H_i = self.get_entropy(p_d_i, p_b_i)
            H_pi_i = self.get_entropy(p_d_pi_i, p_b_pi_i)
            
            # H_diff_tot is wighted by P("i photon counts") = P("i photon counts" | |bright>) + P("i photon counts" | |dark>)
            H_diff_tot+= (H_pi_i - H_i) * (poisson.pmf(i, mean_b) + poisson.pmf(i, mean_d))
        
        if H_diff_tot > 0:
            # if the entropy of the likelihoods is greater (e.g there is more uncertainty on average), it is advantages to make an observation/measurement in this regime
            self.pi_pulse = True
        else:
            self.pi_pulse = False
            
    def pi_pulse_applied(self):
        self.n_pi_pulses+= 1
        self.state_flipped = not self.state_flipped
    
            
    def update_prediction(self, dt_subbin):
        # only update if not ready, this is relevant for later simulations were multiple estimators can run in an simulation environment
        if not self.ready:
            # count bin
            self.n_subbins+=1
            
            # update photon tally
            self.n_photons_tot += self.n_photons_subbin
            # update likelihoods
            self.update_likelihoods(self.n_photons_subbin, dt_subbin)
            #evaluate errors
            sum_p = self.p_b + self.p_d
            e_d = self.p_b / sum_p
            e_b = self.p_d / sum_p
            # check if threshold met
            if min(e_d, e_b) < self.e_c:
                self.ready = True
                if (self.p_d > self.p_b):
                    self.prediction = 1
                else:
                    self.prediction = 0
            else:
                # check if the entropy of the likelihood function averaged over all/most future outcomes is greater if we applied a Pi pulse
                self.is_pi_pulse_needed(dt_subbin)
            
            if (self.n_subbins >= self.n_subbins_max):
                self.ready = True
                if (self.p_d > self.p_b):
                    self.prediction = 1
                else:
                    self.prediction = 0
            # set subbin photon count to 0 for next bin
            self.n_photons_subbin = 0
            
    def get_stats(self):
        stats = {'prediction': self.prediction,
                 'n_subbins' : self.n_subbins,
                 'n_pi_pulses' : self.n_pi_pulses,
                 'n_photons_tot' : self.n_photons_tot}
        return stats
    
    def reset(self):
        self.ready = False
        self.prediction = None
        self.pi_pulse = False
        
        self.p_b = 1
        self.p_d = 1
        self.state_flipped = False
        
        self.n_subbins = 0
        self.n_pi_pulses = 0
        self.n_photons_tot = 0
