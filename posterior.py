from __future__ import print_function
from cobaya.likelihood import Likelihood
import csv 
import pandas as pd
import numpy as np
import os, sys 
os.chdir('/Users/sophiatonelli/library_script/script/work')
sys.path.append(os.getcwd())
import script
sys.path.append(os.getcwd()) 

import matplotlib.pyplot as plt 
import matplotlib
from scipy.stats import gaussian_kde
matplotlib.rcParams.update({'font.size': 11})


#load the observed peak values from the CSV file
#obs_Rpeak_vals = []
#with open('/Users/sophiatonelli/Desktop/w6_code/summary_R_peaks.csv', 'r') as file:
    #reader = csv.reader(file)
   # for row in reader:
       # if row[3] == 'Peak':  #skip the header row
         #   continue
       # print(row[3])  #print the observed 'Peak' column for PC-zeta(M-dependent) @ res 
      # obs_Rpeak_vals.append(row[3]) 
#obs_PC_Rpeaks = np.array(obs_Rpeak_vals, dtype=float)  


obs_PC_Rpeaks = np.load('peaks_pc.npy')  #load the observed peaks for PC method
print(obs_PC_Rpeaks)

one_obs_peak = obs_PC_Rpeaks[0]  #get the first value as a scenario peak
print("chosen observed peak value:", one_obs_peak)

#############
def apply_periodic_index(idx, ngrid):
    return idx % ngrid

def apply_periodic_pos(pos, ngrid):
    return np.mod(pos, ngrid)

def choose_random_direction(): 
    theta = np.arccos(np.random.uniform(-1, 1))  #theta range: 0 to pi
    phi = np.random.uniform(0, 2 * np.pi)
    x_vec = np.sin(theta) * np.cos(phi)
    y_vec = np.sin(theta) * np.sin(phi)
    z_vec = np.cos(theta)
    return np.array([x_vec, y_vec, z_vec], dtype=np.float64)

def launching_rays(start_idx, random_direction_vector, ionized_mask, ngrid, cell_size, step_size=1.0): 
    pos = np.array(start_idx, dtype=np.float64)
    distance=0.0
    max_distance = 3 * ngrid 
    while distance < max_distance:
        pos += random_direction_vector * step_size
        pos = apply_periodic_pos(pos, ngrid) 
        idx = np.floor(pos).astype(int) 
        idx = apply_periodic_index(idx, ngrid)
        if not ionized_mask[tuple(idx)]:
            return distance * cell_size 
        distance += step_size      
    return max_distance * cell_size  

gadget_snap = '/Users/sophiatonelli/Downloads/snap_140' 
outpath = '/Users/sophiatonelli/library_script/script/work/script_files' 
scaledist = 1e-3 
default_simulation_data = script.default_simulation_data(gadget_snap, outpath, sigma_8=0.829, ns=0.961, omega_b=0.0482, scaledist=scaledist) 
#########
def some_model_photon_conserving(zeta, logMmin):
    """this function should return the model prediction 
    for the photon-conserving method"""
    ngrid = 64 #CHECK GRID SIZE CSV FILE 
    box_size = default_simulation_data.box  #in cMpc/h
    cell_size = box_size / ngrid
    matter_fields = script.matter_fields(default_simulation_data, ngrid, outpath, overwrite_files=False)
    fcoll_arr = matter_fields.get_fcoll_for_Mmin(logMmin)
    ionization_map = script.ionization_map(matter_fields)
    qi_arr = ionization_map.get_qi(zeta * fcoll_arr)

    threshold = 0.5 
    num_iterations = 100000

    PC_mean_free_paths = []
    ionized_mask = (qi_arr >= threshold)
 
    for _ in range(num_iterations):
        idx = np.random.randint(0, ngrid, size=3)
        if not ionized_mask[tuple(idx)]:
            continue
        vector_direction = choose_random_direction()
        mfp = launching_rays(idx, vector_direction, ionized_mask, ngrid, cell_size)
        PC_mean_free_paths.append(mfp)
    PC_physical_mfp = np.array(PC_mean_free_paths)

    R_min = cell_size #4 cMPc/h
    R_max = box_size
    smooth_R = np.linspace(R_min, R_max, 64 ) #128,64) #only 64 bins in this case for pre-fixed resolution

    PC_kde = gaussian_kde(PC_physical_mfp)
    PC_smooth_pdf = PC_kde(smooth_R)
    PC_bsd = smooth_R * PC_smooth_pdf
    PC_bsd /= np.trapz(PC_bsd, smooth_R)

    R_peak = smooth_R[np.argmax(PC_bsd)]

    return R_peak  #return the R value at the peak of the BSD for the photon-conserving method



#cobaya-run likelihood.yml
class BSDlikelihood(Likelihood):

    def setup(self):
        #store observed data and uncertainty
        self.obs_PC_peak_val = one_obs_peak
        self.sigma_peak_val = 0.10 * self.obs_PC_peak_val  #10% uncertainty

        #self.obs_PC_fwhm = one_obs_fwhm
        #self.sigma_fwhm = 0.10 * self.obs_PC_fwhm

    def logp(self, **params):
        zeta = params["zeta"]
        logMmin = params["logMmin"]
        #beta = params["beta"]

        peak_model  = self.model_sigma_peaks(zeta, logMmin) #, beta)

        chi_squared = ((self.obs_PC_peak_val - peak_model) ** 2) / (self.sigma_peak_val ** 2)  #+ np.sum((self.obs_PC_peak_vals - peak_model) ** 2 / self.sigma_peak_vals)
        return -0.5 * chi_squared


    def model_sigma_peaks(self, zeta, logMmin): #, beta):
        peak_PC = some_model_photon_conserving(zeta, logMmin) #, beta)
        #peak_ES = some_model_excursion_set(zeta, Mmin, beta)
        return peak_PC  #np.array([sigma1, sigma2])
    
    