from __future__ import print_function
from scipy.ndimage import gaussian_filter1d
import tools21cm as t2c
import os, sys 
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.stats import gaussian_kde
os.chdir('/Users/sophiatonelli/library_script/script/work')
sys.path.append(os.getcwd())
import script
sys.path.append(os.getcwd())
import numpy as np 
import matplotlib.pyplot as plt 
import script 
import matplotlib
matplotlib.rcParams.update({'font.size': 11})


gadget_snap = '/Users/sophiatonelli/Downloads/snap_130'  #CHECK SNAPSHOT NAME
outpath = '/Users/sophiatonelli/library_script/script/work/script_files' 
scaledist = 1e-3 #sets the scale for box size. 1e-3 for box of length in kpc/h and 1. for length in cMpc/h
default_simulation_data = script.default_simulation_data(gadget_snap, outpath, sigma_8=0.829, ns=0.961, omega_b=0.0482, scaledist=scaledist) 

print("Simulation box size:", default_simulation_data.box, "cMpc/h") 
print("Simulation redshift:", default_simulation_data.z)


ngrid = 64 #NGRID: 128, 64, 42, 32
matter_fields = script.matter_fields(default_simulation_data, ngrid, outpath, overwrite_files=False) 
list_xhi_vals = [0.2, 0.5, 0.7]    #target mean neutral fractions
list_log10Mmin_vals = [8.0, 9.0, 10.0]  #log10(Mmin)


#LOADING ZETA VALUES FROM THE PANEL PLOTS GENERATED IN THE NOTEBOOKS
#These are the zeta values for the different Mmin and xhi combinations WITH ES AND PC METHODS
pc_zeta_load = np.load('zeta_pc_array_redsh8.npy')
es_zeta_load = np.load('zeta_es_array_redsh8.npy')
print("es zeta vals", es_zeta_load)
print("pc zeta vals", pc_zeta_load)

#sys.exit()

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

def launching_rays(start_idx, random_direction_vector, ionized_mask, ngrid, cell_size, step_size=1.0): #step_size=0.05
    pos = np.array(start_idx, dtype=np.float64)
    distance=0.0
    max_distance = 3 * ngrid #3*boxsize
    while distance < max_distance:
        pos += random_direction_vector * step_size
        pos = apply_periodic_pos(pos, ngrid) 
        idx = np.floor(pos).astype(int) 
        idx = apply_periodic_index(idx, ngrid)
        if not ionized_mask[tuple(idx)]:
            return distance * cell_size   
        distance += step_size      
    return max_distance * cell_size 


#MAIN CODE
#This code generates the bsd for the two methods (PC and ES) for different Mmin and xhi values.
#It calculates the mean free path distribution and plots the results.
#It also saves the peaks of the bsd for both methods in a summary table.    

nrows, ncols = 3, 3  #3 log10Mmin, 3 zeta per Mmin
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)

#store for summary table
Rpeaks_pc=[] 
Rpeaks_es=[]

threshold = 0.5 #1.0
box_size = default_simulation_data.box
cell_size = box_size / ngrid

R_min = cell_size #such as 4, 2cMPc/h, etc. 
R_max = box_size
smooth_R = np.linspace(R_min, R_max, 64) #127
R_dense = np.linspace(smooth_R[0], smooth_R[-1], 1000)

#if method == 'PC': then zeta = pc_zeta_load[j]
#if method == 'ES': then zeta = es_zeta_load[j]
for j in range(9):
    i = j // 3  #which log10Mmin (row)
    k = j % 3   #which zeta in this row (column)

    log10Mmin = list_log10Mmin_vals[i]
    fcoll_arr = matter_fields.get_fcoll_for_Mmin(log10Mmin)

    xhi = list_xhi_vals[k]

    print(f"Processing log10(Mmin)={log10Mmin}")

    ax = axes[i, k]

    #for j, method in enumerate(['PC', 'ES']):
    for method, color in zip(['PC', 'ES'], ['black', 'green']):
        if method == 'PC':
            zeta = pc_zeta_load[j]       
            ionization_map = script.ionization_map(matter_fields)
            qi_arr = ionization_map.get_qi(zeta * fcoll_arr)
            print(f"Processing PC: log10(Mmin)={log10Mmin}, zeta={zeta:.2f},xhi = {1-qi_arr.mean():.2f}") 
            ionized_mask = (qi_arr >= threshold) #mask for ionized cells
            mean_free_paths = []
            num_iterations = 1000000 #1e6 -> 1e7

            for _ in range(num_iterations):
                idx = np.random.randint(0, ngrid, size=3)
                if not ionized_mask[tuple(idx)]: #check if ionized
                    continue
                vector_direction = choose_random_direction()
                mfp = launching_rays(idx, vector_direction, ionized_mask, ngrid, cell_size) #, box_size
                mean_free_paths.append(mfp)
            physical_mfp = np.array(mean_free_paths)
            kde = gaussian_kde(physical_mfp)
            smooth_pdf = kde(smooth_R)
            print("area pdf",np.trapz(smooth_pdf, smooth_R))
            bsd = smooth_R * smooth_pdf
            bsd /= np.trapz(bsd, smooth_R)
            print("bsd", np.trapz(bsd, smooth_R))

            curve_PC_bsd = interp1d(smooth_R, bsd, kind='cubic', fill_value="extrapolate")
            dense_PC_bsd = curve_PC_bsd(R_dense)

            peak_idx = np.argmax(dense_PC_bsd) 
            R_peak = R_dense[peak_idx]
            peak = dense_PC_bsd[peak_idx]
            Rpeaks_pc.append(R_peak)
            
            ax.plot(R_dense, curve_PC_bsd, label=f'This work - {method}', color=color)
            ax.vlines(R_peak, ymin=0, ymax=peak, linestyle='-.', color='black', label='Peak (PC)')

        elif method=='ES':
            zeta_es = es_zeta_load[j]
                  
            es_ionization_map = script.ionization_map(matter_fields, method='ES') #HERE WE SET THE DIFFERENCE
            qi_arr_es = es_ionization_map.get_qi(zeta_es * fcoll_arr)
            print(f"Processing ES: log10(Mmin)={log10Mmin}), zeta={zeta_es:.2f}, xhi = {1-qi_arr.mean():.2f}") 
            es_ionized_mask = (qi_arr_es >= threshold) #mask for ionized cells
            es_mean_free_paths = []
            num_iterations = 1000000 #1e6 -> 1e7

            for _ in range(num_iterations):
                idx = np.random.randint(0, ngrid, size=3)
                if not es_ionized_mask[tuple(idx)]: #check if ionized
                    continue
                vector_direction = choose_random_direction()
                es_mfp = launching_rays(idx, vector_direction, es_ionized_mask, ngrid, cell_size) #, box_size
                es_mean_free_paths.append(es_mfp)

            es_physical_mfp = np.array(es_mean_free_paths)
            kde = gaussian_kde(es_physical_mfp)
            smooth_pdf = kde(smooth_R)
            print("area pdf",np.trapz(smooth_pdf, smooth_R))
            bsd = smooth_R * smooth_pdf
            bsd /= np.trapz(bsd, smooth_R)
            print("bsd", np.trapz(bsd, smooth_R))

            curve_ES_bsd = interp1d(smooth_R, bsd, kind='cubic', fill_value="extrapolate")
            dense_ES_bsd = curve_ES_bsd(R_dense)

            peak_idx_es = np.argmax(dense_PC_bsd) 
            R_peak_es = R_dense[peak_idx_es]
            peak_es = dense_ES_bsd[peak_idx_es]
            Rpeaks_es.append(R_peak_es)

            ax.plot(R_dense, dense_ES_bsd, label=f'This work - {method}', color=color)
            ax.vlines(R_peak_es, ymin=0, ymax=peak_es, linestyle='-.', color='blue', label='Peak (ES)')

    
    ax.axvline(cell_size, linestyle = '--', color='grey', alpha=0.3)
    ax.legend(fontsize=11, loc='upper right', frameon=False)

    ax.set_title(rf'$z = {default_simulation_data.z}$, '
    rf'$\log_{{10}}(M_{{\min}}) = {log10Mmin}$, '
    rf'$x_{{HI}} = {xhi:.2f}$', pad=10)


    ax.set_xscale('log')
    ax.set_xlabel("R (cMpc/h)", fontsize=14)
    ax.set_ylabel(r"$R\,dP/dR$", fontsize=14)
    ax.tick_params(labelsize=14)
plt.savefig('/Users/sophiatonelli/Desktop/pngs_mpia/w7/interp_redsh7_bsd_pc_es_64res.png', dpi=300, bbox_inches='tight')
plt.show()

#print("Summary PC R peaks:", Rpeaks_pc)
#print("Summary ES R peaks:", Rpeaks_es)
#np.save('Rpeaks_pc_redsh8.npy', np.array(Rpeaks_pc))
#np.save('Rpeaks_es_redsh8.npy', np.array(Rpeaks_es))

import pandas as pd #store peaks for MCMC calculations

rows = []
peak_idx = 0 #start counting peaks idx to go through all len(peaks)
for i, log10Mmin in enumerate(list_log10Mmin_vals):
    for j, xhi in enumerate(list_xhi_vals):
            if peak_idx < len(Rpeaks_pc):
                rows.append( { "log10Mmin": log10Mmin,"xHI": round(xhi, 3),"R Peak for PC": round(Rpeaks_pc[peak_idx], 4)} )
                peak_idx += 1

df_peaks = pd.DataFrame(rows) #display created table
print(df_peaks.to_string(index=False))
df_peaks.to_csv("/Users/sophiatonelli/Desktop/w6_code/summary_PC_peaks_zeta_redsh7.csv", index=False)

##########################




