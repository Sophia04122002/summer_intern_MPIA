from __future__ import print_function 
from scipy.ndimage import gaussian_filter1d
import os, sys 
os.chdir('/Users/sophiatonelli/library_script/script/work')
sys.path.append(os.getcwd())

import script
sys.path.append(os.getcwd())
import numpy as np 
import matplotlib.pyplot as plt 
import script # for using SCRIPT functionalities
import matplotlib
matplotlib.rcParams.update({'font.size': 11})# increase fontsize of text on plots

#load snap-file 
gadget_snap = '/Users/sophiatonelli/Downloads/snap_120' #'../music_snap/snap_130' # import data file 
outpath = '/Users/sophiatonelli/library_script/script/work/script_files' #./script_files' # the generated density and velocity fields will be saved in script_files/
scaledist = 1e-3 # sets the scale for box size. 1e-3 for box of length in kpc/h and 1. for length in cMpc/h
default_simulation_data = script.default_simulation_data(gadget_snap, outpath, sigma_8=0.829, ns=0.961, omega_b=0.0482, scaledist=scaledist)

print("Simulation box size:", default_simulation_data.box, "cMpc/h") #box size in cMpc/h
print("Simulation redshift:", default_simulation_data.z) #redshift of the simulation
print("Simulation box size in Mpc/h:", default_simulation_data.box * 1e-3) #box size in Mpc/h


ngrid = 128 #define RESOLUTION, number of cells to which density fields are smoothened (i.e. resolution 256/128 cMpc/h)
matter_fields = script.matter_fields(default_simulation_data, ngrid, outpath, overwrite_files=False) 
log10Mmin = 9.0 #minimum mass of halo for efficient star formation, Only halos above a certain mass threshold are considered for star formation.
fcoll_arr = matter_fields.get_fcoll_for_Mmin(log10Mmin) #calculate collapse fraction of each cell
zeta = 40.13 #ionizing efficiency parameter, i.e. number of ionizing photons per collapsed baryon

#plot density field #1 + δ is used because you can’t take log10 of a negative number 
#δ>0: overdense region → filaments, halos, galaxies vs δ<0: underdense region → voids
#yellow bright= high density, likely forming halos vs red dark= low density, likely void = COSMIC WEB VIEW
im = plt.imshow(np.log10(1+matter_fields.densitycontr_arr[:,:,16]),extent=[0,default_simulation_data.box,0,default_simulation_data.box], cmap='hot')
cbar = plt.colorbar(im, label=r'$\log \Delta$')
cbar.ax.yaxis.label.set_size(16)  # or any fontsize you want
cbar.ax.tick_params(labelsize=14)
#plt.title(r'Density ($\log \Delta$)',fontsize=18)
plt.xlabel('x (cMpc/h)', fontsize=16)
plt.ylabel('x (cMpc/h)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.savefig('/Users/sophiatonelli/Desktop/pngs_mpia/week3_script_combinations/ex_logM9_zeta40/density_field.png', bbox_inches='tight') #save density field plot
plt.show()


#plot collapse fraction field
im_fcoll = plt.imshow(np.log10(fcoll_arr[:,:,16]), extent=[0,default_simulation_data.box,0,default_simulation_data.box],vmin=-3,vmax=0,cmap='magma')
cbar= plt.colorbar(im_fcoll, label= r'$\log f_{\mathrm{coll}}$')
cbar.ax.yaxis.label.set_size(16)  # or any fontsize you want
cbar.ax.tick_params(labelsize=14)
#plt.title(r'$\log f_{\mathrm{coll}}$', fontsize=18)
plt.xlabel('x (cMpc/h)', fontsize=16)
plt.ylabel('y (cMpc/h)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.savefig('/Users/sophiatonelli/Desktop/pngs_mpia/week3_script_combinations/ex_logM9_zeta40/fcoll_field.png', bbox_inches='tight') #save collapse fraction field plot
plt.show()

print ("Collapse fraction ="+  '{:.2f}'.format(np.mean(fcoll_arr))) 



ionization_map = script.ionization_map(matter_fields)
qi_arr = ionization_map.get_qi(zeta * fcoll_arr) 
print(qi_arr)
im_q = plt.imshow(1-qi_arr[:,:,16],extent=[0,default_simulation_data.box,0,default_simulation_data.box], cmap='viridis')
cbar=plt.colorbar(im_q, label=r'$x_{\mathrm{HI}}$')
cbar.ax.yaxis.label.set_size(16)  # or any fontsize you want
cbar.ax.tick_params(labelsize=14)
#plt.title(r'Neutral fraction $x_{\mathrm{HI}}$',fontsize=16, pad=15)
plt.xlabel('x (cMpc/h)', fontsize=16)
plt.ylabel('y (cMpc/h)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.savefig('/Users/sophiatonelli/Desktop/pngs_mpia/week3_script_combinations/ex_logM9_zeta40/ionization_map.png', bbox_inches='tight') #save ionization map plot
plt.show()
##########


matter_fields.initialize_powspec()
k_edges, k_bins = matter_fields.set_k_edges(nbins=20, log_bins=True)

Delta_HI_arr = (1 - qi_arr) * (1 + matter_fields.densitycontr_arr) #compute neutral hydrogen overdensity 
#binned power spectrum in mK 
powspec_21cm_binned, kount = ionization_map.get_binned_powspec(Delta_HI_arr,k_edges,units='mK')
Delta2_21 = k_bins[kount > 0]**3 * powspec_21cm_binned[kount > 0] / (2 * np.pi**2) #dimensionless form: Delta^2_21(k) 

plt.plot(k_bins[kount > 0], Delta2_21) #, marker='o'
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$k$ (h/cMpc)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(r"$\Delta^2_{21}(k)$ (mK$^2$)", fontsize=16)
plt.title("21 cm Power Spectrum", fontsize=16, pad=20)
#plt.savefig('/Users/sophiatonelli/Desktop/pngs_mpia/week3_script_combinations/ex_logM9_zeta40/powspec_21cm_binned.png', bbox_inches='tight') #save power spectrum plot
plt.show()

#maity reference values
omega_b = 0.0482 #fractional densities
omega_m = 0.308
h = 0.678  
omega_b_h2= omega_b * h**2  #convert to h^2 units. physical densities for mK
omega_m_h2 = omega_m * h**2
z=default_simulation_data.z #redshift of the simulation
#compute T0 from theory Furlanetto+2006 
T0 = 27 *  ((1 + z) / 10)**0.5 * ((0.15 / (omega_m_h2))**0.5 *(omega_b_h2 / 0.023))

#brightness temperature cube δT = T0 ×(1−qi)×(1+δ)
dT = T0 * Delta_HI_arr 
dT_subtracted = dT 

print('Mean Tb (mK): {:.4f}'.format(np.mean(dT)))
print('Mean after subtraction: {:.4e}'.format(np.mean(dT_subtracted)))

#sliceplot temperature fluctuations (i.e.at zbox=16)
plt.rcParams['figure.figsize'] = [6, 5]
plt.title('21 cm Brightness Temperature Fluctuation', fontsize=16, pad=20)
plt.imshow(dT_subtracted[:, :, 16], extent=[0, default_simulation_data.box, 0, default_simulation_data.box],  cmap='RdBu_r' )
plt.xlabel('x (cMpc/h)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('y (cMpc/h)', fontsize=16)
cbar= plt.colorbar(label='mK')
cbar.ax.yaxis.label.set_size(16)  
cbar.ax.tick_params(labelsize=14)
#plt.savefig('/Users/sophiatonelli/Desktop/pngs_mpia/week3_script_combinations/ex_logM9_zeta40/brightness_temperature_fluctuation_slice.png', bbox_inches='tight') #save brightness temperature fluctuation plot
plt.show() 






sys.exit()

########

#check PC CONSERVATION
qi_mean = np.mean(qi_arr * (1 + matter_fields.densitycontr_arr)) #it means: how much ionized hydrogen mass is in each cell !
print ( 'total hydrogen mass is ionized in the simulation box(mean) = ' + '{:.2f}'.format(qi_mean) ) #mass averaged ionized fraction of Hydrogen
print ( 'zeta times fcoll = ' +  '{:.2f}'.format(zeta * np.mean(fcoll_arr * (1 + matter_fields.densitycontr_arr))) ) #predicted ionized fraction

ionized_mask = (qi_arr > 0.5) #mask for ionized cells
grid_shape = qi_arr.shape
print("Grid shape:", grid_shape) #shape of the ionization map array
box_size = default_simulation_data.box  #in cMpc/h
ngrid = grid_shape[0]
print("No. cells x dimension:", ngrid) #number of cells in each dimension
cell_size = box_size / ngrid
print("Resolution:", cell_size, "cMpc/h") #resolution of the grid in cMpc/h


########
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

def launching_rays(ray_centre_location, random_direction_vector, ionized_mask, ngrid, cell_size, max_step_size, step_size=0.05, max_steps=1000):
    pos = np.array(ray_centre_location, dtype=np.float64)
    for _ in range(max_steps):
        pos += random_direction_vector * step_size
        pos = apply_periodic_pos(pos, ngrid) #continuity position boundary condition
        idx = np.floor(pos).astype(int) #discrete index boundary condition (associated index cell for vector-position travelled)
        idx = apply_periodic_index(idx, ngrid)
        if not ionized_mask[tuple(idx)]:
            return step_size * _ * cell_size #0.05 x ith x 2cMpc/h = total distance travelled
    return max_step_size #step_size * max_steps * cell_size #if it never hits a neutral regions, it means fully ionized, then take the full box size

#main 
mean_free_paths = []

mean_mfp_list = []  #for summary table
var_mfp_list = []  

peak_mfp_list = []

num_iterations = 1000
ionized_coords = np.argwhere(ionized_mask == 1)

for i in range(num_iterations):
    if len(ionized_coords) == 0:
        continue

    starting_index = np.random.choice(len(ionized_coords))
    ray_centre_idx = ionized_coords[starting_index]
    vector_direction = choose_random_direction()
    mfp = launching_rays(ray_centre_idx, vector_direction, ionized_mask, ngrid, cell_size, box_size)
    mean_free_paths.append(mfp)

mfp_array = np.array(mean_free_paths)
#print("Mean Free Paths:", mfp_array)

mean_mfp = np.mean(mfp_array)
var_mfp = np.var(mfp_array)
mean_mfp_list.append(mean_mfp)
var_mfp_list.append(var_mfp)

#histogram mfps
plt.hist(mfp_array, bins=20, density=True)
plt.xlabel(r'$R$ ($\mathrm{cMpc}\ h^{-1}$)', fontsize=16)
plt.ylabel('PDF', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.savefig('/Users/sophiatonelli/Desktop/pngs_mpia/week3_script_combinations/ex_logM9_zeta40/mfp_histogram.png', bbox_inches='tight')
plt.show()

print(mfp_array)
print(mfp_array.min())
print(mfp_array.min())


R_min = 2   #or physical minimum radius
R_max = 256
num_bins = 100
#logarithmic binning:
log_bins = np.logspace(np.log10(R_min), np.log10(R_max), num_bins + 1)
print("Logarithmic bins:", log_bins)

#sys.exit() #stop here to check the log_bins
counts_over_radius2, log_bin_edges = np.histogram(mfp_array, bins=log_bins, density=True) #https://numpy.org/devdocs/reference/generated/numpy.histogram.html
print( log_bin_edges[1:] - log_bin_edges[:-1])

counts_over_radius, bin_edges = np.histogram(mfp_array, bins=100, density=True) #https://numpy.org/devdocs/reference/generated/numpy.histogram.html
#bins=20
print("Bin edges:", bin_edges)
print("Counts over radius:", counts_over_radius)
bin_widths = bin_edges[1:] - bin_edges[:-1]
print("Bin widths:", bin_widths)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) 
print("Bin centers:", bin_centers)
#sys.exit() #stop here to check the bin_edges and bin_widths
r_times_dP_dR = bin_centers * counts_over_radius / np.array(bin_widths)

r_times_dN_dR_smooth = gaussian_filter1d(r_times_dP_dR, sigma=1.0)

peak_mfp = bin_centers[np.argmax(r_times_dP_dR)]
peak_mfp_list.append(peak_mfp)
print(f" Peak MFP: {peak_mfp:.2f}")

plt.plot( (0.5*(log_bin_edges[:-1] + log_bin_edges[1:])) , (0.5*(log_bin_edges[:-1] + log_bin_edges[1:]))  * counts_over_radius2 / (log_bin_edges[1:] - log_bin_edges[:-1]), drawstyle='steps-mid', color='blue', label="Log Step trend")
plt.plot(bin_centers, r_times_dP_dR, drawstyle='steps-mid', color='black', label="Step trend")
plt.plot(bin_centers, r_times_dN_dR_smooth, color='green', label="smooth trend")
#plt.plot(bin_centers, r_times_dN_dR, drawstyle='steps-mid', color='green')
#plt.axvline(peak_mfp, color='red', linestyle='--', label=rf"$R_{{\rm peak}} = {np.ceil(peak_mfp):.2f}$")
#plt.xscale('log')
plt.xlabel(r'$R$ ($\mathrm{cMpc}\ h^{-1}$)', fontsize=16)
plt.ylabel(r'$R \cdot \frac{dP}{dR}$', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
#plt.savefig('/Users/sophiatonelli/Desktop/pngs_mpia/week3_script_combinations/ex_logM9_zeta40/2_R_times_dN_dR.png', bbox_inches='tight')
plt.show()

print("\n=== MFP Summary ===")
print(f"Mean = {np.array(mean_mfp_list):.2f}, variance = {np.array(var_mfp_list):.2f}, peak = {np.array(peak_mfp_list):.2f} (cMpc/h)")



"""
Stage of Reionization	Redshift	Typical Bubble Size
Early	z > 10	0.1 , 1 cMpc/h
Mid (overlap phase)	z ≈ 8	1 , 5 cMpc/h
Late	z ≈ 6	10 , 30 cMpc/h

So a 2.9 cMpc/h peak at z = 8 suggests we are:
In the mid-stage of reionization, where multiple ionized regions (H II bubbles) have grown and started to merge, but the universe is not yet fully ionized.
"""

