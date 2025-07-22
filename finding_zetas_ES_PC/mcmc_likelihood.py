from __future__ import print_function
import numpy as np
import multiprocessing
import os, sys
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import emcee
import corner
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import time
import arviz as az
from IPython.display import display, Math
import matplotlib
matplotlib.rcParams.update({'font.size': 11})
os.chdir('/Users/sophiatonelli/library_script/script/work')
sys.path.append(os.getcwd())
import script 

# load data R @ PEAK + FWHM from selected observed scenario
one_obs_peak = np.load('R_peak.npy')
print("chosen observed R @ peak for MCMC: ", one_obs_peak)
one_obs_fwhm = np.load('width.npy')
print("associated observed FWHM value: ", one_obs_fwhm)


# helper functions to find BSDs for the model
def apply_periodic_index(idx, ngrid):
    return idx % ngrid

def apply_periodic_pos(pos, ngrid):
    return np.mod(pos, ngrid)

def choose_random_direction():
    theta = np.arccos(np.random.uniform(-1, 1))
    phi = np.random.uniform(0, 2 * np.pi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z], dtype=np.float64)

def launching_rays(start_idx, direction_vec, ionized_mask, ngrid, cell_size, step_size=1.0):
    pos = np.array(start_idx, dtype=np.float64)
    distance = 0.0
    max_distance = 3 * ngrid
    while distance < max_distance:
        pos += direction_vec * step_size
        pos = apply_periodic_pos(pos, ngrid)
        idx = np.floor(pos).astype(int)
        idx = apply_periodic_index(idx, ngrid)
        if not ionized_mask[tuple(idx)]:
            return distance * cell_size
        distance += step_size
    return max_distance * cell_size

# load SNAPSHOT simulation data
gadget_snap = '/Users/sophiatonelli/Downloads/snap_120'
outpath = '/Users/sophiatonelli/library_script/script/work/script_files'
scaledist = 1e-3
default_simulation_data = script.default_simulation_data(gadget_snap, outpath, sigma_8=0.829, ns=0.961,omega_b=0.0482, scaledist=scaledist)


# define the model function for photon-conserving method PC
def some_model_photon_conserving(zeta, logMmin, ngrid=32, num_iterations=100000):  #ngrid=64
    box_size = default_simulation_data.box  # in cMpc/h
    cell_size = box_size / ngrid
    matter_fields = script.matter_fields(default_simulation_data, ngrid, outpath, overwrite_files=False)
    fcoll_arr = matter_fields.get_fcoll_for_Mmin(logMmin)
    ionization_map = script.ionization_map(matter_fields)
    qi_arr = ionization_map.get_qi(zeta * fcoll_arr)

    ionized_mask = (qi_arr >= 0.5)
    PC_mean_free_paths = []

    for _ in range(num_iterations):
        idx = np.random.randint(0, ngrid, size=3)
        if not ionized_mask[tuple(idx)]:
            continue
        direction = choose_random_direction()
        mfp = launching_rays(idx, direction, ionized_mask, ngrid, cell_size)
        PC_mean_free_paths.append(mfp)

    if len(PC_mean_free_paths) == 0:
        return -np.inf, -np.inf  # prevent crashes

    PC_physical_mfp = np.array(PC_mean_free_paths)
    smooth_R = np.linspace(cell_size, box_size, 64)

    try:
        kde = gaussian_kde(PC_physical_mfp)
        pdf = kde(smooth_R)
        bsd = smooth_R * pdf
    
        area = np.trapz(bsd, smooth_R)
        if area == 0 or np.isnan(area) or np.isinf(area):
            return -np.inf, -np.inf # this means log-likelihood has not got a good point and it will discard it

        bsd /= area

        #R @ peak
        model_R_peak = smooth_R[np.argmax(bsd)]

        #FWHM
        bsd_func = interp1d(smooth_R, bsd, kind='cubic', fill_value="extrapolate")
        y_target = bsd[np.argmax(bsd)] / 2.0
        def equation(x):
            return bsd_func(x) - y_target

        #x-values near the peak for initial guesses
        x1 = fsolve(equation, model_R_peak - 2)[0] # fsolve: find the roots of a function
        x2 = fsolve(equation, model_R_peak + 2)[0]
        width = x2 - cell_size
        return model_R_peak, width # return postion R of the peak of the bsd + fwhm
    
    except Exception as e:
        print(f"KDE failure for zeta={zeta}, logMmin={logMmin}: {e}")
        return 0.0

# likelihood + prior 
obs_peak = one_obs_peak
sigma_peak = 0.1 * obs_peak # we assume 10% uncertainty
obs_fwhm = one_obs_fwhm
sigma_fwhm = 0.1 * obs_fwhm # we assume 10% uncertainty

def log_prior(theta):
    """
    define a flat uniform prior for zeta and logMmin
    """
    zeta, logMmin = theta
    if 1.0 < zeta < 50.0 and 7.0 < logMmin < 11.0: # set prior ranges #zeta 1<100
        return 0.0  # approx constant prior because logP1 - logP2 = 0, from log(1)=0
    return -np.inf # if it is out of bounds prior

def log_likelihood(theta):
    """
    define log-likelihood as the sum of R @ peak + FWHM
    """
    zeta, logMmin = theta
    try:
        model_peak, model_width = some_model_photon_conserving(zeta, logMmin)
    except Exception as e:
        print(f"error at zeta={zeta}, logMmin={logMmin}: {e}")
        return -np.inf

    if model_peak == -np.inf or model_width == -np.inf:
        return -np.inf # the rejection of points with -np.inf log-likelihood is completely automatic
    
    delta_peak = obs_peak - model_peak
    delta_fwhm = obs_fwhm - model_width
    chi2 = (delta_peak / sigma_peak) ** 2 + (delta_fwhm / sigma_fwhm) ** 2

    return -0.5 * chi2 

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta) #logPrior + logLikelihood = logPosterior: summation of log probabilities is product of probabilities

# MCMC with emcee
ndim = 2  # number of dimensions in the parameter space (zeta, logMmin)
nwalkers = 4  # number of independent Markov chains (walkers) that run concurrently. recommended: at least 4 times ndim ??
nsteps = 10000 # number of iterations each walker will take. #real runs: 10_000 ?

initial = np.array([25.08, 9.0])
#initial = np.array([(np.random.uniform(1.0,50.0), np.random.uniform(7.0, 11.0)) for _ in range(nwalkers)])
#INITIAL LOSE TO MY TRUE VALUE
pos = initial +  0.1 * np.random.randn(nwalkers, ndim) # add a small gaussian perturbation to the initial positions to ensure diversity among walkers 

if __name__ == '__main__':
    with multiprocessing.Pool(processes=4) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior) #, pool=pool)

        start = time.time()
        print("Running MCMC...")

        sampler.run_mcmc(pos, nsteps, progress=True)

        end = time.time()
        print("MCMC completed.")

        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        # autocorrelation time
        print("Calculating autocorrelation time...")
        try:    
            tau = sampler.get_autocorr_time()
            print("Autocorrelation time:", tau) # only about X steps are needed for the chain to “forget” where it started => discard = X for sampler.get_chain()
        except emcee.autocorr.AutocorrError as e:
            print(f"Autocorrelation error: {e}")
            tau = np.inf
        
    # results
    samples = sampler.get_chain(discard=100, thin=1, flat=True) #  time series of the parameters in the chain. Check what discard and thin values are needed
#100

    # for the cluster maybe write and save them in a txt output file 
    np.save("mcmc_samples.npy", samples)
    np.save("mcmc_chain.npy", sampler.get_chain())
    np.save("mcmc_log_prob.npy", sampler.get_log_prob())

    idata = az.from_emcee(sampler)
    az.summary(idata)


    # OK, trace plot: positions of each walker as a function of the number of steps in the chain TO CHECK CONVERGENCE
    fig, axes = plt.subplots(ndim, figsize=(10, 5), sharex=True)
    for i, label in zip(range(ndim), [r'$\zeta$', r'$\log_{10}(M_{\mathrm{min}})$']):
        axes[i].plot(sampler.get_chain()[:, :, i], "k", alpha=0.3)
        axes[i].set_ylabel(f"{label}")
    axes[-1].set_xlabel("step number")
    plt.tight_layout()
    plt.savefig("/Users/sophiatonelli/Desktop/pngs_mpia/w7/mcmc/trace3_plot.png")
    plt.show()



    # OK, corner plot: visualize the posterior distribution of the parameters (what your data imply about model parameters = parameter inference)
    truths = np.array([25.08, 9.0]) # 4th value is: XHI=0.5 and logMmin=9
    #means = np.mean(samples, axis=0)
    #stds = np.std(samples, axis=0)
    fig = corner.corner(samples, labels=[r'$\zeta$', r'$\log_{10}(M_{\mathrm{min}})$'], truths=truths, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12},)
    axes = np.array(fig.axes).reshape((2, 2)) # since ndim=2
    plt.savefig("/Users/sophiatonelli/Desktop/pngs_mpia/w7/mcmc/corner_plot3.png")
    plt.show()

    ########



    # another diagnostic plot is the projection of MCMC results into the observed data space. TO  TEST HOW GOOD MY MODEL is at describing data (model checking)
    print("Plotting model predictions from posterior samples...")
    predicted_peaks = []

    inds = np.random.choice(len(samples), size=100, replace=False) # sample 100 random points from posterior
    for ind in inds:
        zeta_sample, logMmin_sample = samples[ind]
        try:
            predicted_peak = some_model_photon_conserving(zeta_sample, logMmin_sample)
            predicted_peaks.append(predicted_peak)
        except Exception as e:
            print(f"Error for sample {ind}: {e}")
            predicted_peaks.append(np.nan)

    predicted_peaks = np.array(predicted_peaks)
    # remove failed samples
    predicted_peaks = predicted_peaks[~np.isnan(predicted_peaks)]
    plt.figure(figsize=(8, 5))
    plt.hist(predicted_peaks, bins=30, alpha=0.7, label='Predicted $R_\\mathrm{peak}$ from posterior')
    plt.axvline(obs_peak, color='r', linestyle='--', label='Observed $R_\\mathrm{peak}$')
    plt.xlabel("$R_\\mathrm{peak}$ [cMpc/h]")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Posterior Predictive Check: $R_\\mathrm{peak}$")
    plt.tight_layout()
    plt.savefig("/Users/sophiatonelli/Desktop/pngs_mpia/w7/mcmc/model_check3.png")
    plt.show()

    mean_pred = np.mean(predicted_peaks)
    std_pred = np.std(predicted_peaks)
    print(f"Predicted R_peak: {mean_pred:.2f} ± {std_pred:.2f} cMpc/h")
    print(f"Observed R_peak: {obs_peak:.2f} cMpc/h")
 

    # OK, print parameter estimates
    print("\nParameter estimates:")
    labels = [r'$\zeta$', r'$\log_{10}(M_{\mathrm{min}})$']
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))


    # OK, posterior summary
    for i, label in enumerate([r'$\zeta$', r'$\log_{10}(M_{\mathrm{min}})$']):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{label}: {mcmc[1]:.3f} (+{q[1]:.3f}, -{q[0]:.3f})")
