from cobaya.analyse import Analyzer 
import corner
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11})

output_path = "/Users/sophiatonelli/Desktop/w6_code/finding_zetas_ES_PC/chains/mcmc_output"

#analyze the chain
analyzer = Analyzer(output_path)
samples = analyzer.get_chain()
param_names = analyzer.products["ordered_params"]
truths = [10, 1e9]  # true values for zeta, Mmin

#plot
fig = corner.corner(samples, labels=param_names, show_titles=True)
fig = corner.corner(samples, labels=param_names, truths=truths, show_titles=True)
plt.show()
