import matlab.engine

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne_connectivity import phase_slope_index

eng = matlab.engine.start_matlab()

# add MVGC2 paths
eng.addpath("/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/core")
eng.addpath("/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/utils")
eng.addpath("/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/demo")

# test data generation
n_trials = float(4)
n_obs = float(500)
sfreq = float(200)

# actual (ground truth) VAR model generation parameters
tnet = eng.tnet5()
moact = float(6)
rho = 0.95
wvar = 0.5
rmi = 0.5

# generate random VAR test data

# set RNG
seed = float(123)
eng.rng(seed)

# generate random VAR coefficients for test network
AA = eng.var_rand(tnet, moact, rho, wvar)
nvars = eng.size(AA, 1)

# generate random residuals covariance (in fact correlation) matrix
VV = eng.corr_rand(nvars, rmi)

# report information on the generated VAR model and check for errors
info = eng.var_info(AA, VV)
# assert info["error"], "VAR error(s) found - bailing out"

# generate multi-trial VAR time series data with normally distributed residuals
# for generated VAR coefficients and residuals covariance matrix
X = eng.varfima_to_tsdata(AA, [], [], VV, n_obs, n_trials)

# convert matlab matrix to np array and to mne raw array
X = np.array(X)
n_nodes = 5
info = mne.create_info(n_nodes, sfreq)
example_trial = mne.io.RawArray(X[:, :, 0], info=info)

# plot example trial (# nodes x # observations)
example_trial.plot()
#plt.show()

# reshape data for computing phase slope index (PSI)
reshaped_X = np.transpose(X, (2, 0, 1))

# compute PSI
indices = (np.array([0]), np.array([1]))
fmin = 10
fmax = 20
tmin_con = 0
psi = phase_slope_index(reshaped_X, mode='multitaper', indices=indices,
                        sfreq=sfreq, fmin=fmin, fmax=fmax, tmin=tmin_con)
print()
print(f"Phase slope index is of type {type(psi)}")
print()
print(f"The shape of PSI is {psi.shape}")
