from utils.setup_matlab import setup_matlab
from utils.simulate_data import simulate_data

import numpy as np
import mne
from mne_connectivity import phase_slope_index


eng = setup_matlab()

# data generation parameters
n_trials = float(4)
n_obs = float(500)
net = eng.tnet5()
p = float(6)
rho = 0.95
wvar = 0.5
rmi = 0.5
sfreq = 200

# phase slope index (PSI) parameters
fmin = 10
fmax = 20
tmin = 0

# set RNG
seed = 123
eng.rng_seed(seed)

# generate random VAR test data
data = simulate_data(n_trials, n_obs, net, p, rho, wvar, rmi, py=True)

# compute PSI
data = np.transpose(data, (2, 0, 1))
psi = phase_slope_index(data, mode='multitaper', sfreq=sfreq, fmin=fmin,
                        fmax=fmax, tmin=tmin)
