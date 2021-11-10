from utils import *
import numpy as np
import matplotlib.pyplot as plt
from mne_connectivity import phase_slope_index


eng = setup_matlab()

# data generation parameters
n_trials = float(4)
n_obs = float(500)
adj_net = eng.tnet9()
n_nodes = len(adj_net)
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

# plot test network and its connectivity matrix
plot_network(adj_net, save=True)

# generate and plot random VAR test data
data = simulate_data(n_trials, n_obs, adj_net, p, rho, wvar, rmi, py=True)
plot_timeseries(data, sfreq, save=True)

# compute PSI
data = np.transpose(data, (2, 0, 1))
psi_connectivity = phase_slope_index(data, mode='multitaper', sfreq=sfreq,
                                     fmin=fmin, fmax=fmax, tmin=tmin)

# plot estimated connectivity matrix
psi = psi_connectivity.get_data()
psi = psi.reshape((n_nodes, n_nodes))
plot_psi(psi, save=True)
plt.show()
