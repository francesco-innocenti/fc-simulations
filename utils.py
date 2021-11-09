import matlab.engine
import numpy as np
import networkx as nx
import mne


def setup_matlab():
    eng = matlab.engine.start_matlab()

    # add MVGC2 paths
    eng.addpath("/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/core")
    eng.addpath("/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/gc/var")
    eng.addpath("/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/utils")
    eng.addpath("/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/stats")
    eng.addpath("/Users/FrancescoInnocenti/Documents/MATLAB/MVGC2/demo")

    return eng


def simulate_data(n_trials, n_obs, adj_net, moact, rho, wvar, rmi, demean=True,
                  py=False):
    """Generates data from a random VAR model for a given network.

    Args:
        n_trials (float): number of trials or epochs.
        n_obs (float): number of observations (sampled time points) per trial.
        adj_net: adjacency matrix of a network/graph.
        moact (float): VAR model order (number of time lags).
        rho (float): spectral radius.
        wvar (float): VAR coefficients decay weighting factor.
        rmi (float): residuals log-generalised correlation (multi-information).
        demean (bool): whether to remove temporal mean and normalise by temporal
            variance. True by default.
        py (bool): whether to convert returned matlab array into a numpy
            array for Python. False by default.

    Returns:
        data (array): 3D matrix containing number of network nodes, number of
            observations or samples and number of trials or epochs.
    """

    eng = setup_matlab()

    # generate random VAR coefficients for test network
    AA = eng.var_rand(adj_net, moact, rho, wvar)
    n_vars = eng.size(AA, 1)

    # generate random residuals covariance (in fact correlation) matrix
    VV = eng.corr_rand(n_vars, rmi)

    # report information on the generated VAR model
    info = eng.var_info(AA, VV)

    # generate multi-trial VAR time series data with normally distributed
    # residuals for generated VAR coefficients and residuals covariance matrix
    data = eng.var_to_tsdata(AA, VV, n_obs, n_trials)

    if demean:
        data = eng.demean(data, True)

    if py:
        data = np.array(data)

    return data


def plot_network(adj_net):
    """Plots graph network from adjacency matrix.

    Args:
        adj_net (array): (2D) matrix representing graph/network.

    """
    if not isinstance(adj_net, np.ndarray):
        adj_net = np.array(adj_net)

    if len(adj_net.shape) != 2:
        raise ValueError("Adjacency matrix must have two dimensions")

    g = nx.convert_matrix.from_numpy_matrix(adj_net)
    labels = {n: n + 1 for n in range(len(adj_net))}
    nx.draw(g, arrows=True, with_labels=True, labels=labels, node_size=1400,
            node_color='red', alpha=0.8, font_size=15, edgecolors='black',
            arrowsize=12, pos=nx.circular_layout(g))


def plot_timeseries(data, sfreq):
    """Plots timeseries for given trials at a given sampling rate.

    Args:
        data (array): 3D matrix containing number of network nodes, number of
            observations or samples and number of trials or epochs.
        sfreq (int): sampling frequency (Hz).

    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    n_nodes = data.shape[0]
    n_trials = data.shape[2]
    info = mne.create_info(n_nodes, sfreq)

    # plot data for all trials/epochs
    for i in range(n_trials):
        trial = mne.io.RawArray(data[:, :, i], info=info)
        trial.plot()
