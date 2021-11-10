import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import networkx as nx
import mne


def setup_matlab():
    eng = matlab.engine.start_matlab()

    # add MVGC2 paths
    eng.addpath("~/Documents/MATLAB/MVGC2/core")
    eng.addpath("~/Documents/MATLAB/MVGC2/gc/var")
    eng.addpath("~/Documents/MATLAB/MVGC2/utils")
    eng.addpath("~/Documents/MATLAB/MVGC2/stats")
    eng.addpath("~/Documents/MATLAB/MVGC2/demo")

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
        data (3D array): matrix containing number of network nodes, number of
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


def plot_network(adj_net, save=True):
    """Plots network with its adjacency or connectivity matrix.

    Args:
        adj_net (2D array): matrix representing a graph/network.
        save (bool): whether to save figures (true by default).

    """
    if not isinstance(adj_net, np.ndarray):
        adj_net = np.array(adj_net)

    if len(adj_net.shape) != 2:
        raise ValueError("Adjacency matrix must have two dimensions")

    if adj_net.shape[0] != adj_net.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    n_nodes = len(adj_net)

    # network
    g = nx.convert_matrix.from_numpy_matrix(adj_net, create_using=nx.DiGraph())
    labels = {n: n + 1 for n in range(n_nodes)}
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(g, arrows=True, with_labels=True, labels=labels, node_size=1400,
            node_color='red', alpha=0.8, font_size=15, edgecolors='black',
            arrowsize=12, pos=nx.circular_layout(g), ax=ax)

    if save:
        fig.savefig(f"figures/net{n_nodes}/{n_nodes}-node network.pdf")

    # remove self-connections from connectivity matrix
    for r in range(adj_net.shape[0]):
        for c in range(adj_net.shape[1]):
            if r == c:
                adj_net[r, c] = 0

    # connectivity
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [0, 1, 2, 3, 4, 5]
    img = ax.imshow(adj_net, cmap='Purples', origin='lower')
    ax.set_title("True connectivity matrix", fontsize=18, fontweight='bold')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("From", labelpad=15, fontsize=20)
    ax.set_ylabel("To", labelpad=15, fontsize=20)
    cbar = fig.colorbar(img)
    cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()

    if save:
        fig.savefig(f"figures/net{n_nodes}/{n_nodes}-node network connectivity matrix.pdf")


def plot_timeseries(data, sfreq, save=True):
    """Plots timeseries for given trials at a given sampling rate.

    Args:
        data (3D array): matrix containing number of network nodes, number of
            observations or samples and number of trials or epochs.
        sfreq (int): sampling frequency (Hz).
        save (bool): whether to save figures (true by default).

    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    n_nodes = data.shape[0]
    n_trials = data.shape[2]
    info = mne.create_info(n_nodes, sfreq)

    # plot data for all trials/epochs
    for n in range(n_trials):
        trial = mne.io.RawArray(data[:, :, n], info=info)
        trial.plot()
        if save:
            plt.savefig(f"figures/net{n_nodes}/Time series of trial #{n}.pdf")


def plot_psi(psi, save=True):
    """Plots connectivity matrix estimated with phase slope index (PSI).

    Args:
        psi (2D array): PSI-estimated connectivity matrix.
        save (bool): whether to save figure (true by default).

    """

    if len(psi.shape) != 2:
        raise ValueError("Connectivity matrix must have two dimensions")

    if psi.shape[0] != psi.shape[1]:
        raise ValueError("Connectivity matrix must be square")

    n_nodes = len(psi)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [n for n in range(n_nodes+1)]
    img = ax.imshow(psi, cmap='bwr', norm=colors.CenteredNorm(), origin='lower')
    ax.set_title("PSI connectivity matrix", fontsize=18, fontweight='bold')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("From", labelpad=15, fontsize=20)
    ax.set_ylabel("To", labelpad=15, fontsize=20)
    cbar = fig.colorbar(img)
    cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()

    if save:
        fig.savefig(f"figures/net{n_nodes}/PSI connectivity matrix.pdf")
