import numpy as np
import matplotlib.pyplot as plt
import mne


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

    plt.show()
