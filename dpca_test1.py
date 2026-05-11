import numpy as np
from numpy.strings import title
from dPCA import dPCA
from matplotlib import pyplot as plt


def dpca_test():
    # We are working with 100 neurons, 250 time points and 6 different stimuli
    n_neuron, time_points, stim = 100, 250, 6

    # We are working with 10 trials
    noise, n_samples = 0.2, 10

    zt = np.arange(time_points) / float(time_points)
    zs = np.arange(stim) / float(stim)

    trialR = noise * np.random.randn(n_samples, n_neuron, stim, time_points)

    # add a time-dependent signal and stimulus dependent signal to every neuron
    trialR += np.random.randn(n_neuron)[None, :, None, None] * zt[None, None, None, :]
    trialR += np.random.randn(n_neuron)[None, :, None, None] * zs[None, None, :, None]

    R = np.mean(trialR, 0)
    R -= np.mean(R.reshape((n_neuron, -1)), 1)[:, None, None]

    dpca = dPCA.dPCA(labels="st", regularizer="auto")
    dpca.protect = ["t"]

    Z = dpca.fit_transform(R, trialR)

    """ significance_masks = dpca.significance_analysis(
        R, trialR, n_shuffles=10, n_splits=10, n_consecutive=10
    )
    print(significance_masks) """

    for s in range(stim):
        x = Z["s"][0, s, :]  # stimulus component 1
        y = Z["t"][0, s, :]  # time component 1
        plt.plot(x, y, label=f"stim {s}")

        # mark start and end
        plt.scatter(x[0], y[0], marker="o")
        plt.scatter(x[-1], y[-1], marker="x")

    plt.xlabel("dPC 1: stimulus")
    plt.ylabel("dPC 1: time")
    plt.title("dPCA state-space trajectories")
    plt.legend()
    plt.show()
