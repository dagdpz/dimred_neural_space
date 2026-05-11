import numpy as np
from numpy.strings import title
from scipy import signal
from dPCA import dPCA
from matplotlib import pyplot as plt


def main(seed=0):
    rng = np.random.default_rng(seed)

    # Building the dataset
    N_trials = 30
    N_units = 3
    N_space = 2
    N_hand = 2
    N_task = 2
    N_time = 500
    dt = 0.01

    noise = 0.08
    signal = 1.0

    time = np.linspace(0, 1, N_time)
    space_vals = np.arange(N_space)  # 0 = left, 1 = right
    hand_vals = np.arange(N_hand)  # 0 = left, 1 = right
    task_vals = np.arange(N_task)

    space_code = 2 * space_vals - 1
    hand_code = 2 * hand_vals - 1
    task_code = 2 * task_vals - 1

    cue = np.exp(-0.5 * ((time - 0.0) / 0.15) ** 2)
    delay = np.exp(-0.5 * ((time - 0.55) / 0.35) ** 2)
    move = np.exp(-0.5 * ((time - 1.0) / 0.18) ** 2)
    ramp = 1 / (1 + np.exp(-(time - 0.25) / 0.12))

    cue /= cue.max()
    delay /= delay.max()
    move /= move.max()
    ramp /= ramp.max()

    baseline = rng.lognormal(mean=np.log(8), sigma=0.5, size=N_units)

    w_time = rng.normal(0, 3.0, N_units)

    w_space = rng.normal(0, 5.0, N_units)
    w_hand = rng.normal(0, 5.0, N_units)
    w_task = rng.normal(0, 4.0, N_units)

    # mixed selectivity terms
    w_space_hand = rng.normal(0, 3.0, N_units)
    w_space_task = rng.normal(0, 3.0, N_units)
    w_hand_task = rng.normal(0, 3.0, N_units)
    w_space_hand_task = rng.normal(0, 2.0, N_units)

    a_cue = rng.uniform(0.2, 1.2, N_units)
    a_delay = rng.uniform(0.2, 1.2, N_units)
    a_move = rng.uniform(0.2, 1.2, N_units)

    temporal_kernel = (
        a_cue[:, None] * cue[None, :]
        + a_delay[:, None] * delay[None, :]
        + a_move[:, None] * move[None, :]
    )

    # shape: units × time
    temporal_kernel /= temporal_kernel.max(axis=1, keepdims=True)

    rate = np.zeros((N_units, N_space, N_hand, N_task, N_time))

    for s in range(N_space):
        for h in range(N_hand):
            for k in range(N_task):

                sc = space_code[s]
                hc = hand_code[h]
                kc = task_code[k]

                # condition-dependent modulation per neuron
                condition_drive = (
                    w_space * sc
                    + w_hand * hc
                    + w_task * kc
                    + w_space_hand * sc * hc
                    + w_space_task * sc * kc
                    + w_hand_task * hc * kc
                    + w_space_hand_task * sc * hc * kc
                )

                # time-dependent response
                # condition_drive is modulated by temporal kernel
                rate[:, s, h, k, :] = (
                    baseline[:, None]
                    + w_time[:, None] * ramp[None, :]
                    + condition_drive[:, None] * temporal_kernel
                )

    # keep firing rates positive
    rate = np.clip(rate, 0.1, None)

    trialR = np.zeros((N_trials, N_units, N_space, N_hand, N_task, N_time))

    for tr in range(N_trials):
        # trial-to-trial multiplicative gain noise
        gain = rng.lognormal(mean=0.0, sigma=0.12, size=(N_units, 1, 1, 1, 1))

        # slow additive shared noise across time
        slow_noise = rng.normal(0, 0.6, size=(N_units, 1, 1, 1, 1))

        trial_rate = gain * rate[None, :, :, :, :, :] + slow_noise[None, :, :, :, :, :]
        trial_rate = np.clip(trial_rate, 0.1, None)

        # Poisson spike counts in bins, converted back to Hz
        counts = rng.poisson(trial_rate * dt)
        trialR[tr] = counts / dt

    # Trial-average response for dPCA
    R = trialR.mean(axis=0)

    # Center each neuron over all conditions and time
    R -= R.reshape(N_units, -1).mean(axis=1)[:, None, None, None, None]

    dpca = dPCA.dPCA(labels="shkt", n_components=5, regularizer="auto")
    dpca.protect = ["t"]

    Z = dpca.fit_transform(R, trialR)

    # -----------------------------
    # Plot raw neural state space
    # using first three neurons
    # -----------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for space in range(2):
        for hand in range(2):
            for task in range(2):
                x = R[0, space, hand, task, :]
                y = R[1, space, hand, task, :]
                z = R[2, space, hand, task, :]

                ax.plot(x, y, z, label=f"s={space}, h={hand}, k={task}")

    ax.set_xlabel("Neuron 1")
    ax.set_ylabel("Neuron 2")
    ax.set_zlabel("Neuron 3")
    ax.set_title("Raw neural state space: first 3 neurons")
    ax.legend()
    plt.show()

    # -----------------------------
    # Plot dPCA state space
    # -----------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for space in range(2):
        for hand in range(2):
            for task in range(2):
                x = Z["st"][0, space, hand, task, :]
                y = Z["ht"][0, space, hand, task, :]
                z = Z["kt"][0, space, hand, task, :]

                ax.plot(x, y, z, label=f"s={space}, h={hand}, k={task}")
                ax.scatter(x[0], y[0], z[0], marker="o")
                ax.scatter(x[-1], y[-1], z[-1], marker="x")

    ax.set_xlabel("space-time dPC 1")
    ax.set_ylabel("hand-time dPC 1")
    ax.set_zlabel("task-time dPC 1")
    ax.set_title("dPCA-demixed state space")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
