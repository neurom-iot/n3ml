import numpy as np
import matplotlib.pyplot as plt

def rasterplot(time, spikes, **kwargs):
    ax = plt.gca()

    n_spike, n_neuron = spikes.shape

    kwargs.setdefault("linestyle", "None")
    kwargs.setdefault("marker", "|")

    spiketimes = []

    for i in range(n_neuron):
        temp = time[spikes[:, i] > 0].ravel()
        spiketimes.append(temp)

    spiketimes = np.array(spiketimes)

    indexes = np.zeros(n_neuron, dtype=np.int)

    for t in range(time.shape[0]):
        for i in range(spiketimes.shape[0]):
            if spiketimes[i].shape[0] <= 0:
                continue
            if indexes[i] < spiketimes[i].shape[0] and \
                    time[t] == spiketimes[i][indexes[i]]:
                ax.plot(
                    spiketimes[i][indexes[i]],
                    i + 1,
                    'k',
                    **kwargs
                )

                plt.draw()
                plt.pause(0.001)

                indexes[i] += 1