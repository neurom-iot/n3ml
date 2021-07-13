import numpy as np
import matplotlib.pyplot as plt


class PlotW:
    def __init__(self):
        self.mat = plt.matshow()

    def __call__(self, w):
        pass


def plot_w(fig, mat, w):

    if mat is None or fig is None:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mat = ax.matshow(w[0].reshape(28, 28))
        mat.set_clim(0, 1)
        fig.colorbar(mat)
        return fig, mat
    plt.gcf()
    # print(w[0].reshape(28, 28))
    mat.set_data(w[0].reshape(28, 28))
    fig.canvas.draw()
    return fig, mat


def plot(fig, mat, w):
    if mat is None or fig is None:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ww = np.zeros((280, 280))
        # ww = np.zeros((28 * 20, 28 * 20))
        for r in range(10):
            for c in range(10):
                ww[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = w[r * 10 + c].reshape(28, 28)
        # for r in range(20):
        #     for c in range(20):
        #         ww[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = w[r * 20 + c].reshape(28, 28)
        mat = ax.matshow(ww, cmap='hot_r')
        mat.set_clim(0, 1)
        fig.colorbar(mat)
        return fig, mat
    ww = np.zeros((280, 280))
    # ww = np.zeros((28 * 20, 28 * 20))
    for r in range(10):
        for c in range(10):
            ww[r*28:(r+1)*28, c*28:(c+1)*28] = w[r*10+c].reshape(28, 28)
    # for r in range(20):
    #     for c in range(20):
    #         ww[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = w[r * 20 + c].reshape(28, 28)
    plt.gcf()
    mat.set_data(ww)
    fig.canvas.draw()
    return fig, mat
