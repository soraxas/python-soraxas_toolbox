import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable


class MatrixPlotter:
    """An object that display and update 2D array plot."""

    def __init__(self, disable: bool = False):
        self.disable = disable
        if self.disable:
            return
        self.fig = plt.figure()
        # ax = self.fig.add_subplot(111)
        self.ax = self.fig.add_subplot(111)
        self.colorbar = None

    def update(self, data: np.ndarray, normalise: bool = False, block: bool = False):
        _data = data / data.sum() if normalise else data

        cax = self.ax.matshow(_data, interpolation="nearest")
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(cax)
        else:
            self.colorbar.update_bruteforce(cax)
        plt.draw()
        plt.pause(0.00001)
        if block:
            plt.show()


def imshow_with_colorbar(fig: plt.Figure, ax: plt.Axes, data: np.ndarray, **kwargs):
    """
    Normally subplot cannot have colorbar. This function is to add colorbar to the subplot.
    """
    im1 = ax.imshow(data, **kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax, orientation="vertical")
