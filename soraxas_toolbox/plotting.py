from __future__ import annotations
# pyright: reportPrivateImportUsage=false

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


def imshow_with_cbar(fig: plt.Figure, ax: plt.Axes, data: np.ndarray, **kwargs):
    """
    Normally subplot cannot have colorbar. This function is to add colorbar to the subplot.
    """
    im1 = ax.imshow(data, **kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax, orientation="vertical")


def subplots(
    nrows,
    ncols: int = 1,
    figsize=None,
    base_row_size: int = 5,
    base_col_size: int = 4,
    row_factor: float = 1.0,
    col_factor: float = 1.0,
    scale_factor: float = 1.0,
    tight_layout: bool = True,
    **kwargs,
):
    if figsize is not None:
        raise ValueError("Cannot use figsize when auto-sizing")
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            base_col_size * ncols * col_factor * scale_factor,
            base_row_size * nrows * row_factor * scale_factor,
        ),
        **kwargs,
    )
    if tight_layout:
        fig.set_tight_layout(True)
    return fig, axes


def histogram(
    img: np.ndarray,
    bins: int = 150,
    ylog: bool = False,
    partile_title: str = "",
    title="",
) -> plt.Figure:
    plt.figure()
    if img.ndim == 2:
        plt.hist(img.ravel(), bins=bins, color="gray", alpha=0.7, density=True)
        partial_title = "Grayscale Histogram"
    elif img.ndim == 3:
        colors = ["r", "g", "b"]
        for i, color in enumerate(colors):
            plt.hist(
                img[..., i].ravel(),
                bins=bins,
                color=color,
                alpha=0.5,
                label=f"{color.upper()} channel",
                density=True,
            )
        partial_title = "RGB Histogram"
        plt.legend()
    else:
        print("Unsupported image format.")
        raise ValueError()

    if not title and partile_title:
        title = f"{partial_title} ({partile_title})"

    plt.title(title)

    pctile_99 = np.percentile(img, 99)
    pctile_1 = np.percentile(img, 1)

    plt.xlabel(
        f"(min: {img.min()}, max: {img.max()}) Percentile: [1% {pctile_1:.1f} | 99% {pctile_99:.1f}]"
    )
    if ylog:
        plt.yscale("log")
        plt.ylabel("Density (log scale)")
    else:
        plt.ylabel("Density")
    return plt.gcf()
