import os
from shutil import which
from subprocess import Popen, PIPE

from typing import IO, Callable

import tqdm
from PIL.Image import Image

from . import easy_with_blocks, notebook

if not which("timg"):
    raise ValueError("timg not installed")


class TerminalImageViewer:
    def __init__(self, get_stdout: bool = False):
        cmd = ["timg", "-"]
        if all(token in os.environ for token in ("TERMINAL_WIDTH", "TERMINAL_HEIGHT")):
            cmd.append(
                f"-g{os.getenv('TERMINAL_WIDTH')}x{os.getenv('TERMINAL_HEIGHT')}"
            )
        self.program = Popen(
            cmd,
            stdin=PIPE,
            stdout=PIPE if get_stdout else None,
            bufsize=-1,
        )

    def __enter__(self):
        self.program.__enter__()
        return self

    def __exit__(self, exc_type, value, traceback):
        self.program.__exit__(exc_type, value, traceback)

    @property
    def stream(self) -> IO:
        return self.program.stdin


def __send_to_display(save_to_straem: Callable[[IO], None], pbar: tqdm.tqdm = None):
    if notebook.is_notebook():
        from io import BytesIO
        from IPython.display import Image, display

        with BytesIO() as stream:
            save_to_straem(stream)
            stream.seek(0)

            display(Image(stream.read()))
    else:
        with TerminalImageViewer(get_stdout=pbar is not None) as viewer:
            save_to_straem(viewer.stream)
            if pbar is not None:
                out, err = viewer.program.communicate()
                pbar.write(out.decode())


import sys


def module_was_imported(module_name: str):
    return module_name in sys.modules


def display(image, pbar: tqdm.tqdm = None, format: str = "bmp") -> None:
    if isinstance(image, Image):
        return __send_to_display(
            lambda stream: image.save(stream, format=format), pbar=pbar
        )

    if module_was_imported("torch"):
        import torch

        if isinstance(image, torch.Tensor):
            with easy_with_blocks.NoMissingModuleError(strong_warning=True):
                import torchvision

                return __send_to_display(
                    lambda stream: torchvision.utils.save_image(
                        image, stream, format=format
                    ),
                    pbar=pbar,
                )

    if module_was_imported("matplotlib"):
        import matplotlib.figure

        if isinstance(image, matplotlib.figure.Figure):
            return __send_to_display(lambda stream: image.savefig(stream), pbar=pbar)

    raise ValueError(f"Unknown format of type {type(image)} with input {image}")


def view_high_dimensional_embeddings(
    x: "np.ndarray", label=None, title="High-d embeddings"
):
    from sklearn.manifold import TSNE
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    if label is not None:
        assert len(label) == x.shape[0], f"{len(label)} != {x.shape}"

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x)

    df = pd.DataFrame()
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    kwargs = dict()
    if label is not None:
        df["y"] = label
        num_classes = len(np.unique(label))

        kwargs["hue"] = df.y.tolist()
        kwargs["palette"] = sns.color_palette("hls", num_classes)

    sns.scatterplot(x="comp-1", y="comp-2", data=df, **kwargs).set(title=title)

    display(plt.gcf())
    plt.clf()
