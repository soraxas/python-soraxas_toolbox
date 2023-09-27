import os
import sys
import math
import warnings
from shutil import which
from subprocess import Popen, PIPE

from typing import IO, Callable, Optional, Tuple, Union, List, Any

import tqdm
import PIL.Image

from . import easy_with_blocks, notebook

if not which("timg"):
    raise ValueError("timg not installed")


############################################################
##             Turn any matplotlib plt to img             ##
############################################################


def plt_fig_to_nparray(fig):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
    Return:
        np.array with shape = (x,y,d) where d=3
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # remove white padding
    fig.tight_layout()

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8
    plt.close(fig)
    return img
    # # Add figure in numpy "image" to TensorBoard writer
    # writer.add_image('confusion_matrix', img, step)


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


def module_was_imported(module_name: str):
    return module_name in sys.modules


def _get_new_shape_maintain_ratio(
    target_size: Union[Tuple, List, float, int], current_shape: Tuple[int, int]
):
    if isinstance(target_size, (tuple, list)):
        # no need to do anything
        return target_size

    assert isinstance(target_size, (int, float))

    if current_shape[0] >= current_shape[1]:
        i1, i2 = 0, 1
    else:
        i1, i2 = 1, 0
    ratio = current_shape[i2] / current_shape[i1]
    new_shape = [0, 0]
    new_shape[i1] = int(target_size)
    new_shape[i2] = int(math.ceil(target_size * ratio))
    return new_shape


from abc import ABC


class ArrayAutoFixer(ABC):
    color_channel_idx: int
    auto_fix_channel_idx: int
    module: Any

    @classmethod
    def float_types(cls):
        return cls.module.float16, cls.module.float32, cls.module.float64

    @classmethod
    def cls_var_setter(cls, module):
        cls.module = module

    @classmethod
    def fix_channel(cls, x):
        # if color channel is at the last dim, move it to the first dim (if not batched)
        # _module is either torch or numpy

        if len(x.shape) < 3:
            return x

        # color_channel_idx = 0 (the following can handle batched image as well)

        # move color channel to corresponding idx
        # if color channel is at the last dim, move it to the first dim (if not batched)
        if x.shape[cls.color_channel_idx] not in (1, 3, 4) and x.shape[
            cls.auto_fix_channel_idx
        ] in (1, 3, 4):
            x = cls.module.moveaxis(x, cls.auto_fix_channel_idx, cls.color_channel_idx)

        return x

    @classmethod
    def fix_dtype(cls, x):
        if x.dtype != cls.module.uint8 and x.dtype in cls.float_types():
            x = (x * 255).astype(cls.module.uint8)
        return x

    @classmethod
    def fix_float_range(cls, image, normalise: bool):
        # works for either numpy or torch
        _min = image.min()
        _max = image.max()
        _warning_msg = None
        if normalise:
            image = (image - _min) / (_max - _min)
        elif _min < 0 or _max > 1:
            if normalise is not False:
                _warning_msg = (
                    "Given image is of float-type, but its value is not between [0, 1]."
                )
                if _min >= 0 and _max <= 255:
                    _warning_msg = f"{_warning_msg} Seems it's within [0, 255]. Gonna normalising it based on this."
                image = image / 255

        if _warning_msg is not None:
            warnings.warn(_warning_msg, RuntimeWarning)

        return image


class NumpyArrayAutoFixer(ArrayAutoFixer):
    # Width X Height X Color
    color_channel_idx: int = -1
    auto_fix_channel_idx: int = -3


class TorchArrayAutoFixer(ArrayAutoFixer):
    #  Batch X ... X Color X Width X Height
    color_channel_idx: int = -3
    auto_fix_channel_idx: int = -1

    @classmethod
    def infer_is_batch(cls, image):
        # try to infer it
        if len(image.shape) >= 4:
            return True
        elif len(image.shape) == 2:
            return False
        elif len(image.shape) == 3:
            if image.shape[0] in (1, 3):
                # probably is gray-scale or RGB image
                return False
            else:
                # since it has non-standard number of channels, probably its a batched grey-scale image?
                return True
        raise NotImplementedError()


def display(
    image,
    pbar: tqdm.tqdm = None,
    format: str = "bmp",
    normalise: Optional[bool] = None,
    target_size: Tuple[int, int] = None,
    is_batched: Optional[bool] = None,
    is_grayscale: Optional[bool] = None,
) -> None:
    if module_was_imported("numpy") and module_was_imported("PIL"):
        import numpy

        if isinstance(image, numpy.ndarray):
            NumpyArrayAutoFixer.cls_var_setter(module=numpy)
            if image.dtype in NumpyArrayAutoFixer.float_types():
                # scale if necessary
                image = NumpyArrayAutoFixer.fix_float_range(image, normalise=normalise)
            # to uint8 if necessary
            image = NumpyArrayAutoFixer.fix_channel(image)
            image = NumpyArrayAutoFixer.fix_dtype(image)
            # convert to pillow image
            image = PIL.Image.fromarray(image)

    if isinstance(image, PIL.Image.Image):
        if target_size is not None:
            image = image.resize(_get_new_shape_maintain_ratio(target_size, image.size))
        return __send_to_display(
            lambda stream: image.save(stream, format=format), pbar=pbar
        )

    if module_was_imported("torch"):
        import torch

        if isinstance(image, torch.Tensor):
            TorchArrayAutoFixer.cls_var_setter(module=torch)

            with torch.no_grad():
                with easy_with_blocks.NoMissingModuleError(strong_warning=True):
                    import torchvision
                    from PIL import Image as PILImage

                    image = image.to("cpu")

                    if image.dtype == torch.uint8:
                        image = image.float() / 255
                        # # directly display it
                        # return __send_to_display(
                        #     lambda stream: PILImage.fromarray(
                        #         image.to("cpu", torch.uint8).numpy()
                        #     ).save(stream, format=format),
                        #     pbar=pbar,
                        # )

                    # for float or double
                    image = TorchArrayAutoFixer.fix_float_range(
                        image, normalise=normalise
                    )

                    if is_grayscale:
                        if len(image.shape) > 2:
                            is_batched = True
                            image = image.reshape(-1, image.shape[-2], image.shape[-1])

                    # potentially fix color channel index
                    image = TorchArrayAutoFixer.fix_channel(image)

                    if is_batched is None:
                        is_batched = TorchArrayAutoFixer.infer_is_batch(image)

                    if is_batched:
                        if len(image.shape) == 2:
                            warnings.warn(
                                "Input is said to be batched, but it only has 2 dim. Skipping any remaining action as it doesn't make sense.",
                                RuntimeWarning,
                            )
                        elif len(image.shape) == 3:
                            # is gray scale
                            image = image.unsqueeze(1)
                    else:
                        if len(image.shape) == 2:
                            # grey scale
                            image = image.reshape(1, 1, *image.shape)
                        elif len(image.shape) == 3:
                            # color non-batch
                            image = image.unsqueeze(0)
                    if target_size is not None:
                        import torch.nn.functional as F

                        image = F.interpolate(
                            image,
                            size=_get_new_shape_maintain_ratio(
                                target_size, image.shape[-2:]
                            ),
                        )

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
