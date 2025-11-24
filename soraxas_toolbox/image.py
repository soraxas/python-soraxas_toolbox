from __future__ import annotations

import math
import os
import warnings
from abc import ABC
from shutil import which
from subprocess import PIPE, Popen
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import lazy_import_plus
import pip_ensure_version

import soraxas_toolbox as st

from . import easy_with_blocks, notebook, utils
from ._lazy_import_workaround import MatplotlibTorchImportWorkaround

# the future annotation allows type hinting for modules that does not exists
# pyright: reportPrivateImportUsage=false
# pyright: reportAssignmentType=false


DisplayBackendT = Literal["auto", "timg", "term_image"]

if TYPE_CHECKING:
    import io
    import numbers
    import re

    import matplotlib as mpl
    import matplotlib.image as mpl_image
    import matplotlib.pyplot as plt
    import numpy as np
    import PIL.Image
    import pydot.core
    import torch
    import torchvision
    import tqdm
else:
    PIL = lazy_import_plus.lazy_module("PIL.Image", level="base")
    np = lazy_import_plus.lazy_module("numpy")
    tqdm = lazy_import_plus.lazy_module("tqdm")
    pydot = lazy_import_plus.lazy_module("pydot")
    plt = lazy_import_plus.lazy_module("matplotlib.pyplot")
    mpl = lazy_import_plus.lazy_module("matplotlib")
    mpl_image = lazy_import_plus.lazy_module("matplotlib.image")
    torch = lazy_import_plus.lazy_module(
        "torch", on_import=MatplotlibTorchImportWorkaround()
    )
    torchvision = lazy_import_plus.lazy_module("torchvision")
    re = lazy_import_plus.lazy_module("re")
    numbers = lazy_import_plus.lazy_module("numbers")

    io = lazy_import_plus.lazy_module("io")

############################################################
##             Turn any matplotlib plt to img             ##
############################################################


def read_as_array(path: str) -> np.ndarray:
    return np.array(PIL.Image.open(path))


def plt_fig_to_nparray(fig: plt.Figure, normalize: bool = False) -> np.ndarray:
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
    Return:
        np.array with shape = (x,y,d) where d=3
    """

    # remove white padding
    fig.tight_layout()

    # Draw figure on canvas
    fig.canvas.draw()

    is_argb = False
    # Convert the figure to numpy array, read the pixel values and reshape the array
    if hasattr(fig.canvas, "tostring_rgb"):
        _img_str = fig.canvas.tostring_rgb()  # type: ignore[attr-defined]
        n_channel = 3
    elif hasattr(fig.canvas, "tostring_argb"):
        is_argb = True
        _img_str = fig.canvas.tostring_argb()  # type: ignore[attr-defined]
        n_channel = 4
    else:
        raise NotImplementedError(f"{fig.canvas}")
    img = np.fromstring(_img_str, dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (n_channel,))

    if is_argb:
        # need to reorder from argb to rgba. Use fancy indexing
        img = img[..., [1, 2, 3, 0]]

    if normalize:
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

        # the following messes the linking path if this python is running inside
        # a contained environment (e.g. pixi), while timg is in another contained
        # environment (e.g. in junest)
        clean_env = os.environ.copy()
        clean_env.pop("LD_LIBRARY_PATH", None)

        self.program = Popen(
            cmd,
            stdin=PIPE,
            stdout=PIPE if get_stdout else None,
            bufsize=-1,
            env=clean_env,
        )

    def __enter__(self):
        self.program.__enter__()
        return self

    def __exit__(self, exc_type, value, traceback):
        self.program.__exit__(exc_type, value, traceback)

    @property
    def stream(self) -> IO:
        assert self.program.stdin is not None
        return self.program.stdin


class DisplayableImage:
    """
    A wrapper that can hold either a PIL image or a stream save functor.
    """

    mode: Literal["pil", "stream"]
    __tmp_file_buffer: Optional["io.BytesIO"] = None

    def __init__(
        self,
        pil_image: PIL.Image.Image | None = None,
        stream_save_functor: Callable[[IO], None] | None = None,
        stream_format: str = "bmp",
    ):
        if pil_image is not None and stream_save_functor is None:
            self.mode = "pil"
        elif stream_save_functor is not None and pil_image is None:
            self.mode = "stream"
        else:
            raise ValueError(
                "Either pil_image or stream must be provided, but not both."
            )
        self.__pil_image: PIL.Image.Image | None = pil_image
        self.__stream_save_functor = stream_save_functor
        self.__stream_format = stream_format

    def into_stream_save_functor(self) -> Callable[[IO], None]:
        """
        Turn the image into stream if it is not already.
        """
        if self.mode == "pil":
            __pil_image: PIL.Image.Image = cast(PIL.Image.Image, self.__pil_image)
            return lambda stream: __pil_image.save(stream, format=self.__stream_format)
        elif self.mode == "stream":
            assert self.__stream_save_functor is not None
            return self.__stream_save_functor
        raise ValueError(f"Unknown mode {self.mode}")

    def into_pil(self):
        """
        Turn the image into PIL if it is not already.
        """
        if self.mode == "stream":
            # create this tmp buffer on SELF to keep the file in memory
            print(io.BytesIO)
            self.__tmp_file_buffer = io.BytesIO()
            self.into_stream_save_functor()(self.__tmp_file_buffer)
            return PIL.Image.open(self.__tmp_file_buffer)

        elif self.mode == "pil":
            return self.__pil_image
        raise ValueError(f"Unknown mode {self.mode}")


def __send_to_display(
    displayable_image: DisplayableImage,
    backend: DisplayBackendT,
    pbar: tqdm.tqdm | None = None,
):
    if notebook.is_notebook():
        import IPython.display  # type: ignore[import-untyped]

        if backend != "auto":
            raise ValueError(f"Cannot use backend '{backend}' in notebook")

        IPython.display.display(displayable_image.into_pil())
    else:
        if backend in ("auto", "timg"):
            if which("timg"):
                with TerminalImageViewer(get_stdout=pbar is not None) as viewer:
                    displayable_image.into_stream_save_functor()(viewer.stream)
                    if pbar is not None:
                        out, err = viewer.program.communicate()
                        pbar.write(out.decode())
            elif backend == "timg":
                raise ValueError(f"Cannot use backend '{backend}' as binary not found!")
        elif backend in ("auto", "term_image"):
            pip_ensure_version.require_package("term_image")
            from term_image.image import AutoImage

            AutoImage(displayable_image.into_pil()).draw()
        else:
            raise NotImplementedError(f"Unknown backend {backend}")


def get_new_shape_maintain_ratio(
    target_size: Union[Sequence[int], float, int],
    current_shape: Tuple[int, int],
    mode: Literal["max", "min"] = "max",
) -> tuple[int, int]:
    if isinstance(target_size, (tuple, list)):
        target_size = max(target_size) if mode == "max" else min(target_size)

    assert isinstance(target_size, (int, float))

    if current_shape[0] >= current_shape[1]:
        i1, i2 = 0, 1
    else:
        i1, i2 = 1, 0
    ratio = current_shape[i2] / current_shape[i1]
    new_shape = [0, 0]
    new_shape[i1] = int(target_size)
    new_shape[i2] = int(math.ceil(target_size * ratio))
    return cast(tuple[int, int], new_shape)


def ensure_is_numpy(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, PIL.Image.Image):
        return np.asarray(img)
    else:
        raise NotImplementedError(f"unknown type: {type(img)}")


def ensure_is_pillow(img) -> PIL.Image.Image:
    if isinstance(img, np.ndarray):
        return PIL.Image.fromarray(img)
    elif isinstance(img, PIL.Image.Image):
        return img
    else:
        raise NotImplementedError(f"unknown type: {type(img)}")


T = TypeVar("T", np.ndarray, "PIL.Image.Image")


def resize(
    image: T,  # pyright: ignore[reportRedeclaration]
    target_size: Union[Sequence[int], float, int],
    backend: Literal["pillow", "cv2"] = "pillow",
) -> T:
    image: Union[np.ndarray, PIL.Image.Image] = cast(
        Union[np.ndarray, PIL.Image.Image], image
    )

    _original_type: Literal["array", "pillow"] | None = None
    if isinstance(image, np.ndarray):
        _original_type = "array"
        image = cast(np.ndarray, image)
        new_shape = get_new_shape_maintain_ratio(target_size, image.shape[:2])
    elif isinstance(image, PIL.Image):
        _original_type = "pillow"
        image = cast(PIL.Image.Image, image)
        new_shape = get_new_shape_maintain_ratio(target_size, image.size)
    else:
        raise ValueError(f"Unsupported type: {_original_type}")

    if backend == "pillow":
        try:
            image = ensure_is_pillow(image)
        except TypeError as e:
            raise TypeError(
                "There seems to be some type error."
                "Maybe opencv (with backend='cv2') would support your image format?"
            ) from e
        image = image.resize(new_shape)  # type: ignore[assignment]
    elif backend == "cv2":
        import cv2

        image: np.ndarray = ensure_is_numpy(image)
        image = cv2.resize(image, (new_shape[1], new_shape[0]))

    if _original_type == "array":
        image = ensure_is_numpy(image)  # type: ignore[assignment]
    elif _original_type == "pillow":
        image = ensure_is_pillow(image)  # type: ignore[assignment]
    return cast(T, image)


def normalise(image: np.ndarray):
    _min = image.min()
    _max = image.max()
    return (image - _min) / (_max - _min)


normalize = normalise


# add a deprecated warning
def stats(image: "np.ndarray | torch.Tensor", **kwargs) -> "st.stats.AnalyseResult":
    warnings.warn(
        "soraxas_toolbox.image.stats is deprecated. Use soraxas_toolbox.stats.get(...) instead.",
        DeprecationWarning,
    )

    return st.stats.get(image, **kwargs)


def make_displayable_image(img: PIL.Image.Image) -> PIL.Image.Image:
    bit_size = re.findall(r"\d+", img.mode)
    bit_size = int(bit_size[0]) if bit_size else 8
    if bit_size not in [8, 16, 32]:
        raise ValueError(f"Unsupported file type, supported bit size is {bit_size}")
    if bit_size != 8:
        max_value = 2**bit_size - 1
        img_arr = (np.array(img) / max_value) * 255.0
        img = PIL.Image.fromarray(img_arr.astype(np.uint8))
    return img.convert("L")


def ensure_uint8_image(image: np.ndarray, as_uint16: bool = False):
    _min = image.min()
    _max = image.max()

    _is_float = image.dtype in (np.float32, np.float64)
    if image.dtype == np.uint8:
        # no op
        pass
    elif as_uint16 and _is_float:
        if _min < 0 or _max > 2**16 - 1:
            warnings.warn(
                f"Given image is of float-type, but its value is not between [0, 2^16 - 1]. (min: {_min}, max: {_max}). ",
                RuntimeWarning,
            )
        # first cast it back to uint16
        image = image.astype(np.uint16)
    elif _is_float:
        if _min < 0 or _max > 1:
            warnings.warn(
                f"Given image is of float-type, but its value is not between [0, 1]. (min: {_min}, max: {_max}). "
                "You might want to normalise it first, or treat it as uint16 image.",
                RuntimeWarning,
            )
        image = (image * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    else:
        raise NotImplementedError(f"Given image is of unknown dtype {image.dtype}")
    return image


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
    def fix_float_range(cls, image, normalise: bool | None = None):
        # works for either numpy or torch
        _min = image.min()
        _max = image.max()
        _warning_msg = None
        if normalise:
            image = normalize(image)
        elif _min < 0 or _max > 1:
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


def __handle_torch_image(image, normalise, is_grayscale, is_batched, target_size):
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
    image = TorchArrayAutoFixer.fix_float_range(image, normalise=normalise)

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
            size=get_new_shape_maintain_ratio(target_size, image.shape[-2:]),
        )
    return image


def cumulative_sum_starts_at(my_list):
    """
    Given [7, 5, 2, 9]
    returns [0, 7, 12, 14]
    """
    result = [0 for _ in range(len(my_list))]
    for i in range(len(my_list) - 1):
        result[i + 1] = result[i] + my_list[i]
    return result


def concat_images(
    images: List[PIL.Image.Image], max_cols: int | None = 0, boarder: int = 5
) -> PIL.Image.Image:
    # no need to concat
    if len(images) <= 1:
        return images[0]

    # compute grid size
    if max_cols:
        ncols = min(max_cols, len(images))
        nrows = int(math.ceil(len(images) / ncols))
    else:
        ncols = len(images)
        nrows = 1

    # partition into per rows and per cols max values
    max_height_per_row = [[] for _ in range(nrows)]
    max_width_per_col = [[] for _ in range(ncols)]
    for idx, _im in enumerate(images):
        col = idx % ncols
        row = idx // ncols
        max_height_per_row[row].append(_im.height + boarder)
        max_width_per_col[col].append(_im.width + boarder)
    max_height_per_row = [max(it) for it in max_height_per_row]
    max_width_per_col = [max(it) for it in max_width_per_col]
    ########################################
    # remove the board for the last row & col
    max_height_per_row[-1] -= boarder
    max_width_per_col[-1] -= boarder

    # Create canvas for the final image with total size
    image_size = (sum(max_width_per_col), sum(max_height_per_row))
    image = PIL.Image.new("RGB", image_size)

    # compute where the starting location for each images
    _width_starts_at = cumulative_sum_starts_at(max_width_per_col)
    _height_starts_at = cumulative_sum_starts_at(max_height_per_row)
    # Paste images into final image
    for idx, _im in enumerate(images):
        col = idx % ncols
        row = idx // ncols
        offset = _width_starts_at[col], _height_starts_at[row]
        image.paste(_im, offset)

    return image


def __to_pil_image(
    image: Union["np.ndarray", torch.Tensor, "PIL.Image.Image"],
    normalise: Optional[bool] = None,
    target_size: Tuple[int, int] | int | None = None,
    is_batched: Optional[bool] = None,
    is_grayscale: Optional[bool] = None,
) -> "PIL.Image.Image":
    if utils.module_was_imported("torch") and not utils.module_was_imported(
        "torchvision"
    ):
        # fallback as numpy
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

    if utils.module_was_imported("numpy") and isinstance(image, np.ndarray):
        NumpyArrayAutoFixer.cls_var_setter(module=np)

        image: np.ndarray = NumpyArrayAutoFixer.fix_channel(image)

        if target_size is not None:
            # we are doing resize first as it might reduce work needed for normalise
            # COSTLY bi-directional roundtrip
            image = cast(np.ndarray, resize(image, target_size=target_size))

        if image.dtype not in NumpyArrayAutoFixer.float_types() and normalise:
            # need to be in float to do normalise
            image = image.astype(np.float32)

        if image.dtype in NumpyArrayAutoFixer.float_types():
            image = NumpyArrayAutoFixer.fix_float_range(image, normalise=normalise)

        # to uint8 if necessary
        image = NumpyArrayAutoFixer.fix_dtype(image)
        return PIL.Image.fromarray(ensure_uint8_image(image))

    # image would either be a torch tensor or a PIL image
    image = cast(Union["torch.Tensor", PIL.Image.Image], image)

    if utils.module_was_imported("torchvision") and isinstance(image, torch.Tensor):
        TorchArrayAutoFixer.cls_var_setter(module=torch)
        with torch.no_grad():
            with easy_with_blocks.NoMissingModuleError(strong_warning=True):
                # the following should be a list of 3D array
                image = (
                    torchvision.utils.make_grid(
                        __handle_torch_image(
                            image=image,
                            normalise=normalise,
                            is_grayscale=is_grayscale,
                            target_size=target_size,
                            is_batched=is_batched,
                        )
                    )
                    .mul(255)
                    .clamp_(0, 255)
                    .permute(1, 2, 0)
                    .to("cpu", torch.uint8)
                    .numpy()
                )
                # .add_(0.5)
                return PIL.Image.fromarray(image)

    image = cast("PIL.Image.Image", image)

    if target_size is not None:
        image = resize(image, target_size=target_size)
    return image


SupportedImageType = Union[
    np.ndarray,
    "PIL.Image.Image",
    "torch.Tensor",
    "plt.Figure",
    "pydot.Dot",
]


def _display_preflight_check(
    image: SupportedImageType,
    normalise: bool | None = None,
):
    # if we are normalising, we need to convert the image to float32
    if normalise:
        if utils.module_was_imported("torch"):
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)
        return image

    if isinstance(image, PIL.Image.Image):
        return image
    if utils.module_was_imported("matplotlib") and isinstance(image, plt.Figure):
        # no need to do anything else, but will raise an error if there are more images
        return image
    if utils.module_was_imported("torch") and isinstance(image, torch.Tensor):
        return image
    if utils.module_was_imported("numpy") and isinstance(image, np.ndarray):
        return image
    if utils.module_was_imported("pydot") and isinstance(image, pydot.Dot):
        return dot_to_image(image)
    raise ValueError(f"Unsupported type {type(image)}")


def display(
    image: SupportedImageType,
    *more_images: SupportedImageType,
    max_cols: int | None = None,
    target_size: Tuple[int, int] | None = None,
    pbar: tqdm.tqdm | None = None,
    format: str = "bmp",
    #
    normalise: Optional[bool] = None,
    is_batched: Optional[bool] = None,
    is_grayscale: Optional[bool] = None,
    backend: DisplayBackendT = "auto",
) -> None:
    images = list(
        map(
            lambda x: _display_preflight_check(x, normalise=normalise),
            (image, *more_images),
        )
    )
    if utils.module_was_imported("matplotlib"):
        if len(images) > 1 and any(isinstance(x, plt.Figure) for x in images):
            raise NotImplementedError("matplotlib does not support multi image")

    # early return for matplotlib figure
    if isinstance(images[0], plt.Figure):
        if len(more_images) > 0:
            raise NotImplementedError("matplotlib does not support multi image")

        return __send_to_display(
            displayable_image=DisplayableImage(
                stream_save_functor=lambda stream: images[0].savefig(stream),
                stream_format=format,
            ),
            pbar=pbar,
            backend=backend,
        )

    #####################################################
    # for list of nparray or tensor

    all_pil_images = [
        __to_pil_image(
            im,
            normalise=normalise,
            target_size=target_size,
            is_batched=is_batched,
            is_grayscale=is_grayscale,
        )
        for im in images
    ]
    output_image = concat_images(
        all_pil_images,
        max_cols=max_cols,
    )

    # if image is in 16bit, convert it to 8bit
    if output_image.mode == "I;16":
        output_image = make_displayable_image(output_image)

    return __send_to_display(
        displayable_image=DisplayableImage(pil_image=output_image),
        pbar=pbar,
        backend=backend,
    )


def view_high_dimensional_embeddings(
    x: np.ndarray, label=None, title="High-d embeddings"
):
    import pandas as pd
    import seaborn as sns
    from sklearn.manifold import TSNE

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


def dot_to_image(dot_graph: "pydot.core.Dot") -> "np.ndarray":
    # render the `pydot` by calling `dot`, no file saved to disk
    png_str = dot_graph.create(prog="dot", format="png")
    # treat the DOT output as an image file
    sio = io.BytesIO()
    sio.write(png_str)
    sio.seek(0)
    img = mpl_image.imread(sio)
    return img
