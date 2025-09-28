import typer

from pathlib import Path
from typing import TYPE_CHECKING

import lazy_import_plus
import soraxas_toolbox as st

if TYPE_CHECKING:
    import PIL.Image
    import numpy as np
    import matplotlib.pyplot as plt
    import loguru
else:
    PIL = lazy_import_plus.lazy_module("PIL.Image", level="base")
    np = lazy_import_plus.lazy_module("numpy")
    plt = lazy_import_plus.lazy_module("matplotlib.pyplot")
    loguru = lazy_import_plus.lazy_module("loguru")

app = typer.Typer()


@app.command()
def version():
    """Show the version of soraxas_toolbox."""
    print("soraxas_toolbox version 1.0.0")


@app.command()
def stats(image_path: Path):
    img = st.image.read_as_array(image_path)
    print(st.image.stats(img))


@app.command()
def histogram(
    image_path: Path,
    bins: int = 150,
    ylog: bool = False,
    display_backend: st.image.DisplayBackendT = "auto",
):
    """Plot the histogram of the given image file."""
    img = st.image.read_as_array(image_path)

    try:
        fig = st.plotting.histogram(
            img=img, bins=bins, ylog=ylog, partile_title=image_path.name
        )
    except ValueError:
        typer.Exit(code=1)

    st.image.display(fig, backend=display_backend)


@app.command()
def display(
    image_path: Path,
    normalise: bool = False,
    clip_low: float | None = None,
    clip_high: float | None = None,
    display_backend: st.image.DisplayBackendT = "auto",
):
    """Display the given image file."""
    img = PIL.Image.open(image_path)

    if clip_low is not None or clip_high is not None:
        # need to convert to numpy array first
        img = np.array(img)

        if img.dtype in (np.uint8, np.uint16, np.uint32):
            divisor = np.iinfo(img.dtype).max
        else:
            divisor = 1.0

        _old_range = (img.min(), img.max())

        pctile_low, pctile_high = None, None

        if clip_low is not None:
            pctile_low = np.percentile(img, clip_low)
        if clip_high is not None:
            pctile_high = np.percentile(img, clip_high)

        new_range = (img.min(), img.max())

        loguru.logger.info(
            f"Clipping image from original range {_old_range} to [{new_range}]"
        )

        img = np.clip(img, a_min=pctile_low, a_max=pctile_high)
        img = img / divisor

        loguru.logger.info(
            f"Image data type after clipping and normalization: {img.dtype}, range: ({img.min()}, {img.max()})"
        )

    st.image.display(img, normalise=normalise, backend=display_backend)


def main():
    app()
