from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import lazy_import_plus
import typer

import soraxas_toolbox as st

if TYPE_CHECKING:
    import loguru
    import matplotlib.pyplot as plt
    import numpy as np
    import PIL.Image
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
    normalise: Annotated[
        bool, typer.Option(help="Apply normalization image to 0-1 range.")
    ] = False,
    clip_low: Annotated[
        float | None,
        typer.Option(help="The low percentile to clip the image (0-100 range)."),
    ] = None,
    clip_high: Annotated[
        float | None,
        typer.Option(help="The high percentile to clip the image (0-100 range)."),
    ] = None,
    display_backend: Annotated[
        st.image.DisplayBackendT,
        typer.Option(help="The backend to use for displaying the image."),
    ] = "auto",
    # cv2_op
    cv2_ops: Annotated[
        str | None,
        typer.Option(
            help=(
                "Arbitrary cv2 operations to apply to the image before displaying it."
                "e.g. `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`\n`cv2.cvtColor(img, cv2.COLOR_BayerBG2RGB)`"
            )
        ),
    ] = None,
):
    """
    Display the given image file.
    """
    img = PIL.Image.open(image_path)

    if cv2_ops is not None:
        import cv2

        img = np.asarray(img)
        img = eval(cv2_ops, {"cv2": cv2}, {"img": img})

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
