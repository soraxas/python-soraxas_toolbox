import typer

from pathlib import Path

app = typer.Typer()


@app.command()
def version():
    """Show the version of soraxas_toolbox."""
    print("soraxas_toolbox version 1.0.0")


@app.command()
def histogram(image_path: Path, bins: int = 150):
    """Plot the histogram of the given image file."""
    import matplotlib.pyplot as plt
    import soraxas_toolbox.image as st_image
    import numpy as np

    from PIL import Image

    img = np.array(Image.open(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        raise typer.Exit(code=1)
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
        raise typer.Exit(code=1)

    plt.title(f"{partial_title} ({Path(image_path).name})")

    pctile_99 = np.percentile(img, 99)
    pctile_1 = np.percentile(img, 1)

    plt.xlabel(
        f"(min: {img.min()}, max: {img.max()}) Percentile: [1% {pctile_1:.1f} | 99% {pctile_99:.1f}]"
    )
    plt.ylabel("Density")

    # ax = plt.gca()

    # # add a second y-axis for pixel count
    # ax2 = ax.twinx()
    # total_pixels = img.size
    # ax2.set_ylabel('Count')
    # # ax2.set_ylim(0, total_pixels)
    # yticks = ax.get_yticks()
    # print(yticks)
    # # yticks here is density, convert to count
    # yticks = (yticks * total_pixels).astype(int)
    # print(yticks)
    # ax2.set_yticks(yticks)
    # ax2.set_yticklabels([f'{y}' for y in yticks])
    # # plt.show()

    st_image.display(plt.gcf())


@app.command()
def display(image_path: Path):
    """Display the given image file."""
    import soraxas_toolbox.image as st_image
    from PIL import Image

    img = Image.open(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        raise typer.Exit(code=1)
    st_image.display(img)


def main():
    app()
