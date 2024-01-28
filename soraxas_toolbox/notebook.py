from typing import Union, Iterable

from . import easy_with_blocks

with easy_with_blocks.NoMissingModuleError(strong_warning=False):
    from IPython.display import Markdown, display


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def display_color(color: Union[str, Iterable[Union[float, int]]]):
    """
    Given an object that is either a hex-string, or a list of RGB
    values, we will display it in the notebook.

    We will interpret the given values as 0-255 if they are integers.
    Otherwise, if any of them is float, and all of them are <= 1, we
    will infer them as RGB in the scale of 0-1, and help you to scale
    them to the full range of 255.
    """
    if not isinstance(color, str):
        # treat it as iterable
        color_it = tuple(color)
        if any(
            # some of them is float
            map(lambda c: isinstance(c, float), color_it)
        ) and all(
            # all of them are less than 1
            map(lambda c: 0 <= c <= 1, color_it)
        ):
            # scale them to range of 0-255
            color_it = list(map(lambda x: 255 * x, color_it))

        color = f"rgb({','.join(map(str, color_it))})"

    display(
        Markdown(
            f'<span style="font-family: monospace">{color} '
            f'<span style="color: {color}">████████</span></span>'
        )
    )
