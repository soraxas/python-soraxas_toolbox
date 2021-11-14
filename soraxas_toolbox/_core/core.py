import abc
import contextlib
import datetime
import os
from pathlib import Path
from typing import Callable, Union


class MagicDict(dict):
    """Content is accessible like property."""

    def __getattr__(self, attr):
        """called when self.attr doesn't exist."""
        return self[attr]

    def __setattr__(self, attr, val):
        """called setting attribute of this dict."""
        self[attr] = val


###########################################################################


def convert_time_to_factor_and_unit(elapsed: float, fix_width=True):
    if elapsed < 1e-6:
        factor = 1e9
        unit = "ns"
    elif elapsed < 1e-3:
        factor = 1e6
        unit = "µs"
    elif elapsed < 1:
        factor = 1e3
        unit = "ms"
    else:
        factor = 1
        unit = "s"
        if fix_width:
            unit = " " + unit
    return factor, unit


def format_time2readable(elapsed, precision=3, decimal_place=2, width=6):
    import numpy as np

    def _f():
        return f">{width}.{decimal_place}f"

    try:
        iter(elapsed)
        # is list-type
        _mean = np.mean(elapsed)
        _std = np.std(elapsed)
        factor_and_unit = convert_time_to_factor_and_unit(_mean, True)
        return (
            f"{_mean * factor_and_unit[0]:{_f()}}"
            f"±{_std * factor_and_unit[0]:{_f()}}{factor_and_unit[1]}"
        )

    except TypeError:
        # is a single value
        factor_and_unit = convert_time_to_factor_and_unit(elapsed, True)
        return f"{elapsed * factor_and_unit[0]:{_f()}}{factor_and_unit[1]}"


def get_current_timestamp(template="%Y-%m-%d_%H-%M") -> str:
    """Return a string that represent the current timestamp."""
    return datetime.datetime.now().strftime(template.format(template))


def get_non_existing_filename(
    *parent_folders: str,
    filename_prefix: Union[str, Callable] = get_current_timestamp,
    filename_suffix: Union[str, Callable] = "",
    create_folders: bool = False,
):
    """
    Return a string that represent a path to file that does not exists.
    The filename is the current timestamp.
    If such a filename exists, it will append a suffix after the given name.

    Default template: %Y-%m-%d_%H-%M{}.csv
    E.g.            : 2021-10-29_01-05.0.csv
    E.g.            : 2021-10-29_01-05.1.csv
    E.g.            : 2021-10-29_01-05.2.csv
    """
    if create_folders and len(parent_folders) > 0:
        Path(os.path.join(*parent_folders)).mkdir(parents=True, exist_ok=True)
    suffix_num = 0
    while True:
        # join filename parts if they exist
        _prefix = filename_prefix() if callable(filename_prefix) else filename_prefix
        _suffix = filename_suffix() if callable(filename_suffix) else filename_suffix
        # _path_parts = list(parent_folders) + [f"{_prefix}.{suffix_num}{_suffix}"]
        # construct format
        filename = os.path.join(*parent_folders, f"{_prefix}.{suffix_num}{_suffix}")
        if not os.path.exists(filename):
            break
        suffix_num += 1
    return filename


class ContextManager(metaclass=abc.ABCMeta):
    """
    Class which can be used as `contextmanager`.
    Following patterns from: https://stackoverflow.com/questions/8720179/nesting-python-context-managers
    """

    def __init__(self):
        self.__cm = None

    @abc.abstractmethod
    @contextlib.contextmanager
    def contextmanager(self):
        raise NotImplementedError("Abstract method")

    def __enter__(self):
        self.__cm = self.contextmanager()
        return self.__cm.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.__cm.__exit__(exc_type, exc_value, traceback)
