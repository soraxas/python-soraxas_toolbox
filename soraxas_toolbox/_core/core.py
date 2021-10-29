import abc
import contextlib
import datetime
import os


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


def get_non_existing_filename(
    *filename_parts: str, csv_template: str = "%Y-%m-%d_%H-%M{}.csv"
):
    """
    Return a string that represent a path to file that does not exists.
    The filename is the current timestamp.
    If such a filename exists, it will append a suffix after the given name.

    Default template: %Y-%m-%d_%H-%M{}.csv
    E.g.            : 2021-10-29_01-05.csv
    E.g.            : 2021-10-29_01-05.1.csv
    E.g.            : 2021-10-29_01-05.2.csv
    """
    suffix_num = 0
    while True:
        # join filename parts if they exist
        _path_parts = list(filename_parts) + [csv_template]
        target_filename_template = os.path.join(*_path_parts)
        # apply datetime format
        fname = datetime.datetime.now().strftime(
            target_filename_template.format("" if suffix_num == 0 else f".{suffix_num}")
        )
        if not os.path.exists(fname):
            break
        suffix_num += 1
    return fname


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
