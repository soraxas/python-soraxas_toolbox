from __future__ import annotations

import abc
import contextlib
import sys

import lazy_import_plus


def module_was_imported(module_name: str):
    """
    Determine if a module is already loaded (for efficient isinstance checking)

    We determine if any attributes had been actually accessed by the system
    """

    mod = sys.modules.get(module_name, None)
    if mod is None:
        return False
    if isinstance(mod, lazy_import_plus.LazyModule):
        # determine if this lazy module has actually been used
        # if it still hasn't it will have the following attribute
        # in its class
        modclass = type(mod)
        if hasattr(modclass, "_lazy_import_plus_error_msgs"):
            return False
    return True


class MagicDict(dict):
    """Content is accessible like property."""

    def __getattr__(self, attr):
        """called when self.attr doesn't exist."""
        return self[attr]

    def __setattr__(self, attr, val):
        """called setting attribute of this dict."""
        self[attr] = val


###########################################################################


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
