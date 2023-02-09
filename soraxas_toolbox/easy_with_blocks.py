import contextlib
import warnings

from . import ContextManager


class NoMissingModuleError(ContextManager):
    def __init__(self, strong_warning: bool = False):
        super().__init__()
        self.strong_warning = strong_warning

    @contextlib.contextmanager
    def contextmanager(self):
        try:
            yield
        except ModuleNotFoundError as e:
            warning_type = ImportWarning
            if self.strong_warning:
                warning_type = UserWarning

            warnings.warn(str(e), warning_type)
