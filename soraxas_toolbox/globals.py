from types import SimpleNamespace
from typing import Callable

ns = dict()


def create_if_not_exists(name: str, initialiser: Callable):
    if name not in ns:
        ns[name] = initialiser()
