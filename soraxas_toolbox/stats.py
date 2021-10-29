import contextlib
import csv
import os
from pathlib import Path
from typing import List, Dict

from . import ContextManager


class StatsSaver(ContextManager):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    @contextlib.contextmanager
    def contextmanager(self):
        # create any non-existing folder
        Path(os.path.dirname(self.filename)).mkdir(parents=True, exist_ok=True)

        with open(self.filename, "w") as file:
            yield self.StatsSaverInternal(file)

    class StatsSaverInternal:
        def __init__(self, opened_file):
            super().__init__()
            self.csv_writer = csv.writer(opened_file)
            self._header = None

        def writeheader(self, headers: List[str]):
            if self._header is not None:
                raise RuntimeError(f"Header had already been set to: {self._header}")
            self._header = headers
            self.csv_writer.writerow(self._header)

        def writerow(self, row: List[str]):
            assert self._header is not None
            assert len(self._header) == len(row), (
                f"Inconsistent length detected. Was "
                f"{len(self._header)}, "
                f"now {len(row)}."
            )
            self.csv_writer.writerow(row)

        def writerow_dict(self, row: Dict[str, str]):
            assert self._header is not None
            assert self._header == len(row), "inconsistent length detected. "
            self.csv_writer.writerow([row[h] for h in self._header])
