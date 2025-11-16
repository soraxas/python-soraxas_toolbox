import contextlib
import csv
import numbers
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import lazy_import_plus

from . import utils

if TYPE_CHECKING:
    import numpy as np
    import torch
else:
    np = lazy_import_plus.lazy_module("numpy")
    torch = lazy_import_plus.lazy_module("torch")


@dataclass
class AnalyseResult:
    stats: dict
    kwargs: dict

    @classmethod
    def from_array_like(cls, image, **kwargs) -> "AnalyseResult":
        if utils.module_was_imported("torch"):
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()

        return AnalyseResult(
            stats=dict(
                mean=np.mean(image),
                shape=image.shape,
                dtype=image.dtype,
                min=image.min(),
                p01=np.percentile(image, 1),
                p25=np.percentile(image, 25),
                median=np.median(image),
                p75=np.percentile(image, 75),
                p99=np.percentile(image, 99),
                max=image.max(),
                __raw_image=image,
            ),
            kwargs=kwargs,
        )

    def get_inline_histogram(self, bins=16) -> str:
        data = self.stats.get("__raw_image", None)
        if data is None:
            return ""

        if "display_min" in self.kwargs or "display_max" in self.kwargs:
            _range = (
                self.kwargs.get("display_min", self.stats["min"]),
                self.kwargs.get("display_max", self.stats["max"]),
            )
        else:
            _range = None
        hist, _ = np.histogram(data, bins=bins, range=_range)
        spark_chars = " ▁▂▃▄▅▆▇█"

        _max_value = len(spark_chars) - 1
        # Normalize histogram to [0, 7]
        # hist_scaled = np.clip(
        #     (hist / hist.max()) * _max_value, 0, _max_value
        # ).astype(int)
        hist_scaled = np.ceil((hist / hist.max()) * _max_value).astype(int)
        sparkline = "".join(spark_chars[val] for val in hist_scaled)

        return f"{sparkline}"

    def __str__(self) -> str:
        n_span_space = 2

        def _pos(i, total):
            if i == 0:
                return "<"
            elif i == total - 1:
                return ">"
            return "^"

        def fmt(value, width: int = 8, precision: int = 3, pos: str = "<"):
            if isinstance(value, numbers.Number):
                value_as_int = int(value)
                if abs(value_as_int - value) > 1e-9:
                    return f"{float(value):{pos}{width}.{precision}f}"
                else:
                    value = value_as_int
            return f"{value:{pos}{width}}"

        def format_line(
            keys,
            values,
            # 1 sign, 1 decimal, 3 precision, 1-3 leading?
            width=8,
            precision=3,
        ):
            value_strs = [
                fmt(v, width, precision, pos=_pos(i, len(values)))
                for i, v in enumerate(values)
            ]

            def _format(k, i):
                if i == 0:
                    # add indicator
                    k = f"^{k}"
                elif i == len(keys) - 1:
                    k = f"{k}^"
                    # direction = ">"
                return f"{k:{_pos(i, len(keys))}{width}}"

            key_strs = [_format(k, i) for i, k in enumerate(keys)]
            return (" " * n_span_space).join(value_strs), "  ".join(key_strs)

        # Layout groups
        # inject dummy to make it align with line2
        line1_keys = ["min", "", "mean", "", "max"]
        line1_values = [self.stats.get(k, "") for k in line1_keys]

        line2_keys = ["p01", "p25", "median", "p75", "p99"]
        line2_values = [self.stats[k] for k in line2_keys]

        width = 8

        # Format both lines
        values1, keys1 = format_line(line1_keys, line1_values, width=width)
        values2, keys2 = format_line(line2_keys, line2_values, width=width)

        # Shape and dtype
        shape_str = f"shape: {self.stats['shape']}"
        dtype_str = f"dtype: {self.stats['dtype']}"

        inline_hist = self.get_inline_histogram(
            bins=width * len(line2_keys) + n_span_space * (len(line2_keys) - 1)
        )

        return f"""{shape_str}
{dtype_str}\

 .{inline_hist}.
  {values1}
  {keys1}
  {values2}
  {keys2}\

"""


def get(image: "np.ndarray | torch.Tensor", **kwargs) -> "AnalyseResult":
    return AnalyseResult.from_array_like(image, **kwargs)


class StatsSaver(utils.ContextManager):
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
