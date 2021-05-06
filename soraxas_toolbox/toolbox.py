class MagicDict(dict):
    """Content is accessable like property."""

    def __getattr__(self, attr):
        """called when self.attr doesn't exist."""
        return self[attr]

    def __setattr__(self, attr, val):
        """called setting attribute of this dict."""
        self[attr] = val


class PerformanceLogger:
    """Help you to write/log things to file."""

    @staticmethod
    def write_dict_file(dictionary, output_filename):
        """Given a dict file, wrtie the contents into a file."""
        with open(output_filename, "w") as outfile:
            for key, val in dictionary.items():
                outfile.write("{} : {}\n".format(key, val))


class MatrixPlotter:
    """An object that display and update 2D array plot."""

    def __init__(self, disable=False):
        import matplotlib.pyplot as plt

        self.disable = disable
        if self.disable:
            return
        self.fig = plt.figure()
        # ax = self.fig.add_subplot(111)
        self.ax = self.fig.add_subplot(111)
        self.colorbar = None

    def update(self, data, normalise=False, block=False):
        import matplotlib.pyplot as plt

        _data = data / data.sum() if normalise else data

        cax = self.ax.matshow(_data, interpolation="nearest")
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(cax)
        else:
            self.colorbar.update_bruteforce(cax)
        plt.draw()
        plt.pause(0.00001)
        if block:
            plt.show()


###########################################################################


def _get_time_factor_and_unit(elapsed, fix_width=True):
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
        factor_and_unit = _get_time_factor_and_unit(_mean, True)
        return (
            f"{_mean * factor_and_unit[0]:{_f()}}"
            f"±{_std * factor_and_unit[0]:{_f()}}{factor_and_unit[1]}"
        )

    except TypeError:
        # is a single value
        factor_and_unit = _get_time_factor_and_unit(elapsed, True)
        return f"{elapsed * factor_and_unit[0]:{_f()}}{factor_and_unit[1]}"


class Timer:
    def __init__(self):
        self.all_stamps = {}
        self._last_stamp_str = None
        self._last_stamp_time = None

    def stamp(self, stamped_str):
        import time

        if self._last_stamp_str is not None:
            key = (self._last_stamp_str, stamped_str)
            if key not in self.all_stamps:
                self.all_stamps[key] = []
            self.all_stamps[key].append(time.time() - self._last_stamp_time)
        self._last_stamp_str = stamped_str
        self._last_stamp_time = time.time()

    def print_stats(self, print_title=True):
        string_1_max_len = 0
        string_2_max_len = 0
        stats_size_max_len = 0
        total_time_spent = 0
        for k, v in self.all_stamps.items():
            # obtain various max length across all stamped results
            string_1_max_len = max(len(k[0]), string_1_max_len)
            string_2_max_len = max(len(k[1]), string_2_max_len)
            stats_size_max_len = max(len(str(len(v))), stats_size_max_len)
            total_time_spent += sum(v)
        _f = format_time2readable

        print(f"{'=' * 28} Stamped Results {'=' * 28}")
        if print_title:
            print(
                f"{'from':<{string_1_max_len}} -> {'to':{string_2_max_len}}: "
                f" mean ± stdev   "
                f"(   min  ~  max   ) [Σ^N = total | pct]"
            )
            print("-" * 73)
        for k, v in self.all_stamps.items():
            _sum = sum(v)
            print(
                f"{k[0]:<{string_1_max_len}} -> {k[1]:{string_2_max_len}}: {_f(v)} "
                f"({_f(min(v))}~{_f(max(v))}) "
                f"[Σ^{len(v):<{stats_size_max_len}}"
                f"={_f(_sum)}|{_sum / total_time_spent:>5.1%}]"
            )
        print("=" * 73)
