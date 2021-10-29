from .core import format_time2readable


class PerformanceLogger:
    """Help you to write/log things to file."""

    @staticmethod
    def write_dict_file(dictionary, output_filename):
        """Given a dict file, wrtie the contents into a file."""
        with open(output_filename, "w") as outfile:
            for key, val in dictionary.items():
                outfile.write("{} : {}\n".format(key, val))


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
