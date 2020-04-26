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
        with open(output_filename, 'w') as outfile:
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

        cax = self.ax.matshow(_data, interpolation='nearest')
        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(cax)
        else:
            self.colorbar.update_bruteforce(cax)
        plt.draw()
        plt.pause(0.00001)
        if block:
            plt.show()

