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
