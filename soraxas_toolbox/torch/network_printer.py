try:
    import datetime
    import glob
    import inspect
    import os
    import shutil

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from tensorboardX import SummaryWriter
except Exception as e:
    print("Error occured when importing dependencies:")
    print(e)


############################################################
##                  TorchNetowrkPrinter                  ##
############################################################


class TorchNetworkPrinter(object):
    """This class patches nn.Module so it would print the netowrk
    architecture automatically as it performs forward pass."""

    def __init__(
        self,
        initialise=True,
        auto_cleanup=True,
        use_buffer=False,
        width_name=25,
        width_insize=20,
        width_outsize=20,
        ignore_modules=[],
        ignore_modules_within=[],
    ):
        """You can delay (and manually call patch) by passing initialise as False.
        Give the model as the parameters to denote which model to print. When the
        forward pass of that model is encountered again, it will cleans up and stop printing
        any further details.
        IF auto_cleanup is True:
            When the forward pass of the first encountered module is called again (second time),
            it will begins to clean up and stop any further output.
        ELSE:
            you should call torch_net_printer.cleanup() at the end of your loop.

        You can ignore printing certain module by putting a list of names (string) in the args ignore_modules
        """
        try:
            torch.nn.Module.__patched
            # if exists.
            print("WARN: Another TorchNetworkPrinter already exists!")
            print("      Trying to clean up previous instance.")
            print("      Use with caution.")
            print()
            torch.nn.Module.__patched.cleanup()
        except AttributeError:
            pass
        self._ori_nn_module_init = None
        self.sentence_ended = True
        self.auto_cleanup = auto_cleanup
        self.last_module_printed = None
        self.banner_printed = False
        self.stack_depth_map = []
        self.hook_handlers = []
        self.use_buffer = use_buffer
        self.buffer = []

        self.ignore_modules = set(ignore_modules)
        self.ignore_modules_within = set(ignore_modules_within)
        # we use this to detect finishing printing the entire network
        self.first_module = None

        self.width_name = width_name
        self.width_insize = width_insize
        self.width_outsize = width_outsize

        self.off = False

        if initialise:
            self.patch()

    def print_network(self):
        for text, kwargs in self.buffer:
            print(text, **kwargs)
        self.buffer = []

    def cleanup(self):
        for h in self.hook_handlers:
            # clean ups after itself :) remove hooks.
            h.remove()
        self.hook_handlers = []
        self.unpatch()

    def patch(self):
        """Patch the module __init__."""
        if self._ori_nn_module_init is None:
            self._ori_nn_module_init = torch.nn.Module.__init__
        torch.nn.Module.__init__ = self.create_wrapper(torch.nn.Module.__init__)
        torch.nn.Module.__patched = self

    def unpatch(self):
        """Restore the original module __init__."""
        if self._ori_nn_module_init is None:
            print("WARN: No stored nn.Module.__init__")
            print("      The module is never patched?")
        else:
            torch.nn.Module.__init__ = self._ori_nn_module_init
            del torch.nn.Module.__patched
            self._ori_nn_module_init = None

    def print_stack_depth(self, _depth, offset=0, symbol="|"):
        """Return a value that represent how long did we printed the stack depth."""
        if _depth not in self.stack_depth_map:
            # use a list to simplify the stack depth
            self.stack_depth_map.append(_depth)
            self.stack_depth_map = sorted(self.stack_depth_map)
        depth = self.stack_depth_map.index(_depth) + offset
        if depth > 0:
            # if depth is more than one, print any bar before the given symbol
            _prefix = "{}{} ".format("| " * (depth - 1), symbol)
            self.p(_prefix, require_newline=True)
            return len(_prefix)
        else:
            return 0

    def print_net_name_insize(self, module, inx, depth):
        # first time printing
        if not self.banner_printed:
            print_seq_line = lambda: self.p(
                "-" * (self.width_name + self.width_insize + self.width_outsize),
                end=True,
            )
            print_seq_line()
            self._print_name("Network Name")
            self._print_insize("In Size")
            self._print_outsize("Out Size")
            print_seq_line()
            self.banner_printed = True
        # check if this main model had been printed before
        if self.auto_cleanup:
            if self.first_module is None:
                self.first_module = module
            else:
                if hash(self.first_module) == hash(module):
                    self.cleanup()
                    return
        #     # we had printed the
        self._check_same_guard(module)
        printed_stack_depth = self.print_stack_depth(depth)
        self._print_name(module.__class__.__name__, width_offset=-printed_stack_depth)
        self._print_insize(self.get_tensor_size(inx))

    def print_net_outsize(self, module, outx, depth):
        if not self._check_same_guard(module):
            # this is a line with only outsize.
            # print the correct spacing. The 1 offset is to force the arrow line up with previous innser-netowrks
            printed_stack_depth = self.print_stack_depth(depth, offset=1, symbol="↳")
            self._print_name("", width_offset=-printed_stack_depth)
            self._print_insize("")
        self._print_outsize(self.get_tensor_size(outx))
        self.last_module_printed = hash(module)

    def create_wrapper(self, func):
        """The wrapper around the nn.Module.__init__"""

        def prehook(module, inx):
            """Hook that is called before the forward pass."""
            if module.__class__.__name__ in self.ignore_modules:
                self.off = True

            if not self.off:
                self.print_net_name_insize(module, inx, depth=len(inspect.stack()))

            if module.__class__.__name__ in self.ignore_modules_within:
                self.off = True

        def posthook(module, inx, outx):
            """Hook that is called after the forward pass."""
            if module.__class__.__name__ in self.ignore_modules_within:
                self.off = False

            if not self.off:
                self.print_net_outsize(module, outx, depth=len(inspect.stack()))

            if module.__class__.__name__ in self.ignore_modules:
                self.off = False

        # def prehook(module, inx):
        #     """Hook that is called before the forward pass."""
        #     if module.__class__.__name__ not in self.ignore_modules:
        #         self.print_net_name_insize(
        #             module, inx, depth=len(inspect.stack()))
        #
        # def posthook(module, inx, outx):
        #     """Hook that is called after the forward pass."""
        #     if module.__class__.__name__ not in self.ignore_modules:
        #         self.print_net_outsize(
        #             module, outx, depth=len(inspect.stack()))

        def call(*args, **kwargs):
            # register hook on ourselves
            # first arg should be self
            result = func(*args, **kwargs)
            # set up some variable to clean up later
            module = args[0]
            self.hook_handlers.append(module.register_forward_pre_hook(prehook))
            self.hook_handlers.append(module.register_forward_hook(posthook))
            return result

        return call

    ############################################################
    ##                        HELPERS                         ##
    ############################################################

    def p(self, text, end=False, require_newline=False):
        """Wrapper around print that take notes on using buffer or not."""
        # end the sentence for it if not already
        if require_newline and not self.sentence_ended:
            text = "\n" + text
        kwargs = {}
        if not end:
            kwargs["end"] = ""
            self.sentence_ended = False
        else:
            self.sentence_ended = True

        if self.use_buffer:
            self.buffer.append((text, kwargs))
        else:
            print(text, **kwargs)

    def _print_name(self, name, width_offset=0):
        width = self.width_name + width_offset
        width = max(0, width)
        self.p("{0:<{name_width}} ".format(name, name_width=width))

    def _print_insize(self, insize, width_offset=0):
        width = self.width_insize + width_offset
        self.p("{0:<{insize_width}} ".format(insize, insize_width=width))

    def _print_outsize(self, outsize, width_offset=0):
        width = self.width_outsize + width_offset
        self.p("{0:<{outsize_width}} ".format(outsize, outsize_width=width), end=True)

    def _check_same_guard(self, module):
        """Check if last printed module is the same as this given module."""
        if self.last_module_printed is None:
            result = True
        else:
            result = self.last_module_printed == hash(module)
        self.last_module_printed = hash(module)
        return result

    def get_tensor_size(self, inx):
        if type(inx) in (tuple, list):
            # recursively call itself
            _shape = ""
            for i in inx:
                _shape = "{},{}".format(_shape, self.get_tensor_size(i))
            # remove first extra comma
            shape = _shape[1:]
        elif type(inx) in (torch.Tensor, torch.nn.parameter.Parameter):
            if not inx.shape:
                shape = "RealVal"
            else:
                shape = "x".join(str(x) for x in inx.shape)
        elif type(inx) == bool:
            shape = "bool"
        elif inx is None:
            shape = "None"
        elif isinstance(inx, dict):
            shape = "dict("
            shape += ",".join(f"{k}:{self.get_tensor_size(v)}" for k, v in inx.items())
            shape += ")"
        else:
            shape = f"{type(inx)}[{inx}]"
            # raise Exception("Unimplemented type {}".format(type(inx)))
        return "{}".format(shape)
