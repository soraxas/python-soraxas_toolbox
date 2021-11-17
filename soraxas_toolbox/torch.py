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


class Filename:
    def __init__(self, filename):
        self._dir = os.path.dirname(filename)
        _filename = os.path.splitext(os.path.basename(filename))
        self._filename = _filename[0]
        self._ext = _filename[1]

    @property
    def filename(self):
        return self._filename

    @property
    def dir(self):
        return self._dir

    @property
    def ext(self):
        return self._ext

    @property
    def full_filename(self):
        return os.path.join(self.dir, "{}{}".format(self.filename, self.ext))


class TorchCheckpointSaver:
    def __init__(self, filename, save_as=None, read_only=False, verbose=1):
        """save_as: you can specific a different name to save to."""
        self.read_only = read_only
        self.verbose = verbose
        # name for loading
        self.load_fn = Filename(filename)
        # name for saving
        if save_as is None:
            self.save_fn = self.load_fn
        else:
            self.save_fn = Filename(save_as)

    def _print(self, *args, level=1, **kwargs):
        """Level 0 is debug, 1 is useful info, 2 is error."""
        if level >= self.verbose:
            print(*args, **kwargs)

    def save(
        self, state, best=False, custom_extra_txt="", timestamp=False, keep_last=5
    ):
        """If timestamp is True, it will only keep the latest `keep_last` files."""
        """
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        """
        if self.read_only:
            self._print(
                "ERROR: 'TorchCheckpointSaver' refusing to save because you made a promise of this is read only!",
                level=2,
            )
            return
        f = self.save_fn
        if timestamp:
            _stamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
            filename = "{}-{}".format(f.filename, _stamp)
            # remove extras
            all_checkpoints = self._get_existing_ckpt()
            if len(all_checkpoints) >= keep_last:
                for _old_ckpt in all_checkpoints[: keep_last - 1]:
                    os.remove(_old_ckpt)
            dest = os.path.join(f.dir, "{}{}".format(filename, f.ext))
        else:
            filename = f"{f.filename}{custom_extra_txt}"
            dest = os.path.join(f.dir, "{}{}".format(filename, f.ext))
            if os.path.exists(dest):
                history_folder = f"{dest}.history"
                os.makedirs(history_folder, exist_ok=True)
                historical_weights = sorted(glob.glob(f"{history_folder}/*.tar"))
                # remove old files
                while len(historical_weights) >= keep_last:
                    historical_weights = sorted(glob.glob(f"{history_folder}/*.tar"))
                    os.remove(historical_weights[0])
                # get file modified time
                mtime = os.path.getmtime(dest)
                _stamp = datetime.datetime.fromtimestamp(mtime).strftime(
                    "%y%m%d-%H%M%S"
                )
                # move to history folder
                os.rename(dest, f"{history_folder}/{filename}-{_stamp}{f.ext}")

        os.makedirs(f.dir, exist_ok=True)
        torch.save(state, dest)
        self._print("=> Saved checkpoint at '{}'".format(dest), level=0)
        if best:
            best_name = "{}-(best){}".format(filename, f.ext)
            shutil.copyfile(dest, os.path.join(f.dir, best_name))
            self._print(
                "  => copied this (best) result to '{}'".format(best_name), level=0
            )

    def _get_existing_ckpt(self, load=False):
        """Return the list of existing checkpoint files in reverse order (latest first)."""
        f = self.save_fn if not load else self.load_fn
        all_checkpoints = glob.glob(
            os.path.join(f.dir, "{}*{}".format(f.filename, f.ext))
        )
        return sorted(all_checkpoints, reverse=True)

    def load(self):
        f = self.load_fn
        # load checkpoint with latest timestamp
        all_checkpoints = self._get_existing_ckpt(load=True)
        if len(all_checkpoints) > 0:
            filename = all_checkpoints[0]
            self._print("=> loading latest checkpoint '{}'".format(filename), level=1)
            checkpoint = torch.load(filename)
            # args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            # model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' (epoch {})".format(
            #     self.filename, checkpoint['epoch']))
            return checkpoint
        else:
            self._print(
                "=> no checkpoint found at '{}'".format(f.full_filename), level=2
            )


############################################################
##  ~For logging to TensorBoard~  ##
############################################################


def tensorboard_logdir_new_subdir(directory):
    """Return a unique new subdir within the given directory."""
    _id = len(os.listdir(directory))
    filename = None
    while filename is None or os.path.exists(filename):
        _id += 1
        filename = os.path.join(directory, str(_id))
    return filename


class TensorboardSummary(object):
    def __init__(self, directory, id=None):
        assert type(directory) == str
        self.directory = directory
        if id is not None:
            self.directory += "_{}".format(id)

    def create_summary(self):
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(log_dir=self.directory)
        return writer


############################################################
##  ~Enhancement towards torch component~  ##
############################################################


class _RepeatSampler(object):
    """Sampler that repeats forever. (So that sampling with thread won't ends)"""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    """From https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048.
    This reuse worker process, which is extremely beneficial to short epoch, as it
    does not need to re-spawn threads from num_workers>0 every epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class CachedDataset(object):
    """A wrapper around torch.utils.data.dataset.Dataset that cache __getitem__."""

    def __init__(self, dataset, cache_across_multiprocess=True):
        """This lightweight wrapper enables caching compare to normal torch dataset.
        If cache_across_multiprocess is set to `True`, it will try to use
            multiprocessing.Manager's shared dict to share the cache.
        """
        if not isinstance(dataset, torch.utils.data.dataset.Dataset):
            raise ValueError(f"Unknown type {type(dataset)}")

        # mock this as the input dataset
        self.__class__ = type(
            dataset.__class__.__name__, (self.__class__, dataset.__class__), {}
        )
        self.__dict__ = dataset.__dict__

        # assign accessible variables for cached items
        self.__dataset = dataset
        if cache_across_multiprocess:
            import multiprocessing

            cache_manager = multiprocessing.Manager()
            self.__cached_items = cache_manager.dict()
        else:
            self.__cached_items = {}

    def __getitem__(self, index):
        if index not in self.__cached_items:
            self.__cached_items[index] = self.__dataset[index]
        return self.__cached_items[index]

    def prefetch(self):
        """This helps to avoid halt lock on multiple process trying to
        access the same file when num_workers > 0. This prefetch all data
        in-advanced. This should be called (if needed) before passing to
        DataLoader."""
        import tqdm

        for item in tqdm.tqdm(self, desc=f"Prefetching {self.__class__.__name__}"):
            pass


############################################################
##  ~For quick display image~  ##
############################################################
def get_clean_npimg(x, auto_reorder_dim=True):
    """Given x, return a clean npimg for imshow."""
    x_type = type(x)
    SUPPORTED_TYPE = [torch.Tensor, np.ndarray]
    if x_type not in SUPPORTED_TYPE:
        print("WARN: img is an unsupported tpye: {}.".format(x_type))
        return None
    if x_type == torch.Tensor:
        tensor_x = torch.Tensor.cpu(x).clone().detach()
        if len(tensor_x.shape) < 2 or len(tensor_x.shape) > 4:
            print("WARN: unsupported Tensor dimension: {}.".format(len(tensor_x.shape)))
            return None
        if len(tensor_x.shape) == 4:
            # we assume the format is in NxCxHxW. We only take the first one.
            if tensor_x.shape[0] > 1:
                print(
                    "INFO: Found bathc size > 1 (N={}), only going to take idx 0.".format(
                        tensor_x.shape[0]
                    )
                )
            tensor_x = tensor_x[0]
        x = tensor_x.numpy()
    ##  ~Should be all in numpy array format now~  ##
    if auto_reorder_dim:
        if len(x.shape) == 3:
            # attempt to reorder channel if needed
            # convention is either: (M,N), (M,N,3) or (M,N,4)
            acceptable_channel_num = (3, 4)
            if (
                x.shape[2] not in acceptable_channel_num
                and x.shape[0] in acceptable_channel_num
            ):
                # this is a common case and should be correct most of the time.
                print(
                    "> Possible wrong channel order detected. "
                    "Attempting to reorder dimension."
                )
                x = np.transpose(x, (1, 2, 0))
            elif x.shape[2] not in acceptable_channel_num and x_type == torch.Tensor:
                # this assume the Tensor convension channel ordering.
                print(
                    "> Possible wrong channel order detected. "
                    "Reordering dimension based on Tensor NCHW convention."
                )
                x = np.transpose(x, (1, 2, 0))
    return x


def show(img, show_info=False, d=None, dimension_order=None, auto_reorder_dim=True):
    """Display image from tensor."""

    def out(*argv, **kwargs):
        """Wrapper around print"""
        if show_info:
            print(*argv, **kwargs)

    def _imshow(_img, d):
        """Wrapper to display requested dimension only."""
        if d is None and len(_img.shape) == 3 and _img.shape[2] not in (3, 4):
            print("WARN: No [dim] is given and image channel number is not correct.")
            print("      Defaulting to d=0.")
            d = 0
        if d is None:
            plt.imshow(_img, interpolation="nearest")
        else:
            out("Showing only dimension: {}".format(d))
            plt.imshow(_img[:, :, d], interpolation="nearest")
            plt.colorbar()

    ###########################################################
    npimg = get_clean_npimg(img, auto_reorder_dim)
    if npimg is None:
        print("Ignoring display request.")
    if dimension_order:
        npimg = np.transpose(img, dimension_order)
    try:
        _imshow(npimg, d)
    except TypeError as e:
        print("========== DEBUG MESSAGE ==========")
        print("Input type: {}".format(type(img)))
        print("Input shape: {}".format(img.shape))
        print("Input shape after auto_reordering: {}".format(npimg.shape))
        print("Input repr: {}".format(repr(img)))
        print("===================================")
        raise e
    plt.show()
    out(npimg)


############################################################
##             Turn any matplotlib plt to img             ##
############################################################


def plt_fig_to_nparray(fig):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
    Return:
        np.array with shape = (x,y,d) where d=3
    """
    # remove white padding
    fig.tight_layout()

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    # img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8
    plt.close(fig)
    return img
    # # Add figure in numpy "image" to TensorBoard writer
    # writer.add_image('confusion_matrix', img, step)


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
        else:
            raise Exception("Unimplemented type {}".format(type(inx)))
        return "{}".format(shape)
