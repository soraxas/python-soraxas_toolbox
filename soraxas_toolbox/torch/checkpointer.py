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
