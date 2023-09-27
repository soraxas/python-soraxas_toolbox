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
    """A wrapper around torch.utils.data.dat aset.Dataset that cache __getitem__."""

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
