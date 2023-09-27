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
