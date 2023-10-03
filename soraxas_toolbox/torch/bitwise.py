import functools

import torch
from math import log2

from typing import List

MAX_BITWISE_SHIFT = 63


def decode_bits(
    x: torch.LongTensor,
    output_digit: bool = True,
    debug: bool = False,
) -> List[int]:
    """Given a binary integer, returns a list of decoded bits of digit.

    >>> decode_bits(1)
    [0]
    >>> decode_bits((1 << 2) | (1 << 5))
    [2, 5]
    >>> decode_bits(torch.bitwise_left_shift(1, torch.LongTensor([2, 1, 0, 2, 4])))
    [0, 1, 2, 4]
    """

    def _format_bin(binary, extract_digit=True):
        digit = "-"
        if extract_digit:
            digit = int(log2(binary))
        return f"0b{binary:08b} [{digit}] val:{binary}"

    if not isinstance(x, torch.Tensor):
        x = torch.LongTensor([x])

    if debug:
        _unique = x.unique()
        print(f">> unique:{_unique}")
        for u in _unique:
            print(f"ie {_format_bin(u, extract_digit=False)}")

    out = []
    for i in range(MAX_BITWISE_SHIFT):
        val = 1 << i
        if (((x & val) == val) > 0).any():
            if output_digit:
                out.append(int(log2(val)))
            else:
                out.append(val)

            if debug:
                print(f"|> {_format_bin(val)}")
    return out


def bitwise_or_reduce(x: torch.LongTensor, dim: int) -> torch.LongTensor:
    """Reduction operation on a tensor along a dimension

    >>> b = bitwise_or_reduce(torch.bitwise_left_shift(1, torch.LongTensor([2, 1, 1, 1, 4])), dim=0)
    >>> b
    tensor(22)
    >>> decode_bits(b)
    [1, 2, 4]
    """

    # build slicing index
    indices = [slice(None)] * len(x.shape)

    def get_indices(i):
        indices[dim] = i
        return tuple(indices)

    ######################

    return functools.reduce(
        lambda bit_mask, i: torch.bitwise_or(bit_mask, x[get_indices(i)]),
        range(1, x.shape[dim]),
        x[get_indices(0)],  # initializer
    )


def has_common_bitset(
    a: torch.LongTensor,
    b: torch.LongTensor,
    discard_bit_zero: bool = True,
) -> torch.BoolTensor:
    """Return a boolean mask depending on whether the two tensor has commont bit set.

    >>> has_common_bitset(torch.LongTensor([0, 1<<2]), torch.LongTensor([0, 1<<3])).tolist()
    [False, False]
    >>> has_common_bitset(torch.LongTensor([0, 1<<2]), torch.LongTensor([0, 1<<3]), False).tolist()
    [True, False]
    >>> has_common_bitset(torch.LongTensor([0, 1<<2]), torch.LongTensor([0, 1<<3 | 1<<2])).tolist()
    [False, True]
    """
    if discard_bit_zero:
        # all common bit should be greater than zero
        return torch.bitwise_and(a, b) > 0
    else:
        return torch.bitwise_and(a, b) == a


def build_combined_bitmask(
    mask: torch.LongTensor, dim: int = 0, remove_class_zero: bool = True
) -> torch.LongTensor:
    """Given a tensor in the shape of AxBxCx...xZ, and the input tensor is a class
    variable from [0,C) for there are C classes.

    If dim is 1,
    returns a combined bitmask tensor with shape AxCx...xZ where all
    bits along dim=1 will be bit-or. This reduce the dimensionality of the tensor

    >>> segmentation_class_data = torch.LongTensor([[[0, 1], [2, 3], [5,9]], [[6, 7], [6, 0], [1,7]]])
    >>> segmentation_class_data.shape
    torch.Size([2, 3, 2])

    >>> out = build_combined_bitmask(segmentation_class_data, dim=0)
    >>> out
    tensor([[ 64, 130],
            [ 68,   8],
            [ 34, 640]])
    >>> decode_bits(out[0])
    [1, 6, 7]
    >>> segmentation_class_data[:, 0]
    tensor([[0, 1],
            [6, 7]])

    >>> out = build_combined_bitmask(segmentation_class_data, dim=1, remove_class_zero=False)
    >>> out
    tensor([[ 37, 522],
            [ 66, 129]])
    >>> out[..., 1].tolist()
    [522, 129]
    >>> decode_bits(out[..., 1])
    [0, 1, 3, 7, 9]
    >>> segmentation_class_data[..., 1]
    tensor([[1, 3, 9],
            [7, 0, 7]])
    """
    assert mask.dtype in (torch.int64,), f"Unsupported dtype: {mask.dtype}"
    assert mask.min() >= 0 and mask.max() < MAX_BITWISE_SHIFT

    shifted_bits = torch.bitwise_left_shift(1, mask)

    # if a class is zero, its bit-shifted will have value 1
    if remove_class_zero:
        shifted_bits[shifted_bits == 1] = 0
    return bitwise_or_reduce(shifted_bits, dim=dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
