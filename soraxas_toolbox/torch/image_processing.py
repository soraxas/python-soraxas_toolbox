import torch
import einops


def subdivide_img_into_blocks(img, blk_height=8, blk_width=8):
    """Subdivide an input image into fix-size smaller blocks.

    >>> _input = torch.linspace(1, 6 * 6, 6 * 6).long().reshape(1, 6, 6)
    >>> _input.shape
    torch.Size([1, 6, 6])
    >>> _input
    tensor([[[ 1,  2,  3,  4,  5,  6],
             [ 7,  8,  9, 10, 11, 12],
             [13, 14, 15, 16, 17, 18],
             [19, 20, 21, 22, 23, 24],
             [25, 26, 27, 28, 29, 30],
             [31, 32, 33, 34, 35, 36]]])

    >>> block_dims = (2, 2)
    >>> divided = subdivide_img_into_blocks(_input, *block_dims)
    >>> divided.shape
    torch.Size([1, 9, 2, 2])
    >>> divided.squeeze(0)[0]
    tensor([[1, 2],
            [7, 8]])
    >>> divided.squeeze(0)[4]
    tensor([[15, 16],
            [21, 22]])
    >>> divided.squeeze(0)[7]
    tensor([[27, 28],
            [33, 34]])

    >>> block_dims = (3, 2)
    >>> divided = subdivide_img_into_blocks(_input, *block_dims)
    >>> divided.squeeze(0)[0]
    tensor([[ 1,  2],
            [ 7,  8],
            [13, 14]])
    >>> divided.squeeze(0)[-1]
    tensor([[23, 24],
            [29, 30],
            [35, 36]])
    """
    assert len(img.shape) >= 2
    # image is in Batch X Height X Width
    return einops.rearrange(
        img,
        "... (n_blks_h blk_height) (n_blks_w blk_width) -> "
        "... (n_blks_h n_blks_w) blk_height blk_width",
        blk_height=blk_height,
        blk_width=blk_width,
    )


def pairwise_mask_subpatch_cross_matching(
    mask0: torch.LongTensor,
    mask1: torch.LongTensor,
    patch_height: int,
    patch_width: int,
):
    """Perform a cross-wise comparison based on sub-patch

    >>> a = torch.LongTensor([[1, 2, 0, 1], [3, 4, 3, 5], [1, 5, 3, 1], [1, 2, 0, 0]])
    >>> b = torch.LongTensor([[2, 2, 0, 1], [2, 2, 1, 4], [8, 5, 2, 0], [5, 0, 3, 3]])
    >>> a
    tensor([[1, 2, 0, 1],
            [3, 4, 3, 5],
            [1, 5, 3, 1],
            [1, 2, 0, 0]])
    >>> b
    tensor([[2, 2, 0, 1],
            [2, 2, 1, 4],
            [8, 5, 2, 0],
            [5, 0, 3, 3]])

    >>> out = pairwise_mask_subpatch_cross_matching(a, b, 2, 2)
    >>> out.shape
    torch.Size([4, 4])

    >>> # comparing top-left corner to all other patches in 'b'
    >>> out[0].reshape(2, 2)
    tensor([[ True,  True],
            [False,  True]])

    >>> # comparing bottom-right corner to all other patches in 'b'
    >>> out[-1].reshape(2, 2)
    tensor([[False,  True],
            [False,  True]])

    """
    from . import bitwise

    blocks_of_mask0 = subdivide_img_into_blocks(
        mask0, blk_height=patch_height, blk_width=patch_width
    )
    blocks_of_mask1 = subdivide_img_into_blocks(
        mask1, blk_height=patch_height, blk_width=patch_width
    )

    m1 = bitwise.build_combined_bitmask(
        einops.rearrange(blocks_of_mask0, "... h w -> ... (h w)"), dim=-1
    )

    m2 = bitwise.build_combined_bitmask(
        einops.rearrange(blocks_of_mask1, "... h w -> ... (h w)"), dim=-1
    )

    cross_compare = bitwise.has_common_bitset(m1[..., :, None], m2[..., None, :])
    return cross_compare


if __name__ == "__main__":
    import doctest

    doctest.testmod()
