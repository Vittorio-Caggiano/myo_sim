"""wp.launch with a block size chosen from the grid size.

Warp's default of 256 threads per block runs a per-world kernel at 1024 worlds on only 4 blocks. The heuristic follows
Bolt (https://github.com/willwng/bolt, warp_util.block_dim_for, William Wang et al., Apache-2.0).
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import warp as wp

_MIN_BLOCKS_PER_SM = 2
_MIN_BLOCK_DIM, _MAX_BLOCK_DIM = 32, 256


def block_dim_for(nthreads: int, device: wp.Device) -> int:
    """Largest power of two in [32, 256] that still gives every SM at least two blocks."""
    per_block = nthreads // (_MIN_BLOCKS_PER_SM * device.sm_count)
    if per_block < _MIN_BLOCK_DIM:
        return _MIN_BLOCK_DIM
    return min(_MAX_BLOCK_DIM, 1 << int(math.log2(per_block)))


def launch(kernel, dim, *args, **kwargs):
    """wp.launch that picks block_dim from the grid size on CUDA devices."""
    device = wp.get_device(kwargs.get("device"))
    if device.is_cuda and kwargs.get("block_dim") is None:
        nthreads = math.prod(dim) if isinstance(dim, (tuple, list)) else int(dim)
        kwargs["block_dim"] = block_dim_for(nthreads, device)
    return wp.launch(kernel, dim, *args, **kwargs)


@contextmanager
def auto_block_dim():
    """Applies block_dim_for to every wp.launch in the block, including mujoco_warp's. Use around a graph capture."""
    plain = wp.launch

    def patched(kernel, dim, *args, **kwargs):
        device = wp.get_device(kwargs.get("device"))
        if device.is_cuda and kwargs.get("block_dim") is None:
            nthreads = math.prod(dim) if isinstance(dim, (tuple, list)) else int(dim)
            kwargs["block_dim"] = block_dim_for(nthreads, device)
        return plain(kernel, dim, *args, **kwargs)

    wp.launch = patched
    try:
        yield
    finally:
        wp.launch = plain
