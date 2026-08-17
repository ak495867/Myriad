from __future__ import annotations

import numpy as np

from myriad.models import PruningResult, Tensor


def _validate_sparsity(sparsity: float) -> None:
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be in the interval [0, 1)")


def magnitude_prune(tensor: Tensor, sparsity: float) -> PruningResult:
    _validate_sparsity(sparsity)
    values = tensor.data.astype(np.float32, copy=True)
    if values.size == 0 or sparsity == 0:
        return PruningResult(tensor=tensor, achieved_sparsity=0.0, threshold=0.0, removed_values=0)
    count = int(np.floor(values.size * sparsity))
    if count == 0:
        return PruningResult(tensor=tensor, achieved_sparsity=0.0, threshold=0.0, removed_values=0)
    magnitudes = np.abs(values).reshape(-1)
    threshold = float(np.partition(magnitudes, count - 1)[count - 1])
    mask = np.abs(values) <= threshold
    if int(np.sum(mask)) > count:
        candidates = np.flatnonzero(mask.reshape(-1))
        keep_zero_count = int(np.sum(mask)) - count
        mask.reshape(-1)[candidates[:keep_zero_count]] = False
    values[mask] = 0
    removed = int(np.sum(values == 0))
    sparse_tensor = Tensor(
        name=tensor.name,
        data=values.astype(tensor.data.dtype),
        dtype=tensor.dtype,
        scale=tensor.scale,
        zero_point=tensor.zero_point,
    )
    return PruningResult(
        tensor=sparse_tensor,
        achieved_sparsity=removed / values.size,
        threshold=threshold,
        removed_values=removed,
    )


def block_prune(tensor: Tensor, sparsity: float, block_shape: tuple[int, int]) -> PruningResult:
    _validate_sparsity(sparsity)
    if len(block_shape) != 2 or min(block_shape) <= 0:
        raise ValueError("block_shape must contain two positive dimensions")
    values = tensor.data.astype(np.float32, copy=True)
    if values.ndim != 2:
        raise ValueError("block pruning requires a two-dimensional tensor")
    rows, cols = values.shape
    block_rows, block_cols = block_shape
    padded_rows = int(np.ceil(rows / block_rows) * block_rows)
    padded_cols = int(np.ceil(cols / block_cols) * block_cols)
    padded = np.zeros((padded_rows, padded_cols), dtype=np.float32)
    padded[:rows, :cols] = values
    blocks = padded.reshape(padded_rows // block_rows, block_rows, padded_cols // block_cols, block_cols)
    scores = np.mean(np.abs(blocks), axis=(1, 3))
    count = int(np.floor(scores.size * sparsity))
    if count == 0:
        return PruningResult(tensor=tensor, achieved_sparsity=0.0, threshold=0.0, removed_values=0)
    threshold = float(np.partition(scores.reshape(-1), count - 1)[count - 1])
    block_mask = scores <= threshold
    if int(np.sum(block_mask)) > count:
        candidates = np.flatnonzero(block_mask.reshape(-1))
        keep_count = int(np.sum(block_mask)) - count
        block_mask.reshape(-1)[candidates[:keep_count]] = False
    blocks[block_mask, :, :] = 0
    result = padded[:rows, :cols]
    removed = int(np.sum(result == 0))
    sparse_tensor = Tensor(
        name=tensor.name,
        data=result.astype(tensor.data.dtype),
        dtype=tensor.dtype,
        scale=tensor.scale,
        zero_point=tensor.zero_point,
    )
    return PruningResult(
        tensor=sparse_tensor,
        achieved_sparsity=removed / values.size,
        threshold=threshold,
        removed_values=removed,
    )
