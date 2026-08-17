import numpy as np

from myriad.compression import (
    block_prune,
    compress_tensors,
    distillation_loss,
    magnitude_prune,
    quantize_tensor,
)
from myriad.models import CompressionConfig, DataType, Tensor


def test_quantization_preserves_shape_and_records_metrics():
    tensor = Tensor("weights", np.linspace(-1, 1, 100, dtype=np.float32))
    result = quantize_tensor(tensor, DataType.INT8)
    assert result.tensor.shape == tensor.shape
    assert result.tensor.dtype is DataType.INT8
    assert result.mse >= 0
    assert np.isfinite(result.snr_db)


def test_magnitude_pruning_hits_requested_count():
    tensor = Tensor("weights", np.arange(10, dtype=np.float32))
    result = magnitude_prune(tensor, 0.3)
    assert result.removed_values == 3
    assert result.achieved_sparsity == 0.3


def test_block_pruning_removes_full_blocks():
    tensor = Tensor("weights", np.arange(16, dtype=np.float32).reshape(4, 4))
    result = block_prune(tensor, 0.5, (2, 2))
    assert result.removed_values == 8
    assert np.all(result.tensor.data[:2, :2] == 0)


def test_distillation_loss_is_zero_for_equal_logits():
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert distillation_loss(logits, logits) < 1e-8


def test_compression_reduces_storage():
    tensor = Tensor("weights", np.ones((64, 64), dtype=np.float32))
    report = compress_tensors([tensor], CompressionConfig(target_dtype=DataType.INT4, global_sparsity=0.5))
    assert report.compressed_bytes < report.original_bytes
    assert report.achieved_sparsity == 0.5
