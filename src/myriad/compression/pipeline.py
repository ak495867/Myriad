from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from myriad.compression.pruning import block_prune, magnitude_prune
from myriad.compression.quantization import quantize_model
from myriad.models import CompressionConfig, DataType, Tensor


@dataclass(frozen=True)
class CompressionReport:
    tensors: tuple[Tensor, ...]
    original_bytes: int
    compressed_bytes: int
    average_snr_db: float
    achieved_sparsity: float

    @property
    def size_reduction(self) -> float:
        if self.original_bytes == 0:
            return 0.0
        return 1 - self.compressed_bytes / self.original_bytes


def distillation_loss(student: np.ndarray, teacher: np.ndarray, temperature: float = 2.0) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher logits must have equal shapes")
    student_logits = student.astype(np.float64) / temperature
    teacher_logits = teacher.astype(np.float64) / temperature
    student_logits -= np.max(student_logits, axis=-1, keepdims=True)
    teacher_logits -= np.max(teacher_logits, axis=-1, keepdims=True)
    student_prob = np.exp(student_logits)
    teacher_prob = np.exp(teacher_logits)
    student_prob /= np.sum(student_prob, axis=-1, keepdims=True)
    teacher_prob /= np.sum(teacher_prob, axis=-1, keepdims=True)
    divergence = teacher_prob * (
        np.log(np.maximum(teacher_prob, 1e-12)) - np.log(np.maximum(student_prob, 1e-12))
    )
    return float(np.mean(np.sum(divergence, axis=-1)) * temperature * temperature)


def compress_tensors(tensors: list[Tensor], config: CompressionConfig) -> CompressionReport:
    rng = np.random.default_rng(config.seed)
    compressed: list[Tensor] = []
    snr_values: list[float] = []
    removed_values = 0
    total_values = 0
    original_bytes = sum(item.numel * DataType.FP32.bits // 8 for item in tensors)
    for tensor in tensors:
        current = Tensor(name=tensor.name, data=tensor.data.astype(np.float32), dtype=DataType.FP32)
        if config.global_sparsity > 0:
            if config.block_shape is not None and current.data.ndim == 2:
                pruned = block_prune(current, config.global_sparsity, config.block_shape)
            else:
                pruned = magnitude_prune(current, config.global_sparsity)
            current = pruned.tensor
            removed_values += pruned.removed_values
            total_values += current.numel
        if config.target_dtype is not DataType.FP32:
            quantized = quantize_model([current], config.target_dtype, config.percentile)[0]
            current = quantized.tensor
            snr_values.append(quantized.snr_db)
        compressed.append(current)
    compressed_bytes = sum(item.bytes for item in compressed)
    rng.shuffle(compressed)
    compressed.sort(key=lambda item: item.name)
    average_snr = float(np.mean(snr_values)) if snr_values else float("inf")
    achieved_sparsity = removed_values / total_values if total_values else 0.0
    return CompressionReport(
        tensors=tuple(compressed),
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        average_snr_db=average_snr,
        achieved_sparsity=achieved_sparsity,
    )
