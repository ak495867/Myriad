from __future__ import annotations

import numpy as np

from myriad.models import DataType, QuantizationResult, Tensor


def _validate_dtype(dtype: DataType) -> None:
    if dtype not in {DataType.INT8, DataType.INT4, DataType.INT2, DataType.BINARY}:
        raise ValueError(f"quantization requires an integer dtype, received {dtype.value}")


def _bounds(dtype: DataType) -> tuple[int, int]:
    if dtype is DataType.BINARY:
        return -1, 1
    upper = 2 ** (dtype.bits - 1) - 1
    return -(2 ** (dtype.bits - 1)), upper


def quantize_tensor(
    tensor: Tensor,
    dtype: DataType,
    percentile: float = 99.9,
    symmetric: bool = True,
) -> QuantizationResult:
    _validate_dtype(dtype)
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in the interval (0, 100]")
    values = tensor.data.astype(np.float32, copy=False)
    if values.size == 0:
        raise ValueError("cannot quantize an empty tensor")
    limit = float(np.percentile(np.abs(values), percentile))
    lower, upper = _bounds(dtype)
    if symmetric:
        peak = max(limit, np.finfo(np.float32).eps)
        scale = peak / max(abs(lower), abs(upper))
        zero_point = 0
        quantized = np.clip(np.rint(values / scale), lower, upper).astype(np.int8)
    else:
        minimum = float(np.percentile(values, 100 - percentile))
        maximum = float(np.percentile(values, percentile))
        span = max(maximum - minimum, np.finfo(np.float32).eps)
        scale = span / (upper - lower)
        zero_point = int(np.clip(np.rint(lower - minimum / scale), lower, upper))
        quantized = np.clip(np.rint(values / scale + zero_point), lower, upper).astype(np.int8)
    dequantized = (quantized.astype(np.float32) - zero_point) * scale
    error = values - dequantized
    mse = float(np.mean(error * error))
    signal = float(np.mean(values * values))
    snr_db = float(10 * np.log10(max(signal, np.finfo(np.float32).eps) / max(mse, np.finfo(np.float32).eps)))
    result_tensor = Tensor(
        name=tensor.name,
        data=quantized,
        dtype=dtype,
        scale=float(scale),
        zero_point=zero_point,
    )
    return QuantizationResult(
        tensor=result_tensor,
        scale=float(scale),
        zero_point=zero_point,
        min_value=float(np.min(values)),
        max_value=float(np.max(values)),
        mse=mse,
        snr_db=snr_db,
    )


def dequantize_tensor(tensor: Tensor) -> np.ndarray:
    if tensor.scale is None or tensor.zero_point is None:
        raise ValueError("tensor does not contain quantization parameters")
    return (tensor.data.astype(np.float32) - tensor.zero_point) * tensor.scale


def quantize_model(tensors: list[Tensor], dtype: DataType, percentile: float = 99.9) -> list[QuantizationResult]:
    return [quantize_tensor(item, dtype, percentile=percentile) for item in tensors]
