from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DataType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    BINARY = "binary"

    @property
    def bits(self) -> int:
        return {
            DataType.FP32: 32,
            DataType.FP16: 16,
            DataType.INT8: 8,
            DataType.INT4: 4,
            DataType.INT2: 2,
            DataType.BINARY: 1,
        }[self]


@dataclass(frozen=True)
class Tensor:
    name: str
    data: np.ndarray
    dtype: DataType = DataType.FP32
    scale: float | None = None
    zero_point: int | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def numel(self) -> int:
        return int(self.data.size)

    @property
    def bytes(self) -> int:
        return int(np.ceil(self.numel * self.dtype.bits / 8))


@dataclass(frozen=True)
class LayerSpec:
    name: str
    op: str
    input_names: tuple[str, ...]
    output_name: str
    shape: tuple[int, ...]
    weight_shape: tuple[int, ...] | None = None
    macs: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompressionConfig:
    target_dtype: DataType = DataType.INT8
    global_sparsity: float = 0.0
    block_shape: tuple[int, int] | None = None
    calibration_samples: int = 128
    percentile: float = 99.9
    distillation_weight: float = 0.0
    seed: int = 7


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    process_nm: float
    frequency_mhz: float
    voltage_v: float
    sram_kb: int
    dram_bandwidth_gbps: float
    mac_units: int
    macs_per_cycle: int
    energy_per_mac_pj: float
    energy_per_sram_byte_pj: float
    energy_per_dram_byte_pj: float
    area_per_mac_um2: float
    area_per_sram_kb_um2: float
    static_power_mw: float = 0.0

    @property
    def peak_macs_per_second(self) -> float:
        return self.mac_units * self.macs_per_cycle * self.frequency_mhz * 1e6

    @property
    def dram_bandwidth_bytes_per_second(self) -> float:
        return self.dram_bandwidth_gbps * 1e9 / 8


@dataclass(frozen=True)
class LayerEstimate:
    layer: LayerSpec
    latency_us: float
    energy_uj: float
    area_um2: float
    dram_bytes: int
    sram_bytes: int
    operational_intensity: float


@dataclass(frozen=True)
class DesignEstimate:
    hardware: HardwareSpec
    layers: tuple[LayerEstimate, ...]
    total_latency_us: float
    total_energy_uj: float
    total_area_um2: float
    peak_memory_bytes: int
    model_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def throughput_per_second(self) -> float:
        if self.total_latency_us <= 0:
            return float("inf")
        return 1e6 / self.total_latency_us


@dataclass(frozen=True)
class QuantizationResult:
    tensor: Tensor
    scale: float
    zero_point: int
    min_value: float
    max_value: float
    mse: float
    snr_db: float


@dataclass(frozen=True)
class PruningResult:
    tensor: Tensor
    achieved_sparsity: float
    threshold: float
    removed_values: int


@dataclass(frozen=True)
class CompileArtifact:
    graph_name: str
    source: str
    memory_plan: dict[str, tuple[int, int]]
    schedules: dict[str, tuple[str, ...]]
    metadata: dict[str, Any] = field(default_factory=dict)
