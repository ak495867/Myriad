from __future__ import annotations

import math

from myriad.models import DesignEstimate, HardwareSpec, LayerEstimate, LayerSpec


def estimate_layer(
    layer: LayerSpec,
    hardware: HardwareSpec,
    parameter_bytes: int = 1,
    activation_bytes: int = 1,
) -> LayerEstimate:
    if parameter_bytes <= 0 or activation_bytes <= 0:
        raise ValueError("byte sizes must be positive")
    activation_elements = math.prod(layer.shape)
    weight_elements = math.prod(layer.weight_shape) if layer.weight_shape else 0
    sram_bytes = (
        activation_elements * activation_bytes + weight_elements * parameter_bytes
    )
    dram_bytes = (
        weight_elements * parameter_bytes + activation_elements * activation_bytes
    )
    compute_seconds = layer.macs / hardware.peak_macs_per_second if layer.macs else 0.0
    memory_seconds = dram_bytes / hardware.dram_bandwidth_bytes_per_second
    latency_seconds = max(compute_seconds, memory_seconds)
    mac_energy = layer.macs * hardware.energy_per_mac_pj
    sram_energy = sram_bytes * hardware.energy_per_sram_byte_pj
    dram_energy = dram_bytes * hardware.energy_per_dram_byte_pj
    energy_uj = (mac_energy + sram_energy + dram_energy) / 1_000_000
    mac_area = hardware.mac_units * hardware.area_per_mac_um2
    sram_area = hardware.sram_kb * hardware.area_per_sram_kb_um2
    area_um2 = mac_area + sram_area
    operational_intensity = layer.macs / max(dram_bytes, 1)
    return LayerEstimate(
        layer=layer,
        latency_us=latency_seconds * 1e6,
        energy_uj=energy_uj,
        area_um2=area_um2,
        dram_bytes=dram_bytes,
        sram_bytes=sram_bytes,
        operational_intensity=operational_intensity,
    )


def estimate_design(
    layers: list[LayerSpec],
    hardware: HardwareSpec,
    parameter_bytes: int = 1,
    activation_bytes: int = 1,
    model_bytes: int | None = None,
) -> DesignEstimate:
    estimates = tuple(
        estimate_layer(
            layer,
            hardware,
            parameter_bytes=parameter_bytes,
            activation_bytes=activation_bytes,
        )
        for layer in layers
    )
    peak_memory = max((item.sram_bytes for item in estimates), default=0)
    return DesignEstimate(
        hardware=hardware,
        layers=estimates,
        total_latency_us=sum(item.latency_us for item in estimates),
        total_energy_uj=sum(item.energy_uj for item in estimates),
        total_area_um2=max((item.area_um2 for item in estimates), default=0.0),
        peak_memory_bytes=peak_memory,
        model_bytes=(
            model_bytes
            if model_bytes is not None
            else sum(item.dram_bytes for item in estimates)
        ),
        metadata={
            "parameter_bytes": parameter_bytes,
            "activation_bytes": activation_bytes,
        },
    )


def summarize_design(estimate: DesignEstimate) -> dict[str, float | int | str]:
    return {
        "hardware": estimate.hardware.name,
        "latency_us": estimate.total_latency_us,
        "energy_uj": estimate.total_energy_uj,
        "area_mm2": estimate.total_area_um2 / 1_000_000,
        "peak_memory_kb": estimate.peak_memory_bytes / 1024,
        "model_size_mb": estimate.model_bytes / (1024 * 1024),
        "throughput_per_second": estimate.throughput_per_second,
    }
