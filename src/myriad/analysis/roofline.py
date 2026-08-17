from __future__ import annotations

from myriad.analysis.estimator import estimate_layer
from myriad.models import DesignEstimate, HardwareSpec, LayerSpec


def classify_layer(layer: LayerSpec, hardware: HardwareSpec, parameter_bytes: int = 1, activation_bytes: int = 1) -> str:
    estimate = estimate_layer(layer, hardware, parameter_bytes=parameter_bytes, activation_bytes=activation_bytes)
    machine_balance = hardware.peak_macs_per_second / hardware.dram_bandwidth_bytes_per_second
    return "compute_bound" if estimate.operational_intensity >= machine_balance else "memory_bound"


def roofline_points(layers: list[LayerSpec], hardware: HardwareSpec, parameter_bytes: int = 1, activation_bytes: int = 1) -> list[dict[str, float | str]]:
    machine_balance = hardware.peak_macs_per_second / hardware.dram_bandwidth_bytes_per_second
    points = []
    for layer in layers:
        estimate = estimate_layer(layer, hardware, parameter_bytes=parameter_bytes, activation_bytes=activation_bytes)
        attainable = min(hardware.peak_macs_per_second, estimate.operational_intensity * hardware.dram_bandwidth_bytes_per_second)
        points.append(
            {
                "layer": layer.name,
                "operational_intensity": estimate.operational_intensity,
                "attainable_macs_per_second": attainable,
                "machine_balance": machine_balance,
                "bound": "compute_bound" if estimate.operational_intensity >= machine_balance else "memory_bound",
            }
        )
    return points


def silicon_cost(estimate: DesignEstimate, wafer_diameter_mm: float = 300.0, die_yield: float = 0.9, wafer_cost_usd: float = 17000.0) -> dict[str, float]:
    if wafer_diameter_mm <= 0 or not 0 < die_yield <= 1 or wafer_cost_usd <= 0:
        raise ValueError("wafer and yield parameters are invalid")
    die_area_mm2 = estimate.total_area_um2 / 1_000_000
    wafer_area_mm2 = 3.141592653589793 * (wafer_diameter_mm / 2) ** 2
    usable_dies = max((wafer_area_mm2 / max(die_area_mm2, 1e-9)) * 0.9, 1.0)
    good_dies = usable_dies * die_yield
    cost_per_die = wafer_cost_usd / good_dies
    return {
        "die_area_mm2": die_area_mm2,
        "estimated_good_dies_per_wafer": good_dies,
        "estimated_cost_per_die_usd": cost_per_die,
        "wafer_diameter_mm": wafer_diameter_mm,
        "die_yield": die_yield,
    }
