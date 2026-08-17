from __future__ import annotations

from myriad.models import HardwareSpec


def validate_hardware(spec: HardwareSpec) -> None:
    values = {
        "process_nm": spec.process_nm,
        "frequency_mhz": spec.frequency_mhz,
        "voltage_v": spec.voltage_v,
        "sram_kb": spec.sram_kb,
        "dram_bandwidth_gbps": spec.dram_bandwidth_gbps,
        "mac_units": spec.mac_units,
        "macs_per_cycle": spec.macs_per_cycle,
        "energy_per_mac_pj": spec.energy_per_mac_pj,
        "energy_per_sram_byte_pj": spec.energy_per_sram_byte_pj,
        "energy_per_dram_byte_pj": spec.energy_per_dram_byte_pj,
        "area_per_mac_um2": spec.area_per_mac_um2,
        "area_per_sram_kb_um2": spec.area_per_sram_kb_um2,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError("hardware quantities must be positive")
    if spec.static_power_mw < 0:
        raise ValueError("static power cannot be negative")


def edge_npu() -> HardwareSpec:
    return HardwareSpec(
        name="edge-npu",
        process_nm=7,
        frequency_mhz=800,
        voltage_v=0.75,
        sram_kb=2048,
        dram_bandwidth_gbps=25.6,
        mac_units=1024,
        macs_per_cycle=2,
        energy_per_mac_pj=0.35,
        energy_per_sram_byte_pj=0.08,
        energy_per_dram_byte_pj=12.0,
        area_per_mac_um2=2.4,
        area_per_sram_kb_um2=120.0,
        static_power_mw=75.0,
    )


def datacenter_npu() -> HardwareSpec:
    return HardwareSpec(
        name="datacenter-npu",
        process_nm=5,
        frequency_mhz=1200,
        voltage_v=0.8,
        sram_kb=32768,
        dram_bandwidth_gbps=900,
        mac_units=65536,
        macs_per_cycle=2,
        energy_per_mac_pj=0.25,
        energy_per_sram_byte_pj=0.06,
        energy_per_dram_byte_pj=8.0,
        area_per_mac_um2=1.9,
        area_per_sram_kb_um2=105.0,
        static_power_mw=450.0,
    )


def reference_hardware(name: str) -> HardwareSpec:
    presets = {"edge-npu": edge_npu, "datacenter-npu": datacenter_npu}
    try:
        spec = presets[name.lower()]()
    except KeyError as error:
        raise ValueError(f"unknown hardware preset: {name}") from error
    validate_hardware(spec)
    return spec
