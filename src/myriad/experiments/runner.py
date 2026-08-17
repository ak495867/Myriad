from __future__ import annotations

from collections.abc import Iterable
from itertools import product

from myriad.analysis.estimator import estimate_design, summarize_design
from myriad.compiler.ir import Graph
from myriad.compression.pipeline import compress_tensors
from myriad.hardware.spec import reference_hardware
from myriad.models import CompressionConfig, DataType, LayerSpec, Tensor


def run_experiment(
    tensors: list[Tensor],
    layers: list[LayerSpec],
    compression: CompressionConfig,
    hardware_name: str,
) -> dict[str, object]:
    compressed = compress_tensors(tensors, compression)
    parameter_bytes = max(1, compression.target_dtype.bits // 8)
    design = estimate_design(
        layers,
        reference_hardware(hardware_name),
        parameter_bytes=parameter_bytes,
        activation_bytes=parameter_bytes,
        model_bytes=compressed.compressed_bytes,
    )
    return {
        "compression": {
            "target_dtype": compression.target_dtype.value,
            "global_sparsity": compression.global_sparsity,
            "block_shape": compression.block_shape,
            "size_reduction": compressed.size_reduction,
            "average_snr_db": compressed.average_snr_db,
            "achieved_sparsity": compressed.achieved_sparsity,
        },
        "design": summarize_design(design),
        "silicon": {
            "hardware": design.hardware.name,
            "process_nm": design.hardware.process_nm,
            "frequency_mhz": design.hardware.frequency_mhz,
            "mac_units": design.hardware.mac_units,
        },
    }


def sweep(
    tensors: list[Tensor],
    layers: list[LayerSpec],
    dtypes: Iterable[DataType],
    sparsities: Iterable[float],
    hardware_names: Iterable[str],
) -> list[dict[str, object]]:
    results = []
    for dtype, sparsity, hardware_name in product(dtypes, sparsities, hardware_names):
        config = CompressionConfig(target_dtype=dtype, global_sparsity=sparsity)
        results.append(run_experiment(tensors, layers, config, hardware_name))
    return results


def graph_from_layers(name: str, layers: list[LayerSpec]) -> Graph:
    graph = Graph(name=name)
    for layer in layers:
        for input_name in layer.input_names:
            if input_name not in graph.values:
                shape = (layer.shape[0], layer.weight_shape[1]) if layer.weight_shape else layer.shape
                graph.add_input(input_name, shape)
        graph.add_layer(layer)
    return graph
