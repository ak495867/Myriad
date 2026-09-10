from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from myriad.analysis import estimate_design, silicon_cost
from myriad.compiler import Graph, compile_graph, lower_dense_layer
from myriad.compression import compress_tensors
from myriad.hardware import edge_npu
from myriad.models import CompressionConfig, DataType, Tensor


def main() -> None:
    rng = np.random.default_rng(7)
    tensors = [
        Tensor("encoder.weight", rng.normal(size=(256, 128)).astype(np.float32)),
        Tensor("projection.weight", rng.normal(size=(128, 64)).astype(np.float32)),
    ]
    compression = compress_tensors(
        tensors,
        CompressionConfig(
            target_dtype=DataType.INT4, global_sparsity=0.5, block_shape=(4, 4)
        ),
    )
    graph = Graph("myriad_example")
    graph.add_input("features", (1, 128))
    graph.add_layer(lower_dense_layer("encoder", "features", "encoded", 1, 128, 256))
    graph.add_layer(lower_dense_layer("projection", "encoded", "output", 1, 256, 128))
    hardware = edge_npu()
    design = estimate_design(
        graph.layers,
        hardware,
        parameter_bytes=1,
        activation_bytes=1,
        model_bytes=compression.compressed_bytes,
    )
    artifact = compile_graph(graph, bytes_per_element=1, parallelism=2)
    payload = {
        "compression": {
            "original_bytes": compression.original_bytes,
            "compressed_bytes": compression.compressed_bytes,
            "size_reduction": compression.size_reduction,
            "average_snr_db": compression.average_snr_db,
            "achieved_sparsity": compression.achieved_sparsity,
        },
        "design": {
            "hardware": design.hardware.name,
            "latency_us": design.total_latency_us,
            "energy_uj": design.total_energy_uj,
            "area_mm2": design.total_area_um2 / 1_000_000,
            "peak_memory_kb": design.peak_memory_bytes / 1024,
        },
        "silicon": silicon_cost(design),
        "compiler": {
            "source": artifact.source,
            "memory_plan": artifact.memory_plan,
            "schedules": artifact.schedules,
        },
    }
    output = Path("artifacts/end_to_end.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
