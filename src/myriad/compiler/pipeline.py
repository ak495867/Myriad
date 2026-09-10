from __future__ import annotations

import json

from myriad.compiler.ir import Graph
from myriad.compiler.scheduler import memory_plan, schedule_layers, topological_order
from myriad.models import CompileArtifact, DataType


def compile_graph(
    graph: Graph, bytes_per_element: int = 1, parallelism: int = 4
) -> CompileArtifact:
    graph.validate()
    plan = memory_plan(graph, bytes_per_element=bytes_per_element)
    schedules = schedule_layers(graph, parallelism=parallelism)
    order = topological_order(graph)
    lines = [
        f"module {graph.name}",
        f"inputs {len([value for value in graph.values.values() if value.producer is None])}",
        f"layers {len(graph.layers)}",
    ]
    for layer_name in order:
        layer = next(layer for layer in graph.layers if layer.name == layer_name)
        lines.append(
            f"op {layer.name} {layer.op} inputs={','.join(layer.input_names)} output={layer.output_name} macs={layer.macs}"
        )
    lines.append("schedule " + json.dumps(schedules, sort_keys=True))
    lines.append("memory " + json.dumps(plan, sort_keys=True))
    source = "\n".join(lines) + "\n"
    metadata = {
        "dtype": DataType.INT8.value if bytes_per_element == 1 else DataType.FP32.value,
        "bytes_per_element": bytes_per_element,
        "parallelism": parallelism,
        "layer_order": order,
    }
    return CompileArtifact(
        graph_name=graph.name,
        source=source,
        memory_plan=plan,
        schedules=schedules,
        metadata=metadata,
    )
