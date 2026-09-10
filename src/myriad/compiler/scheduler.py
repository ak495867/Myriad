from __future__ import annotations

from collections import defaultdict

from myriad.compiler.ir import Graph


def topological_order(graph: Graph) -> tuple[str, ...]:
    graph.validate()
    return tuple(layer.name for layer in graph.layers)


def schedule_layers(graph: Graph, parallelism: int = 1) -> dict[str, tuple[str, ...]]:
    if parallelism <= 0:
        raise ValueError("parallelism must be positive")
    graph.validate()
    groups: dict[int, list[str]] = defaultdict(list)
    for index, layer in enumerate(graph.layers):
        groups[index // parallelism].append(layer.name)
    return {f"cycle_{cycle}": tuple(names) for cycle, names in sorted(groups.items())}


def memory_plan(graph: Graph, bytes_per_element: int = 4) -> dict[str, tuple[int, int]]:
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive")
    graph.validate()
    uses: dict[str, list[int]] = defaultdict(list)
    for index, layer in enumerate(graph.layers):
        for name in layer.input_names:
            uses[name].append(index)
    intervals: list[tuple[str, int, int, int]] = []
    for name, value in graph.values.items():
        if value.producer is None:
            start = 0
        else:
            start = next(
                index
                for index, layer in enumerate(graph.layers)
                if layer.name == value.producer
            )
        end = max(uses.get(name, [start]))
        size = value.numel * bytes_per_element
        intervals.append((name, start, end, size))
    intervals.sort(key=lambda item: (item[1], item[0]))
    active: list[tuple[str, int, int, int]] = []
    placements: dict[str, tuple[int, int]] = {}
    for name, start, end, size in intervals:
        active = [item for item in active if item[2] >= start]
        offset = 0
        for active_name, _, _, _ in sorted(
            active, key=lambda item: placements[item[0]][0]
        ):
            occupied_start, occupied_size = placements[active_name]
            if (
                offset + size <= occupied_start
                or offset >= occupied_start + occupied_size
            ):
                continue
            offset = occupied_start + occupied_size
        placements[name] = (offset, size)
        active.append((name, start, end, size))
    return placements
