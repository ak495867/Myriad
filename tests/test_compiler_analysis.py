from myriad.analysis import classify_layer, estimate_design, roofline_points, silicon_cost
from myriad.compiler import Graph, compile_graph, lower_dense_layer, memory_plan
from myriad.hardware import edge_npu


def build_graph() -> Graph:
    graph = Graph("test")
    graph.add_input("features", (1, 8))
    graph.add_layer(lower_dense_layer("first", "features", "hidden", 1, 8, 16))
    graph.add_layer(lower_dense_layer("second", "hidden", "output", 1, 16, 4))
    return graph


def test_graph_compiles_with_schedule_and_memory_plan():
    graph = build_graph()
    artifact = compile_graph(graph, bytes_per_element=1, parallelism=2)
    assert "module test" in artifact.source
    assert set(artifact.memory_plan) == {"features", "hidden", "output"}
    assert artifact.schedules["cycle_0"] == ("first", "second")


def test_memory_plan_reuses_dead_values():
    graph = build_graph()
    plan = memory_plan(graph, bytes_per_element=1)
    assert plan["features"][0] == plan["output"][0]


def test_hardware_estimate_is_positive():
    graph = build_graph()
    estimate = estimate_design(graph.layers, edge_npu(), parameter_bytes=1, activation_bytes=1)
    assert estimate.total_latency_us > 0
    assert estimate.total_energy_uj > 0
    assert estimate.total_area_um2 > 0
    assert estimate.throughput_per_second > 0


def test_roofline_and_cost_results_are_finite():
    graph = build_graph()
    hardware = edge_npu()
    points = roofline_points(graph.layers, hardware)
    estimate = estimate_design(graph.layers, hardware)
    cost = silicon_cost(estimate)
    assert len(points) == 2
    assert all(point["bound"] in {"compute_bound", "memory_bound"} for point in points)
    assert cost["estimated_cost_per_die_usd"] > 0
    assert classify_layer(graph.layers[0], hardware) in {"compute_bound", "memory_bound"}
