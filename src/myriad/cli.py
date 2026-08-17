from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from myriad.analysis.roofline import roofline_points, silicon_cost
from myriad.compiler.ir import Graph, lower_dense_layer
from myriad.compiler.pipeline import compile_graph
from myriad.compression.pipeline import compress_tensors
from myriad.experiments.report import markdown_table, write_json
from myriad.experiments.runner import sweep
from myriad.hardware.spec import reference_hardware
from myriad.models import CompressionConfig, DataType, Tensor


def demo_tensors(seed: int = 7) -> list[Tensor]:
    rng = np.random.default_rng(seed)
    return [
        Tensor(name="encoder.weight", data=rng.normal(0, 0.4, size=(256, 128)).astype(np.float32)),
        Tensor(name="projection.weight", data=rng.normal(0, 0.2, size=(128, 64)).astype(np.float32)),
        Tensor(name="head.weight", data=rng.normal(0, 0.1, size=(64, 10)).astype(np.float32)),
    ]


def demo_layers() -> list:
    return [
        lower_dense_layer("encoder", "features", "encoded", 1, 128, 256),
        lower_dense_layer("projection", "encoded", "projected", 1, 256, 128),
        lower_dense_layer("head", "projected", "logits", 1, 128, 10),
    ]


def command_compress(args: argparse.Namespace) -> int:
    config = CompressionConfig(
        target_dtype=DataType(args.dtype),
        global_sparsity=args.sparsity,
        block_shape=(args.block_rows, args.block_cols) if args.block_rows and args.block_cols else None,
        percentile=args.percentile,
    )
    report = compress_tensors(demo_tensors(), config)
    payload = {
        "original_bytes": report.original_bytes,
        "compressed_bytes": report.compressed_bytes,
        "size_reduction": report.size_reduction,
        "average_snr_db": report.average_snr_db,
        "achieved_sparsity": report.achieved_sparsity,
        "tensors": [
            {"name": tensor.name, "shape": tensor.shape, "dtype": tensor.dtype.value, "bytes": tensor.bytes}
            for tensor in report.tensors
        ],
    }
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2))
    return 0


def command_compile(args: argparse.Namespace) -> int:
    graph = Graph(name="myriad_demo")
    graph.add_input("features", (1, 128))
    graph.add_layer(lower_dense_layer("encoder", "features", "encoded", 1, 128, 256))
    graph.add_layer(lower_dense_layer("projection", "encoded", "projected", 1, 256, 128))
    graph.add_layer(lower_dense_layer("head", "projected", "logits", 1, 128, 10))
    artifact = compile_graph(graph, bytes_per_element=args.bytes_per_element, parallelism=args.parallelism)
    Path(args.output).write_text(artifact.source, encoding="utf-8")
    print(artifact.source, end="")
    return 0


def command_sweep(args: argparse.Namespace) -> int:
    results = sweep(
        demo_tensors(),
        demo_layers(),
        [DataType.INT8, DataType.INT4, DataType.INT2],
        [0.0, 0.5, 0.8],
        [args.hardware],
    )
    write_json(results, args.output)
    print(markdown_table(results))
    return 0


def command_roofline(args: argparse.Namespace) -> int:
    hardware = reference_hardware(args.hardware)
    points = roofline_points(demo_layers(), hardware, parameter_bytes=args.bytes_per_element, activation_bytes=args.bytes_per_element)
    payload = {"hardware": hardware.name, "points": points}
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2))
    return 0


def command_cost(args: argparse.Namespace) -> int:
    hardware = reference_hardware(args.hardware)
    from myriad.analysis.estimator import estimate_design

    estimate = estimate_design(demo_layers(), hardware, parameter_bytes=args.bytes_per_element, activation_bytes=args.bytes_per_element)
    payload = silicon_cost(estimate, wafer_diameter_mm=args.wafer_diameter, die_yield=args.yield_rate, wafer_cost_usd=args.wafer_cost)
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="myriad", description="Silicon model compression and hardware co-design lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress = subparsers.add_parser("compress")
    compress.add_argument("--dtype", choices=[dtype.value for dtype in DataType if dtype is not DataType.FP32], default="int8")
    compress.add_argument("--sparsity", type=float, default=0.5)
    compress.add_argument("--percentile", type=float, default=99.9)
    compress.add_argument("--block-rows", type=int)
    compress.add_argument("--block-cols", type=int)
    compress.add_argument("--output", default="artifacts/compression.json")
    compress.set_defaults(handler=command_compress)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--bytes-per-element", type=int, default=1)
    compile_parser.add_argument("--parallelism", type=int, default=4)
    compile_parser.add_argument("--output", default="artifacts/myriad.ir")
    compile_parser.set_defaults(handler=command_compile)

    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--hardware", choices=["edge-npu", "datacenter-npu"], default="edge-npu")
    sweep_parser.add_argument("--output", default="artifacts/sweep.json")
    sweep_parser.set_defaults(handler=command_sweep)

    roofline_parser = subparsers.add_parser("roofline")
    roofline_parser.add_argument("--hardware", choices=["edge-npu", "datacenter-npu"], default="edge-npu")
    roofline_parser.add_argument("--bytes-per-element", type=int, default=1)
    roofline_parser.add_argument("--output", default="artifacts/roofline.json")
    roofline_parser.set_defaults(handler=command_roofline)

    cost_parser = subparsers.add_parser("cost")
    cost_parser.add_argument("--hardware", choices=["edge-npu", "datacenter-npu"], default="edge-npu")
    cost_parser.add_argument("--bytes-per-element", type=int, default=1)
    cost_parser.add_argument("--wafer-diameter", type=float, default=300.0)
    cost_parser.add_argument("--yield-rate", type=float, default=0.9)
    cost_parser.add_argument("--wafer-cost", type=float, default=17000.0)
    cost_parser.add_argument("--output", default="artifacts/silicon_cost.json")
    cost_parser.set_defaults(handler=command_cost)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
