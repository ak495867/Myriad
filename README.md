# Myriad

**Myriad** is a Python research toolkit for silicon model compression and hardware co-design. It connects four views of the same deployment problem: numerical compression algorithms, compiler representations and schedules, energy and performance estimation, and first-order silicon area and wafer-cost projections.

The repository is designed for reproducible design-space exploration rather than production silicon sign-off. Its estimators are intentionally explicit and inspectable so that researchers can replace assumptions with measurements from a target process, accelerator RTL, simulator, or technology library.

## Research scope

| Area | Included capability |
|---|---|
| Compression | Affine INT8, INT4, INT2, and binary quantization with calibration metrics |
| Sparsity | Unstructured magnitude pruning and two-dimensional block pruning |
| Training objective | Temperature-scaled knowledge-distillation loss |
| Compiler | Graph IR, dense and matmul lowering, topological validation, liveness memory planning, parallel schedules, and textual artifacts |
| Hardware | Edge and datacenter NPU reference presets |
| Energy | MAC, SRAM-byte, and DRAM-byte energy accounting |
| Performance | Compute-versus-memory roofline classification and latency estimation |
| Silicon | MAC and SRAM area model with wafer-level good-die and cost-per-die projection |
| Experimentation | Deterministic compression sweeps and JSON or Markdown reports |

## Installation

Myriad targets Python 3.11 or newer. The standard editable installation is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The runtime dependencies are NumPy and Pydantic. The current implementation uses standard-library dataclasses for the core model objects and keeps the numerical kernels NumPy-based.

## Command-line usage

The CLI exposes independent workflows:

```bash
myriad compress --dtype int8 --sparsity 0.5
myriad compile --bytes-per-element 1 --parallelism 4
myriad sweep --hardware edge-npu
myriad roofline --hardware datacenter-npu
myriad cost --hardware edge-npu
```

Each command writes a machine-readable artifact under `artifacts/` by default. The `sweep` command also prints a compact Markdown table suitable for experiment logs.

## Python usage

```python
import numpy as np

from myriad.analysis import estimate_design, silicon_cost
from myriad.compression import compress_tensors
from myriad.compiler import Graph, compile_graph, lower_dense_layer
from myriad.hardware import edge_npu
from myriad.models import CompressionConfig, DataType, Tensor

weights = [Tensor("projection.weight", np.random.default_rng(4).normal(size=(128, 64)).astype(np.float32))]
compressed = compress_tensors(
    weights,
    CompressionConfig(target_dtype=DataType.INT4, global_sparsity=0.5),
)

graph = Graph("projection")
graph.add_input("features", (1, 64))
graph.add_layer(lower_dense_layer("projection", "features", "output", 1, 64, 128))
design = estimate_design(graph.layers, edge_npu(), parameter_bytes=1, activation_bytes=1)
artifact = compile_graph(graph, bytes_per_element=1)
cost = silicon_cost(design)
```

## Repository layout

```text
src/myriad/
  analysis/       Energy, latency, area, roofline, and silicon-cost estimation
  compression/   Quantization, pruning, distillation, and compression pipelines
  compiler/      Graph IR, scheduling, memory planning, and artifact generation
  experiments/   Sweeps and report serialization
  hardware/      Reference hardware specifications
  cli.py         Command-line interface
  models.py      Shared immutable data models
examples/        End-to-end demonstration scripts
configs/         Reproducible experiment configurations
tests/           Unit and integration tests
```

## Design assumptions

Myriad’s hardware numbers are reference assumptions, not claims about a particular commercial chip. Latency uses the larger of compute time and external-memory transfer time. Energy adds MAC, SRAM traffic, and DRAM traffic. Area assumes a fixed MAC and SRAM area for a selected hardware preset. Silicon cost uses a geometric wafer estimate, a usable-die factor, a yield factor, and wafer cost. These assumptions are exposed in code so that experiments remain auditable.

## Quality checks

Run the local checks with:

```bash
pytest
ruff check .
python -m compileall -q src tests examples
```

The repository intentionally contains **no source-code comments**. Explanatory material lives in this README and in the public API naming and structure.

## License

Myriad is released under the Apache License 2.0. See `LICENSE`.
