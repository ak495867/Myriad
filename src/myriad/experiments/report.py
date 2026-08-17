from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(results: list[dict[str, object]]) -> str:
    rows = [
        "| Hardware | Dtype | Sparsity | Size reduction | Latency (us) | Energy (uJ) | Area (mm2) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        compression = result["compression"]
        design = result["design"]
        rows.append(
            "| {hardware} | {dtype} | {sparsity:.2f} | {reduction:.3f} | {latency:.3f} | {energy:.3f} | {area:.3f} |".format(
                hardware=design["hardware"],
                dtype=compression["target_dtype"],
                sparsity=compression["achieved_sparsity"],
                reduction=compression["size_reduction"],
                latency=design["latency_us"],
                energy=design["energy_uj"],
                area=design["area_mm2"],
            )
        )
    return "\n".join(rows) + "\n"
