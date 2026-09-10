from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from myriad.models import DataType, LayerSpec


@dataclass(frozen=True)
class Value:
    name: str
    shape: tuple[int, ...]
    dtype: DataType
    producer: str | None = None

    @property
    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


@dataclass
class Graph:
    name: str
    layers: list[LayerSpec] = field(default_factory=list)
    values: dict[str, Value] = field(default_factory=dict)

    def add_input(
        self, name: str, shape: tuple[int, ...], dtype: DataType = DataType.FP32
    ) -> Value:
        if name in self.values:
            raise ValueError(f"value already exists: {name}")
        value = Value(name=name, shape=shape, dtype=dtype)
        self.values[name] = value
        return value

    def add_layer(
        self, layer: LayerSpec, output_dtype: DataType = DataType.FP32
    ) -> Value:
        if layer.output_name in self.values:
            raise ValueError(f"output already exists: {layer.output_name}")
        missing = [name for name in layer.input_names if name not in self.values]
        if missing:
            raise ValueError(f"layer {layer.name} has missing inputs: {missing}")
        self.layers.append(layer)
        value = Value(
            name=layer.output_name,
            shape=layer.shape,
            dtype=output_dtype,
            producer=layer.name,
        )
        self.values[layer.output_name] = value
        return value

    def validate(self) -> None:
        known = set()
        for value in self.values.values():
            if value.producer is None:
                known.add(value.name)
        for layer in self.layers:
            if not set(layer.input_names).issubset(known):
                raise ValueError(
                    f"graph is not topologically ordered at layer {layer.name}"
                )
            if layer.output_name in known:
                raise ValueError(f"duplicate graph output: {layer.output_name}")
            known.add(layer.output_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": [
                {"name": value.name, "shape": value.shape, "dtype": value.dtype.value}
                for value in self.values.values()
                if value.producer is None
            ],
            "layers": [
                {
                    "name": layer.name,
                    "op": layer.op,
                    "inputs": layer.input_names,
                    "output": layer.output_name,
                    "shape": layer.shape,
                    "weight_shape": layer.weight_shape,
                    "macs": layer.macs,
                    "attributes": layer.attributes,
                }
                for layer in self.layers
            ],
        }


def lower_dense_layer(
    name: str,
    input_name: str,
    output_name: str,
    batch: int,
    input_features: int,
    output_features: int,
) -> LayerSpec:
    return LayerSpec(
        name=name,
        op="dense",
        input_names=(input_name,),
        output_name=output_name,
        shape=(batch, output_features),
        weight_shape=(output_features, input_features),
        macs=batch * input_features * output_features,
        attributes={
            "input_features": input_features,
            "output_features": output_features,
        },
    )


def lower_matmul_layer(
    name: str,
    left: str,
    right: str,
    output_name: str,
    batch: int,
    rows: int,
    inner: int,
    cols: int,
) -> LayerSpec:
    return LayerSpec(
        name=name,
        op="matmul",
        input_names=(left, right),
        output_name=output_name,
        shape=(batch, rows, cols),
        weight_shape=None,
        macs=batch * rows * inner * cols,
        attributes={"rows": rows, "inner": inner, "cols": cols},
    )
