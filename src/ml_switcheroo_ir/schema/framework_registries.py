"""Framework operator registries and universal Operation Definition Language (ODL) catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml_switcheroo_ir.schema.custom_ops import CustomAttributeSchema, CustomOpSchema
from ml_switcheroo_ir.schema.onnx_registry import OpSchema
from ml_switcheroo_ir.snapshots import find_schema_file

ATEN_REGISTRY: dict[str, OpSchema] = {}
ARRAY_API_REGISTRY: dict[str, OpSchema] = {}
ODL_CATALOG: dict[str, OpSchema] = {}
ODL_DIALECT_MAPPINGS: dict[str, dict[str, str]] = {}

_CANONICAL_ODL_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="add",
        domain="odl",
        attributes=[],
        inputs=["a", "b"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="subtract",
        domain="odl",
        attributes=[],
        inputs=["a", "b"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="multiply",
        domain="odl",
        attributes=[],
        inputs=["a", "b"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="divide",
        domain="odl",
        attributes=[],
        inputs=["a", "b"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="matmul",
        domain="odl",
        attributes=[],
        inputs=["a", "b"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="relu",
        domain="odl",
        attributes=[],
        inputs=["x"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="gelu",
        domain="odl",
        attributes=[],
        inputs=["x"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="silu",
        domain="odl",
        attributes=[],
        inputs=["x"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="softmax",
        domain="odl",
        attributes=[
            CustomAttributeSchema(
                name="dim",
                type="int",
                required=False,
                default=-1,
            )
        ],
        inputs=["x"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="layer_norm",
        domain="odl",
        attributes=[
            CustomAttributeSchema(
                name="eps",
                type="float",
                required=False,
                default=1e-5,
            )
        ],
        inputs=["x", "scale", "bias"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="rms_norm",
        domain="odl",
        attributes=[
            CustomAttributeSchema(
                name="eps",
                type="float",
                required=False,
                default=1e-6,
            )
        ],
        inputs=["x", "scale"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="conv2d",
        domain="odl",
        attributes=[],
        inputs=["x", "weight"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="scaled_dot_product_attention",
        domain="odl",
        attributes=[],
        inputs=["query", "key", "value"],
        outputs=["result"],
    ),
]

_CANONICAL_ODL_DIALECT_MAPPINGS: dict[str, dict[str, str]] = {
    "matmul": {
        "torch": "torch.matmul",
        "jax": "jax.numpy.matmul",
        "tensorflow": "tf.linalg.matmul",
        "keras3": "keras.ops.matmul",
        "mlx": "mlx.core.matmul",
        "onnx": "MatMul",
        "stablehlo": "stablehlo.dot_general",
    },
    "add": {
        "torch": "torch.add",
        "jax": "jax.numpy.add",
        "tensorflow": "tf.math.add",
        "keras3": "keras.ops.add",
        "mlx": "mlx.core.add",
        "onnx": "Add",
        "stablehlo": "stablehlo.add",
    },
}

_CANONICAL_ATEN_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="add",
        domain="aten",
        attributes=[
            CustomAttributeSchema(name="alpha", type="float", required=False),
        ],
        inputs=["self", "other"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="matmul",
        domain="aten",
        attributes=[],
        inputs=["self", "other"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="relu",
        domain="aten",
        attributes=[],
        inputs=["self"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="layer_norm",
        domain="aten",
        attributes=[],
        inputs=["input", "normalized_shape", "weight", "bias"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="scaled_dot_product_attention",
        domain="aten",
        attributes=[],
        inputs=["query", "key", "value"],
        outputs=["out"],
    ),
]

_CANONICAL_ARRAY_API_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="abs",
        domain="array_api",
        attributes=[],
        inputs=["x"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="add",
        domain="array_api",
        attributes=[],
        inputs=["x1", "x2"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="matmul",
        domain="array_api",
        attributes=[],
        inputs=["x1", "x2"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="mean",
        domain="array_api",
        attributes=[],
        inputs=["x"],
        outputs=["out"],
    ),
    CustomOpSchema(
        name="sum",
        domain="array_api",
        attributes=[
            CustomAttributeSchema(name="axis", type="int", required=False),
            CustomAttributeSchema(name="keepdims", type="bool", required=False),
        ],
        inputs=["x"],
        outputs=["out"],
    ),
]


def _load_framework_schema_file(
    filename: str, domain_default: str, json_path: Path | str | None = None
) -> dict[str, OpSchema]:
    """Load operator schemas from a JSON file into OpSchema mappings.

    Args:
        filename (str): JSON filename in schema directory.
        domain_default (str): Default domain identifier.
        json_path (Optional[Union[Path, str]]): Explicit path override for schema JSON.

    Returns:
        dict[str, OpSchema]: Mapping from operator names to OpSchema instances.
    """
    registry: dict[str, OpSchema] = {}
    target = find_schema_file(filename, override_path=json_path)
    if target and target.exists():
        with open(target, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        for op_data in data.get("ops", []):
            name = str(op_data["name"])
            domain = str(op_data.get("domain", domain_default))
            attrs = [
                CustomAttributeSchema(
                    name=a["name"],
                    type=a.get("type", "str"),
                    required=a.get("required", False),
                    default=a.get("default", None),
                )
                for a in op_data.get("attributes", [])
            ]
            schema = CustomOpSchema(
                name=name,
                domain=domain,
                attributes=attrs,
                inputs=op_data.get("inputs", []),
                outputs=op_data.get("outputs", []),
            )
            registry[schema.name] = schema.to_op_schema()
            if "dialects" in op_data and isinstance(op_data["dialects"], dict):
                ODL_DIALECT_MAPPINGS[name] = dict(op_data["dialects"])
    return registry


def _initialize_framework_registries() -> None:
    """Initialize ATen, Array API, and ODL catalog registries."""
    loaded_aten = _load_framework_schema_file("aten_ops.json", "aten")
    if not loaded_aten:
        for s in _CANONICAL_ATEN_OPS:
            loaded_aten[s.name] = s.to_op_schema()
    ATEN_REGISTRY.update(loaded_aten)

    loaded_array_api = _load_framework_schema_file("array_api_ops.json", "array_api")
    if not loaded_array_api:
        for s in _CANONICAL_ARRAY_API_OPS:
            loaded_array_api[s.name] = s.to_op_schema()
    ARRAY_API_REGISTRY.update(loaded_array_api)

    loaded_odl = _load_framework_schema_file("odl_catalog.json", "odl")
    if not loaded_odl:
        for s in _CANONICAL_ODL_OPS:
            loaded_odl[s.name] = s.to_op_schema()
        ODL_DIALECT_MAPPINGS.update(_CANONICAL_ODL_DIALECT_MAPPINGS)
    ODL_CATALOG.update(loaded_odl)


_initialize_framework_registries()


def get_abstract_op(name: str) -> OpSchema | None:
    """Retrieve canonical OpSchema definition for an abstract mathematical operation.

    Args:
        name (str): Abstract operation name (e.g. 'add', 'matmul', 'relu', 'layer_norm').

    Returns:
        OpSchema | None: Canonical OpSchema if defined in ODL catalog, else None.
    """
    return ODL_CATALOG.get(name)
