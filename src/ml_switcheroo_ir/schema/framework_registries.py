"""Framework operator registries and universal Operation Definition Language (ODL) catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ml_switcheroo_ir.schema.base import (
    CustomAttributeSchema,
    CustomOpSchema,
    OpSchema,
    load_op_schema_catalog,
)
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

    def _record_dialects(op_data: Mapping[str, Any]) -> None:
        """Record dialect mappings from an ODL operator entry if present.

        Args:
            op_data (Mapping[str, Any]): Raw operator definition dictionary.
        """
        if "dialects" in op_data and isinstance(op_data["dialects"], dict):
            ODL_DIALECT_MAPPINGS[str(op_data["name"])] = dict(op_data["dialects"])

    return load_op_schema_catalog(
        filename,
        domain_default=domain_default,
        json_path=json_path,
        find_file_fn=find_schema_file,
        on_entry=_record_dialects,
    )


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
