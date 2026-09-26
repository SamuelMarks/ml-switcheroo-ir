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
    ATEN_REGISTRY.update(_load_framework_schema_file("aten_ops.json", "aten"))
    ARRAY_API_REGISTRY.update(
        _load_framework_schema_file("array_api_ops.json", "array_api")
    )
    ODL_CATALOG.update(_load_framework_schema_file("odl_catalog.json", "odl"))


_initialize_framework_registries()


def get_abstract_op(name: str) -> OpSchema | None:
    """Retrieve canonical OpSchema definition for an abstract mathematical operation.

    Args:
        name (str): Abstract operation name (e.g. 'add', 'matmul', 'relu', 'layer_norm').

    Returns:
        OpSchema | None: Canonical OpSchema if defined in ODL catalog, else None.
    """
    return ODL_CATALOG.get(name)
