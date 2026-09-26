"""Low-level hardware and shading language dialect registries for ml_switcheroo_ir."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml_switcheroo_ir.schema.custom_ops import CustomAttributeSchema, CustomOpSchema
from ml_switcheroo_ir.schema.onnx_registry import OpSchema
from ml_switcheroo_ir.snapshots import find_schema_file

PTX_REGISTRY: dict[str, OpSchema] = {}
METAL_REGISTRY: dict[str, OpSchema] = {}
WASM_REGISTRY: dict[str, OpSchema] = {}
WEBGL_REGISTRY: dict[str, OpSchema] = {}
WGSL_REGISTRY: dict[str, OpSchema] = {}


def _load_schema_file(
    filename: str, domain_default: str, json_path: Path | str | None = None
) -> dict[str, OpSchema]:
    """Load operator schemas from a bundled JSON file.

    Args:
        filename (str): Name of the JSON file in schema directory.
        domain_default (str): Fallback domain identifier.
        json_path (Optional[Union[Path, str]]): Explicit path override for schema JSON.

    Returns:
        dict[str, OpSchema]: Mapping from operator names to OpSchema definitions.
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
    return registry


def _initialize_low_level_registries() -> None:
    """Initialize all low-level hardware and shader registries."""
    PTX_REGISTRY.update(_load_schema_file("ptx_ops.json", "nvidia_ptx"))
    METAL_REGISTRY.update(_load_schema_file("metal_ops.json", "metal_msl"))
    WASM_REGISTRY.update(_load_schema_file("wasm_ops.json", "wasm_simd"))
    WEBGL_REGISTRY.update(_load_schema_file("webgl_ops.json", "webgl"))
    WGSL_REGISTRY.update(_load_schema_file("wgsl_ops.json", "wgsl"))


_initialize_low_level_registries()
