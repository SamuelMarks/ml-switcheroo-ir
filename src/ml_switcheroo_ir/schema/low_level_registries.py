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

_CANONICAL_PTX_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="add",
        domain="nvidia_ptx",
        attributes=[],
        inputs=["a", "b"],
        outputs=["d"],
    ),
    CustomOpSchema(
        name="sub",
        domain="nvidia_ptx",
        attributes=[],
        inputs=["a", "b"],
        outputs=["d"],
    ),
    CustomOpSchema(
        name="mul",
        domain="nvidia_ptx",
        attributes=[],
        inputs=["a", "b"],
        outputs=["d"],
    ),
    CustomOpSchema(
        name="fma",
        domain="nvidia_ptx",
        attributes=[],
        inputs=["a", "b", "c"],
        outputs=["d"],
    ),
]

_CANONICAL_METAL_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="simdgroup_multiply_accumulate",
        domain="metal_msl",
        attributes=[],
        inputs=["dest", "a", "b", "c"],
        outputs=["res"],
    ),
]

_CANONICAL_WASM_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="f32x4.add",
        domain="wasm_simd",
        attributes=[],
        inputs=["lhs", "rhs"],
        outputs=["val"],
    ),
    CustomOpSchema(
        name="f32x4.mul",
        domain="wasm_simd",
        attributes=[],
        inputs=["lhs", "rhs"],
        outputs=["val"],
    ),
]

_CANONICAL_WEBGL_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="texelFetch",
        domain="webgl",
        attributes=[],
        inputs=["sampler", "P", "lod"],
        outputs=["rgba"],
    ),
]

_CANONICAL_WGSL_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="storageStore",
        domain="wgsl",
        attributes=[],
        inputs=["buffer", "index", "value"],
        outputs=[],
    ),
    CustomOpSchema(
        name="workgroupBarrier",
        domain="wgsl",
        attributes=[],
        inputs=[],
        outputs=[],
    ),
    CustomOpSchema(
        name="storageBarrier",
        domain="wgsl",
        attributes=[],
        inputs=[],
        outputs=[],
    ),
    CustomOpSchema(
        name="atomicAdd",
        domain="wgsl",
        attributes=[],
        inputs=["atomic_ptr", "value"],
        outputs=["old_value"],
    ),
]


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
    loaded_ptx = _load_schema_file("ptx_ops.json", "nvidia_ptx")
    if not loaded_ptx:
        for s in _CANONICAL_PTX_OPS:
            loaded_ptx[s.name] = s.to_op_schema()
    PTX_REGISTRY.update(loaded_ptx)

    loaded_metal = _load_schema_file("metal_ops.json", "metal_msl")
    if not loaded_metal:
        for s in _CANONICAL_METAL_OPS:
            loaded_metal[s.name] = s.to_op_schema()
    METAL_REGISTRY.update(loaded_metal)

    loaded_wasm = _load_schema_file("wasm_ops.json", "wasm_simd")
    if not loaded_wasm:
        for s in _CANONICAL_WASM_OPS:
            loaded_wasm[s.name] = s.to_op_schema()
    WASM_REGISTRY.update(loaded_wasm)

    loaded_webgl = _load_schema_file("webgl_ops.json", "webgl")
    if not loaded_webgl:
        for s in _CANONICAL_WEBGL_OPS:
            loaded_webgl[s.name] = s.to_op_schema()
    WEBGL_REGISTRY.update(loaded_webgl)

    loaded_wgsl = _load_schema_file("wgsl_ops.json", "wgsl")
    if not loaded_wgsl:
        for s in _CANONICAL_WGSL_OPS:
            loaded_wgsl[s.name] = s.to_op_schema()
    WGSL_REGISTRY.update(loaded_wgsl)


_initialize_low_level_registries()
