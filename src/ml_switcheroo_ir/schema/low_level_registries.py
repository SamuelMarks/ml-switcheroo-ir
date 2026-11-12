"""Low-level hardware and shading language dialect registries for ml_switcheroo_ir."""

from __future__ import annotations

from pathlib import Path

from ml_switcheroo_ir.schema.base import (
    REGISTRY_LOCK,
    OpSchema,
    SchemaRegistry,
    load_op_schema_catalog,
)
from ml_switcheroo_ir.schema.ghost import WGSL_PRIMITIVE_SIGNATURES
from ml_switcheroo_ir.snapshots import find_schema_file

_CANONICAL_PTX_OPS: list[OpSchema] = [
    OpSchema(
        name="add",
        domain="nvidia_ptx",
        inputs=["a", "b"],
        outputs=["d"],
    ),
    OpSchema(
        name="sub",
        domain="nvidia_ptx",
        inputs=["a", "b"],
        outputs=["d"],
    ),
    OpSchema(
        name="mul",
        domain="nvidia_ptx",
        inputs=["a", "b"],
        outputs=["d"],
    ),
    OpSchema(
        name="fma",
        domain="nvidia_ptx",
        inputs=["a", "b", "c"],
        outputs=["d"],
    ),
]

_CANONICAL_METAL_OPS: list[OpSchema] = [
    OpSchema(
        name="simdgroup_multiply_accumulate",
        domain="metal_msl",
        inputs=["dest", "a", "b", "c"],
        outputs=["res"],
    ),
]

_CANONICAL_WASM_OPS: list[OpSchema] = [
    OpSchema(
        name="f32x4.add",
        domain="wasm_simd",
        inputs=["lhs", "rhs"],
        outputs=["val"],
    ),
    OpSchema(
        name="f32x4.mul",
        domain="wasm_simd",
        inputs=["lhs", "rhs"],
        outputs=["val"],
    ),
]

_CANONICAL_WEBGL_OPS: list[OpSchema] = [
    OpSchema(
        name="texelFetch",
        domain="webgl",
        inputs=["sampler", "P", "lod"],
        outputs=["rgba"],
    ),
]

_CANONICAL_WGSL_OPS: list[OpSchema] = [
    OpSchema(
        name=op_name,
        domain="wgsl",
        inputs=list(sig.inputs),
        outputs=list(sig.outputs),
    )
    for op_name, sig in WGSL_PRIMITIVE_SIGNATURES.items()
]


def _load_schema_file(filename: str, domain_default: str, json_path: Path | str | None = None) -> dict[str, OpSchema]:
    """Load operator schemas from a bundled or snapshot JSON file.

    Args:
        filename (str): Name of the JSON file in schema/catalogs directory.
        domain_default (str): Fallback domain identifier.
        json_path (Path | str | None): Explicit path override for schema JSON.

    Returns:
        dict[str, OpSchema]: Mapping from operator names to OpSchema definitions.
    """
    return load_op_schema_catalog(
        filename,
        domain_default=domain_default,
        json_path=json_path,
        find_file_fn=find_schema_file,
    )


def _initialize_low_level_registries() -> None:
    """Initialize all low-level hardware and shader registries under REGISTRY_LOCK."""
    specs = (
        (PTX_REGISTRY, "ptx_ops.json", "nvidia_ptx", _CANONICAL_PTX_OPS),
        (METAL_REGISTRY, "metal_ops.json", "metal_msl", _CANONICAL_METAL_OPS),
        (WASM_REGISTRY, "wasm_ops.json", "wasm_simd", _CANONICAL_WASM_OPS),
        (WEBGL_REGISTRY, "webgl_ops.json", "webgl", _CANONICAL_WEBGL_OPS),
        (WGSL_REGISTRY, "wgsl_ops.json", "wgsl", _CANONICAL_WGSL_OPS),
    )
    with REGISTRY_LOCK:
        for reg, _filename, _domain, _canonical_ops in specs:
            reg._loaded = True
        for reg, filename, domain, canonical_ops in specs:
            loaded = _load_schema_file(filename, domain)
            if not loaded:
                for schema in canonical_ops:
                    loaded[schema.name] = schema
            reg.update(loaded)


PTX_REGISTRY: SchemaRegistry = SchemaRegistry("PTX_REGISTRY", loader=_initialize_low_level_registries)
METAL_REGISTRY: SchemaRegistry = SchemaRegistry("METAL_REGISTRY", loader=_initialize_low_level_registries)
WASM_REGISTRY: SchemaRegistry = SchemaRegistry("WASM_REGISTRY", loader=_initialize_low_level_registries)
WEBGL_REGISTRY: SchemaRegistry = SchemaRegistry("WEBGL_REGISTRY", loader=_initialize_low_level_registries)
WGSL_REGISTRY: SchemaRegistry = SchemaRegistry("WGSL_REGISTRY", loader=_initialize_low_level_registries)

__all__ = [
    "METAL_REGISTRY",
    "PTX_REGISTRY",
    "WASM_REGISTRY",
    "WEBGL_REGISTRY",
    "WGSL_REGISTRY",
]
