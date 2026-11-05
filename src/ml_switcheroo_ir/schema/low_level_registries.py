"""Low-level hardware and shading language dialect registries for ml_switcheroo_ir."""

from __future__ import annotations

from pathlib import Path

from ml_switcheroo_ir.schema.base import (
    REGISTRY_LOCK,
    CustomOpSchema,
    OpSchema,
    load_op_schema_catalog,
)
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


def _load_schema_file(filename: str, domain_default: str, json_path: Path | str | None = None) -> dict[str, OpSchema]:
    """Load operator schemas from a bundled JSON file.

    Args:
        filename (str): Name of the JSON file in schema directory.
        domain_default (str): Fallback domain identifier.
        json_path (Optional[Union[Path, str]]): Explicit path override for schema JSON.

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
    """Initialize all low-level hardware and shader registries."""
    specs = (
        (PTX_REGISTRY, "ptx_ops.json", "nvidia_ptx", _CANONICAL_PTX_OPS),
        (METAL_REGISTRY, "metal_ops.json", "metal_msl", _CANONICAL_METAL_OPS),
        (WASM_REGISTRY, "wasm_ops.json", "wasm_simd", _CANONICAL_WASM_OPS),
        (WEBGL_REGISTRY, "webgl_ops.json", "webgl", _CANONICAL_WEBGL_OPS),
        (WGSL_REGISTRY, "wgsl_ops.json", "wgsl", _CANONICAL_WGSL_OPS),
    )
    with REGISTRY_LOCK:
        for reg, filename, domain, canonical_ops in specs:
            loaded = _load_schema_file(filename, domain)
            if not loaded:
                for s in canonical_ops:
                    loaded[s.name] = s.to_op_schema()
            reg.update(loaded)


_initialize_low_level_registries()
