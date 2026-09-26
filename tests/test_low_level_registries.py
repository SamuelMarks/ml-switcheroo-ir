"""Tests for low-level hardware and shading language dialect schemas and registries."""

from __future__ import annotations

import pytest

from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_ir.schema.low_level_registries import (
    METAL_REGISTRY,
    PTX_REGISTRY,
    WASM_REGISTRY,
    WEBGL_REGISTRY,
    WGSL_REGISTRY,
)
from ml_switcheroo_ir.schema.rdna_registry import RDNA_REGISTRY
from ml_switcheroo_ir.schema.sass_registry import SASS_REGISTRY
from ml_switcheroo_ir.validator import ValidationLevel, Validator


def test_low_level_registries_loaded() -> None:
    """Verify that all low-level dialect registries are loaded and populated with schemas."""
    assert len(PTX_REGISTRY) > 0
    assert len(METAL_REGISTRY) > 0
    assert len(WASM_REGISTRY) > 0
    assert len(WEBGL_REGISTRY) > 0
    assert len(WGSL_REGISTRY) > 0
    assert len(RDNA_REGISTRY) > 0
    assert len(SASS_REGISTRY) > 0

    # Key operations presence
    assert "add" in PTX_REGISTRY
    assert "simdgroup_multiply_accumulate" in METAL_REGISTRY
    assert "f32x4.add" in WASM_REGISTRY
    assert "texelFetch" in WEBGL_REGISTRY
    assert "storageStore" in WGSL_REGISTRY
    assert "V_DUAL_FMAC_F32" in RDNA_REGISTRY
    assert "FFMA" in SASS_REGISTRY


def test_validator_low_level_domain_lookups() -> None:
    """Verify that Validator._get_schema resolves schemas for all low-level hardware and shader domains."""
    v = Validator(level=ValidationLevel.STRICT)

    # PTX
    n_ptx = LogicalNode(
        id="ptx1",
        op_type="add",
        domain="nvidia_ptx",
        inputs=["a", "b"],
        outputs=["d"],
    )
    schema_ptx = v._get_schema(n_ptx)
    assert schema_ptx is not None
    assert schema_ptx.name == "add"

    # PTX alias domain
    n_ptx_alias = LogicalNode(
        id="ptx2",
        op_type="add",
        domain="ptx",
        inputs=["a", "b"],
        outputs=["d"],
    )
    assert v._get_schema(n_ptx_alias) is not None

    # Metal MSL
    n_metal = LogicalNode(
        id="m1",
        op_type="simdgroup_multiply_accumulate",
        domain="metal_msl",
        inputs=["dest", "a", "b", "c"],
        outputs=["res"],
    )
    schema_metal = v._get_schema(n_metal)
    assert schema_metal is not None
    assert schema_metal.name == "simdgroup_multiply_accumulate"

    # Metal alias domain
    n_metal_alias = LogicalNode(
        id="m2",
        op_type="simdgroup_multiply_accumulate",
        domain="metal",
        inputs=["dest", "a", "b", "c"],
        outputs=["res"],
    )
    assert v._get_schema(n_metal_alias) is not None

    # WASM SIMD
    n_wasm = LogicalNode(
        id="w1",
        op_type="f32x4.add",
        domain="wasm_simd",
        inputs=["lhs", "rhs"],
        outputs=["val"],
    )
    schema_wasm = v._get_schema(n_wasm)
    assert schema_wasm is not None
    assert schema_wasm.name == "f32x4.add"

    # WASM alias domain
    n_wasm_alias = LogicalNode(
        id="w2",
        op_type="f32x4.add",
        domain="wasm",
        inputs=["lhs", "rhs"],
        outputs=["val"],
    )
    assert v._get_schema(n_wasm_alias) is not None

    # WebGL
    n_webgl = LogicalNode(
        id="gl1",
        op_type="texelFetch",
        domain="webgl",
        inputs=["sampler", "P", "lod"],
        outputs=["rgba"],
    )
    schema_webgl = v._get_schema(n_webgl)
    assert schema_webgl is not None
    assert schema_webgl.name == "texelFetch"

    # WGSL
    n_wgsl = LogicalNode(
        id="wgsl1",
        op_type="storageStore",
        domain="wgsl",
        inputs=["buffer", "index", "value"],
    )
    schema_wgsl = v._get_schema(n_wgsl)
    assert schema_wgsl is not None
    assert schema_wgsl.name == "storageStore"

    # RDNA & SASS
    n_rdna = LogicalNode(
        id="rd1",
        op_type="V_DUAL_FMAC_F32",
        domain="amd_rdna",
        inputs=["src0", "src1"],
        outputs=["dst"],
    )
    assert v._get_schema(n_rdna) is not None

    n_rdna_alias = LogicalNode(
        id="rd2",
        op_type="V_DUAL_FMAC_F32",
        domain="rdna",
        inputs=["src0", "src1"],
        outputs=["dst"],
    )
    assert v._get_schema(n_rdna_alias) is not None

    n_sass = LogicalNode(
        id="sa1",
        op_type="FFMA",
        domain="nvidia_sass",
        inputs=["src0", "src1"],
        outputs=["dst"],
    )
    assert v._get_schema(n_sass) is not None

    n_sass_alias = LogicalNode(
        id="sa2",
        op_type="FFMA",
        domain="sass",
        inputs=["src0", "src1"],
        outputs=["dst"],
    )
    assert v._get_schema(n_sass_alias) is not None


def test_validate_kind_unknown_ops_across_low_level_domains() -> None:
    """Verify that validate_kind validates valid ops and raises errors for unknown ops across low-level domains."""
    v = Validator(level=ValidationLevel.STRICT)

    domains_and_valid_ops = [
        ("amd_rdna", "V_DUAL_FMAC_F32"),
        ("nvidia_sass", "FFMA"),
        ("webgpu_wgsl", "storageStore"),
        ("nvidia_ptx", "add"),
        ("metal_msl", "simdgroup_multiply_accumulate"),
        ("wasm_simd", "f32x4.add"),
        ("webgl", "texelFetch"),
    ]
    for dom, valid_op in domains_and_valid_ops:
        good_node = LogicalNode(
            id=f"good_{dom}",
            op_type=valid_op,
            domain=dom,
            shape_metadata=(1,),
        )
        assert v.validate_kind(good_node) == []

        bad_node = LogicalNode(
            id=f"bad_{dom}",
            op_type="UNKNOWN_OP_NAME",
            domain=dom,
            shape_metadata=(1,),
        )
        errors = v.validate_kind(bad_node)
        assert len(errors) == 1
        assert "not found in domain" in errors[0].message


def test_missing_schema_json_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify graceful handling when bundled low-level JSON schema files do not exist."""
    from pathlib import Path

    import ml_switcheroo_ir.schema.low_level_registries as ll_mod
    import ml_switcheroo_ir.schema.rdna_registry as rdna_mod
    import ml_switcheroo_ir.schema.sass_registry as sass_mod

    monkeypatch.setattr(Path, "exists", lambda self: False)

    # Calling loaders with nonexistent path
    rdna_mod._load_rdna_schemas()
    sass_mod._load_sass_schemas()
    assert ll_mod._load_schema_file("missing.json", "dummy") == {}
