"""Tests for universal Operation Definition Language (ODL) catalog and abstract op resolution."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalNode, get_abstract_op
from ml_switcheroo_ir.schema.framework_registries import (
    ODL_CATALOG,
    ODL_DIALECT_MAPPINGS,
)
from ml_switcheroo_ir.validator import ValidationLevel, Validator


def test_odl_catalog_loaded() -> None:
    """Verify that ODL catalog contains core abstract mathematical operations."""
    assert len(ODL_CATALOG) > 10

    # Key operations
    for op in [
        "add",
        "subtract",
        "multiply",
        "divide",
        "matmul",
        "relu",
        "gelu",
        "silu",
        "softmax",
        "layer_norm",
        "rms_norm",
        "conv2d",
        "scaled_dot_product_attention",
    ]:
        schema = get_abstract_op(op)
        assert schema is not None
        assert schema.name == op
        assert schema.domain == "odl"

    assert get_abstract_op("unknown_math_op") is None


def test_odl_dialect_mappings() -> None:
    """Verify that ODL operations provide multi-dialect mappings to frameworks."""
    matmul_dialects = ODL_DIALECT_MAPPINGS.get("matmul", {})
    assert "torch" in matmul_dialects
    assert "jax" in matmul_dialects
    assert "tensorflow" in matmul_dialects
    assert "keras3" in matmul_dialects
    assert "mlx" in matmul_dialects
    assert "onnx" in matmul_dialects
    assert "stablehlo" in matmul_dialects

    assert matmul_dialects["torch"] == "torch.matmul"
    assert matmul_dialects["onnx"] == "MatMul"
    assert matmul_dialects["stablehlo"] == "stablehlo.dot_general"


def test_validator_odl_domain_validation() -> None:
    """Verify that Validator._get_schema and validate_kind accept valid and reject unknown ODL ops."""
    v = Validator(level=ValidationLevel.STRICT)

    # Valid ODL node
    n_odl = LogicalNode(
        id="odl1",
        op_type="layer_norm",
        domain="odl",
        inputs=["x", "w", "b"],
        outputs=["out"],
    )
    schema = v._get_schema(n_odl)
    assert schema is not None
    assert schema.name == "layer_norm"
    assert v.validate_kind(n_odl) == []

    # Valid abstract alias domain
    n_abs = LogicalNode(
        id="abs1",
        op_type="rms_norm",
        domain="abstract",
        inputs=["x", "w"],
        outputs=["out"],
    )
    schema_abs = v._get_schema(n_abs)
    assert schema_abs is not None
    assert schema_abs.name == "rms_norm"
    assert v.validate_kind(n_abs) == []

    # Invalid ODL node
    n_bad = LogicalNode(
        id="bad1",
        op_type="unknown_math_op",
        domain="odl",
        inputs=["x"],
        outputs=["out"],
    )
    errs = v.validate_kind(n_bad)
    assert len(errs) == 1
    assert "not found in domain 'odl'" in errs[0].message
