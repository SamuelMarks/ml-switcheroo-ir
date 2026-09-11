"""Tests for pre-registered modern custom neural operators."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_ir.schema.custom_ops import CUSTOM_OPS_REGISTRY
from ml_switcheroo_ir.validator import ValidationLevel, Validator


def test_custom_ops_registry_contains_modern_primitives() -> None:
    """Test that all modern primitives are in CUSTOM_OPS_REGISTRY."""
    expected = [
        "RMSNorm",
        "SwiGLU",
        "RoPE",
        "FlashAttention",
        "VisionPatchEmbedding",
    ]
    for name in expected:
        assert name in CUSTOM_OPS_REGISTRY
        assert CUSTOM_OPS_REGISTRY[name].domain == "ml.switcheroo.custom"


def test_rmsnorm_schema_validation_success() -> None:
    """Validate correct RMSNorm node."""
    v = Validator()
    node = LogicalNode(
        id="norm1",
        op_type="RMSNorm",
        domain="ml.switcheroo.custom",
        inputs=["X", "weight"],
        attributes={"eps": 1e-5},
    )
    assert not v.validate_kind(node)
    assert not v.validate_required_attributes(node)
    assert not v.validate_attribute_types(node)


def test_rmsnorm_schema_populate_defaults() -> None:
    """Test populating default eps for RMSNorm."""
    v = Validator()
    node = LogicalNode(
        id="norm1",
        op_type="RMSNorm",
        domain="ml.switcheroo.custom",
        inputs=["X", "weight"],
    )
    v.populate_defaults(node)
    assert node.attributes["eps"] == 1e-6


def test_swiglu_schema_validation_success() -> None:
    """Validate correct SwiGLU node."""
    v = Validator()
    node = LogicalNode(
        id="glu1",
        op_type="SwiGLU",
        domain="ml.switcheroo.custom",
        inputs=["X"],
        attributes={"dim": -1},
    )
    assert not v.validate_kind(node)
    assert not v.validate_required_attributes(node)
    assert not v.validate_attribute_types(node)


def test_rope_schema_validation_success() -> None:
    """Validate correct RoPE node."""
    v = Validator()
    node = LogicalNode(
        id="rope1",
        op_type="RoPE",
        domain="ml.switcheroo.custom",
        inputs=["X", "cos", "sin"],
        attributes={"dim": -1},
    )
    assert not v.validate_kind(node)
    assert not v.validate_required_attributes(node)
    assert not v.validate_attribute_types(node)


def test_flash_attention_schema_validation() -> None:
    """Validate FlashAttention with causal flag and scale."""
    v = Validator()
    node = LogicalNode(
        id="fa1",
        op_type="FlashAttention",
        domain="ml.switcheroo.custom",
        inputs=["Q", "K", "V"],
        attributes={"causal": True, "scale": 0.125},
    )
    assert not v.validate_kind(node)
    assert not v.validate_required_attributes(node)
    assert not v.validate_attribute_types(node)

    # Invalid attribute type
    invalid_node = LogicalNode(
        id="fa2",
        op_type="FlashAttention",
        domain="ml.switcheroo.custom",
        inputs=["Q", "K", "V"],
        attributes={"causal": "not_a_bool"},
    )
    errors = v.validate_attribute_types(invalid_node)
    assert len(errors) == 1
    assert errors[0].level == ValidationLevel.ERROR


def test_vision_patch_embedding_validation() -> None:
    """Validate VisionPatchEmbedding with patch_size and embed_dim."""
    v = Validator()
    node = LogicalNode(
        id="patch_embed",
        op_type="VisionPatchEmbedding",
        domain="ml.switcheroo.custom",
        inputs=["X", "weight"],
        attributes={"patch_size": [16, 16], "embed_dim": 768},
    )
    assert not v.validate_kind(node)
    assert not v.validate_required_attributes(node)
    assert not v.validate_attribute_types(node)

    # Missing required attribute
    missing_attr_node = LogicalNode(
        id="patch_embed2",
        op_type="VisionPatchEmbedding",
        domain="ml.switcheroo.custom",
        inputs=["X", "weight"],
        attributes={"patch_size": [16, 16]},
    )
    errors = v.validate_required_attributes(missing_attr_node)
    assert len(errors) == 1
    assert errors[0].attribute == "embed_dim"
