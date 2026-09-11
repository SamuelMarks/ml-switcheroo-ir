"""Additional unit tests ensuring 100% line and branch coverage in Validator."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalMesh, LogicalNode, PartitionSpec
from ml_switcheroo_ir.schema.onnx_registry import OpAttribute, OpSchema
from ml_switcheroo_ir.validator import Validator


def test_validator_custom_registry_override() -> None:
    """Test passing explicit custom_registry to Validator."""
    custom_op = OpSchema(
        name="SpecialOp",
        domain="ml.switcheroo.custom",
        version=1,
        attributes={
            "factor": OpAttribute(
                name="factor", type="float", required=True, default=1.0
            )
        },
        inputs=["in"],
        outputs=["out"],
    )
    v = Validator(custom_registry={"SpecialOp": custom_op})
    node = LogicalNode(id="n1", op_type="SpecialOp", domain="ml.switcheroo.custom")

    # validate_kind should succeed because SpecialOp is in custom_registry
    assert not v.validate_kind(node)
    # validate_required_attributes should flag missing factor
    errors = v.validate_required_attributes(node)
    assert len(errors) == 1
    assert errors[0].attribute == "factor"


def test_validator_sharding_deeply_nested_and_none() -> None:
    """Test deeply nested axis tuples including None and unrecognized axes."""
    mesh = LogicalMesh(shape={"data": 2, "model": 4})
    v = Validator()

    # Nested tuple with valid and None axes
    spec_valid = PartitionSpec(axes=((("data", None), "model"), None))  # type: ignore[arg-type]
    node_valid = LogicalNode(id="nv", op_type="Relu", sharding=spec_valid)
    assert not v.validate_sharding(node_valid, mesh)

    # Nested tuple with invalid axis
    spec_invalid = PartitionSpec(axes=(("data", "bad_mesh_dim"),))
    node_invalid = LogicalNode(id="ni", op_type="Relu", sharding=spec_invalid)
    errors = v.validate_sharding(node_invalid, mesh)
    assert len(errors) == 1
    assert "bad_mesh_dim" in errors[0].message


def test_validator_custom_op_in_onnx_registry_fallback() -> None:
    """Test custom op that is present in base registry instead of custom_registry."""
    fake_onnx = {
        "CustomInOnnx": OpSchema(
            name="CustomInOnnx",
            domain="ml.switcheroo.custom",
            version=1,
            attributes={},
            inputs=[],
            outputs=[],
        )
    }
    v = Validator(registry=fake_onnx, custom_registry={})
    node = LogicalNode(id="c1", op_type="CustomInOnnx", domain="ml.switcheroo.custom")
    assert not v.validate_kind(node)
    assert v._get_schema(node) is fake_onnx["CustomInOnnx"]

    # Op not in onnx or custom registry should return None
    missing_node = LogicalNode(
        id="m1", op_type="NonExistent", domain="ml.switcheroo.custom"
    )
    assert v._get_schema(missing_node) is None

    # Domain other than ai.onnx or ml.switcheroo.custom
    other_node = LogicalNode(id="o1", op_type="CustomInOnnx", domain="ai.custom")
    assert v._get_schema(other_node) is fake_onnx["CustomInOnnx"]


def test_validator_sharding_empty_list_axis() -> None:
    """Test sharding axis containing an empty list or other types."""
    mesh = LogicalMesh(shape={"data": 2})
    v = Validator()
    spec = PartitionSpec(axes=([], 123))  # type: ignore[arg-type]
    node = LogicalNode(id="n_empty", op_type="Relu", sharding=spec)
    assert not v.validate_sharding(node, mesh)
