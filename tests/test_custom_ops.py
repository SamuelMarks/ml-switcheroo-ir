"""Tests for custom ops extensibility."""

import json
from tempfile import NamedTemporaryFile

from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_ir.schema.custom_ops import (
    CustomOpSchema,
    CustomAttributeSchema,
    Registry,
)
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY
from ml_switcheroo_ir.validator import Validator, ValidationLevel


def test_register_and_validate_custom_op():
    """Test registering a custom op and validating it."""
    # Define custom FlashAttention
    flash_attn_schema = CustomOpSchema(
        name="FlashAttention",
        domain="ml.switcheroo.custom",
        attributes=[
            CustomAttributeSchema(
                name="dropout_p", type="float", required=False, default=0.0
            ),
            CustomAttributeSchema(name="causal", type="bool", required=True),
        ],
        inputs=["q", "k", "v"],
        outputs=["out"],
    )

    registry = Registry(base_registry=ONNX_REGISTRY)
    registry.register_custom_op(flash_attn_schema)

    validator = Validator(registry=registry.schemas)

    # Valid node
    valid_node = LogicalNode(
        id="fa1",
        op_type="FlashAttention",
        domain="ml.switcheroo.custom",
        attributes={"causal": True},
    )

    assert not validator.validate_kind(valid_node)
    assert not validator.validate_required_attributes(valid_node)
    assert not validator.validate_attribute_types(valid_node)

    validator.populate_defaults(valid_node)
    assert valid_node.attributes["dropout_p"] == 0.0


def test_custom_op_type_checking():
    """Test custom attribute type enforcement."""
    schema = CustomOpSchema(
        name="CustomAdd",
        domain="ai.custom",
        attributes=[CustomAttributeSchema(name="scale", type="float", required=True)],
    )
    registry = Registry()
    registry.register_custom_op(schema)
    validator = Validator(registry=registry.schemas)

    invalid_node = LogicalNode(
        id="c1",
        op_type="CustomAdd",
        domain="ai.custom",
        attributes={"scale": "1.0"},  # should be float
    )

    errors = validator.validate_attribute_types(invalid_node)
    assert len(errors) == 1
    assert errors[0].level == ValidationLevel.ERROR
    assert "Expected float" in errors[0].message


def test_load_custom_ops_from_json():
    """Test loading custom operators from a JSON file."""
    json_data = json.dumps(
        {
            "ops": [
                {
                    "name": "MyOp",
                    "domain": "my.domain",
                    "attributes": [{"name": "size", "type": "int", "required": True}],
                    "inputs": ["x"],
                    "outputs": ["y"],
                }
            ]
        }
    )

    with NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        f.write(json_data)
        file_path = f.name

    registry = Registry()
    registry.load_custom_ops_from_json(file_path)

    assert "MyOp" in registry.schemas
    op = registry.schemas["MyOp"]
    assert op.domain == "my.domain"
    assert op.inputs == ["x"]
    assert op.attributes["size"].type == "int"
    assert op.attributes["size"].required is True
