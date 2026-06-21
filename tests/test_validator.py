"""Tests for the validation module."""

import pytest
from ml_switcheroo_ir import LogicalNode, LogicalGraph
from ml_switcheroo_ir.validator import Validator, ValidationLevel
from ml_switcheroo_ir.schema.onnx_registry import OpSchema, OpAttribute

# Create a small mock registry for controlled testing
MOCK_REGISTRY = {
    "Gemm": OpSchema(
        name="Gemm",
        domain="ai.onnx",
        version=11,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=1.0
            ),
            "beta": OpAttribute(name="beta", type="float", required=False, default=1.0),
            "transA": OpAttribute(name="transA", type="int", required=False, default=0),
            "transB": OpAttribute(name="transB", type="int", required=False, default=0),
        },
        inputs=["A", "B", "C"],
        outputs=["Y"],
    ),
    "Conv": OpSchema(
        name="Conv",
        domain="ai.onnx",
        version=11,
        attributes={
            "kernel_shape": OpAttribute(
                name="kernel_shape", type="List[int]", required=True, default=None
            ),
            "strides": OpAttribute(
                name="strides", type="List[int]", required=False, default=[1, 1]
            ),
            "group": OpAttribute(name="group", type="int", required=False, default=1),
        },
        inputs=["X", "W", "B"],
        outputs=["Y"],
    ),
}


@pytest.fixture
def validator() -> Validator:
    """Returns a Validator initialized with the mock registry."""
    return Validator(registry=MOCK_REGISTRY)


def test_validator_valid_gemm(validator: Validator) -> None:
    """Test a fully compliant Gemm node returning no errors."""
    node = LogicalNode(id="gemm1", op_type="Gemm", attributes={"alpha": 2.0})

    errors = validator.validate_kind(node)
    assert not errors

    errors = validator.validate_required_attributes(node)
    assert not errors

    errors = validator.validate_attribute_types(node)
    assert not errors


def test_validator_missing_required_attr(validator: Validator) -> None:
    """Test a Conv node lacking kernel_shape asserting a specific ValidationError."""
    node = LogicalNode(id="conv1", op_type="Conv", attributes={"strides": [2, 2]})

    errors = validator.validate_required_attributes(node)
    assert len(errors) == 1
    assert errors[0].attribute == "kernel_shape"
    assert errors[0].level == ValidationLevel.ERROR


def test_validator_invalid_type(validator: Validator) -> None:
    """Test providing a string '1' when an integer 1 is required."""
    node = LogicalNode(
        id="conv1", op_type="Conv", attributes={"group": "1", "kernel_shape": [3, 3]}
    )

    errors = validator.validate_attribute_types(node)
    assert len(errors) == 1
    assert errors[0].attribute == "group"
    assert "Expected int" in errors[0].message
    assert errors[0].level == ValidationLevel.ERROR


def test_validator_default_population(validator: Validator) -> None:
    """Test to ensure missing optional attributes are injected."""
    node = LogicalNode(id="conv1", op_type="Conv", attributes={"kernel_shape": [3, 3]})
    validator.populate_defaults(node)

    assert "strides" in node.metadata
    assert node.metadata["strides"] == [1, 1]
    assert "group" in node.metadata
    assert node.metadata["group"] == 1


def test_validator_invalid_edge(validator: Validator) -> None:
    """Test for dangling edge references."""
    node1 = LogicalNode(id="n1", op_type="Gemm", inputs=["n0", "n2"])
    graph = LogicalGraph(nodes={"n1": node1})

    errors = validator.validate_edges(graph)
    assert len(errors) == 2
    assert errors[0].node_id == "n1"
    assert errors[1].node_id == "n1"


def test_validator_unknown_kind(validator: Validator) -> None:
    """Test validating an unknown operator."""
    node = LogicalNode(id="n1", op_type="UnknownOp")
    errors = validator.validate_kind(node)
    assert len(errors) == 1
    assert errors[0].attribute == "kind"
    assert errors[0].level == ValidationLevel.ERROR


def test_validator_unknown_attribute(validator: Validator) -> None:
    """Test providing an unregistered attribute issues a warning."""
    node = LogicalNode(id="gemm1", op_type="Gemm", attributes={"unknown_attr": 42})
    errors = validator.validate_attribute_types(node)
    assert len(errors) == 1
    assert errors[0].attribute == "unknown_attr"
    assert errors[0].level == ValidationLevel.WARNING


def test_validator_graph_integration(validator: Validator) -> None:
    """Test the integrated validate_graph method."""
    node1 = LogicalNode(id="n1", op_type="Gemm", attributes={"alpha": 2.0})
    node2 = LogicalNode(
        id="n2", op_type="Conv", inputs=["n1", "n3"]
    )  # missing kernel_shape, dangling n3

    graph = LogicalGraph(nodes={"n1": node1, "n2": node2})
    errors = validator.validate_graph(graph)

    # We expect:
    # 1. Conv missing kernel_shape
    # 2. node n2 has dangling input n3
    assert len(errors) == 2

    error_attrs = {e.attribute for e in errors}
    assert "kernel_shape" in error_attrs
    assert "inputs" in error_attrs

    # Check default was populated on Gemm
    assert node1.metadata.get("transA") == 0


def test_validator_default_registry() -> None:
    """Test that Validator defaults to ONNX_REGISTRY."""
    v = Validator()
    assert v.registry is not None
    assert "Add" in v.registry


def test_validator_list_type_check_errors(validator: Validator) -> None:
    """Test invalid list items."""
    node = LogicalNode(
        id="conv1", op_type="Conv", attributes={"kernel_shape": [3, "3"]}
    )
    errors = validator.validate_attribute_types(node)
    assert len(errors) == 1
    assert errors[0].attribute == "kernel_shape"

    # Int type valid check
    node2 = LogicalNode(
        id="conv2", op_type="Conv", attributes={"group": 2.5, "kernel_shape": [3]}
    )
    errors2 = validator.validate_attribute_types(node2)
    assert errors2[0].attribute == "group"

    # Float type valid check
    node3 = LogicalNode(id="gemm1", op_type="Gemm", attributes={"alpha": "2.0"})
    errors3 = validator.validate_attribute_types(node3)
    assert errors3[0].attribute == "alpha"

    # Float type with int value (should be valid)
    node4 = LogicalNode(id="gemm2", op_type="Gemm", attributes={"alpha": 2})
    errors4 = validator.validate_attribute_types(node4)
    assert not errors4


def test_validator_various_types() -> None:
    """Test string, List[float], List[str], bool, etc."""
    mock_reg = {
        "TestOp": OpSchema(
            name="TestOp",
            domain="ai.onnx",
            version=1,
            attributes={
                "s": OpAttribute(name="s", type="str", required=False, default=""),
                "lf": OpAttribute(
                    name="lf", type="List[float]", required=False, default=[]
                ),
                "ls": OpAttribute(
                    name="ls", type="List[str]", required=False, default=[]
                ),
                "b": OpAttribute(name="b", type="bool", required=False, default=False),
                "any": OpAttribute(
                    name="any", type="Any", required=False, default=None
                ),
            },
            inputs=[],
            outputs=[],
        )
    }
    v = Validator(registry=mock_reg)

    # Valid
    node_valid = LogicalNode(
        id="n1",
        op_type="TestOp",
        attributes={"s": "ok", "lf": [1.0, 2.0], "ls": ["a"], "b": True, "any": {}},
    )
    assert not v.validate_attribute_types(node_valid)

    # Invalid str
    assert v.validate_attribute_types(
        LogicalNode(id="n", op_type="TestOp", attributes={"s": 1})
    )
    # Invalid List[float]
    assert v.validate_attribute_types(
        LogicalNode(id="n", op_type="TestOp", attributes={"lf": ["a"]})
    )
    assert v.validate_attribute_types(
        LogicalNode(id="n", op_type="TestOp", attributes={"lf": 1.0})
    )
    # Invalid List[str]
    assert v.validate_attribute_types(
        LogicalNode(id="n", op_type="TestOp", attributes={"ls": [1]})
    )
    assert v.validate_attribute_types(
        LogicalNode(id="n", op_type="TestOp", attributes={"ls": "a"})
    )
    # Invalid bool
    assert v.validate_attribute_types(
        LogicalNode(id="n", op_type="TestOp", attributes={"b": 1})
    )


def test_validator_custom_domain() -> None:
    """Test that custom domains don't trigger kind validation errors if not checked."""
    v = Validator(registry=MOCK_REGISTRY)
    node = LogicalNode(id="n1", op_type="CustomOp", domain="ai.custom")
    assert not v.validate_kind(node)


def test_validator_missing_kind_methods() -> None:
    """Test methods return quickly if kind not in registry."""
    v = Validator(registry=MOCK_REGISTRY)
    node = LogicalNode(id="n1", op_type="UnknownOp")
    assert not v.validate_required_attributes(node)
    assert not v.validate_attribute_types(node)

    # populate defaults returns quickly
    v.populate_defaults(node)
    assert not node.metadata
