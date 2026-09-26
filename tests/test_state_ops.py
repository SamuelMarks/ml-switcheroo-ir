"""Tests for state-mutation operation schemas and validation."""

from __future__ import annotations

from pathlib import Path

from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalNode,
)
from ml_switcheroo_ir.schema.custom_ops import (
    STATE_OPS_REGISTRY,
    _load_state_schemas,
)
from ml_switcheroo_ir.validator import Validator


def test_state_ops_registry_loaded() -> None:
    """Test that canonical state mutation operations are defined in STATE_OPS_REGISTRY."""
    assert "ReadVariable" in STATE_OPS_REGISTRY
    assert "AssignVariable" in STATE_OPS_REGISTRY
    assert "ScatterUpdate" in STATE_OPS_REGISTRY

    read_schema = STATE_OPS_REGISTRY["ReadVariable"]
    assert read_schema.domain == "ml.switcheroo.state"
    assert "variable_name" in read_schema.attributes
    assert read_schema.attributes["variable_name"].required is True

    assign_schema = STATE_OPS_REGISTRY["AssignVariable"]
    assert "variable_name" in assign_schema.attributes
    assert assign_schema.attributes["variable_name"].required is True

    scatter_schema = STATE_OPS_REGISTRY["ScatterUpdate"]
    assert "axis" in scatter_schema.attributes
    assert scatter_schema.attributes["axis"].default == 0


def test_state_ops_strict_validation_success() -> None:
    """Test that valid state mutation operations pass Validator under ValidationLevel.STRICT."""
    validator = Validator(strict=True)

    read_node = LogicalNode(
        id="read_w",
        op_type="ReadVariable",
        domain="ml.switcheroo.state",
        attributes={"variable_name": "dense/weight", "dtype": "float32"},
        shape_metadata=(10, 20),
        outputs=["w_val"],
    )
    assign_node = LogicalNode(
        id="assign_w",
        op_type="AssignVariable",
        domain="ml.switcheroo.state",
        attributes={"variable_name": "dense/weight"},
        shape_metadata=(10, 20),
        inputs=["w_val"],
        outputs=["w_updated"],
    )
    scatter_node = LogicalNode(
        id="scatter_w",
        op_type="ScatterUpdate",
        domain="ml.switcheroo.state",
        attributes={"variable_name": "dense/weight", "axis": 0},
        shape_metadata=(10, 20),
        inputs=["w_val", "indices", "updates"],
        outputs=["w_scattered"],
    )

    graph = LogicalGraph(
        name="StateGraph",
        nodes={
            "read_w": read_node,
            "assign_w": assign_node,
            "scatter_w": scatter_node,
        },
        inputs=["indices", "updates"],
        outputs=["w_scattered"],
    )

    errors = validator.validate(graph)
    assert not errors


def test_state_ops_validation_missing_required_attribute() -> None:
    """Test that missing required attribute on a state op produces a validation error."""
    validator = Validator(strict=True)

    bad_read = LogicalNode(
        id="bad_read",
        op_type="ReadVariable",
        domain="ml.switcheroo.state",
        attributes={},  # missing variable_name
        shape_metadata=(10, 20),
        outputs=["w_val"],
    )

    graph = LogicalGraph(nodes={"bad_read": bad_read}, outputs=["bad_read"])
    errors = validator.validate(graph)
    assert len(errors) == 1
    assert "variable_name" in errors[0].message


def test_state_ops_loader_missing_file(tmp_path: Path) -> None:
    """Test _load_state_schemas with non-existent path handles missing file gracefully.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    nonexistent = tmp_path / "does_not_exist.json"
    # Should not raise an error
    _load_state_schemas(json_path=nonexistent)


def test_state_ops_custom_state_registry_and_unknown_op() -> None:
    """Test Validator with custom state_registry and unknown state op handling."""
    custom_state_reg = dict(STATE_OPS_REGISTRY)
    val = Validator(state_registry=custom_state_reg)
    assert val.state_registry is custom_state_reg

    unknown_node = LogicalNode(
        id="bad_state",
        op_type="UnknownStateOp",
        domain="ml.switcheroo.state",
    )
    errs = val.validate_kind(unknown_node)
    assert len(errs) == 1
    assert "UnknownStateOp" in errs[0].message
    assert "state registry" in errs[0].message
