"""Tests for sharding and mesh validation in Validator."""

from __future__ import annotations

from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
)
from ml_switcheroo_ir.validator import ValidationLevel, Validator


def test_validate_sharding_valid_mesh() -> None:
    """Test valid sharding axes matching mesh."""
    v = Validator()
    mesh = LogicalMesh(shape={"data": 4, "model": 2})
    spec = PartitionSpec(axes=("data", ("model",), None))
    node = LogicalNode(id="n1", op_type="Relu", sharding=spec)

    errors = v.validate_sharding(node, mesh)
    assert not errors


def test_validate_sharding_missing_mesh() -> None:
    """Test error raised when sharding without mesh."""
    v = Validator()
    spec = PartitionSpec(axes=("data", None))
    node = LogicalNode(id="n1", op_type="Relu", sharding=spec)

    errors = v.validate_sharding(node, None)
    assert len(errors) == 1
    assert errors[0].attribute == "sharding"
    assert errors[0].level == ValidationLevel.ERROR
    assert "no LogicalMesh defined" in errors[0].message


def test_validate_sharding_unknown_axis() -> None:
    """Test error raised when axis not in mesh."""
    v = Validator()
    mesh = LogicalMesh(shape={"data": 4})
    spec = PartitionSpec(axes=("data", "unknown_axis", None))
    node = LogicalNode(id="n1", op_type="Relu", sharding=spec)

    errors = v.validate_sharding(node, mesh)
    assert len(errors) == 1
    assert errors[0].attribute == "sharding"
    assert "unknown_axis" in errors[0].message


def test_validate_sharding_none_spec() -> None:
    """Test node without sharding returns no errors."""
    v = Validator()
    node = LogicalNode(id="n1", op_type="Relu")
    assert not v.validate_sharding(node, None)


def test_validate_edges_and_graph() -> None:
    """Test validate_edges and validate_graph with missing source and targets."""
    v = Validator()
    mesh = LogicalMesh(shape={"data": 4})
    spec = PartitionSpec(axes=("data",))
    n1 = LogicalNode(id="n1", op_type="Relu", sharding=spec)
    n2 = LogicalNode(id="n2", op_type="Relu", inputs=["n1", "missing_node"])

    graph = LogicalGraph(nodes={"n1": n1, "n2": n2}, mesh=mesh)

    errors = v.validate_graph(graph)
    messages = [e.message for e in errors]
    assert any("Node input 'missing_node' does not exist." in m for m in messages)

    invalid_edge_graph = LogicalGraph(nodes={"n1": n1})
    n1.inputs = ["missing_source"]
    edge_errors = v.validate_edges(invalid_edge_graph)
    edge_messages = [e.message for e in edge_errors]
    assert any(
        "Node input 'missing_source' does not exist." in m for m in edge_messages
    )


def test_validate_kind_unrecognized_domain() -> None:
    """Test validating a node with an unrecognized domain."""
    v = Validator()
    node = LogicalNode(id="n1", op_type="CustomOp", domain="unknown.domain.org")
    errors = v.validate_kind(node)
    assert len(errors) == 1
    assert errors[0].attribute == "domain"
    assert "Unrecognized domain" in errors[0].message


def test_validate_kind_custom_missing_op() -> None:
    """Test validating a node with ml.switcheroo.custom domain but missing op."""
    v = Validator()
    node = LogicalNode(
        id="n1", op_type="NonExistentCustomOp", domain="ml.switcheroo.custom"
    )
    errors = v.validate_kind(node)
    assert len(errors) == 1
    assert errors[0].attribute == "kind"
    assert "NonExistentCustomOp" in errors[0].message
