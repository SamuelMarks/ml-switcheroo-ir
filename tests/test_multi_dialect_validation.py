"""Tests for multi-dialect operator validation and GroundingValidator."""

from __future__ import annotations

import json
from pathlib import Path

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.validator import (
    GroundingAuditReport,
    GroundingValidator,
    ValidationLevel,
    Validator,
    audit_graph_grounding,
)


def test_stablehlo_validation_success() -> None:
    """Test validating valid StableHLO operations."""
    v = Validator()
    dot_node = LogicalNode(
        id="dot1",
        op_type="dot_general",
        domain="stablehlo",
        inputs=["lhs", "rhs"],
        attributes={
            "dot_dimension_numbers": {
                "lhs_batch_dimensions": [0],
                "rhs_batch_dimensions": [0],
                "lhs_contracting_dimensions": [2],
                "rhs_contracting_dimensions": [1],
            }
        },
    )
    assert not v.validate_kind(dot_node)
    assert not v.validate_required_attributes(dot_node)
    assert not v.validate_attribute_types(dot_node)


def test_stablehlo_missing_required_attribute() -> None:
    """Test StableHLO op with missing required attribute."""
    v = Validator()
    conv_node = LogicalNode(
        id="conv1",
        op_type="convolution",
        domain="stablehlo",
        inputs=["lhs", "rhs"],
        attributes={
            "dimension_numbers": {},
            "window_strides": [1, 1],
            # missing padding
        },
    )
    errors = v.validate_required_attributes(conv_node)
    assert len(errors) == 1
    assert errors[0].attribute == "padding"


def test_stablehlo_unrecognized_op() -> None:
    """Test unrecognized StableHLO operation."""
    v = Validator()
    node = LogicalNode(
        id="bad_op",
        op_type="non_existent_op",
        domain="stablehlo",
    )
    errors = v.validate_kind(node)
    assert len(errors) == 1
    assert "non_existent_op" in errors[0].message


def test_mlir_dialect_validation_success() -> None:
    """Test validating core MLIR dialects."""
    v = Validator()
    add_node = LogicalNode(
        id="add1",
        op_type="addf",
        domain="arith",
        inputs=["%a", "%b"],
    )
    assert not v.validate_kind(add_node)

    # Missing operands
    bad_add_node = LogicalNode(
        id="add2",
        op_type="addf",
        domain="arith",
        inputs=["%a"],  # requires at least 2
    )
    errors = v.validate_kind(bad_add_node)
    assert len(errors) == 1
    assert "expects at least 2 operands" in errors[0].message

    # Unrecognized MLIR op
    bad_op_node = LogicalNode(
        id="unknown1",
        op_type="fake_op",
        domain="math",
    )
    errors = v.validate_kind(bad_op_node)
    assert len(errors) == 1
    assert "fake_op" in errors[0].message


def test_grounding_validator_dict_manifest() -> None:
    """Test GroundingValidator with in-memory manifest dict."""
    manifest = {
        "categories": {
            "core": [
                {
                    "name": "relu",
                    "api_path": "torch.relu",
                    "params": [{"name": "input"}],
                    "attributes": {"inplace": {}},
                }
            ]
        }
    }
    gv = GroundingValidator(snapshot_manifest=manifest)

    # Grounded node
    valid_node = LogicalNode(
        id="n1",
        op_type="relu",
        domain="torch",
        attributes={"inplace": False},
    )
    assert not gv.validate_grounding(valid_node)

    # Hallucinated op
    hallucinated_node = LogicalNode(
        id="n2",
        op_type="hallucinated_relu",
        domain="torch",
    )
    errors = gv.validate_grounding(hallucinated_node)
    assert len(errors) == 1
    assert errors[0].level == ValidationLevel.ERROR
    assert "Ungrounded symbol 'hallucinated_relu'" in errors[0].message

    # Hallucinated attribute
    bad_attr_node = LogicalNode(
        id="n3",
        op_type="relu",
        domain="torch",
        attributes={"non_existent_kwarg": 123},
    )
    attr_errors = gv.validate_grounding(bad_attr_node)
    assert len(attr_errors) == 1
    assert "Ungrounded attribute 'non_existent_kwarg'" in attr_errors[0].message


def test_grounding_validator_file_audit(tmp_path: Path) -> None:
    """Test audit_graph_grounding with JSON file snapshot."""
    manifest_file = tmp_path / "snapshot_audit.json"
    data = {
        "linalg.matmul": {
            "name": "matmul",
            "api_path": "linalg.matmul",
            "params": [{"name": "ins"}, {"name": "outs"}],
        }
    }
    manifest_file.write_text(json.dumps(data), encoding="utf-8")

    n1 = LogicalNode(id="n1", op_type="matmul", domain="linalg")
    n2 = LogicalNode(id="n2", op_type="matmul", domain="linalg", attributes={"bad": 1})
    n3 = LogicalNode(id="n3", op_type="unknown", domain="linalg")
    graph = LogicalGraph(nodes={"n1": n1, "n2": n2, "n3": n3})

    report: GroundingAuditReport = audit_graph_grounding(
        graph, snapshots_path=str(manifest_file)
    )
    assert report.total_nodes == 3
    assert report.grounded_count == 1
    assert report.ungrounded_count == 2
    assert report.hallucination_score > 0.0
    assert len(report.diagnostics) >= 2
