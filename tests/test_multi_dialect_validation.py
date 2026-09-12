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


def test_stablehlo_expanded_ops_validation() -> None:
    """Test validating expanded StableHLO operators."""
    v = Validator()

    # broadcast_in_dim
    bcast = LogicalNode(
        id="b1",
        op_type="broadcast_in_dim",
        domain="stablehlo",
        inputs=["op1"],
        attributes={"broadcast_dimensions": [1, 2]},
    )
    assert not v.validate_kind(bcast)
    assert not v.validate_required_attributes(bcast)

    # iota
    iota_node = LogicalNode(
        id="i1",
        op_type="iota",
        domain="stablehlo",
        attributes={"iota_dimension": 0},
    )
    assert not v.validate_kind(iota_node)
    assert not v.validate_required_attributes(iota_node)

    # cholesky
    cholesky_node = LogicalNode(
        id="ch1",
        op_type="cholesky",
        domain="stablehlo",
        inputs=["mat"],
        attributes={"lower": True},
    )
    assert not v.validate_kind(cholesky_node)
    assert not v.validate_required_attributes(cholesky_node)


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

    # Test all added MLIR ops
    exp2_node = LogicalNode(id="m1", op_type="exp2", domain="math", inputs=["%x"])
    assert not v.validate_kind(exp2_node)

    rsqrt_node = LogicalNode(id="m2", op_type="rsqrt", domain="math", inputs=["%x"])
    assert not v.validate_kind(rsqrt_node)

    erf_node = LogicalNode(id="m3", op_type="erf", domain="math", inputs=["%x"])
    assert not v.validate_kind(erf_node)

    expand_node = LogicalNode(
        id="t1", op_type="expand_shape", domain="tensor", inputs=["%t"]
    )
    assert not v.validate_kind(expand_node)

    collapse_node = LogicalNode(
        id="t2", op_type="collapse_shape", domain="tensor", inputs=["%t"]
    )
    assert not v.validate_kind(collapse_node)

    conv2d_node = LogicalNode(
        id="l1", op_type="conv2d", domain="linalg", inputs=["%in", "%f", "%out"]
    )
    assert not v.validate_kind(conv2d_node)

    bmm_node = LogicalNode(
        id="l2",
        op_type="batch_matmul",
        domain="linalg",
        inputs=["%a", "%b", "%out"],
    )
    assert not v.validate_kind(bmm_node)

    cond_node = LogicalNode(
        id="s1", op_type="condition", domain="scf", inputs=["%flag"]
    )
    assert not v.validate_kind(cond_node)


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
