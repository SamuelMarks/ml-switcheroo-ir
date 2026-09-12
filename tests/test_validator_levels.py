"""Unit tests for ValidationLevel alignment, severity thresholds, and Validator API."""

from __future__ import annotations

from typing import Any

import pytest

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.validator import (
    GroundingValidator,
    ValidationError,
    ValidationLevel,
    Validator,
)


def test_validation_error_str() -> None:
    """Test string representation of ValidationError."""
    err = ValidationError(
        node_id="n1",
        attribute="kernel_shape",
        message="Missing required attribute",
        level=ValidationLevel.ERROR,
    )
    assert (
        "[ERROR] Node 'n1' attribute 'kernel_shape': Missing required attribute"
        in str(err)
    )


def test_validator_init_levels_and_aliases() -> None:
    """Test Validator initialization with various levels and backward compatibility aliases."""
    v_default = Validator()
    assert v_default.level == ValidationLevel.WARNING

    v_strict_flag = Validator(strict=True)
    assert v_strict_flag.level == ValidationLevel.STRICT

    v_strict_level = Validator(level=ValidationLevel.STRICT)
    assert v_strict_level.level == ValidationLevel.STRICT

    v_lenient = Validator(level=ValidationLevel.LENIENT)
    assert v_lenient.level == ValidationLevel.LENIENT

    v_error_alias = Validator(level=ValidationLevel.ERROR)
    assert v_error_alias.level == ValidationLevel.LENIENT


def test_validator_strict_mode_promotions() -> None:
    """Test that STRICT mode promotes unknown attributes and missing shape metadata to ERROR."""
    v_strict = Validator(strict=True)

    # Node with missing shape_metadata and unknown attribute
    node = LogicalNode(
        id="gemm1",
        op_type="Gemm",
        attributes={"unknown_attr": 42},
        shape_metadata=None,
    )

    errors = v_strict.validate_node(node)
    attr_errs = [e for e in errors if e.attribute == "unknown_attr"]
    shape_errs = [e for e in errors if e.attribute == "shape_metadata"]

    assert len(attr_errs) == 1
    assert attr_errs[0].level == ValidationLevel.ERROR

    assert len(shape_errs) == 1
    assert shape_errs[0].level == ValidationLevel.ERROR

    # Also test validate_kind with custom op in strict mode
    custom_node = LogicalNode(
        id="c1",
        op_type="UnknownCustom",
        domain="ml.switcheroo.custom",
    )
    kind_errs = v_strict.validate_kind(custom_node)
    assert len(kind_errs) == 1
    assert kind_errs[0].level == ValidationLevel.ERROR


def test_validator_warning_mode_accumulates() -> None:
    """Test that WARNING mode accumulates unknown attributes without promoting them to ERROR."""
    v_warning = Validator(level=ValidationLevel.WARNING)

    node = LogicalNode(
        id="gemm1",
        op_type="Gemm",
        attributes={"unknown_attr": 42},
        shape_metadata=None,
    )

    errors = v_warning.validate_node(node)
    attr_errs = [e for e in errors if e.attribute == "unknown_attr"]
    shape_errs = [e for e in errors if e.attribute == "shape_metadata"]

    assert len(attr_errs) == 1
    assert attr_errs[0].level == ValidationLevel.WARNING
    assert len(shape_errs) == 0  # Not strictly required in WARNING mode

    # Custom op in warning mode is a non-fatal warning
    custom_node = LogicalNode(
        id="c1",
        op_type="UnknownCustom",
        domain="ml.switcheroo.custom",
    )
    kind_errs = v_warning.validate_kind(custom_node)
    assert len(kind_errs) == 1
    assert kind_errs[0].level == ValidationLevel.WARNING


def test_validator_lenient_mode_filters_warnings() -> None:
    """Test that LENIENT mode filters out non-fatal warnings."""
    v_lenient = Validator(level=ValidationLevel.LENIENT)

    # Node with unknown attribute (warning) and invalid type (fatal error)
    node = LogicalNode(
        id="gemm1",
        op_type="Gemm",
        attributes={"unknown_attr": 42, "alpha": "not_a_float"},
    )

    errors = v_lenient.validate_node(node)
    # Only the fatal invalid type error should remain
    assert len(errors) == 1
    assert errors[0].attribute == "alpha"
    assert errors[0].level == ValidationLevel.ERROR

    # Test validate_graph in lenient mode
    graph = LogicalGraph(name="LenientGraph", nodes=[node])
    graph_errors = v_lenient.validate_graph(graph)
    assert len(graph_errors) == 1
    assert graph_errors[0].attribute == "alpha"


def test_validator_grounding_integration() -> None:
    """Test Validator with integrated GroundingValidator."""
    manifest: dict[str, Any] = {
        "classes": {},
        "functions": {"ai.onnx.Add": {}},
        "methods": {},
    }
    gv = GroundingValidator(snapshot_manifest=manifest)

    # In WARNING mode: Relu is valid ONNX op but not in snapshot manifest
    v_warn = Validator(level=ValidationLevel.WARNING, grounding_validator=gv)
    node_ungrounded = LogicalNode(
        id="n1",
        op_type="Relu",
        domain="ai.onnx",
    )
    errs_warn = v_warn.validate_node(node_ungrounded)
    grounding_warns = [
        e for e in errs_warn if "Symbol not found in framework snapshot" in e.message
    ]
    assert len(grounding_warns) == 1
    assert grounding_warns[0].level == ValidationLevel.WARNING

    # In STRICT mode: grounding error is promoted to ERROR
    v_strict = Validator(strict=True, grounding_validator=gv)
    errs_strict = v_strict.validate_node(node_ungrounded)
    grounding_strict = [
        e for e in errs_strict if "Symbol not found in framework snapshot" in e.message
    ]
    assert len(grounding_strict) == 1
    assert grounding_strict[0].level == ValidationLevel.ERROR

    # In LENIENT mode: non-fatal grounding warnings are omitted
    v_lenient = Validator(level=ValidationLevel.LENIENT, grounding_validator=gv)
    errs_lenient = v_lenient.validate_node(node_ungrounded)
    grounding_lenient = [
        e for e in errs_lenient if "Symbol not found in framework snapshot" in e.message
    ]
    assert len(grounding_lenient) == 0


def test_validator_validate_dispatch_and_raise_on_error() -> None:
    """Test validate() method with LogicalGraph and LogicalNode, and raise_on_error flag."""
    v = Validator(strict=True)

    valid_node = LogicalNode(
        id="x",
        op_type="Relu",
        shape_metadata=(1, 10),
    )
    # validate node without error
    assert v.validate(valid_node, raise_on_error=True) == []

    invalid_node = LogicalNode(
        id="bad",
        op_type="Relu",
        shape_metadata=None,  # triggers ERROR in STRICT mode
    )
    # validate node with raise_on_error=False
    errs = v.validate(invalid_node, raise_on_error=False)
    assert len(errs) >= 1

    # validate node with raise_on_error=True
    with pytest.raises(ValidationError) as excinfo:
        v.validate(invalid_node, raise_on_error=True)
    assert excinfo.value.node_id == "bad"

    # validate graph with raise_on_error=True
    graph = LogicalGraph(name="G", nodes=[invalid_node])
    with pytest.raises(ValidationError):
        v.validate(graph, raise_on_error=True)

    # validate valid graph
    valid_graph = LogicalGraph(name="ValidG", nodes=[valid_node])
    assert v.validate(valid_graph, raise_on_error=True) == []
