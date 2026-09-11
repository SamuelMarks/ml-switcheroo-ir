"""Additional coverage tests for validator and GroundingValidator."""

from __future__ import annotations

from pathlib import Path

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.schema.onnx_registry import OpSchema
from ml_switcheroo_ir.validator import GroundingValidator, Validator


def test_validator_stablehlo_custom_registry_and_dict_attr() -> None:
    """Test custom stablehlo registry and dict attribute type checking."""
    fake_hlo = {
        "custom_hlo": OpSchema(
            name="custom_hlo",
            domain="stablehlo",
            version=1,
            attributes={},
            inputs=[],
            outputs=[],
        )
    }
    v_custom = Validator(stablehlo_registry=fake_hlo)
    node = LogicalNode(id="n1", op_type="custom_hlo", domain="stablehlo")
    assert not v_custom.validate_kind(node)
    assert v_custom._get_schema(node) is fake_hlo["custom_hlo"]

    # Fallback to default stablehlo registry
    v_default = Validator()
    dot_node_bad_dict = LogicalNode(
        id="dot_bad",
        op_type="dot_general",
        domain="stablehlo",
        inputs=["lhs", "rhs"],
        attributes={"dot_dimension_numbers": "not_a_dict"},
    )
    errors = v_default.validate_attribute_types(dot_node_bad_dict)
    assert len(errors) == 1
    assert "Expected dict" in errors[0].message


def test_grounding_validator_edge_branches(tmp_path: Path) -> None:
    """Test GroundingValidator branch conditions."""
    # Non-existent file path
    gv_bad_path = GroundingValidator(
        snapshot_manifest=str(tmp_path / "non_existent.json")
    )
    assert gv_bad_path.grounded_symbols == {}

    # None manifest
    gv_none = GroundingValidator(snapshot_manifest=None)
    assert gv_none.grounded_symbols == {}

    # Manifest with flat dict where value is non-dict
    manifest_mixed = {
        "meta": "not_a_dict",
        "valid_op": {
            "name": "valid_op",
            "params": ["not_a_dict_param", {"name": "arg1"}],
            "attributes": {"attr1": {}},
        },
    }
    gv_mixed = GroundingValidator(snapshot_manifest=manifest_mixed)
    assert "valid_op" in gv_mixed.grounded_symbols

    # Node with attributes matching known params
    node_good = LogicalNode(
        id="n_good",
        op_type="valid_op",
        domain="test",
        attributes={"arg1": 10, "attr1": "value"},
    )
    assert not gv_mixed.validate_grounding(node_good)

    # Matched symbol with empty params and attributes
    manifest_empty_params = {
        "op_no_params": {
            "name": "op_no_params",
        }
    }
    gv_empty_params = GroundingValidator(snapshot_manifest=manifest_empty_params)
    node_no_params = LogicalNode(
        id="n_np",
        op_type="op_no_params",
        domain="test",
        attributes={"any_attr": 42},
    )
    assert not gv_empty_params.validate_grounding(node_no_params)

    # Empty graph audit
    empty_graph = LogicalGraph(nodes={})
    report = gv_mixed.audit_graph(empty_graph)
    assert report.total_nodes == 0
    assert report.hallucination_score == 0.0

    # Categorized manifest with items missing api_path and name or non-dict items
    cat_manifest = {
        "categories": {
            "cat1": [
                "not_a_dict",
                {"other_key": "val"},
                {"name": "only_name"},
            ],
            "cat2": "not_a_list",
        }
    }
    gv_cat = GroundingValidator(snapshot_manifest=cat_manifest)
    assert "only_name" in gv_cat.grounded_symbols
