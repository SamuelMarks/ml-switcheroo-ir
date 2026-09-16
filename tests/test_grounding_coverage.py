"""Additional coverage tests for validator and GroundingValidator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

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

    # Default if none when directory exists
    default_dir = tmp_path / "default_snaps"
    default_dir.mkdir()
    (default_dir / "sample.json").write_text(
        json.dumps({"test_op": {"name": "test_op"}}), encoding="utf-8"
    )
    with patch("ml_switcheroo_ir.validator.DEFAULT_SNAPSHOT_DIR", str(default_dir)):
        gv_default = GroundingValidator(
            snapshot_manifest=None, use_default_if_none=True
        )
        assert len(gv_default.grounded_symbols) > 0

    # Default if none when directory does not exist
    with patch(
        "ml_switcheroo_ir.validator.DEFAULT_SNAPSHOT_DIR",
        str(tmp_path / "non_existent_snaps"),
    ):
        gv_no_default = GroundingValidator(
            snapshot_manifest=None, use_default_if_none=True
        )
        assert len(gv_no_default.grounded_symbols) == 0

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
            "cat2": {
                "sub_dict_item": {"name": "sub_dict_name"},
                "invalid_sub": "not_a_dict",
            },
            "cat3": "not_a_list",
        }
    }
    gv_cat = GroundingValidator(snapshot_manifest=cat_manifest)
    assert "only_name" in gv_cat.grounded_symbols
    assert "sub_dict_name" in gv_cat.grounded_symbols

    # Top-level list in non-categorized dict
    manifest_top_list = {
        "ops_list": [{"name": "list_op_name"}, "not_a_dict"],
    }
    gv_top_list = GroundingValidator(snapshot_manifest=manifest_top_list)
    assert "list_op_name" in gv_top_list.grounded_symbols


def test_compute_levenshtein() -> None:
    """Test compute_levenshtein edit distance calculations and edge cases."""
    from ml_switcheroo_ir.validator import compute_levenshtein

    assert compute_levenshtein("", "") == 0
    assert compute_levenshtein("a", "abc") == 2
    assert compute_levenshtein("abc", "") == 3
    assert compute_levenshtein("kitten", "sitting") == 3
    assert compute_levenshtein("same", "same") == 0


def test_grounding_validator_multi_format_and_directory(tmp_path: Path) -> None:
    """Test GroundingValidator ingesting top-level lists, gzip files, and directories."""
    import gzip
    import json

    # 1. Top-level list of dicts
    list_manifest = [
        {
            "name": "dot_general",
            "api_path": "stablehlo.dot_general",
            "params": [{"name": "lhs"}, {"name": "rhs"}],
            "attributes": [
                {"name": "dot_dimension_numbers", "type": "dict"},
                "plain_attr_str",
            ],
            "operands": [
                {"name": "lhs_op"},
                "plain_op_str",
                123,  # non-dict, non-str branch
            ],
        },
        {"mnemonic": "V_ADD_F32", "api_path": "amd_rdna.v_add_f32"},
        "invalid_non_dict_entry",
    ]

    gv_list = GroundingValidator(snapshot_manifest=list_manifest)
    assert "stablehlo.dot_general" in gv_list.grounded_symbols
    assert "dot_general" in gv_list.grounded_symbols
    assert "V_ADD_F32" in gv_list.grounded_symbols

    # Test attributes list[dict] and list[str], operands list[dict] and list[str]
    valid_hlo_node = LogicalNode(
        id="hlo1",
        op_type="dot_general",
        domain="stablehlo",
        attributes={
            "dot_dimension_numbers": {},
            "plain_attr_str": "test",
            "lhs_op": "in1",
            "plain_op_str": "in2",
        },
    )
    assert not gv_list.validate_grounding(valid_hlo_node)

    # 2. Directory traversal with .json, .json.gz, and corrupt file
    snap_dir = tmp_path / "snapshots_dir"
    snap_dir.mkdir()

    # Regular json file
    json_file = snap_dir / "ops1.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump([{"name": "op_plain", "api_path": "dial.op_plain"}], f)

    # Gzipped json file
    gz_file = snap_dir / "ops2.json.gz"
    with gzip.open(gz_file, "wt", encoding="utf-8") as f:
        json.dump([{"name": "op_gz", "api_path": "dial.op_gz"}], f)

    # Corrupt/non-json file
    bad_file = snap_dir / "corrupt.json"
    with open(bad_file, "w", encoding="utf-8") as f:
        f.write("not valid json")

    # Corrupt .json.gz file
    bad_gz = snap_dir / "corrupt.json.gz"
    bad_gz.write_bytes(b"not valid gzip data")

    # Ignored non-json file in directory
    ignored_file = snap_dir / "ignored.txt"
    ignored_file.write_text("should be ignored", encoding="utf-8")

    # Load directory
    gv_dir = GroundingValidator(snapshot_manifest=str(snap_dir))
    assert "dial.op_plain" in gv_dir.grounded_symbols
    assert "dial.op_gz" in gv_dir.grounded_symbols

    # Ingest target of unsupported type and load unsupported data type
    gv_unsupported = GroundingValidator(snapshot_manifest=12345)  # type: ignore[arg-type]
    assert gv_unsupported.grounded_symbols == {}
    gv_unsupported._load_snapshot_data("not_list_or_dict")  # type: ignore[arg-type]

    # 3. Collection of targets (list of file paths and dicts)
    manifest_collection = [
        str(json_file),
        {"custom_target": {"name": "custom_target"}},
    ]
    gv_coll = GroundingValidator(snapshot_manifest=manifest_collection)
    assert "dial.op_plain" in gv_coll.grounded_symbols
    assert "custom_target" in gv_coll.grounded_symbols

    # 4. audit_graph_grounding function with list and directory
    from ml_switcheroo_ir.validator import audit_graph_grounding

    test_graph = LogicalGraph(
        nodes={"p1": LogicalNode(id="p1", op_type="op_plain", domain="dial")}
    )
    rep_dir = audit_graph_grounding(test_graph, snapshots_path=str(snap_dir))
    assert rep_dir.grounded_count == 1

    rep_list = audit_graph_grounding(test_graph, snapshots_path=list_manifest)
    assert rep_list.total_nodes == 1

    # Symbol with attributes and operands that have dicts without 'name'
    unnamed_manifest = [
        {
            "name": "unnamed_op",
            "attributes": [{"no_name_field": 1}],
            "operands": [{"no_name_field": 2}],
        }
    ]
    gv_unnamed = GroundingValidator(snapshot_manifest=unnamed_manifest)
    node_unnamed = LogicalNode(id="u1", op_type="unnamed_op", domain="test")
    assert not gv_unnamed.validate_grounding(node_unnamed)


def test_grounding_validator_fuzzy_suggestions() -> None:
    """Test fuzzy typo suggestions for hallucinated operators and attributes."""
    snapshot = {
        "stablehlo.dot_general": {
            "name": "dot_general",
            "api_path": "stablehlo.dot_general",
            "params": [{"name": "lhs"}, {"name": "rhs"}],
            "attributes": {"window_strides": {}},
        },
        "arith.addf": {
            "name": "addf",
            "api_path": "arith.addf",
            "params": [{"name": "lhs"}, {"name": "rhs"}],
        },
    }
    gv = GroundingValidator(snapshot_manifest=snapshot)

    # 1. Known synonym: stablehlo matmul -> dot_general
    node_syn = LogicalNode(id="n1", op_type="matmul", domain="stablehlo")
    errors_syn = gv.validate_grounding(node_syn)
    assert len(errors_syn) == 1
    assert "Did you mean 'stablehlo.dot_general'?" in errors_syn[0].message

    # 2. Domain prefix fuzzy match: stablehlo.dot_genral -> dot_general
    node_fuzzy_domain = LogicalNode(id="n2", op_type="dot_genral", domain="stablehlo")
    errors_fuzzy_domain = gv.validate_grounding(node_fuzzy_domain)
    assert len(errors_fuzzy_domain) == 1
    assert "Did you mean 'stablehlo.dot_general'?" in errors_fuzzy_domain[0].message

    # 3. Global fuzzy match: addff -> addf / arith.addf (domain without prefix match)
    node_fuzzy_global = LogicalNode(id="n3", op_type="addff", domain="other")
    errors_fuzzy_global = gv.validate_grounding(node_fuzzy_global)
    assert len(errors_fuzzy_global) == 1
    assert (
        "Did you mean 'addf'?" in errors_fuzzy_global[0].message
        or "Did you mean 'arith.addf'?" in errors_fuzzy_global[0].message
    )

    # 3b. Empty domain fuzzy match
    node_fuzzy_empty_dom = LogicalNode(id="n3b", op_type="addff", domain="")
    errors_fuzzy_empty = gv.validate_grounding(node_fuzzy_empty_dom)
    assert len(errors_fuzzy_empty) == 1
    assert (
        "Did you mean 'addf'?" in errors_fuzzy_empty[0].message
        or "Did you mean 'arith.addf'?" in errors_fuzzy_empty[0].message
    )

    # 4. Far distance hallucination (> 3 edit distance): no suggestion
    node_far = LogicalNode(
        id="n4", op_type="completely_unknown_super_long_op", domain="other"
    )
    errors_far = gv.validate_grounding(node_far)
    assert len(errors_far) == 1
    assert "Did you mean" not in errors_far[0].message

    # 5. Fuzzy attribute match: window_stride -> window_strides
    node_bad_attr_fuzzy = LogicalNode(
        id="n5",
        op_type="dot_general",
        domain="stablehlo",
        attributes={"window_stride": [1, 1]},
    )
    errors_bad_attr_fuzzy = gv.validate_grounding(node_bad_attr_fuzzy)
    assert len(errors_bad_attr_fuzzy) == 1
    assert "Did you mean 'window_strides'?" in errors_bad_attr_fuzzy[0].message

    # 6. Attribute far distance hallucination (> 3 edit distance): no suggestion
    node_bad_attr_far = LogicalNode(
        id="n6",
        op_type="dot_general",
        domain="stablehlo",
        attributes={"completely_unknown_attr": 42},
    )
    errors_bad_attr_far = gv.validate_grounding(node_bad_attr_far)
    assert len(errors_bad_attr_far) == 1
    assert "Did you mean" not in errors_bad_attr_far[0].message


def test_grounding_against_ml_framework_snapshots_golden() -> None:
    """Verify GroundingValidator against real ml-framework-snapshots datasets if present."""
    snapshots_repo = (
        Path(__file__).resolve().parent.parent.parent / "ml-framework-snapshots"
    )
    snapshots_dir = snapshots_repo / "src" / "ml_framework_snapshots" / "snapshots"
    if not snapshots_dir.exists():
        pytest.skip(
            "ml-framework-snapshots repository not present in sibling directory."
        )

    stablehlo_file = snapshots_dir / "stablehlo_v1.0.0.json"
    if not stablehlo_file.exists():
        candidates = sorted(snapshots_dir.glob("stablehlo*.json"))
        if not candidates:
            pytest.skip("StableHLO snapshot dataset not present.")
        stablehlo_file = candidates[-1]

    gv = GroundingValidator(snapshot_manifest=str(stablehlo_file))
    assert "stablehlo.dot_general" in gv.grounded_symbols
    assert "stablehlo.convolution" in gv.grounded_symbols
    assert "stablehlo.reduce" in gv.grounded_symbols

    # Audit a valid StableHLO node
    valid_node = LogicalNode(
        id="dot1",
        op_type="dot_general",
        domain="stablehlo",
        attributes={"dot_dimension_numbers": {}},
    )
    assert not gv.validate_grounding(valid_node)

    ir_file = snapshots_dir / "ir_v0.0.3.json"
    if not ir_file.exists():
        candidates = sorted(snapshots_dir.glob("ir_v*.json"))
        if not candidates:
            pytest.skip("IR snapshot dataset not present.")
        ir_file = candidates[-1]
    assert ir_file.exists()
    gv_ir = GroundingValidator(snapshot_manifest=str(ir_file))
    assert (
        "LogicalGraph" in gv_ir.grounded_symbols
        or "ml_switcheroo_ir.LogicalGraph" in gv_ir.grounded_symbols
    )
    assert (
        "LogicalNode" in gv_ir.grounded_symbols
        or "ml_switcheroo_ir.LogicalNode" in gv_ir.grounded_symbols
    )
    assert (
        "PartitionSpec" in gv_ir.grounded_symbols
        or "ml_switcheroo_ir.PartitionSpec" in gv_ir.grounded_symbols
    )

    node_graph = LogicalNode(id="lg", op_type="LogicalGraph", domain="ml_switcheroo_ir")
    assert not gv_ir.validate_grounding(node_graph)
    node_node = LogicalNode(id="ln", op_type="LogicalNode", domain="ml_switcheroo_ir")
    assert not gv_ir.validate_grounding(node_node)
    node_spec = LogicalNode(id="ps", op_type="PartitionSpec", domain="ml_switcheroo_ir")
    assert not gv_ir.validate_grounding(node_spec)
