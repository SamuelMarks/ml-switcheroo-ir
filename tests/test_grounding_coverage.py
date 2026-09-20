"""Additional coverage tests for validator and GroundingValidator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    # 4. audit_graph_grounding function with list, directory, and None (with/without DEFAULT_SNAPSHOT_DIR)
    from ml_switcheroo_ir.validator import audit_graph_grounding

    test_graph = LogicalGraph(
        nodes={"p1": LogicalNode(id="p1", op_type="op_plain", domain="dial")}
    )
    rep_dir = audit_graph_grounding(test_graph, snapshots_path=str(snap_dir))
    assert rep_dir.grounded_count == 1

    rep_list = audit_graph_grounding(test_graph, snapshots_path=list_manifest)
    assert rep_list.total_nodes == 1

    # audit_graph_grounding with snapshots_path=None and existing DEFAULT_SNAPSHOT_DIR
    with patch("ml_switcheroo_ir.validator.DEFAULT_SNAPSHOT_DIR", str(snap_dir)):
        rep_default_dir = audit_graph_grounding(test_graph, snapshots_path=None)
        assert rep_default_dir.grounded_count == 1

    # audit_graph_grounding with snapshots_path=None and non-existing DEFAULT_SNAPSHOT_DIR
    with patch(
        "ml_switcheroo_ir.validator.DEFAULT_SNAPSHOT_DIR",
        str(tmp_path / "non_existing_dir"),
    ):
        rep_no_dir = audit_graph_grounding(test_graph, snapshots_path=None)
        assert rep_no_dir.grounded_count == 0

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


def test_grounding_validator_accepted_kwargs() -> None:
    """Test GroundingValidator properly grounds accepted_kwargs and rejects hallucinated kwargs."""
    manifest = {
        "categories": {
            "layers": [
                {
                    "name": "CustomLayer",
                    "api_path": "torch.nn.CustomLayer",
                    "accepted_kwargs": [
                        "valid_kw1",
                        {"name": "valid_kw2"},
                        {"no_name": 123},
                        12345,
                    ],
                }
            ]
        }
    }
    gv = GroundingValidator(snapshot_manifest=manifest)

    # Valid kwargs should have 0 errors
    valid_node = LogicalNode(
        id="n_valid",
        op_type="CustomLayer",
        domain="torch.nn",
        attributes={"valid_kw1": True, "valid_kw2": 128},
    )
    assert gv.validate_grounding(valid_node) == []

    # Hallucinated kwarg should trigger error
    bad_node = LogicalNode(
        id="n_bad",
        op_type="CustomLayer",
        domain="torch.nn",
        attributes={"hallucinated_kwarg": 999},
    )
    errs = gv.validate_grounding(bad_node)
    assert len(errs) == 1
    assert "Ungrounded attribute 'hallucinated_kwarg'" in errs[0].message


def test_grounding_against_ml_framework_snapshots_golden(tmp_path: Path) -> None:
    """Verify GroundingValidator against real ml-framework-snapshots datasets if present.

    Args:
        tmp_path (Path): Temporary path fixture for generating test snapshots.
    """
    from ml_switcheroo_ir.validator import DEFAULT_SNAPSHOT_DIR

    snapshots_dir = Path(DEFAULT_SNAPSHOT_DIR)
    stablehlo_files = sorted(snapshots_dir.glob("stablehlo*.json"))
    if not snapshots_dir.exists() or not stablehlo_files:
        pytest.skip("StableHLO snapshot dataset not present.")

    stablehlo_file = stablehlo_files[-1]
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

    # Validate that loading default snapshot directory loads at least 28 manifests and > 5000 symbols
    gv_all = GroundingValidator(use_default_if_none=True)
    assert len(gv_all.grounded_symbols) > 5000


def test_grounding_against_dumped_ir_snapshot(tmp_path: Path) -> None:
    """Verify GroundingValidator against locally dumped snapshot manifest.

    Args:
        tmp_path (Path): Temporary path fixture for generating test snapshot.
    """
    from ml_switcheroo_ir.cli import main as cli_main

    ir_file = tmp_path / "ir_v0.0.3.json"
    cli_main(["dump-snapshot", "--output", str(ir_file)])
    assert ir_file.exists()

    gv_ir = GroundingValidator(snapshot_manifest=str(ir_file))
    assert "ai.onnx.Relu" in gv_ir.grounded_symbols
    assert "ml.switcheroo.custom.RMSNorm" in gv_ir.grounded_symbols

    node_good = LogicalNode(
        id="n_good",
        op_type="RMSNorm",
        domain="ml.switcheroo.custom",
    )
    assert not gv_ir.validate_grounding(node_good)


def test_grounding_against_ml_framework_snapshots_golden_skip(tmp_path: Path) -> None:
    """Test skip branch when snapshot dataset is not present.

    Args:
        tmp_path (Path): Temporary path fixture.
    """
    with patch("pathlib.Path.glob", return_value=[]), pytest.raises(
        pytest.skip.Exception
    ):
        test_grounding_against_ml_framework_snapshots_golden(tmp_path)


def test_grounding_validator_multi_format_and_snapshots_dir(tmp_path: Path) -> None:
    """Test get_default_snapshots_dir resolution branches and multi-format snapshot ingestion.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    from ml_switcheroo_ir.validator import get_default_snapshots_dir

    # 1. Environment variable branch
    env_dir = tmp_path / "env_snapshots"
    env_dir.mkdir()
    with patch.dict(os.environ, {"ML_FRAMEWORK_SNAPSHOTS_DIR": str(env_dir)}):
        assert get_default_snapshots_dir() == str(env_dir)

    # 2. Package find_spec failure branch falling back to sibling directory
    with patch("importlib.util.find_spec", return_value=None):
        sibling_path = get_default_snapshots_dir()
        assert "ml-framework-snapshots" in sibling_path

    # Spec found with origin=None falling back to sibling directory
    spec_no_origin = MagicMock()
    spec_no_origin.origin = None
    with patch("importlib.util.find_spec", return_value=spec_no_origin):
        assert "ml-framework-snapshots" in get_default_snapshots_dir()

    # Spec found but snapshots dir does not exist
    fake_spec = MagicMock()
    fake_spec.origin = str(tmp_path / "nonexistent" / "__init__.py")
    with patch("importlib.util.find_spec", return_value=fake_spec):
        assert "ml-framework-snapshots" in get_default_snapshots_dir()

    # Spec found with empty snapshots dir (no .json/.json.gz) falling back to sibling
    pkg_empty = tmp_path / "fake_pkg_empty"
    pkg_empty_snaps = pkg_empty / "snapshots"
    pkg_empty_snaps.mkdir(parents=True)
    (pkg_empty_snaps / "readme.txt").write_text("not json", encoding="utf-8")
    spec_empty = MagicMock()
    spec_empty.origin = str(pkg_empty / "__init__.py")
    with patch("importlib.util.find_spec", return_value=spec_empty):
        assert "ml-framework-snapshots" in get_default_snapshots_dir()

    # Spec found with valid snapshots dir containing .json file
    pkg_valid_json = tmp_path / "fake_pkg_json"
    pkg_json_snaps = pkg_valid_json / "snapshots"
    pkg_json_snaps.mkdir(parents=True)
    (pkg_json_snaps / "manifest.json").write_text("{}", encoding="utf-8")
    spec_json = MagicMock()
    spec_json.origin = str(pkg_valid_json / "__init__.py")
    with patch("importlib.util.find_spec", return_value=spec_json):
        assert get_default_snapshots_dir() == os.path.abspath(str(pkg_json_snaps))

    # Spec found with valid snapshots dir containing .json.gz file
    pkg_valid_gz = tmp_path / "fake_pkg_gz"
    pkg_gz_snaps = pkg_valid_gz / "snapshots"
    pkg_gz_snaps.mkdir(parents=True)
    (pkg_gz_snaps / "manifest.json.gz").write_bytes(b"")
    spec_gz = MagicMock()
    spec_gz.origin = str(pkg_valid_gz / "__init__.py")
    with patch("importlib.util.find_spec", return_value=spec_gz):
        assert get_default_snapshots_dir() == os.path.abspath(str(pkg_gz_snaps))

    # 3. Exception in find_spec
    with patch("importlib.util.find_spec", side_effect=ValueError("spec error")):
        assert "ml-framework-snapshots" in get_default_snapshots_dir()

    # 4. Ingestion of _parameter_translations, operations list, and custom list format
    manifest_data = {
        "_parameter_translations": {"matmul": {"roles": {"lhs": {"torch": ["input"]}}}},
        "operations": [{"name": "hlo_op_1", "params": []}],
        "custom_category": [{"name": "custom_list_op", "params": []}],
        "rms_norm": {"torch": ["torch.nn.RMSNorm"]},
    }
    gv = GroundingValidator(snapshot_manifest=manifest_data)
    assert "matmul" in gv.parameter_translations
    assert "hlo_op_1" in gv.grounded_symbols
    assert "custom_list_op" in gv.grounded_symbols
    assert "rms_norm" in gv.concept_map
    assert "rms_norm" in gv.grounded_symbols
