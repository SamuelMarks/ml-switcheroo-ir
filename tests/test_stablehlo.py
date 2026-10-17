"""Tests for StableHLO and MLIR schema registries and loaders."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from ml_switcheroo_ir.schema import mlir_registry, stablehlo
from ml_switcheroo_ir.snapshots import find_schema_file


def test_load_stablehlo_schemas_missing_file() -> None:
    """Test _load_stablehlo_schemas when JSON file does not exist."""
    with mock.patch("pathlib.Path.exists", return_value=False):
        stablehlo._load_stablehlo_schemas()


def test_load_mlir_schemas_missing_file() -> None:
    """Test _load_mlir_schemas when JSON file does not exist."""
    with mock.patch("pathlib.Path.exists", return_value=False):
        mlir_registry._load_mlir_schemas()


def test_find_schema_file_branches() -> None:
    """Test find_schema_file branches and fallback lookups."""
    with TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        test_file = tmppath / "dummy_ops.json"
        test_file.write_text("{}", encoding="utf-8")

        # 1. Override path exists
        assert find_schema_file("dummy_ops.json", override_path=test_file) == test_file
        assert (
            find_schema_file("dummy_ops.json", override_path=str(test_file))
            == test_file
        )

        # 2. Override path does not exist
        non_existent = tmppath / "missing.json"
        assert find_schema_file("dummy_ops.json", override_path=non_existent) is None

        # 3. Candidate path exists in DEFAULT_SNAPSHOT_DIR
        snap_sub = tmppath / "snapshots_dir"
        snap_sub.mkdir(parents=True, exist_ok=True)
        cand_file = snap_sub / "cand_ops.json"
        cand_file.write_text("{}", encoding="utf-8")
        with mock.patch(
            "ml_switcheroo_ir.snapshots.DEFAULT_SNAPSHOT_DIR", str(snap_sub)
        ):
            assert find_schema_file("cand_ops.json") == cand_file

            # 4. Parent candidate path exists in DEFAULT_SNAPSHOT_DIR.parent
            parent_file = tmppath / "parent_ops.json"
            parent_file.write_text("{}", encoding="utf-8")
            assert find_schema_file("parent_ops.json") == parent_file

            # 5. Sibling schema path fallback when not in DEFAULT_SNAPSHOT_DIR or parent
            fake_pkg_init = tmppath / "pkg" / "snapshots.py"
            sibling_schema_dir = tmppath / "pkg" / "schema"
            sibling_schema_dir.mkdir(parents=True, exist_ok=True)
            sibling_file = sibling_schema_dir / "sibling_ops.json"
            sibling_file.write_text("{}", encoding="utf-8")
            with mock.patch("ml_switcheroo_ir.snapshots.__file__", str(fake_pkg_init)):
                assert find_schema_file("sibling_ops.json") == sibling_file
                assert find_schema_file("nonexistent_schema.json") is None
