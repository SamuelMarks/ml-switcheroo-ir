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
        with mock.patch(
            "pathlib.Path.is_file",
            side_effect=lambda: True,
        ):
            res = find_schema_file("dummy_ops.json")
            assert res is not None

        # 4. Sibling schema path fallback
        def mock_is_file_sibling(self: Path) -> bool:
            """Check if path string contains schema."""
            return "schema" in str(self)

        with mock.patch.object(Path, "is_file", mock_is_file_sibling):
            res_sibling = find_schema_file("dummy_ops.json")
            assert res_sibling is not None
            assert "schema" in str(res_sibling)

        # 5. Neither candidate nor sibling exists
        with mock.patch.object(Path, "is_file", return_value=False):
            assert find_schema_file("nonexistent_schema.json") is None
