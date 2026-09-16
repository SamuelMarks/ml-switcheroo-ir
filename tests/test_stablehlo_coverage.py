"""Tests for stablehlo module coverage."""

from __future__ import annotations

from unittest import mock

from ml_switcheroo_ir.schema import mlir_registry, stablehlo


def test_load_stablehlo_schemas_missing_file() -> None:
    """Test _load_stablehlo_schemas when JSON file does not exist."""
    with mock.patch("pathlib.Path.exists", return_value=False):
        stablehlo._load_stablehlo_schemas()


def test_load_mlir_schemas_missing_file() -> None:
    """Test _load_mlir_schemas when JSON file does not exist."""
    with mock.patch("pathlib.Path.exists", return_value=False):
        mlir_registry._load_mlir_schemas()
