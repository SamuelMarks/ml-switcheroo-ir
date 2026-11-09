"""Tests for PyTorch ATen and Python Data API (Array API) dialect schemas and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_ir.schema.framework_registries import (
    ARRAY_API_REGISTRY,
    ATEN_REGISTRY,
    CHEX_REGISTRY,
    GRAIN_REGISTRY,
    OPTAX_REGISTRY,
    ORBAX_REGISTRY,
    _load_framework_schema_file,
)
from ml_switcheroo_ir.validator import ValidationLevel, Validator


def test_aten_and_array_api_registries_loaded() -> None:
    """Verify that ATen, Array API, Chex, Grain, Orbax, and Optax registries load operations."""
    assert len(ATEN_REGISTRY) > 0
    assert len(ARRAY_API_REGISTRY) > 0
    assert len(CHEX_REGISTRY) > 0
    assert len(GRAIN_REGISTRY) > 0
    assert len(ORBAX_REGISTRY) > 0
    assert len(OPTAX_REGISTRY) > 0

    # Key ATen operators
    assert "add" in ATEN_REGISTRY
    assert "matmul" in ATEN_REGISTRY
    assert "relu" in ATEN_REGISTRY
    assert "layer_norm" in ATEN_REGISTRY
    assert "scaled_dot_product_attention" in ATEN_REGISTRY

    # Key Array API operators
    assert "abs" in ARRAY_API_REGISTRY
    assert "add" in ARRAY_API_REGISTRY
    assert "matmul" in ARRAY_API_REGISTRY
    assert "mean" in ARRAY_API_REGISTRY
    assert "sum" in ARRAY_API_REGISTRY

    # Key Chex, Grain, Orbax, Optax operators
    assert "assert_shape" in CHEX_REGISTRY
    assert "DataLoader" in GRAIN_REGISTRY
    assert "PyTreeCheckpointer" in ORBAX_REGISTRY
    assert "adamw" in OPTAX_REGISTRY


def test_validator_aten_and_array_api_validation() -> None:
    """Verify that Validator._get_schema and validate_kind accept valid and reject unknown ATen, Array API, Chex, Grain, Orbax, and Optax ops."""
    v = Validator(level=ValidationLevel.STRICT)

    # Valid ATen node
    n_aten = LogicalNode(
        id="aten1",
        op_type="relu",
        domain="aten",
        inputs=["x"],
        outputs=["out"],
    )
    schema_aten = v._get_schema(n_aten)
    assert schema_aten is not None
    assert schema_aten.name == "relu"
    assert v.validate_kind(n_aten) == []

    # Invalid ATen node
    n_bad_aten = LogicalNode(
        id="bad_aten",
        op_type="nonexistent_aten_op",
        domain="aten",
        inputs=["x"],
        outputs=["out"],
    )
    errs_aten = v.validate_kind(n_bad_aten)
    assert len(errs_aten) == 1
    assert "not found in domain 'aten'" in errs_aten[0].message

    # Valid Array API node
    n_aapi = LogicalNode(
        id="aapi1",
        op_type="matmul",
        domain="array_api",
        inputs=["x1", "x2"],
        outputs=["out"],
    )
    schema_aapi = v._get_schema(n_aapi)
    assert schema_aapi is not None
    assert schema_aapi.name == "matmul"
    assert v.validate_kind(n_aapi) == []

    # Invalid Array API node
    n_bad_aapi = LogicalNode(
        id="bad_aapi",
        op_type="nonexistent_array_api_op",
        domain="array_api",
        inputs=["x1", "x2"],
        outputs=["out"],
    )
    errs_aapi = v.validate_kind(n_bad_aapi)
    assert len(errs_aapi) == 1
    assert "not found in domain 'array_api'" in errs_aapi[0].message

    # Valid Chex, Grain, Orbax, Optax nodes
    for domain, op_type in [
        ("chex", "assert_shape"),
        ("grain", "DataLoader"),
        ("orbax", "PyTreeCheckpointer"),
        ("orbax.checkpoint", "CheckpointManager"),
        ("optax", "adamw"),
    ]:
        node = LogicalNode(id=f"n_{domain}", op_type=op_type, domain=domain)
        schema = v._get_schema(node)
        assert schema is not None
        assert schema.name == op_type
        assert v.validate_kind(node) == []


def test_missing_framework_schema_file(tmp_path: Path) -> None:
    """Verify graceful handling when framework JSON schema file does not exist.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    res = _load_framework_schema_file(
        "nonexistent.json",
        "dummy",
        json_path=tmp_path / "nonexistent.json",
    )
    assert res == {}


def test_cli_list_ops_ecosystem_domains(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that CLI list-ops includes chex, grain, orbax, and optax registries.

    Args:
        capsys (pytest.CaptureFixture[str]): Pytest output capture fixture.
    """
    from ml_switcheroo_ir.cli import main

    for domain, expected_op in (
        ("chex", "assert_shape"),
        ("grain", "DataLoader"),
        ("orbax", "PyTreeCheckpointer"),
        ("optax", "adamw"),
    ):
        main(["list-ops", "--domain", domain])
        captured = capsys.readouterr()
        assert expected_op in captured.out
        assert domain in captured.out
