"""Tests for CLI dump-snapshot command and SnapshotEnvelope schema conformance across all registries."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_switcheroo_ir.cli import main as cli_main
from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope


def test_cli_dump_snapshot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test dump-snapshot exports all 21+ registries inside a valid SnapshotEnvelope.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    out_file = tmp_path / "snapshot.json"
    exit_code = cli_main(["dump-snapshot", "--output", str(out_file)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert f"Dumped snapshot to {out_file}" in captured.out
    assert out_file.exists()

    raw_text = out_file.read_text(encoding="utf-8")
    envelope = SnapshotEnvelope.model_validate_json(raw_text)
    assert envelope.schema_version == "2.0.0"
    assert envelope.target == "ml-switcheroo-ir"

    expected_categories = {
        "classes",
        "functions",
        "onnx",
        "custom",
        "stablehlo",
        "mlir",
        "rdna",
        "sass",
        "wgsl",
        "ptx",
        "metal",
        "wasm",
        "webgl",
        "aten",
        "array_api",
        "chex",
        "grain",
        "orbax",
        "optax",
        "collective",
        "quantization",
        "state",
        "odl",
    }
    assert expected_categories.issubset(set(envelope.categories.keys()))

    assert isinstance(envelope.operations, list)
    op_paths = {(entry["api_path"] if isinstance(entry, dict) else entry.api_path) for entry in envelope.operations}
    assert "ai.onnx.Relu" in op_paths
    assert "ml.switcheroo.custom.RMSNorm" in op_paths
    assert "ml.switcheroo.custom.SwiGLU" in op_paths
    assert "ml.switcheroo.custom.FlashAttention" in op_paths
    assert "stablehlo.dot_general" in op_paths
    assert "ml_switcheroo_ir.LogicalGraph" in op_paths
    assert "ml_switcheroo_ir.topological_sort" in op_paths


def test_cli_ast_symbol_extractor_edge_cases() -> None:
    """Test AST symbol and parameter extraction edge branches in cli.py."""
    import ast

    from ml_switcheroo_ir.cli import (
        _build_symbol_ghost_params,
        _extract_ast_args_to_ghost_params,
        _resolve_package_symbol_ast,
    )
    from ml_switcheroo_ir.schema.ghost import ParameterKind

    visited_set = {("ml_switcheroo_ir", "LogicalNode")}
    assert _resolve_package_symbol_ast("ml_switcheroo_ir", "LogicalNode", visited=visited_set) is None
    assert _resolve_package_symbol_ast("ml_switcheroo_ir.nonexistent_submod", "Foo") is None
    assert _build_symbol_ghost_params("NonExistentPublicSymbol") == []

    fn_src = """def f(a: int, /, b: str = 'x', *args: float, kw: bool = True, **kwargs: int) -> None:
    pass
"""
    fn_ast = ast.parse(fn_src).body[0]
    assert isinstance(fn_ast, ast.FunctionDef)
    extracted = _extract_ast_args_to_ghost_params(fn_ast.args, skip_self=False)
    kinds = {p.name: p.kind for p in extracted}
    assert kinds["a"] == ParameterKind.POSITIONAL_ONLY
    assert kinds["args"] == ParameterKind.VAR_POSITIONAL
    assert kinds["kw"] == ParameterKind.KEYWORD_ONLY
    assert kinds["kwargs"] == ParameterKind.VAR_KEYWORD
