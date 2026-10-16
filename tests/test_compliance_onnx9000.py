"""Tests for CLI compliance scanning of ONNX9000 importers, decorators, and multi-target paths."""

import pathlib

import pytest

from ml_switcheroo_ir.cli import main as cli_main
from ml_switcheroo_ir.compliance import run_compliance_check


def test_cli_compliance_no_targets(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI compliance subcommand exits when no target arguments are given.

    Args:
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    with pytest.raises(SystemExit):
        cli_main(["compliance"])


def test_cli_compliance_empty_targets(capsys: pytest.CaptureFixture[str]) -> None:
    """Test run_compliance_check exit behavior with warn=False and deprecation warning with warn=True.

    Args:
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    with pytest.raises(SystemExit):
        run_compliance_check([], warn=False)

    with pytest.warns(
        DeprecationWarning, match="ml_switcheroo_ir.compliance is deprecated"
    ), pytest.raises(SystemExit):
        run_compliance_check([], warn=True)
    captured = capsys.readouterr()
    assert "Error: No targets provided." in captured.out


def test_cli_compliance_invalid_json(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify compliance check gracefully ignores malformed JSON files.

    Args:
        tmp_path (pathlib.Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    f = tmp_path / "bad.json"
    f.write_text("{bad json")
    cli_main(["compliance", str(tmp_path)])
    # should not raise, just ignore


def test_cli_compliance_onnx9000_adapter_and_decorator(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify compliance detection of JaxprImporter and @register_op decorators.

    Args:
        tmp_path (pathlib.Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    f = tmp_path / "my_importer.py"
    f.write_text("""
class JaxprImporter:
    def import_func(self): pass
    def import_jaxpr(self): pass

@register_op("jax", "Add")
def _map_add(): pass

@register_op("jax", "abs")
def _map_abs(): pass
""")
    cli_main(["compliance", str(tmp_path)])
    captured = capsys.readouterr()
    assert "FrameworkAdapter Compliance:" in captured.out
    assert "100%" in captured.out
    assert "DIALECT Compliance" in captured.out
    assert "2/" in captured.out  # Add and Abs


def test_cli_compliance_dynamic_defs_adjacent(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify load_definitions resolves JSON definitions from an adjacent definitions/ folder.

    Args:
        tmp_path (pathlib.Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    d = tmp_path / "definitions"
    d.mkdir()
    (d / "jax.json").write_text('{"Add": {"api": "foo"}}')

    f = tmp_path / "my_fw.py"
    f.write_text("""
class MyAdapter:
    def definitions(self):
        return load_definitions("jax")
""")
    cli_main(["compliance", str(f)])
    captured = capsys.readouterr()
    assert "DIALECT Compliance" in captured.out
    assert "1/" in captured.out  # Add is in json


def test_cli_compliance_multiple_targets(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify compliance report aggregates multiple file targets under a Multiple Targets label.

    Args:
        tmp_path (pathlib.Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    f1 = tmp_path / "f1.py"
    f2 = tmp_path / "f2.py"
    f1.write_text("def Abs(x): pass")
    f2.write_text("def Add(x): pass")
    cli_main(["compliance", str(f1), str(f2)])
    captured = capsys.readouterr()
    assert "Multiple Targets (2)" in captured.out
