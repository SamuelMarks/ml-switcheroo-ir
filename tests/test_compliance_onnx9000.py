"""Tests module."""

import pathlib

import pytest

from ml_switcheroo_ir.cli import main as cli_main
from ml_switcheroo_ir.compliance import run_compliance_check


def test_cli_compliance_no_targets(capsys: pytest.CaptureFixture) -> None:
    """Test."""
    with pytest.raises(SystemExit):
        cli_main(["compliance"])


def test_cli_compliance_empty_targets(capsys: pytest.CaptureFixture) -> None:
    """Test."""
    with pytest.raises(SystemExit):
        run_compliance_check([])


def test_cli_compliance_invalid_json(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    """Test."""
    f = tmp_path / "bad.json"
    f.write_text("{bad json")
    cli_main(["compliance", str(tmp_path)])
    # should not raise, just ignore


def test_cli_compliance_onnx9000_adapter_and_decorator(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    """Test."""
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
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    # simulate load_definitions falling back to adjacent folder
    """Test."""
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
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    """Test."""
    f1 = tmp_path / "f1.py"
    f2 = tmp_path / "f2.py"
    f1.write_text("def Abs(x): pass")
    f2.write_text("def Add(x): pass")
    cli_main(["compliance", str(f1), str(f2)])
    captured = capsys.readouterr()
    assert "Multiple Targets (2)" in captured.out
