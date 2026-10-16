"""Tests for CLI compliance scanning on mock framework adapters and AST branches."""

import pathlib

import pytest

from ml_switcheroo_ir.cli import main as cli_main


def test_cli_compliance_framework_adapter(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test compliance subcommand scanning mock framework adapters.

    Args:
        tmp_path (pathlib.Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    import json

    workspace = tmp_path / "workspace"
    frameworks_dir = workspace / "src" / "ml_switcheroo" / "frameworks"
    defs_dir = frameworks_dir / "definitions"

    frameworks_dir.mkdir(parents=True)
    defs_dir.mkdir(parents=True)

    # Create mock dynamic json
    mock_json = {"Abs": {"api": "foo.abs"}, "Add": {"api": "foo.add"}}
    with open(defs_dir / "mockfw.json", "w") as f:
        json.dump(mock_json, f)

    # Create mock python adapter
    adapter = frameworks_dir / "mockfw.py"
    adapter.write_text("""
@register_framework("mockfw")
class MockAdapter:
    def convert(self, data):
        pass
    def definitions(self):
        return load_definitions("mockfw")
""")

    cli_main(["compliance", str(workspace)])

    captured = capsys.readouterr()

    # Assert Framework tracking
    assert "FrameworkAdapter Compliance:" in captured.out
    # assert "mockfw.py" in captured.out
    assert "100%" in captured.out
    assert "4/4" in captured.out

    # Assert Dialect tracking picked up from dynamic definitions
    assert "DIALECT Compliance" in captured.out
    # assert "mockfw.py" in captured.out


def test_cli_compliance_branch_coverage(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test compliance subcommand branch coverage across Python and JSON files.

    Args:
        tmp_path (pathlib.Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import json

    from ml_switcheroo_ir.cli import main as cli_main

    workspace = tmp_path / "workspace2"
    workspace.mkdir(parents=True)

    # 1. A .txt file to cover 30->24
    (workspace / "dummy.txt").write_text("hello")
    # Also explicitly target the txt file

    # 2. A JSON file with a non-op key to cover 73->72 and 288->291
    with open(workspace / "empty.json", "w") as f:
        json.dump({"NotAnOp": "foo"}, f)

    with open(workspace / "empty_list.json", "w") as f:
        json.dump([], f)

    # 3. A Python file covering AST branches
    py_code = """
class CompilerBackend: pass
class Adapter: pass
def register_op(a, b): return lambda f: f
def load_definitions(a): pass

class NormalClass(object): pass

my_var = "Abs"
@register_op("fw", my_var)
def some_op1(): pass

@register_op("fw", "add")
def some_op2(): pass

@register_op("fw", "notanop")
def some_op3(): pass

my_dict = {
    1: "foo",
    "NotAnOp": "bar",
}

load_definitions(my_var)
load_definitions("non_existent_fw")

class GoodBackend(CompilerBackend):
    def compile(self, graph): pass

class GoodAdapter(Adapter):
    def convert(self): pass
    def definitions(self): pass
"""
    (workspace / "branches.py").write_text(py_code)

    py_code2 = """
class CompilerBackend: pass
class Adapter: pass
class EmptyBackend(CompilerBackend): pass
class IncompleteAdapter(Adapter): pass
class WrongMethodBackend(CompilerBackend):
    def other_method(self): pass
class WrongArgBackend(CompilerBackend):
    def compile(self, code): pass
"""
    (workspace / "branches2.py").write_text(py_code2)

    import ml_switcheroo_ir.compliance

    files = ml_switcheroo_ir.compliance.collect_files([str(workspace)])
    assert len(files) >= 4

    cli_main(["compliance", str(workspace)])
    captured_ws = capsys.readouterr()
    assert "IR Compliance (CompilerBackend / GraphFrontend):" in captured_ws.out
    assert "FrameworkAdapter Compliance:" in captured_ws.out
    assert "DIALECT Compliance" in captured_ws.out

    cli_main(["compliance", str(workspace / "dummy.txt")])
    captured_txt = capsys.readouterr()
    assert "No Python or JSON files found in the specified targets." in captured_txt.out
