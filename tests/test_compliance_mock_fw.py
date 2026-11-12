"""Tests for ComplianceReport model, static IR class scoring, and CLI compliance formatting."""

from __future__ import annotations

import json
import pathlib

import pytest

from ml_switcheroo_ir.cli import main as cli_main
from ml_switcheroo_ir.compliance import (
    ComplianceReport,
    analyze_python_file,
    collect_files,
    evaluate_compliance,
    format_compliance_report,
    get_dialect_ops,
    run_compliance_check,
)


def test_cli_compliance_framework_adapter(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify evaluate_compliance and CLI output on a FrameworkAdapter with dynamic JSON definitions.

    Args:
        tmp_path (pathlib.Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    workspace = tmp_path / "workspace"
    frameworks_dir = workspace / "src" / "ml_switcheroo" / "frameworks"
    defs_dir = frameworks_dir / "definitions"
    (workspace / "node_modules").mkdir(parents=True)
    (workspace / ".venv").mkdir(parents=True)
    frameworks_dir.mkdir(parents=True)
    defs_dir.mkdir(parents=True)

    mock_json = {"Abs": {"api": "foo.abs"}, "Add": {"api": "foo.add"}}
    (defs_dir / "mockfw.json").write_text(json.dumps(mock_json), encoding="utf-8")

    adapter = frameworks_dir / "mockfw.py"
    adapter.write_text(
        """@register_framework("mockfw")
class MockAdapter:
    def convert(self, data):
        pass
    def definitions(self):
        return load_definitions("mockfw")
""",
        encoding="utf-8",
    )

    report = run_compliance_check([str(workspace)])
    assert isinstance(report, ComplianceReport)
    assert report.found_fw is True
    assert report.fw_score == (4, 4)
    assert report.found_dialect is True
    assert set(report.implemented_ops) == {"Abs", "Add"}

    assert cli_main(["compliance", str(workspace)]) == 0
    captured = capsys.readouterr()
    assert "FrameworkAdapter Compliance:" in captured.out
    assert "4/4" in captured.out
    assert "DIALECT Compliance" in captured.out


def test_compliance_ir_scoring_and_multi_file_aggregation(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify 0/5..5/5 genuine IR scoring across qualified/subscripted bases and parameter names.

    Args:
        tmp_path (pathlib.Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    workspace = tmp_path / "workspace2"
    workspace.mkdir(parents=True)

    (workspace / "dummy.txt").write_text("hello", encoding="utf-8")
    (workspace / "empty.json").write_text(json.dumps({"NotAnOp": "foo"}), encoding="utf-8")
    (workspace / "empty_list.json").write_text(json.dumps([]), encoding="utf-8")
    (workspace / "broken.py").write_text(
        """def broken(:
""",
        encoding="utf-8",
    )

    dialect_ops = get_dialect_ops()
    assert analyze_python_file(str(workspace / "broken.py"), dialect_ops).ir_score == (0, 5)

    full_ir_py = workspace / "full_ir.py"
    full_ir_py.write_text(
        """class QualifiedBackend(sw_ir.CompilerBackend[str]):
    def compile(self, ir_graph):
        return ''

class QualifiedFrontend(sw_ir.GraphFrontend):
    @staticmethod
    def parse_to_graph(src_code):
        return None

class DynamicBase(get_base()):
    pass

my_var = 'Abs'
@register_op('fw', my_var)
def some_op1():
    pass

@register_op('fw', 'add')
def some_op2():
    pass

@register_op('fw', 'notanop')
def some_op3():
    pass

my_dict = {1: 'foo', 'NotAnOp': 'bar', 'Relu': 1}
load_definitions(my_var)
load_definitions('non_existent_fw')
class GoodAdapter:
    def convert(self): pass
    def definitions(self): pass
""",
        encoding="utf-8",
    )

    partial_ir_py = workspace / "partial_ir.py"
    partial_ir_py.write_text(
        """class EmptyBackend(CompilerBackend):
    pass
class ZeroPositionalBackend(CompilerBackend):
    def compile():
        pass
class StaticZeroArgBackend(CompilerBackend):
    @staticmethod
    def compile():
        pass
class ReceiverOnlyBackend(CompilerBackend):
    def compile(self):
        pass
class IncompleteAdapter:
    pass
""",
        encoding="utf-8",
    )

    partial_res = analyze_python_file(str(partial_ir_py), dialect_ops)
    assert partial_res.is_ir is True
    assert partial_res.ir_score == (4, 5)
    assert partial_res.fw_score == (2, 4)

    empty_only = workspace / "only_empty_backend.py"
    empty_only.write_text(
        """class B(CompilerBackend):
    def other(self): pass
""",
        encoding="utf-8",
    )
    assert analyze_python_file(str(empty_only), dialect_ops).ir_score == (2, 5)

    files = collect_files([str(workspace)])
    assert len(files) >= 5
    rep_ws = evaluate_compliance([str(workspace)])
    assert rep_ws.ir_score == (5, 5)

    rep_a = evaluate_compliance([str(full_ir_py), str(partial_ir_py)])
    rep_b = evaluate_compliance([str(partial_ir_py), str(full_ir_py)])
    assert rep_a.ir_score == (5, 5) and rep_b.ir_score == (5, 5)
    assert rep_a.fw_score == (4, 4) and rep_b.fw_score == (4, 4)
    assert {"Add", "Relu"}.issubset(set(rep_a.implemented_ops))
    assert "PASS" in format_compliance_report(rep_a)

    rep_txt = evaluate_compliance([str(workspace / "dummy.txt")])
    assert rep_txt.analyzed_files == []
    assert cli_main(["compliance", str(workspace / "dummy.txt")]) == 0
    assert "No Python or JSON files found in the specified targets." in capsys.readouterr().out
