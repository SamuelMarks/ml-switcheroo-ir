"""Tests for verbose missing operations reporting and API mapping in CLI compliance."""

from __future__ import annotations

import pathlib

import pytest

from ml_switcheroo_ir.cli import main as cli_main


def test_cli_compliance_verbose(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test compliance subcommand verbose output.

    Args:
        tmp_path (pathlib.Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import ml_switcheroo_ir.compliance
    from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY

    original_get_dialect_ops = ml_switcheroo_ir.compliance.get_dialect_ops

    def mock_get_dialect_ops() -> set[str]:
        """Return dialect ops extended with a synthetic FakeOp.

        Returns:
            set[str]: Set of dialect operation names.
        """
        ops = original_get_dialect_ops()
        ops.add("FakeOp")
        return ops

    monkeypatch.setattr(ml_switcheroo_ir.compliance, "get_dialect_ops", mock_get_dialect_ops)

    # We just need to mock an incomplete dialect file so the verbose block triggers
    dialect_file = tmp_path / "my_dialect.py"
    dialect_file.write_text("def Abs(x):\n    pass\n")

    cli_main(["compliance", str(tmp_path), "--verbose"])
    captured = capsys.readouterr()

    assert "Verbose Missing Operations Report" in captured.out
    assert "### " in captured.out

    # Find some op we know is missing, e.g., 'Add' if it's in the registry
    dialect_ops = {k.split(".")[-1] for k in ONNX_REGISTRY}
    assert "Add" in dialect_ops
    assert "- [ ] **Add**" in captured.out
    assert "```json" in captured.out
    assert '"domain":' in captured.out
    assert '"inputs":' in captured.out


def test_cli_compliance_verbose_mapping(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test compliance subcommand verbose output with mapping.

    Args:
        tmp_path (pathlib.Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import importlib
    import json
    import types

    orig_import = importlib.import_module

    def mock_import_module(name: str, package: str | None = None) -> types.ModuleType | types.SimpleNamespace:
        """Mock import_module to avoid importing heavy third-party packages.

        Args:
            name (str): Module name to import.
            package (Optional[str]): Package anchor.

        Returns:
            types.ModuleType | types.SimpleNamespace: Mocked or actual imported module.
        """
        if name in ("tensorflow", "jax.numpy"):
            return types.SimpleNamespace(
                add=lambda x, y: x + y,
                exp=lambda x: x,
            )
        return orig_import(name, package)

    monkeypatch.setattr("importlib.import_module", mock_import_module)

    dialect_file = tmp_path / "my_dialect.py"
    dialect_file.write_text("def Abs(x):\n    pass\n")

    mapping_file = tmp_path / "mapping.json"
    mapping_data = {
        "Add": {"api": "os.path.join"},  # valid function with signature and doc
        "Sub": {"api": "builtins.int"},  # covers type/class branch
        "Mul": {"api": "tf.add"},  # tests tf. replacement
        "Div": {"api": "sys.stdout"},  # not callable / no signature
        "Mod": {"api": "os.name"},  # valid module attribute
        "Exp": {"api": "jnp.exp"},  # tests jnp. replacement
    }
    with open(mapping_file, "w") as f:
        json.dump(mapping_data, f)

    cli_main(["compliance", str(dialect_file), "--verbose", "--mapping", str(mapping_file)])
    captured = capsys.readouterr()

    assert "Verbose Missing Operations Report" in captured.out
    assert "Mapped API targets" in captured.out
    assert "os.path" in captured.out
    assert "join" in captured.out


def test_cli_compliance_verbose_mapping_empty(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test compliance subcommand verbose output with empty mapping.

    Args:
        tmp_path (pathlib.Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    import json

    dialect_file = tmp_path / "my_dialect.py"
    dialect_file.write_text("def Abs(x):\n    pass\n")

    mapping_file = tmp_path / "mapping.json"
    with open(mapping_file, "w") as f:
        json.dump({}, f)

    cli_main(["compliance", str(dialect_file), "--verbose", "--mapping", str(mapping_file)])
    captured = capsys.readouterr()

    assert "Verbose Missing Operations Report" in captured.out
    assert "No mappings found in" in captured.out


def test_cli_compliance_verbose_mapping_exception(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test compliance subcommand verbose output with mapping when getdoc raises exception.

    Args:
        tmp_path: Pytest fixture for temporary directory.
        capsys: Pytest fixture for capturing stdout/stderr.
        monkeypatch: Pytest fixture for monkeypatching.
    """
    import inspect
    import json
    from collections.abc import Callable

    def mock_getdoc(obj: Callable[[str], str] | type) -> str:
        """Mock inspect.getdoc to raise a RuntimeError.

        Args:
            obj (Callable[[str], str] | type): Inspected object.

        Returns:
            str: Never returns (raises RuntimeError).

        Raises:
            RuntimeError: Always raised to test error resilience.
        """
        del obj
        raise RuntimeError("Mocked exception")

    monkeypatch.setattr(inspect, "getdoc", mock_getdoc)

    dialect_file = tmp_path / "my_dialect.py"
    dialect_file.write_text("def Abs(x):\n    pass\n")

    mapping_file = tmp_path / "mapping.json"
    mapping_data = {
        "Add": {"api": "os.path.join"},  # valid function
    }
    with open(mapping_file, "w") as f:
        json.dump(mapping_data, f)

    cli_main(["compliance", str(dialect_file), "--verbose", "--mapping", str(mapping_file)])
    captured = capsys.readouterr()

    assert "Verbose Missing Operations Report" in captured.out
    assert "Mapped API targets" in captured.out


def test_compliance_snapshot_api_metadata_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test snapshot metadata fallback in compliance._resolve_mapped_api_row and helper formatters.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_ir.compliance import (
        _format_snapshot_param,
        _format_snapshot_signature,
        _lookup_snapshot_api_metadata,
        _resolve_mapped_api_row,
    )
    from ml_switcheroo_ir.schema.ghost import ExtendedGhostRef, GhostParam, ParameterKind

    p_var = GhostParam(name="args", kind=ParameterKind.VAR_POSITIONAL, annotation="int")
    p_kw = GhostParam(name="kwargs", kind=ParameterKind.VAR_KEYWORD, default="{}")
    assert _format_snapshot_param(p_var) == "*args: int"
    assert _format_snapshot_param(p_kw) == "**kwargs = {}"

    cls_ref = ExtendedGhostRef(name="MyClass", api_path="pkg.MyClass", kind="class", params=[])
    assert _format_snapshot_signature(cls_ref) == "Class/Type"

    fn_ref = ExtendedGhostRef(
        name="my_fn",
        api_path="pkg.my_fn",
        kind="function",
        params=[p_var, p_kw],
        returns_type="Tensor",
        docstring="A" * 160,
    )
    assert _format_snapshot_signature(fn_ref) == "(*args: int, **kwargs = {}) -> Tensor"

    no_doc_ref = ExtendedGhostRef(
        name="no_doc_fn",
        api_path="pkg.no_doc_fn",
        kind="function",
        params=[],
        docstring=None,
    )

    fake_engine_cls = MagicMock()
    fake_engine_inst = MagicMock()
    fake_engine_cls.return_value = fake_engine_inst

    with patch("ml_ecosystem_snapshots.grounding.engine.GroundingEngine", fake_engine_cls):
        fake_engine_inst.get_symbol.return_value = fn_ref
        sig, doc = _lookup_snapshot_api_metadata("pkg", "pkg.my_fn", "my_fn")
        assert sig == "(*args: int, **kwargs = {}) -> Tensor"
        assert doc.endswith("...")
        assert len(doc) == 150

        fake_engine_inst.get_symbol.return_value = no_doc_ref
        sig2, doc2 = _lookup_snapshot_api_metadata("pkg", "pkg.no_doc_fn", "no_doc_fn")
        assert sig2 == "()"
        assert doc2 == "No docstring available."

        row = _resolve_mapped_api_row("uninstalled_fw.submod.my_fn")
        assert row[5] == "()"

    with patch(
        "ml_ecosystem_snapshots.grounding.engine.GroundingEngine",
        side_effect=RuntimeError("engine error"),
    ):
        assert _lookup_snapshot_api_metadata("pkg", "pkg.fn", "fn") == (
            "Unknown",
            "No docstring available.",
        )

    # Branch where live module provides docstring, so only signature or neither needs snapshot update
    import types

    def _undocumented_fn(x: int) -> int:
        """Short doc.

        Args:
            x (int): Input.

        Returns:
            int: Output.
        """
        return x

    monkeypatch.setattr(
        "importlib.import_module",
        lambda name, package=None: types.SimpleNamespace(fn=_undocumented_fn),
    )
    monkeypatch.setattr(
        "inspect.signature",
        MagicMock(side_effect=ValueError("no sig")),
    )
    row_live_doc = _resolve_mapped_api_row("custom_mod.fn")
    assert row_live_doc[5] == "Built-in / No signature"
    assert row_live_doc[6] == "Short doc."
