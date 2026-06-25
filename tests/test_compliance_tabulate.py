"""Test tabulate fallback in compliance module."""

import sys
import pytest


def test_compliance_tabulate_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the fallback logic when tabulate is not installed.

    Args:
        monkeypatch: The monkeypatch fixture.
    """
    monkeypatch.setitem(sys.modules, "tabulate", None)

    # Force reload of compliance.py so it hits the ImportError
    import ml_switcheroo_ir.compliance
    import importlib

    try:
        importlib.reload(ml_switcheroo_ir.compliance)

        from ml_switcheroo_ir.compliance import tabulate

        res = tabulate([["a", "b"]], ["A", "B"])
        assert "A | B" in res
        assert "a | b" in res

        res = tabulate([["a", "b"]])
        assert "a | b" in res
    finally:
        # Restore normal tabulate if it exists
        monkeypatch.undo()
        importlib.reload(ml_switcheroo_ir.compliance)
