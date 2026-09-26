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
    import importlib

    import ml_switcheroo_ir.compliance

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


def test_compliance_deprecation_warning() -> None:
    """Test that importing or reloading compliance emits a DeprecationWarning."""
    import importlib
    import warnings

    import ml_switcheroo_ir.compliance

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.reload(ml_switcheroo_ir.compliance)
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "ml_switcheroo_ir.compliance is deprecated" in str(w.message)
            for w in record
        )
