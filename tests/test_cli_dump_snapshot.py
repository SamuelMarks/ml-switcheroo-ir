"""Tests for CLI dump-snapshot command and validate command enhancements."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_switcheroo_ir.cli import main as cli_main


def test_cli_dump_snapshot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test dump-snapshot command generating GhostRef schema JSON."""
    out_file = tmp_path / "snapshot.json"
    cli_main(["dump-snapshot", "--output", str(out_file)])

    captured = capsys.readouterr()
    assert f"Dumped snapshot to {out_file}" in captured.out
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify both ONNX and Custom operators are present
    assert "ai.onnx.Relu" in data
    assert "ml.switcheroo.custom.RMSNorm" in data
    assert "ml.switcheroo.custom.SwiGLU" in data
    assert "ml.switcheroo.custom.FlashAttention" in data

    rmsnorm_entry = data["ml.switcheroo.custom.RMSNorm"]
    assert rmsnorm_entry["name"] == "RMSNorm"
    assert rmsnorm_entry["api_path"] == "ml.switcheroo.custom.RMSNorm"
    assert any(p["name"] == "eps" for p in rmsnorm_entry["params"])
