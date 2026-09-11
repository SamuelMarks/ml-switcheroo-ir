"""Tests for CLI ground command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.cli import main as cli_main


def test_cli_ground_fully_grounded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI ground command on fully grounded graph."""
    snapshot_file = tmp_path / "snapshot.json"
    snapshot_data = {
        "torch.relu": {
            "name": "relu",
            "api_path": "torch.relu",
            "params": [{"name": "input"}],
        }
    }
    snapshot_file.write_text(json.dumps(snapshot_data), encoding="utf-8")

    graph = LogicalGraph(
        nodes={"n1": LogicalNode(id="n1", op_type="relu", domain="torch")}
    )
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(graph.to_json(), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["ground", str(graph_file), "--snapshots-dir", str(snapshot_file)])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "Grounding Audit: 1/1 nodes grounded" in captured.out
    assert "Graph is fully grounded against framework snapshots." in captured.out


def test_cli_ground_with_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI ground command with ungrounded nodes reporting errors."""
    snapshot_file = tmp_path / "snapshot.json"
    snapshot_data = {
        "torch.relu": {
            "name": "relu",
            "api_path": "torch.relu",
            "params": [{"name": "input"}],
        }
    }
    snapshot_file.write_text(json.dumps(snapshot_data), encoding="utf-8")

    graph = LogicalGraph(
        nodes={
            "n1": LogicalNode(id="n1", op_type="relu", domain="torch"),
            "n2": LogicalNode(id="n2", op_type="hallucinated_layer", domain="torch"),
        }
    )
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(graph.to_json(), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["ground", str(graph_file), "--snapshots-dir", str(snapshot_file)])
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "Grounding Audit: 1/2 nodes grounded" in captured.out
    assert "Ungrounded symbol 'hallucinated_layer'" in captured.out


def test_cli_ground_directory_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI ground command when --snapshots-dir points to a directory of snapshots."""
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    (snap_dir / "s1.json").write_text(
        json.dumps([{"name": "addf", "api_path": "arith.addf"}]), encoding="utf-8"
    )

    graph = LogicalGraph(
        nodes={"n1": LogicalNode(id="n1", op_type="addf", domain="arith")}
    )
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(graph.to_json(), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["ground", str(graph_file), "--snapshots-dir", str(snap_dir)])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "Grounding Audit: 1/1 nodes grounded" in captured.out
    assert "Graph is fully grounded against framework snapshots." in captured.out
