"""Tests for CLI ground command including .json, .json.gz, and .json.zst inputs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.cli import main as cli_main


def test_cli_ground_fully_grounded_and_compressed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI ground and toposort commands on .json, .json.gz, and .json.zst graph files.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    snapshot_file = tmp_path / "snapshot.json"
    snapshot_data = {
        "torch.relu": {
            "name": "relu",
            "api_path": "torch.relu",
            "params": [{"name": "input"}],
        }
    }
    snapshot_file.write_text(json.dumps(snapshot_data), encoding="utf-8")

    graph = LogicalGraph(nodes={"n1": LogicalNode(id="n1", op_type="relu", domain="torch")})
    for suffix in (".json", ".json.gz", ".json.zst"):
        graph_file = tmp_path / f"graph{suffix}"
        graph.to_file(graph_file)

        code = cli_main(["ground", str(graph_file), "--snapshots-dir", str(snapshot_file)])
        assert code == 0
        captured = capsys.readouterr()
        assert "Grounding Audit: 1/1 nodes grounded" in captured.out
        assert "Graph is fully grounded against framework snapshots." in captured.out

        sort_code = cli_main(["toposort", str(graph_file)])
        assert sort_code == 0
        assert "n1 (relu)" in capsys.readouterr().out


def test_cli_ground_with_diagnostics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI ground command with ungrounded nodes returning exit code 1.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
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

    code = cli_main(["ground", str(graph_file), "--snapshots-dir", str(snapshot_file)])
    assert code == 1

    captured = capsys.readouterr()
    assert "Grounding Audit: 1/2 nodes grounded" in captured.out
    assert "Ungrounded symbol 'hallucinated_layer'" in captured.out


def test_cli_ground_directory_and_default_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI ground command with directory --snapshots-dir and default snapshot discovery.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest stdout/stderr capture fixture.
    """
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    (snap_dir / "s1.json").write_text(json.dumps([{"name": "addf", "api_path": "arith.addf"}]), encoding="utf-8")

    graph = LogicalGraph(nodes={"n1": LogicalNode(id="n1", op_type="addf", domain="arith")})
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(graph.to_json(), encoding="utf-8")

    assert cli_main(["ground", str(graph_file), "--snapshots-dir", str(snap_dir)]) == 0
    assert "Grounding Audit: 1/1 nodes grounded" in capsys.readouterr().out

    with patch("ml_switcheroo_ir.snapshots.get_effective_snapshot_dir", return_value=str(snap_dir)):
        assert cli_main(["ground", str(graph_file)]) == 0
    assert "Graph is fully grounded against framework snapshots." in capsys.readouterr().out
