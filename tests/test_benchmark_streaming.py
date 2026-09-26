"""Benchmark and streaming I/O performance tests for LogicalGraph with >10,000 nodes."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ml_switcheroo_ir import (
    DType,
    LogicalGraph,
    LogicalNode,
    TensorSpec,
)


def _build_large_streaming_graph(num_nodes: int = 10005) -> LogicalGraph:
    """Build a synthetic graph with >10,000 nodes using explicit inputs/outputs.

    Args:
        num_nodes (int): Number of nodes in the synthetic graph.

    Returns:
        LogicalGraph: Constructed synthetic graph.
    """
    nodes: dict[str, LogicalNode] = {}
    prev_id = "input"

    # Input specification
    input_specs = {"input": TensorSpec(shape=(32, 512), dtype=DType.float32)}

    for i in range(num_nodes):
        node_id = f"node_{i}"
        nodes[node_id] = LogicalNode(
            id=node_id,
            op_type="Relu" if i % 2 == 0 else "Gelu",
            domain="ai.onnx",
            inputs=[prev_id],
            outputs=[node_id],
            output_specs=[TensorSpec(shape=(32, 512), dtype=DType.float32)],
        )
        prev_id = node_id

    return LogicalGraph(
        name="LargeStreamingNet",
        nodes=nodes,
        inputs=["input"],
        input_specs=input_specs,
        outputs=[prev_id],
    )


def test_streaming_json_roundtrip_10000_nodes(tmp_path: Path) -> None:
    """Benchmark and test streaming serialization/deserialization on 10,000+ nodes.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
    """
    graph = _build_large_streaming_graph(10005)
    assert len(graph.nodes) == 10005

    # 1. Test uncompressed JSON streaming roundtrip
    json_path = tmp_path / "large_graph.json"
    t0 = time.perf_counter()
    graph.to_file(json_path)
    write_time = time.perf_counter() - t0

    assert json_path.is_file()
    assert json_path.stat().st_size > 0

    t1 = time.perf_counter()
    loaded_graph = LogicalGraph.from_file(json_path)
    read_time = time.perf_counter() - t1

    assert len(loaded_graph.nodes) == 10005
    assert loaded_graph.inputs == ["input"]
    assert loaded_graph.outputs == ["node_10004"]
    assert write_time < 30.0
    assert read_time < 30.0

    # 2. Test gzip compressed streaming roundtrip (.json.gz)
    gz_path = tmp_path / "large_graph.json.gz"
    t2 = time.perf_counter()
    graph.to_file(gz_path)
    gz_write_time = time.perf_counter() - t2

    assert gz_path.is_file()
    assert gz_path.stat().st_size < json_path.stat().st_size

    t3 = time.perf_counter()
    loaded_gz_graph = LogicalGraph.from_file(gz_path)
    gz_read_time = time.perf_counter() - t3

    assert len(loaded_gz_graph.nodes) == 10005
    assert loaded_gz_graph.outputs == ["node_10004"]
    assert gz_write_time < 30.0
    assert gz_read_time < 30.0


def test_streaming_zstd_roundtrip_10000_nodes(tmp_path: Path) -> None:
    """Benchmark and test zstd streaming roundtrip on 10,000+ nodes if zstandard installed.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
    """
    import importlib.util

    if importlib.util.find_spec("zstandard") is None:
        pytest.skip("zstandard not installed, skipping .zst streaming test")

    graph = _build_large_streaming_graph(10005)
    zst_path = tmp_path / "large_graph.json.zst"

    t0 = time.perf_counter()
    graph.to_file(zst_path)
    zst_write_time = time.perf_counter() - t0

    assert zst_path.is_file()

    t1 = time.perf_counter()
    loaded_zst = LogicalGraph.from_file(zst_path)
    zst_read_time = time.perf_counter() - t1

    assert len(loaded_zst.nodes) == 10005
    assert loaded_zst.outputs == ["node_10004"]
    assert zst_write_time < 30.0
    assert zst_read_time < 30.0


def test_streaming_zstd_missing_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that missing zstandard gracefully triggers pytest.skip.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(pytest.skip.Exception):
        test_streaming_zstd_roundtrip_10000_nodes(tmp_path)
