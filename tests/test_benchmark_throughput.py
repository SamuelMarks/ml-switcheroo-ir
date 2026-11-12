"""Throughput, memory-bound streaming, and compression integration tests on connected Transformer DAGs."""

from __future__ import annotations

import io
import json
import sys
import tracemalloc
from pathlib import Path

import pytest

from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
)
from tests.test_benchmark_streaming import build_connected_transformer_dag


def create_synthetic_transformer_graph(num_nodes: int = 280) -> LogicalGraph:
    """Create a connected multi-layer Transformer DAG with at least ``num_nodes`` nodes.

    Args:
        num_nodes (int): Target minimum number of nodes across stacked Transformer blocks.

    Returns:
        LogicalGraph: Connected, schema-valid multi-layer Transformer DAG.
    """
    num_layers = max(1, (num_nodes + 13) // 14)
    return build_connected_transformer_dag(num_layers=num_layers)


def test_streaming_and_compressed_file_io(tmp_path: Path) -> None:
    """Test streaming to_stream, to_file, and from_file with raw, .gz, and .zst formats.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    node1 = LogicalNode(
        id="n1",
        op_type="Relu",
        domain="ai.onnx",
        inputs=["x_in"],
        sharding=PartitionSpec(axes=("data", None)),
    )
    node2 = LogicalNode(
        id="n2",
        op_type="MatMul",
        domain="ai.onnx",
        inputs=["n1", "w_in"],
    )
    graph = LogicalGraph(
        name="TestStreaming",
        nodes={n.id: n for n in [node1, node2]},
        inputs=["x_in"],
        initializers={"w_in": [[1.0, 0.0], [0.0, 1.0]]},
        outputs=["n2"],
        mesh=LogicalMesh(shape={"data": 4}),
    )

    buf = io.StringIO()
    graph.to_stream(buf)
    buf.seek(0)
    loaded_from_stream = LogicalGraph.from_json(buf)
    assert len(loaded_from_stream.nodes) == 2
    assert loaded_from_stream.nodes["n1"].sharding is not None

    buf_custom = io.StringIO()
    graph.to_stream(buf_custom, indent=4)
    assert '"edges":' in buf_custom.getvalue()

    raw_json_file = tmp_path / "model.json"
    graph.to_file(raw_json_file)
    loaded_from_file = LogicalGraph.from_file(raw_json_file)
    assert loaded_from_file.name == "TestStreaming"
    assert len(loaded_from_file.nodes) == 2

    raw_none_file = tmp_path / "model_none.json"
    graph.to_file(raw_none_file, compression="none")
    loaded_none = LogicalGraph.from_file(raw_none_file, compression="identity")
    assert loaded_none.name == "TestStreaming"

    loaded_from_str_path = LogicalGraph.from_file(str(raw_json_file))
    assert loaded_from_str_path.name == "TestStreaming"
    with pytest.raises(json.JSONDecodeError):
        LogicalGraph.from_json(str(raw_json_file))

    raw_bytes = graph.to_json().encode("utf-8")
    loaded_from_bytes = LogicalGraph.from_json(raw_bytes)
    assert loaded_from_bytes.name == "TestStreaming"

    gz_file = tmp_path / "model.json.gz"
    graph.to_file(gz_file)
    loaded_from_gz = LogicalGraph.from_file(gz_file)
    assert loaded_from_gz.name == "TestStreaming"
    assert len(loaded_from_gz.nodes) == 2

    explicit_gz = tmp_path / "model_custom_gz"
    graph.to_file(explicit_gz, compression="gzip")
    loaded_explicit_gz = LogicalGraph.from_file(explicit_gz, compression="gz")
    assert loaded_explicit_gz.name == "TestStreaming"

    zst_file = tmp_path / "model.json.zst"
    graph.to_file(zst_file)
    loaded_from_zst = LogicalGraph.from_file(zst_file)
    assert loaded_from_zst.name == "TestStreaming"
    assert len(loaded_from_zst.nodes) == 2

    explicit_zst = tmp_path / "model_custom_zst"
    graph.to_file(explicit_zst, compression="zstd")
    loaded_explicit_zst = LogicalGraph.from_file(explicit_zst, compression="zst")
    assert loaded_explicit_zst.name == "TestStreaming"


def test_file_io_error_handling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error handling for unsupported compression and missing zstandard library.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    from ml_switcheroo_ir import _load_zstandard

    graph = LogicalGraph(name="ErrGraph", nodes={"x": LogicalNode(id="x", op_type="Input")}, outputs=["x"])

    with pytest.raises(ValueError, match="Unsupported compression format"):
        graph.to_file(tmp_path / "test.bad", compression="lz4")

    with pytest.raises(ValueError, match="Unsupported compression format"):
        LogicalGraph.from_file(tmp_path / "test.bad", compression="lz4")

    assert _load_zstandard() is not None
    monkeypatch.setitem(sys.modules, "zstandard", None)
    assert _load_zstandard() is None

    monkeypatch.setattr("ml_switcheroo_ir.serialization.zstandard", None)
    monkeypatch.setattr("ml_switcheroo_ir.graph.zstandard", None)

    with pytest.raises(ImportError, match="zstandard library is required"):
        graph.to_file(tmp_path / "test.zst")

    with pytest.raises(ImportError, match="zstandard library is required"):
        LogicalGraph.from_file(tmp_path / "test.zst")


@pytest.mark.benchmark
def test_benchmark_connected_transformer_throughput_and_memory(tmp_path: Path) -> None:
    """Verify multi-layer Transformer serialization round-trip integrity and bounded streaming memory.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    graph = create_synthetic_transformer_graph(num_nodes=560)
    num_nodes = len(graph.nodes)
    assert num_nodes == 560

    # Verify large single-node attribute payload dominates streaming write buffer O(max_node_size)
    large_blob = b"x" * (64 * 1024)
    graph.nodes["l0_ln1"].attributes["calibration_table"] = large_blob

    json_path = tmp_path / "synth_transformer.json"
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        graph.to_file(json_path)
        _cur_w, peak_write_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Peak heap during streaming write stays within a small constant multiple of max_node_size (64 KiB)
    assert peak_write_bytes < 1024 * 1024

    loaded_raw = LogicalGraph.from_file(json_path)
    assert len(loaded_raw.nodes) == num_nodes
    assert loaded_raw.nodes["l0_ln1"].attributes["calibration_table"] == large_blob

    gz_path = tmp_path / "synth_transformer.json.gz"
    zst_path = tmp_path / "synth_transformer.json.zst"
    graph.to_file(gz_path)
    graph.to_file(zst_path)

    loaded_gz = LogicalGraph.from_file(gz_path)
    loaded_zst = LogicalGraph.from_file(zst_path)
    assert len(loaded_gz.nodes) == num_nodes
    assert len(loaded_zst.nodes) == num_nodes
    assert loaded_zst.name == graph.name
    assert loaded_zst.outputs == graph.outputs
