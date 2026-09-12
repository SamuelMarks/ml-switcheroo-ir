"""Throughput benchmark and compression integration tests for LogicalGraph.

Tests streaming JSON deserialization, compression formats (.gz, .zst),
and high-throughput roundtrip against 10,000+ node synthetic graphs.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import pytest

from ml_switcheroo_ir import (
    LogicalEdge,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
)


def create_synthetic_transformer_graph(num_nodes: int = 10005) -> LogicalGraph:
    """Create a synthetic multi-layer Transformer graph with 10,000+ nodes.

    Args:
        num_nodes (int): Target number of nodes (must be >= 10). Defaults to 10005.

    Returns:
        LogicalGraph: Synthetic deep Transformer graph.
    """
    nodes: dict[str, LogicalNode] = {}
    edges: list[LogicalEdge] = []

    # Initial input node
    input_node = LogicalNode(
        id="input_tokens",
        op_type="Input",
        domain="ml.switcheroo.custom",
        shape_metadata=("B", "T", 1024),
    )
    nodes[input_node.id] = input_node

    current_output = input_node.id
    node_counter = 1

    # Build transformer blocks (each block has 10 nodes)
    while node_counter < num_nodes - 2:
        layer_idx = node_counter // 10

        # 1. Pre-norm
        norm1_id = f"layer_{layer_idx}_norm1"
        nodes[norm1_id] = LogicalNode(
            id=norm1_id,
            op_type="RMSNorm",
            domain="ml.switcheroo.custom",
            inputs=[current_output],
            attributes={"eps": 1e-6},
            shape_metadata=("B", "T", 1024),
        )
        edges.append(LogicalEdge(source=current_output, target=norm1_id))

        # 2. QKV Projection
        qkv_id = f"layer_{layer_idx}_qkv"
        nodes[qkv_id] = LogicalNode(
            id=qkv_id,
            op_type="MatMul",
            domain="ai.onnx",
            inputs=[norm1_id],
            attributes={"transB": 1},
            shape_metadata=("B", "T", 3072),
        )
        edges.append(LogicalEdge(source=norm1_id, target=qkv_id))

        # 3. ScaledDotProductAttention
        attn_id = f"layer_{layer_idx}_attn"
        nodes[attn_id] = LogicalNode(
            id=attn_id,
            op_type="ScaledDotProductAttention",
            domain="ml.switcheroo.custom",
            inputs=[qkv_id],
            attributes={"is_causal": True},
            shape_metadata=("B", "T", 1024),
        )
        edges.append(LogicalEdge(source=qkv_id, target=attn_id))

        # 4. Attention Out Proj
        attn_out_id = f"layer_{layer_idx}_attn_out"
        nodes[attn_out_id] = LogicalNode(
            id=attn_out_id,
            op_type="MatMul",
            domain="ai.onnx",
            inputs=[attn_id],
            shape_metadata=("B", "T", 1024),
        )
        edges.append(LogicalEdge(source=attn_id, target=attn_out_id))

        # 5. Residual 1
        res1_id = f"layer_{layer_idx}_res1"
        nodes[res1_id] = LogicalNode(
            id=res1_id,
            op_type="Add",
            domain="ai.onnx",
            inputs=[current_output, attn_out_id],
            shape_metadata=("B", "T", 1024),
        )
        edges.append(LogicalEdge(source=current_output, target=res1_id))
        edges.append(LogicalEdge(source=attn_out_id, target=res1_id))

        # 6. Pre-norm 2
        norm2_id = f"layer_{layer_idx}_norm2"
        nodes[norm2_id] = LogicalNode(
            id=norm2_id,
            op_type="RMSNorm",
            domain="ml.switcheroo.custom",
            inputs=[res1_id],
            attributes={"eps": 1e-6},
            shape_metadata=("B", "T", 1024),
        )
        edges.append(LogicalEdge(source=res1_id, target=norm2_id))

        # 7. MLP Gate/Up Proj
        mlp_gate_id = f"layer_{layer_idx}_mlp_gate"
        nodes[mlp_gate_id] = LogicalNode(
            id=mlp_gate_id,
            op_type="MatMul",
            domain="ai.onnx",
            inputs=[norm2_id],
            shape_metadata=("B", "T", 4096),
        )
        edges.append(LogicalEdge(source=norm2_id, target=mlp_gate_id))

        # 8. Activation (GELU)
        act_id = f"layer_{layer_idx}_act"
        nodes[act_id] = LogicalNode(
            id=act_id,
            op_type="GELU",
            domain="ml.switcheroo.custom",
            inputs=[mlp_gate_id],
            shape_metadata=("B", "T", 4096),
        )
        edges.append(LogicalEdge(source=mlp_gate_id, target=act_id))

        # 9. MLP Down Proj
        mlp_down_id = f"layer_{layer_idx}_mlp_down"
        nodes[mlp_down_id] = LogicalNode(
            id=mlp_down_id,
            op_type="MatMul",
            domain="ai.onnx",
            inputs=[act_id],
            shape_metadata=("B", "T", 1024),
        )
        edges.append(LogicalEdge(source=act_id, target=mlp_down_id))

        # 10. Residual 2
        res2_id = f"layer_{layer_idx}_res2"
        nodes[res2_id] = LogicalNode(
            id=res2_id,
            op_type="Add",
            domain="ai.onnx",
            inputs=[res1_id, mlp_down_id],
            shape_metadata=("B", "T", 1024),
        )
        edges.append(LogicalEdge(source=res1_id, target=res2_id))
        edges.append(LogicalEdge(source=mlp_down_id, target=res2_id))

        current_output = res2_id
        node_counter += 10

    # Final Output node
    final_output = LogicalNode(
        id="logits",
        op_type="Output",
        domain="ml.switcheroo.custom",
        inputs=[current_output],
        shape_metadata=("B", "T", 32000),
    )
    nodes[final_output.id] = final_output
    edges.append(LogicalEdge(source=current_output, target=final_output.id))

    return LogicalGraph(
        name="SyntheticDeepTransformer",
        nodes=nodes,
        outputs=[final_output.id],
        mesh=LogicalMesh(shape={"data": 2, "tensor": 4}),
        edges=edges,
    )


def test_streaming_and_compressed_file_io(tmp_path: Path) -> None:
    """Test streaming to_stream, to_file, and from_file with raw, .gz, and .zst formats.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    node1 = LogicalNode(
        id="n1",
        op_type="Relu",
        domain="ai.onnx",
        sharding=PartitionSpec(axes=("data", None)),
    )
    node2 = LogicalNode(
        id="n2",
        op_type="MatMul",
        domain="ai.onnx",
        inputs=["n1"],
    )
    graph = LogicalGraph(
        name="TestStreaming",
        nodes=[node1, node2],
        outputs=["n2"],
        mesh=LogicalMesh(shape={"data": 4}),
    )

    # 1. Test streaming to StringIO and reading from StringIO stream
    buf = io.StringIO()
    graph.to_stream(buf)
    buf.seek(0)
    loaded_from_stream = LogicalGraph.from_json(buf)
    assert len(loaded_from_stream.nodes) == 2
    assert "n1" in loaded_from_stream.nodes
    assert loaded_from_stream.nodes["n1"].sharding is not None

    # Test legacy format streaming
    buf_legacy = io.StringIO()
    graph.to_stream(buf_legacy, format="legacy")
    assert "edges" not in buf_legacy.getvalue()

    # 2. Test raw JSON file I/O
    raw_json_file = tmp_path / "model.json"
    graph.to_file(raw_json_file, format="legacy")
    loaded_from_file = LogicalGraph.from_file(raw_json_file)
    assert loaded_from_file.name == "TestStreaming"
    assert len(loaded_from_file.nodes) == 2

    # Test explicit compression='none'
    raw_none_file = tmp_path / "model_none.json"
    graph.to_file(raw_none_file, compression="none")
    loaded_none = LogicalGraph.from_file(raw_none_file, compression="identity")
    assert loaded_none.name == "TestStreaming"

    # Test from_json with Path object
    loaded_from_path = LogicalGraph.from_json(raw_json_file)
    assert loaded_from_path.name == "TestStreaming"

    # Test from_json with string path on disk
    loaded_from_str_path = LogicalGraph.from_json(str(raw_json_file))
    assert loaded_from_str_path.name == "TestStreaming"

    # Test from_json with raw bytes
    raw_bytes = graph.to_json().encode("utf-8")
    loaded_from_bytes = LogicalGraph.from_json(raw_bytes)
    assert loaded_from_bytes.name == "TestStreaming"

    # 3. Test Gzip (.gz) file I/O
    gz_file = tmp_path / "model.json.gz"
    graph.to_file(gz_file)
    loaded_from_gz = LogicalGraph.from_file(gz_file)
    assert loaded_from_gz.name == "TestStreaming"
    assert len(loaded_from_gz.nodes) == 2

    # Test explicit compression="gzip"
    explicit_gz = tmp_path / "model_custom_gz"
    graph.to_file(explicit_gz, compression="gzip")
    loaded_explicit_gz = LogicalGraph.from_file(explicit_gz, compression="gz")
    assert loaded_explicit_gz.name == "TestStreaming"

    # 4. Test Zstandard (.zst) file I/O
    zst_file = tmp_path / "model.json.zst"
    graph.to_file(zst_file)
    loaded_from_zst = LogicalGraph.from_file(zst_file)
    assert loaded_from_zst.name == "TestStreaming"
    assert len(loaded_from_zst.nodes) == 2

    # Test explicit compression="zstd"
    explicit_zst = tmp_path / "model_custom_zst"
    graph.to_file(explicit_zst, compression="zstd")
    loaded_explicit_zst = LogicalGraph.from_file(explicit_zst, compression="zst")
    assert loaded_explicit_zst.name == "TestStreaming"


def test_file_io_error_handling(tmp_path: Path, monkeypatch: Any) -> None:
    """Test error handling for unsupported compression and missing libraries.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        monkeypatch (Any): Pytest monkeypatch fixture.
    """
    graph = LogicalGraph(name="ErrGraph", nodes=[LogicalNode(id="x", op_type="Input")])

    # Unsupported compression format
    with pytest.raises(ValueError, match="Unsupported compression format"):
        graph.to_file(tmp_path / "test.bad", compression="lz4")

    with pytest.raises(ValueError, match="Unsupported compression format"):
        LogicalGraph.from_file(tmp_path / "test.bad", compression="lz4")

    # Unsupported type in from_json
    with pytest.raises(TypeError, match="Unsupported source type"):
        LogicalGraph.from_json(12345)  # type: ignore[arg-type]

    # Missing zstandard library
    import ml_switcheroo_ir

    monkeypatch.setattr(ml_switcheroo_ir, "zstandard", None)

    with pytest.raises(ImportError, match="zstandard library is required"):
        graph.to_file(tmp_path / "test.zst")

    with pytest.raises(ImportError, match="zstandard library is required"):
        LogicalGraph.from_file(tmp_path / "test.zst")


def test_benchmark_10000_node_transformer_throughput(tmp_path: Path) -> None:
    """Benchmark serialization and deserialization throughput with 10,000+ nodes.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    # 1. Generate 10,000+ node synthetic Transformer graph
    t0 = time.perf_counter()
    graph = create_synthetic_transformer_graph(num_nodes=10005)
    gen_time = time.perf_counter() - t0
    num_nodes = len(graph.nodes)
    assert num_nodes >= 10000
    print(f"\nGenerated {num_nodes} node graph in {gen_time:.3f}s")

    # 2. Benchmark to_json()
    t_start = time.perf_counter()
    json_str = graph.to_json()
    to_json_time = time.perf_counter() - t_start
    to_json_rate = num_nodes / to_json_time
    print(f"to_json(): {to_json_time:.3f}s ({to_json_rate:,.0f} nodes/sec)")

    # 3. Benchmark from_json() from string
    t_start = time.perf_counter()
    deserialized_graph = LogicalGraph.from_json(json_str)
    from_json_time = time.perf_counter() - t_start
    from_json_rate = num_nodes / from_json_time
    print(f"from_json(str): {from_json_time:.3f}s ({from_json_rate:,.0f} nodes/sec)")
    assert len(deserialized_graph.nodes) == num_nodes

    # 4. Benchmark streaming to_file() and from_file() (uncompressed)
    json_path = tmp_path / "synth_transformer.json"
    t_start = time.perf_counter()
    graph.to_file(json_path)
    to_file_time = time.perf_counter() - t_start
    print(f"to_file(raw): {to_file_time:.3f}s")

    t_start = time.perf_counter()
    loaded_raw = LogicalGraph.from_file(json_path)
    from_file_time = time.perf_counter() - t_start
    print(f"from_file(raw stream): {from_file_time:.3f}s")
    assert len(loaded_raw.nodes) == num_nodes

    # 5. Benchmark compressed .json.gz file I/O
    gz_path = tmp_path / "synth_transformer.json.gz"
    t_start = time.perf_counter()
    graph.to_file(gz_path)
    to_gz_time = time.perf_counter() - t_start
    print(
        f"to_file(gz): {to_gz_time:.3f}s (size: {gz_path.stat().st_size / 1024:.1f} KB)"
    )

    t_start = time.perf_counter()
    loaded_gz = LogicalGraph.from_file(gz_path)
    from_gz_time = time.perf_counter() - t_start
    print(f"from_file(gz stream): {from_gz_time:.3f}s")
    assert len(loaded_gz.nodes) == num_nodes

    # 6. Benchmark compressed .json.zst file I/O
    zst_path = tmp_path / "synth_transformer.json.zst"
    t_start = time.perf_counter()
    graph.to_file(zst_path)
    to_zst_time = time.perf_counter() - t_start
    print(
        f"to_file(zst): {to_zst_time:.3f}s (size: {zst_path.stat().st_size / 1024:.1f} KB)"
    )

    t_start = time.perf_counter()
    loaded_zst = LogicalGraph.from_file(zst_path)
    from_zst_time = time.perf_counter() - t_start
    print(f"from_file(zst stream): {from_zst_time:.3f}s")
    assert len(loaded_zst.nodes) == num_nodes

    # Assert structural roundtrip equality
    assert loaded_zst.name == graph.name
    assert loaded_zst.outputs == graph.outputs
