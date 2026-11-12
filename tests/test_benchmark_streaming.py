"""Streaming I/O and O(max_node_size) peak heap allocation benchmarks on connected Transformer DAGs."""

from __future__ import annotations

import io
import tracemalloc
from pathlib import Path

import pytest

from ml_switcheroo_ir import (
    DType,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
    TensorSpec,
)
from ml_switcheroo_ir.validator import Validator


class _SinkWriter(io.StringIO):
    """Discard text writer tracking total emitted UTF-8 character count without buffering in RAM.

    Attributes:
        total_chars (int): Total character count written to this sink.
    """

    total_chars: int

    def __init__(self) -> None:
        """Initialize _SinkWriter with zero character count."""
        super().__init__()
        self.total_chars = 0

    def write(self, s: str) -> int:
        """Record character count of chunk ``s`` without retaining ``s`` in memory.

        Args:
            s (str): Text chunk emitted by ``LogicalGraph.to_stream``.

        Returns:
            int: Length of ``s``.
        """
        n = len(s)
        self.total_chars += n
        return n


def build_connected_transformer_dag(num_layers: int = 32) -> LogicalGraph:
    """Construct a connected, schema-valid multi-layer Transformer DAG with residual skip edges.

    Each layer wires ``RMSNorm -> Q/K/V MatMul -> ScaledDotProductAttention -> Out MatMul -> Add ->
    RMSNorm -> Gate/Up MatMul -> GELU -> Mul -> Down MatMul -> Add``.

    Args:
        num_layers (int): Number of stacked Transformer layers.

    Returns:
        LogicalGraph: Connected, schema-valid multi-layer Transformer DAG.
    """
    nodes: dict[str, LogicalNode] = {}
    initializers: dict[str, bytes] = {}
    input_specs = {"tokens": TensorSpec(shape=(4, 128, 256), dtype=DType.float32)}
    hidden = "tokens"

    for layer in range(num_layers):
        pfx = f"l{layer}"
        for w_name in (
            f"{pfx}_ln1_w",
            f"{pfx}_wq",
            f"{pfx}_wk",
            f"{pfx}_wv",
            f"{pfx}_wo",
            f"{pfx}_ln2_w",
            f"{pfx}_wg",
            f"{pfx}_wu",
            f"{pfx}_wd",
        ):
            initializers[w_name] = b"\x00\x00\x80?" * 4

        ln1 = f"{pfx}_ln1"
        nodes[ln1] = LogicalNode(
            id=ln1,
            op_type="RMSNorm",
            domain="ml.switcheroo.custom",
            inputs=[hidden, f"{pfx}_ln1_w"],
            outputs=[ln1],
            attributes={"eps": 1e-6},
            shape_metadata=(4, 128, 256),
        )
        q_id, k_id, v_id = f"{pfx}_q", f"{pfx}_k", f"{pfx}_v"
        for proj_id, w_id in ((q_id, f"{pfx}_wq"), (k_id, f"{pfx}_wk"), (v_id, f"{pfx}_wv")):
            nodes[proj_id] = LogicalNode(
                id=proj_id,
                op_type="MatMul",
                domain="ai.onnx",
                inputs=[ln1, w_id],
                outputs=[proj_id],
                shape_metadata=(4, 128, 256),
                sharding=PartitionSpec(axes=("data", None, "tp")),
            )
        attn_id = f"{pfx}_sdpa"
        nodes[attn_id] = LogicalNode(
            id=attn_id,
            op_type="ScaledDotProductAttention",
            domain="ml.switcheroo.custom",
            inputs=[q_id, k_id, v_id],
            outputs=[attn_id],
            attributes={"is_causal": True},
            shape_metadata=(4, 128, 256),
        )
        out_id = f"{pfx}_attn_out"
        nodes[out_id] = LogicalNode(
            id=out_id,
            op_type="MatMul",
            domain="ai.onnx",
            inputs=[attn_id, f"{pfx}_wo"],
            outputs=[out_id],
            shape_metadata=(4, 128, 256),
        )
        res1_id = f"{pfx}_res1"
        nodes[res1_id] = LogicalNode(
            id=res1_id,
            op_type="Add",
            domain="ai.onnx",
            inputs=[hidden, out_id],
            outputs=[res1_id],
            shape_metadata=(4, 128, 256),
        )
        ln2 = f"{pfx}_ln2"
        nodes[ln2] = LogicalNode(
            id=ln2,
            op_type="RMSNorm",
            domain="ml.switcheroo.custom",
            inputs=[res1_id, f"{pfx}_ln2_w"],
            outputs=[ln2],
            attributes={"eps": 1e-6},
            shape_metadata=(4, 128, 256),
        )
        gate_id, up_id = f"{pfx}_gate", f"{pfx}_up"
        nodes[gate_id] = LogicalNode(
            id=gate_id,
            op_type="MatMul",
            domain="ai.onnx",
            inputs=[ln2, f"{pfx}_wg"],
            outputs=[gate_id],
            shape_metadata=(4, 128, 512),
        )
        nodes[up_id] = LogicalNode(
            id=up_id,
            op_type="MatMul",
            domain="ai.onnx",
            inputs=[ln2, f"{pfx}_wu"],
            outputs=[up_id],
            shape_metadata=(4, 128, 512),
        )
        act_id = f"{pfx}_gelu"
        nodes[act_id] = LogicalNode(
            id=act_id,
            op_type="GELU",
            domain="ml.switcheroo.custom",
            inputs=[gate_id],
            outputs=[act_id],
            shape_metadata=(4, 128, 512),
        )
        gated_id = f"{pfx}_gated"
        nodes[gated_id] = LogicalNode(
            id=gated_id,
            op_type="Mul",
            domain="ai.onnx",
            inputs=[act_id, up_id],
            outputs=[gated_id],
            shape_metadata=(4, 128, 512),
        )
        down_id = f"{pfx}_down"
        nodes[down_id] = LogicalNode(
            id=down_id,
            op_type="MatMul",
            domain="ai.onnx",
            inputs=[gated_id, f"{pfx}_wd"],
            outputs=[down_id],
            shape_metadata=(4, 128, 256),
        )
        res2_id = f"{pfx}_res2"
        nodes[res2_id] = LogicalNode(
            id=res2_id,
            op_type="Add",
            domain="ai.onnx",
            inputs=[res1_id, down_id],
            outputs=[res2_id],
            shape_metadata=(4, 128, 256),
        )
        hidden = res2_id

    return LogicalGraph(
        name="ConnectedTransformerDAG",
        nodes=nodes,
        inputs=["tokens"],
        input_specs=input_specs,
        initializers=initializers,
        outputs=[hidden],
        mesh=LogicalMesh(shape={"data": 2, "tp": 2}),
    )


def _measure_stream_write_peak_bytes(graph: LogicalGraph) -> tuple[int, int]:
    """Measure peak Python heap allocation during ``graph.to_stream`` via ``tracemalloc``.

    Args:
        graph (LogicalGraph): Connected DAG to stream out.

    Returns:
        tuple[int, int]: ``(peak_allocated_bytes, total_emitted_chars)``.
    """
    sink = _SinkWriter()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        graph.to_stream(sink)
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, sink.total_chars


@pytest.mark.benchmark
def test_streaming_peak_heap_is_bounded_by_max_node_size() -> None:
    """Verify via tracemalloc that ``to_stream`` peak heap scales O(max_node_size), not O(|V| + |E|)."""
    small_dag = build_connected_transformer_dag(num_layers=8)
    large_dag = build_connected_transformer_dag(num_layers=64)

    assert Validator().validate_edges(small_dag) == []
    assert len(large_dag.nodes) == 8 * len(small_dag.nodes)

    peak_small, chars_small = _measure_stream_write_peak_bytes(small_dag)
    peak_large, chars_large = _measure_stream_write_peak_bytes(large_dag)

    # Emitted JSON grows ~8x with layer count, while streaming heap overhead stays bounded (< 512 KiB)
    assert chars_large > 7 * chars_small
    assert peak_small < 512 * 1024
    assert peak_large < 512 * 1024
    # Streaming peak heap does NOT grow 8x when graph size grows 8x
    assert peak_large < max(peak_small * 3, 128 * 1024)


@pytest.mark.benchmark
def test_streaming_transformer_dag_json_gzip_and_zstd_roundtrip(tmp_path: Path) -> None:
    """Verify lossless streaming round-trips across .json, .json.gz, and .json.zst on a connected Transformer DAG.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
    """
    dag = build_connected_transformer_dag(num_layers=24)
    expected_nodes = 24 * 14
    assert len(dag.nodes) == expected_nodes
    assert len(dag.edges) > expected_nodes

    json_path = tmp_path / "transformer_dag.json"
    gz_path = tmp_path / "transformer_dag.json.gz"
    zst_path = tmp_path / "transformer_dag.json.zst"

    dag.to_file(json_path)
    dag.to_file(gz_path)
    dag.to_file(zst_path)

    assert gz_path.stat().st_size < json_path.stat().st_size
    assert zst_path.stat().st_size < json_path.stat().st_size

    for path in (json_path, gz_path, zst_path):
        loaded = LogicalGraph.from_file(path)
        assert len(loaded.nodes) == expected_nodes
        assert len(loaded.edges) == len(dag.edges)
        assert loaded.inputs == dag.inputs
        assert loaded.outputs == dag.outputs
        assert loaded.initializers.keys() == dag.initializers.keys()
        assert loaded.mesh == dag.mesh
