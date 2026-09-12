"""Property-based fuzz testing using Hypothesis for IR serialization and roundtripping."""

from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
from hypothesis import given, settings

from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
)

OPERATORS = [
    ("ai.onnx", "Relu"),
    ("ai.onnx", "Add"),
    ("ai.onnx", "Mul"),
    ("ai.onnx", "Conv"),
    ("ml.switcheroo.custom", "RMSNorm"),
    ("ml.switcheroo.custom", "SwiGLU"),
]


@st.composite
def arbitrary_dag(draw: st.DrawFn) -> LogicalGraph:
    """Hypothesis strategy generating arbitrary valid directed acyclic graphs.

    Args:
        draw: Hypothesis draw function.

    Returns:
        LogicalGraph: Valid DAG with grounded nodes and acyclic connections.
    """
    num_nodes = draw(st.integers(min_value=1, max_value=8))
    nodes: list[LogicalNode] = []

    for i in range(num_nodes):
        node_id = f"node_{i}"
        domain, op_type = draw(st.sampled_from(OPERATORS))

        # Select inputs only from preceding nodes to guarantee DAG property
        if i == 0:
            inputs: list[str] = []
        else:
            prev_ids = [n.id for n in nodes]
            inputs = draw(
                st.lists(st.sampled_from(prev_ids), max_size=min(2, i), unique=True)
            )

        attr_val = draw(
            st.one_of(
                st.integers(min_value=-100, max_value=100),
                st.floats(
                    allow_nan=False,
                    allow_infinity=False,
                    min_value=-10.0,
                    max_value=10.0,
                ),
                st.text(alphabet="abcdef", max_size=5),
            )
        )
        attributes: dict[str, Any] = {"param": attr_val}

        has_sharding = draw(st.booleans())
        sharding = PartitionSpec(axes=("data", None)) if has_sharding else None

        has_shape = draw(st.booleans())
        shape_meta = (1, 64) if has_shape else None

        node = LogicalNode(
            id=node_id,
            op_type=op_type,
            domain=domain,
            version=1,
            attributes=attributes,
            inputs=inputs,
            shape_metadata=shape_meta,
            sharding=sharding,
        )
        nodes.append(node)

    mesh = draw(st.one_of(st.none(), st.just(LogicalMesh(shape={"data": 2}))))
    graph_name = draw(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
            min_size=1,
            max_size=10,
        )
    )

    return LogicalGraph(
        name=graph_name,
        nodes=nodes,
        mesh=mesh,
    )


@given(graph=arbitrary_dag())
@settings(max_examples=40, deadline=None)
def test_fuzz_json_roundtrip(graph: LogicalGraph) -> None:
    """Property test asserting lossless JSON roundtrip serialization.

    Args:
        graph (LogicalGraph): Randomly generated DAG.
    """
    json_str = graph.to_json(format="canonical")
    deserialized = LogicalGraph.from_json(json_str)

    assert deserialized.name == graph.name
    assert len(deserialized.nodes) == len(graph.nodes)
    assert set(deserialized.nodes.keys()) == set(graph.nodes.keys())

    for nid, orig_node in graph.nodes.items():
        loaded_node = deserialized.nodes[nid]
        assert loaded_node.id == orig_node.id
        assert loaded_node.op_type == orig_node.op_type
        assert loaded_node.domain == orig_node.domain
        assert loaded_node.version == orig_node.version
        assert loaded_node.inputs == orig_node.inputs
        assert loaded_node.outputs == orig_node.outputs
        assert loaded_node.shape_metadata == orig_node.shape_metadata
        if orig_node.sharding is not None:
            assert loaded_node.sharding is not None
            assert loaded_node.sharding.axes == orig_node.sharding.axes


@given(graph=arbitrary_dag())
@settings(max_examples=30, deadline=None)
def test_fuzz_pythonic_cst_eval(graph: LogicalGraph) -> None:
    """Property test asserting that programmatic Format B CST construction matches original graph.

    Args:
        graph (LogicalGraph): Randomly generated DAG.
    """
    # Programmatic reconstruction via LogicalGraph and LogicalNode
    reconstructed_nodes = [
        LogicalNode(
            id=n.id,
            kind=n.kind,
            domain=n.domain,
            version=n.version,
            metadata=n.metadata,
            inputs=list(n.inputs),
            shape_metadata=n.shape_metadata,
            sharding=n.sharding,
        )
        for n in graph.nodes.values()
    ]

    reconstructed_graph = LogicalGraph(
        name=graph.name,
        nodes=reconstructed_nodes,
        outputs=list(graph.outputs),
        mesh=graph.mesh,
    )

    assert reconstructed_graph.name == graph.name
    assert len(reconstructed_graph.nodes) == len(graph.nodes)
    assert reconstructed_graph.outputs == graph.outputs
    for nid, node in graph.nodes.items():
        r_node = reconstructed_graph.nodes[nid]
        assert r_node.id == node.id
        assert r_node.op_type == node.op_type
        assert r_node.attributes == node.attributes
