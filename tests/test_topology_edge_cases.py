"""Unit tests for cyclic, disconnected, and complex multi-output graph topologies."""

import pytest

from ml_switcheroo_ir import (
    CyclicGraphError,
    LogicalGraph,
    LogicalNode,
    topological_sort,
)


def test_topological_sort_direct_2_node_cycle() -> None:
    """Test direct 2-node cycle raising CyclicGraphError with strict=True and surviving with strict=False."""
    # A -> B -> A
    node_a = LogicalNode(id="a", op_type="Relu", inputs=["b"])
    node_b = LogicalNode(id="b", op_type="Relu", inputs=["a"])
    graph = LogicalGraph(name="DirectCycle", nodes={n.id: n for n in [node_a, node_b]})

    with pytest.raises(CyclicGraphError, match="Cycle detected"):
        topological_sort(graph, strict=True)

    # In non-strict mode, topological sort returns all nodes
    sorted_nodes = topological_sort(graph, strict=False)
    assert len(sorted_nodes) == 2
    assert {n.id for n in sorted_nodes} == {"a", "b"}


def test_topological_sort_5_node_indirect_cycle() -> None:
    """Test 5-node indirect cycle (c1 -> c2 -> c3 -> c4 -> c5 -> c1)."""
    c1 = LogicalNode(id="c1", op_type="Relu", inputs=["c5"])
    c2 = LogicalNode(id="c2", op_type="Relu", inputs=["c1"])
    c3 = LogicalNode(id="c3", op_type="Relu", inputs=["c2"])
    c4 = LogicalNode(id="c4", op_type="Relu", inputs=["c3"])
    c5 = LogicalNode(id="c5", op_type="Relu", inputs=["c4"])
    graph = LogicalGraph(
        name="IndirectCycle5", nodes={n.id: n for n in [c1, c2, c3, c4, c5]}
    )

    with pytest.raises(CyclicGraphError, match="Cycle detected"):
        topological_sort(graph, strict=True)

    sorted_nodes = topological_sort(graph, strict=False)
    assert len(sorted_nodes) == 5
    assert [n.id for n in sorted_nodes] == ["c1", "c2", "c3", "c4", "c5"]


def test_topological_sort_self_referential_cycle() -> None:
    """Test self-referential cycle where a node consumes its own output."""
    self_node = LogicalNode(id="self_op", op_type="Relu", inputs=["self_op"])
    graph = LogicalGraph(name="SelfCycle", nodes={self_node.id: self_node})

    with pytest.raises(CyclicGraphError, match="Cycle detected"):
        topological_sort(graph, strict=True)

    sorted_nodes = topological_sort(graph, strict=False)
    assert len(sorted_nodes) == 1
    assert sorted_nodes[0].id == "self_op"


def test_disconnected_subgraphs_and_isolated_nodes() -> None:
    """Test graph with disconnected components and isolated input nodes."""
    # Component 1: in1 -> relu1
    in1 = LogicalNode(id="in1", op_type="Input")
    relu1 = LogicalNode(id="relu1", op_type="Relu", inputs=["in1"])

    # Component 2: in2 -> relu2
    in2 = LogicalNode(id="in2", op_type="Input")
    relu2 = LogicalNode(id="relu2", op_type="Relu", inputs=["in2"])

    # Isolated node: isolated
    isolated = LogicalNode(id="isolated", op_type="Constant")

    graph = LogicalGraph(
        name="DisconnectedGraph",
        nodes={n.id: n for n in [in1, relu1, in2, relu2, isolated]},
    )

    sorted_nodes = topological_sort(graph, strict=True)
    sorted_ids = [n.id for n in sorted_nodes]

    # Roots should appear before their children
    assert sorted_ids.index("in1") < sorted_ids.index("relu1")
    assert sorted_ids.index("in2") < sorted_ids.index("relu2")
    assert "isolated" in sorted_ids
    assert len(sorted_ids) == 5


def test_dead_code_subgraphs() -> None:
    """Test graph containing dead-code subgraphs not connected to the main output."""
    # Main path: x -> y
    x = LogicalNode(id="x", op_type="Input")
    y = LogicalNode(id="y", op_type="Relu", inputs=["x"])

    # Dead path: dead_in -> dead_out
    dead_in = LogicalNode(id="dead_in", op_type="Input")
    dead_out = LogicalNode(id="dead_out", op_type="Relu", inputs=["dead_in"])

    graph = LogicalGraph(
        name="DeadCodeGraph",
        nodes={n.id: n for n in [x, y, dead_in, dead_out]},
        outputs=["y"],  # dead_out is not in outputs
    )

    sorted_nodes = topological_sort(graph, strict=True)
    sorted_ids = [n.id for n in sorted_nodes]
    assert sorted_ids.index("x") < sorted_ids.index("y")
    assert sorted_ids.index("dead_in") < sorted_ids.index("dead_out")
    assert graph.outputs == ["y"]


def test_multi_output_multiple_consumers_different_indices() -> None:
    """Test multi-output node resolution with multiple consumers consuming different output SSA indices."""
    # split produces out0, out1, out2
    split = LogicalNode(
        id="split_node",
        op_type="Split",
        outputs=["s_out0", "s_out1", "s_out2"],
    )

    # consumer0 consumes s_out0
    consumer0 = LogicalNode(id="c0", op_type="Relu", inputs=["s_out0"])
    # consumer1 consumes s_out1
    consumer1 = LogicalNode(id="c1", op_type="Gelu", inputs=["s_out1"])
    # consumer_joint consumes both s_out0 and s_out2
    consumer_joint = LogicalNode(
        id="c_joint", op_type="Add", inputs=["s_out0", "s_out2"]
    )

    graph = LogicalGraph(
        name="MultiOutputMultiConsumer",
        nodes={n.id: n for n in [split, consumer0, consumer1, consumer_joint]},
    )

    assert graph.get_output_producer("s_out0") == split
    assert graph.get_output_producer("s_out1") == split
    assert graph.get_output_producer("s_out2") == split

    sorted_nodes = topological_sort(graph, strict=True)
    sorted_ids = [n.id for n in sorted_nodes]

    # Producer must be sorted before all consumers
    split_idx = sorted_ids.index("split_node")
    assert split_idx < sorted_ids.index("c0")
    assert split_idx < sorted_ids.index("c1")
    assert split_idx < sorted_ids.index("c_joint")
