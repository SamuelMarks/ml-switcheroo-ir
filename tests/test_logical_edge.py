"""Tests for LogicalEdge and graph topology enhancements."""

from __future__ import annotations

import pytest

from ml_switcheroo_ir import (
    CyclicGraphError,
    LogicalEdge,
    LogicalGraph,
    LogicalNode,
    topological_sort,
)


def test_logical_edge_init() -> None:
    """Test initialization and fields of LogicalEdge."""
    edge = LogicalEdge(source="conv1", target="relu1")
    assert edge.source == "conv1"
    assert edge.target == "relu1"
    assert edge.source_idx == 0
    assert edge.target_idx == 0
    assert edge.value_name is None
    assert edge.model_dump() == {
        "source": "conv1",
        "target": "relu1",
        "source_idx": 0,
        "target_idx": 0,
        "value_name": None,
    }


def test_logical_edge_equality() -> None:
    """Test value equality between LogicalEdge instances."""
    edge1 = LogicalEdge(source="a", target="b")
    edge2 = LogicalEdge(source="a", target="b")
    edge3 = LogicalEdge(source="a", target="c")
    assert edge1 == edge2
    assert edge1 != edge3


def test_graph_edges_property_getter() -> None:
    """Test constructing edges dynamically from node inputs."""
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Conv", inputs=["n1"])
    n3 = LogicalNode(id="n3", op_type="Relu", inputs=["n2", "n1"])
    graph = LogicalGraph(nodes={"n1": n1, "n2": n2, "n3": n3})

    edges = graph.edges
    assert len(edges) == 3
    assert LogicalEdge("n1", "n2", target_idx=0) in edges
    assert LogicalEdge("n2", "n3", target_idx=0) in edges
    assert LogicalEdge("n1", "n3", target_idx=1) in edges


def test_graph_add_edge_and_node_helpers() -> None:
    """Test add_node, add_edge, get_inputs, and get_outputs helper methods."""
    graph = LogicalGraph()
    n1 = LogicalNode(id="x", op_type="Input")
    n2 = LogicalNode(id="y", op_type="Linear")
    n3 = LogicalNode(id="z", op_type="Output")

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)

    assert len(graph) == 3
    assert graph["x"] == n1

    graph.add_edge("x", "y")
    graph.add_edge("x", "y")  # Deduplication
    graph.add_edge("y", "z")
    graph.add_edge("missing_source", "z")
    graph.add_edge("x", "non_existent_target")

    assert graph.get_inputs("y") == [n1]
    assert graph.get_outputs("x") == [n2]
    assert graph.get_inputs("non_existent") == []
    assert graph.get_outputs("z") == []


def test_graph_nodes_list_property() -> None:
    """Test that nodes_list returns all nodes in insertion order."""
    n1 = LogicalNode(id="a", op_type="Input")
    n2 = LogicalNode(id="b", op_type="Relu")
    graph = LogicalGraph(nodes={"a": n1, "b": n2})
    assert graph.nodes_list == [n1, n2]


def test_graph_iteration_and_indexing() -> None:
    """Test iteration protocol and dictionary-like indexing on LogicalGraph."""
    n1 = LogicalNode(id="a", op_type="Input")
    n2 = LogicalNode(id="b", op_type="Relu")
    graph = LogicalGraph(nodes={"a": n1, "b": n2})

    nodes_iterated = list(graph)
    assert nodes_iterated == [n1, n2]
    assert len(graph) == 2
    assert graph["a"] is n1
    assert graph["b"] is n2


def test_topological_sort_strict_cycle() -> None:
    """Test strict cycle detection raises CyclicGraphError."""
    n1 = LogicalNode("n1", "Node1", inputs=["n2"])
    n2 = LogicalNode("n2", "Node2", inputs=["n1"])
    graph = LogicalGraph(nodes={"n1": n1, "n2": n2})

    with pytest.raises(CyclicGraphError, match="Cycle detected"):
        topological_sort(graph, strict=True)

    # Non-strict mode should still handle it by appending
    nodes = topological_sort(graph, strict=False)
    assert len(nodes) == 2


def test_graph_edges_unresolved_colon_input() -> None:
    """Test edges property derivation when an input has colon syntax but no producing node."""
    node = LogicalNode(id="n1", op_type="Relu", inputs=["missing_node:3"])
    graph = LogicalGraph(nodes={"n1": node})
    edges = graph.edges
    assert len(edges) == 1
    assert edges[0].source == "missing_node:3"
    assert edges[0].source_idx == 0


def test_graph_edges_setter_synchronization() -> None:
    """Test LogicalGraph.edges setter synchronizing single and multi-output edges into node inputs."""
    split_node = LogicalNode(id="split", op_type="Split", outputs=["s0", "s1"])
    consumer_node = LogicalNode(id="cons", op_type="Add", inputs=["old_input"])
    graph = LogicalGraph(nodes={"split": split_node, "cons": consumer_node})

    graph.edges = [
        LogicalEdge(source="split", target="cons", source_idx=1),
        LogicalEdge(source="split", target="cons", source_idx=1),
        LogicalEdge(source="external_input", target="cons", source_idx=None),
        LogicalEdge(source="split", target="nonexistent_target", source_idx=0),
    ]
    assert graph.nodes["split"].inputs == []
    assert graph.nodes["cons"].inputs == ["s1", "external_input"]


def test_add_edge_multi_port_value_name_and_nodedict_mutations() -> None:
    """Test add_edge with source_idx, target_idx, value_name, and NodeDict mutation methods."""
    split_node = LogicalNode(id="split", op_type="Split", outputs=["s0", "s1"])
    add_node = LogicalNode(id="add", op_type="Add")
    sub_graph = LogicalGraph(
        name="SubG",
        nodes={"sub_in": LogicalNode(id="sub_in", op_type="Input")},
    )
    if_node = LogicalNode(
        id="if_node",
        op_type="If",
        subgraphs={"then_branch": sub_graph, "attr_meta": "scalar_meta"},
    )
    graph = LogicalGraph(
        nodes={"split": split_node, "add": add_node, "if_node": if_node}
    )

    graph.add_edge("split", "add", source_idx=1, target_idx=0, value_name="s1")
    graph.add_edge("ext_src", "add", source_idx=0, target_idx=1, value_name="ext_ssa")
    assert add_node.inputs == ["s1", "ext_ssa"]
    assert any(e.value_name == "s1" for e in graph.edges)

    cloned = graph.clone()
    assert any(e.value_name == "s1" for e in cloned.edges)
    dumped = graph.to_dict()
    assert dumped["nodes"]["if_node"]["subgraphs"]["attr_meta"] == "scalar_meta"
    then_branch = dumped["nodes"]["if_node"]["subgraphs"]["then_branch"]
    assert isinstance(then_branch, dict)
    assert "edges" in then_branch
    restored = LogicalGraph.from_dict(dumped)
    assert any(e.value_name == "s1" for e in restored.edges)

    # NodeDict mutation methods: __delitem__, pop, update, clear
    del cloned.nodes["if_node"]
    assert "if_node" not in cloned.nodes
    popped = cloned.nodes.pop("add")
    assert popped.id == "add"
    cloned.nodes.update({"new_n": LogicalNode(id="new_n", op_type="Relu")})
    assert "new_n" in cloned.nodes
    cloned.nodes.clear()
    assert len(cloned.nodes) == 0
