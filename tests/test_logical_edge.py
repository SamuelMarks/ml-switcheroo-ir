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
    with pytest.raises(KeyError, match="non_existent_target"):
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

    nodes_iterated = list(graph.nodes_list)
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
    graph = LogicalGraph(nodes={"split": split_node, "add": add_node, "if_node": if_node})

    graph.add_edge("split", "add", source_idx=1, target_idx=0, value_name="s1")
    graph.add_edge("ext_src", "add", source_idx=0, target_idx=1, value_name="ext_ssa")
    assert add_node.inputs == ["s1", "ext_ssa"]
    assert any(e.value_name == "s1" for e in graph.edges)

    cloned = graph.clone()
    assert any(e.value_name == "s1" for e in cloned.edges)
    dumped = graph.to_dict()
    nodes_dumped = dumped["nodes"]
    assert isinstance(nodes_dumped, dict)
    if_node_dumped = nodes_dumped["if_node"]
    assert isinstance(if_node_dumped, dict)
    subgraphs_dumped = if_node_dumped["subgraphs"]
    assert isinstance(subgraphs_dumped, dict)
    assert subgraphs_dumped["attr_meta"] == "scalar_meta"
    then_branch = subgraphs_dumped["then_branch"]
    assert isinstance(then_branch, dict)
    assert "edges" in then_branch
    restored = LogicalGraph.from_dict(dumped)
    assert any(e.value_name == "s1" for e in restored.edges)

    # NodeDict mutation methods: __delitem__, pop, update, clear
    del cloned.nodes["if_node"]
    assert "if_node" not in cloned.nodes
    popped = cloned.nodes.pop("add")
    assert popped.id == "add"
    assert cloned.nodes.pop("nonexistent", None) is None
    cloned.nodes.update({"new_n": LogicalNode(id="new_n", op_type="Relu")})
    assert "new_n" in cloned.nodes
    cloned.nodes.clear()
    assert len(cloned.nodes) == 0


def test_versioned_output_list_and_nodedict_full_invalidation() -> None:
    """Verify every VersionedOutputList and NodeDict mutator invalidates LogicalGraph._output_map."""
    from ml_switcheroo_ir import NodeDict, VersionedOutputList

    n1 = LogicalNode(id="n1", op_type="Split", outputs=["a0", "a1"])
    n2 = LogicalNode(id="n2", op_type="Split", outputs=["b0", "b1"])
    g = LogicalGraph(nodes={"n1": n1, "n2": n2}, outputs=["b1"])

    assert [n.id for n in g.iter_nodes()] == ["n1", "n2"]
    assert g.nodes[0] is n1
    assert g.nodes[1] is n2
    assert g.get_output_producer("b0") is n2

    # Mutate SECOND node's outputs via every mutating sequence method and reassignment
    n2.outputs[0] = "b0_set"
    assert g.get_output_producer("b0") is None
    assert g.get_output_producer("b0_set") is n2

    n2.outputs[0:1] = ["b0_slice1", "b0_slice2"]
    assert g.get_output_producer("b0_slice2") is n2

    n2.outputs[0:1] = "single_str_slice"
    assert g.get_output_producer("single_str_slice") is n2

    n2.outputs.insert(0, "b_ins")
    assert g.get_output_producer("b_ins") is n2

    n2.outputs.extend(["b_ext"])
    assert g.get_output_producer("b_ext") is n2

    popped_out = n2.outputs.pop()
    assert popped_out == "b_ext" and g.get_output_producer("b_ext") is None

    n2.outputs.remove("b_ins")
    assert g.get_output_producer("b_ins") is None

    n2.outputs.reverse()
    assert g.get_producing_output_index("b1") == (n2, 0)

    del n2.outputs[0]
    assert g.get_output_producer("b1") is None

    n2.outputs *= 2
    assert len(n2.outputs) >= 2

    n2.outputs.clear()
    assert g.get_output_producer("single_str_slice") is None

    # Reassigning .outputs on the SECOND node must invalidate g._output_map!
    n2.outputs = VersionedOutputList(["reassigned_b0"])
    assert g.get_output_producer("reassigned_b0") is n2

    # NodeDict kwargs init, iterable update, setdefault, popitem, and assignment revalidation
    nd_kw = NodeDict(k1=LogicalNode(id="k1", op_type="Relu"))
    assert nd_kw["k1"].id == "k1"
    nd_kw.update([("k2", LogicalNode(id="k2", op_type="Relu"))], k3=LogicalNode(id="k3", op_type="Relu"))
    assert len(nd_kw) == 3

    existing = g.nodes.setdefault("n1", LogicalNode(id="n1", op_type="Relu"))
    assert existing is n1
    with pytest.raises(TypeError, match="requires a LogicalNode default"):
        g.nodes.setdefault("missing_key")
    inserted = g.nodes.setdefault("n3", LogicalNode(id="n3", op_type="Relu", outputs=["c0"]))
    assert g.get_output_producer("c0") is inserted

    k_pop, v_pop = g.nodes.popitem()
    assert k_pop == "n3" and v_pop is inserted and g.get_output_producer("c0") is None

    g.nodes = g.nodes
    assert g.get_output_producer("reassigned_b0") is n2


def test_nodedict_merge_key_validation_and_add_edge_ports() -> None:
    """Test NodeDict |=, |, copy, eq, key mismatch ValueError, and add_edge port positioning."""
    import io

    from ml_switcheroo_ir import NodeDict

    a = LogicalNode(id="a", op_type="Input")
    split = LogicalNode(id="split", op_type="Split", inputs=["a"], outputs=["s0", "s1"])
    sub = LogicalNode(id="sub", op_type="Sub")
    add = LogicalNode(id="add", op_type="Add")
    barrier = LogicalNode(id="bar", op_type="Barrier", outputs=[])

    nd = NodeDict({"a": a})
    nd |= {"split": split}
    nd |= [("sub", sub)]
    merged = nd | {"add": add, "bar": barrier}
    assert set(merged.keys()) == {"a", "split", "sub", "add", "bar"}
    assert merged.copy() == merged
    assert merged == dict(merged)
    assert (merged == 42) is False
    assert "split" in repr(merged)

    with pytest.raises(ValueError, match="does not match LogicalNode.id"):
        nd["wrong_key"] = LogicalNode(id="actual_id", op_type="Relu")

    # Unconsumed Input pseudo-node is NOT inferred as a graph output
    g_only_in = LogicalGraph(nodes={"a": LogicalNode(id="a", op_type="Input")})
    assert g_only_in.outputs == []

    g = LogicalGraph(nodes=merged, outputs=["sub", "add"])
    # source_idx=0 on multi-output Split resolves to 's0'
    g.add_edge("split", "sub", source_idx=1, target_idx=1, value_name="v_s1")
    g.add_edge("split", "sub", source_idx=0, target_idx=0, value_name="v_s0")
    assert g.nodes["sub"].inputs == ["s0", "s1"]

    # Repeated operand Add(s0, s0) and out-of-range source_idx / zero-output barrier
    g.add_edge("split", "add", source_idx=0, target_idx=0)
    g.add_edge("split", "add", source_idx=0, target_idx=1)
    assert g.nodes["add"].inputs == ["s0", "s0"]

    g.add_edge("split", "bar", source_idx=5)
    g.add_edge("bar", "add", source_idx=0, value_name="bar_tok")
    assert "split:5" in g.nodes["bar"].inputs

    with pytest.raises(ValueError, match="non-negative integer"):
        g.add_edge("a", "add", target_idx=-1)
    streamed = LogicalGraph.from_stream(io.StringIO(g.to_json()))
    assert streamed.nodes["add"].inputs == ["s0", "s0", "bar_tok"]
