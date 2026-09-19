"""Tests for graph optimization passes: DCE, CSE, and Shape Propagation."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ml_switcheroo_ir import (
    LogicalEdge,
    LogicalGraph,
    LogicalNode,
    eliminate_common_subexpressions,
    eliminate_dead_nodes,
    propagate_shapes_and_constants,
)


def test_eliminate_dead_nodes_basic() -> None:
    """Verify that dead nodes are pruned while outputs and their dependencies are preserved."""
    n_in = LogicalNode(id="in1", op_type="Input", shape_metadata=(4, 4))
    n_live = LogicalNode(
        id="live1", op_type="Relu", inputs=["in1"], shape_metadata=(4, 4)
    )
    n_dead1 = LogicalNode(id="dead1", op_type="Neg", inputs=["in1"])
    n_dead2 = LogicalNode(id="dead2", op_type="Exp", inputs=["dead1"])

    graph = LogicalGraph(
        name="DeadCodeGraph",
        nodes={n.id: n for n in [n_in, n_live, n_dead1, n_dead2]},
        outputs=["live1"],
        edges=[
            LogicalEdge(source="in1", target="live1"),
            LogicalEdge(source="in1", target="dead1"),
            LogicalEdge(source="dead1", target="dead2"),
        ],
    )

    cleaned = eliminate_dead_nodes(graph)
    assert set(cleaned.nodes.keys()) == {"in1", "live1"}
    assert len(cleaned.edges) == 1
    assert cleaned.edges[0].source == "in1"
    assert cleaned.edges[0].target == "live1"
    assert cleaned.outputs == ["live1"]


def test_eliminate_dead_nodes_preserves_side_effects() -> None:
    """Verify that nodes with side-effects or custom call contracts are safely preserved."""
    n_in = LogicalNode(id="in1", op_type="Input")
    n_out = LogicalNode(id="out1", op_type="Relu", inputs=["in1"])
    n_print = LogicalNode(id="side_print", op_type="Print", inputs=["in1"])
    n_custom = LogicalNode(
        id="side_custom",
        op_type="custom_call",
        inputs=["in1"],
        attributes={"has_side_effect": True},
    )
    n_dead = LogicalNode(id="pure_dead", op_type="Tanh", inputs=["in1"])

    graph = LogicalGraph(
        nodes={n.id: n for n in [n_in, n_out, n_print, n_custom, n_dead]},
        outputs=["out1"],
    )

    cleaned = eliminate_dead_nodes(graph)
    assert "pure_dead" not in cleaned.nodes
    assert "out1" in cleaned.nodes
    assert "side_print" in cleaned.nodes
    assert "side_custom" in cleaned.nodes


def test_eliminate_dead_nodes_idempotence() -> None:
    """Verify that DCE is idempotent: eliminate_dead_nodes(g) == eliminate_dead_nodes(eliminate_dead_nodes(g))."""
    n_in = LogicalNode(id="in1", op_type="Input")
    n_mid = LogicalNode(id="mid1", op_type="Relu", inputs=["in1"])
    n_out = LogicalNode(id="out1", op_type="Add", inputs=["mid1", "in1"])
    n_dead = LogicalNode(id="dead1", op_type="GELU", inputs=["in1"])

    graph = LogicalGraph(
        nodes={n.id: n for n in [n_in, n_mid, n_out, n_dead]},
        outputs=["out1"],
    )

    once = eliminate_dead_nodes(graph)
    twice = eliminate_dead_nodes(once)

    assert set(once.nodes.keys()) == set(twice.nodes.keys())
    assert len(once.edges) == len(twice.edges)
    assert once.outputs == twice.outputs


@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
)
def test_eliminate_dead_nodes_property_idempotence(
    num_live: int, num_dead: int
) -> None:
    """Property-based fuzz testing verifying DCE idempotence across arbitrary graphs.

    Args:
        num_live (int): Number of live chained nodes.
        num_dead (int): Number of dead nodes.
    """
    nodes: list[LogicalNode] = []
    edges: list[LogicalEdge] = []

    # Build live chain
    prev_id = "in_0"
    nodes.append(LogicalNode(id=prev_id, op_type="Input"))
    for i in range(num_live):
        nid = f"live_{i}"
        nodes.append(LogicalNode(id=nid, op_type="Relu", inputs=[prev_id]))
        edges.append(LogicalEdge(source=prev_id, target=nid))
        prev_id = nid

    # Build dead nodes
    for j in range(num_dead):
        did = f"dead_{j}"
        nodes.append(LogicalNode(id=did, op_type="Tanh", inputs=["in_0"]))
        edges.append(LogicalEdge(source="in_0", target=did))

    graph = LogicalGraph(nodes={n.id: n for n in nodes}, outputs=[prev_id], edges=edges)

    pass1 = eliminate_dead_nodes(graph)
    pass2 = eliminate_dead_nodes(pass1)

    assert set(pass1.nodes.keys()) == set(pass2.nodes.keys())
    assert pass1.outputs == pass2.outputs
    assert len(pass1.edges) == len(pass2.edges)


def test_eliminate_common_subexpressions_basic() -> None:
    """Verify that duplicate pure operations are merged and consumer inputs updated."""
    n_in = LogicalNode(id="in1", op_type="Input", shape_metadata=(4, 4))
    # Two identical pure nodes
    n_relu1 = LogicalNode(
        id="relu1", op_type="Relu", inputs=["in1"], shape_metadata=(4, 4)
    )
    n_relu2 = LogicalNode(
        id="relu2", op_type="Relu", inputs=["in1"], shape_metadata=(4, 4)
    )
    # Consumer using both
    n_add = LogicalNode(
        id="add1", op_type="Add", inputs=["relu1", "relu2"], shape_metadata=(4, 4)
    )

    graph = LogicalGraph(
        nodes={n.id: n for n in [n_in, n_relu1, n_relu2, n_add]},
        outputs=["add1"],
    )

    cse_graph = eliminate_common_subexpressions(graph)
    assert len(cse_graph.nodes) == 3
    assert "relu1" in cse_graph.nodes
    assert "relu2" not in cse_graph.nodes

    # Consumer add1 should now consume relu1 twice
    assert cse_graph.nodes["add1"].inputs == ["relu1", "relu1"]


def test_eliminate_common_subexpressions_preserves_side_effects() -> None:
    """Verify that duplicate side-effecting operations are NOT merged."""
    n_in = LogicalNode(id="in1", op_type="Input")
    p1 = LogicalNode(id="p1", op_type="Print", inputs=["in1"])
    p2 = LogicalNode(id="p2", op_type="Print", inputs=["in1"])
    graph = LogicalGraph(nodes={n.id: n for n in [n_in, p1, p2]}, outputs=["p1", "p2"])

    cse_graph = eliminate_common_subexpressions(graph)
    assert len(cse_graph.nodes) == 3
    assert "p1" in cse_graph.nodes
    assert "p2" in cse_graph.nodes


def test_propagate_shapes_and_constants() -> None:
    """Verify static shape propagation across Shape, Reshape, Transpose, and BroadcastInDim."""
    n_in = LogicalNode(id="in1", op_type="Input", shape_metadata=(2, 4, 8))
    n_shape = LogicalNode(id="s1", op_type="Shape", inputs=["in1"])
    n_transpose = LogicalNode(
        id="t1",
        op_type="Transpose",
        inputs=["in1"],
        attributes={"perm": [2, 0, 1]},
    )
    n_reshape = LogicalNode(
        id="r1",
        op_type="Reshape",
        inputs=["in1"],
        attributes={"shape": [8, -1]},  # (2*4*8 = 64) -> (8, 8)
    )
    n_bcast = LogicalNode(
        id="b1",
        op_type="BroadcastInDim",
        inputs=["in1"],
        attributes={"output_shape": [2, 4, 8, 16]},
    )

    graph = LogicalGraph(
        nodes={n.id: n for n in [n_in, n_shape, n_transpose, n_reshape, n_bcast]},
        outputs=["r1"],
    )

    evaluated = propagate_shapes_and_constants(graph)
    assert evaluated.nodes["s1"].shape_metadata == (3,)
    assert evaluated.nodes["t1"].shape_metadata == (8, 2, 4)
    assert evaluated.nodes["r1"].shape_metadata == (8, 8)
    assert evaluated.nodes["b1"].shape_metadata == (2, 4, 8, 16)


def test_propagate_shapes_error_detection() -> None:
    """Verify detection of static shape mismatches during shape propagation."""
    # 1. Multiple -1 dimensions in Reshape
    n_in = LogicalNode(id="in1", op_type="Input", shape_metadata=(16,))
    n_bad_reshape1 = LogicalNode(
        id="r_bad1",
        op_type="Reshape",
        inputs=["in1"],
        attributes={"shape": [-1, -1]},
    )
    with pytest.raises(ValueError, match="multiple -1 dimensions"):
        propagate_shapes_and_constants(
            LogicalGraph(nodes={n.id: n for n in [n_in, n_bad_reshape1]})
        )

    # 2. Indivisible dimension in Reshape with -1
    n_bad_reshape2 = LogicalNode(
        id="r_bad2",
        op_type="Reshape",
        inputs=["in1"],
        attributes={"shape": [3, -1]},  # 16 is not divisible by 3!
    )
    with pytest.raises(ValueError, match="not divisible"):
        propagate_shapes_and_constants(
            LogicalGraph(nodes={n.id: n for n in [n_in, n_bad_reshape2]})
        )

    # 3. Explicit element count mismatch in Reshape without -1
    n_bad_reshape3 = LogicalNode(
        id="r_bad3",
        op_type="Reshape",
        inputs=["in1"],
        attributes={"shape": [4, 5]},  # 20 != 16
    )
    with pytest.raises(ValueError, match="mismatches target elements"):
        propagate_shapes_and_constants(
            LogicalGraph(nodes={n.id: n for n in [n_in, n_bad_reshape3]})
        )


def test_transforms_edge_cases() -> None:
    """Verify edge cases for DCE, CSE, and shape propagation passes."""
    # 1. DCE with missing output in graph.nodes, external input, duplicate inputs, and duplicate outputs
    n_node = LogicalNode(id="n1", op_type="Relu", inputs=["external_input"])
    n_dup = LogicalNode(id="n2", op_type="Add", inputs=["n1", "n1", "n1"])
    g_dce_edge = LogicalGraph(
        nodes={n.id: n for n in [n_node, n_dup]},
        outputs=["n2", "n2", "missing_node"],
    )
    res_dce = eliminate_dead_nodes(g_dce_edge)
    assert "n1" in res_dce.nodes
    assert "n2" in res_dce.nodes

    # 2. CSE with complex attributes (dicts and lists), no shape_metadata, and mutating op
    n_in = LogicalNode(id="in1", op_type="Input")
    c1 = LogicalNode(
        id="c1",
        op_type="CustomOp",
        inputs=["in1"],
        attributes={"dict_attr": {"k": "v", "nested": [1, 2]}, "list_attr": [3, 4]},
    )
    c2 = LogicalNode(
        id="c2",
        op_type="CustomOp",
        inputs=["in1"],
        attributes={"dict_attr": {"k": "v", "nested": [1, 2]}, "list_attr": [3, 4]},
    )
    mut_node = LogicalNode(
        id="mut1",
        op_type="CustomOp",
        inputs=["in1"],
        attributes={"is_mutating": True},
    )
    rng_node = LogicalNode(
        id="rng1",
        op_type="rng_bit_generator",
        inputs=["in1"],
    )
    g_cse_edge = LogicalGraph(
        nodes={n.id: n for n in [n_in, c1, c2, mut_node, rng_node]},
        outputs=["c1", "c2"],
    )
    res_cse = eliminate_common_subexpressions(g_cse_edge)
    assert len(res_cse.nodes) == 4
    assert "c1" in res_cse.nodes
    assert "c2" not in res_cse.nodes
    assert "mut1" in res_cse.nodes
    assert "rng1" in res_cse.nodes

    # 3. Shape propagation: Reshape without -1, new_shape alias, symbolic dimensions, and broadcast shape alias
    n_sym_in = LogicalNode(id="sym_in", op_type="Input", shape_metadata=("B", 16))
    n_reshape_exact = LogicalNode(
        id="r_exact",
        op_type="Reshape",
        inputs=["sym_in"],
        attributes={"new_shape": ["B", 4, 4]},
    )
    n_concrete_in = LogicalNode(id="c_in", op_type="Input", shape_metadata=(2, 8))
    n_reshape_concrete = LogicalNode(
        id="r_conc",
        op_type="Reshape",
        inputs=["c_in"],
        attributes={"shape": [4, 4]},  # exact match: 2*8 == 4*4
    )
    n_bcast_shape = LogicalNode(
        id="b_shape",
        op_type="broadcast_in_dim",
        inputs=["c_in"],
        attributes={"shape": [2, 8, 16]},
    )
    n_trans_bad_len = LogicalNode(
        id="t_bad",
        op_type="Transpose",
        inputs=["c_in"],
        attributes={"perm": [0]},  # Mismatched perm length!
        shape_metadata=(2, 8),
    )
    n_shape_empty = LogicalNode(id="s_empty", op_type="Shape", inputs=[])
    n_reshape_empty = LogicalNode(id="r_empty", op_type="Reshape", inputs=[])
    n_trans_empty = LogicalNode(id="t_empty", op_type="Transpose", inputs=[])
    n_bcast_empty = LogicalNode(id="b_empty", op_type="BroadcastInDim", inputs=[])

    n_unshaped_in = LogicalNode(id="unshaped_in", op_type="Input")
    n_shape_unshaped = LogicalNode(
        id="s_unshaped", op_type="Shape", inputs=["unshaped_in"]
    )
    n_reshape_unshaped = LogicalNode(
        id="r_unshaped",
        op_type="Reshape",
        inputs=["unshaped_in"],
        attributes={"shape": [4, 4]},
    )
    n_trans_unshaped = LogicalNode(
        id="t_unshaped",
        op_type="Transpose",
        inputs=["unshaped_in"],
        attributes={"perm": [0, 1]},
    )
    n_bcast_unshaped = LogicalNode(
        id="b_unshaped",
        op_type="BroadcastInDim",
        inputs=["unshaped_in"],
    )

    g_shape_edge = LogicalGraph(
        nodes={
            n.id: n
            for n in [
                n_sym_in,
                n_reshape_exact,
                n_concrete_in,
                n_reshape_concrete,
                n_bcast_shape,
                n_trans_bad_len,
                n_shape_empty,
                n_reshape_empty,
                n_trans_empty,
                n_bcast_empty,
                n_unshaped_in,
                n_shape_unshaped,
                n_reshape_unshaped,
                n_trans_unshaped,
                n_bcast_unshaped,
            ]
        },
    )
    res_shape = propagate_shapes_and_constants(g_shape_edge)
    assert res_shape.nodes["r_exact"].shape_metadata == ("B", 4, 4)
    assert res_shape.nodes["r_conc"].shape_metadata == (4, 4)
    assert res_shape.nodes["b_shape"].shape_metadata == (2, 8, 16)
    assert res_shape.nodes["t_bad"].shape_metadata == (2, 8)
