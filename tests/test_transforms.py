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
    n_live = LogicalNode(id="live1", op_type="Relu", inputs=["in1"], shape_metadata=(4, 4))
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
def test_eliminate_dead_nodes_property_idempotence(num_live: int, num_dead: int) -> None:
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
    n_relu1 = LogicalNode(id="relu1", op_type="Relu", inputs=["in1"], shape_metadata=(4, 4))
    n_relu2 = LogicalNode(id="relu2", op_type="Relu", inputs=["in1"], shape_metadata=(4, 4))
    # Consumer using both
    n_add = LogicalNode(id="add1", op_type="Add", inputs=["relu1", "relu2"], shape_metadata=(4, 4))

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
    """Verify that duplicate side-effecting operations are NOT merged and subgraphs are traversed."""
    sub_graph = LogicalGraph(
        name="Sub",
        nodes={
            "s_in": LogicalNode(id="s_in", op_type="Input"),
            "s_r1": LogicalNode(id="s_r1", op_type="Relu", inputs=["s_in"]),
            "s_r2": LogicalNode(id="s_r2", op_type="Relu", inputs=["s_in"]),
        },
        outputs=["s_r1", "s_r2"],
    )
    n_in = LogicalNode(id="in1", op_type="Input")
    p1 = LogicalNode(
        id="p1",
        op_type="Print",
        inputs=["in1"],
        subgraphs={"body": sub_graph, "tag": "raw_str"},
    )
    p2 = LogicalNode(id="p2", op_type="Print", inputs=["in1"])
    graph = LogicalGraph(nodes={n.id: n for n in [n_in, p1, p2]}, outputs=["p1", "p2"])

    cse_graph = eliminate_common_subexpressions(graph)
    assert len(cse_graph.nodes) == 3
    assert "p1" in cse_graph.nodes
    assert "p2" in cse_graph.nodes
    assert isinstance(cse_graph.nodes["p1"].subgraphs["body"], LogicalGraph)
    assert len(cse_graph.nodes["p1"].subgraphs["body"].nodes) == 2
    assert cse_graph.nodes["p1"].subgraphs["tag"] == "raw_str"


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
        propagate_shapes_and_constants(LogicalGraph(nodes={n.id: n for n in [n_in, n_bad_reshape1]}))

    # 2. Indivisible dimension in Reshape with -1
    n_bad_reshape2 = LogicalNode(
        id="r_bad2",
        op_type="Reshape",
        inputs=["in1"],
        attributes={"shape": [3, -1]},  # 16 is not divisible by 3!
    )
    with pytest.raises(ValueError, match="not divisible"):
        propagate_shapes_and_constants(LogicalGraph(nodes={n.id: n for n in [n_in, n_bad_reshape2]}))

    # 3. Explicit element count mismatch in Reshape without -1
    n_bad_reshape3 = LogicalNode(
        id="r_bad3",
        op_type="Reshape",
        inputs=["in1"],
        attributes={"shape": [4, 5]},  # 20 != 16
    )
    with pytest.raises(ValueError, match="mismatches target elements"):
        propagate_shapes_and_constants(LogicalGraph(nodes={n.id: n for n in [n_in, n_bad_reshape3]}))


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
    n_shape_unshaped = LogicalNode(id="s_unshaped", op_type="Shape", inputs=["unshaped_in"])
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


def test_dce_and_cse_multi_output_and_self_edges() -> None:
    """Verify DCE and CSE preserve multi-output SSA producers, self-edges, version, and source_ast_ref."""
    n_in = LogicalNode(
        id="in0",
        op_type="Input",
        version=7,
        source_ast_ref="model.py:10",
    )
    n_split1 = LogicalNode(
        id="split1",
        op_type="Split",
        version=13,
        inputs=["in0"],
        outputs=["q1", "k1"],
        source_ast_ref="model.py:20",
    )
    n_split2 = LogicalNode(
        id="split2",
        op_type="Split",
        version=13,
        inputs=["in0"],
        outputs=["q2", "k2"],
        source_ast_ref="model.py:21",
    )
    n_add = LogicalNode(
        id="self_add",
        op_type="Add",
        version=14,
        inputs=["k2", "k2"],
        outputs=["sum_out"],
        source_ast_ref="model.py:30",
    )
    n_dead = LogicalNode(
        id="dead_op",
        op_type="Relu",
        inputs=["q1"],
    )

    graph = LogicalGraph(
        name="MultiOutSelfEdgeGraph",
        nodes={
            "in0": n_in,
            "split1": n_split1,
            "split2": n_split2,
            "self_add": n_add,
            "dead_op": n_dead,
        },
        inputs=["in0"],
        outputs=["sum_out"],
    )

    cse_graph = eliminate_common_subexpressions(graph)
    assert "split1" in cse_graph.nodes
    assert "split2" not in cse_graph.nodes
    assert cse_graph.nodes["self_add"].inputs == ["k1", "k1"]
    assert len([e for e in cse_graph.edges if e.target == "self_add"]) == 2
    assert cse_graph.nodes["self_add"].version == 14
    assert cse_graph.nodes["self_add"].source_ast_ref == "model.py:30"

    dce_graph = eliminate_dead_nodes(cse_graph)
    assert "dead_op" not in dce_graph.nodes
    assert "split1" in dce_graph.nodes
    assert "in0" in dce_graph.nodes
    assert "self_add" in dce_graph.nodes
    assert dce_graph.outputs == ["sum_out"]
    assert dce_graph.nodes["split1"].version == 13
    assert dce_graph.nodes["split1"].source_ast_ref == "model.py:20"


def test_cse_control_flow_subgraphs_and_multi_output_consumers() -> None:
    """Verify CSE on control-flow nodes with distinct vs identical subgraphs and multi-output consumers."""
    # CSE on control-flow nodes with distinct vs identical subgraphs
    sub_a = LogicalGraph(
        name="sub_a",
        nodes={"op1": LogicalNode(id="op1", op_type="Relu", inputs=["x"])},
        inputs=["x"],
        outputs=["op1"],
    )
    sub_b = LogicalGraph(
        name="sub_b",
        nodes={"op1": LogicalNode(id="op1", op_type="Sigmoid", inputs=["x"])},
        inputs=["x"],
        outputs=["op1"],
    )
    cf_g = LogicalGraph(
        nodes={
            "x": LogicalNode(id="x", op_type="Input"),
            "if1": LogicalNode(
                id="if1",
                op_type="If",
                inputs=["x"],
                subgraphs={"then_branch": sub_a},
            ),
            "if2": LogicalNode(
                id="if2",
                op_type="If",
                inputs=["x"],
                subgraphs={"then_branch": sub_b},
            ),
            "if3": LogicalNode(
                id="if3",
                op_type="If",
                inputs=["x"],
                subgraphs={"then_branch": sub_a},
            ),
        },
        inputs=["x"],
        outputs=["if1", "if2", "if3"],
    )
    cse_cf = eliminate_common_subexpressions(cf_g)
    assert "if1" in cse_cf.nodes
    assert "if2" in cse_cf.nodes
    assert "if3" not in cse_cf.nodes
    assert cse_cf.outputs == ["if1", "if2", "if1"]

    # Multi-output Split shape propagation populates output_specs for unequal splits
    split_g = LogicalGraph(
        nodes={
            "x": LogicalNode(id="x", op_type="Input", shape_metadata=(4, 30)),
            "sp": LogicalNode(
                id="sp",
                op_type="Split",
                inputs=["x"],
                outputs=["sp0", "sp1"],
                attributes={"axis": 1, "split": [10, 20]},
            ),
        },
        inputs=["x"],
        outputs=["sp0", "sp1"],
    )
    prop_sp = propagate_shapes_and_constants(split_g)
    assert len(prop_sp.nodes["sp"].output_specs) == 2
    assert prop_sp.nodes["sp"].output_specs[0].shape == (4, 10)
    assert prop_sp.nodes["sp"].output_specs[1].shape == (4, 20)

    # Multi-output Split with num_outputs (no split list attribute)
    split_num_g = LogicalGraph(
        nodes={
            "x": LogicalNode(id="x", op_type="Input", shape_metadata=(4, 30)),
            "sp_n": LogicalNode(
                id="sp_n",
                op_type="Split",
                inputs=["x"],
                outputs=["sp0", "sp1"],
                attributes={"axis": 1, "num_outputs": 2},
            ),
        },
        inputs=["x"],
        outputs=["sp0", "sp1"],
    )
    prop_sp_n = propagate_shapes_and_constants(split_num_g)
    assert prop_sp_n.nodes["sp_n"].shape_metadata == (4, 15)
    assert len(prop_sp_n.nodes["sp_n"].output_specs) == 2

    # Multi-output Split with unshaped input
    split_unshaped_g = LogicalGraph(
        nodes={
            "x_un": LogicalNode(id="x_un", op_type="Input"),
            "sp_un": LogicalNode(
                id="sp_un",
                op_type="Split",
                inputs=["x_un"],
                outputs=["sp_un0", "sp_un1"],
            ),
        },
        inputs=["x_un"],
        outputs=["sp_un0", "sp_un1"],
    )
    prop_sp_un = propagate_shapes_and_constants(split_unshaped_g)
    assert prop_sp_un.nodes["sp_un"].output_specs == []

    # Multi-output Split port resolution via node_id:port and named outputs
    from ml_switcheroo_ir import DType, TensorSpec

    split_multi_g = LogicalGraph(
        nodes={
            "x_in": LogicalNode(id="x_in", op_type="Input", shape_metadata=(6, 4)),
            "sp": LogicalNode(
                id="sp",
                op_type="Split",
                inputs=["x_in"],
                outputs=["s0", "s1"],
                output_specs=[
                    TensorSpec((2, 4), DType.float32),
                    TensorSpec((4, 4), DType.float32),
                ],
                attributes={"axis": 0, "split": [2, 4]},
            ),
            "r_named": LogicalNode(id="r_named", op_type="Relu", inputs=["s1"]),
            "r_colon": LogicalNode(id="r_colon", op_type="Relu", inputs=["sp:1"]),
            "sp_no_specs": LogicalNode(
                id="sp_no_specs",
                op_type="Split",
                inputs=["x_in"],
                outputs=["ns0", "ns1"],
                attributes={"axis": 0, "num_outputs": 2},
            ),
            "r_colon_fallback": LogicalNode(id="r_colon_fallback", op_type="Relu", inputs=["sp_no_specs:5"]),
            "sp_unshaped": LogicalNode(
                id="sp_unshaped",
                op_type="CustomMulti",
                outputs=["u0", "u1"],
            ),
            "r_unshaped_colon": LogicalNode(id="r_unshaped_colon", op_type="Relu", inputs=["sp_unshaped:1"]),
            "r_unshaped_named": LogicalNode(id="r_unshaped_named", op_type="Relu", inputs=["u1"]),
        },
        inputs=["x_in"],
        outputs=[
            "r_named",
            "r_colon",
            "r_colon_fallback",
            "r_unshaped_colon",
            "r_unshaped_named",
        ],
    )
    prop_multi = propagate_shapes_and_constants(split_multi_g)
    assert prop_multi.nodes["r_named"].shape_metadata == (4, 4)
    assert prop_multi.nodes["r_colon"].shape_metadata == (4, 4)
    assert prop_multi.nodes["r_colon_fallback"].shape_metadata == (3, 4)
    assert prop_multi.nodes["r_unshaped_colon"].shape_metadata is None
    assert prop_multi.nodes["r_unshaped_named"].shape_metadata is None

    # Indivisible Split raises ValueError
    indiv_split_g = LogicalGraph(
        nodes={
            "x_in": LogicalNode(id="x_in", op_type="Input", shape_metadata=(5, 4)),
            "sp_bad": LogicalNode(
                id="sp_bad",
                op_type="Split",
                inputs=["x_in"],
                outputs=["b0", "b1"],
                attributes={"axis": 0, "num_outputs": 2},
            ),
        },
        inputs=["x_in"],
        outputs=["b0", "b1"],
    )
    with pytest.raises(ValueError, match="not evenly divisible"):
        propagate_shapes_and_constants(indiv_split_g)

    # Reshape with 0-dim copy semantics (concrete and symbolic) and error cases
    reshape_0_g = LogicalGraph(
        nodes={
            "x_in": LogicalNode(id="x_in", op_type="Input", shape_metadata=(4, 6)),
            "r0": LogicalNode(
                id="r0",
                op_type="Reshape",
                inputs=["x_in"],
                attributes={"shape": [0, 3, -1]},
            ),
            "x_sym": LogicalNode(id="x_sym", op_type="Input", shape_metadata=("B", 6)),
            "r0_sym": LogicalNode(
                id="r0_sym",
                op_type="Reshape",
                inputs=["x_sym"],
                attributes={"shape": [0, 2, 3]},
            ),
        },
        inputs=["x_in", "x_sym"],
        outputs=["r0", "r0_sym"],
    )
    prop_r0 = propagate_shapes_and_constants(reshape_0_g)
    assert prop_r0.nodes["r0"].shape_metadata == (4, 3, 2)
    assert prop_r0.nodes["r0_sym"].shape_metadata == ("B", 2, 3)

    # Reshape 0-dim out of bounds
    reshape_oob_g = LogicalGraph(
        nodes={
            "x_in": LogicalNode(id="x_in", op_type="Input", shape_metadata=(4,)),
            "r_oob": LogicalNode(
                id="r_oob",
                op_type="Reshape",
                inputs=["x_in"],
                attributes={"shape": [4, 0]},
            ),
        },
        inputs=["x_in"],
        outputs=["r_oob"],
    )
    with pytest.raises(ValueError, match="cannot copy dimension 0"):
        propagate_shapes_and_constants(reshape_oob_g)

    # Reshape negative dimension != -1
    reshape_neg_g = LogicalGraph(
        nodes={
            "x_in": LogicalNode(id="x_in", op_type="Input", shape_metadata=(4, 6)),
            "r_neg": LogicalNode(
                id="r_neg",
                op_type="Reshape",
                inputs=["x_in"],
                attributes={"shape": [-2, 12]},
            ),
        },
        inputs=["x_in"],
        outputs=["r_neg"],
    )
    with pytest.raises(ValueError, match="invalid non-positive dimension"):
        propagate_shapes_and_constants(reshape_neg_g)

    # Invalid Transpose permutation raises ValueError
    bad_trans_g = LogicalGraph(
        nodes={
            "x_in": LogicalNode(id="x_in", op_type="Input", shape_metadata=(4, 6)),
            "t_bad": LogicalNode(
                id="t_bad",
                op_type="Transpose",
                inputs=["x_in"],
                attributes={"perm": [0, 0]},
            ),
        },
        inputs=["x_in"],
        outputs=["t_bad"],
    )
    with pytest.raises(ValueError, match="Invalid Transpose permutation"):
        propagate_shapes_and_constants(bad_trans_g)


def test_constant_folding_and_downstream_shape_inference() -> None:
    """Verify Shape, Add, Sub, Mul, Div, Neg, Cast, Concat, and Slice constant folding and downstream consumption."""
    from ml_switcheroo_ir import DType, TensorSpec
    from ml_switcheroo_ir.shapes import infer_op_output_shape

    assert infer_op_output_shape("UnknownMysteryOp", [(2, 3)]) is None

    graph = LogicalGraph(
        name="ConstFoldGraph",
        nodes={
            "x": LogicalNode(id="x", op_type="Input", shape_metadata=(2, 6)),
            "sh": LogicalNode(id="sh", op_type="Shape", inputs=["x"], outputs=["sh_out"]),
            "c_one": LogicalNode(id="c_one", op_type="Constant", attributes={"value": [1, 0]}),
            "add_sh": LogicalNode(id="add_sh", op_type="Add", inputs=["sh_out", "c_one"]),
            "sub_c": LogicalNode(id="sub_c", op_type="Sub", inputs=["add_sh", "c_one"]),
            "mul_c": LogicalNode(id="mul_c", op_type="Mul", inputs=["sub_c", "init_scale"]),
            "div_c": LogicalNode(id="div_c", op_type="Div", inputs=["mul_c", "init_scale"]),
            "div_zero": LogicalNode(id="div_zero", op_type="Div", inputs=["div_c", "init_zero"]),
            "div_float": LogicalNode(id="div_float", op_type="arith.divf", inputs=["init_f", "init_f2"]),
            "neg_c": LogicalNode(id="neg_c", op_type="Neg", inputs=["c_one"]),
            "cast_i": LogicalNode(id="cast_i", op_type="Cast", inputs=["init_f"], attributes={"to": 7}),
            "cast_b": LogicalNode(id="cast_b", op_type="Cast", inputs=["c_one"], attributes={"to": DType.bool}),
            "cast_f": LogicalNode(id="cast_f", op_type="Cast", inputs=["c_one"], attributes={"dtype": "float32"}),
            "cast_dt": LogicalNode(id="cast_dt", op_type="Cast", inputs=["c_one"], dtype=DType.float64),
            "cast_unk": LogicalNode(id="cast_unk", op_type="Cast", inputs=["c_one"], attributes={"to": "not_a_dtype"}),
            "cat_2d": LogicalNode(
                id="cat_2d",
                op_type="Concat",
                inputs=["init_2d_a", "init_2d_b"],
                attributes={"axis": 1},
            ),
            "slice_2d": LogicalNode(
                id="slice_2d",
                op_type="Slice",
                inputs=["cat_2d", "init_starts", "init_ends", "init_axes", "init_steps"],
            ),
            "reshape_from_folded": LogicalNode(
                id="reshape_from_folded",
                op_type="Reshape",
                inputs=["x", "init_target_shape"],
            ),
            "unsq": LogicalNode(id="unsq", op_type="Unsqueeze", inputs=["reshape_from_folded", "init_ax0"]),
            "sq": LogicalNode(id="sq", op_type="Squeeze", inputs=["unsq", "init_ax0"]),
            "split_c": LogicalNode(
                id="split_c",
                op_type="Split",
                inputs=["sq", "init_splits"],
                outputs=["sp0", "sp1"],
                attributes={"axis": 0},
            ),
            "gemm_sp": LogicalNode(id="gemm_sp", op_type="Gemm", inputs=["sp0", "sp1"], attributes={"transB": 1}),
        },
        inputs=["x", "ext_spec"],
        input_specs={"ext_spec": TensorSpec(shape=(3, 4), dtype=DType.float32)},
        initializers={
            "init_scale": 2,
            "init_zero": 0,
            "init_f": [[4.0], [8.0]],
            "init_f2": [2.0, 4.0],
            "init_2d_a": [[1, 2], [3, 4]],
            "init_2d_b": [[5], [6]],
            "init_starts": [0],
            "init_ends": [2],
            "init_axes": [1],
            "init_steps": [1],
            "init_target_shape": [3, 4],
            "init_ax0": [0],
            "init_splits": [1, 2],
        },
        outputs=["gemm_sp", "slice_2d", "cast_i", "cast_b", "cast_f", "cast_dt", "neg_c", "div_float", "div_zero"],
        attributes={"custom_tag": "preserved"},
        loop_options={"unroll": 2},
    )

    out = propagate_shapes_and_constants(graph)
    assert out.attributes == {"custom_tag": "preserved"}
    assert out.loop_options == {"unroll": 2}
    assert out.nodes["sh"].op_type == "Constant"
    assert out.nodes["sh"].attributes["value"] == [2, 6]
    assert out.nodes["sh"].dtype == DType.int64
    assert out.nodes["div_c"].attributes["value"] == [2, 6]
    assert out.nodes["div_zero"].op_type == "Div"
    assert out.nodes["div_float"].attributes["value"] == [[2.0, 1.0], [4.0, 2.0]]
    assert out.nodes["neg_c"].attributes["value"] == [-1, 0]
    assert out.nodes["cast_i"].attributes["value"] == [[4], [8]]
    assert out.nodes["cast_b"].attributes["value"] == [True, False]
    assert out.nodes["cast_f"].attributes["value"] == [1.0, 0.0]
    assert out.nodes["cast_dt"].dtype == DType.float64
    assert out.nodes["cast_unk"].op_type == "Cast"
    assert out.nodes["cat_2d"].attributes["value"] == [[1, 2, 5], [3, 4, 6]]
    assert out.nodes["slice_2d"].attributes["value"] == [[1, 2], [3, 4]]
    assert out.nodes["reshape_from_folded"].shape_metadata == (3, 4)
    assert out.nodes["unsq"].shape_metadata == (1, 3, 4)
    assert out.nodes["sq"].shape_metadata == (3, 4)
    assert out.nodes["gemm_sp"].shape_metadata == (1, 2)


def test_constant_folding_edge_branches_and_subgraph_dce_cse() -> None:
    """Cover constant folding edge cases, commutative/metadata CSE, and subgraph capture DCE/CSE."""
    from ml_switcheroo_ir import DType, TensorSpec
    from ml_switcheroo_ir.transforms import (
        _concat_const_along_axis,
        _infer_const_dtype,
        _infer_const_value_shape,
        _is_const_tensor_value,
        _normalize_const_lists,
        _slice_const_single_axis,
    )

    assert _is_const_tensor_value([[1, 2], [3, 4]])
    assert not _is_const_tensor_value("not_numeric")
    assert _normalize_const_lists(["bad"]) is None
    assert _normalize_const_lists([[1, 2], [3]]) is None
    assert _infer_const_value_shape(TensorSpec(shape=("B", 4), dtype=DType.float32)) is None
    assert _infer_const_value_shape(TensorSpec(shape=(2, 4), dtype=DType.float32)) == (2, 4)
    assert _infer_const_value_shape([]) == (0,)
    assert _infer_const_dtype([]) == DType.float32
    assert _concat_const_along_axis([1, 2], 0) is None
    assert _concat_const_along_axis([[[1]], [[2], [3]]], 1) is None
    assert _slice_const_single_axis([1, 2], 0, 0, 1, 0) is None
    assert _slice_const_single_axis([[1, 2]], 1, 0, 1, 0) is None
    assert _slice_const_single_axis([1, 2], 1, 0, 1, 1) is None
    assert _infer_const_value_shape("not_a_tensor") is None
    assert _infer_const_dtype(True) == DType.bool
    assert _infer_const_dtype(1, hint=DType.int32) == DType.int32

    from ml_switcheroo_ir.transforms import (
        _broadcast_const_to_shape,
        _collect_subgraph_bound_names,
        _enrich_slice_attrs,
        _fold_binary_arith_op,
        _fold_concat_const_op,
        _fold_slice_const_op,
        _fold_unary_or_cast_op,
        _resolve_cast_target_dtype,
        _resolve_constant_value,
        _zip_binary_const,
    )
    from ml_switcheroo_ir.types import AttributeValue

    assert _resolve_cast_target_dtype(LogicalNode(id="c_bool_to", op_type="Cast", attributes={"to": True})) is None
    sl_attrs: dict[str, AttributeValue] = {}
    _enrich_slice_attrs(
        LogicalNode(id="sl_sc", op_type="Slice", inputs=["x", "s_scalar"]),
        sl_attrs,
        {},
        {"s_scalar": 0},
    )
    assert sl_attrs == {}

    assert _broadcast_const_to_shape(5, ()) == 5
    assert (
        _resolve_constant_value(
            "named_c",
            {"c_node": LogicalNode(id="c_node", op_type="Constant", outputs=["named_c"], attributes={"value": 3})},
        )
        == 3
    )
    assert (
        _resolve_constant_value(
            "nv",
            {"nv": LogicalNode(id="nv", op_type="Constant", attributes={})},
        )
        is None
    )
    bound_with_specs = _collect_subgraph_bound_names(
        LogicalGraph(
            name="SpecSub",
            nodes={
                "sp": LogicalNode(
                    id="sp",
                    op_type="Split",
                    output_specs=[TensorSpec(shape=(2,), dtype=DType.float32)],
                )
            },
        )
    )
    assert "sp:0" in bound_with_specs
    assert _fold_binary_arith_op("Add", LogicalNode(id="b_unk", op_type="Add", inputs=["x", "y"]), (2,), {}, {}) is None

    # Surviving multi-output port edge with source_idx > 0 in CSE
    surv_port_g = LogicalGraph(
        name="SurvPort",
        nodes={
            "x": LogicalNode(id="x", op_type="Input"),
            "sp": LogicalNode(
                id="sp",
                op_type="Split",
                inputs=["x"],
                output_specs=[TensorSpec(shape=(2,), dtype=DType.float32), TensorSpec(shape=(2,), dtype=DType.float32)],
            ),
            "c": LogicalNode(id="c", op_type="Relu", inputs=["sp:1"]),
        },
        inputs=["x"],
        outputs=["c"],
    )
    surv_cse = eliminate_common_subexpressions(surv_port_g)
    assert any(e.source_idx == 1 for e in surv_cse.edges)

    assert _zip_binary_const([1], 2, lambda a, b: a) == [1]
    assert _fold_unary_or_cast_op(LogicalNode(id="empty_neg", op_type="Neg"), {}, {}) is None
    assert _fold_unary_or_cast_op(LogicalNode(id="unk_neg", op_type="Neg", inputs=["missing"]), {}, {}) is None
    assert _fold_binary_arith_op("Add", LogicalNode(id="b1", op_type="Add", inputs=["x"]), (2,), {}, {}) is None
    assert _fold_concat_const_op(LogicalNode(id="cc1", op_type="Concat", inputs=["missing"]), {}, {}, {}) is None
    assert _fold_concat_const_op(LogicalNode(id="cc2", op_type="Concat", inputs=["sc"]), {}, {}, {"sc": 5}) is None
    assert (
        _fold_concat_const_op(
            LogicalNode(id="cc3", op_type="Concat", inputs=["a", "b"]),
            {"axis": 2},
            {},
            {"a": [[[1]]], "b": [[[2], [3]]]},
        )
        is None
    )
    assert _fold_slice_const_op(LogicalNode(id="sl1", op_type="Slice", inputs=["missing"]), {}, {}, {}) is None
    assert (
        _fold_slice_const_op(
            LogicalNode(id="sl2", op_type="Slice", inputs=["a"]),
            {"starts": "bad", "ends": [1]},
            {},
            {"a": [1, 2]},
        )
        is None
    )
    assert (
        _fold_slice_const_op(
            LogicalNode(id="sl3", op_type="Slice", inputs=["a"]),
            {"starts": [True], "ends": [1]},
            {},
            {"a": [1, 2]},
        )
        is None
    )
    assert (
        _fold_slice_const_op(
            LogicalNode(id="sl4", op_type="Slice", inputs=["a"]),
            {"starts": [0], "ends": [1], "steps": [0]},
            {},
            {"a": [1, 2]},
        )
        is None
    )


def test_symbolic_propagation_and_subgraph_dce_cse() -> None:
    """Verify Reshape new_shape, symbolic Shape skip, commutative CSE, and closure-capture DCE."""
    from ml_switcheroo_ir import DType, LogicalEdge, PartitionSpec, TensorSpec
    from ml_switcheroo_ir.transforms import (
        _is_ssa_live,
        _resolve_input_shape,
        _rewrite_subgraph_captures,
    )

    # Reshape with new_shape attribute, Shape on symbolic input, Input resolved from input_specs, and nested propagate
    sym_fold_g = LogicalGraph(
        name="SymAndSpecsProp",
        nodes={
            "x_unshaped": LogicalNode(id="x_unshaped", op_type="Input"),
            "sym_in": LogicalNode(id="sym_in", op_type="Input", shape_metadata=("B", 4)),
            "sym_sh": LogicalNode(id="sym_sh", op_type="Shape", inputs=["sym_in"]),
            "r_new": LogicalNode(
                id="r_new",
                op_type="Reshape",
                inputs=["x_unshaped"],
                attributes={"new_shape": [2, 3]},
                subgraphs={
                    "sub": LogicalGraph(
                        name="SubProp",
                        nodes={"s": LogicalNode(id="s", op_type="Constant", attributes={"value": [1, 2]})},
                        outputs=["s"],
                    ),
                    "meta": "str",
                },
            ),
            "s_attr": LogicalNode(
                id="s_attr",
                op_type="Slice",
                inputs=["x_unshaped", "non_const_inp", "non_const_inp"],
                attributes={"starts": [0], "ends": [1]},
            ),
            "non_const_inp": LogicalNode(id="non_const_inp", op_type="Input", shape_metadata=(1,)),
            "r_unshaped_attr": LogicalNode(id="r_unshaped_attr", op_type="Reshape", inputs=["x_unshaped"]),
            "c_preset_sh": LogicalNode(
                id="c_preset_sh",
                op_type="Constant",
                attributes={"value": [1, 2]},
                shape_metadata=(2,),
            ),
            "sq_nonconst": LogicalNode(id="sq_nonconst", op_type="Squeeze", inputs=["x_unshaped", "non_const_inp"]),
        },
        inputs=["x_unshaped", "sym_in", "non_const_inp"],
        input_specs={"x_unshaped": TensorSpec(shape=(6,), dtype=DType.float32)},
        outputs=["sym_sh", "r_new", "s_attr", "r_unshaped_attr", "c_preset_sh", "sq_nonconst"],
    )
    prop_sym = propagate_shapes_and_constants(sym_fold_g)
    assert prop_sym.nodes["x_unshaped"].shape_metadata == (6,)
    assert prop_sym.nodes["r_new"].shape_metadata == (2, 3)
    assert prop_sym.nodes["sym_sh"].op_type == "Shape"

    # Subgraph shadow bound alias in _rewrite_subgraph_captures and _is_ssa_live multi-output producer
    shadow_sub = LogicalGraph(name="Shadow", nodes={}, inputs=["shadow_x"], outputs=["shadow_x"])
    assert _rewrite_subgraph_captures(shadow_sub, {"shadow_x": "canon_x"}).outputs == ["shadow_x"]
    prod_g = LogicalGraph(
        name="ProdLive",
        nodes={"p": LogicalNode(id="p", op_type="Split", outputs=["p0", "p1"])},
        outputs=["p0"],
    )
    assert _is_ssa_live("p1", {"p"}, prod_g)

    # Resolve input shape via TensorSpec input_specs, node:idx without output_specs, and single-output named port
    n_unnamed_multi = LogicalNode(id="um", op_type="Split", shape_metadata=(4, 4))
    n_single_named = LogicalNode(id="sn", op_type="Relu", outputs=["sn_out"], shape_metadata=(2, 8))
    n_unshaped = LogicalNode(id="unsh", op_type="Relu")
    node_map = {"um": n_unnamed_multi, "sn": n_single_named, "unsh": n_unshaped}
    assert _resolve_input_shape("um:0", node_map) == (4, 4)
    assert _resolve_input_shape("sn_out", node_map) == (2, 8)
    assert _resolve_input_shape("unsh", node_map) is None
    assert _resolve_input_shape("sp_in", {}, input_specs={"sp_in": TensorSpec(shape=(5,), dtype=DType.float32)}) == (5,)

    # Commutative CSE, sharding/version separation, multi-output unnamed port CSE, and nested subgraph captures
    inner_body = LogicalGraph(
        name="SubBody",
        nodes={
            "sub_use": LogicalNode(id="sub_use", op_type="Relu", inputs=["dup_b_out"]),
            "sub_dead": LogicalNode(id="sub_dead", op_type="Neg", inputs=["unused_outer"]),
        },
        outputs=["sub_use"],
    )
    outer = LogicalGraph(
        name="OuterCseDce",
        nodes={
            "a": LogicalNode(id="a", op_type="Input"),
            "b": LogicalNode(id="b", op_type="Input"),
            "unused_outer": LogicalNode(id="unused_outer", op_type="Relu", inputs=["a"]),
            "dup_a": LogicalNode(id="dup_a", op_type="Add", inputs=["a", "b"], outputs=["dup_a_out"]),
            "dup_b": LogicalNode(id="dup_b", op_type="Add", inputs=["b", "a"], outputs=["dup_b_out"]),
            "sharded_diff": LogicalNode(
                id="sharded_diff",
                op_type="Add",
                inputs=["a", "b"],
                sharding=PartitionSpec(axes=("tp",)),
            ),
            "mp1": LogicalNode(
                id="mp1",
                op_type="Split",
                inputs=["a"],
                output_specs=[TensorSpec(shape=(2,), dtype=DType.float32), TensorSpec(shape=(2,), dtype=DType.float32)],
            ),
            "mp2": LogicalNode(
                id="mp2",
                op_type="Split",
                inputs=["a"],
                output_specs=[TensorSpec(shape=(2,), dtype=DType.float32), TensorSpec(shape=(2,), dtype=DType.float32)],
            ),
            "mp_cons": LogicalNode(id="mp_cons", op_type="Relu", inputs=[]),
            "loop_op": LogicalNode(
                id="loop_op",
                op_type="Loop",
                inputs=["b"],
                subgraphs={"body": inner_body, "tag": "keep"},
            ),
        },
        edges=[
            LogicalEdge(source="a", target="dup_a", target_idx=0),
            LogicalEdge(source="b", target="dup_a", target_idx=1),
            LogicalEdge(source="b", target="dup_b", target_idx=0),
            LogicalEdge(source="a", target="dup_b", target_idx=1),
            LogicalEdge(source="a", target="sharded_diff", target_idx=0),
            LogicalEdge(source="b", target="sharded_diff", target_idx=1),
            LogicalEdge(source="a", target="mp1", target_idx=0),
            LogicalEdge(source="a", target="mp2", target_idx=0),
            LogicalEdge(source="mp2", target="mp_cons", source_idx=1, target_idx=0, value_name="mp2:1"),
            LogicalEdge(source="b", target="loop_op", target_idx=0),
        ],
        initializers={"used_init": [1.0], "dead_init": [9.0]},
        inputs=["a", "b"],
        outputs=["loop_op", "sharded_diff", "mp_cons", "used_init"],
        attributes={"outer_attr": 42},
        loop_options={"max_trip": 8},
    )

    cse_res = eliminate_common_subexpressions(outer)
    assert "dup_b" not in cse_res.nodes
    assert "sharded_diff" in cse_res.nodes
    assert "mp2" not in cse_res.nodes
    assert cse_res.nodes["mp_cons"].inputs == ["mp1:1"]
    rewritten_sub = cse_res.nodes["loop_op"].subgraphs["body"]
    assert isinstance(rewritten_sub, LogicalGraph)
    assert rewritten_sub.nodes["sub_use"].inputs == ["dup_a_out"]

    dce_res = eliminate_dead_nodes(cse_res)
    # dup_a is ONLY referenced inside loop_op.subgraphs['body'] and MUST stay alive!
    assert "dup_a" in dce_res.nodes
    # unused_outer was only referenced by dead sub_dead inside inner_body, so BOTH are pruned!
    assert "unused_outer" not in dce_res.nodes
    cleaned_sub = dce_res.nodes["loop_op"].subgraphs["body"]
    assert isinstance(cleaned_sub, LogicalGraph)
    assert "sub_dead" not in cleaned_sub.nodes
    assert "used_init" in dce_res.initializers
    assert "dead_init" not in dce_res.initializers
    assert dce_res.attributes == {"outer_attr": 42}
    assert dce_res.loop_options == {"max_trip": 8}
