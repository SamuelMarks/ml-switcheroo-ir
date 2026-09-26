"""Tests for enhanced shape propagation with symbolic dimensions, broadcasting, and low-level ops."""

from __future__ import annotations

import pytest

from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalNode,
    propagate_shapes_and_constants,
)
from ml_switcheroo_ir.shapes import SymInt, SymVar


def test_propagate_unary_and_binary_ops() -> None:
    """Test shape propagation through elementwise unary and broadcasting binary operations."""
    n_in1 = LogicalNode(
        id="in1", op_type="Placeholder", shape_metadata=(SymInt("B"), 1, 64)
    )
    n_in2 = LogicalNode(id="in2", op_type="Placeholder", shape_metadata=(128, 64))
    n_relu = LogicalNode(id="relu1", op_type="Relu", inputs=["in1"])
    n_add = LogicalNode(id="add1", op_type="Add", inputs=["relu1", "in2"])

    g = LogicalGraph(
        nodes={n.id: n for n in [n_in1, n_in2, n_relu, n_add]},
        inputs=["in1", "in2"],
        outputs=["add1"],
    )

    evaluated = propagate_shapes_and_constants(g)
    assert evaluated.nodes["relu1"].shape_metadata == (SymInt("B"), 1, 64)
    assert evaluated.nodes["add1"].shape_metadata == (SymInt("B"), 128, 64)


def test_propagate_matmul_and_gemm() -> None:
    """Test shape propagation through MatMul and Gemm operations with transA, transB, and bias."""
    # MatMul
    n_a = LogicalNode(
        id="a", op_type="Placeholder", shape_metadata=(SymInt("B"), 128, 64)
    )
    n_b = LogicalNode(id="b", op_type="Placeholder", shape_metadata=(64, 256))
    n_matmul = LogicalNode(id="mm", op_type="MatMul", inputs=["a", "b"])

    # Gemm: (M, K) x (N, K)^T + (N,) -> (M, N)
    n_g_a = LogicalNode(id="ga", op_type="Placeholder", shape_metadata=(16, 32))
    n_g_b = LogicalNode(id="gb", op_type="Placeholder", shape_metadata=(64, 32))
    n_g_c = LogicalNode(id="gc", op_type="Placeholder", shape_metadata=(64,))
    n_gemm = LogicalNode(
        id="gemm1",
        op_type="Gemm",
        inputs=["ga", "gb", "gc"],
        attributes={"transB": 1},
    )

    g = LogicalGraph(
        nodes={n.id: n for n in [n_a, n_b, n_matmul, n_g_a, n_g_b, n_g_c, n_gemm]},
        inputs=["a", "b", "ga", "gb", "gc"],
        outputs=["mm", "gemm1"],
    )

    evaluated = propagate_shapes_and_constants(g)
    assert evaluated.nodes["mm"].shape_metadata == (SymInt("B"), 128, 256)
    assert evaluated.nodes["gemm1"].shape_metadata == (16, 64)

    # Gemm error branch: inner contraction mismatch
    n_bad_gb = LogicalNode(id="bad_gb", op_type="Placeholder", shape_metadata=(64, 30))
    n_bad_gemm = LogicalNode(
        id="bad_gemm",
        op_type="Gemm",
        inputs=["ga", "bad_gb"],
        attributes={"transB": 1},
    )
    with pytest.raises(ValueError, match="contraction dimension mismatch"):
        propagate_shapes_and_constants(
            LogicalGraph(
                nodes={n.id: n for n in [n_g_a, n_bad_gb, n_bad_gemm]},
                inputs=["ga", "bad_gb"],
            )
        )


def test_propagate_squeeze_unsqueeze_flatten() -> None:
    """Test shape propagation through Squeeze, Unsqueeze, and Flatten operations."""
    n_in = LogicalNode(id="in1", op_type="Placeholder", shape_metadata=(2, 1, 4, 1))

    # Squeeze: remove size-1 dims
    n_sq_all = LogicalNode(id="sq_all", op_type="Squeeze", inputs=["in1"])
    # Squeeze: remove specific axis 1
    n_sq_ax = LogicalNode(
        id="sq_ax",
        op_type="Squeeze",
        inputs=["in1"],
        attributes={"axes": [1]},
    )

    # Unsqueeze: insert 1 at index 1 and 3
    n_in2 = LogicalNode(id="in2", op_type="Placeholder", shape_metadata=(2, 4))
    n_unsq = LogicalNode(
        id="unsq1",
        op_type="Unsqueeze",
        inputs=["in2"],
        attributes={"axes": [1, 3]},
    )

    # Flatten: axis=2 -> (2*1 = 2, 4*1 = 4)
    n_fl = LogicalNode(
        id="fl1",
        op_type="Flatten",
        inputs=["in1"],
        attributes={"axis": 2},
    )

    g = LogicalGraph(
        nodes={n.id: n for n in [n_in, n_sq_all, n_sq_ax, n_in2, n_unsq, n_fl]},
        inputs=["in1", "in2"],
        outputs=["sq_all"],
    )

    evaluated = propagate_shapes_and_constants(g)
    assert evaluated.nodes["sq_all"].shape_metadata == (2, 4)
    assert evaluated.nodes["sq_ax"].shape_metadata == (2, 4, 1)
    assert evaluated.nodes["unsq1"].shape_metadata == (2, 1, 4, 1)
    assert evaluated.nodes["fl1"].shape_metadata == (2, 4)


def test_propagate_concat_split_slice() -> None:
    """Test shape propagation through Concat, Split, and Slice operations."""
    n_c1 = LogicalNode(id="c1", op_type="Placeholder", shape_metadata=(2, 10, 8))
    n_c2 = LogicalNode(id="c2", op_type="Placeholder", shape_metadata=(2, 20, 8))
    n_concat = LogicalNode(
        id="cat1",
        op_type="Concat",
        inputs=["c1", "c2"],
        attributes={"axis": 1},
    )

    # Split: axis 1 into 2 partitions of size 15
    n_split = LogicalNode(
        id="sp1",
        op_type="Split",
        inputs=["cat1"],
        attributes={"axis": 1, "num_outputs": 2},
    )

    # Slice: axis 1 from 5 to 15 step 2 -> (15 - 5 + 1) // 2 = 5
    n_slice = LogicalNode(
        id="sl1",
        op_type="Slice",
        inputs=["cat1"],
        attributes={
            "starts": [5],
            "ends": [15],
            "axes": [1],
            "steps": [2],
        },
    )

    g = LogicalGraph(
        nodes={n.id: n for n in [n_c1, n_c2, n_concat, n_split, n_slice]},
        inputs=["c1", "c2"],
        outputs=["cat1", "sp1", "sl1"],
    )

    evaluated = propagate_shapes_and_constants(g)
    assert evaluated.nodes["cat1"].shape_metadata == (2, 30, 8)
    assert evaluated.nodes["sp1"].shape_metadata == (2, 15, 8)
    assert evaluated.nodes["sl1"].shape_metadata == (2, 5, 8)

    # Concat rank mismatch error
    n_bad_rank = LogicalNode(
        id="bad_rank", op_type="Placeholder", shape_metadata=(2, 10)
    )
    n_bad_cat = LogicalNode(
        id="bad_cat",
        op_type="Concat",
        inputs=["c1", "bad_rank"],
        attributes={"axis": 1},
    )
    with pytest.raises(ValueError, match="Concat rank mismatch"):
        propagate_shapes_and_constants(
            LogicalGraph(
                nodes={n.id: n for n in [n_c1, n_bad_rank, n_bad_cat]},
                inputs=["c1", "bad_rank"],
            )
        )


def test_nanogpt_symbolic_shape_propagation() -> None:
    """Test full NanoGPT multi-head attention block with variable batch B and sequence length T."""
    si_b = SymInt("B")
    si_t = SymInt("T")
    c_embed = 768
    num_heads = 12
    head_dim = c_embed // num_heads  # 64

    # Inputs: x is (B, T, 768)
    n_x = LogicalNode(
        id="x", op_type="Placeholder", shape_metadata=(si_b, si_t, c_embed)
    )

    # QKV projection: (B, T, 768) x (768, 3 * 768) -> (B, T, 2304)
    n_w_qkv = LogicalNode(
        id="w_qkv", op_type="Placeholder", shape_metadata=(c_embed, 3 * c_embed)
    )
    n_qkv = LogicalNode(id="qkv", op_type="MatMul", inputs=["x", "w_qkv"])

    # Split Q, K, V -> each (B, T, 768)
    n_q = LogicalNode(
        id="q",
        op_type="Split",
        inputs=["qkv"],
        attributes={"axis": 2, "num_outputs": 3},
    )

    # Reshape Q into (B, T, num_heads, head_dim) -> (B, T, 12, 64)
    n_q_reshaped = LogicalNode(
        id="q_reshaped",
        op_type="Reshape",
        inputs=["q"],
        attributes={"shape": [si_b, si_t, num_heads, head_dim]},
    )

    # Transpose to (B, num_heads, T, head_dim) -> (B, 12, T, 64)
    n_q_transposed = LogicalNode(
        id="q_transposed",
        op_type="Transpose",
        inputs=["q_reshaped"],
        attributes={"perm": [0, 2, 1, 3]},
    )

    # Softmax / Unary activation
    n_q_act = LogicalNode(id="q_act", op_type="Relu", inputs=["q_transposed"])

    g = LogicalGraph(
        nodes={
            n.id: n
            for n in [
                n_x,
                n_w_qkv,
                n_qkv,
                n_q,
                n_q_reshaped,
                n_q_transposed,
                n_q_act,
            ]
        },
        inputs=["x", "w_qkv"],
        outputs=["q_act"],
    )

    evaluated = propagate_shapes_and_constants(g)
    assert evaluated.nodes["qkv"].shape_metadata == (si_b, si_t, 2304)
    assert evaluated.nodes["q"].shape_metadata == (si_b, si_t, 768)
    assert evaluated.nodes["q_reshaped"].shape_metadata == (si_b, si_t, 12, 64)
    assert evaluated.nodes["q_transposed"].shape_metadata == (si_b, 12, si_t, 64)
    assert evaluated.nodes["q_act"].shape_metadata == (si_b, 12, si_t, 64)


def test_transforms_helper_branches_and_edge_cases() -> None:
    """Test helper edge branches in shape propagation when inputs are empty or special attributes are passed."""
    from ml_switcheroo_ir.shapes import SymbolicConstraintTracker
    from ml_switcheroo_ir.transforms import (
        _propagate_binary_op,
        _propagate_concat_op,
        _propagate_flatten_op,
        _propagate_gemm_op,
        _propagate_matmul_op,
        _propagate_slice_op,
        _propagate_split_op,
        _propagate_squeeze_op,
        _propagate_unary_op,
        _propagate_unsqueeze_op,
    )

    tracker = SymbolicConstraintTracker()

    # Empty inputs across helpers
    empty_node = LogicalNode(id="empty", op_type="Custom", inputs=[])
    assert _propagate_unary_op(empty_node, {}) is None
    assert _propagate_binary_op(empty_node, {}, tracker) is None
    assert _propagate_matmul_op(empty_node, {}) is None
    assert _propagate_gemm_op(empty_node, {}, tracker) is None
    assert _propagate_squeeze_op(empty_node, {}) is None
    assert _propagate_unsqueeze_op(empty_node, {}) is None
    assert _propagate_flatten_op(empty_node, {}) is None
    assert _propagate_concat_op(empty_node, {}) is None
    assert _propagate_split_op(empty_node, {}) is None
    assert _propagate_slice_op(empty_node, {}) is None

    # Gemm with transA=1 and transB=0
    n_a = LogicalNode(id="ga", op_type="Placeholder", shape_metadata=(32, 16))
    n_b = LogicalNode(id="gb", op_type="Placeholder", shape_metadata=(32, 64))
    n_gemm = LogicalNode(
        id="gemm_trans_a",
        op_type="Gemm",
        inputs=["ga", "gb"],
        attributes={"transA": 1, "transB": 0},
    )
    processed = {"ga": n_a, "gb": n_b}
    assert _propagate_gemm_op(n_gemm, processed, tracker) == (16, 64)

    # Missing inputs in new_nodes
    n_miss = LogicalNode(id="miss", op_type="Custom", inputs=["missing1", "missing2"])
    assert _propagate_binary_op(n_miss, {}, tracker) is None
    assert _propagate_matmul_op(n_miss, {}) is None
    assert _propagate_gemm_op(n_miss, {}, tracker) is None

    # Gemm with rank != 2 inputs
    n_1d = LogicalNode(id="n1d", op_type="Placeholder", shape_metadata=(16,))
    n_gemm_bad_rank = LogicalNode(id="gbr", op_type="Gemm", inputs=["n1d", "gb"])
    assert (
        _propagate_gemm_op(n_gemm_bad_rank, {"n1d": n_1d, "gb": n_b}, tracker) is None
    )

    # Gemm without bias C and with unshaped C
    assert _propagate_gemm_op(n_gemm, {"ga": n_a, "gb": n_b}, tracker) == (16, 64)
    n_gc_unshaped = LogicalNode(id="gc_un", op_type="Placeholder", shape_metadata=None)
    n_gemm_unshaped_c = LogicalNode(
        id="gemm_unc",
        op_type="Gemm",
        inputs=["ga", "gb", "gc_un"],
        attributes={"transA": 1, "transB": 0},
    )
    assert _propagate_gemm_op(
        n_gemm_unshaped_c,
        {"ga": n_a, "gb": n_b, "gc_un": n_gc_unshaped},
        tracker,
    ) == (16, 64)

    # Squeeze with axes passed as second input node
    n_x = LogicalNode(id="x", op_type="Placeholder", shape_metadata=(2, 1, 4))
    n_axes_const = LogicalNode(id="ax", op_type="Constant", attributes={"value": [1]})
    n_sq_inp = LogicalNode(id="sq", op_type="Squeeze", inputs=["x", "ax"])
    processed_sq = {"x": n_x, "ax": n_axes_const}
    assert _propagate_squeeze_op(n_sq_inp, processed_sq) == (2, 4)

    # Squeeze with second input node lacking value attribute
    n_ax_empty = LogicalNode(id="ax_emp", op_type="Constant", attributes={})
    n_sq_ax_emp = LogicalNode(id="sq_emp", op_type="Squeeze", inputs=["x", "ax_emp"])
    assert _propagate_squeeze_op(n_sq_ax_emp, {"x": n_x, "ax_emp": n_ax_empty}) == (
        2,
        4,
    )

    # Squeeze with single int axis
    n_sq_int = LogicalNode(
        id="sq_int", op_type="Squeeze", inputs=["x"], attributes={"axes": 1}
    )
    assert _propagate_squeeze_op(n_sq_int, {"x": n_x}) == (2, 4)

    # Squeeze without axes (squeezes all 1s)
    n_sq_no_ax = LogicalNode(id="sq_no_ax", op_type="Squeeze", inputs=["x"])
    assert _propagate_squeeze_op(n_sq_no_ax, {"x": n_x}) == (2, 4)

    # Unsqueeze with axes passed as second input node
    n_unsq_inp = LogicalNode(id="unsq", op_type="Unsqueeze", inputs=["x", "ax"])
    assert _propagate_unsqueeze_op(n_unsq_inp, processed_sq) == (2, 1, 1, 4)

    # Unsqueeze with second input node lacking value attribute
    n_unsq_ax_emp = LogicalNode(
        id="unsq_emp", op_type="Unsqueeze", inputs=["x", "ax_emp"]
    )
    assert _propagate_unsqueeze_op(n_unsq_ax_emp, {"x": n_x, "ax_emp": n_ax_empty}) == (
        2,
        1,
        4,
    )

    # Unsqueeze with single int axis
    n_unsq_int = LogicalNode(
        id="unsq_int", op_type="Unsqueeze", inputs=["x"], attributes={"axes": 0}
    )
    assert _propagate_unsqueeze_op(n_unsq_int, {"x": n_x}) == (1, 2, 1, 4)

    # Unsqueeze without axes (returns original shape tuple)
    n_unsq_no_ax = LogicalNode(id="unsq_no_ax", op_type="Unsqueeze", inputs=["x"])
    assert _propagate_unsqueeze_op(n_unsq_no_ax, {"x": n_x}) == (2, 1, 4)

    # Slice without starts (returns None)
    n_sl_no_starts = LogicalNode(id="sl_ns", op_type="Slice", inputs=["x"])
    assert _propagate_slice_op(n_sl_no_starts, {"x": n_x}) is None

    # Slice with negative step on concrete shape
    n_sl_step_neg = LogicalNode(
        id="sl_sn",
        op_type="Slice",
        inputs=["x"],
        attributes={"starts": [2], "ends": [0], "axes": [0], "steps": [-1]},
    )
    assert _propagate_slice_op(n_sl_step_neg, {"x": n_x}) == (0, 1, 4)

    # Slice with negative start and end bounds on concrete shape
    n_sl_neg_idx = LogicalNode(
        id="sl_ni",
        op_type="Slice",
        inputs=["x"],
        attributes={"starts": [-2], "ends": [-1], "axes": [0], "steps": [1]},
    )
    assert _propagate_slice_op(n_sl_neg_idx, {"x": n_x}) == (1, 1, 4)

    # Slice with non-integer starts and ends (skips calculation)
    n_sl_str = LogicalNode(
        id="sl_str",
        op_type="Slice",
        inputs=["x"],
        attributes={"starts": ["s"], "ends": ["e"], "axes": [0], "steps": [1]},
    )
    assert _propagate_slice_op(n_sl_str, {"x": n_x}) == (2, 1, 4)

    # Split with symbolic dimension
    si_n = SymInt("N")
    n_sym = LogicalNode(id="sym_in", op_type="Placeholder", shape_metadata=(si_n, 64))
    n_split_sym = LogicalNode(
        id="sp_sym",
        op_type="Split",
        inputs=["sym_in"],
        attributes={"axis": 0, "num_outputs": 2},
    )
    res_split = _propagate_split_op(n_split_sym, {"sym_in": n_sym})
    assert res_split is not None
    assert str(res_split[0]) == "(N // 2)"

    # Slice with symbolic dimension and negative bounds
    si_t = SymInt("T")
    n_sym_t = LogicalNode(id="sym_t", op_type="Placeholder", shape_metadata=(si_t,))
    n_sl_sym = LogicalNode(
        id="sl_sym",
        op_type="Slice",
        inputs=["sym_t"],
        attributes={"starts": [0], "ends": [10], "axes": [0], "steps": [1]},
    )
    res_sl = _propagate_slice_op(n_sl_sym, {"sym_t": n_sym_t})
    assert res_sl == (10,)

    # Flatten with symbolic dimension
    n_fl_sym = LogicalNode(
        id="fl_sym",
        op_type="Flatten",
        inputs=["sym_in"],
        attributes={"axis": 1},
    )
    res_fl = _propagate_flatten_op(n_fl_sym, {"sym_in": n_sym})
    assert res_fl == (SymVar("N"), 64)

    # Concat along symbolic dimension
    n_cat_s1 = LogicalNode(
        id="cat_s1", op_type="Placeholder", shape_metadata=(si_n, 32)
    )
    n_cat_s2 = LogicalNode(
        id="cat_s2", op_type="Placeholder", shape_metadata=(si_n, 32)
    )
    n_cat_sym = LogicalNode(
        id="cat_sym",
        op_type="Concat",
        inputs=["cat_s1", "cat_s2"],
        attributes={"axis": 0},
    )
    res_cat = _propagate_concat_op(n_cat_sym, {"cat_s1": n_cat_s1, "cat_s2": n_cat_s2})
    assert res_cat is not None
    assert str(res_cat[0]) == "(2 * N)"

    # Split with explicit split list
    n_split_list = LogicalNode(
        id="sp_list",
        op_type="Split",
        inputs=["sym_in"],
        attributes={"axis": 1, "split": [20, 44]},
    )
    res_split_l = _propagate_split_op(n_split_list, {"sym_in": n_sym})
    assert res_split_l == (si_n, 20)

    # Slice with negative step
    n_sl_neg = LogicalNode(
        id="sl_neg",
        op_type="Slice",
        inputs=["sym_t"],
        attributes={"starts": [10], "ends": [0], "axes": [0], "steps": [-1]},
    )
    res_sl_neg = _propagate_slice_op(n_sl_neg, {"sym_t": n_sym_t})
    assert res_sl_neg == (-10,)

    # Graph with unshaped nodes across all ops maintaining shape_metadata
    unshaped_ops = [
        ("u1", "Relu"),
        ("u2", "Add"),
        ("u3", "MatMul"),
        ("u4", "Gemm"),
        ("u5", "Squeeze"),
        ("u6", "Unsqueeze"),
        ("u7", "Flatten"),
        ("u8", "Concat"),
        ("u9", "Split"),
        ("u10", "Slice"),
    ]
    unshaped_nodes = {
        nid: LogicalNode(id=nid, op_type=op, inputs=["missing"])
        for nid, op in unshaped_ops
    }
    g_unshaped = LogicalGraph(
        nodes=unshaped_nodes,
        inputs=[],
        outputs=[nid for nid, _ in unshaped_ops],
    )
    res_unshaped = propagate_shapes_and_constants(g_unshaped)
    for nid, _ in unshaped_ops:
        assert res_unshaped.nodes[nid].shape_metadata is None
