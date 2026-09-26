"""Tests for collective communication verification (AllReduce, AllGather, ReduceScatter, AllToAll)."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalMesh, LogicalNode
from ml_switcheroo_ir.validator import ValidationLevel, Validator


def test_all_reduce_verification() -> None:
    """Verify AllReduce validation for valid operators, invalid operators, and unknown mesh axes."""
    mesh = LogicalMesh(shape={"data": 4, "model": 2})
    v = Validator(level=ValidationLevel.STRICT)

    # Valid AllReduce
    n_valid = LogicalNode(
        id="ar_ok",
        op_type="AllReduce",
        inputs=["x"],
        attributes={"reduction": "sum", "mesh_axis": "data"},
    )
    g_valid = LogicalGraph(
        nodes={"x": LogicalNode(id="x", op_type="Placeholder"), "ar_ok": n_valid},
        inputs=["x"],
        outputs=["ar_ok"],
        mesh=mesh,
    )
    assert len(v.validate_sharding_propagation(g_valid)) == 0

    # Invalid reduction operator
    n_bad_red = LogicalNode(
        id="ar_bad_red",
        op_type="AllReduce",
        inputs=["x"],
        attributes={"reduction": "invalid_reduction_op", "mesh_axis": "data"},
    )
    g_bad_red = LogicalGraph(
        nodes={
            "x": LogicalNode(id="x", op_type="Placeholder"),
            "ar_bad_red": n_bad_red,
        },
        inputs=["x"],
        outputs=["ar_bad_red"],
        mesh=mesh,
    )
    errs = v.validate_sharding_propagation(g_bad_red)
    assert any("invalid reduction operator" in e.message for e in errs)

    # Unknown mesh axis
    n_bad_axis = LogicalNode(
        id="ar_bad_axis",
        op_type="all_reduce",
        inputs=["x"],
        attributes={"reduction": "min", "axis": "nonexistent_mesh_axis"},
    )
    g_bad_axis = LogicalGraph(
        nodes={
            "x": LogicalNode(id="x", op_type="Placeholder"),
            "ar_bad_axis": n_bad_axis,
        },
        inputs=["x"],
        outputs=["ar_bad_axis"],
        mesh=mesh,
    )
    errs_axis = v.validate_sharding_propagation(g_bad_axis)
    assert any("specifies unknown mesh axis" in e.message for e in errs_axis)


def test_all_gather_verification() -> None:
    """Verify AllGather dimension expansion factor against mesh axis partition factor."""
    mesh = LogicalMesh(shape={"tp": 4})
    v = Validator(level=ValidationLevel.STRICT)

    n_in = LogicalNode(id="in_tensor", op_type="Placeholder", shape_metadata=(2, 64))

    # Valid AllGather: input is (2, 64), gathered on axis 1 by factor 4 -> (2, 256)
    n_ag_ok = LogicalNode(
        id="ag_ok",
        op_type="AllGather",
        inputs=["in_tensor"],
        shape_metadata=(2, 256),
        attributes={"axis": 1, "mesh_axis": "tp"},
    )
    g_ok = LogicalGraph(
        nodes={"in_tensor": n_in, "ag_ok": n_ag_ok},
        inputs=["in_tensor"],
        outputs=["ag_ok"],
        mesh=mesh,
    )
    assert len(v.validate_sharding_propagation(g_ok)) == 0

    # Mismatched AllGather: output is (2, 128) instead of 256
    n_ag_bad = LogicalNode(
        id="ag_bad",
        op_type="all_gather",
        inputs=["in_tensor"],
        shape_metadata=(2, 128),
        attributes={"gather_dim": 1, "mesh_axis": "tp"},
    )
    g_bad = LogicalGraph(
        nodes={"in_tensor": n_in, "ag_bad": n_ag_bad},
        inputs=["in_tensor"],
        outputs=["ag_bad"],
        mesh=mesh,
    )
    errs = v.validate_sharding_propagation(g_bad)
    assert any("does not match input size" in e.message for e in errs)


def test_reduce_scatter_verification() -> None:
    """Verify ReduceScatter dimension contraction factor against mesh axis partition factor."""
    mesh = LogicalMesh(shape={"tp": 4})
    v = Validator(level=ValidationLevel.STRICT)

    n_in = LogicalNode(id="in_tensor", op_type="Placeholder", shape_metadata=(2, 256))

    # Valid ReduceScatter: input is (2, 256), scattered on axis 1 by factor 4 -> (2, 64)
    n_rs_ok = LogicalNode(
        id="rs_ok",
        op_type="ReduceScatter",
        inputs=["in_tensor"],
        shape_metadata=(2, 64),
        attributes={"axis": 1, "mesh_axis": "tp"},
    )
    g_ok = LogicalGraph(
        nodes={"in_tensor": n_in, "rs_ok": n_rs_ok},
        inputs=["in_tensor"],
        outputs=["rs_ok"],
        mesh=mesh,
    )
    assert len(v.validate_sharding_propagation(g_ok)) == 0

    # Mismatched ReduceScatter: output is (2, 128) instead of 64
    n_rs_bad = LogicalNode(
        id="rs_bad",
        op_type="reduce_scatter",
        inputs=["in_tensor"],
        shape_metadata=(2, 128),
        attributes={"scatter_dim": 1, "mesh_axis": "tp"},
    )
    g_bad = LogicalGraph(
        nodes={"in_tensor": n_in, "rs_bad": n_rs_bad},
        inputs=["in_tensor"],
        outputs=["rs_bad"],
        mesh=mesh,
    )
    errs = v.validate_sharding_propagation(g_bad)
    assert any("does not match input size" in e.message for e in errs)


def test_all_to_all_verification() -> None:
    """Verify AllToAll axis bounds verification against tensor rank."""
    v = Validator(level=ValidationLevel.STRICT)

    # Valid AllToAll: rank 2, split_axis=0, concat_axis=1
    n_a2a_ok = LogicalNode(
        id="a2a_ok",
        op_type="AllToAll",
        shape_metadata=(8, 16),
        attributes={"split_axis": 0, "concat_axis": 1},
    )
    g_ok = LogicalGraph(
        nodes={"a2a_ok": n_a2a_ok},
        inputs=["a2a_ok"],
        outputs=["a2a_ok"],
    )
    assert len(v.validate_sharding_propagation(g_ok)) == 0

    # Out of bounds axis: split_axis=3 on rank 2 tensor
    n_a2a_bad = LogicalNode(
        id="a2a_bad",
        op_type="all_to_all",
        shape_metadata=(8, 16),
        attributes={"split_dim": 3, "concat_dim": 1},
    )
    g_bad = LogicalGraph(
        nodes={"a2a_bad": n_a2a_bad},
        inputs=["a2a_bad"],
        outputs=["a2a_bad"],
    )
    errs = v.validate_sharding_propagation(g_bad)
    assert any("are out of bounds for tensor of rank 2" in e.message for e in errs)


def test_collective_verification_edge_branches() -> None:
    """Verify edge conditions for AllReduce, AllGather, ReduceScatter, and AllToAll."""
    v = Validator(level=ValidationLevel.STRICT)

    # AllReduce prod without mesh
    n_prod = LogicalNode(
        id="ar_prod",
        op_type="AllReduce",
        inputs=["x"],
        attributes={"reduction": "prod"},
    )
    g_prod = LogicalGraph(
        nodes={"x": LogicalNode(id="x", op_type="Placeholder"), "ar_prod": n_prod},
        inputs=["x"],
        outputs=["ar_prod"],
    )
    assert len(v.validate_sharding_propagation(g_prod)) == 0

    # AllGather without mesh or gather_dim, or gather_dim out of bounds
    n_ag_no_mesh = LogicalNode(
        id="ag_nm",
        op_type="AllGather",
        inputs=["x"],
        shape_metadata=(2, 64),
    )
    n_ag_oob = LogicalNode(
        id="ag_oob",
        op_type="AllGather",
        inputs=["x"],
        shape_metadata=(2, 64),
        attributes={"axis": 10, "mesh_axis": "tp"},
    )
    mesh = LogicalMesh(shape={"tp": 2})
    g_ag_edge = LogicalGraph(
        nodes={
            "x": LogicalNode(id="x", op_type="Placeholder", shape_metadata=(2, 32)),
            "ag_nm": n_ag_no_mesh,
            "ag_oob": n_ag_oob,
        },
        inputs=["x"],
        outputs=["ag_nm", "ag_oob"],
        mesh=mesh,
    )
    assert len(v.validate_sharding_propagation(g_ag_edge)) == 0

    # ReduceScatter without mesh or scatter_dim, or scatter_dim out of bounds
    n_rs_no_mesh = LogicalNode(
        id="rs_nm",
        op_type="ReduceScatter",
        inputs=["x"],
        shape_metadata=(2, 16),
    )
    n_rs_oob = LogicalNode(
        id="rs_oob",
        op_type="ReduceScatter",
        inputs=["x"],
        shape_metadata=(2, 16),
        attributes={"axis": 10, "mesh_axis": "tp"},
    )
    g_rs_edge = LogicalGraph(
        nodes={
            "x": LogicalNode(id="x", op_type="Placeholder", shape_metadata=(2, 32)),
            "rs_nm": n_rs_no_mesh,
            "rs_oob": n_rs_oob,
        },
        inputs=["x"],
        outputs=["rs_nm", "rs_oob"],
        mesh=mesh,
    )
    assert len(v.validate_sharding_propagation(g_rs_edge)) == 0

    # AllToAll without shape_metadata
    n_a2a_no_shape = LogicalNode(
        id="a2a_ns",
        op_type="AllToAll",
        attributes={"split_axis": 0, "concat_axis": 1},
    )
    g_a2a_edge = LogicalGraph(
        nodes={"a2a_ns": n_a2a_no_shape},
        inputs=["a2a_ns"],
        outputs=["a2a_ns"],
    )
    assert len(v.validate_sharding_propagation(g_a2a_edge)) == 0
