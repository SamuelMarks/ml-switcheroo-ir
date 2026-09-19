"""Tests for distributed sharding, collective primitives, and communication cost estimation."""

from __future__ import annotations

from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
    estimate_communication_volume,
    estimate_graph_communication_volume,
)
from ml_switcheroo_ir.validator import ValidationLevel, Validator


def test_sharding_rank_and_divisibility_validation() -> None:
    """Verify PartitionSpec rank bounds and dimension divisibility against LogicalMesh."""
    v = Validator(level=ValidationLevel.STRICT)
    mesh = LogicalMesh(shape={"data": 4, "model": 2})

    # 1. Valid sharding with divisible dimensions: (16, 8) sharded on (data=4, model=2) -> 16 % 4 == 0, 8 % 2 == 0
    valid_node = LogicalNode(
        id="valid_node",
        op_type="Relu",
        domain="ai.onnx",
        shape_metadata=(16, 8),
        sharding=PartitionSpec(axes=("data", "model")),
    )
    assert v.validate_sharding(valid_node, mesh) == []

    # 2. PartitionSpec rank exceeds tensor rank
    excess_rank_node = LogicalNode(
        id="excess_node",
        op_type="Relu",
        domain="ai.onnx",
        shape_metadata=(16, 8),
        sharding=PartitionSpec(axes=("data", "model", None)),
    )
    errs = v.validate_sharding(excess_rank_node, mesh)
    assert any(
        "PartitionSpec rank (3) exceeds tensor rank (2)" in e.message for e in errs
    )

    # 3. Indivisible dimension: 15 is not divisible by 4 (mesh axis data=4)
    indivisible_node = LogicalNode(
        id="indivisible_node",
        op_type="Relu",
        domain="ai.onnx",
        shape_metadata=(15, 8),
        sharding=PartitionSpec(axes=("data", "model")),
    )
    errs = v.validate_sharding(indivisible_node, mesh)
    assert any(
        "Tensor dimension 0 size 15 is not evenly divisible by mesh axis 'data' size 4"
        in e.message
        for e in errs
    )


def test_sharding_propagation_elementwise_invariants() -> None:
    """Verify sharding propagation checks across elementwise operators."""
    v = Validator(level=ValidationLevel.STRICT)
    mesh = LogicalMesh(shape={"data": 4})

    n_in = LogicalNode(
        id="n_in",
        op_type="Input",
        domain="ml.switcheroo.custom",
        shape_metadata=(16, 16),
        sharding=PartitionSpec(axes=("data", None)),
    )

    # Mismatched elementwise sharding between input and output
    n_add_mismatch = LogicalNode(
        id="n_add_mismatch",
        op_type="Add",
        domain="ai.onnx",
        inputs=["n_in"],
        shape_metadata=(16, 16),
        sharding=PartitionSpec(axes=(None, "data")),  # Disagrees with n_in!
    )
    graph_mismatch = LogicalGraph(
        nodes={n.id: n for n in [n_in, n_add_mismatch]},
        mesh=mesh,
    )
    errs = v.validate_sharding_propagation(graph_mismatch)
    assert len(errs) == 1
    assert "has mismatched sharding between input" in errs[0].message

    # Matching elementwise sharding
    n_add_match = LogicalNode(
        id="n_add_match",
        op_type="Add",
        domain="ai.onnx",
        inputs=["n_in"],
        shape_metadata=(16, 16),
        sharding=PartitionSpec(axes=("data", None)),
    )
    graph_match = LogicalGraph(
        nodes={n.id: n for n in [n_in, n_add_match]},
        mesh=mesh,
    )
    assert v.validate_sharding_propagation(graph_match) == []


def test_sharding_propagation_contraction_and_reduction() -> None:
    """Verify sharding propagation invariants for contraction (MatMul) and reduction operators."""
    v = Validator(level=ValidationLevel.STRICT)
    mesh = LogicalMesh(shape={"data": 4, "model": 2})

    # Contraction mismatch: A contracts axis K sharded on 'model', B contracts axis K sharded on 'data'
    node_a = LogicalNode(
        id="node_a",
        op_type="Relu",
        shape_metadata=(16, 32),
        sharding=PartitionSpec(axes=("data", "model")),
    )
    node_b = LogicalNode(
        id="node_b",
        op_type="Relu",
        shape_metadata=(32, 64),
        sharding=PartitionSpec(
            axes=("data", None)
        ),  # Contracted axis is axis -2 ('data' != 'model')
    )
    node_matmul = LogicalNode(
        id="node_matmul",
        op_type="MatMul",
        inputs=["node_a", "node_b"],
        shape_metadata=(16, 64),
    )
    graph_matmul = LogicalGraph(
        nodes={n.id: n for n in [node_a, node_b, node_matmul]}, mesh=mesh
    )
    errs = v.validate_sharding_propagation(graph_matmul)
    assert any(
        "has mismatched contracted dimension sharding" in e.message for e in errs
    )

    # Reduction leak: Input is sharded on ('data', 'model'), reduction reduces both [0, 1], but output retains both
    node_red_in = LogicalNode(
        id="node_red_in",
        op_type="Input",
        shape_metadata=(16, 32),
        sharding=PartitionSpec(axes=("data", "model")),
    )
    node_reduce = LogicalNode(
        id="node_reduce",
        op_type="ReduceSum",
        inputs=["node_red_in"],
        attributes={"axes": [0, 1]},
        shape_metadata=(16,),
        sharding=PartitionSpec(axes=("data", "model")),  # Both reduced axes retained!
    )
    graph_reduce = LogicalGraph(
        nodes={n.id: n for n in [node_red_in, node_reduce]}, mesh=mesh
    )
    errs_red = v.validate_sharding_propagation(graph_reduce)
    assert len(errs_red) == 2
    assert any("reduces sharded axis 'data'" in e.message for e in errs_red)
    assert any("reduces sharded axis 'model'" in e.message for e in errs_red)


def test_collective_communication_schemas() -> None:
    """Verify registration and validation of collective communication operator schemas."""
    v = Validator(level=ValidationLevel.STRICT)

    # 1. Valid collective.all_reduce
    ar_node = LogicalNode(
        id="ar1",
        op_type="collective.all_reduce",
        domain="collective",
        shape_metadata=(16, 16),
        attributes={
            "reduction_op": "sum",
            "mesh_axis": "data",
            "channel_id": 0,
        },
    )
    assert v.validate_node(ar_node) == []

    # 2. Missing required reduction_op attribute in all_reduce
    ar_bad = LogicalNode(
        id="ar_bad",
        op_type="all_reduce",
        domain="collective",
        shape_metadata=(16, 16),
        attributes={"mesh_axis": "data"},
    )
    errs = v.validate_node(ar_bad)
    assert any(
        "reduction_op" in e.message and "missing" in e.message.lower() for e in errs
    )

    # 3. Valid collective.all_gather
    ag_node = LogicalNode(
        id="ag1",
        op_type="collective.all_gather",
        domain="collective",
        shape_metadata=(16, 16),
        attributes={"gather_dimension": 0, "mesh_axis": "data"},
    )
    assert v.validate_node(ag_node) == []

    # 4. Valid collective.reduce_scatter
    rs_node = LogicalNode(
        id="rs1",
        op_type="collective.reduce_scatter",
        domain="collective",
        shape_metadata=(16, 16),
        attributes={
            "scatter_dimension": 0,
            "reduction_op": "mean",
            "mesh_axis": "data",
        },
    )
    assert v.validate_node(rs_node) == []

    # 5. Valid collective.all_to_all
    a2a_node = LogicalNode(
        id="a2a1",
        op_type="collective.all_to_all",
        domain="collective",
        shape_metadata=(16, 16),
        attributes={
            "split_dimension": 0,
            "concat_dimension": 1,
            "mesh_axis": "model",
        },
    )
    assert v.validate_node(a2a_node) == []

    # 6. Unregistered collective op
    bad_col = LogicalNode(
        id="bad_col",
        op_type="collective.unknown",
        domain="collective",
    )
    assert any(
        "Operator 'collective.unknown' not found" in e.message
        for e in v.validate_node(bad_col)
    )


def test_analytical_communication_cost_estimation() -> None:
    """Verify analytical communication volume calculation for collective operators."""
    mesh = LogicalMesh(shape={"data": 4, "model": 2})

    # Tensor shape: (1024, 1024), dtype float32 (4 bytes) -> 1,048,576 elements * 4 = 4,194,304 bytes (4 MB)
    # Ring all-reduce on 4 devices: 2 * ((4 - 1) / 4) * 4MB = 2 * 0.75 * 4MB = 6 MB = 6,291,456 bytes
    ar_node = LogicalNode(
        id="ar_cost",
        op_type="collective.all_reduce",
        domain="collective",
        shape_metadata=(1024, 1024),
        attributes={"reduction_op": "sum", "mesh_axis": "data", "dtype": "float32"},
    )
    vol_ar = estimate_communication_volume(ar_node, mesh)
    expected_ar = int(2 * (3 / 4) * (1024 * 1024 * 4))
    assert vol_ar == expected_ar

    # All-gather on 2 devices (model=2), dtype float16 (2 bytes):
    # ((2 - 1) / 2) * (1024 * 1024 * 2) = 0.5 * 2 MB = 1 MB = 1,048,576 bytes
    ag_node = LogicalNode(
        id="ag_cost",
        op_type="collective.all_gather",
        domain="collective",
        shape_metadata=(1024, 1024),
        attributes={
            "gather_dimension": 0,
            "mesh_axis": "model",
            "dtype": "float16",
        },
    )
    vol_ag = estimate_communication_volume(ag_node, mesh)
    expected_ag = int(0.5 * (1024 * 1024 * 2))
    assert vol_ag == expected_ag

    # Test single-device (no communication needed)
    single_mesh = LogicalMesh(shape={"data": 1})
    assert estimate_communication_volume(ar_node, single_mesh) == 0

    # Aggregate graph communication volume with both collective and non-collective nodes
    regular_node = LogicalNode(id="reg_relu", op_type="Relu", domain="ai.onnx")
    graph = LogicalGraph(
        name="CollectiveGraph",
        nodes={n.id: n for n in [ar_node, ag_node, regular_node]},
        mesh=mesh,
    )
    summary = estimate_graph_communication_volume(graph)
    assert summary["total_volume_bytes"] == expected_ar + expected_ag
    assert summary["by_axis"]["data"] == expected_ar
    assert summary["by_axis"]["model"] == expected_ag
    assert "collective.all_reduce" in summary["by_op"]

    # Also test Validator instance methods
    v = Validator(level=ValidationLevel.STRICT)
    assert v.estimate_communication_volume(ar_node, mesh) == expected_ar
    assert (
        v.estimate_graph_communication_volume(graph)["total_volume_bytes"]
        == expected_ar + expected_ag
    )

    # Test symbolic shape metadata (e.g. ("B", 10)) where "B" is ignored in multiplication
    node_sym = LogicalNode(
        id="col_sym",
        op_type="all_reduce",
        domain="collective",
        shape_metadata=("B", 10),
        attributes={"mesh_axis": "data", "dtype": "float32", "reduction_op": "sum"},
    )
    vol_sym = v.estimate_communication_volume(node_sym, mesh)
    assert vol_sym == int(2 * (3 / 4) * (10 * 4))

    # Test int64 dtype (8 bytes) and int8 dtype (1 byte)
    node_64 = LogicalNode(
        id="col_64",
        op_type="all_reduce",
        domain="collective",
        shape_metadata=(10, 10),
        attributes={"mesh_axis": "data", "dtype": "int64", "reduction_op": "sum"},
    )
    vol_64 = v.estimate_communication_volume(node_64, mesh)
    assert vol_64 == int(2 * (3 / 4) * (100 * 8))

    node_8 = LogicalNode(
        id="col_8",
        op_type="all_reduce",
        domain="collective",
        shape_metadata=(10, 10),
        attributes={"mesh_axis": "data", "dtype": "int8", "reduction_op": "sum"},
    )
    vol_8 = v.estimate_communication_volume(node_8, mesh)
    assert vol_8 == int(2 * (3 / 4) * (100 * 1))

    # Test fallback collective op (e.g. broadcast) and fallback without shape_metadata
    node_broadcast = LogicalNode(
        id="col_bcast",
        op_type="collective.broadcast",
        domain="collective",
        attributes={"mesh_axis": "unknown_axis", "tensor_numel": 50, "device_count": 4},
    )
    assert v.estimate_communication_volume(node_broadcast, None) == 50 * 4


def test_sharding_and_pipeline_edge_branches() -> None:
    """Verify edge case branches for sharding propagation, checkpointing, and pipeline boundaries."""
    # 1. Elementwise in WARNING mode
    v_warn = Validator(level=ValidationLevel.WARNING)
    n_in = LogicalNode(
        id="n_in",
        op_type="Input",
        shape_metadata=(8, 8),
        sharding=PartitionSpec(axes=("data", None)),
    )
    n_add_warn = LogicalNode(
        id="n_add_warn",
        op_type="Add",
        inputs=["n_in"],
        shape_metadata=(8, 8),
        sharding=PartitionSpec(axes=(None, "data")),
    )
    g_warn = LogicalGraph(
        nodes={n.id: n for n in [n_in, n_add_warn]},
        mesh=LogicalMesh(shape={"data": 2}),
    )
    errs_warn = v_warn.validate_sharding_propagation(g_warn)
    assert len(errs_warn) == 1
    assert errs_warn[0].level == ValidationLevel.WARNING

    # 2. MatMul with matching contracted dimension sharding
    node_a_ok = LogicalNode(
        id="na_ok",
        op_type="Relu",
        shape_metadata=(8, 16),
        sharding=PartitionSpec(axes=("data", "model")),
    )
    node_b_ok = LogicalNode(
        id="nb_ok",
        op_type="Relu",
        shape_metadata=(16, 32),
        sharding=PartitionSpec(axes=("model", None)),
    )
    node_mm_ok = LogicalNode(
        id="nmm_ok",
        op_type="MatMul",
        inputs=["na_ok", "nb_ok"],
        shape_metadata=(8, 32),
    )
    g_mm_ok = LogicalGraph(
        nodes={n.id: n for n in [node_a_ok, node_b_ok, node_mm_ok]},
        mesh=LogicalMesh(shape={"data": 2, "model": 2}),
    )
    assert v_warn.validate_sharding_propagation(g_mm_ok) == []

    # 3. Activation checkpoint invalid type (not str, not bool)
    node_bad_chk_type = LogicalNode(
        id="n_bad_chk",
        op_type="Relu",
        shape_metadata=(1,),
        attributes={"activation_checkpoint": 12345},
    )
    errs_chk_type = v_warn.validate_pipeline_and_checkpointing(
        LogicalGraph(nodes={node_bad_chk_type.id: node_bad_chk_type})
    )
    assert any(
        "Invalid activation checkpoint tag type" in e.message for e in errs_chk_type
    )

    # 4. Pipeline stage skip with boundary marker
    s0 = LogicalNode(
        id="s0", op_type="Relu", shape_metadata=(1,), attributes={"pipeline_stage": 0}
    )
    s2_boundary = LogicalNode(
        id="s2_b",
        op_type="Relu",
        shape_metadata=(1,),
        inputs=["s0"],
        attributes={"pipeline_stage": 2, "pipeline_boundary": True},
    )
    g_skip_b = LogicalGraph(nodes={n.id: n for n in [s0, s2_boundary]})
    assert v_warn.validate_pipeline_and_checkpointing(g_skip_b) == []

    # Pipeline stage skip with P2P op
    s2_p2p = LogicalNode(
        id="s2_p2p",
        op_type="P2P",
        shape_metadata=(1,),
        inputs=["s0"],
        attributes={"pipeline_stage": 2},
    )
    g_skip_p2p = LogicalGraph(nodes={n.id: n for n in [s0, s2_p2p]})
    assert v_warn.validate_pipeline_and_checkpointing(g_skip_p2p) == []

    # Same-stage connection and stage_id alias
    s1_a = LogicalNode(
        id="s1_a",
        op_type="Relu",
        shape_metadata=(1,),
        attributes={"stage_id": 1},
    )
    s1_b = LogicalNode(
        id="s1_b",
        op_type="Relu",
        shape_metadata=(1,),
        inputs=["s1_a"],
        attributes={"stage_id": 1},
    )
    g_same_stage = LogicalGraph(nodes={n.id: n for n in [s1_a, s1_b]})
    assert v_warn.validate_pipeline_and_checkpointing(g_same_stage) == []

    # Valid reduction without sharding leak
    node_red_ok_in = LogicalNode(
        id="red_in_ok",
        op_type="Input",
        shape_metadata=(8, 16, 32),
        sharding=PartitionSpec(axes=("batch", "data", "model")),
    )
    node_red_ok = LogicalNode(
        id="red_ok",
        op_type="ReduceSum",
        inputs=["red_in_ok"],
        attributes={"axes": [2, 1]},
        shape_metadata=(8,),
        sharding=PartitionSpec(axes=("batch",)),  # Does not retain 'model' or 'data'!
    )
    g_red_ok = LogicalGraph(
        nodes={n.id: n for n in [node_red_ok_in, node_red_ok]},
        mesh=LogicalMesh(shape={"batch": 2, "data": 2, "model": 2}),
    )
    assert v_warn.validate_sharding_propagation(g_red_ok) == []

    # MatMul with inputs that have no sharding
    node_a_none = LogicalNode(id="na_n", op_type="Relu", shape_metadata=(8, 16))
    node_b_none = LogicalNode(id="nb_n", op_type="Relu", shape_metadata=(16, 32))
    node_mm_none = LogicalNode(id="nmm_n", op_type="MatMul", inputs=["na_n", "nb_n"])
    g_mm_none = LogicalGraph(
        nodes={n.id: n for n in [node_a_none, node_b_none, node_mm_none]}
    )
    assert v_warn.validate_sharding_propagation(g_mm_none) == []

    # Reduction where input has no sharding, and reduction where sharded axis is None
    node_red_none_in = LogicalNode(
        id="r_none_in", op_type="Input", shape_metadata=(8, 8)
    )
    node_red_none = LogicalNode(
        id="r_none",
        op_type="ReduceSum",
        inputs=["r_none_in"],
        attributes={"axes": [0]},
        sharding=PartitionSpec(axes=("data",)),
    )
    g_red_none = LogicalGraph(
        nodes={n.id: n for n in [node_red_none_in, node_red_none]}
    )
    assert v_warn.validate_sharding_propagation(g_red_none) == []

    node_red_none_axis_in = LogicalNode(
        id="r_na_in",
        op_type="Input",
        shape_metadata=(8, 8),
        sharding=PartitionSpec(axes=(None, "data")),
    )
    node_red_none_axis = LogicalNode(
        id="r_na",
        op_type="ReduceSum",
        inputs=["r_na_in"],
        attributes={"axes": [0]},
        sharding=PartitionSpec(axes=("data",)),
    )
    g_red_na = LogicalGraph(
        nodes={n.id: n for n in [node_red_none_axis_in, node_red_none_axis]}
    )
    assert v_warn.validate_sharding_propagation(g_red_na) == []

    # Pipeline node connected to input with no pipeline stage and boolean pipeline stage
    s_nostage = LogicalNode(id="s_no", op_type="Input")
    s_bool = LogicalNode(
        id="s_bool", op_type="Input", attributes={"pipeline_stage": True}
    )
    s_stage1 = LogicalNode(
        id="s1_step",
        op_type="Relu",
        inputs=["s_no", "s_bool"],
        attributes={"pipeline_stage": 1},
    )
    g_no_stage = LogicalGraph(nodes={n.id: n for n in [s_nostage, s_bool, s_stage1]})
    errs_stage = v_warn.validate_pipeline_and_checkpointing(g_no_stage)
    assert any(
        "Pipeline stage ID must be a non-negative integer, got True" in e.message
        for e in errs_stage
    )

    # Normal single-stage progression: stage 0 -> stage 1
    s_norm0 = LogicalNode(id="sn0", op_type="Relu", attributes={"pipeline_stage": 0})
    s_norm1 = LogicalNode(
        id="sn1", op_type="Relu", inputs=["sn0"], attributes={"pipeline_stage": 1}
    )
    g_norm = LogicalGraph(nodes={n.id: n for n in [s_norm0, s_norm1]})
    assert v_warn.validate_pipeline_and_checkpointing(g_norm) == []

    # Out-of-bounds reduction axis
    node_red_oob = LogicalNode(
        id="r_oob",
        op_type="ReduceSum",
        inputs=["red_in_ok"],
        attributes={"axes": [99]},
        sharding=PartitionSpec(axes=("batch",)),
    )
    g_red_oob = LogicalGraph(nodes={n.id: n for n in [node_red_ok_in, node_red_oob]})
    assert v_warn.validate_sharding_propagation(g_red_oob) == []

    # Pipeline node with external input not in graph.nodes
    s_ext = LogicalNode(
        id="s_ext",
        op_type="Relu",
        inputs=["unresolved_graph_input"],
        attributes={"pipeline_stage": 1},
    )
    g_ext = LogicalGraph(nodes={s_ext.id: s_ext})
    assert v_warn.validate_pipeline_and_checkpointing(g_ext) == []


def test_pipeline_and_checkpointing_validation() -> None:
    """Verify pipeline stage dependency orderings and activation checkpointing tag validation."""
    v = Validator(level=ValidationLevel.STRICT)

    # 1. Valid activation checkpointing tags
    for tag in ["recompute", "offload", "save", "full", True, False]:
        node = LogicalNode(
            id="chk_node",
            op_type="Relu",
            domain="ai.onnx",
            shape_metadata=(1,),
            attributes={"activation_checkpoint": tag},
        )
        assert (
            v.validate_pipeline_and_checkpointing(LogicalGraph(nodes={node.id: node}))
            == []
        )

    # 2. Invalid activation checkpointing tag
    bad_tag_node = LogicalNode(
        id="chk_bad",
        op_type="Relu",
        domain="ai.onnx",
        shape_metadata=(1,),
        attributes={"activation_checkpoint": "invalid_policy"},
    )
    errs = v.validate_pipeline_and_checkpointing(
        LogicalGraph(nodes={bad_tag_node.id: bad_tag_node})
    )
    assert any(
        "Invalid activation checkpoint tag 'invalid_policy'" in e.message for e in errs
    )

    # 3. Invalid negative pipeline stage
    neg_stage_node = LogicalNode(
        id="neg_stage",
        op_type="Relu",
        domain="ai.onnx",
        shape_metadata=(1,),
        attributes={"pipeline_stage": -1},
    )
    errs = v.validate_pipeline_and_checkpointing(
        LogicalGraph(nodes={neg_stage_node.id: neg_stage_node})
    )
    assert any(
        "Pipeline stage ID must be a non-negative integer" in e.message for e in errs
    )

    # 4. Pipeline stage hazard: backward edge from stage 2 to stage 0
    node_stage2 = LogicalNode(
        id="stage2_op",
        op_type="Relu",
        shape_metadata=(1,),
        attributes={"pipeline_stage": 2},
    )
    node_stage0 = LogicalNode(
        id="stage0_op",
        op_type="Relu",
        shape_metadata=(1,),
        inputs=["stage2_op"],  # Backward edge!
        attributes={"pipeline_stage": 0},
    )
    graph_backward = LogicalGraph(nodes={n.id: n for n in [node_stage2, node_stage0]})
    errs_back = v.validate_pipeline_and_checkpointing(graph_backward)
    assert any(
        "Pipeline stage dependency hazard: backward edge from stage 2" in e.message
        for e in errs_back
    )

    # 5. Skipped stage without boundary marker produces warning
    node_s0 = LogicalNode(
        id="s0_op",
        op_type="Relu",
        shape_metadata=(1,),
        attributes={"pipeline_stage": 0},
    )
    node_s2 = LogicalNode(
        id="s2_op",
        op_type="Relu",
        shape_metadata=(1,),
        inputs=["s0_op"],
        attributes={"pipeline_stage": 2},  # Skipped stage 1 without boundary
    )
    graph_skip = LogicalGraph(nodes={n.id: n for n in [node_s0, node_s2]})
    errs_skip = v.validate_pipeline_and_checkpointing(graph_skip)
    assert any(
        "Cross-stage dataflow skipping stages (0 -> 2)" in e.message for e in errs_skip
    )
    assert any(e.level == ValidationLevel.WARNING for e in errs_skip)
