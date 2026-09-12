"""Tests for NVIDIA SASS control codes, barrier masks, and scoreboarding dependency latency."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.validator import ValidationLevel, Validator


def test_sass_stall_count_validation() -> None:
    """Verify SASS instruction stall count attribute validation (0 to 15 cycles)."""
    v = Validator(level=ValidationLevel.STRICT)

    # Valid stall counts
    valid_node_0 = LogicalNode(
        id="s0",
        op_type="FFMA",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"stall_count": 0},
    )
    assert v.validate_node(valid_node_0) == []

    valid_node_15 = LogicalNode(
        id="s1",
        op_type="FFMA",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"stall": 15},
    )
    assert v.validate_node(valid_node_15) == []

    # Invalid negative stall count
    neg_node = LogicalNode(
        id="s_neg",
        op_type="FFMA",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"stall_count": -1},
    )
    errs = v.validate_node(neg_node)
    assert any(
        "stall count must be an integer between 0 and 15" in e.message for e in errs
    )

    # Invalid stall count > 15
    large_node = LogicalNode(
        id="s_large",
        op_type="FFMA",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"stall_count": 16},
    )
    errs = v.validate_node(large_node)
    assert any(
        "stall count must be an integer between 0 and 15" in e.message for e in errs
    )

    # Invalid boolean or float stall count
    bad_type_node = LogicalNode(
        id="s_bad",
        op_type="FFMA",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"stall_count": True},
    )
    errs = v.validate_node(bad_type_node)
    assert any(
        "stall count must be an integer between 0 and 15" in e.message for e in errs
    )


def test_sass_yield_flag_validation() -> None:
    """Verify SASS instruction yield flag attribute validation."""
    v = Validator(level=ValidationLevel.STRICT)

    # Valid yield flags
    for yf in ["Y", "-", "yield", "noyield", True, False]:
        node = LogicalNode(
            id="s_y",
            op_type="FFMA",
            domain="nvidia_sass",
            shape_metadata=(1,),
            attributes={"yield_flag": yf},
        )
        assert v.validate_node(node) == []

    # Invalid yield flag string
    bad_yield_str = LogicalNode(
        id="s_bad_y",
        op_type="FFMA",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"yield_flag": "INVALID"},
    )
    errs = v.validate_node(bad_yield_str)
    assert any("Invalid SASS yield flag" in e.message for e in errs)

    # Invalid yield flag type
    bad_yield_type = LogicalNode(
        id="s_bad_yt",
        op_type="FFMA",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"yield_flag": 123},
    )
    errs = v.validate_node(bad_yield_type)
    assert any("Invalid SASS yield flag" in e.message for e in errs)


def test_sass_barrier_mask_validation() -> None:
    """Verify SASS read, write, and wait barrier mask attribute validation."""
    v = Validator(level=ValidationLevel.STRICT)

    # Valid barrier configurations
    node_valid = LogicalNode(
        id="s_bar",
        op_type="LDG",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={
            "read_barrier": 1,
            "write_barrier": 0,
            "wait_barrier_mask": 3,
        },
    )
    assert v.validate_node(node_valid) == []

    # Valid list of barriers
    node_valid_list = LogicalNode(
        id="s_bar_list",
        op_type="LDG",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"read_barrier": [0, 1, 5]},
    )
    assert v.validate_node(node_valid_list) == []

    # Valid string mask
    node_valid_str = LogicalNode(
        id="s_bar_str",
        op_type="LDG",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"wait_barrier_mask": "--01--"},
    )
    assert v.validate_node(node_valid_str) == []

    # Invalid integer barrier (> 63)
    node_bad_int = LogicalNode(
        id="s_bad_int",
        op_type="LDG",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"read_barrier": 64},
    )
    assert any(
        "Must be in range 0..63" in e.message for e in v.validate_node(node_bad_int)
    )

    # Invalid list entry (> 5)
    node_bad_list = LogicalNode(
        id="s_bad_list",
        op_type="LDG",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"write_barrier": [0, 6]},
    )
    assert any(
        "Entries must be barrier indices 0..5" in e.message
        for e in v.validate_node(node_bad_list)
    )

    # Invalid string mask
    node_bad_str = LogicalNode(
        id="s_bad_str",
        op_type="LDG",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"wait_barrier_mask": "012345678"},
    )
    assert any(
        "Invalid SASS barrier mask string" in e.message
        for e in v.validate_node(node_bad_str)
    )

    # Invalid barrier type
    node_bad_type = LogicalNode(
        id="s_bad_t",
        op_type="LDG",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"read_barrier": 3.14},
    )
    assert any(
        "Invalid SASS barrier mask type" in e.message
        for e in v.validate_node(node_bad_type)
    )


def test_sass_scoreboarding_latency_hazard_detection() -> None:
    """Verify detection of scoreboard dependency hazards between SASS instructions."""
    v = Validator(level=ValidationLevel.STRICT)

    # 1. Pipeline latency hazard: Producer FFMA writes R1 (latency 4), Consumer reads R1 with stall 0
    producer = LogicalNode(
        id="inst_prod",
        op_type="FFMA",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"dst_reg": "R1", "stall_count": 0},
        outputs=["R1"],
    )
    consumer = LogicalNode(
        id="inst_cons",
        op_type="FADD",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"src_regs": ["R1", "R2"], "dst_reg": "R3"},
        inputs=["R1"],
    )

    hazards = v.validate_sass_scoreboarding([producer, consumer])
    assert len(hazards) == 1
    assert "Scoreboard dependency hazard" in hazards[0].message
    assert "reads register 'R1' produced by 'inst_prod'" in hazards[0].message

    # 2. Hazard avoided by sufficient stall count (stall 4 on producer)
    producer_stalled = LogicalNode(
        id="inst_prod_stall",
        op_type="FFMA",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"dst_reg": "R1", "stall_count": 4},
        outputs=["R1"],
    )
    assert v.validate_sass_scoreboarding([producer_stalled, consumer]) == []

    # 3. Hazard avoided by barrier synchronization: LDG (latency 200) write_barrier=0, consumer wait_barrier_mask=1 (bit 0)
    ldg = LogicalNode(
        id="inst_ldg",
        op_type="LDG",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"dst_reg": ["R5"], "write_barrier": 0},
        outputs=["R5"],
    )
    sync_consumer_int = LogicalNode(
        id="inst_sync_cons_int",
        op_type="FADD",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"src_regs": "R5", "wait_barrier_mask": 1},
        inputs=["R5"],
    )
    assert v.validate_sass_scoreboarding([ldg, sync_consumer_int]) == []

    sync_consumer_list = LogicalNode(
        id="inst_sync_cons_list",
        op_type="FADD",
        domain="nvidia_sass",
        shape_metadata=(1,),
        attributes={"src_regs": "R5", "wait_barrier_mask": [0]},
        inputs=["R5"],
    )
    assert v.validate_sass_scoreboarding([ldg, sync_consumer_list]) == []

    # 4. Scoreboarding check with non-SASS node skipped
    onnx_node = LogicalNode(
        id="onnx_node",
        op_type="Relu",
        domain="ai.onnx",
        shape_metadata=(1,),
    )
    assert v.validate_sass_scoreboarding([onnx_node]) == []

    # 5. Scoreboarding check in WARNING mode
    v_warn = Validator(level=ValidationLevel.WARNING)
    warn_hazards = v_warn.validate_sass_scoreboarding([producer, consumer])
    assert len(warn_hazards) == 1
    assert warn_hazards[0].level == ValidationLevel.WARNING

    # 6. Scoreboarding check integrated into LogicalGraph validation
    graph = LogicalGraph(
        name="SassHazardGraph",
        nodes=[producer, consumer],
        outputs=["inst_cons"],
    )
    graph_errors = v.validate_graph(graph)
    assert any("Scoreboard dependency hazard" in e.message for e in graph_errors)
