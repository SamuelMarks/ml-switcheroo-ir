"""Tests for AMD RDNA3 / GFX11 dual-issue VOPD pairing and register bank conflict validation."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_ir.validator import (
    ValidationLevel,
    Validator,
    _extract_dst_vgpr,
    _extract_vgpr_indices,
    validate_vopd_pairing,
)


def test_valid_vopd_instruction_pairing() -> None:
    """Verify that legal VOPD pairs without bank or destination conflicts pass validation."""
    opX = LogicalNode(
        id="inst_x",
        op_type="V_DUAL_FMAC_F32",
        domain="amd_rdna",
        attributes={
            "dst_vgpr": 2,
            "src_vgprs": [0, 1],  # Bank 0, Bank 1
        },
    )
    opY = LogicalNode(
        id="inst_y",
        op_type="V_DUAL_ADD_F32",
        domain="amd_rdna",
        attributes={
            "dst_vgpr": 3,
            "src_vgprs": [2, 3],  # Bank 2, Bank 3
        },
    )

    errors = validate_vopd_pairing(opX, opY)
    assert errors == []

    # Also test via Validator instance method
    v = Validator(level=ValidationLevel.STRICT)
    assert v.validate_vopd_pairing(opX, opY) == []


def test_vopd_identical_vgpr_same_bank_allowed() -> None:
    """Verify that identical VGPR reads across opX and opY on the same bank are legal."""
    opX = LogicalNode(
        id="inst_x",
        op_type="V_FMAC_F32",  # Mapped via RDNA_TO_VOPD_MAP
        domain="amd_rdna",
        attributes={
            "dst": "v10",
            "vgpr_operands": [0, 1],  # Bank 0, Bank 1
        },
    )
    opY = LogicalNode(
        id="inst_y",
        op_type="V_ADD_F32",  # Mapped via RDNA_TO_VOPD_MAP
        domain="amd_rdna",
        attributes={
            "dst": "v11",
            "vgpr_operands": [0, 2],  # Bank 0 (same v0!), Bank 2
        },
    )

    errors = validate_vopd_pairing(opX, opY)
    assert errors == []


def test_vopd_invalid_domain() -> None:
    """Verify error when instructions belong to a non-AMD domain."""
    opX = LogicalNode(
        id="inst_x",
        op_type="V_DUAL_FMAC_F32",
        domain="nvidia_sass",
    )
    opY = LogicalNode(
        id="inst_y",
        op_type="V_DUAL_ADD_F32",
        domain="ai.onnx",
    )

    errors = validate_vopd_pairing(opX, opY)
    assert len(errors) == 2
    assert any("opX domain must be 'amd_rdna'" in e.message for e in errors)
    assert any("opY domain must be 'amd_rdna'" in e.message for e in errors)


def test_vopd_unsupported_opcodes() -> None:
    """Verify error when instructions cannot be issued as VOPD opcodes."""
    opX = LogicalNode(
        id="inst_x",
        op_type="V_UNKNOWN_OP",
        domain="amd_rdna",
    )
    opY = LogicalNode(
        id="inst_y",
        op_type="V_DOT4_I32_I8",  # Not in VOPD opcode set
        domain="amd_rdna",
    )

    errors = validate_vopd_pairing(opX, opY)
    assert any(
        "Opcode 'V_UNKNOWN_OP' is not a valid RDNA3 VOPD instruction." in e.message
        for e in errors
    )
    assert any(
        "Opcode 'V_DOT4_I32_I8' is not a valid RDNA3 VOPD instruction." in e.message
        for e in errors
    )


def test_vopd_illegal_slot_assignment() -> None:
    """Verify that V_DUAL_CNDMASK_B32 cannot be issued in Slot Y and V_DUAL_DOT2ACC_F32_F16 cannot be in Slot X."""
    opX = LogicalNode(
        id="inst_x",
        op_type="V_DUAL_FMAC_F32",
        domain="amd_rdna",
        attributes={"dst": 0},
    )
    opY = LogicalNode(
        id="inst_y",
        op_type="V_DUAL_CNDMASK_B32",  # Only permitted in Slot X
        domain="amd_rdna",
        attributes={"dst": 1},
    )

    errors = validate_vopd_pairing(opX, opY)
    assert any(
        "Opcode 'V_DUAL_CNDMASK_B32' cannot be issued in VOPD Slot Y." in e.message
        for e in errors
    )

    # Test Slot X restriction: V_DUAL_DOT2ACC_F32_F16 is only permitted in Slot Y
    opX_bad_slot = LogicalNode(
        id="inst_x_bad",
        op_type="V_DUAL_DOT2ACC_F32_F16",
        domain="amd_rdna",
        attributes={"dst": 0},
    )
    opY_good = LogicalNode(
        id="inst_y_good",
        op_type="V_DUAL_ADD_F32",
        domain="amd_rdna",
        attributes={"dst": 1},
    )
    errors_x = validate_vopd_pairing(opX_bad_slot, opY_good)
    assert any(
        "Opcode 'V_DUAL_DOT2ACC_F32_F16' cannot be issued in VOPD Slot X." in e.message
        for e in errors_x
    )


def test_vopd_destination_register_conflict() -> None:
    """Verify that opX and opY cannot write to the same destination VGPR."""
    opX = LogicalNode(
        id="inst_x",
        op_type="V_DUAL_FMAC_F32",
        domain="amd_rdna",
        attributes={"dst_vgpr": 4},
        outputs=["v4"],
    )
    opY = LogicalNode(
        id="inst_y",
        op_type="V_DUAL_ADD_F32",
        domain="amd_rdna",
        attributes={"dst_vgpr": 4},
        outputs=["v4"],
    )

    errors = validate_vopd_pairing(opX, opY)
    assert any("VOPD destination conflict" in e.message for e in errors)


def test_vopd_register_bank_conflict() -> None:
    """Verify register bank conflict detection when different registers access the same bank."""
    # Bank 0 conflict: v0 (0 % 4 == 0) and v4 (4 % 4 == 0, 4 != 0)
    opX = LogicalNode(
        id="inst_x",
        op_type="V_DUAL_FMAC_F32",
        domain="amd_rdna",
        attributes={"dst_vgpr": 10, "src_vgprs": [0]},
    )
    opY = LogicalNode(
        id="inst_y",
        op_type="V_DUAL_ADD_F32",
        domain="amd_rdna",
        attributes={"dst_vgpr": 11, "src_vgprs": [4]},
    )

    errors = validate_vopd_pairing(opX, opY)
    assert any("VOPD register bank conflict on bank 0" in e.message for e in errors)


def test_extract_vgpr_helpers() -> None:
    """Verify VGPR index extraction from various attributes, inputs, and outputs."""
    # Test from inputs and outputs with string names
    node1 = LogicalNode(
        id="n1",
        op_type="V_ADD_F32",
        domain="amd_rdna",
        inputs=["v8", "vgpr9"],
        outputs=["v12"],
    )
    assert _extract_vgpr_indices(node1) == [8, 9]
    assert _extract_dst_vgpr(node1) == 12

    # Test with no vgpr information
    node2 = LogicalNode(
        id="n2",
        op_type="V_ADD_F32",
        domain="amd_rdna",
        inputs=["x", "y"],
        outputs=["z"],
    )
    assert _extract_vgpr_indices(node2) == []
    assert _extract_dst_vgpr(node2) is None

    # Test with string attributes
    node3 = LogicalNode(
        id="n3",
        op_type="V_ADD_F32",
        domain="amd_rdna",
        attributes={
            "vgpr_operands": ["vgpr5", 6],
            "src_vgprs": ["v7"],
            "dst_vgpr": "vgpr15",
        },
    )
    assert _extract_vgpr_indices(node3) == [5, 6, 7]
    assert _extract_dst_vgpr(node3) == 15

    # Test dst with no digits falling through to outputs
    node4 = LogicalNode(
        id="n4",
        op_type="V_ADD_F32",
        domain="amd_rdna",
        attributes={"dst_vgpr": "invalid_no_digit"},
        outputs=["v20"],
    )
    assert _extract_dst_vgpr(node4) == 20
