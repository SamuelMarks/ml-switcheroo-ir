"""Tests for WebGPU WGSL dialect enhancements: uniform buffer alignment, builtins, and storage access modes."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_ir.validator import ValidationLevel, Validator


def test_wgsl_uniform_buffer_alignment_validation() -> None:
    """Verify WGSL uniform buffer struct alignment and array stride 16-byte divisibility rules."""
    v = Validator(level=ValidationLevel.STRICT)

    # Valid 16-byte aligned uniform struct and array stride
    node_valid = LogicalNode(
        id="u_valid",
        op_type="fma",
        domain="webgpu_wgsl",
        shape_metadata=(1,),
        attributes={
            "address_space": "uniform",
            "struct_alignment": 16,
            "array_stride": 32,
        },
    )
    assert v.validate_node(node_valid) == []

    # Invalid struct alignment (not multiple of 16)
    node_bad_align = LogicalNode(
        id="u_bad_align",
        op_type="fma",
        domain="webgpu_wgsl",
        shape_metadata=(1,),
        attributes={
            "address_space": "uniform",
            "struct_alignment": 8,
        },
    )
    errs = v.validate_node(node_bad_align)
    assert any(
        "uniform buffer struct alignment must be a multiple of 16 bytes" in e.message
        for e in errs
    )

    # Invalid array stride (not multiple of 16)
    node_bad_stride = LogicalNode(
        id="u_bad_stride",
        op_type="fma",
        domain="webgpu_wgsl",
        shape_metadata=(1,),
        attributes={
            "address_space": "uniform",
            "array_stride": 12,
        },
    )
    errs = v.validate_node(node_bad_stride)
    assert any(
        "uniform buffer array stride must be a multiple of 16 bytes" in e.message
        for e in errs
    )


def test_wgsl_compute_builtins_validation() -> None:
    """Verify WGSL compute entrypoint builtin attributes."""
    v = Validator(level=ValidationLevel.STRICT)

    # Valid builtins
    valid_builtins = [
        "workgroup_id",
        "local_invocation_id",
        "global_invocation_id",
        "local_invocation_index",
        "num_workgroups",
    ]
    for b in valid_builtins:
        node = LogicalNode(
            id=f"b_{b}",
            op_type="dot",
            domain="webgpu_wgsl",
            shape_metadata=(1,),
            attributes={"builtin": b},
        )
        assert v.validate_node(node) == []

    # Invalid builtin
    node_invalid_builtin = LogicalNode(
        id="b_invalid",
        op_type="dot",
        domain="webgpu_wgsl",
        shape_metadata=(1,),
        attributes={"builtin": "vertex_index"},  # Not allowed in compute entrypoint
    )
    errs = v.validate_node(node_invalid_builtin)
    assert any("Invalid WGSL compute builtin 'vertex_index'" in e.message for e in errs)


def test_wgsl_storage_buffer_access_modes() -> None:
    """Verify storage buffer access mode checks (read vs read_write) against mutating operations."""
    v = Validator(level=ValidationLevel.STRICT)

    # Mutating op on read-only storage buffer is illegal
    store_node = LogicalNode(
        id="s_store",
        op_type="storageStore",
        domain="webgpu_wgsl",
        shape_metadata=(1,),
        attributes={"address_space": "storage, read"},
    )
    errs = v.validate_node(store_node)
    assert any(
        "Mutating operation 'storageStore' is illegal on read-only storage buffer"
        in e.message
        for e in errs
    )

    # Atomic op on read-only storage buffer is illegal
    atomic_node = LogicalNode(
        id="s_atomic",
        op_type="atomicAdd",
        domain="webgpu_wgsl",
        shape_metadata=(1,),
        attributes={"address_space": "storage_read"},
    )
    errs = v.validate_node(atomic_node)
    assert any(
        "Mutating operation 'atomicAdd' is illegal on read-only storage buffer"
        in e.message
        for e in errs
    )

    # Custom mutating flag on read-only storage buffer
    custom_mutating = LogicalNode(
        id="s_custom_mut",
        op_type="dot",
        domain="webgpu_wgsl",
        shape_metadata=(1,),
        attributes={"address_space": "storage, read", "is_mutating": True},
    )
    errs = v.validate_node(custom_mutating)
    assert any(
        "Mutating operation 'dot' is illegal on read-only storage buffer" in e.message
        for e in errs
    )

    # Mutating op on read_write storage buffer is legal
    store_rw_node = LogicalNode(
        id="s_store_rw",
        op_type="storageStore",
        domain="webgpu_wgsl",
        shape_metadata=(1,),
        attributes={"address_space": "storage, read_write"},
    )
    assert v.validate_node(store_rw_node) == []

    # Non-mutating op on read-only storage buffer is legal
    dot_read_node = LogicalNode(
        id="s_dot_read",
        op_type="dot",
        domain="webgpu_wgsl",
        shape_metadata=(1,),
        attributes={"address_space": "storage, read"},
    )
    assert v.validate_node(dot_read_node) == []
