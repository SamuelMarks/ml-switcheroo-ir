"""Tests for Quantization Dialect, low-precision DTypes, microscaling, and grouped INT4 formats."""

from __future__ import annotations

from ml_switcheroo_ir import DType, LogicalNode
from ml_switcheroo_ir.validator import ValidationLevel, Validator


def test_ocp_fp8_dtypes() -> None:
    """Verify first-class DType support for modern OCP FP8 formats."""
    assert DType.fp8_e4m3fn.value == "fp8_e4m3fn"
    assert DType.fp8_e4m3fnuz.value == "fp8_e4m3fnuz"
    assert DType.fp8_e5m2.value == "fp8_e5m2"
    assert DType.fp8_e5m2fnuz.value == "fp8_e5m2fnuz"


def test_block_quantize_validation() -> None:
    """Verify microscaling block_quantize operator validation and tile divisibility."""
    v = Validator(level=ValidationLevel.STRICT)

    # 1. Valid block_quantize (dim 64 divisible by block_size 32)
    valid_bq = LogicalNode(
        id="bq1",
        op_type="quantization.block_quantize",
        domain="quantization",
        shape_metadata=(128, 64),
        attributes={
            "block_size": 32,
            "quant_dtype": "fp8_e4m3fn",
            "axis": -1,
        },
    )
    assert v.validate_node(valid_bq) == []

    # 2. Indivisible dimension (65 % 32 != 0)
    indivisible_bq = LogicalNode(
        id="bq_bad_dim",
        op_type="block_quantize",
        domain="quantization",
        shape_metadata=(128, 65),
        attributes={
            "block_size": 32,
            "quant_dtype": "fp8_e4m3fn",
            "axis": -1,
        },
    )
    errs = v.validate_node(indivisible_bq)
    assert any("is not evenly divisible by block_size" in e.message for e in errs)

    # 3. Invalid block_size (non-positive or non-int)
    bad_bs_node = LogicalNode(
        id="bq_bad_bs",
        op_type="block_quantize",
        domain="quantization",
        shape_metadata=(128, 64),
        attributes={
            "block_size": 0,
            "quant_dtype": "fp8_e4m3fn",
        },
    )
    errs_bs = v.validate_node(bad_bs_node)
    assert any("block_size must be a positive integer" in e.message for e in errs_bs)


def test_dequantize_grouped_int4_validation() -> None:
    """Verify weight-only grouped INT4 dequantization, packing formats, and scale alignment."""
    v = Validator(level=ValidationLevel.STRICT)

    # 1. Valid dequantize_grouped_int4 across supported packing formats
    for fmt in ["marlin", "exllama", "tensorrt_llm", "awq", "gptq"]:
        node = LogicalNode(
            id=f"dequant_{fmt}",
            op_type="quantization.dequantize_grouped_int4",
            domain="quantization",
            shape_metadata=(2048, 1024),
            attributes={
                "group_size": 128,
                "packing_format": fmt,
                "symmetric": True,
                "axis": 0,
                "scales_shape": [16, 1024],  # 2048 // 128 == 16
            },
        )
        assert v.validate_node(node) == []

    # 2. Invalid packing format
    bad_fmt_node = LogicalNode(
        id="dequant_bad_fmt",
        op_type="dequantize_grouped_int4",
        domain="quantization",
        shape_metadata=(2048, 1024),
        attributes={
            "group_size": 128,
            "packing_format": "unknown_packer",
        },
    )
    errs = v.validate_node(bad_fmt_node)
    assert any("Invalid packing format 'unknown_packer'" in e.message for e in errs)

    # 3. Invalid group_size
    bad_gs_node = LogicalNode(
        id="dequant_bad_gs",
        op_type="dequantize_grouped_int4",
        domain="quantization",
        shape_metadata=(2048, 1024),
        attributes={
            "group_size": -32,
            "packing_format": "awq",
        },
    )
    errs_gs = v.validate_node(bad_gs_node)
    assert any("group_size must be a positive integer" in e.message for e in errs_gs)

    # 4. Weight matrix dimension not divisible by group_size
    indivisible_weight = LogicalNode(
        id="dequant_bad_weight",
        op_type="dequantize_grouped_int4",
        domain="quantization",
        shape_metadata=(2000, 1024),  # 2000 not divisible by 128
        attributes={
            "group_size": 128,
            "packing_format": "marlin",
            "axis": 0,
        },
    )
    errs_w = v.validate_node(indivisible_weight)
    assert any("is not evenly divisible by group_size 128" in e.message for e in errs_w)

    # 5. Scales shape mismatch (expected 16, got 15)
    scale_mismatch_node = LogicalNode(
        id="dequant_scale_mismatch",
        op_type="dequantize_grouped_int4",
        domain="quantization",
        shape_metadata=(2048, 1024),
        attributes={
            "group_size": 128,
            "packing_format": "marlin",
            "axis": 0,
            "scales_shape": [15, 1024],  # 15 != 16
        },
    )
    errs_scale = v.validate_node(scale_mismatch_node)
    assert any(
        "Scales shape dimension 0 (15) does not match expected groups 16" in e.message
        for e in errs_scale
    )


def test_quantization_registry_and_errors() -> None:
    """Verify operator schema registry lookup and missing attribute diagnostics."""
    v = Validator(level=ValidationLevel.STRICT)

    # 1. Unregistered operator in quantization domain
    bad_op = LogicalNode(
        id="bad_qop",
        op_type="unknown_quant_op",
        domain="quantization",
    )
    errs = v.validate_kind(bad_op)
    assert any("not found in quantization registry" in e.message for e in errs)

    # 2. Missing required attribute (block_size missing in block_quantize)
    missing_attr_node = LogicalNode(
        id="missing_attr_node",
        op_type="block_quantize",
        domain="quantization",
        shape_metadata=(32,),
        attributes={"quant_dtype": "fp8_e4m3fn"},
    )
    errs_attr = v.validate_required_attributes(missing_attr_node)
    assert any(
        "block_size" in e.message and "missing" in e.message.lower() for e in errs_attr
    )


def test_quantization_edge_cases() -> None:
    """Verify edge cases for block_quantize, dequantize_grouped_int4, and unknown quant ops."""
    v = Validator(level=ValidationLevel.WARNING)

    # 1. Unknown op in validate_quantization
    unk_node = LogicalNode(
        id="unk_q", op_type="unknown_quant_op", domain="quantization"
    )
    assert v.validate_quantization(unk_node) == []

    # 2. block_quantize without shape_metadata and with out-of-bounds axis
    bq_no_shape = LogicalNode(
        id="bq_no_shape",
        op_type="block_quantize",
        domain="quantization",
        attributes={"block_size": 32, "quant_dtype": "fp8_e4m3fn"},
    )
    assert v.validate_quantization(bq_no_shape) == []

    bq_oob_axis = LogicalNode(
        id="bq_oob",
        op_type="block_quantize",
        domain="quantization",
        shape_metadata=(32,),
        attributes={"block_size": 32, "quant_dtype": "fp8_e4m3fn", "axis": 99},
    )
    assert v.validate_quantization(bq_oob_axis) == []

    # 3. dequantize_grouped_int4 without shape_metadata and with out-of-bounds axis
    deq_no_shape = LogicalNode(
        id="deq_no_shape",
        op_type="dequantize_grouped_int4",
        domain="quantization",
        attributes={"group_size": 32, "packing_format": "marlin"},
    )
    assert v.validate_quantization(deq_no_shape) == []

    deq_oob_axis = LogicalNode(
        id="deq_oob",
        op_type="dequantize_grouped_int4",
        domain="quantization",
        shape_metadata=(32, 32),
        attributes={"group_size": 32, "packing_format": "marlin", "axis": 99},
    )
    assert v.validate_quantization(deq_oob_axis) == []

    deq_bad_scales_axis = LogicalNode(
        id="deq_sc_oob",
        op_type="dequantize_grouped_int4",
        domain="quantization",
        shape_metadata=(32, 32),
        attributes={
            "group_size": 32,
            "packing_format": "marlin",
            "axis": 0,
            "scales_shape": [],
        },
    )
    assert v.validate_quantization(deq_bad_scales_axis) == []
