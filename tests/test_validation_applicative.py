"""Unit tests for Phase 4 Semiring/Ring shape algebra & *_result APIs and Phase 5 Applicative Validation."""

from __future__ import annotations

from ml_switcheroo_ir.graph import LogicalGraph, LogicalNode
from ml_switcheroo_ir.shapes import (
    DIMENSION_EQ,
    DIMENSION_SEMIRING,
    POLYNOMIAL_EQ,
    POLYNOMIAL_HASH,
    POLYNOMIAL_RING,
    POLYNOMIAL_SEMIRING,
    SHAPE_TRANSFER_WITNESSES,
    Polynomial,
    ShapeTransferContext,
    SymbolicConstraintTracker,
    SymbolicParseError,
    SymVar,
    broadcast_dimension_result,
    broadcast_shapes_result,
    infer_op_output_shape_result,
    matmul_shape_result,
    normalize_axis_result,
    parse_symbolic_expr_result,
    tokenize_symbolic_expr_result,
)
from ml_switcheroo_ir.typeclasses import Err, Invalid, Ok, Valid
from ml_switcheroo_ir.validation import (
    COMMUNICATION_VOLUME_MONOID,
    GROUNDING_DIAGNOSTICS_MONOID,
    HARDWARE_SEMANTICS_BY_DOMAIN,
    PTX_ISA_SEMANTICS,
    QUANTIZATION_NODE_RULE,
    RDNA_ISA_SEMANTICS,
    SASS_ISA_SEMANTICS,
    VALIDATION_ERRORS_MONOID,
    VALIDATION_LEVEL_EQ,
    VALIDATION_LEVEL_LATTICE,
    VALIDATION_LEVEL_ORD,
    WGSL_ISA_SEMANTICS,
    CommunicationVolumeReport,
    DiagnosticSeverity,
    GroundingDiagnostic,
    ValidationError,
    ValidationLevel,
    ValidationRule,
    Validator,
    combine_rules,
    errors_from_validation,
    rule_from_errors_fn,
    when_predicate,
)


def test_polynomial_ring_and_dimension_semiring() -> None:
    """Test immutable Polynomial, POLYNOMIAL_RING, DIMENSION_SEMIRING, and total parser *_result APIs."""
    px = Polynomial.from_var("x")
    p3 = Polynomial.from_const(3)
    summed = POLYNOMIAL_SEMIRING.add(px, p3)

    import types

    assert len(summed.canonical_terms) == 2
    assert isinstance(summed.terms, types.MappingProxyType)
    assert POLYNOMIAL_EQ.eq(
        POLYNOMIAL_RING.semiring.add(summed, POLYNOMIAL_RING.neg(summed)), POLYNOMIAL_SEMIRING.zero()
    )
    assert POLYNOMIAL_HASH.hash_value(summed) == hash(summed)
    assert POLYNOMIAL_RING.sub(summed, p3) == px
    assert POLYNOMIAL_SEMIRING.mul(POLYNOMIAL_SEMIRING.one(), px) == px

    assert DIMENSION_SEMIRING.zero() == 0
    assert DIMENSION_SEMIRING.one() == 1
    assert DIMENSION_SEMIRING.add(4, 6) == 10
    assert DIMENSION_SEMIRING.mul(3, 5) == 15
    sym_sum = DIMENSION_SEMIRING.add(SymVar("B"), 2)
    sym_prod = DIMENSION_SEMIRING.mul(SymVar("B"), 4)
    assert DIMENSION_EQ.eq(sym_sum, "B + 2") is True
    assert DIMENSION_EQ.eq(sym_prod, "4 * B") is True

    tok_ok = tokenize_symbolic_expr_result("B + 1")
    tok_err = tokenize_symbolic_expr_result("B @ 1")
    assert isinstance(tok_ok, Ok) and tok_ok.value == ("B", "+", "1")
    assert isinstance(tok_err, Err) and isinstance(tok_err.error, SymbolicParseError)

    parse_ok = parse_symbolic_expr_result("2 * N + 1")
    parse_err = parse_symbolic_expr_result("2 * (")
    assert isinstance(parse_ok, Ok)
    assert isinstance(parse_err, Err) and isinstance(parse_err.error, SymbolicParseError)


def test_total_shape_inference_result_apis_and_witnesses() -> None:
    """Test normalize_axis_result, broadcast_*_result, matmul_shape_result, and ShapeTransferWitness."""
    assert normalize_axis_result(-1, 3) == Ok(2)
    assert normalize_axis_result([0, -1], 3) == Ok((0, 2))
    assert normalize_axis_result([0, 9], 3).is_err()
    assert normalize_axis_result(True, 3).is_err()

    assert broadcast_dimension_result(1, 8) == Ok(8)
    assert broadcast_dimension_result(3, 8).is_err()

    assert broadcast_shapes_result((2, 1), (1, 4)) == Ok((2, 4))
    assert broadcast_shapes_result((2, 3), (2, 5)).is_err()

    assert matmul_shape_result((2, 4), (4, 8)) == Ok((2, 8))
    assert matmul_shape_result((2, 4), (5, 8)).is_err()

    assert infer_op_output_shape_result("Relu", [(2, 4)]) == Ok((2, 4))
    assert infer_op_output_shape_result("Concat", [(2, 3), (2,)], {"axis": 0}).is_err()

    concat_witness = next(w for w in SHAPE_TRANSFER_WITNESSES if "Concat" in [str(o) for o in w.op_names])
    ctx_ok = ShapeTransferContext(
        node_id="c1",
        input_shapes=((2, 3), (2, 5)),
        attributes={"axis": 1},
        num_outputs=1,
        tracker=SymbolicConstraintTracker(),
    )
    ctx_bad = ShapeTransferContext(
        node_id="c2",
        input_shapes=((2, 3), (2,)),
        attributes={"axis": 0},
        num_outputs=1,
        tracker=SymbolicConstraintTracker(),
    )
    assert concat_witness.infer_result(ctx_ok) == Ok((2, 8))
    assert concat_witness.infer_result(ctx_bad).is_err()


def test_validation_monoids_lattice_and_rule_combinators() -> None:
    """Test ValidationError equality/hash, VALIDATION_LEVEL_LATTICE, monoids, and ValidationRule."""
    e1 = ValidationError(node_id="n1", attribute="a", message="m1", level=ValidationLevel.WARNING)
    e1_dup = ValidationError(node_id="n1", attribute="a", message="m1", level=ValidationLevel.WARNING)
    e2 = ValidationError(node_id="n2", attribute="b", message="m2", level=ValidationLevel.ERROR)

    assert e1 == e1_dup
    assert e1 != "not_an_error"
    assert hash(e1) == hash(e1_dup)
    assert VALIDATION_ERRORS_MONOID.mconcat([(e1,), (), (e2,)]) == (e1, e2)

    assert VALIDATION_LEVEL_ORD.lt(ValidationLevel.LENIENT, ValidationLevel.WARNING)
    assert VALIDATION_LEVEL_EQ.eq(ValidationLevel.STRICT, ValidationLevel.STRICT)
    assert VALIDATION_LEVEL_LATTICE.bottom() is ValidationLevel.LENIENT
    assert VALIDATION_LEVEL_LATTICE.top() is ValidationLevel.STRICT
    assert VALIDATION_LEVEL_LATTICE.join(ValidationLevel.WARNING, ValidationLevel.ERROR) is ValidationLevel.ERROR
    assert VALIDATION_LEVEL_LATTICE.meet(ValidationLevel.WARNING, ValidationLevel.ERROR) is ValidationLevel.WARNING

    r1 = CommunicationVolumeReport(total_volume_bytes=100, by_axis={"tp": 100}, by_op={"all_reduce": 100})
    r2 = CommunicationVolumeReport(total_volume_bytes=50, by_axis={"tp": 20, "dp": 30}, by_op={"all_gather": 50})
    merged = COMMUNICATION_VOLUME_MONOID.mconcat([r1, COMMUNICATION_VOLUME_MONOID.empty(), r2])
    assert merged.total_volume_bytes == 150
    assert merged.by_axis == {"tp": 120, "dp": 30}
    assert merged.by_op == {"all_reduce": 100, "all_gather": 50}

    gd = GroundingDiagnostic(field="op", message="ungrounded", severity=DiagnosticSeverity.ERROR)
    assert GROUNDING_DIAGNOSTICS_MONOID.mconcat([(gd,), (e2,)]) == (gd, e2)

    r_always: ValidationRule[LogicalNode] = rule_from_errors_fn(
        "r_err",
        lambda n: [ValidationError(n.id, "k", "bad")],
    )
    r_guarded = when_predicate(lambda n: n.op_type == "Trigger", r_always)
    combined = combine_rules((r_guarded, QUANTIZATION_NODE_RULE))

    relu = LogicalNode(id="r", op_type="Relu")
    trig = LogicalNode(id="t", op_type="Trigger")
    assert errors_from_validation(combined.run(relu)) == []
    assert len(errors_from_validation(combined.run(trig))) == 1


def test_hardware_dialect_semantics_and_applicative_graph_validator() -> None:
    """Test RDNA/SASS/WGSL/PTX HardwareDialectSemantics witnesses and Validator.validate_graph_applicative."""
    assert HARDWARE_SEMANTICS_BY_DOMAIN["rdna"] is RDNA_ISA_SEMANTICS
    assert HARDWARE_SEMANTICS_BY_DOMAIN["sass"] is SASS_ISA_SEMANTICS
    assert HARDWARE_SEMANTICS_BY_DOMAIN["wgsl"] is WGSL_ISA_SEMANTICS
    assert HARDWARE_SEMANTICS_BY_DOMAIN["ptx"] is PTX_ISA_SEMANTICS

    rdna_a = LogicalNode(
        id="rx",
        op_type="v_fmac_f32",
        domain="rdna",
        attributes={"dst_vgpr": 0, "src_vgprs": [1, 2], "vopd_slot": "X"},
    )
    rdna_b = LogicalNode(
        id="ry",
        op_type="v_fmac_f32",
        domain="rdna",
        attributes={"dst_vgpr": 0, "src_vgprs": [1, 2], "vopd_slot": "Y"},
    )
    rdna_no_dst = LogicalNode(id="rn", op_type="s_nop", domain="rdna")
    assert RDNA_ISA_SEMANTICS.extract_src_operands(rdna_a) == (1, 2)
    assert RDNA_ISA_SEMANTICS.extract_dst_operands(rdna_a) == (0,)
    assert RDNA_ISA_SEMANTICS.extract_dst_operands(rdna_no_dst) == ()
    assert isinstance(RDNA_ISA_SEMANTICS.validate_sequence((rdna_a, rdna_b), ValidationLevel.STRICT), Invalid)
    assert isinstance(
        RDNA_ISA_SEMANTICS.validate_sequence((rdna_no_dst, rdna_no_dst), ValidationLevel.WARNING),
        Valid,
    )

    sass_p = LogicalNode(
        id="sp",
        op_type="FFMA",
        domain="sass",
        attributes={"dst_reg": "R1", "src_regs": ["R2", "R3"], "stall_count": 1},
    )
    sass_c = LogicalNode(
        id="sc",
        op_type="FADD",
        domain="sass",
        attributes={"dst_reg": "R4", "src_regs": ["R1", "R2"], "stall_count": 1},
    )
    assert SASS_ISA_SEMANTICS.extract_src_operands(sass_p) == ("R2", "R3")
    assert SASS_ISA_SEMANTICS.extract_dst_operands(sass_p) == ("R1",)
    assert isinstance(SASS_ISA_SEMANTICS.validate_sequence((sass_p, sass_c), ValidationLevel.STRICT), Invalid)

    wgsl_n = LogicalNode(id="wn", op_type="workgroupBarrier", domain="wgsl", inputs=["x"], outputs=["y"])
    assert WGSL_ISA_SEMANTICS.extract_src_operands(wgsl_n) == ("x",)
    assert WGSL_ISA_SEMANTICS.extract_dst_operands(wgsl_n) == ("y",)
    assert isinstance(WGSL_ISA_SEMANTICS.validate_sequence((wgsl_n,), ValidationLevel.WARNING), Valid)

    ptx_n = LogicalNode(id="pn", op_type="fma.rn.f32", domain="ptx", inputs=["a", "b"], outputs=["d"])
    assert PTX_ISA_SEMANTICS.extract_src_operands(ptx_n) == ("a", "b")
    assert PTX_ISA_SEMANTICS.extract_dst_operands(ptx_n) == ("d",)
    assert isinstance(PTX_ISA_SEMANTICS.validate_instruction(ptx_n, ValidationLevel.STRICT), Valid)
    assert isinstance(PTX_ISA_SEMANTICS.validate_sequence((ptx_n,), ValidationLevel.STRICT), Valid)

    val = Validator()
    ok_g = LogicalGraph(
        nodes={"r": LogicalNode(id="r", op_type="Relu", inputs=["x"], outputs=["y"])},
        inputs=["x"],
        outputs=["y"],
    )
    bad_g = LogicalGraph(
        nodes={"c": LogicalNode(id="c", op_type="Concat", inputs=["x"], attributes={"axis": "bad"}, outputs=["y"])},
        inputs=["x"],
        outputs=["y"],
    )
    assert isinstance(val.validate_graph_applicative(ok_g), Valid)
    assert isinstance(val.validate_graph_applicative(bad_g), Invalid)
