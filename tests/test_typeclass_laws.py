"""Hypothesis property-based algebraic law verification suite (Phase 8.1 QuickCheck laws)."""

from __future__ import annotations

from collections.abc import Callable

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import ml_switcheroo_ir.compliance as compliance_mod
from ml_switcheroo_ir.distributed import PartitionSpec
from ml_switcheroo_ir.graph import LogicalGraph, LogicalNode
from ml_switcheroo_ir.identifiers import (
    NON_BOOL_INT_ORD,
    PORT_REF_EQ,
    PORT_REF_HASH,
    PORT_REF_ORD,
    NonBoolInt,
    NonNegativeInt,
    PortRef,
)
from ml_switcheroo_ir.serialization import (
    ATTRIBUTE_VALUE_JSON_CODEC,
    DTYPE_JSON_CODEC,
    LOGICAL_GRAPH_JSON_CODEC,
    LOGICAL_NODE_JSON_CODEC,
    PARTITION_SPEC_JSON_CODEC,
    SYM_EXPR_JSON_CODEC,
    TENSOR_SHAPE_JSON_CODEC,
    TENSOR_SPEC_JSON_CODEC,
)
from ml_switcheroo_ir.shapes import (
    DIMENSION_EQ,
    DIMENSION_SEMIRING,
    POLYNOMIAL_EQ,
    POLYNOMIAL_HASH,
    POLYNOMIAL_RING,
    SYM_EXPR_EQ,
    SYM_EXPR_HASH,
    DimensionType,
    Polynomial,
    SymBinaryOp,
    SymConst,
    SymExpr,
    SymUnaryOp,
    SymVar,
)
from ml_switcheroo_ir.transforms import ATTR_HASH_WITNESS, LOGICAL_GRAPH_EQ, NODE_CSE_WITNESS
from ml_switcheroo_ir.translation import ParameterTranslationEngine
from ml_switcheroo_ir.typeclasses import (
    BOOL_ALL_MONOID,
    BOOL_ANY_MONOID,
    INT_EQ,
    INT_HASH,
    INT_ORD,
    INT_PRODUCT_MONOID,
    INT_RING,
    INT_SEMIRING,
    INT_SUM_MONOID,
    STR_CONCAT_MONOID,
    STR_EQ,
    STR_ORD,
    Endo,
    Err,
    Invalid,
    Monoid,
    NonEmpty,
    Ok,
    Ordering,
    Result,
    Semigroup,
    Valid,
    Validation,
    and_then,
    ap_validation,
    dict_monoid,
    endo_monoid,
    nonempty_eq,
    nonempty_hash,
    nonempty_ord,
    nonempty_semigroup,
    product_monoid,
    traverse_validation,
    traverse_validation_ne,
    tuple_monoid,
)
from ml_switcheroo_ir.types import AttributeValue, DType, TensorShape, TensorSpec
from ml_switcheroo_ir.validation.core import (
    VALIDATION_ERRORS_MONOID,
    VALIDATION_ERRORS_SEMIGROUP,
    VALIDATION_LEVEL_LATTICE,
    VALIDATION_LEVEL_ORD,
    ValidationError,
    ValidationLevel,
)
from ml_switcheroo_ir.validation.distributed import (
    COMMUNICATION_VOLUME_MONOID,
    CommunicationVolumeReport,
)
from tests.conftest import seal_hypothesis_test

# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

_SMALL_INT = st.integers(min_value=-20, max_value=20)
_VAR_NAME = st.sampled_from(["B", "N", "M", "K", "T"])
_DTYPE_ST = st.sampled_from(list(DType))
_LEVEL_ST = st.sampled_from(list(ValidationLevel))


def _build_sym_expr_strategy() -> st.SearchStrategy[SymExpr]:
    """Construct recursive Hypothesis strategy generating closed ``SymExpr`` ASTs.

    Returns:
        st.SearchStrategy[SymExpr]: Bounded-depth ``SymExpr`` strategy.
    """
    leaf: st.SearchStrategy[SymExpr] = st.one_of(
        _SMALL_INT.map(SymConst),
        _VAR_NAME.map(SymVar),
    )
    return st.recursive(
        leaf,
        lambda children: st.one_of(
            st.builds(
                SymBinaryOp,
                st.sampled_from(["+", "-", "*"]),
                children,
                children,
            ),
            st.builds(
                SymUnaryOp,
                st.sampled_from(["-", "abs"]),
                children,
            ),
        ),
        max_leaves=5,
    )


_SYM_EXPR_ST: st.SearchStrategy[SymExpr] = _build_sym_expr_strategy()

_POLYNOMIAL_ST: st.SearchStrategy[Polynomial] = st.one_of(
    _SMALL_INT.map(Polynomial.from_const),
    _VAR_NAME.map(Polynomial.from_var),
    st.tuples(_SMALL_INT, _VAR_NAME, _SMALL_INT).map(
        lambda t: Polynomial.from_const(t[0]).add(Polynomial.from_var(t[1]).mul(Polynomial.from_const(t[2])))
    ),
)

_DIMENSION_ST: st.SearchStrategy[DimensionType] = st.one_of(
    st.integers(min_value=0, max_value=32),
    _VAR_NAME,
    _SYM_EXPR_ST,
)

_VALIDATION_ERROR_ST: st.SearchStrategy[ValidationError] = st.builds(
    ValidationError,
    node_id=st.sampled_from(["n0", "n1", "n2"]),
    attribute=st.sampled_from(["kind", "axis", "dtype"]),
    message=st.text(alphabet="abcdef ", min_size=1, max_size=12),
    level=_LEVEL_ST,
)

_COMM_REPORT_ST: st.SearchStrategy[CommunicationVolumeReport] = st.builds(
    CommunicationVolumeReport,
    total_volume_bytes=st.integers(min_value=0, max_value=4096),
    by_axis=st.dictionaries(
        st.sampled_from(["dp", "tp", "pp"]),
        st.integers(min_value=0, max_value=1024),
        max_size=3,
    ),
    by_op=st.dictionaries(
        st.sampled_from(["all_reduce", "all_gather"]),
        st.integers(min_value=0, max_value=1024),
        max_size=2,
    ),
)

_COMPLIANCE_RESULT_ST: st.SearchStrategy[compliance_mod.FileComplianceResult] = st.deferred(
    lambda: st.builds(
        compliance_mod.FileComplianceResult,
        is_ir=st.booleans(),
        is_fw=st.booleans(),
        implemented_ops=st.sets(st.sampled_from(["Add", "Mul", "Relu", "MatMul"]), max_size=3),
        ir_score=st.tuples(st.integers(min_value=0, max_value=5), st.just(5)),
        fw_score=st.tuples(st.integers(min_value=0, max_value=4), st.just(4)),
    )
)

_PORT_REF_ST: st.SearchStrategy[PortRef] = st.builds(
    PortRef.parse,
    st.one_of(
        st.sampled_from(["x", "y", "node_0", "attn_out"]),
        st.tuples(st.sampled_from(["n0", "n1", "qkv"]), st.integers(min_value=0, max_value=4)).map(
            lambda p: f"{p[0]}:{p[1]}"
        ),
    ),
)

_ATTR_VALUE_ST: st.SearchStrategy[AttributeValue] = st.one_of(
    st.none(),
    st.booleans(),
    _SMALL_INT,
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.text(alphabet="abcxyz01", max_size=8),
    st.binary(max_size=6),
    _DTYPE_ST,
    st.lists(_SMALL_INT, max_size=4),
    st.dictionaries(st.sampled_from(["k1", "k2"]), _SMALL_INT, max_size=2),
)

_TENSOR_SHAPE_ST: st.SearchStrategy[TensorShape] = st.lists(
    st.one_of(st.integers(min_value=1, max_value=16), _SYM_EXPR_ST),
    max_size=4,
).map(lambda ds: TensorShape(tuple(ds)))

_TENSOR_SPEC_ST: st.SearchStrategy[TensorSpec] = st.builds(
    TensorSpec,
    shape=st.lists(st.integers(min_value=1, max_value=16), max_size=3).map(tuple),
    dtype=_DTYPE_ST,
    sparsity=st.sampled_from([None, "2:4", "csr"]),
)

_PARTITION_SPEC_ST: st.SearchStrategy[PartitionSpec] = st.lists(
    st.one_of(
        st.none(),
        st.sampled_from(["dp", "tp", "cp"]),
        st.tuples(st.sampled_from(["dp", "tp"]), st.sampled_from(["cp", "ep"])),
    ),
    max_size=3,
).map(lambda axes: PartitionSpec(tuple(axes)))

_LOGICAL_NODE_ST: st.SearchStrategy[LogicalNode] = st.builds(
    LogicalNode,
    id=st.sampled_from(["n0", "n1", "n2"]),
    op_type=st.sampled_from(["Add", "Mul", "Relu", "MatMul"]),
    domain=st.sampled_from(["ai.onnx", "stablehlo"]),
    attributes=st.dictionaries(st.sampled_from(["alpha", "axis"]), _SMALL_INT, max_size=2),
    inputs=st.lists(st.sampled_from(["x", "y"]), max_size=2),
    outputs=st.lists(st.sampled_from(["z", "w"]), min_size=1, max_size=2),
    dtype=st.one_of(st.none(), _DTYPE_ST),
    sharding=st.one_of(st.none(), _PARTITION_SPEC_ST),
)


# ---------------------------------------------------------------------------
# 1. Eq & HashableWitness Coherence Laws
# ---------------------------------------------------------------------------


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(a=_POLYNOMIAL_ST, b=_POLYNOMIAL_ST, c=_POLYNOMIAL_ST)
def test_polynomial_eq_and_hash_coherence_laws(a: Polynomial, b: Polynomial, c: Polynomial) -> None:
    """Verify Eq laws (reflexivity, symmetry, transitivity) and HashableWitness coherence on Polynomial.

    Args:
        a (Polynomial): First polynomial sample.
        b (Polynomial): Second polynomial sample.
        c (Polynomial): Third polynomial sample.
    """
    assert POLYNOMIAL_EQ.eq(a, a)
    assert POLYNOMIAL_EQ.eq(a, b) == POLYNOMIAL_EQ.eq(b, a)
    if POLYNOMIAL_EQ.eq(a, b) and POLYNOMIAL_EQ.eq(b, c):
        assert POLYNOMIAL_EQ.eq(a, c)

    rebuilt_a = Polynomial(dict(reversed(list(a.terms.items()))))
    assert POLYNOMIAL_EQ.eq(a, rebuilt_a)
    assert POLYNOMIAL_HASH.hash_value(a) == POLYNOMIAL_HASH.hash_value(rebuilt_a)
    if POLYNOMIAL_EQ.eq(a, b):
        assert POLYNOMIAL_HASH.hash_value(a) == POLYNOMIAL_HASH.hash_value(b)


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(e1=_SYM_EXPR_ST, e2=_SYM_EXPR_ST, e3=_SYM_EXPR_ST, p1=_PORT_REF_ST, p2=_PORT_REF_ST)
def test_sym_expr_and_port_ref_eq_and_hash_laws(
    e1: SymExpr,
    e2: SymExpr,
    e3: SymExpr,
    p1: PortRef,
    p2: PortRef,
) -> None:
    """Verify Eq and HashableWitness coherence laws on SymExpr and PortRef.

    Args:
        e1 (SymExpr): First symbolic expression.
        e2 (SymExpr): Second symbolic expression.
        e3 (SymExpr): Third symbolic expression.
        p1 (PortRef): First port reference.
        p2 (PortRef): Second port reference.
    """
    assert SYM_EXPR_EQ.eq(e1, e1)
    assert SYM_EXPR_EQ.eq(e1, e2) == SYM_EXPR_EQ.eq(e2, e1)
    if SYM_EXPR_EQ.eq(e1, e2) and SYM_EXPR_EQ.eq(e2, e3):
        assert SYM_EXPR_EQ.eq(e1, e3)
    if SYM_EXPR_EQ.eq(e1, e2):
        assert SYM_EXPR_HASH.hash_value(e1) == SYM_EXPR_HASH.hash_value(e2)

    assert PORT_REF_EQ.eq(p1, p1)
    assert PORT_REF_EQ.eq(p1, p2) == PORT_REF_EQ.eq(p2, p1)
    p1_roundtrip = PortRef.parse(p1.render())
    assert PORT_REF_EQ.eq(p1, p1_roundtrip)
    assert PORT_REF_HASH.hash_value(p1) == PORT_REF_HASH.hash_value(p1_roundtrip)


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(attr1=_ATTR_VALUE_ST, attr2=_ATTR_VALUE_ST, n1=_LOGICAL_NODE_ST, n2=_LOGICAL_NODE_ST)
def test_attr_and_node_cse_hash_coherence_laws(
    attr1: AttributeValue,
    attr2: AttributeValue,
    n1: LogicalNode,
    n2: LogicalNode,
) -> None:
    """Verify HashableWitness coherence on AttributeValue and LogicalNode (NODE_CSE_WITNESS).

    Args:
        attr1 (AttributeValue): First attribute value.
        attr2 (AttributeValue): Second attribute value.
        n1 (LogicalNode): First IR node.
        n2 (LogicalNode): Second IR node.
    """
    assert ATTR_HASH_WITNESS.eq_witness.eq(attr1, attr1)
    assert ATTR_HASH_WITNESS.eq_witness.eq(attr1, attr2) == ATTR_HASH_WITNESS.eq_witness.eq(attr2, attr1)
    if ATTR_HASH_WITNESS.eq_witness.eq(attr1, attr2):
        assert ATTR_HASH_WITNESS.hash_value(attr1) == ATTR_HASH_WITNESS.hash_value(attr2)

    n1_renamed = n1.clone(id="renamed_cse_twin", outputs=["alt_out"])
    assert NODE_CSE_WITNESS.eq_witness.eq(n1, n1_renamed)
    assert NODE_CSE_WITNESS.hash_value(n1) == NODE_CSE_WITNESS.hash_value(n1_renamed)
    if NODE_CSE_WITNESS.eq_witness.eq(n1, n2):
        assert NODE_CSE_WITNESS.hash_value(n1) == NODE_CSE_WITNESS.hash_value(n2)

    assert INT_EQ.eq(5, 5) and INT_HASH.hash_value(5) == INT_HASH.hash_value(5)
    assert STR_EQ.eq("ir", "ir")


# ---------------------------------------------------------------------------
# 2. Ord & BoundedLattice Laws
# ---------------------------------------------------------------------------


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    x=_SMALL_INT,
    y=_SMALL_INT,
    z=_SMALL_INT,
    s1=st.text(max_size=6),
    s2=st.text(max_size=6),
    p1=_PORT_REF_ST,
    p2=_PORT_REF_ST,
    p3=_PORT_REF_ST,
)
def test_ord_laws_totality_antisymmetry_transitivity_and_flip(
    x: int,
    y: int,
    z: int,
    s1: str,
    s2: str,
    p1: PortRef,
    p2: PortRef,
    p3: PortRef,
) -> None:
    """Verify Ord totality, antisymmetry, transitivity, and Ordering.flip involution.

    Args:
        x (int): First integer.
        y (int): Second integer.
        z (int): Third integer.
        s1 (str): First string.
        s2 (str): Second string.
        p1 (PortRef): First port reference.
        p2 (PortRef): Second port reference.
        p3 (PortRef): Third port reference.
    """
    cmp_xy = INT_ORD.compare(x, y)
    assert cmp_xy.flip().flip() == cmp_xy
    assert INT_ORD.compare(y, x) == cmp_xy.flip()
    assert INT_ORD.le(x, y) or INT_ORD.le(y, x)
    if INT_ORD.le(x, y) and INT_ORD.le(y, x):
        assert INT_ORD.eq_witness.eq(x, y)
    if INT_ORD.le(x, y) and INT_ORD.le(y, z):
        assert INT_ORD.le(x, z)

    assert NON_BOOL_INT_ORD.compare(NonBoolInt(x), NonBoolInt(y)) == cmp_xy
    assert STR_ORD.compare(s2, s1) == STR_ORD.compare(s1, s2).flip()

    assert PORT_REF_ORD.le(p1, p2) or PORT_REF_ORD.le(p2, p1)
    if PORT_REF_ORD.le(p1, p2) and PORT_REF_ORD.le(p2, p3):
        assert PORT_REF_ORD.le(p1, p3)
    assert Ordering.EQ.flip() == Ordering.EQ


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(a=_LEVEL_ST, b=_LEVEL_ST, c=_LEVEL_ST)
def test_bounded_lattice_laws_on_validation_level(
    a: ValidationLevel,
    b: ValidationLevel,
    c: ValidationLevel,
) -> None:
    """Verify BoundedLattice idempotence, commutativity, associativity, absorption, and bounds.

    Args:
        a (ValidationLevel): First validation level.
        b (ValidationLevel): Second validation level.
        c (ValidationLevel): Third validation level.
    """
    lat = VALIDATION_LEVEL_LATTICE
    assert lat.join(a, a) == a
    assert lat.meet(a, a) == a
    assert lat.join(a, b) == lat.join(b, a)
    assert lat.meet(a, b) == lat.meet(b, a)
    assert lat.join(lat.join(a, b), c) == lat.join(a, lat.join(b, c))
    assert lat.meet(lat.meet(a, b), c) == lat.meet(a, lat.meet(b, c))
    assert lat.join(a, lat.meet(a, b)) == a
    assert lat.meet(a, lat.join(a, b)) == a
    assert lat.join(lat.bottom(), a) == a
    assert lat.meet(lat.top(), a) == a
    assert lat.leq(a, b) == VALIDATION_LEVEL_ORD.le(a, b)


# ---------------------------------------------------------------------------
# 3. Semigroup & Monoid Laws
# ---------------------------------------------------------------------------


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    i1=_SMALL_INT,
    i2=_SMALL_INT,
    i3=_SMALL_INT,
    e1=st.lists(_VALIDATION_ERROR_ST, max_size=2).map(tuple),
    e2=st.lists(_VALIDATION_ERROR_ST, max_size=2).map(tuple),
    e3=st.lists(_VALIDATION_ERROR_ST, max_size=2).map(tuple),
)
def test_scalar_and_validation_monoid_laws(
    i1: int,
    i2: int,
    i3: int,
    e1: tuple[ValidationError, ...],
    e2: tuple[ValidationError, ...],
    e3: tuple[ValidationError, ...],
) -> None:
    """Verify associativity and left/right identity on INT and VALIDATION_ERRORS Monoids.

    Args:
        i1 (int): First integer.
        i2 (int): Second integer.
        i3 (int): Third integer.
        e1 (tuple[ValidationError, ...]): First error tuple.
        e2 (tuple[ValidationError, ...]): Second error tuple.
        e3 (tuple[ValidationError, ...]): Third error tuple.
    """
    for mon in (INT_SUM_MONOID, INT_PRODUCT_MONOID):
        assert mon.combine(mon.combine(i1, i2), i3) == mon.combine(i1, mon.combine(i2, i3))
        assert mon.combine(mon.empty(), i1) == i1
        assert mon.combine(i1, mon.empty()) == i1

    assert VALIDATION_ERRORS_MONOID.combine(
        VALIDATION_ERRORS_MONOID.combine(e1, e2), e3
    ) == VALIDATION_ERRORS_MONOID.combine(e1, VALIDATION_ERRORS_MONOID.combine(e2, e3))
    assert VALIDATION_ERRORS_MONOID.combine(VALIDATION_ERRORS_MONOID.empty(), e1) == e1
    assert VALIDATION_ERRORS_MONOID.combine(e1, VALIDATION_ERRORS_MONOID.empty()) == e1


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    v1=_COMM_REPORT_ST,
    v2=_COMM_REPORT_ST,
    v3=_COMM_REPORT_ST,
    f1=_COMPLIANCE_RESULT_ST,
    f2=_COMPLIANCE_RESULT_ST,
    f3=_COMPLIANCE_RESULT_ST,
)
def test_communication_and_compliance_monoid_laws(
    v1: CommunicationVolumeReport,
    v2: CommunicationVolumeReport,
    v3: CommunicationVolumeReport,
    f1: compliance_mod.FileComplianceResult,
    f2: compliance_mod.FileComplianceResult,
    f3: compliance_mod.FileComplianceResult,
) -> None:
    """Verify associativity and left/right identity on COMMUNICATION_VOLUME and FILE_COMPLIANCE Monoids.

    Args:
        v1 (CommunicationVolumeReport): First communication volume report.
        v2 (CommunicationVolumeReport): Second communication volume report.
        v3 (CommunicationVolumeReport): Third communication volume report.
        f1 (compliance_mod.FileComplianceResult): First compliance result.
        f2 (compliance_mod.FileComplianceResult): Second compliance result.
        f3 (compliance_mod.FileComplianceResult): Third compliance result.
    """
    assert COMMUNICATION_VOLUME_MONOID.combine(
        COMMUNICATION_VOLUME_MONOID.combine(v1, v2), v3
    ) == COMMUNICATION_VOLUME_MONOID.combine(v1, COMMUNICATION_VOLUME_MONOID.combine(v2, v3))
    assert COMMUNICATION_VOLUME_MONOID.combine(COMMUNICATION_VOLUME_MONOID.empty(), v1) == v1
    assert COMMUNICATION_VOLUME_MONOID.combine(v1, COMMUNICATION_VOLUME_MONOID.empty()) == v1

    fc_monoid = compliance_mod.FILE_COMPLIANCE_MONOID
    assert fc_monoid.combine(fc_monoid.combine(f1, f2), f3) == fc_monoid.combine(f1, fc_monoid.combine(f2, f3))
    assert fc_monoid.combine(fc_monoid.empty(), f1) == f1
    assert fc_monoid.combine(f1, fc_monoid.empty()) == f1


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    d1=st.dictionaries(st.sampled_from(["a", "b"]), _SMALL_INT, max_size=2),
    d2=st.dictionaries(st.sampled_from(["a", "b"]), _SMALL_INT, max_size=2),
    d3=st.dictionaries(st.sampled_from(["a", "b"]), _SMALL_INT, max_size=2),
    x=_SMALL_INT,
)
def test_dict_product_and_endo_monoid_laws(
    d1: dict[str, int],
    d2: dict[str, int],
    d3: dict[str, int],
    x: int,
) -> None:
    """Verify Monoid laws on dict_monoid, product_monoid, BOOL monoids, STR monoid, and endo_monoid.

    Args:
        d1 (dict[str, int]): First dictionary.
        d2 (dict[str, int]): Second dictionary.
        d3 (dict[str, int]): Third dictionary.
        x (int): Input integer for endomorphism evaluation.
    """
    dm: Monoid[dict[str, int]] = dict_monoid(INT_SUM_MONOID.semigroup)
    assert dm.combine(dm.combine(d1, d2), d3) == dm.combine(d1, dm.combine(d2, d3))
    assert dm.combine(dm.empty(), d1) == d1
    assert dm.combine(d1, dm.empty()) == d1

    pm = product_monoid(INT_SUM_MONOID, STR_CONCAT_MONOID)
    p_a, p_b, p_c = (x, "a"), (2, "b"), (-1, "c")
    assert pm.combine(pm.combine(p_a, p_b), p_c) == pm.combine(p_a, pm.combine(p_b, p_c))
    assert pm.combine(pm.empty(), p_a) == p_a
    assert pm.combine(p_a, pm.empty()) == p_a

    assert BOOL_ALL_MONOID.combine(BOOL_ALL_MONOID.empty(), False) is False
    assert BOOL_ANY_MONOID.combine(BOOL_ANY_MONOID.empty(), True) is True

    em: Monoid[Endo[int]] = endo_monoid()
    f: Endo[int] = Endo(lambda n: n + 3)
    g: Endo[int] = Endo(lambda n: n * 2)
    h: Endo[int] = Endo(lambda n: n - 5)
    lhs = em.combine(em.combine(f, g), h)
    rhs = em.combine(f, em.combine(g, h))
    assert lhs.app(x) == rhs.app(x)
    assert em.combine(em.empty(), f).app(x) == f.app(x)
    assert em.combine(f, em.empty()).app(x) == f.app(x)


# ---------------------------------------------------------------------------
# 4. Semiring & Ring Laws
# ---------------------------------------------------------------------------


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    i1=_SMALL_INT,
    i2=_SMALL_INT,
    i3=_SMALL_INT,
    p1=_POLYNOMIAL_ST,
    p2=_POLYNOMIAL_ST,
    p3=_POLYNOMIAL_ST,
)
def test_int_and_polynomial_ring_laws(
    i1: int,
    i2: int,
    i3: int,
    p1: Polynomial,
    p2: Polynomial,
    p3: Polynomial,
) -> None:
    """Verify Semiring and Ring laws on INT_RING and POLYNOMIAL_RING.

    Args:
        i1 (int): First integer.
        i2 (int): Second integer.
        i3 (int): Third integer.
        p1 (Polynomial): First polynomial.
        p2 (Polynomial): Second polynomial.
        p3 (Polynomial): Third polynomial.
    """
    sr_i = INT_SEMIRING
    assert sr_i.add(i1, i2) == sr_i.add(i2, i1)
    assert sr_i.add(sr_i.add(i1, i2), i3) == sr_i.add(i1, sr_i.add(i2, i3))
    assert sr_i.add(sr_i.zero(), i1) == i1
    assert sr_i.mul(sr_i.mul(i1, i2), i3) == sr_i.mul(i1, sr_i.mul(i2, i3))
    assert sr_i.mul(sr_i.one(), i1) == i1 and sr_i.mul(i1, sr_i.one()) == i1
    assert sr_i.mul(i1, sr_i.add(i2, i3)) == sr_i.add(sr_i.mul(i1, i2), sr_i.mul(i1, i3))
    assert sr_i.mul(sr_i.zero(), i1) == sr_i.zero()
    assert sr_i.add(i1, INT_RING.neg(i1)) == sr_i.zero()
    assert INT_RING.sub(i1, i2) == sr_i.add(i1, INT_RING.neg(i2))

    sr_p = POLYNOMIAL_RING.semiring
    assert sr_p.add(p1, p2) == sr_p.add(p2, p1)
    assert sr_p.add(sr_p.add(p1, p2), p3) == sr_p.add(p1, sr_p.add(p2, p3))
    assert sr_p.add(sr_p.zero(), p1) == p1
    assert sr_p.mul(sr_p.mul(p1, p2), p3) == sr_p.mul(p1, sr_p.mul(p2, p3))
    assert sr_p.mul(sr_p.one(), p1) == p1 and sr_p.mul(p1, sr_p.one()) == p1
    assert sr_p.mul(p1, sr_p.add(p2, p3)) == sr_p.add(sr_p.mul(p1, p2), sr_p.mul(p1, p3))
    assert sr_p.mul(sr_p.add(p1, p2), p3) == sr_p.add(sr_p.mul(p1, p3), sr_p.mul(p2, p3))
    assert sr_p.mul(sr_p.zero(), p1) == sr_p.zero()
    assert sr_p.add(p1, POLYNOMIAL_RING.neg(p1)) == sr_p.zero()


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(d1=_DIMENSION_ST, d2=_DIMENSION_ST, d3=_DIMENSION_ST)
def test_dimension_semiring_laws(
    d1: DimensionType,
    d2: DimensionType,
    d3: DimensionType,
) -> None:
    """Verify Semiring laws (commutativity, associativity, identity, distributivity) on DIMENSION_SEMIRING.

    Args:
        d1 (DimensionType): First dimension.
        d2 (DimensionType): Second dimension.
        d3 (DimensionType): Third dimension.
    """
    sr_d = DIMENSION_SEMIRING
    assert DIMENSION_EQ.eq(sr_d.add(d1, d2), sr_d.add(d2, d1))
    assert DIMENSION_EQ.eq(sr_d.add(sr_d.add(d1, d2), d3), sr_d.add(d1, sr_d.add(d2, d3)))
    assert DIMENSION_EQ.eq(sr_d.add(sr_d.zero(), d1), d1)
    assert DIMENSION_EQ.eq(sr_d.mul(sr_d.mul(d1, d2), d3), sr_d.mul(d1, sr_d.mul(d2, d3)))
    assert DIMENSION_EQ.eq(sr_d.mul(sr_d.one(), d1), d1)
    assert DIMENSION_EQ.eq(sr_d.mul(d1, sr_d.add(d2, d3)), sr_d.add(sr_d.mul(d1, d2), sr_d.mul(d1, d3)))
    assert DIMENSION_EQ.eq(sr_d.mul(sr_d.add(d1, d2), d3), sr_d.add(sr_d.mul(d1, d3), sr_d.mul(d2, d3)))
    assert DIMENSION_EQ.eq(sr_d.mul(sr_d.zero(), d1), sr_d.zero())


def test_dimension_semiring_mixed_pow_and_abs_regressions() -> None:
    """Verify deterministic Semiring distributivity and associativity regressions with ``T**2`` and ``abs(B)``."""
    sr_d = DIMENSION_SEMIRING
    abs_b = SymUnaryOp("abs", SymVar("B"))
    assert DIMENSION_EQ.eq(
        sr_d.mul("T", sr_d.add(abs_b, "T")),
        sr_d.add(sr_d.mul("T", abs_b), sr_d.mul("T", "T")),
    )
    assert DIMENSION_EQ.eq(
        sr_d.mul(sr_d.mul("T", "T"), abs_b),
        sr_d.mul("T", sr_d.mul("T", abs_b)),
    )


# ---------------------------------------------------------------------------
# 5. Result Monad & Applicative Validation Error-Accumulation Laws
# ---------------------------------------------------------------------------


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    x=_SMALL_INT,
    res=st.one_of(_SMALL_INT.map(Ok), st.text(min_size=1, max_size=6).map(Err)),
)
def test_result_monad_laws(x: int, res: Result[str, int]) -> None:
    """Verify left identity, right identity, and associativity of Result Kleisli bind (and_then).

    Args:
        x (int): Seed integer for left-identity test.
        res (Result[str, int]): Arbitrary Ok/Err sample.
    """

    def f(n: int) -> Result[str, int]:
        """Kleisli arrow f.

        Args:
            n (int): Input integer.

        Returns:
            Result[str, int]: Doubled positive or Err.
        """
        return Ok(n * 2) if n >= -5 else Err("too_neg")

    def g(n: int) -> Result[str, str]:
        """Kleisli arrow g.

        Args:
            n (int): Input integer.

        Returns:
            Result[str, str]: Formatted string or Err.
        """
        return Err("zero_forbidden") if n == 0 else Ok(f"val:{n}")

    assert and_then(Ok(x), f) == f(x)
    assert and_then(res, Ok) == res
    assert and_then(and_then(res, f), g) == and_then(res, lambda a: and_then(f(a), g))


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    items=st.lists(
        st.one_of(
            _SMALL_INT.map(Valid),
            st.lists(_VALIDATION_ERROR_ST, min_size=1, max_size=2).map(lambda es: Invalid(tuple(es))),
        ),
        max_size=5,
    )
)
def test_applicative_validation_error_accumulation_law(
    items: list[Validation[tuple[ValidationError, ...], int]],
) -> None:
    """Verify Applicative Validation accumulates every Invalid branch in exact Semigroup order.

    Args:
        items (list[Validation[tuple[ValidationError, ...], int]]): Mixed Valid/Invalid steps.
    """
    sg = VALIDATION_ERRORS_SEMIGROUP
    traversed = traverse_validation(sg, items, lambda v: v)
    expected_errors = tuple(err for item in items if isinstance(item, Invalid) for err in item.errors)
    if expected_errors:
        assert traversed == Invalid(expected_errors)
    else:
        expected_vals = tuple(item.value for item in items if isinstance(item, Valid))
        assert traversed == Valid(expected_vals)

    str_men: Monoid[tuple[str, ...]] = tuple_monoid()
    str_sg: Semigroup[tuple[str, ...]] = str_men.semigroup
    fn_inv: Validation[tuple[str, ...], Callable[[int], int]] = Invalid(("e_fn",))
    arg_inv: Validation[tuple[str, ...], int] = Invalid(("e_arg",))
    both_inv = ap_validation(str_sg, fn_inv, arg_inv)
    assert both_inv == Invalid(("e_fn", "e_arg"))


# ---------------------------------------------------------------------------
# 6. JsonCodec & PartialIso Round-Trip Laws
# ---------------------------------------------------------------------------


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    dt=_DTYPE_ST,
    expr=_SYM_EXPR_ST,
    shape=_TENSOR_SHAPE_ST,
    spec=_TENSOR_SPEC_ST,
    pspec=_PARTITION_SPEC_ST,
    node=_LOGICAL_NODE_ST,
)
def test_json_codec_roundtrip_laws(
    dt: DType,
    expr: SymExpr,
    shape: TensorShape,
    spec: TensorSpec,
    pspec: PartitionSpec,
    node: LogicalNode,
) -> None:
    """Verify decode(encode(x)) == Ok(x) across DType, SymExpr, TensorShape, TensorSpec, PartitionSpec, Node, Graph.

    Args:
        dt (DType): Sampled DType.
        expr (SymExpr): Sampled closed SymExpr.
        shape (TensorShape): Sampled TensorShape.
        spec (TensorSpec): Sampled TensorSpec.
        pspec (PartitionSpec): Sampled PartitionSpec.
        node (LogicalNode): Sampled LogicalNode.
    """
    assert DTYPE_JSON_CODEC.decode(DTYPE_JSON_CODEC.encode(dt)) == Ok(dt)
    assert SYM_EXPR_JSON_CODEC.decode(SYM_EXPR_JSON_CODEC.encode(expr)) == Ok(expr)
    assert TENSOR_SHAPE_JSON_CODEC.decode(TENSOR_SHAPE_JSON_CODEC.encode(shape)) == Ok(shape)
    assert TENSOR_SPEC_JSON_CODEC.decode(TENSOR_SPEC_JSON_CODEC.encode(spec)) == Ok(spec)
    assert PARTITION_SPEC_JSON_CODEC.decode(PARTITION_SPEC_JSON_CODEC.encode(pspec)) == Ok(pspec)
    assert LOGICAL_NODE_JSON_CODEC.decode(LOGICAL_NODE_JSON_CODEC.encode(node)) == Ok(node)
    assert ATTRIBUTE_VALUE_JSON_CODEC.decode(ATTRIBUTE_VALUE_JSON_CODEC.encode(spec)) == Ok(spec)

    graph = LogicalGraph(
        name="LawGraph",
        nodes={node.id: node},
        inputs=list(node.inputs),
        input_specs={"x": spec} if "x" in node.inputs else {},
        outputs=list(node.outputs),
    )
    decoded_graph = LOGICAL_GRAPH_JSON_CODEC.decode(LOGICAL_GRAPH_JSON_CODEC.encode(graph))
    assert isinstance(decoded_graph, Ok)
    assert LOGICAL_GRAPH_EQ.eq(decoded_graph.value, graph)


@seal_hypothesis_test
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    vals=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=16, max_size=16
    ),
    src_dst=st.sampled_from([("OIHW", "HWIO"), ("HWIO", "OIHW"), ("NCHW", "NHWC"), ("NHWC", "NCHW")]),
)
def test_partial_iso_roundtrip_laws_conv_and_qkv(
    vals: list[float],
    src_dst: tuple[str, str],
) -> None:
    """Verify PartialIso round-trip laws (backward(forward(w)) == Ok(w)) on conv layouts and QKV split/merge.

    Args:
        vals (list[float]): 16 concrete float weights for 2x2x2x2 kernel and 6x2 QKV matrix.
        src_dst (tuple[str, str]): Source and target 4D layout tags.
    """
    eng = ParameterTranslationEngine()
    kernel_4d = [
        [[[vals[0], vals[1]], [vals[2], vals[3]]], [[vals[4], vals[5]], [vals[6], vals[7]]]],
        [[[vals[8], vals[9]], [vals[10], vals[11]]], [[vals[12], vals[13]], [vals[14], vals[15]]]],
    ]
    conv_iso = eng.conv_layout_iso(src_dst[0], src_dst[1])
    fwd = conv_iso.forward(kernel_4d)
    assert isinstance(fwd, Ok)
    assert conv_iso.backward(fwd.value) == Ok(kernel_4d)

    qkv_matrix = [vals[i : i + 2] for i in range(0, 12, 2)]
    qkv_iso = eng.qkv_split_merge_iso(num_heads=1, num_kv_heads=1, head_dim=2, dim=0)
    split = qkv_iso.forward(qkv_matrix)
    assert isinstance(split, Ok)
    assert qkv_iso.backward(split.value) == Ok(qkv_matrix)
    assert NonNegativeInt(NonBoolInt(0)) == 0


def test_generalized_dimension_equivalence_non_polynomial_atoms() -> None:
    """Verify SymUnaryOp.to_polynomial and generalized polynomial equivalence over non-ring atoms."""
    from ml_switcheroo_ir.shapes.ast import SymPiecewise
    from ml_switcheroo_ir.shapes.parser import SymbolicSolver

    var_b = SymVar("B")
    var_n = SymVar("N")
    assert SymUnaryOp("-", var_b).to_polynomial() == Polynomial.from_var("B").neg()
    assert SymUnaryOp("+", var_b).to_polynomial() == Polynomial.from_var("B")
    assert SymUnaryOp("abs", var_b).to_polynomial() is None
    assert SymUnaryOp("abs", SymBinaryOp("//", var_b, SymConst(2))).to_polynomial() is None
    assert DIMENSION_EQ.eq(SymUnaryOp("-", var_b), SymBinaryOp("-", SymConst(0), var_b))

    assert SymbolicSolver._to_generalized_polynomial(SymUnaryOp("+", var_b)) == Polynomial.from_var("B")
    assert DIMENSION_EQ.eq(
        SymUnaryOp("-", SymUnaryOp("abs", var_b)),
        SymBinaryOp("-", SymConst(0), SymUnaryOp("abs", var_b)),
    )

    max_ab = SymBinaryOp("+", SymConst(1), SymBinaryOp("max", var_b, var_n))
    max_ba = SymBinaryOp("+", SymBinaryOp("max", var_n, var_b), SymConst(1))
    assert DIMENSION_EQ.eq(max_ab, max_ba)

    fdiv_1 = SymBinaryOp("+", SymConst(2), SymBinaryOp("//", var_b, SymConst(4)))
    fdiv_2 = SymBinaryOp("+", SymBinaryOp("//", var_b, SymConst(4)), SymConst(2))
    assert DIMENSION_EQ.eq(fdiv_1, fdiv_2)

    pw = SymPiecewise(cases=[("B > 0", var_b)], default=SymConst(1))
    assert DIMENSION_EQ.eq(SymBinaryOp("+", SymConst(3), pw), SymBinaryOp("+", pw, SymConst(3)))
    assert DIMENSION_EQ.eq(SymBinaryOp("-", pw, SymConst(1)), SymBinaryOp("+", SymConst(-1), pw))


_NONEMPTY_INT_ST: st.SearchStrategy[NonEmpty[int]] = st.builds(
    NonEmpty,
    head=_SMALL_INT,
    tail=st.lists(_SMALL_INT, max_size=3).map(tuple),
)


@seal_hypothesis_test
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    a=_NONEMPTY_INT_ST,
    b=_NONEMPTY_INT_ST,
    c=_NONEMPTY_INT_ST,
    steps=st.lists(
        st.one_of(
            _SMALL_INT.map(Valid),
            st.builds(
                NonEmpty, head=_VALIDATION_ERROR_ST, tail=st.lists(_VALIDATION_ERROR_ST, max_size=2).map(tuple)
            ).map(Invalid),
        ),
        max_size=5,
    ),
)
def test_nonempty_algebraic_and_validation_laws(
    a: NonEmpty[int],
    b: NonEmpty[int],
    c: NonEmpty[int],
    steps: list[Validation[NonEmpty[ValidationError], int]],
) -> None:
    """Verify NonEmpty Semigroup associativity, Eq/Hash coherence, Ord total order, and Validation accumulation.

    Args:
        a (NonEmpty[int]): First non-empty sequence.
        b (NonEmpty[int]): Second non-empty sequence.
        c (NonEmpty[int]): Third non-empty sequence.
        steps (list[Validation[NonEmpty[ValidationError], int]]): Mixed Valid/Invalid NonEmpty steps.
    """
    sg: Semigroup[NonEmpty[int]] = nonempty_semigroup()
    assert sg.combine(sg.combine(a, b), c) == sg.combine(a, sg.combine(b, c))

    ne_eq = nonempty_eq(INT_EQ)
    ne_ord = nonempty_ord(INT_ORD)
    ne_hash = nonempty_hash(INT_HASH)

    assert ne_eq.eq(a, a)
    assert ne_eq.eq(a, b) == ne_eq.eq(b, a)
    if ne_eq.eq(a, b) and ne_eq.eq(b, c):
        assert ne_eq.eq(a, c)
    if ne_eq.eq(a, b):
        assert ne_hash.hash_value(a) == ne_hash.hash_value(b)

    assert ne_ord.compare(b, a) == ne_ord.compare(a, b).flip()
    assert ne_ord.le(a, b) or ne_ord.le(b, a)
    if ne_ord.le(a, b) and ne_ord.le(b, a):
        assert ne_eq.eq(a, b)
    if ne_ord.le(a, b) and ne_ord.le(b, c):
        assert ne_ord.le(a, c)

    traversed = traverse_validation_ne(steps, lambda v: v)
    expected_errs = [err for step in steps if isinstance(step, Invalid) for err in step.errors]
    if expected_errs:
        assert traversed == Invalid(NonEmpty(head=expected_errs[0], tail=tuple(expected_errs[1:])))
    else:
        assert traversed == Valid(tuple(step.value for step in steps if isinstance(step, Valid)))
