"""Unit tests for Phase 2 closed ADTs (SymExpr, AttrTerm) and assert_never exhaustiveness helpers."""

from __future__ import annotations

from ml_switcheroo_ir.distributed import PartitionSpec
from ml_switcheroo_ir.graph import LogicalGraph, LogicalNode
from ml_switcheroo_ir.schema.base import AttributeType, describe_attribute_type
from ml_switcheroo_ir.shapes import (
    SYM_EXPR_EQ,
    SYM_EXPR_HASH,
    SYM_EXPR_REWRITE,
    Polynomial,
    SymBinaryOp,
    SymConst,
    SymExpr,
    SymInt,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
    is_binary_sym_op,
    is_comparison_op,
    is_sym_expr,
    is_unary_sym_op,
    poly_from_sym_expr,
    substitute_sym_expr,
    sym_expr_eval,
    sym_expr_format,
    sym_expr_free_vars,
    sym_expr_simplify,
)
from ml_switcheroo_ir.types import (
    AttrBool,
    AttrBytes,
    AttrDType,
    AttrFloat,
    AttributeValue,
    AttrInt,
    AttrMap,
    AttrModel,
    AttrNone,
    AttrProtocolLeaf,
    AttrSeq,
    AttrShape,
    AttrStr,
    AttrSymInt,
    AttrSymNode,
    AttrTensorSpec,
    DType,
    TensorShape,
    TensorSpec,
    embed_attribute_value,
    project_attribute_value,
)
from ml_switcheroo_ir.validation.core import ValidationLevel, validation_level_rank
from ml_switcheroo_ir.validation.grounding import DiagnosticSeverity, GroundingDiagnostic
from ml_switcheroo_ir.validation.schema_validator import check_attribute_type_value


def test_closed_sym_expr_guards_and_folds() -> None:
    """Test SymExpr TypeGuards, exhaustive free_vars, eval, simplify, format, and poly_from_sym_expr."""
    assert is_binary_sym_op("+") is True
    assert is_binary_sym_op("??") is False
    assert is_unary_sym_op("abs") is True
    assert is_unary_sym_op("??") is False
    assert is_comparison_op("<=") is True
    assert is_comparison_op("~=") is False

    c = SymConst(4)
    v = SymVar("N")
    b = SymBinaryOp("+", v, c)
    u = SymUnaryOp("-", v)
    pw = SymPiecewise(cases=[(SymBinaryOp(">", v, SymConst(0)), b), ("N == 0", c)], default=u)

    for expr in (c, v, b, u, pw):
        assert is_sym_expr(expr) is True
    assert is_sym_expr(SymNode()) is False

    assert sym_expr_free_vars(c) == set()
    assert sym_expr_free_vars(v) == {"N"}
    assert sym_expr_free_vars(b) == {"N"}
    assert sym_expr_free_vars(u) == {"N"}
    assert sym_expr_free_vars(pw) == {"N"}

    env = {"N": 5}
    assert sym_expr_eval(c, env) == 4
    assert sym_expr_eval(v, env) == 5
    assert sym_expr_eval(b, env) == 9
    assert sym_expr_eval(u, env) == -5
    assert sym_expr_eval(pw, env) == 9

    assert sym_expr_simplify(c) == c
    assert sym_expr_simplify(v) == v
    assert sym_expr_simplify(SymBinaryOp("+", SymConst(2), SymConst(3))) == SymConst(5)
    assert sym_expr_simplify(SymUnaryOp("-", SymConst(3))) == SymConst(-3)
    assert sym_expr_simplify(SymPiecewise(cases=[(SymBinaryOp("==", SymConst(1), SymConst(1)), c)], default=v)) == c

    assert sym_expr_format(c) == "4"
    assert sym_expr_format(v) == "N"
    assert "N" in sym_expr_format(b)
    assert "-" in sym_expr_format(u)
    assert len(sym_expr_format(pw)) > 0

    assert poly_from_sym_expr(c) == Polynomial.from_const(4)
    assert poly_from_sym_expr(v) == Polynomial.from_var("N")
    assert poly_from_sym_expr(b) == Polynomial.from_var("N").add(Polynomial.from_const(4))
    assert poly_from_sym_expr(u) == Polynomial.from_const(0).sub(Polynomial.from_var("N"))
    assert poly_from_sym_expr(SymUnaryOp("+", v)) == Polynomial.from_var("N")
    assert poly_from_sym_expr(SymUnaryOp("abs", v)) is None
    assert poly_from_sym_expr(pw) is None


def test_sym_expr_witnesses_and_rewrites() -> None:
    """Test SYM_EXPR_EQ, SYM_EXPR_HASH, SYM_EXPR_REWRITE, and substitute_sym_expr."""
    c = SymConst(4)
    v = SymVar("N")
    b = SymBinaryOp("+", v, c)
    u = SymUnaryOp("-", v)
    open_leaf = SymNode()
    b_open = SymBinaryOp("+", open_leaf, c)
    pw = SymPiecewise(
        cases=[(SymBinaryOp(">", v, SymConst(0)), b), ("N == 0", c), ("bad$cond", u)],
        default=u,
    )

    assert SYM_EXPR_EQ.eq(b, SymBinaryOp("+", SymVar("N"), SymConst(4))) is True
    assert SYM_EXPR_HASH.hash_value(b) == SYM_EXPR_HASH.hash_value(SymBinaryOp("+", SymVar("N"), SymConst(4)))

    doubled: SymExpr = SYM_EXPR_REWRITE.bottom_up(
        pw,
        lambda node: SymConst(node.value * 2) if isinstance(node, SymConst) else node,
    )
    assert isinstance(doubled, SymPiecewise)
    rebuilt_open = SYM_EXPR_REWRITE.bottom_up(b_open, lambda node: node)
    assert isinstance(rebuilt_open, SymBinaryOp) and rebuilt_open.left is open_leaf

    assert substitute_sym_expr(pw, {}) == pw
    assert substitute_sym_expr(pw, {"N": 5}) == SymConst(9)


class _DummyDevice:
    """Minimal DeviceLikeProtocol implementation for AttrProtocolLeaf testing.

    Attributes:
        type (str): Backend device type name.
        index (int | None): Optional device index.
    """

    def __init__(self) -> None:
        """Initialize dummy CUDA:0 device."""
        self.type: str = "cuda"
        self.index: int | None = 0


def test_attr_term_projection_and_embedding_roundtrip() -> None:
    """Test project_attribute_value and embed_attribute_value across all 15 AttrTerm constructors."""
    dev = _DummyDevice()
    p_spec = PartitionSpec(("tp",))
    t_shape = TensorShape((2, 4))
    t_spec = TensorSpec(shape=(2, 4), dtype=DType.float32)
    sym_n = SymVar("B")
    sym_i = SymInt("S")

    assert isinstance(project_attribute_value(None), AttrNone)
    assert embed_attribute_value(project_attribute_value(None)) is None

    assert isinstance(project_attribute_value(True), AttrBool)
    assert embed_attribute_value(project_attribute_value(True)) is True

    assert isinstance(project_attribute_value(7), AttrInt)
    assert embed_attribute_value(project_attribute_value(7)) == 7

    assert isinstance(project_attribute_value(1.5), AttrFloat)
    assert embed_attribute_value(project_attribute_value(1.5)) == 1.5

    assert isinstance(project_attribute_value("relu"), AttrStr)
    assert embed_attribute_value(project_attribute_value("relu")) == "relu"

    assert isinstance(project_attribute_value(b"\x01\x02"), AttrBytes)
    assert embed_attribute_value(project_attribute_value(b"\x01\x02")) == b"\x01\x02"

    assert isinstance(project_attribute_value(DType.int64), AttrDType)
    assert embed_attribute_value(project_attribute_value(DType.int64)) is DType.int64

    assert isinstance(project_attribute_value(t_shape), AttrShape)
    assert embed_attribute_value(project_attribute_value(t_shape)) == t_shape

    assert isinstance(project_attribute_value(t_spec), AttrTensorSpec)
    assert embed_attribute_value(project_attribute_value(t_spec)) == t_spec

    assert isinstance(project_attribute_value(sym_n), AttrSymNode)
    assert embed_attribute_value(project_attribute_value(sym_n)) == sym_n

    assert isinstance(project_attribute_value(sym_i), AttrSymInt)
    assert embed_attribute_value(project_attribute_value(sym_i)) == sym_i

    assert isinstance(project_attribute_value(p_spec), AttrModel)
    assert embed_attribute_value(project_attribute_value(p_spec)) == p_spec

    assert isinstance(project_attribute_value(dev), AttrProtocolLeaf)
    assert embed_attribute_value(project_attribute_value(dev)) is dev

    nested: AttributeValue = {"a": [1, True, (2, "x")]}
    projected = project_attribute_value(nested)
    assert isinstance(projected, AttrMap)
    assert isinstance(projected.entries[0][1], AttrSeq)
    assert embed_attribute_value(projected) == {"a": [1, True, (2, "x")]}


def test_exhaustive_attribute_type_and_validation_level_dispatch() -> None:
    """Test describe_attribute_type, check_attribute_type_value, validation_level_rank, and GroundingDiagnostic."""
    for member in AttributeType:
        desc = describe_attribute_type(member)
        assert len(desc) > 0

    sub_g = LogicalGraph(nodes={"n": LogicalNode(id="n", op_type="Relu", outputs=["y"])})
    t_spec = TensorSpec(shape=(2,), dtype=DType.float32)

    assert check_attribute_type_value(AttributeType.INT, 3) is True
    assert check_attribute_type_value(AttributeType.INT, True) is False
    assert check_attribute_type_value(AttributeType.FLOAT, 1.5) is True
    assert check_attribute_type_value(AttributeType.FLOAT, False) is False
    assert check_attribute_type_value(AttributeType.STR, "ok") is True
    assert check_attribute_type_value(AttributeType.BOOL, True) is True
    assert check_attribute_type_value(AttributeType.DICT, {"k": 1}) is True
    assert check_attribute_type_value(AttributeType.LIST_INT, [1, 2]) is True
    assert check_attribute_type_value(AttributeType.LIST_INT, [1, True]) is False
    assert check_attribute_type_value(AttributeType.LIST_FLOAT, [1.0, 2]) is True
    assert check_attribute_type_value(AttributeType.LIST_STR, ["a", "b"]) is True
    assert check_attribute_type_value(AttributeType.LIST, [1, "a"]) is True
    assert check_attribute_type_value(AttributeType.TENSOR, t_spec) is True
    assert check_attribute_type_value(AttributeType.GRAPH, sub_g) is True
    assert check_attribute_type_value(AttributeType.DTYPE, DType.float32) is True
    assert check_attribute_type_value(AttributeType.ANY, None) is True

    assert validation_level_rank(ValidationLevel.LENIENT) == 0
    assert validation_level_rank(ValidationLevel.WARNING) == 1
    assert validation_level_rank(ValidationLevel.ERROR) == 2
    assert validation_level_rank(ValidationLevel.STRICT) == 3

    d_err = GroundingDiagnostic(field="op", message="m", severity=DiagnosticSeverity.ERROR)
    d_warn = GroundingDiagnostic(field="op", message="m", severity=DiagnosticSeverity.WARNING)
    d_info = GroundingDiagnostic(field="op", message="m", severity=DiagnosticSeverity.INFO)
    assert d_err.level is ValidationLevel.ERROR
    assert d_warn.level is ValidationLevel.WARNING
    assert d_info.level is ValidationLevel.WARNING
