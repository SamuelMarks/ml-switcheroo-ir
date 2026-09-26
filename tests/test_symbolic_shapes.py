"""Tests for symbolic shape primitives, polynomial canonicalization, and SymInt algebra."""

from __future__ import annotations

import pytest

from ml_switcheroo_ir.shapes import (
    Polynomial,
    SymBinaryOp,
    SymbolicSolver,
    SymConst,
    SymInt,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
    _parse_sym_str,
    _poly_to_symnode,
    _tokenize_expr,
)
from ml_switcheroo_ir.types import DType, TensorShape, TensorSpec


def test_polynomial_arithmetic_and_canonicalization() -> None:
    """Test Polynomial construction, addition, subtraction, multiplication, and division."""
    p_zero = Polynomial()
    assert p_zero.is_zero()
    assert p_zero.is_const()
    assert p_zero.get_const() == 0
    assert str(p_zero) == "0"
    assert repr(p_zero) == "Polynomial(0)"

    p_c5 = Polynomial.from_const(5)
    assert not p_c5.is_zero()
    assert p_c5.is_const()
    assert p_c5.get_const() == 5
    assert str(p_c5) == "5"

    p_c0 = Polynomial.from_const(0)
    assert p_c0.is_zero()

    p_x = Polynomial.from_var("x")
    assert not p_x.is_const()
    assert str(p_x) == "x"

    p_y = Polynomial.from_var("y")

    # Addition
    p_sum = p_x.add(p_y)
    assert "x" in str(p_sum) and "y" in str(p_sum)
    p_diff = p_x.sub(p_x)
    assert p_diff.is_zero()

    # Subtraction with negative coefficient
    p_neg = p_x.sub(p_y).sub(p_c5)
    assert "-" in str(p_neg)

    # Multiplication
    p_xy = p_x.mul(p_y)
    assert p_xy.eval({"x": 3, "y": 4}) == 12

    p_sq = p_x.mul(p_x)
    assert "x^2" in str(p_sq)
    assert p_sq.eval({"x": 5}) == 25

    # Equality and hashing
    p_x2 = Polynomial.from_var("x")
    assert p_x == p_x2
    assert p_x != "not_a_poly"
    assert hash(p_x) == hash(p_x2)

    # Missing env binding raises KeyError
    with pytest.raises(KeyError, match="Variable 'x' not bound"):
        p_x.eval({})

    # Exact division
    assert p_sq.div_exact(p_x) == p_x
    assert p_sq.div_exact(p_zero) is None
    assert p_sq.div_exact(p_sq) == Polynomial.from_const(1)

    p_c10 = Polynomial.from_const(10)
    p_c2 = Polynomial.from_const(2)
    assert p_c10.div_exact(p_c2) == p_c5
    assert p_c10.div_exact(Polynomial.from_const(3)) is None
    assert p_c10.div_exact(p_c0) is None

    # Monomial division failure
    assert p_x.div_exact(p_y) is None
    p_2x = p_x.add(p_x)
    p_3x = p_2x.add(p_x)
    assert p_3x.div_exact(p_2x) is None  # coeff indivisible


def test_poly_to_symnode() -> None:
    """Test converting Polynomial back into SymNode expression trees."""
    p_zero = Polynomial()
    n_zero = _poly_to_symnode(p_zero)
    assert isinstance(n_zero, SymConst) and n_zero.value == 0

    p_c7 = Polynomial.from_const(7)
    n_c7 = _poly_to_symnode(p_c7)
    assert isinstance(n_c7, SymConst) and n_c7.value == 7

    p_neg_x = Polynomial({((("x", 1),)): -2})
    n_neg_x = _poly_to_symnode(p_neg_x)
    assert n_neg_x.eval({"x": 3}) == -6

    p_poly = Polynomial({((("x", 1),)): 2, ((("y", 1),)): -1, (): 3})
    n_poly = _poly_to_symnode(p_poly)
    assert n_poly.eval({"x": 4, "y": 5}) == (8 - 5 + 3)


def test_tokenize_and_parse_sym_str() -> None:
    """Test string tokenizer and recursive descent expression parser."""
    assert _tokenize_expr("") == []
    tokens = _tokenize_expr("2 * (batch + 4) // 2 - seq_len % 3 ** 2")
    assert tokens == [
        "2",
        "*",
        "(",
        "batch",
        "+",
        "4",
        ")",
        "//",
        "2",
        "-",
        "seq_len",
        "%",
        "3",
        "**",
        "2",
    ]

    single_slash = _tokenize_expr("batch / 2")
    assert single_slash == ["batch", "//", "2"]

    # Parsing expressions
    n1 = _parse_sym_str("batch + 4")
    assert isinstance(n1, SymBinaryOp)
    assert n1.eval({"batch": 10}) == 14

    n2 = _parse_sym_str("2 * -x")
    assert n2.eval({"x": 3}) == -6

    n3 = _parse_sym_str("x ** 2")
    assert n3.eval({"x": 4}) == 16

    # Empty string or fallback
    assert isinstance(_parse_sym_str(""), SymVar)
    assert isinstance(_parse_sym_str("???"), SymVar)


def test_sym_node_base_methods() -> None:
    """Test SymNode base class conversion and default methods."""
    n_const = SymNode.to_node(10)
    assert isinstance(n_const, SymConst) and n_const.value == 10

    n_var = SymNode.to_node("batch")
    assert isinstance(n_var, (SymVar, SymNode))

    sym_int = SymInt("seq_len")
    assert SymNode.to_node(sym_int) is sym_int.node
    assert SymNode.to_node(n_const) is n_const

    with pytest.raises(TypeError, match="Cannot convert"):
        SymNode.to_node(1.5)  # type: ignore[arg-type]

    # Base SymNode eval raises NotImplementedError
    base = SymNode()
    assert base.simplify() is base
    with pytest.raises(NotImplementedError):
        base.eval({"x": 1})


def test_sym_const_and_sym_var() -> None:
    """Test SymConst and SymVar operations, evaluation, and representation."""
    c3 = SymConst(3)
    assert c3.simplify() is c3
    assert c3.eval({}) == 3
    assert c3.evaluate({}) == 3
    assert c3.free_vars() == set()
    assert c3.is_constant
    assert int(c3) == 3
    assert str(c3) == "3"
    assert repr(c3) == "SymConst(3)"
    assert c3 == 3
    assert c3 == SymConst(3)
    assert c3 != "3"
    assert hash(c3) == hash(3)

    vx = SymVar("x")
    assert vx.simplify() is vx
    assert vx.eval({"x": 12}) == 12
    assert vx.evaluate({"x": 12}) == 12
    assert vx.free_vars() == {"x"}
    assert not vx.is_constant
    assert str(vx) == "x"
    assert repr(vx) == "SymVar('x')"
    assert vx == SymVar("x")
    assert vx != "x"
    assert hash(vx) == hash("x")

    with pytest.raises(KeyError, match="Variable 'x' not bound"):
        vx.eval({})


def test_sym_binary_op_algebraic_simplifications() -> None:
    """Test algebraic simplifications and constant folding in SymBinaryOp."""
    vx = SymVar("x")
    vy = SymVar("y")
    c0 = SymConst(0)
    c1 = SymConst(1)
    c2 = SymConst(2)
    c3 = SymConst(3)

    # Constant folding
    assert (c2 + c3).simplify() == SymConst(5)
    assert (c3 - c2).simplify() == SymConst(1)
    assert (c2 * c3).simplify() == SymConst(6)
    assert (SymConst(7) // c2).simplify() == SymConst(3)
    assert (SymConst(7) % c2).simplify() == SymConst(1)
    assert (c2**c3).simplify() == SymConst(8)
    assert SymBinaryOp("min", c2, c3).simplify() == SymConst(2)
    assert SymBinaryOp("max", c2, c3).simplify() == SymConst(3)

    # Identity rules: +
    assert (vx + c0).simplify() == vx
    assert (c0 + vx).simplify() == vx

    # Identity rules: -
    assert (vx - c0).simplify() == vx
    assert (vx - vx).simplify() == c0
    assert ((vx + vy) - vy).simplify() == vx
    assert ((vy + vx) - vy).simplify() == vx

    # Identity rules: *
    assert (vx * c1).simplify() == vx
    assert (c1 * vx).simplify() == vx
    assert (vx * c0).simplify() == c0
    assert (c0 * vx).simplify() == c0

    # Identity rules: //
    assert (vx // c1).simplify() == vx
    assert (c0 // vx).simplify() == c0
    assert (vx // vx).simplify() == c1
    assert ((vx * vy) // vy).simplify() == vx
    assert ((vy * vx) // vy).simplify() == vx
    assert ((vx * SymConst(6)) // c2).simplify() == (vx * c3).simplify()
    assert ((SymConst(6) * vx) // c2).simplify() == (vx * c3).simplify()

    # Identity rules: %
    assert (vx % c1).simplify() == c0
    assert (c0 % vx).simplify() == c0
    assert (vx % vx).simplify() == c0

    # Identity rules: **
    assert (vx**c1).simplify() == vx
    assert (vx**c0).simplify() == c1
    assert (c0**vx).simplify() == c0
    assert (c1**vx).simplify() == c1

    # Identity rules: min / max
    assert SymBinaryOp("min", vx, vx).simplify() == vx
    min_nested = SymBinaryOp("min", SymBinaryOp("min", vx, vy), vx)
    assert min_nested.simplify() == SymBinaryOp("min", vx, vy)
    min_nested2 = SymBinaryOp("min", vx, SymBinaryOp("min", vx, vy))
    assert min_nested2.simplify() == SymBinaryOp("min", vx, vy)

    assert SymBinaryOp("max", vx, vx).simplify() == vx
    max_nested = SymBinaryOp("max", SymBinaryOp("max", vx, vy), vx)
    assert max_nested.simplify() == SymBinaryOp("max", vx, vy)
    max_nested2 = SymBinaryOp("max", vx, SymBinaryOp("max", vx, vy))
    assert max_nested2.simplify() == SymBinaryOp("max", vx, vy)


def test_sym_binary_op_evaluation_and_errors() -> None:
    """Test SymBinaryOp evaluations and division/modulo by zero exceptions."""
    vx = SymVar("x")
    vy = SymVar("y")

    # Comparisons and logic
    eq_op = SymBinaryOp("==", vx, vy)
    assert eq_op.eval({"x": 5, "y": 5}) == 1
    assert eq_op.eval({"x": 5, "y": 6}) == 0

    ne_op = SymBinaryOp("!=", vx, vy)
    assert ne_op.eval({"x": 5, "y": 6}) == 1

    lt_op = SymBinaryOp("<", vx, vy)
    assert lt_op.eval({"x": 4, "y": 5}) == 1
    le_op = SymBinaryOp("<=", vx, vy)
    assert le_op.eval({"x": 5, "y": 5}) == 1

    gt_op = SymBinaryOp(">", vx, vy)
    assert gt_op.eval({"x": 6, "y": 5}) == 1
    ge_op = SymBinaryOp(">=", vx, vy)
    assert ge_op.eval({"x": 5, "y": 5}) == 1

    div_zero = SymBinaryOp("//", vx, vy)
    with pytest.raises(ZeroDivisionError):
        div_zero.eval({"x": 10, "y": 0})

    mod_zero = SymBinaryOp("%", vx, vy)
    with pytest.raises(ZeroDivisionError):
        mod_zero.eval({"x": 10, "y": 0})

    unsupported = SymBinaryOp("&", vx, vy)
    with pytest.raises(ValueError, match="Unsupported operator"):
        unsupported.eval({"x": 1, "y": 2})

    # Equality and representation
    b1 = SymBinaryOp("+", vx, vy)
    b2 = SymBinaryOp("+", vx, vy)
    assert b1 == b2
    assert b1 != "not_op"
    assert hash(b1) == hash(b2)
    assert repr(b1) == f"SymBinaryOp('+', {vx!r}, {vy!r})"
    assert str(b1) == "(x + y)"
    assert b1.free_vars() == {"x", "y"}


def test_sym_unary_op() -> None:
    """Test SymUnaryOp simplification, evaluation, and representation."""
    vx = SymVar("x")
    c_neg5 = SymConst(-5)

    u_neg = -vx
    assert u_neg.eval({"x": 8}) == -8
    assert (-c_neg5).simplify() == SymConst(5)
    assert (-SymUnaryOp("-", vx)).simplify() == vx

    u_abs = abs(vx)
    assert u_abs.eval({"x": -9}) == 9
    assert abs(c_neg5).simplify() == SymConst(5)
    assert abs(abs(vx)).simplify() == u_abs

    bad_unary = SymUnaryOp("invalid", vx)
    with pytest.raises(ValueError, match="Unsupported unary operator"):
        bad_unary.eval({"x": 1})

    assert repr(u_neg) == f"SymUnaryOp('-', {vx!r})"
    assert str(u_neg) == "-(x)"
    assert str(u_abs) == "abs(x)"
    assert u_neg == -vx
    assert u_neg != u_abs
    assert u_neg != "not_unary"
    assert hash(u_neg) == hash(SymUnaryOp("-", vx))
    assert u_neg.free_vars() == {"x"}


def test_sym_piecewise() -> None:
    """Test SymPiecewise condition evaluation and simplification."""
    vx = SymVar("x")

    # Conditions can be callable, SymNode, bool, or str
    pw = SymPiecewise(
        cases=[
            (lambda env: env["x"] < 0, SymConst(0)),
            (SymBinaryOp("==", vx, SymConst(1)), SymConst(100)),
            (True, vx * 2),
        ],
        default=SymConst(-1),
    )

    assert pw.eval({"x": -5}) == 0
    assert pw.eval({"x": 1}) == 100
    assert pw.eval({"x": 10}) == 20

    # Default fallback
    pw2 = SymPiecewise(
        conditions=[
            (False, SymConst(42)),
            ("x == 99", SymConst(999)),
        ],
        default=SymConst(777),
    )
    assert pw2.eval({"x": 5}) == 777
    assert pw2.eval({"x": 99}) == 999

    pw_simp = pw.simplify()
    assert pw_simp.eval({"x": 10}) == 20
    pw_clone = SymPiecewise(list(pw.cases), pw.default)
    assert pw == pw_clone
    assert pw != pw2
    assert pw != "other"
    assert repr(pw).startswith("SymPiecewise")
    assert str(pw).startswith("Piecewise")
    assert "x" in pw.free_vars()
    assert hash(pw) is not None


def test_sym_int_wrapper_and_operations() -> None:
    """Test SymInt wrapper arithmetic dunders, comparisons, int casting, and canonicalization."""
    si_x = SymInt("x")
    si_y = SymInt(SymVar("y"))
    si_5 = SymInt(5)
    si_copy = SymInt(si_x)

    assert si_x.name == "x"
    assert si_copy.node == si_x.node
    assert repr(si_x) == "SymInt(x)"
    assert str(si_x) == "x"
    assert hash(si_x) == hash("x")

    # Evaluation
    assert si_x.eval({"x": 20}) == 20
    assert si_x.evaluate({"x": 20}) == 20
    assert not si_x.is_constant
    assert si_5.is_constant
    assert int(si_5) == 5

    with pytest.raises(TypeError, match="Cannot convert dynamic SymInt"):
        int(si_x)

    # Arithmetic
    s_add = si_x + 3
    assert s_add.eval({"x": 7}) == 10
    s_radd = 3 + si_x
    assert s_radd.eval({"x": 7}) == 10

    s_sub = si_x - 4
    assert s_sub.eval({"x": 10}) == 6
    s_rsub = 10 - si_x
    assert s_rsub.eval({"x": 4}) == 6

    s_mul = si_x * 4
    assert s_mul.eval({"x": 3}) == 12
    s_rmul = 4 * si_x
    assert s_rmul.eval({"x": 3}) == 12

    s_div = si_x // 2
    assert s_div.eval({"x": 15}) == 7
    s_rdiv = 100 // si_x
    assert s_rdiv.eval({"x": 10}) == 10

    s_mod = si_x % 3
    assert s_mod.eval({"x": 10}) == 1
    s_rmod = 25 % si_x
    assert s_rmod.eval({"x": 6}) == 1

    s_pow = si_x**2
    assert s_pow.eval({"x": 4}) == 16
    s_rpow = 2**si_x
    assert s_rpow.eval({"x": 3}) == 8

    s_neg = -si_x
    assert s_neg.eval({"x": 8}) == -8

    s_abs = abs(si_x)
    assert s_abs.eval({"x": -15}) == 15

    # Canonical and simplify
    si_expr = (si_x + 0) * 1
    assert str(si_expr.simplify()) == "x"
    assert str(si_expr.canonical()) == "x"

    # Equivalence using SymbolicSolver
    assert si_x == SymInt("x")
    assert si_x != si_y
    assert (si_x + 2) == (2 + si_x)
    assert si_x != 123


def test_symbolic_solver_consistency() -> None:
    """Test SymbolicSolver equivalence checks between polynomials and strings."""
    assert SymbolicSolver.is_consistent(5, 5)
    assert not SymbolicSolver.is_consistent(5, 6)

    # Commutative addition
    assert SymbolicSolver.is_consistent("x + y", "y + x")
    # Algebraic expansion
    assert SymbolicSolver.is_consistent("2 * (x + 1)", "2 * x + 2")
    # Distributive multiplication
    assert not SymbolicSolver.is_consistent("x + 1", "x + 2")


def test_tensor_shape_and_spec_symbolic_integration() -> None:
    """Test TensorShape and TensorSpec integration with SymNode and SymInt."""
    si_b = SymInt("batch")
    si_s = SymInt("seq_len")
    shape = TensorShape((si_b, si_s, 64))

    assert shape.rank == 3
    assert shape.is_dynamic
    with pytest.raises(ValueError, match="contains dynamic"):
        _ = shape.static_shape

    evaluated = shape.evaluate({"batch": 4, "seq_len": 128})
    assert evaluated == (4, 128, 64)

    # String dimension evaluation
    shape_str = TensorShape((si_b, "16", "seq_len", "batch + 2"))
    assert shape_str.evaluate({"batch": 4, "seq_len": 8}) == (4, 16, 8, 6)

    # Static shape evaluation
    static_shape = TensorShape((SymConst(4), 128, 64))
    assert not static_shape.is_dynamic
    assert static_shape.static_shape == (4, 128, 64)

    # TensorSpec integration
    spec = TensorSpec(shape=shape, dtype=DType.float32)
    assert spec.is_dynamic
    assert spec.rank == 3
    with pytest.raises(ValueError, match="dynamic dimensions"):
        _ = spec.static_shape

    spec_const = TensorSpec(shape=(SymConst(2), 64), dtype=DType.float32)
    assert not spec_const.is_dynamic
    assert spec_const.static_shape == (2, 64)

    spec_evaluated = spec.evaluate({"batch": 2, "seq_len": 64})
    assert not spec_evaluated.is_dynamic
    assert spec_evaluated.static_shape == (2, 64, 64)
    assert spec_evaluated.dtype == DType.float32

    # broadcast_with and matmul_with methods on TensorShape
    ts_base = TensorShape((2, 1, 8))
    assert ts_base.broadcast_with((4, 8)) == TensorShape((2, 4, 8))
    assert ts_base.broadcast_with(TensorShape((4, 8))) == TensorShape((2, 4, 8))

    ts_m1 = TensorShape((16, 32))
    assert ts_m1.matmul_with((32, 64)) == TensorShape((16, 64))
    assert ts_m1.matmul_with(TensorShape((32, 64))) == TensorShape((16, 64))

    # Test types.__getattr__ for ZeroTangent and NoTangent
    import ml_switcheroo_ir.types as types_mod

    assert types_mod.ZeroTangent is not None
    assert types_mod.NoTangent is not None
    with pytest.raises(AttributeError):
        _ = types_mod.NonExistentType


def test_symbolic_shapes_exhaustive_branches() -> None:
    """Test exhaustive branches in Polynomial, SymNode, SymBinaryOp, and SymInt."""
    vx = SymVar("x")
    vy = SymVar("y")
    c0 = SymConst(0)
    c1 = SymConst(1)
    c2 = SymConst(2)

    # Polynomial index 0 negative constant and multi-power monomials
    p_neg_const = Polynomial.from_const(-8)
    assert str(p_neg_const) == "-8"

    p_multi = Polynomial({((("x", 2), ("y", 3))): -4, (): 5})
    assert p_multi.eval({"x": 2, "y": 1}) == (-4 * 4 * 1 + 5)
    assert p_multi.canonical_str() != ""

    # div_exact branches
    p_cube = Polynomial({((("x", 3),)): 1})
    p_quad = Polynomial({((("x", 4),)): 1})
    assert p_cube.div_exact(p_quad) is None

    # _poly_to_symnode with power >= 2
    n_cube = _poly_to_symnode(p_cube)
    assert n_cube.eval({"x": 2}) == 8

    # Tokenizer and parser edge cases
    tokens = _tokenize_expr("   x  +  1   ")
    assert tokens == ["x", "+", "1"]
    assert _tokenize_expr("x < 5") == ["x", "<", "5"]
    assert _tokenize_expr("x > 5") == ["x", ">", "5"]
    assert _tokenize_expr("x = 5") == ["x", "=", "5"]
    assert _tokenize_expr("x ! 5") == ["x", "!", "5"]
    assert _tokenize_expr("") == []
    assert isinstance(_parse_sym_str("   "), SymVar)
    assert _parse_sym_str("x *").eval({"x": 5}) == 0
    assert isinstance(_parse_sym_str("x + 1 extra_token"), SymVar)
    n_unclosed = _parse_sym_str("(x + 1")
    assert n_unclosed.eval({"x": 5}) == 6
    n_paren_pow = _parse_sym_str("(x + 1) ** 2")
    assert n_paren_pow.eval({"x": 3}) == 16

    # SymNode right dunders
    assert (2 + vx).eval({"x": 3}) == 5
    assert (10 - vx).eval({"x": 3}) == 7
    assert (4 * vx).eval({"x": 3}) == 12
    assert (12 // vx).eval({"x": 3}) == 4
    assert (10 % vx).eval({"x": 3}) == 1
    assert (2**vx).eval({"x": 3}) == 8

    # SymNode.canonical and eval fallback
    base_node = SymNode()
    assert base_node.to_polynomial() is None
    assert base_node.canonical() is base_node
    assert base_node.free_vars() == set()

    class SimplifyingNode(SymNode):
        """Mock custom node with simplify override."""

        def simplify(self) -> SymNode:
            """Simplify to constant.

            Returns:
                SymNode: Constant node.
            """
            return SymConst(99)

    assert SimplifyingNode().eval({}) == 99

    # Pydantic core schemas
    assert SymNode.__get_pydantic_core_schema__(SymNode, None) is not None
    assert SymInt.__get_pydantic_core_schema__(SymInt, None) is not None

    # SymBinaryOp identities
    assert ((vx + vy) - vx).simplify() == vy
    assert (vx * c0).simplify() == c0
    assert (c0 * vx).simplify() == c0
    assert (vx * c1).simplify() == vx
    assert (c1 * vx).simplify() == vx
    assert (c0 // vx).simplify() == c0
    assert (vx // c1).simplify() == vx
    assert (vx // vx).simplify() == c1
    assert ((vx * vy) // vx).simplify() == vy
    assert ((vy * vx) // vx).simplify() == vy
    assert ((SymConst(6) * vx) // c2).simplify() == (vx * SymConst(3)).simplify()
    assert ((vx * SymConst(6)) // c2).simplify() == (vx * SymConst(3)).simplify()
    assert (vx % c1).simplify() == c0
    assert (c0 % vx).simplify() == c0
    assert (vx % vx).simplify() == c0
    assert (vx**c0).simplify() == c1
    assert (vx**c1).simplify() == vx
    assert (c0**vx).simplify() == c0
    assert (c1**vx).simplify() == c1

    # Polynomial cancellation and zero coefficient branches
    p_zero_coeff = Polynomial({(): 0})
    assert p_zero_coeff.is_zero()

    p_x = Polynomial.from_var("x")
    p_y = Polynomial.from_var("y")
    p_canc_sum = p_x.add(Polynomial({((("x", 1),)): -1}))
    assert p_canc_sum.is_zero()

    p_c1 = Polynomial.from_const(1)
    p_canc_mul = p_x.add(p_c1).mul(p_x.sub(p_c1))
    assert p_canc_mul.eval({"x": 5}) == 24

    assert p_x.mul(p_y).div_exact(p_x) == p_y

    class PolyNode(SymNode):
        """Mock custom node with polynomial override."""

        def to_polynomial(self) -> Polynomial:
            """Convert to polynomial.

            Returns:
                Polynomial: Constant polynomial.
            """
            return Polynomial.from_const(123)

    assert PolyNode().eval({}) == 123

    # Nested min / max
    min1 = SymBinaryOp("min", vx, SymBinaryOp("min", vx, vy)).simplify()
    assert min1 == SymBinaryOp("min", vx, vy)
    min2 = SymBinaryOp("min", vx, SymBinaryOp("min", vy, vx)).simplify()
    assert min2 == SymBinaryOp("min", vy, vx)
    min3 = SymBinaryOp("min", vy, SymBinaryOp("min", vx, vy)).simplify()
    assert min3 == SymBinaryOp("min", vx, vy)

    max1 = SymBinaryOp("max", vx, SymBinaryOp("max", vx, vy)).simplify()
    assert max1 == SymBinaryOp("max", vx, vy)
    max2 = SymBinaryOp("max", vx, SymBinaryOp("max", vy, vx)).simplify()
    assert max2 == SymBinaryOp("max", vy, vx)
    max3 = SymBinaryOp("max", vy, SymBinaryOp("max", vx, vy)).simplify()
    assert max3 == SymBinaryOp("max", vx, vy)

    # Directly constructed SymBinaryOp division with integer factor on left
    assert (
        SymBinaryOp("//", SymBinaryOp("*", SymConst(6), vx), SymConst(2)).simplify()
        == (SymConst(3) * vx).simplify()
    )

    # SymBinaryOp.to_polynomial
    assert (vx + vy).to_polynomial() is not None
    assert (vx - vy).to_polynomial() is not None
    assert (vx * vy).to_polynomial() is not None
    assert ((vx * vy) // vy).to_polynomial() is not None
    assert (vx // vy).to_polynomial() is None
    assert (vx % vy).to_polynomial() is None
    assert SymBinaryOp("+", vx % 2, vy).to_polynomial() is None

    # SymBinaryOp eval
    assert SymBinaryOp("min", vx, vy).eval({"x": 3, "y": 7}) == 3
    assert SymBinaryOp("max", vx, vy).eval({"x": 3, "y": 7}) == 7
    assert (vx**vy).eval({"x": 2, "y": 3}) == 8

    # SymUnaryOp
    assert (-SymUnaryOp("-", vx)).simplify() == vx
    assert abs(abs(vx)).simplify() == abs(vx)

    # SymPiecewise free_vars
    pw_vars = SymPiecewise([(SymBinaryOp("<", vx, SymConst(5)), vy)], default=c0)
    assert pw_vars.free_vars() == {"x", "y"}

    # SymInt comparisons
    si_x = SymInt("x")
    si_y = SymInt("y")
    assert si_x != si_y
    assert (si_x == 10) is False
    assert (si_x != 10) is True
    assert (si_x == [1, 2]) is False
    assert SymbolicSolver.is_consistent(vx % 2, vx % 2) is True
    assert (
        SymbolicSolver.is_consistent(vx % 2, SymBinaryOp("%", vx + 0, SymConst(2)))
        is True
    )

    # div_exact with multi-term divisor returns None
    assert p_x.div_exact(p_x.add(p_y)) is None

    # Irreducible binary operations
    assert SymBinaryOp("&", vx, vy).simplify() == SymBinaryOp("&", vx, vy)
    assert SymBinaryOp("-", vx, vy).simplify() == (vx - vy)
    assert SymBinaryOp("//", vx, vy).simplify() == (vx // vy)
    assert SymBinaryOp("%", vx, vy).simplify() == (vx % vy)
    assert SymBinaryOp("min", vx, vy).simplify() == SymBinaryOp("min", vx, vy)
    assert SymBinaryOp("max", vx, vy).simplify() == SymBinaryOp("max", vx, vy)

    # Non-polynomial binary simplifications (+, -, // with non-poly left)
    assert SymBinaryOp("+", vx % 2, vy).simplify() == SymBinaryOp("+", vx % 2, vy)
    assert SymBinaryOp("+", vx % 2, SymConst(3)).simplify() == SymBinaryOp(
        "+", vx % 2, SymConst(3)
    )
    assert SymBinaryOp("+", SymConst(3), vx % 2).simplify() == SymBinaryOp(
        "+", SymConst(3), vx % 2
    )
    assert SymBinaryOp("-", vx % 2, vy).simplify() == SymBinaryOp("-", vx % 2, vy)
    assert SymBinaryOp("-", vx % 2, SymConst(3)).simplify() == SymBinaryOp(
        "-", vx % 2, SymConst(3)
    )
    assert SymBinaryOp("-", (vx % 2) + vy, SymConst(3)).simplify() == SymBinaryOp(
        "-", (vx % 2) + vy, SymConst(3)
    )
    assert SymBinaryOp("//", vx % 2, vy).simplify() == SymBinaryOp("//", vx % 2, vy)
    assert SymBinaryOp("//", vx % 2, SymConst(3)).simplify() == SymBinaryOp(
        "//", vx % 2, SymConst(3)
    )
    assert SymBinaryOp("//", SymConst(3), vx % 2).simplify() == SymBinaryOp(
        "//", SymConst(3), vx % 2
    )
    assert SymBinaryOp("//", (vx % 2) * vy, SymVar("z")).simplify() == SymBinaryOp(
        "//", (vx % 2) * vy, SymVar("z")
    )
    assert SymBinaryOp(
        "//", SymBinaryOp("*", vx % 2, SymConst(5)), SymConst(2)
    ).simplify() == SymBinaryOp(
        "//", SymBinaryOp("*", vx % 2, SymConst(5)), SymConst(2)
    )
    assert SymBinaryOp(
        "//", SymBinaryOp("*", SymConst(5), vx % 2), SymConst(2)
    ).simplify() == SymBinaryOp(
        "//", SymBinaryOp("*", SymConst(5), vx % 2), SymConst(2)
    )
    assert SymBinaryOp("%", vx, SymConst(3)).simplify() == SymBinaryOp(
        "%", vx, SymConst(3)
    )
    assert SymBinaryOp("%", SymConst(3), vx).simplify() == SymBinaryOp(
        "%", SymConst(3), vx
    )
    assert SymBinaryOp(
        "min", SymBinaryOp("min", vx, vy), SymVar("z")
    ).simplify() == SymBinaryOp("min", SymBinaryOp("min", vx, vy), SymVar("z"))
    assert SymBinaryOp(
        "min", SymBinaryOp("+", vx % 2, vy), SymVar("z")
    ).simplify() == SymBinaryOp("min", SymBinaryOp("+", vx % 2, vy), SymVar("z"))
    assert SymBinaryOp(
        "min", SymVar("z"), SymBinaryOp("min", vx, vy)
    ).simplify() == SymBinaryOp("min", SymVar("z"), SymBinaryOp("min", vx, vy))
    assert SymBinaryOp(
        "min", SymVar("z"), SymBinaryOp("+", vx % 2, vy)
    ).simplify() == SymBinaryOp("min", SymVar("z"), SymBinaryOp("+", vx % 2, vy))
    assert SymBinaryOp(
        "max", SymBinaryOp("max", vx, vy), SymVar("z")
    ).simplify() == SymBinaryOp("max", SymBinaryOp("max", vx, vy), SymVar("z"))
    assert SymBinaryOp(
        "max", SymBinaryOp("+", vx % 2, vy), SymVar("z")
    ).simplify() == SymBinaryOp("max", SymBinaryOp("+", vx % 2, vy), SymVar("z"))
    assert SymBinaryOp(
        "max", SymVar("z"), SymBinaryOp("max", vx, vy)
    ).simplify() == SymBinaryOp("max", SymVar("z"), SymBinaryOp("max", vx, vy))
    assert SymBinaryOp(
        "max", SymVar("z"), SymBinaryOp("+", vx % 2, vy)
    ).simplify() == SymBinaryOp("max", SymVar("z"), SymBinaryOp("+", vx % 2, vy))

    # Binary op & with SymConst
    assert SymBinaryOp("&", SymConst(1), SymConst(2)).simplify() == SymBinaryOp(
        "&", SymConst(1), SymConst(2)
    )

    # Power simplify with non-polynomial subexpressions
    assert SymBinaryOp("**", vx % 2, SymConst(3)).simplify() == SymBinaryOp(
        "**", vx % 2, SymConst(3)
    )
    assert SymBinaryOp("**", vx % 2, vy).simplify() == SymBinaryOp("**", vx % 2, vy)
    assert SymBinaryOp("**", SymConst(3), vx % 2).simplify() == SymBinaryOp(
        "**", SymConst(3), vx % 2
    )

    # Unary simplify custom op and negated abs
    assert SymUnaryOp("custom", SymConst(5)).simplify() == SymUnaryOp(
        "custom", SymConst(5)
    )
    assert SymUnaryOp("-", SymUnaryOp("abs", vx)).simplify() == SymUnaryOp(
        "-", SymUnaryOp("abs", vx)
    )

    # Piecewise with unsupported condition type and empty cases free_vars
    assert SymPiecewise([(123, SymConst(5))], default=SymConst(0)).eval({}) == 0
    assert SymPiecewise([], default=SymConst(0)).free_vars() == set()
    assert SymUnaryOp("abs", vx).simplify() == SymUnaryOp("abs", vx)

    # Factor division where s_left.right is SymConst
    assert (
        SymBinaryOp("//", SymBinaryOp("*", vx % 2, SymConst(6)), SymConst(2)).simplify()
        == ((vx % 2) * SymConst(3)).simplify()
    )

    # Non-SymNode condition in Piecewise free_vars
    pw_non_sym = SymPiecewise([(True, vx), (lambda env: True, vy)], default=c0)
    assert pw_non_sym.free_vars() == {"x", "y"}
