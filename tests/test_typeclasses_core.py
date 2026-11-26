"""Comprehensive unit tests for Phase 1 core typeclass witnesses, outcome ADTs, codecs, and folds."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

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
    STR_HASH,
    STR_ORD,
    BoundedLattice,
    Codec,
    CodecError,
    Endo,
    EndoRewrite,
    Eq,
    Err,
    HashableWitness,
    Invalid,
    JsonCodec,
    Monoid,
    NonEmpty,
    Ok,
    Ord,
    Ordering,
    Pretty,
    PythonExprEmit,
    RecursiveFold,
    Result,
    Ring,
    Semigroup,
    Valid,
    Validation,
    and_then,
    ap_validation,
    collect_unit_validations,
    collect_unit_validations_ne,
    compose_codec,
    contramap_eq,
    contramap_hash,
    contramap_ord,
    dict_monoid,
    dict_str_codec,
    dual_semigroup,
    endo_monoid,
    first_semigroup,
    fold_result,
    fold_validation,
    from_error_sequence,
    frozenset_union_monoid,
    imap_codec,
    invalid_one,
    last_semigroup,
    lexical_pair_ord,
    list_codec,
    map_err,
    map_result,
    map_validation,
    map_validation_err,
    max_semigroup,
    min_semigroup,
    nonempty_eq,
    nonempty_hash,
    nonempty_ord,
    nonempty_pretty,
    nonempty_semigroup,
    optional_codec,
    pair_hash_witness,
    partition_results,
    product_monoid,
    raise_on_err,
    refine_codec,
    result_to_validation,
    traverse_result,
    traverse_validation,
    traverse_validation_ne,
    tuple_codec,
    tuple_hash_witness,
    tuple_monoid,
    validation_ne_to_tuple,
    validation_to_result,
    validation_tuple_to_ne,
    value_or,
    value_or_else,
    zip_validation2,
    zip_validation3,
)
from ml_switcheroo_ir.types import JsonValue, describe_value_type


def test_ordering_and_eq_ord_witnesses() -> None:
    """Test Ordering, Eq, Ord, clamp, sort_with, and contravariant/pair combinators."""
    assert Ordering.LT.flip() is Ordering.GT
    assert Ordering.GT.flip() is Ordering.LT
    assert Ordering.EQ.flip() is Ordering.EQ

    assert INT_EQ.eq(3, 3) is True
    assert INT_EQ.neq(3, 4) is True
    assert INT_EQ.neq(3, 3) is False
    assert STR_EQ.eq("a", "a") is True
    assert STR_EQ.neq("a", "b") is True

    len_eq: Eq[str] = contramap_eq(INT_EQ, len)
    assert len_eq.eq("abc", "xyz") is True
    assert len_eq.neq("ab", "xyz") is True

    assert INT_ORD.lt(1, 2) is True
    assert INT_ORD.lt(2, 1) is False
    assert INT_ORD.le(2, 2) is True
    assert INT_ORD.le(3, 2) is False
    assert INT_ORD.gt(5, 2) is True
    assert INT_ORD.gt(2, 5) is False
    assert INT_ORD.ge(2, 2) is True
    assert INT_ORD.ge(1, 2) is False
    assert INT_ORD.min_val(3, 7) == 3
    assert INT_ORD.min_val(9, 4) == 4
    assert INT_ORD.max_val(3, 7) == 7
    assert INT_ORD.max_val(9, 4) == 9

    assert INT_ORD.clamp(-5, 0, 10) == 0
    assert INT_ORD.clamp(15, 0, 10) == 10
    assert INT_ORD.clamp(6, 0, 10) == 6
    with pytest.raises(ValueError, match="lower <= upper"):
        INT_ORD.clamp(5, 10, 0)

    assert INT_ORD.sort_with([3, 1, 2, 1]) == [1, 1, 2, 3]
    assert STR_ORD.sort_with(["b", "a", "c", "a"]) == ["a", "a", "b", "c"]

    len_ord: Ord[str] = contramap_ord(INT_ORD, len)
    assert len_ord.lt("a", "bbb") is True
    assert len_ord.eq_witness.eq("ab", "cd") is True


def test_lexical_ord_and_hashable_witnesses() -> None:
    """Test lexical_pair_ord and HashableWitness combinators."""
    pair_ord = lexical_pair_ord(INT_ORD, STR_ORD)
    assert pair_ord.eq_witness.eq((1, "a"), (1, "a")) is True
    assert pair_ord.eq_witness.eq((1, "a"), (1, "b")) is False
    assert pair_ord.lt((1, "b"), (2, "a")) is True
    assert pair_ord.lt((1, "a"), (1, "b")) is True
    assert pair_ord.compare((2, "x"), (2, "x")) is Ordering.EQ

    len_hash: HashableWitness[str] = contramap_hash(INT_HASH, len)
    assert len_hash.eq_witness.eq("hi", "yo") is True
    assert len_hash.hash_value("hi") == len_hash.hash_value("yo")

    tup_hash = tuple_hash_witness(STR_HASH)
    assert tup_hash.eq_witness.eq(("a", "b"), ("a", "b")) is True
    assert tup_hash.eq_witness.eq(("a",), ("a", "b")) is False
    assert tup_hash.eq_witness.eq(("a", "c"), ("a", "b")) is False
    assert tup_hash.hash_value(("a", "b")) == tup_hash.hash_value(("a", "b"))

    pr_hash = pair_hash_witness(INT_HASH, STR_HASH)
    assert pr_hash.eq_witness.eq((1, "k"), (1, "k")) is True
    assert pr_hash.eq_witness.eq((1, "k"), (2, "k")) is False
    assert pr_hash.hash_value((1, "k")) == pr_hash.hash_value((1, "k"))


def test_semigroup_and_monoid_combinators() -> None:
    """Test Semigroup and Monoid primitive and higher-order combinators."""
    f_sg: Semigroup[str] = first_semigroup()
    l_sg: Semigroup[str] = last_semigroup()
    d_sg: Semigroup[str] = dual_semigroup(STR_CONCAT_MONOID.semigroup)
    mx_sg: Semigroup[int] = max_semigroup(INT_ORD)
    mn_sg: Semigroup[int] = min_semigroup(INT_ORD)

    assert f_sg.sconcat("first", ["second", "third"]) == "first"
    assert l_sg.sconcat("first", ["second", "third"]) == "third"
    assert d_sg.combine("a", "b") == "ba"
    assert mx_sg.sconcat(2, [9, 4, 7]) == 9
    assert mn_sg.sconcat(5, [9, 1, 7]) == 1

    assert INT_SUM_MONOID.mconcat([1, 2, 3, 4]) == 10
    assert INT_SUM_MONOID.combine(4, 6) == 10
    assert INT_PRODUCT_MONOID.mconcat([2, 3, 4]) == 24
    assert BOOL_ALL_MONOID.mconcat([True, True]) is True
    assert BOOL_ALL_MONOID.mconcat([True, False]) is False
    assert BOOL_ANY_MONOID.mconcat([False, False]) is False
    assert BOOL_ANY_MONOID.mconcat([False, True]) is True
    assert STR_CONCAT_MONOID.mconcat(["ml", "-", "ir"]) == "ml-ir"

    tm: Monoid[tuple[int, ...]] = tuple_monoid()
    assert tm.mconcat([(1, 2), (), (3,)]) == (1, 2, 3)

    fsm: Monoid[frozenset[str]] = frozenset_union_monoid()
    assert fsm.mconcat([frozenset({"a"}), frozenset({"b", "a"})]) == frozenset({"a", "b"})

    dm: Monoid[dict[str, int]] = dict_monoid(INT_SUM_MONOID.semigroup)
    assert dm.empty() == {}
    assert dm.mconcat([{"x": 2, "y": 1}, {"x": 5, "z": 3}]) == {"x": 7, "y": 1, "z": 3}

    pm = product_monoid(INT_SUM_MONOID, STR_CONCAT_MONOID)
    assert pm.empty() == (0, "")
    assert pm.combine((2, "a"), (3, "b")) == (5, "ab")

    endo_fn: Monoid[Endo[int]] = endo_monoid()
    composed = endo_fn.mconcat([Endo(lambda x: x + 1), Endo(lambda x: x * 2)])
    assert endo_fn.empty()(10) == 10
    assert composed(4) == 9


def test_semiring_ring_and_bounded_lattice() -> None:
    """Test Semiring, Ring, and BoundedLattice witnesses."""
    assert INT_SEMIRING.sum_all([1, 2, 3]) == 6
    assert INT_SEMIRING.product_all([2, 3, 5]) == 30
    assert INT_SEMIRING.pow_nat(3, 0) == 1
    assert INT_SEMIRING.pow_nat(2, 5) == 32
    assert INT_SEMIRING.pow_nat(3, 4) == 81
    with pytest.raises(ValueError, match="non-negative integer"):
        INT_SEMIRING.pow_nat(2, -1)
    with pytest.raises(ValueError, match="non-negative integer"):
        INT_SEMIRING.pow_nat(2, True)

    assert INT_RING.neg(7) == -7
    assert INT_RING.sub(10, 4) == 6
    custom_ring = Ring(semiring=INT_SEMIRING, neg=INT_RING.neg, sub=INT_RING.sub)
    assert custom_ring.semiring.one() == 1

    bool_eq = Eq[bool](eq=lambda a, b: a == b)
    bool_lattice = BoundedLattice[bool](
        eq_witness=bool_eq,
        bottom=lambda: False,
        top=lambda: True,
        join=lambda a, b: a or b,
        meet=lambda a, b: a and b,
    )
    assert bool_lattice.leq(False, True) is True
    assert bool_lattice.leq(True, False) is False
    assert bool_lattice.meet(bool_lattice.top(), False) is False
    assert bool_lattice.join(bool_lattice.bottom(), True) is True


def test_result_monad_combinators() -> None:
    """Test Result (Ok/Err) Functor, Bifunctor, Monad, and traversal helpers."""
    ok_v: Result[str, int] = Ok(10)
    err_v: Result[str, int] = Err("boom")

    assert ok_v.is_ok() is True
    assert ok_v.is_err() is False
    assert err_v.is_ok() is False
    assert err_v.is_err() is True

    assert fold_result(ok_v, lambda e: f"E:{e}", lambda v: f"V:{v}") == "V:10"
    assert fold_result(err_v, lambda e: f"E:{e}", lambda v: f"V:{v}") == "E:boom"

    assert map_result(ok_v, lambda x: x + 5) == Ok(15)
    assert map_result(err_v, lambda x: x + 5) == Err("boom")

    assert map_err(ok_v, str.upper) == Ok(10)
    assert map_err(err_v, str.upper) == Err("BOOM")

    assert and_then(ok_v, lambda x: Ok(x * 2) if x > 0 else Err("neg")) == Ok(20)
    assert and_then(err_v, lambda x: Ok(x * 2)) == Err("boom")

    assert value_or(ok_v, 0) == 10
    assert value_or(err_v, 0) == 0
    assert value_or_else(ok_v, len) == 10
    assert value_or_else(err_v, len) == 4

    ok_exc: Result[ValueError, str] = Ok("good")
    assert raise_on_err(ok_exc) == "good"
    with pytest.raises(ValueError, match="bad_input"):
        raise_on_err(Err(ValueError("bad_input")))

    def _parse_pos(n: int) -> Result[str, int]:
        """Validate positive integer.

        Args:
            n (int): Candidate integer.

        Returns:
            Result[str, int]: Ok(n) if positive else Err.
        """
        return Ok(n) if n > 0 else Err(f"non-pos:{n}")

    assert traverse_result([1, 2, 3], _parse_pos) == Ok((1, 2, 3))
    assert traverse_result([1, -2, -3], _parse_pos) == Err("non-pos:-2")
    assert partition_results((Ok(1), Err("e1"), Ok(2), Err("e2"))) == (("e1", "e2"), (1, 2))


def test_applicative_validation_combinators() -> None:
    """Test Validation (Valid/Invalid) applicative error accumulation and Result bridges."""
    err_monoid: Monoid[tuple[str, ...]] = tuple_monoid()
    err_sg = err_monoid.semigroup

    v_ok: Validation[tuple[str, ...], int] = Valid(7)
    v_bad1: Validation[tuple[str, ...], int] = Invalid(("e1",))
    v_bad2: Validation[tuple[str, ...], int] = Invalid(("e2",))

    assert v_ok.is_valid() is True
    assert v_ok.is_invalid() is False
    assert v_bad1.is_valid() is False
    assert v_bad1.is_invalid() is True

    assert fold_validation(v_ok, lambda es: ",".join(es), str) == "7"
    assert fold_validation(v_bad1, lambda es: ",".join(es), str) == "e1"

    assert map_validation(v_ok, lambda x: x + 1) == Valid(8)
    assert map_validation(v_bad1, lambda x: x + 1) == Invalid(("e1",))
    assert map_validation_err(v_ok, lambda es: tuple(e.upper() for e in es)) == Valid(7)
    assert map_validation_err(v_bad1, lambda es: tuple(e.upper() for e in es)) == Invalid(("E1",))

    assert ap_validation(err_sg, Valid(lambda x: x * 3), v_ok) == Valid(21)
    assert ap_validation(err_sg, Valid(lambda x: x * 3), v_bad1) == Invalid(("e1",))
    assert ap_validation(err_sg, Invalid(("ef",)), v_ok) == Invalid(("ef",))
    assert ap_validation(err_sg, Invalid(("ef",)), v_bad2) == Invalid(("ef", "e2"))

    assert zip_validation2(err_sg, v_ok, Valid(3), lambda a, b: a + b) == Valid(10)
    assert zip_validation2(err_sg, v_bad1, v_bad2, lambda a, b: a + b) == Invalid(("e1", "e2"))

    assert zip_validation3(err_sg, v_ok, Valid(2), Valid(1), lambda a, b, c: a + b + c) == Valid(10)
    assert zip_validation3(
        err_sg,
        v_bad1,
        v_bad2,
        Invalid(("e3",)),
        lambda a, b, c: a + b + c,
    ) == Invalid(("e1", "e2", "e3"))

    def _validate_pos(n: int) -> Validation[tuple[str, ...], int]:
        """Validate positive integer applicatively.

        Args:
            n (int): Candidate integer.

        Returns:
            Validation[tuple[str, ...], int]: Valid(n) or Invalid tuple.
        """
        return Valid(n) if n > 0 else Invalid((f"bad:{n}",))

    assert traverse_validation(err_sg, [1, 2, 3], _validate_pos) == Valid((1, 2, 3))
    assert traverse_validation(err_sg, [-1, 2, -3], _validate_pos) == Invalid(("bad:-1", "bad:-3"))

    assert collect_unit_validations(err_monoid, [Valid(None), Invalid(("u1",)), Invalid(("u2",))]) == ("u1", "u2")
    assert collect_unit_validations(err_monoid, [Valid(None), Valid(None)]) == ()

    assert result_to_validation(Ok(10)) == Valid(10)
    assert result_to_validation(Err("boom")) == Invalid("boom")
    assert validation_to_result(v_ok) == Ok(7)
    assert validation_to_result(v_bad1) == Err(("e1",))


@dataclass(frozen=True)
class _ToyTree:
    """Minimal binary expression tree for testing EndoRewrite and RecursiveFold.

    Attributes:
        val (int): Leaf or node integer payload.
        children (tuple[_ToyTree, ...]): Child subtrees.
    """

    val: int
    children: tuple[_ToyTree, ...] = ()


def test_codec_and_pretty_combinators() -> None:
    """Test Codec combinators, CodecError paths, Pretty, and PythonExprEmit."""

    def _dec_int(raw: JsonValue) -> Result[CodecError, int]:
        """Decode non-boolean JSON integer.

        Args:
            raw (JsonValue): JSON value.

        Returns:
            Result[CodecError, int]: Decoded int or CodecError.
        """
        if isinstance(raw, int) and not isinstance(raw, bool):
            return Ok(raw)
        return Err(CodecError(path=(), expected="int", actual_description=describe_value_type(raw), message="Not int"))

    int_codec: JsonCodec[int] = Codec(encode=lambda x: x, decode=_dec_int)
    str_wrapped_int: Codec[str, int] = compose_codec(
        Codec[str, JsonValue](
            encode=str,
            decode=lambda s: (
                Ok(int(s))
                if s.lstrip("-").isdigit()
                else Err(CodecError(path=(), expected="digit str", actual_description="str", message="Bad digits"))
            ),
        ),
        int_codec,
    )
    assert str_wrapped_int.encode(42) == "42"
    assert str_wrapped_int.decode("42") == Ok(42)
    assert str_wrapped_int.decode("nope").is_err()

    shifted = imap_codec(int_codec, lambda x: x + 1, lambda y: y - 1)
    assert shifted.encode(10) == 9
    assert shifted.decode(9) == Ok(10)
    assert shifted.decode("bad").is_err()

    pos_codec = refine_codec(
        int_codec,
        lambda x: (
            Ok(x)
            if x > 0
            else Err(CodecError(path=(), expected="> 0", actual_description="int", message="Non-positive"))
        ),
        lambda x: x,
    )
    assert pos_codec.encode(5) == 5
    assert pos_codec.decode(5) == Ok(5)
    assert pos_codec.decode(0).is_err()
    assert pos_codec.decode("bad").is_err()

    tup_c = tuple_codec(pos_codec)
    assert tup_c.encode((1, 2)) == [1, 2]
    assert tup_c.decode([1, 2]) == Ok((1, 2))
    bad_wires: tuple[JsonValue, ...] = (None, True, 5, 1.5, "not_list", {"k": 1})
    for bad_wire in bad_wires:
        assert tup_c.decode(bad_wire).is_err()
    bad_elem = tup_c.decode([1, 0])
    assert isinstance(bad_elem, Err)
    assert bad_elem.error.path == (1,)

    dict_c = dict_str_codec(list_codec(pos_codec))
    assert dict_c.encode({"a": [1, 2]}) == {"a": [1, 2]}
    assert dict_c.decode({"a": [1, 2]}) == Ok({"a": [1, 2]})
    assert dict_c.decode([1, 2]).is_err()
    nested_err = dict_c.decode({"items": [3, -1]})
    assert isinstance(nested_err, Err)
    assert nested_err.error.path == ("items", 1)
    assert "$.items[1]" in str(nested_err.error)

    opt_c = optional_codec(pos_codec)
    assert opt_c.encode(None) is None
    assert opt_c.encode(4) == 4
    assert opt_c.decode(None) == Ok(None)
    assert opt_c.decode(4) == Ok(4)
    assert opt_c.decode(-4).is_err()

    pretty_int = Pretty[int](render=lambda n: f"#{n}")
    py_emit_int = PythonExprEmit[int](emit_python=lambda n, pfx: f"{pfx}Const({n})")
    assert pretty_int.render(7) == "#7"
    assert py_emit_int.emit_python(7, "sw_ir.") == "sw_ir.Const(7)"


def test_recursion_schemes_endorewrite_and_fold() -> None:
    """Test EndoRewrite bottom_up, top_down, fixpoint, and RecursiveFold."""
    rewriter = EndoRewrite[_ToyTree](
        map_children=lambda node, fn: _ToyTree(node.val, tuple(fn(c) for c in node.children)),
    )
    tree_eq = Eq[_ToyTree](eq=lambda a, b: a == b)
    sample = _ToyTree(1, (_ToyTree(2), _ToyTree(3, (_ToyTree(4),))))

    bu = rewriter.bottom_up(sample, lambda n: _ToyTree(n.val + sum(c.val for c in n.children), n.children))
    assert bu.val == 1 + 2 + 7

    td = rewriter.top_down(sample, lambda n: _ToyTree(n.val * 10, n.children))
    assert td.val == 10
    assert td.children[0].val == 20

    stabilized = rewriter.fixpoint(
        _ToyTree(3),
        tree_eq,
        lambda n: _ToyTree(n.val - 1) if n.val > 0 else n,
        max_steps=10,
    )
    assert stabilized == _ToyTree(0)

    bounded = rewriter.fixpoint(
        _ToyTree(10),
        tree_eq,
        lambda n: _ToyTree(n.val - 1),
        max_steps=2,
    )
    assert bounded == _ToyTree(8)

    with pytest.raises(ValueError, match="non-negative int"):
        rewriter.fixpoint(sample, tree_eq, lambda n: n, max_steps=-1)
    with pytest.raises(ValueError, match="non-negative int"):
        rewriter.fixpoint(sample, tree_eq, lambda n: n, max_steps=True)

    summer = RecursiveFold[_ToyTree, int](
        fold_step=lambda n, rec: n.val + sum(rec(c) for c in n.children),
    )
    assert summer.fold(sample) == 10


def test_nonempty_and_nonempty_validation_combinators() -> None:
    """Test NonEmpty methods, witnesses (Semigroup/Eq/Ord/Hash/Pretty), and NonEmpty Validation helpers."""
    single = NonEmpty.singleton(10)
    assert len(single) == 1
    assert list(single) == [10]
    assert single.to_tuple() == (10,)
    assert NonEmpty.from_sequence(()) is None

    multi = NonEmpty.from_sequence([1, 2, 3])
    assert multi == NonEmpty(head=1, tail=(2, 3))
    assert len(multi) == 3
    assert list(multi) == [1, 2, 3]
    assert multi.map(lambda x: x * 10) == NonEmpty(head=10, tail=(20, 30))

    sg: Semigroup[NonEmpty[int]] = nonempty_semigroup()
    sg_typed = nonempty_semigroup(int)
    assert sg.combine(single, multi) == NonEmpty(head=10, tail=(1, 2, 3))
    assert sg_typed.combine(single, single) == NonEmpty(head=10, tail=(10,))

    ne_eq = nonempty_eq(INT_EQ)
    assert ne_eq.eq(multi, NonEmpty(1, (2, 3))) is True
    assert ne_eq.eq(multi, single) is False
    assert ne_eq.eq(multi, NonEmpty(9, (2, 3))) is False
    assert ne_eq.eq(multi, NonEmpty(1, (2, 9))) is False

    ne_ord = nonempty_ord(INT_ORD)
    assert ne_ord.compare(NonEmpty(1, (2,)), NonEmpty(2, (0,))) is Ordering.LT
    assert ne_ord.compare(NonEmpty(1, (5,)), NonEmpty(1, (3,))) is Ordering.GT
    assert ne_ord.compare(NonEmpty(1, (2,)), NonEmpty(1, (2, 3))) is Ordering.LT
    assert ne_ord.compare(NonEmpty(1, (2,)), NonEmpty(1, (2,))) is Ordering.EQ

    ne_hash = nonempty_hash(INT_HASH)
    assert ne_hash.hash_value(multi) == ne_hash.hash_value(NonEmpty(1, (2, 3)))

    ne_pretty = nonempty_pretty(Pretty[int](render=lambda n: f"e{n}"), separator=", ")
    assert ne_pretty.render(multi) == "e1, e2, e3"

    inv1 = invalid_one("bad1")
    assert inv1 == Invalid(NonEmpty("bad1"))
    assert from_error_sequence([], 42) == Valid(42)
    assert from_error_sequence(["e1", "e2"], 42) == Invalid(NonEmpty("e1", ("e2",)))

    def _check_pos_ne(n: int) -> Validation[NonEmpty[str], int]:
        """Validate positive integer returning NonEmpty error on failure.

        Args:
            n (int): Candidate integer.

        Returns:
            Validation[NonEmpty[str], int]: Valid(n) or singleton Invalid.
        """
        return Valid(n) if n > 0 else invalid_one(f"neg:{n}")

    assert traverse_validation_ne([1, 2], _check_pos_ne) == Valid((1, 2))
    assert traverse_validation_ne([-1, 2, -3], _check_pos_ne) == Invalid(NonEmpty("neg:-1", ("neg:-3",)))

    assert collect_unit_validations_ne((Valid(None), Valid(None))) == Valid(None)
    assert collect_unit_validations_ne((Valid(None), invalid_one("u1"), invalid_one("u2"))) == Invalid(
        NonEmpty("u1", ("u2",))
    )

    assert validation_ne_to_tuple(Valid(5)) == Valid(5)
    assert validation_ne_to_tuple(Invalid(NonEmpty("a", ("b",)))) == Invalid(("a", "b"))

    assert validation_tuple_to_ne(Valid(5), lambda: "fallback") == Valid(5)
    assert validation_tuple_to_ne(Invalid(("a", "b")), lambda: "fallback") == Invalid(NonEmpty("a", ("b",)))
    assert validation_tuple_to_ne(Invalid(()), lambda: "fallback") == Invalid(NonEmpty("fallback"))
