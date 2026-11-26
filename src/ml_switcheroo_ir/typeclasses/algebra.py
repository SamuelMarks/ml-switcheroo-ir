"""Fundamental algebraic typeclass witnesses (Eq, Ord, Hashable, Semigroup, Monoid, Semiring, Ring, Lattice)."""

from __future__ import annotations

import functools
import typing
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum

import typing_extensions

_T = typing.TypeVar("_T")
_U = typing.TypeVar("_U")
_A = typing.TypeVar("_A")
_B = typing.TypeVar("_B")
_K = typing.TypeVar("_K")
_V = typing.TypeVar("_V")

__all__ = [
    "BOOL_ALL_MONOID",
    "BOOL_ANY_MONOID",
    "BoundedLattice",
    "Endo",
    "Eq",
    "HashableWitness",
    "INT_EQ",
    "INT_HASH",
    "INT_ORD",
    "INT_PRODUCT_MONOID",
    "INT_RING",
    "INT_SEMIRING",
    "INT_SUM_MONOID",
    "Monoid",
    "NonEmpty",
    "Ord",
    "Ordering",
    "Ring",
    "STR_CONCAT_MONOID",
    "STR_EQ",
    "STR_HASH",
    "STR_ORD",
    "Semigroup",
    "Semiring",
    "contramap_eq",
    "contramap_hash",
    "contramap_ord",
    "dict_monoid",
    "dual_semigroup",
    "endo_monoid",
    "first_semigroup",
    "frozenset_union_monoid",
    "last_semigroup",
    "lexical_pair_ord",
    "max_semigroup",
    "min_semigroup",
    "nonempty_eq",
    "nonempty_hash",
    "nonempty_ord",
    "nonempty_semigroup",
    "pair_hash_witness",
    "product_monoid",
    "tuple_hash_witness",
    "tuple_monoid",
]


class Ordering(str, Enum):
    """Total comparison outcome enumeration corresponding to Haskell ``Data.Ord.Ordering``.

    Attributes:
        LT (Ordering): Left operand is strictly less than right operand.
        EQ (Ordering): Left and right operands are equivalent.
        GT (Ordering): Left operand is strictly greater than right operand.
    """

    LT = "LT"
    EQ = "EQ"
    GT = "GT"

    def flip(self) -> Ordering:
        """Invert the comparison direction (``LT <-> GT``, ``EQ -> EQ``).

        Returns:
            Ordering: Inverted ordering value.
        """
        if self is Ordering.LT:
            return Ordering.GT
        elif self is Ordering.GT:
            return Ordering.LT
        elif self is Ordering.EQ:
            return Ordering.EQ
        else:
            typing_extensions.assert_never(self)


@dataclass(frozen=True)
class Eq(typing.Generic[_T]):
    """Typeclass witness dictionary for equivalence relations over ``_T``.

    Laws:
        - Reflexivity: ``eq(x, x) is True``
        - Symmetry: ``eq(x, y) == eq(y, x)``
        - Transitivity: ``(eq(x, y) and eq(y, z)) ==> eq(x, z)``

    Attributes:
        eq (Callable[[_T, _T], bool]): Equivalence predicate returning True iff two values are equal.
    """

    eq: Callable[[_T, _T], bool]

    def neq(self, lhs: _T, rhs: _T) -> bool:
        """Return True when ``lhs`` and ``rhs`` are not equivalent under ``eq``.

        Args:
            lhs (_T): Left operand.
            rhs (_T): Right operand.

        Returns:
            bool: Negation of ``self.eq(lhs, rhs)``.
        """
        return not self.eq(lhs, rhs)


def contramap_eq(witness: Eq[_A], fn: Callable[[_B], _A]) -> Eq[_B]:
    """Pull an ``Eq[_A]`` witness back along a projection ``fn: _B -> _A``.

    Args:
        witness (Eq[_A]): Target equivalence witness.
        fn (Callable[[_B], _A]): Projection function mapping ``_B`` into ``_A``.

    Returns:
        Eq[_B]: Derived equivalence witness over ``_B``.
    """

    def _projected_eq(lhs: _B, rhs: _B) -> bool:
        """Compare projected values using ``witness``.

        Args:
            lhs (_B): Left input value.
            rhs (_B): Right input value.

        Returns:
            bool: True if projected values are equal.
        """
        return witness.eq(fn(lhs), fn(rhs))

    return Eq(eq=_projected_eq)


@dataclass(frozen=True)
class Ord(typing.Generic[_T]):
    """Typeclass witness dictionary for total orderings over ``_T``.

    Attributes:
        eq_witness (Eq[_T]): Underlying coherent equality witness.
        compare (Callable[[_T, _T], Ordering]): Total three-way comparison function.
    """

    eq_witness: Eq[_T]
    compare: Callable[[_T, _T], Ordering]

    def lt(self, lhs: _T, rhs: _T) -> bool:
        """Return True iff ``lhs < rhs``.

        Args:
            lhs (_T): Left operand.
            rhs (_T): Right operand.

        Returns:
            bool: True when ``compare(lhs, rhs) is Ordering.LT``.
        """
        return self.compare(lhs, rhs) is Ordering.LT

    def le(self, lhs: _T, rhs: _T) -> bool:
        """Return True iff ``lhs <= rhs``.

        Args:
            lhs (_T): Left operand.
            rhs (_T): Right operand.

        Returns:
            bool: True when ``compare(lhs, rhs)`` is ``LT`` or ``EQ``.
        """
        res = self.compare(lhs, rhs)
        return res is Ordering.LT or res is Ordering.EQ

    def gt(self, lhs: _T, rhs: _T) -> bool:
        """Return True iff ``lhs > rhs``.

        Args:
            lhs (_T): Left operand.
            rhs (_T): Right operand.

        Returns:
            bool: True when ``compare(lhs, rhs) is Ordering.GT``.
        """
        return self.compare(lhs, rhs) is Ordering.GT

    def ge(self, lhs: _T, rhs: _T) -> bool:
        """Return True iff ``lhs >= rhs``.

        Args:
            lhs (_T): Left operand.
            rhs (_T): Right operand.

        Returns:
            bool: True when ``compare(lhs, rhs)`` is ``GT`` or ``EQ``.
        """
        res = self.compare(lhs, rhs)
        return res is Ordering.GT or res is Ordering.EQ

    def min_val(self, lhs: _T, rhs: _T) -> _T:
        """Return the smaller of ``lhs`` and ``rhs`` (left-biased on ``EQ``).

        Args:
            lhs (_T): Left candidate.
            rhs (_T): Right candidate.

        Returns:
            _T: Minimum value under ``self``.
        """
        return lhs if self.le(lhs, rhs) else rhs

    def max_val(self, lhs: _T, rhs: _T) -> _T:
        """Return the larger of ``lhs`` and ``rhs`` (left-biased on ``EQ``).

        Args:
            lhs (_T): Left candidate.
            rhs (_T): Right candidate.

        Returns:
            _T: Maximum value under ``self``.
        """
        return lhs if self.ge(lhs, rhs) else rhs

    def clamp(self, val: _T, lower: _T, upper: _T) -> _T:
        """Clamp ``val`` into the closed interval ``[lower, upper]``.

        Args:
            val (_T): Input value to bound.
            lower (_T): Inclusive lower bound.
            upper (_T): Inclusive upper bound.

        Returns:
            _T: Clamped value.

        Raises:
            ValueError: If ``lower`` is strictly greater than ``upper``.
        """
        if self.gt(lower, upper):
            raise ValueError("Ord.clamp requires lower <= upper.")
        if self.lt(val, lower):
            return lower
        if self.gt(val, upper):
            return upper
        return val

    def sort_with(self, items: Iterable[_T]) -> list[_T]:
        """Deterministically sort ``items`` in ascending order according to ``compare``.

        Args:
            items (Iterable[_T]): Elements to sort.

        Returns:
            list[_T]: Sorted list of elements.
        """

        def _cmp_int(lhs: _T, rhs: _T) -> int:
            """Convert ``Ordering`` into ``functools.cmp_to_key`` integer sign.

            Args:
                lhs (_T): Left element.
                rhs (_T): Right element.

            Returns:
                int: -1, 0, or 1.
            """
            rel = self.compare(lhs, rhs)
            if rel is Ordering.LT:
                return -1
            elif rel is Ordering.EQ:
                return 0
            elif rel is Ordering.GT:
                return 1
            else:
                typing_extensions.assert_never(rel)

        return sorted(items, key=functools.cmp_to_key(_cmp_int))


def contramap_ord(witness: Ord[_A], fn: Callable[[_B], _A]) -> Ord[_B]:
    """Pull an ``Ord[_A]`` witness back along a projection ``fn: _B -> _A``.

    Args:
        witness (Ord[_A]): Target total order witness.
        fn (Callable[[_B], _A]): Projection mapping ``_B`` into ``_A``.

    Returns:
        Ord[_B]: Derived total ordering over ``_B``.
    """

    def _projected_compare(lhs: _B, rhs: _B) -> Ordering:
        """Compare projected values via ``witness.compare``.

        Args:
            lhs (_B): Left value.
            rhs (_B): Right value.

        Returns:
            Ordering: Three-way comparison result.
        """
        return witness.compare(fn(lhs), fn(rhs))

    return Ord(eq_witness=contramap_eq(witness.eq_witness, fn), compare=_projected_compare)


def lexical_pair_ord(fst_ord: Ord[_A], snd_ord: Ord[_B]) -> Ord[tuple[_A, _B]]:
    """Construct lexicographic ``Ord`` witness over 2-tuples ``tuple[_A, _B]``.

    Args:
        fst_ord (Ord[_A]): Ordering witness for first tuple element.
        snd_ord (Ord[_B]): Ordering witness for second tuple element.

    Returns:
        Ord[tuple[_A, _B]]: Lexicographic ordering witness.
    """

    def _pair_eq(lhs: tuple[_A, _B], rhs: tuple[_A, _B]) -> bool:
        """Check componentwise equality of pairs.

        Args:
            lhs (tuple[_A, _B]): Left pair.
            rhs (tuple[_A, _B]): Right pair.

        Returns:
            bool: True if both components are equal.
        """
        return fst_ord.eq_witness.eq(lhs[0], rhs[0]) and snd_ord.eq_witness.eq(lhs[1], rhs[1])

    def _pair_cmp(lhs: tuple[_A, _B], rhs: tuple[_A, _B]) -> Ordering:
        """Compare pairs lexicographically.

        Args:
            lhs (tuple[_A, _B]): Left pair.
            rhs (tuple[_A, _B]): Right pair.

        Returns:
            Ordering: Lexicographic comparison outcome.
        """
        first_rel = fst_ord.compare(lhs[0], rhs[0])
        if first_rel is not Ordering.EQ:
            return first_rel
        return snd_ord.compare(lhs[1], rhs[1])

    return Ord(eq_witness=Eq(eq=_pair_eq), compare=_pair_cmp)


@dataclass(frozen=True)
class HashableWitness(typing.Generic[_T]):
    """Typeclass witness dictionary pairing coherent equality and deterministic integer hashing.

    Law (Hash-Equality Coherence):
        ``eq_witness.eq(x, y) ==> hash_value(x) == hash_value(y)``

    Attributes:
        eq_witness (Eq[_T]): Underlying equality witness.
        hash_value (Callable[[_T], int]): Deterministic hash function coherent with ``eq_witness``.
    """

    eq_witness: Eq[_T]
    hash_value: Callable[[_T], int]


def contramap_hash(witness: HashableWitness[_A], fn: Callable[[_B], _A]) -> HashableWitness[_B]:
    """Pull a ``HashableWitness[_A]`` back along ``fn: _B -> _A``.

    Args:
        witness (HashableWitness[_A]): Target hash witness.
        fn (Callable[[_B], _A]): Projection function.

    Returns:
        HashableWitness[_B]: Derived coherent hash witness over ``_B``.
    """

    def _projected_hash(val: _B) -> int:
        """Hash projected value via ``witness.hash_value``.

        Args:
            val (_B): Input value.

        Returns:
            int: Integer hash code.
        """
        return witness.hash_value(fn(val))

    return HashableWitness(eq_witness=contramap_eq(witness.eq_witness, fn), hash_value=_projected_hash)


def tuple_hash_witness(elem_witness: HashableWitness[_T]) -> HashableWitness[tuple[_T, ...]]:
    """Construct a ``HashableWitness`` for homogeneous variable-length tuples.

    Args:
        elem_witness (HashableWitness[_T]): Element hashable witness.

    Returns:
        HashableWitness[tuple[_T, ...]]: Coherent tuple hashable witness.
    """

    def _tuple_eq(lhs: tuple[_T, ...], rhs: tuple[_T, ...]) -> bool:
        """Compare two tuples elementwise.

        Args:
            lhs (tuple[_T, ...]): Left tuple.
            rhs (tuple[_T, ...]): Right tuple.

        Returns:
            bool: True iff lengths match and all elements are equivalent.
        """
        if len(lhs) != len(rhs):
            return False
        return all(elem_witness.eq_witness.eq(a, b) for a, b in zip(lhs, rhs))

    def _tuple_hash(items: tuple[_T, ...]) -> int:
        """Hash tuple from element hashes.

        Args:
            items (tuple[_T, ...]): Tuple of elements.

        Returns:
            int: Combined tuple hash.
        """
        return hash(tuple(elem_witness.hash_value(x) for x in items))

    return HashableWitness(eq_witness=Eq(eq=_tuple_eq), hash_value=_tuple_hash)


def pair_hash_witness(
    fst_witness: HashableWitness[_A], snd_witness: HashableWitness[_B]
) -> HashableWitness[tuple[_A, _B]]:
    """Construct a ``HashableWitness`` for heterogeneous 2-tuples ``tuple[_A, _B]``.

    Args:
        fst_witness (HashableWitness[_A]): First element hash witness.
        snd_witness (HashableWitness[_B]): Second element hash witness.

    Returns:
        HashableWitness[tuple[_A, _B]]: Coherent 2-tuple hash witness.
    """

    def _pair_eq(lhs: tuple[_A, _B], rhs: tuple[_A, _B]) -> bool:
        """Check componentwise equality of pairs.

        Args:
            lhs (tuple[_A, _B]): Left pair.
            rhs (tuple[_A, _B]): Right pair.

        Returns:
            bool: True iff both elements match.
        """
        return fst_witness.eq_witness.eq(lhs[0], rhs[0]) and snd_witness.eq_witness.eq(lhs[1], rhs[1])

    def _pair_hash(val: tuple[_A, _B]) -> int:
        """Compute combined hash of pair components.

        Args:
            val (tuple[_A, _B]): Input pair.

        Returns:
            int: Integer hash value.
        """
        return hash((fst_witness.hash_value(val[0]), snd_witness.hash_value(val[1])))

    return HashableWitness(eq_witness=Eq(eq=_pair_eq), hash_value=_pair_hash)


@dataclass(frozen=True)
class Semigroup(typing.Generic[_T]):
    """Typeclass witness dictionary for associative binary operations over ``_T``.

    Law (Associativity):
        ``combine(combine(a, b), c) == combine(a, combine(b, c))``

    Attributes:
        combine (Callable[[_T, _T], _T]): Associative binary combination function.
    """

    combine: Callable[[_T, _T], _T]

    def sconcat(self, head: _T, tail: Iterable[_T]) -> _T:
        """Fold a non-empty sequence ``(head, *tail)`` left-to-right via ``combine``.

        Args:
            head (_T): Initial guaranteed element.
            tail (Iterable[_T]): Remaining elements to combine.

        Returns:
            _T: Associatively combined value.
        """
        acc = head
        for item in tail:
            acc = self.combine(acc, item)
        return acc


def first_semigroup() -> Semigroup[_T]:
    """Construct a left-biased ``Semigroup`` keeping the first value (Haskell ``Data.Semigroup.First``).

    Returns:
        Semigroup[_T]: Left-absorbing semigroup witness.
    """

    def _keep_first(lhs: _T, _rhs: _T) -> _T:
        """Return ``lhs``.

        Args:
            lhs (_T): Left value.
            _rhs (_T): Ignored right value.

        Returns:
            _T: ``lhs``.
        """
        return lhs

    return Semigroup(combine=_keep_first)


def last_semigroup() -> Semigroup[_T]:
    """Construct a right-biased ``Semigroup`` keeping the last value (Haskell ``Data.Semigroup.Last``).

    Returns:
        Semigroup[_T]: Right-absorbing semigroup witness.
    """

    def _keep_last(_lhs: _T, rhs: _T) -> _T:
        """Return ``rhs``.

        Args:
            _lhs (_T): Ignored left value.
            rhs (_T): Right value.

        Returns:
            _T: ``rhs``.
        """
        return rhs

    return Semigroup(combine=_keep_last)


def dual_semigroup(sg: Semigroup[_T]) -> Semigroup[_T]:
    """Flip the operand order of a ``Semigroup`` (Haskell ``Data.Semigroup.Dual``).

    Args:
        sg (Semigroup[_T]): Base semigroup witness.

    Returns:
        Semigroup[_T]: Dual semigroup witness with ``combine(a, b) = sg.combine(b, a)``.
    """

    def _flipped(lhs: _T, rhs: _T) -> _T:
        """Combine in reversed operand order.

        Args:
            lhs (_T): Left value.
            rhs (_T): Right value.

        Returns:
            _T: ``sg.combine(rhs, lhs)``.
        """
        return sg.combine(rhs, lhs)

    return Semigroup(combine=_flipped)


def max_semigroup(ord_witness: Ord[_T]) -> Semigroup[_T]:
    """Construct a ``Semigroup`` taking the maximum under ``ord_witness``.

    Args:
        ord_witness (Ord[_T]): Total ordering witness.

    Returns:
        Semigroup[_T]: Maximum-selecting semigroup witness.
    """
    return Semigroup(combine=ord_witness.max_val)


def min_semigroup(ord_witness: Ord[_T]) -> Semigroup[_T]:
    """Construct a ``Semigroup`` taking the minimum under ``ord_witness``.

    Args:
        ord_witness (Ord[_T]): Total ordering witness.

    Returns:
        Semigroup[_T]: Minimum-selecting semigroup witness.
    """
    return Semigroup(combine=ord_witness.min_val)


@dataclass(frozen=True)
class NonEmpty(typing.Generic[_T]):
    """Guaranteed non-empty immutable sequence corresponding to Haskell ``Data.List.NonEmpty``.

    Makes empty error collections such as ``Invalid(())`` statically unrepresentable.

    Attributes:
        head (_T): Guaranteed first element.
        tail (tuple[_T, ...]): Zero or more trailing elements.
    """

    head: _T
    tail: tuple[_T, ...] = ()

    def __iter__(self) -> Iterator[_T]:
        """Yield ``head`` followed by each element of ``tail``.

        Yields:
            _T: Elements of the non-empty sequence in order.
        """
        yield self.head
        yield from self.tail

    def __len__(self) -> int:
        """Return the positive element count ``1 + len(self.tail)``.

        Returns:
            int: Strictly positive length (>= 1).
        """
        return 1 + len(self.tail)

    def to_tuple(self) -> tuple[_T, ...]:
        """Materialize ``(self.head, *self.tail)`` as a standard homogeneous tuple.

        Returns:
            tuple[_T, ...]: Non-empty tuple representation.
        """
        return (self.head, *self.tail)

    @classmethod
    def singleton(cls, item: _T) -> NonEmpty[_T]:
        """Construct a single-element ``NonEmpty`` sequence.

        Args:
            item (_T): Sole element.

        Returns:
            NonEmpty[_T]: ``NonEmpty(head=item, tail=())``.
        """
        return cls(head=item, tail=())

    @classmethod
    def from_sequence(cls, items: Sequence[_T]) -> NonEmpty[_T] | None:
        """Construct ``NonEmpty[_T]`` from ``items`` or return ``None`` when ``items`` is empty.

        Args:
            items (Sequence[_T]): Candidate sequence of elements.

        Returns:
            NonEmpty[_T] | None: Non-empty wrapper if ``len(items) >= 1``, else ``None``.
        """
        if not items:
            return None
        return cls(head=items[0], tail=tuple(items[1:]))

    def map(self, fn: Callable[[_T], _U]) -> NonEmpty[_U]:
        """Covariantly map ``fn`` across ``head`` and ``tail`` (``Functor`` on ``NonEmpty``).

        Args:
            fn (Callable[[_T], _U]): Pure element transformation function.

        Returns:
            NonEmpty[_U]: Mapped non-empty sequence of identical length.
        """
        return NonEmpty(head=fn(self.head), tail=tuple(fn(x) for x in self.tail))


@typing.overload
def nonempty_semigroup() -> Semigroup[NonEmpty[_T]]:
    """Construct the free ``Semigroup`` over ``NonEmpty[_T]`` via contextual type inference.

    Returns:
        Semigroup[NonEmpty[_T]]: Associative non-empty concatenation semigroup witness.
    """


@typing.overload
def nonempty_semigroup(dummy_type: type[_T]) -> Semigroup[NonEmpty[_T]]:
    """Construct the free ``Semigroup`` over ``NonEmpty[_T]`` parameterized by ``dummy_type``.

    Args:
        dummy_type (type[_T]): Phantom element type token guiding static type inference.

    Returns:
        Semigroup[NonEmpty[_T]]: Associative non-empty concatenation semigroup witness.
    """


def nonempty_semigroup(dummy_type: type[_T] | None = None) -> Semigroup[NonEmpty[_T]]:
    """Construct the free ``Semigroup`` over ``NonEmpty[_T]`` via concatenation.

    Args:
        dummy_type (type[_T] | None): Optional phantom element type token guiding static type inference.

    Returns:
        Semigroup[NonEmpty[_T]]: Associative non-empty concatenation semigroup witness.
    """
    del dummy_type

    def _concat_nonempty(lhs: NonEmpty[_T], rhs: NonEmpty[_T]) -> NonEmpty[_T]:
        """Concatenate two ``NonEmpty`` sequences preserving order.

        Args:
            lhs (NonEmpty[_T]): Left non-empty sequence.
            rhs (NonEmpty[_T]): Right non-empty sequence.

        Returns:
            NonEmpty[_T]: Combined non-empty sequence.
        """
        return NonEmpty(head=lhs.head, tail=(*lhs.tail, rhs.head, *rhs.tail))

    return Semigroup(combine=_concat_nonempty)


def nonempty_eq(elem_eq: Eq[_T]) -> Eq[NonEmpty[_T]]:
    """Lift an element ``Eq[_T]`` witness to ``Eq[NonEmpty[_T]]``.

    Args:
        elem_eq (Eq[_T]): Equivalence witness for individual elements.

    Returns:
        Eq[NonEmpty[_T]]: Elementwise equivalence witness over ``NonEmpty[_T]``.
    """

    def _eq_ne(lhs: NonEmpty[_T], rhs: NonEmpty[_T]) -> bool:
        """Compare two ``NonEmpty`` sequences elementwise.

        Args:
            lhs (NonEmpty[_T]): Left sequence.
            rhs (NonEmpty[_T]): Right sequence.

        Returns:
            bool: True iff lengths match and every corresponding element pair is equal.
        """
        if len(lhs.tail) != len(rhs.tail):
            return False
        if not elem_eq.eq(lhs.head, rhs.head):
            return False
        return all(elem_eq.eq(a, b) for a, b in zip(lhs.tail, rhs.tail))

    return Eq(eq=_eq_ne)


def nonempty_ord(elem_ord: Ord[_T]) -> Ord[NonEmpty[_T]]:
    """Lift an element ``Ord[_T]`` witness to lexicographic ``Ord[NonEmpty[_T]]``.

    Args:
        elem_ord (Ord[_T]): Total order witness for individual elements.

    Returns:
        Ord[NonEmpty[_T]]: Lexicographic total order witness over ``NonEmpty[_T]``.
    """

    def _cmp_ne(lhs: NonEmpty[_T], rhs: NonEmpty[_T]) -> Ordering:
        """Compare two ``NonEmpty`` sequences lexicographically.

        Args:
            lhs (NonEmpty[_T]): Left sequence.
            rhs (NonEmpty[_T]): Right sequence.

        Returns:
            Ordering: Lexicographic comparison outcome.
        """
        head_rel = elem_ord.compare(lhs.head, rhs.head)
        if head_rel is not Ordering.EQ:
            return head_rel
        for a, b in zip(lhs.tail, rhs.tail):
            step = elem_ord.compare(a, b)
            if step is not Ordering.EQ:
                return step
        return INT_ORD.compare(len(lhs.tail), len(rhs.tail))

    return Ord(eq_witness=nonempty_eq(elem_ord.eq_witness), compare=_cmp_ne)


def nonempty_hash(elem_hash: HashableWitness[_T]) -> HashableWitness[NonEmpty[_T]]:
    """Lift an element ``HashableWitness[_T]`` to ``HashableWitness[NonEmpty[_T]]``.

    Args:
        elem_hash (HashableWitness[_T]): Hashable witness for individual elements.

    Returns:
        HashableWitness[NonEmpty[_T]]: Coherent hashable witness over ``NonEmpty[_T]``.
    """
    tup_hash = tuple_hash_witness(elem_hash)
    return HashableWitness(
        eq_witness=nonempty_eq(elem_hash.eq_witness),
        hash_value=lambda ne: tup_hash.hash_value(ne.to_tuple()),
    )


@dataclass(frozen=True)
class Monoid(typing.Generic[_T]):
    """Typeclass witness dictionary extending ``Semigroup[_T]`` with a neutral identity thunk ``empty``.

    Laws:
        - Associativity: ``combine(combine(a, b), c) == combine(a, combine(b, c))``
        - Left Identity: ``combine(empty(), a) == a``
        - Right Identity: ``combine(a, empty()) == a``

    Attributes:
        empty (Callable[[], _T]): Thunk returning a fresh neutral identity element.
        semigroup (Semigroup[_T]): Underlying associative semigroup witness.
    """

    empty: Callable[[], _T]
    semigroup: Semigroup[_T]

    def combine(self, lhs: _T, rhs: _T) -> _T:
        """Combine ``lhs`` and ``rhs`` via ``self.semigroup.combine``.

        Args:
            lhs (_T): Left operand.
            rhs (_T): Right operand.

        Returns:
            _T: Combined value.
        """
        return self.semigroup.combine(lhs, rhs)

    def mconcat(self, items: Iterable[_T]) -> _T:
        """Fold a potentially empty iterable left-to-right starting from ``self.empty()``.

        Args:
            items (Iterable[_T]): Sequence of monoidal values.

        Returns:
            _T: Accumulated monoid value.
        """
        acc = self.empty()
        for item in items:
            acc = self.semigroup.combine(acc, item)
        return acc


def tuple_monoid() -> Monoid[tuple[_T, ...]]:
    """Construct the free concatenation ``Monoid`` over immutable tuples ``tuple[_T, ...]``.

    Returns:
        Monoid[tuple[_T, ...]]: Tuple concatenation monoid witness.
    """

    def _empty_tuple() -> tuple[_T, ...]:
        """Return empty tuple identity.

        Returns:
            tuple[_T, ...]: Empty tuple ``()``.
        """
        return ()

    def _concat_tuples(lhs: tuple[_T, ...], rhs: tuple[_T, ...]) -> tuple[_T, ...]:
        """Concatenate two tuples.

        Args:
            lhs (tuple[_T, ...]): Left tuple.
            rhs (tuple[_T, ...]): Right tuple.

        Returns:
            tuple[_T, ...]: Concatenated tuple.
        """
        return lhs + rhs

    return Monoid(empty=_empty_tuple, semigroup=Semigroup(combine=_concat_tuples))


def frozenset_union_monoid() -> Monoid[frozenset[_T]]:
    """Construct the set-union ``Monoid`` over ``frozenset[_T]``.

    Returns:
        Monoid[frozenset[_T]]: Frozenset union monoid witness.
    """

    def _empty_frozenset() -> frozenset[_T]:
        """Return empty frozenset identity.

        Returns:
            frozenset[_T]: Empty frozenset.
        """
        return frozenset()

    def _union_sets(lhs: frozenset[_T], rhs: frozenset[_T]) -> frozenset[_T]:
        """Compute union of two frozensets.

        Args:
            lhs (frozenset[_T]): Left set.
            rhs (frozenset[_T]): Right set.

        Returns:
            frozenset[_T]: Set union.
        """
        return lhs | rhs

    return Monoid(empty=_empty_frozenset, semigroup=Semigroup(combine=_union_sets))


def dict_monoid(val_semigroup: Semigroup[_V]) -> Monoid[dict[_K, _V]]:
    """Construct ``Monoid[dict[_K, _V]]`` merging colliding keys via ``val_semigroup`` (Haskell ``Map`` monoid).

    Args:
        val_semigroup (Semigroup[_V]): Semigroup used to combine values when a key exists in both maps.

    Returns:
        Monoid[dict[_K, _V]]: Fresh-copy map-merging monoid witness.
    """

    def _empty_dict() -> dict[_K, _V]:
        """Return a fresh empty dictionary.

        Returns:
            dict[_K, _V]: Empty dictionary.
        """
        return {}

    def _merge_dicts(lhs: dict[_K, _V], rhs: dict[_K, _V]) -> dict[_K, _V]:
        """Purely merge two dictionaries using ``val_semigroup`` on duplicate keys.

        Args:
            lhs (dict[_K, _V]): Left dictionary.
            rhs (dict[_K, _V]): Right dictionary.

        Returns:
            dict[_K, _V]: Merged dictionary without mutating inputs.
        """
        merged: dict[_K, _V] = dict(lhs)
        for k, v in rhs.items():
            if k in merged:
                merged[k] = val_semigroup.combine(merged[k], v)
            else:
                merged[k] = v
        return merged

    return Monoid(empty=_empty_dict, semigroup=Semigroup(combine=_merge_dicts))


def product_monoid(fst: Monoid[_A], snd: Monoid[_B]) -> Monoid[tuple[_A, _B]]:
    """Construct the Cartesian product ``Monoid`` over ``tuple[_A, _B]``.

    Args:
        fst (Monoid[_A]): Monoid for first tuple component.
        snd (Monoid[_B]): Monoid for second tuple component.

    Returns:
        Monoid[tuple[_A, _B]]: Componentwise product monoid witness.
    """

    def _empty_pair() -> tuple[_A, _B]:
        """Return componentwise neutral pair.

        Returns:
            tuple[_A, _B]: ``(fst.empty(), snd.empty())``.
        """
        return (fst.empty(), snd.empty())

    def _combine_pairs(lhs: tuple[_A, _B], rhs: tuple[_A, _B]) -> tuple[_A, _B]:
        """Combine pairs componentwise.

        Args:
            lhs (tuple[_A, _B]): Left pair.
            rhs (tuple[_A, _B]): Right pair.

        Returns:
            tuple[_A, _B]: Combined pair.
        """
        return (fst.combine(lhs[0], rhs[0]), snd.combine(lhs[1], rhs[1]))

    return Monoid(empty=_empty_pair, semigroup=Semigroup(combine=_combine_pairs))


@dataclass(frozen=True)
class Endo(typing.Generic[_T]):
    """Endomorphism wrapper corresponding to Haskell ``Data.Monoid.Endo``.

    Attributes:
        app (Callable[[_T], _T]): Wrapped endomorphism ``_T -> _T``.
    """

    app: Callable[[_T], _T]

    def __call__(self, val: _T) -> _T:
        """Apply the wrapped endomorphism ``self.app`` to ``val``.

        Args:
            val (_T): Input value.

        Returns:
            _T: Transformed output value.
        """
        return self.app(val)


def endo_monoid() -> Monoid[Endo[_T]]:
    """Construct the endomorphism composition ``Monoid`` (Haskell ``Data.Monoid.Endo``).

    ``combine(Endo(f), Endo(g))(x)`` applies ``f(g(x))`` (mathematical composition ``f . g``),
    with neutral identity ``Endo(lambda x: x)``.

    Returns:
        Monoid[Endo[_T]]: Endomorphism monoid witness.
    """

    def _identity(val: _T) -> _T:
        """Identity endomorphism.

        Args:
            val (_T): Input value.

        Returns:
            _T: Unchanged input value.
        """
        return val

    def _empty_endo() -> Endo[_T]:
        """Return identity ``Endo`` wrapper.

        Returns:
            Endo[_T]: Identity endomorphism wrapper.
        """
        return Endo(app=_identity)

    def _compose_endo(outer: Endo[_T], inner: Endo[_T]) -> Endo[_T]:
        """Compose two ``Endo`` wrappers as ``outer(inner(x))``.

        Args:
            outer (Endo[_T]): Outer endomorphism.
            inner (Endo[_T]): Inner endomorphism.

        Returns:
            Endo[_T]: Composed endomorphism wrapper.
        """

        def _composed(val: _T) -> _T:
            """Apply ``inner`` then ``outer``.

            Args:
                val (_T): Input value.

            Returns:
                _T: Transformed value.
            """
            return outer.app(inner.app(val))

        return Endo(app=_composed)

    return Monoid(empty=_empty_endo, semigroup=Semigroup(combine=_compose_endo))


@dataclass(frozen=True)
class Semiring(typing.Generic[_T]):
    """Typeclass witness dictionary for commutative additive monoids with distributive multiplication.

    Laws:
        - ``(add, zero)`` is a commutative ``Monoid``
        - ``(mul, one)`` is a ``Monoid``
        - Left/right distributivity: ``mul(a, add(b, c)) == add(mul(a, b), mul(a, c))``
          and ``mul(add(a, b), c) == add(mul(a, c), mul(b, c))``
        - Annihilation: ``mul(zero(), a) == zero() == mul(a, zero())``

    Attributes:
        zero (Callable[[], _T]): Additive identity factory.
        one (Callable[[], _T]): Multiplicative identity factory.
        add (Callable[[_T, _T], _T]): Commutative, associative addition function.
        mul (Callable[[_T, _T], _T]): Associative multiplication function distributing over ``add``.
    """

    zero: Callable[[], _T]
    one: Callable[[], _T]
    add: Callable[[_T, _T], _T]
    mul: Callable[[_T, _T], _T]

    def sum_all(self, items: Iterable[_T]) -> _T:
        """Fold ``items`` via ``add`` starting from ``zero()``.

        Args:
            items (Iterable[_T]): Elements to sum.

        Returns:
            _T: Semiring sum.
        """
        acc = self.zero()
        for item in items:
            acc = self.add(acc, item)
        return acc

    def product_all(self, items: Iterable[_T]) -> _T:
        """Fold ``items`` via ``mul`` starting from ``one()``.

        Args:
            items (Iterable[_T]): Elements to multiply.

        Returns:
            _T: Semiring product.
        """
        acc = self.one()
        for item in items:
            acc = self.mul(acc, item)
        return acc

    def pow_nat(self, base: _T, exp: int) -> _T:
        """Compute ``base ** exp`` for a non-negative integer ``exp`` via binary exponentiation.

        Args:
            base (_T): Semiring base value.
            exp (int): Non-negative non-boolean integer exponent.

        Returns:
            _T: ``base`` raised to ``exp``.

        Raises:
            ValueError: If ``exp`` is a boolean or negative integer.
        """
        if isinstance(exp, bool) or exp < 0:
            raise ValueError(f"Semiring.pow_nat requires a non-negative integer exponent, got {exp!r}.")
        result = self.one()
        curr = base
        n = exp
        while n > 0:
            if n & 1:
                result = self.mul(result, curr)
            n >>= 1
            if n > 0:
                curr = self.mul(curr, curr)
        return result


@dataclass(frozen=True)
class Ring(typing.Generic[_T]):
    """Typeclass witness dictionary extending ``Semiring[_T]`` with additive inverses.

    Laws:
        - All ``Semiring[_T]`` laws on ``semiring``
        - Additive inverse: ``semiring.add(a, neg(a)) == semiring.zero() == semiring.add(neg(a), a)``
        - Subtraction coherence: ``sub(a, b) == semiring.add(a, neg(b))``

    Attributes:
        semiring (Semiring[_T]): Underlying semiring witness.
        neg (Callable[[_T], _T]): Unary additive negation.
        sub (Callable[[_T, _T], _T]): Binary subtraction operation.
    """

    semiring: Semiring[_T]
    neg: Callable[[_T], _T]
    sub: Callable[[_T, _T], _T]


@dataclass(frozen=True)
class BoundedLattice(typing.Generic[_T]):
    """Typeclass witness dictionary for bounded join/meet lattices (abstract interpretation & severity).

    Laws:
        - Idempotence, commutativity, and associativity of ``join`` and ``meet``
        - Absorption: ``join(a, meet(a, b)) == a`` and ``meet(a, join(a, b)) == a``
        - Bounds: ``join(bottom(), a) == a`` and ``meet(top(), a) == a``

    Attributes:
        eq_witness (Eq[_T]): Underlying equality witness.
        bottom (Callable[[], _T]): Least element factory.
        top (Callable[[], _T]): Greatest element factory.
        join (Callable[[_T, _T], _T]): Least upper bound (supremum) operation.
        meet (Callable[[_T, _T], _T]): Greatest lower bound (infimum) operation.
    """

    eq_witness: Eq[_T]
    bottom: Callable[[], _T]
    top: Callable[[], _T]
    join: Callable[[_T, _T], _T]
    meet: Callable[[_T, _T], _T]

    def leq(self, lhs: _T, rhs: _T) -> bool:
        """Return True iff ``lhs <= rhs`` in the induced lattice partial order ``join(lhs, rhs) == rhs``.

        Args:
            lhs (_T): Lower candidate.
            rhs (_T): Upper candidate.

        Returns:
            bool: True if ``join(lhs, rhs)`` equals ``rhs``.
        """
        return self.eq_witness.eq(self.join(lhs, rhs), rhs)


def _int_eq(lhs: int, rhs: int) -> bool:
    """Compare two integers for equality.

    Args:
        lhs (int): Left integer.
        rhs (int): Right integer.

    Returns:
        bool: True iff ``lhs == rhs``.
    """
    return lhs == rhs


def _int_cmp(lhs: int, rhs: int) -> Ordering:
    """Three-way compare two integers.

    Args:
        lhs (int): Left integer.
        rhs (int): Right integer.

    Returns:
        Ordering: ``LT``, ``EQ``, or ``GT``.
    """
    if lhs < rhs:
        return Ordering.LT
    if lhs > rhs:
        return Ordering.GT
    return Ordering.EQ


def _str_eq(lhs: str, rhs: str) -> bool:
    """Compare two strings for equality.

    Args:
        lhs (str): Left string.
        rhs (str): Right string.

    Returns:
        bool: True iff ``lhs == rhs``.
    """
    return lhs == rhs


def _str_cmp(lhs: str, rhs: str) -> Ordering:
    """Three-way compare two strings lexicographically.

    Args:
        lhs (str): Left string.
        rhs (str): Right string.

    Returns:
        Ordering: ``LT``, ``EQ``, or ``GT``.
    """
    if lhs < rhs:
        return Ordering.LT
    if lhs > rhs:
        return Ordering.GT
    return Ordering.EQ


INT_EQ: typing.Final[Eq[int]] = Eq(eq=_int_eq)
INT_ORD: typing.Final[Ord[int]] = Ord(eq_witness=INT_EQ, compare=_int_cmp)
INT_HASH: typing.Final[HashableWitness[int]] = HashableWitness(eq_witness=INT_EQ, hash_value=hash)

STR_EQ: typing.Final[Eq[str]] = Eq(eq=_str_eq)
STR_ORD: typing.Final[Ord[str]] = Ord(eq_witness=STR_EQ, compare=_str_cmp)
STR_HASH: typing.Final[HashableWitness[str]] = HashableWitness(eq_witness=STR_EQ, hash_value=hash)

INT_SUM_MONOID: typing.Final[Monoid[int]] = Monoid(
    empty=lambda: 0,
    semigroup=Semigroup(combine=lambda a, b: a + b),
)
INT_PRODUCT_MONOID: typing.Final[Monoid[int]] = Monoid(
    empty=lambda: 1,
    semigroup=Semigroup(combine=lambda a, b: a * b),
)
BOOL_ALL_MONOID: typing.Final[Monoid[bool]] = Monoid(
    empty=lambda: True,
    semigroup=Semigroup(combine=lambda a, b: a and b),
)
BOOL_ANY_MONOID: typing.Final[Monoid[bool]] = Monoid(
    empty=lambda: False,
    semigroup=Semigroup(combine=lambda a, b: a or b),
)
STR_CONCAT_MONOID: typing.Final[Monoid[str]] = Monoid(
    empty=lambda: "",
    semigroup=Semigroup(combine=lambda a, b: a + b),
)
INT_SEMIRING: typing.Final[Semiring[int]] = Semiring(
    zero=lambda: 0,
    one=lambda: 1,
    add=lambda a, b: a + b,
    mul=lambda a, b: a * b,
)
INT_RING: typing.Final[Ring[int]] = Ring(
    semiring=INT_SEMIRING,
    neg=lambda a: -a,
    sub=lambda a, b: a - b,
)
