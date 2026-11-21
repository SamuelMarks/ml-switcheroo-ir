"""Closed total outcome ADTs: monadic Result (Either) and applicative error-accumulating Validation."""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import typing_extensions

from ml_switcheroo_ir.typeclasses.algebra import Monoid, Semigroup

_A = typing.TypeVar("_A")
_B = typing.TypeVar("_B")
_C = typing.TypeVar("_C")
_D = typing.TypeVar("_D")
_E = typing.TypeVar("_E")
_F = typing.TypeVar("_F")
_R = typing.TypeVar("_R")
_A_co = typing.TypeVar("_A_co", covariant=True)
_E_co = typing.TypeVar("_E_co", covariant=True)
_ExcT = typing.TypeVar("_ExcT", bound=Exception)

__all__ = [
    "Err",
    "Invalid",
    "Ok",
    "Result",
    "Valid",
    "Validation",
    "and_then",
    "ap_validation",
    "collect_unit_validations",
    "fold_result",
    "fold_validation",
    "map_err",
    "map_result",
    "map_validation",
    "map_validation_err",
    "partition_results",
    "raise_on_err",
    "result_to_validation",
    "traverse_result",
    "traverse_validation",
    "validation_to_result",
    "value_or",
    "value_or_else",
    "zip_validation2",
    "zip_validation3",
]


@dataclass(frozen=True)
class Ok(typing.Generic[_A_co]):
    """Right/success constructor of the closed ``Result[_E, _A]`` sum type.

    Attributes:
        value (_A_co): Contained success payload.
    """

    value: _A_co

    def is_ok(self) -> bool:
        """Return True because this variant is ``Ok``.

        Returns:
            bool: Always ``True``.
        """
        return True

    def is_err(self) -> bool:
        """Return False because this variant is ``Ok``.

        Returns:
            bool: Always ``False``.
        """
        return False


@dataclass(frozen=True)
class Err(typing.Generic[_E_co]):
    """Left/failure constructor of the closed ``Result[_E, _A]`` sum type.

    Attributes:
        error (_E_co): Contained error payload.
    """

    error: _E_co

    def is_ok(self) -> bool:
        """Return False because this variant is ``Err``.

        Returns:
            bool: Always ``False``.
        """
        return False

    def is_err(self) -> bool:
        """Return True because this variant is ``Err``.

        Returns:
            bool: Always ``True``.
        """
        return True


Result: typing_extensions.TypeAlias = typing.Union[Err[_E_co], Ok[_A_co]]


def fold_result(
    res: Result[_E, _A],
    on_err: Callable[[_E], _R],
    on_ok: Callable[[_A], _R],
) -> _R:
    """Exhaustively eliminate a ``Result[_E, _A]`` (Haskell ``Data.Either.either``).

    Args:
        res (Result[_E, _A]): Input result sum value.
        on_err (Callable[[_E], _R]): Continuation invoked when ``res`` is ``Err``.
        on_ok (Callable[[_A], _R]): Continuation invoked when ``res`` is ``Ok``.

    Returns:
        _R: Eliminated value.
    """
    if isinstance(res, Ok):
        return on_ok(res.value)
    elif isinstance(res, Err):
        return on_err(res.error)
    else:
        typing_extensions.assert_never(res)


def map_result(res: Result[_E, _A], fn: Callable[[_A], _B]) -> Result[_E, _B]:
    """Covariantly map ``fn`` over the success payload of ``res`` (``Functor`` on ``Ok``).

    Args:
        res (Result[_E, _A]): Input result value.
        fn (Callable[[_A], _B]): Pure function transforming ``_A`` to ``_B``.

    Returns:
        Result[_E, _B]: Mapped ``Ok(fn(value))`` or unchanged ``Err``.
    """
    if isinstance(res, Ok):
        return Ok(fn(res.value))
    elif isinstance(res, Err):
        return res
    else:
        typing_extensions.assert_never(res)


def map_err(res: Result[_E, _A], fn: Callable[[_E], _F]) -> Result[_F, _A]:
    """Map ``fn`` over the error payload of ``res`` (``Bifunctor.first`` on ``Err``).

    Args:
        res (Result[_E, _A]): Input result value.
        fn (Callable[[_E], _F]): Pure function transforming ``_E`` to ``_F``.

    Returns:
        Result[_F, _A]: Unchanged ``Ok`` or mapped ``Err(fn(error))``.
    """
    if isinstance(res, Ok):
        return res
    elif isinstance(res, Err):
        return Err(fn(res.error))
    else:
        typing_extensions.assert_never(res)


def and_then(res: Result[_E, _A], fn: Callable[[_A], Result[_E, _B]]) -> Result[_E, _B]:
    """Monadic bind (``>>=``) sequencing ``fn`` when ``res`` is ``Ok`` and short-circuiting ``Err``.

    Args:
        res (Result[_E, _A]): Upstream result computation.
        fn (Callable[[_A], Result[_E, _B]]): Kleisli arrow taking ``_A`` to ``Result[_E, _B]``.

    Returns:
        Result[_E, _B]: Sequenced result.
    """
    if isinstance(res, Ok):
        return fn(res.value)
    elif isinstance(res, Err):
        return res
    else:
        typing_extensions.assert_never(res)


def value_or(res: Result[_E, _A], default: _A) -> _A:
    """Extract the ``Ok`` value or return ``default`` when ``res`` is ``Err``.

    Args:
        res (Result[_E, _A]): Input result value.
        default (_A): Fallback value returned on ``Err``.

    Returns:
        _A: Extracted or fallback value.
    """
    if isinstance(res, Ok):
        return res.value
    elif isinstance(res, Err):
        return default
    else:
        typing_extensions.assert_never(res)


def value_or_else(res: Result[_E, _A], fallback: Callable[[_E], _A]) -> _A:
    """Extract the ``Ok`` value or compute a fallback from the ``Err`` payload.

    Args:
        res (Result[_E, _A]): Input result value.
        fallback (Callable[[_E], _A]): Recovery function receiving the error value.

    Returns:
        _A: Extracted or recovered value.
    """
    if isinstance(res, Ok):
        return res.value
    elif isinstance(res, Err):
        return fallback(res.error)
    else:
        typing_extensions.assert_never(res)


def raise_on_err(res: Result[_ExcT, _A]) -> _A:
    """Return ``res.value`` on ``Ok`` or raise ``res.error`` at legacy imperative boundaries.

    Args:
        res (Result[_ExcT, _A]): Result whose error type is an ``Exception`` subclass.

    Returns:
        _A: Success value when ``res`` is ``Ok``.

    Raises:
        Exception: Re-raises ``res.error`` when ``res`` is ``Err``.
    """
    if isinstance(res, Ok):
        return res.value
    elif isinstance(res, Err):
        raise res.error
    else:
        typing_extensions.assert_never(res)


def traverse_result(
    items: Iterable[_A],
    fn: Callable[[_A], Result[_E, _B]],
) -> Result[_E, tuple[_B, ...]]:
    """Map ``fn`` across ``items``, short-circuiting on the first ``Err`` encountered.

    Args:
        items (Iterable[_A]): Input elements to traverse.
        fn (Callable[[_A], Result[_E, _B]]): Effectful step function.

    Returns:
        Result[_E, tuple[_B, ...]]: ``Ok`` of collected results or first ``Err``.
    """
    collected: list[_B] = []
    for item in items:
        step = fn(item)
        if isinstance(step, Ok):
            collected.append(step.value)
        elif isinstance(step, Err):
            return step
        else:
            typing_extensions.assert_never(step)
    return Ok(tuple(collected))


def partition_results(
    items: Iterable[Result[_E, _A]],
) -> tuple[tuple[_E, ...], tuple[_A, ...]]:
    """Partition an iterable of ``Result`` values into ``(errors, oks)`` (Haskell ``partitionEithers``).

    Args:
        items (Iterable[Result[_E, _A]]): Sequence of results.

    Returns:
        tuple[tuple[_E, ...], tuple[_A, ...]]: Tuple of ``(all_errors, all_values)``.
    """
    errs: list[_E] = []
    oks: list[_A] = []
    for item in items:
        if isinstance(item, Ok):
            oks.append(item.value)
        elif isinstance(item, Err):
            errs.append(item.error)
        else:
            typing_extensions.assert_never(item)
    return (tuple(errs), tuple(oks))


@dataclass(frozen=True)
class Valid(typing.Generic[_A_co]):
    """Success constructor of the applicative error-accumulating ``Validation[_E, _A]`` ADT.

    Attributes:
        value (_A_co): Validated payload.
    """

    value: _A_co

    def is_valid(self) -> bool:
        """Return True because this variant is ``Valid``.

        Returns:
            bool: Always ``True``.
        """
        return True

    def is_invalid(self) -> bool:
        """Return False because this variant is ``Valid``.

        Returns:
            bool: Always ``False``.
        """
        return False


@dataclass(frozen=True)
class Invalid(typing.Generic[_E_co]):
    """Failure constructor of the applicative error-accumulating ``Validation[_E, _A]`` ADT.

    Attributes:
        errors (_E_co): Semigroup-accumulated error collection.
    """

    errors: _E_co

    def is_valid(self) -> bool:
        """Return False because this variant is ``Invalid``.

        Returns:
            bool: Always ``False``.
        """
        return False

    def is_invalid(self) -> bool:
        """Return True because this variant is ``Invalid``.

        Returns:
            bool: Always ``True``.
        """
        return True


Validation: typing_extensions.TypeAlias = typing.Union[Invalid[_E_co], Valid[_A_co]]


def fold_validation(
    val: Validation[_E, _A],
    on_invalid: Callable[[_E], _R],
    on_valid: Callable[[_A], _R],
) -> _R:
    """Exhaustively eliminate a ``Validation[_E, _A]`` value.

    Args:
        val (Validation[_E, _A]): Input validation value.
        on_invalid (Callable[[_E], _R]): Handler for ``Invalid``.
        on_valid (Callable[[_A], _R]): Handler for ``Valid``.

    Returns:
        _R: Eliminated output.
    """
    if isinstance(val, Valid):
        return on_valid(val.value)
    elif isinstance(val, Invalid):
        return on_invalid(val.errors)
    else:
        typing_extensions.assert_never(val)


def map_validation(val: Validation[_E, _A], fn: Callable[[_A], _B]) -> Validation[_E, _B]:
    """Covariantly map ``fn`` over a ``Valid`` payload.

    Args:
        val (Validation[_E, _A]): Input validation value.
        fn (Callable[[_A], _B]): Pure function on valid payload.

    Returns:
        Validation[_E, _B]: Mapped validation outcome.
    """
    if isinstance(val, Valid):
        return Valid(fn(val.value))
    elif isinstance(val, Invalid):
        return val
    else:
        typing_extensions.assert_never(val)


def map_validation_err(val: Validation[_E, _A], fn: Callable[[_E], _F]) -> Validation[_F, _A]:
    """Map ``fn`` over an ``Invalid`` error payload.

    Args:
        val (Validation[_E, _A]): Input validation value.
        fn (Callable[[_E], _F]): Error transformation function.

    Returns:
        Validation[_F, _A]: Validation with transformed error payload.
    """
    if isinstance(val, Valid):
        return val
    elif isinstance(val, Invalid):
        return Invalid(fn(val.errors))
    else:
        typing_extensions.assert_never(val)


def ap_validation(
    sg: Semigroup[_E],
    fn_val: Validation[_E, Callable[[_A], _B]],
    arg_val: Validation[_E, _A],
) -> Validation[_E, _B]:
    """Applicative ``<*>`` accumulating errors via ``sg.combine`` when both sides are ``Invalid``.

    Unlike monadic ``Result``, ``Validation`` evaluates both ``fn_val`` and ``arg_val``
    independently and combines their ``Invalid`` payloads via ``sg``.

    Args:
        sg (Semigroup[_E]): Semigroup witness for error accumulation.
        fn_val (Validation[_E, Callable[[_A], _B]]): Validated function.
        arg_val (Validation[_E, _A]): Validated argument.

    Returns:
        Validation[_E, _B]: Applied ``Valid`` value or combined ``Invalid`` errors.
    """
    if isinstance(fn_val, Valid):
        if isinstance(arg_val, Valid):
            return Valid(fn_val.value(arg_val.value))
        elif isinstance(arg_val, Invalid):
            return arg_val
        else:
            typing_extensions.assert_never(arg_val)
    elif isinstance(fn_val, Invalid):
        if isinstance(arg_val, Valid):
            return fn_val
        elif isinstance(arg_val, Invalid):
            return Invalid(sg.combine(fn_val.errors, arg_val.errors))
        else:
            typing_extensions.assert_never(arg_val)
    else:
        typing_extensions.assert_never(fn_val)


def zip_validation2(
    sg: Semigroup[_E],
    first: Validation[_E, _A],
    second: Validation[_E, _B],
    combine_ok: Callable[[_A, _B], _C],
) -> Validation[_E, _C]:
    """Applicatively combine two ``Validation`` values (Haskell ``liftA2``), accumulating errors.

    Args:
        sg (Semigroup[_E]): Error semigroup witness.
        first (Validation[_E, _A]): First validated input.
        second (Validation[_E, _B]): Second validated input.
        combine_ok (Callable[[_A, _B], _C]): Pure binary combiner when both inputs are ``Valid``.

    Returns:
        Validation[_E, _C]: Combined ``Valid`` result or accumulated ``Invalid`` errors.
    """

    def _curried(a: _A) -> Callable[[_B], _C]:
        """Partially apply ``combine_ok`` to ``a``.

        Args:
            a (_A): First valid value.

        Returns:
            Callable[[_B], _C]: Unary continuation awaiting ``_B``.
        """

        def _apply_b(b: _B) -> _C:
            """Apply ``combine_ok(a, b)``.

            Args:
                b (_B): Second valid value.

            Returns:
                _C: Combined value.
            """
            return combine_ok(a, b)

        return _apply_b

    return ap_validation(sg, map_validation(first, _curried), second)


def zip_validation3(
    sg: Semigroup[_E],
    first: Validation[_E, _A],
    second: Validation[_E, _B],
    third: Validation[_E, _C],
    combine_ok: Callable[[_A, _B, _C], _D],
) -> Validation[_E, _D]:
    """Applicatively combine three ``Validation`` values (Haskell ``liftA3``), accumulating all errors.

    Args:
        sg (Semigroup[_E]): Error semigroup witness.
        first (Validation[_E, _A]): First validated input.
        second (Validation[_E, _B]): Second validated input.
        third (Validation[_E, _C]): Third validated input.
        combine_ok (Callable[[_A, _B, _C], _D]): Pure ternary combiner when all inputs are ``Valid``.

    Returns:
        Validation[_E, _D]: Combined ``Valid`` result or accumulated ``Invalid`` errors.
    """

    def _pair(a: _A, b: _B) -> tuple[_A, _B]:
        """Pack first two valid inputs into a pair.

        Args:
            a (_A): First value.
            b (_B): Second value.

        Returns:
            tuple[_A, _B]: Tuple ``(a, b)``.
        """
        return (a, b)

    def _finish(ab: tuple[_A, _B], c: _C) -> _D:
        """Apply ``combine_ok`` to unpacked pair and third value.

        Args:
            ab (tuple[_A, _B]): First two valid values.
            c (_C): Third valid value.

        Returns:
            _D: Combined output.
        """
        return combine_ok(ab[0], ab[1], c)

    return zip_validation2(sg, zip_validation2(sg, first, second, _pair), third, _finish)


def traverse_validation(
    sg: Semigroup[_E],
    items: Iterable[_A],
    fn: Callable[[_A], Validation[_E, _B]],
) -> Validation[_E, tuple[_B, ...]]:
    """Applicatively traverse ``items``, accumulating EVERY ``Invalid`` error via ``sg`` without short-circuiting.

    Args:
        sg (Semigroup[_E]): Semigroup witness for error accumulation.
        items (Iterable[_A]): Input elements to validate.
        fn (Callable[[_A], Validation[_E, _B]]): Element validation function.

    Returns:
        Validation[_E, tuple[_B, ...]]: ``Valid`` tuple if all succeed, else ``Invalid`` of all accumulated errors.
    """
    oks: list[_B] = []
    errs: list[_E] = []
    for item in items:
        step = fn(item)
        if isinstance(step, Valid):
            oks.append(step.value)
        elif isinstance(step, Invalid):
            errs.append(step.errors)
        else:
            typing_extensions.assert_never(step)
    if errs:
        return Invalid(sg.sconcat(errs[0], errs[1:]))
    return Valid(tuple(oks))


def collect_unit_validations(
    monoid: Monoid[_E],
    checks: Iterable[Validation[_E, None]],
) -> _E:
    """Fold rule checks returning ``Validation[_E, None]`` into an accumulated error collection ``_E``.

    Args:
        monoid (Monoid[_E]): Monoid witness for error collections (yielding ``empty()`` when all pass).
        checks (Iterable[Validation[_E, None]]): Validation rule outcomes to accumulate.

    Returns:
        _E: Accumulated error collection (``monoid.empty()`` when every check is ``Valid``).
    """
    acc = monoid.empty()
    for check in checks:
        if isinstance(check, Valid):
            continue
        elif isinstance(check, Invalid):
            acc = monoid.combine(acc, check.errors)
        else:
            typing_extensions.assert_never(check)
    return acc


def result_to_validation(res: Result[_E, _A]) -> Validation[_E, _A]:
    """Convert a monadic ``Result[_E, _A]`` into an applicative ``Validation[_E, _A]``.

    Args:
        res (Result[_E, _A]): Input result sum value.

    Returns:
        Validation[_E, _A]: ``Valid(value)`` on ``Ok`` or ``Invalid(error)`` on ``Err``.
    """
    if isinstance(res, Ok):
        return Valid(res.value)
    elif isinstance(res, Err):
        return Invalid(res.error)
    else:
        typing_extensions.assert_never(res)


def validation_to_result(val: Validation[_E, _A]) -> Result[_E, _A]:
    """Convert an applicative ``Validation[_E, _A]`` into a monadic ``Result[_E, _A]``.

    Args:
        val (Validation[_E, _A]): Input validation sum value.

    Returns:
        Result[_E, _A]: ``Ok(value)`` on ``Valid`` or ``Err(errors)`` on ``Invalid``.
    """
    if isinstance(val, Valid):
        return Ok(val.value)
    elif isinstance(val, Invalid):
        return Err(val.errors)
    else:
        typing_extensions.assert_never(val)
