"""Core validation types, levels, errors, and Levenshtein edit distance."""

from __future__ import annotations

import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import pydantic
import typing_extensions

from ml_switcheroo_ir.schema.base import REGISTRY_LOCK, LazyMetadataMap
from ml_switcheroo_ir.schema.mlir_registry import build_core_mlir_dialects
from ml_switcheroo_ir.typeclasses.algebra import (
    INT_ORD,
    BoundedLattice,
    Eq,
    Monoid,
    NonEmpty,
    Ord,
    Semigroup,
    contramap_ord,
    nonempty_semigroup,
    tuple_monoid,
)
from ml_switcheroo_ir.typeclasses.outcome import (
    Invalid,
    Valid,
    Validation,
    collect_unit_validations_ne,
    from_error_sequence,
)

_TargetT = typing.TypeVar("_TargetT")

__all__ = [
    "CORE_MLIR_DIALECTS",
    "KNOWN_SYNONYMS",
    "VALIDATION_ERRORS_MONOID",
    "VALIDATION_ERRORS_NE_SEMIGROUP",
    "VALIDATION_ERRORS_SEMIGROUP",
    "VALIDATION_LEVEL_EQ",
    "VALIDATION_LEVEL_LATTICE",
    "VALIDATION_LEVEL_ORD",
    "ValidationError",
    "ValidationLevel",
    "ValidationRule",
    "combine_rules",
    "compute_levenshtein",
    "errors_from_validation",
    "rule_from_errors_fn",
    "validation_level_rank",
    "when_predicate",
]


def compute_levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.

    Returns:
        int: Integer edit distance between s1 and s2.
    """
    if len(s1) < len(s2):
        return compute_levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


KNOWN_SYNONYMS: dict[tuple[str, str], str] = {
    ("stablehlo", "matmul"): "stablehlo.dot_general",
    ("stablehlo", "linear"): "stablehlo.dot_general",
    ("stablehlo", "conv2d"): "stablehlo.convolution",
    ("arith", "mul"): "arith.mulf",
    ("arith", "add"): "arith.addf",
    ("math", "exp2"): "math.exp",
    ("torch", "rms_norm"): "torch.nn.RMSNorm",
    ("torch", "layer_norm"): "torch.nn.LayerNorm",
    ("torch", "group_norm"): "torch.nn.GroupNorm",
    ("torch", "batch_norm"): "torch.nn.BatchNorm2d",
    ("torch", "gelu"): "torch.nn.GELU",
    ("torch", "relu"): "torch.nn.ReLU",
    ("torch", "silu"): "torch.nn.SiLU",
    ("torch", "all_reduce"): "torch.distributed.all_reduce",
    ("torch", "all_gather"): "torch.distributed.all_gather",
    ("torch", "reduce_scatter"): "torch.distributed.reduce_scatter",
    ("jax", "rms_norm"): "jax.nn.standardize",
    ("jax", "gelu"): "jax.nn.gelu",
    ("jax", "relu"): "jax.nn.relu",
    ("jax", "silu"): "jax.nn.silu",
    ("jax", "softmax"): "jax.nn.softmax",
    ("tf", "rms_norm"): "tf.keras.layers.RMSNormalization",
    ("tf", "layer_norm"): "tf.keras.layers.LayerNormalization",
    ("tf", "gelu"): "tf.nn.gelu",
    ("tf", "relu"): "tf.nn.relu",
    ("tf", "silu"): "tf.nn.silu",
    ("optax", "cross_entropy"): "optax.losses.softmax_cross_entropy",
    ("optax", "adam_w"): "optax.adamw",
    ("orbax", "checkpointer"): "orbax.checkpoint.PyTreeCheckpointer",
    ("chex", "assert_allclose"): "chex.assert_trees_all_close",
    ("grain", "DataLoader"): "grain.DataLoader",
}


def _load_core_mlir_dialects() -> None:
    """Populate ``CORE_MLIR_DIALECTS`` lazily from the unified ``MLIR_REGISTRY``."""
    with REGISTRY_LOCK:
        CORE_MLIR_DIALECTS.mark_loaded()
        for dialect, ops in build_core_mlir_dialects().items():
            CORE_MLIR_DIALECTS[dialect] = ops


CORE_MLIR_DIALECTS: LazyMetadataMap[dict[str, int]] = LazyMetadataMap(
    "CORE_MLIR_DIALECTS", loader=_load_core_mlir_dialects
)


class ValidationLevel(Enum):
    """Severity levels for validation errors and validator configuration.

    Attributes:
        STRICT (ValidationLevel): Treat all errors and warnings as fatal errors.
        WARNING (ValidationLevel): Emit non-fatal warning diagnostics.
        LENIENT (ValidationLevel): Suppress warning-level diagnostics.
        ERROR (ValidationLevel): Standard fatal validation error level.
    """

    STRICT = "STRICT"
    WARNING = "WARNING"
    LENIENT = "LENIENT"
    ERROR = "ERROR"


def validation_level_rank(level: ValidationLevel) -> int:
    """Map ``ValidationLevel`` exhaustively to its monotonic lattice rank ``0..3``.

    Order: ``LENIENT (0) < WARNING (1) < ERROR (2) < STRICT (3)``.

    Args:
        level (ValidationLevel): Validation severity/strictness enum value.

    Returns:
        int: Monotonic integer rank in ``{0, 1, 2, 3}``.
    """
    rank: int
    if level is ValidationLevel.LENIENT:
        rank = 0
    elif level is ValidationLevel.WARNING:
        rank = 1
    elif level is ValidationLevel.ERROR:
        rank = 2
    elif level is ValidationLevel.STRICT:
        rank = 3
    else:
        typing_extensions.assert_never(level)
    return rank


class _ValidationErrorPayload(pydantic.BaseModel):
    """Pydantic schema validating ValidationError fields.

    Attributes:
        node_id (str): Node identifier where the error occurred.
        attribute (str): Attribute or structural slot name.
        message (str): Detailed diagnostic description.
        level (ValidationLevel): Diagnostic severity level.
        suggested_fix (str | None): Optional suggested correction string.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    node_id: str
    attribute: str
    message: str
    level: ValidationLevel = ValidationLevel.ERROR
    suggested_fix: str | None = None


class ValidationError(Exception):
    """Represents an error found during graph or node validation.

    Attributes:
        node_id (str): The ID of the node where the error occurred.
        attribute (str): The name of the attribute involved, or a general descriptor.
        message (str): The detailed error message.
        level (ValidationLevel): The severity of the error.
        suggested_fix (str | None): Suggested correction or candidate symbol.
    """

    node_id: str
    attribute: str
    message: str
    level: ValidationLevel
    suggested_fix: str | None

    def __init__(
        self,
        node_id: str,
        attribute: str,
        message: str,
        level: ValidationLevel = ValidationLevel.ERROR,
        suggested_fix: str | None = None,
    ) -> None:
        """Initialize ValidationError with Pydantic field validation.

        Args:
            node_id (str): The ID of the node where the error occurred.
            attribute (str): The name of the attribute involved.
            message (str): The detailed error message.
            level (ValidationLevel): The severity of the error.
            suggested_fix (str | None): Suggested correction or candidate symbol.
        """
        payload = _ValidationErrorPayload(
            node_id=node_id,
            attribute=attribute,
            message=message,
            level=level,
            suggested_fix=suggested_fix,
        )
        super().__init__(payload.message)
        self.node_id = payload.node_id
        self.attribute = payload.attribute
        self.message = payload.message
        self.level = payload.level
        self.suggested_fix = payload.suggested_fix

    def __str__(self) -> str:
        """Return formatted string description of the validation error.

        Returns:
            str: Human-readable error message with severity level.
        """
        return f"[{self.level.value}] Node '{self.node_id}' attribute '{self.attribute}': {self.message}"

    def _as_tuple(self) -> tuple[str, str, str, ValidationLevel, str | None]:
        """Project ``ValidationError`` fields into an immutable canonical comparison tuple.

        Returns:
            tuple[str, str, str, ValidationLevel, str | None]: Canonical field tuple.
        """
        return (self.node_id, self.attribute, self.message, self.level, self.suggested_fix)

    def __eq__(self, other: object) -> bool:
        """Check structural equality between two ``ValidationError`` instances.

        Args:
            other (object): Candidate comparison object.

        Returns:
            bool: True iff ``other`` is a ``ValidationError`` with identical fields.
        """
        if not isinstance(other, ValidationError):
            return False
        return self._as_tuple() == other._as_tuple()

    def __hash__(self) -> int:
        """Compute deterministic hash coherent with ``ValidationError.__eq__``.

        Returns:
            int: Structural hash code.
        """
        return hash(self._as_tuple())


VALIDATION_LEVEL_ORD: typing.Final[Ord[ValidationLevel]] = contramap_ord(INT_ORD, validation_level_rank)
VALIDATION_LEVEL_EQ: typing.Final[Eq[ValidationLevel]] = VALIDATION_LEVEL_ORD.eq_witness
VALIDATION_LEVEL_LATTICE: typing.Final[BoundedLattice[ValidationLevel]] = BoundedLattice(
    eq_witness=VALIDATION_LEVEL_EQ,
    bottom=lambda: ValidationLevel.LENIENT,
    top=lambda: ValidationLevel.STRICT,
    join=VALIDATION_LEVEL_ORD.max_val,
    meet=VALIDATION_LEVEL_ORD.min_val,
)

VALIDATION_ERRORS_MONOID: typing.Final[Monoid[tuple[ValidationError, ...]]] = tuple_monoid()
VALIDATION_ERRORS_SEMIGROUP: typing.Final[Semigroup[tuple[ValidationError, ...]]] = VALIDATION_ERRORS_MONOID.semigroup
VALIDATION_ERRORS_NE_SEMIGROUP: typing.Final[Semigroup[NonEmpty[ValidationError]]] = nonempty_semigroup()


@dataclass(frozen=True)
class ValidationRule(typing.Generic[_TargetT]):
    """Composable applicative validation rule over target ``_TargetT`` (node, sequence, or graph).

    Attributes:
        name (str): Descriptive rule identifier.
        run (Callable[[_TargetT], Validation[NonEmpty[ValidationError], None]]):
            Total check returning ``Valid(None)`` or ``Invalid(NonEmpty(...))``.
    """

    name: str
    run: Callable[[_TargetT], Validation[NonEmpty[ValidationError], None]]


def rule_from_errors_fn(
    name: str,
    fn: Callable[[_TargetT], Sequence[ValidationError]],
) -> ValidationRule[_TargetT]:
    """Lift a legacy ``_TargetT -> Sequence[ValidationError]`` function into a ``ValidationRule``.

    Args:
        name (str): Rule identifier.
        fn (Callable[[_TargetT], Sequence[ValidationError]]): Diagnostic list producer.

    Returns:
        ValidationRule[_TargetT]: Applicative ``ValidationRule`` witness.
    """

    def _execute(target: _TargetT) -> Validation[NonEmpty[ValidationError], None]:
        """Run ``fn(target)`` and wrap non-empty errors in ``Invalid``.

        Args:
            target (_TargetT): Target value under validation.

        Returns:
            Validation[NonEmpty[ValidationError], None]: ``Valid(None)`` or ``Invalid(NonEmpty(...))``.
        """
        return from_error_sequence(tuple(fn(target)), None)

    return ValidationRule(name=name, run=_execute)


def combine_rules(rules: Sequence[ValidationRule[_TargetT]], name: str = "combined") -> ValidationRule[_TargetT]:
    """Applicatively compose ``rules`` via ``VALIDATION_ERRORS_NE_SEMIGROUP`` without short-circuiting.

    Args:
        rules (Sequence[ValidationRule[_TargetT]]): Ordered rules to execute independently.
        name (str): Composite rule identifier.

    Returns:
        ValidationRule[_TargetT]: Combined rule accumulating all diagnostics across ``rules``.
    """
    frozen_rules = tuple(rules)

    def _run_all(target: _TargetT) -> Validation[NonEmpty[ValidationError], None]:
        """Run every rule in ``frozen_rules`` and accumulate non-empty errors.

        Args:
            target (_TargetT): Target value under validation.

        Returns:
            Validation[NonEmpty[ValidationError], None]: ``Valid(None)`` if all pass else ``Invalid``.
        """
        return collect_unit_validations_ne(rule.run(target) for rule in frozen_rules)

    return ValidationRule(name=name, run=_run_all)


def when_predicate(
    pred: Callable[[_TargetT], bool],
    rule: ValidationRule[_TargetT],
) -> ValidationRule[_TargetT]:
    """Guard ``rule`` so it runs only when ``pred(target)`` holds, yielding ``Valid(None)`` otherwise.

    Args:
        pred (Callable[[_TargetT], bool]): Activation predicate on ``_TargetT``.
        rule (ValidationRule[_TargetT]): Underlying rule to execute when ``pred`` returns True.

    Returns:
        ValidationRule[_TargetT]: Guarded validation rule.
    """

    def _guarded(target: _TargetT) -> Validation[NonEmpty[ValidationError], None]:
        """Evaluate ``pred(target)`` before delegating to ``rule.run``.

        Args:
            target (_TargetT): Target value under validation.

        Returns:
            Validation[NonEmpty[ValidationError], None]: ``Valid(None)`` when skipped else ``rule.run(target)``.
        """
        if not pred(target):
            return Valid(None)
        return rule.run(target)

    return ValidationRule(name=f"when({rule.name})", run=_guarded)


def errors_from_validation(
    outcome: Validation[NonEmpty[ValidationError] | tuple[ValidationError, ...], None],
) -> list[ValidationError]:
    """Extract a mutable ``list[ValidationError]`` from a unit ``Validation`` outcome.

    Args:
        outcome (Validation[NonEmpty[ValidationError] | tuple[ValidationError, ...], None]):
            Validation outcome.

    Returns:
        list[ValidationError]: Empty list on ``Valid``, or list of accumulated errors on ``Invalid``.
    """
    if isinstance(outcome, Valid):
        return []
    elif isinstance(outcome, Invalid):
        return list(outcome.errors)
    else:
        typing_extensions.assert_never(outcome)
