"""Closed algebraic IrDomainError sum type and exhaustive diagnostics formatting."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import typing_extensions

from ml_switcheroo_ir.identifiers import IdentifierKind, IdentifierParseError, NodeId
from ml_switcheroo_ir.translation import TranslationIsoError
from ml_switcheroo_ir.typeclasses import CodecError, Eq, NonEmpty, Pretty
from ml_switcheroo_ir.types import DimensionValue, TensorShape
from ml_switcheroo_ir.validation.core import ValidationError
from ml_switcheroo_ir.validation.grounding import GroundingDiagnostic

__all__ = [
    "GraphCycleDiagnosticError",
    "GroundingAuditBatchError",
    "IR_DOMAIN_ERROR_EQ",
    "IR_DOMAIN_ERROR_PRETTY",
    "IdentifierKind",
    "IdentifierParseError",
    "IrBoundaryException",
    "IrCodecDecodeError",
    "IrDomainError",
    "SchemaValidationBatchError",
    "ShapeUnificationError",
    "ShardingValidationBatchError",
    "TranslationIsoError",
    "format_ir_domain_error",
    "to_boundary_exception",
]


@dataclass(frozen=True)
class GraphCycleDiagnosticError:
    """DAG acyclicity violation variant of ``IrDomainError``.

    Attributes:
        cycle_nodes (tuple[NodeId, ...]): Ordered sequence of node identifiers participating in the cycle.
        message (str): Human-readable topological cycle summary.
    """

    cycle_nodes: tuple[NodeId, ...]
    message: str


@dataclass(frozen=True)
class SchemaValidationBatchError:
    """Schema validation failure batch variant of ``IrDomainError``.

    Attributes:
        diagnostics (NonEmpty[ValidationError]): Non-empty batch of schema validation errors.
    """

    diagnostics: NonEmpty[ValidationError]


@dataclass(frozen=True)
class ShardingValidationBatchError:
    """Distributed sharding/pipeline validation failure batch variant of ``IrDomainError``.

    Attributes:
        diagnostics (NonEmpty[ValidationError]): Non-empty batch of sharding validation errors.
    """

    diagnostics: NonEmpty[ValidationError]


@dataclass(frozen=True)
class GroundingAuditBatchError:
    """Empirical snapshot grounding failure batch variant of ``IrDomainError``.

    Attributes:
        diagnostics (NonEmpty[GroundingDiagnostic | ValidationError]): Non-empty grounding diagnostics.
    """

    diagnostics: NonEmpty[GroundingDiagnostic | ValidationError]


@dataclass(frozen=True)
class IrCodecDecodeError:
    """Wire deserialization failure variant of ``IrDomainError``.

    Attributes:
        codec_error (CodecError): Underlying path-annotated codec failure.
    """

    codec_error: CodecError


@dataclass(frozen=True)
class ShapeUnificationError:
    """Symbolic or static shape/dimension unification failure variant of ``IrDomainError``.

    Attributes:
        lhs (TensorShape | DimensionValue): Left-hand shape or dimension operand.
        rhs (TensorShape | DimensionValue): Right-hand shape or dimension operand.
        reason (str): Detailed explanation of the shape mismatch.
    """

    lhs: TensorShape | DimensionValue
    rhs: TensorShape | DimensionValue
    reason: str


IrDomainError: typing_extensions.TypeAlias = typing.Union[
    IdentifierParseError,
    GraphCycleDiagnosticError,
    SchemaValidationBatchError,
    ShardingValidationBatchError,
    GroundingAuditBatchError,
    TranslationIsoError,
    IrCodecDecodeError,
    ShapeUnificationError,
]


def _render_grounding_item(item: GroundingDiagnostic | ValidationError) -> str:
    """Format a single grounding or validation diagnostic item.

    Args:
        item (GroundingDiagnostic | ValidationError): Diagnostic element.

    Returns:
        str: Formatted single-line diagnostic text.
    """
    if isinstance(item, ValidationError):
        return str(item)
    return f"[{item.severity.value}] {item.field}: {item.message}"


def format_ir_domain_error(err: IrDomainError) -> str:
    """Exhaustively format any ``IrDomainError`` variant using ``assert_never``.

    Args:
        err (IrDomainError): Closed domain error sum value.

    Returns:
        str: Deterministic human-readable diagnostic message.
    """
    msg: str
    if isinstance(err, IdentifierParseError):
        msg = f"IdentifierParseError({err.kind.value}, {err.raw_value!r}): {err.reason}"
    elif isinstance(err, GraphCycleDiagnosticError):
        nodes_txt = ", ".join(str(n) for n in err.cycle_nodes)
        msg = f"GraphCycleDiagnosticError([{nodes_txt}]): {err.message}"
    elif isinstance(err, SchemaValidationBatchError):
        joined = "; ".join(str(d) for d in err.diagnostics)
        msg = f"SchemaValidationBatchError({len(err.diagnostics)}): {joined}"
    elif isinstance(err, ShardingValidationBatchError):
        joined = "; ".join(str(d) for d in err.diagnostics)
        msg = f"ShardingValidationBatchError({len(err.diagnostics)}): {joined}"
    elif isinstance(err, GroundingAuditBatchError):
        joined = "; ".join(_render_grounding_item(d) for d in err.diagnostics)
        msg = f"GroundingAuditBatchError({len(err.diagnostics)}): {joined}"
    elif isinstance(err, TranslationIsoError):
        param_part = f", param={err.parameter}" if err.parameter is not None else ""
        msg = f"TranslationIsoError(op={err.op_name}, dir={err.direction}{param_part}): {err.reason}"
    elif isinstance(err, IrCodecDecodeError):
        msg = f"IrCodecDecodeError: {err.codec_error}"
    elif isinstance(err, ShapeUnificationError):
        msg = f"ShapeUnificationError(lhs={err.lhs!r}, rhs={err.rhs!r}): {err.reason}"
    else:
        typing_extensions.assert_never(err)
    return msg


IR_DOMAIN_ERROR_EQ: Eq[IrDomainError] = Eq(eq=lambda lhs, rhs: lhs == rhs)

IR_DOMAIN_ERROR_PRETTY: Pretty[IrDomainError] = Pretty(render=format_ir_domain_error)


class IrBoundaryException(Exception):
    """Single imperative boundary exception wrapping a structured ``IrDomainError``.

    Attributes:
        error (IrDomainError): Underlying closed algebraic domain error payload.
    """

    error: IrDomainError

    def __init__(self, error: IrDomainError) -> None:
        """Initialize ``IrBoundaryException`` from a closed ``IrDomainError``.

        Args:
            error (IrDomainError): Structured domain error value.
        """
        self.error = error
        super().__init__(format_ir_domain_error(error))


def to_boundary_exception(err: IrDomainError) -> IrBoundaryException:
    """Wrap an ``IrDomainError`` in ``IrBoundaryException`` for ``raise_on_err`` boundaries.

    Args:
        err (IrDomainError): Closed domain error value.

    Returns:
        IrBoundaryException: Imperative exception wrapper carrying ``err``.
    """
    return IrBoundaryException(err)
