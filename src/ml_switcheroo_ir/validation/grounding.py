"""Empirical anti-hallucination grounding diagnostic and report models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from ml_switcheroo_ir.schema.ghost import ExtendedGhostRef
from ml_switcheroo_ir.validation.core import ValidationError

__all__ = [
    "DiagnosticSeverity",
    "GroundingAuditReport",
    "GroundingDiagnostic",
    "GroundingReport",
    "GroundingSeverity",
]


@dataclass
class GroundingAuditReport:
    """Audit report detailing graph grounding against ground-truth framework snapshots.

    Attributes:
        total_nodes (int): Total number of nodes in graph.
        grounded_count (int): Number of nodes successfully grounded against snapshot.
        ungrounded_count (int): Number of ungrounded or hallucinated nodes.
        hallucination_score (float): Percentage of nodes that are ungrounded (0.0 to 1.0).
        diagnostics (List[ValidationError]): Diagnostic error reports.
    """

    total_nodes: int
    grounded_count: int
    ungrounded_count: int
    hallucination_score: float
    diagnostics: list[ValidationError]


class DiagnosticSeverity(str, Enum):
    """Severity classification for grounding diagnostics."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class GroundingDiagnostic(BaseModel):
    """Specific diagnostic message emitted during symbol or operation verification.

    Attributes:
        field (str): The component, attribute, or operand evaluated.
        message (str): Human-readable diagnostic explanation.
        severity (DiagnosticSeverity): Severity level of the diagnostic.
        suggested_fix (Optional[str]): Suggested replacement or typo correction.
    """

    field: str = Field(description="The component, attribute, or operand evaluated.")
    message: str = Field(description="Human-readable diagnostic explanation.")
    severity: DiagnosticSeverity = Field(
        default=DiagnosticSeverity.ERROR,
        description="Severity level of the diagnostic.",
    )
    suggested_fix: str | None = Field(
        default=None,
        description="Suggested replacement or typo correction.",
    )


class GroundingReport(BaseModel):
    """Comprehensive validation report for a verified operation or symbol.

    Attributes:
        is_grounded (bool): True if the symbol is grounded and valid without fatal errors.
        target (str): Target framework, dialect, or ISA.
        symbol (str): Queried symbol, mnemonic, or operation identifier.
        diagnostics (list[GroundingDiagnostic]): Collection of diagnostics produced.
        matched_ref (Optional[ExtendedGhostRef]): Resolved ground-truth reference object if discovered.
    """

    is_grounded: bool = Field(
        description="True if the symbol is grounded and valid without fatal errors."
    )
    target: str = Field(description="Target framework, dialect, or ISA.")
    symbol: str = Field(
        description="Queried symbol, mnemonic, or operation identifier."
    )
    diagnostics: list[GroundingDiagnostic] = Field(
        default_factory=list,
        description="Collection of diagnostics produced during verification.",
    )
    matched_ref: ExtendedGhostRef | None = Field(
        default=None,
        description="Resolved ground-truth reference object if discovered.",
    )

    @property
    def has_errors(self) -> bool:
        """Check if any diagnostics have ERROR severity.

        Returns:
            bool: True if any diagnostic is an ERROR, False otherwise.
        """
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)

    def add_diagnostic(
        self,
        field: str,
        message: str,
        severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
        suggested_fix: str | None = None,
    ) -> None:
        """Append a new diagnostic message and update is_grounded status.

        Args:
            field (str): Component or attribute path.
            message (str): Human-readable error message.
            severity (DiagnosticSeverity): Severity level of the diagnostic.
            suggested_fix (Optional[str]): Suggested correction or candidate symbol.
        """
        if severity == DiagnosticSeverity.ERROR:
            self.is_grounded = False
        self.diagnostics.append(
            GroundingDiagnostic(
                field=field,
                message=message,
                severity=severity,
                suggested_fix=suggested_fix,
            )
        )


GroundingSeverity = DiagnosticSeverity
