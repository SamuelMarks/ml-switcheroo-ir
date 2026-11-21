"""Phantom-typed proof-carrying graph witnesses (ProvenGraph[PhaseT])."""

from __future__ import annotations

from ml_switcheroo_ir.graph import (
    AcyclicSortedPhase,
    CyclicGraphError,
    GroundedPhase,
    LogicalGraph,
    ProvenGraph,
    SchemaValidPhase,
    ShardingValidPhase,
    topological_sort,
)
from ml_switcheroo_ir.typeclasses.outcome import Err, Invalid, Ok, Result, Valid, Validation
from ml_switcheroo_ir.validation.core import ValidationError, ValidationLevel
from ml_switcheroo_ir.validation.distributed import (
    validate_pipeline_and_checkpointing_graph,
    validate_sharding_propagation_graph,
)
from ml_switcheroo_ir.validation.grounding import (
    GroundingDiagnostic,
    SnapshotManifestType,
    audit_graph_grounding,
)
from ml_switcheroo_ir.validation.schema_validator import Validator

__all__ = [
    "AcyclicSortedPhase",
    "GroundedPhase",
    "ProvenGraph",
    "SchemaValidPhase",
    "ShardingValidPhase",
    "prove_acyclic_sorted",
    "prove_grounded",
    "prove_schema_valid",
    "prove_sharding_valid",
]


def prove_acyclic_sorted(
    graph: LogicalGraph,
) -> Result[CyclicGraphError, ProvenGraph[AcyclicSortedPhase]]:
    """Verify strict DAG acyclicity and mint a ``ProvenGraph[AcyclicSortedPhase]``.

    Args:
        graph (LogicalGraph): Candidate computation graph.

    Returns:
        Result[CyclicGraphError, ProvenGraph[AcyclicSortedPhase]]: ``Ok`` proof witness or ``Err(CyclicGraphError)``.
    """
    cloned = graph.clone()
    try:
        ordered = topological_sort(cloned, strict=True)
    except CyclicGraphError as exc:
        return Err(exc)
    return Ok(ProvenGraph(graph=cloned, ordered_nodes=tuple(ordered)))


def prove_schema_valid(
    graph: LogicalGraph,
    validator: Validator | None = None,
) -> Validation[tuple[ValidationError, ...], ProvenGraph[SchemaValidPhase]]:
    """Run ``Validator.validate_graph`` and mint ``ProvenGraph[SchemaValidPhase]`` if no ``ERROR`` occurs.

    Args:
        graph (LogicalGraph): Candidate computation graph.
        validator (Validator | None): Optional pre-configured ``Validator`` instance.

    Returns:
        Validation[tuple[ValidationError, ...], ProvenGraph[SchemaValidPhase]]:
            ``Valid`` proven graph or ``Invalid`` tuple of fatal ``ValidationError`` items.
    """
    cloned = graph.clone()
    active_validator = validator if validator is not None else Validator()
    diagnostics = active_validator.validate_graph(cloned)
    fatal = tuple(err for err in diagnostics if err.level == ValidationLevel.ERROR)
    if fatal:
        return Invalid(fatal)
    return Valid(ProvenGraph(graph=cloned, ordered_nodes=tuple(cloned.nodes.values())))


def prove_sharding_valid(
    graph: LogicalGraph,
    level: ValidationLevel = ValidationLevel.STRICT,
) -> Validation[tuple[ValidationError, ...], ProvenGraph[ShardingValidPhase]]:
    """Verify SPMD sharding propagation and pipeline stage invariants, minting ``ProvenGraph[ShardingValidPhase]``.

    Args:
        graph (LogicalGraph): Candidate computation graph.
        level (ValidationLevel): Validation strictness level (defaults to ``ValidationLevel.STRICT``).

    Returns:
        Validation[tuple[ValidationError, ...], ProvenGraph[ShardingValidPhase]]:
            ``Valid`` proven graph or ``Invalid`` tuple of fatal sharding/pipeline ``ValidationError`` items.
    """
    cloned = graph.clone()
    sharding_errs = validate_sharding_propagation_graph(cloned, level=level)
    pipeline_errs = validate_pipeline_and_checkpointing_graph(cloned)
    fatal = tuple(err for err in (*sharding_errs, *pipeline_errs) if err.level == ValidationLevel.ERROR)
    if fatal:
        return Invalid(fatal)
    return Valid(ProvenGraph(graph=cloned, ordered_nodes=tuple(cloned.nodes.values())))


def prove_grounded(
    graph: LogicalGraph,
    snapshots_path: SnapshotManifestType = None,
) -> Validation[tuple[GroundingDiagnostic | ValidationError, ...], ProvenGraph[GroundedPhase]]:
    """Audit ``graph`` against ground-truth snapshots and mint ``ProvenGraph[GroundedPhase]`` when grounded.

    Args:
        graph (LogicalGraph): Candidate computation graph.
        snapshots_path (SnapshotManifestType): Optional snapshot directory, manifest file, or in-memory payload.

    Returns:
        Validation[tuple[GroundingDiagnostic | ValidationError, ...], ProvenGraph[GroundedPhase]]:
            ``Valid`` proven graph when ``report.is_grounded`` holds, else ``Invalid`` diagnostics tuple.
    """
    cloned = graph.clone()
    report = audit_graph_grounding(cloned, snapshots_path=snapshots_path)
    if not report.is_grounded:
        return Invalid(tuple(report.diagnostics))
    return Valid(ProvenGraph(graph=cloned, ordered_nodes=tuple(cloned.nodes.values())))
