"""Phantom-typed proof-carrying graph witnesses and cumulative phase combinators (ProvenGraph[PhaseT])."""

from __future__ import annotations

from ml_switcheroo_ir.errors import (
    GraphCycleDiagnosticError,
    GroundingAuditBatchError,
    IrDomainError,
    SchemaValidationBatchError,
    ShardingValidationBatchError,
)
from ml_switcheroo_ir.graph import (
    AcyclicAndSchemaPhase,
    AcyclicSchemaShardingPhase,
    AcyclicSortedPhase,
    CyclicGraphError,
    FullyVerifiedPhase,
    GroundedPhase,
    LogicalGraph,
    ProvenGraph,
    SchemaValidPhase,
    ShardingValidPhase,
    topological_sort,
)
from ml_switcheroo_ir.identifiers import NodeId
from ml_switcheroo_ir.typeclasses.algebra import NonEmpty
from ml_switcheroo_ir.typeclasses.outcome import (
    Err,
    Invalid,
    Ok,
    Result,
    Validation,
    from_error_sequence,
)
from ml_switcheroo_ir.validation.core import ValidationError, ValidationLevel
from ml_switcheroo_ir.validation.distributed import (
    validate_pipeline_and_checkpointing_graph,
    validate_sharding_propagation_graph,
)
from ml_switcheroo_ir.validation.grounding import (
    DiagnosticSeverity,
    GroundingDiagnostic,
    SnapshotManifestType,
    audit_graph_grounding,
)
from ml_switcheroo_ir.validation.schema_validator import Validator

__all__ = [
    "AcyclicAndSchemaPhase",
    "AcyclicSchemaShardingPhase",
    "AcyclicSortedPhase",
    "FullyVerifiedPhase",
    "GroundedPhase",
    "ProvenGraph",
    "SchemaValidPhase",
    "ShardingValidPhase",
    "prove_acyclic_sorted",
    "prove_fully_verified",
    "prove_grounded",
    "prove_schema_valid",
    "prove_sharding_valid",
    "upgrade_to_fully_verified",
    "upgrade_to_schema_valid",
    "upgrade_to_sharding_valid",
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
    return Ok(ProvenGraph.mint(cloned, ordered_nodes=tuple(ordered)))


def prove_schema_valid(
    graph: LogicalGraph,
    validator: Validator | None = None,
) -> Validation[NonEmpty[ValidationError], ProvenGraph[SchemaValidPhase]]:
    """Run ``Validator.validate_graph`` and mint ``ProvenGraph[SchemaValidPhase]`` if no ``ERROR`` occurs.

    Args:
        graph (LogicalGraph): Candidate computation graph.
        validator (Validator | None): Optional pre-configured ``Validator`` instance.

    Returns:
        Validation[NonEmpty[ValidationError], ProvenGraph[SchemaValidPhase]]:
            ``Valid`` proven graph or ``Invalid`` non-empty batch of fatal ``ValidationError`` items.
    """
    cloned = graph.clone()
    active_validator = validator if validator is not None else Validator()
    diagnostics = active_validator.validate_graph(cloned)
    fatal = tuple(err for err in diagnostics if err.level == ValidationLevel.ERROR)
    proven: ProvenGraph[SchemaValidPhase] = ProvenGraph.mint(
        cloned,
        ordered_nodes=tuple(cloned.nodes.values()),
    )
    return from_error_sequence(fatal, proven)


def prove_sharding_valid(
    graph: LogicalGraph,
    level: ValidationLevel = ValidationLevel.STRICT,
) -> Validation[NonEmpty[ValidationError], ProvenGraph[ShardingValidPhase]]:
    """Verify SPMD sharding propagation and pipeline stage invariants, minting ``ProvenGraph[ShardingValidPhase]``.

    Args:
        graph (LogicalGraph): Candidate computation graph.
        level (ValidationLevel): Validation strictness level (defaults to ``ValidationLevel.STRICT``).

    Returns:
        Validation[NonEmpty[ValidationError], ProvenGraph[ShardingValidPhase]]:
            ``Valid`` proven graph or ``Invalid`` non-empty batch of fatal sharding/pipeline ``ValidationError`` items.
    """
    cloned = graph.clone()
    sharding_errs = validate_sharding_propagation_graph(cloned, level=level)
    pipeline_errs = validate_pipeline_and_checkpointing_graph(cloned)
    fatal = tuple(err for err in (*sharding_errs, *pipeline_errs) if err.level == ValidationLevel.ERROR)
    proven: ProvenGraph[ShardingValidPhase] = ProvenGraph.mint(
        cloned,
        ordered_nodes=tuple(cloned.nodes.values()),
    )
    return from_error_sequence(fatal, proven)


def _extract_grounding_failures(
    graph: LogicalGraph,
    snapshots_path: SnapshotManifestType,
) -> tuple[GroundingDiagnostic | ValidationError, ...]:
    """Run grounding audit and return a guaranteed non-empty failure tuple whenever ``is_grounded`` is False.

    Args:
        graph (LogicalGraph): Cloned graph under grounding audit.
        snapshots_path (SnapshotManifestType): Optional snapshot directory, manifest path, or payload.

    Returns:
        tuple[GroundingDiagnostic | ValidationError, ...]: Empty tuple iff ``report.is_grounded`` holds.
    """
    report = audit_graph_grounding(graph, snapshots_path=snapshots_path)
    if report.is_grounded:
        return ()
    if report.diagnostics:
        return tuple(report.diagnostics)
    return (
        GroundingDiagnostic(
            field="graph",
            message="Graph failed grounding check with zero explicit diagnostics.",
            severity=DiagnosticSeverity.ERROR,
        ),
    )


def prove_grounded(
    graph: LogicalGraph,
    snapshots_path: SnapshotManifestType = None,
) -> Validation[NonEmpty[GroundingDiagnostic | ValidationError], ProvenGraph[GroundedPhase]]:
    """Audit ``graph`` against ground-truth snapshots and mint ``ProvenGraph[GroundedPhase]`` when grounded.

    Args:
        graph (LogicalGraph): Candidate computation graph.
        snapshots_path (SnapshotManifestType): Optional snapshot directory, manifest file, or in-memory payload.

    Returns:
        Validation[NonEmpty[GroundingDiagnostic | ValidationError], ProvenGraph[GroundedPhase]]:
            ``Valid`` proven graph when ``report.is_grounded`` holds, else ``Invalid`` non-empty diagnostics.
    """
    cloned = graph.clone()
    failures = _extract_grounding_failures(cloned, snapshots_path)
    proven: ProvenGraph[GroundedPhase] = ProvenGraph.mint(
        cloned,
        ordered_nodes=tuple(cloned.nodes.values()),
    )
    return from_error_sequence(failures, proven)


def upgrade_to_schema_valid(
    proven: ProvenGraph[AcyclicSortedPhase],
    validator: Validator | None = None,
) -> Validation[NonEmpty[ValidationError], ProvenGraph[AcyclicAndSchemaPhase]]:
    """Upgrade ``ProvenGraph[AcyclicSortedPhase]`` to cumulative ``ProvenGraph[AcyclicAndSchemaPhase]``.

    Preserves the existing ``frozen_graph`` and ``ordered_node_ids`` without re-running topological sort.

    Args:
        proven (ProvenGraph[AcyclicSortedPhase]): Acyclic topologically sorted graph witness.
        validator (Validator | None): Optional pre-configured ``Validator`` instance.

    Returns:
        Validation[NonEmpty[ValidationError], ProvenGraph[AcyclicAndSchemaPhase]]:
            ``Valid`` upgraded proof witness or ``Invalid`` non-empty schema errors.
    """
    active_validator = validator if validator is not None else Validator()
    diagnostics = active_validator.validate_graph(proven.thaw())
    fatal = tuple(err for err in diagnostics if err.level == ValidationLevel.ERROR)
    upgraded: ProvenGraph[AcyclicAndSchemaPhase] = ProvenGraph(
        frozen_graph=proven.frozen_graph,
        ordered_node_ids=proven.ordered_node_ids,
    )
    return from_error_sequence(fatal, upgraded)


def upgrade_to_sharding_valid(
    proven: ProvenGraph[AcyclicAndSchemaPhase],
    level: ValidationLevel = ValidationLevel.STRICT,
) -> Validation[NonEmpty[ValidationError], ProvenGraph[AcyclicSchemaShardingPhase]]:
    """Upgrade ``ProvenGraph[AcyclicAndSchemaPhase]`` to ``ProvenGraph[AcyclicSchemaShardingPhase]``.

    Args:
        proven (ProvenGraph[AcyclicAndSchemaPhase]): Acyclic + schema-valid graph witness.
        level (ValidationLevel): Sharding validation strictness level.

    Returns:
        Validation[NonEmpty[ValidationError], ProvenGraph[AcyclicSchemaShardingPhase]]:
            ``Valid`` upgraded proof witness or ``Invalid`` non-empty sharding/pipeline errors.
    """
    candidate = proven.thaw()
    sharding_errs = validate_sharding_propagation_graph(candidate, level=level)
    pipeline_errs = validate_pipeline_and_checkpointing_graph(candidate)
    fatal = tuple(err for err in (*sharding_errs, *pipeline_errs) if err.level == ValidationLevel.ERROR)
    upgraded: ProvenGraph[AcyclicSchemaShardingPhase] = ProvenGraph(
        frozen_graph=proven.frozen_graph,
        ordered_node_ids=proven.ordered_node_ids,
    )
    return from_error_sequence(fatal, upgraded)


def upgrade_to_fully_verified(
    proven: ProvenGraph[AcyclicSchemaShardingPhase],
    snapshots_path: SnapshotManifestType = None,
) -> Validation[NonEmpty[GroundingDiagnostic | ValidationError], ProvenGraph[FullyVerifiedPhase]]:
    """Upgrade ``ProvenGraph[AcyclicSchemaShardingPhase]`` to ``ProvenGraph[FullyVerifiedPhase]``.

    Args:
        proven (ProvenGraph[AcyclicSchemaShardingPhase]): Acyclic + schema + sharding verified graph witness.
        snapshots_path (SnapshotManifestType): Optional snapshot directory, manifest file, or payload.

    Returns:
        Validation[NonEmpty[GroundingDiagnostic | ValidationError], ProvenGraph[FullyVerifiedPhase]]:
            ``Valid`` fully verified proof witness or ``Invalid`` non-empty grounding diagnostics.
    """
    failures = _extract_grounding_failures(proven.thaw(), snapshots_path)
    upgraded: ProvenGraph[FullyVerifiedPhase] = ProvenGraph(
        frozen_graph=proven.frozen_graph,
        ordered_node_ids=proven.ordered_node_ids,
    )
    return from_error_sequence(failures, upgraded)


def prove_fully_verified(
    graph: LogicalGraph,
    validator: Validator | None = None,
    level: ValidationLevel = ValidationLevel.STRICT,
    snapshots_path: SnapshotManifestType = None,
) -> Result[IrDomainError, ProvenGraph[FullyVerifiedPhase]]:
    """Run all 4 verification phases sequentially, minting ``ProvenGraph[FullyVerifiedPhase]``.

    Args:
        graph (LogicalGraph): Candidate computation graph.
        validator (Validator | None): Optional ``Validator`` instance for schema checks.
        level (ValidationLevel): Strictness level for sharding validation.
        snapshots_path (SnapshotManifestType): Optional grounding snapshot source.

    Returns:
        Result[IrDomainError, ProvenGraph[FullyVerifiedPhase]]:
            ``Ok(ProvenGraph[FullyVerifiedPhase])`` or structured ``Err(IrDomainError)``.
    """
    acyclic_res = prove_acyclic_sorted(graph)
    if isinstance(acyclic_res, Err):
        cycle_ids = tuple(NodeId(nid) for nid in graph.nodes)
        return Err(GraphCycleDiagnosticError(cycle_nodes=cycle_ids, message=str(acyclic_res.error)))

    schema_res = upgrade_to_schema_valid(acyclic_res.value, validator=validator)
    if isinstance(schema_res, Invalid):
        return Err(SchemaValidationBatchError(diagnostics=schema_res.errors))

    sharding_res = upgrade_to_sharding_valid(schema_res.value, level=level)
    if isinstance(sharding_res, Invalid):
        return Err(ShardingValidationBatchError(diagnostics=sharding_res.errors))

    grounded_res = upgrade_to_fully_verified(sharding_res.value, snapshots_path=snapshots_path)
    if isinstance(grounded_res, Invalid):
        return Err(GroundingAuditBatchError(diagnostics=grounded_res.errors))

    return Ok(grounded_res.value)
