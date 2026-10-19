"""Modular validation package for ML-Switcheroo IR."""

from __future__ import annotations

from ml_switcheroo_ir.validation.core import (
    CORE_MLIR_DIALECTS,
    KNOWN_SYNONYMS,
    ValidationError,
    ValidationLevel,
    compute_levenshtein,
)
from ml_switcheroo_ir.validation.distributed import (
    CommunicationVolumeReport,
    estimate_communication_volume,
    estimate_graph_communication_volume,
    validate_pipeline_and_checkpointing_graph,
    validate_sharding_node,
    validate_sharding_propagation_graph,
)
from ml_switcheroo_ir.validation.grounding import (
    DiagnosticSeverity,
    GroundingAuditReport,
    GroundingDiagnostic,
    GroundingEngineProtocol,
    GroundingReport,
    GroundingValidator,
    SnapshotManifestType,
    audit_graph_grounding,
)
from ml_switcheroo_ir.validation.hardware import (
    validate_isa_instruction_node,
    validate_sass_scoreboarding_sequence,
    validate_vopd_pairing,
)
from ml_switcheroo_ir.validation.quantization import (
    validate_quantization_node,
)
from ml_switcheroo_ir.validation.schema_validator import (
    Validator,
)

__all__ = [
    "CORE_MLIR_DIALECTS",
    "CommunicationVolumeReport",
    "DiagnosticSeverity",
    "GroundingAuditReport",
    "GroundingDiagnostic",
    "GroundingEngineProtocol",
    "GroundingReport",
    "GroundingValidator",
    "KNOWN_SYNONYMS",
    "SnapshotManifestType",
    "ValidationError",
    "ValidationLevel",
    "Validator",
    "audit_graph_grounding",
    "compute_levenshtein",
    "estimate_communication_volume",
    "estimate_graph_communication_volume",
    "validate_isa_instruction_node",
    "validate_pipeline_and_checkpointing_graph",
    "validate_quantization_node",
    "validate_sass_scoreboarding_sequence",
    "validate_sharding_node",
    "validate_sharding_propagation_graph",
    "validate_vopd_pairing",
]
