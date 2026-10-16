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
    GroundingReport,
    GroundingSeverity,
    GroundingValidator,
    audit_graph_grounding,
    audit_grounding,
)
from ml_switcheroo_ir.validation.hardware import (
    _extract_dst_vgpr,
    _extract_sass_registers,
    _extract_vgpr_indices,
    validate_isa_instruction_node,
    validate_sass_scoreboarding_sequence,
    validate_vopd_pairing,
)
from ml_switcheroo_ir.validation.quantization import validate_quantization_node
from ml_switcheroo_ir.validation.schema_validator import (
    MultiDialectValidator,
    Validator,
)

__all__ = [
    "CORE_MLIR_DIALECTS",
    "DiagnosticSeverity",
    "GroundingAuditReport",
    "GroundingDiagnostic",
    "GroundingReport",
    "GroundingSeverity",
    "GroundingValidator",
    "KNOWN_SYNONYMS",
    "MultiDialectValidator",
    "ValidationError",
    "ValidationLevel",
    "Validator",
    "_extract_dst_vgpr",
    "_extract_sass_registers",
    "_extract_vgpr_indices",
    "audit_graph_grounding",
    "audit_grounding",
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
