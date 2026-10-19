"""Validator facade re-exporting schema, dialect, hardware, distributed, and grounding validators."""

from __future__ import annotations

from ml_switcheroo_ir.schema.ghost import (
    WGSL_COMPUTE_BUILTINS,
    WGSL_MUTATING_OPS,
)
from ml_switcheroo_ir.snapshots import (
    DEFAULT_SNAPSHOT_DIR,
    get_default_snapshots_dir,
    get_default_snapshots_dir as _get_default_snapshots_dir,
)
from ml_switcheroo_ir.validation.hardware import (
    _extract_dst_vgpr,
    _extract_sass_registers,
    _extract_vgpr_indices,
    validate_isa_instruction_node,
    validate_sass_scoreboarding_sequence,
    validate_vopd_pairing,
)
from ml_switcheroo_ir.validation import (
    CORE_MLIR_DIALECTS,
    KNOWN_SYNONYMS,
    CommunicationVolumeReport,
    DiagnosticSeverity,
    GroundingAuditReport,
    GroundingDiagnostic,
    GroundingReport,
    GroundingValidator,
    ValidationError,
    ValidationLevel,
    Validator,
    audit_graph_grounding,
    compute_levenshtein,
    estimate_communication_volume,
    estimate_graph_communication_volume,
    validate_pipeline_and_checkpointing_graph,
    validate_quantization_node,
    validate_sharding_node,
    validate_sharding_propagation_graph,
)

__all__ = [
    "CORE_MLIR_DIALECTS",
    "DEFAULT_SNAPSHOT_DIR",
    "KNOWN_SYNONYMS",
    "WGSL_COMPUTE_BUILTINS",
    "WGSL_MUTATING_OPS",
    "CommunicationVolumeReport",
    "DiagnosticSeverity",
    "GroundingAuditReport",
    "GroundingDiagnostic",
    "GroundingReport",
    "GroundingValidator",
    "ValidationError",
    "ValidationLevel",
    "Validator",
    "_extract_dst_vgpr",
    "_extract_sass_registers",
    "_extract_vgpr_indices",
    "_get_default_snapshots_dir",
    "audit_graph_grounding",
    "compute_levenshtein",
    "estimate_communication_volume",
    "estimate_graph_communication_volume",
    "get_default_snapshots_dir",
    "validate_isa_instruction_node",
    "validate_pipeline_and_checkpointing_graph",
    "validate_quantization_node",
    "validate_sass_scoreboarding_sequence",
    "validate_sharding_node",
    "validate_sharding_propagation_graph",
    "validate_vopd_pairing",
]
