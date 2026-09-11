"""Schema and validation for ml_switcheroo_ir."""

from ml_switcheroo_ir.schema.custom_ops import (
    CUSTOM_OPS_REGISTRY,
    CustomAttributeSchema,
    CustomOpSchema,
    Registry,
)
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostIsaRef,
    GhostMlirRef,
    GhostParam,
    GhostRef,
    GhostResult,
    IRParameterRole,
    OperandDirection,
    ParameterKind,
    SemanticTier,
    SnapshotEnvelope,
    migrate_ghost_ref,
    migrate_ghost_ref_v2,
)
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY, OpAttribute, OpSchema
from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY

__all__ = [
    "CUSTOM_OPS_REGISTRY",
    "ONNX_REGISTRY",
    "STABLEHLO_REGISTRY",
    "CustomAttributeSchema",
    "CustomOpSchema",
    "ExtendedGhostParam",
    "ExtendedGhostRef",
    "GhostIsaRef",
    "GhostMlirRef",
    "GhostParam",
    "GhostRef",
    "GhostResult",
    "IRParameterRole",
    "OpAttribute",
    "OpSchema",
    "OperandDirection",
    "ParameterKind",
    "Registry",
    "SemanticTier",
    "SnapshotEnvelope",
    "migrate_ghost_ref",
    "migrate_ghost_ref_v2",
]
