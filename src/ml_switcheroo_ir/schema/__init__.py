"""Schema and validation for ml_switcheroo_ir."""

from ml_switcheroo_ir.schema.custom_ops import (
    CUSTOM_OPS_REGISTRY,
    STATE_OPS_REGISTRY,
    CustomAttributeSchema,
    CustomOpSchema,
    Registry,
)
from ml_switcheroo_ir.schema.framework_registries import (
    ARRAY_API_REGISTRY,
    ATEN_REGISTRY,
    ODL_CATALOG,
    ODL_DIALECT_MAPPINGS,
    get_abstract_op,
)
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostInstructionRef,
    GhostIsaRef,
    GhostMlirRef,
    GhostOperationRef,
    GhostParam,
    GhostPythonRef,
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
from ml_switcheroo_ir.schema.low_level_registries import (
    METAL_REGISTRY,
    PTX_REGISTRY,
    WASM_REGISTRY,
    WEBGL_REGISTRY,
    WGSL_REGISTRY,
)
from ml_switcheroo_ir.schema.mlir_registry import MLIR_REGISTRY
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY, OpAttribute, OpSchema
from ml_switcheroo_ir.schema.rdna_registry import (
    RDNA_REGISTRY,
    RDNA_TO_VOPD_MAP,
    RDNA_VOPD_SLOTS,
)
from ml_switcheroo_ir.schema.sass_registry import (
    SASS_PIPELINE_LATENCIES,
    SASS_REGISTRY,
)
from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY

__all__ = [
    "ARRAY_API_REGISTRY",
    "ATEN_REGISTRY",
    "CUSTOM_OPS_REGISTRY",
    "METAL_REGISTRY",
    "MLIR_REGISTRY",
    "ODL_CATALOG",
    "ODL_DIALECT_MAPPINGS",
    "ONNX_REGISTRY",
    "PTX_REGISTRY",
    "RDNA_REGISTRY",
    "RDNA_TO_VOPD_MAP",
    "RDNA_VOPD_SLOTS",
    "SASS_PIPELINE_LATENCIES",
    "SASS_REGISTRY",
    "STABLEHLO_REGISTRY",
    "STATE_OPS_REGISTRY",
    "WASM_REGISTRY",
    "WEBGL_REGISTRY",
    "WGSL_REGISTRY",
    "CustomAttributeSchema",
    "CustomOpSchema",
    "ExtendedGhostParam",
    "ExtendedGhostRef",
    "GhostInstructionRef",
    "GhostIsaRef",
    "GhostMlirRef",
    "GhostOperationRef",
    "GhostParam",
    "GhostPythonRef",
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
    "get_abstract_op",
    "migrate_ghost_ref",
    "migrate_ghost_ref_v2",
]
