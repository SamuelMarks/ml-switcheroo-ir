"""Schema and validation models for ml_switcheroo_ir."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ml_switcheroo_ir.identifiers import DialectDomain
from ml_switcheroo_ir.schema.base import (
    REGISTRY_LOCK,
    AttributeType,
    DialectCatalogWitness,
    LazyMetadataMap,
    OpSchemaConfig,
    SchemaRegistry,
    describe_attribute_type,
    normalize_attribute_type,
    override_registry,
)
from ml_switcheroo_ir.schema.catalogs import get_catalog_path
from ml_switcheroo_ir.schema.custom_ops import (
    COLLECTIVE_OPS_REGISTRY,
    CUSTOM_OPS_REGISTRY,
    QUANTIZATION_OPS_REGISTRY,
    STATE_OPS_REGISTRY,
    CustomAttributeSchema,
    CustomOpSchema,
    Registry,
)
from ml_switcheroo_ir.schema.framework_registries import (
    ARRAY_API_REGISTRY,
    ATEN_REGISTRY,
    CHEX_REGISTRY,
    GRAIN_REGISTRY,
    ODL_CATALOG,
    ODL_DIALECT_MAPPINGS,
    OPTAX_REGISTRY,
    ORBAX_REGISTRY,
    get_abstract_op,
)
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostInstructionRef,
    GhostIsaRef,
    GhostMlirRef,
    GhostOperationRef,
    GhostOverload,
    GhostParam,
    GhostPythonRef,
    GhostRef,
    GhostResult,
    IRParameterRole,
    LogicConstraint,
    LogicOp,
    OperandDirection,
    ParameterKind,
    SemanticTier,
    SnapshotEnvelope,
    StandardMap,
)
from ml_switcheroo_ir.schema.low_level_registries import (
    METAL_REGISTRY,
    PTX_REGISTRY,
    WASM_REGISTRY,
    WEBGL_REGISTRY,
    WGSL_REGISTRY,
)
from ml_switcheroo_ir.schema.mlir_registry import (
    CORE_MLIR_DIALECT_NAMES,
    MLIR_REGISTRY,
    build_core_mlir_dialects,
    resolve_mlir_op_schema,
)
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


def _make_catalog_witness(
    registry_name: str,
    domains: Sequence[str],
    registry: Mapping[str, OpSchema],
    snapshot_glob: str | None = None,
) -> DialectCatalogWitness[OpSchema]:
    """Construct a frozen ``DialectCatalogWitness[OpSchema]`` for a ``SchemaRegistry``.

    Args:
        registry_name (str): Canonical registry constant name.
        domains (Sequence[str]): Dialect namespace strings routed to this catalog.
        registry (Mapping[str, OpSchema]): Lazy or preloaded operator schema mapping.
        snapshot_glob (str | None): Optional ground-truth snapshot glob pattern.

    Returns:
        DialectCatalogWitness[OpSchema]: Typed catalog witness record.
    """
    return DialectCatalogWitness(
        registry_name=registry_name,
        domains=tuple(DialectDomain(d) for d in domains),
        lookup=registry.get,
        list_ops=lambda: tuple(sorted(registry.keys())),
        snapshot_glob=snapshot_glob,
    )


ALL_DIALECT_CATALOG_WITNESSES: tuple[DialectCatalogWitness[OpSchema], ...] = (
    _make_catalog_witness("ONNX_REGISTRY", ("", "onnx", "ai.onnx"), ONNX_REGISTRY, "onnx*.json"),
    _make_catalog_witness("STABLEHLO_REGISTRY", ("stablehlo",), STABLEHLO_REGISTRY, "stablehlo*.json"),
    _make_catalog_witness(
        "MLIR_REGISTRY",
        ("mlir", *(f"mlir.{d}" for d in CORE_MLIR_DIALECT_NAMES)),
        MLIR_REGISTRY,
        "mlir*.json",
    ),
    _make_catalog_witness("CUSTOM_OPS_REGISTRY", ("ml.switcheroo.custom",), CUSTOM_OPS_REGISTRY),
    _make_catalog_witness(
        "COLLECTIVE_OPS_REGISTRY", ("collective", "ml.switcheroo.collective"), COLLECTIVE_OPS_REGISTRY
    ),
    _make_catalog_witness(
        "QUANTIZATION_OPS_REGISTRY",
        ("quantization", "ml.switcheroo.quant", "ml.switcheroo.quantization"),
        QUANTIZATION_OPS_REGISTRY,
    ),
    _make_catalog_witness("STATE_OPS_REGISTRY", ("state", "ml.switcheroo.state"), STATE_OPS_REGISTRY),
    _make_catalog_witness("ARRAY_API_REGISTRY", ("array_api",), ARRAY_API_REGISTRY, "*array_api*.json"),
    _make_catalog_witness("ATEN_REGISTRY", ("aten",), ATEN_REGISTRY, "*aten*.json"),
    _make_catalog_witness("CHEX_REGISTRY", ("chex",), CHEX_REGISTRY, "chex_*.json"),
    _make_catalog_witness("GRAIN_REGISTRY", ("grain",), GRAIN_REGISTRY, "grain_*.json"),
    _make_catalog_witness(
        "ORBAX_REGISTRY", ("orbax", "orbax_checkpoint", "orbax.checkpoint"), ORBAX_REGISTRY, "orbax*.json"
    ),
    _make_catalog_witness("OPTAX_REGISTRY", ("optax",), OPTAX_REGISTRY, "optax_*.json"),
    _make_catalog_witness("ODL_CATALOG", ("odl", "abstract"), ODL_CATALOG),
    _make_catalog_witness("RDNA_REGISTRY", ("amd_rdna", "rdna"), RDNA_REGISTRY, "*rdna*.json"),
    _make_catalog_witness("SASS_REGISTRY", ("nvidia_sass", "sass"), SASS_REGISTRY, "*sass*.json"),
    _make_catalog_witness("WGSL_REGISTRY", ("webgpu_wgsl", "wgsl"), WGSL_REGISTRY, "wgsl*.json"),
    _make_catalog_witness("PTX_REGISTRY", ("nvidia_ptx", "ptx"), PTX_REGISTRY, "*ptx*.json"),
    _make_catalog_witness("METAL_REGISTRY", ("metal_msl", "metal"), METAL_REGISTRY, "*metal*.json"),
    _make_catalog_witness("WASM_REGISTRY", ("wasm_simd", "wasm"), WASM_REGISTRY, "*wasm*.json"),
    _make_catalog_witness("WEBGL_REGISTRY", ("webgl",), WEBGL_REGISTRY, "*webgl*.json"),
)

__all__ = [
    "ALL_DIALECT_CATALOG_WITNESSES",
    "ARRAY_API_REGISTRY",
    "ATEN_REGISTRY",
    "AttributeType",
    "CHEX_REGISTRY",
    "COLLECTIVE_OPS_REGISTRY",
    "CORE_MLIR_DIALECT_NAMES",
    "CUSTOM_OPS_REGISTRY",
    "DialectCatalogWitness",
    "GRAIN_REGISTRY",
    "LazyMetadataMap",
    "LogicConstraint",
    "LogicOp",
    "METAL_REGISTRY",
    "MLIR_REGISTRY",
    "ODL_CATALOG",
    "ODL_DIALECT_MAPPINGS",
    "ONNX_REGISTRY",
    "OPTAX_REGISTRY",
    "ORBAX_REGISTRY",
    "PTX_REGISTRY",
    "QUANTIZATION_OPS_REGISTRY",
    "RDNA_REGISTRY",
    "RDNA_TO_VOPD_MAP",
    "RDNA_VOPD_SLOTS",
    "REGISTRY_LOCK",
    "SASS_PIPELINE_LATENCIES",
    "SASS_REGISTRY",
    "STABLEHLO_REGISTRY",
    "STATE_OPS_REGISTRY",
    "SchemaRegistry",
    "StandardMap",
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
    "GhostOverload",
    "GhostParam",
    "GhostPythonRef",
    "GhostRef",
    "GhostResult",
    "IRParameterRole",
    "OpAttribute",
    "OpSchema",
    "OpSchemaConfig",
    "OperandDirection",
    "ParameterKind",
    "Registry",
    "SemanticTier",
    "SnapshotEnvelope",
    "build_core_mlir_dialects",
    "describe_attribute_type",
    "get_abstract_op",
    "get_catalog_path",
    "normalize_attribute_type",
    "override_registry",
    "resolve_mlir_op_schema",
]
