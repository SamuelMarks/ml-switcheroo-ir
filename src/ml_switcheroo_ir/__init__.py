"""Intermediate Representation (IR).

This module defines the language-agnostic graph data structures used to represent
Deep Learning models after ingestion from source code (e.g. Python/LibCST) or
explicit definition.

It acts as the contract between the Frontend (Ingestion) and the Backend (Synthesis).
"""

from __future__ import annotations

from ml_switcheroo_ir.distributed import (
    DependencyConfig,
    MeshMappingConfig,
    MicrobatchSplittingConfig,
    PipelineScheduleConfig,
    PipelineTopologyConfig,
    PipelineTopologyDict,
    SchedulePhaseConfig,
    StageCommunicationConfig,
    WebRTCPeerConfig,
    WebRTCSignalingTopology,
    WebRTCSignalingTopologyDict,
)
from ml_switcheroo_ir.export import (
    export_schemas,
    export_to_python,
    generate_typescript_definitions,
    get_json_schema,
)
from ml_switcheroo_ir.graph import (
    BaseFrontend,
    CompilerBackend,
    CyclicGraphError,
    EdgeList,
    GraphFrontend,
    LogicalAxis,
    LogicalEdge,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    NodeDict,
    PartitionSpec,
    _load_zstandard,
    topological_sort,
    zstandard,
)
from ml_switcheroo_ir.schema.framework_registries import (
    ARRAY_API_REGISTRY,
    ATEN_REGISTRY,
    ODL_CATALOG,
    get_abstract_op,
)
from ml_switcheroo_ir.sentinels import (
    NoTangent,
    ZeroTangent,
)
from ml_switcheroo_ir.shapes import (
    DimensionType,
    Polynomial,
    ShapeMismatchError,
    ShapeTracker,
    SymBinaryOp,
    SymbolicConstraintTracker,
    SymbolicSolver,
    SymCondition,
    SymConst,
    SymInt,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
    broadcast_dimension,
    broadcast_shapes,
    matmul_shape,
    normalize_axis,
)
from ml_switcheroo_ir.transforms import (
    eliminate_common_subexpressions,
    eliminate_dead_nodes,
    propagate_shapes_and_constants,
)
from ml_switcheroo_ir.translation import ParameterTranslationEngine
from ml_switcheroo_ir.types import (
    AttributeValue,
    AutodiffSentinelProtocol,
    DType,
    InitializerValue,
    ShapeMetadataType,
    TensorShape,
    TensorSpec,
)
from ml_switcheroo_ir.validator import (
    DiagnosticSeverity,
    GroundingDiagnostic,
    GroundingReport,
    estimate_communication_volume,
    estimate_graph_communication_volume,
)

__version__ = "0.0.4"

__all__ = [
    "ARRAY_API_REGISTRY",
    "ATEN_REGISTRY",
    "ODL_CATALOG",
    "AttributeValue",
    "AutodiffSentinelProtocol",
    "BaseFrontend",
    "CompilerBackend",
    "CyclicGraphError",
    "DType",
    "DependencyConfig",
    "DiagnosticSeverity",
    "DimensionType",
    "EdgeList",
    "GraphFrontend",
    "GroundingDiagnostic",
    "GroundingReport",
    "InitializerValue",
    "LogicalAxis",
    "LogicalEdge",
    "LogicalGraph",
    "LogicalMesh",
    "LogicalNode",
    "MeshMappingConfig",
    "MicrobatchSplittingConfig",
    "NoTangent",
    "NodeDict",
    "ParameterTranslationEngine",
    "PartitionSpec",
    "PipelineScheduleConfig",
    "PipelineTopologyConfig",
    "PipelineTopologyDict",
    "Polynomial",
    "SchedulePhaseConfig",
    "ShapeMetadataType",
    "ShapeMismatchError",
    "ShapeTracker",
    "StageCommunicationConfig",
    "SymBinaryOp",
    "SymCondition",
    "SymConst",
    "SymInt",
    "SymNode",
    "SymPiecewise",
    "SymUnaryOp",
    "SymVar",
    "SymbolicConstraintTracker",
    "SymbolicSolver",
    "TensorShape",
    "TensorSpec",
    "WebRTCPeerConfig",
    "WebRTCSignalingTopology",
    "WebRTCSignalingTopologyDict",
    "ZeroTangent",
    "__version__",
    "_load_zstandard",
    "broadcast_dimension",
    "broadcast_shapes",
    "eliminate_common_subexpressions",
    "eliminate_dead_nodes",
    "estimate_communication_volume",
    "estimate_graph_communication_volume",
    "export_schemas",
    "export_to_python",
    "generate_typescript_definitions",
    "get_abstract_op",
    "get_json_schema",
    "matmul_shape",
    "normalize_axis",
    "propagate_shapes_and_constants",
    "topological_sort",
    "zstandard",
]
