"""JSON Schema, TypeScript definitions, and Python code generator for ML-Switcheroo IR.

Provides utilities for emitting canonical Draft 2020-12 conforming JSON schemas
for LogicalGraph, LogicalNode, and SnapshotEnvelope, TypeScript interface definitions,
and readable Format B Python IR builder code.
"""

from __future__ import annotations

import ast
import json
import keyword
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pydantic
import typing_extensions

from ml_switcheroo_ir.distributed import (
    LogicalMesh,
    PartitionSpec,
    PipelineTopologyConfig,
    WebRTCSignalingTopology,
)
from ml_switcheroo_ir.graph import LogicalGraph, LogicalNode
from ml_switcheroo_ir.protocols import (
    SymbolicIntWrapperProtocol,
    SymbolicNodeProtocol,
)
from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope
from ml_switcheroo_ir.sentinels import NoTangent, ZeroTangent
from ml_switcheroo_ir.shapes import (
    DimensionType,
    SymBinaryOp,
    SymConst,
    SymExpr,
    SymInt,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
    format_symbolic_protocol_expr,
    is_sym_expr,
)
from ml_switcheroo_ir.typeclasses import PythonExprEmit
from ml_switcheroo_ir.types import (
    AttrBool,
    AttrBytes,
    AttrCompositeTerm,
    AttrDomainTerm,
    AttrDType,
    AttrFloat,
    AttributeValue,
    AttrInt,
    AttrMap,
    AttrModel,
    AttrNone,
    AttrProtocolLeaf,
    AttrScalarTerm,
    AttrSeq,
    AttrShape,
    AttrStr,
    AttrSymInt,
    AttrSymNode,
    AttrTensorSpec,
    AttrTerm,
    DType,
    InitializerValue,
    JsonValue,
    TensorShape,
    TensorSpec,
    _encode_json_safe_protocol_leaf,
    project_attribute_value,
)

__all__ = [
    "ATTRIBUTE_VALUE_PYTHON_EMIT",
    "DIMENSION_PYTHON_EMIT",
    "DTYPE_PYTHON_EMIT",
    "LOGICAL_GRAPH_PYTHON_EMIT",
    "LOGICAL_MESH_PYTHON_EMIT",
    "LOGICAL_NODE_PYTHON_EMIT",
    "PARTITION_SPEC_PYTHON_EMIT",
    "SYM_EXPR_PYTHON_EMIT",
    "TENSOR_SHAPE_PYTHON_EMIT",
    "TENSOR_SPEC_PYTHON_EMIT",
    "export_schemas",
    "export_to_python",
    "generate_typescript_definitions",
    "get_json_schema",
]

DRAFT_2020_12_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_NL = chr(10)
_BS = chr(92)

_MODEL_MAP: dict[str, type[pydantic.BaseModel]] = {
    "LogicalGraph": LogicalGraph,
    "LogicalNode": LogicalNode,
    "SnapshotEnvelope": SnapshotEnvelope,
    "ZeroTangent": ZeroTangent,
    "NoTangent": NoTangent,
    "PipelineTopologyConfig": PipelineTopologyConfig,
    "WebRTCSignalingTopology": WebRTCSignalingTopology,
}


def get_json_schema(target: str = "all") -> dict[str, JsonValue]:
    """Generate JSON Schema (Draft 2020-12) for one or all IR models.

    Args:
        target (str): Target model name ('LogicalGraph', 'LogicalNode',
            'SnapshotEnvelope', 'ZeroTangent', 'NoTangent', 'PipelineTopologyConfig',
            'WebRTCSignalingTopology') or 'all'.

    Returns:
        dict[str, JsonValue]: JSON Schema dictionary for the requested model, or a mapping
            of model names to their JSON Schemas when target == 'all'.

    Raises:
        ValueError: If an unsupported target name is requested.
    """
    if target == "all":
        schemas: dict[str, JsonValue] = {}
        for name, model_cls in _MODEL_MAP.items():
            schema = model_cls.model_json_schema()
            schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
            schemas[name] = schema
        return schemas

    if target not in _MODEL_MAP:
        valid = ", ".join([*_MODEL_MAP.keys(), "all"])
        raise ValueError(f"Unsupported schema target '{target}'. Valid targets: {valid}")

    model_cls = _MODEL_MAP[target]
    schema = model_cls.model_json_schema()
    schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
    return schema


def generate_typescript_definitions() -> str:
    """Generate clean, idiomatic TypeScript (.d.ts) interfaces matching IR Pydantic models.

    Synchronizes literal ``DType`` union values directly from ``ml_switcheroo_ir.types.DType``,
    types ``PartitionSpec`` axes with ``string | number | null`` and nested arrays, and
    aligns ``LogicalGraph``, ``LogicalNode``, and ``AttributeValue`` with runtime schemas.

    Returns:
        str: TypeScript declaration file content.
    """
    dtype_literals = _NL.join(f'  | "{dt.value}"' for dt in DType) + ";"

    return f"""/**
 * Auto-generated TypeScript definitions for ML-Switcheroo IR.
 * Conforms to JSON Schema Draft 2020-12 exports.
 */

export type DType =
{dtype_literals}

export interface TensorShape {{
  dims: DimensionType[];
}}

export type AttributeValue =
  | string
  | number
  | boolean
  | Uint8Array
  | null
  | DType
  | SymNode
  | SymInt
  | TensorShape
  | TensorSpec
  | AttributeValue[]
  | {{ [key: string]: AttributeValue }};

export interface WebRTCPeerConfig {{
  peer_id: string;
  signaling_role?: string;
  ice_candidates?: string[];
  device_capability?: string;
  network_transport?: string;
}}

export interface WebRTCSignalingTopology {{
  signaling_url: string;
  ice_servers?: string[];
  peers?: WebRTCPeerConfig[];
  connectivity_matrix?: number[][];
  ordered?: boolean;
  max_retransmits?: number | null;
  chunk_size_bytes?: number;
  high_watermark_bytes?: number;
  low_watermark_bytes?: number;
  dtls_fingerprint_sha256?: string | null;
  sctp_max_message_size?: number;
}}

export interface LogicalMesh {{
  shape: Record<string, number>;
  webrtc_topology?: WebRTCSignalingTopology | null;
}}

export type PartitionAxisAtom = string | number | null;
export type PartitionAxisSubTuple = PartitionAxisAtom[];
export type PartitionAxisElement =
  | PartitionAxisAtom
  | (PartitionAxisAtom | PartitionAxisSubTuple)[];

export interface PartitionSpec {{
  axes: PartitionAxisElement[];
}}

export interface MicrobatchSplittingConfig {{
  strategy?: string;
  num_microbatches: number;
  chunk_axis?: number;
}}

export interface MeshMappingConfig {{
  pipeline_axis?: string;
  stage_assignment?: string;
  devices_per_stage: number;
}}

export interface StageCommunicationConfig {{
  protocol: string;
  SendOp?: string;
  RecvOp?: string;
}}

export interface DependencyConfig {{
  source_stage: string;
  target_stage: string;
  offset_mb?: number;
}}

export interface SchedulePhaseConfig {{
  type: string;
  operations?: string[];
  count_expression?: string;
}}

export interface PipelineScheduleConfig {{
  phases?: SchedulePhaseConfig[];
}}

export interface PipelineTopologyConfig {{
  microbatch_splitting: MicrobatchSplittingConfig;
  mesh_mapping: MeshMappingConfig;
  stage_communication: StageCommunicationConfig;
  dependencies?: DependencyConfig[];
  schedule?: PipelineScheduleConfig | null;
}}

export interface SymVar {{
  type: "var";
  name: string;
}}

export interface SymConst {{
  type: "const";
  value: number;
}}

export interface SymBinaryOp {{
  type: "binary_op";
  op: string;
  left: SymNode;
  right: SymNode;
}}

export interface SymUnaryOp {{
  type: "unary_op";
  op: string;
  operand: SymNode;
}}

export interface SymPiecewise {{
  type: "piecewise";
  cases: [string | boolean | SymNode, SymNode][];
  default: SymNode;
}}

export type SymNode =
  | SymVar
  | SymConst
  | SymBinaryOp
  | SymUnaryOp
  | SymPiecewise;

export interface SymInt {{
  node: SymNode;
}}

export type DimensionType = number | string | SymNode | SymInt;

export interface TensorSpec {{
  shape: DimensionType[];
  dtype: DType;
  sparsity?: string | null;
}}

export interface LogicalEdge {{
  source: string;
  target: string;
  source_idx?: number;
  target_idx?: number;
  value_name?: string | null;
}}

export interface LogicalNode {{
  id: string;
  op_type?: string;
  domain?: string;
  version?: number;
  attributes?: Record<string, AttributeValue>;
  inputs?: string[];
  outputs?: string[];
  shape_metadata?: DimensionType[] | null;
  source_ast_ref?: string | number | null;
  sharding?: PartitionSpec | string | Record<string, AttributeValue> | null;
  dtype?: DType | null;
  output_specs?: TensorSpec[];
  subgraphs?: Record<string, LogicalGraph | AttributeValue>;
  device?: string | null;
  stream?: string | number | null;
}}

export interface ZeroTangent extends LogicalNode {{
  op_type: "ZeroTangent";
  domain: "ml.switcheroo.ad";
}}

export interface NoTangent extends LogicalNode {{
  op_type: "NoTangent";
  domain: "ml.switcheroo.ad";
}}

export interface LogicalGraph {{
  name?: string;
  nodes?: Record<string, LogicalNode>;
  inputs?: string[];
  input_specs?: Record<string, TensorSpec>;
  outputs?: string[];
  initializers?: Record<string, AttributeValue>;
  mesh?: LogicalMesh | AttributeValue | null;
  pipeline_topology?: PipelineTopologyConfig | null;
  attributes?: Record<string, AttributeValue>;
  annotations?: Record<string, AttributeValue>;
  metadata?: Record<string, AttributeValue>;
  sorted_nodes?: LogicalNode[] | null;
  loop_options?: Record<string, AttributeValue> | null;
  edges?: LogicalEdge[];
}}

export interface GhostParam {{
  name: string;
  kind?: string;
  has_default?: boolean;
  default?: string | null;
  annotation?: string | null;
}}

export interface GhostControlCode {{
  stall_count?: number | null;
  yield_flag?: boolean;
  write_barrier?: number | null;
  read_barrier?: number | null;
  wait_barrier_mask?: number;
  reuse_flags?: Record<string, boolean>;
}}

export interface GhostIsaRef {{
  mnemonic: string;
  architecture?: string;
  isa_family?: string;
  instruction_category?: string;
  opcode_hex?: string | null;
  encoding_width_bits?: number;
  operands?: string[];
  instruction_modifiers?: string[];
  allowed_modifiers?: string[];
  pipeline_unit?: string | null;
  execution_latency?: number | null;
  throughput_cycles?: number | null;
  vopd_slot?: string | null;
  vopd_profile?: string | null;
  register_alignment?: number;
  max_vgpr_count?: number | null;
  vgpr_granularity?: number;
  control_codes?: GhostControlCode | null;
  scoreboard_barrier_mask?: number | null;
  predicate_support?: boolean;
  description?: string | null;
}}

export interface GhostRef {{
  name: string;
  api_path: string;
  kind: string;
  params?: GhostParam[];
  return_annotation?: string | null;
  docstring?: string | null;
  deprecated?: boolean;
  experimental?: boolean;
  members?: GhostRef[];
  isa_metadata?: GhostIsaRef | null;
}}

export interface SnapshotEnvelope {{
  schema_version?: string;
  target: string;
  version: string;
  upstream_version?: string | null;
  source_type: string;
  upstream_commit?: string | null;
  supported_microarchitectures?: string[] | null;
  generated_at: string;
  environment?: Record<string, string>;
  categories?: Record<string, unknown>;
  operations?: Record<string, GhostRef> | unknown[] | null;
  instructions?: Record<string, GhostIsaRef> | unknown[] | null;
}}
"""


def _format_sym_expr_python(expr: SymExpr, prefix: str = "sw_ir.") -> str:
    """Exhaustively emit Format-B Python constructor code for a closed ``SymExpr``.

    Args:
        expr (SymExpr): Closed symbolic expression node.
        prefix (str): Module qualification prefix (defaults to ``'sw_ir.'``).

    Returns:
        str: Executable Python constructor expression string.
    """
    code: str
    if isinstance(expr, SymVar):
        code = f"{prefix}SymVar({expr.name!r})"
    elif isinstance(expr, SymConst):
        code = f"{prefix}SymConst({expr.value!r})"
    elif isinstance(expr, SymBinaryOp):
        code = (
            f"{prefix}SymBinaryOp({expr.op!r}, "
            f"{_format_sym_node(expr.left, prefix)}, {_format_sym_node(expr.right, prefix)})"
        )
    elif isinstance(expr, SymUnaryOp):
        code = f"{prefix}SymUnaryOp({expr.op!r}, {_format_sym_node(expr.operand, prefix)})"
    elif isinstance(expr, SymPiecewise):
        cases_items: list[str] = []
        for cond, branch in expr.cases:
            cond_str = _format_sym_node(cond, prefix) if isinstance(cond, SymNode) else repr(cond)
            cases_items.append(f"({cond_str}, {_format_sym_node(branch, prefix)})")
        cases_str = "[" + ", ".join(cases_items) + "]"
        code = f"{prefix}SymPiecewise(cases={cases_str}, default={_format_sym_node(expr.default, prefix)})"
    else:
        typing_extensions.assert_never(expr)
    return code


def _format_sym_node(val: DimensionType, prefix: str = "sw_ir.") -> str:
    """Format a DimensionType or SymNode into clean Python code representation.

    Args:
        val (DimensionType): Dimension or symbolic node.
        prefix (str): Module qualification prefix (defaults to ``'sw_ir.'``).

    Returns:
        str: Executable Python expression string.
    """
    if is_sym_expr(val):
        return _format_sym_expr_python(val, prefix)
    if isinstance(val, SymInt):
        return f"{prefix}SymInt({_format_sym_node(val.node, prefix)})"
    return (
        repr(format_symbolic_protocol_expr(val))
        if isinstance(val, (SymbolicNodeProtocol, SymbolicIntWrapperProtocol))
        else repr(val)
    )


def _format_shape_tuple(
    shape: Sequence[DimensionType] | TensorShape | None,
    prefix: str = "sw_ir.",
) -> str:
    """Format shape metadata or dimension tuple into Python representation.

    Args:
        shape (Sequence[DimensionType] | TensorShape | None): Shape tuple, list, or None.
        prefix (str): Module qualification prefix.

    Returns:
        str: Formatted Python tuple expression or 'None'.
    """
    if shape is None:
        return "None"
    dims = shape.dims if isinstance(shape, TensorShape) else tuple(shape)
    if len(dims) == 0:
        return "()"
    if len(dims) == 1:
        return f"({_format_sym_node(dims[0], prefix)},)"
    items = ", ".join(_format_sym_node(d, prefix) for d in dims)
    return f"({items})"


def _format_tensor_spec(spec: TensorSpec, prefix: str = "sw_ir.") -> str:
    """Format a TensorSpec instance into Python constructor code.

    Args:
        spec (TensorSpec): TensorSpec instance.
        prefix (str): Module qualification prefix.

    Returns:
        str: Python code constructing TensorSpec.
    """
    dtype_str = f"{prefix}DType.{spec.dtype.name}"
    shape_str = _format_shape_tuple(spec.shape, prefix)
    if spec.sparsity is not None:
        return f"{prefix}TensorSpec(shape={shape_str}, dtype={dtype_str}, sparsity={spec.sparsity!r})"
    return f"{prefix}TensorSpec(shape={shape_str}, dtype={dtype_str})"


def _format_model_attr_value(val: pydantic.BaseModel, prefix: str = "sw_ir.") -> str:
    """Format a ``pydantic.BaseModel`` leaf in ``AttributeValue`` into Python code.

    Args:
        val (pydantic.BaseModel): Pydantic model attribute value (excluding ``TensorShape`` and ``TensorSpec``).
        prefix (str): Module qualification prefix.

    Returns:
        str: Executable Python constructor or dictionary literal expression.
    """
    if isinstance(val, LogicalGraph):
        return _format_graph_expr(val, prefix)
    if isinstance(val, PartitionSpec):
        return f"{prefix}PartitionSpec(axes={val.axes!r})"
    return repr(val.model_dump(mode="json"))


def _emit_scalar_attr_term_python(term: AttrScalarTerm) -> str:
    """Exhaustively format an ``AttrScalarTerm`` into a Python literal expression.

    Args:
        term (AttrScalarTerm): Closed scalar attribute term.

    Returns:
        str: Valid Python literal string.
    """
    code: str
    if isinstance(term, AttrNone):
        code = "None"
    elif isinstance(term, AttrBool):
        code = repr(term.value)
    elif isinstance(term, AttrInt):
        code = repr(term.value)
    elif isinstance(term, AttrFloat):
        code = repr(term.value)
    elif isinstance(term, AttrStr):
        code = repr(term.value)
    elif isinstance(term, AttrBytes):
        code = repr(term.value)
    else:
        typing_extensions.assert_never(term)
    return code


def _emit_domain_attr_term_python(term: AttrDomainTerm, prefix: str) -> str:
    """Exhaustively format an ``AttrDomainTerm`` into a Format-B Python constructor expression.

    Args:
        term (AttrDomainTerm): Closed domain/model/protocol attribute term.
        prefix (str): Module alias prefix (e.g. ``'sw_ir.'``).

    Returns:
        str: Executable Python constructor expression.
    """
    code: str
    if isinstance(term, AttrDType):
        code = f"{prefix}DType.{term.value.name}"
    elif isinstance(term, AttrShape):
        code = f"{prefix}TensorShape({_format_shape_tuple(term.value.dims, prefix)})"
    elif isinstance(term, AttrTensorSpec):
        code = _format_tensor_spec(term.value, prefix)
    elif isinstance(term, AttrSymNode):
        code = _format_sym_node(term.value, prefix)
    elif isinstance(term, AttrSymInt):
        code = _format_sym_node(term.value, prefix)
    elif isinstance(term, AttrModel):
        code = _format_model_attr_value(term.value, prefix)
    elif isinstance(term, AttrProtocolLeaf):
        code = repr(_encode_json_safe_protocol_leaf(term.value))
    else:
        typing_extensions.assert_never(term)
    return code


def _emit_composite_attr_term_python(term: AttrCompositeTerm, prefix: str) -> str:
    """Exhaustively format an ``AttrCompositeTerm`` into a Python tuple, list, or dict expression.

    Args:
        term (AttrCompositeTerm): Closed sequence or mapping term.
        prefix (str): Module alias prefix.

    Returns:
        str: Executable Python container literal expression.
    """
    code: str
    if isinstance(term, AttrSeq):
        rendered_items = ", ".join(_emit_attr_term_python(item, prefix) for item in term.items)
        if term.is_tuple:
            if len(term.items) == 0:
                code = "()"
            elif len(term.items) == 1:
                code = f"({rendered_items},)"
            else:
                code = f"({rendered_items})"
        else:
            code = f"[{rendered_items}]"
    elif isinstance(term, AttrMap):
        sorted_entries = sorted(term.entries, key=lambda kv: kv[0])
        entries = ", ".join(f"{k!r}: {_emit_attr_term_python(v, prefix)}" for k, v in sorted_entries)
        code = "{" + entries + "}"
    else:
        typing_extensions.assert_never(term)
    return code


def _emit_attr_term_python(term: AttrTerm, prefix: str = "sw_ir.") -> str:
    """Exhaustively format a closed ``AttrTerm`` into Python code with ``assert_never``.

    Args:
        term (AttrTerm): Closed normal-form attribute term.
        prefix (str): Module alias prefix.

    Returns:
        str: Executable Python expression string.
    """
    code: str
    if isinstance(term, (AttrNone, AttrBool, AttrInt, AttrFloat, AttrStr, AttrBytes)):
        code = _emit_scalar_attr_term_python(term)
    elif isinstance(
        term,
        (AttrDType, AttrShape, AttrTensorSpec, AttrSymNode, AttrSymInt, AttrModel, AttrProtocolLeaf),
    ):
        code = _emit_domain_attr_term_python(term, prefix)
    elif isinstance(term, (AttrSeq, AttrMap)):
        code = _emit_composite_attr_term_python(term, prefix)
    else:
        typing_extensions.assert_never(term)
    return code


def _format_attr_value(val: AttributeValue, prefix: str = "sw_ir.") -> str:
    """Format an ``AttributeValue`` recursively into an executable Python expression via ``AttrTerm``.

    Args:
        val (AttributeValue): Attribute value (primitives, bytes, DType, SymNode, SymInt,
            TensorShape, TensorSpec, models, protocols, sequences, or mappings).
        prefix (str): Module qualification prefix (defaults to ``'sw_ir.'``).

    Returns:
        str: Executable Python expression reconstructing ``val`` losslessly.
    """
    return _emit_attr_term_python(project_attribute_value(val), prefix)


def _format_initializer_value(val: InitializerValue) -> str:
    """Format an ``InitializerValue`` recursively into an executable Python expression.

    Args:
        val (InitializerValue): Initializer constant payload.

    Returns:
        str: Executable Python expression reconstructing ``val``.
    """
    from ml_switcheroo_ir.protocols import ArrayBufferProtocol

    if isinstance(val, (bool, int, float, str, bytes)):
        return repr(val)
    if isinstance(val, ArrayBufferProtocol):
        return repr(val.tobytes())
    if isinstance(val, Mapping):
        entries = ", ".join(
            f"{str(k)!r}: {_format_initializer_value(v)}" for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + entries + "}"
    return "[" + ", ".join(_format_initializer_value(x) for x in val) + "]"


def _sanitize_python_identifier(name: str, fallback: str = "build_graph") -> str:
    """Sanitize a string into a valid, non-keyword Python function identifier.

    Args:
        name (str): Candidate identifier string.
        fallback (str): Default identifier when sanitized result is empty.

    Returns:
        str: Valid Python identifier.
    """
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip()).strip("_")
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def _emit_logical_mesh_python(mesh: LogicalMesh, prefix: str) -> str:
    """Emit Python constructor expression for ``LogicalMesh``.

    Args:
        mesh (LogicalMesh): Logical device mesh.
        prefix (str): Module qualification prefix.

    Returns:
        str: Executable Python ``LogicalMesh(...)`` expression.
    """
    shape_str = json.dumps(mesh.shape, sort_keys=True)
    if mesh.webrtc_topology is not None:
        webrtc_dict = mesh.webrtc_topology.to_dict()
        return (
            f"{prefix}LogicalMesh(shape={shape_str}, "
            f"webrtc_topology={prefix}WebRTCSignalingTopology.from_dict({webrtc_dict!r}))"
        )
    return f"{prefix}LogicalMesh(shape={shape_str})"


def _format_mesh_expr(graph: LogicalGraph, prefix: str = "sw_ir.") -> str:
    """Format ``graph.mesh`` into an inline Python constructor expression via ``LOGICAL_MESH_PYTHON_EMIT``.

    Args:
        graph (LogicalGraph): Graph containing optional ``mesh``.
        prefix (str): Module qualification prefix (defaults to ``'sw_ir.'``).

    Returns:
        str: Python expression constructing ``LogicalMesh`` or mesh literal.
    """
    from ml_switcheroo_ir.protocols import MeshLikeProtocol

    if graph.mesh is None:
        return "None"
    if isinstance(graph.mesh, LogicalMesh):
        return LOGICAL_MESH_PYTHON_EMIT.emit_python(graph.mesh, prefix)
    if isinstance(graph.mesh, MeshLikeProtocol):
        return repr([str(a) for a in graph.mesh.axis_names])
    return ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(graph.mesh, prefix)


def _format_graph_expr(graph: LogicalGraph, prefix: str = "sw_ir.") -> str:
    """Format a ``LogicalGraph`` into an inline ``LogicalGraph(...)`` constructor expression.

    Emits pure Python constructors for all nested nodes, rich attributes, DTypes,
    symbolic dimensions, topologies, and subgraphs without ``to_dict()!r`` enum repr leaks.

    Args:
        graph (LogicalGraph): Graph or nested subgraph to render.
        prefix (str): Module qualification prefix (defaults to ``'sw_ir.'``).

    Returns:
        str: Valid Python expression constructing the ``LogicalGraph``.
    """
    nodes_entries = [
        f"{nid!r}: {LOGICAL_NODE_PYTHON_EMIT.emit_python(node, prefix)}" for nid, node in graph.nodes.items()
    ]
    nodes_expr = "{" + ", ".join(nodes_entries) + "}"
    specs_expr = (
        "{"
        + ", ".join(
            f"{k!r}: {TENSOR_SPEC_PYTHON_EMIT.emit_python(spec, prefix)}"
            for k, spec in sorted(graph.input_specs.items())
        )
        + "}"
    )
    inits_expr = (
        "{"
        + ", ".join(f"{k!r}: {_format_initializer_value(val)}" for k, val in sorted(graph.initializers.items()))
        + "}"
    )
    mesh_expr = _format_mesh_expr(graph, prefix)
    pipe_expr = (
        f"{prefix}PipelineTopologyConfig.from_dict({graph.pipeline_topology.to_dict()!r})"
        if graph.pipeline_topology is not None
        else "None"
    )
    attrs_expr = ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(graph.attributes, prefix)
    annotations_expr = ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(graph.annotations, prefix)
    metadata_expr = ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(graph.metadata, prefix)
    loop_expr = (
        ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(graph.loop_options, prefix)
        if graph.loop_options is not None
        else "None"
    )
    edges_expr = _format_edges_expr(graph, prefix)
    edges_kw = f", edges={edges_expr}" if edges_expr is not None else ""
    return (
        f"{prefix}LogicalGraph("
        f"name={graph.name!r}, "
        f"nodes={nodes_expr}, "
        f"inputs={list(graph.inputs)!r}, "
        f"input_specs={specs_expr}, "
        f"outputs={list(graph.outputs)!r}, "
        f"initializers={inits_expr}, "
        f"mesh={mesh_expr}, "
        f"pipeline_topology={pipe_expr}, "
        f"attributes={attrs_expr}, "
        f"annotations={annotations_expr}, "
        f"metadata={metadata_expr}, "
        f"loop_options={loop_expr}"
        f"{edges_kw})"
    )


def _format_edges_expr(graph: LogicalGraph, prefix: str = "sw_ir.") -> str | None:
    """Format explicit ``edges`` constructor list when edge ``value_name`` metadata is present.

    Args:
        graph (LogicalGraph): Graph to inspect.
        prefix (str): Module qualification prefix (defaults to ``'sw_ir.'``).

    Returns:
        str | None: Python list literal of ``LogicalEdge(...)`` calls, or ``None`` if unneeded.
    """
    if not graph._edge_value_names:
        return None
    edge_items = [
        f"{prefix}LogicalEdge(source={e.source!r}, target={e.target!r}, "
        f"source_idx={e.source_idx!r}, target_idx={e.target_idx!r}, value_name={e.value_name!r})"
        for e in graph.iter_edges()
    ]
    return "[" + ", ".join(edge_items) + "]"


def _format_subgraphs_expr(subgraphs: Mapping[str, LogicalGraph | AttributeValue], prefix: str = "sw_ir.") -> str:
    """Format a node's ``subgraphs`` dictionary into a Python dict literal expression.

    Args:
        subgraphs (Mapping[str, LogicalGraph | AttributeValue]): Node subgraphs mapping.
        prefix (str): Module qualification prefix (defaults to ``'sw_ir.'``).

    Returns:
        str: Formatted Python dictionary expression.
    """
    if not subgraphs:
        return "{}"
    entries: list[str] = []
    for k, v in sorted(subgraphs.items()):
        val_expr = (
            LOGICAL_GRAPH_PYTHON_EMIT.emit_python(v, prefix)
            if isinstance(v, LogicalGraph)
            else ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(v, prefix)
        )
        entries.append(f"{k!r}: {val_expr}")
    return "{" + ", ".join(entries) + "}"


def _format_node_ctor_lines(node: LogicalNode, prefix: str = "sw_ir.") -> list[str]:
    """Build unindented constructor lines for ``LogicalNode``, ``ZeroTangent``, or ``NoTangent``.

    Args:
        node (LogicalNode): Node or autodiff sentinel to render.
        prefix (str): Module qualification prefix (defaults to ``'sw_ir.'``).

    Returns:
        list[str]: Multi-line constructor expression lines ``["LogicalNode(", "    id=...", ..., ")"]``.
    """
    from ml_switcheroo_ir.protocols import DeviceLikeProtocol

    is_zero = isinstance(node, ZeroTangent)
    is_no = isinstance(node, NoTangent)
    ctor = "ZeroTangent" if is_zero else ("NoTangent" if is_no else "LogicalNode")

    if isinstance(node.sharding, PartitionSpec):
        sharding_str = PARTITION_SPEC_PYTHON_EMIT.emit_python(node.sharding, prefix)
    elif node.sharding is not None:
        sharding_str = ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(node.sharding, prefix)
    else:
        sharding_str = "None"
    device_val: str | None = (
        (f"{node.device.type}:{node.device.index}" if node.device.index is not None else node.device.type)
        if isinstance(node.device, DeviceLikeProtocol)
        else node.device
    )
    dtype_str = DTYPE_PYTHON_EMIT.emit_python(node.dtype, prefix) if node.dtype is not None else "None"
    shape_str = _format_shape_tuple(node.shape_metadata, prefix)
    out_specs_str = "[" + ", ".join(TENSOR_SPEC_PYTHON_EMIT.emit_python(s, prefix) for s in node.output_specs) + "]"
    attrs_str = ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(node.attributes, prefix)
    subgraphs_str = _format_subgraphs_expr(node.subgraphs, prefix)

    lines: list[str] = [
        f"{prefix}{ctor}(",
        f"    id={node.id!r},",
    ]
    if not (is_zero or is_no):
        lines.append(f"    op_type={node.op_type!r},")
        lines.append(f"    domain={node.domain!r},")
    lines.extend(
        [
            f"    version={node.version!r},",
            f"    attributes={attrs_str},",
            f"    inputs={list(node.inputs)!r},",
            f"    outputs={list(node.outputs)!r},",
            f"    shape_metadata={shape_str},",
            f"    source_ast_ref={node.source_ast_ref!r},",
            f"    sharding={sharding_str},",
            f"    dtype={dtype_str},",
            f"    output_specs={out_specs_str},",
            f"    subgraphs={subgraphs_str},",
            f"    device={device_val!r},",
            f"    stream={node.stream!r},",
        ]
    )
    if node.strides is not None:
        lines.append(f"    strides={node.strides!r},")
    lines.append(")")
    return lines


def _format_node_entry_lines(nid: str, node: LogicalNode, prefix: str = "sw_ir.") -> list[str]:
    """Generate indented source lines for a single ``nodes`` entry in ``export_to_python``.

    Args:
        nid (str): Node key in ``graph.nodes``.
        node (LogicalNode): Node instance to format.
        prefix (str): Module qualification prefix (defaults to ``'sw_ir.'``).

    Returns:
        list[str]: Indented Python source lines for this dictionary item.
    """
    node_expr = LOGICAL_NODE_PYTHON_EMIT.emit_python(node, prefix)
    ctor_lines = node_expr.splitlines()
    out = [f"        {nid!r}: {ctor_lines[0]}"]
    for middle in ctor_lines[1:-1]:
        out.append(f"        {middle}")
    out.append(f"        {ctor_lines[-1]},")
    return out


def _append_topology_lines(lines: list[str], graph: LogicalGraph) -> tuple[str, str]:
    """Append mesh and pipeline topology builder statements to ``lines``.

    Args:
        lines (list[str]): Output statement lines list.
        graph (LogicalGraph): Source graph.

    Returns:
        tuple[str, str]: ``(mesh_var, pipeline_var)`` variable expressions.
    """
    from ml_switcheroo_ir.protocols import MeshLikeProtocol

    mesh_var = "None"
    if isinstance(graph.mesh, LogicalMesh):
        shape_str = json.dumps(graph.mesh.shape, sort_keys=True)
        if graph.mesh.webrtc_topology is not None:
            webrtc_dict = graph.mesh.webrtc_topology.to_dict()
            lines.append(f"    webrtc_topology = sw_ir.WebRTCSignalingTopology.from_dict({webrtc_dict!r})")
            lines.append(f"    mesh = sw_ir.LogicalMesh(shape={shape_str}, webrtc_topology=webrtc_topology)")
        else:
            lines.append(f"    mesh = sw_ir.LogicalMesh(shape={shape_str})")
        mesh_var = "mesh"
    elif isinstance(graph.mesh, MeshLikeProtocol):
        mesh_var = repr([str(a) for a in graph.mesh.axis_names])
    elif graph.mesh is not None:
        mesh_var = _format_attr_value(graph.mesh)

    pipeline_var = "None"
    if graph.pipeline_topology is not None:
        pipe_dict = graph.pipeline_topology.to_dict()
        lines.append(f"    pipeline_topology = sw_ir.PipelineTopologyConfig.from_dict({pipe_dict!r})")
        pipeline_var = "pipeline_topology"

    return mesh_var, pipeline_var


def _format_python_source(raw_code: str) -> str:
    """Format generated Python source using ``black`` if installed, falling back to ``ast.unparse``.

    Args:
        raw_code (str): Unformatted valid Python source code.

    Returns:
        str: Formatted Python source code ending with a newline.
    """
    try:
        import black

        return black.format_str(raw_code, mode=black.Mode())
    except ImportError:
        return ast.unparse(ast.parse(raw_code)) + _NL


def export_to_python(
    graph: LogicalGraph,
    function_name: str = "build_graph",
    standalone: bool = False,
) -> str:
    """Export a LogicalGraph into formatted, executable Python IR builder source code.

    Generates readable Python code (Format B) that reconstructs the entire LogicalGraph,
    including nodes, subgraphs, symbolic shapes, sharding, Sentinels, Mesh, WebRTC,
    Pipeline topologies, graph attributes, and loop_options.

    Args:
        graph (LogicalGraph): The LogicalGraph instance to export.
        function_name (str): Name of the generated builder function (default: 'build_graph').
        standalone (bool): If True, appends a ``__main__`` execution block printing graph JSON.

    Returns:
        str: Valid, formatted Python source code.
    """
    safe_fn_name = _sanitize_python_identifier(function_name)
    safe_doc_title = graph.name.replace(_BS, _BS * 2).replace('"""', (_BS + '"') * 3).replace(_NL, " ")
    lines: list[str] = [
        '"""Auto-generated ML-Switcheroo IR Python graph definition."""',
        "",
        "from __future__ import annotations",
        "",
        "import ml_switcheroo_ir as sw_ir",
        "",
        "",
        f"def {safe_fn_name}() -> sw_ir.LogicalGraph:",
        f'    """Construct LogicalGraph {safe_doc_title!r}."""',
    ]

    mesh_var, pipeline_var = _append_topology_lines(lines, graph)

    lines.append("    nodes: dict[str, sw_ir.LogicalNode] = {")
    for nid, node in graph.nodes.items():
        lines.extend(_format_node_entry_lines(nid, node))
    lines.append("    }")

    lines.append("    input_specs: dict[str, sw_ir.TensorSpec] = {")
    for k, spec in sorted(graph.input_specs.items()):
        lines.append(f"        {k!r}: {TENSOR_SPEC_PYTHON_EMIT.emit_python(spec, 'sw_ir.')},")
    lines.append("    }")

    lines.append("    initializers: dict[str, sw_ir.InitializerValue] = {")
    for k, val in sorted(graph.initializers.items()):
        lines.append(f"        {k!r}: {_format_initializer_value(val)},")
    lines.append("    }")

    attrs_expr = ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(graph.attributes, "sw_ir.")
    annotations_expr = ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(graph.annotations, "sw_ir.")
    metadata_expr = ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(graph.metadata, "sw_ir.")
    loop_expr = (
        ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python(graph.loop_options, "sw_ir.")
        if graph.loop_options is not None
        else "None"
    )
    edges_expr = _format_edges_expr(graph, "sw_ir.")

    lines.extend(
        [
            "    return sw_ir.LogicalGraph(",
            f"        name={graph.name!r},",
            "        nodes=nodes,",
            f"        inputs={list(graph.inputs)!r},",
            "        input_specs=input_specs,",
            f"        outputs={list(graph.outputs)!r},",
            "        initializers=initializers,",
            f"        mesh={mesh_var},",
            f"        pipeline_topology={pipeline_var},",
            f"        attributes={attrs_expr},",
            f"        annotations={annotations_expr},",
            f"        metadata={metadata_expr},",
            f"        loop_options={loop_expr},",
        ]
    )
    if edges_expr is not None:
        lines.append(f"        edges={edges_expr},")
    lines.append("    )")

    if standalone:
        lines.extend(
            [
                "",
                "",
                'if __name__ == "__main__":',
                f"    g = {safe_fn_name}()",
                "    print(g.to_json(indent=2))",
            ]
        )

    raw_code = _NL.join(lines) + _NL
    return _format_python_source(raw_code)


def export_schemas(
    out_dir: str | Path,
    include_typescript: bool = True,
) -> dict[str, Path]:
    """Export all canonical JSON Schemas and optional TypeScript definitions to a directory.

    Args:
        out_dir (str | Path): Target directory path. Created if it does not exist.
        include_typescript (bool): Whether to also write `logical_graph.d.ts`.

    Returns:
        dict[str, Path]: Mapping of exported artifact names to written file paths.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}
    schemas = get_json_schema("all")

    file_mapping = {
        "LogicalGraph": "logical_graph.schema.json",
        "LogicalNode": "logical_node.schema.json",
        "SnapshotEnvelope": "snapshot_envelope.schema.json",
        "ZeroTangent": "zero_tangent.schema.json",
        "NoTangent": "no_tangent.schema.json",
        "PipelineTopologyConfig": "pipeline_topology.schema.json",
        "WebRTCSignalingTopology": "webrtc_signaling_topology.schema.json",
    }

    for model_name, filename in file_mapping.items():
        file_path = directory / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(schemas[model_name], f, indent=2, sort_keys=True)
            f.write(_NL)
        exported[model_name] = file_path

    if include_typescript:
        ts_file = directory / "logical_graph.d.ts"
        with open(ts_file, "w", encoding="utf-8") as f:
            f.write(generate_typescript_definitions())
        exported["TypeScript"] = ts_file

    return exported


def _emit_logical_node_python(node: LogicalNode, prefix: str) -> str:
    """Emit single-expression Python constructor for ``LogicalNode`` (including sentinels).

    Args:
        node (LogicalNode): IR node or autodiff sentinel.
        prefix (str): Module qualification prefix.

    Returns:
        str: Executable Python node constructor expression.
    """
    return _NL.join(_format_node_ctor_lines(node, prefix))


def _emit_logical_graph_python(graph: LogicalGraph, prefix: str) -> str:
    """Emit inline Python ``LogicalGraph(...)`` constructor expression.

    Args:
        graph (LogicalGraph): Computation graph.
        prefix (str): Module qualification prefix.

    Returns:
        str: Executable Python ``LogicalGraph(...)`` expression.
    """
    return _format_graph_expr(graph, prefix)


DTYPE_PYTHON_EMIT: PythonExprEmit[DType] = PythonExprEmit(
    emit_python=lambda dt, prefix: f"{prefix}DType.{dt.name}",
)
SYM_EXPR_PYTHON_EMIT: PythonExprEmit[SymExpr] = PythonExprEmit(
    emit_python=_format_sym_expr_python,
)
DIMENSION_PYTHON_EMIT: PythonExprEmit[DimensionType] = PythonExprEmit(
    emit_python=_format_sym_node,
)
TENSOR_SHAPE_PYTHON_EMIT: PythonExprEmit[TensorShape] = PythonExprEmit(
    emit_python=lambda sh, prefix: f"{prefix}TensorShape({_format_shape_tuple(sh.dims, prefix)})",
)
TENSOR_SPEC_PYTHON_EMIT: PythonExprEmit[TensorSpec] = PythonExprEmit(
    emit_python=_format_tensor_spec,
)
PARTITION_SPEC_PYTHON_EMIT: PythonExprEmit[PartitionSpec] = PythonExprEmit(
    emit_python=lambda ps, prefix: f"{prefix}PartitionSpec(axes={ps.axes!r})",
)
LOGICAL_MESH_PYTHON_EMIT: PythonExprEmit[LogicalMesh] = PythonExprEmit(
    emit_python=_emit_logical_mesh_python,
)
ATTRIBUTE_VALUE_PYTHON_EMIT: PythonExprEmit[AttributeValue] = PythonExprEmit(
    emit_python=_format_attr_value,
)
LOGICAL_NODE_PYTHON_EMIT: PythonExprEmit[LogicalNode] = PythonExprEmit(
    emit_python=_emit_logical_node_python,
)
LOGICAL_GRAPH_PYTHON_EMIT: PythonExprEmit[LogicalGraph] = PythonExprEmit(
    emit_python=_emit_logical_graph_python,
)
