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

from ml_switcheroo_ir.distributed import (
    PipelineTopologyConfig,
    WebRTCSignalingTopology,
)
from ml_switcheroo_ir.graph import LogicalGraph, LogicalNode
from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope
from ml_switcheroo_ir.sentinels import NoTangent, ZeroTangent
from ml_switcheroo_ir.shapes import (
    DimensionType,
    SymBinaryOp,
    SymConst,
    SymInt,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
)
from ml_switcheroo_ir.types import (
    AttributeValue,
    DType,
    InitializerValue,
    JsonValue,
    TensorShape,
    TensorSpec,
)

__all__ = [
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
  source_ast_ref?: string | null;
  sharding?: PartitionSpec | null;
  dtype?: DType | null;
  output_specs?: TensorSpec[];
  subgraphs?: Record<string, LogicalGraph | AttributeValue>;
  device?: string | null;
  stream?: string | null;
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
  mesh?: LogicalMesh | null;
  pipeline_topology?: PipelineTopologyConfig | null;
  attributes?: Record<string, AttributeValue>;
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


def _format_sym_node(val: DimensionType) -> str:
    """Format a DimensionType or SymNode into clean Python code representation.

    Args:
        val (DimensionType): Dimension or symbolic node.

    Returns:
        str: Executable Python expression string.
    """
    if isinstance(val, (SymVar, SymConst)):
        return f"sw_ir.SymVar({val.name!r})" if isinstance(val, SymVar) else f"sw_ir.SymConst({val.value!r})"
    if isinstance(val, SymBinaryOp):
        return f"sw_ir.SymBinaryOp({val.op!r}, {_format_sym_node(val.left)}, {_format_sym_node(val.right)})"
    if isinstance(val, SymUnaryOp):
        return f"sw_ir.SymUnaryOp({val.op!r}, {_format_sym_node(val.operand)})"
    if isinstance(val, SymInt):
        return f"sw_ir.SymInt({_format_sym_node(val.node)})"
    if isinstance(val, SymPiecewise):
        cases_items: list[str] = []
        for cond, expr in val.cases:
            cond_str = _format_sym_node(cond) if isinstance(cond, SymNode) else repr(cond)
            cases_items.append(f"({cond_str}, {_format_sym_node(expr)})")
        cases_str = "[" + ", ".join(cases_items) + "]"
        return f"sw_ir.SymPiecewise(cases={cases_str}, default={_format_sym_node(val.default)})"
    return repr(val)


def _format_shape_tuple(
    shape: Sequence[DimensionType] | TensorShape | None,
) -> str:
    """Format shape metadata or dimension tuple into Python representation.

    Args:
        shape (Sequence[DimensionType] | TensorShape | None): Shape tuple, list, or None.

    Returns:
        str: Formatted Python tuple expression or 'None'.
    """
    if shape is None:
        return "None"
    dims = shape.dims if isinstance(shape, TensorShape) else tuple(shape)
    if len(dims) == 0:
        return "()"
    if len(dims) == 1:
        return f"({_format_sym_node(dims[0])},)"
    items = ", ".join(_format_sym_node(d) for d in dims)
    return f"({items})"


def _format_tensor_spec(spec: TensorSpec) -> str:
    """Format a TensorSpec instance into Python constructor code.

    Args:
        spec (TensorSpec): TensorSpec instance.

    Returns:
        str: Python code constructing TensorSpec.
    """
    dtype_str = f"sw_ir.DType.{spec.dtype.name}"
    shape_str = _format_shape_tuple(spec.shape)
    if spec.sparsity is not None:
        return f"sw_ir.TensorSpec(shape={shape_str}, dtype={dtype_str}, sparsity={spec.sparsity!r})"
    return f"sw_ir.TensorSpec(shape={shape_str}, dtype={dtype_str})"


def _format_rich_attr_leaf(val: DType | SymNode | SymInt | TensorShape | TensorSpec) -> str:
    """Format a rich non-primitive leaf ``AttributeValue`` into Python code.

    Args:
        val (DType | SymNode | SymInt | TensorShape | TensorSpec): Rich attribute leaf.

    Returns:
        str: Formatted Python constructor expression.
    """
    if isinstance(val, DType):
        return f"sw_ir.DType.{val.name}"
    if isinstance(val, (SymNode, SymInt)):
        return _format_sym_node(val)
    if isinstance(val, TensorShape):
        return f"sw_ir.TensorShape({_format_shape_tuple(val.dims)})"
    return _format_tensor_spec(val)


def _format_attr_value(val: AttributeValue) -> str:
    """Format an ``AttributeValue`` recursively into an executable Python expression.

    Args:
        val (AttributeValue): Attribute value (primitives, bytes, DType, SymNode, SymInt,
            TensorShape, TensorSpec, sequences, or mappings).

    Returns:
        str: Executable Python expression reconstructing ``val`` losslessly.
    """
    if isinstance(val, (DType, SymNode, SymInt, TensorShape, TensorSpec)):
        return _format_rich_attr_leaf(val)
    if val is None or isinstance(val, (bool, int, float, str, bytes)):
        return repr(val)
    if isinstance(val, Mapping):
        entries = ", ".join(
            f"{str(k)!r}: {_format_attr_value(v)}" for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + entries + "}"
    if isinstance(val, tuple):
        if len(val) <= 1:
            return "()" if len(val) == 0 else f"({_format_attr_value(val[0])},)"
        return "(" + ", ".join(_format_attr_value(x) for x in val) + ")"
    return "[" + ", ".join(_format_attr_value(x) for x in val) + "]"


def _format_initializer_value(val: InitializerValue) -> str:
    """Format an ``InitializerValue`` recursively into an executable Python expression.

    Args:
        val (InitializerValue): Initializer constant payload.

    Returns:
        str: Executable Python expression reconstructing ``val``.
    """
    if isinstance(val, (bool, int, float, str, bytes)):
        return repr(val)
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


def _format_mesh_expr(graph: LogicalGraph) -> str:
    """Format ``graph.mesh`` into an inline Python constructor expression.

    Args:
        graph (LogicalGraph): Graph containing optional ``mesh``.

    Returns:
        str: Python expression constructing ``LogicalMesh`` or ``'None'``.
    """
    if graph.mesh is None:
        return "None"
    shape_str = json.dumps(graph.mesh.shape, sort_keys=True)
    if graph.mesh.webrtc_topology is not None:
        webrtc_expr = f"sw_ir.WebRTCSignalingTopology.from_dict({graph.mesh.webrtc_topology.to_dict()!r})"
        return f"sw_ir.LogicalMesh(shape={shape_str}, webrtc_topology={webrtc_expr})"
    return f"sw_ir.LogicalMesh(shape={shape_str})"


def _format_graph_expr(graph: LogicalGraph) -> str:
    """Format a nested ``LogicalGraph`` into an inline ``sw_ir.LogicalGraph(...)`` constructor expression.

    Emits pure Python constructors for all nested nodes, rich attributes, DTypes,
    symbolic dimensions, topologies, and subgraphs without ``to_dict()!r`` enum repr leaks.

    Args:
        graph (LogicalGraph): Nested subgraph to render.

    Returns:
        str: Valid Python expression constructing the nested ``LogicalGraph``.
    """
    nodes_entries: list[str] = []
    for nid, node in graph.nodes.items():
        node_lines = _format_node_entry_lines(nid, node)
        nodes_entries.append(_NL.join(node_lines).strip().rstrip(","))
    nodes_expr = "{" + ", ".join(nodes_entries) + "}"
    specs_expr = (
        "{" + ", ".join(f"{k!r}: {_format_tensor_spec(spec)}" for k, spec in sorted(graph.input_specs.items())) + "}"
    )
    inits_expr = (
        "{"
        + ", ".join(f"{k!r}: {_format_initializer_value(val)}" for k, val in sorted(graph.initializers.items()))
        + "}"
    )
    mesh_expr = _format_mesh_expr(graph)
    pipe_expr = (
        f"sw_ir.PipelineTopologyConfig.from_dict({graph.pipeline_topology.to_dict()!r})"
        if graph.pipeline_topology is not None
        else "None"
    )
    attrs_expr = _format_attr_value(graph.attributes)
    loop_expr = _format_attr_value(graph.loop_options) if graph.loop_options is not None else "None"
    edges_expr = _format_edges_expr(graph)
    edges_kw = f", edges={edges_expr}" if edges_expr is not None else ""
    return (
        f"sw_ir.LogicalGraph("
        f"name={graph.name!r}, "
        f"nodes={nodes_expr}, "
        f"inputs={list(graph.inputs)!r}, "
        f"input_specs={specs_expr}, "
        f"outputs={list(graph.outputs)!r}, "
        f"initializers={inits_expr}, "
        f"mesh={mesh_expr}, "
        f"pipeline_topology={pipe_expr}, "
        f"attributes={attrs_expr}, "
        f"loop_options={loop_expr}"
        f"{edges_kw})"
    )


def _format_edges_expr(graph: LogicalGraph) -> str | None:
    """Format explicit ``edges`` constructor list when edge ``value_name`` metadata is present.

    Args:
        graph (LogicalGraph): Graph to inspect.

    Returns:
        str | None: Python list literal of ``sw_ir.LogicalEdge(...)`` calls, or ``None`` if unneeded.
    """
    if not graph._edge_value_names:
        return None
    edge_items = [
        f"sw_ir.LogicalEdge(source={e.source!r}, target={e.target!r}, "
        f"source_idx={e.source_idx!r}, target_idx={e.target_idx!r}, value_name={e.value_name!r})"
        for e in graph.iter_edges()
    ]
    return "[" + ", ".join(edge_items) + "]"


def _format_subgraphs_expr(subgraphs: Mapping[str, LogicalGraph | AttributeValue]) -> str:
    """Format a node's ``subgraphs`` dictionary into a Python dict literal expression.

    Args:
        subgraphs (Mapping[str, LogicalGraph | AttributeValue]): Node subgraphs mapping.

    Returns:
        str: Formatted Python dictionary expression.
    """
    if not subgraphs:
        return "{}"
    entries: list[str] = []
    for k, v in sorted(subgraphs.items()):
        val_expr = _format_graph_expr(v) if isinstance(v, LogicalGraph) else _format_attr_value(v)
        entries.append(f"{k!r}: {val_expr}")
    return "{" + ", ".join(entries) + "}"


def _format_node_entry_lines(nid: str, node: LogicalNode) -> list[str]:
    """Generate indented source lines for a single ``nodes`` entry in ``export_to_python``.

    Preserves all fields (``version``, ``attributes``, ``inputs``, ``outputs``,
    ``shape_metadata``, ``source_ast_ref``, ``sharding``, ``dtype``, ``output_specs``,
    ``subgraphs``, ``device``, ``stream``) across ``LogicalNode``, ``ZeroTangent``,
    and ``NoTangent``.

    Args:
        nid (str): Node key in ``graph.nodes``.
        node (LogicalNode): Node instance to format.

    Returns:
        list[str]: Indented Python source lines for this dictionary item.
    """
    is_zero = isinstance(node, ZeroTangent)
    is_no = isinstance(node, NoTangent)
    ctor = "ZeroTangent" if is_zero else ("NoTangent" if is_no else "LogicalNode")

    sharding_str = f"sw_ir.PartitionSpec(axes={node.sharding.axes!r})" if node.sharding is not None else "None"
    dtype_str = f"sw_ir.DType.{node.dtype.name}" if node.dtype is not None else "None"
    shape_str = _format_shape_tuple(node.shape_metadata)
    out_specs_str = "[" + ", ".join(_format_tensor_spec(s) for s in node.output_specs) + "]"
    attrs_str = _format_attr_value(node.attributes)
    subgraphs_str = _format_subgraphs_expr(node.subgraphs)

    lines: list[str] = [
        f"        {nid!r}: sw_ir.{ctor}(",
        f"            id={node.id!r},",
    ]
    if not (is_zero or is_no):
        lines.append(f"            op_type={node.op_type!r},")
        lines.append(f"            domain={node.domain!r},")
    lines.extend(
        [
            f"            version={node.version!r},",
            f"            attributes={attrs_str},",
            f"            inputs={list(node.inputs)!r},",
            f"            outputs={list(node.outputs)!r},",
            f"            shape_metadata={shape_str},",
            f"            source_ast_ref={node.source_ast_ref!r},",
            f"            sharding={sharding_str},",
            f"            dtype={dtype_str},",
            f"            output_specs={out_specs_str},",
            f"            subgraphs={subgraphs_str},",
            f"            device={node.device!r},",
            f"            stream={node.stream!r},",
            "        ),",
        ]
    )
    return lines


def _append_topology_lines(lines: list[str], graph: LogicalGraph) -> tuple[str, str]:
    """Append mesh and pipeline topology builder statements to ``lines``.

    Args:
        lines (list[str]): Output statement lines list.
        graph (LogicalGraph): Source graph.

    Returns:
        tuple[str, str]: ``(mesh_var, pipeline_var)`` variable expressions.
    """
    mesh_var = "None"
    if graph.mesh is not None:
        shape_str = json.dumps(graph.mesh.shape, sort_keys=True)
        if graph.mesh.webrtc_topology is not None:
            webrtc_dict = graph.mesh.webrtc_topology.to_dict()
            lines.append(f"    webrtc_topology = sw_ir.WebRTCSignalingTopology.from_dict({webrtc_dict!r})")
            lines.append(f"    mesh = sw_ir.LogicalMesh(shape={shape_str}, webrtc_topology=webrtc_topology)")
        else:
            lines.append(f"    mesh = sw_ir.LogicalMesh(shape={shape_str})")
        mesh_var = "mesh"

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
        lines.append(f"        {k!r}: {_format_tensor_spec(spec)},")
    lines.append("    }")

    lines.append("    initializers: dict[str, sw_ir.InitializerValue] = {")
    for k, val in sorted(graph.initializers.items()):
        lines.append(f"        {k!r}: {_format_initializer_value(val)},")
    lines.append("    }")

    attrs_expr = _format_attr_value(graph.attributes)
    loop_expr = _format_attr_value(graph.loop_options) if graph.loop_options is not None else "None"
    edges_expr = _format_edges_expr(graph)

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
