"""JSON Schema, TypeScript definitions, and Python code generator for ML-Switcheroo IR.

Provides utilities for emitting canonical Draft 2020-12 conforming JSON schemas
for LogicalGraph, LogicalNode, and SnapshotEnvelope, TypeScript interfaces
for web playground and frontend integrations, and programmatic Python code generation.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import keyword
from pathlib import Path
import re
import typing
from typing_extensions import TypeAlias

from pydantic import TypeAdapter

from ml_switcheroo_ir.distributed import (
    PipelineTopologyConfig,
    WebRTCSignalingTopology,
)
from ml_switcheroo_ir.graph import (
    LogicalGraph,
    LogicalNode,
    NoTangent,
    ZeroTangent,
)
from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope
from ml_switcheroo_ir.shapes import (
    DimensionType,
    SymBinaryOp,
    SymConst,
    SymInt,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
)
from ml_switcheroo_ir.types import DType, JsonValue, TensorShape, TensorSpec

DRAFT_2020_12_SCHEMA_URI: str = "https://json-schema.org/draft/2020-12/schema"

SchemaTargetLiteral: TypeAlias = typing.Literal[
    "LogicalGraph",
    "LogicalNode",
    "SnapshotEnvelope",
    "ZeroTangent",
    "NoTangent",
    "PipelineTopologyConfig",
    "WebRTCSignalingTopology",
    "all",
]


def get_json_schema(
    target: SchemaTargetLiteral | str = "LogicalGraph",
) -> dict[str, JsonValue]:
    """Emit canonical JSON Schema conforming to Draft 2020-12 specification.

    Args:
        target (SchemaTargetLiteral | str): Target model ('LogicalGraph', 'LogicalNode', 'SnapshotEnvelope',
            'ZeroTangent', 'NoTangent', 'PipelineTopologyConfig',
            'WebRTCSignalingTopology', or 'all'). Defaults to 'LogicalGraph'.

    Returns:
        dict[str, JsonValue]: Generated JSON Schema dictionary.

    Raises:
        ValueError: If target is not recognized.
    """
    target_clean = target.strip()
    if target_clean == "LogicalGraph":
        schema: dict[str, JsonValue] = LogicalGraph.model_json_schema()
        schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
        return schema
    if target_clean == "LogicalNode":
        schema = LogicalNode.model_json_schema()
        schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
        return schema
    if target_clean == "SnapshotEnvelope":
        schema = SnapshotEnvelope.model_json_schema()
        schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
        return schema
    if target_clean == "ZeroTangent":
        from ml_switcheroo_ir import sentinels as _sentinels  # noqa: F401

        schema = TypeAdapter(ZeroTangent).json_schema()
        schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
        return schema
    if target_clean == "NoTangent":
        from ml_switcheroo_ir import sentinels as _sentinels  # noqa: F401

        schema = TypeAdapter(NoTangent).json_schema()
        schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
        return schema
    if target_clean == "PipelineTopologyConfig":
        schema = PipelineTopologyConfig.model_json_schema()
        schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
        return schema
    if target_clean == "WebRTCSignalingTopology":
        schema = WebRTCSignalingTopology.model_json_schema()
        schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
        return schema
    if target_clean == "all":
        return {
            "LogicalGraph": get_json_schema("LogicalGraph"),
            "LogicalNode": get_json_schema("LogicalNode"),
            "SnapshotEnvelope": get_json_schema("SnapshotEnvelope"),
            "ZeroTangent": get_json_schema("ZeroTangent"),
            "NoTangent": get_json_schema("NoTangent"),
            "PipelineTopologyConfig": get_json_schema("PipelineTopologyConfig"),
            "WebRTCSignalingTopology": get_json_schema("WebRTCSignalingTopology"),
        }
    raise ValueError(
        f"Unsupported schema target: '{target}'. Supported targets: 'LogicalGraph', "
        "'LogicalNode', 'SnapshotEnvelope', 'ZeroTangent', 'NoTangent', "
        "'PipelineTopologyConfig', 'WebRTCSignalingTopology', 'all'."
    )


def generate_typescript_definitions() -> str:
    """Generate typed TypeScript interfaces matching LogicalGraph for frontend integration.

    Returns:
        str: TypeScript interface definitions.
    """
    dtype_literals = "\n".join(f'  | "{dt.value}"' for dt in DType)
    return (
        (
            """/**
 * ML-Switcheroo IR - Canonical TypeScript Definitions
 * Auto-generated for Abstract ML Machine Compiler Ecosystem (Tier 1)
 */

export type DType =
"""
            + dtype_literals
            + """
  | string;

export type AttributeValue =
  | string
  | number
  | boolean
  | null
  | AttributeValue[]
  | { [key: string]: AttributeValue };"""
        )
        + """

export interface PartitionSpec {
  axes: (string | string[] | null)[];
}

export interface WebRTCPeerConfig {
  peer_id: string;
  device_capability?: string;
  network_transport?: string;
  ice_transport_policy?: string;
}

export interface WebRTCSignalingTopology {
  signaling_url?: string;
  ice_servers?: string[];
  high_watermark_bytes?: number;
  low_watermark_bytes?: number;
  chunk_size_bytes?: number;
  connectivity_matrix?: number[][];
  peers?: WebRTCPeerConfig[];
}

export interface LogicalMesh {
  shape: Record<string, number>;
  webrtc_topology?: WebRTCSignalingTopology | null;
}

export interface LogicalEdge {
  source: string;
  target: string;
  source_idx?: number;
  target_idx?: number;
  value_name?: string | null;
}

export interface SymVar {
  type: "var";
  name: string;
}

export interface SymConst {
  type: "const";
  value: number;
}

export interface SymBinaryOp {
  type: "binary_op";
  op: string;
  left: SymNode;
  right: SymNode;
}

export interface SymUnaryOp {
  type: "unary_op";
  op: string;
  operand: SymNode;
}

export interface SymPiecewise {
  type: "piecewise";
  cases?: [SymNode | string | boolean, SymNode][];
  default?: SymNode;
}

export type SymNode =
  | SymVar
  | SymConst
  | SymBinaryOp
  | SymUnaryOp
  | SymPiecewise;

export interface SymInt {
  node: SymNode;
}

export type DimensionType = number | string | SymNode | SymInt;

export interface TensorSpec {
  shape: DimensionType[];
  dtype: DType;
  sparsity?: string | null;
}

export interface MicrobatchSplittingConfig {
  strategy?: string;
  num_microbatches?: number;
}

export interface MeshMappingConfig {
  devices_per_stage?: number;
}

export interface StageCommunicationConfig {
  protocol?: string;
}

export interface DependencyConfig {
  source_stage: string;
  target_stage: string;
  offset_mb?: number;
}

export interface SchedulePhaseConfig {
  type?: string;
  operations?: string[];
  count_expression?: string;
}

export interface PipelineScheduleConfig {
  phases?: SchedulePhaseConfig[];
}

export interface PipelineTopologyConfig {
  microbatch_splitting: MicrobatchSplittingConfig;
  mesh_mapping: MeshMappingConfig;
  stage_communication: StageCommunicationConfig;
  dependencies?: DependencyConfig[];
  schedule?: PipelineScheduleConfig | null;
}

export interface LogicalNode {
  id: string;
  op_type: string;
  domain?: string;
  version?: number;
  attributes?: Record<string, AttributeValue>;
  inputs?: string[];
  outputs?: string[];
  shape_metadata?: DimensionType[] | null;
  source_ast_ref?: string | null;
  dtype?: DType | null;
  output_specs?: TensorSpec[];
  subgraphs?: Record<string, LogicalGraph>;
  device?: string | null;
  stream?: string | null;
  sharding?: PartitionSpec | null;
}

export interface ZeroTangent extends LogicalNode {
  op_type: "ZeroTangent";
  domain: "ml.switcheroo.ad";
}

export interface NoTangent extends LogicalNode {
  op_type: "NoTangent";
  domain: "ml.switcheroo.ad";
}

export interface LogicalGraph {
  name?: string;
  nodes: Record<string, LogicalNode>;
  inputs?: string[];
  input_specs?: Record<string, TensorSpec>;
  outputs?: string[];
  initializers?: Record<string, unknown>;
  mesh?: LogicalMesh | null;
  pipeline_topology?: PipelineTopologyConfig | null;
  edges?: LogicalEdge[];
}

export interface SnapshotEnvelope {
  schema_version: string;
  target: string;
  version: string;
  upstream_version?: string | null;
  source_type: string;
  upstream_commit?: string | null;
  supported_microarchitectures?: string[] | null;
  generated_at: string;
  environment?: Record<string, string>;
  categories?: Record<string, unknown>;
  operations?: unknown[] | null;
  instructions?: unknown[] | null;
}
"""
    )


def _format_sym_node(node: DimensionType) -> str:
    """Recursively format a symbolic shape node into executable Python code.

    Args:
        node (DimensionType): Symbolic node, integer, or string dimension.

    Returns:
        str: Python code expression creating the symbol.
    """
    if isinstance(node, SymInt):
        return f"sw_ir.SymInt({_format_sym_node(node.node)})"
    if isinstance(node, SymVar):
        return f"sw_ir.SymVar({node.name!r})"
    if isinstance(node, SymConst):
        return f"sw_ir.SymConst({node.value!r})"
    if isinstance(node, SymBinaryOp):
        left_str = _format_sym_node(node.left)
        right_str = _format_sym_node(node.right)
        return f"sw_ir.SymBinaryOp({node.op!r}, {left_str}, {right_str})"
    if isinstance(node, SymUnaryOp):
        operand_str = _format_sym_node(node.operand)
        return f"sw_ir.SymUnaryOp({node.op!r}, {operand_str})"
    if isinstance(node, SymPiecewise):
        cases = [f"({c!r}, {_format_sym_node(n)})" for c, n in node.cases]
        def_str = _format_sym_node(node.default)
        return f"sw_ir.SymPiecewise(cases=[{', '.join(cases)}], default={def_str})"
    return repr(node)


def _format_shape_tuple(
    shape: Sequence[DimensionType] | TensorShape | None,
) -> str:
    """Format shape metadata or dimension tuple into Python representation.

    Args:
        shape (Sequence[DimensionType] | TensorShape | None): Shape tuple, list, or None.

    Returns:
        str: Python tuple representation.
    """
    if shape is None:
        return "None"
    formatted_dims = [_format_sym_node(d) for d in shape]
    if len(formatted_dims) == 1:
        return f"({formatted_dims[0]},)"
    return f"({', '.join(formatted_dims)})"


def _format_tensor_spec(spec: TensorSpec) -> str:
    """Format a TensorSpec into executable Python instantiation code.

    Args:
        spec (TensorSpec): Tensor specification instance.

    Returns:
        str: Python instantiation expression.
    """
    shape_str = _format_shape_tuple(spec.shape)
    dtype_str = f"sw_ir.DType.{spec.dtype.name}"
    sparsity_str = repr(spec.sparsity)
    return (
        f"sw_ir.TensorSpec(shape={shape_str}, dtype={dtype_str}, "
        f"sparsity={sparsity_str})"
    )


def _format_graph_expr(graph: LogicalGraph, indent: int = 0) -> str:
    """Format a nested LogicalGraph into a Python expression string.

    Args:
        graph (LogicalGraph): Nested subgraph to serialize.
        indent (int): Leading indentation level in spaces.

    Returns:
        str: Executable Python expression reconstructing the subgraph.
    """
    prefix = " " * max(0, indent)
    return f"{prefix}sw_ir.LogicalGraph.from_dict({graph.to_dict()!r})"


def _sanitize_python_identifier(name: str, fallback: str = "build_graph") -> str:
    """Sanitize an arbitrary string into a valid, non-keyword Python identifier.

    Args:
        name (str): Candidate function or variable identifier.
        fallback (str): Fallback identifier if sanitized name is empty.

    Returns:
        str: Valid Python identifier string.
    """
    cleaned = re.sub(r"\W+", "_", name.strip()).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def export_to_python(
    graph: LogicalGraph,
    function_name: str = "build_graph",
    standalone: bool = False,
) -> str:
    """Emit valid, formatted Python source code constructing the LogicalGraph.

    Generates code that constructs the `LogicalGraph`, `LogicalNode`s,
    `LogicalMesh`, `PartitionSpec`s, `TensorSpec`s, and distributed topologies
    using `ml_switcheroo_ir`.

    Args:
        graph (LogicalGraph): LogicalGraph instance to serialize.
        function_name (str): Name of the builder function. Defaults to 'build_graph'.
        standalone (bool): If True, appends a ``if __name__ == '__main__':`` entrypoint. Defaults to False.

    Returns:
        str: Formatted Python source code.
    """
    safe_fn_name = _sanitize_python_identifier(function_name)
    safe_doc_name = (
        graph.name.replace("\\", "\\\\").replace('"""', "'''").splitlines()[0]
        if graph.name
        else "Model"
    )
    lines: list[str] = [
        '"""Auto-generated LogicalGraph definition.',
        "",
        "Emitted by ml_switcheroo_ir.export.export_to_python.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import ml_switcheroo_ir as sw_ir",
        "",
        "",
        f"def {safe_fn_name}() -> sw_ir.LogicalGraph:",
        f'    """Construct and return the {safe_doc_name} LogicalGraph.',
        "",
        "    Returns:",
        "        sw_ir.LogicalGraph: Reconstructed computational graph.",
        '    """',
    ]

    # Pipeline topology
    if graph.pipeline_topology is not None:
        pipe_dict = graph.pipeline_topology.to_dict()
        lines.append(
            f"    pipeline_topology = sw_ir.PipelineTopologyConfig.from_dict({pipe_dict!r})"
        )
    else:
        lines.append("    pipeline_topology = None")

    # Mesh
    if graph.mesh is not None:
        mesh_shape_repr = repr(dict(sorted(graph.mesh.shape.items())))
        if graph.mesh.webrtc_topology is not None:
            webrtc_dict = graph.mesh.webrtc_topology.to_dict()
            lines.append(
                f"    webrtc_topology = sw_ir.WebRTCSignalingTopology.from_dict({webrtc_dict!r})"
            )
            lines.append(
                f"    mesh = sw_ir.LogicalMesh(shape={mesh_shape_repr}, webrtc_topology=webrtc_topology)"
            )
        else:
            lines.append(f"    mesh = sw_ir.LogicalMesh(shape={mesh_shape_repr})")
    else:
        lines.append("    mesh = None")

    # Input specs
    if graph.input_specs:
        lines.append("    input_specs: dict[str, sw_ir.TensorSpec] = {")
        for in_name in sorted(graph.input_specs.keys()):
            spec_str = _format_tensor_spec(graph.input_specs[in_name])
            lines.append(f"        {in_name!r}: {spec_str},")
        lines.append("    }")
    else:
        lines.append("    input_specs: dict[str, sw_ir.TensorSpec] = {}")

    # Nodes
    lines.append("    nodes: dict[str, sw_ir.LogicalNode] = {")
    for nid, node in graph.nodes.items():
        if isinstance(node, ZeroTangent):
            shape_str = _format_shape_tuple(node.shape_metadata)
            dtype_str = (
                f"sw_ir.DType.{node.dtype.name}" if node.dtype is not None else "None"
            )
            lines.append(
                f"        {nid!r}: sw_ir.ZeroTangent(id={node.id!r}, "
                f"shape_metadata={shape_str}, dtype={dtype_str}),"
            )
            continue
        if isinstance(node, NoTangent):
            lines.append(f"        {nid!r}: sw_ir.NoTangent(id={node.id!r}),")
            continue

        sharding_str = "None"
        if node.sharding is not None:
            sharding_str = f"sw_ir.PartitionSpec(axes={node.sharding.axes!r})"

        dtype_val_str = "None"
        if node.dtype is not None:
            dtype_val_str = f"sw_ir.DType.{node.dtype.name}"

        shape_meta_str = _format_shape_tuple(node.shape_metadata)

        out_specs_list = (
            f"[{', '.join(_format_tensor_spec(s) for s in node.output_specs)}]"
        )

        sorted_attrs = dict(sorted(node.attributes.items(), key=lambda kv: str(kv[0])))
        subgraphs_entries: list[str] = []
        for sub_k, sub_v in sorted(node.subgraphs.items(), key=lambda kv: str(kv[0])):
            if isinstance(sub_v, LogicalGraph):
                subgraphs_entries.append(f"{sub_k!r}: {_format_graph_expr(sub_v)}")
            else:
                subgraphs_entries.append(f"{sub_k!r}: {sub_v!r}")
        subgraphs_str = "{" + ", ".join(subgraphs_entries) + "}"

        lines.append(f"        {nid!r}: sw_ir.LogicalNode(")
        lines.append(f"            id={node.id!r},")
        lines.append(f"            op_type={node.op_type!r},")
        lines.append(f"            domain={node.domain!r},")
        lines.append(f"            version={node.version},")
        lines.append(f"            attributes={sorted_attrs!r},")
        lines.append(f"            inputs={node.inputs!r},")
        lines.append(f"            outputs={node.outputs!r},")
        lines.append(f"            shape_metadata={shape_meta_str},")
        lines.append(f"            source_ast_ref={node.source_ast_ref!r},")
        lines.append(f"            sharding={sharding_str},")
        lines.append(f"            dtype={dtype_val_str},")
        lines.append(f"            output_specs={out_specs_list},")
        lines.append(f"            subgraphs={subgraphs_str},")
        lines.append(f"            device={node.device!r},")
        lines.append(f"            stream={node.stream!r},")
        lines.append("        ),")
    lines.append("    }")

    # Return LogicalGraph
    lines.append("    return sw_ir.LogicalGraph(")
    lines.append(f"        name={graph.name!r},")
    lines.append("        nodes=nodes,")
    lines.append(f"        inputs={graph.inputs!r},")
    lines.append("        input_specs=input_specs,")
    lines.append(f"        outputs={graph.outputs!r},")
    lines.append(f"        initializers={graph.initializers!r},")
    lines.append("        mesh=mesh,")
    lines.append("        pipeline_topology=pipeline_topology,")
    lines.append("    )")

    if standalone:
        lines.extend(
            [
                "",
                "",
                'if __name__ == "__main__":',
                f"    graph = {safe_fn_name}()",
                "    print(f'Constructed graph {graph.name} with {len(graph.nodes)} nodes.')",
            ]
        )

    lines.append("")
    code = "\n".join(lines)

    try:
        import black

        return black.format_str(code, mode=black.Mode())
    except ImportError:
        return code


def export_schemas(
    out_dir: Path | str,
    include_typescript: bool = False,
) -> dict[str, Path]:
    """Export canonical JSON schemas and optional TypeScript definitions to a directory.

    Args:
        out_dir (Path | str): Output directory path.
        include_typescript (bool): Whether to also write TypeScript definition file. Defaults to False.

    Returns:
        dict[str, Path]: Mapping from schema name to exported file path.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}

    for name in (
        "LogicalGraph",
        "LogicalNode",
        "SnapshotEnvelope",
        "ZeroTangent",
        "NoTangent",
        "PipelineTopologyConfig",
        "WebRTCSignalingTopology",
    ):
        schema = get_json_schema(name)
        snake_name = "".join(
            ["_" + c.lower() if c.isupper() else c for c in name]
        ).lstrip("_")
        out_file = directory / f"{snake_name}.schema.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, sort_keys=True)
        exported[name] = out_file

    if include_typescript:
        ts_file = directory / "logical_graph.d.ts"
        with open(ts_file, "w", encoding="utf-8") as f:
            f.write(generate_typescript_definitions())
        exported["TypeScript"] = ts_file

    return exported
