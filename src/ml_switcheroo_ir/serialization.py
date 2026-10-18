"""Streaming JSON serialization, chunked deserialization, and compression utilities for LogicalGraph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import string
from typing import TYPE_CHECKING, IO, Any, TextIO, cast

from ml_switcheroo_ir.distributed import (
    LogicalMesh,
    PartitionSpec,
    PipelineTopologyConfig,
    WebRTCSignalingTopology,
)
from ml_switcheroo_ir.types import (
    AttributeValue,
    DType,
    TensorSpec,
)

if TYPE_CHECKING:
    from ml_switcheroo_ir.graph import LogicalEdge, LogicalGraph, LogicalNode, NodeDict

__all__ = [
    "_apply_edges_list",
    "_build_graph_from_parsed_sections",
    "_decode_next_json_value",
    "_indent_json_fragment",
    "_load_zstandard",
    "_parse_input_specs_dict",
    "_parse_mesh_dict",
    "_parse_node_dict",
    "_parse_subgraphs_from_node_dict",
    "_read_next_chunk",
    "_serialize_node_dict",
    "_skip_ws_and_tokens",
    "_stream_parse_graph",
    "_stream_parse_nodes_section",
    "_stream_write_graph",
    "zstandard",
]


def _load_zstandard() -> Any:
    """Import and return the optional zstandard module if installed.

    Returns:
        Any: The zstandard module, or None if unavailable.
    """
    try:
        import zstandard as zstd
    except ImportError:
        return None
    return zstd


zstandard: Any = _load_zstandard()


def _parse_node_dict(
    ndata: Mapping[str, Any],
    graph_cls: type[LogicalGraph],
) -> LogicalNode:
    """Deserialize a single LogicalNode from a raw dictionary representation.

    Args:
        ndata (Mapping[str, Any]): Raw dictionary describing a node.
        graph_cls (type[LogicalGraph]): LogicalGraph class used for recursive subgraphs.

    Returns:
        LogicalNode: Deserialized LogicalNode instance.
    """
    from ml_switcheroo_ir.graph import LogicalNode
    from ml_switcheroo_ir.sentinels import NoTangent, ZeroTangent

    ndata_copy = dict(ndata)
    if "kind" in ndata_copy and "op_type" not in ndata_copy:
        ndata_copy["op_type"] = ndata_copy.pop("kind")
    if "metadata" in ndata_copy and "attributes" not in ndata_copy:
        ndata_copy["attributes"] = ndata_copy.pop("metadata")
    sharding = ndata_copy.get("sharding")
    if sharding:
        ndata_copy["sharding"] = PartitionSpec(
            axes=tuple(tuple(a) if isinstance(a, list) else a for a in sharding["axes"])
        )
    if "outputs" in ndata_copy and ndata_copy["outputs"] is not None:
        ndata_copy["outputs"] = list(ndata_copy["outputs"])

    dtype_val = ndata_copy.get("dtype")
    if dtype_val is not None and not isinstance(dtype_val, DType):
        ndata_copy["dtype"] = DType.from_str(str(dtype_val))

    output_specs_raw = ndata_copy.get("output_specs")
    if output_specs_raw:
        ndata_copy["output_specs"] = [
            TensorSpec(**s) if isinstance(s, dict) else s for s in output_specs_raw
        ]

    ndata_copy["subgraphs"] = _parse_subgraphs_from_node_dict(ndata_copy, graph_cls)
    op_type_val = ndata_copy.get("op_type")
    domain_val = ndata_copy.get("domain", "ai.onnx")
    if domain_val == "ml.switcheroo.ad" and op_type_val in ("ZeroTangent", "NoTangent"):
        sentinel_cls = ZeroTangent if op_type_val == "ZeroTangent" else NoTangent
        return sentinel_cls(**ndata_copy)
    return LogicalNode(**ndata_copy)


def _parse_subgraphs_from_node_dict(
    ndata_copy: dict[str, Any],
    graph_cls: type[LogicalGraph],
) -> dict[str, LogicalGraph | AttributeValue]:
    """Extract and deserialize nested subgraphs from node dictionary and legacy attributes.

    Args:
        ndata_copy (dict[str, Any]): Mutable node dictionary.
        graph_cls (type[LogicalGraph]): LogicalGraph class for recursive deserialization.

    Returns:
        dict[str, LogicalGraph | AttributeValue]: Mapping of subgraph names to LogicalGraph instances or AttributeValues.
    """
    from ml_switcheroo_ir.graph import LogicalGraph as _LogicalGraph

    resolved_subgraphs: dict[str, LogicalGraph | AttributeValue] = {}
    subgraphs_data = ndata_copy.get("subgraphs")
    if subgraphs_data and isinstance(subgraphs_data, dict):
        for sub_k, sub_v in subgraphs_data.items():
            if isinstance(sub_v, dict) and ("nodes" in sub_v or "name" in sub_v):
                resolved_subgraphs[sub_k] = graph_cls.from_dict(sub_v)
            elif isinstance(sub_v, _LogicalGraph):
                resolved_subgraphs[sub_k] = sub_v
            else:
                resolved_subgraphs[sub_k] = cast("AttributeValue", sub_v)

    attrs = ndata_copy.get("attributes")
    if isinstance(attrs, dict):
        for legacy_key in (
            "subgraph",
            "body_subgraph",
            "bwd_graph",
            "custom_backward_subgraph",
            "jvp_graph",
        ):
            if legacy_key in attrs and legacy_key not in resolved_subgraphs:
                val = attrs[legacy_key]
                canonical_key = (
                    "body"
                    if "body" in legacy_key
                    else ("bwd" if "bwd" in legacy_key else legacy_key)
                )
                if isinstance(val, dict):
                    resolved_subgraphs[canonical_key] = graph_cls.from_dict(val)
                elif isinstance(val, _LogicalGraph):
                    resolved_subgraphs[canonical_key] = val
    return resolved_subgraphs


def _parse_mesh_dict(mesh_data: Mapping[str, Any] | None) -> LogicalMesh | None:
    """Deserialize an optional LogicalMesh dictionary.

    Args:
        mesh_data (Optional[Mapping[str, Any]]): Raw mesh dictionary or None.

    Returns:
        Optional[LogicalMesh]: Deserialized LogicalMesh or None.
    """
    if not mesh_data:
        return None
    webrtc_topo = (
        WebRTCSignalingTopology.from_dict(mesh_data["webrtc_topology"])
        if mesh_data.get("webrtc_topology")
        else None
    )
    return LogicalMesh(shape=mesh_data["shape"], webrtc_topology=webrtc_topo)


def _parse_input_specs_dict(
    input_specs_raw: Mapping[str, Any] | None,
) -> dict[str, TensorSpec]:
    """Deserialize an optional input_specs dictionary into TensorSpec instances.

    Args:
        input_specs_raw (Optional[Mapping[str, Any]]): Raw input specs dictionary.

    Returns:
        dict[str, TensorSpec]: Parsed input specifications.
    """
    input_specs: dict[str, TensorSpec] = {}
    if input_specs_raw and isinstance(input_specs_raw, dict):
        for k, v in input_specs_raw.items():
            if isinstance(v, dict):
                input_specs[k] = TensorSpec(**v)
            elif isinstance(v, TensorSpec):
                input_specs[k] = v
    return input_specs


def _read_next_chunk(stream: IO[Any], chunk_size: int) -> str:
    """Read up to chunk_size from a text or binary stream as a UTF-8 string.

    Args:
        stream (IO[Any]): Readable text or binary stream.
        chunk_size (int): Maximum number of characters/bytes to read.

    Returns:
        str: Decoded string chunk, or empty string on EOF.
    """
    raw = stream.read(chunk_size)
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _skip_ws_and_tokens(
    buf: str,
    pos: int,
    stream: IO[Any],
    chunk_size: int,
    tokens: str,
) -> tuple[str, int]:
    """Advance buffer cursor past whitespace and delimiter characters, refilling from stream as needed.

    Args:
        buf (str): Current active buffer slice.
        pos (int): Current index within buf.
        stream (IO[Any]): Source input stream.
        chunk_size (int): Chunk size for incremental reads and compaction.
        tokens (str): Characters to skip over.

    Returns:
        tuple[str, int]: Updated (buf, pos) positioned at the next non-skipped character or EOF.
    """
    if pos >= chunk_size:
        buf = buf[pos:]
        pos = 0
    while True:
        while pos < len(buf) and buf[pos] in tokens:
            pos += 1
        if pos < len(buf):
            return buf, pos
        chunk = _read_next_chunk(stream, chunk_size)
        if not chunk:
            return buf, pos
        buf = buf[pos:] + chunk
        pos = 0


_WS = string.whitespace
_WS_COMMA = string.whitespace + ","
_WS_COLON = string.whitespace + ":"


def _decode_next_json_value(
    decoder: json.JSONDecoder,
    buf: str,
    pos: int,
    stream: IO[Any],
    chunk_size: int,
) -> tuple[Any, str, int]:
    """Decode a single JSON value starting at pos, incrementally reading chunks on boundary straddles.

    Args:
        decoder (json.JSONDecoder): Reusable JSONDecoder instance.
        buf (str): Current buffer slice.
        pos (int): Current position within buf.
        stream (IO[Any]): Source input stream.
        chunk_size (int): Chunk size for incremental reads.

    Returns:
        tuple[Any, str, int]: Decoded Python value and updated (buf, next_pos).

    Raises:
        json.JSONDecodeError: If the stream ends before a complete JSON value can be decoded.
    """
    buf, pos = _skip_ws_and_tokens(buf, pos, stream, chunk_size, _WS)
    while True:
        try:
            val, next_pos = decoder.raw_decode(buf, pos)
            if next_pos == len(buf) and buf[pos] not in ('"', "{", "["):
                chunk = _read_next_chunk(stream, chunk_size)
                if chunk:
                    buf = buf[pos:] + chunk
                    pos = 0
                    continue
            return val, buf, next_pos
        except json.JSONDecodeError:
            chunk = _read_next_chunk(stream, chunk_size)
            if not chunk:
                raise
            buf = buf[pos:] + chunk
            pos = 0


def _stream_parse_nodes_section(
    decoder: json.JSONDecoder,
    buf: str,
    pos: int,
    stream: IO[Any],
    chunk_size: int,
    graph_cls: type[LogicalGraph],
) -> tuple[NodeDict, str, int]:
    """Incrementally parse the 'nodes' JSON object or array one LogicalNode at a time.

    Ensures peak memory during node deserialization is O(max_node_size) rather than
    allocating all raw node dictionaries simultaneously.

    Args:
        decoder (json.JSONDecoder): Reusable JSONDecoder instance.
        buf (str): Current buffer slice.
        pos (int): Current index within buf.
        stream (IO[Any]): Source input stream.
        chunk_size (int): Chunk size for incremental reads.
        graph_cls (type[LogicalGraph]): LogicalGraph class for recursive subgraph parsing.

    Returns:
        tuple[NodeDict, str, int]: Populated NodeDict and updated (buf, pos).

    Raises:
        json.JSONDecodeError: If the nodes section is not a JSON object or array.
    """
    from ml_switcheroo_ir.graph import NodeDict as _NodeDict

    nodes = _NodeDict()
    buf, pos = _skip_ws_and_tokens(buf, pos, stream, chunk_size, _WS)
    if pos >= len(buf) or buf[pos] not in ("{", "["):
        raise json.JSONDecodeError("Expected '{' or '[' for nodes section", buf, pos)

    opener = buf[pos]
    pos += 1
    if opener == "{":
        while True:
            buf, pos = _skip_ws_and_tokens(buf, pos, stream, chunk_size, _WS_COMMA)
            if pos >= len(buf):
                raise json.JSONDecodeError("Unterminated nodes object", buf, pos)
            if buf[pos] == "}":
                return nodes, buf, pos + 1
            nid, buf, pos = _decode_next_json_value(
                decoder, buf, pos, stream, chunk_size
            )
            buf, pos = _skip_ws_and_tokens(buf, pos, stream, chunk_size, _WS_COLON)
            ndata, buf, pos = _decode_next_json_value(
                decoder, buf, pos, stream, chunk_size
            )
            nodes[str(nid)] = _parse_node_dict(ndata, graph_cls)
    else:
        while True:
            buf, pos = _skip_ws_and_tokens(buf, pos, stream, chunk_size, _WS_COMMA)
            if pos >= len(buf):
                raise json.JSONDecodeError("Unterminated nodes array", buf, pos)
            if buf[pos] == "]":
                return nodes, buf, pos + 1
            ndata, buf, pos = _decode_next_json_value(
                decoder, buf, pos, stream, chunk_size
            )
            node = _parse_node_dict(ndata, graph_cls)
            nodes[node.id] = node


def _apply_edges_list(
    nodes: dict[str, LogicalNode],
    edges_list: Sequence[LogicalEdge | Mapping[str, Any]],
    clear_existing: bool = False,
    edge_value_names: dict[tuple[str, int], str] | None = None,
) -> None:
    """Apply an explicit list of LogicalEdge instances or dicts onto target node inputs.

    Supports both multi-input ports (via explicit distinct ``target_idx`` values,
    e.g. ``Add(x, x)``) and idempotent deduplication when ``target_idx`` is repeated.

    Args:
        nodes (dict[str, LogicalNode]): Graph nodes dictionary.
        edges_list (Sequence[LogicalEdge | Mapping[str, Any]]): Edges to apply.
        clear_existing (bool): Whether to clear existing target_node.inputs first.
        edge_value_names (Optional[dict[tuple[str, int], str]]): Optional mapping of
            (target_id, target_idx) to explicit edge value_name metadata.
    """
    from ml_switcheroo_ir.graph import LogicalEdge as _LogicalEdge

    active_val_names = edge_value_names if edge_value_names is not None else {}
    if clear_existing:
        for node in nodes.values():
            node.inputs.clear()
        active_val_names.clear()

    seen_ports: set[tuple[str, int, str]] = set()

    for edge in edges_list:
        if isinstance(edge, _LogicalEdge):
            src = edge.source
            tgt = edge.target
            src_idx = edge.source_idx
            tgt_idx = edge.target_idx
            val_name = edge.value_name
        else:
            src = str(edge.get("source", ""))
            tgt = str(edge.get("target", ""))
            src_idx = edge.get("source_idx", 0)
            tgt_idx = int(edge.get("target_idx", 0) or 0)
            val_name = edge.get("value_name")

        if tgt not in nodes:
            continue

        target_node = nodes[tgt]
        if val_name and src not in nodes:
            src_name = str(val_name)
        elif src in nodes and src_idx is not None and (clear_existing or src_idx != 0):
            src_name = nodes[src].get_output_name(int(src_idx))
        elif val_name:
            src_name = str(val_name)
        else:
            src_name = src

        port_key = (tgt, tgt_idx, src_name)
        if port_key not in seen_ports:
            seen_ports.add(port_key)
            if src_name not in target_node.inputs or tgt_idx >= len(target_node.inputs):
                actual_idx = len(target_node.inputs)
                target_node.inputs.append(src_name)
            else:
                actual_idx = target_node.inputs.index(src_name)
            if val_name:
                active_val_names[(tgt, actual_idx)] = str(val_name)


def _build_graph_from_parsed_sections(
    graph_cls: type[LogicalGraph],
    nodes: NodeDict,
    meta: Mapping[str, Any],
) -> LogicalGraph:
    """Assemble a LogicalGraph from parsed nodes and top-level metadata sections.

    Args:
        graph_cls (type[LogicalGraph]): Target LogicalGraph class.
        nodes (NodeDict): Deserialized nodes dictionary.
        meta (Mapping[str, Any]): Top-level graph metadata fields.

    Returns:
        LogicalGraph: Constructed LogicalGraph instance.
    """
    edge_value_names: dict[tuple[str, int], str] = {}
    edges_data = meta.get("edges")
    if edges_data and isinstance(edges_data, list):
        _apply_edges_list(
            nodes,
            edges_data,
            clear_existing=False,
            edge_value_names=edge_value_names,
        )

    pipe_topo_raw = meta.get("pipeline_topology")
    pipe_topo = (
        PipelineTopologyConfig.from_dict(pipe_topo_raw)
        if pipe_topo_raw is not None
        else None
    )
    inputs_data = meta.get("inputs")
    graph = graph_cls(
        name=meta.get("name", "Model"),
        nodes=nodes,
        inputs=list(inputs_data) if inputs_data is not None else [],
        input_specs=_parse_input_specs_dict(meta.get("input_specs")),
        outputs=meta.get("outputs"),
        initializers=meta.get("initializers", {}),
        mesh=_parse_mesh_dict(meta.get("mesh")),
        pipeline_topology=pipe_topo,
    )
    if edge_value_names:
        graph._edge_value_names.update(edge_value_names)
    return graph


def _stream_parse_graph(
    stream: IO[Any],
    graph_cls: type[LogicalGraph],
    chunk_size: int = 65536,
) -> LogicalGraph:
    """Incrementally parse a LogicalGraph from a JSON stream using chunked token scanning.

    Maintains O(max_node_size) peak buffer memory by scanning the top-level JSON object
    and decoding each node in the 'nodes' section individually.

    Args:
        stream (IO[Any]): Readable text or binary stream containing JSON.
        graph_cls (type[LogicalGraph]): LogicalGraph class to instantiate.
        chunk_size (int): Read chunk size in bytes/characters (default: 65536).

    Returns:
        LogicalGraph: Deserialized LogicalGraph instance.

    Raises:
        json.JSONDecodeError: If the stream does not contain a valid JSON object.
    """
    from ml_switcheroo_ir.graph import NodeDict as _NodeDict

    decoder = json.JSONDecoder()
    buf = ""
    pos = 0
    buf, pos = _skip_ws_and_tokens(buf, pos, stream, chunk_size, _WS)
    if pos >= len(buf) or buf[pos] != "{":
        raise json.JSONDecodeError("Expected top-level JSON object '{'", buf, pos)
    pos += 1

    nodes = _NodeDict()
    meta: dict[str, Any] = {}

    while True:
        buf, pos = _skip_ws_and_tokens(buf, pos, stream, chunk_size, _WS_COMMA)
        if pos >= len(buf):
            raise json.JSONDecodeError("Unterminated top-level JSON object", buf, pos)
        if buf[pos] == "}":
            break
        key, buf, pos = _decode_next_json_value(decoder, buf, pos, stream, chunk_size)
        buf, pos = _skip_ws_and_tokens(buf, pos, stream, chunk_size, _WS_COLON)
        if key == "nodes":
            nodes, buf, pos = _stream_parse_nodes_section(
                decoder, buf, pos, stream, chunk_size, graph_cls
            )
        else:
            val, buf, pos = _decode_next_json_value(
                decoder, buf, pos, stream, chunk_size
            )
            meta[str(key)] = val

    return _build_graph_from_parsed_sections(graph_cls, nodes, meta)


_NL = chr(10)


def _indent_json_fragment(fragment: str, base_indent: str) -> str:
    """Indent lines 2..N of a multi-line JSON fragment by base_indent.

    Args:
        fragment (str): Multi-line JSON string produced by json.dumps.
        base_indent (str): Indentation prefix for continuation lines.

    Returns:
        str: Indented JSON fragment.
    """
    lines = fragment.splitlines()
    if len(lines) <= 1:
        return fragment
    return lines[0] + _NL + _NL.join(base_indent + line for line in lines[1:])


def _serialize_node_dict(
    node: LogicalNode, format: str = "canonical"
) -> dict[str, Any]:
    """Serialize a LogicalNode to a dictionary while preserving canonical nested subgraphs.

    Args:
        node (LogicalNode): Node to convert to a dictionary.
        format (str): Serialization format ('canonical' or 'legacy').

    Returns:
        dict[str, Any]: Dictionary representation of the node.
    """
    from ml_switcheroo_ir.graph import LogicalGraph as _LogicalGraph

    node_dict = asdict(node)
    if node.subgraphs:
        sub_dict: dict[str, Any] = {}
        for k, v in node.subgraphs.items():
            if isinstance(v, _LogicalGraph):
                sub_dict[k] = v.to_dict(format=format)
            else:
                sub_dict[k] = node_dict["subgraphs"][k]
        node_dict["subgraphs"] = sub_dict
    return node_dict


def _stream_write_graph(
    graph: LogicalGraph,
    fp: IO[Any] | TextIO,
    format: str = "canonical",
    indent: int = 2,
) -> None:
    """Incrementally stream-serialize a LogicalGraph to a writable stream in O(max_node_size) RAM.

    Produces output byte-for-byte identical to ``json.dumps(graph.to_dict(format), sort_keys=True, indent=indent)``
    while streaming ``edges`` and ``nodes`` item by item without materializing the full graph dictionary.

    Args:
        graph (LogicalGraph): Graph to serialize.
        fp (IO[Any] | TextIO): Writable text stream.
        format (str): Serialization format ('canonical' or 'legacy').
        indent (int): Indentation spaces (default: 2).
    """
    pad1 = " " * indent
    pad2 = " " * (indent * 2)

    mesh_val = (
        {
            "shape": dict(graph.mesh.shape),
            "webrtc_topology": (
                graph.mesh.webrtc_topology.to_dict()
                if graph.mesh.webrtc_topology is not None
                else None
            ),
        }
        if graph.mesh is not None
        else None
    )
    pipe_val = (
        graph.pipeline_topology.to_dict()
        if graph.pipeline_topology is not None
        else None
    )
    input_specs_val = {k: asdict(v) for k, v in sorted(graph.input_specs.items())}
    inits_val = dict(sorted(graph.initializers.items()))

    keys_order = [
        "initializers",
        "input_specs",
        "inputs",
        "mesh",
        "name",
        "nodes",
        "outputs",
        "pipeline_topology",
    ]
    if format == "canonical":
        keys_order.insert(0, "edges")

    val_map: dict[str, Any] = {
        "initializers": inits_val,
        "input_specs": input_specs_val,
        "inputs": list(graph.inputs),
        "mesh": mesh_val,
        "name": graph.name,
        "nodes": {},
        "outputs": list(graph.outputs),
        "pipeline_topology": pipe_val,
    }
    fp.write("{" + _NL)
    total_keys = len(keys_order)
    for k_idx, key in enumerate(keys_order):
        comma = "," if k_idx < total_keys - 1 else ""
        fp.write(f'{pad1}"{key}": ')
        if key == "edges":
            edges = graph.edges
            if not edges:
                fp.write(f"[]{comma}{_NL}")
            else:
                fp.write("[" + _NL)
                for e_idx, edge in enumerate(edges):
                    e_comma = "," if e_idx < len(edges) - 1 else ""
                    edge_str = json.dumps(
                        asdict(edge), sort_keys=True, indent=indent, default=str
                    )
                    fp.write(
                        f"{pad2}{_indent_json_fragment(edge_str, pad2)}{e_comma}{_NL}"
                    )
                fp.write(f"{pad1}]{comma}{_NL}")
        elif key == "nodes" and graph.nodes:
            sorted_nids = sorted(graph.nodes.keys())
            fp.write("{" + _NL)
            for n_idx, nid in enumerate(sorted_nids):
                n_comma = "," if n_idx < len(sorted_nids) - 1 else ""
                node_dict = _serialize_node_dict(graph.nodes[nid], format=format)
                node_str = json.dumps(
                    node_dict, sort_keys=True, indent=indent, default=str
                )
                key_json = json.dumps(nid)
                fp.write(
                    f"{pad2}{key_json}: {_indent_json_fragment(node_str, pad2)}{n_comma}{_NL}"
                )
            fp.write(f"{pad1}}}{comma}{_NL}")
        else:
            val_str = json.dumps(
                val_map[key], sort_keys=True, indent=indent, default=str
            )
            fp.write(f"{_indent_json_fragment(val_str, pad1)}{comma}{_NL}")
    fp.write("}")
