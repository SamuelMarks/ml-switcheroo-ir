"""Streaming JSON serialization, chunked deserialization, and compression utilities for LogicalGraph."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import copy
import json
import string
import typing
import typing_extensions

import pydantic

from ml_switcheroo_ir.distributed import (
    LogicalMesh,
    PartitionAxisAtom,
    PartitionAxisElement,
    PartitionSpec,
    PipelineTopologyConfig,
    WebRTCSignalingTopology,
)
from ml_switcheroo_ir.shapes import DimensionType
from ml_switcheroo_ir.types import (
    AttributeValue,
    DType,
    InitializerValue,
    JsonValue,
    TensorSpec,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir.graph import LogicalEdge, LogicalGraph, LogicalNode, NodeDict

    RawGraphValue: typing_extensions.TypeAlias = (
        AttributeValue
        | InitializerValue
        | JsonValue
        | LogicalGraph
        | LogicalMesh
        | PartitionSpec
        | PipelineTopologyConfig
        | Collection["RawGraphValue"]
        | Sequence["RawGraphValue"]
        | Mapping[str, "RawGraphValue"]
    )
else:
    RawGraphValue: typing_extensions.TypeAlias = JsonValue

__all__ = [
    "RawGraphValue",
    "SerializedEdgeModel",
    "SerializedGraphModel",
    "SerializedNodeModel",
    "ZstdCompressorProtocol",
    "ZstdDecompressorProtocol",
    "ZstdModuleProtocol",
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


class SerializedEdgeModel(pydantic.BaseModel):
    """Pydantic model for a serialized LogicalEdge."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    source: str
    source_idx: int | None
    target: str
    target_idx: int
    value_name: str | None


class SerializedNodeModel(pydantic.BaseModel):
    """Pydantic model for a serialized LogicalNode."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    attributes: dict[str, AttributeValue]
    device: str | None
    domain: str
    dtype: DType | None
    id: str
    inputs: list[str]
    op_type: str
    output_specs: list[TensorSpec]
    outputs: list[str]
    shape_metadata: tuple[DimensionType, ...] | None
    sharding: PartitionSpec | None
    source_ast_ref: str | None
    stream: str | None
    subgraphs: dict[str, SerializedGraphModel | AttributeValue]
    version: int


class SerializedGraphModel(pydantic.BaseModel):
    """Pydantic model for a serialized LogicalGraph."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    edges: list[SerializedEdgeModel]
    initializers: dict[str, InitializerValue]
    input_specs: dict[str, TensorSpec]
    inputs: list[str]
    mesh: LogicalMesh | None
    name: str
    nodes: dict[str, SerializedNodeModel]
    outputs: list[str]
    pipeline_topology: PipelineTopologyConfig | None


SerializedNodeModel.model_rebuild()
SerializedGraphModel.model_rebuild()


class ZstdCompressorProtocol(typing.Protocol):
    """Protocol for zstandard.ZstdCompressor."""

    def stream_writer(
        self, writer: typing.BinaryIO
    ) -> typing.ContextManager[typing.BinaryIO]:
        """Return context-managed binary stream writer.

        Args:
            writer (typing.BinaryIO): Underlying binary writer.

        Returns:
            typing.ContextManager[typing.BinaryIO]: Context manager yielding binary stream.
        """


class ZstdDecompressorProtocol(typing.Protocol):
    """Protocol for zstandard.ZstdDecompressor."""

    def stream_reader(
        self, source: typing.BinaryIO
    ) -> typing.ContextManager[typing.BinaryIO]:
        """Return context-managed binary stream reader.

        Args:
            source (typing.BinaryIO): Underlying binary reader.

        Returns:
            typing.ContextManager[typing.BinaryIO]: Context manager yielding binary stream.
        """


class ZstdModuleProtocol(typing.Protocol):
    """Protocol for optional zstandard module."""

    def ZstdCompressor(self) -> ZstdCompressorProtocol:
        """Instantiate a ZstdCompressor.

        Returns:
            ZstdCompressorProtocol: Compressor instance.
        """

    def ZstdDecompressor(self) -> ZstdDecompressorProtocol:
        """Instantiate a ZstdDecompressor.

        Returns:
            ZstdDecompressorProtocol: Decompressor instance.
        """


def _load_zstandard() -> ZstdModuleProtocol | None:
    """Import and return the optional zstandard module if installed.

    Returns:
        ZstdModuleProtocol | None: The zstandard module, or None if unavailable.
    """
    try:
        import zstandard as zstd
    except ImportError:
        return None
    return zstd


zstandard: ZstdModuleProtocol | None = _load_zstandard()


def _normalize_partition_axis_atom(
    raw: RawGraphValue | PartitionAxisElement,
) -> PartitionAxisAtom:
    """Normalize a single leaf partition axis element.

    Args:
        raw (RawGraphValue | PartitionAxisElement): Raw leaf element.

    Returns:
        PartitionAxisAtom: Normalized atom.
    """
    return (
        raw
        if raw is None or (isinstance(raw, (str, int)) and not isinstance(raw, bool))
        else str(raw)
    )


def _normalize_partition_axis(
    raw: RawGraphValue | PartitionAxisElement,
) -> PartitionAxisElement:
    """Recursively normalize raw JSON sharding axis into PartitionAxisElement.

    Args:
        raw (RawGraphValue | PartitionAxisElement): Raw axis element from JSON.

    Returns:
        PartitionAxisElement: Normalized axis element.
    """
    if isinstance(raw, (list, tuple)):
        return tuple(
            tuple(_normalize_partition_axis_atom(sub) for sub in item)
            if isinstance(item, (list, tuple))
            else _normalize_partition_axis_atom(item)
            for item in raw
        )
    return _normalize_partition_axis_atom(raw)


def _coerce_attribute_value(val: RawGraphValue) -> AttributeValue:
    """Recursively narrow a RawGraphValue into an AttributeValue.

    Args:
        val (RawGraphValue): Raw graph value.

    Returns:
        AttributeValue: Narrowed AttributeValue.
    """
    from ml_switcheroo_ir.shapes import SymInt, SymNode
    from ml_switcheroo_ir.types import TensorShape

    if val is None or isinstance(
        val,
        (
            int,
            float,
            str,
            bool,
            bytes,
            SymNode,
            SymInt,
            DType,
            TensorShape,
            TensorSpec,
        ),
    ):
        return val
    if isinstance(val, Mapping):
        return {str(k): _coerce_attribute_value(v) for k, v in val.items()}
    return (
        str(val)
        if isinstance(val, (LogicalMesh, PartitionSpec, PipelineTopologyConfig))
        else [_coerce_attribute_value(item) for item in val]
    )


def _coerce_initializer_value(val: RawGraphValue) -> InitializerValue:
    """Recursively narrow a RawGraphValue into an InitializerValue.

    Args:
        val (RawGraphValue): Raw graph value.

    Returns:
        InitializerValue: Narrowed InitializerValue.
    """
    if isinstance(val, (int, float, bool, str, bytes)):
        return val
    if isinstance(val, Mapping):
        return {str(k): _coerce_initializer_value(v) for k, v in val.items()}
    return (
        str(val)
        if isinstance(
            val,
            (
                LogicalMesh,
                PartitionSpec,
                PipelineTopologyConfig,
                DType,
                TensorSpec,
            ),
        )
        or not isinstance(val, (list, tuple, set, frozenset))
        else [_coerce_initializer_value(item) for item in val]
    )


def _parse_node_dict(
    ndata: Mapping[str, RawGraphValue],
    graph_cls: type[LogicalGraph],
) -> LogicalNode:
    """Deserialize a single LogicalNode from a raw dictionary representation.

    Args:
        ndata (Mapping[str, RawGraphValue]):
            Raw dictionary describing a node.
        graph_cls (type[LogicalGraph]): LogicalGraph class used for recursive subgraphs.

    Returns:
        LogicalNode: Deserialized LogicalNode instance.
    """
    from ml_switcheroo_ir.graph import LogicalNode
    from ml_switcheroo_ir.sentinels import NoTangent, ZeroTangent

    node_id = str(ndata.get("id", ""))
    op_type_val = str(ndata.get("op_type", ""))
    domain_val = str(ndata.get("domain", "ai.onnx"))
    raw_ver = ndata.get("version", 1)
    version_val = int(raw_ver) if isinstance(raw_ver, (int, str)) else 1

    raw_attrs = ndata.get("attributes")
    attributes_val: dict[str, AttributeValue] = (
        {str(k): _coerce_attribute_value(v) for k, v in raw_attrs.items()}
        if isinstance(raw_attrs, Mapping)
        else {}
    )

    raw_inputs = ndata.get("inputs")
    inputs_val: list[str] = (
        [str(x) for x in raw_inputs]
        if isinstance(raw_inputs, Sequence) and not isinstance(raw_inputs, (str, bytes))
        else []
    )

    raw_outputs = ndata.get("outputs")
    outputs_val: list[str] | None = (
        [str(x) for x in raw_outputs]
        if isinstance(raw_outputs, Sequence)
        and not isinstance(raw_outputs, (str, bytes))
        else None
    )

    raw_shape = ndata.get("shape_metadata")
    shape_val: tuple[DimensionType, ...] | None = (
        tuple(int(d) if isinstance(d, int) else str(d) for d in raw_shape)
        if isinstance(raw_shape, Sequence) and not isinstance(raw_shape, (str, bytes))
        else None
    )

    sharding_raw = ndata.get("sharding")
    sharding_val: PartitionSpec | None = None
    if isinstance(sharding_raw, PartitionSpec):
        sharding_val = sharding_raw
    elif isinstance(sharding_raw, Mapping):
        raw_axes = sharding_raw.get("axes", ())
        axes_seq = (
            raw_axes
            if isinstance(raw_axes, Sequence) and not isinstance(raw_axes, (str, bytes))
            else ()
        )
        sharding_val = PartitionSpec(
            axes=tuple(_normalize_partition_axis(a) for a in axes_seq)
        )

    dtype_raw = ndata.get("dtype")
    dtype_val: DType | None = (
        dtype_raw
        if isinstance(dtype_raw, DType)
        else (DType.from_str(str(dtype_raw)) if dtype_raw is not None else None)
    )

    output_specs_raw = ndata.get("output_specs")
    output_specs_val: list[TensorSpec] = []
    for s in (
        output_specs_raw
        if isinstance(output_specs_raw, Sequence)
        and not isinstance(output_specs_raw, (str, bytes))
        else []
    ):
        if isinstance(s, TensorSpec):
            output_specs_val.append(s)
        else:
            s_map = s if isinstance(s, Mapping) else {}
            raw_s_shape = s_map.get("shape", ())
            s_shape: list[DimensionType] = (
                [int(x) if isinstance(x, int) else str(x) for x in raw_s_shape]
                if isinstance(raw_s_shape, Sequence)
                and not isinstance(raw_s_shape, (str, bytes))
                else []
            )
            s_dtype = str(s_map.get("dtype", "float32"))
            raw_sparsity = s_map.get("sparsity")
            s_sparsity = str(raw_sparsity) if raw_sparsity is not None else None
            output_specs_val.append(
                TensorSpec(shape=s_shape, dtype=s_dtype, sparsity=s_sparsity)
            )

    subgraphs_val = _parse_subgraphs_from_node_dict(ndata, graph_cls)
    raw_ast_ref = ndata.get("source_ast_ref")
    ast_ref_val = str(raw_ast_ref) if raw_ast_ref is not None else None
    raw_device = ndata.get("device")
    device_val = str(raw_device) if raw_device is not None else None
    raw_stream = ndata.get("stream")
    stream_val = str(raw_stream) if raw_stream is not None else None

    if domain_val == "ml.switcheroo.ad" and op_type_val == "ZeroTangent":
        return ZeroTangent(
            id=node_id,
            op_type=op_type_val,
            domain=domain_val,
            version=version_val,
            attributes=attributes_val,
            inputs=inputs_val,
            outputs=outputs_val,
            shape_metadata=shape_val,
            source_ast_ref=ast_ref_val,
            sharding=sharding_val,
            dtype=dtype_val,
            output_specs=output_specs_val,
            subgraphs=subgraphs_val,
            device=device_val,
            stream=stream_val,
        )
    if domain_val == "ml.switcheroo.ad" and op_type_val == "NoTangent":
        return NoTangent(
            id=node_id,
            op_type=op_type_val,
            domain=domain_val,
            version=version_val,
            attributes=attributes_val,
            inputs=inputs_val,
            outputs=outputs_val,
            shape_metadata=shape_val,
            source_ast_ref=ast_ref_val,
            sharding=sharding_val,
            dtype=dtype_val,
            output_specs=output_specs_val,
            subgraphs=subgraphs_val,
            device=device_val,
            stream=stream_val,
        )
    return LogicalNode(
        id=node_id,
        op_type=op_type_val,
        domain=domain_val,
        version=version_val,
        attributes=attributes_val,
        inputs=inputs_val,
        outputs=outputs_val,
        shape_metadata=shape_val,
        source_ast_ref=ast_ref_val,
        sharding=sharding_val,
        dtype=dtype_val,
        output_specs=output_specs_val,
        subgraphs=subgraphs_val,
        device=device_val,
        stream=stream_val,
    )


def _parse_subgraphs_from_node_dict(
    ndata: Mapping[str, RawGraphValue],
    graph_cls: type[LogicalGraph],
) -> dict[str, LogicalGraph | AttributeValue]:
    """Extract and deserialize nested subgraphs from a node dictionary.

    Args:
        ndata (Mapping[str, RawGraphValue]): Node dictionary.
        graph_cls (type[LogicalGraph]): LogicalGraph class for recursive deserialization.

    Returns:
        dict[str, LogicalGraph | AttributeValue]: Mapping of subgraph names to LogicalGraph instances or AttributeValues.
    """
    resolved_subgraphs: dict[str, LogicalGraph | AttributeValue] = {}
    subgraphs_data = ndata.get("subgraphs")
    if isinstance(subgraphs_data, Mapping):
        for sub_k, sub_v in subgraphs_data.items():
            if isinstance(sub_v, graph_cls):
                resolved_subgraphs[str(sub_k)] = sub_v
            elif isinstance(sub_v, Mapping) and ("nodes" in sub_v or "name" in sub_v):
                resolved_subgraphs[str(sub_k)] = graph_cls.from_dict(sub_v)
            else:
                resolved_subgraphs[str(sub_k)] = _coerce_attribute_value(sub_v)
    return resolved_subgraphs


def _parse_mesh_dict(
    mesh_data: RawGraphValue,
) -> LogicalMesh | None:
    """Deserialize an optional LogicalMesh dictionary.

    Args:
        mesh_data (RawGraphValue): Raw mesh dictionary or None.

    Returns:
        LogicalMesh | None: Deserialized LogicalMesh or None.
    """
    if not isinstance(mesh_data, Mapping) or not mesh_data:
        return None
    raw_webrtc = mesh_data.get("webrtc_topology")
    webrtc_topo = (
        WebRTCSignalingTopology.model_validate(raw_webrtc)
        if isinstance(raw_webrtc, Mapping)
        else None
    )
    raw_shape = mesh_data.get("shape", {})
    shape_dict: dict[str, int] = (
        {str(k): int(v) for k, v in raw_shape.items() if isinstance(v, (int, str))}
        if isinstance(raw_shape, Mapping)
        else {}
    )
    return LogicalMesh(shape=shape_dict, webrtc_topology=webrtc_topo)


def _parse_input_specs_dict(
    input_specs_raw: RawGraphValue,
) -> dict[str, TensorSpec]:
    """Deserialize an optional input_specs dictionary into TensorSpec instances.

    Args:
        input_specs_raw (RawGraphValue):
            Raw input specs dictionary.

    Returns:
        dict[str, TensorSpec]: Parsed input specifications.
    """
    input_specs: dict[str, TensorSpec] = {}
    if isinstance(input_specs_raw, Mapping):
        for k, v in input_specs_raw.items():
            if isinstance(v, TensorSpec):
                input_specs[str(k)] = v
            elif isinstance(v, Mapping):
                raw_shape = v.get("shape", ())
                shape_seq: list[DimensionType] = (
                    [int(x) if isinstance(x, int) else str(x) for x in raw_shape]
                    if isinstance(raw_shape, Sequence)
                    and not isinstance(raw_shape, (str, bytes))
                    else []
                )
                dtype_str = str(v.get("dtype", "float32"))
                raw_sparsity = v.get("sparsity")
                sparsity_str = str(raw_sparsity) if raw_sparsity is not None else None
                input_specs[str(k)] = TensorSpec(
                    shape=shape_seq, dtype=dtype_str, sparsity=sparsity_str
                )
    return input_specs


def _read_next_chunk(
    stream: typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO,
    chunk_size: int,
) -> str:
    """Read up to chunk_size from a text or binary stream as a UTF-8 string.

    Args:
        stream (typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO): Readable text or binary stream.
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
    stream: typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO,
    chunk_size: int,
    tokens: str,
) -> tuple[str, int]:
    """Advance buffer cursor past whitespace and delimiter characters, refilling from stream as needed.

    Args:
        buf (str): Current active buffer slice.
        pos (int): Current index within buf.
        stream (typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO): Source input stream.
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
    stream: typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO,
    chunk_size: int,
) -> tuple[JsonValue, str, int]:
    """Decode a single JSON value starting at pos, incrementally reading chunks on boundary straddles.

    Args:
        decoder (json.JSONDecoder): Reusable JSONDecoder instance.
        buf (str): Current buffer slice.
        pos (int): Current position within buf.
        stream (typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO): Source input stream.
        chunk_size (int): Chunk size for incremental reads.

    Returns:
        tuple[JsonValue, str, int]: Decoded JSON value and updated (buf, next_pos).

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
    stream: typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO,
    chunk_size: int,
    graph_cls: type[LogicalGraph],
) -> tuple[NodeDict, str, int]:
    """Incrementally parse the 'nodes' JSON object one LogicalNode at a time.

    Ensures peak memory during node deserialization is O(max_node_size) rather than
    allocating all raw node dictionaries simultaneously.

    Args:
        decoder (json.JSONDecoder): Reusable JSONDecoder instance.
        buf (str): Current buffer slice.
        pos (int): Current index within buf.
        stream (typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO): Source input stream.
        chunk_size (int): Chunk size for incremental reads.
        graph_cls (type[LogicalGraph]): LogicalGraph class for recursive subgraph parsing.

    Returns:
        tuple[NodeDict, str, int]: Populated NodeDict and updated (buf, pos).

    Raises:
        json.JSONDecodeError: If the nodes section is not a JSON object.
    """
    from ml_switcheroo_ir.graph import NodeDict as _NodeDict

    nodes = _NodeDict()
    buf, pos = _skip_ws_and_tokens(buf, pos, stream, chunk_size, _WS)
    if pos >= len(buf) or buf[pos] != "{":
        raise json.JSONDecodeError("Expected '{' for nodes section", buf, pos)

    pos += 1
    while True:
        buf, pos = _skip_ws_and_tokens(buf, pos, stream, chunk_size, _WS_COMMA)
        if pos >= len(buf):
            raise json.JSONDecodeError("Unterminated nodes object", buf, pos)
        if buf[pos] == "}":
            return nodes, buf, pos + 1
        nid, buf, pos = _decode_next_json_value(decoder, buf, pos, stream, chunk_size)
        buf, pos = _skip_ws_and_tokens(buf, pos, stream, chunk_size, _WS_COLON)
        ndata, buf, pos = _decode_next_json_value(decoder, buf, pos, stream, chunk_size)
        assert isinstance(ndata, Mapping)
        nodes[str(nid)] = _parse_node_dict(ndata, graph_cls)


def _apply_edges_list(
    nodes: dict[str, LogicalNode],
    edges_list: Sequence[LogicalEdge | Mapping[str, RawGraphValue]],
    clear_existing: bool = False,
    edge_value_names: dict[tuple[str, int], str] | None = None,
) -> None:
    """Apply an explicit list of LogicalEdge instances or dicts onto target node inputs.

    Supports both multi-input ports (via explicit distinct ``target_idx`` values,
    e.g. ``Add(x, x)``) and idempotent deduplication when ``target_idx`` is repeated.

    Args:
        nodes (dict[str, LogicalNode]): Graph nodes dictionary.
        edges_list (Sequence[LogicalEdge | Mapping[str, RawGraphValue]]): Edges to apply.
        clear_existing (bool): Whether to clear existing target_node.inputs first.
        edge_value_names (dict[tuple[str, int], str] | None): Optional mapping of
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
            raw_src_idx = edge.get("source_idx", 0)
            src_idx = int(raw_src_idx) if isinstance(raw_src_idx, int) else None
            raw_tgt_idx = edge.get("target_idx", 0)
            tgt_idx = int(raw_tgt_idx) if isinstance(raw_tgt_idx, int) else 0
            raw_val_name = edge.get("value_name")
            val_name = str(raw_val_name) if raw_val_name is not None else None

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
    meta: Mapping[str, RawGraphValue],
) -> LogicalGraph:
    """Assemble a LogicalGraph from parsed nodes and top-level metadata sections.

    Args:
        graph_cls (type[LogicalGraph]): Target LogicalGraph class.
        nodes (NodeDict): Deserialized nodes dictionary.
        meta (Mapping[str, RawGraphValue]):
            Top-level graph metadata fields.

    Returns:
        LogicalGraph: Constructed LogicalGraph instance.
    """
    edge_value_names: dict[tuple[str, int], str] = {}
    edges_data = meta.get("edges")
    if isinstance(edges_data, list):
        typed_edges: list[LogicalEdge | Mapping[str, RawGraphValue]] = [
            e for e in edges_data if isinstance(e, Mapping)
        ]
        _apply_edges_list(
            nodes,
            typed_edges,
            clear_existing=False,
            edge_value_names=edge_value_names,
        )

    pipe_topo_raw = meta.get("pipeline_topology")
    pipe_topo = (
        PipelineTopologyConfig.model_validate(pipe_topo_raw)
        if isinstance(pipe_topo_raw, Mapping)
        else None
    )
    inputs_data = meta.get("inputs")
    inputs_list: list[str] = (
        [str(x) for x in inputs_data]
        if isinstance(inputs_data, Sequence)
        and not isinstance(inputs_data, (str, bytes))
        else []
    )
    outputs_data = meta.get("outputs")
    outputs_list: list[str] | None = (
        [str(x) for x in outputs_data]
        if isinstance(outputs_data, Sequence)
        and not isinstance(outputs_data, (str, bytes))
        else None
    )
    inits_raw = meta.get("initializers")
    inits_dict: dict[str, InitializerValue] = (
        {str(k): _coerce_initializer_value(v) for k, v in inits_raw.items()}
        if isinstance(inits_raw, Mapping)
        else {}
    )
    graph = graph_cls(
        name=str(meta.get("name", "Model")),
        nodes=nodes,
        inputs=inputs_list,
        input_specs=_parse_input_specs_dict(meta.get("input_specs")),
        outputs=outputs_list,
        initializers=inits_dict,
        mesh=_parse_mesh_dict(meta.get("mesh")),
        pipeline_topology=pipe_topo,
    )
    graph._edge_value_names.update(edge_value_names)
    return graph


def _stream_parse_graph(
    stream: typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO,
    graph_cls: type[LogicalGraph],
    chunk_size: int = 65536,
) -> LogicalGraph:
    """Incrementally parse a LogicalGraph from a JSON stream using chunked token scanning.

    Maintains O(max_node_size) peak buffer memory by scanning the top-level JSON object
    and decoding each node in the 'nodes' section individually.

    Args:
        stream (typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO): Readable text or binary stream containing JSON.
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
    meta: dict[str, JsonValue] = {}

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


def _serialize_node_dict(node: LogicalNode) -> dict[str, AttributeValue]:
    """Serialize a LogicalNode to a dictionary while preserving canonical nested subgraphs.

    Args:
        node (LogicalNode): Node to convert to a dictionary.

    Returns:
        dict[str, AttributeValue]: Dictionary representation of the node.
    """
    from ml_switcheroo_ir.graph import LogicalGraph as _LogicalGraph

    sub_dict: dict[str, AttributeValue] = {}
    for k, v in node.subgraphs.items():
        if isinstance(v, _LogicalGraph):
            sub_dict[k] = v.to_dict()
        else:
            sub_dict[k] = copy.deepcopy(v)

    sharding_dict: dict[str, tuple[PartitionAxisElement, ...]] | None = (
        {"axes": node.sharding.axes} if node.sharding is not None else None
    )
    output_specs_list: list[dict[str, AttributeValue]] = [
        {"dtype": spec.dtype, "shape": spec.shape, "sparsity": spec.sparsity}
        for spec in node.output_specs
    ]

    return {
        "attributes": copy.deepcopy(node.attributes),
        "device": node.device,
        "domain": node.domain,
        "dtype": node.dtype,
        "id": node.id,
        "inputs": list(node.inputs),
        "op_type": node.op_type,
        "output_specs": output_specs_list,
        "outputs": list(node.outputs),
        "shape_metadata": node.shape_metadata,
        "sharding": sharding_dict,
        "source_ast_ref": node.source_ast_ref,
        "stream": node.stream,
        "subgraphs": sub_dict,
        "version": node.version,
    }


def _stream_write_graph(
    graph: LogicalGraph,
    fp: typing.IO[str] | typing.TextIO,
    indent: int = 2,
) -> None:
    """Incrementally stream-serialize a LogicalGraph to a writable stream in O(max_node_size) RAM.

    Produces output byte-for-byte identical to ``json.dumps(graph.to_dict(), sort_keys=True, indent=indent)``
    while streaming ``edges`` and ``nodes`` item by item without materializing the full graph dictionary.

    Args:
        graph (LogicalGraph): Graph to serialize.
        fp (typing.IO[str] | typing.TextIO): Writable text stream.
        indent (int): Indentation spaces (default: 2).
    """
    pad1 = " " * indent
    pad2 = " " * (indent * 2)

    mesh_val: dict[str, JsonValue] | None = (
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
    input_specs_val: dict[str, dict[str, AttributeValue]] = {
        k: {"dtype": v.dtype, "shape": v.shape, "sparsity": v.sparsity}
        for k, v in sorted(graph.input_specs.items())
    }
    inits_val = dict(sorted(graph.initializers.items()))

    keys_order = [
        "edges",
        "initializers",
        "input_specs",
        "inputs",
        "mesh",
        "name",
        "nodes",
        "outputs",
        "pipeline_topology",
    ]

    val_map: dict[str, AttributeValue | dict[str, InitializerValue]] = {
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
                    edge_dict: dict[str, str | int | None] = {
                        "source": edge.source,
                        "source_idx": edge.source_idx,
                        "target": edge.target,
                        "target_idx": edge.target_idx,
                        "value_name": edge.value_name,
                    }
                    edge_str = json.dumps(
                        edge_dict, sort_keys=True, indent=indent, default=str
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
                node_dict = _serialize_node_dict(graph.nodes[nid])
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
