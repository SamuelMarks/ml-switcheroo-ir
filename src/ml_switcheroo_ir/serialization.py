"""Lossless canonical dictionary, streaming JSON, and compressed serialization for LogicalGraph."""

from __future__ import annotations

import base64
import copy
import json
import string
import typing
from collections.abc import Collection, Mapping, MutableMapping, Sequence

import typing_extensions

from ml_switcheroo_ir.distributed import (
    LogicalMesh,
    PartitionAxisAtom,
    PartitionAxisElement,
    PartitionSpec,
    PipelineTopologyConfig,
    WebRTCSignalingTopology,
)
from ml_switcheroo_ir.shapes import (
    DimensionType,
    SymBinaryOp,
    SymCondition,
    SymConst,
    SymInt,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
    parse_symbolic_expr,
)
from ml_switcheroo_ir.types import (
    AttributeValue,
    DType,
    InitializerValue,
    JsonValue,
    TensorShape,
    TensorSpec,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir.graph import LogicalEdge, LogicalGraph, LogicalNode, NodeDict

RawGraphValue: typing_extensions.TypeAlias = typing.Union[
    AttributeValue,
    InitializerValue,
    JsonValue,
    "LogicalGraph",
    LogicalMesh,
    PartitionSpec,
    PipelineTopologyConfig,
    Collection["RawGraphValue"],
    Sequence["RawGraphValue"],
    Mapping[str, "RawGraphValue"],
]

_TAG_KEY: typing.Final[str] = "__sw_type__"

__all__ = [
    "RawGraphValue",
    "ZstdCompressorProtocol",
    "ZstdDecompressorProtocol",
    "ZstdModuleProtocol",
    "_apply_edges_list",
    "_build_graph_from_parsed_sections",
    "_decode_next_json_value",
    "_encode_special_types",
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
    "graph_to_dict",
    "zstandard",
]


class ZstdCompressorProtocol(typing.Protocol):
    """Protocol for zstandard.ZstdCompressor."""

    def stream_writer(self, writer: typing.BinaryIO) -> typing.ContextManager[typing.BinaryIO]:
        """Return context-managed binary stream writer.

        Args:
            writer (typing.BinaryIO): Underlying binary writer.

        Returns:
            typing.ContextManager[typing.BinaryIO]: Context manager yielding binary stream.
        """


class ZstdDecompressorProtocol(typing.Protocol):
    """Protocol for zstandard.ZstdDecompressor."""

    def stream_reader(self, source: typing.BinaryIO) -> typing.ContextManager[typing.BinaryIO]:
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


def _encode_sym_condition(cond: SymCondition) -> JsonValue:
    """Encode a ``SymPiecewise`` branch condition into a JSON-compatible representation.

    Args:
        cond (SymCondition): Piecewise branch condition predicate.

    Returns:
        JsonValue: Encoded condition payload.
    """
    if isinstance(cond, SymNode):
        return _encode_symnode(cond)
    if isinstance(cond, (bool, str)):
        return cond
    return str(cond)


def _encode_symnode(node: SymNode) -> dict[str, JsonValue]:
    """Encode a ``SymNode`` AST into a lossless tagged JSON dictionary.

    Args:
        node (SymNode): Symbolic expression node to encode.

    Returns:
        dict[str, JsonValue]: Tagged JSON dictionary preserving exact AST structure.
    """
    if isinstance(node, SymVar):
        return {_TAG_KEY: "SymNode", "expr": str(node), "kind": "var", "name": node.name}
    if isinstance(node, SymConst):
        return {_TAG_KEY: "SymNode", "expr": str(node), "kind": "const", "value": node.value}
    if isinstance(node, SymBinaryOp):
        return {
            _TAG_KEY: "SymNode",
            "expr": str(node),
            "kind": "binary_op",
            "left": _encode_symnode(node.left),
            "op": node.op,
            "right": _encode_symnode(node.right),
        }
    if isinstance(node, SymUnaryOp):
        return {
            _TAG_KEY: "SymNode",
            "expr": str(node),
            "kind": "unary_op",
            "op": node.op,
            "operand": _encode_symnode(node.operand),
        }
    if isinstance(node, SymPiecewise):
        return {
            _TAG_KEY: "SymNode",
            "cases": [[_encode_sym_condition(c), _encode_symnode(e)] for c, e in node.cases],
            "default": _encode_symnode(node.default),
            "kind": "piecewise",
        }
    return {_TAG_KEY: "SymNode", "expr": str(node)}


def _decode_sym_condition(raw: RawGraphValue) -> SymCondition:
    """Decode a serialized ``SymPiecewise`` condition back into a ``SymCondition``.

    Args:
        raw (RawGraphValue): Raw JSON condition entry.

    Returns:
        SymCondition: Reconstructed condition predicate.
    """
    if isinstance(raw, Mapping) and raw.get(_TAG_KEY) == "SymNode":
        return _decode_symnode(raw)
    if isinstance(raw, bool):
        return raw
    return str(raw)


def _decode_piecewise_symnode(raw: Mapping[str, RawGraphValue]) -> SymPiecewise:
    """Decode a ``kind == 'piecewise'`` serialized ``SymNode`` dictionary.

    Args:
        raw (Mapping[str, RawGraphValue]): Serialized piecewise dictionary.

    Returns:
        SymPiecewise: Reconstructed ``SymPiecewise`` AST node.

    Raises:
        ValueError: If ``default`` or ``cases`` are malformed.
    """
    raw_cases = raw.get("cases", ())
    def_raw = raw.get("default")
    if not isinstance(def_raw, Mapping) or not isinstance(raw_cases, Sequence):
        raise ValueError("Malformed piecewise fields in serialized SymNode.")
    cases: list[tuple[SymCondition, SymNode]] = [
        (_decode_sym_condition(pair[0]), _decode_symnode(pair[1]))
        for pair in raw_cases
        if isinstance(pair, Sequence) and len(pair) == 2 and isinstance(pair[1], Mapping)
    ]
    return SymPiecewise(cases=cases, default=_decode_symnode(def_raw))


def _decode_symnode(raw: Mapping[str, RawGraphValue]) -> SymNode:
    """Decode a tagged ``SymNode`` dictionary back into a ``SymNode`` AST.

    Args:
        raw (Mapping[str, RawGraphValue]): Tagged dictionary with ``__sw_type__ == 'SymNode'``.

    Returns:
        SymNode: Reconstructed symbolic expression node.

    Raises:
        ValueError: If the tagged ``SymNode`` payload is malformed.
    """
    kind = raw.get("kind")
    if kind == "var":
        return SymVar(str(raw["name"]))
    if kind == "const":
        val = raw.get("value")
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"Invalid SymConst value in serialized SymNode: {val!r}")
        return SymConst(val)
    if kind == "binary_op":
        left_raw, right_raw = raw.get("left"), raw.get("right")
        if not isinstance(left_raw, Mapping) or not isinstance(right_raw, Mapping):
            raise ValueError("Malformed binary_op operands in serialized SymNode.")
        return SymBinaryOp(str(raw["op"]), _decode_symnode(left_raw), _decode_symnode(right_raw))
    if kind == "unary_op":
        op_raw = raw.get("operand")
        if not isinstance(op_raw, Mapping):
            raise ValueError("Malformed unary_op operand in serialized SymNode.")
        return SymUnaryOp(str(raw["op"]), _decode_symnode(op_raw))
    if kind == "piecewise":
        return _decode_piecewise_symnode(raw)
    expr = raw.get("expr")
    if isinstance(expr, str):
        return parse_symbolic_expr(expr)
    raise ValueError(f"Unsupported serialized SymNode payload: {raw!r}")


def _decode_symint_mapping(d: Mapping[str, RawGraphValue]) -> SymInt:
    """Decode a ``__sw_type__ == 'SymInt'`` mapping into a ``SymInt`` instance.

    Args:
        d (Mapping[str, RawGraphValue]): Serialized ``SymInt`` mapping.

    Returns:
        SymInt: Reconstructed ``SymInt``.
    """
    node_raw = d.get("node")
    if isinstance(node_raw, Mapping):
        return SymInt(_decode_symnode(node_raw))
    return SymInt(parse_symbolic_expr(str(d.get("expr", "0"))))


def _parse_raw_dim_entry(d: RawGraphValue) -> DimensionType:
    """Parse a raw JSON dimension entry into ``int``, ``str``, ``SymNode``, or ``SymInt``, rejecting ``bool``.

    Args:
        d (RawGraphValue): Raw JSON dimension item.

    Returns:
        DimensionType: Reconstructed dimension value.

    Raises:
        TypeError: If ``d`` is a boolean value.
    """
    if isinstance(d, bool):
        raise TypeError(f"Boolean value {d!r} is not a valid serialized tensor dimension.")
    if isinstance(d, (SymNode, SymInt, int)):
        return int(d) if isinstance(d, int) else d
    if isinstance(d, Mapping):
        tag = d.get(_TAG_KEY)
        if tag == "SymNode":
            return _decode_symnode(d)
        if tag == "SymInt":
            return _decode_symint_mapping(d)
    s = str(d)
    if s.isidentifier() or s.lstrip("-").isdigit():
        return s
    try:
        return parse_symbolic_expr(s)
    except ValueError:
        return s


def _encode_dimension(dim: DimensionType) -> JsonValue:
    """Encode a single ``DimensionType`` entry losslessly for JSON serialization.

    Args:
        dim (DimensionType): Dimension value (``int``, ``str``, ``SymNode``, or ``SymInt``).

    Returns:
        JsonValue: JSON-compatible representation preserving exact symbolic type.
    """
    if isinstance(dim, SymInt):
        return {_TAG_KEY: "SymInt", "expr": str(dim), "node": _encode_symnode(dim.node)}
    if isinstance(dim, SymNode):
        return _encode_symnode(dim)
    if isinstance(dim, int) and not isinstance(dim, bool):
        return dim
    return str(dim)


def _encode_rich_attribute_leaf(
    val: bytes | DType | SymInt | SymNode | TensorShape | TensorSpec,
) -> dict[str, JsonValue]:
    """Encode a non-primitive leaf ``AttributeValue`` into a ``__sw_type__`` dictionary.

    Args:
        val (bytes | DType | SymInt | SymNode | TensorShape | TensorSpec): Rich attribute leaf.

    Returns:
        dict[str, JsonValue]: Tagged dictionary preserving exact runtime type.
    """
    if isinstance(val, (bytes, DType)):
        return (
            {_TAG_KEY: "bytes", "data": base64.b64encode(val).decode("ascii")}
            if isinstance(val, bytes)
            else {_TAG_KEY: "DType", "value": val.value}
        )
    if isinstance(val, (SymInt, SymNode)):
        return (
            {_TAG_KEY: "SymInt", "expr": str(val), "node": _encode_symnode(val.node)}
            if isinstance(val, SymInt)
            else _encode_symnode(val)
        )
    if isinstance(val, TensorShape):
        return {_TAG_KEY: "TensorShape", "dims": [_encode_dimension(d) for d in val.dims]}
    return {
        _TAG_KEY: "TensorSpec",
        "dtype": val.dtype.value,
        "shape": [_encode_dimension(d) for d in val.shape],
        "sparsity": val.sparsity,
    }


def _encode_attribute_value(val: AttributeValue) -> JsonValue:
    """Encode an ``AttributeValue`` into pure JSON types using lossless ``__sw_type__`` envelopes.

    Args:
        val (AttributeValue): IR attribute value.

    Returns:
        JsonValue: Pure JSON-serializable structure.
    """
    if isinstance(val, (bytes, DType, SymInt, SymNode, TensorShape, TensorSpec)):
        return _encode_rich_attribute_leaf(val)
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, Mapping):
        return {str(k): _encode_attribute_value(v) for k, v in val.items()}
    return [_encode_attribute_value(item) for item in val]


def _encode_initializer_value(val: InitializerValue) -> JsonValue:
    """Encode an ``InitializerValue`` into pure JSON types, preserving ``bytes`` via base64 tags.

    Args:
        val (InitializerValue): Initializer constant payload.

    Returns:
        JsonValue: Pure JSON-serializable value.
    """
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, bytes):
        return {_TAG_KEY: "bytes", "data": base64.b64encode(val).decode("ascii")}
    if isinstance(val, Mapping):
        return {str(k): _encode_initializer_value(v) for k, v in val.items()}
    return [_encode_initializer_value(item) for item in val]


def _decode_tensor_shape_or_spec(
    tag: RawGraphValue,
    val: Mapping[str, RawGraphValue],
) -> TensorShape | TensorSpec | None:
    """Decode ``TensorShape`` or ``TensorSpec`` tagged mappings.

    Args:
        tag (RawGraphValue): Value of ``__sw_type__``.
        val (Mapping[str, RawGraphValue]): Tagged mapping payload.

    Returns:
        TensorShape | TensorSpec | None: Reconstructed shape/spec or ``None``.
    """
    if tag == "TensorShape":
        raw_dims = val.get("dims", ())
        dims = (
            tuple(_parse_raw_dim_entry(d) for d in raw_dims)
            if isinstance(raw_dims, Sequence) and not isinstance(raw_dims, (str, bytes))
            else ()
        )
        return TensorShape(dims)
    if tag == "TensorSpec":
        raw_shape = val.get("shape", ())
        shape_seq = (
            [_parse_raw_dim_entry(d) for d in raw_shape]
            if isinstance(raw_shape, Sequence) and not isinstance(raw_shape, (str, bytes))
            else []
        )
        raw_sp = val.get("sparsity")
        return TensorSpec(
            shape=shape_seq,
            dtype=str(val.get("dtype", "float32")),
            sparsity=(str(raw_sp) if raw_sp is not None else None),
        )
    return None


def _decode_tagged_mapping(val: Mapping[str, RawGraphValue]) -> AttributeValue | None:
    """Decode a ``__sw_type__`` tagged mapping back into its rich IR runtime object.

    Args:
        val (Mapping[str, RawGraphValue]): Candidate mapping with ``__sw_type__`` tag.

    Returns:
        AttributeValue | None: Reconstructed rich object, or ``None`` if untagged.
    """
    tag = val.get(_TAG_KEY)
    if tag == "bytes":
        return base64.b64decode(str(val.get("data", "")).encode("ascii"))
    if tag == "DType":
        return DType.from_str(str(val["value"]))
    if tag == "SymNode":
        return _decode_symnode(val)
    if tag == "SymInt":
        return _decode_symint_mapping(val)
    return _decode_tensor_shape_or_spec(tag, val)


def _normalize_partition_axis_atom(
    raw: RawGraphValue | PartitionAxisElement,
) -> PartitionAxisAtom:
    """Normalize a single leaf partition axis element.

    Args:
        raw (RawGraphValue | PartitionAxisElement): Raw leaf element.

    Returns:
        PartitionAxisAtom: Normalized atom.
    """
    return raw if raw is None or (isinstance(raw, (str, int)) and not isinstance(raw, bool)) else str(raw)


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
    """Recursively narrow and decode a ``RawGraphValue`` into an ``AttributeValue``.

    Args:
        val (RawGraphValue): Raw graph value.

    Returns:
        AttributeValue: Decoded ``AttributeValue``.
    """
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
        decoded = _decode_tagged_mapping(val)
        if decoded is not None:
            return decoded
        return {str(k): _coerce_attribute_value(v) for k, v in val.items()}
    return (
        str(val)
        if isinstance(val, (LogicalMesh, PartitionSpec, PipelineTopologyConfig))
        else [_coerce_attribute_value(item) for item in val]
    )


def _coerce_initializer_value(val: RawGraphValue) -> InitializerValue:
    """Recursively narrow and decode a ``RawGraphValue`` into an ``InitializerValue``.

    Args:
        val (RawGraphValue): Raw graph value.

    Returns:
        InitializerValue: Decoded ``InitializerValue``.
    """
    if isinstance(val, (int, float, bool, str, bytes)):
        return val
    if val is None:
        return ""
    if isinstance(val, Mapping):
        if val.get(_TAG_KEY) == "bytes":
            return base64.b64decode(str(val.get("data", "")).encode("ascii"))
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


def _parse_sharding_field(sharding_raw: RawGraphValue) -> PartitionSpec | None:
    """Deserialize an optional ``PartitionSpec`` from a node dictionary.

    Args:
        sharding_raw (RawGraphValue): Raw sharding value.

    Returns:
        PartitionSpec | None: Parsed ``PartitionSpec`` or ``None``.
    """
    if isinstance(sharding_raw, PartitionSpec):
        return sharding_raw
    if isinstance(sharding_raw, Mapping):
        raw_axes = sharding_raw.get("axes", ())
        axes_seq = raw_axes if isinstance(raw_axes, Sequence) and not isinstance(raw_axes, (str, bytes)) else ()
        return PartitionSpec(axes=tuple(_normalize_partition_axis(a) for a in axes_seq))
    return None


def _parse_output_specs_field(output_specs_raw: RawGraphValue) -> list[TensorSpec]:
    """Deserialize ``output_specs`` from a node dictionary.

    Args:
        output_specs_raw (RawGraphValue): Raw ``output_specs`` sequence.

    Returns:
        list[TensorSpec]: Parsed ``TensorSpec`` list.
    """
    specs: list[TensorSpec] = []
    seq = (
        output_specs_raw
        if isinstance(output_specs_raw, Sequence) and not isinstance(output_specs_raw, (str, bytes))
        else ()
    )
    for s in seq:
        if isinstance(s, TensorSpec):
            specs.append(s)
        elif isinstance(s, Mapping):
            raw_s_shape = s.get("shape", ())
            s_shape: list[DimensionType] = (
                [_parse_raw_dim_entry(x) for x in raw_s_shape]
                if isinstance(raw_s_shape, Sequence) and not isinstance(raw_s_shape, (str, bytes))
                else []
            )
            s_dtype = str(s.get("dtype", "float32"))
            raw_sparsity = s.get("sparsity")
            s_sparsity = str(raw_sparsity) if raw_sparsity is not None else None
            specs.append(TensorSpec(shape=s_shape, dtype=s_dtype, sparsity=s_sparsity))
    return specs


def _parse_node_dict(
    ndata: Mapping[str, RawGraphValue],
    graph_cls: type[LogicalGraph],
) -> LogicalNode:
    """Deserialize a single LogicalNode (or ZeroTangent/NoTangent sentinel) from a raw dictionary.

    Args:
        ndata (Mapping[str, RawGraphValue]): Raw dictionary describing a node.
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
    version_val = int(raw_ver) if isinstance(raw_ver, (int, str)) and not isinstance(raw_ver, bool) else 1

    raw_attrs = ndata.get("attributes")
    attributes_val: dict[str, AttributeValue] = (
        {str(k): _coerce_attribute_value(v) for k, v in raw_attrs.items()} if isinstance(raw_attrs, Mapping) else {}
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
        if isinstance(raw_outputs, Sequence) and not isinstance(raw_outputs, (str, bytes))
        else None
    )
    raw_shape = ndata.get("shape_metadata")
    shape_val: tuple[DimensionType, ...] | None = (
        tuple(_parse_raw_dim_entry(d) for d in raw_shape)
        if isinstance(raw_shape, Sequence) and not isinstance(raw_shape, (str, bytes))
        else None
    )
    sharding_val = _parse_sharding_field(ndata.get("sharding"))
    dtype_raw = ndata.get("dtype")
    dtype_val: DType | None = (
        dtype_raw
        if isinstance(dtype_raw, DType)
        else (DType.from_str(str(dtype_raw)) if dtype_raw is not None else None)
    )
    output_specs_val = _parse_output_specs_field(ndata.get("output_specs"))
    subgraphs_val = _parse_subgraphs_from_node_dict(ndata, graph_cls)
    raw_ast_ref = ndata.get("source_ast_ref")
    ast_ref_val = str(raw_ast_ref) if raw_ast_ref is not None else None
    raw_device = ndata.get("device")
    device_val = str(raw_device) if raw_device is not None else None
    raw_stream = ndata.get("stream")
    stream_val = str(raw_stream) if raw_stream is not None else None

    if domain_val in ("ml.switcheroo.ad", "ad") and op_type_val == "ZeroTangent":
        return ZeroTangent(
            id=node_id,
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
    if domain_val in ("ml.switcheroo.ad", "ad") and op_type_val == "NoTangent":
        return NoTangent(
            id=node_id,
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
        dict[str, LogicalGraph | AttributeValue]:
            Mapping of subgraph names to LogicalGraph instances or AttributeValues.
    """
    resolved_subgraphs: dict[str, LogicalGraph | AttributeValue] = {}
    subgraphs_data = ndata.get("subgraphs")
    if isinstance(subgraphs_data, Mapping):
        for sub_k, sub_v in subgraphs_data.items():
            if isinstance(sub_v, graph_cls):
                resolved_subgraphs[str(sub_k)] = sub_v
            elif isinstance(sub_v, Mapping) and _TAG_KEY not in sub_v and ("nodes" in sub_v or "name" in sub_v):
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
    webrtc_topo = WebRTCSignalingTopology.model_validate(raw_webrtc) if isinstance(raw_webrtc, Mapping) else None
    raw_shape = mesh_data.get("shape", {})
    shape_dict: dict[str, int] = (
        {str(k): int(v) for k, v in raw_shape.items() if isinstance(v, (int, str)) and not isinstance(v, bool)}
        if isinstance(raw_shape, Mapping)
        else {}
    )
    return LogicalMesh(shape=shape_dict, webrtc_topology=webrtc_topo)


def _parse_input_specs_dict(
    input_specs_raw: RawGraphValue,
) -> dict[str, TensorSpec]:
    """Deserialize an optional input_specs dictionary into TensorSpec instances.

    Args:
        input_specs_raw (RawGraphValue): Raw input specs dictionary.

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
                    [_parse_raw_dim_entry(x) for x in raw_shape]
                    if isinstance(raw_shape, Sequence) and not isinstance(raw_shape, (str, bytes))
                    else []
                )
                dtype_str = str(v.get("dtype", "float32"))
                raw_sparsity = v.get("sparsity")
                sparsity_str = str(raw_sparsity) if raw_sparsity is not None else None
                input_specs[str(k)] = TensorSpec(shape=shape_seq, dtype=dtype_str, sparsity=sparsity_str)
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
_MAX_STREAM_TOKEN_CHARS: typing.Final[int] = 64 * 1024 * 1024


def _encode_special_types(val: AttributeValue | InitializerValue | JsonValue) -> JsonValue:
    """Fallback JSON encoder for IR-specific types passed directly to ``json.dumps``.

    Args:
        val (AttributeValue | InitializerValue | JsonValue): Value to serialize.

    Returns:
        JsonValue: JSON-compatible representation of special IR value.
    """
    if isinstance(val, DType):
        return val.value
    if isinstance(val, bytes):
        return {_TAG_KEY: "bytes", "data": base64.b64encode(val).decode("ascii")}
    if isinstance(val, SymInt):
        return {_TAG_KEY: "SymInt", "expr": str(val), "node": _encode_symnode(val.node)}
    if isinstance(val, SymNode):
        return _encode_symnode(val)
    return str(val)


def _decode_next_json_value(
    decoder: json.JSONDecoder,
    buf: str,
    pos: int,
    stream: typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO,
    chunk_size: int,
    max_token_chars: int = _MAX_STREAM_TOKEN_CHARS,
) -> tuple[JsonValue, str, int]:
    """Decode a single JSON value starting at pos, incrementally reading chunks on boundary straddles.

    Args:
        decoder (json.JSONDecoder): Reusable JSONDecoder instance.
        buf (str): Current buffer slice.
        pos (int): Current position within buf.
        stream (typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO): Source input stream.
        chunk_size (int): Chunk size for incremental reads.
        max_token_chars (int): Maximum buffer characters for a single JSON token before failing fast.

    Returns:
        tuple[JsonValue, str, int]: Decoded JSON value and updated (buf, next_pos).

    Raises:
        json.JSONDecodeError: If the stream ends before a complete JSON value can be decoded or exceeds max_token_chars.
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
            if len(buf) - pos >= max_token_chars:
                raise
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
        ValueError: If an individual node entry is not a JSON object mapping.
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
        if not isinstance(ndata, Mapping):
            raise ValueError(f"Expected JSON object mapping for node '{nid}', got {type(ndata)}.")
        nodes[str(nid)] = _parse_node_dict(ndata, graph_cls)


def _unpack_edge_item(
    edge: LogicalEdge | Mapping[str, RawGraphValue],
) -> tuple[str, str, int | None, int, str | None]:
    """Extract normalized ``(source, target, source_idx, target_idx, value_name)`` from an edge item.

    Args:
        edge (LogicalEdge | Mapping[str, RawGraphValue]): Edge instance or dictionary.

    Returns:
        tuple[str, str, int | None, int, str | None]: Normalized edge fields.
    """
    from ml_switcheroo_ir.graph import LogicalEdge as _LogicalEdge

    if isinstance(edge, _LogicalEdge):
        return (
            edge.source,
            edge.target,
            edge.source_idx,
            edge.target_idx,
            edge.value_name,
        )
    src = str(edge.get("source", ""))
    tgt = str(edge.get("target", ""))
    raw_src_idx = edge.get("source_idx", 0)
    src_idx = int(raw_src_idx) if isinstance(raw_src_idx, int) and not isinstance(raw_src_idx, bool) else None
    raw_tgt_idx = edge.get("target_idx", 0)
    tgt_idx = int(raw_tgt_idx) if isinstance(raw_tgt_idx, int) and not isinstance(raw_tgt_idx, bool) else 0
    raw_val_name = edge.get("value_name")
    val_name = str(raw_val_name) if raw_val_name is not None else None
    return src, tgt, src_idx, tgt_idx, val_name


def _resolve_edge_src_name(
    src: str,
    src_idx: int | None,
    val_name: str | None,
    nodes: Mapping[str, LogicalNode],
    clear_existing: bool,
) -> str:
    """Resolve the upstream SSA value name for an edge.

    Args:
        src (str): Source node ID or SSA value name.
        src_idx (int | None): Optional source port index.
        val_name (str | None): Explicit edge value name.
        nodes (Mapping[str, LogicalNode]): Graph nodes dictionary.
        clear_existing (bool): Whether existing node inputs were cleared.

    Returns:
        str: Resolved upstream SSA value name.
    """
    if src in nodes and src_idx is not None:
        src_node = nodes[src]
        if 0 <= src_idx < len(src_node.outputs) and (clear_existing or src_idx != 0 or src_node.outputs[0] != src):
            return src_node.outputs[src_idx]
        if clear_existing or src_idx != 0:
            return src_node.get_output_name(int(src_idx))
        return src
    if any(src in n.outputs for n in nodes.values()):
        return src
    if val_name:
        return str(val_name)
    return src


def _apply_edges_list(
    nodes: MutableMapping[str, LogicalNode],
    edges_list: Sequence[LogicalEdge | Mapping[str, RawGraphValue]],
    clear_existing: bool = False,
    edge_value_names: dict[tuple[str, int], str] | None = None,
) -> None:
    """Apply an explicit list of LogicalEdge instances or dicts onto target node inputs.

    Supports both multi-input ports (via explicit distinct ``target_idx`` values,
    e.g. ``Add(x, x)``) and idempotent deduplication when ``target_idx`` is repeated.

    Args:
        nodes (MutableMapping[str, LogicalNode]): Graph nodes dictionary.
        edges_list (Sequence[LogicalEdge | Mapping[str, RawGraphValue]]): Edges to apply.
        clear_existing (bool): Whether to clear existing target_node.inputs first.
        edge_value_names (dict[tuple[str, int], str] | None): Optional mapping of
            (target_id, target_idx) to explicit edge value_name metadata.
    """
    active_val_names = edge_value_names if edge_value_names is not None else {}
    if clear_existing:
        for node in nodes.values():
            node.inputs.clear()
        active_val_names.clear()

    seen_ports: set[tuple[str, int, str]] = set()

    for edge in edges_list:
        src, tgt, src_idx, tgt_idx, val_name = _unpack_edge_item(edge)
        if tgt not in nodes:
            continue

        target_node = nodes[tgt]
        src_name = _resolve_edge_src_name(src, src_idx, val_name, nodes, clear_existing)
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
        meta (Mapping[str, RawGraphValue]): Top-level graph metadata fields.

    Returns:
        LogicalGraph: Constructed LogicalGraph instance.
    """
    edge_value_names: dict[tuple[str, int], str] = {}
    edges_data = meta.get("edges")
    if isinstance(edges_data, list):
        typed_edges: list[LogicalEdge | Mapping[str, RawGraphValue]] = [e for e in edges_data if isinstance(e, Mapping)]
        _apply_edges_list(
            nodes,
            typed_edges,
            clear_existing=False,
            edge_value_names=edge_value_names,
        )

    pipe_topo_raw = meta.get("pipeline_topology")
    pipe_topo = PipelineTopologyConfig.model_validate(pipe_topo_raw) if isinstance(pipe_topo_raw, Mapping) else None
    inputs_data = meta.get("inputs")
    inputs_list: list[str] = (
        [str(x) for x in inputs_data]
        if isinstance(inputs_data, Sequence) and not isinstance(inputs_data, (str, bytes))
        else []
    )
    outputs_data = meta.get("outputs")
    outputs_list: list[str] | None = (
        [str(x) for x in outputs_data]
        if isinstance(outputs_data, Sequence) and not isinstance(outputs_data, (str, bytes))
        else None
    )
    inits_raw = meta.get("initializers")
    inits_dict: dict[str, InitializerValue] = (
        {str(k): _coerce_initializer_value(v) for k, v in inits_raw.items()} if isinstance(inits_raw, Mapping) else {}
    )
    attrs_raw = meta.get("attributes")
    attrs_dict: dict[str, AttributeValue] = (
        {str(k): _coerce_attribute_value(v) for k, v in attrs_raw.items()} if isinstance(attrs_raw, Mapping) else {}
    )
    loop_raw = meta.get("loop_options")
    loop_dict: dict[str, AttributeValue] | None = (
        {str(k): _coerce_attribute_value(v) for k, v in loop_raw.items()} if isinstance(loop_raw, Mapping) else None
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
        attributes=attrs_dict,
        loop_options=loop_dict,
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
        stream (typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO):
            Readable text or binary stream containing JSON.
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
            nodes, buf, pos = _stream_parse_nodes_section(decoder, buf, pos, stream, chunk_size, graph_cls)
        else:
            val, buf, pos = _decode_next_json_value(decoder, buf, pos, stream, chunk_size)
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


def _serialize_edge_dict(edge: LogicalEdge) -> dict[str, JsonValue]:
    """Serialize a ``LogicalEdge`` into its canonical JSON dictionary form.

    Args:
        edge (LogicalEdge): Edge instance to serialize.

    Returns:
        dict[str, JsonValue]: Canonical serialized edge dictionary.
    """
    return {
        "source": edge.source,
        "source_idx": edge.source_idx,
        "target": edge.target,
        "target_idx": edge.target_idx,
        "value_name": edge.value_name,
    }


def _serialize_tensor_spec_dict(spec: TensorSpec) -> dict[str, JsonValue]:
    """Serialize a ``TensorSpec`` into its canonical JSON dictionary form.

    Args:
        spec (TensorSpec): Tensor specification instance.

    Returns:
        dict[str, JsonValue]: Canonical serialized TensorSpec dictionary.
    """
    return {
        "dtype": spec.dtype.value,
        "shape": [_encode_dimension(d) for d in spec.shape],
        "sparsity": spec.sparsity,
    }


def _serialize_node_dict(node: LogicalNode) -> dict[str, JsonValue]:
    """Serialize a LogicalNode to a JSON-compatible dictionary preserving rich IR types and subgraphs.

    Args:
        node (LogicalNode): Node to convert to a dictionary.

    Returns:
        dict[str, JsonValue]: Canonical dictionary representation of the node.
    """
    from ml_switcheroo_ir.graph import LogicalGraph as _LogicalGraph

    sub_dict: dict[str, JsonValue] = {}
    for k, v in node.subgraphs.items():
        if isinstance(v, _LogicalGraph):
            sub_dict[k] = graph_to_dict(v)
        else:
            sub_dict[k] = _encode_attribute_value(copy.deepcopy(v))

    sharding_dict: dict[str, JsonValue] | None = None
    if node.sharding is not None:
        sharding_dict = {"axes": json.loads(json.dumps(node.sharding.axes))}

    shape_meta_val: list[JsonValue] | None = (
        [_encode_dimension(d) for d in node.shape_metadata] if node.shape_metadata is not None else None
    )

    return {
        "attributes": {k: _encode_attribute_value(v) for k, v in node.attributes.items()},
        "device": node.device,
        "domain": node.domain,
        "dtype": node.dtype.value if node.dtype is not None else None,
        "id": node.id,
        "inputs": list(node.inputs),
        "op_type": node.op_type,
        "output_specs": [_serialize_tensor_spec_dict(spec) for spec in node.output_specs],
        "outputs": list(node.outputs),
        "shape_metadata": shape_meta_val,
        "sharding": sharding_dict,
        "source_ast_ref": node.source_ast_ref,
        "stream": node.stream,
        "subgraphs": sub_dict,
        "version": node.version,
    }


def _serialize_graph_header_fields(graph: LogicalGraph) -> dict[str, JsonValue]:
    """Serialize lightweight top-level scalar/metadata fields shared by ``graph_to_dict`` and ``_stream_write_graph``.

    Args:
        graph (LogicalGraph): Graph instance.

    Returns:
        dict[str, JsonValue]: Mapping of top-level scalar/metadata fields (excluding nodes, edges, initializers).
    """
    mesh_val: dict[str, JsonValue] | None = (
        {
            "shape": {k: int(v) for k, v in graph.mesh.shape.items()},
            "webrtc_topology": (
                dict(graph.mesh.webrtc_topology.to_dict()) if graph.mesh.webrtc_topology is not None else None
            ),
        }
        if graph.mesh is not None
        else None
    )
    pipe_val: dict[str, JsonValue] | None = (
        dict(graph.pipeline_topology.to_dict()) if graph.pipeline_topology is not None else None
    )
    attrs_val: dict[str, JsonValue] = {k: _encode_attribute_value(v) for k, v in graph.attributes.items()}
    loop_val: dict[str, JsonValue] | None = (
        {k: _encode_attribute_value(v) for k, v in graph.loop_options.items()}
        if graph.loop_options is not None
        else None
    )
    input_specs_val: dict[str, JsonValue] = {
        k: _serialize_tensor_spec_dict(v) for k, v in sorted(graph.input_specs.items())
    }
    return {
        "attributes": attrs_val,
        "input_specs": input_specs_val,
        "inputs": list(graph.inputs),
        "loop_options": loop_val,
        "mesh": mesh_val,
        "name": graph.name,
        "outputs": list(graph.outputs),
        "pipeline_topology": pipe_val,
    }


def graph_to_dict(graph: LogicalGraph) -> dict[str, JsonValue]:
    """Convert a ``LogicalGraph`` into a canonical, lossless JSON-compatible dictionary.

    Args:
        graph (LogicalGraph): Graph to serialize.

    Returns:
        dict[str, JsonValue]: Canonical dictionary representation.
    """
    header = _serialize_graph_header_fields(graph)
    return {
        "attributes": header["attributes"],
        "edges": [_serialize_edge_dict(edge) for edge in graph.iter_edges()],
        "initializers": {k: _encode_initializer_value(v) for k, v in graph.initializers.items()},
        "input_specs": header["input_specs"],
        "inputs": header["inputs"],
        "loop_options": header["loop_options"],
        "mesh": header["mesh"],
        "name": header["name"],
        "nodes": {nid: _serialize_node_dict(node) for nid, node in graph.nodes.items()},
        "outputs": header["outputs"],
        "pipeline_topology": header["pipeline_topology"],
    }


def _stream_write_edges_section(
    graph: LogicalGraph,
    fp: typing.IO[str] | typing.TextIO,
    indent: int,
    pad1: str,
    pad2: str,
    comma: str,
) -> None:
    """Stream ``graph.iter_edges()`` lazily without materializing a full edge list in RAM.

    Args:
        graph (LogicalGraph): Graph whose edges are streamed.
        fp (typing.IO[str] | typing.TextIO): Writable text stream.
        indent (int): JSON indentation width.
        pad1 (str): First-level indentation spaces.
        pad2 (str): Second-level indentation spaces.
        comma (str): Trailing comma after closing bracket.
    """
    edge_iter = graph.iter_edges()
    first_edge = next(edge_iter, None)
    if first_edge is None:
        fp.write(f"[]{comma}{_NL}")
        return
    fp.write("[" + _NL)
    prev_edge = first_edge
    for next_edge in edge_iter:
        edge_str = json.dumps(
            _serialize_edge_dict(prev_edge), sort_keys=True, indent=indent, default=_encode_special_types
        )
        fp.write(f"{pad2}{_indent_json_fragment(edge_str, pad2)},{_NL}")
        prev_edge = next_edge
    last_str = json.dumps(_serialize_edge_dict(prev_edge), sort_keys=True, indent=indent, default=_encode_special_types)
    fp.write(f"{pad2}{_indent_json_fragment(last_str, pad2)}{_NL}")
    fp.write(f"{pad1}]{comma}{_NL}")


def _stream_write_initializers_section(
    initializers: Mapping[str, InitializerValue],
    fp: typing.IO[str] | typing.TextIO,
    indent: int,
    pad1: str,
    pad2: str,
    comma: str,
) -> None:
    """Stream ``initializers`` entry-by-entry without serializing all weight buffers at once.

    Args:
        initializers (Mapping[str, InitializerValue]): Graph initializers map.
        fp (typing.IO[str] | typing.TextIO): Writable text stream.
        indent (int): JSON indentation width.
        pad1 (str): First-level indentation spaces.
        pad2 (str): Second-level indentation spaces.
        comma (str): Trailing comma after closing brace.
    """
    if not initializers:
        fp.write(f"{{}}{comma}{_NL}")
        return
    sorted_keys = sorted(initializers.keys())
    fp.write("{" + _NL)
    for idx, init_key in enumerate(sorted_keys):
        i_comma = "," if idx < len(sorted_keys) - 1 else ""
        encoded_val = _encode_initializer_value(initializers[init_key])
        val_str = json.dumps(encoded_val, sort_keys=True, indent=indent, default=_encode_special_types)
        fp.write(f"{pad2}{json.dumps(init_key)}: {_indent_json_fragment(val_str, pad2)}{i_comma}{_NL}")
    fp.write(f"{pad1}}}{comma}{_NL}")


def _stream_write_nodes_section(
    graph: LogicalGraph,
    fp: typing.IO[str] | typing.TextIO,
    indent: int,
    pad1: str,
    pad2: str,
    comma: str,
) -> None:
    """Stream ``graph.nodes`` entry-by-entry in sorted key order.

    Args:
        graph (LogicalGraph): Graph whose nodes are streamed.
        fp (typing.IO[str] | typing.TextIO): Writable text stream.
        indent (int): JSON indentation width.
        pad1 (str): First-level indentation spaces.
        pad2 (str): Second-level indentation spaces.
        comma (str): Trailing comma after closing brace.
    """
    if not graph.nodes:
        fp.write(f"{{}}{comma}{_NL}")
        return
    sorted_nids = sorted(graph.nodes.keys())
    fp.write("{" + _NL)
    for n_idx, nid in enumerate(sorted_nids):
        n_comma = "," if n_idx < len(sorted_nids) - 1 else ""
        node_dict = _serialize_node_dict(graph.nodes[nid])
        node_str = json.dumps(node_dict, sort_keys=True, indent=indent, default=_encode_special_types)
        fp.write(f"{pad2}{json.dumps(nid)}: {_indent_json_fragment(node_str, pad2)}{n_comma}{_NL}")
    fp.write(f"{pad1}}}{comma}{_NL}")


def _stream_write_graph(
    graph: LogicalGraph,
    fp: typing.IO[str] | typing.TextIO,
    indent: int = 2,
) -> None:
    """Incrementally stream-serialize a LogicalGraph to a writable stream in O(max_node_size) RAM.

    Produces output byte-for-byte identical to ``json.dumps(graph.to_dict(), sort_keys=True, indent=indent)``
    while streaming ``edges``, ``initializers``, and ``nodes`` item by item without materializing
    a full-graph dictionary or monolithic JSON string in memory.

    Args:
        graph (LogicalGraph): Graph to serialize.
        fp (typing.IO[str] | typing.TextIO): Writable text stream.
        indent (int): Indentation spaces (default: 2).
    """
    pad1 = " " * indent
    pad2 = " " * (indent * 2)
    header = _serialize_graph_header_fields(graph)

    keys_order = (
        "attributes",
        "edges",
        "initializers",
        "input_specs",
        "inputs",
        "loop_options",
        "mesh",
        "name",
        "nodes",
        "outputs",
        "pipeline_topology",
    )

    fp.write("{" + _NL)
    total_keys = len(keys_order)
    for k_idx, key in enumerate(keys_order):
        comma = "," if k_idx < total_keys - 1 else ""
        fp.write(f'{pad1}"{key}": ')
        if key == "edges":
            _stream_write_edges_section(graph, fp, indent, pad1, pad2, comma)
        elif key == "initializers":
            _stream_write_initializers_section(graph.initializers, fp, indent, pad1, pad2, comma)
        elif key == "nodes":
            _stream_write_nodes_section(graph, fp, indent, pad1, pad2, comma)
        else:
            val_str = json.dumps(header[key], sort_keys=True, indent=indent, default=_encode_special_types)
            fp.write(f"{_indent_json_fragment(val_str, pad1)}{comma}{_NL}")
    fp.write("}")
