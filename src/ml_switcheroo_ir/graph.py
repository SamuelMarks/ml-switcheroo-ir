"""Core graph intermediate representation Pydantic models and topological utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
import copy
from enum import Enum
import gzip
import io
import json
import os
from pathlib import Path
import typing

import pydantic
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import pydantic_core

from ml_switcheroo_ir.distributed import (
    LogicalAxis,
    LogicalMesh,
    PartitionSpec,
    PipelineTopologyConfig,
)
from ml_switcheroo_ir.serialization import (
    RawGraphValue,
    SerializedEdgeModel,
    SerializedGraphModel,
    SerializedNodeModel,
    _apply_edges_list,
    _build_graph_from_parsed_sections,
    _decode_next_json_value,
    _indent_json_fragment,
    _load_zstandard,
    _parse_input_specs_dict,
    _parse_mesh_dict,
    _parse_node_dict,
    _parse_subgraphs_from_node_dict,
    _read_next_chunk,
    _serialize_node_dict,
    _skip_ws_and_tokens,
    _stream_parse_graph,
    _stream_parse_nodes_section,
    _stream_write_graph,
    zstandard,
)
from ml_switcheroo_ir.shapes import DimensionType
from ml_switcheroo_ir.types import (
    AttributeValue,
    DType,
    InitializerValue,
    JsonValue,
    ShapeMetadataType,
    TensorShape,
    TensorSpec,
    _AttributeValuePydanticAnnotation,
    _validate_dimension_tuple,
    is_dynamic_dimension,
)

__all__ = [
    "BaseFrontend",
    "CompiledArtifactT",
    "CompilerBackend",
    "CyclicGraphError",
    "EdgeList",
    "GraphFrontend",
    "LogicalAxis",
    "LogicalEdge",
    "LogicalGraph",
    "LogicalMesh",
    "LogicalNode",
    "NoTangent",
    "NodeDict",
    "PartitionSpec",
    "SerializedEdgeModel",
    "SerializedGraphModel",
    "SerializedNodeModel",
    "ZeroTangent",
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
    "topological_sort",
    "zstandard",
]


class _OmittedSentinel(Enum):
    """Sentinel enum indicating an omitted optional parameter."""

    OMITTED = "OMITTED"


_CLONE_OMITTED: typing.Final[_OmittedSentinel] = _OmittedSentinel.OMITTED


class LogicalEdge(BaseModel):
    """Represents a directed data-flow edge between two nodes in the graph.

    Attributes:
        source (str): Source node identifier emitting data.
        target (str): Target node identifier consuming data.
        source_idx (int | None): Output port index of the source node (default: 0).
        target_idx (int): Input argument index of the target node (default: 0).
        value_name (str | None): Name of the intermediate SSA value (default: None).
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    source_idx: int | None = 0
    target_idx: int = 0
    value_name: str | None = None

    def __init__(
        self,
        source: str,
        target: str,
        source_idx: int | None = 0,
        target_idx: int = 0,
        value_name: str | None = None,
    ) -> None:
        """Initialize LogicalEdge.

        Args:
            source (str): Source node identifier emitting data.
            target (str): Target node identifier consuming data.
            source_idx (int | None): Output port index of the source node (default: 0).
            target_idx (int): Input argument index of the target node (default: 0).
            value_name (str | None): Name of the intermediate SSA value (default: None).
        """
        super().__init__(
            source=source,
            target=target,
            source_idx=source_idx,
            target_idx=target_idx,
            value_name=value_name,
        )


class CyclicGraphError(Exception):
    """Raised when a cycle is detected in a graph during topological sorting."""


class _TrackedOutputsList(list[str]):
    """List subclass for LogicalNode.outputs that increments a mutation counter on change."""

    version: int = 0

    def __init__(self, iterable: Iterable[str] = ()) -> None:
        """Initialize _TrackedOutputsList.

        Args:
            iterable (Iterable[str]): Initial output names.
        """
        self.version = 0
        super().__init__(iterable)

    def append(self, item: str) -> None:
        """Append an output name and increment mutation version.

        Args:
            item (str): Output SSA name.
        """
        super().append(item)
        self.version += 1


def _normalize_shape_metadata(
    shape_metadata: ShapeMetadataType | Iterable[DimensionType],
) -> tuple[DimensionType, ...] | None:
    """Normalize shape_metadata input into a canonical tuple of DimensionType or None.

    Args:
        shape_metadata (ShapeMetadataType | Iterable[DimensionType]): Raw shape metadata argument.

    Returns:
        tuple[DimensionType, ...] | None: Normalized tuple of dimensions or None.

    Raises:
        TypeError: If any dimension in a shape sequence is a boolean or invalid DimensionType.
    """
    if shape_metadata is None:
        return None
    if isinstance(shape_metadata, (str, bytes, Mapping)):
        raise TypeError(
            f"Invalid shape_metadata {shape_metadata!r}; expected TensorShape, sequence of DimensionType, or None."
        )
    return _validate_dimension_tuple(shape_metadata)


class LogicalNode(BaseModel):
    """Represents a computation unit (Layer) in the graph.

    Attributes:
        id (str): Unique identifier (e.g. 'conv1').
        op_type (str): Operation type. Standard types include 'Conv', 'Gemm', as well as advanced primitives.
        domain (str): Operator domain (e.g., 'ai.onnx').
        version (int): Operator set version (e.g., 1).
        attributes (dict[str, AttributeValue]): Dictionary of configuration parameters (e.g. ``kernel_size=3``).
        inputs (list[str]): Ordered list of upstream LogicalNode IDs.
        outputs (list[str]): Ordered list of output names/SSA value identifiers produced by this node.
        shape_metadata (tuple[DimensionType, ...] | None): Tuple of dimensions or None.
        source_ast_ref (str | None): Traceback to exact file path, line number, and cdd-python AST node ID.
        sharding (PartitionSpec | None): Optional layout specification for distributed placement of this node's output.
        dtype (DType | None): Data type of the node computation or primary output.
        output_specs (list[TensorSpec]): Explicit tensor specifications for all node outputs.
        subgraphs (dict[str, LogicalGraph | AttributeValue]): Nested subgraphs for control flow and autodiff.
        device (str | None): Target device placement string (e.g., 'cuda:0', 'cpu').
        stream (str | None): Asynchronous execution stream name.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    id: str
    op_type: str = ""
    domain: str = "ai.onnx"
    version: int = 1
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    shape_metadata: tuple[DimensionType, ...] | None = None
    source_ast_ref: str | None = None
    sharding: PartitionSpec | None = None
    dtype: DType | None = None
    output_specs: list[TensorSpec] = Field(default_factory=list)
    subgraphs: dict[
        str,
        typing.Annotated[
            LogicalGraph | AttributeValue, _AttributeValuePydanticAnnotation
        ],
    ] = Field(default_factory=dict)
    device: str | None = None
    stream: str | None = None

    @pydantic.field_validator("shape_metadata", mode="before")
    @classmethod
    def _validate_shape_metadata_field(
        cls, val: ShapeMetadataType | Iterable[DimensionType]
    ) -> tuple[DimensionType, ...] | None:
        """Validate and normalize shape_metadata on assignment and construction.

        Args:
            val (ShapeMetadataType | Iterable[DimensionType]): Raw shape metadata value.

        Returns:
            tuple[DimensionType, ...] | None: Validated tuple of dimensions or None.
        """
        return _normalize_shape_metadata(val)

    @pydantic.field_validator("outputs", mode="after")
    @classmethod
    def _ensure_tracked_outputs(cls, val: list[str]) -> list[str]:
        """Wrap outputs in _TrackedOutputsList for mutation version tracking.

        Args:
            val (list[str]): Validated outputs list.

        Returns:
            list[str]: _TrackedOutputsList instance.
        """
        return _TrackedOutputsList(val)

    def __init__(
        self,
        id: str,
        op_type: str,
        domain: str = "ai.onnx",
        version: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: ShapeMetadataType | Iterable[DimensionType] = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        dtype: DType | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None = None,
        device: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Initialize a LogicalNode instance.

        Args:
            id (str): Unique identifier for the node.
            op_type (str): Operation type (e.g. 'Conv', 'Relu').
            domain (str): Operator domain (default: 'ai.onnx').
            version (int): Operator set version (default: 1).
            attributes (Mapping[str, AttributeValue] | None): Operator configuration attributes.
            inputs (list[str] | None): Ordered list of upstream LogicalNode IDs.
            outputs (list[str] | None): Ordered list of output SSA names.
            shape_metadata (ShapeMetadataType | Iterable[DimensionType]): Tensor shape metadata.
            source_ast_ref (str | None): Source AST trace reference.
            sharding (PartitionSpec | None): Distributed partition layout.
            dtype (DType | None): Data type of the node.
            output_specs (list[TensorSpec] | None): Explicit output tensor specs.
            subgraphs (Mapping[str, LogicalGraph | AttributeValue] | None): Nested subgraphs dictionary.
            device (str | None): Target device placement.
            stream (str | None): Target stream identifier.

        Raises:
            ValueError: If id or op_type is empty.
        """
        if not id:
            raise ValueError("LogicalNode 'id' must be a non-empty string.")
        if not op_type:
            raise ValueError("LogicalNode 'op_type' must be a non-empty string.")
        super().__init__(
            id=id,
            op_type=op_type,
            domain=domain,
            version=version,
            attributes=dict(attributes) if attributes is not None else {},
            inputs=list(inputs) if inputs is not None else [],
            outputs=_TrackedOutputsList(outputs if outputs is not None else [id]),
            shape_metadata=_normalize_shape_metadata(shape_metadata),
            source_ast_ref=source_ast_ref,
            sharding=sharding,
            dtype=dtype,
            output_specs=list(output_specs) if output_specs is not None else [],
            subgraphs=dict(subgraphs) if subgraphs is not None else {},
            device=device,
            stream=stream,
        )

    @property
    def shape(self) -> TensorShape | None:
        """Return structured TensorShape if shape_metadata is available.

        Returns:
            TensorShape | None: Standardized shape instance if available.
        """
        if self.shape_metadata is None:
            return None
        return TensorShape(self.shape_metadata)

    @shape.setter
    def shape(self, val: ShapeMetadataType | Iterable[DimensionType]) -> None:
        """Set shape metadata via shape property.

        Args:
            val (ShapeMetadataType | Iterable[DimensionType]): Shape dimensions or TensorShape instance.
        """
        self.shape_metadata = _normalize_shape_metadata(val)

    @property
    def static_shape(self) -> tuple[int, ...]:
        """Return static shape or raise ValueError if dynamic or missing.

        Returns:
            tuple[int, ...]: Static shape dimensions.

        Raises:
            ValueError: If node has no shape metadata or shape is dynamic.
        """
        if self.shape_metadata is None:
            raise ValueError(f"Node {self.id} has no shape metadata.")
        return TensorShape(self.shape_metadata).static_shape

    @property
    def is_dynamic_shape(self) -> bool:
        """Check if shape metadata contains dynamic or symbolic dimensions.

        Returns:
            bool: True if dynamic dimensions are present.
        """
        if self.shape_metadata is None:
            return False
        return any(is_dynamic_dimension(dim) for dim in self.shape_metadata)

    @property
    def rank(self) -> int:
        """Return rank of shape metadata, or 0 if missing.

        Returns:
            int: Tensor rank.
        """
        if self.shape_metadata is None:
            return 0
        return len(self.shape_metadata)

    def clone(
        self,
        id: str | None = None,
        op_type: str | None = None,
        domain: str | None = None,
        version: int | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: ShapeMetadataType
        | Iterable[DimensionType]
        | _OmittedSentinel = _CLONE_OMITTED,
        source_ast_ref: str | None | _OmittedSentinel = _CLONE_OMITTED,
        sharding: PartitionSpec | None | _OmittedSentinel = _CLONE_OMITTED,
        dtype: DType | None | _OmittedSentinel = _CLONE_OMITTED,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None = None,
        device: str | None | _OmittedSentinel = _CLONE_OMITTED,
        stream: str | None | _OmittedSentinel = _CLONE_OMITTED,
        **kwargs: AttributeValue,
    ) -> LogicalNode:
        """Create a deep clone of this LogicalNode with optional keyword overrides.

        Args:
            id (str | None): Override node identifier.
            op_type (str | None): Override operation type.
            domain (str | None): Override operator domain.
            version (int | None): Override operator version.
            attributes (Mapping[str, AttributeValue] | None): Override attributes dictionary.
            inputs (list[str] | None): Override input IDs.
            outputs (list[str] | None): Override output names.
            shape_metadata (ShapeMetadataType | Iterable[DimensionType] | _OmittedSentinel): Override tensor shape metadata.
            source_ast_ref (str | None | _OmittedSentinel): Override source AST reference.
            sharding (PartitionSpec | None | _OmittedSentinel): Override partition specification.
            dtype (DType | None | _OmittedSentinel): Override data type.
            output_specs (list[TensorSpec] | None): Override output tensor specifications.
            subgraphs (Mapping[str, LogicalGraph | AttributeValue] | None): Override nested subgraphs.
            device (str | None | _OmittedSentinel): Override target device placement.
            stream (str | None | _OmittedSentinel): Override asynchronous execution stream.
            **kwargs (AttributeValue): Additional attribute overrides.

        Returns:
            LogicalNode: A cloned LogicalNode with deep-copied collections and applied overrides.
        """
        cloned_id = id if id is not None else self.id
        cloned_op = op_type if op_type is not None else self.op_type
        cloned_domain = domain if domain is not None else self.domain
        cloned_version = version if version is not None else self.version
        cloned_attrs = copy.deepcopy(
            dict(attributes) if attributes is not None else self.attributes
        )
        if kwargs:
            cloned_attrs.update(kwargs)

        raw_subgraphs = subgraphs if subgraphs is not None else self.subgraphs
        cloned_subgraphs: dict[str, LogicalGraph | AttributeValue] = {
            k: v.clone() if isinstance(v, LogicalGraph) else copy.deepcopy(v)
            for k, v in raw_subgraphs.items()
        }

        if isinstance(shape_metadata, _OmittedSentinel):
            resolved_shape_meta = copy.deepcopy(self.shape_metadata)
        else:
            resolved_shape_meta = _normalize_shape_metadata(shape_metadata)

        resolved_ast_ref = (
            self.source_ast_ref
            if isinstance(source_ast_ref, _OmittedSentinel)
            else source_ast_ref
        )
        resolved_sharding = (
            copy.deepcopy(self.sharding)
            if isinstance(sharding, _OmittedSentinel)
            else copy.deepcopy(sharding)
        )
        resolved_dtype = self.dtype if isinstance(dtype, _OmittedSentinel) else dtype
        resolved_device = (
            self.device if isinstance(device, _OmittedSentinel) else device
        )
        resolved_stream = (
            self.stream if isinstance(stream, _OmittedSentinel) else stream
        )

        cls = self.__class__
        return cls(
            id=cloned_id,
            op_type=cloned_op,
            domain=cloned_domain,
            version=cloned_version,
            attributes=cloned_attrs,
            inputs=list(inputs) if inputs is not None else list(self.inputs),
            outputs=list(outputs) if outputs is not None else list(self.outputs),
            shape_metadata=resolved_shape_meta,
            source_ast_ref=resolved_ast_ref,
            sharding=resolved_sharding,
            dtype=resolved_dtype,
            output_specs=[
                copy.deepcopy(s)
                for s in (
                    output_specs if output_specs is not None else self.output_specs
                )
            ],
            subgraphs=cloned_subgraphs,
            device=resolved_device,
            stream=resolved_stream,
        )

    def get_output_name(self, index: int) -> str:
        """Return explicit SSA output name or generate '{node.id}:{index}' default.

        Args:
            index (int): Zero-based output port index.

        Returns:
            str: Output name for the specified index.
        """
        if 0 <= index < len(self.outputs):
            return self.outputs[index]
        return f"{self.id}:{index}"

    @property
    def has_multiple_outputs(self) -> bool:
        """Return True if this node produces multiple SSA outputs or output specs.

        Returns:
            bool: True if outputs count or output_specs count exceeds 1.
        """
        return len(self.outputs) > 1 or len(self.output_specs) > 1


_DefaultT = typing.TypeVar("_DefaultT")


class _SupportsNodeKeysAndGetItem(typing.Protocol):
    """Protocol for mapping-like objects accepted by NodeDict.update."""

    def keys(self) -> Iterable[str]:
        """Return iterable of node IDs.

        Returns:
            Iterable[str]: Node IDs.
        """

    def __getitem__(self, key: str, /) -> LogicalNode:
        """Retrieve LogicalNode by ID.

        Args:
            key (str): Node ID.

        Returns:
            LogicalNode: Node instance.
        """


class NodeDict(dict[str, LogicalNode]):
    """Strictly typed mapping from node identifiers to LogicalNode instances with mutation tracking."""

    version: int = 0

    def __init__(
        self,
        mapping_or_iterable: Mapping[str, LogicalNode]
        | Iterable[tuple[str, LogicalNode]]
        | None = None,
        **kwargs: LogicalNode,
    ) -> None:
        """Initialize NodeDict and set initial version counter.

        Args:
            mapping_or_iterable (Mapping[str, LogicalNode] | Iterable[tuple[str, LogicalNode]] | None):
                Initial mapping or key-value pairs.
            **kwargs (LogicalNode): Keyword arguments forwarded to dict.
        """
        self.version = 0
        if mapping_or_iterable is not None:
            super().__init__(mapping_or_iterable, **kwargs)
        else:
            super().__init__(**kwargs)

    def __setitem__(self, key: str, value: LogicalNode) -> None:
        """Insert or update a node and increment version.

        Args:
            key (str): Node ID.
            value (LogicalNode): LogicalNode instance.
        """
        super().__setitem__(key, value)
        self.version += 1

    def __delitem__(self, key: str) -> None:
        """Delete a node by ID and increment version.

        Args:
            key (str): Node ID.
        """
        super().__delitem__(key)
        self.version += 1

    @typing.overload
    def pop(self, key: str) -> LogicalNode:
        """Remove and return a node by ID and increment version.

        Args:
            key (str): Node ID.

        Returns:
            LogicalNode: Removed LogicalNode.
        """

    @typing.overload
    def pop(
        self, key: str, default: LogicalNode | _DefaultT
    ) -> LogicalNode | _DefaultT:
        """Remove and return a node by ID or default and increment version.

        Args:
            key (str): Node ID.
            default (LogicalNode | _DefaultT): Default value if key is absent.

        Returns:
            LogicalNode | _DefaultT: Removed LogicalNode or default.
        """

    def pop(
        self,
        key: str,
        default: LogicalNode | _DefaultT | _OmittedSentinel = _CLONE_OMITTED,
    ) -> LogicalNode | _DefaultT:
        """Remove and return a node by ID and increment version.

        Args:
            key (str): Node ID.
            default (LogicalNode | _DefaultT | _OmittedSentinel): Optional default value.

        Returns:
            LogicalNode | _DefaultT: Removed LogicalNode or default.
        """
        if isinstance(default, _OmittedSentinel):
            res: LogicalNode | _DefaultT = super().pop(key)
        else:
            res = super().pop(key, default)
        self.version += 1
        return res

    def clear(self) -> None:
        """Clear all nodes and increment version."""
        super().clear()
        self.version += 1

    @typing.overload
    def update(
        self,
        other: _SupportsNodeKeysAndGetItem,
        /,
        **kwargs: LogicalNode,
    ) -> None:
        """Update mapping from a mapping-like object of node IDs to LogicalNode instances.

        Args:
            other (_SupportsNodeKeysAndGetItem): Mapping of node IDs to nodes.
            **kwargs (LogicalNode): Additional keyword node entries.
        """

    @typing.overload
    def update(
        self,
        other: Iterable[tuple[str, LogicalNode]] = (),
        /,
        **kwargs: LogicalNode,
    ) -> None:
        """Update mapping from an iterable of (node_id, LogicalNode) pairs or keyword entries.

        Args:
            other (Iterable[tuple[str, LogicalNode]]): Iterable of key-value pairs.
            **kwargs (LogicalNode): Additional keyword node entries.
        """

    def update(
        self,
        other: _SupportsNodeKeysAndGetItem | Iterable[tuple[str, LogicalNode]] = (),
        /,
        **kwargs: LogicalNode,
    ) -> None:
        """Update mapping with new nodes and increment version.

        Args:
            other (_SupportsNodeKeysAndGetItem | Iterable[tuple[str, LogicalNode]]):
                Mapping or iterable of key-value pairs.
            **kwargs (LogicalNode): Keyword node entries.
        """
        super().update(other, **kwargs)
        self.version += 1

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: object,
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.CoreSchema:
        """Generate Pydantic core schema for NodeDict as dict[str, LogicalNode].

        Args:
            source_type (object): Source type annotation.
            handler (pydantic.GetCoreSchemaHandler): Core schema handler.

        Returns:
            pydantic_core.CoreSchema: Core schema wrapping dict[str, LogicalNode].
        """
        from pydantic_core import core_schema

        dict_schema = handler.generate_schema(dict[str, LogicalNode])
        return core_schema.no_info_after_validator_function(
            cls,
            dict_schema,
        )


class EdgeList(list["LogicalEdge"]):
    """Typed list of LogicalEdge instances."""


def _normalize_graph_nodes(
    nodes: Mapping[str, LogicalNode] | NodeDict | None,
) -> NodeDict:
    """Normalize graph nodes argument from None or mapping into a NodeDict.

    Args:
        nodes (Mapping[str, LogicalNode] | NodeDict | None): Input nodes mapping.

    Returns:
        NodeDict: Mapping from node IDs to LogicalNode instances.

    Raises:
        TypeError: If nodes is not a Mapping or None.
    """
    if nodes is None:
        return NodeDict()
    if isinstance(nodes, NodeDict):
        return nodes
    if isinstance(nodes, Mapping):
        return NodeDict(nodes)
    raise TypeError(
        f"LogicalGraph 'nodes' must be a Mapping[str, LogicalNode] or None, got {type(nodes).__name__}."
    )


def _auto_detect_input_pseudo_nodes(
    nodes: dict[str, LogicalNode],
    inputs: list[str],
    input_specs: dict[str, TensorSpec],
) -> None:
    """Populate graph inputs and input_specs from Input pseudo-nodes.

    Args:
        nodes (dict[str, LogicalNode]): Graph node dictionary.
        inputs (list[str]): Graph input list to update in place.
        input_specs (dict[str, TensorSpec]): Graph input_specs dictionary to update in place.
    """
    for nid, node in list(nodes.items()):
        if node.op_type == "Input":
            param_name = str(node.attributes.get("name", nid))
            if param_name not in inputs:
                inputs.append(param_name)
            if (
                node.dtype is not None or node.shape_metadata is not None
            ) and param_name not in input_specs:
                spec_dtype = node.dtype or DType.float32
                spec_shape = (
                    node.shape_metadata
                    if isinstance(node.shape_metadata, tuple)
                    else ()
                )
                input_specs[param_name] = TensorSpec(shape=spec_shape, dtype=spec_dtype)


def _infer_graph_outputs(
    nodes: dict[str, LogicalNode],
    explicit_outputs: list[str] | None,
) -> list[str]:
    """Resolve explicit graph outputs or infer leaf/Output nodes when omitted.

    Args:
        nodes (dict[str, LogicalNode]): Graph node dictionary.
        explicit_outputs (list[str] | None): Explicit output list if provided.

    Returns:
        list[str]: Resolved output identifiers.
    """
    if explicit_outputs is not None:
        return list(explicit_outputs)
    consumed: set[str] = {inp for n in nodes.values() for inp in n.inputs}
    out_ids: list[str] = []
    for nid, n in nodes.items():
        if n.op_type == "Output":
            for inp in n.inputs:
                if inp not in out_ids:
                    out_ids.append(inp)
        elif nid not in consumed and not any(o in consumed for o in n.outputs):
            out_ids.append(nid)
    return out_ids


class LogicalGraph(BaseModel):
    """Represents a computational model or function definition.

    Attributes:
        name (str): Name of the graph or class.
        nodes (NodeDict): Mapping of node IDs to LogicalNode instances.
        inputs (list[str]): Ordered list of graph input variable names.
        input_specs (dict[str, TensorSpec]): Type and shape specifications for graph inputs.
        outputs (list[str]): Ordered list of graph output identifiers or SSA names.
        initializers (dict[str, InitializerValue]): Constant parameters and weights keyed by tensor name.
        mesh (LogicalMesh | None): Optional logical device mesh for distributed execution.
        pipeline_topology (PipelineTopologyConfig | None): Optional distributed pipeline topology.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    name: str = "Model"
    nodes: NodeDict = Field(default_factory=NodeDict)
    inputs: list[str] = Field(default_factory=list)
    input_specs: dict[str, TensorSpec] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)
    initializers: dict[str, InitializerValue] = Field(default_factory=dict)
    mesh: LogicalMesh | None = None
    pipeline_topology: PipelineTopologyConfig | None = None

    _cached_output_map: dict[str, tuple[LogicalNode, int]] | None = PrivateAttr(
        default=None
    )
    _cached_nodes_version: int = PrivateAttr(default=-1)
    _cached_outputs_node: LogicalNode | None = PrivateAttr(default=None)
    _cached_outputs_list_ref: list[str] | None = PrivateAttr(default=None)
    _cached_outputs_version: int = PrivateAttr(default=-1)
    _edge_value_names: dict[tuple[str, int], str] = PrivateAttr(default_factory=dict)

    @pydantic.field_validator("nodes", mode="before")
    @classmethod
    def _validate_nodes_field(
        cls, val: Mapping[str, LogicalNode] | NodeDict | None
    ) -> NodeDict:
        """Validate and normalize nodes into a NodeDict.

        Args:
            val (Mapping[str, LogicalNode] | NodeDict | None): Input nodes mapping.

        Returns:
            NodeDict: Normalized NodeDict instance.
        """
        return _normalize_graph_nodes(val)

    def __init__(
        self,
        name: str = "Model",
        nodes: Mapping[str, LogicalNode] | NodeDict | None = None,
        outputs: list[str] | None = None,
        mesh: LogicalMesh | None = None,
        inputs: list[str] | None = None,
        input_specs: Mapping[str, TensorSpec] | None = None,
        initializers: Mapping[str, InitializerValue] | None = None,
        *,
        edges: list[LogicalEdge] | None = None,
        pipeline_topology: PipelineTopologyConfig | None = None,
    ) -> None:
        """Initialize a LogicalGraph instance.

        Args:
            name (str): Name of the graph model or class (default: 'Model').
            nodes (Mapping[str, LogicalNode] | NodeDict | None): Mapping of node IDs
                to LogicalNode instances.
            outputs (list[str] | None): List of explicit output IDs. If omitted, deduced from
                nodes with out-degree zero (leaf nodes).
            mesh (LogicalMesh | None): Optional logical device mesh for distributed execution.
            inputs (list[str] | None): List of graph input variable names.
            input_specs (Mapping[str, TensorSpec] | None): Mapping of input names to TensorSpecs.
            initializers (Mapping[str, InitializerValue] | None): Constant tensor initializers.
            edges (list[LogicalEdge] | None): Optional list of directed edges to populate
                directly into each target node's inputs list.
            pipeline_topology (PipelineTopologyConfig | None): Optional distributed pipeline topology.
        """
        norm_nodes = _normalize_graph_nodes(nodes)
        norm_inputs = list(inputs) if inputs is not None else []
        norm_specs = dict(input_specs) if input_specs is not None else {}
        norm_inits = dict(initializers) if initializers is not None else {}

        _auto_detect_input_pseudo_nodes(norm_nodes, norm_inputs, norm_specs)

        edge_val_names: dict[tuple[str, int], str] = {}
        if edges is not None:
            _apply_edges_list(
                norm_nodes,
                edges,
                clear_existing=False,
                edge_value_names=edge_val_names,
            )

        norm_outputs = _infer_graph_outputs(norm_nodes, outputs)

        super().__init__(
            name=name,
            nodes=norm_nodes,
            inputs=norm_inputs,
            input_specs=norm_specs,
            outputs=norm_outputs,
            initializers=norm_inits,
            mesh=mesh,
            pipeline_topology=pipeline_topology,
        )
        self._edge_value_names = edge_val_names

    @property
    def nodes_list(self) -> list[LogicalNode]:
        """Return list of LogicalNode instances in dictionary insertion order.

        Time Complexity:
            O(V) where V is the number of nodes in the graph.

        Returns:
            list[LogicalNode]: Ordered list of nodes.
        """
        return list(self.nodes.values())

    def _resolve_output_port(
        self,
        output_name: str,
        output_map: dict[str, tuple[LogicalNode, int]] | None = None,
    ) -> tuple[LogicalNode, int] | None:
        """Resolve an SSA output name or 'node_id:idx' token to its (producer_node, port_index).

        Args:
            output_name (str): SSA output tensor identifier or 'node_id:idx'.
            output_map (dict[str, tuple[LogicalNode, int]] | None): Pre-fetched output map if available.

        Returns:
            tuple[LogicalNode, int] | None: Producing node and 0-based output index, or None.
        """
        active_map = self._output_map if output_map is None else output_map
        res = active_map.get(output_name)
        if res is not None:
            return res
        if ":" in output_name:
            node_part, idx_part = output_name.rsplit(":", 1)
            if node_part in self.nodes and idx_part.lstrip("-").isdigit():
                return self.nodes[node_part], int(idx_part)
        return None

    def get_producing_output_index(
        self, output_name: str
    ) -> tuple[LogicalNode, int] | None:
        """Resolve both the producing node and the specific output port index.

        Time Complexity:
            O(1) amortized via cached ``_output_map`` lookup.

        Args:
            output_name (str): SSA output tensor identifier or 'node_id:idx'.

        Returns:
            tuple[LogicalNode, int] | None: Producing node and 0-based output index, or None.
        """
        return self._resolve_output_port(output_name)

    @property
    def edges(self) -> list[LogicalEdge]:
        """Derive and return directed edges dynamically from each node's inputs.

        Time Complexity:
            O(V + E) where V is the number of nodes and E is the number of input connections.

        Returns:
            list[LogicalEdge]: List of directed edges from upstream inputs to target nodes.
        """
        output_map = self._output_map
        result: list[LogicalEdge] = []
        for target_node in self.nodes.values():
            for target_idx, src in enumerate(target_node.inputs):
                prod_info = self._resolve_output_port(src, output_map=output_map)
                src_idx = prod_info[1] if prod_info is not None else 0
                val_name = self._edge_value_names.get((target_node.id, target_idx))
                result.append(
                    LogicalEdge(
                        source=src,
                        target=target_node.id,
                        source_idx=src_idx,
                        target_idx=target_idx,
                        value_name=val_name,
                    )
                )
        return result

    @edges.setter
    def edges(self, edge_list: list[LogicalEdge]) -> None:
        """Synchronize edges into target nodes' inputs.

        Args:
            edge_list (list[LogicalEdge]): List of edges to configure.
        """
        _apply_edges_list(
            self.nodes,
            edge_list,
            clear_existing=True,
            edge_value_names=self._edge_value_names,
        )

    def set_pipeline_topology(self, topology: PipelineTopologyConfig) -> None:
        """Set distributed pipeline topology configuration for this graph.

        Args:
            topology (PipelineTopologyConfig): Distributed pipeline topology configuration.
        """
        self.pipeline_topology = topology

    def add_node(self, node: LogicalNode) -> None:
        """Add or update a node in the graph and invalidate the output map cache.

        Args:
            node (LogicalNode): The node to add or update.
        """
        self.nodes[node.id] = node
        self._cached_output_map = None

    def add_edge(
        self,
        source: str,
        target: str,
        source_idx: int | None = 0,
        target_idx: int | None = None,
        value_name: str | None = None,
    ) -> None:
        """Connect source node to target node, updating target inputs.

        Supports optional multi-output ``source_idx``, explicit ``target_idx`` for
        multi-input self-edges (e.g. ``Add(x, x)``), and ``value_name`` metadata.

        Args:
            source (str): Identifier of the upstream source node or SSA value.
            target (str): Identifier of the downstream target node.
            source_idx (int | None): Output port index of the source node (default: 0).
            target_idx (int | None): Input port index on the target node. When specified
                at or beyond ``len(target_node.inputs)``, allows duplicate connections from
                the same source (e.g. ``Add(x, x)``).
            value_name (str | None): Optional SSA value name for the edge.
        """
        if target not in self.nodes:
            return
        target_node = self.nodes[target]
        if source in self.nodes and source_idx is not None and source_idx != 0:
            src_name = self.nodes[source].get_output_name(source_idx)
        elif value_name and source not in self.nodes:
            src_name = value_name
        else:
            src_name = source

        allow_multi = target_idx is not None and target_idx >= len(target_node.inputs)
        if src_name not in target_node.inputs or allow_multi:
            actual_idx = len(target_node.inputs)
            target_node.inputs.append(src_name)
            if value_name is not None:
                self._edge_value_names[(target, actual_idx)] = value_name

    @property
    def _output_map(self) -> dict[str, tuple[LogicalNode, int]]:
        """Cached O(1) mapping of SSA names to (producer_node, output_index).

        Uses O(1) version counters on ``NodeDict`` and tracked ``outputs`` lists to
        avoid O(V) rehashing on every lookup.

        Returns:
            dict[str, tuple[LogicalNode, int]]: Mapping from output SSA name to (node, port_index).
        """
        nodes_dict = self.nodes
        dict_ver = (
            nodes_dict.version if isinstance(nodes_dict, NodeDict) else len(nodes_dict)
        )
        tracked_node = self._cached_outputs_node
        if (
            self._cached_output_map is not None
            and dict_ver == self._cached_nodes_version
            and (
                tracked_node is None
                or (
                    tracked_node.outputs is self._cached_outputs_list_ref
                    and getattr(tracked_node.outputs, "version", -1)
                    == self._cached_outputs_version
                )
            )
        ):
            return self._cached_output_map

        mapping: dict[str, tuple[LogicalNode, int]] = {}
        first_multi_node: LogicalNode | None = None
        for node in nodes_dict.values():
            if first_multi_node is None:
                first_multi_node = node
            for idx, out_name in enumerate(node.outputs):
                mapping[out_name] = (node, idx)

        self._cached_output_map = mapping
        self._cached_nodes_version = dict_ver
        self._cached_outputs_node = first_multi_node
        self._cached_outputs_list_ref = (
            first_multi_node.outputs if first_multi_node is not None else None
        )
        self._cached_outputs_version = (
            getattr(first_multi_node.outputs, "version", 0)
            if first_multi_node is not None
            else 0
        )
        return mapping

    def get_output_producer(self, output_name: str) -> LogicalNode | None:
        """Retrieve the LogicalNode that produces the specified output name.

        Time Complexity:
            O(1) amortized via cached ``_output_map`` lookup.

        Args:
            output_name (str): Identifier or SSA name of the output.

        Returns:
            LogicalNode | None: The producing node if found, otherwise None.
        """
        res = self.get_producing_output_index(output_name)
        if res is not None:
            return res[0]
        return None

    def clone(self) -> LogicalGraph:
        """Create a complete, isolated clone of this graph and all its nodes.

        Returns:
            LogicalGraph: A deep copy of this graph with isolated nodes and edges.
        """
        cloned_nodes = {nid: node.clone() for nid, node in self.nodes.items()}
        cloned_specs = {k: copy.deepcopy(v) for k, v in self.input_specs.items()}
        cloned_inits = copy.deepcopy(self.initializers)
        cloned_mesh = copy.deepcopy(self.mesh) if self.mesh is not None else None
        cloned_topology = (
            copy.deepcopy(self.pipeline_topology)
            if self.pipeline_topology is not None
            else None
        )

        new_g = LogicalGraph(
            name=self.name,
            nodes=cloned_nodes,
            inputs=list(self.inputs),
            input_specs=cloned_specs,
            outputs=list(self.outputs),
            mesh=cloned_mesh,
            initializers=cloned_inits,
        )
        new_g._edge_value_names = dict(self._edge_value_names)
        if cloned_topology is not None:
            new_g.set_pipeline_topology(cloned_topology)
        return new_g

    def get_inputs(self, node_id: str) -> list[LogicalNode]:
        """Retrieve upstream input LogicalNode instances for a node.

        Args:
            node_id (str): Identifier of the target node.

        Returns:
            list[LogicalNode]: List of upstream input nodes.
        """
        if node_id not in self.nodes:
            return []
        target_node = self.nodes[node_id]
        result: list[LogicalNode] = []
        for inp_id in target_node.inputs:
            if inp_id in self.nodes:
                result.append(self.nodes[inp_id])
            else:
                producer = self.get_output_producer(inp_id)
                if producer is not None:
                    result.append(producer)
        return result

    def get_outputs(self, node_id: str) -> list[LogicalNode]:
        """Retrieve downstream consumer LogicalNode instances for a node.

        Args:
            node_id (str): Identifier of the source node.

        Returns:
            list[LogicalNode]: List of downstream consumer nodes.
        """
        if node_id not in self.nodes:
            return [node for node in self.nodes.values() if node_id in node.inputs]
        source_node = self.nodes[node_id]
        output_ids = set(source_node.outputs) | {node_id}
        return [
            node
            for node in self.nodes.values()
            if any(
                inp in output_ids or inp.startswith(f"{node_id}:")
                for inp in node.inputs
            )
        ]

    def __len__(self) -> int:
        """Return total node count in graph.

        Returns:
            int: Number of nodes.
        """
        return len(self.nodes)

    def __getitem__(self, node_id: str) -> LogicalNode:
        """Direct lookup of node by its unique identifier.

        Args:
            node_id (str): Node identifier.

        Returns:
            LogicalNode: The requested node.
        """
        return self.nodes[node_id]

    def to_dict(self) -> dict[str, AttributeValue | dict[str, InitializerValue]]:
        """Convert LogicalGraph to a serializable dictionary.

        Returns:
            dict[str, AttributeValue | dict[str, InitializerValue]]: Dictionary representation of the graph.
        """
        mesh_dict: dict[str, JsonValue] | None = None
        if self.mesh is not None:
            mesh_dict = {
                "shape": {k: int(v) for k, v in self.mesh.shape.items()},
                "webrtc_topology": (
                    dict(self.mesh.webrtc_topology.to_dict())
                    if self.mesh.webrtc_topology is not None
                    else None
                ),
            }
        pipe_dict: dict[str, JsonValue] | None = (
            dict(self.pipeline_topology.to_dict())
            if self.pipeline_topology is not None
            else None
        )
        return {
            "edges": [
                {
                    "source": edge.source,
                    "source_idx": edge.source_idx,
                    "target": edge.target,
                    "target_idx": edge.target_idx,
                    "value_name": edge.value_name,
                }
                for edge in self.edges
            ],
            "initializers": copy.deepcopy(self.initializers),
            "input_specs": {
                k: {
                    "dtype": v.dtype,
                    "shape": v.shape,
                    "sparsity": v.sparsity,
                }
                for k, v in self.input_specs.items()
            },
            "inputs": list(self.inputs),
            "mesh": mesh_dict,
            "name": self.name,
            "nodes": {
                nid: _serialize_node_dict(node) for nid, node in self.nodes.items()
            },
            "outputs": list(self.outputs),
            "pipeline_topology": pipe_dict,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the graph to a deterministic JSON string.

        Args:
            indent (int): Indentation spaces for JSON formatting.

        Returns:
            str: Deterministic JSON string representation.
        """
        buf = io.StringIO()
        self.to_stream(buf, indent=indent)
        return buf.getvalue()

    def to_stream(
        self,
        fp: typing.IO[str] | typing.TextIO,
        indent: int = 2,
    ) -> None:
        """Stream the serialized graph directly into a file-like object.

        Avoids allocating a single massive intermediate JSON string or full-graph
        dictionary in memory by streaming edges and nodes entry-by-entry.

        Args:
            fp (typing.IO[str] | typing.TextIO): Writable file-like stream.
            indent (int): Indentation spaces for JSON formatting.
        """
        _stream_write_graph(self, fp, indent=indent)

    def to_file(
        self,
        path: str | Path,
        indent: int = 2,
        compression: str | None = None,
    ) -> None:
        """Serialize graph directly to a file, supporting optional .gz or .zst compression.

        Args:
            path (str | Path): Destination file path.
            indent (int): Indentation spaces for JSON formatting.
            compression (str | None): Optional compression algorithm ('gzip', 'gz', 'zstd', 'zst').
                If None, compression is inferred from file path suffix.

        Raises:
            ImportError: If zstandard is requested but the library is not installed.
            ValueError: If an unsupported compression algorithm is specified.
        """
        path_str = str(path)
        comp = compression.lower() if compression else None
        if comp in ("gzip", "gz") or (comp is None and path_str.endswith(".gz")):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                self.to_stream(f, indent=indent)
        elif comp in ("zstd", "zst") or (
            comp is None and path_str.endswith((".zst", ".zstd"))
        ):
            if zstandard is None:
                raise ImportError("zstandard library is required for .zst compression.")
            cctx = zstandard.ZstdCompressor()
            with open(path, "wb") as raw_f, cctx.stream_writer(raw_f) as compressor:
                text_wrapper = io.TextIOWrapper(compressor, encoding="utf-8")
                self.to_stream(text_wrapper, indent=indent)
                text_wrapper.flush()
        elif comp is None or comp in ("none", "identity"):
            with open(path, "w", encoding="utf-8") as f:
                self.to_stream(f, indent=indent)
        else:
            raise ValueError(f"Unsupported compression format: {compression}")

    @classmethod
    def from_dict(
        cls,
        data: SerializedGraphModel | Mapping[str, RawGraphValue],
    ) -> LogicalGraph:
        """Deserialize a graph from a dictionary representation.

        Args:
            data (SerializedGraphModel | Mapping[str, RawGraphValue]):
                Dictionary or model containing graph fields.

        Returns:
            LogicalGraph: Deserialized graph instance.

        Raises:
            TypeError: If the 'nodes' entry is not a Mapping.
        """
        raw_map = data.model_dump() if isinstance(data, SerializedGraphModel) else data
        raw_nodes = raw_map.get("nodes", {})
        if not isinstance(raw_nodes, Mapping):
            raise TypeError(
                f"LogicalGraph.from_dict expected 'nodes' to be a Mapping, got {type(raw_nodes).__name__}."
            )
        nodes = NodeDict(
            {
                str(nid): _parse_node_dict(ndata, cls)
                for nid, ndata in raw_nodes.items()
                if isinstance(ndata, Mapping)
            }
        )
        return _build_graph_from_parsed_sections(cls, nodes, raw_map)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        compression: str | None = None,
    ) -> LogicalGraph:
        """Deserialize graph directly from a file or compressed archive.

        Supports streaming deserialization from uncompressed JSON, gzip (.gz),
        and zstandard (.zst) formats without intermediate full-file string allocations.

        Args:
            path (str | Path): Path to the input file.
            compression (str | None): Optional compression algorithm ('gzip', 'gz', 'zstd', 'zst').
                If None, compression is inferred from file path suffix.

        Returns:
            LogicalGraph: Deserialized graph instance.

        Raises:
            ImportError: If zstandard is requested but the library is not installed.
            ValueError: If an unsupported compression algorithm is specified.
        """
        path_str = str(path)
        comp = compression.lower() if compression else None
        if comp in ("gzip", "gz") or (comp is None and path_str.endswith(".gz")):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return _stream_parse_graph(f, cls)
        elif comp in ("zstd", "zst") or (
            comp is None and path_str.endswith((".zst", ".zstd"))
        ):
            if zstandard is None:
                raise ImportError(
                    "zstandard library is required for .zst decompression."
                )
            dctx = zstandard.ZstdDecompressor()
            with open(path, "rb") as raw_f, dctx.stream_reader(raw_f) as reader:
                text_wrapper = io.TextIOWrapper(reader, encoding="utf-8")
                return _stream_parse_graph(text_wrapper, cls)
        elif comp is None or comp in ("none", "identity"):
            with open(path, "r", encoding="utf-8") as f:
                return _stream_parse_graph(f, cls)
        else:
            raise ValueError(f"Unsupported compression format: {compression}")

    @classmethod
    def from_json(
        cls,
        json_str: (
            str
            | bytes
            | typing.IO[str]
            | typing.IO[bytes]
            | typing.TextIO
            | typing.BinaryIO
            | Path
        ),
    ) -> LogicalGraph:
        """Deserialize a graph from a JSON string, stream, or file path.

        Supports chunked incremental streaming deserialization from file-like objects
        without allocating intermediate full-graph string representations.

        Args:
            json_str (str | bytes | typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO | Path):
                JSON string, bytes, file-like stream with read() method, or filesystem path.

        Returns:
            LogicalGraph: Deserialized graph instance.

        Raises:
            TypeError: If json_str is of an unsupported type.
        """
        if isinstance(json_str, Path):
            return cls.from_file(json_str)
        if not isinstance(json_str, (str, bytes)) and hasattr(json_str, "read"):
            return _stream_parse_graph(json_str, cls)
        if isinstance(json_str, bytes):
            data = json.loads(json_str.decode("utf-8"))
        elif isinstance(json_str, str):
            stripped = json_str.strip()
            if (
                not stripped.startswith(("{", "["))
                and len(json_str) <= 4096
                and len(json_str.splitlines()) <= 1
                and chr(0) not in json_str
                and os.path.exists(json_str)
            ):
                return cls.from_file(json_str)
            data = json.loads(json_str)
        else:
            raise TypeError(
                f"Unsupported source type for from_json: {type(json_str).__name__}"
            )
        return cls.from_dict(data)


def topological_sort(graph: LogicalGraph, strict: bool = False) -> list[LogicalNode]:
    """Sort graph nodes by dependency order.

    Ensures that for every edge u -> v, u appears before v in the returned list.
    Handles disconnected components and cycles gracefully by appending
    unreachable nodes in their original definition order, or raises
    CyclicGraphError if strict=True.

    Time Complexity:
        O(V + E) where V is the number of nodes and E is the number of edges.

    Args:
        graph (LogicalGraph): The logical graph to sort.
        strict (bool): If True, raises CyclicGraphError on cycle detection.

    Returns:
        list[LogicalNode]: List of nodes in execution order.

    Raises:
        CyclicGraphError: If cycles exist and strict is True.
    """
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {nid: 0 for nid in graph.nodes}

    for nid, node in graph.nodes.items():
        seen_upstreams: set[str] = set()
        for inp_id in node.inputs:
            upstream_id: str | None = None
            if inp_id in graph.nodes:
                upstream_id = inp_id
            else:
                producer = graph.get_output_producer(inp_id)
                if producer is not None and producer.id in graph.nodes:
                    upstream_id = producer.id
            if upstream_id is not None and upstream_id not in seen_upstreams:
                seen_upstreams.add(upstream_id)
                adj[upstream_id].append(nid)
                in_degree[nid] += 1

    initial_roots = sorted([nid for nid in graph.nodes if in_degree[nid] == 0])
    queue = deque(initial_roots)
    sorted_nodes: list[LogicalNode] = []

    while queue:
        u = queue.popleft()
        sorted_nodes.append(graph.nodes[u])

        for v in sorted(adj[u]):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(sorted_nodes) < len(graph.nodes):
        if strict:
            raise CyclicGraphError("Cycle detected in graph during topological sort.")
        seen = {n.id for n in sorted_nodes}
        for nid, n in graph.nodes.items():
            if nid not in seen:
                sorted_nodes.append(n)

    return sorted_nodes


CompiledArtifactT = typing.TypeVar("CompiledArtifactT")


class CompilerBackend(ABC, typing.Generic[CompiledArtifactT]):
    """Abstract base class for compilation backends."""

    @abstractmethod
    def compile(self, graph: LogicalGraph) -> CompiledArtifactT:
        """Compile the Logical Intermediate Representation (IR) into a target artifact.

        Args:
            graph (LogicalGraph): The intermediate representation of the model structure.

        Returns:
            CompiledArtifactT: The compiled output (e.g., source code string, binary buffer, or AST).
        """
        raise NotImplementedError


class BaseFrontend(ABC):
    """Abstract base for registry typing."""


class GraphFrontend(BaseFrontend):
    """Produces LogicalGraph from code via parse/lift chain."""

    @abstractmethod
    def parse_to_graph(self, code: str) -> LogicalGraph:
        """Parse source code into a LogicalGraph.

        Args:
            code (str): The source code to parse.

        Returns:
            LogicalGraph: The constructed intermediate representation.
        """
        raise NotImplementedError


LogicalNode.model_rebuild()
LogicalGraph.model_rebuild()

from ml_switcheroo_ir.sentinels import NoTangent, ZeroTangent  # noqa: E402
