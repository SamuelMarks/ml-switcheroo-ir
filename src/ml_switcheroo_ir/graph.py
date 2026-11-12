"""Core graph intermediate representation Pydantic models and topological utilities."""

from __future__ import annotations

import builtins
import copy
import gzip
import io
import json
import typing
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from enum import Enum
from pathlib import Path

import pydantic
import pydantic_core
import typing_extensions

from ml_switcheroo_ir.distributed import (
    LogicalAxis,
    LogicalMesh,
    PartitionSpec,
    PipelineTopologyConfig,
)
from ml_switcheroo_ir.serialization import (
    RawGraphValue,
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
    graph_to_dict,
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
    describe_value_type,
    is_dynamic_dimension,
)

__all__ = [
    "BaseFrontend",
    "CloneOverrideValue",
    "CompiledArtifactT",
    "CompilerBackend",
    "CyclicGraphError",
    "EdgeList",
    "GraphFrontend",
    "LogicalAxis",
    "LogicalEdge",
    "LogicalGraph",
    "LogicalGraphKwargs",
    "LogicalMesh",
    "LogicalNode",
    "LogicalNodeKwargs",
    "NodeDict",
    "PartitionSpec",
    "VersionedOutputList",
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
    """Sentinel enum indicating an omitted optional parameter.

    Attributes:
        OMITTED (_OmittedSentinel): Singleton marker indicating no override was passed.
    """

    OMITTED = "OMITTED"


_CLONE_OMITTED: typing.Final[_OmittedSentinel] = _OmittedSentinel.OMITTED


class LogicalEdge(pydantic.BaseModel):
    """Represents a directed data-flow edge between two nodes in the graph.

    Attributes:
        source (str): Source node identifier emitting data.
        target (str): Target node identifier consuming data.
        source_idx (int | None): Output port index of the source node (default: 0).
        target_idx (int): Input argument index of the target node (default: 0).
        value_name (str | None): Name of the intermediate SSA value (default: None).
    """

    model_config = pydantic.ConfigDict(extra="forbid")

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


class VersionedOutputList(list[str]):
    """List subclass for LogicalNode.outputs tracking every mutating operation and notifying listeners.

    Attributes:
        version (int): Monotonically increasing local mutation version counter.
    """

    version: int = 0
    _listeners: tuple[typing.Callable[[], None], ...] = ()

    def __init__(self, iterable: Iterable[str] = ()) -> None:
        """Initialize VersionedOutputList.

        Args:
            iterable (Iterable[str]): Initial output names.
        """
        self.version = 0
        self._listeners = ()
        super().__init__(iterable)

    def _register_listener(self, callback: typing.Callable[[], None]) -> None:
        """Register an invalidation callback if not already subscribed.

        Args:
            callback (Callable[[], None]): Zero-argument invalidation callback.
        """
        if callback not in self._listeners:
            self._listeners = (*self._listeners, callback)

    def _bump_version(self) -> None:
        """Increment local version counter and invoke all registered invalidation listeners."""
        self.version += 1
        for callback in self._listeners:
            callback()

    def append(self, item: str) -> None:
        """Append an output name and increment mutation version.

        Args:
            item (str): Output SSA name.
        """
        super().append(item)
        self._bump_version()

    def extend(self, iterable: Iterable[str]) -> None:
        """Extend output names from ``iterable`` and increment mutation version.

        Args:
            iterable (Iterable[str]): Output SSA names to append.
        """
        super().extend(iterable)
        self._bump_version()

    def insert(self, index: typing.SupportsIndex, item: str) -> None:
        """Insert an output name at ``index`` and increment mutation version.

        Args:
            index (SupportsIndex): Insertion index.
            item (str): Output SSA name.
        """
        super().insert(index, item)
        self._bump_version()

    def pop(self, index: typing.SupportsIndex = -1) -> str:
        """Remove and return output name at ``index`` and increment mutation version.

        Args:
            index (SupportsIndex): Element index to pop (default -1).

        Returns:
            str: Removed output SSA name.
        """
        val = super().pop(index)
        self._bump_version()
        return val

    def remove(self, item: str) -> None:
        """Remove first occurrence of ``item`` and increment mutation version.

        Args:
            item (str): Output SSA name to remove.
        """
        super().remove(item)
        self._bump_version()

    def clear(self) -> None:
        """Remove all output names and increment mutation version."""
        super().clear()
        self._bump_version()

    def reverse(self) -> None:
        """Reverse output names in place and increment mutation version."""
        super().reverse()
        self._bump_version()

    @typing.overload
    def __setitem__(self, index: typing.SupportsIndex, value: str) -> None:
        """Assign single output name at integer index.

        Args:
            index (SupportsIndex): Target index.
            value (str): Output SSA name.
        """

    @typing.overload
    def __setitem__(self, index: builtins.slice[int | None, int | None, int | None], value: Iterable[str]) -> None:
        """Assign iterable of output names to slice.

        Args:
            index (slice[int | None, int | None, int | None]): Target slice.
            value (Iterable[str]): Replacement output SSA names.
        """

    def __setitem__(
        self,
        index: typing.SupportsIndex | builtins.slice[int | None, int | None, int | None],
        value: str | Iterable[str],
    ) -> None:
        """Assign output name(s) at index or slice and increment mutation version.

        Args:
            index (SupportsIndex | slice[int | None, int | None, int | None]): Target index or slice.
            value (str | Iterable[str]): Output name or sequence of output names.
        """
        if isinstance(index, builtins.slice):
            super().__setitem__(index, [value] if isinstance(value, str) else value)
        else:
            super().__setitem__(index, str(value))
        self._bump_version()

    def __delitem__(self, index: typing.SupportsIndex | builtins.slice[int | None, int | None, int | None]) -> None:
        """Delete output name(s) at index or slice and increment mutation version.

        Args:
            index (SupportsIndex | slice[int | None, int | None, int | None]): Target index or slice.
        """
        super().__delitem__(index)
        self._bump_version()

    def __imul__(self, count: typing.SupportsIndex, /) -> typing_extensions.Self:
        """In-place repeat output names via ``*=`` and increment mutation version.

        Args:
            count (SupportsIndex): Repetition multiplier.

        Returns:
            Self: Mutated VersionedOutputList instance.
        """
        super().__imul__(count)
        self._bump_version()
        return self

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: type[VersionedOutputList],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.CoreSchema:
        """Return Pydantic core schema for VersionedOutputList.

        Args:
            source_type (type[VersionedOutputList]): Target class.
            handler (pydantic.GetCoreSchemaHandler): Core schema handler.

        Returns:
            pydantic_core.CoreSchema: Core schema wrapping list[str] in VersionedOutputList.
        """
        del source_type
        from pydantic_core import core_schema

        list_schema = handler.generate_schema(list[str])
        return core_schema.no_info_after_validator_function(
            cls,
            list_schema,
        )


_TrackedOutputsList = VersionedOutputList


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


def _coerce_clone_extra_attr(val: CloneOverrideValue) -> AttributeValue:
    """Narrow a non-field ``CloneOverrideValue`` entry into an ``AttributeValue``.

    Args:
        val (CloneOverrideValue): Candidate attribute override value.

    Returns:
        AttributeValue: Validated deep-copied ``AttributeValue``.
    """
    from ml_switcheroo_ir.shapes import SymInt, SymNode

    if val is None or isinstance(
        val,
        (int, float, str, bool, bytes, SymNode, SymInt, DType, TensorShape, TensorSpec),
    ):
        return copy.deepcopy(val)
    if isinstance(val, Mapping):
        return {str(k): _coerce_clone_extra_attr(v) for k, v in val.items() if not isinstance(v, LogicalGraph)}
    if isinstance(val, Sequence):
        return [_coerce_clone_extra_attr(x) for x in val]
    return [_coerce_clone_extra_attr(x) for x in val] if isinstance(val, Iterable) else str(val)


def _coerce_shape_override(val: CloneOverrideValue) -> ShapeMetadataType | Iterable[DimensionType]:
    """Narrow a ``shape_metadata`` override value into ``ShapeMetadataType | Iterable[DimensionType]``.

    Args:
        val (CloneOverrideValue): Candidate shape override.

    Returns:
        ShapeMetadataType | Iterable[DimensionType]: Narrowed shape metadata candidate.

    Raises:
        TypeError: If ``val`` is not ``None``, ``TensorShape``, or an ``Iterable`` of dimensions.
    """
    if val is None or isinstance(val, TensorShape):
        return val
    if isinstance(val, Iterable) and not isinstance(val, (str, bytes, Mapping)):
        from ml_switcheroo_ir.shapes import SymInt, SymNode

        return [x for x in val if isinstance(x, (int, str, SymNode, SymInt))]
    raise TypeError(f"Invalid shape_metadata override: {val!r}")


def _resolve_optional_str_override(
    overrides: Mapping[str, CloneOverrideValue],
    key: str,
    fallback: str | None,
) -> str | None:
    """Resolve an optional ``str | None`` field override from ``overrides``.

    Args:
        overrides (Mapping[str, CloneOverrideValue]): Overrides mapping.
        key (str): Field name to inspect.
        fallback (str | None): Existing node field value when omitted.

    Returns:
        str | None: Resolved optional string value.
    """
    raw = overrides.get(key, _CLONE_OMITTED)
    if isinstance(raw, _OmittedSentinel):
        return fallback
    return None if raw is None else str(raw)


def _resolve_sharding_override(
    overrides: Mapping[str, CloneOverrideValue],
    fallback: PartitionSpec | None,
) -> PartitionSpec | None:
    """Resolve an optional ``PartitionSpec | None`` field override from ``overrides``.

    Args:
        overrides (Mapping[str, CloneOverrideValue]): Overrides mapping.
        fallback (PartitionSpec | None): Existing sharding value when omitted.

    Returns:
        PartitionSpec | None: Deep-copied resolved ``PartitionSpec`` or ``None``.
    """
    raw = overrides.get("sharding", _CLONE_OMITTED)
    if isinstance(raw, _OmittedSentinel):
        return copy.deepcopy(fallback)
    return copy.deepcopy(raw) if isinstance(raw, PartitionSpec) else None


def _resolve_dtype_override(
    overrides: Mapping[str, CloneOverrideValue],
    fallback: DType | None,
) -> DType | None:
    """Resolve an optional ``DType | None`` field override from ``overrides``.

    Args:
        overrides (Mapping[str, CloneOverrideValue]): Overrides mapping.
        fallback (DType | None): Existing ``DType`` value when omitted.

    Returns:
        DType | None: Resolved ``DType`` or ``None``.
    """
    raw = overrides.get("dtype", _CLONE_OMITTED)
    if isinstance(raw, _OmittedSentinel):
        return fallback
    if isinstance(raw, DType):
        return raw
    return None if raw is None else DType.from_str(str(raw))


class LogicalNodeKwargs(typing_extensions.TypedDict, total=False):
    """Structured optional keyword parameters for ``LogicalNode`` construction.

    Attributes:
        attributes (Mapping[str, AttributeValue] | None): Operator configuration attributes.
        inputs (list[str] | None): Ordered list of upstream LogicalNode IDs or SSA names.
        outputs (Iterable[str] | None): Ordered list of output SSA names.
        shape_metadata (ShapeMetadataType | Iterable[DimensionType]): Primary output shape tuple.
        source_ast_ref (str | None): Source AST traceback reference.
        sharding (PartitionSpec | None): Distributed SPMD partition layout.
        dtype (DType | None): Primary output tensor data type.
        output_specs (list[TensorSpec] | None): Explicit output tensor specifications.
        subgraphs (Mapping[str, LogicalGraph | AttributeValue] | None): Nested control-flow subgraphs.
        device (str | None): Target execution device placement.
        stream (str | None): Asynchronous execution stream name.
    """

    attributes: Mapping[str, AttributeValue] | None
    inputs: list[str] | None
    outputs: Iterable[str] | None
    shape_metadata: ShapeMetadataType | Iterable[DimensionType]
    source_ast_ref: str | None
    sharding: PartitionSpec | None
    dtype: DType | None
    output_specs: list[TensorSpec] | None
    subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None
    device: str | None
    stream: str | None


CloneOverrideValue: typing_extensions.TypeAlias = typing.Union[
    AttributeValue,
    Mapping[str, AttributeValue],
    list[str],
    Iterable[DimensionType],
    PartitionSpec,
    list[TensorSpec],
    Mapping[str, typing.Union["LogicalGraph", AttributeValue]],
    _OmittedSentinel,
]


_NODE_CLONE_FIELD_KEYS: typing.Final[frozenset[str]] = frozenset(
    {
        "domain",
        "version",
        "attributes",
        "inputs",
        "outputs",
        "shape_metadata",
        "source_ast_ref",
        "sharding",
        "dtype",
        "output_specs",
        "subgraphs",
        "device",
        "stream",
    }
)


class LogicalNode(pydantic.BaseModel):
    """Represents a computation unit (Layer) in the graph.

    Attributes:
        id (str): Unique identifier (e.g. 'conv1').
        op_type (str): Operation type. Standard types include 'Conv', 'Gemm', as well as advanced primitives.
        domain (str): Operator domain (e.g., 'ai.onnx').
        version (int): Operator set version (e.g., 1).
        attributes (dict[str, AttributeValue]): Dictionary of configuration parameters (e.g. ``kernel_size=3``).
        inputs (list[str]): Ordered list of upstream LogicalNode IDs.
        outputs (VersionedOutputList): Ordered list of output names/SSA value identifiers produced by this node.
        shape_metadata (tuple[DimensionType, ...] | None): Tuple of dimensions or None.
        source_ast_ref (str | None): Traceback to exact file path, line number, and cdd-python AST node ID.
        sharding (PartitionSpec | None): Optional layout specification for distributed placement of this node's output.
        dtype (DType | None): Data type of the node computation or primary output.
        output_specs (list[TensorSpec]): Explicit tensor specifications for all node outputs.
        subgraphs (dict[str, LogicalGraph | AttributeValue]): Nested subgraphs for control flow and autodiff.
        device (str | None): Target device placement string (e.g., 'cuda:0', 'cpu').
        stream (str | None): Asynchronous execution stream name.
    """

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    id: str
    op_type: str = ""
    domain: str = "ai.onnx"
    version: int = 1
    attributes: dict[str, AttributeValue] = pydantic.Field(default_factory=dict)
    inputs: list[str] = pydantic.Field(default_factory=list)
    outputs: VersionedOutputList = pydantic.Field(default_factory=VersionedOutputList)
    shape_metadata: tuple[DimensionType, ...] | None = None
    source_ast_ref: str | None = None
    sharding: PartitionSpec | None = None
    dtype: DType | None = None
    output_specs: list[TensorSpec] = pydantic.Field(default_factory=list)
    subgraphs: dict[
        str,
        typing.Annotated[LogicalGraph | AttributeValue, _AttributeValuePydanticAnnotation],
    ] = pydantic.Field(default_factory=dict)
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

    @pydantic.field_validator("outputs", mode="before")
    @classmethod
    def _ensure_tracked_outputs(cls, val: Iterable[str] | VersionedOutputList) -> VersionedOutputList:
        """Wrap outputs in VersionedOutputList for mutation version tracking.

        Args:
            val (Iterable[str] | VersionedOutputList): Validated outputs sequence.

        Returns:
            VersionedOutputList: VersionedOutputList instance.
        """
        return VersionedOutputList(val)

    def __init__(
        self,
        id: str,
        op_type: str,
        domain: str = "ai.onnx",
        version: int = 1,
        **kwargs: typing_extensions.Unpack[LogicalNodeKwargs],
    ) -> None:
        """Initialize a LogicalNode instance.

        Args:
            id (str): Unique identifier for the node.
            op_type (str): Operation type (e.g. 'Conv', 'Relu').
            domain (str): Operator domain (default: 'ai.onnx').
            version (int): Operator set version (default: 1).
            **kwargs (typing_extensions.Unpack[LogicalNodeKwargs]): Structured optional node fields.

        Raises:
            ValueError: If id or op_type is empty.
        """
        if not id:
            raise ValueError("LogicalNode 'id' must be a non-empty string.")
        if not op_type:
            raise ValueError("LogicalNode 'op_type' must be a non-empty string.")
        attributes = kwargs.get("attributes")
        inputs = kwargs.get("inputs")
        outputs = kwargs.get("outputs")
        output_specs = kwargs.get("output_specs")
        subgraphs = kwargs.get("subgraphs")
        super().__init__(
            id=id,
            op_type=op_type,
            domain=domain,
            version=version,
            attributes=dict(attributes) if attributes is not None else {},
            inputs=list(inputs) if inputs is not None else [],
            outputs=VersionedOutputList(outputs if outputs is not None else [id]),
            shape_metadata=_normalize_shape_metadata(kwargs.get("shape_metadata")),
            source_ast_ref=kwargs.get("source_ast_ref"),
            sharding=kwargs.get("sharding"),
            dtype=kwargs.get("dtype"),
            output_specs=list(output_specs) if output_specs is not None else [],
            subgraphs=dict(subgraphs) if subgraphs is not None else {},
            device=kwargs.get("device"),
            stream=kwargs.get("stream"),
        )

    def set_shape_metadata(self, shape_metadata: ShapeMetadataType | Iterable[DimensionType]) -> None:
        """Set normalized shape metadata on this node.

        Args:
            shape_metadata (ShapeMetadataType | Iterable[DimensionType]): Shape dimensions or TensorShape instance.
        """
        self.shape_metadata = _normalize_shape_metadata(shape_metadata)

    def update_attributes(
        self,
        attributes: Mapping[str, AttributeValue] | None = None,
        **kwargs: AttributeValue,
    ) -> None:
        """Update configuration attributes on this node in-place.

        Args:
            attributes (Mapping[str, AttributeValue] | None): Optional mapping of attributes to merge.
            **kwargs (AttributeValue): Keyword attribute overrides to merge.
        """
        if attributes is not None:
            self.attributes.update(attributes)
        if kwargs:
            self.attributes.update(kwargs)

    def set_outputs(self, outputs: Iterable[str]) -> None:
        """Replace output SSA names on this node with a new VersionedOutputList.

        Args:
            outputs (Iterable[str]): Replacement sequence of output SSA names.
        """
        self.outputs = VersionedOutputList(outputs)

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

    def _build_cloned_node_kwargs(
        self,
        overrides: Mapping[str, CloneOverrideValue],
    ) -> LogicalNodeKwargs:
        """Build resolved ``LogicalNodeKwargs`` dictionary for ``clone()`` from ``overrides``.

        Args:
            overrides (Mapping[str, CloneOverrideValue]): Field and attribute override mapping.

        Returns:
            LogicalNodeKwargs: Deep-copied and normalized field keyword arguments.
        """
        raw_attrs = overrides.get("attributes")
        cloned_attrs: dict[str, AttributeValue] = copy.deepcopy(
            dict(raw_attrs) if isinstance(raw_attrs, Mapping) else self.attributes
        )
        for k, v in overrides.items():
            if k not in _NODE_CLONE_FIELD_KEYS and not isinstance(v, (_OmittedSentinel, PartitionSpec)):
                cloned_attrs[k] = _coerce_clone_extra_attr(v)

        raw_sub = overrides.get("subgraphs")
        active_sub = raw_sub if isinstance(raw_sub, Mapping) else self.subgraphs
        cloned_subgraphs: dict[str, LogicalGraph | AttributeValue] = {
            str(k): (v.clone() if isinstance(v, LogicalGraph) else copy.deepcopy(v)) for k, v in active_sub.items()
        }

        raw_shape = overrides.get("shape_metadata", _CLONE_OMITTED)
        resolved_shape: tuple[DimensionType, ...] | None = (
            copy.deepcopy(self.shape_metadata)
            if isinstance(raw_shape, _OmittedSentinel)
            else _normalize_shape_metadata(_coerce_shape_override(raw_shape))
        )

        raw_inputs = overrides.get("inputs")
        raw_outputs = overrides.get("outputs")
        raw_specs = overrides.get("output_specs")
        resolved_inputs = [str(x) for x in raw_inputs] if isinstance(raw_inputs, list) else list(self.inputs)
        resolved_outputs = [str(x) for x in raw_outputs] if isinstance(raw_outputs, list) else list(self.outputs)
        resolved_specs = [
            copy.deepcopy(s)
            for s in (
                [x for x in raw_specs if isinstance(x, TensorSpec)]
                if isinstance(raw_specs, list)
                else self.output_specs
            )
        ]

        return {
            "attributes": cloned_attrs,
            "inputs": resolved_inputs,
            "outputs": resolved_outputs,
            "shape_metadata": resolved_shape,
            "source_ast_ref": _resolve_optional_str_override(overrides, "source_ast_ref", self.source_ast_ref),
            "sharding": _resolve_sharding_override(overrides, self.sharding),
            "dtype": _resolve_dtype_override(overrides, self.dtype),
            "output_specs": resolved_specs,
            "subgraphs": cloned_subgraphs,
            "device": _resolve_optional_str_override(overrides, "device", self.device),
            "stream": _resolve_optional_str_override(overrides, "stream", self.stream),
        }

    def clone(
        self,
        id: str | None = None,
        op_type: str | None = None,
        **kwargs: CloneOverrideValue,
    ) -> typing_extensions.Self:
        """Create a deep clone of this LogicalNode with optional keyword overrides.

        Args:
            id (str | None): Override node identifier.
            op_type (str | None): Override operation type.
            **kwargs (CloneOverrideValue): Field overrides (``domain``, ``version``, ``attributes``,
                ``inputs``, ``outputs``, ``shape_metadata``, ``source_ast_ref``, ``sharding``,
                ``dtype``, ``output_specs``, ``subgraphs``, ``device``, ``stream``) and extra
                attribute key-value overrides.

        Returns:
            typing_extensions.Self: A cloned node instance of the same concrete class.
        """
        cloned_id = id if id is not None else self.id
        cloned_op = op_type if op_type is not None else self.op_type
        raw_dom = kwargs.get("domain")
        cloned_domain = str(raw_dom) if isinstance(raw_dom, str) else self.domain
        raw_ver = kwargs.get("version")
        cloned_version = int(raw_ver) if isinstance(raw_ver, int) and not isinstance(raw_ver, bool) else self.version
        node_kwargs = self._build_cloned_node_kwargs(kwargs)
        cls = type(self)
        return cls(
            id=cloned_id,
            op_type=cloned_op,
            domain=cloned_domain,
            version=cloned_version,
            **node_kwargs,
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


@typing.runtime_checkable
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


class NodeDict(MutableMapping[str, LogicalNode]):
    """Strictly typed mutable mapping from node IDs to ``LogicalNode`` instances with version tracking.

    Uses explicit composition over ``dict[str, LogicalNode]`` so every insertion, update,
    ``setdefault``, ``pop``, ``popitem``, ``clear``, and ``|=`` merge enforces ``key == node.id``
    and invalidates cached SSA producer maps deterministically.

    Attributes:
        version (int): Monotonically increasing mutation version counter.
    """

    version: int

    def __init__(
        self,
        mapping_or_iterable: Mapping[str, LogicalNode] | Iterable[tuple[str, LogicalNode]] | None = None,
        **kwargs: LogicalNode,
    ) -> None:
        """Initialize NodeDict and set initial version counter.

        Args:
            mapping_or_iterable (Mapping[str, LogicalNode] | Iterable[tuple[str, LogicalNode]] | None):
                Initial mapping or key-value pairs.
            **kwargs (LogicalNode): Keyword arguments forwarded to ``update``.
        """
        self.version = 0
        self._data: dict[str, LogicalNode] = {}
        self._tracked_refs: list[tuple[LogicalNode, VersionedOutputList]] = []
        if mapping_or_iterable is not None:
            self.update(mapping_or_iterable, **kwargs)
        elif kwargs:
            self.update(**kwargs)

    def _bump_version(self) -> None:
        """Increment mutation version counter when nodes or their outputs change."""
        self.version += 1

    def _bind_entry(self, key: str, value: LogicalNode) -> None:
        """Validate ``key == value.id``, subscribe ``self._bump_version`` to ``value.outputs``, and track identity.

        Args:
            key (str): Mapping key in ``NodeDict``.
            value (LogicalNode): Target LogicalNode instance.

        Raises:
            ValueError: If ``key != value.id``.
        """
        if key != value.id:
            raise ValueError(f"NodeDict key {key!r} does not match LogicalNode.id {value.id!r}.")
        out_list = value.outputs
        out_list._register_listener(self._bump_version)
        self._tracked_refs = [*self._tracked_refs, (value, out_list)]

    @property
    def effective_version(self) -> int:
        """Return version counter after checking for ``node.outputs = [...]`` reassignments.

        Returns:
            int: Up-to-date monotonic mutation version counter.
        """
        reassigned = len(self._tracked_refs) != len(self._data)
        new_refs: list[tuple[LogicalNode, VersionedOutputList]] = []
        prev_by_id = {id(n): lst for n, lst in self._tracked_refs}
        for node in self._data.values():
            cur_list = node.outputs
            if prev_by_id.get(id(node)) is not cur_list:
                cur_list._register_listener(self._bump_version)
                reassigned = True
            new_refs.append((node, cur_list))
        if reassigned:
            self._tracked_refs = new_refs
            self.version += 1
        return self.version

    def __len__(self) -> int:
        """Return total number of nodes in the mapping.

        Returns:
            int: Node count.
        """
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        """Iterate over node ID keys in insertion order.

        Yields:
            str: Node identifier key.
        """
        yield from self._data

    def __repr__(self) -> str:
        """Return dictionary-style string representation.

        Returns:
            str: String representation of underlying mapping.
        """
        return repr(self._data)

    def __eq__(self, other: object) -> bool:
        """Compare equality against any ``Mapping[str, LogicalNode]``.

        Args:
            other (object): Mapping or object to compare against.

        Returns:
            bool: True if ``other`` is a ``Mapping`` with equal key-value pairs.
        """
        if isinstance(other, NodeDict):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return False

    def copy(self) -> NodeDict:
        """Return a shallow ``NodeDict`` copy referencing the same ``LogicalNode`` entries.

        Returns:
            NodeDict: Shallow copy of this ``NodeDict``.
        """
        return NodeDict(self._data)

    def __getitem__(self, key: str | int) -> LogicalNode:
        """Retrieve a LogicalNode by string ID or 0-based insertion-order integer index.

        Args:
            key (str | int): Node ID string or integer positional index.

        Returns:
            LogicalNode: Matched LogicalNode instance.
        """
        if isinstance(key, int) and not isinstance(key, bool):
            return list(self._data.values())[key]
        return self._data[str(key)]

    def __setitem__(self, key: str, value: LogicalNode) -> None:
        """Insert or update a node, validate ``key == value.id``, attach output listeners, and increment version.

        Args:
            key (str): Node ID.
            value (LogicalNode): LogicalNode instance.
        """
        self._bind_entry(key, value)
        self._data[key] = value
        self.version += 1

    def __delitem__(self, key: str) -> None:
        """Delete a node by ID and increment version.

        Args:
            key (str): Node ID.
        """
        del self._data[key]
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
    def pop(self, key: str, default: LogicalNode | _DefaultT) -> LogicalNode | _DefaultT:
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
            res: LogicalNode | _DefaultT = self._data.pop(key)
        else:
            res = self._data.pop(key, default)
        self.version += 1
        return res

    def clear(self) -> None:
        """Clear all nodes and increment version."""
        self._data.clear()
        self._tracked_refs = []
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
        """Update mapping with new nodes, validate ``key == node.id``, and increment version.

        Args:
            other (_SupportsNodeKeysAndGetItem | Iterable[tuple[str, LogicalNode]]):
                Mapping or iterable of key-value pairs.
            **kwargs (LogicalNode): Keyword node entries.
        """
        if isinstance(other, _SupportsNodeKeysAndGetItem):
            for k in other.keys():
                val = other[k]
                self._bind_entry(k, val)
                self._data[k] = val
        else:
            for k, v in other:
                self._bind_entry(k, v)
                self._data[k] = v
        for k, v in kwargs.items():
            self._bind_entry(k, v)
            self._data[k] = v
        self.version += 1

    def __ior__(
        self,
        other: _SupportsNodeKeysAndGetItem | Iterable[tuple[str, LogicalNode]],
        /,
    ) -> typing_extensions.Self:
        """In-place merge ``other`` into this ``NodeDict`` (``|=``), validating keys and bumping version.

        Args:
            other (_SupportsNodeKeysAndGetItem | Iterable[tuple[str, LogicalNode]]):
                Mapping or iterable of ``(node_id, LogicalNode)`` pairs.

        Returns:
            typing_extensions.Self: Updated ``NodeDict`` instance.
        """
        self.update(other)
        return self

    def __or__(
        self,
        other: _SupportsNodeKeysAndGetItem | Iterable[tuple[str, LogicalNode]],
        /,
    ) -> NodeDict:
        """Return a new ``NodeDict`` merging ``self`` and ``other`` (``|``).

        Args:
            other (_SupportsNodeKeysAndGetItem | Iterable[tuple[str, LogicalNode]]):
                Mapping or iterable of ``(node_id, LogicalNode)`` pairs.

        Returns:
            NodeDict: Newly merged ``NodeDict`` instance.
        """
        merged = NodeDict(self._data)
        merged.update(other)
        return merged

    @typing.overload
    def setdefault(self, key: str, default: None = None, /) -> LogicalNode | None:
        """Return node for ``key`` or None when default is omitted.

        Args:
            key (str): Node ID.
            default (None): None default.

        Returns:
            LogicalNode | None: Existing node or None.
        """

    @typing.overload
    def setdefault(self, key: str, default: LogicalNode, /) -> LogicalNode:
        """Return existing node for ``key`` or insert and return ``default``.

        Args:
            key (str): Node ID.
            default (LogicalNode): Default LogicalNode to insert if absent.

        Returns:
            LogicalNode: Existing or newly inserted LogicalNode.
        """

    def setdefault(self, key: str, default: LogicalNode | None = None, /) -> LogicalNode | None:
        """Return existing node for ``key`` or insert ``default`` with version tracking.

        Args:
            key (str): Node ID.
            default (LogicalNode | None): Default node to insert when ``key`` is absent.

        Returns:
            LogicalNode | None: Existing or inserted LogicalNode.

        Raises:
            TypeError: If ``key`` is absent and ``default`` is None.
        """
        if key in self._data:
            return self._data[key]
        if default is None:
            raise TypeError("NodeDict.setdefault requires a LogicalNode default when key is absent.")
        self[key] = default
        return default

    def popitem(self) -> tuple[str, LogicalNode]:
        """Remove and return the last ``(node_id, LogicalNode)`` pair and increment version.

        Returns:
            tuple[str, LogicalNode]: Removed ``(key, node)`` entry.
        """
        pair = self._data.popitem()
        self.version += 1
        return pair

    def _revalidate_all(self) -> None:
        """Ensure keys match node IDs, output listeners are attached, and increment version."""
        self._tracked_refs = []
        for k, v in self._data.items():
            self._bind_entry(k, v)
        self.version += 1

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: type[NodeDict],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.CoreSchema:
        """Generate Pydantic core schema for NodeDict accepting NodeDict or dict[str, LogicalNode].

        Args:
            source_type (type[NodeDict]): Source type annotation.
            handler (pydantic.GetCoreSchemaHandler): Core schema handler.

        Returns:
            pydantic_core.CoreSchema: Core schema wrapping NodeDict and dict[str, LogicalNode].
        """
        del source_type
        from pydantic_core import core_schema

        dict_schema = handler.generate_schema(dict[str, LogicalNode])
        return core_schema.json_or_python_schema(
            json_schema=dict_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(cls),
                    core_schema.no_info_after_validator_function(cls, dict_schema),
                ]
            ),
        )


class EdgeList(tuple["LogicalEdge", ...]):
    """Immutable tuple of ``LogicalEdge`` instances returned by ``LogicalGraph.edges``."""


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
        nodes._revalidate_all()
        return nodes
    if isinstance(nodes, Mapping):
        return NodeDict(nodes)
    raise TypeError(
        f"LogicalGraph 'nodes' must be a Mapping[str, LogicalNode] or None, got {describe_value_type(nodes)}."
    )


def _auto_detect_input_pseudo_nodes(
    nodes: Mapping[str, LogicalNode],
    inputs: list[str],
    input_specs: dict[str, TensorSpec],
) -> None:
    """Populate graph inputs and input_specs from Input pseudo-nodes.

    Args:
        nodes (Mapping[str, LogicalNode]): Graph node dictionary.
        inputs (list[str]): Graph input list to update in place.
        input_specs (dict[str, TensorSpec]): Graph input_specs dictionary to update in place.
    """
    for nid, node in list(nodes.items()):
        if node.op_type == "Input":
            param_name = str(node.attributes.get("name", nid))
            if param_name not in inputs:
                inputs.append(param_name)
            if (node.dtype is not None or node.shape_metadata is not None) and param_name not in input_specs:
                spec_dtype = node.dtype or DType.float32
                spec_shape = node.shape_metadata if isinstance(node.shape_metadata, tuple) else ()
                input_specs[param_name] = TensorSpec(shape=spec_shape, dtype=spec_dtype)


def _infer_graph_outputs(
    nodes: Mapping[str, LogicalNode],
    explicit_outputs: list[str] | None,
) -> list[str]:
    """Resolve explicit graph outputs or infer non-Input leaf/Output nodes when omitted.

    Args:
        nodes (Mapping[str, LogicalNode]): Graph node dictionary.
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
        elif n.op_type != "Input" and nid not in consumed and not any(o in consumed for o in n.outputs):
            out_ids.append(nid)
    return out_ids


class LogicalGraphKwargs(typing_extensions.TypedDict, total=False):
    """Structured optional keyword parameters for ``LogicalGraph`` construction.

    Attributes:
        inputs (list[str] | None): Ordered list of graph input variable names.
        input_specs (Mapping[str, TensorSpec] | None): Type and shape specifications for inputs.
        initializers (Mapping[str, InitializerValue] | None): Constant parameter/weight tensors.
        edges (Sequence[LogicalEdge] | None): Explicit directed edges to populate into node inputs.
        pipeline_topology (PipelineTopologyConfig | None): Distributed pipeline topology config.
        attributes (Mapping[str, AttributeValue] | None): Graph-level metadata attributes.
        loop_options (Mapping[str, AttributeValue] | None): Optional compiler loop configuration.
    """

    inputs: list[str] | None
    input_specs: Mapping[str, TensorSpec] | None
    initializers: Mapping[str, InitializerValue] | None
    edges: Sequence[LogicalEdge] | None
    pipeline_topology: PipelineTopologyConfig | None
    attributes: Mapping[str, AttributeValue] | None
    loop_options: Mapping[str, AttributeValue] | None


class LogicalGraph(pydantic.BaseModel):
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
        attributes (dict[str, AttributeValue]): Validated graph-level metadata attributes.
        loop_options (dict[str, AttributeValue] | None): Optional compiler loop configuration options.
    """

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    name: str = "Model"
    nodes: NodeDict = pydantic.Field(default_factory=NodeDict)
    inputs: list[str] = pydantic.Field(default_factory=list)
    input_specs: dict[str, TensorSpec] = pydantic.Field(default_factory=dict)
    outputs: list[str] = pydantic.Field(default_factory=list)
    initializers: dict[str, InitializerValue] = pydantic.Field(default_factory=dict)
    mesh: LogicalMesh | None = None
    pipeline_topology: PipelineTopologyConfig | None = None
    attributes: dict[str, AttributeValue] = pydantic.Field(default_factory=dict)
    loop_options: dict[str, AttributeValue] | None = None

    _cached_output_map: dict[str, tuple[LogicalNode, int]] | None = pydantic.PrivateAttr(default=None)
    _cached_nodes_version: int = pydantic.PrivateAttr(default=-1)
    _edge_value_names: dict[tuple[str, int], str] = pydantic.PrivateAttr(default_factory=dict)

    @pydantic.field_validator("nodes", mode="before")
    @classmethod
    def _validate_nodes_field(cls, val: Mapping[str, LogicalNode] | NodeDict | None) -> NodeDict:
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
        **kwargs: typing_extensions.Unpack[LogicalGraphKwargs],
    ) -> None:
        """Initialize a LogicalGraph instance.

        Args:
            name (str): Name of the graph model or class (default: 'Model').
            nodes (Mapping[str, LogicalNode] | NodeDict | None): Mapping of node IDs
                to LogicalNode instances.
            outputs (list[str] | None): List of explicit output IDs. If omitted, deduced from
                non-Input nodes with out-degree zero (leaf nodes).
            mesh (LogicalMesh | None): Optional logical device mesh for distributed execution.
            **kwargs (typing_extensions.Unpack[LogicalGraphKwargs]): Optional graph fields
                (``inputs``, ``input_specs``, ``initializers``, ``edges``,
                ``pipeline_topology``, ``attributes``, ``loop_options``).
        """
        norm_nodes = _normalize_graph_nodes(nodes)
        raw_inputs = kwargs.get("inputs")
        raw_specs = kwargs.get("input_specs")
        raw_inits = kwargs.get("initializers")
        raw_edges = kwargs.get("edges")
        raw_attrs = kwargs.get("attributes")
        raw_loop = kwargs.get("loop_options")

        norm_inputs = list(raw_inputs) if raw_inputs is not None else []
        norm_specs = dict(raw_specs) if raw_specs is not None else {}
        norm_inits = dict(raw_inits) if raw_inits is not None else {}
        norm_attrs = dict(raw_attrs) if raw_attrs is not None else {}
        norm_loop = dict(raw_loop) if raw_loop is not None else None

        _auto_detect_input_pseudo_nodes(norm_nodes, norm_inputs, norm_specs)

        edge_val_names: dict[tuple[str, int], str] = {}
        if raw_edges is not None:
            _apply_edges_list(
                norm_nodes,
                list(raw_edges),
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
            pipeline_topology=kwargs.get("pipeline_topology"),
            attributes=norm_attrs,
            loop_options=norm_loop,
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

    def get_producing_output_index(self, output_name: str) -> tuple[LogicalNode, int] | None:
        """Resolve both the producing node and the specific output port index.

        Time Complexity:
            O(1) amortized via cached ``_output_map`` lookup.

        Args:
            output_name (str): SSA output tensor identifier or 'node_id:idx'.

        Returns:
            tuple[LogicalNode, int] | None: Producing node and 0-based output index, or None.
        """
        return self._resolve_output_port(output_name)

    def iter_edges(self) -> Iterator[LogicalEdge]:
        """Lazily yield directed ``LogicalEdge`` instances derived from each node's inputs.

        Yields:
            LogicalEdge: Directed edge from upstream SSA source to target node operand slot.
        """
        output_map = self._output_map
        for target_node in self.nodes.values():
            for target_idx, src in enumerate(target_node.inputs):
                prod_info = self._resolve_output_port(src, output_map=output_map)
                src_idx = prod_info[1] if prod_info is not None else 0
                val_name = self._edge_value_names.get((target_node.id, target_idx))
                yield LogicalEdge(
                    source=src,
                    target=target_node.id,
                    source_idx=src_idx,
                    target_idx=target_idx,
                    value_name=val_name,
                )

    @property
    def edges(self) -> EdgeList:
        """Derive and return an immutable ``EdgeList`` tuple of directed edges from each node's inputs.

        Returning an immutable ``tuple[LogicalEdge, ...]`` prevents accidental silent loss
        of mutations such as ``graph.edges.append(...)``.

        Time Complexity:
            O(V + E) where V is the number of nodes and E is the number of input connections.

        Returns:
            EdgeList: Immutable tuple of directed edges.
        """
        return EdgeList(self.iter_edges())

    @edges.setter
    def edges(self, edge_list: Sequence[LogicalEdge]) -> None:
        """Synchronize edges into target nodes' inputs.

        Args:
            edge_list (Sequence[LogicalEdge]): Sequence of edges to configure.
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

    def replace_nodes(
        self,
        nodes: Mapping[str, LogicalNode] | Iterable[tuple[str, LogicalNode]],
    ) -> None:
        """Replace all nodes in the graph in-place while preserving NodeDict version tracking.

        Args:
            nodes (Mapping[str, LogicalNode] | Iterable[tuple[str, LogicalNode]]):
                Replacement nodes mapping or iterable.
        """
        new_items = dict(nodes)
        self.nodes.clear()
        self.nodes.update(new_items)
        self._cached_output_map = None

    def _resolve_add_edge_src_name(
        self,
        source: str,
        source_idx: int | None,
        value_name: str | None,
    ) -> str:
        """Resolve the SSA input string to wire into ``target_node.inputs`` for ``add_edge``.

        Args:
            source (str): Source node identifier or external SSA token.
            source_idx (int | None): Optional output port index on ``source``.
            value_name (str | None): Explicit SSA value name override.

        Returns:
            str: Resolved SSA value name.
        """
        if value_name is not None and source not in self.nodes:
            return value_name
        if source in self.nodes and source_idx is not None:
            src_node = self.nodes[source]
            if 0 <= source_idx < len(src_node.outputs):
                return src_node.outputs[source_idx]
            if source_idx != 0:
                return src_node.get_output_name(source_idx)
        return value_name if value_name is not None else source

    def add_edge(
        self,
        source: str,
        target: str,
        source_idx: int | None = 0,
        target_idx: int | None = None,
        value_name: str | None = None,
    ) -> None:
        """Connect source node to target node, updating target inputs at the requested port.

        Resolves ``source_idx`` to ``source_node.outputs[source_idx]`` for multi-output
        producers and honors explicit ``target_idx`` operand positioning (including
        repeated operands such as ``Add(x, x)`` and out-of-order operand wiring).

        Args:
            source (str): Identifier of the upstream source node or SSA value.
            target (str): Identifier of the downstream target node.
            source_idx (int | None): Output port index of the source node (default: 0).
            target_idx (int | None): Zero-based operand slot index on the target node.
                When ``None``, appends ``src_name`` if not already present.
            value_name (str | None): Optional SSA value name for the edge.

        Raises:
            KeyError: If ``target`` is not present in ``self.nodes``.
            ValueError: If ``target_idx`` is negative or a boolean.
        """
        if target not in self.nodes:
            raise KeyError(f"Target node '{target}' not found in LogicalGraph.")
        if target_idx is not None and (isinstance(target_idx, bool) or target_idx < 0):
            raise ValueError(f"target_idx must be a non-negative integer, got {target_idx!r}.")

        target_node = self.nodes[target]
        src_name = self._resolve_add_edge_src_name(source, source_idx, value_name)

        if target_idx is None:
            if src_name not in target_node.inputs:
                actual_idx = len(target_node.inputs)
                target_node.inputs.append(src_name)
            else:
                actual_idx = target_node.inputs.index(src_name)
        elif target_idx < len(target_node.inputs):
            target_node.inputs[target_idx] = src_name
            actual_idx = target_idx
        else:
            while len(target_node.inputs) < target_idx:
                target_node.inputs.append("")
            target_node.inputs.append(src_name)
            actual_idx = target_idx

        if value_name is not None:
            self._edge_value_names[(target, actual_idx)] = value_name

    def iter_nodes(self) -> Iterator[LogicalNode]:
        """Iterate over LogicalNode instances in deterministic insertion order.

        Yields:
            LogicalNode: Graph nodes in insertion order.
        """
        yield from self.nodes.values()

    @property
    def _output_map(self) -> dict[str, tuple[LogicalNode, int]]:
        """Cached O(1) mapping of SSA names to (producer_node, output_index).

        Uses O(1) observer-driven version tracking on ``NodeDict`` to avoid O(V)
        rehashing on repeated lookups while invalidating on any node or output mutation.

        Returns:
            dict[str, tuple[LogicalNode, int]]: Mapping from output SSA name to (node, port_index).
        """
        nodes_dict = self.nodes
        dict_ver = nodes_dict.effective_version
        if self._cached_output_map is not None and dict_ver == self._cached_nodes_version:
            return self._cached_output_map

        mapping: dict[str, tuple[LogicalNode, int]] = {}
        for node in nodes_dict.values():
            for idx, out_name in enumerate(node.outputs):
                mapping[out_name] = (node, idx)

        self._cached_output_map = mapping
        self._cached_nodes_version = dict_ver
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
        """Create a complete, isolated clone of this graph and all its nodes and metadata.

        Returns:
            LogicalGraph: A deep copy of this graph with isolated nodes, edges, and attributes.
        """
        cloned_nodes = {nid: node.clone() for nid, node in self.nodes.items()}
        cloned_specs = {k: copy.deepcopy(v) for k, v in self.input_specs.items()}
        cloned_inits = copy.deepcopy(self.initializers)
        cloned_mesh = copy.deepcopy(self.mesh) if self.mesh is not None else None
        cloned_topology = copy.deepcopy(self.pipeline_topology) if self.pipeline_topology is not None else None
        cloned_attrs = copy.deepcopy(self.attributes)
        cloned_loop = copy.deepcopy(self.loop_options) if self.loop_options is not None else None

        new_g = LogicalGraph(
            name=self.name,
            nodes=cloned_nodes,
            inputs=list(self.inputs),
            input_specs=cloned_specs,
            outputs=list(self.outputs),
            mesh=cloned_mesh,
            initializers=cloned_inits,
            pipeline_topology=cloned_topology,
            attributes=cloned_attrs,
            loop_options=cloned_loop,
        )
        new_g._edge_value_names = dict(self._edge_value_names)
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
            if any(inp in output_ids or inp.startswith(f"{node_id}:") for inp in node.inputs)
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

    def __eq__(self, other: object) -> bool:
        """Compare structural equality against another ``LogicalGraph``, ignoring lazy output map caches.

        Args:
            other (object): Object to compare against.

        Returns:
            bool: True if ``other`` is a ``LogicalGraph`` with identical semantic fields and edge metadata.
        """
        if not isinstance(other, LogicalGraph):
            return False
        return (
            self.name == other.name
            and self.nodes == other.nodes
            and self.inputs == other.inputs
            and self.input_specs == other.input_specs
            and self.outputs == other.outputs
            and self.initializers == other.initializers
            and self.mesh == other.mesh
            and self.pipeline_topology == other.pipeline_topology
            and self.attributes == other.attributes
            and self.loop_options == other.loop_options
            and self._edge_value_names == other._edge_value_names
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert LogicalGraph to a canonical lossless JSON-compatible dictionary.

        Returns:
            dict[str, JsonValue]: Canonical dictionary representation of the graph.
        """
        return graph_to_dict(self)

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
        dictionary in memory by streaming edges, initializers, and nodes entry-by-entry.

        Args:
            fp (typing.IO[str] | typing.TextIO): Writable file-like stream.
            indent (int): Indentation spaces for JSON formatting.
        """
        _stream_write_graph(self, fp, indent=indent)

    @classmethod
    def from_stream(
        cls,
        fp: typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO,
    ) -> LogicalGraph:
        """Incrementally stream-deserialize a LogicalGraph from a readable file-like object.

        Args:
            fp (typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO):
                Readable text or binary JSON stream.

        Returns:
            LogicalGraph: Deserialized graph instance.
        """
        return _stream_parse_graph(fp, cls)

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
        elif comp in ("zstd", "zst") or (comp is None and path_str.endswith((".zst", ".zstd"))):
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
        data: Mapping[str, RawGraphValue],
    ) -> LogicalGraph:
        """Deserialize a graph from a dictionary representation.

        Args:
            data (Mapping[str, RawGraphValue]): Dictionary containing graph fields.

        Returns:
            LogicalGraph: Deserialized graph instance.

        Raises:
            TypeError: If the 'nodes' entry is not a Mapping.
        """
        raw_nodes = data.get("nodes", {})
        if not isinstance(raw_nodes, Mapping):
            raise TypeError(
                f"LogicalGraph.from_dict expected 'nodes' to be a Mapping, got {describe_value_type(raw_nodes)}."
            )
        nodes = NodeDict(
            {str(nid): _parse_node_dict(ndata, cls) for nid, ndata in raw_nodes.items() if isinstance(ndata, Mapping)}
        )
        return _build_graph_from_parsed_sections(cls, nodes, data)

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
        elif comp in ("zstd", "zst") or (comp is None and path_str.endswith((".zst", ".zstd"))):
            if zstandard is None:
                raise ImportError("zstandard library is required for .zst decompression.")
            dctx = zstandard.ZstdDecompressor()
            with open(path, "rb") as raw_f, dctx.stream_reader(raw_f) as reader:
                text_wrapper = io.TextIOWrapper(reader, encoding="utf-8")
                return _stream_parse_graph(text_wrapper, cls)
        elif comp is None or comp in ("none", "identity"):
            with open(path, encoding="utf-8") as f:
                return _stream_parse_graph(f, cls)
        else:
            raise ValueError(f"Unsupported compression format: {compression}")

    @classmethod
    def from_json(
        cls,
        json_str: (str | bytes | typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO),
    ) -> LogicalGraph:
        """Deserialize a graph from a JSON string, UTF-8 bytes, or readable stream.

        Does not perform filesystem probing on string inputs; use ``LogicalGraph.from_file``
        exclusively for filesystem paths.

        Args:
            json_str (str | bytes | typing.IO[str] | typing.IO[bytes] | typing.TextIO | typing.BinaryIO):
                JSON string, UTF-8 bytes, or readable stream with ``read()``.

        Returns:
            LogicalGraph: Deserialized graph instance.
        """
        if not isinstance(json_str, (str, bytes)):
            return _stream_parse_graph(json_str, cls)
        text = json_str.decode("utf-8") if isinstance(json_str, bytes) else json_str
        data = json.loads(text)
        return cls.from_dict(data)


def _build_toposort_adjacency(
    graph: LogicalGraph,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Build adjacency list and in-degree map for topological sorting.

    Args:
        graph (LogicalGraph): The logical graph to inspect.

    Returns:
        tuple[dict[str, list[str]], dict[str, int]]: ``(adj, in_degree)`` mappings.
    """
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {nid: 0 for nid in graph.nodes}

    for nid, node in graph.nodes.items():
        seen_upstreams: set[str] = set()
        for inp_id in node.inputs:
            upstream_id: str | None = inp_id if inp_id in graph.nodes else None
            if upstream_id is None:
                producer = graph.get_output_producer(inp_id)
                if producer is not None and producer.id in graph.nodes:
                    upstream_id = producer.id
            if upstream_id is not None and upstream_id not in seen_upstreams:
                seen_upstreams.add(upstream_id)
                adj[upstream_id].append(nid)
                in_degree[nid] += 1
    return adj, in_degree


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
    adj, in_degree = _build_toposort_adjacency(graph)
    queue = deque(sorted(nid for nid in graph.nodes if in_degree[nid] == 0))
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
        sorted_nodes.extend(n for nid, n in graph.nodes.items() if nid not in seen)

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

        Raises:
            NotImplementedError: When not implemented by concrete backend subclass.
        """
        raise NotImplementedError


class BaseFrontend:
    """Base class for frontend registry typing."""


class GraphFrontend(BaseFrontend, ABC):
    """Produces LogicalGraph from code via parse/lift chain."""

    @abstractmethod
    def parse_to_graph(self, code: str) -> LogicalGraph:
        """Parse source code into a LogicalGraph.

        Args:
            code (str): The source code to parse.

        Returns:
            LogicalGraph: The constructed intermediate representation.

        Raises:
            NotImplementedError: When not implemented by concrete frontend subclass.
        """
        raise NotImplementedError


LogicalNode.model_rebuild()
LogicalGraph.model_rebuild()
