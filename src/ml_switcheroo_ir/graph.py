"""Core graph intermediate representation dataclasses and topological utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
import gzip
import io
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, IO, Any, BinaryIO, TextIO, cast, overload
import warnings

from ml_switcheroo_ir.distributed import (
    LogicalAxis,
    LogicalMesh,
    PartitionSpec,
    PipelineTopologyConfig,
)
from ml_switcheroo_ir.serialization import (
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
    ShapeMetadataType,
    TensorShape,
    TensorSpec,
    is_dynamic_dimension,
)

if TYPE_CHECKING:
    from ml_switcheroo_ir.sentinels import NoTangent, ZeroTangent

__all__ = [
    "BaseFrontend",
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


@dataclass
class LogicalEdge:
    """Represents a directed data-flow edge between two nodes in the graph.

    Attributes:
        source (str): Source node identifier emitting data.
        target (str): Target node identifier consuming data.
        source_idx (Optional[int]): Output port index of the source node (default: 0).
        target_idx (int): Input argument index of the target node (default: 0).
        value_name (Optional[str]): Name of the intermediate SSA value (default: None).

    """

    source: str
    target: str
    source_idx: int | None = 0
    target_idx: int = 0
    value_name: str | None = None


class CyclicGraphError(Exception):
    """Raised when a cycle is detected in a graph during topological sorting."""


class _UnsetSentinel(Enum):
    """Internal sentinel enum for distinguishing omitted optional arguments."""

    UNSET_OP_TYPE = "UNSET_OP_TYPE"
    CLONE_UNSET = "CLONE_UNSET"


_UNSET_OP_TYPE = _UnsetSentinel.UNSET_OP_TYPE
_CLONE_UNSET = _UnsetSentinel.CLONE_UNSET


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

    def extend(self, iterable: Iterable[str]) -> None:
        """Extend output names and increment mutation version.

        Args:
            iterable (Iterable[str]): Output SSA names.
        """
        super().extend(iterable)
        self.version += 1


def _normalize_shape_metadata(
    shape_metadata: ShapeMetadataType | Sequence[DimensionType] | object,
) -> ShapeMetadataType:
    """Normalize arbitrary shape_metadata input into a canonical tuple or object.

    Args:
        shape_metadata (ShapeMetadataType | Sequence[DimensionType] | object): Raw shape metadata argument.

    Returns:
        ShapeMetadataType: Normalized tuple of dimensions or preserved custom metadata object.

    Raises:
        TypeError: If any dimension in a shape sequence is a boolean value.
    """
    if shape_metadata is None:
        return None
    if isinstance(shape_metadata, TensorShape):
        return shape_metadata.dims
    if isinstance(shape_metadata, (tuple, list)):
        res_tuple = tuple(shape_metadata)
        for d in res_tuple:
            if isinstance(d, bool):
                raise TypeError(
                    f"Boolean value {d!r} is not a valid tensor dimension in shape_metadata."
                )
        return res_tuple
    if hasattr(shape_metadata, "shape"):
        return _normalize_shape_metadata(shape_metadata.shape)
    if not isinstance(shape_metadata, (str, bytes, dict)) and isinstance(
        shape_metadata, Iterable
    ):
        res_iter = tuple(shape_metadata)
        for d in res_iter:
            if isinstance(d, bool):
                raise TypeError(
                    f"Boolean value {d!r} is not a valid tensor dimension in shape_metadata."
                )
        return res_iter
    return cast("ShapeMetadataType", shape_metadata)


def _normalize_legacy_node_args(
    op_type: str | _UnsetSentinel | None,
    domain: Mapping[str, AttributeValue] | str,
    attributes: Mapping[str, AttributeValue] | None,
    kind: str | None,
    metadata: Mapping[str, AttributeValue] | None,
) -> tuple[str, str, dict[str, AttributeValue]]:
    """Resolve legacy LogicalNode constructor parameters and emit deprecation warnings.

    Args:
        op_type (str | _UnsetSentinel | None): Canonical operation type or sentinel.
        domain (Mapping[str, AttributeValue] | str): Domain identifier string or legacy attribute mapping.
        attributes (Optional[Mapping[str, AttributeValue]]): Canonical attributes mapping.
        kind (Optional[str]): Deprecated alias for op_type.
        metadata (Optional[Mapping[str, AttributeValue]]): Deprecated alias for attributes.

    Returns:
        tuple[str, str, dict[str, AttributeValue]]: Resolved (op_type, domain, attributes).

    Raises:
        ValueError: If neither op_type nor kind is provided, or if both conflict.
    """
    if kind is not None:
        warnings.warn(
            "The 'kind' parameter is deprecated; use 'op_type' instead.",
            DeprecationWarning,
            stacklevel=4,
        )
    if metadata is not None:
        warnings.warn(
            "The 'metadata' parameter is deprecated; use 'attributes' instead.",
            DeprecationWarning,
            stacklevel=4,
        )

    if op_type is _UNSET_OP_TYPE and kind is None:
        raise ValueError(
            "Either 'op_type' or 'kind' must be specified for LogicalNode."
        )
    if (
        op_type is not _UNSET_OP_TYPE
        and op_type is not None
        and kind is not None
        and op_type != kind
    ):
        raise ValueError(
            f"Conflicting op_type ({op_type!r}) and kind ({kind!r}) specified for LogicalNode."
        )

    resolved_op_type = (
        str(op_type)
        if (op_type is not _UNSET_OP_TYPE and op_type is not None)
        else str(kind)
    )

    resolved_domain = "ai.onnx"
    resolved_attributes: dict[str, AttributeValue] = {}
    if isinstance(domain, Mapping):
        warnings.warn(
            "Passing a mapping as 'domain' is deprecated; use 'attributes' instead.",
            DeprecationWarning,
            stacklevel=4,
        )
        resolved_attributes.update(domain)
    else:
        resolved_domain = str(domain)

    if metadata is not None:
        resolved_attributes.update(metadata)
    if attributes is not None:
        resolved_attributes.update(attributes)

    return resolved_op_type, resolved_domain, resolved_attributes


def _normalize_node_init_args(
    op_type: str | _UnsetSentinel | None,
    domain: Mapping[str, AttributeValue] | str,
    attributes: Mapping[str, AttributeValue] | None,
    kind: str | None,
    metadata: Mapping[str, AttributeValue] | None,
) -> tuple[str, str, dict[str, AttributeValue]]:
    """Normalize canonical and legacy LogicalNode constructor arguments.

    Separates the fast canonical construction path from legacy positional/mapping
    coercion and deprecated alias resolution.

    Args:
        op_type (str | _UnsetSentinel | None): Canonical operation type or sentinel.
        domain (Mapping[str, AttributeValue] | str): Domain identifier string or legacy attribute dictionary.
        attributes (Optional[Mapping[str, AttributeValue]]): Canonical attributes dictionary.
        kind (Optional[str]): Deprecated alias for op_type.
        metadata (Optional[Mapping[str, AttributeValue]]): Deprecated alias for attributes.

    Returns:
        tuple[str, str, dict[str, AttributeValue]]: Resolved (op_type, domain, attributes).

    Raises:
        ValueError: If neither op_type nor kind is provided, or if both conflict.
    """
    if (
        kind is None
        and metadata is None
        and isinstance(op_type, str)
        and isinstance(domain, str)
    ):
        return (
            op_type,
            domain,
            dict(attributes) if attributes is not None else {},
        )
    return _normalize_legacy_node_args(op_type, domain, attributes, kind, metadata)


@dataclass
class LogicalNode:
    """Represents a computation unit (Layer) in the graph.

    Attributes:
        id (str): Unique identifier (e.g. 'conv1').
        op_type (str): Operation type. Standard types include 'Conv', 'Gemm', as well as advanced primitives.
        domain (str): Operator domain (e.g., 'ai.onnx').
        version (int): Operator set version (e.g., 1).
        attributes (Dict[str, AttributeValue]): Dictionary of configuration parameters (e.g. ``kernel_size=3``).
        inputs (List[str]): Ordered list of upstream LogicalNode IDs.
        outputs (List[str]): Ordered list of output names/SSA value identifiers produced by this node.
        shape_metadata (ShapeMetadataType): Tuple of dimensions or TensorShape.
        source_ast_ref (Optional[str]): Traceback to exact file path, line number, and cdd-python AST node ID.
        sharding (Optional[PartitionSpec]): Optional layout specification for distributed placement of this node's output.
        dtype (Optional[DType]): Data type of the node computation or primary output.
        output_specs (List[TensorSpec]): Explicit tensor specifications for all node outputs.
        subgraphs (Dict[str, Union[LogicalGraph, AttributeValue]]): Nested subgraphs for control flow and autodiff.
        device (Optional[str]): Target device placement string (e.g., 'cuda:0', 'cpu').
        stream (Optional[str]): Asynchronous execution stream name.

    """

    id: str
    op_type: str = ""
    domain: str = "ai.onnx"
    version: int = 1
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    shape_metadata: ShapeMetadataType = None
    source_ast_ref: str | None = None
    sharding: PartitionSpec | None = None
    dtype: DType | None = None
    output_specs: list[TensorSpec] = field(default_factory=list)
    subgraphs: dict[str, LogicalGraph | AttributeValue] = field(default_factory=dict)
    device: str | None = None
    stream: str | None = None

    @overload
    def __init__(
        self,
        id: str,
        op_type: str = ...,
        domain: Mapping[str, AttributeValue] | str = "ai.onnx",
        version: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: ShapeMetadataType | Sequence[DimensionType] | object = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        dtype: DType | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None = None,
        device: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Initialize LogicalNode using canonical op_type and attributes.

        Args:
            id (str): Unique identifier for the node.
            op_type (str): Operation type.
            domain (Union[Mapping[str, AttributeValue], str]): Operator domain or attributes mapping.
            version (int): Operator version (default: 1).
            attributes (Optional[Mapping[str, AttributeValue]]): Operator attributes.
            inputs (Optional[List[str]]): Ordered list of upstream input IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (ShapeMetadataType | Sequence[DimensionType] | object): Tensor shape metadata.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            dtype (Optional[DType]): Data type of the node.
            output_specs (Optional[List[TensorSpec]]): Explicit output tensor specs.
            subgraphs (Optional[Mapping[str, Union[LogicalGraph, AttributeValue]]]): Nested subgraphs dictionary.
            device (Optional[str]): Target device placement.
            stream (Optional[str]): Target stream identifier.
        """

    @overload
    def __init__(
        self,
        id: str,
        *,
        kind: str,
        domain: Mapping[str, AttributeValue] | str = "ai.onnx",
        version: int = 1,
        metadata: Mapping[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: ShapeMetadataType | Sequence[DimensionType] | object = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        dtype: DType | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None = None,
        device: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Initialize LogicalNode using legacy/alternative kind and metadata.

        Args:
            id (str): Unique identifier for the node.
            kind (str): Alternative alias for op_type (deprecated).
            domain (Union[Mapping[str, AttributeValue], str]): Operator domain or attributes mapping.
            version (int): Operator version (default: 1).
            metadata (Optional[Mapping[str, AttributeValue]]): Alternative alias for attributes (deprecated).
            inputs (Optional[List[str]]): Ordered list of upstream input IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (ShapeMetadataType | Sequence[DimensionType] | object): Tensor shape metadata.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            dtype (Optional[DType]): Data type of the node.
            output_specs (Optional[List[TensorSpec]]): Explicit output tensor specs.
            subgraphs (Optional[Mapping[str, Union[LogicalGraph, AttributeValue]]]): Nested subgraphs dictionary.
            device (Optional[str]): Target device placement.
            stream (Optional[str]): Target stream identifier.
        """

    @overload
    def __init__(
        self,
        id: str,
        op_type: str | None = None,
        domain: Mapping[str, AttributeValue] | str = "ai.onnx",
        version: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: ShapeMetadataType | Sequence[DimensionType] | object = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        dtype: DType | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None = None,
        device: str | None = None,
        stream: str | None = None,
        *,
        kind: str | None = None,
        metadata: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        """Initialize LogicalNode with flexible parameter naming conventions.

        Args:
            id (str): Unique identifier for the node.
            op_type (Optional[str]): Operation type.
            domain (Union[Mapping[str, AttributeValue], str]): Operator domain or attributes mapping.
            version (int): Operator version (default: 1).
            attributes (Optional[Mapping[str, AttributeValue]]): Operator attributes.
            inputs (Optional[List[str]]): Ordered list of upstream input IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (ShapeMetadataType | Sequence[DimensionType] | object): Tensor shape metadata.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            dtype (Optional[DType]): Data type of the node.
            output_specs (Optional[List[TensorSpec]]): Explicit output tensor specs.
            subgraphs (Optional[Mapping[str, Union[LogicalGraph, AttributeValue]]]): Nested subgraphs dictionary.
            device (Optional[str]): Target device placement.
            stream (Optional[str]): Target stream identifier.
            kind (Optional[str]): Alternative alias for op_type (deprecated).
            metadata (Optional[Mapping[str, AttributeValue]]): Alternative alias for attributes (deprecated).
        """

    def __init__(
        self,
        id: str,
        op_type: str | _UnsetSentinel | None = _UNSET_OP_TYPE,
        domain: Mapping[str, AttributeValue] | str = "ai.onnx",
        version: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: ShapeMetadataType | Sequence[DimensionType] | object = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        dtype: DType | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None = None,
        device: str | None = None,
        stream: str | None = None,
        *,
        kind: str | None = None,
        metadata: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        """Initialize a LogicalNode instance with dual parameter naming support.

        Args:
            id (str): Unique identifier for the node.
            op_type (str | _UnsetSentinel | None): Operation type (e.g. 'Conv', 'Relu').
            domain (Mapping[str, AttributeValue] | str): Operator domain or dictionary of attributes for 3-arg legacy signature.
            version (int): Operator set version (default: 1).
            attributes (Optional[Mapping[str, AttributeValue]]): Operator configuration attributes.
            inputs (Optional[List[str]]): Ordered list of upstream LogicalNode IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (ShapeMetadataType | Sequence[DimensionType] | object): Tensor shape metadata.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            dtype (Optional[DType]): Data type of the node.
            output_specs (Optional[List[TensorSpec]]): Explicit output tensor specs.
            subgraphs (Optional[Mapping[str, Union[LogicalGraph, AttributeValue]]]): Nested subgraphs dictionary.
            device (Optional[str]): Target device placement.
            stream (Optional[str]): Target stream identifier.
            kind (Optional[str]): Alternative alias for op_type (deprecated).
            metadata (Optional[Mapping[str, AttributeValue]]): Alternative alias for attributes (deprecated).

        Raises:
            ValueError: If neither op_type nor kind is provided, or if both are provided with conflicting values.
        """
        resolved_op, resolved_dom, resolved_attrs = _normalize_node_init_args(
            op_type, domain, attributes, kind, metadata
        )
        self.id = id
        self.op_type = resolved_op
        self.domain = resolved_dom
        self.version = version
        self.attributes = resolved_attrs
        self.inputs = list(inputs) if inputs is not None else []
        self.outputs = _TrackedOutputsList(outputs if outputs is not None else [id])
        self.shape_metadata = _normalize_shape_metadata(shape_metadata)
        self.source_ast_ref = source_ast_ref
        self.sharding = sharding
        self.dtype = dtype
        self.output_specs = list(output_specs) if output_specs is not None else []
        self.subgraphs = dict(subgraphs) if subgraphs is not None else {}
        self.device = device
        self.stream = stream

    @property
    def kind(self) -> str:
        """Alias for op_type for backward compatibility.

        Returns:
            str: The operation type of this node.
        """
        warnings.warn(
            "The 'kind' property is deprecated and will be removed in version 0.1.0; use 'op_type' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.op_type

    @kind.setter
    def kind(self, value: str) -> None:
        """Set op_type via kind alias.

        Args:
            value (str): The operation type to set.
        """
        warnings.warn(
            "The 'kind' setter is deprecated and will be removed in version 0.1.0; use 'op_type' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.op_type = value

    @property
    def metadata(self) -> dict[str, AttributeValue]:
        """Alias for attributes for backward compatibility.

        Returns:
            dict[str, AttributeValue]: Dictionary of attribute metadata.
        """
        warnings.warn(
            "The 'metadata' property is deprecated and will be removed in version 0.1.0; use 'attributes' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.attributes

    @metadata.setter
    def metadata(self, value: Mapping[str, AttributeValue]) -> None:
        """Set attributes via metadata alias.

        Args:
            value (Mapping[str, AttributeValue]): Dictionary of attribute metadata.
        """
        warnings.warn(
            "The 'metadata' setter is deprecated and will be removed in version 0.1.0; use 'attributes' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.attributes = dict(value)

    @property
    def shape(self) -> TensorShape | None:
        """Return structured TensorShape if shape_metadata is available.

        Returns:
            Optional[TensorShape]: Standardized shape instance if available.
        """
        if self.shape_metadata is None or not isinstance(
            self.shape_metadata, (tuple, list)
        ):
            return None
        return TensorShape(self.shape_metadata)

    @shape.setter
    def shape(self, val: ShapeMetadataType | Sequence[DimensionType] | object) -> None:
        """Set shape metadata via shape property.

        Args:
            val (ShapeMetadataType | Sequence[DimensionType] | object): Shape dimensions or TensorShape instance.
        """
        self.shape_metadata = _normalize_shape_metadata(val)

    @property
    def static_shape(self) -> tuple[int, ...]:
        """Return static shape or raise ValueError if dynamic or missing.

        Returns:
            Tuple[int, ...]: Static shape dimensions.

        Raises:
            ValueError: If node has no shape metadata, is not a sequence, or shape is dynamic.
        """
        if self.shape_metadata is None or not isinstance(
            self.shape_metadata, (tuple, list)
        ):
            raise ValueError(f"Node {self.id} has no shape metadata.")
        return TensorShape(self.shape_metadata).static_shape

    @property
    def is_dynamic_shape(self) -> bool:
        """Check if shape metadata contains dynamic or symbolic dimensions.

        Returns:
            bool: True if dynamic dimensions are present.
        """
        if self.shape_metadata is None or not isinstance(
            self.shape_metadata, (tuple, list)
        ):
            return False
        return any(is_dynamic_dimension(dim) for dim in self.shape_metadata)

    @property
    def rank(self) -> int:
        """Return rank of shape metadata, or 0 if missing.

        Returns:
            int: Tensor rank.
        """
        if self.shape_metadata is None or not isinstance(
            self.shape_metadata, (tuple, list)
        ):
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
        | Sequence[DimensionType]
        | object
        | _UnsetSentinel = _CLONE_UNSET,
        source_ast_ref: str | None | _UnsetSentinel = _CLONE_UNSET,
        sharding: PartitionSpec | None | _UnsetSentinel = _CLONE_UNSET,
        dtype: DType | None | _UnsetSentinel = _CLONE_UNSET,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None = None,
        device: str | None | _UnsetSentinel = _CLONE_UNSET,
        stream: str | None | _UnsetSentinel = _CLONE_UNSET,
        **kwargs: AttributeValue,
    ) -> LogicalNode:
        """Create a deep clone of this LogicalNode with optional keyword overrides.

        Args:
            id (Optional[str]): Override node identifier.
            op_type (Optional[str]): Override operation type.
            domain (Optional[str]): Override operator domain.
            version (Optional[int]): Override operator version.
            attributes (Optional[Mapping[str, AttributeValue]]): Override attributes dictionary.
            inputs (Optional[List[str]]): Override input IDs.
            outputs (Optional[List[str]]): Override output names.
            shape_metadata (ShapeMetadataType | Sequence[DimensionType] | object | _UnsetSentinel): Override tensor shape metadata.
            source_ast_ref (Optional[str | _UnsetSentinel]): Override source AST reference.
            sharding (Optional[PartitionSpec | _UnsetSentinel]): Override partition specification.
            dtype (Optional[DType | _UnsetSentinel]): Override data type.
            output_specs (Optional[List[TensorSpec]]): Override output tensor specifications.
            subgraphs (Optional[Mapping[str, Union[LogicalGraph, AttributeValue]]]): Override nested subgraphs.
            device (Optional[str | _UnsetSentinel]): Override target device placement.
            stream (Optional[str | _UnsetSentinel]): Override asynchronous execution stream.
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

        cls = self.__class__
        return cls(
            id=cloned_id,
            op_type=cloned_op,
            domain=cloned_domain,
            version=cloned_version,
            attributes=cloned_attrs,
            inputs=list(inputs) if inputs is not None else list(self.inputs),
            outputs=list(outputs) if outputs is not None else list(self.outputs),
            shape_metadata=copy.deepcopy(
                shape_metadata
                if shape_metadata is not _CLONE_UNSET
                else self.shape_metadata
            ),
            source_ast_ref=source_ast_ref
            if not isinstance(source_ast_ref, _UnsetSentinel)
            else self.source_ast_ref,
            sharding=copy.deepcopy(
                sharding if not isinstance(sharding, _UnsetSentinel) else self.sharding
            ),
            dtype=dtype if not isinstance(dtype, _UnsetSentinel) else self.dtype,
            output_specs=[
                copy.deepcopy(s)
                for s in (
                    output_specs if output_specs is not None else self.output_specs
                )
            ],
            subgraphs=cloned_subgraphs,
            device=device if not isinstance(device, _UnsetSentinel) else self.device,
            stream=stream if not isinstance(stream, _UnsetSentinel) else self.stream,
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


class NodeDict(dict[str, LogicalNode]):
    """Strictly typed mapping from node identifiers to LogicalNode instances with mutation tracking."""

    version: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize NodeDict and set initial version counter.

        Args:
            *args (Any): Positional arguments forwarded to dict.
            **kwargs (Any): Keyword arguments forwarded to dict.
        """
        self.version = 0
        super().__init__(*args, **kwargs)

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

    def pop(self, key: str, *args: Any) -> Any:
        """Remove and return a node by ID and increment version.

        Args:
            key (str): Node ID.
            *args (Any): Optional default value.

        Returns:
            Any: Removed LogicalNode or default.
        """
        res = super().pop(key, *args)
        self.version += 1
        return res

    def clear(self) -> None:
        """Clear all nodes and increment version."""
        super().clear()
        self.version += 1

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update mapping with new nodes and increment version.

        Args:
            *args (Any): Mapping or iterable of key-value pairs.
            **kwargs (Any): Keyword node entries.
        """
        super().update(*args, **kwargs)
        self.version += 1


class EdgeList(list["LogicalEdge"]):
    """Typed list of LogicalEdge instances."""


def _normalize_graph_nodes(
    nodes: Mapping[str, LogicalNode] | list[LogicalNode] | None,
) -> NodeDict:
    """Normalize graph nodes argument from None, list, or mapping into a NodeDict.

    Args:
        nodes (Optional[Union[Mapping[str, LogicalNode], List[LogicalNode]]]): Input nodes collection.

    Returns:
        NodeDict: Mapping from node IDs to LogicalNode instances.
    """
    if nodes is None:
        return NodeDict()
    if isinstance(nodes, list):
        warnings.warn(
            "Passing a list of nodes to LogicalGraph is deprecated and will be removed in version 0.1.0; provide a dict[str, LogicalNode] instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return NodeDict({node.id: node for node in nodes})
    return NodeDict(nodes)


def _auto_detect_input_pseudo_nodes(
    nodes: dict[str, LogicalNode],
    inputs: list[str],
    input_specs: dict[str, TensorSpec],
) -> None:
    """Populate graph inputs and input_specs from legacy Input pseudo-nodes.

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
        explicit_outputs (Optional[list[str]]): Explicit output list if provided.

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


@dataclass
class LogicalGraph:
    """Represents a computational model or function definition.

    Attributes:
        name (str): Name of the graph or class.
        nodes (Dict[str, LogicalNode]): Mapping of node IDs to LogicalNode instances.
        inputs (List[str]): Ordered list of graph input variable names.
        input_specs (Dict[str, TensorSpec]): Type and shape specifications for graph inputs.
        outputs (List[str]): Ordered list of graph output identifiers or SSA names.
        initializers (Dict[str, InitializerValue]): Constant parameters and weights keyed by tensor name.
        mesh (Optional[LogicalMesh]): Optional logical device mesh for distributed execution.
        pipeline_topology (Optional[PipelineTopologyConfig]): Optional distributed pipeline topology.

    """

    name: str = "Model"
    nodes: dict[str, LogicalNode] = field(default_factory=NodeDict)
    inputs: list[str] = field(default_factory=list)
    input_specs: dict[str, TensorSpec] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    initializers: dict[str, InitializerValue] = field(default_factory=dict)
    mesh: LogicalMesh | None = None
    pipeline_topology: PipelineTopologyConfig | None = None

    @overload
    def __init__(
        self,
        name: str = "Model",
        nodes: Mapping[str, LogicalNode] | None = None,
        outputs: list[str] | None = None,
        mesh: LogicalMesh | None = None,
        inputs: list[str] | None = None,
        input_specs: Mapping[str, TensorSpec] | None = None,
        initializers: Mapping[str, InitializerValue] | None = None,
        *,
        edges: list[LogicalEdge] | None = None,
        pipeline_topology: PipelineTopologyConfig | None = None,
    ) -> None:
        """Initialize LogicalGraph with nodes specified as a dictionary.

        Args:
            name (str): Name of the graph model/class.
            nodes (Optional[Mapping[str, LogicalNode]]): Mapping of node IDs to LogicalNode instances.
            outputs (Optional[List[str]]): List of explicit output IDs.
            mesh (Optional[LogicalMesh]): Optional logical device mesh.
            inputs (Optional[List[str]]): List of graph input variable names.
            input_specs (Optional[Mapping[str, TensorSpec]]): Mapping of input names to TensorSpecs.
            initializers (Optional[Mapping[str, InitializerValue]]): Constant tensor initializers.
            edges (Optional[List[LogicalEdge]]): Optional list of directed edges to populate.
            pipeline_topology (Optional[PipelineTopologyConfig]): Optional distributed pipeline topology.
        """

    @overload
    def __init__(
        self,
        name: str = "Model",
        nodes: list[LogicalNode] | None = None,
        outputs: list[str] | None = None,
        mesh: LogicalMesh | None = None,
        inputs: list[str] | None = None,
        input_specs: Mapping[str, TensorSpec] | None = None,
        initializers: Mapping[str, InitializerValue] | None = None,
        *,
        edges: list[LogicalEdge] | None = None,
        pipeline_topology: PipelineTopologyConfig | None = None,
    ) -> None:
        """Initialize LogicalGraph with nodes specified as a list of LogicalNode instances (deprecated).

        Args:
            name (str): Name of the graph model/class.
            nodes (Optional[List[LogicalNode]]): List of LogicalNode instances (deprecated).
            outputs (Optional[List[str]]): List of explicit output IDs.
            mesh (Optional[LogicalMesh]): Optional logical device mesh.
            inputs (Optional[List[str]]): List of graph input variable names.
            input_specs (Optional[Mapping[str, TensorSpec]]): Mapping of input names to TensorSpecs.
            initializers (Optional[Mapping[str, InitializerValue]]): Constant tensor initializers.
            edges (Optional[List[LogicalEdge]]): Optional list of directed edges to populate.
            pipeline_topology (Optional[PipelineTopologyConfig]): Optional distributed pipeline topology.
        """

    def __init__(
        self,
        name: str = "Model",
        nodes: Mapping[str, LogicalNode] | list[LogicalNode] | None = None,
        outputs: list[str] | None = None,
        mesh: LogicalMesh | None = None,
        inputs: list[str] | None = None,
        input_specs: Mapping[str, TensorSpec] | None = None,
        initializers: Mapping[str, InitializerValue] | None = None,
        *,
        edges: list[LogicalEdge] | None = None,
        pipeline_topology: PipelineTopologyConfig | None = None,
    ) -> None:
        """Initialize a LogicalGraph instance with flexible node and edge specifications.

        Args:
            name (str): Name of the graph model or class (default: 'Model').
            nodes (Optional[Union[Mapping[str, LogicalNode], List[LogicalNode]]]): Mapping of node IDs
                to LogicalNode instances, or a list of LogicalNode instances.
            outputs (Optional[List[str]]): List of explicit output IDs. If omitted, deduced from
                nodes with out-degree zero (leaf nodes).
            mesh (Optional[LogicalMesh]): Optional logical device mesh for distributed execution.
            inputs (Optional[List[str]]): List of graph input variable names.
            input_specs (Optional[Mapping[str, TensorSpec]]): Mapping of input names to TensorSpecs.
            initializers (Optional[Mapping[str, InitializerValue]]): Constant tensor initializers.
            edges (Optional[List[LogicalEdge]]): Optional list of directed edges to populate
                directly into each target node's inputs list.
            pipeline_topology (Optional[PipelineTopologyConfig]): Optional distributed pipeline topology.
        """
        self.name = name
        self.nodes = _normalize_graph_nodes(nodes)
        self._cached_output_map: dict[str, tuple[LogicalNode, int]] | None = None
        self._cached_nodes_version: int = -1
        self._cached_outputs_node: LogicalNode | None = None
        self._cached_outputs_list_ref: list[str] | None = None
        self._cached_outputs_version: int = -1
        self._edge_value_names: dict[tuple[str, int], str] = {}
        self.inputs = list(inputs) if inputs is not None else []
        self.input_specs = dict(input_specs) if input_specs is not None else {}
        self.initializers = dict(initializers) if initializers is not None else {}

        _auto_detect_input_pseudo_nodes(self.nodes, self.inputs, self.input_specs)

        if edges is not None:
            _apply_edges_list(
                self.nodes,
                edges,
                clear_existing=False,
                edge_value_names=self._edge_value_names,
            )

        self.outputs = _infer_graph_outputs(self.nodes, outputs)
        self.mesh = mesh
        self.pipeline_topology = pipeline_topology

    @property
    def nodes_list(self) -> list[LogicalNode]:
        """Return list of LogicalNode instances in dictionary insertion order.

        Time Complexity:
            O(V) where V is the number of nodes in the graph.

        Returns:
            List[LogicalNode]: Ordered list of nodes.
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
            source_idx (Optional[int]): Output port index of the source node (default: 0).
            target_idx (Optional[int]): Input port index on the target node. When specified
                at or beyond ``len(target_node.inputs)``, allows duplicate connections from
                the same source (e.g. ``Add(x, x)``).
            value_name (Optional[str]): Optional SSA value name for the edge.
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
            if not isinstance(node.outputs, _TrackedOutputsList):
                node.outputs = _TrackedOutputsList(node.outputs)
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
            Optional[LogicalNode]: The producing node if found, otherwise None.
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
            List[LogicalNode]: List of upstream input nodes.
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
            List[LogicalNode]: List of downstream consumer nodes.
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

    def __iter__(self) -> Iterator[LogicalNode]:
        """Yield nodes in dictionary insertion order.

        Returns:
            Iterator[LogicalNode]: Node iterator.
        """
        return iter(self.nodes.values())

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

    def to_dict(self, format: str = "canonical") -> dict[str, Any]:
        """Convert LogicalGraph to a serializable dictionary.

        Args:
            format (str): Serialization format ('canonical' or 'legacy').

        Returns:
            Dict[str, Any]: Dictionary representation of the graph.
        """
        data = asdict(self)
        data["nodes"] = {
            nid: _serialize_node_dict(node, format=format)
            for nid, node in self.nodes.items()
        }
        if self.mesh is not None:
            data["mesh"]["webrtc_topology"] = (
                self.mesh.webrtc_topology.to_dict()
                if self.mesh.webrtc_topology is not None
                else None
            )
        data["pipeline_topology"] = (
            self.pipeline_topology.to_dict()
            if self.pipeline_topology is not None
            else None
        )
        if format == "canonical":
            data["edges"] = [asdict(edge) for edge in self.edges]
        return data

    def to_json(self, format: str = "canonical", indent: int = 2) -> str:
        """Serialize the graph to a deterministic JSON string.

        Args:
            format (str): Serialization format ('canonical' or 'legacy').
            indent (int): Indentation spaces for JSON formatting.

        Returns:
            str: Deterministic JSON string representation.
        """
        buf = io.StringIO()
        self.to_stream(buf, format=format, indent=indent)
        return buf.getvalue()

    def to_stream(
        self,
        fp: IO[Any] | TextIO,
        format: str = "canonical",
        indent: int = 2,
    ) -> None:
        """Stream the serialized graph directly into a file-like object.

        Avoids allocating a single massive intermediate JSON string or full-graph
        dictionary in memory by streaming edges and nodes entry-by-entry.

        Args:
            fp (Union[IO[Any], TextIO]): Writable file-like stream.
            format (str): Serialization format ('canonical' or 'legacy').
            indent (int): Indentation spaces for JSON formatting.
        """
        _stream_write_graph(self, fp, format=format, indent=indent)

    def to_file(
        self,
        path: str | Path,
        format: str = "canonical",
        indent: int = 2,
        compression: str | None = None,
    ) -> None:
        """Serialize graph directly to a file, supporting optional .gz or .zst compression.

        Args:
            path (str | Path): Destination file path.
            format (str): Serialization format ('canonical' or 'legacy').
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
                self.to_stream(f, format=format, indent=indent)
        elif comp in ("zstd", "zst") or (
            comp is None and path_str.endswith((".zst", ".zstd"))
        ):
            if zstandard is None:
                raise ImportError("zstandard library is required for .zst compression.")
            cctx = zstandard.ZstdCompressor()
            with open(path, "wb") as raw_f, cctx.stream_writer(raw_f) as compressor:
                text_wrapper = io.TextIOWrapper(compressor, encoding="utf-8")
                self.to_stream(text_wrapper, format=format, indent=indent)
                text_wrapper.flush()
        elif comp is None or comp in ("none", "identity"):
            with open(path, "w", encoding="utf-8") as f:
                self.to_stream(f, format=format, indent=indent)
        else:
            raise ValueError(f"Unsupported compression format: {compression}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LogicalGraph:
        """Deserialize a graph from a dictionary representation.

        Args:
            data (Mapping[str, Any]): Dictionary containing graph fields.

        Returns:
            LogicalGraph: Deserialized graph instance.
        """
        raw_nodes = data.get("nodes", {})
        nodes_map = (
            {n["id"]: n for n in raw_nodes}
            if isinstance(raw_nodes, list)
            else raw_nodes
        )
        nodes = NodeDict(
            {nid: _parse_node_dict(ndata, cls) for nid, ndata in nodes_map.items()}
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
        json_str: str | bytes | IO[Any] | TextIO | BinaryIO | Path,
    ) -> LogicalGraph:
        """Deserialize a graph from a JSON string, stream, or file path.

        Supports chunked incremental streaming deserialization from file-like objects
        without allocating intermediate full-graph string representations.

        Args:
            json_str (Union[str, bytes, IO[Any], TextIO, BinaryIO, Path]): JSON string, bytes,
                file-like stream with read() method, or filesystem path.

        Returns:
            LogicalGraph: Deserialized graph instance.

        Raises:
            TypeError: If json_str is of an unsupported type.
        """
        if isinstance(json_str, Path):
            return cls.from_file(json_str)
        if hasattr(json_str, "read"):
            return _stream_parse_graph(cast("IO[Any]", json_str), cls)
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
        List[LogicalNode]: List of nodes in execution order.

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


class CompilerBackend(ABC):
    """Abstract base class for compilation backends."""

    @abstractmethod
    def compile(self, graph: LogicalGraph) -> object:
        """Compile the Logical Intermediate Representation (IR) into a target artifact.

        Args:
            graph (LogicalGraph): The intermediate representation of the model structure.

        Returns:
            Any: The compiled output (e.g., source code string, binary buffer, or AST).

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


def __getattr__(name: str) -> Any:
    """Lazily resolve ZeroTangent and NoTangent from ml_switcheroo_ir.sentinels.

    Args:
        name (str): Attribute name to retrieve.

    Returns:
        Any: Exported sentinel class.

    Raises:
        AttributeError: If attribute is not found.
    """
    if name in ("ZeroTangent", "NoTangent"):
        from ml_switcheroo_ir import sentinels

        return getattr(sentinels, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
