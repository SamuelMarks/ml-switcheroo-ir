"""Core graph intermediate representation dataclasses and serialization."""

from __future__ import annotations

import copy
import gzip
import io
import json
import os
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, BinaryIO, Iterator, Mapping, Sequence, TextIO, overload

from ml_switcheroo_ir.distributed import (
    PipelineTopologyConfig,
    WebRTCSignalingTopology,
)
from ml_switcheroo_ir.types import (
    AttributeValue,
    DType,
    TensorShape,
    TensorSpec,
)


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
    "PartitionSpec",
    "ZeroTangent",
    "_load_zstandard",
    "topological_sort",
    "zstandard",
]


@dataclass
class LogicalAxis:
    """Represents a named dimension for tensor sizes and sharding (e.g., 'batch', 'embed', 'heads').

    Attributes:
        name (str): Name of the logical axis.
        size (Optional[int]): Optional fixed size of the axis.

    """

    name: str
    size: int | None = None


@dataclass
class PartitionSpec:
    """Describes how a tensor's dimensions are mapped to a logical mesh.

    Each element in `axes` corresponds to a tensor dimension. An element can be:
    - A string representing the mesh axis name (e.g., 'data').
    - A tuple of strings for multi-axis sharding (e.g., ('data', 'model')).
    - None for a replicated/unsharded dimension.

    Attributes:
        axes (Tuple[Optional[Union[str, Tuple[str, ...]]], ...]): Tuple mapping tensor dimensions to mesh axes.

    """

    axes: tuple[str | tuple[str, ...] | None, ...]


@dataclass
class LogicalMesh:
    """Represents a multi-dimensional grid of devices for distributed execution.

    Attributes:
        shape (Dict[str, int]): Mapping of mesh axis names to their sizes (e.g., {'data': 4, 'model': 2}).
        webrtc_topology (Optional[WebRTCSignalingTopology]): WebRTC cluster mesh signaling configuration.

    """

    shape: dict[str, int]
    webrtc_topology: WebRTCSignalingTopology | None = None


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


_UNSET_OP_TYPE: Any = object()
_CLONE_UNSET: Any = object()


def _normalize_shape_metadata(
    shape_metadata: tuple[int | str, ...] | Sequence[int | str] | Any,
) -> Any:
    """Normalize arbitrary shape_metadata input into a canonical tuple or object.

    Args:
        shape_metadata (Any): Raw shape metadata argument.

    Returns:
        Any: Normalized tuple of dimensions or preserved custom metadata object.
    """
    if shape_metadata is None:
        return None
    if isinstance(shape_metadata, tuple):
        return shape_metadata
    if isinstance(shape_metadata, list):
        return tuple(shape_metadata)
    if hasattr(shape_metadata, "shape"):
        return _normalize_shape_metadata(shape_metadata.shape)
    if not isinstance(shape_metadata, (str, bytes, dict)):
        try:
            return tuple(shape_metadata)
        except TypeError:
            return shape_metadata
    return shape_metadata


def _normalize_node_init_args(
    op_type: Any,
    domain: Any,
    attributes: Mapping[str, Any] | None,
    kind: str | None,
    metadata: Mapping[str, Any] | None,
) -> tuple[str, str, dict[str, AttributeValue]]:
    """Normalize canonical and legacy LogicalNode constructor arguments.

    Args:
        op_type (Any): Canonical operation type or sentinel.
        domain (Any): Domain identifier string or legacy attribute dictionary.
        attributes (Optional[Mapping[str, Any]]): Canonical attributes dictionary.
        kind (Optional[str]): Deprecated alias for op_type.
        metadata (Optional[Mapping[str, Any]]): Deprecated alias for attributes.

    Returns:
        tuple[str, str, dict[str, AttributeValue]]: Resolved (op_type, domain, attributes).

    Raises:
        ValueError: If neither op_type nor kind is provided, or if both conflict.
    """
    if kind is not None:
        warnings.warn(
            "The 'kind' parameter is deprecated; use 'op_type' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    if metadata is not None:
        warnings.warn(
            "The 'metadata' parameter is deprecated; use 'attributes' instead.",
            DeprecationWarning,
            stacklevel=3,
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
    if isinstance(domain, dict):
        resolved_attributes.update(domain)
    else:
        resolved_domain = str(domain)

    if metadata is not None:
        resolved_attributes.update(metadata)
    if attributes is not None:
        resolved_attributes.update(attributes)

    return resolved_op_type, resolved_domain, resolved_attributes


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
        shape_metadata (Optional[Tuple[Union[int, str], ...]]): Tuple of integers or string symbols ("B", "T").
        source_ast_ref (Optional[str]): Traceback to exact file path, line number, and cdd-python AST node ID.
        sharding (Optional[PartitionSpec]): Optional layout specification for distributed placement of this node's output.
        dtype (Optional[DType]): Data type of the node computation or primary output.
        output_specs (List[TensorSpec]): Explicit tensor specifications for all node outputs.
        subgraphs (Dict[str, Any]): Nested subgraphs for control flow, autodiff, and checkpointing.
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
    shape_metadata: tuple[int | str, ...] | Sequence[int | str] | Any = None
    source_ast_ref: str | None = None
    sharding: PartitionSpec | None = None
    dtype: DType | None = None
    output_specs: list[TensorSpec] = field(default_factory=list)
    subgraphs: dict[str, Any] = field(default_factory=dict)
    device: str | None = None
    stream: str | None = None

    @overload
    def __init__(
        self,
        id: str,
        op_type: str = ...,
        domain: Mapping[str, Any] | str = "ai.onnx",
        version: int = 1,
        attributes: Mapping[str, Any] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: tuple[int | str, ...] | Sequence[int | str] | Any = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        dtype: DType | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: dict[str, Any] | None = None,
        device: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Initialize LogicalNode using canonical op_type and attributes.

        Args:
            id (str): Unique identifier for the node.
            op_type (str): Operation type.
            domain (Union[Mapping[str, Any], str]): Operator domain or attributes mapping.
            version (int): Operator version (default: 1).
            attributes (Optional[Mapping[str, Any]]): Operator attributes.
            inputs (Optional[List[str]]): Ordered list of upstream input IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (Optional[Union[Tuple[Union[int, str], ...], Sequence[Union[int, str]], Any]]): Tensor
                shape metadata, accepting standard integer/string dimension tuples/lists, duck-typed metadata
                objects defining a `.shape` attribute, or arbitrary custom non-iterable objects.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            dtype (Optional[DType]): Data type of the node.
            output_specs (Optional[List[TensorSpec]]): Explicit output tensor specs.
            subgraphs (Optional[Dict[str, Any]]): Nested subgraphs dictionary.
            device (Optional[str]): Target device placement.
            stream (Optional[str]): Target stream identifier.
        """

    @overload
    def __init__(
        self,
        id: str,
        *,
        kind: str,
        domain: Mapping[str, Any] | str = "ai.onnx",
        version: int = 1,
        metadata: Mapping[str, Any] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: tuple[int | str, ...] | Sequence[int | str] | Any = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        dtype: DType | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: dict[str, Any] | None = None,
        device: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Initialize LogicalNode using legacy/alternative kind and metadata.

        Args:
            id (str): Unique identifier for the node.
            kind (str): Alternative alias for op_type (deprecated).
            domain (Union[Mapping[str, Any], str]): Operator domain or attributes mapping.
            version (int): Operator version (default: 1).
            metadata (Optional[Mapping[str, Any]]): Alternative alias for attributes (deprecated).
            inputs (Optional[List[str]]): Ordered list of upstream input IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (Optional[Union[Tuple[Union[int, str], ...], Sequence[Union[int, str]], Any]]): Tensor
                shape metadata, accepting standard integer/string dimension tuples/lists, duck-typed metadata
                objects defining a `.shape` attribute, or arbitrary custom non-iterable objects.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            dtype (Optional[DType]): Data type of the node.
            output_specs (Optional[List[TensorSpec]]): Explicit output tensor specs.
            subgraphs (Optional[Dict[str, Any]]): Nested subgraphs dictionary.
            device (Optional[str]): Target device placement.
            stream (Optional[str]): Target stream identifier.
        """

    @overload
    def __init__(
        self,
        id: str,
        op_type: str | None = None,
        domain: Mapping[str, Any] | str = "ai.onnx",
        version: int = 1,
        attributes: Mapping[str, Any] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: tuple[int | str, ...] | Sequence[int | str] | Any = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        dtype: DType | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: dict[str, Any] | None = None,
        device: str | None = None,
        stream: str | None = None,
        *,
        kind: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize LogicalNode with flexible parameter naming conventions.

        Args:
            id (str): Unique identifier for the node.
            op_type (Optional[str]): Operation type.
            domain (Union[Mapping[str, Any], str]): Operator domain or attributes mapping.
            version (int): Operator version (default: 1).
            attributes (Optional[Mapping[str, Any]]): Operator attributes.
            inputs (Optional[List[str]]): Ordered list of upstream input IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (Optional[Union[Tuple[Union[int, str], ...], Sequence[Union[int, str]], Any]]): Tensor
                shape metadata, accepting standard integer/string dimension tuples/lists, duck-typed metadata
                objects defining a `.shape` attribute, or arbitrary custom non-iterable objects.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            dtype (Optional[DType]): Data type of the node.
            output_specs (Optional[List[TensorSpec]]): Explicit output tensor specs.
            subgraphs (Optional[Dict[str, Any]]): Nested subgraphs dictionary.
            device (Optional[str]): Target device placement.
            stream (Optional[str]): Target stream identifier.
            kind (Optional[str]): Alternative alias for op_type (deprecated).
            metadata (Optional[Mapping[str, Any]]): Alternative alias for attributes (deprecated).
        """

    def __init__(
        self,
        id: str,
        op_type: Any = _UNSET_OP_TYPE,
        domain: Any = "ai.onnx",
        version: int = 1,
        attributes: Mapping[str, Any] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: tuple[int | str, ...] | Sequence[int | str] | Any = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        dtype: DType | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: dict[str, Any] | None = None,
        device: str | None = None,
        stream: str | None = None,
        *,
        kind: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a LogicalNode instance with dual parameter naming support.

        Args:
            id (str): Unique identifier for the node.
            op_type (Any): Operation type (e.g. 'Conv', 'Relu').
            domain (Any): Operator domain or dictionary of attributes for 3-arg legacy signature.
            version (int): Operator set version (default: 1).
            attributes (Optional[Dict[str, AttributeValue]]): Operator configuration attributes.
            inputs (Optional[List[str]]): Ordered list of upstream LogicalNode IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (Optional[Union[Tuple[Union[int, str], ...], Sequence[Union[int, str]], Any]]): Tensor
                shape metadata, accepting standard integer/string dimension tuples/lists, duck-typed metadata
                objects defining a `.shape` attribute, or arbitrary custom non-iterable objects.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            dtype (Optional[DType]): Data type of the node.
            output_specs (Optional[List[TensorSpec]]): Explicit output tensor specs.
            subgraphs (Optional[Dict[str, Any]]): Nested subgraphs dictionary.
            device (Optional[str]): Target device placement.
            stream (Optional[str]): Target stream identifier.
            kind (Optional[str]): Alternative alias for op_type (deprecated).
            metadata (Optional[Dict[str, AttributeValue]]): Alternative alias for attributes (deprecated).

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
        self.outputs = (
            list(outputs) if outputs is not None and len(outputs) > 0 else [id]
        )
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
    def metadata(self) -> dict[str, Any]:
        """Alias for attributes for backward compatibility.

        Returns:
            dict[str, Any]: Dictionary of attribute metadata.
        """
        warnings.warn(
            "The 'metadata' property is deprecated and will be removed in version 0.1.0; use 'attributes' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.attributes

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        """Set attributes via metadata alias.

        Args:
            value (dict[str, Any]): Dictionary of attribute metadata.
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
    def shape(self, val: Any) -> None:
        """Set shape metadata via shape property.

        Args:
            val (Any): Shape dimensions or TensorShape instance.
        """
        if isinstance(val, TensorShape):
            self.shape_metadata = val.dims
        elif isinstance(val, (tuple, list)):
            self.shape_metadata = tuple(val)
        else:
            self.shape_metadata = val

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
        return any(
            isinstance(dim, str)
            or (isinstance(dim, int) and dim < 0)
            or not isinstance(dim, int)
            for dim in self.shape_metadata
        )

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
        attributes: dict[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: tuple[int | str, ...]
        | Sequence[int | str]
        | Any = _CLONE_UNSET,
        source_ast_ref: str | None = _CLONE_UNSET,
        sharding: PartitionSpec | None = _CLONE_UNSET,
        dtype: DType | None = _CLONE_UNSET,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: dict[str, Any] | None = None,
        device: str | None = _CLONE_UNSET,
        stream: str | None = _CLONE_UNSET,
        **kwargs: Any,
    ) -> LogicalNode:
        """Create a deep clone of this LogicalNode with optional keyword overrides.

        Args:
            id (Optional[str]): Override node identifier.
            op_type (Optional[str]): Override operation type.
            domain (Optional[str]): Override operator domain.
            version (Optional[int]): Override operator version.
            attributes (Optional[Dict[str, AttributeValue]]): Override attributes dictionary.
            inputs (Optional[List[str]]): Override input IDs.
            outputs (Optional[List[str]]): Override output names.
            shape_metadata (Optional[Union[Tuple[Union[int, str], ...], Sequence[Union[int, str]], Any]]):
                Override tensor shape metadata.
            source_ast_ref (Optional[str]): Override source AST reference.
            sharding (Optional[PartitionSpec]): Override partition specification.
            dtype (Optional[DType]): Override data type.
            output_specs (Optional[List[TensorSpec]]): Override output tensor specifications.
            subgraphs (Optional[Dict[str, Any]]): Override nested subgraphs dictionary.
            device (Optional[str]): Override target device placement.
            stream (Optional[str]): Override asynchronous execution stream.
            **kwargs (Any): Additional attribute overrides.

        Returns:
            LogicalNode: A cloned LogicalNode with deep-copied collections and applied overrides.
        """
        cloned_id = id if id is not None else self.id
        cloned_op = op_type if op_type is not None else self.op_type
        cloned_domain = domain if domain is not None else self.domain
        cloned_version = version if version is not None else self.version
        cloned_attrs = copy.deepcopy(
            attributes if attributes is not None else self.attributes
        )
        if kwargs:
            cloned_attrs.update(kwargs)

        raw_subgraphs = subgraphs if subgraphs is not None else self.subgraphs
        cloned_subgraphs = {
            k: v.clone() if hasattr(v, "clone") else copy.deepcopy(v)
            for k, v in raw_subgraphs.items()
        }

        cls = self.__class__ if self.__class__ is not LogicalNode else LogicalNode
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
            if source_ast_ref is not _CLONE_UNSET
            else self.source_ast_ref,
            sharding=copy.deepcopy(
                sharding if sharding is not _CLONE_UNSET else self.sharding
            ),
            dtype=dtype if dtype is not _CLONE_UNSET else self.dtype,
            output_specs=[
                copy.deepcopy(s)
                for s in (
                    output_specs if output_specs is not None else self.output_specs
                )
            ],
            subgraphs=cloned_subgraphs,
            device=device if device is not _CLONE_UNSET else self.device,
            stream=stream if stream is not _CLONE_UNSET else self.stream,
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


@dataclass
class ZeroTangent(LogicalNode):
    """Represents a mathematically zero tangent vector with explicit shape and dtype.

    Attributes:
        id (str): Unique node identifier.
        shape_metadata (Optional[Union[Tuple[Union[int, str], ...], Sequence[Union[int, str]], Any]]):
            Shape of the zero tangent vector.
        dtype (Optional[DType]): Data type of the zero tangent vector.
    """

    def __init__(
        self,
        id: str,
        shape: tuple[int | str, ...] | Sequence[int | str] | Any = None,
        dtype: DType | None = DType.float32,
        **kwargs: Any,
    ) -> None:
        """Initialize ZeroTangent sentinel node.

        Args:
            id (str): Unique node identifier.
            shape (Optional[Union[Tuple[Union[int, str], ...], Sequence[Union[int, str]], Any]]): Vector shape.
            dtype (Optional[DType]): Tensor data type.
            **kwargs (Any): Additional node keyword arguments.
        """
        kwargs.pop("op_type", None)
        kwargs.pop("domain", None)
        shape_val = kwargs.pop("shape_metadata", shape)
        dtype_val = kwargs.pop("dtype", dtype)
        super().__init__(
            id=id,
            op_type="ZeroTangent",
            domain="ml.switcheroo.ad",
            shape_metadata=shape_val,
            dtype=dtype_val,
            **kwargs,
        )


@dataclass
class NoTangent(LogicalNode):
    """Represents a non-differentiable or structurally missing cotangent path.

    Attributes:
        id (str): Unique node identifier.
    """

    def __init__(
        self,
        id: str,
        **kwargs: Any,
    ) -> None:
        """Initialize NoTangent sentinel node.

        Args:
            id (str): Unique node identifier.
            **kwargs (Any): Additional node keyword arguments.
        """
        kwargs.pop("op_type", None)
        kwargs.pop("domain", None)
        super().__init__(
            id=id,
            op_type="NoTangent",
            domain="ml.switcheroo.ad",
            **kwargs,
        )


class EdgeList(list["LogicalEdge"]):
    """List of LogicalEdge instances without magic graph synchronization."""


def _normalize_graph_nodes(
    nodes: dict[str, Any] | list[LogicalNode] | None,
) -> dict[str, Any]:
    """Normalize graph nodes argument from None, list, or dict into a node dictionary.

    Args:
        nodes (Optional[Union[Dict[str, Any], List[LogicalNode]]]): Input nodes collection.

    Returns:
        dict[str, Any]: Mapping from node IDs to LogicalNode instances.
    """
    if nodes is None:
        return {}
    if isinstance(nodes, list):
        warnings.warn(
            "Passing a list of nodes to LogicalGraph is deprecated and will be removed in version 0.1.0; provide a dict[str, LogicalNode] instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return {node.id: node for node in nodes}
    return dict(nodes)


def _auto_detect_input_pseudo_nodes(
    nodes: dict[str, Any],
    inputs: list[str],
    input_specs: dict[str, TensorSpec],
) -> None:
    """Populate graph inputs and input_specs from legacy Input pseudo-nodes.

    Args:
        nodes (dict[str, Any]): Graph node dictionary.
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
    nodes: dict[str, Any],
    explicit_outputs: list[str] | None,
) -> list[str]:
    """Resolve explicit graph outputs or infer leaf/Output nodes when omitted.

    Args:
        nodes (dict[str, Any]): Graph node dictionary.
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
    return LogicalNode(**ndata_copy)


def _parse_subgraphs_from_node_dict(
    ndata_copy: dict[str, Any],
    graph_cls: type[LogicalGraph],
) -> dict[str, Any]:
    """Extract and deserialize nested subgraphs from node dictionary and legacy attributes.

    Args:
        ndata_copy (dict[str, Any]): Mutable node dictionary.
        graph_cls (type[LogicalGraph]): LogicalGraph class for recursive deserialization.

    Returns:
        dict[str, Any]: Mapping of subgraph names to LogicalGraph instances.
    """
    resolved_subgraphs: dict[str, Any] = {}
    subgraphs_data = ndata_copy.get("subgraphs")
    if subgraphs_data and isinstance(subgraphs_data, dict):
        for sub_k, sub_v in subgraphs_data.items():
            if isinstance(sub_v, dict):
                resolved_subgraphs[sub_k] = graph_cls.from_dict(sub_v)
            elif isinstance(sub_v, LogicalGraph):
                resolved_subgraphs[sub_k] = sub_v

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
                elif isinstance(val, LogicalGraph):
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


@dataclass
class LogicalGraph:
    """Represents a computational model or function definition.

    Attributes:
        name (str): Name of the graph or class.
        nodes (Dict[str, LogicalNode]): Mapping of node IDs to LogicalNode instances.
        inputs (List[str]): Ordered list of graph input variable names.
        input_specs (Dict[str, TensorSpec]): Type and shape specifications for graph inputs.
        outputs (List[str]): Ordered list of graph output identifiers or SSA names.
        initializers (Dict[str, Any]): Constant parameters and weights keyed by tensor name.
        mesh (Optional[LogicalMesh]): Optional logical device mesh for distributed execution.

    """

    name: str = "Model"
    nodes: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    input_specs: dict[str, TensorSpec] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    initializers: dict[str, Any] = field(default_factory=dict)
    mesh: LogicalMesh | None = None
    pipeline_topology: PipelineTopologyConfig | None = None

    @overload
    def __init__(
        self,
        name: str = "Model",
        nodes: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
        mesh: LogicalMesh | None = None,
        inputs: list[str] | None = None,
        input_specs: dict[str, TensorSpec] | None = None,
        initializers: dict[str, Any] | None = None,
        *,
        edges: list[LogicalEdge] | None = None,
        pipeline_topology: PipelineTopologyConfig | None = None,
    ) -> None:
        """Initialize LogicalGraph with nodes specified as a dictionary.

        Args:
            name (str): Name of the graph model/class.
            nodes (Optional[Dict[str, LogicalNode]]): Mapping of node IDs to LogicalNode instances.
            outputs (Optional[List[str]]): List of explicit output IDs.
            mesh (Optional[LogicalMesh]): Optional logical device mesh.
            inputs (Optional[List[str]]): List of graph input variable names.
            input_specs (Optional[Dict[str, TensorSpec]]): Mapping of input names to TensorSpecs.
            initializers (Optional[Dict[str, Any]]): Constant tensor initializers.
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
        input_specs: dict[str, TensorSpec] | None = None,
        initializers: dict[str, Any] | None = None,
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
            input_specs (Optional[Dict[str, TensorSpec]]): Mapping of input names to TensorSpecs.
            initializers (Optional[Dict[str, Any]]): Constant tensor initializers.
            edges (Optional[List[LogicalEdge]]): Optional list of directed edges to populate.
            pipeline_topology (Optional[PipelineTopologyConfig]): Optional distributed pipeline topology.
        """

    def __init__(
        self,
        name: str = "Model",
        nodes: dict[str, Any] | list[LogicalNode] | None = None,
        outputs: list[str] | None = None,
        mesh: LogicalMesh | None = None,
        inputs: list[str] | None = None,
        input_specs: dict[str, TensorSpec] | None = None,
        initializers: dict[str, Any] | None = None,
        *,
        edges: list[LogicalEdge] | None = None,
        pipeline_topology: PipelineTopologyConfig | None = None,
    ) -> None:
        """Initialize a LogicalGraph instance with flexible node and edge specifications.

        Args:
            name (str): Name of the graph model or class (default: 'Model').
            nodes (Optional[Union[Dict[str, LogicalNode], List[LogicalNode]]]): Mapping of node IDs
                to LogicalNode instances, or a list of LogicalNode instances.
            outputs (Optional[List[str]]): List of explicit output IDs. If omitted, deduced from
                nodes with out-degree zero (leaf nodes).
            mesh (Optional[LogicalMesh]): Optional logical device mesh for distributed execution.
            inputs (Optional[List[str]]): List of graph input variable names.
            input_specs (Optional[Dict[str, TensorSpec]]): Mapping of input names to TensorSpecs.
            initializers (Optional[Dict[str, Any]]): Constant tensor initializers.
            edges (Optional[List[LogicalEdge]]): Optional list of directed edges to populate
                directly into each target node's inputs list.
            pipeline_topology (Optional[PipelineTopologyConfig]): Optional distributed pipeline topology.
        """
        self.name = name
        self.nodes = _normalize_graph_nodes(nodes)
        self.inputs = list(inputs) if inputs is not None else []
        self.input_specs = dict(input_specs) if input_specs is not None else {}
        self.initializers = dict(initializers) if initializers is not None else {}

        _auto_detect_input_pseudo_nodes(self.nodes, self.inputs, self.input_specs)

        if edges is not None:
            for edge in edges:
                self.add_edge(edge.source, edge.target)

        self.outputs = _infer_graph_outputs(self.nodes, outputs)
        self.mesh = mesh
        self.pipeline_topology = pipeline_topology

    @property
    def nodes_list(self) -> list[LogicalNode]:
        """Return list of LogicalNode instances in dictionary insertion order.

        Returns:
            List[LogicalNode]: Ordered list of nodes.
        """
        return list(self.nodes.values())

    def get_producing_output_index(
        self, output_name: str
    ) -> tuple[LogicalNode, int] | None:
        """Resolve both the producing node and the specific output port index.

        Args:
            output_name (str): SSA output tensor identifier or 'node_id:idx'.

        Returns:
            tuple[LogicalNode, int] | None: Producing node and 0-based output index, or None.
        """
        return self._output_map.get(output_name)

    @property
    def edges(self) -> list[LogicalEdge]:
        """Derive and return directed edges dynamically from each node's inputs.

        Returns:
            list[LogicalEdge]: List of directed edges from upstream inputs to target nodes.
        """
        result: list[LogicalEdge] = []
        out_map = self._output_map
        for target_node in self.nodes.values():
            for target_idx, src in enumerate(target_node.inputs):
                prod_info = out_map.get(src)
                src_idx = prod_info[1] if prod_info is not None else 0
                result.append(
                    LogicalEdge(
                        source=src,
                        target=target_node.id,
                        source_idx=src_idx,
                        target_idx=target_idx,
                    )
                )
        return result

    @edges.setter
    def edges(self, edge_list: list[LogicalEdge]) -> None:
        """Synchronize edges into target nodes' inputs.

        Args:
            edge_list (list[LogicalEdge]): List of edges to configure.
        """
        for node in self.nodes.values():
            node.inputs.clear()
        for edge in edge_list:
            if edge.target in self.nodes:
                target_node = self.nodes[edge.target]
                src_name = edge.source
                if edge.source in self.nodes and edge.source_idx is not None:
                    src_name = self.nodes[edge.source].get_output_name(edge.source_idx)
                if src_name not in target_node.inputs:
                    target_node.inputs.append(src_name)

    def set_pipeline_topology(self, topology: PipelineTopologyConfig) -> None:
        """Set distributed pipeline topology configuration for this graph.

        Args:
            topology (PipelineTopologyConfig): Distributed pipeline topology configuration.
        """
        self.pipeline_topology = topology

    def add_node(self, node: LogicalNode) -> None:
        """Add or update a node in the graph.

        Args:
            node (LogicalNode): The node to add or update.
        """
        self.nodes[node.id] = node

    def add_edge(self, source: str, target: str) -> None:
        """Connect source node to target node, updating target inputs.

        Args:
            source (str): Identifier of the upstream source node.
            target (str): Identifier of the downstream target node.
        """
        if target in self.nodes:
            target_node = self.nodes[target]
            if source not in target_node.inputs:
                target_node.inputs.append(source)

    @property
    def _output_map(self) -> dict[str, tuple[LogicalNode, int]]:
        """O(1) mapping of SSA names to (producer_node, output_index)."""
        mapping: dict[str, tuple[LogicalNode, int]] = {}
        for node in self.nodes.values():
            for idx, out_name in enumerate(node.outputs):
                mapping[out_name] = (node, idx)
        return mapping

    def get_output_producer(self, output_name: str) -> LogicalNode | None:
        """Retrieve the LogicalNode that produces the specified output name.

        Args:
            output_name (str): Identifier or SSA name of the output.

        Returns:
            Optional[LogicalNode]: The producing node if found, otherwise None.
        """
        res = self._output_map.get(output_name)
        if res:
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
        cloned_topology = getattr(self, "pipeline_topology", None)
        if cloned_topology is not None:
            cloned_topology = copy.deepcopy(cloned_topology)

        new_g = LogicalGraph(
            name=self.name,
            nodes=cloned_nodes,
            inputs=list(self.inputs),
            input_specs=cloned_specs,
            outputs=list(self.outputs),
            mesh=cloned_mesh,
            initializers=cloned_inits,
        )
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
            if any(inp in output_ids for inp in node.inputs)
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
        from typing import cast

        return cast("LogicalNode", self.nodes[node_id])

    def to_dict(self, format: str = "canonical") -> dict[str, Any]:
        """Convert LogicalGraph to a serializable dictionary.

        Args:
            format (str): Serialization format ('canonical' or 'legacy').

        Returns:
            Dict[str, Any]: Dictionary representation of the graph.
        """
        data = asdict(self)
        if format == "canonical":
            data["edges"] = [asdict(edge) for edge in self.edges]
        if self.pipeline_topology is not None:
            data["pipeline_topology"] = self.pipeline_topology.to_dict()
        if self.mesh is not None and self.mesh.webrtc_topology is not None:
            data["mesh"]["webrtc_topology"] = self.mesh.webrtc_topology.to_dict()
        return data

    def to_json(self, format: str = "canonical", indent: int = 2) -> str:
        """Serialize the graph to a deterministic JSON string.

        Args:
            format (str): Serialization format ('canonical' or 'legacy').
            indent (int): Indentation spaces for JSON formatting.

        Returns:
            str: Deterministic JSON string representation.
        """
        data = self.to_dict(format=format)
        return json.dumps(data, sort_keys=True, indent=indent, default=str)

    def to_stream(
        self,
        fp: IO[Any] | TextIO,
        format: str = "canonical",
        indent: int = 2,
    ) -> None:
        """Stream the serialized graph directly into a file-like object.

        Avoids allocating a single massive intermediate JSON string in memory.

        Args:
            fp (Union[IO[Any], TextIO]): Writable file-like stream.
            format (str): Serialization format ('canonical' or 'legacy').
            indent (int): Indentation spaces for JSON formatting.
        """
        data = self.to_dict(format=format)
        json.dump(data, fp, sort_keys=True, indent=indent, default=str)

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
        import ml_switcheroo_ir

        zstd_mod = getattr(ml_switcheroo_ir, "zstandard", zstandard)
        path_str = str(path)
        comp = compression.lower() if compression else None
        if comp in ("gzip", "gz") or (comp is None and path_str.endswith(".gz")):
            with gzip.open(path, "wt", encoding="utf-8") as f:
                self.to_stream(f, format=format, indent=indent)
        elif comp in ("zstd", "zst") or (
            comp is None and path_str.endswith((".zst", ".zstd"))
        ):
            if zstd_mod is None:
                raise ImportError("zstandard library is required for .zst compression.")
            cctx = zstd_mod.ZstdCompressor()
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
    def from_dict(cls, data: dict[str, Any]) -> LogicalGraph:
        """Deserialize a graph from a dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary containing graph fields.

        Returns:
            LogicalGraph: Deserialized graph instance.
        """
        raw_nodes = data.get("nodes", {})
        nodes_map = (
            {n["id"]: n for n in raw_nodes}
            if isinstance(raw_nodes, list)
            else raw_nodes
        )
        nodes: dict[str, Any] = {
            nid: _parse_node_dict(ndata, cls) for nid, ndata in nodes_map.items()
        }

        edges_data = data.get("edges")
        if edges_data and isinstance(edges_data, list):
            for edge_item in edges_data:
                src = edge_item.get("source")
                tgt = edge_item.get("target")
                if tgt and tgt in nodes and src and src not in nodes[tgt].inputs:
                    nodes[tgt].inputs.append(src)

        pipe_topo_raw = data.get("pipeline_topology")
        pipe_topo = (
            PipelineTopologyConfig.from_dict(pipe_topo_raw)
            if pipe_topo_raw is not None
            else None
        )
        inputs_data = data.get("inputs")
        graph = cls(
            name=data.get("name", "Model"),
            nodes=nodes,
            inputs=list(inputs_data) if inputs_data is not None else [],
            input_specs=_parse_input_specs_dict(data.get("input_specs")),
            outputs=data.get("outputs"),
            initializers=data.get("initializers", {}),
            mesh=_parse_mesh_dict(data.get("mesh")),
        )
        if pipe_topo is not None:
            graph.pipeline_topology = pipe_topo
        return graph

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        compression: str | None = None,
    ) -> LogicalGraph:
        """Deserialize graph directly from a file or compressed archive.

        Supports streaming deserialization from uncompressed JSON, gzip (.gz),
        and zstandard (.zst) formats without intermediate string allocations.

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
        import ml_switcheroo_ir

        zstd_mod = getattr(ml_switcheroo_ir, "zstandard", zstandard)
        path_str = str(path)
        comp = compression.lower() if compression else None
        if comp in ("gzip", "gz") or (comp is None and path_str.endswith(".gz")):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return cls.from_json(f)
        elif comp in ("zstd", "zst") or (
            comp is None and path_str.endswith((".zst", ".zstd"))
        ):
            if zstd_mod is None:
                raise ImportError(
                    "zstandard library is required for .zst decompression."
                )
            dctx = zstd_mod.ZstdDecompressor()
            with open(path, "rb") as raw_f, dctx.stream_reader(raw_f) as reader:
                text_wrapper = io.TextIOWrapper(reader, encoding="utf-8")
                return cls.from_json(text_wrapper)
        elif comp is None or comp in ("none", "identity"):
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_json(f)
        else:
            raise ValueError(f"Unsupported compression format: {compression}")

    @classmethod
    def from_json(
        cls,
        json_str: str | bytes | IO[Any] | TextIO | BinaryIO | Path,
    ) -> LogicalGraph:
        """Deserialize a graph from a JSON string, stream, or file path.

        Supports streaming deserialization from file-like objects (streams)
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
            data = json.load(json_str)
        elif isinstance(json_str, bytes):
            data = json.loads(json_str.decode("utf-8"))
        elif isinstance(json_str, str):
            stripped = json_str.strip()
            if (
                not stripped.startswith(("{", "["))
                and "\n" not in json_str
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
        for inp_id in node.inputs:
            if inp_id in graph.nodes:
                adj[inp_id].append(nid)
                in_degree[nid] += 1
            else:
                producer = graph.get_output_producer(inp_id)
                if producer is not None and producer.id in graph.nodes:
                    adj[producer.id].append(nid)
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
