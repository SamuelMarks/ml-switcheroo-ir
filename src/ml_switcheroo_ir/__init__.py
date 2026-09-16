"""Intermediate Representation (IR).

This module defines the language-agnostic graph data structures used to represent
Deep Learning models after ingestion from source code (e.g. Python/LibCST) or
explicit definition.

It acts as the contract between the Frontend (Ingestion) and the Backend (Synthesis).
"""

from __future__ import annotations

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

try:
    import zstandard
except ImportError:  # pragma: no cover
    zstandard = None  # type: ignore[assignment]

from ml_switcheroo_ir.types import (
    AttributeValue,
    DType,
    TensorShape,
    TensorSpec,
)

__version__ = "0.0.3"

__all__ = [
    "AttributeValue",
    "BaseFrontend",
    "CompilerBackend",
    "CyclicGraphError",
    "DType",
    "GraphFrontend",
    "LogicalAxis",
    "LogicalEdge",
    "LogicalGraph",
    "LogicalMesh",
    "LogicalNode",
    "NodeDict",
    "PartitionSpec",
    "TensorShape",
    "TensorSpec",
    "__version__",
    "eliminate_common_subexpressions",
    "eliminate_dead_nodes",
    "estimate_communication_volume",
    "estimate_graph_communication_volume",
    "export_schemas",
    "generate_typescript_definitions",
    "get_json_schema",
    "propagate_shapes_and_constants",
    "topological_sort",
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

    """

    shape: dict[str, int]


@dataclass
class LogicalEdge:
    """Represents a directed data-flow edge between two nodes in the graph.

    Attributes:
        source (str): Source node identifier emitting data.
        target (str): Target node identifier consuming data.
        source_idx (int): Output port index of the source node (default: 0).
        target_idx (int): Input argument index of the target node (default: 0).
        value_name (Optional[str]): Name of the intermediate SSA value (default: None).

    """

    source: str
    target: str
    source_idx: int = 0
    target_idx: int = 0
    value_name: str | None = None


class CyclicGraphError(Exception):
    """Raised when a cycle is detected in a graph during topological sorting."""


_UNSET_OP_TYPE: Any = object()


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
        if kind is not None:
            warnings.warn(
                "The 'kind' parameter is deprecated; use 'op_type' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if metadata is not None:
            warnings.warn(
                "The 'metadata' parameter is deprecated; use 'attributes' instead.",
                DeprecationWarning,
                stacklevel=2,
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

        resolved_op_type: str = ""
        if op_type is not _UNSET_OP_TYPE and op_type is not None:
            resolved_op_type = str(op_type)
        elif kind is not None:
            resolved_op_type = kind

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

        self.id = id
        self.op_type = resolved_op_type
        self.domain = resolved_domain
        self.version = version
        self.attributes = resolved_attributes
        self.inputs = list(inputs) if inputs is not None else []
        self.outputs = (
            list(outputs) if outputs is not None and len(outputs) > 0 else [id]
        )
        if shape_metadata is None:
            self.shape_metadata = None
        elif isinstance(shape_metadata, tuple):
            self.shape_metadata = shape_metadata
        elif isinstance(shape_metadata, list):
            self.shape_metadata = tuple(shape_metadata)
        elif hasattr(shape_metadata, "shape"):
            inner_shape = shape_metadata.shape
            if isinstance(inner_shape, tuple):
                self.shape_metadata = inner_shape
            elif isinstance(inner_shape, list):
                self.shape_metadata = tuple(inner_shape)
            elif not isinstance(inner_shape, (str, bytes, dict)):
                try:
                    self.shape_metadata = tuple(inner_shape)
                except TypeError:
                    self.shape_metadata = inner_shape
            else:
                self.shape_metadata = inner_shape
        elif not isinstance(shape_metadata, (str, bytes, dict)):
            try:
                self.shape_metadata = tuple(shape_metadata)
            except TypeError:
                self.shape_metadata = shape_metadata
        else:
            self.shape_metadata = shape_metadata

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
            "The 'kind' property is deprecated; use 'op_type' instead.",
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
            "The 'kind' setter is deprecated; use 'op_type' instead.",
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
            "The 'metadata' property is deprecated; use 'attributes' instead.",
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
            "The 'metadata' setter is deprecated; use 'attributes' instead.",
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


class EdgeList(list["LogicalEdge"]):
    """List of LogicalEdge instances with bidirectional synchronization to node inputs."""

    def __init__(
        self,
        graph: LogicalGraph | None = None,
        iterable: Sequence[LogicalEdge] | Any = (),
    ) -> None:
        """Initialize EdgeList linked to parent LogicalGraph.

        Args:
            graph (Optional[LogicalGraph]): Linked parent graph.
            iterable (Union[Sequence[LogicalEdge], Any]): Initial edges iterable.
        """
        self._graph = graph
        super().__init__(iterable)

    def append(self, edge: LogicalEdge) -> None:
        """Append directed edge and synchronize into target node inputs.

        Args:
            edge (LogicalEdge): The edge to append.
        """
        super().append(edge)
        graph = getattr(self, "_graph", None)
        if graph is not None:
            if edge.target in graph.nodes:
                target_node = graph.nodes[edge.target]
                if edge.source not in target_node.inputs:
                    target_node.inputs.append(edge.source)
            else:
                graph._pending_edges.append(edge)

    def extend(self, edges: Sequence[LogicalEdge] | Any) -> None:
        """Extend EdgeList with multiple edges and synchronize into node inputs.

        Args:
            edges (Union[Sequence[LogicalEdge], Any]): Edges to append.
        """
        for edge in edges:
            self.append(edge)


class NodeDict(dict[str, "LogicalNode"]):
    """Dictionary mapping node IDs to LogicalNodes with dual dict and sequence ergonomics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize NodeDict with optional parent graph and dictionary data.

        Args:
            *args (Any): Optional graph instance or dictionary mapping.
            **kwargs (Any): Keyword dictionary arguments.
        """
        self._graph: Any = kwargs.pop("graph", None)
        if len(args) == 2:
            self._graph = args[0]
            super().__init__()
            for k, v in args[1].items():
                self[k] = v
        elif (
            len(args) == 1
            and hasattr(args[0], "__dataclass_fields__")
            and not hasattr(args[0], "items")
        ):
            self._graph = args[0]
            super().__init__()
        elif len(args) == 1:
            super().__init__()
            items = args[0].items() if hasattr(args[0], "items") else args[0]
            for k, v in (items if hasattr(items, "items") else dict(items).items()):
                self[k] = v
        else:
            super().__init__(*args, **kwargs)

    def __getstate__(self) -> dict[str, Any]:
        """Return state dictionary for pickle serialization.

        Returns:
            dict[str, Any]: State dictionary containing node items and linked graph.
        """
        return {"_items": dict(self), "_graph": getattr(self, "_graph", None)}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from pickle dictionary.

        Args:
            state (dict[str, Any]): Pickled state dictionary.
        """
        self._graph = state.get("_graph")
        self.update(state.get("_items", {}))

    def __setitem__(self, key: str, node: LogicalNode) -> None:
        """Insert or update node by ID string and wire any pending edges.

        Args:
            key (str): String node ID.
            node (LogicalNode): LogicalNode instance.
        """
        super().__setitem__(key, node)
        graph = getattr(self, "_graph", None)
        if graph is not None and hasattr(graph, "_pending_edges"):
            remaining: list[LogicalEdge] = []
            for edge in graph._pending_edges:
                if edge.target == key:
                    if edge.source not in node.inputs:
                        node.inputs.append(edge.source)
                else:
                    remaining.append(edge)
            graph._pending_edges = remaining

    def __getitem__(self, key: Any) -> LogicalNode:
        """Retrieve node by ID string or integer sequence index.

        Args:
            key (Any): String node ID or integer index.

        Returns:
            LogicalNode: The requested node.

        Raises:
            IndexError: If integer index is out of bounds.
            KeyError: If string key is not found.
        """
        if isinstance(key, int):
            try:
                return list(self.values())[key]
            except IndexError as err:
                raise IndexError(
                    f"NodeDict index {key} out of range (length {len(self)})"
                ) from err
        return super().__getitem__(key)

    def __delitem__(self, key: Any) -> None:
        """Delete node by ID string or integer sequence index.

        Args:
            key (Any): String node ID or integer index.
        """
        if isinstance(key, int):
            target_key = list(self.keys())[key]
            super().__delitem__(target_key)
        else:
            super().__delitem__(key)

    def __contains__(self, key: Any) -> bool:
        """Check presence of node by ID string or LogicalNode object.

        Args:
            key (Any): String ID or LogicalNode instance.

        Returns:
            bool: True if node ID is present.
        """
        if isinstance(key, str):
            return super().__contains__(key)
        if hasattr(key, "id"):
            return super().__contains__(key.id)
        return False

    def append(self, node: LogicalNode) -> None:
        """Add node by ID with deprecation warning for legacy list ergonomics.

        Args:
            node (LogicalNode): The node to add.
        """
        warnings.warn(
            "Using graph.nodes.append() is deprecated; use graph.nodes[node.id] = node or graph.add_node(node) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self[node.id] = node

    def extend(self, nodes: Sequence[LogicalNode] | Any) -> None:
        """Extend NodeDict with an iterable of LogicalNode instances.

        Args:
            nodes (Union[Sequence[LogicalNode], Any]): Iterable of nodes to append.
        """
        warnings.warn(
            "Using graph.nodes.extend() is deprecated; use graph.nodes.update() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        for node in nodes:
            self.append(node)

    def insert(self, index: int, node: LogicalNode) -> None:
        """Insert node with sequence ergonomics.

        Args:
            index (int): Sequence index.
            node (LogicalNode): Node to insert.
        """
        self[node.id] = node

    def pop(self, key: Any, *args: Any) -> Any:
        """Remove and return node by string ID or integer index.

        Args:
            key (Any): String ID or integer index.
            *args (Any): Default fallback value.

        Returns:
            Any: The removed LogicalNode or default value.
        """
        if isinstance(key, int):
            target_key = list(self.keys())[key]
            return super().pop(target_key, *args)
        return super().pop(key, *args)

    def index(self, node: Any) -> int:
        """Find the integer sequence index of a node by instance or ID.

        Args:
            node (Any): LogicalNode instance or ID string.

        Returns:
            int: Integer sequence index.

        Raises:
            ValueError: If node is not found.
        """
        target_id = getattr(node, "id", node)
        for idx, k in enumerate(self.keys()):
            if k == target_id:
                return idx
        raise ValueError(f"{node} is not in NodeDict")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        """Generate Pydantic core schema for NodeDict as a dictionary mapping string to LogicalNode.

        Args:
            source_type (Any): Source type to generate schema for.
            handler (Any): Schema generation handler.

        Returns:
            Any: Pydantic core schema representation.
        """
        from pydantic_core import core_schema

        return core_schema.dict_schema(
            keys_schema=core_schema.str_schema(),
            values_schema=handler.generate_schema(LogicalNode),
        )


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
    nodes: NodeDict = field(default_factory=NodeDict)
    inputs: list[str] = field(default_factory=list)
    input_specs: dict[str, TensorSpec] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    initializers: dict[str, Any] = field(default_factory=dict)
    mesh: LogicalMesh | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept assignments to ensure nodes and edges remain synchronized.

        Args:
            name (str): Attribute name.
            value (Any): Assigned value.
        """
        if name == "nodes":
            old_edges: list[LogicalEdge] = []
            try:
                old_edges = list(self.edges)
            except (AttributeError, TypeError, KeyError):
                old_edges = []

            if isinstance(value, NodeDict):
                value._graph = self
                super().__setattr__("nodes", value)
            elif isinstance(value, dict):
                super().__setattr__("nodes", NodeDict(self, value))
            elif isinstance(value, (list, tuple)):
                super().__setattr__(
                    "nodes",
                    NodeDict(self, {node.id: node for node in value}),
                )
            else:
                super().__setattr__("nodes", NodeDict(self))

            for e in old_edges:
                if e.target not in self.nodes:
                    self._pending_edges.append(e)
            return
        super().__setattr__(name, value)

    @overload
    def __init__(
        self,
        name: str = "Model",
        nodes: dict[str, LogicalNode] | None = None,
        outputs: list[str] | None = None,
        mesh: LogicalMesh | None = None,
        inputs: list[str] | None = None,
        input_specs: dict[str, TensorSpec] | None = None,
        initializers: dict[str, Any] | None = None,
        *,
        edges: list[LogicalEdge] | None = None,
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
        """

    def __init__(
        self,
        name: str = "Model",
        nodes: dict[str, LogicalNode] | list[LogicalNode] | None = None,
        outputs: list[str] | None = None,
        mesh: LogicalMesh | None = None,
        inputs: list[str] | None = None,
        input_specs: dict[str, TensorSpec] | None = None,
        initializers: dict[str, Any] | None = None,
        *,
        edges: list[LogicalEdge] | None = None,
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
        """
        self.name = name
        self._pending_edges = []

        if nodes is None:
            self.nodes = NodeDict(self)
        elif isinstance(nodes, list):
            warnings.warn(
                "Passing a list of nodes to LogicalGraph is deprecated; provide a dict[str, LogicalNode] instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.nodes = NodeDict(self, {node.id: node for node in nodes})
        else:
            self.nodes = NodeDict(self, nodes)

        self.inputs = list(inputs) if inputs is not None else []
        self.input_specs = dict(input_specs) if input_specs is not None else {}
        self.initializers = dict(initializers) if initializers is not None else {}

        # Auto-detect legacy Input pseudo-nodes into graph.inputs and input_specs
        for nid, node in list(self.nodes.items()):
            if node.op_type == "Input":
                param_name = str(node.attributes.get("name", nid))
                if param_name not in self.inputs:
                    self.inputs.append(param_name)
                if node.dtype is not None or node.shape_metadata is not None:
                    spec_dtype = node.dtype or DType.float32
                    spec_shape = (
                        node.shape_metadata
                        if isinstance(node.shape_metadata, tuple)
                        else ()
                    )
                    if param_name not in self.input_specs:
                        self.input_specs[param_name] = TensorSpec(
                            shape=spec_shape, dtype=spec_dtype
                        )

        if edges is not None:
            for edge in edges:
                if edge.target in self.nodes:
                    target_node = self.nodes[edge.target]
                    if edge.source not in target_node.inputs:
                        target_node.inputs.append(edge.source)
                else:
                    self._pending_edges.append(edge)

        if outputs is not None:
            self.outputs = list(outputs)
        else:
            consumed: set[str] = set()
            for n in self.nodes.values():
                for inp in n.inputs:
                    consumed.add(inp)
            out_ids: list[str] = []
            for nid, n in self.nodes.items():
                if n.op_type == "Output":
                    for inp in n.inputs:
                        if inp not in out_ids:
                            out_ids.append(inp)
                elif nid not in consumed and not any(o in consumed for o in n.outputs):
                    out_ids.append(nid)
            self.outputs = out_ids

        self.mesh = mesh

    @property
    def nodes_list(self) -> list[LogicalNode]:
        """Return list of LogicalNode instances in dictionary insertion order.

        Returns:
            List[LogicalNode]: Ordered list of nodes.
        """
        return list(self.nodes.values())

    @property
    def edges(self) -> EdgeList:
        """Derive and return directed edges dynamically from each node's inputs.

        Returns:
            EdgeList: List of directed edges from upstream inputs to target nodes.
        """
        result = EdgeList(self)
        for target_node in self.nodes.values():
            for target_idx, src in enumerate(target_node.inputs):
                src_idx = 0
                producer = self.get_output_producer(src)
                if producer is not None and src in producer.outputs:
                    src_idx = producer.outputs.index(src)
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
            edge_list (List[LogicalEdge]): List of edges to configure.
        """
        warnings.warn(
            "Setting 'edges' directly is deprecated; configure node.inputs directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        for node in self.nodes.values():
            node.inputs.clear()
        self._pending_edges.clear()
        for edge in edge_list:
            if edge.target in self.nodes:
                target_node = self.nodes[edge.target]
                if edge.source not in target_node.inputs:
                    target_node.inputs.append(edge.source)
            else:
                self._pending_edges.append(edge)

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

    def get_output_producer(self, output_name: str) -> LogicalNode | None:
        """Retrieve the LogicalNode that produces the specified output name.

        Args:
            output_name (str): Identifier or SSA name of the output.

        Returns:
            Optional[LogicalNode]: The producing node if found, otherwise None.
        """
        for node in self.nodes.values():
            if output_name in node.outputs:
                return node
        return None

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
        return self.nodes[node_id]

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
    def from_dict(cls, data: dict[str, Any]) -> LogicalGraph:
        """Deserialize a graph from a dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary containing graph fields.

        Returns:
            LogicalGraph: Deserialized graph instance.
        """
        nodes_data = data.get("nodes", {})
        nodes: dict[str, LogicalNode] = {}
        # Support both Dict and List representation of nodes for backward compatibility
        if isinstance(nodes_data, list):
            nodes_data = {n["id"]: n for n in nodes_data}

        for nid, ndata in nodes_data.items():
            ndata_copy = dict(ndata)
            if "kind" in ndata_copy and "op_type" not in ndata_copy:
                ndata_copy["op_type"] = ndata_copy.pop("kind")
            if "metadata" in ndata_copy and "attributes" not in ndata_copy:
                ndata_copy["attributes"] = ndata_copy.pop("metadata")
            sharding = ndata_copy.get("sharding")
            if sharding:
                ndata_copy["sharding"] = PartitionSpec(
                    axes=tuple(
                        tuple(a) if isinstance(a, list) else a for a in sharding["axes"]
                    )
                )
            if "outputs" in ndata_copy and ndata_copy["outputs"] is not None:
                ndata_copy["outputs"] = list(ndata_copy["outputs"])

            dtype_val = ndata_copy.get("dtype")
            if dtype_val is not None and not isinstance(dtype_val, DType):
                ndata_copy["dtype"] = DType.from_str(str(dtype_val))

            output_specs_raw = ndata_copy.get("output_specs")
            if output_specs_raw:
                ndata_copy["output_specs"] = [
                    TensorSpec(**s) if isinstance(s, dict) else s
                    for s in output_specs_raw
                ]

            # Deserialize nested subgraphs if present
            subgraphs_data = ndata_copy.get("subgraphs")
            resolved_subgraphs: dict[str, LogicalGraph] = {}
            if subgraphs_data and isinstance(subgraphs_data, dict):
                for sub_k, sub_v in subgraphs_data.items():
                    if isinstance(sub_v, dict):
                        resolved_subgraphs[sub_k] = cls.from_dict(sub_v)
                    elif isinstance(sub_v, LogicalGraph):
                        resolved_subgraphs[sub_k] = sub_v

            # Check for legacy subgraph attributes in attributes dict
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
                            resolved_subgraphs[canonical_key] = cls.from_dict(val)
                        elif isinstance(val, LogicalGraph):
                            resolved_subgraphs[canonical_key] = val
            ndata_copy["subgraphs"] = resolved_subgraphs

            nodes[nid] = LogicalNode(**ndata_copy)

        # Wire explicit edges if provided and inputs were not already populated
        edges_data = data.get("edges")
        if edges_data and isinstance(edges_data, list):
            for edge_item in edges_data:
                src = edge_item.get("source")
                tgt = edge_item.get("target")
                if tgt and tgt in nodes and src and src not in nodes[tgt].inputs:
                    nodes[tgt].inputs.append(src)

        mesh_data = data.get("mesh")
        mesh = LogicalMesh(shape=mesh_data["shape"]) if mesh_data else None

        inputs_data = data.get("inputs")
        inputs = list(inputs_data) if inputs_data is not None else []

        input_specs_raw = data.get("input_specs")
        input_specs: dict[str, TensorSpec] = {}
        if input_specs_raw and isinstance(input_specs_raw, dict):
            for k, v in input_specs_raw.items():
                if isinstance(v, dict):
                    input_specs[k] = TensorSpec(**v)
                elif isinstance(v, TensorSpec):
                    input_specs[k] = v

        initializers = data.get("initializers", {})

        outputs = data.get("outputs")
        return cls(
            name=data.get("name", "Model"),
            nodes=nodes,
            inputs=inputs,
            input_specs=input_specs,
            outputs=outputs,
            initializers=initializers,
            mesh=mesh,
        )

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
        path_str = str(path)
        comp = compression.lower() if compression else None
        if comp in ("gzip", "gz") or (comp is None and path_str.endswith(".gz")):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return cls.from_json(f)
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
    """Sorts graph nodes by dependency order.

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
    in_degree: dict[str, int] = defaultdict(int)

    # Initialize in-degree for all nodes
    for nid in graph.nodes:
        in_degree[nid] = 0

    # Build adjacency and degree maps based on node inputs and explicit edges
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

    # Simple queue-based toposort
    initial_roots = sorted([nid for nid in graph.nodes if in_degree[nid] == 0])
    queue = deque(initial_roots)
    sorted_nodes = []

    while queue:
        u = queue.popleft()
        sorted_nodes.append(graph.nodes[u])

        for v in sorted(adj[u]):  # Sorting for determinism
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Handle disconnected components or cycles
    if len(sorted_nodes) < len(graph.nodes):
        if strict:
            raise CyclicGraphError("Cycle detected in graph during topological sort.")
        seen = {n.id for n in sorted_nodes}
        # Append remaining nodes in dictionary iteration order (fallback)
        for nid, n in graph.nodes.items():
            if nid not in seen:
                sorted_nodes.append(n)

    return sorted_nodes


class CompilerBackend(ABC):
    """Abstract base class for compilation backends."""

    @abstractmethod
    def compile(self, graph: LogicalGraph) -> object:
        """Compiles the Logical Intermediate Representation (IR) into a target artifact.

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


# Import export utilities after LogicalGraph and LogicalNode are defined
from ml_switcheroo_ir.export import (
    export_schemas,
    generate_typescript_definitions,
    get_json_schema,
)
from ml_switcheroo_ir.transforms import (
    eliminate_common_subexpressions,
    eliminate_dead_nodes,
    propagate_shapes_and_constants,
)
from ml_switcheroo_ir.validator import (
    estimate_communication_volume,
    estimate_graph_communication_volume,
)
