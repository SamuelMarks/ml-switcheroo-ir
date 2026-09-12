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
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, BinaryIO, Iterator, Sequence, TextIO, overload

try:
    import zstandard
except ImportError:  # pragma: no cover
    zstandard = None  # type: ignore[assignment]

from ml_switcheroo_ir.types import AttributeValue, DType

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
    "PartitionSpec",
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

    """

    source: str
    target: str


class CyclicGraphError(Exception):
    """Raised when a cycle is detected in a graph during topological sorting."""


@dataclass
class LogicalNode:
    """Represents a computation unit (Layer) in the graph.

    Attributes:
        id (str): Unique identifier (e.g. 'conv1').
        op_type (str): Operation type. Standard types include 'Conv2d', 'Linear', 'Input', 'Output', as well as advanced primitives.
        domain (str): Operator domain (e.g., 'ai.onnx').
        version (int): Operator set version (e.g., 1).
        attributes (Dict[str, AttributeValue]): Dictionary of configuration parameters (e.g. ``kernel_size=3``).
        inputs (List[str]): Ordered list of upstream LogicalNode IDs.
        outputs (List[str]): Ordered list of output names/SSA value identifiers produced by this node.
        shape_metadata (Optional[Tuple[Union[int, str], ...]]): Tuple of integers or string symbols ("B", "T").
        source_ast_ref (Optional[str]): Traceback to exact file path, line number, and cdd-python AST node ID.
        sharding (Optional[PartitionSpec]): Optional layout specification for distributed placement of this node's output.

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

    @overload
    def __init__(
        self,
        id: str,
        op_type: str = ...,
        domain: str = "ai.onnx",
        version: int = 1,
        attributes: dict[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: tuple[int | str, ...] | Sequence[int | str] | Any = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
    ) -> None:
        """Initialize LogicalNode using canonical op_type and attributes.

        Args:
            id (str): Unique identifier for the node.
            op_type (str): Operation type.
            domain (str): Operator domain (default: 'ai.onnx').
            version (int): Operator version (default: 1).
            attributes (Optional[Dict[str, AttributeValue]]): Operator attributes.
            inputs (Optional[List[str]]): Ordered list of upstream input IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (Optional[Union[Tuple[Union[int, str], ...], Sequence[Union[int, str]], Any]]): Tensor
                shape metadata, accepting standard integer/string dimension tuples/lists, duck-typed metadata
                objects defining a `.shape` attribute, or arbitrary custom non-iterable objects.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
        """

    @overload
    def __init__(
        self,
        id: str,
        *,
        kind: str,
        domain: str = "ai.onnx",
        version: int = 1,
        metadata: dict[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: tuple[int | str, ...] | Sequence[int | str] | Any = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
    ) -> None:
        """Initialize LogicalNode using legacy/alternative kind and metadata.

        Args:
            id (str): Unique identifier for the node.
            kind (str): Alternative alias for op_type.
            domain (str): Operator domain (default: 'ai.onnx').
            version (int): Operator version (default: 1).
            metadata (Optional[Dict[str, AttributeValue]]): Alternative alias for attributes.
            inputs (Optional[List[str]]): Ordered list of upstream input IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (Optional[Union[Tuple[Union[int, str], ...], Sequence[Union[int, str]], Any]]): Tensor
                shape metadata, accepting standard integer/string dimension tuples/lists, duck-typed metadata
                objects defining a `.shape` attribute, or arbitrary custom non-iterable objects.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
        """

    @overload
    def __init__(
        self,
        id: str,
        op_type: str | None = None,
        domain: str = "ai.onnx",
        version: int = 1,
        attributes: dict[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: tuple[int | str, ...] | Sequence[int | str] | Any = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        *,
        kind: str | None = None,
        metadata: dict[str, AttributeValue] | None = None,
    ) -> None:
        """Initialize LogicalNode with flexible parameter naming conventions.

        Args:
            id (str): Unique identifier for the node.
            op_type (Optional[str]): Operation type.
            domain (str): Operator domain (default: 'ai.onnx').
            version (int): Operator version (default: 1).
            attributes (Optional[Dict[str, AttributeValue]]): Operator attributes.
            inputs (Optional[List[str]]): Ordered list of upstream input IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (Optional[Union[Tuple[Union[int, str], ...], Sequence[Union[int, str]], Any]]): Tensor
                shape metadata, accepting standard integer/string dimension tuples/lists, duck-typed metadata
                objects defining a `.shape` attribute, or arbitrary custom non-iterable objects.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            kind (Optional[str]): Alternative alias for op_type.
            metadata (Optional[Dict[str, AttributeValue]]): Alternative alias for attributes.
        """

    def __init__(
        self,
        id: str,
        op_type: str | None = None,
        domain: str = "ai.onnx",
        version: int = 1,
        attributes: dict[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: tuple[int | str, ...] | Sequence[int | str] | Any = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        *,
        kind: str | None = None,
        metadata: dict[str, AttributeValue] | None = None,
    ) -> None:
        """Initialize a LogicalNode instance with dual parameter naming support.

        Args:
            id (str): Unique identifier for the node.
            op_type (Optional[str]): Operation type (e.g. 'Conv', 'Relu').
            domain (str): Operator domain (default: 'ai.onnx').
            version (int): Operator set version (default: 1).
            attributes (Optional[Dict[str, AttributeValue]]): Operator configuration attributes.
            inputs (Optional[List[str]]): Ordered list of upstream LogicalNode IDs.
            outputs (Optional[List[str]]): Ordered list of output SSA names.
            shape_metadata (Optional[Union[Tuple[Union[int, str], ...], Sequence[Union[int, str]], Any]]): Tensor
                shape metadata, accepting standard integer/string dimension tuples/lists, duck-typed metadata
                objects defining a `.shape` attribute, or arbitrary custom non-iterable objects.
            source_ast_ref (Optional[str]): Source AST trace reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            kind (Optional[str]): Alternative alias for op_type.
            metadata (Optional[Dict[str, AttributeValue]]): Alternative alias for attributes.

        Raises:
            ValueError: If neither op_type nor kind is provided, or if both are provided with conflicting values.
        """
        if op_type is None and kind is None:
            raise ValueError(
                "Either 'op_type' or 'kind' must be specified for LogicalNode."
            )
        if op_type is not None and kind is not None and op_type != kind:
            raise ValueError(
                f"Conflicting op_type ({op_type!r}) and kind ({kind!r}) specified for LogicalNode."
            )

        resolved_op_type = op_type if op_type is not None else kind
        assert resolved_op_type is not None

        resolved_attributes: dict[str, AttributeValue] = {}
        if metadata is not None:
            resolved_attributes.update(metadata)
        if attributes is not None:
            resolved_attributes.update(attributes)

        self.id = id
        self.op_type = resolved_op_type
        self.domain = domain
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
            else:
                try:
                    self.shape_metadata = tuple(inner_shape)
                except TypeError:
                    self.shape_metadata = inner_shape
        else:
            try:
                self.shape_metadata = tuple(shape_metadata)
            except TypeError:
                self.shape_metadata = shape_metadata
        self.source_ast_ref = source_ast_ref
        self.sharding = sharding

    @property
    def kind(self) -> str:
        """Alias for op_type for backward compatibility.

        Returns:
            str: The operation type of this node.
        """
        return self.op_type

    @kind.setter
    def kind(self, value: str) -> None:
        """Set op_type via kind alias.

        Args:
            value (str): The operation type to set.
        """
        self.op_type = value

    @property
    def metadata(self) -> dict[str, AttributeValue]:
        """Alias for attributes for backward compatibility.

        Returns:
            dict[str, AttributeValue]: Dictionary of attribute metadata.
        """
        return self.attributes

    @metadata.setter
    def metadata(self, value: dict[str, AttributeValue]) -> None:
        """Set attributes via metadata alias.

        Args:
            value (dict[str, AttributeValue]): Dictionary of attribute metadata to set.
        """
        self.attributes = value


@dataclass
class LogicalGraph:
    """Language-agnostic representation of the neural network structure.

    Attributes:
        name (str): Name of the graph model/class.
        nodes (Dict[str, LogicalNode]): Map of id -> LogicalNode.
        outputs (List[str]): List of explicit output ids.
        mesh (Optional[LogicalMesh]): Optional logical device mesh for distributed training/inference.

    """

    name: str = "Model"
    nodes: dict[str, LogicalNode] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    mesh: LogicalMesh | None = None

    @overload
    def __init__(
        self,
        name: str = "Model",
        nodes: dict[str, LogicalNode] | None = None,
        outputs: list[str] | None = None,
        mesh: LogicalMesh | None = None,
        *,
        edges: list[LogicalEdge] | None = None,
    ) -> None:
        """Initialize LogicalGraph with nodes specified as a dictionary mapping id to LogicalNode.

        Args:
            name (str): Name of the graph model/class.
            nodes (Optional[Dict[str, LogicalNode]]): Map of node ID to LogicalNode instance.
            outputs (Optional[List[str]]): List of explicit output IDs.
            mesh (Optional[LogicalMesh]): Optional logical device mesh.
            edges (Optional[List[LogicalEdge]]): Optional list of directed edges to populate.
        """

    @overload
    def __init__(
        self,
        name: str = "Model",
        nodes: list[LogicalNode] | None = None,
        outputs: list[str] | None = None,
        mesh: LogicalMesh | None = None,
        *,
        edges: list[LogicalEdge] | None = None,
    ) -> None:
        """Initialize LogicalGraph with nodes specified as a list of LogicalNode instances.

        Args:
            name (str): Name of the graph model/class.
            nodes (Optional[List[LogicalNode]]): List of LogicalNode instances.
            outputs (Optional[List[str]]): List of explicit output IDs.
            mesh (Optional[LogicalMesh]): Optional logical device mesh.
            edges (Optional[List[LogicalEdge]]): Optional list of directed edges to populate.
        """

    def __init__(
        self,
        name: str = "Model",
        nodes: dict[str, LogicalNode] | list[LogicalNode] | None = None,
        outputs: list[str] | None = None,
        mesh: LogicalMesh | None = None,
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
            edges (Optional[List[LogicalEdge]]): Optional list of directed edges to populate
                directly into each target node's inputs list.
        """
        self.name = name

        if nodes is None:
            self.nodes = {}
        elif isinstance(nodes, list):
            self.nodes = {node.id: node for node in nodes}
        else:
            self.nodes = dict(nodes)

        if edges is not None:
            for edge in edges:
                if edge.target in self.nodes:
                    target_node = self.nodes[edge.target]
                    if edge.source not in target_node.inputs:
                        target_node.inputs.append(edge.source)

        if outputs is not None:
            self.outputs = list(outputs)
        else:
            consumed: set[str] = set()
            for n in self.nodes.values():
                for inp in n.inputs:
                    consumed.add(inp)
            out_ids: list[str] = []
            for nid, n in self.nodes.items():
                if nid not in consumed and not any(o in consumed for o in n.outputs):
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
    def edges(self) -> list[LogicalEdge]:
        """Derive and return directed edges from each node's inputs.

        Returns:
            List[LogicalEdge]: List of directed edges from upstream inputs to target nodes.
        """
        result: list[LogicalEdge] = []
        for target_node in self.nodes.values():
            for src in target_node.inputs:
                result.append(LogicalEdge(source=src, target=target_node.id))
        return result

    @edges.setter
    def edges(self, edge_list: list[LogicalEdge]) -> None:
        """Synchronize edges into target nodes' inputs.

        Args:
            edge_list (List[LogicalEdge]): List of edges to configure.
        """
        for node in self.nodes.values():
            node.inputs.clear()
        for edge in edge_list:
            if edge.target in self.nodes:
                target_node = self.nodes[edge.target]
                if edge.source not in target_node.inputs:
                    target_node.inputs.append(edge.source)

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

    def to_json(self, format: str = "canonical", indent: int = 2) -> str:
        """Serialize the graph to a deterministic JSON string.

        Args:
            format (str): Serialization format ('canonical' or 'legacy').
            indent (int): Indentation spaces for JSON formatting.

        Returns:
            str: Deterministic JSON string representation.
        """
        data = asdict(self)
        if format == "canonical":
            data["edges"] = [asdict(edge) for edge in self.edges]
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
            fp (IO[Any] | TextIO): Writable file-like stream.
            format (str): Serialization format ('canonical' or 'legacy').
            indent (int): Indentation spaces for JSON formatting.
        """
        data = asdict(self)
        if format == "canonical":
            data["edges"] = [asdict(edge) for edge in self.edges]
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
            json_str (str | bytes | IO[Any] | TextIO | BinaryIO | Path): JSON string, bytes,
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

        outputs = data.get("outputs")
        return cls(
            name=data.get("name", "Model"), nodes=nodes, outputs=outputs, mesh=mesh
        )


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
