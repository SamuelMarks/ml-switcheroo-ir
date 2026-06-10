"""Intermediate Representation (IR).

This module defines the language-agnostic graph data structures used to represent
Deep Learning models after ingestion from source code (e.g. Python/LibCST) or
explicit definition.

It acts as the contract between the Frontend (Ingestion) and the Backend (Synthesis).
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import json

from ml_switcheroo_ir.types import AttributeValue


@dataclass
class LogicalAxis:
    """Represents a named dimension for tensor sizes and sharding (e.g., 'batch', 'embed', 'heads').

    Attributes:
        name (str): Name of the logical axis.
        size (Optional[int]): Optional fixed size of the axis.

    """

    name: str
    size: Optional[int] = None


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

    axes: Tuple[Optional[Union[str, Tuple[str, ...]]], ...]


@dataclass
class LogicalMesh:
    """Represents a multi-dimensional grid of devices for distributed execution.

    Attributes:
        shape (Dict[str, int]): Mapping of mesh axis names to their sizes (e.g., {'data': 4, 'model': 2}).

    """

    shape: Dict[str, int]


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
        shape_metadata (Optional[Tuple[Union[int, str], ...]]): Tuple of integers or string symbols ("B", "T").
        source_ast_ref (Optional[str]): Traceback to exact file path, line number, and cdd-python AST node ID.
        sharding (Optional[PartitionSpec]): Optional layout specification for distributed placement of this node's output.

    """

    id: str
    op_type: str
    domain: str = "ai.onnx"
    version: int = 1
    attributes: Dict[str, AttributeValue] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    shape_metadata: Optional[Tuple[Union[int, str], ...]] = None
    source_ast_ref: Optional[str] = None
    sharding: Optional[PartitionSpec] = None

    @property
    def kind(self) -> str:
        """Alias for op_type for backward compatibility."""
        return self.op_type

    @property
    def metadata(self) -> Dict[str, AttributeValue]:
        """Alias for attributes for backward compatibility."""
        return self.attributes


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
    nodes: Dict[str, LogicalNode] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    mesh: Optional[LogicalMesh] = None

    def to_json(self) -> str:
        """Serialize the graph to a deterministic JSON string."""
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "LogicalGraph":
        """Deserialize a graph from a JSON string."""
        data = json.loads(json_str)
        nodes_data = data.get("nodes", {})
        nodes = {}
        # Support both Dict and List representation of nodes for backward compatibility
        if isinstance(nodes_data, list):
            nodes_data = {n["id"]: n for n in nodes_data}

        for nid, ndata in nodes_data.items():
            if "kind" in ndata and "op_type" not in ndata:
                ndata["op_type"] = ndata.pop("kind")
            if "metadata" in ndata and "attributes" not in ndata:
                ndata["attributes"] = ndata.pop("metadata")
            sharding = ndata.get("sharding")
            if sharding:
                ndata["sharding"] = PartitionSpec(axes=tuple(sharding["axes"]))
            if "shape_metadata" in ndata and ndata["shape_metadata"]:
                ndata["shape_metadata"] = tuple(ndata["shape_metadata"])
            nodes[nid] = LogicalNode(**ndata)

        mesh_data = data.get("mesh")
        mesh = LogicalMesh(shape=mesh_data["shape"]) if mesh_data else None

        # Build output list from explicit 'outputs' or legacy 'edges'
        outputs = data.get("outputs", [])
        if "edges" in data and not outputs:
            # legacy logic: nodes with no outgoing edges might be outputs, or output nodes.
            # We don't try to reconstruct the entire output list, just keep it empty if not provided.
            pass

        return cls(
            name=data.get("name", "Model"), nodes=nodes, outputs=outputs, mesh=mesh
        )


def topological_sort(graph: LogicalGraph) -> List[LogicalNode]:
    """Sorts graph nodes by dependency order.

    Ensures that for every edge u -> v, u appears before v in the returned list.
    Handles disconnected components and cycles gracefully by appending
    unreachable nodes in their original definition order.

    Args:
        graph (LogicalGraph): The logical graph to sort.

    Returns:
        List[LogicalNode]: List of nodes in execution order.

    """
    adj: Dict[str, List[str]] = defaultdict(list)
    in_degree: Dict[str, int] = defaultdict(int)

    # Initialize in-degree for all nodes
    for nid in graph.nodes:
        in_degree[nid] = 0

    # Build adjacency and degree maps based on node inputs
    for nid, node in graph.nodes.items():
        for inp_id in node.inputs:
            if inp_id in graph.nodes:
                adj[inp_id].append(nid)
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

    # Handle disconnected components or cycles by appending remaining nodes
    if len(sorted_nodes) < len(graph.nodes):
        seen = {n.id for n in sorted_nodes}
        # Append remaining nodes in dictionary iteration order (fallback)
        for nid, n in graph.nodes.items():
            if nid not in seen:
                sorted_nodes.append(n)

    return sorted_nodes


class CompilerBackend(ABC):
    """Abstract base class for compilation backends."""

    @abstractmethod
    def compile(self, graph: LogicalGraph) -> Any:
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
