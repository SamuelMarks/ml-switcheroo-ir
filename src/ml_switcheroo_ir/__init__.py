"""Intermediate Representation (IR).

This module defines the language-agnostic graph data structures used to represent
Deep Learning models after ingestion from source code (e.g. Python/LibCST) or
explicit definition.

It acts as the contract between the Frontend (Ingestion) and the Backend (Synthesis).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import defaultdict, deque
from abc import ABC, abstractmethod

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
        kind (str): Operation type. Standard types include 'Conv2d', 'Linear', 'Input', 'Output', as well as advanced primitives.
        domain (str): Operator domain (e.g., 'ai.onnx').
        version (int): Operator set version (e.g., 1).
        metadata (Dict[str, AttributeValue]): Dictionary of configuration parameters (e.g. ``kernel_size=3``).
        sharding (Optional[PartitionSpec]): Optional layout specification for distributed placement of this node's output.

    """

    id: str
    kind: str
    domain: str = "ai.onnx"
    version: int = 1
    metadata: Dict[str, AttributeValue] = field(default_factory=dict)
    sharding: Optional[PartitionSpec] = None


@dataclass
class LogicalEdge:
    """Represents data flow between two nodes.

    Attributes:
        source (str): Source node ID.
        target (str): Target node ID.

    """

    source: str
    target: str


@dataclass
class LogicalGraph:
    """Language-agnostic representation of the neural network structure.

    Attributes:
        name (str): Name of the graph model/class.
        nodes (List[LogicalNode]): Ordered list of nodes in the graph.
        edges (List[LogicalEdge]): List of directed edges between nodes.
        mesh (Optional[LogicalMesh]): Optional logical device mesh for distributed training/inference.

    """

    name: str = "Model"
    nodes: List[LogicalNode] = field(default_factory=list)
    edges: List[LogicalEdge] = field(default_factory=list)
    mesh: Optional[LogicalMesh] = None


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
    nodes_by_id = {n.id: n for n in graph.nodes}

    # Initialize in-degree for all nodes
    for n in graph.nodes:
        in_degree[n.id] = 0

    # Build adjacency and degree maps
    for edge in graph.edges:
        if edge.source in nodes_by_id and edge.target in nodes_by_id:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1

    # Simple queue-based toposort
    # Note: Using sorted keys for determinism in queue initialization
    initial_roots = sorted([n.id for n in graph.nodes if in_degree[n.id] == 0])
    queue = deque(initial_roots)
    sorted_nodes = []

    while queue:
        u = queue.popleft()
        sorted_nodes.append(nodes_by_id[u])

        for v in sorted(adj[u]):  # Sorting for determinism
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Handle disconnected components or cycles by appending remaining nodes
    if len(sorted_nodes) < len(graph.nodes):
        seen = {n.id for n in sorted_nodes}
        # Append remaining nodes in definition order (fallback)
        for n in graph.nodes:
            if n.id not in seen:
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
