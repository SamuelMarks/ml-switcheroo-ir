"""Tests for dual serialization and deserialization in LogicalGraph."""

from __future__ import annotations

import json

from ml_switcheroo_ir import (
    LogicalEdge,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
)


def test_from_json_with_nodes_as_dict() -> None:
    """Test parsing dictionary-based node layout."""
    data = {
        "name": "DictModel",
        "nodes": {
            "n1": {
                "id": "n1",
                "op_type": "Conv",
                "attributes": {"kernel_shape": [3, 3]},
            }
        },
        "outputs": ["n1"],
    }
    graph = LogicalGraph.from_json(json.dumps(data))
    assert graph.name == "DictModel"
    assert "n1" in graph.nodes
    assert graph.nodes["n1"].op_type == "Conv"
    assert graph.outputs == ["n1"]


def test_from_json_with_nodes_as_list() -> None:
    """Test parsing list-based node layout."""
    data = {
        "name": "ListModel",
        "nodes": [
            {
                "id": "node_a",
                "kind": "Relu",
                "metadata": {"test_attr": 42},
                "inputs": ["node_input"],
            }
        ],
    }
    graph = LogicalGraph.from_json(json.dumps(data))
    assert graph.name == "ListModel"
    assert "node_a" in graph.nodes
    assert graph.nodes["node_a"].op_type == "Relu"
    assert graph.nodes["node_a"].attributes == {"test_attr": 42}
    assert graph.nodes["node_a"].inputs == ["node_input"]


def test_from_json_with_explicit_edges() -> None:
    """Test parsing explicit edges array and populating node inputs."""
    data = {
        "name": "EdgeModel",
        "nodes": [
            {"id": "a", "op_type": "Input"},
            {"id": "b", "op_type": "Relu"},
        ],
        "edges": [
            {"source": "a", "target": "b"},
            {
                "source": "a",
                "target": "b",
            },  # Duplicate edge shouldn't duplicate in inputs
            {"source": "unknown", "target": "unknown_target"},  # Safely handled
        ],
    }
    graph = LogicalGraph.from_json(json.dumps(data))
    assert graph.nodes["b"].inputs == ["a"]
    assert graph.edges == [LogicalEdge(source="a", target="b")]


def test_to_json_deterministic() -> None:
    """Test that to_json() produces deterministic string outputs across multiple runs."""
    mesh = LogicalMesh(shape={"data": 2, "model": 4})
    spec = PartitionSpec(axes=("data", ("model", "pipeline"), None))
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(
        id="n2",
        op_type="Linear",
        inputs=["n1"],
        sharding=spec,
        shape_metadata=(16, 64),
    )
    graph = LogicalGraph(
        name="DeterministicGraph", nodes={"n1": n1, "n2": n2}, mesh=mesh
    )

    json1 = graph.to_json(format="canonical", indent=2)
    json2 = graph.to_json(format="canonical", indent=2)
    assert json1 == json2

    json_legacy = graph.to_json(format="legacy", indent=2)
    assert "edges" not in json.loads(json_legacy)


def test_roundtrip_json_serialization() -> None:
    """Test LogicalGraph -> JSON -> LogicalGraph identity."""
    mesh = LogicalMesh(shape={"data": 8})
    spec = PartitionSpec(axes=("data", ("tensor", "expert"), None))
    n1 = LogicalNode(id="x", op_type="Input")
    n2 = LogicalNode(
        id="y",
        op_type="MatMul",
        inputs=["x"],
        sharding=spec,
        shape_metadata=("B", 128),
    )
    graph = LogicalGraph(
        name="RoundtripModel",
        nodes={"x": n1, "y": n2},
        outputs=["y"],
        mesh=mesh,
    )

    serialized = graph.to_json()
    deserialized = LogicalGraph.from_json(serialized)

    assert deserialized.name == graph.name
    assert set(deserialized.nodes.keys()) == set(graph.nodes.keys())
    assert deserialized.nodes["y"].op_type == "MatMul"
    assert deserialized.nodes["y"].inputs == ["x"]
    assert deserialized.nodes["y"].shape_metadata == ("B", 128)
    assert deserialized.nodes["y"].sharding is not None
    assert deserialized.nodes["y"].sharding.axes == ("data", ("tensor", "expert"), None)
    assert deserialized.mesh is not None
    assert deserialized.mesh.shape == {"data": 8}
    assert deserialized.outputs == ["y"]
    assert deserialized.edges == graph.edges
