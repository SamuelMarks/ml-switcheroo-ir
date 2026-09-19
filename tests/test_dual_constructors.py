"""Unit tests for dual constructor ergonomics and symmetrical API compatibility."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from ml_switcheroo_ir import (
    LogicalEdge,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
)


def test_logical_node_op_type_and_attributes() -> None:
    """Test LogicalNode constructor with canonical op_type and attributes."""
    node = LogicalNode(
        id="conv1",
        op_type="Conv",
        attributes={"kernel_size": 3, "stride": 1},
    )
    assert node.id == "conv1"
    assert node.op_type == "Conv"
    with pytest.deprecated_call():
        assert node.kind == "Conv"
    assert node.attributes == {"kernel_size": 3, "stride": 1}
    with pytest.deprecated_call():
        assert node.metadata == {"kernel_size": 3, "stride": 1}
    assert node.outputs == ["conv1"]


def test_logical_node_kind_and_metadata() -> None:
    """Test LogicalNode constructor with legacy kind and metadata."""
    with pytest.deprecated_call():
        node = LogicalNode(
            id="relu1",
            kind="Relu",
            metadata={"alpha": 0.2},
        )
    assert node.id == "relu1"
    assert node.op_type == "Relu"
    with pytest.deprecated_call():
        assert node.kind == "Relu"
    assert node.attributes == {"alpha": 0.2}
    with pytest.deprecated_call():
        assert node.metadata == {"alpha": 0.2}


def test_logical_node_both_matching() -> None:
    """Test LogicalNode constructor when both op_type and kind are provided and match."""
    with pytest.deprecated_call():
        node = LogicalNode(
            id="n1",
            op_type="Relu",
            kind="Relu",
            attributes={"a": 1},
            metadata={"b": 2},
        )
    assert node.op_type == "Relu"
    with pytest.deprecated_call():
        assert node.kind == "Relu"
    assert node.attributes == {"a": 1, "b": 2}


def test_logical_node_conflicting_op_type_and_kind() -> None:
    """Test LogicalNode constructor raises ValueError when op_type and kind conflict."""
    with pytest.warns(DeprecationWarning), pytest.raises(
        ValueError, match="Conflicting op_type"
    ):
        LogicalNode(id="n1", op_type="Conv", kind="Relu")


def test_logical_node_neither_op_type_nor_kind() -> None:
    """Test LogicalNode constructor raises ValueError when neither is provided."""
    with pytest.raises(
        ValueError, match="Either 'op_type' or 'kind' must be specified"
    ):
        LogicalNode(id="n1")


def test_logical_node_setters() -> None:
    """Test kind and metadata property setters on LogicalNode."""
    node = LogicalNode(id="n1", op_type="Relu")
    with pytest.deprecated_call():
        node.kind = "Gelu"
    assert node.op_type == "Gelu"
    with pytest.deprecated_call():
        assert node.kind == "Gelu"

    with pytest.deprecated_call():
        node.metadata = {"approximate": "tanh"}
    assert node.attributes == {"approximate": "tanh"}
    with pytest.deprecated_call():
        assert node.metadata == {"approximate": "tanh"}


def test_logical_node_dataclass_asdict_and_repr() -> None:
    """Test that dataclass asdict and repr function correctly with custom __init__."""
    spec = PartitionSpec(axes=("data", None))
    node = LogicalNode(
        id="n1",
        op_type="MatMul",
        shape_metadata=(128, 256),
        sharding=spec,
        source_ast_ref="test.py:10",
    )
    d = asdict(node)
    assert d["id"] == "n1"
    assert d["op_type"] == "MatMul"
    assert d["outputs"] == ["n1"]
    assert "LogicalNode(id='n1', op_type='MatMul'" in repr(node)


def test_logical_graph_nodes_as_dict() -> None:
    """Test LogicalGraph initialized with a dictionary of nodes."""
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Relu", inputs=["n1"])
    graph = LogicalGraph(name="DictGraph", nodes={"n1": n1, "n2": n2})
    assert len(graph.nodes) == 2
    assert graph.nodes["n1"] == n1
    assert graph.outputs == ["n2"]


def test_logical_graph_nodes_as_list() -> None:
    """Test LogicalGraph initialized with a list of nodes."""
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Relu", inputs=["n1"])
    with pytest.deprecated_call():
        graph = LogicalGraph(name="ListGraph", nodes=[n1, n2])
    assert set(graph.nodes.keys()) == {"n1", "n2"}
    assert graph.nodes["n1"] == n1
    assert graph.outputs == ["n2"]


def test_logical_graph_explicit_edges() -> None:
    """Test LogicalGraph initialized with explicit edges in constructor."""
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Relu")
    edges = [
        LogicalEdge(source="n1", target="n2"),
        LogicalEdge(source="n1", target="n2"),  # Duplicate to test deduplication
        LogicalEdge(source="n1", target="non_existent"),  # Target not in graph
    ]
    graph = LogicalGraph(name="EdgeGraph", nodes={"n1": n1, "n2": n2}, edges=edges)
    assert n2.inputs == ["n1"]
    assert graph.edges == [LogicalEdge(source="n1", target="n2")]
    assert graph.outputs == ["n2"]


def test_logical_graph_outputs_deduction_and_override() -> None:
    """Test deduction of graph outputs and explicit override."""
    # Explicit outputs
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Relu", inputs=["n1"])
    graph_explicit = LogicalGraph(
        name="ExplicitOutputs", nodes={"n1": n1, "n2": n2}, outputs=["custom_out"]
    )
    assert graph_explicit.outputs == ["custom_out"]

    # Empty graph
    graph_empty = LogicalGraph()
    assert graph_empty.outputs == []

    # Cycle: both consume each other -> empty out-degree zero
    c1 = LogicalNode(id="c1", op_type="Op", inputs=["c2"])
    c2 = LogicalNode(id="c2", op_type="Op", inputs=["c1"])
    graph_cycle = LogicalGraph(nodes={"c1": c1, "c2": c2})
    assert graph_cycle.outputs == []

    # Multi-output SSA node consumed by downstream
    split = LogicalNode(id="split", op_type="Split", outputs=["s0", "s1"])
    c_s0 = LogicalNode(id="c_s0", op_type="Relu", inputs=["s0"])
    graph_multi = LogicalGraph(nodes={"split": split, "c_s0": c_s0})
    # split's s0 is consumed, but split's id is not directly consumed, yet s0 is part of split.outputs
    # c_s0 is completely unconsumed, so c_s0 is the output
    assert "c_s0" in graph_multi.outputs


def test_logical_graph_roundtrip_instantiation() -> None:
    """Test roundtrip construction using Format B CST syntax."""
    mesh = LogicalMesh(shape={"data": 2, "model": 4})
    with pytest.deprecated_call():
        node1 = LogicalNode(
            id="x",
            kind="Input",
            domain="ai.onnx",
            version=17,
            metadata={"dtype": "float32"},
            sharding=PartitionSpec(axes=("data", None, "model")),
        )
        node2 = LogicalNode(
            id="norm1",
            kind="RMSNorm",
            domain="ml.switcheroo.custom",
            version=1,
            metadata={"eps": 1e-6},
        )
        graph = LogicalGraph(
            name="TransformerLayer",
            mesh=mesh,
            nodes=[node1, node2],
            edges=[LogicalEdge("x", "norm1")],
        )

    assert graph.name == "TransformerLayer"
    assert graph.nodes["norm1"].inputs == ["x"]
    assert graph.outputs == ["norm1"]
    assert graph.edges == [LogicalEdge(source="x", target="norm1")]


class BadShape:
    """Mock shape object that raises TypeError on iteration."""

    def __iter__(self) -> BadShape:
        """Iterate over elements, deliberately raising TypeError.

        Raises:
            TypeError: Always raised to simulate non-iterable custom shape objects.
        """
        raise TypeError("BadShape iteration not supported")


class ArbitraryObject:
    """Mock custom object without __iter__ or shape attribute."""


class MockMeta:
    """Duck-typed metadata object defining a shape property."""

    def __init__(self, shape: tuple[int, ...] | list[int] | range | str) -> None:
        """Initialize MockMeta with shape.

        Args:
            shape (Union[Tuple[int, ...], List[int], range, str]): Inner shape representation.
        """
        self.shape = shape


class DummyShape:
    """Duck-typed metadata object defining a shape property with non-iterable shape."""

    def __init__(self, shape: int) -> None:
        """Initialize DummyShape with integer shape.

        Args:
            shape (int): Integer scalar shape.
        """
        self.shape = shape


def test_logical_node_bad_shape_and_arbitrary_object() -> None:
    """Test LogicalNode initialization with non-iterable BadShape and arbitrary objects."""
    bad_shape = BadShape()
    node = LogicalNode(id="n1", op_type="CustomOp", shape_metadata=bad_shape)
    assert node.shape_metadata is bad_shape

    arb = ArbitraryObject()
    node2 = LogicalNode(id="n2", op_type="CustomOp", shape_metadata=arb)
    assert node2.shape_metadata is arb


def test_logical_node_duck_typed_shape() -> None:
    """Test LogicalNode initialization with duck-typed MockMeta and DummyShape."""
    meta_list = MockMeta(shape=[3, 224, 224])
    node1 = LogicalNode(id="n1", op_type="Conv", shape_metadata=meta_list)
    assert node1.shape_metadata == (3, 224, 224)

    meta_tuple = MockMeta(shape=(1, 10))
    node2 = LogicalNode(id="n2", op_type="Linear", shape_metadata=meta_tuple)
    assert node2.shape_metadata == (1, 10)

    meta_iter = MockMeta(shape=range(3))
    node_iter = LogicalNode(id="n_iter", op_type="Linear", shape_metadata=meta_iter)
    assert node_iter.shape_metadata == (0, 1, 2)

    dummy = DummyShape(shape=512)
    node3 = LogicalNode(id="n3", op_type="Embedding", shape_metadata=dummy)
    assert node3.shape_metadata == 512


def test_logical_node_dynamic_string_dimensions() -> None:
    """Test LogicalNode initialization with dynamic string dimensions."""
    dynamic_shape = ("B", "T", 128)
    node = LogicalNode(id="n1", op_type="Attention", shape_metadata=dynamic_shape)
    assert node.shape_metadata == ("B", "T", 128)


def test_logical_node_standard_tuples_and_lists() -> None:
    """Test LogicalNode initialization with standard integer tuples and lists."""
    node_tuple = LogicalNode(id="n1", op_type="Relu", shape_metadata=(16, 32))
    assert node_tuple.shape_metadata == (16, 32)

    node_list = LogicalNode(id="n2", op_type="Relu", shape_metadata=[16, 32])
    assert node_list.shape_metadata == (16, 32)


def test_logical_node_scalar_and_none_shapes() -> None:
    """Test LogicalNode initialization with scalar shapes and None."""
    node_empty = LogicalNode(id="n1", op_type="Constant", shape_metadata=())
    assert node_empty.shape_metadata == ()

    node_none = LogicalNode(id="n2", op_type="Constant", shape_metadata=None)
    assert node_none.shape_metadata is None
