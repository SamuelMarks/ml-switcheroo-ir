"""Unit tests for canonical LogicalNode, LogicalGraph, and streaming parser constructors."""

from __future__ import annotations

import io
import json

import pytest

from ml_switcheroo_ir import (
    LogicalEdge,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
    TensorShape,
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
    assert node.attributes == {"kernel_size": 3, "stride": 1}
    assert node.outputs == ["conv1"]


def test_logical_node_empty_id_or_op_type() -> None:
    """Test LogicalNode constructor raises ValueError when id or op_type is empty."""
    with pytest.raises(ValueError, match="LogicalNode 'id' must be a non-empty string"):
        LogicalNode(id="", op_type="Relu")
    with pytest.raises(
        ValueError, match="LogicalNode 'op_type' must be a non-empty string"
    ):
        LogicalNode(id="n1", op_type="")


def test_logical_node_setters() -> None:
    """Test op_type and attributes assignment on LogicalNode."""
    node = LogicalNode(id="n1", op_type="Relu")
    node.op_type = "Gelu"
    assert node.op_type == "Gelu"

    node.attributes = {"approximate": "tanh"}
    assert node.attributes == {"approximate": "tanh"}


def test_logical_node_model_dump_and_repr() -> None:
    """Test that Pydantic model_dump and repr function correctly on LogicalNode."""
    spec = PartitionSpec(axes=("data", None))
    node = LogicalNode(
        id="n1",
        op_type="MatMul",
        shape_metadata=(128, 256),
        sharding=spec,
        source_ast_ref="test.py:10",
    )
    d = node.model_dump()
    assert d["id"] == "n1"
    assert d["op_type"] == "MatMul"
    assert d["outputs"] == ["n1"]
    assert "id='n1'" in repr(node)
    assert "op_type='MatMul'" in repr(node)


def test_logical_graph_nodes_as_dict() -> None:
    """Test LogicalGraph initialized with a dictionary of nodes."""
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Relu", inputs=["n1"])
    graph = LogicalGraph(name="DictGraph", nodes={"n1": n1, "n2": n2})
    assert len(graph.nodes) == 2
    assert graph.nodes["n1"] == n1
    assert graph.outputs == ["n2"]


def test_logical_graph_rejects_nodes_as_list() -> None:
    """Test LogicalGraph raises TypeError when initialized with a list of nodes."""
    n1 = LogicalNode(id="n1", op_type="Input")
    with pytest.raises(TypeError, match="must be a Mapping"):
        LogicalGraph(name="ListGraph", nodes=[n1])  # type: ignore[arg-type]


def test_logical_graph_explicit_edges() -> None:
    """Test LogicalGraph initialized with explicit edges in constructor."""
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Relu")
    edges = [
        LogicalEdge(source="n1", target="n2"),
        LogicalEdge(source="n1", target="n2"),
        LogicalEdge(source="n1", target="non_existent"),
    ]
    graph = LogicalGraph(name="EdgeGraph", nodes={"n1": n1, "n2": n2}, edges=edges)
    assert n2.inputs == ["n1"]
    assert graph.edges == [LogicalEdge(source="n1", target="n2")]
    assert graph.outputs == ["n2"]


def test_logical_graph_outputs_deduction_and_override() -> None:
    """Test deduction of graph outputs and explicit override."""
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Relu", inputs=["n1"])
    graph_explicit = LogicalGraph(
        name="ExplicitOutputs", nodes={"n1": n1, "n2": n2}, outputs=["custom_out"]
    )
    assert graph_explicit.outputs == ["custom_out"]

    graph_empty = LogicalGraph()
    assert graph_empty.outputs == []

    c1 = LogicalNode(id="c1", op_type="Op", inputs=["c2"])
    c2 = LogicalNode(id="c2", op_type="Op", inputs=["c1"])
    graph_cycle = LogicalGraph(nodes={"c1": c1, "c2": c2})
    assert graph_cycle.outputs == []

    split = LogicalNode(id="split", op_type="Split", outputs=["s0", "s1"])
    c_s0 = LogicalNode(id="c_s0", op_type="Relu", inputs=["s0"])
    graph_multi = LogicalGraph(nodes={"split": split, "c_s0": c_s0})
    assert "c_s0" in graph_multi.outputs

    in_n = LogicalNode(id="in_n", op_type="Input")
    out_n = LogicalNode(id="out_n", op_type="Output", inputs=["in_n", "in_n"])
    graph_with_output_op = LogicalGraph(nodes={"in_n": in_n, "out_n": out_n})
    assert graph_with_output_op.outputs == ["in_n"]


def test_logical_graph_roundtrip_instantiation() -> None:
    """Test roundtrip construction using canonical LogicalNode and LogicalGraph syntax."""
    mesh = LogicalMesh(shape={"data": 2, "model": 4})
    node1 = LogicalNode(
        id="x",
        op_type="Input",
        domain="ai.onnx",
        version=17,
        attributes={"dtype": "float32"},
        sharding=PartitionSpec(axes=("data", None, "model")),
    )
    node2 = LogicalNode(
        id="norm1",
        op_type="RMSNorm",
        domain="ml.switcheroo.custom",
        version=1,
        attributes={"eps": 1e-6},
    )
    graph = LogicalGraph(
        name="TransformerLayer",
        mesh=mesh,
        nodes={"x": node1, "norm1": node2},
        edges=[LogicalEdge("x", "norm1")],
    )

    assert graph.name == "TransformerLayer"
    assert graph.nodes["norm1"].inputs == ["x"]
    assert graph.outputs == ["norm1"]
    assert graph.edges == [LogicalEdge(source="x", target="norm1")]


def test_logical_node_rejects_invalid_shape_metadata() -> None:
    """Test that LogicalNode rejects arbitrary objects and invalid shape metadata."""

    class BadShape:
        """A class simulating an invalid shape."""

    with pytest.raises(TypeError, match="not a valid DimensionType"):
        LogicalNode(id="n4", op_type="Linear", shape_metadata=BadShape())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="not a valid DimensionType"):
        LogicalNode(id="n5", op_type="Custom", shape_metadata=object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Invalid shape_metadata"):
        LogicalNode(id="n6", op_type="Custom", shape_metadata="invalid_str")


def test_logical_node_shape_from_tensor_shape_and_iterables() -> None:
    """Test LogicalNode initialization with TensorShape, tuples, lists, and ranges."""
    node1 = LogicalNode(
        id="n1", op_type="Conv", shape_metadata=TensorShape((3, 224, 224))
    )
    assert node1.shape_metadata == (3, 224, 224)

    node2 = LogicalNode(id="n2", op_type="Linear", shape_metadata=(1, 10))
    assert node2.shape_metadata == (1, 10)

    node_iter = LogicalNode(id="n_iter", op_type="Linear", shape_metadata=range(3))
    assert node_iter.shape_metadata == (0, 1, 2)


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

    node_zero_out = LogicalNode(id="sink", op_type="Barrier", outputs=[])
    assert node_zero_out.outputs == []


def test_stream_parse_graph_chunked_boundaries() -> None:
    """Test chunked incremental streaming JSON parser with small chunk boundaries and error paths."""
    from ml_switcheroo_ir.graph import _decode_next_json_value, _stream_parse_graph

    raw_dict_nodes = (
        '{"name": "ChunkedModel", "nodes": {"n1": {"id": "n1", "op_type": "Relu"}, '
        '"n2": {"id": "n2", "op_type": "Add", "inputs": []}}, '
        '"edges": [{"source": "n1", "target": "n2"}], "outputs": ["n2"]}'
    )
    g1 = _stream_parse_graph(io.StringIO(raw_dict_nodes), LogicalGraph, chunk_size=8)
    assert g1.name == "ChunkedModel"
    assert list(g1.nodes.keys()) == ["n1", "n2"]
    assert g1.nodes["n2"].inputs == ["n1"]

    raw_bytes_nodes = b'{"name": "BytesModel", "nodes": {"n1": {"id": "n1", "op_type": "Relu"}}, "outputs": ["n1"]}'
    g2 = _stream_parse_graph(io.BytesIO(raw_bytes_nodes), LogicalGraph, chunk_size=7)
    assert g2.name == "BytesModel"
    assert "n1" in g2.nodes

    with pytest.raises(json.JSONDecodeError, match="Expected top-level JSON object"):
        _stream_parse_graph(io.StringIO("   [1, 2]"), LogicalGraph, chunk_size=4)

    with pytest.raises(
        json.JSONDecodeError, match="Unterminated top-level JSON object"
    ):
        _stream_parse_graph(io.StringIO('{"name": "X"'), LogicalGraph, chunk_size=4)

    with pytest.raises(json.JSONDecodeError, match=r"Expected '\{' for nodes section"):
        _stream_parse_graph(io.StringIO('{"nodes": 123}'), LogicalGraph, chunk_size=4)

    with pytest.raises(json.JSONDecodeError, match="Unterminated nodes object"):
        _stream_parse_graph(
            io.StringIO('{"nodes": {"n1": {"id": "n1", "op_type": "Relu"}'),
            LogicalGraph,
            chunk_size=8,
        )

    with pytest.raises(json.JSONDecodeError):
        _stream_parse_graph(
            io.StringIO('{"name": "unterminated_str'),
            LogicalGraph,
            chunk_size=4,
        )

    dec = json.JSONDecoder()
    val_split, _, _ = _decode_next_json_value(dec, "12", 0, io.StringIO("345}"), 4)
    assert val_split == 12345

    val_eof, _, _ = _decode_next_json_value(dec, "123", 0, io.StringIO(""), 3)
    assert val_eof == 123
