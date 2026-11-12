"""Tests for canonical serialization and deserialization in LogicalGraph."""

from __future__ import annotations

import json

import pytest

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


def test_from_json_rejects_nodes_as_list() -> None:
    """Test that from_json rejects legacy list-based node layout."""
    data = {
        "name": "ListModel",
        "nodes": [
            {
                "id": "node_a",
                "op_type": "Relu",
                "attributes": {"test_attr": 42},
                "inputs": ["node_input"],
            }
        ],
    }
    with pytest.raises(TypeError, match="expected 'nodes' to be a Mapping"):
        LogicalGraph.from_json(json.dumps(data))


def test_from_json_with_explicit_edges() -> None:
    """Test parsing explicit edges array and populating node inputs."""
    data = {
        "name": "EdgeModel",
        "nodes": {
            "a": {"id": "a", "op_type": "Input"},
            "b": {"id": "b", "op_type": "Relu"},
        },
        "edges": [
            {"source": "a", "target": "b"},
            {
                "source": "a",
                "target": "b",
            },
            {"source": "unknown", "target": "unknown_target"},
        ],
    }
    graph = LogicalGraph.from_json(json.dumps(data))
    assert graph.nodes["b"].inputs == ["a"]
    assert graph.edges == (LogicalEdge(source="a", target="b"),)


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
    graph = LogicalGraph(name="DeterministicGraph", nodes={"n1": n1, "n2": n2}, mesh=mesh)

    json1 = graph.to_json(indent=2)
    json2 = graph.to_json(indent=2)
    assert json1 == json2
    assert "edges" in json.loads(json1)
    assert "edges" in graph.to_dict()


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


def test_stream_parser_rejects_non_mapping_node_entry() -> None:
    """Verify streaming parser raises ValueError when a node entry in 'nodes' is not a JSON object."""
    import io

    bad_stream = io.StringIO('{"name": "Bad", "nodes": {"n1": 123}}')
    with pytest.raises(ValueError, match="Expected JSON object mapping for node 'n1'"):
        LogicalGraph.from_json(bad_stream)


def test_lossless_symbolic_piecewise_and_error_paths() -> None:
    """Test SymUnaryOp, SymPiecewise, untagged dim expressions, empty streaming, and malformed SymNode errors."""
    import io

    from ml_switcheroo_ir.serialization import (
        RawGraphValue,
        _coerce_attribute_value,
        _coerce_initializer_value,
        _decode_symint_mapping,
        _decode_symnode,
        _encode_special_types,
        _encode_sym_condition,
        _encode_symnode,
        _parse_raw_dim_entry,
    )
    from ml_switcheroo_ir.shapes import SymBinaryOp, SymConst, SymInt, SymNode, SymPiecewise, SymUnaryOp, SymVar
    from ml_switcheroo_ir.types import DType, TensorShape, TensorSpec

    pw = SymPiecewise(
        cases=[
            (SymBinaryOp(">", SymVar("N"), SymConst(0)), SymUnaryOp("abs", SymVar("N"))),
            (True, SymConst(2)),
            ("N == 0", SymConst(1)),
        ],
        default=SymConst(4),
    )
    node = LogicalNode(
        id="op1",
        op_type="Custom",
        shape_metadata=(pw, SymInt("N + 2")),
        attributes={
            "raw_bytes": b"\x00\xff\x10",
            "dt": DType.int4,
            "tshape": TensorShape((SymVar("N"), 8)),
            "tspec": TensorSpec(shape=[SymInt("N"), 8], dtype=DType.float16, sparsity="2:4"),
        },
    )
    g = LogicalGraph(
        name="RichTest",
        nodes={"op1": node},
        initializers={"blob": b"\x01\x02", "nested": {"raw": b"\x03"}, "arr": [1, 2]},
        attributes={"pw": pw},
        loop_options={"max_k": SymInt(8)},
    )
    restored = LogicalGraph.from_json(g.to_json())
    assert restored.nodes["op1"].attributes["raw_bytes"] == b"\x00\xff\x10"
    assert restored.nodes["op1"].attributes["dt"] is DType.int4
    assert restored.initializers["blob"] == b"\x01\x02"
    assert restored.initializers["nested"] == {"raw": b"\x03"}
    assert restored.attributes["pw"] == pw
    assert restored.loop_options == {"max_k": SymInt(8)}

    # Empty graph streaming covers empty edges, initializers, and nodes sections
    empty_g = LogicalGraph(name="Empty")
    assert LogicalGraph.from_json(empty_g.to_json()).nodes == {}

    # Fallback and malformed branches in symnode/dim/initializer codecs
    cond_enc = _encode_sym_condition(lambda env: True)
    assert isinstance(cond_enc, str) and cond_enc.startswith("<function")
    base_encoded = _encode_symnode(SymNode())
    assert _decode_symnode({"__sw_type__": "SymNode", "expr": "N + 1"}) == SymVar("N") + 1
    assert _decode_symint_mapping({"__sw_type__": "SymInt", "expr": "N * 2"}) == SymInt("N * 2")
    assert _parse_raw_dim_entry(SymVar("K")) == SymVar("K")
    assert _parse_raw_dim_entry("N + 1") == SymVar("N") + 1
    assert _parse_raw_dim_entry("bad$dim") == "bad$dim"
    assert isinstance(_parse_raw_dim_entry({"untagged": 1}), str)
    assert _encode_special_types(b"hi") == {"__sw_type__": "bytes", "data": "aGk="}
    assert _encode_special_types(TensorShape((2,))) == "dims=(2,)"
    g_skip_spec = LogicalGraph.from_dict({"nodes": {"n": {"id": "n", "op_type": "Relu", "output_specs": [123]}}})
    assert g_skip_spec.nodes["n"].output_specs == []
    enc_k = _encode_special_types(SymInt("K"))
    assert isinstance(enc_k, dict) and enc_k["__sw_type__"] == "SymInt"
    assert _coerce_initializer_value(None) == ""
    assert isinstance(_coerce_attribute_value(LogicalMesh(shape={"d": 2})), str)

    bad_syms: list[dict[str, RawGraphValue]] = [
        {"__sw_type__": "SymNode", "kind": "const", "value": True},
        {"__sw_type__": "SymNode", "kind": "binary_op", "op": "+", "left": 1, "right": 2},
        {"__sw_type__": "SymNode", "kind": "unary_op", "op": "abs", "operand": 1},
        {"__sw_type__": "SymNode", "kind": "piecewise", "cases": [], "default": 1},
        {"__sw_type__": "SymNode"},
    ]
    for bad_sym in bad_syms:
        with pytest.raises(ValueError):
            _decode_symnode(bad_sym)
    assert base_encoded["__sw_type__"] == "SymNode"
    assert isinstance(io.StringIO(), io.StringIO)
