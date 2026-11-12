"""Tests for export_to_python CST code generator and execution roundtrips."""

from __future__ import annotations

import pytest

import ml_switcheroo_ir as sw_ir
from ml_switcheroo_ir.export import (
    _format_shape_tuple,
    _format_sym_node,
    _format_tensor_spec,
    export_to_python,
)


def _eval_exported_code(code: str, fn_name: str = "build_graph") -> sw_ir.LogicalGraph:
    """Load exported Python code as a temporary module and invoke its builder function.

    Args:
        code (str): Generated Python source code.
        fn_name (str): Builder function name to invoke.

    Returns:
        sw_ir.LogicalGraph: Reconstructed graph.
    """
    import runpy
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8") as tmp:
        tmp.write(code)
        tmp.flush()
        ns = runpy.run_path(tmp.name)
        builder = ns[fn_name]
        assert callable(builder)
        graph = builder()
        assert isinstance(graph, sw_ir.LogicalGraph)
        return graph


def test_export_to_python_basic_roundtrip() -> None:
    """Test exporting a basic LogicalGraph and roundtripping via exec()."""
    graph = sw_ir.LogicalGraph(
        name="LinearModel",
        inputs=["x"],
        input_specs={
            "x": sw_ir.TensorSpec(
                shape=(sw_ir.SymVar("B"), 128),
                dtype=sw_ir.DType.float32,
            )
        },
        outputs=["y"],
    )
    node = sw_ir.LogicalNode(
        id="fc1",
        op_type="MatMul",
        domain="ai.onnx",
        version=1,
        inputs=["x"],
        outputs=["y"],
        attributes={"transB": 1},
        shape_metadata=(sw_ir.SymVar("B"), 64),
        dtype=sw_ir.DType.float32,
        sharding=sw_ir.PartitionSpec(axes=("data", None)),
        output_specs=[
            sw_ir.TensorSpec(
                shape=(sw_ir.SymVar("B"), 64),
                dtype=sw_ir.DType.float32,
            )
        ],
        device="cuda:0",
        stream="main",
    )
    graph.nodes["fc1"] = node

    code = export_to_python(graph, function_name="create_linear_graph", standalone=True)
    assert "def create_linear_graph() -> sw_ir.LogicalGraph:" in code
    assert 'if __name__ == "__main__":' in code

    reconstructed = _eval_exported_code(code, "create_linear_graph")

    assert isinstance(reconstructed, sw_ir.LogicalGraph)
    assert reconstructed.name == graph.name
    assert reconstructed.inputs == graph.inputs
    assert reconstructed.outputs == graph.outputs
    assert reconstructed.nodes["fc1"] == graph.nodes["fc1"]
    assert reconstructed == graph


def test_export_to_python_default_node_fields() -> None:
    """Test exporting LogicalNode with default None/empty fields."""
    graph = sw_ir.LogicalGraph(name="DefaultFieldsGraph")
    node = sw_ir.LogicalNode(id="default_node", op_type="Relu")
    graph.nodes["default_node"] = node

    code = export_to_python(graph)
    assert "sharding=None" in code
    assert "dtype=None" in code
    assert "output_specs=[]" in code

    reconstructed = _eval_exported_code(code, "build_graph")
    assert reconstructed.nodes["default_node"].sharding is None
    assert reconstructed.nodes["default_node"].dtype is None
    assert reconstructed.nodes["default_node"].output_specs == []
    assert reconstructed == graph


def test_export_to_python_empty_graph() -> None:
    """Test exporting an empty LogicalGraph with default settings."""
    graph = sw_ir.LogicalGraph(name="EmptyGraph")
    code = export_to_python(graph)
    assert "def build_graph() -> sw_ir.LogicalGraph:" in code
    assert '__name__ == "__main__"' not in code

    reconstructed = _eval_exported_code(code, "build_graph")
    assert reconstructed == graph


def test_export_to_python_autodiff_sentinels() -> None:
    """Test exporting ZeroTangent and NoTangent sentinel nodes."""
    graph = sw_ir.LogicalGraph(name="AutodiffGraph")
    zero_t = sw_ir.ZeroTangent(
        id="zero_grad",
        shape_metadata=(sw_ir.SymVar("B"), 32),
        dtype=sw_ir.DType.float32,
    )
    no_t = sw_ir.NoTangent(id="no_grad")
    graph.nodes["zero_grad"] = zero_t
    graph.nodes["no_grad"] = no_t

    code = export_to_python(graph, function_name="build_ad_graph")
    assert "sw_ir.ZeroTangent(" in code
    assert "sw_ir.NoTangent(" in code

    reconstructed = _eval_exported_code(code, "build_ad_graph")

    assert isinstance(reconstructed.nodes["zero_grad"], sw_ir.ZeroTangent)
    assert isinstance(reconstructed.nodes["no_grad"], sw_ir.NoTangent)
    assert reconstructed == graph


def test_export_to_python_distributed_topologies() -> None:
    """Test exporting LogicalGraph with Mesh, WebRTC, and Pipeline topologies."""
    webrtc = sw_ir.WebRTCSignalingTopology(
        signaling_url="wss://broker.ai:443",
        ice_servers=["stun:stun.l.google.com:19302"],
        peers=[
            sw_ir.WebRTCPeerConfig(
                peer_id="node-0",
                device_capability="webgpu",
                network_transport="data_channel",
            )
        ],
    )
    mesh = sw_ir.LogicalMesh(shape={"data": 2, "model": 4}, webrtc_topology=webrtc)

    pipeline = sw_ir.PipelineTopologyConfig(
        microbatch_splitting=sw_ir.MicrobatchSplittingConfig(strategy="adaptive", num_microbatches=8),
        mesh_mapping=sw_ir.MeshMappingConfig(devices_per_stage=2),
        stage_communication=sw_ir.StageCommunicationConfig(protocol="webrtc"),
        dependencies=[sw_ir.DependencyConfig(source_stage="stage_0", target_stage="stage_1", offset_mb=0)],
        schedule=sw_ir.PipelineScheduleConfig(
            phases=[
                sw_ir.SchedulePhaseConfig(
                    type="1F1B",
                    operations=["fwd", "bwd"],
                    count_expression="num_mb",
                )
            ]
        ),
    )

    graph = sw_ir.LogicalGraph(
        name="DistributedPipeline",
        mesh=mesh,
        pipeline_topology=pipeline,
    )

    code = export_to_python(graph, function_name="build_distributed")
    assert "pipeline_topology = sw_ir.PipelineTopologyConfig.from_dict(" in code
    assert "webrtc_topology = sw_ir.WebRTCSignalingTopology.from_dict(" in code
    assert "mesh = sw_ir.LogicalMesh(" in code

    reconstructed = _eval_exported_code(code, "build_distributed")
    assert reconstructed == graph
    assert reconstructed.mesh is not None
    assert reconstructed.mesh.webrtc_topology is not None
    assert reconstructed.mesh.webrtc_topology.peers[0].peer_id == webrtc.peers[0].peer_id
    assert reconstructed.pipeline_topology is not None
    assert reconstructed.pipeline_topology.microbatch_splitting.num_microbatches == 8


def test_export_to_python_sym_nodes() -> None:
    """Test formatting and evaluating various SymNode algebraic trees."""
    var_b = sw_ir.SymVar("B")
    var_t = sw_ir.SymVar("T")
    c2 = sw_ir.SymConst(2)
    bin_op = sw_ir.SymBinaryOp("+", var_b, c2)
    un_op = sw_ir.SymUnaryOp("-", var_t)
    sym_int = sw_ir.SymInt(var_b)
    piecewise = sw_ir.SymPiecewise(
        cases=[("B > 0", var_b)],
        default=c2,
    )

    assert _format_sym_node(var_b) == "sw_ir.SymVar('B')"
    assert _format_sym_node(c2) == "sw_ir.SymConst(2)"
    assert _format_sym_node(bin_op) == "sw_ir.SymBinaryOp('+', sw_ir.SymVar('B'), sw_ir.SymConst(2))"
    assert _format_sym_node(un_op) == "sw_ir.SymUnaryOp('-', sw_ir.SymVar('T'))"
    assert _format_sym_node(sym_int) == "sw_ir.SymInt(sw_ir.SymVar('B'))"
    assert (
        _format_sym_node(piecewise)
        == "sw_ir.SymPiecewise(cases=[('B > 0', sw_ir.SymVar('B'))], default=sw_ir.SymConst(2))"
    )

    # Test single-dimension shape tuple formatting
    assert _format_shape_tuple(None) == "None"
    assert _format_shape_tuple([var_b]) == "(sw_ir.SymVar('B'),)"
    assert _format_shape_tuple([var_b, 64]) == "(sw_ir.SymVar('B'), 64)"

    # Test tensor spec formatting
    spec = sw_ir.TensorSpec(shape=(var_b, 128), dtype=sw_ir.DType.float32)
    formatted = _format_tensor_spec(spec)
    assert "sw_ir.TensorSpec(" in formatted
    assert "sw_ir.DType.float32" in formatted


def test_export_to_python_mesh_without_webrtc() -> None:
    """Test exporting LogicalGraph with LogicalMesh having no WebRTC topology."""
    graph = sw_ir.LogicalGraph(
        name="MeshNoWebRTC",
        mesh=sw_ir.LogicalMesh(shape={"data": 4}),
    )
    code = export_to_python(graph)
    assert 'mesh = sw_ir.LogicalMesh(shape={"data": 4})' in code
    assert _eval_exported_code(code, "build_graph") == graph


def test_export_to_python_subgraphs_and_missing_black(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test exporting LogicalGraph with nested subgraphs and when black is unavailable.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import sys

    sub = sw_ir.LogicalGraph(
        name="BodySubgraph",
        nodes={"sub_relu": sw_ir.LogicalNode(id="sub_relu", op_type="Relu")},
        outputs=["sub_relu"],
    )
    parent_node = sw_ir.LogicalNode(
        id="scan_op",
        op_type="Scan",
        subgraphs={"body": sub, "meta": "custom_tag"},
    )
    graph = sw_ir.LogicalGraph(
        name="ParentWithSubgraph",
        nodes={"scan_op": parent_node},
        outputs=["scan_op"],
    )

    code = export_to_python(graph)
    assert "sw_ir.LogicalGraph(" in code
    assert "LogicalGraph.from_dict(" not in code
    reconstructed = _eval_exported_code(code, "build_graph")
    body_sub = reconstructed.nodes["scan_op"].subgraphs["body"]
    assert isinstance(body_sub, sw_ir.LogicalGraph)
    assert body_sub.name == "BodySubgraph"
    assert reconstructed.nodes["scan_op"].subgraphs["meta"] == "custom_tag"

    monkeypatch.setitem(sys.modules, "black", None)
    unformatted_code = export_to_python(graph)
    assert "def build_graph() -> sw_ir.LogicalGraph:" in unformatted_code


def test_export_to_python_special_characters_and_sanitization() -> None:
    """Verify export_to_python handles special characters in node IDs, attributes, and function names."""
    special_node = sw_ir.LogicalNode(
        id="node/0:out'\"\\",
        op_type="Custom",
        attributes={"z-key'\"": "val\nwith\nnewlines", "a$key": 42},
    )
    graph = sw_ir.LogicalGraph(
        name='Model"""WithQuotes\nAndNewlines',
        nodes={special_node.id: special_node},
        outputs=[special_node.id],
    )

    code_digit = export_to_python(graph, function_name="123-my.func!")
    reconstructed = _eval_exported_code(code_digit, "_123_my_func")
    assert reconstructed == graph

    code_kw = export_to_python(sw_ir.LogicalGraph(name=""), function_name="return")
    assert "def return_() -> sw_ir.LogicalGraph:" in code_kw

    code_empty = export_to_python(graph, function_name="!!!")
    assert "def build_graph() -> sw_ir.LogicalGraph:" in code_empty


def test_export_to_python_sentinels_tuples_and_rich_nested_subgraph() -> None:
    """Verify export_to_python preserves all Sentinel fields, tuple/scalar shapes, and rich nested subgraphs."""
    webrtc = sw_ir.WebRTCSignalingTopology(
        signaling_url="wss://sub.local",
        peers=[sw_ir.WebRTCPeerConfig(peer_id="p0")],
    )
    pipe = sw_ir.PipelineTopologyConfig(
        microbatch_splitting=sw_ir.MicrobatchSplittingConfig(num_microbatches=2),
        mesh_mapping=sw_ir.MeshMappingConfig(devices_per_stage=1),
        stage_communication=sw_ir.StageCommunicationConfig(protocol="nccl"),
    )
    sub_in = sw_ir.LogicalNode(id="s_in", op_type="Input", shape_metadata=(), dtype=sw_ir.DType.float32)
    sub_out = sw_ir.LogicalNode(id="s_out", op_type="Relu", shape_metadata=(4,))
    nested_with_webrtc = sw_ir.LogicalGraph(
        name="RichSub",
        nodes={"s_in": sub_in, "s_out": sub_out},
        outputs=["s_out"],
        initializers={"sub_w": {"buf": b"\x10\x20", "arr": [1, 2]}},
        mesh=sw_ir.LogicalMesh(shape={"d": 1}, webrtc_topology=webrtc),
        pipeline_topology=pipe,
        attributes={"t0": (), "t1": (1,), "t2": (1, 2)},
        loop_options={"iters": 3},
        edges=[sw_ir.LogicalEdge("s_in", "s_out", value_name="s_edge")],
    )
    nested_plain_mesh = sw_ir.LogicalGraph(
        name="PlainMeshSub",
        nodes={"m": sw_ir.LogicalNode(id="m", op_type="Relu")},
        outputs=["m"],
        mesh=sw_ir.LogicalMesh(shape={"tp": 2}),
    )

    pw = sw_ir.SymPiecewise(
        cases=[
            (sw_ir.SymBinaryOp(">", sw_ir.SymVar("B"), sw_ir.SymConst(1)), sw_ir.SymVar("B")),
            ("B == 1", sw_ir.SymConst(1)),
        ],
        default=sw_ir.SymConst(2),
    )
    zt = sw_ir.ZeroTangent(
        id="zt_full",
        shape=(sw_ir.SymUnaryOp("abs", sw_ir.SymVar("B")), pw),
        dtype=sw_ir.DType.bfloat16,
        version=2,
        attributes={
            "dt": sw_ir.DType.float16,
            "sym": sw_ir.SymInt("B + 1"),
            "tshape": sw_ir.TensorShape((2, 4)),
            "tspec": sw_ir.TensorSpec(shape=(2,), dtype=sw_ir.DType.int32),
            "raw": b"abc",
        },
        inputs=["ext_in"],
        outputs=["zt_ssa"],
        source_ast_ref="model.py:10",
        sharding=sw_ir.PartitionSpec(axes=("d", None)),
        output_specs=[sw_ir.TensorSpec(shape=(2, 4), dtype=sw_ir.DType.bfloat16, sparsity="2:4")],
        subgraphs={"sub1": nested_with_webrtc, "sub2": nested_plain_mesh},
        device="cuda:0",
        stream="s0",
    )
    nt = sw_ir.NoTangent(
        id="nt_full",
        version=3,
        attributes={"note": "detached"},
        inputs=["zt_ssa"],
        outputs=["nt_ssa"],
        shape_metadata=(),
        source_ast_ref="model.py:11",
        sharding=sw_ir.PartitionSpec(axes=(None,)),
        dtype=sw_ir.DType.float32,
        device="cpu",
        stream="s1",
    )
    top = sw_ir.LogicalGraph(
        name="TopSentinelExport",
        nodes={"zt_full": zt, "nt_full": nt},
        outputs=["nt_ssa"],
        initializers={"top_w": [1.0, 2.0]},
        edges=[sw_ir.LogicalEdge("zt_full", "nt_full", value_name="cotangent_edge")],
    )
    assert (top == 123) is False
    code = export_to_python(top)
    reconstructed = _eval_exported_code(code)
    assert isinstance(reconstructed.nodes["zt_full"], sw_ir.ZeroTangent)
    assert isinstance(reconstructed.nodes["nt_full"], sw_ir.NoTangent)
    assert reconstructed.nodes["zt_full"] == zt
    assert reconstructed.nodes["nt_full"] == nt
    assert reconstructed.edges == top.edges
