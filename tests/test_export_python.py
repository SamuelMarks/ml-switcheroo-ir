"""Tests for export_to_python CST code generator and execution roundtrips."""

from __future__ import annotations

import ml_switcheroo_ir as sw_ir
from ml_switcheroo_ir.export import (
    _format_shape_tuple,
    _format_sym_node,
    _format_tensor_spec,
    export_to_python,
)


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

    scope: dict[str, object] = {}
    exec(code, scope)  # noqa: S102
    builder_fn = scope["create_linear_graph"]
    assert callable(builder_fn)
    reconstructed = builder_fn()

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

    scope: dict[str, object] = {}
    exec(code, scope)  # noqa: S102
    builder = scope["build_graph"]
    assert callable(builder)
    reconstructed = builder()
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

    scope: dict[str, object] = {}
    exec(code, scope)  # noqa: S102
    builder = scope["build_graph"]
    assert callable(builder)
    reconstructed = builder()
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

    scope: dict[str, object] = {}
    exec(code, scope)  # noqa: S102
    builder = scope["build_ad_graph"]
    assert callable(builder)
    reconstructed = builder()

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
        microbatch_splitting=sw_ir.MicrobatchSplittingConfig(
            strategy="adaptive", num_microbatches=8
        ),
        mesh_mapping=sw_ir.MeshMappingConfig(devices_per_stage=2),
        stage_communication=sw_ir.StageCommunicationConfig(protocol="webrtc"),
        dependencies=[
            sw_ir.DependencyConfig(
                source_stage="stage_0", target_stage="stage_1", offset_mb=0
            )
        ],
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

    scope: dict[str, object] = {}
    exec(code, scope)  # noqa: S102
    builder = scope["build_distributed"]
    assert callable(builder)
    reconstructed = builder()
    assert reconstructed == graph
    assert reconstructed.mesh is not None
    assert reconstructed.mesh.webrtc_topology is not None
    assert (
        reconstructed.mesh.webrtc_topology.peers[0].peer_id == webrtc.peers[0].peer_id
    )
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
    assert (
        _format_sym_node(bin_op)
        == "sw_ir.SymBinaryOp('+', sw_ir.SymVar('B'), sw_ir.SymConst(2))"
    )
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
    scope: dict[str, object] = {}
    exec(code, scope)  # noqa: S102
    builder = scope["build_graph"]
    assert callable(builder)
    assert builder() == graph
