"""Tests for LogicalNode.clone and LogicalGraph.clone ergonomics."""

from __future__ import annotations

from ml_switcheroo_ir import (
    DType,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    NoTangent,
    PartitionSpec,
    TensorSpec,
    ZeroTangent,
)


def test_node_clone_defaults_and_isolation() -> None:
    """Test that LogicalNode.clone performs a deep copy of all mutable collections."""
    orig = LogicalNode(
        id="node1",
        op_type="Conv",
        domain="ai.onnx",
        version=1,
        attributes={"kernel_shape": [3, 3], "strides": [1, 1]},
        inputs=["in1", "in2"],
        outputs=["out1", "out2"],
        shape_metadata=(1, 32, 64, 64),
        source_ast_ref="test.py:10:ast_1",
        sharding=PartitionSpec(axes=("data", "model")),
        dtype=DType.float32,
        output_specs=[TensorSpec(shape=(1, 32, 64, 64), dtype=DType.float32)],
        subgraphs={"body": LogicalGraph(name="Inner")},
        device="cuda:0",
        stream="compute_0",
    )

    cloned = orig.clone()

    # Structural equality
    assert cloned.id == orig.id
    assert cloned.op_type == orig.op_type
    assert cloned.domain == orig.domain
    assert cloned.version == orig.version
    assert cloned.attributes == orig.attributes
    assert cloned.inputs == orig.inputs
    assert cloned.outputs == orig.outputs
    assert cloned.shape_metadata == orig.shape_metadata
    assert cloned.source_ast_ref == orig.source_ast_ref
    assert cloned.sharding == orig.sharding
    assert cloned.dtype == orig.dtype
    assert cloned.device == orig.device
    assert cloned.stream == orig.stream

    # Deep copy isolation: mutating cloned must not affect orig
    cloned.attributes["new_key"] = "new_val"
    assert "new_key" not in orig.attributes
    assert orig.attributes["kernel_shape"] == [3, 3]

    cloned.inputs.append("in3")
    assert orig.inputs == ["in1", "in2"]

    cloned.outputs.append("out3")
    assert orig.outputs == ["out1", "out2"]

    cloned.output_specs[0].shape = (2, 64)
    assert orig.output_specs[0].shape == (1, 32, 64, 64)

    cloned.subgraphs["body"].name = "MutatedInner"
    assert orig.subgraphs["body"].name == "Inner"


def test_node_clone_partial_overrides() -> None:
    """Test keyword overrides during LogicalNode.clone."""
    orig = LogicalNode(
        id="conv1",
        op_type="Conv",
        domain="ai.onnx",
        version=1,
        attributes={"pads": [0, 0, 0, 0]},
        inputs=["x"],
        outputs=["y"],
    )

    cloned = orig.clone(
        id="conv1_override",
        op_type="ConvTranspose",
        domain="ai.onnx.preview",
        version=2,
        attributes={"pads": [1, 1, 1, 1]},
        inputs=["x_new"],
        outputs=["y_new"],
        shape_metadata=(4, 16),
        source_ast_ref="new.py:42",
        sharding=PartitionSpec(axes=("batch",)),
        dtype=DType.float16,
        output_specs=[TensorSpec(shape=(4, 16), dtype=DType.float16)],
        subgraphs={"new_sub": LogicalGraph(name="Sub")},
        device="cpu",
        stream="main",
        extra_attr=99,
    )

    assert cloned.id == "conv1_override"
    assert cloned.op_type == "ConvTranspose"
    assert cloned.domain == "ai.onnx.preview"
    assert cloned.version == 2
    assert cloned.attributes["pads"] == [1, 1, 1, 1]
    assert cloned.attributes["extra_attr"] == 99
    assert cloned.inputs == ["x_new"]
    assert cloned.outputs == ["y_new"]
    assert cloned.shape_metadata == (4, 16)
    assert cloned.source_ast_ref == "new.py:42"
    assert cloned.sharding is not None and cloned.sharding.axes == ("batch",)
    assert cloned.dtype == DType.float16
    assert len(cloned.output_specs) == 1
    assert cloned.device == "cpu"
    assert cloned.stream == "main"


def test_node_clone_subclass_preservation() -> None:
    """Test that cloning ZeroTangent and NoTangent preserves their types."""
    zt = ZeroTangent(id="zt0", shape=(2, 3), dtype=DType.float32)
    zt_cloned = zt.clone(id="zt1")
    assert isinstance(zt_cloned, ZeroTangent)
    assert zt_cloned.id == "zt1"
    assert zt_cloned.op_type == "ZeroTangent"
    assert zt_cloned.domain == "ml.switcheroo.ad"

    nt = NoTangent(id="nt0")
    nt_cloned = nt.clone(id="nt1")
    assert isinstance(nt_cloned, NoTangent)
    assert nt_cloned.id == "nt1"
    assert nt_cloned.op_type == "NoTangent"


def test_graph_clone_complete_isolation() -> None:
    """Test that LogicalGraph.clone produces completely isolated duplicates."""
    mesh = LogicalMesh(shape={"data": 2, "model": 4})
    n1 = LogicalNode(id="n1", op_type="Relu", inputs=["in1"], outputs=["n1_out"])
    n2 = LogicalNode(id="n2", op_type="Relu", inputs=["n1_out"], outputs=["n2_out"])
    graph = LogicalGraph(
        name="OrigModel",
        nodes={"n1": n1, "n2": n2},
        inputs=["in1"],
        input_specs={"in1": TensorSpec(shape=(8, 16), dtype=DType.float32)},
        outputs=["n2_out"],
        mesh=mesh,
        initializers={"w": [1.0, 2.0]},
    )

    cloned = graph.clone()

    assert cloned.name == "OrigModel"
    assert len(cloned.nodes) == 2
    assert cloned.inputs == ["in1"]
    assert cloned.outputs == ["n2_out"]
    assert cloned.input_specs["in1"].shape == (8, 16)

    # Mutate cloned graph nodes and inputs to ensure complete isolation
    cloned.nodes["n1"].inputs.append("extra_in")
    assert graph.nodes["n1"].inputs == ["in1"]

    cloned.inputs.append("in2")
    assert graph.inputs == ["in1"]

    cloned.outputs.append("n3_out")
    assert graph.outputs == ["n2_out"]

    cloned.initializers["w"].append(3.0)
    assert graph.initializers["w"] == [1.0, 2.0]


def test_graph_clone_with_pipeline_topology() -> None:
    """Test LogicalGraph.clone preserves pipeline_topology when present."""
    from ml_switcheroo_ir.distributed import (
        MeshMappingConfig,
        MicrobatchSplittingConfig,
        PipelineTopologyConfig,
        StageCommunicationConfig,
    )

    g = LogicalGraph(name="PipeGraph")
    topo = PipelineTopologyConfig(
        microbatch_splitting=MicrobatchSplittingConfig(num_microbatches=4),
        mesh_mapping=MeshMappingConfig(devices_per_stage=1),
        stage_communication=StageCommunicationConfig(protocol="p2p"),
    )
    g.set_pipeline_topology(topo)

    cloned = g.clone()
    assert hasattr(cloned, "pipeline_topology")
    assert cloned.pipeline_topology is not None
    assert cloned.pipeline_topology.microbatch_splitting.num_microbatches == 4
    assert cloned.pipeline_topology.mesh_mapping.devices_per_stage == 1
