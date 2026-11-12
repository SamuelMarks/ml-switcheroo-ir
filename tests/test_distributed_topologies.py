"""Tests for distributed pipeline topologies, schedules, and LogicalGraph integration."""

from __future__ import annotations

from ml_switcheroo_ir import (
    DependencyConfig,
    LogicalGraph,
    LogicalNode,
    MeshMappingConfig,
    MicrobatchSplittingConfig,
    PipelineScheduleConfig,
    PipelineTopologyConfig,
    SchedulePhaseConfig,
    StageCommunicationConfig,
)


def test_pipeline_topology_models_and_graph_integration() -> None:
    """Verify construction, validation, and LogicalGraph integration of pipeline topologies."""
    microbatch_cfg = MicrobatchSplittingConfig(
        strategy="uniform",
        num_microbatches=8,
    )
    assert microbatch_cfg.strategy == "uniform"
    assert microbatch_cfg.num_microbatches == 8

    mesh_cfg = MeshMappingConfig(devices_per_stage=2)
    assert mesh_cfg.devices_per_stage == 2

    comm_cfg = StageCommunicationConfig(protocol="nccl")
    assert comm_cfg.protocol == "nccl"

    dep1 = DependencyConfig(
        source_stage="stage_0",
        target_stage="stage_1",
        offset_mb=0,
    )
    assert dep1.source_stage == "stage_0"
    assert dep1.target_stage == "stage_1"
    assert dep1.offset_mb == 0

    phase1 = SchedulePhaseConfig(
        type="warmup",
        operations=["forward"],
        count_expression="num_stages - 1",
    )
    phase2 = SchedulePhaseConfig(
        type="1F1B",
        operations=["forward", "backward"],
        count_expression="num_microbatches - num_stages + 1",
    )
    schedule = PipelineScheduleConfig(phases=[phase1, phase2])
    assert len(schedule.phases) == 2

    pipe_topo = PipelineTopologyConfig(
        microbatch_splitting=microbatch_cfg,
        mesh_mapping=mesh_cfg,
        stage_communication=comm_cfg,
        dependencies=[dep1],
        schedule=schedule,
    )

    # Test dictionary export and import
    topo_dict = pipe_topo.to_dict()
    mb_dict = topo_dict["microbatch_splitting"]
    assert isinstance(mb_dict, dict)
    assert mb_dict["num_microbatches"] == 8
    reconstructed_topo = PipelineTopologyConfig.from_dict(topo_dict)
    assert reconstructed_topo.mesh_mapping.devices_per_stage == 2

    # Graph integration and set_pipeline_topology
    n1 = LogicalNode(id="n1", op_type="Relu", shape_metadata=(4, 4))
    graph = LogicalGraph(
        name="PipelineGraph",
        nodes={"n1": n1},
        inputs=["n1"],
        outputs=["n1"],
    )

    def _get_pipe_topo(g: LogicalGraph) -> PipelineTopologyConfig | None:
        """Return pipeline_topology from a LogicalGraph.

        Args:
            g (LogicalGraph): Graph instance.

        Returns:
            PipelineTopologyConfig | None: Pipeline topology if set.
        """
        return g.pipeline_topology

    assert _get_pipe_topo(graph) is None
    graph.set_pipeline_topology(pipe_topo)
    active_topo = _get_pipe_topo(graph)
    assert active_topo is not None
    assert active_topo.stage_communication.protocol == "nccl"

    # Serialization roundtrip
    g_dict = graph.to_dict()
    assert "pipeline_topology" in g_dict
    raw_pipe = g_dict["pipeline_topology"]
    assert isinstance(raw_pipe, dict)
    raw_comm = raw_pipe["stage_communication"]
    assert isinstance(raw_comm, dict)
    assert raw_comm["protocol"] == "nccl"

    g_restored = LogicalGraph.from_dict(g_dict)
    assert g_restored.pipeline_topology is not None
    assert g_restored.pipeline_topology.mesh_mapping.devices_per_stage == 2
    assert g_restored.pipeline_topology.microbatch_splitting.num_microbatches == 8

    # Clone graph with pipeline_topology
    g_cloned = graph.clone()
    assert g_cloned.pipeline_topology is not None
    assert g_cloned.pipeline_topology.mesh_mapping.devices_per_stage == 2


def test_distributed_and_webrtc_domain_invariants() -> None:
    """Verify domain invariants on LogicalMesh, PartitionSpec, PipelineTopologyConfig, and WebRTCSignalingTopology."""
    import pydantic
    import pytest

    from ml_switcheroo_ir import LogicalMesh, PartitionSpec, WebRTCPeerConfig, WebRTCSignalingTopology

    with pytest.raises(pydantic.ValidationError, match="must be a sequence"):
        PartitionSpec.model_validate({"axes": "not_a_seq"})

    with pytest.raises(pydantic.ValidationError, match="non-empty mapping"):
        LogicalMesh(shape={})

    with pytest.raises(pydantic.ValidationError, match="positive integer size >= 1"):
        LogicalMesh(shape={"data": 0})

    with pytest.raises(pydantic.ValidationError, match="positive integer size >= 1"):
        LogicalMesh(shape={"data": True})

    with pytest.raises(pydantic.ValidationError, match="num_microbatches must be >= 1"):
        PipelineTopologyConfig(
            microbatch_splitting=MicrobatchSplittingConfig(num_microbatches=0),
            mesh_mapping=MeshMappingConfig(devices_per_stage=1),
            stage_communication=StageCommunicationConfig(),
        )

    with pytest.raises(pydantic.ValidationError, match="devices_per_stage must be >= 1"):
        PipelineTopologyConfig(
            microbatch_splitting=MicrobatchSplittingConfig(num_microbatches=2),
            mesh_mapping=MeshMappingConfig(devices_per_stage=0),
            stage_communication=StageCommunicationConfig(),
        )

    with pytest.raises(pydantic.ValidationError, match="non-empty strings"):
        PipelineTopologyConfig(
            microbatch_splitting=MicrobatchSplittingConfig(num_microbatches=2),
            mesh_mapping=MeshMappingConfig(devices_per_stage=1),
            stage_communication=StageCommunicationConfig(),
            dependencies=[DependencyConfig(source_stage=" ", target_stage="s1")],
        )

    with pytest.raises(pydantic.ValidationError, match="chunk_size_bytes must be >= 1"):
        WebRTCSignalingTopology(chunk_size_bytes=0)

    with pytest.raises(pydantic.ValidationError, match="Watermarks must satisfy"):
        WebRTCSignalingTopology(low_watermark_bytes=2000, high_watermark_bytes=1000)

    with pytest.raises(pydantic.ValidationError, match="unique peer_id"):
        WebRTCSignalingTopology(peers=[WebRTCPeerConfig(peer_id="p0"), WebRTCPeerConfig(peer_id="p0")])

    with pytest.raises(pydantic.ValidationError, match="connectivity_matrix must have dimensions"):
        WebRTCSignalingTopology(
            peers=[WebRTCPeerConfig(peer_id="p0"), WebRTCPeerConfig(peer_id="p1")],
            connectivity_matrix=[[1]],
        )

    with pytest.raises(pydantic.ValidationError, match="binary integers 0 or 1"):
        WebRTCSignalingTopology(
            peers=[WebRTCPeerConfig(peer_id="p0")],
            connectivity_matrix=[[2]],
        )
