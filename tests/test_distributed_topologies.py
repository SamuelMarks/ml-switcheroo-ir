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
    assert topo_dict["microbatch_splitting"]["num_microbatches"] == 8
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
    assert graph.pipeline_topology is None
    graph.set_pipeline_topology(pipe_topo)
    assert graph.pipeline_topology is not None
    assert graph.pipeline_topology.stage_communication.protocol == "nccl"

    # Serialization roundtrip
    g_dict = graph.to_dict()
    assert "pipeline_topology" in g_dict
    assert g_dict["pipeline_topology"]["stage_communication"]["protocol"] == "nccl"

    g_restored = LogicalGraph.from_dict(g_dict)
    assert g_restored.pipeline_topology is not None
    assert g_restored.pipeline_topology.mesh_mapping.devices_per_stage == 2
    assert g_restored.pipeline_topology.microbatch_splitting.num_microbatches == 8

    # Clone graph with pipeline_topology
    g_cloned = graph.clone()
    assert g_cloned.pipeline_topology is not None
    assert g_cloned.pipeline_topology.mesh_mapping.devices_per_stage == 2
