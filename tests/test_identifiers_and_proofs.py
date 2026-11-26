"""Unit tests for Phase 3 NewType identifiers, refined integers, PortRef, and ProvenGraph witnesses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ml_switcheroo_ir.distributed import (
    LogicalMesh,
    MeshMappingConfig,
    MicrobatchSplittingConfig,
    PartitionSpec,
    PipelineTopologyConfig,
    StageCommunicationConfig,
)
from ml_switcheroo_ir.errors import (
    GraphCycleDiagnosticError,
    GroundingAuditBatchError,
    SchemaValidationBatchError,
    ShardingValidationBatchError,
)
from ml_switcheroo_ir.graph import (
    FROZEN_LOGICAL_GRAPH_EQ,
    FROZEN_LOGICAL_GRAPH_HASH,
    CyclicGraphError,
    FrozenLogicalGraph,
    FrozenLogicalNode,
    LogicalGraph,
    LogicalNode,
    freeze_graph,
    thaw_graph,
)
from ml_switcheroo_ir.identifiers import (
    DIALECT_DOMAIN_EQ,
    DIALECT_DOMAIN_HASH,
    DIALECT_DOMAIN_ORD,
    MESH_AXIS_NAME_EQ,
    MESH_AXIS_NAME_HASH,
    MESH_AXIS_NAME_ORD,
    NODE_ID_EQ,
    NODE_ID_HASH,
    NODE_ID_ORD,
    NON_BOOL_INT_EQ,
    NON_BOOL_INT_HASH,
    NON_BOOL_INT_ORD,
    OP_NAME_EQ,
    OP_NAME_HASH,
    OP_NAME_ORD,
    PORT_REF_EQ,
    PORT_REF_HASH,
    PORT_REF_ORD,
    QUALIFIED_OP_KEY_EQ,
    QUALIFIED_OP_KEY_HASH,
    QUALIFIED_OP_KEY_ORD,
    SSA_VALUE_NAME_EQ,
    SSA_VALUE_NAME_HASH,
    SSA_VALUE_NAME_ORD,
    DialectDomain,
    NodeId,
    OpName,
    PortRef,
    QualifiedOpKey,
    is_non_bool_int,
    parse_dialect_domain,
    parse_mesh_axis_name,
    parse_node_id,
    parse_non_negative_int,
    parse_op_name,
    parse_positive_int,
    parse_ssa_name,
    require_non_bool_int,
)
from ml_switcheroo_ir.proofs import (
    AcyclicAndSchemaPhase,
    AcyclicSchemaShardingPhase,
    AcyclicSortedPhase,
    FullyVerifiedPhase,
    GroundedPhase,
    ProvenGraph,
    SchemaValidPhase,
    ShardingValidPhase,
    prove_acyclic_sorted,
    prove_fully_verified,
    prove_grounded,
    prove_schema_valid,
    prove_sharding_valid,
)
from ml_switcheroo_ir.protocols import LoopOptionsProtocol, MeshLikeProtocol
from ml_switcheroo_ir.typeclasses import Err, Invalid, Ok, Valid
from ml_switcheroo_ir.types import DType, TensorSpec
from ml_switcheroo_ir.validation.grounding import GroundingAuditReport, SnapshotManifestType
from ml_switcheroo_ir.validation.schema_validator import Validator


def test_refined_integers_and_smart_constructors() -> None:
    """Test is_non_bool_int, require_non_bool_int, parse_non_negative_int, and parse_positive_int."""
    assert is_non_bool_int(0) is True
    assert is_non_bool_int(-5) is True
    assert is_non_bool_int(True) is False
    assert is_non_bool_int(False) is False
    assert is_non_bool_int("3") is False

    assert int(require_non_bool_int(42)) == 42
    with pytest.raises(TypeError, match="non-boolean int"):
        require_non_bool_int(True, label="axis")

    assert parse_non_negative_int(0).is_ok()
    assert parse_non_negative_int(5).is_ok()
    assert parse_non_negative_int(-1).is_err()
    assert parse_non_negative_int(False).is_err()

    assert parse_positive_int(1).is_ok()
    assert parse_positive_int(0).is_err()
    assert parse_positive_int(-3).is_err()
    assert parse_positive_int(True).is_err()

    n1 = require_non_bool_int(3)
    n2 = require_non_bool_int(7)
    assert NON_BOOL_INT_EQ.eq(n1, n1) is True
    assert NON_BOOL_INT_ORD.lt(n1, n2) is True
    assert NON_BOOL_INT_HASH.hash_value(n1) == NON_BOOL_INT_HASH.hash_value(n1)


def test_string_identifiers_qualified_keys_and_port_ref() -> None:
    """Test NodeId, SsaValueName, DialectDomain, OpName, MeshAxisName, QualifiedOpKey, and PortRef."""
    nid_ok = parse_node_id("n1")
    assert isinstance(nid_ok, Ok)
    assert parse_node_id("").is_err()
    assert NODE_ID_EQ.eq(nid_ok.value, nid_ok.value)
    assert NODE_ID_ORD.le(nid_ok.value, nid_ok.value)
    assert isinstance(NODE_ID_HASH.hash_value(nid_ok.value), int)

    ssa_ok = parse_ssa_name("t0")
    assert isinstance(ssa_ok, Ok)
    assert parse_ssa_name("").is_err()
    assert SSA_VALUE_NAME_EQ.eq(ssa_ok.value, ssa_ok.value)
    assert SSA_VALUE_NAME_ORD.le(ssa_ok.value, ssa_ok.value)
    assert isinstance(SSA_VALUE_NAME_HASH.hash_value(ssa_ok.value), int)

    dom_empty = parse_dialect_domain("")
    dom_shlo = parse_dialect_domain("stablehlo")
    assert isinstance(dom_empty, Ok) and isinstance(dom_shlo, Ok)
    assert parse_dialect_domain("   ").is_err()
    assert DIALECT_DOMAIN_EQ.neq(dom_empty.value, dom_shlo.value)
    assert DIALECT_DOMAIN_ORD.lt(dom_empty.value, dom_shlo.value)
    assert isinstance(DIALECT_DOMAIN_HASH.hash_value(dom_shlo.value), int)

    op_ok = parse_op_name("Add")
    assert isinstance(op_ok, Ok)
    assert parse_op_name("").is_err()
    assert OP_NAME_EQ.eq(op_ok.value, op_ok.value)
    assert OP_NAME_ORD.le(op_ok.value, op_ok.value)
    assert isinstance(OP_NAME_HASH.hash_value(op_ok.value), int)

    ax_ok = parse_mesh_axis_name("tp")
    assert isinstance(ax_ok, Ok)
    assert parse_mesh_axis_name("").is_err()
    assert MESH_AXIS_NAME_EQ.eq(ax_ok.value, ax_ok.value)
    assert MESH_AXIS_NAME_ORD.le(ax_ok.value, ax_ok.value)
    assert isinstance(MESH_AXIS_NAME_HASH.hash_value(ax_ok.value), int)

    q_core = QualifiedOpKey(DialectDomain(""), OpName("Relu"))
    q_shlo = QualifiedOpKey(DialectDomain("stablehlo"), OpName("add"))
    assert q_core.qualified_str() == "Relu"
    assert q_shlo.qualified_str() == "stablehlo.add"
    assert QUALIFIED_OP_KEY_EQ.neq(q_core, q_shlo)
    assert QUALIFIED_OP_KEY_ORD.lt(q_core, q_shlo)
    assert isinstance(QUALIFIED_OP_KEY_HASH.hash_value(q_shlo), int)


def test_port_ref_parsing_rendering_and_ordering() -> None:
    """Test PortRef.parse, PortRef.base_identifier, PortRef.render, and witness instances."""
    indexed = PortRef.parse("split:2")
    plain = PortRef.parse("act_out")
    non_digit_colon = PortRef.parse("ns:tensor")
    leading_colon = PortRef.parse(":0")

    assert indexed.port_index == 2
    assert indexed.render() == "split:2"
    assert plain.port_index is None
    assert plain.render() == "act_out"
    assert non_digit_colon.port_index is None
    assert leading_colon.port_index is None

    assert PortRef.base_identifier("split:2") == "split"
    assert PortRef.base_identifier("ns:tensor") == "ns"
    assert PortRef.base_identifier("plain") == "plain"

    assert PORT_REF_EQ.eq(indexed, PortRef.parse("split:2"))
    assert PORT_REF_ORD.lt(plain, indexed)
    assert PORT_REF_HASH.hash_value(indexed) == PORT_REF_HASH.hash_value(PortRef.parse("split:2"))


def test_proven_graph_acyclic_and_schema_witnesses() -> None:
    """Test prove_acyclic_sorted and prove_schema_valid including clone isolation and failure paths."""
    n1 = LogicalNode(id="x", op_type="Input", outputs=["x_out"])
    n2 = LogicalNode(id="r", op_type="Relu", inputs=["x_out"], outputs=["y_out"])
    dag = LogicalGraph(name="Dag", nodes={"x": n1, "r": n2}, outputs=["y_out"])

    acyclic_res = prove_acyclic_sorted(dag)
    assert isinstance(acyclic_res, Ok)
    assert [n.id for n in acyclic_res.value.ordered_nodes] == ["x", "r"]

    # Mutating caller's original graph must not mutate proven snapshot
    dag.nodes["extra"] = LogicalNode(id="extra", op_type="Relu", inputs=["y_out"], outputs=["z"])
    assert "extra" not in acyclic_res.value.graph.nodes

    cyc_a = LogicalNode(id="a", op_type="Relu", inputs=["b_out"], outputs=["a_out"])
    cyc_b = LogicalNode(id="b", op_type="Relu", inputs=["a_out"], outputs=["b_out"])
    cyc_g = LogicalGraph(nodes={"a": cyc_a, "b": cyc_b})
    cyc_res = prove_acyclic_sorted(cyc_g)
    assert isinstance(cyc_res, Err)
    assert isinstance(cyc_res.error, CyclicGraphError)

    valid_g = LogicalGraph(
        nodes={"r": LogicalNode(id="r", op_type="Relu", inputs=["in0"], outputs=["out0"])},
        inputs=["in0"],
        outputs=["out0"],
    )
    schema_ok = prove_schema_valid(valid_g)
    assert isinstance(schema_ok, Valid)
    assert prove_schema_valid(valid_g, validator=Validator()).is_valid()

    bad_node = LogicalNode(id="c", op_type="Concat", inputs=["in0"], attributes={"axis": "not_int"}, outputs=["o"])
    bad_g = LogicalGraph(nodes={"c": bad_node}, inputs=["in0"], outputs=["o"])
    schema_bad = prove_schema_valid(bad_g)
    assert isinstance(schema_bad, Invalid)
    assert len(schema_bad.errors) >= 1


def test_proven_graph_sharding_and_grounding_witnesses() -> None:
    """Test prove_sharding_valid and prove_grounded for both Valid and Invalid graphs."""
    mesh = LogicalMesh(shape={"dp": 2})
    s_in = LogicalNode(id="in", op_type="Input", outputs=["x"], sharding=PartitionSpec(("dp",)))
    s_ok = LogicalNode(id="r", op_type="Relu", inputs=["x"], outputs=["y"], sharding=PartitionSpec(("dp",)))
    g_shard_ok = LogicalGraph(nodes={"in": s_in, "r": s_ok}, mesh=mesh, outputs=["y"])
    assert isinstance(prove_sharding_valid(g_shard_ok), Valid)

    s_bad = LogicalNode(id="r", op_type="Relu", inputs=["x"], outputs=["y"], sharding=PartitionSpec((None,)))
    g_shard_bad = LogicalGraph(nodes={"in": s_in, "r": s_bad}, mesh=mesh, outputs=["y"])
    assert isinstance(prove_sharding_valid(g_shard_bad), Invalid)

    manifest: SnapshotManifestType = {"operations": {"Relu": {"params": []}}}
    g_ground_ok = LogicalGraph(
        nodes={"r": LogicalNode(id="r", op_type="Relu", domain="onnx", inputs=["x"], outputs=["y"])},
        inputs=["x"],
        outputs=["y"],
    )
    assert isinstance(prove_grounded(g_ground_ok, snapshots_path=manifest), Valid)

    g_ground_bad = LogicalGraph(
        nodes={"u": LogicalNode(id="u", op_type="TotallyFakeOp999", domain="torch", inputs=["x"], outputs=["y"])},
        inputs=["x"],
        outputs=["y"],
    )
    assert isinstance(prove_grounded(g_ground_bad, snapshots_path=manifest), Invalid)


def test_frozen_logical_graph_roundtrip_eq_hash_and_immutability() -> None:
    """Test freeze_graph/thaw_graph across rich node/graph variants, get_node, EQ/HASH, and immutability."""

    class _DummyMesh:
        """Minimal MeshLikeProtocol implementation for freeze/thaw testing."""

        @property
        def shape(self) -> dict[str, int]:
            """Return mesh axis sizes.

            Returns:
                dict[str, int]: Axis map.
            """
            return {"tp": 2}

        @property
        def axis_names(self) -> tuple[str, ...]:
            """Return mesh axis names.

            Returns:
                tuple[str, ...]: Axis tuple.
            """
            return ("tp",)

    class _DummyLoopOpts:
        """Minimal LoopOptionsProtocol implementation for freeze/thaw testing."""

        @property
        def parallel_iterations(self) -> int | None:
            """Return parallel iteration count.

            Returns:
                int | None: Parallel count.
            """
            return 4

        @property
        def swap_memory(self) -> bool | None:
            """Return memory swap flag.

            Returns:
                bool | None: Swap flag.
            """
            return False

        @property
        def maximum_iterations(self) -> int | None:
            """Return maximum iteration count.

            Returns:
                int | None: Max iterations.
            """
            return 16

    assert isinstance(_DummyMesh(), MeshLikeProtocol)
    assert isinstance(_DummyLoopOpts(), LoopOptionsProtocol)

    sub_g = LogicalGraph(name="Sub", nodes={"s1": LogicalNode(id="s1", op_type="Relu", sharding="replicated")})
    topo = PipelineTopologyConfig(
        microbatch_splitting=MicrobatchSplittingConfig(num_microbatches=2),
        mesh_mapping=MeshMappingConfig(devices_per_stage=1),
        stage_communication=StageCommunicationConfig(),
    )
    rich_g = LogicalGraph(
        name="Rich",
        nodes={
            "n1": LogicalNode(
                id="n1",
                op_type="If",
                attributes={"k": [1, 2]},
                outputs=["n1_out"],
                sharding={"custom": 1},
                dtype=DType.float32,
                output_specs=[TensorSpec(shape=(2, 4), dtype=DType.float32)],
                subgraphs={"then": sub_g, "meta": {"tag": "v"}},
                strides=(4, 1),
            ),
            "n2": LogicalNode(
                id="n2",
                op_type="Relu",
                inputs=["n1_out"],
                outputs=["out"],
                sharding=PartitionSpec(("dp", None)),
            ),
        },
        inputs=["in0"],
        input_specs={"in0": TensorSpec(shape=(2, 4), dtype=DType.float32)},
        outputs=["out"],
        initializers={"w": [1.0, 2.0]},
        mesh=LogicalMesh(shape={"dp": 2}),
        pipeline_topology=topo,
        attributes={"ga": 1},
        annotations={"ann": "ok"},
        metadata={"m_dict": {"x": 1}, "m_loop": _DummyLoopOpts()},
        loop_options={"max_iter": 8},
    )
    rich_g.add_edge("n1", "n2", source_idx=0, target_idx=0, value_name="n1_out")

    frozen = freeze_graph(rich_g)
    assert frozen.get_node("n1") is not None
    assert frozen.get_node("missing") is None
    thawed = thaw_graph(frozen)
    assert thawed == rich_g
    assert FROZEN_LOGICAL_GRAPH_EQ.eq(frozen, freeze_graph(thawed))
    assert FROZEN_LOGICAL_GRAPH_HASH.hash_value(frozen) == FROZEN_LOGICAL_GRAPH_HASH.hash_value(freeze_graph(thawed))

    init_int_f = freeze_graph(LogicalGraph(name="I", initializers={"w": [1]}))
    init_bool_f = freeze_graph(LogicalGraph(name="I", initializers={"w": [True]}))
    assert FROZEN_LOGICAL_GRAPH_EQ.neq(init_int_f, init_bool_f)

    alt_edge_g = rich_g.clone()
    alt_edge_g.add_edge("n1", "n2", source_idx=0, target_idx=0, value_name="alt_ssa_name")
    assert FROZEN_LOGICAL_GRAPH_EQ.neq(frozen, freeze_graph(alt_edge_g))

    # Protocol mesh & loop_options + None mesh/loop_options branches + unindexed get_node fallback
    proto_g = LogicalGraph(name="Proto", mesh=_DummyMesh(), loop_options=_DummyLoopOpts())
    assert thaw_graph(freeze_graph(proto_g)).name == "Proto"
    plain_g = LogicalGraph(name="Plain")
    assert thaw_graph(freeze_graph(plain_g)) == plain_g

    fn_direct = FrozenLogicalNode(id=NodeId("d1"), op_type=OpName("Relu"))
    fg_unindexed = FrozenLogicalGraph(nodes=((NodeId("d1"), fn_direct),))
    assert fg_unindexed.get_node("d1") == fn_direct
    assert fg_unindexed.get_node("d2") is None

    mp = pytest.MonkeyPatch()
    with pytest.raises(FrozenInstanceError):
        mp.setattr(frozen, "name", "Mutated")
    with pytest.raises(FrozenInstanceError):
        mp.setattr(fn_direct, "version", 99)


def test_cumulative_proven_graph_covariance_and_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ProvenGraph covariance, defensive .graph/.ordered_frozen_nodes, upgrade_to_*, and prove_fully_verified.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    manifest: SnapshotManifestType = {"operations": {"Relu": {"params": []}}}
    mesh = LogicalMesh(shape={"dp": 2})
    g_ok = LogicalGraph(
        name="FullOk",
        nodes={
            "r1": LogicalNode(
                id="r1",
                op_type="Relu",
                domain="onnx",
                inputs=["x"],
                outputs=["h"],
                sharding=PartitionSpec(("dp",)),
            ),
            "r2": LogicalNode(
                id="r2",
                op_type="Relu",
                domain="onnx",
                inputs=["h"],
                outputs=["y"],
                sharding=PartitionSpec(("dp",)),
            ),
        },
        inputs=["x"],
        mesh=mesh,
        outputs=["y"],
    )

    full_res = prove_fully_verified(g_ok, snapshots_path=manifest)
    assert isinstance(full_res, Ok)
    proven_full: ProvenGraph[FullyVerifiedPhase] = full_res.value

    # Covariant assignment across all 4 base and intermediate phase types
    as_acyclic: ProvenGraph[AcyclicSortedPhase] = proven_full
    as_schema: ProvenGraph[SchemaValidPhase] = proven_full
    as_sharding: ProvenGraph[ShardingValidPhase] = proven_full
    as_grounded: ProvenGraph[GroundedPhase] = proven_full
    as_a_s: ProvenGraph[AcyclicAndSchemaPhase] = proven_full
    as_a_s_sh: ProvenGraph[AcyclicSchemaShardingPhase] = proven_full
    assert len(as_acyclic.ordered_frozen_nodes) == 2
    assert as_schema.ordered_node_ids == as_sharding.ordered_node_ids == as_grounded.ordered_node_ids
    assert as_a_s.frozen_graph == as_a_s_sh.frozen_graph

    # Cover ordered_frozen_nodes skipping unknown node IDs
    sparse_proven: ProvenGraph[AcyclicSortedPhase] = ProvenGraph(
        frozen_graph=proven_full.frozen_graph,
        ordered_node_ids=(NodeId("r1"), NodeId("nonexistent")),
    )
    assert len(sparse_proven.ordered_frozen_nodes) == 1
    assert len(sparse_proven.ordered_nodes) == 1

    # Mutating proven_full.graph view must never corrupt proven_full
    mut_view = proven_full.graph
    mut_view.nodes.clear()
    assert len(proven_full.graph.nodes) == 2
    mp_local = pytest.MonkeyPatch()
    with pytest.raises(FrozenInstanceError):
        mp_local.setattr(proven_full, "ordered_node_ids", ())

    # Mint without explicit ordered_nodes defaults to frozen.nodes order
    minted_default: ProvenGraph[AcyclicSortedPhase] = ProvenGraph.mint(g_ok)
    assert len(minted_default.ordered_node_ids) == 2

    # Failure branches of prove_fully_verified across all 4 stages
    cyc_g = LogicalGraph(
        nodes={
            "a": LogicalNode(id="a", op_type="Relu", inputs=["b"], outputs=["a"]),
            "b": LogicalNode(id="b", op_type="Relu", inputs=["a"], outputs=["b"]),
        }
    )
    cyc_full = prove_fully_verified(cyc_g, snapshots_path=manifest)
    assert isinstance(cyc_full, Err) and isinstance(cyc_full.error, GraphCycleDiagnosticError)

    bad_schema_g = LogicalGraph(
        nodes={"c": LogicalNode(id="c", op_type="Concat", inputs=["x"], attributes={"axis": "bad"}, outputs=["y"])},
        inputs=["x"],
        outputs=["y"],
    )
    sch_full = prove_fully_verified(bad_schema_g, snapshots_path=manifest)
    assert isinstance(sch_full, Err) and isinstance(sch_full.error, SchemaValidationBatchError)

    bad_shard_g = LogicalGraph(
        nodes={
            "in": LogicalNode(id="in", op_type="Input", outputs=["x"], sharding=PartitionSpec(("dp",))),
            "r": LogicalNode(id="r", op_type="Relu", inputs=["x"], outputs=["y"], sharding=PartitionSpec((None,))),
        },
        mesh=mesh,
        outputs=["y"],
    )
    shd_full = prove_fully_verified(bad_shard_g, snapshots_path=manifest)
    assert isinstance(shd_full, Err) and isinstance(shd_full.error, ShardingValidationBatchError)

    # Test _extract_grounding_failures fallback + GroundingAuditBatchError in prove_fully_verified
    import ml_switcheroo_ir.proofs as proofs_mod

    monkeypatch.setattr(
        proofs_mod,
        "audit_graph_grounding",
        lambda _g, snapshots_path=None: GroundingAuditReport(is_grounded=False, diagnostics=[]),
    )
    empty_diag_res = prove_grounded(g_ok, snapshots_path=manifest)
    assert isinstance(empty_diag_res, Invalid) and len(empty_diag_res.errors) == 1
    grd_full = prove_fully_verified(g_ok, snapshots_path=manifest)
    assert isinstance(grd_full, Err) and isinstance(grd_full.error, GroundingAuditBatchError)
