"""Unit tests for Phase 3 NewType identifiers, refined integers, PortRef, and ProvenGraph witnesses."""

from __future__ import annotations

import pytest

from ml_switcheroo_ir.distributed import LogicalMesh, PartitionSpec
from ml_switcheroo_ir.graph import CyclicGraphError, LogicalGraph, LogicalNode
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
    prove_acyclic_sorted,
    prove_grounded,
    prove_schema_valid,
    prove_sharding_valid,
)
from ml_switcheroo_ir.typeclasses import Err, Invalid, Ok, Valid
from ml_switcheroo_ir.validation.grounding import SnapshotManifestType
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
