"""Unit tests for Phase 6 JsonCodec / PythonExprEmit / Passes and Phase 7 Catalogs / PartialIso / Compliance."""

from __future__ import annotations

import pytest

from ml_switcheroo_ir.compliance import FILE_COMPLIANCE_MONOID, FileComplianceResult
from ml_switcheroo_ir.distributed import (
    LogicalMesh,
    PartitionSpec,
    WebRTCPeerConfig,
    WebRTCSignalingTopology,
)
from ml_switcheroo_ir.export import (
    ATTRIBUTE_VALUE_PYTHON_EMIT,
    DIMENSION_PYTHON_EMIT,
    DTYPE_PYTHON_EMIT,
    LOGICAL_GRAPH_PYTHON_EMIT,
    LOGICAL_MESH_PYTHON_EMIT,
    LOGICAL_NODE_PYTHON_EMIT,
    PARTITION_SPEC_PYTHON_EMIT,
    SYM_EXPR_PYTHON_EMIT,
    TENSOR_SHAPE_PYTHON_EMIT,
    TENSOR_SPEC_PYTHON_EMIT,
)
from ml_switcheroo_ir.graph import LogicalGraph, LogicalNode
from ml_switcheroo_ir.schema import ALL_DIALECT_CATALOG_WITNESSES
from ml_switcheroo_ir.serialization import (
    ATTRIBUTE_VALUE_JSON_CODEC,
    DIMENSION_JSON_CODEC,
    DTYPE_JSON_CODEC,
    LOGICAL_GRAPH_JSON_CODEC,
    LOGICAL_NODE_JSON_CODEC,
    PARTITION_SPEC_JSON_CODEC,
    SYM_EXPR_JSON_CODEC,
    TENSOR_SHAPE_JSON_CODEC,
    TENSOR_SPEC_JSON_CODEC,
    graph_from_dict_result,
    graph_from_json_result,
)
from ml_switcheroo_ir.shapes import SymBinaryOp, SymConst, SymVar
from ml_switcheroo_ir.transforms import (
    ATTR_HASH_WITNESS,
    CANONICAL_OPTIMIZE_PIPELINE,
    CSE_PASS,
    DCE_PASS,
    LOGICAL_GRAPH_EQ,
    NODE_CSE_WITNESS,
    SHAPE_CONSTANT_PROP_PASS,
    GraphTransformPass,
    compose_passes,
    fixpoint_graph_pass,
)
from ml_switcheroo_ir.translation import (
    OperationTranslationSpec,
    ParameterTranslationEngine,
    PartialIso,
)
from ml_switcheroo_ir.typeclasses import Err, Ok
from ml_switcheroo_ir.types import DType, TensorShape, TensorSpec


def test_scalar_and_tensor_json_codecs_and_error_paths() -> None:
    """Test DTYPE, SYM_EXPR, DIMENSION, TENSOR_SHAPE, TENSOR_SPEC, and PARTITION_SPEC JsonCodecs."""
    assert DTYPE_JSON_CODEC.decode(DTYPE_JSON_CODEC.encode(DType.bfloat16)) == Ok(DType.bfloat16)
    assert DTYPE_JSON_CODEC.decode("int32") == Ok(DType.int32)
    assert DTYPE_JSON_CODEC.decode("not_a_dtype").is_err()
    assert DTYPE_JSON_CODEC.decode(123).is_err()

    expr = SymBinaryOp("+", SymVar("B"), SymConst(2))
    assert SYM_EXPR_JSON_CODEC.decode(SYM_EXPR_JSON_CODEC.encode(expr)) == Ok(expr)
    assert SYM_EXPR_JSON_CODEC.decode("B + 2") == Ok(expr)
    assert SYM_EXPR_JSON_CODEC.decode("B @ 2").is_err()
    assert SYM_EXPR_JSON_CODEC.decode(99).is_err()

    assert DIMENSION_JSON_CODEC.decode(DIMENSION_JSON_CODEC.encode(8)) == Ok(8)
    assert DIMENSION_JSON_CODEC.decode(None).is_err()
    assert DIMENSION_JSON_CODEC.decode(1.5).is_err()
    assert DIMENSION_JSON_CODEC.decode([1]).is_err()
    assert DIMENSION_JSON_CODEC.decode(True).is_err()

    t_shape = TensorShape((2, SymVar("N")))
    assert TENSOR_SHAPE_JSON_CODEC.decode(TENSOR_SHAPE_JSON_CODEC.encode(t_shape)) == Ok(t_shape)
    assert TENSOR_SHAPE_JSON_CODEC.decode([2, "N"]).is_ok()
    assert TENSOR_SHAPE_JSON_CODEC.decode({"__sw_type__": "TensorShape", "dims": "bad"}).is_err()
    assert TENSOR_SHAPE_JSON_CODEC.decode("not_list").is_err()
    bad_dim = TENSOR_SHAPE_JSON_CODEC.decode([2, True])
    assert isinstance(bad_dim, Err) and bad_dim.error.path == (1,)

    t_spec = TensorSpec(shape=(4, 8), dtype=DType.float16, sparsity="2:4")
    assert TENSOR_SPEC_JSON_CODEC.decode(TENSOR_SPEC_JSON_CODEC.encode(t_spec)) == Ok(t_spec)
    assert TENSOR_SPEC_JSON_CODEC.decode("bad").is_err()
    assert TENSOR_SPEC_JSON_CODEC.decode({"shape": "bad", "dtype": "float32"}).is_err()
    assert TENSOR_SPEC_JSON_CODEC.decode({"shape": [1], "dtype": "bad_dt"}).is_err()
    assert TENSOR_SPEC_JSON_CODEC.decode({"shape": [1], "dtype": "float32", "sparsity": 5}).is_err()

    p_spec = PartitionSpec(("dp", ("tp", "cp"), None))
    assert PARTITION_SPEC_JSON_CODEC.decode(PARTITION_SPEC_JSON_CODEC.encode(p_spec)) == Ok(p_spec)
    assert PARTITION_SPEC_JSON_CODEC.decode({"axes": "not_list"}).is_err()
    assert PARTITION_SPEC_JSON_CODEC.decode(42).is_err()
    assert PARTITION_SPEC_JSON_CODEC.decode({"axes": [[]]}).is_err()


def test_attribute_node_and_graph_json_codecs_and_breadcrumbs() -> None:
    """Test ATTRIBUTE_VALUE, LOGICAL_NODE, LOGICAL_GRAPH codecs and total *_result error paths."""
    from ml_switcheroo_ir.typeclasses.codec import dict_str_codec, list_codec, optional_codec, tuple_codec
    from ml_switcheroo_ir.types import AttributeValue

    attr: AttributeValue = {"b": b"\x00\xff", "dt": DType.float32, "nums": [1, 2]}
    assert ATTRIBUTE_VALUE_JSON_CODEC.decode(ATTRIBUTE_VALUE_JSON_CODEC.encode(attr)) == Ok(attr)
    assert ATTRIBUTE_VALUE_JSON_CODEC.decode({"__sw_type__": "DType", "value": "invalid_dt"}).is_err()
    nested_err = ATTRIBUTE_VALUE_JSON_CODEC.decode({"outer": [{"__sw_type__": "DType", "value": "invalid_dt"}]})
    assert isinstance(nested_err, Err) and nested_err.error.path == ("outer", 0)
    assert "$.outer[0]" in nested_err.error.format_diagnostic()

    composed = dict_str_codec(list_codec(optional_codec(tuple_codec(DTYPE_JSON_CODEC))))
    valid_payload = composed.encode({"dtypes": [(DType.float16,), None]})
    assert composed.decode(valid_payload) == Ok({"dtypes": [(DType.float16,), None]})
    malformed = composed.decode({"dtypes": [["not_a_dtype"], None]})
    assert isinstance(malformed, Err) and malformed.error.path == ("dtypes", 0, 0)

    node = LogicalNode(id="n1", op_type="Relu", inputs=["x"], outputs=["y"], attributes={"alpha": 1.0})
    assert LOGICAL_NODE_JSON_CODEC.decode(LOGICAL_NODE_JSON_CODEC.encode(node)) == Ok(node)
    assert LOGICAL_NODE_JSON_CODEC.decode("not_a_dict").is_err()
    assert LOGICAL_NODE_JSON_CODEC.decode({"id": "n1", "op_type": "Relu", "dtype": "bad_dtype"}).is_err()

    graph = LogicalGraph(name="G", nodes={"n1": node}, inputs=["x"], outputs=["y"])
    assert LOGICAL_GRAPH_JSON_CODEC.decode(LOGICAL_GRAPH_JSON_CODEC.encode(graph)) == Ok(graph)
    assert LOGICAL_GRAPH_JSON_CODEC.decode("not_mapping").is_err()

    assert LogicalGraph.from_json_result(graph.to_json().encode("utf-8")) == Ok(graph)
    assert graph_from_json_result("{malformed_json").is_err()
    assert graph_from_json_result(b"\xff\xfe").is_err()
    assert graph_from_json_result("[1, 2, 3]").is_err()

    bad_nodes_field = graph_from_dict_result({"nodes": [1, 2]})
    assert isinstance(bad_nodes_field, Err) and bad_nodes_field.error.path == ("nodes",)
    bad_node_entry = graph_from_dict_result({"nodes": {"n1": "not_mapping"}})
    assert isinstance(bad_node_entry, Err) and bad_node_entry.error.path == ("nodes", "n1")
    bad_node_schema = graph_from_dict_result({"nodes": {"n1": {"id": "n1", "op_type": "Relu", "dtype": "nope"}}})
    assert isinstance(bad_node_schema, Err) and bad_node_schema.error.path == ("nodes", "n1")
    bad_envelope = graph_from_dict_result({"nodes": {}, "mesh": {"shape": {}}})
    assert isinstance(bad_envelope, Err) and bad_envelope.error.path == ()


def test_python_expr_emit_witnesses_and_transform_passes() -> None:
    """Test Phase 6.2 PythonExprEmit witnesses and Phase 6.3 CSE/GraphTransformPass pipelines."""
    assert DTYPE_PYTHON_EMIT.emit_python(DType.float32, "") == "DType.float32"
    assert SYM_EXPR_PYTHON_EMIT.emit_python(SymVar("B"), "") == "SymVar('B')"
    assert DIMENSION_PYTHON_EMIT.emit_python(16, "sw_ir.") == "16"
    assert "TensorShape" in TENSOR_SHAPE_PYTHON_EMIT.emit_python(TensorShape((2,)), "")
    assert "TensorSpec" in TENSOR_SPEC_PYTHON_EMIT.emit_python(TensorSpec(shape=(2,), dtype=DType.int32), "")
    assert "PartitionSpec" in PARTITION_SPEC_PYTHON_EMIT.emit_python(PartitionSpec(("tp",)), "")
    assert "{}" == ATTRIBUTE_VALUE_PYTHON_EMIT.emit_python({}, "")

    plain_mesh = LogicalMesh(shape={"tp": 2})
    rtc_mesh = LogicalMesh(
        shape={"tp": 2},
        webrtc_topology=WebRTCSignalingTopology(
            signaling_url="wss://sig.example.com",
            peers=[WebRTCPeerConfig(peer_id="p0"), WebRTCPeerConfig(peer_id="p1")],
        ),
    )
    assert "LogicalMesh" in LOGICAL_MESH_PYTHON_EMIT.emit_python(plain_mesh, "")
    assert "WebRTCSignalingTopology" in LOGICAL_MESH_PYTHON_EMIT.emit_python(rtc_mesh, "")

    n1 = LogicalNode(id="r1", op_type="Relu", inputs=["x"], outputs=["y1"])
    n2 = LogicalNode(id="r2", op_type="Relu", inputs=["x"], outputs=["y2"])
    dead = LogicalNode(id="dead", op_type="Neg", inputs=["x"], outputs=["unused"])
    g = LogicalGraph(nodes={"r1": n1, "r2": n2, "dead": dead}, inputs=["x"], outputs=["y1", "y2"])

    assert "LogicalNode" in LOGICAL_NODE_PYTHON_EMIT.emit_python(n1, "")
    assert "LogicalNode" in LOGICAL_NODE_PYTHON_EMIT.emit_python(n1, "sw_ir.")
    assert "LogicalGraph" in LOGICAL_GRAPH_PYTHON_EMIT.emit_python(g, "")
    assert "LogicalGraph" in LOGICAL_GRAPH_PYTHON_EMIT.emit_python(g, "sw_ir.")

    assert ATTR_HASH_WITNESS.eq_witness.eq([1, 2], [1, 2])
    assert ATTR_HASH_WITNESS.hash_value([1, 2]) == ATTR_HASH_WITNESS.hash_value([1, 2])
    assert NODE_CSE_WITNESS.eq_witness.eq(n1, n2)
    assert NODE_CSE_WITNESS.hash_value(n1) == NODE_CSE_WITNESS.hash_value(n2)

    id_pass = compose_passes()
    assert LOGICAL_GRAPH_EQ.eq(id_pass(g), g)

    opt = CANONICAL_OPTIMIZE_PIPELINE(g)
    assert "dead" not in opt.nodes
    assert len(opt.nodes) == 1
    assert SHAPE_CONSTANT_PROP_PASS.name == "propagate_shapes_and_constants"
    assert CSE_PASS.name == "cse" and DCE_PASS.name == "dce"

    toggle = GraphTransformPass(
        name="rename",
        apply=lambda gr: gr.clone() if gr.name == "done" else LogicalGraph(name=gr.name + "!"),
    )
    capped = fixpoint_graph_pass(toggle, max_iterations=2)
    assert capped(LogicalGraph(name="a")).name == "a!!"
    with pytest.raises(ValueError, match="non-negative int"):
        fixpoint_graph_pass(DCE_PASS, max_iterations=-1)


def test_dialect_catalog_witnesses_and_operation_translation_spec() -> None:
    """Test Phase 7 ALL_DIALECT_CATALOG_WITNESSES, OperationTranslationSpec, and compliance monoid."""
    import pydantic

    assert len(ALL_DIALECT_CATALOG_WITNESSES) == 21
    for cat in ALL_DIALECT_CATALOG_WITNESSES:
        ops = cat.list_ops()
        assert len(ops) >= 1
        first_schema = cat.lookup(ops[0])
        assert first_schema is not None
        assert cat.lookup("__nonexistent_op_xyz__") is None

    spec_plain = OperationTranslationSpec.model_validate({"roles": {"input": {"torch": ["x"]}}})
    assert spec_plain.metadata is None
    assert OperationTranslationSpec.model_validate(spec_plain) == spec_plain
    spec_extra = OperationTranslationSpec.model_validate(
        {"roles": {}, "metadata": {"k0": 0}, "source_commit": "abc123"}
    )
    assert spec_extra.metadata == {"k0": 0, "source_commit": "abc123"}
    spec_extra_no_meta = OperationTranslationSpec.model_validate({"roles": {}, "source_commit": "def456"})
    assert spec_extra_no_meta.metadata == {"source_commit": "def456"}
    with pytest.raises(pydantic.ValidationError):
        OperationTranslationSpec.model_validate("not_a_mapping")

    fc1 = FileComplianceResult(is_ir=True, implemented_ops={"Add"}, ir_score=(3, 5), fw_score=(1, 4))
    fc2 = FileComplianceResult(is_fw=True, implemented_ops={"Mul"}, ir_score=(2, 5), fw_score=(4, 4))
    fc_merged = FILE_COMPLIANCE_MONOID.mconcat([fc1, FILE_COMPLIANCE_MONOID.empty(), fc2])
    assert fc_merged.is_ir is True and fc_merged.is_fw is True
    assert fc_merged.implemented_ops == {"Add", "Mul"}
    assert fc_merged.ir_score == (3, 5) and fc_merged.fw_score == (4, 4)


def test_partial_iso_and_parameter_translation_results() -> None:
    """Test PartialIso combinators, conv_layout_iso, qkv_split_merge_iso, and *_result methods."""
    eng = ParameterTranslationEngine(
        translations={
            "softmax": {
                "roles": {
                    "axis": {"torch": ["dim"], "jax": ["axis"]},
                }
            }
        }
    )
    assert eng.translate_parameter_result("softmax", "dim", "torch", "jax") == Ok("axis")
    assert eng.translate_parameter_result("softmax", "missing", "torch", "jax").is_err()
    assert eng.translate_attributes_result("softmax", {"dim": -1}, "torch", "jax") == Ok({"axis": -1})
    assert eng.translate_attributes_result("softmax", {"bad": 1}, "torch", "jax").is_err()

    iso_ab: PartialIso[int, str] = PartialIso(
        forward=lambda n: Ok(str(n)),
        backward=lambda s: Ok(int(s)) if s.isdigit() else Err("not digits"),
    )
    iso_bc: PartialIso[str, list[str]] = PartialIso(
        forward=lambda s: Ok([s]),
        backward=lambda xs: Ok(xs[0]) if len(xs) == 1 else Err("bad len"),
    )
    composed = iso_ab.compose(iso_bc)
    inv = composed.inverse()
    assert composed.forward(7) == Ok(["7"])
    assert inv.forward(["7"]) == Ok(7)

    conv_iso = eng.conv_layout_iso("OIHW", "HWIO")
    w_oihw = [[[[1.0, 2.0]]]]
    w_hwio_res = conv_iso.forward(w_oihw)
    assert isinstance(w_hwio_res, Ok)
    assert conv_iso.backward(w_hwio_res.value) == Ok(w_oihw)
    assert eng.conv_layout_iso("BAD", "HWIO").forward(w_oihw).is_err()
    assert eng.conv_layout_iso("OIHW", "BAD").backward(w_oihw).is_err()
    assert conv_iso.forward("symbolic_weight_ref").is_err()
    assert conv_iso.backward("symbolic_weight_ref").is_err()

    qkv_iso = eng.qkv_split_merge_iso(num_heads=1, num_kv_heads=1, head_dim=2, dim=0)
    fused = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    split_res = qkv_iso.forward(fused)
    assert isinstance(split_res, Ok)
    assert qkv_iso.backward(split_res.value) == Ok(fused)
    assert qkv_iso.forward("sym_qkv").is_err()
    assert qkv_iso.backward(("q_ref", "k_ref", "v_ref")).is_err()
    bad_qkv_iso = eng.qkv_split_merge_iso(num_heads=2, num_kv_heads=2, head_dim=4, dim=1)
    assert bad_qkv_iso.forward([[1.0]]).is_err()
    assert bad_qkv_iso.backward(([[1.0]], [[2.0], [3.0]], [[4.0]])).is_err()
