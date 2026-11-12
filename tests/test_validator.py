"""Tests for the validation module."""

import pytest

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.schema.onnx_registry import OpAttribute, OpSchema
from ml_switcheroo_ir.validator import ValidationLevel, Validator

# Create a small mock registry for controlled testing
MOCK_REGISTRY = {
    "Gemm": OpSchema(
        name="Gemm",
        domain="ai.onnx",
        version=11,
        attributes={
            "alpha": OpAttribute(name="alpha", type="float", required=False, default=1.0),
            "beta": OpAttribute(name="beta", type="float", required=False, default=1.0),
            "transA": OpAttribute(name="transA", type="int", required=False, default=0),
            "transB": OpAttribute(name="transB", type="int", required=False, default=0),
        },
        inputs=["A", "B", "C"],
        outputs=["Y"],
    ),
    "Conv": OpSchema(
        name="Conv",
        domain="ai.onnx",
        version=11,
        attributes={
            "kernel_shape": OpAttribute(name="kernel_shape", type="List[int]", required=True, default=None),
            "strides": OpAttribute(name="strides", type="List[int]", required=False, default=[1, 1]),
            "group": OpAttribute(name="group", type="int", required=False, default=1),
        },
        inputs=["X", "W", "B"],
        outputs=["Y"],
    ),
}


@pytest.fixture
def validator() -> Validator:
    """Return a Validator initialized with the mock registry.

    Returns:
        Validator: Configured Validator instance.
    """
    return Validator(registry=MOCK_REGISTRY)


def test_validator_valid_gemm(validator: Validator) -> None:
    """Test a fully compliant Gemm node returning no errors.

    Args:
        validator (Validator): Mock-registry Validator fixture.
    """
    node = LogicalNode(id="gemm1", op_type="Gemm", attributes={"alpha": 2.0})

    errors = validator.validate_kind(node)
    assert not errors

    errors = validator.validate_required_attributes(node)
    assert not errors

    errors = validator.validate_attribute_types(node)
    assert not errors


def test_validator_missing_required_attr(validator: Validator) -> None:
    """Test a Conv node lacking kernel_shape asserting a specific ValidationError.

    Args:
        validator (Validator): Mock-registry Validator fixture.
    """
    node = LogicalNode(id="conv1", op_type="Conv", attributes={"strides": [2, 2]})

    errors = validator.validate_required_attributes(node)
    assert len(errors) == 1
    assert errors[0].attribute == "kernel_shape"
    assert errors[0].level == ValidationLevel.ERROR


def test_validator_invalid_type(validator: Validator) -> None:
    """Test providing a string '1' when an integer 1 is required.

    Args:
        validator (Validator): Mock-registry Validator fixture.
    """
    node = LogicalNode(id="conv1", op_type="Conv", attributes={"group": "1", "kernel_shape": [3, 3]})

    errors = validator.validate_attribute_types(node)
    assert len(errors) == 1
    assert errors[0].attribute == "group"
    assert "Expected int" in errors[0].message
    assert errors[0].level == ValidationLevel.ERROR


def test_validator_default_population(validator: Validator) -> None:
    """Test to ensure missing optional attributes are injected.

    Args:
        validator (Validator): Mock-registry Validator fixture.
    """
    node = LogicalNode(id="conv1", op_type="Conv", attributes={"kernel_shape": [3, 3]})
    validator.populate_defaults(node)

    assert "strides" in node.attributes
    assert node.attributes["strides"] == [1, 1]
    assert "group" in node.attributes
    assert node.attributes["group"] == 1


def test_validator_invalid_edge(validator: Validator) -> None:
    """Test for dangling edge references.

    Args:
        validator (Validator): Mock-registry Validator fixture.
    """
    node1 = LogicalNode(id="n1", op_type="Gemm", inputs=["n0", "n2"])
    graph = LogicalGraph(nodes={"n1": node1})

    errors = validator.validate_edges(graph)
    assert len(errors) == 2
    assert errors[0].node_id == "n1"
    assert errors[1].node_id == "n1"


def test_validator_unknown_kind(validator: Validator) -> None:
    """Test validating an unknown operator.

    Args:
        validator (Validator): Mock-registry Validator fixture.
    """
    node = LogicalNode(id="n1", op_type="UnknownOp")
    errors = validator.validate_kind(node)
    assert len(errors) == 1
    assert errors[0].attribute == "kind"
    assert errors[0].level == ValidationLevel.ERROR


def test_validator_unknown_attribute(validator: Validator) -> None:
    """Test providing an unregistered attribute issues a warning.

    Args:
        validator (Validator): Mock-registry Validator fixture.
    """
    node = LogicalNode(id="gemm1", op_type="Gemm", attributes={"unknown_attr": 42})
    errors = validator.validate_attribute_types(node)
    assert len(errors) == 1
    assert errors[0].attribute == "unknown_attr"
    assert errors[0].level == ValidationLevel.WARNING


def test_validator_graph_integration(validator: Validator) -> None:
    """Test the integrated validate_graph method.

    Args:
        validator (Validator): Mock-registry Validator fixture.
    """
    node1 = LogicalNode(id="n1", op_type="Gemm", attributes={"alpha": 2.0})
    node2 = LogicalNode(id="n2", op_type="Conv", inputs=["n1", "n3"])  # missing kernel_shape, dangling n3

    graph = LogicalGraph(nodes={"n1": node1, "n2": node2})
    errors = validator.validate_graph(graph)

    # We expect:
    # 1. Conv missing kernel_shape
    # 2. node n2 has dangling input n3
    assert len(errors) == 2

    error_attrs = {e.attribute for e in errors}
    assert "kernel_shape" in error_attrs
    assert "inputs" in error_attrs

    # Side-effect-free by default: Gemm attributes are NOT mutated
    assert "transA" not in node1.attributes

    # Explicit populate_defaults=True populates defaults in place
    validator.validate_graph(graph, populate_defaults=True)
    assert node1.attributes.get("transA") == 0


def test_validator_default_registry() -> None:
    """Test that Validator defaults to ONNX_REGISTRY."""
    v = Validator()
    assert v.registry is not None
    assert "Add" in v.registry

    # Test that a custom primary registry overriding a stablehlo op is respected
    from ml_switcheroo_ir import LogicalNode
    from ml_switcheroo_ir.schema.custom_ops import CustomOpSchema

    s = CustomOpSchema(name="dot_general", domain="stablehlo", attributes=[], inputs=[], outputs=[])
    v_custom = Validator(registry={"dot_general": s.to_op_schema()})
    n = LogicalNode(id="x", op_type="dot_general", domain="stablehlo")
    assert v_custom._get_schema(n) is not None


def test_validator_list_type_check_errors(validator: Validator) -> None:
    """Test invalid list items.

    Args:
        validator (Validator): Mock-registry Validator fixture.
    """
    node = LogicalNode(id="conv1", op_type="Conv", attributes={"kernel_shape": [3, "3"]})
    errors = validator.validate_attribute_types(node)
    assert len(errors) == 1
    assert errors[0].attribute == "kernel_shape"

    # Int type valid check
    node2 = LogicalNode(id="conv2", op_type="Conv", attributes={"group": 2.5, "kernel_shape": [3]})
    errors2 = validator.validate_attribute_types(node2)
    assert errors2[0].attribute == "group"

    # Float type valid check
    node3 = LogicalNode(id="gemm1", op_type="Gemm", attributes={"alpha": "2.0"})
    errors3 = validator.validate_attribute_types(node3)
    assert errors3[0].attribute == "alpha"

    # Float type with int value (should be valid)
    node4 = LogicalNode(id="gemm2", op_type="Gemm", attributes={"alpha": 2})
    errors4 = validator.validate_attribute_types(node4)
    assert not errors4


def test_validator_various_types() -> None:
    """Test string, List[float], List[str], bool, etc."""
    mock_reg = {
        "TestOp": OpSchema(
            name="TestOp",
            domain="ai.onnx",
            version=1,
            attributes={
                "s": OpAttribute(name="s", type="str", required=False, default=""),
                "lf": OpAttribute(name="lf", type="List[float]", required=False, default=[]),
                "ls": OpAttribute(name="ls", type="List[str]", required=False, default=[]),
                "b": OpAttribute(name="b", type="bool", required=False, default=False),
                "any": OpAttribute(name="any", type="Any", required=False, default=None),
            },
            inputs=[],
            outputs=[],
        )
    }
    v = Validator(registry=mock_reg)

    # Valid
    node_valid = LogicalNode(
        id="n1",
        op_type="TestOp",
        attributes={"s": "ok", "lf": [1.0, 2.0], "ls": ["a"], "b": True, "any": {}},
    )
    assert not v.validate_attribute_types(node_valid)

    # Invalid str
    assert v.validate_attribute_types(LogicalNode(id="n", op_type="TestOp", attributes={"s": 1}))
    # Invalid List[float]
    assert v.validate_attribute_types(LogicalNode(id="n", op_type="TestOp", attributes={"lf": ["a"]}))
    assert v.validate_attribute_types(LogicalNode(id="n", op_type="TestOp", attributes={"lf": 1.0}))
    # Invalid List[str]
    assert v.validate_attribute_types(LogicalNode(id="n", op_type="TestOp", attributes={"ls": [1]}))
    assert v.validate_attribute_types(LogicalNode(id="n", op_type="TestOp", attributes={"ls": "a"}))
    # Invalid bool
    assert v.validate_attribute_types(LogicalNode(id="n", op_type="TestOp", attributes={"b": 1}))


def test_validator_custom_domain() -> None:
    """Test that custom domains don't trigger kind validation errors if not checked."""
    v = Validator(registry=MOCK_REGISTRY)
    node = LogicalNode(id="n1", op_type="CustomOp", domain="ai.custom")
    assert not v.validate_kind(node)


def test_validator_missing_kind_methods() -> None:
    """Test methods return quickly if kind not in registry."""
    v = Validator(registry=MOCK_REGISTRY)
    node = LogicalNode(id="n1", op_type="UnknownOp")
    assert not v.validate_required_attributes(node)
    assert not v.validate_attribute_types(node)

    # populate defaults returns quickly
    v.populate_defaults(node)
    assert not node.attributes


def test_validator_edge_source_idx_out_of_bounds() -> None:
    """Test validator catches out of bounds source_idx for multi-output nodes."""
    from ml_switcheroo_ir import LogicalEdge, LogicalGraph, LogicalNode
    from ml_switcheroo_ir.validator import Validator

    producer = LogicalNode(id="prod", op_type="Split", domain="ai.onnx", outputs=["out0", "out1"])
    consumer = LogicalNode(id="cons", op_type="Relu", domain="ai.onnx")

    edge = LogicalEdge(source="prod", target="cons", source_idx=2)
    graph = LogicalGraph(nodes={"prod": producer, "cons": consumer}, edges=[edge])

    validator = Validator()
    errors = validator.validate_graph(graph)

    assert any("out of bounds" in str(e.message) for e in errors), f"Errors were: {errors}"

    # Also test when edge.source is an explicit SSA output name with out-of-bounds source_idx
    from unittest import mock

    with mock.patch("ml_switcheroo_ir.LogicalGraph.edges", new_callable=mock.PropertyMock) as mock_edges:
        mock_edges.return_value = [
            LogicalEdge(source="prod", target="cons", source_idx=2),
            LogicalEdge(source="out1", target="cons", source_idx=2),
        ]
        errors_direct = validator.validate_graph(graph)
    assert len([e for e in errors_direct if "out of bounds" in str(e.message)]) == 2


def test_validator_exhaustive_attribute_types_subgraphs_cycles_and_arity() -> None:
    """Verify every AttributeType checker, nested subgraphs, cycles, outputs, arity, and suggested_fix."""
    import pytest

    from ml_switcheroo_ir import DType, LogicalGraph, LogicalNode, TensorSpec
    from ml_switcheroo_ir.schema.base import AttributeType, OpAttribute, OpSchema
    from ml_switcheroo_ir.validator import GroundingValidator, ValidationError, ValidationLevel, Validator

    rich_schema = OpSchema(
        name="RichOp",
        domain="ai.onnx",
        inputs=["x", "y"],
        outputs=["z"],
        attributes={
            "d": OpAttribute("d", AttributeType.DICT, False),
            "lst": OpAttribute("lst", AttributeType.LIST, False),
            "ten": OpAttribute("ten", AttributeType.TENSOR, False),
            "gr": OpAttribute("gr", AttributeType.GRAPH, False),
            "dt": OpAttribute("dt", AttributeType.DTYPE, False),
        },
    )
    v = Validator(registry={"RichOp": rich_schema, "Add": OpSchema("Add", "ai.onnx", 14, {}, ["A", "B"], ["C"])})

    from ml_switcheroo_ir.validation.schema_validator import _is_valid_graph_attr

    inner_sub = LogicalGraph(name="Sub", nodes={"s0": LogicalNode(id="s0", op_type="Input")}, outputs=["s0"])
    assert _is_valid_graph_attr(inner_sub)
    for valid_dt in (DType.float32, 1, "float16"):
        ok_node = LogicalNode(
            id="ok",
            op_type="RichOp",
            attributes={
                "d": {"k": 1},
                "lst": [1, "two"],
                "ten": TensorSpec(shape=(2,), dtype=DType.float32),
                "gr": {"nodes": {}},
                "dt": valid_dt,
            },
        )
        assert v.validate_node(ok_node) == []

    bad_attr_node = LogicalNode(
        id="bad_attrs",
        op_type="RichOp",
        attributes={"d": 5, "lst": "not_list", "ten": "not_tensor", "gr": 123, "dt": "not_a_dtype"},
    )
    attr_errs = v.validate_attribute_types(bad_attr_node)
    assert {e.attribute for e in attr_errs} == {"d", "lst", "ten", "gr", "dt"}
    assert len(v.validate_attribute_types(LogicalNode(id="bad_dt2", op_type="RichOp", attributes={"dt": True}))) == 1

    # Arity underflow, overflow, unresolved output, key != node.id, cycle, and nested subgraph errors
    bad_sub = LogicalGraph(
        name="BadSub",
        nodes={"add_under": LogicalNode(id="add_under", op_type="Add", inputs=["outer_in"])},
        outputs=["missing_sub_out"],
    )
    cyc_a = LogicalNode(id="ca", op_type="Add", inputs=["cb", "cb"], subgraphs={"body": bad_sub})
    cyc_b = LogicalNode(id="cb", op_type="Add", inputs=["ca", "ca", "ca"])
    cyclic_g = LogicalGraph(name="CyclicG", nodes={"ca": cyc_a, "cb": cyc_b}, inputs=["outer_in"], outputs=["bad_out"])

    g_errs = v.validate_graph(cyclic_g)
    messages = [e.message for e in g_errs]
    assert any("expects at least 2 positional input" in m for m in messages)
    assert any("accepts at most 2 positional input" in m for m in messages)
    assert any("does not resolve to any node" in m for m in messages)
    assert any("Cycle" in m or "cycle" in m for m in messages)

    edge_errs = v.validate_edges(cyclic_g)
    assert any("missing_sub_out" in e.message for e in edge_errs)

    corrupted_g = LogicalGraph(name="CorruptedKeyG", nodes={"ok_n": LogicalNode(id="ok_n", op_type="Input")})
    corrupted_g.nodes._data["wrong_key"] = LogicalNode(id="actual_id", op_type="Input")
    key_errs = v.validate_edges(corrupted_g)
    assert any("does not match LogicalNode.id" in e.message for e in key_errs)

    # Preserve suggested_fix on GroundingValidator conversion and validate(raise_on_error=True)
    gv = GroundingValidator(
        snapshot_manifest={"categories": {"ops": [{"name": "RichOp", "api_path": "ai.custom.RichOp"}]}}
    )
    mlir_custom = OpSchema("arith.custom_mul", "mlir.arith", 1, {}, ["a", "b"], ["c"])
    opt_none_schema = OpSchema(
        "OptNoneOp",
        "ai.onnx",
        1,
        {"opt_a": OpAttribute("opt_a", AttributeType.INT, required=False, default=None)},
        [],
        ["out"],
    )
    v_strict = Validator(
        registry={"RichOp": rich_schema, "arith.custom_mul": mlir_custom, "OptNoneOp": opt_none_schema},
        mlir_registry={"other.op": OpSchema("other.op", "mlir.other", 1, {}, ["a", "b"], ["c"])},
        level=ValidationLevel.STRICT,
        grounding_validator=gv,
    )
    typo_node = LogicalNode(id="typo", op_type="Rich0p", domain="ai.custom", shape_metadata=(1,))
    with pytest.raises(ValidationError) as exc_info:
        v_strict.validate(typo_node, raise_on_error=True)
    assert exc_info.value.suggested_fix == "ai.custom.RichOp"

    # Cover MLIR override in self.registry (line 415), CORE_MLIR_DIALECTS fallback arity (454),
    # optional None attribute (650), non-LogicalGraph subgraph entry (805->804, 1142->1141), and ISA arity skip (975)
    assert v_strict._get_schema(LogicalNode(id="mc", op_type="arith.custom_mul", domain="mlir.arith")) is mlir_custom
    v_core_custom = Validator(
        mlir_registry={"arith.brand_new": OpSchema("arith.brand_new", "mlir.arith", 1, {}, ["a", "b"], ["c"])}
    )
    errs_core = v_core_custom.validate_kind(
        LogicalNode(id="bn", op_type="brand_new", domain="arith", inputs=["only_one"])
    )
    assert any("expects at least 2 operands" in e.message for e in errs_core)
    assert (
        v_strict.validate_attribute_types(LogicalNode(id="on", op_type="OptNoneOp", attributes={"opt_a": None})) == []
    )
    isa_g = LogicalGraph(
        name="IsaSubMeta",
        nodes={
            "i0": LogicalNode(
                id="i0",
                op_type="FFMA",
                domain="nvidia_sass",
                inputs=["x_in", "x_in", "x_in"],
                shape_metadata=(1,),
                subgraphs={"meta": "not_a_graph"},
            ),
            "c0": LogicalNode(
                id="c0",
                op_type="CustomUnregistered",
                domain="ai.custom",
                inputs=["x_in"],
                shape_metadata=(1,),
            ),
        },
        inputs=["x_in"],
        outputs=["i0", "c0"],
    )
    assert v_core_custom.validate_edges(isa_g) == []
    _ = v_core_custom.validate_graph(isa_g)
