"""Tests for extended DTypes, quantization formats, and multi-output node modeling."""

from __future__ import annotations

import json
import pickle
from typing import Any

import pytest

from ml_switcheroo_ir import (
    AttributeValue,
    DType,
    LogicalGraph,
    LogicalNode,
    TensorShape,
    TensorSpec,
    eliminate_dead_nodes,
    topological_sort,
)
from ml_switcheroo_ir.validator import (
    Validator,
)


def test_extended_dtypes_and_quantization_vocabulary() -> None:
    """Test all extended DType enum members and modern quantization formats."""
    expected_types = {
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "float64": "float64",
        "int64": "int64",
        "int32": "int32",
        "int16": "int16",
        "int8": "int8",
        "uint64": "uint64",
        "uint32": "uint32",
        "uint16": "uint16",
        "uint8": "uint8",
        "bool": "bool",
        "complex64": "complex64",
        "complex128": "complex128",
        "float8_e4m3fn": "float8_e4m3fn",
        "float8_e5m2": "float8_e5m2",
        "int4": "int4",
        "uint4": "uint4",
        "int2": "int2",
    }
    for name, val in expected_types.items():
        enum_val = getattr(DType, name)
        assert enum_val == val
        assert DType(val) == enum_val
        assert isinstance(enum_val, str)


def test_nested_attribute_value_for_tablegen_ods() -> None:
    """Test nested dictionary and sequence types in AttributeValue."""
    nested_tablegen_attrs: dict[str, AttributeValue] = {
        "strides": [1, 2],
        "dilation": (1, 1),
        "nested_dict": {
            "sub_key": "sub_value",
            "nested_list": [10, 20, 30],
            "nested_mapping": {"a": 1.5, "b": True},
        },
        "flag": True,
        "scalar_int": 42,
        "scalar_float": 3.14,
    }
    node = LogicalNode(
        id="tablegen_op",
        op_type="custom_call",
        domain="stablehlo",
        attributes=nested_tablegen_attrs,
    )
    assert node.attributes["strides"] == [1, 2]
    assert node.attributes["flag"] is True
    assert isinstance(node.attributes["nested_dict"], dict)
    assert node.attributes["nested_dict"]["nested_list"] == [10, 20, 30]


def test_logical_node_outputs_default_and_explicit() -> None:
    """Test LogicalNode outputs defaulting to [self.id] and accepting explicit outputs."""
    # Omitted outputs defaults to [self.id]
    n_default = LogicalNode(id="conv1", op_type="Conv")
    assert n_default.outputs == ["conv1"]

    # Explicit multi-output SSA values (e.g. Split, BatchNorm, custom_call)
    n_multi = LogicalNode(
        id="split1",
        op_type="Split",
        outputs=["split_out0", "split_out1"],
    )
    assert n_multi.outputs == ["split_out0", "split_out1"]


def test_get_output_producer() -> None:
    """Test get_output_producer on LogicalGraph."""
    split_node = LogicalNode(
        id="split1",
        op_type="Split",
        outputs=["y0", "y1"],
    )
    relu_node = LogicalNode(
        id="relu1",
        op_type="Relu",
    )
    graph = LogicalGraph(nodes={"split1": split_node, "relu1": relu_node})

    # Producer of multi-output values
    assert graph.get_output_producer("y0") == split_node
    assert graph.get_output_producer("y1") == split_node

    # Producer of default output
    assert graph.get_output_producer("relu1") == relu_node

    # Non-existent output
    assert graph.get_output_producer("non_existent") is None


def test_get_inputs_and_get_outputs_with_multi_output_nodes() -> None:
    """Test get_inputs and get_outputs with SSA values from multi-output nodes."""
    split_node = LogicalNode(
        id="split1",
        op_type="Split",
        outputs=["y0", "y1"],
    )
    consumer0 = LogicalNode(
        id="c0",
        op_type="Relu",
        inputs=["y0"],
    )
    consumer1 = LogicalNode(
        id="c1",
        op_type="Sigmoid",
        inputs=["y1", "dangling_ssa"],
    )
    graph = LogicalGraph(
        nodes={
            "split1": split_node,
            "c0": consumer0,
            "c1": consumer1,
        }
    )

    # get_inputs resolves SSA producers
    assert graph.get_inputs("c0") == [split_node]
    # c1 has y1 (produced by split1) and dangling_ssa (unproduced)
    assert graph.get_inputs("c1") == [split_node]

    # get_outputs finds consumers of both node ID and explicit SSA output names
    outputs_split = graph.get_outputs("split1")
    assert {n.id for n in outputs_split} == {"c0", "c1"}

    # get_outputs for missing node
    assert graph.get_outputs("missing") == []


def test_topological_sort_with_multi_output_nodes() -> None:
    """Test topological_sort when downstream nodes depend on SSA outputs."""
    split_node = LogicalNode(
        id="split1",
        op_type="Split",
        outputs=["y0", "y1"],
    )
    c0 = LogicalNode(id="c0", op_type="Relu", inputs=["y0"])
    c1 = LogicalNode(id="c1", op_type="Sigmoid", inputs=["y1"])
    sink = LogicalNode(id="sink", op_type="Add", inputs=["c0", "c1"])

    graph = LogicalGraph(
        nodes={
            "sink": sink,
            "c1": c1,
            "c0": c0,
            "split1": split_node,
        }
    )

    order = topological_sort(graph)
    order_ids = [n.id for n in order]

    assert order_ids.index("split1") < order_ids.index("c0")
    assert order_ids.index("split1") < order_ids.index("c1")
    assert order_ids.index("c0") < order_ids.index("sink")
    assert order_ids.index("c1") < order_ids.index("sink")


def test_topological_sort_with_unresolved_ssa_input() -> None:
    """Test topological_sort when an SSA input cannot be resolved."""
    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Relu", inputs=["unresolved_ssa"])
    graph = LogicalGraph(nodes={"n1": n1, "n2": n2})
    order = topological_sort(graph)
    assert len(order) == 2


def test_serialization_roundtrip_with_outputs() -> None:
    """Test JSON serialization and deserialization preserving outputs attribute."""
    split_node = LogicalNode(
        id="split1",
        op_type="Split",
        outputs=["branch_a", "branch_b"],
    )
    graph = LogicalGraph(nodes={"split1": split_node})

    json_str = graph.to_json()
    deserialized = LogicalGraph.from_json(json_str)

    assert deserialized.nodes["split1"].outputs == ["branch_a", "branch_b"]


def test_from_json_outputs_omitted_and_null() -> None:
    """Test from_json handles omitted outputs and null outputs."""
    raw_json = json.dumps(
        {
            "name": "TestModel",
            "nodes": [
                {"id": "n1", "op_type": "Relu"},
                {"id": "n2", "op_type": "Sigmoid", "outputs": None},
            ],
        }
    )
    graph = LogicalGraph.from_json(raw_json)
    assert graph.nodes["n1"].outputs == ["n1"]
    assert graph.nodes["n2"].outputs == ["n2"]


def test_validator_multi_output_edge_validation() -> None:
    """Test Validator.validate_edges recognizes multi-output SSA names."""
    split_node = LogicalNode(
        id="split1",
        op_type="Split",
        outputs=["out_a", "out_b"],
    )
    consumer_valid = LogicalNode(
        id="c_valid",
        op_type="Relu",
        inputs=["out_a"],
    )
    consumer_invalid = LogicalNode(
        id="c_invalid",
        op_type="Relu",
        inputs=["out_missing"],
    )

    graph_valid = LogicalGraph(nodes={"split1": split_node, "c_valid": consumer_valid})
    validator = Validator()
    errors_valid = validator.validate_edges(graph_valid)
    assert errors_valid == []

    graph_invalid = LogicalGraph(
        nodes={"split1": split_node, "c_invalid": consumer_invalid}
    )
    errors_invalid = validator.validate_edges(graph_invalid)
    assert len(errors_invalid) == 1
    assert errors_invalid[0].node_id == "c_invalid"
    assert "out_missing" in errors_invalid[0].message


def test_dtype_from_str_and_format_conversions() -> None:
    """Test DType.from_str, to_torch_str, and to_onnx_type methods across all supported types."""
    # Test normalization map
    assert DType.from_str("float") == DType.float32
    assert DType.from_str("double") == DType.float64
    assert DType.from_str("half") == DType.float16
    assert DType.from_str("int") == DType.int32
    assert DType.from_str("long") == DType.int64
    assert DType.from_str("short") == DType.int16
    assert DType.from_str("byte") == DType.uint8
    assert DType.from_str("char") == DType.int8
    assert DType.from_str("boolean") == DType.bool
    assert DType.from_str("str") == DType.string
    assert DType.from_str("obj") == DType.object
    assert DType.from_str("torch.float32") == DType.float32
    assert DType.from_str("torch.float16") == DType.float16
    assert DType.from_str("torch.bfloat16") == DType.bfloat16
    assert DType.from_str("torch.float64") == DType.float64
    assert DType.from_str("torch.int64") == DType.int64
    assert DType.from_str("torch.int32") == DType.int32
    assert DType.from_str("torch.bool") == DType.bool

    # Test member lookup
    assert DType.from_str("qint8") == DType.qint8
    assert DType.from_str("QINT4") == DType.qint4
    assert DType.from_str("float8_e4m3b11fnuz") == DType.float8_e4m3b11fnuz

    with pytest.raises(ValueError, match="Unknown or unsupported DType"):
        DType.from_str("invalid_nonexistent_dtype")

    # to_torch_str
    assert DType.float32.to_torch_str() == "torch.float32"
    assert DType.bfloat16.to_torch_str() == "torch.bfloat16"

    # to_onnx_type
    assert DType.float32.to_onnx_type() == "FLOAT"
    assert DType.float16.to_onnx_type() == "FLOAT16"
    assert DType.bfloat16.to_onnx_type() == "BFLOAT16"
    assert DType.float64.to_onnx_type() == "DOUBLE"
    assert DType.int64.to_onnx_type() == "INT64"
    assert DType.int32.to_onnx_type() == "INT32"
    assert DType.int16.to_onnx_type() == "INT16"
    assert DType.int8.to_onnx_type() == "INT8"
    assert DType.uint64.to_onnx_type() == "UINT64"
    assert DType.uint32.to_onnx_type() == "UINT32"
    assert DType.uint16.to_onnx_type() == "UINT16"
    assert DType.uint8.to_onnx_type() == "UINT8"
    assert DType.bool.to_onnx_type() == "BOOL"
    assert DType.string.to_onnx_type() == "STRING"
    assert DType.complex64.to_onnx_type() == "COMPLEX64"
    assert DType.complex128.to_onnx_type() == "COMPLEX128"
    assert DType.float8_e4m3fn.to_onnx_type() == "FLOAT8E4M3FN"
    assert DType.float8_e4m3b11fnuz.to_onnx_type() == "FLOAT8E4M3FNUZ"
    assert DType.float8_e5m2.to_onnx_type() == "FLOAT8E5M2"
    assert DType.int4.to_onnx_type() == "INT4"
    assert DType.uint4.to_onnx_type() == "UINT4"
    assert DType.qint8.to_onnx_type() == "QINT8"


def test_tensor_shape_operations_and_properties() -> None:
    """Test TensorShape constructors, queries, dynamic checks, and dunders."""
    # From tuple
    s1 = TensorShape((1, 3, 224, 224))
    assert s1.rank == 4
    assert not s1.is_dynamic
    assert s1.static_shape == (1, 3, 224, 224)
    assert len(s1) == 4
    assert list(s1) == [1, 3, 224, 224]
    assert s1[1] == 3

    # From list
    s2 = TensorShape([2, 4])
    assert s2.dims == (2, 4)

    # From object with .dims attribute
    class DummyShapeObj:
        """Mock object providing dims tuple for shape normalization."""

        dims = (8, 16)

    s3 = TensorShape(DummyShapeObj())
    assert s3.dims == (8, 16)

    # From scalar/non-iterable
    s4 = TensorShape(42)
    assert s4.dims == (42,)

    # Dynamic shape with string
    sd1 = TensorShape(("B", 128, "T"))
    assert sd1.is_dynamic
    assert sd1.rank == 3
    with pytest.raises(ValueError, match="contains dynamic or symbolic dimensions"):
        _ = sd1.static_shape

    # Dynamic shape with negative dimension
    sd2 = TensorShape((-1, 64))
    assert sd2.is_dynamic
    with pytest.raises(ValueError, match="contains dynamic"):
        _ = sd2.static_shape


def test_tensor_spec_operations_and_properties() -> None:
    """Test TensorSpec constructors, properties, and static shape conversions."""
    # From TensorShape and string dtype
    shape_obj = TensorShape((10, 20))
    spec1 = TensorSpec(shape=shape_obj, dtype="float32", sparsity="dense")
    assert spec1.shape == (10, 20)
    assert spec1.dtype == DType.float32
    assert spec1.sparsity == "dense"
    assert not spec1.is_dynamic
    assert spec1.static_shape == (10, 20)
    assert spec1.rank == 2

    # Dynamic shape
    spec_dyn = TensorSpec(shape=["B", -1], dtype=DType.int32)
    assert spec_dyn.is_dynamic
    assert spec_dyn.rank == 2
    with pytest.raises(ValueError, match="has dynamic dimensions"):
        _ = spec_dyn.static_shape


def test_logical_node_shape_and_metadata_properties() -> None:
    """Test LogicalNode shape properties (shape, static_shape, is_dynamic_shape, rank)."""
    node_none = LogicalNode(id="n0", op_type="Relu")
    assert node_none.shape is None
    assert not node_none.is_dynamic_shape
    assert node_none.rank == 0
    with pytest.raises(ValueError, match="no shape metadata"):
        _ = node_none.static_shape

    node_static = LogicalNode(id="n1", op_type="Conv", shape_metadata=(1, 32, 14, 14))
    assert node_static.shape == TensorShape((1, 32, 14, 14))
    assert node_static.static_shape == (1, 32, 14, 14)
    assert not node_static.is_dynamic_shape
    assert node_static.rank == 4

    node_dyn = LogicalNode(id="n2", op_type="Gemm", shape_metadata=("B", 768))
    assert node_dyn.is_dynamic_shape
    assert node_dyn.rank == 2
    with pytest.raises(ValueError, match="contains dynamic"):
        _ = node_dyn.static_shape


def test_first_class_graph_inputs_and_initializers() -> None:
    """Test LogicalGraph first-class inputs, input_specs, and initializers."""
    spec_x = TensorSpec(shape=(1, 10), dtype=DType.float32)
    spec_w = TensorSpec(shape=(10, 20), dtype=DType.float32)

    node = LogicalNode(id="mm", op_type="MatMul", inputs=["x", "w"])
    graph = LogicalGraph(
        name="LinearModel",
        nodes={"mm": node},
        inputs=["x", "w"],
        input_specs={"x": spec_x, "w": spec_w},
        outputs=["mm"],
        initializers={"w": [0.1] * 200},
    )

    assert graph.inputs == ["x", "w"]
    assert graph.input_specs["x"] == spec_x
    assert graph.initializers["w"] == [0.1] * 200

    # Validator respects graph.inputs and initializers as valid edge sources
    validator = Validator()
    errors = validator.validate_edges(graph)
    assert errors == []

    # Roundtrip serialization
    json_data = graph.to_json()
    loaded = LogicalGraph.from_json(json_data)
    assert loaded.inputs == ["x", "w"]
    assert loaded.input_specs["x"].shape == (1, 10)
    assert loaded.input_specs["x"].dtype == DType.float32
    assert loaded.initializers["w"] == [0.1] * 200


def test_nested_subgraphs_and_transforms() -> None:
    """Test first-class subgraphs on LogicalNode and recursive Dead Code Elimination."""
    sub_n1 = LogicalNode(id="sub_live", op_type="Relu")
    sub_n2 = LogicalNode(id="sub_dead", op_type="Neg")
    sub_g = LogicalGraph(
        name="BodySubgraph",
        nodes={"sub_live": sub_n1, "sub_dead": sub_n2},
        outputs=["sub_live"],
    )

    outer_node = LogicalNode(
        id="loop",
        op_type="Loop",
        subgraphs={"body": sub_g},
        outputs=["loop_out"],
        device="cuda:0",
        stream="compute_stream_0",
        dtype=DType.float32,
        output_specs=[TensorSpec(shape=(1, 4), dtype=DType.float32)],
    )

    outer_graph = LogicalGraph(
        name="Outer",
        nodes={"loop": outer_node},
        outputs=["loop_out"],
    )

    # Verify eliminate_dead_nodes recursively processes subgraphs
    cleaned = eliminate_dead_nodes(outer_graph)
    cleaned_loop = cleaned.nodes["loop"]
    assert "body" in cleaned_loop.subgraphs
    cleaned_sub = cleaned_loop.subgraphs["body"]
    assert isinstance(cleaned_sub, LogicalGraph)
    assert "sub_live" in cleaned_sub.nodes
    assert "sub_dead" not in cleaned_sub.nodes
    assert cleaned_loop.device == "cuda:0"
    assert cleaned_loop.stream == "compute_stream_0"
    assert cleaned_loop.dtype == DType.float32
    assert len(cleaned_loop.output_specs) == 1

    # Roundtrip serialization of nested subgraphs
    serialized = outer_graph.to_json()
    restored = LogicalGraph.from_json(serialized)
    restored_loop = restored.nodes["loop"]
    assert "body" in restored_loop.subgraphs
    assert isinstance(restored_loop.subgraphs["body"], LogicalGraph)
    assert "sub_live" in restored_loop.subgraphs["body"].nodes
    assert restored_loop.device == "cuda:0"
    assert restored_loop.stream == "compute_stream_0"
    assert restored_loop.dtype == DType.float32


def test_legacy_subgraph_attributes_migration() -> None:
    """Test from_dict migrates legacy subgraph attributes into node.subgraphs."""
    sub_raw: dict[str, Any] = {
        "name": "LegacySub",
        "nodes": {"n1": {"id": "n1", "op_type": "Relu"}},
        "outputs": ["n1"],
    }
    raw_data: dict[str, Any] = {
        "name": "ModelWithLegacyAttrs",
        "nodes": {
            "node_body": {
                "id": "node_body",
                "op_type": "While",
                "attributes": {"body_subgraph": sub_raw},
            },
            "node_bwd": {
                "id": "node_bwd",
                "op_type": "CustomGrad",
                "attributes": {"bwd_graph": sub_raw},
            },
        },
    }
    g = LogicalGraph.from_dict(raw_data)
    assert "body" in g.nodes["node_body"].subgraphs
    assert isinstance(g.nodes["node_body"].subgraphs["body"], LogicalGraph)
    assert "bwd" in g.nodes["node_bwd"].subgraphs
    assert isinstance(g.nodes["node_bwd"].subgraphs["bwd"], LogicalGraph)


def test_tensor_shape_and_node_edge_coverage() -> None:
    """Test TensorShape equality, LogicalNode shape setters, and NodeDict pickle serialization."""
    ts = TensorShape((1, 2, 3))
    assert ts == (1, 2, 3)
    assert ts == [1, 2, 3]
    assert ts != "not_a_shape"
    assert ts != (1, 2)

    # Non-sequence shape_metadata property checks
    node_custom = LogicalNode(id="nc", op_type="Custom", shape_metadata=999)
    assert node_custom.shape is None
    assert node_custom.rank == 0

    # Float dimension in dynamic shape check
    node_float_dim = LogicalNode(id="nf", op_type="Custom", shape_metadata=(1.5, 2))
    assert node_float_dim.is_dynamic_shape

    # Shape property setter tests
    node_prop = LogicalNode(id="np", op_type="Relu")
    node_prop.shape = TensorShape((4, 5))
    assert node_prop.shape_metadata == (4, 5)

    node_prop.shape = (6, 7)
    assert node_prop.shape_metadata == (6, 7)

    node_prop.shape = [8, 9]
    assert node_prop.shape_metadata == (8, 9)

    node_prop.shape = 42
    assert node_prop.shape_metadata == 42

    # Duck-typed metadata.shape handling branches
    class MockMetaStr:
        """Mock metadata with string shape."""

        shape = "static_str"

    class MockMetaInt:
        """Mock metadata with integer shape."""

        shape = 123

    node_inner_str = LogicalNode(id="nis", op_type="Op", shape_metadata=MockMetaStr())
    assert node_inner_str.shape_metadata == "static_str"

    node_inner_int = LogicalNode(id="nii", op_type="Op", shape_metadata=MockMetaInt())
    assert node_inner_int.shape_metadata == 123

    node_str = LogicalNode(id="n_str", op_type="Custom", shape_metadata="not_a_seq")
    assert node_str.shape_metadata == "not_a_seq"

    node_dict = LogicalNode(
        id="n_dict", op_type="Custom", shape_metadata={"shape": (2, 2)}
    )
    assert node_dict.shape_metadata == {"shape": (2, 2)}

    # NodeDict pickle serialization roundtrip
    g = LogicalGraph(
        name="PickleGraph",
        nodes={"n1": LogicalNode(id="n1", op_type="Relu")},
    )
    pickled = pickle.dumps(g.nodes)
    unpickled = pickle.loads(pickled)
    assert "n1" in unpickled
    assert unpickled["n1"].op_type == "Relu"
