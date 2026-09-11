"""Tests for extended DTypes, quantization formats, and multi-output node modeling."""

from __future__ import annotations

import json

from ml_switcheroo_ir import (
    AttributeValue,
    DType,
    LogicalGraph,
    LogicalNode,
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
