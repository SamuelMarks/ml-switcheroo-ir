"""Tests for multi-output SSA unpacking and verification ergonomics."""

from __future__ import annotations

from ml_switcheroo_ir import (
    DType,
    LogicalGraph,
    LogicalNode,
    TensorSpec,
)


def test_get_output_name() -> None:
    """Test LogicalNode.get_output_name returns explicit name or default index notation."""
    node = LogicalNode(id="split1", op_type="Split", outputs=["chunk_0", "chunk_1"])

    assert node.get_output_name(0) == "chunk_0"
    assert node.get_output_name(1) == "chunk_1"
    assert node.get_output_name(2) == "split1:2"
    assert node.get_output_name(99) == "split1:99"

    empty_node = LogicalNode(id="relu1", op_type="Relu")
    assert empty_node.get_output_name(0) == "relu1"
    assert empty_node.get_output_name(1) == "relu1:1"


def test_has_multiple_outputs() -> None:
    """Test LogicalNode.has_multiple_outputs detection."""
    single_out = LogicalNode(id="n1", op_type="Relu", outputs=["y"])
    assert single_out.has_multiple_outputs is False

    multi_out = LogicalNode(id="split1", op_type="Split", outputs=["o1", "o2"])
    assert multi_out.has_multiple_outputs is True

    multi_spec = LogicalNode(
        id="lstm1",
        op_type="LSTM",
        output_specs=[
            TensorSpec(shape=(1, 10), dtype=DType.float32),
            TensorSpec(shape=(1, 10), dtype=DType.float32),
        ],
    )
    assert multi_spec.has_multiple_outputs is True


def test_get_producing_output_index() -> None:
    """Test LogicalGraph.get_producing_output_index resolution paths."""
    split = LogicalNode(id="split_node", op_type="Split", outputs=["part_a", "part_b"])
    relu = LogicalNode(id="relu_node", op_type="Relu", outputs=["relu_out"])
    graph = LogicalGraph(
        nodes={"split_node": split, "relu_node": relu},
        outputs=["part_a", "part_b", "relu_out"],
    )

    # 1. Output port by explicit name
    res1 = graph.get_producing_output_index("part_a")
    assert res1 is not None
    assert res1[0].id == "split_node"
    assert res1[1] == 0

    res2 = graph.get_producing_output_index("part_b")
    assert res2 is not None
    assert res2[0].id == "split_node"
    assert res2[1] == 1

    # 4. Unknown output name
    assert graph.get_producing_output_index("non_existent_output") is None
    assert graph.get_producing_output_index("missing_node:0") is None
