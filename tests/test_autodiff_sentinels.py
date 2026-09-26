"""Tests for ZeroTangent and NoTangent autodiff sentinel nodes."""

from __future__ import annotations

from ml_switcheroo_ir import (
    DType,
    LogicalGraph,
    LogicalNode,
    NoTangent,
    ZeroTangent,
)
from ml_switcheroo_ir.types import (
    NoTangent as NoTangentFromTypes,
)
from ml_switcheroo_ir.types import (
    ZeroTangent as ZeroTangentFromTypes,
)
from ml_switcheroo_ir.validator import Validator


def test_zero_tangent_initialization() -> None:
    """Test ZeroTangent node initialization, attributes, and inheritance."""
    zt = ZeroTangent(id="zt_0", shape=(4, 8), dtype=DType.float32)

    assert isinstance(zt, LogicalNode)
    assert zt.id == "zt_0"
    assert zt.op_type == "ZeroTangent"
    assert zt.domain == "ml.switcheroo.ad"
    assert zt.shape_metadata == (4, 8)
    assert zt.dtype == DType.float32


def test_no_tangent_initialization() -> None:
    """Test NoTangent node initialization and inheritance."""
    nt = NoTangent(id="nt_0")

    assert isinstance(nt, LogicalNode)
    assert nt.id == "nt_0"
    assert nt.op_type == "NoTangent"
    assert nt.domain == "ml.switcheroo.ad"


def test_import_from_types() -> None:
    """Test importing ZeroTangent and NoTangent from ml_switcheroo_ir.types."""
    import pytest

    import ml_switcheroo_ir.types as ir_types

    zt = ZeroTangentFromTypes(id="zt_types")
    nt = NoTangentFromTypes(id="nt_types")

    assert isinstance(zt, ZeroTangent)
    assert isinstance(nt, NoTangent)

    with pytest.raises(AttributeError):
        _ = ir_types.UnknownAttr123


def test_autodiff_sentinels_strict_validation() -> None:
    """Test that ZeroTangent and NoTangent pass Validator under ValidationLevel.STRICT."""
    validator = Validator(strict=True)

    zt = ZeroTangent(id="zt", shape=(2, 4))
    nt = NoTangent(id="nt")

    graph = LogicalGraph(
        name="ADGraph",
        nodes={"zt": zt, "nt": nt},
        outputs=["zt", "nt"],
    )

    errors = validator.validate(graph)
    assert not errors

    # Check ungrounded/unknown op under ml.switcheroo.ad fails
    bad_ad_node = LogicalNode(
        id="bad_ad",
        op_type="UnknownTangentOp",
        domain="ml.switcheroo.ad",
        shape_metadata=(2, 4),
    )
    bad_graph = LogicalGraph(nodes={"bad": bad_ad_node}, outputs=["bad"])
    bad_errors = validator.validate(bad_graph)
    assert len(bad_errors) == 1
    assert "UnknownTangentOp" in bad_errors[0].message


def test_autodiff_sentinels_serialization_roundtrip() -> None:
    """Test serialization roundtrip (to_dict, from_dict, to_json, from_json) of autodiff sentinels."""
    zt = ZeroTangent(id="zt", shape=(2, 4), dtype=DType.float16)
    nt = NoTangent(id="nt")
    graph = LogicalGraph(nodes={"zt": zt, "nt": nt}, outputs=["zt", "nt"])

    # Dictionary roundtrip
    data = graph.to_dict()
    restored = LogicalGraph.from_dict(data)

    assert "zt" in restored.nodes
    assert restored.nodes["zt"].op_type == "ZeroTangent"
    assert restored.nodes["zt"].domain == "ml.switcheroo.ad"
    assert restored.nodes["zt"].shape_metadata == [2, 4] or restored.nodes[
        "zt"
    ].shape_metadata == (2, 4)
    assert restored.nodes["zt"].dtype == DType.float16

    assert "nt" in restored.nodes
    assert restored.nodes["nt"].op_type == "NoTangent"
    assert restored.nodes["nt"].domain == "ml.switcheroo.ad"

    # JSON roundtrip
    json_str = graph.to_json()
    from_json_graph = LogicalGraph.from_json(json_str)

    assert "zt" in from_json_graph.nodes
    assert "nt" in from_json_graph.nodes
