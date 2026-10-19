"""Extended unit tests for validation rules and edge cases in Validator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo_ir import LogicalGraph, LogicalMesh, LogicalNode, PartitionSpec
from ml_switcheroo_ir.distributed import PartitionAxisElement
from ml_switcheroo_ir.schema.onnx_registry import OpAttribute, OpSchema
from ml_switcheroo_ir.validator import GroundingValidator, ValidationLevel, Validator


def test_validator_custom_registry_override() -> None:
    """Test passing explicit custom_registry to Validator."""
    custom_op = OpSchema(
        name="SpecialOp",
        domain="ml.switcheroo.custom",
        version=1,
        attributes={
            "factor": OpAttribute(
                name="factor", type="float", required=True, default=1.0
            )
        },
        inputs=["in"],
        outputs=["out"],
    )
    v = Validator(custom_registry={"SpecialOp": custom_op})
    node = LogicalNode(id="n1", op_type="SpecialOp", domain="ml.switcheroo.custom")

    assert not v.validate_kind(node)
    errors = v.validate_required_attributes(node)
    assert len(errors) == 1
    assert errors[0].attribute == "factor"


def test_validator_sharding_deeply_nested_and_none() -> None:
    """Test deeply nested axis tuples including None and unrecognized axes."""
    mesh = LogicalMesh(shape={"data": 2, "model": 4})
    v = Validator()

    nested_axes: tuple[PartitionAxisElement, ...] = (
        (("data", None), "model"),
        None,
    )
    spec_valid = PartitionSpec(axes=nested_axes)
    node_valid = LogicalNode(id="nv", op_type="Relu", sharding=spec_valid)
    assert not v.validate_sharding(node_valid, mesh)

    spec_invalid = PartitionSpec(axes=(("data", "bad_mesh_dim"),))
    node_invalid = LogicalNode(id="ni", op_type="Relu", sharding=spec_invalid)
    errors = v.validate_sharding(node_invalid, mesh)
    assert len(errors) == 1
    assert "bad_mesh_dim" in errors[0].message


def test_validator_custom_op_in_onnx_registry_fallback() -> None:
    """Test custom op that is present in base registry instead of custom_registry."""
    fake_onnx = {
        "CustomInOnnx": OpSchema(
            name="CustomInOnnx",
            domain="ml.switcheroo.custom",
            version=1,
            attributes={},
            inputs=[],
            outputs=[],
        )
    }
    v = Validator(registry=fake_onnx, custom_registry={})
    node = LogicalNode(id="c1", op_type="CustomInOnnx", domain="ml.switcheroo.custom")
    assert not v.validate_kind(node)
    assert v._get_schema(node) is fake_onnx["CustomInOnnx"]

    missing_node = LogicalNode(
        id="m1", op_type="NonExistent", domain="ml.switcheroo.custom"
    )
    assert v._get_schema(missing_node) is None

    other_node = LogicalNode(id="o1", op_type="CustomInOnnx", domain="ai.custom")
    assert v._get_schema(other_node) is fake_onnx["CustomInOnnx"]


def test_validator_sharding_empty_list_axis() -> None:
    """Test sharding axis containing an empty list or other non-string runtime types."""
    mesh = LogicalMesh(shape={"data": 2})
    v = Validator()
    malformed_axes: tuple[PartitionAxisElement, ...] = ([], 123)
    spec = PartitionSpec(axes=malformed_axes)
    node = LogicalNode(id="n_empty", op_type="Relu", sharding=spec)
    assert not v.validate_sharding(node, mesh)


def test_validator_custom_domain_pass_through() -> None:
    """Test that custom domain nodes pass through validate_kind without error."""
    v = Validator()
    node = LogicalNode(id="c1", op_type="MySpecialCustomOp", domain="custom")
    assert not v.validate_kind(node)


def test_grounding_validator_engine_exception_handling(tmp_path: Path) -> None:
    """Test exception resilience in GroundingValidator when GroundingEngine fails.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    test_dir = tmp_path / "snaps"
    test_dir.mkdir()
    gv = GroundingValidator(snapshots_dir=str(test_dir))
    assert isinstance(gv.grounded_symbols, dict)
    assert gv.grounded_symbols == {}

    gv2 = GroundingValidator(
        snapshot_manifest={"torch.relu": {"name": "relu", "api_path": "torch.relu"}},
        snapshots_dir=str(test_dir),
    )
    assert "relu" in gv2.grounded_symbols
    assert "torch.relu" in gv2.grounded_symbols

    with patch(
        "ml_ecosystem_snapshots.grounding.engine.GroundingEngine",
        side_effect=RuntimeError("init failed"),
    ):
        gv_fail = GroundingValidator()
        assert gv_fail._engine is None

    with (
        patch.dict("sys.modules", {"ml_ecosystem_snapshots.grounding.engine": None}),
        patch(
            "ml_framework_snapshots.grounding.engine.GroundingEngine",
            side_effect=RuntimeError("init failed"),
        ),
    ):
        gv_fail_fallback = GroundingValidator()
        assert gv_fail_fallback._engine is None

    broken_engine = MagicMock()
    broken_engine.get_symbol.side_effect = RuntimeError("engine lookup failed")
    broken_engine.suggest_closest_symbol.side_effect = RuntimeError(
        "engine suggest failed"
    )

    gv_resilient = GroundingValidator()
    gv_resilient._engine = broken_engine

    node = LogicalNode(id="n1", op_type="BrokenOp", domain="torch")
    errs = gv_resilient.validate_grounding(node)
    assert len(errs) == 1
    assert "BrokenOp" in errs[0].message


def test_validator_bool_rejected_for_int_and_float_attributes() -> None:
    """Verify bool values (True/False) are rejected for int, float, List[int], and List[float] attributes."""
    schema = OpSchema(
        name="TypedOp",
        domain="ai.onnx",
        version=1,
        attributes={
            "axis": OpAttribute(name="axis", type="int", required=True, default=None),
            "scale": OpAttribute(
                name="scale", type="float", required=True, default=None
            ),
            "pads": OpAttribute(
                name="pads", type="List[int]", required=False, default=None
            ),
            "ratios": OpAttribute(
                name="ratios", type="List[float]", required=False, default=None
            ),
        },
        inputs=["x"],
        outputs=["y"],
    )
    v = Validator(registry={"TypedOp": schema})
    bad_node = LogicalNode(
        id="n_bool",
        op_type="TypedOp",
        domain="ai.onnx",
        attributes={
            "axis": True,
            "scale": False,
            "pads": [1, True, 2],
            "ratios": [0.5, False],
        },
    )
    errs = v.validate_attribute_types(bad_node)
    invalid_attrs = {e.attribute for e in errs}
    assert invalid_attrs == {"axis", "scale", "pads", "ratios"}


def test_validator_domain_aliases_and_side_effect_free_option() -> None:
    """Verify webgpu_wgsl, rdna, sass, and wgsl domain aliases and populate_defaults=False."""
    v = Validator(level=ValidationLevel.WARNING)

    wgsl_node = LogicalNode(
        id="w1",
        op_type="storageStore",
        domain="webgpu_wgsl",
        attributes={"address_space": "storage, read"},
    )
    assert v._get_schema(wgsl_node) is not None
    wgsl_errs = v.validate_node(wgsl_node, populate_defaults=False)
    assert any("Mutating operation" in e.message for e in wgsl_errs)

    sass_n1 = LogicalNode(
        id="s1",
        op_type="HMMA.16816.F32",
        domain="sass",
        outputs=["R0"],
        attributes={"stall_count": 1},
    )
    sass_n2 = LogicalNode(
        id="s2",
        op_type="FADD",
        domain="sass",
        inputs=["R0"],
        outputs=["R2"],
    )
    g_sass = LogicalGraph(
        name="SassAliasGraph",
        nodes={"s1": sass_n1, "s2": sass_n2},
        outputs=["s2"],
    )
    sass_errs = v.validate_graph(g_sass, populate_defaults=False)
    assert any(e.attribute == "scoreboarding" for e in sass_errs)

    # Test populate_defaults=False leaves optional default attributes untouched
    op_with_def = OpSchema(
        name="OpWithDefault",
        domain="ai.onnx",
        version=1,
        attributes={
            "alpha": OpAttribute(
                name="alpha", type="float", required=False, default=1.0
            )
        },
        inputs=["x"],
        outputs=["y"],
    )
    v_def = Validator(registry={"OpWithDefault": op_with_def})
    pure_node = LogicalNode(id="p1", op_type="OpWithDefault", attributes={})
    v_def.validate(pure_node, populate_defaults=False)
    assert "alpha" not in pure_node.attributes
    v_def.validate(pure_node, populate_defaults=True)
    assert pure_node.attributes["alpha"] == 1.0
