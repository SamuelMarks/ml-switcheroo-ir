"""Tests for the Ghost Protocol schema in ml_switcheroo_ir."""

import json

import pydantic
import pytest

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostParam,
    GhostRef,
    LogicOp,
    ParameterKind,
    SemanticTier,
    SnapshotEnvelope,
    StandardMap,
)


def test_ghost_param_model() -> None:
    """Test GhostParam serialization and deserialization."""
    param = GhostParam(
        name="x",
        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
        default="1",
        annotation="int",
    )
    assert param.name == "x"
    assert param.kind == ParameterKind.POSITIONAL_OR_KEYWORD
    assert param.default == "1"
    assert param.annotation == "int"

    # Test ignore extra fields
    param_dict = {
        "name": "y",
        "kind": "POSITIONAL_OR_KEYWORD",
        "extra_field": "ignore_me",
    }
    param2 = GhostParam.model_validate(param_dict)
    assert param2.name == "y"
    assert "extra_field" not in param2.model_dump()


def test_ghost_ref_model() -> None:
    """Test GhostRef serialization, deserialization, and methods."""
    param = GhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD)
    ref = GhostRef(name="foo", api_path="pkg.foo", kind="function", params=[param])

    assert ref.name == "foo"
    assert ref.api_path == "pkg.foo"
    assert ref.kind == "function"
    assert ref.schema_version == "2.0.0"
    assert not ref.has_varargs

    assert ref.has_arg("x") is True
    assert ref.has_arg("y") is False


def test_standard_map_model() -> None:
    """Test StandardMap serialization and defaults."""
    sm = StandardMap(api="foo.bar")
    assert sm.api == "foo.bar"
    assert sm.args is None
    assert sm.inject_args is None
    assert sm.requires_plugin is None
    assert sm.transformation_type is None
    assert sm.operator is None
    assert sm.pack_to_tuple is None
    assert sm.macro_template is None


def test_extended_ghost_ref_validation() -> None:
    """Test validating schema dicts into v2.0.0 ExtendedGhostRef."""
    v2_data = {
        "name": "new_func",
        "api_path": "new.func",
        "kind": "function",
        "has_varargs": False,
        "schema_version": "2.0.0",
        "params": [
            {
                "name": "c",
                "kind": "POSITIONAL_ONLY",
            }
        ],
    }

    ref = ExtendedGhostRef.model_validate(v2_data)
    assert ref.schema_version == "2.0.0"
    assert ref.name == "new_func"
    assert ref.params[0].kind == ParameterKind.POSITIONAL_ONLY


def test_extended_ghost_ref_no_params() -> None:
    """Test validating ghost ref with no params key."""
    data = {
        "name": "no_params",
        "api_path": "no.params",
        "kind": "function",
        "has_varargs": False,
    }
    ref = ExtendedGhostRef.model_validate(data)
    assert ref.schema_version == "2.0.0"
    assert len(ref.params) == 0


def test_extended_ghost_ref_param_no_kind() -> None:
    """Test validating ghost ref where a param lacks a kind."""
    bad_data = {
        "name": "bad_param",
        "api_path": "bad.param",
        "kind": "function",
        "params": [{"name": "x"}],
    }
    with pytest.raises(ValueError):
        ExtendedGhostRef.model_validate(bad_data)


def test_extended_ghost_param_canonical_constraints() -> None:
    """Test ExtendedGhostParam canonical allowed_dtypes and rank_constraint fields."""
    p1 = ExtendedGhostParam(
        name="p1",
        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
        allowed_dtypes=["float32", "bfloat16"],
        rank_constraint="==2",
    )
    assert p1.allowed_dtypes == ["float32", "bfloat16"]
    assert p1.rank_constraint == "==2"

    p2 = ExtendedGhostParam(
        name="p2",
        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
        allowed_dtypes=["int32"],
        rank_constraint=">=1",
    )
    assert p2.allowed_dtypes == ["int32"]
    assert p2.rank_constraint == ">=1"

    p3 = ExtendedGhostParam(
        name="p3",
        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
        rank_constraint=None,
    )
    assert p3.rank_constraint is None

    env = SnapshotEnvelope(
        target="mlir",
        operations=[{"name": "arith.addi"}],
        instructions=[{"name": "V_ADD_F32"}],
    )
    assert env.operations == [{"name": "arith.addi"}]
    assert env.instructions == [{"name": "V_ADD_F32"}]


def test_parameter_kind_enum_roundtrip() -> None:
    """Test ParameterKind enumeration functional roundtrip and validation."""

    class DummyModel(pydantic.BaseModel):
        """Dummy model wrapping ParameterKind.

        Attributes:
            kind (ParameterKind): Parameter kind under test.
        """

        kind: ParameterKind

    model = DummyModel.model_validate_json('{"kind": "POSITIONAL_OR_KEYWORD"}')
    assert model.kind == ParameterKind.POSITIONAL_OR_KEYWORD

    serialized = json.loads(model.model_dump_json())
    assert serialized["kind"] == "POSITIONAL_OR_KEYWORD"

    with pytest.raises(pydantic.ValidationError):
        DummyModel.model_validate_json('{"kind": "INVALID_KIND"}')


def test_semantic_tier_enum_roundtrip() -> None:
    """Test SemanticTier enumeration functional roundtrip and validation."""

    class DummyModel(pydantic.BaseModel):
        """Dummy model wrapping SemanticTier.

        Attributes:
            tier (SemanticTier): Semantic tier under test.
        """

        tier: SemanticTier

    model = DummyModel.model_validate_json('{"tier": "neural_ops"}')
    assert model.tier == SemanticTier.NEURAL_OPS

    serialized = json.loads(model.model_dump_json())
    assert serialized["tier"] == "neural_ops"

    with pytest.raises(pydantic.ValidationError):
        DummyModel.model_validate_json('{"tier": "nonexistent_tier"}')


def test_logic_op_enum_roundtrip() -> None:
    """Test LogicOp enumeration functional roundtrip and validation."""

    class DummyModel(pydantic.BaseModel):
        """Dummy model wrapping LogicOp.

        Attributes:
            op (LogicOp): Logic operator under test.
        """

        op: LogicOp

    model = DummyModel.model_validate_json('{"op": "gte"}')
    assert model.op == LogicOp.GTE

    serialized = json.loads(model.model_dump_json())
    assert serialized["op"] == "gte"

    with pytest.raises(pydantic.ValidationError):
        DummyModel.model_validate_json('{"op": "not_an_operator"}')


def test_logic_constraints_isa_aliases_and_standard_map_integration() -> None:
    """Verify LogicConstraint, ExtendedGhostParam.check_value_constraints, GhostIsaRef aliases, and StandardMap."""
    from ml_switcheroo_ir.schema.ghost import (
        LogicConstraint,
    )

    assert LogicConstraint(op=LogicOp.EQ, target=4).evaluate(4)
    assert not LogicConstraint(op=LogicOp.EQ, target=1).evaluate(True)
    assert LogicConstraint(op=LogicOp.NEQ, target=1).evaluate(True)
    assert LogicConstraint(op=LogicOp.GT, target=2).evaluate(5)
    assert LogicConstraint(op=LogicOp.LT, target=5).evaluate(2)
    assert LogicConstraint(op=LogicOp.GTE, target=2).evaluate(2)
    assert LogicConstraint(op=LogicOp.LTE, target=2).evaluate(2)
    assert not LogicConstraint(op=LogicOp.GT, target="bad").evaluate(5)
    assert LogicConstraint(op=LogicOp.IN, target=["a", "b"]).evaluate("a")
    assert not LogicConstraint(op=LogicOp.IN, target="not_list").evaluate("a")
    assert LogicConstraint(op=LogicOp.NOT_IN, target=["a", "b"]).evaluate("c")
    assert LogicConstraint(op=LogicOp.IS_TYPE, target="int").evaluate(3)
    assert not LogicConstraint(op=LogicOp.IS_TYPE, target="int").evaluate(True)
    assert LogicConstraint(op=LogicOp.IS_TYPE, target="bool").evaluate(False)
    assert LogicConstraint(op=LogicOp.IS_TYPE, target="float").evaluate(1.5)
    assert LogicConstraint(op=LogicOp.IS_TYPE, target="str").evaluate("ok")
    assert LogicConstraint(op=LogicOp.IS_TYPE, target="list").evaluate([1])
    assert LogicConstraint(op=LogicOp.IS_TYPE, target="dict").evaluate({"k": 1})
    assert LogicConstraint(op=LogicOp.IS_TYPE, target="none").evaluate(None)
    assert not LogicConstraint(op=LogicOp.IS_TYPE, target="unknown_kind").evaluate(1)

    param = ExtendedGhostParam(
        name="mode",
        kind=ParameterKind.KEYWORD_ONLY,
        is_mandatory=True,
        allowed_values=["fast", "exact"],
        constraints=[LogicConstraint(op=LogicOp.IS_TYPE, target="str")],
    )
    assert not param.check_value_constraints(None)
    assert not param.check_value_constraints("invalid_mode")
    assert param.check_value_constraints("fast")

    opt_param = ExtendedGhostParam(name="opt", kind=ParameterKind.KEYWORD_ONLY, is_mandatory=False)
    assert opt_param.check_value_constraints(None)


def test_ghost_isa_aliases_and_standard_map_integration() -> None:
    """Verify GhostIsaRef alias folding and ParameterTranslationEngine StandardMap integration."""
    from ml_switcheroo_ir.schema.ghost import (
        GhostIsaRef,
        IsaControlCodesSpec,
        StandardMap,
        VopdProfileSpec,
    )
    from ml_switcheroo_ir.translation import ParameterTranslationEngine

    # GhostIsaRef alias folding branches
    r_str_prof = GhostIsaRef(
        name="V_ADD_F32",
        api_path="rdna.V_ADD_F32",
        kind="function",
        vopd_profile="Y",
        allowed_modifiers=["clamp"],
    )
    assert r_str_prof.vopd_slot == "Y"
    assert r_str_prof.allowed_modifiers == ["clamp"]

    r_preset_slot = GhostIsaRef(
        name="V_PRESET",
        api_path="rdna.V_PRESET",
        kind="function",
        vopd_profile={"slot": "X"},
        vopd_slot="Y",
        execution_latency="not_int",
        scoreboard_barrier_mask=2,
    )
    base_ref = GhostRef(name="V_BASE", api_path="rdna.V_BASE", kind="function")
    assert GhostIsaRef.model_validate(base_ref, from_attributes=True).name == "V_BASE"
    assert r_preset_slot.vopd_slot == "X"
    r_lat_only = GhostIsaRef(name="V_LAT", api_path="rdna.V_LAT", kind="function", execution_latency=4)
    assert r_lat_only.execution_latency == 4

    r_model_prof = GhostIsaRef(
        name="V_MUL_F32",
        api_path="rdna.V_MUL_F32",
        kind="function",
        vopd_profile=VopdProfileSpec(slot=None),
        vopd_slot="X",
        control_codes={"stall_count": 2},
        execution_latency=(4, 8),
        scoreboard_barrier_mask="0x1",
    )
    assert r_model_prof.vopd_slot == "X"
    assert r_model_prof.execution_latency == (4, 8)
    assert r_model_prof.scoreboard_barrier_mask == "0x1"

    r_dict_prof = GhostIsaRef(
        name="V_SUB_F32",
        api_path="rdna.V_SUB_F32",
        kind="function",
        vopd_profile={"can_pair": True},
        vopd_slot="BOTH",
    )
    assert r_dict_prof.vopd_slot == "BOTH"
    assert GhostIsaRef.model_validate(r_dict_prof).vopd_slot == "BOTH"
    assert IsaControlCodesSpec(execution_latency=6, scoreboard_barrier_mask=3).latency == 6

    # StandardMap integration with ParameterTranslationEngine
    engine = ParameterTranslationEngine()
    smap = StandardMap(
        api="torch.optim.AdamW",
        args={"b1": "beta1", "b2": "beta2", "drop_me": None, "const_eps": 1e-8},
        pack_to_tuple="betas",
        inject_args={"foreach": True},
    )
    engine.register_standard_map("custom_adam", "torch", smap)
    engine.register_standard_map("custom_adam", "torch", smap)
    engine.register_standard_map("custom_adam", "torch", StandardMap(api="torch.optim.AdamW", args=None))
    assert engine.translations["custom_adam"].roles["b1"]["torch"] == ["betas"]
    mapped = engine.apply_standard_map(
        smap, {"b1": 0.9, "b2": 0.999, "drop_me": 0, "const_eps": 0.0, "extra": 5}, allow_passthrough=True
    )
    assert mapped == {"betas": (0.9, 0.999), "const_eps": 1e-8, "extra": 5, "foreach": True}

    smap_rename = StandardMap(api="jax.nn.gelu", args={"approx": "approximate"})
    engine.register_standard_map("custom_gelu", "jax", smap_rename)
    assert engine.apply_standard_map(smap_rename, {"approx": True}) == {"approximate": True}
    with pytest.raises(ValueError, match="Ungrounded attribute"):
        engine.apply_standard_map(smap_rename, {"unmapped_arg": 1})
