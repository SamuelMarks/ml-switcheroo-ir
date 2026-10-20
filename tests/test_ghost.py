"""Tests for the Ghost Protocol schema in ml_switcheroo_ir."""

import json

import pytest
from pydantic import BaseModel, ValidationError

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
    assert not hasattr(param2, "extra_field")


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

    class DummyModel(BaseModel):
        """Dummy docstring."""

        kind: ParameterKind

    model = DummyModel.model_validate_json('{"kind": "POSITIONAL_OR_KEYWORD"}')
    assert model.kind == ParameterKind.POSITIONAL_OR_KEYWORD

    serialized = json.loads(model.model_dump_json())
    assert serialized["kind"] == "POSITIONAL_OR_KEYWORD"

    with pytest.raises(ValidationError):
        DummyModel.model_validate_json('{"kind": "INVALID_KIND"}')


def test_semantic_tier_enum_roundtrip() -> None:
    """Test SemanticTier enumeration functional roundtrip and validation."""

    class DummyModel(BaseModel):
        """Dummy docstring."""

        tier: SemanticTier

    model = DummyModel.model_validate_json('{"tier": "neural_ops"}')
    assert model.tier == SemanticTier.NEURAL_OPS

    serialized = json.loads(model.model_dump_json())
    assert serialized["tier"] == "neural_ops"

    with pytest.raises(ValidationError):
        DummyModel.model_validate_json('{"tier": "nonexistent_tier"}')


def test_logic_op_enum_roundtrip() -> None:
    """Test LogicOp enumeration functional roundtrip and validation."""

    class DummyModel(BaseModel):
        """Dummy docstring."""

        op: LogicOp

    model = DummyModel.model_validate_json('{"op": "gte"}')
    assert model.op == LogicOp.GTE

    serialized = json.loads(model.model_dump_json())
    assert serialized["op"] == "gte"

    with pytest.raises(ValidationError):
        DummyModel.model_validate_json('{"op": "not_an_operator"}')
