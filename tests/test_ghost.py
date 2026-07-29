"""Tests for the Ghost Protocol schema in ml_switcheroo_ir."""

import pytest

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    LogicOp,
    ParameterKind,
    SemanticTier,
    StandardMap,
    migrate_ghost_ref,
)


def test_parameter_kind_enum() -> None:
    """Test ParameterKind enumeration values."""
    assert ParameterKind.POSITIONAL_ONLY.value == "POSITIONAL_ONLY"
    assert ParameterKind.POSITIONAL_OR_KEYWORD.value == "POSITIONAL_OR_KEYWORD"
    assert ParameterKind.VAR_POSITIONAL.value == "VAR_POSITIONAL"
    assert ParameterKind.KEYWORD_ONLY.value == "KEYWORD_ONLY"
    assert ParameterKind.VAR_KEYWORD.value == "VAR_KEYWORD"


def test_semantic_tier_enum() -> None:
    """Test SemanticTier enumeration values."""
    assert SemanticTier.ARRAY_API.value == "array"
    assert SemanticTier.NEURAL.value == "neural"
    assert SemanticTier.NEURAL_OPS.value == "neural_ops"
    assert SemanticTier.EXTRAS.value == "extras"


def test_logic_op_enum() -> None:
    """Test LogicOp enumeration values."""
    assert LogicOp.EQ.value == "eq"
    assert LogicOp.NEQ.value == "neq"
    assert LogicOp.GT.value == "gt"
    assert LogicOp.LT.value == "lt"
    assert LogicOp.GTE.value == "gte"
    assert LogicOp.LTE.value == "lte"
    assert LogicOp.IN.value == "in"
    assert LogicOp.NOT_IN.value == "not_in"
    assert LogicOp.IS_TYPE.value == "is_type"


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
    assert ref.schema_version == "1.2"
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


def test_migrate_ghost_ref_v1() -> None:
    """Test migrating v1.x schema dict to v2.x GhostRef."""
    v1_data = {
        "name": "old_func",
        "api_path": "old.func",
        "kind": "function",
        "has_varargs": True,
        "params": [
            {
                "name": "a",
                "kind": "POSITIONAL_OR_KEYWORD",
            },
            {
                "name": "b",
                "kind": "_ParameterKind.POSITIONAL_OR_KEYWORD",
            },
        ],
    }

    ref = migrate_ghost_ref(v1_data)

    assert ref.schema_version == "1.2"
    assert ref.name == "old_func"
    assert ref.has_varargs is True
    assert len(ref.params) == 2
    assert ref.params[0].name == "a"
    assert ref.params[0].kind == ParameterKind.POSITIONAL_OR_KEYWORD
    assert ref.params[1].name == "b"
    assert ref.params[1].kind == ParameterKind.POSITIONAL_OR_KEYWORD


def test_migrate_ghost_ref_v2() -> None:
    """Test migrating v2.x schema dict (noop behavior) to GhostRef."""
    v2_data = {
        "name": "new_func",
        "api_path": "new.func",
        "kind": "function",
        "has_varargs": False,
        "schema_version": "1.3",
        "params": [
            {
                "name": "c",
                "kind": "POSITIONAL_ONLY",
            }
        ],
    }

    ref = migrate_ghost_ref(v2_data)
    assert ref.schema_version == "1.3"
    assert ref.name == "new_func"
    assert ref.params[0].kind == ParameterKind.POSITIONAL_ONLY


def test_migrate_ghost_ref_no_params() -> None:
    """Test migrating ghost ref with no params key."""
    v1_data = {
        "name": "no_params",
        "api_path": "no.params",
        "kind": "function",
        "has_varargs": False,
    }
    ref = migrate_ghost_ref(v1_data)
    assert ref.schema_version == "1.2"
    assert len(ref.params) == 0


def test_migrate_ghost_ref_param_no_kind() -> None:
    """Test migrating ghost ref where a param lacks a kind (for branch coverage)."""
    v1_data = {
        "name": "bad_param",
        "api_path": "bad.param",
        "kind": "function",
        "params": [{"name": "x"}],
    }
    with pytest.raises(ValueError):
        # pydantic will raise validation error since kind is missing
        migrate_ghost_ref(v1_data)
