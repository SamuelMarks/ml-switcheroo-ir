"""Unit tests for ParameterTranslationEngine and zero-hallucination parameter translations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_switcheroo_ir.translation import ParameterTranslationEngine


def test_parameter_translation_engine_defaults() -> None:
    """Test ParameterTranslationEngine loads default translations and translates roles correctly."""
    engine = ParameterTranslationEngine()
    assert "matmul" in engine.translations

    # Matmul: torch input -> jax a / stablehlo lhs
    assert engine.translate_parameter("matmul", "input", "torch", "jax") == "a"
    assert engine.translate_parameter("matmul", "mat1", "torch", "stablehlo") == "lhs"
    assert engine.translate_parameter("matmul", "other", "torch", "numpy") == "b"

    # Reduction: dim -> axis, keepdim -> keepdims
    assert engine.translate_parameter("reduction", "dim", "torch", "jax") == "axis"
    assert (
        engine.translate_parameter("reduction", "keepdim", "torch", "numpy")
        == "keepdims"
    )

    # Normalization: eps -> epsilon, weight -> scale
    assert (
        engine.translate_parameter("normalization", "eps", "torch", "tf") == "epsilon"
    )
    assert (
        engine.translate_parameter("normalization", "weight", "torch", "jax") == "scale"
    )


def test_parameter_translation_engine_translate_attributes() -> None:
    """Test translate_attributes with valid parameters and strict zero-hallucination policy."""
    engine = ParameterTranslationEngine()

    torch_norm_attrs = {"eps": 1e-5, "weight": 1.0}
    tf_norm_attrs = engine.translate_attributes(
        operation="normalization",
        attributes=torch_norm_attrs,
        source_framework="torch",
        target_framework="tf",
    )
    assert tf_norm_attrs == {"epsilon": 1e-5, "gamma": 1.0}

    # Strict mode raises ValueError on hallucinated parameter
    with pytest.raises(ValueError, match="Ungrounded parameter 'hallucinated_param'"):
        engine.translate_attributes(
            operation="normalization",
            attributes={"hallucinated_param": 42},
            source_framework="torch",
            target_framework="tf",
            allow_passthrough=False,
        )

    # Passthrough mode preserves untranslated parameter
    passthrough_attrs = engine.translate_attributes(
        operation="normalization",
        attributes={"hallucinated_param": 42, "eps": 1e-5},
        source_framework="torch",
        target_framework="tf",
        allow_passthrough=True,
    )
    assert passthrough_attrs == {"hallucinated_param": 42, "epsilon": 1e-5}


def test_parameter_translation_engine_error_branches(tmp_path: Path) -> None:
    """Test error branches in ParameterTranslationEngine including missing operations and file errors.

    Args:
        tmp_path (Path): Temporary test directory fixture.
    """
    engine = ParameterTranslationEngine(translations={})

    # Operation not found
    with pytest.raises(KeyError, match="No parameter translations defined"):
        engine.translate_parameter("nonexistent_op", "x", "torch", "jax")

    custom_trans = {
        "custom_op": {
            "roles": {
                "input_role": {
                    "torch": ["src_x"],
                    "jax": [],  # Target framework has no mapping
                }
            }
        }
    }
    engine_custom = ParameterTranslationEngine(translations=custom_trans)

    # Role exists but target framework does not support it
    with pytest.raises(ValueError, match="does not support role 'input_role'"):
        engine_custom.translate_parameter("custom_op", "src_x", "torch", "jax")

    # Parameter not recognized in source framework
    with pytest.raises(
        KeyError, match="is not recognized as a valid 'torch' parameter"
    ):
        engine_custom.translate_parameter(
            "custom_op", "unregistered_param", "torch", "jax"
        )

    # Custom file loading branches
    valid_file = tmp_path / "valid_cmap.json"
    with open(valid_file, "w", encoding="utf-8") as f:
        json.dump({"_parameter_translations": {"op_from_file": {"roles": {}}}}, f)
    engine_file = ParameterTranslationEngine(concept_map_path=str(valid_file))
    assert "op_from_file" in engine_file.translations

    # Malformed JSON file
    bad_file = tmp_path / "bad_cmap.json"
    with open(bad_file, "w", encoding="utf-8") as f:
        f.write("{malformed json")
    engine_bad = ParameterTranslationEngine(concept_map_path=str(bad_file))
    assert engine_bad.translations == {}

    # Valid JSON without _parameter_translations
    no_trans_file = tmp_path / "no_trans.json"
    with open(no_trans_file, "w", encoding="utf-8") as f:
        json.dump({"other_key": 123}, f)
    engine_no_trans = ParameterTranslationEngine(concept_map_path=str(no_trans_file))
    assert engine_no_trans.translations == {}


def test_parameter_translation_engine_bundled_fallback() -> None:
    """Test ParameterTranslationEngine fallback to bundled concept_map when snapshot directory is missing."""
    from unittest.mock import patch

    with patch(
        "ml_switcheroo_ir.validator.DEFAULT_SNAPSHOT_DIR",
        "/nonexistent_snapshots_dir",
    ), patch(
        "ml_switcheroo_ir.translation.DEFAULT_SNAPSHOT_DIR",
        "/nonexistent_snapshots_dir",
    ):
        engine = ParameterTranslationEngine()
        assert "matmul" in engine.translations
        assert "normalization" in engine.translations
        assert (
            engine.translate_parameter("normalization", "eps", "torch", "tf")
            == "epsilon"
        )
