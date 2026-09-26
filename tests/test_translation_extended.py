"""Tests for extended cross-framework parameter and weight translations and transformer concepts."""

from __future__ import annotations

import pytest

from ml_switcheroo_ir.translation import ParameterTranslationEngine


def test_translate_weight_names_across_frameworks() -> None:
    """Test weight identifier translation for LayerNorm, RMSNorm, Conv2D, and Linear."""
    engine = ParameterTranslationEngine()

    # LayerNorm weights
    assert (
        engine.translate_weight_name("layer_norm", "weight", "torch", "jax") == "scale"
    )
    assert (
        engine.translate_weight_name("layer_norm", "weight", "torch", "keras3")
        == "gamma"
    )
    assert (
        engine.translate_weight_name("layer_norm", "weight", "torch", "tensorflow")
        == "gamma"
    )
    assert (
        engine.translate_weight_name("layer_norm", "weight", "torch", "mlx") == "weight"
    )
    assert (
        engine.translate_weight_name("layer_norm", "weight", "torch", "onnx") == "scale"
    )

    # LayerNorm bias
    assert (
        engine.translate_weight_name("layer_norm", "bias", "torch", "keras3") == "beta"
    )
    assert (
        engine.translate_weight_name("layer_norm", "bias", "torch", "tensorflow")
        == "beta"
    )
    assert engine.translate_weight_name("layer_norm", "bias", "torch", "onnx") == "bias"

    # RMSNorm weights
    assert engine.translate_weight_name("rms_norm", "weight", "torch", "jax") == "scale"
    assert (
        engine.translate_weight_name("rms_norm", "weight", "torch", "keras3") == "gamma"
    )

    # Convolution weights
    assert (
        engine.translate_weight_name("convolution", "weight", "torch", "jax") == "rhs"
    )
    assert (
        engine.translate_weight_name("conv2d", "weight", "torch", "keras3") == "kernel"
    )
    assert (
        engine.translate_weight_name("conv", "weight", "torch", "tensorflow")
        == "filters"
    )
    assert engine.translate_weight_name("convolution", "weight", "torch", "onnx") == "W"

    # Linear weights
    assert engine.translate_weight_name("linear", "weight", "torch", "jax") == "kernel"
    assert (
        engine.translate_weight_name("dense", "weight", "torch", "keras3") == "kernel"
    )
    assert engine.translate_weight_name("linear", "bias", "torch", "onnx") == "bias"

    # General op fallback
    assert engine.translate_weight_name("matmul", "mat1", "torch", "jax") == "a"


def test_permute_conv_weights() -> None:
    """Test convolution weight permutation across memory formats (OIHW, HWIO, HWOI)."""
    engine = ParameterTranslationEngine()

    # Identity permutation
    raw = [1, 2, 3]
    assert engine.permute_conv_weights(raw, "OIHW", "OIHW") is raw

    # Object with transpose method
    class MockTensorTranspose:
        """Mock tensor implementing transpose method."""

        def __init__(self, data: object) -> None:
            """Initialize with data."""
            self.data = data

        def transpose(self, perm: tuple[int, ...]) -> tuple[object, tuple[int, ...]]:
            """Transpose with permutation."""
            return (self.data, perm)

    t_trans = MockTensorTranspose("weight_data")
    res_trans = engine.permute_conv_weights(t_trans, "OIHW", "HWIO")
    assert res_trans == ("weight_data", (2, 3, 1, 0))

    # Object with permute method
    class MockTensorPermute:
        """Mock tensor implementing permute method."""

        def __init__(self, data: object) -> None:
            """Initialize with data."""
            self.data = data

        def permute(self, perm: tuple[int, ...]) -> tuple[object, tuple[int, ...]]:
            """Permute with permutation."""
            return (self.data, perm)

    t_perm = MockTensorPermute("weight_data")
    res_perm = engine.permute_conv_weights(t_perm, "OIHW", "HWOI")
    assert res_perm == ("weight_data", (2, 3, 0, 1))

    # Fallback tuple
    res_fallback = engine.permute_conv_weights("raw_tensor", "OIHW", "HWIO")
    assert res_fallback == ("raw_tensor", (2, 3, 1, 0))

    # Errors on invalid format length or characters
    with pytest.raises(ValueError, match="must have rank 4"):
        engine.permute_conv_weights(raw, "OIH", "HWIO")
    with pytest.raises(ValueError, match="permutations of 'OIHW'"):
        engine.permute_conv_weights(raw, "ABCD", "HWIO")


def test_split_and_merge_qkv_weights() -> None:
    """Test QKV weight splitting and merging for multi-head attention."""
    engine = ParameterTranslationEngine()

    num_heads = 4
    head_dim = 16
    proj_size = num_heads * head_dim  # 64
    combined_len = 3 * proj_size  # 192

    # Split list
    qkv_list = list(range(combined_len))
    q, k, v = engine.split_qkv_weights(qkv_list, num_heads, head_dim)
    assert len(q) == proj_size
    assert len(k) == proj_size
    assert len(v) == proj_size
    assert q == list(range(64))
    assert k == list(range(64, 128))
    assert v == list(range(128, 192))

    # Merge list
    merged = engine.merge_qkv_weights(q, k, v)
    assert merged == qkv_list

    # Object with split method
    class MockQkvTensor:
        """Mock QKV tensor implementing split method."""

        def split(self, size: int, dim: int = 0) -> list[str]:
            """Split into chunks."""
            return [
                f"chunk_q_{size}_{dim}",
                f"chunk_k_{size}_{dim}",
                f"chunk_v_{size}_{dim}",
            ]

    t_qkv = MockQkvTensor()
    sq, sk, sv = engine.split_qkv_weights(t_qkv, num_heads, head_dim)
    assert sq == "chunk_q_64_0"
    assert sk == "chunk_k_64_0"
    assert sv == "chunk_v_64_0"

    # Fallback merge and split
    fb_q, fb_k, fb_v = engine.split_qkv_weights("tensor_qkv", 1, 10)
    assert fb_q == ("q", "tensor_qkv")
    assert fb_k == ("k", "tensor_qkv")
    assert fb_v == ("v", "tensor_qkv")

    merged_fb = engine.merge_qkv_weights("wq", "wk", "wv", concat_dim=1)
    assert merged_fb == ("qkv", ("wq", "wk", "wv"), 1)

    # Invalid head count or dimension error
    with pytest.raises(ValueError, match="must be positive"):
        engine.split_qkv_weights(qkv_list, 0, 16)
    with pytest.raises(ValueError, match="must be positive"):
        engine.split_qkv_weights(qkv_list, 4, -1)


def test_concept_map_transformer_blocks() -> None:
    """Verify that concept_map.json defines mappings for transformer building blocks."""
    import json
    from pathlib import Path

    from ml_switcheroo_ir.snapshots import DEFAULT_SNAPSHOT_DIR

    cmap_path = Path(DEFAULT_SNAPSHOT_DIR) / "concept_map.json"
    with open(cmap_path, "r", encoding="utf-8") as f:
        cmap = json.load(f)

    for block in [
        "grouped_query_attention",
        "multi_head_latent_attention",
        "rope_rotary_position_embedding",
        "expert_moe_routing",
    ]:
        assert block in cmap
        assert "torch" in cmap[block] or "jax" in cmap[block]
