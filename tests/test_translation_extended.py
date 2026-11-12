"""Tests for extended cross-framework parameter and weight translations and transformer concepts."""

from __future__ import annotations

import pytest

from ml_switcheroo_ir.translation import ParameterTranslationEngine


def test_translate_weight_names_across_frameworks() -> None:
    """Test weight identifier translation for LayerNorm, RMSNorm, Conv2D, and Linear."""
    engine = ParameterTranslationEngine()

    # LayerNorm weights
    assert engine.translate_weight_name("layer_norm", "weight", "torch", "jax") == "scale"
    assert engine.translate_weight_name("layer_norm", "weight", "torch", "keras3") == "gamma"
    assert engine.translate_weight_name("layer_norm", "weight", "torch", "tensorflow") == "gamma"
    assert engine.translate_weight_name("layer_norm", "weight", "torch", "mlx") == "weight"
    assert engine.translate_weight_name("layer_norm", "weight", "torch", "onnx") == "scale"

    # LayerNorm bias
    assert engine.translate_weight_name("layer_norm", "bias", "torch", "keras3") == "beta"
    assert engine.translate_weight_name("layer_norm", "bias", "torch", "tensorflow") == "beta"
    assert engine.translate_weight_name("layer_norm", "bias", "torch", "onnx") == "bias"

    # RMSNorm weights
    assert engine.translate_weight_name("rms_norm", "weight", "torch", "jax") == "scale"
    assert engine.translate_weight_name("rms_norm", "weight", "torch", "keras3") == "gamma"

    # Convolution weights
    assert engine.translate_weight_name("convolution", "weight", "torch", "jax") == "rhs"
    assert engine.translate_weight_name("conv2d", "weight", "torch", "keras3") == "kernel"
    assert engine.translate_weight_name("conv", "weight", "torch", "tensorflow") == "filters"
    assert engine.translate_weight_name("convolution", "weight", "torch", "onnx") == "W"

    # Linear weights
    assert engine.translate_weight_name("linear", "weight", "torch", "jax") == "kernel"
    assert engine.translate_weight_name("dense", "weight", "torch", "keras3") == "kernel"
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

        def __init__(self, data: str) -> None:
            """Initialize with data.

            Args:
                data (str): Underlying payload.
            """
            self.data = data

        def transpose(self, perm: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
            """Transpose with permutation.

            Args:
                perm (tuple[int, ...]): Permutation axes.

            Returns:
                tuple[str, tuple[int, ...]]: Data and permutation tuple.
            """
            return (self.data, perm)

    t_trans = MockTensorTranspose("weight_data")
    res_trans = engine.permute_conv_weights(t_trans, "OIHW", "HWIO")
    assert res_trans == ("weight_data", (2, 3, 1, 0))

    # Object with permute method
    class MockTensorPermute:
        """Mock tensor implementing permute method."""

        def __init__(self, data: str) -> None:
            """Initialize with data.

            Args:
                data (str): Underlying payload.
            """
            self.data = data

        def permute(self, perm: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
            """Permute with permutation.

            Args:
                perm (tuple[int, ...]): Permutation axes.

            Returns:
                tuple[str, tuple[int, ...]]: Data and permutation tuple.
            """
            return (self.data, perm)

    t_perm = MockTensorPermute("weight_data")
    res_perm = engine.permute_conv_weights(t_perm, "OIHW", "HWOI")
    assert res_perm == ("weight_data", (2, 3, 0, 1))

    # Real 4D nested list permutation: shape (2, 1, 1, 3) in OIHW -> (1, 3, 1, 2) in HWIO
    nested_4d = [[[[10, 20, 30]]], [[[40, 50, 60]]]]
    permuted_4d = engine.permute_conv_weights(nested_4d, "OIHW", "HWIO")
    assert permuted_4d == [[[[10, 40]], [[20, 50]], [[30, 60]]]]

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
    assert isinstance(q, list)
    assert isinstance(k, list)
    assert isinstance(v, list)
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
            """Split into chunks.

            Args:
                size (int): Chunk size.
                dim (int): Dimension along which to split.

            Returns:
                list[str]: Split chunk identifiers.
            """
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
    """Verify GroundingValidator concept_map ingestion and CUSTOM_OPS_REGISTRY transformer blocks."""
    from ml_switcheroo_ir import LogicalNode
    from ml_switcheroo_ir.schema.custom_ops import CUSTOM_OPS_REGISTRY
    from ml_switcheroo_ir.validator import GroundingValidator, Validator

    manifest = {
        "grouped_query_attention": {
            "torch": ["torch.nn.functional.scaled_dot_product_attention"],
            "jax": ["jax.nn.dot_product_attention"],
        },
        "multi_head_latent_attention": {
            "torch": ["ml.switcheroo.custom.MultiHeadLatentAttention"],
            "jax": ["ml.switcheroo.custom.MultiHeadLatentAttention"],
        },
        "rope_rotary_position_embedding": {
            "torch": ["ml.switcheroo.custom.RoPE"],
            "jax": ["ml.switcheroo.custom.RoPE"],
        },
        "expert_moe_routing": {
            "torch": ["ml.switcheroo.custom.MoETopKRouter"],
            "jax": ["ml.switcheroo.custom.MoETopKRouter"],
        },
    }

    gv = GroundingValidator(snapshot_manifest=manifest)
    for block in (
        "grouped_query_attention",
        "multi_head_latent_attention",
        "rope_rotary_position_embedding",
        "expert_moe_routing",
    ):
        assert block in gv.concept_map
        assert "torch" in gv.concept_map[block]
        assert "jax" in gv.concept_map[block]
        node = LogicalNode(id=f"n_{block}", op_type=block, domain="concept")
        assert gv.validate_grounding(node) == []

    validator = Validator()
    for custom_op_name in (
        "FlashAttention",
        "PagedAttention",
        "RoPE",
        "SwiGLU",
        "RMSNorm",
        "EmbeddingLookup",
    ):
        assert custom_op_name in CUSTOM_OPS_REGISTRY
        custom_node = LogicalNode(
            id=f"custom_{custom_op_name}",
            op_type=custom_op_name,
            domain="ml.switcheroo.custom",
        )
        assert validator.validate_kind(custom_node) == []


def test_optimizer_betas_packing_unpacking_and_gqa_2d_qkv() -> None:
    """Test b1/b2 <-> betas many-to-1 / 1-to-many translation and GQA + 2D QKV weight split/merge."""
    engine = ParameterTranslationEngine()

    # Optax b1 + b2 -> PyTorch betas=(0.9, 0.999)
    to_torch = engine.translate_attributes(
        "optimizer",
        {"learning_rate": 1e-3, "b1": 0.9, "b2": 0.999},
        source_framework="optax",
        target_framework="torch",
    )
    assert to_torch == {"lr": 1e-3, "betas": (0.9, 0.999)}

    # PyTorch betas=(0.9, 0.999) -> Optax b1=0.9, b2=0.999
    to_optax = engine.translate_attributes(
        "optimizer",
        {"lr": 1e-3, "betas": (0.9, 0.999)},
        source_framework="torch",
        target_framework="optax",
    )
    assert to_optax == {"learning_rate": 1e-3, "b1": 0.9, "b2": 0.999}

    # Unpack fallback branches: wrong tuple length, missing target framework, duplicate target keys
    custom_eng = ParameterTranslationEngine(
        translations={
            "custom_opt": {
                "roles": {
                    "r1": {"src": ["pair"], "tgt_same": ["same"]},
                    "r2": {"src": ["pair"], "tgt_same": ["same"]},
                }
            }
        }
    )
    assert engine.translate_attributes("optimizer", {"betas": (0.9,)}, "torch", "optax") == {"b1": (0.9,)}
    with pytest.raises(ValueError, match="Ungrounded parameter"):
        engine.translate_attributes("optimizer", {"betas": (0.9, 0.999)}, "torch", "unknown_fw")
    with pytest.raises(ValueError, match="No parameter translations defined"):
        engine.translate_attributes("unknown_op_xyz", {"x": 1}, "torch", "jax")
    assert custom_eng.translate_attributes("custom_opt", {"pair": (1, 2)}, "src", "tgt_same") == {"same": (1, 2)}

    # GQA 1D list & tuple QKV split (num_heads=4, num_kv_heads=1, head_dim=2 -> q=8, k=2, v=2)
    gqa_list = list(range(12))
    q_gqa, k_gqa, v_gqa = engine.split_qkv_weights(gqa_list, num_heads=4, head_dim=2, num_kv_heads=1)
    assert q_gqa == list(range(8)) and k_gqa == [8, 9] and v_gqa == [10, 11]
    assert engine.merge_qkv_weights(q_gqa, k_gqa, v_gqa) == gqa_list

    gqa_tup = tuple(range(12))
    qt, kt, vt = engine.split_qkv_weights(gqa_tup, num_heads=4, head_dim=2, num_kv_heads=1)
    assert qt == tuple(range(8)) and kt == (8, 9) and vt == (10, 11)
    assert engine.merge_qkv_weights(qt, kt, vt) == gqa_tup

    with pytest.raises(ValueError, match="does not match expected"):
        engine.split_qkv_weights([1, 2, 3], num_heads=2, head_dim=2)

    # 2D list and 2D tuple split_dim=1 / concat_dim=1
    qkv_2d_list = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    q2l, k2l, v2l = engine.split_qkv_weights(qkv_2d_list, num_heads=1, head_dim=2, split_dim=1)
    assert q2l == [[1, 2], [7, 8]] and k2l == [[3, 4], [9, 10]] and v2l == [[5, 6], [11, 12]]
    assert engine.merge_qkv_weights(q2l, k2l, v2l, concat_dim=1) == qkv_2d_list

    qkv_2d_tup = ((1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12))
    q2t, k2t, v2t = engine.split_qkv_weights(qkv_2d_tup, num_heads=1, head_dim=2, split_dim=-1)
    assert q2t == ((1, 2), (7, 8))
    assert engine.merge_qkv_weights(q2t, k2t, v2t, concat_dim=-1) == qkv_2d_tup

    with pytest.raises(ValueError, match="2D QKV sequence inner size"):
        engine.split_qkv_weights([[1, 2]], num_heads=1, head_dim=2, split_dim=1)

    with pytest.raises(ValueError, match="mismatched row counts"):
        engine.merge_qkv_weights([[1, 2]], [[3, 4], [5, 6]], [[7, 8]], concat_dim=1)
