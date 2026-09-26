"""Engine for translating parameter names across ML frameworks and dialects without hallucinations."""

from __future__ import annotations

import json
import os
from typing import Any

from ml_switcheroo_ir.validator import DEFAULT_SNAPSHOT_DIR

_CANONICAL_FALLBACK_TRANSLATIONS: dict[str, Any] = {
    "matmul": {
        "roles": {
            "lhs": {
                "torch": ["input", "a"],
                "jax": ["lhs", "x"],
                "tf": ["a", "x"],
                "tensorflow": ["a", "x"],
                "stablehlo": ["lhs"],
                "numpy": ["a", "x1"],
            },
            "rhs": {
                "torch": ["other", "b"],
                "jax": ["rhs", "y"],
                "tf": ["b", "y"],
                "tensorflow": ["b", "y"],
                "stablehlo": ["rhs"],
                "numpy": ["b", "x2"],
            },
            "transpose_a": {
                "torch": ["transpose_a"],
                "jax": ["transpose_a"],
                "tf": ["transpose_a"],
                "tensorflow": ["transpose_a"],
                "stablehlo": ["transpose_a"],
            },
            "transpose_b": {
                "torch": ["transpose_b"],
                "jax": ["transpose_b"],
                "tf": ["transpose_b"],
                "tensorflow": ["transpose_b"],
                "stablehlo": ["transpose_b"],
            },
        }
    },
    "normalization": {
        "roles": {
            "scale": {
                "torch": ["weight"],
                "jax": ["scale"],
                "tensorflow": ["gamma"],
                "tf": ["gamma"],
                "stablehlo": ["scale"],
            },
            "bias": {
                "torch": ["bias"],
                "jax": ["bias"],
                "tensorflow": ["beta"],
                "tf": ["beta"],
                "stablehlo": ["bias"],
            },
            "epsilon": {
                "torch": ["eps"],
                "jax": ["epsilon", "eps"],
                "tensorflow": ["epsilon"],
                "tf": ["epsilon"],
                "stablehlo": ["epsilon"],
            },
        }
    },
    "reduction": {
        "roles": {
            "axis": {
                "torch": ["dim"],
                "jax": ["axis"],
                "tensorflow": ["axis"],
                "tf": ["axis"],
                "stablehlo": ["dimensions"],
                "numpy": ["axis"],
            },
            "keepdims": {
                "torch": ["keepdim"],
                "jax": ["keepdims"],
                "tensorflow": ["keepdims"],
                "tf": ["keepdims"],
                "stablehlo": ["keep_dimensions"],
                "numpy": ["keepdims"],
            },
        }
    },
    "convolution": {
        "roles": {
            "stride": {
                "torch": ["stride"],
                "jax": ["strides"],
                "tensorflow": ["strides"],
                "tf": ["strides"],
                "stablehlo": ["window_strides"],
            },
            "padding": {
                "torch": ["padding"],
                "jax": ["padding"],
                "tensorflow": ["padding"],
                "tf": ["padding"],
                "stablehlo": ["padding"],
            },
            "dilation": {
                "torch": ["dilation"],
                "jax": ["rhs_dilation"],
                "tensorflow": ["dilations"],
                "tf": ["dilations"],
                "stablehlo": ["rhs_dilation"],
            },
        }
    },
}


class ParameterTranslationEngine:
    """Translates parameter and attribute names across frameworks based on concept maps.

    Ensures zero hallucinated parameters by validating that every translated
    parameter corresponds to an empirical, grounded framework parameter role.
    """

    def __init__(
        self,
        translations: dict[str, Any] | None = None,
        concept_map_path: str | None = None,
    ) -> None:
        """Initialize translation engine with parameter translation mapping.

        Args:
            translations (Optional[dict[str, Any]]): Pre-loaded parameter translations mapping.
            concept_map_path (Optional[str]): Path to concept_map.json file.
        """
        self.translations: dict[str, Any] = translations or {}
        if not self.translations:
            self._load_default_translations(concept_map_path)

    def _load_default_translations(self, path: str | None = None) -> None:
        """Load default translations from concept_map.json in snapshot directory or package root.

        Args:
            path (Optional[str]): Optional path override.
        """
        if path is not None:
            candidates = [path]
        else:
            candidates = [
                os.path.join(DEFAULT_SNAPSHOT_DIR, "concept_map.json"),
                os.path.join(os.path.dirname(DEFAULT_SNAPSHOT_DIR), "concept_map.json"),
                os.path.join(os.path.dirname(__file__), "schema", "concept_map.json"),
                os.path.join(os.path.dirname(__file__), "concept_map.json"),
            ]
        for cand in candidates:
            if cand and os.path.isfile(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "_parameter_translations" in data and isinstance(
                        data["_parameter_translations"], dict
                    ):
                        self.translations.update(data["_parameter_translations"])
                        break
                except (json.JSONDecodeError, OSError):
                    pass
        if not self.translations and path is None:
            self.translations.update(_CANONICAL_FALLBACK_TRANSLATIONS)

    def translate_parameter(
        self,
        operation: str,
        param_name: str,
        source_framework: str,
        target_framework: str,
    ) -> str:
        """Translate a single parameter name from source framework to target framework.

        Args:
            operation (str): Canonical operation name (e.g. 'matmul', 'reduction', 'convolution', 'normalization').
            param_name (str): Parameter name in source framework.
            source_framework (str): Source framework identifier (e.g. 'torch', 'jax', 'tf', 'stablehlo', 'numpy').
            target_framework (str): Target framework identifier.

        Returns:
            str: Translated parameter name in target framework.

        Raises:
            KeyError: If operation or parameter role is not found in empirical translations.
            ValueError: If source or target framework has no mapping for the parameter role.
        """
        if operation not in self.translations:
            raise KeyError(
                f"No parameter translations defined for operation '{operation}'."
            )

        roles = self.translations[operation].get("roles", {})
        for role_name, fw_mappings in roles.items():
            src_params = fw_mappings.get(source_framework, [])
            if param_name in src_params:
                target_params = fw_mappings.get(target_framework, [])
                if not target_params:
                    raise ValueError(
                        f"Target framework '{target_framework}' does not support role '{role_name}' for operation '{operation}'."
                    )
                return str(target_params[0])

        raise KeyError(
            f"Parameter '{param_name}' is not recognized as a valid '{source_framework}' parameter for '{operation}'."
        )

    def translate_attributes(
        self,
        operation: str,
        attributes: dict[str, Any],
        source_framework: str,
        target_framework: str,
        allow_passthrough: bool = False,
    ) -> dict[str, Any]:
        """Translate all attributes/keyword arguments from source framework to target framework.

        Args:
            operation (str): Canonical operation name.
            attributes (dict[str, Any]): Dictionary of parameter names and values.
            source_framework (str): Source framework identifier.
            target_framework (str): Target framework identifier.
            allow_passthrough (bool): If True, parameters without translation are kept; if False,
                unrecognized parameters raise ValueError (zero-hallucination policy).

        Returns:
            dict[str, Any]: Translated attributes dictionary.

        Raises:
            ValueError: If unknown parameter is encountered and allow_passthrough is False.
        """
        result: dict[str, Any] = {}
        for k, v in attributes.items():
            try:
                translated_key = self.translate_parameter(
                    operation=operation,
                    param_name=k,
                    source_framework=source_framework,
                    target_framework=target_framework,
                )
                result[translated_key] = v
            except (KeyError, ValueError) as err:
                if allow_passthrough:
                    result[k] = v
                else:
                    raise ValueError(
                        f"Ungrounded parameter '{k}' during translation from '{source_framework}' to '{target_framework}': {err}"
                    ) from err
        return result

    def translate_weight_name(
        self,
        layer_type: str,
        weight_name: str,
        source_framework: str,
        target_framework: str,
    ) -> str:
        """Translate weight or bias tensor names across frameworks.

        Supports LayerNorm/RMSNorm (weight <-> scale <-> gamma, bias <-> beta),
        Convolution (weight <-> kernel <-> filters), and Linear layers.

        Args:
            layer_type (str): Layer category ('layer_norm', 'rms_norm', 'convolution', 'linear').
            weight_name (str): Original weight identifier.
            source_framework (str): Source framework identifier.
            target_framework (str): Target framework identifier.

        Returns:
            str: Translated weight identifier.

        Raises:
            KeyError: If layer_type or weight_name is unrecognized.
        """
        op = layer_type.lower().strip()
        if op in ("layernorm", "layer_norm"):
            op = "layer_norm"
        elif op in ("rmsnorm", "rms_norm"):
            op = "rms_norm"
        elif op in ("conv", "conv2d", "convolution"):
            op = "convolution"
        elif op in ("dense", "linear"):
            op = "linear"
        return self.translate_parameter(
            operation=op,
            param_name=weight_name,
            source_framework=source_framework,
            target_framework=target_framework,
        )

    def permute_conv_weights(
        self,
        weights: Any,
        source_format: str,
        target_format: str,
    ) -> Any:
        """Permute 2D convolution weight tensors across framework memory layouts.

        Permutes between PyTorch format (OIHW / [out_channels, in_channels, H, W])
        and TensorFlow/JAX format (HWIO / [H, W, in_channels, out_channels]) or HWOI.

        Args:
            weights (Any): Weight array, list, or tensor object with shape and transpose attributes.
            source_format (str): Source layout string (e.g. 'OIHW', 'HWIO', 'HWOI').
            target_format (str): Target layout string (e.g. 'OIHW', 'HWIO', 'HWOI').

        Returns:
            Any: Transposed weight array or list with target layout.

        Raises:
            ValueError: If source_format or target_format is invalid or ranks mismatch.
        """
        src = source_format.upper().strip()
        tgt = target_format.upper().strip()
        if len(src) != 4 or len(tgt) != 4:
            raise ValueError(f"Conv weight formats must have rank 4: '{src}', '{tgt}'.")
        if sorted(src) != sorted("OIHW") or sorted(tgt) != sorted("OIHW"):
            raise ValueError(
                f"Formats must be permutations of 'OIHW': '{src}', '{tgt}'."
            )
        if src == tgt:
            return weights

        perm = [src.index(ch) for ch in tgt]
        if hasattr(weights, "transpose"):
            return weights.transpose(tuple(perm))
        if hasattr(weights, "permute"):
            return weights.permute(tuple(perm))
        return (weights, tuple(perm))

    def split_qkv_weights(
        self,
        qkv_weight: Any,
        num_heads: int,
        head_dim: int,
        split_dim: int = 0,
    ) -> tuple[Any, Any, Any]:
        """Split combined QKV projection weights into separate Q, K, and V weights.

        Args:
            qkv_weight (Any): Combined QKV weight array or sequence.
            num_heads (int): Number of attention heads.
            head_dim (int): Dimensionality of each attention head.
            split_dim (int): Dimension along which to split (default 0).

        Returns:
            tuple[Any, Any, Any]: (q_weight, k_weight, v_weight) tuple.

        Raises:
            ValueError: If num_heads or head_dim is non-positive.
        """
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError(
                f"num_heads ({num_heads}) and head_dim ({head_dim}) must be positive."
            )

        proj_size = num_heads * head_dim
        if not isinstance(qkv_weight, (str, bytes)) and hasattr(qkv_weight, "split"):
            parts = qkv_weight.split(proj_size, dim=split_dim)
            return (parts[0], parts[1], parts[2])
        if isinstance(qkv_weight, (list, tuple)) and len(qkv_weight) == 3 * proj_size:
            q = qkv_weight[:proj_size]
            k = qkv_weight[proj_size : 2 * proj_size]
            v = qkv_weight[2 * proj_size :]
            return (q, k, v)
        return (("q", qkv_weight), ("k", qkv_weight), ("v", qkv_weight))

    def merge_qkv_weights(
        self,
        q_weight: Any,
        k_weight: Any,
        v_weight: Any,
        concat_dim: int = 0,
    ) -> Any:
        """Merge separate Q, K, and V projection weights into a single combined QKV tensor.

        Args:
            q_weight (Any): Query projection weight.
            k_weight (Any): Key projection weight.
            v_weight (Any): Value projection weight.
            concat_dim (int): Dimension along which to concatenate (default 0).

        Returns:
            Any: Merged QKV weight object.
        """
        if (
            isinstance(q_weight, list)
            and isinstance(k_weight, list)
            and isinstance(v_weight, list)
        ):
            return q_weight + k_weight + v_weight
        return ("qkv", (q_weight, k_weight, v_weight), concat_dim)
