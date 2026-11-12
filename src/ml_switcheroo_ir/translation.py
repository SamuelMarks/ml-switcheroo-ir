"""Engine for translating parameter names across ML frameworks and dialects without hallucinations."""

from __future__ import annotations

import copy
import json
import logging
import os
import typing
from collections.abc import Iterable, Mapping, Sequence

import pydantic
import typing_extensions

from ml_switcheroo_ir.schema.ghost import StandardMap
from ml_switcheroo_ir.snapshots import get_effective_snapshot_dir
from ml_switcheroo_ir.types import AttributeValue, JsonValue

_logger = logging.getLogger(__name__)

RoleFrameworkMap: typing_extensions.TypeAlias = dict[str, list[str]]


def _is_4d_nested_sequence(
    val: TensorLike,
) -> typing.TypeGuard[Sequence[Sequence[Sequence[Sequence[JsonValue]]]]]:
    """Check whether val is a non-empty 4D nested sequence (list or tuple).

    Args:
        val (TensorLike): Candidate tensor or sequence.

    Returns:
        typing.TypeGuard[Sequence[Sequence[Sequence[Sequence[JsonValue]]]]]: True if 4D nested sequence.
    """
    return (
        isinstance(val, (list, tuple))
        and len(val) > 0
        and isinstance(val[0], (list, tuple))
        and len(val[0]) > 0
        and isinstance(val[0][0], (list, tuple))
        and len(val[0][0]) > 0
        and isinstance(val[0][0][0], (list, tuple))
    )


class OperationTranslationSpec(pydantic.BaseModel):
    """Pydantic model for operation parameter role translations across frameworks.

    Attributes:
        roles (dict[str, RoleFrameworkMap]): Mapping of semantic role names to framework parameter lists.
        defaults (dict[str, dict[str, JsonValue]] | None): Optional default parameter values per framework.
    """

    model_config = pydantic.ConfigDict(extra="allow")

    roles: dict[str, RoleFrameworkMap] = pydantic.Field(default_factory=dict)
    defaults: dict[str, dict[str, JsonValue]] | None = None


class TranslationMap(pydantic.BaseModel):
    """Pydantic model wrapping a mapping of operation names to OperationTranslationSpec.

    Attributes:
        operations (dict[str, OperationTranslationSpec]): Operation translation specifications.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    operations: dict[str, OperationTranslationSpec] = pydantic.Field(default_factory=dict)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _coerce_mapping(
        cls,
        data: Mapping[str, OperationTranslationSpec | Mapping[str, JsonValue]],
    ) -> dict[str, dict[str, OperationTranslationSpec | Mapping[str, JsonValue]]]:
        """Coerce raw dictionary of operation specs into the 'operations' field.

        Args:
            data (Mapping[str, OperationTranslationSpec | Mapping[str, JsonValue]]):
                Input mapping.

        Returns:
            dict[str, dict[str, OperationTranslationSpec | Mapping[str, JsonValue]]]:
                Normalized dictionary for Pydantic validation.
        """
        return {"operations": dict(data)}

    def __getitem__(self, key: str) -> OperationTranslationSpec:
        """Retrieve operation translation spec by operation name.

        Args:
            key (str): Operation name.

        Returns:
            OperationTranslationSpec: Spec for the operation.
        """
        return self.operations[key]

    def __setitem__(
        self,
        key: str,
        value: Mapping[str, dict[str, RoleFrameworkMap]],
    ) -> None:
        """Set operation translation spec by operation name.

        Args:
            key (str): Operation name.
            value (Mapping[str, dict[str, RoleFrameworkMap]]): Spec dict.
        """
        self.operations[key] = OperationTranslationSpec.model_validate(value)

    def __contains__(self, key: str) -> bool:
        """Check whether an operation exists in the translation map.

        Args:
            key (str): Operation name.

        Returns:
            bool: True if operation is present.
        """
        return key in self.operations

    def __bool__(self) -> bool:
        """Return True if at least one operation is defined.

        Returns:
            bool: True if non-empty.
        """
        return bool(self.operations)

    def __eq__(
        self,
        other: object,
    ) -> bool:
        """Compare equality against another TranslationMap or dict.

        Args:
            other (object): Other value to compare.

        Returns:
            bool: True if equivalent.
        """
        return isinstance(other, Mapping) and {
            k: v.model_dump(exclude_none=True) for k, v in self.operations.items()
        } == dict(other)

    def values(self) -> Iterable[OperationTranslationSpec]:
        """Return operation translation specs.

        Returns:
            Iterable[OperationTranslationSpec]: Values view.
        """
        return self.operations.values()

    def clear(self) -> None:
        """Clear all operation translation entries."""
        self.operations.clear()

    def update(
        self,
        other: Mapping[str, Mapping[str, dict[str, RoleFrameworkMap]]],
    ) -> None:
        """Update operations from another mapping.

        Args:
            other (Mapping[str, Mapping[str, dict[str, RoleFrameworkMap]]]):
                Source mapping to merge.
        """
        for k, v in other.items():
            self[k] = v


@typing.runtime_checkable
class TensorLikeProtocol(typing.Protocol):
    """Structural protocol for tensor-like objects supporting layout and shape permutations.

    Attributes:
        shape (Sequence[int]): Tensor dimensions.
    """

    shape: Sequence[int]

    def transpose(self, *axes: int | tuple[int, ...]) -> TensorLikeProtocol:
        """Return tensor with permuted dimensions.

        Args:
            *axes (int | tuple[int, ...]): Target axis permutation.

        Returns:
            TensorLikeProtocol: Transposed tensor-like object.
        """

    def reshape(self, *shape: int | tuple[int, ...]) -> TensorLikeProtocol:
        """Return tensor reshaped to target dimensions.

        Args:
            *shape (int | tuple[int, ...]): Target dimensions.

        Returns:
            TensorLikeProtocol: Reshaped tensor-like object.
        """


@typing.runtime_checkable
class TransposableTensorProtocol(typing.Protocol):
    """Structural protocol for tensor-like objects providing .transpose()."""

    def transpose(self, perm: tuple[int, ...]) -> TransposableTensorProtocol | tuple[str, tuple[int, ...]] | JsonValue:
        """Return tensor with transposed dimensions.

        Args:
            perm (tuple[int, ...]): Target axis permutation.

        Returns:
            TransposableTensorProtocol | tuple[str, tuple[int, ...]] | JsonValue: Transposed tensor-like object.
        """


@typing.runtime_checkable
class PermutableTensorProtocol(typing.Protocol):
    """Structural protocol for PyTorch-like tensors providing .permute()."""

    def permute(self, perm: tuple[int, ...]) -> PermutableTensorProtocol | tuple[str, tuple[int, ...]] | JsonValue:
        """Return tensor with permuted dimensions.

        Args:
            perm (tuple[int, ...]): Target dimension order.

        Returns:
            PermutableTensorProtocol | tuple[str, tuple[int, ...]] | JsonValue: Permuted tensor-like object.
        """


@typing.runtime_checkable
class SplittableTensorProtocol(typing.Protocol):
    """Structural protocol for tensor-like objects providing .split()."""

    def split(self, split_size: int, dim: int = 0) -> Sequence[SplittableTensorProtocol | JsonValue]:
        """Split tensor into chunks along a dimension.

        Args:
            split_size (int): Size of each chunk.
            dim (int): Dimension along which to split.

        Returns:
            Sequence[SplittableTensorProtocol | JsonValue]: Sequence of split tensor chunks.
        """


NestedTensorSequence: typing_extensions.TypeAlias = Sequence[
    typing.Union[int, float, bool, str, "NestedTensorSequence"]
]
TensorLike: typing_extensions.TypeAlias = typing.Union[
    TensorLikeProtocol,
    TransposableTensorProtocol,
    PermutableTensorProtocol,
    SplittableTensorProtocol,
    list[list[list[list[JsonValue]]]],
    Sequence[Sequence[JsonValue]],
    NestedTensorSequence,
    Sequence[JsonValue],
    Sequence["TensorLike"],
    Mapping[str, JsonValue],
    tuple[str, tuple[int, ...]],
    JsonValue,
]


class PermutedWeightDescriptor(typing.NamedTuple):
    """Typed tuple descriptor for a symbolic weight permutation ``(weights, perm)``.

    Attributes:
        weights (TensorLike): Symbolic or raw weight reference.
        permutation (tuple[int, ...]): Target axis permutation.
    """

    weights: TensorLike
    permutation: tuple[int, ...]


class SplitQKVSliceDescriptor(typing.NamedTuple):
    """Typed tuple descriptor for a single symbolic Q, K, or V projection slice ``(role, qkv_weight)``.

    Attributes:
        role (str): Projection component role ('q', 'k', or 'v').
        qkv_weight (TensorLike): Combined QKV weight reference.
    """

    role: str
    qkv_weight: TensorLike


class MergedQKVWeightDescriptor(typing.NamedTuple):
    """Typed tuple descriptor for a merged QKV weight ``('qkv', (q, k, v), concat_dim)``.

    Attributes:
        tag (str): Descriptor tag ('qkv').
        weights (tuple[TensorLike, TensorLike, TensorLike]): Tuple of (q_weight, k_weight, v_weight).
        concat_dim (int): Concatenation dimension index.
    """

    tag: str
    weights: tuple[TensorLike, TensorLike, TensorLike]
    concat_dim: int


__all__ = [
    "MergedQKVWeightDescriptor",
    "NestedTensorSequence",
    "OperationTranslationSpec",
    "ParameterTranslationEngine",
    "PermutableTensorProtocol",
    "PermutedWeightDescriptor",
    "RoleFrameworkMap",
    "SplitQKVSliceDescriptor",
    "SplittableTensorProtocol",
    "TensorLike",
    "TensorLikeProtocol",
    "TranslationMap",
    "TransposableTensorProtocol",
]

_CANONICAL_FALLBACK_TRANSLATIONS: dict[str, dict[str, dict[str, RoleFrameworkMap]]] = {
    "matmul": {
        "roles": {
            "lhs": {
                "torch": ["input", "mat1", "a"],
                "jax": ["a", "lhs", "x"],
                "tf": ["a", "x"],
                "tensorflow": ["a", "x"],
                "stablehlo": ["lhs"],
                "numpy": ["a", "x1"],
            },
            "rhs": {
                "torch": ["other", "mat2", "b"],
                "jax": ["b", "rhs", "y"],
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
    "layer_norm": {
        "roles": {
            "weight": {
                "torch": ["weight"],
                "jax": ["scale"],
                "keras3": ["gamma"],
                "tensorflow": ["gamma"],
                "tf": ["gamma"],
                "mlx": ["weight"],
                "onnx": ["scale"],
            },
            "bias": {
                "torch": ["bias"],
                "jax": ["bias"],
                "keras3": ["beta"],
                "tensorflow": ["beta"],
                "tf": ["beta"],
                "mlx": ["bias"],
                "onnx": ["bias"],
            },
        }
    },
    "rms_norm": {
        "roles": {
            "weight": {
                "torch": ["weight"],
                "jax": ["scale"],
                "keras3": ["gamma"],
                "tensorflow": ["gamma"],
                "tf": ["gamma"],
                "mlx": ["weight"],
                "onnx": ["scale"],
            },
        }
    },
    "linear": {
        "roles": {
            "weight": {
                "torch": ["weight"],
                "jax": ["kernel"],
                "keras3": ["kernel"],
                "tensorflow": ["kernel"],
                "tf": ["kernel"],
                "mlx": ["weight"],
                "onnx": ["weight"],
            },
            "bias": {
                "torch": ["bias"],
                "jax": ["bias"],
                "keras3": ["bias"],
                "tensorflow": ["bias"],
                "tf": ["bias"],
                "mlx": ["bias"],
                "onnx": ["bias"],
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
            "weight": {
                "torch": ["weight"],
                "jax": ["rhs", "kernel"],
                "keras3": ["kernel"],
                "tensorflow": ["filters", "kernel"],
                "tf": ["filters", "kernel"],
                "mlx": ["weight"],
                "onnx": ["W"],
            },
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
    "optimizer": {
        "roles": {
            "learning_rate": {
                "torch": ["lr"],
                "optax": ["learning_rate"],
                "jax": ["learning_rate", "step_size"],
                "tensorflow": ["learning_rate"],
                "tf": ["learning_rate"],
                "keras3": ["learning_rate"],
                "mlx": ["learning_rate"],
            },
            "beta1": {
                "torch": ["betas"],
                "optax": ["b1"],
                "jax": ["b1"],
                "tensorflow": ["beta_1"],
                "tf": ["beta_1"],
                "keras3": ["beta_1"],
            },
            "beta2": {
                "torch": ["betas"],
                "optax": ["b2"],
                "jax": ["b2"],
                "tensorflow": ["beta_2"],
                "tf": ["beta_2"],
                "keras3": ["beta_2"],
            },
            "epsilon": {
                "torch": ["eps"],
                "optax": ["eps"],
                "jax": ["eps"],
                "tensorflow": ["epsilon"],
                "tf": ["epsilon"],
                "keras3": ["epsilon"],
            },
            "weight_decay": {
                "torch": ["weight_decay"],
                "optax": ["weight_decay"],
                "jax": ["weight_decay"],
                "tensorflow": ["weight_decay"],
                "tf": ["weight_decay"],
                "keras3": ["weight_decay"],
            },
        }
    },
    "loss": {
        "roles": {
            "predictions": {
                "torch": ["input"],
                "optax": ["logits", "predictions"],
                "jax": ["logits", "predictions"],
                "tensorflow": ["y_pred", "logits"],
                "tf": ["y_pred", "logits"],
                "keras3": ["y_pred"],
            },
            "targets": {
                "torch": ["target"],
                "optax": ["labels", "targets"],
                "jax": ["labels", "targets"],
                "tensorflow": ["y_true", "labels"],
                "tf": ["y_true", "labels"],
                "keras3": ["y_true"],
            },
        }
    },
    "checkpoint": {
        "roles": {
            "path": {
                "torch": ["f"],
                "orbax": ["directory"],
                "orbax.checkpoint": ["directory"],
                "tensorflow": ["filepath", "save_path"],
                "tf": ["filepath", "save_path"],
                "keras3": ["filepath"],
            },
            "state": {
                "torch": ["obj", "state_dict"],
                "orbax": ["item", "items", "args"],
                "orbax.checkpoint": ["item", "items", "args"],
                "tensorflow": ["checkpoint"],
                "tf": ["checkpoint"],
                "keras3": ["weights"],
            },
        }
    },
    "assertion": {
        "roles": {
            "inputs": {
                "torch": ["actual", "input"],
                "chex": ["inputs", "trees"],
                "numpy": ["actual", "x"],
                "tensorflow": ["x"],
                "tf": ["x"],
            },
            "rtol": {
                "torch": ["rtol"],
                "chex": ["rtol"],
                "numpy": ["rtol"],
                "tensorflow": ["rtol"],
                "tf": ["rtol"],
            },
            "atol": {
                "torch": ["atol"],
                "chex": ["atol"],
                "numpy": ["atol"],
                "tensorflow": ["atol"],
                "tf": ["atol"],
            },
        }
    },
    "dataloader": {
        "roles": {
            "dataset": {
                "torch": ["dataset"],
                "grain": ["data_source", "dataset", "parents"],
                "tensorflow": ["dataset"],
                "tf": ["dataset"],
            },
            "batch_size": {
                "torch": ["batch_size"],
                "grain": ["batch_size"],
                "tensorflow": ["batch_size"],
                "tf": ["batch_size"],
            },
            "drop_last": {
                "torch": ["drop_last"],
                "grain": ["drop_remainder"],
                "tensorflow": ["drop_remainder"],
                "tf": ["drop_remainder"],
            },
            "num_workers": {
                "torch": ["num_workers"],
                "grain": ["worker_count"],
                "tensorflow": ["num_parallel_calls"],
                "tf": ["num_parallel_calls"],
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
        translations: (
            TranslationMap
            | Mapping[
                str,
                OperationTranslationSpec | Mapping[str, dict[str, RoleFrameworkMap]],
            ]
            | None
        ) = None,
        concept_map_path: str | None = None,
    ) -> None:
        """Initialize translation engine with parameter translation mapping.

        Args:
            translations: Pre-loaded parameter translations mapping.
            concept_map_path (str | None): Path to concept_map.json file.
        """
        self.translations: TranslationMap = (
            TranslationMap.model_validate(translations) if translations is not None else TranslationMap()
        )
        if not self.translations:
            self._load_default_translations(concept_map_path)

    def _merge_fallback_translations(self) -> None:
        """Merge canonical fallback translations into self.translations without overwriting."""
        for op, op_dict in _CANONICAL_FALLBACK_TRANSLATIONS.items():
            if op not in self.translations:
                self.translations[op] = copy.deepcopy(op_dict)
                continue
            roles = self.translations[op].roles
            for r_name, r_map in op_dict.get("roles", {}).items():
                if r_name not in roles:
                    roles[r_name] = copy.deepcopy(r_map)
                else:
                    for fw, fw_params in r_map.items():
                        if fw not in roles[r_name]:
                            roles[r_name][fw] = list(fw_params)
                        else:
                            for p in fw_params:
                                if p not in roles[r_name][fw]:
                                    roles[r_name][fw].append(p)

    def _canonicalize_role_aliases(self) -> None:
        """Synchronize role aliases (e.g. 'kernel' <-> 'weight') across all operations using isolated deep copies."""
        role_aliases = {"kernel": "weight"}
        for op_data in self.translations.values():
            roles = op_data.roles
            for alias, canonical in role_aliases.items():
                if alias in roles and canonical in roles:
                    for fw, fw_params in roles[alias].items():
                        if fw not in roles[canonical]:
                            roles[canonical][fw] = list(fw_params)
                        else:
                            for p in fw_params:
                                if p not in roles[canonical][fw]:
                                    roles[canonical][fw].append(p)
                    roles[alias] = copy.deepcopy(roles[canonical])
                elif alias in roles:
                    roles[canonical] = copy.deepcopy(roles[alias])
                elif canonical in roles:
                    roles[alias] = copy.deepcopy(roles[canonical])

    def _load_default_translations(self, path: str | None = None) -> None:
        """Load default translations from concept_map.json in snapshot directory or package root.

        Args:
            path (str | None): Optional path override.
        """
        if path is not None:
            candidates = [path]
        else:
            effective_dir = get_effective_snapshot_dir()
            candidates = [
                os.path.join(effective_dir, "concept_map.json"),
                os.path.join(os.path.dirname(effective_dir), "concept_map.json"),
            ]
        for cand in candidates:
            if cand and os.path.isfile(cand):
                try:
                    with open(cand, encoding="utf-8") as f:
                        data = json.load(f)
                    if "_parameter_translations" in data and isinstance(data["_parameter_translations"], dict):
                        self.translations.update(data["_parameter_translations"])
                        break
                except (json.JSONDecodeError, OSError) as exc:
                    _logger.debug("Skipping invalid concept_map candidate %s: %s", cand, exc)
        if path is None:
            self._merge_fallback_translations()
        self._canonicalize_role_aliases()

    def _find_matching_roles(
        self,
        operation: str,
        param_name: str,
        source_framework: str,
    ) -> list[tuple[str, RoleFrameworkMap]]:
        """Return all ``(role_name, fw_mappings)`` pairs matching ``param_name`` in ``source_framework``.

        Args:
            operation (str): Canonical operation name.
            param_name (str): Source parameter name.
            source_framework (str): Source framework identifier.

        Returns:
            list[tuple[str, RoleFrameworkMap]]: Matching role specifications in declaration order.
        """
        roles = self.translations[operation].roles
        return [
            (role_name, fw_mappings)
            for role_name, fw_mappings in roles.items()
            if param_name in fw_mappings.get(source_framework, [])
        ]

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
            raise KeyError(f"No parameter translations defined for operation '{operation}'.")

        matches = self._find_matching_roles(operation, param_name, source_framework)
        if matches:
            role_name, fw_mappings = matches[0]
            target_params = fw_mappings.get(target_framework, [])
            if not target_params:
                raise ValueError(
                    f"Target framework '{target_framework}' does not support role '{role_name}' "
                    f"for operation '{operation}'."
                )
            return str(target_params[0])

        raise KeyError(
            f"Parameter '{param_name}' is not recognized as a valid '{source_framework}' parameter for '{operation}'."
        )

    def _unpack_packed_tuple_attribute(
        self,
        matches: list[tuple[str, RoleFrameworkMap]],
        val: Sequence[AttributeValue],
        target_framework: str,
        result: dict[str, AttributeValue],
    ) -> bool:
        """Unpack a packed tuple attribute (e.g. PyTorch ``betas=(0.9, 0.999)``) into distinct target scalar roles.

        Args:
            matches (list[tuple[str, RoleFrameworkMap]]): Roles sharing this source parameter name.
            val (Sequence[AttributeValue]): Sequence value to unpack.
            target_framework (str): Target framework identifier.
            result (dict[str, AttributeValue]): Output dictionary to populate.

        Returns:
            bool: True if unpacked across distinct target scalar parameters, False otherwise.
        """
        if len(matches) < 2 or len(val) != len(matches):
            return False
        tgt_keys: list[str] = []
        for _, fw_map in matches:
            t_list = fw_map.get(target_framework, [])
            if not t_list:
                return False
            tgt_keys.append(str(t_list[0]))
        if len(set(tgt_keys)) != len(tgt_keys):
            return False
        for tgt_k, elem in zip(tgt_keys, val):
            result[tgt_k] = elem
        return True

    def translate_attributes(
        self,
        operation: str,
        attributes: Mapping[str, AttributeValue],
        source_framework: str,
        target_framework: str,
        allow_passthrough: bool = False,
    ) -> dict[str, AttributeValue]:
        """Translate all attributes/keyword arguments from source framework to target framework.

        Supports 1-to-1 renames, 1-to-many tuple unpacking (e.g. ``betas`` -> ``b1``, ``b2``),
        and many-to-1 tuple packing (e.g. ``b1``, ``b2`` -> ``betas``).

        Args:
            operation (str): Canonical operation name.
            attributes (Mapping[str, AttributeValue]): Mapping of parameter names and values.
            source_framework (str): Source framework identifier.
            target_framework (str): Target framework identifier.
            allow_passthrough (bool): If True, parameters without translation are kept; if False,
                unrecognized parameters raise ValueError (zero-hallucination policy).

        Returns:
            dict[str, AttributeValue]: Translated attributes dictionary.

        Raises:
            ValueError: If unknown parameter is encountered and allow_passthrough is False.
        """
        result: dict[str, AttributeValue] = {}
        packed_buckets: dict[str, list[AttributeValue]] = {}

        for k, v in attributes.items():
            try:
                if operation not in self.translations:
                    raise KeyError(f"No parameter translations defined for operation '{operation}'.")
                matches = self._find_matching_roles(operation, k, source_framework)
                if (
                    len(matches) > 1
                    and isinstance(v, (tuple, list))
                    and self._unpack_packed_tuple_attribute(matches, v, target_framework, result)
                ):
                    continue
                translated_key = self.translate_parameter(
                    operation=operation,
                    param_name=k,
                    source_framework=source_framework,
                    target_framework=target_framework,
                )
                # Detect many-to-1 collision onto a shared packed parameter (e.g. b1 + b2 -> betas)
                target_role_count = sum(
                    1
                    for r_map in self.translations[operation].roles.values()
                    if translated_key in r_map.get(target_framework, []) and len(r_map.get(source_framework, [])) > 0
                )
                if target_role_count > 1 and not isinstance(v, (tuple, list)):
                    packed_buckets.setdefault(translated_key, []).append(v)
                    result[translated_key] = (
                        tuple(packed_buckets[translated_key]) if len(packed_buckets[translated_key]) > 1 else v
                    )
                else:
                    result[translated_key] = v
            except (KeyError, ValueError) as err:
                if allow_passthrough:
                    result[k] = v
                else:
                    raise ValueError(
                        f"Ungrounded parameter '{k}' during translation from "
                        f"'{source_framework}' to '{target_framework}': {err}"
                    ) from err
        return result

    def register_standard_map(
        self,
        operation: str,
        framework: str,
        spec: StandardMap,
    ) -> None:
        """Register a Ghost Protocol StandardMap specification for an operation and framework.

        Args:
            operation (str): Canonical middle-layer operation name.
            framework (str): Target framework identifier (e.g. 'torch', 'jax', 'optax').
            spec (StandardMap): Framework parameter mapping specification.
        """
        if operation not in self.translations:
            self.translations[operation] = {"roles": {}}
        roles = self.translations[operation].roles
        if spec.args is not None:
            for role_name, target_spec in spec.args.items():
                target_param: str | None = None
                if isinstance(target_spec, str):
                    target_param = spec.pack_to_tuple if spec.pack_to_tuple is not None else target_spec
                elif spec.pack_to_tuple is not None and target_spec is not None:
                    target_param = spec.pack_to_tuple
                if target_param is not None:
                    role_entry = roles.setdefault(role_name, {})
                    fw_list = role_entry.setdefault(framework, [])
                    if target_param not in fw_list:
                        fw_list.append(target_param)
        self._canonicalize_role_aliases()

    def apply_standard_map(
        self,
        spec: StandardMap,
        attributes: Mapping[str, AttributeValue],
        allow_passthrough: bool = False,
    ) -> dict[str, AttributeValue]:
        """Transform middle-layer attributes using a Ghost Protocol StandardMap specification.

        Args:
            spec (StandardMap): Target framework mapping rule (renames, tuple packing, injections).
            attributes (Mapping[str, AttributeValue]): Canonical role-keyed attribute values.
            allow_passthrough (bool): Whether unmapped attributes are preserved verbatim.

        Returns:
            dict[str, AttributeValue]: Transformed framework keyword arguments.

        Raises:
            ValueError: If an unmapped attribute is encountered while ``allow_passthrough`` is False.
        """
        result: dict[str, AttributeValue] = {}
        packed_values: list[AttributeValue] = []
        args_map = spec.args or {}

        for key, value in attributes.items():
            if key in args_map:
                mapped = args_map[key]
                if mapped is None:
                    continue
                if spec.pack_to_tuple is not None and isinstance(mapped, str):
                    packed_values.append(value)
                elif isinstance(mapped, str):
                    result[mapped] = value
                else:
                    result[key] = mapped
            elif allow_passthrough:
                result[key] = value
            else:
                raise ValueError(f"Ungrounded attribute '{key}' is not mapped in StandardMap '{spec.api}'.")

        if spec.pack_to_tuple is not None and packed_values:
            result[spec.pack_to_tuple] = tuple(packed_values)
        if spec.inject_args is not None:
            for inj_k, inj_v in spec.inject_args.items():
                result[inj_k] = inj_v
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

    @staticmethod
    def _permute_4d_nested_sequence(
        weights: Sequence[Sequence[Sequence[Sequence[JsonValue]]]],
        perm: Sequence[int],
    ) -> list[list[list[list[JsonValue]]]]:
        """Permute a 4D nested Python sequence according to axis permutation ``perm``.

        Args:
            weights (Sequence[Sequence[Sequence[Sequence[JsonValue]]]]): 4D nested list/tuple.
            perm (Sequence[int]): 4-element permutation indices.

        Returns:
            list[list[list[list[JsonValue]]]]: Permuted 4D nested list.
        """
        dims = [
            len(weights),
            len(weights[0]),
            len(weights[0][0]),
            len(weights[0][0][0]),
        ]
        out_dims = [dims[p] for p in perm]
        out_4d: list[list[list[list[JsonValue]]]] = []
        for i0 in range(out_dims[0]):
            plane1: list[list[list[JsonValue]]] = []
            for i1 in range(out_dims[1]):
                plane2: list[list[JsonValue]] = []
                for i2 in range(out_dims[2]):
                    row: list[JsonValue] = []
                    for i3 in range(out_dims[3]):
                        out_idx = (i0, i1, i2, i3)
                        src_idx = [0, 0, 0, 0]
                        for d_idx, p_axis in enumerate(perm):
                            src_idx[p_axis] = out_idx[d_idx]
                        row.append(weights[src_idx[0]][src_idx[1]][src_idx[2]][src_idx[3]])
                    plane2.append(row)
                plane1.append(plane2)
            out_4d.append(plane1)
        return out_4d

    def permute_conv_weights(
        self,
        weights: TensorLike,
        source_format: str,
        target_format: str,
    ) -> TensorLike | tuple[TensorLike, tuple[int, ...]]:
        """Permute 2D convolution weight tensors across framework memory layouts.

        Permutes between PyTorch format (OIHW / [out_channels, in_channels, H, W])
        and TensorFlow/JAX format (HWIO / [H, W, in_channels, out_channels]) or HWOI.

        Args:
            weights (TensorLike): Weight array, list, or tensor object with shape and transpose attributes.
            source_format (str): Source layout string (e.g. 'OIHW', 'HWIO', 'HWOI').
            target_format (str): Target layout string (e.g. 'OIHW', 'HWIO', 'HWOI').

        Returns:
            TensorLike | tuple[TensorLike, tuple[int, ...]]: Transposed weight array or list with target layout.

        Raises:
            ValueError: If source_format or target_format is invalid or ranks mismatch.
        """
        src = source_format.upper().strip()
        tgt = target_format.upper().strip()
        if len(src) != 4 or len(tgt) != 4:
            raise ValueError(f"Conv weight formats must have rank 4: '{src}', '{tgt}'.")
        if sorted(src) != sorted("OIHW") or sorted(tgt) != sorted("OIHW"):
            raise ValueError(f"Formats must be permutations of 'OIHW': '{src}', '{tgt}'.")
        if src == tgt:
            return weights

        perm = [src.index(ch) for ch in tgt]
        if isinstance(weights, TransposableTensorProtocol):
            return weights.transpose(tuple(perm))
        if isinstance(weights, PermutableTensorProtocol):
            return weights.permute(tuple(perm))
        if _is_4d_nested_sequence(weights):
            return self._permute_4d_nested_sequence(weights, perm)
        return PermutedWeightDescriptor(weights, tuple(perm))

    @staticmethod
    def _split_sequence_qkv(
        seq: Sequence[TensorLike],
        q_size: int,
        kv_size: int,
        split_dim: int,
    ) -> tuple[TensorLike, TensorLike, TensorLike]:
        """Split a 1D or 2D Python list/tuple QKV weight along ``split_dim``.

        Args:
            seq (Sequence[TensorLike]): Input 1D or 2D Python sequence.
            q_size (int): Query projection size (`num_heads * head_dim`).
            kv_size (int): Key/Value projection size (`num_kv_heads * head_dim`).
            split_dim (int): Axis along which Q, K, and V are concatenated (0 or 1 / -1).

        Returns:
            tuple[TensorLike, TensorLike, TensorLike]: ``(q, k, v)`` sequences preserving list/tuple type.

        Raises:
            ValueError: If sequence dimensions do not match ``q_size + 2 * kv_size`` along ``split_dim``.
        """
        total_expected = q_size + 2 * kv_size
        is_tup = isinstance(seq, tuple)
        if split_dim in (1, -1) and seq and all(isinstance(r, (list, tuple)) for r in seq):
            rows: list[Sequence[TensorLike]] = [r for r in seq if isinstance(r, (list, tuple))]
            if any(len(r) != total_expected for r in rows):
                raise ValueError(f"2D QKV sequence inner size must equal {total_expected} along split_dim={split_dim}.")
            if is_tup:
                q_tup: Sequence[Sequence[TensorLike]] = tuple(tuple(r[:q_size]) for r in rows)
                k_tup: Sequence[Sequence[TensorLike]] = tuple(tuple(r[q_size : q_size + kv_size]) for r in rows)
                v_tup: Sequence[Sequence[TensorLike]] = tuple(tuple(r[q_size + kv_size :]) for r in rows)
                return (q_tup, k_tup, v_tup)
            q_lst: Sequence[Sequence[TensorLike]] = [list(r[:q_size]) for r in rows]
            k_lst: Sequence[Sequence[TensorLike]] = [list(r[q_size : q_size + kv_size]) for r in rows]
            v_lst: Sequence[Sequence[TensorLike]] = [list(r[q_size + kv_size :]) for r in rows]
            return (q_lst, k_lst, v_lst)

        if len(seq) != total_expected:
            raise ValueError(
                f"QKV sequence length ({len(seq)}) does not match expected {total_expected} (q={q_size}, kv={kv_size})."
            )
        q_flat: TensorLike = tuple(seq[:q_size]) if is_tup else list(seq[:q_size])
        k_flat: TensorLike = tuple(seq[q_size : q_size + kv_size]) if is_tup else list(seq[q_size : q_size + kv_size])
        v_flat: TensorLike = tuple(seq[q_size + kv_size :]) if is_tup else list(seq[q_size + kv_size :])
        return (q_flat, k_flat, v_flat)

    def split_qkv_weights(
        self,
        qkv_weight: TensorLike,
        num_heads: int,
        head_dim: int,
        split_dim: int = 0,
        num_kv_heads: int | None = None,
    ) -> tuple[
        TensorLike | SplitQKVSliceDescriptor,
        TensorLike | SplitQKVSliceDescriptor,
        TensorLike | SplitQKVSliceDescriptor,
    ]:
        """Split combined QKV projection weights into separate Q, K, and V weights (MHA, GQA, and MQA).

        Args:
            qkv_weight (TensorLike): Combined QKV weight array or sequence.
            num_heads (int): Number of query attention heads.
            head_dim (int): Dimensionality of each attention head.
            split_dim (int): Dimension along which to split (default 0).
            num_kv_heads (int | None): Optional key/value head count for GQA/MQA (defaults to ``num_heads``).

        Returns:
            tuple[
                TensorLike | SplitQKVSliceDescriptor,
                TensorLike | SplitQKVSliceDescriptor,
                TensorLike | SplitQKVSliceDescriptor,
            ]: ``(q_weight, k_weight, v_weight)`` tuple.

        Raises:
            ValueError: If head dimensions are non-positive or concrete sequence length mismatches.
        """
        kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads <= 0 or head_dim <= 0 or kv_heads <= 0:
            raise ValueError(
                f"num_heads ({num_heads}), num_kv_heads ({kv_heads}), and head_dim ({head_dim}) must be positive."
            )

        q_size = num_heads * head_dim
        kv_size = kv_heads * head_dim
        if (
            not isinstance(qkv_weight, (str, bytes))
            and isinstance(qkv_weight, SplittableTensorProtocol)
            and q_size == kv_size
        ):
            parts = qkv_weight.split(q_size, dim=split_dim)
            return (parts[0], parts[1], parts[2])
        if isinstance(qkv_weight, (list, tuple)):
            return self._split_sequence_qkv(qkv_weight, q_size, kv_size, split_dim)
        return (
            SplitQKVSliceDescriptor("q", qkv_weight),
            SplitQKVSliceDescriptor("k", qkv_weight),
            SplitQKVSliceDescriptor("v", qkv_weight),
        )

    def merge_qkv_weights(
        self,
        q_weight: TensorLike,
        k_weight: TensorLike,
        v_weight: TensorLike,
        concat_dim: int = 0,
    ) -> TensorLike | MergedQKVWeightDescriptor:
        """Merge separate Q, K, and V projection weights into a single combined QKV tensor.

        Args:
            q_weight (TensorLike): Query projection weight.
            k_weight (TensorLike): Key projection weight.
            v_weight (TensorLike): Value projection weight.
            concat_dim (int): Dimension along which to concatenate (default 0).

        Returns:
            TensorLike | MergedQKVWeightDescriptor:
                Merged QKV weight object.

        Raises:
            ValueError: If 2D sequences concatenated along dim 1 have mismatched outer row counts.
        """
        if (
            isinstance(q_weight, (list, tuple))
            and isinstance(k_weight, (list, tuple))
            and isinstance(v_weight, (list, tuple))
        ):
            is_tup = isinstance(q_weight, tuple) and isinstance(k_weight, tuple) and isinstance(v_weight, tuple)
            if (
                concat_dim in (1, -1)
                and q_weight
                and all(isinstance(r, (list, tuple)) for r in q_weight)
                and all(isinstance(r, (list, tuple)) for r in k_weight)
                and all(isinstance(r, (list, tuple)) for r in v_weight)
            ):
                if not (len(q_weight) == len(k_weight) == len(v_weight)):
                    raise ValueError("Cannot concatenate 2D Q, K, V sequences along dim 1 with mismatched row counts.")
                rows_q = [r for r in q_weight if isinstance(r, (list, tuple))]
                rows_k = [r for r in k_weight if isinstance(r, (list, tuple))]
                rows_v = [r for r in v_weight if isinstance(r, (list, tuple))]
                merged_rows: list[Sequence[TensorLike]] = [
                    tuple([*rq, *rk, *rv]) if is_tup else [*rq, *rk, *rv] for rq, rk, rv in zip(rows_q, rows_k, rows_v)
                ]
                return tuple(merged_rows) if is_tup else merged_rows
            merged_flat = [*q_weight, *k_weight, *v_weight]
            return tuple(merged_flat) if is_tup else merged_flat
        return MergedQKVWeightDescriptor("qkv", (q_weight, k_weight, v_weight), concat_dim)
