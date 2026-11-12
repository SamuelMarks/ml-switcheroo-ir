"""Core validation types, levels, errors, and Levenshtein edit distance."""

from __future__ import annotations

from enum import Enum

import pydantic

from ml_switcheroo_ir.schema.base import REGISTRY_LOCK, LazyMetadataMap
from ml_switcheroo_ir.schema.mlir_registry import build_core_mlir_dialects

__all__ = [
    "CORE_MLIR_DIALECTS",
    "KNOWN_SYNONYMS",
    "ValidationError",
    "ValidationLevel",
    "compute_levenshtein",
]


def compute_levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.

    Returns:
        int: Integer edit distance between s1 and s2.
    """
    if len(s1) < len(s2):
        return compute_levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


KNOWN_SYNONYMS: dict[tuple[str, str], str] = {
    ("stablehlo", "matmul"): "stablehlo.dot_general",
    ("stablehlo", "linear"): "stablehlo.dot_general",
    ("stablehlo", "conv2d"): "stablehlo.convolution",
    ("arith", "mul"): "arith.mulf",
    ("arith", "add"): "arith.addf",
    ("math", "exp2"): "math.exp",
    ("torch", "rms_norm"): "torch.nn.RMSNorm",
    ("torch", "layer_norm"): "torch.nn.LayerNorm",
    ("torch", "group_norm"): "torch.nn.GroupNorm",
    ("torch", "batch_norm"): "torch.nn.BatchNorm2d",
    ("torch", "gelu"): "torch.nn.GELU",
    ("torch", "relu"): "torch.nn.ReLU",
    ("torch", "silu"): "torch.nn.SiLU",
    ("torch", "all_reduce"): "torch.distributed.all_reduce",
    ("torch", "all_gather"): "torch.distributed.all_gather",
    ("torch", "reduce_scatter"): "torch.distributed.reduce_scatter",
    ("jax", "rms_norm"): "jax.nn.standardize",
    ("jax", "gelu"): "jax.nn.gelu",
    ("jax", "relu"): "jax.nn.relu",
    ("jax", "silu"): "jax.nn.silu",
    ("jax", "softmax"): "jax.nn.softmax",
    ("tf", "rms_norm"): "tf.keras.layers.RMSNormalization",
    ("tf", "layer_norm"): "tf.keras.layers.LayerNormalization",
    ("tf", "gelu"): "tf.nn.gelu",
    ("tf", "relu"): "tf.nn.relu",
    ("tf", "silu"): "tf.nn.silu",
    ("optax", "cross_entropy"): "optax.losses.softmax_cross_entropy",
    ("optax", "adam_w"): "optax.adamw",
    ("orbax", "checkpointer"): "orbax.checkpoint.PyTreeCheckpointer",
    ("chex", "assert_allclose"): "chex.assert_trees_all_close",
    ("grain", "DataLoader"): "grain.DataLoader",
}


def _load_core_mlir_dialects() -> None:
    """Populate ``CORE_MLIR_DIALECTS`` lazily from the unified ``MLIR_REGISTRY``."""
    with REGISTRY_LOCK:
        CORE_MLIR_DIALECTS.mark_loaded()
        for dialect, ops in build_core_mlir_dialects().items():
            CORE_MLIR_DIALECTS[dialect] = ops


CORE_MLIR_DIALECTS: LazyMetadataMap[dict[str, int]] = LazyMetadataMap(
    "CORE_MLIR_DIALECTS", loader=_load_core_mlir_dialects
)


class ValidationLevel(Enum):
    """Severity levels for validation errors and validator configuration.

    Attributes:
        STRICT (ValidationLevel): Treat all errors and warnings as fatal errors.
        WARNING (ValidationLevel): Emit non-fatal warning diagnostics.
        LENIENT (ValidationLevel): Suppress warning-level diagnostics.
        ERROR (ValidationLevel): Standard fatal validation error level.
    """

    STRICT = "STRICT"
    WARNING = "WARNING"
    LENIENT = "LENIENT"
    ERROR = "ERROR"


class _ValidationErrorPayload(pydantic.BaseModel):
    """Pydantic schema validating ValidationError fields.

    Attributes:
        node_id (str): Node identifier where the error occurred.
        attribute (str): Attribute or structural slot name.
        message (str): Detailed diagnostic description.
        level (ValidationLevel): Diagnostic severity level.
        suggested_fix (str | None): Optional suggested correction string.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    node_id: str
    attribute: str
    message: str
    level: ValidationLevel = ValidationLevel.ERROR
    suggested_fix: str | None = None


class ValidationError(Exception):
    """Represents an error found during graph or node validation.

    Attributes:
        node_id (str): The ID of the node where the error occurred.
        attribute (str): The name of the attribute involved, or a general descriptor.
        message (str): The detailed error message.
        level (ValidationLevel): The severity of the error.
        suggested_fix (str | None): Suggested correction or candidate symbol.
    """

    node_id: str
    attribute: str
    message: str
    level: ValidationLevel
    suggested_fix: str | None

    def __init__(
        self,
        node_id: str,
        attribute: str,
        message: str,
        level: ValidationLevel = ValidationLevel.ERROR,
        suggested_fix: str | None = None,
    ) -> None:
        """Initialize ValidationError with Pydantic field validation.

        Args:
            node_id (str): The ID of the node where the error occurred.
            attribute (str): The name of the attribute involved.
            message (str): The detailed error message.
            level (ValidationLevel): The severity of the error.
            suggested_fix (str | None): Suggested correction or candidate symbol.
        """
        payload = _ValidationErrorPayload(
            node_id=node_id,
            attribute=attribute,
            message=message,
            level=level,
            suggested_fix=suggested_fix,
        )
        super().__init__(payload.message)
        self.node_id = payload.node_id
        self.attribute = payload.attribute
        self.message = payload.message
        self.level = payload.level
        self.suggested_fix = payload.suggested_fix

    def __str__(self) -> str:
        """Return formatted string description of the validation error.

        Returns:
            str: Human-readable error message with severity level.
        """
        return f"[{self.level.value}] Node '{self.node_id}' attribute '{self.attribute}': {self.message}"
