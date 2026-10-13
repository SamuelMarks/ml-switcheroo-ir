"""Core validation types, levels, errors, and Levenshtein edit distance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
}

CORE_MLIR_DIALECTS: dict[str, dict[str, int]] = {
    "arith": {
        "addf": 2,
        "subf": 2,
        "mulf": 2,
        "divf": 2,
        "addi": 2,
        "subi": 2,
        "muli": 2,
        "constant": 0,
        "cmpf": 2,
        "cmpi": 2,
    },
    "math": {
        "exp": 1,
        "exp2": 1,
        "log": 1,
        "sin": 1,
        "cos": 1,
        "tanh": 1,
        "sqrt": 1,
        "rsqrt": 1,
        "erf": 1,
        "absf": 1,
    },
    "tensor": {
        "extract": 2,
        "insert": 3,
        "empty": 0,
        "cast": 1,
        "dim": 2,
        "expand_shape": 1,
        "collapse_shape": 1,
    },
    "linalg": {
        "generic": 2,
        "matmul": 3,
        "conv2d": 3,
        "fill": 2,
        "batch_matmul": 3,
        "dot": 3,
    },
    "scf": {
        "for": 3,
        "if": 1,
        "while": 1,
        "yield": 1,
        "condition": 1,
    },
    "func": {
        "func": 0,
        "call": 1,
        "return": 1,
    },
}


class ValidationLevel(Enum):
    """Severity levels for validation errors and validator configuration."""

    STRICT = "STRICT"
    WARNING = "WARNING"
    LENIENT = "LENIENT"
    ERROR = "ERROR"


@dataclass
class ValidationError(Exception):
    """Represents an error found during graph or node validation.

    Attributes:
        node_id (str): The ID of the node where the error occurred.
        attribute (str): The name of the attribute involved, or a general descriptor.
        message (str): The detailed error message.
        level (ValidationLevel): The severity of the error.
        suggested_fix (Optional[str]): Suggested correction or candidate symbol.
    """

    node_id: str
    attribute: str
    message: str
    level: ValidationLevel = ValidationLevel.ERROR
    suggested_fix: str | None = None

    def __str__(self) -> str:
        """Return formatted string description of the validation error.

        Returns:
            str: Human-readable error message with severity level.
        """
        return f"[{self.level.value}] Node '{self.node_id}' attribute '{self.attribute}': {self.message}"
