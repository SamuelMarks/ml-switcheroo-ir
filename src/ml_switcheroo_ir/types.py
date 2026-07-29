"""Type definitions for ml_switcheroo_ir."""

from enum import Enum
from typing import Any, List, Union

AttributeValue = Union[int, float, str, List[int], List[float], List[str], Any]


class DType(str, Enum):
    """Data types for tensors."""

    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
    int64 = "int64"
    int32 = "int32"
    bool = "bool"
