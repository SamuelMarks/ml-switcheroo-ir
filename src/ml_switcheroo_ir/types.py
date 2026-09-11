"""Type definitions for ml_switcheroo_ir."""

from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence, Union

AttributeValue = Union[
    int,
    float,
    str,
    bool,
    List[int],
    List[float],
    List[str],
    Sequence[Any],
    Dict[str, Any],
    Mapping[str, Any],
    Any,
]


class DType(str, Enum):
    """Data types for tensors."""

    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
    float64 = "float64"
    int64 = "int64"
    int32 = "int32"
    int16 = "int16"
    int8 = "int8"
    uint64 = "uint64"
    uint32 = "uint32"
    uint16 = "uint16"
    uint8 = "uint8"
    bool = "bool"
    complex64 = "complex64"
    complex128 = "complex128"
    float8_e4m3fn = "float8_e4m3fn"
    float8_e5m2 = "float8_e5m2"
    int4 = "int4"
    uint4 = "uint4"
    int2 = "int2"
