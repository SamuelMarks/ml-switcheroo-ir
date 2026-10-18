"""Type definitions for ml_switcheroo_ir."""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Final,
    List,
    Protocol,
    Tuple,
    Union,
    overload,
)
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from ml_switcheroo_ir.sentinels import NoTangent, ZeroTangent

from ml_switcheroo_ir.shapes import (
    DimensionType,
    SymInt,
    SymNode,
    broadcast_shapes,
    matmul_shape,
)

__all__ = [
    "AttributeValue",
    "AutodiffSentinelProtocol",
    "DType",
    "DimensionType",
    "InitializerValue",
    "NoTangent",
    "ShapeMetadataType",
    "TensorShape",
    "TensorSpec",
    "ZeroTangent",
    "is_dynamic_dimension",
]

if TYPE_CHECKING:
    InitializerValue: TypeAlias = Union[
        int,
        float,
        bool,
        str,
        bytes,
        List["InitializerValue"],
        Dict[str, "InitializerValue"],
    ]

    AttributeValue: TypeAlias = Union[
        int,
        float,
        str,
        bool,
        bytes,
        None,
        SymNode,
        SymInt,
        "DType",
        "TensorShape",
        "TensorSpec",
        Sequence["AttributeValue"],
        Mapping[str, "AttributeValue"],
    ]
else:
    _ScalarAttribute: TypeAlias = Union[
        int,
        float,
        str,
        bool,
        bytes,
        None,
        SymNode,
        SymInt,
        "DType",
        "TensorShape",
        "TensorSpec",
    ]
    InitializerValue: TypeAlias = Union[
        int,
        float,
        bool,
        str,
        bytes,
        List[object],
        Dict[str, object],
    ]
    AttributeValue: TypeAlias = Union[
        _ScalarAttribute,
        Sequence[object],
        Mapping[str, object],
    ]


def is_dynamic_dimension(dim: object) -> bool:
    """Check whether a single dimension value is symbolic or dynamic.

    Args:
        dim (object): Dimension entry to inspect.

    Returns:
        bool: True if the dimension is dynamic, symbolic, negative, or non-integer.
    """
    if isinstance(dim, int) and not isinstance(dim, bool):
        return dim < 0
    if isinstance(dim, (SymNode, SymInt)):
        return not dim.is_constant or dim.eval({}) < 0
    return True


class AutodiffSentinelProtocol(Protocol):
    """Protocol representing autodiff sentinel nodes (ZeroTangent, NoTangent).

    Attributes:
        id (str): Unique node identifier.
        op_type (str): Sentinel operation type ('ZeroTangent' or 'NoTangent').
        domain (str): Operator domain ('ml.switcheroo.ad').
    """

    id: str
    op_type: str
    domain: str


_DTYPE_ALIASES: Final[Mapping[str, str]] = {
    "float": "float32",
    "double": "float64",
    "half": "float16",
    "int": "int32",
    "long": "int64",
    "short": "int16",
    "byte": "uint8",
    "char": "int8",
    "boolean": "bool",
    "str": "string",
    "obj": "object",
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.float64": "float64",
    "torch.int64": "int64",
    "torch.int32": "int32",
    "torch.bool": "bool",
}

_DTYPE_ONNX_MAP: Final[Mapping[str, str]] = {
    "float32": "FLOAT",
    "float16": "FLOAT16",
    "bfloat16": "BFLOAT16",
    "float64": "DOUBLE",
    "int64": "INT64",
    "int32": "INT32",
    "int16": "INT16",
    "int8": "INT8",
    "uint64": "UINT64",
    "uint32": "UINT32",
    "uint16": "UINT16",
    "uint8": "UINT8",
    "bool": "BOOL",
    "string": "STRING",
    "complex64": "COMPLEX64",
    "complex128": "COMPLEX128",
    "float8_e4m3fn": "FLOAT8E4M3FN",
    "float8_e4m3b11fnuz": "FLOAT8E4M3FNUZ",
    "float8_e5m2": "FLOAT8E5M2",
    "fp8_e4m3fn": "FLOAT8E4M3FN",
    "fp8_e4m3fnuz": "FLOAT8E4M3FNUZ",
    "fp8_e5m2": "FLOAT8E5M2",
    "fp8_e5m2fnuz": "FLOAT8E5M2FNUZ",
    "int4": "INT4",
    "uint4": "UINT4",
}

_DTYPE_ITEMSIZE_MAP: Final[Mapping[str, float]] = {
    "complex128": 16.0,
    "float64": 8.0,
    "int64": 8.0,
    "uint64": 8.0,
    "complex64": 8.0,
    "float32": 4.0,
    "int32": 4.0,
    "uint32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int16": 2.0,
    "uint16": 2.0,
    "int8": 1.0,
    "uint8": 1.0,
    "bool": 1.0,
    "qint8": 1.0,
    "quint8": 1.0,
    "float8_e4m3fn": 1.0,
    "float8_e4m3b11fnuz": 1.0,
    "float8_e5m2": 1.0,
    "fp8_e4m3fn": 1.0,
    "fp8_e4m3fnuz": 1.0,
    "fp8_e5m2": 1.0,
    "fp8_e5m2fnuz": 1.0,
    "int4": 0.5,
    "uint4": 0.5,
    "qint4": 0.5,
    "int2": 0.25,
    "string": 4.0,
    "object": 4.0,
}


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
    string = "string"
    object = "object"
    complex64 = "complex64"
    complex128 = "complex128"
    float8_e4m3fn = "float8_e4m3fn"
    float8_e4m3b11fnuz = "float8_e4m3b11fnuz"
    float8_e5m2 = "float8_e5m2"
    fp8_e4m3fn = "fp8_e4m3fn"
    fp8_e4m3fnuz = "fp8_e4m3fnuz"
    fp8_e5m2 = "fp8_e5m2"
    fp8_e5m2fnuz = "fp8_e5m2fnuz"
    int4 = "int4"
    uint4 = "uint4"
    int2 = "int2"
    qint8 = "qint8"
    quint8 = "quint8"
    qint4 = "qint4"

    @classmethod
    def from_str(cls, val: str) -> DType:
        """Parse string or framework-specific name into a canonical DType.

        Args:
            val (str): String representation of the data type (case-insensitive).

        Returns:
            DType: The corresponding canonical DType enum member.

        Raises:
            ValueError: If the string cannot be resolved to a known DType.
        """
        clean = val.strip().lower()
        alias_target = _DTYPE_ALIASES.get(clean)
        if alias_target is not None:
            return cls(alias_target)
        for member in cls:
            if member.value == clean or member.name.lower() == clean:
                return member
        raise ValueError(f"Unknown or unsupported DType representation: {val!r}")

    def to_torch_str(self) -> str:
        """Return PyTorch-compatible string name.

        Returns:
            str: PyTorch type representation (e.g. 'torch.float32').
        """
        return f"torch.{self.value}"

    def to_onnx_type(self) -> str:
        """Return ONNX-compatible type string name.

        Returns:
            str: ONNX TensorProto type string identifier.
        """
        return _DTYPE_ONNX_MAP.get(self.value, self.value.upper())

    @property
    def is_variable_width(self) -> builtins.bool:
        """Return True if this DType represents an opaque or variable-width descriptor type.

        Returns:
            bool: True for string and object DTypes, False for fixed-width numeric types.
        """
        return self in (DType.string, DType.object)

    @property
    def itemsize(self) -> float:
        """Return the storage size in bytes per element for this data type.

        Sub-byte quantized types (e.g. int4, uint4, qint4) return fractional byte sizes
        (e.g. 0.5 bytes for 4-bit, 0.25 bytes for 2-bit). Variable-width descriptor types
        (string, object) return their nominal 4.0-byte reference slot size.

        Returns:
            float: Number of bytes per element.
        """
        return _DTYPE_ITEMSIZE_MAP[self.value]


class _HasDimsProtocol(Protocol):
    """Protocol for duck-typed shape objects carrying a .dims sequence."""

    @property
    def dims(self) -> Sequence[DimensionType]:
        """Return sequence of dimensions.

        Returns:
            Sequence[DimensionType]: Dimension sequence.
        """


@dataclass
class TensorShape:
    """Represents a tensor shape with dimension query and symbolic manipulation capabilities.

    Attributes:
        dims (tuple[DimensionType, ...]): Tensor dimension sizes, symbolic names, or SymNode expressions.
    """

    dims: tuple[DimensionType, ...]

    def __init__(
        self,
        dims: tuple[DimensionType, ...]
        | Sequence[DimensionType]
        | TensorShape
        | _HasDimsProtocol
        | DimensionType,
    ) -> None:
        """Initialize TensorShape normalizing inputs into a tuple.

        Args:
            dims (tuple[DimensionType, ...] | Sequence[DimensionType] | TensorShape | _HasDimsProtocol | DimensionType): Dimensions.

        Raises:
            TypeError: If any dimension is a boolean value.
        """
        if isinstance(dims, tuple):
            raw_dims = dims
        elif hasattr(dims, "dims"):
            raw_dims = tuple(dims.dims)
        elif not isinstance(dims, str) and isinstance(dims, Iterable):
            raw_dims = tuple(dims)
        else:
            raw_dims = (dims,)

        for d in raw_dims:
            if isinstance(d, bool):
                raise TypeError(
                    f"Boolean value {d!r} is not a valid tensor dimension in TensorShape."
                )
        self.dims = raw_dims

    @property
    def rank(self) -> int:
        """Return tensor rank (number of dimensions).

        Returns:
            int: Number of dimensions.
        """
        return len(self.dims)

    @property
    def is_dynamic(self) -> bool:
        """Check if shape contains dynamic or symbolic dimensions.

        Returns:
            bool: True if dynamic or ungrounded symbolic dimensions are present.
        """
        return any(is_dynamic_dimension(d) for d in self.dims)

    @property
    def static_shape(self) -> tuple[int, ...]:
        """Return static shape or raise ValueError if dynamic.

        Returns:
            tuple[int, ...]: Tuple of non-negative integer dimension sizes.

        Raises:
            ValueError: If the shape contains dynamic or symbolic dimensions.
        """
        if self.is_dynamic:
            raise ValueError(
                f"Shape {self.dims} contains dynamic or symbolic dimensions."
            )
        res: list[int] = []
        for d in self.dims:
            if isinstance(d, (SymNode, SymInt)):
                res.append(d.eval({}))
            else:
                res.append(int(d))
        return tuple(res)

    def evaluate(self, bindings: dict[str, int]) -> tuple[int, ...]:
        """Evaluate all symbolic dimensions into concrete integers using runtime bindings.

        Args:
            bindings (dict[str, int]): Mapping from dimension variable names to concrete integers.

        Returns:
            tuple[int, ...]: Tuple of concrete integer dimension sizes.
        """
        res: list[int] = []
        for d in self.dims:
            if isinstance(d, int) and not isinstance(d, bool):
                res.append(d)
            elif isinstance(d, (SymNode, SymInt)):
                res.append(d.evaluate(bindings))
            elif isinstance(d, str) and d.isdigit():
                res.append(int(d))
            elif isinstance(d, str) and d in bindings:
                res.append(bindings[d])
            else:
                res.append(SymNode.to_node(d).evaluate(bindings))
        return tuple(res)

    def broadcast_with(
        self, other: TensorShape | Sequence[DimensionType]
    ) -> TensorShape:
        """Broadcast this shape with another shape using NumPy-compliant broadcasting.

        Args:
            other (TensorShape | Sequence[DimensionType]): Target shape to broadcast with.

        Returns:
            TensorShape: Resulting broadcasted TensorShape.
        """
        other_dims = other.dims if isinstance(other, TensorShape) else tuple(other)
        return TensorShape(broadcast_shapes(self.dims, other_dims))

    def matmul_with(self, other: TensorShape | Sequence[DimensionType]) -> TensorShape:
        """Infer the resulting matrix multiplication shape with another shape.

        Args:
            other (TensorShape | Sequence[DimensionType]): Target shape to matrix multiply with.

        Returns:
            TensorShape: Resulting matrix multiplication TensorShape.
        """
        other_dims = other.dims if isinstance(other, TensorShape) else tuple(other)
        return TensorShape(matmul_shape(self.dims, other_dims))

    def __len__(self) -> int:
        """Return number of dimensions.

        Returns:
            int: Number of dimensions.
        """
        return len(self.dims)

    def __iter__(self) -> Iterator[DimensionType]:
        """Iterate over dimension sizes.

        Returns:
            Iterator[DimensionType]: Iterator over dimensions.
        """
        return iter(self.dims)

    @overload
    def __getitem__(self, idx: int) -> DimensionType:
        """Retrieve a single dimension by integer index.

        Args:
            idx (int): Zero-based dimension index.

        Returns:
            DimensionType: Dimension at index.
        """

    @overload
    def __getitem__(self, idx: slice) -> tuple[DimensionType, ...]:
        """Retrieve a slice of dimensions.

        Args:
            idx (slice): Slice object.

        Returns:
            tuple[DimensionType, ...]: Sliced tuple of dimensions.
        """

    def __getitem__(
        self, idx: int | slice
    ) -> DimensionType | tuple[DimensionType, ...]:
        """Retrieve dimension size by index or slice.

        Args:
            idx (int | slice): Index or slice.

        Returns:
            DimensionType | tuple[DimensionType, ...]: Dimension size or tuple of sizes.
        """
        return self.dims[idx]

    def __eq__(self, other: object) -> bool:
        """Check equality against TensorShape, tuple, or list.

        Args:
            other (object): Other shape object.

        Returns:
            bool: True if dimensions match.
        """
        if isinstance(other, TensorShape):
            return self.dims == other.dims
        if isinstance(other, (tuple, list)):
            return self.dims == tuple(other)
        return False


ShapeMetadataType: TypeAlias = Union[
    Tuple[DimensionType, ...],
    TensorShape,
    None,
]


@dataclass
class TensorSpec:
    """Specification for tensor values including shape, dtype, and sparsity.

    Attributes:
        shape (tuple[DimensionType, ...]): Dimensions of the tensor.
        dtype (DType): Tensor data type.
        sparsity (str | None): Sparsity layout or format if applicable.
    """

    shape: tuple[DimensionType, ...]
    dtype: DType
    sparsity: str | None = None

    def __init__(
        self,
        shape: tuple[DimensionType, ...] | Sequence[DimensionType] | TensorShape,
        dtype: DType | str,
        sparsity: str | None = None,
    ) -> None:
        """Initialize TensorSpec with shape and dtype normalization.

        Args:
            shape (tuple[DimensionType, ...] | Sequence[DimensionType] | TensorShape): Tensor dimensions.
            dtype (DType | str): Data type as DType enum or string name.
            sparsity (str | None): Optional sparsity layout.
        """
        if isinstance(shape, TensorShape):
            self.shape = shape.dims
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            self.shape = tuple(shape)

        if isinstance(dtype, DType):
            self.dtype = dtype
        else:
            self.dtype = DType.from_str(dtype)

        self.sparsity = sparsity

    @property
    def is_dynamic(self) -> bool:
        """Check if shape has dynamic dimensions.

        Returns:
            bool: True if dynamic dimensions are present.
        """
        return any(is_dynamic_dimension(d) for d in self.shape)

    @property
    def static_shape(self) -> tuple[int, ...]:
        """Return static shape tuple if static, else raises ValueError.

        Returns:
            tuple[int, ...]: Non-negative integer dimension tuple.

        Raises:
            ValueError: If the shape is dynamic.
        """
        if self.is_dynamic:
            raise ValueError(f"TensorSpec shape {self.shape} has dynamic dimensions.")
        return TensorShape(self.shape).static_shape

    @property
    def rank(self) -> int:
        """Return tensor rank.

        Returns:
            int: Tensor rank.
        """
        return len(self.shape)

    def evaluate(self, bindings: dict[str, int]) -> TensorSpec:
        """Evaluate symbolic dimensions into concrete integers using runtime bindings.

        Args:
            bindings (dict[str, int]): Variable bindings mapping names to integer sizes.

        Returns:
            TensorSpec: New TensorSpec instance with concrete evaluated shape.
        """
        shape_obj = TensorShape(self.shape)
        concrete_shape = shape_obj.evaluate(bindings)
        return TensorSpec(
            shape=concrete_shape,
            dtype=self.dtype,
            sparsity=self.sparsity,
        )


def __getattr__(name: str) -> Any:
    """Lazily resolve ZeroTangent and NoTangent from ml_switcheroo_ir.sentinels.

    Args:
        name (str): Attribute name to retrieve.

    Returns:
        Any: Exported sentinel class.

    Raises:
        AttributeError: If attribute is not found.
    """
    if name in ("ZeroTangent", "NoTangent"):
        from ml_switcheroo_ir import sentinels

        return getattr(sentinels, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
