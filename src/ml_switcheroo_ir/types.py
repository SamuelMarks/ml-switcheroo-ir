"""Type definitions for ml_switcheroo_ir."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Sequence,
    Union,
)

if TYPE_CHECKING:
    from ml_switcheroo_ir import NoTangent, ZeroTangent

from ml_switcheroo_ir.shapes import (
    DimensionType,
    SymInt,
    SymNode,
    broadcast_shapes,
    matmul_shape,
)

__all__ = [
    "AttributeValue",
    "DType",
    "DimensionType",
    "NoTangent",
    "TensorShape",
    "TensorSpec",
    "ZeroTangent",
]

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
        # Normalization map for common framework variations
        mapping: dict[str, DType] = {
            "float": cls.float32,
            "double": cls.float64,
            "half": cls.float16,
            "int": cls.int32,
            "long": cls.int64,
            "short": cls.int16,
            "byte": cls.uint8,
            "char": cls.int8,
            "boolean": cls.bool,
            "str": cls.string,
            "obj": cls.object,
            "torch.float32": cls.float32,
            "torch.float16": cls.float16,
            "torch.bfloat16": cls.bfloat16,
            "torch.float64": cls.float64,
            "torch.int64": cls.int64,
            "torch.int32": cls.int32,
            "torch.bool": cls.bool,
        }
        if clean in mapping:
            return mapping[clean]
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
        onnx_map: dict[str, str] = {
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
        return onnx_map.get(self.value, self.value.upper())


@dataclass
class TensorShape:
    """Represents a tensor shape with dimension query and symbolic manipulation capabilities.

    Attributes:
        dims (tuple[DimensionType, ...]): Tensor dimension sizes, symbolic names, or SymNode expressions.
    """

    dims: tuple[DimensionType, ...]

    def __init__(
        self, dims: tuple[DimensionType, ...] | Sequence[DimensionType] | Any
    ) -> None:
        """Initialize TensorShape normalizing inputs into a tuple.

        Args:
            dims (tuple[DimensionType, ...] | Sequence[DimensionType] | Any): Dimensions.
        """
        if isinstance(dims, tuple):
            self.dims = dims
        elif isinstance(dims, (list, Sequence)):
            self.dims = tuple(dims)
        elif hasattr(dims, "dims"):
            self.dims = tuple(dims.dims)
        else:
            try:
                self.dims = tuple(dims)
            except TypeError:
                self.dims = (dims,)

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
        return any(
            isinstance(d, str)
            or (isinstance(d, int) and d < 0)
            or (isinstance(d, (SymNode, SymInt)) and not d.is_constant)
            for d in self.dims
        )

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

        Raises:
            KeyError: If an unbound symbolic variable is encountered.
        """
        res: list[int] = []
        for d in self.dims:
            if isinstance(d, int):
                res.append(d)
            elif isinstance(d, (SymNode, SymInt)):
                res.append(d.evaluate(bindings))
            elif d.isdigit():
                res.append(int(d))
            elif d in bindings:
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

    def __getitem__(self, idx: Any) -> Any:
        """Retrieve dimension size by index or slice.

        Args:
            idx (Any): Index or slice.

        Returns:
            Any: Dimension size or tuple of sizes.
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
        return any(
            isinstance(d, str)
            or (isinstance(d, int) and d < 0)
            or (isinstance(d, (SymNode, SymInt)) and not d.is_constant)
            for d in self.shape
        )

    @property
    def static_shape(self) -> tuple[int, ...]:
        """Return static shape tuple if static, else raises ValueError.

        Returns:
            tuple[int, ...]: Non-negative integer dimension tuple.

        Raises:
            ValueError: If the shape is dynamic.
        """
        if self.is_dynamic:
            raise ValueError(f"Shape {self.shape} has dynamic dimensions.")
        res: list[int] = []
        for d in self.shape:
            if isinstance(d, (SymNode, SymInt)):
                res.append(d.eval({}))
            else:
                res.append(int(d))
        return tuple(res)

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
    """Lazily export ZeroTangent and NoTangent from ml_switcheroo_ir.

    Args:
        name (str): Attribute name to retrieve.

    Returns:
        Any: Exported object.

    Raises:
        AttributeError: If attribute is not found.
    """
    if name in ("ZeroTangent", "NoTangent"):
        import ml_switcheroo_ir

        val = getattr(ml_switcheroo_ir, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
