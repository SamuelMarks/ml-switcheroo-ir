"""Type definitions for ml_switcheroo_ir."""

from __future__ import annotations

import base64
import builtins
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import pydantic
import pydantic_core
import typing_extensions
from pydantic_core import core_schema

from ml_switcheroo_ir.protocols import (
    ArrayBufferProtocol,
    CustomRuleHandleProtocol,
    DeviceLikeProtocol,
    DTypeLikeProtocol,
    LoopOptionsProtocol,
    MeshLikeProtocol,
    SerializationInfoProtocol,
    SubgraphLikeProtocol,
    SymbolicDimProtocol,
    SymbolicIntWrapperProtocol,
    SymbolicNodeProtocol,
)
from ml_switcheroo_ir.shapes import (
    DimensionType,
    DimensionValue,
    SymInt,
    SymNode,
    broadcast_shapes,
    format_symbolic_protocol_expr,
    matmul_shape,
)

__all__ = [
    "ArrayBufferProtocol",
    "AttrBool",
    "AttrBytes",
    "AttrCompositeTerm",
    "AttrDType",
    "AttrDomainTerm",
    "AttrFloat",
    "AttrInt",
    "AttrMap",
    "AttrModel",
    "AttrNone",
    "AttrProtocolLeaf",
    "AttrScalarTerm",
    "AttrSeq",
    "AttrShape",
    "AttrStr",
    "AttrSymInt",
    "AttrSymNode",
    "AttrTensorSpec",
    "AttrTerm",
    "AttributeValue",
    "AutodiffSentinelProtocol",
    "CustomRuleHandleProtocol",
    "DType",
    "DTypeLikeProtocol",
    "DeviceLikeProtocol",
    "DimensionType",
    "DimensionValue",
    "InitializerValue",
    "JsonValue",
    "LoopOptionsProtocol",
    "MeshLikeProtocol",
    "ProtocolAttributeLeaf",
    "ShapeMetadataType",
    "SubgraphLikeProtocol",
    "SymbolicDimProtocol",
    "SymbolicIntWrapperProtocol",
    "SymbolicNodeProtocol",
    "TensorShape",
    "TensorSpec",
    "_validate_dimension_tuple",
    "describe_value_type",
    "embed_attribute_value",
    "is_dynamic_dimension",
    "is_symbolic_dimension",
    "normalize_dimension_tuple",
    "normalize_dtype",
    "normalize_optional_dtype",
    "project_attribute_value",
]

JsonScalar: typing_extensions.TypeAlias = typing.Union[str, int, float, bool, None]

JsonValue: typing_extensions.TypeAlias = pydantic.JsonValue


def _is_valid_array_buffer(val: ArrayBufferProtocol) -> bool:
    """Validate structural invariants of an ``ArrayBufferProtocol`` instance.

    Args:
        val (ArrayBufferProtocol): Candidate array buffer protocol object.

    Returns:
        bool: True if ``shape`` contains no boolean dimensions and ``tobytes()`` succeeds.
    """
    _ = val.tobytes()
    return all(not isinstance(dim, bool) for dim in val.shape)


def _validate_initializer_value(val: InitializerValue) -> InitializerValue:
    """Recursively validate a runtime ``InitializerValue`` tree for Pydantic fields.

    Args:
        val (InitializerValue): Candidate initializer payload.

    Returns:
        InitializerValue: Validated initializer payload.

    Raises:
        ValueError: If ``val`` or any nested element is not a valid ``InitializerValue``.
    """
    if isinstance(val, (int, float, bool, str, bytes)):
        return val
    if isinstance(val, ArrayBufferProtocol):
        if _is_valid_array_buffer(val):
            return val
        raise ValueError(f"Invalid ArrayBufferProtocol payload in InitializerValue: {val!r}")
    if isinstance(val, Sequence):
        for item in val:
            _validate_initializer_value(item)
        return val
    if isinstance(val, Mapping):
        for key, item in val.items():
            if not isinstance(key, str):
                raise ValueError(f"Initializer dictionary key must be str, got {describe_value_type(key)}.")
            _validate_initializer_value(item)
        return val
    raise ValueError(f"Unsupported InitializerValue of type {describe_value_type(val)}: {val!r}")


def _encode_json_safe_initializer(val: InitializerValue) -> JsonValue:
    """Convert an ``InitializerValue`` into a deterministic JSON-serializable structure.

    Args:
        val (InitializerValue): Validated initializer payload.

    Returns:
        JsonValue: Deterministic JSON-compatible payload.
    """
    if isinstance(val, (int, float, bool, str)):
        return val
    if isinstance(val, bytes):
        return {"__sw_type__": "bytes", "data": base64.b64encode(val).decode("ascii")}
    if isinstance(val, ArrayBufferProtocol):
        return {
            "__sw_type__": "ArrayBuffer",
            "data": base64.b64encode(val.tobytes()).decode("ascii"),
            "shape": list(val.shape),
        }
    if isinstance(val, Mapping):
        return {str(k): _encode_json_safe_initializer(v) for k, v in val.items()}
    return [_encode_json_safe_initializer(item) for item in val]


def _serialize_initializer_pydantic(
    val: InitializerValue,
    info: SerializationInfoProtocol,
) -> InitializerValue | JsonValue:
    """Serialize ``InitializerValue`` for Pydantic Python or JSON mode.

    Args:
        val (InitializerValue): Validated initializer payload.
        info (SerializationInfoProtocol): Pydantic serialization context.

    Returns:
        InitializerValue | JsonValue: Raw value in Python mode, or JSON-safe payload in JSON mode.
    """
    if info.mode == "json":
        return _encode_json_safe_initializer(val)
    return val


class _InitializerValuePydanticAnnotation:
    """Recursive Pydantic v2 validator and JSON Schema provider for ``InitializerValue``."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: type[InitializerValue],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.CoreSchema:
        """Build recursive runtime validation CoreSchema for ``InitializerValue``.

        Args:
            source_type (type[InitializerValue]): Annotated alias type.
            handler (pydantic.GetCoreSchemaHandler): Pydantic core schema handler.

        Returns:
            pydantic_core.CoreSchema: Validating CoreSchema instance.
        """
        del source_type, handler
        return core_schema.no_info_plain_validator_function(
            _validate_initializer_value,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_initializer_pydantic,
                info_arg=True,
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema_obj: pydantic_core.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> dict[str, JsonValue]:
        """Generate Draft 2020-12 JSON Schema for ``InitializerValue``.

        Args:
            core_schema_obj (pydantic_core.CoreSchema): Core schema object.
            handler (pydantic.GetJsonSchemaHandler): JSON schema handler.

        Returns:
            dict[str, JsonValue]: Constrained union JSON Schema definition.
        """
        del core_schema_obj, handler
        return {
            "anyOf": [
                {"type": "integer"},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "string"},
                {"type": "array"},
                {"type": "object"},
            ]
        }


InitializerValue: typing_extensions.TypeAlias = typing.Annotated[
    typing.Union[
        int,
        float,
        bool,
        str,
        bytes,
        ArrayBufferProtocol,
        Sequence["InitializerValue"],
        Mapping[str, "InitializerValue"],
    ],
    _InitializerValuePydanticAnnotation,
]


def is_symbolic_dimension(
    dim: DimensionType | JsonValue,
) -> typing.TypeGuard[str | SymNode | SymInt | SymbolicNodeProtocol | SymbolicIntWrapperProtocol]:
    """Narrow a dimension entry strictly to symbolic identifier/AST/protocol types (never negative ``int``).

    Args:
        dim (DimensionType | JsonValue): Dimension entry to inspect.

    Returns:
        typing.TypeGuard[str | SymNode | SymInt | SymbolicNodeProtocol | SymbolicIntWrapperProtocol]:
            True iff ``dim`` is a symbolic ``str``, ``SymNode``, ``SymInt``, or symbolic protocol.

    Raises:
        TypeError: If ``dim`` is a boolean value.
    """
    if isinstance(dim, bool):
        raise TypeError(f"Boolean value {dim!r} is not a valid tensor dimension.")
    if isinstance(dim, str):
        return not dim.isdigit()
    if isinstance(dim, (SymNode, SymInt)):
        return True
    if isinstance(dim, SymbolicIntWrapperProtocol):
        return isinstance(dim.node, SymbolicNodeProtocol)
    return isinstance(dim, SymbolicNodeProtocol)


def is_dynamic_dimension(
    dim: DimensionType | JsonValue,
) -> bool:
    """Check whether a single dimension value is dynamic, negative, or ungrounded symbolic.

    Args:
        dim (DimensionType | JsonValue): Dimension entry to inspect.

    Returns:
        bool: True if the dimension is dynamic, symbolic, negative, or non-static.

    Raises:
        TypeError: If ``dim`` is a boolean value.
    """
    if isinstance(dim, bool):
        raise TypeError(f"Boolean value {dim!r} is not a valid tensor dimension.")
    if isinstance(dim, int):
        return dim < 0
    if isinstance(dim, str):
        return not dim.isdigit()
    if isinstance(dim, (SymNode, SymInt)):
        return not dim.is_constant or dim.eval({}) < 0
    return True


class AutodiffSentinelProtocol(typing.Protocol):
    """Protocol representing autodiff sentinel nodes (ZeroTangent, NoTangent).

    Attributes:
        id (str): Unique node identifier.
        op_type (str): Sentinel operation type ('ZeroTangent' or 'NoTangent').
        domain (str): Operator domain ('ml.switcheroo.ad').
    """

    id: str
    op_type: str
    domain: str


_DTYPE_ALIASES: typing.Final[Mapping[str, str]] = {
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

_DTYPE_ONNX_MAP: typing.Final[Mapping[str, str]] = {
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

_DTYPE_ITEMSIZE_MAP: typing.Final[Mapping[str, float]] = {
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
}


class DType(str, Enum):
    """Canonical tensor element data types across ML frameworks and hardware dialects.

    Attributes:
        float32 (DType): 32-bit IEEE 754 floating-point type.
        float16 (DType): 16-bit IEEE 754 half-precision floating-point type.
        bfloat16 (DType): 16-bit Brain Floating-Point type.
        float64 (DType): 64-bit IEEE 754 double-precision floating-point type.
        int64 (DType): 64-bit signed integer type.
        int32 (DType): 32-bit signed integer type.
        int16 (DType): 16-bit signed integer type.
        int8 (DType): 8-bit signed integer type.
        uint64 (DType): 64-bit unsigned integer type.
        uint32 (DType): 32-bit unsigned integer type.
        uint16 (DType): 16-bit unsigned integer type.
        uint8 (DType): 8-bit unsigned integer type.
        bool (DType): Boolean predicate/mask type.
        string (DType): Variable-width UTF-8 string tensor type.
        object (DType): Variable-width opaque reference descriptor type.
        complex64 (DType): 64-bit complex number (two float32 components).
        complex128 (DType): 128-bit complex number (two float64 components).
        float8_e4m3fn (DType): 8-bit floating-point E4M3 finite-only format.
        float8_e4m3b11fnuz (DType): 8-bit floating-point E4M3 bias-11 FNUZ format.
        float8_e5m2 (DType): 8-bit floating-point E5M2 format.
        fp8_e4m3fn (DType): Alias for 8-bit floating-point E4M3 finite-only format.
        fp8_e4m3fnuz (DType): 8-bit floating-point E4M3 FNUZ format.
        fp8_e5m2 (DType): Alias for 8-bit floating-point E5M2 format.
        fp8_e5m2fnuz (DType): 8-bit floating-point E5M2 FNUZ format.
        int4 (DType): 4-bit signed sub-byte quantized integer type.
        uint4 (DType): 4-bit unsigned sub-byte quantized integer type.
        int2 (DType): 2-bit signed sub-byte quantized integer type.
        qint8 (DType): Quantized 8-bit signed integer type.
        quint8 (DType): Quantized 8-bit unsigned integer type.
        qint4 (DType): Quantized 4-bit signed integer type.
    """

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
    def is_fixed_width(self) -> builtins.bool:
        """Return True if this DType has a deterministic fixed storage width in bytes.

        Returns:
            bool: True for fixed-width numeric and quantized types, False for string/object.
        """
        return not self.is_variable_width

    @property
    def itemsize(self) -> float:
        """Return the fixed storage size in bytes per element for this data type.

        Sub-byte quantized types (e.g. int4, uint4, qint4) return fractional byte sizes
        (e.g. 0.5 bytes for 4-bit, 0.25 bytes for 2-bit).

        Returns:
            float: Number of bytes per element.

        Raises:
            ValueError: If called on a variable-width DType (``DType.string`` or ``DType.object``).
        """
        if self.is_variable_width:
            raise ValueError(f"Variable-width DType {self.value!r} does not have a fixed byte itemsize.")
        return _DTYPE_ITEMSIZE_MAP[self.value]


DescribeValueTarget: typing_extensions.TypeAlias = typing.Union[
    "AttributeValue",
    InitializerValue,
    DimensionType,
    DTypeLikeProtocol,
    DeviceLikeProtocol,
    SymbolicNodeProtocol,
    SymbolicIntWrapperProtocol,
    CustomRuleHandleProtocol,
    LoopOptionsProtocol,
    MeshLikeProtocol,
    ArrayBufferProtocol,
    SubgraphLikeProtocol,
    pydantic.BaseModel,
    Iterable[typing.Union["AttributeValue", InitializerValue, pydantic.BaseModel, "DescribeValueTarget"]],
    Mapping[str, typing.Union["AttributeValue", InitializerValue, pydantic.BaseModel, "DescribeValueTarget"]],
]

_SIMPLE_TYPE_LABELS: tuple[tuple[type, str], ...] = (
    (bool, "bool"),
    (int, "int"),
    (float, "float"),
    (str, "str"),
    (bytes, "bytes"),
    (DType, "DType"),
    (SymInt, "SymInt"),
    (SymNode, "SymNode"),
    (tuple, "tuple"),
    (list, "list"),
    (dict, "dict"),
    (set, "set"),
)


def describe_value_type(
    val: DescribeValueTarget,
) -> str:
    """Return the unqualified Python runtime type name of a value without reflection or fragile splits.

    Args:
        val (DescribeValueTarget): Value whose runtime type name to format.

    Returns:
        str: Unqualified type name (e.g. 'int', 'float', 'list', 'dict', 'NoneType').
    """
    if val is None:
        return "NoneType"
    if isinstance(val, TensorShape):
        return "TensorShape"
    if isinstance(val, TensorSpec):
        return "TensorSpec"
    for candidate_cls, label in _SIMPLE_TYPE_LABELS:
        if isinstance(val, candidate_cls):
            return label
    raw_repr = str(type(val))
    segments = raw_repr.split("'")
    if len(segments) >= 2 and segments[1]:
        return segments[1].rsplit(".", 1)[-1]
    return raw_repr.strip("<>").rsplit(".", 1)[-1]


def normalize_dtype(val: DType | DTypeLikeProtocol | str) -> DType:
    """Normalize a ``DType``, ``DTypeLikeProtocol``, or string into a canonical ``DType``.

    Args:
        val (DType | DTypeLikeProtocol | str): Input data type representation.

    Returns:
        DType: Canonical ``DType`` enum member.

    Raises:
        TypeError: If ``val`` does not satisfy ``DType``, ``DTypeLikeProtocol``, or ``str``.
        ValueError: If the string representation is not a supported data type.
    """
    if isinstance(val, DType):
        return val
    if isinstance(val, str):
        return DType.from_str(val)
    if isinstance(val, DTypeLikeProtocol):
        if isinstance(val.value, str):
            return DType.from_str(val.value)
    raise TypeError(f"Unsupported dtype of type {describe_value_type(val)}: {val!r}")


def normalize_optional_dtype(val: DType | DTypeLikeProtocol | str | None) -> DType | None:
    """Normalize an optional ``DType | DTypeLikeProtocol | str | None`` into ``DType | None``.

    Args:
        val (DType | DTypeLikeProtocol | str | None): Optional input data type representation.

    Returns:
        DType | None: Canonical ``DType`` enum member or ``None``.
    """
    if val is None:
        return None
    return normalize_dtype(val)


def _validate_dimension_tuple(
    dims: (
        tuple[DimensionType, ...]
        | Sequence[DimensionType]
        | Iterable[DimensionType]
        | TensorShape
        | DimensionType
        | float
    ),
) -> tuple[DimensionType, ...]:
    """Validate and normalize dimensions for TensorShape.

    Args:
        dims: Input dimensions sequence, iterable, TensorShape, or single dimension.

    Returns:
        tuple[DimensionType, ...]: Validated dimension tuple.

    Raises:
        TypeError: If any dimension is a boolean or not a valid DimensionType.
    """
    if isinstance(dims, TensorShape):
        raw_dims: Sequence[DimensionType | float] = dims.dims
    elif isinstance(
        dims,
        (int, str, SymNode, SymInt, SymbolicNodeProtocol, SymbolicIntWrapperProtocol),
    ):
        raw_dims = (dims,)
    else:
        raw_dims = tuple(dims) if isinstance(dims, Iterable) else (dims,)

    validated: list[DimensionType] = []
    for d in raw_dims:
        if isinstance(d, bool):
            raise TypeError(f"Boolean value {d!r} is not a valid tensor dimension in TensorShape.")
        if isinstance(d, (int, str, SymNode, SymInt)):
            validated.append(d)
        elif isinstance(d, SymbolicIntWrapperProtocol):
            if not isinstance(d.node, SymbolicNodeProtocol):
                raise TypeError(f"SymbolicIntWrapperProtocol.node {d.node!r} does not satisfy SymbolicNodeProtocol.")
            validated.append(d)
        elif isinstance(d, SymbolicNodeProtocol):
            validated.append(d)
        else:
            raise TypeError(f"Value {d!r} of type {describe_value_type(d)} is not a valid DimensionType.")
    return tuple(validated)


def normalize_dimension_tuple(
    dims: (tuple[DimensionType, ...] | Sequence[DimensionType] | Iterable[DimensionType] | TensorShape | DimensionType),
) -> tuple[DimensionType, ...]:
    """Public helper normalizing a shape sequence, iterable, single dimension, or ``TensorShape`` into a tuple.

    Args:
        dims (Sequence[DimensionType] | Iterable[DimensionType] | TensorShape | DimensionType):
            Input dimensions or ``TensorShape``.

    Returns:
        tuple[DimensionType, ...]: Normalized tuple of ``DimensionType`` entries.
    """
    return _validate_dimension_tuple(dims)


class TensorShape(pydantic.BaseModel):
    """Represents a tensor shape with dimension query and symbolic manipulation capabilities.

    Attributes:
        dims (tuple[DimensionType, ...]): Tensor dimension sizes, symbolic names, or SymNode expressions.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    dims: tuple[DimensionType, ...] = ()

    def __init__(
        self,
        dims: tuple[DimensionType, ...]
        | Sequence[DimensionType]
        | Iterable[DimensionType]
        | TensorShape
        | DimensionType = (),
    ) -> None:
        """Initialize TensorShape normalizing inputs into a tuple.

        Args:
            dims: Input dimensions sequence, iterable, TensorShape, or single dimension.

        Raises:
            TypeError: If any dimension is a boolean value or invalid DimensionType.
        """
        validated_dims = _validate_dimension_tuple(dims)
        super().__init__(dims=validated_dims)

    @classmethod
    def from_sequence(
        cls,
        seq: Sequence[DimensionType] | Iterable[DimensionType] | TensorShape,
    ) -> TensorShape:
        """Construct a TensorShape from a sequence or iterable of dimensions.

        Args:
            seq (Sequence[DimensionType] | Iterable[DimensionType] | TensorShape): Dimension sequence.

        Returns:
            TensorShape: Constructed TensorShape instance.
        """
        return cls(seq)

    def to_list(self) -> list[DimensionType]:
        """Return dimensions as a list.

        Returns:
            list[DimensionType]: List of dimensions.
        """
        return list(self.dims)

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
            raise ValueError(f"Shape {self.dims} contains dynamic or symbolic dimensions.")
        res: list[int] = []
        for d in self.dims:
            if isinstance(d, int):
                res.append(d)
            elif isinstance(d, (SymNode, SymInt)):
                res.append(d.eval({}))
            else:
                res.append(int(str(d)))
        return tuple(res)

    def evaluate(self, bindings: Mapping[str, int]) -> tuple[int, ...]:
        """Evaluate all symbolic dimensions into concrete integers using runtime bindings.

        Args:
            bindings (Mapping[str, int]): Mapping from dimension variable names to concrete integers.

        Returns:
            tuple[int, ...]: Tuple of concrete integer dimension sizes.
        """
        env_dict = dict(bindings)
        res: list[int] = []
        for d in self.dims:
            if isinstance(d, int) and not isinstance(d, bool):
                res.append(d)
            elif isinstance(d, (SymNode, SymInt)):
                res.append(d.evaluate(env_dict))
            elif isinstance(d, SymbolicIntWrapperProtocol):
                res.append(int(d.node.eval(env_dict)))
            elif isinstance(d, SymbolicNodeProtocol):
                res.append(int(d.eval(env_dict)))
            elif isinstance(d, str) and d.isdigit():
                res.append(int(d))
            elif isinstance(d, str) and d in env_dict:
                res.append(env_dict[d])
            else:
                res.append(SymNode.to_node(d).evaluate(env_dict))
        return tuple(res)

    def broadcast_with(self, other: TensorShape | Sequence[DimensionType]) -> TensorShape:
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

    @typing.overload
    def __getitem__(self, idx: int) -> DimensionType:
        """Retrieve a single dimension by integer index.

        Args:
            idx (int): Zero-based dimension index.

        Returns:
            DimensionType: Dimension at index.
        """

    @typing.overload
    def __getitem__(self, idx: builtins.slice[int | None, int | None, int | None]) -> tuple[DimensionType, ...]:
        """Retrieve a slice of dimensions.

        Args:
            idx (slice[int | None, int | None, int | None]): Slice object.

        Returns:
            tuple[DimensionType, ...]: Sliced tuple of dimensions.
        """

    def __getitem__(
        self, idx: int | builtins.slice[int | None, int | None, int | None]
    ) -> DimensionType | tuple[DimensionType, ...]:
        """Retrieve dimension size by index or slice.

        Args:
            idx (int | slice[int | None, int | None, int | None]): Index or slice.

        Returns:
            DimensionType | tuple[DimensionType, ...]: Dimension size or tuple of sizes.
        """
        return self.dims[idx]

    def __eq__(self, other: object) -> bool:
        """Check equality against TensorShape, tuple, or list.

        Args:
            other (object): Other value to compare against.

        Returns:
            bool: True if dimensions match.
        """
        if isinstance(other, TensorShape):
            return self.dims == other.dims
        if isinstance(other, (tuple, list)):
            return self.dims == tuple(other)
        return False

    def __hash__(self) -> int:
        """Compute hash from dimensions tuple.

        Returns:
            int: Hash code.
        """
        return hash(self.dims)


ShapeMetadataType: typing_extensions.TypeAlias = typing.Union[
    tuple[DimensionType, ...],
    Sequence[DimensionType],
    TensorShape,
    DimensionType,
    None,
]


class TensorSpec(pydantic.BaseModel):
    """Specification for tensor values including shape, dtype, and sparsity.

    Attributes:
        shape (tuple[DimensionType, ...]): Dimensions of the tensor.
        dtype (DType): Tensor data type.
        sparsity (str | None): Sparsity layout or format if applicable.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    shape: tuple[DimensionType, ...]
    dtype: DType
    sparsity: str | None = None

    @pydantic.field_validator("dtype", mode="before")
    @classmethod
    def _validate_dtype_field(cls, val: DType | DTypeLikeProtocol | str) -> DType:
        """Normalize ``dtype`` field input from ``DType``, ``DTypeLikeProtocol``, or ``str``.

        Args:
            val (DType | DTypeLikeProtocol | str): Raw dtype value.

        Returns:
            DType: Canonical ``DType`` enum member.
        """
        return normalize_dtype(val)

    def __init__(
        self,
        shape: tuple[DimensionType, ...] | Sequence[DimensionType] | TensorShape,
        dtype: DType | DTypeLikeProtocol | str,
        sparsity: str | None = None,
    ) -> None:
        """Initialize TensorSpec with shape and dtype normalization.

        Args:
            shape (tuple[DimensionType, ...] | Sequence[DimensionType] | TensorShape): Tensor dimensions.
            dtype (DType | DTypeLikeProtocol | str): Data type as DType, DTypeLikeProtocol, or string name.
            sparsity (str | None): Optional sparsity layout.
        """
        norm_shape = _validate_dimension_tuple(shape)
        norm_dtype = normalize_dtype(dtype)
        super().__init__(shape=norm_shape, dtype=norm_dtype, sparsity=sparsity)

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

    def evaluate(self, bindings: Mapping[str, int]) -> TensorSpec:
        """Evaluate symbolic dimensions into concrete integers using runtime bindings.

        Args:
            bindings (Mapping[str, int]): Variable bindings mapping names to integer sizes.

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


def is_valid_loop_options_protocol(val: LoopOptionsProtocol) -> bool:
    """Validate field types on a ``LoopOptionsProtocol`` instance.

    Args:
        val (LoopOptionsProtocol): Candidate loop options object.

    Returns:
        bool: True if ``parallel_iterations`` and ``maximum_iterations`` are not booleans.
    """
    return not isinstance(val.parallel_iterations, bool) and not isinstance(val.maximum_iterations, bool)


def is_valid_device_protocol(val: DeviceLikeProtocol) -> bool:
    """Validate field types on a ``DeviceLikeProtocol`` instance.

    Args:
        val (DeviceLikeProtocol): Candidate device descriptor object.

    Returns:
        bool: True if ``index`` is not a boolean value.
    """
    return not isinstance(val.index, bool)


def _is_valid_protocol_attribute_leaf(val: AttributeValue) -> bool:
    """Check whether ``val`` is a valid structural protocol leaf for ``AttributeValue``.

    Args:
        val (AttributeValue): Candidate attribute leaf value.

    Returns:
        bool: True if ``val`` satisfies one of the accepted structural protocol contracts.
    """
    if isinstance(val, (DTypeLikeProtocol, CustomRuleHandleProtocol, SymbolicNodeProtocol)):
        return True
    if isinstance(val, DeviceLikeProtocol):
        return is_valid_device_protocol(val)
    if isinstance(val, SymbolicIntWrapperProtocol):
        return isinstance(val.node, SymbolicNodeProtocol)
    if isinstance(val, LoopOptionsProtocol):
        return is_valid_loop_options_protocol(val)
    if isinstance(val, ArrayBufferProtocol):
        return _is_valid_array_buffer(val)
    return False


def _validate_attribute_value(val: AttributeValue) -> AttributeValue:
    """Recursively validate an ``AttributeValue`` at runtime during Pydantic validation.

    Args:
        val (AttributeValue): Input attribute value.

    Returns:
        AttributeValue: Validated attribute value preserving tuples, lists, dicts, protocols, and IR types.

    Raises:
        ValueError: If ``val`` or any nested element is not a valid ``AttributeValue``.
    """
    if val is None or isinstance(
        val,
        (int, float, str, bool, bytes, SymNode, SymInt, DType, TensorShape, TensorSpec, pydantic.BaseModel),
    ):
        return val
    if _is_valid_protocol_attribute_leaf(val):
        return val
    if isinstance(val, Mapping):
        for key, item in val.items():
            if not isinstance(key, str):
                raise ValueError(f"Attribute mapping key must be str, got {describe_value_type(key)}.")
            _validate_attribute_value(item)
        return val
    if isinstance(val, Sequence):
        for item in val:
            _validate_attribute_value(item)
        return val
    raise ValueError(f"Unsupported AttributeValue of type {describe_value_type(val)}: {val!r}")


def _encode_json_safe_protocol_leaf(
    val: (
        DTypeLikeProtocol
        | DeviceLikeProtocol
        | SymbolicNodeProtocol
        | SymbolicIntWrapperProtocol
        | CustomRuleHandleProtocol
        | LoopOptionsProtocol
        | ArrayBufferProtocol
    ),
) -> dict[str, JsonValue]:
    """Encode a structural protocol ``AttributeValue`` leaf into a deterministic tagged JSON dictionary.

    Args:
        val: Structural protocol leaf instance.

    Returns:
        dict[str, JsonValue]: Deterministic JSON-compatible dictionary.
    """
    if isinstance(val, DTypeLikeProtocol):
        return {"__sw_type__": "DType", "value": val.value}
    if isinstance(val, DeviceLikeProtocol):
        return {"__sw_type__": "Device", "index": val.index, "type": val.type}
    if isinstance(val, (SymbolicNodeProtocol, SymbolicIntWrapperProtocol)):
        return {"__sw_type__": "SymDim", "expr": format_symbolic_protocol_expr(val)}
    if isinstance(val, CustomRuleHandleProtocol):
        return {"__sw_type__": "CustomRuleHandle", "handle_id": val.handle_id, "kind": val.kind}
    if isinstance(val, LoopOptionsProtocol):
        return {
            "__sw_type__": "LoopOptions",
            "maximum_iterations": val.maximum_iterations,
            "parallel_iterations": val.parallel_iterations,
            "swap_memory": val.swap_memory,
        }
    return {
        "__sw_type__": "ArrayBuffer",
        "data": base64.b64encode(val.tobytes()).decode("ascii"),
        "shape": list(val.shape),
    }


def _encode_json_safe_ir_leaf(
    val: bytes | DType | SymNode | SymInt | TensorShape | TensorSpec | pydantic.BaseModel,
) -> JsonValue:
    """Serialize a core IR or ``pydantic.BaseModel`` leaf into a JSON-compatible representation.

    Args:
        val (bytes | DType | SymNode | SymInt | TensorShape | TensorSpec | pydantic.BaseModel):
            IR leaf value.

    Returns:
        JsonValue: Deterministic JSON-compatible representation.
    """
    if isinstance(val, (bytes, DType)):
        return (
            {"__sw_type__": "bytes", "data": base64.b64encode(val).decode("ascii")}
            if isinstance(val, bytes)
            else {"__sw_type__": "DType", "value": val.value}
        )
    if isinstance(val, (SymNode, SymInt)):
        return str(val)
    if isinstance(val, TensorShape):
        return {"__sw_type__": "TensorShape", "dims": [_encode_json_safe_attribute(d) for d in val.dims]}
    if isinstance(val, TensorSpec):
        return {
            "__sw_type__": "TensorSpec",
            "dtype": val.dtype.value,
            "shape": [_encode_json_safe_attribute(d) for d in val.shape],
            "sparsity": val.sparsity,
        }
    dumped: JsonValue = val.model_dump(mode="json")
    return dumped


def _encode_json_safe_attribute(val: AttributeValue) -> JsonValue:
    """Recursively serialize an ``AttributeValue`` into a deterministic JSON-compatible structure.

    Args:
        val (AttributeValue): Validated attribute value.

    Returns:
        JsonValue: Deterministic JSON-compatible representation.
    """
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (bytes, DType, SymNode, SymInt, TensorShape, TensorSpec, pydantic.BaseModel)):
        return _encode_json_safe_ir_leaf(val)
    if isinstance(
        val,
        (
            DTypeLikeProtocol,
            DeviceLikeProtocol,
            SymbolicNodeProtocol,
            SymbolicIntWrapperProtocol,
            CustomRuleHandleProtocol,
            LoopOptionsProtocol,
            ArrayBufferProtocol,
        ),
    ):
        return _encode_json_safe_protocol_leaf(val)
    if isinstance(val, Mapping):
        return {str(k): _encode_json_safe_attribute(v) for k, v in val.items()}
    return [_encode_json_safe_attribute(item) for item in val]


def _serialize_attribute_pydantic(
    val: AttributeValue,
    info: SerializationInfoProtocol,
) -> AttributeValue | JsonValue:
    """Serialize ``AttributeValue`` cleanly for Pydantic Python and JSON serialization modes.

    Args:
        val (AttributeValue): Validated attribute value.
        info (SerializationInfoProtocol): Pydantic serialization context.

    Returns:
        AttributeValue | JsonValue: Raw value in Python mode, or deterministic JSON structure in JSON mode.
    """
    if info.mode == "json":
        return _encode_json_safe_attribute(val)
    return val


class _AttributeValuePydanticAnnotation:
    """Recursive Pydantic v2 validator and JSON Schema provider for ``AttributeValue``."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: type[AttributeValue],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.CoreSchema:
        """Return recursive runtime validator schema for ``AttributeValue``.

        Args:
            source_type (type[AttributeValue]): Annotated type.
            handler (pydantic.GetCoreSchemaHandler): Pydantic core schema handler.

        Returns:
            pydantic_core.CoreSchema: Pydantic CoreSchema instance.
        """
        del source_type, handler
        return core_schema.no_info_plain_validator_function(
            _validate_attribute_value,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_attribute_pydantic,
                info_arg=True,
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema_obj: pydantic_core.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> dict[str, JsonValue]:
        """Return a Draft 2020-12 union JSON Schema object for ``AttributeValue`` fields.

        Args:
            core_schema_obj (pydantic_core.CoreSchema): Core schema object.
            handler (pydantic.GetJsonSchemaHandler): JSON schema handler.

        Returns:
            dict[str, JsonValue]: Structured JSON Schema dictionary.
        """
        del core_schema_obj, handler
        return {
            "anyOf": [
                {"type": "integer"},
                {"type": "number"},
                {"type": "string"},
                {"type": "boolean"},
                {"type": "null"},
                {"type": "array"},
                {"type": "object"},
            ]
        }


AttributeValue: typing_extensions.TypeAlias = typing.Annotated[
    typing.Union[
        int,
        float,
        str,
        bool,
        bytes,
        None,
        SymNode,
        SymInt,
        DType,
        DTypeLikeProtocol,
        DeviceLikeProtocol,
        SymbolicNodeProtocol,
        SymbolicIntWrapperProtocol,
        CustomRuleHandleProtocol,
        LoopOptionsProtocol,
        ArrayBufferProtocol,
        TensorShape,
        TensorSpec,
        pydantic.BaseModel,
        Sequence["AttributeValue"],
        Mapping[str, "AttributeValue"],
    ],
    _AttributeValuePydanticAnnotation,
]

ProtocolAttributeLeaf: typing_extensions.TypeAlias = typing.Union[
    DTypeLikeProtocol,
    DeviceLikeProtocol,
    SymbolicNodeProtocol,
    SymbolicIntWrapperProtocol,
    CustomRuleHandleProtocol,
    LoopOptionsProtocol,
    ArrayBufferProtocol,
]


@dataclass(frozen=True)
class AttrNone:
    """Closed ``AttrTerm`` constructor representing ``None``.

    Attributes:
        value (None): Literal ``None`` payload.
    """

    value: None = None


@dataclass(frozen=True)
class AttrBool:
    """Closed ``AttrTerm`` constructor representing a boolean scalar (distinct from ``int``).

    Attributes:
        value (bool): Boolean payload.
    """

    value: bool


@dataclass(frozen=True)
class AttrInt:
    """Closed ``AttrTerm`` constructor representing a non-boolean integer scalar.

    Attributes:
        value (int): Non-boolean signed integer payload.
    """

    value: int


@dataclass(frozen=True)
class AttrFloat:
    """Closed ``AttrTerm`` constructor representing an IEEE-754 float scalar.

    Attributes:
        value (float): Floating-point payload.
    """

    value: float


@dataclass(frozen=True)
class AttrStr:
    """Closed ``AttrTerm`` constructor representing a UTF-8 string scalar (distinct from ``DType``).

    Attributes:
        value (str): String payload.
    """

    value: str


@dataclass(frozen=True)
class AttrBytes:
    """Closed ``AttrTerm`` constructor representing raw ``bytes``.

    Attributes:
        value (bytes): Raw byte string payload.
    """

    value: bytes


@dataclass(frozen=True)
class AttrDType:
    """Closed ``AttrTerm`` constructor representing a canonical ``DType`` enum member.

    Attributes:
        value (DType): Element data type payload.
    """

    value: DType


@dataclass(frozen=True)
class AttrShape:
    """Closed ``AttrTerm`` constructor representing a ``TensorShape``.

    Attributes:
        value (TensorShape): TensorShape payload.
    """

    value: TensorShape


@dataclass(frozen=True)
class AttrTensorSpec:
    """Closed ``AttrTerm`` constructor representing a ``TensorSpec``.

    Attributes:
        value (TensorSpec): TensorSpec payload.
    """

    value: TensorSpec


@dataclass(frozen=True)
class AttrSymNode:
    """Closed ``AttrTerm`` constructor representing a ``SymNode`` expression tree.

    Attributes:
        value (SymNode): Symbolic AST node payload.
    """

    value: SymNode


@dataclass(frozen=True)
class AttrSymInt:
    """Closed ``AttrTerm`` constructor representing a ``SymInt`` dimension wrapper.

    Attributes:
        value (SymInt): SymInt wrapper payload.
    """

    value: SymInt


@dataclass(frozen=True)
class AttrModel:
    """Closed ``AttrTerm`` constructor representing a ``pydantic.BaseModel`` (e.g. ``LogicalGraph``, ``PartitionSpec``).

    Attributes:
        value (pydantic.BaseModel): Validated Pydantic model payload.
    """

    value: pydantic.BaseModel


@dataclass(frozen=True)
class AttrProtocolLeaf:
    """Closed ``AttrTerm`` constructor representing an external Tier-1 structural protocol leaf.

    Attributes:
        value (ProtocolAttributeLeaf): External protocol-satisfying value.
    """

    value: ProtocolAttributeLeaf


@dataclass(frozen=True)
class AttrSeq:
    """Closed ``AttrTerm`` constructor representing a homogeneous or heterogeneous sequence.

    Attributes:
        items (tuple[AttrTerm, ...]): Recursively projected child terms.
        is_tuple (bool): True when the original sequence was a ``tuple`` rather than a ``list``.
    """

    items: tuple[AttrTerm, ...]
    is_tuple: bool = False


@dataclass(frozen=True)
class AttrMap:
    """Closed ``AttrTerm`` constructor representing a string-keyed mapping.

    Attributes:
        entries (tuple[tuple[str, AttrTerm], ...]): Deterministic ``(key, term)`` pairs.
    """

    entries: tuple[tuple[str, AttrTerm], ...]


AttrScalarTerm: typing_extensions.TypeAlias = typing.Union[
    AttrNone,
    AttrBool,
    AttrInt,
    AttrFloat,
    AttrStr,
    AttrBytes,
]
AttrDomainTerm: typing_extensions.TypeAlias = typing.Union[
    AttrDType,
    AttrShape,
    AttrTensorSpec,
    AttrSymNode,
    AttrSymInt,
    AttrModel,
    AttrProtocolLeaf,
]
AttrCompositeTerm: typing_extensions.TypeAlias = typing.Union[AttrSeq, AttrMap]
AttrTerm: typing_extensions.TypeAlias = typing.Union[
    AttrScalarTerm,
    AttrDomainTerm,
    AttrCompositeTerm,
]


def _project_scalar_or_domain_leaf(
    val: AttributeValue,
) -> AttrScalarTerm | AttrDomainTerm | None:
    """Project non-container ``AttributeValue`` leaves into ``AttrScalarTerm | AttrDomainTerm``.

    Orders checks carefully (``bool`` before ``int``, ``DType`` before ``str``/``DTypeLikeProtocol``,
    ``TensorShape``/``TensorSpec`` before ``BaseModel``, ``SymInt``/``SymNode`` before protocols).

    Args:
        val (AttributeValue): Candidate attribute value.

    Returns:
        AttrScalarTerm | AttrDomainTerm | None: Projected leaf term, or ``None`` if container.
    """
    res: AttrScalarTerm | AttrDomainTerm | None
    if val is None:
        res = AttrNone()
    elif isinstance(val, bool):
        res = AttrBool(val)
    elif isinstance(val, int):
        res = AttrInt(val)
    elif isinstance(val, float):
        res = AttrFloat(val)
    elif isinstance(val, DType):
        res = AttrDType(val)
    elif isinstance(val, str):
        res = AttrStr(val)
    elif isinstance(val, bytes):
        res = AttrBytes(val)
    else:
        res = _project_domain_object_leaf(val)
    return res


def _project_domain_object_leaf(
    val: AttributeValue,
) -> AttrDomainTerm | None:
    """Project rich domain/model/protocol ``AttributeValue`` leaves into ``AttrDomainTerm``.

    Args:
        val (AttributeValue): Non-primitive candidate attribute value.

    Returns:
        AttrDomainTerm | None: Projected domain term, or ``None`` if sequence/mapping container.
    """
    res: AttrDomainTerm | None = None
    if isinstance(val, TensorShape):
        res = AttrShape(val)
    elif isinstance(val, TensorSpec):
        res = AttrTensorSpec(val)
    elif isinstance(val, SymInt):
        res = AttrSymInt(val)
    elif isinstance(val, SymNode):
        res = AttrSymNode(val)
    elif isinstance(val, pydantic.BaseModel):
        res = AttrModel(val)
    elif isinstance(
        val,
        (
            DTypeLikeProtocol,
            DeviceLikeProtocol,
            SymbolicNodeProtocol,
            SymbolicIntWrapperProtocol,
            CustomRuleHandleProtocol,
            LoopOptionsProtocol,
            ArrayBufferProtocol,
        ),
    ):
        res = AttrProtocolLeaf(val)
    return res


def project_attribute_value(val: AttributeValue) -> AttrTerm:
    """Total projection of any ``AttributeValue`` into its closed ``AttrTerm`` normal-form ADT.

    Centralizes ``bool <: int``, ``DType <: str``, ``TensorShape <: BaseModel``, and
    ``SymNode <: SymbolicNodeProtocol`` precedence in one single place.

    Args:
        val (AttributeValue): Validated IR attribute value.

    Returns:
        AttrTerm: Disjoint tagged ``AttrTerm`` node.
    """
    leaf = _project_scalar_or_domain_leaf(val)
    if leaf is not None:
        return leaf
    if isinstance(val, Mapping):
        return AttrMap(tuple((str(k), project_attribute_value(v)) for k, v in val.items()))
    seq_items = tuple(project_attribute_value(item) for item in val) if isinstance(val, Sequence) else ()
    return AttrSeq(items=seq_items, is_tuple=isinstance(val, tuple))


def _embed_scalar_term(term: AttrScalarTerm) -> AttributeValue:
    """Exhaustively embed an ``AttrScalarTerm`` back into ``AttributeValue``.

    Args:
        term (AttrScalarTerm): Closed scalar term.

    Returns:
        AttributeValue: Embedded primitive value.
    """
    out: AttributeValue
    if isinstance(term, AttrNone):
        out = None
    elif isinstance(term, AttrBool):
        out = term.value
    elif isinstance(term, AttrInt):
        out = term.value
    elif isinstance(term, AttrFloat):
        out = term.value
    elif isinstance(term, AttrStr):
        out = term.value
    elif isinstance(term, AttrBytes):
        out = term.value
    else:
        typing_extensions.assert_never(term)
    return out


def _embed_domain_term(term: AttrDomainTerm) -> AttributeValue:
    """Exhaustively embed an ``AttrDomainTerm`` back into ``AttributeValue``.

    Args:
        term (AttrDomainTerm): Closed domain/model/protocol term.

    Returns:
        AttributeValue: Embedded rich domain value.
    """
    out: AttributeValue
    if isinstance(term, AttrDType):
        out = term.value
    elif isinstance(term, AttrShape):
        out = term.value
    elif isinstance(term, AttrTensorSpec):
        out = term.value
    elif isinstance(term, AttrSymNode):
        out = term.value
    elif isinstance(term, AttrSymInt):
        out = term.value
    elif isinstance(term, AttrModel):
        out = term.value
    elif isinstance(term, AttrProtocolLeaf):
        out = term.value
    else:
        typing_extensions.assert_never(term)
    return out


def _embed_composite_term(term: AttrCompositeTerm) -> AttributeValue:
    """Exhaustively embed an ``AttrCompositeTerm`` back into ``AttributeValue``.

    Args:
        term (AttrCompositeTerm): Closed sequence or mapping term.

    Returns:
        AttributeValue: Embedded tuple, list, or dict value.
    """
    out: AttributeValue
    if isinstance(term, AttrSeq):
        children = [embed_attribute_value(item) for item in term.items]
        out = tuple(children) if term.is_tuple else children
    elif isinstance(term, AttrMap):
        out = {k: embed_attribute_value(v) for k, v in term.entries}
    else:
        typing_extensions.assert_never(term)
    return out


def embed_attribute_value(term: AttrTerm) -> AttributeValue:
    """Exhaustively embed a closed ``AttrTerm`` normal-form value back into ``AttributeValue``.

    Args:
        term (AttrTerm): Closed normal-form attribute term.

    Returns:
        AttributeValue: Reconstructed runtime ``AttributeValue``.
    """
    out: AttributeValue
    if isinstance(term, (AttrNone, AttrBool, AttrInt, AttrFloat, AttrStr, AttrBytes)):
        out = _embed_scalar_term(term)
    elif isinstance(
        term,
        (AttrDType, AttrShape, AttrTensorSpec, AttrSymNode, AttrSymInt, AttrModel, AttrProtocolLeaf),
    ):
        out = _embed_domain_term(term)
    elif isinstance(term, (AttrSeq, AttrMap)):
        out = _embed_composite_term(term)
    else:
        typing_extensions.assert_never(term)
    return out
