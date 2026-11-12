"""Autodiff sentinel node definitions (ZeroTangent and NoTangent)."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping

import typing_extensions

from ml_switcheroo_ir.distributed import PartitionSpec
from ml_switcheroo_ir.graph import LogicalGraph, LogicalNode, LogicalNodeKwargs
from ml_switcheroo_ir.shapes import DimensionType
from ml_switcheroo_ir.types import (
    AttributeValue,
    AutodiffSentinelProtocol,
    DType,
    ShapeMetadataType,
    TensorSpec,
)

__all__ = [
    "NoTangent",
    "SentinelNodeKwargs",
    "ZeroTangent",
    "ZeroTangentKwargs",
    "_AutodiffSentinelBase",
    "verify_sentinel_protocol",
]


class ZeroTangentKwargs(typing_extensions.TypedDict, total=False):
    """Structured optional keyword parameters for ``ZeroTangent`` construction.

    Attributes:
        op_type (str): Optional ignored operation type override (forced to ``'ZeroTangent'``).
        domain (str): Optional ignored domain override (forced to ``'ml.switcheroo.ad'``).
        version (int): Operator set version (default: 1).
        attributes (Mapping[str, AttributeValue] | None): Operator configuration attributes.
        inputs (list[str] | None): Ordered list of upstream LogicalNode IDs or SSA names.
        outputs (Iterable[str] | None): Ordered list of output SSA names.
        shape_metadata (ShapeMetadataType | Iterable[DimensionType]): Optional explicit shape_metadata override.
        source_ast_ref (str | None): Source AST traceback reference.
        sharding (PartitionSpec | None): Distributed SPMD partition layout.
        output_specs (list[TensorSpec] | None): Explicit output tensor specifications.
        subgraphs (Mapping[str, LogicalGraph | AttributeValue] | None): Nested control-flow subgraphs.
        device (str | None): Target execution device placement.
        stream (str | None): Asynchronous execution stream name.
    """

    op_type: str
    domain: str
    version: int
    attributes: Mapping[str, AttributeValue] | None
    inputs: list[str] | None
    outputs: Iterable[str] | None
    shape_metadata: ShapeMetadataType | Iterable[DimensionType]
    source_ast_ref: str | None
    sharding: PartitionSpec | None
    output_specs: list[TensorSpec] | None
    subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None
    device: str | None
    stream: str | None


class SentinelNodeKwargs(ZeroTangentKwargs, total=False):
    """Structured optional keyword parameters for ``NoTangent`` construction.

    Attributes:
        dtype (DType | None): Optional tensor data type.
    """

    dtype: DType | None


class _AutodiffSentinelBase(LogicalNode):
    """Shared base Pydantic model for autodiff sentinel nodes in the 'ml.switcheroo.ad' domain.

    Attributes:
        _SENTINEL_OP_TYPE (typing.ClassVar[str]): Canonical operation type for the sentinel subclass.
    """

    _SENTINEL_OP_TYPE: typing.ClassVar[str] = "AutodiffSentinel"

    def _init_sentinel(
        self,
        id: str,
        resolved_shape: ShapeMetadataType | Iterable[DimensionType],
        resolved_dtype: DType | None,
        extra: ZeroTangentKwargs | SentinelNodeKwargs,
    ) -> None:
        """Initialize the underlying LogicalNode for an autodiff sentinel.

        Args:
            id (str): Unique node identifier.
            resolved_shape (ShapeMetadataType | Iterable[DimensionType]): Resolved tensor shape metadata.
            resolved_dtype (DType | None): Tensor data type.
            extra (ZeroTangentKwargs | SentinelNodeKwargs): Structured optional node fields.
        """
        version = extra.get("version", 1)
        node_kwargs: LogicalNodeKwargs = {
            "attributes": extra.get("attributes"),
            "inputs": extra.get("inputs"),
            "outputs": extra.get("outputs"),
            "shape_metadata": resolved_shape,
            "source_ast_ref": extra.get("source_ast_ref"),
            "sharding": extra.get("sharding"),
            "dtype": resolved_dtype,
            "output_specs": extra.get("output_specs"),
            "subgraphs": extra.get("subgraphs"),
            "device": extra.get("device"),
            "stream": extra.get("stream"),
        }
        super().__init__(
            id=id,
            op_type=self._SENTINEL_OP_TYPE,
            domain="ml.switcheroo.ad",
            version=version,
            **node_kwargs,
        )


class ZeroTangent(_AutodiffSentinelBase):
    """Represents a mathematically zero tangent vector with explicit shape and dtype.

    Attributes:
        id (str): Unique node identifier.
        op_type (str): Sentinel operation type ('ZeroTangent').
        domain (str): Operator domain ('ml.switcheroo.ad').
        shape_metadata (tuple[DimensionType, ...] | None): Shape of the zero tangent vector.
        dtype (DType | None): Data type of the zero tangent vector.
    """

    _SENTINEL_OP_TYPE: typing.ClassVar[str] = "ZeroTangent"

    def __init__(
        self,
        id: str,
        shape: ShapeMetadataType | Iterable[DimensionType] = None,
        dtype: DType | None = DType.float32,
        **kwargs: typing_extensions.Unpack[ZeroTangentKwargs],
    ) -> None:
        """Initialize ZeroTangent sentinel node.

        Args:
            id (str): Unique node identifier.
            shape (ShapeMetadataType | Iterable[DimensionType]): Vector shape.
            dtype (DType | None): Tensor data type (default: ``DType.float32``).
            **kwargs (typing_extensions.Unpack[ZeroTangentKwargs]): Structured optional node fields.
        """
        explicit_shape = kwargs.get("shape_metadata")
        resolved_shape = explicit_shape if explicit_shape is not None else shape
        self._init_sentinel(
            id=id,
            resolved_shape=resolved_shape,
            resolved_dtype=dtype,
            extra=kwargs,
        )


class NoTangent(_AutodiffSentinelBase):
    """Represents a non-differentiable or structurally missing cotangent path.

    Attributes:
        id (str): Unique node identifier.
        op_type (str): Sentinel operation type ('NoTangent').
        domain (str): Operator domain ('ml.switcheroo.ad').
    """

    _SENTINEL_OP_TYPE: typing.ClassVar[str] = "NoTangent"

    def __init__(
        self,
        id: str,
        **kwargs: typing_extensions.Unpack[SentinelNodeKwargs],
    ) -> None:
        """Initialize NoTangent sentinel node.

        Args:
            id (str): Unique node identifier.
            **kwargs (typing_extensions.Unpack[SentinelNodeKwargs]): Structured optional node fields.
        """
        self._init_sentinel(
            id=id,
            resolved_shape=kwargs.get("shape_metadata"),
            resolved_dtype=kwargs.get("dtype"),
            extra=kwargs,
        )


def verify_sentinel_protocol(
    node: ZeroTangent | NoTangent,
) -> AutodiffSentinelProtocol:
    """Verify statically and at runtime that sentinel nodes satisfy AutodiffSentinelProtocol.

    Args:
        node (ZeroTangent | NoTangent): Sentinel instance to verify.

    Returns:
        AutodiffSentinelProtocol: The verified sentinel node.

    Raises:
        TypeError: If node is not an instance of _AutodiffSentinelBase or has invalid domain.
    """
    if node.domain != "ml.switcheroo.ad":
        raise TypeError(f"Expected autodiff sentinel node in domain 'ml.switcheroo.ad', got {node!r}.")
    return node
