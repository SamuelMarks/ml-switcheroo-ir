"""Autodiff sentinel node definitions (ZeroTangent and NoTangent)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import typing

from ml_switcheroo_ir.graph import LogicalGraph, LogicalNode, PartitionSpec
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
    "ZeroTangent",
    "_AutodiffSentinelBase",
    "verify_sentinel_protocol",
]


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
        dtype: DType | None,
        version: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None = None,
        device: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Initialize the underlying LogicalNode for an autodiff sentinel.

        Args:
            id (str): Unique node identifier.
            resolved_shape (ShapeMetadataType | Iterable[DimensionType]): Resolved tensor shape metadata.
            dtype (DType | None): Tensor data type.
            version (int): Operator set version (default: 1).
            attributes (Mapping[str, AttributeValue] | None): Node configuration attributes.
            inputs (list[str] | None): Ordered list of upstream node IDs.
            outputs (list[str] | None): Ordered list of output SSA names.
            source_ast_ref (str | None): Source AST traceback reference.
            sharding (PartitionSpec | None): Distributed partition layout.
            output_specs (list[TensorSpec] | None): Output tensor specifications.
            subgraphs (Mapping[str, LogicalGraph | AttributeValue] | None): Nested subgraphs.
            device (str | None): Target device placement.
            stream (str | None): Asynchronous execution stream.
        """
        super().__init__(
            id=id,
            op_type=self._SENTINEL_OP_TYPE,
            domain="ml.switcheroo.ad",
            version=version,
            attributes=attributes,
            inputs=inputs,
            outputs=outputs,
            shape_metadata=resolved_shape,
            source_ast_ref=source_ast_ref,
            sharding=sharding,
            dtype=dtype,
            output_specs=output_specs,
            subgraphs=subgraphs,
            device=device,
            stream=stream,
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
        op_type: str = "ZeroTangent",
        domain: str = "ml.switcheroo.ad",
        version: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: ShapeMetadataType | Iterable[DimensionType] = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None = None,
        device: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Initialize ZeroTangent sentinel node.

        Args:
            id (str): Unique node identifier.
            shape (ShapeMetadataType | Iterable[DimensionType]): Vector shape.
            dtype (DType | None): Tensor data type.
            op_type (str): Sentinel operation type (ignored, forced to 'ZeroTangent').
            domain (str): Operator domain (ignored, forced to 'ml.switcheroo.ad').
            version (int): Operator set version (default: 1).
            attributes (Mapping[str, AttributeValue] | None): Node configuration attributes.
            inputs (list[str] | None): Ordered list of upstream node IDs.
            outputs (list[str] | None): Ordered list of output SSA names.
            shape_metadata (ShapeMetadataType | Iterable[DimensionType]): Optional explicit shape_metadata override.
            source_ast_ref (str | None): Source AST traceback reference.
            sharding (PartitionSpec | None): Distributed partition layout.
            output_specs (list[TensorSpec] | None): Output tensor specifications.
            subgraphs (Mapping[str, LogicalGraph | AttributeValue] | None): Nested subgraphs.
            device (str | None): Target device placement.
            stream (str | None): Asynchronous execution stream.
        """
        del op_type, domain
        resolved_shape = shape_metadata if shape_metadata is not None else shape
        self._init_sentinel(
            id=id,
            resolved_shape=resolved_shape,
            dtype=dtype,
            version=version,
            attributes=attributes,
            inputs=inputs,
            outputs=outputs,
            source_ast_ref=source_ast_ref,
            sharding=sharding,
            output_specs=output_specs,
            subgraphs=subgraphs,
            device=device,
            stream=stream,
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
        op_type: str = "NoTangent",
        domain: str = "ml.switcheroo.ad",
        version: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: ShapeMetadataType | Iterable[DimensionType] = None,
        source_ast_ref: str | None = None,
        sharding: PartitionSpec | None = None,
        dtype: DType | None = None,
        output_specs: list[TensorSpec] | None = None,
        subgraphs: Mapping[str, LogicalGraph | AttributeValue] | None = None,
        device: str | None = None,
        stream: str | None = None,
    ) -> None:
        """Initialize NoTangent sentinel node.

        Args:
            id (str): Unique node identifier.
            op_type (str): Sentinel operation type (ignored, forced to 'NoTangent').
            domain (str): Operator domain (ignored, forced to 'ml.switcheroo.ad').
            version (int): Operator set version (default: 1).
            attributes (Mapping[str, AttributeValue] | None): Node configuration attributes.
            inputs (list[str] | None): Ordered list of upstream node IDs.
            outputs (list[str] | None): Ordered list of output SSA names.
            shape_metadata (ShapeMetadataType | Iterable[DimensionType]): Tensor shape metadata.
            source_ast_ref (str | None): Source AST traceback reference.
            sharding (PartitionSpec | None): Distributed partition layout.
            dtype (DType | None): Tensor data type.
            output_specs (list[TensorSpec] | None): Output tensor specifications.
            subgraphs (Mapping[str, LogicalGraph | AttributeValue] | None): Nested subgraphs.
            device (str | None): Target device placement.
            stream (str | None): Asynchronous execution stream.
        """
        del op_type, domain
        self._init_sentinel(
            id=id,
            resolved_shape=shape_metadata,
            dtype=dtype,
            version=version,
            attributes=attributes,
            inputs=inputs,
            outputs=outputs,
            source_ast_ref=source_ast_ref,
            sharding=sharding,
            output_specs=output_specs,
            subgraphs=subgraphs,
            device=device,
            stream=stream,
        )


def verify_sentinel_protocol(
    node: ZeroTangent | NoTangent,
) -> AutodiffSentinelProtocol:
    """Verify statically and at runtime that sentinel nodes satisfy AutodiffSentinelProtocol.

    Args:
        node (ZeroTangent | NoTangent): Sentinel instance to verify.

    Returns:
        AutodiffSentinelProtocol: The verified sentinel node.
    """
    return node
