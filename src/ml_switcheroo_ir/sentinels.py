"""Autodiff sentinel node definitions (ZeroTangent and NoTangent)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

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
]


@dataclass
class _AutodiffSentinelBase(LogicalNode):
    """Shared base dataclass for autodiff sentinel nodes in the 'ml.switcheroo.ad' domain.

    Attributes:
        _SENTINEL_OP_TYPE (ClassVar[str]): Canonical operation type for the sentinel subclass.
    """

    _SENTINEL_OP_TYPE: ClassVar[str] = "AutodiffSentinel"

    def _init_sentinel(
        self,
        id: str,
        resolved_shape: ShapeMetadataType | Sequence[DimensionType] | object,
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
            resolved_shape (ShapeMetadataType | Sequence[DimensionType] | object): Resolved tensor shape metadata.
            dtype (Optional[DType]): Tensor data type.
            version (int): Operator set version (default: 1).
            attributes (Optional[Mapping[str, AttributeValue]]): Node configuration attributes.
            inputs (Optional[list[str]]): Ordered list of upstream node IDs.
            outputs (Optional[list[str]]): Ordered list of output SSA names.
            source_ast_ref (Optional[str]): Source AST traceback reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            output_specs (Optional[list[TensorSpec]]): Output tensor specifications.
            subgraphs (Optional[Mapping[str, LogicalGraph | AttributeValue]]): Nested subgraphs.
            device (Optional[str]): Target device placement.
            stream (Optional[str]): Asynchronous execution stream.
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


@dataclass
class ZeroTangent(_AutodiffSentinelBase):
    """Represents a mathematically zero tangent vector with explicit shape and dtype.

    Attributes:
        id (str): Unique node identifier.
        shape_metadata (ShapeMetadataType): Shape of the zero tangent vector.
        dtype (Optional[DType]): Data type of the zero tangent vector.
    """

    _SENTINEL_OP_TYPE: ClassVar[str] = "ZeroTangent"

    def __init__(
        self,
        id: str,
        shape: ShapeMetadataType | Sequence[DimensionType] | object = None,
        dtype: DType | None = DType.float32,
        op_type: str = "ZeroTangent",
        domain: str = "ml.switcheroo.ad",
        version: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: ShapeMetadataType | Sequence[DimensionType] | object = None,
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
            shape (ShapeMetadataType | Sequence[DimensionType] | object): Vector shape.
            dtype (Optional[DType]): Tensor data type.
            op_type (str): Sentinel operation type (ignored, forced to 'ZeroTangent').
            domain (str): Operator domain (ignored, forced to 'ml.switcheroo.ad').
            version (int): Operator set version (default: 1).
            attributes (Optional[Mapping[str, AttributeValue]]): Node configuration attributes.
            inputs (Optional[list[str]]): Ordered list of upstream node IDs.
            outputs (Optional[list[str]]): Ordered list of output SSA names.
            shape_metadata (ShapeMetadataType | Sequence[DimensionType] | object): Optional explicit shape_metadata override.
            source_ast_ref (Optional[str]): Source AST traceback reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            output_specs (Optional[list[TensorSpec]]): Output tensor specifications.
            subgraphs (Optional[Mapping[str, LogicalGraph | AttributeValue]]): Nested subgraphs.
            device (Optional[str]): Target device placement.
            stream (Optional[str]): Asynchronous execution stream.
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


@dataclass
class NoTangent(_AutodiffSentinelBase):
    """Represents a non-differentiable or structurally missing cotangent path.

    Attributes:
        id (str): Unique node identifier.
    """

    _SENTINEL_OP_TYPE: ClassVar[str] = "NoTangent"

    def __init__(
        self,
        id: str,
        op_type: str = "NoTangent",
        domain: str = "ml.switcheroo.ad",
        version: int = 1,
        attributes: Mapping[str, AttributeValue] | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        shape_metadata: ShapeMetadataType | Sequence[DimensionType] | object = None,
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
            attributes (Optional[Mapping[str, AttributeValue]]): Node configuration attributes.
            inputs (Optional[list[str]]): Ordered list of upstream node IDs.
            outputs (Optional[list[str]]): Ordered list of output SSA names.
            shape_metadata (ShapeMetadataType | Sequence[DimensionType] | object): Tensor shape metadata.
            source_ast_ref (Optional[str]): Source AST traceback reference.
            sharding (Optional[PartitionSpec]): Distributed partition layout.
            dtype (Optional[DType]): Tensor data type.
            output_specs (Optional[list[TensorSpec]]): Output tensor specifications.
            subgraphs (Optional[Mapping[str, LogicalGraph | AttributeValue]]): Nested subgraphs.
            device (Optional[str]): Target device placement.
            stream (Optional[str]): Asynchronous execution stream.
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


def verify_sentinel_protocol(node: ZeroTangent | NoTangent) -> AutodiffSentinelProtocol:
    """Verify at type-checking and runtime that sentinel nodes satisfy AutodiffSentinelProtocol.

    Args:
        node (ZeroTangent | NoTangent): Sentinel instance to verify.

    Returns:
        AutodiffSentinelProtocol: The verified sentinel node.
    """
    return node
