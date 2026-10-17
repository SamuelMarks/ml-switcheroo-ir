"""Autodiff sentinel node definitions (ZeroTangent and NoTangent)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from ml_switcheroo_ir.graph import LogicalGraph, LogicalNode, PartitionSpec
from ml_switcheroo_ir.shapes import DimensionType
from ml_switcheroo_ir.types import (
    AttributeValue,
    DType,
    ShapeMetadataType,
    TensorSpec,
)

__all__ = [
    "NoTangent",
    "ZeroTangent",
]


@dataclass
class ZeroTangent(LogicalNode):
    """Represents a mathematically zero tangent vector with explicit shape and dtype.

    Attributes:
        id (str): Unique node identifier.
        shape_metadata (ShapeMetadataType): Shape of the zero tangent vector.
        dtype (Optional[DType]): Data type of the zero tangent vector.
    """

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
        super().__init__(
            id=id,
            op_type="ZeroTangent",
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
class NoTangent(LogicalNode):
    """Represents a non-differentiable or structurally missing cotangent path.

    Attributes:
        id (str): Unique node identifier.
    """

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
        super().__init__(
            id=id,
            op_type="NoTangent",
            domain="ml.switcheroo.ad",
            version=version,
            attributes=attributes,
            inputs=inputs,
            outputs=outputs,
            shape_metadata=shape_metadata,
            source_ast_ref=source_ast_ref,
            sharding=sharding,
            dtype=dtype,
            output_specs=output_specs,
            subgraphs=subgraphs,
            device=device,
            stream=stream,
        )
