"""Autodiff sentinel node definitions (ZeroTangent and NoTangent)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ml_switcheroo_ir.graph import LogicalNode
from ml_switcheroo_ir.shapes import DimensionType
from ml_switcheroo_ir.types import DType, ShapeMetadataType

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
        **kwargs: Any,
    ) -> None:
        """Initialize ZeroTangent sentinel node.

        Args:
            id (str): Unique node identifier.
            shape (ShapeMetadataType | Sequence[DimensionType] | object): Vector shape.
            dtype (Optional[DType]): Tensor data type.
            **kwargs (Any): Additional node keyword arguments.
        """
        kwargs.pop("op_type", None)
        kwargs.pop("domain", None)
        shape_val = kwargs.pop("shape_metadata", shape)
        dtype_val = kwargs.pop("dtype", dtype)
        super().__init__(
            id=id,
            op_type="ZeroTangent",
            domain="ml.switcheroo.ad",
            shape_metadata=shape_val,
            dtype=dtype_val,
            **kwargs,
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
        **kwargs: Any,
    ) -> None:
        """Initialize NoTangent sentinel node.

        Args:
            id (str): Unique node identifier.
            **kwargs (Any): Additional node keyword arguments.
        """
        kwargs.pop("op_type", None)
        kwargs.pop("domain", None)
        super().__init__(
            id=id,
            op_type="NoTangent",
            domain="ml.switcheroo.ad",
            **kwargs,
        )
