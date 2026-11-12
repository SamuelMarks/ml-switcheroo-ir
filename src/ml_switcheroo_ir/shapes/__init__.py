"""Symbolic shape system, dimension algebra, broadcasting, and shape inference.

Provides zero-dependency symbolic dimension representation, algebraic expression trees,
polynomial canonicalization, NumPy-compliant broadcasting, and matrix multiplication shape inference.
"""

from __future__ import annotations

from ml_switcheroo_ir.shapes.ast import (
    DimensionType,
    ShapeType,
    SymBinaryOp,
    SymCondition,
    SymConst,
    SymInt,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
)
from ml_switcheroo_ir.shapes.inference import (
    SHAPE_TRANSFER_REGISTRY,
    ShapeNodeProtocol,
    ShapeTracker,
    ShapeTransferContext,
    SupportsDirectShapeProtocol,
    SupportsShapeMetadataProtocol,
    SupportsShapeNodesProtocol,
    TelemetryValueType,
    broadcast_dimension,
    broadcast_shapes,
    infer_node_shapes,
    infer_op_output_shape,
    matmul_shape,
    normalize_axis,
    resolve_symbolic_bounds,
)
from ml_switcheroo_ir.shapes.parser import (
    SymbolicExpressionParser,
    SymbolicSolver,
    parse_symbolic_expr,
    tokenize_symbolic_expr,
)
from ml_switcheroo_ir.shapes.polynomial import (
    MonomialType,
    Polynomial,
    poly_to_symnode,
)
from ml_switcheroo_ir.shapes.tracker import (
    DimensionBound,
    ShapeMismatchError,
    SymbolicConstraintTracker,
)

__all__ = [
    "DimensionBound",
    "DimensionType",
    "MonomialType",
    "Polynomial",
    "SHAPE_TRANSFER_REGISTRY",
    "ShapeMismatchError",
    "ShapeNodeProtocol",
    "ShapeTracker",
    "ShapeTransferContext",
    "ShapeType",
    "SupportsDirectShapeProtocol",
    "SupportsShapeMetadataProtocol",
    "SupportsShapeNodesProtocol",
    "SymBinaryOp",
    "SymCondition",
    "SymConst",
    "SymInt",
    "SymNode",
    "SymPiecewise",
    "SymUnaryOp",
    "SymVar",
    "SymbolicConstraintTracker",
    "SymbolicExpressionParser",
    "SymbolicSolver",
    "TelemetryValueType",
    "broadcast_dimension",
    "broadcast_shapes",
    "infer_node_shapes",
    "infer_op_output_shape",
    "matmul_shape",
    "normalize_axis",
    "parse_symbolic_expr",
    "poly_to_symnode",
    "resolve_symbolic_bounds",
    "tokenize_symbolic_expr",
]
