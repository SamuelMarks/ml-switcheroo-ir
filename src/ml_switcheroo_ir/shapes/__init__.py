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
    ShapeNodeProtocol,
    ShapeTracker,
    SupportsShapeNodesProtocol,
    TelemetryValueType,
    broadcast_dimension,
    broadcast_shapes,
    infer_node_shapes,
    matmul_shape,
    normalize_axis,
    resolve_symbolic_bounds,
)
from ml_switcheroo_ir.shapes.parser import (
    SymbolicExpressionParser,
    SymbolicSolver,
    _parse_sym_str,
    _tokenize_expr,
)
from ml_switcheroo_ir.shapes.polynomial import (
    MonomialType,
    Polynomial,
    _poly_to_symnode,
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
    "ShapeMismatchError",
    "ShapeNodeProtocol",
    "ShapeTracker",
    "ShapeType",
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
    "_parse_sym_str",
    "_poly_to_symnode",
    "_tokenize_expr",
    "broadcast_dimension",
    "broadcast_shapes",
    "infer_node_shapes",
    "matmul_shape",
    "normalize_axis",
    "resolve_symbolic_bounds",
]
