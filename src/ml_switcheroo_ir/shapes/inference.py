"""Broadcasting, matrix multiplication shape inference, axis normalization, and feedback tracking."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, overload

if TYPE_CHECKING:
    from ml_switcheroo_ir.types import TensorSpec

from ml_switcheroo_ir.shapes.ast import (
    DimensionType,
    ShapeType,
    SymConst,
    SymInt,
    SymNode,
)
from ml_switcheroo_ir.shapes.parser import SymbolicSolver
from ml_switcheroo_ir.shapes.tracker import (
    ShapeMismatchError,
    SymbolicConstraintTracker,
)

__all__ = [
    "ShapeTracker",
    "SupportsShapeNodesProtocol",
    "broadcast_dimension",
    "broadcast_shapes",
    "infer_node_shapes",
    "matmul_shape",
    "normalize_axis",
    "resolve_symbolic_bounds",
]


class SupportsShapeNodesProtocol(Protocol):
    """Protocol for graph containers carrying a mapping or sequence of nodes."""

    @property
    def nodes(self) -> Mapping[str, Any] | Iterable[Any]:
        """Return mapping or iterable of graph nodes.

        Returns:
            Mapping[str, Any] | Iterable[Any]: Graph nodes collection.
        """


def broadcast_dimension(
    dim_a: DimensionType,
    dim_b: DimensionType,
    tracker: SymbolicConstraintTracker | None = None,
) -> DimensionType:
    """Resolve broadcasting between two dimensions, raising ShapeMismatchError if incompatible.

    Args:
        dim_a (DimensionType): First dimension size or symbolic expression.
        dim_b (DimensionType): Second dimension size or symbolic expression.
        tracker (SymbolicConstraintTracker | None): Optional constraint tracker for unification.

    Returns:
        DimensionType: Resulting broadcasted dimension.

    Raises:
        ShapeMismatchError: If the dimensions are incompatible for broadcasting.
    """
    if dim_a == dim_b:
        return dim_a

    val_a = dim_a.value if isinstance(dim_a, SymConst) else dim_a
    val_b = dim_b.value if isinstance(dim_b, SymConst) else dim_b

    if val_a == 1 or (isinstance(val_a, str) and val_a == "1"):
        return dim_b
    if val_b == 1 or (isinstance(val_b, str) and val_b == "1"):
        return dim_a

    if SymbolicSolver.is_consistent(val_a, val_b):
        return dim_a

    if isinstance(val_a, int) and isinstance(val_b, int):
        raise ShapeMismatchError(
            f"Incompatible dimensions for broadcasting: {val_a} and {val_b}"
        )

    if tracker is not None:
        return tracker.unify(dim_a, dim_b)

    raise ShapeMismatchError(
        f"Incompatible dimensions for broadcasting: {dim_a} and {dim_b}"
    )


def broadcast_shapes(
    shape_a: ShapeType,
    shape_b: ShapeType,
    tracker: SymbolicConstraintTracker | None = None,
) -> tuple[DimensionType, ...]:
    """Broadcast two multidimensional shapes together following standard right-aligned rules.

    Args:
        shape_a (ShapeType): First shape sequence.
        shape_b (ShapeType): Second shape sequence.
        tracker (SymbolicConstraintTracker | None): Optional constraint tracker for unification.

    Returns:
        tuple[DimensionType, ...]: Broadcasted resulting shape tuple.
    """
    max_len = max(len(shape_a), len(shape_b))
    pad_a = (1,) * (max_len - len(shape_a)) + tuple(shape_a)
    pad_b = (1,) * (max_len - len(shape_b)) + tuple(shape_b)
    return tuple(
        broadcast_dimension(a, b, tracker=tracker) for a, b in zip(pad_a, pad_b)
    )


def matmul_shape(shape_a: ShapeType, shape_b: ShapeType) -> tuple[DimensionType, ...]:
    """Calculate the output shape for matrix multiplication conforming to standard rules.

    Supports 1D x 1D (dot product, scalar output ()), 1D x 2D ((N,)), 2D x 1D ((M,)),
    2D x 2D ((M, N)), and batched (...B, M, K) x (...B, K, N) -> (...B, M, N).

    Args:
        shape_a (ShapeType): Left-hand side shape.
        shape_b (ShapeType): Right-hand side shape.

    Returns:
        tuple[DimensionType, ...]: Output shape after matrix multiplication.

    Raises:
        ShapeMismatchError: If shapes are scalars or inner contracting dimensions mismatch.
    """
    len_a = len(shape_a)
    len_b = len(shape_b)

    if len_a == 0 or len_b == 0:
        raise ShapeMismatchError("Scalars cannot be matrix multiplied.")

    if len_a == 1 and len_b == 1:
        if not SymbolicSolver.is_consistent(shape_a[0], shape_b[0]):
            raise ShapeMismatchError(
                f"Incompatible 1D dot product shapes: {shape_a[0]} and {shape_b[0]}"
            )
        return ()

    if len_a == 1:
        k_a = shape_a[0]
        k_b = shape_b[-2]
        if not SymbolicSolver.is_consistent(k_a, k_b):
            raise ShapeMismatchError(
                f"Incompatible inner dimensions for matmul: {k_a} and {k_b}"
            )
        batch = shape_b[:-2]
        n_dim = shape_b[-1]
        return tuple(batch) + (n_dim,)

    if len_b == 1:
        k_a = shape_a[-1]
        k_b = shape_b[0]
        if not SymbolicSolver.is_consistent(k_a, k_b):
            raise ShapeMismatchError(
                f"Incompatible inner dimensions for matmul: {k_a} and {k_b}"
            )
        batch = shape_a[:-2]
        m_dim = shape_a[-2]
        return tuple(batch) + (m_dim,)

    k_a = shape_a[-1]
    k_b = shape_b[-2]
    if not SymbolicSolver.is_consistent(k_a, k_b):
        raise ShapeMismatchError(
            f"Incompatible inner dimensions for matmul: {k_a} and {k_b}"
        )

    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]
    out_batch = broadcast_shapes(batch_a, batch_b)
    m_dim = shape_a[-2]
    n_dim = shape_b[-1]
    return out_batch + (m_dim, n_dim)


@overload
def normalize_axis(axis: int, rank: int) -> int:
    """Normalize a single axis index to positive 0-indexed integer.

    Args:
        axis (int): Single axis index.
        rank (int): Tensor rank.

    Returns:
        int: Normalized 0-indexed positive axis.
    """


@overload
def normalize_axis(axis: Sequence[int], rank: int) -> tuple[int, ...]:
    """Normalize a sequence of axis indices to positive 0-indexed integers.

    Args:
        axis (Sequence[int]): Sequence of axis indices.
        rank (int): Tensor rank.

    Returns:
        tuple[int, ...]: Tuple of normalized 0-indexed positive axes.
    """


def normalize_axis(
    axis: int | Sequence[int],
    rank: int,
) -> int | tuple[int, ...]:
    """Convert negative axis indices to positive 0-indexed bounds.

    Args:
        axis (int | Sequence[int]): Single axis or sequence of axes.
        rank (int): Rank (total dimension count) of the tensor.

    Returns:
        int | tuple[int, ...]: Normalized positive axis index or indices.

    Raises:
        TypeError: If axis has an unsupported type.
    """

    def _normalize_single(a: int) -> int:
        """Normalize a single axis index.

        Args:
            a (int): Single axis index.

        Returns:
            int: Normalized 0-indexed positive axis.

        Raises:
            ValueError: If axis is out of bounds.
        """
        if a < -rank or a >= rank:
            raise ValueError(f"Axis {a} is out of bounds for tensor of rank {rank}.")
        return a + rank if a < 0 else a

    if isinstance(axis, int) and not isinstance(axis, bool):
        return _normalize_single(axis)
    if isinstance(axis, (tuple, list, Sequence)) and not isinstance(axis, (str, bytes)):
        return tuple(_normalize_single(a) for a in axis)
    raise TypeError(f"Invalid type for axis: {type(axis)}")


class ShapeTracker:
    """Utility class to calculate output shapes given input specs and resolve runtime feedback."""

    @staticmethod
    def infer_elementwise(
        inputs: Sequence[TensorSpec],
    ) -> tuple[DimensionType, ...]:
        """Infer shape for elementwise operations requiring broadcasting.

        Args:
            inputs (Sequence[TensorSpec]): Input tensor specifications.

        Returns:
            tuple[DimensionType, ...]: Broadcasted output shape.
        """
        if not inputs:
            return ()
        out_shape: tuple[DimensionType, ...] = tuple(inputs[0].shape)
        for i in range(1, len(inputs)):
            out_shape = broadcast_shapes(out_shape, inputs[i].shape)
        return out_shape

    @staticmethod
    def infer_matmul(
        input1: TensorSpec,
        input2: TensorSpec,
    ) -> tuple[DimensionType, ...]:
        """Infer shape for matrix multiplication between two specs.

        Args:
            input1 (TensorSpec): First tensor specification.
            input2 (TensorSpec): Second tensor specification.

        Returns:
            tuple[DimensionType, ...]: Resulting matrix multiplication shape.
        """
        return matmul_shape(input1.shape, input2.shape)

    @staticmethod
    def update_from_feedback(
        feedback_telemetry: Mapping[str, Any],
        tracker: SymbolicConstraintTracker | None = None,
    ) -> dict[str, tuple[int, ...]]:
        """Update shape tracking state and record constraints from observed runtime telemetry.

        Args:
            feedback_telemetry (Mapping[str, Any]):
                Observed telemetry mapping node IDs or 1D dimension variable names to concrete shapes.
            tracker (SymbolicConstraintTracker | None): Optional constraint tracker to unify.

        Returns:
            dict[str, tuple[int, ...]]: Resolved concrete shapes per node.
        """
        raw_shapes: dict[str, tuple[int, ...]] = {}
        shapes_val = feedback_telemetry.get("shapes")
        if isinstance(shapes_val, dict):
            for k, val in shapes_val.items():
                if isinstance(val, (list, tuple)):
                    raw_shapes[str(k)] = tuple(int(x) for x in val)
        else:
            for k, v in feedback_telemetry.items():
                if isinstance(v, (list, tuple)):
                    raw_shapes[str(k)] = tuple(int(x) for x in v)

        resolved: dict[str, tuple[int, ...]] = {}
        for node_id, shape_val in raw_shapes.items():
            concrete_tuple = tuple(int(dim) for dim in shape_val)
            resolved[str(node_id)] = concrete_tuple
            if (
                tracker is not None
                and len(concrete_tuple) == 1
                and not str(node_id).isdigit()
            ):
                tracker.record_equality(str(node_id), concrete_tuple[0])

        return resolved

    @staticmethod
    def _record_dynamic_dim_bound(
        dim: Any,
        concrete_val: int,
        resolved_bounds: dict[str, int],
        tracker: SymbolicConstraintTracker | None,
    ) -> None:
        """Record a concrete integer bound for a single symbolic dimension.

        Args:
            dim (Any): Symbolic dimension entry.
            concrete_val (int): Observed concrete dimension size.
            resolved_bounds (dict[str, int]): Output bounds dictionary to update.
            tracker (SymbolicConstraintTracker | None): Optional constraint tracker.
        """
        var_name: str | None = None
        if isinstance(dim, SymInt):
            var_name = dim.name
        elif isinstance(dim, SymNode):
            var_name = str(dim)
        elif isinstance(dim, str) and not dim.isdigit():
            var_name = dim

        if var_name is not None:
            resolved_bounds[var_name] = concrete_val
            if tracker is not None:
                tracker.record_equality(dim, concrete_val)

    @staticmethod
    def resolve_dynamic_bounds(
        graph: SupportsShapeNodesProtocol | Any,
        feedback_telemetry: Mapping[str, Any],
        tracker: SymbolicConstraintTracker | None = None,
        strict_rank: bool = False,
    ) -> dict[str, int]:
        """Resolve dynamic SymInt variables across a graph into static integer bounds using telemetry.

        Args:
            graph (SupportsShapeNodesProtocol | Any): LogicalGraph instance containing nodes with symbolic dimensions.
            feedback_telemetry (Mapping[str, Any]):
                Observed runtime telemetry with concrete shapes.
            tracker (SymbolicConstraintTracker | None): Optional constraint tracker for unification.
            strict_rank (bool): If True, raises ShapeMismatchError when observed telemetry rank
                differs from the node's symbolic shape rank.

        Returns:
            dict[str, int]: Mapping of symbolic dimension names to their resolved static integer bounds.

        Raises:
            ShapeMismatchError: If strict_rank is True and observed shape rank mismatches node rank.
        """
        concrete_shapes = ShapeTracker.update_from_feedback(feedback_telemetry, tracker)
        resolved_bounds: dict[str, int] = {}

        raw_nodes = getattr(graph, "nodes", {})
        node_iterable = (
            raw_nodes.values() if isinstance(raw_nodes, Mapping) else raw_nodes
        )

        for node in node_iterable:
            node_id = str(getattr(node, "id", ""))
            if node_id not in concrete_shapes:
                continue

            observed_shape = concrete_shapes[node_id]
            shape_meta = getattr(node, "shape_metadata", None)
            if shape_meta is None:
                shape_meta = getattr(node, "shape", None)

            if isinstance(shape_meta, (tuple, list)):
                if strict_rank and len(observed_shape) != len(shape_meta):
                    raise ShapeMismatchError(
                        f"Node '{node_id}' observed rank {len(observed_shape)} mismatches "
                        f"symbolic shape rank {len(shape_meta)}."
                    )
                new_shape: list[int] = []
                for idx, dim in enumerate(shape_meta):
                    concrete_val = (
                        observed_shape[idx] if idx < len(observed_shape) else 1
                    )
                    ShapeTracker._record_dynamic_dim_bound(
                        dim, concrete_val, resolved_bounds, tracker
                    )
                    new_shape.append(concrete_val)

                if hasattr(node, "shape_metadata"):
                    node.shape_metadata = tuple(new_shape)
                if hasattr(node, "shape"):
                    node.shape = tuple(new_shape)

        return resolved_bounds


infer_node_shapes = ShapeTracker.infer_elementwise
resolve_symbolic_bounds = ShapeTracker.resolve_dynamic_bounds
