"""Broadcasting, matrix multiplication shape inference, axis normalization, and feedback tracking."""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable, Mapping, Sequence

import typing_extensions

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir.types import AttributeValue, TensorSpec

from ml_switcheroo_ir.shapes.ast import (
    DimensionType,
    ShapeType,
    SymBinaryOp,
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
    "SHAPE_TRANSFER_REGISTRY",
    "ShapeNodeProtocol",
    "ShapeTracker",
    "ShapeTransferContext",
    "SupportsDirectShapeProtocol",
    "SupportsShapeMetadataProtocol",
    "SupportsShapeNodesProtocol",
    "TelemetryValueType",
    "broadcast_dimension",
    "broadcast_shapes",
    "infer_node_shapes",
    "infer_op_output_shape",
    "matmul_shape",
    "normalize_axis",
    "resolve_symbolic_bounds",
]

TelemetryValueType: typing_extensions.TypeAlias = typing.Union[
    Sequence[int],
    Mapping[str, typing.Union[Sequence[int], int, str, float, bool, None]],
    int,
    str,
    float,
    bool,
    None,
]


class ShapeNodeProtocol(typing.Protocol):
    """Protocol for graph nodes carrying an ID and optional shape metadata."""

    @property
    def id(self) -> str:
        """Return node identifier.

        Returns:
            str: Node identifier.
        """


@typing.runtime_checkable
class SupportsShapeMetadataProtocol(ShapeNodeProtocol, typing.Protocol):
    """Runtime-checkable protocol for nodes carrying a mutable shape_metadata attribute.

    Attributes:
        shape_metadata (tuple[DimensionType, ...] | None): Optional tuple of tensor dimensions.
    """

    shape_metadata: tuple[DimensionType, ...] | None


@typing.runtime_checkable
class SupportsDirectShapeProtocol(ShapeNodeProtocol, typing.Protocol):
    """Runtime-checkable protocol for nodes carrying a mutable shape attribute.

    Attributes:
        shape (tuple[DimensionType, ...] | None): Optional tuple of tensor dimensions.
    """

    shape: tuple[DimensionType, ...] | None


class SupportsShapeNodesProtocol(typing.Protocol):
    """Protocol for graph containers carrying a mapping or iterable of nodes."""

    @property
    def nodes(self) -> Mapping[str, ShapeNodeProtocol] | Iterable[ShapeNodeProtocol]:
        """Return mapping or iterable of graph nodes.

        Returns:
            Mapping[str, ShapeNodeProtocol] | Iterable[ShapeNodeProtocol]: Graph nodes collection.
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
        TypeError: If either dimension is a boolean value.
        ShapeMismatchError: If the dimensions are incompatible for broadcasting.
    """
    if isinstance(dim_a, bool) or isinstance(dim_b, bool):
        raise TypeError("Boolean values are not valid tensor dimensions.")

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
        raise ShapeMismatchError(f"Incompatible dimensions for broadcasting: {val_a} and {val_b}")

    if tracker is not None:
        return tracker.unify(dim_a, dim_b)

    raise ShapeMismatchError(f"Incompatible dimensions for broadcasting: {dim_a} and {dim_b}")


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

    Raises:
        TypeError: If any dimension in ``shape_a`` or ``shape_b`` is a boolean value.
    """
    if any(isinstance(d, bool) for d in shape_a) or any(isinstance(d, bool) for d in shape_b):
        raise TypeError("Boolean values are not valid tensor dimensions.")
    max_len = max(len(shape_a), len(shape_b))
    pad_a = (1,) * (max_len - len(shape_a)) + tuple(shape_a)
    pad_b = (1,) * (max_len - len(shape_b)) + tuple(shape_b)
    return tuple(broadcast_dimension(a, b, tracker=tracker) for a, b in zip(pad_a, pad_b))


def _check_contracting_dims(k_a: DimensionType, k_b: DimensionType, label: str) -> None:
    """Verify that contracting dimensions ``k_a`` and ``k_b`` are symbolically consistent.

    Args:
        k_a (DimensionType): Left contracting dimension.
        k_b (DimensionType): Right contracting dimension.
        label (str): Context description for error reporting.

    Raises:
        ShapeMismatchError: If ``k_a`` and ``k_b`` are incompatible.
    """
    if not SymbolicSolver.is_consistent(k_a, k_b):
        raise ShapeMismatchError(f"Incompatible {label}: {k_a} and {k_b}")


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
        TypeError: If any dimension in ``shape_a`` or ``shape_b`` is a boolean value.
        ShapeMismatchError: If shapes are scalars or inner contracting dimensions mismatch.
    """
    if any(isinstance(d, bool) for d in shape_a) or any(isinstance(d, bool) for d in shape_b):
        raise TypeError("Boolean values are not valid tensor dimensions.")

    len_a = len(shape_a)
    len_b = len(shape_b)

    if len_a == 0 or len_b == 0:
        raise ShapeMismatchError("Scalars cannot be matrix multiplied.")

    if len_a == 1 and len_b == 1:
        _check_contracting_dims(shape_a[0], shape_b[0], "1D dot product shapes")
        return ()

    if len_a == 1:
        _check_contracting_dims(shape_a[0], shape_b[-2], "inner dimensions for matmul")
        return tuple(shape_b[:-2]) + (shape_b[-1],)

    if len_b == 1:
        _check_contracting_dims(shape_a[-1], shape_b[0], "inner dimensions for matmul")
        return tuple(shape_a[:-2]) + (shape_a[-2],)

    _check_contracting_dims(shape_a[-1], shape_b[-2], "inner dimensions for matmul")
    out_batch = broadcast_shapes(shape_a[:-2], shape_b[:-2])
    return out_batch + (shape_a[-2], shape_b[-1])


@typing.overload
def normalize_axis(axis: int, rank: int) -> int:
    """Normalize a single axis index to positive 0-indexed integer.

    Args:
        axis (int): Single axis index.
        rank (int): Tensor rank.

    Returns:
        int: Normalized 0-indexed positive axis.
    """


@typing.overload
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
        TypeError: If ``rank``, ``axis``, or any axis element is a boolean or non-integer type.
    """
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"Invalid type for rank: {type(rank)}")

    def _normalize_single(a: int) -> int:
        """Normalize a single axis index.

        Args:
            a (int): Single axis index.

        Returns:
            int: Normalized 0-indexed positive axis.

        Raises:
            TypeError: If ``a`` is a boolean or not an ``int``.
            ValueError: If axis is out of bounds.
        """
        if isinstance(a, bool) or not isinstance(a, int):
            raise TypeError(f"Invalid type for axis element: {type(a)}")
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
        feedback_telemetry: Mapping[str, TelemetryValueType],
        tracker: SymbolicConstraintTracker | None = None,
    ) -> dict[str, tuple[int, ...]]:
        """Update shape tracking state and record constraints from observed runtime telemetry.

        Args:
            feedback_telemetry (Mapping[str, TelemetryValueType]):
                Observed telemetry mapping node IDs or 1D dimension variable names to concrete shapes.
            tracker (SymbolicConstraintTracker | None): Optional constraint tracker to unify.

        Returns:
            dict[str, tuple[int, ...]]: Resolved concrete shapes per node.
        """
        raw_shapes: dict[str, tuple[int, ...]] = {}
        shapes_val = feedback_telemetry.get("shapes")
        if isinstance(shapes_val, Mapping):
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
            if tracker is not None and len(concrete_tuple) == 1 and not str(node_id).isdigit():
                tracker.record_equality(str(node_id), concrete_tuple[0])

        return resolved

    @staticmethod
    def _record_dynamic_dim_bound(
        dim: DimensionType,
        concrete_val: int,
        resolved_bounds: dict[str, int],
        tracker: SymbolicConstraintTracker | None,
    ) -> None:
        """Record a concrete integer bound for a single symbolic dimension.

        Args:
            dim (DimensionType): Symbolic dimension entry.
            concrete_val (int): Observed concrete dimension size.
            resolved_bounds (dict[str, int]): Output bounds dictionary to update.
            tracker (SymbolicConstraintTracker | None): Optional constraint tracker.
        """
        var_name: str | None = None
        if isinstance(dim, SymInt):
            var_name = dim.name
        elif isinstance(dim, SymNode):
            var_name = str(dim)
        elif isinstance(dim, str) and not dim.lstrip("-").isdigit():
            var_name = dim

        if var_name is not None:
            if var_name.isidentifier():
                resolved_bounds[var_name] = concrete_val
            active_tracker = tracker if tracker is not None else SymbolicConstraintTracker()
            active_tracker.record_equality(dim, concrete_val)
            resolved_bounds.update(active_tracker.solve())

    @staticmethod
    def _get_node_shape_meta(
        node: ShapeNodeProtocol,
    ) -> tuple[DimensionType, ...] | None:
        """Extract shape dimension tuple from a node supporting shape_metadata or shape.

        Args:
            node (ShapeNodeProtocol): Graph node instance.

        Returns:
            tuple[DimensionType, ...] | None: Tuple of dimensions if available, else None.
        """
        if isinstance(node, SupportsShapeMetadataProtocol):
            return node.shape_metadata
        if isinstance(node, SupportsDirectShapeProtocol):
            return node.shape
        return None

    @staticmethod
    def _set_node_shape_meta(
        node: ShapeNodeProtocol,
        resolved_tuple: tuple[DimensionType, ...],
    ) -> None:
        """Assign resolved concrete shape tuple onto a node supporting shape_metadata or shape.

        Args:
            node (ShapeNodeProtocol): Graph node instance.
            resolved_tuple (tuple[DimensionType, ...]): Resolved dimension tuple.
        """
        if isinstance(node, SupportsShapeMetadataProtocol):
            node.shape_metadata = resolved_tuple
        elif isinstance(node, SupportsDirectShapeProtocol):
            node.shape = resolved_tuple

    @staticmethod
    def _resolve_single_node_bounds(
        node: ShapeNodeProtocol,
        observed_shape: tuple[int, ...],
        resolved_bounds: dict[str, int],
        tracker: SymbolicConstraintTracker | None,
        strict_rank: bool,
        mutate: bool,
    ) -> None:
        """Resolve dynamic dimension bounds for a single observed graph node.

        Args:
            node (ShapeNodeProtocol): Graph node with optional symbolic shape metadata.
            observed_shape (tuple[int, ...]): Concrete observed telemetry dimensions.
            resolved_bounds (dict[str, int]): Output dictionary accumulating solved bounds.
            tracker (SymbolicConstraintTracker | None): Optional shared constraint tracker.
            strict_rank (bool): Whether rank mismatches raise ``ShapeMismatchError``.
            mutate (bool): Whether to mutate ``node`` shape metadata in place.

        Raises:
            ShapeMismatchError: If ``strict_rank`` is True and ranks differ.
        """
        shape_meta = ShapeTracker._get_node_shape_meta(node)
        if not isinstance(shape_meta, (tuple, list)):
            return
        if strict_rank and len(observed_shape) != len(shape_meta):
            raise ShapeMismatchError(
                f"Node '{node.id}' observed rank {len(observed_shape)} "
                f"mismatches symbolic shape rank {len(shape_meta)}."
            )
        new_shape: list[DimensionType] = []
        for idx, dim in enumerate(shape_meta):
            if idx < len(observed_shape):
                concrete_val = observed_shape[idx]
                ShapeTracker._record_dynamic_dim_bound(dim, concrete_val, resolved_bounds, tracker)
                new_shape.append(concrete_val)
            else:
                new_shape.append(dim)

        if mutate:
            ShapeTracker._set_node_shape_meta(node, tuple(new_shape))

    @staticmethod
    def resolve_dynamic_bounds(
        graph: SupportsShapeNodesProtocol,
        feedback_telemetry: Mapping[str, TelemetryValueType],
        tracker: SymbolicConstraintTracker | None = None,
        strict_rank: bool = False,
        mutate: bool = False,
    ) -> dict[str, int]:
        """Resolve dynamic symbolic variables across a graph into static integer bounds using telemetry.

        Pure by default (``mutate=False``): computes and returns solved bounds without mutating
        graph nodes unless ``mutate=True`` is explicitly passed.

        Args:
            graph (SupportsShapeNodesProtocol): LogicalGraph instance containing nodes with symbolic dimensions.
            feedback_telemetry (Mapping[str, TelemetryValueType]):
                Observed runtime telemetry with concrete shapes.
            tracker (SymbolicConstraintTracker | None): Optional constraint tracker for unification.
            strict_rank (bool): If True, raises ShapeMismatchError when observed telemetry rank
                differs from the node's symbolic shape rank.
            mutate (bool): If True, mutates ``shape_metadata`` / ``shape`` on observed nodes in place.

        Returns:
            dict[str, int]: Mapping of symbolic dimension names to their resolved static integer bounds.
        """
        concrete_shapes = ShapeTracker.update_from_feedback(feedback_telemetry, tracker)
        resolved_bounds: dict[str, int] = {}

        raw_nodes = graph.nodes
        node_iterable: Iterable[ShapeNodeProtocol] = raw_nodes.values() if isinstance(raw_nodes, Mapping) else raw_nodes

        for node in node_iterable:
            node_id = str(node.id)
            if node_id in concrete_shapes:
                ShapeTracker._resolve_single_node_bounds(
                    node,
                    concrete_shapes[node_id],
                    resolved_bounds,
                    tracker,
                    strict_rank,
                    mutate,
                )

        return resolved_bounds


infer_node_shapes = ShapeTracker.infer_elementwise
resolve_symbolic_bounds = ShapeTracker.resolve_dynamic_bounds


class ShapeTransferContext(typing.NamedTuple):
    """Structured evaluation context passed to unified operator shape transfer rules.

    Attributes:
        node_id (str): Graph node identifier for diagnostic error messages.
        input_shapes (tuple[tuple[DimensionType, ...] | None, ...]): Resolved upstream input shapes.
        attributes (Mapping[str, AttributeValue]): Resolved operator attributes (including folded operand values).
        num_outputs (int): Declared output port count on the node.
        tracker (SymbolicConstraintTracker): Symbolic constraint unification tracker.
    """

    node_id: str
    input_shapes: tuple[tuple[DimensionType, ...] | None, ...]
    attributes: Mapping[str, AttributeValue]
    num_outputs: int
    tracker: SymbolicConstraintTracker


ShapeTransferFn: typing_extensions.TypeAlias = Callable[
    [ShapeTransferContext],
    typing.Optional[tuple[DimensionType, ...]],
]


def _transfer_unary(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer output shape for unary elementwise operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: Propagated first input shape or None.
    """
    return ctx.input_shapes[0] if ctx.input_shapes else None


def _transfer_binary(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer broadcasted output shape for binary elementwise operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: Broadcasted shape or None.
    """
    if len(ctx.input_shapes) >= 2 and ctx.input_shapes[0] is not None and ctx.input_shapes[1] is not None:
        return broadcast_shapes(ctx.input_shapes[0], ctx.input_shapes[1], tracker=ctx.tracker)
    return None


def _transfer_matmul(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer output shape for matrix multiplication operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: MatMul output shape or None.
    """
    if len(ctx.input_shapes) >= 2 and ctx.input_shapes[0] is not None and ctx.input_shapes[1] is not None:
        return matmul_shape(ctx.input_shapes[0], ctx.input_shapes[1])
    return None


def _transfer_gemm(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer output shape for 2D General Matrix Multiply (Gemm) operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: Gemm output shape or None.

    Raises:
        ValueError: If inner contraction dimensions mismatch.
    """
    if len(ctx.input_shapes) < 2:
        return None
    s_a, s_b = ctx.input_shapes[0], ctx.input_shapes[1]
    if s_a is None or s_b is None or len(s_a) != 2 or len(s_b) != 2:
        return None
    trans_a = bool(ctx.attributes.get("transA", 0))
    trans_b = bool(ctx.attributes.get("transB", 0))
    m = s_a[1] if trans_a else s_a[0]
    k_a = s_a[0] if trans_a else s_a[1]
    k_b = s_b[1] if trans_b else s_b[0]
    n = s_b[0] if trans_b else s_b[1]
    if not SymbolicSolver.is_consistent(k_a, k_b):
        raise ValueError(f"Gemm inner contraction dimension mismatch: {k_a} vs {k_b} in node '{ctx.node_id}'.")
    out_shape: tuple[DimensionType, ...] = (m, n)
    if len(ctx.input_shapes) > 2 and ctx.input_shapes[2] is not None:
        out_shape = broadcast_shapes(out_shape, ctx.input_shapes[2], tracker=ctx.tracker)
    return out_shape


def _extract_int_axes_attr(axes: AttributeValue) -> list[int]:
    """Coerce an integer or sequence of integer/string axes into ``list[int]``.

    Args:
        axes (AttributeValue): Raw axis or axes attribute value.

    Returns:
        list[int]: Parsed integer axes.
    """
    raw_list: Sequence[AttributeValue] = (
        [axes]
        if isinstance(axes, int) and not isinstance(axes, bool)
        else (axes if isinstance(axes, (list, tuple)) else [])
    )
    return [int(a) for a in raw_list if isinstance(a, (int, str)) and not isinstance(a, bool)]


def _transfer_squeeze(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer output shape for Squeeze operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: Squeezed shape or None.
    """
    if not ctx.input_shapes or ctx.input_shapes[0] is None:
        return None
    s = ctx.input_shapes[0]
    axes = ctx.attributes.get("axes")
    if axes is None:
        axes = ctx.attributes.get("axis")
    if axes is not None:
        norm_axes = set(normalize_axis(_extract_int_axes_attr(axes), len(s)))
        return tuple(d for i, d in enumerate(s) if i not in norm_axes)
    return tuple(d for d in s if d != 1 and str(d) != "1")


def _transfer_unsqueeze(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer output shape for Unsqueeze operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: Unsqueezed shape or None.
    """
    if not ctx.input_shapes or ctx.input_shapes[0] is None:
        return None
    s = list(ctx.input_shapes[0])
    axes = ctx.attributes.get("axes")
    if axes is None:
        axes = ctx.attributes.get("axis")
    if axes is not None:
        raw_axes = _extract_int_axes_attr(axes)
        target_rank = len(s) + len(raw_axes)
        for ax in sorted(normalize_axis(a, target_rank) for a in raw_axes):
            s.insert(ax, 1)
    return tuple(s)


def _product_dimensions(dims: Sequence[DimensionType]) -> DimensionType:
    """Compute cumulative symbolic or integer product across a sequence of dimensions.

    Args:
        dims (Sequence[DimensionType]): Dimension sequence.

    Returns:
        DimensionType: Product dimension.
    """
    total: DimensionType = 1
    for d in dims:
        if isinstance(d, int) and not isinstance(d, bool) and isinstance(total, int):
            total *= d
        else:
            total = SymBinaryOp("*", SymNode.to_node(total), SymNode.to_node(d)).simplify()
    return total


def _transfer_flatten(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer 2D output shape for Flatten operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: Flattened 2D shape or None.
    """
    if not ctx.input_shapes or ctx.input_shapes[0] is None:
        return None
    s = ctx.input_shapes[0]
    axis_val = ctx.attributes.get("axis", 1)
    axis = int(axis_val) if isinstance(axis_val, (int, str)) and not isinstance(axis_val, bool) else 1
    norm_axis = normalize_axis(axis, len(s))
    return (_product_dimensions(s[:norm_axis]), _product_dimensions(s[norm_axis:]))


def _transfer_concat(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer output shape for Concat operations joining tensors along an axis.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: Concatenated shape or None.

    Raises:
        ValueError: If input tensor ranks mismatch.
    """
    if not ctx.input_shapes or any(sh is None for sh in ctx.input_shapes):
        return None
    shapes = [sh for sh in ctx.input_shapes if sh is not None]
    rank = len(shapes[0])
    axis_val = ctx.attributes.get("axis", 0)
    axis = int(axis_val) if isinstance(axis_val, (int, str)) and not isinstance(axis_val, bool) else 0
    norm_axis = normalize_axis(axis, rank)

    concat_dim: DimensionType = 0
    for s in shapes:
        if len(s) != rank:
            raise ValueError(f"Concat rank mismatch in node '{ctx.node_id}': expected rank {rank}, got {len(s)}.")
        d = s[norm_axis]
        if isinstance(d, int) and not isinstance(d, bool) and isinstance(concat_dim, int):
            concat_dim += d
        else:
            concat_dim = SymBinaryOp("+", SymNode.to_node(concat_dim), SymNode.to_node(d)).simplify()
    out_s = list(shapes[0])
    out_s[norm_axis] = concat_dim
    return tuple(out_s)


def _transfer_split(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer first-partition output shape for Split operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: First split partition shape or None.

    Raises:
        ValueError: If a concrete dimension is not evenly divisible by ``num_outputs``.
    """
    if not ctx.input_shapes or ctx.input_shapes[0] is None:
        return None
    s = ctx.input_shapes[0]
    axis_val = ctx.attributes.get("axis", 0)
    axis = int(axis_val) if isinstance(axis_val, (int, str)) and not isinstance(axis_val, bool) else 0
    norm_axis = normalize_axis(axis, len(s))
    split = ctx.attributes.get("split")
    num_out_raw = ctx.attributes.get("num_outputs") or ctx.num_outputs or 2
    num_outputs = int(num_out_raw) if isinstance(num_out_raw, (int, str)) and not isinstance(num_out_raw, bool) else 2

    first_split: DimensionType
    if isinstance(split, (list, tuple)) and len(split) > 0:
        raw_first = split[0]
        first_split = (
            raw_first
            if isinstance(raw_first, (int, str, SymNode, SymInt)) and not isinstance(raw_first, bool)
            else int(str(raw_first))
        )
    else:
        dim_val = s[norm_axis]
        if isinstance(dim_val, int) and not isinstance(dim_val, bool):
            if dim_val % num_outputs != 0:
                raise ValueError(
                    f"Split dimension {dim_val} along axis {norm_axis} in node '{ctx.node_id}' "
                    f"is not evenly divisible by num_outputs={num_outputs}."
                )
            first_split = dim_val // num_outputs
        else:
            first_split = SymBinaryOp("//", SymNode.to_node(dim_val), SymConst(num_outputs)).simplify()
    out_s = list(s)
    out_s[norm_axis] = first_split
    return tuple(out_s)


def _sliced_axis_length(dim_len: DimensionType, st: int, en: int, sp: int) -> DimensionType:
    """Compute the sliced length along one axis for integer or symbolic dimension.

    Args:
        dim_len (DimensionType): Input dimension size.
        st (int): Slice start index.
        en (int): Slice end index.
        sp (int): Slice step stride.

    Returns:
        DimensionType: Resulting sliced dimension size.
    """
    if sp == 0:
        return 0
    if isinstance(dim_len, int) and not isinstance(dim_len, bool):
        st_norm = max(0, min(dim_len, st + dim_len if st < 0 else st))
        en_norm = max(0, min(dim_len, en + dim_len if en < 0 else en))
        if sp > 0:
            return max(0, (en_norm - st_norm + sp - 1) // sp)
        abs_sp = -sp
        return max(0, (st_norm - en_norm + abs_sp - 1) // abs_sp)

    sym_dim = SymNode.to_node(dim_len)
    st_sym = SymBinaryOp("+", sym_dim, SymConst(st)).simplify() if st < 0 else SymConst(st)
    en_sym = SymBinaryOp("+", sym_dim, SymConst(en)).simplify() if en < 0 else SymConst(en)
    abs_sp = sp if sp > 0 else -sp
    span = SymBinaryOp("-", en_sym, st_sym) if sp > 0 else SymBinaryOp("-", st_sym, en_sym)
    diff_node = SymBinaryOp("+", span, SymConst(abs_sp - 1)).simplify()
    res_sym = SymBinaryOp("//", diff_node, SymConst(abs_sp)).simplify()
    return max(0, res_sym.value) if isinstance(res_sym, SymConst) else res_sym


def _transfer_slice(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer output shape for Slice operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: Sliced shape or None.
    """
    if not ctx.input_shapes or ctx.input_shapes[0] is None:
        return None
    starts = ctx.attributes.get("starts")
    ends = ctx.attributes.get("ends")
    if not isinstance(starts, (list, tuple)) or not isinstance(ends, (list, tuple)):
        return None
    s = list(ctx.input_shapes[0])
    axes = ctx.attributes.get("axes")
    steps = ctx.attributes.get("steps")
    axes_list = list(axes) if isinstance(axes, (list, tuple)) else list(range(len(starts)))
    steps_list = list(steps) if isinstance(steps, (list, tuple)) else [1] * len(starts)
    for st, en, ax, sp in zip(starts, ends, axes_list, steps_list):
        ax_int = int(ax) if isinstance(ax, (int, str)) and not isinstance(ax, bool) else 0
        norm_ax = normalize_axis(ax_int, len(s))
        if (
            isinstance(st, int)
            and not isinstance(st, bool)
            and isinstance(en, int)
            and not isinstance(en, bool)
            and isinstance(sp, int)
            and not isinstance(sp, bool)
        ):
            s[norm_ax] = _sliced_axis_length(s[norm_ax], st, en, sp)
    return tuple(s)


def _transfer_shape(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer 1D ``(rank,)`` output shape for Shape operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: ``(len(input_shape),)`` or None.
    """
    if ctx.input_shapes and ctx.input_shapes[0] is not None:
        return (len(ctx.input_shapes[0]),)
    return None


def _concrete_numel(shape: Sequence[DimensionType]) -> tuple[int, bool]:
    """Compute product of concrete positive integer dimensions and whether all dims are concrete.

    Args:
        shape (Sequence[DimensionType]): Input shape tuple.

    Returns:
        tuple[int, bool]: ``(product_numel, all_concrete)``.
    """
    numel = 1
    for d in shape:
        if isinstance(d, int) and not isinstance(d, bool) and d > 0:
            numel *= d
        else:
            return numel, False
    return numel, True


def _resolve_reshape_dim_entry(
    node_id: str,
    idx: int,
    d: AttributeValue,
    inp_shape: Sequence[DimensionType],
    allowzero: bool,
) -> tuple[DimensionType, int, bool]:
    """Resolve one target dimension in a Reshape operation.

    Args:
        node_id (str): Node identifier for error messages.
        idx (int): Target dimension index.
        d (AttributeValue): Raw target dimension value.
        inp_shape (Sequence[DimensionType]): Input tensor shape.
        allowzero (bool): Whether 0 denotes literal zero rather than input copy.

    Returns:
        tuple[DimensionType, int, bool]: ``(resolved_dim, factor, is_concrete)``.

    Raises:
        ValueError: If 0-copy index is out of bounds or integer dim is non-positive.
    """
    if d == 0 and not isinstance(d, bool) and not allowzero:
        if idx >= len(inp_shape):
            raise ValueError(
                f"Reshape in node '{node_id}' cannot copy dimension 0 at index {idx} "
                f"from input of rank {len(inp_shape)}."
            )
        copied_d = inp_shape[idx]
        if isinstance(copied_d, int) and not isinstance(copied_d, bool):
            return copied_d, copied_d, True
        return copied_d, 1, False
    if isinstance(d, int) and not isinstance(d, bool):
        if d <= 0:
            raise ValueError(f"Reshape in node '{node_id}' has invalid non-positive dimension {d}.")
        return d, d, True
    return (d if isinstance(d, (SymNode, SymInt)) else str(d)), 1, False


def _transfer_reshape(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer output shape and verify element-count invariants for Reshape operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: Resolved target shape or None.

    Raises:
        ValueError: On multiple -1 dims, out-of-bounds 0-copy, or mismatched static element counts.
    """
    if not ctx.input_shapes or ctx.input_shapes[0] is None:
        return None
    target_shape_attr = ctx.attributes.get("shape")
    if target_shape_attr is None:
        target_shape_attr = ctx.attributes.get("new_shape")
    if not isinstance(target_shape_attr, (list, tuple)):
        return None

    inp_shape = ctx.input_shapes[0]
    in_numel, all_concrete = _concrete_numel(inp_shape)
    allowzero = bool(ctx.attributes.get("allowzero", 0))
    resolved_dims: list[DimensionType] = []
    neg_idx = -1
    target_known_numel = 1

    for idx, d in enumerate(target_shape_attr):
        if d == -1 and not isinstance(d, bool):
            if neg_idx != -1:
                raise ValueError(f"Reshape in node '{ctx.node_id}' contains multiple -1 dimensions.")
            neg_idx = idx
            resolved_dims.append(-1)
            continue
        res_d, factor, is_conc = _resolve_reshape_dim_entry(ctx.node_id, idx, d, inp_shape, allowzero)
        resolved_dims.append(res_d)
        target_known_numel *= factor
        all_concrete = all_concrete and is_conc

    if all_concrete and neg_idx != -1:
        if in_numel % target_known_numel != 0:
            raise ValueError(
                f"Static shape mismatch in Reshape '{ctx.node_id}': input elements ({in_numel}) "
                f"not divisible by target dimensions ({target_known_numel})."
            )
        resolved_dims[neg_idx] = in_numel // target_known_numel
    elif all_concrete and neg_idx == -1 and in_numel != target_known_numel:
        raise ValueError(
            f"Static shape mismatch in Reshape '{ctx.node_id}': input elements ({in_numel}) "
            f"mismatches target elements ({target_known_numel})."
        )
    return tuple(resolved_dims)


def _transfer_transpose(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer output shape for Transpose operations according to ``perm`` attribute.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: Permuted shape or None.

    Raises:
        ValueError: If ``perm`` has matching length with duplicate or out-of-range axes.
    """
    if not ctx.input_shapes or ctx.input_shapes[0] is None:
        return None
    inp_shape = ctx.input_shapes[0]
    perm = ctx.attributes.get("perm")
    if isinstance(perm, (list, tuple)) and len(perm) == len(inp_shape):
        rank = len(inp_shape)
        valid_perm = [p for p in perm if isinstance(p, int) and not isinstance(p, bool) and 0 <= p < rank]
        if len(valid_perm) != rank or len(set(valid_perm)) != rank:
            raise ValueError(f"Invalid Transpose permutation {list(perm)} for rank {rank} in node '{ctx.node_id}'.")
        return tuple(inp_shape[p] for p in valid_perm)
    return None


def _transfer_broadcast_in_dim(ctx: ShapeTransferContext) -> tuple[DimensionType, ...] | None:
    """Infer output shape for BroadcastInDim operations.

    Args:
        ctx (ShapeTransferContext): Shape transfer context.

    Returns:
        tuple[DimensionType, ...] | None: Target broadcasted shape or None.
    """
    if not ctx.input_shapes:
        return None
    out_shape = ctx.attributes.get("output_shape")
    if out_shape is None:
        out_shape = ctx.attributes.get("shape")
    if isinstance(out_shape, (list, tuple)):
        return tuple(
            d if isinstance(d, (int, str, SymNode, SymInt)) and not isinstance(d, bool) else str(d) for d in out_shape
        )
    return None


_UNARY_TRANSFER_OPS: tuple[str, ...] = (
    "Abs",
    "Neg",
    "Exp",
    "Log",
    "Sqrt",
    "Relu",
    "Sigmoid",
    "Tanh",
    "Sin",
    "Cos",
    "Floor",
    "Ceil",
    "Round",
    "Identity",
    "Cast",
    "relu",
    "sigmoid",
    "tanh",
    "cast",
)

_BINARY_TRANSFER_OPS: tuple[str, ...] = (
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Mod",
    "Equal",
    "Greater",
    "Less",
    "Max",
    "Min",
    "And",
    "Or",
    "add",
    "sub",
    "mul",
    "div",
    "arith.addi",
    "arith.addf",
    "arith.subi",
    "arith.subf",
    "arith.muli",
    "arith.mulf",
    "arith.divf",
)

SHAPE_TRANSFER_REGISTRY: dict[str, ShapeTransferFn] = {
    **{op: _transfer_unary for op in _UNARY_TRANSFER_OPS},
    **{op: _transfer_binary for op in _BINARY_TRANSFER_OPS},
    "MatMul": _transfer_matmul,
    "BatchMatMul": _transfer_matmul,
    "matmul": _transfer_matmul,
    "batch_matmul": _transfer_matmul,
    "Gemm": _transfer_gemm,
    "gemm": _transfer_gemm,
    "Squeeze": _transfer_squeeze,
    "squeeze": _transfer_squeeze,
    "Unsqueeze": _transfer_unsqueeze,
    "unsqueeze": _transfer_unsqueeze,
    "Flatten": _transfer_flatten,
    "flatten": _transfer_flatten,
    "Concat": _transfer_concat,
    "concat": _transfer_concat,
    "Split": _transfer_split,
    "split": _transfer_split,
    "Slice": _transfer_slice,
    "slice": _transfer_slice,
    "Shape": _transfer_shape,
    "Reshape": _transfer_reshape,
    "reshape": _transfer_reshape,
    "Transpose": _transfer_transpose,
    "transpose": _transfer_transpose,
    "BroadcastInDim": _transfer_broadcast_in_dim,
    "broadcast_in_dim": _transfer_broadcast_in_dim,
}


def infer_op_output_shape(
    op_type: str,
    input_shapes: Sequence[tuple[DimensionType, ...] | None],
    attributes: Mapping[str, AttributeValue] | None = None,
    *,
    node_id: str = "",
    num_outputs: int = 1,
    tracker: SymbolicConstraintTracker | None = None,
) -> tuple[DimensionType, ...] | None:
    """Infer output tensor shape for an operator using the unified ``SHAPE_TRANSFER_REGISTRY``.

    Args:
        op_type (str): Operator mnemonic (e.g. ``'Add'``, ``'MatMul'``, ``'Reshape'``).
        input_shapes (Sequence[tuple[DimensionType, ...] | None]): Upstream input tensor shapes.
        attributes (Mapping[str, AttributeValue] | None): Operator attributes mapping.
        node_id (str): Optional node identifier for error messages.
        num_outputs (int): Declared output port count.
        tracker (SymbolicConstraintTracker | None): Optional symbolic constraint tracker.

    Returns:
        tuple[DimensionType, ...] | None: Inferred output shape tuple or None.
    """
    rule = SHAPE_TRANSFER_REGISTRY.get(op_type)
    if rule is None:
        return None
    ctx = ShapeTransferContext(
        node_id=node_id,
        input_shapes=tuple(input_shapes),
        attributes=attributes if attributes is not None else {},
        num_outputs=num_outputs,
        tracker=tracker if tracker is not None else SymbolicConstraintTracker(),
    )
    return rule(ctx)
