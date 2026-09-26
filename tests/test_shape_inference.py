"""Tests for broadcasting, matmul shape inference, axis normalization, and constraint tracking."""

from __future__ import annotations

import pytest

from ml_switcheroo_ir.shapes import (
    ShapeMismatchError,
    ShapeTracker,
    SymbolicConstraintTracker,
    SymConst,
    SymInt,
    SymVar,
    broadcast_dimension,
    broadcast_shapes,
    matmul_shape,
    normalize_axis,
)
from ml_switcheroo_ir.types import DType, TensorShape, TensorSpec


def test_broadcast_dimension() -> None:
    """Test broadcast_dimension with concrete integers, 1s, and symbolic values."""
    assert broadcast_dimension(1, 10) == 10
    assert broadcast_dimension(10, 1) == 10
    assert broadcast_dimension(8, 8) == 8

    # 1 as string or SymConst
    assert broadcast_dimension("1", 16) == 16
    assert broadcast_dimension(16, "1") == 16
    assert broadcast_dimension(SymConst(1), 32) == 32
    assert broadcast_dimension(32, SymConst(1)) == 32

    # Incompatible concrete dimensions
    with pytest.raises(ShapeMismatchError, match="Incompatible dimensions"):
        broadcast_dimension(4, 5)

    # Symbolic dimensions
    assert broadcast_dimension("batch", "batch") == "batch"
    assert broadcast_dimension("batch", 1) == "batch"
    assert broadcast_dimension(1, "batch") == "batch"

    si = SymInt("batch")
    assert broadcast_dimension(si, 1) == si
    assert broadcast_dimension(1, si) == si
    assert broadcast_dimension(si, si) == si

    # Tracker unification
    tracker = SymbolicConstraintTracker()
    assert broadcast_dimension("dim_a", "dim_b", tracker=tracker) == "dim_a"
    assert tracker.is_consistent()

    # Contradiction with tracker
    tracker.add_constraint("dim_a", 128)
    tracker.add_constraint("dim_c", 256)
    with pytest.raises(ShapeMismatchError, match="Contradictory"):
        broadcast_dimension("dim_a", "dim_c", tracker=tracker)


def test_broadcast_shapes() -> None:
    """Test broadcast_shapes across multidimensional static and dynamic shapes."""
    # Right-aligned broadcasting
    s1 = (2, 1, 8)
    s2 = (4, 8)
    assert broadcast_shapes(s1, s2) == (2, 4, 8)

    # Incompatible shapes
    with pytest.raises(ShapeMismatchError):
        broadcast_shapes((3, 4), (5, 4))

    # Symbolic broadcasting
    si_b = SymInt("batch")
    s_sym1 = (si_b, 1, 64)
    s_sym2 = (128, 64)
    res = broadcast_shapes(s_sym1, s_sym2)
    assert res == (si_b, 128, 64)

    # TensorShape.broadcast_with
    ts1 = TensorShape((2, 1, 8))
    ts2 = TensorShape((4, 8))
    assert ts1.broadcast_with(ts2) == TensorShape((2, 4, 8))
    assert ts1.broadcast_with((4, 8)) == TensorShape((2, 4, 8))


def test_matmul_shape() -> None:
    """Test matmul_shape across 1D, 2D, and batched matrix multiplication cases."""
    # 1D x 1D dot product -> ()
    assert matmul_shape((4,), (4,)) == ()
    with pytest.raises(ShapeMismatchError, match="Incompatible 1D"):
        matmul_shape((4,), (5,))

    # 1D x 2D -> (N,)
    assert matmul_shape((4,), (4, 8)) == (8,)
    with pytest.raises(ShapeMismatchError, match="Incompatible inner"):
        matmul_shape((3,), (4, 8))

    # 2D x 1D -> (M,)
    assert matmul_shape((8, 4), (4,)) == (8,)
    with pytest.raises(ShapeMismatchError, match="Incompatible inner"):
        matmul_shape((8, 3), (4,))

    # 2D x 2D -> (M, N)
    assert matmul_shape((16, 32), (32, 64)) == (16, 64)
    with pytest.raises(ShapeMismatchError, match="Incompatible inner"):
        matmul_shape((16, 30), (32, 64))

    # Batched matmul (...B, M, K) x (...B, K, N)
    assert matmul_shape((2, 1, 16, 32), (4, 32, 64)) == (2, 4, 16, 64)

    # Batched with 1D
    assert matmul_shape((4,), (2, 4, 8)) == (2, 8)
    assert matmul_shape((2, 8, 4), (4,)) == (2, 8)

    # Scalar cannot be matmul
    with pytest.raises(ShapeMismatchError, match="Scalars cannot"):
        matmul_shape((), (4, 4))
    with pytest.raises(ShapeMismatchError, match="Scalars cannot"):
        matmul_shape((4, 4), ())

    # TensorShape.matmul_with
    ts1 = TensorShape((16, 32))
    ts2 = TensorShape((32, 64))
    assert ts1.matmul_with(ts2) == TensorShape((16, 64))
    assert ts1.matmul_with((32, 64)) == TensorShape((16, 64))


def test_normalize_axis() -> None:
    """Test normalize_axis with single and multiple axes, negative bounds, and errors."""
    assert normalize_axis(0, 4) == 0
    assert normalize_axis(3, 4) == 3
    assert normalize_axis(-1, 4) == 3
    assert normalize_axis(-4, 4) == 0

    assert normalize_axis((0, -1, -2), 4) == (0, 3, 2)
    assert normalize_axis([-1, 1], 3) == (2, 1)

    with pytest.raises(ValueError, match="out of bounds"):
        normalize_axis(4, 4)
    with pytest.raises(ValueError, match="out of bounds"):
        normalize_axis(-5, 4)

    with pytest.raises(TypeError, match="Invalid type"):
        normalize_axis("invalid", 4)  # type: ignore[arg-type]


def test_symbolic_constraint_tracker() -> None:
    """Test SymbolicConstraintTracker record equality, unification, substitution, and solving."""
    tracker = SymbolicConstraintTracker()
    assert tracker.is_consistent()

    tracker.record_equality(4, 4)
    tracker.add_constraint("batch", 8)
    assert tracker.solve() == {"batch": 8}

    # Record equality between symbols
    tracker.record_equality("dim_x", "dim_y")
    tracker.record_equality("dim_y", 16)
    solution = tracker.solve()
    assert solution["dim_x"] == 16
    assert solution["dim_y"] == 16

    # Substitute
    assert tracker.substitute("dim_x") == 16
    assert tracker.substitute(SymInt("dim_x")) == SymInt(16)
    assert tracker.substitute(SymVar("dim_x")) == SymConst(16)
    assert tracker.substitute(123) == 123
    assert tracker.substitute("unknown_var") == "unknown_var"
    assert tracker.substitute(("dim_x", 32)) == (16, 32)

    # Unify
    assert tracker.unify(1, "dim_z") == "dim_z"
    assert tracker.unify("dim_z", 1) == "dim_z"
    assert tracker.unify("batch", "batch") == 8
    assert tracker.unify("dim_unbound", "dim_unbound") == "dim_unbound"
    assert tracker.unify("batch", 8) == 8

    # Contradiction
    with pytest.raises(ShapeMismatchError, match="Contradictory"):
        tracker.record_equality(4, 5)

    with pytest.raises(ShapeMismatchError, match="Contradictory"):
        tracker.record_equality("batch", 16)


def test_shape_tracker_inference_and_feedback() -> None:
    """Test ShapeTracker elementwise, matmul, and runtime feedback resolution."""
    s1 = TensorSpec(shape=(2, 1, 8), dtype=DType.float32)
    s2 = TensorSpec(shape=(4, 8), dtype=DType.float32)
    s3 = TensorSpec(shape=(8, 16), dtype=DType.float32)

    # Elementwise
    assert ShapeTracker.infer_elementwise([]) == ()
    assert ShapeTracker.infer_elementwise([s1, s2]) == (2, 4, 8)

    # Matmul
    assert ShapeTracker.infer_matmul(s2, s3) == (4, 16)

    # Telemetry update
    telemetry = {
        "node_1": [2, 4, 8],
        "node_2": (4, 16),
    }
    resolved = ShapeTracker.update_from_feedback(telemetry)
    assert resolved["node_1"] == (2, 4, 8)
    assert resolved["node_2"] == (4, 16)

    # Nested shapes format
    telemetry_nested = {
        "shapes": {
            "node_1": [2, 4, 8],
        }
    }
    resolved_nested = ShapeTracker.update_from_feedback(telemetry_nested)
    assert resolved_nested["node_1"] == (2, 4, 8)

    # Resolve dynamic bounds in mock graph
    class MockNode:
        """Mock node for testing dynamic bound resolution."""

        def __init__(
            self,
            node_id: str,
            shape: tuple[object, ...],
        ) -> None:
            """Initialize mock node with id and shape."""
            self.id = node_id
            self.shape_metadata = shape
            self.shape = shape

    class MockGraph:
        """Mock graph containing nodes."""

        def __init__(self, nodes: list[MockNode]) -> None:
            """Initialize mock graph with node list."""
            self.nodes = {n.id: n for n in nodes}

    si_b = SymInt("batch")
    sv_s = SymVar("seq_len")
    n1 = MockNode("n1", (si_b, sv_s, "embed_dim", 64))
    g = MockGraph([n1])

    tracker = SymbolicConstraintTracker()
    bounds = ShapeTracker.resolve_dynamic_bounds(
        g,
        {"n1": [8, 128, 512, 64]},
        tracker=tracker,
    )
    assert bounds["batch"] == 8
    assert bounds["seq_len"] == 128
    assert bounds["embed_dim"] == 512
    assert n1.shape_metadata == (8, 128, 512, 64)

    # Test tracker unification and record_equality branches
    t2 = SymbolicConstraintTracker()
    t2.record_equality(SymConst(16), 16)
    t2.record_equality(16, SymConst(16))
    t2.record_equality(SymConst(16), "dim_k")
    t2.record_equality("dim_m", SymConst(32))
    assert t2.unify("1", "dim_x") == "dim_x"
    assert t2.unify(SymConst(1), "dim_x") == "dim_x"
    assert t2.unify("dim_x", "1") == "dim_x"
    assert t2.unify("dim_x", SymConst(1)) == "dim_x"
    assert t2.unify(SymInt("dim_m"), 32) == SymInt(32)
    assert t2.unify(SymVar("dim_m"), 32) == SymConst(32)
    assert t2.unify("dim_m", 32) == 32

    # broadcast_dimension with equivalent expressions
    assert broadcast_dimension("batch + 2", "2 + batch") == "batch + 2"

    # ShapeTracker edge branches: node with shape only and fewer dims in telemetry
    class MockNodeShapeOnly:
        """Mock node with only shape attribute."""

        def __init__(self, node_id: str, shape: tuple[object, ...]) -> None:
            """Initialize mock node with shape."""
            self.id = node_id
            self.shape = shape

    n_so = MockNodeShapeOnly(
        "n_so", (SymInt("si_track_none"), SymVar("sv_track_none"), "var_dim", 64, 128)
    )
    g_so = MockGraph([n_so])  # type: ignore[list-item]

    # Resolve bounds with tracker=None and shorter telemetry (dim 3 pads with 1)
    bounds_so = ShapeTracker.resolve_dynamic_bounds(
        g_so,
        {"n_so": [16, 32, 64]},
        tracker=None,
    )
    assert bounds_so["si_track_none"] == 16
    assert bounds_so["sv_track_none"] == 32
    assert bounds_so["var_dim"] == 64
    assert n_so.shape == (16, 32, 64, 1, 1)

    # Unify unbound symbols
    assert t2.unify("dim_u", "dim_v") == "dim_u"
    t2.record_equality(SymConst(64), SymConst(64))

    # Record equality when a or b is in _const_map or a digit string
    t2.record_equality("dim_x", 16)
    t2.record_equality("dim_x", "dim_y")
    t2.record_equality("dim_z", "dim_y")
    t2.record_equality("16", "dim_p")
    t2.record_equality("dim_n", "32")
    assert t2.solve()["dim_p"] == 16
    assert t2.solve()["dim_n"] == 32

    # broadcast_dimension between two dynamic expressions without tracker
    assert broadcast_dimension("dim_foo", "dim_bar", tracker=None) == "dim_foo"

    # ShapeTracker with list-based graph nodes and unobserved node skipped
    class MockGraphWithList:
        """Mock graph with nodes as a list."""

        def __init__(self, nodes_list: list[MockNode]) -> None:
            """Initialize mock graph with list."""
            self.nodes = nodes_list

    n_list = MockNode("n_list", (si_b, "dynamic_axis", 64))
    n_unobserved = MockNode("n_unobserved", (128, 256))
    g_list = MockGraphWithList([n_list, n_unobserved])
    bounds_list = ShapeTracker.resolve_dynamic_bounds(
        g_list,
        {"n_list": [4, 16, 64]},
        tracker=t2,
    )
    assert bounds_list["batch"] == 4
    assert bounds_list["dynamic_axis"] == 16

    # Empty tuple substitution and empty feedback telemetry
    assert t2.substitute(()) == ()
    assert ShapeTracker.update_from_feedback({}) == {}
    assert ShapeTracker.update_from_feedback({"shapes": {"inv": "not_list"}}) == {}
    assert ShapeTracker.update_from_feedback({"inv": "not_list"}) == {}

    # Exact rank match in resolve_dynamic_bounds
    n_exact = MockNode("n_exact", (SymInt("A"), SymVar("B"), "C", 10))
    n_none = MockNode("n_none", None)  # type: ignore[arg-type]

    class MockNodeShapeMetaOnly:
        """Mock node with shape_metadata only."""

        def __init__(self, node_id: str, shape_meta: tuple[object, ...]) -> None:
            """Initialize mock node with shape_metadata."""
            self.id = node_id
            self.shape_metadata = shape_meta

    n_smo = MockNodeShapeMetaOnly("n_smo", (16, 32))
    g_exact = MockGraph([n_exact, n_none, n_smo])  # type: ignore[list-item]
    bounds_exact = ShapeTracker.resolve_dynamic_bounds(
        g_exact,
        {"n_exact": [1, 2, 3, 10], "n_none": [1], "n_smo": [16, 32]},
        tracker=t2,
    )
    assert bounds_exact["A"] == 1
    assert bounds_exact["B"] == 2
    assert bounds_exact["C"] == 3

    # broadcast_dimension raising when incompatible static and dynamic without tracker
    with pytest.raises(ShapeMismatchError, match="Incompatible dimensions"):
        broadcast_dimension(5, "dim_x", tracker=None)
