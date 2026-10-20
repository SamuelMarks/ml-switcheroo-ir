"""Disjoint-set constraint tracking and dimension equality unification."""

from __future__ import annotations

import typing
import typing_extensions

from ml_switcheroo_ir.shapes.ast import DimensionType, SymConst, SymInt, SymNode

DimensionBound: typing_extensions.TypeAlias = typing.Union[int, str, SymNode, SymInt]

__all__ = [
    "DimensionBound",
    "ShapeMismatchError",
    "SymbolicConstraintTracker",
]


class ShapeMismatchError(ValueError):
    """Raised when shapes or symbolic dimensions are incompatible or contradict constraints."""

    def __init__(self, message: str) -> None:
        """Initialize ShapeMismatchError with diagnostic message.

        Args:
            message (str): Description of the shape mismatch.
        """
        super().__init__(message)


class SymbolicConstraintTracker:
    """Disjoint-set constraint tracker to record dimension equalities across graphs."""

    def __init__(self) -> None:
        """Initialize SymbolicConstraintTracker."""
        self._parent: dict[str, str] = {}
        self._const_map: dict[str, int] = {}
        self._contradiction: bool = False
        self._error_msg: str = ""

    def _canonical_key(self, dim: int | str | SymNode | SymInt) -> str:
        """Return string representation key for dimension.

        Args:
            dim (int | str | SymNode | SymInt): Dimension to stringify.

        Returns:
            str: Canonical key string.
        """
        if isinstance(dim, SymInt):
            return str(dim.canonical())
        if isinstance(dim, SymNode):
            return str(dim.canonical())
        return str(dim)

    def _find(self, key: str) -> str:
        """Find representative with path compression.

        Args:
            key (str): Dimension key.

        Returns:
            str: Representative key.
        """
        if key not in self._parent:
            self._parent[key] = key
            return key
        if self._parent[key] != key:
            self._parent[key] = self._find(self._parent[key])
        return self._parent[key]

    def _extract_const_value(
        self,
        dim: int | str | SymNode | SymInt,
        key: str,
        root: str,
    ) -> int | None:
        """Resolve an integer constant bound for a dimension if known.

        Args:
            dim (int | str | SymNode | SymInt): Dimension value.
            key (str): Canonical string key for dim.
            root (str): Disjoint-set root key for dim.

        Returns:
            int | None: Resolved integer constant or None.

        Raises:
            TypeError: If dim is a boolean value.
        """
        if isinstance(dim, bool):
            raise TypeError(f"Boolean value {dim!r} is not a valid dimension bound.")
        if isinstance(dim, int):
            return dim
        if isinstance(dim, SymConst):
            return dim.value
        if isinstance(dim, (SymNode, SymInt)) and dim.is_constant:
            return dim.eval({})
        if key.lstrip("-").isdigit():
            return int(key)
        return self._const_map.get(root)

    @staticmethod
    def _is_unit_dimension(dim: int | str | SymNode | SymInt) -> bool:
        """Return True if dim represents a unit (size-1) dimension.

        Args:
            dim (int | str | SymNode | SymInt): Dimension value.

        Returns:
            bool: True if unit dimension.
        """
        if isinstance(dim, bool):
            return False
        return (
            (isinstance(dim, int) and dim == 1)
            or (isinstance(dim, SymConst) and dim.value == 1)
            or str(dim) == "1"
        )

    def record_equality(
        self,
        a: int | str | SymNode | SymInt,
        b: int | str | SymNode | SymInt,
    ) -> None:
        """Record an equality constraint between two dimensions.

        Args:
            a (int | str | SymNode | SymInt): First dimension.
            b (int | str | SymNode | SymInt): Second dimension.

        Raises:
            TypeError: If either dimension is a boolean value.
            ShapeMismatchError: If the recorded equality causes a contradiction.
        """
        if isinstance(a, bool) or isinstance(b, bool):
            raise TypeError("Boolean values are not valid tensor dimensions.")
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                self._contradiction = True
                self._error_msg = f"Contradictory dimension constraints: {a} != {b}"
                raise ShapeMismatchError(self._error_msg)
            return

        key_a = self._canonical_key(a)
        key_b = self._canonical_key(b)

        root_a = self._find(key_a)
        root_b = self._find(key_b)

        const_a = self._extract_const_value(a, key_a, root_a)
        const_b = self._extract_const_value(b, key_b, root_b)

        if const_a is not None and const_b is not None and const_a != const_b:
            self._contradiction = True
            self._error_msg = f"Contradictory dimension constraints: {a} == {b} ({const_a} != {const_b})"
            raise ShapeMismatchError(self._error_msg)

        if root_a != root_b:
            self._parent[root_b] = root_a
            chosen_const = const_a if const_a is not None else const_b
            if chosen_const is not None:
                self._const_map[root_a] = chosen_const
        elif const_a is not None:
            self._const_map[root_a] = const_a

    def unify(
        self,
        a: int | str | SymNode | SymInt,
        b: int | str | SymNode | SymInt,
    ) -> int | str | SymNode | SymInt:
        """Unify two dimensions according to broadcasting semantics.

        Args:
            a (int | str | SymNode | SymInt): First dimension.
            b (int | str | SymNode | SymInt): Second dimension.

        Returns:
            int | str | SymNode | SymInt: Unified dimension representative.
        """
        if self._is_unit_dimension(a):
            return b
        if self._is_unit_dimension(b):
            return a

        self.record_equality(a, b)

        root = self._find(self._canonical_key(a))
        if root in self._const_map:
            val = self._const_map[root]
            if isinstance(a, SymInt) or isinstance(b, SymInt):
                return SymInt(val)
            if isinstance(a, SymNode) or isinstance(b, SymNode):
                return SymConst(val)
            return val

        return a

    def add_constraint(self, dim: str | SymNode | SymInt, val: int) -> None:
        """Assign a concrete integer value constraint to a symbolic dimension.

        Args:
            dim (str | SymNode | SymInt): Symbolic dimension.
            val (int): Concrete integer value.
        """
        self.record_equality(dim, val)

    def is_consistent(self) -> bool:
        """Check if all recorded constraints are mutually consistent.

        Returns:
            bool: True if no contradictions have been encountered.
        """
        return not self._contradiction

    def solve(self) -> dict[str, int]:
        """Solve for all bound variables and return their resolved values.

        Only returns valid identifier keys (excluding integer literals and
        composite algebraic expressions).

        Returns:
            dict[str, int]: Mapping of variable names to integer values.
        """
        solution: dict[str, int] = {}
        for var, root in list(self._parent.items()):
            if not var.isidentifier():
                continue
            actual_root = self._find(root)
            if actual_root in self._const_map:
                solution[var] = self._const_map[actual_root]
        return solution

    @typing.overload
    def substitute(
        self, dim_or_shape: tuple[DimensionType, ...]
    ) -> tuple[DimensionType, ...]:
        """Substitute known resolved values into a shape tuple.

        Args:
            dim_or_shape (tuple[DimensionType, ...]): Input shape tuple.

        Returns:
            tuple[DimensionType, ...]: Substituted shape tuple.
        """

    @typing.overload
    def substitute(self, dim_or_shape: DimensionType) -> DimensionType:
        """Substitute known resolved values into a single dimension.

        Args:
            dim_or_shape (DimensionType): Input dimension.

        Returns:
            DimensionType: Substituted dimension.
        """

    def substitute(
        self,
        dim_or_shape: (
            int | str | SymNode | SymInt | tuple[int | str | SymNode | SymInt, ...]
        ),
    ) -> int | str | SymNode | SymInt | tuple[int | str | SymNode | SymInt, ...]:
        """Substitute known resolved values into dimension or shape tuple.

        Args:
            dim_or_shape (int | str | SymNode | SymInt | tuple[int | str | SymNode | SymInt, ...]):
                Input dimension or shape.

        Returns:
            int | str | SymNode | SymInt | tuple[int | str | SymNode | SymInt, ...]:
                Substituted dimension or shape.
        """
        if isinstance(dim_or_shape, tuple):
            sub_res: list[DimensionType] = [self.substitute(d) for d in dim_or_shape]
            return tuple(sub_res)

        key = self._canonical_key(dim_or_shape)
        root = self._find(key)
        if root in self._const_map:
            val = self._const_map[root]
            if isinstance(dim_or_shape, SymInt):
                return SymInt(val)
            if isinstance(dim_or_shape, SymNode):
                return SymConst(val)
            return val

        return dim_or_shape
