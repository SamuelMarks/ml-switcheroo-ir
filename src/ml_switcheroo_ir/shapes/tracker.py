"""Disjoint-set constraint tracking, affine equation solving, and dimension unification."""

from __future__ import annotations

import typing

import typing_extensions

from ml_switcheroo_ir.shapes.ast import (
    DimensionType,
    SymBinaryOp,
    SymCondition,
    SymConst,
    SymInt,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
)
from ml_switcheroo_ir.shapes.parser import parse_symbolic_expr
from ml_switcheroo_ir.shapes.polynomial import MonomialType, Polynomial

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


def _eval_poly_with_knowns(poly: Polynomial, knowns: dict[str, int]) -> Polynomial:
    """Partially evaluate ``poly`` by substituting known integer bindings from ``knowns``.

    Args:
        poly (Polynomial): Input multivariate polynomial.
        knowns (dict[str, int]): Mapping of solved variable names to concrete integers.

    Returns:
        Polynomial: Reduced polynomial over remaining unbound variables.
    """
    if not knowns:
        return poly
    new_terms: dict[MonomialType, int] = {}
    for mono, coeff in poly.terms.items():
        new_coeff = coeff
        rem_factors: list[tuple[str, int]] = []
        for var, power in mono:
            if var in knowns:
                new_coeff *= knowns[var] ** power
            else:
                rem_factors.append((var, power))
        rem_mono: MonomialType = tuple(rem_factors)
        updated = new_terms.get(rem_mono, 0) + new_coeff
        if updated == 0:
            new_terms.pop(rem_mono, None)
        else:
            new_terms[rem_mono] = updated
    return Polynomial(new_terms)


def _substitute_piecewise(node: SymPiecewise, solved: dict[str, int]) -> SymNode:
    """Substitute ``solved`` variable bindings into a ``SymPiecewise`` node and simplify.

    Args:
        node (SymPiecewise): Piecewise symbolic node.
        solved (dict[str, int]): Solved variable-to-integer dictionary.

    Returns:
        SymNode: Simplified substituted piecewise node or folded branch.
    """
    new_cases: list[tuple[SymCondition, SymNode]] = []
    for cond, expr in node.cases:
        sub_expr = _substitute_symnode(expr, solved)
        sub_cond: SymCondition = cond
        if isinstance(cond, SymNode):
            sub_cond = _substitute_symnode(cond, solved)
        elif isinstance(cond, str):
            parsed_cond = _safe_parse_dim_node(cond)
            if parsed_cond is not None:
                sub_cond = _substitute_symnode(parsed_cond, solved)
        new_cases.append((sub_cond, sub_expr))
    sub_def = _substitute_symnode(node.default, solved)
    return SymPiecewise(cases=new_cases, default=sub_def).simplify()


def _substitute_symnode(node: SymNode, solved: dict[str, int]) -> SymNode:
    """Recursively substitute ``solved`` variable bindings inside ``node`` and simplify.

    Args:
        node (SymNode): Symbolic expression node.
        solved (dict[str, int]): Mapping of solved variable names to integer constants.

    Returns:
        SymNode: Substituted and simplified ``SymNode``.
    """
    if not solved or isinstance(node, SymConst):
        return node
    if isinstance(node, SymVar):
        return SymConst(solved[node.name]) if node.name in solved else node
    if isinstance(node, SymBinaryOp):
        return SymBinaryOp(
            node.op,
            _substitute_symnode(node.left, solved),
            _substitute_symnode(node.right, solved),
        ).simplify()
    if isinstance(node, SymUnaryOp):
        return SymUnaryOp(node.op, _substitute_symnode(node.operand, solved)).simplify()
    if isinstance(node, SymPiecewise):
        return _substitute_piecewise(node, solved)
    return node


def _safe_parse_dim_node(dim: int | str | SymNode | SymInt) -> SymNode | None:
    """Convert ``dim`` to a ``SymNode`` if syntactically valid, returning ``None`` on parse failure.

    Args:
        dim (int | str | SymNode | SymInt): Dimension bound candidate.

    Returns:
        SymNode | None: Parsed ``SymNode`` or ``None`` if ``dim`` is an unparseable raw string.
    """
    if isinstance(dim, SymNode):
        return dim
    if isinstance(dim, SymInt):
        return dim.node
    if isinstance(dim, int):
        return SymConst(dim)
    try:
        return parse_symbolic_expr(dim)
    except ValueError:
        return None


class SymbolicConstraintTracker:
    """Disjoint-set and affine algebraic constraint tracker for symbolic dimensions."""

    def __init__(self) -> None:
        """Initialize SymbolicConstraintTracker."""
        self._parent: dict[str, str] = {}
        self._const_map: dict[str, int] = {}
        self._poly_constraints: list[tuple[Polynomial, str]] = []
        self._contradiction: bool = False
        self._error_msg: str = ""

    def _canonical_key(self, dim: int | str | SymNode | SymInt) -> str:
        """Return canonical string representation key for ``dim``.

        Args:
            dim (int | str | SymNode | SymInt): Dimension to stringify.

        Returns:
            str: Canonical key string.
        """
        if isinstance(dim, (SymInt, SymNode)):
            return str(dim.canonical())
        return str(dim)

    def _find(self, key: str) -> str:
        """Find disjoint-set representative with path compression.

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
        """Resolve an integer constant bound for ``dim`` if known.

        Args:
            dim (int | str | SymNode | SymInt): Dimension value.
            key (str): Canonical string key for ``dim``.
            root (str): Disjoint-set root key for ``dim``.

        Returns:
            int | None: Resolved integer constant or None.

        Raises:
            TypeError: If ``dim`` is a boolean value.
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
        """Return True if ``dim`` represents a unit (size-1) dimension.

        Args:
            dim (int | str | SymNode | SymInt): Dimension value.

        Returns:
            bool: True if unit dimension.
        """
        if isinstance(dim, bool):
            return False
        return (isinstance(dim, int) and dim == 1) or (isinstance(dim, SymConst) and dim.value == 1) or str(dim) == "1"

    def _fail_contradiction(self, message: str) -> typing.NoReturn:
        """Mark tracker inconsistent and raise ``ShapeMismatchError``.

        Args:
            message (str): Diagnostic contradiction message.

        Returns:
            typing.NoReturn: Never returns; always raises ``ShapeMismatchError``.

        Raises:
            ShapeMismatchError: Always raised with ``message``.
        """
        self._contradiction = True
        self._error_msg = message
        raise ShapeMismatchError(message)

    def _bind_var_constant(self, var_name: str, val: int) -> None:
        """Bind unbound identifier ``var_name`` to ``val`` in the disjoint-set map.

        Args:
            var_name (str): Unbound variable identifier name.
            val (int): Solved integer value.
        """
        root = self._find(var_name)
        self._const_map[root] = val

    def _try_solve_affine_poly(self, poly: Polynomial, context: str) -> tuple[bool, bool]:
        """Attempt to resolve a polynomial equality ``poly == 0`` as constant or univariate affine.

        Solves ``a * X + b == 0`` when ``(-b) % a == 0`` and raises ``ShapeMismatchError``
        when ``poly`` reduces to a non-zero constant or has no integer solution.

        Args:
            poly (Polynomial): Polynomial representing ``lhs - rhs``.
            context (str): Human-readable equation description.

        Returns:
            tuple[bool, bool]: ``(is_fully_satisfied, bound_new_variable)``.
        """
        reduced = _eval_poly_with_knowns(poly, self.solve())
        if reduced.is_const():
            if reduced.get_const() != 0:
                self._fail_contradiction(f"Contradictory dimension constraints: {context}")
            return True, False

        vars_in_poly = {v for mono in reduced.terms for v, _ in mono}
        max_deg = max((sum(p for _, p in mono) for mono in reduced.terms), default=0)
        if len(vars_in_poly) == 1 and max_deg == 1:
            var_name = next(iter(vars_in_poly))
            a_coeff = reduced.terms[((var_name, 1),)]
            b_const = reduced.get_const()
            if (-b_const) % a_coeff != 0:
                self._fail_contradiction(
                    f"Unsolvable integer affine constraint: {context} ({a_coeff}*{var_name} + {b_const} == 0)"
                )
            self._bind_var_constant(var_name, (-b_const) // a_coeff)
            return True, True
        return False, False

    def _propagate_poly_constraints(self) -> None:
        """Iteratively reduce pending polynomial constraints until a fixed point is reached."""
        progressed = True
        while progressed and self._poly_constraints:
            progressed = False
            remaining: list[tuple[Polynomial, str]] = []
            for poly, ctx in self._poly_constraints:
                satisfied, newly_bound = self._try_solve_affine_poly(poly, ctx)
                if newly_bound:
                    progressed = True
                if not satisfied:
                    remaining.append((poly, ctx))
            self._poly_constraints = remaining

    def _record_algebraic_equality(
        self,
        a: int | str | SymNode | SymInt,
        b: int | str | SymNode | SymInt,
    ) -> None:
        """Record polynomial difference ``a - b == 0`` and solve affine constraints.

        Args:
            a (int | str | SymNode | SymInt): Left-hand dimension expression.
            b (int | str | SymNode | SymInt): Right-hand dimension expression.
        """
        node_a = _safe_parse_dim_node(a)
        node_b = _safe_parse_dim_node(b)
        if node_a is None or node_b is None:
            return
        poly_a = node_a.to_polynomial()
        poly_b = node_b.to_polynomial()
        if poly_a is None or poly_b is None:
            return
        diff = poly_a.sub(poly_b)
        ctx = f"{a} == {b}"
        satisfied, newly_bound = self._try_solve_affine_poly(diff, ctx)
        if not satisfied:
            self._poly_constraints.append((diff, ctx))
        if newly_bound:
            self._propagate_poly_constraints()

    def record_equality(
        self,
        a: int | str | SymNode | SymInt,
        b: int | str | SymNode | SymInt,
    ) -> None:
        """Record an equality constraint between two dimensions and solve affine bindings.

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
                self._fail_contradiction(f"Contradictory dimension constraints: {a} != {b}")
            return

        key_a = self._canonical_key(a)
        key_b = self._canonical_key(b)
        root_a = self._find(key_a)
        root_b = self._find(key_b)
        const_a = self._extract_const_value(a, key_a, root_a)
        const_b = self._extract_const_value(b, key_b, root_b)

        if const_a is not None and const_b is not None and const_a != const_b:
            self._fail_contradiction(f"Contradictory dimension constraints: {a} == {b} ({const_a} != {const_b})")

        if root_a != root_b:
            self._parent[root_b] = root_a
            chosen_const = const_a if const_a is not None else const_b
            if chosen_const is not None:
                self._const_map[root_a] = chosen_const
        elif const_a is not None:
            self._const_map[root_a] = const_a

        self._propagate_poly_constraints()
        self._record_algebraic_equality(a, b)

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

    def _substitute_composite(self, dim: DimensionType, solved: dict[str, int]) -> DimensionType:
        """Substitute ``solved`` variable bindings into a composite ``SymNode``, ``SymInt``, or ``str``.

        Args:
            dim (DimensionType): Single dimension expression.
            solved (dict[str, int]): Mapping of solved variable names to integers.

        Returns:
            DimensionType: Substituted dimension value.
        """
        if isinstance(dim, SymInt):
            sub_node = _substitute_symnode(dim.node, solved)
            return SymInt(sub_node.value) if isinstance(sub_node, SymConst) else SymInt(sub_node)
        if isinstance(dim, SymNode):
            return _substitute_symnode(dim, solved)
        if isinstance(dim, str):
            parsed = _safe_parse_dim_node(dim)
            if parsed is not None and (parsed.free_vars() & solved.keys()):
                sub_node = _substitute_symnode(parsed, solved)
                return sub_node.value if isinstance(sub_node, SymConst) else str(sub_node)
        return dim

    @typing.overload
    def substitute(self, dim_or_shape: tuple[DimensionType, ...]) -> tuple[DimensionType, ...]:
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
        dim_or_shape: DimensionType | tuple[DimensionType, ...],
    ) -> DimensionType | tuple[DimensionType, ...]:
        """Substitute known resolved values into a single dimension or shape tuple.

        Recursively substitutes solved variables inside composite ``SymNode``, ``SymInt``,
        and symbolic ``str`` expressions (e.g., ``substitute(SymInt("B * 2")) -> SymInt(16)``).

        Args:
            dim_or_shape (DimensionType | tuple[DimensionType, ...]): Input dimension or shape tuple.

        Returns:
            DimensionType | tuple[DimensionType, ...]: Substituted dimension or shape tuple.

        Raises:
            TypeError: If ``dim_or_shape`` is a boolean value.
        """
        if isinstance(dim_or_shape, bool):
            raise TypeError(f"Boolean value {dim_or_shape!r} is not a valid dimension.")
        if isinstance(dim_or_shape, tuple):
            return tuple(self.substitute(d) for d in dim_or_shape)

        key = self._canonical_key(dim_or_shape)
        root = self._find(key)
        if root in self._const_map:
            val = self._const_map[root]
            if isinstance(dim_or_shape, SymInt):
                return SymInt(val)
            if isinstance(dim_or_shape, SymNode):
                return SymConst(val)
            return val

        solved = self.solve()
        if not solved:
            return dim_or_shape
        return self._substitute_composite(dim_or_shape, solved)
