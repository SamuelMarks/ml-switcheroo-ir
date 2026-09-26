"""Symbolic shape system, dimension algebra, broadcasting, and shape inference.

Provides zero-dependency symbolic dimension representation, algebraic expression trees,
polynomial canonicalization, NumPy-compliant broadcasting, and matrix multiplication shape inference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Union, overload

if TYPE_CHECKING:
    from ml_switcheroo_ir.types import TensorSpec

DimensionType = Union[int, str, "SymNode", "SymInt"]
ShapeType = Sequence[DimensionType]


class ShapeMismatchError(ValueError):
    """Raised when shapes or symbolic dimensions are incompatible or contradict constraints."""

    def __init__(self, message: str) -> None:
        """Initialize ShapeMismatchError with diagnostic message.

        Args:
            message (str): Description of the shape mismatch.
        """
        super().__init__(message)


MonomialType = tuple[tuple[str, int], ...]
SymCondition = Union[Callable[[dict[str, int]], bool], "SymNode", bool, str]


class Polynomial:
    """Canonical multivariate polynomial over integer coefficients for symbolic equivalence.

    Attributes:
        terms (dict[MonomialType, int]): Mapping from canonical monomials to integer coefficients.
    """

    def __init__(self, terms: dict[MonomialType, int] | None = None) -> None:
        """Initialize a canonical polynomial.

        Args:
            terms (dict[MonomialType, int] | None): Dictionary mapping canonical monomials
                (tuples of (var_name, power) sorted) to non-zero integer coefficients.
        """
        self.terms: dict[MonomialType, int] = {}
        if terms:
            for mono, coeff in terms.items():
                if coeff != 0:
                    self.terms[mono] = coeff

    @classmethod
    def from_const(cls, val: int) -> Polynomial:
        """Construct a constant polynomial.

        Args:
            val (int): Constant integer value.

        Returns:
            Polynomial: Resulting constant polynomial.
        """
        if val == 0:
            return cls({})
        return cls({(): val})

    @classmethod
    def from_var(cls, name: str) -> Polynomial:
        """Construct a single variable polynomial.

        Args:
            name (str): Variable identifier.

        Returns:
            Polynomial: Resulting variable polynomial.
        """
        return cls({(((name, 1),)): 1})

    def is_zero(self) -> bool:
        """Check if the polynomial is identically zero.

        Returns:
            bool: True if the polynomial has no non-zero terms.
        """
        return len(self.terms) == 0

    def is_const(self) -> bool:
        """Check if the polynomial is a constant.

        Returns:
            bool: True if polynomial contains at most a constant term.
        """
        if self.is_zero():
            return True
        return len(self.terms) == 1 and () in self.terms

    def get_const(self) -> int:
        """Return the constant integer value of the polynomial.

        Returns:
            int: The constant term value (0 if not present).
        """
        return self.terms.get((), 0)

    def add(self, other: Polynomial) -> Polynomial:
        """Add two polynomials.

        Args:
            other (Polynomial): The other polynomial.

        Returns:
            Polynomial: Sum of the two polynomials.
        """
        new_terms = dict(self.terms)
        for mono, coeff in other.terms.items():
            new_coeff = new_terms.get(mono, 0) + coeff
            if new_coeff == 0:
                new_terms.pop(mono, None)
            else:
                new_terms[mono] = new_coeff
        return Polynomial(new_terms)

    def sub(self, other: Polynomial) -> Polynomial:
        """Subtract another polynomial from self.

        Args:
            other (Polynomial): The polynomial to subtract.

        Returns:
            Polynomial: Difference of the two polynomials.
        """
        new_terms = dict(self.terms)
        for mono, coeff in other.terms.items():
            new_coeff = new_terms.get(mono, 0) - coeff
            if new_coeff == 0:
                new_terms.pop(mono, None)
            else:
                new_terms[mono] = new_coeff
        return Polynomial(new_terms)

    def mul(self, other: Polynomial) -> Polynomial:
        """Multiply two polynomials.

        Args:
            other (Polynomial): The other polynomial.

        Returns:
            Polynomial: Product of the two polynomials.
        """
        new_terms: dict[MonomialType, int] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                merged_dict: dict[str, int] = {}
                for v, p in m1:
                    merged_dict[v] = merged_dict.get(v, 0) + p
                for v, p in m2:
                    merged_dict[v] = merged_dict.get(v, 0) + p
                merged_mono = tuple(
                    sorted((v, p) for v, p in merged_dict.items() if p > 0)
                )
                coeff = c1 * c2
                curr = new_terms.get(merged_mono, 0) + coeff
                if curr == 0:
                    new_terms.pop(merged_mono, None)
                else:
                    new_terms[merged_mono] = curr
        return Polynomial(new_terms)

    def div_exact(self, other: Polynomial) -> Polynomial | None:
        """Attempt exact polynomial division.

        Args:
            other (Polynomial): The divisor polynomial.

        Returns:
            Polynomial | None: Quotient polynomial if division is exact, else None.
        """
        if other.is_zero():
            return None
        if self == other:
            return Polynomial.from_const(1)
        if other.is_const():
            c = other.get_const()
            new_terms: dict[MonomialType, int] = {}
            for mono, coeff in self.terms.items():
                if coeff % c != 0:
                    return None
                new_terms[mono] = coeff // c
            return Polynomial(new_terms)

        if len(other.terms) == 1:
            div_mono, div_coeff = next(iter(other.terms.items()))
            new_terms_exact: dict[MonomialType, int] = {}
            for mono, coeff in self.terms.items():
                if coeff % div_coeff != 0:
                    return None
                mono_dict = dict(mono)
                div_dict = dict(div_mono)
                for var, power in div_dict.items():
                    if mono_dict.get(var, 0) < power:
                        return None
                    mono_dict[var] -= power
                    if mono_dict[var] == 0:
                        del mono_dict[var]
                quot_mono = tuple(sorted(mono_dict.items()))
                new_terms_exact[quot_mono] = coeff // div_coeff
            return Polynomial(new_terms_exact)

        return None

    def canonical_str(self) -> str:
        """Return canonical deterministic string representation of polynomial.

        Returns:
            str: Deterministic string format with terms sorted consistently.
        """
        if self.is_zero():
            return "0"

        def sort_key(mono: MonomialType) -> tuple[int, list[str]]:
            """Sort key for monomial terms by descending degree and variable names.

            Args:
                mono (MonomialType): Monomial representation.

            Returns:
                tuple[int, list[str]]: Comparison sort key.
            """
            degree = sum(p for _, p in mono)
            vars_sorted = [v for v, _ in mono]
            return (-degree, vars_sorted)

        sorted_monos = sorted(self.terms.keys(), key=sort_key)
        parts: list[str] = []
        for i, mono in enumerate(sorted_monos):
            coeff = self.terms[mono]
            sign = ""
            if i > 0:
                if coeff > 0:
                    sign = " + "
                else:
                    sign = " - "
                    coeff = abs(coeff)
            elif coeff < 0:
                sign = "-"
                coeff = abs(coeff)

            if len(mono) == 0:
                part = f"{coeff}"
            else:
                var_factors: list[str] = []
                for var, pow_val in mono:
                    if pow_val == 1:
                        var_factors.append(var)
                    else:
                        var_factors.append(f"{var}^{pow_val}")
                var_str = "*".join(var_factors)
                if coeff == 1:
                    part = var_str
                else:
                    part = f"{coeff}*{var_str}"
            parts.append(f"{sign}{part}")

        return "".join(parts)

    def __eq__(self, other: object) -> bool:
        """Evaluate equality between two polynomials.

        Args:
            other (object): Other object to compare against.

        Returns:
            bool: True if both polynomials have identical canonical terms.
        """
        if not isinstance(other, Polynomial):
            return False
        return self.terms == other.terms

    def __hash__(self) -> int:
        """Compute hash of polynomial terms.

        Returns:
            int: Hash code.
        """
        return hash(tuple(sorted(self.terms.items())))

    def __str__(self) -> str:
        """Return string representation of polynomial.

        Returns:
            str: Canonical string format.
        """
        return self.canonical_str()

    def __repr__(self) -> str:
        """Return debugging representation.

        Returns:
            str: Representation string.
        """
        return f"Polynomial({self.canonical_str()})"

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate polynomial expression given variable bindings.

        Args:
            env (dict[str, int]): Variable assignment dictionary.

        Returns:
            int: Evaluated integer result.

        Raises:
            KeyError: If a variable in the polynomial is not in env.
        """
        total = 0
        for mon, coeff in self.terms.items():
            term_val = coeff
            for var, power in mon:
                if var not in env:
                    raise KeyError(f"Variable '{var}' not bound in environment.")
                term_val *= env[var] ** power
            total += term_val
        return total


def _poly_to_symnode(poly: Polynomial) -> SymNode:
    """Convert a canonical Polynomial back into a simplified SymNode expression tree.

    Args:
        poly (Polynomial): Canonical polynomial instance.

    Returns:
        SymNode: Equivalent structured symbolic expression tree.
    """
    if poly.is_zero():
        return SymConst(0)
    if poly.is_const():
        return SymConst(poly.get_const())

    def sort_key(mono: MonomialType) -> tuple[int, list[str]]:
        """Sort key for monomial terms by descending degree and variable names.

        Args:
            mono (MonomialType): Monomial representation.

        Returns:
            tuple[int, list[str]]: Comparison sort key.
        """
        degree = sum(p for _, p in mono)
        vars_sorted = [v for v, _ in mono]
        return (-degree, vars_sorted)

    sorted_monos = sorted(poly.terms.keys(), key=sort_key)
    res_node: SymNode | None = None

    for mono in sorted_monos:
        raw_coeff = poly.terms[mono]
        coeff = abs(raw_coeff)

        var_factors: list[SymNode] = []
        if coeff != 1 or len(mono) == 0:
            var_factors.append(SymConst(coeff))

        for var, power in mono:
            for _ in range(power):
                var_factors.append(SymVar(var))

        term_node = var_factors[0]
        for f in var_factors[1:]:
            term_node = SymBinaryOp("*", term_node, f)

        if res_node is None:
            if raw_coeff < 0:
                res_node = SymBinaryOp("-", SymConst(0), term_node)
            else:
                res_node = term_node
        elif raw_coeff < 0:
            res_node = SymBinaryOp("-", res_node, term_node)
        else:
            res_node = SymBinaryOp("+", res_node, term_node)

    return res_node or SymConst(0)


def _tokenize_expr(expr_str: str) -> list[str]:
    """Tokenize a symbolic expression string into component tokens.

    Args:
        expr_str (str): Input expression string.

    Returns:
        list[str]: Sequence of string tokens.
    """
    tokens: list[str] = []
    i = 0
    s = expr_str.strip()
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "()+*":
            if c == "*" and i + 1 < n and s[i + 1] == "*":
                tokens.append("**")
                i += 2
            else:
                tokens.append(c)
                i += 1
        elif c in ("<", ">", "=", "!"):
            if i + 1 < n and s[i + 1] == "=":
                tokens.append(s[i : i + 2])
                i += 2
            else:
                tokens.append(c)
                i += 1
        elif c == "-":
            tokens.append(c)
            i += 1
        elif c == "/":
            if i + 1 < n and s[i + 1] == "/":
                tokens.append("//")
                i += 2
            else:
                tokens.append("//")
                i += 1
        elif c == "%":
            tokens.append("%")
            i += 1
        elif c.isdigit():
            start = i
            while i < n and s[i].isdigit():
                i += 1
            tokens.append(s[start:i])
        elif c.isalpha() or c == "_":
            start = i
            while i < n and (s[i].isalnum() or s[i] == "_"):
                i += 1
            tokens.append(s[start:i])
        else:
            i += 1
    return tokens


def _parse_sym_str(s: str) -> SymNode:
    """Parse an algebraic string expression into a structured SymNode.

    Args:
        s (str): String expression.

    Returns:
        SymNode: Structured symbolic expression node.
    """
    tokens = _tokenize_expr(s)
    if not tokens:
        return SymVar(s)

    idx = 0

    def parse_comparison() -> SymNode:
        """Parse comparison expressions.

        Returns:
            SymNode: Comparison binary expression node or additive expression.
        """
        nonlocal idx
        node = parse_expr()
        if idx < len(tokens) and tokens[idx] in (
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
        ):
            op = tokens[idx]
            idx += 1
            right = parse_expr()
            node = SymBinaryOp(op, node, right)
        return node

    def parse_expr() -> SymNode:
        """Parse additive binary expressions.

        Returns:
            SymNode: Root parsed expression node.
        """
        nonlocal idx
        node = parse_term()
        while idx < len(tokens) and tokens[idx] in ("+", "-"):
            op = tokens[idx]
            idx += 1
            right = parse_term()
            node = SymBinaryOp(op, node, right)
        return node

    def parse_term() -> SymNode:
        """Parse multiplicative binary expressions.

        Returns:
            SymNode: Root parsed term node.
        """
        nonlocal idx
        node = parse_factor()
        while idx < len(tokens) and tokens[idx] in ("*", "//", "%"):
            op = tokens[idx]
            idx += 1
            right = parse_factor()
            node = SymBinaryOp(op, node, right)
        return node

    def parse_factor() -> SymNode:
        """Parse factor, power expression, parenthesized expression, or unary negative.

        Returns:
            SymNode: Root parsed factor node.
        """
        nonlocal idx
        if idx >= len(tokens):
            return SymConst(0)
        tok = tokens[idx]
        if tok == "(":
            idx += 1
            node = parse_comparison()
            if idx < len(tokens) and tokens[idx] == ")":
                idx += 1
            if idx < len(tokens) and tokens[idx] == "**":
                idx += 1
                exponent = parse_factor()
                node = SymBinaryOp("**", node, exponent)
            return node
        if tok == "-":
            idx += 1
            inner = parse_factor()
            return SymBinaryOp("-", SymConst(0), inner)
        idx += 1
        base_node: SymNode
        if tok.isdigit():
            base_node = SymConst(int(tok))
        else:
            base_node = SymVar(tok)
        if idx < len(tokens) and tokens[idx] == "**":
            idx += 1
            exponent = parse_factor()
            base_node = SymBinaryOp("**", base_node, exponent)
        return base_node

    node = parse_comparison()
    if idx == len(tokens):
        return node

    return SymVar(s)


class SymNode:
    """Base class for all symbolic expression tree nodes."""

    @classmethod
    def to_node(cls, val: int | str | SymNode | SymInt) -> SymNode:
        """Convert an int, str, SymNode, or SymInt into a SymNode.

        Args:
            val (int | str | SymNode | SymInt): Value to convert.

        Returns:
            SymNode: Structured symbolic expression node.

        Raises:
            TypeError: If val cannot be converted into a SymNode.
        """
        if isinstance(val, SymNode):
            return val
        if isinstance(val, SymInt):
            return val.node
        if isinstance(val, int):
            return SymConst(val)
        if isinstance(val, str):
            return _parse_sym_str(val)
        raise TypeError(f"Cannot convert {type(val)} to SymNode")

    def simplify(self) -> SymNode:
        """Apply algebraic simplification rules recursively.

        Returns:
            SymNode: Simplified expression tree.
        """
        return self

    def canonical(self) -> SymNode:
        """Reduce expression to canonical polynomial form.

        Returns:
            SymNode: Canonical expression tree.
        """
        poly = self.to_polynomial()
        if poly is not None:
            return _poly_to_symnode(poly)
        return self.simplify()

    def to_polynomial(self) -> Polynomial | None:
        """Convert node to canonical Polynomial representation if possible.

        Returns:
            Polynomial | None: Canonical polynomial or None if non-polynomial.
        """
        return None

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate expression given variable bindings.

        Args:
            env (dict[str, int]): Variable assignment dictionary.

        Returns:
            int: Evaluated integer result.

        Raises:
            KeyError: If an unbound variable is encountered.
            ZeroDivisionError: If dividing or taking modulo by zero.
            NotImplementedError: If evaluated on an unspecialized base node.
        """
        poly = self.to_polynomial()
        if poly is not None:
            return poly.eval(env)
        simplified = self.simplify()
        if simplified is not self:
            return simplified.eval(env)
        raise NotImplementedError(
            f"Evaluation not implemented for node of type {type(self).__name__}"
        )

    def evaluate(self, bindings: dict[str, int]) -> int:
        """Evaluate expression given runtime variable bindings.

        Args:
            bindings (dict[str, int]): Variable assignment dictionary.

        Returns:
            int: Evaluated integer result.
        """
        return self.eval(bindings)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        """Generate Pydantic core schema for SymNode serializing as string or integer.

        Args:
            source_type (Any): Source type.
            handler (Any): Schema generation handler.

        Returns:
            Any: Pydantic core schema representation.
        """
        try:
            from pydantic_core import core_schema

            return core_schema.union_schema(
                [
                    core_schema.int_schema(),
                    core_schema.str_schema(),
                    core_schema.is_instance_schema(cls),
                ]
            )
        except ImportError:  # pragma: no cover
            return None

    @property
    def is_constant(self) -> bool:
        """Return True if node is completely constant and contains no free variables.

        Returns:
            bool: True if constant.
        """
        return len(self.free_vars()) == 0

    def free_vars(self) -> set[str]:
        """Return set of free variable names in expression.

        Returns:
            set[str]: Set of variable name strings.
        """
        return set()

    def __add__(self, other: int | str | SymNode) -> SymBinaryOp:
        """Add two symbolic expressions.

        Args:
            other (int | str | SymNode): Operand to add.

        Returns:
            SymBinaryOp: Addition expression node.
        """
        return SymBinaryOp("+", self, SymNode.to_node(other))

    def __radd__(self, other: int | str) -> SymBinaryOp:
        """Right-add two symbolic expressions.

        Args:
            other (int | str): Left operand.

        Returns:
            SymBinaryOp: Addition expression node.
        """
        return SymBinaryOp("+", SymNode.to_node(other), self)

    def __sub__(self, other: int | str | SymNode) -> SymBinaryOp:
        """Subtract symbolic expressions.

        Args:
            other (int | str | SymNode): Operand to subtract.

        Returns:
            SymBinaryOp: Subtraction expression node.
        """
        return SymBinaryOp("-", self, SymNode.to_node(other))

    def __rsub__(self, other: int | str) -> SymBinaryOp:
        """Right-subtract symbolic expressions.

        Args:
            other (int | str): Left operand.

        Returns:
            SymBinaryOp: Subtraction expression node.
        """
        return SymBinaryOp("-", SymNode.to_node(other), self)

    def __mul__(self, other: int | str | SymNode) -> SymBinaryOp:
        """Multiply symbolic expressions.

        Args:
            other (int | str | SymNode): Operand to multiply.

        Returns:
            SymBinaryOp: Multiplication expression node.
        """
        return SymBinaryOp("*", self, SymNode.to_node(other))

    def __rmul__(self, other: int | str) -> SymBinaryOp:
        """Right-multiply symbolic expressions.

        Args:
            other (int | str): Left operand.

        Returns:
            SymBinaryOp: Multiplication expression node.
        """
        return SymBinaryOp("*", SymNode.to_node(other), self)

    def __floordiv__(self, other: int | str | SymNode) -> SymBinaryOp:
        """Floor-divide symbolic expressions.

        Args:
            other (int | str | SymNode): Divisor operand.

        Returns:
            SymBinaryOp: Floor division expression node.
        """
        return SymBinaryOp("//", self, SymNode.to_node(other))

    def __rfloordiv__(self, other: int | str) -> SymBinaryOp:
        """Right-floor-divide symbolic expressions.

        Args:
            other (int | str): Dividend operand.

        Returns:
            SymBinaryOp: Floor division expression node.
        """
        return SymBinaryOp("//", SymNode.to_node(other), self)

    def __mod__(self, other: int | str | SymNode) -> SymBinaryOp:
        """Modulo symbolic expressions.

        Args:
            other (int | str | SymNode): Divisor operand.

        Returns:
            SymBinaryOp: Modulo expression node.
        """
        return SymBinaryOp("%", self, SymNode.to_node(other))

    def __rmod__(self, other: int | str) -> SymBinaryOp:
        """Right-modulo symbolic expressions.

        Args:
            other (int | str): Dividend operand.

        Returns:
            SymBinaryOp: Modulo expression node.
        """
        return SymBinaryOp("%", SymNode.to_node(other), self)

    def __pow__(self, other: int | str | SymNode) -> SymBinaryOp:
        """Exponentiate symbolic expression.

        Args:
            other (int | str | SymNode): Exponent operand.

        Returns:
            SymBinaryOp: Power expression node.
        """
        return SymBinaryOp("**", self, SymNode.to_node(other))

    def __rpow__(self, other: int | str) -> SymBinaryOp:
        """Right-exponentiate symbolic expression.

        Args:
            other (int | str): Base operand.

        Returns:
            SymBinaryOp: Power expression node.
        """
        return SymBinaryOp("**", SymNode.to_node(other), self)

    def __neg__(self) -> SymUnaryOp:
        """Negate symbolic expression.

        Returns:
            SymUnaryOp: Negated expression node.
        """
        return SymUnaryOp("-", self)

    def __abs__(self) -> SymUnaryOp:
        """Absolute value of symbolic expression.

        Returns:
            SymUnaryOp: Absolute value node.
        """
        return SymUnaryOp("abs", self)


class SymConst(SymNode):
    """Constant integer symbolic node.

    Attributes:
        value (int): Integer constant value.
    """

    def __init__(self, value: int) -> None:
        """Initialize SymConst.

        Args:
            value (int): Constant integer value.
        """
        self.value: int = int(value)

    def simplify(self) -> SymConst:
        """Simplify constant node.

        Returns:
            SymConst: Self.
        """
        return self

    def to_polynomial(self) -> Polynomial:
        """Convert constant to polynomial.

        Returns:
            Polynomial: Constant polynomial.
        """
        return Polynomial.from_const(self.value)

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate constant value.

        Args:
            env (dict[str, int]): Variable bindings.

        Returns:
            int: Constant integer value.
        """
        return self.value

    def free_vars(self) -> set[str]:
        """Return free variables for constant.

        Returns:
            set[str]: Empty set.
        """
        return set()

    def __int__(self) -> int:
        """Convert constant to integer.

        Returns:
            int: Integer constant value.
        """
        return self.value

    def __str__(self) -> str:
        """Return string representation of constant.

        Returns:
            str: String of integer value.
        """
        return str(self.value)

    def __repr__(self) -> str:
        """Return debugging representation.

        Returns:
            str: Representation string.
        """
        return f"SymConst({self.value})"

    def __eq__(self, other: object) -> bool:
        """Check equality with constant.

        Args:
            other (object): Other object.

        Returns:
            bool: True if values match.
        """
        if isinstance(other, SymConst):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        return False

    def __hash__(self) -> int:
        """Return hash for constant.

        Returns:
            int: Hash integer.
        """
        return hash(self.value)


class SymVar(SymNode):
    """Symbolic variable node representing a dynamic dimension.

    Attributes:
        name (str): Identifier name of the variable.
    """

    def __init__(self, name: str) -> None:
        """Initialize SymVar.

        Args:
            name (str): Variable name.
        """
        self.name: str = str(name)

    def simplify(self) -> SymVar:
        """Simplify variable node.

        Returns:
            SymVar: Self.
        """
        return self

    def to_polynomial(self) -> Polynomial:
        """Convert variable to polynomial.

        Returns:
            Polynomial: Variable polynomial.
        """
        return Polynomial.from_var(self.name)

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate variable given environment.

        Args:
            env (dict[str, int]): Environment dictionary.

        Returns:
            int: Bound value.

        Raises:
            KeyError: If variable is not bound.
        """
        if self.name not in env:
            raise KeyError(f"Variable '{self.name}' not bound in environment.")
        return env[self.name]

    def free_vars(self) -> set[str]:
        """Return set containing variable name.

        Returns:
            set[str]: Set with variable name.
        """
        return {self.name}

    def __str__(self) -> str:
        """Return variable name string.

        Returns:
            str: Name.
        """
        return self.name

    def __repr__(self) -> str:
        """Return debugging representation.

        Returns:
            str: Representation string.
        """
        return f"SymVar({self.name!r})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another variable.

        Args:
            other (object): Other object.

        Returns:
            bool: True if variable names match.
        """
        if isinstance(other, SymVar):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        """Compute hash for variable.

        Returns:
            int: Hash integer.
        """
        return hash(self.name)


class SymBinaryOp(SymNode):
    """Binary operation node in symbolic expression tree.

    Attributes:
        op (str): Binary operator (+, -, *, //, %, **, min, max).
        left (SymNode): Left operand node.
        right (SymNode): Right operand node.
    """

    def __init__(self, op: str, left: SymNode, right: SymNode) -> None:
        """Initialize SymBinaryOp.

        Args:
            op (str): Operator string.
            left (SymNode): Left child node.
            right (SymNode): Right child node.
        """
        self.op: str = str(op)
        self.left: SymNode = left
        self.right: SymNode = right

    def simplify(self) -> SymNode:
        """Apply algebraic rewrite rules to simplify binary operation.

        Returns:
            SymNode: Simplified expression node.
        """
        s_left = self.left.simplify()
        s_right = self.right.simplify()

        if isinstance(s_left, SymConst) and isinstance(s_right, SymConst):
            v1, v2 = s_left.value, s_right.value
            if self.op == "+":
                return SymConst(v1 + v2)
            if self.op == "-":
                return SymConst(v1 - v2)
            if self.op == "*":
                return SymConst(v1 * v2)
            if self.op == "//" and v2 != 0:
                return SymConst(v1 // v2)
            if self.op == "%" and v2 != 0:
                return SymConst(v1 % v2)
            if self.op == "**":
                return SymConst(int(v1**v2))
            if self.op == "min":
                return SymConst(min(v1, v2))
            if self.op == "max":
                return SymConst(max(v1, v2))

        if self.op == "+":
            if isinstance(s_right, SymConst) and s_right.value == 0:
                return s_left
            if isinstance(s_left, SymConst) and s_left.value == 0:
                return s_right

        elif self.op == "-":
            if isinstance(s_right, SymConst) and s_right.value == 0:
                return s_left
            if s_left == s_right:
                return SymConst(0)
            if isinstance(s_left, SymBinaryOp) and s_left.op == "+":
                if s_left.right == s_right:
                    return s_left.left
                if s_left.left == s_right:
                    return s_left.right

        elif self.op == "*":
            if isinstance(s_right, SymConst):
                if s_right.value == 1:
                    return s_left
                if s_right.value == 0:
                    return SymConst(0)
            if isinstance(s_left, SymConst):
                if s_left.value == 1:
                    return s_right
                if s_left.value == 0:
                    return SymConst(0)

        elif self.op == "//":
            if isinstance(s_right, SymConst) and s_right.value == 1:
                return s_left
            if isinstance(s_left, SymConst) and s_left.value == 0:
                return SymConst(0)
            if s_left == s_right and not (
                isinstance(s_right, SymConst) and s_right.value == 0
            ):
                return SymConst(1)
            if isinstance(s_left, SymBinaryOp) and s_left.op == "*":
                if s_left.right == s_right:
                    return s_left.left
                if s_left.left == s_right:
                    return s_left.right
                if isinstance(s_right, SymConst) and s_right.value != 0:
                    if (
                        isinstance(s_left.right, SymConst)
                        and s_left.right.value % s_right.value == 0
                    ):
                        return SymBinaryOp(
                            "*",
                            s_left.left,
                            SymConst(s_left.right.value // s_right.value),
                        ).simplify()
                    if (
                        isinstance(s_left.left, SymConst)
                        and s_left.left.value % s_right.value == 0
                    ):
                        return SymBinaryOp(
                            "*",
                            s_left.right,
                            SymConst(s_left.left.value // s_right.value),
                        ).simplify()

        elif self.op == "%":
            if isinstance(s_right, SymConst) and s_right.value == 1:
                return SymConst(0)
            if isinstance(s_left, SymConst) and s_left.value == 0:
                return SymConst(0)
            if s_left == s_right and not (
                isinstance(s_right, SymConst) and s_right.value == 0
            ):
                return SymConst(0)

        elif self.op == "**":
            if isinstance(s_right, SymConst):
                if s_right.value == 1:
                    return s_left
                if s_right.value == 0:
                    return SymConst(1)
            if isinstance(s_left, SymConst):
                if s_left.value == 0:
                    return SymConst(0)
                if s_left.value == 1:
                    return SymConst(1)

        elif self.op == "min":
            if s_left == s_right:
                return s_left
            if (
                isinstance(s_left, SymBinaryOp)
                and s_left.op == "min"
                and (s_left.left == s_right or s_left.right == s_right)
            ):
                return s_left
            if (
                isinstance(s_right, SymBinaryOp)
                and s_right.op == "min"
                and (s_right.left == s_left or s_right.right == s_left)
            ):
                return s_right

        elif self.op == "max":
            if s_left == s_right:
                return s_left
            if (
                isinstance(s_left, SymBinaryOp)
                and s_left.op == "max"
                and (s_left.left == s_right or s_left.right == s_right)
            ):
                return s_left
            if (
                isinstance(s_right, SymBinaryOp)
                and s_right.op == "max"
                and (s_right.left == s_left or s_right.right == s_left)
            ):
                return s_right

        poly = self.to_polynomial()
        if poly is not None:
            return _poly_to_symnode(poly)

        return SymBinaryOp(self.op, s_left, s_right)

    def to_polynomial(self) -> Polynomial | None:
        """Convert binary operation to canonical Polynomial representation.

        Returns:
            Polynomial | None: Resulting polynomial or None if irreducible.
        """
        p_left = self.left.to_polynomial()
        p_right = self.right.to_polynomial()
        if p_left is None or p_right is None:
            return None

        if self.op == "+":
            return p_left.add(p_right)
        if self.op == "-":
            return p_left.sub(p_right)
        if self.op == "*":
            return p_left.mul(p_right)
        if self.op == "//":
            return p_left.div_exact(p_right)
        return None

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate binary operation given variable bindings.

        Args:
            env (dict[str, int]): Environment dictionary.

        Returns:
            int: Evaluated integer result.

        Raises:
            ZeroDivisionError: If dividing or taking modulo by zero.
            ValueError: If operator is unsupported.
        """
        v1 = self.left.eval(env)
        v2 = self.right.eval(env)
        if self.op == "+":
            return v1 + v2
        if self.op == "-":
            return v1 - v2
        if self.op == "*":
            return v1 * v2
        if self.op == "//":
            if v2 == 0:
                raise ZeroDivisionError("Division by zero in symbolic evaluation.")
            return v1 // v2
        if self.op == "%":
            if v2 == 0:
                raise ZeroDivisionError("Modulo by zero in symbolic evaluation.")
            return v1 % v2
        if self.op == "**":
            return int(v1**v2)
        if self.op == "min":
            return min(v1, v2)
        if self.op == "max":
            return max(v1, v2)
        if self.op == "==":
            return int(v1 == v2)
        if self.op == "!=":
            return int(v1 != v2)
        if self.op == "<":
            return int(v1 < v2)
        if self.op == "<=":
            return int(v1 <= v2)
        if self.op == ">":
            return int(v1 > v2)
        if self.op == ">=":
            return int(v1 >= v2)
        raise ValueError(f"Unsupported operator '{self.op}'.")

    def free_vars(self) -> set[str]:
        """Return free variables of both operands.

        Returns:
            set[str]: Union of free variables.
        """
        return self.left.free_vars() | self.right.free_vars()

    def __str__(self) -> str:
        """Format binary operation as a string with parentheses.

        Returns:
            str: Parenthesized string expression.
        """
        return f"({self.left} {self.op} {self.right})"

    def __repr__(self) -> str:
        """Return debugging representation.

        Returns:
            str: Representation string.
        """
        return f"SymBinaryOp({self.op!r}, {self.left!r}, {self.right!r})"

    def __eq__(self, other: object) -> bool:
        """Check structural equality with another binary operation node.

        Args:
            other (object): Other object.

        Returns:
            bool: True if other is a SymBinaryOp with identical op, left, and right operands.
        """
        if not isinstance(other, SymBinaryOp):
            return False
        return (
            self.op == other.op
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self) -> int:
        """Compute hash for binary operation.

        Returns:
            int: Hash integer based on op, left, and right operands.
        """
        return hash((self.op, self.left, self.right))


class SymUnaryOp(SymNode):
    """Unary operation node in symbolic expression tree (abs, neg).

    Attributes:
        op (str): Unary operator name.
        operand (SymNode): Target operand node.
    """

    def __init__(self, op: str, operand: SymNode) -> None:
        """Initialize SymUnaryOp.

        Args:
            op (str): Operator string ('abs', '-').
            operand (SymNode): Child operand node.
        """
        self.op: str = str(op)
        self.operand: SymNode = operand

    def simplify(self) -> SymNode:
        """Simplify unary operation.

        Returns:
            SymNode: Simplified expression node.
        """
        s_op = self.operand.simplify()
        if isinstance(s_op, SymConst):
            v = s_op.value
            if self.op == "abs":
                return SymConst(abs(v))
            if self.op == "-":
                return SymConst(-v)
        if isinstance(s_op, SymUnaryOp):
            if self.op == "-" and s_op.op == "-":
                return s_op.operand
            if self.op == "abs" and s_op.op == "abs":
                return s_op
        return SymUnaryOp(self.op, s_op)

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate unary operation with variable bindings.

        Args:
            env (dict[str, int]): Variable assignment dictionary.

        Returns:
            int: Evaluated integer value.

        Raises:
            ValueError: If operator is unsupported.
        """
        val = self.operand.eval(env)
        if self.op == "abs":
            return abs(val)
        if self.op == "-":
            return -val
        raise ValueError(f"Unsupported unary operator '{self.op}'.")

    def free_vars(self) -> set[str]:
        """Return free variables of the operand.

        Returns:
            set[str]: Set of variable names.
        """
        return self.operand.free_vars()

    def __str__(self) -> str:
        """Return string representation.

        Returns:
            str: Operator and operand string.
        """
        if self.op == "-":
            return f"-({self.operand})"
        return f"{self.op}({self.operand})"

    def __repr__(self) -> str:
        """Return debugging string representation.

        Returns:
            str: Debugging string.
        """
        return f"SymUnaryOp({self.op!r}, {self.operand!r})"

    def __eq__(self, other: object) -> bool:
        """Check structural equality.

        Args:
            other (object): Other object.

        Returns:
            bool: True if op and operand match.
        """
        if not isinstance(other, SymUnaryOp):
            return False
        return self.op == other.op and self.operand == other.operand

    def __hash__(self) -> int:
        """Compute hash for unary operation.

        Returns:
            int: Hash integer based on op and operand.
        """
        return hash((self.op, self.operand))


class SymPiecewise(SymNode):
    """Piecewise conditional symbolic node.

    Attributes:
        cases (list[tuple[object, SymNode]]): Sequence of (condition, expr) pairs.
        default (SymNode): Default fallback expression node.
    """

    def __init__(
        self,
        cases: list[tuple[object, SymNode]] | None = None,
        default: SymNode | None = None,
        conditions: list[tuple[object, SymNode]] | None = None,
    ) -> None:
        """Initialize SymPiecewise.

        Args:
            cases (list[tuple[object, SymNode]] | None): List of (condition, expr_node).
            default (SymNode | None): Fallback node when no conditions are met.
            conditions (list[tuple[object, SymNode]] | None): Alias for cases.
        """
        raw_cases = cases if cases is not None else (conditions or [])
        self.cases: list[tuple[object, SymNode]] = list(raw_cases)
        self.default: SymNode = default if default is not None else SymConst(0)

    def simplify(self) -> SymNode:
        """Simplify piecewise expression.

        Returns:
            SymNode: Simplified expression node.
        """
        simplified_cases: list[tuple[object, SymNode]] = []
        for cond, expr in self.cases:
            if isinstance(cond, SymNode):
                cond = cond.simplify()
            simplified_cases.append((cond, expr.simplify()))
        return SymPiecewise(simplified_cases, self.default.simplify())

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate piecewise expression given environment bindings.

        Args:
            env (dict[str, int]): Variable bindings dictionary.

        Returns:
            int: Evaluated integer result from the first matching branch or default.
        """
        for cond, expr in self.cases:
            is_met = False
            if callable(cond):
                is_met = bool(cond(env))
            elif isinstance(cond, SymNode):
                is_met = bool(cond.eval(env))
            elif isinstance(cond, bool):
                is_met = cond
            elif isinstance(cond, str):
                parsed = _parse_sym_str(cond)
                is_met = bool(parsed.eval(env))
            if is_met:
                return expr.eval(env)
        return self.default.eval(env)

    def free_vars(self) -> set[str]:
        """Return union of free variables across all branches.

        Returns:
            set[str]: Set of variable names.
        """
        vars_set = set(self.default.free_vars())
        for cond, expr in self.cases:
            if isinstance(cond, SymNode):
                vars_set.update(cond.free_vars())
            vars_set.update(expr.free_vars())
        return vars_set

    def __str__(self) -> str:
        """Return string representation of piecewise node.

        Returns:
            str: Piecewise string representation.
        """
        return f"Piecewise({self.cases}, default={self.default})"

    def __repr__(self) -> str:
        """Return debugging string representation.

        Returns:
            str: Debugging string.
        """
        return f"SymPiecewise(cases={self.cases!r}, default={self.default!r})"

    def __eq__(self, other: object) -> bool:
        """Check structural equality.

        Args:
            other (object): Other object.

        Returns:
            bool: True if cases and default match.
        """
        if not isinstance(other, SymPiecewise):
            return False
        return self.cases == other.cases and self.default == other.default

    def __hash__(self) -> int:
        """Compute hash code.

        Returns:
            int: Hash integer.
        """
        return hash((tuple(self.cases), self.default))


class SymInt:
    """Symbolic Integer wrapper with Python arithmetic dunders for dynamic dimensions.

    Attributes:
        node (SymNode): Underlying symbolic expression tree node.
        expr (str): Expression or identifier string.
    """

    def __init__(self, name_or_expr: str | int | SymNode | SymInt) -> None:
        """Initialize SymInt.

        Args:
            name_or_expr (str | int | SymNode | SymInt): The variable name, expression,
                integer, or underlying SymNode.
        """
        if isinstance(name_or_expr, SymInt):
            self.node: SymNode = name_or_expr.node
            self.expr: str = name_or_expr.expr
        elif isinstance(name_or_expr, SymNode):
            self.node = name_or_expr
            self.expr = str(name_or_expr)
        elif isinstance(name_or_expr, int):
            self.node = SymConst(name_or_expr)
            self.expr = str(name_or_expr)
        else:
            str_val = str(name_or_expr)
            self.node = SymNode.to_node(str_val)
            self.expr = str_val

    @property
    def name(self) -> str:
        """Return the symbolic variable identifier or expression string.

        Returns:
            str: Identifier or expression string.
        """
        return self.expr

    def simplify(self) -> SymInt:
        """Return simplified SymInt instance.

        Returns:
            SymInt: Simplified expression.
        """
        return SymInt(self.node.simplify())

    def canonical(self) -> SymInt:
        """Return canonicalized SymInt instance.

        Returns:
            SymInt: Canonical polynomial reduced expression.
        """
        return SymInt(self.node.canonical())

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate symbolic expression given environment bindings.

        Args:
            env (dict[str, int]): Variable bindings.

        Returns:
            int: Evaluated integer value.
        """
        return self.node.eval(env)

    def evaluate(self, bindings: dict[str, int]) -> int:
        """Evaluate symbolic expression given runtime bindings.

        Args:
            bindings (dict[str, int]): Variable assignment dictionary.

        Returns:
            int: Evaluated integer value.
        """
        return self.node.evaluate(bindings)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        """Generate Pydantic core schema for SymInt serializing as string or integer.

        Args:
            source_type (Any): Source type.
            handler (Any): Schema generation handler.

        Returns:
            Any: Pydantic core schema representation.
        """
        try:
            from pydantic_core import core_schema

            return core_schema.union_schema(
                [
                    core_schema.int_schema(),
                    core_schema.str_schema(),
                    core_schema.is_instance_schema(cls),
                ]
            )
        except ImportError:  # pragma: no cover
            return None

    @property
    def is_constant(self) -> bool:
        """Check if wrapped expression is completely constant.

        Returns:
            bool: True if constant.
        """
        return self.node.is_constant

    def __add__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate addition.

        Args:
            other (SymInt | int | SymNode | str): Operand to add.

        Returns:
            SymInt: Result of addition.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("+", self.node, other_node))

    def __radd__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right addition.

        Args:
            other (int | SymNode | str): Operand to add.

        Returns:
            SymInt: Result of addition.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("+", other_node, self.node))

    def __sub__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate subtraction.

        Args:
            other (SymInt | int | SymNode | str): Operand to subtract.

        Returns:
            SymInt: Result of subtraction.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("-", self.node, other_node))

    def __rsub__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right subtraction.

        Args:
            other (int | SymNode | str): Operand to subtract.

        Returns:
            SymInt: Result of subtraction.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("-", other_node, self.node))

    def __mul__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate multiplication.

        Args:
            other (SymInt | int | SymNode | str): Operand to multiply.

        Returns:
            SymInt: Result of multiplication.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("*", self.node, other_node))

    def __rmul__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right multiplication.

        Args:
            other (int | SymNode | str): Operand to multiply.

        Returns:
            SymInt: Result of multiplication.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("*", other_node, self.node))

    def __floordiv__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate floor division.

        Args:
            other (SymInt | int | SymNode | str): Divisor operand.

        Returns:
            SymInt: Result of floor division.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("//", self.node, other_node))

    def __rfloordiv__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right floor division.

        Args:
            other (int | SymNode | str): Dividend operand.

        Returns:
            SymInt: Result of floor division.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("//", other_node, self.node))

    def __mod__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate modulo operation.

        Args:
            other (SymInt | int | SymNode | str): Divisor operand.

        Returns:
            SymInt: Result of modulo operation.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("%", self.node, other_node))

    def __rmod__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right modulo operation.

        Args:
            other (int | SymNode | str): Dividend operand.

        Returns:
            SymInt: Result of modulo operation.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("%", other_node, self.node))

    def __pow__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate exponentiation.

        Args:
            other (SymInt | int | SymNode | str): Exponent operand.

        Returns:
            SymInt: Result of exponentiation.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("**", self.node, other_node))

    def __rpow__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right exponentiation.

        Args:
            other (int | SymNode | str): Base operand.

        Returns:
            SymInt: Result of exponentiation.
        """
        other_node = SymNode.to_node(other)
        return SymInt(SymBinaryOp("**", other_node, self.node))

    def __neg__(self) -> SymInt:
        """Evaluate negation.

        Returns:
            SymInt: Negated expression.
        """
        return SymInt(SymUnaryOp("-", self.node))

    def __abs__(self) -> SymInt:
        """Evaluate absolute value.

        Returns:
            SymInt: Absolute value expression.
        """
        return SymInt(SymUnaryOp("abs", self.node))

    def __str__(self) -> str:
        """Return string expression.

        Returns:
            str: String expression.
        """
        return str(self.node)

    def __repr__(self) -> str:
        """Return debugging representation.

        Returns:
            str: Debugging representation.
        """
        return f"SymInt({self.node})"

    def __hash__(self) -> int:
        """Return hash integer.

        Returns:
            int: Hash integer.
        """
        return hash(self.expr)

    def __eq__(self, other: object) -> bool:
        """Evaluate equality using canonical polynomial consistency.

        Args:
            other (object): The other object to compare.

        Returns:
            bool: True if expressions are mathematically consistent.
        """
        if isinstance(other, (SymInt, SymNode, int, str)):
            return SymbolicSolver.is_consistent(self, other)
        return False

    def __int__(self) -> int:
        """Convert constant SymInt to integer.

        Returns:
            int: Integer constant value.

        Raises:
            TypeError: If SymInt is dynamic and contains free variables.
        """
        if self.is_constant:
            return self.node.eval({})
        raise TypeError(f"Cannot convert dynamic SymInt '{self.expr}' to concrete int.")


class SymbolicSolver:
    """Symbolic Expression Solver to validate shape consistency."""

    @staticmethod
    def is_consistent(
        expr1: SymInt | SymNode | int | str,
        expr2: SymInt | SymNode | int | str,
    ) -> bool:
        """Check if two symbolic expressions are mathematically equivalent.

        Args:
            expr1 (SymInt | SymNode | int | str): First expression.
            expr2 (SymInt | SymNode | int | str): Second expression.

        Returns:
            bool: True if expressions evaluate to identical canonical representations.
        """
        if isinstance(expr1, int) and isinstance(expr2, int):
            return expr1 == expr2

        n1 = SymNode.to_node(expr1)
        n2 = SymNode.to_node(expr2)

        if str(n1) == str(n2):
            return True

        p1 = n1.to_polynomial()
        p2 = n2.to_polynomial()
        if p1 is not None and p2 is not None:
            return p1 == p2

        s1 = n1.simplify()
        s2 = n2.simplify()
        return str(s1) == str(s2)


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
            ShapeMismatchError: If the recorded equality causes a contradiction.
        """
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

        const_a: int | None = None
        const_b: int | None = None

        if isinstance(a, int):
            const_a = a
        elif isinstance(a, SymConst):
            const_a = a.value
        elif key_a.isdigit():
            const_a = int(key_a)
        elif root_a in self._const_map:
            const_a = self._const_map[root_a]

        if isinstance(b, int):
            const_b = b
        elif isinstance(b, SymConst):
            const_b = b.value
        elif key_b.isdigit():
            const_b = int(key_b)
        elif root_b in self._const_map:
            const_b = self._const_map[root_b]

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

        Raises:
            ShapeMismatchError: If the dimensions cannot be unified.
        """
        if (
            (isinstance(a, int) and a == 1)
            or (isinstance(a, SymConst) and a.value == 1)
            or str(a) == "1"
        ):
            return b
        if (
            (isinstance(b, int) and b == 1)
            or (isinstance(b, SymConst) and b.value == 1)
            or str(b) == "1"
        ):
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

        Returns:
            dict[str, int]: Mapping of variable names to integer values.
        """
        solution: dict[str, int] = {}
        for var, root in list(self._parent.items()):
            if var.isdigit():
                continue
            actual_root = self._find(root)
            if actual_root in self._const_map:
                solution[var] = self._const_map[actual_root]
        return solution

    @overload
    def substitute(
        self, dim_or_shape: tuple[DimensionType, ...]
    ) -> tuple[DimensionType, ...]:
        """Substitute known resolved values into a shape tuple.

        Args:
            dim_or_shape (tuple[DimensionType, ...]): Input shape tuple.

        Returns:
            tuple[DimensionType, ...]: Substituted shape tuple.
        """

    @overload
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

    if isinstance(dim_a, (str, SymNode, SymInt)) and isinstance(
        dim_b, (str, SymNode, SymInt)
    ):
        return dim_a

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
        ValueError: If any axis index is out of [-rank, rank - 1] bounds.
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

    if isinstance(axis, int):
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
                Observed telemetry mapping node IDs to concrete shapes.
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
            if tracker is not None:
                for dim in concrete_tuple:
                    tracker.record_equality(dim, dim)

        return resolved

    @staticmethod
    def resolve_dynamic_bounds(
        graph: Any,
        feedback_telemetry: Mapping[str, Any],
        tracker: SymbolicConstraintTracker | None = None,
    ) -> dict[str, int]:
        """Resolve dynamic SymInt variables across a graph into static integer bounds using telemetry.

        Args:
            graph (Any): LogicalGraph instance containing nodes with symbolic dimensions.
            feedback_telemetry (Mapping[str, Any]):
                Observed runtime telemetry with concrete shapes.
            tracker (SymbolicConstraintTracker | None): Optional constraint tracker for unification.

        Returns:
            dict[str, int]: Mapping of symbolic dimension names to their resolved static integer bounds.
        """
        concrete_shapes = ShapeTracker.update_from_feedback(feedback_telemetry, tracker)
        resolved_bounds: dict[str, int] = {}

        raw_nodes = getattr(graph, "nodes", {})
        node_iterable = raw_nodes.values() if isinstance(raw_nodes, dict) else raw_nodes

        for node in node_iterable:
            node_id = str(getattr(node, "id", ""))
            if node_id not in concrete_shapes:
                continue

            observed_shape = concrete_shapes[node_id]
            shape_meta = getattr(node, "shape_metadata", None)
            if shape_meta is None:
                shape_meta = getattr(node, "shape", None)

            if isinstance(shape_meta, (tuple, list)):
                new_shape: list[int] = []
                for idx, dim in enumerate(shape_meta):
                    concrete_val = (
                        observed_shape[idx] if idx < len(observed_shape) else 1
                    )
                    if isinstance(dim, SymInt):
                        var_name = dim.name
                        resolved_bounds[var_name] = concrete_val
                        if tracker is not None:
                            tracker.record_equality(dim, concrete_val)
                    elif isinstance(dim, SymNode):
                        var_name = str(dim)
                        resolved_bounds[var_name] = concrete_val
                        if tracker is not None:
                            tracker.record_equality(dim, concrete_val)
                    elif isinstance(dim, str) and not dim.isdigit():
                        resolved_bounds[dim] = concrete_val
                        if tracker is not None:
                            tracker.record_equality(dim, concrete_val)
                    new_shape.append(concrete_val)

                if hasattr(node, "shape_metadata"):
                    node.shape_metadata = tuple(new_shape)
                if hasattr(node, "shape"):
                    node.shape = tuple(new_shape)

        return resolved_bounds
