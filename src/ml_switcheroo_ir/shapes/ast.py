"""Symbolic expression AST nodes and SymInt algebraic wrapper."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import typing
import typing_extensions

import pydantic
import pydantic_core

from ml_switcheroo_ir.shapes.polynomial import Polynomial, _poly_to_symnode

__all__ = [
    "DimensionType",
    "ShapeType",
    "SymBinaryOp",
    "SymCondition",
    "SymConst",
    "SymInt",
    "SymNode",
    "SymPiecewise",
    "SymUnaryOp",
    "SymVar",
]


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
            from ml_switcheroo_ir.shapes.parser import _parse_sym_str

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
    def __get_pydantic_core_schema__(
        cls,
        source_type: type[SymNode],
        handler: pydantic.GetCoreSchemaHandler | None = None,
    ) -> pydantic_core.CoreSchema | None:
        """Generate Pydantic core schema for SymNode serializing as string or integer.

        Args:
            source_type (type[SymNode]): Source type.
            handler (pydantic.GetCoreSchemaHandler | None): Schema generation handler.

        Returns:
            pydantic_core.CoreSchema | None: Pydantic core schema representation.
        """
        del source_type, handler
        try:
            from pydantic_core import core_schema

            return core_schema.union_schema(
                [
                    core_schema.int_schema(),
                    core_schema.str_schema(),
                    core_schema.is_instance_schema(cls),
                ]
            )
        except ImportError:
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

    @staticmethod
    def _simplify_constants(
        op: str, s_left: SymConst, s_right: SymConst
    ) -> SymNode | None:
        """Fold two constant operands if the operator is statically evaluable.

        Args:
            op (str): Binary operator symbol.
            s_left (SymConst): Left constant operand.
            s_right (SymConst): Right constant operand.

        Returns:
            SymNode | None: Folded SymConst or None if division by zero or unknown op.
        """
        v1, v2 = s_left.value, s_right.value
        if op == "+":
            return SymConst(v1 + v2)
        if op == "-":
            return SymConst(v1 - v2)
        if op == "*":
            return SymConst(v1 * v2)
        if op == "//" and v2 != 0:
            return SymConst(v1 // v2)
        if op == "%" and v2 != 0:
            return SymConst(v1 % v2)
        if op == "**":
            return SymConst(int(v1**v2))
        if op == "min":
            return SymConst(min(v1, v2))
        if op == "max":
            return SymConst(max(v1, v2))
        return None

    @staticmethod
    def _simplify_add(s_left: SymNode, s_right: SymNode) -> SymNode | None:
        """Apply additive identity rules (x + 0 -> x, 0 + x -> x).

        Args:
            s_left (SymNode): Simplified left operand.
            s_right (SymNode): Simplified right operand.

        Returns:
            SymNode | None: Simplified node or None.
        """
        if isinstance(s_right, SymConst) and s_right.value == 0:
            return s_left
        if isinstance(s_left, SymConst) and s_left.value == 0:
            return s_right
        return None

    @staticmethod
    def _simplify_sub(s_left: SymNode, s_right: SymNode) -> SymNode | None:
        """Apply subtractive identity and cancellation rules.

        Args:
            s_left (SymNode): Simplified left operand.
            s_right (SymNode): Simplified right operand.

        Returns:
            SymNode | None: Simplified node or None.
        """
        if isinstance(s_right, SymConst) and s_right.value == 0:
            return s_left
        if s_left == s_right:
            return SymConst(0)
        if isinstance(s_left, SymBinaryOp) and s_left.op == "+":
            if s_left.right == s_right:
                return s_left.left
            if s_left.left == s_right:
                return s_left.right
        return None

    @staticmethod
    def _simplify_mul(s_left: SymNode, s_right: SymNode) -> SymNode | None:
        """Apply multiplicative identity and zero-annihilation rules.

        Args:
            s_left (SymNode): Simplified left operand.
            s_right (SymNode): Simplified right operand.

        Returns:
            SymNode | None: Simplified node or None.
        """
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
        return None

    @staticmethod
    def _simplify_floordiv(s_left: SymNode, s_right: SymNode) -> SymNode | None:
        """Apply floor-division identity and factor cancellation rules.

        Args:
            s_left (SymNode): Simplified left operand.
            s_right (SymNode): Simplified right operand.

        Returns:
            SymNode | None: Simplified node or None.
        """
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
        return None

    @staticmethod
    def _simplify_mod(s_left: SymNode, s_right: SymNode) -> SymNode | None:
        """Apply modulo identity rules.

        Args:
            s_left (SymNode): Simplified left operand.
            s_right (SymNode): Simplified right operand.

        Returns:
            SymNode | None: Simplified node or None.
        """
        if isinstance(s_right, SymConst) and s_right.value == 1:
            return SymConst(0)
        if isinstance(s_left, SymConst) and s_left.value == 0:
            return SymConst(0)
        if s_left == s_right and not (
            isinstance(s_right, SymConst) and s_right.value == 0
        ):
            return SymConst(0)
        return None

    @staticmethod
    def _simplify_pow(s_left: SymNode, s_right: SymNode) -> SymNode | None:
        """Apply exponentiation identity rules.

        Args:
            s_left (SymNode): Simplified left operand.
            s_right (SymNode): Simplified right operand.

        Returns:
            SymNode | None: Simplified node or None.
        """
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
        return None

    @staticmethod
    def _simplify_minmax(op: str, s_left: SymNode, s_right: SymNode) -> SymNode | None:
        """Apply idempotence and absorption rules for min and max operations.

        Args:
            op (str): Either 'min' or 'max'.
            s_left (SymNode): Simplified left operand.
            s_right (SymNode): Simplified right operand.

        Returns:
            SymNode | None: Simplified node or None.
        """
        if s_left == s_right:
            return s_left
        if (
            isinstance(s_left, SymBinaryOp)
            and s_left.op == op
            and (s_left.left == s_right or s_left.right == s_right)
        ):
            return s_left
        if (
            isinstance(s_right, SymBinaryOp)
            and s_right.op == op
            and (s_right.left == s_left or s_right.right == s_left)
        ):
            return s_right
        return None

    def simplify(self) -> SymNode:
        """Apply algebraic rewrite rules to simplify binary operation.

        Returns:
            SymNode: Simplified expression node.
        """
        s_left = self.left.simplify()
        s_right = self.right.simplify()

        if isinstance(s_left, SymConst) and isinstance(s_right, SymConst):
            folded = self._simplify_constants(self.op, s_left, s_right)
            if folded is not None:
                return folded

        op_handlers = {
            "+": self._simplify_add,
            "-": self._simplify_sub,
            "*": self._simplify_mul,
            "//": self._simplify_floordiv,
            "%": self._simplify_mod,
            "**": self._simplify_pow,
        }
        handler = op_handlers.get(self.op)
        if handler is not None:
            res = handler(s_left, s_right)
            if res is not None:
                return res
        elif self.op in ("min", "max"):
            res = self._simplify_minmax(self.op, s_left, s_right)
            if res is not None:
                return res

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
        cases (list[tuple[SymCondition, SymNode]]): Sequence of (condition, expr) pairs.
        default (SymNode): Default fallback expression node.
    """

    def __init__(
        self,
        cases: list[tuple[SymCondition, SymNode]] | None = None,
        default: SymNode | None = None,
    ) -> None:
        """Initialize SymPiecewise.

        Args:
            cases (list[tuple[SymCondition, SymNode]] | None): List of (condition, expr_node).
            default (SymNode | None): Fallback node when no conditions are met.
        """
        self.cases: list[tuple[SymCondition, SymNode]] = list(cases or [])
        self.default: SymNode = default if default is not None else SymConst(0)

    def simplify(self) -> SymNode:
        """Simplify piecewise expression.

        Returns:
            SymNode: Simplified expression node.
        """
        simplified_cases: list[tuple[SymCondition, SymNode]] = []
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

        Raises:
            TypeError: If a condition predicate is not a callable, SymNode, bool, or str.
        """
        from ml_switcheroo_ir.shapes.parser import _parse_sym_str

        for cond, expr in self.cases:
            if callable(cond):
                is_met = bool(cond(env))
            elif isinstance(cond, SymNode):
                is_met = bool(cond.eval(env))
            elif isinstance(cond, bool):
                is_met = cond
            elif isinstance(cond, str):
                parsed = _parse_sym_str(cond)
                is_met = bool(parsed.eval(env))
            else:
                raise TypeError(
                    f"Unsupported SymPiecewise condition type: {type(cond).__name__}"
                )
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
    def __get_pydantic_core_schema__(
        cls,
        source_type: type[SymInt],
        handler: pydantic.GetCoreSchemaHandler | None = None,
    ) -> pydantic_core.CoreSchema | None:
        """Generate Pydantic core schema for SymInt serializing as string or integer.

        Args:
            source_type (type[SymInt]): Source type.
            handler (pydantic.GetCoreSchemaHandler | None): Schema generation handler.

        Returns:
            pydantic_core.CoreSchema | None: Pydantic core schema representation.
        """
        del source_type, handler
        try:
            from pydantic_core import core_schema

            return core_schema.union_schema(
                [
                    core_schema.int_schema(),
                    core_schema.str_schema(),
                    core_schema.is_instance_schema(cls),
                ]
            )
        except ImportError:
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
        return SymInt(self.node + SymNode.to_node(other))

    def __radd__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right addition.

        Args:
            other (int | SymNode | str): Operand to add.

        Returns:
            SymInt: Result of addition.
        """
        return SymInt(SymNode.to_node(other) + self.node)

    def __sub__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate subtraction.

        Args:
            other (SymInt | int | SymNode | str): Operand to subtract.

        Returns:
            SymInt: Result of subtraction.
        """
        return SymInt(self.node - SymNode.to_node(other))

    def __rsub__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right subtraction.

        Args:
            other (int | SymNode | str): Operand to subtract.

        Returns:
            SymInt: Result of subtraction.
        """
        return SymInt(SymNode.to_node(other) - self.node)

    def __mul__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate multiplication.

        Args:
            other (SymInt | int | SymNode | str): Operand to multiply.

        Returns:
            SymInt: Result of multiplication.
        """
        return SymInt(self.node * SymNode.to_node(other))

    def __rmul__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right multiplication.

        Args:
            other (int | SymNode | str): Operand to multiply.

        Returns:
            SymInt: Result of multiplication.
        """
        return SymInt(SymNode.to_node(other) * self.node)

    def __floordiv__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate floor division.

        Args:
            other (SymInt | int | SymNode | str): Divisor operand.

        Returns:
            SymInt: Result of floor division.
        """
        return SymInt(self.node // SymNode.to_node(other))

    def __rfloordiv__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right floor division.

        Args:
            other (int | SymNode | str): Dividend operand.

        Returns:
            SymInt: Result of floor division.
        """
        return SymInt(SymNode.to_node(other) // self.node)

    def __mod__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate modulo operation.

        Args:
            other (SymInt | int | SymNode | str): Divisor operand.

        Returns:
            SymInt: Result of modulo operation.
        """
        return SymInt(self.node % SymNode.to_node(other))

    def __rmod__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right modulo operation.

        Args:
            other (int | SymNode | str): Dividend operand.

        Returns:
            SymInt: Result of modulo operation.
        """
        return SymInt(SymNode.to_node(other) % self.node)

    def __pow__(self, other: SymInt | int | SymNode | str) -> SymInt:
        """Evaluate exponentiation.

        Args:
            other (SymInt | int | SymNode | str): Exponent operand.

        Returns:
            SymInt: Result of exponentiation.
        """
        return SymInt(self.node ** SymNode.to_node(other))

    def __rpow__(self, other: int | SymNode | str) -> SymInt:
        """Evaluate right exponentiation.

        Args:
            other (int | SymNode | str): Base operand.

        Returns:
            SymInt: Result of exponentiation.
        """
        return SymInt(SymNode.to_node(other) ** self.node)

    def __neg__(self) -> SymInt:
        """Evaluate negation.

        Returns:
            SymInt: Negated expression.
        """
        return SymInt(-self.node)

    def __abs__(self) -> SymInt:
        """Evaluate absolute value.

        Returns:
            SymInt: Absolute value expression.
        """
        return SymInt(abs(self.node))

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
        """Return hash integer consistent with canonical equivalence.

        Returns:
            int: Hash integer.
        """
        canon = self.node.canonical()
        if isinstance(canon, SymConst):
            return hash(canon.value)
        return hash(str(canon))

    def __eq__(self, other: object) -> bool:
        """Evaluate equality using canonical polynomial consistency.

        Args:
            other (object): The other object to compare.

        Returns:
            bool: True if expressions are mathematically consistent.
        """
        if isinstance(other, bool):
            return False
        if isinstance(other, (SymInt, SymNode, int)):
            from ml_switcheroo_ir.shapes.parser import SymbolicSolver

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


DimensionType: typing_extensions.TypeAlias = typing.Union[int, str, SymNode, SymInt]
ShapeType: typing_extensions.TypeAlias = Sequence[DimensionType]
SymCondition: typing_extensions.TypeAlias = typing.Union[
    Callable[[dict[str, int]], bool], SymNode, bool, str
]
