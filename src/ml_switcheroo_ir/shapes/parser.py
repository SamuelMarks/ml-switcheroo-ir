"""Recursive-descent expression parser, tokenizer, and algebraic solver."""

from __future__ import annotations

from ml_switcheroo_ir.shapes.ast import (
    SymBinaryOp,
    SymConst,
    SymInt,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
)

__all__ = [
    "SymbolicExpressionParser",
    "SymbolicSolver",
    "parse_symbolic_expr",
    "tokenize_symbolic_expr",
]


class SymbolicExpressionParser:
    """Recursive-descent parser and tokenizer for symbolic dimension expressions.

    Attributes:
        tokens (list[str]): Sequence of lexical tokens to parse.
        idx (int): Current zero-based cursor index in ``tokens``.
    """

    def __init__(self, tokens: list[str]) -> None:
        """Initialize parser with a token stream.

        Args:
            tokens (list[str]): Sequence of lexical tokens.
        """
        self.tokens: list[str] = tokens
        self.idx: int = 0

    @staticmethod
    def _consume_alnum_span(s: str, i: int, n: int) -> tuple[str, int]:
        """Consume a contiguous numeric or identifier span starting at ``i``.

        Args:
            s (str): Expression string.
            i (int): Current zero-based character index.
            n (int): Total length of ``s``.

        Returns:
            tuple[str, int]: ``(token, next_index)``.

        Raises:
            ValueError: If an unrecognized character is encountered at index ``i``.
        """
        c = s[i]
        if c.isdigit():
            end = i
            while end < n and s[end].isdigit():
                end += 1
            return s[i:end], end
        if c.isalpha() or c == "_":
            end = i
            while end < n and (s[end].isalnum() or s[end] == "_"):
                end += 1
            return s[i:end], end
        raise ValueError(f"Unrecognized character {c!r} at position {i} in symbolic expression {s!r}.")

    @staticmethod
    def _consume_next_token(s: str, i: int, n: int) -> tuple[str | None, int]:
        """Consume a single token starting at index ``i`` in ``s``.

        Args:
            s (str): Expression string.
            i (int): Current index in ``s``.
            n (int): Total length of ``s``.

        Returns:
            tuple[str | None, int]: ``(token_or_none, next_index)``.

        Raises:
            ValueError: If an unrecognized character is encountered at index ``i``.
        """
        c = s[i]
        if c.isspace() or c in "(),+-%":
            return (None if c.isspace() else c), i + 1
        if c in ("^", "*"):
            step = 2 if (c == "*" and i + 1 < n and s[i + 1] == "*") else 1
            return ("**" if (c == "^" or step == 2) else "*"), i + step
        if c in ("<", ">", "=", "!"):
            if i + 1 < n and s[i + 1] == "=":
                return s[i : i + 2], i + 2
            if c in ("<", ">"):
                return c, i + 1
        if c == "/":
            return "//", (i + 2 if (i + 1 < n and s[i + 1] == "/") else i + 1)
        return SymbolicExpressionParser._consume_alnum_span(s, i, n)

    @staticmethod
    def tokenize(expr_str: str) -> list[str]:
        """Tokenize a symbolic expression string into component tokens.

        Supports arithmetic operators (``+``, ``-``, ``*``, ``//``, ``/``, ``%``,
        ``**``, ``^``), comparison operators (``==``, ``!=``, ``<``, ``<=``,
        ``>``, ``>=``), parentheses, commas, identifiers, and integer literals.
        Single ``/`` is normalized to integer floor division ``//`` and ``^`` is
        normalized to exponentiation ``**`` so canonical polynomial strings
        (e.g. ``x^2``) roundtrip losslessly.

        Args:
            expr_str (str): Input expression string.

        Returns:
            list[str]: Sequence of string tokens.

        Raises:
            ValueError: If an unrecognized character is encountered in ``expr_str``.
        """
        tokens: list[str] = []
        i = 0
        n = len(expr_str)
        while i < n:
            tok, i = SymbolicExpressionParser._consume_next_token(expr_str, i, n)
            if tok is not None:
                tokens.append(tok)
        return tokens

    def parse_comparison(self) -> SymNode:
        """Parse comparison expressions.

        Returns:
            SymNode: Comparison binary expression node or additive expression.
        """
        node = self.parse_expr()
        if self.idx < len(self.tokens) and self.tokens[self.idx] in (
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
        ):
            op = self.tokens[self.idx]
            self.idx += 1
            right = self.parse_expr()
            node = SymBinaryOp(op, node, right)
        return node

    def parse_expr(self) -> SymNode:
        """Parse additive binary expressions.

        Returns:
            SymNode: Root parsed expression node.
        """
        node = self.parse_term()
        while self.idx < len(self.tokens) and self.tokens[self.idx] in ("+", "-"):
            op = self.tokens[self.idx]
            self.idx += 1
            right = self.parse_term()
            node = SymBinaryOp(op, node, right)
        return node

    def parse_term(self) -> SymNode:
        """Parse multiplicative binary expressions.

        Returns:
            SymNode: Root parsed term node.
        """
        node = self.parse_factor()
        while self.idx < len(self.tokens) and self.tokens[self.idx] in (
            "*",
            "//",
            "%",
        ):
            op = self.tokens[self.idx]
            self.idx += 1
            right = self.parse_factor()
            node = SymBinaryOp(op, node, right)
        return node

    def _parse_call_args(self, func_name: str) -> list[SymNode]:
        """Consume parenthesized comma-separated arguments for ``func_name``.

        Args:
            func_name (str): Invoked function identifier.

        Returns:
            list[SymNode]: Parsed argument AST nodes.

        Raises:
            ValueError: If argument list is empty or missing closing ``')'``.
        """
        self.idx += 1  # consume '('
        if self.idx < len(self.tokens) and self.tokens[self.idx] == ")":
            raise ValueError(f"Function '{func_name}' requires at least one argument.")
        args: list[SymNode] = [self.parse_comparison()]
        while self.idx < len(self.tokens) and self.tokens[self.idx] == ",":
            self.idx += 1
            args.append(self.parse_comparison())
        if self.idx >= len(self.tokens) or self.tokens[self.idx] != ")":
            raise ValueError(f"Unclosed '(' in call to '{func_name}'.")
        self.idx += 1
        return args

    def _parse_func_call(self, func_name: str) -> SymNode:
        """Parse ``abs(x)``, ``min(a, b, ...)``, or ``max(a, b, ...)`` into ``SymUnaryOp`` or ``SymPiecewise``.

        Args:
            func_name (str): Name of the function (``'abs'``, ``'min'``, or ``'max'``).

        Returns:
            SymNode: Lowered ``SymUnaryOp`` or ``SymPiecewise`` node.

        Raises:
            ValueError: If ``func_name`` is unsupported, unclosed, or called with invalid arity.
        """
        if func_name not in ("abs", "min", "max"):
            raise ValueError(f"Unsupported function '{func_name}' in symbolic expression.")
        args = self._parse_call_args(func_name)
        if func_name == "abs":
            if len(args) != 1:
                raise ValueError(f"Function 'abs' expects 1 argument, got {len(args)}.")
            return SymUnaryOp("abs", args[0])
        if len(args) < 2:
            raise ValueError(f"Function '{func_name}' expects at least 2 arguments, got {len(args)}.")
        cmp_op = "<=" if func_name == "min" else ">="
        acc = args[0]
        for next_arg in args[1:]:
            acc = SymPiecewise(
                cases=[(SymBinaryOp(cmp_op, acc, next_arg), acc)],
                default=next_arg,
            )
        return acc

    def _maybe_parse_exponent(self, base_node: SymNode) -> SymNode:
        """Parse optional right-associative ``**`` exponentiation suffix on ``base_node``.

        Args:
            base_node (SymNode): Base expression node.

        Returns:
            SymNode: Exponentiated node if followed by ``**``, else ``base_node``.
        """
        if self.idx < len(self.tokens) and self.tokens[self.idx] == "**":
            self.idx += 1
            exponent = self.parse_factor()
            return SymBinaryOp("**", base_node, exponent)
        return base_node

    def _parse_atom(self, tok: str) -> SymNode:
        """Parse an atomic literal, identifier, or function call starting with ``tok``.

        Args:
            tok (str): Leading token already consumed at ``self.idx - 1``.

        Returns:
            SymNode: Parsed atomic expression node.

        Raises:
            ValueError: If ``tok`` is not a valid integer literal or identifier.
        """
        if tok.isdigit():
            return SymConst(int(tok))
        if tok.isidentifier():
            if self.idx < len(self.tokens) and self.tokens[self.idx] == "(":
                return self._parse_func_call(tok)
            return SymVar(tok)
        raise ValueError(f"Unexpected token {tok!r} in symbolic expression.")

    def parse_factor(self) -> SymNode:
        """Parse factor, power expression, function call, parenthesized expression, or unary operator.

        Returns:
            SymNode: Root parsed factor node.

        Raises:
            ValueError: If expression ends prematurely, has unclosed parentheses, or contains invalid syntax.
        """
        if self.idx >= len(self.tokens):
            raise ValueError("Unexpected end of symbolic expression.")
        tok = self.tokens[self.idx]
        if tok == "(":
            self.idx += 1
            node = self.parse_comparison()
            if self.idx >= len(self.tokens) or self.tokens[self.idx] != ")":
                raise ValueError("Unclosed '(' in symbolic expression.")
            self.idx += 1
            return self._maybe_parse_exponent(node)
        if tok == "+":
            self.idx += 1
            return self.parse_factor()
        if tok == "-":
            self.idx += 1
            inner = self.parse_factor()
            return SymBinaryOp("-", SymConst(0), inner)
        self.idx += 1
        base_node = self._parse_atom(tok)
        return self._maybe_parse_exponent(base_node)


def tokenize_symbolic_expr(expr_str: str) -> list[str]:
    """Tokenize a symbolic expression string into component tokens.

    Args:
        expr_str (str): Input expression string.

    Returns:
        list[str]: Sequence of string tokens.
    """
    return SymbolicExpressionParser.tokenize(expr_str)


def parse_symbolic_expr(expr_str: str) -> SymNode:
    """Parse an algebraic string expression into a structured ``SymNode``.

    Args:
        expr_str (str): String expression to parse.

    Returns:
        SymNode: Structured symbolic expression node.

    Raises:
        ValueError: If ``expr_str`` is empty, contains unrecognized characters, or is syntactically malformed.
    """
    tokens = tokenize_symbolic_expr(expr_str)
    if not tokens:
        raise ValueError(f"Empty symbolic expression: {expr_str!r}.")

    parser = SymbolicExpressionParser(tokens)
    node = parser.parse_comparison()
    if parser.idx != len(tokens):
        raise ValueError(f"Unexpected trailing token {tokens[parser.idx]!r} in symbolic expression {expr_str!r}.")
    return node


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
        if (
            isinstance(expr1, int)
            and not isinstance(expr1, bool)
            and isinstance(expr2, int)
            and not isinstance(expr2, bool)
        ):
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
        return s1 == s2 or str(s1) == str(s2)
