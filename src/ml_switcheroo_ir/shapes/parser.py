"""Recursive-descent expression parser, tokenizer, and algebraic solver."""

from __future__ import annotations

from ml_switcheroo_ir.shapes.ast import (
    SymBinaryOp,
    SymConst,
    SymInt,
    SymNode,
    SymVar,
)

__all__ = [
    "SymbolicExpressionParser",
    "SymbolicSolver",
    "_ExprParser",
    "_parse_sym_str",
    "_tokenize_expr",
]


class SymbolicExpressionParser:
    """Recursive-descent parser and tokenizer for symbolic dimension expressions."""

    def __init__(self, tokens: list[str]) -> None:
        """Initialize parser with a token stream.

        Args:
            tokens (list[str]): Sequence of lexical tokens.
        """
        self.tokens: list[str] = tokens
        self.idx: int = 0
        self.failed: bool = False

    @staticmethod
    def tokenize(expr_str: str) -> list[str]:
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
            elif c in "()+*":
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
            elif c in ("-", "%"):
                tokens.append(c)
                i += 1
            elif c == "/":
                tokens.append("//")
                i += 2 if (i + 1 < n and s[i + 1] == "/") else 1
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

    def parse_factor(self) -> SymNode:
        """Parse factor, power expression, parenthesized expression, or unary negative.

        Returns:
            SymNode: Root parsed factor node.
        """
        if self.idx >= len(self.tokens):
            self.failed = True
            return SymConst(0)
        tok = self.tokens[self.idx]
        if tok == "(":
            self.idx += 1
            node = self.parse_comparison()
            if self.idx < len(self.tokens) and self.tokens[self.idx] == ")":
                self.idx += 1
            else:
                self.failed = True
            if self.idx < len(self.tokens) and self.tokens[self.idx] == "**":
                self.idx += 1
                exponent = self.parse_factor()
                node = SymBinaryOp("**", node, exponent)
            return node
        if tok == "-":
            self.idx += 1
            inner = self.parse_factor()
            return SymBinaryOp("-", SymConst(0), inner)
        self.idx += 1
        base_node: SymNode = SymConst(int(tok)) if tok.isdigit() else SymVar(tok)
        if self.idx < len(self.tokens) and self.tokens[self.idx] == "**":
            self.idx += 1
            exponent = self.parse_factor()
            base_node = SymBinaryOp("**", base_node, exponent)
        return base_node


_ExprParser = SymbolicExpressionParser


def _tokenize_expr(expr_str: str) -> list[str]:
    """Tokenize a symbolic expression string into component tokens.

    Args:
        expr_str (str): Input expression string.

    Returns:
        list[str]: Sequence of string tokens.
    """
    return SymbolicExpressionParser.tokenize(expr_str)


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

    parser = SymbolicExpressionParser(tokens)
    node = parser.parse_comparison()
    if not parser.failed and parser.idx == len(tokens):
        return node

    return SymVar(s)


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
