"""Canonical multivariate polynomials over integer coefficients for symbolic equivalence."""

from __future__ import annotations

import typing
import typing_extensions

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir.shapes.ast import SymNode

MonomialType: typing_extensions.TypeAlias = tuple[tuple[str, int], ...]

__all__ = [
    "MonomialType",
    "Polynomial",
    "_poly_to_symnode",
]


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
    from ml_switcheroo_ir.shapes.ast import SymBinaryOp, SymConst, SymVar

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
