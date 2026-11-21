"""First-order recursion scheme witnesses (EndoRewrite and RecursiveFold) without Higher-Kinded Types."""

from __future__ import annotations

import typing
from collections.abc import Callable
from dataclasses import dataclass

from ml_switcheroo_ir.typeclasses.algebra import Eq

_NodeT = typing.TypeVar("_NodeT")
_AccT = typing.TypeVar("_AccT")

__all__ = [
    "EndoRewrite",
    "RecursiveFold",
]


@dataclass(frozen=True)
class EndoRewrite(typing.Generic[_NodeT]):
    """First-order structural tree/DAG rewrite witness (Haskell ``Uniplate`` / ``Plated``).

    Encodes single-layer child mapping ``map_children`` so generic ``bottom_up``,
    ``top_down``, and bounded ``fixpoint`` rewrites work over closed recursive ADTs
    without requiring Higher-Kinded Types, ``Any``, or ``typing.cast``.

    Attributes:
        map_children (Callable[[_NodeT, Callable[[_NodeT], _NodeT]], _NodeT]): Single-layer child mapper.
    """

    map_children: Callable[[_NodeT, Callable[[_NodeT], _NodeT]], _NodeT]

    def bottom_up(self, node: _NodeT, rule: Callable[[_NodeT], _NodeT]) -> _NodeT:
        """Apply ``rule`` recursively from leaves up to the root (post-order rewrite).

        Args:
            node (_NodeT): Root AST/IR node.
            rule (Callable[[_NodeT], _NodeT]): Local single-step rewrite rule.

        Returns:
            _NodeT: Rewritten root node.
        """

        def _visit(curr: _NodeT) -> _NodeT:
            """Rewrite children first, then apply ``rule`` to the rebuilt node.

            Args:
                curr (_NodeT): Current node.

            Returns:
                _NodeT: Rewritten subtree.
            """
            rebuilt = self.map_children(curr, _visit)
            return rule(rebuilt)

        return _visit(node)

    def top_down(self, node: _NodeT, rule: Callable[[_NodeT], _NodeT]) -> _NodeT:
        """Apply ``rule`` recursively from the root down to the leaves (pre-order rewrite).

        Args:
            node (_NodeT): Root AST/IR node.
            rule (Callable[[_NodeT], _NodeT]): Local single-step rewrite rule.

        Returns:
            _NodeT: Rewritten root node.
        """

        def _visit(curr: _NodeT) -> _NodeT:
            """Apply ``rule`` to ``curr`` first, then recursively rewrite its children.

            Args:
                curr (_NodeT): Current node.

            Returns:
                _NodeT: Rewritten subtree.
            """
            transformed = rule(curr)
            return self.map_children(transformed, _visit)

        return _visit(node)

    def fixpoint(
        self,
        node: _NodeT,
        eq_witness: Eq[_NodeT],
        rule: Callable[[_NodeT], _NodeT],
        max_steps: int = 64,
    ) -> _NodeT:
        """Repeatedly apply ``bottom_up(curr, rule)`` until ``eq_witness`` stabilizes or ``max_steps`` is hit.

        Args:
            node (_NodeT): Initial node to normalize.
            eq_witness (Eq[_NodeT]): Equivalence witness detecting stabilization.
            rule (Callable[[_NodeT], _NodeT]): Local rewrite rule.
            max_steps (int): Maximum rewrite iterations (must be a non-negative non-boolean integer).

        Returns:
            _NodeT: Stabilized or step-bounded node.

        Raises:
            ValueError: If ``max_steps`` is a boolean or negative integer.
        """
        if isinstance(max_steps, bool) or max_steps < 0:
            raise ValueError(f"EndoRewrite.fixpoint requires non-negative int max_steps, got {max_steps!r}.")
        curr = node
        for _ in range(max_steps):
            nxt = self.bottom_up(curr, rule)
            if eq_witness.eq(curr, nxt):
                return nxt
            curr = nxt
        return curr


@dataclass(frozen=True)
class RecursiveFold(typing.Generic[_NodeT, _AccT]):
    """First-order structural catamorphism witness folding ``_NodeT`` into carrier ``_AccT``.

    Attributes:
        fold_step (Callable[[_NodeT, Callable[[_NodeT], _AccT]], _AccT]): Open-recursive algebra step.
    """

    fold_step: Callable[[_NodeT, Callable[[_NodeT], _AccT]], _AccT]

    def fold(self, node: _NodeT) -> _AccT:
        """Evaluate the structural catamorphism from ``node``.

        Args:
            node (_NodeT): Root node to fold.

        Returns:
            _AccT: Accumulated carrier result.
        """

        def _recurse(curr: _NodeT) -> _AccT:
            """Recursively fold ``curr`` via ``self.fold_step``.

            Args:
                curr (_NodeT): Subtree node.

            Returns:
                _AccT: Folded result for ``curr``.
            """
            return self.fold_step(curr, _recurse)

        return _recurse(node)
