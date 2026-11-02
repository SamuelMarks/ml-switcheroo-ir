"""AST-based static typing and anti-reflection linter for ml-switcheroo-ir.

Enforces zero usage of:
- ``getattr``, ``setattr``, ``hasattr``, ``delattr``
- ``object.__setattr__``, ``object.__getattribute__``, ``object.__delattr__``
- ``typing.Any`` / ``Any``
- ``typing.cast`` / ``cast``
- ``# type: ignore`` comments
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
import io
from pathlib import Path
import sys
import tokenize

_FORBIDDEN_BUILTIN_CALLS: frozenset[str] = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "cast",
    }
)

_FORBIDDEN_DUNDER_ATTRS: frozenset[str] = frozenset(
    {
        "__setattr__",
        "__getattribute__",
        "__delattr__",
    }
)

_TYPING_MODULES: frozenset[str] = frozenset(
    {
        "typing",
        "typing_extensions",
    }
)

_FORBIDDEN_TYPING_NAMES: frozenset[str] = frozenset(
    {
        "Any",
        "cast",
    }
)


class _TypingVisitor(ast.NodeVisitor):
    """AST visitor detecting forbidden reflection and typing constructs."""

    def __init__(self, file_path: Path) -> None:
        """Initialize the visitor for a single file.

        Args:
            file_path (Path): Path of the Python file being inspected.
        """
        self.file_path = file_path
        self.violations: list[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Inspect ``from ... import ...`` statements for Any, cast, or reflection functions.

        Args:
            node (ast.ImportFrom): ImportFrom AST node.
        """
        mod = node.module or ""
        if mod in _TYPING_MODULES:
            for alias in node.names:
                if alias.name in _FORBIDDEN_TYPING_NAMES:
                    self.violations.append(
                        f"{self.file_path}:{node.lineno}:{node.col_offset}: "
                        f"Forbidden typing import '{mod}.{alias.name}'"
                    )
        elif mod == "builtins":
            for alias in node.names:
                if alias.name in _FORBIDDEN_BUILTIN_CALLS:
                    self.violations.append(
                        f"{self.file_path}:{node.lineno}:{node.col_offset}: "
                        f"Forbidden reflection import 'builtins.{alias.name}'"
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Inspect attribute accesses for ``typing.Any``, ``typing.cast``, or ``object.__setattr__``.

        Args:
            node (ast.Attribute): Attribute AST node.
        """
        if isinstance(node.value, ast.Name):
            if (
                node.value.id in _TYPING_MODULES
                and node.attr in _FORBIDDEN_TYPING_NAMES
            ):
                self.violations.append(
                    f"{self.file_path}:{node.lineno}:{node.col_offset}: "
                    f"Forbidden typing construct '{node.value.id}.{node.attr}'"
                )
            elif node.value.id == "builtins" and node.attr in _FORBIDDEN_BUILTIN_CALLS:
                self.violations.append(
                    f"{self.file_path}:{node.lineno}:{node.col_offset}: "
                    f"Forbidden reflection construct 'builtins.{node.attr}'"
                )
            elif node.value.id == "object" and node.attr in _FORBIDDEN_DUNDER_ATTRS:
                self.violations.append(
                    f"{self.file_path}:{node.lineno}:{node.col_offset}: "
                    f"Forbidden reflection construct 'object.{node.attr}'"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Inspect function calls for ``getattr``, ``setattr``, ``hasattr``, ``delattr``, or ``cast``.

        Args:
            node (ast.Call): Call AST node.
        """
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_BUILTIN_CALLS:
            self.violations.append(
                f"{self.file_path}:{node.lineno}:{node.col_offset}: "
                f"Forbidden call to '{node.func.id}()'"
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Inspect bare ``Any`` name references.

        Args:
            node (ast.Name): Name AST node.
        """
        if node.id == "Any":
            self.violations.append(
                f"{self.file_path}:{node.lineno}:{node.col_offset}: "
                "Forbidden reference to 'Any'"
            )
        self.generic_visit(node)


def check_source(source: str, file_path: Path) -> list[str]:
    """Check Python source text for forbidden reflection, Any, cast, and type-ignore comments.

    Args:
        source (str): Python source code text.
        file_path (Path): File path for diagnostic reporting.

    Returns:
        list[str]: List of formatted violation messages.
    """
    violations: list[str] = []
    tree = ast.parse(source, filename=str(file_path))
    visitor = _TypingVisitor(file_path)
    visitor.visit(tree)
    violations.extend(visitor.violations)

    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            normalized = tok.string.replace(" ", "").lower()
            if "type:ignore" in normalized:
                line_no, col_no = tok.start
                violations.append(
                    f"{file_path}:{line_no}:{col_no}: "
                    "Forbidden comment directive 'type: ignore'"
                )
    return violations


def collect_python_files(targets: Sequence[str | Path]) -> list[Path]:
    """Collect all Python files under the given target files or directories.

    Args:
        targets (Sequence[str | Path]): File or directory paths to scan.

    Returns:
        list[Path]: Sorted list of discovered ``.py`` files.
    """
    files: list[Path] = []
    for raw_target in targets:
        path = Path(raw_target)
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return sorted(set(files))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the typing linter across target directories or files.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments; defaults to ``('src', 'scripts', 'tests')``.

    Returns:
        int: Exit code (0 if clean, 1 if violations found).
    """
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    targets: Sequence[str] = raw_args if raw_args else ("src", "scripts", "tests")
    py_files = collect_python_files(targets)

    all_violations: list[str] = []
    for py_file in py_files:
        source = py_file.read_text(encoding="utf-8")
        all_violations.extend(check_source(source, py_file))

    if all_violations:
        for violation in all_violations:
            print(violation, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
