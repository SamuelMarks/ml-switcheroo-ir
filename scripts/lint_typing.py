"""AST-based static typing and anti-reflection linter for ml-switcheroo-ir.

Enforces zero usage of:
- ``getattr``, ``setattr``, ``hasattr``, ``delattr``, ``vars``, ``globals``, ``locals``, ``eval``, ``exec``, ``compile``, ``cast``
- ``operator.attrgetter``, ``operator.itemgetter``, ``operator.methodcaller``
- ``types.FunctionType`` and ``core_schema.any_schema``
- ``object`` annotations, base classes, and calls (except ``other: object`` in ``__eq__`` / ``__ne__``)
- ``object.__setattr__``, ``object.__getattribute__``, ``object.__delattr__``, ``.__dict__``, ``.__mro__``, ``.__subclasses__``, ``.__class__``, ``.__get_pydantic_core_schema__``, ``.__builtins__``, ``.__globals__``, ``.__code__``, ``.__closure__``
- Direct dunder method invocations ``obj.__<name>__(...)`` (only ``super().__<name>__(...)`` is permitted)
- ``typing.Any``, ``typing_extensions.Any``
- ``typing.cast``, ``typing_extensions.cast``
- ``# type: ignore`` comments
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from collections.abc import Sequence
from pathlib import Path

_FORBIDDEN_BUILTIN_CALLS: frozenset[str] = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "eval",
        "exec",
        "compile",
        "cast",
        "object",
    }
)

_FORBIDDEN_OPERATOR_NAMES: frozenset[str] = frozenset(
    {
        "attrgetter",
        "itemgetter",
        "methodcaller",
    }
)

_FORBIDDEN_TYPES_NAMES: frozenset[str] = frozenset(
    {
        "FunctionType",
    }
)

_FORBIDDEN_DUNDER_ATTRS: frozenset[str] = frozenset(
    {
        "__dict__",
        "__mro__",
        "__subclasses__",
        "__setattr__",
        "__getattribute__",
        "__delattr__",
        "__class__",
        "__get_pydantic_core_schema__",
        "__builtins__",
        "__globals__",
        "__code__",
        "__closure__",
    }
)

_TYPING_MODULES: frozenset[str] = frozenset({"typing", "typing_extensions"})
_OPERATOR_MODULES: frozenset[str] = frozenset({"operator", "_operator"})
_FORBIDDEN_TYPING_NAMES: frozenset[str] = frozenset({"Any", "cast"})
_MODULE_FORBIDDEN_TABLE: tuple[tuple[frozenset[str], frozenset[str], str], ...] = (
    (_TYPING_MODULES, _FORBIDDEN_TYPING_NAMES, "Forbidden reference"),
    (frozenset({"builtins"}), _FORBIDDEN_BUILTIN_CALLS, "Forbidden builtin reference"),
    (_OPERATOR_MODULES, _FORBIDDEN_OPERATOR_NAMES, "Forbidden operator helper"),
    (frozenset({"types"}), _FORBIDDEN_TYPES_NAMES, "Forbidden dynamic type reference"),
)


class _TypingAndReflectionVisitor(ast.NodeVisitor):
    """AST visitor collecting static typing, anti-reflection, and anti-dunder violations."""

    def __init__(self, file_path: Path) -> None:
        """Initialize the visitor.

        Args:
            file_path: Path of the file being checked.
        """
        self.file_path = file_path
        self.violations: list[str] = []
        self._allowed_eq_object_nodes: set[int] = set()

    def _record(self, lineno: int, col_offset: int, message: str) -> None:
        """Record a lint violation.

        Args:
            lineno: 1-based line number.
            col_offset: 0-based column offset.
            message: Human-readable violation description.
        """
        self.violations.append(f"{self.file_path}:{lineno}:{col_offset + 1}: {message}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Allow ``other: object`` parameter annotation only on ``__eq__`` and ``__ne__``.

        Args:
            node: FunctionDef AST node.
        """
        if node.name in ("__eq__", "__ne__"):
            for arg in node.args.args:
                if isinstance(arg.annotation, ast.Name) and arg.annotation.id == "object":
                    self._allowed_eq_object_nodes.add(id(arg.annotation))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check forbidden imports from ``typing``, ``builtins``, ``operator``, and ``types``.

        Args:
            node: ImportFrom AST node.
        """
        module = node.module or ""
        for mod_set, forbidden_names, _ in _MODULE_FORBIDDEN_TABLE:
            if module in mod_set:
                for alias in node.names:
                    if alias.name in forbidden_names:
                        self._record(
                            node.lineno,
                            node.col_offset,
                            f"Forbidden import '{module}.{alias.name}'",
                        )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check forbidden module attributes, dunder attributes, and ``any_schema``.

        Args:
            node: Attribute AST node.
        """
        if node.attr in _FORBIDDEN_DUNDER_ATTRS:
            self._record(
                node.lineno,
                node.col_offset,
                f"Forbidden dunder attribute access '.{node.attr}'",
            )
        elif node.attr == "any_schema":
            self._record(
                node.lineno,
                node.col_offset,
                "Forbidden untyped schema reference '.any_schema'",
            )
        elif isinstance(node.value, ast.Name):
            for mod_set, forbidden_names, prefix in _MODULE_FORBIDDEN_TABLE:
                if node.value.id in mod_set and node.attr in forbidden_names:
                    self._record(
                        node.lineno,
                        node.col_offset,
                        f"{prefix} '{node.value.id}.{node.attr}'",
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check direct calls to forbidden builtins, operator helpers, and dunder methods.

        Args:
            node: Call AST node.
        """
        if isinstance(node.func, ast.Name):
            fn_name = node.func.id
            if fn_name in _FORBIDDEN_BUILTIN_CALLS or fn_name in _FORBIDDEN_OPERATOR_NAMES or fn_name in _FORBIDDEN_TYPES_NAMES:
                self._record(
                    node.lineno,
                    node.col_offset,
                    f"Forbidden reflection/cast call '{fn_name}()'",
                )
        elif isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if len(attr) > 4 and attr.startswith("__") and attr.endswith("__"):
                is_super = isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name) and node.func.value.func.id == "super"
                if not is_super:
                    self._record(
                        node.lineno,
                        node.col_offset,
                        f"Forbidden direct dunder method call '.{attr}()'",
                    )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check bare ``Any`` and ``object`` references.

        Args:
            node: Name AST node.
        """
        if node.id == "Any":
            self._record(node.lineno, node.col_offset, "Forbidden reference to 'Any'")
        elif node.id == "object" and isinstance(node.ctx, ast.Load) and id(node) not in self._allowed_eq_object_nodes:
            self._record(node.lineno, node.col_offset, "Forbidden reference to 'object'")
        self.generic_visit(node)


def check_source(source: str, file_path: Path) -> list[str]:
    """Check a single Python source string for typing and reflection violations.

    Args:
        source: Python source code text.
        file_path: Path associated with the source text.

    Returns:
        List of formatted violation messages.
    """
    tree = ast.parse(source, filename=str(file_path))
    visitor = _TypingAndReflectionVisitor(file_path)
    visitor.visit(tree)

    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for tok in tokens:
        if tok.type == tokenize.COMMENT and "type: ignore" in tok.string:
            visitor._record(tok.start[0], tok.start[1], "Forbidden '# type: ignore' comment")

    return visitor.violations


def collect_python_files(paths: Sequence[str | Path]) -> list[Path]:
    """Resolve file and directory arguments into sorted Python file paths.

    Args:
        paths: Sequence of file or directory path strings or Path objects.

    Returns:
        Sorted list of ``.py`` files.
    """
    collected: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".py":
            collected.append(path)
        elif path.is_dir():
            collected.extend(sorted(path.rglob("*.py")))
    return sorted(set(collected))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AST typing and anti-reflection linter across target paths.

    Args:
        argv: Command-line arguments (defaults to ``src``, ``scripts``, ``tests``).

    Returns:
        Exit code (0 if clean, 1 if violations found).
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    targets = args if args else ["src", "scripts", "tests"]
    files = collect_python_files(targets)

    all_violations: list[str] = []
    for py_file in files:
        source = py_file.read_text(encoding="utf-8")
        all_violations.extend(check_source(source, py_file))

    if all_violations:
        for violation in all_violations:
            print(violation, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
