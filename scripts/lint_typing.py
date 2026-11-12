"""AST-based static typing and anti-reflection linter for ml-switcheroo-ir.

Enforces zero usage of:
- ``getattr``, ``setattr``, ``hasattr``, ``delattr``, ``vars``, ``globals``, ``locals``,
  ``dir``, ``eval``, ``exec``, ``compile``, ``cast``
- ``operator.attrgetter``, ``operator.itemgetter``, ``operator.methodcaller``
- ``inspect.getmembers``, ``inspect.getattr_static``, ``sys.modules`` lookup workarounds
- ``types.FunctionType``, ``core_schema.any_schema``, and unconstrained ``{}`` JSON schema shims
- ``if TYPE_CHECKING: ... else: ...`` divergent runtime type-erasure blocks
- Bare ``assert`` statements inside ``src/`` production modules
- ``object`` annotations, base classes, and calls (except ``other: object`` on ``__eq__`` / ``__ne__``)
- All dunder attribute accesses ``obj.__<name>__`` and direct dunder method invocations
  ``obj.__<name>__(...)`` (only ``super().__<name>__(...)`` is permitted)
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
        "dir",
        "eval",
        "exec",
        "compile",
        "cast",
        "object",
        "getmembers",
        "getattr_static",
        "read_runtime_attr",
        "has_runtime_attr",
        "set_runtime_attr",
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

_TYPING_MODULES: frozenset[str] = frozenset({"typing", "typing_extensions"})
_OPERATOR_MODULES: frozenset[str] = frozenset({"operator", "_operator"})
_FORBIDDEN_TYPING_NAMES: frozenset[str] = frozenset({"Any", "cast"})
_MODULE_FORBIDDEN_TABLE: tuple[tuple[frozenset[str], frozenset[str], str], ...] = (
    (_TYPING_MODULES, _FORBIDDEN_TYPING_NAMES, "Forbidden reference"),
    (frozenset({"builtins", "inspect"}), _FORBIDDEN_BUILTIN_CALLS, "Forbidden builtin reference"),
    (_OPERATOR_MODULES, _FORBIDDEN_OPERATOR_NAMES, "Forbidden operator helper"),
    (frozenset({"types"}), _FORBIDDEN_TYPES_NAMES, "Forbidden dynamic type reference"),
)


def _is_dunder_name(name: str) -> bool:
    """Return True if ``name`` is a dunder identifier ``__<name>__``.

    Args:
        name (str): Candidate attribute or method identifier.

    Returns:
        bool: True if ``name`` starts and ends with ``__`` and has length > 4.
    """
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _is_super_call(node: ast.AST) -> bool:
    """Return True if ``node`` is a ``super()`` call expression.

    Args:
        node (ast.AST): AST expression node to inspect.

    Returns:
        bool: True if ``node`` represents ``super(...)``.
    """
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super"


def _is_type_checking_guard(test: ast.expr) -> bool:
    """Return True if ``test`` is ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``.

    Args:
        test (ast.expr): Conditional expression on an ``ast.If`` node.

    Returns:
        bool: True if the condition tests ``TYPE_CHECKING``.
    """
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _is_src_production_file(file_path: Path) -> bool:
    """Return True if ``file_path`` resides under a ``src`` package directory.

    Args:
        file_path (Path): Path of the file being checked.

    Returns:
        bool: True when ``'src'`` is one of the path components.
    """
    return "src" in file_path.parts


def _returns_empty_dict(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if ``__get_pydantic_json_schema__`` directly returns an empty ``{}`` dict literal.

    Args:
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node.

    Returns:
        bool: True if any return statement returns an empty ``ast.Dict`` literal.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict) and not child.value.keys:
            return True
    return False


class _TypingAndReflectionVisitor(ast.NodeVisitor):
    """AST visitor collecting static typing, anti-reflection, and anti-dunder violations.

    Attributes:
        file_path (Path): Path of the source file under inspection.
        violations (list[str]): Collected lint violation strings.
        is_src_file (bool): Whether ``file_path`` is inside ``src/``.
    """

    file_path: Path
    violations: list[str]
    is_src_file: bool

    def __init__(self, file_path: Path) -> None:
        """Initialize the visitor.

        Args:
            file_path (Path): Path of the file being checked.
        """
        self.file_path = file_path
        self.violations = []
        self.is_src_file = _is_src_production_file(file_path)
        self._allowed_object_nodes: set[int] = set()

    def _record(self, lineno: int, col_offset: int, message: str) -> None:
        """Record a lint violation.

        Args:
            lineno (int): 1-based line number.
            col_offset (int): 0-based column offset.
            message (str): Human-readable violation description.
        """
        self.violations.append(f"{self.file_path}:{lineno}:{col_offset + 1}: {message}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Inspect function definitions for LSP ``__eq__``/``__ne__`` object params and empty JSON schemas.

        Args:
            node (ast.FunctionDef): Function definition AST node.
        """
        if node.name in ("__eq__", "__ne__"):
            all_pos = list(node.args.posonlyargs) + list(node.args.args)
            for arg in all_pos[1:2]:
                if isinstance(arg.annotation, ast.Name) and arg.annotation.id == "object":
                    self._allowed_object_nodes.add(id(arg.annotation))
        elif node.name == "__get_pydantic_json_schema__" and _returns_empty_dict(node):
            self._record(
                node.lineno,
                node.col_offset,
                "Forbidden unconstrained empty '{}' return in __get_pydantic_json_schema__",
            )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """Check for ``if TYPE_CHECKING: ... else: ...`` divergent runtime type-erasure blocks.

        Args:
            node (ast.If): Conditional statement AST node.
        """
        if _is_type_checking_guard(node.test) and node.orelse:
            self._record(
                node.lineno,
                node.col_offset,
                "Forbidden 'if TYPE_CHECKING: ... else:' divergent runtime type definition",
            )
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        """Forbid bare ``assert`` statements inside ``src/`` production code.

        Args:
            node (ast.Assert): Assert statement AST node.
        """
        if self.is_src_file:
            self._record(
                node.lineno,
                node.col_offset,
                "Forbidden bare 'assert' in production source; raise explicit exception instead",
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check forbidden imports from ``typing``, ``builtins``, ``inspect``, ``operator``, and ``types``.

        Args:
            node (ast.ImportFrom): ImportFrom AST node.
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
        """Check forbidden module attributes, dunder attributes, ``sys.modules``, and ``any_schema``.

        Args:
            node (ast.Attribute): Attribute AST node.
        """
        if _is_dunder_name(node.attr) and not _is_super_call(node.value):
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
        elif (
            self.is_src_file and isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr == "modules"
        ):
            self._record(
                node.lineno,
                node.col_offset,
                "Forbidden cross-module 'sys.modules' lookup in production source",
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
            node (ast.Call): Call AST node.
        """
        if isinstance(node.func, ast.Name):
            fn_name = node.func.id
            if (
                fn_name in _FORBIDDEN_BUILTIN_CALLS
                or fn_name in _FORBIDDEN_OPERATOR_NAMES
                or fn_name in _FORBIDDEN_TYPES_NAMES
            ):
                self._record(
                    node.lineno,
                    node.col_offset,
                    f"Forbidden reflection/cast call '{fn_name}()'",
                )
        elif isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if _is_dunder_name(attr) and not _is_super_call(node.func.value):
                self._record(
                    node.lineno,
                    node.col_offset,
                    f"Forbidden direct dunder method call '.{attr}()'",
                )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check bare ``Any`` and ``object`` references.

        Args:
            node (ast.Name): Name AST node.
        """
        if node.id == "Any":
            self._record(node.lineno, node.col_offset, "Forbidden reference to 'Any'")
        elif node.id == "object" and isinstance(node.ctx, ast.Load) and id(node) not in self._allowed_object_nodes:
            self._record(node.lineno, node.col_offset, "Forbidden reference to 'object'")
        self.generic_visit(node)


def check_source(source: str, file_path: Path) -> list[str]:
    """Check a single Python source string for typing and reflection violations.

    Args:
        source (str): Python source code text.
        file_path (Path): Path associated with the source text.

    Returns:
        list[str]: Formatted violation messages.
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
        paths (Sequence[str | Path]): Sequence of file or directory path strings or Path objects.

    Returns:
        list[Path]: Sorted list of ``.py`` files.
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
        argv (Sequence[str] | None): Command-line arguments (defaults to ``src``, ``scripts``, ``tests``).

    Returns:
        int: Exit code (0 if clean, 1 if violations found).
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


if __name__ == "__main__":
    sys.exit(main())
