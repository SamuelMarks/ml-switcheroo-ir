"""Granular AST docstring verifier for modules, classes, fields, enums, functions, args, returns, and raises."""

from __future__ import annotations

import ast
import os
import re
import sys
from collections.abc import Sequence

_SECTION_HEADERS: frozenset[str] = frozenset(
    {
        "Args:",
        "Arguments:",
        "Attributes:",
        "Returns:",
        "Yields:",
        "Raises:",
        "Note:",
        "Notes:",
        "Example:",
        "Examples:",
    }
)

_ENTRY_PATTERN: re.Pattern[str] = re.compile(r"^(\s+)\*{0,2}([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(.*\))?\s*:(.*)$")


def _parse_docstring_sections(doc: str) -> dict[str, list[str]]:
    """Split a cleaned Google-style docstring into section line lists keyed by header.

    Args:
        doc (str): Cleaned docstring text from ``ast.get_docstring``.

    Returns:
        dict[str, list[str]]: Mapping from section header (or ``_summary``) to raw section lines.
    """
    sections: dict[str, list[str]] = {"_summary": []}
    current = "_summary"
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped in _SECTION_HEADERS and not line[:1].isspace():
            current = stripped
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return sections


def _parse_named_entries(lines: Sequence[str]) -> dict[str, str]:
    """Parse ``name (type): description`` entries and continuation lines from a section block.

    Args:
        lines (Sequence[str]): Lines belonging to an ``Args:`` or ``Attributes:`` block.

    Returns:
        dict[str, str]: Mapping from entry identifier to stripped description text.
    """
    entries: dict[str, list[str]] = {}
    current_name: str | None = None
    entry_indent: int | None = None

    for raw_line in lines:
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if entry_indent is None or indent <= entry_indent:
            match = _ENTRY_PATTERN.match(raw_line)
            if match is not None:
                entry_indent = len(match.group(1))
                current_name = match.group(2)
                first_desc = match.group(3).strip()
                entries[current_name] = [first_desc] if first_desc else []
                continue
        if current_name is not None and entry_indent is not None and indent > entry_indent:
            entries[current_name].append(raw_line.strip())

    return {name: " ".join(parts).strip() for name, parts in entries.items()}


def _extract_expected_arg_names(args: ast.arguments) -> list[str]:
    """Extract parameter names (excluding leading ``self``/``cls``) from ``ast.arguments``.

    Args:
        args (ast.arguments): AST function arguments node.

    Returns:
        list[str]: Ordered parameter names expected in the docstring ``Args:`` section.
    """
    names: list[str] = []
    all_pos = list(args.posonlyargs) + list(args.args)
    for idx, arg in enumerate(all_pos):
        if idx == 0 and arg.arg in ("self", "cls"):
            continue
        names.append(arg.arg)
    if args.vararg is not None:
        names.append(args.vararg.arg)
    for kwarg in args.kwonlyargs:
        names.append(kwarg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _has_non_none_return(returns: ast.expr | None) -> bool:
    """Check whether a function return annotation is present and not ``None``.

    Args:
        returns (ast.expr | None): Function return annotation AST expression.

    Returns:
        bool: True if a non-``None`` return annotation is present.
    """
    if returns is None:
        return False
    if isinstance(returns, ast.Constant) and returns.value is None:
        return False
    return not (isinstance(returns, ast.Name) and returns.id == "None")


def _is_overload_decorated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if a function is decorated with ``@overload`` or ``@typing.overload``.

    Args:
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node.

    Returns:
        bool: True if decorated with ``@overload``.
    """
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def _scan_direct_body_control(
    body: Sequence[ast.stmt],
) -> tuple[bool, bool]:
    """Inspect a function body (excluding nested classes/functions) for yield and explicit raise statements.

    Args:
        body (Sequence[ast.stmt]): Statements directly inside a function body.

    Returns:
        tuple[bool, bool]: Tuple of ``(has_yield, has_explicit_raise)``.
    """
    has_yield = False
    has_raise = False
    stack: list[ast.AST] = list(body)
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(current, (ast.Yield, ast.YieldFrom)):
            has_yield = True
        elif isinstance(current, ast.Raise) and current.exc is not None:
            has_raise = True
        stack.extend(ast.iter_child_nodes(current))
    return has_yield, has_raise


def _check_prohibited_doc_types(path: str, lineno: int, name: str, doc: str) -> list[str]:
    """Check for prohibited untyped top-type references inside a docstring.

    Permits ``object`` parameter typing exclusively on ``__eq__`` and ``__ne__`` methods
    where Python's Liskov Substitution Principle mandates it.

    Args:
        path (str): File path being inspected.
        lineno (int): Starting line number of the symbol.
        name (str): Function or class name.
        doc (str): Cleaned docstring text.

    Returns:
        list[str]: Violation messages if prohibited type tokens appear.
    """
    has_any = ("(" + "Any)") in doc
    has_obj = ("(" + "object)") in doc and name not in ("__eq__", "__ne__")
    if has_any or has_obj:
        return [f"{path}:{lineno}: prohibited Any/object docstring type in '{name}'"]
    return []


def _check_function_args_section(
    path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    sections: dict[str, list[str]],
) -> list[str]:
    """Verify exact ``Args:`` coverage and absence of stale documented arguments on a function.

    Args:
        path (str): File path being inspected.
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node.
        sections (dict[str, list[str]]): Parsed docstring sections.

    Returns:
        list[str]: Violation messages for missing, empty, or stale argument documentation.
    """
    violations: list[str] = []
    expected_args = _extract_expected_arg_names(node.args)
    args_lines = list(sections.get("Args:", [])) + list(sections.get("Arguments:", []))
    documented_args = _parse_named_entries(args_lines)

    if expected_args and "Args:" not in sections and "Arguments:" not in sections:
        violations.append(f"{path}:{node.lineno}: missing 'Args:' section in '{node.name}'")

    for arg_name in expected_args:
        if arg_name not in documented_args or not documented_args[arg_name]:
            violations.append(f"{path}:{node.lineno}: missing arg '{arg_name}' in '{node.name}'")

    expected_set = set(expected_args)
    for doc_arg in documented_args:
        if doc_arg not in expected_set:
            violations.append(f"{path}:{node.lineno}: stale documented arg '{doc_arg}' in '{node.name}'")

    return violations


def _section_has_content(sections: dict[str, list[str]], header: str) -> bool:
    """Return True if ``header`` exists in ``sections`` and contains non-whitespace text.

    Args:
        sections (dict[str, list[str]]): Parsed docstring section dictionary.
        header (str): Section header name (e.g. ``'Returns:'``).

    Returns:
        bool: True if the section is present and non-empty.
    """
    return header in sections and any(line.strip() for line in sections[header])


def _check_function_node(path: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Check that a function or method AST node has complete argument, return, yield, and raises docs.

    Args:
        path (str): File path being inspected.
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function or async function AST node.

    Returns:
        list[str]: Violation messages for the function.
    """
    if _is_overload_decorated(node):
        return []
    doc = ast.get_docstring(node)
    if not doc or not doc.strip():
        return [f"{path}:{node.lineno}: missing docstring on '{node.name}'"]

    violations = _check_prohibited_doc_types(path, node.lineno, node.name, doc)
    sections = _parse_docstring_sections(doc)
    violations.extend(_check_function_args_section(path, node, sections))

    has_yield, has_raise = _scan_direct_body_control(node.body)
    if has_yield and not _section_has_content(sections, "Yields:"):
        violations.append(f"{path}:{node.lineno}: missing 'Yields:' section in '{node.name}'")
    elif not has_yield and node.name != "__init__" and _has_non_none_return(node.returns):
        if not _section_has_content(sections, "Returns:") and not _section_has_content(sections, "Yields:"):
            violations.append(f"{path}:{node.lineno}: missing 'Returns:'/'Yields:' section in '{node.name}'")

    if has_raise and not _section_has_content(sections, "Raises:"):
        violations.append(f"{path}:{node.lineno}: missing 'Raises:' section in '{node.name}'")

    return violations


def _has_following_inline_docstring(stmts: Sequence[ast.stmt], idx: int) -> bool:
    """Check whether ``stmts[idx + 1]`` is an inline string literal docstring for a field or enum member.

    Args:
        stmts (Sequence[ast.stmt]): Class body statement sequence.
        idx (int): Index of the field/member assignment in ``stmts``.

    Returns:
        bool: True if immediately followed by a non-empty string literal expression.
    """
    if idx + 1 >= len(stmts):
        return False
    nxt = stmts[idx + 1]
    return (
        isinstance(nxt, ast.Expr)
        and isinstance(nxt.value, ast.Constant)
        and isinstance(nxt.value.value, str)
        and bool(nxt.value.value.strip())
    )


def _has_field_description_kwarg(value: ast.expr | None) -> bool:
    """Return True if ``value`` is a ``Field(..., description='...')`` call with non-empty description.

    Args:
        value (ast.expr | None): Assigned value AST expression on a class attribute.

    Returns:
        bool: True if a non-empty ``description`` keyword argument is provided.
    """
    if not isinstance(value, ast.Call):
        return False
    for kw in value.keywords:
        if kw.arg == "description":
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return bool(kw.value.value.strip())
            return True
    return False


def _is_enum_subclass(node: ast.ClassDef) -> bool:
    """Return True if ``node`` inherits from ``Enum`` or ``IntEnum`` or ``StrEnum``.

    Args:
        node (ast.ClassDef): Class definition AST node.

    Returns:
        bool: True if any base class references an Enum base.
    """
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id.endswith("Enum"):
            return True
        if isinstance(base, ast.Attribute) and base.attr.endswith("Enum"):
            return True
    return False


def _check_class_node(path: str, node: ast.ClassDef) -> list[str]:
    """Check class docstring, public annotated attributes, and Enum member documentation.

    Args:
        path (str): File path being inspected.
        node (ast.ClassDef): Class definition AST node.

    Returns:
        list[str]: Violation messages discovered on the class.
    """
    cls_doc = ast.get_docstring(node)
    if not cls_doc or not cls_doc.strip():
        return [f"{path}:{node.lineno}: missing class docstring on '{node.name}'"]

    violations = _check_prohibited_doc_types(path, node.lineno, node.name, cls_doc)
    sections = _parse_docstring_sections(cls_doc)
    documented_attrs = _parse_named_entries(sections.get("Attributes:", []))
    is_enum = _is_enum_subclass(node)

    stmts = node.body
    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fname = stmt.target.id
            if (
                not fname.startswith("_")
                and fname != "model_config"
                and not (
                    (fname in documented_attrs and bool(documented_attrs[fname]))
                    or _has_following_inline_docstring(stmts, idx)
                    or _has_field_description_kwarg(stmt.value)
                )
            ):
                violations.append(f"{path}:{stmt.lineno}: undocumented attribute '{fname}' on class '{node.name}'")
        elif is_enum and isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    mname = target.id
                    if not (
                        (mname in documented_attrs and bool(documented_attrs[mname]))
                        or _has_following_inline_docstring(stmts, idx)
                    ):
                        violations.append(
                            f"{path}:{stmt.lineno}: undocumented enum member '{mname}' on Enum '{node.name}'"
                        )

    return violations


def check_file(path: str) -> list[str]:
    """Check a single Python file for module, class, attribute, enum, function, arg, return, and raises docstrings.

    Args:
        path (str): Path to the Python file to inspect.

    Returns:
        list[str]: Violation messages discovered in the file.
    """
    with open(path, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=path)
    violations: list[str] = []
    mod_doc = ast.get_docstring(tree)
    if not mod_doc or not mod_doc.strip():
        violations.append(f"{path}:1: missing module docstring")
    else:
        violations.extend(_check_prohibited_doc_types(path, 1, "<module>", mod_doc))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            violations.extend(_check_class_node(path, node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_check_function_node(path, node))

    return violations


def check_directory(target_dir: str) -> list[str]:
    """Recursively scan a directory or file for docstring violations.

    Args:
        target_dir (str): Directory or file path to inspect.

    Returns:
        list[str]: Aggregated violation messages.
    """
    if not os.path.exists(target_dir):
        return []
    if os.path.isfile(target_dir):
        return check_file(target_dir) if target_dir.endswith(".py") else []

    violations: list[str] = []
    for root, _, files in sorted(os.walk(target_dir)):
        for fname in sorted(files):
            if fname.endswith(".py"):
                violations.extend(check_file(os.path.join(root, fname)))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the documentation coverage check CLI.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments;
            defaults to ``('src/ml_switcheroo_ir', 'scripts', 'tests')``.

    Returns:
        int: Process exit code (0 when 100% compliant, 1 otherwise).
    """
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    targets: Sequence[str] = raw_args if raw_args else ("src/ml_switcheroo_ir", "scripts", "tests")
    all_violations: list[str] = []
    for target in targets:
        all_violations.extend(check_directory(str(target)))

    if all_violations:
        for v in all_violations:
            print(v, file=sys.stderr)
        return 1
    print("Documentation coverage check passed (0 violations).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
