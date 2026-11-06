"""Granular AST docstring coverage verifier for modules, classes, functions, and arguments."""

from __future__ import annotations

import ast
import os
import re
import sys
from collections.abc import Sequence


def _extract_expected_arg_names(args: ast.arguments) -> list[str]:
    """Extract parameter names (excluding `self`/`cls`, including `*args`/`**kwargs`) from `ast.arguments`.

    Args:
        args (ast.arguments): AST function arguments node.

    Returns:
        list[str]: Parameter names expected in the docstring `Args:` block.
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
    """Check whether a function return annotation is present and not `None`.

    Args:
        returns (ast.expr | None): Function return annotation AST expression.

    Returns:
        bool: True if a non-`None` return annotation is present.
    """
    if returns is None:
        return False
    if isinstance(returns, ast.Constant) and returns.value is None:
        return False
    return not (isinstance(returns, ast.Name) and returns.id == "None")


def _is_overload_decorated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function has an `@overload` or `@typing.overload` decorator.

    Args:
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node.

    Returns:
        bool: True if decorated with `@overload`.
    """
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def _check_function_node(path: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Check that a function or method AST node has a complete docstring.

    Args:
        path (str): File path being inspected.
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function or async function AST node.

    Returns:
        list[str]: List of violation messages for the function.
    """
    if _is_overload_decorated(node):
        return []
    doc = ast.get_docstring(node)
    if not doc or not doc.strip():
        return [f"{path}:{node.lineno}: missing docstring on '{node.name}'"]
    violations: list[str] = []
    if "(" + "Any)" in doc or "(" + "object)" in doc:
        violations.append(f"{path}:{node.lineno}: prohibited Any/object docstring type in '{node.name}'")
    expected_args = _extract_expected_arg_names(node.args)
    if expected_args and "Args:" not in doc:
        violations.append(f"{path}:{node.lineno}: missing 'Args:' section in '{node.name}'")
    for arg_name in expected_args:
        if not re.search(rf"\b{re.escape(arg_name)}\b", doc):
            violations.append(f"{path}:{node.lineno}: missing arg '{arg_name}' in '{node.name}'")
    if _has_non_none_return(node.returns) and "Returns:" not in doc and "Yields:" not in doc:
        violations.append(f"{path}:{node.lineno}: missing 'Returns:'/'Yields:' section in '{node.name}'")
    return violations


def check_file(path: str) -> list[str]:
    """Check a single Python file for module, class, function, and argument docstring coverage.

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

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cls_doc = ast.get_docstring(node)
            if not cls_doc or not cls_doc.strip():
                violations.append(f"{path}:{node.lineno}: missing class docstring on '{node.name}'")
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
        argv (Sequence[str] | None): Optional CLI arguments; defaults to `("src/ml_switcheroo_ir", "scripts", "tests")`.

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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
