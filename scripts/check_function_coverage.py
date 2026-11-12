"""AST and coverage.json verifier enforcing 100% function, line, and branch test coverage."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from ml_switcheroo_ir.types import JsonValue


def _is_overload_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if ``node`` is a ``@overload`` typing stub without runtime body execution.

    Args:
        node (ast.FunctionDef | ast.AsyncFunctionDef): Function AST node.

    Returns:
        bool: True if decorated with ``@overload`` or ``@typing.overload``.
    """
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def _is_protocol_class(node: ast.ClassDef) -> bool:
    """Return True if ``node`` inherits from ``Protocol`` or ``typing.Protocol``.

    Args:
        node (ast.ClassDef): Class definition AST node.

    Returns:
        bool: True if ``node`` is a structural Protocol class.
    """
    for base in node.bases:
        if isinstance(base, ast.Name) and "Protocol" in base.id:
            return True
        if isinstance(base, ast.Attribute) and "Protocol" in base.attr:
            return True
    return False


def _collect_executable_functions(
    tree: ast.AST,
) -> list[tuple[str, int, int]]:
    """Collect ``(qualname, body_start_line, end_line)`` for all concrete functions in an AST.

    Excludes ``@overload`` stubs and abstract docstring-only ``Protocol`` method signatures.

    Args:
        tree (ast.AST): Parsed module AST.

    Returns:
        list[tuple[str, int, int]]: Function name, first executable body line, and ending line.
    """
    collected: list[tuple[str, int, int]] = []

    def _walk(current: ast.AST, prefix: str, in_protocol: bool) -> None:
        """Recursively traverse AST nodes tracking qualified names and Protocol scopes.

        Args:
            current (ast.AST): Current AST node.
            prefix (str): Dotted parent scope prefix.
            in_protocol (bool): True if inside a ``typing.Protocol`` class body.
        """
        for child in ast.iter_child_nodes(current):
            if isinstance(child, ast.ClassDef):
                next_prefix = f"{prefix}.{child.name}" if prefix else child.name
                _walk(child, next_prefix, _is_protocol_class(child))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}.{child.name}" if prefix else child.name
                if not _is_overload_stub(child):
                    body_stmts = [
                        s
                        for idx, s in enumerate(child.body)
                        if not (
                            idx == 0
                            and isinstance(s, ast.Expr)
                            and isinstance(s.value, ast.Constant)
                            and isinstance(s.value.value, str)
                        )
                    ]
                    if body_stmts and not (
                        in_protocol
                        and len(body_stmts) == 1
                        and isinstance(body_stmts[0], ast.Expr)
                        and isinstance(body_stmts[0].value, ast.Constant)
                        and body_stmts[0].value.value is Ellipsis
                    ):
                        start_ln = body_stmts[0].lineno
                        end_ln = child.end_lineno or body_stmts[-1].lineno
                        collected.append((qualname, start_ln, end_ln))
                _walk(child, qualname, False)
            else:
                _walk(child, prefix, in_protocol)

    _walk(tree, "", False)
    return collected


def _load_or_generate_coverage_json(coverage_json_path: str | None) -> dict[str, JsonValue]:
    """Load an existing coverage JSON report or export one from ``.coverage`` via ``coverage json``.

    Args:
        coverage_json_path (str | None): Optional path to an existing ``coverage.json`` report.

    Returns:
        dict[str, JsonValue]: Parsed coverage JSON payload.

    Raises:
        RuntimeError: If ``coverage json`` fails or produces an invalid payload.
    """
    if coverage_json_path is not None:
        with open(coverage_json_path, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise RuntimeError("Coverage JSON root must be an object.")
        return loaded

    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        res = subprocess.run(
            [sys.executable, "-m", "coverage", "json", "-o", tmp_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Failed to export coverage JSON: {res.stderr.strip()}")
        with open(tmp_path, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise RuntimeError("Coverage JSON root must be an object.")
        return loaded
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _extract_int_set(record: Mapping[str, JsonValue], key: str) -> set[int]:
    """Extract a set of line numbers from a coverage file record list field.

    Args:
        record (Mapping[str, JsonValue]): Per-file coverage dictionary.
        key (str): Key name (e.g. ``'executed_lines'`` or ``'missing_lines'``).

    Returns:
        set[int]: Set of integer line numbers.
    """
    raw = record.get(key)
    if not isinstance(raw, list):
        return set()
    return {int(v) for v in raw if isinstance(v, int) and not isinstance(v, bool)}


def _verify_single_file_record(
    rel_path: str,
    file_rec: Mapping[str, JsonValue],
    base: Path,
) -> list[str]:
    """Check a single file coverage record for missing lines, branches, and functions.

    Args:
        rel_path (str): Relative or absolute file path key from ``coverage.json``.
        file_rec (Mapping[str, JsonValue]): Per-file coverage metrics dictionary.
        base (Path): Base repository root directory.

    Returns:
        list[str]: Diagnostic failure strings for the file.
    """
    errors: list[str] = []
    missing_lines = _extract_int_set(file_rec, "missing_lines")
    executed_lines = _extract_int_set(file_rec, "executed_lines")
    raw_missing_branches = file_rec.get("missing_branches")
    if missing_lines:
        errors.append(f"{rel_path}: missing lines {sorted(missing_lines)}")
    if isinstance(raw_missing_branches, list) and raw_missing_branches:
        errors.append(f"{rel_path}: missing branches {raw_missing_branches}")

    full_path = Path(rel_path) if Path(rel_path).is_absolute() else (base / rel_path)
    if not full_path.is_file():
        return errors
    tree = ast.parse(full_path.read_text(encoding="utf-8"), filename=str(full_path))
    for qualname, start_ln, end_ln in _collect_executable_functions(tree):
        fn_lines = set(range(start_ln, end_ln + 1))
        if fn_lines & missing_lines or not (fn_lines & executed_lines):
            errors.append(f"{rel_path}:{start_ln}: uncovered function '{qualname}'")
    return errors


def verify_coverage_data(data: Mapping[str, JsonValue], repo_root: Path | None = None) -> list[str]:
    """Verify 100% function, line, and branch coverage across all measured files in ``data``.

    Args:
        data (Mapping[str, JsonValue]): Parsed ``coverage.json`` payload.
        repo_root (Path | None): Base directory used to resolve relative file paths in ``data['files']``.

    Returns:
        list[str]: Diagnostic failure messages (empty when 100% compliant).
    """
    errors: list[str] = []
    base = repo_root if repo_root is not None else Path.cwd()
    raw_files = data.get("files")
    if not isinstance(raw_files, Mapping) or not raw_files:
        return ["No measured files found in coverage report."]

    for rel_path_obj, file_rec in sorted(raw_files.items()):
        if isinstance(file_rec, Mapping):
            errors.extend(_verify_single_file_record(str(rel_path_obj), file_rec, base))

    totals = data.get("totals")
    if isinstance(totals, Mapping):
        pct = totals.get("percent_covered")
        if isinstance(pct, (int, float)) and float(pct) < 100.0:
            errors.append(f"Total coverage {float(pct):.2f}% is below required 100.00%.")

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI verification for 100% function, line, and branch test coverage.

    Args:
        argv (Sequence[str] | None): Command-line arguments.

    Returns:
        int: Exit status code (0 for 100% coverage, 1 otherwise).
    """
    parser = argparse.ArgumentParser(description="Verify 100% function, line, and branch coverage from coverage data.")
    parser.add_argument(
        "--coverage-json",
        type=str,
        default=None,
        help="Optional path to pre-generated coverage.json file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        cov_data = _load_or_generate_coverage_json(args.coverage_json)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    violations = verify_coverage_data(cov_data)
    if violations:
        for v in violations:
            print(f"[COVERAGE FAILURE] {v}", file=sys.stderr)
        return 1

    print("Function, line, and branch coverage check passed (100.0%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
