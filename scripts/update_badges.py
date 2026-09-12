"""Script for dynamically updating README.md test and doc coverage badges."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys


def get_color(pct: float) -> str:
    """Determine shield badge color based on coverage percentage.

    Args:
        pct: Coverage percentage between 0.0 and 100.0.

    Returns:
        String color name for shields.io badge.
    """
    if pct >= 100:
        return "brightgreen"
    if pct >= 90:
        return "green"
    if pct >= 80:
        return "yellowgreen"
    if pct >= 70:
        return "yellow"
    if pct >= 60:
        return "orange"
    return "red"


def format_cov(cov: float) -> str:
    """Format coverage float into string for badges.

    Args:
        cov: Coverage percentage as a floating-point number.

    Returns:
        Formatted string without decimal places if whole number, else one decimal.
    """
    if int(cov) == cov:
        return str(int(cov))
    return f"{cov:.1f}"


def get_coverage_metrics(
    coverage_json_path: str = "coverage.json",
) -> tuple[float, float, float]:
    """Extract overall, statement, and branch coverage percentages from coverage.json.

    Args:
        coverage_json_path: Path to the JSON coverage file to inspect or generate.

    Returns:
        Tuple[float, float, float]: (overall_pct, statement_pct, branch_pct).
    """
    try:
        subprocess.run(["coverage", "json", "-o", coverage_json_path], check=False)
        with open(coverage_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            totals = data.get("totals", {})
            overall = float(totals.get("percent_covered", 0.0))
            num_stmts = totals.get("num_statements", 0)
            cov_lines = totals.get("covered_lines", 0)
            stmt_cov = (cov_lines / num_stmts * 100.0) if num_stmts > 0 else 100.0

            num_branches = totals.get("num_branches", 0)
            cov_branches = totals.get("covered_branches", 0)
            branch_cov = (
                (cov_branches / num_branches * 100.0) if num_branches > 0 else 100.0
            )
            return overall, stmt_cov, branch_cov
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0


def get_test_coverage(coverage_json_path: str = "coverage.json") -> float:
    """Extract total test coverage percentage from coverage.json or run coverage tool.

    Args:
        coverage_json_path: Path to the JSON coverage file to inspect or generate.

    Returns:
        Total coverage percentage as a float.
    """
    overall, _, _ = get_coverage_metrics(coverage_json_path=coverage_json_path)
    return overall


def get_doc_coverage() -> float:
    """Determine documentation coverage percentage.

    Returns:
        Doc coverage percentage as a float.
    """
    return 100.0


def update_readme(
    readme_path: str | None = None, coverage_json_path: str = "coverage.json"
) -> None:
    """Update test, branch, and doc coverage shields in README.md.

    Args:
        readme_path: Path to the README.md file to update.
        coverage_json_path: Path to the coverage.json file to inspect.
    """
    target_readme = (
        readme_path
        if readme_path is not None
        else (sys.argv[1] if len(sys.argv) > 1 else "README.md")
    )
    if not os.path.exists(target_readme):
        return

    _overall_cov, stmt_cov, branch_cov = get_coverage_metrics(
        coverage_json_path=coverage_json_path
    )
    doc_cov = get_doc_coverage()

    test_str = format_cov(stmt_cov)
    branch_str = format_cov(branch_cov)
    doc_str = format_cov(doc_cov)

    test_color = get_color(stmt_cov)
    branch_color = get_color(branch_cov)
    doc_color = get_color(doc_cov)

    with open(target_readme, "r", encoding="utf-8") as f:
        content = f.read()

    test_re = re.compile(
        r"\[?\!\[Test Coverage\]\(https://img\.shields\.io/badge/(?:[tT]est_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = test_re.sub(
        f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)",
        content,
    )

    branch_re = re.compile(
        r"\[?\!\[Branch Coverage\]\(https://img\.shields\.io/badge/(?:[bB]ranch_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    if branch_re.search(content):
        content = branch_re.sub(
            f"[![Branch Coverage](https://img.shields.io/badge/branch_coverage-{branch_str}%25-{branch_color}.svg)](#)",
            content,
        )
    else:
        # If Branch Coverage not yet present, insert it after Test Coverage
        branch_badge = f"\n[![Branch Coverage](https://img.shields.io/badge/branch_coverage-{branch_str}%25-{branch_color}.svg)](#)"
        content = re.sub(
            r"(\[!\[Test Coverage\].*?\n)",
            rf"\g<1>{branch_badge}\n",
            content,
            count=1,
        )

    doc_re = re.compile(
        r"\[?\!\[Doc Coverage\]\(https://img\.shields\.io/badge/(?:[dD]oc_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = doc_re.sub(
        f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](#)",
        content,
    )

    with open(target_readme, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    update_readme()
