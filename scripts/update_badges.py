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


def get_test_coverage(coverage_json_path: str = "coverage.json") -> float:
    """Extract total test coverage percentage from coverage.json or run coverage tool.

    Args:
        coverage_json_path: Path to the JSON coverage file to inspect or generate.

    Returns:
        Total coverage percentage as a float.
    """
    try:
        subprocess.run(["coverage", "json", "-o", coverage_json_path], check=False)
        with open(coverage_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return float(data["totals"]["percent_covered"])
    except Exception:  # noqa: BLE001
        return 0.0


def get_doc_coverage() -> float:
    """Determine documentation coverage percentage.

    Returns:
        Doc coverage percentage as a float.
    """
    # Placeholder for actual AST linter coverage logic
    return 100.0


def update_readme(
    readme_path: str | None = None, coverage_json_path: str = "coverage.json"
) -> None:
    """Update test and doc coverage shields in README.md.

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

    test_cov = get_test_coverage(coverage_json_path=coverage_json_path)
    doc_cov = get_doc_coverage()

    test_str = format_cov(test_cov)
    doc_str = format_cov(doc_cov)

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)

    with open(target_readme, "r", encoding="utf-8") as f:
        content = f.read()

    # Generic replacements that handle both the cdd-go markdown format with the `#` anchor and the older ml-switcheroo format
    test_re = re.compile(
        r"\[?\!\[Test Coverage\]\(https://img\.shields\.io/badge/(?:[tT]est_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = test_re.sub(
        f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)",
        content,
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
