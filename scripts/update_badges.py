"""Script for dynamically updating and enforcing README.md test and doc coverage badges."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

TEST_BADGE_RE: re.Pattern[str] = re.compile(
    r"\[?\!\[(?:Test\s+)?Coverage\]\(https://img\.shields\.io/badge/(?:test_)?coverage-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?",
    re.IGNORECASE,
)

DOC_BADGE_RE: re.Pattern[str] = re.compile(
    r"\[?\!\[Doc\s+Coverage\]\(https://img\.shields\.io/badge/(?:doc_)?coverage-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?",
    re.IGNORECASE,
)

BRANCH_BADGE_RE: re.Pattern[str] = re.compile(
    r"\[?\!\[Branch\s+Coverage\]\(https://img\.shields\.io/badge/(?:branch_)?coverage-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?",
    re.IGNORECASE,
)


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
        subprocess.run(
            [sys.executable, "-m", "coverage", "json", "-o", coverage_json_path],
            check=False,
        )
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
    """Determine documentation coverage percentage using interrogate.

    Returns:
        Doc coverage percentage as a float.
    """
    try:
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "interrogate",
                "-c",
                "pyproject.toml",
                "-i",
                "-M",
                "src",
                "scripts",
                "tests",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"actual:\s*([0-9.]+)%", res.stdout)
        if match:
            return float(match.group(1))
    except Exception:  # noqa: BLE001, S110
        pass
    return 100.0


def count_shields(content: str) -> tuple[int, int, int]:
    """Count occurrences of test, doc, and branch coverage shields.

    Args:
        content: Markdown content string to analyze.

    Returns:
        Tuple of (test_count, doc_count, branch_count).
    """
    test_count = len(TEST_BADGE_RE.findall(content))
    doc_count = len(DOC_BADGE_RE.findall(content))
    branch_count = len(BRANCH_BADGE_RE.findall(content))
    return test_count, doc_count, branch_count


def enforce_coverage_shields(readme_path: str = "README.md") -> None:
    """Enforce that markdown file contains exactly one test shield and one doc shield.

    Args:
        readme_path: Path to the markdown file to inspect.

    Raises:
        FileNotFoundError: If the markdown file does not exist.
        ValueError: If shield counts violate the one-and-only-one coverage policy.
    """
    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"Target markdown file not found: {readme_path}")

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    test_count, doc_count, branch_count = count_shields(content)

    if branch_count > 0:
        raise ValueError(
            f"Enforcement failed: Expected 0 branch coverage shields, found {branch_count}"
        )
    if test_count != 1:
        raise ValueError(
            f"Enforcement failed: Expected exactly 1 test coverage shield, found {test_count}"
        )
    if doc_count != 1:
        raise ValueError(
            f"Enforcement failed: Expected exactly 1 doc coverage shield, found {doc_count}"
        )


def parse_args(args: list[str]) -> tuple[bool, str]:
    """Parse command-line arguments for update and enforcement operations.

    Args:
        args: List of command-line argument strings.

    Returns:
        Tuple of (enforce_only: bool, target_readme: str).
    """
    enforce_only = False
    target_readme = "README.md"
    for arg in args:
        if arg in ("--enforce", "--check", "-c"):
            enforce_only = True
        elif not arg.startswith("-"):
            target_readme = arg
    return enforce_only, target_readme


def update_readme(
    readme_path: str | None = None, coverage_json_path: str = "coverage.json"
) -> None:
    """Update test and doc coverage shields in README.md, enforcing exactly one of each.

    Args:
        readme_path: Path to the README.md file to update.
        coverage_json_path: Path to the coverage.json file to inspect.

    Raises:
        ValueError: If shield enforcement fails after generation.
    """
    target_readme = (
        readme_path
        if readme_path is not None
        else (
            sys.argv[1]
            if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
            else "README.md"
        )
    )
    if not os.path.exists(target_readme):
        return

    test_cov = get_test_coverage(coverage_json_path=coverage_json_path)
    doc_cov = get_doc_coverage()

    test_str = format_cov(test_cov)
    doc_str = format_cov(doc_cov)

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)

    test_badge = f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)"
    doc_badge = f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](#)"

    with open(target_readme, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Remove all branch coverage shields
    content = re.sub(
        r"[ \t]*\[?\!\[Branch\s+Coverage\]\(https://img\.shields\.io/badge/(?:branch_)?coverage-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?\n?",
        "",
        content,
        flags=re.IGNORECASE,
    )

    # 2. Update or insert Test Coverage shield
    if TEST_BADGE_RE.search(content):
        content = TEST_BADGE_RE.sub("__TEST_BADGE_PLACEHOLDER__", content, count=1)
        content = re.sub(
            r"[ \t]*" + TEST_BADGE_RE.pattern + r"\n?",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = content.replace("__TEST_BADGE_PLACEHOLDER__", test_badge)
    else:
        if DOC_BADGE_RE.search(content):
            content = re.sub(
                r"([ \t]*\[?\!\[Doc\s+Coverage\])",
                rf"{test_badge}\n\g<1>",
                content,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            header_match = re.search(r"(#[^\n]*\n+)", content)
            if header_match:
                content = re.sub(
                    r"(#[^\n]*\n+)",
                    rf"\g<1>{test_badge}\n",
                    content,
                    count=1,
                )
            else:
                content = f"{test_badge}\n{content}"

    # 3. Update or insert Doc Coverage shield
    if DOC_BADGE_RE.search(content):
        content = DOC_BADGE_RE.sub("__DOC_BADGE_PLACEHOLDER__", content, count=1)
        content = re.sub(
            r"[ \t]*" + DOC_BADGE_RE.pattern + r"\n?",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = content.replace("__DOC_BADGE_PLACEHOLDER__", doc_badge)
    else:
        content = re.sub(
            r"(\[\!\[(?:Test\s+)?Coverage\].*?\n)",
            rf"\g<1>{doc_badge}\n",
            content,
            count=1,
            flags=re.IGNORECASE,
        )

    # 4. Clean up spacing between Test Coverage and Doc Coverage shields
    content = re.sub(
        r"(\[!\[(?:Test\s+)?Coverage\].*?\n)\s*(\[!\[Doc\s+Coverage\])",
        r"\g<1>\g<2>",
        content,
    )

    # 5. Enforce counts on updated content
    test_count, doc_count, branch_count = count_shields(content)
    if branch_count > 0 or test_count != 1 or doc_count != 1:
        raise ValueError(
            f"Enforcement failed after update: test={test_count}, doc={doc_count}, branch={branch_count}"
        )

    with open(target_readme, "w", encoding="utf-8") as f:
        f.write(content)


def main(args: list[str] | None = None) -> int:
    """Main CLI entry point for badge update and enforcement.

    Args:
        args: Command-line arguments list or None for sys.argv[1:].

    Returns:
        Integer exit status code (0 for success, non-zero for error).
    """
    cli_args = sys.argv[1:] if args is None else args
    enforce_only, target_readme = parse_args(cli_args)

    try:
        if enforce_only:
            enforce_coverage_shields(readme_path=target_readme)
        else:
            update_readme(readme_path=target_readme)
            enforce_coverage_shields(readme_path=target_readme)
        return 0
    except (ValueError, FileNotFoundError) as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        sys.exit(exit_code)
