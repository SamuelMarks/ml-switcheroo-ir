"""Script for dynamically updating and enforcing README.md test and doc coverage badges."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

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
        pct (float): Coverage percentage between 0.0 and 100.0.

    Returns:
        str: String color name for shields.io badge.
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
        cov (float): Coverage percentage as a floating-point number.

    Returns:
        str: Formatted string without decimal places if whole number, else one decimal.
    """
    if int(cov) == cov:
        return str(int(cov))
    return f"{cov:.1f}"


def get_coverage_metrics(
    coverage_json_path: str | None = None,
) -> tuple[float, float, float]:
    """Extract overall, statement, and branch coverage percentages from coverage.json.

    Args:
        coverage_json_path (str | None): Optional path to an existing JSON coverage file to inspect.

    Returns:
        tuple[float, float, float]: ``(overall_pct, statement_pct, branch_pct)``.
    """
    actual_path = coverage_json_path
    is_temp = False
    if actual_path is None:
        fd, actual_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        is_temp = True
    try:
        subprocess.run(
            [sys.executable, "-m", "coverage", "json", "-o", actual_path],
            check=False,
        )
        with open(actual_path, encoding="utf-8") as f:
            data = json.load(f)
            totals = data.get("totals", {})
            overall = float(totals.get("percent_covered", 0.0))
            num_stmts = totals.get("num_statements", 0)
            cov_lines = totals.get("covered_lines", 0)
            stmt_cov = (cov_lines / num_stmts * 100.0) if num_stmts > 0 else 100.0

            num_branches = totals.get("num_branches", 0)
            cov_branches = totals.get("covered_branches", 0)
            branch_cov = (cov_branches / num_branches * 100.0) if num_branches > 0 else 100.0
            return overall, stmt_cov, branch_cov
    except (
        subprocess.SubprocessError,
        OSError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
    ):
        return 0.0, 0.0, 0.0
    finally:
        if is_temp and os.path.exists(actual_path):
            try:
                os.remove(actual_path)
            except OSError:
                pass


def get_test_coverage(coverage_json_path: str | None = None) -> float:
    """Extract total test coverage percentage from coverage.json or run coverage tool.

    Args:
        coverage_json_path (str | None): Optional path to the JSON coverage file to inspect or generate.

    Returns:
        float: Total coverage percentage as a float.
    """
    overall, _, _ = get_coverage_metrics(coverage_json_path=coverage_json_path)
    return overall


def get_doc_coverage() -> float:
    """Determine documentation coverage percentage using interrogate.

    Returns:
        float: Doc coverage percentage as a float.

    Raises:
        ValueError: If documentation coverage cannot be determined or is 0%.
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
            val = float(match.group(1))
            if val > 0.0:
                return val
    except (
        subprocess.SubprocessError,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as err:
        raise ValueError("Failed to determine documentation coverage from interrogate.") from err
    raise ValueError("Failed to determine documentation coverage from interrogate.")


def count_shields(content: str) -> tuple[int, int, int]:
    """Count occurrences of test, doc, and branch coverage shields.

    Args:
        content (str): Markdown content string to analyze.

    Returns:
        tuple[int, int, int]: Tuple of ``(test_count, doc_count, branch_count)``.
    """
    test_count = len(TEST_BADGE_RE.findall(content))
    doc_count = len(DOC_BADGE_RE.findall(content))
    branch_count = len(BRANCH_BADGE_RE.findall(content))
    return test_count, doc_count, branch_count


def enforce_coverage_shields(readme_path: str = "README.md") -> None:
    """Enforce that markdown file contains exactly one test shield and one doc shield.

    Args:
        readme_path (str): Path to the markdown file to inspect.

    Raises:
        FileNotFoundError: If the markdown file does not exist.
        ValueError: If shield counts violate the one-and-only-one coverage policy.
    """
    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"Target markdown file not found: {readme_path}")

    with open(readme_path, encoding="utf-8") as f:
        content = f.read()

    test_count, doc_count, branch_count = count_shields(content)

    if branch_count > 0:
        raise ValueError(f"Enforcement failed: Expected 0 branch coverage shields, found {branch_count}")
    if test_count != 1:
        raise ValueError(f"Enforcement failed: Expected exactly 1 test coverage shield, found {test_count}")
    if doc_count != 1:
        raise ValueError(f"Enforcement failed: Expected exactly 1 doc coverage shield, found {doc_count}")
    if re.search(r"coverage-0(?:\.0+)?%25", content, flags=re.IGNORECASE):
        raise ValueError("Enforcement failed: Coverage shield cannot be 0%")


def parse_args(args: list[str]) -> tuple[bool, str]:
    """Parse command-line arguments for update and enforcement operations via ``argparse``.

    Args:
        args (list[str]): List of command-line argument strings.

    Returns:
        tuple[bool, str]: Tuple of ``(enforce_only, target_readme)``.
    """
    parser = argparse.ArgumentParser(
        description="Update and enforce README.md test and documentation coverage shields."
    )
    parser.add_argument(
        "readme",
        nargs="?",
        default="README.md",
        help="Path to target README markdown file (default: README.md).",
    )
    parser.add_argument(
        "-c",
        "--check",
        "--enforce",
        dest="enforce",
        action="store_true",
        help="Check shield invariants without modifying the README file.",
    )
    parsed = parser.parse_args(args)
    return bool(parsed.enforce), str(parsed.readme)


_NL: str = chr(10)
_HTAB: str = chr(9)


def _apply_badge_replacements(content: str, test_badge: str, doc_badge: str) -> str:
    """Update or insert test and doc coverage badges in README markdown content.

    Args:
        content (str): Original README markdown text.
        test_badge (str): Rendered Test Coverage markdown badge.
        doc_badge (str): Rendered Doc Coverage markdown badge.

    Returns:
        str: Updated README markdown text.
    """
    ws_cls = f"[ {_HTAB}]*"
    content = re.sub(
        ws_cls
        + r"\[?\!\[Branch\s+Coverage\]\(https://img\.shields\.io/badge/(?:branch_)?coverage-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
        + _NL
        + "?",
        "",
        content,
        flags=re.IGNORECASE,
    )

    if TEST_BADGE_RE.search(content):
        content = TEST_BADGE_RE.sub("__TEST_BADGE_PLACEHOLDER__", content, count=1)
        content = re.sub(
            ws_cls + TEST_BADGE_RE.pattern + _NL + "?",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = content.replace("__TEST_BADGE_PLACEHOLDER__", test_badge)
    elif DOC_BADGE_RE.search(content):
        content = re.sub(
            r"(" + ws_cls + r"\[?\!\[Doc\s+Coverage\])",
            test_badge + _NL + r"\g<1>",
            content,
            count=1,
            flags=re.IGNORECASE,
        )
    elif re.search(f"(#[^\\r{_NL}]*{_NL}+)", content):
        content = re.sub(
            f"(#[^\\r{_NL}]*{_NL}+)",
            r"\g<1>" + test_badge + _NL,
            content,
            count=1,
        )
    else:
        content = f"{test_badge}{_NL}{content}"

    if DOC_BADGE_RE.search(content):
        content = DOC_BADGE_RE.sub("__DOC_BADGE_PLACEHOLDER__", content, count=1)
        content = re.sub(
            ws_cls + DOC_BADGE_RE.pattern + _NL + "?",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = content.replace("__DOC_BADGE_PLACEHOLDER__", doc_badge)
    else:
        content = re.sub(
            r"(\[\!\[(?:Test\s+)?Coverage\].*?" + _NL + ")",
            r"\g<1>" + doc_badge + _NL,
            content,
            count=1,
            flags=re.IGNORECASE,
        )

    return re.sub(
        r"(\[!\[(?:Test\s+)?Coverage\].*?" + _NL + r")\s*(\[!\[Doc\s+Coverage\])",
        r"\g<1>\g<2>",
        content,
    )


def update_readme(readme_path: str | None = None, coverage_json_path: str | None = None) -> None:
    """Update test and doc coverage shields in README.md, enforcing exactly one of each.

    Args:
        readme_path (str | None): Path to the README.md file to update.
        coverage_json_path (str | None): Optional path to an existing coverage.json file to inspect.

    Raises:
        ValueError: If shield enforcement fails after generation or coverage is 0%.
    """
    target_readme = (
        readme_path
        if readme_path is not None
        else (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "README.md")
    )
    if not os.path.exists(target_readme):
        return

    actual_cov_path = coverage_json_path
    if actual_cov_path is None:
        sibling_cov = os.path.join(os.path.dirname(target_readme), "coverage.json")
        if os.path.isfile(sibling_cov):
            actual_cov_path = sibling_cov

    test_cov = get_test_coverage(coverage_json_path=actual_cov_path)
    if test_cov <= 0.0:
        raise ValueError("Failed to determine test coverage or test coverage is 0%.")
    doc_cov = get_doc_coverage()

    test_str = format_cov(test_cov)
    doc_str = format_cov(doc_cov)

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)

    test_badge = f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)"
    doc_badge = f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](#)"

    with open(target_readme, encoding="utf-8") as f:
        content = f.read()

    content = _apply_badge_replacements(content, test_badge, doc_badge)

    test_count, doc_count, branch_count = count_shields(content)
    if branch_count > 0 or test_count != 1 or doc_count != 1:
        raise ValueError(f"Enforcement failed after update: test={test_count}, doc={doc_count}, branch={branch_count}")

    with open(target_readme, "w", encoding="utf-8") as f:
        f.write(content)


def main(args: list[str] | None = None) -> int:
    """Run main CLI entry point for badge update and enforcement.

    Args:
        args (list[str] | None): Command-line arguments list or None for ``sys.argv[1:]``.

    Returns:
        int: Integer exit status code (0 for success, non-zero for error).
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
        print(f"Error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        sys.exit(exit_code)
