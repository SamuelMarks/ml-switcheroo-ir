"""Pytest configuration and benchmark collection hooks for ml-switcheroo-ir."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options to pytest.

    Args:
        parser (pytest.Parser): Pytest command-line argument parser.
    """
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="run heavy benchmarks",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers.

    Args:
        config (pytest.Config): Pytest configuration object.
    """
    config.addinivalue_line("markers", "benchmark: mark test as a heavy benchmark")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip benchmark tests unless --run-benchmarks is explicitly supplied.

    Args:
        config (pytest.Config): Pytest configuration object.
        items (list[pytest.Item]): Collected test items.
    """
    if config.getoption("--run-benchmarks"):
        return
    skip_bench = pytest.mark.skip(reason="need --run-benchmarks option to run")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_bench)
