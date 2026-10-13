from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add pytest options."""
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="run heavy benchmarks",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "benchmark: mark test as a heavy benchmark")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify collection for benchmarks."""
    if config.getoption("--run-benchmarks"):
        return
    skip_bench = pytest.mark.skip(reason="need --run-benchmarks option to run")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_bench)
