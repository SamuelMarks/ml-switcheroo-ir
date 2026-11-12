"""Pytest configuration, global registry isolation guards, and benchmark hooks for ml-switcheroo-ir."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ml_switcheroo_ir.schema import (
    ARRAY_API_REGISTRY,
    ATEN_REGISTRY,
    CHEX_REGISTRY,
    CUSTOM_OPS_REGISTRY,
    GRAIN_REGISTRY,
    METAL_REGISTRY,
    MLIR_REGISTRY,
    ODL_CATALOG,
    ODL_DIALECT_MAPPINGS,
    ONNX_REGISTRY,
    OPTAX_REGISTRY,
    ORBAX_REGISTRY,
    PTX_REGISTRY,
    RDNA_REGISTRY,
    RDNA_TO_VOPD_MAP,
    RDNA_VOPD_SLOTS,
    SASS_PIPELINE_LATENCIES,
    SASS_REGISTRY,
    STABLEHLO_REGISTRY,
    STATE_OPS_REGISTRY,
    WASM_REGISTRY,
    WEBGL_REGISTRY,
    WGSL_REGISTRY,
)
from ml_switcheroo_ir.schema.custom_ops import (
    COLLECTIVE_OPS_REGISTRY,
    QUANTIZATION_OPS_REGISTRY,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options for benchmark execution.

    Args:
        parser (pytest.Parser): Pytest command-line option parser.
    """
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="run long-running throughput and streaming benchmarks",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used across the test suite.

    Args:
        config (pytest.Config): Pytest configuration instance.
    """
    config.addinivalue_line(
        "markers",
        "benchmark: mark test as a throughput or streaming benchmark",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Retain all collected benchmark and unit items so default runs execute with zero skips.

    Args:
        config (pytest.Config): Pytest configuration instance.
        items (list[pytest.Item]): Collected test items.
    """
    _ = (config, items)


_GLOBAL_REGISTRIES = (
    ("ONNX_REGISTRY", ONNX_REGISTRY),
    ("STABLEHLO_REGISTRY", STABLEHLO_REGISTRY),
    ("MLIR_REGISTRY", MLIR_REGISTRY),
    ("CUSTOM_OPS_REGISTRY", CUSTOM_OPS_REGISTRY),
    ("COLLECTIVE_OPS_REGISTRY", COLLECTIVE_OPS_REGISTRY),
    ("QUANTIZATION_OPS_REGISTRY", QUANTIZATION_OPS_REGISTRY),
    ("STATE_OPS_REGISTRY", STATE_OPS_REGISTRY),
    ("ATEN_REGISTRY", ATEN_REGISTRY),
    ("ARRAY_API_REGISTRY", ARRAY_API_REGISTRY),
    ("CHEX_REGISTRY", CHEX_REGISTRY),
    ("GRAIN_REGISTRY", GRAIN_REGISTRY),
    ("ORBAX_REGISTRY", ORBAX_REGISTRY),
    ("OPTAX_REGISTRY", OPTAX_REGISTRY),
    ("ODL_CATALOG", ODL_CATALOG),
    ("RDNA_REGISTRY", RDNA_REGISTRY),
    ("SASS_REGISTRY", SASS_REGISTRY),
    ("WGSL_REGISTRY", WGSL_REGISTRY),
    ("PTX_REGISTRY", PTX_REGISTRY),
    ("METAL_REGISTRY", METAL_REGISTRY),
    ("WASM_REGISTRY", WASM_REGISTRY),
    ("WEBGL_REGISTRY", WEBGL_REGISTRY),
)

_GLOBAL_METADATA_MAPS = (
    ("ODL_DIALECT_MAPPINGS", ODL_DIALECT_MAPPINGS),
    ("RDNA_VOPD_SLOTS", RDNA_VOPD_SLOTS),
    ("RDNA_TO_VOPD_MAP", RDNA_TO_VOPD_MAP),
    ("SASS_PIPELINE_LATENCIES", SASS_PIPELINE_LATENCIES),
)


@pytest.fixture(autouse=True)
def _verify_global_state_isolation() -> Iterator[None]:
    """Verify no test leaks mutations to the 21 global registries, metadata maps, or CLI bindings.

    Yields:
        None: Executes the test within a before/after global-state snapshot guard.
    """
    import ml_switcheroo_ir.cli as cli_mod

    before_keys = {name: set(reg.keys()) for name, reg in _GLOBAL_REGISTRIES}
    before_meta = {name: dict(m.items()) for name, m in _GLOBAL_METADATA_MAPS}
    before_tabulate = cli_mod.tabulate
    yield
    after_keys = {name: set(reg.keys()) for name, reg in _GLOBAL_REGISTRIES}
    after_meta = {name: dict(m.items()) for name, m in _GLOBAL_METADATA_MAPS}
    assert after_keys == before_keys, "Global schema registry state leaked across test boundary!"
    assert after_meta == before_meta, "Global dialect metadata map leaked across test boundary!"
    assert cli_mod.tabulate is before_tabulate, "ml_switcheroo_ir.cli.tabulate binding leaked across test boundary!"
