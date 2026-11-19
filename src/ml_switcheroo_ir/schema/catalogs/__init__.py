"""Bundled grounded operator schema JSON catalogs for standalone ml_switcheroo_ir installations."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

__all__ = [
    "get_catalog_path",
]


def get_catalog_path(filename: str) -> Path:
    """Resolve the filesystem path to a bundled JSON operator catalog resource.

    Args:
        filename (str): Catalog JSON filename (e.g. ``'wgsl_ops.json'``).

    Returns:
        Path: Resolved filesystem ``Path`` to the catalog resource.
    """
    resource = importlib.resources.files("ml_switcheroo_ir.schema.catalogs").joinpath(filename)
    return Path(str(resource))
