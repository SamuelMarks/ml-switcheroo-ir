"""Snapshot resolution and discovery utilities for ml_switcheroo_ir."""

from __future__ import annotations

import os
from pathlib import Path


def get_default_snapshots_dir() -> str:
    """Discover and return default snapshot directory location.

    Checks locations in the following precedence order:
        1. ML_ECOSYSTEM_SNAPSHOTS_DIR environment variable.
        2. Sibling directory ../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots/.
        3. User cache directory (~/.cache/ml_ecosystem_snapshots).
        4. Installed ml_ecosystem_snapshots package directory.
        5. Bundled offline fixture snapshots in tests/fixtures/snapshots.

    Returns:
        str: Absolute path to snapshot directory.
    """
    env_eco = os.environ.get("ML_ECOSYSTEM_SNAPSHOTS_DIR")
    if env_eco and os.path.isdir(env_eco):
        return os.path.abspath(env_eco)

    sibling_eco = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "ml-ecosystem-snapshots",
            "src",
            "ml_ecosystem_snapshots",
            "snapshots",
        )
    )
    if os.path.isdir(sibling_eco) and any(
        f.endswith((".json", ".json.gz")) for f in os.listdir(sibling_eco)
    ):
        return sibling_eco

    user_cache_eco = os.path.expanduser("~/.cache/ml_ecosystem_snapshots")
    if os.path.isdir(user_cache_eco) and any(
        f.endswith((".json", ".json.gz")) for f in os.listdir(user_cache_eco)
    ):
        return os.path.abspath(user_cache_eco)

    try:
        import importlib.util

        spec = importlib.util.find_spec("ml_ecosystem_snapshots")
        if spec and spec.origin:
            pkg_dir = os.path.join(os.path.dirname(spec.origin), "snapshots")
            if os.path.isdir(pkg_dir) and any(
                f.endswith((".json", ".json.gz")) for f in os.listdir(pkg_dir)
            ):
                return os.path.abspath(pkg_dir)
    except (ImportError, AttributeError, ValueError):
        pass

    fixtures_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "tests",
            "fixtures",
            "snapshots",
        )
    )
    if os.path.isdir(fixtures_dir) and any(
        f.endswith((".json", ".json.gz")) for f in os.listdir(fixtures_dir)
    ):
        return fixtures_dir

    return sibling_eco


DEFAULT_SNAPSHOT_DIR: str = get_default_snapshots_dir()


def get_effective_snapshot_dir() -> str:
    """Return the active default snapshot directory, respecting overrides on validator.DEFAULT_SNAPSHOT_DIR.

    Returns:
        str: Effective default snapshot directory path.
    """
    import sys

    val_mod = sys.modules.get("ml_switcheroo_ir.validator")
    return str(getattr(val_mod, "DEFAULT_SNAPSHOT_DIR", DEFAULT_SNAPSHOT_DIR))


def _has_snapshot_files(directory: Path) -> bool:
    """Return True if the directory exists and contains at least one JSON snapshot file.

    Args:
        directory (Path): Directory path to inspect.

    Returns:
        bool: True if at least one .json or .json.gz file is present.
    """
    return directory.is_dir() and any(
        f.name.endswith((".json", ".json.gz")) for f in directory.iterdir()
    )


def resolve_snapshots_path(override_path: Path | str) -> Path | None:
    """Resolve an explicit snapshots directory path, supporting repo-root and sibling layouts.

    Args:
        override_path (Path | str): Explicit path provided via CLI flag or API parameter.

    Returns:
        Path | None: Resolved snapshots directory path if found, or None.
    """
    p = Path(override_path).resolve()
    if p.is_dir():
        nested_eco = p / "src" / "ml_ecosystem_snapshots" / "snapshots"
        if _has_snapshot_files(nested_eco):
            return nested_eco
        if nested_eco.is_dir():
            if (
                p.parent.resolve()
                == Path(__file__).resolve().parent.parent.parent.parent
            ):
                return Path(DEFAULT_SNAPSHOT_DIR).resolve()
            return nested_eco
        return p
    return None


def find_schema_file(
    filename: str, override_path: Path | str | None = None
) -> Path | None:
    """Resolve the location of a schema or snapshot JSON file.

    Args:
        filename (str): Name of the file to resolve (e.g. 'onnx_ops.json').
        override_path (Path | str | None): Explicit file path override.

    Returns:
        Path | None: Resolved Path if the file exists on disk, or None.
    """
    if override_path is not None:
        p = Path(override_path)
        if p.is_file():
            return p
        return None

    candidate = Path(DEFAULT_SNAPSHOT_DIR) / filename
    if candidate.is_file():
        return candidate

    parent_candidate = Path(DEFAULT_SNAPSHOT_DIR).parent / filename
    if parent_candidate.is_file():
        return parent_candidate

    sibling_schema = Path(__file__).parent / "schema" / filename
    if sibling_schema.is_file():
        return sibling_schema

    return None
