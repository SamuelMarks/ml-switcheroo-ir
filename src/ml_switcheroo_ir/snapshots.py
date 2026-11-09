"""Snapshot resolution and discovery utilities for ml_switcheroo_ir."""

from __future__ import annotations

import os
from pathlib import Path


def _dir_has_json_files(directory: str) -> bool:
    """Return True if a directory path exists and contains at least one JSON snapshot file.

    Args:
        directory (str): Directory path to inspect.

    Returns:
        bool: True if at least one .json or .json.gz file is present.
    """
    return os.path.isdir(directory) and any(f.endswith((".json", ".json.gz")) for f in os.listdir(directory))


def _find_installed_package_snapshots_dir() -> str | None:
    """Locate snapshots directory inside an installed snapshot package if available.

    Returns:
        str | None: Absolute path to installed package snapshots directory, or None.
    """
    try:
        import importlib.util

        for pkg_name in ("ml_framework_snapshots", "ml_ecosystem_snapshots"):
            spec = importlib.util.find_spec(pkg_name)
            if spec and spec.origin:
                pkg_dir = os.path.join(os.path.dirname(spec.origin), "snapshots")
                if _dir_has_json_files(pkg_dir):
                    return os.path.abspath(pkg_dir)
    except (ImportError, AttributeError, ValueError):
        pass
    return None


def get_default_snapshots_dir() -> str:
    """Discover and return default snapshot directory location.

    Checks locations in the following precedence order:
        1. ML_FRAMEWORK_SNAPSHOTS_DIR or ML_ECOSYSTEM_SNAPSHOTS_DIR environment variable.
        2. Sibling directory ../ml-framework-snapshots or ../ml-ecosystem-snapshots.
        3. User cache directory (~/.cache/ml_framework_snapshots or ~/.cache/ml_ecosystem_snapshots).
        4. Installed ml_framework_snapshots or ml_ecosystem_snapshots package directory.
        5. Bundled offline fixture snapshots in tests/fixtures/snapshots.

    Returns:
        str: Absolute path to snapshot directory.
    """
    for env_key in ("ML_FRAMEWORK_SNAPSHOTS_DIR", "ML_ECOSYSTEM_SNAPSHOTS_DIR"):
        env_val = os.environ.get(env_key)
        if env_val and os.path.isdir(env_val):
            return os.path.abspath(env_val)

    base_dir = os.path.dirname(__file__)
    sibling_candidates = (
        os.path.abspath(os.path.join(base_dir, "..", "..", "..", "ml-framework-snapshots", "src", "ml_framework_snapshots", "snapshots")),
        os.path.abspath(os.path.join(base_dir, "..", "..", "..", "ml-framework-snapshots", "snapshots")),
        os.path.abspath(os.path.join(base_dir, "..", "..", "..", "ml-framework-snapshots")),
        os.path.abspath(os.path.join(base_dir, "..", "..", "..", "ml-ecosystem-snapshots", "src", "ml_ecosystem_snapshots", "snapshots")),
    )
    for cand in sibling_candidates:
        if _dir_has_json_files(cand):
            return cand

    for cache_rel in ("~/.cache/ml_framework_snapshots", "~/.cache/ml_ecosystem_snapshots"):
        user_cache = os.path.expanduser(cache_rel)
        if _dir_has_json_files(user_cache):
            return os.path.abspath(user_cache)

    installed_dir = _find_installed_package_snapshots_dir()
    if installed_dir is not None:
        return installed_dir

    fixtures_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "tests", "fixtures", "snapshots"))
    if _dir_has_json_files(fixtures_dir):
        return fixtures_dir

    return sibling_candidates[-1]


_INITIAL_DEFAULT_SNAPSHOT_DIR: str = get_default_snapshots_dir()
DEFAULT_SNAPSHOT_DIR: str = _INITIAL_DEFAULT_SNAPSHOT_DIR


def get_effective_snapshot_dir() -> str:
    """Return the active default snapshot directory, respecting runtime overrides.

    Checks whether ``DEFAULT_SNAPSHOT_DIR`` on ``snapshots`` or ``validator`` has been
    explicitly overridden from its initial value; otherwise resolves dynamically via
    ``get_default_snapshots_dir()`` if environment variables are active.

    Returns:
        str: Effective default snapshot directory path.
    """
    if DEFAULT_SNAPSHOT_DIR != _INITIAL_DEFAULT_SNAPSHOT_DIR:
        return DEFAULT_SNAPSHOT_DIR

    import sys

    val_mod = sys.modules.get("ml_switcheroo_ir.validator")
    if val_mod is not None:
        val_dir = str(val_mod.DEFAULT_SNAPSHOT_DIR)
        if val_dir != _INITIAL_DEFAULT_SNAPSHOT_DIR:
            return val_dir

    if "ML_FRAMEWORK_SNAPSHOTS_DIR" in os.environ or "ML_ECOSYSTEM_SNAPSHOTS_DIR" in os.environ:
        return get_default_snapshots_dir()

    return DEFAULT_SNAPSHOT_DIR


def _has_snapshot_files(directory: Path) -> bool:
    """Return True if the directory exists and contains at least one JSON snapshot file.

    Args:
        directory (Path): Directory path to inspect.

    Returns:
        bool: True if at least one .json or .json.gz file is present.
    """
    return directory.is_dir() and any(f.name.endswith((".json", ".json.gz")) for f in directory.iterdir())


def resolve_snapshots_path(override_path: Path | str, allow_default_fallback: bool = False) -> Path | None:
    """Resolve an explicit snapshots directory path, supporting repo-root and sibling layouts.

    Args:
        override_path (Path | str): Explicit path provided via CLI flag or API parameter.
        allow_default_fallback (bool): If True and the target is a standard sibling snapshot
            repo root whose nested snapshots directory is unpopulated on disk, fall back to
            ``get_effective_snapshot_dir()``. Defaults to False so explicit paths are honored.

    Returns:
        Path | None: Resolved snapshots directory path if found, or None.
    """
    p = Path(override_path).resolve()
    if p.is_dir():
        for pkg_sub in ("ml_framework_snapshots", "ml_ecosystem_snapshots"):
            nested = p / "src" / pkg_sub / "snapshots"
            if _has_snapshot_files(nested):
                return nested
            if nested.is_dir() or (p / "src" / pkg_sub).is_dir():
                if allow_default_fallback and p.name in ("ml-framework-snapshots", "ml-ecosystem-snapshots"):
                    return Path(get_effective_snapshot_dir()).resolve()
                return nested
        direct_snaps = p / "snapshots"
        if _has_snapshot_files(direct_snaps):
            return direct_snaps
        return p
    return None


def find_schema_file(filename: str, override_path: Path | str | None = None) -> Path | None:
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

    active_dir = Path(get_effective_snapshot_dir())
    candidate = active_dir / filename
    if candidate.is_file():
        return candidate

    parent_candidate = active_dir.parent / filename
    if parent_candidate.is_file():
        return parent_candidate

    sibling_schema = Path(__file__).parent / "schema" / filename
    if sibling_schema.is_file():
        return sibling_schema

    return None
