"""Snapshot resolution and discovery utilities for ml_switcheroo_ir."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _dir_has_json_files(directory: str) -> bool:
    """Return True if ``directory`` exists and contains at least one ``.json`` or ``.json.gz`` file.

    Args:
        directory (str): Directory path to inspect.

    Returns:
        bool: True if ``directory`` is an existing directory with snapshot JSON files.
    """
    return os.path.isdir(directory) and any(f.endswith((".json", ".json.gz")) for f in os.listdir(directory))


def _has_snapshot_files(directory: str) -> bool:
    """Return True if ``directory`` or any canonical nested snapshot subdirectory contains snapshot files.

    Args:
        directory (str): Base directory path to inspect.

    Returns:
        bool: True if snapshot ``.json`` manifests exist in ``directory`` or standard subdirectories.
    """
    if _dir_has_json_files(directory):
        return True
    for rel in (
        os.path.join("snapshots"),
        os.path.join("src", "ml_framework_snapshots", "snapshots"),
        os.path.join("src", "ml_ecosystem_snapshots", "snapshots"),
    ):
        if _dir_has_json_files(os.path.join(directory, rel)):
            return True
    return False


def _resolve_nested_snapshot_subdir(base_dir: Path) -> Path | None:
    """Resolve standard nested snapshot subdirectories under ``base_dir`` if present.

    Args:
        base_dir (Path): Root snapshot repository or package path.

    Returns:
        Path | None: Resolved nested snapshot directory if discovered, else None.
    """
    nested_candidates = (
        base_dir / "src" / "ml_framework_snapshots" / "snapshots",
        base_dir / "src" / "ml_ecosystem_snapshots" / "snapshots",
        base_dir / "snapshots",
    )
    for cand in nested_candidates:
        if _dir_has_json_files(str(cand)):
            return cand.resolve()
    for cand in nested_candidates:
        if os.path.isdir(str(cand)):
            return cand.resolve()
    return None


def _resolve_populated_candidate(candidate: Path) -> str | None:
    """Return resolved snapshot directory string if ``candidate`` contains snapshot JSON files.

    Args:
        candidate (Path): Candidate root or cache directory.

    Returns:
        str | None: Resolved directory path string if populated, else None.
    """
    if _has_snapshot_files(str(candidate)):
        nested = _resolve_nested_snapshot_subdir(candidate)
        return str(nested if nested is not None else candidate.resolve())
    return None


def _resolve_package_snapshots_dir() -> str | None:
    """Locate snapshots bundled inside an installed ``ml_ecosystem_snapshots`` package via ``find_spec``.

    Returns:
        str | None: Absolute path to package ``snapshots`` directory if populated, else None.
    """
    try:
        spec = importlib.util.find_spec("ml_ecosystem_snapshots")
    except (ImportError, ValueError, TypeError):
        return None
    if spec is not None and spec.origin is not None:
        pkg_snaps = os.path.join(os.path.dirname(os.path.abspath(spec.origin)), "snapshots")
        if _dir_has_json_files(pkg_snaps):
            return pkg_snaps
    return None


def _resolve_fallback_snapshot_dir(repo_parent: Path) -> str:
    """Resolve unpopulated fallback snapshot directory path under ``repo_parent``.

    Args:
        repo_parent (Path): Parent directory of the repository checkout.

    Returns:
        str: Default resolved fallback directory path.
    """
    fallback_fw = repo_parent / "ml-framework-snapshots"
    if os.path.isdir(str(fallback_fw)):
        nested = _resolve_nested_snapshot_subdir(fallback_fw)
        return str(nested if nested is not None else fallback_fw.resolve())
    eco_default = repo_parent / "ml-ecosystem-snapshots" / "src" / "ml_ecosystem_snapshots" / "snapshots"
    return str(eco_default.resolve())


def _resolve_env_snapshots_dir() -> str | None:
    """Resolve snapshot directory from ``ML_FRAMEWORK_SNAPSHOTS_DIR`` or ``ML_ECOSYSTEM_SNAPSHOTS_DIR``.

    Returns:
        str | None: Resolved directory path if configured in ``os.environ``, else None.
    """
    for env_var in ("ML_FRAMEWORK_SNAPSHOTS_DIR", "ML_ECOSYSTEM_SNAPSHOTS_DIR"):
        env_val = os.environ.get(env_var, "").strip()
        if env_val:
            found_env = _resolve_populated_candidate(Path(env_val))
            if found_env is not None:
                return found_env
            if os.path.isdir(env_val):
                nested_env = _resolve_nested_snapshot_subdir(Path(env_val))
                return str(nested_env) if nested_env is not None else os.path.abspath(env_val)
    return None


def get_default_snapshots_dir() -> str:
    """Discover the ground-truth snapshots directory lazily without import-time disk I/O.

    Prioritizes ``ML_FRAMEWORK_SNAPSHOTS_DIR``, ``ML_ECOSYSTEM_SNAPSHOTS_DIR``,
    ``../ml-framework-snapshots``, ``../ml-ecosystem-snapshots``, installed snapshot
    packages, and ``~/.cache/ml_ecosystem_snapshots``.

    Returns:
        str: Absolute path to the discovered snapshots directory.
    """
    env_resolved = _resolve_env_snapshots_dir()
    if env_resolved is not None:
        return env_resolved

    repo_parent = Path(__file__).resolve().parent.parent.parent.parent
    for sib_name in ("ml-framework-snapshots", "ml-ecosystem-snapshots"):
        found_sib = _resolve_populated_candidate(repo_parent / sib_name)
        if found_sib is not None:
            return found_sib

    pkg_dir = _resolve_package_snapshots_dir()
    if pkg_dir is not None:
        return pkg_dir

    for cache_rel in ("~/.cache/ml_framework_snapshots", "~/.cache/ml_ecosystem_snapshots"):
        cache_dir = os.path.expanduser(cache_rel)
        if _dir_has_json_files(cache_dir):
            return os.path.abspath(cache_dir)

    fixtures_dir = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
    if _dir_has_json_files(str(fixtures_dir)):
        return str(fixtures_dir)

    return _resolve_fallback_snapshot_dir(repo_parent)


DEFAULT_SNAPSHOT_DIR: str = ""


def get_effective_snapshot_dir() -> str:
    """Return the active snapshot directory, honoring explicit module overrides and env vars.

    Returns:
        str: Effective snapshot directory path.
    """
    import ml_switcheroo_ir.validator as validator_mod

    if DEFAULT_SNAPSHOT_DIR:
        return DEFAULT_SNAPSHOT_DIR
    if validator_mod.DEFAULT_SNAPSHOT_DIR:
        return validator_mod.DEFAULT_SNAPSHOT_DIR
    return get_default_snapshots_dir()


def resolve_snapshots_path(
    raw_path: str | Path,
    *,
    allow_default_fallback: bool = False,
) -> Path | None:
    """Resolve a user-supplied snapshot directory or sibling repository root to its JSON directory.

    Args:
        raw_path (str | Path): Explicit directory or sibling repository path (e.g. ``../ml-framework-snapshots``).
        allow_default_fallback (bool): When True and ``raw_path`` is a repo shell whose nested
            snapshots live in the local user cache, fall back to ``get_effective_snapshot_dir()``.

    Returns:
        Path | None: Resolved path containing snapshot ``.json`` manifests, or ``None`` if ``raw_path`` does not exist.
    """
    base = Path(raw_path)
    if not os.path.exists(str(base)):
        return None
    if _dir_has_json_files(str(base)):
        return base.resolve()

    nested = _resolve_nested_snapshot_subdir(base)
    if nested is not None:
        if _dir_has_json_files(str(nested)) or not allow_default_fallback:
            return nested
        effective = Path(get_effective_snapshot_dir())
        if _dir_has_json_files(str(effective)):
            return effective.resolve()
        return nested

    return base.resolve()


def require_framework_snapshots_dir(raw_path: str | Path) -> Path:
    """Resolve ``raw_path`` and verify that it exists and contains at least one ``.json`` snapshot manifest.

    Args:
        raw_path (str | Path): Required framework snapshots directory path (e.g. ``../ml-framework-snapshots``).

    Returns:
        Path: Verified directory containing ground-truth ``.json`` snapshots.

    Raises:
        FileNotFoundError: If ``raw_path`` does not exist or contains no ``.json`` snapshot files.
    """
    resolved = resolve_snapshots_path(raw_path, allow_default_fallback=False)
    if resolved is None:
        raise FileNotFoundError(f"Required framework snapshots directory does not exist: {raw_path}")
    if not _dir_has_json_files(str(resolved)):
        raise FileNotFoundError(f"Required framework snapshots directory contains no .json manifests: {resolved}")
    return resolved


def find_schema_file(filename: str, override_path: Path | str | None = None) -> Path | None:
    """Locate a schema JSON file across override paths, snapshot directories, and bundled catalogs.

    Args:
        filename (str): Target JSON filename (e.g. ``'onnx_ops.json'``).
        override_path (Path | str | None): Optional explicit file path taking highest precedence.

    Returns:
        Path | None: Existing ``Path`` to the schema file, or ``None`` if not found.
    """
    if override_path is not None:
        candidate = Path(override_path)
        return candidate if candidate.is_file() else None

    snap_dir = Path(get_effective_snapshot_dir())
    direct_candidate = snap_dir / filename
    if direct_candidate.is_file():
        return direct_candidate

    parent_candidate = snap_dir.parent / filename
    if parent_candidate.is_file():
        return parent_candidate

    bundled_catalog = Path(__file__).parent / "schema" / "catalogs" / filename
    if bundled_catalog.is_file():
        return bundled_catalog

    sibling_schema = Path(__file__).parent / "schema" / filename
    if sibling_schema.is_file():
        return sibling_schema

    return None
