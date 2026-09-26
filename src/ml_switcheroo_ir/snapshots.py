"""Snapshot resolution and discovery utilities for ml_switcheroo_ir."""

from __future__ import annotations

import os
from pathlib import Path


def get_default_snapshots_dir() -> str:
    """Discover and return default snapshot directory location.

    Checks locations in the following precedence order:
        1. ML_ECOSYSTEM_SNAPSHOTS_DIR environment variable.
        2. Sibling directory ../ml-ecosystem-snapshots/src/ml_framework_snapshots/snapshots/
           and ../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots/.
        3. User cache directory (~/.cache/ml_ecosystem_snapshots).
        4. ML_FRAMEWORK_SNAPSHOTS_DIR environment variable.
        5. User cache directory (~/.cache/ml_framework_snapshots).
        6. Installed ml_ecosystem_snapshots or ml_framework_snapshots package directory.
        7. Sibling directory ../ml-framework-snapshots/src/ml_framework_snapshots/snapshots/.
        8. Bundled offline fixture snapshots in tests/fixtures/snapshots.

    Returns:
        str: Absolute path to snapshot directory.
    """
    env_eco = os.environ.get("ML_ECOSYSTEM_SNAPSHOTS_DIR")
    if env_eco and os.path.isdir(env_eco):
        return os.path.abspath(env_eco)

    sibling_eco_fw = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "ml-ecosystem-snapshots",
            "src",
            "ml_framework_snapshots",
            "snapshots",
        )
    )
    if os.path.isdir(sibling_eco_fw) and any(
        f.endswith((".json", ".json.gz")) for f in os.listdir(sibling_eco_fw)
    ):
        return sibling_eco_fw

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

    env_fw = os.environ.get("ML_FRAMEWORK_SNAPSHOTS_DIR")
    if env_fw and os.path.isdir(env_fw):
        return os.path.abspath(env_fw)

    user_cache_fw = os.path.expanduser("~/.cache/ml_framework_snapshots")
    if os.path.isdir(user_cache_fw) and any(
        f.endswith((".json", ".json.gz")) for f in os.listdir(user_cache_fw)
    ):
        return os.path.abspath(user_cache_fw)

    try:
        import importlib.util

        for pkg_name in ("ml_ecosystem_snapshots", "ml_framework_snapshots"):
            spec = importlib.util.find_spec(pkg_name)
            if spec and spec.origin:
                pkg_dir = os.path.join(os.path.dirname(spec.origin), "snapshots")
                if os.path.isdir(pkg_dir) and any(
                    f.endswith((".json", ".json.gz")) for f in os.listdir(pkg_dir)
                ):
                    return os.path.abspath(pkg_dir)
    except (ImportError, AttributeError, ValueError):
        pass

    sibling_fw = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "ml-framework-snapshots",
            "src",
            "ml_framework_snapshots",
            "snapshots",
        )
    )
    if os.path.isdir(sibling_fw) and any(
        f.endswith((".json", ".json.gz")) for f in os.listdir(sibling_fw)
    ):
        return sibling_fw

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


def find_schema_file(
    filename: str, override_path: Path | str | None = None
) -> Path | None:
    """Resolve the location of a schema or snapshot JSON file.

    Args:
        filename (str): Name of the file to resolve (e.g. 'onnx_ops.json').
        override_path (Optional[Union[Path, str]]): Explicit file path override.

    Returns:
        Optional[Path]: Resolved Path if the file exists on disk, or None.
    """
    if override_path is not None:
        p = Path(override_path)
        if p.is_file():
            return p
        return None

    candidate = Path(DEFAULT_SNAPSHOT_DIR) / filename
    if candidate.is_file():
        return candidate

    sibling_schema = Path(__file__).parent / "schema" / filename
    if sibling_schema.is_file():
        return sibling_schema

    return None
