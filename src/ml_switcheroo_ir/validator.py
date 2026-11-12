"""Validator facade re-exporting schema, dialect, hardware, distributed, and grounding validators."""

from __future__ import annotations

from ml_switcheroo_ir.schema.ghost import (
    WGSL_COMPUTE_BUILTINS,
    WGSL_MUTATING_OPS,
)
from ml_switcheroo_ir.snapshots import (
    DEFAULT_SNAPSHOT_DIR,
    get_default_snapshots_dir,
)
from ml_switcheroo_ir.snapshots import (
    get_default_snapshots_dir as _get_default_snapshots_dir,
)
from ml_switcheroo_ir.validation import *  # noqa: F403
from ml_switcheroo_ir.validation import VALIDATION_PUBLIC_API
from ml_switcheroo_ir.validation.hardware import (
    _extract_dst_vgpr,
    _extract_sass_registers,
    _extract_vgpr_indices,
)

__all__ = [
    *VALIDATION_PUBLIC_API,
    "DEFAULT_SNAPSHOT_DIR",
    "WGSL_COMPUTE_BUILTINS",
    "WGSL_MUTATING_OPS",
    "_extract_dst_vgpr",
    "_extract_sass_registers",
    "_extract_vgpr_indices",
    "_get_default_snapshots_dir",
    "get_default_snapshots_dir",
]
