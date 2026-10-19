"""NVIDIA SASS architecture operator schemas and registry for ml_switcheroo_ir."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ml_switcheroo_ir.schema.base import (
    REGISTRY_LOCK,
    OpSchema,
    build_isa_primitive_schemas,
    load_op_schema_catalog,
)
from ml_switcheroo_ir.snapshots import find_schema_file

SASS_REGISTRY: dict[str, OpSchema] = {}
SASS_PIPELINE_LATENCIES: dict[str, int] = {}


def _load_sass_schemas(json_path: Path | str | None = None) -> None:
    """Load NVIDIA SASS instruction schemas from bundled sass_ops.json or snapshot directory.

    Args:
        json_path (Optional[Union[Path, str]]): Explicit path override for sass_ops.json.
    """

    def _record_sass_latency(op_data: Mapping[str, object]) -> None:
        """Record execution latency from a SASS schema entry.

        Args:
            op_data (Mapping[str, object]): Raw operator definition dictionary.
        """
        raw_lat = op_data.get("execution_latency", 4)
        SASS_PIPELINE_LATENCIES[str(op_data["name"])] = (
            int(raw_lat) if isinstance(raw_lat, (int, str)) else 4
        )

    with REGISTRY_LOCK:
        loaded = load_op_schema_catalog(
            "sass_ops.json",
            domain_default="nvidia_sass",
            json_path=json_path,
            find_file_fn=find_schema_file,
            on_entry=_record_sass_latency,
        )
        if loaded:
            SASS_REGISTRY.update(loaded)
        elif json_path is None:
            from ml_switcheroo_ir.schema.ghost import (
                SASS_INSTRUCTION_PRIMITIVES as GHOST_PRIMITIVES,
            )
            from ml_switcheroo_ir.schema.ghost import (
                SASS_PIPELINE_LATENCIES as GHOST_LATENCIES,
            )

            SASS_PIPELINE_LATENCIES.update(GHOST_LATENCIES)
            SASS_REGISTRY.update(
                build_isa_primitive_schemas(GHOST_PRIMITIVES, domain="nvidia_sass")
            )


_load_sass_schemas()
