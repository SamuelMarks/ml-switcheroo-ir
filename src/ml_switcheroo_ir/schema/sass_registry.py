"""NVIDIA SASS architecture operator schemas and registry for ml_switcheroo_ir."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ml_switcheroo_ir.schema.base import (
    REGISTRY_LOCK,
    LazyMetadataMap,
    OpSchema,
    SchemaRegistry,
    build_isa_primitive_schemas,
    load_op_schema_catalog,
)
from ml_switcheroo_ir.snapshots import find_schema_file
from ml_switcheroo_ir.types import JsonValue

_DEFAULT_SASS_PIPELINE_LATENCIES: dict[str, int] = {
    "FFMA": 4,
    "FADD": 4,
    "FMUL": 4,
    "HMMA": 8,
    "LDG": 200,
    "STS": 16,
}


def _load_sass_schemas(json_path: Path | str | None = None) -> None:
    """Load NVIDIA SASS instruction schemas from bundled sass_ops.json or snapshot directory.

    Args:
        json_path (Path | str | None): Explicit path override for sass_ops.json.
    """

    def _record_sass_latency(op_data: Mapping[str, JsonValue]) -> None:
        """Record execution latency from a SASS schema entry.

        Args:
            op_data (Mapping[str, JsonValue]): Raw operator definition dictionary.
        """
        raw_lat = op_data.get("execution_latency")
        lat_val = raw_lat if raw_lat is not None else op_data.get("latency", 4)
        SASS_PIPELINE_LATENCIES[str(op_data["name"])] = int(lat_val) if isinstance(lat_val, (int, str)) else 4

    with REGISTRY_LOCK:
        SASS_PIPELINE_LATENCIES.mark_loaded()
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
            from ml_switcheroo_ir.schema.ghost import SASS_INSTRUCTION_PRIMITIVES

            SASS_PIPELINE_LATENCIES.update(_DEFAULT_SASS_PIPELINE_LATENCIES)
            SASS_REGISTRY.update(build_isa_primitive_schemas(SASS_INSTRUCTION_PRIMITIVES, domain="nvidia_sass"))


SASS_REGISTRY: SchemaRegistry = SchemaRegistry("SASS_REGISTRY", loader=_load_sass_schemas)
SASS_PIPELINE_LATENCIES: LazyMetadataMap[int] = LazyMetadataMap(
    "SASS_PIPELINE_LATENCIES", loader=SASS_REGISTRY._ensure_loaded
)

__all__ = [
    "SASS_PIPELINE_LATENCIES",
    "SASS_REGISTRY",
    "OpSchema",
]
