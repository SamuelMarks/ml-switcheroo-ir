"""NVIDIA SASS architecture operator schemas and registry for ml_switcheroo_ir."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ml_switcheroo_ir.schema.base import (
    CustomOpSchema,
    OpSchema,
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

    def _record_sass_latency(op_data: Mapping[str, Any]) -> None:
        """Record execution latency from a SASS schema entry.

        Args:
            op_data (Mapping[str, Any]): Raw operator definition dictionary.
        """
        SASS_PIPELINE_LATENCIES[str(op_data["name"])] = int(
            op_data.get("execution_latency", 4)
        )

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

        for op_name, op_info in GHOST_PRIMITIVES.items():
            operands = op_info.get("operands", [])
            inputs = [op for op in operands if op != "dst"]
            outputs = ["dst"] if "dst" in operands else []
            schema = CustomOpSchema(
                name=op_name,
                domain="nvidia_sass",
                attributes=[],
                inputs=inputs,
                outputs=outputs,
            )
            SASS_REGISTRY[op_name] = schema.to_op_schema()


_load_sass_schemas()
