"""NVIDIA SASS architecture operator schemas and registry for ml_switcheroo_ir."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml_switcheroo_ir.schema.custom_ops import CustomAttributeSchema, CustomOpSchema
from ml_switcheroo_ir.schema.onnx_registry import OpSchema
from ml_switcheroo_ir.snapshots import find_schema_file

SASS_REGISTRY: dict[str, OpSchema] = {}
SASS_PIPELINE_LATENCIES: dict[str, int] = {}


def _load_sass_schemas(json_path: Path | str | None = None) -> None:
    """Load NVIDIA SASS instruction schemas from bundled sass_ops.json or snapshot directory.

    Args:
        json_path (Optional[Union[Path, str]]): Explicit path override for sass_ops.json.
    """
    target = find_schema_file("sass_ops.json", override_path=json_path)
    if target and target.exists():
        with open(target, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        for op_data in data.get("ops", []):
            name = str(op_data["name"])
            domain = str(op_data.get("domain", "nvidia_sass"))
            lat = int(op_data.get("execution_latency", 4))
            SASS_PIPELINE_LATENCIES[name] = lat

            attrs = [
                CustomAttributeSchema(
                    name=a["name"],
                    type=a.get("type", "str"),
                    required=a.get("required", False),
                    default=a.get("default", None),
                )
                for a in op_data.get("attributes", [])
            ]
            schema = CustomOpSchema(
                name=name,
                domain=domain,
                attributes=attrs,
                inputs=op_data.get("inputs", []),
                outputs=op_data.get("outputs", []),
            )
            SASS_REGISTRY[schema.name] = schema.to_op_schema()


_load_sass_schemas()
