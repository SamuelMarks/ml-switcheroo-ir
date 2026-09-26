"""MLIR core dialect operator schemas and registry for ml_switcheroo_ir."""

from __future__ import annotations

import json
from pathlib import Path

from ml_switcheroo_ir.schema.custom_ops import CustomAttributeSchema, CustomOpSchema
from ml_switcheroo_ir.schema.onnx_registry import OpSchema
from ml_switcheroo_ir.snapshots import find_schema_file

MLIR_REGISTRY: dict[str, OpSchema] = {}


def _load_mlir_schemas(json_path: Path | str | None = None) -> None:
    """Load MLIR schemas from bundled mlir_ops.json or snapshot directory.

    Args:
        json_path (Optional[Union[Path, str]]): Explicit path override for mlir_ops.json.
    """
    target = find_schema_file("mlir_ops.json", override_path=json_path)
    if target and target.exists():
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        for op_data in data.get("ops", []):
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
                name=op_data["name"],
                domain=op_data.get("domain", "mlir"),
                attributes=attrs,
                inputs=op_data.get("inputs", []),
                outputs=op_data.get("outputs", []),
            )
            MLIR_REGISTRY[schema.name] = schema.to_op_schema()


_load_mlir_schemas()
