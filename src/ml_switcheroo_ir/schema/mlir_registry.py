"""MLIR core dialect operator schemas and registry for ml_switcheroo_ir."""

from __future__ import annotations

import json
from pathlib import Path

from ml_switcheroo_ir.schema.custom_ops import CustomAttributeSchema, CustomOpSchema
from ml_switcheroo_ir.schema.onnx_registry import OpSchema

MLIR_REGISTRY: dict[str, OpSchema] = {}


def _load_mlir_schemas() -> None:
    """Load MLIR schemas from bundled mlir_ops.json."""
    json_path = Path(__file__).parent / "mlir_ops.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
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
