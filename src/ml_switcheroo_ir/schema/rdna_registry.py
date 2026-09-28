"""AMD RDNA architecture operator schemas and registry for ml_switcheroo_ir."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml_switcheroo_ir.schema.custom_ops import CustomAttributeSchema, CustomOpSchema
from ml_switcheroo_ir.schema.onnx_registry import OpSchema
from ml_switcheroo_ir.snapshots import find_schema_file

RDNA_REGISTRY: dict[str, OpSchema] = {}
RDNA_VOPD_SLOTS: dict[str, str] = {}
RDNA_TO_VOPD_MAP: dict[str, str] = {}


def _load_rdna_schemas(json_path: Path | str | None = None) -> None:
    """Load AMD RDNA instruction schemas from bundled rdna_ops.json or snapshot directory.

    Args:
        json_path (Optional[Union[Path, str]]): Explicit path override for rdna_ops.json.
    """
    target = find_schema_file("rdna_ops.json", override_path=json_path)
    if target and target.exists():
        with open(target, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        for op_data in data.get("ops", []):
            name = str(op_data["name"])
            domain = str(op_data.get("domain", "amd_rdna"))
            vopd_slot = op_data.get("vopd_slot")
            if vopd_slot:
                RDNA_VOPD_SLOTS[name] = str(vopd_slot)
            vopd_target = op_data.get("vopd_target")
            if vopd_target:
                RDNA_TO_VOPD_MAP[name] = str(vopd_target)

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
            RDNA_REGISTRY[schema.name] = schema.to_op_schema()
    elif json_path is None:
        from ml_switcheroo_ir.schema.ghost import (
            RDNA3_VOPD_OPERATORS as GHOST_VOPD_OPS,
        )
        from ml_switcheroo_ir.schema.ghost import (
            RDNA_INSTRUCTION_PRIMITIVES as GHOST_PRIMITIVES,
        )
        from ml_switcheroo_ir.schema.ghost import (
            RDNA_TO_VOPD_MAP as GHOST_VOPD_MAP,
        )

        for op_name, op_info in GHOST_VOPD_OPS.items():
            slots = op_info.get("slots", [])
            slot_str = (
                "BOTH" if set(slots) == {"X", "Y"} else (slots[0] if slots else "BOTH")
            )
            RDNA_VOPD_SLOTS[op_name] = slot_str

        RDNA_TO_VOPD_MAP.update(GHOST_VOPD_MAP)

        for op_name, op_info in GHOST_PRIMITIVES.items():
            operands = op_info.get("operands", [])
            inputs = [op for op in operands if op != "dst"]
            outputs = ["dst"] if "dst" in operands else []
            schema = CustomOpSchema(
                name=op_name,
                domain="amd_rdna",
                attributes=[],
                inputs=inputs,
                outputs=outputs,
            )
            RDNA_REGISTRY[op_name] = schema.to_op_schema()


_load_rdna_schemas()
