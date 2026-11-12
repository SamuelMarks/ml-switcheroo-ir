"""AMD RDNA architecture operator schemas and registry for ml_switcheroo_ir."""

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

_DEFAULT_RDNA_TO_VOPD_MAP: dict[str, str] = {
    "V_FMAC_F32": "V_DUAL_FMAC_F32",
    "V_FMA_F32": "V_DUAL_FMAC_F32",
    "V_MUL_F32": "V_DUAL_MUL_F32",
    "V_ADD_F32": "V_DUAL_ADD_F32",
    "V_SUB_F32": "V_DUAL_SUB_F32",
    "V_MOV_B32": "V_DUAL_MOV_B32",
    "V_CNDMASK_B32": "V_DUAL_CNDMASK_B32",
    "V_DOT2_F32_F16": "V_DUAL_DOT2ACC_F32_F16",
}


def _load_rdna_schemas(json_path: Path | str | None = None) -> None:
    """Load AMD RDNA instruction schemas from bundled rdna_ops.json or snapshot directory.

    Args:
        json_path (Path | str | None): Explicit path override for rdna_ops.json.
    """

    def _record_rdna_metadata(op_data: Mapping[str, JsonValue]) -> None:
        """Record VOPD slot and target mappings from an RDNA schema entry.

        Args:
            op_data (Mapping[str, JsonValue]): Raw operator definition dictionary.
        """
        name = str(op_data["name"])
        vopd_slot = op_data.get("vopd_slot")
        if vopd_slot:
            RDNA_VOPD_SLOTS[name] = str(vopd_slot)
        vopd_target = op_data.get("vopd_target") or op_data.get("vopd_equiv")
        if vopd_target:
            RDNA_TO_VOPD_MAP[name] = str(vopd_target)

    with REGISTRY_LOCK:
        RDNA_VOPD_SLOTS.mark_loaded()
        RDNA_TO_VOPD_MAP.mark_loaded()
        loaded = load_op_schema_catalog(
            "rdna_ops.json",
            domain_default="amd_rdna",
            json_path=json_path,
            find_file_fn=find_schema_file,
            on_entry=_record_rdna_metadata,
        )
        if loaded:
            RDNA_REGISTRY.update(loaded)
        elif json_path is None:
            from ml_switcheroo_ir.schema.ghost import (
                RDNA3_VOPD_OPERATORS,
                RDNA_INSTRUCTION_PRIMITIVES,
            )

            for op_name, op_info in RDNA3_VOPD_OPERATORS.items():
                slots = op_info.slots
                slot_str = "BOTH" if set(slots) == {"X", "Y"} else (slots[0] if slots else "BOTH")
                RDNA_VOPD_SLOTS[op_name] = slot_str

            RDNA_TO_VOPD_MAP.update(_DEFAULT_RDNA_TO_VOPD_MAP)
            RDNA_REGISTRY.update(build_isa_primitive_schemas(RDNA_INSTRUCTION_PRIMITIVES, domain="amd_rdna"))


RDNA_REGISTRY: SchemaRegistry = SchemaRegistry("RDNA_REGISTRY", loader=_load_rdna_schemas)
RDNA_VOPD_SLOTS: LazyMetadataMap[str] = LazyMetadataMap("RDNA_VOPD_SLOTS", loader=RDNA_REGISTRY._ensure_loaded)
RDNA_TO_VOPD_MAP: LazyMetadataMap[str] = LazyMetadataMap("RDNA_TO_VOPD_MAP", loader=RDNA_REGISTRY._ensure_loaded)

__all__ = [
    "RDNA_REGISTRY",
    "RDNA_TO_VOPD_MAP",
    "RDNA_VOPD_SLOTS",
    "OpSchema",
]
