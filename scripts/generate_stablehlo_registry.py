"""Generator for StableHLO operator schemas from ground-truth snapshots."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import sys

from pydantic import BaseModel, ConfigDict, Field

from ml_switcheroo_ir.types import JsonValue
from ml_switcheroo_ir.validator import DEFAULT_SNAPSHOT_DIR


class StableHloAttrSpecModel(BaseModel):
    """Pydantic model for a generated StableHLO operator attribute schema."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = "str"
    required: bool = False
    default: JsonValue = None


class StableHloOpSpecModel(BaseModel):
    """Pydantic model for a generated StableHLO operator schema."""

    model_config = ConfigDict(extra="forbid")

    name: str
    domain: str = "stablehlo"
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    attributes: list[StableHloAttrSpecModel] = Field(default_factory=list)


StableHloAttrSpecModel.model_rebuild()
StableHloOpSpecModel.model_rebuild()


_OPTIONAL_STABLEHLO_ATTRS: frozenset[str] = frozenset(
    {
        "precision_config",
        "algorithm",
        "backend_config",
        "operand_layouts",
        "result_layouts",
        "result_tilings",
        "lhs_dilation",
        "rhs_dilation",
        "window_reversal",
        "known_expanding_dimensions",
        "known_nonexpanding_dimensions",
        "channel_handle",
        "source_target_pairs",
        "layout",
        "batch_group_count",
        "feature_group_count",
    }
)

_ARRAY_STABLEHLO_ATTRS: frozenset[str] = frozenset(
    {
        "window_strides",
        "padding",
        "dimensions",
        "slice_sizes",
        "window_dimensions",
        "base_dilations",
        "window_dilations",
        "precision_config",
        "operand_layouts",
        "result_layouts",
        "result_tilings",
        "start_indices",
        "limit_indices",
        "strides",
        "permutation",
        "broadcast_dimensions",
    }
)

_DICT_STABLEHLO_ATTRS: frozenset[str] = frozenset(
    {
        "dot_dimension_numbers",
        "dimension_numbers",
        "scatter_dimension_numbers",
        "backend_config",
    }
)


def _normalize_stablehlo_op_name(raw: Mapping[str, JsonValue]) -> str:
    """Resolve canonical snake_case StableHLO operator name from a raw snapshot entry.

    Args:
        raw (Mapping[str, JsonValue]): Raw operator snapshot dictionary.

    Returns:
        str: Canonical operator name.
    """
    api_path = str(raw.get("api_path", ""))
    raw_name = str(raw.get("name", ""))
    if api_path.startswith("stablehlo."):
        return api_path.split(".", 1)[1]
    if raw_name.endswith("Op"):
        stem = raw_name[:-2]
        return "".join(
            "_" + c.lower() if c.isupper() and i > 0 else c.lower()
            for i, c in enumerate(stem)
        )
    return raw_name.lower()


def _infer_stablehlo_attr_spec(
    op_name: str, a_name: str, a_info: JsonValue
) -> dict[str, JsonValue]:
    """Infer Python type, required flag, and default value for a StableHLO attribute.

    Args:
        op_name (str): Canonical StableHLO operator name.
        a_name (str): Attribute identifier.
        a_info (JsonValue): Raw attribute metadata from snapshot.

    Returns:
        dict[str, JsonValue]: Normalized attribute schema dictionary.
    """
    a_type = str(a_info.get("type", "")) if isinstance(a_info, dict) else str(a_info)
    is_optional = (
        "OptionalAttr" in a_type
        or "DefaultValued" in a_type
        or a_name in _OPTIONAL_STABLEHLO_ATTRS
    )
    if op_name == "convolution" and a_name in (
        "padding",
        "window_strides",
        "dimension_numbers",
    ):
        is_optional = False

    default_val: JsonValue = (
        1 if a_name in ("batch_group_count", "feature_group_count") else None
    )
    py_type = "str"
    if "BoolAttr" in a_type or a_type == "boolean":
        py_type = "bool"
    elif (
        "IntegerAttr" in a_type
        or "I64" in a_type
        or "I32" in a_type
        or a_type == "integer"
    ):
        py_type = "int"
    elif (
        "FloatAttr" in a_type
        or "F32" in a_type
        or "F64" in a_type
        or a_type == "number"
    ):
        py_type = "float"
    elif (
        "Array" in a_type
        or a_type == "array"
        or (isinstance(a_info, dict) and "items" in a_info)
        or a_name in _ARRAY_STABLEHLO_ATTRS
    ):
        py_type = "List[int]" if a_name != "precision_config" else "List[str]"
    elif (
        "Dictionary" in a_type
        or "NumbersAttr" in a_type
        or "Numbers" in a_type
        or a_type == "object"
        or (isinstance(a_info, dict) and "properties" in a_info)
        or a_name in _DICT_STABLEHLO_ATTRS
    ):
        py_type = "dict"

    model = StableHloAttrSpecModel(
        name=a_name,
        type=py_type,
        required=not is_optional,
        default=default_val,
    )
    return model.model_dump()


def _extract_stablehlo_attributes(
    op_name: str, attrs_raw: JsonValue
) -> list[dict[str, JsonValue]]:
    """Extract normalized attribute schema list from a StableHLO snapshot record.

    Args:
        op_name (str): Canonical operator name.
        attrs_raw (JsonValue): Raw attributes dict or list.

    Returns:
        list[dict[str, JsonValue]]: List of attribute schema dictionaries.
    """
    attributes: list[dict[str, JsonValue]] = []
    if isinstance(attrs_raw, dict):
        for a_name, a_info in sorted(attrs_raw.items()):
            if a_name:
                attributes.append(
                    _infer_stablehlo_attr_spec(op_name, str(a_name), a_info)
                )
    elif isinstance(attrs_raw, list):
        for a in attrs_raw:
            if isinstance(a, dict):
                a_name = str(a.get("name") or "").strip()
                if a_name:
                    raw_def = a.get("default", None)
                    def_val: JsonValue = (
                        raw_def
                        if isinstance(raw_def, (str, int, float, bool, list, dict))
                        or raw_def is None
                        else str(raw_def)
                    )
                    model = StableHloAttrSpecModel(
                        name=a_name,
                        type=str(a.get("type", "str")),
                        required=bool(a.get("required", False)),
                        default=def_val,
                    )
                    attributes.append(model.model_dump())
    return attributes


def extract_stablehlo_schemas(
    snapshot_path: str | Path,
) -> list[dict[str, JsonValue]]:
    """Extract grounded StableHLO operator schemas from snapshot file.

    Args:
        snapshot_path (str | Path): Path to stablehlo JSON snapshot manifest.

    Returns:
        list[dict[str, JsonValue]]: List of grounded operator definition dictionaries.
    """
    with open(snapshot_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_ops: list[Mapping[str, JsonValue]] = []
    if isinstance(data, list):
        raw_ops = [item for item in data if isinstance(item, Mapping)]
    elif isinstance(data, dict):
        if "categories" in data and isinstance(data["categories"], dict):
            util_list = data["categories"].get("util")
            if isinstance(util_list, list):
                raw_ops = [item for item in util_list if isinstance(item, Mapping)]
        else:
            ops_val = data.get("operations")
            if isinstance(ops_val, list):
                raw_ops = [item for item in ops_val if isinstance(item, Mapping)]

    ops: list[dict[str, JsonValue]] = []
    seen: set[str] = set()

    for raw in raw_ops:
        op_name = _normalize_stablehlo_op_name(raw)
        if not op_name or op_name in seen:
            continue
        seen.add(op_name)

        raw_operands = raw.get("operands")
        inputs = (
            [
                str(opnd.get("name") or "").strip()
                for opnd in raw_operands
                if isinstance(opnd, dict) and str(opnd.get("name") or "").strip()
            ]
            if isinstance(raw_operands, list)
            else []
        )
        raw_returns = raw.get("returns")
        outputs = (
            [
                str(ret.get("name") or "").strip()
                for ret in raw_returns
                if isinstance(ret, dict) and str(ret.get("name") or "").strip()
            ]
            if isinstance(raw_returns, list)
            else []
        ) or ["result"]

        op_model = StableHloOpSpecModel(
            name=op_name,
            domain="stablehlo",
            inputs=inputs,
            outputs=outputs,
            attributes=[
                StableHloAttrSpecModel.model_validate(attr)
                for attr in _extract_stablehlo_attributes(
                    op_name, raw.get("attributes")
                )
            ],
        )
        ops.append(op_model.model_dump())

    return ops


def main(snapshot_path: str | None = None, output_path: str | None = None) -> None:
    """Generate StableHLO schemas from snapshot manifest.

    Args:
        snapshot_path (str | None): Source snapshot file path.
        output_path (str | None): Target stablehlo_ops.json path.
    """
    target_snap = (
        snapshot_path
        if snapshot_path is not None
        else (
            sys.argv[1]
            if len(sys.argv) > 1
            else os.path.join(DEFAULT_SNAPSHOT_DIR, "stablehlo_v1.9.0.json")
        )
    )
    target_out = (
        output_path
        if output_path is not None
        else (
            sys.argv[2]
            if len(sys.argv) > 2
            else "src/ml_switcheroo_ir/schema/stablehlo_ops.json"
        )
    )

    ops = extract_stablehlo_schemas(target_snap)
    with open(target_out, "w", encoding="utf-8") as f:
        json.dump({"ops": ops}, f, indent=2)

    print(f"Extracted {len(ops)} StableHLO operations into {target_out}")


if __name__ == "__main__":
    main()
