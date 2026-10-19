"""Generated ONNX Operator Registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import logging
from pathlib import Path
import re

from ml_switcheroo_ir.schema.base import REGISTRY_LOCK, OpAttribute, OpSchema
from ml_switcheroo_ir.snapshots import find_schema_file
from ml_switcheroo_ir.types import AttributeValue

_logger = logging.getLogger(__name__)

__all__ = [
    "ONNX_REGISTRY",
    "OpAttribute",
    "OpSchema",
    "_parse_onnx_fallback_ops",
    "_parse_onnx_proto_default",
    "load_onnx_schemas",
]


def _parse_onnx_proto_default(raw: AttributeValue) -> AttributeValue:
    """Parse protobuf text format attribute default value into a native Python scalar or list.

    Args:
        raw (AttributeValue): Raw default representation from snapshot.

    Returns:
        AttributeValue: Parsed Python int, float, string, or original value.
    """
    if not isinstance(raw, str):
        return raw
    m_i = re.search(r"\bi:\s*(-?\d+)", raw)
    if m_i:
        return int(m_i.group(1))
    m_f = re.search(r"\bf:\s*(-?[\d\.]+)", raw)
    if m_f:
        return float(m_f.group(1))
    m_s = re.search(r'\bs:\s*"([^"]*)"', raw)
    if m_s:
        return m_s.group(1)
    return raw


def _parse_onnx_fallback_ops(
    raw_ops: Sequence[Mapping[str, object]],
) -> dict[str, OpSchema]:
    """Convert raw upstream ONNX snapshot operator entries into OpSchema instances.

    Args:
        raw_ops (Sequence[Mapping[str, object]]): Raw operator records from _load_onnx_ops().

    Returns:
        dict[str, OpSchema]: Mapping of operator names to OpSchema definitions.
    """
    loaded: dict[str, OpSchema] = {}
    for op_data in raw_ops:
        name = str(op_data.get("name", ""))
        if not name:
            continue
        attributes: dict[str, OpAttribute] = {}
        raw_attrs = op_data.get("attributes", [])
        if isinstance(raw_attrs, list):
            for a in raw_attrs:
                if isinstance(a, dict) and "name" in a:
                    a_name = str(a["name"])
                    parsed_default = _parse_onnx_proto_default(a.get("default"))
                    attributes[a_name] = OpAttribute(
                        name=a_name,
                        type=str(a.get("type", "Any")),
                        required=bool(a.get("required", False)),
                        default=parsed_default,
                    )
        elif isinstance(raw_attrs, dict):
            for a_name, a_val in raw_attrs.items():
                val_type = (
                    a_val.get("type", "Any") if isinstance(a_val, dict) else "Any"
                )
                val_req = (
                    a_val.get("required", False) if isinstance(a_val, dict) else False
                )
                val_def = (
                    _parse_onnx_proto_default(a_val.get("default"))
                    if isinstance(a_val, dict)
                    else None
                )
                attributes[str(a_name)] = OpAttribute(
                    name=str(a_name),
                    type=str(val_type),
                    required=bool(val_req),
                    default=val_def,
                )
        raw_ver = op_data.get("since_version") or op_data.get("version") or 1
        ver = int(raw_ver) if isinstance(raw_ver, (int, str)) else 1
        raw_sv = op_data.get("schema_version")
        raw_inputs = op_data.get("inputs", [])
        inputs_list = (
            [str(i) for i in raw_inputs]
            if isinstance(raw_inputs, Sequence)
            and not isinstance(raw_inputs, (str, bytes))
            else []
        )
        raw_outputs = op_data.get("outputs", [])
        outputs_list = (
            [str(o) for o in raw_outputs]
            if isinstance(raw_outputs, Sequence)
            and not isinstance(raw_outputs, (str, bytes))
            else []
        )
        loaded[name] = OpSchema(
            name=name,
            domain=str(op_data.get("domain") or "ai.onnx"),
            version=ver,
            attributes=attributes,
            inputs=inputs_list,
            outputs=outputs_list,
            schema_version=int(raw_sv) if isinstance(raw_sv, int) else ver,
            deprecated=bool(op_data.get("deprecated", False)),
        )
    return loaded


ONNX_REGISTRY: dict[str, OpSchema] = {}


def load_onnx_schemas(json_path: Path | str | None = None) -> dict[str, OpSchema]:
    """Dynamically load ONNX schemas from onnx_ops.json or upstream snapshot into ONNX_REGISTRY.

    Args:
        json_path (Optional[Union[Path, str]]): Optional custom path to onnx_ops.json.

    Returns:
        dict[str, OpSchema]: Mapping of operator names to OpSchema objects.
    """
    with REGISTRY_LOCK:
        if ONNX_REGISTRY and json_path is None:
            return ONNX_REGISTRY

        target = find_schema_file("onnx_ops.json", override_path=json_path)
        if (target is None or not target.exists()) and json_path is not None:
            return dict(ONNX_REGISTRY)

        loaded: dict[str, OpSchema] = {}

        if target is not None and target.exists():
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)

            for op_name, op_data in data.items():
                attributes = {
                    attr_name: OpAttribute(
                        name=attr_name,
                        type=attr_data.get("type", "Any"),
                        required=attr_data.get("required", False),
                        default=attr_data.get("default", None),
                    )
                    for attr_name, attr_data in op_data.get("attributes", {}).items()
                }
                ver = int(op_data.get("version", 1))
                raw_sv = op_data.get("schema_version")
                loaded[op_name] = OpSchema(
                    name=op_name,
                    domain=op_data.get("domain", "ai.onnx"),
                    version=ver,
                    attributes=attributes,
                    inputs=op_data.get("inputs", []),
                    outputs=op_data.get("outputs", []),
                    schema_version=int(raw_sv) if isinstance(raw_sv, int) else ver,
                    deprecated=bool(op_data.get("deprecated", False)),
                )
        else:
            try:
                from ml_ecosystem_snapshots.frameworks.onnx_spec import _load_onnx_ops

                loaded.update(_parse_onnx_fallback_ops(_load_onnx_ops()))
            except (ImportError, AttributeError, ValueError, OSError) as exc:
                _logger.debug("Fallback ONNX schema loading skipped: %s", exc)

        if json_path is None:
            ONNX_REGISTRY.clear()
            ONNX_REGISTRY.update(loaded)
            return ONNX_REGISTRY
        return loaded


load_onnx_schemas()
