"""Generated ONNX Operator Registry."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pydantic

from ml_switcheroo_ir.schema.base import (
    REGISTRY_LOCK,
    AttributeType,
    OpAttribute,
    OpSchema,
    SchemaRegistry,
    load_bundled_catalog_json,
)
from ml_switcheroo_ir.snapshots import find_schema_file
from ml_switcheroo_ir.types import AttributeValue, JsonValue

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
    raw_ops: Sequence[Mapping[str, JsonValue | Sequence[pydantic.BaseModel | dict[str, JsonValue]]]],
) -> dict[str, OpSchema]:
    """Convert raw upstream ONNX snapshot operator entries into OpSchema instances.

    Args:
        raw_ops (Sequence[Mapping[str, JsonValue | Sequence[pydantic.BaseModel | dict[str, JsonValue]]]]):
            Raw operator records from ``_load_onnx_ops()``.

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
            for a_raw in raw_attrs:
                a_dict = (
                    a_raw.model_dump()
                    if isinstance(a_raw, pydantic.BaseModel)
                    else (a_raw if isinstance(a_raw, dict) else {})
                )
                if "name" in a_dict:
                    a_name = str(a_dict["name"])
                    parsed_default = _parse_onnx_proto_default(a_dict.get("default"))
                    attributes[a_name] = OpAttribute(
                        name=a_name,
                        type=str(a_dict.get("type", AttributeType.ANY.value)),
                        required=bool(a_dict.get("required", False)),
                        default=parsed_default,
                    )
        elif isinstance(raw_attrs, dict):
            for a_name, a_val in raw_attrs.items():
                val_type = (
                    a_val.get("type", AttributeType.ANY.value) if isinstance(a_val, dict) else AttributeType.ANY.value
                )
                val_req = a_val.get("required", False) if isinstance(a_val, dict) else False
                val_def = _parse_onnx_proto_default(a_val.get("default")) if isinstance(a_val, dict) else None
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
            if isinstance(raw_inputs, Sequence) and not isinstance(raw_inputs, (str, bytes))
            else []
        )
        raw_outputs = op_data.get("outputs", [])
        outputs_list = (
            [str(o) for o in raw_outputs]
            if isinstance(raw_outputs, Sequence) and not isinstance(raw_outputs, (str, bytes))
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
        )
    return loaded


def _parse_onnx_json_dict(data: Mapping[str, JsonValue]) -> dict[str, OpSchema]:
    """Parse an ONNX JSON catalog dictionary into OpSchema instances.

    Args:
        data (Mapping[str, JsonValue]): Dictionary mapping operator names to raw schema specs.

    Returns:
        dict[str, OpSchema]: Parsed ONNX operator schemas.
    """
    loaded: dict[str, OpSchema] = {}
    for op_name, op_data in data.items():
        if not isinstance(op_data, Mapping):
            continue
        raw_attrs = op_data.get("attributes", {})
        attributes: dict[str, OpAttribute] = {}
        if isinstance(raw_attrs, Mapping):
            for attr_name, attr_data in raw_attrs.items():
                if isinstance(attr_data, Mapping):
                    attributes[str(attr_name)] = OpAttribute(
                        name=str(attr_name),
                        type=str(attr_data.get("type", AttributeType.ANY.value)),
                        required=bool(attr_data.get("required", False)),
                        default=_parse_onnx_proto_default(attr_data.get("default", None)),
                    )
        raw_ver = op_data.get("version", 1)
        ver = int(raw_ver) if isinstance(raw_ver, (int, str)) else 1
        raw_sv = op_data.get("schema_version")
        raw_inputs = op_data.get("inputs", [])
        inputs_list = (
            [str(x) for x in raw_inputs]
            if isinstance(raw_inputs, Sequence) and not isinstance(raw_inputs, (str, bytes))
            else []
        )
        raw_outputs = op_data.get("outputs", [])
        outputs_list = (
            [str(x) for x in raw_outputs]
            if isinstance(raw_outputs, Sequence) and not isinstance(raw_outputs, (str, bytes))
            else []
        )
        loaded[str(op_name)] = OpSchema(
            name=str(op_name),
            domain=str(op_data.get("domain", "ai.onnx")),
            version=ver,
            attributes=attributes,
            inputs=inputs_list,
            outputs=outputs_list,
            schema_version=int(raw_sv) if isinstance(raw_sv, int) else ver,
        )
    return loaded


def _initialize_onnx_registry() -> None:
    """Lazy loader populating ``ONNX_REGISTRY`` on first access under ``REGISTRY_LOCK``."""
    load_onnx_schemas()


ONNX_REGISTRY: SchemaRegistry = SchemaRegistry("ONNX_REGISTRY", loader=_initialize_onnx_registry)


def load_onnx_schemas(json_path: Path | str | None = None) -> dict[str, OpSchema]:
    """Dynamically load ONNX schemas from external snapshot, bundled catalog, or upstream spec.

    Args:
        json_path (Path | str | None): Optional custom path to onnx_ops.json.

    Returns:
        dict[str, OpSchema]: Mapping of operator names to OpSchema objects.
    """
    with REGISTRY_LOCK:
        if ONNX_REGISTRY.is_loaded and len(ONNX_REGISTRY) > 0 and json_path is None:
            return ONNX_REGISTRY.copy()

        target = find_schema_file("onnx_ops.json", override_path=json_path)
        if (target is None or not target.exists()) and json_path is not None:
            return ONNX_REGISTRY.copy()

        loaded: dict[str, OpSchema] = {}

        if target is not None and target.exists():
            with open(target, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, Mapping):
                loaded = _parse_onnx_json_dict(data)
        else:
            bundled = load_bundled_catalog_json("onnx_ops.json")
            if bundled is not None:
                loaded = _parse_onnx_json_dict(bundled)

        if not loaded and json_path is None:
            try:
                from ml_ecosystem_snapshots.frameworks.onnx_spec import _load_onnx_ops

                loaded.update(_parse_onnx_fallback_ops(_load_onnx_ops()))
            except (ImportError, SyntaxError, AttributeError, ValueError, OSError) as exc:
                _logger.debug("Fallback ONNX schema loading skipped: %s", exc)

        if json_path is None:
            ONNX_REGISTRY.clear()
            ONNX_REGISTRY.update(loaded)
            return ONNX_REGISTRY.copy()
        return loaded
