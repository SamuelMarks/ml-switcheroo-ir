"""Generated ONNX Operator Registry."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_switcheroo_ir.snapshots import find_schema_file


@dataclass
class OpAttribute:
    """Represents a single operator attribute schema.

    Attributes:
        name: Name of the attribute.
        type: Type descriptor string of the attribute.
        required: Whether the attribute must be provided.
        default: Default value if optional.
    """

    name: str
    type: str
    required: bool
    default: Any


@dataclass
class OpSchema:
    """Represents a single operator schema.

    Attributes:
        name: Operator identifier name.
        domain: Domain namespace of the operator.
        version: Operator schema version integer.
        attributes: Mapping of attribute names to OpAttribute instances.
        inputs: List of formal input operand names.
        outputs: List of formal output operand names.
    """

    name: str
    domain: str
    version: int
    attributes: dict[str, OpAttribute]
    inputs: list[str]
    outputs: list[str]


def _parse_onnx_proto_default(raw: Any) -> Any:
    """Parse protobuf text format attribute default value into a native Python scalar or list.

    Args:
        raw (Any): Raw default representation from snapshot.

    Returns:
        Any: Parsed Python int, float, string, or original value.
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


_LOCK = threading.Lock()
ONNX_REGISTRY: dict[str, OpSchema] = {}


def load_onnx_schemas(json_path: Path | str | None = None) -> dict[str, OpSchema]:
    """Dynamically load ONNX schemas from onnx_ops.json or upstream snapshot into ONNX_REGISTRY.

    Args:
        json_path: Optional custom path to onnx_ops.json.

    Returns:
        Mapping of operator names to OpSchema objects.
    """
    with _LOCK:
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
                loaded[op_name] = OpSchema(
                    name=op_name,
                    domain=op_data.get("domain", "ai.onnx"),
                    version=op_data.get("version", 1),
                    attributes=attributes,
                    inputs=op_data.get("inputs", []),
                    outputs=op_data.get("outputs", []),
                )
        else:
            try:
                from ml_ecosystem_snapshots.frameworks.onnx_spec import _load_onnx_ops

                for op_data in _load_onnx_ops():
                    name = str(op_data.get("name", ""))
                    if not name:
                        continue
                    attributes = {}
                    raw_attrs = op_data.get("attributes", [])
                    if isinstance(raw_attrs, list):
                        for a in raw_attrs:
                            if isinstance(a, dict) and "name" in a:
                                a_name = str(a["name"])
                                parsed_default = _parse_onnx_proto_default(
                                    a.get("default")
                                )
                                attributes[a_name] = OpAttribute(
                                    name=a_name,
                                    type=str(a.get("type", "Any")),
                                    required=bool(a.get("required", False)),
                                    default=parsed_default,
                                )
                    elif isinstance(raw_attrs, dict):
                        for a_name, a_val in raw_attrs.items():
                            val_type = (
                                a_val.get("type", "Any")
                                if isinstance(a_val, dict)
                                else "Any"
                            )
                            val_req = (
                                a_val.get("required", False)
                                if isinstance(a_val, dict)
                                else False
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
                    loaded[name] = OpSchema(
                        name=name,
                        domain=str(op_data.get("domain") or "ai.onnx"),
                        version=int(
                            op_data.get("since_version") or op_data.get("version") or 1
                        ),
                        attributes=attributes,
                        inputs=[str(i) for i in op_data.get("inputs", [])],
                        outputs=[str(o) for o in op_data.get("outputs", [])],
                    )
            except (ImportError, AttributeError, ValueError, OSError):
                pass

        if json_path is None:
            ONNX_REGISTRY.clear()
            ONNX_REGISTRY.update(loaded)
            return ONNX_REGISTRY
        return loaded


load_onnx_schemas()
