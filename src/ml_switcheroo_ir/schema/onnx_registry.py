"""Generated ONNX Operator Registry."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


from ml_switcheroo_ir.snapshots import find_schema_file

_LOCK = threading.Lock()
ONNX_REGISTRY: dict[str, OpSchema] = {}


def load_onnx_schemas(json_path: Path | str | None = None) -> dict[str, OpSchema]:
    """Dynamically load ONNX schemas from onnx_ops.json into ONNX_REGISTRY.

    Args:
        json_path: Optional custom path to onnx_ops.json.

    Returns:
        Mapping of operator names to OpSchema objects.
    """
    with _LOCK:
        if ONNX_REGISTRY and json_path is None:
            return ONNX_REGISTRY

        target = find_schema_file("onnx_ops.json", override_path=json_path)
        if target is None or not target.exists():
            return dict(ONNX_REGISTRY) if json_path is not None else ONNX_REGISTRY

        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded: dict[str, OpSchema] = {}
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

        if json_path is None:
            ONNX_REGISTRY.clear()
            ONNX_REGISTRY.update(loaded)
            return ONNX_REGISTRY
        return loaded


load_onnx_schemas()
