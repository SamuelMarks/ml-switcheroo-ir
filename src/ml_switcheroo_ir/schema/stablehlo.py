"""StableHLO operator schemas for ml_switcheroo_ir."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from ml_switcheroo_ir.schema.base import (
    REGISTRY_LOCK,
    AttributeType,
    OpAttribute,
    OpSchema,
    SchemaRegistry,
    load_op_schema_catalog,
)
from ml_switcheroo_ir.snapshots import find_schema_file
from ml_switcheroo_ir.types import JsonValue

_logger = logging.getLogger(__name__)


def parse_stablehlo_exhaustive_operations(
    operations: Sequence[Mapping[str, JsonValue]],
) -> dict[str, OpSchema]:
    """Parse raw operations from stablehlo_exhaustive.json into OpSchema objects.

    Args:
        operations (Sequence[Mapping[str, JsonValue]]): Raw StableHLO operation dictionaries.

    Returns:
        dict[str, OpSchema]: Mapping of StableHLO operator names to OpSchema definitions.
    """
    parsed: dict[str, OpSchema] = {}
    for op_data in operations:
        name = str(op_data.get("name", ""))
        if not name:
            continue
        attrs: list[OpAttribute] = []
        raw_attrs = op_data.get("attributes", [])
        for a in raw_attrs if isinstance(raw_attrs, Sequence) and not isinstance(raw_attrs, (str, bytes)) else []:
            if not (isinstance(a, dict) and "name" in a):
                continue
            attr_name = str(a.get("name", ""))
            req = bool(a.get("required", False))
            if name == "convolution" and attr_name in ("dimension_numbers", "padding"):
                req = True
            attr_type: AttributeType | str = str(a.get("type", "str"))
            if attr_name in ("dimension_numbers", "dot_dimension_numbers"):
                attr_type = AttributeType.DICT
            attrs.append(
                OpAttribute(
                    name=attr_name,
                    type=attr_type,
                    required=req,
                    default=a.get("default", None),
                )
            )
        raw_operands = op_data.get("operands", [])
        inputs = (
            [str(inp.get("name", "")) for inp in raw_operands if isinstance(inp, dict)]
            if isinstance(raw_operands, Sequence) and not isinstance(raw_operands, (str, bytes))
            else []
        )
        raw_results = op_data.get("results", [])
        outputs = (
            [str(out.get("name", "")) for out in raw_results if isinstance(out, dict)]
            if isinstance(raw_results, Sequence) and not isinstance(raw_results, (str, bytes))
            else []
        )
        schema = OpSchema(
            name=name,
            domain="stablehlo",
            attributes=attrs,
            inputs=inputs,
            outputs=outputs,
        )
        parsed[schema.name] = schema
    return parsed


def _load_stablehlo_schemas(json_path: Path | str | None = None) -> None:
    """Load StableHLO schemas from bundled stablehlo_ops.json or snapshot directory.

    Args:
        json_path (Path | str | None): Explicit path override for stablehlo_ops.json.
    """
    with REGISTRY_LOCK:
        loaded: dict[str, OpSchema] = {}
        try:
            loaded = load_op_schema_catalog(
                "stablehlo_ops.json",
                domain_default="stablehlo",
                json_path=json_path,
                find_file_fn=find_schema_file,
            )
        except OSError as exc:
            _logger.debug("Could not read stablehlo_ops.json: %s", exc)
            loaded = {}
        if loaded:
            STABLEHLO_REGISTRY.update(loaded)
        elif json_path is None:
            try:
                from ml_ecosystem_snapshots.frameworks import (
                    __file__ as fw_file,
                )

                pkg_exhaustive = Path(fw_file).parent / "stablehlo_exhaustive.json"
                exhaustive_path = (
                    pkg_exhaustive if pkg_exhaustive.exists() else find_schema_file("stablehlo_exhaustive.json")
                )
                if exhaustive_path is not None and exhaustive_path.exists():
                    with open(exhaustive_path, encoding="utf-8") as f:
                        data = json.load(f)
                    STABLEHLO_REGISTRY.update(parse_stablehlo_exhaustive_operations(data.get("operations", [])))
            except (ImportError, SyntaxError, AttributeError, ValueError, OSError) as exc:
                _logger.debug("Fallback StableHLO schema loading skipped: %s", exc)


STABLEHLO_REGISTRY: SchemaRegistry = SchemaRegistry("STABLEHLO_REGISTRY", loader=_load_stablehlo_schemas)

__all__ = [
    "STABLEHLO_REGISTRY",
    "parse_stablehlo_exhaustive_operations",
]
