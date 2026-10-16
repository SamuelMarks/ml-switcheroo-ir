"""StableHLO operator schemas for ml_switcheroo_ir."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ml_switcheroo_ir.schema.base import (
    REGISTRY_LOCK,
    CustomAttributeSchema,
    CustomOpSchema,
    OpSchema,
    load_op_schema_catalog,
)
from ml_switcheroo_ir.snapshots import find_schema_file

_logger = logging.getLogger(__name__)

STABLEHLO_REGISTRY: dict[str, OpSchema] = {}


def _load_stablehlo_schemas(json_path: Path | str | None = None) -> None:
    """Load StableHLO schemas from bundled stablehlo_ops.json or snapshot directory.

    Args:
        json_path (Optional[Union[Path, str]]): Explicit path override for stablehlo_ops.json.
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

                exhaustive_path = Path(fw_file).parent / "stablehlo_exhaustive.json"
                if exhaustive_path.exists():
                    with open(exhaustive_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for op_data in data.get("operations", []):
                        name = str(op_data.get("name", ""))
                        if not name:
                            continue
                        attrs = []
                        for a in op_data.get("attributes", []):
                            if not (isinstance(a, dict) and "name" in a):
                                continue
                            attr_name = str(a.get("name", ""))
                            req = bool(a.get("required", False))
                            if name == "convolution" and attr_name in (
                                "dimension_numbers",
                                "padding",
                            ):
                                req = True
                            attr_type = str(a.get("type", "str"))
                            if attr_name in (
                                "dimension_numbers",
                                "dot_dimension_numbers",
                            ):
                                attr_type = "dict"
                            attrs.append(
                                CustomAttributeSchema(
                                    name=attr_name,
                                    type=attr_type,
                                    required=req,
                                    default=a.get("default", None),
                                )
                            )
                        inputs = [
                            str(inp.get("name", ""))
                            for inp in op_data.get("operands", [])
                            if isinstance(inp, dict)
                        ]
                        outputs = [
                            str(out.get("name", ""))
                            for out in op_data.get("results", [])
                            if isinstance(out, dict)
                        ]
                        schema = CustomOpSchema(
                            name=name,
                            domain="stablehlo",
                            attributes=attrs,
                            inputs=inputs,
                            outputs=outputs,
                        )
                        STABLEHLO_REGISTRY[schema.name] = schema.to_op_schema()
            except (ImportError, AttributeError, ValueError, OSError) as exc:
                _logger.debug("Fallback StableHLO schema loading skipped: %s", exc)


_load_stablehlo_schemas()
