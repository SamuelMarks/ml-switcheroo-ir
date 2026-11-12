"""Unified MLIR core dialect operator schemas and registry for ml_switcheroo_ir."""

from __future__ import annotations

from collections.abc import Mapping
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

_CANONICAL_MLIR_OPS: list[OpSchema] = [
    # arith dialect
    OpSchema(
        name="arith.constant",
        domain="mlir.arith",
        attributes=[OpAttribute(name="value", type=AttributeType.ANY, required=True)],
        inputs=[],
        outputs=["result"],
    ),
    OpSchema(name="arith.addi", domain="mlir.arith", inputs=["lhs", "rhs"], outputs=["result"]),
    OpSchema(name="arith.subi", domain="mlir.arith", inputs=["lhs", "rhs"], outputs=["result"]),
    OpSchema(name="arith.muli", domain="mlir.arith", inputs=["lhs", "rhs"], outputs=["result"]),
    OpSchema(name="arith.addf", domain="mlir.arith", inputs=["lhs", "rhs"], outputs=["result"]),
    OpSchema(name="arith.subf", domain="mlir.arith", inputs=["lhs", "rhs"], outputs=["result"]),
    OpSchema(name="arith.mulf", domain="mlir.arith", inputs=["lhs", "rhs"], outputs=["result"]),
    OpSchema(name="arith.divf", domain="mlir.arith", inputs=["lhs", "rhs"], outputs=["result"]),
    OpSchema(
        name="arith.cmpf",
        domain="mlir.arith",
        attributes=[OpAttribute(name="predicate", type=AttributeType.STR, required=True)],
        inputs=["lhs", "rhs"],
        outputs=["result"],
    ),
    OpSchema(
        name="arith.cmpi",
        domain="mlir.arith",
        attributes=[OpAttribute(name="predicate", type=AttributeType.STR, required=True)],
        inputs=["lhs", "rhs"],
        outputs=["result"],
    ),
    # math dialect
    OpSchema(name="math.exp", domain="mlir.math", inputs=["operand"], outputs=["result"]),
    OpSchema(name="math.exp2", domain="mlir.math", inputs=["operand"], outputs=["result"]),
    OpSchema(name="math.log", domain="mlir.math", inputs=["operand"], outputs=["result"]),
    OpSchema(name="math.sqrt", domain="mlir.math", inputs=["operand"], outputs=["result"]),
    OpSchema(name="math.rsqrt", domain="mlir.math", inputs=["operand"], outputs=["result"]),
    OpSchema(name="math.sin", domain="mlir.math", inputs=["operand"], outputs=["result"]),
    OpSchema(name="math.cos", domain="mlir.math", inputs=["operand"], outputs=["result"]),
    OpSchema(name="math.tanh", domain="mlir.math", inputs=["operand"], outputs=["result"]),
    OpSchema(name="math.erf", domain="mlir.math", inputs=["operand"], outputs=["result"]),
    OpSchema(name="math.absf", domain="mlir.math", inputs=["operand"], outputs=["result"]),
    # tensor dialect
    OpSchema(
        name="tensor.empty",
        domain="mlir.tensor",
        attributes=[
            OpAttribute(name="staticSizes", type=AttributeType.LIST_INT, required=False),
            OpAttribute(name="dynamicSizes", type=AttributeType.LIST_INT, required=False),
        ],
        inputs=[],
        outputs=["result"],
    ),
    OpSchema(name="tensor.extract", domain="mlir.tensor", inputs=["tensor", "indices"], outputs=["result"]),
    OpSchema(
        name="tensor.insert",
        domain="mlir.tensor",
        inputs=["scalar", "dest", "indices"],
        outputs=["result"],
    ),
    OpSchema(name="tensor.cast", domain="mlir.tensor", inputs=["source"], outputs=["dest"]),
    OpSchema(name="tensor.reshape", domain="mlir.tensor", inputs=["source", "shape"], outputs=["result"]),
    OpSchema(name="tensor.dim", domain="mlir.tensor", inputs=["source", "index"], outputs=["result"]),
    OpSchema(name="tensor.expand_shape", domain="mlir.tensor", inputs=["src"], outputs=["result"]),
    OpSchema(name="tensor.collapse_shape", domain="mlir.tensor", inputs=["src"], outputs=["result"]),
    # linalg dialect
    OpSchema(name="linalg.matmul", domain="mlir.linalg", inputs=["lhs", "rhs", "outs"], outputs=["result"]),
    OpSchema(name="linalg.conv2d", domain="mlir.linalg", inputs=["input", "filter", "outs"], outputs=["result"]),
    OpSchema(
        name="linalg.batch_matmul",
        domain="mlir.linalg",
        inputs=["lhs", "rhs", "outs"],
        outputs=["result"],
    ),
    OpSchema(name="linalg.dot", domain="mlir.linalg", inputs=["lhs", "rhs", "outs"], outputs=["result"]),
    OpSchema(
        name="linalg.generic",
        domain="mlir.linalg",
        attributes=[
            OpAttribute(name="indexing_maps", type=AttributeType.LIST, required=True),
            OpAttribute(name="iterator_types", type=AttributeType.LIST_STR, required=True),
        ],
        inputs=["inputs", "outputs"],
        outputs=["result"],
    ),
    OpSchema(name="linalg.fill", domain="mlir.linalg", inputs=["value", "outs"], outputs=["result"]),
    # scf dialect
    OpSchema(name="scf.for", domain="mlir.scf", inputs=["lower", "upper", "step"], outputs=["results"]),
    OpSchema(name="scf.if", domain="mlir.scf", inputs=["condition"], outputs=["results"]),
    OpSchema(name="scf.yield", domain="mlir.scf", inputs=["operands"], outputs=[]),
    OpSchema(name="scf.while", domain="mlir.scf", inputs=["inits"], outputs=["results"]),
    OpSchema(name="scf.condition", domain="mlir.scf", inputs=["condition"], outputs=[]),
    # func dialect
    OpSchema(
        name="func.func",
        domain="mlir.func",
        attributes=[
            OpAttribute(name="sym_name", type=AttributeType.STR, required=True),
            OpAttribute(name="function_type", type=AttributeType.STR, required=False, default="() -> ()"),
        ],
        inputs=[],
        outputs=[],
    ),
    OpSchema(name="func.call", domain="mlir.func", inputs=["callee_arg"], outputs=["results"]),
    OpSchema(name="func.return", domain="mlir.func", inputs=["operands"], outputs=[]),
]

CORE_MLIR_DIALECT_NAMES: frozenset[str] = frozenset({"arith", "math", "tensor", "linalg", "scf", "func"})


def _load_mlir_schemas(json_path: Path | str | None = None) -> None:
    """Load unified MLIR schemas from bundled mlir_ops.json or snapshot directory.

    Args:
        json_path (Path | str | None): Explicit path override for mlir_ops.json.
    """
    with REGISTRY_LOCK:
        loaded = load_op_schema_catalog(
            "mlir_ops.json",
            domain_default="mlir",
            json_path=json_path,
            find_file_fn=find_schema_file,
        )
        if loaded:
            MLIR_REGISTRY.update(loaded)
        elif json_path is None:
            for schema in _CANONICAL_MLIR_OPS:
                MLIR_REGISTRY[schema.name] = schema


MLIR_REGISTRY: SchemaRegistry = SchemaRegistry("MLIR_REGISTRY", loader=_load_mlir_schemas)


def resolve_mlir_op_schema(
    domain: str,
    op_type: str,
    registry: Mapping[str, OpSchema] | None = None,
) -> OpSchema | None:
    """Resolve an MLIR op across both ``mlir.<dialect>`` and short ``<dialect>`` addressing conventions.

    Args:
        domain (str): Node domain string (e.g. ``'mlir.arith'``, ``'mlir'``, or ``'arith'``).
        op_type (str): Operation mnemonic (e.g. ``'arith.addf'`` or ``'addf'``).
        registry (Mapping[str, OpSchema] | None): Optional custom MLIR registry mapping.

    Returns:
        OpSchema | None: Matching canonical OpSchema if registered for ``domain``, else None.
    """
    active = MLIR_REGISTRY if registry is None else registry
    if domain == "mlir" or domain.startswith("mlir."):
        direct = active.get(op_type)
        if direct is not None:
            if domain == "mlir" or direct.domain in (domain, "mlir"):
                return direct
        if domain.startswith("mlir.") and "." not in op_type:
            dialect = domain.split(".", 1)[1]
            prefixed = active.get(f"{dialect}.{op_type}")
            if prefixed is not None:
                return prefixed
        return None
    if domain in CORE_MLIR_DIALECT_NAMES:
        candidate_key = op_type if op_type.startswith(f"{domain}.") else f"{domain}.{op_type}"
        found = active.get(candidate_key)
        if found is not None and found.domain in (f"mlir.{domain}", domain, "mlir"):
            return found
    return None


def build_core_mlir_dialects(
    registry: Mapping[str, OpSchema] | None = None,
) -> dict[str, dict[str, int]]:
    """Derive a ``{dialect: {short_op_name: min_operands}}`` table from the unified ``MLIR_REGISTRY``.

    Args:
        registry (Mapping[str, OpSchema] | None): Optional custom MLIR registry mapping.

    Returns:
        dict[str, dict[str, int]]: Mapping of dialect names to short op names and minimum operand counts.
    """
    active = MLIR_REGISTRY if registry is None else registry
    dialects: dict[str, dict[str, int]] = {d: {} for d in sorted(CORE_MLIR_DIALECT_NAMES)}
    for full_name, schema in active.items():
        if "." not in full_name:
            continue
        dialect, short_op = full_name.split(".", 1)
        if dialect in dialects:
            dialects[dialect][short_op] = len(schema.inputs)
    return dialects


__all__ = [
    "CORE_MLIR_DIALECT_NAMES",
    "MLIR_REGISTRY",
    "build_core_mlir_dialects",
    "resolve_mlir_op_schema",
]
