"""MLIR core dialect operator schemas and registry for ml_switcheroo_ir."""

from __future__ import annotations

import json
from pathlib import Path

from ml_switcheroo_ir.schema.custom_ops import CustomAttributeSchema, CustomOpSchema
from ml_switcheroo_ir.schema.onnx_registry import OpSchema
from ml_switcheroo_ir.snapshots import find_schema_file

MLIR_REGISTRY: dict[str, OpSchema] = {}

_CANONICAL_MLIR_OPS: list[CustomOpSchema] = [
    CustomOpSchema(
        name="arith.constant",
        domain="mlir.arith",
        attributes=[CustomAttributeSchema(name="value", type="Any", required=True)],
        inputs=[],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="arith.addi",
        domain="mlir.arith",
        attributes=[],
        inputs=["lhs", "rhs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="arith.subi",
        domain="mlir.arith",
        attributes=[],
        inputs=["lhs", "rhs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="arith.muli",
        domain="mlir.arith",
        attributes=[],
        inputs=["lhs", "rhs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="arith.addf",
        domain="mlir.arith",
        attributes=[],
        inputs=["lhs", "rhs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="arith.subf",
        domain="mlir.arith",
        attributes=[],
        inputs=["lhs", "rhs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="arith.mulf",
        domain="mlir.arith",
        attributes=[],
        inputs=["lhs", "rhs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="arith.divf",
        domain="mlir.arith",
        attributes=[],
        inputs=["lhs", "rhs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="arith.cmpf",
        domain="mlir.arith",
        attributes=[CustomAttributeSchema(name="predicate", type="str", required=True)],
        inputs=["lhs", "rhs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="arith.cmpi",
        domain="mlir.arith",
        attributes=[CustomAttributeSchema(name="predicate", type="str", required=True)],
        inputs=["lhs", "rhs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="math.exp",
        domain="mlir.math",
        attributes=[],
        inputs=["operand"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="math.log",
        domain="mlir.math",
        attributes=[],
        inputs=["operand"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="math.sqrt",
        domain="mlir.math",
        attributes=[],
        inputs=["operand"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="math.sin",
        domain="mlir.math",
        attributes=[],
        inputs=["operand"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="math.cos",
        domain="mlir.math",
        attributes=[],
        inputs=["operand"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="math.tanh",
        domain="mlir.math",
        attributes=[],
        inputs=["operand"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="tensor.empty",
        domain="mlir.tensor",
        attributes=[
            CustomAttributeSchema(name="staticSizes", type="List[int]", required=False),
            CustomAttributeSchema(
                name="dynamicSizes", type="List[int]", required=False
            ),
        ],
        inputs=[],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="tensor.extract",
        domain="mlir.tensor",
        attributes=[],
        inputs=["tensor", "indices"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="tensor.insert",
        domain="mlir.tensor",
        attributes=[],
        inputs=["scalar", "dest", "indices"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="tensor.cast",
        domain="mlir.tensor",
        attributes=[],
        inputs=["source"],
        outputs=["dest"],
    ),
    CustomOpSchema(
        name="tensor.reshape",
        domain="mlir.tensor",
        attributes=[],
        inputs=["source", "shape"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="linalg.matmul",
        domain="mlir.linalg",
        attributes=[],
        inputs=["lhs", "rhs", "outs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="linalg.generic",
        domain="mlir.linalg",
        attributes=[
            CustomAttributeSchema(
                name="indexing_maps", type="List[Any]", required=True
            ),
            CustomAttributeSchema(
                name="iterator_types", type="List[str]", required=True
            ),
        ],
        inputs=["inputs", "outputs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="linalg.fill",
        domain="mlir.linalg",
        attributes=[],
        inputs=["value", "outs"],
        outputs=["result"],
    ),
    CustomOpSchema(
        name="scf.for",
        domain="mlir.scf",
        attributes=[],
        inputs=["lower", "upper", "step", "init_args"],
        outputs=["results"],
    ),
    CustomOpSchema(
        name="scf.if",
        domain="mlir.scf",
        attributes=[],
        inputs=["condition"],
        outputs=["results"],
    ),
    CustomOpSchema(
        name="scf.yield",
        domain="mlir.scf",
        attributes=[],
        inputs=["operands"],
        outputs=[],
    ),
    CustomOpSchema(
        name="scf.while",
        domain="mlir.scf",
        attributes=[],
        inputs=["inits"],
        outputs=["results"],
    ),
    CustomOpSchema(
        name="func.func",
        domain="mlir.func",
        attributes=[
            CustomAttributeSchema(name="sym_name", type="str", required=True),
            CustomAttributeSchema(name="function_type", type="str", required=True),
        ],
        inputs=[],
        outputs=[],
    ),
    CustomOpSchema(
        name="func.return",
        domain="mlir.func",
        attributes=[],
        inputs=["operands"],
        outputs=[],
    ),
]


def _load_mlir_schemas(json_path: Path | str | None = None) -> None:
    """Load MLIR schemas from bundled mlir_ops.json or snapshot directory.

    Args:
        json_path (Optional[Union[Path, str]]): Explicit path override for mlir_ops.json.
    """
    target = find_schema_file("mlir_ops.json", override_path=json_path)
    if target and target.exists():
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        for op_data in data.get("ops", []):
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
                name=op_data["name"],
                domain=op_data.get("domain", "mlir"),
                attributes=attrs,
                inputs=op_data.get("inputs", []),
                outputs=op_data.get("outputs", []),
            )
            MLIR_REGISTRY[schema.name] = schema.to_op_schema()
    elif json_path is None:
        for s in _CANONICAL_MLIR_OPS:
            MLIR_REGISTRY[s.name] = s.to_op_schema()


_load_mlir_schemas()
