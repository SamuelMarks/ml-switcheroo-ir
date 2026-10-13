"""Tests for ONNX operator schema compliance and registry loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_ir.schema.onnx_registry import (
    ONNX_REGISTRY,
    OpAttribute,
    OpSchema,
    load_onnx_schemas,
)
from ml_switcheroo_ir.validator import Validator


@pytest.mark.parametrize("op_name", sorted(ONNX_REGISTRY.keys()))
def test_onnx_schema_compliance(op_name: str) -> None:
    """Verify that each registered ONNX operator schema is well-formed and validates nodes.

    Args:
        op_name (str): Canonical ONNX operator name in ONNX_REGISTRY.
    """
    assert op_name in ONNX_REGISTRY
    schema = ONNX_REGISTRY[op_name]
    assert schema.name == op_name
    assert schema.domain in ("ai.onnx", "ai.onnx.preview.training", "ai.onnx.ml")
    assert schema.version >= 1
    assert isinstance(schema.inputs, list)
    assert isinstance(schema.outputs, list)
    assert isinstance(schema.attributes, dict)

    node = LogicalNode(
        id=f"node_{op_name}",
        op_type=op_name,
        domain=schema.domain,
    )
    validator = Validator()
    assert validator.validate_kind(node) == []
    assert validator.validate_attribute_types(node) == []


def test_load_onnx_schemas_branches(tmp_path: Path) -> None:
    """Test load_onnx_schemas caching, missing file, and custom file loading.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    # 1. Cached load returns existing registry with canonical ONNX ops
    reg = load_onnx_schemas()
    assert len(reg) >= 205
    for core_op in ("Abs", "Add", "Conv", "Gemm", "MatMul", "Relu", "Transpose"):
        assert core_op in reg
        assert reg[core_op].domain == "ai.onnx"

    # 2. Missing file load falls back to upstream snapshot specs
    nonexistent = tmp_path / "does_not_exist.json"
    res = load_onnx_schemas(json_path=nonexistent)
    assert len(res) == len(reg)
    assert "MatMul" in res

    # 3. Custom valid JSON file
    custom_json = tmp_path / "custom_ops.json"
    custom_json.write_text(
        json.dumps(
            {
                "CustomOp": {
                    "domain": "ai.onnx.custom",
                    "version": 1,
                    "attributes": {
                        "alpha": {"type": "float", "required": False, "default": 1.0}
                    },
                    "inputs": ["x"],
                    "outputs": ["y"],
                }
            }
        ),
        encoding="utf-8",
    )
    custom_reg = load_onnx_schemas(json_path=custom_json)
    assert "CustomOp" in custom_reg
    assert custom_reg["CustomOp"].domain == "ai.onnx.custom"
    assert "alpha" in custom_reg["CustomOp"].attributes
    assert custom_reg["CustomOp"].attributes["alpha"].default == 1.0

    # 4. OpAttribute and OpSchema dataclass fields
    attr = OpAttribute(name="dim", type="int", required=True, default=0)
    assert attr.name == "dim"
    assert attr.type == "int"
    assert attr.required is True
    assert attr.default == 0

    schema = OpSchema(
        name="Op",
        domain="ai.onnx",
        version=1,
        attributes={"dim": attr},
        inputs=["in"],
        outputs=["out"],
    )
    assert schema.name == "Op"
    assert schema.domain == "ai.onnx"
    assert schema.version == 1
    assert schema.inputs == ["in"]
    assert schema.outputs == ["out"]
