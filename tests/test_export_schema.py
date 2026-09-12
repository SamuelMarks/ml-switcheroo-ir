"""Tests for JSON Schema and TypeScript export utilities and CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema.validators import Draft202012Validator  # type: ignore[import-untyped]

from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalNode,
    export_schemas,
    generate_typescript_definitions,
    get_json_schema,
)
from ml_switcheroo_ir.cli import main
from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope


def test_get_json_schema_draft_2020_12_conformance() -> None:
    """Verify that emitted JSON schemas conform to Draft 2020-12 specification."""
    # 1. LogicalGraph schema
    graph_schema = get_json_schema("LogicalGraph")
    assert graph_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(graph_schema)

    # 2. LogicalNode schema
    node_schema = get_json_schema("LogicalNode")
    assert node_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(node_schema)

    # 3. SnapshotEnvelope schema
    envelope_schema = get_json_schema("SnapshotEnvelope")
    assert envelope_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator.check_schema(envelope_schema)

    # 4. Target 'all'
    all_schemas = get_json_schema("all")
    assert "LogicalGraph" in all_schemas
    assert "LogicalNode" in all_schemas
    assert "SnapshotEnvelope" in all_schemas

    # 5. Invalid target raises ValueError
    with pytest.raises(ValueError, match="Unsupported schema target"):
        get_json_schema("InvalidTarget")


def test_validate_instances_against_emitted_json_schemas() -> None:
    """Verify that serialized IR instances validate against the emitted JSON schemas."""
    graph_schema = get_json_schema("LogicalGraph")
    validator = Draft202012Validator(graph_schema)

    node = LogicalNode(
        id="conv1",
        op_type="Conv",
        domain="ai.onnx",
        attributes={"pads": [1, 1, 1, 1]},
    )
    graph = LogicalGraph(
        name="SimpleConv",
        nodes=[node],
        outputs=["conv1"],
    )

    raw_json = json.loads(graph.to_json())
    validator.validate(raw_json)

    # Validate node schema
    node_schema = get_json_schema("LogicalNode")
    node_validator = Draft202012Validator(node_schema)
    node_dict = {
        "id": "gemm1",
        "op_type": "Gemm",
        "domain": "ai.onnx",
        "attributes": {"alpha": 1.0},
    }
    node_validator.validate(node_dict)

    # Validate envelope schema
    envelope_schema = get_json_schema("SnapshotEnvelope")
    envelope_validator = Draft202012Validator(envelope_schema)
    env = SnapshotEnvelope(
        schema_version="2.0.0",
        target="stablehlo",
        version="1.0.0",
        source_type="tablegen",
        generated_at="2026-09-12T00:00:00Z",
    )
    envelope_validator.validate(env.model_dump())


def test_generate_typescript_definitions() -> None:
    """Verify generated TypeScript definitions contain required interfaces and types."""
    ts_code = generate_typescript_definitions()
    assert "export interface LogicalGraph" in ts_code
    assert "export interface LogicalNode" in ts_code
    assert "export interface LogicalEdge" in ts_code
    assert "export interface LogicalMesh" in ts_code
    assert "export interface PartitionSpec" in ts_code
    assert "export interface SnapshotEnvelope" in ts_code
    assert "export type DType =" in ts_code
    assert "export type AttributeValue =" in ts_code


def test_export_schemas_directory(tmp_path: Path) -> None:
    """Test exporting JSON schemas and TypeScript definitions to disk.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    out_dir = tmp_path / "schemas"
    exported = export_schemas(out_dir, include_typescript=True)

    assert "LogicalGraph" in exported
    assert "LogicalNode" in exported
    assert "SnapshotEnvelope" in exported
    assert "TypeScript" in exported

    assert exported["LogicalGraph"].is_file()
    assert exported["LogicalNode"].is_file()
    assert exported["SnapshotEnvelope"].is_file()
    assert exported["TypeScript"].is_file()

    with open(exported["LogicalGraph"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["$schema"] == "https://json-schema.org/draft/2020-12/schema"

    with open(exported["TypeScript"], "r", encoding="utf-8") as f:
        content = f.read()
        assert "export interface LogicalGraph" in content

    # Test export without TypeScript
    out_dir_no_ts = tmp_path / "schemas_no_ts"
    exported_no_ts = export_schemas(out_dir_no_ts, include_typescript=False)
    assert "TypeScript" not in exported_no_ts
    assert len(exported_no_ts) == 3


def test_cli_export_schema_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI export-schema command outputting to stdout.

    Args:
        capsys (pytest.CaptureFixture[str]): Pytest capsys fixture.
    """
    # 1. Default export all schemas
    main(["export-schema"])
    captured = capsys.readouterr()
    assert "LogicalGraph" in captured.out
    assert "https://json-schema.org/draft/2020-12/schema" in captured.out

    # 2. Export single schema target
    main(["export-schema", "--target", "LogicalNode"])
    captured = capsys.readouterr()
    node_schema = json.loads(captured.out)
    assert node_schema["title"] == "LogicalNode"

    # 3. Export TypeScript to stdout
    main(["export-schema", "--typescript"])
    captured = capsys.readouterr()
    assert "export interface LogicalGraph" in captured.out


def test_cli_export_schema_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI export-schema command exporting to directory and custom file paths.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest capsys fixture.
    """
    # 1. Export schemas to directory with TypeScript
    out_dir = tmp_path / "cli_schemas"
    main(["export-schema", "--out-dir", str(out_dir), "--typescript"])
    captured = capsys.readouterr()
    assert f"Exported 4 schemas to {out_dir}" in captured.out
    assert (out_dir / "logical_graph.schema.json").is_file()
    assert (out_dir / "logical_graph.d.ts").is_file()

    # 2. Export TypeScript to custom output file
    ts_out = tmp_path / "types" / "graph.d.ts"
    main(["export-schema", "--ts-out", str(ts_out)])
    captured = capsys.readouterr()
    assert f"Exported TypeScript interfaces to {ts_out}" in captured.out
    assert ts_out.is_file()
