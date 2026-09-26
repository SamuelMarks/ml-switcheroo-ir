"""Tests for JSON Schema and TypeScript export utilities and CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_switcheroo_ir.cli import main
from ml_switcheroo_ir.export import (
    export_schemas,
    generate_typescript_definitions,
    get_json_schema,
)


def test_get_json_schema_logical_graph() -> None:
    """Test generating JSON Schema for LogicalGraph."""
    schema = get_json_schema("LogicalGraph")
    assert isinstance(schema, dict)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "LogicalGraph" in schema.get("title", "") or "properties" in schema
    assert "nodes" in schema.get("properties", {})


def test_get_json_schema_logical_node() -> None:
    """Test generating JSON Schema for LogicalNode."""
    schema = get_json_schema("LogicalNode")
    assert isinstance(schema, dict)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "LogicalNode" in schema.get("title", "") or "properties" in schema
    assert "id" in schema.get("properties", {})


def test_get_json_schema_snapshot_envelope() -> None:
    """Test generating JSON Schema for SnapshotEnvelope."""
    schema = get_json_schema("SnapshotEnvelope")
    assert isinstance(schema, dict)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "SnapshotEnvelope" in schema.get("title", "") or "properties" in schema
    assert "schema_version" in schema.get("properties", {})


def test_get_json_schema_extended_targets() -> None:
    """Test generating JSON Schema for ZeroTangent, NoTangent, and topologies."""
    for target in [
        "ZeroTangent",
        "NoTangent",
        "PipelineTopologyConfig",
        "WebRTCSignalingTopology",
    ]:
        schema = get_json_schema(target)
        assert isinstance(schema, dict)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert "properties" in schema


def test_get_json_schema_all() -> None:
    """Test generating all schemas at once."""
    schemas = get_json_schema("all")
    assert isinstance(schemas, dict)
    assert "LogicalGraph" in schemas
    assert "LogicalNode" in schemas
    assert "SnapshotEnvelope" in schemas
    assert "ZeroTangent" in schemas
    assert "NoTangent" in schemas
    assert "PipelineTopologyConfig" in schemas
    assert "WebRTCSignalingTopology" in schemas
    assert (
        schemas["LogicalGraph"]["$schema"]
        == "https://json-schema.org/draft/2020-12/schema"
    )


def test_get_json_schema_invalid_target() -> None:
    """Test requesting unsupported schema target raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported schema target"):
        get_json_schema("InvalidModelName")


def test_generate_typescript_definitions() -> None:
    """Test TypeScript definitions generation and syntax validation."""
    ts_code = generate_typescript_definitions()
    assert isinstance(ts_code, str)
    assert "export interface LogicalGraph" in ts_code
    assert "export interface LogicalNode" in ts_code
    assert "export interface LogicalEdge" in ts_code
    assert "export interface LogicalMesh" in ts_code
    assert "export interface PartitionSpec" in ts_code
    assert "export interface SnapshotEnvelope" in ts_code
    assert "export interface ZeroTangent" in ts_code
    assert "export interface NoTangent" in ts_code
    assert "export interface PipelineTopologyConfig" in ts_code
    assert "export interface WebRTCSignalingTopology" in ts_code
    assert "export type SymNode =" in ts_code
    assert "export interface SymInt" in ts_code
    assert "export type DType =" in ts_code
    assert "export type AttributeValue =" in ts_code


def test_export_schemas_directory(tmp_path: Path) -> None:
    """Test exporting JSON schemas and TypeScript definitions to disk.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    out_dir = tmp_path / "schemas"
    out_dir_no_ts = tmp_path / "schemas_no_ts"

    exported = export_schemas(out_dir, include_typescript=True)

    assert "LogicalGraph" in exported
    assert "LogicalNode" in exported
    assert "SnapshotEnvelope" in exported
    assert "ZeroTangent" in exported
    assert "NoTangent" in exported
    assert "PipelineTopologyConfig" in exported
    assert "WebRTCSignalingTopology" in exported
    assert "TypeScript" in exported

    assert exported["LogicalGraph"].is_file()
    assert exported["LogicalNode"].is_file()
    assert exported["SnapshotEnvelope"].is_file()
    assert exported["TypeScript"].is_file()

    with open(exported["LogicalGraph"], "r", encoding="utf-8") as f:
        schema = json.load(f)
        assert "$schema" in schema

    with open(exported["TypeScript"], "r", encoding="utf-8") as f:
        content = f.read()
        assert "export interface LogicalGraph" in content

    # Test export without TypeScript
    exported_no_ts = export_schemas(out_dir_no_ts, include_typescript=False)
    assert "TypeScript" not in exported_no_ts
    assert len(exported_no_ts) == 7


def test_cli_export_schema_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI export-schema command outputting to stdout.

    Args:
        capsys: Pytest capture fixture.
    """
    # 1. Default export all schemas
    main(["export-schema"])
    captured = capsys.readouterr()
    assert '"LogicalGraph":' in captured.out
    assert '"LogicalNode":' in captured.out
    assert '"SnapshotEnvelope":' in captured.out

    # 2. Export single schema target
    main(["export-schema", "--target", "LogicalNode"])
    captured = capsys.readouterr()
    assert '"title": "LogicalNode"' in captured.out or '"id":' in captured.out

    # 3. Export TypeScript to stdout
    main(["export-schema", "--typescript"])
    captured = capsys.readouterr()
    assert "export interface LogicalGraph" in captured.out


def test_cli_export_schema_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test CLI export-schema command exporting to directory and custom file paths.

    Args:
        tmp_path: Pytest temporary directory fixture.
        capsys: Pytest capture fixture.
    """
    out_dir = tmp_path / "cli_schemas"
    ts_out = tmp_path / "custom_types.d.ts"

    # 1. Export schemas to directory with TypeScript
    main(["export-schema", "--out-dir", str(out_dir), "--typescript"])
    captured = capsys.readouterr()
    assert f"Exported 8 schemas to {out_dir}" in captured.out
    assert (out_dir / "logical_graph.schema.json").is_file()
    assert (out_dir / "logical_graph.d.ts").is_file()

    # 2. Export TypeScript to custom output file
    main(["export-schema", "--ts-out", str(ts_out)])
    captured = capsys.readouterr()
    assert f"Exported TypeScript interfaces to {ts_out}" in captured.out
    assert ts_out.is_file()
