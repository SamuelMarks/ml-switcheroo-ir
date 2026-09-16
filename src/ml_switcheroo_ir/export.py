"""JSON Schema and TypeScript definitions exporter for ML-Switcheroo IR.

Provides utilities for emitting canonical Draft 2020-12 conforming JSON schemas
for LogicalGraph, LogicalNode, and SnapshotEnvelope, as well as TypeScript
interfaces for web playground and frontend integrations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope

DRAFT_2020_12_SCHEMA_URI: str = "https://json-schema.org/draft/2020-12/schema"


def get_json_schema(target: str = "LogicalGraph") -> dict[str, Any]:
    """Emit canonical JSON Schema conforming to Draft 2020-12 specification.

    Args:
        target (str): Target model ('LogicalGraph', 'LogicalNode', 'SnapshotEnvelope', or 'all').
            Defaults to 'LogicalGraph'.

    Returns:
        dict[str, Any]: Generated JSON Schema dictionary.

    Raises:
        ValueError: If target is not recognized.
    """
    target_clean = target.strip()
    if target_clean == "LogicalGraph":
        schema: dict[str, Any] = TypeAdapter(LogicalGraph).json_schema()
        schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
        return schema
    if target_clean == "LogicalNode":
        schema = TypeAdapter(LogicalNode).json_schema()
        schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
        return schema
    if target_clean == "SnapshotEnvelope":
        schema = SnapshotEnvelope.model_json_schema()
        schema["$schema"] = DRAFT_2020_12_SCHEMA_URI
        return schema
    if target_clean == "all":
        return {
            "LogicalGraph": get_json_schema("LogicalGraph"),
            "LogicalNode": get_json_schema("LogicalNode"),
            "SnapshotEnvelope": get_json_schema("SnapshotEnvelope"),
        }
    raise ValueError(
        f"Unsupported schema target: '{target}'. Supported targets: 'LogicalGraph', 'LogicalNode', 'SnapshotEnvelope', 'all'."
    )


def generate_typescript_definitions() -> str:
    """Generate typed TypeScript interfaces matching LogicalGraph for frontend integration.

    Returns:
        str: TypeScript interface definitions.
    """
    return """/**
 * ML-Switcheroo IR - Canonical TypeScript Definitions
 * Auto-generated for Abstract ML Machine Compiler Ecosystem (Tier 1)
 */

export type DType =
  | "float32"
  | "float16"
  | "bfloat16"
  | "float64"
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64"
  | "bool"
  | "string"
  | "object"
  | "complex64"
  | "complex128"
  | "float8_e4m3fn"
  | "float8_e4m3b11fnuz"
  | "float8_e5m2"
  | "fp8_e4m3fn"
  | "fp8_e4m3fnuz"
  | "fp8_e5m2"
  | "fp8_e5m2fnuz"
  | "int4"
  | "uint4"
  | "int2"
  | "qint8"
  | "quint8"
  | "qint4"
  | string;

export type AttributeValue =
  | string
  | number
  | boolean
  | null
  | AttributeValue[]
  | { [key: string]: AttributeValue };

export interface PartitionSpec {
  axes: (string | string[] | null)[];
}

export interface LogicalMesh {
  shape: Record<string, number>;
}

export interface LogicalEdge {
  source: string;
  target: string;
  source_idx?: number;
  target_idx?: number;
  value_name?: string | null;
}

export interface TensorSpec {
  shape: (number | string)[];
  dtype: DType;
  sparsity?: string | null;
}

export interface LogicalNode {
  id: string;
  op_type: string;
  domain?: string;
  version?: number;
  attributes?: Record<string, AttributeValue>;
  inputs?: string[];
  outputs?: string[];
  shape_metadata?: (number | string)[] | null;
  dtype?: DType | null;
  output_specs?: TensorSpec[];
  subgraphs?: Record<string, LogicalGraph>;
  device?: string | null;
  stream?: string | null;
  sharding?: PartitionSpec | null;
  subgraph?: LogicalGraph | null;
}

export interface LogicalGraph {
  name?: string;
  nodes: Record<string, LogicalNode> | LogicalNode[];
  inputs?: string[];
  input_specs?: Record<string, TensorSpec>;
  outputs?: string[];
  initializers?: Record<string, unknown>;
  mesh?: LogicalMesh | null;
  edges?: LogicalEdge[];
}

export interface SnapshotEnvelope {
  schema_version: string;
  target: string;
  version: string;
  upstream_version?: string | null;
  source_type: string;
  upstream_commit?: string | null;
  supported_microarchitectures?: string[] | null;
  generated_at: string;
  environment?: Record<string, string>;
  categories?: Record<string, unknown>;
}
"""


def export_schemas(
    out_dir: Path | str,
    include_typescript: bool = False,
) -> dict[str, Path]:
    """Export canonical JSON schemas and optional TypeScript definitions to a directory.

    Args:
        out_dir (Path | str): Output directory path.
        include_typescript (bool): Whether to also write TypeScript definition file. Defaults to False.

    Returns:
        dict[str, Path]: Mapping from schema name to exported file path.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}

    for name in ("LogicalGraph", "LogicalNode", "SnapshotEnvelope"):
        schema = get_json_schema(name)
        if name == "LogicalGraph":
            out_file = directory / "logical_graph.schema.json"
        elif name == "LogicalNode":
            out_file = directory / "logical_node.schema.json"
        else:
            out_file = directory / "snapshot_envelope.schema.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, sort_keys=True)
        exported[name] = out_file

    if include_typescript:
        ts_file = directory / "logical_graph.d.ts"
        with open(ts_file, "w", encoding="utf-8") as f:
            f.write(generate_typescript_definitions())
        exported["TypeScript"] = ts_file

    return exported
