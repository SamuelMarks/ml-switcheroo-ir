"""Tests for TypeScript schema structural parity and negative type checking with tsc."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ml_switcheroo_ir import (
    LogicalEdge,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
    PipelineTopologyConfig,
    TensorSpec,
    WebRTCSignalingTopology,
)
from ml_switcheroo_ir.export import generate_typescript_definitions, get_json_schema
from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope


def _parse_ts_interface_properties(ts_source: str) -> dict[str, set[str]]:
    """Parse ``export interface`` declarations into per-interface property sets.

    Resolves single-inheritance ``extends BaseInterface`` blocks so derived interfaces
    include their inherited properties.

    Args:
        ts_source (str): TypeScript declaration source text.

    Returns:
        dict[str, set[str]]: Mapping from interface name to declared/inherited property names.
    """
    pattern = re.compile(
        r"export\s+interface\s+(?P<name>\w+)(?:\s+extends\s+(?P<base>\w+))?\s*\{(?P<body>[^}]*)\}",
        re.MULTILINE,
    )
    prop_pattern = re.compile(r"^\s*(?P<prop>[A-Za-z_]\w*)\??\s*:", re.MULTILINE)
    raw_props: dict[str, set[str]] = {}
    bases: dict[str, str] = {}
    for match in pattern.finditer(ts_source):
        iface_name = match.group("name")
        base_name = match.group("base")
        body = match.group("body")
        raw_props[iface_name] = {m.group("prop") for m in prop_pattern.finditer(body)}
        if base_name:
            bases[iface_name] = base_name

    resolved: dict[str, set[str]] = {}
    for iface_name, props in raw_props.items():
        merged = set(props)
        cur_base = bases.get(iface_name)
        while cur_base is not None and cur_base in raw_props:
            merged |= raw_props[cur_base]
            cur_base = bases.get(cur_base)
        resolved[iface_name] = merged
    return resolved


def _verify_ts_compilation(work_dir: Path) -> None:
    """Compile generated TypeScript definitions with positive and ``@ts-expect-error`` negative fixtures.

    Runs ``tsc --noEmit --strict`` so unused ``@ts-expect-error`` annotations fail compilation
    if invalid types (such as arbitrary ``DType`` strings or boolean partition axes) are accepted.

    Args:
        work_dir (Path): Working directory for temporary TypeScript files.
    """
    tsc_path = shutil.which("tsc")
    if tsc_path is None:
        pytest.fail("TypeScript compiler (tsc) is required for schema validation.")
    assert tsc_path is not None

    ts_code = generate_typescript_definitions()
    ts_file = work_dir / "schema.d.ts"
    ts_file.write_text(ts_code, encoding="utf-8")

    consumer_file = work_dir / "consumer.ts"
    consumer_code = """
import {
  DType,
  LogicalGraph,
  LogicalNode,
  LogicalMesh,
  PartitionSpec,
  ZeroTangent,
  NoTangent,
  PipelineTopologyConfig,
} from "./schema";

const validDtype: DType = "bfloat16";

// @ts-expect-error arbitrary non-DType string must be rejected
const invalidDtype: DType = "not_a_real_dtype";

const validSharding: PartitionSpec = {
  axes: ["data", 0, ["tensor", 1], null],
};

// @ts-expect-error boolean is not a valid PartitionAxisElement
const invalidSharding: PartitionSpec = { axes: [true] };

const mesh: LogicalMesh = {
  shape: { data: 2, model: 4 },
  webrtc_topology: {
    signaling_url: "ws://localhost:8080",
    peers: [{ peer_id: "p1", device_capability: "webgpu" }],
  },
};

const pipeline: PipelineTopologyConfig = {
  microbatch_splitting: { strategy: "uniform", num_microbatches: 4 },
  mesh_mapping: { devices_per_stage: 1 },
  stage_communication: { protocol: "webrtc" },
  dependencies: [{ source_stage: "s0", target_stage: "s1", offset_mb: 0 }],
};

const node: LogicalNode = {
  id: "conv1",
  op_type: "Conv",
  domain: "ai.onnx",
  version: 1,
  attributes: { kernel_shape: [3, 3], target_dtype: validDtype },
  inputs: ["input"],
  outputs: ["output"],
  output_specs: [{ shape: ["B", 64, 32, 32], dtype: "float32" }],
  sharding: validSharding,
  subgraphs: { meta_flag: 1 },
};

const zeroNode: ZeroTangent = {
  id: "zt",
  op_type: "ZeroTangent",
  domain: "ml.switcheroo.ad",
};

const noNode: NoTangent = {
  id: "nt",
  op_type: "NoTangent",
  domain: "ml.switcheroo.ad",
};

// @ts-expect-error wrong op_type literal for ZeroTangent
const badZero: ZeroTangent = { id: "bad", op_type: "Relu", domain: "ml.switcheroo.ad" };

const graph: LogicalGraph = {
  name: "SampleGraph",
  nodes: { conv1: node, zt: zeroNode, nt: noNode },
  inputs: ["input"],
  outputs: ["output"],
  mesh,
  pipeline_topology: pipeline,
  attributes: { compiled_tier: "gpu" },
  loop_options: { unroll: 4 },
};

// @ts-expect-error unknown property on LogicalGraph must fail excess property check
const badGraph: LogicalGraph = { name: "Bad", nonexistent_field: 123 };

void [invalidDtype, invalidSharding, badZero, graph, badGraph];
"""
    consumer_file.write_text(consumer_code, encoding="utf-8")

    result = subprocess.run(
        [tsc_path, "--noEmit", "--strict", str(consumer_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"tsc strict validation failed: {result.stdout} {result.stderr}"


def test_typescript_definitions_compile(tmp_path: Path) -> None:
    """Validate that generate_typescript_definitions compiles positive and negative fixtures under tsc --strict.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    _verify_ts_compilation(tmp_path)
    assert (tmp_path / "schema.d.ts").is_file()


def test_typescript_missing_tsc_and_structural_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify exact per-interface property parity against Pydantic models and JSON Schemas.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    ts_defs = generate_typescript_definitions()
    ts_interfaces = _parse_ts_interface_properties(ts_defs)

    for schema_name in (
        "LogicalGraph",
        "LogicalNode",
        "SnapshotEnvelope",
        "PipelineTopologyConfig",
        "WebRTCSignalingTopology",
    ):
        assert schema_name in ts_interfaces
        json_schema = get_json_schema(schema_name)
        props = json_schema.get("properties", {})
        assert isinstance(props, dict)
        assert set(props.keys()).issubset(ts_interfaces[schema_name])

    for cls_name, pyd_cls in (
        ("LogicalGraph", LogicalGraph),
        ("LogicalNode", LogicalNode),
        ("LogicalEdge", LogicalEdge),
        ("LogicalMesh", LogicalMesh),
        ("PartitionSpec", PartitionSpec),
        ("TensorSpec", TensorSpec),
        ("SnapshotEnvelope", SnapshotEnvelope),
        ("PipelineTopologyConfig", PipelineTopologyConfig),
        ("WebRTCSignalingTopology", WebRTCSignalingTopology),
    ):
        assert cls_name in ts_interfaces
        assert set(pyd_cls.model_fields.keys()).issubset(ts_interfaces[cls_name])

    assert "edges" in ts_interfaces["LogicalGraph"]
    assert "| string;" not in ts_defs.split("export interface TensorShape")[0]

    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    with pytest.raises(pytest.fail.Exception, match="TypeScript compiler .* is required"):
        _verify_ts_compilation(tmp_path)
