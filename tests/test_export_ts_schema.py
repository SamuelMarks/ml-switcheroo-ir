"""Tests for TypeScript schema validity with tsc compiler."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import shutil
import subprocess

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


def _verify_ts_compilation(work_dir: Path) -> None:
    """Compile generated TypeScript definitions and sample consumer code using tsc.

    Args:
        work_dir (Path): Working directory for temporary TypeScript files.
    """
    tsc_path = shutil.which("tsc")
    if tsc_path is None:
        pytest.fail("TypeScript compiler (tsc) is required for schema validation.")

    ts_code = generate_typescript_definitions()
    ts_file = work_dir / "schema.d.ts"
    ts_file.write_text(ts_code, encoding="utf-8")

    consumer_file = work_dir / "consumer.ts"
    consumer_code = """
import {
  LogicalGraph,
  LogicalNode,
  LogicalMesh,
  PartitionSpec,
  ZeroTangent,
  NoTangent,
  PipelineTopologyConfig,
  WebRTCSignalingTopology,
} from "./schema";

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
  attributes: { kernel_shape: [3, 3] },
  inputs: ["input"],
  outputs: ["output"],
  output_specs: [{ shape: ["B", 64, 32, 32], dtype: "float32" }],
  sharding: { axes: ["data", null, null, null] },
};

const zeroNode: ZeroTangent = {
  id: "zt",
  op_type: "ZeroTangent",
  domain: "ml.switcheroo.ad",
};

const graph: LogicalGraph = {
  name: "SampleGraph",
  nodes: { conv1: node, zt: zeroNode },
  inputs: ["input"],
  outputs: ["output"],
  mesh,
  pipeline_topology: pipeline,
};
"""
    consumer_file.write_text(consumer_code, encoding="utf-8")

    result = subprocess.run(
        [tsc_path, "--noEmit", str(consumer_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"tsc validation failed: {result.stdout} {result.stderr}"
    )


def test_typescript_definitions_compile(tmp_path: Path) -> None:
    """Validate that generate_typescript_definitions compiles without error using tsc.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    _verify_ts_compilation(tmp_path)


def test_typescript_missing_tsc_and_structural_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test missing tsc failure handling and verify TypeScript/Python field parity across all IR models.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    ts_defs = generate_typescript_definitions()
    for schema_name in (
        "LogicalGraph",
        "LogicalNode",
        "SnapshotEnvelope",
        "PipelineTopologyConfig",
        "WebRTCSignalingTopology",
    ):
        json_schema = get_json_schema(schema_name)
        props = json_schema.get("properties", {})
        assert isinstance(props, dict)
        for prop_name in props:
            assert prop_name in ts_defs

    for dc_cls in (
        LogicalGraph,
        LogicalNode,
        LogicalEdge,
        LogicalMesh,
        PartitionSpec,
        TensorSpec,
    ):
        for f in fields(dc_cls):
            assert f.name in ts_defs

    for pyd_cls in (
        SnapshotEnvelope,
        PipelineTopologyConfig,
        WebRTCSignalingTopology,
    ):
        for field_name in pyd_cls.model_fields:
            assert field_name in ts_defs

    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    with pytest.raises(
        pytest.fail.Exception, match="TypeScript compiler .* is required"
    ):
        _verify_ts_compilation(tmp_path)
