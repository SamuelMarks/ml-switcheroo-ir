"""Tests for TypeScript schema validity with tsc compiler."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ml_switcheroo_ir.export import generate_typescript_definitions


def test_typescript_definitions_compile(tmp_path: Path) -> None:
    """Validate that generate_typescript_definitions compiles without error using tsc."""
    tsc_path = shutil.which("tsc")
    if tsc_path is None:
        pytest.fail("TypeScript compiler (tsc) is required for schema validation.")

    ts_code = generate_typescript_definitions()
    ts_file = tmp_path / "schema.d.ts"
    ts_file.write_text(ts_code, encoding="utf-8")

    # Add a sample usage test file to ensure interfaces are sound
    consumer_file = tmp_path / "consumer.ts"
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
        f"tsc validation failed:\n{result.stdout}\n{result.stderr}"
    )


def test_typescript_missing_tsc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that missing tsc compiler triggers pytest.fail.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    with pytest.raises(
        pytest.fail.Exception, match="TypeScript compiler .* is required"
    ):
        test_typescript_definitions_compile(tmp_path)
