# Comprehensive Usage & Developer Guide

`ml-switcheroo-ir` provides both a Python API and an operational CLI utility for the `ml-switcheroo` and `zero-*` compilation ecosystem.

This guide details programmatic graph manipulation, multi-dialect schema validation, anti-hallucination grounding audits with `ml-ecosystem-snapshots`, static compliance analysis, pure-IR transformations, cross-framework parameter translation, and hardware ISA validation.

---

## Table of Contents

1. [Programmatic Python API](#1-programmatic-python-api)
   - [Constructing Graphs, Nodes & Nested Subgraphs](#constructing-graphs-nodes--nested-subgraphs)
   - [Multi-Output Nodes & SSA Value Tracking](#multi-output-nodes--ssa-value-tracking)
   - [Distributed Sharding & Device Meshes](#distributed-sharding--device-meshes)
   - [SPMD Sharding Propagation & Pipeline Validation](#spmd-sharding-propagation--pipeline-validation)
   - [Analytical Collective Communication Cost Estimation](#analytical-collective-communication-cost-estimation)
   - [Topological Sorting & Cycle Detection](#topological-sorting--cycle-detection)
   - [Dual JSON Serialization & High-Throughput Streaming I/O](#dual-json-serialization--high-throughput-streaming-io)
   - [Native Graph Optimizations (DCE, CSE, Shape Propagation)](#native-graph-optimizations-dce-cse-shape-propagation)
   - [Zero-Hallucination Parameter Translation](#zero-hallucination-parameter-translation)
   - [Hardware ISA & GPU Shader Validation](#hardware-isa--gpu-shader-validation)
   - [Extending Custom Operator Schemas](#extending-custom-operator-schemas)
   - [Validating Graphs](#validating-graphs)
   - [Auditing Graphs with `ml-ecosystem-snapshots`](#auditing-graphs-with-ml-ecosystem-snapshots)
   - [Programmatic Schema Export & TypeScript Generation](#programmatic-schema-export--typescript-generation)
2. [Interface Protocols for Compilers & Frontends](#2-interface-protocols-for-compilers--frontends)
   - [Implementing `GraphFrontend`](#implementing-graphfrontend)
   - [Implementing `CompilerBackend`](#implementing-compilerbackend)
3. [CLI Reference & Workflows](#3-cli-reference--workflows)
   - [Graph Validation (`validate`)](#1-graph-validation-validate)
   - [Snapshot Grounding (`ground`)](#2-snapshot-grounding-ground)
   - [Compliance & Coverage Auditing (`compliance`)](#3-compliance--coverage-auditing-compliance)
   - [Operator Discovery (`list-ops`)](#4-operator-discovery-list-ops)
   - [Topological Sorting (`toposort`)](#5-topological-sorting-toposort)
   - [Backend Interface Verification (`verify-backend`)](#6-backend-interface-verification-verify-backend)
   - [Exporting GhostRef Snapshots (`dump-snapshot`)](#7-exporting-ghostref-snapshots-dump-snapshot)
   - [Exporting JSON Schemas & TypeScript (`export-schema`)](#8-exporting-json-schemas--typescript-export-schema)
4. [End-to-End Developer Recipes](#4-end-to-end-developer-recipes)
   - [Recipe A: Tracking `zero-*` Dialect Implementation Progress](#recipe-a-tracking-zero--dialect-implementation-progress)
   - [Recipe B: Pre-Compilation Sanitization in CI/CD](#recipe-b-pre-compilation-sanitization-in-cicd)
   - [Recipe C: Validating StableHLO & MLIR Lowerings](#recipe-c-validating-stablehlo--mlir-lowerings)
   - [Recipe D: Hardware-Specific SASS Scoreboarding & RDNA3 VOPD Validation](#recipe-d-hardware-specific-sass-scoreboarding--rdna3-vopd-validation)
   - [Recipe E: Pre-Lowering Graph Optimization Pipeline](#recipe-e-pre-lowering-graph-optimization-pipeline)

---

## 1. Programmatic Python API

### Constructing Graphs, Nodes & Nested Subgraphs

A computation graph is represented by `LogicalGraph`, containing `LogicalNode` instances connected by input strings or `LogicalEdge` objects. Nodes can also contain nested subgraphs for control flow (`body`, `then`, `else`) or autodiff (`bwd`, `jvp`):

```python
from ml_switcheroo_ir import (
    DType,
    LogicalGraph,
    LogicalNode,
)

# Create input node
x_node = LogicalNode(
    id="x",
    op_type="Input",
    domain="ai.onnx",
    attributes={"dtype": DType.float32.value, "shape": [1, 128, 768]},
    outputs=["x"],
)

# Create compute node (RMSNorm)
norm_node = LogicalNode(
    id="norm",
    op_type="RMSNorm",
    domain="ml.switcheroo.custom",
    attributes={"eps": 1e-6},
    inputs=["x"],
    outputs=["norm_out"],
)

# Assemble graph
graph = LogicalGraph(
    name="LayerNormBlock",
    nodes={"x": x_node, "norm": norm_node},
    outputs=["norm_out"],
)

# Access nodes using dictionary key or sequence iteration
assert graph["norm"].op_type == "RMSNorm"
assert len(graph) == 2
for node in graph:
    print(f"Node: {node.id} ({node.op_type})")

# Nested Subgraph Example: Loop or Conditional Block
body_node = LogicalNode(
    id="body_add",
    op_type="Add",
    domain="ai.onnx",
    inputs=["sub_x", "sub_y"],
    outputs=["sub_out"],
)
body_subgraph = LogicalGraph(
    name="LoopBody",
    nodes={"body_add": body_node},
    outputs=["sub_out"],
)

loop_node = LogicalNode(
    id="loop1",
    op_type="Loop",
    domain="ai.onnx",
    inputs=["max_trip", "cond", "init_val"],
    outputs=["final_val"],
    subgraphs={"body": body_subgraph},
)
control_graph = LogicalGraph(nodes={"loop1": loop_node})
assert "body" in control_graph["loop1"].subgraphs
```

### Multi-Output Nodes & SSA Value Tracking

Nodes that produce multiple tensors (e.g., `Split`, `BatchNorm`, or multi-result MLIR ops) specify output identifiers in `LogicalNode.outputs`:

```python
from ml_switcheroo_ir import LogicalGraph, LogicalNode

# Split tensor into 2 parts
split_node = LogicalNode(
    id="split1",
    op_type="Split",
    domain="ai.onnx",
    attributes={"axis": -1},
    inputs=["x"],
    outputs=["chunk_a", "chunk_b"],
)

graph = LogicalGraph(nodes={"split1": split_node})

# Resolve which node produced an SSA value
producer = graph.get_output_producer("chunk_b")
assert producer is not None and producer.id == "split1"

# Query consumer nodes
downstream = graph.get_outputs("split1")
```

### Distributed Sharding & Device Meshes

For SPMD distributed execution (compatible with JAX/XLA device meshes):

```python
from ml_switcheroo_ir import LogicalGraph, LogicalMesh, LogicalNode, PartitionSpec

# Define a 2D device mesh (4 data parallel devices x 2 tensor parallel devices)
mesh = LogicalMesh(shape={"data": 4, "model": 2})

# Shard dimension 0 along "data" and replicate dimension 1
sharded_node = LogicalNode(
    id="weights",
    op_type="Constant",
    domain="ai.onnx",
    attributes={"value": [1.0, 2.0]},
    sharding=PartitionSpec(axes=("data", None)),
)

graph = LogicalGraph(nodes={"weights": sharded_node}, mesh=mesh)
```

### SPMD Sharding Propagation & Pipeline Validation

Verify distributed invariants, cross-stage pipelining boundaries, and activation checkpointing scopes:

```python
from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
)
from ml_switcheroo_ir.validator import Validator

mesh = LogicalMesh(shape={"data": 4, "model": 2})

# 1. Sharding Propagation Verification (Elementwise, Contraction, Reduction)
n1 = LogicalNode(
    id="n1",
    op_type="Relu",
    domain="ai.onnx",
    sharding=PartitionSpec(axes=("data", None)),
)
n2 = LogicalNode(
    id="n2",
    op_type="Relu",
    domain="ai.onnx",
    inputs=["n1"],
    sharding=PartitionSpec(axes=("data", None)),
)
dist_graph = LogicalGraph(nodes={"n1": n1, "n2": n2}, mesh=mesh)

validator = Validator()
sharding_errors = validator.validate_sharding_propagation(dist_graph)
assert not sharding_errors

# 2. Pipeline Stage Progression & Checkpoint Scope Validation
s0 = LogicalNode(
    id="stage0",
    op_type="Relu",
    domain="ai.onnx",
    attributes={"pipeline_stage": 0},
)
s1 = LogicalNode(
    id="stage1",
    op_type="Relu",
    domain="ai.onnx",
    inputs=["stage0"],
    attributes={"pipeline_stage": 1, "checkpoint_tag": "recompute"},
)
pipe_graph = LogicalGraph(nodes={"stage0": s0, "stage1": s1})
pipe_errors = validator.validate_pipeline_and_checkpointing(pipe_graph)
assert not pipe_errors
```

### Analytical Collective Communication Cost Estimation

Calculate exact closed-form byte-level transfer volumes for distributed operations (`AllReduce`, `AllGather`, `ReduceScatter`, `P2P`):

```python
from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    estimate_communication_volume,
    estimate_graph_communication_volume,
)

mesh = LogicalMesh(shape={"data": 4, "model": 2})

# 1. Estimate single collective volume
# 1024x1024 float32 tensor (4 MB) AllReduce along 4 data devices: 2 * (3/4) * 4 MB = 6 MB
ar_node = LogicalNode(
    id="all_reduce1",
    op_type="collective.all_reduce",
    domain="collective",
    shape_metadata=(1024, 1024),
    attributes={"reduction_op": "sum", "mesh_axis": "data", "dtype": "float32"},
)
vol_bytes = estimate_communication_volume(ar_node, mesh)
assert vol_bytes == 6 * 1024 * 1024

# 2. Aggregate whole-graph communication volume
coll_graph = LogicalGraph(nodes={"all_reduce1": ar_node}, mesh=mesh)
summary = estimate_graph_communication_volume(coll_graph)
print(f"Total Transfer: {summary['total_volume_bytes']} bytes")
print(f"By Mesh Axis: {summary['by_axis']}")
```

### Topological Sorting & Cycle Detection

Sort nodes into valid execution order:

```python
from ml_switcheroo_ir import CyclicGraphError, LogicalGraph, topological_sort

try:
    ordered_nodes = topological_sort(graph, strict=True)
    print("Execution order:", [n.id for n in ordered_nodes])
except CyclicGraphError as e:
    print("Graph contains dependency cycles:", e)
```

### Dual JSON Serialization & High-Throughput Streaming I/O

Serialize and deserialize graphs deterministically across frontends and backends, including zero-allocation streaming and compressed formats:

```python
# 1. Deterministic JSON serialization
json_str = graph.to_json(format="canonical", indent=2)
restored_graph = LogicalGraph.from_json(json_str)
assert len(restored_graph) == len(graph)

# 2. Streaming directly to a writable stream (avoids giant string allocations)
with open("model.json", "w", encoding="utf-8") as f:
    graph.to_stream(f)

# 3. High-throughput compressed disk I/O (gzip and zstandard)
graph.to_file("model.json.gz")
graph_gz = LogicalGraph.from_file("model.json.gz")
assert len(graph_gz) == len(graph)
```

### Native Graph Optimizations (DCE, CSE, Shape Propagation)

`ml-switcheroo-ir` includes native, framework-agnostic transformation passes in `ml_switcheroo_ir.transforms`:

```python
from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalNode,
    eliminate_common_subexpressions,
    eliminate_dead_nodes,
    propagate_shapes_and_constants,
)

# Dead Code Elimination (DCE): preserves explicit outputs and side-effecting ops (e.g. Print, custom_call)
clean_graph = eliminate_dead_nodes(graph)

# Common Subexpression Elimination (CSE): eliminates identical redundant computations
deduped_graph = eliminate_common_subexpressions(clean_graph)

# Shape Propagation & Constant Folding: propagates tensor shapes and folds dead constant subgraphs
optimized_graph = propagate_shapes_and_constants(deduped_graph)
```

### Zero-Hallucination Parameter Translation

Translate parameter and attribute names across frameworks (`torch`, `jax`, `tf`, `stablehlo`, `numpy`) driven by `concept_map.json`:

```python
from ml_switcheroo_ir.translation import ParameterTranslationEngine

engine = ParameterTranslationEngine()

# 1. Translate a single parameter name
# Normalization: PyTorch 'eps' -> TensorFlow 'epsilon'
tf_param = engine.translate_parameter(
    operation="normalization",
    param_name="eps",
    source_framework="torch",
    target_framework="tf",
)
assert tf_param == "epsilon"

# 2. Translate an entire attributes dictionary (strict zero-hallucination policy)
torch_attrs = {"eps": 1e-5, "weight": 1.0}
tf_attrs = engine.translate_attributes(
    operation="normalization",
    attributes=torch_attrs,
    source_framework="torch",
    target_framework="tf",
)
assert tf_attrs == {"epsilon": 1e-5, "gamma": 1.0}
```

### Hardware ISA & GPU Shader Validation

Validate hardware GPU assembly instructions and WebGPU WGSL shaders against architectural constraints:

```python
from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_ir.validator import Validator

validator = Validator()

# 1. AMD RDNA3 / GFX11 Dual-Issue VOPD Pairing Validation
v_add = LogicalNode(id="op1", op_type="V_DUAL_ADD_F32", domain="amd_rdna")
v_mul = LogicalNode(id="op2", op_type="V_DUAL_MUL_F32", domain="amd_rdna")
vopd_errors = validator.validate_vopd_pairing(v_add, v_mul)
assert not vopd_errors

# 2. NVIDIA SASS Scoreboard Latency Hazard Detection
sass_write = LogicalNode(
    id="i1",
    op_type="FFMA",
    domain="nvidia_sass",
    attributes={"dst_reg": "R0", "stall_count": 0},
    outputs=["R0"],
)
sass_read = LogicalNode(
    id="i2",
    op_type="FADD",
    domain="nvidia_sass",
    attributes={"src_regs": ["R0"], "dst_reg": "R1"},
    inputs=["R0"],
)
hazards = validator.validate_sass_scoreboarding([sass_write, sass_read])
assert len(hazards) == 1  # Flags hazard: FFMA has latency 4 but stall_count was 0
```

### Extending Custom Operator Schemas

Register new cutting-edge or domain-specific operations dynamically:

```python
from ml_switcheroo_ir.schema.custom_ops import (
    CustomAttributeSchema,
    CustomOpSchema,
    Registry,
)

registry = Registry()
registry.register(
    CustomOpSchema(
        name="ChunkedPrefillAttention",
        domain="ml.switcheroo.custom",
        inputs=["query", "key", "value", "block_table"],
        outputs=["output"],
        attributes={
            "block_size": CustomAttributeSchema(type="int", required=True, default=16),
            "is_causal": CustomAttributeSchema(
                type="bool", required=False, default=True
            ),
        },
    )
)
```

### Validating Graphs

Ensure graphs satisfy operator schemas across ONNX, Modern Custom Ops, StableHLO, and MLIR:

```python
from ml_switcheroo_ir.validator import ValidationLevel, Validator

validator = Validator()
errors = validator.validate_graph(graph)

for err in errors:
    if err.level == ValidationLevel.ERROR:
        print(f"ERROR on {err.node_id} ({err.attribute}): {err.message}")
    else:
        print(f"WARNING on {err.node_id} ({err.attribute}): {err.message}")
```

### Auditing Graphs with `ml-ecosystem-snapshots`

Audit an IR graph against ground-truth framework snapshots to eliminate hallucinations:

```python
from ml_switcheroo_ir.validator import audit_graph_grounding

# Point to snapshots generated by SamuelMarks/ml-ecosystem-snapshots
report = audit_graph_grounding(graph, snapshots_path="path/to/snapshots.json")

print(f"Total Nodes: {report.total_nodes}")
print(f"Grounded Nodes: {report.grounded_count}")
print(f"Hallucination Score: {report.hallucination_score:.1%}")

if report.diagnostics:
    for err in report.diagnostics:
        print(f"  [Hallucination] Node {err.node_id}: {err.message}")
```

### Programmatic Schema Export & TypeScript Generation

Emit Draft 2020-12 conforming JSON Schemas and TypeScript interface definitions for frontend and playground integrations:

```python
from ml_switcheroo_ir import (
    export_schemas,
    generate_typescript_definitions,
    get_json_schema,
)

# 1. Fetch Draft 2020-12 JSON Schema dictionary
schema = get_json_schema("LogicalGraph")
assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"

# 2. Generate TypeScript interfaces mirroring LogicalGraph and LogicalNode
ts_code = generate_typescript_definitions()
assert "export interface LogicalGraph" in ts_code

# 3. Export all schemas to disk
export_schemas(out_dir="dist/schemas")
```

---

## 2. Interface Protocols for Compilers & Frontends

### Implementing `GraphFrontend`

Ingestion plugins (in `ml-switcheroo` or `zero-pytorch`) parse source code or framework traces into a `LogicalGraph`:

```python
from ml_switcheroo_ir import GraphFrontend, LogicalGraph, LogicalNode


class PyTorchASTFrontend(GraphFrontend):
    """Parses PyTorch nn.Module source code into an IR graph."""

    def parse_to_graph(self, code: str) -> LogicalGraph:
        # Ingest AST and map linear layers to ONNX Gemm
        node = LogicalNode(
            id="fc1",
            op_type="Gemm",
            domain="ai.onnx",
            attributes={"alpha": 1.0, "beta": 1.0, "transB": 1},
            inputs=["input_tensor"],
            outputs=["fc1_out"],
        )
        return LogicalGraph(
            name="ParsedModel", nodes={"fc1": node}, outputs=["fc1_out"]
        )
```

### Implementing `CompilerBackend`

Synthesis plugins (in `ml-switcheroo-compiler` or target code generators) consume a `LogicalGraph`:

```python
from ml_switcheroo_ir import CompilerBackend, LogicalGraph


class WebGPUBackend(CompilerBackend):
    """Compiles LogicalGraph into WGSL shader bindings."""

    def compile(self, graph: LogicalGraph) -> str:
        shader_lines = ["// Generated by Abstract ML Compiler"]
        for node in graph.nodes_list:
            shader_lines.append(
                f"// Op: {node.domain}.{node.op_type} (Node: {node.id})"
            )
        return "\n".join(shader_lines)
```

---

## 3. CLI Reference & Workflows

The `ml-switcheroo-ir` CLI utility provides tools for graph verification, operator search, static compliance scoring, and snapshot generation.

### 1. Graph Validation (`validate`)

Validates a JSON model graph against the canonical schema.

```bash
ml-switcheroo-ir validate model.json
```

**Options:**
- `--strict`: Treat warnings as fatal errors (exit code 1).
- `--custom-ops <path.json>`: Load an external JSON custom operations manifest.

**Example Failure:**
```text
Node linear1:
  [ERROR] transB: Attribute 'transB' expected type 'int', got 'str'
  [ERROR] inputs: Node input 'nonexistent_layer' does not exist.
```

### 2. Snapshot Grounding (`ground`)

Audits an IR graph against ground-truth framework snapshots from `SamuelMarks/ml-ecosystem-snapshots` to detect phantom operators and illegal parameters:

```bash
ml-switcheroo-ir ground model.json --snapshots-dir /path/to/snapshots/
```

**Output:**
```text
Grounding Audit: 42/45 nodes grounded. Hallucination score: 6.7%
  [ERROR] Node attn1 (kind): Ungrounded symbol 'NonExistentAttn' in domain 'ai.onnx'. Symbol not found in framework snapshot.
  [ERROR] Node conv1 (dilation): Ungrounded attribute 'dilation' on 'Conv'. Allowed attributes/params: ['auto_pad', 'dilations', 'group', 'kernel_shape', 'pads', 'strides'].
```

### 3. Compliance & Coverage Auditing (`compliance`)

The `compliance` command performs static AST inspection without executing code or importing heavy frameworks. It evaluates:
1. IR Frontend & Backend compliance (`GraphFrontend`, `CompilerBackend`).
2. Framework Adapter compliance (`register_framework`, `convert`, `definitions`).
3. Dialect coverage percentage against canonical operators.

```bash
ml-switcheroo-ir compliance path/to/my_framework/
```

**Output:**
```text
Compliance Report
=================

FrameworkAdapter Compliance:
Target                  Compliance %    Reqs Met    Status
----------------------  --------------  ----------  --------
my_framework (Dir)      100%            4/4         PASS

DIALECT Compliance (ONNX Superset, 205 total ops):
Target                  Compliance %    Ops Implemented    Status      Missing Ops
----------------------  --------------  -----------------  --------  -------------
my_framework (Dir)      56.6%           116/205            FAIL                 89
```

#### Generating Missing-Symbol Checklists (`-v` / `-m`)

Pass a mapping manifest (such as `torch.json` or `jax.json`) with `--verbose` to generate a markdown checklist of missing APIs:

```bash
ml-switcheroo-ir compliance src/zero_torch/ -m definitions/torch.json -v
```

**Output:**
```text
Verbose Missing Operations Report
=================================

### zero_torch (Directory) (Mapped API targets)
| Implemented   | Framework | Namespace    | Symbol      | FQN                     | Signature                       | Docstring               |
|---------------|-----------|--------------|-------------|-------------------------|---------------------------------|-------------------------|
| [ ]           | torch     | torch.nn     | RMSNorm     | torch.nn.RMSNorm        | (dim, eps=1e-6)                 | Root Mean Square Norm   |
| [ ]           | torch     | torch.nn.fn  | scaled_dot  | torch.nn.functional.sdp | (q, k, v, attn_mask=None, ...)  | Flash/SDP Attention     |
```

This checklist can be directly pasted into an issue, PR, or fed into an automated code-generation prompt.

### 4. Operator Discovery (`list-ops`)

Inspect built-in operator specifications, required parameters, and domains:

```bash
# List all operators in the canonical ai.onnx domain
ml-switcheroo-ir list-ops --domain ai.onnx

# Search for convolution or attention operators
ml-switcheroo-ir list-ops --search "(Conv|Attention)"
```

**Output:**
```text
Op Name                    Domain                Required Args              Optional Args
-------------------------  --------------------  -------------------------  ----------------------------------------------------------------------------------------------
Attention                  ai.onnx                                          is_causal, kv_num_heads, q_num_heads, qk_matmul_output_mode, scale, softcap, softmax_precision
Conv                       ai.onnx                                          auto_pad, dilations, group, kernel_shape, pads, strides
FlashAttention             ml.switcheroo.custom                             causal, scale
LinearAttention            ai.onnx               kv_num_heads, q_num_heads  chunk_size, scale, update_rule
ScaledDotProductAttention  ml.switcheroo.custom                             scale, dropout_p, is_causal
```

### 5. Topological Sorting (`toposort`)

Order and check graph dependencies from JSON:

```bash
ml-switcheroo-ir toposort model.json
```

**Output:**
```text
Topological Order:
 - input_x (Input)
 - norm1 (RMSNorm)
 - proj1 (Gemm)
```

### 6. Backend Interface Verification (`verify-backend`)

Verify that a target compiler module implements the `CompilerBackend` interface:

```bash
ml-switcheroo-ir verify-backend src/compiler/wasm.py WasmBackend
```

**Output:**
```text
Verification Report:
 [PASS] Module loaded
 [PASS] Class found
 [PASS] Inherits CompilerBackend
 [PASS] Has compile() method
 [PASS] compile() signature takes 'graph'

Compliance: 100% (5/5 requirements met)
```

### 7. Exporting GhostRef Snapshots (`dump-snapshot`)

Export all built-in operator schemas to GhostRef v2 JSON snapshot format:

```bash
ml-switcheroo-ir dump-snapshot --output schemas_snapshot.json
```

### 8. Exporting JSON Schemas & TypeScript (`export-schema`)

Export canonical JSON Schema definitions and TypeScript interfaces:

```bash
# Export all JSON schemas to a directory
ml-switcheroo-ir export-schema --out-dir schemas/

# Export TypeScript definitions
ml-switcheroo-ir export-schema --typescript --ts-out types.ts
```

---

## 4. End-to-End Developer Recipes

### Recipe A: Tracking `zero-*` Dialect Implementation Progress

When building pure-Python replicas of major frameworks (e.g. `zero-pytorch` or `zero-jax`):

1. **Initialize your replica repo** without third-party framework dependencies.
2. **Obtain the target snapshot** from `SamuelMarks/ml-ecosystem-snapshots` (e.g., `torch.json`).
3. **Run compliance audits in CI** to track operator coverage progress:
   ```bash
   ml-switcheroo-ir compliance src/zero_torch/ -m definitions/torch.json -v > MISSING_OPS.md
   ```
4. **Implement missing operations** until Dialect Compliance reaches 100%.

### Recipe B: Pre-Compilation Sanitization in CI/CD

Before running expensive compilation or lowering passes in `ml-switcheroo-compiler`:

```bash
# 1. Ensure topological ordering and cycle freedom
ml-switcheroo-ir toposort graph.json > /dev/null

# 2. Strict schema checking (ONNX and Custom Ops)
ml-switcheroo-ir validate graph.json --strict

# 3. Ground against snapshots to prevent LLM hallucinations
ml-switcheroo-ir ground graph.json --snapshots-dir snapshots/
```

### Recipe C: Validating StableHLO & MLIR Lowerings

When translating high-level graphs to compiler dialects:

```python
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.validator import Validator

# Define input operands
lhs_node = LogicalNode(id="lhs", op_type="Input", domain="stablehlo")
rhs_node = LogicalNode(id="rhs", op_type="Input", domain="stablehlo")

# Construct StableHLO dot_general node
dot_node = LogicalNode(
    id="dot",
    op_type="dot_general",
    domain="stablehlo",
    attributes={
        "dot_dimension_numbers": {
            "lhs_contracting_dimensions": [1],
            "rhs_contracting_dimensions": [0],
        }
    },
    inputs=["lhs", "rhs"],
    outputs=["out"],
)

graph = LogicalGraph(nodes={"lhs": lhs_node, "rhs": rhs_node, "dot": dot_node})
validator = Validator()
errors = validator.validate_graph(graph)
assert not errors, f"StableHLO validation failed: {errors}"
```

### Recipe D: Hardware-Specific SASS Scoreboarding & RDNA3 VOPD Validation

When compiling down to GPU assembly instructions:

```python
from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_ir.validator import Validator

validator = Validator()

# Audit SASS instructions for latency pipeline hazards
sass_instructions = [
    LogicalNode(
        id="i1",
        op_type="FFMA",
        domain="nvidia_sass",
        attributes={"dst_reg": "R0", "stall_count": 4},
        outputs=["R0"],
    ),
    LogicalNode(
        id="i2",
        op_type="FADD",
        domain="nvidia_sass",
        attributes={"src_regs": ["R0"], "dst_reg": "R1"},
        inputs=["R0"],
    ),
]
scoreboard_hazards = validator.validate_sass_scoreboarding(sass_instructions)
assert not scoreboard_hazards, f"Hazard detected: {scoreboard_hazards}"
```

### Recipe E: Pre-Lowering Graph Optimization Pipeline

Chain native graph passes before handing the IR to target code synthesis:

```python
from ml_switcheroo_ir import (
    LogicalGraph,
    eliminate_common_subexpressions,
    eliminate_dead_nodes,
    propagate_shapes_and_constants,
)


def optimize_for_lowering(raw_graph: LogicalGraph) -> LogicalGraph:
    """Run canonical dead-code, subexpression, and constant-folding passes."""
    g = eliminate_dead_nodes(raw_graph)
    g = eliminate_common_subexpressions(g)
    g = propagate_shapes_and_constants(g)
    return g
```
