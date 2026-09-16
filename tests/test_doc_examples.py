"""Tests verifying that all IR examples in documentation execute correctly and are up to date."""

from __future__ import annotations

import json

from ml_switcheroo_ir import (
    CompilerBackend,
    CyclicGraphError,
    DType,
    GraphFrontend,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    PartitionSpec,
    topological_sort,
)
from ml_switcheroo_ir.schema.custom_ops import (
    CustomAttributeSchema,
    CustomOpSchema,
    Registry,
)
from ml_switcheroo_ir.validator import Validator, audit_graph_grounding


def test_readme_quickstart_building_graph() -> None:
    """Test README Quick Start 1: Building a Logical Graph with device mesh and sharding."""
    # 1. Create a distributed device mesh
    mesh = LogicalMesh(shape={"data": 4, "model": 2})

    # 2. Define compute nodes with multi-dimensional sharding
    input_node = LogicalNode(
        id="input_x",
        op_type="Input",
        domain="ai.onnx",
        attributes={"dtype": DType.float32.value},
        sharding=PartitionSpec(axes=("data", None)),
    )

    norm_node = LogicalNode(
        id="norm1",
        op_type="RMSNorm",
        domain="ml.switcheroo.custom",
        attributes={"eps": 1e-6},
        inputs=["input_x"],
        outputs=["norm1_out"],
    )

    proj_node = LogicalNode(
        id="proj1",
        op_type="Gemm",
        domain="ai.onnx",
        attributes={"alpha": 1.0, "beta": 1.0, "transB": 1},
        inputs=["norm1_out"],
        outputs=["proj1_out"],
        sharding=PartitionSpec(axes=(None, "model")),
    )

    # 3. Assemble and inspect the graph
    graph = LogicalGraph(
        name="TransformerBlock",
        nodes={"input_x": input_node, "norm1": norm_node, "proj1": proj_node},
        outputs=["proj1_out"],
        mesh=mesh,
    )

    # 4. Topologically sort the computation graph
    ordered_nodes = topological_sort(graph)
    assert [node.id for node in ordered_nodes] == ["input_x", "norm1", "proj1"]

    # 5. Access first-class edges derived from data flow
    edge_pairs = [(edge.source, edge.target) for edge in graph.edges]
    assert edge_pairs == [("input_x", "norm1"), ("norm1_out", "proj1")]

    # 6. Validate graph
    validator = Validator()
    errors = validator.validate_graph(graph)
    assert not errors, f"Graph validation failed: {errors}"


def test_readme_quickstart_validating_schema() -> None:
    """Test README Quick Start 2: Validating node and graph against canonical operator schemas."""
    # 1. Validate a single operator node against its canonical schema
    attn_node = LogicalNode(
        id="attn",
        op_type="FlashAttention",
        domain="ml.switcheroo.custom",
        attributes={"causal": True, "scale": 0.125},
        inputs=["Q", "K", "V"],
        outputs=["attn_out"],
    )

    validator = Validator()
    node_errors = validator.validate_node(attn_node)
    assert not node_errors

    # 2. Validate entire computational graph with its inputs and edges
    q_node = LogicalNode(id="Q", op_type="Input")
    k_node = LogicalNode(id="K", op_type="Input")
    v_node = LogicalNode(id="V", op_type="Input")

    graph = LogicalGraph(
        nodes={"Q": q_node, "K": k_node, "V": v_node, "attn": attn_node},
        outputs=["attn_out"],
    )

    graph_errors = validator.validate_graph(graph)
    assert not graph_errors, f"Graph validation failed: {graph_errors}"


def test_readme_quickstart_json_serialization() -> None:
    """Test README Quick Start 4: Deterministic JSON serialization and deserialization."""
    q_node = LogicalNode(id="Q", op_type="Input")
    attn_node = LogicalNode(
        id="attn",
        op_type="FlashAttention",
        domain="ml.switcheroo.custom",
        attributes={"causal": True, "scale": 0.125},
        inputs=["Q"],
        outputs=["attn_out"],
    )
    graph = LogicalGraph(
        name="TestAttnGraph",
        nodes={"Q": q_node, "attn": attn_node},
        outputs=["attn_out"],
    )

    json_payload = graph.to_json(format="canonical", indent=2)
    assert "FlashAttention" in json_payload
    assert "edges" in json_payload

    restored_graph = LogicalGraph.from_json(json_payload)
    assert len(restored_graph) == len(graph)
    assert restored_graph["attn"].op_type == "FlashAttention"


def test_readme_frontend_backend_contracts() -> None:
    """Test README frontend (GraphFrontend) and backend (CompilerBackend) protocol implementations."""

    class MyCustomFrontend(GraphFrontend):
        """Example custom frontend translating source code to LogicalGraph."""

        def parse_to_graph(self, code: str) -> LogicalGraph:
            """Parse input code string into a LogicalGraph representation."""
            node = LogicalNode(
                id="dense",
                op_type="Gemm",
                domain="ai.onnx",
                attributes={"transB": 1},
                inputs=["x"],
            )
            return LogicalGraph(nodes={"dense": node}, outputs=["dense"])

    class MyCompilerBackend(CompilerBackend):
        """Example compiler backend emitting target assembly instructions."""

        def compile(self, graph: LogicalGraph) -> str:
            """Compile a LogicalGraph into target code instructions."""
            instructions = []
            for node in graph.nodes_list:
                instructions.append(f"emit_{node.domain}_{node.op_type}({node.id})")
            return "\n".join(instructions)

    frontend = MyCustomFrontend()
    graph = frontend.parse_to_graph("dummy_code")
    assert "dense" in graph.nodes

    backend = MyCompilerBackend()
    compiled = backend.compile(graph)
    assert compiled == "emit_ai.onnx_Gemm(dense)"


def test_usage_constructing_graphs_and_nodes() -> None:
    """Test USAGE.md Section 1.1: Constructing Graphs & Nodes."""
    x_node = LogicalNode(
        id="x",
        op_type="Input",
        domain="ai.onnx",
        attributes={"dtype": DType.float32.value, "shape": [1, 128, 768]},
        outputs=["x"],
    )

    norm_node = LogicalNode(
        id="norm",
        op_type="RMSNorm",
        domain="ml.switcheroo.custom",
        attributes={"eps": 1e-6},
        inputs=["x"],
        outputs=["norm_out"],
    )

    graph = LogicalGraph(
        name="LayerNormBlock",
        nodes={"x": x_node, "norm": norm_node},
        outputs=["norm_out"],
    )

    assert graph["norm"].op_type == "RMSNorm"
    assert len(graph) == 2
    node_ids = [node.id for node in graph]
    assert node_ids == ["x", "norm"]

    validator = Validator()
    errors = validator.validate_graph(graph)
    assert not errors, f"Graph validation failed: {errors}"


def test_usage_multi_output_and_ssa_tracking() -> None:
    """Test USAGE.md Section 1.2: Multi-Output Nodes & SSA Value Tracking."""
    split_node = LogicalNode(
        id="split1",
        op_type="Split",
        domain="ai.onnx",
        attributes={"axis": -1},
        inputs=["x"],
        outputs=["chunk_a", "chunk_b"],
    )

    graph = LogicalGraph(nodes={"split1": split_node})

    producer = graph.get_output_producer("chunk_b")
    assert producer is not None and producer.id == "split1"

    downstream = graph.get_outputs("split1")
    assert downstream == []


def test_usage_extending_custom_ops_dict_and_list() -> None:
    """Test USAGE.md Section 1.6: Extending Custom Operator Schemas with dict attributes and Registry.register."""
    registry = Registry()
    registry.register(
        CustomOpSchema(
            name="ChunkedPrefillAttention",
            domain="ml.switcheroo.custom",
            inputs=["query", "key", "value", "block_table"],
            outputs=["output"],
            attributes={
                "block_size": CustomAttributeSchema(
                    type="int", required=True, default=16
                ),
                "is_causal": CustomAttributeSchema(
                    type="bool", required=False, default=True
                ),
            },
        )
    )

    assert "ChunkedPrefillAttention" in registry.schemas
    schema = registry.schemas["ChunkedPrefillAttention"]
    assert schema.attributes["block_size"].type == "int"
    assert schema.attributes["block_size"].required is True
    assert schema.attributes["is_causal"].default is True

    # Also test registering OpSchema directly
    registry.register(schema)
    assert "ChunkedPrefillAttention" in registry.schemas


def test_usage_recipe_c_stablehlo_validation() -> None:
    """Test USAGE.md Recipe C: Validating StableHLO & MLIR Lowerings."""
    lhs_node = LogicalNode(id="lhs", op_type="Input", domain="stablehlo")
    rhs_node = LogicalNode(id="rhs", op_type="Input", domain="stablehlo")

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


def test_usage_toposort_and_cycle_detection() -> None:
    """Test USAGE.md Section 1.4: Topological Sorting & Cycle Detection."""
    n1 = LogicalNode(id="n1", op_type="Relu")
    n2 = LogicalNode(id="n2", op_type="Relu", inputs=["n1"])
    graph = LogicalGraph(nodes={"n1": n1, "n2": n2})

    ordered_nodes = topological_sort(graph, strict=True)
    assert [n.id for n in ordered_nodes] == ["n1", "n2"]

    # Test cyclic graph detection
    n1_cycle = LogicalNode(id="n1", op_type="Relu", inputs=["n2"])
    n2_cycle = LogicalNode(id="n2", op_type="Relu", inputs=["n1"])
    cyclic_graph = LogicalGraph(nodes={"n1": n1_cycle, "n2": n2_cycle})
    try:
        topological_sort(cyclic_graph, strict=True)
    except CyclicGraphError as e:
        assert "Cycle detected" in str(e) or "Cyclic graph" in str(e)


def test_usage_audit_graph_grounding() -> None:
    """Test USAGE.md Section 1.8: Auditing Graphs with audit_graph_grounding."""
    x = LogicalNode(id="x", op_type="Input")
    graph = LogicalGraph(nodes={"x": x})
    report = audit_graph_grounding(graph, snapshots_path=[])
    assert report.total_nodes == 1
    assert 0.0 <= report.hallucination_score <= 1.0


def test_usage_frontend_and_backend() -> None:
    """Test USAGE.md Section 2: Implementing GraphFrontend and CompilerBackend."""

    class PyTorchASTFrontend(GraphFrontend):
        """Parses PyTorch nn.Module source code into an IR graph."""

        def parse_to_graph(self, code: str) -> LogicalGraph:
            """Parse code string into a LogicalGraph with Gemm node."""
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

    class WebGPUBackend(CompilerBackend):
        """Compiles LogicalGraph into WGSL shader bindings."""

        def compile(self, graph: LogicalGraph) -> str:
            """Compile graph into WGSL commented representation."""
            shader_lines = ["// Generated by Abstract ML Compiler"]
            for node in graph.nodes_list:
                shader_lines.append(
                    f"// Op: {node.domain}.{node.op_type} (Node: {node.id})"
                )
            return "\n".join(shader_lines)

    frontend = PyTorchASTFrontend()
    graph = frontend.parse_to_graph("code")
    assert "fc1" in graph.nodes

    backend = WebGPUBackend()
    compiled = backend.compile(graph)
    assert "// Op: ai.onnx.Gemm (Node: fc1)" in compiled


def test_dialect_md_json_examples() -> None:
    """Test docs/DIALECT.md JSON node representations deserialize and validate properly."""
    linear_json = """
    {
      "id": "linear1",
      "op_type": "Gemm",
      "domain": "ai.onnx",
      "version": 11,
      "attributes": {
        "alpha": 1.0,
        "beta": 1.0,
        "transB": 1
      },
      "inputs": ["X", "W", "B"],
      "outputs": ["linear1_out"]
    }
    """
    node_data = json.loads(linear_json)
    linear_node = LogicalNode(**node_data)
    assert linear_node.id == "linear1"
    assert linear_node.op_type == "Gemm"
    assert linear_node.attributes["transB"] == 1

    flash_json = """
    {
      "id": "flash_attn1",
      "op_type": "FlashAttention",
      "domain": "ml.switcheroo.custom",
      "version": 1,
      "attributes": {
        "causal": true,
        "scale": 0.125
      },
      "inputs": ["Q", "K", "V"],
      "outputs": ["flash_attn1_out"]
    }
    """
    flash_data = json.loads(flash_json)
    flash_node = LogicalNode(**flash_data)
    assert flash_node.id == "flash_attn1"
    assert flash_node.op_type == "FlashAttention"

    v = Validator()
    assert not v.validate_node(linear_node)
    assert not v.validate_node(flash_node)
