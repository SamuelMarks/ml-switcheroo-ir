"""Property-based Hypothesis fuzzing for lossless IR serialization, compression, and Python codegen."""

from __future__ import annotations

import io
import runpy
import tempfile
from pathlib import Path

import hypothesis.strategies as st
from hypothesis import given, settings

import ml_switcheroo_ir as sw_ir
from ml_switcheroo_ir import (
    DependencyConfig,
    DType,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    MeshMappingConfig,
    MicrobatchSplittingConfig,
    NoTangent,
    PartitionSpec,
    PipelineTopologyConfig,
    StageCommunicationConfig,
    SymBinaryOp,
    SymConst,
    SymInt,
    SymNode,
    SymVar,
    TensorShape,
    TensorSpec,
    WebRTCPeerConfig,
    WebRTCSignalingTopology,
    ZeroTangent,
    export_to_python,
)
from ml_switcheroo_ir.types import AttributeValue, DimensionType, InitializerValue

valid_identifiers: st.SearchStrategy[str] = st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,12}", fullmatch=True)

sym_leaf_strategy: st.SearchStrategy[SymNode] = st.one_of(
    valid_identifiers.map(SymVar),
    st.integers(min_value=1, max_value=32).map(SymConst),
)

sym_node_strategy: st.SearchStrategy[SymNode] = st.one_of(
    sym_leaf_strategy,
    st.builds(
        SymBinaryOp,
        op=st.sampled_from(["+", "-", "*"]),
        left=sym_leaf_strategy,
        right=sym_leaf_strategy,
    ),
    st.builds(
        SymBinaryOp,
        op=st.sampled_from(["//", "%"]),
        left=sym_leaf_strategy,
        right=st.integers(min_value=1, max_value=8).map(SymConst),
    ),
    st.builds(
        SymBinaryOp,
        op=st.just("**"),
        left=sym_leaf_strategy,
        right=st.integers(min_value=0, max_value=3).map(SymConst),
    ),
)

sym_int_strategy: st.SearchStrategy[SymInt] = sym_node_strategy.map(SymInt)

dimension_strategy: st.SearchStrategy[DimensionType] = st.one_of(
    st.integers(min_value=1, max_value=64),
    valid_identifiers,
    sym_node_strategy,
    sym_int_strategy,
)

dtype_strategy: st.SearchStrategy[DType] = st.sampled_from(list(DType))

tensor_shape_strategy: st.SearchStrategy[TensorShape] = st.lists(dimension_strategy, min_size=0, max_size=3).map(
    lambda dims: TensorShape(tuple(dims))
)

tensor_spec_strategy: st.SearchStrategy[TensorSpec] = st.builds(
    TensorSpec,
    shape=st.lists(dimension_strategy, min_size=0, max_size=3),
    dtype=dtype_strategy,
    sparsity=st.one_of(st.none(), st.sampled_from(["dense", "2:4"])),
)

rich_attr_leaf: st.SearchStrategy[AttributeValue] = st.one_of(
    st.integers(min_value=-100, max_value=100),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=0, max_size=12),
    st.booleans(),
    st.binary(min_size=0, max_size=16),
    st.none(),
    dtype_strategy,
    sym_node_strategy,
    sym_int_strategy,
    tensor_shape_strategy,
    tensor_spec_strategy,
)

attribute_values: st.SearchStrategy[AttributeValue] = st.recursive(
    rich_attr_leaf,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(valid_identifiers, children, max_size=3),
    ),
    max_leaves=8,
)

initializer_values: st.SearchStrategy[InitializerValue] = st.one_of(
    st.integers(min_value=-50, max_value=50),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.booleans(),
    st.text(alphabet="abcdef0123", min_size=0, max_size=10),
    st.binary(min_size=0, max_size=16),
    st.lists(st.integers(min_value=-10, max_value=10), max_size=4),
)

partition_specs: st.SearchStrategy[PartitionSpec | None] = st.one_of(
    st.none(),
    st.lists(
        st.one_of(st.none(), valid_identifiers, st.integers(min_value=0, max_value=3)),
        min_size=1,
        max_size=3,
    ).map(lambda axes: PartitionSpec(axes=tuple(axes))),
)


@st.composite
def nested_subgraph_strategy(draw: st.DrawFn) -> LogicalGraph:
    """Generate a compact nested ``LogicalGraph`` subgraph for control-flow nodes.

    Args:
        draw (st.DrawFn): Hypothesis draw callable.

    Returns:
        LogicalGraph: Valid nested ``LogicalGraph``.
    """
    sub_id = draw(valid_identifiers)
    sub_node = LogicalNode(
        id=sub_id,
        op_type="Relu",
        attributes=draw(st.dictionaries(valid_identifiers, rich_attr_leaf, max_size=2)),
        shape_metadata=tuple(draw(st.lists(dimension_strategy, max_size=2))),
        dtype=draw(st.one_of(st.none(), dtype_strategy)),
    )
    return LogicalGraph(
        name=draw(valid_identifiers),
        nodes={sub_id: sub_node},
        outputs=[sub_id],
        attributes=draw(st.dictionaries(valid_identifiers, rich_attr_leaf, max_size=1)),
    )


@st.composite
def rich_graph_strategy(draw: st.DrawFn) -> LogicalGraph:
    """Generate arbitrary valid ``LogicalGraph`` instances exercising all rich IR types and topologies.

    Args:
        draw (st.DrawFn): Hypothesis draw callable.

    Returns:
        LogicalGraph: Generated ``LogicalGraph`` instance.
    """
    graph_name = draw(valid_identifiers)
    num_nodes = draw(st.integers(min_value=1, max_value=4))
    node_ids = [f"node_{i}" for i in range(num_nodes)]
    nodes: dict[str, LogicalNode] = {}
    available_ssa: list[str] = []

    for idx, nid in enumerate(node_ids):
        kind = draw(st.sampled_from(["standard", "multi_out", "zero_tangent", "no_tangent"]))
        inps = draw(st.lists(st.sampled_from(available_ssa), max_size=2)) if available_ssa else []
        attrs = draw(st.dictionaries(valid_identifiers, attribute_values, max_size=2))
        shape_meta = draw(st.one_of(st.none(), st.lists(dimension_strategy, max_size=3).map(tuple)))
        sharding = draw(partition_specs)
        dt = draw(st.one_of(st.none(), dtype_strategy))
        out_specs = draw(st.lists(tensor_spec_strategy, max_size=2))
        subgraphs: dict[str, LogicalGraph | AttributeValue] = {}
        if idx == 0 and draw(st.booleans()):
            subgraphs["body"] = draw(nested_subgraph_strategy())

        if kind == "zero_tangent":
            node: LogicalNode = ZeroTangent(
                id=nid,
                shape=shape_meta,
                dtype=dt or DType.float32,
                attributes=attrs,
                inputs=inps,
                sharding=sharding,
                output_specs=out_specs,
                subgraphs=subgraphs,
            )
        elif kind == "no_tangent":
            node = NoTangent(
                id=nid,
                shape_metadata=shape_meta,
                dtype=dt,
                attributes=attrs,
                inputs=inps,
                sharding=sharding,
                output_specs=out_specs,
                subgraphs=subgraphs,
            )
        elif kind == "multi_out":
            outs = [f"{nid}_o0", f"{nid}_o1"]
            node = LogicalNode(
                id=nid,
                op_type="Split",
                attributes=attrs,
                inputs=inps,
                outputs=outs,
                shape_metadata=shape_meta,
                sharding=sharding,
                dtype=dt,
                output_specs=out_specs,
                subgraphs=subgraphs,
            )
        else:
            node = LogicalNode(
                id=nid,
                op_type=draw(st.sampled_from(["Add", "Relu", "MatMul", "Conv"])),
                attributes=attrs,
                inputs=inps,
                shape_metadata=shape_meta,
                sharding=sharding,
                dtype=dt,
                output_specs=out_specs,
                subgraphs=subgraphs,
            )
        nodes[nid] = node
        available_ssa.extend(node.outputs)

    webrtc = None
    if draw(st.booleans()):
        webrtc = WebRTCSignalingTopology(
            signaling_url="wss://coord.local",
            peers=[WebRTCPeerConfig(peer_id="p0"), WebRTCPeerConfig(peer_id="p1")],
            connectivity_matrix=[[1, 1], [1, 1]],
            chunk_size_bytes=1024,
            low_watermark_bytes=256,
            high_watermark_bytes=2048,
        )
    mesh = LogicalMesh(shape={"data": 2, "model": 2}, webrtc_topology=webrtc) if draw(st.booleans()) else None
    pipe = (
        PipelineTopologyConfig(
            microbatch_splitting=MicrobatchSplittingConfig(num_microbatches=2),
            mesh_mapping=MeshMappingConfig(devices_per_stage=1),
            stage_communication=StageCommunicationConfig(protocol="nccl"),
            dependencies=[DependencyConfig(source_stage="s0", target_stage="s1")],
        )
        if draw(st.booleans())
        else None
    )
    input_specs = draw(st.dictionaries(valid_identifiers, tensor_spec_strategy, max_size=2))
    initializers = draw(st.dictionaries(valid_identifiers, initializer_values, max_size=2))
    graph_attrs = draw(st.dictionaries(valid_identifiers, attribute_values, max_size=2))
    loop_opts = draw(st.one_of(st.none(), st.dictionaries(valid_identifiers, rich_attr_leaf, max_size=2)))
    outputs = draw(st.lists(st.sampled_from(available_ssa), min_size=1, max_size=2, unique=True))

    return LogicalGraph(
        name=graph_name,
        nodes=nodes,
        inputs=list(input_specs.keys()),
        input_specs=input_specs,
        outputs=outputs,
        initializers=initializers,
        mesh=mesh,
        pipeline_topology=pipe,
        attributes=graph_attrs,
        loop_options=loop_opts,
    )


def _assert_graphs_losslessly_equal(expected: LogicalGraph, actual: LogicalGraph) -> None:
    """Assert exact structural and runtime-type parity between ``expected`` and ``actual`` graphs.

    Args:
        expected (LogicalGraph): Original graph.
        actual (LogicalGraph): Reconstructed graph.
    """
    assert actual.name == expected.name
    assert actual.inputs == expected.inputs
    assert actual.input_specs == expected.input_specs
    assert actual.outputs == expected.outputs
    assert actual.initializers == expected.initializers
    assert actual.mesh == expected.mesh
    assert actual.pipeline_topology == expected.pipeline_topology
    assert actual.attributes == expected.attributes
    assert actual.loop_options == expected.loop_options
    assert list(actual.nodes.keys()) == list(expected.nodes.keys())
    for nid, exp_node in expected.nodes.items():
        act_node = actual.nodes[nid]
        assert type(act_node) is type(exp_node)
        assert act_node == exp_node


@given(graph=rich_graph_strategy())
@settings(max_examples=12, deadline=None)
def test_fuzz_dual_serialization_and_codegen_roundtrip(graph: LogicalGraph) -> None:
    """Fuzz lossless roundtrips across dict, JSON, stream, .gz, .zst, and export_to_python.

    Args:
        graph (LogicalGraph): Randomly generated rich LogicalGraph.
    """
    # 1. Dictionary & JSON string roundtrip
    from_dict_g = LogicalGraph.from_dict(graph.to_dict())
    _assert_graphs_losslessly_equal(graph, from_dict_g)

    json_str = graph.to_json()
    from_json_g = LogicalGraph.from_json(json_str)
    _assert_graphs_losslessly_equal(graph, from_json_g)
    assert from_json_g.to_json() == json_str

    # 2. Streaming roundtrip
    stream_buf = io.StringIO()
    graph.to_stream(stream_buf)
    stream_buf.seek(0)
    from_stream_g = LogicalGraph.from_stream(stream_buf)
    _assert_graphs_losslessly_equal(graph, from_stream_g)

    # 3. Compressed .gz and .zst roundtrip + export_to_python execution roundtrip
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        gz_path = tmp / "fuzzed.json.gz"
        zst_path = tmp / "fuzzed.json.zst"
        py_path = tmp / "fuzzed_builder.py"

        graph.to_file(gz_path)
        _assert_graphs_losslessly_equal(graph, LogicalGraph.from_file(gz_path))

        graph.to_file(zst_path)
        _assert_graphs_losslessly_equal(graph, LogicalGraph.from_file(zst_path))

        py_code = export_to_python(graph, function_name="build_fuzzed")
        py_path.write_text(py_code, encoding="utf-8")
        ns = runpy.run_path(str(py_path))
        built_graph = ns["build_fuzzed"]()
        assert isinstance(built_graph, sw_ir.LogicalGraph)
        _assert_graphs_losslessly_equal(graph, built_graph)
