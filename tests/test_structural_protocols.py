"""Comprehensive unit tests for Tier-1 structural protocols, schema alignment, and graph extensions."""

from __future__ import annotations

import importlib.resources
import json
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import pydantic
import pytest

import ml_switcheroo_ir as sw_ir
from ml_switcheroo_ir import (
    MLIR_REGISTRY,
    ArrayBufferProtocol,
    AttributeValue,
    CustomRuleHandleProtocol,
    DeviceLikeProtocol,
    DimensionType,
    DType,
    DTypeLikeProtocol,
    ExtendedGhostRef,
    GhostOverload,
    GhostParam,
    InitializerValue,
    IRGraph,
    IRNode,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    LoopOptionsProtocol,
    MeshLikeProtocol,
    NoTangent,
    PartitionSpec,
    SubgraphLikeProtocol,
    SymbolicConstraintTracker,
    SymbolicIntWrapperProtocol,
    SymbolicNodeProtocol,
    SymbolicSolver,
    SymInt,
    SymNode,
    SymVar,
    TensorShape,
    TensorSpec,
    ZeroTangent,
    __version__,
    eliminate_common_subexpressions,
    eliminate_dead_nodes,
    export_to_python,
    get_catalog_path,
    get_json_schema,
    is_dynamic_dimension,
    is_symbolic_dimension,
    normalize_dimension_tuple,
    normalize_dtype,
    normalize_optional_dtype,
    topological_sort,
)
from ml_switcheroo_ir.schema import GhostRef, ParameterKind
from ml_switcheroo_ir.validation.distributed import (
    estimate_graph_communication_volume,
    validate_sharding_node,
    validate_sharding_propagation_graph,
)
from ml_switcheroo_ir.validator import ValidationLevel, Validator


class DummyCompilerDType(Enum):
    """Standalone Enum structurally matching ``DTypeLikeProtocol``.

    Attributes:
        FLOAT32 (str): 32-bit IEEE floating-point member.
        BFLOAT16 (str): Brain 16-bit floating-point member.
    """

    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"


class BadIntDTypeEnum(Enum):
    """Standalone Enum with non-string value for negative ``normalize_dtype`` coverage.

    Attributes:
        INVALID (int): Non-string integer enum value.
    """

    INVALID = 99


@dataclass(frozen=True)
class DummyDevice:
    """Standalone device descriptor structurally matching ``DeviceLikeProtocol``.

    Attributes:
        type (str): Target hardware backend name.
        index (int | None): Optional device ordinal index.
    """

    type: str
    index: int | None = None


class BadBoolDevice:
    """Malformed device with boolean index for negative validation coverage."""

    @property
    def type(self) -> str:
        """Return device type.

        Returns:
            str: Backend string.
        """
        return "cuda"

    @property
    def index(self) -> int | None:
        """Return invalid boolean index.

        Returns:
            int | None: Boolean masquerading as int.
        """
        return True


class DummySymNodeNamed:
    """Standalone symbolic node with custom ``__str__`` matching ``SymbolicNodeProtocol``."""

    def __init__(self, expr: str) -> None:
        """Initialize DummySymNodeNamed.

        Args:
            expr (str): Symbolic variable name or expression.
        """
        self._expr = expr

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate variable lookup from ``env``.

        Args:
            env (dict[str, int]): Concrete symbol bindings.

        Returns:
            int: Concrete dimension size.
        """
        return env.get(self._expr, 1) * 2 if "*" in self._expr else env[self._expr]

    def __str__(self) -> str:
        """Format symbolic expression string.

        Returns:
            str: Expression string.
        """
        return self._expr


class DummySymNodeAnon:
    """Standalone symbolic node without ``__str__`` (exercises deterministic address stripping)."""

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate symbolic node using ``'anon'`` key or fallback 7.

        Args:
            env (dict[str, int]): Concrete symbol bindings.

        Returns:
            int: Evaluated dimension value.
        """
        return env.get("anon", 7)


class DummySymIntWrapper:
    """Standalone symbolic integer wrapper matching ``SymbolicIntWrapperProtocol``."""

    def __init__(self, node: SymbolicNodeProtocol) -> None:
        """Initialize DummySymIntWrapper.

        Args:
            node (SymbolicNodeProtocol): Underlying symbolic expression node.
        """
        self._node = node

    @property
    def node(self) -> SymbolicNodeProtocol:
        """Return wrapped symbolic expression node.

        Returns:
            SymbolicNodeProtocol: Underlying symbolic node.
        """
        return self._node


class BadSymIntWrapper:
    """Malformed wrapper whose ``node`` does not implement ``SymbolicNodeProtocol``.

    Attributes:
        node (int): Non-protocol integer attribute.
    """

    node: int = 123


@dataclass(frozen=True)
class DummyCustomRuleHandle:
    """Standalone callback handle matching ``CustomRuleHandleProtocol``.

    Attributes:
        handle_id (str): Unique rule handle identifier.
        kind (str): Custom rule category.
    """

    handle_id: str
    kind: str


@dataclass(frozen=True)
class DummyLoopOptions:
    """Standalone loop options config matching ``LoopOptionsProtocol``.

    Attributes:
        parallel_iterations (int | None): Parallel loop trip limit.
        swap_memory (bool | None): Host-device memory swap flag.
        maximum_iterations (int | None): Upper bound on iterations.
    """

    parallel_iterations: int | None = 16
    swap_memory: bool | None = True
    maximum_iterations: int | None = 100


class BadLoopOptions:
    """Malformed loop options with boolean ``parallel_iterations``.

    Attributes:
        parallel_iterations (int | None): Invalid boolean parallel iteration count.
        swap_memory (bool | None): Memory swap flag.
        maximum_iterations (int | None): Maximum iterations.
    """

    parallel_iterations: int | None = True
    swap_memory: bool | None = False
    maximum_iterations: int | None = 10


@dataclass(frozen=True)
class DummyDeviceMesh:
    """Standalone device mesh matching ``MeshLikeProtocol``.

    Attributes:
        axis_names (tuple[str, ...]): Ordered mesh axis names.
    """

    axis_names: tuple[str, ...]


class BadStringAxesMesh:
    """Malformed mesh whose ``axis_names`` is a bare string rather than axis tuple/list.

    Attributes:
        axis_names (Sequence[str]): Bare string posing as sequence of strings.
    """

    axis_names: Sequence[str] = "bad_axes_str"


class DummyArrayBuffer:
    """Standalone host tensor buffer matching ``ArrayBufferProtocol``."""

    def __init__(self, shape: tuple[int, ...], payload: bytes) -> None:
        """Initialize DummyArrayBuffer.

        Args:
            shape (tuple[int, ...]): Concrete integer shape tuple.
            payload (bytes): Raw contiguous byte buffer.
        """
        self._shape = shape
        self._payload = payload

    @property
    def shape(self) -> tuple[int, ...]:
        """Return concrete shape tuple.

        Returns:
            tuple[int, ...]: Buffer shape.
        """
        return self._shape

    def tobytes(self) -> bytes:
        """Return raw bytes of buffer.

        Returns:
            bytes: Buffer byte contents.
        """
        return self._payload

    def __eq__(self, other: object) -> bool:
        """Compare buffer equality by shape and byte payload.

        Args:
            other (object): Object to compare against.

        Returns:
            bool: True if ``other`` is a DummyArrayBuffer with identical shape and bytes.
        """
        if not isinstance(other, DummyArrayBuffer):
            return False
        return self._shape == other._shape and self._payload == other._payload


class BadBoolShapeBuffer:
    """Malformed array buffer containing a boolean dimension in ``shape``."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return malformed shape containing boolean entry.

        Returns:
            tuple[int, ...]: Tuple with boolean value.
        """
        return (True, 4)

    def tobytes(self) -> bytes:
        """Return raw bytes.

        Returns:
            bytes: Raw bytes.
        """
        return b"\x00\x01\x02\x03"


class DummySubgraphContainer:
    """Standalone subgraph container matching ``SubgraphLikeProtocol``."""

    def __init__(self, nodes: Sequence[LogicalNode]) -> None:
        """Initialize DummySubgraphContainer.

        Args:
            nodes (Sequence[LogicalNode]): Contained logical nodes.
        """
        self._nodes = tuple(nodes)

    @property
    def nodes(self) -> Sequence[LogicalNode]:
        """Return contained logical nodes.

        Returns:
            Sequence[LogicalNode]: Sequence of LogicalNodes.
        """
        return self._nodes


def test_runtime_checkable_protocols_and_aliases() -> None:
    """Verify @runtime_checkable isinstance checks and IRNode/IRGraph aliases."""
    assert IRNode is LogicalNode
    assert IRGraph is LogicalGraph

    dt = DummyCompilerDType.FLOAT32
    dev = DummyDevice("cuda", 0)
    sym_n = DummySymNodeNamed("B")
    sym_w = DummySymIntWrapper(sym_n)
    rule = DummyCustomRuleHandle("rule_42", "vjp")
    loop = DummyLoopOptions()
    mesh = DummyDeviceMesh(("data", "model"))
    buf = DummyArrayBuffer((2, 2), b"\x01\x02\x03\x04")
    sub = DummySubgraphContainer([LogicalNode(id="s1", op_type="Relu")])

    assert isinstance(dt, DTypeLikeProtocol)
    assert isinstance(dev, DeviceLikeProtocol)
    assert isinstance(sym_n, SymbolicNodeProtocol)
    assert isinstance(sym_w, SymbolicIntWrapperProtocol)
    assert isinstance(rule, CustomRuleHandleProtocol)
    assert isinstance(loop, LoopOptionsProtocol)
    assert isinstance(mesh, MeshLikeProtocol)
    assert isinstance(buf, ArrayBufferProtocol)
    assert isinstance(sub, SubgraphLikeProtocol)
    assert isinstance(LogicalGraph(name="Sub"), SubgraphLikeProtocol)


def test_symbolic_dimension_protocols_in_shapes_and_specs() -> None:
    """Test SymbolicNodeProtocol, SymbolicIntWrapperProtocol, and DTypeLikeProtocol in TensorShape and TensorSpec."""
    sym_b = DummySymNodeNamed("B")
    sym_anon = DummySymNodeAnon()
    sym_wrap = DummySymIntWrapper(DummySymNodeNamed("T"))

    assert is_symbolic_dimension(sym_b)
    assert is_symbolic_dimension(sym_anon)
    assert is_symbolic_dimension(sym_wrap)
    assert not is_symbolic_dimension(8)
    assert is_dynamic_dimension(sym_b)
    assert is_dynamic_dimension(sym_wrap)
    assert is_dynamic_dimension(-1)
    assert not is_dynamic_dimension(16)

    norm = normalize_dimension_tuple([sym_b, sym_wrap, 64])
    assert len(norm) == 3
    assert normalize_dimension_tuple(sym_b) == (sym_b,)

    shape = TensorShape((sym_b, sym_wrap, sym_anon, "8", SymInt(4)))
    assert shape.is_dynamic
    assert shape.evaluate({"B": 3, "T": 5, "anon": 9}) == (3, 5, 9, 8, 4)

    static_s = TensorShape((2, "4", SymInt(8)))
    assert static_s.static_shape == (2, 4, 8)
    assert static_s.model_dump(mode="json") == {"dims": [2, "4", "8"]}

    bcast = TensorShape((sym_b, 1)).broadcast_with((1, sym_wrap))
    assert bcast.evaluate({"B": 4, "T": 6}) == (4, 6)
    assert SymbolicSolver.is_consistent(sym_b, SymVar("B"))

    tracker = SymbolicConstraintTracker()
    tracker.add_constraint(sym_b, 12)
    tracker.add_constraint(sym_wrap, 24)
    assert tracker.unify(sym_b, 12) == 12
    assert SymNode.to_node(sym_wrap).evaluate({"T": 11}) == 11
    assert SymNode.to_node(sym_anon).evaluate({"DummySymNodeAnon": 13}) == 13

    spec = TensorSpec(shape=(sym_b, 32), dtype=DummyCompilerDType.BFLOAT16)
    assert spec.dtype is DType.bfloat16
    evaluated_spec = spec.evaluate({"B": 2})
    assert evaluated_spec.shape == (2, 32)
    assert evaluated_spec.dtype is DType.bfloat16

    spec_validated = TensorSpec.model_validate({"shape": [sym_b, "4"], "dtype": DummyCompilerDType.FLOAT32})
    assert spec_validated.dtype is DType.float32
    assert spec_validated.model_dump(mode="json")["shape"] == ["B", "4"]
    assert spec_validated.model_dump(mode="python")["shape"][0] is sym_b

    assert normalize_dtype(DType.int32) is DType.int32
    assert normalize_dtype("int64") is DType.int64
    assert normalize_dtype(DummyCompilerDType.FLOAT32) is DType.float32
    assert normalize_optional_dtype(None) is None
    assert normalize_optional_dtype(DummyCompilerDType.BFLOAT16) is DType.bfloat16


def test_sentinels_with_protocols_and_strides() -> None:
    """Test ZeroTangent and NoTangent with DTypeLikeProtocol, symbolic protocols, and strides."""
    sym_b = DummySymNodeNamed("B")
    sym_wrap = DummySymIntWrapper(DummySymNodeNamed("D"))

    zt = ZeroTangent(
        id="zt_proto",
        shape=(sym_b, sym_wrap),
        dtype=DummyCompilerDType.BFLOAT16,
        device=DummyDevice("cuda", 1),
        source_ast_ref=101,
        stream=2,
        strides=(16, 1),
    )
    assert zt.dtype is DType.bfloat16
    assert zt.strides == (16, 1)
    assert zt.source_ast_ref == 101
    assert zt.stream == 2

    nt = NoTangent(
        id="nt_proto",
        dtype="float16",
        shape_metadata=(sym_b,),
        strides=[1],
    )
    assert nt.dtype is DType.float16
    assert nt.strides == (1,)

    g = LogicalGraph(name="SentinelProtoGraph", nodes=[zt, nt])
    restored = LogicalGraph.from_json(g.to_json())
    assert isinstance(restored.nodes["zt_proto"], ZeroTangent)
    assert isinstance(restored.nodes["nt_proto"], NoTangent)
    assert restored.nodes["zt_proto"].strides == (16, 1)
    assert restored.nodes["zt_proto"].source_ast_ref == 101
    assert restored.nodes["zt_proto"].stream == 2
    assert restored.nodes["zt_proto"].device == "cuda:1"


def _build_rich_protocol_node() -> tuple[LogicalNode, DummyDevice, DummyCustomRuleHandle, DummySymIntWrapper]:
    """Construct a LogicalNode populated with all Tier-1 structural protocols for node tests.

    Returns:
        tuple[LogicalNode, DummyDevice, DummyCustomRuleHandle, DummySymIntWrapper]:
            Constructed node and protocol fixture instances.
    """
    sym_b = DummySymNodeNamed("B")
    sym_anon = DummySymNodeAnon()
    sym_wrap = DummySymIntWrapper(sym_b)
    dev_idx = DummyDevice("cuda", 0)
    dev_noindex = DummyDevice("cpu", None)
    rule = DummyCustomRuleHandle("custom_bwd_1", "bwd")
    loop = DummyLoopOptions(parallel_iterations=8, swap_memory=False, maximum_iterations=None)
    buf = DummyArrayBuffer((2, 3), b"\x01\x02\x03\x04\x05\x06")
    nested_node = LogicalNode(id="inner_n", op_type="Relu")
    nested_graph = LogicalGraph(name="InnerG", nodes=[nested_node])

    node = LogicalNode(
        id="op_main",
        op_type="CustomOp",
        dtype=DummyCompilerDType.FLOAT32,
        shape_metadata=(sym_b, 16),
        device=dev_idx,
        source_ast_ref=404,
        stream=3,
        sharding={"strategy": "megatron", "tp": 2},
        strides=[16, 1],
        attributes={
            "dtype_proto": DummyCompilerDType.BFLOAT16,
            "dtype_ir": DType.float16,
            "dev_proto": dev_noindex,
            "sym_node": sym_anon,
            "sym_wrap": sym_wrap,
            "sym_ir": SymVar("K"),
            "rule": rule,
            "loop": loop,
            "buf": buf,
            "raw_bytes": b"\xaa\xbb",
            "t_shape": TensorShape((2, 4)),
            "t_spec": TensorSpec((2, 4), DType.int8),
            "pspec": PartitionSpec(axes=("data", None)),
            "sub_node": nested_node,
            "sub_graph": nested_graph,
            "nested_map": {"inner": 1},
            "seq": [rule, (1, 2)],
        },
    )
    return node, dev_idx, rule, sym_wrap


def test_logical_node_fields_strides_and_json_dump() -> None:
    """Test LogicalNode strides, shape_tuple, widened fields, protocol attributes, and JSON dump."""
    node, dev_idx, rule, _sym_wrap = _build_rich_protocol_node()
    dev_noindex = DummyDevice("cpu", None)

    assert node.dtype is DType.float32
    assert node.shape_tuple is not None and node.shape_tuple[1] == 16
    assert node.strides == (16, 1)
    assert node.device == dev_idx
    assert node.source_ast_ref == 404
    assert node.stream == 3
    assert node.sharding == {"strategy": "megatron", "tp": 2}

    node.strides = (32, 2)
    assert node.strides == (32, 2)
    node.strides = None
    assert node.strides is None
    node.strides = [16, 1]

    node_axes_dict = LogicalNode.model_validate(
        {"id": "n_ax", "op_type": "Relu", "sharding": {"axes": ["data", None]}, "device": dev_noindex}
    )
    assert isinstance(node_axes_dict.sharding, PartitionSpec)
    assert node_axes_dict.sharding.axes == ("data", None)
    assert node_axes_dict.model_dump(mode="json")["device"] == "cpu"
    assert node_axes_dict.model_dump(mode="json")["sharding"] == {"axes": ["data", None]}

    node_str_shard = LogicalNode(id="n_str_s", op_type="Relu", sharding="replicated")
    assert node_str_shard.sharding == "replicated"
    assert node_str_shard.model_dump(mode="json")["sharding"] == "replicated"

    py_dump = node.model_dump(mode="python")
    assert node.device is dev_idx
    assert py_dump["device"] == {"type": "cuda", "index": 0}
    assert node.attributes["rule"] is rule

    json_dump = node.model_dump(mode="json")
    assert json_dump["device"] == "cuda:0"
    assert json_dump["attributes"]["rule"] == {
        "__sw_type__": "CustomRuleHandle",
        "handle_id": "custom_bwd_1",
        "kind": "bwd",
    }
    assert json_dump["attributes"]["sym_node"] == {
        "__sw_type__": "SymDim",
        "expr": "DummySymNodeAnon",
    }

    rich_g = LogicalGraph(name="RichNodeG", nodes=[node])
    assert "CustomRuleHandle" in rich_g.to_json()
    assert "CustomRuleHandle" in export_to_python(rich_g)


def test_logical_node_cloning_with_protocols_and_strides() -> None:
    """Test LogicalNode.clone() preservation and overrides across strides and protocol fields."""
    node, dev_idx, _rule, sym_wrap = _build_rich_protocol_node()
    dev_noindex = DummyDevice("cpu", None)

    cloned = node.clone()
    assert cloned.strides == (16, 1)
    assert cloned.source_ast_ref == 404
    assert cloned.stream == 3
    assert cloned.device == dev_idx
    assert cloned.sharding == {"strategy": "megatron", "tp": 2}

    override_shape: list[DimensionType] = [sym_wrap, 8]
    overridden = node.clone(
        id="op_override",
        strides=(64, 4),
        source_ast_ref=None,
        stream="cuda_s1",
        device=dev_noindex,
        sharding="replicated",
        dtype=DummyCompilerDType.BFLOAT16,
        shape_metadata=override_shape,
    )
    assert overridden.strides == (64, 4)
    assert overridden.source_ast_ref is None
    assert overridden.stream == "cuda_s1"
    assert overridden.device == dev_noindex
    assert overridden.sharding == "replicated"
    assert overridden.dtype is DType.bfloat16
    assert overridden.shape_tuple == (sym_wrap, 8)

    cleared = overridden.clone(strides=None, sharding=None, device=None, dtype=None)
    assert cleared.strides is None
    assert cleared.sharding is None
    assert cleared.device is None
    assert cleared.dtype is None

    coerced_clone = node.clone(
        device=123,
        source_ast_ref=999,
        stream=SymVar("s0"),
        dtype=DType.int32,
        sharding={"custom": 1},
    )
    assert coerced_clone.device == "123"
    assert coerced_clone.source_ast_ref == 999
    assert coerced_clone.stream == "s0"
    assert coerced_clone.sharding == {"custom": 1}

    coerced_dtype_clone = node.clone(dtype=SymVar("int16"))
    assert coerced_dtype_clone.dtype is DType.int16


def _build_rich_protocol_graph() -> tuple[LogicalGraph, DummyLoopOptions, DummyDeviceMesh, DummyArrayBuffer]:
    """Construct a LogicalGraph populated with annotations, metadata, sorted_nodes, and protocols.

    Returns:
        tuple[LogicalGraph, DummyLoopOptions, DummyDeviceMesh, DummyArrayBuffer]:
            Constructed graph and protocol fixture instances.
    """
    n_in = LogicalNode(id="x", op_type="Input", dtype="float32", shape_metadata=(2, 4))
    n_a = LogicalNode(
        id="a",
        op_type="Relu",
        inputs=["x"],
        device=DummyDevice("cuda", 0),
        sharding="replicated",
        stream=1,
        strides=(4, 1),
    )
    n_b = LogicalNode(
        id="b",
        op_type="Relu",
        inputs=["x"],
        device=DummyDevice("cuda", 0),
        sharding="replicated",
        stream=1,
        strides=(4, 1),
    )
    n_add = LogicalNode(
        id="out",
        op_type="Add",
        inputs=["a", "b"],
        device=DummyDevice("cpu", None),
        sharding={"axes_custom": "none"},
    )

    loop_proto = DummyLoopOptions(parallel_iterations=4, swap_memory=True, maximum_iterations=50)
    mesh_proto = DummyDeviceMesh(("dp", "tp"))
    buf = DummyArrayBuffer((2, 4), b"\x00" * 8)
    orphan_node = LogicalNode(id="orphan_cached", op_type="Identity")

    graph = LogicalGraph(
        name="ProtocolGraph",
        nodes=[n_in, n_a, n_b, n_add],
        inputs=("x",),
        outputs=("out",),
        mesh=mesh_proto,
        initializers={
            "w_buf": buf,
            "w_tuple": (1.0, 2.0),
            "w_bytes": b"\x10\x20",
            "w_map": {"scale": 0.5},
        },
        annotations={"pass": "spmd_partitioner", "version": 2},
        metadata={"loop_cfg": loop_proto, "note": None},
        sorted_nodes=[n_in, n_a, n_b, n_add, orphan_node],
        loop_options=loop_proto,
    )
    return graph, loop_proto, mesh_proto, buf


def test_logical_graph_fields_protocols_and_serialization() -> None:
    """Test LogicalGraph annotations, metadata, sorted_nodes, mesh/loop protocols, and JSON round-trip."""
    graph, loop_proto, mesh_proto, buf = _build_rich_protocol_graph()

    assert graph.inputs == ["x"]
    assert graph.outputs == ["out"]
    assert graph.annotations == {"pass": "spmd_partitioner", "version": 2}
    assert graph.metadata["loop_cfg"] is loop_proto
    assert graph.loop_options is loop_proto
    assert graph.mesh is mesh_proto
    assert graph.sorted_nodes is not None and len(graph.sorted_nodes) == 5

    py_dump = graph.model_dump(mode="python")
    assert graph.mesh is mesh_proto
    assert py_dump["mesh"] == {"axis_names": ("dp", "tp")}
    assert py_dump["loop_options"] == {
        "parallel_iterations": 4,
        "swap_memory": True,
        "maximum_iterations": 50,
    }
    assert py_dump["initializers"]["w_buf"] is buf

    json_dump = graph.model_dump(mode="json")
    assert json_dump["mesh"] == {"__sw_type__": "MeshLike", "axis_names": ["dp", "tp"]}
    assert json_dump["loop_options"]["__sw_type__"] == "LoopOptions"
    assert json_dump["initializers"]["w_buf"]["__sw_type__"] == "ArrayBuffer"
    assert json_dump["initializers"]["w_bytes"]["__sw_type__"] == "bytes"
    assert json_dump["initializers"]["w_tuple"] == [1.0, 2.0]
    assert json_dump["initializers"]["w_map"] == {"scale": 0.5}

    dict_repr = graph.to_dict()
    assert "sorted_nodes" not in dict_repr
    assert dict_repr["annotations"] == {"pass": "spmd_partitioner", "version": 2}
    json_text = graph.to_json(indent=2)
    assert json_text == json.dumps(dict_repr, sort_keys=True, indent=2)

    restored = LogicalGraph.from_json(json_text)
    assert restored.annotations == {"pass": "spmd_partitioner", "version": 2}
    assert restored.nodes["a"].strides == (4, 1)
    assert restored.nodes["a"].sharding == "replicated"
    assert restored.nodes["out"].sharding == {"axes_custom": "none"}

    cloned = graph.clone()
    assert cloned == graph
    assert cloned.annotations == graph.annotations
    assert cloned.sorted_nodes is not None and len(cloned.sorted_nodes) == 5
    assert cloned.sorted_nodes[0] is cloned.nodes["x"]
    assert cloned.sorted_nodes[-1].id == "orphan_cached"


def test_logical_graph_passes_validation_and_export() -> None:
    """Test CSE, DCE, SchemaValidator, distributed validators, and Python/JSON Schema export on protocol graphs."""
    graph, _loop_proto, mesh_proto, _buf = _build_rich_protocol_graph()

    cse_graph = eliminate_common_subexpressions(graph)
    assert cse_graph.annotations == graph.annotations
    dce_graph = eliminate_dead_nodes(cse_graph)
    assert len(topological_sort(dce_graph)) == 3

    validator = Validator(level=ValidationLevel.LENIENT)
    assert validator.validate_graph(graph) == []
    assert validate_sharding_node(graph.nodes["a"], None) == []
    assert validate_sharding_propagation_graph(graph) == []
    comm = estimate_graph_communication_volume(
        LogicalGraph(
            nodes=[
                LogicalNode(
                    id="ar",
                    op_type="all_reduce",
                    domain="collective",
                    shape_metadata=(4,),
                    dtype="float32",
                )
            ],
            mesh=mesh_proto,
        )
    )
    assert comm.total_volume_bytes == 16

    nested_host = LogicalGraph(
        name="HostWithSubgraph",
        nodes=[
            LogicalNode(
                id="call",
                op_type="If",
                subgraphs={"then_branch": graph},
                attributes={"nested_p": PartitionSpec(axes=("dp",)), "extra_m": GhostOverload()},
            )
        ],
        mesh="string_mesh_tag",
    )
    py_code = export_to_python(nested_host, function_name="build_proto_host")
    assert "annotations=" in py_code
    assert "strides=(4, 1)" in py_code
    assert "def build_proto_host()" in py_code
    assert '["dp", "tp"]' in export_to_python(graph, function_name="build_direct_mesh_proto")
    assert DummyArrayBuffer((1,), b"\x00") != "not_a_buffer"

    g_mesh_dict = LogicalGraph.model_validate(
        {"name": "MD", "mesh": {"shape": {"data": 4}}, "loop_options": {"unroll": 2}}
    )
    assert isinstance(g_mesh_dict.mesh, LogicalMesh)
    assert g_mesh_dict.mesh.shape == {"data": 4}
    assert g_mesh_dict.model_dump(mode="json")["loop_options"] == {"unroll": 2}
    assert g_mesh_dict.model_dump(mode="json")["mesh"]["shape"] == {"data": 4}

    g_mesh_scalar = LogicalGraph(name="MS", mesh="tpu_v5e_pod")
    assert g_mesh_scalar.mesh == "tpu_v5e_pod"
    assert LogicalGraph.from_json(g_mesh_scalar.to_json()).mesh == "tpu_v5e_pod"
    assert g_mesh_scalar.model_dump(mode="json")["mesh"] == "tpu_v5e_pod"
    nested_scalar_mesh_host = LogicalGraph(
        name="HostScalarSub",
        nodes=[LogicalNode(id="if_s", op_type="If", subgraphs={"b": g_mesh_scalar})],
    )
    assert '"tpu_v5e_pod"' in export_to_python(nested_scalar_mesh_host)

    g_mesh_custom_dict = LogicalGraph.from_dict(
        {
            "name": "MCD",
            "mesh": {"custom_mesh": 2},
            "nodes": {"p": {"id": "p", "op_type": "Relu", "sharding": PartitionSpec(axes=("d",))}},
        }
    )
    assert g_mesh_custom_dict.mesh == {"custom_mesh": 2}
    assert LogicalGraph.from_dict({"name": "EmptyMesh", "mesh": {}}).mesh is None

    all_schemas = get_json_schema("all")
    assert "LogicalGraph" in all_schemas
    assert "LogicalNode" in all_schemas


def test_schema_catalogs_mlir_and_ghost_overload_exports() -> None:
    """Verify wgsl_ops.json resource path, MLIR_REGISTRY, ExtendedGhostRef, GhostParam, and GhostOverload."""
    assert __version__ == "0.0.7"
    assert len(MLIR_REGISTRY) > 0

    wgsl_trav = importlib.resources.files("ml_switcheroo_ir.schema.catalogs").joinpath("wgsl_ops.json")
    assert wgsl_trav.is_file()

    wgsl_path = get_catalog_path("wgsl_ops.json")
    assert wgsl_path.is_file()
    catalog_data = json.loads(wgsl_path.read_text(encoding="utf-8"))
    assert isinstance(catalog_data, (dict, list))

    param = GhostParam(name="x", kind=ParameterKind.POSITIONAL_OR_KEYWORD, annotation="Tensor")
    overload = GhostOverload(
        params=[param],
        return_annotation="Tensor",
        returns_type="Tensor",
        docstring="Tensor overload.",
        has_varargs=False,
        accepted_kwargs=["out"],
    )
    raw_overload_dict: dict[str, sw_ir.types.JsonValue] = {"raw_sig": "(Tensor, Scalar) -> Tensor"}
    overload_entries: list[dict[str, sw_ir.types.JsonValue] | GhostOverload | GhostRef] = [
        overload,
        raw_overload_dict,
    ]
    ref = GhostRef(
        name="add",
        api_path="torch.add",
        kind="function",
        params=[param],
        overloads=overload_entries,
    )
    assert ref.overloads is not None and len(ref.overloads) == 2
    assert isinstance(ref.overloads[0], GhostOverload)

    ext_ref = ExtendedGhostRef(name="linalg.matmul", api_path="mlir.linalg.matmul", kind="operation")
    assert ext_ref.name == "linalg.matmul"


def test_negative_type_and_attribute_validation_branches() -> None:
    """Verify error/rejection branches for invalid DimensionType, DType, InitializerValue, and AttributeValue."""
    dim_adapter: pydantic.TypeAdapter[DimensionType] = pydantic.TypeAdapter(DimensionType)
    init_adapter: pydantic.TypeAdapter[InitializerValue] = pydantic.TypeAdapter(InitializerValue)
    attr_adapter: pydantic.TypeAdapter[AttributeValue] = pydantic.TypeAdapter(AttributeValue)

    with pytest.raises(pydantic.ValidationError):
        dim_adapter.validate_python(True)
    with pytest.raises(pydantic.ValidationError):
        dim_adapter.validate_python(BadSymIntWrapper())
    with pytest.raises(pydantic.ValidationError):
        dim_adapter.validate_python(3.14)
    assert dim_adapter.validate_python(DummySymNodeNamed("N")) is not None
    assert dim_adapter.validate_python(DummySymIntWrapper(DummySymNodeNamed("N"))) is not None

    with pytest.raises(TypeError, match="does not satisfy SymbolicNodeProtocol"):
        TensorShape.model_validate({"dims": [BadSymIntWrapper()]})
    shape_tuple_adapter: pydantic.TypeAdapter[tuple[DimensionType, ...]] = pydantic.TypeAdapter(
        typing.Annotated[tuple[DimensionType, ...], pydantic.BeforeValidator(sw_ir.types._validate_dimension_tuple)]
    )
    with pytest.raises(TypeError, match="does not satisfy SymbolicNodeProtocol"):
        shape_tuple_adapter.validate_python([BadSymIntWrapper()])
    with pytest.raises(TypeError, match="not a valid DimensionType"):
        sw_ir.types._validate_dimension_tuple(1.5)

    with pytest.raises(TypeError, match="Unsupported dtype"):
        TensorSpec.model_validate({"shape": [1], "dtype": BadIntDTypeEnum.INVALID})
    with pytest.raises(TypeError, match="Unsupported dtype"):
        TensorSpec.model_validate({"shape": [1], "dtype": 123})

    with pytest.raises(pydantic.ValidationError):
        init_adapter.validate_python(BadBoolShapeBuffer())
    with pytest.raises(pydantic.ValidationError):
        init_adapter.validate_python({123: "bad_key"})
    with pytest.raises(pydantic.ValidationError):
        init_adapter.validate_python({1, 2, 3})

    with pytest.raises(pydantic.ValidationError):
        attr_adapter.validate_python(BadBoolDevice())
    with pytest.raises(pydantic.ValidationError):
        attr_adapter.validate_python(BadLoopOptions())
    with pytest.raises(pydantic.ValidationError):
        attr_adapter.validate_python(BadSymIntWrapper())
    with pytest.raises(pydantic.ValidationError):
        attr_adapter.validate_python(BadBoolShapeBuffer())
    with pytest.raises(pydantic.ValidationError):
        attr_adapter.validate_python({1: "bad_key"})
    with pytest.raises(pydantic.ValidationError):
        attr_adapter.validate_python({1, 2})


def test_negative_node_and_graph_validation_branches() -> None:
    """Verify error/rejection branches for invalid LogicalNode and LogicalGraph field inputs."""
    strides_adapter: pydantic.TypeAdapter[tuple[int, ...] | None] = pydantic.TypeAdapter(
        typing.Annotated[
            typing.Union[tuple[int, ...], None],
            pydantic.BeforeValidator(sw_ir.graph._normalize_strides),
        ]
    )
    with pytest.raises(TypeError, match="strides"):
        strides_adapter.validate_python("bad")
    with pytest.raises(TypeError, match="Stride element must be int"):
        LogicalNode(id="n", op_type="Relu", strides=[True])
    with pytest.raises(pydantic.ValidationError):
        LogicalNode.model_validate({"id": "n", "op_type": "Relu", "source_ast_ref": True})
    with pytest.raises(pydantic.ValidationError):
        LogicalNode.model_validate({"id": "n", "op_type": "Relu", "device": BadBoolDevice()})
    with pytest.raises(pydantic.ValidationError):
        LogicalNode.model_validate({"id": "n", "op_type": "Relu", "device": 123})
    with pytest.raises(TypeError, match="Unsupported LogicalNode 'sharding'"):
        LogicalNode.model_validate({"id": "n", "op_type": "Relu", "sharding": 123})
    with pytest.raises(TypeError, match="mapping key must be str"):
        LogicalNode.model_validate({"id": "n", "op_type": "Relu", "sharding": {1: "bad"}})

    base_node = LogicalNode(id="n_ok", op_type="Relu")
    with pytest.raises(TypeError, match="Invalid strides override"):
        base_node.clone(strides="not_seq")
    with pytest.raises(TypeError, match="Invalid stride element"):
        base_node.clone(strides=[True])
    with pytest.raises(TypeError, match="Invalid shape_metadata override"):
        base_node.clone(shape_metadata=123)
    with pytest.raises(ValueError, match="Unknown or unsupported DType"):
        sw_ir.graph._resolve_dtype_override({"dtype": TensorShape((1,))}, None)

    with pytest.raises(TypeError, match="LogicalGraph 'nodes' must be"):
        LogicalGraph.model_validate({"name": "G", "nodes": "bad_str"})
    with pytest.raises(TypeError, match="LogicalGraph 'nodes' must be"):
        LogicalGraph.model_validate({"name": "G", "nodes": 123})
    with pytest.raises(TypeError, match="LogicalGraph 'inputs' must be Sequence"):
        LogicalGraph.model_validate({"name": "G", "inputs": "not_a_list"})
    with pytest.raises(TypeError, match="LogicalGraph 'inputs' element must be str"):
        LogicalGraph.model_validate({"name": "G", "inputs": [123]})
    with pytest.raises(pydantic.ValidationError):
        LogicalGraph.model_validate({"name": "G", "loop_options": 123})
    with pytest.raises(pydantic.ValidationError):
        LogicalGraph.model_validate({"name": "G", "loop_options": {1: "bad"}})
    with pytest.raises(pydantic.ValidationError):
        LogicalGraph.model_validate({"name": "G", "loop_options": BadLoopOptions()})
    with pytest.raises(pydantic.ValidationError):
        LogicalGraph.model_validate({"name": "G", "mesh": BadStringAxesMesh()})
