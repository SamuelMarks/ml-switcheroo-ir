"""Graph optimization and canonical transformation passes for ML-Switcheroo IR.

Provides pure-IR transformations including Dead Code Elimination (DCE),
Common Subexpression Elimination (CSE), and Unified Shape Propagation & Constant Folding.
"""

from __future__ import annotations

import typing
from collections import deque
from collections.abc import Callable, Mapping, Sequence

import typing_extensions

from ml_switcheroo_ir.graph import (
    LogicalEdge,
    LogicalGraph,
    LogicalNode,
    topological_sort,
)
from ml_switcheroo_ir.schema.custom_ops import (
    COLLECTIVE_OPS_REGISTRY,
    STATE_OPS_REGISTRY,
)
from ml_switcheroo_ir.shapes import (
    DimensionType,
    SymbolicConstraintTracker,
    SymInt,
    SymNode,
    infer_op_output_shape,
    normalize_axis,
)
from ml_switcheroo_ir.types import AttributeValue, DType, InitializerValue, TensorShape, TensorSpec

HashableAttrKey: typing_extensions.TypeAlias = typing.Union[
    str,
    int,
    float,
    bool,
    None,
    tuple["HashableAttrKey", ...],
]
NodeCseSignatureKey: typing_extensions.TypeAlias = tuple[
    str,
    str,
    int,
    tuple[str, ...],
    tuple[tuple[str, HashableAttrKey], ...],
    typing.Optional[tuple[DimensionType, ...]],
    typing.Optional[str],
    HashableAttrKey,
    tuple[tuple[str, tuple[DimensionType, ...]], ...],
    typing.Optional[str],
    typing.Optional[str],
    HashableAttrKey,
]
ConstScalar: typing_extensions.TypeAlias = typing.Union[int, float, bool]
ConstTensorValue: typing_extensions.TypeAlias = typing.Union[
    ConstScalar,
    list["ConstTensorValue"],
]

SIDE_EFFECT_OPS: set[str] = {
    "Print",
    "Output",
    "custom_call",
    "storageStore",
    "atomicAdd",
    "atomicSub",
    "atomicMax",
    "atomicMin",
    "atomicAnd",
    "atomicOr",
    "atomicXor",
    "atomicExchange",
    "atomicCompareExchangeWeak",
    "textureStore",
}

NON_DETERMINISTIC_OPS: set[str] = {
    "Input",
    "rng_bit_generator",
    "Dropout",
    "RandomUniform",
    "RandomNormal",
    "RandomUniformLike",
    "RandomNormalLike",
    "Multinomial",
    "Bernoulli",
}

COMMUTATIVE_OPS: frozenset[str] = frozenset(
    {
        "Add",
        "add",
        "arith.addi",
        "arith.addf",
        "Mul",
        "mul",
        "arith.muli",
        "arith.mulf",
        "And",
        "Or",
        "Equal",
        "Max",
        "Min",
    }
)


def _clone_graph_envelope(
    graph: LogicalGraph,
    *,
    nodes: Mapping[str, LogicalNode],
    outputs: list[str],
    initializers: Mapping[str, InitializerValue],
    edges: Sequence[LogicalEdge] | None,
) -> LogicalGraph:
    """Construct a transformed ``LogicalGraph`` preserving all graph-level metadata fields.

    Args:
        graph (LogicalGraph): Source graph providing envelope metadata.
        nodes (Mapping[str, LogicalNode]): Transformed node dictionary.
        outputs (list[str]): Transformed output list.
        initializers (Mapping[str, InitializerValue]): Transformed initializers dictionary.
        edges (Sequence[LogicalEdge] | None): Explicit reconstructed edges or None.

    Returns:
        LogicalGraph: Reconstructed graph preserving attributes, loop_options, mesh, and topology.
    """
    return LogicalGraph(
        name=graph.name,
        nodes=dict(nodes),
        inputs=list(graph.inputs),
        input_specs=dict(graph.input_specs),
        outputs=outputs,
        initializers=dict(initializers),
        mesh=graph.mesh,
        edges=edges,
        pipeline_topology=graph.pipeline_topology,
        attributes=dict(graph.attributes),
        loop_options=dict(graph.loop_options) if graph.loop_options is not None else None,
    )


def _is_side_effecting_node(node: LogicalNode, include_sources: bool = False) -> bool:
    """Return True if a node has side effects, state mutation, or collective semantics.

    Args:
        node (LogicalNode): Node to inspect.
        include_sources (bool): Whether to also treat Input and non-deterministic RNG ops as impure.

    Returns:
        bool: True if the node cannot be eliminated or merged.
    """
    is_effect = (
        bool(node.attributes.get("has_side_effect", False))
        or bool(node.attributes.get("side_effects", False))
        or bool(node.attributes.get("is_mutating", False))
        or node.op_type in SIDE_EFFECT_OPS
        or node.op_type in STATE_OPS_REGISTRY
        or node.op_type in COLLECTIVE_OPS_REGISTRY
        or node.domain
        in (
            "state",
            "ml.switcheroo.state",
            "collective",
            "ml.switcheroo.collective",
        )
    )
    if include_sources:
        return is_effect or node.op_type in NON_DETERMINISTIC_OPS
    return is_effect


def _infer_const_value_shape(val: AttributeValue | InitializerValue) -> tuple[int, ...] | None:
    """Infer concrete integer shape tuple from a scalar or nested constant sequence.

    Args:
        val (AttributeValue | InitializerValue): Constant literal value or TensorSpec.

    Returns:
        tuple[int, ...] | None: Concrete shape tuple if ``val`` is a well-formed tensor literal.
    """
    if isinstance(val, TensorSpec):
        ints = [d for d in val.shape if isinstance(d, int) and not isinstance(d, bool)]
        return tuple(ints) if len(ints) == len(val.shape) else None
    if isinstance(val, (int, float, bool)):
        return ()
    if not isinstance(val, (list, tuple)):
        return None
    if len(val) == 0:
        return (0,)
    sub_shapes = [_infer_const_value_shape(elem) for elem in val]
    first_sub = sub_shapes[0]
    if first_sub is None or any(sh != first_sub for sh in sub_shapes):
        return None
    return (len(val), *first_sub)


def _resolve_multi_output_port_shape(
    inp_id: str,
    new_nodes: Mapping[str, LogicalNode],
) -> tuple[DimensionType, ...] | None:
    """Resolve shape for ``'node_id:port'`` or named multi-output SSA port in ``new_nodes``.

    Args:
        inp_id (str): Input token (`'node_id:port'` or named output).
        new_nodes (Mapping[str, LogicalNode]): Processed upstream nodes.

    Returns:
        tuple[DimensionType, ...] | None: Resolved output port shape or None.
    """
    node_part, sep, idx_part = inp_id.rpartition(":")
    if sep and node_part in new_nodes and idx_part.isdigit():
        prod = new_nodes[node_part]
        port_idx = int(idx_part)
        if 0 <= port_idx < len(prod.output_specs):
            return tuple(prod.output_specs[port_idx].shape)
        return tuple(prod.shape_metadata) if isinstance(prod.shape_metadata, (list, tuple)) else None

    for prod in new_nodes.values():
        if inp_id in prod.outputs:
            port_idx = prod.outputs.index(inp_id)
            if 0 <= port_idx < len(prod.output_specs):
                return tuple(prod.output_specs[port_idx].shape)
            if len(prod.outputs) == 1 and isinstance(prod.shape_metadata, (list, tuple)):
                return tuple(prod.shape_metadata)
    return None


def _resolve_input_shape(
    inp_id: str,
    new_nodes: Mapping[str, LogicalNode],
    input_specs: Mapping[str, TensorSpec] | None = None,
    initializers: Mapping[str, InitializerValue] | None = None,
) -> tuple[DimensionType, ...] | None:
    """Resolve the tensor shape for an input identifier, multi-output SSA port, input_spec, or initializer.

    Args:
        inp_id (str): Input node ID or SSA value name.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.
        input_specs (Mapping[str, TensorSpec] | None): Graph input tensor specifications.
        initializers (Mapping[str, InitializerValue] | None): Graph constant initializers.

    Returns:
        tuple[DimensionType, ...] | None: Resolved input shape tuple or None.
    """
    direct = new_nodes.get(inp_id)
    if direct is not None:
        if isinstance(direct.shape_metadata, (list, tuple)):
            return tuple(direct.shape_metadata)
        return _infer_const_value_shape(direct.attributes["value"]) if "value" in direct.attributes else None

    port_shape = _resolve_multi_output_port_shape(inp_id, new_nodes)
    if port_shape is not None:
        return port_shape
    if input_specs is not None and inp_id in input_specs:
        return tuple(input_specs[inp_id].shape)
    if initializers is not None and inp_id in initializers:
        return _infer_const_value_shape(initializers[inp_id])
    return None


def _is_ssa_live(
    ssa_name: str,
    live_nodes: set[str],
    graph: LogicalGraph,
) -> bool:
    """Return True if an SSA value, graph input, initializer, or node ID is live.

    Args:
        ssa_name (str): Node ID or SSA value identifier.
        live_nodes (set[str]): Set of live node IDs.
        graph (LogicalGraph): Owning LogicalGraph.

    Returns:
        bool: True if the producer of ``ssa_name`` is live or is an external input/initializer.
    """
    if ssa_name in live_nodes or ssa_name in graph.inputs or ssa_name in graph.initializers:
        return True
    producer = graph.get_output_producer(ssa_name)
    return producer is not None and producer.id in live_nodes


def _collect_subgraph_bound_names(subgraph: LogicalGraph) -> set[str]:
    """Return all SSA identifiers locally bound inside ``subgraph``.

    Args:
        subgraph (LogicalGraph): Nested computational graph.

    Returns:
        set[str]: Locally bound input, initializer, node ID, and node output names.
    """
    bound = set(subgraph.inputs) | set(subgraph.initializers) | set(subgraph.nodes)
    for sub_node in subgraph.nodes.values():
        bound.update(sub_node.outputs)
        for idx in range(len(sub_node.output_specs)):
            bound.add(f"{sub_node.id}:{idx}")
    return bound


def _collect_subgraph_free_vars(
    subgraphs: Mapping[str, LogicalGraph | AttributeValue],
) -> set[str]:
    """Collect outer closure-captured SSA identifiers referenced inside nested ``LogicalGraph`` regions.

    Args:
        subgraphs (Mapping[str, LogicalGraph | AttributeValue]): Node subgraphs mapping.

    Returns:
        set[str]: Free SSA names referenced inside nested subgraphs.
    """
    free_vars: set[str] = set()
    for sub_val in subgraphs.values():
        if not isinstance(sub_val, LogicalGraph):
            continue
        local_bound = _collect_subgraph_bound_names(sub_val)
        referenced: set[str] = set(sub_val.outputs)
        for sub_node in sub_val.nodes.values():
            referenced.update(sub_node.inputs)
            referenced.update(_collect_subgraph_free_vars(sub_node.subgraphs))
        for ref in referenced:
            base = ref.rpartition(":")[0] if ":" in ref else ref
            if ref not in local_bound and base not in local_bound:
                free_vars.add(ref)
    return free_vars


def _enqueue_ssa_target(
    ssa_or_id: str,
    graph: LogicalGraph,
    live_nodes: set[str],
    queue: deque[str],
) -> None:
    """Resolve an SSA value or node ID to its producer node and enqueue if not yet live.

    Args:
        ssa_or_id (str): Node ID or output SSA value name.
        graph (LogicalGraph): Computational graph.
        live_nodes (set[str]): Set of discovered live node IDs.
        queue (deque[str]): BFS worklist queue.
    """
    target_id: str | None = ssa_or_id if ssa_or_id in graph.nodes else None
    if target_id is None:
        producer = graph.get_output_producer(ssa_or_id)
        if producer is not None and producer.id in graph.nodes:
            target_id = producer.id
    if target_id is not None and target_id not in live_nodes:
        live_nodes.add(target_id)
        queue.append(target_id)


def _collect_live_nodes(
    graph: LogicalGraph,
    cleaned_subgraphs: Mapping[str, dict[str, LogicalGraph | AttributeValue]],
) -> set[str]:
    """Perform backward BFS from outputs, side-effecting nodes, and live subgraph captures.

    Args:
        graph (LogicalGraph): Input computational graph.
        cleaned_subgraphs (Mapping[str, dict[str, LogicalGraph | AttributeValue]]):
            Pre-cleaned subgraphs per node ID so dead inner nodes do not retain outer captures.

    Returns:
        set[str]: Set of reachable live node IDs.
    """
    live_nodes: set[str] = set()
    queue: deque[str] = deque()

    for out_id in graph.outputs:
        _enqueue_ssa_target(out_id, graph, live_nodes, queue)

    for node_id, node in graph.nodes.items():
        if _is_side_effecting_node(node) and node_id not in live_nodes:
            live_nodes.add(node_id)
            queue.append(node_id)

    while queue:
        curr_id = queue.popleft()
        for inp_id in graph.nodes[curr_id].inputs:
            _enqueue_ssa_target(inp_id, graph, live_nodes, queue)
        for captured_ssa in _collect_subgraph_free_vars(cleaned_subgraphs[curr_id]):
            _enqueue_ssa_target(captured_ssa, graph, live_nodes, queue)

    return live_nodes


def eliminate_dead_nodes(graph: LogicalGraph) -> LogicalGraph:
    """Eliminate unreachable nodes, dead nested subgraph nodes, and unreferenced initializers.

    Marks outer SSA values referenced inside live ``node.subgraphs`` as live so closure-captured
    producers are never deleted, prunes unreferenced ``initializers``, and preserves
    ``LogicalGraph.attributes`` and ``LogicalGraph.loop_options``.

    Args:
        graph (LogicalGraph): Input computational graph.

    Returns:
        LogicalGraph: Cleaned graph containing only live reachable nodes and initializers.
    """
    cleaned_subgraphs: dict[str, dict[str, LogicalGraph | AttributeValue]] = {
        nid: {
            sub_k: (eliminate_dead_nodes(sub_v) if isinstance(sub_v, LogicalGraph) else sub_v)
            for sub_k, sub_v in node.subgraphs.items()
        }
        for nid, node in graph.nodes.items()
    }
    live_nodes = _collect_live_nodes(graph, cleaned_subgraphs)

    new_nodes: dict[str, LogicalNode] = {}
    referenced_names: set[str] = set(graph.outputs)
    for nid, node in graph.nodes.items():
        if nid in live_nodes:
            sub_map = cleaned_subgraphs[nid]
            new_nodes[nid] = node.clone(subgraphs=sub_map)
            referenced_names.update(node.inputs)
            referenced_names.update(_collect_subgraph_free_vars(sub_map))

    pruned_initializers = {
        init_k: init_v for init_k, init_v in graph.initializers.items() if init_k in referenced_names
    }

    new_edges: list[LogicalEdge] = [
        LogicalEdge(
            source=edge.source,
            target=edge.target,
            source_idx=edge.source_idx,
            target_idx=edge.target_idx,
            value_name=edge.value_name,
        )
        for edge in graph.edges
        if _is_ssa_live(edge.source, live_nodes, graph) and edge.target in live_nodes
    ]

    new_outputs = [out for out in graph.outputs if _is_ssa_live(out, live_nodes, graph)]

    return _clone_graph_envelope(
        graph,
        nodes=new_nodes,
        outputs=new_outputs,
        initializers=pruned_initializers,
        edges=new_edges,
    )


def _serialize_attr_for_key(
    val: (
        AttributeValue
        | InitializerValue
        | LogicalGraph
        | Mapping[str, LogicalGraph | AttributeValue]
        | Mapping[str, AttributeValue | dict[str, InitializerValue]]
    ),
) -> HashableAttrKey:
    """Normalize attribute values with explicit type tags for deterministic equivalence hashing.

    Args:
        val: Attribute, initializer, LogicalGraph, or mapping value to normalize.

    Returns:
        HashableAttrKey: Hashable normalized representation.
    """
    if isinstance(val, LogicalGraph):
        return ("LogicalGraph", _serialize_attr_for_key(val.to_dict()))
    if isinstance(val, Mapping):
        return ("dict", tuple(sorted((str(k), _serialize_attr_for_key(v)) for k, v in val.items())))
    if isinstance(val, (list, tuple)):
        return ("seq", tuple(_serialize_attr_for_key(v) for v in val))
    return f"{type(val)!r}:{val!r}"


def _build_node_cse_signature(
    node: LogicalNode,
    updated_inputs: Sequence[str],
    transformed_subgraphs: Mapping[str, LogicalGraph | AttributeValue],
) -> NodeCseSignatureKey:
    """Build deterministic CSE equivalence signature including version, dtype, sharding, and output_specs.

    Args:
        node (LogicalNode): Candidate pure node.
        updated_inputs (Sequence[str]): Rewritten SSA operand names.
        transformed_subgraphs (Mapping[str, LogicalGraph | AttributeValue]): Canonicalized subgraphs.

    Returns:
        NodeCseSignatureKey: Hashable CSE signature tuple.
    """
    canonical_inputs = tuple(sorted(updated_inputs)) if node.op_type in COMMUTATIVE_OPS else tuple(updated_inputs)
    attr_key = tuple(sorted((k, _serialize_attr_for_key(v)) for k, v in node.attributes.items()))
    shape_key = tuple(node.shape_metadata) if isinstance(node.shape_metadata, (list, tuple)) else None
    sharding_key = _serialize_attr_for_key(node.sharding.model_dump()) if node.sharding is not None else None
    out_specs_key = tuple((spec.dtype.value, tuple(spec.shape)) for spec in node.output_specs)
    subgraph_key = _serialize_attr_for_key(transformed_subgraphs)
    return (
        node.domain,
        node.op_type,
        node.version,
        canonical_inputs,
        attr_key,
        shape_key,
        node.dtype.value if node.dtype is not None else None,
        sharding_key,
        out_specs_key,
        node.device,
        node.stream,
        subgraph_key,
    )


def _rewrite_subgraph_captures(subgraph: LogicalGraph, outer_aliases: Mapping[str, str]) -> LogicalGraph:
    """Rewrite free outer SSA captures inside ``subgraph`` according to ``outer_aliases``.

    Args:
        subgraph (LogicalGraph): Nested computational graph.
        outer_aliases (Mapping[str, str]): Outer CSE ``old_ssa -> canonical_ssa`` mapping.

    Returns:
        LogicalGraph: Subgraph with outer captures rewritten and inner CSE applied.
    """
    if not outer_aliases:
        return eliminate_common_subexpressions(subgraph)
    local_bound = _collect_subgraph_bound_names(subgraph)
    filtered_aliases = {k: v for k, v in outer_aliases.items() if k not in local_bound}
    if not filtered_aliases:
        return eliminate_common_subexpressions(subgraph)

    rewritten_nodes: dict[str, LogicalNode] = {}
    for nid, snode in subgraph.nodes.items():
        new_inps = [filtered_aliases.get(inp, inp) for inp in snode.inputs]
        new_subs: dict[str, LogicalGraph | AttributeValue] = {
            sk: (_rewrite_subgraph_captures(sv, filtered_aliases) if isinstance(sv, LogicalGraph) else sv)
            for sk, sv in snode.subgraphs.items()
        }
        rewritten_nodes[nid] = snode.clone(inputs=new_inps, subgraphs=new_subs)

    rewritten_outputs = [filtered_aliases.get(out, out) for out in subgraph.outputs]
    rewritten_graph = _clone_graph_envelope(
        subgraph,
        nodes=rewritten_nodes,
        outputs=rewritten_outputs,
        initializers=subgraph.initializers,
        edges=None,
    )
    return eliminate_common_subexpressions(rewritten_graph)


def _register_node_cse_aliases(
    node: LogicalNode,
    canonical_node: LogicalNode,
    alias_map: dict[str, str],
) -> None:
    """Populate ``alias_map`` for a duplicate node and all of its named/indexed output ports.

    Args:
        node (LogicalNode): Duplicate node being eliminated.
        canonical_node (LogicalNode): Surviving canonical node.
        alias_map (dict[str, str]): Mutable alias map to update.
    """
    alias_map[node.id] = canonical_node.id
    port_count = max(len(node.outputs), len(node.output_specs), 1)
    for out_idx in range(port_count):
        target_name = canonical_node.get_output_name(out_idx)
        alias_map[f"{node.id}:{out_idx}"] = target_name
        if out_idx < len(node.outputs):
            alias_map[node.outputs[out_idx]] = target_name


def _resolve_surviving_port(
    inp: str,
    surviving_nodes: Mapping[str, LogicalNode],
) -> tuple[str, int, str]:
    """Resolve an SSA input reference to ``(source_token, source_idx, value_name)`` for LogicalEdge.

    Args:
        inp (str): Rewritten input identifier on a surviving consumer node.
        surviving_nodes (Mapping[str, LogicalNode]): Map of surviving nodes.

    Returns:
        tuple[str, int, str]: ``(edge_source, source_port_idx, value_name)``.
    """
    node_part, sep, idx_part = inp.rpartition(":")
    if sep and node_part in surviving_nodes and idx_part.isdigit():
        prod = surviving_nodes[node_part]
        port_idx = int(idx_part)
        return inp, port_idx, prod.get_output_name(port_idx)
    for prod in surviving_nodes.values():
        if inp in prod.outputs:
            port_idx = prod.outputs.index(inp)
            return inp, port_idx, inp
    return inp, 0, inp


def _build_incoming_port_lookup(graph: LogicalGraph) -> dict[tuple[str, int], str | None]:
    """Index original incoming ``value_name`` overrides by ``(target_node_id, target_idx)``.

    Args:
        graph (LogicalGraph): Original input graph.

    Returns:
        dict[tuple[str, int], str | None]: Port value_name lookup dictionary.
    """
    lookup: dict[tuple[str, int], str | None] = {}
    for edge in graph.edges:
        lookup[(edge.target, edge.target_idx)] = edge.value_name
    return lookup


def eliminate_common_subexpressions(graph: LogicalGraph) -> LogicalGraph:
    """Eliminate common subexpressions by merging structurally identical pure operations.

    Includes ``version``, ``dtype``, ``sharding``, and ``output_specs`` in equivalence keys,
    canonicalizes operand ordering for commutative operators, preserves multi-output port
    indices, and rewrites outer SSA captures inside nested ``node.subgraphs``.

    Args:
        graph (LogicalGraph): Input computational graph.

    Returns:
        LogicalGraph: Canonical graph with redundant subexpressions eliminated.
    """
    sorted_nodes = topological_sort(graph)
    incoming_val_names = _build_incoming_port_lookup(graph)
    alias_map: dict[str, str] = {}
    sig_map: dict[NodeCseSignatureKey, str] = {}
    surviving_nodes: dict[str, LogicalNode] = {}

    for node in sorted_nodes:
        updated_inputs = [alias_map.get(inp, inp) for inp in node.inputs]
        transformed_subgraphs: dict[str, LogicalGraph | AttributeValue] = {
            sub_k: (_rewrite_subgraph_captures(sub_v, alias_map) if isinstance(sub_v, LogicalGraph) else sub_v)
            for sub_k, sub_v in node.subgraphs.items()
        }

        if not _is_side_effecting_node(node, include_sources=True):
            sig = _build_node_cse_signature(node, updated_inputs, transformed_subgraphs)
            if sig in sig_map:
                _register_node_cse_aliases(node, surviving_nodes[sig_map[sig]], alias_map)
                continue
            sig_map[sig] = node.id

        surviving_nodes[node.id] = node.clone(
            inputs=updated_inputs,
            subgraphs=transformed_subgraphs,
        )

    new_outputs = [alias_map.get(out, out) for out in graph.outputs]

    new_edges: list[LogicalEdge] = []
    for node in surviving_nodes.values():
        for target_idx, inp in enumerate(node.inputs):
            src_tok, src_idx, val_name = _resolve_surviving_port(inp, surviving_nodes)
            orig_val_name = incoming_val_names.get((node.id, target_idx))
            if orig_val_name is not None:
                val_name = alias_map.get(orig_val_name, orig_val_name)
            new_edges.append(
                LogicalEdge(
                    source=src_tok,
                    target=node.id,
                    source_idx=src_idx,
                    target_idx=target_idx,
                    value_name=val_name,
                )
            )

    return _clone_graph_envelope(
        graph,
        nodes=surviving_nodes,
        outputs=new_outputs,
        initializers=graph.initializers,
        edges=new_edges,
    )


# --- Constant Folding Engine ---


def _is_const_tensor_value(val: AttributeValue | InitializerValue) -> typing.TypeGuard[ConstTensorValue]:
    """Return True if ``val`` is a numeric/boolean scalar or well-formed nested list/tuple literal.

    Args:
        val (AttributeValue | InitializerValue): Candidate value.

    Returns:
        typing.TypeGuard[ConstTensorValue]: True if ``val`` is foldable constant data.
    """
    if isinstance(val, (int, float, bool)):
        return True
    if isinstance(val, (list, tuple)):
        return all(_is_const_tensor_value(elem) for elem in val)
    return False


def _normalize_const_lists(val: AttributeValue | InitializerValue) -> ConstTensorValue | None:
    """Convert a constant scalar or nested list/tuple into canonical ``ConstTensorValue`` (lists).

    Args:
        val (AttributeValue | InitializerValue): Raw attribute or initializer value.

    Returns:
        ConstTensorValue | None: Normalized scalar or nested list if constant, else None.
    """
    if isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, (list, tuple)):
        converted: list[ConstTensorValue] = []
        for elem in val:
            sub = _normalize_const_lists(elem)
            if sub is None:
                return None
            converted.append(sub)
        if _infer_const_value_shape(converted) is None:
            return None
        return converted
    return None


def _resolve_constant_value(
    inp_id: str,
    new_nodes: Mapping[str, LogicalNode],
    initializers: Mapping[str, InitializerValue] | None = None,
) -> ConstTensorValue | None:
    """Resolve a compile-time constant tensor value from ``Constant`` nodes or static ``initializers``.

    Args:
        inp_id (str): Input node ID or SSA value name.
        new_nodes (Mapping[str, LogicalNode]): Processed upstream nodes.
        initializers (Mapping[str, InitializerValue] | None): Graph initializers.

    Returns:
        ConstTensorValue | None: Resolved constant value if statically known, else None.
    """
    node = new_nodes.get(inp_id)
    if node is None:
        for cand in new_nodes.values():
            if len(cand.outputs) == 1 and cand.outputs[0] == inp_id:
                node = cand
                break
    if node is not None and node.op_type in ("Constant", "constant", "arith.constant"):
        if "value" in node.attributes:
            return _normalize_const_lists(node.attributes["value"])
    if initializers is not None and inp_id in initializers:
        return _normalize_const_lists(initializers[inp_id])
    return None


def _infer_const_dtype(val: ConstTensorValue, hint: DType | None = None) -> DType:
    """Infer canonical ``DType`` for a folded constant value.

    Args:
        val (ConstTensorValue): Folded scalar or nested list.
        hint (DType | None): Optional explicit dtype hint.

    Returns:
        DType: Inferred element data type.
    """
    if hint is not None:
        return hint
    probe: ConstTensorValue = val
    while isinstance(probe, list):
        if not probe:
            return DType.float32
        probe = probe[0]
    if isinstance(probe, bool):
        return DType.bool
    if isinstance(probe, int):
        return DType.int64
    return DType.float32


def _map_unary_const(val: ConstTensorValue, fn: Callable[[ConstScalar], ConstScalar]) -> ConstTensorValue:
    """Apply ``fn`` recursively across a scalar or nested constant tensor list.

    Args:
        val (ConstTensorValue): Input constant value.
        fn (Callable[[ConstScalar], ConstScalar]): Scalar unary operator.

    Returns:
        ConstTensorValue: Transformed constant value.
    """
    if isinstance(val, list):
        return [_map_unary_const(elem, fn) for elem in val]
    return fn(val)


def _broadcast_const_to_shape(val: ConstTensorValue, target_shape: tuple[int, ...]) -> ConstTensorValue:
    """Broadcast a constant value ``val`` to ``target_shape`` following right-aligned rules.

    Args:
        val (ConstTensorValue): Constant scalar or nested list.
        target_shape (tuple[int, ...]): Concrete target shape.

    Returns:
        ConstTensorValue: Broadcasted nested list (or scalar when ``target_shape == ()``).
    """
    cur_shape = _infer_const_value_shape(val)
    if cur_shape is None or cur_shape == target_shape:
        return val
    while len(cur_shape) < len(target_shape):
        val = [val]
        cur_shape = (1, *cur_shape)
    return _expand_aligned_const(val, cur_shape, target_shape)


def _expand_aligned_const(
    val: ConstTensorValue,
    cur_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> ConstTensorValue:
    """Expand rank-aligned constant list ``val`` from ``cur_shape`` to ``target_shape``.

    Args:
        val (ConstTensorValue): Rank-aligned nested list or scalar.
        cur_shape (tuple[int, ...]): Current concrete shape.
        target_shape (tuple[int, ...]): Target concrete shape of equal rank.

    Returns:
        ConstTensorValue: Expanded constant tensor value.
    """
    if not target_shape or not isinstance(val, list):
        return val
    expanded_children = [_expand_aligned_const(child, cur_shape[1:], target_shape[1:]) for child in val]
    if cur_shape[0] == 1 and target_shape[0] != 1:
        return [_clone_const_value(expanded_children[0]) for _ in range(target_shape[0])]
    return expanded_children


def _clone_const_value(val: ConstTensorValue) -> ConstTensorValue:
    """Deep-copy a ``ConstTensorValue`` nested list structure.

    Args:
        val (ConstTensorValue): Constant value to copy.

    Returns:
        ConstTensorValue: Independent copy.
    """
    if isinstance(val, list):
        return [_clone_const_value(x) for x in val]
    return val


def _zip_binary_const(
    lhs: ConstTensorValue,
    rhs: ConstTensorValue,
    fn: Callable[[ConstScalar, ConstScalar], ConstScalar],
) -> ConstTensorValue:
    """Apply ``fn`` elementwise across two identically shaped constant tensors.

    Args:
        lhs (ConstTensorValue): Left constant tensor.
        rhs (ConstTensorValue): Right constant tensor.
        fn (Callable[[ConstScalar, ConstScalar], ConstScalar]): Binary scalar operator.

    Returns:
        ConstTensorValue: Resulting constant tensor.
    """
    if isinstance(lhs, list) and isinstance(rhs, list):
        return [_zip_binary_const(a, b, fn) for a, b in zip(lhs, rhs)]
    if not isinstance(lhs, list) and not isinstance(rhs, list):
        return fn(lhs, rhs)
    return lhs


def _eval_binary_scalar(op: str, a: ConstScalar, b: ConstScalar) -> ConstScalar:
    """Evaluate a pure binary arithmetic operation on two constant scalars.

    Args:
        op (str): Operator name ('Add', 'Sub', 'Mul', 'Div', etc.).
        a (ConstScalar): Left scalar.
        b (ConstScalar): Right scalar.

    Returns:
        ConstScalar: Computed scalar result.

    Raises:
        ZeroDivisionError: If dividing by zero.
    """
    na = int(a) if isinstance(a, bool) else a
    nb = int(b) if isinstance(b, bool) else b
    if op in ("Add", "add", "arith.addi", "arith.addf"):
        return na + nb
    if op in ("Sub", "sub", "arith.subi", "arith.subf"):
        return na - nb
    if op in ("Mul", "mul", "arith.muli", "arith.mulf"):
        return na * nb
    if nb == 0:
        raise ZeroDivisionError("Division by zero during constant folding.")
    if isinstance(na, int) and isinstance(nb, int) and op != "arith.divf":
        return int(na / nb)
    return na / nb


def _resolve_cast_target_dtype(node: LogicalNode) -> DType | None:
    """Resolve target ``DType`` from a ``Cast`` node's ``to`` or ``dtype`` attribute.

    Args:
        node (LogicalNode): Cast node.

    Returns:
        DType | None: Target DType if recognized, else None.
    """
    onnx_elem_map: dict[int, DType] = {
        1: DType.float32,
        2: DType.uint8,
        3: DType.int8,
        6: DType.int32,
        7: DType.int64,
        9: DType.bool,
        10: DType.float16,
        11: DType.float64,
        16: DType.bfloat16,
    }
    raw_to = node.attributes.get("to")
    if raw_to is None:
        raw_to = node.attributes.get("dtype")
    if raw_to is None and node.dtype is not None:
        return node.dtype
    if isinstance(raw_to, DType):
        return raw_to
    if isinstance(raw_to, int) and not isinstance(raw_to, bool):
        return onnx_elem_map.get(raw_to)
    if isinstance(raw_to, str):
        cleaned = raw_to.split(".")[-1].lower()
        for member in DType:
            if member.value.lower() == cleaned:
                return member
    return None


def _cast_scalar(val: ConstScalar, target_dtype: DType) -> ConstScalar:
    """Cast a constant scalar to Python bool, int, or float matching ``target_dtype``.

    Args:
        val (ConstScalar): Input scalar.
        target_dtype (DType): Target IR DType.

    Returns:
        ConstScalar: Casted scalar.
    """
    if target_dtype == DType.bool:
        return bool(val)
    if "int" in target_dtype.value:
        return int(val)
    return float(val)


def _concat_const_along_axis(tensors: Sequence[ConstTensorValue], axis: int) -> ConstTensorValue | None:
    """Concatenate constant nested-list tensors recursively along normalized ``axis``.

    Args:
        tensors (Sequence[ConstTensorValue]): Non-empty sequence of rank-matched constant lists.
        axis (int): Normalized 0-based concatenation axis.

    Returns:
        ConstTensorValue | None: Concatenated constant list or None if invalid.
    """
    if not all(isinstance(t, list) for t in tensors):
        return None
    list_tensors = [t for t in tensors if isinstance(t, list)]
    if axis == 0:
        merged: list[ConstTensorValue] = []
        for t in list_tensors:
            merged.extend(_clone_const_value(item) for item in t)
        return merged
    first_len = len(list_tensors[0])
    if any(len(t) != first_len for t in list_tensors):
        return None
    out: list[ConstTensorValue] = []
    for idx in range(first_len):
        sub = _concat_const_along_axis([t[idx] for t in list_tensors], axis - 1)
        if sub is None:
            return None
        out.append(sub)
    return out


def _slice_const_single_axis(
    tensor: ConstTensorValue,
    axis: int,
    start: int,
    end: int,
    step: int,
) -> ConstTensorValue | None:
    """Slice a constant nested-list tensor along a single normalized ``axis``.

    Args:
        tensor (ConstTensorValue): Input constant list.
        axis (int): Normalized 0-based axis.
        start (int): Slice start index.
        end (int): Slice end index.
        step (int): Non-zero slice step.

    Returns:
        ConstTensorValue | None: Sliced constant list or None if ``step == 0``.
    """
    if step == 0 or not isinstance(tensor, list):
        return None
    if axis == 0:
        return [_clone_const_value(x) for x in tensor[slice(start, end, step)]]
    out: list[ConstTensorValue] = []
    for child in tensor:
        sliced_child = _slice_const_single_axis(child, axis - 1, start, end, step)
        if sliced_child is None:
            return None
        out.append(sliced_child)
    return out


def _enrich_slice_attrs(
    node: LogicalNode,
    attrs: dict[str, AttributeValue],
    new_nodes: Mapping[str, LogicalNode],
    initializers: Mapping[str, InitializerValue],
) -> None:
    """Populate ``starts``, ``ends``, ``axes``, and ``steps`` on ``attrs`` from constant Slice inputs.

    Args:
        node (LogicalNode): Slice node.
        attrs (dict[str, AttributeValue]): Mutable attribute dictionary to populate.
        new_nodes (Mapping[str, LogicalNode]): Evaluated upstream nodes.
        initializers (Mapping[str, InitializerValue]): Static graph initializers.
    """
    for slot_idx, attr_key in ((1, "starts"), (2, "ends"), (3, "axes"), (4, "steps")):
        if attr_key not in attrs and len(node.inputs) > slot_idx:
            c_val = _resolve_constant_value(node.inputs[slot_idx], new_nodes, initializers)
            if isinstance(c_val, list):
                attrs[attr_key] = c_val


def _enrich_node_attrs_from_constants(
    node: LogicalNode,
    new_nodes: Mapping[str, LogicalNode],
    initializers: Mapping[str, InitializerValue],
) -> dict[str, AttributeValue]:
    """Populate shape-controlling attributes from folded constant operand inputs when omitted.

    Args:
        node (LogicalNode): Current node being evaluated.
        new_nodes (Mapping[str, LogicalNode]): Already-evaluated upstream nodes.
        initializers (Mapping[str, InitializerValue]): Static graph initializers.

    Returns:
        dict[str, AttributeValue]: Effective attributes dictionary for shape inference and folding.
    """
    attrs: dict[str, AttributeValue] = dict(node.attributes)
    op = node.op_type
    if op in ("Slice", "slice") and len(node.inputs) >= 3:
        _enrich_slice_attrs(node, attrs, new_nodes, initializers)
        return attrs
    if len(node.inputs) < 2:
        return attrs
    c_arg1 = _resolve_constant_value(node.inputs[1], new_nodes, initializers)
    if op in ("Reshape", "reshape") and "shape" not in attrs and "new_shape" not in attrs and isinstance(c_arg1, list):
        attrs["shape"] = c_arg1
    elif op in ("Squeeze", "squeeze", "Unsqueeze", "unsqueeze") and "axes" not in attrs and "axis" not in attrs:
        if c_arg1 is not None:
            attrs["axes"] = c_arg1
    elif op in ("Split", "split") and "split" not in attrs and isinstance(c_arg1, list):
        attrs["split"] = c_arg1
    return attrs


_FOLDABLE_BINARY_ARITH_OPS: frozenset[str] = frozenset(
    {
        "Add",
        "add",
        "Sub",
        "sub",
        "Mul",
        "mul",
        "Div",
        "div",
        "arith.addi",
        "arith.addf",
        "arith.subi",
        "arith.subf",
        "arith.muli",
        "arith.mulf",
        "arith.divf",
    }
)


def _fold_unary_or_cast_op(
    node: LogicalNode,
    new_nodes: Mapping[str, LogicalNode],
    initializers: Mapping[str, InitializerValue],
) -> tuple[ConstTensorValue, DType, tuple[DimensionType, ...]] | None:
    """Fold ``Neg`` or ``Cast`` when its operand is a compile-time constant.

    Args:
        node (LogicalNode): Unary Neg or Cast node.
        new_nodes (Mapping[str, LogicalNode]): Evaluated upstream nodes.
        initializers (Mapping[str, InitializerValue]): Static graph initializers.

    Returns:
        tuple[ConstTensorValue, DType, tuple[DimensionType, ...]] | None: Folded result or None.
    """
    if not node.inputs:
        return None
    c0 = _resolve_constant_value(node.inputs[0], new_nodes, initializers)
    sh0 = _infer_const_value_shape(c0) if c0 is not None else None
    if c0 is None or sh0 is None:
        return None
    if node.op_type == "Neg":
        folded = _map_unary_const(c0, lambda x: -(int(x) if isinstance(x, bool) else x))
        return folded, _infer_const_dtype(folded, node.dtype), sh0
    tgt_dt = _resolve_cast_target_dtype(node)
    if tgt_dt is not None:
        return _map_unary_const(c0, lambda x: _cast_scalar(x, tgt_dt)), tgt_dt, sh0
    return None


def _try_fold_pure_constant_op(
    node: LogicalNode,
    effective_attrs: Mapping[str, AttributeValue],
    inferred_shape: tuple[DimensionType, ...] | None,
    new_nodes: Mapping[str, LogicalNode],
    initializers: Mapping[str, InitializerValue],
) -> tuple[ConstTensorValue, DType, tuple[DimensionType, ...]] | None:
    """Attempt compile-time constant folding for Shape, arithmetic, Neg, Cast, Concat, and Slice.

    Args:
        node (LogicalNode): Node candidate for folding.
        effective_attrs (Mapping[str, AttributeValue]): Effective attributes including constant operands.
        inferred_shape (tuple[DimensionType, ...] | None): Inferred output shape.
        new_nodes (Mapping[str, LogicalNode]): Evaluated upstream nodes.
        initializers (Mapping[str, InitializerValue]): Static graph initializers.

    Returns:
        tuple[ConstTensorValue, DType, tuple[DimensionType, ...]] | None:
            ``(folded_value, folded_dtype, folded_shape)`` if foldable, else None.
    """
    op = node.op_type
    if op == "Shape" and node.inputs:
        src_shape = _resolve_input_shape(node.inputs[0], new_nodes, initializers=initializers)
        if src_shape is not None:
            conc = [d for d in src_shape if isinstance(d, int) and not isinstance(d, bool) and d >= 0]
            if len(conc) == len(src_shape):
                shape_vals: ConstTensorValue = list(conc)
                return shape_vals, DType.int64, (len(conc),)
    elif op in _FOLDABLE_BINARY_ARITH_OPS:
        return _fold_binary_arith_op(op, node, inferred_shape, new_nodes, initializers)
    elif op in ("Neg", "Cast", "cast"):
        return _fold_unary_or_cast_op(node, new_nodes, initializers)
    elif op in ("Concat", "concat") and node.inputs:
        return _fold_concat_const_op(node, effective_attrs, new_nodes, initializers)
    elif op in ("Slice", "slice") and node.inputs:
        return _fold_slice_const_op(node, effective_attrs, new_nodes, initializers)
    return None


def _fold_binary_arith_op(
    op: str,
    node: LogicalNode,
    inferred_shape: tuple[DimensionType, ...] | None,
    new_nodes: Mapping[str, LogicalNode],
    initializers: Mapping[str, InitializerValue],
) -> tuple[ConstTensorValue, DType, tuple[DimensionType, ...]] | None:
    """Fold a pure binary arithmetic node when both operands are compile-time constants.

    Args:
        op (str): Binary arithmetic mnemonic.
        node (LogicalNode): Binary operator node.
        inferred_shape (tuple[DimensionType, ...] | None): Broadcasted output shape.
        new_nodes (Mapping[str, LogicalNode]): Evaluated upstream nodes.
        initializers (Mapping[str, InitializerValue]): Static graph initializers.

    Returns:
        tuple[ConstTensorValue, DType, tuple[DimensionType, ...]] | None: Folded result or None.
    """
    if len(node.inputs) < 2 or inferred_shape is None:
        return None
    conc_dims = [d for d in inferred_shape if isinstance(d, int) and not isinstance(d, bool)]
    if len(conc_dims) != len(inferred_shape):
        return None
    lhs = _resolve_constant_value(node.inputs[0], new_nodes, initializers)
    rhs = _resolve_constant_value(node.inputs[1], new_nodes, initializers)
    if lhs is None or rhs is None:
        return None
    concrete_shape = tuple(conc_dims)
    b_lhs = _broadcast_const_to_shape(lhs, concrete_shape)
    b_rhs = _broadcast_const_to_shape(rhs, concrete_shape)
    try:
        folded = _zip_binary_const(b_lhs, b_rhs, lambda a, b: _eval_binary_scalar(op, a, b))
    except ZeroDivisionError:
        return None
    return folded, _infer_const_dtype(folded, node.dtype), concrete_shape


def _fold_concat_const_op(
    node: LogicalNode,
    effective_attrs: Mapping[str, AttributeValue],
    new_nodes: Mapping[str, LogicalNode],
    initializers: Mapping[str, InitializerValue],
) -> tuple[ConstTensorValue, DType, tuple[DimensionType, ...]] | None:
    """Fold a ``Concat`` node when all input tensors are compile-time constants.

    Args:
        node (LogicalNode): Concat node.
        effective_attrs (Mapping[str, AttributeValue]): Effective attributes.
        new_nodes (Mapping[str, LogicalNode]): Evaluated upstream nodes.
        initializers (Mapping[str, InitializerValue]): Static graph initializers.

    Returns:
        tuple[ConstTensorValue, DType, tuple[DimensionType, ...]] | None: Folded result or None.
    """
    vals: list[ConstTensorValue] = []
    for inp_id in node.inputs:
        cv = _resolve_constant_value(inp_id, new_nodes, initializers)
        if cv is None:
            return None
        vals.append(cv)
    first_sh = _infer_const_value_shape(vals[0])
    if first_sh is None or len(first_sh) == 0:
        return None
    axis_raw = effective_attrs.get("axis", 0)
    axis = int(axis_raw) if isinstance(axis_raw, (int, str)) and not isinstance(axis_raw, bool) else 0
    norm_ax = normalize_axis(axis, len(first_sh))
    folded = _concat_const_along_axis(vals, norm_ax)
    out_sh = _infer_const_value_shape(folded) if folded is not None else None
    if folded is None or out_sh is None:
        return None
    return folded, _infer_const_dtype(folded, node.dtype), out_sh


def _fold_slice_const_op(
    node: LogicalNode,
    effective_attrs: Mapping[str, AttributeValue],
    new_nodes: Mapping[str, LogicalNode],
    initializers: Mapping[str, InitializerValue],
) -> tuple[ConstTensorValue, DType, tuple[DimensionType, ...]] | None:
    """Fold a ``Slice`` node when its input data and bounds are compile-time constants.

    Args:
        node (LogicalNode): Slice node.
        effective_attrs (Mapping[str, AttributeValue]): Effective attributes including starts/ends/axes/steps.
        new_nodes (Mapping[str, LogicalNode]): Evaluated upstream nodes.
        initializers (Mapping[str, InitializerValue]): Static graph initializers.

    Returns:
        tuple[ConstTensorValue, DType, tuple[DimensionType, ...]] | None: Folded result or None.
    """
    data_val = _resolve_constant_value(node.inputs[0], new_nodes, initializers)
    starts = effective_attrs.get("starts")
    ends = effective_attrs.get("ends")
    data_sh = _infer_const_value_shape(data_val) if data_val is not None else None
    if data_val is None or data_sh is None or len(data_sh) == 0:
        return None
    if not isinstance(starts, (list, tuple)) or not isinstance(ends, (list, tuple)):
        return None
    axes_raw = effective_attrs.get("axes")
    steps_raw = effective_attrs.get("steps")
    axes_list = list(axes_raw) if isinstance(axes_raw, (list, tuple)) else list(range(len(starts)))
    steps_list = list(steps_raw) if isinstance(steps_raw, (list, tuple)) else [1] * len(starts)

    cur: ConstTensorValue | None = data_val
    for st, en, ax, sp in zip(starts, ends, axes_list, steps_list):
        if cur is None or not (
            isinstance(st, int)
            and not isinstance(st, bool)
            and isinstance(en, int)
            and not isinstance(en, bool)
            and isinstance(sp, int)
            and not isinstance(sp, bool)
        ):
            return None
        ax_int = int(ax) if isinstance(ax, (int, str)) and not isinstance(ax, bool) else 0
        cur = _slice_const_single_axis(cur, normalize_axis(ax_int, len(data_sh)), st, en, sp)
    out_sh = _infer_const_value_shape(cur) if cur is not None else None
    if cur is None or out_sh is None:
        return None
    return cur, _infer_const_dtype(cur, node.dtype), out_sh


# --- Unified Operator Shape Propagator Wrappers (routing through SHAPE_TRANSFER_REGISTRY) ---


def _node_input_shapes(
    node: LogicalNode,
    new_nodes: Mapping[str, LogicalNode],
    input_specs: Mapping[str, TensorSpec] | None = None,
    initializers: Mapping[str, InitializerValue] | None = None,
) -> list[tuple[DimensionType, ...] | None]:
    """Resolve all input shapes for ``node`` including multi-output SSA port references.

    Args:
        node (LogicalNode): Target node.
        new_nodes (Mapping[str, LogicalNode]): Processed upstream nodes.
        input_specs (Mapping[str, TensorSpec] | None): Graph input tensor specs.
        initializers (Mapping[str, InitializerValue] | None): Graph initializers.

    Returns:
        list[tuple[DimensionType, ...] | None]: Resolved input shapes in operand order.
    """
    return [_resolve_input_shape(inp, new_nodes, input_specs, initializers) for inp in node.inputs]


def _propagate_unary_op(node: LogicalNode, new_nodes: Mapping[str, LogicalNode]) -> tuple[DimensionType, ...] | None:
    """Propagate shape for elementwise unary operations via ``SHAPE_TRANSFER_REGISTRY``.

    Args:
        node (LogicalNode): Current unary node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Propagated shape or None if unshaped.
    """
    return infer_op_output_shape("Relu", _node_input_shapes(node, new_nodes), node.attributes, node_id=node.id)


def _propagate_binary_op(
    node: LogicalNode,
    new_nodes: Mapping[str, LogicalNode],
    tracker: SymbolicConstraintTracker,
) -> tuple[DimensionType, ...] | None:
    """Propagate broadcasted output shape for elementwise binary operations.

    Args:
        node (LogicalNode): Current binary node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.
        tracker (SymbolicConstraintTracker): Constraint tracker for symbolic unification.

    Returns:
        tuple[DimensionType, ...] | None: Broadcasted shape or None if inputs are unshaped.
    """
    return infer_op_output_shape(
        "Add", _node_input_shapes(node, new_nodes), node.attributes, node_id=node.id, tracker=tracker
    )


def _propagate_matmul_op(node: LogicalNode, new_nodes: Mapping[str, LogicalNode]) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for matrix multiplication operations.

    Args:
        node (LogicalNode): Current MatMul or BatchMatMul node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Resulting matmul shape or None if unshaped.
    """
    return infer_op_output_shape("MatMul", _node_input_shapes(node, new_nodes), node.attributes, node_id=node.id)


def _propagate_gemm_op(
    node: LogicalNode,
    new_nodes: Mapping[str, LogicalNode],
    tracker: SymbolicConstraintTracker,
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for General Matrix Multiply (Gemm) operations.

    Args:
        node (LogicalNode): Current Gemm node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.
        tracker (SymbolicConstraintTracker): Constraint tracker for broadcast unification.

    Returns:
        tuple[DimensionType, ...] | None: Gemm output shape or None if inputs are unshaped.
    """
    return infer_op_output_shape(
        "Gemm", _node_input_shapes(node, new_nodes), node.attributes, node_id=node.id, tracker=tracker
    )


def _propagate_squeeze_op(node: LogicalNode, new_nodes: Mapping[str, LogicalNode]) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Squeeze operations removing specified or size-1 axes.

    Args:
        node (LogicalNode): Current Squeeze node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Squeezed shape or None if unshaped.
    """
    attrs = _enrich_node_attrs_from_constants(node, new_nodes, {})
    return infer_op_output_shape("Squeeze", _node_input_shapes(node, new_nodes), attrs, node_id=node.id)


def _propagate_unsqueeze_op(
    node: LogicalNode, new_nodes: Mapping[str, LogicalNode]
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Unsqueeze operations inserting size-1 dimensions.

    Args:
        node (LogicalNode): Current Unsqueeze node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Unsqueezed shape or None if unshaped.
    """
    attrs = _enrich_node_attrs_from_constants(node, new_nodes, {})
    return infer_op_output_shape("Unsqueeze", _node_input_shapes(node, new_nodes), attrs, node_id=node.id)


def _propagate_flatten_op(node: LogicalNode, new_nodes: Mapping[str, LogicalNode]) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Flatten operations collapsing into a 2D tensor.

    Args:
        node (LogicalNode): Current Flatten node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Flattened 2D shape or None if unshaped.
    """
    return infer_op_output_shape("Flatten", _node_input_shapes(node, new_nodes), node.attributes, node_id=node.id)


def _propagate_concat_op(node: LogicalNode, new_nodes: Mapping[str, LogicalNode]) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Concat operations joining tensors along an axis.

    Args:
        node (LogicalNode): Current Concat node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Concatenated shape or None if unshaped.
    """
    return infer_op_output_shape("Concat", _node_input_shapes(node, new_nodes), node.attributes, node_id=node.id)


def _propagate_split_op(node: LogicalNode, new_nodes: Mapping[str, LogicalNode]) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Split operations slicing tensor along an axis.

    Args:
        node (LogicalNode): Current Split node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Shape of first output partition or None.
    """
    attrs = _enrich_node_attrs_from_constants(node, new_nodes, {})
    return infer_op_output_shape(
        "Split",
        _node_input_shapes(node, new_nodes),
        attrs,
        node_id=node.id,
        num_outputs=len(node.outputs),
    )


def _propagate_slice_op(node: LogicalNode, new_nodes: Mapping[str, LogicalNode]) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Slice operations given starts, ends, axes, steps.

    Args:
        node (LogicalNode): Current Slice node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Sliced shape or None if unshaped.
    """
    attrs = _enrich_node_attrs_from_constants(node, new_nodes, {})
    return infer_op_output_shape("Slice", _node_input_shapes(node, new_nodes), attrs, node_id=node.id)


def _propagate_shape_op(node: LogicalNode, new_nodes: Mapping[str, LogicalNode]) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Shape operator returning 1D rank tensor.

    Args:
        node (LogicalNode): Current Shape node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: 1D shape tuple ``(rank,)`` or None.
    """
    return infer_op_output_shape("Shape", _node_input_shapes(node, new_nodes), node.attributes, node_id=node.id)


def _propagate_reshape_op(node: LogicalNode, new_nodes: Mapping[str, LogicalNode]) -> tuple[DimensionType, ...] | None:
    """Propagate output shape and validate element count for Reshape operations.

    Args:
        node (LogicalNode): Current Reshape node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Resolved target shape or None.
    """
    attrs = _enrich_node_attrs_from_constants(node, new_nodes, {})
    return infer_op_output_shape("Reshape", _node_input_shapes(node, new_nodes), attrs, node_id=node.id)


def _propagate_transpose_op(
    node: LogicalNode, new_nodes: Mapping[str, LogicalNode]
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Transpose operations according to perm attribute.

    Args:
        node (LogicalNode): Current Transpose node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Permuted shape or None.
    """
    return infer_op_output_shape("Transpose", _node_input_shapes(node, new_nodes), node.attributes, node_id=node.id)


def _propagate_broadcast_in_dim_op(
    node: LogicalNode, new_nodes: Mapping[str, LogicalNode]
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for BroadcastInDim operations.

    Args:
        node (LogicalNode): Current BroadcastInDim node.
        new_nodes (Mapping[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Broadcasted target shape or None.
    """
    return infer_op_output_shape(
        "BroadcastInDim", _node_input_shapes(node, new_nodes), node.attributes, node_id=node.id
    )


def _populate_split_output_specs(
    node: LogicalNode,
    effective_attrs: Mapping[str, AttributeValue],
    inferred_shape: tuple[DimensionType, ...] | None,
    new_nodes: Mapping[str, LogicalNode],
    input_specs: Mapping[str, TensorSpec],
    initializers: Mapping[str, InitializerValue],
) -> list[TensorSpec]:
    """Build per-port ``TensorSpec`` list for multi-output ``Split`` nodes when omitted.

    Args:
        node (LogicalNode): Split node.
        effective_attrs (Mapping[str, AttributeValue]): Effective attributes (including folded split sizes).
        inferred_shape (tuple[DimensionType, ...] | None): Inferred first-port shape.
        new_nodes (Mapping[str, LogicalNode]): Processed upstream nodes.
        input_specs (Mapping[str, TensorSpec]): Graph input tensor specs.
        initializers (Mapping[str, InitializerValue]): Graph initializers.

    Returns:
        list[TensorSpec]: Populated output_specs list.
    """
    if node.output_specs or node.op_type not in ("Split", "split") or len(node.outputs) <= 1 or not node.inputs:
        return list(node.output_specs)
    inp_shape = _resolve_input_shape(node.inputs[0], new_nodes, input_specs, initializers)
    if inp_shape is None or inferred_shape is None:
        return []
    axis_raw = effective_attrs.get("axis", 0)
    axis = int(axis_raw) if isinstance(axis_raw, (int, str)) and not isinstance(axis_raw, bool) else 0
    norm_ax = normalize_axis(axis, len(inp_shape))
    inp_node = new_nodes.get(node.inputs[0])
    dt = node.dtype or (inp_node.dtype if inp_node is not None else None) or DType.float32
    split_attr = effective_attrs.get("split")
    specs: list[TensorSpec] = []
    if isinstance(split_attr, (list, tuple)):
        for sz in split_attr:
            s_i = list(inp_shape)
            s_i[norm_ax] = (
                sz
                if isinstance(sz, (int, SymNode, SymInt)) and not isinstance(sz, bool)
                else (int(str(sz)) if str(sz).isdigit() else str(sz))
            )
            specs.append(TensorSpec(shape=TensorShape(s_i), dtype=dt))
    else:
        for _ in node.outputs:
            specs.append(TensorSpec(shape=TensorShape(inferred_shape), dtype=dt))
    return specs


def propagate_shapes_and_constants(graph: LogicalGraph) -> LogicalGraph:
    """Propagate statically computable tensor shapes and fold constant operations.

    Routes all operator shape rules through ``SHAPE_TRANSFER_REGISTRY``, resolves multi-output
    SSA input shapes accurately, folds ``Shape`` on statically shaped inputs and pure constant
    ``Add``/``Sub``/``Mul``/``Div``/``Neg``/``Cast``/``Concat``/``Slice`` expressions into
    ``Constant`` nodes, and consumes folded constant tensors in downstream shape inference.

    Args:
        graph (LogicalGraph): Input computational graph.

    Returns:
        LogicalGraph: Graph with propagated shapes and folded ``Constant`` nodes.
    """
    sorted_nodes = topological_sort(graph)
    new_nodes: dict[str, LogicalNode] = {}
    folded_node_ids: set[str] = set()
    tracker = SymbolicConstraintTracker()

    for node in sorted_nodes:
        effective_attrs = _enrich_node_attrs_from_constants(node, new_nodes, graph.initializers)
        inp_shapes = _node_input_shapes(node, new_nodes, graph.input_specs, graph.initializers)

        computed = infer_op_output_shape(
            node.op_type,
            inp_shapes,
            effective_attrs,
            node_id=node.id,
            num_outputs=len(node.outputs),
            tracker=tracker,
        )
        if computed is None and node.op_type in ("Constant", "constant", "arith.constant"):
            if node.shape_metadata is None and "value" in node.attributes:
                computed = _infer_const_value_shape(node.attributes["value"])
        if (
            computed is None
            and node.op_type == "Input"
            and node.shape_metadata is None
            and node.id in graph.input_specs
        ):
            computed = tuple(graph.input_specs[node.id].shape)

        inferred_shape = computed if computed is not None else node.shape_metadata

        transformed_subgraphs: dict[str, LogicalGraph | AttributeValue] = {
            sub_k: (propagate_shapes_and_constants(sub_v) if isinstance(sub_v, LogicalGraph) else sub_v)
            for sub_k, sub_v in node.subgraphs.items()
        }

        folded = (
            None
            if _is_side_effecting_node(node, include_sources=True)
            else _try_fold_pure_constant_op(node, effective_attrs, inferred_shape, new_nodes, graph.initializers)
        )
        if folded is not None:
            folded_val, folded_dt, folded_shape = folded
            folded_node_ids.add(node.id)
            new_nodes[node.id] = node.clone(
                op_type="Constant",
                domain="ai.onnx",
                inputs=[],
                attributes={"value": folded_val},
                dtype=folded_dt,
                shape_metadata=folded_shape,
                subgraphs=transformed_subgraphs,
            )
            continue

        updated_output_specs = _populate_split_output_specs(
            node, effective_attrs, inferred_shape, new_nodes, graph.input_specs, graph.initializers
        )
        new_nodes[node.id] = node.clone(
            shape_metadata=inferred_shape,
            output_specs=updated_output_specs,
            subgraphs=transformed_subgraphs,
        )

    kept_edges = [edge for edge in graph.edges if edge.target not in folded_node_ids]
    return _clone_graph_envelope(
        graph,
        nodes=new_nodes,
        outputs=list(graph.outputs),
        initializers=graph.initializers,
        edges=kept_edges,
    )
