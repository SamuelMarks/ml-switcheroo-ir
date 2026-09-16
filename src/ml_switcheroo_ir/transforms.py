"""Graph optimization and canonical transformation passes for ML-Switcheroo IR.

Provides pure-IR transformations including Dead Code Elimination (DCE),
Common Subexpression Elimination (CSE), and Shape Propagation / Constant Folding.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ml_switcheroo_ir import (
    LogicalEdge,
    LogicalGraph,
    LogicalNode,
    topological_sort,
)

SIDE_EFFECT_OPS: set[str] = {
    "Print",
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


def eliminate_dead_nodes(graph: LogicalGraph) -> LogicalGraph:
    """Eliminate non-output reachable and dead subgraphs from a LogicalGraph.

    Preserves explicit graph outputs and nodes with side-effects or custom call contracts.

    Args:
        graph (LogicalGraph): Input computational graph.

    Returns:
        LogicalGraph: Cleaned graph containing only live reachable nodes.
    """
    # 1. Identify live roots: outputs and side-effecting operations
    live_nodes: set[str] = set()
    queue: deque[str] = deque()

    # Add graph explicit outputs
    for out_id in graph.outputs:
        if out_id in graph.nodes and out_id not in live_nodes:
            live_nodes.add(out_id)
            queue.append(out_id)
        else:
            producer = graph.get_output_producer(out_id)
            if (
                producer is not None
                and producer.id in graph.nodes
                and producer.id not in live_nodes
            ):
                live_nodes.add(producer.id)
                queue.append(producer.id)

    # Add nodes marked with side-effects or known mutating ops
    for node_id, node in graph.nodes.items():
        has_effect = (
            bool(node.attributes.get("has_side_effect", False))
            or bool(node.attributes.get("side_effects", False))
            or bool(node.attributes.get("is_mutating", False))
            or node.op_type in SIDE_EFFECT_OPS
        )
        if has_effect and node_id not in live_nodes:
            live_nodes.add(node_id)
            queue.append(node_id)

    # 2. Backward BFS traversal along node inputs
    while queue:
        curr_id = queue.popleft()
        curr_node = graph.nodes[curr_id]
        for inp_id in curr_node.inputs:
            if inp_id in graph.nodes and inp_id not in live_nodes:
                live_nodes.add(inp_id)
                queue.append(inp_id)

    # 3. Filter surviving nodes and recursively process nested subgraphs
    new_nodes: dict[str, LogicalNode] = {}
    for nid, node in graph.nodes.items():
        if nid in live_nodes:
            transformed_subgraphs: dict[str, Any] = {}
            for sub_k, sub_v in node.subgraphs.items():
                if isinstance(sub_v, LogicalGraph):
                    transformed_subgraphs[sub_k] = eliminate_dead_nodes(sub_v)
                else:
                    transformed_subgraphs[sub_k] = sub_v

            new_nodes[nid] = LogicalNode(
                id=node.id,
                op_type=node.op_type,
                domain=node.domain,
                inputs=list(node.inputs),
                attributes=dict(node.attributes),
                outputs=list(node.outputs) if node.outputs is not None else None,
                shape_metadata=node.shape_metadata,
                sharding=node.sharding,
                dtype=node.dtype,
                output_specs=list(node.output_specs),
                subgraphs=transformed_subgraphs,
                device=node.device,
                stream=node.stream,
            )

    # 4. Filter surviving edges
    new_edges: list[LogicalEdge] = [
        LogicalEdge(
            source=edge.source,
            target=edge.target,
            source_idx=edge.source_idx,
            target_idx=edge.target_idx,
            value_name=edge.value_name,
        )
        for edge in graph.edges
        if edge.source in live_nodes and edge.target in live_nodes
    ]

    new_outputs = [out for out in graph.outputs if out in live_nodes]

    return LogicalGraph(
        name=graph.name,
        nodes=new_nodes,
        inputs=list(graph.inputs),
        input_specs=dict(graph.input_specs),
        outputs=new_outputs,
        initializers=dict(graph.initializers),
        mesh=graph.mesh,
        edges=new_edges,
    )


def _serialize_attr_for_key(val: Any) -> Any:
    """Normalize attribute values for deterministic equivalence hashing.

    Args:
        val (Any): Attribute value to normalize.

    Returns:
        Any: Hashable normalized representation.
    """
    if isinstance(val, dict):
        return tuple(sorted((k, _serialize_attr_for_key(v)) for k, v in val.items()))
    if isinstance(val, (list, tuple)):
        return tuple(_serialize_attr_for_key(v) for v in val)
    return repr(val)


def eliminate_common_subexpressions(graph: LogicalGraph) -> LogicalGraph:
    """Eliminate common subexpressions by merging structurally identical pure operations.

    Args:
        graph (LogicalGraph): Input computational graph.

    Returns:
        LogicalGraph: Canonical graph with redundant subexpressions eliminated.
    """
    sorted_nodes = topological_sort(graph)
    alias_map: dict[str, str] = {}
    sig_map: dict[tuple[Any, ...], str] = {}
    surviving_nodes: dict[str, LogicalNode] = {}

    for node in sorted_nodes:
        # Update inputs based on previous CSE aliases
        updated_inputs = [alias_map.get(inp, inp) for inp in node.inputs]

        # Check if pure and eligible for CSE
        has_effect = (
            bool(node.attributes.get("has_side_effect", False))
            or bool(node.attributes.get("side_effects", False))
            or bool(node.attributes.get("is_mutating", False))
            or node.op_type in SIDE_EFFECT_OPS
            or node.op_type in ("Input", "rng_bit_generator")
        )

        if not has_effect:
            attr_key = tuple(
                sorted(
                    (k, _serialize_attr_for_key(v)) for k, v in node.attributes.items()
                )
            )
            shape_key = (
                tuple(node.shape_metadata)
                if isinstance(node.shape_metadata, (list, tuple))
                else None
            )
            sig = (
                node.domain,
                node.op_type,
                tuple(updated_inputs),
                attr_key,
                shape_key,
                node.dtype.value if node.dtype is not None else None,
                node.device,
                node.stream,
            )

            if sig in sig_map:
                canonical_id = sig_map[sig]
                alias_map[node.id] = canonical_id
                continue

            sig_map[sig] = node.id

        new_node = LogicalNode(
            id=node.id,
            op_type=node.op_type,
            domain=node.domain,
            inputs=updated_inputs,
            attributes=dict(node.attributes),
            outputs=list(node.outputs) if node.outputs is not None else None,
            shape_metadata=node.shape_metadata,
            sharding=node.sharding,
            dtype=node.dtype,
            output_specs=list(node.output_specs),
            subgraphs=dict(node.subgraphs),
            device=node.device,
            stream=node.stream,
        )
        surviving_nodes[node.id] = new_node

    new_outputs = [alias_map.get(out, out) for out in graph.outputs]

    # Rebuild edges
    new_edges: list[LogicalEdge] = []
    seen_edges: set[tuple[str, str]] = set()
    for node in surviving_nodes.values():
        for target_idx, inp in enumerate(node.inputs):
            edge_tuple = (inp, node.id)
            if edge_tuple not in seen_edges:
                seen_edges.add(edge_tuple)
                new_edges.append(
                    LogicalEdge(
                        source=inp,
                        target=node.id,
                        target_idx=target_idx,
                        value_name=inp,
                    )
                )

    return LogicalGraph(
        name=graph.name,
        nodes=surviving_nodes,
        inputs=list(graph.inputs),
        input_specs=dict(graph.input_specs),
        outputs=new_outputs,
        initializers=dict(graph.initializers),
        mesh=graph.mesh,
        edges=new_edges,
    )


def propagate_shapes_and_constants(graph: LogicalGraph) -> LogicalGraph:
    """Propagate statically computable tensor shapes and fold constant metadata.

    Evaluates Shape, Reshape, Transpose, and BroadcastInDim operations to infer
    output shapes and detect static dimension mismatches early.

    Args:
        graph (LogicalGraph): Input computational graph.

    Returns:
        LogicalGraph: Graph with updated shape_metadata across evaluated operations.

    Raises:
        ValueError: If a static shape mismatch is detected (e.g. incompatible reshape).
    """
    sorted_nodes = topological_sort(graph)
    new_nodes: dict[str, LogicalNode] = {}

    for node in sorted_nodes:
        inferred_shape = node.shape_metadata

        if node.op_type == "Shape":
            if node.inputs:
                inp_node = new_nodes.get(node.inputs[0])
                if inp_node is not None and isinstance(
                    inp_node.shape_metadata, (list, tuple)
                ):
                    inferred_shape = (len(inp_node.shape_metadata),)

        elif node.op_type == "Reshape":
            if node.inputs:
                inp_node = new_nodes.get(node.inputs[0])
                target_shape_attr = node.attributes.get("shape")
                if target_shape_attr is None:
                    target_shape_attr = node.attributes.get("new_shape")
                if (
                    inp_node is not None
                    and isinstance(inp_node.shape_metadata, (list, tuple))
                    and isinstance(target_shape_attr, (list, tuple))
                ):
                    in_numel = 1
                    all_concrete = True
                    for d in inp_node.shape_metadata:
                        if isinstance(d, int) and d > 0:
                            in_numel *= d
                        else:
                            all_concrete = False
                            break

                    resolved_dims: list[int | str] = []
                    neg_idx = -1
                    target_known_numel = 1

                    for idx, d in enumerate(target_shape_attr):
                        if d == -1:
                            if neg_idx != -1:
                                raise ValueError(
                                    f"Reshape in node '{node.id}' contains multiple -1 dimensions."
                                )
                            neg_idx = idx
                            resolved_dims.append(-1)
                        elif isinstance(d, int):
                            target_known_numel *= d
                            resolved_dims.append(d)
                        else:
                            resolved_dims.append(str(d))

                    if all_concrete and neg_idx != -1:
                        if in_numel % target_known_numel != 0:
                            raise ValueError(
                                f"Static shape mismatch in Reshape '{node.id}': input elements ({in_numel}) "
                                f"not divisible by target dimensions ({target_known_numel})."
                            )
                        resolved_dims[neg_idx] = in_numel // target_known_numel
                    elif (
                        all_concrete
                        and neg_idx == -1
                        and in_numel != target_known_numel
                    ):
                        raise ValueError(
                            f"Static shape mismatch in Reshape '{node.id}': input elements ({in_numel}) "
                            f"mismatches target elements ({target_known_numel})."
                        )

                    inferred_shape = tuple(resolved_dims)

        elif node.op_type == "Transpose":
            if node.inputs:
                inp_node = new_nodes.get(node.inputs[0])
                perm = node.attributes.get("perm")
                if (
                    inp_node is not None
                    and isinstance(inp_node.shape_metadata, (list, tuple))
                    and isinstance(perm, (list, tuple))
                    and len(perm) == len(inp_node.shape_metadata)
                ):
                    inferred_shape = tuple(
                        inp_node.shape_metadata[p]
                        for p in perm
                        if isinstance(p, int)
                        and not isinstance(p, bool)
                        and 0 <= p < len(inp_node.shape_metadata)
                    )

        elif node.op_type in ("BroadcastInDim", "broadcast_in_dim") and node.inputs:
            out_shape = node.attributes.get("output_shape")
            if out_shape is None:
                out_shape = node.attributes.get("shape")
            if isinstance(out_shape, (list, tuple)):
                inferred_shape = tuple(out_shape)

        new_node = LogicalNode(
            id=node.id,
            op_type=node.op_type,
            domain=node.domain,
            inputs=list(node.inputs),
            attributes=dict(node.attributes),
            outputs=list(node.outputs) if node.outputs is not None else None,
            shape_metadata=inferred_shape,
            sharding=node.sharding,
            dtype=node.dtype,
            output_specs=list(node.output_specs),
            subgraphs=dict(node.subgraphs),
            device=node.device,
            stream=node.stream,
        )
        new_nodes[node.id] = new_node

    return LogicalGraph(
        name=graph.name,
        nodes=new_nodes,
        inputs=list(graph.inputs),
        input_specs=dict(graph.input_specs),
        outputs=list(graph.outputs),
        initializers=dict(graph.initializers),
        mesh=graph.mesh,
        edges=list(graph.edges),
    )
