"""Graph optimization and canonical transformation passes for ML-Switcheroo IR.

Provides pure-IR transformations including Dead Code Elimination (DCE),
Common Subexpression Elimination (CSE), and Shape Propagation / Constant Folding.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

from ml_switcheroo_ir import (
    LogicalEdge,
    LogicalGraph,
    LogicalNode,
    topological_sort,
)
from ml_switcheroo_ir.shapes import (
    DimensionType,
    SymBinaryOp,
    SymbolicConstraintTracker,
    SymConst,
    SymNode,
    broadcast_shapes,
    matmul_shape,
    normalize_axis,
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


def _propagate_unary_op(
    node: LogicalNode, new_nodes: dict[str, LogicalNode]
) -> tuple[DimensionType, ...] | None:
    """Propagate shape for elementwise unary operations.

    Args:
        node (LogicalNode): Current unary node.
        new_nodes (dict[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Propagated shape or None if unshaped.
    """
    if node.inputs:
        inp = new_nodes.get(node.inputs[0])
        if inp is not None and isinstance(inp.shape_metadata, (list, tuple)):
            return tuple(inp.shape_metadata)
    return None


def _propagate_binary_op(
    node: LogicalNode,
    new_nodes: dict[str, LogicalNode],
    tracker: SymbolicConstraintTracker,
) -> tuple[DimensionType, ...] | None:
    """Propagate broadcasted output shape for elementwise binary operations.

    Args:
        node (LogicalNode): Current binary node.
        new_nodes (dict[str, LogicalNode]): Map of processed preceding nodes.
        tracker (SymbolicConstraintTracker): Constraint tracker for symbolic unification.

    Returns:
        tuple[DimensionType, ...] | None: Broadcasted shape or None if inputs are unshaped.
    """
    if len(node.inputs) >= 2:
        inp1 = new_nodes.get(node.inputs[0])
        inp2 = new_nodes.get(node.inputs[1])
        if (
            inp1 is not None
            and inp2 is not None
            and isinstance(inp1.shape_metadata, (list, tuple))
            and isinstance(inp2.shape_metadata, (list, tuple))
        ):
            return broadcast_shapes(
                inp1.shape_metadata, inp2.shape_metadata, tracker=tracker
            )
    return None


def _propagate_matmul_op(
    node: LogicalNode, new_nodes: dict[str, LogicalNode]
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for matrix multiplication operations.

    Args:
        node (LogicalNode): Current MatMul or BatchMatMul node.
        new_nodes (dict[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Resulting matmul shape or None if unshaped.
    """
    if len(node.inputs) >= 2:
        inp1 = new_nodes.get(node.inputs[0])
        inp2 = new_nodes.get(node.inputs[1])
        if (
            inp1 is not None
            and inp2 is not None
            and isinstance(inp1.shape_metadata, (list, tuple))
            and isinstance(inp2.shape_metadata, (list, tuple))
        ):
            return matmul_shape(inp1.shape_metadata, inp2.shape_metadata)
    return None


def _propagate_gemm_op(
    node: LogicalNode,
    new_nodes: dict[str, LogicalNode],
    tracker: SymbolicConstraintTracker,
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for General Matrix Multiply (Gemm) operations.

    Args:
        node (LogicalNode): Current Gemm node.
        new_nodes (dict[str, LogicalNode]): Map of processed preceding nodes.
        tracker (SymbolicConstraintTracker): Constraint tracker for broadcast unification.

    Returns:
        tuple[DimensionType, ...] | None: Gemm output shape or None if inputs are unshaped.

    Raises:
        ValueError: If inner matrix dimensions do not match.
    """
    if len(node.inputs) >= 2:
        inp_a = new_nodes.get(node.inputs[0])
        inp_b = new_nodes.get(node.inputs[1])
        if (
            inp_a is not None
            and inp_b is not None
            and isinstance(inp_a.shape_metadata, (list, tuple))
            and isinstance(inp_b.shape_metadata, (list, tuple))
            and len(inp_a.shape_metadata) == 2
            and len(inp_b.shape_metadata) == 2
        ):
            trans_a = bool(node.attributes.get("transA", 0))
            trans_b = bool(node.attributes.get("transB", 0))
            s_a = inp_a.shape_metadata
            s_b = inp_b.shape_metadata

            m = s_a[1] if trans_a else s_a[0]
            k_a = s_a[0] if trans_a else s_a[1]
            k_b = s_b[1] if trans_b else s_b[0]
            n = s_b[0] if trans_b else s_b[1]

            from ml_switcheroo_ir.shapes import SymbolicSolver

            if not SymbolicSolver.is_consistent(k_a, k_b):
                raise ValueError(
                    f"Gemm inner contraction dimension mismatch: {k_a} vs {k_b} in node '{node.id}'."
                )

            out_shape: tuple[DimensionType, ...] = (m, n)
            if len(node.inputs) > 2 and node.inputs[2] in new_nodes:
                inp_c = new_nodes[node.inputs[2]]
                if isinstance(inp_c.shape_metadata, (list, tuple)):
                    out_shape = broadcast_shapes(
                        out_shape, inp_c.shape_metadata, tracker=tracker
                    )
            return out_shape
    return None


def _propagate_squeeze_op(
    node: LogicalNode, new_nodes: dict[str, LogicalNode]
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Squeeze operations removing specified or size-1 axes.

    Args:
        node (LogicalNode): Current Squeeze node.
        new_nodes (dict[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Squeezed shape or None if unshaped.
    """
    if node.inputs:
        inp = new_nodes.get(node.inputs[0])
        if inp is not None and isinstance(inp.shape_metadata, (list, tuple)):
            s = tuple(inp.shape_metadata)
            axes = node.attributes.get("axes")
            if axes is None:
                axes = node.attributes.get("axis")
            if axes is None and len(node.inputs) > 1 and node.inputs[1] in new_nodes:
                axes_node = new_nodes[node.inputs[1]]
                if "value" in axes_node.attributes:
                    axes = axes_node.attributes["value"]

            if axes is not None:
                raw_list = [axes] if isinstance(axes, int) else list(axes)  # type: ignore[arg-type]
                norm_axes = set(
                    normalize_axis(
                        [int(a) for a in raw_list if isinstance(a, (int, str))],
                        len(s),
                    )
                )
                return tuple(d for i, d in enumerate(s) if i not in norm_axes)
            return tuple(d for d in s if d != 1 and str(d) != "1")
    return None


def _propagate_unsqueeze_op(
    node: LogicalNode, new_nodes: dict[str, LogicalNode]
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Unsqueeze operations inserting size-1 dimensions.

    Args:
        node (LogicalNode): Current Unsqueeze node.
        new_nodes (dict[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Unsqueezed shape or None if unshaped.
    """
    if node.inputs:
        inp = new_nodes.get(node.inputs[0])
        if inp is not None and isinstance(inp.shape_metadata, (list, tuple)):
            s = list(inp.shape_metadata)
            axes = node.attributes.get("axes")
            if axes is None:
                axes = node.attributes.get("axis")
            if axes is None and len(node.inputs) > 1 and node.inputs[1] in new_nodes:
                axes_node = new_nodes[node.inputs[1]]
                if "value" in axes_node.attributes:
                    axes = axes_node.attributes["value"]

            if axes is not None:
                raw_list = [axes] if isinstance(axes, int) else list(axes)  # type: ignore[arg-type]
                raw_axes = [int(a) for a in raw_list if isinstance(a, (int, str))]
                target_rank = len(s) + len(raw_axes)
                norm_axes = sorted([normalize_axis(a, target_rank) for a in raw_axes])
                for ax in norm_axes:
                    s.insert(ax, 1)
                return tuple(s)
            return tuple(s)
    return None


def _propagate_flatten_op(
    node: LogicalNode, new_nodes: dict[str, LogicalNode]
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Flatten operations collapsing into a 2D tensor.

    Args:
        node (LogicalNode): Current Flatten node.
        new_nodes (dict[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Flattened 2D shape or None if unshaped.
    """
    if node.inputs:
        inp = new_nodes.get(node.inputs[0])
        if inp is not None and isinstance(inp.shape_metadata, (list, tuple)):
            s = tuple(inp.shape_metadata)
            axis_val = node.attributes.get("axis", 1)
            axis = int(axis_val) if isinstance(axis_val, (int, str)) else 1
            norm_axis = normalize_axis(axis, len(s))

            def _prod(dims: Sequence[DimensionType]) -> DimensionType:
                """Compute cumulative product across sequence of dimensions.

                Args:
                    dims (Sequence[DimensionType]): Dimension sizes.

                Returns:
                    DimensionType: Product dimension.
                """
                total: DimensionType = 1
                for d in dims:
                    if isinstance(d, int) and isinstance(total, int):
                        total = total * d
                    else:
                        total = SymBinaryOp(
                            "*", SymNode.to_node(total), SymNode.to_node(d)
                        ).simplify()
                return total

            return (_prod(s[:norm_axis]), _prod(s[norm_axis:]))
    return None


def _propagate_concat_op(
    node: LogicalNode, new_nodes: dict[str, LogicalNode]
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Concat operations joining tensors along an axis.

    Args:
        node (LogicalNode): Current Concat node.
        new_nodes (dict[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Concatenated shape or None if unshaped.

    Raises:
        ValueError: If input tensor ranks do not match.
    """
    shapes: list[tuple[DimensionType, ...]] = []
    for inp_id in node.inputs:
        inp_n = new_nodes.get(inp_id)
        if inp_n is not None and isinstance(inp_n.shape_metadata, (list, tuple)):
            shapes.append(tuple(inp_n.shape_metadata))

    if len(shapes) == len(node.inputs) and len(shapes) > 0:
        rank = len(shapes[0])
        axis_val = node.attributes.get("axis", 0)
        axis = int(axis_val) if isinstance(axis_val, (int, str)) else 0
        norm_axis = normalize_axis(axis, rank)

        concat_dim: DimensionType = 0
        for s in shapes:
            if len(s) != rank:
                raise ValueError(
                    f"Concat rank mismatch in node '{node.id}': expected rank {rank}, got {len(s)}."
                )
            d = s[norm_axis]
            if isinstance(d, int) and isinstance(concat_dim, int):
                concat_dim += d
            else:
                concat_dim = SymBinaryOp(
                    "+", SymNode.to_node(concat_dim), SymNode.to_node(d)
                ).simplify()

        out_s = list(shapes[0])
        out_s[norm_axis] = concat_dim
        return tuple(out_s)
    return None


def _propagate_split_op(
    node: LogicalNode, new_nodes: dict[str, LogicalNode]
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Split operations slicing tensor along an axis.

    Args:
        node (LogicalNode): Current Split node.
        new_nodes (dict[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Shape of first output partition or None.
    """
    if node.inputs:
        inp = new_nodes.get(node.inputs[0])
        if inp is not None and isinstance(inp.shape_metadata, (list, tuple)):
            s = tuple(inp.shape_metadata)
            axis_val = node.attributes.get("axis", 0)
            axis = int(axis_val) if isinstance(axis_val, (int, str)) else 0
            norm_axis = normalize_axis(axis, len(s))
            split = node.attributes.get("split")
            num_outputs_val = (
                node.attributes.get("num_outputs") or len(node.outputs or []) or 2
            )
            num_outputs = (
                int(num_outputs_val) if isinstance(num_outputs_val, (int, str)) else 2
            )
            first_split: DimensionType
            if isinstance(split, (list, tuple)) and len(split) > 0:
                first_split = split[0]  # type: ignore[assignment]
            else:
                dim_val = s[norm_axis]
                if isinstance(dim_val, int):
                    first_split = dim_val // num_outputs
                else:
                    first_split = SymBinaryOp(
                        "//", SymNode.to_node(dim_val), SymConst(num_outputs)
                    ).simplify()
            out_s = list(s)
            out_s[norm_axis] = first_split
            return tuple(out_s)
    return None


def _propagate_slice_op(
    node: LogicalNode, new_nodes: dict[str, LogicalNode]
) -> tuple[DimensionType, ...] | None:
    """Propagate output shape for Slice operations given starts, ends, axes, steps.

    Args:
        node (LogicalNode): Current Slice node.
        new_nodes (dict[str, LogicalNode]): Map of processed preceding nodes.

    Returns:
        tuple[DimensionType, ...] | None: Sliced shape or None if unshaped.
    """
    if node.inputs:
        inp = new_nodes.get(node.inputs[0])
        if inp is not None and isinstance(inp.shape_metadata, (list, tuple)):
            s = list(inp.shape_metadata)
            starts = node.attributes.get("starts")
            ends = node.attributes.get("ends")
            axes = node.attributes.get("axes")
            steps = node.attributes.get("steps")
            if isinstance(starts, (list, tuple)) and isinstance(ends, (list, tuple)):
                axes_list = (
                    list(axes)
                    if isinstance(axes, (list, tuple))
                    else list(range(len(starts)))
                )
                steps_list = (
                    list(steps)
                    if isinstance(steps, (list, tuple))
                    else [1] * len(starts)
                )
                for st, en, ax, sp in zip(starts, ends, axes_list, steps_list):
                    ax_int = int(ax) if isinstance(ax, (int, str)) else 0
                    norm_ax = normalize_axis(ax_int, len(s))
                    dim_len = s[norm_ax]
                    if (
                        isinstance(st, int)
                        and isinstance(en, int)
                        and isinstance(sp, int)
                    ):
                        if isinstance(dim_len, int):
                            st_norm = st + dim_len if st < 0 else st
                            en_norm = en + dim_len if en < 0 else en
                            st_norm = max(0, min(dim_len, st_norm))
                            en_norm = max(0, min(dim_len, en_norm))
                            s[norm_ax] = (
                                max(0, (en_norm - st_norm + sp - 1) // sp)
                                if sp > 0
                                else 0
                            )
                        else:
                            s[norm_ax] = (
                                max(0, (en - st + sp - 1) // sp) if sp > 0 else en - st
                            )
                return tuple(s)
    return None


def propagate_shapes_and_constants(graph: LogicalGraph) -> LogicalGraph:
    """Propagate statically computable tensor shapes and fold constant metadata.

    Evaluates Shape, Reshape, Transpose, BroadcastInDim, elementwise unary/binary,
    matrix multiplication, Gemm, Squeeze, Unsqueeze, Flatten, Concat, Split, and Slice
    operations to infer output shapes and detect static dimension mismatches early.

    Args:
        graph (LogicalGraph): Input computational graph.

    Returns:
        LogicalGraph: Graph with updated shape_metadata across evaluated operations.

    Raises:
        ValueError: If a static shape mismatch is detected (e.g. incompatible reshape).
    """
    sorted_nodes = topological_sort(graph)
    new_nodes: dict[str, LogicalNode] = {}
    tracker = SymbolicConstraintTracker()

    unary_ops: set[str] = {
        "Abs",
        "Neg",
        "Exp",
        "Log",
        "Sqrt",
        "Relu",
        "Sigmoid",
        "Tanh",
        "Sin",
        "Cos",
        "Floor",
        "Ceil",
        "Round",
        "Identity",
        "relu",
        "sigmoid",
        "tanh",
    }
    binary_ops: set[str] = {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Mod",
        "Equal",
        "Greater",
        "Less",
        "Max",
        "Min",
        "add",
        "sub",
        "mul",
        "div",
    }
    matmul_ops: set[str] = {
        "MatMul",
        "BatchMatMul",
        "matmul",
        "batch_matmul",
    }

    for node in sorted_nodes:
        inferred_shape = node.shape_metadata

        if node.op_type in unary_ops:
            u_shape = _propagate_unary_op(node, new_nodes)
            if u_shape is not None:
                inferred_shape = u_shape

        elif node.op_type in binary_ops:
            b_shape = _propagate_binary_op(node, new_nodes, tracker)
            if b_shape is not None:
                inferred_shape = b_shape

        elif node.op_type in matmul_ops:
            m_shape = _propagate_matmul_op(node, new_nodes)
            if m_shape is not None:
                inferred_shape = m_shape

        elif node.op_type in ("Gemm", "gemm"):
            g_shape = _propagate_gemm_op(node, new_nodes, tracker)
            if g_shape is not None:
                inferred_shape = g_shape

        elif node.op_type in ("Squeeze", "squeeze"):
            sq_shape = _propagate_squeeze_op(node, new_nodes)
            if sq_shape is not None:
                inferred_shape = sq_shape

        elif node.op_type in ("Unsqueeze", "unsqueeze"):
            unsq_shape = _propagate_unsqueeze_op(node, new_nodes)
            if unsq_shape is not None:
                inferred_shape = unsq_shape

        elif node.op_type in ("Flatten", "flatten"):
            fl_shape = _propagate_flatten_op(node, new_nodes)
            if fl_shape is not None:
                inferred_shape = fl_shape

        elif node.op_type in ("Concat", "concat"):
            cat_shape = _propagate_concat_op(node, new_nodes)
            if cat_shape is not None:
                inferred_shape = cat_shape

        elif node.op_type in ("Split", "split"):
            sp_shape = _propagate_split_op(node, new_nodes)
            if sp_shape is not None:
                inferred_shape = sp_shape

        elif node.op_type in ("Slice", "slice"):
            sl_shape = _propagate_slice_op(node, new_nodes)
            if sl_shape is not None:
                inferred_shape = sl_shape

        elif node.op_type == "Shape":
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

                    resolved_dims: list[DimensionType] = []
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
