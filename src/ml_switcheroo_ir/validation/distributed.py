"""Distributed sharding, SPMD propagation, pipeline stage, and communication cost validation."""

from __future__ import annotations

import pydantic

from ml_switcheroo_ir.distributed import PartitionAxisElement
from ml_switcheroo_ir.graph import LogicalGraph, LogicalMesh, LogicalNode
from ml_switcheroo_ir.types import DType
from ml_switcheroo_ir.validation.core import ValidationError, ValidationLevel

__all__ = [
    "CommunicationVolumeReport",
    "estimate_communication_volume",
    "estimate_graph_communication_volume",
    "validate_pipeline_and_checkpointing_graph",
    "validate_sharding_node",
    "validate_sharding_propagation_graph",
]


class CommunicationVolumeReport(pydantic.BaseModel):
    """Pydantic model reporting analytical collective communication volume.

    Attributes:
        total_volume_bytes (int): Total bytes transferred across all collective nodes.
        by_axis (dict[str, int]): Breakdown of communication volume in bytes per mesh axis.
        by_op (dict[str, int]): Breakdown of communication volume in bytes per operator type.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    total_volume_bytes: int
    by_axis: dict[str, int]
    by_op: dict[str, int]


def _resolve_itemsize_bytes(node: LogicalNode) -> float:
    """Resolve bytes per element for a node's dtype or dtype attribute.

    Args:
        node (LogicalNode): Target node.

    Returns:
        float: Bytes per element.
    """
    dtype_val = node.dtype or node.attributes.get("dtype", "float32")
    if isinstance(dtype_val, DType):
        return dtype_val.itemsize
    dtype_str = str(dtype_val).lower()
    if "64" in dtype_str:
        return 8.0
    if "16" in dtype_str:
        return 2.0
    if "8" in dtype_str:
        return 1.0
    return 4.0


def _resolve_node_element_count(node: LogicalNode) -> int:
    """Resolve total element count from ``shape_metadata`` or ``tensor_numel`` attribute.

    Args:
        node (LogicalNode): Target node.

    Returns:
        int: Element count.
    """
    if isinstance(node.shape_metadata, (list, tuple)) and len(node.shape_metadata) > 0:
        element_count = 1
        for dim in node.shape_metadata:
            if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
                element_count *= dim
        return element_count
    numel = node.attributes.get("tensor_numel", 1024)
    return int(numel) if isinstance(numel, int) and not isinstance(numel, bool) else 1024


def estimate_communication_volume(node: LogicalNode, mesh: LogicalMesh | None = None) -> int:
    """Calculate analytical communication volume in bytes for a collective communication node.

    Args:
        node (LogicalNode): The collective operation node.
        mesh (Optional[LogicalMesh]): The logical device mesh defined on the graph.

    Returns:
        int: Analytical volume in bytes transferred per device during the collective.
    """
    tensor_bytes = int(_resolve_node_element_count(node) * _resolve_itemsize_bytes(node))
    axis_name = str(node.attributes.get("mesh_axis", "data"))
    if mesh is not None and axis_name in mesh.shape:
        n_devices = mesh.shape[axis_name]
    else:
        dev_count = node.attributes.get("device_count", 2)
        n_devices = int(dev_count) if isinstance(dev_count, int) and not isinstance(dev_count, bool) else 2

    if n_devices <= 1:
        return 0

    clean_op = node.op_type.replace("collective.", "").lower()
    if clean_op == "all_reduce":
        return int(2 * ((n_devices - 1) / n_devices) * tensor_bytes)
    if clean_op in ("all_gather", "reduce_scatter", "all_to_all"):
        return int(((n_devices - 1) / n_devices) * tensor_bytes)
    return tensor_bytes


def estimate_graph_communication_volume(
    graph: LogicalGraph,
) -> CommunicationVolumeReport:
    """Aggregate analytical communication volume across all collective operators in a graph.

    Args:
        graph (LogicalGraph): The computational graph to analyze.

    Returns:
        CommunicationVolumeReport: Dictionary with total communication volume in bytes,
            and breakdowns by mesh axis and operator type.
    """
    total_bytes = 0
    by_axis: dict[str, int] = {}
    by_op: dict[str, int] = {}

    for node in graph.nodes.values():
        if node.domain in ("collective", "ml.switcheroo.collective") or node.op_type.startswith("collective.") or node.op_type in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all"):
            vol = estimate_communication_volume(node, graph.mesh)
            total_bytes += vol
            axis = str(node.attributes.get("mesh_axis", "unknown"))
            by_axis[axis] = by_axis.get(axis, 0) + vol
            by_op[node.op_type] = by_op.get(node.op_type, 0) + vol

    return CommunicationVolumeReport(
        total_volume_bytes=total_bytes,
        by_axis=by_axis,
        by_op=by_op,
    )


def _check_single_sharding_axis(
    node: LogicalNode,
    mesh: LogicalMesh,
    axis: PartitionAxisElement,
    dim_idx: int,
) -> list[ValidationError]:
    """Recursively check a single PartitionSpec axis element against mesh shape and tensor dimension.

    Args:
        node (LogicalNode): Node being validated.
        mesh (LogicalMesh): Active logical device mesh.
        axis (PartitionAxisElement): Axis element (None, str, or nested tuple/list).
        dim_idx (int): Tensor dimension index.

    Returns:
        list[ValidationError]: Detected axis validation errors.
    """
    if not isinstance(axis, (str, list, tuple)):
        return []
    if isinstance(axis, (list, tuple)):
        errs: list[ValidationError] = []
        for sub_axis in axis:
            errs.extend(_check_single_sharding_axis(node, mesh, sub_axis, dim_idx))
        return errs
    if axis not in mesh.shape:
        return [
            ValidationError(
                node_id=node.id,
                attribute="sharding",
                message=f"Sharding axis '{axis}' is not defined in LogicalMesh shape {list(mesh.shape.keys())}.",
                level=ValidationLevel.ERROR,
            )
        ]
    if isinstance(node.shape_metadata, (list, tuple)) and dim_idx < len(node.shape_metadata):
        dim_size = node.shape_metadata[dim_idx]
        axis_size = mesh.shape[axis]
        if isinstance(dim_size, int) and not isinstance(dim_size, bool) and axis_size > 0 and dim_size % axis_size != 0:
            return [
                ValidationError(
                    node_id=node.id,
                    attribute="sharding",
                    message=(f"Tensor dimension {dim_idx} size {dim_size} is not evenly divisible by mesh axis '{axis}' size {axis_size}."),
                    level=ValidationLevel.ERROR,
                )
            ]
    return []


def validate_sharding_node(node: LogicalNode, mesh: LogicalMesh | None) -> list[ValidationError]:
    """Validate that node sharding specification is compatible with the logical mesh.

    Args:
        node (LogicalNode): The node whose sharding to validate.
        mesh (Optional[LogicalMesh]): The logical mesh defined on the graph.

    Returns:
        list[ValidationError]: A list of sharding validation errors.
    """
    if node.sharding is None:
        return []

    if mesh is None:
        return [
            ValidationError(
                node_id=node.id,
                attribute="sharding",
                message="Node specifies sharding layout but graph has no LogicalMesh defined.",
                level=ValidationLevel.ERROR,
            )
        ]

    errors: list[ValidationError] = []
    if isinstance(node.shape_metadata, (list, tuple)) and len(node.shape_metadata) > 0 and len(node.sharding.axes) > len(node.shape_metadata):
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="sharding",
                message=(f"PartitionSpec rank ({len(node.sharding.axes)}) exceeds tensor rank ({len(node.shape_metadata)})."),
                level=ValidationLevel.ERROR,
            )
        )

    for idx, axis in enumerate(node.sharding.axes):
        errors.extend(_check_single_sharding_axis(node, mesh, axis, idx))

    return errors


_ELEMENTWISE_SHARDING_OPS: frozenset[str] = frozenset(
    {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Relu",
        "GELU",
        "SwiGLU",
        "Exp",
        "Log",
        "Sqrt",
        "Tanh",
        "Neg",
        "Abs",
    }
)
_REDUCTION_SHARDING_OPS: frozenset[str] = frozenset({"ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"})


def _validate_elementwise_sharding(node: LogicalNode, graph: LogicalGraph, level: ValidationLevel) -> list[ValidationError]:
    """Validate matching input/output PartitionSpecs on an elementwise node.

    Args:
        node (LogicalNode): Elementwise operator node.
        graph (LogicalGraph): Enclosing graph.
        level (ValidationLevel): Active validation severity level.

    Returns:
        list[ValidationError]: Detected sharding propagation errors.
    """
    if node.sharding is None:
        return []
    lvl = ValidationLevel.ERROR if level == ValidationLevel.STRICT else ValidationLevel.WARNING
    errors: list[ValidationError] = []
    for inp_id in node.inputs:
        inp_node = graph.nodes.get(inp_id) or graph.get_output_producer(inp_id)
        if inp_node is not None and inp_node.sharding is not None and inp_node.sharding.axes != node.sharding.axes:
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="sharding_propagation",
                    message=(f"Elementwise op '{node.id}' ({node.op_type}) has mismatched sharding between input '{inp_node.id}' ({inp_node.sharding.axes}) and output ({node.sharding.axes})."),
                    level=lvl,
                )
            )
    return errors


def _select_contract_axis(axes: tuple[PartitionAxisElement, ...], pick_second_last: bool) -> PartitionAxisElement:
    """Select the contracting PartitionSpec axis from a MatMul/Gemm operand.

    Args:
        axes (tuple[PartitionAxisElement, ...]): Operand PartitionSpec axes.
        pick_second_last (bool): True to pick ``axes[-2]`` (falling back to ``axes[0]``), False for ``axes[-1]``.

    Returns:
        PartitionAxisElement: Selected contracting axis or None.
    """
    if pick_second_last:
        return axes[-2] if len(axes) >= 2 else (axes[0] if len(axes) >= 1 else None)
    return axes[-1] if len(axes) >= 1 else None


def _validate_contraction_sharding(node: LogicalNode, graph: LogicalGraph) -> list[ValidationError]:
    """Validate matching contracted-dimension sharding on a MatMul/Gemm node.

    Args:
        node (LogicalNode): MatMul or Gemm operator node.
        graph (LogicalGraph): Enclosing graph.

    Returns:
        list[ValidationError]: Detected contraction sharding errors.
    """
    if len(node.inputs) < 2:
        return []
    node0 = graph.nodes.get(node.inputs[0]) or graph.get_output_producer(node.inputs[0])
    node1 = graph.nodes.get(node.inputs[1]) or graph.get_output_producer(node.inputs[1])
    if node0 is None or node1 is None or node0.sharding is None or node1.sharding is None:
        return []
    trans_a = bool(node.attributes.get("transA", 0))
    trans_b = bool(node.attributes.get("transB", 0))
    contract_axis_0 = _select_contract_axis(node0.sharding.axes, trans_a)
    contract_axis_1 = _select_contract_axis(node1.sharding.axes, not trans_b)
    if contract_axis_0 is not None and contract_axis_1 is not None and contract_axis_0 != contract_axis_1:
        return [
            ValidationError(
                node_id=node.id,
                attribute="sharding_propagation",
                message=(f"Contraction op '{node.id}' ({node.op_type}) has mismatched contracted dimension sharding: '{node0.id}' axis is '{contract_axis_0}', '{node1.id}' axis is '{contract_axis_1}'."),
                level=ValidationLevel.ERROR,
            )
        ]
    return []


def _validate_reduction_sharding(node: LogicalNode, graph: LogicalGraph) -> list[ValidationError]:
    """Validate that reduced input axes do not leak into output sharding on a reduction node.

    Args:
        node (LogicalNode): Reduction operator node.
        graph (LogicalGraph): Enclosing graph.

    Returns:
        list[ValidationError]: Detected reduction sharding errors.
    """
    reduced_axes = node.attributes.get("axes")
    inp_node = graph.nodes.get(node.inputs[0]) or graph.get_output_producer(node.inputs[0])
    if not isinstance(reduced_axes, (list, tuple)) or inp_node is None or inp_node.sharding is None:
        return []
    assert node.sharding is not None
    errors: list[ValidationError] = []
    for r_axis in reduced_axes:
        if isinstance(r_axis, int) and not isinstance(r_axis, bool) and 0 <= r_axis < len(inp_node.sharding.axes):
            sharded_mesh_axis = inp_node.sharding.axes[r_axis]
            if sharded_mesh_axis is not None and sharded_mesh_axis in node.sharding.axes:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="sharding_propagation",
                        message=(f"Reduction op '{node.id}' reduces sharded axis '{sharded_mesh_axis}', but output retains sharding on it."),
                        level=ValidationLevel.ERROR,
                    )
                )
    return errors


def _validate_gather_or_scatter_sharding(node: LogicalNode, graph: LogicalGraph, is_gather: bool) -> list[ValidationError]:
    """Validate dimension scaling for AllGather or ReduceScatter operations.

    Args:
        node (LogicalNode): AllGather or ReduceScatter node.
        graph (LogicalGraph): Enclosing graph.
        is_gather (bool): True for AllGather, False for ReduceScatter.

    Returns:
        list[ValidationError]: Detected dimension scaling errors.
    """
    alt_key = "gather_dim" if is_gather else "scatter_dim"
    dim_attr = node.attributes.get("axis")
    if dim_attr is None:
        dim_attr = node.attributes.get(alt_key)
    mesh_axis = node.attributes.get("mesh_axis")
    if not isinstance(dim_attr, int) or graph.mesh is None or mesh_axis is None or str(mesh_axis) not in graph.mesh.shape or not node.inputs:
        return []
    inp_node = graph.nodes.get(node.inputs[0]) or graph.get_output_producer(node.inputs[0])
    if inp_node is None or not isinstance(inp_node.shape_metadata, (list, tuple)) or not isinstance(node.shape_metadata, (list, tuple)) or not (0 <= dim_attr < len(inp_node.shape_metadata)) or dim_attr >= len(node.shape_metadata):
        return []
    in_d = inp_node.shape_metadata[dim_attr]
    out_d = node.shape_metadata[dim_attr]
    partition_factor = graph.mesh.shape[str(mesh_axis)]
    if is_gather and isinstance(in_d, int) and isinstance(out_d, int) and out_d != in_d * partition_factor:
        return [
            ValidationError(
                node_id=node.id,
                attribute="gather_dim",
                message=(f"AllGather operation '{node.id}' gathered dimension size {out_d} does not match input size {in_d} * partition factor {partition_factor}."),
                level=ValidationLevel.ERROR,
            )
        ]
    if not is_gather and isinstance(in_d, int) and isinstance(out_d, int) and in_d != out_d * partition_factor:
        return [
            ValidationError(
                node_id=node.id,
                attribute="scatter_dim",
                message=(f"ReduceScatter operation '{node.id}' scattered dimension size {out_d} does not match input size {in_d} // partition factor {partition_factor}."),
                level=ValidationLevel.ERROR,
            )
        ]
    return []


def _validate_collective_node_sharding(node: LogicalNode, graph: LogicalGraph) -> list[ValidationError]:
    """Validate collective communication operator attributes and dimension invariants.

    Args:
        node (LogicalNode): Collective operator node.
        graph (LogicalGraph): Enclosing graph.

    Returns:
        list[ValidationError]: Detected collective validation errors.
    """
    errors: list[ValidationError] = []
    if node.op_type in ("AllReduce", "all_reduce"):
        red_op = str(node.attributes.get("reduction") or node.attributes.get("op", "sum")).lower()
        if red_op not in ("sum", "min", "max", "prod"):
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="reduction",
                    message=f"AllReduce operation '{node.id}' has invalid reduction operator '{red_op}'.",
                    level=ValidationLevel.ERROR,
                )
            )
        mesh_axis = node.attributes.get("axis") or node.attributes.get("mesh_axis")
        if mesh_axis is not None and graph.mesh is not None and str(mesh_axis) not in graph.mesh.shape:
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="axis",
                    message=f"AllReduce operation '{node.id}' specifies unknown mesh axis '{mesh_axis}'.",
                    level=ValidationLevel.ERROR,
                )
            )
    elif node.op_type in ("AllGather", "all_gather"):
        errors.extend(_validate_gather_or_scatter_sharding(node, graph, is_gather=True))
    elif node.op_type in ("ReduceScatter", "reduce_scatter"):
        errors.extend(_validate_gather_or_scatter_sharding(node, graph, is_gather=False))
    elif node.op_type in ("AllToAll", "all_to_all"):
        split_axis = node.attributes.get("split_axis") or node.attributes.get("split_dim", 0)
        concat_axis = node.attributes.get("concat_axis") or node.attributes.get("concat_dim", 1)
        if isinstance(split_axis, int) and isinstance(concat_axis, int) and isinstance(node.shape_metadata, (list, tuple)):
            rank = len(node.shape_metadata)
            if not (0 <= split_axis < rank and 0 <= concat_axis < rank):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="all_to_all_axes",
                        message=(f"AllToAll operation '{node.id}' axes ({split_axis}, {concat_axis}) are out of bounds for tensor of rank {rank}."),
                        level=ValidationLevel.ERROR,
                    )
                )
    return errors


def validate_sharding_propagation_graph(
    graph: LogicalGraph,
    level: ValidationLevel = ValidationLevel.WARNING,
) -> list[ValidationError]:
    """Validate SPaDe/GSPMD sharding propagation invariants across graph operations.

    Audits elementwise operators for matching input/output partition specs,
    contraction operators for matching contracted axis bindings, and reduction
    operators for proper axis reduction without illegal sharding leaks.

    Args:
        graph (LogicalGraph): Computational graph to validate.
        level (ValidationLevel): Active validation severity level.

    Returns:
        list[ValidationError]: Detected sharding propagation violations.
    """
    errors: list[ValidationError] = []
    for node in graph.nodes.values():
        if node.op_type in _ELEMENTWISE_SHARDING_OPS:
            errors.extend(_validate_elementwise_sharding(node, graph, level))
        elif node.op_type in ("MatMul", "Gemm"):
            errors.extend(_validate_contraction_sharding(node, graph))
        elif node.op_type in _REDUCTION_SHARDING_OPS and node.sharding is not None and len(node.inputs) >= 1:
            errors.extend(_validate_reduction_sharding(node, graph))
        else:
            errors.extend(_validate_collective_node_sharding(node, graph))
    return errors


def _validate_node_checkpoint_tag(node: LogicalNode) -> list[ValidationError]:
    """Validate activation checkpointing policy tag on a node.

    Args:
        node (LogicalNode): Node to inspect.

    Returns:
        list[ValidationError]: Detected checkpoint tag errors.
    """
    tag = node.attributes.get("activation_checkpoint") if "activation_checkpoint" in node.attributes else node.attributes.get("checkpoint_policy")
    if tag is None:
        return []
    if isinstance(tag, str):
        if tag not in {"recompute", "offload", "save", "full"}:
            return [
                ValidationError(
                    node_id=node.id,
                    attribute="activation_checkpoint",
                    message=(f"Invalid activation checkpoint tag '{tag}', expected one of {{'recompute', 'offload', 'save', 'full'}}."),
                    level=ValidationLevel.ERROR,
                )
            ]
        return []
    if not isinstance(tag, bool):
        return [
            ValidationError(
                node_id=node.id,
                attribute="activation_checkpoint",
                message=f"Invalid activation checkpoint tag type: {type(tag).__name__}.",
                level=ValidationLevel.ERROR,
            )
        ]
    return []


def _validate_node_pipeline_stage(node: LogicalNode, graph: LogicalGraph) -> list[ValidationError]:
    """Validate pipeline stage ID and stage ordering transitions for a node.

    Args:
        node (LogicalNode): Node to inspect.
        graph (LogicalGraph): Enclosing graph.

    Returns:
        list[ValidationError]: Detected pipeline stage errors.
    """
    stage = node.attributes.get("pipeline_stage") if "pipeline_stage" in node.attributes else node.attributes.get("stage_id")
    if stage is None:
        return []
    if not isinstance(stage, int) or isinstance(stage, bool) or stage < 0:
        return [
            ValidationError(
                node_id=node.id,
                attribute="pipeline_stage",
                message=f"Pipeline stage ID must be a non-negative integer, got {stage}.",
                level=ValidationLevel.ERROR,
            )
        ]

    errors: list[ValidationError] = []
    for inp_id in node.inputs:
        src_node = graph.nodes.get(inp_id) or graph.get_output_producer(inp_id)
        if src_node is None:
            continue
        src_stage = src_node.attributes.get("pipeline_stage") if "pipeline_stage" in src_node.attributes else src_node.attributes.get("stage_id")
        if not isinstance(src_stage, int) or isinstance(src_stage, bool):
            continue
        if src_stage > stage:
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="pipeline_stage",
                    message=(f"Pipeline stage dependency hazard: backward edge from stage {src_stage} (node '{src_node.id}') to stage {stage} (node '{node.id}')."),
                    level=ValidationLevel.ERROR,
                )
            )
        elif stage - src_stage > 1 and not (bool(node.attributes.get("pipeline_boundary", False)) or node.op_type in ("P2P", "collective.all_to_all")):
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="pipeline_boundary",
                    message=(f"Cross-stage dataflow skipping stages ({src_stage} -> {stage}) requires an explicit boundary marker."),
                    level=ValidationLevel.WARNING,
                )
            )
    return errors


def validate_pipeline_and_checkpointing_graph(
    graph: LogicalGraph,
) -> list[ValidationError]:
    """Validate pipeline parallel stage orderings, boundary markers, and activation checkpointing tags.

    Args:
        graph (LogicalGraph): Computational graph to validate.

    Returns:
        list[ValidationError]: Detected pipeline or checkpointing violations.
    """
    errors: list[ValidationError] = []
    for node in graph.nodes.values():
        errors.extend(_validate_node_checkpoint_tag(node))
        errors.extend(_validate_node_pipeline_stage(node, graph))
    return errors
