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


def estimate_communication_volume(
    node: LogicalNode, mesh: LogicalMesh | None = None
) -> int:
    """Calculate analytical communication volume in bytes for a collective communication node.

    Args:
        node (LogicalNode): The collective operation node.
        mesh (Optional[LogicalMesh]): The logical device mesh defined on the graph.

    Returns:
        int: Analytical volume in bytes transferred per device during the collective.
    """
    element_count = 1
    if isinstance(node.shape_metadata, (list, tuple)) and len(node.shape_metadata) > 0:
        for dim in node.shape_metadata:
            if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
                element_count *= dim
    else:
        numel = node.attributes.get("tensor_numel", 1024)
        element_count = (
            int(numel)
            if isinstance(numel, int) and not isinstance(numel, bool)
            else 1024
        )

    dtype_val = node.dtype or node.attributes.get("dtype", "float32")
    if isinstance(dtype_val, DType):
        itemsize = dtype_val.itemsize
    else:
        dtype_str = str(dtype_val).lower()
        if "64" in dtype_str:
            itemsize = 8.0
        elif "16" in dtype_str:
            itemsize = 2.0
        elif "8" in dtype_str:
            itemsize = 1.0
        else:
            itemsize = 4.0
    tensor_bytes = int(element_count * itemsize)

    axis_name = str(node.attributes.get("mesh_axis", "data"))
    if mesh is not None and axis_name in mesh.shape:
        n_devices = mesh.shape[axis_name]
    else:
        dev_count = node.attributes.get("device_count", 2)
        n_devices = (
            int(dev_count)
            if isinstance(dev_count, int) and not isinstance(dev_count, bool)
            else 2
        )

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
        if (
            node.domain in ("collective", "ml.switcheroo.collective")
            or node.op_type.startswith("collective.")
            or node.op_type
            in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")
        ):
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


def validate_sharding_node(
    node: LogicalNode, mesh: LogicalMesh | None
) -> list[ValidationError]:
    """Validate that node sharding specification is compatible with the logical mesh.

    Args:
        node (LogicalNode): The node whose sharding to validate.
        mesh (Optional[LogicalMesh]): The logical mesh defined on the graph.

    Returns:
        list[ValidationError]: A list of sharding validation errors.
    """
    errors: list[ValidationError] = []
    if node.sharding is None:
        return errors

    if mesh is None:
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="sharding",
                message="Node specifies sharding layout but graph has no LogicalMesh defined.",
                level=ValidationLevel.ERROR,
            )
        )
        return errors

    if (
        isinstance(node.shape_metadata, (list, tuple))
        and len(node.shape_metadata) > 0
        and len(node.sharding.axes) > len(node.shape_metadata)
    ):
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="sharding",
                message=(
                    f"PartitionSpec rank ({len(node.sharding.axes)}) "
                    f"exceeds tensor rank ({len(node.shape_metadata)})."
                ),
                level=ValidationLevel.ERROR,
            )
        )

    def check_axis(
        axis: PartitionAxisElement,
        dim_idx: int | None = None,
    ) -> None:
        """Recursively check axis names and dimension divisibility against mesh shape.

        Args:
            axis (PartitionAxisElement): Sharding axis specification.
            dim_idx (int | None): Tensor dimension index being checked.
        """
        if axis is None:
            return
        if isinstance(axis, str):
            if axis not in mesh.shape:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="sharding",
                        message=f"Sharding axis '{axis}' is not defined in LogicalMesh shape {list(mesh.shape.keys())}.",
                        level=ValidationLevel.ERROR,
                    )
                )
            elif (
                dim_idx is not None
                and isinstance(node.shape_metadata, (list, tuple))
                and dim_idx < len(node.shape_metadata)
            ):
                dim_size = node.shape_metadata[dim_idx]
                axis_size = mesh.shape[axis]
                if (
                    isinstance(dim_size, int)
                    and not isinstance(dim_size, bool)
                    and axis_size > 0
                    and dim_size % axis_size != 0
                ):
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="sharding",
                            message=(
                                f"Tensor dimension {dim_idx} size {dim_size} is not evenly "
                                f"divisible by mesh axis '{axis}' size {axis_size}."
                            ),
                            level=ValidationLevel.ERROR,
                        )
                    )
        elif isinstance(axis, (list, tuple)):
            for sub_axis in axis:
                check_axis(sub_axis, dim_idx)

    for idx, axis in enumerate(node.sharding.axes):
        check_axis(axis, idx)

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
    elementwise_ops = {
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
    reduction_ops = {"ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin"}

    for node in graph.nodes.values():
        if node.op_type in elementwise_ops and node.sharding is not None:
            for inp_id in node.inputs:
                inp_node = graph.nodes.get(inp_id) or graph.get_output_producer(inp_id)
                if (
                    inp_node is not None
                    and inp_node.sharding is not None
                    and inp_node.sharding.axes != node.sharding.axes
                ):
                    lvl = (
                        ValidationLevel.ERROR
                        if level == ValidationLevel.STRICT
                        else ValidationLevel.WARNING
                    )
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="sharding_propagation",
                            message=(
                                f"Elementwise op '{node.id}' ({node.op_type}) has mismatched "
                                f"sharding between input '{inp_node.id}' ({inp_node.sharding.axes}) "
                                f"and output ({node.sharding.axes})."
                            ),
                            level=lvl,
                        )
                    )

        elif node.op_type in ("MatMul", "Gemm") and len(node.inputs) >= 2:
            node0 = graph.nodes.get(node.inputs[0]) or graph.get_output_producer(
                node.inputs[0]
            )
            node1 = graph.nodes.get(node.inputs[1]) or graph.get_output_producer(
                node.inputs[1]
            )
            if (
                node0 is not None
                and node1 is not None
                and node0.sharding is not None
                and node1.sharding is not None
            ):
                trans_a = bool(node.attributes.get("transA", 0))
                trans_b = bool(node.attributes.get("transB", 0))
                if trans_a:
                    contract_axis_0 = (
                        node0.sharding.axes[-2]
                        if len(node0.sharding.axes) >= 2
                        else node0.sharding.axes[0]
                        if len(node0.sharding.axes) >= 1
                        else None
                    )
                else:
                    contract_axis_0 = (
                        node0.sharding.axes[-1]
                        if len(node0.sharding.axes) >= 1
                        else None
                    )

                if trans_b:
                    contract_axis_1 = (
                        node1.sharding.axes[-1]
                        if len(node1.sharding.axes) >= 1
                        else None
                    )
                else:
                    contract_axis_1 = (
                        node1.sharding.axes[-2]
                        if len(node1.sharding.axes) >= 2
                        else node1.sharding.axes[0]
                        if len(node1.sharding.axes) >= 1
                        else None
                    )
                if (
                    contract_axis_0 is not None
                    and contract_axis_1 is not None
                    and contract_axis_0 != contract_axis_1
                ):
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="sharding_propagation",
                            message=(
                                f"Contraction op '{node.id}' ({node.op_type}) has mismatched "
                                f"contracted dimension sharding: '{node0.id}' axis is "
                                f"'{contract_axis_0}', '{node1.id}' axis is '{contract_axis_1}'."
                            ),
                            level=ValidationLevel.ERROR,
                        )
                    )

        elif (
            node.op_type in reduction_ops
            and node.sharding is not None
            and len(node.inputs) >= 1
        ):
            reduced_axes = node.attributes.get("axes")
            inp_node = graph.nodes.get(node.inputs[0]) or graph.get_output_producer(
                node.inputs[0]
            )
            if (
                isinstance(reduced_axes, (list, tuple))
                and inp_node is not None
                and inp_node.sharding is not None
            ):
                for r_axis in reduced_axes:
                    if (
                        isinstance(r_axis, int)
                        and not isinstance(r_axis, bool)
                        and 0 <= r_axis < len(inp_node.sharding.axes)
                    ):
                        sharded_mesh_axis = inp_node.sharding.axes[r_axis]
                        if (
                            sharded_mesh_axis is not None
                            and sharded_mesh_axis in node.sharding.axes
                        ):
                            errors.append(
                                ValidationError(
                                    node_id=node.id,
                                    attribute="sharding_propagation",
                                    message=(
                                        f"Reduction op '{node.id}' reduces sharded axis "
                                        f"'{sharded_mesh_axis}', but output retains sharding on it."
                                    ),
                                    level=ValidationLevel.ERROR,
                                )
                            )

        elif node.op_type in ("AllReduce", "all_reduce"):
            red_op = str(
                node.attributes.get("reduction") or node.attributes.get("op", "sum")
            ).lower()
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
            if (
                mesh_axis is not None
                and graph.mesh is not None
                and str(mesh_axis) not in graph.mesh.shape
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="axis",
                        message=f"AllReduce operation '{node.id}' specifies unknown mesh axis '{mesh_axis}'.",
                        level=ValidationLevel.ERROR,
                    )
                )

        elif node.op_type in ("AllGather", "all_gather"):
            gather_dim = node.attributes.get("axis")
            if gather_dim is None:
                gather_dim = node.attributes.get("gather_dim")
            mesh_axis = node.attributes.get("mesh_axis")
            if (
                gather_dim is not None
                and graph.mesh is not None
                and mesh_axis is not None
                and str(mesh_axis) in graph.mesh.shape
                and node.inputs
            ):
                inp_node = graph.nodes.get(node.inputs[0]) or graph.get_output_producer(
                    node.inputs[0]
                )
                partition_factor = graph.mesh.shape[str(mesh_axis)]
                if (
                    inp_node is not None
                    and isinstance(inp_node.shape_metadata, (list, tuple))
                    and isinstance(node.shape_metadata, (list, tuple))
                    and isinstance(gather_dim, int)
                    and 0 <= gather_dim < len(inp_node.shape_metadata)
                    and gather_dim < len(node.shape_metadata)
                ):
                    in_d = inp_node.shape_metadata[gather_dim]
                    out_d = node.shape_metadata[gather_dim]
                    if (
                        isinstance(in_d, int)
                        and isinstance(out_d, int)
                        and out_d != in_d * partition_factor
                    ):
                        errors.append(
                            ValidationError(
                                node_id=node.id,
                                attribute="gather_dim",
                                message=(
                                    f"AllGather operation '{node.id}' gathered dimension size {out_d} "
                                    f"does not match input size {in_d} * partition factor {partition_factor}."
                                ),
                                level=ValidationLevel.ERROR,
                            )
                        )

        elif node.op_type in ("ReduceScatter", "reduce_scatter"):
            scatter_dim = node.attributes.get("axis")
            if scatter_dim is None:
                scatter_dim = node.attributes.get("scatter_dim")
            mesh_axis = node.attributes.get("mesh_axis")
            if (
                scatter_dim is not None
                and graph.mesh is not None
                and mesh_axis is not None
                and str(mesh_axis) in graph.mesh.shape
                and node.inputs
            ):
                inp_node = graph.nodes.get(node.inputs[0]) or graph.get_output_producer(
                    node.inputs[0]
                )
                partition_factor = graph.mesh.shape[str(mesh_axis)]
                if (
                    inp_node is not None
                    and isinstance(inp_node.shape_metadata, (list, tuple))
                    and isinstance(node.shape_metadata, (list, tuple))
                    and isinstance(scatter_dim, int)
                    and 0 <= scatter_dim < len(inp_node.shape_metadata)
                    and scatter_dim < len(node.shape_metadata)
                ):
                    in_d = inp_node.shape_metadata[scatter_dim]
                    out_d = node.shape_metadata[scatter_dim]
                    if (
                        isinstance(in_d, int)
                        and isinstance(out_d, int)
                        and in_d != out_d * partition_factor
                    ):
                        errors.append(
                            ValidationError(
                                node_id=node.id,
                                attribute="scatter_dim",
                                message=(
                                    f"ReduceScatter operation '{node.id}' scattered dimension size {out_d} "
                                    f"does not match input size {in_d} // partition factor {partition_factor}."
                                ),
                                level=ValidationLevel.ERROR,
                            )
                        )

        elif node.op_type in ("AllToAll", "all_to_all"):
            split_axis = node.attributes.get("split_axis") or node.attributes.get(
                "split_dim", 0
            )
            concat_axis = node.attributes.get("concat_axis") or node.attributes.get(
                "concat_dim", 1
            )
            if (
                isinstance(split_axis, int)
                and isinstance(concat_axis, int)
                and node.shape_metadata is not None
                and isinstance(node.shape_metadata, (list, tuple))
            ):
                rank = len(node.shape_metadata)
                if not (0 <= split_axis < rank and 0 <= concat_axis < rank):
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="all_to_all_axes",
                            message=(
                                f"AllToAll operation '{node.id}' axes ({split_axis}, {concat_axis}) "
                                f"are out of bounds for tensor of rank {rank}."
                            ),
                            level=ValidationLevel.ERROR,
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
        tag = (
            node.attributes.get("activation_checkpoint")
            if "activation_checkpoint" in node.attributes
            else node.attributes.get("checkpoint_policy")
        )
        if tag is not None:
            if isinstance(tag, str):
                if tag not in {"recompute", "offload", "save", "full"}:
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="activation_checkpoint",
                            message=(
                                f"Invalid activation checkpoint tag '{tag}', "
                                f"expected one of {{'recompute', 'offload', 'save', 'full'}}."
                            ),
                            level=ValidationLevel.ERROR,
                        )
                    )
            elif not isinstance(tag, bool):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="activation_checkpoint",
                        message=f"Invalid activation checkpoint tag type: {type(tag).__name__}.",
                        level=ValidationLevel.ERROR,
                    )
                )

        stage = (
            node.attributes.get("pipeline_stage")
            if "pipeline_stage" in node.attributes
            else node.attributes.get("stage_id")
        )
        if stage is not None and (
            not isinstance(stage, int) or isinstance(stage, bool) or stage < 0
        ):
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="pipeline_stage",
                    message=f"Pipeline stage ID must be a non-negative integer, got {stage}.",
                    level=ValidationLevel.ERROR,
                )
            )
        elif isinstance(stage, int) and not isinstance(stage, bool):
            for inp_id in node.inputs:
                src_node = graph.nodes.get(inp_id) or graph.get_output_producer(inp_id)
                if src_node is not None:
                    src_stage = (
                        src_node.attributes.get("pipeline_stage")
                        if "pipeline_stage" in src_node.attributes
                        else src_node.attributes.get("stage_id")
                    )
                    if isinstance(src_stage, int) and not isinstance(src_stage, bool):
                        if src_stage > stage:
                            errors.append(
                                ValidationError(
                                    node_id=node.id,
                                    attribute="pipeline_stage",
                                    message=(
                                        f"Pipeline stage dependency hazard: backward edge from "
                                        f"stage {src_stage} (node '{src_node.id}') to "
                                        f"stage {stage} (node '{node.id}')."
                                    ),
                                    level=ValidationLevel.ERROR,
                                )
                            )
                        elif stage - src_stage > 1 and not (
                            bool(node.attributes.get("pipeline_boundary", False))
                            or node.op_type in ("P2P", "collective.all_to_all")
                        ):
                            errors.append(
                                ValidationError(
                                    node_id=node.id,
                                    attribute="pipeline_boundary",
                                    message=(
                                        f"Cross-stage dataflow skipping stages ({src_stage} -> {stage}) "
                                        f"requires an explicit boundary marker."
                                    ),
                                    level=ValidationLevel.WARNING,
                                )
                            )

    return errors
