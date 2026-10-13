"""Quantization operator validation (block_quantize and dequantize_grouped_int4)."""

from __future__ import annotations

from ml_switcheroo_ir.graph import LogicalNode
from ml_switcheroo_ir.validation.core import ValidationError, ValidationLevel

__all__ = ["validate_quantization_node"]


def validate_quantization_node(node: LogicalNode) -> list[ValidationError]:
    """Validate quantization operator attributes, alignment constraints, and packing formats.

    Checks microscaling block divisibility for block_quantize, and group size,
    packing format, and scale alignment for dequantize_grouped_int4.

    Args:
        node (LogicalNode): The quantization operator node.

    Returns:
        list[ValidationError]: Detected quantization validation errors.
    """
    errors: list[ValidationError] = []
    clean_op = node.op_type.replace("quantization.", "").lower()

    if clean_op == "block_quantize":
        block_size = node.attributes.get("block_size")
        if (
            not isinstance(block_size, int)
            or isinstance(block_size, bool)
            or block_size <= 0
        ):
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="block_size",
                    message=f"Quantization block_size must be a positive integer, got {block_size}.",
                    level=ValidationLevel.ERROR,
                )
            )
        elif (
            isinstance(node.shape_metadata, (list, tuple))
            and len(node.shape_metadata) > 0
        ):
            axis_attr = node.attributes.get("axis", -1)
            axis = (
                axis_attr
                if isinstance(axis_attr, int) and not isinstance(axis_attr, bool)
                else -1
            )
            if -len(node.shape_metadata) <= axis < len(node.shape_metadata):
                dim = node.shape_metadata[axis]
                if (
                    isinstance(dim, int)
                    and not isinstance(dim, bool)
                    and dim % block_size != 0
                ):
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="block_size",
                            message=(
                                f"Input tensor dimension {axis} size {dim} is not evenly "
                                f"divisible by block_size {block_size}."
                            ),
                            level=ValidationLevel.ERROR,
                        )
                    )

    elif clean_op == "dequantize_grouped_int4":
        valid_packing = {"marlin", "exllama", "tensorrt_llm", "awq", "gptq"}
        fmt = node.attributes.get("packing_format")
        if fmt not in valid_packing:
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="packing_format",
                    message=(
                        f"Invalid packing format '{fmt}', expected one of {sorted(valid_packing)}."
                    ),
                    level=ValidationLevel.ERROR,
                )
            )

        group_size = node.attributes.get("group_size")
        if (
            not isinstance(group_size, int)
            or isinstance(group_size, bool)
            or group_size <= 0
        ):
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="group_size",
                    message=f"Quantization group_size must be a positive integer, got {group_size}.",
                    level=ValidationLevel.ERROR,
                )
            )
        elif (
            isinstance(node.shape_metadata, (list, tuple))
            and len(node.shape_metadata) > 0
        ):
            axis_attr = node.attributes.get("axis", 0)
            axis = (
                axis_attr
                if isinstance(axis_attr, int) and not isinstance(axis_attr, bool)
                else 0
            )
            if -len(node.shape_metadata) <= axis < len(node.shape_metadata):
                dim = node.shape_metadata[axis]
                if (
                    isinstance(dim, int)
                    and not isinstance(dim, bool)
                    and dim % group_size != 0
                ):
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="group_size",
                            message=(
                                f"Weight matrix dimension {axis} size {dim} is not evenly "
                                f"divisible by group_size {group_size}."
                            ),
                            level=ValidationLevel.ERROR,
                        )
                    )

        scales_shape = node.attributes.get("scales_shape")
        if (
            isinstance(scales_shape, (list, tuple))
            and isinstance(node.shape_metadata, (list, tuple))
            and len(node.shape_metadata) > 0
        ):
            axis_attr = node.attributes.get("axis", 0)
            axis = (
                axis_attr
                if isinstance(axis_attr, int) and not isinstance(axis_attr, bool)
                else 0
            )
            if -len(node.shape_metadata) <= axis < len(node.shape_metadata) and -len(
                scales_shape
            ) <= axis < len(scales_shape):
                dim = node.shape_metadata[axis]
                sc_dim = scales_shape[axis]
                if (
                    isinstance(dim, int)
                    and isinstance(sc_dim, int)
                    and isinstance(group_size, int)
                    and group_size > 0
                    and dim % group_size == 0
                    and sc_dim != dim // group_size
                ):
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="scales_shape",
                            message=(
                                f"Scales shape dimension {axis} ({sc_dim}) does not match "
                                f"expected groups {dim // group_size}."
                            ),
                            level=ValidationLevel.ERROR,
                        )
                    )

    return errors
