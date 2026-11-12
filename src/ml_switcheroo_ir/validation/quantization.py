"""Quantization operator validation (block_quantize and dequantize_grouped_int4)."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_ir.graph import LogicalNode
from ml_switcheroo_ir.shapes import DimensionType
from ml_switcheroo_ir.validation.core import ValidationError, ValidationLevel

__all__ = ["validate_quantization_node"]

_VALID_MX_SCALE_DTYPES: frozenset[str] = frozenset({"float8_e8m0fnu", "e8m0", "fp8_e8m0", "uint8"})
_VALID_INT4_PACKING: frozenset[str] = frozenset({"marlin", "exllama", "tensorrt_llm", "awq", "gptq"})


def _extract_validated_axis(node: LogicalNode, default_axis: int) -> tuple[int, list[ValidationError]]:
    """Extract and validate integer ``axis`` attribute, rejecting boolean values.

    Args:
        node (LogicalNode): Target quantization node.
        default_axis (int): Default axis index if ``'axis'`` is absent.

    Returns:
        tuple[int, list[ValidationError]]: ``(resolved_axis, errors)``.
    """
    if "axis" not in node.attributes:
        return default_axis, []
    axis_attr = node.attributes["axis"]
    if isinstance(axis_attr, bool) or not isinstance(axis_attr, int):
        return default_axis, [
            ValidationError(
                node_id=node.id,
                attribute="axis",
                message=f"Quantization axis must be an integer, got {axis_attr!r}.",
                level=ValidationLevel.ERROR,
            )
        ]
    return axis_attr, []


def _validate_block_quantize(node: LogicalNode) -> list[ValidationError]:
    """Validate attributes, MX scale_dtype, and axis divisibility for a ``block_quantize`` node.

    Args:
        node (LogicalNode): The ``block_quantize`` operator node.

    Returns:
        list[ValidationError]: Detected quantization validation errors.
    """
    errors: list[ValidationError] = []
    block_size = node.attributes.get("block_size")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        return [
            ValidationError(
                node_id=node.id,
                attribute="block_size",
                message=f"Quantization block_size must be a positive integer, got {block_size}.",
                level=ValidationLevel.ERROR,
            )
        ]

    mx_fmt = str(node.attributes.get("microscaling_format", node.attributes.get("format", ""))).lower()
    if mx_fmt in ("mxfp4", "mxfp8"):
        raw_scale_dt = node.attributes.get("scale_dtype")
        scale_dt = str(raw_scale_dt).lower() if raw_scale_dt is not None else ""
        if scale_dt not in _VALID_MX_SCALE_DTYPES:
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="scale_dtype",
                    message=(
                        f"Microscaling format '{mx_fmt}' requires explicit E8M0 scale_dtype "
                        f"in {sorted(_VALID_MX_SCALE_DTYPES)}, got {raw_scale_dt!r}."
                    ),
                    level=ValidationLevel.ERROR,
                )
            )

    axis, axis_errs = _extract_validated_axis(node, default_axis=-1)
    errors.extend(axis_errs)
    if not axis_errs and isinstance(node.shape_metadata, (list, tuple)) and len(node.shape_metadata) > 0:
        if -len(node.shape_metadata) <= axis < len(node.shape_metadata):
            dim = node.shape_metadata[axis]
            if isinstance(dim, int) and not isinstance(dim, bool) and dim % block_size != 0:
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
    return errors


def _validate_scales_shape_alignment(
    node: LogicalNode,
    axis: int,
    group_size: int,
    shape_meta: Sequence[DimensionType],
) -> list[ValidationError]:
    """Validate ``scales_shape`` element types and group-count alignment along ``axis``.

    Args:
        node (LogicalNode): Target ``dequantize_grouped_int4`` node.
        axis (int): Quantization axis index.
        group_size (int): Positive integer group size.
        shape_meta (Sequence[DimensionType]): Node tensor dimensions.

    Returns:
        list[ValidationError]: Detected ``scales_shape`` validation errors.
    """
    scales_shape = node.attributes.get("scales_shape")
    if not isinstance(scales_shape, (list, tuple)):
        return []
    if any(isinstance(el, bool) for el in scales_shape):
        return [
            ValidationError(
                node_id=node.id,
                attribute="scales_shape",
                message=f"Quantization scales_shape must not contain boolean elements: {scales_shape!r}.",
                level=ValidationLevel.ERROR,
            )
        ]
    if -len(shape_meta) <= axis < len(shape_meta) and -len(scales_shape) <= axis < len(scales_shape):
        dim = shape_meta[axis]
        sc_dim = scales_shape[axis]
        if (
            isinstance(dim, int)
            and not isinstance(dim, bool)
            and isinstance(sc_dim, int)
            and dim % group_size == 0
            and sc_dim != dim // group_size
        ):
            return [
                ValidationError(
                    node_id=node.id,
                    attribute="scales_shape",
                    message=(
                        f"Scales shape dimension {axis} ({sc_dim}) does not match expected groups {dim // group_size}."
                    ),
                    level=ValidationLevel.ERROR,
                )
            ]
    return []


def _validate_dequantize_grouped_int4(node: LogicalNode) -> list[ValidationError]:
    """Validate packing format, group size, and scale alignment for ``dequantize_grouped_int4``.

    Args:
        node (LogicalNode): The ``dequantize_grouped_int4`` operator node.

    Returns:
        list[ValidationError]: Detected quantization validation errors.
    """
    errors: list[ValidationError] = []
    fmt = node.attributes.get("packing_format")
    if fmt not in _VALID_INT4_PACKING:
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="packing_format",
                message=(f"Invalid packing format '{fmt}', expected one of {sorted(_VALID_INT4_PACKING)}."),
                level=ValidationLevel.ERROR,
            )
        )

    axis, axis_errs = _extract_validated_axis(node, default_axis=0)
    errors.extend(axis_errs)

    group_size = node.attributes.get("group_size")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        errors.append(
            ValidationError(
                node_id=node.id,
                attribute="group_size",
                message=f"Quantization group_size must be a positive integer, got {group_size}.",
                level=ValidationLevel.ERROR,
            )
        )
        return errors

    if axis_errs or not isinstance(node.shape_metadata, (list, tuple)) or not node.shape_metadata:
        return errors

    if -len(node.shape_metadata) <= axis < len(node.shape_metadata):
        dim = node.shape_metadata[axis]
        if isinstance(dim, int) and not isinstance(dim, bool) and dim % group_size != 0:
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="group_size",
                    message=(
                        f"Weight matrix dimension {axis} size {dim} is not evenly divisible by group_size {group_size}."
                    ),
                    level=ValidationLevel.ERROR,
                )
            )

    errors.extend(_validate_scales_shape_alignment(node, axis, group_size, node.shape_metadata))
    return errors


def validate_quantization_node(node: LogicalNode) -> list[ValidationError]:
    """Validate quantization operator attributes, alignment constraints, and packing formats.

    Checks microscaling block divisibility and E8M0 scale_dtype for block_quantize,
    and group size, packing format, and scale alignment for dequantize_grouped_int4.

    Args:
        node (LogicalNode): The quantization operator node.

    Returns:
        list[ValidationError]: Detected quantization validation errors.
    """
    clean_op = node.op_type.replace("quantization.", "").lower()
    if clean_op == "block_quantize":
        return _validate_block_quantize(node)
    if clean_op == "dequantize_grouped_int4":
        return _validate_dequantize_grouped_int4(node)
    return []
