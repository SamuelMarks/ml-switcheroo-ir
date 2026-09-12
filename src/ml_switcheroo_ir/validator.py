"""Validator module for ml_switcheroo_ir schemas."""

from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from ml_switcheroo_ir import LogicalGraph, LogicalMesh, LogicalNode
from ml_switcheroo_ir.schema.custom_ops import (
    COLLECTIVE_OPS_REGISTRY,
    CUSTOM_OPS_REGISTRY,
    QUANTIZATION_OPS_REGISTRY,
)
from ml_switcheroo_ir.schema.ghost import (
    RDNA3_VOPD_OPERATORS,
    RDNA_INSTRUCTION_PRIMITIVES,
    RDNA_TO_VOPD_MAP,
    SASS_INSTRUCTION_PRIMITIVES,
    SASS_PIPELINE_LATENCIES,
    WGSL_COMPUTE_BUILTINS,
    WGSL_MUTATING_OPS,
    WGSL_PRIMITIVE_SIGNATURES,
)
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY, OpSchema
from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY


def compute_levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Args:
        s1 (str): First input string.
        s2 (str): Second input string.

    Returns:
        int: Integer edit distance between s1 and s2.
    """
    if len(s1) < len(s2):
        return compute_levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


KNOWN_SYNONYMS: dict[tuple[str, str], str] = {
    ("stablehlo", "matmul"): "stablehlo.dot_general",
    ("stablehlo", "linear"): "stablehlo.dot_general",
    ("stablehlo", "conv2d"): "stablehlo.convolution",
    ("arith", "mul"): "arith.mulf",
    ("arith", "add"): "arith.addf",
    ("math", "exp2"): "math.exp",
}

CORE_MLIR_DIALECTS: dict[str, dict[str, int]] = {
    "arith": {
        "addf": 2,
        "subf": 2,
        "mulf": 2,
        "divf": 2,
        "addi": 2,
        "subi": 2,
        "muli": 2,
        "constant": 0,
        "cmpf": 2,
        "cmpi": 2,
    },
    "math": {
        "exp": 1,
        "exp2": 1,
        "log": 1,
        "sin": 1,
        "cos": 1,
        "tanh": 1,
        "sqrt": 1,
        "rsqrt": 1,
        "erf": 1,
        "absf": 1,
    },
    "tensor": {
        "extract": 2,
        "insert": 3,
        "empty": 0,
        "cast": 1,
        "dim": 2,
        "expand_shape": 1,
        "collapse_shape": 1,
    },
    "linalg": {
        "generic": 2,
        "matmul": 3,
        "conv2d": 3,
        "fill": 2,
        "batch_matmul": 3,
        "dot": 3,
    },
    "scf": {
        "for": 3,
        "if": 1,
        "while": 1,
        "yield": 1,
        "condition": 1,
    },
    "func": {
        "func": 0,
        "call": 1,
        "return": 1,
    },
}


class ValidationLevel(Enum):
    """Severity levels for validation errors and validator configuration."""

    STRICT = "STRICT"
    WARNING = "WARNING"
    LENIENT = "LENIENT"
    ERROR = "ERROR"


@dataclass
class ValidationError(Exception):
    """Represents an error found during graph or node validation.

    Attributes:
        node_id (str): The ID of the node where the error occurred.
        attribute (str): The name of the attribute involved, or a general descriptor.
        message (str): The detailed error message.
        level (ValidationLevel): The severity of the error.
    """

    node_id: str
    attribute: str
    message: str
    level: ValidationLevel = ValidationLevel.ERROR

    def __str__(self) -> str:
        """Return formatted string description of the validation error.

        Returns:
            str: Human-readable error message with severity level.
        """
        return f"[{self.level.value}] Node '{self.node_id}' attribute '{self.attribute}': {self.message}"


def _extract_vgpr_indices(node: LogicalNode) -> list[int]:
    """Extract VGPR operand integer indices from a node.

    Args:
        node (LogicalNode): The node to extract VGPR indices from.

    Returns:
        list[int]: List of VGPR register indices.
    """
    indices: list[int] = []
    for key in ("vgpr_operands", "src_vgprs"):
        val = node.attributes.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, int) and not isinstance(item, bool):
                    indices.append(item)
                else:
                    for digit in re.findall(r"\d+", str(item)):
                        indices.append(int(digit))

    for inp in node.inputs:
        for digit in re.findall(r"\d+", str(inp)):
            indices.append(int(digit))

    return indices


def _extract_dst_vgpr(node: LogicalNode) -> int | None:
    """Extract destination VGPR index from a node.

    Args:
        node (LogicalNode): The node to extract destination VGPR from.

    Returns:
        int | None: The destination VGPR index, or None if unspecified.
    """
    dst = node.attributes.get("dst_vgpr")
    if dst is None:
        dst = node.attributes.get("dst")
    if isinstance(dst, int) and not isinstance(dst, bool):
        return dst
    if dst is not None:
        for digit in re.findall(r"\d+", str(dst)):
            return int(digit)
    for out in node.outputs:
        for digit in re.findall(r"\d+", str(out)):
            return int(digit)
    return None


def _extract_sass_registers(node: LogicalNode, role: str = "src") -> list[str]:
    """Extract SASS register operands (e.g., 'R0', 'R1') from a node.

    Args:
        node (LogicalNode): The SASS instruction node.
        role (str): Operand role to extract ('src' or 'dst'). Defaults to 'src'.

    Returns:
        list[str]: Normalized register names.
    """
    regs: list[str] = []
    if role == "dst":
        dst = node.attributes.get("dst_reg")
        if dst is None:
            dst = node.attributes.get("dst")
        if isinstance(dst, list):
            for d in dst:
                regs.append(str(d).upper())
        elif dst is not None:
            regs.append(str(dst).upper())
        for out in node.outputs:
            for reg in re.findall(r"^R\d+$", str(out), re.IGNORECASE):
                regs.append(reg.upper())
    else:
        srcs = node.attributes.get("src_regs")
        if srcs is None:
            srcs = node.attributes.get("srcs")
        if isinstance(srcs, list):
            for s in srcs:
                regs.append(str(s).upper())
        elif srcs is not None:
            regs.append(str(srcs).upper())
        for inp in node.inputs:
            for reg in re.findall(r"^R\d+$", str(inp), re.IGNORECASE):
                regs.append(reg.upper())

    seen: set[str] = set()
    unique_regs: list[str] = []
    for r in regs:
        if r not in seen:
            seen.add(r)
            unique_regs.append(r)
    return unique_regs


def validate_vopd_pairing(
    opX: LogicalNode,
    opY: LogicalNode,
) -> list[ValidationError]:
    """Validate AMD RDNA3 / GFX11 dual-issue VOPD instruction pairing rules.

    Checks opcode pairing matrix, Slot X/Y compatibility, register bank conflict
    exclusivity, and destination register collisions.

    Args:
        opX (LogicalNode): Instruction for VOPD Slot X.
        opY (LogicalNode): Instruction for VOPD Slot Y.

    Returns:
        list[ValidationError]: Detected VOPD pairing violations.
    """
    errors: list[ValidationError] = []

    # 1. Domain verification
    if opX.domain != "amd_rdna":
        errors.append(
            ValidationError(
                node_id=opX.id,
                attribute="domain",
                message=f"opX domain must be 'amd_rdna', got '{opX.domain}'.",
                level=ValidationLevel.ERROR,
            )
        )
    if opY.domain != "amd_rdna":
        errors.append(
            ValidationError(
                node_id=opY.id,
                attribute="domain",
                message=f"opY domain must be 'amd_rdna', got '{opY.domain}'.",
                level=ValidationLevel.ERROR,
            )
        )

    # 2. Canonicalize opcodes to VOPD format
    canon_x = RDNA_TO_VOPD_MAP.get(opX.op_type, opX.op_type)
    canon_y = RDNA_TO_VOPD_MAP.get(opY.op_type, opY.op_type)

    # 3. Check VOPD opcode support and slot assignments
    if canon_x not in RDNA3_VOPD_OPERATORS:
        errors.append(
            ValidationError(
                node_id=opX.id,
                attribute="vopd_opcode",
                message=f"Opcode '{opX.op_type}' is not a valid RDNA3 VOPD instruction.",
                level=ValidationLevel.ERROR,
            )
        )
    elif "X" not in RDNA3_VOPD_OPERATORS[canon_x]["slots"]:
        errors.append(
            ValidationError(
                node_id=opX.id,
                attribute="vopd_slot",
                message=f"Opcode '{opX.op_type}' cannot be issued in VOPD Slot X.",
                level=ValidationLevel.ERROR,
            )
        )

    if canon_y not in RDNA3_VOPD_OPERATORS:
        errors.append(
            ValidationError(
                node_id=opY.id,
                attribute="vopd_opcode",
                message=f"Opcode '{opY.op_type}' is not a valid RDNA3 VOPD instruction.",
                level=ValidationLevel.ERROR,
            )
        )
    elif "Y" not in RDNA3_VOPD_OPERATORS[canon_y]["slots"]:
        errors.append(
            ValidationError(
                node_id=opY.id,
                attribute="vopd_slot",
                message=f"Opcode '{opY.op_type}' cannot be issued in VOPD Slot Y.",
                level=ValidationLevel.ERROR,
            )
        )

    # 4. Destination register collision
    dst_x = _extract_dst_vgpr(opX)
    dst_y = _extract_dst_vgpr(opY)
    if dst_x is not None and dst_y is not None and dst_x == dst_y:
        errors.append(
            ValidationError(
                node_id=opY.id,
                attribute="vopd_destination",
                message=f"VOPD destination conflict: opX and opY both write to VGPR v{dst_x}.",
                level=ValidationLevel.ERROR,
            )
        )

    # 5. Read port register bank conflict avoidance
    src_x = _extract_vgpr_indices(opX)
    src_y = _extract_vgpr_indices(opY)

    for bank in range(4):
        bank_x_regs = {r for r in src_x if r % 4 == bank}
        bank_y_regs = {r for r in src_y if r % 4 == bank}
        if (
            bank_x_regs
            and bank_y_regs
            and (
                bank_x_regs != bank_y_regs
                or len(bank_x_regs) > 1
                or len(bank_y_regs) > 1
            )
        ):
            errors.append(
                ValidationError(
                    node_id=opX.id,
                    attribute="vopd_bank_conflict",
                    message=(
                        f"VOPD register bank conflict on bank {bank}: "
                        f"opX reads VGPRs {sorted(bank_x_regs)} and opY reads VGPRs {sorted(bank_y_regs)}."
                    ),
                    level=ValidationLevel.ERROR,
                )
            )

    return errors


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

    dtype_val = getattr(node, "dtype", None) or node.attributes.get("dtype", "float32")
    dtype_str = str(dtype_val).lower()
    if "64" in dtype_str:
        itemsize = 8
    elif "16" in dtype_str:
        itemsize = 2
    elif "8" in dtype_str:
        itemsize = 1
    else:
        itemsize = 4
    tensor_bytes = element_count * itemsize

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
) -> dict[str, Any]:
    """Aggregate analytical communication volume across all collective operators in a graph.

    Args:
        graph (LogicalGraph): The computational graph to analyze.

    Returns:
        dict[str, Any]: Dictionary with total communication volume in bytes,
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

    return {
        "total_volume_bytes": total_bytes,
        "by_axis": by_axis,
        "by_op": by_op,
    }


class Validator:
    """Validates LogicalGraph and LogicalNode instances against schemas."""

    def __init__(
        self,
        registry: dict[str, OpSchema] | None = None,
        custom_registry: dict[str, OpSchema] | None = None,
        stablehlo_registry: dict[str, OpSchema] | None = None,
        level: ValidationLevel = ValidationLevel.WARNING,
        strict: bool = False,
        grounding_validator: GroundingValidator | None = None,
    ) -> None:
        """Initialize the validator with configurable registries and severity thresholds.

        Args:
            registry (Optional[Dict[str, OpSchema]]): The operator registry to use.
                Defaults to the built-in ONNX_REGISTRY.
            custom_registry (Optional[Dict[str, OpSchema]]): The custom operator registry.
                Defaults to the built-in CUSTOM_OPS_REGISTRY.
            stablehlo_registry (Optional[Dict[str, OpSchema]]): The StableHLO operator registry.
                Defaults to the built-in STABLEHLO_REGISTRY.
            level (ValidationLevel): The validation severity threshold (default: ValidationLevel.WARNING).
            strict (bool): Convenience flag; if True, sets level to ValidationLevel.STRICT.
            grounding_validator (Optional[GroundingValidator]): Optional grounding validator for snapshot verification.
        """
        if registry is None:
            self.registry = ONNX_REGISTRY
        else:
            self.registry = registry

        if custom_registry is None:
            self.custom_registry = CUSTOM_OPS_REGISTRY
        else:
            self.custom_registry = custom_registry

        if stablehlo_registry is None:
            self.stablehlo_registry = STABLEHLO_REGISTRY
        else:
            self.stablehlo_registry = stablehlo_registry

        self.collective_registry = COLLECTIVE_OPS_REGISTRY
        self.quantization_registry = QUANTIZATION_OPS_REGISTRY

        if strict:
            self.level = ValidationLevel.STRICT
        elif level == ValidationLevel.ERROR:
            self.level = ValidationLevel.LENIENT
        else:
            self.level = level

        self.grounding_validator = grounding_validator

    def _get_schema(self, node: LogicalNode) -> OpSchema | None:
        """Retrieve the schema for a node across registered domains.

        Args:
            node (LogicalNode): The node to inspect.

        Returns:
            Optional[OpSchema]: The schema if found, or None.
        """
        if node.domain == "ai.onnx":
            return self.registry.get(node.op_type)
        if node.domain == "ml.switcheroo.custom":
            return self.registry.get(node.op_type) or self.custom_registry.get(
                node.op_type
            )
        if node.domain == "stablehlo":
            return self.registry.get(node.op_type) or self.stablehlo_registry.get(
                node.op_type
            )
        if node.domain in ("collective", "ml.switcheroo.collective"):
            return self.registry.get(node.op_type) or self.collective_registry.get(
                node.op_type
            )
        if node.domain in ("quantization", "ml.switcheroo.quantization"):
            return self.registry.get(node.op_type) or self.quantization_registry.get(
                node.op_type
            )
        return self.registry.get(node.op_type)

    def validate_kind(self, node: LogicalNode) -> list[ValidationError]:
        """Validate that the node's kind exists in the registry for its domain.

        Args:
            node (LogicalNode): The node to validate.

        Returns:
            List[ValidationError]: A list of errors found.
        """
        errors: list[ValidationError] = []
        if node.domain == "ai.onnx":
            if node.op_type not in self.registry:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain == "ml.switcheroo.custom":
            if (
                node.op_type not in self.custom_registry
                and node.op_type not in self.registry
            ):
                lvl = (
                    ValidationLevel.ERROR
                    if self.level == ValidationLevel.STRICT
                    else ValidationLevel.WARNING
                )
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=lvl,
                    )
                )
        elif node.domain == "stablehlo":
            if (
                node.op_type not in self.stablehlo_registry
                and node.op_type not in self.registry
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain in ("collective", "ml.switcheroo.collective"):
            if (
                node.op_type not in self.collective_registry
                and node.op_type not in self.registry
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in collective registry.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain in ("quantization", "ml.switcheroo.quantization"):
            if (
                node.op_type not in self.quantization_registry
                and node.op_type not in self.registry
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in quantization registry.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain in CORE_MLIR_DIALECTS:
            dialect_ops = CORE_MLIR_DIALECTS[node.domain]
            if node.op_type not in dialect_ops and node.op_type not in self.registry:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in MLIR dialect '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
            else:
                expected_inputs = dialect_ops.get(node.op_type)
                if expected_inputs is not None and len(node.inputs) < expected_inputs:
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="inputs",
                            message=f"MLIR op '{node.domain}.{node.op_type}' expects at least {expected_inputs} operands, got {len(node.inputs)}.",
                            level=ValidationLevel.ERROR,
                        )
                    )
        elif node.domain == "amd_rdna":
            if (
                node.op_type not in RDNA_INSTRUCTION_PRIMITIVES
                and node.op_type not in self.registry
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain == "nvidia_sass":
            if (
                node.op_type not in SASS_INSTRUCTION_PRIMITIVES
                and node.op_type not in self.registry
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain == "webgpu_wgsl":
            if (
                node.op_type not in WGSL_PRIMITIVE_SIGNATURES
                and node.op_type not in self.registry
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif (
            node.domain not in ("ai.custom", "custom")
            and node.op_type not in self.registry
        ):
            # Any unrecognized domain outside custom and not in registry
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="domain",
                    message=f"Unrecognized domain '{node.domain}'.",
                    level=ValidationLevel.ERROR,
                )
            )
        return errors

    def validate_required_attributes(self, node: LogicalNode) -> list[ValidationError]:
        """Validate that all required attributes for the node's kind are present.

        Args:
            node (LogicalNode): The node to validate.

        Returns:
            List[ValidationError]: A list of errors found.
        """
        errors: list[ValidationError] = []
        schema = self._get_schema(node)
        if schema is None:
            return errors

        for attr_name, attr_schema in schema.attributes.items():
            if attr_schema.required and attr_name not in node.attributes:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute=attr_name,
                        message=f"Required attribute '{attr_name}' is missing.",
                        level=ValidationLevel.ERROR,
                    )
                )
        return errors

    def validate_attribute_types(self, node: LogicalNode) -> list[ValidationError]:
        """Validate that attributes have the correct types according to the schema.

        Args:
            node (LogicalNode): The node to validate.

        Returns:
            List[ValidationError]: A list of errors found.
        """
        errors: list[ValidationError] = []
        schema = self._get_schema(node)
        if schema is None:
            return errors

        for key, value in node.attributes.items():
            if key not in schema.attributes:
                lvl = (
                    ValidationLevel.ERROR
                    if self.level == ValidationLevel.STRICT
                    else ValidationLevel.WARNING
                )
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute=key,
                        message=f"Attribute '{key}' is not recognized for '{node.op_type}'.",
                        level=lvl,
                    )
                )
                continue

            attr_schema = schema.attributes[key]
            expected_type = attr_schema.type

            # Simplified type checking mapping
            valid = True
            if expected_type == "int" and not isinstance(value, int):
                valid = False
            elif expected_type == "float" and not isinstance(value, float):
                # allow int to substitute for float in python json parsing safely
                if not isinstance(value, (float, int)):
                    valid = False
            elif (
                expected_type == "str"
                and not isinstance(value, str)
                or expected_type == "dict"
                and not isinstance(value, dict)
            ):
                valid = False
            elif expected_type == "List[int]":
                if not isinstance(value, list) or not all(
                    isinstance(x, int) for x in value
                ):
                    valid = False
            elif expected_type == "List[float]":
                if not isinstance(value, list) or not all(
                    isinstance(x, (float, int)) for x in value
                ):
                    valid = False
            elif expected_type == "List[str]":
                if not isinstance(value, list) or not all(
                    isinstance(x, str) for x in value
                ):
                    valid = False
            elif expected_type == "bool" and not isinstance(value, bool):
                valid = False

            if not valid:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute=key,
                        message=f"Attribute '{key}' has invalid type. Expected {expected_type}, got {type(value).__name__}.",
                        level=ValidationLevel.ERROR,
                    )
                )

        return errors

    def validate_sharding(
        self, node: LogicalNode, mesh: LogicalMesh | None
    ) -> list[ValidationError]:
        """Validate that node sharding specification is compatible with the logical mesh.

        Args:
            node (LogicalNode): The node whose sharding to validate.
            mesh (Optional[LogicalMesh]): The logical mesh defined on the graph.

        Returns:
            List[ValidationError]: A list of sharding validation errors.
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

        # Validate partition spec rank against tensor shape metadata rank if known
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

        def check_axis(axis: Any, dim_idx: int | None = None) -> None:
            """Recursively check axis names and dimension divisibility against mesh shape."""
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

    def populate_defaults(self, node: LogicalNode) -> None:
        """Inject missing optional attributes with default values from the schema.

        Args:
            node (LogicalNode): The node to mutate.
        """
        schema = self._get_schema(node)
        if schema is None:
            return

        for attr_name, attr_schema in schema.attributes.items():
            if (
                not attr_schema.required
                and attr_schema.default is not None
                and attr_name not in node.attributes
            ):
                node.attributes[attr_name] = attr_schema.default

    def validate_isa_instruction(self, node: LogicalNode) -> list[ValidationError]:
        """Validate GPU accelerator ISA instructions (AMD RDNA, NVIDIA SASS, and WebGPU WGSL).

        Args:
            node (LogicalNode): The node representing an ISA instruction or WGSL primitive.

        Returns:
            List[ValidationError]: A list of detected validation errors or warnings.
        """
        errors: list[ValidationError] = []

        if node.domain == "amd_rdna":
            wavefront = node.attributes.get("wavefront_size")
            if wavefront is not None and wavefront not in (32, 64):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="wavefront_size",
                        message=f"AMD RDNA wavefront_size must be 32 or 64, got {wavefront}.",
                        level=ValidationLevel.ERROR,
                    )
                )

            reg_classes = node.attributes.get("register_classes")
            if isinstance(reg_classes, dict):
                valid_classes = {"VGPR", "SGPR", "AGPR"}
                for op_name, rclass in reg_classes.items():
                    if rclass not in valid_classes:
                        errors.append(
                            ValidationError(
                                node_id=node.id,
                                attribute=f"register_classes.{op_name}",
                                message=f"Invalid RDNA register class '{rclass}', expected one of {valid_classes}.",
                                level=ValidationLevel.ERROR,
                            )
                        )

            vgpr_indices = node.attributes.get("vgpr_operands")
            if isinstance(vgpr_indices, list) and len(vgpr_indices) >= 2:
                banks = [idx % 4 for idx in vgpr_indices if isinstance(idx, int)]
                if len(banks) != len(set(banks)):
                    lvl = (
                        ValidationLevel.ERROR
                        if self.level == ValidationLevel.STRICT
                        else ValidationLevel.WARNING
                    )
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="vgpr_operands",
                            message=f"VGPR bank conflict detected in instruction '{node.op_type}': operands map to same bank ({banks}).",
                            level=lvl,
                        )
                    )

        elif node.domain == "nvidia_sass":
            barrier = node.attributes.get("barrier_predicate")
            if (
                barrier is not None
                and isinstance(barrier, str)
                and not (
                    barrier.startswith(("@P", "@!P")) or barrier in ("@PT", "@!PT")
                )
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="barrier_predicate",
                        message=f"Invalid SASS barrier predicate '{barrier}'.",
                        level=ValidationLevel.ERROR,
                    )
                )

            sync = node.attributes.get("warp_sync")
            if sync is not None and sync not in (
                "sync",
                "yield",
                "diverge",
                "arrive",
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="warp_sync",
                        message=f"Invalid warp synchronization marker '{sync}'.",
                        level=ValidationLevel.ERROR,
                    )
                )

            mem_space = node.attributes.get("memory_space")
            if mem_space is not None:
                valid_spaces = {"global", "shared", "constant", "local"}
                if mem_space not in valid_spaces:
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="memory_space",
                            message=f"Invalid SASS memory space '{mem_space}', expected one of {valid_spaces}.",
                            level=ValidationLevel.ERROR,
                        )
                    )

            # Stall count validation (0 to 15 clock cycles)
            stall = (
                node.attributes.get("stall_count")
                if "stall_count" in node.attributes
                else node.attributes.get("stall")
            )
            if stall is not None and (
                not isinstance(stall, int)
                or isinstance(stall, bool)
                or not (0 <= stall <= 15)
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="stall_count",
                        message=f"SASS instruction stall count must be an integer between 0 and 15, got {stall}.",
                        level=ValidationLevel.ERROR,
                    )
                )

            # Yield flag validation
            yield_val = (
                node.attributes.get("yield_flag")
                if "yield_flag" in node.attributes
                else node.attributes.get("yield")
            )
            if yield_val is not None:
                if isinstance(yield_val, str):
                    if yield_val not in ("Y", "-", "yield", "noyield"):
                        errors.append(
                            ValidationError(
                                node_id=node.id,
                                attribute="yield_flag",
                                message=f"Invalid SASS yield flag '{yield_val}'. Expected 'Y' or '-'.",
                                level=ValidationLevel.ERROR,
                            )
                        )
                elif not isinstance(yield_val, bool):
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="yield_flag",
                            message=f"Invalid SASS yield flag '{yield_val}'.",
                            level=ValidationLevel.ERROR,
                        )
                    )

            # Read / Write / Wait Barrier masks
            for barrier_key in ("read_barrier", "write_barrier", "wait_barrier_mask"):
                b_val = node.attributes.get(barrier_key)
                if b_val is not None:
                    if isinstance(b_val, int) and not isinstance(b_val, bool):
                        if not (0 <= b_val <= 63):
                            errors.append(
                                ValidationError(
                                    node_id=node.id,
                                    attribute=barrier_key,
                                    message=f"Invalid SASS barrier mask '{b_val}'. Must be in range 0..63.",
                                    level=ValidationLevel.ERROR,
                                )
                            )
                    elif isinstance(b_val, list):
                        if not all(
                            isinstance(x, int)
                            and not isinstance(x, bool)
                            and 0 <= x <= 5
                            for x in b_val
                        ):
                            errors.append(
                                ValidationError(
                                    node_id=node.id,
                                    attribute=barrier_key,
                                    message=f"Invalid SASS barrier list '{b_val}'. Entries must be barrier indices 0..5.",
                                    level=ValidationLevel.ERROR,
                                )
                            )
                    elif isinstance(b_val, str):
                        if len(b_val) > 6 or not set(b_val).issubset(
                            {"0", "1", "2", "3", "4", "5", "-", " "}
                        ):
                            errors.append(
                                ValidationError(
                                    node_id=node.id,
                                    attribute=barrier_key,
                                    message=f"Invalid SASS barrier mask string '{b_val}'.",
                                    level=ValidationLevel.ERROR,
                                )
                            )
                    else:
                        errors.append(
                            ValidationError(
                                node_id=node.id,
                                attribute=barrier_key,
                                message=f"Invalid SASS barrier mask type for '{barrier_key}'.",
                                level=ValidationLevel.ERROR,
                            )
                        )

        elif node.domain == "webgpu_wgsl":
            wg_size = node.attributes.get("workgroup_size")
            if wg_size is not None and (
                not isinstance(wg_size, (list, tuple))
                or len(wg_size) < 1
                or len(wg_size) > 3
                or not all(
                    isinstance(x, int) and not isinstance(x, bool) and x > 0
                    for x in wg_size
                )
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="workgroup_size",
                        message=f"WGSL @workgroup_size must be 1 to 3 positive integers, got {wg_size}.",
                        level=ValidationLevel.ERROR,
                    )
                )

            qualifier = node.attributes.get("address_space")
            if qualifier is not None:
                valid_qualifiers = {
                    "uniform",
                    "storage, read",
                    "storage, read_write",
                    "workgroup",
                    "private",
                    "function",
                }
                if qualifier not in valid_qualifiers:
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="address_space",
                            message=f"Invalid WGSL address space qualifier '{qualifier}', expected one of {valid_qualifiers}.",
                            level=ValidationLevel.ERROR,
                        )
                    )

            # Uniform buffer 16-byte alignment rules
            if qualifier == "uniform":
                align = (
                    node.attributes.get("struct_alignment")
                    if "struct_alignment" in node.attributes
                    else node.attributes.get("alignment")
                )
                if align is not None and (
                    not isinstance(align, int)
                    or isinstance(align, bool)
                    or align % 16 != 0
                ):
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="struct_alignment",
                            message=f"WGSL uniform buffer struct alignment must be a multiple of 16 bytes, got {align}.",
                            level=ValidationLevel.ERROR,
                        )
                    )

                stride = node.attributes.get("array_stride")
                if stride is not None and (
                    not isinstance(stride, int)
                    or isinstance(stride, bool)
                    or stride % 16 != 0
                ):
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="array_stride",
                            message=f"WGSL uniform buffer array stride must be a multiple of 16 bytes, got {stride}.",
                            level=ValidationLevel.ERROR,
                        )
                    )

            # Compute entrypoint builtins validation
            builtin = node.attributes.get("builtin")
            if builtin is not None and builtin not in WGSL_COMPUTE_BUILTINS:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="builtin",
                        message=f"Invalid WGSL compute builtin '{builtin}', expected one of {sorted(WGSL_COMPUTE_BUILTINS)}.",
                        level=ValidationLevel.ERROR,
                    )
                )

            # Storage buffer access mode checks against mutating operation kinds
            if qualifier in ("storage, read", "storage_read") and (
                node.op_type in WGSL_MUTATING_OPS
                or bool(node.attributes.get("is_mutating", False))
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="address_space",
                        message=f"Mutating operation '{node.op_type}' is illegal on read-only storage buffer with address space '{qualifier}'.",
                        level=ValidationLevel.ERROR,
                    )
                )

        return errors

    def validate_quantization(self, node: LogicalNode) -> list[ValidationError]:
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

            # Check scale alignment if provided
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
                if -len(node.shape_metadata) <= axis < len(
                    node.shape_metadata
                ) and -len(scales_shape) <= axis < len(scales_shape):
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

    def validate_node(
        self, node: LogicalNode, mesh: LogicalMesh | None = None
    ) -> list[ValidationError]:
        """Validate a single LogicalNode against schemas and configured severity level.

        Args:
            node (LogicalNode): The node to validate.
            mesh (Optional[LogicalMesh]): Optional logical device mesh for sharding validation.

        Returns:
            List[ValidationError]: Aggregated list of validation errors and warnings for the node.
        """
        errors: list[ValidationError] = []
        errors.extend(self.validate_kind(node))
        errors.extend(self.validate_required_attributes(node))
        errors.extend(self.validate_attribute_types(node))
        errors.extend(self.validate_sharding(node, mesh))
        if node.domain in ("amd_rdna", "nvidia_sass", "webgpu_wgsl"):
            errors.extend(self.validate_isa_instruction(node))
        if (
            node.domain in ("quantization", "ml.switcheroo.quantization")
            or node.op_type.startswith("quantization.")
            or node.op_type in ("block_quantize", "dequantize_grouped_int4")
        ):
            errors.extend(self.validate_quantization(node))

        # Check shape metadata in STRICT mode
        if self.level == ValidationLevel.STRICT and node.shape_metadata is None:
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="shape_metadata",
                    message=f"Node '{node.id}' missing required shape_metadata in STRICT validation mode.",
                    level=ValidationLevel.ERROR,
                )
            )

        # Check grounding if grounding validator is configured
        if self.grounding_validator is not None:
            grounding_errs = self.grounding_validator.validate_grounding(node)
            for g_err in grounding_errs:
                if self.level == ValidationLevel.STRICT:
                    errors.append(
                        ValidationError(
                            node_id=g_err.node_id,
                            attribute=g_err.attribute,
                            message=g_err.message,
                            level=ValidationLevel.ERROR,
                        )
                    )
                elif self.level == ValidationLevel.WARNING:
                    errors.append(
                        ValidationError(
                            node_id=g_err.node_id,
                            attribute=g_err.attribute,
                            message=g_err.message,
                            level=ValidationLevel.WARNING,
                        )
                    )

        self.populate_defaults(node)

        # In LENIENT mode, filter out non-fatal WARNINGs
        if self.level == ValidationLevel.LENIENT:
            errors = [e for e in errors if e.level == ValidationLevel.ERROR]

        return errors

    def validate_graph(self, graph: LogicalGraph) -> list[ValidationError]:
        """Validate all nodes, sharding, and edges in a LogicalGraph.

        Args:
            graph (LogicalGraph): The graph to validate.

        Returns:
            List[ValidationError]: A list of all aggregated errors.
        """
        errors: list[ValidationError] = []

        # Validate nodes
        for node in graph.nodes.values():
            errors.extend(self.validate_node(node, graph.mesh))

        # Validate edges
        errors.extend(self.validate_edges(graph))

        # Validate SASS scoreboarding if SASS instructions are present
        sass_nodes = [n for n in graph.nodes.values() if n.domain == "nvidia_sass"]
        if sass_nodes:
            errors.extend(self.validate_sass_scoreboarding(sass_nodes))

        # Validate SPaDe / GSPMD sharding propagation
        errors.extend(self.validate_sharding_propagation(graph))

        # Validate pipeline stages and activation checkpointing
        errors.extend(self.validate_pipeline_and_checkpointing(graph))

        if self.level == ValidationLevel.LENIENT:
            errors = [e for e in errors if e.level == ValidationLevel.ERROR]

        return errors

    def validate_vopd_pairing(
        self, opX: LogicalNode, opY: LogicalNode
    ) -> list[ValidationError]:
        """Validate AMD RDNA3 / GFX11 dual-issue VOPD instruction pairing rules.

        Checks opcode pairing matrix, Slot X/Y compatibility, register bank conflict
        exclusivity, and destination register collisions.

        Args:
            opX (LogicalNode): The instruction proposed for VOPD Slot X.
            opY (LogicalNode): The instruction proposed for VOPD Slot Y.

        Returns:
            list[ValidationError]: Detected VOPD pairing violations.
        """
        return validate_vopd_pairing(opX, opY)

    def validate_sass_scoreboarding(
        self, nodes: Sequence[LogicalNode]
    ) -> list[ValidationError]:
        """Validate NVIDIA SASS scoreboard dependency latencies between consecutive instructions.

        Args:
            nodes (Sequence[LogicalNode]): Sequential list of SASS instructions to audit.

        Returns:
            list[ValidationError]: Detected scoreboard hazards or dependency violations.
        """
        errors: list[ValidationError] = []
        pending_writes: dict[str, tuple[str, str, int, int | None]] = {}

        for node in nodes:
            if node.domain != "nvidia_sass":
                continue

            # Check if any wait barrier clears pending writes
            wait_mask = node.attributes.get("wait_barrier_mask")
            cleared_barriers: set[int] = set()
            if isinstance(wait_mask, int) and not isinstance(wait_mask, bool):
                for b_idx in range(6):
                    if (wait_mask >> b_idx) & 1:
                        cleared_barriers.add(b_idx)
            elif isinstance(wait_mask, list):
                cleared_barriers.update(
                    b
                    for b in wait_mask
                    if isinstance(b, int) and not isinstance(b, bool)
                )

            if cleared_barriers:
                pending_writes = {
                    r: info
                    for r, info in pending_writes.items()
                    if info[3] not in cleared_barriers
                }

            # Check consumed registers
            src_regs = _extract_sass_registers(node, role="src")
            for reg in src_regs:
                if reg in pending_writes:
                    prod_id, prod_op, rem_cycles, _ = pending_writes[reg]
                    lvl = (
                        ValidationLevel.ERROR
                        if self.level == ValidationLevel.STRICT
                        else ValidationLevel.WARNING
                    )
                    errors.append(
                        ValidationError(
                            node_id=node.id,
                            attribute="scoreboarding",
                            message=(
                                f"Scoreboard dependency hazard: consumer '{node.id}' ({node.op_type}) "
                                f"reads register '{reg}' produced by '{prod_id}' ({prod_op}) "
                                f"before pipeline latency satisfied ({rem_cycles} stall cycles remaining) "
                                f"without barrier synchronization."
                            ),
                            level=lvl,
                        )
                    )

            # Register writes from this node
            dst_regs = _extract_sass_registers(node, role="dst")
            producer_latency = SASS_PIPELINE_LATENCIES.get(node.op_type, 4)
            wb_attr = node.attributes.get("write_barrier")
            wb_id: int | None = (
                wb_attr
                if isinstance(wb_attr, int) and not isinstance(wb_attr, bool)
                else None
            )
            for reg in dst_regs:
                pending_writes[reg] = (
                    node.id,
                    node.op_type,
                    producer_latency,
                    wb_id,
                )

            # Deduct stall count of this instruction from all pending writes
            stall = node.attributes.get("stall_count")
            if stall is None:
                stall = node.attributes.get("stall", 0)
            stall_cycles = (
                stall if isinstance(stall, int) and not isinstance(stall, bool) else 0
            )

            # Advance cycles
            new_pending: dict[str, tuple[str, str, int, int | None]] = {}
            for r, (p_id, p_op, rem, wb) in pending_writes.items():
                updated_rem = max(0, rem - stall_cycles)
                if updated_rem > 0:
                    new_pending[r] = (p_id, p_op, updated_rem, wb)
            pending_writes = new_pending

        return errors

    def validate_sharding_propagation(
        self, graph: LogicalGraph
    ) -> list[ValidationError]:
        """Validate SPaDe/GSPMD sharding propagation invariants across graph operations.

        Audits elementwise operators for matching input/output partition specs,
        contraction operators for matching contracted axis bindings, and reduction
        operators for proper axis reduction without illegal sharding leaks.

        Args:
            graph (LogicalGraph): Computational graph to validate.

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
            # 1. Elementwise operations: input shardings must match each other and output sharding
            if node.op_type in elementwise_ops and node.sharding is not None:
                for inp_id in node.inputs:
                    inp_node = graph.nodes.get(inp_id)
                    if (
                        inp_node is not None
                        and inp_node.sharding is not None
                        and inp_node.sharding.axes != node.sharding.axes
                    ):
                        lvl = (
                            ValidationLevel.ERROR
                            if self.level == ValidationLevel.STRICT
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

            # 2. Contraction operations (e.g. MatMul): contracted dimensions must match
            elif node.op_type in ("MatMul", "Gemm") and len(node.inputs) >= 2:
                node0 = graph.nodes.get(node.inputs[0])
                node1 = graph.nodes.get(node.inputs[1])
                if (
                    node0 is not None
                    and node1 is not None
                    and node0.sharding is not None
                    and node1.sharding is not None
                ):
                    contract_axis_0 = (
                        node0.sharding.axes[-1]
                        if len(node0.sharding.axes) >= 1
                        else None
                    )
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

            # 3. Reduction operations: reduced axis cannot remain sharded in output
            elif (
                node.op_type in reduction_ops
                and node.sharding is not None
                and len(node.inputs) >= 1
            ):
                reduced_axes = node.attributes.get("axes")
                inp_node = graph.nodes.get(node.inputs[0])
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

        return errors

    def validate_pipeline_and_checkpointing(
        self, graph: LogicalGraph
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
            if stage is not None and (type(stage) is not int or stage < 0):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="pipeline_stage",
                        message=f"Pipeline stage ID must be a non-negative integer, got {stage}.",
                        level=ValidationLevel.ERROR,
                    )
                )
            elif type(stage) is int:
                for inp_id in node.inputs:
                    src_node = graph.nodes.get(inp_id)
                    if src_node is not None:
                        src_stage = (
                            src_node.attributes.get("pipeline_stage")
                            if "pipeline_stage" in src_node.attributes
                            else src_node.attributes.get("stage_id")
                        )
                        if type(src_stage) is int:
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

    def estimate_communication_volume(
        self, node: LogicalNode, mesh: LogicalMesh | None = None
    ) -> int:
        """Calculate analytical communication volume in bytes for a collective communication node.

        Args:
            node (LogicalNode): The collective operation node.
            mesh (Optional[LogicalMesh]): The logical device mesh defined on the graph.

        Returns:
            int: Analytical volume in bytes transferred per device during the collective.
        """
        return estimate_communication_volume(node, mesh)

    def estimate_graph_communication_volume(
        self, graph: LogicalGraph
    ) -> dict[str, Any]:
        """Aggregate analytical communication volume across all collective operators in a graph.

        Args:
            graph (LogicalGraph): The computational graph to analyze.

        Returns:
            dict[str, Any]: Dictionary with total communication volume in bytes,
                and breakdowns by mesh axis and operator type.
        """
        return estimate_graph_communication_volume(graph)

    def validate(
        self,
        target: LogicalGraph | LogicalNode,
        raise_on_error: bool = False,
    ) -> list[ValidationError]:
        """Validate a LogicalGraph or LogicalNode instance against configured validation level.

        Args:
            target (Union[LogicalGraph, LogicalNode]): Graph or node instance to validate.
            raise_on_error (bool): If True, raises the first fatal ValidationError encountered.

        Returns:
            List[ValidationError]: List of validation errors and warnings.

        Raises:
            ValidationError: If raise_on_error is True and a fatal ValidationError is found.
        """
        if isinstance(target, LogicalGraph):
            errors = self.validate_graph(target)
        else:
            errors = self.validate_node(target)

        if raise_on_error:
            fatal_errors = [e for e in errors if e.level == ValidationLevel.ERROR]
            if fatal_errors:
                raise fatal_errors[0]

        return errors

    def validate_edges(self, graph: LogicalGraph) -> list[ValidationError]:
        """Validate that all node inputs and graph edges exist in the graph.

        Args:
            graph (LogicalGraph): The graph to validate.

        Returns:
            List[ValidationError]: A list of edge validation errors.
        """
        errors: list[ValidationError] = []
        node_ids = set(graph.nodes.keys())

        # Validate inputs
        for node_id, node in graph.nodes.items():
            for inp in node.inputs:
                if inp not in node_ids and graph.get_output_producer(inp) is None:
                    errors.append(
                        ValidationError(
                            node_id=node_id,
                            attribute="inputs",
                            message=f"Node input '{inp}' does not exist.",
                            level=ValidationLevel.ERROR,
                        )
                    )

        return errors


@dataclass
class GroundingAuditReport:
    """Audit report detailing graph grounding against ground-truth framework snapshots.

    Attributes:
        total_nodes (int): Total number of nodes in graph.
        grounded_count (int): Number of nodes successfully grounded against snapshot.
        ungrounded_count (int): Number of ungrounded or hallucinated nodes.
        hallucination_score (float): Percentage of nodes that are ungrounded (0.0 to 1.0).
        diagnostics (List[ValidationError]): Diagnostic error reports.
    """

    total_nodes: int
    grounded_count: int
    ungrounded_count: int
    hallucination_score: float
    diagnostics: list[ValidationError]


DEFAULT_SNAPSHOT_DIR = os.environ.get("ML_FRAMEWORK_SNAPSHOTS_DIR") or os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "ml-framework-snapshots",
        "src",
        "ml_framework_snapshots",
        "snapshots",
    )
)


class GroundingValidator(Validator):
    """Validator that audits graphs strictly against external ground-truth snapshot manifests."""

    def __init__(
        self,
        snapshot_manifest: dict[str, Any] | list[Any] | str | None = None,
        registry: dict[str, OpSchema] | None = None,
        use_default_if_none: bool = False,
    ) -> None:
        """Initialize GroundingValidator with external snapshot file, directory, list, or dict.

        Args:
            snapshot_manifest (Union[Dict[str, Any], List[Any], str, None]): Snapshot dictionary,
                list of records, file path, directory path, or collection.
            registry (Optional[Dict[str, OpSchema]]): Base operator registry.
            use_default_if_none (bool): If True and snapshot_manifest is None, load from DEFAULT_SNAPSHOT_DIR.
        """
        super().__init__(registry=registry)
        self.grounded_symbols: dict[str, dict[str, Any]] = {}

        if snapshot_manifest is not None:
            self._ingest_manifest_target(snapshot_manifest)
        elif use_default_if_none and os.path.isdir(DEFAULT_SNAPSHOT_DIR):
            self._load_directory(DEFAULT_SNAPSHOT_DIR)

    def _ingest_manifest_target(self, target: Any) -> None:
        """Ingest snapshot manifest from diverse target types.

        Args:
            target (Any): Directory path, file path, dict, or list.
        """
        if isinstance(target, str):
            if os.path.isdir(target):
                self._load_directory(target)
            elif os.path.isfile(target):
                self._load_file(target)
        elif isinstance(target, dict):
            self._load_snapshot_data(target)
        elif isinstance(target, list):
            if target and all(
                isinstance(item, str)
                or (
                    isinstance(item, dict)
                    and not ("api_path" in item or "name" in item or "mnemonic" in item)
                )
                for item in target
            ):
                for item in target:
                    self._ingest_manifest_target(item)
            else:
                self._load_snapshot_data(target)

    def _load_file(self, filepath: str) -> None:
        """Load and parse a single snapshot file (.json or .json.gz).

        Args:
            filepath (str): Path to snapshot file.
        """
        try:
            if filepath.endswith(".gz"):
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            self._load_snapshot_data(data)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
            return

    def _load_directory(self, dirpath: str) -> None:
        """Recursively search and load all .json and .json.gz snapshot files in directory.

        Args:
            dirpath (str): Path to root directory.
        """
        for root, _, files in os.walk(dirpath):
            for fname in sorted(files):
                if fname.endswith((".json", ".json.gz")):
                    self._load_file(os.path.join(root, fname))

    def _register_symbol_record(
        self, item: dict[str, Any], default_key: str | None = None
    ) -> None:
        """Register an individual symbol record under primary and short keys.

        Args:
            item (Dict[str, Any]): Symbol definition record.
            default_key (Optional[str]): Fallback key if neither api_path, name, nor mnemonic is present.
        """
        api_path = item.get("api_path")
        name = item.get("name")
        mnemonic = item.get("mnemonic")

        keys = [k for k in [api_path, name, mnemonic, default_key] if k]
        for k in keys:
            self.grounded_symbols[k] = item

        if api_path and "." in api_path:
            unqualified = api_path.split(".")[-1]
            if unqualified not in self.grounded_symbols:
                self.grounded_symbols[unqualified] = item

    def _load_snapshot_data(self, data: dict[str, Any] | list[Any]) -> None:
        """Load snapshot entries from categorized, flat, or top-level list formats.

        Args:
            data (Union[Dict[str, Any], List[Any]]): Parsed snapshot data.
        """
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    self._register_symbol_record(item)
        elif isinstance(data, dict):
            if "categories" in data and isinstance(data["categories"], dict):
                for cat_list in data["categories"].values():
                    if isinstance(cat_list, list):
                        for item in cat_list:
                            if isinstance(item, dict):
                                self._register_symbol_record(item)
                    elif isinstance(cat_list, dict):
                        for item in cat_list.values():
                            if isinstance(item, dict):
                                self._register_symbol_record(item)
            else:
                for k, v in data.items():
                    if isinstance(v, dict):
                        self._register_symbol_record(v, default_key=k)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                self._register_symbol_record(item)

    def _find_best_symbol_match(self, op_type: str, domain: str) -> str | None:
        """Find the closest matching symbol in the snapshot within an edit distance threshold.

        Args:
            op_type (str): The ungrounded operation type.
            domain (str): The operation domain.

        Returns:
            Optional[str]: Closest matching symbol name, or None.
        """
        synonym = KNOWN_SYNONYMS.get((domain, op_type))
        if synonym and (
            synonym in self.grounded_symbols
            or synonym.split(".")[-1] in self.grounded_symbols
        ):
            return synonym

        best_match: str | None = None
        best_dist = 4
        full_candidate = f"{domain}.{op_type}" if domain else op_type
        domain_prefix = f"{domain}." if domain else ""

        if domain_prefix:
            for sym in self.grounded_symbols:
                if sym.startswith(domain_prefix):
                    unqual = sym[len(domain_prefix) :]
                    d = min(
                        compute_levenshtein(op_type, unqual),
                        compute_levenshtein(full_candidate, sym),
                    )
                    if d < best_dist:
                        best_dist = d
                        best_match = sym

        if best_match is not None:
            return best_match

        for sym in self.grounded_symbols:
            d = compute_levenshtein(op_type, sym)
            if d < best_dist:
                best_dist = d
                best_match = sym

        return best_match

    def _find_best_attr_match(self, attr_key: str, candidates: set[str]) -> str | None:
        """Find closest matching attribute or parameter name.

        Args:
            attr_key (str): Hallucinated attribute key.
            candidates (Set[str]): Valid known attribute/parameter names.

        Returns:
            Optional[str]: Closest candidate name if edit distance <= 3, else None.
        """
        best_match: str | None = None
        best_dist = 4
        for cand in candidates:
            d = compute_levenshtein(attr_key, cand)
            if d < best_dist:
                best_dist = d
                best_match = cand
        return best_match

    def validate_grounding(self, node: LogicalNode) -> list[ValidationError]:
        """Check if node corresponds to a grounded symbol in the snapshot.

        Args:
            node (LogicalNode): The node to inspect.

        Returns:
            List[ValidationError]: Diagnostic errors if symbol is ungrounded.
        """
        errors: list[ValidationError] = []
        full_path = f"{node.domain}.{node.op_type}"

        match = self.grounded_symbols.get(full_path) or self.grounded_symbols.get(
            node.op_type
        )
        if match is None:
            suggestion = self._find_best_symbol_match(node.op_type, node.domain)
            msg = f"Ungrounded symbol '{node.op_type}' in domain '{node.domain}'. Symbol not found in framework snapshot."
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="kind",
                    message=msg,
                    level=ValidationLevel.ERROR,
                )
            )
        else:
            known_params: set[str] = set()
            for p in match.get("params", []):
                if isinstance(p, dict) and "name" in p:
                    known_params.add(p["name"])

            attrs = match.get("attributes")
            if isinstance(attrs, dict):
                known_params.update(attrs.keys())
            elif isinstance(attrs, list):
                for a in attrs:
                    if isinstance(a, dict) and "name" in a:
                        known_params.add(a["name"])
                    elif isinstance(a, str):
                        known_params.add(a)

            operands = match.get("operands")
            if isinstance(operands, list):
                for op in operands:
                    if isinstance(op, dict) and "name" in op:
                        known_params.add(op["name"])
                    elif isinstance(op, str):
                        known_params.add(op)

            if known_params:
                for attr_key in node.attributes:
                    if attr_key not in known_params:
                        attr_suggestion = self._find_best_attr_match(
                            attr_key, known_params
                        )
                        msg = f"Ungrounded attribute '{attr_key}' on '{node.op_type}'. Allowed attributes/params: {sorted(known_params)}."
                        if attr_suggestion:
                            msg += f" Did you mean '{attr_suggestion}'?"
                        errors.append(
                            ValidationError(
                                node_id=node.id,
                                attribute=attr_key,
                                message=msg,
                                level=ValidationLevel.ERROR,
                            )
                        )
        return errors

    def audit_graph(self, graph: LogicalGraph) -> GroundingAuditReport:
        """Audit an entire graph for anti-hallucination grounding against snapshots.

        Args:
            graph (LogicalGraph): Graph to audit.

        Returns:
            GroundingAuditReport: Comprehensive grounding report.
        """
        diagnostics: list[ValidationError] = []
        grounded_count = 0
        total_nodes = len(graph.nodes)

        for node in graph.nodes.values():
            node_errors = self.validate_grounding(node)
            if node_errors:
                diagnostics.extend(node_errors)
            else:
                grounded_count += 1

        ungrounded_count = total_nodes - grounded_count
        score = (ungrounded_count / total_nodes) if total_nodes > 0 else 0.0

        return GroundingAuditReport(
            total_nodes=total_nodes,
            grounded_count=grounded_count,
            ungrounded_count=ungrounded_count,
            hallucination_score=round(score, 4),
            diagnostics=diagnostics,
        )


def audit_graph_grounding(
    graph: LogicalGraph,
    snapshots_path: str | list[Any] | dict[str, Any] | None = None,
) -> GroundingAuditReport:
    """Audit a LogicalGraph against framework snapshots.

    Args:
        graph (LogicalGraph): Logical graph to audit.
        snapshots_path (Optional[Union[str, List[Any], Dict[str, Any]]]): Path to snapshot manifest,
            directory, or loaded dictionary/list.

    Returns:
        GroundingAuditReport: Resulting grounding audit report.
    """
    resolved_path = (
        DEFAULT_SNAPSHOT_DIR
        if snapshots_path is None and os.path.isdir(DEFAULT_SNAPSHOT_DIR)
        else snapshots_path
    )
    validator = GroundingValidator(snapshot_manifest=resolved_path)
    return validator.audit_graph(graph)
