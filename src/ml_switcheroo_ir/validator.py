"""Validator module for ml_switcheroo_ir schemas."""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ml_switcheroo_ir import LogicalGraph, LogicalMesh, LogicalNode
from ml_switcheroo_ir.schema.custom_ops import CUSTOM_OPS_REGISTRY
from ml_switcheroo_ir.schema.ghost import (
    RDNA_INSTRUCTION_PRIMITIVES,
    SASS_INSTRUCTION_PRIMITIVES,
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

        def check_axis(axis: Any) -> None:
            """Recursively check axis names against mesh shape."""
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
            elif isinstance(axis, (list, tuple)):
                for sub_axis in axis:
                    check_axis(sub_axis)

        for axis in node.sharding.axes:
            check_axis(axis)

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

        elif node.domain == "webgpu_wgsl":
            wg_size = node.attributes.get("workgroup_size")
            if wg_size is not None and (
                not isinstance(wg_size, (list, tuple))
                or len(wg_size) < 1
                or len(wg_size) > 3
                or not all(isinstance(x, int) and x > 0 for x in wg_size)
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

        if self.level == ValidationLevel.LENIENT:
            errors = [e for e in errors if e.level == ValidationLevel.ERROR]

        return errors

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
