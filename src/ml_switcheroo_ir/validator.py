"""Validator module for ml_switcheroo_ir schemas."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ml_switcheroo_ir import LogicalGraph, LogicalMesh, LogicalNode
from ml_switcheroo_ir.schema.custom_ops import CUSTOM_OPS_REGISTRY
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY, OpSchema
from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY

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
        "log": 1,
        "sin": 1,
        "cos": 1,
        "sqrt": 1,
        "tanh": 1,
        "absf": 1,
    },
    "tensor": {
        "extract": 2,
        "insert": 3,
        "empty": 0,
        "cast": 1,
        "dim": 2,
    },
    "linalg": {
        "matmul": 3,
        "generic": 2,
        "fill": 2,
        "dot": 3,
    },
    "scf": {
        "for": 3,
        "while": 1,
        "if": 1,
        "yield": 1,
    },
    "func": {
        "func": 0,
        "call": 1,
        "return": 1,
    },
}


class ValidationLevel(Enum):
    """Severity levels for validation errors."""

    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ValidationError:
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
    level: ValidationLevel


class Validator:
    """Validates LogicalGraph and LogicalNode instances against schemas."""

    def __init__(
        self,
        registry: dict[str, OpSchema] | None = None,
        custom_registry: dict[str, OpSchema] | None = None,
        stablehlo_registry: dict[str, OpSchema] | None = None,
    ) -> None:
        """Initialize the validator.

        Args:
            registry (Dict[str, OpSchema], optional): The operator registry to use.
                Defaults to the built-in ONNX_REGISTRY.
            custom_registry (Dict[str, OpSchema], optional): The custom operator registry.
                Defaults to the built-in CUSTOM_OPS_REGISTRY.
            stablehlo_registry (Dict[str, OpSchema], optional): The StableHLO operator registry.
                Defaults to the built-in STABLEHLO_REGISTRY.
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
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
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
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute=key,
                        message=f"Attribute '{key}' is not recognized for '{node.op_type}'.",
                        level=ValidationLevel.WARNING,
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
            errors.extend(self.validate_kind(node))
            errors.extend(self.validate_required_attributes(node))
            errors.extend(self.validate_attribute_types(node))
            errors.extend(self.validate_sharding(node, graph.mesh))
            self.populate_defaults(node)

        # Validate edges
        errors.extend(self.validate_edges(graph))

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


class GroundingValidator(Validator):
    """Validator that audits graphs strictly against external ground-truth snapshot manifests."""

    def __init__(
        self,
        snapshot_manifest: dict[str, Any] | str | None = None,
        registry: dict[str, OpSchema] | None = None,
    ) -> None:
        """Initialize GroundingValidator with external snapshot file or dict.

        Args:
            snapshot_manifest (Union[Dict[str, Any], str], optional): Snapshot dictionary or JSON file path.
            registry (Dict[str, OpSchema], optional): Base operator registry.
        """
        super().__init__(registry=registry)
        self.grounded_symbols: dict[str, dict[str, Any]] = {}

        if isinstance(snapshot_manifest, str):
            if os.path.exists(snapshot_manifest):
                with open(snapshot_manifest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._load_snapshot_data(data)
        elif isinstance(snapshot_manifest, dict):
            self._load_snapshot_data(snapshot_manifest)

    def _load_snapshot_data(self, data: dict[str, Any]) -> None:
        """Load snapshot entries from categorized or flat snapshot format."""
        if "categories" in data and isinstance(data["categories"], dict):
            for cat_list in data["categories"].values():
                if isinstance(cat_list, list):
                    for item in cat_list:
                        if isinstance(item, dict):
                            key = item.get("api_path") or item.get("name")
                            if key:
                                self.grounded_symbols[key] = item
        else:
            for k, v in data.items():
                if isinstance(v, dict):
                    self.grounded_symbols[k] = v

    def validate_grounding(self, node: LogicalNode) -> list[ValidationError]:
        """Check if node corresponds to a grounded symbol in the snapshot.

        Args:
            node (LogicalNode): The node to inspect.

        Returns:
            List[ValidationError]: Diagnostic errors if symbol is ungrounded.
        """
        errors: list[ValidationError] = []
        full_path = f"{node.domain}.{node.op_type}"

        # Match against full api_path, or op_type directly
        match = self.grounded_symbols.get(full_path) or self.grounded_symbols.get(
            node.op_type
        )
        if match is None:
            # Hallucination detected
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="kind",
                    message=f"Ungrounded symbol '{node.op_type}' in domain '{node.domain}'. Symbol not found in framework snapshot.",
                    level=ValidationLevel.ERROR,
                )
            )
        else:
            # Validate attributes against parameter/attribute keys if present
            known_params: set[str] = set()
            for p in match.get("params", []):
                if isinstance(p, dict) and "name" in p:
                    known_params.add(p["name"])
            if "attributes" in match and isinstance(match["attributes"], dict):
                known_params.update(match["attributes"].keys())

            if known_params:
                for attr_key in node.attributes:
                    if attr_key not in known_params:
                        errors.append(
                            ValidationError(
                                node_id=node.id,
                                attribute=attr_key,
                                message=f"Ungrounded attribute '{attr_key}' on '{node.op_type}'. Allowed attributes/params: {sorted(known_params)}.",
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
    graph: LogicalGraph, snapshots_path: str | None = None
) -> GroundingAuditReport:
    """Audit a LogicalGraph against framework snapshots.

    Args:
        graph (LogicalGraph): Logical graph to audit.
        snapshots_path (Optional[str]): Path to snapshot manifest or JSON file.

    Returns:
        GroundingAuditReport: Resulting grounding audit report.
    """
    validator = GroundingValidator(snapshot_manifest=snapshots_path)
    return validator.audit_graph(graph)
