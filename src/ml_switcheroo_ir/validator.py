"""Validator module for ml_switcheroo_ir schemas."""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

from ml_switcheroo_ir import LogicalNode, LogicalGraph
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY, OpSchema


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

    def __init__(self, registry: Dict[str, OpSchema] = None) -> None:
        """Initialize the validator.

        Args:
            registry (Dict[str, OpSchema], optional): The operator registry to use.
                Defaults to the built-in ONNX_REGISTRY.
        """
        if registry is None:
            self.registry = ONNX_REGISTRY
        else:
            self.registry = registry

    def validate_kind(self, node: LogicalNode) -> List[ValidationError]:
        """Validate that the node's kind exists in the registry for its domain.

        Args:
            node (LogicalNode): The node to validate.

        Returns:
            List[ValidationError]: A list of errors found.
        """
        errors: List[ValidationError] = []
        if node.domain != "ai.onnx":
            return errors  # Custom domains not handled strictly here unless in registry

        if node.op_type not in self.registry:
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="kind",
                    message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                    level=ValidationLevel.ERROR,
                )
            )
        return errors

    def validate_required_attributes(self, node: LogicalNode) -> List[ValidationError]:
        """Validate that all required attributes for the node's kind are present.

        Args:
            node (LogicalNode): The node to validate.

        Returns:
            List[ValidationError]: A list of errors found.
        """
        errors: List[ValidationError] = []
        if node.op_type not in self.registry:
            return errors

        schema = self.registry[node.op_type]
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

    def validate_attribute_types(self, node: LogicalNode) -> List[ValidationError]:
        """Validate that attributes have the correct types according to the schema.

        Args:
            node (LogicalNode): The node to validate.

        Returns:
            List[ValidationError]: A list of errors found.
        """
        errors: List[ValidationError] = []
        if node.op_type not in self.registry:
            return errors

        schema = self.registry[node.op_type]
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
            elif expected_type == "str" and not isinstance(value, str):
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

    def populate_defaults(self, node: LogicalNode) -> None:
        """Inject missing optional attributes with default values from the schema.

        Args:
            node (LogicalNode): The node to mutate.
        """
        if node.op_type not in self.registry:
            return

        schema = self.registry[node.op_type]
        for attr_name, attr_schema in schema.attributes.items():
            if not attr_schema.required and attr_schema.default is not None:
                if attr_name not in node.attributes:
                    node.attributes[attr_name] = attr_schema.default

    def validate_graph(self, graph: LogicalGraph) -> List[ValidationError]:
        """Validate all nodes and edges in a LogicalGraph.

        Args:
            graph (LogicalGraph): The graph to validate.

        Returns:
            List[ValidationError]: A list of all aggregated errors.
        """
        errors: List[ValidationError] = []

        # Validate nodes
        for node in graph.nodes.values():
            errors.extend(self.validate_kind(node))
            errors.extend(self.validate_required_attributes(node))
            errors.extend(self.validate_attribute_types(node))
            self.populate_defaults(node)

        # Validate edges
        errors.extend(self.validate_edges(graph))

        return errors

    def validate_edges(self, graph: LogicalGraph) -> List[ValidationError]:
        """Validate that all node inputs exist in the graph.

        Args:
            graph (LogicalGraph): The graph to validate.

        Returns:
            List[ValidationError]: A list of edge validation errors.
        """
        errors: List[ValidationError] = []
        node_ids = set(graph.nodes.keys())

        for node_id, node in graph.nodes.items():
            for inp in node.inputs:
                if inp not in node_ids:
                    errors.append(
                        ValidationError(
                            node_id=node_id,
                            attribute="inputs",
                            message=f"Node input '{inp}' does not exist.",
                            level=ValidationLevel.ERROR,
                        )
                    )
        return errors
