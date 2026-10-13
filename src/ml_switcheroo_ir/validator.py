"""Validator module for ml_switcheroo_ir schemas."""

from __future__ import annotations

import gzip
import json
import logging
import os
from typing import Any, Sequence

from ml_switcheroo_ir.graph import LogicalGraph, LogicalMesh, LogicalNode
from ml_switcheroo_ir.schema.custom_ops import (
    COLLECTIVE_OPS_REGISTRY,
    CUSTOM_OPS_REGISTRY,
    QUANTIZATION_OPS_REGISTRY,
    STATE_OPS_REGISTRY,
)
from ml_switcheroo_ir.schema.framework_registries import (
    ARRAY_API_REGISTRY,
    ATEN_REGISTRY,
    ODL_CATALOG,
)
from ml_switcheroo_ir.schema.ghost import (
    RDNA_INSTRUCTION_PRIMITIVES,
    SASS_INSTRUCTION_PRIMITIVES,
    WGSL_COMPUTE_BUILTINS,
    WGSL_MUTATING_OPS,
    WGSL_PRIMITIVE_SIGNATURES,
    ExtendedGhostRef,
)
from ml_switcheroo_ir.schema.low_level_registries import (
    METAL_REGISTRY,
    PTX_REGISTRY,
    WASM_REGISTRY,
    WEBGL_REGISTRY,
    WGSL_REGISTRY,
)
from ml_switcheroo_ir.schema.mlir_registry import MLIR_REGISTRY
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY, OpSchema
from ml_switcheroo_ir.schema.rdna_registry import (
    RDNA_REGISTRY,
    RDNA_TO_VOPD_MAP,
    RDNA_VOPD_SLOTS,
)
from ml_switcheroo_ir.schema.sass_registry import (
    SASS_PIPELINE_LATENCIES,
    SASS_REGISTRY,
)
from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY
from ml_switcheroo_ir.snapshots import (
    DEFAULT_SNAPSHOT_DIR,
    get_default_snapshots_dir,
)
from ml_switcheroo_ir.validation.core import (
    CORE_MLIR_DIALECTS,
    KNOWN_SYNONYMS,
    ValidationError,
    ValidationLevel,
    compute_levenshtein,
)
from ml_switcheroo_ir.validation.distributed import (
    estimate_communication_volume,
    estimate_graph_communication_volume,
    validate_pipeline_and_checkpointing_graph,
    validate_sharding_node,
    validate_sharding_propagation_graph,
)
from ml_switcheroo_ir.validation.grounding import (
    DiagnosticSeverity,
    GroundingAuditReport,
    GroundingDiagnostic,
    GroundingReport,
    GroundingSeverity,
)
from ml_switcheroo_ir.validation.hardware import (
    _extract_dst_vgpr,
    _extract_sass_registers,
    _extract_vgpr_indices,
    validate_isa_instruction_node,
    validate_sass_scoreboarding_sequence,
    validate_vopd_pairing,
)
from ml_switcheroo_ir.validation.quantization import validate_quantization_node

logger = logging.getLogger(__name__)

__all__ = [
    "CORE_MLIR_DIALECTS",
    "DEFAULT_SNAPSHOT_DIR",
    "DiagnosticSeverity",
    "ExtendedGhostRef",
    "GroundingAuditReport",
    "GroundingDiagnostic",
    "GroundingReport",
    "GroundingSeverity",
    "GroundingValidator",
    "KNOWN_SYNONYMS",
    "MultiDialectValidator",
    "RDNA_TO_VOPD_MAP",
    "RDNA_VOPD_SLOTS",
    "SASS_PIPELINE_LATENCIES",
    "ValidationError",
    "ValidationLevel",
    "Validator",
    "WGSL_COMPUTE_BUILTINS",
    "WGSL_MUTATING_OPS",
    "_extract_dst_vgpr",
    "_extract_sass_registers",
    "_extract_vgpr_indices",
    "audit_graph_grounding",
    "compute_levenshtein",
    "estimate_communication_volume",
    "estimate_graph_communication_volume",
    "get_default_snapshots_dir",
    "validate_vopd_pairing",
]


class Validator:
    """Validates LogicalGraph and LogicalNode instances against schemas."""

    def __init__(
        self,
        registry: dict[str, OpSchema] | None = None,
        custom_registry: dict[str, OpSchema] | None = None,
        stablehlo_registry: dict[str, OpSchema] | None = None,
        mlir_registry: dict[str, OpSchema] | None = None,
        state_registry: dict[str, OpSchema] | None = None,
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
            mlir_registry (Optional[Dict[str, OpSchema]]): The Core MLIR operator registry.
                Defaults to the built-in MLIR_REGISTRY.
            state_registry (Optional[Dict[str, OpSchema]]): The State mutation operator registry.
                Defaults to the built-in STATE_OPS_REGISTRY.
            level (ValidationLevel): The validation severity threshold (default: ValidationLevel.WARNING).
            strict (bool): Convenience flag; if True, sets level to ValidationLevel.STRICT.
            grounding_validator (Optional[GroundingValidator]): Optional grounding validator for snapshot verification.
        """
        self.registry = ONNX_REGISTRY if registry is None else registry
        self.custom_registry = (
            CUSTOM_OPS_REGISTRY if custom_registry is None else custom_registry
        )
        self.stablehlo_registry = (
            STABLEHLO_REGISTRY if stablehlo_registry is None else stablehlo_registry
        )
        self.mlir_registry = MLIR_REGISTRY if mlir_registry is None else mlir_registry
        self.state_registry = (
            STATE_OPS_REGISTRY if state_registry is None else state_registry
        )
        self.collective_registry = COLLECTIVE_OPS_REGISTRY
        self.quantization_registry = QUANTIZATION_OPS_REGISTRY

        if strict:
            self.level = ValidationLevel.STRICT
        elif level == ValidationLevel.ERROR:
            self.level = ValidationLevel.LENIENT
        else:
            self.level = level

        self.grounding_validator = grounding_validator

    def _get_domain_registry(self, domain: str) -> dict[str, OpSchema] | None:
        """Resolve the secondary operator registry corresponding to a dialect domain.

        Args:
            domain (str): Operator domain identifier.

        Returns:
            dict[str, OpSchema] | None: Matching secondary registry or None.
        """
        domain_map: dict[str, dict[str, OpSchema]] = {
            "ml.switcheroo.custom": self.custom_registry,
            "stablehlo": self.stablehlo_registry,
            "mlir": self.mlir_registry,
            "mlir.arith": self.mlir_registry,
            "mlir.math": self.mlir_registry,
            "mlir.tensor": self.mlir_registry,
            "mlir.linalg": self.mlir_registry,
            "mlir.scf": self.mlir_registry,
            "collective": self.collective_registry,
            "ml.switcheroo.collective": self.collective_registry,
            "quantization": self.quantization_registry,
            "ml.switcheroo.quantization": self.quantization_registry,
            "state": self.state_registry,
            "ml.switcheroo.state": self.state_registry,
            "amd_rdna": RDNA_REGISTRY,
            "rdna": RDNA_REGISTRY,
            "nvidia_sass": SASS_REGISTRY,
            "sass": SASS_REGISTRY,
            "nvidia_ptx": PTX_REGISTRY,
            "ptx": PTX_REGISTRY,
            "metal_msl": METAL_REGISTRY,
            "metal": METAL_REGISTRY,
            "wasm_simd": WASM_REGISTRY,
            "wasm": WASM_REGISTRY,
            "webgl": WEBGL_REGISTRY,
            "wgsl": WGSL_REGISTRY,
            "aten": ATEN_REGISTRY,
            "array_api": ARRAY_API_REGISTRY,
            "odl": ODL_CATALOG,
            "abstract": ODL_CATALOG,
        }
        return domain_map.get(domain)

    def _get_schema(self, node: LogicalNode) -> OpSchema | None:
        """Retrieve the schema for a node across registered domains.

        Args:
            node (LogicalNode): The node to inspect.

        Returns:
            Optional[OpSchema]: The schema if found, or None.
        """
        if node.domain in ("ml.switcheroo.custom", "stablehlo"):
            s = self.registry.get(node.op_type)
            if s and s.domain == node.domain:
                return s
            sec = self._get_domain_registry(node.domain)
            return sec.get(node.op_type) if sec is not None else None

        if node.domain in ("ad", "ml.switcheroo.ad") and node.op_type in (
            "ZeroTangent",
            "NoTangent",
        ):
            return OpSchema(
                name=node.op_type,
                domain=node.domain,
                version=1,
                attributes={},
                inputs=[],
                outputs=["tangent"],
            )

        primary = self.registry.get(node.op_type)
        if primary is not None:
            return primary
        secondary = self._get_domain_registry(node.domain)
        if secondary is not None:
            return secondary.get(node.op_type)
        return None

    def validate_kind(self, node: LogicalNode) -> list[ValidationError]:
        """Validate that the node's kind exists in the registry for its domain.

        Args:
            node (LogicalNode): The node to validate.

        Returns:
            List[ValidationError]: A list of errors found.
        """
        if node.op_type in ("Input", "Output"):
            return []

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
        elif node.domain in (
            "mlir",
            "mlir.arith",
            "mlir.math",
            "mlir.tensor",
            "mlir.linalg",
            "mlir.scf",
        ):
            if (
                node.op_type not in self.mlir_registry
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
        elif node.domain in ("amd_rdna", "rdna"):
            if (
                node.op_type not in RDNA_REGISTRY
                and node.op_type not in RDNA_INSTRUCTION_PRIMITIVES
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
        elif node.domain in ("nvidia_sass", "sass"):
            if (
                node.op_type not in SASS_REGISTRY
                and node.op_type not in SASS_INSTRUCTION_PRIMITIVES
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
        elif node.domain in ("webgpu_wgsl", "wgsl"):
            if (
                node.op_type not in WGSL_REGISTRY
                and node.op_type not in WGSL_PRIMITIVE_SIGNATURES
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
        elif node.domain in ("nvidia_ptx", "ptx"):
            if node.op_type not in PTX_REGISTRY and node.op_type not in self.registry:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain in ("metal_msl", "metal"):
            if node.op_type not in METAL_REGISTRY and node.op_type not in self.registry:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain in ("wasm_simd", "wasm"):
            if node.op_type not in WASM_REGISTRY and node.op_type not in self.registry:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain in ("webgl",):
            if node.op_type not in WEBGL_REGISTRY and node.op_type not in self.registry:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain in ("state", "ml.switcheroo.state"):
            if (
                node.op_type not in self.state_registry
                and node.op_type not in self.registry
            ):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in state registry.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain in ("ad", "ml.switcheroo.ad"):
            if node.op_type not in ("ZeroTangent", "NoTangent"):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in autodiff sentinel registry.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain in ("aten",):
            if node.op_type not in ATEN_REGISTRY and node.op_type not in self.registry:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
        elif node.domain in ("array_api",):
            if (
                node.op_type not in ARRAY_API_REGISTRY
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
        elif node.domain in ("odl", "abstract"):
            if node.op_type not in ODL_CATALOG and node.op_type not in self.registry:
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

        isa_attributes = {
            "amd_rdna": {
                "wavefront_size",
                "register_classes",
                "vgpr_operands",
                "dst_vgpr",
                "src_vgprs",
                "dst",
                "vopd_slot",
                "vopd_target",
                "modifiers",
            },
            "rdna": {
                "wavefront_size",
                "register_classes",
                "vgpr_operands",
                "dst_vgpr",
                "src_vgprs",
                "dst",
                "vopd_slot",
                "vopd_target",
                "modifiers",
            },
            "nvidia_sass": {
                "barrier_predicate",
                "warp_sync",
                "stall",
                "stall_count",
                "yield_flag",
                "read_barrier",
                "write_barrier",
                "wait_barrier_mask",
                "barrier_mask",
                "execution_latency",
                "modifiers",
                "control_codes",
                "register_classes",
            },
            "sass": {
                "barrier_predicate",
                "warp_sync",
                "stall",
                "stall_count",
                "yield_flag",
                "read_barrier",
                "write_barrier",
                "wait_barrier_mask",
                "barrier_mask",
                "execution_latency",
                "modifiers",
                "control_codes",
                "register_classes",
            },
            "webgpu_wgsl": {"workgroup_size", "address_space", "qualifier"},
            "wgsl": {"workgroup_size", "address_space", "qualifier"},
            "nvidia_ptx": {"vector_widths", "min_sm", "address_space"},
            "ptx": {"vector_widths", "min_sm", "address_space"},
        }
        allowed_isa_keys = isa_attributes.get(node.domain, set())

        for key, value in node.attributes.items():
            if key not in schema.attributes:
                if key in allowed_isa_keys:
                    continue
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

            valid = True
            if expected_type == "int" and not isinstance(value, int):
                valid = False
            elif expected_type == "float" and not isinstance(value, float):
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
        return validate_sharding_node(node, mesh)

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
        return validate_isa_instruction_node(node, self.level)

    def validate_quantization(self, node: LogicalNode) -> list[ValidationError]:
        """Validate quantization operator attributes, alignment constraints, and packing formats.

        Args:
            node (LogicalNode): The quantization operator node.

        Returns:
            list[ValidationError]: Detected quantization validation errors.
        """
        return validate_quantization_node(node)

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

        if (
            self.level == ValidationLevel.STRICT
            and node.shape_metadata is None
            and node.op_type != "NoTangent"
        ):
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="shape_metadata",
                    message=f"Node '{node.id}' missing required shape_metadata in STRICT validation mode.",
                    level=ValidationLevel.ERROR,
                )
            )

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

        for node in graph.nodes.values():
            errors.extend(self.validate_node(node, graph.mesh))

        errors.extend(self.validate_edges(graph))

        sass_nodes = [n for n in graph.nodes.values() if n.domain == "nvidia_sass"]
        if sass_nodes:
            errors.extend(self.validate_sass_scoreboarding(sass_nodes))

        errors.extend(self.validate_sharding_propagation(graph))
        errors.extend(self.validate_pipeline_and_checkpointing(graph))

        if self.level == ValidationLevel.LENIENT:
            errors = [e for e in errors if e.level == ValidationLevel.ERROR]

        return errors

    def validate_vopd_pairing(
        self, opX: LogicalNode, opY: LogicalNode
    ) -> list[ValidationError]:
        """Validate AMD RDNA3 / GFX11 dual-issue VOPD instruction pairing rules.

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
        return validate_sass_scoreboarding_sequence(nodes, self.level)

    def validate_sharding_propagation(
        self, graph: LogicalGraph
    ) -> list[ValidationError]:
        """Validate SPaDe/GSPMD sharding propagation invariants across graph operations.

        Args:
            graph (LogicalGraph): Computational graph to validate.

        Returns:
            list[ValidationError]: Detected sharding propagation violations.
        """
        return validate_sharding_propagation_graph(graph, self.level)

    def validate_pipeline_and_checkpointing(
        self, graph: LogicalGraph
    ) -> list[ValidationError]:
        """Validate pipeline parallel stage orderings, boundary markers, and activation checkpointing tags.

        Args:
            graph (LogicalGraph): Computational graph to validate.

        Returns:
            list[ValidationError]: Detected pipeline or checkpointing violations.
        """
        return validate_pipeline_and_checkpointing_graph(graph)

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
                first_err = fatal_errors[0]
                raise ValidationError(
                    node_id=first_err.node_id,
                    attribute=first_err.attribute,
                    message=first_err.message,
                    level=first_err.level,
                )

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

        for node_id, node in graph.nodes.items():
            for inp in node.inputs:
                if (
                    inp not in node_ids
                    and graph.get_output_producer(inp) is None
                    and inp not in graph.inputs
                    and inp not in graph.initializers
                ):
                    errors.append(
                        ValidationError(
                            node_id=node_id,
                            attribute="inputs",
                            message=f"Node input '{inp}' does not exist.",
                            level=ValidationLevel.ERROR,
                        )
                    )

        for edge in graph.edges:
            producer = graph.nodes.get(edge.source)
            if producer is None:
                prod_tuple = graph.get_producing_output_index(edge.source)
                if prod_tuple is not None:
                    producer = prod_tuple[0]

            if producer is not None and producer.has_multiple_outputs:
                num_outputs = (
                    len(producer.outputs)
                    if len(producer.outputs) > 1
                    else len(producer.output_specs)
                )
                if edge.source_idx is not None and (
                    edge.source_idx < 0 or edge.source_idx >= num_outputs
                ):
                    errors.append(
                        ValidationError(
                            node_id=edge.target,
                            attribute="edges",
                            message=(
                                f"Edge references invalid source_idx {edge.source_idx} "
                                f"(out of bounds) for multi-output node '{producer.id}' (has {num_outputs} outputs)."
                            ),
                            level=ValidationLevel.ERROR,
                        )
                    )

        return errors


class GroundingValidator(Validator):
    """Validator that audits graphs strictly against external ground-truth snapshot manifests."""

    def __init__(
        self,
        snapshot_manifest: dict[str, Any] | list[Any] | str | None = None,
        snapshots_dir: str | None = None,
        registry: dict[str, OpSchema] | None = None,
        use_default_if_none: bool = False,
    ) -> None:
        """Initialize GroundingValidator with external snapshot file, directory, list, or dict.

        Args:
            snapshot_manifest (Union[Dict[str, Any], List[Any], str, None]): Snapshot dictionary,
                list of records, file path, directory path, or collection.
            snapshots_dir (Optional[str]): Explicit path to snapshots directory.
            registry (Optional[Dict[str, OpSchema]]): Base operator registry.
            use_default_if_none (bool): If True and snapshot_manifest is None, load from DEFAULT_SNAPSHOT_DIR.
        """
        super().__init__(registry=registry)
        self.grounded_symbols: dict[str, dict[str, Any]] = {}
        self.concept_map: dict[str, Any] = {}
        self.parameter_translations: dict[str, Any] = {}
        self._engine: Any = None

        target = snapshot_manifest if snapshot_manifest is not None else snapshots_dir

        try:
            from ml_ecosystem_snapshots.grounding.engine import GroundingEngine

            search_dirs: list[str] = []
            if isinstance(target, str) and os.path.isdir(target):
                search_dirs.append(target)
            if (
                snapshots_dir
                and os.path.isdir(snapshots_dir)
                and snapshots_dir not in search_dirs
            ):
                search_dirs.append(snapshots_dir)
            if (
                os.path.isdir(DEFAULT_SNAPSHOT_DIR)
                and DEFAULT_SNAPSHOT_DIR not in search_dirs
            ):
                search_dirs.append(DEFAULT_SNAPSHOT_DIR)
            self._engine = GroundingEngine(
                base_dirs=search_dirs if search_dirs else None
            )
        except (ImportError, AttributeError, ValueError, OSError, RuntimeError) as exc:
            logger.debug("Failed initializing GroundingEngine: %s", exc)
            self._engine = None

        if target is not None:
            self._ingest_manifest_target(target)
        elif use_default_if_none and os.path.isdir(DEFAULT_SNAPSHOT_DIR):
            self._load_directory(DEFAULT_SNAPSHOT_DIR)

    def _ingest_manifest_target(
        self, target: str | dict[str, Any] | list[Any] | object
    ) -> None:
        """Ingest snapshot manifest from diverse target types.

        Args:
            target (Union[str, dict[str, Any], list[Any], object]): Directory path, file path, dict, or list.
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
            if "_parameter_translations" in data and isinstance(
                data["_parameter_translations"], dict
            ):
                self.parameter_translations.update(data["_parameter_translations"])

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
                    if k == "_parameter_translations":
                        continue
                    if isinstance(v, dict):
                        if any(
                            fw in v
                            for fw in (
                                "torch",
                                "jax",
                                "tensorflow",
                                "tf",
                                "stablehlo",
                                "numpy",
                            )
                        ):
                            self.concept_map[k] = v
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

        if (
            best_match is None
            and self._engine is not None
            and domain
            and (
                not hasattr(self._engine, "_discover_target_files")
                or bool(self._engine._discover_target_files(domain))
            )
        ):
            try:
                engine_cand = self._engine.suggest_closest_symbol(
                    domain, op_type, max_distance=3
                )
                if engine_cand:
                    best_match = engine_cand
            except (
                AttributeError,
                KeyError,
                ValueError,
                RuntimeError,
                TypeError,
            ) as exc:
                logger.debug("Failed symbol suggestion from engine: %s", exc)

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
        if (
            match is None
            and self._engine is not None
            and node.domain
            and (
                not hasattr(self._engine, "_discover_target_files")
                or bool(self._engine._discover_target_files(node.domain))
            )
        ):
            try:
                ref = self._engine.get_symbol(node.domain, node.op_type)
                if ref is not None:
                    match = (
                        ref.model_dump() if hasattr(ref, "model_dump") else ref.__dict__
                    )
            except (
                AttributeError,
                KeyError,
                ValueError,
                RuntimeError,
                TypeError,
            ) as exc:
                logger.debug("Failed symbol lookup from engine: %s", exc)

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
                    suggested_fix=suggestion,
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

            accepted_kwargs = match.get("accepted_kwargs") or match.get("kwargs")
            if isinstance(accepted_kwargs, list):
                for kw in accepted_kwargs:
                    if isinstance(kw, str):
                        known_params.add(kw)
                    elif isinstance(kw, dict) and "name" in kw:
                        known_params.add(kw["name"])

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
                                suggested_fix=attr_suggestion,
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
    resolved_path: str | list[Any] | dict[str, Any] | None
    if snapshots_path is None and os.path.isdir(DEFAULT_SNAPSHOT_DIR):
        resolved_path = DEFAULT_SNAPSHOT_DIR
    else:
        resolved_path = snapshots_path
    validator = GroundingValidator(snapshot_manifest=resolved_path)
    return validator.audit_graph(graph)


MultiDialectValidator = Validator
