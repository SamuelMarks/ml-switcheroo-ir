"""Schema, attribute, edge, and multi-dialect graph validator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import typing

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir.validation.grounding import GroundingValidator

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
    WGSL_PRIMITIVE_SIGNATURES,
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
from ml_switcheroo_ir.schema.rdna_registry import RDNA_REGISTRY
from ml_switcheroo_ir.schema.sass_registry import SASS_REGISTRY
from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY
from ml_switcheroo_ir.types import AttributeValue
from ml_switcheroo_ir.validation.core import (
    CORE_MLIR_DIALECTS,
    ValidationError,
    ValidationLevel,
)
from ml_switcheroo_ir.validation.distributed import (
    CommunicationVolumeReport,
    estimate_communication_volume,
    estimate_graph_communication_volume,
    validate_pipeline_and_checkpointing_graph,
    validate_sharding_node,
    validate_sharding_propagation_graph,
)
from ml_switcheroo_ir.validation.hardware import (
    validate_isa_instruction_node,
    validate_sass_scoreboarding_sequence,
    validate_vopd_pairing,
)
from ml_switcheroo_ir.validation.quantization import validate_quantization_node

__all__ = [
    "Validator",
]

_ISA_ALLOWED_ATTRIBUTES: dict[str, set[str]] = {
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
        "memory_space",
        "stall",
        "stall_count",
        "yield",
        "yield_flag",
        "read_barrier",
        "write_barrier",
        "wait_barrier_mask",
        "barrier_mask",
        "execution_latency",
        "modifiers",
        "control_codes",
        "register_classes",
        "src_regs",
        "srcs",
        "dst_reg",
        "dst",
    },
    "sass": {
        "barrier_predicate",
        "warp_sync",
        "memory_space",
        "stall",
        "stall_count",
        "yield",
        "yield_flag",
        "read_barrier",
        "write_barrier",
        "wait_barrier_mask",
        "barrier_mask",
        "execution_latency",
        "modifiers",
        "control_codes",
        "register_classes",
        "src_regs",
        "srcs",
        "dst_reg",
        "dst",
    },
    "webgpu_wgsl": {
        "workgroup_size",
        "address_space",
        "qualifier",
        "struct_alignment",
        "alignment",
        "array_stride",
        "builtin",
        "is_mutating",
    },
    "wgsl": {
        "workgroup_size",
        "address_space",
        "qualifier",
        "struct_alignment",
        "alignment",
        "array_stride",
        "builtin",
        "is_mutating",
    },
    "nvidia_ptx": {"vector_widths", "min_sm", "address_space"},
    "ptx": {"vector_widths", "min_sm", "address_space"},
}

_ATTRIBUTE_TYPE_CHECKERS: dict[str, Callable[[AttributeValue], bool]] = {
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (float, int)) and not isinstance(v, bool),
    "str": lambda v: isinstance(v, str),
    "dict": lambda v: isinstance(v, dict),
    "List[int]": lambda v: (
        isinstance(v, list)
        and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
    ),
    "List[float]": lambda v: (
        isinstance(v, list)
        and all(isinstance(x, (float, int)) and not isinstance(x, bool) for x in v)
    ),
    "List[str]": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
    "bool": lambda v: isinstance(v, bool),
}


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
            registry (dict[str, OpSchema] | None): The operator registry to use.
                Defaults to the built-in ONNX_REGISTRY.
            custom_registry (dict[str, OpSchema] | None): The custom operator registry.
                Defaults to the built-in CUSTOM_OPS_REGISTRY.
            stablehlo_registry (dict[str, OpSchema] | None): The StableHLO operator registry.
                Defaults to the built-in STABLEHLO_REGISTRY.
            mlir_registry (dict[str, OpSchema] | None): The Core MLIR operator registry.
                Defaults to the built-in MLIR_REGISTRY.
            state_registry (dict[str, OpSchema] | None): The State mutation operator registry.
                Defaults to the built-in STATE_OPS_REGISTRY.
            level (ValidationLevel): The validation severity threshold (default: ValidationLevel.WARNING).
            strict (bool): Convenience flag; if True, sets level to ValidationLevel.STRICT.
            grounding_validator (GroundingValidator | None): Optional grounding validator for snapshot verification.
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
            "webgpu_wgsl": WGSL_REGISTRY,
            "wgsl": WGSL_REGISTRY,
            "aten": ATEN_REGISTRY,
            "array_api": ARRAY_API_REGISTRY,
            "odl": ODL_CATALOG,
            "abstract": ODL_CATALOG,
        }
        return domain_map.get(domain)

    def _filter_by_level(self, errors: list[ValidationError]) -> list[ValidationError]:
        """Filter accumulated validation diagnostics according to the active severity level.

        Args:
            errors (list[ValidationError]): Raw validation diagnostics.

        Returns:
            list[ValidationError]: Filtered validation diagnostics.
        """
        if self.level == ValidationLevel.LENIENT:
            return [e for e in errors if e.level == ValidationLevel.ERROR]
        return errors

    def _get_schema(self, node: LogicalNode) -> OpSchema | None:
        """Retrieve the schema for a node across registered domains.

        Prioritizes the domain-specific registry when ``node.domain != 'ai.onnx'``
        so ONNX schemas never shadow ATen, Array API, ODL, or low-level dialect schemas.

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

        if node.domain != "ai.onnx":
            secondary = self._get_domain_registry(node.domain)
            if secondary is not None and node.op_type in secondary:
                return secondary[node.op_type]

        return self.registry.get(node.op_type)

    def _validate_core_mlir_dialect(self, node: LogicalNode) -> list[ValidationError]:
        """Validate a node belonging to one of the CORE_MLIR_DIALECTS.

        Args:
            node (LogicalNode): Node in a core MLIR dialect.

        Returns:
            list[ValidationError]: Detected kind or operand-count errors.
        """
        errors: list[ValidationError] = []
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
        return errors

    def validate_kind(self, node: LogicalNode) -> list[ValidationError]:
        """Validate that the node's kind exists in the registry for its domain.

        Args:
            node (LogicalNode): The node to validate.

        Returns:
            List[ValidationError]: A list of errors found.
        """
        if node.op_type in ("Input", "Output"):
            return []

        if node.domain == "ai.onnx":
            if node.op_type not in self.registry:
                return [
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                ]
            return []

        if node.domain in ("ad", "ml.switcheroo.ad"):
            if node.op_type not in ("ZeroTangent", "NoTangent"):
                return [
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in autodiff sentinel registry.",
                        level=ValidationLevel.ERROR,
                    )
                ]
            return []

        secondary = self._get_domain_registry(node.domain)
        if secondary is not None:
            extra_primitives: Mapping[str, object] = {}
            if node.domain in ("amd_rdna", "rdna"):
                extra_primitives = RDNA_INSTRUCTION_PRIMITIVES
            elif node.domain in ("nvidia_sass", "sass"):
                extra_primitives = SASS_INSTRUCTION_PRIMITIVES
            elif node.domain in ("webgpu_wgsl", "wgsl"):
                extra_primitives = WGSL_PRIMITIVE_SIGNATURES

            if (
                node.op_type in secondary
                or node.op_type in extra_primitives
                or node.op_type in self.registry
            ):
                return []

            lvl = (
                ValidationLevel.WARNING
                if node.domain == "ml.switcheroo.custom"
                and self.level != ValidationLevel.STRICT
                else ValidationLevel.ERROR
            )
            custom_labels = {
                "collective": "collective registry.",
                "ml.switcheroo.collective": "collective registry.",
                "quantization": "quantization registry.",
                "ml.switcheroo.quantization": "quantization registry.",
                "state": "state registry.",
                "ml.switcheroo.state": "state registry.",
            }
            target_desc = custom_labels.get(node.domain, f"domain '{node.domain}'.")
            return [
                ValidationError(
                    node_id=node.id,
                    attribute="kind",
                    message=f"Operator '{node.op_type}' not found in {target_desc}",
                    level=lvl,
                )
            ]

        if node.domain in CORE_MLIR_DIALECTS:
            return self._validate_core_mlir_dialect(node)

        if (
            node.domain not in ("ai.custom", "custom")
            and node.op_type not in self.registry
        ):
            return [
                ValidationError(
                    node_id=node.id,
                    attribute="domain",
                    message=f"Unrecognized domain '{node.domain}'.",
                    level=ValidationLevel.ERROR,
                )
            ]
        return []

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

        allowed_isa_keys = _ISA_ALLOWED_ATTRIBUTES.get(node.domain, set())

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
            checker = _ATTRIBUTE_TYPE_CHECKERS.get(expected_type)
            valid = checker(value) if checker is not None else True

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
        self,
        node: LogicalNode,
        mesh: LogicalMesh | None = None,
        populate_defaults: bool = True,
    ) -> list[ValidationError]:
        """Validate a single LogicalNode against schemas and configured severity level.

        Args:
            node (LogicalNode): The node to validate.
            mesh (Optional[LogicalMesh]): Optional logical device mesh for sharding validation.
            populate_defaults (bool): Whether to populate missing default attributes from schema (default: True).

        Returns:
            List[ValidationError]: Aggregated list of validation errors and warnings for the node.
        """
        errors: list[ValidationError] = []
        errors.extend(self.validate_kind(node))
        errors.extend(self.validate_required_attributes(node))
        errors.extend(self.validate_attribute_types(node))
        errors.extend(self.validate_sharding(node, mesh))
        if node.domain in (
            "amd_rdna",
            "rdna",
            "nvidia_sass",
            "sass",
            "webgpu_wgsl",
            "wgsl",
        ):
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

        if populate_defaults:
            self.populate_defaults(node)

        return self._filter_by_level(errors)

    def validate_graph(
        self,
        graph: LogicalGraph,
        populate_defaults: bool = True,
    ) -> list[ValidationError]:
        """Validate all nodes, sharding, and edges in a LogicalGraph.

        Args:
            graph (LogicalGraph): The graph to validate.
            populate_defaults (bool): Whether to populate missing default attributes on nodes (default: True).

        Returns:
            List[ValidationError]: A list of all aggregated errors.
        """
        errors: list[ValidationError] = []

        for node in graph.nodes.values():
            errors.extend(
                self.validate_node(
                    node, graph.mesh, populate_defaults=populate_defaults
                )
            )

        errors.extend(self.validate_edges(graph))

        sass_nodes = [
            n for n in graph.nodes.values() if n.domain in ("nvidia_sass", "sass")
        ]
        if sass_nodes:
            errors.extend(self.validate_sass_scoreboarding(sass_nodes))

        errors.extend(self.validate_sharding_propagation(graph))
        errors.extend(self.validate_pipeline_and_checkpointing(graph))

        return self._filter_by_level(errors)

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
    ) -> CommunicationVolumeReport:
        """Aggregate analytical communication volume across all collective operators in a graph.

        Args:
            graph (LogicalGraph): The computational graph to analyze.

        Returns:
            CommunicationVolumeReport: Dictionary with total communication volume in bytes,
                and breakdowns by mesh axis and operator type.
        """
        return estimate_graph_communication_volume(graph)

    def validate(
        self,
        target: LogicalGraph | LogicalNode,
        raise_on_error: bool = False,
        populate_defaults: bool = True,
    ) -> list[ValidationError]:
        """Validate a LogicalGraph or LogicalNode instance against configured validation level.

        Args:
            target (Union[LogicalGraph, LogicalNode]): Graph or node instance to validate.
            raise_on_error (bool): If True, raises the first fatal ValidationError encountered.
            populate_defaults (bool): Whether to populate missing default attributes (default: True).

        Returns:
            List[ValidationError]: List of validation errors and warnings.

        Raises:
            ValidationError: If raise_on_error is True and a fatal ValidationError is found.
        """
        if isinstance(target, LogicalGraph):
            errors = self.validate_graph(target, populate_defaults=populate_defaults)
        else:
            errors = self.validate_node(target, populate_defaults=populate_defaults)

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
