"""Schema, attribute, edge, acyclicity, and multi-dialect graph validator."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import typing_extensions

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir.validation.grounding import GroundingValidator

from ml_switcheroo_ir.graph import (
    CyclicGraphError,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    ProvenGraph,
    SchemaValidPhase,
    topological_sort,
)
from ml_switcheroo_ir.identifiers import is_non_bool_int
from ml_switcheroo_ir.schema import ALL_DIALECT_CATALOG_WITNESSES
from ml_switcheroo_ir.schema.base import AttributeType, DialectCatalogWitness, normalize_attribute_type
from ml_switcheroo_ir.schema.custom_ops import (
    COLLECTIVE_OPS_REGISTRY,
    CUSTOM_OPS_REGISTRY,
    QUANTIZATION_OPS_REGISTRY,
    STATE_OPS_REGISTRY,
)
from ml_switcheroo_ir.schema.ghost import (
    RDNA_INSTRUCTION_PRIMITIVES,
    SASS_INSTRUCTION_PRIMITIVES,
    WGSL_PRIMITIVE_SIGNATURES,
    RdnaPrimitiveSpec,
    SassPrimitiveSpec,
    WgslPrimitiveSpec,
)
from ml_switcheroo_ir.schema.mlir_registry import (
    MLIR_REGISTRY,
    resolve_mlir_op_schema,
)
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY, OpSchema
from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY
from ml_switcheroo_ir.typeclasses.algebra import NonEmpty
from ml_switcheroo_ir.typeclasses.outcome import Validation, from_error_sequence
from ml_switcheroo_ir.types import AttributeValue, DType, TensorSpec, describe_value_type
from ml_switcheroo_ir.validation.core import (
    CORE_MLIR_DIALECTS,
    ValidationError,
    ValidationLevel,
    ValidationRule,
    combine_rules,
    errors_from_validation,
    rule_from_errors_fn,
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
    HARDWARE_SEMANTICS_BY_DOMAIN,
    validate_isa_instruction_node,
    validate_sass_scoreboarding_sequence,
    validate_vopd_pairing,
)
from ml_switcheroo_ir.validation.quantization import validate_quantization_node

__all__ = [
    "Validator",
    "ValidatorConfig",
    "ValidatorKwargs",
    "check_attribute_type_value",
]


def _is_valid_dtype_attr(value: AttributeValue) -> bool:
    """Return True if ``value`` is a valid ``DType``, ONNX TensorProto elem_type int, or DType string.

    Args:
        value (AttributeValue): Candidate attribute value.

    Returns:
        bool: True if ``value`` represents a valid element dtype.
    """
    if isinstance(value, DType):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return 1 <= value <= 40
    if isinstance(value, str):
        try:
            DType.from_str(value)
            return True
        except ValueError:
            return False
    return False


def _is_valid_tensor_attr(value: AttributeValue) -> bool:
    """Return True if ``value`` is a valid constant tensor literal, ``bytes``, or ``TensorSpec``.

    Args:
        value (AttributeValue): Candidate attribute value.

    Returns:
        bool: True if ``value`` is a valid Tensor attribute payload.
    """
    return isinstance(value, (TensorSpec, bytes, int, float, bool, list, tuple))


def _is_valid_graph_attr(value: AttributeValue | LogicalGraph) -> bool:
    """Return True if ``value`` is a ``LogicalGraph`` or serialized graph dictionary.

    Args:
        value (AttributeValue | LogicalGraph): Candidate attribute value.

    Returns:
        bool: True if ``value`` represents a nested graph region.
    """
    if isinstance(value, LogicalGraph):
        return True
    return isinstance(value, Mapping) and "nodes" in value and isinstance(value["nodes"], Mapping)


_ScalarSchemaAttr: typing_extensions.TypeAlias = typing.Literal[
    AttributeType.INT,
    AttributeType.FLOAT,
    AttributeType.STR,
    AttributeType.BOOL,
    AttributeType.DTYPE,
]
_CollectionSchemaAttr: typing_extensions.TypeAlias = typing.Literal[
    AttributeType.DICT,
    AttributeType.LIST_INT,
    AttributeType.LIST_FLOAT,
    AttributeType.LIST_STR,
    AttributeType.LIST,
]
_ObjectSchemaAttr: typing_extensions.TypeAlias = typing.Literal[
    AttributeType.TENSOR,
    AttributeType.GRAPH,
    AttributeType.ANY,
]


def _check_object_attr_type(attr_type: _ObjectSchemaAttr, value: AttributeValue) -> bool:
    """Exhaustively validate object/polymorphic ``AttributeType`` members against ``value``.

    Args:
        attr_type (_ObjectSchemaAttr): Object or wildcard attribute enum variant.
        value (AttributeValue): Candidate attribute value.

    Returns:
        bool: True iff ``value`` satisfies ``attr_type``.
    """
    ok: bool
    if attr_type is AttributeType.TENSOR:
        ok = _is_valid_tensor_attr(value)
    elif attr_type is AttributeType.GRAPH:
        ok = _is_valid_graph_attr(value)
    elif attr_type is AttributeType.ANY:
        ok = True
    else:
        typing_extensions.assert_never(attr_type)
    return ok


def _check_non_scalar_attr_type(
    attr_type: _CollectionSchemaAttr | _ObjectSchemaAttr,
    value: AttributeValue,
) -> bool:
    """Cascade collection ``AttributeType`` members and delegate remaining object variants.

    Args:
        attr_type (_CollectionSchemaAttr | _ObjectSchemaAttr): Non-scalar attribute enum variant.
        value (AttributeValue): Candidate attribute value.

    Returns:
        bool: True iff ``value`` satisfies ``attr_type``.
    """
    ok: bool
    if attr_type is AttributeType.DICT:
        ok = isinstance(value, Mapping)
    elif attr_type is AttributeType.LIST_INT:
        ok = isinstance(value, (list, tuple)) and all(is_non_bool_int(x) for x in value)
    elif attr_type is AttributeType.LIST_FLOAT:
        ok = isinstance(value, (list, tuple)) and all(
            isinstance(x, (float, int)) and not isinstance(x, bool) for x in value
        )
    elif attr_type is AttributeType.LIST_STR:
        ok = isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value)
    elif attr_type is AttributeType.LIST:
        ok = isinstance(value, (list, tuple))
    else:
        ok = _check_object_attr_type(attr_type, value)
    return ok


def check_attribute_type_value(attr_type: AttributeType, value: AttributeValue) -> bool:
    """Exhaustively check whether ``value`` conforms to ``attr_type`` with ``assert_never`` safety.

    Args:
        attr_type (AttributeType): Canonical operator attribute schema type.
        value (AttributeValue): Candidate node attribute value.

    Returns:
        bool: True iff ``value`` conforms to ``attr_type``.
    """
    ok: bool
    if attr_type is AttributeType.INT:
        ok = is_non_bool_int(value)
    elif attr_type is AttributeType.FLOAT:
        ok = isinstance(value, (float, int)) and not isinstance(value, bool)
    elif attr_type is AttributeType.STR:
        ok = isinstance(value, str)
    elif attr_type is AttributeType.BOOL:
        ok = isinstance(value, bool)
    elif attr_type is AttributeType.DTYPE:
        ok = _is_valid_dtype_attr(value)
    else:
        ok = _check_non_scalar_attr_type(attr_type, value)
    return ok


_VARIADIC_OPERAND_OPS: frozenset[str] = frozenset(
    {
        "Concat",
        "concat",
        "Sum",
        "Max",
        "Min",
        "Mean",
        "SequenceConstruct",
        "Loop",
        "Scan",
        "If",
        "scf.for",
        "scf.while",
        "scf.if",
        "scf.yield",
        "func.call",
        "func.return",
        "linalg.generic",
        "call",
        "return",
        "yield",
        "for",
        "while",
        "if",
    }
)

_OPTIONAL_TRAILING_MIN_INPUTS: dict[str, int] = {
    "Gemm": 2,
    "Conv": 2,
    "ConvTranspose": 2,
    "BatchNormalization": 5,
    "LayerNormalization": 2,
    "LayerNorm": 1,
    "GroupNormalization": 3,
    "GroupNorm": 1,
    "InstanceNormalization": 3,
    "RMSNorm": 1,
    "Reshape": 1,
    "Squeeze": 1,
    "Unsqueeze": 1,
    "Split": 1,
    "Slice": 1,
    "Clip": 1,
    "Pad": 1,
    "Resize": 1,
    "Dropout": 1,
    "TopK": 1,
    "CumSum": 1,
    "ReduceSum": 1,
    "ReduceMean": 1,
    "ReduceMax": 1,
    "ReduceMin": 1,
    "ReduceProd": 1,
    "ScaledDotProductAttention": 3,
    "RoPE": 1,
    "Where": 3,
}


def _collect_ssa_scope(graph: LogicalGraph, outer_scope: set[str] | None = None) -> set[str]:
    """Collect all valid SSA value identifiers visible inside ``graph``.

    Args:
        graph (LogicalGraph): Target graph scope.
        outer_scope (set[str] | None): Enclosing parent SSA names visible to nested subgraphs.

    Returns:
        set[str]: All valid node IDs, inputs, initializers, and multi-output port names.
    """
    scope: set[str] = set(outer_scope) if outer_scope is not None else set()
    scope.update(graph.nodes.keys())
    scope.update(graph.inputs)
    scope.update(graph.initializers.keys())
    for node_id, node in graph.nodes.items():
        scope.update(node.outputs)
        port_count = max(len(node.outputs), len(node.output_specs))
        for idx in range(port_count):
            scope.add(f"{node_id}:{idx}")
    return scope


_SECONDARY_CATALOG_BY_DOMAIN: typing.Final[dict[str, DialectCatalogWitness[OpSchema]]] = {
    str(dom): witness
    for witness in ALL_DIALECT_CATALOG_WITNESSES
    if witness.registry_name != "ONNX_REGISTRY"
    for dom in witness.domains
}


@dataclass(frozen=True)
class ValidatorConfig:
    """Immutable configuration struct for ``Validator`` construction.

    Attributes:
        registry (Mapping[str, OpSchema] | None): Primary ONNX/canonical operator registry.
        custom_registry (Mapping[str, OpSchema] | None): Custom operator registry override.
        stablehlo_registry (Mapping[str, OpSchema] | None): StableHLO operator registry override.
        mlir_registry (Mapping[str, OpSchema] | None): Core MLIR operator registry override.
        state_registry (Mapping[str, OpSchema] | None): State-mutation operator registry override.
        level (ValidationLevel): Minimum diagnostic severity threshold.
        strict (bool): Convenience flag promoting ``level`` to ``ValidationLevel.STRICT``.
        grounding_validator (GroundingValidator | None): Optional snapshot grounding validator.
    """

    registry: Mapping[str, OpSchema] | None = None
    custom_registry: Mapping[str, OpSchema] | None = None
    stablehlo_registry: Mapping[str, OpSchema] | None = None
    mlir_registry: Mapping[str, OpSchema] | None = None
    state_registry: Mapping[str, OpSchema] | None = None
    level: ValidationLevel = ValidationLevel.WARNING
    strict: bool = False
    grounding_validator: GroundingValidator | None = None


class ValidatorKwargs(typing_extensions.TypedDict, total=False):
    """Typed keyword options accepted by ``Validator.__init__`` alongside ``ValidatorConfig``.

    Attributes:
        registry (Mapping[str, OpSchema] | None): Primary ONNX/canonical operator registry.
        custom_registry (Mapping[str, OpSchema] | None): Custom operator registry override.
        stablehlo_registry (Mapping[str, OpSchema] | None): StableHLO operator registry override.
        mlir_registry (Mapping[str, OpSchema] | None): Core MLIR operator registry override.
        state_registry (Mapping[str, OpSchema] | None): State-mutation operator registry override.
        level (ValidationLevel): Minimum diagnostic severity threshold.
        strict (bool): Convenience flag promoting ``level`` to ``ValidationLevel.STRICT``.
        grounding_validator (GroundingValidator | None): Optional snapshot grounding validator.
    """

    registry: Mapping[str, OpSchema] | None
    custom_registry: Mapping[str, OpSchema] | None
    stablehlo_registry: Mapping[str, OpSchema] | None
    mlir_registry: Mapping[str, OpSchema] | None
    state_registry: Mapping[str, OpSchema] | None
    level: ValidationLevel
    strict: bool
    grounding_validator: GroundingValidator | None


class Validator:
    """Validates LogicalGraph and LogicalNode instances against schemas without mutating inputs by default."""

    def __init__(
        self,
        config: ValidatorConfig | None = None,
        **kwargs: typing_extensions.Unpack[ValidatorKwargs],
    ) -> None:
        """Initialize the validator from ``ValidatorConfig`` and/or typed keyword overrides.

        Args:
            config (ValidatorConfig | None): Optional frozen configuration struct.
            **kwargs (typing_extensions.Unpack[ValidatorKwargs]): Backwards-compatible keyword overrides.
        """
        base = config if config is not None else ValidatorConfig()
        registry = kwargs["registry"] if "registry" in kwargs else base.registry
        custom_registry = kwargs["custom_registry"] if "custom_registry" in kwargs else base.custom_registry
        stablehlo_registry = kwargs["stablehlo_registry"] if "stablehlo_registry" in kwargs else base.stablehlo_registry
        mlir_registry = kwargs["mlir_registry"] if "mlir_registry" in kwargs else base.mlir_registry
        state_registry = kwargs["state_registry"] if "state_registry" in kwargs else base.state_registry
        level = kwargs["level"] if "level" in kwargs else base.level
        strict = kwargs["strict"] if "strict" in kwargs else base.strict
        grounding_validator = (
            kwargs["grounding_validator"] if "grounding_validator" in kwargs else base.grounding_validator
        )

        self.registry: Mapping[str, OpSchema] = ONNX_REGISTRY if registry is None else registry
        self.custom_registry: Mapping[str, OpSchema] = (
            CUSTOM_OPS_REGISTRY if custom_registry is None else custom_registry
        )
        self.stablehlo_registry: Mapping[str, OpSchema] = (
            STABLEHLO_REGISTRY if stablehlo_registry is None else stablehlo_registry
        )
        self.mlir_registry: Mapping[str, OpSchema] = MLIR_REGISTRY if mlir_registry is None else mlir_registry
        self.state_registry: Mapping[str, OpSchema] = STATE_OPS_REGISTRY if state_registry is None else state_registry
        self.collective_registry: Mapping[str, OpSchema] = COLLECTIVE_OPS_REGISTRY
        self.quantization_registry: Mapping[str, OpSchema] = QUANTIZATION_OPS_REGISTRY

        self.level = ValidationLevel.STRICT if strict else level
        self.grounding_validator = grounding_validator

    def _resolve_secondary_lookup(self, domain: str) -> typing.Callable[[str], OpSchema | None] | None:
        """Resolve the secondary operator lookup callable corresponding to a dialect domain.

        Args:
            domain (str): Operator domain identifier.

        Returns:
            typing.Callable[[str], OpSchema | None] | None: Schema lookup function or ``None``.
        """
        override_map: dict[str, Mapping[str, OpSchema]] = {
            "ml.switcheroo.custom": self.custom_registry,
            "stablehlo": self.stablehlo_registry,
            "mlir": self.mlir_registry,
            "mlir.arith": self.mlir_registry,
            "mlir.math": self.mlir_registry,
            "mlir.tensor": self.mlir_registry,
            "mlir.linalg": self.mlir_registry,
            "mlir.scf": self.mlir_registry,
            "mlir.func": self.mlir_registry,
            "collective": self.collective_registry,
            "ml.switcheroo.collective": self.collective_registry,
            "quantization": self.quantization_registry,
            "ml.switcheroo.quant": self.quantization_registry,
            "ml.switcheroo.quantization": self.quantization_registry,
            "state": self.state_registry,
            "ml.switcheroo.state": self.state_registry,
        }
        overridden = override_map.get(domain)
        if overridden is not None:
            return overridden.get
        witness = _SECONDARY_CATALOG_BY_DOMAIN.get(domain)
        return witness.lookup if witness is not None else None

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
        """Retrieve the schema for a node across registered domains and unified MLIR dialects.

        Args:
            node (LogicalNode): The node to inspect.

        Returns:
            OpSchema | None: The schema if found, or None.
        """
        if node.domain in ("ad", "ml.switcheroo.ad") and node.op_type in ("ZeroTangent", "NoTangent"):
            return OpSchema(
                name=node.op_type,
                domain=node.domain,
                version=1,
                attributes={},
                inputs=[],
                outputs=["tangent"],
            )

        if node.domain == "mlir" or node.domain.startswith("mlir.") or node.domain in CORE_MLIR_DIALECTS:
            override = self.registry.get(node.op_type)
            if override is not None and override.domain == node.domain:
                return override
            return resolve_mlir_op_schema(node.domain, node.op_type, self.mlir_registry)

        if node.domain != "ai.onnx":
            s = self.registry.get(node.op_type)
            if s is not None and s.domain == node.domain:
                return s
            secondary_lookup = self._resolve_secondary_lookup(node.domain)
            return secondary_lookup(node.op_type) if secondary_lookup is not None else self.registry.get(node.op_type)

        return self.registry.get(node.op_type)

    def _validate_core_mlir_dialect(self, node: LogicalNode) -> list[ValidationError]:
        """Validate a node belonging to one of the unified CORE_MLIR_DIALECTS.

        Args:
            node (LogicalNode): Node in a core MLIR dialect.

        Returns:
            list[ValidationError]: Detected kind or operand-count errors.
        """
        errors: list[ValidationError] = []
        dialect_ops = CORE_MLIR_DIALECTS[node.domain]
        resolved_mlir = resolve_mlir_op_schema(node.domain, node.op_type, self.mlir_registry)
        short_op = node.op_type.split(".", 1)[1] if node.op_type.startswith(f"{node.domain}.") else node.op_type
        if short_op not in dialect_ops and resolved_mlir is None and node.op_type not in self.registry:
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="kind",
                    message=f"Operator '{node.op_type}' not found in MLIR dialect '{node.domain}'.",
                    level=ValidationLevel.ERROR,
                )
            )
        else:
            expected_inputs = dialect_ops.get(short_op)
            if expected_inputs is None and resolved_mlir is not None:
                expected_inputs = len(resolved_mlir.inputs)
            if expected_inputs is not None and len(node.inputs) < expected_inputs:
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute="inputs",
                        message=(
                            f"MLIR op '{node.domain}.{short_op}' expects at least "
                            f"{expected_inputs} operands, got {len(node.inputs)}."
                        ),
                        level=ValidationLevel.ERROR,
                    )
                )
        return errors

    def _validate_secondary_domain_kind(
        self, node: LogicalNode, secondary_lookup: typing.Callable[[str], OpSchema | None]
    ) -> list[ValidationError]:
        """Validate node kind against a secondary domain lookup callable and hardware primitive table.

        Args:
            node (LogicalNode): Node to validate.
            secondary_lookup (typing.Callable[[str], OpSchema | None]): Secondary domain lookup function.

        Returns:
            list[ValidationError]: Validation error list (empty if valid).
        """
        if node.domain == "mlir" or node.domain.startswith("mlir."):
            reg_mlir = self.registry.get(node.op_type)
            if (reg_mlir is not None and reg_mlir.domain == node.domain) or (
                resolve_mlir_op_schema(node.domain, node.op_type, self.mlir_registry) is not None
            ):
                return []

        extra_primitives: (
            Mapping[str, RdnaPrimitiveSpec] | Mapping[str, SassPrimitiveSpec] | Mapping[str, WgslPrimitiveSpec]
        ) = {}
        if node.domain in ("amd_rdna", "rdna"):
            extra_primitives = RDNA_INSTRUCTION_PRIMITIVES
        elif node.domain in ("nvidia_sass", "sass"):
            extra_primitives = SASS_INSTRUCTION_PRIMITIVES
        elif node.domain in ("webgpu_wgsl", "wgsl"):
            extra_primitives = WGSL_PRIMITIVE_SIGNATURES

        reg_schema = self.registry.get(node.op_type)
        if (
            secondary_lookup(node.op_type) is not None
            or node.op_type in extra_primitives
            or (reg_schema is not None and reg_schema.domain == node.domain)
        ):
            return []

        lvl = (
            ValidationLevel.WARNING
            if node.domain == "ml.switcheroo.custom" and self.level != ValidationLevel.STRICT
            else ValidationLevel.ERROR
        )
        custom_labels = {
            "collective": "collective registry.",
            "ml.switcheroo.collective": "collective registry.",
            "quantization": "quantization registry.",
            "ml.switcheroo.quant": "quantization registry.",
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

    def validate_kind(self, node: LogicalNode) -> list[ValidationError]:
        """Validate that the node's kind exists in the registry for its domain.

        Args:
            node (LogicalNode): The node to validate.

        Returns:
            list[ValidationError]: A list of errors found.
        """
        if node.op_type in ("Input", "Output"):
            return []

        if node.domain == "ai.onnx":
            return (
                [
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in domain '{node.domain}'.",
                        level=ValidationLevel.ERROR,
                    )
                ]
                if node.op_type not in self.registry
                else []
            )

        if node.domain in ("ad", "ml.switcheroo.ad"):
            return (
                [
                    ValidationError(
                        node_id=node.id,
                        attribute="kind",
                        message=f"Operator '{node.op_type}' not found in autodiff sentinel registry.",
                        level=ValidationLevel.ERROR,
                    )
                ]
                if node.op_type not in ("ZeroTangent", "NoTangent")
                else []
            )

        secondary_lookup = self._resolve_secondary_lookup(node.domain)
        if secondary_lookup is not None:
            return self._validate_secondary_domain_kind(node, secondary_lookup)

        if node.domain in CORE_MLIR_DIALECTS:
            return self._validate_core_mlir_dialect(node)

        return (
            [
                ValidationError(
                    node_id=node.id,
                    attribute="domain",
                    message=f"Unrecognized domain '{node.domain}'.",
                    level=ValidationLevel.ERROR,
                )
            ]
            if node.domain not in ("ai.custom", "custom") and node.op_type not in self.registry
            else []
        )

    def validate_required_attributes(self, node: LogicalNode) -> list[ValidationError]:
        """Validate that all required attributes for the node's kind are present.

        Args:
            node (LogicalNode): The node to validate.

        Returns:
            list[ValidationError]: A list of errors found.
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
        """Validate that attributes match every canonical ``AttributeType`` in the schema.

        Args:
            node (LogicalNode): The node to validate.

        Returns:
            list[ValidationError]: A list of errors found.
        """
        errors: list[ValidationError] = []
        schema = self._get_schema(node)
        if schema is None:
            return errors

        isa_sem = HARDWARE_SEMANTICS_BY_DOMAIN.get(node.domain)
        allowed_isa_keys = isa_sem.allowed_attributes if isa_sem is not None else frozenset()

        for key, value in node.attributes.items():
            if key not in schema.attributes:
                if key in allowed_isa_keys:
                    continue
                lvl = ValidationLevel.ERROR if self.level == ValidationLevel.STRICT else ValidationLevel.WARNING
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
            canonical_type = normalize_attribute_type(attr_schema.type)
            if value is None and not attr_schema.required and attr_schema.default is None:
                continue
            if not check_attribute_type_value(canonical_type, value):
                errors.append(
                    ValidationError(
                        node_id=node.id,
                        attribute=key,
                        message=(
                            f"Attribute '{key}' has invalid type. Expected {canonical_type.value}, "
                            f"got {describe_value_type(value)}."
                        ),
                        level=ValidationLevel.ERROR,
                    )
                )

        return errors

    def validate_sharding(self, node: LogicalNode, mesh: LogicalMesh | None) -> list[ValidationError]:
        """Validate that node sharding specification is compatible with the logical mesh.

        Args:
            node (LogicalNode): The node whose sharding to validate.
            mesh (LogicalMesh | None): The logical mesh defined on the graph.

        Returns:
            list[ValidationError]: A list of sharding validation errors.
        """
        return validate_sharding_node(node, mesh)

    def populate_defaults(self, node: LogicalNode) -> None:
        """Explicitly inject missing optional attributes with default values from the schema.

        Args:
            node (LogicalNode): The node to mutate in place.
        """
        schema = self._get_schema(node)
        if schema is None:
            return

        for attr_name, attr_schema in schema.attributes.items():
            if not attr_schema.required and attr_schema.default is not None and attr_name not in node.attributes:
                node.attributes[attr_name] = attr_schema.default

    def validate_isa_instruction(self, node: LogicalNode) -> list[ValidationError]:
        """Validate GPU accelerator ISA instructions (AMD RDNA, NVIDIA SASS, and WebGPU WGSL).

        Args:
            node (LogicalNode): The node representing an ISA instruction or WGSL primitive.

        Returns:
            list[ValidationError]: A list of detected validation errors or warnings.
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

    def _collect_grounding_node_errors(self, node: LogicalNode) -> list[ValidationError]:
        """Run ``grounding_validator`` on ``node`` preserving ``suggested_fix``.

        Args:
            node (LogicalNode): Node to audit.

        Returns:
            list[ValidationError]: Level-mapped grounding errors with ``suggested_fix`` preserved.
        """
        if self.grounding_validator is None or self.level == ValidationLevel.LENIENT:
            return []
        target_lvl = ValidationLevel.ERROR if self.level == ValidationLevel.STRICT else ValidationLevel.WARNING
        return [
            ValidationError(
                node_id=g_err.node_id,
                attribute=g_err.attribute,
                message=g_err.message,
                level=target_lvl,
                suggested_fix=g_err.suggested_fix,
            )
            for g_err in self.grounding_validator.validate_grounding(node)
        ]

    def validate_node(
        self,
        node: LogicalNode,
        mesh: LogicalMesh | None = None,
        populate_defaults: bool = False,
    ) -> list[ValidationError]:
        """Validate a single LogicalNode (and any nested subgraphs) without mutating by default.

        Args:
            node (LogicalNode): The node to validate.
            mesh (LogicalMesh | None): Optional logical device mesh for sharding validation.
            populate_defaults (bool): Whether to populate missing default attributes (default: False).

        Returns:
            list[ValidationError]: Aggregated list of validation errors and warnings for the node.
        """
        errors: list[ValidationError] = []
        errors.extend(self.validate_kind(node))
        errors.extend(self.validate_required_attributes(node))
        if populate_defaults:
            self.populate_defaults(node)
        errors.extend(self.validate_attribute_types(node))
        errors.extend(self.validate_sharding(node, mesh))
        if node.domain in ("amd_rdna", "rdna", "nvidia_sass", "sass", "webgpu_wgsl", "wgsl"):
            errors.extend(self.validate_isa_instruction(node))
        if (
            node.domain in ("quantization", "ml.switcheroo.quant", "ml.switcheroo.quantization")
            or node.op_type.startswith("quantization.")
            or node.op_type in ("block_quantize", "dequantize_grouped_int4")
        ):
            errors.extend(self.validate_quantization(node))

        if self.level == ValidationLevel.STRICT and node.shape_metadata is None and node.op_type != "NoTangent":
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="shape_metadata",
                    message=f"Node '{node.id}' missing required shape_metadata in STRICT validation mode.",
                    level=ValidationLevel.ERROR,
                )
            )

        errors.extend(self._collect_grounding_node_errors(node))
        return self._filter_by_level(errors)

    def _validate_graph_recursive(
        self,
        graph: LogicalGraph,
        populate_defaults: bool,
        outer_scope: set[str] | None,
    ) -> list[ValidationError]:
        """Recursively validate ``graph`` and all nested ``node.subgraphs``.

        Args:
            graph (LogicalGraph): Current graph scope.
            populate_defaults (bool): Whether to populate missing defaults in place.
            outer_scope (set[str] | None): Enclosing parent SSA names visible to nested subgraphs.

        Returns:
            list[ValidationError]: Aggregated validation errors across ``graph`` and nested subgraphs.
        """
        errors: list[ValidationError] = []
        current_scope = _collect_ssa_scope(graph, outer_scope)

        logical_mesh = graph.mesh if isinstance(graph.mesh, LogicalMesh) else None
        for node in graph.nodes.values():
            errors.extend(self.validate_node(node, logical_mesh, populate_defaults=populate_defaults))
            for sub_val in node.subgraphs.values():
                if isinstance(sub_val, LogicalGraph):
                    errors.extend(
                        self._validate_graph_recursive(
                            sub_val,
                            populate_defaults=populate_defaults,
                            outer_scope=current_scope,
                        )
                    )

        errors.extend(self._validate_edges_scoped(graph, outer_scope=outer_scope, recurse_subgraphs=False))

        sass_nodes = [n for n in graph.nodes.values() if n.domain in ("nvidia_sass", "sass")]
        if sass_nodes:
            errors.extend(self.validate_sass_scoreboarding(sass_nodes))

        errors.extend(self.validate_sharding_propagation(graph))
        errors.extend(self.validate_pipeline_and_checkpointing(graph))
        return errors

    def validate_graph(
        self,
        graph: LogicalGraph,
        populate_defaults: bool = False,
    ) -> list[ValidationError]:
        """Validate all nodes, nested subgraphs, sharding, and edges in a LogicalGraph.

        Args:
            graph (LogicalGraph): The graph to validate.
            populate_defaults (bool): Whether to populate missing default attributes (default: False).

        Returns:
            list[ValidationError]: A list of all aggregated errors.
        """
        return self._filter_by_level(
            self._validate_graph_recursive(graph, populate_defaults=populate_defaults, outer_scope=None)
        )

    def build_graph_rules(self, populate_defaults: bool = False) -> tuple[ValidationRule[LogicalGraph], ...]:
        """Return the declarative ``ValidationRule[LogicalGraph]`` pipeline for this ``Validator``.

        Args:
            populate_defaults (bool): Whether node validation populates default schema attributes.

        Returns:
            tuple[ValidationRule[LogicalGraph], ...]: Composable graph-level validation rules.
        """
        return (
            rule_from_errors_fn(
                "graph_structure_and_nodes",
                lambda g: self._filter_by_level(
                    self._validate_graph_recursive(g, populate_defaults=populate_defaults, outer_scope=None)
                ),
            ),
        )

    def validate_graph_applicative(
        self,
        graph: LogicalGraph,
        populate_defaults: bool = False,
    ) -> Validation[NonEmpty[ValidationError], ProvenGraph[SchemaValidPhase]]:
        """Applicatively validate ``graph`` without short-circuiting and mint ``ProvenGraph[SchemaValidPhase]``.

        Args:
            graph (LogicalGraph): Computation graph to validate.
            populate_defaults (bool): Whether to populate missing default attributes on the cloned snapshot.

        Returns:
            Validation[NonEmpty[ValidationError], ProvenGraph[SchemaValidPhase]]:
                ``Valid`` proven graph when zero diagnostics pass active ``self.level`` filter,
                or ``Invalid`` non-empty batch of all accumulated ``ValidationError`` items.
        """
        cloned = graph.clone()
        composite = combine_rules(self.build_graph_rules(populate_defaults=populate_defaults), name="validate_graph")
        errs = errors_from_validation(composite.run(cloned))
        proven: ProvenGraph[SchemaValidPhase] = ProvenGraph.mint(
            cloned,
            ordered_nodes=tuple(cloned.nodes.values()),
        )
        return from_error_sequence(errs, proven)

    def validate_vopd_pairing(self, opX: LogicalNode, opY: LogicalNode) -> list[ValidationError]:
        """Validate AMD RDNA3 / GFX11 dual-issue VOPD instruction pairing rules.

        Args:
            opX (LogicalNode): The instruction proposed for VOPD Slot X.
            opY (LogicalNode): The instruction proposed for VOPD Slot Y.

        Returns:
            list[ValidationError]: Detected VOPD pairing violations.
        """
        return validate_vopd_pairing(opX, opY)

    def validate_sass_scoreboarding(self, nodes: Sequence[LogicalNode]) -> list[ValidationError]:
        """Validate NVIDIA SASS scoreboard dependency latencies between consecutive instructions.

        Args:
            nodes (Sequence[LogicalNode]): Sequential list of SASS instructions to audit.

        Returns:
            list[ValidationError]: Detected scoreboard hazards or dependency violations.
        """
        return validate_sass_scoreboarding_sequence(nodes, self.level)

    def validate_sharding_propagation(self, graph: LogicalGraph) -> list[ValidationError]:
        """Validate SPaDe/GSPMD sharding propagation invariants across graph operations.

        Args:
            graph (LogicalGraph): Computational graph to validate.

        Returns:
            list[ValidationError]: Detected sharding propagation violations.
        """
        return validate_sharding_propagation_graph(graph, self.level)

    def validate_pipeline_and_checkpointing(self, graph: LogicalGraph) -> list[ValidationError]:
        """Validate pipeline parallel stage orderings, boundary markers, and activation checkpointing tags.

        Args:
            graph (LogicalGraph): Computational graph to validate.

        Returns:
            list[ValidationError]: Detected pipeline or checkpointing violations.
        """
        return validate_pipeline_and_checkpointing_graph(graph)

    def estimate_communication_volume(
        self,
        node: LogicalNode,
        mesh: LogicalMesh | None = None,
        symbol_bindings: Mapping[str, int] | None = None,
    ) -> int:
        """Calculate analytical communication volume in bytes for a collective communication node.

        Args:
            node (LogicalNode): The collective operation node.
            mesh (LogicalMesh | None): The logical device mesh defined on the graph.
            symbol_bindings (Mapping[str, int] | None): Optional bindings for symbolic dimensions.

        Returns:
            int: Analytical volume in bytes transferred per device during the collective.
        """
        return estimate_communication_volume(node, mesh, symbol_bindings=symbol_bindings)

    def estimate_graph_communication_volume(
        self,
        graph: LogicalGraph,
        symbol_bindings: Mapping[str, int] | None = None,
    ) -> CommunicationVolumeReport:
        """Aggregate analytical communication volume across all collective operators in a graph.

        Args:
            graph (LogicalGraph): The computational graph to analyze.
            symbol_bindings (Mapping[str, int] | None): Optional bindings for symbolic dimensions.

        Returns:
            CommunicationVolumeReport: Report with total communication volume in bytes,
                and breakdowns by mesh axis and operator type.
        """
        return estimate_graph_communication_volume(graph, symbol_bindings=symbol_bindings)

    def validate(
        self,
        target: LogicalGraph | LogicalNode,
        raise_on_error: bool = False,
        populate_defaults: bool = False,
    ) -> list[ValidationError]:
        """Validate a LogicalGraph or LogicalNode instance without mutating inputs by default.

        Args:
            target (LogicalGraph | LogicalNode): Graph or node instance to validate.
            raise_on_error (bool): If True, raises the first fatal ValidationError encountered.
            populate_defaults (bool): Whether to populate missing default attributes (default: False).

        Returns:
            list[ValidationError]: List of validation errors and warnings.

        Raises:
            ValidationError: If ``raise_on_error`` is True and a fatal ValidationError is found.
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
                    suggested_fix=first_err.suggested_fix,
                )

        return errors

    def _validate_node_operand_arity(self, node: LogicalNode) -> list[ValidationError]:
        """Check positional operand minimum/maximum arity when ``node.inputs`` are wired.

        Args:
            node (LogicalNode): Node to check.

        Returns:
            list[ValidationError]: Arity violation errors if any.
        """
        if not node.inputs or node.op_type in ("Input", "Output") or node.op_type in _VARIADIC_OPERAND_OPS:
            return []
        if node.domain in ("amd_rdna", "rdna", "nvidia_sass", "sass", "webgpu_wgsl", "wgsl", "nvidia_ptx", "ptx"):
            return []
        schema = self._get_schema(node)
        if schema is None or not schema.inputs:
            return []

        max_arity = len(schema.inputs)
        min_arity = min(_OPTIONAL_TRAILING_MIN_INPUTS.get(node.op_type, max_arity), max_arity)
        got = len(node.inputs)
        if got < min_arity:
            return [
                ValidationError(
                    node_id=node.id,
                    attribute="inputs",
                    message=(f"Operator '{node.op_type}' expects at least {min_arity} positional input(s), got {got}."),
                    level=ValidationLevel.ERROR,
                )
            ]
        if got > max_arity:
            return [
                ValidationError(
                    node_id=node.id,
                    attribute="inputs",
                    message=(f"Operator '{node.op_type}' accepts at most {max_arity} positional input(s), got {got}."),
                    level=ValidationLevel.ERROR,
                )
            ]
        return []

    @staticmethod
    def _is_resolvable_ssa(name: str, graph: LogicalGraph, node_ids: set[str], inherited: set[str]) -> bool:
        """Return True if ``name`` resolves to a node, SSA output, input, initializer, or outer scope.

        Args:
            name (str): Candidate SSA identifier.
            graph (LogicalGraph): Current graph scope.
            node_ids (set[str]): Keys in ``graph.nodes``.
            inherited (set[str]): Enclosing parent SSA identifiers.

        Returns:
            bool: True if ``name`` is defined.
        """
        return (
            name in node_ids
            or graph.get_output_producer(name) is not None
            or name in graph.inputs
            or name in graph.initializers
            or name in inherited
        )

    def _check_graph_nodes_and_outputs(
        self,
        graph: LogicalGraph,
        inherited: set[str],
    ) -> list[ValidationError]:
        """Verify ``key == node.id``, operand references, operand arity, and ``graph.outputs`` resolution.

        Args:
            graph (LogicalGraph): Graph scope to inspect.
            inherited (set[str]): Outer enclosing SSA identifiers.

        Returns:
            list[ValidationError]: Detected node/input/output/arity errors.
        """
        errors: list[ValidationError] = []
        node_ids = set(graph.nodes.keys())
        for dict_key, node in graph.nodes.items():
            if dict_key != node.id:
                errors.append(
                    ValidationError(
                        node_id=dict_key,
                        attribute="id",
                        message=f"Graph nodes key '{dict_key}' does not match LogicalNode.id '{node.id}'.",
                        level=ValidationLevel.ERROR,
                    )
                )
            for inp in node.inputs:
                if not self._is_resolvable_ssa(inp, graph, node_ids, inherited):
                    errors.append(
                        ValidationError(
                            node_id=dict_key,
                            attribute="inputs",
                            message=f"Node input '{inp}' does not exist.",
                            level=ValidationLevel.ERROR,
                        )
                    )
            errors.extend(self._validate_node_operand_arity(node))

        for out_name in graph.outputs:
            if not self._is_resolvable_ssa(out_name, graph, node_ids, inherited):
                errors.append(
                    ValidationError(
                        node_id=graph.name,
                        attribute="outputs",
                        message=f"Graph output '{out_name}' does not resolve to any node, SSA output, or input.",
                        level=ValidationLevel.ERROR,
                    )
                )
        return errors

    @staticmethod
    def _check_multi_output_edge_ports(graph: LogicalGraph) -> list[ValidationError]:
        """Verify that every ``LogicalEdge.source_idx`` is in bounds for multi-output producers.

        Args:
            graph (LogicalGraph): Target graph to check.

        Returns:
            list[ValidationError]: Out-of-bounds ``source_idx`` errors.
        """
        errors: list[ValidationError] = []
        for edge in graph.edges:
            producer = graph.nodes.get(edge.source)
            if producer is None:
                prod_tuple = graph.get_producing_output_index(edge.source)
                if prod_tuple is not None:
                    producer = prod_tuple[0]

            if producer is not None and producer.has_multiple_outputs:
                num_outputs = len(producer.outputs) if len(producer.outputs) > 1 else len(producer.output_specs)
                if edge.source_idx is not None and (edge.source_idx < 0 or edge.source_idx >= num_outputs):
                    errors.append(
                        ValidationError(
                            node_id=edge.target,
                            attribute="edges",
                            message=(
                                f"Edge references invalid source_idx {edge.source_idx} (out of bounds) "
                                f"for multi-output node '{producer.id}' (has {num_outputs} outputs)."
                            ),
                            level=ValidationLevel.ERROR,
                        )
                    )
        return errors

    def _validate_edges_scoped(
        self,
        graph: LogicalGraph,
        *,
        outer_scope: set[str] | None,
        recurse_subgraphs: bool,
    ) -> list[ValidationError]:
        """Validate node ID keys, inputs, outputs, port indices, arity, acyclicity, and subgraphs.

        Args:
            graph (LogicalGraph): Target graph to check.
            outer_scope (set[str] | None): Enclosing parent SSA names visible to nested subgraphs.
            recurse_subgraphs (bool): Whether to recursively validate ``node.subgraphs``.

        Returns:
            list[ValidationError]: Detected edge, topology, output, cycle, or arity errors.
        """
        inherited = outer_scope if outer_scope is not None else set()
        errors = self._check_graph_nodes_and_outputs(graph, inherited)
        errors.extend(self._check_multi_output_edge_ports(graph))

        try:
            topological_sort(graph, strict=True)
        except CyclicGraphError as exc:
            errors.append(
                ValidationError(
                    node_id=graph.name,
                    attribute="edges",
                    message=str(exc),
                    level=ValidationLevel.ERROR,
                )
            )

        if recurse_subgraphs:
            current_scope = _collect_ssa_scope(graph, outer_scope)
            for node in graph.nodes.values():
                for sub_val in node.subgraphs.values():
                    if isinstance(sub_val, LogicalGraph):
                        errors.extend(
                            self._validate_edges_scoped(
                                sub_val,
                                outer_scope=current_scope,
                                recurse_subgraphs=True,
                            )
                        )
        return errors

    def validate_edges(self, graph: LogicalGraph) -> list[ValidationError]:
        """Validate node keys, inputs, outputs, operand arity, acyclicity, and nested subgraph edges.

        Args:
            graph (LogicalGraph): The graph to validate.

        Returns:
            list[ValidationError]: A list of edge and topology validation errors.
        """
        return self._validate_edges_scoped(graph, outer_scope=None, recurse_subgraphs=True)
