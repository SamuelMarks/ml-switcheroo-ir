"""Shared Data Models for ML Framework Snapshot introspection.

Provides the Ghost Protocol (GhostRef, GhostParam) schemas used to communicate
API structures between the ml-ecosystem-snapshots scraper and the ml-switcheroo compiler.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from enum import Enum

import pydantic
import typing_extensions

from ml_switcheroo_ir.types import JsonValue


class ParameterKind(str, Enum):
    """Standardized enumeration for parameter kinds.

    Attributes:
        POSITIONAL_ONLY (ParameterKind): Positional-only function parameter.
        POSITIONAL_OR_KEYWORD (ParameterKind): Standard positional-or-keyword parameter.
        VAR_POSITIONAL (ParameterKind): Variadic positional ``*args`` parameter.
        KEYWORD_ONLY (ParameterKind): Keyword-only function parameter.
        VAR_KEYWORD (ParameterKind): Variadic keyword ``**kwargs`` parameter.
    """

    POSITIONAL_ONLY = "POSITIONAL_ONLY"
    POSITIONAL_OR_KEYWORD = "POSITIONAL_OR_KEYWORD"
    VAR_POSITIONAL = "VAR_POSITIONAL"
    KEYWORD_ONLY = "KEYWORD_ONLY"
    VAR_KEYWORD = "VAR_KEYWORD"


class OperandDirection(str, Enum):
    """Structured operand directionality for assembly and low-level instructions.

    Attributes:
        READ (OperandDirection): Read-only source operand.
        WRITE (OperandDirection): Write-only destination operand.
        READ_WRITE (OperandDirection): Read-modify-write in-place operand.
        PREDICATE (OperandDirection): Predicate guard register operand.
    """

    READ = "READ"
    WRITE = "WRITE"
    READ_WRITE = "READ_WRITE"
    PREDICATE = "PREDICATE"


class IRParameterRole(str, Enum):
    """Distinguishes parameter roles for compiler intermediate representations.

    Attributes:
        OPERAND (IRParameterRole): Dynamic SSA value operand.
        ATTRIBUTE (IRParameterRole): Compile-time static attribute.
        RESULT (IRParameterRole): Operation output SSA result.
        SUCCESSOR (IRParameterRole): Control-flow block successor.
        REGION (IRParameterRole): Nested IR region or subgraph body.
    """

    OPERAND = "OPERAND"
    ATTRIBUTE = "ATTRIBUTE"
    RESULT = "RESULT"
    SUCCESSOR = "SUCCESSOR"
    REGION = "REGION"


class RegisterClass(str, Enum):
    """Register classes for GPU architectures (AMD RDNA/CDNA and NVIDIA).

    Attributes:
        VGPR (RegisterClass): Vector General-Purpose Register (AMD).
        SGPR (RegisterClass): Scalar General-Purpose Register (AMD).
        AGPR (RegisterClass): Accumulator General-Purpose Register (AMD CDNA).
        GPR (RegisterClass): General-Purpose Register (NVIDIA SASS/PTX).
        PRED (RegisterClass): Predicate register.
        ACCUM (RegisterClass): Tensor/matrix accumulator register.
    """

    VGPR = "VGPR"
    SGPR = "SGPR"
    AGPR = "AGPR"
    GPR = "GPR"
    PRED = "PRED"
    ACCUM = "ACCUM"


class MemorySpace(str, Enum):
    """Memory space qualifiers for low-level GPU and shader ISAs.

    Attributes:
        GLOBAL (MemorySpace): Device global DRAM address space.
        SHARED (MemorySpace): On-chip threadblock/workgroup shared SRAM.
        CONSTANT (MemorySpace): Read-only constant cache memory.
        LOCAL (MemorySpace): Per-thread private/spill local memory.
        UNIFORM (MemorySpace): Uniform constant buffer memory.
        STORAGE_READ (MemorySpace): Read-only shader storage buffer.
        STORAGE_READ_WRITE (MemorySpace): Read-write shader storage buffer.
    """

    GLOBAL = "global"
    SHARED = "shared"
    CONSTANT = "constant"
    LOCAL = "local"
    UNIFORM = "uniform"
    STORAGE_READ = "storage, read"
    STORAGE_READ_WRITE = "storage, read_write"


class WGSLQualifier(str, Enum):
    """Address space qualifiers for WebGPU WGSL shaders.

    Attributes:
        UNIFORM (WGSLQualifier): Uniform buffer address space (16-byte aligned).
        STORAGE_READ (WGSLQualifier): Read-only storage buffer address space.
        STORAGE_READ_WRITE (WGSLQualifier): Read-write storage buffer address space.
        WORKGROUP (WGSLQualifier): Workgroup shared address space.
        PRIVATE (WGSLQualifier): Module-scope per-invocation private space.
        FUNCTION (WGSLQualifier): Function-scope local variable space.
    """

    UNIFORM = "uniform"
    STORAGE_READ = "storage, read"
    STORAGE_READ_WRITE = "storage, read_write"
    WORKGROUP = "workgroup"
    PRIVATE = "private"
    FUNCTION = "function"


class SemanticTier(str, Enum):
    """Categorization of API operations to distinct knowledge base tiers.

    Attributes:
        ARRAY_API (SemanticTier): Python Array API foundational tensor operations.
        NEURAL (SemanticTier): High-level neural network modules.
        NEURAL_OPS (SemanticTier): Stateless functional neural operators.
        EXTRAS (SemanticTier): Framework utility extensions.
        LOSS (SemanticTier): Objective and loss functions.
        OPTIMIZER (SemanticTier): Gradient optimization algorithms.
        LAYER (SemanticTier): Stateful layer primitives.
        ACTIVATION (SemanticTier): Non-linear activation functions.
        METRIC (SemanticTier): Evaluation metric computations.
        UTIL (SemanticTier): General helper utilities.
        SCHEDULER (SemanticTier): Learning-rate schedule primitives.
        MODEL (SemanticTier): End-to-end model architectures.
        INITIALIZER (SemanticTier): Parameter weight initialization schemes.
        DATALOADER (SemanticTier): Data pipeline and batching iterators.
    """

    ARRAY_API = "array"
    NEURAL = "neural"
    NEURAL_OPS = "neural_ops"
    EXTRAS = "extras"
    LOSS = "loss"
    OPTIMIZER = "optimizer"
    LAYER = "layer"
    ACTIVATION = "activation"
    METRIC = "metric"
    UTIL = "util"
    SCHEDULER = "scheduler"
    MODEL = "model"
    INITIALIZER = "initializer"
    DATALOADER = "dataloader"


class IsaControlCodesSpec(pydantic.BaseModel):
    """Pydantic model for ISA instruction control codes and scheduling metadata.

    Attributes:
        latency (int | tuple[int, int] | None): Execution pipeline latency or cycle bounds.
        stall_count (int | None): Programmed issue stall cycle count (0-15).
        yield_flag (bool | None): Warp scheduler yield hint flag.
        write_barrier (int | None): Write dependency scoreboard barrier index.
        read_barrier (int | None): Read dependency scoreboard barrier index.
        wait_barrier_mask (int | str | None): Bitmask of barriers to await prior to issue.
    """

    model_config = pydantic.ConfigDict(extra="allow", populate_by_name=True)

    latency: int | tuple[int, int] | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices("latency", "execution_latency"),
        description="Execution pipeline latency or cycle bounds.",
    )
    stall_count: int | None = None
    yield_flag: bool | None = None
    write_barrier: int | None = None
    read_barrier: int | None = None
    wait_barrier_mask: int | str | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices("wait_barrier_mask", "scoreboard_barrier_mask"),
        description="Bitmask of dependency barriers to await prior to issue.",
    )

    def __init__(self, **data: JsonValue | tuple[int, int]) -> None:
        """Initialize IsaControlCodesSpec from canonical or alias keyword arguments.

        Args:
            **data (JsonValue | tuple[int, int]): Field or alias keyword values.
        """
        super().__init__(**data)


class IsaStructuredOperandSpec(pydantic.BaseModel):
    """Pydantic model for structured ISA operand specifications.

    Attributes:
        name (str | None): Formal operand name.
        role (str | None): Operand direction or semantic role.
        reg (str | None): Bound architectural register identifier.
        register_class (str | None): Hardware register bank/class constraint.
    """

    model_config = pydantic.ConfigDict(extra="allow")

    name: str | None = None
    role: str | None = None
    reg: str | None = None
    register_class: str | None = None


class VopdProfileSpec(pydantic.BaseModel):
    """Pydantic model for AMD RDNA3 VOPD dual-issue profile.

    Attributes:
        slot (str | None): Single legal VOPD slot ('X', 'Y', or 'BOTH').
        slots (list[str] | None): List of legal VOPD issue slots.
        max_src_operands (int | None): Maximum source VGPR operands allowed in dual-issue.
    """

    model_config = pydantic.ConfigDict(extra="allow")

    slot: str | None = None
    slots: list[str] | None = None
    max_src_operands: int | None = None


class RdnaPrimitiveSpec(pydantic.BaseModel):
    """Pydantic model for AMD RDNA instruction primitive specifications.

    Attributes:
        mnemonic (str): Hardware instruction mnemonic.
        operands (list[str]): Ordered operand slot names.
        register_classes (dict[str, RegisterClass]): Required register class per operand.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    mnemonic: str
    operands: list[str]
    register_classes: dict[str, RegisterClass]


class VopdOperatorSpec(pydantic.BaseModel):
    """Pydantic model for RDNA3 VOPD dual-issue operator specifications.

    Attributes:
        mnemonic (str): Dual-issue VOPD instruction mnemonic.
        slots (list[str]): Legal dual-issue slots ('X', 'Y').
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    mnemonic: str
    slots: list[str]


class SassPrimitiveSpec(pydantic.BaseModel):
    """Pydantic model for NVIDIA SASS instruction primitive specifications.

    Attributes:
        mnemonic (str): NVIDIA SASS opcode mnemonic.
        operands (list[str]): Ordered operand names.
        barrier_predicates (list[str] | None): Optional predicate/barrier qualifiers.
        memory_space (MemorySpace | None): Target memory address space if applicable.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    mnemonic: str
    operands: list[str]
    barrier_predicates: list[str] | None = None
    memory_space: MemorySpace | None = None


class WgslPrimitiveSpec(pydantic.BaseModel):
    """Pydantic model for WebGPU WGSL primitive signature specifications.

    Attributes:
        inputs (list[str]): Ordered input parameter names.
        outputs (list[str]): Ordered output value names.
        qualifier (WGSLQualifier | None): Required WGSL address space qualifier.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    inputs: list[str]
    outputs: list[str]
    qualifier: WGSLQualifier | None = None


class GhostParam(pydantic.BaseModel):
    """Serializable representation of a function parameter."""

    model_config = pydantic.ConfigDict(extra="ignore")

    name: str = pydantic.Field(description="Parameter name.")
    kind: ParameterKind = pydantic.Field(description="Kind of parameter (e.g. POSITIONAL_OR_KEYWORD).")
    default: JsonValue = pydantic.Field(default=None, description="Default value as string.")
    annotation: str | None = pydantic.Field(default=None, description="Type annotation as string.")
    description: str | None = pydantic.Field(default=None, description="Description")

    standardized_name: str | None = pydantic.Field(default=None, description="Standardized name")


class GhostResult(pydantic.BaseModel):
    """Structured SSA return or result for compiler IR operations."""

    model_config = pydantic.ConfigDict(extra="allow")

    name: str | None = pydantic.Field(default=None, description="Result SSA name or output identifier.")
    type: str | None = pydantic.Field(default=None, description="Result type (e.g. tensor<?x?xf32>).")
    description: str | None = pydantic.Field(default=None, description="Description of the result.")


class LogicOp(str, Enum):
    """Supported operators for conditional logic rules in operations.

    Attributes:
        EQ (LogicOp): Equality predicate ('eq').
        NEQ (LogicOp): Inequality predicate ('neq').
        GT (LogicOp): Strictly greater-than predicate ('gt').
        LT (LogicOp): Strictly less-than predicate ('lt').
        GTE (LogicOp): Greater-than-or-equal predicate ('gte').
        LTE (LogicOp): Less-than-or-equal predicate ('lte').
        IN (LogicOp): Set membership predicate ('in').
        NOT_IN (LogicOp): Negated set membership predicate ('not_in').
        IS_TYPE (LogicOp): Runtime type predicate ('is_type').
    """

    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    IS_TYPE = "is_type"


def _evaluate_numeric_logic_op(op: LogicOp, value: int | float, target: int | float) -> bool:
    """Evaluate a relational LogicOp comparison between two numeric values.

    Args:
        op (LogicOp): Relational operator (`GT`, `LT`, `GTE`, or `LTE`).
        value (int | float): Candidate numeric value.
        target (int | float): Threshold numeric target.

    Returns:
        bool: True if the comparison holds.
    """
    if op == LogicOp.GT:
        return value > target
    if op == LogicOp.LT:
        return value < target
    if op == LogicOp.GTE:
        return value >= target
    return value <= target


_LOGIC_TYPE_CHECKERS: dict[str, typing.Callable[[JsonValue], bool]] = {
    "bool": lambda v: isinstance(v, bool),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "str": lambda v: isinstance(v, str),
    "string": lambda v: isinstance(v, str),
    "list": lambda v: isinstance(v, list),
    "sequence": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
    "mapping": lambda v: isinstance(v, dict),
    "none": lambda v: v is None,
    "null": lambda v: v is None,
}


def _matches_logic_type_name(value: JsonValue, target_type: str) -> bool:
    """Check whether a runtime JsonValue matches a symbolic type descriptor name.

    Args:
        value (JsonValue): Runtime parameter value to inspect.
        target_type (str): Expected type name ('int', 'float', 'str', 'bool', 'list', 'dict', 'none').

    Returns:
        bool: True if ``value`` matches ``target_type``.
    """
    checker = _LOGIC_TYPE_CHECKERS.get(target_type.strip().lower())
    return checker(value) if checker is not None else False


class LogicConstraint(pydantic.BaseModel):
    """Conditional predicate constraint evaluated via a LogicOp on a GhostParam value.

    Attributes:
        op (LogicOp): Relational, membership, or runtime type predicate operator.
        target (JsonValue): Expected target scalar, collection, or type name.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    op: LogicOp = pydantic.Field(description="Predicate operator.")
    target: JsonValue = pydantic.Field(default=None, description="Expected target value, list, or type name.")

    def evaluate(self, value: JsonValue) -> bool:
        """Evaluate whether ``value`` satisfies this conditional logic constraint.

        Args:
            value (JsonValue): Candidate parameter value.

        Returns:
            bool: True when ``value`` satisfies ``(self.op, self.target)``.
        """
        same_bool = isinstance(value, bool) == isinstance(self.target, bool)
        if self.op in (LogicOp.EQ, LogicOp.NEQ):
            equal = bool(value == self.target) and same_bool
            return equal if self.op == LogicOp.EQ else not equal
        if self.op in (LogicOp.GT, LogicOp.LT, LogicOp.GTE, LogicOp.LTE):
            both_num = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isinstance(self.target, (int, float))
                and not isinstance(self.target, bool)
            )
            if both_num and isinstance(value, (int, float)) and isinstance(self.target, (int, float)):
                return _evaluate_numeric_logic_op(self.op, value, self.target)
            return False
        if self.op in (LogicOp.IN, LogicOp.NOT_IN):
            if not isinstance(self.target, list):
                return False
            return (value in self.target) if self.op == LogicOp.IN else (value not in self.target)
        return isinstance(self.target, str) and _matches_logic_type_name(value, self.target)


class ExtendedGhostParam(GhostParam):
    """Extended GhostParam supporting operand directionality, IR roles, dtypes, rank, and logic constraints."""

    model_config = pydantic.ConfigDict(extra="allow")

    default: JsonValue = pydantic.Field(
        default=None,
        description="Default value representation.",
    )
    direction: OperandDirection | None = pydantic.Field(
        default=None,
        description="Operand directionality (READ, WRITE, READ_WRITE, PREDICATE).",
    )
    role: IRParameterRole | None = pydantic.Field(
        default=None,
        description="IR parameter role (OPERAND, ATTRIBUTE, RESULT, etc.).",
    )
    allowed_dtypes: list[str] | None = pydantic.Field(
        default=None,
        description="Canonical allowed tensor dtypes (e.g. ['float32', 'bfloat16']).",
    )
    allowed_values: list[str] | None = pydantic.Field(
        default=None,
        description="Allowed enum or literal string values (e.g. ['none', 'mean', 'sum']).",
    )
    rank_constraint: str | None = pydantic.Field(
        default=None,
        description="Allowed tensor rank constraint (e.g. '==2', '>=2', 'scalar').",
    )
    is_contracting_dim: bool | None = pydantic.Field(
        default=None,
        description="Whether this parameter represents a contracting tensor dimension.",
    )
    default_factory: str | None = pydantic.Field(
        default=None,
        description="Name or representation of factory function producing default value.",
    )
    is_mandatory: bool | None = pydantic.Field(
        default=None,
        description="Whether parameter is mandatory (no default value).",
    )
    constraints: list[LogicConstraint] = pydantic.Field(
        default_factory=list,
        description="Explicit conditional logic constraints evaluated against parameter values.",
    )

    def check_value_constraints(self, value: JsonValue) -> bool:
        """Validate a runtime parameter value against allowed_values and LogicOp constraints.

        Args:
            value (JsonValue): Candidate parameter value to validate.

        Returns:
            bool: True if ``value`` satisfies ``is_mandatory``, ``allowed_values``, and all ``constraints``.
        """
        if value is None:
            if self.is_mandatory:
                return False
            return all(c.evaluate(value) for c in self.constraints)
        if self.allowed_values is not None and str(value) not in self.allowed_values:
            return False
        return all(c.evaluate(value) for c in self.constraints)


class GhostRef(pydantic.BaseModel):
    """Serializable snapshot of a Framework API component."""

    model_config = pydantic.ConfigDict(extra="ignore")

    name: str = pydantic.Field(description="Short name of the object.")
    api_path: str = pydantic.Field(description="Fully qualified import path.")
    kind: str = pydantic.Field(description="One of: 'class', 'function'")
    params: list[GhostParam] = pydantic.Field(default_factory=list, description="List of parameters.")
    docstring: str | None = pydantic.Field(default=None, description="Extracted docstring.")
    has_varargs: bool = pydantic.Field(default=False, description="True if signature accepts *args.")
    schema_version: str = pydantic.Field(default="2.0.0", description="Version of the schema format.")

    is_public: bool | None = pydantic.Field(default=None, description="Is public")
    is_c_extension: bool | None = pydantic.Field(
        default=False,
        description="Whether the symbol originates from a compiled C/C++ extension.",
    )
    accepted_kwargs: list[str] | None = pydantic.Field(
        default=None,
        description="Explicit list of accepted keyword arguments when **kwargs is present.",
    )

    aliases: list[str] | None = pydantic.Field(default_factory=list, description="Aliases")
    returns_type: str | None = pydantic.Field(default=None, description="Returns type")
    returns_description: str | None = pydantic.Field(default=None, description="Returns description")
    raises: list[str] | None = pydantic.Field(default_factory=list, description="Raises")
    environment_tags: list[str] | None = pydantic.Field(default_factory=list, description="Environment tags")
    overloads: list[dict[str, JsonValue] | GhostRef] | None = pydantic.Field(
        default_factory=list, description="Overloads"
    )

    def __init__(self, **data: GhostModelInitValue) -> None:
        """Initialize GhostRef from field or alias keyword arguments.

        Args:
            **data (GhostModelInitValue): Field keyword values.
        """
        super().__init__(**data)

    def has_arg(self, arg_name: str) -> bool:
        """Check if a specific argument exists in the signature.

        Args:
            arg_name (str): The argument name to find.

        Returns:
            bool: True if found.
        """
        return any(p.name == arg_name for p in self.params)


class ExtendedGhostRef(GhostRef):
    """Extended GhostRef with support for domain metadata, multiple SSA returns, and IR operands."""

    model_config = pydantic.ConfigDict(extra="allow")

    params: list[ExtendedGhostParam | GhostParam] = pydantic.Field(
        default_factory=list,
        description="List of extended parameter specifications.",
    )
    returns: list[GhostResult] | None = pydantic.Field(default=None, description="Multiple SSA returns or results.")
    domain_metadata: dict[str, JsonValue] | None = pydantic.Field(
        default=None,
        description="Structured domain metadata for ISAs and compilers.",
    )
    signature_completeness: typing.Literal["exact", "heuristic", "opaque"] | None = pydantic.Field(
        default="exact",
        description="Completeness of signature resolution: exact, heuristic, or opaque.",
    )
    is_c_extension: bool | None = pydantic.Field(
        default=False,
        description="Whether the symbol originates from a compiled C/C++ extension.",
    )
    accepted_kwargs: list[str] | None = pydantic.Field(
        default=None,
        description="Explicit list of accepted keyword arguments when **kwargs is present.",
    )

    def __init__(self, **data: GhostModelInitValue) -> None:
        """Initialize ExtendedGhostRef from field or alias keyword arguments.

        Args:
            **data (GhostModelInitValue): Field keyword values.
        """
        super().__init__(**data)


class GhostPythonRef(ExtendedGhostRef):
    """GhostRef specialized for high-level Python ML frameworks (PyTorch, JAX, TF, Keras).

    Attributes:
        domain_type (Literal['python']): Discriminator tag for Python framework references.
    """

    model_config = pydantic.ConfigDict(extra="allow")
    domain_type: typing.Literal["python"] = "python"


GhostIsaPayloadValue: typing_extensions.TypeAlias = typing.Union[
    JsonValue,
    tuple[int, int],
    VopdProfileSpec,
    IsaControlCodesSpec,
    typing.Sequence[str],
    typing.Sequence[typing.Sequence[str]],
    list[IsaStructuredOperandSpec],
    list[ExtendedGhostParam],
    list[GhostParam],
    list[typing.Union[ExtendedGhostParam, GhostParam]],
    list[GhostResult],
    list[GhostRef],
    list[typing.Union[dict[str, JsonValue], GhostRef]],
]
GhostModelInitValue: typing_extensions.TypeAlias = GhostIsaPayloadValue


def _merge_vopd_slot_alias(data: dict[str, GhostIsaPayloadValue]) -> None:
    """Fold ``vopd_slot`` or string ``vopd_profile`` into canonical ``VopdProfileSpec`` dict.

    Args:
        data (dict[str, GhostIsaPayloadValue]): Mutable GhostIsaRef payload dictionary.
    """
    raw_slot = data.pop("vopd_slot", None)
    raw_prof = data.get("vopd_profile")
    if isinstance(raw_prof, str):
        data["vopd_profile"] = {"slot": raw_prof}
    elif isinstance(raw_slot, str):
        if raw_prof is None:
            data["vopd_profile"] = {"slot": raw_slot}
        elif isinstance(raw_prof, VopdProfileSpec) and raw_prof.slot is None:
            prof_dump = raw_prof.model_dump()
            prof_dump["slot"] = raw_slot
            data["vopd_profile"] = prof_dump
        elif isinstance(raw_prof, dict) and raw_prof.get("slot") is None:
            merged_prof = dict(raw_prof)
            merged_prof["slot"] = raw_slot
            data["vopd_profile"] = merged_prof


def _normalize_ghost_isa_payload(raw: Mapping[str, GhostIsaPayloadValue]) -> dict[str, GhostIsaPayloadValue]:
    """Normalize aliased modifier, VOPD, latency, and barrier fields into canonical GhostIsaRef fields.

    Args:
        raw (Mapping[str, GhostIsaPayloadValue]): Raw input mapping passed to GhostIsaRef validation.

    Returns:
        dict[str, GhostIsaPayloadValue]: Normalized dictionary with canonical fields populated.
    """
    data = dict(raw)
    allowed_mods = data.pop("allowed_modifiers", None)
    if data.get("instruction_modifiers") is None and allowed_mods is not None:
        data["instruction_modifiers"] = allowed_mods

    _merge_vopd_slot_alias(data)

    exec_lat = data.pop("execution_latency", None)
    sb_mask = data.pop("scoreboard_barrier_mask", None)
    if exec_lat is not None or sb_mask is not None:
        raw_cc = data.get("control_codes")
        cc_dict: dict[str, JsonValue | tuple[int, int]] = (
            raw_cc.model_dump()
            if isinstance(raw_cc, IsaControlCodesSpec)
            else (dict(raw_cc) if isinstance(raw_cc, dict) else {})
        )
        if cc_dict.get("latency") is None:
            if isinstance(exec_lat, int) and not isinstance(exec_lat, bool):
                cc_dict["latency"] = exec_lat
            elif (
                isinstance(exec_lat, (list, tuple))
                and len(exec_lat) == 2
                and isinstance(exec_lat[0], int)
                and isinstance(exec_lat[1], int)
            ):
                cc_dict["latency"] = (exec_lat[0], exec_lat[1])
        if isinstance(sb_mask, (int, str)) and cc_dict.get("wait_barrier_mask") is None:
            cc_dict["wait_barrier_mask"] = sb_mask
        data["control_codes"] = IsaControlCodesSpec.model_validate(cc_dict)

    return data


class GhostIsaRef(ExtendedGhostRef):
    """GhostRef specialized for GPU assembly ISAs (NVIDIA SASS, AMD RDNA/CDNA).

    Attributes:
        domain_type (Literal['isa', 'instruction']): Discriminator tag for hardware ISA references.
        predicate_guards (list[str] | None): Allowed predicate guard registers.
        register_classes (dict[str, str] | None): Register classes for instruction operands.
        control_codes (IsaControlCodesSpec | None): Instruction control code and scheduling schema.
        instruction_modifiers (list[str] | None): Valid instruction modifiers.
        structured_modifiers (dict[str, JsonValue] | None): Structured modifier bitfields.
        structured_operands (list[IsaStructuredOperandSpec] | None): Detailed operand records.
        vopd_profile (VopdProfileSpec | None): VOPD dual-issue profile and pairing rules.
        condition_codes (list[str] | None): Allowed condition codes or flags.
        supported_architectures (list[str] | None): Microarchitectures supporting this instruction.
        operand_signatures (list[list[str]] | None): Explicit register or immediate operand signatures.
        issue_rate (float | None): Throughput instructions issued per clock cycle.
    """

    model_config = pydantic.ConfigDict(extra="allow", populate_by_name=True)
    domain_type: typing.Literal["isa", "instruction"] = "isa"
    predicate_guards: list[str] | None = pydantic.Field(
        default=None,
        description="Allowed predicate guard registers (e.g. ['@P0', '@!P1', '@PT']).",
    )
    register_classes: dict[str, str] | None = pydantic.Field(
        default=None,
        description="Register classes for operands (e.g. {'op0': 'VGPR_32', 'op1': 'VReg_64'}).",
    )
    control_codes: IsaControlCodesSpec | None = pydantic.Field(
        default=None,
        description="Instruction control code and scheduling schema.",
    )
    instruction_modifiers: list[str] | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices("instruction_modifiers", "allowed_modifiers"),
        description="Valid instruction modifiers (e.g. ['.SAT', '.FTZ', 'omod:2']).",
    )
    structured_modifiers: dict[str, JsonValue] | None = pydantic.Field(
        default=None,
        description="Structured modifier bitfields (e.g. rounding, cache, saturation).",
    )
    structured_operands: list[IsaStructuredOperandSpec] | None = pydantic.Field(
        default=None,
        description="Detailed operand records with roles, register classes, and immediate constraints.",
    )
    vopd_profile: VopdProfileSpec | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices("vopd_profile", "vopd_slot"),
        description="VOPD dual-issue profile and pairing rules for RDNA3/GFX11.",
    )
    condition_codes: list[str] | None = pydantic.Field(
        default=None,
        description="Allowed condition codes or flags (e.g. ['CC.EQ', 'CC.LT', 'vcc']).",
    )
    supported_architectures: list[str] | None = pydantic.Field(
        default=None,
        description="Microarchitectures supporting this instruction.",
    )
    operand_signatures: list[list[str]] | None = pydantic.Field(
        default=None,
        description="Explicit register or immediate operand signatures (e.g. [['vgpr', 'vgpr', 'sgpr']]).",
    )
    issue_rate: float | None = pydantic.Field(
        default=None,
        description="Throughput instructions issued per clock cycle.",
    )

    def __init__(self, **data: GhostIsaPayloadValue) -> None:
        """Initialize GhostIsaRef from canonical fields or legacy ISA aliases.

        Args:
            **data (GhostIsaPayloadValue): Field or alias keyword values.
        """
        super().__init__(**data)

    @pydantic.model_validator(mode="before")
    @classmethod
    def _unify_isa_aliases(
        cls,
        values: Mapping[str, GhostIsaPayloadValue] | GhostIsaRef,
    ) -> dict[str, GhostIsaPayloadValue] | GhostIsaRef:
        """Fold legacy modifier, VOPD slot, latency, and scoreboard aliases into canonical fields.

        Args:
            values (Mapping[str, GhostIsaPayloadValue] | GhostIsaRef): Raw input dictionary or model instance.

        Returns:
            dict[str, GhostIsaPayloadValue] | GhostIsaRef: Normalized validation payload.
        """
        if isinstance(values, Mapping):
            return _normalize_ghost_isa_payload(values)
        return values

    @property
    def allowed_modifiers(self) -> list[str] | None:
        """Return legal instruction modifier flags (alias of ``instruction_modifiers``).

        Returns:
            list[str] | None: Allowed instruction modifier flags if defined.
        """
        return self.instruction_modifiers

    @property
    def vopd_slot(self) -> str | None:
        """Return VOPD dual-issue slot designation derived from ``vopd_profile``.

        Returns:
            str | None: Dual-issue slot ('X', 'Y', or 'BOTH') if configured.
        """
        return self.vopd_profile.slot if self.vopd_profile is not None else None

    @property
    def execution_latency(self) -> int | tuple[int, int] | None:
        """Return analytical execution cycle latency derived from ``control_codes``.

        Returns:
            int | tuple[int, int] | None: Execution latency value or bounds if configured.
        """
        return self.control_codes.latency if self.control_codes is not None else None

    @property
    def scoreboard_barrier_mask(self) -> int | str | None:
        """Return hardware dependency barrier mask derived from ``control_codes``.

        Returns:
            int | str | None: Dependency barrier mask if configured.
        """
        return self.control_codes.wait_barrier_mask if self.control_codes is not None else None


class GhostMlirRef(ExtendedGhostRef):
    """GhostRef specialized for compiler IR dialects (Core MLIR and StableHLO).

    Attributes:
        domain_type (Literal['mlir', 'operation']): Discriminator tag for MLIR dialect operations.
    """

    model_config = pydantic.ConfigDict(extra="allow")
    domain_type: typing.Literal["mlir", "operation"] = "mlir"
    traits: list[str] | None = pydantic.Field(
        default=None,
        description="Dialect verification traits (e.g. ['SameOperandsAndResultType', 'Commutative']).",
    )
    operands: list[ExtendedGhostParam | GhostParam] | None = pydantic.Field(
        default=None,
        description="Strictly decoupled SSA value arguments (operands).",
    )
    attributes: dict[str, JsonValue] | None = pydantic.Field(
        default=None,
        description="Structured attribute specifications and schemas.",
    )
    regions: dict[str, JsonValue] | None = pydantic.Field(
        default=None,
        description="Region definitions with block arguments and yield types.",
    )
    successors: list[str] | None = pydantic.Field(
        default=None,
        description="Successor block identifiers for control flow operations.",
    )
    type_constraints: dict[str, str] | None = pydantic.Field(
        default=None,
        description="Type constraints for operands and results (e.g. RankedTensorOf, AnyFloat).",
    )


# First-class domain IR and ISA schema aliases
GhostInstructionRef = GhostIsaRef
GhostOperationRef = GhostMlirRef

RDNA_INSTRUCTION_PRIMITIVES: dict[str, RdnaPrimitiveSpec] = {
    "V_FMA_F32": RdnaPrimitiveSpec(
        mnemonic="V_FMA_F32",
        operands=["dst", "src0", "src1", "src2"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
            "src2": RegisterClass.VGPR,
        },
    ),
    "V_ADD_F32": RdnaPrimitiveSpec(
        mnemonic="V_ADD_F32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_MUL_F32": RdnaPrimitiveSpec(
        mnemonic="V_MUL_F32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_DOT2_F32_F16": RdnaPrimitiveSpec(
        mnemonic="V_DOT2_F32_F16",
        operands=["dst", "src0", "src1", "src2"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
            "src2": RegisterClass.VGPR,
        },
    ),
    "V_DOT4_I32_I8": RdnaPrimitiveSpec(
        mnemonic="V_DOT4_I32_I8",
        operands=["dst", "src0", "src1", "src2"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
            "src2": RegisterClass.VGPR,
        },
    ),
    "V_FMAC_F32": RdnaPrimitiveSpec(
        mnemonic="V_FMAC_F32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_MOV_B32": RdnaPrimitiveSpec(
        mnemonic="V_MOV_B32",
        operands=["dst", "src0"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
        },
    ),
    "V_CNDMASK_B32": RdnaPrimitiveSpec(
        mnemonic="V_CNDMASK_B32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_SUB_F32": RdnaPrimitiveSpec(
        mnemonic="V_SUB_F32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_DUAL_FMAC_F32": RdnaPrimitiveSpec(
        mnemonic="V_DUAL_FMAC_F32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_DUAL_FMAAK_F32": RdnaPrimitiveSpec(
        mnemonic="V_DUAL_FMAAK_F32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_DUAL_FMAMK_F32": RdnaPrimitiveSpec(
        mnemonic="V_DUAL_FMAMK_F32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_DUAL_MUL_F32": RdnaPrimitiveSpec(
        mnemonic="V_DUAL_MUL_F32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_DUAL_ADD_F32": RdnaPrimitiveSpec(
        mnemonic="V_DUAL_ADD_F32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_DUAL_SUB_F32": RdnaPrimitiveSpec(
        mnemonic="V_DUAL_SUB_F32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_DUAL_SUBREV_F32": RdnaPrimitiveSpec(
        mnemonic="V_DUAL_SUBREV_F32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_DUAL_MOV_B32": RdnaPrimitiveSpec(
        mnemonic="V_DUAL_MOV_B32",
        operands=["dst", "src0"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
        },
    ),
    "V_DUAL_CNDMASK_B32": RdnaPrimitiveSpec(
        mnemonic="V_DUAL_CNDMASK_B32",
        operands=["dst", "src0", "src1"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    ),
    "V_DUAL_DOT2ACC_F32_F16": RdnaPrimitiveSpec(
        mnemonic="V_DUAL_DOT2ACC_F32_F16",
        operands=["dst", "src0", "src1", "src2"],
        register_classes={
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
            "src2": RegisterClass.VGPR,
        },
    ),
}

# RDNA3 / GFX11 VOPD Dual-Issue Instruction Opcode Matrix
RDNA3_VOPD_OPERATORS: dict[str, VopdOperatorSpec] = {
    "V_DUAL_FMAC_F32": VopdOperatorSpec(mnemonic="V_DUAL_FMAC_F32", slots=["X", "Y"]),
    "V_DUAL_FMAAK_F32": VopdOperatorSpec(mnemonic="V_DUAL_FMAAK_F32", slots=["X", "Y"]),
    "V_DUAL_FMAMK_F32": VopdOperatorSpec(mnemonic="V_DUAL_FMAMK_F32", slots=["X", "Y"]),
    "V_DUAL_MUL_F32": VopdOperatorSpec(mnemonic="V_DUAL_MUL_F32", slots=["X", "Y"]),
    "V_DUAL_ADD_F32": VopdOperatorSpec(mnemonic="V_DUAL_ADD_F32", slots=["X", "Y"]),
    "V_DUAL_SUB_F32": VopdOperatorSpec(mnemonic="V_DUAL_SUB_F32", slots=["X", "Y"]),
    "V_DUAL_SUBREV_F32": VopdOperatorSpec(mnemonic="V_DUAL_SUBREV_F32", slots=["X", "Y"]),
    "V_DUAL_MOV_B32": VopdOperatorSpec(mnemonic="V_DUAL_MOV_B32", slots=["X", "Y"]),
    "V_DUAL_CNDMASK_B32": VopdOperatorSpec(mnemonic="V_DUAL_CNDMASK_B32", slots=["X"]),
    "V_DUAL_DOT2ACC_F32_F16": VopdOperatorSpec(mnemonic="V_DUAL_DOT2ACC_F32_F16", slots=["Y"]),
}

SASS_INSTRUCTION_PRIMITIVES: dict[str, SassPrimitiveSpec] = {
    "FFMA": SassPrimitiveSpec(
        mnemonic="FFMA",
        operands=["dst", "src0", "src1", "src2"],
        barrier_predicates=["@P0", "@!P0", "@PT"],
    ),
    "FADD": SassPrimitiveSpec(
        mnemonic="FADD",
        operands=["dst", "src0", "src1"],
        barrier_predicates=["@P0", "@!P0", "@PT"],
    ),
    "FMUL": SassPrimitiveSpec(
        mnemonic="FMUL",
        operands=["dst", "src0", "src1"],
        barrier_predicates=["@P0", "@!P0", "@PT"],
    ),
    "HMMA": SassPrimitiveSpec(
        mnemonic="HMMA",
        operands=["dst", "src0", "src1", "src2"],
        barrier_predicates=["@P0", "@!P0", "@PT"],
    ),
    "LDG": SassPrimitiveSpec(
        mnemonic="LDG",
        operands=["dst", "src0"],
        memory_space=MemorySpace.GLOBAL,
    ),
    "STS": SassPrimitiveSpec(
        mnemonic="STS",
        operands=["dst", "src0"],
        memory_space=MemorySpace.SHARED,
    ),
}

WGSL_PRIMITIVE_SIGNATURES: dict[str, WgslPrimitiveSpec] = {
    "workgroupBarrier": WgslPrimitiveSpec(
        inputs=[],
        outputs=[],
        qualifier=WGSLQualifier.WORKGROUP,
    ),
    "storageBarrier": WgslPrimitiveSpec(
        inputs=[],
        outputs=[],
        qualifier=WGSLQualifier.STORAGE_READ_WRITE,
    ),
    "fma": WgslPrimitiveSpec(
        inputs=["a", "b", "c"],
        outputs=["result"],
    ),
    "dot": WgslPrimitiveSpec(
        inputs=["a", "b"],
        outputs=["result"],
    ),
    "textureSample": WgslPrimitiveSpec(
        inputs=["texture", "sampler", "coords"],
        outputs=["result"],
    ),
    "storageStore": WgslPrimitiveSpec(
        inputs=["buffer", "index", "value"],
        outputs=[],
        qualifier=WGSLQualifier.STORAGE_READ_WRITE,
    ),
    "atomicAdd": WgslPrimitiveSpec(
        inputs=["atomic_ptr", "value"],
        outputs=["old_value"],
        qualifier=WGSLQualifier.STORAGE_READ_WRITE,
    ),
}

WGSL_COMPUTE_BUILTINS: set[str] = {
    "workgroup_id",
    "local_invocation_id",
    "global_invocation_id",
    "local_invocation_index",
    "num_workgroups",
}

WGSL_MUTATING_OPS: set[str] = {
    "storageStore",
    "atomicAdd",
    "atomicSub",
    "atomicMax",
    "atomicMin",
    "atomicAnd",
    "atomicOr",
    "atomicXor",
    "atomicExchange",
    "atomicCompareExchangeWeak",
    "textureStore",
}


class SnapshotEnvelope(pydantic.BaseModel):
    """Structured provenance envelope for framework and ISA/IR snapshots."""

    model_config = pydantic.ConfigDict(extra="allow")

    schema_version: str = pydantic.Field(default="2.0.0", description="Snapshot schema version.")
    target: str = pydantic.Field(..., description="Target framework, dialect, or hardware ISA.")
    version: str | None = pydantic.Field(
        default=None,
        description="Upstream framework version or toolkit release.",
    )
    upstream_version: str | None = pydantic.Field(
        default=None,
        description="Upstream hardware specification or compiler version.",
    )
    source_type: str | None = pydantic.Field(
        default=None,
        description="Extraction source (tablegen, binary_disassembly, python_ast).",
    )
    upstream_commit: str | None = pydantic.Field(default=None, description="Upstream git commit hash or release tag.")
    supported_microarchitectures: list[str] | None = pydantic.Field(
        default=None,
        description="Explicit list of supported GPU compute capabilities or target architectures.",
    )
    generated_at: str | None = pydantic.Field(default=None, description="ISO-8601 generation timestamp.")
    environment: dict[str, JsonValue] | None = pydantic.Field(
        default=None, description="Build host environment metadata."
    )
    categories: dict[str, list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef]] = pydantic.Field(
        default_factory=dict, description="Categorized symbol dictionaries."
    )
    operations: list[dict[str, JsonValue] | GhostMlirRef] | None = pydantic.Field(
        default=None, description="Flat list of IR or dialect operations."
    )
    instructions: list[dict[str, JsonValue] | GhostIsaRef] | None = pydantic.Field(
        default=None, description="Flat list of hardware ISA instructions."
    )


class StandardMap(pydantic.BaseModel):
    """Defines how a Framework implements a Middle Layer standard.

    Attributes:
        api (str | None): Target framework dotted API callable path.
        args (dict[str, str | float | int | None] | None): Parameter rename or constant mapping.
        inject_args (dict[str, JsonValue] | None): Extra keyword arguments injected at call site.
        requires_plugin (str | None): Optional plugin dependency identifier.
        transformation_type (str | None): AST rewrite strategy ('call', 'infix', 'macro').
        operator (str | None): Infix or unary operator token when transformation_type is 'infix'.
        pack_to_tuple (str | None): Destination tuple parameter name for packed scalar roles.
        macro_template (str | None): Source template pattern for macro expansions.
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    api: str | None = pydantic.Field(default=None)
    args: dict[str, str | float | int | None] | None = pydantic.Field(default=None)
    inject_args: dict[str, JsonValue] | None = pydantic.Field(default=None)
    requires_plugin: str | None = pydantic.Field(default=None)
    transformation_type: str | None = pydantic.Field(default=None)
    operator: str | None = pydantic.Field(default=None)
    pack_to_tuple: str | None = pydantic.Field(default=None)
    macro_template: str | None = pydantic.Field(default=None)
