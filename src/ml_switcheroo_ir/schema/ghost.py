"""Shared Data Models for ML Framework Snapshot introspection.

Provides the Ghost Protocol (GhostRef, GhostParam) schemas used to communicate
API structures between the ml-framework-snapshots scraper and the ml-switcheroo compiler.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ParameterKind(str, Enum):
    """Standardized enumeration for parameter kinds."""

    POSITIONAL_ONLY = "POSITIONAL_ONLY"
    POSITIONAL_OR_KEYWORD = "POSITIONAL_OR_KEYWORD"
    VAR_POSITIONAL = "VAR_POSITIONAL"
    KEYWORD_ONLY = "KEYWORD_ONLY"
    VAR_KEYWORD = "VAR_KEYWORD"


class OperandDirection(str, Enum):
    """Structured operand directionality for assembly and low-level instructions."""

    READ = "READ"
    WRITE = "WRITE"
    READ_WRITE = "READ_WRITE"
    PREDICATE = "PREDICATE"


class IRParameterRole(str, Enum):
    """Distinguishes parameter roles for compiler intermediate representations."""

    OPERAND = "OPERAND"
    ATTRIBUTE = "ATTRIBUTE"
    RESULT = "RESULT"
    SUCCESSOR = "SUCCESSOR"
    REGION = "REGION"


class RegisterClass(str, Enum):
    """Register classes for GPU architectures (AMD RDNA/CDNA and NVIDIA)."""

    VGPR = "VGPR"
    SGPR = "SGPR"
    AGPR = "AGPR"
    GPR = "GPR"
    PRED = "PRED"
    ACCUM = "ACCUM"


class MemorySpace(str, Enum):
    """Memory space qualifiers for low-level GPU and shader ISAs."""

    GLOBAL = "global"
    SHARED = "shared"
    CONSTANT = "constant"
    LOCAL = "local"
    UNIFORM = "uniform"
    STORAGE_READ = "storage, read"
    STORAGE_READ_WRITE = "storage, read_write"


class WGSLQualifier(str, Enum):
    """Address space qualifiers for WebGPU WGSL shaders."""

    UNIFORM = "uniform"
    STORAGE_READ = "storage, read"
    STORAGE_READ_WRITE = "storage, read_write"
    WORKGROUP = "workgroup"
    PRIVATE = "private"
    FUNCTION = "function"


class SemanticTier(str, Enum):
    """Categorization of API operations to distinct knowledge base tiers."""

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


class GhostParam(BaseModel):
    """Serializable representation of a function parameter."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Parameter name.")
    kind: ParameterKind = Field(
        description="Kind of parameter (e.g. POSITIONAL_OR_KEYWORD)."
    )
    default: str | None = Field(default=None, description="Default value as string.")
    annotation: str | None = Field(
        default=None, description="Type annotation as string."
    )
    description: str | None = Field(default=None, description="Description")

    standardized_name: str | None = Field(default=None, description="Standardized name")


class GhostResult(BaseModel):
    """Structured SSA return or result for compiler IR operations."""

    model_config = ConfigDict(extra="allow")

    name: str | None = Field(
        default=None, description="Result SSA name or output identifier."
    )
    type: str | None = Field(
        default=None, description="Result type (e.g. tensor<?x?xf32>)."
    )
    description: str | None = Field(
        default=None, description="Description of the result."
    )


class ExtendedGhostParam(GhostParam):
    """Extended GhostParam supporting operand directionality, IR roles, dtypes, rank, and factory defaults."""

    model_config = ConfigDict(extra="allow")

    default: Any | None = Field(
        default=None,
        description="Default value representation.",
    )
    direction: OperandDirection | None = Field(
        default=None,
        description="Operand directionality (READ, WRITE, READ_WRITE, PREDICATE).",
    )
    role: IRParameterRole | None = Field(
        default=None,
        description="IR parameter role (OPERAND, ATTRIBUTE, RESULT, etc.).",
    )
    dtypes: list[str] | None = Field(
        default=None,
        description="Allowed tensor dtypes (e.g. ['float32', 'bfloat16', 'float16']).",
    )
    allowed_dtypes: list[str] | None = Field(
        default=None,
        description="Canonical allowed tensor dtypes (e.g. ['float32', 'bfloat16']).",
    )
    allowed_values: list[str] | None = Field(
        default=None,
        description="Allowed enum or literal string values (e.g. ['none', 'mean', 'sum']).",
    )
    rank: int | str | None = Field(
        default=None,
        description="Allowed tensor rank (e.g. 0 for scalar, 1, 2, 'N-D').",
    )
    rank_constraint: str | None = Field(
        default=None,
        description="Allowed tensor rank constraint (e.g. '==2', '>=2', 'scalar').",
    )
    is_contracting_dim: bool | None = Field(
        default=None,
        description="Whether this parameter represents a contracting tensor dimension.",
    )
    default_factory: str | None = Field(
        default=None,
        description="Name or representation of factory function producing default value.",
    )
    is_mandatory: bool | None = Field(
        default=None,
        description="Whether parameter is mandatory (no default value).",
    )


class GhostRef(BaseModel):
    """Serializable snapshot of a Framework API component."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Short name of the object.")
    api_path: str = Field(description="Fully qualified import path.")
    kind: str = Field(description="One of: 'class', 'function'")
    params: list[GhostParam] = Field(
        default_factory=list, description="List of parameters."
    )
    docstring: str | None = Field(default=None, description="Extracted docstring.")
    has_varargs: bool = Field(
        default=False, description="True if signature accepts *args."
    )
    schema_version: str = Field(
        default="1.2", description="Version of the schema format."
    )

    is_public: bool | None = Field(default=None, description="Is public")

    aliases: list[str] | None = Field(default_factory=list, description="Aliases")
    returns_type: str | None = Field(default=None, description="Returns type")
    returns_description: str | None = Field(
        default=None, description="Returns description"
    )
    raises: list[str] | None = Field(default_factory=list, description="Raises")
    environment_tags: list[str] | None = Field(
        default_factory=list, description="Environment tags"
    )
    overloads: list[Any] | None = Field(default_factory=list, description="Overloads")

    def has_arg(self, arg_name: str) -> bool:
        """Check if a specific argument exists in the signature.

        Args:
            arg_name: The argument name to find.

        Returns:
            True if found.
        """
        return any(p.name == arg_name for p in self.params)


class ExtendedGhostRef(GhostRef):
    """Extended GhostRef with support for domain metadata, multiple SSA returns, and IR operands."""

    model_config = ConfigDict(extra="allow")

    params: list[ExtendedGhostParam | GhostParam] = Field(
        default_factory=list,
        description="List of extended parameter specifications.",
    )
    returns: list[GhostResult] | None = Field(
        default=None, description="Multiple SSA returns or results."
    )
    domain_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Structured domain metadata for ISAs and compilers.",
    )
    signature_completeness: Literal["exact", "heuristic", "opaque"] | None = Field(
        default="exact",
        description="Completeness of signature resolution: exact, heuristic, or opaque.",
    )


class GhostPythonRef(ExtendedGhostRef):
    """GhostRef specialized for high-level Python ML frameworks (PyTorch, JAX, TF, Keras)."""

    model_config = ConfigDict(extra="allow")
    domain_type: Literal["python"] = "python"


class GhostIsaRef(ExtendedGhostRef):
    """GhostRef specialized for GPU assembly ISAs (NVIDIA SASS, AMD RDNA/CDNA)."""

    model_config = ConfigDict(extra="allow")
    domain_type: Literal["isa", "instruction"] = "isa"
    predicate_guards: list[str] | None = Field(
        default=None,
        description="Allowed predicate guard registers (e.g. ['@P0', '@!P1', '@PT']).",
    )
    register_classes: dict[str, str] | None = Field(
        default=None,
        description="Register classes for operands (e.g. {'op0': 'VGPR_32', 'op1': 'VReg_64'}).",
    )
    control_codes: dict[str, Any] | None = Field(
        default=None,
        description="Instruction control code and scheduling schema.",
    )
    instruction_modifiers: list[str] | None = Field(
        default=None,
        description="Valid instruction modifiers (e.g. ['.SAT', '.FTZ', 'omod:2']).",
    )
    structured_modifiers: dict[str, Any] | None = Field(
        default=None,
        description="Structured modifier bitfields (e.g. rounding, cache, saturation).",
    )
    structured_operands: list[dict[str, Any]] | None = Field(
        default=None,
        description="Detailed operand records with roles, register classes, and immediate constraints.",
    )
    vopd_profile: dict[str, Any] | None = Field(
        default=None,
        description="VOPD dual-issue profile and pairing rules for RDNA3/GFX11.",
    )
    condition_codes: list[str] | None = Field(
        default=None,
        description="Allowed condition codes or flags (e.g. ['CC.EQ', 'CC.LT', 'vcc']).",
    )
    supported_architectures: list[str] | None = Field(
        default=None,
        description="Microarchitectures supporting this instruction.",
    )


class GhostMlirRef(ExtendedGhostRef):
    """GhostRef specialized for compiler IR dialects (Core MLIR and StableHLO)."""

    model_config = ConfigDict(extra="allow")
    domain_type: Literal["mlir", "operation"] = "mlir"
    traits: list[str] | None = Field(
        default=None,
        description="Dialect verification traits (e.g. ['SameOperandsAndResultType', 'Commutative']).",
    )
    operands: list[ExtendedGhostParam | GhostParam] | None = Field(
        default=None,
        description="Strictly decoupled SSA value arguments (operands).",
    )
    attributes: dict[str, Any] | None = Field(
        default=None,
        description="Structured attribute specifications and schemas.",
    )
    regions: dict[str, Any] | None = Field(
        default=None,
        description="Region definitions with block arguments and yield types.",
    )
    successors: list[str] | None = Field(
        default=None,
        description="Successor block identifiers for control flow operations.",
    )
    type_constraints: dict[str, str] | None = Field(
        default=None,
        description="Type constraints for operands and results (e.g. RankedTensorOf, AnyFloat).",
    )


# First-class domain IR and ISA schema aliases
GhostInstructionRef = GhostIsaRef
GhostOperationRef = GhostMlirRef

RDNA_INSTRUCTION_PRIMITIVES: dict[str, dict[str, Any]] = {
    "V_FMA_F32": {
        "mnemonic": "V_FMA_F32",
        "operands": ["dst", "src0", "src1", "src2"],
        "register_classes": {
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
            "src2": RegisterClass.VGPR,
        },
    },
    "V_ADD_F32": {
        "mnemonic": "V_ADD_F32",
        "operands": ["dst", "src0", "src1"],
        "register_classes": {
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    },
    "V_MUL_F32": {
        "mnemonic": "V_MUL_F32",
        "operands": ["dst", "src0", "src1"],
        "register_classes": {
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
        },
    },
    "V_DOT2_F32_F16": {
        "mnemonic": "V_DOT2_F32_F16",
        "operands": ["dst", "src0", "src1", "src2"],
        "register_classes": {
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
            "src2": RegisterClass.VGPR,
        },
    },
    "V_DOT4_I32_I8": {
        "mnemonic": "V_DOT4_I32_I8",
        "operands": ["dst", "src0", "src1", "src2"],
        "register_classes": {
            "dst": RegisterClass.VGPR,
            "src0": RegisterClass.VGPR,
            "src1": RegisterClass.VGPR,
            "src2": RegisterClass.VGPR,
        },
    },
}

SASS_INSTRUCTION_PRIMITIVES: dict[str, dict[str, Any]] = {
    "FFMA": {
        "mnemonic": "FFMA",
        "operands": ["dst", "src0", "src1", "src2"],
        "barrier_predicates": ["@P0", "@!P0", "@PT"],
    },
    "FADD": {
        "mnemonic": "FADD",
        "operands": ["dst", "src0", "src1"],
        "barrier_predicates": ["@P0", "@!P0", "@PT"],
    },
    "FMUL": {
        "mnemonic": "FMUL",
        "operands": ["dst", "src0", "src1"],
        "barrier_predicates": ["@P0", "@!P0", "@PT"],
    },
    "HMMA": {
        "mnemonic": "HMMA",
        "operands": ["dst", "src0", "src1", "src2"],
        "barrier_predicates": ["@P0", "@!P0", "@PT"],
    },
    "LDG": {
        "mnemonic": "LDG",
        "operands": ["dst", "src0"],
        "memory_space": MemorySpace.GLOBAL,
    },
    "STS": {
        "mnemonic": "STS",
        "operands": ["dst", "src0"],
        "memory_space": MemorySpace.SHARED,
    },
}

WGSL_PRIMITIVE_SIGNATURES: dict[str, dict[str, Any]] = {
    "workgroupBarrier": {
        "inputs": [],
        "outputs": [],
        "qualifier": WGSLQualifier.WORKGROUP,
    },
    "storageBarrier": {
        "inputs": [],
        "outputs": [],
        "qualifier": WGSLQualifier.STORAGE_READ_WRITE,
    },
    "fma": {
        "inputs": ["a", "b", "c"],
        "outputs": ["result"],
    },
    "dot": {
        "inputs": ["a", "b"],
        "outputs": ["result"],
    },
    "textureSample": {
        "inputs": ["texture", "sampler", "coords"],
        "outputs": ["result"],
    },
}


class SnapshotEnvelope(BaseModel):
    """Structured provenance envelope for framework and ISA/IR snapshots."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(default="2.0.0", description="Snapshot schema version.")
    target: str = Field(..., description="Target framework, dialect, or hardware ISA.")
    version: str | None = Field(
        default=None,
        description="Upstream framework version or toolkit release.",
    )
    upstream_version: str | None = Field(
        default=None,
        description="Upstream hardware specification or compiler version.",
    )
    source_type: str | None = Field(
        default=None,
        description="Extraction source (tablegen, binary_disassembly, python_ast).",
    )
    upstream_commit: str | None = Field(
        default=None, description="Upstream git commit hash or release tag."
    )
    supported_microarchitectures: list[str] | None = Field(
        default=None,
        description="Explicit list of supported GPU compute capabilities or target architectures.",
    )
    generated_at: str | None = Field(
        default=None, description="ISO-8601 generation timestamp."
    )
    environment: dict[str, Any] | None = Field(
        default=None, description="Build host environment metadata."
    )
    categories: dict[str, list[Any]] = Field(
        default_factory=dict, description="Categorized symbol dictionaries."
    )


class LogicOp(str, Enum):
    """Supported operators for conditional logic rules in operations."""

    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    IS_TYPE = "is_type"


class StandardMap(BaseModel):
    """Defines how a Framework implements a Middle Layer standard."""

    model_config = ConfigDict(extra="ignore")

    api: str | None = Field(default=None)
    args: dict[str, str | float | int | None] | None = Field(default=None)
    inject_args: dict[str, Any] | None = Field(default=None)
    requires_plugin: str | None = Field(default=None)
    transformation_type: str | None = Field(default=None)
    operator: str | None = Field(default=None)
    pack_to_tuple: str | None = Field(default=None)
    macro_template: str | None = Field(default=None)


def migrate_ghost_ref(data: dict[str, Any]) -> GhostRef:
    """Migrates a v1.x JSON dict to a v1.2 compatible GhostRef instance.

    Args:
        data: The dictionary representation of a GhostRef, possibly from an older schema.

    Returns:
        A valid GhostRef object.
    """
    if "schema_version" not in data:
        data["schema_version"] = "1.2"

    # Ensure parameter kinds map to valid ParameterKind enum values
    if "params" in data and isinstance(data["params"], list):
        for param in data["params"]:
            if "kind" in param and isinstance(param["kind"], str):
                # Standardize inspect._ParameterKind strings to our enum values
                # e.g., 'POSITIONAL_OR_KEYWORD' or '_ParameterKind.POSITIONAL_OR_KEYWORD'
                kind_str = param["kind"].split(".")[-1]
                param["kind"] = kind_str

    return GhostRef.model_validate(data)


def migrate_ghost_ref_v2(data: dict[str, Any]) -> ExtendedGhostRef:
    """Seamlessly upgrade v1.x or v2.x dictionaries to ExtendedGhostRef structure.

    Args:
        data (Dict[str, Any]): Dictionary representation of ghost reference.

    Returns:
        ExtendedGhostRef: Validated ExtendedGhostRef instance.
    """
    data_copy = dict(data)
    if "schema_version" not in data_copy or data_copy["schema_version"] in (
        "1.0",
        "1.0.0",
        "1.2",
    ):
        data_copy["schema_version"] = "2.0.0"

    if "kind" not in data_copy:
        data_copy["kind"] = "function"

    if "name" not in data_copy:
        if "mnemonic" in data_copy:
            data_copy["name"] = data_copy["mnemonic"]
        elif "api_path" in data_copy:
            data_copy["name"] = data_copy["api_path"].split(".")[-1]
        else:
            data_copy["name"] = "unknown"

    if "api_path" not in data_copy:
        data_copy["api_path"] = data_copy.get("name", "unknown")

    if "params" in data_copy and isinstance(data_copy["params"], list):
        for param in data_copy["params"]:
            if (
                isinstance(param, dict)
                and "kind" in param
                and isinstance(param["kind"], str)
            ):
                kind_str = param["kind"].split(".")[-1]
                param["kind"] = kind_str

    domain_type = data_copy.get("domain_type")
    if domain_type == "python":
        return GhostPythonRef.model_validate(data_copy)
    if domain_type == "isa" or domain_type == "instruction" or "mnemonic" in data_copy:
        return GhostIsaRef.model_validate(data_copy)
    if domain_type == "mlir" or domain_type == "operation":
        return GhostMlirRef.model_validate(data_copy)

    return ExtendedGhostRef.model_validate(data_copy)
