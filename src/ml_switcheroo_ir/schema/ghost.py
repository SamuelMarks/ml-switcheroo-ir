"""Shared Data Models for ML Framework Snapshot introspection.

Provides the Ghost Protocol (GhostRef, GhostParam) schemas used to communicate
API structures between the ml-ecosystem-snapshots scraper and the ml-switcheroo compiler.
"""

from __future__ import annotations

from enum import Enum
import typing

from pydantic import BaseModel, ConfigDict, Field

from ml_switcheroo_ir.types import JsonValue


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


class IsaControlCodesSpec(BaseModel):
    """Pydantic model for ISA instruction control codes and scheduling metadata."""

    model_config = ConfigDict(extra="allow")

    latency: int | None = None
    stall_count: int | None = None
    yield_flag: bool | None = None
    write_barrier: int | None = None
    read_barrier: int | None = None
    wait_barrier_mask: int | str | None = None


class IsaStructuredOperandSpec(BaseModel):
    """Pydantic model for structured ISA operand specifications."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    role: str | None = None
    reg: str | None = None
    register_class: str | None = None


class VopdProfileSpec(BaseModel):
    """Pydantic model for AMD RDNA3 VOPD dual-issue profile."""

    model_config = ConfigDict(extra="allow")

    slot: str | None = None
    slots: list[str] | None = None
    max_src_operands: int | None = None


class RdnaPrimitiveSpec(BaseModel):
    """Pydantic model for AMD RDNA instruction primitive specifications."""

    model_config = ConfigDict(extra="forbid")

    mnemonic: str
    operands: list[str]
    register_classes: dict[str, RegisterClass]


class VopdOperatorSpec(BaseModel):
    """Pydantic model for RDNA3 VOPD dual-issue operator specifications."""

    model_config = ConfigDict(extra="forbid")

    mnemonic: str
    slots: list[str]


class SassPrimitiveSpec(BaseModel):
    """Pydantic model for NVIDIA SASS instruction primitive specifications."""

    model_config = ConfigDict(extra="forbid")

    mnemonic: str
    operands: list[str]
    barrier_predicates: list[str] | None = None
    memory_space: MemorySpace | None = None


class WgslPrimitiveSpec(BaseModel):
    """Pydantic model for WebGPU WGSL primitive signature specifications."""

    model_config = ConfigDict(extra="forbid")

    inputs: list[str]
    outputs: list[str]
    qualifier: WGSLQualifier | None = None


class GhostParam(BaseModel):
    """Serializable representation of a function parameter."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Parameter name.")
    kind: ParameterKind = Field(
        description="Kind of parameter (e.g. POSITIONAL_OR_KEYWORD)."
    )
    default: JsonValue = Field(default=None, description="Default value as string.")
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
    """Extended GhostParam supporting operand directionality, IR roles, allowed_dtypes, rank_constraint, and factory defaults."""

    model_config = ConfigDict(extra="allow")

    default: JsonValue = Field(
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
    allowed_dtypes: list[str] | None = Field(
        default=None,
        description="Canonical allowed tensor dtypes (e.g. ['float32', 'bfloat16']).",
    )
    allowed_values: list[str] | None = Field(
        default=None,
        description="Allowed enum or literal string values (e.g. ['none', 'mean', 'sum']).",
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
        default="2.0.0", description="Version of the schema format."
    )

    is_public: bool | None = Field(default=None, description="Is public")
    is_c_extension: bool | None = Field(
        default=False,
        description="Whether the symbol originates from a compiled C/C++ extension.",
    )
    accepted_kwargs: list[str] | None = Field(
        default=None,
        description="Explicit list of accepted keyword arguments when **kwargs is present.",
    )

    aliases: list[str] | None = Field(default_factory=list, description="Aliases")
    returns_type: str | None = Field(default=None, description="Returns type")
    returns_description: str | None = Field(
        default=None, description="Returns description"
    )
    raises: list[str] | None = Field(default_factory=list, description="Raises")
    environment_tags: list[str] | None = Field(
        default_factory=list, description="Environment tags"
    )
    overloads: list[dict[str, JsonValue] | GhostRef] | None = Field(
        default_factory=list, description="Overloads"
    )

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

    model_config = ConfigDict(extra="allow")

    params: list[ExtendedGhostParam | GhostParam] = Field(
        default_factory=list,
        description="List of extended parameter specifications.",
    )
    returns: list[GhostResult] | None = Field(
        default=None, description="Multiple SSA returns or results."
    )
    domain_metadata: dict[str, JsonValue] | None = Field(
        default=None,
        description="Structured domain metadata for ISAs and compilers.",
    )
    signature_completeness: typing.Literal["exact", "heuristic", "opaque"] | None = (
        Field(
            default="exact",
            description="Completeness of signature resolution: exact, heuristic, or opaque.",
        )
    )
    is_c_extension: bool | None = Field(
        default=False,
        description="Whether the symbol originates from a compiled C/C++ extension.",
    )
    accepted_kwargs: list[str] | None = Field(
        default=None,
        description="Explicit list of accepted keyword arguments when **kwargs is present.",
    )


class GhostPythonRef(ExtendedGhostRef):
    """GhostRef specialized for high-level Python ML frameworks (PyTorch, JAX, TF, Keras)."""

    model_config = ConfigDict(extra="allow")
    domain_type: typing.Literal["python"] = "python"


class GhostIsaRef(ExtendedGhostRef):
    """GhostRef specialized for GPU assembly ISAs (NVIDIA SASS, AMD RDNA/CDNA)."""

    model_config = ConfigDict(extra="allow")
    domain_type: typing.Literal["isa", "instruction"] = "isa"
    predicate_guards: list[str] | None = Field(
        default=None,
        description="Allowed predicate guard registers (e.g. ['@P0', '@!P1', '@PT']).",
    )
    register_classes: dict[str, str] | None = Field(
        default=None,
        description="Register classes for operands (e.g. {'op0': 'VGPR_32', 'op1': 'VReg_64'}).",
    )
    control_codes: IsaControlCodesSpec | None = Field(
        default=None,
        description="Instruction control code and scheduling schema.",
    )
    instruction_modifiers: list[str] | None = Field(
        default=None,
        description="Valid instruction modifiers (e.g. ['.SAT', '.FTZ', 'omod:2']).",
    )
    structured_modifiers: dict[str, JsonValue] | None = Field(
        default=None,
        description="Structured modifier bitfields (e.g. rounding, cache, saturation).",
    )
    structured_operands: list[IsaStructuredOperandSpec] | None = Field(
        default=None,
        description="Detailed operand records with roles, register classes, and immediate constraints.",
    )
    vopd_profile: VopdProfileSpec | None = Field(
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
    operand_signatures: list[list[str]] | None = Field(
        default=None,
        description="Explicit register or immediate operand signatures (e.g. [['vgpr', 'vgpr', 'sgpr']]).",
    )
    allowed_modifiers: list[str] | None = Field(
        default=None,
        description="Legal instruction modifier flags (e.g. ['.SAT', '.FTZ', 'omod:2']).",
    )
    execution_latency: int | tuple[int, int] | None = Field(
        default=None,
        description="Analytical execution clock cycle latency bounds.",
    )
    issue_rate: float | None = Field(
        default=None,
        description="Throughput instructions issued per clock cycle.",
    )
    scoreboard_barrier_mask: int | None = Field(
        default=None,
        description="Hardware dependency barrier index mask (0-5).",
    )
    vopd_slot: str | None = Field(
        default=None,
        description="Dual-issue slot designation ('X', 'Y', or 'BOTH') for AMD RDNA3 VOPD instructions.",
    )


class GhostMlirRef(ExtendedGhostRef):
    """GhostRef specialized for compiler IR dialects (Core MLIR and StableHLO)."""

    model_config = ConfigDict(extra="allow")
    domain_type: typing.Literal["mlir", "operation"] = "mlir"
    traits: list[str] | None = Field(
        default=None,
        description="Dialect verification traits (e.g. ['SameOperandsAndResultType', 'Commutative']).",
    )
    operands: list[ExtendedGhostParam | GhostParam] | None = Field(
        default=None,
        description="Strictly decoupled SSA value arguments (operands).",
    )
    attributes: dict[str, JsonValue] | None = Field(
        default=None,
        description="Structured attribute specifications and schemas.",
    )
    regions: dict[str, JsonValue] | None = Field(
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
    "V_DUAL_SUBREV_F32": VopdOperatorSpec(
        mnemonic="V_DUAL_SUBREV_F32", slots=["X", "Y"]
    ),
    "V_DUAL_MOV_B32": VopdOperatorSpec(mnemonic="V_DUAL_MOV_B32", slots=["X", "Y"]),
    "V_DUAL_CNDMASK_B32": VopdOperatorSpec(mnemonic="V_DUAL_CNDMASK_B32", slots=["X"]),
    "V_DUAL_DOT2ACC_F32_F16": VopdOperatorSpec(
        mnemonic="V_DUAL_DOT2ACC_F32_F16", slots=["Y"]
    ),
}

# Standard RDNA equivalents that can be mapped to VOPD opcodes
RDNA_TO_VOPD_MAP: dict[str, str] = {
    "V_FMAC_F32": "V_DUAL_FMAC_F32",
    "V_FMA_F32": "V_DUAL_FMAC_F32",
    "V_MUL_F32": "V_DUAL_MUL_F32",
    "V_ADD_F32": "V_DUAL_ADD_F32",
    "V_SUB_F32": "V_DUAL_SUB_F32",
    "V_MOV_B32": "V_DUAL_MOV_B32",
    "V_CNDMASK_B32": "V_DUAL_CNDMASK_B32",
    "V_DOT2_F32_F16": "V_DUAL_DOT2ACC_F32_F16",
}

SASS_PIPELINE_LATENCIES: dict[str, int] = {
    "FFMA": 4,
    "FADD": 4,
    "FMUL": 4,
    "HMMA": 8,
    "LDG": 200,
    "STS": 16,
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
    environment: dict[str, JsonValue] | None = Field(
        default=None, description="Build host environment metadata."
    )
    categories: dict[str, list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef]] = (
        Field(default_factory=dict, description="Categorized symbol dictionaries.")
    )
    operations: list[dict[str, JsonValue] | GhostMlirRef] | None = Field(
        default=None, description="Flat list of IR or dialect operations."
    )
    instructions: list[dict[str, JsonValue] | GhostIsaRef] | None = Field(
        default=None, description="Flat list of hardware ISA instructions."
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
    inject_args: dict[str, JsonValue] | None = Field(default=None)
    requires_plugin: str | None = Field(default=None)
    transformation_type: str | None = Field(default=None)
    operator: str | None = Field(default=None)
    pack_to_tuple: str | None = Field(default=None)
    macro_template: str | None = Field(default=None)
