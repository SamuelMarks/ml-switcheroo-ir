"""Tests for Ghost Protocol v2 data models and schema migration."""

from __future__ import annotations

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostIsaRef,
    GhostMlirRef,
    GhostParam,
    GhostResult,
    IRParameterRole,
    OperandDirection,
    ParameterKind,
    SnapshotEnvelope,
    migrate_ghost_ref_v2,
)


def test_ghost_v2_enums_and_results() -> None:
    """Test OperandDirection, IRParameterRole, and GhostResult instantiation."""
    assert OperandDirection.READ == "READ"
    assert OperandDirection.WRITE == "WRITE"
    assert OperandDirection.READ_WRITE == "READ_WRITE"
    assert OperandDirection.PREDICATE == "PREDICATE"

    assert IRParameterRole.OPERAND == "OPERAND"
    assert IRParameterRole.ATTRIBUTE == "ATTRIBUTE"
    assert IRParameterRole.RESULT == "RESULT"
    assert IRParameterRole.SUCCESSOR == "SUCCESSOR"
    assert IRParameterRole.REGION == "REGION"

    res = GhostResult(
        name="out0",
        type="tensor<1x32xf32>",
        description="Activation output",
    )
    assert res.name == "out0"
    assert res.type == "tensor<1x32xf32>"


def test_extended_ghost_param() -> None:
    """Test ExtendedGhostParam with direction, role, dtypes, rank, and constraints."""
    param = ExtendedGhostParam(
        name="lhs",
        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
        direction=OperandDirection.READ,
        role=IRParameterRole.OPERAND,
        dtypes=["float32", "bfloat16"],
        rank=2,
        rank_constraint="==2",
        is_contracting_dim=True,
    )
    assert param.name == "lhs"
    assert param.direction == OperandDirection.READ
    assert param.role == IRParameterRole.OPERAND
    assert param.dtypes == ["float32", "bfloat16"]
    assert param.rank == 2
    assert param.rank_constraint == "==2"
    assert param.is_contracting_dim is True


def test_extended_ghost_ref() -> None:
    """Test ExtendedGhostRef with returns, domain_metadata, and signature_completeness."""
    param = ExtendedGhostParam(
        name="x",
        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
        direction=OperandDirection.READ,
        role=IRParameterRole.OPERAND,
    )
    res = GhostResult(name="y", type="tensor<?xf32>")
    ref = ExtendedGhostRef(
        name="relu",
        api_path="arith.relu",
        kind="operation",
        params=[param],
        returns=[res],
        domain_metadata={"lowering_tier": "linalg"},
        signature_completeness="exact",
    )
    assert ref.name == "relu"
    assert len(ref.params) == 1
    assert ref.returns == [res]
    assert ref.domain_metadata == {"lowering_tier": "linalg"}
    assert ref.signature_completeness == "exact"


def test_ghost_isa_ref() -> None:
    """Test GhostIsaRef with GPU assembly attributes."""
    ref = GhostIsaRef(
        name="FFMA",
        api_path="nvidia_sass.ffma",
        kind="instruction",
        domain_type="isa",
        predicate_guards=["@P0", "@!PT"],
        register_classes={"op0": "R0", "op1": "R1"},
        control_codes={"latency": 4},
        instruction_modifiers=[".SAT", ".FTZ"],
        vopd_profile=None,
    )
    assert ref.domain_type == "isa"
    assert ref.predicate_guards == ["@P0", "@!PT"]
    assert ref.register_classes == {"op0": "R0", "op1": "R1"}
    assert ref.control_codes == {"latency": 4}
    assert ref.instruction_modifiers == [".SAT", ".FTZ"]


def test_ghost_mlir_ref() -> None:
    """Test GhostMlirRef with traits, operands, attributes, and regions."""
    operand = ExtendedGhostParam(
        name="lhs",
        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
        role=IRParameterRole.OPERAND,
    )
    ref = GhostMlirRef(
        name="dot_general",
        api_path="stablehlo.dot_general",
        kind="operation",
        domain_type="mlir",
        traits=["SameOperandsAndResultType"],
        operands=[operand],
        attributes={"dot_dimension_numbers": {}},
        regions=None,
        successors=None,
        type_constraints={"T": "RankedTensorOf"},
    )
    assert ref.domain_type == "mlir"
    assert ref.traits == ["SameOperandsAndResultType"]
    assert len(ref.operands or []) == 1
    assert "dot_dimension_numbers" in (ref.attributes or {})
    assert ref.type_constraints == {"T": "RankedTensorOf"}


def test_snapshot_envelope() -> None:
    """Test SnapshotEnvelope provenance metadata model."""
    env = SnapshotEnvelope(
        schema_version="2.0.0",
        target="stablehlo",
        version="1.0.0",
        upstream_version="llvm-19",
        source_type="tablegen",
        upstream_commit="abcdef123",
        supported_microarchitectures=["sm_90", "gfx942"],
        generated_at="2026-09-11T00:00:00Z",
        environment={"python": "3.12"},
        categories={"core": []},
    )
    assert env.target == "stablehlo"
    assert env.schema_version == "2.0.0"
    assert env.supported_microarchitectures == ["sm_90", "gfx942"]


def test_migrate_ghost_ref_v2_backward_compatibility() -> None:
    """Test migrating v1.x and v2.x dicts with migrate_ghost_ref_v2."""
    v1_dict = {
        "name": "Linear",
        "api_path": "torch.nn.Linear",
        "kind": "class",
        "schema_version": "1.0",
        "params": [
            {
                "name": "in_features",
                "kind": "inspect._ParameterKind.POSITIONAL_OR_KEYWORD",
            },
            {"name": "bias", "kind": "KEYWORD_ONLY"},
        ],
    }
    migrated_v1 = migrate_ghost_ref_v2(v1_dict)
    assert migrated_v1.schema_version == "2.0.0"

    # Test no schema_version provided defaults to 2.0.0
    no_version_dict = {
        "name": "Linear0",
        "api_path": "torch.nn.Linear0",
        "kind": "class",
    }
    assert migrate_ghost_ref_v2(no_version_dict).schema_version == "2.0.0"

    # Test dict without params key
    no_params_dict = {
        "name": "LinearDict",
        "api_path": "torch.nn.LinearDict",
        "kind": "class",
    }
    assert migrate_ghost_ref_v2(no_params_dict).schema_version == "2.0.0"

    # Test params containing GhostParam instance directly
    gp = GhostParam(name="y", kind=ParameterKind.KEYWORD_ONLY)
    param_int_dict = {
        "name": "LinearParam",
        "api_path": "torch.nn.LinearParam",
        "kind": "class",
        "params": [
            {"name": "x", "kind": ParameterKind.POSITIONAL_ONLY},
            gp,
        ],
    }
    assert len(migrate_ghost_ref_v2(param_int_dict).params) == 2

    # Test existing 2.0.0 schema version preserved
    v2_dict = {
        "name": "Linear2",
        "api_path": "torch.nn.Linear2",
        "kind": "class",
        "schema_version": "2.0.0",
    }
    assert migrate_ghost_ref_v2(v2_dict).schema_version == "2.0.0"

    isa_dict = {
        "name": "V_ADD_F32",
        "api_path": "amd_rdna.v_add_f32",
        "kind": "instruction",
        "domain_type": "isa",
        "predicate_guards": ["@vcc"],
    }
    migrated_isa = migrate_ghost_ref_v2(isa_dict)
    assert isinstance(migrated_isa, GhostIsaRef)
    assert migrated_isa.predicate_guards == ["@vcc"]

    isa_instr_dict = {
        "name": "V_ADD_F32_INSTR",
        "api_path": "amd_rdna.v_add_f32_instr",
        "kind": "instruction",
        "domain_type": "instruction",
    }
    assert isinstance(migrate_ghost_ref_v2(isa_instr_dict), GhostIsaRef)

    mlir_dict = {
        "name": "matmul",
        "api_path": "linalg.matmul",
        "kind": "operation",
        "domain_type": "mlir",
        "traits": ["ContractionOpInterface"],
    }
    migrated_mlir = migrate_ghost_ref_v2(mlir_dict)
    assert isinstance(migrated_mlir, GhostMlirRef)
    assert migrated_mlir.traits == ["ContractionOpInterface"]

    mlir_op_dict = {
        "name": "matmul_op",
        "api_path": "linalg.matmul_op",
        "kind": "operation",
        "domain_type": "operation",
    }
    assert isinstance(migrate_ghost_ref_v2(mlir_op_dict), GhostMlirRef)
