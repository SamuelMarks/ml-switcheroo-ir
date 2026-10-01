"""Exhaustive tests covering schema registries, loaders, snapshot discovery, and translation edge cases."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from ml_switcheroo_ir.schema import (
    custom_ops,
    framework_registries,
    low_level_registries,
    mlir_registry,
    onnx_registry,
    rdna_registry,
    sass_registry,
    stablehlo,
)
from ml_switcheroo_ir.snapshots import (
    find_schema_file,
    get_default_snapshots_dir,
)
from ml_switcheroo_ir.translation import ParameterTranslationEngine
from scripts.verify_grounding import verify_stablehlo_grounding


def test_state_ops_loader_with_custom_valid_json(tmp_path: Path) -> None:
    """Verify that _load_state_schemas loads custom operator schemas from JSON.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    custom_file = tmp_path / "custom_state.json"
    custom_data = {
        "ops": [
            {
                "name": "CustomStateOp",
                "domain": "ml.switcheroo.state",
                "inputs": ["in_var"],
                "outputs": ["out_var"],
                "attributes": [
                    {
                        "name": "param_a",
                        "type": "int",
                        "required": True,
                        "default": 1,
                    }
                ],
            }
        ]
    }
    with open(custom_file, "w", encoding="utf-8") as f:
        json.dump(custom_data, f)

    try:
        custom_ops._load_state_schemas(json_path=custom_file)
        assert "CustomStateOp" in custom_ops.STATE_OPS_REGISTRY
        schema = custom_ops.STATE_OPS_REGISTRY["CustomStateOp"]
        assert schema.name == "CustomStateOp"
        assert schema.domain == "ml.switcheroo.state"
        assert "param_a" in schema.attributes
    finally:
        custom_ops.STATE_OPS_REGISTRY.pop("CustomStateOp", None)


def test_framework_registries_loader_with_json_and_dialects(
    tmp_path: Path,
) -> None:
    """Verify _load_framework_schema_file handles custom files and dialect mappings.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    custom_file = tmp_path / "custom_fw.json"
    custom_data = {
        "ops": [
            {
                "name": "custom_add",
                "domain": "custom_fw",
                "inputs": ["a", "b"],
                "outputs": ["out"],
                "attributes": [
                    {
                        "name": "alpha",
                        "type": "float",
                        "required": False,
                        "default": 1.0,
                    }
                ],
                "dialects": {"torch": "torch.add", "jax": "jax.numpy.add"},
            },
            {
                "name": "custom_no_dialects",
                "domain": "custom_fw",
                "inputs": [],
                "outputs": [],
                "attributes": [],
            },
        ]
    }
    with open(custom_file, "w", encoding="utf-8") as f:
        json.dump(custom_data, f)

    res = framework_registries._load_framework_schema_file(
        "custom_fw.json", "custom_fw", json_path=custom_file
    )
    assert "custom_add" in res
    assert "custom_no_dialects" in res
    assert framework_registries.ODL_DIALECT_MAPPINGS.get("custom_add") == {
        "torch": "torch.add",
        "jax": "jax.numpy.add",
    }


def test_framework_registries_non_empty_initialization() -> None:
    """Verify _initialize_framework_registries branch when schemas are already loaded from files."""
    dummy_op = mock.MagicMock()
    with mock.patch(
        "ml_switcheroo_ir.schema.framework_registries._load_framework_schema_file",
        return_value={"mock_op": dummy_op},
    ):
        try:
            framework_registries._initialize_framework_registries()
            assert "mock_op" in framework_registries.ATEN_REGISTRY
            assert "mock_op" in framework_registries.ARRAY_API_REGISTRY
            assert "mock_op" in framework_registries.ODL_CATALOG
        finally:
            framework_registries.ATEN_REGISTRY.pop("mock_op", None)
            framework_registries.ARRAY_API_REGISTRY.pop("mock_op", None)
            framework_registries.ODL_CATALOG.pop("mock_op", None)
    framework_registries._initialize_framework_registries()


def test_low_level_registries_loader_with_custom_json(tmp_path: Path) -> None:
    """Verify _load_schema_file for low level registries with custom JSON content.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    custom_file = tmp_path / "custom_ll.json"
    custom_data = {
        "ops": [
            {
                "name": "custom_ll_op",
                "domain": "custom_ll",
                "inputs": ["src"],
                "outputs": ["dst"],
                "attributes": [{"name": "mode", "type": "str", "required": False}],
            }
        ]
    }
    with open(custom_file, "w", encoding="utf-8") as f:
        json.dump(custom_data, f)

    res = low_level_registries._load_schema_file(
        "custom_ll.json", "custom_ll", json_path=custom_file
    )
    assert "custom_ll_op" in res
    assert res["custom_ll_op"].domain == "custom_ll"


def test_low_level_registries_initialization_branches() -> None:
    """Verify _initialize_low_level_registries branches when loaders return empty or non-empty."""
    dummy_op = mock.MagicMock()
    with mock.patch(
        "ml_switcheroo_ir.schema.low_level_registries._load_schema_file",
        side_effect=lambda *args, **kwargs: {"mock_ll": dummy_op},
    ):
        try:
            low_level_registries._initialize_low_level_registries()
            assert "mock_ll" in low_level_registries.PTX_REGISTRY
            assert "mock_ll" in low_level_registries.METAL_REGISTRY
            assert "mock_ll" in low_level_registries.WASM_REGISTRY
            assert "mock_ll" in low_level_registries.WEBGL_REGISTRY
            assert "mock_ll" in low_level_registries.WGSL_REGISTRY
        finally:
            low_level_registries.PTX_REGISTRY.pop("mock_ll", None)
            low_level_registries.METAL_REGISTRY.pop("mock_ll", None)
            low_level_registries.WASM_REGISTRY.pop("mock_ll", None)
            low_level_registries.WEBGL_REGISTRY.pop("mock_ll", None)
            low_level_registries.WGSL_REGISTRY.pop("mock_ll", None)

    # Empty branch
    with mock.patch(
        "ml_switcheroo_ir.schema.low_level_registries._load_schema_file",
        side_effect=lambda *args, **kwargs: {},
    ):
        low_level_registries._initialize_low_level_registries()
        assert "storageStore" in low_level_registries.WGSL_REGISTRY


def test_mlir_schemas_loader_with_custom_json(tmp_path: Path) -> None:
    """Verify _load_mlir_schemas loads custom MLIR operations from JSON.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    custom_file = tmp_path / "custom_mlir.json"
    custom_data = {
        "ops": [
            {
                "name": "custom.op",
                "domain": "mlir.custom",
                "inputs": ["in"],
                "outputs": ["out"],
                "attributes": [{"name": "attr1", "type": "int", "required": True}],
            }
        ]
    }
    with open(custom_file, "w", encoding="utf-8") as f:
        json.dump(custom_data, f)

    try:
        mlir_registry._load_mlir_schemas(json_path=custom_file)
        assert "custom.op" in mlir_registry.MLIR_REGISTRY
    finally:
        mlir_registry.MLIR_REGISTRY.pop("custom.op", None)


def test_registry_loaders_with_nonexistent_custom_json(tmp_path: Path) -> None:
    """Verify registry loaders handle non-existent json_path overrides without exceptions.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    nonexistent = tmp_path / "does_not_exist.json"
    mlir_registry._load_mlir_schemas(json_path=nonexistent)
    rdna_registry._load_rdna_schemas(json_path=nonexistent)
    sass_registry._load_sass_schemas(json_path=nonexistent)
    stablehlo._load_stablehlo_schemas(json_path=nonexistent)


def test_rdna_schemas_loader_with_custom_json(tmp_path: Path) -> None:
    """Verify _load_rdna_schemas loads custom RDNA operations with and without slot mappings.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    custom_file = tmp_path / "custom_rdna.json"
    custom_data = {
        "ops": [
            {
                "name": "V_CUSTOM_F32",
                "domain": "amd_rdna",
                "vopd_slot": "BOTH",
                "vopd_target": "V_DUAL_CUSTOM_F32",
                "inputs": ["src0", "src1"],
                "outputs": ["dst"],
                "attributes": [],
            },
            {
                "name": "V_PLAIN_OP",
                "domain": "amd_rdna",
                "inputs": ["src0"],
                "outputs": ["dst"],
                "attributes": [],
            },
        ]
    }
    with open(custom_file, "w", encoding="utf-8") as f:
        json.dump(custom_data, f)

    try:
        rdna_registry._load_rdna_schemas(json_path=custom_file)
        assert "V_CUSTOM_F32" in rdna_registry.RDNA_REGISTRY
        assert "V_PLAIN_OP" in rdna_registry.RDNA_REGISTRY
        assert rdna_registry.RDNA_VOPD_SLOTS.get("V_CUSTOM_F32") == "BOTH"
        assert rdna_registry.RDNA_TO_VOPD_MAP.get("V_CUSTOM_F32") == "V_DUAL_CUSTOM_F32"
    finally:
        rdna_registry.RDNA_REGISTRY.pop("V_CUSTOM_F32", None)
        rdna_registry.RDNA_REGISTRY.pop("V_PLAIN_OP", None)
        rdna_registry.RDNA_VOPD_SLOTS.pop("V_CUSTOM_F32", None)
        rdna_registry.RDNA_TO_VOPD_MAP.pop("V_CUSTOM_F32", None)


def test_sass_schemas_loader_with_custom_json(tmp_path: Path) -> None:
    """Verify _load_sass_schemas loads custom SASS operations and latencies.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    custom_file = tmp_path / "custom_sass.json"
    custom_data = {
        "ops": [
            {
                "name": "CUSTOM_SASS",
                "domain": "nvidia_sass",
                "execution_latency": 12,
                "inputs": ["src0"],
                "outputs": ["dst"],
                "attributes": [],
            }
        ]
    }
    with open(custom_file, "w", encoding="utf-8") as f:
        json.dump(custom_data, f)

    try:
        sass_registry._load_sass_schemas(json_path=custom_file)
        assert "CUSTOM_SASS" in sass_registry.SASS_REGISTRY
        assert sass_registry.SASS_PIPELINE_LATENCIES.get("CUSTOM_SASS") == 12
    finally:
        sass_registry.SASS_REGISTRY.pop("CUSTOM_SASS", None)
        sass_registry.SASS_PIPELINE_LATENCIES.pop("CUSTOM_SASS", None)


def test_stablehlo_schemas_loader_with_custom_json(tmp_path: Path) -> None:
    """Verify _load_stablehlo_schemas loads custom StableHLO operations.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    custom_file = tmp_path / "custom_hlo.json"
    custom_data = {
        "ops": [
            {
                "name": "custom_hlo_op",
                "domain": "stablehlo",
                "inputs": ["x"],
                "outputs": ["y"],
                "attributes": [{"name": "dim", "type": "int", "required": False}],
            }
        ]
    }
    with open(custom_file, "w", encoding="utf-8") as f:
        json.dump(custom_data, f)

    try:
        stablehlo._load_stablehlo_schemas(json_path=custom_file)
        assert "custom_hlo_op" in stablehlo.STABLEHLO_REGISTRY
    finally:
        stablehlo.STABLEHLO_REGISTRY.pop("custom_hlo_op", None)


def test_stablehlo_loader_error_and_malformed_entries() -> None:
    """Verify _load_stablehlo_schemas error branch and malformed entry skips."""
    from pathlib import Path

    with mock.patch("builtins.open", side_effect=OSError("Read error")), mock.patch(
        "ml_switcheroo_ir.schema.stablehlo.find_schema_file",
        return_value=Path("/dummy/stablehlo_ops.json"),
    ), mock.patch("pathlib.Path.exists", return_value=True):
        stablehlo._load_stablehlo_schemas()

    # Hit the fallback branch exception
    with mock.patch("builtins.open", side_effect=OSError("Read error")), mock.patch(
        "ml_switcheroo_ir.schema.stablehlo.find_schema_file", return_value=None
    ), mock.patch("pathlib.Path.exists", return_value=True):
        stablehlo._load_stablehlo_schemas()

    malformed_data = {
        "operations": [
            {"name": ""},
            {
                "name": "op_with_malformed_attr",
                "attributes": [
                    "not_a_dict",
                    {},
                    {"name": "valid_attr", "type": "int"},
                ],
                "operands": [],
                "results": [],
            },
        ]
    }
    try:
        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data=json.dumps(malformed_data)),
        ), mock.patch("pathlib.Path.exists", return_value=True):
            stablehlo._load_stablehlo_schemas()
    finally:
        stablehlo.STABLEHLO_REGISTRY.pop("op_with_malformed_attr", None)


def test_onnx_schemas_loader_edge_cases(tmp_path: Path) -> None:
    """Verify load_onnx_schemas handling of dict-based attributes and proto parser.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    custom_file = tmp_path / "custom_onnx.json"
    custom_data = {
        "CustomOp": {
            "domain": "ai.onnx",
            "version": 1,
            "inputs": ["x"],
            "outputs": ["y"],
            "attributes": {
                "dict_attr": {
                    "type": "int",
                    "required": True,
                    "default": 42,
                }
            },
        }
    }
    with open(custom_file, "w", encoding="utf-8") as f:
        json.dump(custom_data, f)

    res = onnx_registry.load_onnx_schemas(json_path=custom_file)
    assert "CustomOp" in res
    assert "dict_attr" in res["CustomOp"].attributes


def test_onnx_proto_default_parser_branches() -> None:
    """Verify _parse_onnx_proto_default handles non-string and string values correctly."""
    assert onnx_registry._parse_onnx_proto_default(None) is None
    assert onnx_registry._parse_onnx_proto_default(100) == 100
    assert onnx_registry._parse_onnx_proto_default(3.14) == 3.14
    assert onnx_registry._parse_onnx_proto_default('s: "sample_str"') == "sample_str"
    assert onnx_registry._parse_onnx_proto_default("unmatched") == "unmatched"


def test_onnx_loader_framework_dict_attrs_and_exceptions() -> None:
    """Verify load_onnx_schemas handles dict attributes from upstream and error paths."""
    mock_ops = [
        {"name": ""},
        {
            "name": "OpWithDictAttrs",
            "domain": "ai.onnx",
            "attributes": {
                "flag": {"type": "bool", "required": False, "default": "i: 1"},
                "str_flag": "just_a_string",
            },
        },
        {
            "name": "OpWithMixedListAttrs",
            "domain": "ai.onnx",
            "attributes": [
                "not_a_dict",
                {},
                {
                    "name": "valid_attr",
                    "type": "int",
                    "required": True,
                    "default": "i: 1",
                },
            ],
        },
        {
            "name": "OpWithNoneAttrs",
            "domain": "ai.onnx",
            "attributes": None,
        },
    ]
    with mock.patch(
        "ml_ecosystem_snapshots.frameworks.onnx_spec._load_onnx_ops",
        return_value=mock_ops,
    ), mock.patch.dict(onnx_registry.ONNX_REGISTRY, {}, clear=True):
        res = onnx_registry.load_onnx_schemas()
        assert "OpWithDictAttrs" in res
        assert "OpWithMixedListAttrs" in res
        assert "OpWithNoneAttrs" in res

    # Exception path
    with mock.patch(
        "ml_ecosystem_snapshots.frameworks.onnx_spec._load_onnx_ops",
        side_effect=ImportError("Mock import failure"),
    ), mock.patch.dict(onnx_registry.ONNX_REGISTRY, {}, clear=True):
        res_err = onnx_registry.load_onnx_schemas()
        assert res_err == {}

    onnx_registry.load_onnx_schemas()


def test_snapshots_discovery_precedence(tmp_path: Path) -> None:
    """Verify get_default_snapshots_dir precedence paths and environment variables.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    env_dir = tmp_path / "env_snapshots"
    env_dir.mkdir()
    (env_dir / "sample.json").write_text("{}", encoding="utf-8")

    with mock.patch.dict("os.environ", {"ML_ECOSYSTEM_SNAPSHOTS_DIR": str(env_dir)}):
        resolved = get_default_snapshots_dir()
        assert resolved == str(env_dir.resolve())

    # User cache path
    cache_dir = tmp_path / "cache_snapshots"
    cache_dir.mkdir()
    (cache_dir / "sample.json").write_text("{}", encoding="utf-8")
    with mock.patch.dict("os.environ", {}, clear=True), mock.patch(
        "os.path.expanduser", return_value=str(cache_dir)
    ), mock.patch("os.path.isdir", side_effect=lambda p: str(p) == str(cache_dir)):
        resolved_cache = get_default_snapshots_dir()
        assert resolved_cache == str(cache_dir.resolve())


def test_find_schema_file_all_branches(tmp_path: Path) -> None:
    """Verify find_schema_file handles override paths, candidate resolution, and missing files.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    override_file = tmp_path / "explicit_schema.json"
    override_file.write_text("{}", encoding="utf-8")

    # 1. Existing override path
    assert find_schema_file("dummy.json", override_path=override_file) == override_file

    # 2. Non-existent override path
    assert (
        find_schema_file("dummy.json", override_path=tmp_path / "nonexistent.json")
        is None
    )

    # 3. Non-existent filename anywhere
    assert find_schema_file("completely_unknown_file_12345.json") is None

    # 4. Parent candidate
    with mock.patch("pathlib.Path.is_file", side_effect=[False, True]):
        parent_res = find_schema_file("parent_candidate_dummy.json")
        assert parent_res is not None


def test_translation_engine_corrupt_file_handling(tmp_path: Path) -> None:
    """Verify ParameterTranslationEngine gracefully ignores corrupt JSON candidates.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    bad_file = tmp_path / "corrupt_concept_map.json"
    bad_file.write_text("{invalid json", encoding="utf-8")

    engine = ParameterTranslationEngine(concept_map_path=str(bad_file))
    assert engine.translations == {}


def test_translation_engine_fallback_merge_when_no_concept_map() -> None:
    """Verify ParameterTranslationEngine populates fallback translations when concept_map.json is missing."""
    engine = ParameterTranslationEngine()
    engine.translations.clear()
    with mock.patch(
        "builtins.open", side_effect=FileNotFoundError("Missing concept map")
    ):
        engine._load_default_translations(None)
    assert "matmul" in engine.translations
    assert "layer_norm" in engine.translations


def test_verify_stablehlo_grounding_discovery_branches(tmp_path: Path) -> None:
    """Verify verify_stablehlo_grounding handles frameworks/ fallback and missing snapshots.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    # 1. No snapshot file in empty directory
    empty_dir = tmp_path / "empty_snaps"
    empty_dir.mkdir()
    errs = verify_stablehlo_grounding(empty_dir)
    assert len(errs) == 1
    assert "StableHLO snapshot file not found" in errs[0]

    # 2. Frameworks fallback directory
    fw_dir = tmp_path / "frameworks"
    fw_dir.mkdir()
    hlo_file = fw_dir / "stablehlo_exhaustive.json"
    hlo_file.write_text(json.dumps({"operations": []}), encoding="utf-8")

    snaps_sub = tmp_path / "snapshots"
    snaps_sub.mkdir()
    errs_fw = verify_stablehlo_grounding(snaps_sub)
    assert isinstance(errs_fw, list)
