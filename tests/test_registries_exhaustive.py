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

    saved_odl_mappings = framework_registries.ODL_DIALECT_MAPPINGS.copy()
    try:
        res = framework_registries._load_framework_schema_file("custom_fw.json", "custom_fw", json_path=custom_file)
        assert "custom_add" in res
        assert "custom_no_dialects" in res
        assert framework_registries.ODL_DIALECT_MAPPINGS.get("custom_add") == {
            "torch": "torch.add",
            "jax": "jax.numpy.add",
        }
    finally:
        framework_registries.ODL_DIALECT_MAPPINGS.clear()
        framework_registries.ODL_DIALECT_MAPPINGS.update(saved_odl_mappings)


def test_framework_registries_non_empty_initialization() -> None:
    """Verify _initialize_framework_registries branch when schemas are already loaded from files."""
    from ml_switcheroo_ir.schema.base import OpSchema

    dummy_op = OpSchema(
        name="mock_op",
        domain="aten",
        version=1,
        attributes={},
        inputs=["x"],
        outputs=["y"],
    )
    with mock.patch(
        "ml_switcheroo_ir.schema.framework_registries._load_framework_schema_file",
        return_value={"mock_op": dummy_op},
    ):
        try:
            framework_registries._initialize_framework_registries()
            assert framework_registries.ATEN_REGISTRY.get("mock_op") is dummy_op
            assert framework_registries.ARRAY_API_REGISTRY.get("mock_op") is dummy_op
            assert framework_registries.ODL_CATALOG.get("mock_op") is dummy_op
        finally:
            framework_registries.ATEN_REGISTRY.pop("mock_op", None)
            framework_registries.ARRAY_API_REGISTRY.pop("mock_op", None)
            framework_registries.CHEX_REGISTRY.pop("mock_op", None)
            framework_registries.GRAIN_REGISTRY.pop("mock_op", None)
            framework_registries.ORBAX_REGISTRY.pop("mock_op", None)
            framework_registries.OPTAX_REGISTRY.pop("mock_op", None)
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

    res = low_level_registries._load_schema_file("custom_ll.json", "custom_ll", json_path=custom_file)
    assert "custom_ll_op" in res
    assert res["custom_ll_op"].domain == "custom_ll"


def test_low_level_registries_initialization_branches() -> None:
    """Verify _initialize_low_level_registries branches when loaders return empty or non-empty."""
    from ml_switcheroo_ir.schema.base import OpSchema

    dummy_op = OpSchema(
        name="mock_ll",
        domain="wgsl",
        version=1,
        attributes={},
        inputs=["src"],
        outputs=["dst"],
    )
    with mock.patch(
        "ml_switcheroo_ir.schema.low_level_registries._load_schema_file",
        side_effect=lambda *args, **kwargs: {"mock_ll": dummy_op},
    ):
        try:
            low_level_registries._initialize_low_level_registries()
            assert low_level_registries.PTX_REGISTRY.get("mock_ll") is dummy_op
            assert low_level_registries.METAL_REGISTRY.get("mock_ll") is dummy_op
            assert low_level_registries.WASM_REGISTRY.get("mock_ll") is dummy_op
            assert low_level_registries.WEBGL_REGISTRY.get("mock_ll") is dummy_op
            assert low_level_registries.WGSL_REGISTRY.get("mock_ll") is dummy_op
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
    """Verify registry loaders leave global registries intact when an explicit json_path does not exist.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    nonexistent = tmp_path / "does_not_exist.json"
    mlir_registry._load_mlir_schemas(json_path=nonexistent)
    rdna_registry._load_rdna_schemas(json_path=nonexistent)
    sass_registry._load_sass_schemas(json_path=nonexistent)
    stablehlo._load_stablehlo_schemas(json_path=nonexistent)
    assert len(mlir_registry.MLIR_REGISTRY) > 0
    assert len(rdna_registry.RDNA_REGISTRY) > 0
    assert len(sass_registry.SASS_REGISTRY) > 0
    assert len(stablehlo.STABLEHLO_REGISTRY) > 0


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

    with (
        mock.patch("builtins.open", side_effect=OSError("Read error")),
        mock.patch(
            "ml_switcheroo_ir.schema.stablehlo.find_schema_file",
            return_value=Path("/dummy/stablehlo_ops.json"),
        ),
        mock.patch("pathlib.Path.exists", return_value=True),
    ):
        stablehlo._load_stablehlo_schemas()

    # Hit the fallback branch exception
    with (
        mock.patch("builtins.open", side_effect=OSError("Read error")),
        mock.patch("ml_switcheroo_ir.schema.stablehlo.find_schema_file", return_value=None),
        mock.patch("pathlib.Path.exists", return_value=True),
    ):
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
        with (
            mock.patch(
                "builtins.open",
                mock.mock_open(read_data=json.dumps(malformed_data)),
            ),
            mock.patch("pathlib.Path.exists", return_value=True),
        ):
            stablehlo._load_stablehlo_schemas()
            assert "op_with_malformed_attr" in stablehlo.STABLEHLO_REGISTRY
            loaded_op = stablehlo.STABLEHLO_REGISTRY["op_with_malformed_attr"]
            assert "valid_attr" in loaded_op.attributes
            assert loaded_op.attributes["valid_attr"].type == "int"
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
            "schema_version": 18,
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
    assert res["CustomOp"].schema_version == 18
    assert "dict_attr" in res["CustomOp"].attributes
    assert res["CustomOp"].attributes["dict_attr"].required is True
    assert res["CustomOp"].attributes["dict_attr"].default == 42
    assert res["CustomOp"].inputs == ["x"]
    assert res["CustomOp"].outputs == ["y"]


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
    fake_onnx_spec = mock.MagicMock(_load_onnx_ops=lambda: mock_ops)
    fake_fw_pkg = mock.MagicMock(onnx_spec=fake_onnx_spec)
    fake_root_pkg = mock.MagicMock(frameworks=fake_fw_pkg)

    with (
        mock.patch.dict(
            "sys.modules",
            {
                "ml_ecosystem_snapshots": fake_root_pkg,
                "ml_ecosystem_snapshots.frameworks": fake_fw_pkg,
                "ml_ecosystem_snapshots.frameworks.onnx_spec": fake_onnx_spec,
            },
        ),
        mock.patch.dict(onnx_registry.ONNX_REGISTRY, {}, clear=True),
        mock.patch("ml_switcheroo_ir.schema.onnx_registry.find_schema_file", return_value=None),
        mock.patch("ml_switcheroo_ir.schema.onnx_registry.load_bundled_catalog_json", return_value=None),
    ):
        res = onnx_registry.load_onnx_schemas()
        assert "OpWithDictAttrs" in res
        assert "OpWithMixedListAttrs" in res
        assert "OpWithNoneAttrs" in res

    # Standalone bundled importlib.resources catalog loads when external snapshots are absent
    with (
        mock.patch.dict(onnx_registry.ONNX_REGISTRY, {}, clear=True),
        mock.patch("ml_switcheroo_ir.schema.onnx_registry.find_schema_file", return_value=None),
    ):
        bundled_res = onnx_registry.load_onnx_schemas()
        assert "Add" in bundled_res
        assert "Conv" in bundled_res

    # Exception path
    def _raise_import_err() -> list[dict[str, str]]:
        """Raise ImportError to test fallback handling.

        Returns:
            list[dict[str, str]]: Never returns.

        Raises:
            ImportError: Always raised to simulate missing upstream spec loader.
        """
        raise ImportError("Mock import failure")

    broken_onnx_spec = mock.MagicMock(_load_onnx_ops=_raise_import_err)
    with (
        mock.patch.dict(
            "sys.modules",
            {
                "ml_ecosystem_snapshots": fake_root_pkg,
                "ml_ecosystem_snapshots.frameworks": fake_fw_pkg,
                "ml_ecosystem_snapshots.frameworks.onnx_spec": broken_onnx_spec,
            },
        ),
        mock.patch.dict(onnx_registry.ONNX_REGISTRY, {}, clear=True),
        mock.patch("ml_switcheroo_ir.schema.onnx_registry.find_schema_file", return_value=None),
        mock.patch("ml_switcheroo_ir.schema.onnx_registry.load_bundled_catalog_json", return_value=None),
    ):
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
    with (
        mock.patch.dict("os.environ", {}, clear=True),
        mock.patch("os.path.expanduser", return_value=str(cache_dir)),
        mock.patch("os.path.isdir", side_effect=lambda p: str(p) == str(cache_dir)),
    ):
        resolved_cache = get_default_snapshots_dir()
        assert resolved_cache == str(cache_dir.resolve())

    env_fw_dir = tmp_path / "env_fw_snapshots"
    env_fw_dir.mkdir()
    (env_fw_dir / "sample.json").write_text("{}", encoding="utf-8")
    from ml_switcheroo_ir.snapshots import get_effective_snapshot_dir, resolve_snapshots_path

    with mock.patch.dict("os.environ", {"ML_FRAMEWORK_SNAPSHOTS_DIR": str(env_fw_dir)}):
        assert get_default_snapshots_dir() == str(env_fw_dir.resolve())
        assert get_effective_snapshot_dir() == str(env_fw_dir.resolve())

    # Hermetic repo shell with empty nested snapshots subdir honors explicit path or falls back when enabled
    empty_sibling_shell = tmp_path / "empty_ml_ecosystem_snapshots"
    empty_nested_snaps = empty_sibling_shell / "src" / "ml_ecosystem_snapshots" / "snapshots"
    empty_nested_snaps.mkdir(parents=True)
    with mock.patch.dict("os.environ", {"ML_FRAMEWORK_SNAPSHOTS_DIR": str(env_fw_dir)}, clear=True):
        assert resolve_snapshots_path(empty_sibling_shell) == empty_nested_snaps.resolve()
        assert resolve_snapshots_path(empty_sibling_shell, allow_default_fallback=True) == env_fw_dir.resolve()

    populated_sibling_shell = tmp_path / "populated_ml_ecosystem_snapshots"
    populated_nested_snaps = populated_sibling_shell / "src" / "ml_ecosystem_snapshots" / "snapshots"
    populated_nested_snaps.mkdir(parents=True)
    (populated_nested_snaps / "snap.json").write_text("{}", encoding="utf-8")
    assert (
        resolve_snapshots_path(populated_sibling_shell, allow_default_fallback=True) == populated_nested_snaps.resolve()
    )

    direct_snaps_root = tmp_path / "custom_repo"
    direct_snaps = direct_snaps_root / "snapshots"
    direct_snaps.mkdir(parents=True)
    (direct_snaps / "snap.json").write_text("{}", encoding="utf-8")
    assert resolve_snapshots_path(direct_snaps_root) == direct_snaps.resolve()


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
    assert find_schema_file("dummy.json", override_path=tmp_path / "nonexistent.json") is None

    # 3. Non-existent filename anywhere
    assert find_schema_file("completely_unknown_file_12345.json") is None

    # 4. Parent candidate via real directory hierarchy
    parent_dir = tmp_path / "snap_root"
    snap_subdir = parent_dir / "snapshots"
    snap_subdir.mkdir(parents=True)
    parent_file = parent_dir / "parent_candidate_dummy.json"
    parent_file.write_text("{}", encoding="utf-8")
    with mock.patch("ml_switcheroo_ir.snapshots.DEFAULT_SNAPSHOT_DIR", str(snap_subdir)):
        parent_res = find_schema_file("parent_candidate_dummy.json")
        assert parent_res == parent_file


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
    with mock.patch("builtins.open", side_effect=FileNotFoundError("Missing concept map")):
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
    assert len(errs) == 0

    # 2. Frameworks fallback directory
    fw_dir = tmp_path / "frameworks"
    fw_dir.mkdir()
    hlo_file = fw_dir / "stablehlo_exhaustive.json"
    hlo_file.write_text(json.dumps({"operations": []}), encoding="utf-8")

    snaps_sub = tmp_path / "snapshots"
    snaps_sub.mkdir()
    errs_fw = verify_stablehlo_grounding(snaps_sub)
    assert len(errs_fw) == len(stablehlo.STABLEHLO_REGISTRY)
    assert any("is not grounded in" in err for err in errs_fw)


def test_override_registry_context_manager() -> None:
    """Verify thread-safe override_registry context manager restores registries cleanly."""
    from ml_switcheroo_ir.schema.base import OpSchema, override_registry

    dummy_op = OpSchema(
        name="TempOp",
        domain="ai.onnx",
        version=1,
        attributes={},
        inputs=["x"],
        outputs=["y"],
    )
    orig_len = len(onnx_registry.ONNX_REGISTRY)
    with override_registry(onnx_registry.ONNX_REGISTRY, {"TempOp": dummy_op}) as reg:
        assert "TempOp" in reg
        assert len(reg) == orig_len + 1
    assert "TempOp" not in onnx_registry.ONNX_REGISTRY
    assert len(onnx_registry.ONNX_REGISTRY) == orig_len

    with override_registry(onnx_registry.ONNX_REGISTRY, clear=True) as reg:
        assert len(reg) == 0
    assert len(onnx_registry.ONNX_REGISTRY) == orig_len


def test_schema_registry_lazy_metadata_attribute_types_and_fallbacks(tmp_path: Path) -> None:
    """Exercise SchemaRegistry, LazyMetadataMap, normalize_attribute_type, MLIR resolver, and loader fallbacks.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    import pytest

    from ml_switcheroo_ir.schema import (
        AttributeType,
        LazyMetadataMap,
        OpAttribute,
        OpSchema,
        SchemaRegistry,
        normalize_attribute_type,
    )

    # AttributeType normalization branches
    assert normalize_attribute_type(AttributeType.INT) == AttributeType.INT
    assert normalize_attribute_type("AnyAttrOf<[I64Attr, StrAttr]>") == AttributeType.ANY
    assert normalize_attribute_type("ConfinedAttr<I64AttrArray, [NonEmpty]>") == AttributeType.LIST_INT
    assert normalize_attribute_type("DenseI64ArrayAttr") == AttributeType.LIST_INT
    assert normalize_attribute_type("OptionalAttr<DenseI64ArrayAttr>") == AttributeType.LIST_INT
    assert normalize_attribute_type("OptionalAttr<ArrayAttr>") == AttributeType.LIST
    assert normalize_attribute_type("ConvDimensionNumbersAttr") == AttributeType.DICT
    assert normalize_attribute_type("StableHLO_ComparisonDirectionAttr") == AttributeType.STR
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_attribute_type("   ")
    with pytest.raises(ValueError, match="Unsupported AttributeType"):
        normalize_attribute_type("CompletelyUnknownType")

    attr = OpAttribute("x", "int", False, 1)
    assert attr.to_op_attribute() is attr

    # SchemaRegistry initializers, secondary aliases, deletion, and resolve
    op_a = OpSchema("CanonOp", "ai.onnx", aliases=("canon_alias", ""))
    sr_iter = SchemaRegistry("iter_reg", initial=[op_a])
    sr_iter._ensure_loaded()
    assert sr_iter.resolve("canon_alias") is op_a
    assert sr_iter.alias_index == {"canon_alias": "CanonOp"}
    del sr_iter["canon_alias"]
    assert sr_iter.alias_index == {}
    sr_map = SchemaRegistry("map_reg", initial={"CanonOp": op_a, "SecondOp": OpSchema("SecondOp", "ai.onnx")})
    del sr_map["SecondOp"]
    assert sr_map.resolve("canon_alias") is op_a
    del sr_map["CanonOp"]
    assert len(sr_map) == 0
    with pytest.raises(KeyError):
        _ = sr_map["missing"]
    with pytest.raises(KeyError):
        del sr_map["missing"]

    # LazyMetadataMap lazy loader, set/del, copy, clear
    def _populate_lmap() -> None:
        """Populate lmap during lazy initialization."""
        lmap["k"] = 7

    lmap: LazyMetadataMap[int] = LazyMetadataMap("test_meta", loader=_populate_lmap)
    assert lmap["k"] == 7
    lmap._run_loader_locked()
    assert list(lmap) == ["k"]
    assert len(lmap) == 1
    assert lmap.copy() == {"k": 7}
    del lmap["k"]
    lmap.clear()
    assert len(lmap) == 0


def test_schema_catalog_entry_parsing_and_onnx_edge_cases(tmp_path: Path) -> None:
    """Verify parse_op_schema_entry, load_op_schema_catalog, and _parse_onnx_json_dict edge branches.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    import pydantic
    import pytest

    from ml_switcheroo_ir.schema import OpSchema, override_registry, sass_registry
    from ml_switcheroo_ir.schema.base import (
        load_bundled_catalog_json,
        load_op_schema_catalog,
        parse_op_schema_entry,
    )
    from ml_switcheroo_ir.schema.onnx_registry import _parse_onnx_json_dict, _parse_onnx_proto_default
    from ml_switcheroo_ir.schema.stablehlo import parse_stablehlo_exhaustive_operations

    # parse_op_schema_entry dict attributes, OSError in load_bundled_catalog_json, non-dict entry in ops list
    parsed_dict_attrs = parse_op_schema_entry(
        {
            "name": "DictAttrOp",
            "attributes": {"alpha": {"type": "float", "default": 1.0}, "skip_bad": "not_a_mapping"},
        },
        domain_default="ai.onnx",
    )
    assert parsed_dict_attrs.attributes["alpha"].default == 1.0

    with pytest.raises(pydantic.ValidationError):
        parse_op_schema_entry({"name": "IntAttrPayload", "attributes": 123}, domain_default="ai.onnx")
    mixed_ops_json = tmp_path / "mixed_ops.json"
    mixed_ops_json.write_text('{"ops": ["not_a_dict", {"name": "OkOp", "attributes": ["bad_attr"]}]}', encoding="utf-8")
    assert "OkOp" in load_op_schema_catalog("mixed_ops.json", "ai.onnx", json_path=mixed_ops_json)
    with mock.patch("ml_switcheroo_ir.schema.base.find_schema_file", return_value=None):
        bundled_state = load_op_schema_catalog("state_ops.json", "ml.switcheroo.state")
        assert "ReadVariable" in bundled_state
    with mock.patch("importlib.resources.files", side_effect=OSError("bad pkg")):
        assert load_bundled_catalog_json("onnx_ops.json") is None

    # onnx_registry float default & _parse_onnx_json_dict edge branches & non-mapping target json
    assert _parse_onnx_proto_default("f: 0.25") == 0.25
    parsed_onnx_edge = _parse_onnx_json_dict(
        {
            "bad_entry": "not_a_dict",
            "EdgeOp": {"attributes": {"ok": {"type": "int"}, "bad": 5}, "inputs": [], "outputs": []},
            "NonMapAttrs": {"attributes": [], "inputs": ["x"], "outputs": ["y"]},
        }
    )
    assert "EdgeOp" in parsed_onnx_edge and "NonMapAttrs" in parsed_onnx_edge
    list_onnx_json = tmp_path / "list_onnx.json"
    list_onnx_json.write_text("[1, 2]", encoding="utf-8")
    assert onnx_registry.load_onnx_schemas(json_path=list_onnx_json) == {}

    # stablehlo convolution dimension_numbers & sass latency fallback
    shlo_conv = parse_stablehlo_exhaustive_operations(
        [{"name": "convolution", "attributes": [{"name": "dimension_numbers", "type": "str", "required": False}]}]
    )
    assert shlo_conv["convolution"].attributes["dimension_numbers"].required is True
    sass_alt_json = tmp_path / "sass_alt.json"
    sass_alt_json.write_text('{"ops": [{"name": "FADD_ALT", "latency": 5}]}', encoding="utf-8")
    saved_sass_lat = sass_registry.SASS_PIPELINE_LATENCIES.copy()
    try:
        with override_registry(sass_registry.SASS_REGISTRY, clear=True):
            sass_registry._load_sass_schemas(json_path=sass_alt_json)
            assert sass_registry.SASS_PIPELINE_LATENCIES["FADD_ALT"] == 5
    finally:
        sass_registry.SASS_PIPELINE_LATENCIES.clear()
        sass_registry.SASS_PIPELINE_LATENCIES.update(saved_sass_lat)
    _ = OpSchema


def test_unified_mlir_resolution_and_missing_catalogs(tmp_path: Path) -> None:
    """Verify resolve_mlir_op_schema, build_core_mlir_dialects, and missing explicit catalog paths.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    from ml_switcheroo_ir.schema import (
        OpSchema,
        SchemaRegistry,
        build_core_mlir_dialects,
        custom_ops,
        mlir_registry,
        override_registry,
        resolve_mlir_op_schema,
    )
    from ml_switcheroo_ir.schema.base import load_bundled_catalog_json, load_op_schema_catalog

    missing_p = tmp_path / "missing_catalog.json"
    custom_ops._load_collective_schemas(json_path=missing_p)
    custom_ops._load_quantization_schemas(json_path=missing_p)
    custom_ops._load_custom_schemas(json_path=missing_p)
    custom_ops._load_state_schemas(json_path=missing_p)

    # Unified MLIR resolution when direct op has mismatched domain and build_core_mlir_dialects unknown dialect
    custom_mlir = {"arith.addf": OpSchema("arith.addf", "other_dom"), "unk.op": OpSchema("unk.op", "mlir.unk")}
    assert resolve_mlir_op_schema("mlir.arith", "arith.addf", registry=custom_mlir) is None
    assert build_core_mlir_dialects(registry=custom_mlir)["arith"] == {"addf": 0}
    assert build_core_mlir_dialects(registry=custom_mlir)["math"] == {}
    SchemaRegistry("noop_reg", initial=[OpSchema("Op", "ai.onnx")])._run_loader_locked()

    # Bundled catalog loader missing file branch & non-mapping catalog branch
    assert load_bundled_catalog_json("does_not_exist_catalog.json") is None
    list_json = tmp_path / "list_root.json"
    list_json.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_op_schema_catalog("list_root.json", "ai.onnx", json_path=list_json) == {}
    bad_ops_json = tmp_path / "bad_ops.json"
    bad_ops_json.write_text('{"ops": "not_a_list"}', encoding="utf-8")
    assert load_op_schema_catalog("bad_ops.json", "ai.onnx", json_path=bad_ops_json) == {}

    # Unified MLIR resolution across mlir.<dialect>, mlir, and short <dialect> conventions
    assert resolve_mlir_op_schema("mlir.arith", "arith.addf") is not None
    assert resolve_mlir_op_schema("mlir.arith", "addf") is not None
    assert resolve_mlir_op_schema("mlir.arith", "nonexistent") is None
    assert resolve_mlir_op_schema("arith", "addf") is not None
    assert resolve_mlir_op_schema("unknown_dom", "addf") is None
    with override_registry(mlir_registry.MLIR_REGISTRY, {"nodot": OpSchema("nodot", "mlir")}):
        dialects = build_core_mlir_dialects()
        assert dialects["arith"]["addf"] == 2


def test_canonical_in_memory_registry_fallbacks() -> None:
    """Exercise canonical in-memory fallbacks when external and bundled JSON catalogs are absent."""
    from ml_switcheroo_ir.schema import (
        custom_ops,
        mlir_registry,
        override_registry,
        rdna_registry,
        sass_registry,
    )

    # Exercise canonical in-memory fallbacks when external & bundled JSON catalogs are mocked out
    with (
        mock.patch("ml_switcheroo_ir.schema.base.find_schema_file", return_value=None),
        mock.patch("ml_switcheroo_ir.schema.base.load_bundled_catalog_json", return_value=None),
        mock.patch("ml_switcheroo_ir.schema.rdna_registry.find_schema_file", return_value=None),
        mock.patch("ml_switcheroo_ir.schema.sass_registry.find_schema_file", return_value=None),
        mock.patch("ml_switcheroo_ir.schema.mlir_registry.find_schema_file", return_value=None),
        mock.patch("ml_switcheroo_ir.schema.framework_registries.find_schema_file", return_value=None),
    ):
        with override_registry(custom_ops.COLLECTIVE_OPS_REGISTRY, clear=True):
            custom_ops._load_collective_schemas()
            assert "collective.all_reduce" in custom_ops.COLLECTIVE_OPS_REGISTRY
        with override_registry(custom_ops.QUANTIZATION_OPS_REGISTRY, clear=True):
            custom_ops._load_quantization_schemas()
            assert "quantization.block_quantize" in custom_ops.QUANTIZATION_OPS_REGISTRY
        with override_registry(custom_ops.CUSTOM_OPS_REGISTRY, clear=True):
            custom_ops._load_custom_schemas()
            assert "FlashAttention" in custom_ops.CUSTOM_OPS_REGISTRY
        with override_registry(custom_ops.STATE_OPS_REGISTRY, clear=True):
            custom_ops._load_state_schemas()
            assert "ReadVariable" in custom_ops.STATE_OPS_REGISTRY
        with override_registry(mlir_registry.MLIR_REGISTRY, clear=True):
            mlir_registry._load_mlir_schemas()
            assert "arith.addf" in mlir_registry.MLIR_REGISTRY
        saved_slots = rdna_registry.RDNA_VOPD_SLOTS.copy()
        saved_vmap = rdna_registry.RDNA_TO_VOPD_MAP.copy()
        try:
            with override_registry(rdna_registry.RDNA_REGISTRY, clear=True):
                rdna_registry._load_rdna_schemas()
                assert "V_DUAL_FMAC_F32" in rdna_registry.RDNA_REGISTRY
        finally:
            rdna_registry.RDNA_VOPD_SLOTS.clear()
            rdna_registry.RDNA_VOPD_SLOTS.update(saved_slots)
            rdna_registry.RDNA_TO_VOPD_MAP.clear()
            rdna_registry.RDNA_TO_VOPD_MAP.update(saved_vmap)
        saved_lat = sass_registry.SASS_PIPELINE_LATENCIES.copy()
        try:
            with override_registry(sass_registry.SASS_REGISTRY, clear=True):
                sass_registry._load_sass_schemas()
                assert "HMMA" in sass_registry.SASS_REGISTRY
        finally:
            sass_registry.SASS_PIPELINE_LATENCIES.clear()
            sass_registry.SASS_PIPELINE_LATENCIES.update(saved_lat)
        saved_odl_map = framework_registries.ODL_DIALECT_MAPPINGS.copy()
        try:
            with override_registry(framework_registries.ODL_CATALOG, clear=True):
                framework_registries.ODL_DIALECT_MAPPINGS.clear()
                framework_registries._initialize_framework_registries()
                assert "matmul" in framework_registries.ODL_DIALECT_MAPPINGS
        finally:
            framework_registries.ODL_DIALECT_MAPPINGS.clear()
            framework_registries.ODL_DIALECT_MAPPINGS.update(saved_odl_map)
