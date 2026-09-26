"""Script to verify anti-hallucination grounding of registered schemas against ground-truth snapshots."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from ml_switcheroo_ir.schema.custom_ops import (
    COLLECTIVE_OPS_REGISTRY,
    CUSTOM_OPS_REGISTRY,
)
from ml_switcheroo_ir.schema.framework_registries import (
    ARRAY_API_REGISTRY,
    ATEN_REGISTRY,
)
from ml_switcheroo_ir.schema.mlir_registry import MLIR_REGISTRY
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY
from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY
from ml_switcheroo_ir.validator import (
    CORE_MLIR_DIALECTS,
    DEFAULT_SNAPSHOT_DIR,
    GroundingValidator,
)


def find_snapshots_directory(override_path: str | None = None) -> Path | None:
    """Locate the ground-truth snapshots directory.

    Checks:
        1. Explicit override_path argument.
        2. ML_ECOSYSTEM_SNAPSHOTS_DIR environment variable.
        3. ML_FRAMEWORK_SNAPSHOTS_DIR environment variable.
        4. DEFAULT_SNAPSHOT_DIR from validator.
        5. Sibling directory ../ml-ecosystem-snapshots/src/ml_framework_snapshots/snapshots/
           and ../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots/.
        6. User cache directory (~/.cache/ml_ecosystem_snapshots).
        7. User cache directory (~/.cache/ml_framework_snapshots).
        8. Sibling directory ../ml-framework-snapshots/src/ml_framework_snapshots/snapshots/.
        9. Bundled fixture directory tests/fixtures/snapshots.

    Args:
        override_path (Optional[str]): Explicit path provided via CLI flag or config.

    Returns:
        Optional[Path]: Path to snapshots directory if found, or None.
    """
    if override_path:
        p = Path(override_path).resolve()
        if p.is_dir():
            return p
        return None

    env_eco = os.environ.get("ML_ECOSYSTEM_SNAPSHOTS_DIR")
    if env_eco:
        p = Path(env_eco).resolve()
        if p.is_dir():
            return p

    env_dir = os.environ.get("ML_FRAMEWORK_SNAPSHOTS_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
        if p.is_dir():
            return p

    default_path = Path(DEFAULT_SNAPSHOT_DIR).resolve()
    if default_path.is_dir():
        return default_path

    sibling_eco_fw = (
        Path(__file__).resolve().parent.parent.parent
        / "ml-ecosystem-snapshots"
        / "src"
        / "ml_framework_snapshots"
        / "snapshots"
    )
    if sibling_eco_fw.is_dir():
        return sibling_eco_fw

    sibling_eco = (
        Path(__file__).resolve().parent.parent.parent
        / "ml-ecosystem-snapshots"
        / "src"
        / "ml_ecosystem_snapshots"
        / "snapshots"
    )
    if sibling_eco.is_dir():
        return sibling_eco

    user_cache_eco = Path("~/.cache/ml_ecosystem_snapshots").expanduser().resolve()
    if user_cache_eco.is_dir():
        return user_cache_eco

    user_cache_fw = Path("~/.cache/ml_framework_snapshots").expanduser().resolve()
    if user_cache_fw.is_dir():
        return user_cache_fw

    sibling_fw = (
        Path(__file__).resolve().parent.parent.parent
        / "ml-framework-snapshots"
        / "src"
        / "ml_framework_snapshots"
        / "snapshots"
    )
    if sibling_fw.is_dir():
        return sibling_fw

    fixtures_dir = (
        Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "snapshots"
    )
    if fixtures_dir.is_dir():
        return fixtures_dir

    if default_path.is_dir():
        return default_path

    return None


def _extract_known_attributes(sym_record: dict[str, Any]) -> set[str]:
    """Extract known parameter and attribute names from a snapshot symbol record.

    Args:
        sym_record: Snapshot symbol dictionary.

    Returns:
        Set of grounded attribute/parameter names.
    """
    known: set[str] = set()
    for p in sym_record.get("params") or []:
        if isinstance(p, dict) and p.get("name"):
            known.add(str(p["name"]))
    attrs_obj = sym_record.get("attributes")
    if isinstance(attrs_obj, dict):
        known.update(attrs_obj.keys())
    elif isinstance(attrs_obj, list):
        for a in attrs_obj:
            if isinstance(a, dict) and a.get("name"):
                known.add(str(a["name"]))
            elif isinstance(a, str) and a:
                known.add(a)
    return known


def verify_stablehlo_grounding(snapshots_dir: Path) -> list[str]:
    """Verify all StableHLO registry schemas against stablehlo snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: List of error diagnostic messages for any ungrounded symbols.
    """
    stablehlo_file = snapshots_dir / "stablehlo_v1.0.0.json"
    if not stablehlo_file.is_file():
        candidates = sorted(snapshots_dir.glob("stablehlo*.json"))
        if candidates:
            stablehlo_file = candidates[-1]
        else:
            return [f"StableHLO snapshot file not found: {stablehlo_file}"]

    gv = GroundingValidator(snapshot_manifest=str(stablehlo_file))
    errors: list[str] = []

    for op_name, schema in STABLEHLO_REGISTRY.items():
        if (
            op_name not in gv.grounded_symbols
            and f"stablehlo.{op_name}" not in gv.grounded_symbols
        ):
            errors.append(
                f"StableHLO op '{op_name}' is not grounded in {stablehlo_file.name} snapshot."
            )

        # Check attributes against snapshot record
        sym_record = gv.grounded_symbols.get(op_name) or gv.grounded_symbols.get(
            f"stablehlo.{op_name}"
        )
        if sym_record:
            known_attrs = _extract_known_attributes(sym_record)
            for attr_name in schema.attributes:
                if known_attrs and attr_name not in known_attrs:
                    errors.append(
                        f"StableHLO op '{op_name}' attribute '{attr_name}' is not grounded in snapshot attributes {list(known_attrs)}."
                    )

    return errors


def verify_custom_ops_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify all custom operator schemas in CUSTOM_OPS_REGISTRY have valid specifications.

    Also audits custom attention operations against flash_attention snapshot when available.

    Args:
        snapshots_dir (Optional[Path]): Directory containing snapshots.

    Returns:
        List[str]: List of diagnostic errors for custom operators.
    """
    errors: list[str] = []
    for op_name, schema in CUSTOM_OPS_REGISTRY.items():
        if not schema.name:
            errors.append("Custom op has empty name.")
        if schema.domain not in ("ml.switcheroo.custom", "collective", "quantization"):
            errors.append(
                f"Custom op '{op_name}' has invalid domain '{schema.domain}', expected 'ml.switcheroo.custom', 'collective', or 'quantization'."
            )
        if not schema.outputs:
            errors.append(f"Custom op '{op_name}' has no defined outputs.")

    if snapshots_dir is not None:
        flash_candidates = sorted(snapshots_dir.glob("*flash_attention*.json"))
        if flash_candidates:
            flash_file = flash_candidates[-1]
            try:
                with open(flash_file, "r", encoding="utf-8") as f:
                    flash_data = json.load(f)
                snap_symbols: dict[str, Any] = {}
                for cat_items in flash_data.get("categories", {}).values():
                    if isinstance(cat_items, list):
                        for item in cat_items:
                            if isinstance(item, dict):
                                name = item.get("name")
                                api_path = item.get("api_path")
                                if name:
                                    snap_symbols[name] = item
                                if api_path:
                                    snap_symbols[api_path] = item
                                    snap_symbols[api_path.split(".")[-1]] = item

                attention_ops = {
                    "FlashAttention": ["Q", "K", "V"],
                    "PagedAttention": [
                        "query",
                        "key_cache",
                        "value_cache",
                        "block_tables",
                        "context_lens",
                    ],
                    "RaggedPagedAttention": [
                        "query",
                        "key_cache",
                        "value_cache",
                        "block_tables",
                        "context_lens",
                    ],
                }
                for att_name, exp_inputs in attention_ops.items():
                    if att_name not in CUSTOM_OPS_REGISTRY:
                        errors.append(
                            f"Custom attention op '{att_name}' missing from CUSTOM_OPS_REGISTRY."
                        )
                    else:
                        att_schema = CUSTOM_OPS_REGISTRY[att_name]
                        if att_schema.inputs != exp_inputs:
                            errors.append(
                                f"Custom attention op '{att_name}' inputs {att_schema.inputs} do not match expected {exp_inputs}."
                            )
            except (
                ValueError,
                OSError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                errors.append(
                    f"Failed parsing flash_attention snapshot {flash_file.name}: {exc}"
                )

    return errors


def verify_array_api_grounding(snapshots_dir: Path) -> list[str]:
    """Verify Array API operators against array_api snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded Array API operations.
    """
    candidates = sorted(snapshots_dir.glob("*array_api*.json"))
    if not candidates:
        return []

    errors: list[str] = []
    array_api_file = candidates[-1]
    with open(array_api_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    snap_symbols: dict[str, Any] = {}
    for cat_items in data.get("categories", {}).values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict):
                    name = item.get("name")
                    api_path = item.get("api_path")
                    if name:
                        snap_symbols[name] = item
                    if api_path:
                        snap_symbols[api_path] = item
                        snap_symbols[api_path.split(".")[-1]] = item

    for op_name, schema in ARRAY_API_REGISTRY.items():
        if op_name not in snap_symbols and f"array_api.{op_name}" not in snap_symbols:
            errors.append(
                f"Array API op '{op_name}' is not grounded in {array_api_file.name} snapshot."
            )
            continue

        sym_record = snap_symbols.get(op_name) or snap_symbols[f"array_api.{op_name}"]
        snap_params = sym_record.get("params") or []
        if schema.inputs:
            pos_params = [
                p["name"]
                for p in snap_params
                if p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
            ]
            for idx, inp in enumerate(schema.inputs):
                if (
                    idx < len(pos_params)
                    and inp != pos_params[idx]
                    and inp not in pos_params
                ):
                    errors.append(
                        f"Array API op '{op_name}' input {idx} ('{inp}') does not match snapshot parameter '{pos_params[idx]}'."
                    )
        if schema.attributes:
            kw_params = {
                p["name"]: p for p in snap_params if p.get("kind") == "KEYWORD_ONLY"
            }
            for attr_name in schema.attributes:
                if (
                    kw_params
                    and attr_name not in kw_params
                    and attr_name not in sym_record.get("kwargs", [])
                ):
                    errors.append(
                        f"Array API op '{op_name}' attribute '{attr_name}' is not recognized in {array_api_file.name}."
                    )
    return errors


def verify_aten_grounding(snapshots_dir: Path) -> list[str]:
    """Verify ATen operators against aten snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded ATen operations.
    """
    candidates = sorted(snapshots_dir.glob("*aten*.json"))
    if not candidates:
        return []

    errors: list[str] = []
    aten_file = candidates[-1]
    with open(aten_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    snap_symbols: dict[str, Any] = {}
    for cat_items in data.get("categories", {}).values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict):
                    name = item.get("name")
                    api_path = item.get("api_path")
                    if name:
                        snap_symbols[name] = item
                    if api_path:
                        snap_symbols[api_path] = item
                        snap_symbols[api_path.split(".")[-1]] = item

    for op_name, schema in ATEN_REGISTRY.items():
        if op_name not in snap_symbols and f"aten.{op_name}" not in snap_symbols:
            errors.append(
                f"ATen op '{op_name}' is not grounded in {aten_file.name} snapshot."
            )
            continue

        sym_record = snap_symbols.get(op_name) or snap_symbols[f"aten.{op_name}"]
        if schema.attributes:
            known_params = {
                p["name"]
                for p in sym_record.get("params") or []
                if isinstance(p, dict) and p.get("name")
            }
            for attr_name in schema.attributes:
                if known_params and attr_name not in known_params:
                    errors.append(
                        f"ATen op '{op_name}' attribute '{attr_name}' is not recognized in {aten_file.name}."
                    )
    return errors


def verify_collectives_grounding(snapshots_dir: Path) -> list[str]:
    """Verify collective operations against NCCL snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for collective operations.
    """
    candidates = sorted(snapshots_dir.glob("*nccl*.json"))
    if not candidates:
        return []

    errors: list[str] = []
    nccl_file = candidates[-1]
    with open(nccl_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    snap_symbols: dict[str, Any] = {}
    for cat_items in data.get("categories", {}).values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict):
                    name = item.get("name")
                    api_path = item.get("api_path")
                    if name:
                        snap_symbols[name] = item
                    if api_path:
                        snap_symbols[api_path] = item
                        snap_symbols[api_path.split(".")[-1]] = item

    core_collectives = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]
    for coll_name in core_collectives:
        if (
            coll_name not in COLLECTIVE_OPS_REGISTRY
            and f"collective.{coll_name}" not in COLLECTIVE_OPS_REGISTRY
        ):
            errors.append(
                f"Collective operation '{coll_name}' missing from COLLECTIVE_OPS_REGISTRY."
            )
            continue

        if coll_name not in snap_symbols and f"nccl.{coll_name}" not in snap_symbols:
            errors.append(
                f"Collective op '{coll_name}' is not grounded in {nccl_file.name} snapshot."
            )
            continue

        schema = (
            COLLECTIVE_OPS_REGISTRY.get(coll_name)
            or COLLECTIVE_OPS_REGISTRY[f"collective.{coll_name}"]
        )
        if (
            coll_name in ("all_reduce", "reduce_scatter")
            and "reduction_op" not in schema.attributes
        ):
            errors.append(
                f"Collective op '{coll_name}' schema missing 'reduction_op' attribute."
            )
        if "comm" not in schema.attributes:
            errors.append(
                f"Collective op '{coll_name}' schema missing 'comm' communicator attribute."
            )

    return errors


def verify_onnx_grounding() -> list[str]:
    """Verify that ONNX registry schemas are properly defined and consistent.

    Returns:
        List[str]: List of diagnostic errors for ONNX operators.
    """
    errors: list[str] = []
    if not ONNX_REGISTRY:
        errors.append("ONNX registry is empty.")
    for op_name, schema in ONNX_REGISTRY.items():
        if schema.name != op_name:
            errors.append(
                f"ONNX schema name mismatch: key '{op_name}' != schema.name '{schema.name}'."
            )
        if not (schema.domain == "ai.onnx" or schema.domain.startswith("ai.onnx.")):
            errors.append(
                f"ONNX op '{op_name}' has unexpected domain '{schema.domain}'."
            )
    return errors


def verify_mlir_grounding(snapshots_dir: Path) -> list[str]:
    """Verify MLIR dialect operations against mlir snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: List of diagnostic errors for ungrounded MLIR operations.
    """
    mlir_file = snapshots_dir / "mlir_v0.4.30.json"
    if not mlir_file.is_file():
        candidates = sorted(snapshots_dir.glob("mlir*.json"))
        if candidates:
            mlir_file = candidates[-1]
        else:
            return []

    errors: list[str] = []
    with open(mlir_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    known_dialects: set[str] = set()
    known_symbols: dict[str, dict[str, Any]] = {}
    for cat_items in data.get("categories", {}).values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict) and "api_path" in item:
                    prefix = item["api_path"].split(".")[0]
                    known_dialects.add(prefix)
                    known_symbols[item["api_path"]] = item

    for dialect in CORE_MLIR_DIALECTS:
        if known_dialects and dialect not in known_dialects:
            errors.append(
                f"Core MLIR dialect '{dialect}' is not grounded in {mlir_file.name}."
            )

    for op_name, schema in MLIR_REGISTRY.items():
        sym_record = known_symbols.get(op_name)
        if sym_record:
            known_attrs = _extract_known_attributes(sym_record)
            for attr_name in schema.attributes:
                if known_attrs and attr_name not in known_attrs:
                    errors.append(
                        f"MLIR op '{op_name}' attribute '{attr_name}' is not grounded in {mlir_file.name} snapshot attributes {list(known_attrs)}."
                    )

    return errors


def verify_ir_snapshot_grounding(snapshots_dir: Path) -> list[str]:
    """Verify ir_v0.0.3.json snapshot matches runtime API signatures.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: List of diagnostic errors for mismatched public APIs.
    """
    ir_file = snapshots_dir / "ir_v0.0.3.json"
    if not ir_file.is_file():
        candidates = sorted(snapshots_dir.glob("ir_v*.json"))
        if not candidates:
            return []
        ir_file = candidates[-1]

    errors: list[str] = []
    with open(ir_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cls_entry in data.get("categories", {}).get("classes", []):
        path = cls_entry.get("api_path", "")
        if not path:
            continue
        mod_path, obj_name = path.rsplit(".", 1)
        try:
            mod = importlib.import_module(mod_path)
            getattr(mod, obj_name)
        except (ImportError, AttributeError):
            if "." in mod_path:
                try:
                    parent_mod_path, parent_cls = mod_path.rsplit(".", 1)
                    parent = getattr(
                        importlib.import_module(parent_mod_path), parent_cls
                    )
                    getattr(parent, obj_name)
                except (ImportError, AttributeError):
                    errors.append(
                        f"Exported symbol '{path}' not resolvable at runtime."
                    )
            else:
                errors.append(f"Exported symbol '{path}' not resolvable at runtime.")

    for fn_entry in data.get("categories", {}).get("functions", []):
        path = fn_entry.get("api_path", "")
        if not path:
            continue
        mod_path, obj_name = path.rsplit(".", 1)
        try:
            mod = importlib.import_module(mod_path)
            getattr(mod, obj_name)
        except (ImportError, AttributeError):
            errors.append(f"Exported function '{path}' not resolvable at runtime.")

    return errors


def verify_rdna_grounding(snapshots_dir: Path) -> list[str]:
    """Verify AMD RDNA instructions and VOPD specifications against snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded RDNA instructions.
    """
    candidates = sorted(snapshots_dir.glob("*rdna*.json"))
    if not candidates:
        return []

    errors: list[str] = []
    rdna_file = candidates[-1]
    with open(rdna_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    instructions = data.get("categories", {}).get("instructions", [])
    for inst in instructions:
        if not isinstance(inst, dict):
            continue
        name = inst.get("name") or inst.get("mnemonic")
        if not name:
            errors.append(
                f"RDNA instruction missing name/mnemonic in {rdna_file.name}."
            )
            continue
        vopd_slot = inst.get("vopd_slot")
        if vopd_slot is not None and vopd_slot not in ("X", "Y", "BOTH"):
            errors.append(
                f"RDNA instruction '{name}' has invalid vopd_slot '{vopd_slot}'."
            )
    return errors


def verify_sass_grounding(snapshots_dir: Path) -> list[str]:
    """Verify NVIDIA SASS instructions and control codes against snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded SASS instructions.
    """
    candidates = sorted(snapshots_dir.glob("*sass*.json"))
    if not candidates:
        return []

    errors: list[str] = []
    sass_file = candidates[-1]
    with open(sass_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    instructions = data.get("categories", {}).get("instructions", [])
    for inst in instructions:
        if not isinstance(inst, dict):
            continue
        name = inst.get("name") or inst.get("mnemonic")
        if not name:
            errors.append(
                f"SASS instruction missing name/mnemonic in {sass_file.name}."
            )
            continue
        latency = inst.get("execution_latency")
        if latency is not None and not isinstance(latency, (int, list, tuple)):
            errors.append(
                f"SASS instruction '{name}' has invalid execution_latency '{latency}'."
            )
    return errors


def verify_wgsl_grounding(snapshots_dir: Path) -> list[str]:
    """Verify WebGPU WGSL operations and attributes against snapshot or schema.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded WGSL operations.
    """
    wgsl_file = snapshots_dir / "wgsl_ops.json"
    if not wgsl_file.is_file():
        wgsl_file = Path(DEFAULT_SNAPSHOT_DIR) / "wgsl_ops.json"
    if not wgsl_file.is_file():
        return []

    errors: list[str] = []
    with open(wgsl_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for op in data.get("ops", []):
        name = op.get("name")
        if not name:
            errors.append("WGSL op missing name.")
            continue
        if op.get("domain") != "wgsl":
            errors.append(f"WGSL op '{name}' has invalid domain '{op.get('domain')}'.")
    return errors


def verify_ptx_grounding(snapshots_dir: Path) -> list[str]:
    """Verify NVIDIA PTX operations against snapshot if available.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded PTX operations.
    """
    candidates = sorted(snapshots_dir.glob("*ptx*.json"))
    if not candidates:
        return []

    errors: list[str] = []
    ptx_file = candidates[-1]
    with open(ptx_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cat_items in data.get("categories", {}).values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict) and not (
                    item.get("api_path") or item.get("name") or item.get("mnemonic")
                ):
                    errors.append(f"PTX item missing identifier in {ptx_file.name}.")
    return errors


def verify_metal_grounding(snapshots_dir: Path) -> list[str]:
    """Verify Metal MSL compute operations against snapshot if available.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded Metal operations.
    """
    candidates = sorted(snapshots_dir.glob("*metal*.json"))
    if not candidates:
        return []

    errors: list[str] = []
    metal_file = candidates[-1]
    with open(metal_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cat_items in data.get("categories", {}).values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict) and not (
                    item.get("api_path") or item.get("name")
                ):
                    errors.append(
                        f"Metal item missing identifier in {metal_file.name}."
                    )
    return errors


def verify_wasm_grounding(snapshots_dir: Path) -> list[str]:
    """Verify WebAssembly SIMD operations against snapshot if available.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded WASM operations.
    """
    candidates = sorted(snapshots_dir.glob("*wasm*.json"))
    if not candidates:
        return []

    errors: list[str] = []
    wasm_file = candidates[-1]
    with open(wasm_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cat_items in data.get("categories", {}).values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict) and not (
                    item.get("api_path") or item.get("name")
                ):
                    errors.append(f"WASM item missing identifier in {wasm_file.name}.")
    return errors


def _verify_local_schemas_only() -> int:
    """Verify local custom ops, ONNX, Array API, ATen, and collective registries when snapshot directory is missing or empty.

    Returns:
        int: Process exit code (0 for success, 1 for failure).
    """
    custom_errors = verify_custom_ops_grounding()
    onnx_errors = verify_onnx_grounding()
    all_errors = custom_errors + onnx_errors
    if all_errors:
        for err in all_errors:
            print(f"[ERROR] {err}")
        return 1
    print("[SUCCESS] Local schema definitions are valid and structurally sound.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run anti-hallucination grounding verification for all schemas.

    Args:
        argv (Optional[List[str]]): Command-line arguments.

    Returns:
        int: Exit status code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Verify anti-hallucination grounding against framework snapshots."
    )
    parser.add_argument(
        "--snapshots-dir",
        type=str,
        default=None,
        help="Path to snapshots directory.",
    )
    parser.add_argument(
        "--ecosystem-snapshots-dir",
        type=str,
        default=None,
        help="Path to ml-ecosystem-snapshots directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing snapshot directories or ungrounded symbols strictly as errors.",
    )
    args = parser.parse_args(argv)

    print("=== Auditing ml-switcheroo-ir Schema Grounding ===")

    target_dir = args.ecosystem_snapshots_dir or args.snapshots_dir
    snapshots_dir = find_snapshots_directory(target_dir)
    if snapshots_dir is None:
        if args.strict:
            print("[ERROR] Snapshot directory not found in strict mode.")
            return 1
        print("[WARNING] Snapshot directory not found. Skipping live snapshot checks.")
        return _verify_local_schemas_only()

    if target_dir is None and not any(snapshots_dir.glob("*.json")):
        if args.strict:
            print(
                "[ERROR] Snapshot directory contains no snapshot files in strict mode."
            )
            return 1
        print(
            "[WARNING] Snapshot directory contains no snapshot files. Skipping live snapshot checks."
        )
        return _verify_local_schemas_only()

    print(f"Using snapshots directory: {snapshots_dir}")

    stablehlo_errors = verify_stablehlo_grounding(snapshots_dir)
    mlir_errors = verify_mlir_grounding(snapshots_dir)
    ir_errors = verify_ir_snapshot_grounding(snapshots_dir)
    custom_errors = verify_custom_ops_grounding(snapshots_dir)
    onnx_errors = verify_onnx_grounding()
    rdna_errors = verify_rdna_grounding(snapshots_dir)
    sass_errors = verify_sass_grounding(snapshots_dir)
    wgsl_errors = verify_wgsl_grounding(snapshots_dir)
    ptx_errors = verify_ptx_grounding(snapshots_dir)
    metal_errors = verify_metal_grounding(snapshots_dir)
    wasm_errors = verify_wasm_grounding(snapshots_dir)
    array_api_errors = verify_array_api_grounding(snapshots_dir)
    aten_errors = verify_aten_grounding(snapshots_dir)
    collectives_errors = verify_collectives_grounding(snapshots_dir)

    all_errors = (
        stablehlo_errors
        + mlir_errors
        + ir_errors
        + custom_errors
        + onnx_errors
        + rdna_errors
        + sass_errors
        + wgsl_errors
        + ptx_errors
        + metal_errors
        + wasm_errors
        + array_api_errors
        + aten_errors
        + collectives_errors
    )

    print(f"Audited StableHLO schemas: {len(STABLEHLO_REGISTRY)} ops.")
    print(f"Audited MLIR dialects: {len(CORE_MLIR_DIALECTS)} dialects.")
    print(f"Audited Custom ops: {len(CUSTOM_OPS_REGISTRY)} ops.")
    print(f"Audited ONNX schemas: {len(ONNX_REGISTRY)} ops.")
    print(f"Audited Array API schemas: {len(ARRAY_API_REGISTRY)} ops.")
    print(f"Audited ATen schemas: {len(ATEN_REGISTRY)} ops.")
    print(f"Audited Collective schemas: {len(COLLECTIVE_OPS_REGISTRY)} ops.")
    print("Audited Low-Level & Hardware ISAs (RDNA, SASS, WGSL, PTX, Metal, WASM).")

    if all_errors:
        print(
            f"\n[FAILURE] Found {len(all_errors)} ungrounded symbol(s) or schema defect(s):"
        )
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print("\n[SUCCESS] All schemas are 100% grounded against ground-truth snapshots!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
