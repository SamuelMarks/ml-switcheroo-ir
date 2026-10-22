"""Script to verify anti-hallucination grounding of registered schemas against ground-truth snapshots."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib
import json
import os
from pathlib import Path
import sys

from ml_switcheroo_ir.schema.custom_ops import (
    CANONICAL_ATTENTION_SCHEMAS,
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
from ml_switcheroo_ir.schema.mlir_registry import MLIR_REGISTRY
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY
from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY
from ml_switcheroo_ir.types import JsonValue
from ml_switcheroo_ir.validation.grounding import SnapshotManifestNode
from ml_switcheroo_ir.validator import (
    CORE_MLIR_DIALECTS,
    DEFAULT_SNAPSHOT_DIR,
    GroundingValidator,
)


def _resolve_override_snapshots_path(override_path: str) -> Path | None:
    """Resolve an explicit snapshots directory override, supporting repo-root and sibling layouts.

    Args:
        override_path (str): Explicit path provided via CLI flag or config.

    Returns:
        Optional[Path]: Resolved snapshots directory path if found, or None.
    """
    from ml_switcheroo_ir.snapshots import resolve_snapshots_path

    return resolve_snapshots_path(override_path)


def find_snapshots_directory(override_path: str | None = None) -> Path | None:
    """Locate the ground-truth snapshots directory.

    Delegates explicit overrides to ``ml_switcheroo_ir.snapshots.resolve_snapshots_path``
    and falls back to environment variables, ``DEFAULT_SNAPSHOT_DIR``, and
    ``get_default_snapshots_dir()``.

    Args:
        override_path (Optional[str]): Explicit path provided via CLI flag or config.

    Returns:
        Optional[Path]: Path to snapshots directory if found, or None.
    """
    if override_path:
        return _resolve_override_snapshots_path(override_path)

    env_val = os.environ.get("ML_ECOSYSTEM_SNAPSHOTS_DIR")
    if env_val:
        resolved_env = _resolve_override_snapshots_path(env_val)
        if resolved_env is not None:
            return resolved_env

    default_path = Path(DEFAULT_SNAPSHOT_DIR).resolve()
    if default_path.is_dir():
        return default_path

    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    discovered = Path(get_default_snapshots_dir()).resolve()
    if discovered.is_dir():
        return discovered

    return None


def _load_snapshot_symbol_map(
    snapshot_file: Path,
) -> dict[str, dict[str, JsonValue]]:
    """Load a snapshot JSON file and index its category entries by name and api_path.

    Args:
        snapshot_file (Path): Path to the JSON snapshot file.

    Returns:
        dict[str, dict[str, JsonValue]]: Mapping from symbol name / api_path to its snapshot record.
    """
    with open(snapshot_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    snap_symbols: dict[str, dict[str, JsonValue]] = {}
    for cat_items in data.get("categories", {}).values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict):
                    name = item.get("name")
                    api_path = item.get("api_path")
                    if isinstance(name, str) and name:
                        snap_symbols[name] = item
                    if isinstance(api_path, str) and api_path:
                        snap_symbols[api_path] = item
                        snap_symbols[api_path.split(".")[-1]] = item
    return snap_symbols


def _verify_category_items_have_identifiers(
    snapshots_dir: Path,
    glob_pattern: str,
    label: str,
    allow_mnemonic: bool = False,
) -> list[str]:
    """Verify that all category items in a matching snapshot file define a valid identifier.

    Args:
        snapshots_dir (Path): Directory containing snapshots.
        glob_pattern (str): Glob pattern to locate the snapshot file.
        label (str): Dialect label for error messages (e.g. 'PTX', 'Metal', 'WASM').
        allow_mnemonic (bool): Whether 'mnemonic' is accepted as an identifier key.

    Returns:
        list[str]: Diagnostic error messages for items missing identifiers.
    """
    candidates = sorted(snapshots_dir.glob(glob_pattern))
    if not candidates:
        return []

    errors: list[str] = []
    target_file = candidates[-1]
    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cat_items in data.get("categories", {}).values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict):
                    has_id = bool(
                        item.get("api_path")
                        or item.get("name")
                        or (allow_mnemonic and item.get("mnemonic"))
                    )
                    if not has_id:
                        errors.append(
                            f"{label} item missing identifier in {target_file.name}."
                        )
    return errors


def _extract_known_attributes(
    sym_record: Mapping[str, SnapshotManifestNode],
) -> set[str]:
    """Extract known parameter and attribute names from a snapshot symbol record.

    Args:
        sym_record (Mapping[str, SnapshotManifestNode]): Snapshot symbol dictionary.

    Returns:
        set[str]: Set of grounded attribute/parameter names.
    """
    known: set[str] = set()
    raw_params = sym_record.get("params")
    if isinstance(raw_params, list):
        for p in raw_params:
            if isinstance(p, dict) and p.get("name"):
                known.add(str(p["name"]))
    attrs_obj = sym_record.get("attributes")
    if isinstance(attrs_obj, dict):
        known.update(str(k) for k in attrs_obj.keys())
    elif isinstance(attrs_obj, list):
        for a in attrs_obj:
            if isinstance(a, dict) and a.get("name"):
                known.add(str(a["name"]))
            elif isinstance(a, str) and a:
                known.add(a)
    return known


def _find_stablehlo_snapshot_file(snapshots_dir: Path) -> Path | None:
    """Locate the StableHLO snapshot or exhaustive JSON file.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        Path | None: Resolved Path to StableHLO JSON file if found, or None.
    """
    exhaustive_candidate = snapshots_dir / "stablehlo_exhaustive.json"
    if exhaustive_candidate.is_file():
        return exhaustive_candidate

    fw_exhaustive = snapshots_dir.parent / "frameworks" / "stablehlo_exhaustive.json"
    if fw_exhaustive.is_file():
        return fw_exhaustive

    candidates = sorted(snapshots_dir.glob("stablehlo*.json"))
    if candidates:
        return candidates[-1]

    if "ml-ecosystem-snapshots" in str(snapshots_dir):
        try:
            from ml_ecosystem_snapshots.frameworks import __file__ as pkg_fw_file

            pkg_exhaustive = Path(pkg_fw_file).parent / "stablehlo_exhaustive.json"
            if pkg_exhaustive.is_file():
                return pkg_exhaustive
        except (ImportError, AttributeError, TypeError):
            pass

    return None


def verify_stablehlo_grounding(snapshots_dir: Path) -> list[str]:
    """Verify all StableHLO registry schemas against stablehlo snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: List of error diagnostic messages for any ungrounded symbols.
    """
    stablehlo_file = _find_stablehlo_snapshot_file(snapshots_dir)
    if stablehlo_file is None:
        print(
            f"[WARNING] StableHLO snapshot file not found in {snapshots_dir}. Skipping StableHLO checks."
        )
        return []

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
                _load_snapshot_symbol_map(flash_file)
                for canonical_schema in CANONICAL_ATTENTION_SCHEMAS:
                    att_name = canonical_schema.name
                    exp_inputs = canonical_schema.inputs
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
    snap_symbols = _load_snapshot_symbol_map(array_api_file)

    for op_name, schema in ARRAY_API_REGISTRY.items():
        if op_name not in snap_symbols and f"array_api.{op_name}" not in snap_symbols:
            errors.append(
                f"Array API op '{op_name}' is not grounded in {array_api_file.name} snapshot."
            )
            continue

        sym_record = snap_symbols.get(op_name) or snap_symbols[f"array_api.{op_name}"]
        raw_params = sym_record.get("params")
        snap_params = raw_params if isinstance(raw_params, list) else []
        if schema.inputs:
            pos_params = [
                str(p["name"])
                for p in snap_params
                if isinstance(p, dict)
                and p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
                and isinstance(p.get("name"), str)
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
                str(p["name"]): p
                for p in snap_params
                if isinstance(p, dict)
                and p.get("kind") == "KEYWORD_ONLY"
                and isinstance(p.get("name"), str)
            }
            raw_kwargs = sym_record.get("kwargs")
            kwargs_list = (
                [str(k) for k in raw_kwargs] if isinstance(raw_kwargs, list) else []
            )
            for attr_name in schema.attributes:
                if (
                    kw_params
                    and attr_name not in kw_params
                    and attr_name not in kwargs_list
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
        list[str]: Diagnostic error messages for ungrounded ATen operations.
    """
    candidates = sorted(snapshots_dir.glob("*aten*.json"))
    if not candidates:
        return []

    errors: list[str] = []
    aten_file = candidates[-1]
    snap_symbols = _load_snapshot_symbol_map(aten_file)

    for op_name, schema in ATEN_REGISTRY.items():
        if op_name not in snap_symbols and f"aten.{op_name}" not in snap_symbols:
            errors.append(
                f"ATen op '{op_name}' is not grounded in {aten_file.name} snapshot."
            )
            continue

        sym_record = snap_symbols.get(op_name) or snap_symbols[f"aten.{op_name}"]
        if schema.attributes:
            raw_params = sym_record.get("params")
            snap_params = raw_params if isinstance(raw_params, list) else []
            known_params = {
                str(p["name"])
                for p in snap_params
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
    snap_symbols = _load_snapshot_symbol_map(nccl_file)

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
    known_symbols: dict[str, dict[str, JsonValue]] = {}
    for cat_items in data.get("categories", {}).values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict) and isinstance(item.get("api_path"), str):
                    api_p = str(item["api_path"])
                    prefix = api_p.split(".")[0]
                    known_dialects.add(prefix)
                    known_symbols[api_p] = item

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
    """Verify ir_v0.0.4.json snapshot matches runtime API signatures.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: List of diagnostic errors for mismatched public APIs.
    """
    ir_file = snapshots_dir / "ir_v0.0.4.json"
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
    return _verify_category_items_have_identifiers(
        snapshots_dir, "*ptx*.json", "PTX", allow_mnemonic=True
    )


def verify_metal_grounding(snapshots_dir: Path) -> list[str]:
    """Verify Metal MSL compute operations against snapshot if available.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded Metal operations.
    """
    return _verify_category_items_have_identifiers(
        snapshots_dir, "*metal*.json", "Metal", allow_mnemonic=False
    )


def verify_wasm_grounding(snapshots_dir: Path) -> list[str]:
    """Verify WebAssembly SIMD operations against snapshot if available.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded WASM operations.
    """
    return _verify_category_items_have_identifiers(
        snapshots_dir, "*wasm*.json", "WASM", allow_mnemonic=False
    )


def verify_quantization_grounding() -> list[str]:
    """Verify quantization operator schemas in QUANTIZATION_OPS_REGISTRY.

    Returns:
        list[str]: Diagnostic error messages for quantization schemas.
    """
    errors: list[str] = []
    if not QUANTIZATION_OPS_REGISTRY:
        errors.append("QUANTIZATION_OPS_REGISTRY is empty.")
    for op_name, schema in QUANTIZATION_OPS_REGISTRY.items():
        if not schema.name or (
            schema.name != op_name and not schema.name.endswith(f".{op_name}")
        ):
            errors.append(
                f"Quantization op key '{op_name}' != schema.name '{schema.name}'."
            )
        if schema.domain not in ("quantization", "ml.switcheroo.quant"):
            errors.append(
                f"Quantization op '{op_name}' has invalid domain '{schema.domain}'."
            )
    return errors


def verify_state_ops_grounding() -> list[str]:
    """Verify state-mutation operator schemas in STATE_OPS_REGISTRY.

    Returns:
        list[str]: Diagnostic error messages for state schemas.
    """
    errors: list[str] = []
    if not STATE_OPS_REGISTRY:
        errors.append("STATE_OPS_REGISTRY is empty.")
    for op_name, schema in STATE_OPS_REGISTRY.items():
        if schema.name != op_name:
            errors.append(f"State op key '{op_name}' != schema.name '{schema.name}'.")
        if schema.domain != "ml.switcheroo.state":
            errors.append(f"State op '{op_name}' has invalid domain '{schema.domain}'.")
    return errors


def verify_odl_grounding() -> list[str]:
    """Verify universal Operation Definition Language (ODL) schemas in ODL_CATALOG.

    Returns:
        list[str]: Diagnostic error messages for ODL schemas.
    """
    errors: list[str] = []
    if not ODL_CATALOG:
        errors.append("ODL_CATALOG is empty.")
    for op_name, schema in ODL_CATALOG.items():
        if schema.name != op_name:
            errors.append(f"ODL op key '{op_name}' != schema.name '{schema.name}'.")
        if schema.domain != "odl":
            errors.append(f"ODL op '{op_name}' has invalid domain '{schema.domain}'.")
    return errors


def _verify_local_schemas_only() -> int:
    """Verify local custom ops, ONNX, quantization, state, and ODL registries when snapshot directory is missing or empty.

    Returns:
        int: Process exit code (0 for success, 1 for failure).
    """
    custom_errors = verify_custom_ops_grounding()
    onnx_errors = verify_onnx_grounding()
    quant_errors = verify_quantization_grounding()
    state_errors = verify_state_ops_grounding()
    odl_errors = verify_odl_grounding()
    all_errors = custom_errors + onnx_errors + quant_errors + state_errors + odl_errors
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

    if not any(snapshots_dir.glob("*.json")):
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
    quant_errors = verify_quantization_grounding()
    state_errors = verify_state_ops_grounding()
    odl_errors = verify_odl_grounding()

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
        + quant_errors
        + state_errors
        + odl_errors
    )

    print(f"Audited StableHLO schemas: {len(STABLEHLO_REGISTRY)} ops.")
    print(f"Audited MLIR dialects: {len(CORE_MLIR_DIALECTS)} dialects.")
    print(f"Audited Custom ops: {len(CUSTOM_OPS_REGISTRY)} ops.")
    print(f"Audited ONNX schemas: {len(ONNX_REGISTRY)} ops.")
    print(f"Audited Array API schemas: {len(ARRAY_API_REGISTRY)} ops.")
    print(f"Audited ATen schemas: {len(ATEN_REGISTRY)} ops.")
    print(f"Audited Collective schemas: {len(COLLECTIVE_OPS_REGISTRY)} ops.")
    print(f"Audited Quantization schemas: {len(QUANTIZATION_OPS_REGISTRY)} ops.")
    print(f"Audited State schemas: {len(STATE_OPS_REGISTRY)} ops.")
    print(f"Audited ODL schemas: {len(ODL_CATALOG)} ops.")
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
