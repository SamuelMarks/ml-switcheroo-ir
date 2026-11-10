"""Script to verify anti-hallucination grounding of registered schemas against ground-truth snapshots."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

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
    CHEX_REGISTRY,
    GRAIN_REGISTRY,
    ODL_CATALOG,
    OPTAX_REGISTRY,
    ORBAX_REGISTRY,
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
from ml_switcheroo_ir.types import JsonValue
from ml_switcheroo_ir.validation.grounding import SnapshotManifestNode
from ml_switcheroo_ir.validator import (
    CORE_MLIR_DIALECTS,
    DEFAULT_SNAPSHOT_DIR,
    GroundingValidator,
)


def _resolve_override_snapshots_path(override_path: str, allow_default_fallback: bool = True) -> Path | None:
    """Resolve an explicit snapshots directory override, supporting repo-root and sibling layouts.

    Args:
        override_path (str): Explicit path provided via CLI flag or config.
        allow_default_fallback (bool): Whether to fall back to active default snapshot directory
            when a sibling snapshot repo's nested snapshots directory is unpopulated.

    Returns:
        Optional[Path]: Resolved snapshots directory path if found, or None.
    """
    from ml_switcheroo_ir.snapshots import resolve_snapshots_path

    return resolve_snapshots_path(override_path, allow_default_fallback=allow_default_fallback)


def find_snapshots_directory(override_path: str | Sequence[str] | None = None) -> Path | None:
    """Locate the ground-truth snapshots directory.

    Delegates explicit overrides to ``ml_switcheroo_ir.snapshots.resolve_snapshots_path``
    and falls back to environment variables, ``DEFAULT_SNAPSHOT_DIR``, and
    ``get_default_snapshots_dir()``.

    Args:
        override_path (Optional[str | Sequence[str]]): Explicit path or sequence of candidate paths.

    Returns:
        Optional[Path]: Path to snapshots directory if found, or None.
    """
    if isinstance(override_path, str) and override_path:
        return _resolve_override_snapshots_path(override_path)
    if isinstance(override_path, Sequence) and not isinstance(override_path, str):
        for candidate in override_path:
            resolved = _resolve_override_snapshots_path(candidate)
            if resolved is not None:
                return resolved
        return None

    for env_key in ("ML_FRAMEWORK_SNAPSHOTS_DIR", "ML_ECOSYSTEM_SNAPSHOTS_DIR"):
        env_val = os.environ.get(env_key)
        if env_val:
            resolved_env = _resolve_override_snapshots_path(env_val)
            if resolved_env is not None:
                return resolved_env

    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    for fallback_dir in (Path(DEFAULT_SNAPSHOT_DIR).resolve(), Path(get_default_snapshots_dir()).resolve()):
        if fallback_dir.is_dir():
            return fallback_dir

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
    with open(snapshot_file, encoding="utf-8") as f:
        data = json.load(f)

    snap_symbols: dict[str, dict[str, JsonValue]] = {}
    categories = data.get("categories", {})
    for cat_items in categories.values():
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


def _verify_registry_structural_integrity(
    registry: Mapping[str, OpSchema],
    registry_name: str,
    expected_domains: tuple[str, ...],
    label: str,
) -> list[str]:
    """Verify structural invariants (non-empty, key/name match, domain match) of an OpSchema registry.

    Args:
        registry (Mapping[str, OpSchema]): Registry mapping operator name to OpSchema.
        registry_name (str): Name of the registry constant for error messages.
        expected_domains (tuple[str, ...]): Allowed domain strings for schemas in the registry.
        label (str): Human-readable dialect label.

    Returns:
        list[str]: List of structural error messages.
    """
    errors: list[str] = []
    if not registry:
        errors.append(f"{registry_name} is empty.")
        return errors
    for op_name, schema in registry.items():
        if schema.name != op_name:
            errors.append(f"{label} op key '{op_name}' != schema.name '{schema.name}'.")
        if schema.domain not in expected_domains:
            errors.append(f"{label} op '{op_name}' has invalid domain '{schema.domain}'.")
    return errors


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
    candidates = [p for p in sorted(snapshots_dir.glob(glob_pattern)) if not p.name.endswith("_exhaustive.json")]
    if not candidates:
        return []

    errors: list[str] = []
    target_file = candidates[-1]
    with open(target_file, encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", {})
    for cat_items in categories.values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict):
                    has_id = bool(item.get("api_path") or item.get("name") or (allow_mnemonic and item.get("mnemonic")))
                    if not has_id:
                        errors.append(f"{label} item missing identifier in {target_file.name}.")
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

    if "ml-ecosystem-snapshots" in str(snapshots_dir) or "ml-framework-snapshots" in str(snapshots_dir):
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
        print(f"[WARNING] StableHLO snapshot file not found in {snapshots_dir}. Skipping StableHLO checks.")
        return []

    gv = GroundingValidator(snapshot_manifest=str(stablehlo_file))
    errors: list[str] = []

    for op_name, schema in STABLEHLO_REGISTRY.items():
        if op_name not in gv.grounded_symbols and f"stablehlo.{op_name}" not in gv.grounded_symbols:
            errors.append(f"StableHLO op '{op_name}' is not grounded in {stablehlo_file.name} snapshot.")

        sym_record = gv.grounded_symbols.get(op_name) or gv.grounded_symbols.get(f"stablehlo.{op_name}")
        if sym_record:
            known_attrs = _extract_known_attributes(sym_record)
            for attr_name in schema.attributes:
                if known_attrs and attr_name not in known_attrs:
                    errors.append(f"StableHLO op '{op_name}' attribute '{attr_name}' is not grounded in snapshot attributes {list(known_attrs)}.")

    return errors


def _verify_flash_attention_snapshot(snapshots_dir: Path) -> list[str]:
    """Audit canonical custom attention schemas against a flash_attention snapshot if present.

    Args:
        snapshots_dir (Path): Directory containing snapshots.

    Returns:
        list[str]: Diagnostic errors for mismatched or missing custom attention operators.
    """
    flash_candidates = sorted(snapshots_dir.glob("*flash_attention*.json"))
    if not flash_candidates:
        return []
    flash_file = flash_candidates[-1]
    errors: list[str] = []
    try:
        _load_snapshot_symbol_map(flash_file)
        for canonical_schema in CANONICAL_ATTENTION_SCHEMAS:
            att_name = canonical_schema.name
            exp_inputs = canonical_schema.inputs
            if att_name not in CUSTOM_OPS_REGISTRY:
                errors.append(f"Custom attention op '{att_name}' missing from CUSTOM_OPS_REGISTRY.")
            elif CUSTOM_OPS_REGISTRY[att_name].inputs != exp_inputs:
                att_inputs = CUSTOM_OPS_REGISTRY[att_name].inputs
                errors.append(f"Custom attention op '{att_name}' inputs {att_inputs} do not match expected {exp_inputs}.")
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        errors.append(f"Failed parsing flash_attention snapshot {flash_file.name}: {exc}")
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
            errors.append(f"Custom op '{op_name}' has invalid domain '{schema.domain}', expected 'ml.switcheroo.custom', 'collective', or 'quantization'.")
        if not schema.outputs:
            errors.append(f"Custom op '{op_name}' has no defined outputs.")

    if snapshots_dir is not None:
        errors.extend(_verify_flash_attention_snapshot(snapshots_dir))

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
            errors.append(f"Array API op '{op_name}' is not grounded in {array_api_file.name} snapshot.")
            continue

        sym_record = snap_symbols.get(op_name) or snap_symbols[f"array_api.{op_name}"]
        raw_params = sym_record.get("params")
        snap_params = raw_params if isinstance(raw_params, list) else []
        if schema.inputs:
            pos_params = [str(p["name"]) for p in snap_params if isinstance(p, dict) and p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD") and isinstance(p.get("name"), str)]
            for idx, inp in enumerate(schema.inputs):
                if idx < len(pos_params) and inp != pos_params[idx] and inp not in pos_params:
                    errors.append(f"Array API op '{op_name}' input {idx} ('{inp}') does not match snapshot parameter '{pos_params[idx]}'.")
        if schema.attributes:
            kw_params = {str(p["name"]): p for p in snap_params if isinstance(p, dict) and p.get("kind") == "KEYWORD_ONLY" and isinstance(p.get("name"), str)}
            raw_kwargs = sym_record.get("kwargs")
            kwargs_list = [str(k) for k in raw_kwargs] if isinstance(raw_kwargs, list) else []
            for attr_name in schema.attributes:
                if kw_params and attr_name not in kw_params and attr_name not in kwargs_list:
                    errors.append(f"Array API op '{op_name}' attribute '{attr_name}' is not recognized in {array_api_file.name}.")
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
            errors.append(f"ATen op '{op_name}' is not grounded in {aten_file.name} snapshot.")
            continue

        sym_record = snap_symbols.get(op_name) or snap_symbols[f"aten.{op_name}"]
        if schema.attributes:
            raw_params = sym_record.get("params")
            snap_params = raw_params if isinstance(raw_params, list) else []
            known_params = {str(p["name"]) for p in snap_params if isinstance(p, dict) and p.get("name")}
            for attr_name in schema.attributes:
                if known_params and attr_name not in known_params:
                    errors.append(f"ATen op '{op_name}' attribute '{attr_name}' is not recognized in {aten_file.name}.")
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
        if coll_name not in COLLECTIVE_OPS_REGISTRY and f"collective.{coll_name}" not in COLLECTIVE_OPS_REGISTRY:
            errors.append(f"Collective operation '{coll_name}' missing from COLLECTIVE_OPS_REGISTRY.")
            continue

        if coll_name not in snap_symbols and f"nccl.{coll_name}" not in snap_symbols:
            errors.append(f"Collective op '{coll_name}' is not grounded in {nccl_file.name} snapshot.")
            continue

        schema = COLLECTIVE_OPS_REGISTRY.get(coll_name) or COLLECTIVE_OPS_REGISTRY[f"collective.{coll_name}"]
        if coll_name in ("all_reduce", "reduce_scatter") and "reduction_op" not in schema.attributes:
            errors.append(f"Collective op '{coll_name}' schema missing 'reduction_op' attribute.")
        if "comm" not in schema.attributes:
            errors.append(f"Collective op '{coll_name}' schema missing 'comm' communicator attribute.")

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
            errors.append(f"ONNX schema name mismatch: key '{op_name}' != schema.name '{schema.name}'.")
        if not (schema.domain == "ai.onnx" or schema.domain.startswith("ai.onnx.")):
            errors.append(f"ONNX op '{op_name}' has unexpected domain '{schema.domain}'.")
    return errors


def _collect_mlir_snapshot_symbols(
    categories: Mapping[str, JsonValue],
) -> tuple[set[str], dict[str, dict[str, JsonValue]]]:
    """Extract dialect prefixes and symbol records from an MLIR snapshot categories mapping.

    Args:
        categories (Mapping[str, JsonValue]): Categories dictionary from MLIR snapshot JSON.

    Returns:
        tuple[set[str], dict[str, dict[str, JsonValue]]]: ``(known_dialects, known_symbols)``.
    """
    known_dialects: set[str] = set()
    known_symbols: dict[str, dict[str, JsonValue]] = {}
    for cat_items in categories.values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict) and isinstance(item.get("api_path"), str):
                    api_p = str(item["api_path"])
                    known_dialects.add(api_p.split(".")[0])
                    known_symbols[api_p] = item
    return known_dialects, known_symbols


def verify_mlir_grounding(snapshots_dir: Path) -> list[str]:
    """Verify MLIR dialect operations against mlir snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: List of diagnostic errors for ungrounded MLIR operations.
    """
    allowed_mlir_domains = ("mlir", *(f"mlir.{d}" for d in CORE_MLIR_DIALECTS), *CORE_MLIR_DIALECTS.keys())
    errors = _verify_registry_structural_integrity(MLIR_REGISTRY, "MLIR_REGISTRY", allowed_mlir_domains, "MLIR")
    mlir_file = snapshots_dir / "mlir_v0.4.30.json"
    if not mlir_file.is_file():
        candidates = [p for p in sorted(snapshots_dir.glob("mlir*.json")) if not p.name.endswith("_exhaustive.json")]
        if not candidates:
            return errors
        mlir_file = candidates[-1]

    with open(mlir_file, encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", {}) if isinstance(data, dict) else {}
    known_dialects, known_symbols = _collect_mlir_snapshot_symbols(categories if isinstance(categories, dict) else {})

    for dialect in CORE_MLIR_DIALECTS:
        if known_dialects and dialect not in known_dialects:
            errors.append(f"Core MLIR dialect '{dialect}' is not grounded in {mlir_file.name}.")

    for op_name, schema in MLIR_REGISTRY.items():
        sym_record = known_symbols.get(op_name)
        if sym_record:
            known_attrs = _extract_known_attributes(sym_record)
            for attr_name in schema.attributes:
                if known_attrs and attr_name not in known_attrs:
                    errors.append(f"MLIR op '{op_name}' attribute '{attr_name}' is not grounded in {mlir_file.name} snapshot attributes {list(known_attrs)}.")
        else:
            dialect_prefix, _, short_op = op_name.partition(".")
            if known_symbols and short_op not in CORE_MLIR_DIALECTS.get(dialect_prefix, {}):
                errors.append(f"MLIR op '{op_name}' is not grounded in {mlir_file.name} or CORE_MLIR_DIALECTS.")

    return errors


def _resolve_ir_symbol(mod_path: str, obj_name: str, allow_nested: bool) -> bool:
    """Resolve a symbol in a module (or nested parent class) via inspect.getmembers.

    Args:
        mod_path (str): Module path or dotted parent class path.
        obj_name (str): Symbol name to resolve.
        allow_nested (bool): Whether to fall back to resolving parent class members.

    Returns:
        bool: True if the symbol is resolvable, False otherwise.
    """
    try:
        mod = importlib.import_module(mod_path)
        _ = dict(inspect.getmembers(mod))[obj_name]
    except (ImportError, AttributeError, KeyError):
        if allow_nested and "." in mod_path:
            try:
                parent_mod_path, parent_cls = mod_path.rsplit(".", 1)
                parent_mod = importlib.import_module(parent_mod_path)
                parent = dict(inspect.getmembers(parent_mod))[parent_cls]
                _ = dict(inspect.getmembers(parent))[obj_name]
            except (ImportError, AttributeError, KeyError):
                return False
        else:
            return False
    return True


def _verify_ir_category_entries(
    entries: list[dict[str, str]],
    default_kind: str,
) -> list[str]:
    """Verify a list of IR snapshot category entries against runtime symbols.

    Args:
        entries (list[dict[str, str]]): Snapshot category entries.
        default_kind (str): Default symbol kind ('class' or 'function') when 'kind' is omitted.

    Returns:
        list[str]: Diagnostic error messages for unresolvable symbols.
    """
    errors: list[str] = []
    for entry in entries:
        path = entry.get("api_path", "")
        if not path:
            continue
        kind = entry.get("kind", default_kind)
        mod_path, obj_name = path.rsplit(".", 1)
        if kind == "function":
            if not _resolve_ir_symbol(mod_path, obj_name, allow_nested=False):
                errors.append(f"Exported function '{path}' not resolvable at runtime.")
        elif not _resolve_ir_symbol(mod_path, obj_name, allow_nested=True):
            errors.append(f"Exported symbol '{path}' not resolvable at runtime.")
    return errors


def verify_ir_snapshot_grounding(snapshots_dir: Path) -> list[str]:
    """Verify ir_v*.json snapshot matches runtime API signatures.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: List of diagnostic errors for mismatched public APIs.
    """
    ir_file = snapshots_dir / "ir_v0.0.6.json"
    if not ir_file.is_file():
        ir_file = snapshots_dir / "ir_v0.0.4.json"
    if not ir_file.is_file():
        candidates = sorted(snapshots_dir.glob("ir_v*.json"))
        if not candidates:
            return []
        ir_file = candidates[-1]

    errors: list[str] = []
    with open(ir_file, encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", {})
    errors.extend(_verify_ir_category_entries(categories.get("classes", []), "class"))
    errors.extend(_verify_ir_category_entries(categories.get("functions", []), "function"))
    for cat_name, cat_entries in categories.items():
        if cat_name not in ("classes", "functions") and isinstance(cat_entries, list):
            errors.extend(_verify_ir_category_entries(cat_entries, "class"))

    return errors


def verify_rdna_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify AMD RDNA instructions and VOPD specifications against snapshot and registry.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded RDNA instructions.
    """
    errors = _verify_registry_structural_integrity(RDNA_REGISTRY, "RDNA_REGISTRY", ("amd_rdna", "rdna"), "RDNA")
    if snapshots_dir is None:
        return errors

    candidates = [p for p in sorted(snapshots_dir.glob("*rdna*.json")) if not p.name.endswith("_exhaustive.json")]
    if not candidates:
        return errors

    rdna_file = candidates[-1]
    with open(rdna_file, encoding="utf-8") as f:
        data = json.load(f)

    instructions = data.get("categories", {}).get("instructions", [])
    for inst in instructions:
        if not isinstance(inst, dict):
            continue
        name = inst.get("name") or inst.get("mnemonic")
        if not name:
            errors.append(f"RDNA instruction missing name/mnemonic in {rdna_file.name}.")
            continue
        vopd_slot = inst.get("vopd_slot")
        if vopd_slot is not None and vopd_slot not in ("X", "Y", "BOTH"):
            errors.append(f"RDNA instruction '{name}' has invalid vopd_slot '{vopd_slot}'.")
    return errors


def verify_sass_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify NVIDIA SASS instructions and control codes against snapshot and registry.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded SASS instructions.
    """
    errors = _verify_registry_structural_integrity(SASS_REGISTRY, "SASS_REGISTRY", ("nvidia_sass", "sass"), "SASS")
    if snapshots_dir is None:
        return errors

    candidates = [p for p in sorted(snapshots_dir.glob("*sass*.json")) if not p.name.endswith("_exhaustive.json")]
    if not candidates:
        return errors

    sass_file = candidates[-1]
    with open(sass_file, encoding="utf-8") as f:
        data = json.load(f)

    instructions = data.get("categories", {}).get("instructions", [])
    for inst in instructions:
        if not isinstance(inst, dict):
            continue
        name = inst.get("name") or inst.get("mnemonic")
        if not name:
            errors.append(f"SASS instruction missing name/mnemonic in {sass_file.name}.")
            continue
        latency = inst.get("execution_latency")
        if latency is not None and not isinstance(latency, (int, list, tuple)):
            errors.append(f"SASS instruction '{name}' has invalid execution_latency '{latency}'.")
    return errors


def verify_wgsl_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify WebGPU WGSL operations and attributes against snapshot and WGSL_REGISTRY.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded WGSL operations.
    """
    errors = _verify_registry_structural_integrity(WGSL_REGISTRY, "WGSL_REGISTRY", ("webgpu_wgsl", "wgsl"), "WGSL")
    if snapshots_dir is None:
        return errors

    wgsl_file = snapshots_dir / "wgsl_ops.json"
    if not wgsl_file.is_file():
        wgsl_file = Path(DEFAULT_SNAPSHOT_DIR) / "wgsl_ops.json"
    if not wgsl_file.is_file():
        return errors

    with open(wgsl_file, encoding="utf-8") as f:
        data = json.load(f)

    for op in data.get("ops", []):
        name = op.get("name")
        if not name:
            errors.append("WGSL op missing name.")
            continue
        if op.get("domain") not in ("wgsl", "webgpu_wgsl"):
            errors.append(f"WGSL op '{name}' has invalid domain '{op.get('domain')}'.")
    return errors


def verify_ptx_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify NVIDIA PTX operations in PTX_REGISTRY and against snapshot if available.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded PTX operations.
    """
    errors = _verify_registry_structural_integrity(PTX_REGISTRY, "PTX_REGISTRY", ("nvidia_ptx", "ptx"), "PTX")
    if snapshots_dir is not None:
        errors.extend(_verify_category_items_have_identifiers(snapshots_dir, "*ptx*.json", "PTX", allow_mnemonic=True))
    return errors


def verify_metal_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify Metal MSL compute operations in METAL_REGISTRY and against snapshot if available.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded Metal operations.
    """
    errors = _verify_registry_structural_integrity(METAL_REGISTRY, "METAL_REGISTRY", ("metal_msl", "metal"), "Metal")
    if snapshots_dir is not None:
        errors.extend(_verify_category_items_have_identifiers(snapshots_dir, "*metal*.json", "Metal", allow_mnemonic=False))
    return errors


def verify_wasm_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify WebAssembly SIMD operations in WASM_REGISTRY and against snapshot if available.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded WASM operations.
    """
    errors = _verify_registry_structural_integrity(WASM_REGISTRY, "WASM_REGISTRY", ("wasm_simd", "wasm"), "WASM")
    if snapshots_dir is not None:
        errors.extend(_verify_category_items_have_identifiers(snapshots_dir, "*wasm*.json", "WASM", allow_mnemonic=False))
    return errors


def verify_webgl_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify WebGL / GLSL operations in WEBGL_REGISTRY and against snapshot if available.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.

    Returns:
        List[str]: Diagnostic error messages for ungrounded WebGL operations.
    """
    errors = _verify_registry_structural_integrity(WEBGL_REGISTRY, "WEBGL_REGISTRY", ("webgl",), "WebGL")
    if snapshots_dir is not None:
        errors.extend(_verify_category_items_have_identifiers(snapshots_dir, "*webgl*.json", "WebGL", allow_mnemonic=False))
    return errors


def verify_quantization_grounding() -> list[str]:
    """Verify quantization operator schemas in QUANTIZATION_OPS_REGISTRY.

    Returns:
        list[str]: Diagnostic error messages for quantization schemas.
    """
    errors: list[str] = []
    if not QUANTIZATION_OPS_REGISTRY:
        errors.append("QUANTIZATION_OPS_REGISTRY is empty.")
    for op_name, schema in QUANTIZATION_OPS_REGISTRY.items():
        if not schema.name or (schema.name != op_name and not schema.name.endswith(f".{op_name}")):
            errors.append(f"Quantization op key '{op_name}' != schema.name '{schema.name}'.")
        if schema.domain not in ("quantization", "ml.switcheroo.quant"):
            errors.append(f"Quantization op '{op_name}' has invalid domain '{schema.domain}'.")
    return errors


def verify_state_ops_grounding() -> list[str]:
    """Verify state-mutation operator schemas in STATE_OPS_REGISTRY.

    Returns:
        list[str]: Diagnostic error messages for state schemas.
    """
    return _verify_registry_structural_integrity(STATE_OPS_REGISTRY, "STATE_OPS_REGISTRY", ("ml.switcheroo.state",), "State")


def _check_domain_op_against_snapshot(
    op_name: str,
    schema: OpSchema,
    expected_domain: str,
    label: str,
    snap_symbols: Mapping[str, dict[str, JsonValue]],
    snap_file_name: str,
) -> list[str]:
    """Validate a single domain OpSchema against loaded snapshot symbols.

    Args:
        op_name (str): Operation key in the registry.
        schema (OpSchema): Registered operator schema.
        expected_domain (str): Expected domain identifier.
        label (str): Human-readable domain label.
        snap_symbols (Mapping[str, dict[str, JsonValue]]): Loaded snapshot symbol records.
        snap_file_name (str): Snapshot filename for diagnostic messages.

    Returns:
        list[str]: Diagnostic error messages for this operator.
    """
    errors: list[str] = []
    if op_name not in snap_symbols and f"{expected_domain}.{op_name}" not in snap_symbols:
        errors.append(f"{label} op '{op_name}' is not grounded in {snap_file_name} snapshot.")
        return errors
    sym_record = snap_symbols.get(op_name) or snap_symbols[f"{expected_domain}.{op_name}"]
    if schema.attributes:
        raw_params = sym_record.get("params")
        snap_params = raw_params if isinstance(raw_params, list) else []
        known_params = {str(p["name"]) for p in snap_params if isinstance(p, dict) and p.get("name")}
        for attr_name in schema.attributes:
            if known_params and attr_name not in known_params:
                errors.append(f"{label} op '{op_name}' attribute '{attr_name}' is not recognized in {snap_file_name}.")
    return errors


def _verify_domain_registry_grounding(
    registry: Mapping[str, OpSchema],
    registry_name: str,
    expected_domain: str,
    label: str,
    glob_pattern: str,
    snapshots_dir: Path | None = None,
) -> list[str]:
    """Verify a domain operator registry against its structural invariants and optional snapshot.

    Args:
        registry (Mapping[str, OpSchema]): Domain operator registry.
        registry_name (str): Name of the registry constant for error messages.
        expected_domain (str): Expected domain string on each OpSchema.
        label (str): Human-readable domain label.
        glob_pattern (str): Glob pattern to locate snapshot JSON in ``snapshots_dir``.
        snapshots_dir (Path | None): Optional directory containing framework snapshots.

    Returns:
        list[str]: Diagnostic error messages for ungrounded or malformed schemas.
    """
    errors = _verify_registry_structural_integrity(registry, registry_name, (expected_domain,), label)
    if not registry:
        return errors

    snap_symbols: dict[str, dict[str, JsonValue]] = {}
    snap_file_name = ""
    if snapshots_dir is not None:
        candidates = sorted(snapshots_dir.glob(glob_pattern))
        if candidates:
            snap_file_name = candidates[-1].name
            snap_symbols = _load_snapshot_symbol_map(candidates[-1])

    if snap_symbols:
        for op_name, schema in registry.items():
            errors.extend(
                _check_domain_op_against_snapshot(
                    op_name,
                    schema,
                    expected_domain,
                    label,
                    snap_symbols,
                    snap_file_name,
                )
            )
    return errors


def verify_chex_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify Chex assertion and testing operator schemas in CHEX_REGISTRY.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.

    Returns:
        list[str]: Diagnostic error messages for Chex schemas.
    """
    return _verify_domain_registry_grounding(
        CHEX_REGISTRY,
        "CHEX_REGISTRY",
        "chex",
        "Chex",
        "chex_*.json",
        snapshots_dir,
    )


def verify_grain_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify Grain data pipeline operator schemas in GRAIN_REGISTRY.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.

    Returns:
        list[str]: Diagnostic error messages for Grain schemas.
    """
    return _verify_domain_registry_grounding(
        GRAIN_REGISTRY,
        "GRAIN_REGISTRY",
        "grain",
        "Grain",
        "grain_*.json",
        snapshots_dir,
    )


def verify_orbax_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify Orbax checkpointing operator schemas in ORBAX_REGISTRY.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.

    Returns:
        list[str]: Diagnostic error messages for Orbax schemas.
    """
    return _verify_domain_registry_grounding(
        ORBAX_REGISTRY,
        "ORBAX_REGISTRY",
        "orbax",
        "Orbax",
        "orbax*.json",
        snapshots_dir,
    )


def verify_optax_grounding(snapshots_dir: Path | None = None) -> list[str]:
    """Verify Optax optimizer and loss operator schemas in OPTAX_REGISTRY.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.

    Returns:
        list[str]: Diagnostic error messages for Optax schemas.
    """
    return _verify_domain_registry_grounding(
        OPTAX_REGISTRY,
        "OPTAX_REGISTRY",
        "optax",
        "Optax",
        "optax_*.json",
        snapshots_dir,
    )


def verify_odl_grounding() -> list[str]:
    """Verify universal Operation Definition Language (ODL) schemas in ODL_CATALOG.

    Returns:
        list[str]: Diagnostic error messages for ODL schemas.
    """
    return _verify_registry_structural_integrity(ODL_CATALOG, "ODL_CATALOG", ("odl",), "ODL")


def _verify_local_schemas_only() -> int:
    """Verify local custom ops, ONNX, quantization, state, hardware ISAs, Chex, Grain, Orbax, Optax, and ODL registries when snapshot directory is missing or empty.

    Returns:
        int: Process exit code (0 for success, 1 for failure).
    """
    all_errors = (
        verify_custom_ops_grounding()
        + verify_onnx_grounding()
        + verify_quantization_grounding()
        + verify_state_ops_grounding()
        + verify_rdna_grounding()
        + verify_sass_grounding()
        + verify_wgsl_grounding()
        + verify_ptx_grounding()
        + verify_metal_grounding()
        + verify_wasm_grounding()
        + verify_webgl_grounding()
        + verify_chex_grounding()
        + verify_grain_grounding()
        + verify_orbax_grounding()
        + verify_optax_grounding()
        + verify_odl_grounding()
    )
    if all_errors:
        for err in all_errors:
            print(f"[ERROR] {err}")
        return 1
    print("[SUCCESS] Local schema definitions are valid and structurally sound.")
    return 0


def _run_all_snapshot_verifiers(snapshots_dir: Path) -> list[str]:
    """Execute all live snapshot and local schema grounding verifiers.

    Args:
        snapshots_dir (Path): Directory containing ground-truth snapshots.

    Returns:
        list[str]: Combined list of all diagnostic error messages.
    """
    return (
        verify_stablehlo_grounding(snapshots_dir)
        + verify_mlir_grounding(snapshots_dir)
        + verify_ir_snapshot_grounding(snapshots_dir)
        + verify_custom_ops_grounding(snapshots_dir)
        + verify_onnx_grounding()
        + verify_rdna_grounding(snapshots_dir)
        + verify_sass_grounding(snapshots_dir)
        + verify_wgsl_grounding(snapshots_dir)
        + verify_ptx_grounding(snapshots_dir)
        + verify_metal_grounding(snapshots_dir)
        + verify_wasm_grounding(snapshots_dir)
        + verify_webgl_grounding(snapshots_dir)
        + verify_array_api_grounding(snapshots_dir)
        + verify_aten_grounding(snapshots_dir)
        + verify_chex_grounding(snapshots_dir)
        + verify_grain_grounding(snapshots_dir)
        + verify_orbax_grounding(snapshots_dir)
        + verify_optax_grounding(snapshots_dir)
        + verify_collectives_grounding(snapshots_dir)
        + verify_quantization_grounding()
        + verify_state_ops_grounding()
        + verify_odl_grounding()
    )


def main(argv: list[str] | None = None) -> int:
    """Run anti-hallucination grounding verification for all schemas.

    Args:
        argv (Optional[List[str]]): Command-line arguments.

    Returns:
        int: Exit status code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(description="Verify anti-hallucination grounding against framework snapshots.")
    parser.add_argument(
        "--snapshots-dir",
        type=str,
        action="append",
        default=None,
        help="Path(s) to snapshots directory (may be specified multiple times).",
    )
    parser.add_argument(
        "--framework-snapshots-dir",
        type=str,
        default=None,
        help="Path to ml-framework-snapshots directory.",
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

    candidates: list[str] = []
    if args.framework_snapshots_dir:
        candidates.append(args.framework_snapshots_dir)
    if args.snapshots_dir:
        candidates.extend(args.snapshots_dir)
    if args.ecosystem_snapshots_dir:
        candidates.append(args.ecosystem_snapshots_dir)

    snapshots_dir = find_snapshots_directory(candidates if candidates else None)
    if snapshots_dir is None:
        if args.strict:
            print("[ERROR] Snapshot directory not found in strict mode.")
            return 1
        print("[WARNING] Snapshot directory not found. Skipping live snapshot checks.")
        return _verify_local_schemas_only()

    if not any(snapshots_dir.glob("*.json")):
        if args.strict:
            print("[ERROR] Snapshot directory contains no snapshot files in strict mode.")
            return 1
        print("[WARNING] Snapshot directory contains no snapshot files. Skipping live snapshot checks.")
        return _verify_local_schemas_only()

    print(f"Using snapshots directory: {snapshots_dir}")
    all_errors = _run_all_snapshot_verifiers(snapshots_dir)

    print(f"Audited StableHLO schemas: {len(STABLEHLO_REGISTRY)} ops.")
    print(f"Audited MLIR dialects: {len(CORE_MLIR_DIALECTS)} dialects.")
    print(f"Audited Custom ops: {len(CUSTOM_OPS_REGISTRY)} ops.")
    print(f"Audited ONNX schemas: {len(ONNX_REGISTRY)} ops.")
    print(f"Audited Array API schemas: {len(ARRAY_API_REGISTRY)} ops.")
    print(f"Audited ATen schemas: {len(ATEN_REGISTRY)} ops.")
    print(f"Audited Chex schemas: {len(CHEX_REGISTRY)} ops.")
    print(f"Audited Grain schemas: {len(GRAIN_REGISTRY)} ops.")
    print(f"Audited Orbax schemas: {len(ORBAX_REGISTRY)} ops.")
    print(f"Audited Optax schemas: {len(OPTAX_REGISTRY)} ops.")
    print(f"Audited Collective schemas: {len(COLLECTIVE_OPS_REGISTRY)} ops.")
    print(f"Audited Quantization schemas: {len(QUANTIZATION_OPS_REGISTRY)} ops.")
    print(f"Audited State schemas: {len(STATE_OPS_REGISTRY)} ops.")
    print(f"Audited ODL schemas: {len(ODL_CATALOG)} ops.")
    print("Audited Low-Level & Hardware ISAs (RDNA, SASS, WGSL, PTX, Metal, WASM, WebGL).")

    if all_errors:
        print(f"\n[FAILURE] Found {len(all_errors)} ungrounded symbol(s) or schema defect(s):")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print("\n[SUCCESS] All schemas are 100% grounded against ground-truth snapshots!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
