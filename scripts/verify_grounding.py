"""Script to verify anti-hallucination grounding of registered schemas against ground-truth snapshots."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from ml_switcheroo_ir.schema.base import normalize_attribute_type
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
    ODL_DIALECT_MAPPINGS,
    OPTAX_REGISTRY,
    ORBAX_REGISTRY,
)
from ml_switcheroo_ir.schema.ghost import (
    RDNA3_VOPD_OPERATORS,
    WGSL_PRIMITIVE_SIGNATURES,
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
from ml_switcheroo_ir.schema.rdna_registry import (
    RDNA_REGISTRY,
    RDNA_TO_VOPD_MAP,
    RDNA_VOPD_SLOTS,
)
from ml_switcheroo_ir.schema.sass_registry import (
    SASS_PIPELINE_LATENCIES,
    SASS_REGISTRY,
)
from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY
from ml_switcheroo_ir.translation import ParameterTranslationEngine
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
        Path | None: Resolved snapshots directory path if found, or None.
    """
    from ml_switcheroo_ir.snapshots import resolve_snapshots_path

    return resolve_snapshots_path(override_path, allow_default_fallback=allow_default_fallback)


def _resolve_sequence_overrides(candidates: Sequence[str]) -> Path | None:
    """Resolve a sequence of candidate snapshot directories, preferring populated directories.

    Args:
        candidates (Sequence[str]): Ordered candidate directory paths.

    Returns:
        Path | None: First populated or existing resolved directory, else None.
    """
    if len(candidates) == 1:
        return _resolve_override_snapshots_path(candidates[0], allow_default_fallback=False)
    for candidate in candidates:
        resolved = _resolve_override_snapshots_path(candidate, allow_default_fallback=True)
        if resolved is not None and any(resolved.glob("*.json")):
            return resolved
    for candidate in candidates:
        resolved = _resolve_override_snapshots_path(candidate, allow_default_fallback=False)
        if resolved is not None:
            return resolved
    return None


def find_snapshots_directory(override_path: str | Sequence[str] | None = None) -> Path | None:
    """Locate the ground-truth snapshots directory.

    Fails fast returning ``None`` when an explicit single ``override_path`` is provided
    and missing or empty, rather than silently falling back to site-packages caches.

    Args:
        override_path (str | Sequence[str] | None): Explicit path or sequence of candidate paths.

    Returns:
        Path | None: Path to snapshots directory if found, or None.
    """
    if isinstance(override_path, str) and override_path:
        return _resolve_override_snapshots_path(override_path, allow_default_fallback=False)
    if isinstance(override_path, Sequence) and not isinstance(override_path, str):
        return _resolve_sequence_overrides(override_path)

    for env_key in ("ML_FRAMEWORK_SNAPSHOTS_DIR", "ML_ECOSYSTEM_SNAPSHOTS_DIR"):
        env_val = os.environ.get(env_key)
        if env_val:
            resolved_env = _resolve_override_snapshots_path(env_val)
            if resolved_env is not None:
                return resolved_env

    from ml_switcheroo_ir.snapshots import get_default_snapshots_dir

    if DEFAULT_SNAPSHOT_DIR and Path(DEFAULT_SNAPSHOT_DIR).is_dir():
        return Path(DEFAULT_SNAPSHOT_DIR).resolve()
    default_dir_str = get_default_snapshots_dir()
    if not default_dir_str:
        return None
    default_dir = Path(default_dir_str).resolve()
    return default_dir if default_dir.is_dir() else None


def _index_snapshot_record(snap_symbols: dict[str, dict[str, JsonValue]], item: dict[str, JsonValue]) -> None:
    """Index a single snapshot item dictionary by ``name``, ``mnemonic``, and ``api_path``.

    Args:
        snap_symbols (dict[str, dict[str, JsonValue]]): Target symbol lookup dictionary.
        item (dict[str, JsonValue]): Raw snapshot entry dictionary.
    """
    for key_field in ("name", "mnemonic"):
        raw_name = item.get(key_field)
        if isinstance(raw_name, str) and raw_name:
            snap_symbols[raw_name] = item
    api_path = item.get("api_path")
    if isinstance(api_path, str) and api_path:
        snap_symbols[api_path] = item
        snap_symbols[api_path.split(".")[-1]] = item
        if ".inst." in api_path:
            snap_symbols[api_path.split(".inst.", 1)[1]] = item


def _collect_manifest_records(data: JsonValue) -> list[dict[str, JsonValue]]:
    """Extract all record dictionaries from a snapshot JSON payload.

    Args:
        data (JsonValue): Parsed JSON payload.

    Returns:
        list[dict[str, JsonValue]]: Flattened record dictionaries.
    """
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    records: list[dict[str, JsonValue]] = []
    categories = data.get("categories")
    if isinstance(categories, dict):
        for cat_items in categories.values():
            if isinstance(cat_items, list):
                records.extend(item for item in cat_items if isinstance(item, dict))
    for list_key in ("operations", "instructions", "ops"):
        seq = data.get(list_key)
        if isinstance(seq, list):
            records.extend(item for item in seq if isinstance(item, dict))
    return records


def _load_snapshot_symbol_map(snapshot_file: Path) -> dict[str, dict[str, JsonValue]]:
    """Load a snapshot JSON file and index its entries across categories, operations, instructions, and ops.

    Args:
        snapshot_file (Path): Path to the JSON snapshot file.

    Returns:
        dict[str, dict[str, JsonValue]]: Mapping from symbol name / api_path to its snapshot record.
    """
    with open(snapshot_file, encoding="utf-8") as f:
        data = json.load(f)
    snap_symbols: dict[str, dict[str, JsonValue]] = {}
    for item in _collect_manifest_records(data):
        _index_snapshot_record(snap_symbols, item)
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
    if not isinstance(data, dict):
        return errors

    categories = data.get("categories", {})
    if isinstance(categories, dict):
        for cat_items in categories.values():
            if isinstance(cat_items, list):
                for item in cat_items:
                    if isinstance(item, dict):
                        has_id = bool(
                            item.get("api_path") or item.get("name") or (allow_mnemonic and item.get("mnemonic"))
                        )
                        if not has_id:
                            errors.append(f"{label} item missing identifier in {target_file.name}.")
    return errors


def _extract_known_attributes(
    sym_record: Mapping[str, SnapshotManifestNode] | Mapping[str, JsonValue],
) -> set[str]:
    """Extract known parameter, kwarg, and attribute names from a snapshot symbol record.

    Args:
        sym_record (Mapping[str, SnapshotManifestNode] | Mapping[str, JsonValue]): Snapshot symbol dictionary.

    Returns:
        set[str]: Set of grounded attribute/parameter names.
    """
    known: set[str] = set()
    raw_params = sym_record.get("params")
    if isinstance(raw_params, list):
        for p in raw_params:
            if isinstance(p, dict) and p.get("name"):
                known.add(str(p["name"]))
    raw_kwargs = sym_record.get("kwargs")
    if isinstance(raw_kwargs, list):
        known.update(str(k) for k in raw_kwargs if isinstance(k, str) and k)
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


def _missing_snapshot_error(label: str, pattern: str, snapshots_dir: Path, strict: bool) -> list[str]:
    """Build diagnostic error list when a required snapshot file is absent in strict mode.

    Args:
        label (str): Human-readable dialect name.
        pattern (str): Expected filename or glob pattern.
        snapshots_dir (Path): Searched snapshots directory.
        strict (bool): Whether missing snapshot files are fatal errors.

    Returns:
        list[str]: Single-element error list when ``strict`` is True, else empty list.
    """
    if strict:
        return [f"Missing required {label} snapshot ({pattern}) in {snapshots_dir}."]
    return []


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
        except (ImportError, SyntaxError, AttributeError, TypeError):
            pass

    return None


def verify_stablehlo_grounding(snapshots_dir: Path, *, strict: bool = False) -> list[str]:
    """Verify all StableHLO registry schemas against stablehlo snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.
        strict (bool): Fail if no StableHLO snapshot exists in ``snapshots_dir``.

    Returns:
        list[str]: List of error diagnostic messages for any ungrounded symbols.
    """
    errors = _verify_registry_structural_integrity(
        STABLEHLO_REGISTRY, "STABLEHLO_REGISTRY", ("stablehlo",), "StableHLO"
    )
    stablehlo_file = _find_stablehlo_snapshot_file(snapshots_dir)
    if stablehlo_file is None:
        if strict:
            errors.extend(_missing_snapshot_error("StableHLO", "stablehlo*.json", snapshots_dir, strict=True))
        else:
            print(f"[WARNING] StableHLO snapshot file not found in {snapshots_dir}. Skipping StableHLO checks.")
        return errors

    gv = GroundingValidator(snapshot_manifest=str(stablehlo_file))
    for op_name, schema in STABLEHLO_REGISTRY.items():
        if op_name not in gv.grounded_symbols and f"stablehlo.{op_name}" not in gv.grounded_symbols:
            errors.append(f"StableHLO op '{op_name}' is not grounded in {stablehlo_file.name} snapshot.")
            continue

        sym_record = gv.grounded_symbols.get(op_name) or gv.grounded_symbols.get(f"stablehlo.{op_name}")
        if sym_record:
            known_attrs = _extract_known_attributes(sym_record)
            for attr_name in schema.attributes:
                if known_attrs and attr_name not in known_attrs:
                    errors.append(
                        f"StableHLO op '{op_name}' attribute '{attr_name}' is not grounded "
                        f"in snapshot attributes {list(known_attrs)}."
                    )
    return errors


_CUSTOM_ATTENTION_SNAPSHOT_KEYS: dict[str, tuple[str, ...]] = {
    "FlashAttention": ("FlashAttention", "ml.switcheroo.custom.FlashAttention", "flash_attn_func"),
    "ScaledDotProductAttention": ("ScaledDotProductAttention", "ml.switcheroo.custom.ScaledDotProductAttention"),
    "RoPE": ("RoPE", "ml.switcheroo.custom.RoPE"),
    "PagedAttention": ("PagedAttention", "paged_attention_v1", "vllm.paged_attention_v1"),
    "RaggedPagedAttention": (
        "RaggedPagedAttention",
        "ragged_paged_attention",
        "ml_switcheroo_compiler.ops.ragged_paged_attention",
    ),
}


def _find_attention_symbol(
    op_name: str,
    flash_symbols: Mapping[str, dict[str, JsonValue]],
) -> tuple[str, dict[str, JsonValue] | None]:
    """Locate the matching symbol record for a custom attention operator.

    Args:
        op_name (str): Attention operator name.
        flash_symbols (Mapping[str, dict[str, JsonValue]]): Loaded flash_attention symbol map.

    Returns:
        tuple[str, dict[str, JsonValue] | None]: ``(matched_key, symbol_record)``.
    """
    for k in _CUSTOM_ATTENTION_SNAPSHOT_KEYS.get(op_name, (op_name,)):
        if k in flash_symbols:
            return k, flash_symbols[k]
    return "", None


def _check_single_attention_op(
    op_name: str,
    reg_op: OpSchema,
    flash_symbols: Mapping[str, dict[str, JsonValue]],
    flash_file_name: str,
) -> list[str]:
    """Verify a single custom attention OpSchema against bound ``flash_symbols`` metadata.

    Args:
        op_name (str): Attention operator name in ``CUSTOM_OPS_REGISTRY``.
        reg_op (OpSchema): Registered ``OpSchema`` instance.
        flash_symbols (Mapping[str, dict[str, JsonValue]]): Bound symbols from ``flash_attention_*.json``.
        flash_file_name (str): Snapshot filename for error reporting.

    Returns:
        list[str]: Diagnostic error messages for signature mismatches.
    """
    matched_key, sym = _find_attention_symbol(op_name, flash_symbols)
    if sym is None:
        return [f"Custom attention op '{op_name}' is not grounded in {flash_file_name}."]

    errors: list[str] = []
    raw_params = sym.get("params")
    param_names = [
        str(p["name"])
        for p in (raw_params if isinstance(raw_params, list) else [])
        if isinstance(p, dict) and p.get("name")
    ]
    raw_kwargs = sym.get("kwargs")
    kwarg_set = {str(k) for k in (raw_kwargs if isinstance(raw_kwargs, list) else []) if isinstance(k, str)}
    known_all = set(param_names) | kwarg_set | _extract_known_attributes(sym)

    exact_keys = ("FlashAttention", "ml.switcheroo.custom.FlashAttention", "RoPE", "ml.switcheroo.custom.RoPE")
    if matched_key in exact_keys:
        expected_inputs = [p for p in param_names if p not in kwarg_set]
        if expected_inputs and reg_op.inputs != expected_inputs:
            errors.append(
                f"Custom attention op '{op_name}' inputs {reg_op.inputs} do not match "
                f"{flash_file_name} inputs {expected_inputs}."
            )
    elif known_all:
        errors.extend(
            f"Custom attention op '{op_name}' input '{inp}' is not grounded in {flash_file_name} ({matched_key})."
            for inp in reg_op.inputs
            if inp not in known_all
        )
    if known_all:
        errors.extend(
            f"Custom attention op '{op_name}' attribute '{attr_name}' is not grounded "
            f"in {flash_file_name} ({matched_key})."
            for attr_name in reg_op.attributes
            if attr_name not in known_all
        )
    return errors


def _verify_flash_attention_snapshot(snapshots_dir: Path, *, strict: bool = False) -> list[str]:
    """Audit custom attention schemas against bound ``flash_attention_*.json`` symbol signatures.

    Args:
        snapshots_dir (Path): Directory containing snapshots.
        strict (bool): Fail if ``flash_attention_*.json`` is absent.

    Returns:
        list[str]: Diagnostic errors for ungrounded or mismatched attention operators.
    """
    flash_candidates = sorted(snapshots_dir.glob("*flash_attention*.json"))
    if not flash_candidates:
        return _missing_snapshot_error("FlashAttention", "*flash_attention*.json", snapshots_dir, strict)
    flash_file = flash_candidates[-1]
    errors: list[str] = []
    try:
        flash_symbols = _load_snapshot_symbol_map(flash_file)
        if not flash_symbols:
            errors.append(f"FlashAttention snapshot {flash_file.name} contains no symbols.")
            return errors
        checked_names = {s.name for s in CANONICAL_ATTENTION_SCHEMAS} | set(_CUSTOM_ATTENTION_SNAPSHOT_KEYS)
        for att_name in sorted(checked_names):
            if att_name not in CUSTOM_OPS_REGISTRY:
                errors.append(f"Custom attention op '{att_name}' missing from CUSTOM_OPS_REGISTRY.")
                continue
            errors.extend(
                _check_single_attention_op(att_name, CUSTOM_OPS_REGISTRY[att_name], flash_symbols, flash_file.name)
            )
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"Failed parsing flash_attention snapshot {flash_file.name}: {exc}")
    return errors


_KNOWN_CUSTOM_CONCEPT_OPS: frozenset[str] = frozenset(
    {
        "RMSNorm",
        "RMSNormBackward",
        "SwiGLU",
        "RoPE",
        "FlashAttention",
        "PagedAttention",
        "RaggedPagedAttention",
        "VisionPatchEmbedding",
        "ScaledDotProductAttention",
        "LayerNorm",
        "GroupNorm",
        "GELU",
        "EmbeddingLookup",
    }
)


def verify_custom_ops_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify all custom operator schemas in CUSTOM_OPS_REGISTRY against structural and snapshot invariants.

    Args:
        snapshots_dir (Path | None): Directory containing snapshots.
        strict (bool): Fail if required snapshot manifests are absent.

    Returns:
        list[str]: List of diagnostic errors for custom operators.
    """
    errors = _verify_registry_structural_integrity(
        CUSTOM_OPS_REGISTRY,
        "CUSTOM_OPS_REGISTRY",
        ("ml.switcheroo.custom", "collective", "quantization"),
        "Custom",
    )
    for op_name, schema in CUSTOM_OPS_REGISTRY.items():
        if not schema.outputs:
            errors.append(f"Custom op '{op_name}' has no defined outputs.")

    if snapshots_dir is not None:
        errors.extend(_verify_flash_attention_snapshot(snapshots_dir, strict=strict))
        custom_snap_files = sorted(snapshots_dir.glob("custom_ops*.json"))
        custom_snap_syms = _load_snapshot_symbol_map(custom_snap_files[-1]) if custom_snap_files else {}
        for op_name, schema in CUSTOM_OPS_REGISTRY.items():
            if schema.domain == "ml.switcheroo.custom":
                if op_name not in _KNOWN_CUSTOM_CONCEPT_OPS and op_name not in custom_snap_syms:
                    errors.append(f"Custom op '{op_name}' is not grounded in snapshots or canonical custom ops.")
    return errors


def verify_array_api_grounding(snapshots_dir: Path, *, strict: bool = False) -> list[str]:
    """Verify Array API operators against array_api snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.
        strict (bool): Fail if ``*array_api*.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for ungrounded Array API operations.
    """
    errors = _verify_registry_structural_integrity(
        ARRAY_API_REGISTRY, "ARRAY_API_REGISTRY", ("array_api",), "Array API"
    )
    candidates = sorted(snapshots_dir.glob("*array_api*.json"))
    if not candidates:
        return errors + _missing_snapshot_error("Array API", "*array_api*.json", snapshots_dir, strict)

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
            pos_params = [
                str(p["name"])
                for p in snap_params
                if isinstance(p, dict)
                and p.get("kind") in ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
                and isinstance(p.get("name"), str)
            ]
            for idx, inp in enumerate(schema.inputs):
                if idx < len(pos_params) and inp != pos_params[idx] and inp not in pos_params:
                    errors.append(
                        f"Array API op '{op_name}' input {idx} ('{inp}') does not match "
                        f"snapshot parameter '{pos_params[idx]}'."
                    )
        if schema.attributes:
            known_kw = _extract_known_attributes(sym_record)
            for attr_name in schema.attributes:
                if known_kw and attr_name not in known_kw:
                    errors.append(
                        f"Array API op '{op_name}' attribute '{attr_name}' is not recognized in {array_api_file.name}."
                    )
    return errors


def verify_aten_grounding(snapshots_dir: Path, *, strict: bool = False) -> list[str]:
    """Verify ATen operators against aten snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.
        strict (bool): Fail if ``*aten*.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for ungrounded ATen operations.
    """
    errors = _verify_registry_structural_integrity(ATEN_REGISTRY, "ATEN_REGISTRY", ("aten",), "ATen")
    candidates = sorted(snapshots_dir.glob("*aten*.json"))
    if not candidates:
        return errors + _missing_snapshot_error("ATen", "*aten*.json", snapshots_dir, strict)

    aten_file = candidates[-1]
    snap_symbols = _load_snapshot_symbol_map(aten_file)

    for op_name, schema in ATEN_REGISTRY.items():
        if op_name not in snap_symbols and f"aten.{op_name}" not in snap_symbols:
            errors.append(f"ATen op '{op_name}' is not grounded in {aten_file.name} snapshot.")
            continue

        sym_record = snap_symbols.get(op_name) or snap_symbols[f"aten.{op_name}"]
        if schema.attributes:
            known_params = _extract_known_attributes(sym_record)
            for attr_name in schema.attributes:
                if known_params and attr_name not in known_params:
                    errors.append(f"ATen op '{op_name}' attribute '{attr_name}' is not recognized in {aten_file.name}.")
    return errors


def verify_collectives_grounding(snapshots_dir: Path, *, strict: bool = False) -> list[str]:
    """Verify collective operations against NCCL snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.
        strict (bool): Fail if ``*nccl*.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for collective operations.
    """
    errors = _verify_registry_structural_integrity(
        COLLECTIVE_OPS_REGISTRY, "COLLECTIVE_OPS_REGISTRY", ("collective",), "Collective"
    )
    candidates = sorted(snapshots_dir.glob("*nccl*.json"))
    if not candidates:
        return errors + _missing_snapshot_error("NCCL", "*nccl*.json", snapshots_dir, strict)

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

        schema = COLLECTIVE_OPS_REGISTRY[coll_name]
        if coll_name in ("all_reduce", "reduce_scatter") and "reduction_op" not in schema.attributes:
            errors.append(f"Collective op '{coll_name}' schema missing 'reduction_op' attribute.")
        if "comm" not in schema.attributes:
            errors.append(f"Collective op '{coll_name}' schema missing 'comm' communicator attribute.")

    for op_name in COLLECTIVE_OPS_REGISTRY:
        short_name = op_name.split(".", 1)[-1]
        if short_name not in snap_symbols and f"nccl.{short_name}" not in snap_symbols:
            errors.append(f"Collective op '{op_name}' is not grounded in {nccl_file.name} snapshot.")

    return errors


def _extract_onnx_io_names(item: Mapping[str, JsonValue], direct_key: str, struct_key: str) -> list[str]:
    """Extract input or output parameter names from an ONNX snapshot record.

    Args:
        item (Mapping[str, JsonValue]): Raw snapshot entry.
        direct_key (str): Direct string-list key ('inputs' or 'outputs').
        struct_key (str): Structured record-list key ('operands' or 'returns').

    Returns:
        list[str]: Ordered formal parameter names.
    """
    raw_direct = item.get(direct_key)
    if isinstance(raw_direct, list) and all(isinstance(x, str) for x in raw_direct):
        return [str(x) for x in raw_direct]
    structured = item.get(struct_key)
    if isinstance(structured, list):
        return [str(elem["name"]) for elem in structured if isinstance(elem, Mapping) and elem.get("name")]
    return []


def _extract_onnx_attr_map(raw_attrs: JsonValue) -> dict[str, JsonValue]:
    """Extract normalized ``{attr_name: {type, required}}`` dictionary from an ONNX snapshot record.

    Args:
        raw_attrs (JsonValue): Raw attributes dict or list.

    Returns:
        dict[str, JsonValue]: Normalized attribute specs.
    """
    attrs: dict[str, JsonValue] = {}
    if isinstance(raw_attrs, Mapping):
        for a_k, a_v in raw_attrs.items():
            if isinstance(a_v, Mapping):
                attrs[str(a_k)] = {"type": str(a_v.get("type", "Any")), "required": bool(a_v.get("required", False))}
    elif isinstance(raw_attrs, list):
        for a_v in raw_attrs:
            if isinstance(a_v, Mapping) and a_v.get("name"):
                attrs[str(a_v["name"])] = {
                    "type": str(a_v.get("type", "Any")),
                    "required": bool(a_v.get("required", False)),
                }
    return attrs


def _parse_onnx_snapshot_entry(item: Mapping[str, JsonValue]) -> dict[str, JsonValue] | None:
    """Convert a single ONNX category/operation item into a normalized ground-truth spec dict.

    Args:
        item (Mapping[str, JsonValue]): Raw snapshot entry from ``onnx_*.json``.

    Returns:
        dict[str, JsonValue] | None: Normalized ground-truth operator spec or None.
    """
    name = str(item.get("name") or "")
    if not name:
        return None
    meta = item.get("domain_metadata")
    domain = str(item.get("domain") or "ai.onnx")
    raw_ver = item.get("version")
    version = raw_ver if isinstance(raw_ver, int) and not isinstance(raw_ver, bool) else 1
    exact_version = "version" in item
    if isinstance(meta, Mapping):
        if isinstance(meta.get("domain"), str) and meta["domain"]:
            domain = str(meta["domain"])
        up_ver = meta.get("upstream_version")
        if isinstance(up_ver, (int, str)) and str(up_ver).isdigit():
            version = int(up_ver)
            exact_version = False
    inputs_json: list[JsonValue] = list(_extract_onnx_io_names(item, "inputs", "operands"))
    outputs_json: list[JsonValue] = list(_extract_onnx_io_names(item, "outputs", "returns"))
    return {
        "name": name,
        "domain": domain,
        "version": version,
        "exact_version": exact_version,
        "inputs": inputs_json,
        "outputs": outputs_json,
        "attributes": _extract_onnx_attr_map(item.get("attributes")),
    }


def _load_onnx_ground_truth(onnx_file: Path) -> tuple[dict[str, dict[str, JsonValue]], bool]:
    """Load ONNX ground-truth operator specs from an ``onnx_*.json`` manifest.

    Args:
        onnx_file (Path): Path to ONNX JSON snapshot file.

    Returns:
        tuple[dict[str, dict[str, JsonValue]], bool]: ``(specs_by_name, is_partial_category_summary)``.
    """
    with open(onnx_file, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        return {}, False

    specs: dict[str, dict[str, JsonValue]] = {}
    if "categories" in data or "operations" in data or "ops" in data:
        is_partial = "categories" in data and "operations" not in data and "ops" not in data
        for it in _collect_manifest_records(dict(data)):
            parsed = _parse_onnx_snapshot_entry(it)
            if parsed is not None and str(parsed["name"]) not in specs:
                specs[str(parsed["name"])] = parsed
        return specs, is_partial

    for op_name, op_val in data.items():
        if isinstance(op_val, Mapping):
            merged = dict(op_val)
            merged.setdefault("name", str(op_name))
            parsed = _parse_onnx_snapshot_entry(merged)
            if parsed is not None:
                specs[str(op_name)] = parsed
    return specs, False


def _compare_onnx_attributes(
    op_name: str,
    schema: OpSchema,
    gt_attrs: Mapping[str, JsonValue],
    exact_version: bool,
    src: str,
) -> list[str]:
    """Compare registered ONNX OpAttribute definitions against ground-truth attributes.

    Args:
        op_name (str): Operator name.
        schema (OpSchema): Registered ONNX schema.
        gt_attrs (Mapping[str, JsonValue]): Ground-truth attribute mapping.
        exact_version (bool): Whether extra attributes in ``schema`` are disallowed.
        src (str): Source label for diagnostics.

    Returns:
        list[str]: Attribute mismatch diagnostics.
    """
    errors: list[str] = []
    for a_name, a_spec in gt_attrs.items():
        if a_name not in schema.attributes:
            errors.append(f"ONNX op '{op_name}' missing grounded attribute '{a_name}' from {src}.")
            continue
        if not isinstance(a_spec, Mapping):
            continue
        reg_attr = schema.attributes[a_name]
        exp_req = bool(a_spec.get("required", False))
        if reg_attr.required != exp_req:
            errors.append(f"ONNX op '{op_name}' attribute '{a_name}' required={reg_attr.required} != {src} {exp_req}.")
        try:
            gt_type = normalize_attribute_type(str(a_spec.get("type", "Any")))
            if normalize_attribute_type(reg_attr.type) != gt_type and gt_type.value != "Any":
                errors.append(
                    f"ONNX op '{op_name}' attribute '{a_name}' type '{reg_attr.type.value}' != {src} '{gt_type.value}'."
                )
        except ValueError:
            errors.append(f"ONNX op '{op_name}' attribute '{a_name}' has invalid snapshot type in {src}.")
    if exact_version:
        errors.extend(
            f"ONNX op '{op_name}' has ungrounded attribute '{extra}' not in {src}."
            for extra in sorted(set(schema.attributes) - set(gt_attrs))
        )
    return errors


def _compare_onnx_op_against_spec(op_name: str, schema: OpSchema, gt: Mapping[str, JsonValue], src: str) -> list[str]:
    """Verify domain, version, inputs, outputs, and attributes of an ONNX OpSchema against ground truth.

    Args:
        op_name (str): Operator name.
        schema (OpSchema): Registered ONNX schema.
        gt (Mapping[str, JsonValue]): Ground-truth operator specification.
        src (str): Snapshot filename label.

    Returns:
        list[str]: Mismatch error diagnostics.
    """
    errors: list[str] = []
    gt_domain = str(gt.get("domain", "ai.onnx"))
    if schema.domain != gt_domain:
        errors.append(f"ONNX op '{op_name}' domain '{schema.domain}' != {src} domain '{gt_domain}'.")
    raw_gt_ver = gt.get("version")
    gt_ver = raw_gt_ver if isinstance(raw_gt_ver, int) and not isinstance(raw_gt_ver, bool) else 1
    exact = bool(gt.get("exact_version"))
    if (exact and schema.version != gt_ver) or (not exact and schema.version < gt_ver):
        errors.append(f"ONNX op '{op_name}' version {schema.version} incompatible with {src} version {gt_ver}.")
    if schema.inputs != gt.get("inputs", []):
        errors.append(f"ONNX op '{op_name}' inputs {schema.inputs} != {src} inputs {gt.get('inputs')}.")
    if schema.outputs != gt.get("outputs", []):
        errors.append(f"ONNX op '{op_name}' outputs {schema.outputs} != {src} outputs {gt.get('outputs')}.")
    gt_attrs = gt.get("attributes")
    if isinstance(gt_attrs, Mapping):
        errors.extend(_compare_onnx_attributes(op_name, schema, gt_attrs, exact, src))
    return errors


def _supplement_onnx_defs_specs(
    snap_specs: Mapping[str, dict[str, JsonValue]],
    is_partial: bool,
) -> dict[str, dict[str, JsonValue]]:
    """Supplement partial ONNX category snapshots with canonical ``onnx.defs`` schemas.

    Args:
        snap_specs (Mapping[str, dict[str, JsonValue]]): Specs parsed directly from ``onnx_*.json``.
        is_partial (bool): Whether the JSON file was a partial category summary.

    Returns:
        dict[str, dict[str, JsonValue]]: Complete ground-truth spec dictionary.
    """
    full_specs = dict(snap_specs)
    if not is_partial:
        return full_specs
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.generate_registry import extract_onnx_from_defs

    for def_name, def_data in extract_onnx_from_defs().items():
        if def_name not in full_specs:
            raw_in = def_data.get("inputs")
            raw_out = def_data.get("outputs")
            raw_attrs = def_data.get("attributes")
            raw_def_ver = def_data.get("version")
            def_ver = raw_def_ver if isinstance(raw_def_ver, int) and not isinstance(raw_def_ver, bool) else 1
            in_json: list[JsonValue] = [str(x) for x in raw_in] if isinstance(raw_in, list) else []
            out_json: list[JsonValue] = [str(x) for x in raw_out] if isinstance(raw_out, list) else []
            attr_json: dict[str, JsonValue] = (
                {str(k): dict(v) for k, v in raw_attrs.items()} if isinstance(raw_attrs, dict) else {}
            )
            full_specs[def_name] = {
                "name": def_name,
                "domain": str(def_data.get("domain", "ai.onnx")),
                "version": def_ver,
                "exact_version": True,
                "inputs": in_json,
                "outputs": out_json,
                "attributes": attr_json,
            }
    return full_specs


def _verify_onnx_structural() -> list[str]:
    """Verify structural invariants of ``ONNX_REGISTRY``.

    Returns:
        list[str]: Structural error diagnostics.
    """
    if not ONNX_REGISTRY:
        return ["ONNX registry is empty."]
    errors: list[str] = []
    for op_name, schema in ONNX_REGISTRY.items():
        if schema.name != op_name:
            errors.append(f"ONNX schema name mismatch: key '{op_name}' != schema.name '{schema.name}'.")
        if not (schema.domain == "ai.onnx" or schema.domain.startswith("ai.onnx.")):
            errors.append(f"ONNX op '{op_name}' has unexpected domain '{schema.domain}'.")
    return errors


def verify_onnx_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify ONNX registry schemas structurally and cross-verify against ``onnx_*.json`` snapshots.

    Args:
        snapshots_dir (Path | None): Optional directory containing ``onnx_*.json`` manifests.
        strict (bool): Fail if no ``onnx_*.json`` manifest exists in ``snapshots_dir``.

    Returns:
        list[str]: List of diagnostic errors for ONNX operators.
    """
    errors = _verify_onnx_structural()
    if not ONNX_REGISTRY or snapshots_dir is None:
        return errors

    candidates = [p for p in sorted(snapshots_dir.glob("onnx*.json")) if p.name != "onnxruntime.json"]
    if not candidates:
        return errors + _missing_snapshot_error("ONNX", "onnx_*.json", snapshots_dir, strict)

    onnx_file = candidates[-1]
    snap_specs, is_partial = _load_onnx_ground_truth(onnx_file)
    for snap_op_name, snap_gt in snap_specs.items():
        if snap_op_name not in ONNX_REGISTRY:
            errors.append(f"ONNX snapshot op '{snap_op_name}' in {onnx_file.name} missing from ONNX_REGISTRY.")
        else:
            errors.extend(
                _compare_onnx_op_against_spec(snap_op_name, ONNX_REGISTRY[snap_op_name], snap_gt, onnx_file.name)
            )

    full_specs = _supplement_onnx_defs_specs(snap_specs, is_partial)
    for op_name, schema in ONNX_REGISTRY.items():
        if op_name not in full_specs:
            errors.append(f"ONNX op '{op_name}' is not grounded in {onnx_file.name} snapshot.")
        elif op_name not in snap_specs:
            errors.extend(_compare_onnx_op_against_spec(op_name, schema, full_specs[op_name], "onnx.defs"))
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


_CANONICAL_CORE_MLIR_OPS: frozenset[str] = frozenset(
    {
        "arith.constant",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.cmpf",
        "arith.cmpi",
        "math.exp",
        "math.exp2",
        "math.log",
        "math.sqrt",
        "math.rsqrt",
        "math.sin",
        "math.cos",
        "math.tanh",
        "math.erf",
        "math.absf",
        "tensor.empty",
        "tensor.extract",
        "tensor.insert",
        "tensor.cast",
        "tensor.reshape",
        "tensor.dim",
        "tensor.expand_shape",
        "tensor.collapse_shape",
        "linalg.matmul",
        "linalg.conv2d",
        "linalg.batch_matmul",
        "linalg.dot",
        "linalg.generic",
        "linalg.fill",
        "scf.for",
        "scf.if",
        "scf.yield",
        "scf.while",
        "scf.condition",
        "func.func",
        "func.call",
        "func.return",
    }
)


def verify_mlir_grounding(snapshots_dir: Path, *, strict: bool = False) -> list[str]:
    """Verify MLIR dialect operations against mlir snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.
        strict (bool): Fail if ``mlir*.json`` is absent.

    Returns:
        list[str]: List of diagnostic errors for ungrounded MLIR operations.
    """
    allowed_mlir_domains = ("mlir", *(f"mlir.{d}" for d in CORE_MLIR_DIALECTS), *CORE_MLIR_DIALECTS.keys())
    errors = _verify_registry_structural_integrity(MLIR_REGISTRY, "MLIR_REGISTRY", allowed_mlir_domains, "MLIR")
    mlir_file = snapshots_dir / "mlir_v0.4.30.json"
    if not mlir_file.is_file():
        candidates = [p for p in sorted(snapshots_dir.glob("mlir*.json")) if not p.name.endswith("_exhaustive.json")]
        if not candidates:
            return errors + _missing_snapshot_error("MLIR", "mlir_*.json", snapshots_dir, strict)
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
                    errors.append(
                        f"MLIR op '{op_name}' attribute '{attr_name}' is not grounded "
                        f"in {mlir_file.name} snapshot attributes {list(known_attrs)}."
                    )
        elif known_symbols and op_name not in _CANONICAL_CORE_MLIR_OPS:
            errors.append(f"MLIR op '{op_name}' is not grounded in {mlir_file.name} or core MLIR dialect specs.")

    return errors


def _locate_package_module_file(mod_path: str) -> Path | None:
    """Map a dotted ``ml_switcheroo_ir`` module path to its ``.py`` file under ``src/``.

    Args:
        mod_path (str): Dotted Python module path (e.g. ``'ml_switcheroo_ir.graph'``).

    Returns:
        Path | None: Resolved source ``.py`` file if it exists under ``src/``, else None.
    """
    src_root = Path(__file__).resolve().parent.parent / "src"
    parts = mod_path.split(".")
    if not parts or parts[0] != "ml_switcheroo_ir":
        return None
    pkg_init = src_root.joinpath(*parts, "__init__.py")
    if pkg_init.is_file():
        return pkg_init
    mod_file = src_root.joinpath(*parts[:-1], f"{parts[-1]}.py")
    if mod_file.is_file():
        return mod_file
    return None


def _stmt_defines_symbol(stmt: ast.stmt, symbol_name: str) -> bool:
    """Return True if a top-level or class-body statement directly defines ``symbol_name``.

    Args:
        stmt (ast.stmt): AST statement node.
        symbol_name (str): Target identifier name.

    Returns:
        bool: True if ``stmt`` binds ``symbol_name``.
    """
    if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return stmt.name == symbol_name
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id == symbol_name
    if isinstance(stmt, ast.Assign):
        return any(isinstance(tgt, ast.Name) and tgt.id == symbol_name for tgt in stmt.targets)
    return False


def _resolve_ast_symbol_in_module(
    mod_path: str,
    symbol_name: str,
    visited: set[tuple[str, str]] | None = None,
) -> tuple[str, ast.AST] | None:
    """Statically resolve a symbol inside ``mod_path`` via AST parsing and ``from ... import`` traversal.

    Args:
        mod_path (str): Dotted module path inside ``ml_switcheroo_ir``.
        symbol_name (str): Target class, function, or attribute identifier.
        visited (set[tuple[str, str]] | None): Cycle-detection set of visited ``(mod_path, symbol_name)`` pairs.

    Returns:
        tuple[str, ast.AST] | None: Defining ``(mod_path, ast_node)`` if found, else None.
    """
    seen = visited if visited is not None else set()
    if (mod_path, symbol_name) in seen:
        return None
    seen.add((mod_path, symbol_name))

    py_file = _locate_package_module_file(mod_path)
    if py_file is None:
        return None
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

    for stmt in tree.body:
        if _stmt_defines_symbol(stmt, symbol_name):
            return mod_path, stmt
        if isinstance(stmt, ast.ImportFrom) and stmt.module:
            for alias in stmt.names:
                if alias.name == "*":
                    star_hit = _resolve_ast_symbol_in_module(stmt.module, symbol_name, seen)
                    if star_hit is not None:
                        return star_hit
                elif (alias.asname or alias.name) == symbol_name:
                    return _resolve_ast_symbol_in_module(stmt.module, alias.name, seen)
    return None


_PYDANTIC_BASE_METHODS: frozenset[str] = frozenset(
    {"model_dump", "model_validate", "model_dump_json", "model_validate_json", "model_json_schema"}
)


def _class_defines_member_ast(
    defining_mod: str,
    cls_node: ast.ClassDef,
    member_name: str,
    visited: set[tuple[str, str]] | None = None,
) -> bool:
    """Check via AST whether ``cls_node`` (or an in-package base class) defines ``member_name``.

    Args:
        defining_mod (str): Module path where ``cls_node`` is defined.
        cls_node (ast.ClassDef): Target class AST node.
        member_name (str): Method, property, or attribute name to look up.
        visited (set[tuple[str, str]] | None): Cycle guard for base class resolution.

    Returns:
        bool: True if ``member_name`` is defined on ``cls_node`` or its bases.
    """
    if any(_stmt_defines_symbol(stmt, member_name) for stmt in cls_node.body):
        return True

    seen = visited if visited is not None else set()
    for base in cls_node.bases:
        base_name = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else None)
        if base_name in ("BaseModel", "Model") and member_name in _PYDANTIC_BASE_METHODS:
            return True
        if base_name is not None:
            resolved_base = _resolve_ast_symbol_in_module(defining_mod, base_name, seen)
            if resolved_base is not None and isinstance(resolved_base[1], ast.ClassDef):
                if _class_defines_member_ast(resolved_base[0], resolved_base[1], member_name, seen):
                    return True
    return False


def _resolve_ir_symbol(mod_path: str, obj_name: str, allow_nested: bool) -> bool:
    """Statically resolve a symbol in ``src/ml_switcheroo_ir/`` (or nested parent class) via AST.

    Args:
        mod_path (str): Module path or dotted parent class path.
        obj_name (str): Symbol name to resolve.
        allow_nested (bool): Whether to fall back to resolving parent class members.

    Returns:
        bool: True if the symbol is statically resolvable in ``src/ml_switcheroo_ir/``, False otherwise.
    """
    if _resolve_ast_symbol_in_module(mod_path, obj_name) is not None:
        return True
    if allow_nested and "." in mod_path:
        parent_mod_path, parent_cls = mod_path.rsplit(".", 1)
        resolved_parent = _resolve_ast_symbol_in_module(parent_mod_path, parent_cls)
        if resolved_parent is not None and isinstance(resolved_parent[1], ast.ClassDef):
            return _class_defines_member_ast(resolved_parent[0], resolved_parent[1], obj_name)
    return False


def _verify_ir_category_entries(
    entries: list[dict[str, str]],
    default_kind: str,
) -> list[str]:
    """Verify a list of IR snapshot category entries against static AST symbols.

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


def verify_ir_snapshot_grounding(snapshots_dir: Path, *, strict: bool = False) -> list[str]:
    """Verify ir_v*.json snapshot matches static AST API definitions.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.
        strict (bool): Fail if ``ir_v*.json`` is absent.

    Returns:
        list[str]: List of diagnostic errors for mismatched public APIs.
    """
    ir_file = snapshots_dir / "ir_v0.0.6.json"
    if not ir_file.is_file():
        ir_file = snapshots_dir / "ir_v0.0.4.json"
    if not ir_file.is_file():
        candidates = sorted(snapshots_dir.glob("ir_v*.json"))
        if not candidates:
            return _missing_snapshot_error("IR", "ir_v*.json", snapshots_dir, strict)
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


_GFX11_ARCHITECTURAL_RDNA_SUPPLEMENT: frozenset[str] = frozenset(
    {
        *RDNA3_VOPD_OPERATORS.keys(),
        "V_DOT2_F32_F16",
        "V_DOT4_I32_I8",
        "V_CNDMASK_B32",
    }
)


def _extract_rdna_snapshot_records(
    rdna_file: Path,
) -> tuple[dict[str, dict[str, JsonValue]], list[str], bool]:
    """Load RDNA instruction records keyed by uppercase mnemonic and validate snapshot fields.

    Args:
        rdna_file (Path): Path to ``amd_rdna_*.json`` or ``rdna_ops.json``.

    Returns:
        tuple[dict[str, dict[str, JsonValue]], list[str], bool]:
            ``(upper_mnemonic_map, structural_errors, is_partial_tablegen_sample)``.
    """
    with open(rdna_file, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}, [], False

    errors: list[str] = []
    upper_map: dict[str, dict[str, JsonValue]] = {}
    for inst in _collect_manifest_records(data):
        name = inst.get("name") or inst.get("mnemonic")
        if not isinstance(name, str) or not name:
            errors.append(f"RDNA instruction missing name/mnemonic in {rdna_file.name}.")
            continue
        vopd_slot = inst.get("vopd_slot")
        if vopd_slot is not None and vopd_slot not in ("X", "Y", "BOTH"):
            errors.append(f"RDNA instruction '{name}' has invalid vopd_slot '{vopd_slot}'.")
        upper_key = name.upper()
        upper_map[upper_key] = inst
        if upper_key == "V_DOT4C_I32_I8":
            upper_map["V_DOT4_I32_I8"] = inst
    return upper_map, errors, data.get("source_type") == "tablegen"


def _verify_rdna_vopd_tables() -> list[str]:
    """Verify internal consistency of ``RDNA_VOPD_SLOTS`` and ``RDNA_TO_VOPD_MAP`` against ``RDNA_REGISTRY``.

    Returns:
        list[str]: Structural consistency errors.
    """
    errors: list[str] = []
    for slot_op, slot_val in RDNA_VOPD_SLOTS.items():
        if slot_op not in RDNA_REGISTRY:
            errors.append(f"RDNA_VOPD_SLOTS mnemonic '{slot_op}' is missing from RDNA_REGISTRY.")
        if slot_val not in ("X", "Y", "BOTH"):
            errors.append(f"RDNA_VOPD_SLOTS['{slot_op}'] has invalid slot '{slot_val}'.")
    for src_op, dst_op in RDNA_TO_VOPD_MAP.items():
        if src_op not in RDNA_REGISTRY:
            errors.append(f"RDNA_TO_VOPD_MAP source '{src_op}' is missing from RDNA_REGISTRY.")
        if dst_op not in RDNA_REGISTRY or dst_op not in RDNA_VOPD_SLOTS:
            errors.append(f"RDNA_TO_VOPD_MAP target '{dst_op}' is missing from RDNA_REGISTRY / RDNA_VOPD_SLOTS.")
    return errors


def _check_rdna_mnemonic_entry(
    mnemonic: str,
    upper_map: Mapping[str, dict[str, JsonValue]],
    is_partial_sample: bool,
    file_name: str,
) -> list[str]:
    """Verify a single registered RDNA mnemonic and VOPD metadata against loaded snapshot records.

    Args:
        mnemonic (str): Mnemonic key in ``RDNA_REGISTRY``.
        upper_map (Mapping[str, dict[str, JsonValue]]): Uppercase-keyed snapshot records.
        is_partial_sample (bool): Whether snapshot is a partial TableGen sample.
        file_name (str): Snapshot filename.

    Returns:
        list[str]: Mismatch or ungrounded errors.
    """
    upper_m = mnemonic.upper()
    inst = upper_map.get(upper_m)
    if inst is None:
        if is_partial_sample and upper_m in _GFX11_ARCHITECTURAL_RDNA_SUPPLEMENT:
            return []
        return [f"RDNA mnemonic '{mnemonic}' is not grounded in {file_name}."]
    errors: list[str] = []
    snap_slot = inst.get("vopd_slot")
    if isinstance(snap_slot, str) and mnemonic in RDNA_VOPD_SLOTS and RDNA_VOPD_SLOTS[mnemonic] != snap_slot:
        errors.append(f"RDNA VOPD slot mismatch for '{mnemonic}' vs {file_name} ('{snap_slot}').")
    snap_tgt = inst.get("vopd_target")
    if isinstance(snap_tgt, str) and mnemonic in RDNA_TO_VOPD_MAP and RDNA_TO_VOPD_MAP[mnemonic] != snap_tgt:
        errors.append(f"RDNA VOPD target mismatch for '{mnemonic}' vs {file_name} ('{snap_tgt}').")
    return errors


def verify_rdna_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify RDNA mnemonics, ``RDNA_VOPD_SLOTS``, and ``RDNA_TO_VOPD_MAP`` against ``amd_rdna_*.json``.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.
        strict (bool): Fail if ``*rdna*.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for ungrounded RDNA instructions or VOPD mappings.
    """
    errors = _verify_registry_structural_integrity(RDNA_REGISTRY, "RDNA_REGISTRY", ("amd_rdna", "rdna"), "RDNA")
    errors.extend(_verify_rdna_vopd_tables())
    if snapshots_dir is None:
        return errors

    candidates = [p for p in sorted(snapshots_dir.glob("*rdna*.json")) if not p.name.endswith("_exhaustive.json")]
    candidates = candidates or sorted(snapshots_dir.glob("*rdna*.json"))
    if not candidates:
        return errors + _missing_snapshot_error("RDNA", "amd_rdna_*.json", snapshots_dir, strict)

    rdna_file = candidates[-1]
    upper_map, snap_errs, is_partial_sample = _extract_rdna_snapshot_records(rdna_file)
    errors.extend(snap_errs)
    if upper_map:
        for mnemonic in RDNA_REGISTRY:
            errors.extend(_check_rdna_mnemonic_entry(mnemonic, upper_map, is_partial_sample, rdna_file.name))
    return errors


def _extract_sass_snapshot_records(sass_file: Path) -> tuple[dict[str, dict[str, JsonValue]], list[str]]:
    """Load SASS instruction records keyed by uppercase mnemonic and validate snapshot fields.

    Args:
        sass_file (Path): Path to ``nvidia_sass_*.json`` or ``sass_ops.json``.

    Returns:
        tuple[dict[str, dict[str, JsonValue]], list[str]]: ``(upper_mnemonic_map, structural_errors)``.
    """
    with open(sass_file, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}, []

    errors: list[str] = []
    upper_map: dict[str, dict[str, JsonValue]] = {}
    for inst in _collect_manifest_records(data):
        name = inst.get("name") or inst.get("mnemonic")
        if not isinstance(name, str) or not name:
            errors.append(f"SASS instruction missing name/mnemonic in {sass_file.name}.")
            continue
        latency = inst.get("execution_latency")
        if latency is not None and (isinstance(latency, bool) or not isinstance(latency, (int, list, tuple))):
            errors.append(f"SASS instruction '{name}' has invalid execution_latency '{latency}'.")
        upper_map[name.upper()] = inst
    return upper_map, errors


def _verify_sass_latency_table() -> list[str]:
    """Validate ``SASS_PIPELINE_LATENCIES`` entries against ``SASS_REGISTRY``.

    Returns:
        list[str]: Structural latency table errors.
    """
    errors: list[str] = []
    for lat_op, lat_val in SASS_PIPELINE_LATENCIES.items():
        if lat_op not in SASS_REGISTRY:
            errors.append(f"SASS_PIPELINE_LATENCIES mnemonic '{lat_op}' is missing from SASS_REGISTRY.")
        if isinstance(lat_val, bool) or lat_val < 1:
            errors.append(f"SASS_PIPELINE_LATENCIES['{lat_op}'] has invalid latency '{lat_val}'.")
    return errors


def verify_sass_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify SASS mnemonics and ``SASS_PIPELINE_LATENCIES`` against ``nvidia_sass_*.json``.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.
        strict (bool): Fail if ``*sass*.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for ungrounded SASS instructions or latencies.
    """
    errors = _verify_registry_structural_integrity(SASS_REGISTRY, "SASS_REGISTRY", ("nvidia_sass", "sass"), "SASS")
    errors.extend(_verify_sass_latency_table())
    if snapshots_dir is None:
        return errors

    candidates = [p for p in sorted(snapshots_dir.glob("*sass*.json")) if not p.name.endswith("_exhaustive.json")]
    candidates = candidates or sorted(snapshots_dir.glob("*sass*.json"))
    if not candidates:
        return errors + _missing_snapshot_error("SASS", "nvidia_sass_*.json", snapshots_dir, strict)

    sass_file = candidates[-1]
    upper_map, snap_errs = _extract_sass_snapshot_records(sass_file)
    errors.extend(snap_errs)
    if not upper_map:
        return errors

    for mnemonic in SASS_REGISTRY:
        inst = upper_map.get(mnemonic.upper())
        if inst is None:
            errors.append(f"SASS mnemonic '{mnemonic}' is not grounded in {sass_file.name}.")
            continue
        snap_lat = inst.get("execution_latency")
        if isinstance(snap_lat, int) and not isinstance(snap_lat, bool) and mnemonic in SASS_PIPELINE_LATENCIES:
            if SASS_PIPELINE_LATENCIES[mnemonic] != snap_lat:
                errors.append(
                    f"SASS latency mismatch for '{mnemonic}': registry={SASS_PIPELINE_LATENCIES[mnemonic]} "
                    f"vs {sass_file.name}={snap_lat}."
                )
    return errors


def _validate_wgsl_ops_payload(data: JsonValue) -> list[str]:
    """Validate optional ``ops`` list entries in a ``wgsl_ops.json`` file.

    Args:
        data (JsonValue): Parsed WGSL JSON payload.

    Returns:
        list[str]: Structural validation errors.
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return errors
    raw_ops = data.get("ops")
    if not isinstance(raw_ops, list):
        return errors
    for op in raw_ops:
        if isinstance(op, dict):
            name = op.get("name")
            if not name:
                errors.append("WGSL op missing name.")
            elif op.get("domain") not in ("wgsl", "webgpu_wgsl", None):
                errors.append(f"WGSL op '{name}' has invalid domain '{op.get('domain')}'.")
    return errors


def _load_wgsl_grounded_symbols(
    snapshots_dir: Path, wgsl_file: Path, is_category_manifest: bool
) -> dict[str, dict[str, JsonValue]]:
    """Load WGSL symbol map and supplement ``fma``/``dot`` math builtins from shader snapshots if needed.

    Args:
        snapshots_dir (Path): Directory containing snapshots.
        wgsl_file (Path): Primary WGSL snapshot file.
        is_category_manifest (bool): Whether ``wgsl_file`` uses category sampling.

    Returns:
        dict[str, dict[str, JsonValue]]: Grounded WGSL symbols.
    """
    snap_symbols = _load_snapshot_symbol_map(wgsl_file)
    if is_category_manifest:
        for aux_glob in ("webgl*.json", "metal*.json"):
            for aux_file in sorted(snapshots_dir.glob(aux_glob)):
                for sym_k, sym_v in _load_snapshot_symbol_map(aux_file).items():
                    if sym_k in ("fma", "dot") and sym_k not in snap_symbols:
                        snap_symbols[sym_k] = sym_v
    return snap_symbols


def verify_wgsl_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify every primitive in ``WGSL_REGISTRY`` and ``WGSL_PRIMITIVE_SIGNATURES`` against ``wgsl_*.json``.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.
        strict (bool): Fail if ``wgsl*.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for ungrounded WGSL operations.
    """
    errors = _verify_registry_structural_integrity(WGSL_REGISTRY, "WGSL_REGISTRY", ("webgpu_wgsl", "wgsl"), "WGSL")
    for prim_name, prim_sig in WGSL_PRIMITIVE_SIGNATURES.items():
        if prim_name not in WGSL_REGISTRY:
            errors.append(f"WGSL primitive '{prim_name}' missing from WGSL_REGISTRY.")
        elif WGSL_REGISTRY[prim_name].inputs != prim_sig.inputs or WGSL_REGISTRY[prim_name].outputs != prim_sig.outputs:
            errors.append(f"WGSL primitive '{prim_name}' signature mismatch with WGSL_PRIMITIVE_SIGNATURES.")
    if snapshots_dir is None:
        return errors

    candidates = sorted(snapshots_dir.glob("wgsl*.json"))
    if not candidates:
        return errors + _missing_snapshot_error("WGSL", "wgsl_*.json", snapshots_dir, strict)

    wgsl_file = candidates[-1]
    with open(wgsl_file, encoding="utf-8") as f:
        data = json.load(f)
    errors.extend(_validate_wgsl_ops_payload(data))
    snap_symbols = _load_wgsl_grounded_symbols(
        snapshots_dir, wgsl_file, isinstance(data, dict) and "categories" in data
    )
    if snap_symbols:
        errors.extend(
            f"WGSL op '{op_name}' is not grounded in {wgsl_file.name} snapshot."
            for op_name in WGSL_REGISTRY
            if op_name not in snap_symbols and f"wgsl.{op_name}" not in snap_symbols
        )
    return errors


def _find_prefixed_symbol(
    op_name: str,
    expected_domains: tuple[str, ...],
    snap_symbols: Mapping[str, dict[str, JsonValue]],
) -> dict[str, JsonValue] | None:
    """Resolve an operator in ``snap_symbols`` by bare name or ``<domain>.<op_name>``.

    Args:
        op_name (str): Operator name.
        expected_domains (tuple[str, ...]): Candidate domain prefixes.
        snap_symbols (Mapping[str, dict[str, JsonValue]]): Loaded snapshot symbols.

    Returns:
        dict[str, JsonValue] | None: Matched symbol record or None.
    """
    if op_name in snap_symbols:
        return snap_symbols[op_name]
    for dom in expected_domains:
        if f"{dom}.{op_name}" in snap_symbols:
            return snap_symbols[f"{dom}.{op_name}"]
    return None


def _verify_low_level_dialect_grounding(
    registry: Mapping[str, OpSchema],
    registry_name: str,
    expected_domains: tuple[str, ...],
    label: str,
    glob_pattern: str,
    snapshots_dir: Path | None,
    *,
    allow_mnemonic: bool = False,
    strict: bool = False,
) -> list[str]:
    """Verify structural integrity and snapshot cross-grounding for a low-level hardware/shader registry.

    Args:
        registry (Mapping[str, OpSchema]): Target operator registry.
        registry_name (str): Constant name for error reporting.
        expected_domains (tuple[str, ...]): Valid domain identifiers.
        label (str): Human-readable dialect label.
        glob_pattern (str): Snapshot filename glob pattern.
        snapshots_dir (Path | None): Optional snapshots directory.
        allow_mnemonic (bool): Whether ``mnemonic`` counts as an item identifier.
        strict (bool): Fail if no matching snapshot file is found.

    Returns:
        list[str]: Diagnostic error messages.
    """
    errors = _verify_registry_structural_integrity(registry, registry_name, expected_domains, label)
    if snapshots_dir is None:
        return errors

    errors.extend(_verify_category_items_have_identifiers(snapshots_dir, glob_pattern, label, allow_mnemonic))
    candidates = [p for p in sorted(snapshots_dir.glob(glob_pattern)) if not p.name.endswith("_exhaustive.json")]
    candidates = candidates or sorted(snapshots_dir.glob(glob_pattern))
    if not candidates:
        return errors + _missing_snapshot_error(label, glob_pattern, snapshots_dir, strict)

    snap_file = candidates[-1]
    snap_symbols = _load_snapshot_symbol_map(snap_file)
    if not snap_symbols:
        return errors
    for op_name, schema in registry.items():
        sym = _find_prefixed_symbol(op_name, expected_domains, snap_symbols)
        if sym is None:
            errors.append(f"{label} op '{op_name}' is not grounded in {snap_file.name} snapshot.")
            continue
        known_attrs = _extract_known_attributes(sym) if schema.attributes else set()
        if known_attrs:
            errors.extend(
                f"{label} op '{op_name}' attribute '{attr_name}' is not grounded in {snap_file.name}."
                for attr_name in schema.attributes
                if attr_name not in known_attrs
            )
    return errors


def verify_ptx_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify every operator in ``PTX_REGISTRY`` against ``nvidia_ptx_*.json``.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.
        strict (bool): Fail if ``*ptx*.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for ungrounded PTX operations.
    """
    return _verify_low_level_dialect_grounding(
        PTX_REGISTRY,
        "PTX_REGISTRY",
        ("nvidia_ptx", "ptx"),
        "PTX",
        "*ptx*.json",
        snapshots_dir,
        allow_mnemonic=True,
        strict=strict,
    )


def verify_metal_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify every operator in ``METAL_REGISTRY`` against ``metal_*.json``.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.
        strict (bool): Fail if ``*metal*.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for ungrounded Metal operations.
    """
    return _verify_low_level_dialect_grounding(
        METAL_REGISTRY,
        "METAL_REGISTRY",
        ("metal_msl", "metal"),
        "Metal",
        "*metal*.json",
        snapshots_dir,
        strict=strict,
    )


def verify_wasm_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify every operator in ``WASM_REGISTRY`` against ``wasm_simd_*.json``.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.
        strict (bool): Fail if ``*wasm*.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for ungrounded WASM operations.
    """
    return _verify_low_level_dialect_grounding(
        WASM_REGISTRY,
        "WASM_REGISTRY",
        ("wasm_simd", "wasm"),
        "WASM",
        "*wasm*.json",
        snapshots_dir,
        strict=strict,
    )


def verify_webgl_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify every operator in ``WEBGL_REGISTRY`` against ``webgl_*.json``.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.
        strict (bool): Fail if ``*webgl*.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for ungrounded WebGL operations.
    """
    return _verify_low_level_dialect_grounding(
        WEBGL_REGISTRY,
        "WEBGL_REGISTRY",
        ("webgl",),
        "WebGL",
        "*webgl*.json",
        snapshots_dir,
        strict=strict,
    )


_QUANTIZATION_CONCEPT_MAP: dict[str, tuple[str, ...]] = {
    "quantization.block_quantize": ("quantize", "quantization", "mx_fp4", "mxfp8"),
    "quantization.dequantize_grouped_int4": ("dequantize", "quantization"),
}


def verify_quantization_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify quantization operator schemas structurally and against ``concept_map.json`` / ``quantization_*.json``.

    Args:
        snapshots_dir (Path | None): Optional directory containing snapshots.
        strict (bool): Fail if neither ``quantization_*.json`` nor ``concept_map.json`` exists.

    Returns:
        list[str]: Diagnostic error messages for quantization schemas.
    """
    errors = _verify_registry_structural_integrity(
        QUANTIZATION_OPS_REGISTRY,
        "QUANTIZATION_OPS_REGISTRY",
        ("quantization", "ml.switcheroo.quant"),
        "Quantization",
    )
    if snapshots_dir is None:
        return errors

    quant_files = sorted(snapshots_dir.glob("quantization*.json"))
    if quant_files:
        snap_syms = _load_snapshot_symbol_map(quant_files[-1])
        for op_name in QUANTIZATION_OPS_REGISTRY:
            if op_name not in snap_syms:
                errors.append(f"Quantization op '{op_name}' is not grounded in {quant_files[-1].name}.")
        return errors

    cm_file = snapshots_dir / "concept_map.json"
    if not cm_file.is_file():
        return errors + _missing_snapshot_error("Quantization", "concept_map.json", snapshots_dir, strict)

    with open(cm_file, encoding="utf-8") as f:
        cm_data = json.load(f)
    if isinstance(cm_data, Mapping):
        for op_name in QUANTIZATION_OPS_REGISTRY:
            concepts = _QUANTIZATION_CONCEPT_MAP.get(op_name, ())
            if not concepts or not any(c in cm_data for c in concepts):
                errors.append(f"Quantization op '{op_name}' is not grounded in concept_map.json.")
    return errors


_STATE_CONCEPT_MAP: dict[str, tuple[str, ...]] = {
    "ReadVariable": ("memcpy", "gather"),
    "AssignVariable": ("memcpy",),
    "ScatterUpdate": ("scatter",),
}


def verify_state_ops_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify state-mutation operator schemas structurally and against ``state_*.json`` or ``concept_map.json``.

    Args:
        snapshots_dir (Path | None): Optional directory containing snapshots.
        strict (bool): Fail if neither ``state_*.json`` nor ``concept_map.json`` exists.

    Returns:
        list[str]: Diagnostic error messages for state schemas.
    """
    errors = _verify_registry_structural_integrity(
        STATE_OPS_REGISTRY, "STATE_OPS_REGISTRY", ("ml.switcheroo.state",), "State"
    )
    if snapshots_dir is None:
        return errors

    state_files = sorted(snapshots_dir.glob("state*.json"))
    if state_files:
        snap_syms = _load_snapshot_symbol_map(state_files[-1])
        for op_name in STATE_OPS_REGISTRY:
            if op_name not in snap_syms:
                errors.append(f"State op '{op_name}' is not grounded in {state_files[-1].name}.")
        return errors

    cm_file = snapshots_dir / "concept_map.json"
    if not cm_file.is_file():
        return errors + _missing_snapshot_error("State", "concept_map.json", snapshots_dir, strict)

    with open(cm_file, encoding="utf-8") as f:
        cm_data = json.load(f)
    if isinstance(cm_data, Mapping):
        for op_name in STATE_OPS_REGISTRY:
            concepts = _STATE_CONCEPT_MAP.get(op_name, ())
            if not concepts or not any(c in cm_data for c in concepts):
                errors.append(f"State op '{op_name}' is not grounded in concept_map.json.")
    return errors


_ODL_CONCEPT_SYNONYMS: dict[str, str] = {
    "conv2d": "convolution",
}


def _collect_odl_grounded_concepts(snapshots_dir: Path, cm_file: Path) -> set[str]:
    """Collect grounded ODL concept and framework symbol names from ``concept_map.json`` and framework snapshots.

    Args:
        snapshots_dir (Path): Directory containing snapshots.
        cm_file (Path): Path to ``concept_map.json``.

    Returns:
        set[str]: Grounded concept and symbol identifiers.
    """
    with open(cm_file, encoding="utf-8") as f:
        cm_data = json.load(f)
    grounded = set(cm_data.keys()) if isinstance(cm_data, Mapping) else set()
    for glob_pat in ("*array_api*.json", "*aten*.json"):
        for fw_file in sorted(snapshots_dir.glob(glob_pat)):
            grounded.update(_load_snapshot_symbol_map(fw_file).keys())
    return grounded


def verify_odl_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify universal ODL schemas structurally and against ``concept_map.json`` and ``array_api_*.json``.

    Args:
        snapshots_dir (Path | None): Optional directory containing snapshots.
        strict (bool): Fail if ``concept_map.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for ODL schemas.
    """
    errors = _verify_registry_structural_integrity(ODL_CATALOG, "ODL_CATALOG", ("odl",), "ODL")
    errors.extend(
        f"ODL_DIALECT_MAPPINGS key '{mapped_op}' missing from ODL_CATALOG."
        for mapped_op in ODL_DIALECT_MAPPINGS
        if mapped_op not in ODL_CATALOG
    )
    if snapshots_dir is None:
        return errors

    odl_files = sorted(snapshots_dir.glob("odl*.json"))
    if odl_files:
        snap_syms = _load_snapshot_symbol_map(odl_files[-1])
        errors.extend(
            f"ODL op '{op_name}' is not grounded in {odl_files[-1].name}."
            for op_name in ODL_CATALOG
            if op_name not in snap_syms
        )
        return errors

    cm_file = snapshots_dir / "concept_map.json"
    if not cm_file.is_file():
        return errors + _missing_snapshot_error("ODL", "concept_map.json", snapshots_dir, strict)

    grounded_concepts = _collect_odl_grounded_concepts(snapshots_dir, cm_file)
    for op_name in ODL_CATALOG:
        concept_key = _ODL_CONCEPT_SYNONYMS.get(op_name, op_name)
        if concept_key not in grounded_concepts and op_name not in grounded_concepts:
            errors.append(f"ODL op '{op_name}' is not grounded in concept_map.json or framework snapshots.")
    return errors


def _compare_role_table(
    op_name: str,
    snap_roles: Mapping[str, JsonValue],
    eng_roles: Mapping[str, Mapping[str, Sequence[str]]],
) -> list[str]:
    """Compare snapshot parameter translation roles for one operation against ``ParameterTranslationEngine``.

    Args:
        op_name (str): Operation key in ``_parameter_translations``.
        snap_roles (Mapping[str, JsonValue]): Snapshot role mapping from ``concept_map.json``.
        eng_roles (Mapping[str, Mapping[str, Sequence[str]]]): Engine role mapping.

    Returns:
        list[str]: Role/parameter mismatch errors.
    """
    errors: list[str] = []
    for role_name, fw_map in snap_roles.items():
        if role_name not in eng_roles:
            errors.append(
                f"ParameterTranslationEngine op '{op_name}' missing role '{role_name}' from concept_map.json."
            )
            continue
        if not isinstance(fw_map, Mapping):
            continue
        for fw, params in fw_map.items():
            if isinstance(params, list):
                known_fw = eng_roles[role_name].get(str(fw), ())
                errors.extend(
                    f"ParameterTranslationEngine op '{op_name}' role '{role_name}' missing {fw} param '{p}'."
                    for p in params
                    if str(p) not in known_fw
                )
    return errors


def verify_translation_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify ``ParameterTranslationEngine`` rules against ``concept_map.json`` ``_parameter_translations``.

    Args:
        snapshots_dir (Path | None): Optional directory containing ``concept_map.json``.
        strict (bool): Fail if ``concept_map.json`` is absent.

    Returns:
        list[str]: Diagnostic error messages for ungrounded or inconsistent parameter translations.
    """
    engine = ParameterTranslationEngine()
    if not engine.translations:
        return ["ParameterTranslationEngine translations table is empty."]
    errors = [
        f"ParameterTranslationEngine operation '{op_key}' has empty roles."
        for op_key, spec in engine.translations.operations.items()
        if not spec.roles
    ]
    if snapshots_dir is None:
        return errors

    cm_candidate = snapshots_dir / "concept_map.json"
    if not cm_candidate.is_file():
        return errors + _missing_snapshot_error("ParameterTranslationEngine", "concept_map.json", snapshots_dir, strict)

    with open(cm_candidate, encoding="utf-8") as f:
        cm_data = json.load(f)
    if not isinstance(cm_data, Mapping):
        return [*errors, "concept_map.json root must be a JSON object."]

    raw_pt = cm_data.get("_parameter_translations")
    if not isinstance(raw_pt, Mapping):
        return [*errors, "concept_map.json missing '_parameter_translations' section."] if strict else errors

    for op_name, op_entry in raw_pt.items():
        if op_name not in engine.translations:
            errors.append(f"Snapshot parameter translation op '{op_name}' missing from ParameterTranslationEngine.")
        elif isinstance(op_entry, Mapping) and isinstance(op_entry.get("roles"), Mapping):
            errors.extend(_compare_role_table(op_name, op_entry["roles"], engine.translations[op_name].roles))
    return errors


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
        known_params = _extract_known_attributes(sym_record)
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
    *,
    strict: bool = False,
) -> list[str]:
    """Verify a domain operator registry against its structural invariants and optional snapshot.

    Args:
        registry (Mapping[str, OpSchema]): Domain operator registry.
        registry_name (str): Name of the registry constant for error messages.
        expected_domain (str): Expected domain string on each OpSchema.
        label (str): Human-readable domain label.
        glob_pattern (str): Glob pattern to locate snapshot JSON in ``snapshots_dir``.
        snapshots_dir (Path | None): Optional directory containing framework snapshots.
        strict (bool): Fail if no matching snapshot file exists in ``snapshots_dir``.

    Returns:
        list[str]: Diagnostic error messages for ungrounded or malformed schemas.
    """
    errors = _verify_registry_structural_integrity(registry, registry_name, (expected_domain,), label)
    if not registry or snapshots_dir is None:
        return errors

    candidates = sorted(snapshots_dir.glob(glob_pattern))
    if not candidates:
        return errors + _missing_snapshot_error(label, glob_pattern, snapshots_dir, strict)

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


def verify_chex_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify Chex assertion and testing operator schemas in CHEX_REGISTRY.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.
        strict (bool): Fail if ``chex_*.json`` is absent.

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
        strict=strict,
    )


def verify_grain_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify Grain data pipeline operator schemas in GRAIN_REGISTRY.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.
        strict (bool): Fail if ``grain_*.json`` is absent.

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
        strict=strict,
    )


def verify_orbax_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify Orbax checkpointing operator schemas in ORBAX_REGISTRY.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.
        strict (bool): Fail if ``orbax*.json`` is absent.

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
        strict=strict,
    )


def verify_optax_grounding(snapshots_dir: Path | None = None, *, strict: bool = False) -> list[str]:
    """Verify Optax optimizer and loss operator schemas in OPTAX_REGISTRY.

    Args:
        snapshots_dir (Path | None): Optional path to framework snapshots directory.
        strict (bool): Fail if ``optax_*.json`` is absent.

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
        strict=strict,
    )


def _verify_local_schemas_only() -> int:
    """Verify local schemas across all internal registries when external snapshot directories are absent.

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
        + verify_translation_grounding()
    )
    if all_errors:
        for err in all_errors:
            print(f"[ERROR] {err}")
        return 1
    print("[SUCCESS] Local schema definitions are valid and structurally sound.")
    return 0


def _run_all_snapshot_verifiers(snapshots_dir: Path, *, strict: bool = False) -> list[str]:
    """Execute all live snapshot and local schema grounding verifiers.

    Args:
        snapshots_dir (Path): Directory containing ground-truth snapshots.
        strict (bool): Fail whenever any required dialect snapshot file is absent.

    Returns:
        list[str]: Combined list of all diagnostic error messages.
    """
    return (
        verify_stablehlo_grounding(snapshots_dir, strict=strict)
        + verify_mlir_grounding(snapshots_dir, strict=strict)
        + verify_ir_snapshot_grounding(snapshots_dir, strict=strict)
        + verify_custom_ops_grounding(snapshots_dir, strict=strict)
        + verify_onnx_grounding(snapshots_dir, strict=strict)
        + verify_rdna_grounding(snapshots_dir, strict=strict)
        + verify_sass_grounding(snapshots_dir, strict=strict)
        + verify_wgsl_grounding(snapshots_dir, strict=strict)
        + verify_ptx_grounding(snapshots_dir, strict=strict)
        + verify_metal_grounding(snapshots_dir, strict=strict)
        + verify_wasm_grounding(snapshots_dir, strict=strict)
        + verify_webgl_grounding(snapshots_dir, strict=strict)
        + verify_array_api_grounding(snapshots_dir, strict=strict)
        + verify_aten_grounding(snapshots_dir, strict=strict)
        + verify_chex_grounding(snapshots_dir, strict=strict)
        + verify_grain_grounding(snapshots_dir, strict=strict)
        + verify_orbax_grounding(snapshots_dir, strict=strict)
        + verify_optax_grounding(snapshots_dir, strict=strict)
        + verify_collectives_grounding(snapshots_dir, strict=strict)
        + verify_quantization_grounding(snapshots_dir, strict=strict)
        + verify_state_ops_grounding(snapshots_dir, strict=strict)
        + verify_odl_grounding(snapshots_dir, strict=strict)
        + verify_translation_grounding(snapshots_dir, strict=strict)
    )


def _build_grounding_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI ArgumentParser for ``verify_grounding.py``.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
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
        "--require-snapshots-dir",
        action="store_true",
        help="Require resolved snapshots directory to exist and contain snapshot JSON files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing snapshot directories/files or ungrounded symbols strictly as errors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run anti-hallucination grounding verification for all schemas.

    Args:
        argv (list[str] | None): Command-line arguments.

    Returns:
        int: Exit status code (0 for success, non-zero for failure).
    """
    args = _build_grounding_arg_parser().parse_args(argv)
    strict_dir = bool(args.strict or args.require_snapshots_dir)

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
        if strict_dir:
            print("[ERROR] Snapshot directory not found in strict mode.")
            return 1
        print("[WARNING] Snapshot directory not found. Skipping live snapshot checks.")
        return _verify_local_schemas_only()

    if not any(snapshots_dir.glob("*.json")):
        if strict_dir:
            print("[ERROR] Snapshot directory contains no snapshot files in strict mode.")
            return 1
        print("[WARNING] Snapshot directory contains no snapshot files. Skipping live snapshot checks.")
        return _verify_local_schemas_only()

    print(f"Using snapshots directory: {snapshots_dir}")
    all_errors = _run_all_snapshot_verifiers(snapshots_dir, strict=bool(args.strict))

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
        print(f"[FAILURE] Found {len(all_errors)} ungrounded symbol(s) or schema defect(s):")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print("[SUCCESS] All schemas are 100% grounded against ground-truth snapshots!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
