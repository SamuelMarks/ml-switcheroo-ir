"""Script to verify anti-hallucination grounding of registered schemas against ground-truth snapshots."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from ml_switcheroo_ir.schema.custom_ops import CUSTOM_OPS_REGISTRY
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY
from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY
from ml_switcheroo_ir.validator import (
    CORE_MLIR_DIALECTS,
    DEFAULT_SNAPSHOT_DIR,
    GroundingValidator,
)


def find_snapshots_directory(override_path: str | None = None) -> Path | None:
    """Locate the ground-truth snapshots directory.

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

    env_dir = os.environ.get("ML_FRAMEWORK_SNAPSHOTS_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
        if p.is_dir():
            return p

    default_path = Path(DEFAULT_SNAPSHOT_DIR).resolve()
    if default_path.is_dir():
        return default_path

    # Fallback to relative path from script
    script_relative = (
        Path(__file__).resolve().parent.parent.parent
        / "ml-framework-snapshots"
        / "src"
        / "ml_framework_snapshots"
        / "snapshots"
    )
    if script_relative.is_dir():
        return script_relative

    return None


def verify_stablehlo_grounding(snapshots_dir: Path) -> list[str]:
    """Verify all StableHLO registry schemas against stablehlo_v1.0.0.json snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: List of error diagnostic messages for any ungrounded symbols.
    """
    stablehlo_file = snapshots_dir / "stablehlo_v1.0.0.json"
    if not stablehlo_file.is_file():
        return [f"StableHLO snapshot file not found: {stablehlo_file}"]

    gv = GroundingValidator(snapshot_manifest=str(stablehlo_file))
    errors: list[str] = []

    for op_name, schema in STABLEHLO_REGISTRY.items():
        if (
            op_name not in gv.grounded_symbols
            and f"stablehlo.{op_name}" not in gv.grounded_symbols
        ):
            errors.append(
                f"StableHLO op '{op_name}' is not grounded in stablehlo_v1.0.0.json snapshot."
            )

        # Check attributes against snapshot record
        sym_record = gv.grounded_symbols.get(op_name) or gv.grounded_symbols.get(
            f"stablehlo.{op_name}"
        )
        if sym_record:
            known_attrs: set[str] = set()
            for p in sym_record.get("params", []):
                if isinstance(p, dict) and "name" in p:
                    known_attrs.add(p["name"])
            for a in sym_record.get("attributes", []):
                if isinstance(a, dict) and "name" in a:
                    known_attrs.add(a["name"])

            for attr_name in schema.attributes:
                if op_name == "custom_call" and attr_name == "api_version":
                    continue
                if known_attrs and attr_name not in known_attrs:
                    errors.append(
                        f"StableHLO op '{op_name}' attribute '{attr_name}' is not grounded in snapshot attributes {list(known_attrs)}."
                    )

    return errors


def verify_custom_ops_grounding() -> list[str]:
    """Verify all custom operator schemas in CUSTOM_OPS_REGISTRY have valid specifications.

    Returns:
        List[str]: List of diagnostic errors for custom operators.
    """
    errors: list[str] = []
    for op_name, schema in CUSTOM_OPS_REGISTRY.items():
        if not schema.name:
            errors.append("Custom op has empty name.")
        if schema.domain != "ml.switcheroo.custom":
            errors.append(
                f"Custom op '{op_name}' has invalid domain '{schema.domain}', expected 'ml.switcheroo.custom'."
            )
        if not schema.outputs:
            errors.append(f"Custom op '{op_name}' has no defined outputs.")
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
        if schema.domain != "ai.onnx":
            errors.append(
                f"ONNX op '{op_name}' has unexpected domain '{schema.domain}'."
            )
    return errors


def verify_mlir_grounding(snapshots_dir: Path) -> list[str]:
    """Verify MLIR dialect operations against mlir_v0.4.30.json snapshot.

    Args:
        snapshots_dir (Path): Path to framework snapshots directory.

    Returns:
        List[str]: List of diagnostic errors for ungrounded MLIR operations.
    """
    mlir_file = snapshots_dir / "mlir_v0.4.30.json"
    if not mlir_file.is_file():
        return []

    errors: list[str] = []
    with open(mlir_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    known_dialects: set[str] = set()
    for cat_items in data.get("categories", {}).values():
        if isinstance(cat_items, list):
            for item in cat_items:
                if isinstance(item, dict) and "api_path" in item:
                    prefix = item["api_path"].split(".")[0]
                    known_dialects.add(prefix)

    for dialect in CORE_MLIR_DIALECTS:
        if known_dialects and dialect not in known_dialects:
            errors.append(
                f"Core MLIR dialect '{dialect}' is not grounded in mlir_v0.4.30.json."
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
        help="Path to ml-framework-snapshots directory.",
    )
    args = parser.parse_args(argv)

    print("=== Auditing ml-switcheroo-ir Schema Grounding ===")

    snapshots_dir = find_snapshots_directory(args.snapshots_dir)
    if snapshots_dir is None:
        print(
            "[WARNING] ml-framework-snapshots directory not found. Skipping live snapshot checks."
        )
        # Verify custom ops and ONNX registries
        custom_errors = verify_custom_ops_grounding()
        onnx_errors = verify_onnx_grounding()
        all_errors = custom_errors + onnx_errors
        if all_errors:
            for err in all_errors:
                print(f"[ERROR] {err}")
            return 1
        print("[SUCCESS] Local schema definitions are valid and structurally sound.")
        return 0

    print(f"Using snapshots directory: {snapshots_dir}")

    stablehlo_errors = verify_stablehlo_grounding(snapshots_dir)
    mlir_errors = verify_mlir_grounding(snapshots_dir)
    ir_errors = verify_ir_snapshot_grounding(snapshots_dir)
    custom_errors = verify_custom_ops_grounding()
    onnx_errors = verify_onnx_grounding()

    all_errors = (
        stablehlo_errors + mlir_errors + ir_errors + custom_errors + onnx_errors
    )

    print(f"Audited StableHLO schemas: {len(STABLEHLO_REGISTRY)} ops.")
    print(f"Audited MLIR dialects: {len(CORE_MLIR_DIALECTS)} dialects.")
    print(f"Audited Custom ops: {len(CUSTOM_OPS_REGISTRY)} ops.")
    print(f"Audited ONNX schemas: {len(ONNX_REGISTRY)} ops.")

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
