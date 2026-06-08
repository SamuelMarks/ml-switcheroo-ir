"""Module for checking compliance of ML implementations against the IR and dialect."""

import ast
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set

from tabulate import tabulate

from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY


def collect_files(targets: List[str]) -> List[str]:
    """Collect Python and JSON files from a list of target paths.

    Args:
        targets: List of file or directory paths.

    Returns:
        List of absolute file paths to process.
    """
    collected_files = []
    for target in targets:
        if not os.path.exists(target):
            print(f"Error: Target path '{target}' not found.")
            sys.exit(1)

        if os.path.isfile(target):
            if target.endswith(".py") or target.endswith(".json"):
                collected_files.append(os.path.abspath(target))
        else:
            for root, dirs, files in os.walk(target):
                if "node_modules" in dirs:
                    dirs.remove("node_modules")
                if ".venv" in dirs:
                    dirs.remove(".venv")
                for file in files:
                    if (
                        (file.endswith(".py") or file.endswith(".json"))
                        and not file.startswith("test_")
                        and file != "__init__.py"
                    ):
                        collected_files.append(os.path.join(root, file))
    return collected_files


def get_dialect_ops() -> Set[str]:
    """Get the set of all operator names in the ONNX dialect registry.

    Returns:
        Set of operator names.
    """
    return {k.split(".")[-1] for k in ONNX_REGISTRY.keys()}


def parse_json_file(filepath: str, dialect_ops: Set[str]) -> Set[str]:
    """Extract implemented dialect operations from a JSON file.

    Args:
        filepath: Path to the JSON file.
        dialect_ops: Set of all dialect operations.

    Returns:
        Set of implemented operation names found in the JSON file keys.
    """
    implemented = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                for k in data.keys():
                    if k in dialect_ops:
                        implemented.add(k)
    except Exception:
        pass
    return implemented


def extract_dynamic_definitions(
    tree: ast.AST, filepath: str, dialect_ops: Set[str]
) -> Set[str]:
    """Extract dynamically loaded JSON definitions based on load_definitions call.

    Args:
        tree: The AST tree of the python file.
        filepath: The path to the python file.
        dialect_ops: Set of all dialect operations.

    Returns:
        Set of implemented operation names.
    """
    implemented = set()
    loaded_definitions_fw = None

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "load_definitions"
        ):
            if node.args and isinstance(node.args[0], ast.Constant):
                loaded_definitions_fw = node.args[0].value

    if loaded_definitions_fw:
        # Try adjacent or nearby definitions/ folder
        base_dir = os.path.dirname(filepath)
        possible_paths = [
            os.path.join(base_dir, f"{loaded_definitions_fw}.json"),
            os.path.join(base_dir, "definitions", f"{loaded_definitions_fw}.json"),
            os.path.join(
                os.path.dirname(base_dir),
                "definitions",
                f"{loaded_definitions_fw}.json",
            ),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                implemented.update(parse_json_file(p, dialect_ops))
                break

    return implemented


def analyze_python_file(filepath: str, dialect_ops: Set[str]) -> Dict[str, Any]:
    """Analyze a python file for IR, FrameworkAdapter, and Dialect compliance.

    Args:
        filepath: Path to the python file.
        dialect_ops: Set of all dialect operations.

    Returns:
        Dictionary with compliance findings.
    """
    result: Dict[str, Any] = {
        "is_ir": False,
        "is_fw": False,
        "implemented_ops": set(),
        "ir_score": (0, 5),
        "fw_score": (0, 4),
    }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return result

    # Check for IR
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if getattr(base, "id", "") in ("CompilerBackend", "GraphFrontend"):
                    result["is_ir"] = True
                    break

    # Check for FrameworkAdapter
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_adapter_by_name = "Adapter" in node.name or "Importer" in node.name
            is_adapter_by_decorator = any(
                isinstance(dec, ast.Call)
                and getattr(dec.func, "id", "") == "register_framework"
                for dec in node.decorator_list
            )
            if is_adapter_by_name or is_adapter_by_decorator:
                result["is_fw"] = True
                break

    # Check for Dialect
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in dialect_ops:
                result["implemented_ops"].add(node.name)

            # Check for @register_op("framework", "op_name")
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and getattr(dec.func, "id", "") == "register_op"
                ):
                    if len(dec.args) >= 2 and isinstance(dec.args[1], ast.Constant):
                        op_name = dec.args[1].value
                        # Try exact match or capitalized match
                        if op_name in dialect_ops:
                            result["implemented_ops"].add(op_name)
                        elif str(op_name).capitalize() in dialect_ops:
                            result["implemented_ops"].add(str(op_name).capitalize())

        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if key.value in dialect_ops:
                        result["implemented_ops"].add(key.value)

    result["implemented_ops"].update(
        extract_dynamic_definitions(tree, filepath, dialect_ops)
    )

    if result["is_ir"]:
        best_req_met = 2
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                is_backend = any(
                    getattr(b, "id", "") == "CompilerBackend" for b in node.bases
                )
                is_frontend = any(
                    getattr(b, "id", "") == "GraphFrontend" for b in node.bases
                )
                if is_backend or is_frontend:
                    current_met = 3
                    target_method = "compile" if is_backend else "parse_to_graph"
                    target_arg = "graph" if is_backend else "code"
                    for item in node.body:
                        if (
                            isinstance(item, ast.FunctionDef)
                            and item.name == target_method
                        ):
                            current_met += 1
                            arg_names = [arg.arg for arg in item.args.args]
                            if target_arg in arg_names:
                                current_met += 1
                            break
                    best_req_met = max(best_req_met, current_met)
        result["ir_score"] = (best_req_met, 5)

    if result["is_fw"]:
        best_req_met = 2
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and (
                "Adapter" in node.name
                or "Importer" in node.name
                or any(
                    isinstance(d, ast.Call)
                    and getattr(d.func, "id", "") == "register_framework"
                    for d in node.decorator_list
                )
            ):
                current_met = 2
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                if (
                    "convert" in methods
                    or "import_func" in methods
                    or "parse" in methods
                ):
                    current_met += 1
                if "definitions" in methods or "import_jaxpr" in methods:
                    current_met += 1
                best_req_met = max(best_req_met, current_met)
        result["fw_score"] = (best_req_met, 4)

    return result


def run_compliance_check(
    targets: List[str], verbose: bool = False, mapping_file: Optional[str] = None
) -> None:
    """Run the compliance check on the specified targets and print reports.

    Args:
        targets: List of file or directory paths to check.
        verbose: Whether to print detailed missing operation schemas.
        mapping_file: Optional path to a mapping JSON file to report implemented APIs.
    """
    if not targets:
        print("Error: No targets provided.")
        sys.exit(1)

    files = collect_files(targets)
    if not files:
        print("No Python or JSON files found in the specified targets.")
        # sys.exit(1) - we allow it to continue to print empty report

    dialect_ops = get_dialect_ops()
    total_dialect_ops = len(dialect_ops)

    best_ir_score = (0, 5)
    best_fw_score = (0, 4)
    found_any_ir = False
    found_any_fw = False

    aggregated_dialect_ops = set()
    found_any_dialect_targets = False

    for filepath in files:
        if filepath.endswith(".json"):
            ops = parse_json_file(filepath, dialect_ops)
            if ops:
                aggregated_dialect_ops.update(ops)
                found_any_dialect_targets = True
            else:
                _ = True
            continue

        res = analyze_python_file(filepath, dialect_ops)

        if res["is_ir"]:
            found_any_ir = True
            if res["ir_score"][0] > best_ir_score[0]:
                best_ir_score = res["ir_score"]

        if res["is_fw"]:
            found_any_fw = True
            if res["fw_score"][0] > best_fw_score[0]:
                best_fw_score = res["fw_score"]

        if res["implemented_ops"]:
            aggregated_dialect_ops.update(res["implemented_ops"])
            found_any_dialect_targets = True

        if (
            not res["is_ir"]
            and not res["is_fw"]
            and filepath in [os.path.abspath(t) for t in targets if os.path.isfile(t)]
        ):
            found_any_dialect_targets = True

    print("\nCompliance Report")
    print("=================")

    # Determine aggregate label
    if len(targets) == 1 and os.path.isfile(targets[0]):
        target_label = os.path.basename(targets[0])
    elif len(targets) == 1:
        target_label = os.path.basename(os.path.abspath(targets[0])) + " (Directory)"
    else:
        target_label = f"Multiple Targets ({len(targets)})"

    if found_any_ir:
        pct = (best_ir_score[0] / best_ir_score[1]) * 100
        status = "PASS" if pct == 100 else "FAIL"
        report = [
            [
                target_label,
                f"{pct:.0f}%",
                f"{best_ir_score[0]}/{best_ir_score[1]}",
                status,
            ]
        ]
        print("\nIR Compliance (CompilerBackend / GraphFrontend):")
        print(
            tabulate(report, headers=["Target", "Compliance %", "Reqs Met", "Status"])
        )

    if found_any_fw:
        pct = (best_fw_score[0] / best_fw_score[1]) * 100
        status = "PASS" if pct == 100 else "FAIL"
        report = [
            [
                target_label,
                f"{pct:.0f}%",
                f"{best_fw_score[0]}/{best_fw_score[1]}",
                status,
            ]
        ]
        print("\nFrameworkAdapter Compliance:")
        print(
            tabulate(report, headers=["Target", "Compliance %", "Reqs Met", "Status"])
        )

    if found_any_dialect_targets:
        pct = (len(aggregated_dialect_ops) / total_dialect_ops) * 100
        status = "PASS" if pct == 100 else "FAIL"

        dialect_report = [
            [
                target_label,
                f"{pct:.1f}%",
                f"{len(aggregated_dialect_ops)}/{total_dialect_ops}",
                status,
                total_dialect_ops - len(aggregated_dialect_ops),
            ]
        ]

        print(f"\nDIALECT Compliance (ONNX Superset, {total_dialect_ops} total ops):")
        print(
            tabulate(
                dialect_report,
                headers=[
                    "Target",
                    "Compliance %",
                    "Ops Implemented",
                    "Status",
                    "Missing Ops",
                ],
            )
        )

    if not found_any_ir and not found_any_dialect_targets and not found_any_fw:
        print(
            "\nNo IR, FrameworkAdapter, or DIALECT targets detected in the specified path."
        )

    if verbose and found_any_dialect_targets:
        print("\nVerbose Missing Operations Report")
        print("=================================")

        missing_ops_list = sorted(list(dialect_ops - aggregated_dialect_ops))
        dialect_ops_map = {k.split(".")[-1]: v for k, v in ONNX_REGISTRY.items()}

        if mapping_file and os.path.exists(mapping_file):
            import inspect
            import importlib

            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)

            table_data = []
            for op in missing_ops_list:
                if op in mapping_data and "api" in mapping_data[op]:
                    api_path = mapping_data[op]["api"]

                    if api_path.startswith("jnp."):
                        api_path = api_path.replace("jnp.", "jax.numpy.", 1)
                    elif api_path.startswith("tf."):
                        api_path = api_path.replace("tf.", "tensorflow.", 1)

                    parts = api_path.split(".")
                    framework = parts[0]
                    symbol = parts[-1]
                    namespace = ".".join(parts[:-1]) if len(parts) > 1 else framework
                    fqn = api_path

                    signature = "Unknown"
                    docstring = "No docstring available."

                    try:
                        module = importlib.import_module(namespace)
                        obj = getattr(module, symbol)

                        try:
                            sig = inspect.signature(obj)
                            signature = str(sig)
                        except (ValueError, TypeError):
                            if isinstance(obj, type):
                                signature = "Class/Type"
                            else:
                                signature = "Built-in / No signature"

                        doc = inspect.getdoc(obj)
                        if doc:
                            docstring = doc.strip().split("\n\n")[0]
                            docstring = docstring.replace("\n", " ")
                            if len(docstring) > 150:
                                docstring = docstring[:147] + "..."
                    except Exception:
                        pass

                    table_data.append(
                        ["[ ]", framework, namespace, symbol, fqn, signature, docstring]
                    )

            if table_data:
                print(f"\n### {target_label} (Mapped API targets)")
                print(
                    tabulate(
                        table_data,
                        headers=[
                            "Implemented",
                            "Framework",
                            "Namespace",
                            "Symbol",
                            "FQN",
                            "Signature",
                            "Docstring",
                        ],
                        tablefmt="github",
                    )
                )
            else:
                print(f"\n### {target_label} (No mappings found in {mapping_file})")

        else:
            print(f"\n### {target_label}")
            for op in missing_ops_list:
                if op in dialect_ops_map:
                    schema = dialect_ops_map[op]
                    schema_dict = {
                        "domain": schema.domain,
                        "version": schema.version,
                        "inputs": schema.inputs,
                        "outputs": schema.outputs,
                        "attributes": {
                            name: {
                                "type": attr.type,
                                "required": attr.required,
                                "default": attr.default,
                            }
                            for name, attr in schema.attributes.items()
                        },
                    }
                    print(f"- [ ] **{op}**")
                    print("  ```json")
                    json_str = json.dumps(schema_dict, indent=2)
                    for line in json_str.splitlines():
                        print(f"  {line}")
                    print("  ```")
