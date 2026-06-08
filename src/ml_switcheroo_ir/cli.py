"""Command Line Interface for ml-switcheroo-ir.

Provides basic utilities to interact with LogicalGraph representations.
"""

import argparse
import sys
import json
import os
import importlib.util
import inspect
import re
from typing import List

from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalNode,
    LogicalEdge,
    topological_sort,
    CompilerBackend,
)
from ml_switcheroo_ir.validator import Validator, ValidationLevel
from ml_switcheroo_ir.schema.custom_ops import Registry

try:
    from tabulate import tabulate
except ImportError:
    # Fallback if tabulate is not available
    def tabulate(data, headers):
        """Fallback for tabulate."""
        res = " | ".join(headers) + "\n"
        res += "-" * len(res) + "\n"
        for row in data:
            res += " | ".join(str(c) for c in row) + "\n"
        return res


def _parse_graph_from_json(json_str: str) -> LogicalGraph:
    """Parse a LogicalGraph from a JSON string representation.

    Args:
        json_str (str): The JSON string containing graph data.

    Returns:
        LogicalGraph: The parsed graph object.

    """
    data = json.loads(json_str)
    nodes = []
    for n in data.get("nodes", []):
        nodes.append(
            LogicalNode(
                id=n.get("id", ""),
                kind=n.get("kind", ""),
                domain=n.get("domain", "ai.onnx"),
                version=n.get("version", 1),
                metadata=n.get("metadata", {}),
            )
        )
    edges = []
    for e in data.get("edges", []):
        edges.append(
            LogicalEdge(
                source=e.get("source", ""),
                target=e.get("target", ""),
            )
        )
    return LogicalGraph(
        name=data.get("name", "Model"),
        nodes=nodes,
        edges=edges,
    )


def _verify_backend(file_path: str, class_name: str) -> None:
    """Verify that a specified class implements the CompilerBackend interface.

    Args:
        file_path (str): Path to the Python file containing the backend.
        class_name (str): Name of the backend class to verify.

    """
    checks = {
        "Module loaded": False,
        "Class found": False,
        "Inherits CompilerBackend": False,
        "Has compile() method": False,
        "compile() signature takes 'graph'": False,
    }

    def print_report() -> None:
        """Print the verification report to standard output."""
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        percentage = (passed / total) * 100
        print("\nVerification Report:")
        for name, status in checks.items():
            mark = "PASS" if status else "FAIL"
            print(f" [{mark}] {name}")
        print(f"\nCompliance: {percentage:.0f}% ({passed}/{total} requirements met)")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        print_report()
        return

    spec = importlib.util.spec_from_file_location("dynamic_backend", file_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load module from {file_path}")
        print_report()
        return

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        checks["Module loaded"] = True
    except Exception as e:
        print(f"Error executing module: {e}")
        print_report()
        return

    if not hasattr(module, class_name):
        print(f"Error: Class {class_name} not found in {file_path}")
        print_report()
        return

    cls = getattr(module, class_name)
    checks["Class found"] = True

    try:
        if issubclass(cls, CompilerBackend):
            checks["Inherits CompilerBackend"] = True
    except TypeError:
        pass

    if hasattr(cls, "compile") and callable(getattr(cls, "compile")):
        checks["Has compile() method"] = True
        sig = inspect.signature(getattr(cls, "compile"))
        if "graph" in sig.parameters:
            checks["compile() signature takes 'graph'"] = True

    print_report()


def main(args: List[str] = None) -> None:
    """Entrypoint for the CLI.

    Args:
        args (List[str], optional): Command line arguments. Defaults to None (sys.argv).

    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="ml-switcheroo-ir CLI - Utility for IR manipulation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Toposort command
    sort_parser = subparsers.add_parser(
        "toposort", help="Topologically sort a graph from JSON"
    )
    sort_parser.add_argument(
        "infile", type=argparse.FileType("r"), help="Input JSON file"
    )

    # Verify backend command
    verify_parser = subparsers.add_parser(
        "verify-backend", help="Verify a backend implementation against IR interfaces"
    )
    verify_parser.add_argument(
        "file_path", type=str, help="Path to the python file containing the backend"
    )
    verify_parser.add_argument(
        "class_name", type=str, help="Name of the class to verify"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a graph against the operator schema"
    )
    validate_parser.add_argument(
        "infile", type=argparse.FileType("r"), help="Input JSON file"
    )
    validate_parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors"
    )
    validate_parser.add_argument(
        "--custom-ops", type=str, help="Path to a custom ops JSON file"
    )

    # List ops command
    list_ops_parser = subparsers.add_parser(
        "list-ops", help="List registered operators"
    )
    list_ops_parser.add_argument("--domain", type=str, help="Filter by domain")
    list_ops_parser.add_argument(
        "--search", type=str, help="Regex search operator names"
    )

    # Compliance command
    compliance_parser = subparsers.add_parser(
        "compliance",
        help="Check compliance of Python code against ml-switcheroo-ir IR or DIALECT",
    )
    compliance_parser.add_argument(
        "target", type=str, help="Python file or directory to check"
    )
    compliance_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print a detailed markdown checklist of missing operations",
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "toposort":
        json_data = parsed_args.infile.read()
        graph = _parse_graph_from_json(json_data)
        sorted_nodes = topological_sort(graph)
        print("Topological Order:")
        for node in sorted_nodes:
            print(f" - {node.id} ({node.kind})")

    elif parsed_args.command == "verify-backend":
        _verify_backend(parsed_args.file_path, parsed_args.class_name)

    elif parsed_args.command == "validate":
        json_data = parsed_args.infile.read()
        graph = _parse_graph_from_json(json_data)

        registry = Registry()
        if parsed_args.custom_ops:
            registry.load_custom_ops_from_json(parsed_args.custom_ops)

        # Merge with base ONNX registry inside Validator
        from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY

        merged_schemas = {**ONNX_REGISTRY, **registry.schemas}
        validator = Validator(registry=merged_schemas)

        errors = validator.validate_graph(graph)

        if not errors:
            print("Graph is valid.")
            sys.exit(0)

        has_errors = False
        grouped_errors = {}
        for err in errors:
            if err.level == ValidationLevel.ERROR or parsed_args.strict:
                has_errors = True
            grouped_errors.setdefault(err.node_id, []).append(err)

        for node_id, errs in grouped_errors.items():
            print(f"Node {node_id}:")
            for err in errs:
                level_str = (
                    "ERROR"
                    if err.level == ValidationLevel.ERROR or parsed_args.strict
                    else "WARNING"
                )
                print(f"  [{level_str}] {err.attribute}: {err.message}")

        if has_errors:
            sys.exit(1)
        else:
            sys.exit(0)

    elif parsed_args.command == "compliance":
        from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY
        import ast

        target = parsed_args.target

        if not os.path.exists(target):
            print(f"Error: Target path '{target}' not found.")
            sys.exit(1)

        py_files = []
        if os.path.isfile(target):
            if target.endswith(".py"):
                py_files.append(target)
        else:
            for root, dirs, files in os.walk(target):
                if "node_modules" in dirs:
                    dirs.remove("node_modules")
                if ".venv" in dirs:
                    dirs.remove(".venv")
                for file in files:
                    if (
                        file.endswith(".py")
                        and not file.startswith("test_")
                        and file != "__init__.py"
                    ):
                        py_files.append(os.path.join(root, file))

        ir_reports = []
        framework_reports = []
        dialect_reports = []

        dialect_keys = set(ONNX_REGISTRY.keys())
        dialect_ops = {k.split(".")[-1] for k in dialect_keys}
        total_dialect_ops = len(dialect_ops)

        for filepath in py_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    source = f.read()

                tree = ast.parse(source)

                rel_path = (
                    os.path.relpath(filepath, target)
                    if os.path.isdir(target)
                    else os.path.basename(filepath)
                )

                # Check for IR
                is_ir = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for base in node.bases:
                            if getattr(base, "id", "") in (
                                "CompilerBackend",
                                "GraphFrontend",
                            ):
                                is_ir = True
                                break

                # Check for FrameworkAdapter
                is_fw = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for base in node.bases:
                            # Not all adapters inherit from FrameworkAdapter formally in code,
                            # but they all end in Adapter and register_framework
                            pass
                        if "Adapter" in node.name:
                            # Check decorators
                            for dec in node.decorator_list:
                                if (
                                    isinstance(dec, ast.Call)
                                    and getattr(dec.func, "id", "")
                                    == "register_framework"
                                ):
                                    is_fw = True
                                    break

                # Check for Dialect
                implemented_ops = set()
                loaded_definitions_fw = None

                for node in ast.walk(tree):
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        if node.name in dialect_ops:
                            implemented_ops.add(node.name)

                    if isinstance(node, ast.Dict):
                        for key in node.keys:
                            if isinstance(key, ast.Constant) and isinstance(
                                key.value, str
                            ):
                                if key.value in dialect_ops:
                                    implemented_ops.add(key.value)

                    if (
                        isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") == "load_definitions"
                    ):
                        if node.args and isinstance(node.args[0], ast.Constant):
                            loaded_definitions_fw = node.args[0].value

                # If dynamic definitions were found via `load_definitions("fw")`
                if loaded_definitions_fw:
                    try:
                        # Assuming the standard path structure relative to the script if running on workspace
                        if "src/ml_switcheroo" in filepath:
                            ws_root = filepath.split("src/ml_switcheroo")[0]
                            def_path = os.path.join(
                                ws_root,
                                "src",
                                "ml_switcheroo",
                                "frameworks",
                                "definitions",
                                f"{loaded_definitions_fw}.json",
                            )
                            if os.path.exists(def_path):
                                with open(def_path, "r", encoding="utf-8") as f:
                                    dynamic_defs = json.load(f)
                                for k in dynamic_defs.keys():
                                    if k in dialect_ops:
                                        implemented_ops.add(k)
                    except Exception:  # pragma: no cover
                        pass

                is_dialect = len(implemented_ops) > 0

                if is_ir:
                    # Score the file based on the highest-scoring IR class
                    best_req_met = 2  # Module loaded statically, Class found
                    req_total = 5  # Inherits, Has method, Signature takes graph/code

                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            is_backend = any(
                                getattr(b, "id", "") == "CompilerBackend"
                                for b in node.bases
                            )
                            is_frontend = any(
                                getattr(b, "id", "") == "GraphFrontend"
                                for b in node.bases
                            )
                            if is_backend or is_frontend:
                                current_met = 3  # Module(1) + Class(1) + Inherits(1)
                                target_method = (
                                    "compile" if is_backend else "parse_to_graph"
                                )
                                target_arg = "graph" if is_backend else "code"
                                for item in node.body:
                                    if (
                                        isinstance(item, ast.FunctionDef)
                                        and item.name == target_method
                                    ):
                                        current_met += 1  # Has method
                                        arg_names = [arg.arg for arg in item.args.args]
                                        if target_arg in arg_names:
                                            current_met += 1  # Signature
                                        break
                                best_req_met = max(best_req_met, current_met)

                    pct = (best_req_met / req_total) * 100
                    status = "PASS" if pct == 100 else "FAIL"
                    ir_reports.append(
                        [rel_path, f"{pct:.0f}%", f"{best_req_met}/{req_total}", status]
                    )

                if is_fw:
                    req_total = 4  # Module(1) + Class(1) + convert(1) + definitions(1)
                    best_req_met = 2  # Module loaded, Class found
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and (
                            "Adapter" in node.name
                            or any(
                                isinstance(d, ast.Call)
                                and getattr(d.func, "id", "") == "register_framework"
                                for d in node.decorator_list
                            )
                        ):
                            current_met = 2
                            methods = [
                                n.name
                                for n in node.body
                                if isinstance(n, ast.FunctionDef)
                            ]
                            if "convert" in methods:
                                current_met += 1
                            if "definitions" in methods:
                                current_met += 1
                            best_req_met = max(best_req_met, current_met)
                    pct = (best_req_met / req_total) * 100
                    status = "PASS" if pct == 100 else "FAIL"
                    framework_reports.append(
                        [rel_path, f"{pct:.0f}%", f"{best_req_met}/{req_total}", status]
                    )

                if is_dialect or (not is_ir and os.path.isfile(target)):
                    pct = (len(implemented_ops) / total_dialect_ops) * 100
                    status = "PASS" if pct == 100 else "FAIL"
                    missing_ops_list = sorted(list(dialect_ops - implemented_ops))
                    missing_count = len(missing_ops_list)
                    dialect_reports.append(
                        [
                            rel_path,
                            f"{pct:.1f}%",
                            f"{len(implemented_ops)}/{total_dialect_ops}",
                            status,
                            missing_count,
                            missing_ops_list,
                        ]
                    )

            except Exception:
                pass

        print("\nCompliance Report")
        print("=================")

        if ir_reports:
            print("\nIR Compliance (CompilerBackend / GraphFrontend):")
            print(
                tabulate(
                    ir_reports, headers=["File", "Compliance %", "Reqs Met", "Status"]
                )
            )

        if framework_reports:
            print("\nFrameworkAdapter Compliance:")
            print(
                tabulate(
                    framework_reports,
                    headers=["File", "Compliance %", "Reqs Met", "Status"],
                )
            )

        if dialect_reports:
            print(
                f"\nDIALECT Compliance (ONNX Superset, {total_dialect_ops} total ops):"
            )
            print(
                tabulate(
                    [r[:5] for r in dialect_reports],
                    headers=[
                        "File",
                        "Compliance %",
                        "Ops Implemented",
                        "Status",
                        "Missing Ops",
                    ],
                )
            )

        if not ir_reports and not dialect_reports and not framework_reports:
            print(
                "\nNo IR, FrameworkAdapter, or DIALECT targets detected in the specified path."
            )

        if parsed_args.verbose and dialect_reports:
            print("\nVerbose Missing Operations Report")
            print("=================================")

            # Create mapping to recover the full domain prefix
            dialect_ops_map = {k.split(".")[-1]: v for k, v in ONNX_REGISTRY.items()}

            for report in dialect_reports:
                rel_path = report[0]
                missing_ops_list = report[5]

                if not missing_ops_list:  # pragma: no cover
                    continue

                print(f"\n### {rel_path}")
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

    elif parsed_args.command == "list-ops":
        from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY

        schemas = ONNX_REGISTRY.values()

        if parsed_args.domain:
            schemas = [s for s in schemas if s.domain == parsed_args.domain]

        if parsed_args.search:
            pattern = re.compile(parsed_args.search)
            schemas = [s for s in schemas if pattern.search(s.name)]

        table_data = []
        for s in sorted(schemas, key=lambda x: x.name):
            req_args = [name for name, attr in s.attributes.items() if attr.required]
            opt_args = [
                name for name, attr in s.attributes.items() if not attr.required
            ]
            table_data.append(
                [s.name, s.domain, ", ".join(req_args), ", ".join(opt_args)]
            )

        print(
            tabulate(
                table_data,
                headers=["Op Name", "Domain", "Required Args", "Optional Args"],
            )
        )


if __name__ == "__main__":  # pragma: no cover
    main()
