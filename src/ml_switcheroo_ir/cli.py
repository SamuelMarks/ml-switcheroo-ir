"""Command Line Interface for ml-switcheroo-ir.

Provides basic utilities to interact with LogicalGraph representations.
"""

import argparse
import sys
import os
import importlib.util
import inspect
import re
from typing import List

from ml_switcheroo_ir import (
    LogicalGraph,
    topological_sort,
    CompilerBackend,
)
from ml_switcheroo_ir.validator import Validator, ValidationLevel
from ml_switcheroo_ir.schema.custom_ops import Registry

try:
    from tabulate import tabulate
except ImportError:
    # Fallback if tabulate is not available
    def tabulate(data: list[list[object]], headers: list[str]) -> str:
        """Fallback for tabulate."""
        res = " | ".join(headers) + "\n"
        res += "-" * len(res) + "\n"
        for row in data:
            res += " | ".join(str(c) for c in row) + "\n"
        return res


def _parse_graph_from_json(json_str: str) -> LogicalGraph:
    """Parses a LogicalGraph from a JSON string.

    Args:
        json_str (str): The JSON string.

    Returns:
        LogicalGraph: The parsed graph object.

    """
    return LogicalGraph.from_json(json_str)


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
        "target", type=str, nargs="+", help="Python file(s) or directory to check"
    )
    compliance_parser.add_argument(
        "-m",
        "--mapping",
        type=str,
        help="Path to a framework definitions JSON file (e.g. jax.json) to show target API info for missing ops",
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
        from ml_switcheroo_ir.compliance import run_compliance_check

        run_compliance_check(
            parsed_args.target,
            verbose=parsed_args.verbose,
            mapping_file=parsed_args.mapping,
        )

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
