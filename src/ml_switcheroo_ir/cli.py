"""Command Line Interface for ml-switcheroo-ir.

Provides basic utilities to interact with LogicalGraph representations.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import re
import sys

from ml_switcheroo_ir import (
    CompilerBackend,
    LogicalGraph,
    topological_sort,
)
from ml_switcheroo_ir.schema.custom_ops import Registry
from ml_switcheroo_ir.validator import ValidationError, ValidationLevel, Validator

__all__ = ["main", "tabulate"]

try:
    from tabulate import tabulate
except ImportError:
    # Fallback if tabulate is not available
    def tabulate(data: list[list[object]], headers: list[str], **kwargs: object) -> str:
        """Fallback for tabulate.

        Args:
            data: The table data.
            headers: The table headers.
            kwargs: Ignored keyword arguments.

        Returns:
            The tabulated string.
        """
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
    except Exception as e:  # noqa: BLE001
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

    if hasattr(cls, "compile") and callable(cls.compile):
        checks["Has compile() method"] = True
        sig = inspect.signature(cls.compile)
        if "graph" in sig.parameters:
            checks["compile() signature takes 'graph'"] = True

    print_report()


def main(args: list[str] | None = None) -> None:
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

    # Dump snapshot command
    dump_parser = subparsers.add_parser(
        "dump-snapshot",
        help="Dump all classes, methods, parameters, and ops to GhostRef format JSON",
    )
    dump_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output JSON file",
    )

    # Ground command
    ground_parser = subparsers.add_parser(
        "ground",
        help="Ground and audit a graph against ground-truth framework snapshots",
    )
    ground_parser.add_argument(
        "infile", type=argparse.FileType("r"), help="Input model graph JSON file"
    )
    ground_parser.add_argument(
        "--snapshots-dir",
        "--snapshots",
        type=str,
        required=False,
        default=None,
        help="Path to snapshot JSON file or directory",
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
        grouped_errors: dict[str, list[ValidationError]] = {}
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

        schemas = list(ONNX_REGISTRY.values())

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

    elif parsed_args.command == "dump-snapshot":
        import datetime
        import json
        from typing import Any

        from ml_switcheroo_ir.schema.custom_ops import CUSTOM_OPS_REGISTRY
        from ml_switcheroo_ir.schema.ghost import (
            ExtendedGhostParam,
            GhostMlirRef,
            GhostParam,
            GhostRef,
            IRParameterRole,
            ParameterKind,
            SnapshotEnvelope,
        )
        from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY
        from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY

        snapshot_entries: dict[str, dict[str, Any]] = {}
        onnx_list: list[dict[str, Any]] = []
        custom_list: list[dict[str, Any]] = []
        stablehlo_list: list[dict[str, Any]] = []

        # Dump ONNX operators
        for op_name, schema in ONNX_REGISTRY.items():
            params = []
            for inp in schema.inputs:
                params.append(
                    GhostParam(
                        name=inp,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default=None,
                        annotation="Tensor",
                    )
                )
            for attr_name, attr in schema.attributes.items():
                params.append(
                    GhostParam(
                        name=attr_name,
                        kind=ParameterKind.KEYWORD_ONLY,
                        default=str(attr.default) if attr.default is not None else None,
                        annotation=attr.type,
                    )
                )
            ghost_ref = GhostRef(
                name=op_name,
                api_path=f"{schema.domain}.{op_name}",
                kind="function",
                params=params,
                returns_type="Tensor",
                docstring=f"ONNX operator {op_name} in domain {schema.domain}",
                has_varargs=False,
                schema_version="2.0.0",
            )
            dumped = ghost_ref.model_dump()
            snapshot_entries[f"{schema.domain}.{op_name}"] = dumped
            onnx_list.append(dumped)

        # Dump Custom operators
        for op_name, schema in CUSTOM_OPS_REGISTRY.items():
            params = []
            for inp in schema.inputs:
                params.append(
                    GhostParam(
                        name=inp,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        default=None,
                        annotation="Tensor",
                    )
                )
            for attr_name, attr in schema.attributes.items():
                params.append(
                    GhostParam(
                        name=attr_name,
                        kind=ParameterKind.KEYWORD_ONLY,
                        default=str(attr.default) if attr.default is not None else None,
                        annotation=attr.type,
                    )
                )
            ghost_ref = GhostRef(
                name=op_name,
                api_path=f"{schema.domain}.{op_name}",
                kind="function",
                params=params,
                returns_type="Tensor",
                docstring=f"Custom operator {op_name} in domain {schema.domain}",
                has_varargs=False,
                schema_version="2.0.0",
            )
            dumped = ghost_ref.model_dump()
            snapshot_entries[f"{schema.domain}.{op_name}"] = dumped
            custom_list.append(dumped)

        # Dump StableHLO operators
        for op_name, schema in STABLEHLO_REGISTRY.items():
            operands: list[ExtendedGhostParam | GhostParam] = []
            for inp in schema.inputs:
                operands.append(
                    ExtendedGhostParam(
                        name=inp,
                        kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                        role=IRParameterRole.OPERAND,
                    )
                )
            attrs_dict: dict[str, Any] = {}
            for attr_name, attr in schema.attributes.items():
                attrs_dict[attr_name] = {
                    "type": attr.type,
                    "default": attr.default,
                    "required": attr.required,
                }
            mlir_ref = GhostMlirRef(
                name=op_name,
                api_path=f"{schema.domain}.{op_name}",
                kind="function",
                domain_type="mlir",
                operands=operands,
                attributes=attrs_dict,
                docstring=f"StableHLO operator {op_name} in domain {schema.domain}",
                schema_version="2.0.0",
            )
            dumped = mlir_ref.model_dump()
            snapshot_entries[f"{schema.domain}.{op_name}"] = dumped
            stablehlo_list.append(dumped)

        envelope = SnapshotEnvelope(
            schema_version="2.0.0",
            target="ml-switcheroo-ir",
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            environment={"source": "ml_switcheroo_ir.schema"},
            categories={
                "onnx": onnx_list,
                "custom": custom_list,
                "stablehlo": stablehlo_list,
            },
        )
        data_to_dump = envelope.model_dump()
        data_to_dump.update(snapshot_entries)

        with open(parsed_args.output, "w", encoding="utf-8") as f:
            json.dump(data_to_dump, f, indent=2, sort_keys=True)
        print(f"Dumped snapshot to {parsed_args.output}")

    elif parsed_args.command == "ground":
        from ml_switcheroo_ir.validator import audit_graph_grounding

        json_data = parsed_args.infile.read()
        graph = _parse_graph_from_json(json_data)
        report = audit_graph_grounding(graph, snapshots_path=parsed_args.snapshots_dir)

        print(
            f"Grounding Audit: {report.grounded_count}/{report.total_nodes} nodes grounded. "
            f"Hallucination score: {report.hallucination_score:.1%}"
        )

        if report.diagnostics:
            for err in report.diagnostics:
                print(
                    f"  [{err.level.value}] Node {err.node_id} ({err.attribute}): {err.message}"
                )
            sys.exit(1)
        else:
            print("Graph is fully grounded against framework snapshots.")
            sys.exit(0)


if __name__ == "__main__":
    main()
