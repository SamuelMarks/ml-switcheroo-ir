"""Command Line Interface for ml-switcheroo-ir.

Provides pure static AST verification, validation, grounding, compliance, and schema utilities.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ml_switcheroo_ir import LogicalGraph, topological_sort
from ml_switcheroo_ir.compliance import (
    _extract_base_class_name,
    _fallback_tabulate,
    evaluate_compliance,
    format_compliance_report,
)
from ml_switcheroo_ir.schema.base import OpSchema
from ml_switcheroo_ir.schema.custom_ops import Registry
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostMlirRef,
    GhostParam,
    GhostRef,
    IRParameterRole,
    ParameterKind,
    SnapshotEnvelope,
)
from ml_switcheroo_ir.types import JsonValue
from ml_switcheroo_ir.validator import ValidationError, ValidationLevel, Validator

__all__ = ["main", "tabulate"]

_NL = chr(10)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = _fallback_tabulate


def _print_backend_verification_report(checks: Mapping[str, bool]) -> None:
    """Print the backend verification report to standard output.

    Args:
        checks (Mapping[str, bool]): Mapping of check names to pass/fail booleans.
    """
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    percentage = (passed / total) * 100
    print(_NL + "Verification Report:")
    for name, status in checks.items():
        mark = "PASS" if status else "FAIL"
        print(f" [{mark}] {name}")
    print(_NL + f"Compliance: {percentage:.0f}% ({passed}/{total} requirements met)")


def _inspect_backend_class_ast(tree: ast.Module, class_name: str, checks: dict[str, bool]) -> bool:
    """Inspect a parsed Python module AST for ``CompilerBackend`` class compliance without executing code.

    Args:
        tree (ast.Module): Parsed Python module AST.
        class_name (str): Target backend class name.
        checks (dict[str, bool]): Mutable verification check results dictionary.

    Returns:
        bool: True if the symbol was found in the module, False otherwise.
    """
    target_cls: ast.ClassDef | None = None
    symbol_found = False
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == class_name:
            target_cls = stmt
            symbol_found = True
            break
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == class_name for t in stmt.targets):
            symbol_found = True
            break

    if not symbol_found:
        return False

    checks["Class found"] = True
    if target_cls is None:
        return True

    if any(_extract_base_class_name(b) == "CompilerBackend" for b in target_cls.bases):
        checks["Inherits CompilerBackend"] = True

    for body_stmt in target_cls.body:
        if isinstance(body_stmt, ast.FunctionDef) and body_stmt.name == "compile":
            checks["Has compile() method"] = True
            if any(arg.arg == "graph" for arg in body_stmt.args.args):
                checks["compile() signature takes 'graph'"] = True
    return True


def _verify_backend(file_path: str, class_name: str) -> int:
    """Statically verify via AST that a class implements ``CompilerBackend`` without executing code.

    Args:
        file_path (str): Path to the Python file containing the backend.
        class_name (str): Name of the backend class to verify.

    Returns:
        int: ``0`` if all 5 checks pass, ``1`` otherwise.
    """
    checks = {
        "AST parsed": False,
        "Class found": False,
        "Inherits CompilerBackend": False,
        "Has compile() method": False,
        "compile() signature takes 'graph'": False,
    }

    if not os.path.isfile(file_path):
        print(f"Error: File {file_path} not found.")
        _print_backend_verification_report(checks)
        return 1

    try:
        source = Path(file_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=file_path)
        checks["AST parsed"] = True
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
        print(f"Error parsing module AST: {exc}")
        _print_backend_verification_report(checks)
        return 1

    if not _inspect_backend_class_ast(tree, class_name, checks):
        print(f"Error: Class {class_name} not found in {file_path}")
        _print_backend_verification_report(checks)
        return 1

    _print_backend_verification_report(checks)
    return 0 if all(checks.values()) else 1


def _handle_toposort(parsed_args: argparse.Namespace) -> int:
    """Execute the 'toposort' CLI subcommand using ``LogicalGraph.from_file``.

    Args:
        parsed_args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code ``0``.
    """
    graph = LogicalGraph.from_file(Path(parsed_args.infile))
    sorted_nodes = topological_sort(graph)
    print("Topological Order:")
    for node in sorted_nodes:
        print(f" - {node.id} ({node.op_type})")
    return 0


def _handle_validate(parsed_args: argparse.Namespace) -> int:
    """Execute the 'validate' CLI subcommand using ``LogicalGraph.from_file``.

    Args:
        parsed_args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: ``0`` when valid (or warnings-only in non-strict mode), ``1`` on validation errors.
    """
    from ml_switcheroo_ir.schema.custom_ops import CUSTOM_OPS_REGISTRY
    from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY

    graph = LogicalGraph.from_file(Path(parsed_args.infile))

    registry = Registry()
    if parsed_args.custom_ops:
        registry.load_custom_ops_from_json(parsed_args.custom_ops)

    merged_schemas = {**ONNX_REGISTRY, **registry.schemas}
    merged_custom = {**CUSTOM_OPS_REGISTRY, **registry.schemas}
    validator = Validator(
        registry=merged_schemas,
        custom_registry=merged_custom,
        strict=bool(parsed_args.strict),
    )
    errors = validator.validate_graph(graph)

    if not errors:
        print("Graph is valid.")
        return 0

    has_errors = False
    grouped_errors: dict[str, list[ValidationError]] = {}
    for err in errors:
        if err.level == ValidationLevel.ERROR or parsed_args.strict:
            has_errors = True
        grouped_errors.setdefault(err.node_id, []).append(err)

    for node_id, errs in grouped_errors.items():
        print(f"Node {node_id}:")
        for err in errs:
            level_str = "ERROR" if err.level == ValidationLevel.ERROR or parsed_args.strict else "WARNING"
            print(f"  [{level_str}] {err.attribute}: {err.message}")

    return 1 if has_errors else 0


def _get_named_registries() -> tuple[tuple[str, Mapping[str, OpSchema]], ...]:
    """Return all 21 dialect/ecosystem registries paired with their canonical category keys.

    Returns:
        tuple[tuple[str, Mapping[str, OpSchema]], ...]: Ordered ``(category_name, registry_map)`` pairs.
    """
    from ml_switcheroo_ir.schema.custom_ops import (
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
    from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY
    from ml_switcheroo_ir.schema.rdna_registry import RDNA_REGISTRY
    from ml_switcheroo_ir.schema.sass_registry import SASS_REGISTRY
    from ml_switcheroo_ir.schema.stablehlo import STABLEHLO_REGISTRY

    return (
        ("onnx", ONNX_REGISTRY),
        ("custom", CUSTOM_OPS_REGISTRY),
        ("stablehlo", STABLEHLO_REGISTRY),
        ("mlir", MLIR_REGISTRY),
        ("rdna", RDNA_REGISTRY),
        ("sass", SASS_REGISTRY),
        ("wgsl", WGSL_REGISTRY),
        ("ptx", PTX_REGISTRY),
        ("metal", METAL_REGISTRY),
        ("wasm", WASM_REGISTRY),
        ("webgl", WEBGL_REGISTRY),
        ("aten", ATEN_REGISTRY),
        ("array_api", ARRAY_API_REGISTRY),
        ("chex", CHEX_REGISTRY),
        ("grain", GRAIN_REGISTRY),
        ("orbax", ORBAX_REGISTRY),
        ("optax", OPTAX_REGISTRY),
        ("collective", COLLECTIVE_OPS_REGISTRY),
        ("quantization", QUANTIZATION_OPS_REGISTRY),
        ("state", STATE_OPS_REGISTRY),
        ("odl", ODL_CATALOG),
    )


def _handle_list_ops(parsed_args: argparse.Namespace) -> int:
    """Execute the 'list-ops' CLI subcommand.

    Args:
        parsed_args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code ``0``.
    """
    seen_ops: set[tuple[str, str]] = set()
    schemas: list[OpSchema] = []
    for _, reg in _get_named_registries():
        for s in reg.values():
            key = (s.domain, s.name)
            if key not in seen_ops:
                seen_ops.add(key)
                schemas.append(s)

    if parsed_args.domain:
        schemas = [s for s in schemas if s.domain == parsed_args.domain]

    if parsed_args.search:
        pattern = re.compile(parsed_args.search)
        schemas = [s for s in schemas if pattern.search(s.name)]

    table_data = []
    for s in sorted(schemas, key=lambda x: x.name):
        req_args = [name for name, attr in s.attributes.items() if attr.required]
        opt_args = [name for name, attr in s.attributes.items() if not attr.required]
        table_data.append([s.name, s.domain, ", ".join(req_args), ", ".join(opt_args)])

    print(
        tabulate(
            table_data,
            headers=["Op Name", "Domain", "Required Args", "Optional Args"],
        )
    )
    return 0


def _is_non_overload_function_def(stmt: ast.stmt, sym_name: str) -> bool:
    """Return True if ``stmt`` is a concrete (non-``@overload``) function named ``sym_name``.

    Args:
        stmt (ast.stmt): AST statement node.
        sym_name (str): Expected function identifier.

    Returns:
        bool: True when ``stmt`` defines ``sym_name`` and is not an ``@overload`` stub.
    """
    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) or stmt.name != sym_name:
        return False
    return not any(
        (isinstance(d, ast.Name) and d.id == "overload") or (isinstance(d, ast.Attribute) and d.attr == "overload")
        for d in stmt.decorator_list
    )


def _resolve_package_symbol_ast(
    mod_path: str,
    sym_name: str,
    visited: set[tuple[str, str]] | None = None,
) -> ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Resolve an exported class or function AST node across ``ml_switcheroo_ir`` modules.

    Args:
        mod_path (str): Dotted module path inside ``ml_switcheroo_ir``.
        sym_name (str): Symbol name to locate.
        visited (set[tuple[str, str]] | None): Cycle-detection set.

    Returns:
        ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | None: Resolved AST definition if callable, else None.
    """
    seen = visited if visited is not None else set()
    if (mod_path, sym_name) in seen:
        return None
    seen.add((mod_path, sym_name))

    pkg_root = Path(__file__).resolve().parent.parent
    parts = mod_path.split(".")
    cand_init = pkg_root.joinpath(*parts, "__init__.py")
    py_file = cand_init if cand_init.is_file() else pkg_root.joinpath(*parts[:-1], f"{parts[-1]}.py")
    if not py_file.is_file():
        return None

    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    matched_fn: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == sym_name:
            return stmt
        if _is_non_overload_function_def(stmt, sym_name) and isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            matched_fn = stmt
        elif isinstance(stmt, ast.ImportFrom) and stmt.module:
            for alias in stmt.names:
                if (alias.asname or alias.name) == sym_name:
                    return _resolve_package_symbol_ast(stmt.module, alias.name, seen)
    return matched_fn


def _extract_ast_args_to_ghost_params(args_node: ast.arguments, skip_self: bool) -> list[GhostParam]:
    """Convert an ``ast.arguments`` node into a list of ``GhostParam`` specifications.

    Args:
        args_node (ast.arguments): AST function signature node.
        skip_self (bool): Whether to skip leading ``self``/``cls`` parameter.

    Returns:
        list[GhostParam]: Extracted parameter specifications.
    """
    params: list[GhostParam] = []
    pos_all = list(args_node.posonlyargs) + list(args_node.args)
    num_posonly = len(args_node.posonlyargs)
    default_offset = len(pos_all) - len(args_node.defaults)

    for idx, arg in enumerate(pos_all):
        if skip_self and idx == 0 and arg.arg in ("self", "cls"):
            continue
        p_kind = ParameterKind.POSITIONAL_ONLY if idx < num_posonly else ParameterKind.POSITIONAL_OR_KEYWORD
        def_expr = args_node.defaults[idx - default_offset] if idx >= default_offset else None
        params.append(
            GhostParam(
                name=arg.arg,
                kind=p_kind,
                default=ast.unparse(def_expr) if def_expr is not None else None,
                annotation=ast.unparse(arg.annotation) if arg.annotation is not None else None,
            )
        )
    if args_node.vararg is not None:
        v_ann = args_node.vararg.annotation
        params.append(
            GhostParam(
                name=args_node.vararg.arg,
                kind=ParameterKind.VAR_POSITIONAL,
                default=None,
                annotation=ast.unparse(v_ann) if v_ann is not None else None,
            )
        )
    for kw_arg, kw_def in zip(args_node.kwonlyargs, args_node.kw_defaults):
        params.append(
            GhostParam(
                name=kw_arg.arg,
                kind=ParameterKind.KEYWORD_ONLY,
                default=ast.unparse(kw_def) if kw_def is not None else None,
                annotation=ast.unparse(kw_arg.annotation) if kw_arg.annotation is not None else None,
            )
        )
    if args_node.kwarg is not None:
        kw_ann = args_node.kwarg.annotation
        params.append(
            GhostParam(
                name=args_node.kwarg.arg,
                kind=ParameterKind.VAR_KEYWORD,
                default=None,
                annotation=ast.unparse(kw_ann) if kw_ann is not None else None,
            )
        )
    return params


def _build_symbol_ghost_params(sym_name: str) -> list[GhostParam]:
    """Extract GhostParam list from an exported ``ml_switcheroo_ir`` class or function signature.

    Args:
        sym_name (str): Exported symbol name in ``ml_switcheroo_ir``.

    Returns:
        list[GhostParam]: Extracted parameter specifications.
    """
    sym_node = _resolve_package_symbol_ast("ml_switcheroo_ir", sym_name)
    if sym_node is None:
        return []
    if isinstance(sym_node, ast.ClassDef):
        for stmt in sym_node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                return _extract_ast_args_to_ghost_params(stmt.args, skip_self=True)
        return []
    return _extract_ast_args_to_ghost_params(sym_node.args, skip_self=False)


def _collect_ir_public_symbol_snapshots(
    operations_map: dict[str, dict[str, JsonValue] | GhostMlirRef],
) -> tuple[
    list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef],
    list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef],
]:
    """Collect ``GhostRef`` snapshot records for all public classes and functions in ``ml_switcheroo_ir``.

    Args:
        operations_map (dict[str, dict[str, JsonValue] | GhostMlirRef]):
            Envelope ``operations`` map to populate.

    Returns:
        tuple[
            list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef],
            list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef],
        ]: Tuple of ``(classes_list, functions_list)``.
    """
    from ml_switcheroo_ir import __all__ as ir_public_all

    classes_list: list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef] = []
    functions_list: list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef] = []

    for sym_name in ir_public_all:
        sym_node = _resolve_package_symbol_ast("ml_switcheroo_ir", sym_name)
        if sym_node is None:
            continue
        is_cls = isinstance(sym_node, ast.ClassDef)
        kind = "class" if is_cls else "function"
        params = _build_symbol_ghost_params(sym_name)
        doc = ast.get_docstring(sym_node)
        api_path = f"ml_switcheroo_ir.{sym_name}"
        ref = GhostRef(
            name=sym_name,
            api_path=api_path,
            kind=kind,
            params=params,
            docstring=doc,
            has_varargs=any(p.kind == ParameterKind.VAR_POSITIONAL for p in params),
            schema_version="2.0.0",
        )
        dumped = ref.model_dump()
        operations_map[api_path] = dumped
        if is_cls:
            classes_list.append(dumped)
        else:
            functions_list.append(dumped)

    return classes_list, functions_list


def _schema_to_ghost_entry(category_name: str, op_name: str, schema: OpSchema) -> dict[str, JsonValue]:
    """Serialize an ``OpSchema`` into a ``GhostMlirRef`` or ``GhostRef`` dictionary for ``dump-snapshot``.

    Args:
        category_name (str): Registry category key.
        op_name (str): Operator name key in the registry.
        schema (OpSchema): Operator schema instance.

    Returns:
        dict[str, JsonValue]: Validated ``GhostRef`` or ``GhostMlirRef`` dictionary.
    """
    api_path = f"{schema.domain}.{op_name}" if not op_name.startswith(f"{schema.domain}.") else op_name
    if category_name in ("stablehlo", "mlir", "core_mlir"):
        operands: list[ExtendedGhostParam | GhostParam] = [
            ExtendedGhostParam(
                name=inp,
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                role=IRParameterRole.OPERAND,
            )
            for inp in schema.inputs
        ]
        attrs_dict: dict[str, JsonValue] = {
            attr_name: {
                "type": attr.type,
                "default": (
                    attr.default
                    if isinstance(attr.default, (str, int, float, bool)) or attr.default is None
                    else str(attr.default)
                ),
                "required": attr.required,
            }
            for attr_name, attr in schema.attributes.items()
        }
        return GhostMlirRef(
            name=op_name,
            api_path=api_path,
            kind="function",
            domain_type="mlir",
            operands=operands,
            attributes=attrs_dict,
            docstring=f"{category_name} operator {op_name} in domain {schema.domain}",
            schema_version="2.0.0",
        ).model_dump()

    params = [
        GhostParam(
            name=inp,
            kind=ParameterKind.POSITIONAL_OR_KEYWORD,
            default=None,
            annotation="Tensor",
        )
        for inp in schema.inputs
    ] + [
        GhostParam(
            name=attr_name,
            kind=ParameterKind.KEYWORD_ONLY,
            default=str(attr.default) if attr.default is not None else None,
            annotation=attr.type,
        )
        for attr_name, attr in schema.attributes.items()
    ]
    return GhostRef(
        name=op_name,
        api_path=api_path,
        kind="function",
        params=params,
        returns_type="Tensor",
        docstring=f"{category_name} operator {op_name} in domain {schema.domain}",
        has_varargs=False,
        schema_version="2.0.0",
    ).model_dump()


def _handle_dump_snapshot(parsed_args: argparse.Namespace) -> int:
    """Execute 'dump-snapshot', exporting all registries into a valid ``SnapshotEnvelope`` JSON file.

    Args:
        parsed_args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code ``0``.
    """
    from ml_switcheroo_ir import __version__

    operations_map: dict[str, dict[str, JsonValue] | GhostMlirRef] = {}
    classes_list, functions_list = _collect_ir_public_symbol_snapshots(operations_map)
    categories: dict[str, list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef]] = {
        "classes": classes_list,
        "functions": functions_list,
    }

    for cat_name, reg in _get_named_registries():
        cat_entries: list[dict[str, JsonValue] | ExtendedGhostRef | GhostRef] = []
        for op_name, schema in reg.items():
            dumped = _schema_to_ghost_entry(cat_name, op_name, schema)
            api_path = str(dumped["api_path"])
            operations_map[api_path] = dumped
            cat_entries.append(dumped)
        categories[cat_name] = cat_entries

    envelope = SnapshotEnvelope(
        schema_version="2.0.0",
        target="ml-switcheroo-ir",
        version=__version__,
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        environment={"source": "ml_switcheroo_ir.schema"},
        categories=categories,
        operations=list(operations_map.values()),
    )
    json_payload = envelope.model_dump_json(indent=2)
    SnapshotEnvelope.model_validate_json(json_payload)
    Path(parsed_args.output).write_text(json_payload + _NL, encoding="utf-8")
    print(f"Dumped snapshot to {parsed_args.output}")
    return 0


def _handle_ground(parsed_args: argparse.Namespace) -> int:
    """Execute the 'ground' CLI subcommand using ``LogicalGraph.from_file``.

    Args:
        parsed_args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: ``0`` if fully grounded, ``1`` if grounding diagnostics are present.
    """
    from ml_switcheroo_ir.validator import audit_graph_grounding

    graph = LogicalGraph.from_file(Path(parsed_args.infile))
    report = audit_graph_grounding(graph, snapshots_path=parsed_args.snapshots_dir)

    print(
        f"Grounding Audit: {report.grounded_count}/{report.total_nodes} nodes grounded. "
        f"Hallucination score: {report.hallucination_score:.1%}"
    )

    if report.diagnostics:
        for err in report.diagnostics:
            print(f"  [{err.level.value}] Node {err.node_id} ({err.attribute}): {err.message}")
        return 1
    print("Graph is fully grounded against framework snapshots.")
    return 0


def _handle_compliance(parsed_args: argparse.Namespace) -> int:
    """Execute the 'compliance' CLI subcommand and render the resulting ``ComplianceReport``.

    Args:
        parsed_args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: ``0`` on completed audit, ``1`` if target paths are missing or invalid.
    """
    try:
        report = evaluate_compliance(
            parsed_args.target,
            verbose=bool(parsed_args.verbose),
            mapping_file=parsed_args.mapping,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1

    print(format_compliance_report(report, verbose=bool(parsed_args.verbose)))
    return 0


def _handle_export_schema(parsed_args: argparse.Namespace) -> int:
    """Execute the 'export-schema' CLI subcommand.

    Args:
        parsed_args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Process exit code ``0``.
    """
    from ml_switcheroo_ir.export import (
        export_schemas,
        generate_typescript_definitions,
        get_json_schema,
    )

    if parsed_args.out_dir:
        exported = export_schemas(parsed_args.out_dir, include_typescript=parsed_args.typescript)
        print(f"Exported {len(exported)} schemas to {parsed_args.out_dir}:")
        for k, v in exported.items():
            print(f" - {k}: {v}")
    elif parsed_args.ts_out:
        ts_code = generate_typescript_definitions()
        Path(parsed_args.ts_out).parent.mkdir(parents=True, exist_ok=True)
        Path(parsed_args.ts_out).write_text(ts_code, encoding="utf-8")
        print(f"Exported TypeScript interfaces to {parsed_args.ts_out}")
    elif parsed_args.typescript:
        print(generate_typescript_definitions())
    else:
        exported_schema = get_json_schema(parsed_args.target)
        print(json.dumps(exported_schema, indent=2, sort_keys=True))
    return 0


def _build_cli_parser() -> argparse.ArgumentParser:
    """Construct the ``ArgumentParser`` for ``ml-switcheroo-ir``.

    Returns:
        argparse.ArgumentParser: Configured CLI parser.
    """
    parser = argparse.ArgumentParser(description="ml-switcheroo-ir CLI - Utility for IR manipulation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sort_parser = subparsers.add_parser("toposort", help="Topologically sort a graph from JSON/.gz/.zst")
    sort_parser.add_argument("infile", type=Path, help="Input graph file (.json, .json.gz, .json.zst)")

    verify_parser = subparsers.add_parser(
        "verify-backend",
        help="Statically verify a backend implementation against IR interfaces",
    )
    verify_parser.add_argument("file_path", type=str, help="Path to the python file containing the backend")
    verify_parser.add_argument("class_name", type=str, help="Name of the class to verify")

    validate_parser = subparsers.add_parser("validate", help="Validate a graph against the operator schema")
    validate_parser.add_argument("infile", type=Path, help="Input graph file (.json, .json.gz, .json.zst)")
    validate_parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    validate_parser.add_argument("--custom-ops", type=str, help="Path to a custom ops JSON file")

    list_ops_parser = subparsers.add_parser("list-ops", help="List registered operators")
    list_ops_parser.add_argument("--domain", type=str, help="Filter by domain")
    list_ops_parser.add_argument("--search", type=str, help="Regex search operator names")

    compliance_parser = subparsers.add_parser(
        "compliance",
        help="Check compliance of Python code against ml-switcheroo-ir IR or DIALECT",
    )
    compliance_parser.add_argument("target", type=str, nargs="+", help="Python file(s) or directory to check")
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

    dump_parser = subparsers.add_parser(
        "dump-snapshot",
        help="Dump all classes, methods, parameters, and ops to SnapshotEnvelope JSON",
    )
    dump_parser.add_argument("--output", "-o", type=str, required=True, help="Path to output JSON file")

    ground_parser = subparsers.add_parser(
        "ground",
        help="Ground and audit a graph against ground-truth framework snapshots",
    )
    ground_parser.add_argument("infile", type=Path, help="Input graph file (.json, .json.gz, .json.zst)")
    ground_parser.add_argument(
        "--snapshots-dir",
        "--snapshots",
        type=str,
        required=False,
        default=None,
        help="Path to snapshot JSON file or directory",
    )

    export_parser = subparsers.add_parser(
        "export-schema",
        help="Export canonical JSON Schema definitions and TypeScript interfaces",
    )
    export_parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=[
            "all",
            "LogicalGraph",
            "LogicalNode",
            "SnapshotEnvelope",
            "ZeroTangent",
            "NoTangent",
            "PipelineTopologyConfig",
            "WebRTCSignalingTopology",
        ],
        help="Target schema to export (default: all)",
    )
    export_parser.add_argument("--out-dir", "-o", type=str, default=None, help="Directory to write JSON schemas")
    export_parser.add_argument("--typescript", "--ts", action="store_true", help="Export TypeScript definitions")
    export_parser.add_argument("--ts-out", type=str, default=None, help="File path to write TypeScript definitions")
    return parser


def main(args: Sequence[str] | None = None) -> int:
    """Entrypoint for the CLI returning a process exit code.

    Args:
        args (Sequence[str] | None): Command line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        int: Integer process exit code (``0`` on success, non-zero on error).
    """
    active_args = list(sys.argv[1:]) if args is None else list(args)
    parser = _build_cli_parser()
    parsed_args = parser.parse_args(active_args)

    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "toposort": _handle_toposort,
        "verify-backend": lambda ns: _verify_backend(ns.file_path, ns.class_name),
        "validate": _handle_validate,
        "compliance": _handle_compliance,
        "list-ops": _handle_list_ops,
        "dump-snapshot": _handle_dump_snapshot,
        "ground": _handle_ground,
        "export-schema": _handle_export_schema,
    }
    handler = handlers.get(str(parsed_args.command))
    return handler(parsed_args) if handler is not None else 1


if __name__ == "__main__":
    sys.exit(main())
