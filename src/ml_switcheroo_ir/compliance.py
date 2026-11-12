"""Pure side-effect-free compliance evaluation and reporting for ML-Switcheroo IR."""

from __future__ import annotations

import ast
import json
import os
from collections.abc import Sequence

import pydantic

from ml_switcheroo_ir.schema.ghost import ExtendedGhostRef, GhostParam
from ml_switcheroo_ir.schema.onnx_registry import ONNX_REGISTRY
from ml_switcheroo_ir.types import JsonValue

_NL = chr(10)


def _fallback_tabulate(
    tabular_data: Sequence[Sequence[str | int | float]],
    headers: Sequence[str] = (),
    tablefmt: str = "grid",
) -> str:
    """Format tabular data as a pipe-separated ASCII table when tabulate is unavailable.

    Args:
        tabular_data (Sequence[Sequence[str | int | float]]): Rows of table cells.
        headers (Sequence[str]): Sequence of column header labels.
        tablefmt (str): Table formatting style identifier.

    Returns:
        str: Formatted table string.
    """
    del tablefmt
    res = " | ".join(str(h) for h in headers) + _NL
    res += "-" * len(res) + _NL
    for row in tabular_data:
        res += " | ".join(str(c) for c in row) + _NL
    return res


class FileComplianceResult(pydantic.BaseModel):
    """Pydantic model representing the static AST compliance analysis of a single Python file.

    Attributes:
        is_ir (bool): Whether the file defines an IR Backend or Frontend.
        is_fw (bool): Whether the file defines a FrameworkAdapter or Importer.
        implemented_ops (set[str]): Set of implemented dialect operator names.
        ir_score (tuple[int, int]): Tuple of ``(met_requirements, total_requirements)`` for IR.
        fw_score (tuple[int, int]): Tuple of ``(met_requirements, total_requirements)`` for FrameworkAdapter.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    is_ir: bool = False
    is_fw: bool = False
    implemented_ops: set[str] = pydantic.Field(default_factory=set)
    ir_score: tuple[int, int] = (0, 5)
    fw_score: tuple[int, int] = (0, 4)


class ComplianceReport(pydantic.BaseModel):
    """Typed side-effect-free compliance report across one or more target paths.

    Attributes:
        target_label (str): Human-readable label describing the evaluated targets.
        analyzed_files (list[str]): Ordered list of analyzed Python/JSON file paths.
        found_ir (bool): True if at least one IR backend/frontend class was discovered.
        ir_score (tuple[int, int]): Best ``(met, total)`` IR score across analyzed files.
        found_fw (bool): True if at least one FrameworkAdapter/Importer class was discovered.
        fw_score (tuple[int, int]): Best ``(met, total)`` FrameworkAdapter score across analyzed files.
        found_dialect (bool): True if dialect operator targets were detected.
        implemented_ops (list[str]): Sorted list of implemented dialect operator names.
        missing_ops (list[str]): Sorted list of unimplemented dialect operator names.
        total_dialect_ops (int): Total canonical dialect operator count.
        mapped_missing_rows (list[list[str]]): Grounded API mapping rows when ``mapping_file`` is supplied.
        unmapped_missing_schemas (dict[str, dict[str, JsonValue]]): Missing operator schema summaries.
        mapping_file (str | None): Optional mapping JSON path supplied for verbose output.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    target_label: str
    analyzed_files: list[str] = pydantic.Field(default_factory=list)
    found_ir: bool = False
    ir_score: tuple[int, int] = (0, 5)
    found_fw: bool = False
    fw_score: tuple[int, int] = (0, 4)
    found_dialect: bool = False
    implemented_ops: list[str] = pydantic.Field(default_factory=list)
    missing_ops: list[str] = pydantic.Field(default_factory=list)
    total_dialect_ops: int = 0
    mapped_missing_rows: list[list[str]] = pydantic.Field(default_factory=list)
    unmapped_missing_schemas: dict[str, dict[str, JsonValue]] = pydantic.Field(default_factory=dict)
    mapping_file: str | None = None


__all__ = [
    "ComplianceReport",
    "FileComplianceResult",
    "analyze_python_file",
    "collect_files",
    "evaluate_compliance",
    "extract_dynamic_definitions",
    "format_compliance_report",
    "get_dialect_ops",
    "parse_json_file",
    "run_compliance_check",
    "tabulate",
]

try:
    from tabulate import tabulate
except ImportError:
    tabulate = _fallback_tabulate


def collect_files(targets: Sequence[str]) -> list[str]:
    """Collect Python and JSON files from a sequence of target paths without side effects.

    Args:
        targets (Sequence[str]): Sequence of file or directory paths.

    Returns:
        list[str]: List of absolute file paths to process.

    Raises:
        FileNotFoundError: If any target path does not exist on disk.
    """
    collected_files: list[str] = []
    for target in targets:
        if not os.path.exists(target):
            raise FileNotFoundError(f"Target path '{target}' not found.")

        if os.path.isfile(target):
            if target.endswith((".py", ".json")):
                collected_files.append(os.path.abspath(target))
        else:
            for root, dirs, files in os.walk(target):
                for excluded in ("node_modules", ".venv"):
                    if excluded in dirs:
                        dirs.remove(excluded)
                dirs.sort()
                for file in sorted(files):
                    if file.endswith((".py", ".json")) and not file.startswith("test_") and file != "__init__.py":
                        collected_files.append(os.path.join(root, file))
    return collected_files


def get_dialect_ops() -> set[str]:
    """Get the set of all operator names in the ONNX dialect registry.

    Returns:
        set[str]: Set of operator names.
    """
    return {k.split(".")[-1] for k in ONNX_REGISTRY}


def parse_json_file(filepath: str, dialect_ops: set[str]) -> set[str]:
    """Extract implemented dialect operations from a JSON file.

    Args:
        filepath (str): Path to the JSON file.
        dialect_ops (set[str]): Set of all dialect operations.

    Returns:
        set[str]: Set of implemented operation names found in the JSON file keys.
    """
    implemented: set[str] = set()
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                for k in data:
                    if k in dialect_ops:
                        implemented.add(k)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError):
        pass
    return implemented


def extract_dynamic_definitions(tree: ast.AST, filepath: str, dialect_ops: set[str]) -> set[str]:
    """Extract dynamically loaded JSON definitions based on ``load_definitions`` AST calls.

    Args:
        tree (ast.AST): Parsed Python AST tree.
        filepath (str): Path to the Python file.
        dialect_ops (set[str]): Set of all dialect operations.

    Returns:
        set[str]: Set of implemented operation names.
    """
    implemented: set[str] = set()
    loaded_definitions_fw = None

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load_definitions"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            loaded_definitions_fw = node.args[0].value

    if loaded_definitions_fw is not None:
        fw_name = (
            loaded_definitions_fw.decode("utf-8", errors="replace")
            if isinstance(loaded_definitions_fw, bytes)
            else str(loaded_definitions_fw)
        )
        base_dir = os.path.dirname(filepath)
        possible_paths = [
            os.path.join(base_dir, f"{fw_name}.json"),
            os.path.join(base_dir, "definitions", f"{fw_name}.json"),
            os.path.join(os.path.dirname(base_dir), "definitions", f"{fw_name}.json"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                implemented.update(parse_json_file(p, dialect_ops))
                break

    return implemented


def _extract_decorator_dialect_ops(decorator_list: list[ast.expr], dialect_ops: set[str]) -> set[str]:
    """Extract dialect operator names from ``@register_op`` decorators on an AST node.

    Args:
        decorator_list (list[ast.expr]): List of decorator AST expressions.
        dialect_ops (set[str]): Canonical dialect operator names.

    Returns:
        set[str]: Matched dialect operator names from decorators.
    """
    matched: set[str] = set()
    for dec in decorator_list:
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Name)
            and dec.func.id == "register_op"
            and len(dec.args) >= 2
            and isinstance(dec.args[1], ast.Constant)
        ):
            op_name = str(dec.args[1].value)
            if op_name in dialect_ops:
                matched.add(op_name)
            elif op_name.capitalize() in dialect_ops:
                matched.add(op_name.capitalize())
    return matched


def _extract_ast_dialect_ops(tree: ast.AST, dialect_ops: set[str]) -> set[str]:
    """Scan an AST for function/class definitions, ``@register_op`` decorators, and dict keys matching dialect ops.

    Args:
        tree (ast.AST): Parsed Python AST.
        dialect_ops (set[str]): Set of canonical dialect operator names.

    Returns:
        set[str]: Implemented operator names discovered in the AST.
    """
    implemented: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in dialect_ops:
                implemented.add(node.name)
            implemented.update(_extract_decorator_dialect_ops(node.decorator_list, dialect_ops))
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value in dialect_ops:
                    implemented.add(key.value)
    return implemented


def _extract_base_class_name(base_expr: ast.expr) -> str | None:
    """Extract the terminal class identifier from a simple, qualified, or subscripted base expression.

    Recognizes ``CompilerBackend``, ``sw_ir.CompilerBackend``, ``CompilerBackend[str]``,
    and ``sw_ir.CompilerBackend[str]``.

    Args:
        base_expr (ast.expr): Base class AST expression node.

    Returns:
        str | None: Unqualified class name if resolvable, else ``None``.
    """
    target = base_expr.value if isinstance(base_expr, ast.Subscript) else base_expr
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _score_ir_method(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Score receiver and input payload parameter requirements (up to 2 points) on an IR interface method.

    Args:
        fn_node (ast.FunctionDef | ast.AsyncFunctionDef): Target ``compile`` or ``parse_to_graph`` AST node.

    Returns:
        int: Additional points in ``{0, 1, 2}`` (1 for receiver/staticmethod, 1 for input argument).
    """
    is_static = any(
        (isinstance(d, ast.Name) and d.id == "staticmethod")
        or (isinstance(d, ast.Attribute) and d.attr == "staticmethod")
        for d in fn_node.decorator_list
    )
    positional = list(fn_node.args.posonlyargs) + list(fn_node.args.args)
    kwonly = list(fn_node.args.kwonlyargs)
    has_varargs = fn_node.args.vararg is not None

    if is_static:
        has_input = len(positional) + len(kwonly) >= 1 or has_varargs
        return 2 if has_input else 1

    if not positional:
        return 0
    has_input = (len(positional) - 1 + len(kwonly)) >= 1 or has_varargs
    return 2 if has_input else 1


def _score_ir_classes(tree: ast.AST) -> tuple[bool, tuple[int, int]]:
    """Detect ``CompilerBackend`` / ``GraphFrontend`` subclasses and score 5 genuine requirements from ``0/5``.

    Requirements (1 point each):
        1. Valid ``ast.ClassDef`` declaration.
        2. Inherits ``CompilerBackend`` or ``GraphFrontend`` (including qualified/subscripted forms).
        3. Defines ``compile`` (backend) or ``parse_to_graph`` (frontend).
        4. Declares method receiver (``self``/``cls``) or ``@staticmethod``.
        5. Accepts at least one input payload parameter (any valid parameter name).

    Args:
        tree (ast.AST): Parsed Python AST.

    Returns:
        tuple[bool, tuple[int, int]]: ``(is_ir, (met_requirements, 5))``.
    """
    is_ir = False
    best_req_met = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {_extract_base_class_name(b) for b in node.bases}
        is_backend = "CompilerBackend" in base_names
        is_frontend = "GraphFrontend" in base_names
        if not (is_backend or is_frontend):
            continue
        is_ir = True
        current_met = 2  # (1) ClassDef + (2) inherits CompilerBackend / GraphFrontend
        target_method = "compile" if is_backend else "parse_to_graph"
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == target_method:
                current_met += 1 + _score_ir_method(item)
                break
        best_req_met = max(best_req_met, current_met)
    return is_ir, ((best_req_met, 5) if is_ir else (0, 5))


def _score_fw_adapters(tree: ast.AST) -> tuple[bool, tuple[int, int]]:
    """Detect ``FrameworkAdapter`` / ``Importer`` classes and compute their compliance score.

    Args:
        tree (ast.AST): Parsed Python AST.

    Returns:
        tuple[bool, tuple[int, int]]: ``(is_fw, (met_requirements, 4))``.
    """
    is_fw = False
    best_req_met = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        has_fw_dec = any(
            isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "register_framework"
            for d in node.decorator_list
        )
        if "Adapter" in node.name or "Importer" in node.name or has_fw_dec:
            is_fw = True
            current_met = 2
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            if methods & {"convert", "import_func", "parse"}:
                current_met += 1
            if methods & {"definitions", "import_jaxpr"}:
                current_met += 1
            best_req_met = max(best_req_met, current_met)
    return is_fw, ((best_req_met, 4) if is_fw else (0, 4))


def analyze_python_file(filepath: str, dialect_ops: set[str]) -> FileComplianceResult:
    """Analyze a Python file for IR, FrameworkAdapter, and Dialect compliance.

    Args:
        filepath (str): Path to the Python file.
        dialect_ops (set[str]): Set of all dialect operations.

    Returns:
        FileComplianceResult: Model with compliance findings.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return FileComplianceResult()

    is_ir, ir_score = _score_ir_classes(tree)
    is_fw, fw_score = _score_fw_adapters(tree)
    implemented = _extract_ast_dialect_ops(tree, dialect_ops)
    implemented.update(extract_dynamic_definitions(tree, filepath, dialect_ops))

    return FileComplianceResult(
        is_ir=is_ir,
        ir_score=ir_score,
        is_fw=is_fw,
        fw_score=fw_score,
        implemented_ops=implemented,
    )


def _format_snapshot_param(param: GhostParam) -> str:
    """Format a single GhostParam into a Python signature parameter string.

    Args:
        param (GhostParam): Snapshot parameter specification.

    Returns:
        str: Formatted parameter string (e.g. ``'x: Tensor = None'``).
    """
    prefix = "*" if param.kind == "VAR_POSITIONAL" else ("**" if param.kind == "VAR_KEYWORD" else "")
    rendered = f"{prefix}{param.name}"
    if param.annotation:
        rendered += f": {param.annotation}"
    if param.default is not None:
        rendered += f" = {param.default}"
    return rendered


def _format_snapshot_signature(ref: ExtendedGhostRef) -> str:
    """Format an ExtendedGhostRef parameter list and return type as a Python signature.

    Args:
        ref (ExtendedGhostRef): Grounded snapshot symbol reference.

    Returns:
        str: Formatted signature string.
    """
    if not ref.params and ref.kind == "class":
        return "Class/Type"
    joined = ", ".join(_format_snapshot_param(p) for p in ref.params)
    sig = f"({joined})"
    if ref.returns_type:
        sig += f" -> {ref.returns_type}"
    return sig


def _lookup_snapshot_api_metadata(framework: str, api_path: str, symbol: str) -> tuple[str, str]:
    """Resolve signature and docstring strictly from grounded snapshot manifests without live imports.

    Args:
        framework (str): Target framework identifier (e.g. ``'torch'``, ``'jax'``).
        api_path (str): Fully qualified API path.
        symbol (str): Unqualified symbol name.

    Returns:
        tuple[str, str]: Tuple of ``(signature, docstring)``.
    """
    from ml_switcheroo_ir.snapshots import get_effective_snapshot_dir

    try:
        from ml_ecosystem_snapshots.grounding.engine import GroundingEngine

        snap_dir = get_effective_snapshot_dir()
        base_dirs = [snap_dir] if os.path.isdir(snap_dir) else None
        engine = GroundingEngine(base_dirs=base_dirs)
        ref = engine.get_symbol(framework, api_path) or engine.get_symbol(framework, symbol)
        if ref is not None:
            sig = _format_snapshot_signature(ref)
            doc = "No docstring available."
            if ref.docstring:
                doc = ref.docstring.strip().split(_NL * 2)[0].replace(_NL, " ")
                if len(doc) > 150:
                    doc = doc[:147] + "..."
            return sig, doc
    except (ImportError, SyntaxError, AttributeError, KeyError, OSError, ValueError, TypeError, RuntimeError):
        pass
    return "Unknown", "No docstring available."


def _resolve_mapped_api_row(raw_api_path: str) -> list[str]:
    """Resolve framework, namespace, symbol, signature, and docstring for a mapped API path.

    Args:
        raw_api_path (str): Raw API path string from a mapping JSON file.

    Returns:
        list[str]: Row cells ``["[ ]", framework, namespace, symbol, fqn, signature, docstring]``.
    """
    api_path = raw_api_path
    if api_path.startswith("jnp."):
        api_path = api_path.replace("jnp.", "jax.numpy.", 1)
    elif api_path.startswith("tf."):
        api_path = api_path.replace("tf.", "tensorflow.", 1)

    parts = api_path.split(".")
    framework = parts[0]
    symbol = parts[-1]
    namespace = ".".join(parts[:-1]) if len(parts) > 1 else framework
    signature, docstring = _lookup_snapshot_api_metadata(framework, api_path, symbol)

    return ["[ ]", framework, namespace, symbol, api_path, signature, docstring]


def _aggregate_compliance_results(
    files: Sequence[str],
    targets: Sequence[str],
    dialect_ops: set[str],
) -> tuple[bool, tuple[int, int], bool, tuple[int, int], set[str], bool]:
    """Aggregate IR, FrameworkAdapter, and Dialect compliance scores across analyzed files.

    Args:
        files (Sequence[str]): Collected file paths to analyze.
        targets (Sequence[str]): Original target paths.
        dialect_ops (set[str]): Canonical dialect operator set.

    Returns:
        tuple[bool, tuple[int, int], bool, tuple[int, int], set[str], bool]:
            ``(found_ir, ir_score, found_fw, fw_score, dialect_ops, found_dialect)``.
    """
    best_ir_score = (0, 5)
    best_fw_score = (0, 4)
    found_any_ir = False
    found_any_fw = False
    aggregated_dialect_ops: set[str] = set()
    found_any_dialect_targets = False
    explicit_files = {os.path.abspath(t) for t in targets if os.path.isfile(t)}

    for filepath in files:
        if filepath.endswith(".json"):
            ops = parse_json_file(filepath, dialect_ops)
            aggregated_dialect_ops.update(ops)
            found_any_dialect_targets = found_any_dialect_targets or bool(ops)
            continue

        res = analyze_python_file(filepath, dialect_ops)
        if res.is_ir:
            found_any_ir = True
            if res.ir_score[0] > best_ir_score[0]:
                best_ir_score = res.ir_score
        if res.is_fw:
            found_any_fw = True
            if res.fw_score[0] > best_fw_score[0]:
                best_fw_score = res.fw_score
        if res.implemented_ops:
            aggregated_dialect_ops.update(res.implemented_ops)
            found_any_dialect_targets = True
        if not res.is_ir and not res.is_fw and filepath in explicit_files:
            found_any_dialect_targets = True

    return (
        found_any_ir,
        best_ir_score,
        found_any_fw,
        best_fw_score,
        aggregated_dialect_ops,
        found_any_dialect_targets,
    )


def _format_target_label(targets: Sequence[str]) -> str:
    """Compute the display label for a sequence of compliance targets.

    Args:
        targets (Sequence[str]): Target file or directory paths.

    Returns:
        str: Formatted target label.
    """
    if len(targets) == 1 and os.path.isfile(targets[0]):
        return os.path.basename(targets[0])
    if len(targets) == 1:
        return os.path.basename(os.path.abspath(targets[0])) + " (Directory)"
    return f"Multiple Targets ({len(targets)})"


def _populate_verbose_report_details(
    missing_ops_list: list[str],
    mapping_file: str | None,
) -> tuple[list[list[str]], dict[str, dict[str, JsonValue]]]:
    """Build structured verbose missing-operation details without printing.

    Args:
        missing_ops_list (list[str]): Sorted missing operator names.
        mapping_file (str | None): Optional mapping JSON path.

    Returns:
        tuple[list[list[str]], dict[str, dict[str, JsonValue]]]:
            ``(mapped_rows, unmapped_schema_summaries)``.
    """
    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, encoding="utf-8") as f:
            mapping_data = json.load(f)
        rows = [
            _resolve_mapped_api_row(str(mapping_data[op]["api"]))
            for op in missing_ops_list
            if isinstance(mapping_data, dict)
            and op in mapping_data
            and isinstance(mapping_data[op], dict)
            and "api" in mapping_data[op]
        ]
        return rows, {}

    dialect_ops_map = {k.split(".")[-1]: v for k, v in ONNX_REGISTRY.items()}
    summaries: dict[str, dict[str, JsonValue]] = {}
    for op in missing_ops_list:
        if op in dialect_ops_map:
            schema = dialect_ops_map[op]
            attrs_out: dict[str, JsonValue] = {}
            for name, attr in schema.attributes.items():
                def_val: JsonValue = (
                    attr.default
                    if attr.default is None or isinstance(attr.default, (str, int, float, bool))
                    else str(attr.default)
                )
                attrs_out[name] = {
                    "type": attr.type,
                    "required": attr.required,
                    "default": def_val,
                }
            summaries[op] = {
                "domain": schema.domain,
                "version": schema.version,
                "inputs": list(schema.inputs),
                "outputs": list(schema.outputs),
                "attributes": attrs_out,
            }
    return [], summaries


def evaluate_compliance(
    targets: Sequence[str],
    verbose: bool = False,
    mapping_file: str | None = None,
) -> ComplianceReport:
    """Evaluate IR, FrameworkAdapter, and Dialect compliance cleanly without I/O side effects.

    Args:
        targets (Sequence[str]): Sequence of file or directory paths to evaluate.
        verbose (bool): Whether to populate detailed missing-operator metadata.
        mapping_file (str | None): Optional path to a framework mapping JSON file.

    Returns:
        ComplianceReport: Pure typed Pydantic compliance report.

    Raises:
        ValueError: If ``targets`` is empty.
        FileNotFoundError: If any target path does not exist.
    """
    if not targets:
        raise ValueError("No targets provided.")

    files = collect_files(targets)
    dialect_ops = get_dialect_ops()
    total_dialect_ops = len(dialect_ops)
    (
        found_any_ir,
        best_ir_score,
        found_any_fw,
        best_fw_score,
        aggregated_dialect_ops,
        found_any_dialect_targets,
    ) = _aggregate_compliance_results(files, targets, dialect_ops)

    implemented_sorted = sorted(aggregated_dialect_ops)
    missing_sorted = sorted(dialect_ops - aggregated_dialect_ops)
    mapped_rows: list[list[str]] = []
    unmapped_schemas: dict[str, dict[str, JsonValue]] = {}
    if verbose and found_any_dialect_targets and missing_sorted:
        mapped_rows, unmapped_schemas = _populate_verbose_report_details(missing_sorted, mapping_file)

    return ComplianceReport(
        target_label=_format_target_label(targets),
        analyzed_files=files,
        found_ir=found_any_ir,
        ir_score=best_ir_score,
        found_fw=found_any_fw,
        fw_score=best_fw_score,
        found_dialect=found_any_dialect_targets,
        implemented_ops=implemented_sorted,
        missing_ops=missing_sorted,
        total_dialect_ops=total_dialect_ops,
        mapped_missing_rows=mapped_rows,
        unmapped_missing_schemas=unmapped_schemas,
        mapping_file=mapping_file,
    )


def _format_verbose_section(report: ComplianceReport) -> list[str]:
    """Render the verbose missing-operations section lines from a ``ComplianceReport``.

    Args:
        report (ComplianceReport): Evaluated compliance report.

    Returns:
        list[str]: Lines of formatted verbose report text.
    """
    lines: list[str] = ["", "Verbose Missing Operations Report", "================================="]
    if report.mapping_file and os.path.exists(report.mapping_file):
        if report.mapped_missing_rows:
            lines.append(f"{_NL}### {report.target_label} (Mapped API targets)")
            lines.append(
                tabulate(
                    report.mapped_missing_rows,
                    headers=["Implemented", "Framework", "Namespace", "Symbol", "FQN", "Signature", "Docstring"],
                    tablefmt="github",
                )
            )
        else:
            lines.append(f"{_NL}### {report.target_label} (No mappings found in {report.mapping_file})")
        return lines

    lines.append(f"{_NL}### {report.target_label}")
    for op, schema_dict in report.unmapped_missing_schemas.items():
        lines.append(f"- [ ] **{op}**")
        lines.append("  ```json")
        for json_line in json.dumps(schema_dict, indent=2).splitlines():
            lines.append(f"  {json_line}")
        lines.append("  ```")
    return lines


def format_compliance_report(report: ComplianceReport, verbose: bool = False) -> str:
    """Render a ``ComplianceReport`` into human-readable CLI output without side effects.

    Args:
        report (ComplianceReport): Typed compliance report model.
        verbose (bool): Whether to include verbose missing-operation details.

    Returns:
        str: Formatted report text.
    """
    lines: list[str] = []
    if not report.analyzed_files:
        lines.append("No Python or JSON files found in the specified targets.")

    lines.extend(["", "Compliance Report", "================="])

    if report.found_ir:
        pct = (report.ir_score[0] / report.ir_score[1]) * 100
        status = "PASS" if pct == 100 else "FAIL"
        ir_rows = [[report.target_label, f"{pct:.0f}%", f"{report.ir_score[0]}/{report.ir_score[1]}", status]]
        lines.append(f"{_NL}IR Compliance (CompilerBackend / GraphFrontend):")
        lines.append(tabulate(ir_rows, headers=["Target", "Compliance %", "Reqs Met", "Status"]))

    if report.found_fw:
        pct = (report.fw_score[0] / report.fw_score[1]) * 100
        status = "PASS" if pct == 100 else "FAIL"
        fw_rows = [[report.target_label, f"{pct:.0f}%", f"{report.fw_score[0]}/{report.fw_score[1]}", status]]
        lines.append(f"{_NL}FrameworkAdapter Compliance:")
        lines.append(tabulate(fw_rows, headers=["Target", "Compliance %", "Reqs Met", "Status"]))

    if report.found_dialect:
        total = max(report.total_dialect_ops, 1)
        pct = (len(report.implemented_ops) / total) * 100
        status = "PASS" if len(report.implemented_ops) == report.total_dialect_ops else "FAIL"
        d_rows = [
            [
                report.target_label,
                f"{pct:.1f}%",
                f"{len(report.implemented_ops)}/{report.total_dialect_ops}",
                status,
                len(report.missing_ops),
            ]
        ]
        lines.append(f"{_NL}DIALECT Compliance (ONNX Superset, {report.total_dialect_ops} total ops):")
        lines.append(
            tabulate(
                d_rows,
                headers=["Target", "Compliance %", "Ops Implemented", "Status", "Missing Ops"],
            )
        )

    if not report.found_ir and not report.found_dialect and not report.found_fw:
        lines.append("No IR, FrameworkAdapter, or DIALECT targets detected in the specified path.")

    if verbose and report.found_dialect and report.missing_ops:
        lines.extend(_format_verbose_section(report))

    return _NL.join(lines)


def run_compliance_check(
    targets: Sequence[str],
    verbose: bool = False,
    mapping_file: str | None = None,
) -> ComplianceReport:
    """Pure library wrapper returning a ``ComplianceReport`` without ``print`` or ``sys.exit``.

    Args:
        targets (Sequence[str]): Sequence of file or directory paths to check.
        verbose (bool): Whether to populate detailed missing operation schemas.
        mapping_file (str | None): Optional path to a mapping JSON file.

    Returns:
        ComplianceReport: Evaluated compliance report model.
    """
    return evaluate_compliance(targets=targets, verbose=verbose, mapping_file=mapping_file)
