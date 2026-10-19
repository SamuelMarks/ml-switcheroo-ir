"""Empirical anti-hallucination grounding diagnostic, report, and validator models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
import gzip
import json
import logging
import os
import typing
from typing_extensions import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from ml_switcheroo_ir.graph import LogicalGraph, LogicalNode
from ml_switcheroo_ir.schema.ghost import ExtendedGhostRef
from ml_switcheroo_ir.schema.onnx_registry import OpSchema
from ml_switcheroo_ir.snapshots import (
    get_effective_snapshot_dir as _get_snapshot_dir,
)
from ml_switcheroo_ir.validation.core import (
    KNOWN_SYNONYMS,
    ValidationError,
    ValidationLevel,
    compute_levenshtein,
)
from ml_switcheroo_ir.validation.schema_validator import Validator

logger = logging.getLogger(__name__)

__all__ = [
    "DiagnosticSeverity",
    "GroundingAuditReport",
    "GroundingDiagnostic",
    "GroundingEngineProtocol",
    "GroundingReport",
    "GroundingValidator",
    "SnapshotManifestType",
    "audit_graph_grounding",
]

SnapshotManifestType: TypeAlias = typing.Union[
    Mapping[str, object], Sequence[object], str, None
]


class GroundingEngineProtocol(typing.Protocol):
    """Protocol for external ml_ecosystem_snapshots GroundingEngine."""

    def get_symbol(self, target: str, symbol: str) -> ExtendedGhostRef | None:
        """Look up a grounded symbol in the given target.

        Args:
            target (str): Target framework or dialect.
            symbol (str): Symbol name.

        Returns:
            ExtendedGhostRef | None: Grounded symbol reference if found.
        """

    def suggest_closest_symbol(
        self, target: str, symbol: str, max_distance: int = 3
    ) -> str | None:
        """Suggest the closest matching symbol name in the target.

        Args:
            target (str): Target framework or dialect.
            symbol (str): Ungrounded symbol name.
            max_distance (int): Maximum Levenshtein distance.

        Returns:
            str | None: Suggested symbol name if within distance.
        """


class GroundingAuditReport(BaseModel):
    """Audit report detailing graph grounding against ground-truth framework snapshots.

    Attributes:
        total_nodes (int): Total number of nodes in graph.
        grounded_count (int): Number of nodes successfully grounded against snapshot.
        ungrounded_count (int): Number of ungrounded or hallucinated nodes.
        hallucination_score (float): Percentage of nodes that are ungrounded (0.0 to 1.0).
        diagnostics (list[ValidationError]): Diagnostic error reports.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    total_nodes: int
    grounded_count: int
    ungrounded_count: int
    hallucination_score: float
    diagnostics: list[ValidationError]

    def __init__(
        self,
        total_nodes: int,
        grounded_count: int,
        ungrounded_count: int,
        hallucination_score: float,
        diagnostics: list[ValidationError],
    ) -> None:
        """Initialize GroundingAuditReport.

        Args:
            total_nodes (int): Total number of nodes in graph.
            grounded_count (int): Number of nodes successfully grounded against snapshot.
            ungrounded_count (int): Number of ungrounded or hallucinated nodes.
            hallucination_score (float): Percentage of nodes that are ungrounded (0.0 to 1.0).
            diagnostics (list[ValidationError]): Diagnostic error reports.
        """
        super().__init__(
            total_nodes=total_nodes,
            grounded_count=grounded_count,
            ungrounded_count=ungrounded_count,
            hallucination_score=hallucination_score,
            diagnostics=diagnostics,
        )


class DiagnosticSeverity(str, Enum):
    """Severity classification for grounding diagnostics."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class GroundingDiagnostic(BaseModel):
    """Specific diagnostic message emitted during symbol or operation verification.

    Attributes:
        field (str): The component, attribute, or operand evaluated.
        message (str): Human-readable diagnostic explanation.
        severity (DiagnosticSeverity): Severity level of the diagnostic.
        suggested_fix (str | None): Suggested replacement or typo correction.
    """

    field: str = Field(description="The component, attribute, or operand evaluated.")
    message: str = Field(description="Human-readable diagnostic explanation.")
    severity: DiagnosticSeverity = Field(
        default=DiagnosticSeverity.ERROR,
        description="Severity level of the diagnostic.",
    )
    suggested_fix: str | None = Field(
        default=None,
        description="Suggested replacement or typo correction.",
    )


class GroundingReport(BaseModel):
    """Comprehensive validation report for a verified operation or symbol.

    Attributes:
        is_grounded (bool): True if the symbol is grounded and valid without fatal errors.
        target (str): Target framework, dialect, or ISA.
        symbol (str): Queried symbol, mnemonic, or operation identifier.
        diagnostics (list[GroundingDiagnostic]): Collection of diagnostics produced.
        matched_ref (ExtendedGhostRef | None): Resolved ground-truth reference object if discovered.
    """

    is_grounded: bool = Field(
        description="True if the symbol is grounded and valid without fatal errors."
    )
    target: str = Field(description="Target framework, dialect, or ISA.")
    symbol: str = Field(
        description="Queried symbol, mnemonic, or operation identifier."
    )
    diagnostics: list[GroundingDiagnostic] = Field(
        default_factory=list,
        description="Collection of diagnostics produced during verification.",
    )
    matched_ref: ExtendedGhostRef | None = Field(
        default=None,
        description="Resolved ground-truth reference object if discovered.",
    )

    @property
    def has_errors(self) -> bool:
        """Check if any diagnostics have ERROR severity.

        Returns:
            bool: True if any diagnostic is an ERROR, False otherwise.
        """
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)

    def add_diagnostic(
        self,
        field: str,
        message: str,
        severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
        suggested_fix: str | None = None,
    ) -> None:
        """Append a new diagnostic message and update is_grounded status.

        Args:
            field (str): Component or attribute path.
            message (str): Human-readable error message.
            severity (DiagnosticSeverity): Severity level of the diagnostic.
            suggested_fix (str | None): Suggested correction or candidate symbol.
        """
        if severity == DiagnosticSeverity.ERROR:
            self.is_grounded = False
        self.diagnostics.append(
            GroundingDiagnostic(
                field=field,
                message=message,
                severity=severity,
                suggested_fix=suggested_fix,
            )
        )


class GroundingValidator(Validator):
    """Validator that audits graphs strictly against external ground-truth snapshot manifests."""

    def __init__(
        self,
        snapshot_manifest: SnapshotManifestType = None,
        snapshots_dir: str | None = None,
        registry: dict[str, OpSchema] | None = None,
        use_default_if_none: bool = False,
    ) -> None:
        """Initialize GroundingValidator with external snapshot file, directory, list, or dict.

        Args:
            snapshot_manifest (SnapshotManifestType): Snapshot dictionary,
                list of records, file path, directory path, or collection.
            snapshots_dir (str | None): Explicit path to snapshots directory.
            registry (dict[str, OpSchema] | None): Base operator registry.
            use_default_if_none (bool): If True and snapshot_manifest is None, load from DEFAULT_SNAPSHOT_DIR.
        """
        super().__init__(registry=registry)
        self.grounded_symbols: dict[str, dict[str, object]] = {}
        self.concept_map: dict[str, dict[str, object]] = {}
        self.parameter_translations: dict[str, object] = {}
        self._engine: GroundingEngineProtocol | None = None

        target = snapshot_manifest if snapshot_manifest is not None else snapshots_dir
        default_dir = _get_snapshot_dir()

        try:
            from ml_ecosystem_snapshots.grounding.engine import GroundingEngine

            search_dirs: list[str] = []
            if isinstance(target, str) and os.path.isdir(target):
                search_dirs.append(target)
            if (
                snapshots_dir
                and os.path.isdir(snapshots_dir)
                and snapshots_dir not in search_dirs
            ):
                search_dirs.append(snapshots_dir)
            if os.path.isdir(default_dir) and default_dir not in search_dirs:
                search_dirs.append(default_dir)
            self._engine = GroundingEngine(
                base_dirs=search_dirs if search_dirs else None
            )
        except (ImportError, AttributeError, ValueError, OSError, RuntimeError) as exc:
            logger.debug("Failed initializing GroundingEngine: %s", exc)
            self._engine = None

        if target is not None:
            self._ingest_manifest_target(target)
        elif use_default_if_none and os.path.isdir(default_dir):
            self._load_directory(default_dir)

    def _ingest_manifest_target(self, target: object) -> None:
        """Ingest snapshot manifest from diverse target types.

        Args:
            target (object): Directory path, file path, dict, or list.
        """
        if isinstance(target, str):
            if os.path.isdir(target):
                self._load_directory(target)
            elif os.path.isfile(target):
                self._load_file(target)
        elif isinstance(target, Mapping):
            self._load_snapshot_data(target)
        elif isinstance(target, list):
            if target and all(
                isinstance(item, str)
                or (
                    isinstance(item, Mapping)
                    and not ("api_path" in item or "name" in item or "mnemonic" in item)
                )
                for item in target
            ):
                for item in target:
                    self._ingest_manifest_target(item)
            else:
                self._load_snapshot_data(target)

    def _load_file(self, filepath: str) -> None:
        """Load and parse a single snapshot file (.json or .json.gz).

        Args:
            filepath (str): Path to snapshot file.
        """
        try:
            if filepath.endswith(".gz"):
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            self._load_snapshot_data(data)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError) as exc:
            logger.debug("Skipping invalid snapshot file %s: %s", filepath, exc)
            return

    def _load_directory(self, dirpath: str) -> None:
        """Recursively search and load all .json and .json.gz snapshot files in directory.

        Args:
            dirpath (str): Path to root directory.
        """
        for root, _, files in os.walk(dirpath):
            for fname in sorted(files):
                if fname.endswith((".json", ".json.gz")):
                    self._load_file(os.path.join(root, fname))

    def _register_symbol_record(
        self, item: Mapping[str, object], default_key: str | None = None
    ) -> None:
        """Register an individual symbol record under primary and short keys.

        Args:
            item (Mapping[str, object]): Symbol definition record.
            default_key (str | None): Fallback key if neither api_path, name, nor mnemonic is present.
        """
        item_dict = {str(k): v for k, v in item.items()}
        api_path = item_dict.get("api_path")
        name = item_dict.get("name")
        mnemonic = item_dict.get("mnemonic")

        keys = [
            str(k)
            for k in [api_path, name, mnemonic, default_key]
            if isinstance(k, str) and k
        ]
        for k in keys:
            self.grounded_symbols[k] = item_dict

        if isinstance(api_path, str) and "." in api_path:
            unqualified = api_path.split(".")[-1]
            if unqualified not in self.grounded_symbols:
                self.grounded_symbols[unqualified] = item_dict

    def _load_snapshot_data(
        self, data: Mapping[str, object] | Sequence[object]
    ) -> None:
        """Load snapshot entries from categorized, flat, or top-level list formats.

        Args:
            data (Mapping[str, object] | Sequence[object]): Parsed snapshot data.
        """
        if isinstance(data, list):
            for item in data:
                if isinstance(item, Mapping):
                    self._register_symbol_record(item)
        elif isinstance(data, Mapping):
            raw_pt = data.get("_parameter_translations")
            if isinstance(raw_pt, Mapping):
                for pk, pv in raw_pt.items():
                    self.parameter_translations[str(pk)] = pv

            raw_cats = data.get("categories")
            if isinstance(raw_cats, Mapping):
                for cat_list in raw_cats.values():
                    if isinstance(cat_list, list):
                        for item in cat_list:
                            if isinstance(item, Mapping):
                                self._register_symbol_record(item)
                    elif isinstance(cat_list, Mapping):
                        for item in cat_list.values():
                            if isinstance(item, Mapping):
                                self._register_symbol_record(item)
            else:
                for k, v in data.items():
                    if k == "_parameter_translations":
                        continue
                    if isinstance(v, Mapping):
                        v_dict = {str(vk): vv for vk, vv in v.items()}
                        if any(
                            fw in v_dict
                            for fw in (
                                "torch",
                                "jax",
                                "tensorflow",
                                "tf",
                                "stablehlo",
                                "numpy",
                            )
                        ):
                            self.concept_map[str(k)] = v_dict
                        self._register_symbol_record(v_dict, default_key=str(k))
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, Mapping):
                                self._register_symbol_record(item)

    def _find_best_symbol_match(self, op_type: str, domain: str) -> str | None:
        """Find the closest matching symbol in the snapshot within an edit distance threshold.

        Args:
            op_type (str): The ungrounded operation type.
            domain (str): The operation domain.

        Returns:
            str | None: Closest matching symbol name, or None.
        """
        synonym = KNOWN_SYNONYMS.get((domain, op_type))
        if synonym and (
            synonym in self.grounded_symbols
            or synonym.split(".")[-1] in self.grounded_symbols
        ):
            return synonym

        best_match: str | None = None
        best_dist = 4
        full_candidate = f"{domain}.{op_type}" if domain else op_type
        domain_prefix = f"{domain}." if domain else ""

        if domain_prefix:
            for sym in self.grounded_symbols:
                if sym.startswith(domain_prefix):
                    unqual = sym[len(domain_prefix) :]
                    if (
                        abs(len(op_type) - len(unqual)) >= best_dist
                        and abs(len(full_candidate) - len(sym)) >= best_dist
                    ):
                        continue
                    d = min(
                        compute_levenshtein(op_type, unqual),
                        compute_levenshtein(full_candidate, sym),
                    )
                    if d < best_dist:
                        best_dist = d
                        best_match = sym

        if best_match is not None:
            return best_match

        for sym in self.grounded_symbols:
            if abs(len(op_type) - len(sym)) >= best_dist:
                continue
            d = compute_levenshtein(op_type, sym)
            if d < best_dist:
                best_dist = d
                best_match = sym

        discover_fn = getattr(self._engine, "_discover_target_files", None)
        if (
            best_match is None
            and self._engine is not None
            and domain
            and (not callable(discover_fn) or bool(discover_fn(domain)))
        ):
            try:
                engine_cand = self._engine.suggest_closest_symbol(
                    domain, op_type, max_distance=3
                )
                if engine_cand:
                    best_match = engine_cand
            except (
                AttributeError,
                KeyError,
                ValueError,
                RuntimeError,
                TypeError,
            ) as exc:
                logger.debug("Failed symbol suggestion from engine: %s", exc)

        return best_match

    def _find_best_attr_match(self, attr_key: str, candidates: set[str]) -> str | None:
        """Find closest matching attribute or parameter name.

        Args:
            attr_key (str): Hallucinated attribute key.
            candidates (set[str]): Valid known attribute/parameter names.

        Returns:
            str | None: Closest candidate name if edit distance <= 3, else None.
        """
        best_match: str | None = None
        best_dist = 4
        for cand in candidates:
            if abs(len(attr_key) - len(cand)) >= best_dist:
                continue
            d = compute_levenshtein(attr_key, cand)
            if d < best_dist:
                best_dist = d
                best_match = cand
        return best_match

    def validate_grounding(self, node: LogicalNode) -> list[ValidationError]:
        """Check if node corresponds to a grounded symbol in the snapshot.

        Args:
            node (LogicalNode): The node to inspect.

        Returns:
            list[ValidationError]: Diagnostic errors if symbol is ungrounded.
        """
        errors: list[ValidationError] = []
        full_path = f"{node.domain}.{node.op_type}"

        match: Mapping[str, object] | None = self.grounded_symbols.get(
            full_path
        ) or self.grounded_symbols.get(node.op_type)
        discover_fn = getattr(self._engine, "_discover_target_files", None)
        if (
            match is None
            and self._engine is not None
            and node.domain
            and (not callable(discover_fn) or bool(discover_fn(node.domain)))
        ):
            try:
                ref = self._engine.get_symbol(node.domain, node.op_type)
                if ref is not None:
                    match = (
                        ref.model_dump() if hasattr(ref, "model_dump") else vars(ref)
                    )
            except (
                AttributeError,
                KeyError,
                ValueError,
                RuntimeError,
                TypeError,
            ) as exc:
                logger.debug("Failed symbol lookup from engine: %s", exc)

        if match is None:
            suggestion = self._find_best_symbol_match(node.op_type, node.domain)
            msg = f"Ungrounded symbol '{node.op_type}' in domain '{node.domain}'. Symbol not found in framework snapshot."
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            errors.append(
                ValidationError(
                    node_id=node.id,
                    attribute="kind",
                    message=msg,
                    level=ValidationLevel.ERROR,
                    suggested_fix=suggestion,
                )
            )
        else:
            known_params: set[str] = set()
            raw_params = match.get("params")
            if isinstance(raw_params, list):
                for p in raw_params:
                    if isinstance(p, dict) and isinstance(p.get("name"), str):
                        known_params.add(str(p["name"]))

            attrs = match.get("attributes")
            if isinstance(attrs, dict):
                known_params.update(str(k) for k in attrs.keys())
            elif isinstance(attrs, list):
                for a in attrs:
                    if isinstance(a, dict) and isinstance(a.get("name"), str):
                        known_params.add(str(a["name"]))
                    elif isinstance(a, str):
                        known_params.add(a)

            operands = match.get("operands")
            if isinstance(operands, list):
                for op in operands:
                    if isinstance(op, dict) and isinstance(op.get("name"), str):
                        known_params.add(str(op["name"]))
                    elif isinstance(op, str):
                        known_params.add(op)

            accepted_kwargs = match.get("accepted_kwargs") or match.get("kwargs")
            if isinstance(accepted_kwargs, list):
                for kw in accepted_kwargs:
                    if isinstance(kw, str):
                        known_params.add(kw)
                    elif isinstance(kw, dict) and isinstance(kw.get("name"), str):
                        known_params.add(str(kw["name"]))

            if known_params:
                for attr_key in node.attributes:
                    if attr_key not in known_params:
                        attr_suggestion = self._find_best_attr_match(
                            attr_key, known_params
                        )
                        msg = f"Ungrounded attribute '{attr_key}' on '{node.op_type}'. Allowed attributes/params: {sorted(known_params)}."
                        if attr_suggestion:
                            msg += f" Did you mean '{attr_suggestion}'?"
                        errors.append(
                            ValidationError(
                                node_id=node.id,
                                attribute=attr_key,
                                message=msg,
                                level=ValidationLevel.ERROR,
                                suggested_fix=attr_suggestion,
                            )
                        )
        return errors

    def audit_graph(self, graph: LogicalGraph) -> GroundingAuditReport:
        """Audit an entire graph for anti-hallucination grounding against snapshots.

        Args:
            graph (LogicalGraph): Graph to audit.

        Returns:
            GroundingAuditReport: Comprehensive grounding report.
        """
        diagnostics: list[ValidationError] = []
        grounded_count = 0
        total_nodes = len(graph.nodes)

        for node in graph.nodes.values():
            node_errors = self.validate_grounding(node)
            if node_errors:
                diagnostics.extend(node_errors)
            else:
                grounded_count += 1

        ungrounded_count = total_nodes - grounded_count
        score = (ungrounded_count / total_nodes) if total_nodes > 0 else 0.0

        return GroundingAuditReport(
            total_nodes=total_nodes,
            grounded_count=grounded_count,
            ungrounded_count=ungrounded_count,
            hallucination_score=round(score, 4),
            diagnostics=diagnostics,
        )


def audit_graph_grounding(
    graph: LogicalGraph,
    snapshots_path: SnapshotManifestType = None,
) -> GroundingAuditReport:
    """Audit a LogicalGraph against framework snapshots.

    Args:
        graph (LogicalGraph): Logical graph to audit.
        snapshots_path (SnapshotManifestType): Path to snapshot manifest,
            directory, or loaded dictionary/list.

    Returns:
        GroundingAuditReport: Resulting grounding audit report.
    """
    default_dir = _get_snapshot_dir()
    resolved_path: SnapshotManifestType
    if snapshots_path is None and os.path.isdir(default_dir):
        resolved_path = default_dir
    else:
        resolved_path = snapshots_path
    validator = GroundingValidator(snapshot_manifest=resolved_path)
    return validator.audit_graph(graph)
