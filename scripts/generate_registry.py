"""Generator for ONNX operator schema catalog from structured snapshots and onnx.defs."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path

import pydantic
import typing_extensions

from ml_switcheroo_ir.schema.base import AttributeType, normalize_attribute_type
from ml_switcheroo_ir.snapshots import get_default_snapshots_dir
from ml_switcheroo_ir.types import JsonValue


class OnnxAttrSpecModel(pydantic.BaseModel):
    """Pydantic model for a parsed ONNX operator attribute specification.

    Attributes:
        type (str): Canonical AttributeType value string.
        required (bool): Whether the attribute is mandatory.
        default (JsonValue): Default attribute value when omitted.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: str = AttributeType.ANY.value
    required: bool = False
    default: JsonValue = None


class OnnxOpSpecModel(pydantic.BaseModel):
    """Pydantic model for a parsed ONNX operator schema specification.

    Attributes:
        domain (str): Operator domain (default 'ai.onnx').
        version (int): Operator opset since_version.
        attributes (dict[str, dict[str, JsonValue]]): Attribute specification dictionary.
        inputs (list[str]): Formal input tensor names.
        outputs (list[str]): Formal output tensor names.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    domain: str = "ai.onnx"
    version: int = 1
    attributes: dict[str, dict[str, JsonValue]] = pydantic.Field(default_factory=dict)
    inputs: list[str] = pydantic.Field(default_factory=list)
    outputs: list[str] = pydantic.Field(default_factory=list)


OnnxAttrSpecModel.model_rebuild()
OnnxOpSpecModel.model_rebuild()


class OnnxParamProtocol(typing.Protocol):
    """Protocol for an upstream onnx.defs formal parameter.

    Attributes:
        name (str): Formal parameter name.
    """

    name: str


class OnnxDefaultProtoProtocol(typing.Protocol):
    """Protocol for an upstream onnx AttributeProto default value message."""

    def ByteSize(self) -> int:
        """Return serialized protobuf byte size.

        Returns:
            int: Serialized byte count.
        """


class OnnxAttrDefProtocol(typing.Protocol):
    """Protocol for an upstream onnx.defs OpSchema.Attribute descriptor.

    Attributes:
        name (str): Attribute name.
        type (str | int): Upstream attribute type enum or string.
        required (bool): Whether mandatory.
        default_value (OnnxDefaultProtoProtocol): Protobuf default value message.
    """

    name: str
    type: str | int
    required: bool
    default_value: OnnxDefaultProtoProtocol


class OnnxSchemaProtocol(typing.Protocol):
    """Protocol for an upstream onnx.defs OpSchema.

    Attributes:
        name (str): Operator name.
        since_version (int): Opset version introducing this schema.
        domain (str): Operator domain string.
        deprecated (bool): Deprecation flag.
        attributes (Mapping[str, OnnxAttrDefProtocol]): Attribute definitions mapping.
        inputs (Sequence[OnnxParamProtocol]): Formal input parameters.
        outputs (Sequence[OnnxParamProtocol]): Formal output parameters.
    """

    name: str
    since_version: int
    domain: str
    deprecated: bool
    attributes: Mapping[str, OnnxAttrDefProtocol]
    inputs: Sequence[OnnxParamProtocol]
    outputs: Sequence[OnnxParamProtocol]


OnnxOpEntryValue: typing_extensions.TypeAlias = typing.Union[
    str,
    int,
    list[str],
    list[dict[str, JsonValue]],
    dict[str, dict[str, JsonValue]],
]

_RAW_ATTR_TYPE_MAP: dict[str, str] = {
    "int": AttributeType.INT.value,
    "float": AttributeType.FLOAT.value,
    "str": AttributeType.STR.value,
    "string": AttributeType.STR.value,
    "bool": AttributeType.BOOL.value,
    "ints": AttributeType.LIST_INT.value,
    "list[int]": AttributeType.LIST_INT.value,
    "list of ints": AttributeType.LIST_INT.value,
    "floats": AttributeType.LIST_FLOAT.value,
    "list[float]": AttributeType.LIST_FLOAT.value,
    "list of floats": AttributeType.LIST_FLOAT.value,
    "strings": AttributeType.LIST_STR.value,
    "list[str]": AttributeType.LIST_STR.value,
    "list of strings": AttributeType.LIST_STR.value,
    "tensor": AttributeType.TENSOR.value,
    "sparse_tensor": AttributeType.TENSOR.value,
    "tensors": AttributeType.LIST.value,
    "sparse_tensors": AttributeType.LIST.value,
    "graph": AttributeType.GRAPH.value,
    "graphs": AttributeType.LIST.value,
    "type": AttributeType.DTYPE.value,
    "type_proto": AttributeType.DTYPE.value,
    "type_protos": AttributeType.LIST.value,
}


def map_onnx_attribute_type(raw_type: str) -> str:
    """Map an ONNX attribute type descriptor to a canonical ``AttributeType`` string.

    Args:
        raw_type (str): Raw ONNX attribute type (e.g. ``'tensor'``, ``'graph'``, ``'ints'``).

    Returns:
        str: Canonical ``AttributeType`` string value.
    """
    cleaned = raw_type.strip()
    if "." in cleaned:
        cleaned = cleaned.rsplit(".", 1)[-1]
    mapped = _RAW_ATTR_TYPE_MAP.get(cleaned.lower())
    if mapped is not None:
        return mapped
    try:
        return normalize_attribute_type(cleaned).value
    except ValueError:
        return AttributeType.ANY.value


def _parse_proto_default_str(raw: str | None) -> JsonValue:
    """Parse an ONNX AttributeProto text representation into a JSON-compatible default value.

    Args:
        raw (str | None): Raw protobuf text or string literal.

    Returns:
        JsonValue: Extracted scalar default or None.
    """
    if not raw or raw == "None":
        return None
    m_i = re.search(r"\bi:\s*(-?\d+)", raw)
    if m_i:
        return int(m_i.group(1))
    m_f = re.search(r"\bf:\s*(-?[\d\.]+)", raw)
    if m_f:
        return float(m_f.group(1))
    m_s = re.search(r'\bs:\s*"([^"]*)"', raw)
    if m_s:
        return m_s.group(1)
    result: JsonValue = raw
    try:
        parsed: JsonValue = ast.literal_eval(raw)
        if isinstance(parsed, (int, float, str, bool, list, dict)) or parsed is None:
            result = parsed
    except (ValueError, SyntaxError):
        pass
    return result


def _collect_snapshot_items(data: Mapping[str, JsonValue]) -> list[Mapping[str, JsonValue]]:
    """Collect operator item dictionaries from ``categories`` and ``operations`` sections.

    Args:
        data (Mapping[str, JsonValue]): Parsed snapshot manifest dictionary.

    Returns:
        list[Mapping[str, JsonValue]]: Flattened operator item dictionaries.
    """
    raw_items: list[Mapping[str, JsonValue]] = []
    categories = data.get("categories")
    if isinstance(categories, Mapping):
        for items in categories.values():
            if isinstance(items, list):
                raw_items.extend(entry for entry in items if isinstance(entry, Mapping))
    operations = data.get("operations")
    if isinstance(operations, list):
        raw_items.extend(entry for entry in operations if isinstance(entry, Mapping))
    return raw_items


def _snapshot_item_to_onnx_spec(item: Mapping[str, JsonValue]) -> dict[str, OnnxOpEntryValue]:
    """Convert one ONNX snapshot item dictionary into a dumped ``OnnxOpSpecModel`` dictionary.

    Args:
        item (Mapping[str, JsonValue]): Raw snapshot operator dictionary.

    Returns:
        dict[str, OnnxOpEntryValue]: Serialized operator spec dictionary.
    """
    meta = item.get("domain_metadata")
    domain = "ai.onnx"
    version = 1
    if isinstance(meta, Mapping):
        raw_dom = meta.get("domain")
        if isinstance(raw_dom, str) and raw_dom:
            domain = raw_dom
        raw_ver = meta.get("upstream_version") or item.get("version")
        if isinstance(raw_ver, (int, str)) and str(raw_ver).isdigit():
            version = int(raw_ver)

    operands = item.get("operands")
    inputs = (
        [str(o["name"]) for o in operands if isinstance(o, Mapping) and o.get("name")]
        if isinstance(operands, list)
        else []
    )
    returns = item.get("returns")
    outputs = (
        [str(r["name"]) for r in returns if isinstance(r, Mapping) and r.get("name")]
        if isinstance(returns, list)
        else []
    )
    attrs_out: dict[str, dict[str, JsonValue]] = {}
    raw_attrs = item.get("attributes")
    if isinstance(raw_attrs, Mapping):
        for a_name, a_spec in raw_attrs.items():
            if isinstance(a_spec, Mapping):
                def_raw = a_spec.get("default")
                parsed_def = _parse_proto_default_str(def_raw) if isinstance(def_raw, str) else def_raw
                attrs_out[str(a_name)] = OnnxAttrSpecModel(
                    type=map_onnx_attribute_type(str(a_spec.get("type", "Any"))),
                    required=bool(a_spec.get("required", False)),
                    default=parsed_def,
                ).model_dump()
    return OnnxOpSpecModel(
        domain=domain,
        version=version,
        attributes=attrs_out,
        inputs=inputs,
        outputs=outputs,
    ).model_dump()


def extract_onnx_from_snapshot_json(snapshot_file: Path | str) -> dict[str, dict[str, OnnxOpEntryValue]]:
    """Extract ONNX operator schemas from a structured ``onnx_*.json`` snapshot manifest.

    Args:
        snapshot_file (Path | str): Path to ``onnx_*.json`` in ``../ml-framework-snapshots``.

    Returns:
        dict[str, dict[str, OnnxOpEntryValue]]: Extracted operator schema dictionary.
    """
    path = Path(snapshot_file)
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        return {}

    ops: dict[str, dict[str, OnnxOpEntryValue]] = {}
    for item in _collect_snapshot_items(data):
        name = str(item.get("name") or "")
        if name and name not in ops:
            ops[name] = _snapshot_item_to_onnx_spec(item)
    return ops


def _onnx_schema_to_spec_dict(schema: OnnxSchemaProtocol) -> dict[str, OnnxOpEntryValue]:
    """Serialize an ``OnnxSchemaProtocol`` descriptor into a dumped ``OnnxOpSpecModel`` dictionary.

    Args:
        schema (OnnxSchemaProtocol): Upstream ``onnx.defs`` operator schema object.

    Returns:
        dict[str, OnnxOpEntryValue]: Normalized operator specification dictionary.
    """
    attrs_out: dict[str, dict[str, JsonValue]] = {}
    for attr_name, attr_def in schema.attributes.items():
        raw_type_str = str(attr_def.type).split(".")[-1].lower()
        raw_def = str(attr_def.default_value).strip() if attr_def.default_value.ByteSize() > 0 else None
        attrs_out[str(attr_name)] = OnnxAttrSpecModel(
            type=map_onnx_attribute_type(raw_type_str),
            required=bool(attr_def.required),
            default=_parse_proto_default_str(raw_def),
        ).model_dump()
    domain_str = str(schema.domain) if schema.domain else "ai.onnx"
    return OnnxOpSpecModel(
        domain=domain_str,
        version=int(schema.since_version),
        attributes=attrs_out,
        inputs=[str(inp.name) for inp in schema.inputs],
        outputs=[str(out.name) for out in schema.outputs],
    ).model_dump()


def extract_onnx_from_defs(
    target_versions: Mapping[str, int] | None = None,
) -> dict[str, dict[str, OnnxOpEntryValue]]:
    """Extract canonical ONNX operator schemas via ``onnx.defs.get_all_schemas_with_history()``.

    When ``target_versions`` is provided, resolves the exact historical ``since_version``
    matching ``target_versions[op_name]`` when present in ``onnx.defs`` history so that
    pinned catalog opset versions remain grounded across newer ``onnx`` releases.

    Args:
        target_versions (Mapping[str, int] | None): Optional desired ``since_version`` per operator name.

    Returns:
        dict[str, dict[str, OnnxOpEntryValue]]: Grounded operator dictionary keyed by operator name.
    """
    ops: dict[str, dict[str, OnnxOpEntryValue]] = {}
    try:
        import onnx.defs

        history_schemas: Sequence[OnnxSchemaProtocol] = onnx.defs.get_all_schemas_with_history()
        by_name_ver: dict[tuple[str, int], OnnxSchemaProtocol] = {}
        latest_by_name: dict[str, OnnxSchemaProtocol] = {}
        for schema in history_schemas:
            if schema.domain not in ("", "ai.onnx") or schema.deprecated:
                continue
            by_name_ver[(schema.name, int(schema.since_version))] = schema
            if schema.name not in latest_by_name or schema.since_version > latest_by_name[schema.name].since_version:
                latest_by_name[schema.name] = schema

        current_schemas: Sequence[OnnxSchemaProtocol] = onnx.defs.get_all_schemas()
        for schema in current_schemas:
            if schema.domain not in ("", "ai.onnx") or schema.deprecated:
                continue
            by_name_ver[(schema.name, int(schema.since_version))] = schema
            if schema.name not in latest_by_name or schema.since_version >= latest_by_name[schema.name].since_version:
                latest_by_name[schema.name] = schema

        for op_name in sorted(latest_by_name):
            selected = latest_by_name[op_name]
            if target_versions is not None and op_name in target_versions:
                selected = by_name_ver.get((op_name, target_versions[op_name]), selected)
            ops[op_name] = _onnx_schema_to_spec_dict(selected)
    except ImportError:
        return {}
    return ops


def extract_section(block: str, section_name: str) -> str:
    """Extract content of a markdown subsection bounded by next subsection or block end.

    Args:
        block (str): Markdown block of operator.
        section_name (str): Section header title (e.g. 'Attributes', 'Inputs', 'Outputs').

    Returns:
        str: Section content string.
    """
    pattern = rf"^####\s+{section_name}\s*$(.*?)(?=^####\s+|\Z)"
    m = re.search(pattern, block, flags=re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def cross_validate_with_onnx_defs(
    ops: dict[str, dict[str, OnnxOpEntryValue]],
) -> dict[str, dict[str, OnnxOpEntryValue]]:
    """Cross-validate parsed operator schemas against official onnx.defs if available.

    Args:
        ops (dict[str, dict[str, OnnxOpEntryValue]]): Parsed operator dictionary.

    Returns:
        dict[str, dict[str, OnnxOpEntryValue]]: Cleaned, grounded operator dictionary.
    """
    try:
        import onnx.defs

        onnx_schemas: Sequence[OnnxSchemaProtocol] = onnx.defs.get_all_schemas()
        schema_map: dict[str, OnnxSchemaProtocol] = {}
        for s in onnx_schemas:
            if s.name not in schema_map or s.since_version > schema_map[s.name].since_version:
                schema_map[s.name] = s

        for op_name, op_data in list(ops.items()):
            bare_name = op_name.split(".")[-1]
            if bare_name in schema_map:
                s = schema_map[bare_name]
                canonical_attrs = set(s.attributes.keys())
                raw_attrs = op_data.get("attributes")
                op_data["attributes"] = {
                    k: v for k, v in (raw_attrs.items() if isinstance(raw_attrs, dict) else ()) if k in canonical_attrs
                }
                op_data["inputs"] = [inp.name for inp in s.inputs]
                op_data["outputs"] = [out.name for out in s.outputs]
                if s.domain:
                    op_data["domain"] = s.domain
    except ImportError:
        pass

    return ops


def _parse_onnx_default_val(rest: str) -> JsonValue:
    """Parse an ONNX markdown attribute default value description.

    Args:
        rest (str): Attribute description text after the colon.

    Returns:
        JsonValue: Parsed default value or None.
    """
    default_match = re.search(r"\(default is (.*?)\)", rest)
    if not default_match:
        return None
    default_str = default_match.group(1).strip()
    if default_str == "NOTSET":
        return "NOTSET"
    if default_str in ["[]", "()"]:
        return []
    try:
        parsed: JsonValue = ast.literal_eval(default_str)
        return parsed
    except (ValueError, SyntaxError):
        return default_str.strip("'").strip('"')


def _parse_onnx_attributes_section(
    attr_section: str | None,
) -> dict[str, dict[str, JsonValue]]:
    """Parse the Attributes section of an ONNX operator markdown block.

    Args:
        attr_section (str | None): Extracted attributes HTML/markdown section.

    Returns:
        dict[str, dict[str, JsonValue]]: Mapping of attribute names to serialized specs.
    """
    attributes: dict[str, dict[str, JsonValue]] = {}
    if not attr_section:
        return attributes
    for dt in re.findall(r"<dt>(.*?)</dt>", attr_section, flags=re.DOTALL):
        tt_match = re.search(r"<tt>(.*?)</tt>\s*:\s*(.*?)$", dt)
        if not tt_match:
            continue
        attr_name = tt_match.group(1).strip()
        rest = tt_match.group(2).strip()
        raw_type = re.sub(r"\(.*?\)", "", rest).strip()
        attr_model = OnnxAttrSpecModel(
            type=map_onnx_attribute_type(raw_type),
            required="(required)" in rest,
            default=_parse_onnx_default_val(rest),
        )
        attributes[attr_name] = attr_model.model_dump()
    return attributes


def _parse_onnx_io_section(section_text: str | None) -> list[str]:
    """Parse an Inputs or Outputs section of an ONNX operator markdown block.

    Args:
        section_text (str | None): Extracted inputs or outputs HTML/markdown section.

    Returns:
        list[str]: Ordered list of parameter names.
    """
    if not section_text:
        return []
    names: list[str] = []
    for dt in re.findall(r"<dt>(.*?)</dt>", section_text, flags=re.DOTALL):
        tt_match = re.search(r"<tt>(.*?)</tt>", dt)
        if tt_match:
            names.append(tt_match.group(1).strip())
    return names


def parse_onnx_docs(md_file: str, cross_validate: bool = False) -> dict[str, dict[str, OnnxOpEntryValue]]:
    """Parse ONNX Operators.md markdown documentation to extract op schemas.

    Args:
        md_file (str): Path to the Operators.md file.
        cross_validate (bool): Whether to cross-validate against canonical onnx.defs.

    Returns:
        dict[str, dict[str, OnnxOpEntryValue]]: Dictionary mapping operator names to operator schema definitions.
    """
    with open(md_file, encoding="utf-8") as f:
        content = f.read()

    op_blocks = re.split(
        r'^###\s+(?:<sub>.*?</sub>\s+)?<a\s+name="[^"]+"></a>',
        content,
        flags=re.MULTILINE,
    )

    ops: dict[str, dict[str, OnnxOpEntryValue]] = {}

    for block in op_blocks[1:]:
        name_match = re.search(r'^<a\s+name="[^"]+">\*\*(.*?)\*\*</a>', block) or re.search(r"^\*\*(.*?)\*\*", block)
        if not name_match:
            continue

        op_name = name_match.group(1).strip()
        ver_match = re.search(r"has been available since version (\d+)", block)
        version = int(ver_match.group(1)) if ver_match else 1
        domain = op_name.rsplit(".", 1)[0] if op_name.startswith("ai.onnx.") else "ai.onnx"

        op_model = OnnxOpSpecModel(
            domain=domain,
            version=version,
            attributes=_parse_onnx_attributes_section(extract_section(block, "Attributes")),
            inputs=_parse_onnx_io_section(extract_section(block, "Inputs")),
            outputs=_parse_onnx_io_section(extract_section(block, "Outputs")),
        )
        ops[op_name] = op_model.model_dump()

    if cross_validate:
        return cross_validate_with_onnx_defs(ops)
    return ops


def extract_grounded_onnx_catalog(
    snapshots_dir: str | Path | None = None,
) -> dict[str, dict[str, OnnxOpEntryValue]]:
    """Build grounded ONNX catalog preferring structured ``onnx_*.json`` snapshots and ``onnx.defs``.

    Args:
        snapshots_dir (str | Path | None): Optional directory containing ``onnx_*.json`` manifests.

    Returns:
        dict[str, dict[str, OnnxOpEntryValue]]: Grounded operator dictionary.
    """
    resolved_dir_str = str(snapshots_dir) if snapshots_dir is not None else get_default_snapshots_dir()
    merged: dict[str, dict[str, OnnxOpEntryValue]] = extract_onnx_from_defs()

    if resolved_dir_str:
        snap_root = Path(resolved_dir_str)
        if snap_root.is_dir():
            for snap_file in sorted(snap_root.glob("onnx*.json")):
                if snap_file.name == "onnx_ops.json":
                    continue
                snap_ops = extract_onnx_from_snapshot_json(snap_file)
                for op_name, op_spec in snap_ops.items():
                    if op_name not in merged:
                        merged[op_name] = op_spec
    return merged


def render_registry_module() -> str:
    """Render and AST-validate the canonical ``onnx_registry.py`` module source.

    Returns:
        str: Syntactically validated Python module source string.
    """
    canonical_src = (Path(__file__).resolve().parent.parent / "src/ml_switcheroo_ir/schema/onnx_registry.py").read_text(
        encoding="utf-8"
    )
    parsed_ast = ast.parse(canonical_src)
    _ = ast.unparse(parsed_ast)
    return canonical_src


def _resolve_cli_paths(
    md_file: str | None,
    json_path: str | None,
    registry_path: str | None,
    onnx_dir: str | None,
) -> tuple[str | None, str, str]:
    """Resolve CLI input markdown path, output JSON path, and output registry module path.

    Args:
        md_file (str | None): Explicit Operators.md path if supplied.
        json_path (str | None): Explicit output JSON path if supplied.
        registry_path (str | None): Explicit output Python module path if supplied.
        onnx_dir (str | None): Optional ONNX repository directory.

    Returns:
        tuple[str | None, str, str]: Resolved ``(target_md, target_json, target_reg)``.
    """
    resolved_onnx_dir = onnx_dir
    args_list = sys.argv[1:]
    for idx, arg in enumerate(args_list):
        if arg == "--onnx-dir" and idx + 1 < len(args_list):
            resolved_onnx_dir = args_list[idx + 1]

    target_md: str | None
    if md_file is not None:
        target_md = md_file
    elif resolved_onnx_dir is not None:
        cand1 = os.path.join(resolved_onnx_dir, "docs", "Operators.md")
        cand2 = os.path.join(resolved_onnx_dir, "Operators.md")
        target_md = cand1 if os.path.exists(cand1) else cand2
    elif len(args_list) > 0 and not args_list[0].startswith("-"):
        target_md = args_list[0]
    else:
        target_md = None

    target_json = (
        json_path
        if json_path is not None
        else (
            args_list[1]
            if len(args_list) > 1 and not args_list[1].startswith("-")
            else "src/ml_switcheroo_ir/schema/catalogs/onnx_ops.json"
        )
    )
    target_reg = (
        registry_path
        if registry_path is not None
        else (
            args_list[2]
            if len(args_list) > 2 and not args_list[2].startswith("-")
            else "src/ml_switcheroo_ir/schema/onnx_registry.py"
        )
    )
    return target_md, target_json, target_reg


def main(
    md_file: str | None = None,
    json_path: str | None = None,
    registry_path: str | None = None,
    onnx_dir: str | None = None,
) -> None:
    """Generate ONNX JSON schema catalog and registry module from snapshots, onnx.defs, or Operators.md.

    Args:
        md_file (str | None): Optional path to ONNX Operators markdown documentation.
        json_path (str | None): Destination path for generated JSON op schemas.
        registry_path (str | None): Destination path for generated Python registry file.
        onnx_dir (str | None): Optional path to an external ONNX repository directory.
    """
    target_md, target_json, target_reg = _resolve_cli_paths(md_file, json_path, registry_path, onnx_dir)

    if target_md is not None:
        ops = parse_onnx_docs(target_md, cross_validate=True)
    else:
        ops = extract_grounded_onnx_catalog()
        if not ops and os.path.exists("docs/Operators.md"):
            ops = parse_onnx_docs("docs/Operators.md", cross_validate=True)

    Path(target_json).parent.mkdir(parents=True, exist_ok=True)
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(ops, f, indent=2)

    print(f"Parsed {len(ops)} operators.")

    loader_code = render_registry_module()
    Path(target_reg).parent.mkdir(parents=True, exist_ok=True)
    with open(target_reg, "w", encoding="utf-8") as f:
        f.write(loader_code)

    print(f"Generated {target_reg}")


if __name__ == "__main__":
    main()
