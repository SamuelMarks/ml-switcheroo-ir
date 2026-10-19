"""Parser and code generator for ONNX operator schema registry."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import json
import os
import re
import sys
import typing

from pydantic import BaseModel, ConfigDict, Field

from ml_switcheroo_ir.types import JsonValue


class OnnxAttrSpecModel(BaseModel):
    """Pydantic model for a parsed ONNX operator attribute specification."""

    model_config = ConfigDict(extra="forbid")

    type: str = "Any"
    required: bool = False
    default: JsonValue = None


class OnnxOpSpecModel(BaseModel):
    """Pydantic model for a parsed ONNX operator schema specification."""

    model_config = ConfigDict(extra="forbid")

    domain: str = "ai.onnx"
    version: int = 1
    attributes: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


OnnxAttrSpecModel.model_rebuild()
OnnxOpSpecModel.model_rebuild()


class OnnxParamProtocol(typing.Protocol):
    """Protocol for an upstream onnx.defs formal parameter."""

    name: str


class OnnxSchemaProtocol(typing.Protocol):
    """Protocol for an upstream onnx.defs OpSchema."""

    name: str
    since_version: int
    domain: str
    attributes: Mapping[str, object]
    inputs: Sequence[OnnxParamProtocol]
    outputs: Sequence[OnnxParamProtocol]


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
    ops: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Cross-validate parsed operator schemas against official onnx.defs if available.

    Eliminates hallucinated attributes, corrects inputs on zero-input operators, and
    harmonizes domains.

    Args:
        ops (dict[str, dict[str, object]]): Parsed operator dictionary.

    Returns:
        dict[str, dict[str, object]]: Cleaned, grounded operator dictionary.
    """
    try:
        import onnx.defs

        onnx_schemas: Sequence[OnnxSchemaProtocol] = onnx.defs.get_all_schemas()
        schema_map: dict[str, OnnxSchemaProtocol] = {}
        for s in onnx_schemas:
            if (
                s.name not in schema_map
                or s.since_version > schema_map[s.name].since_version
            ):
                schema_map[s.name] = s

        for op_name, op_data in list(ops.items()):
            bare_name = op_name.split(".")[-1]
            if bare_name in schema_map:
                s = schema_map[bare_name]
                canonical_attrs = set(s.attributes.keys())
                raw_attrs = op_data.get("attributes")
                op_data["attributes"] = {
                    k: v
                    for k, v in (
                        raw_attrs.items() if isinstance(raw_attrs, dict) else ()
                    )
                    if k in canonical_attrs
                }
                op_data["inputs"] = [inp.name for inp in s.inputs]
                op_data["outputs"] = [out.name for out in s.outputs]
                if s.domain:
                    op_data["domain"] = s.domain
    except ImportError:
        pass

    return ops


def parse_onnx_docs(
    md_file: str, cross_validate: bool = False
) -> dict[str, dict[str, object]]:
    """Parse ONNX Operators.md markdown documentation to extract op schemas.

    Args:
        md_file (str): Path to the Operators.md file.
        cross_validate (bool): Whether to cross-validate against canonical onnx.defs.

    Returns:
        dict[str, dict[str, object]]: Dictionary mapping operator names to operator schema definitions.
    """
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split the file by operators. The header for an operator is usually like:
    # ### <a name="Name"></a><a name="name">**Name**</a>
    # or with sub experimental tag: ### <sub>experimental</sub> <a name="...">
    op_blocks = re.split(
        r'^###\s+(?:<sub>.*?</sub>\s+)?<a\s+name="[^"]+"></a>',
        content,
        flags=re.MULTILINE,
    )

    ops: dict[str, dict[str, object]] = {}

    for block in op_blocks[1:]:  # skip first block which is preamble
        name_match = re.search(r'^<a\s+name="[^"]+">\*\*(.*?)\*\*</a>', block)
        if not name_match:
            name_match = re.search(r"^\*\*(.*?)\*\*", block)
            if not name_match:
                continue

        op_name = name_match.group(1).strip()

        # Version
        version = 1
        ver_match = re.search(r"has been available since version (\d+)", block)
        if ver_match:
            version = int(ver_match.group(1))

        # Attributes
        attr_section = extract_section(block, "Attributes")
        attributes: dict[str, dict[str, JsonValue]] = {}
        if attr_section:
            dts = re.findall(r"<dt>(.*?)</dt>", attr_section, flags=re.DOTALL)
            for dt in dts:
                tt_match = re.search(r"<tt>(.*?)</tt>\s*:\s*(.*?)$", dt)
                if not tt_match:
                    continue
                attr_name = tt_match.group(1).strip()
                rest = tt_match.group(2).strip()

                required = "(required)" in rest

                default_val: JsonValue = None
                default_match = re.search(r"\(default is (.*?)\)", rest)
                if default_match:
                    default_str = default_match.group(1).strip()
                    if default_str == "NOTSET":
                        default_val = "NOTSET"
                    elif default_str in ["[]", "()"]:
                        default_val = []
                    else:
                        try:
                            default_val = ast.literal_eval(default_str)
                        except (ValueError, SyntaxError):
                            default_val = default_str.strip("'\"")

                raw_type = re.sub(r"\(.*?\)", "", rest).strip()

                py_type = "Any"
                if raw_type == "int":
                    py_type = "int"
                elif raw_type == "float":
                    py_type = "float"
                elif raw_type == "string":
                    py_type = "str"
                elif raw_type == "list of ints":
                    py_type = "List[int]"
                elif raw_type == "list of floats":
                    py_type = "List[float]"
                elif raw_type == "list of strings":
                    py_type = "List[str]"
                elif raw_type in ("tensor", "graph"):
                    py_type = "Any"
                elif raw_type == "type":
                    py_type = "str"
                else:
                    py_type = "Any"

                attr_model = OnnxAttrSpecModel(
                    type=py_type,
                    required=required,
                    default=default_val,
                )
                attributes[attr_name] = attr_model.model_dump()

        # Inputs
        input_section = extract_section(block, "Inputs")
        inputs: list[str] = []
        if input_section:
            dts = re.findall(r"<dt>(.*?)</dt>", input_section, flags=re.DOTALL)
            for dt in dts:
                tt_match = re.search(r"<tt>(.*?)</tt>", dt)
                if tt_match:
                    inputs.append(tt_match.group(1).strip())

        # Outputs
        output_section = extract_section(block, "Outputs")
        outputs: list[str] = []
        if output_section:
            dts = re.findall(r"<dt>(.*?)</dt>", output_section, flags=re.DOTALL)
            for dt in dts:
                tt_match = re.search(r"<tt>(.*?)</tt>", dt)
                if tt_match:
                    outputs.append(tt_match.group(1).strip())

        domain = "ai.onnx"
        if op_name.startswith("ai.onnx."):
            domain = op_name.rsplit(".", 1)[0]

        op_model = OnnxOpSpecModel(
            domain=domain,
            version=version,
            attributes=attributes,
            inputs=inputs,
            outputs=outputs,
        )
        ops[op_name] = op_model.model_dump()

    if cross_validate:
        return cross_validate_with_onnx_defs(ops)
    return ops


_ONNX_REGISTRY_TEMPLATE = '''"""Generated ONNX Operator Registry."""

from __future__ import annotations

import json
from pathlib import Path
import threading

from ml_switcheroo_ir.schema.base import OpAttribute, OpSchema

_LOCK = threading.Lock()
ONNX_REGISTRY: dict[str, OpSchema] = {}


def load_onnx_schemas(json_path: Path | str | None = None) -> dict[str, OpSchema]:
    """Dynamically load ONNX schemas from onnx_ops.json into ONNX_REGISTRY.

    Args:
        json_path: Optional custom path to onnx_ops.json.

    Returns:
        Mapping of operator names to OpSchema objects.
    """
    with _LOCK:
        if ONNX_REGISTRY and json_path is None:
            return ONNX_REGISTRY

        target = (
            Path(json_path)
            if json_path is not None
            else Path(__file__).parent / "onnx_ops.json"
        )
        if not target.exists():
            return dict(ONNX_REGISTRY) if json_path is not None else ONNX_REGISTRY

        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded: dict[str, OpSchema] = {}
        for op_name, op_data in data.items():
            attributes = {
                attr_name: OpAttribute(
                    name=attr_name,
                    type=attr_data.get("type", "Any"),
                    required=attr_data.get("required", False),
                    default=attr_data.get("default", None),
                )
                for attr_name, attr_data in op_data.get("attributes", {}).items()
            }
            loaded[op_name] = OpSchema(
                name=op_name,
                domain=op_data.get("domain", "ai.onnx"),
                version=op_data.get("version", 1),
                attributes=attributes,
                inputs=op_data.get("inputs", []),
                outputs=op_data.get("outputs", []),
            )
        if json_path is None:
            ONNX_REGISTRY.clear()
            ONNX_REGISTRY.update(loaded)
            return ONNX_REGISTRY
        return loaded


load_onnx_schemas()
'''


def render_registry_module() -> str:
    """Render and AST-validate the generated onnx_registry.py source code.

    Returns:
        str: Syntactically validated Python module source string.
    """
    parsed_ast = ast.parse(_ONNX_REGISTRY_TEMPLATE)
    _ = ast.unparse(parsed_ast)
    return _ONNX_REGISTRY_TEMPLATE


def main(
    md_file: str | None = None,
    json_path: str | None = None,
    registry_path: str | None = None,
    onnx_dir: str | None = None,
) -> None:
    """Generate ONNX JSON schema and registry python module from documentation.

    Args:
        md_file: Path to ONNX Operators markdown documentation.
        json_path: Destination path for generated JSON op schemas.
        registry_path: Destination path for generated Python registry file.
        onnx_dir: Optional path to an external ONNX repository directory.
    """
    resolved_onnx_dir = onnx_dir
    args_list = sys.argv[1:]
    for idx, arg in enumerate(args_list):
        if arg == "--onnx-dir" and idx + 1 < len(args_list):
            resolved_onnx_dir = args_list[idx + 1]

    if md_file is not None:
        target_md = md_file
    elif resolved_onnx_dir is not None:
        cand1 = os.path.join(resolved_onnx_dir, "docs", "Operators.md")
        cand2 = os.path.join(resolved_onnx_dir, "Operators.md")
        target_md = cand1 if os.path.exists(cand1) else cand2
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        target_md = sys.argv[1]
    else:
        target_md = "docs/Operators.md"

    target_json = (
        json_path
        if json_path is not None
        else (
            sys.argv[2]
            if len(sys.argv) > 2 and not sys.argv[2].startswith("--")
            else "src/ml_switcheroo_ir/schema/onnx_ops.json"
        )
    )
    target_reg = (
        registry_path
        if registry_path is not None
        else (
            sys.argv[3]
            if len(sys.argv) > 3 and not sys.argv[3].startswith("--")
            else "src/ml_switcheroo_ir/schema/onnx_registry.py"
        )
    )

    ops = parse_onnx_docs(target_md, cross_validate=True)

    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(ops, f, indent=2)

    print(f"Parsed {len(ops)} operators.")

    loader_code = render_registry_module()

    with open(target_reg, "w", encoding="utf-8") as f:
        f.write(loader_code)

    print(f"Generated {target_reg}")


if __name__ == "__main__":
    main()
