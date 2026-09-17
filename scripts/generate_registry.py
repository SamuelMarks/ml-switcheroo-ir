"""Parser and code generator for ONNX operator schema registry."""

from __future__ import annotations

import ast
import json
import re
import sys
from typing import Any


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
    ops: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Cross-validate parsed operator schemas against official onnx.defs if available.

    Eliminates hallucinated attributes, corrects inputs on zero-input operators, and
    harmonizes domains.

    Args:
        ops (dict[str, dict[str, Any]]): Parsed operator dictionary.

    Returns:
        dict[str, dict[str, Any]]: Cleaned, grounded operator dictionary.
    """
    try:
        import onnx.defs

        onnx_schemas = onnx.defs.get_all_schemas()
        schema_map: dict[str, Any] = {}
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
                # Filter out hallucinated attributes not present in official defs
                op_data["attributes"] = {
                    k: v
                    for k, v in op_data["attributes"].items()
                    if k in canonical_attrs
                }
                # Fix inputs and outputs to match canonical defs
                op_data["inputs"] = [inp.name for inp in s.inputs]
                op_data["outputs"] = [out.name for out in s.outputs]
                if s.domain:
                    op_data["domain"] = s.domain
    except ImportError:
        pass

    return ops


def parse_onnx_docs(
    md_file: str, cross_validate: bool = False
) -> dict[str, dict[str, Any]]:
    """Parse ONNX Operators.md markdown documentation to extract op schemas.

    Args:
        md_file: Path to the Operators.md file.
        cross_validate: Whether to cross-validate against canonical onnx.defs.

    Returns:
        Dictionary mapping operator names to operator schema definitions.
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

    ops: dict[str, dict[str, Any]] = {}

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
        attributes: dict[str, dict[str, Any]] = {}
        if attr_section:
            dts = re.findall(r"<dt>(.*?)</dt>", attr_section, flags=re.DOTALL)
            for dt in dts:
                tt_match = re.search(r"<tt>(.*?)</tt>\s*:\s*(.*?)$", dt)
                if not tt_match:
                    continue
                attr_name = tt_match.group(1).strip()
                rest = tt_match.group(2).strip()

                required = "(required)" in rest

                default_val: str | int | float | list[Any] | dict[str, Any] | None = (
                    None
                )
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

                attributes[attr_name] = {
                    "type": py_type,
                    "required": required,
                    "default": default_val,
                }

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

        ops[op_name] = {
            "domain": domain,
            "version": version,
            "attributes": attributes,
            "inputs": inputs,
            "outputs": outputs,
        }

    if cross_validate:
        return cross_validate_with_onnx_defs(ops)
    return ops


def main(
    md_file: str | None = None,
    json_path: str | None = None,
    registry_path: str | None = None,
) -> None:
    """Generate ONNX JSON schema and registry python module from documentation.

    Args:
        md_file: Path to ONNX Operators markdown documentation.
        json_path: Destination path for generated JSON op schemas.
        registry_path: Destination path for generated Python registry file.
    """
    target_md = (
        md_file
        if md_file is not None
        else (
            sys.argv[1] if len(sys.argv) > 1 else "third_party/onnx/docs/Operators.md"
        )
    )
    target_json = (
        json_path
        if json_path is not None
        else (
            sys.argv[2]
            if len(sys.argv) > 2
            else "src/ml_switcheroo_ir/schema/onnx_ops.json"
        )
    )
    target_reg = (
        registry_path
        if registry_path is not None
        else (
            sys.argv[3]
            if len(sys.argv) > 3
            else "src/ml_switcheroo_ir/schema/onnx_registry.py"
        )
    )

    ops = parse_onnx_docs(target_md, cross_validate=True)

    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(ops, f, indent=2)

    print(f"Parsed {len(ops)} operators.")

    # Generate python registry file
    lines = [
        '"""Generated ONNX Operator Registry."""',
        "from __future__ import annotations",
        "from dataclasses import dataclass, field",
        "from typing import Any",
        "",
        "@dataclass",
        "class OpAttribute:",
        '    """Represents a single operator attribute schema."""',
        "    name: str",
        "    type: str",
        "    required: bool",
        "    default: Any",
        "",
        "@dataclass",
        "class OpSchema:",
        '    """Represents a single operator schema."""',
        "    name: str",
        "    domain: str",
        "    version: int",
        "    attributes: dict[str, OpAttribute]",
        "    inputs: list[str]",
        "    outputs: list[str]",
        "",
        "ONNX_REGISTRY: dict[str, OpSchema] = {",
    ]

    for op_name, op_data in sorted(ops.items()):
        lines.append(f'    "{op_name}": OpSchema(')
        lines.append(f'        name="{op_name}",')
        lines.append(f'        domain="{op_data["domain"]}",')
        lines.append(f"        version={op_data['version']},")
        lines.append("        attributes={")
        for attr_name, attr_data in sorted(op_data["attributes"].items()):
            req = str(attr_data["required"])
            default_val = repr(attr_data["default"])
            lines.append(
                f'            "{attr_name}": OpAttribute(name="{attr_name}", type="{attr_data["type"]}", required={req}, default={default_val}),'
            )
        lines.append("        },")
        inputs_repr = repr(op_data["inputs"])
        outputs_repr = repr(op_data["outputs"])
        lines.append(f"        inputs={inputs_repr},")
        lines.append(f"        outputs={outputs_repr},")
        lines.append("    ),")

    lines.append("}")
    lines.append("")

    with open(target_reg, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {target_reg}")


if __name__ == "__main__":
    main()
