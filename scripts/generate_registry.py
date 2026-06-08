import re
import json
import ast


def parse_onnx_docs(md_file):
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split the file by operators. The header for an operator is usually like:
    # ### <a name="Name"></a><a name="name">**Name**</a>
    # or just look for "### <a name="

    op_blocks = re.split(r'^###\s+<a\s+name="[^"]+"></a>', content, flags=re.MULTILINE)

    ops = {}

    for block in op_blocks[1:]:  # skip first block which is preamble
        name_match = re.search(r'^<a\s+name="[^"]+">\*\*(.*?)\*\*</a>', block)
        if not name_match:
            # Maybe it's formatted differently
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
        # find the Attributes section
        attr_section_match = re.search(
            r"#### Attributes\n+<dl>(.*?)</dl>", block, flags=re.DOTALL
        )

        attributes = {}
        if attr_section_match:
            attr_content = attr_section_match.group(1)
            # Find all <dt>
            dts = re.findall(r"<dt>(.*?)</dt>", attr_content, flags=re.DOTALL)
            for dt in dts:
                # e.g., <tt>auto_pad</tt> : string (default is NOTSET)
                # or <tt>kernel_shape</tt> : list of ints (required)
                tt_match = re.search(r"<tt>(.*?)</tt>\s*:\s*(.*?)$", dt)
                if not tt_match:
                    continue
                attr_name = tt_match.group(1).strip()
                rest = tt_match.group(2).strip()

                required = "(required)" in rest

                # Default
                default_val = None
                default_match = re.search(r"\(default is (.*?)\)", rest)
                if default_match:
                    default_str = default_match.group(1).strip()
                    # Try to parse the default string safely, might be 'NOTSET', numbers, lists
                    if default_str == "NOTSET":
                        default_val = "NOTSET"
                    elif default_str in ["[]", "()"]:
                        default_val = []
                    else:
                        try:
                            # It could be a number
                            default_val = ast.literal_eval(default_str)
                        except (ValueError, SyntaxError):
                            # It's just a string, e.g., 'nearest'
                            default_val = default_str.strip("'\"")

                # Type
                # strip out the paren parts
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
                elif raw_type == "tensor":
                    py_type = "Any"
                elif raw_type == "graph":
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
        input_section_match = re.search(
            r"#### Inputs(?:.*?)\n+<dl>(.*?)</dl>", block, flags=re.DOTALL
        )
        inputs = []
        if input_section_match:
            input_content = input_section_match.group(1)
            # Find all <dt>
            dts = re.findall(r"<dt>(.*?)</dt>", input_content, flags=re.DOTALL)
            for dt in dts:
                tt_match = re.search(r"<tt>(.*?)</tt>", dt)
                if tt_match:
                    inputs.append(tt_match.group(1).strip())

        # Outputs
        output_section_match = re.search(
            r"#### Outputs(?:.*?)\n+<dl>(.*?)</dl>", block, flags=re.DOTALL
        )
        outputs = []
        if output_section_match:
            output_content = output_section_match.group(1)
            # Find all <dt>
            dts = re.findall(r"<dt>(.*?)</dt>", output_content, flags=re.DOTALL)
            for dt in dts:
                tt_match = re.search(r"<tt>(.*?)</tt>", dt)
                if tt_match:
                    outputs.append(tt_match.group(1).strip())

        ops[op_name] = {
            "domain": "ai.onnx",
            "version": version,
            "attributes": attributes,
            "inputs": inputs,
            "outputs": outputs,
        }

    return ops


def main():
    md_file = "third_party/onnx/docs/Operators.md"
    ops = parse_onnx_docs(md_file)

    json_path = "src/ml_switcheroo_ir/schema/onnx_ops.json"
    with open(json_path, "w") as f:
        json.dump(ops, f, indent=2)

    print(f"Parsed {len(ops)} operators.")

    # Generate python registry file
    registry_path = "src/ml_switcheroo_ir/schema/onnx_registry.py"

    lines = [
        '"""Generated ONNX Operator Registry."""',
        "from dataclasses import dataclass, field",
        "from typing import Dict, List, Optional, Any",
        "",
        "@dataclass",
        "class OpAttribute:",
        "    name: str",
        "    type: str",
        "    required: bool",
        "    default: Any",
        "",
        "@dataclass",
        "class OpSchema:",
        "    name: str",
        "    domain: str",
        "    version: int",
        "    attributes: Dict[str, OpAttribute]",
        "    inputs: List[str]",
        "    outputs: List[str]",
        "",
        "ONNX_REGISTRY: Dict[str, OpSchema] = {",
    ]

    for op_name, op_data in sorted(ops.items()):
        lines.append(f'    "{op_name}": OpSchema(')
        lines.append(f'        name="{op_name}",')
        lines.append(f'        domain="{op_data["domain"]}",')
        lines.append(f'        version={op_data["version"]},')
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

    with open(registry_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated {registry_path}")


if __name__ == "__main__":
    main()
