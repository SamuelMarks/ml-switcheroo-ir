"""Core operator schema dataclasses and unified JSON catalog loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ml_switcheroo_ir.snapshots import find_schema_file


@dataclass
class OpAttribute:
    """Represents a single operator attribute schema.

    Attributes:
        name (str): Name of the attribute.
        type (str): Type descriptor string of the attribute.
        required (bool): Whether the attribute must be provided.
        default (Any): Default value if optional.
    """

    name: str
    type: str
    required: bool
    default: Any


@dataclass
class OpSchema:
    """Represents a single operator schema.

    Attributes:
        name (str): Operator identifier name.
        domain (str): Domain namespace of the operator.
        version (int): Operator schema version integer.
        attributes (dict[str, OpAttribute]): Mapping of attribute names to OpAttribute instances.
        inputs (list[str]): List of formal input operand names.
        outputs (list[str]): List of formal output operand names.
        schema_version (int | None): Optional upstream opset/schema version.
        deprecated (bool): Whether this operator version is deprecated.
    """

    name: str
    domain: str
    version: int
    attributes: dict[str, OpAttribute]
    inputs: list[str]
    outputs: list[str]
    schema_version: int | None = None
    deprecated: bool = False


@dataclass
class CustomAttributeSchema:
    """Represents an attribute for a custom operator.

    Attributes:
        name (str): The name of the attribute.
        type (str): The expected Python type (e.g. 'int', 'List[float]').
        required (bool): Whether the attribute is required.
        default (Any): The default value if not provided.
    """

    name: str = ""
    type: str = "Any"
    required: bool = False
    default: Any = None

    def to_op_attribute(self) -> OpAttribute:
        """Convert to the internal OpAttribute type.

        Returns:
            OpAttribute: Converted operator attribute schema.
        """
        return OpAttribute(
            name=self.name,
            type=self.type,
            required=self.required,
            default=self.default,
        )


@dataclass
class CustomOpSchema:
    """Represents a custom operator schema.

    Attributes:
        name (str): The name of the operator.
        domain (str): The custom domain (e.g., 'ai.custom').
        attributes (Union[List[CustomAttributeSchema], Dict[str, CustomAttributeSchema]]): The list or dictionary of attributes.
        inputs (List[str]): List of expected inputs.
        outputs (List[str]): List of expected outputs.
        version (int): Operator version (default: 1).
        schema_version (Optional[int]): Optional upstream opset/schema version.
        deprecated (bool): Whether this operator schema is deprecated.
    """

    name: str
    domain: str
    attributes: (
        list[CustomAttributeSchema] | dict[str, CustomAttributeSchema | OpAttribute]
    ) = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    version: int = 1
    schema_version: int | None = None
    deprecated: bool = False

    def to_op_schema(self) -> OpSchema:
        """Convert to the internal OpSchema type.

        Note:
            Version is set to 1 by default for custom ops.

        Returns:
            OpSchema: Converted operator schema.
        """
        if isinstance(self.attributes, dict):
            attr_dict: dict[str, OpAttribute] = {}
            for k, attr in self.attributes.items():
                if isinstance(attr, CustomAttributeSchema):
                    attr_name = attr.name if attr.name else k
                    attr_dict[attr_name] = OpAttribute(
                        name=attr_name,
                        type=attr.type,
                        required=attr.required,
                        default=attr.default,
                    )
                else:
                    attr_dict[k] = attr
        else:
            attr_dict = {attr.name: attr.to_op_attribute() for attr in self.attributes}
        return OpSchema(
            name=self.name,
            domain=self.domain,
            version=self.version,
            attributes=attr_dict,
            inputs=self.inputs,
            outputs=self.outputs,
            schema_version=self.schema_version,
            deprecated=self.deprecated,
        )


def parse_op_schema_entry(
    op_data: Mapping[str, Any],
    domain_default: str,
    attr_type_default: str = "str",
) -> OpSchema:
    """Parse a single JSON operator schema dictionary entry into an OpSchema.

    Args:
        op_data (Mapping[str, Any]): Raw JSON operator definition dictionary.
        domain_default (str): Fallback domain identifier if not present in op_data.
        attr_type_default (str): Fallback attribute type string if not present.

    Returns:
        OpSchema: Parsed operator schema.
    """
    attrs = [
        CustomAttributeSchema(
            name=str(a.get("name", "")),
            type=str(a.get("type", attr_type_default)),
            required=bool(a.get("required", False)),
            default=a.get("default", None),
        )
        for a in op_data.get("attributes", [])
        if isinstance(a, Mapping)
    ]
    raw_sv = op_data.get("schema_version")
    schema = CustomOpSchema(
        name=str(op_data["name"]),
        domain=str(op_data.get("domain", domain_default)),
        attributes=attrs,
        inputs=list(op_data.get("inputs", [])),
        outputs=list(op_data.get("outputs", [])),
        version=int(op_data.get("version", 1)),
        schema_version=int(raw_sv) if isinstance(raw_sv, int) else None,
        deprecated=bool(op_data.get("deprecated", False)),
    )
    return schema.to_op_schema()


def load_op_schema_catalog(
    filename: str,
    domain_default: str,
    json_path: Path | str | None = None,
    find_file_fn: Callable[[str, Path | str | None], Path | None] = find_schema_file,
    attr_type_default: str = "str",
    on_entry: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, OpSchema]:
    """Load operator schemas from a JSON catalog file containing an 'ops' list.

    Args:
        filename (str): Name of the JSON catalog file in the schema/snapshot directory.
        domain_default (str): Default domain identifier for loaded operations.
        json_path (Optional[Union[Path, str]]): Explicit path override for the JSON catalog.
        find_file_fn (Callable[[str, Optional[Union[Path, str]]], Optional[Path]]): File resolution function.
        attr_type_default (str): Default attribute type string when unspecified.
        on_entry (Optional[Callable[[Mapping[str, Any]], None]]): Optional callback invoked per raw op_data entry.

    Returns:
        dict[str, OpSchema]: Mapping from operator names to parsed OpSchema definitions.
    """
    registry: dict[str, OpSchema] = {}
    target = find_file_fn(filename, json_path)
    if target is not None and target.exists():
        with open(target, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        for op_data in data.get("ops", []):
            schema = parse_op_schema_entry(
                op_data,
                domain_default=domain_default,
                attr_type_default=attr_type_default,
            )
            registry[schema.name] = schema
            if on_entry is not None:
                on_entry(op_data)
    return registry
