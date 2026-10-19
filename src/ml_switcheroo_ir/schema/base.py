"""Core operator schema Pydantic models and unified JSON catalog loader."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import json
from pathlib import Path
import threading
import typing

from pydantic import BaseModel, ConfigDict, Field

from ml_switcheroo_ir.snapshots import find_schema_file
from ml_switcheroo_ir.types import AttributeValue

REGISTRY_LOCK = threading.RLock()


class RawOpAttributeDict(typing.TypedDict, total=False):
    """TypedDict for raw JSON attribute entry in an operator catalog."""

    name: str
    type: str
    required: bool
    default: AttributeValue


class RawOpSchemaDict(typing.TypedDict, total=False):
    """TypedDict for raw JSON operator entry in an operator catalog."""

    name: str
    domain: str
    version: int
    schema_version: int | None
    deprecated: bool
    attributes: Sequence[RawOpAttributeDict] | Mapping[str, RawOpAttributeDict]
    inputs: Sequence[str]
    outputs: Sequence[str]
    latency: int
    vopd_slot: str
    vopd_equiv: str


class OpAttribute(BaseModel):
    """Represents a single operator attribute schema.

    Attributes:
        name (str): Name of the attribute.
        type (str): Type descriptor string of the attribute.
        required (bool): Whether the attribute must be provided.
        default (AttributeValue): Default value if optional.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    type: str
    required: bool
    default: AttributeValue = None

    def __init__(
        self,
        name: str,
        type: str,
        required: bool,
        default: AttributeValue = None,
    ) -> None:
        """Initialize OpAttribute.

        Args:
            name (str): Name of the attribute.
            type (str): Type descriptor string of the attribute.
            required (bool): Whether the attribute must be provided.
            default (AttributeValue): Default value if optional.
        """
        super().__init__(
            name=name,
            type=type,
            required=required,
            default=default,
        )


class OpSchema(BaseModel):
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

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    domain: str
    version: int
    attributes: dict[str, OpAttribute]
    inputs: list[str]
    outputs: list[str]
    schema_version: int | None = None
    deprecated: bool = False

    def __init__(
        self,
        name: str,
        domain: str,
        version: int,
        attributes: Mapping[str, OpAttribute],
        inputs: Sequence[str],
        outputs: Sequence[str],
        schema_version: int | None = None,
        deprecated: bool = False,
    ) -> None:
        """Initialize OpSchema.

        Args:
            name (str): Operator identifier name.
            domain (str): Domain namespace of the operator.
            version (int): Operator schema version integer.
            attributes (Mapping[str, OpAttribute]): Mapping of attribute names to OpAttribute instances.
            inputs (Sequence[str]): List of formal input operand names.
            outputs (Sequence[str]): List of formal output operand names.
            schema_version (int | None): Optional upstream opset/schema version.
            deprecated (bool): Whether this operator version is deprecated.
        """
        super().__init__(
            name=name,
            domain=domain,
            version=version,
            attributes=dict(attributes),
            inputs=list(inputs),
            outputs=list(outputs),
            schema_version=schema_version,
            deprecated=deprecated,
        )


class CustomAttributeSchema(BaseModel):
    """Represents an attribute for a custom operator.

    Attributes:
        name (str): The name of the attribute.
        type (str): The expected Python type (e.g. 'int', 'List[float]').
        required (bool): Whether the attribute is required.
        default (AttributeValue): The default value if not provided.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str = ""
    type: str = "Any"
    required: bool = False
    default: AttributeValue = None

    def __init__(
        self,
        name: str = "",
        type: str = "Any",
        required: bool = False,
        default: AttributeValue = None,
    ) -> None:
        """Initialize CustomAttributeSchema.

        Args:
            name (str): The name of the attribute.
            type (str): The expected Python type.
            required (bool): Whether the attribute is required.
            default (AttributeValue): The default value if not provided.
        """
        super().__init__(
            name=name,
            type=type,
            required=required,
            default=default,
        )

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


class CustomOpSchema(BaseModel):
    """Represents a custom operator schema.

    Attributes:
        name (str): The name of the operator.
        domain (str): The custom domain (e.g., 'ai.custom').
        attributes (list[CustomAttributeSchema] | dict[str, CustomAttributeSchema | OpAttribute]):
            The list or dictionary of attributes.
        inputs (list[str]): List of expected inputs.
        outputs (list[str]): List of expected outputs.
        version (int): Operator version (default: 1).
        schema_version (int | None): Optional upstream opset/schema version.
        deprecated (bool): Whether this operator schema is deprecated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    domain: str
    attributes: (
        list[CustomAttributeSchema] | dict[str, CustomAttributeSchema | OpAttribute]
    ) = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    version: int = 1
    schema_version: int | None = None
    deprecated: bool = False

    def __init__(
        self,
        name: str,
        domain: str,
        attributes: (
            Sequence[CustomAttributeSchema]
            | Mapping[str, CustomAttributeSchema | OpAttribute]
            | None
        ) = None,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        version: int = 1,
        schema_version: int | None = None,
        deprecated: bool = False,
    ) -> None:
        """Initialize CustomOpSchema.

        Args:
            name (str): The name of the operator.
            domain (str): The custom domain.
            attributes (Sequence[CustomAttributeSchema] | Mapping[str, CustomAttributeSchema | OpAttribute] | None):
                Attributes list or dictionary.
            inputs (Sequence[str] | None): Expected input names.
            outputs (Sequence[str] | None): Expected output names.
            version (int): Operator version (default: 1).
            schema_version (int | None): Optional upstream opset/schema version.
            deprecated (bool): Whether this operator schema is deprecated.
        """
        norm_attrs: (
            list[CustomAttributeSchema] | dict[str, CustomAttributeSchema | OpAttribute]
        ) = (
            dict(attributes)
            if isinstance(attributes, Mapping)
            else list(attributes or [])
        )
        super().__init__(
            name=name,
            domain=domain,
            attributes=norm_attrs,
            inputs=list(inputs) if inputs is not None else [],
            outputs=list(outputs) if outputs is not None else [],
            version=version,
            schema_version=schema_version,
            deprecated=deprecated,
        )

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
    op_data: Mapping[str, object],
    domain_default: str,
    attr_type_default: str = "str",
) -> OpSchema:
    """Parse a single JSON operator schema dictionary entry into an OpSchema.

    Args:
        op_data (Mapping[str, object]): Raw JSON operator definition dictionary.
        domain_default (str): Fallback domain identifier if not present in op_data.
        attr_type_default (str): Fallback attribute type string if not present.

    Returns:
        OpSchema: Parsed operator schema.
    """
    raw_attrs_obj = op_data.get("attributes", [])
    raw_attrs_seq = (
        raw_attrs_obj
        if isinstance(raw_attrs_obj, Sequence)
        and not isinstance(raw_attrs_obj, (str, bytes))
        else []
    )
    attrs: list[CustomAttributeSchema] = [
        CustomAttributeSchema(
            name=str(a.get("name", "")),
            type=str(a.get("type", attr_type_default)),
            required=bool(a.get("required", False)),
            default=a.get("default", None),
        )
        for a in raw_attrs_seq
        if isinstance(a, Mapping)
    ]

    raw_sv = op_data.get("schema_version")
    raw_inputs = op_data.get("inputs", [])
    inputs_list = (
        [str(x) for x in raw_inputs]
        if isinstance(raw_inputs, Sequence) and not isinstance(raw_inputs, (str, bytes))
        else []
    )
    raw_outputs = op_data.get("outputs", [])
    outputs_list = (
        [str(x) for x in raw_outputs]
        if isinstance(raw_outputs, Sequence)
        and not isinstance(raw_outputs, (str, bytes))
        else []
    )
    raw_ver = op_data.get("version", 1)
    ver_int = int(raw_ver) if isinstance(raw_ver, (int, str)) else 1

    schema = CustomOpSchema(
        name=str(op_data["name"]),
        domain=str(op_data.get("domain", domain_default)),
        attributes=attrs,
        inputs=inputs_list,
        outputs=outputs_list,
        version=ver_int,
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
    on_entry: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, OpSchema]:
    """Load operator schemas from a JSON catalog file containing an 'ops' list.

    Args:
        filename (str): Name of the JSON catalog file in the schema/snapshot directory.
        domain_default (str): Default domain identifier for loaded operations.
        json_path (Path | str | None): Explicit path override for the JSON catalog.
        find_file_fn (Callable[[str, Path | str | None], Path | None]): File resolution function.
        attr_type_default (str): Default attribute type string when unspecified.
        on_entry (Callable[[Mapping[str, object]], None] | None): Optional callback invoked per raw op_data entry.

    Returns:
        dict[str, OpSchema]: Mapping from operator names to parsed OpSchema definitions.
    """
    registry: dict[str, OpSchema] = {}
    target = find_file_fn(filename, json_path)
    if target is not None and target.exists():
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
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


def build_isa_primitive_schemas(
    primitives: Mapping[str, object],
    domain: str,
) -> dict[str, OpSchema]:
    """Convert raw ISA instruction primitive definitions into OpSchema instances.

    Args:
        primitives (Mapping[str, object]): Mapping of instruction mnemonics to operand specs.
        domain (str): Target hardware ISA domain (e.g. 'amd_rdna', 'nvidia_sass').

    Returns:
        dict[str, OpSchema]: Mapping from instruction mnemonic to OpSchema.
    """
    result: dict[str, OpSchema] = {}
    for op_name, op_info in primitives.items():
        raw_operands = (
            op_info.get("operands", [])
            if isinstance(op_info, Mapping)
            else getattr(op_info, "operands", [])
        )
        operands = (
            [str(op) for op in raw_operands]
            if isinstance(raw_operands, Sequence)
            and not isinstance(raw_operands, (str, bytes))
            else []
        )
        inputs = [op for op in operands if op != "dst"]
        outputs = ["dst"] if "dst" in operands else []
        schema = CustomOpSchema(
            name=op_name,
            domain=domain,
            attributes=[],
            inputs=inputs,
            outputs=outputs,
        )
        result[op_name] = schema.to_op_schema()
    return result


@contextlib.contextmanager
def override_registry(
    registry: dict[str, OpSchema],
    overrides: dict[str, OpSchema] | None = None,
    clear: bool = False,
) -> Iterator[dict[str, OpSchema]]:
    """Thread-safe context manager to temporarily modify or replace an operator registry.

    Acquires ``REGISTRY_LOCK`` for the duration of the block and restores the
    original registry contents on exit.

    Args:
        registry (dict[str, OpSchema]): Target operator registry dictionary.
        overrides (dict[str, OpSchema] | None): Optional mapping of schemas to apply.
        clear (bool): If True, clears existing entries before applying overrides.

    Yields:
        dict[str, OpSchema]: The active registry dictionary.
    """
    with REGISTRY_LOCK:
        saved = dict(registry)
        try:
            if clear:
                registry.clear()
            if overrides is not None:
                registry.update(overrides)
            yield registry
        finally:
            registry.clear()
            registry.update(saved)
