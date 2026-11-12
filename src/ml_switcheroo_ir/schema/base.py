"""Core operator schema Pydantic models, canonical attribute types, and lazy registry containers."""

from __future__ import annotations

import contextlib
import importlib.resources
import json
import threading
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from enum import Enum
from pathlib import Path

import pydantic

from ml_switcheroo_ir.schema.ghost import (
    RdnaPrimitiveSpec,
    SassPrimitiveSpec,
)
from ml_switcheroo_ir.snapshots import find_schema_file
from ml_switcheroo_ir.types import AttributeValue, JsonValue

REGISTRY_LOCK = threading.RLock()

_ValT = typing.TypeVar("_ValT")


class AttributeType(str, Enum):
    """Canonical enumeration of supported operator schema attribute types.

    Attributes:
        INT (AttributeType): Scalar 64-bit signed integer ('int').
        FLOAT (AttributeType): Scalar IEEE-754 floating-point number ('float').
        STR (AttributeType): UTF-8 string value ('str').
        BOOL (AttributeType): Boolean flag ('bool').
        DICT (AttributeType): String-keyed mapping of attribute values ('dict').
        LIST_INT (AttributeType): Homogeneous sequence of integers ('List[int]').
        LIST_FLOAT (AttributeType): Homogeneous sequence of floats ('List[float]').
        LIST_STR (AttributeType): Homogeneous sequence of strings ('List[str]').
        LIST (AttributeType): Heterogeneous or untyped sequence ('list').
        TENSOR (AttributeType): Constant tensor literal or TensorSpec ('Tensor').
        GRAPH (AttributeType): Nested LogicalGraph region ('Graph').
        DTYPE (AttributeType): Element data type descriptor ('DType').
        ANY (AttributeType): Polymorphic union of valid AttributeValue types ('Any').
    """

    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    DICT = "dict"
    LIST_INT = "List[int]"
    LIST_FLOAT = "List[float]"
    LIST_STR = "List[str]"
    LIST = "list"
    TENSOR = "Tensor"
    GRAPH = "Graph"
    DTYPE = "DType"
    ANY = "Any"


_ATTRIBUTE_TYPE_SYNONYMS: dict[str, AttributeType] = {
    "int": AttributeType.INT,
    "i32": AttributeType.INT,
    "i64": AttributeType.INT,
    "i64attr": AttributeType.INT,
    "float": AttributeType.FLOAT,
    "f32": AttributeType.FLOAT,
    "f64": AttributeType.FLOAT,
    "f32attr": AttributeType.FLOAT,
    "str": AttributeType.STR,
    "string": AttributeType.STR,
    "strattr": AttributeType.STR,
    "flatsymbolrefattr": AttributeType.STR,
    "bool": AttributeType.BOOL,
    "boolattr": AttributeType.BOOL,
    "unitattr": AttributeType.BOOL,
    "dict": AttributeType.DICT,
    "dictionaryattr": AttributeType.DICT,
    "list[int]": AttributeType.LIST_INT,
    "ints": AttributeType.LIST_INT,
    "densei64arrayattr": AttributeType.LIST_INT,
    "i64elementsattr": AttributeType.LIST_INT,
    "list[float]": AttributeType.LIST_FLOAT,
    "floats": AttributeType.LIST_FLOAT,
    "densef32arrayattr": AttributeType.LIST_FLOAT,
    "list[str]": AttributeType.LIST_STR,
    "strings": AttributeType.LIST_STR,
    "list": AttributeType.LIST,
    "list[any]": AttributeType.LIST,
    "arrayattr": AttributeType.LIST,
    "denseboolarrayattr": AttributeType.LIST,
    "tensor": AttributeType.TENSOR,
    "sparse_tensor": AttributeType.TENSOR,
    "elementsattr": AttributeType.TENSOR,
    "tensors": AttributeType.LIST,
    "graph": AttributeType.GRAPH,
    "graphs": AttributeType.LIST,
    "dtype": AttributeType.DTYPE,
    "type_proto": AttributeType.DTYPE,
    "type_protos": AttributeType.LIST,
    "sparse_tensors": AttributeType.LIST,
    "any": AttributeType.ANY,
}


def _classify_mlir_attr_descriptor(lowered: str) -> AttributeType | None:
    """Classify TableGen/StableHLO composite attribute descriptors into canonical AttributeType.

    Args:
        lowered (str): Lowercased raw attribute type string.

    Returns:
        AttributeType | None: Matched AttributeType if recognized, else None.
    """
    result: AttributeType | None = None
    if "anyattrof" in lowered:
        result = AttributeType.ANY
    elif "i32attr" in lowered or "i64attr" in lowered:
        result = AttributeType.LIST_INT if ("array" in lowered or "elements" in lowered) else AttributeType.INT
    elif "densei64array" in lowered or "i64elements" in lowered:
        result = AttributeType.LIST_INT
    elif any(tok in lowered for tok in ("arrayattr", "arrayof", "precisionconfig", "denseboolarray")):
        result = AttributeType.LIST
    elif any(tok in lowered for tok in ("dimensionnumbers", "channelhandle", "dotalgorithm")):
        result = AttributeType.DICT
    elif lowered.startswith(("stablehlo_", "optionalattr<stablehlo_")):
        result = AttributeType.STR
    return result


def normalize_attribute_type(raw_type: str | AttributeType) -> AttributeType:
    """Normalize a raw schema or TableGen attribute type string into a canonical AttributeType.

    Args:
        raw_type (str | AttributeType): Raw type literal or AttributeType member.

    Returns:
        AttributeType: Canonicalized AttributeType enum member.

    Raises:
        ValueError: If ``raw_type`` is empty or unrecognized.
    """
    if isinstance(raw_type, AttributeType):
        return raw_type
    cleaned = raw_type.strip()
    if not cleaned:
        raise ValueError("AttributeType string cannot be empty.")
    direct = _ATTRIBUTE_TYPE_SYNONYMS.get(cleaned.lower())
    if direct is not None:
        return direct
    mlir_kind = _classify_mlir_attr_descriptor(cleaned.lower())
    if mlir_kind is not None:
        return mlir_kind
    raise ValueError(f"Unsupported AttributeType descriptor: {raw_type!r}")


class RawOpAttributeSpec(pydantic.BaseModel):
    """Pydantic model validating a raw JSON attribute entry in an operator catalog.

    Attributes:
        name (str): Attribute name.
        type (AttributeType): Canonicalized attribute type.
        required (bool): Whether the attribute is required.
        default (AttributeValue): Optional default value.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str = ""
    type: AttributeType = AttributeType.STR
    required: bool = False
    default: AttributeValue = None

    @pydantic.field_validator("type", mode="before")
    @classmethod
    def _coerce_attr_type(cls, value: str | AttributeType) -> AttributeType:
        """Normalize raw attribute type strings into AttributeType.

        Args:
            value (str | AttributeType): Raw type descriptor.

        Returns:
            AttributeType: Canonicalized AttributeType.
        """
        return normalize_attribute_type(value)


class RawOpSchemaSpec(pydantic.BaseModel):
    """Pydantic model validating a raw JSON operator entry in an operator catalog.

    Attributes:
        name (str): Operator name.
        domain (str): Operator domain.
        version (int): Operator version.
        schema_version (int | None): Optional upstream schema version.
        attributes (list[RawOpAttributeSpec] | dict[str, RawOpAttributeSpec]): Operator attributes.
        inputs (list[str]): Input operand names.
        outputs (list[str]): Output operand names.
        aliases (list[str]): Secondary alias names resolving to this operator.
        latency (int | None): Optional execution latency alias.
        execution_latency (int | None): Optional execution pipeline latency.
        vopd_slot (str | None): Optional RDNA VOPD slot.
        vopd_target (str | None): Optional RDNA VOPD target opcode.
        vopd_equiv (str | None): Optional RDNA VOPD equivalent opcode.
        dialects (dict[str, str] | None): Optional cross-dialect lowering map (ODL).
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    domain: str = ""
    version: int = 1
    schema_version: int | None = None
    attributes: list[RawOpAttributeSpec] | dict[str, RawOpAttributeSpec] = pydantic.Field(default_factory=list)
    inputs: list[str] = pydantic.Field(default_factory=list)
    outputs: list[str] = pydantic.Field(default_factory=list)
    aliases: list[str] = pydantic.Field(default_factory=list)
    latency: int | None = None
    execution_latency: int | None = None
    vopd_slot: str | None = None
    vopd_target: str | None = None
    vopd_equiv: str | None = None
    dialects: dict[str, str] | None = None


class OpAttribute(pydantic.BaseModel):
    """Unified operator attribute schema used across built-in and custom dialects.

    Attributes:
        name (str): Name of the attribute.
        type (AttributeType): Canonical AttributeType descriptor.
        required (bool): Whether the attribute must be provided.
        default (AttributeValue): Default value when optional.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str = ""
    type: AttributeType = AttributeType.ANY
    required: bool = False
    default: AttributeValue = None

    @pydantic.field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, value: str | AttributeType) -> AttributeType:
        """Normalize string or enum attribute type into canonical AttributeType.

        Args:
            value (str | AttributeType): Candidate attribute type.

        Returns:
            AttributeType: Validated AttributeType enum member.
        """
        return normalize_attribute_type(value)

    def __init__(
        self,
        name: str = "",
        type: AttributeType | str = AttributeType.ANY,
        required: bool = False,
        default: AttributeValue = None,
    ) -> None:
        """Initialize OpAttribute.

        Args:
            name (str): Name of the attribute.
            type (AttributeType | str): Type descriptor enum or string.
            required (bool): Whether the attribute must be provided.
            default (AttributeValue): Default value if optional.
        """
        super().__init__(
            name=name,
            type=normalize_attribute_type(type),
            required=required,
            default=default,
        )

    def to_op_attribute(self) -> OpAttribute:
        """Return self for backwards-compatible custom attribute conversion.

        Returns:
            OpAttribute: This canonical operator attribute schema.
        """
        return self


def _normalize_op_attributes(
    attributes: Sequence[OpAttribute] | Mapping[str, OpAttribute] | None,
) -> dict[str, OpAttribute]:
    """Normalize sequence or mapping attribute inputs into a canonical ``dict[str, OpAttribute]``.

    Args:
        attributes (Sequence[OpAttribute] | Mapping[str, OpAttribute] | None):
            Sequence or dictionary of OpAttribute instances.

    Returns:
        dict[str, OpAttribute]: Normalized attribute dictionary keyed by canonical attribute name.
    """
    if attributes is None:
        return {}
    attr_dict: dict[str, OpAttribute] = {}
    if isinstance(attributes, Mapping):
        for key, attr in attributes.items():
            resolved_name = attr.name if attr.name else key
            if attr.name == resolved_name:
                attr_dict[resolved_name] = attr
            else:
                attr_dict[resolved_name] = OpAttribute(
                    name=resolved_name,
                    type=attr.type,
                    required=attr.required,
                    default=attr.default,
                )
        return attr_dict
    for attr in attributes:
        attr_dict[attr.name] = attr
    return attr_dict


class OpSchema(pydantic.BaseModel):
    """Unified operator schema supporting canonical attributes and secondary alias names.

    Attributes:
        name (str): Canonical operator identifier name.
        domain (str): Domain namespace of the operator.
        version (int): Operator schema version integer.
        attributes (dict[str, OpAttribute]): Mapping of attribute names to OpAttribute instances.
        inputs (list[str]): List of formal input operand names.
        outputs (list[str]): List of formal output operand names.
        schema_version (int | None): Optional upstream opset/schema version.
        aliases (tuple[str, ...]): Secondary alias names resolving to this canonical schema.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    domain: str
    version: int = 1
    attributes: dict[str, OpAttribute] = pydantic.Field(default_factory=dict)
    inputs: list[str] = pydantic.Field(default_factory=list)
    outputs: list[str] = pydantic.Field(default_factory=list)
    schema_version: int | None = None
    aliases: tuple[str, ...] = ()

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        domain: str,
        version: int = 1,
        attributes: Sequence[OpAttribute] | Mapping[str, OpAttribute] | None = None,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
        schema_version: int | None = None,
        aliases: Sequence[str] = (),
    ) -> None:
        """Initialize OpSchema.

        Args:
            name (str): Canonical operator identifier name.
            domain (str): Domain namespace of the operator.
            version (int): Operator schema version integer (default: 1).
            attributes (Sequence[OpAttribute] | Mapping[str, OpAttribute] | None):
                Sequence or mapping of OpAttribute specifications.
            inputs (Sequence[str] | None): Formal input operand names.
            outputs (Sequence[str] | None): Formal output operand names.
            schema_version (int | None): Optional upstream opset/schema version.
            aliases (Sequence[str]): Secondary alias names for secondary index resolution.
        """
        super().__init__(
            name=name,
            domain=domain,
            version=version,
            attributes=_normalize_op_attributes(attributes),
            inputs=list(inputs) if inputs is not None else [],
            outputs=list(outputs) if outputs is not None else [],
            schema_version=schema_version,
            aliases=tuple(aliases),
        )

    def to_op_schema(self) -> OpSchema:
        """Return self for backwards-compatible custom operator conversion.

        Returns:
            OpSchema: This canonical operator schema.
        """
        return self


# Unified canonical aliases (eliminating duplicate parallel Pydantic classes)
CustomAttributeSchema = OpAttribute
CustomOpSchema = OpSchema


class SchemaRegistry(MutableMapping[str, OpSchema]):
    """Thread-safe lazy operator registry with secondary alias index resolution.

    Stores each canonical ``OpSchema`` once under ``schema.name`` while maintaining a
    secondary ``_alias_index`` mapping ``schema.aliases`` to ``schema.name``. Iterating
    keys, items, or values yields strictly canonical entries where ``key == schema.name``.

    Attributes:
        name (str): Human-readable registry identifier.
    """

    name: str

    def __init__(
        self,
        name: str = "registry",
        loader: Callable[[], None] | None = None,
        initial: Mapping[str, OpSchema] | Iterable[OpSchema] | None = None,
    ) -> None:
        """Initialize SchemaRegistry with optional lazy loader or initial schemas.

        Args:
            name (str): Human-readable registry identifier.
            loader (Callable[[], None] | None): Lazy initialization callback guarded by REGISTRY_LOCK.
            initial (Mapping[str, OpSchema] | Iterable[OpSchema] | None): Optional pre-populated schemas.
        """
        self.name = name
        self._loader = loader
        self._loaded = loader is None
        self._loading = False
        self._data: dict[str, OpSchema] = {}
        self._alias_index: dict[str, str] = {}
        if initial is not None:
            if isinstance(initial, Mapping):
                for k, v in initial.items():
                    self._insert_unlocked(k, v)
            else:
                for schema in initial:
                    self._insert_unlocked(schema.name, schema)

    def _run_loader_locked(self) -> None:
        """Execute ``_loader`` while holding ``REGISTRY_LOCK`` if still unloaded."""
        if not self._loaded and not self._loading and self._loader is not None:
            self._loading = True
            try:
                self._loader()
                self._loaded = True
            finally:
                self._loading = False

    def _ensure_loaded(self) -> None:
        """Execute lazy loader once under ``REGISTRY_LOCK`` if not yet initialized."""
        if self._loaded or self._loading:
            return
        with REGISTRY_LOCK:
            self._run_loader_locked()

    def _insert_unlocked(self, key: str, value: OpSchema) -> None:
        """Insert ``value`` under ``key`` and register ``value.aliases`` in ``_alias_index``.

        Args:
            key (str): Primary dictionary key.
            value (OpSchema): Operator schema to store.
        """
        self._data[key] = value
        for alias in value.aliases:
            if alias and alias != key:
                self._alias_index[alias] = key

    def _rebuild_aliases_unlocked(self) -> None:
        """Rebuild ``_alias_index`` from currently stored canonical schemas."""
        self._alias_index.clear()
        for key, schema in self._data.items():
            for alias in schema.aliases:
                if alias and alias != key and alias not in self._data:
                    self._alias_index[alias] = key

    def __getitem__(self, key: str) -> OpSchema:
        """Retrieve an OpSchema by canonical name or registered secondary alias.

        Args:
            key (str): Canonical operator name or secondary alias.

        Returns:
            OpSchema: Resolved operator schema.

        Raises:
            KeyError: If neither canonical key nor alias exists.
        """
        self._ensure_loaded()
        if key in self._data:
            return self._data[key]
        target = self._alias_index.get(key)
        if target is not None and target in self._data:
            return self._data[target]
        raise KeyError(key)

    def __setitem__(self, key: str, value: OpSchema) -> None:
        """Store an OpSchema under ``key`` and index any declared ``value.aliases``.

        Args:
            key (str): Canonical operator name.
            value (OpSchema): Operator schema instance.
        """
        self._ensure_loaded()
        with REGISTRY_LOCK:
            self._insert_unlocked(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete an operator schema by canonical key or alias and refresh secondary index.

        Args:
            key (str): Canonical operator name or alias to remove.

        Raises:
            KeyError: If ``key`` is not present.
        """
        self._ensure_loaded()
        with REGISTRY_LOCK:
            if key in self._data:
                del self._data[key]
                self._rebuild_aliases_unlocked()
                return
            if key in self._alias_index:
                del self._alias_index[key]
                return
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        """Yield canonical operator names stored in this registry.

        Returns:
            Iterator[str]: Iterator over canonical keys.
        """
        self._ensure_loaded()
        return iter(self._data)

    def __len__(self) -> int:
        """Return number of canonical operator schemas stored in this registry.

        Returns:
            int: Count of canonical entries.
        """
        self._ensure_loaded()
        return len(self._data)

    def clear(self) -> None:
        """Clear all canonical schemas and secondary aliases under ``REGISTRY_LOCK``."""
        with REGISTRY_LOCK:
            self._loaded = True
            self._data.clear()
            self._alias_index.clear()

    def copy(self) -> dict[str, OpSchema]:
        """Return a shallow dictionary copy of canonical schemas.

        Returns:
            dict[str, OpSchema]: Dictionary of canonical key-to-schema entries.
        """
        self._ensure_loaded()
        return dict(self._data)

    def resolve(self, name: str) -> OpSchema | None:
        """Resolve an operator name or secondary alias to its canonical OpSchema.

        Args:
            name (str): Canonical operator name or alias.

        Returns:
            OpSchema | None: Matching OpSchema or None if ungrounded.
        """
        return self.get(name)

    @property
    def alias_index(self) -> dict[str, str]:
        """Return a copy of the secondary ``alias -> canonical_name`` lookup table.

        Returns:
            dict[str, str]: Secondary alias mapping.
        """
        self._ensure_loaded()
        return dict(self._alias_index)

    @property
    def is_loaded(self) -> bool:
        """Return True if lazy initialization has already executed.

        Returns:
            bool: Initialization status flag.
        """
        return self._loaded


class LazyMetadataMap(MutableMapping[str, _ValT]):
    """Thread-safe lazy metadata dictionary synchronized with a dialect's lazy loader.

    Attributes:
        name (str): Metadata table identifier.
    """

    name: str

    def __init__(
        self,
        name: str,
        loader: Callable[[], None] | None = None,
    ) -> None:
        """Initialize LazyMetadataMap.

        Args:
            name (str): Metadata table identifier.
            loader (Callable[[], None] | None): Shared lazy loader guarded by REGISTRY_LOCK.
        """
        self.name = name
        self._loader = loader
        self._loaded = loader is None
        self._loading = False
        self._data: dict[str, _ValT] = {}

    def _run_loader_locked(self) -> None:
        """Execute ``_loader`` while holding ``REGISTRY_LOCK`` if still unloaded."""
        if not self._loaded and not self._loading and self._loader is not None:
            self._loading = True
            try:
                self._loader()
                self._loaded = True
            finally:
                self._loading = False

    def _ensure_loaded(self) -> None:
        """Invoke shared lazy loader once under ``REGISTRY_LOCK`` when first read."""
        if self._loaded or self._loading:
            return
        with REGISTRY_LOCK:
            self._run_loader_locked()

    def mark_loaded(self) -> None:
        """Mark metadata map as initialized after shared loader completes."""
        self._loaded = True

    def __getitem__(self, key: str) -> _ValT:
        """Retrieve metadata entry by key.

        Args:
            key (str): Entry key.

        Returns:
            _ValT: Associated metadata value.
        """
        self._ensure_loaded()
        return self._data[key]

    def __setitem__(self, key: str, value: _ValT) -> None:
        """Set metadata entry under ``REGISTRY_LOCK``.

        Args:
            key (str): Entry key.
            value (_ValT): Metadata value.
        """
        self._ensure_loaded()
        with REGISTRY_LOCK:
            self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete metadata entry under ``REGISTRY_LOCK``.

        Args:
            key (str): Entry key.
        """
        self._ensure_loaded()
        with REGISTRY_LOCK:
            del self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate keys in metadata map.

        Returns:
            Iterator[str]: Key iterator.
        """
        self._ensure_loaded()
        return iter(self._data)

    def __len__(self) -> int:
        """Return number of entries in metadata map.

        Returns:
            int: Entry count.
        """
        self._ensure_loaded()
        return len(self._data)

    def clear(self) -> None:
        """Clear all entries under ``REGISTRY_LOCK``."""
        with REGISTRY_LOCK:
            self._loaded = True
            self._data.clear()

    def copy(self) -> dict[str, _ValT]:
        """Return a shallow dictionary copy of metadata entries.

        Returns:
            dict[str, _ValT]: Copied dictionary.
        """
        self._ensure_loaded()
        return dict(self._data)


def load_bundled_catalog_json(filename: str) -> dict[str, JsonValue] | None:
    """Load a bundled JSON schema catalog via ``importlib.resources`` from ``ml_switcheroo_ir.schema.catalogs``.

    Args:
        filename (str): Catalog JSON filename (e.g. ``'onnx_ops.json'``, ``'stablehlo_ops.json'``).

    Returns:
        dict[str, JsonValue] | None: Parsed JSON dictionary if bundled resource exists and is valid, else None.
    """
    try:
        pkg_files = importlib.resources.files("ml_switcheroo_ir.schema.catalogs")
        resource = pkg_files.joinpath(filename)
        with importlib.resources.as_file(resource) as res_path:
            if not res_path.is_file():
                return None
            with open(res_path, encoding="utf-8") as f:
                loaded = json.load(f)
            return loaded if isinstance(loaded, dict) else None
    except (ModuleNotFoundError, FileNotFoundError, OSError, ValueError):
        return None


def parse_op_schema_entry(
    op_data: Mapping[str, JsonValue],
    domain_default: str,
    attr_type_default: AttributeType | str = AttributeType.STR,
) -> OpSchema:
    """Validate and parse a JSON operator catalog entry through RawOpSchemaSpec into an OpSchema.

    Args:
        op_data (Mapping[str, JsonValue]): Raw JSON operator definition dictionary.
        domain_default (str): Fallback domain identifier if not present in op_data.
        attr_type_default (AttributeType | str): Fallback attribute type when unspecified.

    Returns:
        OpSchema: Validated canonical operator schema.
    """
    normalized_payload: dict[str, JsonValue] = dict(op_data)
    if not normalized_payload.get("domain"):
        normalized_payload["domain"] = domain_default

    default_type_str = (
        attr_type_default.value if isinstance(attr_type_default, AttributeType) else str(attr_type_default)
    )
    raw_attrs_obj = normalized_payload.get("attributes", [])
    if isinstance(raw_attrs_obj, Mapping):
        coerced_map: dict[str, JsonValue] = {}
        for k, v in raw_attrs_obj.items():
            if isinstance(v, Mapping):
                item_dict = dict(v)
                item_dict.setdefault("name", str(k))
                item_dict.setdefault("type", default_type_str)
                coerced_map[str(k)] = item_dict
        normalized_payload["attributes"] = coerced_map
    elif isinstance(raw_attrs_obj, Sequence) and not isinstance(raw_attrs_obj, (str, bytes)):
        coerced_list: list[JsonValue] = []
        for item in raw_attrs_obj:
            if isinstance(item, Mapping):
                item_dict = dict(item)
                item_dict.setdefault("type", default_type_str)
                coerced_list.append(item_dict)
        normalized_payload["attributes"] = coerced_list

    spec = RawOpSchemaSpec.model_validate(normalized_payload)
    if isinstance(spec.attributes, dict):
        attrs: dict[str, OpAttribute] = {
            (a.name if a.name else k): OpAttribute(
                name=a.name if a.name else k,
                type=a.type,
                required=a.required,
                default=a.default,
            )
            for k, a in spec.attributes.items()
        }
    else:
        attrs = {
            a.name: OpAttribute(
                name=a.name,
                type=a.type,
                required=a.required,
                default=a.default,
            )
            for a in spec.attributes
        }

    return OpSchema(
        name=spec.name,
        domain=spec.domain,
        version=spec.version,
        attributes=attrs,
        inputs=spec.inputs,
        outputs=spec.outputs,
        schema_version=spec.schema_version,
        aliases=spec.aliases,
    )


def _populate_catalog_entries(
    data: Mapping[str, JsonValue],
    domain_default: str,
    attr_type_default: AttributeType | str,
    on_entry: Callable[[Mapping[str, JsonValue]], None] | None,
) -> dict[str, OpSchema]:
    """Parse an ``{'ops': [...]}`` JSON dictionary into ``OpSchema`` instances.

    Args:
        data (Mapping[str, JsonValue]): Parsed JSON catalog dictionary.
        domain_default (str): Default operator domain string.
        attr_type_default (AttributeType | str): Default attribute type.
        on_entry (Callable[[Mapping[str, JsonValue]], None] | None): Per-entry hook callback.

    Returns:
        dict[str, OpSchema]: Parsed canonical schemas keyed by ``schema.name``.
    """
    registry: dict[str, OpSchema] = {}
    raw_ops = data.get("ops", [])
    if not isinstance(raw_ops, Sequence) or isinstance(raw_ops, (str, bytes)):
        return registry
    for op_data in raw_ops:
        if isinstance(op_data, Mapping):
            schema = parse_op_schema_entry(
                op_data,
                domain_default=domain_default,
                attr_type_default=attr_type_default,
            )
            registry[schema.name] = schema
            if on_entry is not None:
                on_entry(op_data)
    return registry


def load_op_schema_catalog(
    filename: str,
    domain_default: str,
    json_path: Path | str | None = None,
    find_file_fn: Callable[[str, Path | str | None], Path | None] | None = None,
    attr_type_default: AttributeType | str = AttributeType.STR,
    on_entry: Callable[[Mapping[str, JsonValue]], None] | None = None,
) -> dict[str, OpSchema]:
    """Load operator schemas from an external snapshot file or bundled ``importlib.resources`` catalog.

    Args:
        filename (str): Name of the JSON catalog file in the snapshot or bundled catalogs directory.
        domain_default (str): Default domain identifier for loaded operations.
        json_path (Path | str | None): Explicit path override for the JSON catalog.
        find_file_fn (Callable[[str, Path | str | None], Path | None] | None): File resolution function override.
        attr_type_default (AttributeType | str): Default attribute type when unspecified.
        on_entry (Callable[[Mapping[str, JsonValue]], None] | None): Optional callback invoked per raw op_data entry.

    Returns:
        dict[str, OpSchema]: Mapping from operator names to parsed OpSchema definitions.
    """
    finder = find_schema_file if find_file_fn is None else find_file_fn
    target = finder(filename, json_path)
    if target is not None and target.exists():
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, Mapping):
            return _populate_catalog_entries(data, domain_default, attr_type_default, on_entry)
        return {}

    if json_path is None and find_file_fn is None:
        bundled = load_bundled_catalog_json(filename)
        if bundled is not None:
            return _populate_catalog_entries(bundled, domain_default, attr_type_default, on_entry)
    return {}


def build_isa_primitive_schemas(
    primitives: Mapping[str, RdnaPrimitiveSpec] | Mapping[str, SassPrimitiveSpec],
    domain: str,
) -> dict[str, OpSchema]:
    """Convert raw ISA instruction primitive definitions into OpSchema instances.

    Args:
        primitives (Mapping[str, RdnaPrimitiveSpec] | Mapping[str, SassPrimitiveSpec]):
            Mapping of instruction mnemonics to operand specs.
        domain (str): Target hardware ISA domain (e.g. 'amd_rdna', 'nvidia_sass').

    Returns:
        dict[str, OpSchema]: Mapping from instruction mnemonic to OpSchema.
    """
    result: dict[str, OpSchema] = {}
    for op_name, op_info in primitives.items():
        operands = list(op_info.operands)
        inputs = [op for op in operands if op != "dst"]
        outputs = ["dst"] if "dst" in operands else []
        result[op_name] = OpSchema(
            name=op_name,
            domain=domain,
            attributes={},
            inputs=inputs,
            outputs=outputs,
        )
    return result


@contextlib.contextmanager
def override_registry(
    registry: MutableMapping[str, OpSchema],
    overrides: Mapping[str, OpSchema] | None = None,
    clear: bool = False,
) -> Iterator[MutableMapping[str, OpSchema]]:
    """Thread-safe context manager to temporarily modify or replace an operator registry.

    Acquires ``REGISTRY_LOCK`` for the duration of the block and restores the
    original registry contents on exit.

    Args:
        registry (MutableMapping[str, OpSchema]): Target operator registry mapping.
        overrides (Mapping[str, OpSchema] | None): Optional mapping of schemas to apply.
        clear (bool): If True, clears existing entries before applying overrides.

    Yields:
        MutableMapping[str, OpSchema]: The active registry mapping.
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
