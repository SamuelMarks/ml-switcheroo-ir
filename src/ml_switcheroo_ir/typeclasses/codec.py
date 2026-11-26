"""Bidirectional Codec, Pretty, and PythonExprEmit typeclass witness dictionaries."""

from __future__ import annotations

import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pydantic
import typing_extensions

from ml_switcheroo_ir.typeclasses.algebra import NonEmpty
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

JsonValue: typing_extensions.TypeAlias = pydantic.JsonValue

_WireT = typing.TypeVar("_WireT")
_MidT = typing.TypeVar("_MidT")
_A = typing.TypeVar("_A")
_B = typing.TypeVar("_B")


def _describe_wire_value(val: JsonValue) -> str:
    """Return a deterministic type label for a ``JsonValue`` without reflection or dunder access.

    Args:
        val (JsonValue): Candidate JSON wire value.

    Returns:
        str: Canonical JSON value kind name.
    """
    kind: str
    if val is None:
        kind = "NoneType"
    elif isinstance(val, bool):
        kind = "bool"
    elif isinstance(val, int):
        kind = "int"
    elif isinstance(val, float):
        kind = "float"
    elif isinstance(val, str):
        kind = "str"
    elif isinstance(val, list):
        kind = "list"
    else:
        kind = "dict"
    return kind


__all__ = [
    "Codec",
    "CodecError",
    "JsonCodec",
    "Pretty",
    "PythonExprEmit",
    "compose_codec",
    "dict_str_codec",
    "imap_codec",
    "list_codec",
    "nonempty_pretty",
    "optional_codec",
    "refine_codec",
    "tuple_codec",
]


@dataclass(frozen=True)
class CodecError(ValueError):
    """Structured deserialization failure carrying JSON/wire path breadcrumbs.

    Attributes:
        path (tuple[str | int, ...]): Sequence of key/index breadcrumbs from root to failure site.
        expected (str): Human-readable description of expected wire schema or type.
        actual_description (str): Human-readable description of encountered value type.
        message (str): Detailed diagnostic explanation.
    """

    path: tuple[str | int, ...]
    expected: str
    actual_description: str
    message: str

    def __post_init__(self) -> None:
        """Initialize the underlying ``ValueError`` message string."""
        super().__init__(self.format_diagnostic())

    def prepend_path(self, segment: str | int) -> CodecError:
        """Return a new ``CodecError`` with ``segment`` prepended to ``self.path``.

        Args:
            segment (str | int): Outer field key or sequence index.

        Returns:
            CodecError: Path-enriched codec error.
        """
        return CodecError(
            path=(segment, *self.path),
            expected=self.expected,
            actual_description=self.actual_description,
            message=self.message,
        )

    def format_diagnostic(self) -> str:
        """Render a deterministic human-readable diagnostic string including JSONPointer-like path.

        Returns:
            str: Formatted diagnostic string.
        """
        loc = "$"
        for seg in self.path:
            if isinstance(seg, int) and not isinstance(seg, bool):
                loc += f"[{seg}]"
            else:
                loc += f".{seg}"
        return f"{loc}: expected {self.expected}, got {self.actual_description} ({self.message})"


@dataclass(frozen=True)
class Codec(typing.Generic[_WireT, _A]):
    """Bidirectional serialization witness pairing total ``encode`` with total ``Result``-returning ``decode``.

    Law (Prism / Round-Trip):
        ``decode(encode(x)) == Ok(x)``

    Attributes:
        encode (Callable[[_A], _WireT]): Total serializer mapping ``_A`` to wire representation ``_WireT``.
        decode (Callable[[_WireT], Result[CodecError, _A]]): Total parser mapping ``_WireT`` to
            ``Result[CodecError, _A]``.
    """

    encode: Callable[[_A], _WireT]
    decode: Callable[[_WireT], Result[CodecError, _A]]


JsonCodec: typing_extensions.TypeAlias = Codec[JsonValue, _A]


def imap_codec(
    codec: Codec[_WireT, _A],
    to_b: Callable[[_A], _B],
    from_b: Callable[[_B], _A],
) -> Codec[_WireT, _B]:
    """Invariant functor map (Haskell ``Data.Functor.Contravariant.Divisible`` / ``Profunctor`` iso) on ``Codec``.

    Args:
        codec (Codec[_WireT, _A]): Base codec for ``_A``.
        to_b (Callable[[_A], _B]): Total isomorphism forward map ``_A -> _B``.
        from_b (Callable[[_B], _A]): Total isomorphism backward map ``_B -> _A``.

    Returns:
        Codec[_WireT, _B]: Derived codec for ``_B``.
    """

    def _encode_b(val: _B) -> _WireT:
        """Encode ``_B`` via ``from_b`` then ``codec.encode``.

        Args:
            val (_B): Value to encode.

        Returns:
            _WireT: Encoded wire payload.
        """
        return codec.encode(from_b(val))

    def _decode_b(wire: _WireT) -> Result[CodecError, _B]:
        """Decode ``_WireT`` via ``codec.decode`` then ``to_b``.

        Args:
            wire (_WireT): Wire payload.

        Returns:
            Result[CodecError, _B]: Decoded ``_B`` or ``CodecError``.
        """
        step = codec.decode(wire)
        if isinstance(step, Ok):
            return Ok(to_b(step.value))
        elif isinstance(step, Err):
            return step
        else:
            typing_extensions.assert_never(step)

    return Codec(encode=_encode_b, decode=_decode_b)


def refine_codec(
    codec: Codec[_WireT, _A],
    parse_b: Callable[[_A], Result[CodecError, _B]],
    unparse_b: Callable[[_B], _A],
) -> Codec[_WireT, _B]:
    """Refine a ``Codec[_WireT, _A]`` with a fallible prism ``parse_b: _A -> Result[CodecError, _B]``.

    Args:
        codec (Codec[_WireT, _A]): Underlying wire codec.
        parse_b (Callable[[_A], Result[CodecError, _B]]): Fallible validator/constructor from ``_A``.
        unparse_b (Callable[[_B], _A]): Total projection from ``_B`` back to ``_A``.

    Returns:
        Codec[_WireT, _B]: Refined codec for ``_B``.
    """

    def _encode_b(val: _B) -> _WireT:
        """Project ``val`` to ``_A`` and encode.

        Args:
            val (_B): Domain value.

        Returns:
            _WireT: Wire representation.
        """
        return codec.encode(unparse_b(val))

    def _decode_b(wire: _WireT) -> Result[CodecError, _B]:
        """Decode ``wire`` to ``_A`` then validate via ``parse_b``.

        Args:
            wire (_WireT): Wire representation.

        Returns:
            Result[CodecError, _B]: Validated ``_B`` or ``CodecError``.
        """
        step = codec.decode(wire)
        if isinstance(step, Ok):
            return parse_b(step.value)
        elif isinstance(step, Err):
            return step
        else:
            typing_extensions.assert_never(step)

    return Codec(encode=_encode_b, decode=_decode_b)


def compose_codec(
    outer: Codec[_WireT, _MidT],
    inner: Codec[_MidT, _A],
) -> Codec[_WireT, _A]:
    """Sequentially compose two codecs ``_WireT <-> _MidT <-> _A``.

    Args:
        outer (Codec[_WireT, _MidT]): Outer wire-to-intermediate codec.
        inner (Codec[_MidT, _A]): Inner intermediate-to-domain codec.

    Returns:
        Codec[_WireT, _A]: Composed end-to-end codec.
    """

    def _encode(val: _A) -> _WireT:
        """Encode ``_A`` to ``_MidT`` then ``_WireT``.

        Args:
            val (_A): Domain value.

        Returns:
            _WireT: Wire value.
        """
        return outer.encode(inner.encode(val))

    def _decode(wire: _WireT) -> Result[CodecError, _A]:
        """Decode ``_WireT`` to ``_MidT`` then ``_A``.

        Args:
            wire (_WireT): Wire value.

        Returns:
            Result[CodecError, _A]: Decoded domain value or ``CodecError``.
        """
        mid = outer.decode(wire)
        if isinstance(mid, Ok):
            return inner.decode(mid.value)
        elif isinstance(mid, Err):
            return mid
        else:
            typing_extensions.assert_never(mid)

    return Codec(encode=_encode, decode=_decode)


def list_codec(elem_codec: JsonCodec[_A]) -> JsonCodec[list[_A]]:
    """Lift an element ``JsonCodec[_A]`` to a JSON array ``JsonCodec[list[_A]]`` with index breadcrumbs.

    Args:
        elem_codec (JsonCodec[_A]): Codec for individual list elements.

    Returns:
        JsonCodec[list[_A]]: JSON array codec.
    """

    def _encode_list(items: list[_A]) -> JsonValue:
        """Encode list elements to a JSON list.

        Args:
            items (list[_A]): Domain list.

        Returns:
            JsonValue: Encoded JSON array.
        """
        return [elem_codec.encode(x) for x in items]

    def _decode_list(raw: JsonValue) -> Result[CodecError, list[_A]]:
        """Decode a JSON list, prepending ``[idx]`` to any element ``CodecError``.

        Args:
            raw (JsonValue): Candidate JSON value.

        Returns:
            Result[CodecError, list[_A]]: Decoded list or indexed ``CodecError``.
        """
        if not isinstance(raw, list):
            return Err(
                CodecError(
                    path=(),
                    expected="JSON array",
                    actual_description=_describe_wire_value(raw),
                    message="Expected a list value.",
                )
            )
        out: list[_A] = []
        for idx, item in enumerate(raw):
            step = elem_codec.decode(item)
            if isinstance(step, Ok):
                out.append(step.value)
            elif isinstance(step, Err):
                return Err(step.error.prepend_path(idx))
            else:
                typing_extensions.assert_never(step)
        return Ok(out)

    return Codec(encode=_encode_list, decode=_decode_list)


def tuple_codec(elem_codec: JsonCodec[_A]) -> JsonCodec[tuple[_A, ...]]:
    """Lift an element ``JsonCodec[_A]`` to ``JsonCodec[tuple[_A, ...]]`` encoded as a JSON array.

    Args:
        elem_codec (JsonCodec[_A]): Codec for tuple elements.

    Returns:
        JsonCodec[tuple[_A, ...]]: JSON array-backed tuple codec.
    """
    return imap_codec(list_codec(elem_codec), tuple, list)


def dict_str_codec(val_codec: JsonCodec[_A]) -> JsonCodec[dict[str, _A]]:
    """Lift a value ``JsonCodec[_A]`` to a JSON object ``JsonCodec[dict[str, _A]]`` with key breadcrumbs.

    Args:
        val_codec (JsonCodec[_A]): Codec for dictionary values.

    Returns:
        JsonCodec[dict[str, _A]]: JSON object codec.
    """

    def _encode_dict(mapping: dict[str, _A]) -> JsonValue:
        """Encode string-keyed dictionary to JSON object.

        Args:
            mapping (dict[str, _A]): Domain dictionary.

        Returns:
            JsonValue: Encoded JSON dictionary.
        """
        return {k: val_codec.encode(v) for k, v in mapping.items()}

    def _decode_dict(raw: JsonValue) -> Result[CodecError, dict[str, _A]]:
        """Decode JSON object, prepending key names to nested ``CodecError`` paths.

        Args:
            raw (JsonValue): Candidate JSON value.

        Returns:
            Result[CodecError, dict[str, _A]]: Decoded dictionary or path-annotated ``CodecError``.
        """
        if not isinstance(raw, Mapping):
            return Err(
                CodecError(
                    path=(),
                    expected="JSON object",
                    actual_description=_describe_wire_value(raw),
                    message="Expected a mapping object.",
                )
            )
        out: dict[str, _A] = {}
        for key, item in raw.items():
            step = val_codec.decode(item)
            if isinstance(step, Ok):
                out[key] = step.value
            elif isinstance(step, Err):
                return Err(step.error.prepend_path(key))
            else:
                typing_extensions.assert_never(step)
        return Ok(out)

    return Codec(encode=_encode_dict, decode=_decode_dict)


def optional_codec(inner_codec: JsonCodec[_A]) -> JsonCodec[_A | None]:
    """Lift ``JsonCodec[_A]`` to ``JsonCodec[_A | None]`` mapping ``None`` <-> JSON ``null``.

    Args:
        inner_codec (JsonCodec[_A]): Non-null inner codec.

    Returns:
        JsonCodec[_A | None]: Nullable JSON codec.
    """

    def _encode_opt(val: _A | None) -> JsonValue:
        """Encode ``None`` as ``None`` else delegate to ``inner_codec``.

        Args:
            val (_A | None): Optional domain value.

        Returns:
            JsonValue: JSON value or ``None``.
        """
        if val is None:
            return None
        return inner_codec.encode(val)

    def _decode_opt(raw: JsonValue) -> Result[CodecError, _A | None]:
        """Decode ``None`` as ``Ok(None)`` else delegate to ``inner_codec``.

        Args:
            raw (JsonValue): Candidate JSON value.

        Returns:
            Result[CodecError, _A | None]: Decoded optional value or ``CodecError``.
        """
        if raw is None:
            return Ok(None)
        step = inner_codec.decode(raw)
        if isinstance(step, Ok):
            return Ok(step.value)
        elif isinstance(step, Err):
            return step
        else:
            typing_extensions.assert_never(step)

    return Codec(encode=_encode_opt, decode=_decode_opt)


@dataclass(frozen=True)
class Pretty(typing.Generic[_A]):
    """Typeclass witness dictionary for deterministic human-readable text rendering (Haskell ``Pretty``).

    Attributes:
        render (Callable[[_A], str]): Deterministic string formatting function.
    """

    render: Callable[[_A], str]


def nonempty_pretty(elem_pretty: Pretty[_A], separator: str = "; ") -> Pretty[NonEmpty[_A]]:
    """Lift an element ``Pretty[_A]`` witness to ``Pretty[NonEmpty[_A]]``.

    Args:
        elem_pretty (Pretty[_A]): Pretty-printer witness for individual elements.
        separator (str): Delimiter inserted between rendered elements.

    Returns:
        Pretty[NonEmpty[_A]]: Deterministic pretty-printer over ``NonEmpty[_A]``.
    """

    def _render_ne(items: NonEmpty[_A]) -> str:
        """Render every element of ``items`` joined by ``separator``.

        Args:
            items (NonEmpty[_A]): Non-empty sequence to format.

        Returns:
            str: Joined human-readable string.
        """
        return separator.join(elem_pretty.render(x) for x in items)

    return Pretty(render=_render_ne)


@dataclass(frozen=True)
class PythonExprEmit(typing.Generic[_A]):
    """Typeclass witness dictionary emitting valid Format-B Python constructor source expressions.

    Attributes:
        emit_python (Callable[[_A, str], str]): Function ``(value, module_prefix) -> python_expr_str``.
    """

    emit_python: Callable[[_A, str], str]
