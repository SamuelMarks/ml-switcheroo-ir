"""Zero-cost NewType domain identifiers, refined integer scalars, and multi-output PortRef."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from ml_switcheroo_ir.typeclasses.algebra import (
    INT_HASH,
    INT_ORD,
    STR_HASH,
    STR_ORD,
    Eq,
    HashableWitness,
    Ord,
    Ordering,
    contramap_hash,
    contramap_ord,
    lexical_pair_ord,
    pair_hash_witness,
)
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

_CandidateT = typing.TypeVar("_CandidateT")

NodeId = typing.NewType("NodeId", str)
SsaValueName = typing.NewType("SsaValueName", str)
DialectDomain = typing.NewType("DialectDomain", str)
OpName = typing.NewType("OpName", str)
MeshAxisName = typing.NewType("MeshAxisName", str)

NonBoolInt = typing.NewType("NonBoolInt", int)
NonNegativeInt = typing.NewType("NonNegativeInt", NonBoolInt)
PositiveInt = typing.NewType("PositiveInt", NonBoolInt)

__all__ = [
    "DIALECT_DOMAIN_EQ",
    "DIALECT_DOMAIN_HASH",
    "DIALECT_DOMAIN_ORD",
    "DialectDomain",
    "MESH_AXIS_NAME_EQ",
    "MESH_AXIS_NAME_HASH",
    "MESH_AXIS_NAME_ORD",
    "MeshAxisName",
    "NODE_ID_EQ",
    "NODE_ID_HASH",
    "NODE_ID_ORD",
    "NON_BOOL_INT_EQ",
    "NON_BOOL_INT_HASH",
    "NON_BOOL_INT_ORD",
    "NodeId",
    "NonBoolInt",
    "NonNegativeInt",
    "OP_NAME_EQ",
    "OP_NAME_HASH",
    "OP_NAME_ORD",
    "OpName",
    "PORT_REF_EQ",
    "PORT_REF_HASH",
    "PORT_REF_ORD",
    "PortRef",
    "PositiveInt",
    "QUALIFIED_OP_KEY_EQ",
    "QUALIFIED_OP_KEY_HASH",
    "QUALIFIED_OP_KEY_ORD",
    "QualifiedOpKey",
    "SSA_VALUE_NAME_EQ",
    "SSA_VALUE_NAME_HASH",
    "SSA_VALUE_NAME_ORD",
    "SsaValueName",
    "is_non_bool_int",
    "parse_dialect_domain",
    "parse_mesh_axis_name",
    "parse_node_id",
    "parse_non_negative_int",
    "parse_op_name",
    "parse_positive_int",
    "parse_ssa_name",
    "require_non_bool_int",
]


def is_non_bool_int(val: _CandidateT) -> typing.TypeGuard[NonBoolInt]:
    """Narrow ``val`` to ``NonBoolInt`` iff ``val`` is an ``int`` and NOT a ``bool``.

    Eliminates Python's ``bool <: int`` subtyping footgun with a single reusable ``TypeGuard``.

    Args:
        val (_CandidateT): Candidate runtime value of arbitrary static type.

    Returns:
        typing.TypeGuard[NonBoolInt]: True iff ``isinstance(val, int) and not isinstance(val, bool)``.
    """
    return isinstance(val, int) and not isinstance(val, bool)


def require_non_bool_int(val: int, label: str = "value") -> NonBoolInt:
    """Validate that ``val`` is a genuine ``int`` (not ``bool``) and return ``NonBoolInt``.

    Args:
        val (int): Candidate integer value.
        label (str): Field or parameter name for diagnostic messages.

    Returns:
        NonBoolInt: Validated non-boolean integer.

    Raises:
        TypeError: If ``val`` is a boolean or not an ``int``.
    """
    if not is_non_bool_int(val):
        raise TypeError(f"{label} must be a non-boolean int, got {val!r}.")
    return val


def parse_non_negative_int(val: int) -> Result[str, NonNegativeInt]:
    """Total smart constructor validating ``val >= 0`` and ``not isinstance(val, bool)``.

    Args:
        val (int): Candidate integer.

    Returns:
        Result[str, NonNegativeInt]: ``Ok(NonNegativeInt)`` or ``Err`` diagnostic message.
    """
    if not is_non_bool_int(val):
        return Err(f"Expected non-boolean int, got {val!r}.")
    if val < 0:
        return Err(f"Expected non-negative int (>= 0), got {val}.")
    return Ok(NonNegativeInt(val))


def parse_positive_int(val: int) -> Result[str, PositiveInt]:
    """Total smart constructor validating ``val > 0`` and ``not isinstance(val, bool)``.

    Args:
        val (int): Candidate integer.

    Returns:
        Result[str, PositiveInt]: ``Ok(PositiveInt)`` or ``Err`` diagnostic message.
    """
    if not is_non_bool_int(val):
        return Err(f"Expected non-boolean int, got {val!r}.")
    if val <= 0:
        return Err(f"Expected positive int (> 0), got {val}.")
    return Ok(PositiveInt(val))


def parse_node_id(raw: str) -> Result[str, NodeId]:
    """Total smart constructor validating a non-empty ``NodeId``.

    Args:
        raw (str): Candidate node identifier string.

    Returns:
        Result[str, NodeId]: ``Ok(NodeId(raw))`` or ``Err`` if empty.
    """
    if not raw:
        return Err("NodeId must be a non-empty string.")
    return Ok(NodeId(raw))


def parse_ssa_name(raw: str) -> Result[str, SsaValueName]:
    """Total smart constructor validating a non-empty ``SsaValueName``.

    Args:
        raw (str): Candidate SSA value identifier string.

    Returns:
        Result[str, SsaValueName]: ``Ok(SsaValueName(raw))`` or ``Err`` if empty.
    """
    if not raw:
        return Err("SsaValueName must be a non-empty string.")
    return Ok(SsaValueName(raw))


def parse_dialect_domain(raw: str) -> Result[str, DialectDomain]:
    """Total smart constructor wrapping a dialect domain namespace string (empty allowed for default domain).

    Args:
        raw (str): Raw domain identifier string.

    Returns:
        Result[str, DialectDomain]: ``Ok(DialectDomain(raw))`` or ``Err`` if whitespace-only non-empty.
    """
    if raw and not raw.strip():
        return Err("DialectDomain must not be whitespace-only.")
    return Ok(DialectDomain(raw))


def parse_op_name(raw: str) -> Result[str, OpName]:
    """Total smart constructor validating a non-empty operator name ``OpName``.

    Args:
        raw (str): Candidate operator name.

    Returns:
        Result[str, OpName]: ``Ok(OpName(raw))`` or ``Err`` if empty.
    """
    if not raw:
        return Err("OpName must be a non-empty string.")
    return Ok(OpName(raw))


def parse_mesh_axis_name(raw: str) -> Result[str, MeshAxisName]:
    """Total smart constructor validating a non-empty ``MeshAxisName``.

    Args:
        raw (str): Candidate mesh axis name.

    Returns:
        Result[str, MeshAxisName]: ``Ok(MeshAxisName(raw))`` or ``Err`` if empty.
    """
    if not raw:
        return Err("MeshAxisName must be a non-empty string.")
    return Ok(MeshAxisName(raw))


@dataclass(frozen=True)
class QualifiedOpKey:
    """Structured ``(domain, op_name)`` identifier replacing ad-hoc string concatenation.

    Attributes:
        domain (DialectDomain): Dialect namespace (empty string for default/core domain).
        op_name (OpName): Unqualified operator name.
    """

    domain: DialectDomain
    op_name: OpName

    def qualified_str(self) -> str:
        """Render ``'domain.op_name'`` when ``domain`` is non-empty, else ``'op_name'``.

        Returns:
            str: Canonical qualified operator identifier.
        """
        if self.domain:
            return f"{self.domain}.{self.op_name}"
        return str(self.op_name)


@dataclass(frozen=True)
class PortRef:
    """Structured SSA/multi-output port reference replacing ad-hoc ``'node:idx'`` string splitting.

    Attributes:
        producer (NodeId | SsaValueName): Producing node identifier or standalone SSA value name.
        port_index (NonNegativeInt | None): Zero-based output port index when indexed, else ``None``.
    """

    producer: NodeId | SsaValueName
    port_index: NonNegativeInt | None = None

    @classmethod
    def parse(cls, raw: str) -> PortRef:
        """Total parser converting ``'producer:idx'`` (non-negative digits) or ``'ssa_name'`` into ``PortRef``.

        Args:
            raw (str): Raw input edge reference string.

        Returns:
            PortRef: Structured port reference.
        """
        if ":" in raw:
            head, tail = raw.rsplit(":", 1)
            if head and tail.isdigit():
                return cls(
                    producer=NodeId(head),
                    port_index=NonNegativeInt(NonBoolInt(int(tail))),
                )
        return cls(producer=SsaValueName(raw), port_index=None)

    @classmethod
    def base_identifier(cls, raw: str) -> str:
        """Extract the producer/base prefix before ``':'`` if present, else ``raw`` unchanged.

        Args:
            raw (str): Raw SSA or ``'node:port'`` reference string.

        Returns:
            str: Base producer identifier string.
        """
        parsed = cls.parse(raw)
        if parsed.port_index is not None:
            return str(parsed.producer)
        return raw.rpartition(":")[0] if ":" in raw else raw

    def render(self) -> str:
        """Format back to canonical ``'producer:port_index'`` or ``'producer'`` string.

        Returns:
            str: Canonical edge reference string.
        """
        if self.port_index is not None:
            return f"{self.producer}:{int(self.port_index)}"
        return str(self.producer)


NODE_ID_ORD: typing.Final[Ord[NodeId]] = contramap_ord(STR_ORD, str)
NODE_ID_EQ: typing.Final[Eq[NodeId]] = NODE_ID_ORD.eq_witness
NODE_ID_HASH: typing.Final[HashableWitness[NodeId]] = contramap_hash(STR_HASH, str)

SSA_VALUE_NAME_ORD: typing.Final[Ord[SsaValueName]] = contramap_ord(STR_ORD, str)
SSA_VALUE_NAME_EQ: typing.Final[Eq[SsaValueName]] = SSA_VALUE_NAME_ORD.eq_witness
SSA_VALUE_NAME_HASH: typing.Final[HashableWitness[SsaValueName]] = contramap_hash(STR_HASH, str)

DIALECT_DOMAIN_ORD: typing.Final[Ord[DialectDomain]] = contramap_ord(STR_ORD, str)
DIALECT_DOMAIN_EQ: typing.Final[Eq[DialectDomain]] = DIALECT_DOMAIN_ORD.eq_witness
DIALECT_DOMAIN_HASH: typing.Final[HashableWitness[DialectDomain]] = contramap_hash(STR_HASH, str)

OP_NAME_ORD: typing.Final[Ord[OpName]] = contramap_ord(STR_ORD, str)
OP_NAME_EQ: typing.Final[Eq[OpName]] = OP_NAME_ORD.eq_witness
OP_NAME_HASH: typing.Final[HashableWitness[OpName]] = contramap_hash(STR_HASH, str)

MESH_AXIS_NAME_ORD: typing.Final[Ord[MeshAxisName]] = contramap_ord(STR_ORD, str)
MESH_AXIS_NAME_EQ: typing.Final[Eq[MeshAxisName]] = MESH_AXIS_NAME_ORD.eq_witness
MESH_AXIS_NAME_HASH: typing.Final[HashableWitness[MeshAxisName]] = contramap_hash(STR_HASH, str)

NON_BOOL_INT_ORD: typing.Final[Ord[NonBoolInt]] = contramap_ord(INT_ORD, int)
NON_BOOL_INT_EQ: typing.Final[Eq[NonBoolInt]] = NON_BOOL_INT_ORD.eq_witness
NON_BOOL_INT_HASH: typing.Final[HashableWitness[NonBoolInt]] = contramap_hash(INT_HASH, int)


def _qualified_key_tuple(key: QualifiedOpKey) -> tuple[str, str]:
    """Project ``QualifiedOpKey`` to ``(domain, op_name)`` tuple for witnesses.

    Args:
        key (QualifiedOpKey): Qualified operator key.

    Returns:
        tuple[str, str]: Pair of ``(str(key.domain), str(key.op_name))``.
    """
    return (str(key.domain), str(key.op_name))


QUALIFIED_OP_KEY_ORD: typing.Final[Ord[QualifiedOpKey]] = contramap_ord(
    lexical_pair_ord(STR_ORD, STR_ORD),
    _qualified_key_tuple,
)
QUALIFIED_OP_KEY_EQ: typing.Final[Eq[QualifiedOpKey]] = QUALIFIED_OP_KEY_ORD.eq_witness
QUALIFIED_OP_KEY_HASH: typing.Final[HashableWitness[QualifiedOpKey]] = contramap_hash(
    pair_hash_witness(STR_HASH, STR_HASH),
    _qualified_key_tuple,
)


def _port_ref_tuple(ref: PortRef) -> tuple[str, int]:
    """Project ``PortRef`` to ``(producer_str, port_index_or_neg1)`` for ordering/hashing.

    Args:
        ref (PortRef): Port reference.

    Returns:
        tuple[str, int]: Tuple with ``-1`` representing ``port_index is None``.
    """
    idx = -1 if ref.port_index is None else int(ref.port_index)
    return (str(ref.producer), idx)


def _port_ref_cmp(lhs: PortRef, rhs: PortRef) -> Ordering:
    """Compare two ``PortRef`` values deterministically.

    Args:
        lhs (PortRef): Left port reference.
        rhs (PortRef): Right port reference.

    Returns:
        Ordering: Lexicographic ordering outcome.
    """
    return lexical_pair_ord(STR_ORD, INT_ORD).compare(_port_ref_tuple(lhs), _port_ref_tuple(rhs))


PORT_REF_EQ: typing.Final[Eq[PortRef]] = contramap_ord(
    lexical_pair_ord(STR_ORD, INT_ORD),
    _port_ref_tuple,
).eq_witness
PORT_REF_ORD: typing.Final[Ord[PortRef]] = Ord(eq_witness=PORT_REF_EQ, compare=_port_ref_cmp)
PORT_REF_HASH: typing.Final[HashableWitness[PortRef]] = contramap_hash(
    pair_hash_witness(STR_HASH, INT_HASH),
    _port_ref_tuple,
)
