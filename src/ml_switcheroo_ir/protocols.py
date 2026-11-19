"""Tier-1 structural protocols for zero-dependency compiler interoperability.

Defines ``@typing.runtime_checkable`` structural ``typing.Protocol`` classes so
downstream compiler and ecosystem types (dtypes, devices, symbolic dimensions,
custom rule handles, loop options, device meshes, eager array buffers, and
subgraph containers) interoperate cleanly without upward imports or ``Any``.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence

import typing_extensions

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir.graph import LogicalNode

__all__ = [
    "ArrayBufferProtocol",
    "CustomRuleHandleProtocol",
    "DTypeLikeProtocol",
    "DeviceLikeProtocol",
    "LoopOptionsProtocol",
    "MeshLikeProtocol",
    "SerializationInfoProtocol",
    "SubgraphLikeProtocol",
    "SymbolicDimProtocol",
    "SymbolicIntWrapperProtocol",
    "SymbolicNodeProtocol",
]


class SerializationInfoProtocol(typing.Protocol):
    """Structural protocol for Pydantic v2 field/plain serializer ``info`` context."""

    @property
    def mode(self) -> str:
        """Return active serialization mode (``'python'`` or ``'json'``).

        Returns:
            str: Active serialization mode name.
        """


@typing.runtime_checkable
class DTypeLikeProtocol(typing.Protocol):
    """Structural protocol for external Enum-based or custom dtype objects."""

    @property
    def value(self) -> str:
        """Return canonical string value of the data type.

        Returns:
            str: Canonical string representation (e.g. ``'float32'``).
        """

    @property
    def name(self) -> str:
        """Return symbolic member name of the data type.

        Returns:
            str: Member identifier (e.g. ``'FLOAT32'`` or ``'float32'``).
        """


@typing.runtime_checkable
class DeviceLikeProtocol(typing.Protocol):
    """Structural protocol for external hardware device descriptors."""

    @property
    def type(self) -> str:
        """Return hardware backend or device type string.

        Returns:
            str: Device backend identifier (e.g. ``'cuda'``, ``'cpu'``, ``'tpu'``).
        """

    @property
    def index(self) -> int | None:
        """Return optional zero-based device ordinal index.

        Returns:
            int | None: Device ordinal index or ``None``.
        """


@typing.runtime_checkable
class SymbolicNodeProtocol(typing.Protocol):
    """Structural protocol for external symbolic dimension expression AST nodes."""

    def eval(self, env: dict[str, int]) -> int:
        """Evaluate symbolic expression node under concrete variable bindings.

        Args:
            env (dict[str, int]): Mapping from variable names to concrete integer sizes.

        Returns:
            int: Evaluated concrete integer dimension size.
        """


@typing.runtime_checkable
class SymbolicIntWrapperProtocol(typing.Protocol):
    """Structural protocol for symbolic integer wrappers exposing an underlying ``node``."""

    @property
    def node(self) -> SymbolicNodeProtocol:
        """Return underlying symbolic expression AST node.

        Returns:
            SymbolicNodeProtocol: Underlying symbolic expression node.
        """


SymbolicDimProtocol: typing_extensions.TypeAlias = typing.Union[
    SymbolicNodeProtocol,
    SymbolicIntWrapperProtocol,
]


@typing.runtime_checkable
class CustomRuleHandleProtocol(typing.Protocol):
    """Structural protocol for serializable custom lowering or autodiff rule handles."""

    @property
    def handle_id(self) -> str:
        """Return unique callback registry identifier for the custom rule.

        Returns:
            str: Unique handle identifier.
        """

    @property
    def kind(self) -> str:
        """Return category or dispatch kind of the custom rule.

        Returns:
            str: Rule dispatch kind identifier.
        """


@typing.runtime_checkable
class LoopOptionsProtocol(typing.Protocol):
    """Structural protocol for compiler control-flow loop configuration options."""

    @property
    def parallel_iterations(self) -> int | None:
        """Return maximum number of loop iterations allowed to run in parallel.

        Returns:
            int | None: Parallel iteration limit or ``None``.
        """

    @property
    def swap_memory(self) -> bool | None:
        """Return whether device-to-host activation memory swapping is enabled.

        Returns:
            bool | None: Memory swapping flag or ``None``.
        """

    @property
    def maximum_iterations(self) -> int | None:
        """Return optional upper bound on loop trip count.

        Returns:
            int | None: Maximum iteration count or ``None``.
        """


@typing.runtime_checkable
class MeshLikeProtocol(typing.Protocol):
    """Structural protocol for external distributed SPMD device mesh objects."""

    @property
    def axis_names(self) -> Sequence[str]:
        """Return ordered mesh dimension axis names.

        Returns:
            Sequence[str]: Ordered sequence of mesh axis names.
        """


@typing.runtime_checkable
class ArrayBufferProtocol(typing.Protocol):
    """Structural protocol for eager/host tensor buffers without upward numpy imports."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return concrete integer dimension tuple of the buffer.

        Returns:
            tuple[int, ...]: Buffer shape tuple.
        """

    def tobytes(self) -> bytes:
        """Serialize raw contiguous buffer contents into bytes.

        Returns:
            bytes: Raw byte buffer payload.
        """


@typing.runtime_checkable
class SubgraphLikeProtocol(typing.Protocol):
    """Structural protocol for nested subgraph containers."""

    @property
    def nodes(self) -> Mapping[str, LogicalNode] | Sequence[LogicalNode]:
        """Return mapping or sequence view of contained ``LogicalNode`` instances.

        Returns:
            Mapping[str, LogicalNode] | Sequence[LogicalNode]: Contained graph nodes.
        """
