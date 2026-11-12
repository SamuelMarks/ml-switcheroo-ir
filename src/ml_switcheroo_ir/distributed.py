"""Distributed pipeline topologies, WebRTC mesh signaling schemas, and communication models."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence

import pydantic
import typing_extensions

from ml_switcheroo_ir.types import JsonValue

PartitionAxisAtom: typing_extensions.TypeAlias = typing.Union[str, int, None]
PartitionAxisInner: typing_extensions.TypeAlias = typing.Union[
    PartitionAxisAtom, tuple[PartitionAxisAtom, ...], list[PartitionAxisAtom]
]
PartitionAxisElement: typing_extensions.TypeAlias = typing.Union[
    PartitionAxisInner, tuple[PartitionAxisInner, ...], list[PartitionAxisInner]
]


def _validate_partition_atom(atom: PartitionAxisAtom | JsonValue) -> PartitionAxisAtom:
    """Validate an atomic partition axis entry (``str | int | None``, rejecting ``bool``).

    Args:
        atom (PartitionAxisAtom | JsonValue): Candidate atomic axis value.

    Returns:
        PartitionAxisAtom: Validated atomic axis value.

    Raises:
        ValueError: If ``atom`` is a boolean or unsupported type.
    """
    if isinstance(atom, bool) or not (atom is None or isinstance(atom, (str, int))):
        raise ValueError(f"Invalid PartitionSpec axis atom: {atom!r}; expected str, int, or None.")
    return atom


def _normalize_partition_element(elem: PartitionAxisElement | JsonValue) -> PartitionAxisElement:
    """Validate and normalize a single ``PartitionSpec`` dimension element into tuples/atoms.

    Args:
        elem (PartitionAxisElement | JsonValue): Candidate dimension axis element.

    Returns:
        PartitionAxisElement: Normalized axis element with nested lists converted to non-empty tuples.

    Raises:
        ValueError: If ``elem`` is an empty sub-sequence, boolean, or unsupported container.
    """
    if isinstance(elem, (list, tuple)):
        if not elem:
            raise ValueError("PartitionSpec nested axis tuple/list must not be empty.")
        normalized_items: list[PartitionAxisInner] = []
        for sub in elem:
            if isinstance(sub, (list, tuple)):
                if not sub:
                    raise ValueError("PartitionSpec nested sub-axis tuple/list must not be empty.")
                normalized_items.append(tuple(_validate_partition_atom(a) for a in sub))
            else:
                normalized_items.append(_validate_partition_atom(sub))
        return tuple(normalized_items)
    return _validate_partition_atom(elem)


class LogicalAxis(pydantic.BaseModel):
    """Represents a named dimension for tensor sizes and sharding (e.g., 'batch', 'embed', 'heads').

    Attributes:
        name (str): Name of the logical axis.
        size (int | None): Optional fixed size of the axis.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    size: int | None = None

    def __init__(self, name: str, size: int | None = None) -> None:
        """Initialize LogicalAxis.

        Args:
            name (str): Name of the logical axis.
            size (int | None): Optional fixed size of the axis.
        """
        super().__init__(name=name, size=size)


class PartitionSpec(pydantic.BaseModel):
    """Describes how a tensor's dimensions are mapped to a logical mesh.

    Each element in `axes` corresponds to a tensor dimension. An element can be:
    - A string or integer representing the mesh axis identifier (e.g., 'data').
    - A non-empty tuple of identifiers for multi-axis sharding (e.g., ('data', 'model')).
    - None for a replicated/unsharded dimension.

    Attributes:
        axes (tuple[PartitionAxisElement, ...]): Tuple mapping tensor dimensions to mesh axes.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    axes: tuple[PartitionAxisElement, ...] = ()

    @pydantic.field_validator("axes", mode="before")
    @classmethod
    def _validate_axes_elements(
        cls,
        raw_axes: Sequence[PartitionAxisElement],
    ) -> tuple[PartitionAxisElement, ...]:
        """Validate and normalize PartitionSpec axis elements.

        Args:
            raw_axes (Sequence[PartitionAxisElement]): Raw axis sequence.

        Returns:
            tuple[PartitionAxisElement, ...]: Normalized tuple of axis elements.
        """
        return tuple(_normalize_partition_element(el) for el in raw_axes)

    def __init__(
        self,
        axes: Sequence[PartitionAxisElement] = (),
    ) -> None:
        """Initialize PartitionSpec normalizing axes into a tuple.

        Args:
            axes (Sequence[PartitionAxisElement]): Sequence mapping tensor dimensions to mesh axes.

        Raises:
            ValueError: If ``axes`` is a string, bytes, or non-sequence.
        """
        if isinstance(axes, (str, bytes)) or not isinstance(axes, Sequence):
            raise ValueError(f"PartitionSpec axes must be a sequence, got {axes!r}.")
        super().__init__(axes=tuple(axes))


class LogicalMesh(pydantic.BaseModel):
    """Represents a multi-dimensional grid of devices for distributed execution.

    Attributes:
        shape (dict[str, int]): Mapping of mesh axis names to positive sizes (e.g., ``{'data': 4}``).
        webrtc_topology (WebRTCSignalingTopology | None): WebRTC cluster mesh signaling configuration.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    shape: dict[str, int]
    webrtc_topology: WebRTCSignalingTopology | None = None

    @pydantic.field_validator("shape", mode="before")
    @classmethod
    def _validate_mesh_shape(cls, raw_shape: Mapping[str, int] | JsonValue) -> dict[str, int]:
        """Validate that mesh shape is non-empty and all axis sizes are strictly positive integers.

        Args:
            raw_shape (Mapping[str, int] | JsonValue): Raw mapping of axis names to sizes.

        Returns:
            dict[str, int]: Validated mesh shape dictionary.

        Raises:
            ValueError: If ``raw_shape`` is empty or any axis name/size is invalid.
        """
        if not isinstance(raw_shape, Mapping) or not raw_shape:
            raise ValueError("LogicalMesh 'shape' must be a non-empty mapping of axis names to positive sizes.")
        validated: dict[str, int] = {}
        for axis_name, size in raw_shape.items():
            if not axis_name or isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"LogicalMesh axis {axis_name!r} must have positive integer size >= 1, got {size!r}.")
            validated[str(axis_name)] = size
        return validated

    def __init__(
        self,
        shape: Mapping[str, int],
        webrtc_topology: WebRTCSignalingTopology | None = None,
    ) -> None:
        """Initialize LogicalMesh.

        Args:
            shape (Mapping[str, int]): Mapping of mesh axis names to their sizes.
            webrtc_topology (WebRTCSignalingTopology | None): Optional WebRTC cluster mesh signaling configuration.
        """
        super().__init__(shape=dict(shape), webrtc_topology=webrtc_topology)


class MicrobatchSplittingConfig(pydantic.BaseModel):
    """Configuration for microbatch splitting strategy and count.

    Attributes:
        strategy (str): Microbatch splitting strategy (e.g. 'uniform', 'adaptive').
        num_microbatches (int): Total number of microbatches (>= 1).
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    strategy: str = "uniform"
    num_microbatches: int = 1


class MeshMappingConfig(pydantic.BaseModel):
    """Configuration for mapping pipeline stages to accelerator devices.

    Attributes:
        devices_per_stage (int): Number of devices assigned to each pipeline stage (>= 1).
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    devices_per_stage: int = 1


class StageCommunicationConfig(pydantic.BaseModel):
    """Configuration for inter-stage communication protocol.

    Attributes:
        protocol (str): Transport protocol ('p2p', 'webrtc', 'nccl', 'gloo').
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    protocol: str = "p2p"


class DependencyConfig(pydantic.BaseModel):
    """Configuration for synchronization dependencies between pipeline stages.

    Attributes:
        source_stage (str): Producer stage identifier.
        target_stage (str): Consumer stage identifier.
        offset_mb (int): Microbatch offset index.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    source_stage: str
    target_stage: str
    offset_mb: int = 0


class SchedulePhaseConfig(pydantic.BaseModel):
    """Configuration for an execution phase in a pipeline schedule.

    Attributes:
        type (str): Schedule phase type (e.g. 'warmup', '1F1B', 'cooldown').
        operations (list[str]): Ordered operations executed in this phase.
        count_expression (str): Expression resolving the execution iteration count.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    type: str = "1F1B"
    operations: list[str] = pydantic.Field(default_factory=list)
    count_expression: str = "1"


class PipelineScheduleConfig(pydantic.BaseModel):
    """Configuration for ordered phases in a pipeline execution schedule.

    Attributes:
        phases (list[SchedulePhaseConfig]): Ordered schedule phases.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    phases: list[SchedulePhaseConfig] = pydantic.Field(default_factory=list)


class PipelineTopologyConfig(pydantic.BaseModel):
    """Comprehensive specification for a distributed pipeline topology.

    Attributes:
        microbatch_splitting (MicrobatchSplittingConfig): Microbatch splitting configuration.
        mesh_mapping (MeshMappingConfig): Stage device mapping configuration.
        stage_communication (StageCommunicationConfig): Communication protocol.
        dependencies (list[DependencyConfig]): Inter-stage synchronization dependencies.
        schedule (PipelineScheduleConfig | None): Optional pipeline schedule configuration.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    microbatch_splitting: MicrobatchSplittingConfig
    mesh_mapping: MeshMappingConfig
    stage_communication: StageCommunicationConfig
    dependencies: list[DependencyConfig] = pydantic.Field(default_factory=list)
    schedule: PipelineScheduleConfig | None = None

    @pydantic.model_validator(mode="after")
    def _validate_pipeline_invariants(self) -> PipelineTopologyConfig:
        """Validate positive microbatch/device counts and non-empty stage identifiers.

        Returns:
            PipelineTopologyConfig: Validated pipeline topology instance.

        Raises:
            ValueError: If microbatch count, devices_per_stage, or stage identifiers are invalid.
        """
        if self.microbatch_splitting.num_microbatches < 1:
            raise ValueError(f"num_microbatches must be >= 1, got {self.microbatch_splitting.num_microbatches}.")
        if self.mesh_mapping.devices_per_stage < 1:
            raise ValueError(f"devices_per_stage must be >= 1, got {self.mesh_mapping.devices_per_stage}.")
        for dep in self.dependencies:
            if not dep.source_stage.strip() or not dep.target_stage.strip():
                raise ValueError("DependencyConfig source_stage and target_stage must be non-empty strings.")
        return self

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert pipeline topology configuration to dictionary representation.

        Returns:
            dict[str, JsonValue]: JSON-serializable dictionary representation.
        """
        schedule_dict: JsonValue = (
            {
                "phases": [
                    {
                        "type": phase.type,
                        "operations": list(phase.operations),
                        "count_expression": phase.count_expression,
                    }
                    for phase in self.schedule.phases
                ]
            }
            if self.schedule is not None
            else None
        )
        return {
            "microbatch_splitting": {
                "strategy": self.microbatch_splitting.strategy,
                "num_microbatches": self.microbatch_splitting.num_microbatches,
            },
            "mesh_mapping": {
                "devices_per_stage": self.mesh_mapping.devices_per_stage,
            },
            "stage_communication": {
                "protocol": self.stage_communication.protocol,
            },
            "dependencies": [
                {
                    "source_stage": dep.source_stage,
                    "target_stage": dep.target_stage,
                    "offset_mb": dep.offset_mb,
                }
                for dep in self.dependencies
            ],
            "schedule": schedule_dict,
        }

    @classmethod
    def from_dict(cls, data: PipelineTopologyConfig | Mapping[str, JsonValue]) -> PipelineTopologyConfig:
        """Construct PipelineTopologyConfig from dictionary representation.

        Args:
            data (PipelineTopologyConfig | Mapping[str, JsonValue]): Dictionary containing topology configuration.

        Returns:
            PipelineTopologyConfig: Parsed configuration instance.
        """
        return cls.model_validate(data)


class WebRTCPeerConfig(pydantic.BaseModel):
    """Configuration for a WebRTC mesh participant peer.

    Attributes:
        peer_id (str): Unique peer identifier.
        device_capability (str): Accelerator capability (e.g. 'webgpu', 'wasm', 'cpu').
        network_transport (str): Transport channel type (e.g. 'data_channel').
        ice_transport_policy (str): ICE transport policy ('all', 'relay').
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    peer_id: str
    device_capability: str = "webgpu"
    network_transport: str = "data_channel"
    ice_transport_policy: str = "all"


class WebRTCSignalingTopology(pydantic.BaseModel):
    """Configuration for WebRTC mesh coordination, signaling endpoints, and buffer watermarks.

    Attributes:
        signaling_url (str): WebSocket or HTTP signaling server endpoint.
        ice_servers (list[str]): STUN/TURN server URLs.
        high_watermark_bytes (int): Buffer high-water mark for backpressure management.
        low_watermark_bytes (int): Buffer low-water mark for resumed transmission.
        chunk_size_bytes (int): Maximum chunk payload size for large tensor transfers.
        connectivity_matrix (list[list[int]]): Peer connectivity adjacency matrix.
        peers (list[WebRTCPeerConfig]): List of participating peer specifications.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    signaling_url: str = "ws://localhost:8080"
    ice_servers: list[str] = pydantic.Field(default_factory=lambda: ["stun:stun.l.google.com:19302"])
    high_watermark_bytes: int = 1048576
    low_watermark_bytes: int = 262144
    chunk_size_bytes: int = 65536
    connectivity_matrix: list[list[int]] = pydantic.Field(default_factory=list)
    peers: list[WebRTCPeerConfig] = pydantic.Field(default_factory=list)

    @pydantic.model_validator(mode="after")
    def _validate_webrtc_invariants(self) -> WebRTCSignalingTopology:
        """Validate chunk size, buffer watermarks, unique peer IDs, and adjacency matrix dimensions.

        Returns:
            WebRTCSignalingTopology: Validated WebRTC signaling topology instance.

        Raises:
            ValueError: If buffer watermarks, chunk size, peer IDs, or connectivity matrix are invalid.
        """
        if self.chunk_size_bytes < 1:
            raise ValueError(f"chunk_size_bytes must be >= 1, got {self.chunk_size_bytes}.")
        if self.low_watermark_bytes < 0 or self.low_watermark_bytes > self.high_watermark_bytes:
            raise ValueError(
                f"Watermarks must satisfy 0 <= low_watermark_bytes ({self.low_watermark_bytes}) "
                f"<= high_watermark_bytes ({self.high_watermark_bytes})."
            )
        peer_ids = [p.peer_id for p in self.peers]
        if len(set(peer_ids)) != len(peer_ids):
            raise ValueError(f"WebRTCSignalingTopology peers must have unique peer_id values, got {peer_ids}.")
        if self.connectivity_matrix:
            expected_n = len(self.peers)
            if len(self.connectivity_matrix) != expected_n or any(
                len(row) != expected_n for row in self.connectivity_matrix
            ):
                raise ValueError(
                    f"connectivity_matrix must have dimensions {expected_n}x{expected_n} matching len(peers)."
                )
            for row in self.connectivity_matrix:
                if any(isinstance(cell, bool) or cell not in (0, 1) for cell in row):
                    raise ValueError("connectivity_matrix entries must be binary integers 0 or 1.")
        return self

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert WebRTC signaling topology to dictionary representation.

        Returns:
            dict[str, JsonValue]: JSON-serializable dictionary representation.
        """
        return {
            "signaling_url": self.signaling_url,
            "ice_servers": list(self.ice_servers),
            "high_watermark_bytes": self.high_watermark_bytes,
            "low_watermark_bytes": self.low_watermark_bytes,
            "chunk_size_bytes": self.chunk_size_bytes,
            "connectivity_matrix": [list(row) for row in self.connectivity_matrix],
            "peers": [
                {
                    "peer_id": peer.peer_id,
                    "device_capability": peer.device_capability,
                    "network_transport": peer.network_transport,
                    "ice_transport_policy": peer.ice_transport_policy,
                }
                for peer in self.peers
            ],
        }

    @classmethod
    def from_dict(cls, data: WebRTCSignalingTopology | Mapping[str, JsonValue]) -> WebRTCSignalingTopology:
        """Construct WebRTCSignalingTopology from dictionary representation.

        Args:
            data (WebRTCSignalingTopology | Mapping[str, JsonValue]):
                Dictionary containing signaling topology configuration.

        Returns:
            WebRTCSignalingTopology: Parsed signaling topology instance.
        """
        return cls.model_validate(data)


LogicalMesh.model_rebuild()
