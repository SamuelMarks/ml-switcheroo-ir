"""Distributed pipeline topologies, WebRTC mesh signaling schemas, and communication models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import typing
import typing_extensions

from pydantic import BaseModel, ConfigDict, Field

from ml_switcheroo_ir.types import JsonValue

PartitionAxisAtom: typing_extensions.TypeAlias = typing.Union[str, int, None]
PartitionAxisInner: typing_extensions.TypeAlias = typing.Union[
    PartitionAxisAtom,
    tuple[PartitionAxisAtom, ...],
    list[PartitionAxisAtom],
]
PartitionAxisElement: typing_extensions.TypeAlias = typing.Union[
    PartitionAxisInner,
    tuple[PartitionAxisInner, ...],
    list[PartitionAxisInner],
]


class LogicalAxis(BaseModel):
    """Represents a named dimension for tensor sizes and sharding (e.g., 'batch', 'embed', 'heads').

    Attributes:
        name (str): Name of the logical axis.
        size (int | None): Optional fixed size of the axis.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    size: int | None = None

    def __init__(self, name: str, size: int | None = None) -> None:
        """Initialize LogicalAxis.

        Args:
            name (str): Name of the logical axis.
            size (int | None): Optional fixed size of the axis.
        """
        super().__init__(name=name, size=size)


class PartitionSpec(BaseModel):
    """Describes how a tensor's dimensions are mapped to a logical mesh.

    Each element in `axes` corresponds to a tensor dimension. An element can be:
    - A string representing the mesh axis name (e.g., 'data').
    - A tuple of strings for multi-axis sharding (e.g., ('data', 'model')).
    - None for a replicated/unsharded dimension.

    Attributes:
        axes (tuple[PartitionAxisElement, ...]): Tuple mapping tensor dimensions to mesh axes.
    """

    model_config = ConfigDict(extra="forbid")

    axes: tuple[PartitionAxisElement, ...] = ()

    def __init__(
        self,
        axes: Sequence[PartitionAxisElement] = (),
    ) -> None:
        """Initialize PartitionSpec normalizing axes into a tuple.

        Args:
            axes (Sequence[PartitionAxisElement]): Sequence mapping tensor dimensions to mesh axes.
        """
        super().__init__(axes=tuple(axes))


class LogicalMesh(BaseModel):
    """Represents a multi-dimensional grid of devices for distributed execution.

    Attributes:
        shape (dict[str, int]): Mapping of mesh axis names to their sizes (e.g., {'data': 4, 'model': 2}).
        webrtc_topology (WebRTCSignalingTopology | None): WebRTC cluster mesh signaling configuration.
    """

    model_config = ConfigDict(extra="forbid")

    shape: dict[str, int]
    webrtc_topology: WebRTCSignalingTopology | None = None

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


class MicrobatchSplittingConfig(BaseModel):
    """Configuration for microbatch splitting strategy and count.

    Attributes:
        strategy (str): Microbatch splitting strategy (e.g. 'uniform', 'adaptive').
        num_microbatches (int): Total number of microbatches.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    strategy: str = "uniform"
    num_microbatches: int = 1


class MeshMappingConfig(BaseModel):
    """Configuration for mapping pipeline stages to accelerator devices.

    Attributes:
        devices_per_stage (int): Number of devices assigned to each pipeline stage.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    devices_per_stage: int = 1


class StageCommunicationConfig(BaseModel):
    """Configuration for inter-stage communication protocol.

    Attributes:
        protocol (str): Transport protocol ('p2p', 'webrtc', 'nccl', 'gloo').
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    protocol: str = "p2p"


class DependencyConfig(BaseModel):
    """Configuration for synchronization dependencies between pipeline stages.

    Attributes:
        source_stage (str): Producer stage identifier.
        target_stage (str): Consumer stage identifier.
        offset_mb (int): Microbatch offset index.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    source_stage: str
    target_stage: str
    offset_mb: int = 0


class SchedulePhaseConfig(BaseModel):
    """Configuration for an execution phase in a pipeline schedule.

    Attributes:
        type (str): Schedule phase type (e.g. 'warmup', '1F1B', 'cooldown').
        operations (list[str]): Ordered operations executed in this phase.
        count_expression (str): Expression resolving the execution iteration count.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    type: str = "1F1B"
    operations: list[str] = Field(default_factory=list)
    count_expression: str = "1"


class PipelineScheduleConfig(BaseModel):
    """Configuration for ordered phases in a pipeline execution schedule.

    Attributes:
        phases (list[SchedulePhaseConfig]): Ordered schedule phases.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    phases: list[SchedulePhaseConfig] = Field(default_factory=list)


class PipelineTopologyConfig(BaseModel):
    """Comprehensive specification for a distributed pipeline topology.

    Attributes:
        microbatch_splitting (MicrobatchSplittingConfig): Microbatch splitting configuration.
        mesh_mapping (MeshMappingConfig): Stage device mapping configuration.
        stage_communication (StageCommunicationConfig): Communication protocol.
        dependencies (list[DependencyConfig]): Inter-stage synchronization dependencies.
        schedule (PipelineScheduleConfig | None): Optional pipeline schedule configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    microbatch_splitting: MicrobatchSplittingConfig
    mesh_mapping: MeshMappingConfig
    stage_communication: StageCommunicationConfig
    dependencies: list[DependencyConfig] = Field(default_factory=list)
    schedule: PipelineScheduleConfig | None = None

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
    def from_dict(
        cls, data: PipelineTopologyConfig | Mapping[str, JsonValue]
    ) -> PipelineTopologyConfig:
        """Construct PipelineTopologyConfig from dictionary representation.

        Args:
            data (PipelineTopologyConfig | Mapping[str, JsonValue]): Dictionary containing topology configuration.

        Returns:
            PipelineTopologyConfig: Parsed configuration instance.
        """
        return cls.model_validate(data)


class WebRTCPeerConfig(BaseModel):
    """Configuration for a WebRTC mesh participant peer.

    Attributes:
        peer_id (str): Unique peer identifier.
        device_capability (str): Accelerator capability (e.g. 'webgpu', 'wasm', 'cpu').
        network_transport (str): Transport channel type (e.g. 'data_channel').
        ice_transport_policy (str): ICE transport policy ('all', 'relay').
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    peer_id: str
    device_capability: str = "webgpu"
    network_transport: str = "data_channel"
    ice_transport_policy: str = "all"


class WebRTCSignalingTopology(BaseModel):
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

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    signaling_url: str = "ws://localhost:8080"
    ice_servers: list[str] = Field(
        default_factory=lambda: ["stun:stun.l.google.com:19302"]
    )
    high_watermark_bytes: int = 1048576
    low_watermark_bytes: int = 262144
    chunk_size_bytes: int = 65536
    connectivity_matrix: list[list[int]] = Field(default_factory=list)
    peers: list[WebRTCPeerConfig] = Field(default_factory=list)

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
    def from_dict(
        cls, data: WebRTCSignalingTopology | Mapping[str, JsonValue]
    ) -> WebRTCSignalingTopology:
        """Construct WebRTCSignalingTopology from dictionary representation.

        Args:
            data (WebRTCSignalingTopology | Mapping[str, JsonValue]): Dictionary containing signaling topology configuration.

        Returns:
            WebRTCSignalingTopology: Parsed signaling topology instance.
        """
        return cls.model_validate(data)


LogicalMesh.model_rebuild()
