"""Distributed pipeline topologies, WebRTC mesh signaling schemas, and communication models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field


class MicrobatchSplittingDict(TypedDict):
    """TypedDict representation of MicrobatchSplittingConfig.

    Attributes:
        strategy (str): Microbatch splitting strategy.
        num_microbatches (int): Total number of microbatches.
    """

    strategy: str
    num_microbatches: int


class MeshMappingDict(TypedDict):
    """TypedDict representation of MeshMappingConfig.

    Attributes:
        devices_per_stage (int): Number of devices assigned to each pipeline stage.
    """

    devices_per_stage: int


class StageCommunicationDict(TypedDict):
    """TypedDict representation of StageCommunicationConfig.

    Attributes:
        protocol (str): Transport protocol identifier.
    """

    protocol: str


class DependencyDict(TypedDict):
    """TypedDict representation of DependencyConfig.

    Attributes:
        source_stage (str): Producer stage identifier.
        target_stage (str): Consumer stage identifier.
        offset_mb (int): Microbatch offset index.
    """

    source_stage: str
    target_stage: str
    offset_mb: int


class SchedulePhaseDict(TypedDict):
    """TypedDict representation of SchedulePhaseConfig.

    Attributes:
        type (str): Schedule phase type.
        operations (list[str]): Ordered operations executed in this phase.
        count_expression (str): Expression resolving the execution iteration count.
    """

    type: str
    operations: list[str]
    count_expression: str


class PipelineScheduleDict(TypedDict):
    """TypedDict representation of PipelineScheduleConfig.

    Attributes:
        phases (list[SchedulePhaseDict]): Ordered schedule phases.
    """

    phases: list[SchedulePhaseDict]


class PipelineTopologyDict(TypedDict):
    """TypedDict representation of PipelineTopologyConfig.

    Attributes:
        microbatch_splitting (MicrobatchSplittingDict): Microbatch splitting configuration.
        mesh_mapping (MeshMappingDict): Stage device mapping configuration.
        stage_communication (StageCommunicationDict): Communication protocol.
        dependencies (list[DependencyDict]): Inter-stage synchronization dependencies.
        schedule (PipelineScheduleDict | None): Optional pipeline schedule configuration.
    """

    microbatch_splitting: MicrobatchSplittingDict
    mesh_mapping: MeshMappingDict
    stage_communication: StageCommunicationDict
    dependencies: list[DependencyDict]
    schedule: PipelineScheduleDict | None


class WebRTCPeerDict(TypedDict):
    """TypedDict representation of WebRTCPeerConfig.

    Attributes:
        peer_id (str): Unique peer identifier.
        device_capability (str): Accelerator capability.
        network_transport (str): Transport channel type.
        ice_transport_policy (str): ICE transport policy.
    """

    peer_id: str
    device_capability: str
    network_transport: str
    ice_transport_policy: str


class WebRTCSignalingTopologyDict(TypedDict):
    """TypedDict representation of WebRTCSignalingTopology.

    Attributes:
        signaling_url (str): WebSocket or HTTP signaling server endpoint.
        ice_servers (list[str]): STUN/TURN server URLs.
        high_watermark_bytes (int): Buffer high-water mark for backpressure management.
        low_watermark_bytes (int): Buffer low-water mark for resumed transmission.
        chunk_size_bytes (int): Maximum chunk payload size for large tensor transfers.
        connectivity_matrix (list[list[int]]): Peer connectivity adjacency matrix.
        peers (list[WebRTCPeerDict]): List of participating peer specifications.
    """

    signaling_url: str
    ice_servers: list[str]
    high_watermark_bytes: int
    low_watermark_bytes: int
    chunk_size_bytes: int
    connectivity_matrix: list[list[int]]
    peers: list[WebRTCPeerDict]


class MicrobatchSplittingConfig(BaseModel):
    """Configuration for microbatch splitting strategy and count.

    Attributes:
        strategy (str): Microbatch splitting strategy (e.g. 'uniform', 'adaptive').
        num_microbatches (int): Total number of microbatches.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: str = "uniform"
    num_microbatches: int = 1


class MeshMappingConfig(BaseModel):
    """Configuration for mapping pipeline stages to accelerator devices.

    Attributes:
        devices_per_stage (int): Number of devices assigned to each pipeline stage.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    devices_per_stage: int = 1


class StageCommunicationConfig(BaseModel):
    """Configuration for inter-stage communication protocol.

    Attributes:
        protocol (str): Transport protocol ('p2p', 'webrtc', 'nccl', 'gloo').
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    protocol: str = "p2p"


class DependencyConfig(BaseModel):
    """Configuration for synchronization dependencies between pipeline stages.

    Attributes:
        source_stage (str): Producer stage identifier.
        target_stage (str): Consumer stage identifier.
        offset_mb (int): Microbatch offset index.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str = "1F1B"
    operations: list[str] = Field(default_factory=list)
    count_expression: str = "1"


class PipelineScheduleConfig(BaseModel):
    """Configuration for ordered phases in a pipeline execution schedule.

    Attributes:
        phases (list[SchedulePhaseConfig]): Ordered schedule phases.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    microbatch_splitting: MicrobatchSplittingConfig
    mesh_mapping: MeshMappingConfig
    stage_communication: StageCommunicationConfig
    dependencies: list[DependencyConfig] = Field(default_factory=list)
    schedule: PipelineScheduleConfig | None = None

    def to_dict(self) -> PipelineTopologyDict:
        """Convert pipeline topology configuration to dictionary representation.

        Returns:
            PipelineTopologyDict: Strongly typed dictionary representation.
        """
        return cast("PipelineTopologyDict", self.model_dump())

    @classmethod
    def from_dict(
        cls, data: PipelineTopologyDict | Mapping[str, object]
    ) -> PipelineTopologyConfig:
        """Construct PipelineTopologyConfig from dictionary representation.

        Args:
            data (PipelineTopologyDict | Mapping[str, object]): Dictionary containing topology configuration.

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    signaling_url: str = "ws://localhost:8080"
    ice_servers: list[str] = Field(
        default_factory=lambda: ["stun:stun.l.google.com:19302"]
    )
    high_watermark_bytes: int = 1048576
    low_watermark_bytes: int = 262144
    chunk_size_bytes: int = 65536
    connectivity_matrix: list[list[int]] = Field(default_factory=list)
    peers: list[WebRTCPeerConfig] = Field(default_factory=list)

    def to_dict(self) -> WebRTCSignalingTopologyDict:
        """Convert WebRTC signaling topology to dictionary representation.

        Returns:
            WebRTCSignalingTopologyDict: Strongly typed dictionary representation.
        """
        return cast("WebRTCSignalingTopologyDict", self.model_dump())

    @classmethod
    def from_dict(
        cls, data: WebRTCSignalingTopologyDict | Mapping[str, object]
    ) -> WebRTCSignalingTopology:
        """Construct WebRTCSignalingTopology from dictionary representation.

        Args:
            data (WebRTCSignalingTopologyDict | Mapping[str, object]): Dictionary containing signaling topology configuration.

        Returns:
            WebRTCSignalingTopology: Parsed signaling topology instance.
        """
        return cls.model_validate(data)
