"""Tests for WebRTC mesh signaling schemas, layout models, and LogicalMesh integration."""

from __future__ import annotations

from ml_switcheroo_ir import (
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    WebRTCPeerConfig,
    WebRTCSignalingTopology,
)


def test_webrtc_signaling_topology_and_mesh_integration() -> None:
    """Verify WebRTC peer, signaling configuration, and LogicalMesh serialization."""
    peer1 = WebRTCPeerConfig(
        peer_id="browser_peer_alpha",
        device_capability="webgpu",
        network_transport="data_channel",
        ice_transport_policy="all",
    )
    peer2 = WebRTCPeerConfig(
        peer_id="worker_peer_beta",
        device_capability="wasm",
        network_transport="data_channel",
        ice_transport_policy="all",
    )
    assert peer1.peer_id == "browser_peer_alpha"
    assert peer2.device_capability == "wasm"

    signaling_topo = WebRTCSignalingTopology(
        signaling_url="wss://coordinator.switcheroo.ai:8443",
        ice_servers=["stun:stun.l.google.com:19302", "turn:turn.switcheroo.ai:3478"],
        high_watermark_bytes=2097152,
        low_watermark_bytes=524288,
        chunk_size_bytes=131072,
        connectivity_matrix=[[0, 1], [1, 0]],
        peers=[peer1, peer2],
    )
    assert signaling_topo.signaling_url.startswith("wss://")
    assert signaling_topo.chunk_size_bytes == 131072

    # Dictionary serialization
    topo_dict = signaling_topo.to_dict()
    assert topo_dict["high_watermark_bytes"] == 2097152
    reconstructed = WebRTCSignalingTopology.from_dict(topo_dict)
    assert len(reconstructed.peers) == 2
    assert reconstructed.peers[0].peer_id == "browser_peer_alpha"

    # Integrate into LogicalMesh
    mesh = LogicalMesh(
        shape={"data": 2, "workers": 2},
        webrtc_topology=signaling_topo,
    )
    assert mesh.webrtc_topology is not None
    assert mesh.webrtc_topology.chunk_size_bytes == 131072

    # Graph serialization roundtrip with WebRTC mesh
    n1 = LogicalNode(id="n1", op_type="Relu", shape_metadata=(2, 4))
    graph = LogicalGraph(
        name="WebRTCGraph",
        nodes={"n1": n1},
        inputs=["n1"],
        outputs=["n1"],
        mesh=mesh,
    )

    g_dict = graph.to_dict()
    assert "mesh" in g_dict
    assert "webrtc_topology" in g_dict["mesh"]
    assert g_dict["mesh"]["webrtc_topology"]["chunk_size_bytes"] == 131072

    g_restored = LogicalGraph.from_dict(g_dict)
    assert g_restored.mesh is not None
    assert g_restored.mesh.webrtc_topology is not None
    assert (
        g_restored.mesh.webrtc_topology.signaling_url
        == "wss://coordinator.switcheroo.ai:8443"
    )
    assert len(g_restored.mesh.webrtc_topology.peers) == 2
