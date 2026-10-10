//! Inbound `Gossip` gRPC service.
//!
//! Implements the proto-defined `Gossip` service (see
//! `proto/gossip.proto`). For each accepted connection this file
//! handles both RPCs:
//!
//! - **`PingServer` (unary)**: SWIM-style ping with optional embedded
//!   `state_sync` and `ping_req` indirect-probe forwarding. Used by
//!   the membership channel; no streaming, no per-call task state.
//! - **`SyncStream` (bidirectional streaming)**: spawns **two tasks**
//!   per accepted stream that share a few small pieces of state:
//!   - **Inbound-handler task** — reads frames the dialer sends.
//!     The first non-empty `msg.peer_id` is written into
//!     `learned_peer: Arc<RwLock<Option<String>>>`. Dispatches
//!     received `StreamBatch` payloads into local `MeshKV` via
//!     [`dispatch_stream_batch`](crate::transport::sync_stream::dispatch_stream_batch).
//!     Idle-timeout wraps `incoming.next()` so unhealthy peers don't
//!     pin the task indefinitely.
//!   - **Inbound-sender task** — every 1 Hz, reads the shared
//!     [`RoundBatch`](crate::kv::RoundBatch) slot produced by
//!     [`gossip_controller`](crate::gossip_controller)'s event loop,
//!     filters targeted entries for the learned peer, and emits
//!     `StreamBatch` envelopes on this stream.
//!
//! Asymmetry with the outbound side: the dialer (gossip_controller)
//! knows its counterparty by name at task-spawn time. The acceptor
//! here does not — the only way to associate the in-flight TCP/gRPC
//! connection with a logical mesh identity (today, pre-mTLS-derived
//! identity) is to read the first inbound frame's `peer_id`. That
//! learning step is what `learned_peer` exists for.

use std::{net::SocketAddr, pin::Pin, sync::Arc, time::Duration};

use anyhow::Result;
use futures::Stream;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tonic::{
    transport::{server::TcpIncoming, Server},
    Response, Status,
};
use tracing as log;
use tracing::instrument;

use super::{
    crdt_kv::CrdtWatermark,
    metrics::{record_ack, record_nack, record_peer_reconnect, update_peer_connections},
    mtls::MTLSManager,
    partition::PartitionDetector,
    service::{
        gossip::{
            self,
            gossip_server::{Gossip, GossipServer},
            GossipMessage, NodeState, NodeStatus, NodeUpdate, PingReq, StreamBatch, StreamMessage,
            StreamMessageType,
        },
        try_ping, ClusterState,
    },
    transport::{
        crdt_batch::{
            build_crdt_batches, crdt_ack_to_watermark, dispatch_crdt_batch, wrap_crdt_ack,
            wrap_crdt_batch,
        },
        limits::{MAX_MESSAGE_SIZE, MAX_STREAM_CHUNK_BYTES, STREAM_IDLE_TIMEOUT},
        sync_stream::{
            build_heartbeat, build_peer_stream_batches, dispatch_stream_batch, wrap_stream_batch,
        },
    },
};

/// Server-side handler for the proto-defined `Gossip` service.
///
/// One instance per mesh node. Configured at startup by
/// `service.rs::MeshServerBuilder` and registered with tonic. Holds
/// shared references to the cluster state, mTLS config, partition
/// detector, the per-node `RoundBatch` slot owned by the controller,
/// and the node's `MeshKV`. See module docs for the per-accepted-
/// stream task topology spawned by `sync_stream`.
#[derive(Debug)]
pub struct GossipService {
    state: ClusterState,
    listen_addr: SocketAddr,
    advertise_addr: SocketAddr,
    self_name: String,
    partition_detector: Option<Arc<PartitionDetector>>,
    mtls_manager: Option<Arc<MTLSManager>>,
    /// Shared reference to the current stream RoundBatch, drained once
    /// per round by the GossipController. Server-side handlers read
    /// broadcast drain_entries and also emit targeted_entries addressed
    /// to the remote peer learned from the first inbound message, so
    /// publish_to(peer) works in both directions of a peer pair.
    current_stream_batch: Option<Arc<parking_lot::RwLock<Arc<crate::kv::RoundBatch>>>>,
    /// Node-wide MeshKV handle. Owns the stream buffers, subscriber
    /// registry, and chunk assembler shared with the client-side
    /// SyncStream handlers.
    mesh_kv: Option<Arc<crate::kv::MeshKV>>,
}

impl GossipService {
    pub fn new(
        state: ClusterState,
        listen_addr: SocketAddr,
        advertise_addr: SocketAddr,
        self_name: &str,
    ) -> Self {
        Self {
            state,
            listen_addr,
            advertise_addr,
            self_name: self_name.to_string(),
            partition_detector: None,
            mtls_manager: None,
            current_stream_batch: None,
            mesh_kv: None,
        }
    }

    /// Attach the shared stream RoundBatch reference. Server-side
    /// handlers emit broadcast drain_entries plus targeted_entries
    /// whose target matches the remote peer learned from the first
    /// inbound StreamMessage, so publish_to() works in both directions
    /// of a peer pair.
    pub fn with_current_stream_batch(
        mut self,
        current_stream_batch: Arc<parking_lot::RwLock<Arc<crate::kv::RoundBatch>>>,
    ) -> Self {
        self.current_stream_batch = Some(current_stream_batch);
        self
    }

    /// Attach the node-wide MeshKV handle. Plumbed from the server
    /// builder so stream buffers, subscribers, and the chunk assembler
    /// are shared between the client-side (outbound) and server-side
    /// (inbound) SyncStream handlers.
    pub fn with_mesh_kv(mut self, mesh_kv: Arc<crate::kv::MeshKV>) -> Self {
        self.mesh_kv = Some(mesh_kv);
        self
    }

    pub fn with_partition_detector(mut self, partition_detector: Arc<PartitionDetector>) -> Self {
        self.partition_detector = Some(partition_detector);
        self
    }

    pub fn with_mtls_manager(mut self, mtls_manager: Arc<MTLSManager>) -> Self {
        self.mtls_manager = Some(mtls_manager);
        self
    }

    pub async fn serve_ping_with_shutdown<F: std::future::Future<Output = ()>>(
        self,
        signal: F,
    ) -> Result<()> {
        let listen_addr = self.listen_addr;
        let mut server = self.server_builder().await?;
        let service = GossipServer::new(self)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip);

        server
            .add_service(service)
            .serve_with_shutdown(listen_addr, signal)
            .await?;
        Ok(())
    }

    /// The listener's server: plain gRPC, or TLS with this node's
    /// certificate (client certificates checked against the CA) when mTLS is
    /// configured.
    async fn server_builder(&self) -> Result<Server> {
        let mut server = Server::builder();
        if let Some(mtls_manager) = &self.mtls_manager {
            server = server.tls_config(mtls_manager.server_tls_config().await?)?;
            log::info!(
                "Mesh listener {} serves TLS with the node certificate",
                self.listen_addr
            );
        }
        Ok(server)
    }

    pub async fn serve_ping_with_listener<F: std::future::Future<Output = ()>>(
        self,
        listener: tokio::net::TcpListener,
        signal: F,
    ) -> Result<()> {
        let incoming = TcpIncoming::from(listener);
        let mut server = self.server_builder().await?;
        let service = GossipServer::new(self)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip);
        server
            .add_service(service)
            .serve_with_incoming_shutdown(incoming, signal)
            .await?;
        Ok(())
    }

    fn merge_state(&self, incoming_nodes: Vec<NodeState>) -> bool {
        let mut state = self.state.write();
        let mut updated = false;
        for node in incoming_nodes {
            state
                .entry(node.name.clone())
                .and_modify(|entry| {
                    if node.version > entry.version {
                        *entry = node.clone();
                        updated = true;
                    }
                })
                .or_insert_with(|| {
                    updated = true;
                    node
                });
        }
        if updated {
            log::info!("Cluster state updated. Current nodes: {}", state.len());
        }
        updated
    }
}

#[tonic::async_trait]
impl Gossip for GossipService {
    type SyncStreamStream =
        Pin<Box<dyn Stream<Item = Result<StreamMessage, Status>> + Send + 'static>>;

    #[instrument(fields(name = %self.self_name), skip(self, request))]
    async fn ping_server(
        &self,
        request: tonic::Request<GossipMessage>,
    ) -> std::result::Result<Response<NodeUpdate>, Status> {
        let message = request.into_inner();
        match message.payload {
            Some(gossip::gossip_message::Payload::Ping(ping)) => {
                log::info!("Received {:?}", ping);
                if let Some(stat_sync) = ping.state_sync {
                    log::info!("Merging state from Ping: {} nodes", stat_sync.nodes.len());
                    self.merge_state(stat_sync.nodes);
                }
                // Return current status of self node (could be Alive or Leaving)
                let current_status = {
                    let state = self.state.read();
                    state
                        .get(&self.self_name)
                        .map(|n| n.status)
                        .unwrap_or(NodeStatus::Alive as i32)
                };
                Ok(Response::new(NodeUpdate {
                    name: self.self_name.clone(),
                    address: self.advertise_addr.to_string(),
                    status: current_status,
                }))
            }
            Some(gossip::gossip_message::Payload::PingReq(PingReq { node: Some(node) })) => {
                log::info!("PingReq to node {} addr:{}", node.name, node.address);
                let res = try_ping(&node, None, self.mtls_manager.clone()).await?;
                Ok(Response::new(res))
            }
            _ => Err(Status::invalid_argument("Invalid message payload")),
        }
    }

    #[instrument(fields(name = %self.self_name), skip(self, request))]
    async fn sync_stream(
        &self,
        request: tonic::Request<tonic::Streaming<StreamMessage>>,
    ) -> Result<Response<Self::SyncStreamStream>, Status> {
        let mut incoming = request.into_inner();
        let self_name = self.self_name.clone();
        let mesh_kv = self.mesh_kv.clone();

        const CHANNEL_CAPACITY: usize = 128;
        let (tx, rx) = mpsc::channel::<Result<StreamMessage, Status>>(CHANNEL_CAPACITY);

        // Remote peer identity, learned from the first inbound message and
        // used by the sender to filter targeted_entries.
        let learned_peer: Arc<parking_lot::RwLock<Option<String>>> =
            Arc::new(parking_lot::RwLock::new(None));

        // Per-peer CRDT send watermark: the highest version this peer has acked
        // for each key. The sender filters the op-log by it; the inbound handler
        // advances it on CrdtAck. Keyed by key so a dropped/late op only delays
        // that one key (resent next round), never strands it.
        let acked: Arc<parking_lot::RwLock<CrdtWatermark>> =
            Arc::new(parking_lot::RwLock::new(CrdtWatermark::new()));

        // Server-side stream sender: periodically emit fresh stream batches
        // (broadcast drain_entries + targeted entries addressed to the
        // learned peer). Skipped when no current_stream_batch is attached.
        let sender_handle = if let Some(stream_batch_handle) = self.current_stream_batch.clone() {
            let tx_sender = tx.clone();
            let self_name_sender = self_name.clone();
            let learned_peer_sender = learned_peer.clone();
            let acked_sender = acked.clone();
            #[expect(
                clippy::disallowed_methods,
                reason = "server-side sender bound to sync_stream lifetime; terminates when channel closes or handle is aborted on disconnect"
            )]
            Some(tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(1));
                let mut sequence_counter: u64 = 0;
                let mut last_stream_batch: Option<Arc<crate::kv::RoundBatch>> = None;

                loop {
                    interval.tick().await;

                    // If the paired inbound handler has dropped its end of
                    // the mpsc, we have nobody to send to. Exit cleanly
                    // instead of looping forever — important when peer never
                    // identifies (so we never reach try_send to learn the
                    // channel is closed).
                    if tx_sender.is_closed() {
                        return;
                    }

                    let stream_batch = stream_batch_handle.read().clone();

                    // Stream batches: gated by learned peer + Arc-freshness. A
                    // skipped tick leaves `last_stream_batch` untouched so the
                    // same RoundBatch is re-evaluated later.
                    let stream_tick = {
                        let guard = learned_peer_sender.read();
                        plan_sender_tick(
                            last_stream_batch.as_ref(),
                            &stream_batch,
                            guard.as_deref(),
                        )
                    };
                    if let SenderTick::Emit(batches) = stream_tick {
                        last_stream_batch = Some(stream_batch.clone());
                        for batch in batches {
                            sequence_counter += 1;
                            let msg = wrap_stream_batch(batch, sequence_counter, &self_name_sender);
                            match tx_sender.try_send(Ok(msg)) {
                                Ok(()) => {}
                                Err(mpsc::error::TrySendError::Full(_)) => {
                                    log::debug!("server-side stream batch dropped on backpressure");
                                    break;
                                }
                                Err(mpsc::error::TrySendError::Closed(_)) => return,
                            }
                        }
                    }

                    // CRDT op-log: evaluated every tick (acks shrink the delta
                    // even when the RoundBatch is unchanged). Send only ops the
                    // peer has not acked; the watermark advances solely on
                    // CrdtAck, so unacked keys retry next round.
                    let crdt_ops: Vec<_> = {
                        let acked = acked_sender.read();
                        stream_batch
                            .crdt_ops
                            .operations()
                            .iter()
                            .filter(|op| acked.allows(op))
                            .cloned()
                            .collect()
                    };
                    for crdt_batch in build_crdt_batches(&crdt_ops, MAX_STREAM_CHUNK_BYTES) {
                        sequence_counter += 1;
                        let msg = wrap_crdt_batch(crdt_batch, sequence_counter, &self_name_sender);
                        match tx_sender.try_send(Ok(msg)) {
                            Ok(()) => {}
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                log::debug!("server-side crdt batch dropped on backpressure");
                                break;
                            }
                            Err(mpsc::error::TrySendError::Closed(_)) => return,
                        }
                    }
                }
            }))
        } else {
            None
        };

        let learned_peer_inbound = learned_peer.clone();
        let acked_inbound = acked.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "server-side inbound handler bound to sync_stream lifetime; terminates when the stream closes"
        )]
        tokio::spawn(async move {
            // Close the stream if no inbound message arrives within
            // STREAM_IDLE_TIMEOUT — protects against idle clients
            // pinning the server-side task and mpsc channel indefinitely.
            let mut peer_id = String::new();
            update_peer_connections(&peer_id, true);
            let mut sequence: u64 = 0;

            loop {
                let msg = match tokio::time::timeout(STREAM_IDLE_TIMEOUT, incoming.next()).await {
                    Ok(Some(Ok(msg))) => msg,
                    Ok(Some(Err(e))) => {
                        log::error!("Error receiving stream message: {}", e);
                        break;
                    }
                    Ok(None) => break,
                    Err(_) => {
                        log::warn!(
                            peer = %peer_id,
                            "sync_stream idle timeout ({STREAM_IDLE_TIMEOUT:?}) — closing"
                        );
                        break;
                    }
                };

                // Bind peer_id to the first non-empty inbound id. A later
                // frame whose msg.peer_id (empty or otherwise) doesn't
                // match is treated as identity change and closes the
                // stream. Pre-mTLS-binding defence; mTLS-derived
                // identity is the authoritative long-term fix.
                if peer_id.is_empty() {
                    if !msg.peer_id.is_empty() {
                        peer_id = msg.peer_id.clone();
                        update_peer_connections(&peer_id, true);
                        *learned_peer_inbound.write() = Some(peer_id.clone());
                    }
                } else if msg.peer_id != peer_id {
                    log::warn!(
                        expected_peer_id = %peer_id,
                        received_peer_id = %msg.peer_id,
                        "peer_id changed mid-stream; closing sync_stream"
                    );
                    break;
                }
                sequence = sequence.max(msg.sequence);

                match msg.message_type() {
                    StreamMessageType::Heartbeat => {
                        let heartbeat = build_heartbeat(sequence, &self_name);
                        if tx.send(Ok(heartbeat)).await.is_err() {
                            break;
                        }
                    }
                    StreamMessageType::Ack => {
                        if let Some(gossip::stream_message::Payload::Ack(ack)) = &msg.payload {
                            record_ack(&peer_id, ack.success);
                        }
                    }
                    StreamMessageType::Nack => record_nack(&peer_id),
                    StreamMessageType::StreamBatch => {
                        if let (
                            Some(mesh_kv),
                            Some(gossip::stream_message::Payload::StreamBatch(batch)),
                        ) = (&mesh_kv, msg.payload)
                        {
                            dispatch_stream_batch(mesh_kv, &msg.peer_id, batch.entries);
                        }
                    }
                    StreamMessageType::CrdtBatch => {
                        if let (
                            Some(mesh_kv),
                            Some(gossip::stream_message::Payload::CrdtBatch(batch)),
                        ) = (&mesh_kv, msg.payload)
                        {
                            // Merge, then ack the per-key versions back so the
                            // peer can advance its send watermark. Ack loss is
                            // fine — the peer resends unacked keys next round —
                            // so drop it on a full channel rather than block.
                            let ack = dispatch_crdt_batch(mesh_kv, batch);
                            if !ack.is_empty() {
                                let _ = tx.try_send(Ok(wrap_crdt_ack(&ack, sequence, &self_name)));
                            }
                        }
                    }
                    // CRDT delivery ack: advance this peer's send watermark.
                    StreamMessageType::CrdtAck => {
                        if let Some(gossip::stream_message::Payload::CrdtAck(ack)) = msg.payload {
                            acked_inbound.write().merge_max(&crdt_ack_to_watermark(ack));
                        }
                    }
                    StreamMessageType::IncrementalUpdate
                    | StreamMessageType::SnapshotRequest
                    | StreamMessageType::SnapshotChunk
                    | StreamMessageType::SnapshotComplete => {
                        log::debug!(
                            peer = %peer_id,
                            message_type = ?msg.message_type(),
                            "ignoring v1 wire message (state-sync removed)",
                        );
                    }
                }
            }

            update_peer_connections(&peer_id, false);
            record_peer_reconnect(&peer_id);
            if let Some(handle) = sender_handle {
                handle.abort();
            }
        });

        let output_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(
            Box::pin(output_stream) as Self::SyncStreamStream
        ))
    }
}

/// Outcome of one tick of the inbound sender task.
///
/// The task only marks a `RoundBatch` consumed (advances
/// `last_stream_batch`) when [`SenderTick::Emit`] is returned. The two
/// skip variants preserve `last_stream_batch` so the same `Arc` is
/// retried on the next tick — important because before peer identity
/// is learned, we cannot build the correct per-peer batch, and consuming
/// the batch anyway would silently drop its targeted entries.
#[derive(Debug)]
enum SenderTick {
    /// Peer identity not yet known. Wait until the inbound handler
    /// records the dialer's `peer_id` from the first frame.
    SkipPeerUnknown,
    /// Round batch unchanged since the last successful emit.
    SkipBatchUnchanged,
    /// Build successful — emit these batches and advance the watermark.
    Emit(Vec<StreamBatch>),
}

/// Decide what one tick of the inbound sender should do.
///
/// Splitting this from the async task body makes the per-tick decision
/// pinnable in unit tests without standing up an interval timer, a
/// real mpsc, or a tokio runtime. The async loop is responsible for
/// freshness-watermark advancement and channel I/O; this function is
/// responsible only for the "what should we do this tick" decision.
fn plan_sender_tick(
    last_stream_batch: Option<&Arc<crate::kv::RoundBatch>>,
    current_stream_batch: &Arc<crate::kv::RoundBatch>,
    learned_peer: Option<&str>,
) -> SenderTick {
    let peer_id = match learned_peer {
        Some(peer) if !peer.is_empty() => peer,
        _ => return SenderTick::SkipPeerUnknown,
    };
    let fresh = last_stream_batch.is_none_or(|last| !Arc::ptr_eq(last, current_stream_batch));
    if !fresh {
        return SenderTick::SkipBatchUnchanged;
    }
    SenderTick::Emit(build_peer_stream_batches(current_stream_batch, peer_id))
}

#[cfg(test)]
mod sender_tick_tests {
    use bytes::Bytes;

    use super::*;
    use crate::kv::RoundBatch;

    fn round_batch_with(
        drain: Vec<(&str, &[u8])>,
        targeted: Vec<(&str, &str, &[u8])>,
    ) -> Arc<RoundBatch> {
        Arc::new(RoundBatch {
            drain_entries: drain
                .into_iter()
                .map(|(k, v)| (k.to_string(), Bytes::copy_from_slice(v)))
                .collect(),
            targeted_entries: targeted
                .into_iter()
                .map(|(t, k, v)| (t.to_string(), k.to_string(), Bytes::copy_from_slice(v)))
                .collect(),
            crdt_ops: Arc::default(),
        })
    }

    fn batch_keys(batches: &[StreamBatch]) -> Vec<String> {
        batches
            .iter()
            .flat_map(|b| b.entries.iter().map(|e| e.key.clone()))
            .collect()
    }

    #[test]
    fn skips_when_peer_unknown_and_keeps_watermark_unchanged() {
        // Original race: with learned_peer = None the previous code
        // would consume the batch with peer_id = "", drop targeted
        // entries, and advance last_stream_batch — so the next tick
        // (after peer identity arrived) would see an unchanged Arc and
        // never re-send the lost targeted entries.
        let rb = round_batch_with(
            vec![("td:foo", b"d")],
            vec![("peer_X", "tree:req:abc", b"r")],
        );
        let decision = plan_sender_tick(None, &rb, None);
        assert!(matches!(decision, SenderTick::SkipPeerUnknown));
    }

    #[test]
    fn skips_when_peer_is_empty_string() {
        // Defensive: an empty learned_peer string must not be treated
        // as a valid identity, even though the underlying string-equality
        // would otherwise let a targeted entry with target == "" leak
        // through.
        let rb = round_batch_with(vec![], vec![("", "tree:req:weird", b"r")]);
        let decision = plan_sender_tick(None, &rb, Some(""));
        assert!(matches!(decision, SenderTick::SkipPeerUnknown));
    }

    #[test]
    fn emits_drain_and_targeted_once_peer_is_known() {
        // First tick (peer unknown) returns SkipPeerUnknown. After the
        // inbound handler learns the peer, a second tick on the same
        // Arc must emit both drain and the matching targeted entry.
        let rb = round_batch_with(
            vec![("td:foo", b"d")],
            vec![("peer_X", "tree:req:abc", b"r")],
        );
        let pre = plan_sender_tick(None, &rb, None);
        assert!(matches!(pre, SenderTick::SkipPeerUnknown));

        // last_stream_batch was left unchanged by the skip, so we still
        // pass None here — modeling the watermark not having advanced.
        let post = plan_sender_tick(None, &rb, Some("peer_X"));
        let SenderTick::Emit(batches) = post else {
            panic!("expected Emit after peer learned, got {post:?}");
        };
        let keys = batch_keys(&batches);
        assert!(keys.contains(&"td:foo".to_string()), "drain entry emitted");
        assert!(
            keys.contains(&"tree:req:abc".to_string()),
            "targeted entry for peer_X emitted"
        );
    }

    #[test]
    fn does_not_resend_same_arc_after_consume() {
        // After a successful Emit, the loop advances last_stream_batch
        // to the same Arc. The next tick must report SkipBatchUnchanged
        // and emit nothing.
        let rb = round_batch_with(vec![("td:foo", b"d")], vec![]);
        let decision = plan_sender_tick(Some(&rb), &rb, Some("peer_X"));
        assert!(matches!(decision, SenderTick::SkipBatchUnchanged));
    }

    #[test]
    fn emits_only_targeted_entries_for_learned_peer() {
        // Targeted-entry filtering still works correctly: a batch with
        // entries for multiple peers must only emit entries addressed to
        // this stream's learned peer (plus drain entries which broadcast).
        let rb = round_batch_with(
            vec![("td:bcast", b"b")],
            vec![
                ("peer_X", "tree:req:x", b"x"),
                ("peer_Y", "tree:req:y", b"y"),
            ],
        );
        let SenderTick::Emit(batches) = plan_sender_tick(None, &rb, Some("peer_X")) else {
            panic!("expected Emit");
        };
        let keys = batch_keys(&batches);
        assert!(keys.contains(&"td:bcast".to_string()));
        assert!(keys.contains(&"tree:req:x".to_string()));
        assert!(
            !keys.contains(&"tree:req:y".to_string()),
            "must not include entries addressed to other peers"
        );
    }

    #[test]
    fn drain_only_batch_still_emits_when_peer_learned() {
        // Regression guard: a batch with only drain entries and no
        // targeted entries should still emit after peer is learned —
        // we do not want a future "skip when no targeted for this peer"
        // optimization to reintroduce the original race for drain entries.
        let rb = round_batch_with(vec![("td:foo", b"d"), ("td:bar", b"d")], vec![]);
        let SenderTick::Emit(batches) = plan_sender_tick(None, &rb, Some("peer_X")) else {
            panic!("expected Emit for drain-only batch");
        };
        let keys = batch_keys(&batches);
        assert!(keys.contains(&"td:foo".to_string()));
        assert!(keys.contains(&"td:bar".to_string()));
    }
}
