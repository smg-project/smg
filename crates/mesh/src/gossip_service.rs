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
//!     `learned_peer: Arc<RwLock<Option<String>>>` and later frames are
//!     held to it. Each frame is handled by
//!     [`handle_frame`](crate::transport::peer_stream::handle_frame), the
//!     same way the dialer handles its own; the idle timeout around
//!     `incoming.next()` lives in
//!     [`next_frame`](crate::transport::peer_stream::next_frame).
//!   - **Inbound-sender task** — every 1 Hz, reads the shared
//!     [`RoundBatch`](crate::kv::RoundBatch) slot produced by
//!     [`gossip_controller`](crate::gossip_controller)'s event loop and
//!     runs [`send_round`](crate::transport::peer_stream::send_round)
//!     for the learned peer.
//!
//! Asymmetry with the outbound side: the dialer (gossip_controller)
//! knows its counterparty by name at task-spawn time. The acceptor
//! here does not — the only way to associate the in-flight TCP/gRPC
//! connection with a logical mesh identity (today, pre-mTLS-derived
//! identity) is to read the first inbound frame's `peer_id`. That
//! learning step is what `learned_peer` exists for.

use std::{net::SocketAddr, pin::Pin, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use futures::Stream;
use tokio::sync::mpsc;
use tonic::{
    transport::{
        server::{Router, TcpIncoming},
        Server,
    },
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
            GossipMessage, NodeState, NodeStatus, NodeUpdate, PingReq, StreamMessage,
        },
        try_ping, ClusterState,
    },
    transport::{
        limits::MAX_MESSAGE_SIZE,
        peer_stream::{handle_frame, next_frame, send_round, Frame, RoundOutcome, SenderState},
    },
};

/// Where a prepared gossip server accepts: a listener its caller bound, or
/// the configured address, bound when it serves.
enum Incoming {
    Listener(TcpIncoming),
    Address(SocketAddr),
}

/// A gossip server built by [`GossipService::prepare`] and not yet serving:
/// its TLS configuration, when the node has TLS material, is already built.
/// [`serve`](Self::serve) runs it until the shutdown signal completes.
pub struct PreparedGossipServer {
    router: Router,
    incoming: Incoming,
}

impl PreparedGossipServer {
    /// Serves until `signal` completes.
    pub async fn serve<F: std::future::Future<Output = ()>>(self, signal: F) -> Result<()> {
        match self.incoming {
            Incoming::Listener(incoming) => {
                self.router
                    .serve_with_incoming_shutdown(incoming, signal)
                    .await?;
            }
            Incoming::Address(addr) => {
                self.router.serve_with_shutdown(addr, signal).await?;
            }
        }
        Ok(())
    }
}

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

    /// The tonic server for this service: with the node's TLS material, the
    /// server's TLS configuration is built here, from the files the mTLS
    /// manager reads, so material that exists but cannot be used fails here,
    /// not where the server is polled.
    async fn server_builder(&self) -> Result<Server> {
        let mut server = Server::builder();
        if let Some(mtls_manager) = &self.mtls_manager {
            let tls = mtls_manager
                .server_tls_config()
                .await
                .context("the mesh TLS material cannot be read")?;
            server = server.tls_config(tls).context(
                "the mesh TLS material cannot be used for the mesh listener (the node \
                 certificate, its private key or the CA certificate)",
            )?;
            log::info!(
                "Mesh listener {} is prepared to serve TLS with the node certificate",
                self.listen_addr
            );
        }
        Ok(server)
    }

    /// This service's server, built but not serving: on `listener` when one
    /// is given (the caller bound it), else on the configured listen address,
    /// bound when it serves. Everything that can fail before the server
    /// serves happens here, the TLS configuration first of all, so a caller
    /// learns of it before it serves anything itself.
    pub async fn prepare(
        self,
        listener: Option<tokio::net::TcpListener>,
    ) -> Result<PreparedGossipServer> {
        let incoming = match listener {
            // The socket option the address path sets (tonic's server
            // default): TCP_NODELAY on every accepted connection; no
            // keepalive either way.
            Some(listener) => {
                Incoming::Listener(TcpIncoming::from(listener).with_nodelay(Some(true)))
            }
            None => Incoming::Address(self.listen_addr),
        };
        let mut server = self.server_builder().await?;
        let service = GossipServer::new(self)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip);
        Ok(PreparedGossipServer {
            router: server.add_service(service),
            incoming,
        })
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

        // Server-side stream sender: the shared round every second, for the
        // peer once it is learned (until then only the CRDT ops go out).
        // Skipped when no current_stream_batch is attached.
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
                let mut sender = SenderState::default();

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
                    let peer = learned_peer_sender.read().clone();
                    let outcome = send_round(
                        &mut sender,
                        &stream_batch,
                        peer.as_deref(),
                        &acked_sender,
                        &self_name_sender,
                        &mut || {
                            sequence_counter += 1;
                            sequence_counter
                        },
                        &tx_sender,
                    );
                    if outcome == RoundOutcome::Closed {
                        return;
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
            let mut peer_id = String::new();
            update_peer_connections(&peer_id, true);
            let mut sequence: u64 = 0;

            while let Some(msg) = next_frame(&mut incoming, &peer_id).await {
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

                let frame = handle_frame(
                    msg,
                    &peer_id,
                    &self_name,
                    mesh_kv.as_deref(),
                    &acked_inbound,
                    &mut || sequence,
                    &tx,
                )
                .await;
                match frame {
                    Frame::Continue | Frame::Ack { success: None, .. } => {}
                    Frame::Ack {
                        success: Some(success),
                        ..
                    } => record_ack(&peer_id, success),
                    Frame::Nack { .. } => record_nack(&peer_id),
                    Frame::Closed => break,
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
