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
//!     [`handle_frame`], the
//!     same way the dialer handles its own; the idle timeout around
//!     `incoming.next()` lives in
//!     [`next_frame`].
//!   - **Inbound-sender task** — every 1 Hz, reads the shared
//!     [`RoundBatch`](crate::kv::RoundBatch) slot produced by
//!     [`gossip_controller`](crate::gossip_controller)'s event loop and
//!     runs [`send_round`]
//!     for the learned peer.
//!
//! Asymmetry with the outbound side: the dialer (gossip_controller)
//! knows its counterparty by name at task-spawn time. The acceptor
//! here does not — the only way to associate the in-flight TCP/gRPC
//! connection with a logical mesh identity (today, pre-mTLS-derived
//! identity) is to read the first inbound frame's `peer_id`. That
//! learning step is what `learned_peer` exists for.

use std::{collections::btree_map::Entry, net::SocketAddr, pin::Pin, sync::Arc, time::Duration};

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
    metrics::{peer_connection_opened, record_ack, record_nack, record_peer_reconnect},
    mtls::MTLSManager,
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
            let is_failure_for_self = node.name == self.self_name
                && (node.status == NodeStatus::Down as i32
                    || node.status == NodeStatus::Suspected as i32);
            match state.entry(node.name.clone()) {
                Entry::Occupied(mut entry) => {
                    let current = entry.get_mut();
                    if is_failure_for_self {
                        // A node's local process state is authoritative for itself.
                        if current.status != NodeStatus::Leaving as i32
                            && current.status != NodeStatus::Alive as i32
                        {
                            current.status = NodeStatus::Alive as i32;
                            updated = true;
                        }
                        if node.version >= current.version {
                            let version = node.version.saturating_add(1);
                            if version != current.version {
                                current.version = version;
                                updated = true;
                            }
                        }
                    } else if node.version > current.version
                        || (node.version == current.version
                            && node.status == NodeStatus::Alive as i32
                            && (current.status == NodeStatus::Down as i32
                                || current.status == NodeStatus::Suspected as i32))
                    {
                        *current = node;
                        updated = true;
                    }
                }
                Entry::Vacant(entry) => {
                    updated = true;
                    if is_failure_for_self {
                        entry.insert(NodeState {
                            name: self.self_name.clone(),
                            address: self.advertise_addr.to_string(),
                            status: NodeStatus::Alive as i32,
                            version: node.version.saturating_add(1),
                            metadata: Default::default(),
                        });
                    } else {
                        entry.insert(node);
                    }
                }
            }
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
            // The dialer's name comes with its first frame; the stream is
            // counted under that name only, so no series carries a
            // placeholder for the time before the name is known. The count
            // lives in `counted` and goes when the task ends, however it ends.
            let mut peer_id = String::new();
            let mut counted = None;
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
                        counted = Some(peer_connection_opened(&peer_id));
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

            // The stream is over: its count goes with the guard, and its end
            // is counted under the name it was learned as. A stream that
            // closed before its first frame named the dialer was never a peer
            // connection: nothing is recorded for it.
            if counted.take().is_some() {
                record_peer_reconnect(&peer_id);
            }
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

#[cfg(test)]
mod tests {
    //! The acceptor's connection accounting, read the way an operator reads
    //! it: the text of a Prometheus recorder installed for the test thread
    //! (the gateway installs the same recorder type for the process), with
    //! the mesh nodes on a current-thread runtime so every task of theirs
    //! records into it.

    use std::{collections::BTreeMap, future::Future, net::SocketAddr, sync::Arc, time::Duration};

    use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
    use parking_lot::RwLock;
    use tokio::sync::mpsc;
    use tokio_stream::{wrappers::ReceiverStream, StreamExt};

    use super::GossipService;
    use crate::{
        service::gossip::{gossip_client::GossipClient, NodeState, NodeStatus, StreamMessage},
        tests::test_utils::{bind_node, wait_for},
        transport::sync_stream::build_heartbeat,
        MeshServerBuilder, MeshServerHandler,
    };

    const PEER_GAUGE: &str = "router_mesh_peer_connections";

    /// Runs `scenario` on a current-thread runtime under a Prometheus
    /// recorder installed for this thread; the rendered text, the way
    /// `/metrics` serves it, once the scenario is done.
    fn metrics_of<F, Fut>(scenario: F) -> String
    where
        F: FnOnce(PrometheusHandle) -> Fut,
        Fut: Future<Output = ()>,
    {
        let (rendered, ()) = metrics_and_result_of(scenario);
        rendered
    }

    /// [`metrics_of`] for a scenario with a result of its own: the rendered
    /// text and that result.
    fn metrics_and_result_of<F, Fut, T>(scenario: F) -> (String, T)
    where
        F: FnOnce(PrometheusHandle) -> Fut,
        Fut: Future<Output = T>,
    {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let result = metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(scenario(handle.clone()))
        });
        (handle.render(), result)
    }

    /// The peer gauge's lines of a rendered scrape.
    fn peer_gauge_lines(rendered: &str) -> Vec<&str> {
        rendered
            .lines()
            .filter(|line| line.starts_with(PEER_GAUGE))
            .collect()
    }

    #[test]
    fn merge_state_refutes_down_for_self_but_merges_other_members() {
        let local_addr: SocketAddr = "127.0.0.1:8001".parse().unwrap();
        let state = Arc::new(RwLock::new(BTreeMap::from([
            (
                "node-a".to_string(),
                NodeState {
                    name: "node-a".to_string(),
                    address: local_addr.to_string(),
                    status: NodeStatus::Alive as i32,
                    version: 2,
                    metadata: Default::default(),
                },
            ),
            (
                "node-b".to_string(),
                NodeState {
                    name: "node-b".to_string(),
                    address: "127.0.0.1:8002".to_string(),
                    status: NodeStatus::Alive as i32,
                    version: 1,
                    metadata: Default::default(),
                },
            ),
        ])));
        let service = GossipService::new(state.clone(), local_addr, local_addr, "node-a");

        assert!(service.merge_state(vec![
            NodeState {
                name: "node-a".to_string(),
                address: local_addr.to_string(),
                status: NodeStatus::Down as i32,
                version: 5,
                metadata: Default::default(),
            },
            NodeState {
                name: "node-b".to_string(),
                address: "127.0.0.1:8002".to_string(),
                status: NodeStatus::Down as i32,
                version: 3,
                metadata: Default::default(),
            },
        ]));

        let merged = state.read();
        let local = merged.get("node-a").unwrap();
        assert_eq!(local.status, NodeStatus::Alive as i32);
        assert_eq!(local.version, 6);
        let peer = merged.get("node-b").unwrap();
        assert_eq!(peer.status, NodeStatus::Down as i32);
        assert_eq!(peer.version, 3);
    }

    #[test]
    fn merge_state_refutes_max_version_down_for_self_without_overflow() {
        let local_addr: SocketAddr = "127.0.0.1:8001".parse().unwrap();
        let state = Arc::new(RwLock::new(BTreeMap::from([(
            "node-a".to_string(),
            NodeState {
                name: "node-a".to_string(),
                address: local_addr.to_string(),
                status: NodeStatus::Down as i32,
                version: 4,
                metadata: Default::default(),
            },
        )])));
        let service = GossipService::new(state.clone(), local_addr, local_addr, "node-a");

        assert!(service.merge_state(vec![NodeState {
            name: "node-a".to_string(),
            address: local_addr.to_string(),
            status: NodeStatus::Down as i32,
            version: u64::MAX,
            metadata: Default::default(),
        }]));

        let local = state.read();
        let node = local.get("node-a").unwrap();
        assert_eq!(node.status, NodeStatus::Alive as i32);
        assert_eq!(node.version, u64::MAX);
    }

    #[test]
    fn merge_state_refutes_suspected_for_self_and_advances_version() {
        let local_addr: SocketAddr = "127.0.0.1:8001".parse().unwrap();
        let state = Arc::new(RwLock::new(BTreeMap::from([(
            "node-a".to_string(),
            NodeState {
                name: "node-a".to_string(),
                address: local_addr.to_string(),
                status: NodeStatus::Alive as i32,
                version: 2,
                metadata: Default::default(),
            },
        )])));
        let service = GossipService::new(state.clone(), local_addr, local_addr, "node-a");

        assert!(service.merge_state(vec![NodeState {
            name: "node-a".to_string(),
            address: local_addr.to_string(),
            status: NodeStatus::Suspected as i32,
            version: 5,
            metadata: Default::default(),
        }]));

        let local = state.read();
        let node = local.get("node-a").unwrap();
        assert_eq!(node.status, NodeStatus::Alive as i32);
        assert_eq!(node.version, 6);
    }

    #[test]
    fn merge_state_accepts_alive_over_down_at_equal_max_version() {
        let local_addr: SocketAddr = "127.0.0.1:8001".parse().unwrap();
        let state = Arc::new(RwLock::new(BTreeMap::from([(
            "node-b".to_string(),
            NodeState {
                name: "node-b".to_string(),
                address: "127.0.0.1:8002".to_string(),
                status: NodeStatus::Down as i32,
                version: u64::MAX,
                metadata: Default::default(),
            },
        )])));
        let service = GossipService::new(state.clone(), local_addr, local_addr, "node-a");

        assert!(service.merge_state(vec![NodeState {
            name: "node-b".to_string(),
            address: "127.0.0.1:8002".to_string(),
            status: NodeStatus::Alive as i32,
            version: u64::MAX,
            metadata: Default::default(),
        }]));

        let peer = state.read();
        let node = peer.get("node-b").unwrap();
        assert_eq!(node.status, NodeStatus::Alive as i32);
        assert_eq!(node.version, u64::MAX);
    }

    #[test]
    fn merge_state_accepts_alive_over_suspected_at_equal_version() {
        let local_addr: SocketAddr = "127.0.0.1:8001".parse().unwrap();
        let state = Arc::new(RwLock::new(BTreeMap::from([(
            "node-b".to_string(),
            NodeState {
                name: "node-b".to_string(),
                address: "127.0.0.1:8002".to_string(),
                status: NodeStatus::Suspected as i32,
                version: 5,
                metadata: Default::default(),
            },
        )])));
        let service = GossipService::new(state.clone(), local_addr, local_addr, "node-a");

        assert!(service.merge_state(vec![NodeState {
            name: "node-b".to_string(),
            address: "127.0.0.1:8002".to_string(),
            status: NodeStatus::Alive as i32,
            version: 5,
            metadata: Default::default(),
        }]));

        let peer = state.read();
        let node = peer.get("node-b").unwrap();
        assert_eq!(node.status, NodeStatus::Alive as i32);
        assert_eq!(node.version, 5);
    }

    /// A mesh node on an ephemeral loopback port, joining `init_peer` when
    /// given.
    async fn node(name: &str, init_peer: Option<SocketAddr>) -> (SocketAddr, MeshServerHandler) {
        let (listener, addr) = bind_node().await;
        let (server, handler) =
            MeshServerBuilder::new(name.to_string(), addr, addr, init_peer).build();
        #[expect(
            clippy::disallowed_methods,
            reason = "test node runs in the background for the duration of the assertion"
        )]
        tokio::spawn(async move {
            if let Err(e) = server.start_with_listener(listener).await {
                tracing::error!("Mesh node failed: {e}");
            }
        });
        (addr, handler)
    }

    /// Two members: A, the lower name, dials the sync stream; B accepts it
    /// and records the peer gauge. The gauge appears under A's name and
    /// under nothing else: no series with an empty `peer` for the time
    /// before A's first frame named it.
    #[test]
    fn the_peer_gauge_carries_the_dialers_name_and_no_empty_label() {
        let rendered = metrics_of(|metrics| async move {
            let (addr_a, handler_a) = node("A", None).await;
            let (_addr_b, handler_b) = node("B", Some(addr_a)).await;
            wait_for(
                || {
                    peer_gauge_lines(&metrics.render())
                        .iter()
                        .any(|line| line.contains("peer=\"A\"") && line.ends_with(" 1"))
                },
                Duration::from_secs(20),
                "B accepts A's sync stream",
            )
            .await;
            handler_a.shutdown();
            handler_b.shutdown();
        });

        let lines = peer_gauge_lines(&rendered);
        assert!(
            !lines.iter().any(|line| line.contains("peer=\"\"")),
            "a peer series without a name: {lines:?}"
        );
    }

    /// A stream that closes before its first frame named the dialer was
    /// never a peer connection: neither the gauge nor the reconnect count is
    /// recorded for it, so no series with an empty `peer` is left behind.
    #[test]
    fn a_stream_closed_before_naming_its_peer_leaves_no_series() {
        let rendered = metrics_of(|_| async move {
            let (addr, handler) = node("A", None).await;
            wait_for(
                || std::net::TcpStream::connect(addr).is_ok(),
                Duration::from_secs(5),
                "mesh listener started",
            )
            .await;
            let mut client = GossipClient::connect(format!("http://{addr}"))
                .await
                .expect("connect to the mesh listener");
            // An outbound stream that ends at once: the acceptor reads no
            // frame. Its response stream ends once its handler has run its
            // close path, which is what the assertion is about.
            let response = client
                .sync_stream(tokio_stream::iter(Vec::<StreamMessage>::new()))
                .await
                .expect("open the sync stream");
            let mut inbound = response.into_inner();
            tokio::time::timeout(Duration::from_secs(10), async {
                while inbound.next().await.is_some() {}
            })
            .await
            .expect("the acceptor closes its side of a stream the dialer ended");
            handler.shutdown();
        });

        assert!(
            !rendered.contains("peer=\"\""),
            "a series without a peer name:\n{rendered}"
        );
    }

    /// A stream whose dialer named itself and then ended: the acceptor's
    /// series for that name reads 1 while the stream is open and 0 after,
    /// the closed stream is counted under that name, and no series carries
    /// an empty `peer` at any point.
    #[test]
    fn a_named_stream_that_closed_reads_zero_under_its_name() {
        let rendered = metrics_of(|_| async move {
            let (addr, handler) = node("A", None).await;
            wait_for(
                || std::net::TcpStream::connect(addr).is_ok(),
                Duration::from_secs(5),
                "mesh listener started",
            )
            .await;
            let mut client = GossipClient::connect(format!("http://{addr}"))
                .await
                .expect("connect to the mesh listener");
            // One heartbeat naming the dialer, then the end of the stream.
            let response = client
                .sync_stream(tokio_stream::iter(vec![build_heartbeat(1, "probe")]))
                .await
                .expect("open the sync stream");
            let mut inbound = response.into_inner();
            tokio::time::timeout(Duration::from_secs(10), async {
                while inbound.next().await.is_some() {}
            })
            .await
            .expect("the acceptor closes its side of a stream the dialer ended");
            handler.shutdown();
        });

        assert!(
            rendered.contains("router_mesh_peer_connections{peer=\"probe\"} 0"),
            "the closed stream's series does not read 0 under its name:\n{rendered}"
        );
        assert!(
            rendered.contains("router_mesh_peer_reconnects_total{peer=\"probe\"} 1"),
            "the closed stream is not counted under its name:\n{rendered}"
        );
        assert!(
            !rendered.contains("peer=\"\""),
            "a series without a peer name:\n{rendered}"
        );
    }

    /// The peer gauge's line for `peer` in a rendered scrape.
    fn peer_gauge_line(rendered: &str, peer: &str) -> Option<String> {
        let prefix = format!("{PEER_GAUGE}{{peer=\"{peer}\"}} ");
        rendered
            .lines()
            .find(|line| line.starts_with(&prefix))
            .map(str::to_owned)
    }

    /// Two accepted streams from one dialer overlap, as they do when the
    /// dialer redials before the acceptor's handler of the old stream has
    /// timed out: the gauge counts the open streams, so it reads 2 while
    /// both are open and 1 once the first closed, not 0 while a stream is
    /// live.
    #[test]
    fn overlapping_streams_of_one_dialer_add_up() {
        let (_, (while_both_open, after_first_closed)) =
            metrics_and_result_of(|metrics| async move {
                let (addr, handler) = node("A", None).await;
                wait_for(
                    || std::net::TcpStream::connect(addr).is_ok(),
                    Duration::from_secs(5),
                    "mesh listener started",
                )
                .await;
                let mut client = GossipClient::connect(format!("http://{addr}"))
                    .await
                    .expect("connect to the mesh listener");
                // Each stream names the dialer with one heartbeat; the acceptor's
                // answer to it is the sign that its handler learned the name.
                let mut open = Vec::new();
                for _ in 0..2 {
                    let (tx, rx) = mpsc::channel::<StreamMessage>(4);
                    let mut inbound = client
                        .sync_stream(ReceiverStream::new(rx))
                        .await
                        .expect("open the sync stream")
                        .into_inner();
                    tx.send(build_heartbeat(1, "probe"))
                        .await
                        .expect("send the heartbeat");
                    tokio::time::timeout(Duration::from_secs(10), inbound.next())
                        .await
                        .expect("the acceptor answers the heartbeat")
                        .expect("a frame")
                        .expect("a frame, not a status");
                    open.push((tx, inbound));
                }
                let while_both_open = metrics.render();

                // The first stream ends; its response stream ends once the
                // acceptor's handler has run its close path.
                let (tx_first, mut inbound_first) = open.remove(0);
                drop(tx_first);
                tokio::time::timeout(Duration::from_secs(10), async {
                    while inbound_first.next().await.is_some() {}
                })
                .await
                .expect("the acceptor closes its side of a stream the dialer ended");
                let after_first_closed = metrics.render();

                drop(open);
                handler.shutdown();
                (while_both_open, after_first_closed)
            });

        assert_eq!(
            peer_gauge_line(&after_first_closed, "probe").as_deref(),
            Some("router_mesh_peer_connections{peer=\"probe\"} 1"),
            "one stream is still open after the first closed:\n{after_first_closed}"
        );
        assert_eq!(
            peer_gauge_line(&while_both_open, "probe").as_deref(),
            Some("router_mesh_peer_connections{peer=\"probe\"} 2"),
            "two streams were open:\n{while_both_open}"
        );
    }
}
