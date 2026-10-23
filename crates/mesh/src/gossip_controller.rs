//! Node-wide gossip event loop and outbound stream management.
//!
//! Despite the name, this file does more than outbound gossip. It owns the
//! single per-node tick (1 Hz) that drives several responsibilities:
//!
//! 1. **SWIM-style peer probing.** Picks a peer from cluster state, sends a
//!    `Ping` (with `ping_req` indirect-probe fallback), updates membership.
//! 2. **Outbound `sync_stream` lifecycle.** Dials peers that need a stream,
//!    spawns the per-peer outbound sender task, retries with exponential
//!    backoff on failure, garbage-collects finished connections.
//! 3. **Round collection (node-wide housekeeping).** Drains the local
//!    [`MeshKV`](crate::kv::MeshKV) into a fresh [`RoundBatch`](crate::kv::RoundBatch)
//!    and publishes it into a shared `Arc<RwLock<Arc<RoundBatch>>>` slot
//!    that is read by BOTH the outbound senders here AND the inbound
//!    senders in [`gossip_service`](crate::gossip_service). This step is
//!    not outbound-specific — it produces the shared per-round data that
//!    every outgoing stream (in either direction) consumes.
//! 4. **Periodic housekeeping**: chunk-assembler GC, retry-manager pruning.
//!
//! Per-peer outbound stream tasks also live here. They are spawned by
//! [`GossipController::event_loop`] when this node initiates a stream to a
//! peer. The peer's name is captured as a `String` at task-spawn time, so
//! these senders never need to learn peer identity at runtime — contrast
//! with the inbound senders in `gossip_service.rs`, which must learn the
//! counterparty from the first inbound frame. What both ends do with an open
//! stream (the sender's round, the inbound frames) is
//! [`transport::peer_stream`](crate::transport::peer_stream).

use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use anyhow::Result;
use parking_lot::RwLock;
use rand::seq::{IndexedRandom, SliceRandom};
use tokio::sync::{mpsc, watch, Mutex};
use tonic::transport::Endpoint;
use tracing as log;
use tracing::{instrument, Instrument};

use super::{
    mtls::{tls_server_name, MTLSManager},
    service::{
        broadcast_node_states,
        gossip::{
            gossip_client::GossipClient, gossip_message, NodeState, NodeStatus, NodeUpdate, Ping,
            PingReq, StateSync, StreamMessage,
        },
        try_ping, ClusterState,
    },
};
use crate::{
    crdt_kv::CrdtWatermark,
    metrics,
    partition::PartitionDetector,
    transport::{
        limits::MAX_MESSAGE_SIZE,
        peer_stream::{handle_frame, next_frame, send_round, Frame, RoundOutcome, SenderState},
        sync_stream::build_heartbeat,
    },
};

/// The name the configured join peer (`--mesh-peer-urls`) goes by until
/// its answer names it: in the dial's log lines and in the retry table.
const INIT_PEER_NAME: &str = "init_peer";

/// The per-node event loop driver. Holds the cluster-state reference,
/// self-identity, init-peer address, mTLS config, the live set of
/// outbound sync_stream task handles, and the shared `RoundBatch` slot.
///
/// One instance per mesh node. See module docs for the full set of
/// responsibilities driven by [`event_loop`](Self::event_loop).
pub struct GossipController {
    state: ClusterState,
    self_name: String,
    self_addr: SocketAddr,
    init_peer: Option<SocketAddr>,
    mtls_manager: Option<Arc<MTLSManager>>,
    // Track active sync_stream connections
    sync_connections: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
    /// Current stream round batch, drained once per round from MeshKV.
    /// Per-peer senders read this and filter targeted entries to their
    /// own peer; drain_entries are broadcast to every peer.
    current_stream_batch: Arc<RwLock<Arc<crate::kv::RoundBatch>>>,
    /// Node-wide MeshKV handle. Owns the stream buffers, subscriber
    /// registry, and chunk assembler shared with the server-side
    /// SyncStream handlers.
    mesh_kv: Option<Arc<crate::kv::MeshKV>>,
    partition_detector: Option<Arc<PartitionDetector>>,
}

impl GossipController {
    pub fn new(
        state: ClusterState,
        self_addr: SocketAddr,
        self_name: &str,
        init_peer: Option<SocketAddr>,
        mtls_manager: Option<Arc<MTLSManager>>,
    ) -> Self {
        Self {
            state,
            self_name: self_name.to_string(),
            self_addr,
            init_peer,
            mtls_manager,
            sync_connections: Arc::new(Mutex::new(HashMap::new())),
            current_stream_batch: Arc::new(RwLock::new(Arc::new(crate::kv::RoundBatch::default()))),
            mesh_kv: None,
            partition_detector: None,
        }
    }

    /// Attach the node-wide MeshKV handle. Plumbed from the server
    /// builder so stream buffers, subscribers, and the chunk assembler
    /// are shared between client-side (outbound) and server-side
    /// (inbound) SyncStream handlers.
    pub fn with_mesh_kv(mut self, mesh_kv: Arc<crate::kv::MeshKV>) -> Self {
        self.mesh_kv = Some(mesh_kv);
        self
    }

    pub fn with_partition_detector(mut self, partition_detector: Arc<PartitionDetector>) -> Self {
        self.partition_detector = Some(partition_detector);
        self
    }

    /// Get a handle to the shared stream RoundBatch. Used by GossipService
    /// so server-side sync_stream handlers see the same drained stream
    /// entries as client-side handlers.
    pub fn current_stream_batch(&self) -> Arc<RwLock<Arc<crate::kv::RoundBatch>>> {
        self.current_stream_batch.clone()
    }

    #[instrument(fields(name = %self.self_name), skip(self, signal))]
    pub async fn event_loop(self, mut signal: watch::Receiver<bool>) -> Result<()> {
        let mut round: u64 = 0;
        let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();

        loop {
            log::info!("Round {} Status:{:?}", round, self.state.read());
            self.reap_finished_streams().await;

            if let Some(partition_detector) = &self.partition_detector {
                partition_detector.update_last_seen(&self.self_name);
                let cluster_state = self.state.read().clone();
                partition_detector.detect_partition(&cluster_state);
            }

            let targets = self.probe_targets();
            let peer = self.select_peer(&targets, round);
            round += 1;
            self.periodic_gc(round, &targets, &mut retry_managers);
            self.collect_stream_batch();

            tokio::select! {
                _ = signal.changed() => {
                    log::info!("Gossip app_server {} at {} is shutting down", self.self_name, self.self_addr);
                    break;
                }

                () = tokio::time::sleep(Duration::from_secs(1)) => {
                    let Some(peer) = peer else {
                        log::info!("No peer address available to connect");
                        continue;
                    };
                    let outcome = Self::dial_with_backoff(
                        &mut retry_managers,
                        &peer,
                        |peer| self.connect_to_peer(peer),
                        || self.has_another_member(),
                    )
                    .await;
                    match outcome {
                        DialOutcome::Connected => {
                            log::info!("Successfully connected to peer {}", peer.name);
                        }
                        DialOutcome::JoinPeerAlone { next_delay } => {
                            log::debug!(
                                "The join peer answered but the table has nobody else to probe; next dial in {:?}",
                                next_delay
                            );
                        }
                        DialOutcome::Failed {
                            error,
                            attempt,
                            next_delay,
                        } => {
                            log::warn!(
                                "Error connecting to peer {} (attempt {}): {}. Next retry in {:?}",
                                peer.name,
                                attempt,
                                error,
                                next_delay
                            );
                        }
                        DialOutcome::BackingOff { next_delay } => {
                            log::debug!(
                                "Skipping connection to peer {} (backoff: {:?} remaining)",
                                peer.name,
                                next_delay
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // ---- The round's phases ----

    /// Drop the handles of sync-stream tasks that have ended, so a peer whose
    /// stream closed is dialled again.
    async fn reap_finished_streams(&self) {
        let mut connections = self.sync_connections.lock().await;
        connections.retain(|peer_name, handle| {
            if handle.is_finished() {
                log::info!(
                    "Sync stream connection to {} has finished, removing",
                    peer_name
                );
                false
            } else {
                true
            }
        });
    }

    /// The members this round may dial: the table without this node or
    /// members marked Leaving. Down members remain probeable for recovery.
    fn probe_targets(&self) -> BTreeMap<String, NodeState> {
        let mut targets = self.state.read().clone();
        targets.retain(|name, node| self.is_probe_target(name, node));
        targets
    }

    /// The round's housekeeping. Every fifth round (~5 s) the chunk
    /// assembler drops partial assemblies older than 30 s: partial chunks
    /// the receiver has held for a full assembly timeout are assumed lost,
    /// and the sender re-publishes on its own retry cycle with a fresh
    /// generation. Every sixtieth round (~60 s) the retry managers of peers
    /// no longer in the table go; the join peer's entry stays, as it is never
    /// in the table under its placeholder name and its backoff must not
    /// restart every minute.
    fn periodic_gc(
        &self,
        round: u64,
        targets: &BTreeMap<String, NodeState>,
        retry_managers: &mut HashMap<String, RetryManager>,
    ) {
        if round.is_multiple_of(5) {
            if let Some(mesh_kv) = &self.mesh_kv {
                mesh_kv.chunk_assembler().gc(Duration::from_secs(30));
            }
        }
        if round.is_multiple_of(60) {
            retry_managers.retain(|peer_name, _| {
                targets.contains_key(peer_name) || peer_name == INIT_PEER_NAME
            });
        }
    }

    /// Stream round collection: drain the stream namespace buffers and the
    /// drain callbacks exactly once per round (destructive) into the slot
    /// the per-peer senders read. Per-peer senders filter the targeted
    /// entries by their own peer id and broadcast the drained entries to all
    /// peers. Nothing is collected without a MeshKV attached.
    fn collect_stream_batch(&self) {
        if let Some(mesh_kv) = &self.mesh_kv {
            let stream_batch = mesh_kv.collect_round_batch();
            *self.current_stream_batch.write() = Arc::new(stream_batch);
        }
    }

    /// One dial of `peer` under its retry manager's backoff: nothing is
    /// dialled while the manager still backs off from an earlier failure;
    /// otherwise `connect` runs. Its success resets the manager, unless the
    /// peer is the join peer and `has_another_member` finds the table still
    /// without a member to probe (the join peer is this node itself, or it
    /// is leaving): that counts as an attempt, so the next dial waits out
    /// the backoff instead of coming every round. A failure records the
    /// attempt and lengthens the next wait. The outcome is what the round
    /// logs.
    async fn dial_with_backoff<F, Fut>(
        retry_managers: &mut HashMap<String, RetryManager>,
        peer: &NodeState,
        connect: F,
        has_another_member: impl FnOnce() -> bool,
    ) -> DialOutcome
    where
        F: FnOnce(NodeState) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        let retry_manager = retry_managers.entry(peer.name.clone()).or_default();
        if !retry_manager.should_retry() {
            return DialOutcome::BackingOff {
                next_delay: retry_manager.next_delay(),
            };
        }
        match connect(peer.clone()).await {
            Ok(()) if peer.name == INIT_PEER_NAME && !has_another_member() => {
                retry_manager.record_attempt();
                DialOutcome::JoinPeerAlone {
                    next_delay: retry_manager.next_delay(),
                }
            }
            Ok(()) => {
                retry_manager.reset();
                DialOutcome::Connected
            }
            Err(error) => {
                retry_manager.record_attempt();
                DialOutcome::Failed {
                    error,
                    attempt: retry_manager.attempt_count(),
                    next_delay: retry_manager.next_delay(),
                }
            }
        }
    }

    async fn connect_to_peer(&self, peer: NodeState) -> Result<()> {
        log::info!("Connecting to peer {} at {}", peer.name, peer.address);

        let read_state = self.state.clone();

        // TODO: Maybe we don't need to send the whole state.
        let state_sync = StateSync {
            nodes: read_state.read().values().cloned().collect(),
        };
        let peer_addr = peer.address.parse::<SocketAddr>()?;
        let peer_name = peer.name.clone();
        match try_ping(
            &peer,
            Some(gossip_message::Payload::Ping(Ping {
                state_sync: Some(state_sync),
            })),
            self.mtls_manager.clone(),
        )
        .await
        {
            Ok(node_update) => {
                log::info!("Received NodeUpdate from peer: {:?}", node_update);
                // Update state for Alive or Leaving status
                if let Some(updated_peer) = self.update_membership(&node_update) {
                    // If node is Alive, establish sync_stream connection with freshest address.
                    if node_update.status == NodeStatus::Alive as i32 {
                        if let Err(e) = self
                            .start_sync_stream_connection(updated_peer.clone())
                            .await
                        {
                            log::warn!(
                                "Failed to start sync_stream to {}: {}",
                                updated_peer.name,
                                e
                            );
                            // Connection failure doesn't affect ping flow, will retry in next cycle
                        }
                    }
                }
            }
            Err(e) => {
                log::info!("Failed to connect to peer: {}, now try ping-req", e);
                let mut map = read_state.read().clone();
                map.retain(|k, v| {
                    k.ne(&self.self_name)
                        && k.ne(&peer_name)
                        && v.status == NodeStatus::Alive as i32
                });
                let random_nodes = get_random_values_refs(&map, 3);
                let mut reachable = false;
                for node in random_nodes {
                    log::info!(
                        "Trying to ping-req node {}, req target: {}",
                        node.address,
                        peer_addr
                    );
                    if let Ok(node_update) = try_ping(
                        node,
                        Some(gossip_message::Payload::PingReq(PingReq {
                            node: Some(peer.clone()),
                        })),
                        self.mtls_manager.clone(),
                    )
                    .await
                    {
                        if node_update.status == NodeStatus::Alive as i32
                            && node_update.name == peer_name
                        {
                            self.update_membership(&node_update);
                            reachable = true;
                            break;
                        }
                    }
                }
                if !reachable {
                    let mut target = read_state.read().clone();

                    // Broadcast only the unreachable node's status is enough.
                    if let Some(mut unreachable_node) = target.remove(&peer_name) {
                        unreachable_node.status = if unreachable_node.status
                            == NodeStatus::Suspected as i32
                            || unreachable_node.status == NodeStatus::Down as i32
                        {
                            NodeStatus::Down as i32
                        } else {
                            NodeStatus::Suspected as i32
                        };
                        unreachable_node.version += 1;

                        // Broadcast target nodes should include self.
                        let target_nodes: Vec<NodeState> = target
                            .values()
                            .filter(|v| {
                                v.name.ne(&peer_name)
                                    && v.status == NodeStatus::Alive as i32
                                    && v.status != NodeStatus::Leaving as i32
                            })
                            .cloned()
                            .collect();

                        log::info!(
                            "Broadcasting node status to {} alive nodes, new_state: {:?}",
                            target_nodes.len(),
                            unreachable_node
                        );

                        let (success_count, total_count) = broadcast_node_states(
                            vec![unreachable_node],
                            target_nodes,
                            None, // Use default timeout
                            self.mtls_manager.clone(),
                        )
                        .await;

                        log::info!(
                            "Broadcast node status: {}/{} successful",
                            success_count,
                            total_count
                        );
                    }
                    return Err(anyhow::anyhow!(
                        "Failed to connect to peer {peer_name}: direct ping and ping-req both failed"
                    ));
                }
            }
        }

        log::info!("Successfully connected to peer {}", peer_addr);

        Ok(())
    }

    fn update_membership(&self, node_update: &NodeUpdate) -> Option<NodeState> {
        if node_update.status != NodeStatus::Alive as i32
            && node_update.status != NodeStatus::Leaving as i32
        {
            return None;
        }

        let updated_peer = {
            let mut state = self.state.write();
            let entry = state
                .entry(node_update.name.clone())
                .and_modify(|node| {
                    if node_update.status == NodeStatus::Alive as i32
                        && node.status != NodeStatus::Alive as i32
                    {
                        node.version = node.version.saturating_add(1);
                    }
                    node.status = node_update.status;
                    node.address.clone_from(&node_update.address);
                })
                .or_insert_with(|| NodeState {
                    name: node_update.name.clone(),
                    address: node_update.address.clone(),
                    status: node_update.status,
                    version: 1,
                    metadata: HashMap::new(),
                });
            entry.clone()
        };

        if node_update.status == NodeStatus::Alive as i32 {
            if let Some(partition_detector) = &self.partition_detector {
                partition_detector.update_last_seen(&node_update.name);
            }
        }

        Some(updated_peer)
    }

    /// Whether `node` is one to probe: another node that is not leaving.
    /// Down members stay probeable so a recovered peer can rejoin.
    fn is_probe_target(&self, name: &str, node: &NodeState) -> bool {
        name != self.self_name && node.status != NodeStatus::Leaving as i32
    }

    /// Whether the membership table holds a node to probe.
    fn has_another_member(&self) -> bool {
        self.state.read().iter().any(|(name, node)| {
            self.is_probe_target(name, node) && node.status != NodeStatus::Down as i32
        })
    }

    fn select_peer(
        &self,
        candidates: &BTreeMap<String, NodeState>,
        round: u64,
    ) -> Option<NodeState> {
        let has_non_down_member = candidates
            .values()
            .any(|node| node.status != NodeStatus::Down as i32);
        let dial_init_peer = candidates.is_empty()
            || (self.init_peer.is_some() && !has_non_down_member && round.is_multiple_of(2));

        if dial_init_peer {
            self.init_peer.map(|init_peer| NodeState {
                name: INIT_PEER_NAME.to_string(),
                address: init_peer.to_string(),
                status: NodeStatus::Suspected as i32,
                version: 1,
                metadata: HashMap::new(),
            })
        } else {
            let random_nodes = get_random_values_refs(candidates, 1);
            random_nodes.first().map(|&node| node.clone())
        }
    }

    /// Determine if this node should initiate sync_stream connection
    /// Use lexicographic ordering to avoid duplicate connections
    fn should_initiate_connection(&self, peer_name: &str) -> bool {
        self.self_name.as_str() < peer_name
    }

    /// Spawn a task to handle sync_stream messages
    fn spawn_sync_stream_handler(
        &self,
        mut incoming_stream: tonic::Streaming<StreamMessage>,
        tx: mpsc::Sender<StreamMessage>,
        self_name: String,
        peer_name: String,
    ) -> tokio::task::JoinHandle<()> {
        let sync_connections = self.sync_connections.clone();
        let current_stream_batch = self.current_stream_batch.clone();
        let mesh_kv = self.mesh_kv.clone();

        // Log connection lifecycle: spawn
        log::debug!(
            peer = %peer_name,
            "spawn_sync_stream_handler called — spawning handler task"
        );

        // Create a span for the spawned task
        let span = tracing::info_span!(
            "sync_stream_handler",
            peer = %peer_name
        );

        #[expect(clippy::disallowed_methods, reason = "handle is returned to caller (spawn_sync_stream_handler) and stored in sync_connections map for lifecycle tracking")]
        tokio::spawn(
            async move {
                // Log active connection count at handler start
                let active_connections = sync_connections.lock().await.len();
                log::debug!(
                    peer = %peer_name,
                    active_connections,
                    "Sync stream handler started"
                );

                // This end's frame counter: every frame sent or received
                // advances it.
                let sequence = Arc::new(AtomicU64::new(0));

                // Per-peer CRDT send watermark (per-key acked versions). The
                // sender filters by it; the inbound loop advances it on CrdtAck.
                let acked: Arc<RwLock<CrdtWatermark>> =
                    Arc::new(RwLock::new(CrdtWatermark::new()));

                // Send initial heartbeat
                let heartbeat =
                    build_heartbeat(sequence.fetch_add(1, Ordering::Relaxed), &self_name);
                if tx.send(heartbeat).await.is_err() {
                    log::warn!("Failed to send initial heartbeat to {}", peer_name);
                    return;
                }

                // The periodic sender: the shared round every second, with
                // this end's round metrics around it. The peer is known from
                // the start, so every round can carry its targeted entries.
                let incremental_sender_handle = {
                    let tx_incremental = tx.clone();
                    let self_name_incremental = self_name.clone();
                    let peer_name_incremental = peer_name.clone();
                    let shared_sequence = sequence.clone();
                    let stream_batch_handle = current_stream_batch.clone();
                    let acked_incremental = acked.clone();

                    #[expect(clippy::disallowed_methods, reason = "incremental sender handle is stored and aborted when the parent sync_stream handler exits")]
                    tokio::spawn(async move {
                        let mut interval = tokio::time::interval(Duration::from_secs(1));
                        let mut sender = SenderState::default();

                        loop {
                            interval.tick().await;

                            let round_start = Instant::now();
                            let stream_batch = stream_batch_handle.read().clone();
                            let outcome = send_round(
                                &mut sender,
                                &stream_batch,
                                Some(&peer_name_incremental),
                                &acked_incremental,
                                &self_name_incremental,
                                &mut || shared_sequence.fetch_add(1, Ordering::Relaxed),
                                &tx_incremental,
                            );
                            if outcome == RoundOutcome::Closed {
                                log::warn!(
                                    peer = %peer_name_incremental,
                                    "stream sender: channel closed, stopping"
                                );
                                return;
                            }

                            let round_elapsed = round_start.elapsed();
                            metrics::record_sync_round_duration(
                                &peer_name_incremental,
                                round_elapsed,
                            );
                            if round_elapsed.as_millis() > 10 {
                                log::info!(
                                    peer = %peer_name_incremental,
                                    round_ms = round_elapsed.as_millis(),
                                    "mesh sync round"
                                );
                            }
                        }
                    })
                };

                // Handle incoming messages
                while let Some(msg) = next_frame(&mut incoming_stream, &peer_name).await {
                    sequence.fetch_add(1, Ordering::Relaxed);
                    let frame = handle_frame(
                        msg,
                        &peer_name,
                        &self_name,
                        mesh_kv.as_deref(),
                        &acked,
                        &mut || sequence.fetch_add(1, Ordering::Relaxed),
                        &tx,
                    )
                    .await;
                    match frame {
                        Frame::Continue => {}
                        Frame::Ack { sequence, .. } => {
                            log::trace!("Received ACK from {} (seq: {})", peer_name, sequence);
                        }
                        Frame::Nack { sequence } => {
                            log::warn!("Received NACK from {} (seq: {})", peer_name, sequence);
                        }
                        Frame::Closed => break,
                    }
                }

                incremental_sender_handle.abort();
                let _ = incremental_sender_handle.await;
                log::debug!(
                    peer = %peer_name,
                    "sync_stream_handler exited — handler dropped"
                );
            }
            .instrument(span),
        )
    }

    /// Start a sync_stream connection to a peer
    async fn start_sync_stream_connection(&self, peer: NodeState) -> Result<()> {
        let peer_name = peer.name.clone();
        let peer_addr = peer.address.clone();

        // Check if connection already exists
        {
            let connections = self.sync_connections.lock().await;
            if connections.contains_key(&peer_name) {
                log::debug!("Sync stream connection to {} already exists", peer_name);
                return Ok(());
            }
        }

        // Check if we should initiate connection (avoid duplicates)
        if !self.should_initiate_connection(&peer_name) {
            log::debug!(
                "Skipping sync_stream to {} (peer should initiate)",
                peer_name
            );
            return Ok(());
        }

        log::info!(
            "Starting sync_stream connection to peer {} at address {}",
            peer_name,
            peer_addr
        );

        // Connect to peer's gRPC service via Endpoint so TLS can be configured.
        let connect_url = if self.mtls_manager.is_some() {
            format!("https://{peer_addr}")
        } else {
            format!("http://{peer_addr}")
        };
        log::info!("Connecting to URL: {}", connect_url);

        let mut endpoint = Endpoint::from_shared(connect_url.clone())
            .map_err(|e| anyhow::anyhow!("Invalid peer endpoint {connect_url}: {e}"))?;

        if let Some(mtls_manager) = self.mtls_manager.clone() {
            let tls_domain = endpoint.uri().host().map_or_else(
                || peer_name.clone(),
                |host| tls_server_name(host).to_owned(),
            );
            let tls = mtls_manager
                .client_tls_config(&tls_domain)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to load the mTLS identity: {e}"))?;
            endpoint = endpoint
                .tls_config(tls)
                .map_err(|e| anyhow::anyhow!("Failed to configure TLS endpoint: {e}"))?;
        }

        let channel = endpoint.connect().await.map_err(|e| {
            log::warn!(
                "Failed to connect to peer {} for sync_stream: {}",
                peer_name,
                e
            );
            anyhow::anyhow!("Connection failed: {e}")
        })?;
        let mut client = GossipClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_MESSAGE_SIZE)
            .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
            .send_compressed(tonic::codec::CompressionEncoding::Gzip);

        // Create bidirectional stream
        let (tx, rx) = mpsc::channel::<StreamMessage>(128);
        let outgoing_stream = tokio_stream::wrappers::ReceiverStream::new(rx);

        let response = client.sync_stream(outgoing_stream).await.map_err(|e| {
            log::error!("Failed to establish sync_stream with {}: {}", peer_name, e);
            anyhow::anyhow!("sync_stream RPC failed: {e}")
        })?;

        let incoming_stream = response.into_inner();

        // Spawn task to handle the bidirectional stream
        let self_name = self.self_name.clone();
        let peer_name_clone = peer_name.clone();

        let handle =
            self.spawn_sync_stream_handler(incoming_stream, tx, self_name, peer_name_clone);

        // Store the task handle
        {
            let mut connections = self.sync_connections.lock().await;
            connections.insert(peer_name.clone(), handle);
        }

        log::info!("Sync stream connection to {} established", peer_name);
        Ok(())
    }
}

// TODO: Support weighted random selection. e.g. nodes in INIT state should be more likely to be selected.
fn get_random_values_refs<K, V>(map: &BTreeMap<K, V>, k: usize) -> Vec<&V> {
    let values: Vec<&V> = map.values().collect();

    if k >= values.len() {
        let mut all_values = values;
        all_values.shuffle(&mut rand::rng());
        return all_values;
    }

    let mut rng = rand::rng();

    values.sample(&mut rng, k).copied().collect()
}

/// What one round's dial of a peer came to (see
/// `GossipController::dial_with_backoff`).
#[derive(Debug)]
enum DialOutcome {
    /// The peer answered; its retry state is reset.
    Connected,
    /// The join peer answered but added no member to probe; the attempt is
    /// recorded and the next dial waits `next_delay`.
    JoinPeerAlone { next_delay: Duration },
    /// The dial failed; the attempt is recorded and the next one waits
    /// `next_delay`.
    Failed {
        error: anyhow::Error,
        attempt: u32,
        next_delay: Duration,
    },
    /// Still inside the backoff of an earlier failure; nothing was dialled.
    /// `next_delay` is the slot's whole wait, not what is left of it (the
    /// round's line has always reported it as the backoff remaining).
    BackingOff { next_delay: Duration },
}

/// Exponential backoff calculator used by the per-peer reconnect loop.
#[derive(Debug, Clone)]
struct ExponentialBackoff {
    initial_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
}

impl ExponentialBackoff {
    fn new(initial_delay: Duration, max_delay: Duration, multiplier: f64) -> Self {
        Self {
            initial_delay,
            max_delay,
            multiplier,
        }
    }

    /// Delay for attempt number (0-indexed).
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let max_delay_secs = self.max_delay.as_secs_f64();
        let delay_secs = self.initial_delay.as_secs_f64()
            * self.multiplier.powi(attempt.min(i32::MAX as u32) as i32);
        // Guard against f64 overflow to infinity which would panic in
        // Duration::from_secs_f64.
        let capped = if delay_secs.is_finite() && delay_secs >= 0.0 {
            delay_secs.min(max_delay_secs)
        } else {
            max_delay_secs
        };
        Duration::from_secs_f64(capped)
    }
}

impl Default for ExponentialBackoff {
    fn default() -> Self {
        Self::new(Duration::from_secs(1), Duration::from_secs(60), 2.0)
    }
}

/// Per-peer reconnect state tracker with exponential backoff. Owned
/// by the controller's `HashMap<peer_name, RetryManager>` and only
/// touched on the controller's task, so plain `&mut self` mutation
/// is sufficient — no interior mutability needed.
#[derive(Debug, Default)]
struct RetryManager {
    backoff: ExponentialBackoff,
    last_attempt: Option<Instant>,
    attempt_count: u32,
}

impl RetryManager {
    /// Whether enough time has elapsed since the last attempt to retry.
    /// `attempt_count` counts *completed* attempts; the next retry's
    /// delay slot is therefore the zero-indexed `attempt_count - 1`.
    fn should_retry(&self) -> bool {
        match self.last_attempt {
            Some(last_attempt) => last_attempt.elapsed() >= self.next_delay(),
            None => true,
        }
    }

    fn record_attempt(&mut self) {
        self.last_attempt = Some(Instant::now());
        self.attempt_count = self.attempt_count.saturating_add(1);
    }

    /// Reset on successful connection.
    fn reset(&mut self) {
        self.last_attempt = None;
        self.attempt_count = 0;
    }

    fn attempt_count(&self) -> u32 {
        self.attempt_count
    }

    fn next_delay(&self) -> Duration {
        // `attempt_count` counts completed attempts; the upcoming retry
        // is in the zero-indexed slot one below it.
        self.backoff
            .delay_for_attempt(self.attempt_count.saturating_sub(1))
    }
}

#[cfg(test)]
mod retry_manager_tests {
    use super::*;

    fn controller_with_members(
        members: BTreeMap<String, NodeState>,
        init_peer: Option<SocketAddr>,
    ) -> GossipController {
        GossipController::new(
            Arc::new(RwLock::new(members)),
            "127.0.0.1:8000".parse().unwrap(),
            "node-a",
            init_peer,
            None,
        )
    }

    fn controller() -> GossipController {
        controller_with_members(BTreeMap::new(), None)
    }

    fn node(name: &str, status: NodeStatus) -> NodeState {
        NodeState {
            name: name.to_string(),
            address: "127.0.0.1:8001".to_string(),
            status: status as i32,
            version: 1,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn down_members_remain_probeable_for_recovery() {
        let controller = controller();
        assert!(controller.is_probe_target("node-b", &node("node-b", NodeStatus::Down)));
        assert!(!controller.is_probe_target("node-b", &node("node-b", NodeStatus::Leaving)));
        assert!(!controller.is_probe_target("node-a", &node("node-a", NodeStatus::Alive)));
    }

    #[test]
    fn down_members_do_not_block_join_peer_redial_or_their_own_probes() {
        let members = BTreeMap::from([("node-b".to_string(), node("node-b", NodeStatus::Down))]);
        let candidates = members.clone();
        let init_peer = "127.0.0.1:8002".parse().unwrap();
        let controller = controller_with_members(members, Some(init_peer));

        assert!(!controller.has_another_member());

        let join_peer = controller.select_peer(&candidates, 0).unwrap();
        assert_eq!(join_peer.name, INIT_PEER_NAME);
        assert_eq!(join_peer.address, init_peer.to_string());

        let down_member = controller.select_peer(&candidates, 1).unwrap();
        assert_eq!(down_member.name, "node-b");
        assert_eq!(down_member.status, NodeStatus::Down as i32);
    }

    #[test]
    fn successful_indirect_probe_restores_quorum_membership() {
        let state = Arc::new(RwLock::new(BTreeMap::from([
            ("node-a".to_string(), node("node-a", NodeStatus::Alive)),
            ("node-b".to_string(), node("node-b", NodeStatus::Down)),
            ("node-c".to_string(), node("node-c", NodeStatus::Alive)),
        ])));
        let detector = Arc::new(PartitionDetector::new(crate::partition::PartitionConfig {
            unreachable_timeout: Duration::from_secs(30),
            min_cluster_size: 3,
            quorum_threshold: 3,
        }));
        detector.update_last_seen("node-a");
        detector.update_last_seen("node-c");
        let controller = GossipController::new(
            state.clone(),
            "127.0.0.1:8000".parse().unwrap(),
            "node-a",
            None,
            None,
        )
        .with_partition_detector(detector.clone());

        assert_eq!(
            detector.detect_partition(&state.read()),
            crate::partition::PartitionState::PartitionedWithoutQuorum
        );
        assert!(controller
            .update_membership(&NodeUpdate {
                name: "node-b".to_string(),
                address: "127.0.0.1:8002".to_string(),
                status: NodeStatus::Alive as i32,
            })
            .is_some());
        assert_eq!(
            state.read().get("node-b").unwrap().status,
            NodeStatus::Alive as i32
        );
        assert_eq!(state.read().get("node-b").unwrap().version, 2);
        controller.update_membership(&NodeUpdate {
            name: "node-b".to_string(),
            address: "127.0.0.1:8002".to_string(),
            status: NodeStatus::Alive as i32,
        });
        assert_eq!(state.read().get("node-b").unwrap().version, 2);
        assert_eq!(
            detector.detect_partition(&state.read()),
            crate::partition::PartitionState::Normal
        );
    }

    #[test]
    fn alive_recovery_at_max_version_does_not_overflow() {
        let members = BTreeMap::from([(
            "node-b".to_string(),
            NodeState {
                name: "node-b".to_string(),
                address: "127.0.0.1:8002".to_string(),
                status: NodeStatus::Down as i32,
                version: u64::MAX,
                metadata: HashMap::new(),
            },
        )]);
        let state = Arc::new(RwLock::new(members));
        let controller = GossipController::new(
            state.clone(),
            "127.0.0.1:8000".parse().unwrap(),
            "node-a",
            None,
            None,
        );

        controller.update_membership(&NodeUpdate {
            name: "node-b".to_string(),
            address: "127.0.0.1:8002".to_string(),
            status: NodeStatus::Alive as i32,
        });

        let recovered = state.read().get("node-b").unwrap().clone();
        assert_eq!(recovered.status, NodeStatus::Alive as i32);
        assert_eq!(recovered.version, u64::MAX);
    }

    #[test]
    fn first_retry_uses_initial_delay() {
        let mut mgr = RetryManager::default();
        mgr.record_attempt();
        // ExponentialBackoff::default() = (1s, 60s, 2.0)
        assert_eq!(mgr.next_delay(), Duration::from_secs(1));
    }

    #[test]
    fn subsequent_retries_double_until_capped() {
        let mut mgr = RetryManager::default();
        mgr.record_attempt();
        assert_eq!(mgr.next_delay(), Duration::from_secs(1));
        mgr.record_attempt();
        assert_eq!(mgr.next_delay(), Duration::from_secs(2));
        mgr.record_attempt();
        assert_eq!(mgr.next_delay(), Duration::from_secs(4));
        // Cap is 60s with the default config: 1 * 2^6 = 64 -> clamped.
        for _ in 0..10 {
            mgr.record_attempt();
        }
        assert_eq!(mgr.next_delay(), Duration::from_secs(60));
    }

    #[test]
    fn reset_returns_to_first_retry_state() {
        let mut mgr = RetryManager::default();
        for _ in 0..5 {
            mgr.record_attempt();
        }
        assert_ne!(mgr.next_delay(), Duration::from_secs(1));
        mgr.reset();
        assert!(mgr.should_retry(), "post-reset should always allow retry");
        mgr.record_attempt();
        assert_eq!(mgr.next_delay(), Duration::from_secs(1));
    }

    #[test]
    fn should_retry_before_any_attempt() {
        let mgr = RetryManager::default();
        assert!(mgr.should_retry());
    }
}

#[cfg(test)]
mod dial_tests {
    use std::{
        collections::HashMap,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        time::{Duration, Instant},
    };

    use super::{
        DialOutcome, GossipController, NodeState, NodeStatus, RetryManager, INIT_PEER_NAME,
    };

    fn node(name: &str) -> NodeState {
        NodeState {
            name: name.to_string(),
            address: "127.0.0.1:1".to_string(),
            status: NodeStatus::Alive as i32,
            version: 1,
            metadata: HashMap::new(),
        }
    }

    /// A connect that answers or refuses, counting its calls.
    fn connect(
        calls: &Arc<AtomicUsize>,
        answer: bool,
    ) -> impl FnOnce(NodeState) -> std::future::Ready<anyhow::Result<()>> {
        let calls = Arc::clone(calls);
        move |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            std::future::ready(if answer {
                Ok(())
            } else {
                Err(anyhow::anyhow!("refused"))
            })
        }
    }

    /// A member answering resets its retry state; a failure records the
    /// attempt and backs off; a dial inside that backoff runs no connect at
    /// all; the backoff over, the next dial runs and its success starts the
    /// peer afresh.
    #[tokio::test]
    async fn a_dial_connects_fails_or_backs_off_and_the_retry_state_follows() {
        let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();
        let peer = node("peer-a");
        let calls = Arc::new(AtomicUsize::new(0));
        let member_present = || true;

        let outcome = GossipController::dial_with_backoff(
            &mut retry_managers,
            &peer,
            connect(&calls, true),
            member_present,
        )
        .await;
        assert!(matches!(outcome, DialOutcome::Connected), "{outcome:?}");
        assert_eq!(retry_managers["peer-a"].attempt_count(), 0);

        let outcome = GossipController::dial_with_backoff(
            &mut retry_managers,
            &peer,
            connect(&calls, false),
            member_present,
        )
        .await;
        match outcome {
            DialOutcome::Failed {
                attempt,
                next_delay,
                ..
            } => {
                assert_eq!(attempt, 1);
                assert_eq!(next_delay, Duration::from_secs(1));
            }
            other => panic!("expected a failure, got {other:?}"),
        }
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        let outcome = GossipController::dial_with_backoff(
            &mut retry_managers,
            &peer,
            connect(&calls, true),
            member_present,
        )
        .await;
        assert!(
            matches!(outcome, DialOutcome::BackingOff { next_delay } if next_delay == Duration::from_secs(1)),
            "{outcome:?}"
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "a dial in backoff runs no connect"
        );
        assert_eq!(retry_managers["peer-a"].attempt_count(), 1);

        retry_managers.get_mut("peer-a").unwrap().last_attempt =
            Some(Instant::now() - Duration::from_secs(2));
        let outcome = GossipController::dial_with_backoff(
            &mut retry_managers,
            &peer,
            connect(&calls, true),
            member_present,
        )
        .await;
        assert!(matches!(outcome, DialOutcome::Connected), "{outcome:?}");
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        assert_eq!(retry_managers["peer-a"].attempt_count(), 0);
        assert!(retry_managers["peer-a"].should_retry());
    }

    /// The join peer answering with nobody else in the table counts as an
    /// attempt (the next dial waits out the backoff); answering with a
    /// member in the table is a connection like any other.
    #[tokio::test]
    async fn the_join_peer_answering_alone_records_an_attempt() {
        let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();
        let join_peer = node(INIT_PEER_NAME);
        let calls = Arc::new(AtomicUsize::new(0));

        let outcome = GossipController::dial_with_backoff(
            &mut retry_managers,
            &join_peer,
            connect(&calls, true),
            || false,
        )
        .await;
        assert!(
            matches!(outcome, DialOutcome::JoinPeerAlone { next_delay } if next_delay == Duration::from_secs(1)),
            "{outcome:?}"
        );
        assert_eq!(retry_managers[INIT_PEER_NAME].attempt_count(), 1);
        assert!(!retry_managers[INIT_PEER_NAME].should_retry());

        retry_managers.get_mut(INIT_PEER_NAME).unwrap().last_attempt =
            Some(Instant::now() - Duration::from_secs(2));
        let outcome = GossipController::dial_with_backoff(
            &mut retry_managers,
            &join_peer,
            connect(&calls, true),
            || true,
        )
        .await;
        assert!(matches!(outcome, DialOutcome::Connected), "{outcome:?}");
        assert_eq!(retry_managers[INIT_PEER_NAME].attempt_count(), 0);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    /// Two failures in a row double the wait, and each peer keeps its own
    /// retry state.
    #[tokio::test]
    async fn repeated_failures_lengthen_the_wait_per_peer() {
        let mut retry_managers: HashMap<String, RetryManager> = HashMap::new();
        let peer_a = node("peer-a");
        let peer_b = node("peer-b");
        let calls = Arc::new(AtomicUsize::new(0));

        GossipController::dial_with_backoff(
            &mut retry_managers,
            &peer_a,
            connect(&calls, false),
            || true,
        )
        .await;
        retry_managers.get_mut("peer-a").unwrap().last_attempt =
            Some(Instant::now() - Duration::from_secs(5));
        let outcome = GossipController::dial_with_backoff(
            &mut retry_managers,
            &peer_a,
            connect(&calls, false),
            || true,
        )
        .await;
        assert!(
            matches!(outcome, DialOutcome::Failed { attempt: 2, next_delay, .. } if next_delay == Duration::from_secs(2)),
            "{outcome:?}"
        );

        let outcome = GossipController::dial_with_backoff(
            &mut retry_managers,
            &peer_b,
            connect(&calls, true),
            || true,
        )
        .await;
        assert!(matches!(outcome, DialOutcome::Connected), "{outcome:?}");
        assert_eq!(retry_managers["peer-b"].attempt_count(), 0);
        assert_eq!(retry_managers["peer-a"].attempt_count(), 2);
    }
}
