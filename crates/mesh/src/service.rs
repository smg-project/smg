use std::{
    collections::{BTreeMap, HashMap},
    net::SocketAddr,
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use parking_lot::RwLock;
use tokio::sync::watch;
use tracing as log;

pub mod gossip {
    #![allow(unused_qualifications, clippy::absolute_paths)]
    #![allow(clippy::trivially_copy_pass_by_ref, clippy::allow_attributes)]
    tonic::include_proto!("mesh.gossip");
}
use gossip::{gossip_message, NodeState, NodeStatus, Ping, StateSync};

use crate::{
    dialer::Dialer,
    gossip_controller::GossipController,
    gossip_service::{GossipService, PreparedGossipServer},
    kv::MeshKV,
    mtls::{MTLSConfig, MTLSManager},
    partition::PartitionDetector,
};

pub type ClusterState = Arc<RwLock<BTreeMap<String, NodeState>>>;

pub struct MeshServerConfig {
    pub self_name: String,
    pub bind_addr: SocketAddr,
    pub advertise_addr: SocketAddr,
    /// The join peers (`--mesh-peer-urls`): while the membership table holds
    /// nobody else, they are dialed in turn, one per round, each under its
    /// own retry backoff, until one answers.
    pub init_peers: Vec<SocketAddr>,
    pub mtls_config: Option<MTLSConfig>,
}

/// MeshServerHandler
/// It is the handler for the mesh server, which is responsible for the node management.
/// Includes some basic node management logic, like shutdown,
/// node discovery(TODO), node status update(TODO), etc.
pub struct MeshServerHandler {
    pub state: ClusterState,
    pub self_name: String,
    signal_tx: watch::Sender<bool>,
    partition_detector: Option<Arc<PartitionDetector>>,
    /// Shared with the MeshServer so adapters can subscribe to stream
    /// namespaces (broadcast/targeted) and publish values that reach
    /// peers via the gossip loop.
    mesh_kv: Arc<MeshKV>,
    /// The node's dials, shared with the gossip loop: the `Leaving` broadcast
    /// goes through the channels the probes keep, with the node's TLS
    /// identity when the mesh has one.
    dialer: Arc<Dialer>,
}

impl MeshServerHandler {
    /// Get partition detector
    pub fn partition_detector(&self) -> Option<&Arc<PartitionDetector>> {
        self.partition_detector.as_ref()
    }

    /// Check if we should serve (have quorum)
    pub fn should_serve(&self) -> bool {
        self.partition_detector
            .as_ref()
            .map(|pd| pd.should_serve())
            .unwrap_or(true) // If no partition detector, consider should serve
    }

    /// Shutdown immediately without graceful shutdown
    pub fn shutdown(&self) {
        self.signal_tx.send(true).ok();
    }

    /// Graceful shutdown: broadcast LEAVING status to all alive nodes,
    /// wait for propagation, then shutdown
    pub async fn graceful_shutdown(&self) -> Result<()> {
        log::info!("Graceful shutdown for node {}", self.self_name);

        let maybe_leaving = {
            let mut state = self.state.write();
            // Broadcast targets: the live members other than this node.
            let alive_nodes = state
                .values()
                .filter(|node| {
                    node.status == NodeStatus::Alive as i32 && node.name != self.self_name
                })
                .cloned()
                .collect::<Vec<NodeState>>();
            match state.get_mut(&self.self_name) {
                Some(self_node) if self_node.status != NodeStatus::Leaving as i32 => {
                    // Written into this node's own table first: a probe it
                    // still answers before its loop stops says Leaving too,
                    // instead of turning a member's view of it back to Alive.
                    self_node.status = NodeStatus::Leaving as i32;
                    self_node.version += 1;
                    Some((self_node.clone(), alive_nodes))
                }
                _ => None,
            }
        };
        let (leaving_node, alive_nodes) = match maybe_leaving {
            Some(values) => values,
            None => {
                self.signal_tx.send(true).ok();
                return Ok(());
            }
        };

        log::info!(
            "Broadcasting LEAVING status to {} alive nodes",
            alive_nodes.len()
        );

        // Broadcast LEAVING status to all alive nodes
        let (success_count, total_count) = broadcast_node_states(
            vec![leaving_node],
            alive_nodes,
            Some(Duration::from_secs(3)),
            Arc::clone(&self.dialer),
        )
        .await;

        log::info!(
            "Broadcast LEAVING status: {}/{} successful",
            success_count,
            total_count
        );

        // Wait a bit more for state propagation
        let propagation_delay = Duration::from_secs(1);
        log::info!(
            "Waiting {} seconds for LEAVING status propagation",
            propagation_delay.as_secs()
        );
        tokio::time::sleep(propagation_delay).await;

        log::info!("Signaling shutdown");
        self.signal_tx.send(true).ok();
        Ok(())
    }

    /// Shared MeshKV handle — adapters subscribe to stream namespaces
    /// and publish values through this. The handle is Arc-cloned, so
    /// subscribers created here see the same events as the gossip loop.
    pub fn mesh_kv(&self) -> &Arc<MeshKV> {
        &self.mesh_kv
    }
}

pub struct MeshServerBuilder {
    state: ClusterState,
    self_name: String,
    bind_addr: SocketAddr,
    advertise_addr: SocketAddr,
    init_peers: Vec<SocketAddr>,
    mtls_manager: Option<Arc<MTLSManager>>,
}

impl MeshServerBuilder {
    pub fn new(
        self_name: String,
        bind_addr: SocketAddr,
        advertise_addr: SocketAddr,
        init_peers: Vec<SocketAddr>,
    ) -> Self {
        let state = Arc::new(RwLock::new(BTreeMap::from([(
            self_name.clone(),
            NodeState {
                name: self_name.clone(),
                address: advertise_addr.to_string(),
                status: NodeStatus::Alive as i32,
                version: 1,
                metadata: HashMap::new(),
            },
        )])));
        Self {
            state,
            self_name,
            bind_addr,
            advertise_addr,
            init_peers,
            mtls_manager: None,
        }
    }

    pub fn with_mtls(mut self, mtls_config: MTLSConfig) -> Self {
        self.mtls_manager = Some(Arc::new(MTLSManager::new(mtls_config)));
        self
    }

    pub fn build(&self) -> (MeshServer, MeshServerHandler) {
        let (signal_tx, signal_rx) = watch::channel(false);
        let partition_detector = Arc::new(PartitionDetector::default());
        let mesh_kv = Arc::new(MeshKV::new(self.self_name.clone()));
        // One dialer for the server, its gossip loop and the handler: every
        // dial to a peer, from any of them, goes through the same channel.
        let dialer = Dialer::new(self.mtls_manager.clone());
        (
            MeshServer {
                state: self.state.clone(),
                self_name: self.self_name.clone(),
                bind_addr: self.bind_addr,
                advertise_addr: self.advertise_addr,
                init_peers: self.init_peers.clone(),
                signal_rx,
                partition_detector: Some(partition_detector.clone()),
                dialer: Arc::clone(&dialer),
                mesh_kv: mesh_kv.clone(),
            },
            MeshServerHandler {
                state: self.state.clone(),
                self_name: self.self_name.clone(),
                signal_tx,
                partition_detector: Some(partition_detector),
                mesh_kv,
                dialer,
            },
        )
    }
}

impl From<&MeshServerConfig> for MeshServerBuilder {
    fn from(value: &MeshServerConfig) -> Self {
        let mut builder = MeshServerBuilder::new(
            value.self_name.clone(),
            value.bind_addr,
            value.advertise_addr,
            value.init_peers.clone(),
        );
        if let Some(mtls_config) = &value.mtls_config {
            builder = builder.with_mtls(mtls_config.clone());
        }
        builder
    }
}

pub struct MeshServer {
    state: ClusterState,
    self_name: String,
    bind_addr: SocketAddr,
    advertise_addr: SocketAddr,
    init_peers: Vec<SocketAddr>,
    signal_rx: watch::Receiver<bool>,
    partition_detector: Option<Arc<PartitionDetector>>,
    /// The node's dials, shared by the gossip controller, the service and
    /// the handler.
    dialer: Arc<Dialer>,
    /// Node-wide MeshKV handle shared by the gossip controller and service.
    mesh_kv: Arc<MeshKV>,
}

impl MeshServer {
    fn build_gossip_service(&self) -> GossipService {
        GossipService::new(
            self.state.clone(),
            self.bind_addr,
            self.advertise_addr,
            &self.self_name,
        )
        .with_mesh_kv(self.mesh_kv.clone())
    }

    fn build_controller(&self) -> GossipController {
        GossipController::new(
            self.state.clone(),
            self.advertise_addr,
            &self.self_name,
            self.init_peers.clone(),
            Arc::clone(&self.dialer),
        )
        .with_mesh_kv(self.mesh_kv.clone())
    }

    pub async fn start(self) -> Result<()> {
        self.prepare(None).await?.serve().await
    }

    pub async fn start_with_listener(self, listener: tokio::net::TcpListener) -> Result<()> {
        self.prepare_with_listener(listener).await?.serve().await
    }

    /// Everything this server does before it serves from `listener`, which
    /// the caller bound on the configured bind address: the listener against
    /// that address, the gossip service and its controller, the tonic server
    /// with the TLS configuration from the node's material. Every failure the
    /// mesh server can raise before it serves surfaces here, so a caller can
    /// refuse its own start on it instead of learning of it from the task
    /// the server runs in, after it serves. [`PreparedMeshServer::serve`]
    /// runs the result.
    pub async fn prepare_with_listener(
        self,
        listener: tokio::net::TcpListener,
    ) -> Result<PreparedMeshServer> {
        let bound_addr = listener
            .local_addr()
            .map_err(|e| anyhow::anyhow!("Failed to read listener local addr: {e}"))?;
        if bound_addr != self.bind_addr {
            return Err(anyhow::anyhow!(
                "Listener/bind_addr mismatch: listener={}, bind_addr={}",
                bound_addr,
                self.bind_addr
            ));
        }
        self.prepare(Some(listener)).await
    }

    async fn prepare(
        self,
        listener: Option<tokio::net::TcpListener>,
    ) -> Result<PreparedMeshServer> {
        #[expect(
            clippy::expect_used,
            reason = "partition_detector is always set to Some by MeshServerBuilder::build() before start() is called"
        )]
        let partition_detector = self
            .partition_detector
            .clone()
            .expect("partition detector missing");

        // Build controller first so we can share its current_stream_batch
        // with server-side sync_stream handlers.
        let controller = self.build_controller();

        let mut service = self.build_gossip_service();

        service = service.with_partition_detector(partition_detector);

        // Share the controller's current_stream_batch so server-side
        // sync_stream handlers see the same drained stream entries as
        // client-side.
        service = service.with_current_stream_batch(controller.current_stream_batch());

        // The node's dials, and with them its TLS identity for the listener.
        service = service.with_dialer(Arc::clone(&self.dialer));

        let server = service.prepare(listener).await?;
        Ok(PreparedMeshServer {
            self_name: self.self_name,
            bind_addr: self.bind_addr,
            advertise_addr: self.advertise_addr,
            signal_rx: self.signal_rx,
            controller,
            server,
        })
    }
}

/// A mesh server with everything that can fail before it serves already
/// done ([`MeshServer::prepare_with_listener`]); [`serve`](Self::serve) runs
/// it for the life of the process.
pub struct PreparedMeshServer {
    self_name: String,
    bind_addr: SocketAddr,
    advertise_addr: SocketAddr,
    signal_rx: watch::Receiver<bool>,
    controller: GossipController,
    server: PreparedGossipServer,
}

impl PreparedMeshServer {
    /// Serves the gossip service and runs the gossip loop until the handler
    /// signals shutdown or either of them fails.
    pub async fn serve(self) -> Result<()> {
        log::info!(
            "Mesh server listening on {} and advertising {}",
            self.bind_addr,
            self.advertise_addr
        );
        let mut service_shutdown = self.signal_rx.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "handle is awaited immediately below via tokio::select!, bounded by shutdown signal"
        )]
        let server_handle = tokio::spawn(self.server.serve(async move {
            _ = service_shutdown.changed().await;
        }));
        tokio::time::sleep(Duration::from_secs(1)).await;
        #[expect(
            clippy::disallowed_methods,
            reason = "handle is awaited immediately below via tokio::select!, bounded by shutdown signal"
        )]
        let app_handle = tokio::spawn(self.controller.event_loop(self.signal_rx.clone()));

        tokio::select! {
            res = server_handle => res??,
            res = app_handle => res??,
        }

        log::info!(
            "Mesh server {} at {} is shutting down",
            self.self_name,
            self.advertise_addr
        );
        Ok(())
    }
}

/// The status a table entry carries, for the log lines (an unknown value
/// reads as `Init`).
pub fn status_of(entry: &NodeState) -> NodeStatus {
    NodeStatus::try_from(entry.status).unwrap_or(NodeStatus::Init)
}

/// The one info line a member's status change writes, whoever changes the
/// table: `Member <name> is <Status> at version <n> (<how>)`. `how` names
/// what changed it (the README's table of log lines lists the spellings);
/// the caller writes the line only when the status held before differs.
pub fn member_transition_line(entry: &NodeState, how: impl std::fmt::Display) -> String {
    format!(
        "Member {} is {:?} at version {} ({how})",
        entry.name,
        status_of(entry),
        entry.version
    )
}

/// Broadcast node state updates to target nodes, each dialed through
/// `dialer` (the channel the probes keep to it, with the node's TLS identity
/// when the mesh has one). `timeout` (3 s when `None`) bounds each dial, not
/// the wait for all of them: a peer that stalls costs the broadcast the
/// deadline and nothing else, the answers that arrived are counted, and no
/// dial outlives the call (the dials are the call's own futures, so dropping
/// the call drops them too).
/// Returns (success_count, total_count)
pub(crate) async fn broadcast_node_states(
    nodes_to_broadcast: Vec<NodeState>,
    target_nodes: Vec<NodeState>,
    timeout: Option<Duration>,
    dialer: Arc<Dialer>,
) -> (usize, usize) {
    if nodes_to_broadcast.is_empty() || target_nodes.is_empty() {
        log::debug!(
            "Nothing to broadcast: nodes_to_broadcast={}, target_nodes={}",
            nodes_to_broadcast.len(),
            target_nodes.len()
        );
        return (0, target_nodes.len());
    }

    let deadline = timeout.unwrap_or(Duration::from_secs(3));
    let broadcasts = target_nodes.iter().map(|target_node| {
        let nodes = nodes_to_broadcast.clone();
        let dialer = &dialer;
        async move {
            let ping_payload = gossip_message::Payload::Ping(Ping {
                state_sync: Some(StateSync { nodes }),
            });
            let answer =
                tokio::time::timeout(deadline, dialer.ping(target_node, Some(ping_payload)))
                    .await
                    .unwrap_or_else(|_| {
                        Err(tonic::Status::deadline_exceeded(format!(
                            "no answer within {deadline:?}"
                        )))
                    });
            match answer {
                Ok(_) => {
                    log::debug!("Successfully broadcasted to {}", target_node.name);
                    Ok(())
                }
                Err(e) => {
                    log::warn!("Failed to broadcast to {}: {}", target_node.name, e);
                    Err(e)
                }
            }
        }
    });

    let results: Vec<Result<(), tonic::Status>> = futures::future::join_all(broadcasts).await;
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    let total_count = target_nodes.len();
    log::info!(
        "Broadcast completed: {}/{} successful",
        success_count,
        total_count
    );
    (success_count, total_count)
}

#[macro_export]
macro_rules! mesh_run {
    ($addr:expr, $init_peers:expr) => {{
        mesh_run!($addr.to_string(), $addr, $init_peers)
    }};

    ($name:expr, $addr:expr, $init_peers:expr) => {{
        tracing::info!("Starting mesh server : {}", $addr);
        use $crate::MeshServerBuilder;
        let (server, handler) =
            MeshServerBuilder::new($name.to_string(), $addr, $addr, $init_peers).build();
        #[expect(clippy::disallowed_methods, reason = "test macro: spawned server runs for the test lifetime and handler is returned for assertions")]
        tokio::spawn(async move {
            if let Err(e) = server.start().await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });
        handler
    }};

    ($name:expr, $listener:expr, $addr:expr, $init_peers:expr) => {{
        tracing::info!("Starting mesh server : {}", $addr);
        use $crate::MeshServerBuilder;
        let (server, handler) =
            MeshServerBuilder::new($name.to_string(), $addr, $addr, $init_peers).build();
        #[expect(clippy::disallowed_methods, reason = "test macro: spawned server runs for the test lifetime and handler is returned for assertions")]
        tokio::spawn(async move {
            if let Err(e) = server.start_with_listener($listener).await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });
        handler
    }};
}

#[cfg(test)]
#[path = "../tests/unit/service/tests.rs"]
mod tests;
