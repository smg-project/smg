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
    mesh_kv: Arc<crate::kv::MeshKV>,
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
    pub fn mesh_kv(&self) -> &Arc<crate::kv::MeshKV> {
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
        let mesh_kv = Arc::new(crate::kv::MeshKV::new(self.self_name.clone()));
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
    mesh_kv: Arc<crate::kv::MeshKV>,
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
/// when the mesh has one).
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

    let mut broadcast_tasks = Vec::new();
    for target_node in &target_nodes {
        let target_node_clone = target_node.clone();
        let nodes_for_task = nodes_to_broadcast.clone();
        let dialer = Arc::clone(&dialer);
        #[expect(
            clippy::disallowed_methods,
            reason = "broadcast tasks are collected and awaited via join_all with a timeout immediately below"
        )]
        let task = tokio::spawn(async move {
            let state_sync = StateSync {
                nodes: nodes_for_task,
            };
            let ping_payload = gossip_message::Payload::Ping(Ping {
                state_sync: Some(state_sync),
            });
            match dialer.ping(&target_node_clone, Some(ping_payload)).await {
                Ok(_) => {
                    log::debug!("Successfully broadcasted to {}", target_node_clone.name);
                    Ok(())
                }
                Err(e) => {
                    log::warn!("Failed to broadcast to {}: {}", target_node_clone.name, e);
                    Err(e)
                }
            }
        });
        broadcast_tasks.push(task);
    }

    let timeout_duration = timeout.unwrap_or(Duration::from_secs(3));
    let broadcast_result = tokio::time::timeout(timeout_duration, async {
        futures::future::join_all(broadcast_tasks).await
    })
    .await;

    match broadcast_result {
        Ok(results) => {
            let success_count = results.iter().filter(|r| matches!(r, Ok(Ok(())))).count();
            let total_count = target_nodes.len();
            log::info!(
                "Broadcast completed: {}/{} successful",
                success_count,
                total_count
            );
            (success_count, total_count)
        }
        Err(_) => {
            log::warn!(
                "Broadcast timeout after {} seconds",
                timeout_duration.as_secs()
            );
            (0, target_nodes.len())
        }
    }
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
mod tests {
    use std::{
        net::{IpAddr, Ipv4Addr, Ipv6Addr},
        sync::{
            atomic::{AtomicUsize, Ordering},
            Once,
        },
    };

    use rustls::crypto::ring;
    use tokio::net::TcpListener;
    use tonic::{
        transport::{
            server::TcpIncoming, Certificate, ClientTlsConfig, Endpoint, Identity, Server,
            ServerTlsConfig,
        },
        Request,
    };
    use tracing as log;
    use tracing_subscriber::{
        filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
    };

    use super::{
        gossip::{gossip_client, gossip_server::GossipServer, GossipMessage, NodeUpdate, PingReq},
        *,
    };
    use crate::{
        gossip_controller::ROUND_PERIOD,
        tests::{
            mtls_certs::MtlsTestCerts,
            test_utils::{bind_node, wait_for},
        },
        transport::limits::MAX_MESSAGE_SIZE,
    };

    static INIT: Once = Once::new();
    fn init() {
        INIT.call_once(|| {
            let _ = tracing_subscriber::registry()
                .with(tracing_subscriber::fmt::layer())
                .with(
                    EnvFilter::builder()
                        .with_default_directive(LevelFilter::INFO.into())
                        .from_env_lossy(),
                )
                .try_init();
        });
    }

    #[tokio::test]
    async fn test_ping_advertises_configured_address() {
        init();

        let (listener, bind_addr) = bind_node().await;
        let advertise_addr = SocketAddr::from(([10, 20, 30, 40], bind_addr.port()));
        let (server, handler) =
            MeshServerBuilder::new("A".to_string(), bind_addr, advertise_addr, Vec::new()).build();

        #[expect(
            clippy::disallowed_methods,
            reason = "test server runs in the background for the duration of the assertion"
        )]
        tokio::spawn(async move {
            if let Err(e) = server.start_with_listener(listener).await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });

        wait_for(
            || std::net::TcpStream::connect(bind_addr).is_ok(),
            Duration::from_secs(5),
            "mesh listener started",
        )
        .await;

        let response = Dialer::new(None)
            .ping(
                &NodeState {
                    name: "A".to_string(),
                    address: bind_addr.to_string(),
                    status: NodeStatus::Alive as i32,
                    version: 1,
                    metadata: HashMap::new(),
                },
                Some(gossip_message::Payload::Ping(Ping {
                    state_sync: Some(StateSync { nodes: vec![] }),
                })),
            )
            .await
            .unwrap();

        assert_eq!(response.address, advertise_addr.to_string());
        handler.shutdown();
    }

    /// Membership converges on every member: four form, a fifth joins and is
    /// hard-stopped, one leaves gracefully, and the three left hold the leaver
    /// `Leaving` and the stopped member not `Alive`. Every budget is twenty
    /// rounds of the gossip tick (`ROUND_PERIOD`): a probe carries the
    /// prober's whole table, so a join reaches every member within a few
    /// rounds; the stopped member is known to every live member before the
    /// stop, so each of them probes it with probability one third per round
    /// and twenty rounds miss it with probability (2/3)^60; its closed socket
    /// refuses the probe and the relays at once, and the verdict is
    /// broadcast in the same round.
    #[tokio::test]
    async fn test_state_synchronization() {
        init();
        log::info!("Starting test_state_synchronization");
        let budget = ROUND_PERIOD * 20;

        // 1. Setup A-B cluster
        let (listener_a, addr_a) = bind_node().await;
        let handler_a = mesh_run!("A", listener_a, addr_a, Vec::new());
        let (listener_b, addr_b) = bind_node().await;
        let handler_b = mesh_run!("B", listener_b, addr_b, vec![addr_a]);

        wait_for(
            || handler_a.state.read().len() == 2,
            budget,
            "A-B cluster formed",
        )
        .await;

        // 2. Add C and D
        let (listener_c, addr_c) = bind_node().await;
        let handler_c = mesh_run!("C", listener_c, addr_c, vec![addr_a]);
        let (listener_d, addr_d) = bind_node().await;
        let handler_d = mesh_run!("D", listener_d, addr_d, vec![addr_c]);

        wait_for(
            || handler_a.state.read().len() == 4,
            budget,
            "4-node cluster formed",
        )
        .await;

        // 3. Add E on a runtime of its own, let every member learn of it, then
        //    drop that runtime: a hard stop closes E's listener and streams at
        //    once, as a process exit does. (The handler's `shutdown` ends the
        //    gossip loop and leaves the listener draining the open streams; a
        //    probe meets that as a wedge, accepted and never answered, and
        //    finds it through the dialer's timeouts instead.)
        let member_e = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        member_e.spawn(async move {
            let (listener_e, addr_e) = bind_node().await;
            let (server, _handler_e) =
                MeshServerBuilder::new("E".to_string(), addr_e, addr_e, vec![addr_d]).build();
            if let Err(e) = server.start_with_listener(listener_e).await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });
        for (handler, name) in [(&handler_a, "A"), (&handler_b, "B"), (&handler_c, "C")] {
            wait_for(
                || handler.state.read().len() == 5,
                budget,
                &format!("E joined the cluster on node {name}"),
            )
            .await;
        }
        member_e.shutdown_background();

        // 4. Gracefully shutdown D
        handler_d.graceful_shutdown().await.unwrap();

        // 5. D=Leaving and E not Alive on every remaining node
        let status_on = |handler: &MeshServerHandler, member: &str| {
            handler.state.read().get(member).map(|n| n.status)
        };
        for (handler, name) in [(&handler_a, "A"), (&handler_b, "B"), (&handler_c, "C")] {
            wait_for(
                || status_on(handler, "D") == Some(NodeStatus::Leaving as i32),
                budget,
                &format!("D=Leaving on node {name}"),
            )
            .await;
            wait_for(
                || status_on(handler, "E").is_some_and(|status| status != NodeStatus::Alive as i32),
                budget,
                &format!("E not Alive on node {name}"),
            )
            .await;
        }

        log::info!("All nodes converged to expected state");
    }

    /// A member whose join peer is not up at its first round keeps dialing
    /// it: B starts with A's address while nothing listens there, A comes up
    /// a few rounds later, and the two still find each other.
    #[tokio::test]
    async fn a_member_keeps_dialing_its_join_peer_until_it_is_up() {
        init();
        // A's address, reserved by a bind that is released at once: B's
        // first dials are refused, as they are while the peer is not up yet.
        let (listener_a, addr_a) = bind_node().await;
        drop(listener_a);
        let (listener_b, addr_b) = bind_node().await;
        let handler_b = mesh_run!("B", listener_b, addr_b, vec![addr_a]);

        // B's first round and its dial pass before A binds its address.
        tokio::time::sleep(Duration::from_millis(2500)).await;
        assert_eq!(
            handler_b.state.read().len(),
            1,
            "B found a member while nothing listened on A's address"
        );
        let listener_a = TcpListener::bind(addr_a)
            .await
            .expect("A's address is free again");
        let handler_a = mesh_run!("A", listener_a, addr_a, Vec::new());

        wait_for(
            || handler_a.state.read().len() == 2 && handler_b.state.read().len() == 2,
            Duration::from_secs(20),
            "B joins A once A is up",
        )
        .await;
        handler_a.shutdown();
        handler_b.shutdown();
    }

    /// A gossip peer answering over TLS with the test node certificate (IP
    /// SANs for both loopback addresses), bound to an ephemeral port of
    /// `loopback`.
    async fn gossip_peer_over_tls(loopback: IpAddr, certs: &MtlsTestCerts) -> SocketAddr {
        let listener = TcpListener::bind((loopback, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let service =
            GossipService::new(Arc::new(RwLock::new(BTreeMap::new())), addr, addr, "peer");
        let identity = Identity::from_pem(
            std::fs::read(&certs.node_cert_path).unwrap(),
            std::fs::read(&certs.node_key_path).unwrap(),
        );
        let server = Server::builder()
            .tls_config(ServerTlsConfig::new().identity(identity))
            .unwrap()
            .add_service(
                GossipServer::new(service)
                    .max_decoding_message_size(MAX_MESSAGE_SIZE)
                    .max_encoding_message_size(MAX_MESSAGE_SIZE)
                    .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
                    .send_compressed(tonic::codec::CompressionEncoding::Gzip),
            )
            .serve_with_incoming(TcpIncoming::from(listener));
        #[expect(
            clippy::disallowed_methods,
            reason = "test peer runs in the background for the duration of the assertion"
        )]
        tokio::spawn(server);
        addr
    }

    /// The process-level crypto provider rustls needs before any TLS
    /// configuration is built: the gateway installs it at start-up, the test
    /// binary does it itself, and before the peer's server is built, since a
    /// build with both rustls backends enabled (the workspace's test build)
    /// cannot pick one on its own.
    fn install_crypto_provider() {
        let _ = ring::default_provider().install_default();
    }

    /// A dial with mTLS on, the way the gossip loop dials a peer.
    async fn mtls_ping(
        peer: SocketAddr,
        certs: &MtlsTestCerts,
    ) -> Result<NodeUpdate, tonic::Status> {
        install_crypto_provider();
        let mtls = MTLSManager::new(MTLSConfig {
            ca_cert_path: certs.ca_cert_path.clone(),
            server_cert_path: certs.node_cert_path.clone(),
            server_key_path: certs.node_key_path.clone(),
            require_client_cert: false,
        });
        Dialer::new(Some(Arc::new(mtls)))
            .ping(
                &NodeState {
                    name: "peer".to_string(),
                    address: peer.to_string(),
                    status: NodeStatus::Alive as i32,
                    version: 1,
                    metadata: HashMap::new(),
                },
                Some(gossip_message::Payload::Ping(Ping {
                    state_sync: Some(StateSync { nodes: vec![] }),
                })),
            )
            .await
    }

    #[tokio::test]
    async fn mtls_ping_reaches_an_ipv6_peer_by_its_ip_san() {
        init();
        install_crypto_provider();
        // Nothing to assert on a host without an IPv6 loopback.
        if std::net::TcpListener::bind("[::1]:0").is_err() {
            return;
        }
        let certs = MtlsTestCerts::generate();
        let peer = gossip_peer_over_tls(Ipv6Addr::LOCALHOST.into(), &certs).await;

        let update = mtls_ping(peer, &certs)
            .await
            .unwrap_or_else(|e| panic!("ping over TLS to {peer} failed: {e}"));

        assert_eq!(update.address, peer.to_string());
    }

    #[tokio::test]
    async fn mtls_ping_reaches_an_ipv4_peer_by_its_ip_san() {
        init();
        install_crypto_provider();
        let certs = MtlsTestCerts::generate();
        let peer = gossip_peer_over_tls(Ipv4Addr::LOCALHOST.into(), &certs).await;

        let update = mtls_ping(peer, &certs)
            .await
            .unwrap_or_else(|e| panic!("ping over TLS to {peer} failed: {e}"));

        assert_eq!(update.address, peer.to_string());
    }

    fn mtls_config(certs: &MtlsTestCerts) -> MTLSConfig {
        MTLSConfig {
            ca_cert_path: certs.ca_cert_path.clone(),
            server_cert_path: certs.node_cert_path.clone(),
            server_key_path: certs.node_key_path.clone(),
            require_client_cert: true,
        }
    }

    /// The transition line every writer of the membership shares: the member,
    /// its status, the version and the cause the writer names.
    #[test]
    fn a_member_transition_line_names_the_member_its_status_the_version_and_the_cause() {
        let entry = NodeState {
            name: "r1".to_string(),
            address: "10.1.0.1:7100".to_string(),
            status: NodeStatus::Suspected as i32,
            version: 3,
            metadata: Default::default(),
        };
        assert_eq!(
            member_transition_line(
                &entry,
                format_args!("a peer's table; was {:?}", NodeStatus::Alive)
            ),
            "Member r1 is Suspected at version 3 (a peer's table; was Alive)"
        );
        assert_eq!(
            member_transition_line(&entry, "joined this node's table"),
            "Member r1 is Suspected at version 3 (joined this node's table)"
        );
    }

    fn node_state(name: &str, addr: SocketAddr) -> NodeState {
        NodeState {
            name: name.to_string(),
            address: addr.to_string(),
            status: NodeStatus::Alive as i32,
            version: 1,
            metadata: HashMap::new(),
        }
    }

    /// A mesh node with mTLS on (client certificates required), bound to an
    /// ephemeral port of `ip`, joining `init_peers`.
    async fn mtls_node(
        name: &str,
        ip: IpAddr,
        init_peers: Vec<SocketAddr>,
        certs: &MtlsTestCerts,
    ) -> (SocketAddr, MeshServerHandler) {
        mtls_node_with(name, ip, init_peers, mtls_config(certs)).await
    }

    /// A mesh node with mTLS on under `config`, bound to an ephemeral port
    /// of `ip`, joining `init_peers`.
    async fn mtls_node_with(
        name: &str,
        ip: IpAddr,
        init_peers: Vec<SocketAddr>,
        config: MTLSConfig,
    ) -> (SocketAddr, MeshServerHandler) {
        let listener = TcpListener::bind((ip, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (server, handler) = MeshServerBuilder::new(name.to_string(), addr, addr, init_peers)
            .with_mtls(config)
            .build();
        let name = name.to_string();
        #[expect(
            clippy::disallowed_methods,
            reason = "test node runs in the background for the duration of the assertion"
        )]
        tokio::spawn(async move {
            if let Err(e) = server.start_with_listener(listener).await {
                tracing::error!("Mesh node {name} failed: {e}");
            }
        });
        wait_for(
            || std::net::TcpStream::connect(addr).is_ok(),
            Duration::from_secs(5),
            "mesh listener started",
        )
        .await;
        (addr, handler)
    }

    /// Two nodes with mTLS on `ip`: B joins A (the gossip ping), then a value
    /// put on A reaches B (the sync stream).
    async fn mesh_forms_with_mtls_over(ip: IpAddr) {
        init();
        let _ = ring::default_provider().install_default();
        let certs = MtlsTestCerts::generate();
        let (addr_a, handler_a) = mtls_node("A", ip, Vec::new(), &certs).await;
        let (_addr_b, handler_b) = mtls_node("B", ip, vec![addr_a], &certs).await;

        wait_for(
            || handler_a.state.read().len() == 2 && handler_b.state.read().len() == 2,
            Duration::from_secs(20),
            "A and B see each other over mTLS",
        )
        .await;

        handler_a
            .mesh_kv()
            .configs()
            .put("config:tls", b"on".to_vec());
        wait_for(
            || handler_b.mesh_kv().configs().get("config:tls").as_deref() == Some(b"on".as_slice()),
            Duration::from_secs(20),
            "A's value reaches B over the sync stream",
        )
        .await;

        handler_a.shutdown();
        handler_b.shutdown();
    }

    #[tokio::test]
    async fn mesh_forms_with_mtls_over_ipv4() {
        mesh_forms_with_mtls_over(Ipv4Addr::LOCALHOST.into()).await;
    }

    #[tokio::test]
    async fn mesh_forms_with_mtls_over_ipv6() {
        // Nothing to assert on a host without an IPv6 loopback.
        if std::net::TcpListener::bind("[::1]:0").is_err() {
            return;
        }
        mesh_forms_with_mtls_over(Ipv6Addr::LOCALHOST.into()).await;
    }

    /// A member that leaves tells its peers so with the node's TLS identity
    /// on the dial: B's `Leaving` reaches A under mTLS, whose listener refuses
    /// a dial without a certificate from the CA.
    #[tokio::test]
    async fn a_leaving_member_tells_its_peers_over_mtls() {
        init();
        let _ = ring::default_provider().install_default();
        let certs = MtlsTestCerts::generate();
        let ip: IpAddr = Ipv4Addr::LOCALHOST.into();
        let (addr_a, handler_a) = mtls_node("A", ip, Vec::new(), &certs).await;
        let (_addr_b, handler_b) = mtls_node("B", ip, vec![addr_a], &certs).await;
        wait_for(
            || handler_a.state.read().len() == 2 && handler_b.state.read().len() == 2,
            Duration::from_secs(20),
            "A and B see each other over mTLS",
        )
        .await;

        handler_b.graceful_shutdown().await.expect("B leaves");

        let b_on_a = handler_a.state.read().get("B").cloned();
        assert_eq!(
            b_on_a.as_ref().map(|node| node.status),
            Some(NodeStatus::Leaving as i32),
            "A does not show B as Leaving after B's broadcast: {b_on_a:?}"
        );
        // B's own table says so too, so a probe B still answers before its
        // loop stops cannot turn a member's view of B back to Alive.
        let b_on_b = handler_b.state.read().get("B").cloned();
        assert_eq!(
            b_on_b.as_ref().map(|node| node.status),
            Some(NodeStatus::Leaving as i32),
            "B's own table does not show B as Leaving: {b_on_b:?}"
        );
        handler_a.shutdown();
    }

    /// The status broadcast itself, with no gossip loop on the sending side
    /// to carry the news another way: against a member over mTLS, a dial
    /// without the node's identity is refused and delivers nothing, a dial
    /// with it is accepted and the member records the status.
    #[tokio::test]
    async fn a_status_broadcast_dials_with_the_node_tls_identity() {
        init();
        let _ = ring::default_provider().install_default();
        let certs = MtlsTestCerts::generate();
        let (addr_a, handler_a) =
            mtls_node("A", Ipv4Addr::LOCALHOST.into(), Vec::new(), &certs).await;
        let mut leaving_b = node_state("B", SocketAddr::from((Ipv4Addr::LOCALHOST, 1)));
        leaving_b.status = NodeStatus::Leaving as i32;
        let timeout = Some(Duration::from_secs(3));

        let without_identity = broadcast_node_states(
            vec![leaving_b.clone()],
            vec![node_state("A", addr_a)],
            timeout,
            Dialer::new(None),
        )
        .await;
        assert_eq!(
            without_identity,
            (0, 1),
            "a broadcast dialed without the identity was accepted by an mTLS member"
        );
        assert!(
            handler_a.state.read().get("B").is_none(),
            "A recorded a status from a refused dial: {:?}",
            handler_a.state.read().get("B")
        );

        let with_identity = broadcast_node_states(
            vec![leaving_b],
            vec![node_state("A", addr_a)],
            timeout,
            Dialer::new(Some(Arc::new(MTLSManager::new(mtls_config(&certs))))),
        )
        .await;
        assert_eq!(
            with_identity,
            (1, 1),
            "a broadcast dialed with the identity was refused by an mTLS member"
        );
        let b_on_a = handler_a.state.read().get("B").cloned();
        assert_eq!(
            b_on_a.as_ref().map(|node| node.status),
            Some(NodeStatus::Leaving as i32),
            "A did not record B's status from the broadcast: {b_on_a:?}"
        );
        handler_a.shutdown();
    }

    #[tokio::test]
    async fn a_dialer_without_a_client_certificate_is_refused() {
        init();
        let _ = ring::default_provider().install_default();
        let certs = MtlsTestCerts::generate();
        let (addr_a, handler_a) =
            mtls_node("A", Ipv4Addr::LOCALHOST.into(), Vec::new(), &certs).await;

        let trusting_only = ClientTlsConfig::new()
            .domain_name(addr_a.ip().to_string())
            .ca_certificate(Certificate::from_pem(
                std::fs::read(&certs.ca_cert_path).unwrap(),
            ));
        let endpoint = Endpoint::from_shared(format!("https://{addr_a}"))
            .unwrap()
            .tls_config(trusting_only)
            .unwrap()
            .connect_timeout(Duration::from_secs(5));
        let ping = GossipMessage {
            payload: Some(gossip_message::Payload::Ping(Ping { state_sync: None })),
        };
        let refused = match endpoint.connect().await {
            Err(_) => true,
            Ok(channel) => gossip_client::GossipClient::new(channel)
                .ping_server(Request::new(ping))
                .await
                .is_err(),
        };

        assert!(
            refused,
            "a dialer without a client certificate must be refused"
        );
        handler_a.shutdown();
    }

    #[tokio::test]
    async fn a_dialer_with_a_foreign_ca_is_refused() {
        init();
        let _ = ring::default_provider().install_default();
        let certs = MtlsTestCerts::generate();
        let foreign = MtlsTestCerts::generate();
        let (addr_a, handler_a) =
            mtls_node("A", Ipv4Addr::LOCALHOST.into(), Vec::new(), &certs).await;

        let result = Dialer::new(Some(Arc::new(MTLSManager::new(mtls_config(&foreign)))))
            .ping(
                &node_state("A", addr_a),
                Some(gossip_message::Payload::Ping(Ping { state_sync: None })),
            )
            .await;

        assert!(result.is_err(), "a foreign CA must be refused: {result:?}");
        handler_a.shutdown();
    }

    /// The dialing end of a pair opens one connection to its peer: the
    /// channel the probes keep carries the sync stream too. B advertises a
    /// counting proxy in front of its listener, A joins through it, and once
    /// a value put on A has reached B over the stream, the proxy has
    /// accepted one connection from A. (Before, the stream dialed a
    /// connection of its own: two.)
    #[tokio::test]
    async fn the_sync_stream_runs_on_the_channel_the_probes_keep() {
        init();
        let (listener_b, bind_b) = bind_node().await;
        let (proxy_b, accepted) = counting_proxy(bind_b).await;
        let (server_b, handler_b) =
            MeshServerBuilder::new("B".to_string(), bind_b, proxy_b, Vec::new()).build();
        #[expect(
            clippy::disallowed_methods,
            reason = "test node runs in the background for the duration of the assertion"
        )]
        tokio::spawn(async move {
            if let Err(e) = server_b.start_with_listener(listener_b).await {
                tracing::error!("Mesh node B failed: {e}");
            }
        });
        let (listener_a, addr_a) = bind_node().await;
        let handler_a = mesh_run!("A", listener_a, addr_a, vec![proxy_b]);

        wait_for(
            || handler_a.state.read().len() == 2 && handler_b.state.read().len() == 2,
            Duration::from_secs(20),
            "A and B see each other",
        )
        .await;
        handler_a
            .mesh_kv()
            .configs()
            .put("config:shared", b"channel".to_vec());
        wait_for(
            || {
                handler_b
                    .mesh_kv()
                    .configs()
                    .get("config:shared")
                    .as_deref()
                    == Some(b"channel".as_slice())
            },
            Duration::from_secs(20),
            "A's value reaches B over the sync stream",
        )
        .await;

        assert_eq!(
            accepted.load(Ordering::SeqCst),
            1,
            "A opened more than one connection to B: the stream did not run on the probes' channel"
        );
        handler_a.shutdown();
        handler_b.shutdown();
    }

    /// A loopback proxy in front of `target`: it counts the connections it
    /// accepts and copies bytes both ways.
    async fn counting_proxy(target: SocketAddr) -> (SocketAddr, Arc<AtomicUsize>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let accepted = Arc::new(AtomicUsize::new(0));
        let count = accepted.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "test proxy runs in the background for the duration of the assertion"
        )]
        tokio::spawn(async move {
            while let Ok((mut inbound, _)) = listener.accept().await {
                count.fetch_add(1, Ordering::SeqCst);
                #[expect(
                    clippy::disallowed_methods,
                    reason = "one proxied connection, copied until either end closes"
                )]
                tokio::spawn(async move {
                    if let Ok(mut outbound) = tokio::net::TcpStream::connect(target).await {
                        let _ = tokio::io::copy_bidirectional(&mut inbound, &mut outbound).await;
                    }
                });
            }
        });
        (addr, accepted)
    }

    /// The member asked to probe a peer on another member's behalf sends
    /// the peer a probe without a table and hands back the peer's own
    /// answer: C answers B's relayed probe with its name, address and
    /// status, and nothing joins C's table from it. (The relay used to send
    /// a message without a payload, which the peer refused as invalid, so
    /// no relayed probe ever reached its target.)
    #[tokio::test]
    async fn a_relayed_probe_reaches_its_target_and_brings_back_its_answer() {
        init();
        let (addr_b, handler_b) = plain_node("B").await;
        let (addr_c, handler_c) = plain_node("C").await;

        let answer = Dialer::new(None)
            .ping(
                &node_state("B", addr_b),
                Some(gossip_message::Payload::PingReq(PingReq {
                    node: Some(node_state("C", addr_c)),
                })),
            )
            .await
            .expect("the relayed probe is answered");

        assert_eq!(
            (answer.name, answer.address, answer.status),
            (
                "C".to_string(),
                addr_c.to_string(),
                NodeStatus::Alive as i32
            ),
            "B handed back something other than C's own answer"
        );
        assert_eq!(
            handler_c.state.read().keys().cloned().collect::<Vec<_>>(),
            vec!["C".to_string()],
            "a relayed probe without a table changed C's table"
        );
        handler_b.shutdown();
        handler_c.shutdown();
    }

    /// A relay answers for the node it reached. A table holds C at an
    /// address X answers at now: B's relayed probe to that address brings
    /// back X's answer, which is not C's. It is refused, C's entry stays as
    /// held and nothing of X enters the table; C's own answer through B is
    /// recorded and C is `Alive` again.
    #[tokio::test]
    async fn a_relayed_answer_from_a_stale_address_is_not_the_peers() {
        init();
        let (addr_b, handler_b) = plain_node("B").await;
        let (addr_c, handler_c) = plain_node("C").await;
        let (addr_x, handler_x) = plain_node("X").await;
        let stale = NodeState {
            status: NodeStatus::Suspected as i32,
            version: 3,
            ..node_state("C", addr_x)
        };
        let table: ClusterState = Arc::new(RwLock::new(BTreeMap::from([(
            "C".to_string(),
            stale.clone(),
        )])));
        let dialer = Dialer::new(None);

        let answer = dialer
            .ping(
                &node_state("B", addr_b),
                Some(gossip_message::Payload::PingReq(PingReq {
                    node: Some(stale),
                })),
            )
            .await
            .expect("B relayed the probe to the stale address");
        assert_eq!(
            answer.name, "X",
            "the node at the stale address answers as itself"
        );
        assert!(
            !GossipController::record_relayed_answer(&table, "C", &addr_b.to_string(), &answer),
            "X's answer was taken as C's"
        );
        {
            let table = table.read();
            let c = table.get("C").expect("the table still holds C");
            assert_eq!(
                (c.status, c.address.clone(), c.version),
                (NodeStatus::Suspected as i32, addr_x.to_string(), 3),
                "C's entry moved on X's answer"
            );
            assert!(!table.contains_key("X"), "X's answer entered the table");
        }

        let answer = dialer
            .ping(
                &node_state("B", addr_b),
                Some(gossip_message::Payload::PingReq(PingReq {
                    node: Some(node_state("C", addr_c)),
                })),
            )
            .await
            .expect("B relayed the probe to C");
        assert!(
            GossipController::record_relayed_answer(&table, "C", &addr_b.to_string(), &answer),
            "C's own answer was refused"
        );
        assert_eq!(
            table.read().get("C").map(|c| c.status),
            Some(NodeStatus::Alive as i32),
            "C is not Alive after its own answer through B"
        );
        handler_b.shutdown();
        handler_c.shutdown();
        handler_x.shutdown();
    }

    /// Two members that cannot reach each other stay `Alive` on each other
    /// through the member that reaches both. A's certificate comes from a
    /// CA that B trusts and C does not, and C's from the CA A and B trust:
    /// A's and C's dials to each other fail at the handshake, every dial to
    /// or from B passes, so each failed direct probe between A and C is
    /// relayed through B, and for fifteen seconds neither marks the other
    /// `Suspected`.
    #[tokio::test]
    async fn two_members_that_cannot_reach_each_other_stay_alive_through_the_one_that_reaches_both()
    {
        init();
        install_crypto_provider();
        let certs = MtlsTestCerts::generate();
        let other_ca = MtlsTestCerts::generate();
        // A trust file with both CAs: its holder accepts either certificate
        // from a client and from a server.
        let both = certs.ca_cert_path.with_file_name("ca-both.pem");
        std::fs::write(
            &both,
            [
                std::fs::read(&certs.ca_cert_path).unwrap(),
                std::fs::read(&other_ca.ca_cert_path).unwrap(),
            ]
            .concat(),
        )
        .unwrap();
        let trusting_both = |member: &MtlsTestCerts| MTLSConfig {
            ca_cert_path: both.clone(),
            server_cert_path: member.node_cert_path.clone(),
            server_key_path: member.node_key_path.clone(),
            require_client_cert: true,
        };
        let ip: IpAddr = Ipv4Addr::LOCALHOST.into();
        let (addr_a, handler_a) =
            mtls_node_with("A", ip, Vec::new(), trusting_both(&other_ca)).await;
        let (addr_b, handler_b) =
            mtls_node_with("B", ip, vec![addr_a], trusting_both(&certs)).await;
        let (addr_c, handler_c) = mtls_node_with("C", ip, vec![addr_b], mtls_config(&certs)).await;

        // The bed: A and C refuse each other, B reaches both.
        let a_to_c = handler_a
            .dialer
            .ping(&node_state("C", addr_c), Some(probe()))
            .await;
        assert!(a_to_c.is_err(), "A reached C directly: {a_to_c:?}");
        let c_to_a = handler_c
            .dialer
            .ping(&node_state("A", addr_a), Some(probe()))
            .await;
        assert!(c_to_a.is_err(), "C reached A directly: {c_to_a:?}");
        for (name, addr) in [("A", addr_a), ("C", addr_c)] {
            let from_b = handler_b
                .dialer
                .ping(&node_state(name, addr), Some(probe()))
                .await;
            assert!(from_b.is_ok(), "B did not reach {name}: {from_b:?}");
        }

        wait_for(
            || {
                [&handler_a, &handler_b, &handler_c]
                    .iter()
                    .all(|handler| handler.state.read().len() == 3)
            },
            Duration::from_secs(20),
            "the three members know each other",
        )
        .await;

        let status_on = |holder: &MeshServerHandler, name: &str| {
            holder.state.read().get(name).map(|node| node.status)
        };
        let started = std::time::Instant::now();
        while started.elapsed() < Duration::from_secs(15) {
            for (holder, label, name) in [(&handler_a, "A", "C"), (&handler_c, "C", "A")] {
                let status = status_on(holder, name);
                assert_eq!(
                    status,
                    Some(NodeStatus::Alive as i32),
                    "{label} holds {name} at status {status:?} after {:?} although B reaches {name}",
                    started.elapsed()
                );
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // A verdict from an earlier break: C's own answer, brought back by
        // the relay through B, recorded as the loop records it over a
        // `Suspected` entry for C one above what A holds, and A holds C
        // `Alive` again. The relay is asked here and its answer recorded by
        // hand, not waited for on the loop: the loop could carry the verdict
        // to C through B first, and C would answer with it (the standoff the
        // refutation addresses).
        let answer = handler_a
            .dialer
            .ping(
                &node_state("B", addr_b),
                Some(gossip_message::Payload::PingReq(PingReq {
                    node: Some(node_state("C", addr_c)),
                })),
            )
            .await
            .expect("B relayed the probe to C");
        assert_eq!(
            (answer.name.as_str(), answer.status),
            ("C", NodeStatus::Alive as i32),
            "C's own answer through B"
        );
        {
            let mut table = handler_a.state.write();
            let entry = table.get_mut("C").expect("A holds C");
            entry.status = NodeStatus::Suspected as i32;
            entry.version += 1;
        }
        GossipController::record_in_table(&handler_a.state, &answer);
        assert_eq!(
            status_on(&handler_a, "C"),
            Some(NodeStatus::Alive as i32),
            "A holds C Alive again once the relay's answer is recorded"
        );
        handler_a.shutdown();
        handler_b.shutdown();
        handler_c.shutdown();
    }

    /// A plain mesh node on an ephemeral loopback port, listening.
    async fn plain_node(name: &str) -> (SocketAddr, MeshServerHandler) {
        let (listener, addr) = bind_node().await;
        let handler = mesh_run!(name, listener, addr, Vec::new());
        wait_for(
            || std::net::TcpStream::connect(addr).is_ok(),
            Duration::from_secs(5),
            "mesh listener started",
        )
        .await;
        (addr, handler)
    }

    fn probe() -> gossip_message::Payload {
        gossip_message::Payload::Ping(Ping {
            state_sync: Some(StateSync { nodes: vec![] }),
        })
    }

    /// The dialer keeps one channel per peer address: two dials of one
    /// member share it, another member gets its own, and `retain` drops the
    /// channels of the addresses it refuses.
    #[tokio::test]
    async fn a_dialer_keeps_one_channel_per_peer_address() {
        init();
        let (addr_a, handler_a) = plain_node("A").await;
        let (addr_b, handler_b) = plain_node("B").await;
        let dialer = Dialer::new(None);

        dialer
            .ping(&node_state("A", addr_a), Some(probe()))
            .await
            .expect("A answers");
        dialer
            .ping(&node_state("A", addr_a), Some(probe()))
            .await
            .expect("A answers again");
        assert_eq!(
            dialer.channel_count(),
            1,
            "two dials of one address share one channel"
        );
        dialer
            .ping(&node_state("B", addr_b), Some(probe()))
            .await
            .expect("B answers");
        assert_eq!(
            dialer.channel_count(),
            2,
            "another address gets a channel of its own"
        );

        dialer.retain(|addr| *addr == addr_a);
        assert_eq!(
            dialer.channel_count(),
            1,
            "retain drops the channels of the addresses it refuses"
        );
        handler_a.shutdown();
        handler_b.shutdown();
    }

    /// A node whose first join peer is dead joins through its second: the
    /// entries are dialed in turn, so a dead seed costs a round, not the
    /// mesh.
    #[tokio::test]
    async fn a_node_joins_through_its_second_join_peer_when_the_first_is_dead() {
        init();
        let dead = {
            let (listener, addr) = bind_node().await;
            drop(listener);
            addr
        };
        let (listener_a, addr_a) = bind_node().await;
        let handler_a = mesh_run!("A", listener_a, addr_a, Vec::new());
        let (listener_b, addr_b) = bind_node().await;
        let handler_b = mesh_run!("B", listener_b, addr_b, vec![dead, addr_a]);

        wait_for(
            || handler_a.state.read().len() == 2 && handler_b.state.read().len() == 2,
            Duration::from_secs(20),
            "B joins A through its second join peer",
        )
        .await;
        handler_a.shutdown();
        handler_b.shutdown();
    }

    /// A member that goes away and comes back on its address answers a later
    /// dial through the channel kept for it: a dial into the gap fails
    /// instead of hanging, the channel reconnects by itself, and no second
    /// channel is made for the address.
    #[tokio::test]
    async fn a_peer_back_on_its_address_answers_through_the_kept_channel() {
        init();
        let (addr, handler) = plain_node("A").await;
        let dialer = Dialer::new(None);
        dialer
            .ping(&node_state("A", addr), Some(probe()))
            .await
            .expect("the first life answers");

        handler.shutdown();
        let listener = {
            // The address is free once the first life's listener is gone.
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            loop {
                match TcpListener::bind(addr).await {
                    Ok(listener) => break listener,
                    Err(e) if std::time::Instant::now() < deadline => {
                        log::debug!("the address is still held: {e}");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                    Err(e) => panic!("the address was never released: {e}"),
                }
            }
        };
        let in_the_gap = dialer.ping(&node_state("A", addr), Some(probe())).await;
        assert!(
            in_the_gap.is_err(),
            "a dial into the gap answered: {in_the_gap:?}"
        );

        let handler = mesh_run!("A", listener, addr, Vec::new());
        let mut answer = None;
        for _ in 0..50 {
            match dialer.ping(&node_state("A", addr), Some(probe())).await {
                Ok(update) => {
                    answer = Some(update);
                    break;
                }
                Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
            }
        }
        assert_eq!(
            answer.map(|update| update.name).as_deref(),
            Some("A"),
            "the second life never answered through the kept channel"
        );
        assert_eq!(
            dialer.channel_count(),
            1,
            "no second channel was made for the address"
        );
        handler.shutdown();
    }

    /// Material the node can read but not use fails the preparation, before
    /// anything serves: the listener goes with the error and its port answers
    /// nobody. The caller refuses its own start on that error, where it used
    /// to learn of it from the task the running server failed in.
    #[tokio::test]
    async fn unusable_tls_material_fails_the_preparation_before_anything_serves() {
        init();
        install_crypto_provider();
        let certs = MtlsTestCerts::generate();
        std::fs::write(&certs.node_cert_path, "not a certificate\n").unwrap();
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (server, _handler) = MeshServerBuilder::new("node".to_string(), addr, addr, Vec::new())
            .with_mtls(mtls_config(&certs))
            .build();

        let error = match server.prepare_with_listener(listener).await {
            Ok(_) => panic!("a mesh server was prepared from unusable TLS material"),
            Err(e) => format!("{e:#}"),
        };

        assert!(
            error.contains("mesh TLS material"),
            "the error does not say what is wrong: {error}"
        );
        assert!(
            std::net::TcpStream::connect(addr).is_err(),
            "the mesh port answers after the failed preparation"
        );
    }

    /// The listener handed in must be bound on the configured address: a
    /// mismatch fails the preparation, not the first poll of a running task.
    #[tokio::test]
    async fn a_listener_on_another_address_fails_the_preparation() {
        init();
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let other = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let configured = other.local_addr().unwrap();
        let (server, _handler) =
            MeshServerBuilder::new("node".to_string(), configured, configured, Vec::new()).build();

        let error = match server.prepare_with_listener(listener).await {
            Ok(_) => panic!("a mesh server was prepared on a listener bound elsewhere"),
            Err(e) => e.to_string(),
        };

        assert!(
            error.contains("mismatch"),
            "the error does not name the mismatch: {error}"
        );
    }
}
