use std::{
    collections::{BTreeMap, HashMap},
    net::SocketAddr,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use parking_lot::RwLock;
use tokio::sync::watch;
use tonic::{
    transport::{ClientTlsConfig, Endpoint},
    Request,
};
use tracing as log;

use crate::transport::limits::MAX_MESSAGE_SIZE;

pub mod gossip {
    #![allow(unused_qualifications, clippy::absolute_paths)]
    #![allow(clippy::trivially_copy_pass_by_ref, clippy::allow_attributes)]
    tonic::include_proto!("mesh.gossip");
}
use gossip::{
    gossip_client, gossip_message, GossipMessage, NodeState, NodeStatus, NodeUpdate, Ping,
    StateSync,
};

use crate::{
    gossip_controller::GossipController,
    gossip_service::GossipService,
    mtls::{tls_server_name, MTLSConfig, MTLSManager},
    partition::PartitionDetector,
};

pub type ClusterState = Arc<RwLock<BTreeMap<String, NodeState>>>;

pub struct MeshServerConfig {
    pub self_name: String,
    pub bind_addr: SocketAddr,
    pub advertise_addr: SocketAddr,
    pub init_peer: Option<SocketAddr>,
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
            let state = self.state.read();

            if let Some(self_node) = state.get(&self.self_name) {
                let mut self_node = self_node.clone();
                if self_node.status == NodeStatus::Leaving as i32 {
                    None
                } else {
                    self_node.status = NodeStatus::Leaving as i32;
                    self_node.version += 1;

                    let alive_nodes = state
                        .values()
                        .filter(|node| {
                            node.status == NodeStatus::Alive as i32 && node.name != self.self_name
                            // exclude self from broadcast targets
                        })
                        .cloned()
                        .collect::<Vec<NodeState>>();

                    Some((self_node, alive_nodes))
                }
            } else {
                None
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
    init_peer: Option<SocketAddr>,
    mtls_manager: Option<Arc<MTLSManager>>,
}

impl MeshServerBuilder {
    pub fn new(
        self_name: String,
        bind_addr: SocketAddr,
        advertise_addr: SocketAddr,
        init_peer: Option<SocketAddr>,
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
            init_peer,
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
        (
            MeshServer {
                state: self.state.clone(),
                self_name: self.self_name.clone(),
                bind_addr: self.bind_addr,
                advertise_addr: self.advertise_addr,
                init_peer: self.init_peer,
                signal_rx,
                partition_detector: Some(partition_detector.clone()),
                mtls_manager: self.mtls_manager.clone(),
                mesh_kv: mesh_kv.clone(),
            },
            MeshServerHandler {
                state: self.state.clone(),
                self_name: self.self_name.clone(),
                signal_tx,
                partition_detector: Some(partition_detector),
                mesh_kv,
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
            value.init_peer,
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
    init_peer: Option<SocketAddr>,
    signal_rx: watch::Receiver<bool>,
    partition_detector: Option<Arc<PartitionDetector>>,
    mtls_manager: Option<Arc<MTLSManager>>,
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
            self.init_peer,
            self.mtls_manager.clone(),
        )
        .with_mesh_kv(self.mesh_kv.clone())
    }

    pub async fn start(self) -> Result<()> {
        self.start_inner(None).await
    }

    pub async fn start_with_listener(self, listener: tokio::net::TcpListener) -> Result<()> {
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
        self.start_inner(Some(listener)).await
    }

    async fn start_inner(self, listener: Option<tokio::net::TcpListener>) -> Result<()> {
        log::info!(
            "Mesh server listening on {} and advertising {}",
            self.bind_addr,
            self.advertise_addr
        );
        let self_name = self.self_name.clone();
        let advertise_address = self.advertise_addr;

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

        // Add mTLS support if configured
        if let Some(mtls_manager) = self.mtls_manager.clone() {
            service = service.with_mtls_manager(mtls_manager);
        }

        let mut service_shutdown = self.signal_rx.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "handle is awaited immediately below via tokio::select!, bounded by shutdown signal"
        )]
        let server_handle = if let Some(tcp_listener) = listener {
            tokio::spawn(service.serve_ping_with_listener(tcp_listener, async move {
                _ = service_shutdown.changed().await;
            }))
        } else {
            tokio::spawn(service.serve_ping_with_shutdown(async move {
                _ = service_shutdown.changed().await;
            }))
        };
        tokio::time::sleep(Duration::from_secs(1)).await;
        #[expect(
            clippy::disallowed_methods,
            reason = "handle is awaited immediately below via tokio::select!, bounded by shutdown signal"
        )]
        let app_handle = tokio::spawn(controller.event_loop(self.signal_rx.clone()));

        tokio::select! {
            res = server_handle => res??,
            res = app_handle => res??,
        }

        log::info!(
            "Mesh server {} at {} is shutting down",
            self_name,
            advertise_address
        );
        Ok(())
    }
}

/// Broadcast node state updates to target nodes
/// Returns (success_count, total_count)
pub async fn broadcast_node_states(
    nodes_to_broadcast: Vec<NodeState>,
    target_nodes: Vec<NodeState>,
    timeout: Option<Duration>,
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
            match try_ping(&target_node_clone, Some(ping_payload), None).await {
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

pub async fn try_ping(
    peer_node: &NodeState,
    payload: Option<gossip_message::Payload>,
    mtls_manager: Option<Arc<MTLSManager>>,
) -> Result<NodeUpdate, tonic::Status> {
    let peer_name = peer_node.name.clone();

    let peer_addr = SocketAddr::from_str(&peer_node.address).map_err(|e| {
        tonic::Status::invalid_argument(format!(
            "Invalid address for node {}: {}, {}",
            peer_name, peer_node.address, e
        ))
    })?;

    let connect_url = if mtls_manager.is_some() {
        format!("https://{peer_addr}")
    } else {
        format!("http://{peer_addr}")
    };

    let mut endpoint = Endpoint::from_shared(connect_url.clone())
        .map_err(|e| {
            tonic::Status::invalid_argument(format!(
                "Invalid endpoint for node {peer_name}: {connect_url}, {e}"
            ))
        })?
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10));

    if let Some(mtls_manager) = mtls_manager {
        mtls_manager.load_client_config().await.map_err(|e| {
            tonic::Status::unavailable(format!(
                "Failed to load mTLS client config for {peer_name}: {e}"
            ))
        })?;

        let tls_domain = endpoint.uri().host().map_or_else(
            || peer_name.clone(),
            |host| tls_server_name(host).to_owned(),
        );
        let ca_certificate = mtls_manager.load_ca_certificate().await.map_err(|e| {
            tonic::Status::unavailable(format!(
                "Failed to load mTLS CA certificate for {peer_name}: {e}"
            ))
        })?;

        endpoint = endpoint
            .tls_config(
                ClientTlsConfig::new()
                    .domain_name(tls_domain)
                    .ca_certificate(ca_certificate),
            )
            .map_err(|e| {
                tonic::Status::unavailable(format!(
                    "Failed to configure TLS endpoint for {peer_name}: {e}"
                ))
            })?;
    }

    let channel = endpoint.connect().await.map_err(|e| {
        log::warn!(
            "Failed to connect to peer {} {}: {}.",
            peer_name,
            peer_addr,
            e
        );
        tonic::Status::unavailable("Failed to connect to peer")
    })?;
    let mut client = gossip_client::GossipClient::new(channel)
        .max_decoding_message_size(MAX_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_MESSAGE_SIZE)
        .accept_compressed(tonic::codec::CompressionEncoding::Gzip)
        .send_compressed(tonic::codec::CompressionEncoding::Gzip);

    let ping_message = GossipMessage { payload };
    let response = client.ping_server(Request::new(ping_message)).await?;

    Ok(response.into_inner())
}

#[macro_export]
macro_rules! mesh_run {
    ($addr:expr, $init_peer:expr) => {{
        mesh_run!($addr.to_string(), $addr, $init_peer)
    }};

    ($name:expr, $addr:expr, $init_peer:expr) => {{
        tracing::info!("Starting mesh server : {}", $addr);
        use $crate::MeshServerBuilder;
        let (server, handler) =
            MeshServerBuilder::new($name.to_string(), $addr, $addr, $init_peer).build();
        #[expect(clippy::disallowed_methods, reason = "test macro: spawned server runs for the test lifetime and handler is returned for assertions")]
        tokio::spawn(async move {
            if let Err(e) = server.start().await {
                tracing::error!("Mesh server failed: {}", e);
            }
        });
        handler
    }};

    ($name:expr, $listener:expr, $addr:expr, $init_peer:expr) => {{
        tracing::info!("Starting mesh server : {}", $addr);
        use $crate::MeshServerBuilder;
        let (server, handler) =
            MeshServerBuilder::new($name.to_string(), $addr, $addr, $init_peer).build();
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
        sync::Once,
    };

    use rustls::crypto::ring;
    use tokio::net::TcpListener;
    use tonic::transport::{server::TcpIncoming, Identity, Server, ServerTlsConfig};
    use tracing as log;
    use tracing_subscriber::{
        filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
    };

    use super::{gossip::gossip_server::GossipServer, *};
    use crate::tests::{
        mtls_certs::MtlsTestCerts,
        test_utils::{bind_node, wait_for},
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
            MeshServerBuilder::new("A".to_string(), bind_addr, advertise_addr, None).build();

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

        let response = try_ping(
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
            None,
        )
        .await
        .unwrap();

        assert_eq!(response.address, advertise_addr.to_string());
        handler.shutdown();
    }

    #[tokio::test]
    #[ignore = "SWIM failure detection for hard-shutdown nodes needs many gossip rounds; flaky under parallel CI load"]
    async fn test_state_synchronization() {
        init();
        log::info!("Starting test_state_synchronization");

        // 1. Setup A-B cluster
        let (listener_a, addr_a) = bind_node().await;
        let handler_a = mesh_run!("A", listener_a, addr_a, None);
        let (listener_b, addr_b) = bind_node().await;
        let handler_b = mesh_run!("B", listener_b, addr_b, Some(addr_a));

        wait_for(
            || handler_a.state.read().len() == 2,
            Duration::from_secs(15),
            "A-B cluster formed",
        )
        .await;

        // 2. Add C and D
        let (listener_c, addr_c) = bind_node().await;
        let handler_c = mesh_run!("C", listener_c, addr_c, Some(addr_a));
        let (listener_d, addr_d) = bind_node().await;
        let handler_d = mesh_run!("D", listener_d, addr_d, Some(addr_c));

        wait_for(
            || handler_a.state.read().len() == 4,
            Duration::from_secs(30),
            "4-node cluster formed",
        )
        .await;

        // 3. Add E, let it join, then kill it
        {
            let (listener_e, addr_e) = bind_node().await;
            let handler_e = mesh_run!("E", listener_e, addr_e, Some(addr_d));

            wait_for(
                || handler_a.state.read().len() == 5,
                Duration::from_secs(30),
                "E joined cluster",
            )
            .await;

            handler_e.shutdown();
        }

        // 4. Gracefully shutdown D
        handler_d.graceful_shutdown().await.unwrap();

        // 5. Wait for D=Leaving and E=Down (not Alive) on all remaining nodes
        let check_statuses = |handler: &MeshServerHandler| {
            let state = handler.state.read();
            let d_leaving = state
                .get("D")
                .is_some_and(|n| n.status == NodeStatus::Leaving as i32);
            let e_not_alive = state
                .get("E")
                .is_some_and(|n| n.status != NodeStatus::Alive as i32);
            d_leaving && e_not_alive
        };

        for (handler, name) in [(&handler_a, "A"), (&handler_b, "B"), (&handler_c, "C")] {
            wait_for(
                || check_statuses(handler),
                Duration::from_secs(60),
                &format!("D=Leaving, E not Alive on node {name}"),
            )
            .await;
        }

        log::info!("All nodes converged to expected state");
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

    /// `try_ping` with mTLS on, the way the gossip loop dials a peer.
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
            rotation_check_interval: Duration::from_secs(300),
        });
        try_ping(
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
            Some(Arc::new(mtls)),
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
}
