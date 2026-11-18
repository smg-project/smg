//! What the crate's tests build their workers with: a health-check
//! configuration that leaves the worker as the test built it, a ready
//! worker and a worker of a given role built with it, a labelled gRPC
//! worker and the registry holding a test's workers, and the KV cache event
//! that clears a worker's cache. [`RefusedAddress`] is the loopback address
//! nothing answers at: a dead worker's port, held for as long as the test
//! needs it. [`NoEngine`] is an engine connection with no engine behind it,
//! and [`CountingConnector`] the connector that hands those out.
//! [`LoadReaders`] is the routing side a load monitor's test stands beside
//! it. A test of a metric renders its scrape with [`render_with_recorder`]
//! and reads it with [`assert_metric`] and [`sorted_series`].

use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Weak,
    },
};

use async_trait::async_trait;
use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::{
    model_card::ModelCard,
    worker::{
        HealthCheckConfig, ProfileOptions, RuntimeType, WorkerLoadResponse, WorkerStatus,
        WorkerType,
    },
};
use smg_grpc_client::common_proto::{kv_cache_event, KvCacheCleared, KvCacheEvent};
use tokio::net::{TcpListener, TcpSocket};

use crate::worker::{
    connection::engine::{AdminAnswer, EngineConnection, EngineConnector, ZmqLink},
    BasicWorkerBuilder, ConnectionMode, LoadConsumer, Worker, WorkerRegistry,
};

/// The health checker disabled, everything else default: a test's worker
/// keeps the state it was built with, with no probe behind it.
pub fn no_health_check() -> HealthCheckConfig {
    HealthCheckConfig {
        disable_health_check: true,
        ..Default::default()
    }
}

/// A ready HTTP worker at `url` serving `model`, with no health checker
/// behind it.
pub fn ready_worker(url: &str, model: &str) -> Arc<dyn Worker> {
    let worker = BasicWorkerBuilder::new(url)
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Http)
        .model(ModelCard::new(model))
        .health_config(no_health_check())
        .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// A worker of `worker_type` at `url` with no health checker behind it: what
/// a test fills a pool with when no health checker may run.
pub fn worker_of(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .health_config(no_health_check())
            .build(),
    )
}

/// A regular gRPC worker of `runtime` at `url` serving `model`, with
/// `labels` and no health checker behind it: what the multimodal router
/// tests put in their pools.
pub fn grpc_worker(
    url: &str,
    model: &str,
    runtime: RuntimeType,
    labels: &[(&str, &str)],
) -> Arc<dyn Worker> {
    let mut builder = BasicWorkerBuilder::new(url)
        .model(ModelCard::new(model))
        .worker_type(WorkerType::Regular)
        .runtime_type(runtime)
        .connection_mode(ConnectionMode::Grpc)
        .health_config(no_health_check());
    for (key, value) in labels {
        builder = builder.label(*key, *value);
    }
    Arc::new(builder.build())
}

/// A registry holding `workers`, each registered as the test built it.
#[expect(
    clippy::expect_used,
    reason = "a test fixture: a worker the registry refuses is a test failure"
)]
pub fn registry_with(workers: Vec<Arc<dyn Worker>>) -> WorkerRegistry {
    let registry = WorkerRegistry::new();
    for worker in workers {
        registry.register(worker).expect("register");
    }
    registry
}

/// The KV cache event that clears a worker's whole cache: the state the KV
/// event monitor's and the index backends' tests start over from.
pub fn kv_cache_cleared() -> KvCacheEvent {
    KvCacheEvent {
        event_id: 0,
        data: Some(kv_cache_event::Data::Cleared(KvCacheCleared::default())),
    }
}

/// A loopback address nothing answers at, refused for as long as this value
/// lives.
///
/// The socket behind it is bound and never listening: the host's own stack
/// refuses every connection at once, the way a dead worker's port does, and
/// the port stays taken, so no other test's `bind("127.0.0.1:0")` can draw
/// it in the meantime. A listener bound and dropped for the same purpose
/// gives its port back, and a neighbouring test's stub then answers in the
/// unreachable side's place: a `404` or `405` where a transport failure was
/// expected, or a request held until the test's bound.
///
/// Keep the value in a named binding for the whole case: a `_` pattern or a
/// temporary drops it, and the port is free again. A case whose dead worker
/// comes up later starts it on [`RefusedAddress::listen`]: the same socket,
/// so the port is never free in between. A case whose worker comes up, dies
/// and comes up again reserves with [`RefusedAddress::reserve_shared`].
#[must_use = "the address is refused only while this value lives"]
pub struct RefusedAddress {
    reservation: TcpSocket,
    addr: SocketAddr,
}

impl RefusedAddress {
    /// Reserve a loopback port of the system's choice, and refuse it. For a
    /// port nothing answers at, or one a worker comes up on once: its only
    /// socket is the one [`RefusedAddress::listen`] hands over.
    #[expect(
        clippy::expect_used,
        reason = "a test fixture: a loopback socket the system cannot give is a test failure"
    )]
    pub fn reserve() -> Self {
        let socket = TcpSocket::new_v4().expect("a TCP socket");
        socket
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("a loopback port of the system's choice");
        let addr = socket.local_addr().expect("the bound address");
        Self {
            reservation: socket,
            addr,
        }
    }

    /// Stop refusing and listen on the reserved socket itself, as a dead
    /// worker's port does when the worker comes up. No other test can bind
    /// the port, or be handed it, before the caller's server is on it.
    #[expect(
        clippy::expect_used,
        reason = "a test fixture: a reserved socket that cannot listen is a test failure"
    )]
    pub fn listen(self) -> TcpListener {
        self.reservation
            .listen(1024)
            .expect("the reserved socket listens")
    }

    /// The address, `127.0.0.1:<port>`.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The port alone, for a worker registered with a bootstrap port.
    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    /// The address as a plain HTTP URL, `http://127.0.0.1:<port>`.
    pub fn http_url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

// A reservation shares its port with its listeners through `SO_REUSEPORT`, a
// Unix socket option.
#[cfg(unix)]
impl RefusedAddress {
    /// Reserve a loopback port of the system's choice for a worker that
    /// comes up on it more than once, and refuse it. The reservation shares
    /// the port with the listeners of [`RefusedAddress::listen_shared`], one
    /// for each life of the worker, and outlives them: the port is refused
    /// again when a listener is gone, and stays taken between two lives.
    /// Unix alone: a test module that calls it is gated `#![cfg(unix)]`.
    #[expect(
        clippy::expect_used,
        reason = "a test fixture: a loopback socket the system cannot give is a test failure"
    )]
    pub fn reserve_shared() -> Self {
        let socket = Self::sharing_socket();
        socket
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("a loopback port of the system's choice");
        let addr = socket.local_addr().expect("the bound address");
        Self {
            reservation: socket,
            addr,
        }
    }

    /// A listener beside the reservation, for one life of the worker:
    /// connections are answered while it lives and refused again after it.
    /// The life before must be gone first: a second listener alive at once
    /// is not refused, and the two share the connections.
    /// Only a reservation of [`RefusedAddress::reserve_shared`] shares its
    /// port: on any other the bind is refused, and the test fails here.
    #[expect(
        clippy::expect_used,
        reason = "a test fixture: a shared port that takes no listener is a test failure"
    )]
    pub fn listen_shared(&self) -> TcpListener {
        let socket = Self::sharing_socket();
        socket
            .bind(self.addr)
            .expect("a port reserved by `reserve_shared`");
        socket.listen(1024).expect("the sharing socket listens")
    }

    /// A socket that shares the port it binds with every socket made here.
    #[expect(
        clippy::expect_used,
        reason = "a test fixture: a socket the system cannot give is a test failure"
    )]
    fn sharing_socket() -> TcpSocket {
        let socket = TcpSocket::new_v4().expect("a TCP socket");
        socket
            .set_reuseport(true)
            .expect("a socket that shares its port");
        socket
    }
}

/// A removed worker nothing holds any more: a dead witness for the series
/// store's `retire`.
pub fn gone(url: &str) -> Weak<dyn Worker> {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new(url).build());
    Arc::downgrade(&worker)
}

/// A removed worker a handle still holds: the worker and its witness.
pub fn held(url: &str) -> (Arc<dyn Worker>, Weak<dyn Worker>) {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new(url).build());
    let witness = Arc::downgrade(&worker);
    (worker, witness)
}

/// The routing side a test stands beside the load monitor: it says whether
/// a reader wants worker loads, and reconciles nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadReaders {
    /// No reader wants loads, as under a policy that reads none.
    Blind,
    /// A reader wants loads, as under a load-aware policy.
    Aware,
}

impl LoadConsumer for LoadReaders {
    fn needs_loads(&self) -> bool {
        matches!(self, Self::Aware)
    }

    fn reconcile_in_flight(&self, _worker: &dyn Worker) {}
}

/// Run `f` under a thread-local Prometheus recorder and return the rendered
/// scrape, the text a metrics endpoint serves.
pub fn render_with_recorder(f: impl FnOnce()) -> String {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, f);
    handle.render()
}

/// Assert the rendered scrape has a line for `name` carrying every label in
/// `labels` and ending in `value`. Label order is exporter-defined, so this
/// matches on substrings rather than a fixed label set.
#[expect(
    clippy::panic,
    reason = "a test assertion: a missing series is a test failure"
)]
pub fn assert_metric(rendered: &str, name: &str, labels: &[&str], value: &str) {
    let line = rendered
        .lines()
        .find(|l| l.starts_with(&format!("{name}{{")))
        .unwrap_or_else(|| panic!("metric {name} missing; rendered:\n{rendered}"));
    for label in labels {
        assert!(line.contains(label), "{name} missing label {label}: {line}");
    }
    assert!(
        line.ends_with(&format!(" {value}")),
        "{name} expected value {value}: {line}"
    );
}

/// The lines of a scrape that are series (not `# HELP` or `# TYPE`),
/// sorted, for comparing two recorders' output.
pub fn sorted_series(rendered: &str) -> Vec<&str> {
    let mut series: Vec<&str> = rendered
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .collect();
    series.sort_unstable();
    series
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;

    use super::RefusedAddress;

    /// A connection to the address is refused at once, and no other bind
    /// takes the port while the value lives.
    #[tokio::test]
    async fn the_address_refuses_connections_and_keeps_its_port() {
        let refused = RefusedAddress::reserve();
        let error = tokio::net::TcpStream::connect(refused.addr())
            .await
            .expect_err("nothing answers there");
        assert_eq!(error.kind(), ErrorKind::ConnectionRefused, "{error}");
        let taken = std::net::TcpListener::bind(refused.addr()).expect_err("the port is held");
        assert_eq!(taken.kind(), ErrorKind::AddrInUse, "{taken}");
        assert_eq!(
            refused.http_url(),
            format!("http://127.0.0.1:{}", refused.port())
        );
    }

    /// The reservation becomes the listener: the same address, refused until
    /// the hand-over and answered after it.
    #[tokio::test]
    async fn the_reserved_socket_listens_on_its_own_address() {
        let refused = RefusedAddress::reserve();
        let addr = refused.addr();
        let error = tokio::net::TcpStream::connect(addr)
            .await
            .expect_err("nothing answers there yet");
        assert_eq!(error.kind(), ErrorKind::ConnectionRefused, "{error}");

        let listener = refused.listen();
        assert_eq!(listener.local_addr().expect("the listener's address"), addr);
        let (accepted, connected) =
            tokio::join!(listener.accept(), tokio::net::TcpStream::connect(addr));
        accepted.expect("the listener accepts");
        connected.expect("the connection is answered");
    }

    /// A shared reservation takes a listener twice over: refused, and no
    /// plain bind's, before each of them, answered while one is up, and
    /// refused again after the last.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_shared_reservation_is_refused_between_its_listeners() {
        let reserved = RefusedAddress::reserve_shared();
        let addr = reserved.addr();
        for life in 0..2 {
            let error = tokio::net::TcpStream::connect(addr)
                .await
                .expect_err("nothing listens there");
            assert_eq!(
                error.kind(),
                ErrorKind::ConnectionRefused,
                "before life {life}: {error}"
            );
            let taken = std::net::TcpListener::bind(addr).expect_err("the port is held");
            assert_eq!(
                taken.kind(),
                ErrorKind::AddrInUse,
                "before life {life}: {taken}"
            );

            let listener = reserved.listen_shared();
            assert_eq!(listener.local_addr().expect("the listener's address"), addr);
            let (accepted, connected) =
                tokio::join!(listener.accept(), tokio::net::TcpStream::connect(addr));
            accepted.expect("the listener accepts");
            connected.expect("the connection is answered");
        }
        let error = tokio::net::TcpStream::connect(addr)
            .await
            .expect_err("nothing listens there");
        assert_eq!(error.kind(), ErrorKind::ConnectionRefused, "{error}");
    }

    /// Shared reservations alive together hold a port each: a port is shared
    /// with its own reservation's listeners, never handed to a second
    /// reservation, so two cases of one binary never serve each other.
    #[cfg(unix)]
    #[test]
    fn shared_reservations_alive_together_hold_distinct_ports() {
        let reserved: Vec<RefusedAddress> =
            (0..256).map(|_| RefusedAddress::reserve_shared()).collect();
        let ports: std::collections::HashSet<u16> =
            reserved.iter().map(RefusedAddress::port).collect();
        assert_eq!(
            ports.len(),
            reserved.len(),
            "two reservations hold one port"
        );
    }
}

/// An engine connection with no engine behind it: alive, without admin
/// calls, and every call refused as unavailable. What a test hands a worker
/// whose connection only has to exist.
#[derive(Debug, Default)]
pub struct NoEngine;

fn no_engine<T>() -> Result<T, tonic::Status> {
    Err(tonic::Status::unavailable(
        "no engine behind the test's connection",
    ))
}

#[async_trait]
impl EngineConnection for NoEngine {
    fn carries_admin_calls(&self) -> bool {
        false
    }

    fn is_alive(&self) -> bool {
        true
    }

    async fn health_check(&self) -> Result<bool, tonic::Status> {
        no_engine()
    }

    async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
        no_engine()
    }

    async fn flush_cache(&self, _timeout_s: f32) -> Result<AdminAnswer, tonic::Status> {
        no_engine()
    }

    async fn start_profile(&self, _options: &ProfileOptions) -> Result<AdminAnswer, tonic::Status> {
        no_engine()
    }

    async fn stop_profile(&self) -> Result<AdminAnswer, tonic::Status> {
        no_engine()
    }

    async fn kv_engine_id(&self) -> Result<Option<String>, tonic::Status> {
        no_engine()
    }
}

/// A connector that hands out [`NoEngine`] connections and counts the
/// connects it was asked for.
#[derive(Debug, Default)]
pub struct CountingConnector {
    connects: AtomicUsize,
}

impl CountingConnector {
    /// How many connects it was asked for.
    pub fn connects(&self) -> usize {
        self.connects.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl EngineConnector for CountingConnector {
    async fn connect_grpc(
        &self,
        _base_url: &str,
        _runtime: RuntimeType,
    ) -> Result<Arc<dyn EngineConnection>, String> {
        self.connects.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(NoEngine))
    }

    async fn connect_zmq(&self, _link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
        self.connects.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(NoEngine))
    }
}
