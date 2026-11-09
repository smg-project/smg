//! What the crate's tests build their workers with: a health-check
//! configuration that leaves the worker as the test built it, a ready
//! worker and a worker of a given role built with it, a labelled gRPC
//! worker and the registry holding a test's workers, the admission attempt
//! that selects one worker, and the KV cache event that clears a worker's
//! cache. [`RefusedAddress`] is the loopback address nothing answers at: a
//! dead worker's port, held for as long as the test needs it.

use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
};

use openai_protocol::{
    model_card::ModelCard,
    worker::{HealthCheckConfig, RuntimeType, WorkerStatus, WorkerType},
};
use smg_grpc_client::common_proto::{kv_cache_event, KvCacheCleared, KvCacheEvent};
use tokio::net::TcpSocket;

use crate::worker::{
    prefill_admission::{PrefillAdmissionAttempt, PrefillSelectionContext},
    BasicWorkerBuilder, ConnectionMode, Worker, WorkerRegistry,
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

/// The attempt a test makes when it wants exactly `worker`: it is selected,
/// with a unit payload, whenever it has capacity.
pub fn select_worker(
    worker: &Arc<dyn Worker>,
) -> impl for<'a> FnMut(&PrefillSelectionContext<'a>) -> PrefillAdmissionAttempt<()> + Send {
    let worker = Arc::clone(worker);
    move |capacity| capacity.select(Arc::clone(&worker), ())
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
/// temporary drops it, and the port is free again.
#[must_use = "the address is refused only while this value lives"]
pub struct RefusedAddress {
    _reservation: TcpSocket,
    addr: SocketAddr,
}

impl RefusedAddress {
    /// Reserve a loopback port of the system's choice, and refuse it.
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
            _reservation: socket,
            addr,
        }
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
}
