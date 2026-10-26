//! What the crate's capture tests drive a request through: a worker at an
//! address of the test's choosing, a worker source over such workers, the
//! HTTP client they share, and an address nothing answers at.

use std::{
    fmt,
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use axum::{http::HeaderMap, response::Response};
use openai_protocol::worker::ProviderType;
use rustls::crypto::ring::default_provider;
use tokio::net::TcpSocket;

use crate::{
    error,
    worker::{ExternalWorker, LoadHold, SelectWorkerRequest, WorkerSource, WorkerStats},
    RetryConfig,
};

/// A worker registered with credentials in its URL, at a port nobody listens
/// on: a request to it fails at the connection, and every line written on
/// the way must show the address as [`MASKED_WORKER_URL`].
pub(crate) const CREDENTIALED_WORKER_URL: &str = "https://user:secret@127.0.0.1:1";

/// [`CREDENTIALED_WORKER_URL`] as a log line or a message shows it.
pub(crate) const MASKED_WORKER_URL: &str = "https://***@127.0.0.1:1";

/// The tests' HTTP client, the process TLS crypto provider installed first.
#[expect(
    clippy::disallowed_methods,
    reason = "a test binary's client factory; the process TLS crypto provider is installed first"
)]
pub(crate) fn http_client() -> reqwest::Client {
    let _ = default_provider().install_default();
    reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("a client")
}

/// A worker at one address, serving model `m`, with no API key of its own.
pub(crate) struct StubWorker {
    url: String,
    healthy: bool,
    client: reqwest::Client,
}

impl StubWorker {
    /// A healthy worker at `url`.
    pub(crate) fn at(url: &str) -> Arc<dyn ExternalWorker> {
        Arc::new(Self {
            url: url.to_string(),
            healthy: true,
            client: http_client(),
        })
    }

    /// A worker at `url` whose health check has failed.
    #[cfg_attr(
        not(feature = "openai"),
        expect(dead_code, reason = "read by the health answers' tests alone")
    )]
    pub(crate) fn unhealthy_at(url: &str) -> Arc<dyn ExternalWorker> {
        Arc::new(Self {
            url: url.to_string(),
            healthy: false,
            client: http_client(),
        })
    }
}

impl fmt::Debug for StubWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StubWorker")
            .field("url", &self.url)
            .field("healthy", &self.healthy)
            .finish()
    }
}

impl ExternalWorker for StubWorker {
    fn url(&self) -> &str {
        &self.url
    }

    fn api_key(&self) -> Option<&String> {
        None
    }

    fn model_id(&self) -> &str {
        "m"
    }

    fn is_healthy(&self) -> bool {
        self.healthy
    }

    fn provider_for_model(&self, _model_id: &str) -> Option<&ProviderType> {
        None
    }

    fn record_outcome(&self, _status_code: u16) {}

    fn http_client(&self) -> &reqwest::Client {
        &self.client
    }

    fn hold_load(&self, _headers: Option<&HeaderMap>) -> LoadHold {
        Box::new(())
    }
}

/// A worker source over a fixed set of workers; it selects none of them.
#[cfg_attr(
    not(any(feature = "openai", feature = "gemini")),
    expect(
        dead_code,
        reason = "read by the health answers' and the Gemini step's tests alone"
    )
)]
#[derive(Debug, Default)]
pub(crate) struct StubWorkers(pub(crate) Vec<Arc<dyn ExternalWorker>>);

#[async_trait]
impl WorkerSource for StubWorkers {
    async fn select(
        &self,
        _req: &SelectWorkerRequest<'_>,
    ) -> Result<Arc<dyn ExternalWorker>, Response> {
        Err(error::service_unavailable(
            "service_unavailable",
            "No worker selected",
        ))
    }

    fn retry_config(&self, _model_id: &str) -> Option<RetryConfig> {
        None
    }

    fn stats(&self) -> WorkerStats {
        WorkerStats::default()
    }

    fn external_workers(&self) -> Vec<Arc<dyn ExternalWorker>> {
        self.0.clone()
    }
}

/// A loopback address nothing answers at, refused for as long as this value
/// lives (the shape of the gRPC client's `channel::testing::RefusedEndpoint`).
///
/// The socket behind it is bound and never listening: the host's own stack
/// refuses every connection at once, the way an unreachable peer's port
/// does, and the port stays taken, so no other test's `bind("127.0.0.1:0")`
/// can draw it in the meantime. A listener bound and dropped for the same
/// purpose gives its port back, and a neighbouring test's stub then answers
/// in the unreachable peer's place.
///
/// Keep the value in a named binding for the whole case: a `_` pattern or a
/// temporary drops it, and the port is free again.
#[must_use = "the address is refused only while this value lives"]
pub(crate) struct RefusedAddress {
    _reservation: TcpSocket,
    addr: SocketAddr,
}

impl RefusedAddress {
    /// Reserve a loopback port of the system's choice, and refuse it.
    pub(crate) fn reserve() -> Self {
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
    pub(crate) fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The port alone, for a URL built around it.
    pub(crate) fn port(&self) -> u16 {
        self.addr.port()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::ErrorKind,
        net::{TcpListener, TcpStream},
    };

    use super::RefusedAddress;

    /// A connection to the address is refused at once, and no other bind
    /// takes the port while the value lives.
    #[test]
    fn the_address_refuses_connections_and_keeps_its_port() {
        let refused = RefusedAddress::reserve();
        let error = TcpStream::connect(refused.addr()).expect_err("nothing answers there");
        assert_eq!(error.kind(), ErrorKind::ConnectionRefused, "{error}");
        let taken = TcpListener::bind(refused.addr()).expect_err("the port is held");
        assert_eq!(taken.kind(), ErrorKind::AddrInUse, "{taken}");
        assert_eq!(refused.addr().port(), refused.port());
    }
}
