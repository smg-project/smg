//! What the crate's capture tests drive a request through: a worker at an
//! address of the test's choosing, a worker source over such workers, the
//! HTTP client they share, an address nothing answers at, a provider's
//! answer of the test's making, how a test reads an answer's body back as
//! text, and the counting allocator every unit test of the crate runs
//! under (the allocation pins read it).

use std::{
    fmt,
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use axum::{
    body::to_bytes,
    http::{header::CONTENT_TYPE, HeaderMap},
    response::Response,
};
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

/// A provider's answer of the test's making: `status`, the content type
/// when one is given, and `body`.
pub(crate) fn upstream(
    status: u16,
    content_type: Option<&str>,
    body: impl Into<reqwest::Body>,
) -> reqwest::Response {
    let mut builder = http::Response::builder().status(status);
    if let Some(ct) = content_type {
        builder = builder.header(CONTENT_TYPE, ct);
    }
    reqwest::Response::from(builder.body(body.into()).expect("a response"))
}

/// The whole body of `response`, as text.
pub(crate) async fn text_of(response: Response) -> String {
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("a body");
    String::from_utf8(bytes.to_vec()).expect("utf-8")
}

/// A worker at one address, serving model `m`, with no API key of its own.
pub(crate) struct StubWorker {
    url: String,
    redacted: String,
    healthy: bool,
    client: reqwest::Client,
}

impl StubWorker {
    /// A healthy worker at `url`.
    pub(crate) fn at(url: &str) -> Arc<dyn ExternalWorker> {
        Arc::new(Self {
            url: url.to_string(),
            redacted: crate::worker::redacted_url(url).into_owned(),
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
            redacted: crate::worker::redacted_url(url).into_owned(),
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

    fn redacted_url(&self) -> &str {
        &self.redacted
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

/// The crate's unit tests run under a counting allocator, so an allocation
/// pin can read what a call allocated: the system allocator with a
/// per-thread count of allocation and reallocation calls. Test builds only.
pub(crate) mod counting_alloc {
    #![expect(
        unsafe_code,
        reason = "a counting global allocator is the only way to see what a call allocated; test builds only"
    )]

    use std::{
        alloc::{GlobalAlloc, Layout, System},
        cell::Cell,
    };

    thread_local! {
        /// Allocation and reallocation calls this thread made.
        static CALLS: Cell<usize> = const { Cell::new(0) };
    }

    /// Allocation and reallocation calls this thread made so far.
    pub(crate) fn calls() -> usize {
        CALLS.with(Cell::get)
    }

    struct Counting;

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            CALLS.with(|calls| calls.set(calls.get() + 1));
            // SAFETY: the caller's layout contract, forwarded to the system allocator.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: `ptr` came from `alloc` or `realloc` with this layout.
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            CALLS.with(|calls| calls.set(calls.get() + 1));
            // SAFETY: `ptr` came from this allocator with `layout`; the new size is the caller's.
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static GLOBAL: Counting = Counting;

    /// The count moves by one for one allocation and one reallocation,
    /// and not for a free: what every pin in the crate reads.
    #[test]
    fn the_count_follows_this_threads_allocation_calls() {
        let before = calls();
        let mut bytes: Vec<u8> = Vec::with_capacity(8);
        assert_eq!(calls() - before, 1, "one allocation");
        bytes.extend(std::iter::repeat_n(0u8, 64));
        assert_eq!(calls() - before, 2, "one reallocation on top");
        drop(bytes);
        assert_eq!(calls() - before, 2, "a free is not counted");
    }
}
