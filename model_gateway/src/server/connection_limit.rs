//! A bound on the client connections the HTTP listener holds open at once.
//!
//! The listener accepts without bound: a connection flood, or clients piling
//! up behind a slow upstream, is bounded by the process's file descriptors
//! and memory, not by anything configured. The request-level bounds
//! (admission, the request timeout) see a connection only once it carries a
//! request, and the connection timeouts (`connection_timeouts`) end the
//! connections that carry none, each after its own delay. This module adds
//! the bound every other gateway has (a `maxconn`, a connection-limit
//! filter): at most `max_connections` client connections open at once,
//! counted from accept to close, with idle keep-alive connections,
//! connections inside a TLS handshake and upgraded connections (a
//! WebSocket) all counted.
//!
//! A connection beyond the bound is closed at once, in the acceptor, with
//! nothing read or written on it and before any TLS handshake: the cheapest
//! refusal there is, which is the point under a flood. The client sees its
//! connection end without a response (an end of stream, or a reset when it
//! had already sent its request), as it does when a listener's backlog
//! overflows, and should retry with backoff. Each refusal is counted in
//! `smg_http_connections_rejected_total`; the log carries one WARN per
//! minute while refusals happen, with the running count, so an episode shows
//! in the logs as well as on the counter. Readiness is unaffected: the bound
//! protects the process, which is not the orchestrator's cue to restart it.
//! Probes that reach the gateway through the bounded listener are subject to
//! the bound like every other connection; the dedicated probe listener
//! (`--health-check-port`) serves them outside it.
//!
//! The permit is held by the connection's stream, so it is released when the
//! connection closes, however it closes: by the client, by a connection
//! timeout, by the graceful drain, or at the end of an upgraded stream.

use std::{
    fmt,
    future::Future,
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};

use axum_server::accept::Accept;
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    sync::{OwnedSemaphorePermit, Semaphore},
};
use tracing::{debug, warn};

use crate::observability::{log_throttle::LogThrottle, metrics::Metrics};

/// The bound, as a semaphore with one permit per connection.
#[derive(Debug)]
struct Bound {
    max_connections: usize,
    permits: Arc<Semaphore>,
}

/// How often refusals reach the log at WARN while they go on: the first at
/// once, then at most one per interval, with the running count.
const WARN_INTERVAL: Duration = Duration::from_secs(60);

/// The bound and what became of the connections it saw, readable in
/// process: how many are open under it now, how many it refused. The
/// counter `smg_http_connections_rejected_total` carries the refusals.
pub struct ConnectionLimitStats {
    bound: Option<Bound>,
    rejected: AtomicU64,
    /// One WARN per [`WARN_INTERVAL`] while refusals go on, the shared
    /// throttle the routing refusals use too; the refusals between two
    /// lines are counted and reported as `suppressed` on the next one.
    warn_log: LogThrottle,
}

impl fmt::Debug for ConnectionLimitStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConnectionLimitStats")
            .field("bound", &self.bound)
            .field("rejected", &self.rejected())
            .finish_non_exhaustive()
    }
}

impl Default for ConnectionLimitStats {
    fn default() -> Self {
        Self::new(0)
    }
}

impl ConnectionLimitStats {
    fn new(max_connections: usize) -> Self {
        Self {
            bound: (max_connections > 0).then(|| Bound {
                max_connections,
                // The runtime refuses more permits than this at construction
                // (`ConfigValidator` names the bound; a library caller's
                // larger value is clamped rather than panicked on).
                permits: Arc::new(Semaphore::new(max_connections.min(Semaphore::MAX_PERMITS))),
            }),
            rejected: AtomicU64::new(0),
            warn_log: LogThrottle::new(1, WARN_INTERVAL),
        }
    }

    /// The bound; `None` without one.
    pub fn max_connections(&self) -> Option<usize> {
        self.bound.as_ref().map(|bound| bound.max_connections)
    }

    /// Connections open under the bound right now; `None` without a bound
    /// (nothing is counted then).
    pub fn open(&self) -> Option<usize> {
        self.bound.as_ref().map(|bound| {
            bound
                .max_connections
                .saturating_sub(bound.permits.available_permits())
        })
    }

    /// Connections closed at accept for arriving beyond the bound.
    pub fn rejected(&self) -> u64 {
        self.rejected.load(Ordering::Relaxed)
    }

    /// A permit for one more connection: `Ok(None)` without a bound,
    /// `Err(())` at the bound, with the refusal counted and logged.
    fn admit(&self) -> Result<Option<OwnedSemaphorePermit>, ()> {
        let Some(bound) = &self.bound else {
            return Ok(None);
        };
        match Arc::clone(&bound.permits).try_acquire_owned() {
            Ok(permit) => Ok(Some(permit)),
            Err(_) => {
                let max_connections = bound.max_connections;
                let rejected = self.rejected.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some(admitted) = self.warn_log.admit() {
                    warn!(
                        max_connections,
                        rejected,
                        suppressed = admitted.suppressed,
                        "The HTTP connection limit is reached: a connection beyond it is closed \
                         at once without a response; refusals are counted in \
                         smg_http_connections_rejected_total and logged here at most once a \
                         minute (suppressed: the refusals since the last line)"
                    );
                } else {
                    debug!(
                        max_connections,
                        rejected,
                        "Connection closed at accept: the HTTP connection limit is reached"
                    );
                }
                Metrics::record_http_connection_rejected();
                Err(())
            }
        }
    }
}

/// An acceptor that bounds the connections open through the inner acceptor
/// (the timeout-bounded plain-HTTP or TLS acceptor): a connection within the
/// bound carries its permit in its stream, a [`LimitedConnection`]; one
/// beyond it is closed at once, before the inner acceptor sees it, so a
/// refused TLS connection costs no handshake.
#[derive(Clone, Debug)]
pub struct ConnectionLimitAcceptor<A> {
    inner: A,
    stats: Arc<ConnectionLimitStats>,
}

impl<A> ConnectionLimitAcceptor<A> {
    /// Bounds `inner` to `max_connections` open connections; `0` is no bound
    /// (every connection passes straight through, uncounted).
    pub fn new(inner: A, max_connections: usize) -> Self {
        Self {
            inner,
            stats: Arc::new(ConnectionLimitStats::new(max_connections)),
        }
    }

    /// The bound's counters.
    pub fn stats(&self) -> Arc<ConnectionLimitStats> {
        Arc::clone(&self.stats)
    }
}

impl<A, I, S> Accept<I, S> for ConnectionLimitAcceptor<A>
where
    A: Accept<I, S>,
    A::Future: Send + 'static,
    A::Stream: Send + 'static,
    A::Service: Send + 'static,
{
    type Stream = LimitedConnection<A::Stream>;
    type Service = A::Service;
    type Future = Pin<Box<dyn Future<Output = io::Result<(Self::Stream, Self::Service)>> + Send>>;

    fn accept(&self, stream: I, service: S) -> Self::Future {
        let Ok(permit) = self.stats.admit() else {
            // `stream` goes out of scope here, unread and unwritten: that is
            // the close. The server drops a connection its acceptor refused.
            return Box::pin(std::future::ready(Err(io::Error::new(
                io::ErrorKind::ConnectionRefused,
                "the HTTP connection limit is reached",
            ))));
        };
        let inner = self.inner.accept(stream, service);
        Box::pin(async move {
            let (stream, service) = inner.await?;
            Ok((
                LimitedConnection {
                    inner: stream,
                    _permit: permit,
                },
                service,
            ))
        })
    }
}

/// A connection within the bound, holding its permit for as long as its
/// stream lives: through keep-alive, through an upgrade, until the close.
#[derive(Debug)]
pub struct LimitedConnection<I> {
    inner: I,
    _permit: Option<OwnedSemaphorePermit>,
}

impl<I> LimitedConnection<I> {
    /// The wrapped stream.
    pub fn get_ref(&self) -> &I {
        &self.inner
    }
}

impl<I: AsyncRead + Unpin> AsyncRead for LimitedConnection<I> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl<I: AsyncWrite + Unpin> AsyncWrite for LimitedConnection<I> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use std::{net::SocketAddr, time::Duration};

    use axum::{extract::ws::WebSocketUpgrade, routing::get, Router};
    use axum_server::Handle;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpStream,
        time::{sleep, timeout, Instant},
    };

    use super::*;
    use crate::server::{connection_timeouts::ConnectionTimeouts, http_acceptor};

    /// How long a test waits for a close, or for a permit to return.
    const WINDOW: Duration = Duration::from_secs(5);

    fn test_app() -> Router {
        Router::new().route("/ok", get(|| async { "ok" })).route(
            "/ws",
            get(|upgrade: WebSocketUpgrade| async {
                upgrade.on_upgrade(|mut socket| async move {
                    while let Some(Ok(message)) = socket.recv().await {
                        if socket.send(message).await.is_err() {
                            break;
                        }
                    }
                })
            }),
        )
    }

    /// Serves the test app through the production acceptors under
    /// `max_connections` on a loopback port, runs `check` against it and
    /// returns the bound's counters.
    async fn with_server<F, Fut>(max_connections: usize, check: F) -> Arc<ConnectionLimitStats>
    where
        F: FnOnce(SocketAddr, Arc<ConnectionLimitStats>) -> Fut,
        Fut: Future<Output = ()>,
    {
        let handle = Handle::new();
        let acceptor = ConnectionLimitAcceptor::new(
            http_acceptor(ConnectionTimeouts::disabled()),
            max_connections,
        );
        let stats = acceptor.stats();
        let server = axum_server::bind("127.0.0.1:0".parse().unwrap())
            .acceptor(acceptor)
            .handle(handle.clone());
        let serving = server.serve(test_app().into_make_service_with_connect_info::<SocketAddr>());
        let checking = async {
            let addr = handle
                .listening()
                .await
                .expect("the test server did not bind");
            check(addr, Arc::clone(&stats)).await;
            handle.shutdown();
        };
        let (served, ()) = tokio::join!(serving, checking);
        served.expect("the test server failed");
        stats
    }

    async fn connect(addr: SocketAddr) -> TcpStream {
        TcpStream::connect(addr).await.expect("connect")
    }

    /// Sends a keep-alive GET and reads the whole response.
    async fn get_ok(stream: &mut TcpStream) {
        stream
            .write_all(b"GET /ok HTTP/1.1\r\nHost: test\r\n\r\n")
            .await
            .unwrap();
        let mut bytes = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = timeout(WINDOW, stream.read(&mut buf))
                .await
                .expect("the response did not arrive")
                .expect("read");
            assert!(n > 0, "the connection closed inside the response");
            bytes.extend_from_slice(&buf[..n]);
            if bytes.ends_with(b"ok") {
                break;
            }
        }
        let response = String::from_utf8_lossy(&bytes);
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    }

    /// Sends a request on a connection the server may already have closed,
    /// then waits up to `within` for the close: the bytes that arrived
    /// before it, or `None` while the connection is still open at the end
    /// of the window. A reset counts as the close it is.
    async fn request_then_closed_within(stream: &mut TcpStream, within: Duration) -> Option<usize> {
        let _ = stream
            .write_all(b"GET /ok HTTP/1.1\r\nHost: test\r\n\r\n")
            .await;
        let started = Instant::now();
        let mut received = 0;
        let mut buf = [0u8; 4096];
        loop {
            let left = within.checked_sub(started.elapsed())?;
            match timeout(left, stream.read(&mut buf)).await {
                Ok(Ok(0)) | Ok(Err(_)) => return Some(received),
                Ok(Ok(n)) => received += n,
                Err(_) => return None,
            }
        }
    }

    /// Waits up to `within` for the bound's open count to reach `open`.
    async fn open_settles_at(stats: &ConnectionLimitStats, open: usize, within: Duration) {
        let deadline = Instant::now() + within;
        while stats.open() != Some(open) {
            assert!(
                Instant::now() < deadline,
                "open connections stayed at {:?}, expected {open}",
                stats.open()
            );
            sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test]
    async fn a_connection_beyond_the_bound_is_closed_at_once_and_its_slot_returns_with_a_close() {
        let stats = with_server(2, |addr, stats| async move {
            let mut first = connect(addr).await;
            let mut second = connect(addr).await;
            get_ok(&mut first).await;
            get_ok(&mut second).await;
            assert_eq!(stats.open(), Some(2));

            // The third connection is closed with nothing written on it, the
            // request it sent unanswered.
            let mut third = connect(addr).await;
            assert_eq!(
                request_then_closed_within(&mut third, WINDOW).await,
                Some(0),
                "the refused connection got bytes, or stayed open"
            );
            assert_eq!(stats.rejected(), 1);
            // The two within the bound are untouched and still served.
            get_ok(&mut first).await;
            get_ok(&mut second).await;

            // A close returns its permit, and the next connection is served.
            drop(first);
            open_settles_at(&stats, 1, WINDOW).await;
            let mut fourth = connect(addr).await;
            get_ok(&mut fourth).await;
            assert_eq!(stats.open(), Some(2));
            assert_eq!(stats.rejected(), 1);
        })
        .await;
        assert_eq!(stats.max_connections(), Some(2));
    }

    /// The WARN is admitted for the first refusal and then once per
    /// [`WARN_INTERVAL`], the refusals in between counted as suppressed on
    /// the next line: the stats' throttle is wired with that limit and
    /// window (its boundaries are the throttle's own tests' business).
    #[test]
    fn a_refusal_is_logged_at_warn_once_per_interval() {
        let stats = ConnectionLimitStats::new(1);
        let start = std::time::Instant::now();
        let admitted = |at: Duration| stats.warn_log.admit_at(start + at).map(|a| a.suppressed);
        assert_eq!(admitted(Duration::ZERO), Some(0), "the first refusal");
        assert_eq!(admitted(Duration::from_secs(1)), None);
        assert_eq!(admitted(WARN_INTERVAL - Duration::from_millis(1)), None);
        assert_eq!(
            admitted(WARN_INTERVAL),
            Some(2),
            "the first after the interval, with the two in between"
        );
        assert_eq!(admitted(WARN_INTERVAL + Duration::from_secs(1)), None);
        assert_eq!(
            admitted(WARN_INTERVAL * 2 + Duration::from_secs(1)),
            Some(1)
        );
    }

    #[tokio::test]
    async fn without_a_bound_every_connection_is_served_and_nothing_is_counted() {
        let stats = with_server(0, |addr, stats| async move {
            let mut connections = Vec::new();
            for _ in 0..20 {
                let mut connection = connect(addr).await;
                get_ok(&mut connection).await;
                connections.push(connection);
            }
            assert_eq!(stats.open(), None);
        })
        .await;
        assert_eq!(stats.max_connections(), None);
        assert_eq!(stats.rejected(), 0);
    }

    /// An upgraded connection keeps its permit for as long as the upgraded
    /// stream lives, not only until its service is done: with a bound of
    /// one, a WebSocket holds the bound against a second connection and
    /// releases it when it closes.
    #[tokio::test]
    async fn an_upgraded_connection_holds_its_permit_until_the_stream_closes() {
        let stats = with_server(1, |addr, stats| async move {
            let mut websocket = connect(addr).await;
            websocket
                .write_all(
                    b"GET /ws HTTP/1.1\r\nHost: test\r\nConnection: Upgrade\r\n\
                      Upgrade: websocket\r\nSec-WebSocket-Version: 13\r\n\
                      Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
                )
                .await
                .unwrap();
            let mut head = [0u8; 12];
            timeout(WINDOW, websocket.read_exact(&mut head))
                .await
                .expect("the upgrade response did not arrive")
                .expect("read");
            assert_eq!(&head, b"HTTP/1.1 101");
            // The service of the upgraded connection is done; the stream is
            // not, and the bound is still held.
            sleep(Duration::from_millis(200)).await;
            assert_eq!(stats.open(), Some(1));
            let mut second = connect(addr).await;
            assert_eq!(
                request_then_closed_within(&mut second, WINDOW).await,
                Some(0)
            );
            assert_eq!(stats.rejected(), 1);

            drop(websocket);
            open_settles_at(&stats, 0, WINDOW).await;
            let mut third = connect(addr).await;
            get_ok(&mut third).await;
        })
        .await;
        assert_eq!(stats.rejected(), 1);
    }
}
