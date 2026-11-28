//! A bound on the client connections an HTTP listener holds open at once.
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
//! (`--health-check-port`) serves them outside it. The data listener's bound
//! (`--max-connections`) and the admin listener's (`--admin-max-connections`)
//! are two acceptors over their own permits: the counter is one for both by
//! design, and the WARN and the refusal name the listener.
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

/// The listener a bound belongs to, for the words of its refusals: the
/// counter `smg_http_connections_rejected_total` is one for both by design,
/// so the log and the closed connection's error say which listener refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundedListener {
    /// The data listener, bounded by `--max-connections`.
    Data,
    /// The admin listener, bounded by `--admin-max-connections`.
    Admin,
}

impl BoundedListener {
    /// The words of a refusal, the listener and its flag: shared by the
    /// WARN, the DEBUG line and the closed connection's error.
    pub(crate) fn limit_reached(self) -> &'static str {
        match self {
            Self::Data => "the data listener's connection limit (--max-connections) is reached",
            Self::Admin => {
                "the admin listener's connection limit (--admin-max-connections) is reached"
            }
        }
    }

    /// The listener's name, the `listener` field of the log lines.
    fn name(self) -> &'static str {
        match self {
            Self::Data => "data",
            Self::Admin => "admin",
        }
    }
}

/// The bound, as a semaphore with one permit per connection.
#[derive(Debug)]
struct Bound {
    listener: BoundedListener,
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
        Self::new(BoundedListener::Data, 0)
    }
}

impl ConnectionLimitStats {
    fn new(listener: BoundedListener, max_connections: usize) -> Self {
        Self {
            bound: (max_connections > 0).then(|| Bound {
                listener,
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

    /// A permit for one more connection: `Ok(None)` without a bound, the
    /// refusing listener at the bound, with the refusal counted and logged.
    fn admit(&self) -> Result<Option<OwnedSemaphorePermit>, BoundedListener> {
        let Some(bound) = &self.bound else {
            return Ok(None);
        };
        match Arc::clone(&bound.permits).try_acquire_owned() {
            Ok(permit) => Ok(Some(permit)),
            Err(_) => {
                let listener = bound.listener;
                let max_connections = bound.max_connections;
                let rejected = self.rejected.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some(admitted) = self.warn_log.admit() {
                    warn!(
                        listener = listener.name(),
                        max_connections,
                        rejected,
                        suppressed = admitted.suppressed,
                        "{}: a connection beyond it is closed at once without a response; \
                         refusals are counted in smg_http_connections_rejected_total and \
                         logged here at most once a minute (suppressed: the refusals since \
                         the last line)",
                        listener.limit_reached()
                    );
                } else {
                    debug!(
                        listener = listener.name(),
                        max_connections,
                        rejected,
                        "Connection closed at accept: {}",
                        listener.limit_reached()
                    );
                }
                Metrics::record_http_connection_rejected();
                Err(listener)
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
    /// Bounds `inner`, the `listener`'s acceptor, to `max_connections` open
    /// connections; `0` is no bound (every connection passes straight
    /// through, uncounted).
    pub fn new(inner: A, listener: BoundedListener, max_connections: usize) -> Self {
        Self {
            inner,
            stats: Arc::new(ConnectionLimitStats::new(listener, max_connections)),
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
        let permit = match self.stats.admit() {
            Ok(permit) => permit,
            // `stream` goes out of scope here, unread and unwritten: that is
            // the close. The server drops a connection its acceptor refused.
            Err(listener) => {
                return Box::pin(std::future::ready(Err(io::Error::new(
                    io::ErrorKind::ConnectionRefused,
                    listener.limit_reached(),
                ))));
            }
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
#[path = "../../tests/unit/server/connection_limit/tests.rs"]
mod tests;
