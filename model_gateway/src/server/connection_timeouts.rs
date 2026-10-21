//! Bounds on connections that carry no request: the HTTP edge's header-read
//! timeout and idle keep-alive timeout.
//!
//! The server stack bounds request bodies (`--stream-body-stall-timeout-secs`)
//! and whole requests (`--request-timeout-secs`), but nothing between
//! requests: a connection that never sends a request head, one that stops in
//! the middle of a head, and a keep-alive connection that goes quiet after its
//! last response are all held until the client closes them. hyper's own
//! header-read timer is not wired by the server builder, and it would bound the
//! half-open and the idle case with one value. This module adds the two bounds
//! at the connection level, each with its own setting and counter:
//!
//! * header-read timeout: a connection that has not served a request yet must
//!   deliver a complete request head within this long of being accepted; on a
//!   keep-alive connection the same bound runs from the first byte of the next
//!   head (HTTP/1 only: an idle HTTP/2 connection exchanges control frames
//!   that are not the start of a request);
//! * idle timeout: a keep-alive connection with no request in flight is closed
//!   after this long without a new request.
//!
//! A connection with a request in flight is never closed here. "In flight"
//! runs from the moment the request reaches the service until its response
//! body has been written or dropped, so an open streaming response with a
//! silent producer, a slowly uploaded body and a slow reader all count as
//! active; the request-level timeouts own those. Expiry is signalled to hyper
//! as an end of stream on the read side, which closes the connection the way
//! a peer going away between requests does: an idle connection ends cleanly,
//! a half-received head ends as an incomplete message without a response.
//! The graceful drain is untouched: it disables keep-alive on every
//! connection, which closes idle ones at once and lets in-flight requests
//! finish. A connection that was upgraded (a WebSocket, after its `101`)
//! is a bidirectional stream from then on and is exempt from both bounds.

use std::{
    future::Future,
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    task::{ready, Context, Poll},
    time::Duration,
};

use axum_server::accept::Accept;
use futures::task::AtomicWaker;
use http::{Response, StatusCode};
use http_body::{Body, Frame, SizeHint};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    time::{Instant, Sleep},
};
use tower::Service;
use tracing::debug;

use crate::observability::metrics::Metrics;

/// The two connection-level bounds, each `None` when disabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConnectionTimeouts {
    /// A connection that has not served a request yet must deliver a complete
    /// request head within this long of being accepted; on a keep-alive
    /// connection the same bound runs from the first byte of the next head.
    pub header_read: Option<Duration>,
    /// A keep-alive connection with no request in flight is closed after this
    /// long without a new request.
    pub idle: Option<Duration>,
}

impl ConnectionTimeouts {
    /// Builds the bounds from the configured seconds; `0` disables a bound.
    pub fn from_secs(header_read_secs: u64, idle_secs: u64) -> Self {
        let bound = |secs: u64| (secs > 0).then(|| Duration::from_secs(secs));
        Self {
            header_read: bound(header_read_secs),
            idle: bound(idle_secs),
        }
    }

    /// No bound at all: connections live until the client closes them.
    pub fn disabled() -> Self {
        Self {
            header_read: None,
            idle: None,
        }
    }
}

/// The bound that closed a connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Timeout {
    HeaderRead,
    Idle,
}

impl Timeout {
    /// The `timeout` label of `smg_http_connection_timeouts_total`.
    fn label(self) -> &'static str {
        match self {
            Self::HeaderRead => "header_read",
            Self::Idle => "idle",
        }
    }
}

/// Connections closed by each bound, readable in process; the counter
/// `smg_http_connection_timeouts_total{timeout}` carries the same counts.
#[derive(Debug, Default)]
pub struct ConnectionTimeoutStats {
    header_read: AtomicU64,
    idle: AtomicU64,
}

impl ConnectionTimeoutStats {
    /// Connections closed because no complete request head arrived in time.
    pub fn closed_by_header_read_timeout(&self) -> u64 {
        self.header_read.load(Ordering::Relaxed)
    }

    /// Keep-alive connections closed for carrying no request for too long.
    pub fn closed_by_idle_timeout(&self) -> u64 {
        self.idle.load(Ordering::Relaxed)
    }

    fn record(&self, timeout: Timeout) {
        let counter = match timeout {
            Timeout::HeaderRead => &self.header_read,
            Timeout::Idle => &self.idle,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        Metrics::record_http_connection_timeout(timeout.label());
    }
}

/// What the service side tells the connection side: how many requests are in
/// flight (from the call until the response body ends or is dropped), how
/// many the connection has started at all, and whether it was upgraded.
#[derive(Debug, Default)]
struct RequestCount {
    in_flight: AtomicUsize,
    started: AtomicU64,
    upgraded: AtomicBool,
    /// The connection task's waker, registered while the bounds are
    /// suspended and woken when the last request in flight ends: the server
    /// polls the read side of an idle connection again only when bytes
    /// arrive, so without this wake the idle bound would never be armed.
    idle_waker: AtomicWaker,
}

impl RequestCount {
    /// Whether the bounds are suspended: a request is in flight or the
    /// connection has become a bidirectional stream.
    fn exempt(&self) -> bool {
        self.in_flight.load(Ordering::Acquire) > 0 || self.upgraded.load(Ordering::Acquire)
    }
}

/// Holds one in-flight request on its connection's count.
#[derive(Debug)]
struct InFlightGuard(Arc<RequestCount>);

impl InFlightGuard {
    fn start(count: Arc<RequestCount>) -> Self {
        count.in_flight.fetch_add(1, Ordering::AcqRel);
        count.started.fetch_add(1, Ordering::AcqRel);
        Self(count)
    }

    /// The response switched protocols: the connection is a stream from now
    /// on, outside both bounds for the rest of its life.
    fn mark_upgraded(&self) {
        self.0.upgraded.store(true, Ordering::Release);
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if self.0.in_flight.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.idle_waker.wake();
        }
    }
}

/// An acceptor that applies [`ConnectionTimeouts`] to every connection the
/// inner acceptor produces (the plain-HTTP and the TLS acceptor alike): the
/// stream is wrapped in a [`TimedConnection`], the connection's service in a
/// [`TrackedService`], and the two share the connection's request count.
#[derive(Clone, Debug)]
pub struct ConnectionTimeoutAcceptor<A> {
    inner: A,
    timeouts: ConnectionTimeouts,
    stats: Arc<ConnectionTimeoutStats>,
}

impl<A> ConnectionTimeoutAcceptor<A> {
    pub fn new(inner: A, timeouts: ConnectionTimeouts) -> Self {
        Self {
            inner,
            timeouts,
            stats: Arc::default(),
        }
    }

    /// The counters of connections closed by each bound.
    pub fn stats(&self) -> Arc<ConnectionTimeoutStats> {
        Arc::clone(&self.stats)
    }
}

impl<A, I, S> Accept<I, S> for ConnectionTimeoutAcceptor<A>
where
    A: Accept<I, S>,
    A::Future: Send + 'static,
    A::Stream: Send + 'static,
    A::Service: Send + 'static,
{
    type Stream = TimedConnection<A::Stream>;
    type Service = TrackedService<A::Service>;
    type Future = Pin<Box<dyn Future<Output = io::Result<(Self::Stream, Self::Service)>> + Send>>;

    fn accept(&self, stream: I, service: S) -> Self::Future {
        let inner = self.inner.accept(stream, service);
        let timeouts = self.timeouts;
        let stats = Arc::clone(&self.stats);
        Box::pin(async move {
            let (stream, service) = inner.await?;
            let count = Arc::new(RequestCount::default());
            Ok((
                TimedConnection::new(stream, Arc::clone(&count), timeouts, stats),
                TrackedService {
                    inner: service,
                    count,
                },
            ))
        })
    }
}

/// Which protocol the connection speaks, read off its first bytes the way
/// hyper's version detection does. Only HTTP/1 has a request head that can
/// stall half-way; the frames of an idle HTTP/2 connection (pings, window
/// updates, settings) are not the start of a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Protocol {
    Undecided { preface_matched: usize },
    Http1,
    Http2,
}

impl Protocol {
    fn label(self) -> &'static str {
        match self {
            Self::Undecided { .. } => "undecided",
            Self::Http1 => "http/1",
            Self::Http2 => "http/2",
        }
    }
}

const H2_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// A wait for the next request head.
#[derive(Clone, Copy, Debug)]
struct HeadWait {
    /// When the wait began: the accept for a connection that has not served a
    /// request yet, the end of the previous response otherwise.
    since: Instant,
    /// Whether the connection has served a request before (a keep-alive wait).
    keep_alive: bool,
    /// The connection's request count when the wait began; a request that
    /// started since ends this wait, whether or not the read side was polled
    /// while it was in flight.
    started: u64,
    /// When the first byte of the awaited head arrived, if it has.
    first_byte: Option<Instant>,
}

/// The connection's stream with the two bounds applied on its read side.
#[derive(Debug)]
pub struct TimedConnection<I> {
    inner: I,
    count: Arc<RequestCount>,
    timeouts: ConnectionTimeouts,
    stats: Arc<ConnectionTimeoutStats>,
    opened_at: Instant,
    protocol: Protocol,
    wait: Option<HeadWait>,
    /// The deadline the sleep is armed for.
    armed: Option<Instant>,
    sleep: Option<Pin<Box<Sleep>>>,
    closed_by: Option<Timeout>,
}

impl<I> TimedConnection<I> {
    fn new(
        inner: I,
        count: Arc<RequestCount>,
        timeouts: ConnectionTimeouts,
        stats: Arc<ConnectionTimeoutStats>,
    ) -> Self {
        Self {
            inner,
            count,
            timeouts,
            stats,
            opened_at: Instant::now(),
            protocol: Protocol::Undecided { preface_matched: 0 },
            wait: None,
            armed: None,
            sleep: None,
            closed_by: None,
        }
    }

    /// The wrapped stream.
    pub fn get_ref(&self) -> &I {
        &self.inner
    }

    /// Bytes arrived from the peer: settle the protocol while it is undecided
    /// and, while no request is in flight, mark the start of the next head.
    fn on_bytes(&mut self, bytes: &[u8], now: Instant) {
        if let Protocol::Undecided { preface_matched } = self.protocol {
            let wanted = &H2_PREFACE[preface_matched..];
            let compared = bytes.len().min(wanted.len());
            self.protocol = if bytes[..compared] != wanted[..compared] {
                Protocol::Http1
            } else if preface_matched + compared == H2_PREFACE.len() {
                Protocol::Http2
            } else {
                Protocol::Undecided {
                    preface_matched: preface_matched + compared,
                }
            };
        }
        if !self.count.exempt() {
            self.head_wait(now).first_byte.get_or_insert(now);
        }
    }

    /// The current wait for a request head, started now if none is running
    /// or if a request has started since the running one began (a short
    /// request can come and go between two polls of the read side).
    fn head_wait(&mut self, now: Instant) -> &mut HeadWait {
        let started = self.count.started.load(Ordering::Acquire);
        if self.wait.is_some_and(|wait| wait.started != started) {
            self.wait = None;
        }
        let keep_alive = started > 0;
        let opened_at = self.opened_at;
        self.wait.get_or_insert(HeadWait {
            since: if keep_alive { now } else { opened_at },
            keep_alive,
            started,
            first_byte: None,
        })
    }

    /// The bound that applies to a wait, and when it expires.
    fn deadline_for(&self, wait: &HeadWait) -> Option<(Instant, Timeout)> {
        let ConnectionTimeouts { header_read, idle } = self.timeouts;
        if !wait.keep_alive {
            // Nothing served yet: a complete request head is due within the
            // header-read bound of the accept, whatever the protocol.
            return header_read.map(|bound| (wait.since + bound, Timeout::HeaderRead));
        }
        let idle_at = idle.map(|bound| (wait.since + bound, Timeout::Idle));
        let head_at = match (self.protocol, header_read, wait.first_byte) {
            (Protocol::Http2, _, _) => None,
            (_, Some(bound), Some(first_byte)) => Some((first_byte + bound, Timeout::HeaderRead)),
            _ => None,
        };
        match (idle_at, head_at) {
            (Some(idle_at), Some(head_at)) => Some(if head_at.0 < idle_at.0 {
                head_at
            } else {
                idle_at
            }),
            (idle_at, head_at) => idle_at.or(head_at),
        }
    }

    /// Nothing to read and no request in flight: arm the applicable bound and
    /// report an end of stream once it has expired.
    fn poll_head_wait(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // A request in progress (its body may still be arriving, its response
        // still going out) or an upgraded connection: the request-level
        // timeouts own it. Register for the wake that ends the last request,
        // then look again so a request ending in between is not missed.
        self.count.idle_waker.register(cx.waker());
        if self.count.exempt() {
            self.wait = None;
            self.armed = None;
            return Poll::Pending;
        }
        let now = Instant::now();
        let wait = *self.head_wait(now);
        let Some((deadline, timeout)) = self.deadline_for(&wait) else {
            return Poll::Pending;
        };
        if now >= deadline {
            return self.expire(timeout, now);
        }
        if self.armed != Some(deadline) {
            self.armed = Some(deadline);
            match self.sleep.as_mut() {
                Some(sleep) => sleep.as_mut().reset(deadline),
                None => self.sleep = Some(Box::pin(tokio::time::sleep_until(deadline))),
            }
        }
        match self.sleep.as_mut() {
            Some(sleep) => match sleep.as_mut().poll(cx) {
                Poll::Ready(()) => self.expire(timeout, now),
                Poll::Pending => Poll::Pending,
            },
            None => Poll::Pending,
        }
    }

    fn expire(&mut self, timeout: Timeout, now: Instant) -> Poll<io::Result<()>> {
        self.closed_by = Some(timeout);
        self.stats.record(timeout);
        debug!(
            timeout = timeout.label(),
            protocol = self.protocol.label(),
            age_secs = now.duration_since(self.opened_at).as_secs_f64(),
            requests = self.count.started.load(Ordering::Acquire),
            "closing a connection with no request in progress"
        );
        // An end of stream: hyper closes the connection as it does when the
        // peer goes away between requests.
        Poll::Ready(Ok(()))
    }
}

impl<I: AsyncRead + Unpin> AsyncRead for TimedConnection<I> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if self.closed_by.is_some() {
            return Poll::Ready(Ok(()));
        }
        let before = buf.filled().len();
        match Pin::new(&mut self.inner).poll_read(cx, buf) {
            Poll::Ready(Ok(())) => {
                if buf.filled().len() > before {
                    let now = Instant::now();
                    let (bytes, this) = (&buf.filled()[before..], self.as_mut().get_mut());
                    this.on_bytes(bytes, now);
                }
                Poll::Ready(Ok(()))
            }
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Pending => self.poll_head_wait(cx),
        }
    }
}

impl<I: AsyncWrite + Unpin> AsyncWrite for TimedConnection<I> {
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

/// The connection's service, counting each request from its call until its
/// response body ends or is dropped.
#[derive(Clone, Debug)]
pub struct TrackedService<S> {
    inner: S,
    count: Arc<RequestCount>,
}

impl<S> TrackedService<S> {
    /// The wrapped service.
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S, R, B> Service<R> for TrackedService<S>
where
    S: Service<R, Response = Response<B>>,
    B: Body,
{
    type Response = Response<TrackedBody<B>>;
    type Error = S::Error;
    type Future = TrackedFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: R) -> Self::Future {
        let guard = InFlightGuard::start(Arc::clone(&self.count));
        TrackedFuture {
            inner: Box::pin(self.inner.call(request)),
            guard: Some(guard),
        }
    }
}

/// The service's response future; the request stays in flight across it and
/// then for the life of the response body.
#[derive(Debug)]
pub struct TrackedFuture<F> {
    inner: Pin<Box<F>>,
    guard: Option<InFlightGuard>,
}

impl<F, B, E> Future for TrackedFuture<F>
where
    F: Future<Output = Result<Response<B>, E>>,
    B: Body,
{
    type Output = Result<Response<TrackedBody<B>>, E>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let result = ready!(self.inner.as_mut().poll(cx));
        let guard = self.guard.take();
        Poll::Ready(result.map(|response| {
            if response.status() == StatusCode::SWITCHING_PROTOCOLS {
                if let Some(guard) = &guard {
                    guard.mark_upgraded();
                }
            }
            response.map(|body| TrackedBody::new(body, guard))
        }))
    }
}

/// A response body that keeps its request in flight until the body ends or
/// is dropped; a streaming response with a silent producer is in flight.
#[derive(Debug)]
pub struct TrackedBody<B> {
    inner: B,
    guard: Option<InFlightGuard>,
}

impl<B: Body> TrackedBody<B> {
    fn new(inner: B, guard: Option<InFlightGuard>) -> Self {
        let mut body = Self { inner, guard };
        // An already-complete body (an empty response) may never be polled.
        if body.inner.is_end_stream() {
            body.guard = None;
        }
        body
    }
}

impl<B: Body + Unpin> Body for TrackedBody<B> {
    type Data = B::Data;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let poll = Pin::new(&mut this.inner).poll_frame(cx);
        // Release as soon as the end of the stream is knowable; the drop
        // covers a client that went away.
        match &poll {
            Poll::Ready(None) | Poll::Ready(Some(Err(_))) => this.guard = None,
            Poll::Ready(Some(Ok(_))) if this.inner.is_end_stream() => this.guard = None,
            _ => {}
        }
        poll
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, net::SocketAddr};

    use axum::{
        extract::ws::WebSocketUpgrade,
        response::sse::{Event, Sse},
        routing::get,
        Router,
    };
    use axum_server::Handle;
    use futures::StreamExt;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpStream,
        time::{sleep, timeout},
    };

    use super::*;
    use crate::server::bind_http_server;

    /// The bound under test, in seconds; the other one sits out of reach.
    const SHORT: u64 = 1;
    const LONG: u64 = 30;
    /// How long the streaming route's producer stays silent: longer than
    /// `SHORT`, so a silent stream crosses the bound.
    const PRODUCER_SILENCE: Duration = Duration::from_secs(3);
    /// How long a test waits for a close before calling the connection open.
    const WINDOW: Duration = Duration::from_secs(5);
    /// A close must not come measurably before the bound.
    const NOT_BEFORE: Duration = Duration::from_millis(900);

    fn test_app() -> Router {
        Router::new()
            .route("/ok", get(|| async { "ok" }))
            .route(
                "/stream",
                get(|| async {
                    let first = futures::stream::once(async {
                        Ok::<_, Infallible>(Event::default().data("first"))
                    });
                    let second = futures::stream::once(async {
                        sleep(PRODUCER_SILENCE).await;
                        Ok::<_, Infallible>(Event::default().data("second"))
                    });
                    Sse::new(first.chain(second))
                }),
            )
            .route(
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

    /// Serves the test app through the production acceptor on a loopback
    /// port, runs `check` against it and returns the acceptor's counters.
    async fn with_server<F, Fut>(
        timeouts: ConnectionTimeouts,
        check: F,
    ) -> Arc<ConnectionTimeoutStats>
    where
        F: FnOnce(SocketAddr, Handle<SocketAddr>) -> Fut,
        Fut: Future<Output = ()>,
    {
        let handle = Handle::new();
        let server =
            bind_http_server("127.0.0.1:0".parse().unwrap(), timeouts).handle(handle.clone());
        let stats = server.get_ref().stats();
        let serving = server.serve(test_app().into_make_service_with_connect_info::<SocketAddr>());
        let checking = async {
            let addr = handle
                .listening()
                .await
                .expect("the test server did not bind");
            check(addr, handle.clone()).await;
            handle.shutdown();
        };
        let (served, ()) = tokio::join!(serving, checking);
        served.expect("the test server failed");
        stats
    }

    async fn connect(addr: SocketAddr) -> TcpStream {
        TcpStream::connect(addr).await.expect("connect")
    }

    /// Waits up to `within` for the server to close the connection: the time
    /// it took, or `None` while the connection is still open at the end of
    /// the window. Bytes still arriving are read and discarded.
    async fn closed_within(stream: &mut TcpStream, within: Duration) -> Option<Duration> {
        let started = Instant::now();
        let mut buf = [0u8; 4096];
        loop {
            let left = within.checked_sub(started.elapsed())?;
            match timeout(left, stream.read(&mut buf)).await {
                Ok(Ok(0)) | Ok(Err(_)) => return Some(started.elapsed()),
                Ok(Ok(_)) => {}
                Err(_) => return None,
            }
        }
    }

    fn closed(at: Option<Duration>) -> Duration {
        let at = at.expect("the connection was still open at the end of the window");
        assert!(at >= NOT_BEFORE, "closed before the bound: {at:?}");
        at
    }

    /// Sends a keep-alive GET and reads the whole response.
    async fn get_ok(stream: &mut TcpStream) {
        stream
            .write_all(b"GET /ok HTTP/1.1\r\nHost: test\r\n\r\n")
            .await
            .unwrap();
        let response = read_response(stream).await;
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    }

    /// Reads a response head and its `Content-Length` body.
    async fn read_response(stream: &mut TcpStream) -> String {
        let mut bytes = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = timeout(WINDOW, stream.read(&mut buf))
                .await
                .expect("the response did not arrive")
                .expect("read");
            assert!(n > 0, "the connection closed inside the response");
            bytes.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&bytes);
            if let Some(head_end) = text.find("\r\n\r\n") {
                let content_length: usize = text[..head_end]
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                if bytes.len() >= head_end + 4 + content_length {
                    return text.into_owned();
                }
            }
        }
    }

    /// Reads until `marker` has arrived: the time it took.
    async fn read_until(stream: &mut TcpStream, marker: &str) -> Duration {
        let started = Instant::now();
        let mut bytes = Vec::new();
        let mut buf = [0u8; 4096];
        while !String::from_utf8_lossy(&bytes).contains(marker) {
            let n = timeout(Duration::from_secs(10), stream.read(&mut buf))
                .await
                .expect("the stream stalled")
                .expect("read");
            assert!(n > 0, "the connection closed before {marker:?} arrived");
            bytes.extend_from_slice(&buf[..n]);
        }
        started.elapsed()
    }

    fn counts(stats: &ConnectionTimeoutStats) -> (u64, u64) {
        (
            stats.closed_by_header_read_timeout(),
            stats.closed_by_idle_timeout(),
        )
    }

    #[tokio::test]
    async fn a_connection_that_sends_nothing_is_closed_at_the_header_read_timeout() {
        let stats = with_server(
            ConnectionTimeouts::from_secs(SHORT, LONG),
            |addr, _| async move {
                let mut silent = connect(addr).await;
                closed(closed_within(&mut silent, WINDOW).await);
            },
        )
        .await;
        assert_eq!(counts(&stats), (1, 0));
    }

    #[tokio::test]
    async fn a_connection_that_stops_inside_its_request_head_is_closed_at_the_header_read_timeout()
    {
        let stats = with_server(
            ConnectionTimeouts::from_secs(SHORT, LONG),
            |addr, _| async move {
                let mut partial = connect(addr).await;
                partial
                    .write_all(b"GET /ok HTTP/1.1\r\nHost: test\r\nX-Half: a")
                    .await
                    .unwrap();
                closed(closed_within(&mut partial, WINDOW).await);
            },
        )
        .await;
        assert_eq!(counts(&stats), (1, 0));
    }

    #[tokio::test]
    async fn an_idle_keep_alive_connection_is_closed_at_the_idle_timeout() {
        let stats = with_server(
            ConnectionTimeouts::from_secs(LONG, SHORT),
            |addr, _| async move {
                let mut keep_alive = connect(addr).await;
                get_ok(&mut keep_alive).await;
                closed(closed_within(&mut keep_alive, WINDOW).await);
            },
        )
        .await;
        assert_eq!(counts(&stats), (0, 1));
    }

    #[tokio::test]
    async fn a_keep_alive_connection_that_stops_inside_its_next_head_is_closed_at_the_header_read_timeout(
    ) {
        let stats = with_server(
            ConnectionTimeouts::from_secs(SHORT, LONG),
            |addr, _| async move {
                let mut keep_alive = connect(addr).await;
                get_ok(&mut keep_alive).await;
                // The idle bound is out of reach; the partial head arms the
                // header-read bound from its first byte.
                keep_alive
                    .write_all(b"GET /ok HTTP/1.1\r\nHost: te")
                    .await
                    .unwrap();
                closed(closed_within(&mut keep_alive, WINDOW).await);
            },
        )
        .await;
        assert_eq!(counts(&stats), (1, 0));
    }

    #[tokio::test]
    async fn a_streaming_response_with_a_silent_producer_outlives_both_timeouts() {
        let stats = with_server(
            ConnectionTimeouts::from_secs(SHORT, SHORT),
            |addr, _| async move {
                let mut streaming = connect(addr).await;
                streaming
                    .write_all(b"GET /stream HTTP/1.1\r\nHost: test\r\n\r\n")
                    .await
                    .unwrap();
                read_until(&mut streaming, "data: first").await;
                // The producer is silent for longer than either bound: the
                // stream is in flight, never idle, and its next event arrives.
                let silence = read_until(&mut streaming, "data: second").await;
                assert!(silence + NOT_BEFORE >= PRODUCER_SILENCE, "{silence:?}");
                // Once the stream has ended the connection is a keep-alive one
                // between requests again, and the idle bound closes it.
                closed(closed_within(&mut streaming, WINDOW).await);
            },
        )
        .await;
        assert_eq!(counts(&stats), (0, 1));
    }

    #[tokio::test]
    async fn an_upgraded_connection_is_exempt_from_both_timeouts() {
        let stats = with_server(
            ConnectionTimeouts::from_secs(SHORT, SHORT),
            |addr, _| async move {
                let mut websocket = connect(addr).await;
                websocket
                    .write_all(
                        b"GET /ws HTTP/1.1\r\nHost: test\r\nConnection: Upgrade\r\n\
                          Upgrade: websocket\r\nSec-WebSocket-Version: 13\r\n\
                          Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
                    )
                    .await
                    .unwrap();
                let response = read_response(&mut websocket).await;
                assert!(response.starts_with("HTTP/1.1 101"), "{response}");
                // Silent for longer than both bounds, then a frame: the
                // connection is still there and still echoes.
                let still_open = closed_within(&mut websocket, Duration::from_millis(2500)).await;
                assert_eq!(still_open, None, "the upgraded connection was closed");
                let mask = [1u8, 2, 3, 4];
                let frame = [0x81, 0x82, 1, 2, 3, 4, b'h' ^ mask[0], b'i' ^ mask[1]];
                websocket.write_all(&frame).await.unwrap();
                let mut echo = [0u8; 4];
                timeout(WINDOW, websocket.read_exact(&mut echo))
                    .await
                    .expect("no echo")
                    .unwrap();
                assert_eq!(echo, [0x81, 0x02, b'h', b'i']);
            },
        )
        .await;
        assert_eq!(counts(&stats), (0, 0));
    }

    #[tokio::test]
    async fn disabled_timeouts_leave_connections_open() {
        let stats = with_server(ConnectionTimeouts::from_secs(0, 0), |addr, _| async move {
            let mut silent = connect(addr).await;
            let mut keep_alive = connect(addr).await;
            get_ok(&mut keep_alive).await;
            let window = Duration::from_secs(2);
            let (silent_closed, idle_closed) = tokio::join!(
                closed_within(&mut silent, window),
                closed_within(&mut keep_alive, window)
            );
            assert_eq!((silent_closed, idle_closed), (None, None));
            get_ok(&mut keep_alive).await;
        })
        .await;
        assert_eq!(counts(&stats), (0, 0));
    }

    #[tokio::test]
    async fn the_drain_still_closes_idle_connections_and_lets_streams_finish() {
        let stats = with_server(
            ConnectionTimeouts::from_secs(LONG, LONG),
            |addr, handle| async move {
                let mut keep_alive = connect(addr).await;
                get_ok(&mut keep_alive).await;
                let mut streaming = connect(addr).await;
                streaming
                    .write_all(b"GET /stream HTTP/1.1\r\nHost: test\r\n\r\n")
                    .await
                    .unwrap();
                read_until(&mut streaming, "data: first").await;
                handle.graceful_shutdown(Some(Duration::from_secs(10)));
                // The idle connection goes at once; the stream delivers its
                // last event first and is closed at its end.
                assert!(
                    closed_within(&mut keep_alive, WINDOW).await.is_some(),
                    "the idle connection survived the drain"
                );
                read_until(&mut streaming, "data: second").await;
                assert!(
                    closed_within(&mut streaming, WINDOW).await.is_some(),
                    "the streaming connection outlived its stream"
                );
            },
        )
        .await;
        assert_eq!(counts(&stats), (0, 0));
    }

    fn idle_connection(timeouts: ConnectionTimeouts) -> TimedConnection<tokio::io::Empty> {
        TimedConnection::new(
            tokio::io::empty(),
            Arc::new(RequestCount::default()),
            timeouts,
            Arc::new(ConnectionTimeoutStats::default()),
        )
    }

    #[test]
    fn the_http2_preface_is_recognised_across_reads_and_anything_else_is_http1() {
        let now = Instant::now();
        let mut h2 = idle_connection(ConnectionTimeouts::disabled());
        h2.on_bytes(&H2_PREFACE[..10], now);
        assert_eq!(
            h2.protocol,
            Protocol::Undecided {
                preface_matched: 10
            }
        );
        h2.on_bytes(&H2_PREFACE[10..], now);
        assert_eq!(h2.protocol, Protocol::Http2);

        let mut h1 = idle_connection(ConnectionTimeouts::disabled());
        h1.on_bytes(b"PRI * HTTP/1.1\r\n", now);
        assert_eq!(h1.protocol, Protocol::Http1);
    }

    #[test]
    fn a_partial_head_arms_the_header_read_bound_on_http1_only() {
        let timeouts = ConnectionTimeouts::from_secs(1, 90);
        let connection = |protocol| {
            let mut connection = idle_connection(timeouts);
            connection.protocol = protocol;
            connection
        };
        let now = Instant::now();
        let header_read = Some((now + Duration::from_secs(1), Timeout::HeaderRead));
        let idle = Some((now + Duration::from_secs(90), Timeout::Idle));

        let fresh = HeadWait {
            since: now,
            keep_alive: false,
            started: 0,
            first_byte: None,
        };
        assert_eq!(
            connection(Protocol::Http1).deadline_for(&fresh),
            header_read
        );
        assert_eq!(
            connection(Protocol::Http2).deadline_for(&fresh),
            header_read
        );

        let between_requests = HeadWait {
            since: now,
            keep_alive: true,
            started: 1,
            first_byte: None,
        };
        assert_eq!(
            connection(Protocol::Http1).deadline_for(&between_requests),
            idle
        );
        assert_eq!(
            connection(Protocol::Http2).deadline_for(&between_requests),
            idle
        );

        let partial_next_head = HeadWait {
            since: now,
            keep_alive: true,
            started: 1,
            first_byte: Some(now),
        };
        assert_eq!(
            connection(Protocol::Http1).deadline_for(&partial_next_head),
            header_read
        );
        assert_eq!(
            connection(Protocol::Http2).deadline_for(&partial_next_head),
            idle
        );
    }

    #[test]
    fn zero_seconds_disables_a_bound() {
        assert_eq!(
            ConnectionTimeouts::from_secs(0, 0),
            ConnectionTimeouts::disabled()
        );
        assert_eq!(
            ConnectionTimeouts::from_secs(30, 0),
            ConnectionTimeouts {
                header_read: Some(Duration::from_secs(30)),
                idle: None,
            }
        );
    }
}
