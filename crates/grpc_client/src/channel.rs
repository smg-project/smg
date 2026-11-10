//! The engine clients' HTTP/2 channel: one connection per engine, every call
//! run over h2 from the task that makes it.
//!
//! Each gRPC client of this crate (`vllm_engine`, `sglang_scheduler`,
//! `tokenspeed_scheduler`, `trtllm_service`, `mlx_engine`, and the EPD
//! encode client `tokenspeed_encoder`) connects to its backend the same
//! way: accept either an `http://` or a `grpc://` endpoint, convert the
//! gRPC scheme to its HTTP equivalent, dial under a connect timeout, and
//! run every call over a [`Channel`] with the same keep-alive and window
//! profile. This module centralises that pipeline so adding a new engine,
//! or tuning the transport profile, touches one place instead of one per
//! client.
//!
//! The channel is this crate's own `tower::Service` over h2's client
//! handle rather than tonic's `Channel`. tonic's fronts hyper's HTTP/2
//! client with a `tower::buffer::Buffer` (one worker task per connection,
//! an mpsc send and a oneshot per call), and hyper relays every request
//! through its dispatch task and a task spawned per request for the
//! response: five tasks and seven hand-offs per call, paid in instructions
//! and allocations on every engine call (not in context switches: the
//! hand-offs run in the runtime worker's LIFO slot). Here a call opens its
//! stream on the connection from the calling task, writes its body under
//! the stream's send window and awaits the response head; the connection's
//! one task (the socket, with the keep-alive pings, in `connection`)
//! wakes the call for the head and for every frame of the response body.
//! No task is spawned per call and nothing relays the answer, as the h2
//! crate is built to be used.

mod connection;
#[cfg(test)]
mod tests;

use std::{
    fmt,
    future::{poll_fn, Future},
    pin::{pin, Pin},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, PoisonError,
    },
    task::{ready, Context, Poll, Waker},
    time::Duration,
};

use h2::{client, Reason, RecvStream, SendStream};
use http_body::Frame;
use tonic::{
    codegen::{
        http::{
            header::{HeaderValue, USER_AGENT},
            uri::{Authority, Uri},
            Request, Response, Version,
        },
        Body as _, Bytes, Service,
    },
    ConnectError,
};
use tracing::{debug, info};

use self::connection::{Connection, End};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Default ceiling on a single TCP connect attempt.
///
/// Without a bound of its own, a dial to a black-holing peer (SYN dropped
/// rather than refused, a pod IP whose container is not listening yet) sits
/// in `SYN_SENT` until the kernel gives up: `net.ipv4.tcp_syn_retries`
/// defaults to 6, i.e. ~127s. Bounding the attempt here keeps those sockets
/// from accumulating faster than they drain. Matches the upstream HTTP
/// client's connect timeout in `AppContext`.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// The `user-agent` of every call, in place of the one tonic's channel
/// wrote: engines log it with the request.
const USER_AGENT_VALUE: &str = concat!("smg-grpc-client/", env!("CARGO_PKG_VERSION"));

/// One HTTP/2 connection to an engine, shared by every clone of the channel.
///
/// A call opens its stream on the connection from the task that makes it,
/// writes the request body under the stream's send window (a one-message
/// body, which every call of this crate has, is one DATA frame that ends the
/// stream) and awaits the response head; the connection's task wakes it for
/// the head and the body hands tonic every frame as the socket task reads
/// it. No task is spawned per call and nothing relays the answer.
///
/// A connection that died (the peer went away, a GOAWAY, a keep-alive ping
/// unanswered) fails the calls in flight on it and is re-dialled under the
/// next call: one dial at a time, and the callers that find the connection
/// dead meanwhile wait for that dial and share its connection, or its
/// failure, so N callers on a dead engine cost one dial, not N, and one
/// connect timeout, not N, when the engine black-holes. The gateway caches
/// a worker's client once, so this is the client's only recovery.
#[derive(Clone)]
pub struct Channel {
    shared: Arc<Shared>,
}

/// The response's body: h2's receive stream as an HTTP body, each data
/// frame's bytes released back to the stream's window as it is read, the
/// trailers last. A request still being written when the response head
/// arrived (a peer that answers before it has read the whole request and
/// keeps reading) is written from here, as the body is read: no task for
/// that either.
#[derive(Debug)]
pub struct RecvBody {
    stream: RecvStream,
    write: Option<BodyWrite>,
    end: Arc<End>,
}

impl Service<Request<tonic::body::Body>> for Channel {
    type Response = Response<RecvBody>;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    /// Always ready: whether the connection takes a stream is asked under
    /// the call, where a dead connection is re-dialled, because tonic's
    /// generated clients report a readiness error as an `Unknown` status
    /// without its source, which would lose the fault's nature.
    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: Request<tonic::body::Body>) -> Self::Future {
        let (mut parts, body) = request.into_parts();
        // The origin in front of the call's own path (what tonic's AddOrigin
        // layer wrote).
        let mut uri = self.shared.origin.clone().into_parts();
        uri.path_and_query = parts.uri.path_and_query().cloned();
        parts.uri = match Uri::from_parts(uri) {
            Ok(uri) => uri,
            Err(e) => return Box::pin(std::future::ready(Err(e.into()))),
        };
        parts.version = Version::HTTP_2;
        parts
            .headers
            .entry(USER_AGENT)
            .or_insert(HeaderValue::from_static(USER_AGENT_VALUE));
        let request = Request::from_parts(parts, ());
        let ended = body.is_end_stream();
        let shared = Arc::clone(&self.shared);
        Box::pin(async move {
            let mut connection = shared.ready().await?;
            let (response, stream) = connection
                .requests
                .send_request(request, ended)
                .map_err(|e| connection.end.failure(e))?;
            exchange(response, stream, body, connection.end, ended).await
        })
    }
}

/// Connect a [`Channel`] to the given endpoint with the SMG-standard
/// keep-alive and HTTP/2 window profile applied, using
/// [`DEFAULT_CONNECT_TIMEOUT`].
///
/// The endpoint may use `http://` or `grpc://`; the gRPC scheme is
/// normalised to its HTTP equivalent before the URI is parsed. A userinfo in
/// the endpoint is not forwarded: every call's `:authority` is the host and
/// port. TLS endpoints are refused (see [`normalize_grpc_endpoint`] for the
/// scheme rewrite and the config validator for the refusal at start-up).
pub async fn connect_channel(
    endpoint: &str,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    connect_channel_with_timeout(endpoint, DEFAULT_CONNECT_TIMEOUT).await
}

/// Same as [`connect_channel`], but with an explicit connect timeout.
///
/// Callers that already bound the dial with their own deadline (health and
/// reachability probes) should pass that deadline through, so the socket of
/// a dial that runs past it is closed with the attempt rather than left to
/// the outer future being dropped.
pub async fn connect_channel_with_timeout(
    endpoint: &str,
    connect_timeout: Duration,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    let channel = Channel::unconnected(endpoint, connect_timeout)?;
    channel.shared.dial().await?;
    Ok(channel)
}

/// Whether `status` is what tonic makes of a failure in the HTTP/2 layer
/// under a call, rather than an answer the worker gave: the connection went
/// away (a GOAWAY of either side, the one this client issues once too many
/// of its locally-reset streams received late frames included; the socket
/// failing or closed with the call in flight), the peer reset the stream
/// below gRPC, a protocol fault. tonic picks such a status's code from the
/// frame's reason (`ENHANCE_YOUR_CALM` reads as `RESOURCE_EXHAUSTED`, the
/// code of a worker's own capacity pushback; `PROTOCOL_ERROR` as `INTERNAL`,
/// the code of its own failure on a request; an I/O failure, which h2
/// reports with no reason, as `UNKNOWN`), so the code cannot tell the two
/// apart: the `h2::Error` in the status's source chain can. A status the
/// worker answered with (its trailers) carries no such source. Neither does
/// a dial the peer refuses or that times out (the channel's `UNAVAILABLE`
/// over the connect error, before any call was in flight), nor the
/// `UNAVAILABLE` "keep-alive timed out" of a connection this side dropped
/// for a pong that never came, as tonic answered it.
///
/// The gateway's liveness asks a different question of the ends of its
/// polls and streams, whether the peer is reachable for work, and answers
/// it from the code alone (`worker::liveness::is_connection_code` in the
/// gateway crate); the two are not interchangeable.
pub fn is_transport_failure(status: &tonic::Status) -> bool {
    let mut source = std::error::Error::source(status);
    while let Some(error) = source {
        if error.is::<h2::Error>() {
            return true;
        }
        source = error.source();
    }
    false
}

/// Convert a `grpc://` or `grpcs://` endpoint to a tonic-compatible
/// `http://` or `https://` URI. Other schemes (or schemeless inputs) are
/// returned unchanged so callers can mix `http(s)://` and `grpc(s)://`
/// freely.
pub fn normalize_grpc_endpoint(endpoint: &str) -> String {
    match endpoint.split_once("://") {
        Some(("grpc", rest)) => format!("http://{rest}"),
        Some(("grpcs", rest)) => format!("https://{rest}"),
        _ => endpoint.to_string(),
    }
}

// --- the channel's own machinery --------------------------------------------

struct Shared {
    /// The `http://` origin: the scheme and authority of every call's URI.
    origin: Uri,
    connect_timeout: Duration,
    /// The connection the clones share: the latest dial's, `None` before
    /// the first.
    live: Mutex<Option<Connection>>,
    /// Held for the length of a dial, so a dead connection is re-dialled
    /// once however many callers find it dead; the guard carries the text
    /// of the failure the last dial ended in (`None` after a connection),
    /// for the callers that queued behind that dial.
    dial: tokio::sync::Mutex<Option<String>>,
    /// Dials finished so far: a caller reads it before queuing for the
    /// lock and knows, once it holds the lock, whether a dial ended while
    /// it waited.
    dials: AtomicU64,
}

impl Channel {
    /// A channel to `endpoint` that dials on its first call.
    pub(crate) fn unconnected(endpoint: &str, connect_timeout: Duration) -> Result<Self, BoxError> {
        let origin = origin_of(endpoint)?;
        Ok(Self {
            shared: Arc::new(Shared {
                origin,
                connect_timeout,
                live: Mutex::new(None),
                dial: tokio::sync::Mutex::new(None),
                dials: AtomicU64::new(0),
            }),
        })
    }

    /// The peer as the diagnostics name it: host and port.
    pub(crate) fn peer(&self) -> String {
        peer_of(&self.shared.origin)
    }
}

/// `endpoint` as the origin of a channel's calls: a plain `http://` URI with
/// a host, the gRPC scheme normalised first, its authority the host and port
/// alone.
///
/// A userinfo written into a worker URL (`user:secret@host:port`) stays out
/// of the origin: `http` keeps it in the URI's authority and h2 writes that
/// authority whole into every request's `:authority`, a pseudo-header that
/// names the host and port and nothing else, so it is cut here, once, as the
/// origin is built. The host and port stay as the endpoint wrote them (an
/// IPv6 literal's brackets, a port left to the default).
///
/// `https://` (`grpcs://`) is refused: the channel speaks plain HTTP/2 over
/// TCP. tonic's channel carried its TLS code into the gateway binary (a
/// feature another crate of the workspace enables) but no root store to
/// validate an engine's certificate against, so no TLS endpoint ever
/// completed a validated handshake; the config validator refuses the scheme
/// at start-up now, and a TLS connector, when one is wanted, slots in under
/// `connection::connect_io`, which takes any socket.
fn origin_of(endpoint: &str) -> Result<Uri, BoxError> {
    let origin: Uri = normalize_grpc_endpoint(endpoint).parse()?;
    if origin.scheme_str() != Some("http") {
        return Err(format!(
            "{}: only plain http (grpc) endpoints are supported",
            peer_of(&origin)
        )
        .into());
    }
    // The authority as the calls write it: what follows the userinfo, when
    // the endpoint carries one.
    let host_port = match origin.authority().map(|a| a.as_str().rsplit_once('@')) {
        None => return Err("the endpoint names no host".into()),
        Some(None) => return Ok(origin),
        Some(Some((_userinfo, host_port))) => host_port.parse::<Authority>()?,
    };
    let mut parts = origin.into_parts();
    parts.authority = Some(host_port);
    Ok(Uri::from_parts(parts)?)
}

/// What the diagnostics call `origin`'s peer: host and port, never the
/// origin whole (a worker URL may carry userinfo, which no log line or
/// error gets).
fn peer_of(origin: &Uri) -> String {
    format!(
        "{}:{}",
        origin.host().unwrap_or_default(),
        origin.port_u16().unwrap_or(80)
    )
}

impl Shared {
    /// The connection the clones share, if there is one.
    fn live(&self) -> Option<Connection> {
        self.live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The shared connection, ready to take a stream: the live one once it
    /// has room for a stream (h2 queues the call behind the peer's
    /// concurrency limit here, no error), else the one a dial under the
    /// call gives.
    async fn ready(&self) -> Result<Connection, BoxError> {
        if let Some(mut live) = self.live() {
            if poll_fn(|cx| live.requests.poll_ready(cx)).await.is_ok() {
                return Ok(live);
            }
        }
        let mut connection = self.connected().await?;
        poll_fn(|cx| connection.requests.poll_ready(cx))
            .await
            .map_err(|e| connection.end.failure(e))?;
        Ok(connection)
    }

    /// A connection for a call that found none alive: the one another caller
    /// dialled meanwhile, else this caller's own dial, under the dial lock so
    /// a dead engine is dialled once however many callers queue. A dial that
    /// fails is the call's `UNAVAILABLE`, as tonic reads its connect errors.
    async fn connected(&self) -> Result<Connection, BoxError> {
        let dials_before = self.dials.load(Ordering::Acquire);
        let mut last_dial = self.dial.lock().await;
        if let Some(mut live) = self.live() {
            // Alive unless h2 says the connection is gone: a connection at
            // the peer's concurrency limit is alive and queues the call.
            let mut probe = Context::from_waker(Waker::noop());
            if !matches!(live.requests.poll_ready(&mut probe), Poll::Ready(Err(_))) {
                return Ok(live);
            }
        }
        // A dial that ended in a failure while this caller waited for the
        // lock is this caller's answer too: the callers behind a dial that
        // times out pay one connect timeout between them, not one each.
        let dialled_meanwhile = self.dials.load(Ordering::Acquire) != dials_before;
        if let (true, Some(failure)) = (dialled_meanwhile, last_dial.as_deref()) {
            return Err(ConnectError(std::io::Error::other(failure.to_owned()).into()).into());
        }
        let dialled = self.dial().await;
        *last_dial = dialled.as_ref().err().map(ToString::to_string);
        self.dials.fetch_add(1, Ordering::Release);
        Ok(dialled.map_err(ConnectError)?)
    }

    /// Dials the origin and publishes the connection to the clones.
    async fn dial(&self) -> Result<Connection, BoxError> {
        let connection = connection::connect(&self.origin, self.connect_timeout).await?;
        let previous = self
            .live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .replace(connection.clone());
        if previous.is_some() {
            info!(
                "re-dialled {} after its connection ended",
                peer_of(&self.origin)
            );
        }
        Ok(connection)
    }
}

/// The call once its stream is open: the request body written, the response
/// head awaited, the response body handed over as it arrives. The write and
/// the head are polled together, so a peer that answers before it has read
/// the whole request is heard: an answer that ends the stream, or carries
/// the gRPC status, is the call's whole answer and the rest of the body
/// stays unwritten; any other early head hands the write to the response
/// body, which finishes it as it is read.
async fn exchange(
    response: client::ResponseFuture,
    stream: SendStream<Bytes>,
    body: tonic::body::Body,
    end: Arc<End>,
    ended: bool,
) -> Result<Response<RecvBody>, BoxError> {
    let mut response = pin!(response);
    let mut write = (!ended).then(|| BodyWrite::new(stream, body));
    let head = poll_fn(|cx| {
        if let Some(pending) = &mut write {
            let written = Pin::new(pending).poll(cx);
            match written {
                Poll::Ready(Ok(()) | Err(Unwritten::Stream)) => write = None,
                Poll::Ready(Err(Unwritten::Body(e))) => return Poll::Ready(Err(e)),
                Poll::Pending => {}
            }
        }
        response.as_mut().poll(cx).map(Ok)
    })
    .await?
    .map_err(|e| end.failure(e))?;
    let (parts, stream) = head.into_parts();
    if stream.is_end_stream() || parts.headers.contains_key("grpc-status") {
        write = None;
    }
    Ok(Response::from_parts(parts, RecvBody { stream, write, end }))
}

/// How a request body's write ended short of the stream's end.
enum Unwritten {
    /// The body itself failed (its encoder refused a message): the stream is
    /// reset and the body's error is the call's.
    Body(BoxError),
    /// The stream ended under the write (the peer reset it, the connection
    /// went away): the response says how.
    Stream,
}

fn stream_ended(error: impl fmt::Display) -> Unwritten {
    debug!("the request body was not written in full: {error}");
    Unwritten::Stream
}

/// The request body's write to its stream, under the stream's send window:
/// one DATA frame per message, the one the body ends with flagged as the
/// stream's end (so a one-message body is one frame), the trailers last when
/// the body has them. A future the call polls with the response head, and
/// the response body after it when the head came first; it holds no
/// reference, so it moves between the two.
#[derive(Debug)]
struct BodyWrite {
    stream: SendStream<Bytes>,
    body: tonic::body::Body,
    /// The frame read ahead of the one being written, to learn whether that
    /// one is the body's last (`Some(None)`: it is).
    next: Option<Option<Result<Frame<Bytes>, tonic::Status>>>,
    /// The data waiting for the window, and whether it ends the stream.
    pending: Option<(Bytes, bool)>,
}

impl BodyWrite {
    fn new(stream: SendStream<Bytes>, body: tonic::body::Body) -> Self {
        Self {
            stream,
            body,
            next: None,
            pending: None,
        }
    }

    /// The next frame of the body, read ahead or awaited.
    fn poll_next_frame(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, tonic::Status>>> {
        match self.next.take() {
            Some(frame) => Poll::Ready(frame),
            None => Pin::new(&mut self.body).poll_frame(cx),
        }
    }
}

impl Future for BodyWrite {
    type Output = Result<(), Unwritten>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = &mut *self;
        loop {
            if let Some((data, end)) = &mut this.pending {
                if data.is_empty() {
                    this.stream
                        .send_data(Bytes::new(), *end)
                        .map_err(stream_ended)?;
                } else {
                    this.stream.reserve_capacity(data.len());
                    let granted = match ready!(this.stream.poll_capacity(cx)) {
                        Some(granted) => granted.map_err(stream_ended)?,
                        // The send side closed under the write: the peer
                        // reset the stream, or the connection failed. The
                        // response says how.
                        None => {
                            return Poll::Ready(Err(match ready!(this.stream.poll_reset(cx)) {
                                Ok(reason) => stream_ended(h2::Error::from(reason)),
                                Err(e) => stream_ended(e),
                            }));
                        }
                    };
                    let piece = data.split_to(granted.min(data.len()));
                    let finished = data.is_empty();
                    this.stream
                        .send_data(piece, *end && finished)
                        .map_err(stream_ended)?;
                    if !finished {
                        continue;
                    }
                }
                let ended = *end;
                this.pending = None;
                if ended {
                    return Poll::Ready(Ok(()));
                }
            }
            match ready!(this.poll_next_frame(cx)) {
                // The body ended without a frame flagged as its last: an
                // empty frame ends the stream.
                None => this.pending = Some((Bytes::new(), true)),
                // The body failed, not the stream: nothing of the peer's
                // will end it, so the call does, and its error is the
                // body's.
                Some(Err(e)) => {
                    this.stream.send_reset(Reason::CANCEL);
                    return Poll::Ready(Err(Unwritten::Body(e.into())));
                }
                Some(Ok(frame)) => match frame.into_data() {
                    Ok(data) => {
                        // A look at what follows, without waiting for it: a
                        // body that is already at its end makes this frame
                        // the stream's last.
                        let last = match Pin::new(&mut this.body).poll_frame(cx) {
                            Poll::Ready(None) => true,
                            Poll::Ready(following) => {
                                this.next = Some(following);
                                false
                            }
                            Poll::Pending => false,
                        };
                        this.pending = Some((data, last));
                    }
                    Err(frame) => {
                        if let Ok(trailers) = frame.into_trailers() {
                            this.stream.send_trailers(trailers).map_err(stream_ended)?;
                            return Poll::Ready(Ok(()));
                        }
                    }
                },
            }
        }
    }
}

impl http_body::Body for RecvBody {
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = &mut *self;
        if let Some(write) = &mut this.write {
            let written = Pin::new(write).poll(cx);
            match written {
                Poll::Ready(Ok(()) | Err(Unwritten::Stream)) => this.write = None,
                Poll::Ready(Err(Unwritten::Body(e))) => {
                    this.write = None;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Pending => {}
            }
        }
        if let Some(data) = ready!(this.stream.poll_data(cx)) {
            let data = data.map_err(|e| this.end.failure(e))?;
            this.stream
                .flow_control()
                .release_capacity(data.len())
                .map_err(|e| this.end.failure(e))?;
            return Poll::Ready(Some(Ok(Frame::data(data))));
        }
        match ready!(this.stream.poll_trailers(cx)).map_err(|e| this.end.failure(e))? {
            Some(trailers) => Poll::Ready(Some(Ok(Frame::trailers(trailers)))),
            None => Poll::Ready(None),
        }
    }

    fn is_end_stream(&self) -> bool {
        self.stream.is_end_stream()
    }
}

/// For the tests that need a dial to fail.
#[cfg(test)]
pub(crate) mod testing {
    use std::net::{Ipv4Addr, SocketAddr};

    use tokio::net::TcpSocket;

    /// A loopback endpoint whose dial is refused at once on any host: the
    /// port is bound, so nothing else can take it while this value lives, and
    /// not listening, so the host's own stack answers the connect with a
    /// reset. No name is resolved and nothing leaves the host. (A made-up
    /// scheme such as `invalid://endpoint` is no such endpoint: it is dialled
    /// as plain HTTP/2 at port 80 of the host `endpoint`, and the connect
    /// succeeds wherever that name resolves to a listener.)
    pub(crate) struct RefusedEndpoint {
        _reservation: TcpSocket,
        pub(crate) url: String,
    }

    pub(crate) fn refused_endpoint() -> RefusedEndpoint {
        let socket = TcpSocket::new_v4().expect("a TCP socket");
        socket
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("a loopback port of the system's choice");
        let port = socket.local_addr().expect("the bound port").port();
        RefusedEndpoint {
            _reservation: socket,
            url: format!("grpc://127.0.0.1:{port}"),
        }
    }

    /// A loopback endpoint whose dials black-hole: a listener whose accept
    /// queue is full drops the SYNs that follow (Linux), so a connect runs
    /// into its timeout. The queued connections live as long as this value
    /// and nothing leaves the host. Linux only, as its one test is.
    #[cfg(target_os = "linux")]
    pub(crate) struct BlackHoleEndpoint {
        _listener: tokio::net::TcpListener,
        _queued: Vec<tokio::net::TcpStream>,
        pub(crate) url: String,
    }

    #[cfg(target_os = "linux")]
    pub(crate) async fn black_hole_endpoint() -> BlackHoleEndpoint {
        let socket = TcpSocket::new_v4().expect("a TCP socket");
        socket
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("a loopback port of the system's choice");
        let address = socket.local_addr().expect("the bound address");
        let listener = socket.listen(1).expect("a listener");
        let mut queued = Vec::new();
        // The queue is full once a connect no longer completes.
        while let Ok(Ok(stream)) = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            tokio::net::TcpStream::connect(address),
        )
        .await
        {
            queued.push(stream);
            assert!(
                queued.len() < 16,
                "the accept queue behind a backlog of one did not fill"
            );
        }
        BlackHoleEndpoint {
            _listener: listener,
            _queued: queued,
            url: format!("grpc://{address}"),
        }
    }
}
