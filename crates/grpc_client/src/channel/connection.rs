//! One connection of the channel: the TCP connect (each address the host
//! resolves to in turn, under its share of the connect timeout; nodelay and
//! the TCP keep-alive flag set, the idle time the kernel's), the HTTP/2
//! handshake with the channel's profile, and the one task that drives the
//! socket and keeps the connection alive with pings. `connect_io` takes any
//! socket, so a TLS connector would slot in under it without touching the
//! calls.

use std::{
    fmt,
    future::poll_fn,
    io,
    pin::pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use h2::{
    client::{self, SendRequest},
    Ping, PingPong,
};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::TcpStream,
    time::{sleep, Instant},
};
use tonic::codegen::{http::Uri, Bytes};
use tracing::{debug, info, warn};

use super::BoxError;

/// HTTP/2 keepalive: a ping every 30 seconds, answered within 10. The
/// interval is bounded from below by the engines' gRPC servers, not by how
/// fast a dead peer should be noticed: a grpc-core server answers a ping
/// that arrives within its `min_ping_interval_without_data` (five minutes
/// by default) of the previous one, with no data sent in between, with a
/// strike, and the second strike is a GOAWAY "too_many_pings" that fails
/// every in-flight stream with `UNAVAILABLE`. A ping every second did
/// exactly that whenever a prefill batch left the connection without data
/// for two seconds. A peer that drops its connection still fails its
/// streams at once; one that falls silent is noticed at the keepalive
/// timeout or at the next poll's deadline, whichever comes first, and the
/// gateway's liveness veto turns the failure into exclusion from routing.
///
/// The pings go out on the interval whatever the traffic (hyper's went out
/// after an interval without a read): on a busy connection the engine's
/// data between two pings keeps them off its strike count, and on an idle
/// one the cadence is the same.
pub(super) const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
pub(super) const KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(10);

/// HTTP/2 receive window per stream (one in-flight RPC).
///
/// These windows only govern what the worker sends back; requests flow under
/// the worker's own settings. The stream window caps how much of a response h2
/// buffers while SMG is not reading it, e.g. while a slow SSE client keeps the
/// bounded body channel full. Beyond that, flow control pauses the worker.
///
/// h2 charges every buffered DATA frame under 256 bytes against a single
/// budget per *connection*, and a streamed token chunk is a few dozen bytes.
/// Once the budget is gone the client sends
/// `GOAWAY ENHANCE_YOUR_CALM too_many_data_frames`, failing every RPC on the
/// connection. A 16 MiB stream window let one unread high-rate stream (an
/// `n=128` request behind a slow client) do that in about 90 seconds.
pub(super) const STREAM_WINDOW: u32 = 256 * 1024;

/// HTTP/2 receive window per connection.
///
/// Every stream is already capped at [`STREAM_WINDOW`], so this does not bound
/// memory in practice. It sizes h2's small-DATA-frame budget, which is half of
/// this window.
pub(super) const CONNECTION_WINDOW: u32 = 512 * 1024 * 1024;

// A stream window packed with 1-byte DATA frames costs `STREAM_WINDOW * 255`
// budget units. Require room for four of them, so a few stalled streams
// cannot tear down the connection they share.
const _: () = assert!((STREAM_WINDOW as usize) * 255 * 4 <= (CONNECTION_WINDOW as usize) / 2);

/// The largest header block the connection takes from the peer, advertised
/// in its SETTINGS and enforced on receipt. hyper advertised and enforced
/// the same; h2 left to itself advertises nothing and decodes up to 16 MiB.
pub(super) const MAX_HEADER_LIST_SIZE: u32 = 16 * 1024;

/// How many streams the connection opens before the peer's SETTINGS have
/// arrived. hyper's bound; h2 left to itself opens any number, and a burst
/// of calls right after a dial could run past the peer's limit and have its
/// streams refused.
pub(super) const INITIAL_MAX_SEND_STREAMS: usize = 100;

/// A connection as the calls hold it: the stream opener, and the record of
/// how the connection ended, for the calls that fail with it.
#[derive(Clone, Debug)]
pub(super) struct Connection {
    pub(super) requests: SendRequest<Bytes>,
    pub(super) end: Arc<End>,
}

/// How a connection ended, as the calls on it read their failures. h2
/// reports a connection that went away as an I/O error with no frame
/// reason, the same whether the socket failed under the call or this side
/// dropped the connection at the keep-alive timeout; the difference is
/// recorded here.
#[derive(Debug, Default)]
pub(super) struct End {
    keep_alive_timed_out: AtomicBool,
}

impl End {
    /// The error a call reports for `error`, an h2 fault on this connection.
    /// A fault with a frame reason (a GOAWAY, a stream reset, a protocol
    /// error) is h2's own error, so tonic picks the gRPC code from the
    /// reason. One without a reason is the socket under the call failing: it
    /// reads `transport error` with the I/O detail (the wording tonic's
    /// channel gave it) and keeps the h2 error as its source, for
    /// `is_transport_failure`; when it was this side that ended the
    /// connection, at the keep-alive timeout, it is `UNAVAILABLE`
    /// "keep-alive timed out", the answer tonic gave that case.
    pub(super) fn failure(&self, error: h2::Error) -> BoxError {
        if error.reason().is_some() {
            return Box::new(error);
        }
        if self.keep_alive_timed_out.load(Ordering::Acquire) {
            return Box::new(tonic::Status::unavailable("keep-alive timed out"));
        }
        Box::new(TransportError(error))
    }
}

/// The socket under a call failing, in tonic's words, the h2 error behind it.
#[derive(Debug)]
pub(super) struct TransportError(pub(super) h2::Error);

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "transport error: {}", self.0)
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// A connection to `origin`, dialled under `connect_timeout`: the TCP
/// connect, the handshake, the connection's task started.
pub(super) async fn connect(
    origin: &Uri,
    connect_timeout: Duration,
) -> Result<Connection, BoxError> {
    // `Uri::host()` keeps an IPv6 literal's brackets ("[::1]"); the resolver
    // takes the address inside them (the request's `:authority` keeps the
    // brackets, as the origin wrote them).
    let host = origin
        .host()
        .unwrap_or_default()
        .trim_start_matches('[')
        .trim_end_matches(']');
    let port = origin.port_u16().unwrap_or(80);
    let peer = super::peer_of(origin);
    let socket = tokio::time::timeout(connect_timeout, connect_tcp(host, port, connect_timeout))
        .await
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                format!("no connection to {peer} within {connect_timeout:?}"),
            )
        })??;
    connect_io(socket, peer).await
}

/// The TCP connection to `host:port` within `within`: each address the host
/// resolves to in turn, under an equal share of the time, so one that
/// black-holes does not spend it all (the connector before this one split
/// the time the same way; its 300 ms race between the address families is
/// left out: a worker's endpoint is one address). The socket gets nodelay
/// and the TCP keep-alive flag (the idle time is the kernel's); an address
/// whose socket the host cannot give (a family it lacks) is skipped for the
/// next, as one that refuses the connection is. The error kept for the
/// caller names the address it came from, as the timeout's does: a gateway
/// with many workers logs which one refused.
async fn connect_tcp(host: &str, port: u16, within: Duration) -> io::Result<TcpStream> {
    // A socket of the address's family with the keep-alive flag alone (the
    // idle time stays the kernel's: setting it needs socket2, kept out of
    // this crate) and nodelay.
    fn socket_for(address: std::net::SocketAddr) -> io::Result<tokio::net::TcpSocket> {
        let socket = if address.is_ipv4() {
            tokio::net::TcpSocket::new_v4()?
        } else {
            tokio::net::TcpSocket::new_v6()?
        };
        socket.set_keepalive(true)?;
        socket.set_nodelay(true)?;
        Ok(socket)
    }
    let addresses: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| io::Error::new(e.kind(), format!("{host}:{port}: {e}")))?
        .collect();
    let each = within / u32::try_from(addresses.len()).unwrap_or(u32::MAX).max(1);
    let mut last = io::Error::new(
        io::ErrorKind::NotFound,
        format!("{host} resolves to no address"),
    );
    for address in addresses {
        // No socket for this address (a family the host lacks) is this
        // address's failure, as a refused connection is: the next is tried.
        let socket = match socket_for(address) {
            Ok(socket) => socket,
            Err(e) => {
                last = io::Error::new(e.kind(), format!("{address}: {e}"));
                continue;
            }
        };
        match tokio::time::timeout(each, socket.connect(address)).await {
            Ok(Ok(socket)) => return Ok(socket),
            Ok(Err(e)) => last = io::Error::new(e.kind(), format!("{address}: {e}")),
            Err(_) => {
                last = io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("no connection to {address} within {each:?}"),
                );
            }
        }
    }
    Err(last)
}

/// `io` as a connection of the channel's profile: the handshake, then one
/// task for the connection's socket and keep-alive pings, for as long as it
/// lives. Generic over the socket: a TLS stream would enter here.
pub(super) async fn connect_io<IO>(io: IO, peer: String) -> Result<Connection, BoxError>
where
    IO: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (requests, mut driver) = open(io, STREAM_WINDOW, CONNECTION_WINDOW).await?;
    let pings = driver
        .ping_pong()
        .ok_or("the connection handed out no ping handle")?;
    let end = Arc::new(End::default());
    let ended = Arc::clone(&end);
    #[expect(
        clippy::disallowed_methods,
        reason = "the connection's task lives as long as the connection, not as long as a request"
    )]
    tokio::spawn(async move {
        match drive(driver, pings, KEEPALIVE_INTERVAL, KEEPALIVE_TIMEOUT, &ended).await {
            Ok(()) => debug!("the connection to {peer} closed"),
            Err(e) if ended.keep_alive_timed_out.load(Ordering::Acquire) => {
                warn!("the connection to {peer} was dropped: {e}");
            }
            Err(e) => info!("the connection to {peer} ended: {e}"),
        }
    });
    Ok(Connection { requests, end })
}

/// The HTTP/2 handshake over `io` with the given receive windows and the
/// channel's bounds; the connection future still has to be driven.
pub(super) async fn open<IO>(
    io: IO,
    stream_window: u32,
    connection_window: u32,
) -> Result<(SendRequest<Bytes>, client::Connection<IO, Bytes>), h2::Error>
where
    IO: AsyncRead + AsyncWrite + Unpin,
{
    client::Builder::new()
        .initial_window_size(stream_window)
        .initial_connection_window_size(connection_window)
        .max_header_list_size(MAX_HEADER_LIST_SIZE)
        .initial_max_send_streams(INITIAL_MAX_SEND_STREAMS)
        .enable_push(false)
        .handshake(io)
        .await
}

/// Drives one connection to its end: the socket future, with a ping every
/// `interval` and the connection dropped when its pong is `timeout` late,
/// which `end` records first. Dropping it fails the connection's streams
/// with the broken pipe h2 reports for a connection that went away, and the
/// next call re-dials.
pub(super) async fn drive<IO>(
    connection: client::Connection<IO, Bytes>,
    mut pings: PingPong,
    interval: Duration,
    timeout: Duration,
    end: &End,
) -> Result<(), BoxError>
where
    IO: AsyncRead + AsyncWrite + Unpin,
{
    let mut connection = pin!(connection);
    let mut clock = pin!(sleep(interval));
    loop {
        tokio::select! {
            ended = &mut connection => return ended.map_err(Into::into),
            () = &mut clock => {}
        }
        pings.send_ping(Ping::opaque())?;
        clock.as_mut().reset(Instant::now() + timeout);
        tokio::select! {
            ended = &mut connection => return ended.map_err(Into::into),
            pong = poll_fn(|cx| pings.poll_pong(cx)) => { pong?; }
            () = &mut clock => {
                end.keep_alive_timed_out.store(true, Ordering::Release);
                return Err(format!("no pong within {timeout:?}").into());
            }
        }
        clock.as_mut().reset(Instant::now() + interval);
    }
}
