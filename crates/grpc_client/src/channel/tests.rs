use std::{
    future::poll_fn,
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll},
    time::Duration,
};

use http_body::Frame;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, DuplexStream},
    net::{TcpListener, TcpStream},
};
use tonic::codegen::{http, Body as _, Bytes, Service};

use super::{
    connection::{
        drive, open, Connection, End, CONNECTION_WINDOW, KEEPALIVE_INTERVAL, KEEPALIVE_TIMEOUT,
        STREAM_WINDOW,
    },
    is_transport_failure, normalize_grpc_endpoint, testing, BoxError, Channel, RecvBody, Shared,
    DEFAULT_CONNECT_TIMEOUT,
};
use crate::{tokenspeed_proto, TokenSpeedSchedulerClient};

#[test]
fn default_connect_timeout_is_below_the_kernel_syn_ceiling() {
    // The point of the default is to beat the ~127s kernel SYN timeout
    // (tcp_syn_retries=6); if it ever grows past that it stops doing its job.
    assert!(DEFAULT_CONNECT_TIMEOUT < Duration::from_secs(127));
}

#[test]
fn an_endpoint_without_a_plain_http_origin_is_refused_before_any_dial() {
    for endpoint in ["grpcs://worker:8443", "https://worker:8443", "worker:8080"] {
        let refused = Channel::unconnected(endpoint, DEFAULT_CONNECT_TIMEOUT);
        assert!(refused.is_err(), "{endpoint}");
    }
    assert!(Channel::unconnected("grpc://worker:8080", DEFAULT_CONNECT_TIMEOUT).is_ok());
    assert!(Channel::unconnected("http://127.0.0.1:1", DEFAULT_CONNECT_TIMEOUT).is_ok());
    // The refusal names the peer as host:port, not the endpoint whole.
    let refused = Channel::unconnected("grpcs://user:secret@worker:8443", DEFAULT_CONNECT_TIMEOUT)
        .err()
        .expect("a TLS endpoint is refused")
        .to_string();
    assert!(refused.starts_with("worker:8443: "), "{refused}");
    assert!(
        !refused.contains("secret"),
        "the refusal carries the userinfo: {refused}"
    );
}

/// The origin's authority, the `:authority` of every call, is the host and
/// port as the endpoint wrote them (an IPv6 literal's brackets, a port left
/// to the default), never the userinfo in front of them.
#[test]
fn the_origin_authority_is_the_host_and_port_without_the_userinfo() {
    for (endpoint, authority) in [
        ("grpc://user:secret@worker:8080", "worker:8080"),
        // A raw `@` inside the userinfo: the host follows the last one.
        ("grpc://user:p@ss@worker:8080", "worker:8080"),
        ("http://token@10.0.0.5:8000", "10.0.0.5:8000"),
        ("grpc://user:secret@[::1]:8080", "[::1]:8080"),
        ("grpc://user:secret@worker", "worker"),
        ("grpc://worker:8080", "worker:8080"),
        ("http://[::1]:8080", "[::1]:8080"),
        ("http://worker", "worker"),
    ] {
        let channel = Channel::unconnected(endpoint, DEFAULT_CONNECT_TIMEOUT).expect(endpoint);
        assert_eq!(
            channel.shared.origin.authority().map(|a| a.as_str()),
            Some(authority),
            "{endpoint}"
        );
    }
}

#[test]
fn normalize_grpc_to_http() {
    assert_eq!(
        normalize_grpc_endpoint("grpc://worker:8080"),
        "http://worker:8080"
    );
}

#[test]
fn normalize_grpcs_to_https() {
    assert_eq!(
        normalize_grpc_endpoint("grpcs://worker:8443"),
        "https://worker:8443"
    );
}

#[test]
fn normalize_passes_http_through() {
    assert_eq!(
        normalize_grpc_endpoint("http://worker:8080"),
        "http://worker:8080"
    );
}

#[test]
fn normalize_passes_https_through() {
    assert_eq!(
        normalize_grpc_endpoint("https://worker:8443"),
        "https://worker:8443"
    );
}

#[test]
fn normalize_passes_unknown_scheme_through() {
    // The channel will refuse this, but normalize is not a validator —
    // it only rewrites gRPC schemes.
    assert_eq!(
        normalize_grpc_endpoint("tcp://worker:9000"),
        "tcp://worker:9000"
    );
}

#[test]
fn normalize_passes_schemeless_through() {
    assert_eq!(normalize_grpc_endpoint("worker:8080"), "worker:8080");
}

#[test]
fn normalize_handles_path_after_authority() {
    assert_eq!(
        normalize_grpc_endpoint("grpc://worker:8080/some/path"),
        "http://worker:8080/some/path"
    );
}

#[test]
fn normalize_is_case_sensitive_on_scheme() {
    // Schemes are conventionally lowercase; the URI parser is case
    // sensitive, so we don't rewrite uppercased gRPC.
    assert_eq!(
        normalize_grpc_endpoint("GRPC://worker:8080"),
        "GRPC://worker:8080"
    );
}

// --- the test's own peers and channels ----------------------------------

/// `drive` with a connection end of its own, for a connection the test
/// drives itself.
async fn drive_alone<IO>(
    connection: h2::client::Connection<IO, Bytes>,
    pings: h2::PingPong,
    interval: Duration,
    timeout: Duration,
) -> Result<(), BoxError>
where
    IO: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    drive(connection, pings, interval, timeout, &End::default()).await
}

/// A channel over `io` with the given receive windows, the connection
/// driven with the production keep-alive; the peer is the test's.
async fn channel_over(io: DuplexStream, stream_window: u32, connection_window: u32) -> Channel {
    channel_over_with(
        io,
        stream_window,
        connection_window,
        KEEPALIVE_INTERVAL,
        KEEPALIVE_TIMEOUT,
    )
    .await
}

/// `channel_over` with the keep-alive of the test's choice.
async fn channel_over_with(
    io: DuplexStream,
    stream_window: u32,
    connection_window: u32,
    interval: Duration,
    timeout: Duration,
) -> Channel {
    let (requests, mut driver) = open(io, stream_window, connection_window)
        .await
        .expect("h2 handshake");
    let pings = driver.ping_pong().expect("the connection's pings");
    let end = Arc::new(End::default());
    let ended = Arc::clone(&end);
    #[expect(clippy::disallowed_methods, reason = "test-only connection task")]
    tokio::spawn(async move {
        let _ = drive(driver, pings, interval, timeout, &ended).await;
    });
    Channel {
        shared: Arc::new(Shared {
            origin: "http://unused.invalid".parse().expect("uri"),
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            live: Mutex::new(Some(Connection { requests, end })),
            dial: tokio::sync::Mutex::new(None),
            dials: AtomicU64::new(0),
        }),
    }
}

/// A gRPC request on `path` with `body`.
fn request(path: &str, body: tonic::body::Body) -> http::Request<tonic::body::Body> {
    http::Request::post(path)
        .header("content-type", "application/grpc")
        .header("te", "trailers")
        .body(body)
        .expect("request")
}

/// One call on `channel`, readiness first as tonic's clients do.
async fn call(
    channel: &mut Channel,
    request: http::Request<tonic::body::Body>,
) -> Result<http::Response<RecvBody>, BoxError> {
    poll_fn(|cx| channel.poll_ready(cx)).await?;
    channel.call(request).await
}

/// A body of one data frame.
struct OneFrame(Option<Bytes>);

impl http_body::Body for OneFrame {
    type Data = Bytes;
    type Error = tonic::Status;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, tonic::Status>>> {
        Poll::Ready(self.0.take().map(|data| Ok(Frame::data(data))))
    }

    fn is_end_stream(&self) -> bool {
        self.0.is_none()
    }
}

/// Serve `io` as a gRPC peer that answers every request with an OK
/// status and one empty message, over h2's server API; `on_request` reads
/// each request as it arrives. Ends with the connection.
async fn serve_grpc_ok(io: TcpStream, on_request: impl Fn(&http::Request<h2::RecvStream>)) {
    let mut connection = h2::server::handshake(io).await.expect("h2 handshake");
    while let Some(Ok((request, mut respond))) = connection.accept().await {
        on_request(&request);
        let response = http::Response::builder()
            .status(200)
            .header("content-type", "application/grpc")
            .body(())
            .expect("response");
        let mut send = respond
            .send_response(response, false)
            .expect("send headers");
        // A gRPC frame of an empty message: uncompressed, length 0.
        send.send_data(Bytes::from_static(&[0, 0, 0, 0, 0]), false)
            .expect("send the message");
        let mut trailers = http::HeaderMap::new();
        trailers.insert("grpc-status", http::HeaderValue::from_static("0"));
        send.send_trailers(trailers).expect("send trailers");
        // The request's body, unread, outlives the answer: no reset
        // for a body that arrives after the answer went out.
        drop(request);
    }
}

/// A loopback gRPC peer that answers OK on every connection until it is
/// stopped; counts its connections and its requests, and records the
/// runtime's live task count and the `:authority` at each request.
struct OkPeer {
    url: String,
    /// One task per accepted connection: their count is the peer's connection count.
    connections: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
    requests: Arc<AtomicUsize>,
    tasks_at_request: Arc<AtomicUsize>,
    /// Each request's `:authority`, as the peer decoded it from the HEADERS
    /// frame (h2's server rebuilds the request's URI from the pseudo-headers).
    authorities: Arc<Mutex<Vec<String>>>,
    serving: tokio::task::JoinHandle<()>,
}

impl OkPeer {
    async fn start(address: &str) -> Self {
        let listener = TcpListener::bind(address).await.expect("a loopback port");
        let url = format!(
            "http://{}",
            listener.local_addr().expect("the bound address")
        );
        let connections = Arc::new(Mutex::new(Vec::new()));
        let requests = Arc::new(AtomicUsize::new(0));
        let tasks_at_request = Arc::new(AtomicUsize::new(0));
        let authorities = Arc::new(Mutex::new(Vec::new()));
        let (kept, served, tasks, seen) = (
            Arc::clone(&connections),
            Arc::clone(&requests),
            Arc::clone(&tasks_at_request),
            Arc::clone(&authorities),
        );
        #[expect(
            clippy::disallowed_methods,
            reason = "test-only peer, stopped by the test"
        )]
        let serving = tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.expect("accept");
                let (served, tasks, seen) =
                    (Arc::clone(&served), Arc::clone(&tasks), Arc::clone(&seen));
                #[expect(clippy::disallowed_methods, reason = "test-only peer connection")]
                let connection = tokio::spawn(serve_grpc_ok(stream, move |request| {
                    served.fetch_add(1, Ordering::SeqCst);
                    tasks.store(
                        tokio::runtime::Handle::current()
                            .metrics()
                            .num_alive_tasks(),
                        Ordering::SeqCst,
                    );
                    seen.lock().expect("the peer's authorities").push(
                        request
                            .uri()
                            .authority()
                            .map(ToString::to_string)
                            .unwrap_or_default(),
                    );
                }));
                kept.lock()
                    .expect("the peer's connections")
                    .push(connection);
            }
        });
        Self {
            url,
            connections,
            requests,
            tasks_at_request,
            authorities,
            serving,
        }
    }

    fn connections(&self) -> usize {
        self.connections
            .lock()
            .expect("the peer's connections")
            .len()
    }

    /// The `:authority` of each request so far, in order.
    fn authorities(&self) -> Vec<String> {
        self.authorities
            .lock()
            .expect("the peer's authorities")
            .clone()
    }

    /// The peer stops listening and closes its connections; done once the
    /// runtime has dropped their tasks (the listener's port is free then).
    async fn stop(self) -> String {
        self.serving.abort();
        let _ = self.serving.await;
        let connections =
            std::mem::take(&mut *self.connections.lock().expect("the peer's connections"));
        for connection in connections {
            connection.abort();
            let _ = connection.await;
        }
        self.url
    }
}

// --- no task per call ----------------------------------------------------

/// A call spawns nothing: the runtime's live tasks are the peer's and the
/// connection's, before, during and after the call. (tonic's channel
/// added a worker task per connection; hyper's client a task per
/// request.) The answer is polled by the caller, there being nobody else.
#[tokio::test]
async fn a_call_spawns_no_task() {
    let peer = OkPeer::start("127.0.0.1:0").await;
    let client = TokenSpeedSchedulerClient::connect(&peer.url)
        .await
        .expect("the peer accepts the dial");
    // One call first, so the peer's connection task exists before the count.
    client.health_check().await.expect("the peer answers OK");
    let metrics = tokio::runtime::Handle::current().metrics();
    let before = metrics.num_alive_tasks();
    for _ in 0..3 {
        client.health_check().await.expect("the peer answers OK");
    }
    assert_eq!(peer.requests.load(Ordering::SeqCst), 4);
    assert_eq!(
        peer.tasks_at_request.load(Ordering::SeqCst),
        before,
        "a task was alive during the call that was not before it"
    );
    assert_eq!(metrics.num_alive_tasks(), before);
    assert_eq!(peer.connections(), 1);
}

/// Says why the bracketed dial was not exercised on this host.
#[expect(
    clippy::print_stderr,
    reason = "a host without IPv6 loopback is told why the test ended early"
)]
fn no_ipv6_loopback(error: &io::Error) {
    eprintln!("no IPv6 loopback on this host ({error}); the bracketed dial is not exercised here");
}

/// An origin written as a bracketed IPv6 literal (the form discovery
/// gives a worker on an IPv6 pod address) dials the address inside the
/// brackets: `Uri::host()` keeps them, the resolver cannot take them.
#[tokio::test]
async fn a_bracketed_ipv6_origin_dials_the_address_inside_the_brackets() {
    if let Err(error) = TcpListener::bind("[::1]:0").await {
        no_ipv6_loopback(&error);
        return;
    }
    let peer = OkPeer::start("[::1]:0").await;
    assert!(peer.url.starts_with("http://[::1]:"), "{}", peer.url);
    let client = TokenSpeedSchedulerClient::connect(&peer.url)
        .await
        .expect("the peer on the IPv6 loopback accepts the dial");
    client.health_check().await.expect("the peer answers OK");
    assert_eq!(peer.requests.load(Ordering::SeqCst), 1);
    // The brackets stay on the wire, as the origin wrote them.
    assert_eq!(peer.authorities(), [peer.url.trim_start_matches("http://")]);
    peer.stop().await;
}

/// The `:authority` of every call is the origin's host and port: a worker
/// URL's userinfo (`grpc://user:secret@host:port`) reaches the wire in no
/// header. The peer reads the authority of each request as h2's server
/// rebuilt it from the HEADERS frame, so what it records is what was sent.
#[tokio::test]
async fn the_authority_on_the_wire_is_the_host_and_port_without_the_userinfo() {
    let peer = OkPeer::start("127.0.0.1:0").await;
    let host_port = peer.url.trim_start_matches("http://").to_owned();
    let client = TokenSpeedSchedulerClient::connect(&format!("grpc://user:secret@{host_port}"))
        .await
        .expect("the peer accepts the dial");
    client.health_check().await.expect("the peer answers OK");
    let plain = TokenSpeedSchedulerClient::connect(&peer.url)
        .await
        .expect("the peer accepts the dial");
    plain.health_check().await.expect("the peer answers OK");
    assert_eq!(peer.authorities(), [host_port.clone(), host_port]);
    peer.stop().await;
}

/// A request body that fails on its first frame (an encoder refusing a
/// message) ends the call with the body's error at once, the stream
/// reset: nothing waits for an answer the peer cannot give.
#[tokio::test]
async fn a_request_body_that_fails_ends_the_call_with_its_error() {
    struct Refusing;
    impl http_body::Body for Refusing {
        type Data = Bytes;
        type Error = tonic::Status;
        fn poll_frame(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, tonic::Status>>> {
            Poll::Ready(Some(Err(tonic::Status::resource_exhausted(
                "the message is larger than the limit",
            ))))
        }
    }
    let peer = OkPeer::start("127.0.0.1:0").await;
    let mut channel = Channel::unconnected(&peer.url, DEFAULT_CONNECT_TIMEOUT).expect("a channel");
    poll_fn(|cx| channel.poll_ready(cx))
        .await
        .expect("the channel is ready");
    let request = http::Request::post("/smg.test.Refusing/Call")
        .body(tonic::body::Body::new(Refusing))
        .expect("request");
    let failure = tokio::time::timeout(Duration::from_secs(5), channel.call(request))
        .await
        .expect("the call ends with the body's error, not with a wait for the peer")
        .expect_err("the body refused its message");
    let status = tonic::Status::from_error(failure);
    assert_eq!(status.code(), tonic::Code::ResourceExhausted, "{status:?}");
    assert!(!is_transport_failure(&status), "{status:?}");
    peer.stop().await;
}

// --- reconnect -----------------------------------------------------------

/// A connection the peer closed fails the next call with a transport
/// failure and is re-dialled under the call after that: once, however
/// many callers find it dead.
#[tokio::test]
async fn a_dead_connection_is_redialled_once_under_the_next_calls() {
    let peer = OkPeer::start("127.0.0.1:0").await;
    let client = TokenSpeedSchedulerClient::connect(&peer.url)
        .await
        .expect("the peer accepts the dial");
    client.health_check().await.expect("the peer answers OK");
    let url = peer.stop().await;
    // The call after the close fails one of two ways, by what the
    // connection's task has read by then: the call went out on the dying
    // connection (a transport failure), or the connection was already
    // known dead and the re-dial was refused (the connect error).
    let failure = tokio::time::timeout(Duration::from_secs(10), client.health_check())
        .await
        .expect("the call ends with the connection")
        .expect_err("the connection is gone");
    assert!(
        is_transport_failure(&failure) || failure.code() == tonic::Code::Unavailable,
        "{failure:?}"
    );

    let restarted = OkPeer::start(url.trim_start_matches("http://")).await;
    let calls = (0..8).map(|_| {
        let client = client.clone();
        async move { client.health_check().await }
    });
    for answer in futures::future::join_all(calls).await {
        answer.expect("the restarted peer answers OK");
    }
    assert_eq!(restarted.requests.load(Ordering::SeqCst), 8);
    assert_eq!(
        restarted.connections(),
        1,
        "eight callers on a dead connection dial once"
    );
}

// --- the request on the wire ---------------------------------------------

/// Read one frame from `reader`: (kind, flags, stream id, payload); `None`
/// when the peer closed before a frame.
async fn read_frame<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut R,
) -> Option<(u8, u8, u32, Vec<u8>)> {
    let mut header = [0u8; 9];
    reader.read_exact(&mut header).await.ok()?;
    let len = u32::from_be_bytes([0, header[0], header[1], header[2]]) as usize;
    let id = u32::from_be_bytes([header[5], header[6], header[7], header[8]]) & 0x7fff_ffff;
    let mut payload = vec![0u8; len];
    reader
        .read_exact(&mut payload)
        .await
        .expect("a frame payload");
    Some((header[3], header[4], id, payload))
}

/// Write one frame to `writer`; a write the peer no longer takes is let go
/// (the client closes its side after a GOAWAY).
async fn write_frame<W: tokio::io::AsyncWrite + Unpin>(
    writer: &mut W,
    kind: u8,
    flags: u8,
    stream_id: u32,
    payload: &[u8],
) {
    let mut frame = Vec::with_capacity(9 + payload.len());
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes()[1..]);
    frame.push(kind);
    frame.push(flags);
    frame.extend_from_slice(&stream_id.to_be_bytes());
    frame.extend_from_slice(payload);
    let _ = writer.write_all(&frame).await;
    let _ = writer.flush().await;
}

/// A one-message request is one HEADERS frame and one DATA frame that
/// ends the stream: the message's frame carries END_STREAM, with no empty
/// frame after it.
#[tokio::test]
async fn a_one_message_request_is_one_data_frame_that_ends_the_stream() {
    const DATA: u8 = 0x0;
    const HEADERS: u8 = 0x1;
    const END_STREAM: u8 = 0x1;
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a loopback port");
    let url = format!(
        "http://{}",
        listener.local_addr().expect("the bound address")
    );
    #[expect(clippy::disallowed_methods, reason = "test-only peer, joined below")]
    let peer = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept");
        let mut preface = [0u8; 24];
        stream.read_exact(&mut preface).await.expect("the preface");
        // The server's SETTINGS, so the client's request goes out.
        stream
            .write_all(&[0, 0, 0, 0x4, 0, 0, 0, 0, 0])
            .await
            .expect("the server preface");
        let mut frames = Vec::new();
        loop {
            let (kind, flags, id, payload) = read_frame(&mut stream).await.expect("a frame");
            if id == 0 {
                continue; // SETTINGS, their ACK, WINDOW_UPDATE
            }
            frames.push((kind, flags, payload.len()));
            if flags & END_STREAM != 0 {
                return frames;
            }
        }
    });
    let client = TokenSpeedSchedulerClient::connect(&url)
        .await
        .expect("the peer accepts the dial");
    let call = tokio::time::timeout(Duration::from_millis(500), client.health_check());
    let frames = tokio::select! {
        frames = peer => frames.expect("peer task"),
        _ = call => panic!("the peer answers nothing"),
    };
    assert_eq!(frames.len(), 2, "{frames:?}");
    assert_eq!(frames[0].0, HEADERS, "{frames:?}");
    assert_eq!(frames[0].1 & END_STREAM, 0, "{frames:?}");
    assert_eq!(frames[1].0, DATA, "{frames:?}");
    assert_eq!(frames[1].1 & END_STREAM, END_STREAM, "{frames:?}");
    assert_eq!(
        frames[1].2, 5,
        "a gRPC frame of an empty message: {frames:?}"
    );
}

/// A peer that answers before it has read the request (here: a
/// trailers-only status at once, the body never read) is heard: the
/// call's answer is the peer's status, not the stalled write's end.
#[tokio::test]
async fn a_peer_that_answers_before_reading_the_request_is_heard() {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    #[expect(clippy::disallowed_methods, reason = "test-only h2 peer")]
    tokio::spawn(async move {
        let mut connection = h2::server::handshake(server_io)
            .await
            .expect("h2 handshake");
        while let Some(Ok((_request, mut respond))) = connection.accept().await {
            let response = http::Response::builder()
                .status(200)
                .header("content-type", "application/grpc")
                .header("grpc-status", "8")
                .body(())
                .expect("response");
            respond
                .send_response(response, true)
                .expect("the trailers-only answer");
        }
    });
    let mut channel = channel_over(client_io, STREAM_WINDOW, CONNECTION_WINDOW).await;
    // Four times the peer's default stream window: the write stalls.
    let body = tonic::body::Body::new(OneFrame(Some(Bytes::from(vec![0u8; 256 * 1024]))));
    let response = tokio::time::timeout(
        Duration::from_secs(10),
        call(&mut channel, request("/smg.test.Early/Unary", body)),
    )
    .await
    .expect("the answer arrives while the write waits")
    .expect("the peer's answer, not the write's end");
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers().get("grpc-status").map(|v| v.as_bytes()),
        Some(&b"8"[..])
    );
}

// --- HTTP/2 window on the wire ------------------------------------------
//
// What reaches the peer can only be checked on the wire. h2 sizes its
// budget for sub-256-byte DATA frames from the window
// (`max(window / 2, 25_600)`), which is what turns a backlog of streamed
// token frames into `GOAWAY ENHANCE_YOUR_CALM too_many_data_frames`.

/// Read `SETTINGS_INITIAL_WINDOW_SIZE` from the client's first frame.
/// `None` means the client omitted it, which h2 does at the 65535 default.
#[tokio::test]
async fn the_channel_advertises_its_stream_window() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");

    let read_settings = async {
        let (mut sock, _) = listener.accept().await.expect("accept");
        let mut preface = [0u8; 24];
        sock.read_exact(&mut preface).await.expect("read preface");
        assert_eq!(&preface[..16], b"PRI * HTTP/2.0\r\n");
        let (kind, _flags, _id, payload) = read_frame(&mut sock).await.expect("a frame");
        assert_eq!(kind, 0x4, "first frame must be SETTINGS");
        payload
    };
    // The dial resolves once the preface is flushed, so join rather
    // than race the two.
    let url = format!("http://{addr}");
    let (payload, connected) = tokio::join!(read_settings, super::connect_channel(&url));
    connected.expect("connect");

    let (entries, _) = payload.as_chunks::<6>();
    let advertised = entries
        .iter()
        .find(|e| u16::from_be_bytes([e[0], e[1]]) == 0x4)
        .map(|e| u32::from_be_bytes([e[2], e[3], e[4], e[5]]));
    assert_eq!(advertised, Some(STREAM_WINDOW));
}

#[derive(Debug)]
enum BlastOutcome {
    ClientGoAway(h2::Reason),
    StillOpen,
}

/// Have an in-process h2 server push `frames` one-byte DATA frames onto a
/// stream whose body the client never polls, and report whether the client
/// tore the connection down.
async fn blast_unread_data_frames(
    stream_window: u32,
    connection_window: u32,
    frames: usize,
) -> BlastOutcome {
    // Keep the HTTP/2 exchange in memory: closing TCP with unread DATA
    // can surface as a connection reset on macOS instead of the GOAWAY.
    // Buffer the entire blast (including frame headers) so the peer can
    // finish writing before reading the client's GOAWAY.
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);

    #[expect(clippy::disallowed_methods, reason = "test-only h2 peer, joined below")]
    let server = tokio::spawn(async move {
        let mut conn = h2::server::handshake(server_io)
            .await
            .expect("h2 handshake");
        let (_request, mut respond) = conn
            .accept()
            .await
            .expect("client opened a stream")
            .expect("valid request");
        let response = http::Response::builder()
            .status(200)
            .header("content-type", "application/grpc")
            .body(())
            .expect("response");
        let mut send = respond
            .send_response(response, false)
            .expect("send headers");
        send.reserve_capacity(frames);
        for sent in 0..frames {
            // Every frame must be queued; a short blast would let
            // `StillOpen` below pass vacuously.
            send.send_data(Bytes::from_static(b"x"), false)
                .unwrap_or_else(|err| panic!("send_data failed after {sent} frames: {err}"));
        }
        match tokio::time::timeout(Duration::from_secs(5), conn.accept()).await {
            Ok(Some(Err(err))) if err.is_go_away() && err.is_remote() => {
                BlastOutcome::ClientGoAway(err.reason().expect("go_away reason"))
            }
            Ok(Some(Err(err))) => panic!("unexpected h2 error: {err}"),
            Ok(Some(Ok(_))) => panic!("client opened a second stream"),
            Ok(None) => panic!("client closed mid-blast"),
            Err(_elapsed) => BlastOutcome::StillOpen,
        }
    });

    let mut channel = channel_over(client_io, stream_window, connection_window).await;
    let response = call(
        &mut channel,
        request("/smg.test.Blast/Stream", tonic::body::Body::empty()),
    )
    .await
    .expect("response headers");
    let _unread_body = response.into_body();

    server.await.expect("server task")
}

/// The previous profile (16 MiB stream / 32 MiB connection window) let a
/// single unread stream buffer enough token frames to exhaust the 16 MiB
/// budget (~66k one-byte frames), and the client GOAWAYed every stream on
/// the connection.
#[tokio::test]
async fn oversized_stream_window_goaways_on_unread_token_frames() {
    const MIB: u32 = 1024 * 1024;
    let outcome = blast_unread_data_frames(16 * MIB, 32 * MIB, 70_000).await;
    assert!(
        matches!(
            outcome,
            BlastOutcome::ClientGoAway(h2::Reason::ENHANCE_YOUR_CALM)
        ),
        "expected GOAWAY ENHANCE_YOUR_CALM, got {outcome:?}"
    );
}

/// Flow control stops the peer once an unread stream holds `STREAM_WINDOW`
/// bytes, and a full window of 1-byte frames still fits in the budget. The
/// connection therefore survives no matter how much the peer tries to send.
#[tokio::test]
async fn the_profile_survives_unread_token_frames() {
    let outcome = blast_unread_data_frames(
        STREAM_WINDOW,
        CONNECTION_WINDOW,
        STREAM_WINDOW as usize + 1_024,
    )
    .await;
    assert!(
        matches!(outcome, BlastOutcome::StillOpen),
        "expected the connection to survive, got {outcome:?}"
    );
}

// --- keep-alive ----------------------------------------------------------

/// Serve `io` at the frame level: the server preface, SETTINGS and
/// PING acknowledged (pings only when `answer_pings`), a count of the
/// pings received, everything else ignored. Ends when the client closes.
async fn serve_frames(io: DuplexStream, answer_pings: bool, pings: Arc<AtomicUsize>) {
    let (mut reader, mut writer) = tokio::io::split(io);
    let mut preface = [0u8; 24];
    reader
        .read_exact(&mut preface)
        .await
        .expect("client preface");
    writer
        .write_all(&[0, 0, 0, 0x4, 0, 0, 0, 0, 0])
        .await
        .expect("the server's SETTINGS");
    // Ends when the client closes.
    while let Some((kind, flags, _id, payload)) = read_frame(&mut reader).await {
        let ack = match kind {
            0x4 if flags & 0x1 == 0 => Some(vec![0, 0, 0, 0x4, 0x1, 0, 0, 0, 0]),
            0x6 if flags & 0x1 == 0 => {
                pings.fetch_add(1, Ordering::SeqCst);
                answer_pings.then(|| {
                    let mut frame = vec![0, 0, 8, 0x6, 0x1, 0, 0, 0, 0];
                    frame.extend_from_slice(&payload);
                    frame
                })
            }
            _ => None,
        };
        if let Some(frame) = ack {
            let _ = writer.write_all(&frame).await;
        }
    }
}

/// A peer that never answers a ping loses its connection at the
/// keep-alive timeout, and the call in flight on it fails with a
/// transport failure.
#[tokio::test]
async fn a_silent_peer_is_dropped_at_the_keep_alive_timeout() {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let pings = Arc::new(AtomicUsize::new(0));
    #[expect(clippy::disallowed_methods, reason = "test-only frame-level peer")]
    tokio::spawn(serve_frames(server_io, false, Arc::clone(&pings)));
    let (mut connection, mut driver) = open(client_io, STREAM_WINDOW, CONNECTION_WINDOW)
        .await
        .expect("h2 handshake");
    let ping_pong = driver.ping_pong().expect("the connection's pings");
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only connection task, joined below"
    )]
    let driven = tokio::spawn(drive_alone(
        driver,
        ping_pong,
        Duration::from_millis(50),
        Duration::from_millis(50),
    ));
    poll_fn(|cx| connection.poll_ready(cx))
        .await
        .expect("the connection takes a stream");
    let (response, _stream) = connection
        .send_request(
            http::Request::post("http://unused.invalid/smg.test.Silent/Call")
                .body(())
                .expect("request"),
            true,
        )
        .expect("the stream opens");

    let ended = tokio::time::timeout(Duration::from_secs(5), driven)
        .await
        .expect("the connection ends at the keep-alive timeout")
        .expect("connection task");
    assert!(
        ended.is_err(),
        "the keep-alive timeout is the connection's end"
    );
    assert!(pings.load(Ordering::SeqCst) >= 1, "a ping was sent");
    let failure = tokio::time::timeout(Duration::from_secs(5), response)
        .await
        .expect("the call ends with the connection")
        .expect_err("the connection is gone");
    assert!(
        is_transport_failure(&tonic::Status::from_error(Box::new(failure))),
        "the call in flight fails with a transport failure"
    );
}

/// A peer that answers its pings keeps its connection: the pings go out
/// on the interval, idle or not.
#[tokio::test]
async fn pings_go_out_on_the_interval_while_the_peer_answers_them() {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let pings = Arc::new(AtomicUsize::new(0));
    #[expect(clippy::disallowed_methods, reason = "test-only frame-level peer")]
    tokio::spawn(serve_frames(server_io, true, Arc::clone(&pings)));
    let (_connection, mut driver) = open(client_io, STREAM_WINDOW, CONNECTION_WINDOW)
        .await
        .expect("h2 handshake");
    let ping_pong = driver.ping_pong().expect("the connection's pings");
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only connection task, aborted below"
    )]
    let driven = tokio::spawn(drive_alone(
        driver,
        ping_pong,
        Duration::from_millis(20),
        Duration::from_millis(500),
    ));
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert!(
        !driven.is_finished(),
        "the connection outlives many intervals"
    );
    assert!(
        pings.load(Ordering::SeqCst) >= 3,
        "pings on the interval: {}",
        pings.load(Ordering::SeqCst)
    );
    driven.abort();
}

// --- transport faults under a call ---------------------------------------

/// tonic's status for an HTTP/2 fault carries the h2 error as its source,
/// whatever code it picked from the frame's reason; a status a worker
/// answers with, or one made of some other error, carries none.
#[test]
fn a_status_made_of_an_http2_fault_is_a_transport_failure() {
    for (reason, code) in [
        (
            h2::Reason::ENHANCE_YOUR_CALM,
            tonic::Code::ResourceExhausted,
        ),
        (h2::Reason::PROTOCOL_ERROR, tonic::Code::Internal),
        (h2::Reason::REFUSED_STREAM, tonic::Code::Unavailable),
        (h2::Reason::CANCEL, tonic::Code::Cancelled),
    ] {
        let status = tonic::Status::from_error(Box::new(h2::Error::from(reason)));
        assert_eq!(status.code(), code, "tonic's own code for {reason:?}");
        assert!(is_transport_failure(&status), "{reason:?}");
    }
    assert!(!is_transport_failure(&tonic::Status::resource_exhausted(
        "busy"
    )));
    assert!(!is_transport_failure(&tonic::Status::internal("boom")));
    // An I/O error with no HTTP layer's account over it is not one either.
    let other = tonic::Status::from_error(Box::new(io::Error::other("not http/2")));
    assert_eq!(other.code(), tonic::Code::Unknown);
    assert!(!is_transport_failure(&other));
}

/// How a loopback peer ends the socket once the request has reached it.
#[derive(Clone, Copy, Debug)]
enum SocketEnd {
    /// A close under a zero linger: a reset (RST) rather than FIN.
    Reset,
    /// A plain close with the call in flight: FIN.
    Close,
}

/// Read `stream` until the client's request is in: the 24-byte
/// connection preface, then frames up to a complete HEADERS or DATA
/// frame that ends the stream. Nothing of the request is left unread, so
/// a plain close afterwards is a FIN, not the reset the kernel sends for
/// a close over unread data.
async fn read_the_request(stream: &mut TcpStream) {
    const DATA: u8 = 0x0;
    const HEADERS: u8 = 0x1;
    const END_STREAM: u8 = 0x1;
    let mut preface = [0u8; 24];
    stream.read_exact(&mut preface).await.expect("the preface");
    loop {
        let (kind, flags, _id, _payload) = read_frame(stream).await.expect("a frame");
        if matches!(kind, DATA | HEADERS) && flags & END_STREAM != 0 {
            return;
        }
    }
}

/// A loopback peer that takes a call as far as the socket goes: it
/// answers the connection preface with its own (an empty SETTINGS
/// frame, so the client's request goes out at once), reads the request
/// and ends the socket, on every connection. The call is answered by
/// the layers below gRPC alone.
async fn a_peer_that_ends_the_socket(end: SocketEnd) -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a loopback port of the system's choice");
    let url = format!(
        "http://{}",
        listener.local_addr().expect("the bound address")
    );
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only peer; it ends with the test's runtime"
    )]
    tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.expect("accept");
            stream
                .write_all(&[0, 0, 0, 0x4, 0, 0, 0, 0, 0])
                .await
                .expect("the server preface");
            read_the_request(&mut stream).await;
            if let SocketEnd::Reset = end {
                stream.set_zero_linger().expect("a zero linger");
            }
            drop(stream);
        }
    });
    url
}

/// The `io::Error` in `status`'s source chain, by kind: one of its own,
/// or the one an `h2::Error` carries (h2 keeps it inside, not as a source).
fn io_error_kind_in(status: &tonic::Status) -> Option<io::ErrorKind> {
    let mut source = std::error::Error::source(status);
    while let Some(error) = source {
        if let Some(io) = error.downcast_ref::<io::Error>() {
            return Some(io.kind());
        }
        if let Some(io) = error
            .downcast_ref::<h2::Error>()
            .and_then(h2::Error::get_io)
        {
            return Some(io.kind());
        }
        source = error.source();
    }
    None
}

/// The socket under a call failing (the peer reset the connection, or
/// closed it with the call in flight) reaches tonic as the channel's
/// "transport error" over h2's I/O error, which carries no frame reason,
/// so tonic has no code for it but `Unknown` and the status reads as it
/// did through tonic's channel. Which `io::Error` sits behind it is the lower layers'
/// choice (h2 closes the streams of a connection whose socket ended with
/// a broken pipe of its own, a read or a write that failed carries the
/// OS error); it is the worker's connection going away under the call
/// all the same: a transport failure, either way the socket ends. A dial
/// the peer refuses is the channel's `UNAVAILABLE`, with the `io::Error`
/// under its connect error and no HTTP/2 layer's account over it: not a
/// failure under a call.
#[tokio::test]
async fn a_status_made_of_the_socket_failing_under_the_call_is_a_transport_failure() {
    for end in [SocketEnd::Reset, SocketEnd::Close] {
        let url = a_peer_that_ends_the_socket(end).await;
        let client = TokenSpeedSchedulerClient::connect(&url)
            .await
            .expect("the peer accepts the dial");
        let status = tokio::time::timeout(Duration::from_secs(10), client.health_check())
            .await
            .expect("the call ends with the socket")
            .expect_err("the peer answers no call");
        assert_eq!(status.code(), tonic::Code::Unknown, "{end:?}: {status:?}");
        assert!(
            status.message().starts_with("transport error"),
            "{end:?}: {status:?}"
        );
        assert!(
            matches!(
                io_error_kind_in(&status),
                Some(io::ErrorKind::ConnectionReset | io::ErrorKind::BrokenPipe)
            ),
            "{end:?}: {status:?}"
        );
        assert!(is_transport_failure(&status), "{end:?}: {status:?}");
    }

    let (_, status) = a_refused_dial().await;
    assert_eq!(status.code(), tonic::Code::Unavailable, "{status:?}");
    assert_eq!(
        io_error_kind_in(&status),
        Some(io::ErrorKind::ConnectionRefused),
        "{status:?}"
    );
    assert!(!is_transport_failure(&status), "{status:?}");
}

/// A call over a channel to a reserved loopback port that nothing listens
/// on: the status the refused dial produces, and the peer dialled as host:port.
async fn a_refused_dial() -> (String, tonic::Status) {
    let refused = testing::refused_endpoint();
    let peer = refused.url.trim_start_matches("grpc://").to_owned();
    let channel = Channel::unconnected(&refused.url, DEFAULT_CONNECT_TIMEOUT).expect("a channel");
    let mut client =
        tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel);
    let status = client
        .health_check(tonic::Request::new(tokenspeed_proto::HealthCheckRequest {}))
        .await
        .expect_err("nothing listens");
    (peer, status)
}

/// A dial that is refused names the peer in its error, as the dial that
/// runs into its timeout does: the log line of a failed call on a gateway
/// with many workers says which one refused.
#[tokio::test]
async fn a_refused_dial_names_the_peer_in_its_error() {
    let (peer, status) = a_refused_dial().await;
    assert_eq!(status.code(), tonic::Code::Unavailable, "{status:?}");
    assert!(
        status.message().contains(&peer),
        "the refusal does not name the peer {peer}: {status:?}"
    );
}

/// A loopback peer that accepts the connect, reads the client's
/// connection preface and resets the socket without answering it: the
/// TCP connect and the handshake's write complete, the handshake's read
/// fails. A dying engine whose accept queue is reset looks like this
/// from the client.
async fn a_peer_that_resets_the_handshake() -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a loopback port of the system's choice");
    let url = format!(
        "http://{}",
        listener.local_addr().expect("the bound address")
    );
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only peer; it ends with the test's runtime"
    )]
    tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let mut preface = [0u8; 24];
            stream
                .read_exact(&mut preface)
                .await
                .expect("the client's connection preface");
            stream.set_zero_linger().expect("a zero linger");
            drop(stream);
        }
    });
    url
}

/// A dial the peer resets during the HTTP/2 handshake, after the connect,
/// is reported like the reset under a call: the channel dials under the
/// call, the preface goes out, and the connection's first read fails, so
/// the call ends with h2's I/O error (not with the connect error a reset
/// at the TCP connect would be, which the channel reports as
/// `UNAVAILABLE`). It is the worker's connection going away under the
/// call: a transport failure.
#[tokio::test]
async fn a_dial_the_peer_resets_during_the_handshake_is_a_transport_failure() {
    let url = a_peer_that_resets_the_handshake().await;
    let channel = Channel::unconnected(&url, DEFAULT_CONNECT_TIMEOUT).expect("a channel");
    let mut client =
        tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel);
    let status = tokio::time::timeout(
        Duration::from_secs(10),
        client.health_check(tonic::Request::new(tokenspeed_proto::HealthCheckRequest {})),
    )
    .await
    .expect("the dial ends with the reset")
    .expect_err("the peer answers no handshake");
    assert_eq!(status.code(), tonic::Code::Unknown, "{status:?}");
    assert!(
        status.message().starts_with("transport error"),
        "{status:?}"
    );
    assert!(
        matches!(
            io_error_kind_in(&status),
            Some(io::ErrorKind::ConnectionReset | io::ErrorKind::BrokenPipe)
        ),
        "{status:?}"
    );
    assert!(is_transport_failure(&status), "{status:?}");
}

// --- late frames on forgotten streams ------------------------------------
//
// A stream the client reset (a dropped response body) is remembered for
// one second, fifty at a time; a frame the peer sends on it after that is
// a frame on an unknown stream, which h2 answers with a library reset
// that counts toward the connection's limit of 1024, and past the limit
// it tears the connection down: GOAWAY ENHANCE_YOUR_CALM
// "too_many_internal_resets", every call in flight failed. The status
// tonic makes of that failure is a transport fault.

/// Serve `io` at the frame level: answer every request's HEADERS with
/// response headers at once and one DATA frame `late` afterwards, whatever
/// RST_STREAM the client sent in between; report the client's GOAWAY as
/// its (error code, debug data).
async fn serve_late_frames(io: DuplexStream, late: Duration) -> (u32, Vec<u8>) {
    use tokio::{
        io::{AsyncReadExt as _, AsyncWriteExt as _, WriteHalf},
        sync::Mutex,
    };

    async fn write_frame(
        writer: &Mutex<WriteHalf<DuplexStream>>,
        kind: u8,
        flags: u8,
        stream_id: u32,
        payload: &[u8],
    ) {
        let mut frame = Vec::with_capacity(9 + payload.len());
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes()[1..]);
        frame.push(kind);
        frame.push(flags);
        frame.extend_from_slice(&stream_id.to_be_bytes());
        frame.extend_from_slice(payload);
        let mut writer = writer.lock().await;
        // The client closes its side after its GOAWAY: late writes may fail.
        let _ = writer.write_all(&frame).await;
        let _ = writer.flush().await;
    }

    let (mut reader, writer) = tokio::io::split(io);
    let writer = Arc::new(Mutex::new(writer));
    let mut preface = [0u8; 24];
    reader
        .read_exact(&mut preface)
        .await
        .expect("client preface");
    assert_eq!(&preface[..16], b"PRI * HTTP/2.0\r\n");
    write_frame(&writer, 0x4, 0, 0, &[]).await; // the server's SETTINGS
    loop {
        let mut header = [0u8; 9];
        reader
            .read_exact(&mut header)
            .await
            .expect("a frame header (the client left without a GOAWAY)");
        let len = u32::from_be_bytes([0, header[0], header[1], header[2]]) as usize;
        let (kind, flags) = (header[3], header[4]);
        let stream_id =
            u32::from_be_bytes([header[5], header[6], header[7], header[8]]) & 0x7fff_ffff;
        let mut payload = vec![0u8; len];
        reader
            .read_exact(&mut payload)
            .await
            .expect("a frame payload");
        match kind {
            // SETTINGS and PING are acknowledged; everything else the
            // client sends, RST_STREAM above all, is ignored.
            0x4 if flags & 0x1 == 0 => write_frame(&writer, 0x4, 0x1, 0, &[]).await,
            0x6 if flags & 0x1 == 0 => write_frame(&writer, 0x6, 0x1, 0, &payload).await,
            0x1 => {
                // `:status: 200` (static table index 8), END_HEADERS.
                write_frame(&writer, 0x1, 0x4, stream_id, &[0x88]).await;
                let writer = Arc::clone(&writer);
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test-only h2 peer; the task ends with its one frame"
                )]
                tokio::spawn(async move {
                    tokio::time::sleep(late).await;
                    write_frame(&writer, 0x0, 0, stream_id, b"x").await;
                });
            }
            0x7 => {
                let code = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
                return (code, payload[8..].to_vec());
            }
            _ => {}
        }
    }
}

/// The limit under the production profile, reached by late frames alone:
/// the connection goes away with "too_many_internal_resets", and the
/// status a call in flight fails with is a transport fault (tonic reads
/// the reason as RESOURCE_EXHAUSTED, which answered as a 429 would tell
/// the client of a rate limit it never hit).
#[tokio::test]
async fn late_frames_on_forgotten_streams_end_the_connection_with_a_transport_failure() {
    // Past h2's limit of 1024 local resets per connection.
    const ABANDONED: usize = 1_100;
    // Past the one second h2 remembers a reset stream for.
    const LATE: Duration = Duration::from_millis(1_500);

    let (client_io, server_io) = tokio::io::duplex(256 * 1024);
    #[expect(clippy::disallowed_methods, reason = "test-only h2 peer, joined below")]
    let server = tokio::spawn(serve_late_frames(server_io, LATE));

    let mut channel = channel_over(client_io, STREAM_WINDOW, CONNECTION_WINDOW).await;
    let late = || request("/smg.test.Late/Stream", tonic::body::Body::empty());
    // One call stays in flight, to read the connection's end from.
    let mut probe = call(&mut channel, late())
        .await
        .expect("probe headers")
        .into_body();
    for _ in 0..ABANDONED {
        let response = call(&mut channel, late()).await.expect("response headers");
        // The body goes with it: RST_STREAM CANCEL.
        drop(response);
    }

    let (code, debug) = tokio::time::timeout(Duration::from_secs(20), server)
        .await
        .expect("the client went away within the deadline")
        .expect("server task");
    assert_eq!(h2::Reason::from(code), h2::Reason::ENHANCE_YOUR_CALM);
    assert_eq!(debug, b"too_many_internal_resets");

    // The probe's own late frame arrives before the connection's end.
    let failure = loop {
        match poll_fn(|cx| Pin::new(&mut probe).poll_frame(cx)).await {
            Some(Ok(_frame)) => continue,
            Some(Err(failure)) => break failure,
            None => panic!("the probe ended cleanly, not with the connection"),
        }
    };
    let status = tonic::Status::from_error(failure);
    assert!(is_transport_failure(&status), "{status:?}");
    assert_eq!(status.code(), tonic::Code::ResourceExhausted, "{status:?}");
}

// --- the wire under load: windows, early answers, GOAWAY, resets ---------

/// Serve `io` as a gRPC peer over h2's server API with a receive window of
/// `window` per stream, reading every request's body to its end and
/// counting its bytes into `received`; the OK answer goes out before the
/// read when `answer_first`, after it otherwise.
async fn serve_reading(
    io: DuplexStream,
    window: u32,
    answer_first: bool,
    received: Arc<AtomicUsize>,
) {
    let mut connection = h2::server::Builder::new()
        .initial_window_size(window)
        .handshake::<_, Bytes>(io)
        .await
        .expect("h2 handshake");
    // The connection is driven by `accept`, so each request's reading runs
    // on a task of its own.
    while let Some(Ok((request, mut respond))) = connection.accept().await {
        let received = Arc::clone(&received);
        #[expect(clippy::disallowed_methods, reason = "test-only h2 peer stream")]
        tokio::spawn(async move {
            let response = http::Response::builder()
                .status(200)
                .header("content-type", "application/grpc")
                .body(())
                .expect("response");
            let mut answer = answer_first.then(|| {
                respond
                    .send_response(response.clone(), false)
                    .expect("the early head")
            });
            let mut body = request.into_body();
            while let Some(chunk) = body.data().await {
                let chunk = chunk.expect("the request's data");
                body.flow_control()
                    .release_capacity(chunk.len())
                    .expect("release the window");
                received.fetch_add(chunk.len(), Ordering::SeqCst);
            }
            let mut send = match answer.take() {
                Some(send) => send,
                None => respond
                    .send_response(response, false)
                    .expect("the head after the read"),
            };
            let mut trailers = http::HeaderMap::new();
            trailers.insert("grpc-status", http::HeaderValue::from_static("0"));
            send.send_trailers(trailers).expect("send trailers");
        });
    }
}

/// Reads `body` to its end: the bytes of its data frames and its trailers.
async fn read_to_end(mut body: RecvBody) -> (usize, Option<http::HeaderMap>) {
    let mut bytes = 0;
    let mut trailers = None;
    while let Some(frame) = poll_fn(|cx| Pin::new(&mut body).poll_frame(cx)).await {
        match frame.expect("a frame of the response").into_data() {
            Ok(data) => bytes += data.len(),
            Err(frame) => trailers = frame.into_trailers().ok(),
        }
    }
    (bytes, trailers)
}

/// A request four times the peer's stream window is written to its end
/// under the window, piece by piece, against a peer that reads it before
/// answering.
#[tokio::test]
async fn a_request_larger_than_the_peers_window_is_written_in_full() {
    let window = 16 * 1024;
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let received = Arc::new(AtomicUsize::new(0));
    #[expect(clippy::disallowed_methods, reason = "test-only h2 peer")]
    tokio::spawn(serve_reading(
        server_io,
        window,
        false,
        Arc::clone(&received),
    ));
    let mut channel = channel_over(client_io, STREAM_WINDOW, CONNECTION_WINDOW).await;
    let size = 4 * window as usize;
    let body = tonic::body::Body::new(OneFrame(Some(Bytes::from(vec![3u8; size]))));
    let response = tokio::time::timeout(
        Duration::from_secs(10),
        call(&mut channel, request("/smg.test.Large/Request", body)),
    )
    .await
    .expect("the write completes under the window")
    .expect("the peer answers");
    assert_eq!(response.status(), 200);
    let (_, trailers) = read_to_end(response.into_body()).await;
    assert_eq!(
        trailers.and_then(|t| t.get("grpc-status").cloned()),
        Some(http::HeaderValue::from_static("0"))
    );
    assert_eq!(received.load(Ordering::SeqCst), size);
}

/// A peer that answers first and reads the request afterwards (its head
/// neither ends the stream nor carries the status) receives all of it:
/// the write moves to the response body and finishes as that is read.
#[tokio::test]
async fn a_peer_that_answers_first_and_then_reads_the_request_receives_all_of_it() {
    let window = 16 * 1024;
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let received = Arc::new(AtomicUsize::new(0));
    #[expect(clippy::disallowed_methods, reason = "test-only h2 peer")]
    tokio::spawn(serve_reading(
        server_io,
        window,
        true,
        Arc::clone(&received),
    ));
    let mut channel = channel_over(client_io, STREAM_WINDOW, CONNECTION_WINDOW).await;
    let size = 4 * window as usize;
    let body = tonic::body::Body::new(OneFrame(Some(Bytes::from(vec![5u8; size]))));
    let response = tokio::time::timeout(
        Duration::from_secs(10),
        call(&mut channel, request("/smg.test.Early/Read", body)),
    )
    .await
    .expect("the head arrives before the request is read")
    .expect("the peer's head");
    assert_eq!(response.status(), 200);
    assert!(
        received.load(Ordering::SeqCst) < size,
        "the head came first"
    );
    let (_, trailers) =
        tokio::time::timeout(Duration::from_secs(10), read_to_end(response.into_body()))
            .await
            .expect("the rest of the request goes out as the response is read");
    assert_eq!(
        trailers.and_then(|t| t.get("grpc-status").cloned()),
        Some(http::HeaderValue::from_static("0"))
    );
    assert_eq!(received.load(Ordering::SeqCst), size);
}

/// A response four times the stream window is read to its end: every
/// frame's bytes are released back to the window as it is read, so the
/// peer is never held beyond the window.
#[tokio::test]
async fn a_response_larger_than_the_stream_window_is_read_to_the_end() {
    let size = 4 * STREAM_WINDOW as usize;
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    #[expect(clippy::disallowed_methods, reason = "test-only h2 peer")]
    tokio::spawn(async move {
        let mut connection = h2::server::handshake(server_io)
            .await
            .expect("h2 handshake");
        while let Some(Ok((_request, mut respond))) = connection.accept().await {
            // `accept` drives the connection: the sending runs on its own task.
            #[expect(clippy::disallowed_methods, reason = "test-only h2 peer stream")]
            tokio::spawn(async move {
                let response = http::Response::builder()
                    .status(200)
                    .header("content-type", "application/grpc")
                    .body(())
                    .expect("response");
                let mut send = respond
                    .send_response(response, false)
                    .expect("send headers");
                let mut remaining = size;
                while remaining > 0 {
                    send.reserve_capacity(remaining.min(64 * 1024));
                    let granted = poll_fn(|cx| send.poll_capacity(cx))
                        .await
                        .expect("the stream is open")
                        .expect("capacity");
                    let piece = granted.min(remaining);
                    send.send_data(Bytes::from(vec![9u8; piece]), false)
                        .expect("send data");
                    remaining -= piece;
                }
                let mut trailers = http::HeaderMap::new();
                trailers.insert("grpc-status", http::HeaderValue::from_static("0"));
                send.send_trailers(trailers).expect("send trailers");
            });
        }
    });
    let mut channel = channel_over(client_io, STREAM_WINDOW, CONNECTION_WINDOW).await;
    let response = call(
        &mut channel,
        request("/smg.test.Large/Response", tonic::body::Body::empty()),
    )
    .await
    .expect("the peer answers");
    let (bytes, trailers) =
        tokio::time::timeout(Duration::from_secs(10), read_to_end(response.into_body()))
            .await
            .expect("the body streams past the window");
    assert_eq!(bytes, size);
    assert_eq!(
        trailers.and_then(|t| t.get("grpc-status").cloned()),
        Some(http::HeaderValue::from_static("0"))
    );
}

/// A gRPC peer over TCP at the frame level that answers every stream's
/// HEADERS with a trailers-only OK at once, except that its first
/// connection, on its second stream, sends a GOAWAY naming the first
/// stream as the last it processed, answers that one and leaves the
/// second unanswered. `connections` counts the connections it accepted.
async fn serve_goaway_on_the_second_stream(listener: TcpListener, connections: Arc<AtomicUsize>) {
    let mut first_connection = true;
    loop {
        let (socket, _) = listener.accept().await.expect("accept");
        connections.fetch_add(1, Ordering::SeqCst);
        let goaway_on_second = std::mem::take(&mut first_connection);
        #[expect(clippy::disallowed_methods, reason = "test-only frame-level peer")]
        tokio::spawn(async move {
            let (mut reader, mut writer) = tokio::io::split(socket);
            let mut preface = [0u8; 24];
            reader
                .read_exact(&mut preface)
                .await
                .expect("client preface");
            write_frame(&mut writer, 0x4, 0, 0, &[]).await;
            let mut opened: Vec<u32> = Vec::new();
            while let Some((kind, flags, id, payload)) = read_frame(&mut reader).await {
                match kind {
                    0x4 if flags & 0x1 == 0 => write_frame(&mut writer, 0x4, 0x1, 0, &[]).await,
                    0x6 if flags & 0x1 == 0 => {
                        write_frame(&mut writer, 0x6, 0x1, 0, &payload).await;
                    }
                    0x1 => {
                        opened.push(id);
                        if goaway_on_second && opened.len() == 1 {
                            continue;
                        }
                        if goaway_on_second && opened.len() == 2 {
                            let mut goaway = opened[0].to_be_bytes().to_vec();
                            goaway.extend_from_slice(&0u32.to_be_bytes()); // NO_ERROR
                            write_frame(&mut writer, 0x7, 0, 0, &goaway).await;
                            write_frame(&mut writer, 0x1, 0x5, opened[0], &trailers_only_ok())
                                .await;
                            continue;
                        }
                        write_frame(&mut writer, 0x1, 0x5, id, &trailers_only_ok()).await;
                    }
                    _ => {}
                }
            }
        });
    }
}

/// The header block of a trailers-only OK: `:status: 200` (static table
/// index 8) and `grpc-status: 0` as a literal without indexing.
fn trailers_only_ok() -> Vec<u8> {
    let mut block = vec![0x88, 0x00];
    block.push(b"grpc-status".len() as u8);
    block.extend_from_slice(b"grpc-status");
    block.push(1);
    block.push(b'0');
    block
}

/// A GOAWAY with streams in flight: the stream at or below its last stream
/// id completes with the peer's answer, the one above fails with the
/// GOAWAY's reason (`NO_ERROR`, which tonic reads as `INTERNAL`, a
/// transport failure), and the call after it re-dials.
#[tokio::test]
async fn a_goaway_finishes_the_stream_below_its_last_id_and_fails_the_one_above() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a loopback port");
    let url = format!(
        "http://{}",
        listener.local_addr().expect("the bound address")
    );
    let connections = Arc::new(AtomicUsize::new(0));
    #[expect(
        clippy::disallowed_methods,
        reason = "test-only peer, ends with the runtime"
    )]
    tokio::spawn(serve_goaway_on_the_second_stream(
        listener,
        Arc::clone(&connections),
    ));
    let mut channel = super::connect_channel(&url)
        .await
        .expect("the peer accepts the dial");
    let mut other = channel.clone();
    let body = || tonic::body::Body::new(OneFrame(Some(Bytes::from_static(&[0, 0, 0, 0, 0]))));
    let first = call(&mut channel, request("/smg.test.GoAway/First", body()));
    let second = async {
        // The second stream opens after the first went out.
        tokio::time::sleep(Duration::from_millis(50)).await;
        call(&mut other, request("/smg.test.GoAway/Second", body())).await
    };
    let (first, second) = tokio::time::timeout(
        Duration::from_secs(10),
        futures::future::join(first, second),
    )
    .await
    .expect("both calls end with the GOAWAY");
    let first = first.expect("the stream below the last id completes");
    assert_eq!(
        first.headers().get("grpc-status").map(|v| v.as_bytes()),
        Some(&b"0"[..])
    );
    let failure =
        tonic::Status::from_error(second.expect_err("the stream above the last id fails"));
    assert_eq!(failure.code(), tonic::Code::Internal, "{failure:?}");
    assert!(is_transport_failure(&failure), "{failure:?}");
    let third = tokio::time::timeout(
        Duration::from_secs(10),
        call(&mut channel, request("/smg.test.GoAway/Third", body())),
    )
    .await
    .expect("the call after the GOAWAY ends")
    .expect("the re-dialled connection answers");
    assert_eq!(third.status(), 200);
    assert_eq!(
        connections.load(Ordering::SeqCst),
        2,
        "the GOAWAY cost one re-dial"
    );
}

/// A dropped call future and a dropped response body each reset their
/// stream with CANCEL, as the peer sees it; no stream is left behind.
#[tokio::test]
async fn a_dropped_call_and_a_dropped_body_each_reset_their_stream() {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let resets: Arc<Mutex<Vec<(String, h2::Reason)>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::new(AtomicUsize::new(0));
    let (peer_resets, peer_seen) = (Arc::clone(&resets), Arc::clone(&seen));
    #[expect(clippy::disallowed_methods, reason = "test-only h2 peer")]
    tokio::spawn(async move {
        let mut connection = h2::server::handshake(server_io)
            .await
            .expect("h2 handshake");
        while let Some(Ok((request, mut respond))) = connection.accept().await {
            let path = request.uri().path().to_owned();
            peer_seen.fetch_add(1, Ordering::SeqCst);
            let (resets, path_for_task) = (Arc::clone(&peer_resets), path.clone());
            #[expect(clippy::disallowed_methods, reason = "test-only h2 peer stream")]
            tokio::spawn(async move {
                let reason = if path_for_task.ends_with("/Call") {
                    // No answer: the client gives up on the call.
                    poll_fn(|cx| respond.poll_reset(cx)).await
                } else {
                    let response = http::Response::builder()
                        .status(200)
                        .header("content-type", "application/grpc")
                        .body(())
                        .expect("response");
                    let mut send = respond
                        .send_response(response, false)
                        .expect("send headers");
                    send.send_data(Bytes::from_static(&[0, 0, 0, 0, 0]), false)
                        .expect("a first frame");
                    poll_fn(|cx| send.poll_reset(cx)).await
                };
                resets
                    .lock()
                    .expect("the peer's resets")
                    .push((path_for_task, reason.expect("the stream's reset")));
                drop(request);
            });
        }
    });
    let mut channel = channel_over(client_io, STREAM_WINDOW, CONNECTION_WINDOW).await;
    let body = || tonic::body::Body::new(OneFrame(Some(Bytes::from_static(&[0, 0, 0, 0, 0]))));
    // A call dropped while the peer holds its answer.
    let pending = call(&mut channel, request("/smg.test.Drop/Call", body()));
    let unanswered = tokio::time::timeout(Duration::from_millis(300), pending).await;
    assert!(unanswered.is_err(), "the peer holds the answer");
    // A body dropped after the head and a first frame.
    let response = call(&mut channel, request("/smg.test.Drop/Body", body()))
        .await
        .expect("the peer's head");
    assert_eq!(response.status(), 200);
    drop(response);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while resets.lock().expect("the peer's resets").len() < 2 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "both streams are reset within the deadline"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(seen.load(Ordering::SeqCst), 2);
    let mut resets = resets.lock().expect("the peer's resets").clone();
    resets.sort_by_key(|reset| reset.0.clone());
    assert_eq!(
        resets,
        vec![
            ("/smg.test.Drop/Body".to_owned(), h2::Reason::CANCEL),
            ("/smg.test.Drop/Call".to_owned(), h2::Reason::CANCEL),
        ]
    );
}

/// Eight callers behind a dial the peer refuses each end `UNAVAILABLE`,
/// and all of them within a connect timeout or two: nothing waits on the
/// dial lock for longer than its own dial.
#[tokio::test]
async fn callers_behind_a_refused_dial_each_end_unavailable_at_once() {
    let refused = testing::refused_endpoint();
    let channel =
        Channel::unconnected(&refused.url, Duration::from_millis(200)).expect("a channel");
    let client =
        tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel);
    let started = tokio::time::Instant::now();
    let calls = (0..8).map(|_| {
        let mut client = client.clone();
        async move {
            client
                .health_check(tonic::Request::new(tokenspeed_proto::HealthCheckRequest {}))
                .await
        }
    });
    let answers = tokio::time::timeout(Duration::from_secs(5), futures::future::join_all(calls))
        .await
        .expect("every caller ends");
    for answer in answers {
        let status = answer.expect_err("nothing listens");
        assert_eq!(status.code(), tonic::Code::Unavailable, "{status:?}");
        assert!(!is_transport_failure(&status), "{status:?}");
    }
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "eight refused dials took {:?}",
        started.elapsed()
    );
}

/// Eight callers behind a dial that times out (an endpoint that drops the
/// SYN) end `UNAVAILABLE` together: the first dial's timeout is shared, the
/// seven callers queued behind it do not dial in turn.
#[cfg(target_os = "linux")]
#[tokio::test]
async fn callers_behind_a_dial_that_times_out_share_its_failure() {
    let black_hole = testing::black_hole_endpoint().await;
    let timeout = Duration::from_millis(300);
    let channel = Channel::unconnected(&black_hole.url, timeout).expect("a channel");
    let client =
        tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel);
    let started = tokio::time::Instant::now();
    let calls = (0..8).map(|_| {
        let mut client = client.clone();
        async move {
            client
                .health_check(tonic::Request::new(tokenspeed_proto::HealthCheckRequest {}))
                .await
        }
    });
    let answers = tokio::time::timeout(Duration::from_secs(5), futures::future::join_all(calls))
        .await
        .expect("every caller ends");
    for answer in answers {
        let status = answer.expect_err("nothing answers the SYN");
        assert_eq!(status.code(), tonic::Code::Unavailable, "{status:?}");
        assert!(!is_transport_failure(&status), "{status:?}");
    }
    let elapsed = started.elapsed();
    assert!(elapsed >= timeout, "the dial did not time out: {elapsed:?}");
    assert!(
        elapsed < timeout * 3,
        "eight callers paid more than one connect timeout: {elapsed:?}"
    );
}

/// A channel dropped with no call in flight ends its connection: the
/// connection task returns and the peer sees the socket close, so a probe
/// that connects once leaves nothing behind.
#[tokio::test]
async fn a_dropped_channel_ends_its_connection_task_and_closes_the_socket() {
    let peer = OkPeer::start("127.0.0.1:0").await;
    let metrics = tokio::runtime::Handle::current().metrics();
    let before = metrics.num_alive_tasks();
    let client = TokenSpeedSchedulerClient::connect(&peer.url)
        .await
        .expect("the peer accepts the dial");
    client.health_check().await.expect("the peer answers OK");
    assert_eq!(peer.connections(), 1);
    assert_eq!(
        metrics.num_alive_tasks(),
        before + 2,
        "the connection's task on each side"
    );
    drop(client);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while metrics.num_alive_tasks() > before {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the connection tasks end once the channel is gone"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    peer.stop().await;
}

/// A connection this side drops for a pong that never came answers the
/// call in flight `UNAVAILABLE` "keep-alive timed out", as tonic did; it
/// is not a transport failure (nothing of the peer's ended it).
#[tokio::test]
async fn a_keep_alive_timeout_under_a_call_is_unavailable() {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let pings = Arc::new(AtomicUsize::new(0));
    #[expect(clippy::disallowed_methods, reason = "test-only frame-level peer")]
    tokio::spawn(serve_frames(server_io, false, Arc::clone(&pings)));
    let mut channel = channel_over_with(
        client_io,
        STREAM_WINDOW,
        CONNECTION_WINDOW,
        Duration::from_millis(50),
        Duration::from_millis(50),
    )
    .await;
    let body = tonic::body::Body::new(OneFrame(Some(Bytes::from_static(&[0, 0, 0, 0, 0]))));
    let failure = tokio::time::timeout(
        Duration::from_secs(5),
        call(&mut channel, request("/smg.test.Silent/Call", body)),
    )
    .await
    .expect("the call ends with the connection")
    .expect_err("the silent peer answers nothing");
    let status = tonic::Status::from_error(failure);
    assert_eq!(status.code(), tonic::Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), "keep-alive timed out");
    assert!(!is_transport_failure(&status), "{status:?}");
    assert!(pings.load(Ordering::SeqCst) >= 1, "a ping was sent");
}

/// A subscriber that keeps the message of every event this crate logs, for
/// the tests that read what the channel says. The connection's task runs
/// on the test's current-thread runtime, so a thread-local default sees it.
struct Capture(Arc<Mutex<Vec<String>>>);

impl tracing::Subscriber for Capture {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target().starts_with(env!("CARGO_CRATE_NAME"))
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Message(String);
        impl tracing::field::Visit for Message {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0 = format!("{value:?}");
                }
            }
        }
        let mut message = Message(String::new());
        event.record(&mut message);
        self.0
            .lock()
            .expect("the captured messages")
            .push(message.0);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// Yields until a kept message contains `text` (the connection's task logs
/// between the test's polls); five seconds at most.
async fn logged(logs: &Mutex<Vec<String>>, text: &str) {
    let kept = || {
        logs.lock()
            .expect("the captured messages")
            .iter()
            .any(|m| m.contains(text))
    };
    let in_time = tokio::time::timeout(Duration::from_secs(5), async {
        while !kept() {
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(in_time.is_ok(), "{text:?} was not logged in time");
}

/// Every message the channel renders names the peer as host:port: a worker
/// URL's userinfo reaches no log line (the client's connect line, the
/// connection's end, the re-dial) and, on Linux, no error (a black-holed
/// dial's timeout as the caller reads it in the status).
#[tokio::test]
async fn diagnostics_name_the_peer_without_the_url_userinfo() {
    let logs = Arc::new(Mutex::new(Vec::new()));
    let _subscriber = tracing::subscriber::set_default(Capture(Arc::clone(&logs)));
    let without_userinfo = |message: &str| !message.contains("secret") && !message.contains("user");
    let url = a_peer_that_ends_the_socket(SocketEnd::Close).await;
    let with_secret = url.replacen("http://", "grpc://user:secret@", 1);
    // The crate's own client: its connect line is read with the channel's.
    let client = TokenSpeedSchedulerClient::connect(&with_secret)
        .await
        .expect("the peer accepts the dial");
    logged(&logs, " gRPC server at 127.0.0.1:").await;
    // The first call's connection ends under it; once its task has said so,
    // the second call re-dials.
    let _ = client.health_check().await;
    logged(&logs, "the connection to 127.0.0.1:").await;
    let _ = client.health_check().await;
    logged(&logs, "re-dialled 127.0.0.1:").await;
    for message in logs.lock().expect("the captured messages").iter() {
        assert!(
            without_userinfo(message),
            "a log line carries the URL's userinfo: {message}"
        );
    }
    #[cfg(target_os = "linux")]
    {
        let black_hole = testing::black_hole_endpoint().await;
        let with_secret = black_hole.url.replacen("://", "://user:secret@", 1);
        let channel =
            Channel::unconnected(&with_secret, Duration::from_millis(300)).expect("a channel");
        let mut client =
            tokenspeed_proto::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel);
        let status = client
            .health_check(tonic::Request::new(tokenspeed_proto::HealthCheckRequest {}))
            .await
            .expect_err("a black-holed dial");
        assert!(
            status.message().contains("no connection to 127.0.0.1:"),
            "the connect timeout names the peer: {status:?}"
        );
        assert!(
            without_userinfo(status.message()),
            "the connect timeout carries the URL's userinfo: {status:?}"
        );
    }
}
