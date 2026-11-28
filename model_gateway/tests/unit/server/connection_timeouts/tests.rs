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
use crate::server::{bind_http_server, http_client_builder};

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
/// port, runs `check` against it (with the acceptor's counters, for a
/// check that watches a close it cannot see from its end) and returns
/// the counters.
async fn with_server<F, Fut>(timeouts: ConnectionTimeouts, check: F) -> Arc<ConnectionTimeoutStats>
where
    F: FnOnce(SocketAddr, Handle<SocketAddr>, Arc<ConnectionTimeoutStats>) -> Fut,
    Fut: Future<Output = ()>,
{
    let handle = Handle::new();
    let server = bind_http_server("127.0.0.1:0".parse().unwrap(), timeouts).handle(handle.clone());
    let stats = server.get_ref().stats();
    let serving = server.serve(test_app().into_make_service_with_connect_info::<SocketAddr>());
    let checking = async {
        let addr = handle
            .listening()
            .await
            .expect("the test server did not bind");
        check(addr, handle.clone(), Arc::clone(&stats)).await;
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
        |addr, _, _| async move {
            let mut silent = connect(addr).await;
            closed(closed_within(&mut silent, WINDOW).await);
        },
    )
    .await;
    assert_eq!(counts(&stats), (1, 0));
}

#[tokio::test]
async fn a_connection_that_stops_inside_its_request_head_is_closed_at_the_header_read_timeout() {
    let stats = with_server(
        ConnectionTimeouts::from_secs(SHORT, LONG),
        |addr, _, _| async move {
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

/// The request ends under the connection task's own poll (HTTP/1 polls
/// the response there) and its end wakes nobody: the bound is armed by
/// the read hyper makes once the response is flushed, and this fails if
/// that read stops coming.
#[tokio::test]
async fn an_idle_keep_alive_connection_is_closed_at_the_idle_timeout() {
    let stats = with_server(
        ConnectionTimeouts::from_secs(LONG, SHORT),
        |addr, _, _| async move {
            let mut keep_alive = connect(addr).await;
            get_ok(&mut keep_alive).await;
            closed(closed_within(&mut keep_alive, WINDOW).await);
        },
    )
    .await;
    assert_eq!(counts(&stats), (0, 1));
}

/// An HTTP/2 client over the plain port, with prior knowledge; one
/// connection, kept alive between requests.
fn http2_client() -> reqwest::Client {
    http_client_builder()
        .http2_prior_knowledge()
        .build()
        .expect("the HTTP/2 client")
}

/// Waits up to `within` for the acceptor's idle counter to reach `n`:
/// the time it took, or `None` while it has not at the end of the
/// window.
async fn idle_closes_within(
    stats: &ConnectionTimeoutStats,
    n: u64,
    within: Duration,
) -> Option<Duration> {
    let started = Instant::now();
    while stats.closed_by_idle_timeout() < n {
        if started.elapsed() >= within {
            return None;
        }
        sleep(Duration::from_millis(20)).await;
    }
    Some(started.elapsed())
}

/// On HTTP/2 the request ends on its stream's task, which hands the
/// stream's last frame to the connection after that; the connection's
/// read after writing it arms the bound, with no wake.
#[tokio::test]
async fn an_idle_keep_alive_http2_connection_is_closed_at_the_idle_timeout() {
    let stats = with_server(
        ConnectionTimeouts::from_secs(LONG, SHORT),
        |addr, _, stats| async move {
            let client = http2_client();
            let response = client
                .get(format!("http://{addr}/ok"))
                .send()
                .await
                .expect("the request");
            assert_eq!(response.version(), http::Version::HTTP_2);
            assert_eq!(response.text().await.expect("the body"), "ok");
            closed(idle_closes_within(&stats, 1, WINDOW).await);
        },
    )
    .await;
    assert_eq!(counts(&stats), (0, 1));
}

/// A response body dropped before its end (the client reset the stream)
/// ends the request where no read of the connection follows: that end
/// wakes the connection task, which arms the bound, so the close comes
/// at the bound and not at the stream's own end. The test pins that
/// outcome (h2 wakes the task for the reset as well); the guard's own
/// wake on a drop is pinned on its waker by
/// `a_request_that_ends_under_a_poll_does_not_wake_the_connection_a_dropped_one_does`.
#[tokio::test]
async fn a_stream_reset_by_the_client_still_leaves_the_http2_connection_to_the_idle_timeout() {
    let stats = with_server(
        ConnectionTimeouts::from_secs(LONG, SHORT),
        |addr, _, stats| async move {
            let client = http2_client();
            let mut response = client
                .get(format!("http://{addr}/stream"))
                .send()
                .await
                .expect("the request");
            assert_eq!(response.version(), http::Version::HTTP_2);
            let first = response.chunk().await.expect("the first event");
            assert!(first.is_some(), "the stream ended before its first event");
            // The producer is silent for longer than the bound from here;
            // dropping the response resets the stream.
            drop(response);
            let at = closed(idle_closes_within(&stats, 1, WINDOW).await);
            assert!(
                at < PRODUCER_SILENCE,
                "the close waited for the stream's own end, not the reset: {at:?}"
            );
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
        |addr, _, _| async move {
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
        |addr, _, _| async move {
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
        |addr, _, _| async move {
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
    let stats = with_server(
        ConnectionTimeouts::from_secs(0, 0),
        |addr, _, _| async move {
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
        },
    )
    .await;
    assert_eq!(counts(&stats), (0, 0));
}

#[tokio::test]
async fn the_drain_still_closes_idle_connections_and_lets_streams_finish() {
    let stats = with_server(
        ConnectionTimeouts::from_secs(LONG, LONG),
        |addr, handle, _| async move {
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

/// A waker that counts its wakes.
struct CountingWaker(AtomicUsize);

impl std::task::Wake for CountingWaker {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn counting_waker() -> (Arc<CountingWaker>, std::task::Waker) {
    let wakes = Arc::new(CountingWaker(AtomicUsize::new(0)));
    let waker = std::task::Waker::from(Arc::clone(&wakes));
    (wakes, waker)
}

#[test]
fn a_request_that_ends_under_a_poll_does_not_wake_the_connection_a_dropped_one_does() {
    let count = Arc::new(RequestCount::default());
    let (wakes, waker) = counting_waker();

    count.idle_waker.register(&waker);
    InFlightGuard::start(Arc::clone(&count)).end();
    assert!(!count.exempt());
    assert_eq!(wakes.0.load(Ordering::Relaxed), 0);

    count.idle_waker.register(&waker);
    drop(InFlightGuard::start(Arc::clone(&count)));
    assert!(!count.exempt());
    assert_eq!(wakes.0.load(Ordering::Relaxed), 1);
}

#[test]
fn a_response_body_polled_to_its_end_releases_the_request_without_a_wake() {
    let count = Arc::new(RequestCount::default());
    let (wakes, waker) = counting_waker();
    count.idle_waker.register(&waker);

    let guard = InFlightGuard::start(Arc::clone(&count));
    let mut body = TrackedBody::new(axum::body::Body::from("ok"), Some(guard));
    assert!(count.exempt());
    let mut cx = Context::from_waker(std::task::Waker::noop());
    while let Poll::Ready(Some(frame)) = Pin::new(&mut body).poll_frame(&mut cx) {
        frame.expect("the body failed");
    }
    assert!(!count.exempt());
    assert_eq!(wakes.0.load(Ordering::Relaxed), 0);

    count.idle_waker.register(&waker);
    let guard = InFlightGuard::start(Arc::clone(&count));
    let empty = TrackedBody::new(axum::body::Body::empty(), Some(guard));
    assert!(!count.exempt());
    assert_eq!(wakes.0.load(Ordering::Relaxed), 0);
    drop(empty);
    assert_eq!(wakes.0.load(Ordering::Relaxed), 0);
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
