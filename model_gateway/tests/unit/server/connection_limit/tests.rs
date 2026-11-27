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
        BoundedListener::Data,
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
    let stats = ConnectionLimitStats::new(BoundedListener::Data, 1);
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

/// A refusal's words name the listener and its flag, per listener, and
/// the refusing listener comes back from the bound so the closed
/// connection's error carries the same words.
#[test]
fn a_refusal_names_the_listener_and_its_flag() {
    assert_eq!(
        BoundedListener::Data.limit_reached(),
        "the data listener's connection limit (--max-connections) is reached"
    );
    assert_eq!(
        BoundedListener::Admin.limit_reached(),
        "the admin listener's connection limit (--admin-max-connections) is reached"
    );
    let stats = ConnectionLimitStats::new(BoundedListener::Admin, 1);
    let held = stats.admit().expect("within the bound");
    assert_eq!(
        stats.admit().expect_err("at the bound"),
        BoundedListener::Admin
    );
    drop(held);
    assert!(stats.admit().is_ok(), "the permit returned");
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
