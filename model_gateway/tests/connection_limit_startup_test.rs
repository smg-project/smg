//! `--max-connections` bounds the client connections the gateway holds open
//! at once. With a bound of two and two connections held: a third connection
//! is closed without a response, the two held ones go on being served, and
//! once one of them closes a new connection is served; the metrics listener,
//! outside the bound, counts the refusal. The gateway runs as a process on
//! purpose: the bound sits in the acceptor the binary wires.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

#[path = "common/gateway_launch.rs"]
mod gateway_launch;

use std::{
    fs::{self, File},
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use gateway_launch::{base_args, free_ports, serial, EXIT_TIMEOUT, IP};

/// A running gateway, killed when dropped.
struct Gateway {
    child: Child,
    log: PathBuf,
}

impl Gateway {
    /// Starts the gateway with `args`, its output in `log`, and waits until
    /// its health route answers on `addr`.
    fn start(args: &[String], addr: SocketAddr, log: &Path) -> Self {
        let file = File::create(log).expect("log file");
        let child = Command::new(env!("CARGO_BIN_EXE_smg"))
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::from(file.try_clone().expect("log handle")))
            .stderr(Stdio::from(file))
            .spawn()
            .expect("spawn the gateway");
        let mut gateway = Self {
            child,
            log: log.to_path_buf(),
        };
        let deadline = Instant::now() + EXIT_TIMEOUT;
        loop {
            if let Some(status) = gateway.child.try_wait().expect("poll the gateway") {
                panic!(
                    "the gateway exited ({status}) before serving\n{}",
                    gateway.log()
                );
            }
            if let Some(mut probe) = connect(addr) {
                if ok(request(&mut probe, "/health")) {
                    return gateway;
                }
            }
            assert!(
                Instant::now() < deadline,
                "the gateway did not serve within {EXIT_TIMEOUT:?}\n{}",
                gateway.log()
            );
            thread::sleep(Duration::from_millis(50));
        }
    }

    fn log(&self) -> String {
        fs::read_to_string(&self.log).unwrap_or_default()
    }
}

impl Drop for Gateway {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn connect(addr: SocketAddr) -> Option<TcpStream> {
    let stream = TcpStream::connect_timeout(&addr, Duration::from_millis(500)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    Some(stream)
}

/// Sends a keep-alive GET for `path` and reads its response (the head and
/// the `Content-Length` body): the text, or `None` when the connection
/// closed, or was reset, before a byte of response arrived.
fn request(stream: &mut TcpStream, path: &str) -> Option<String> {
    let _ = stream.write_all(format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes());
    let mut bytes = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) if bytes.is_empty() => return None,
            Ok(0) | Err(_) => break,
            Ok(n) => bytes.extend_from_slice(&buf[..n]),
        }
        let text = String::from_utf8_lossy(&bytes);
        if let Some(head_end) = text.find("\r\n\r\n") {
            let content_length: usize = text[..head_end]
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if bytes.len() >= head_end + 4 + content_length {
                break;
            }
        }
    }
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

fn ok(response: Option<String>) -> bool {
    response.is_some_and(|response| response.starts_with("HTTP/1.1 200"))
}

/// A connection the gateway serves: connected and answered within `within`,
/// retried while the gateway refuses (a permit on its way back).
fn served_connection(addr: SocketAddr, within: Duration) -> TcpStream {
    let deadline = Instant::now() + within;
    loop {
        if let Some(mut stream) = connect(addr) {
            if ok(request(&mut stream, "/health")) {
                return stream;
            }
        }
        assert!(
            Instant::now() < deadline,
            "no connection was served within {within:?}"
        );
        thread::sleep(Duration::from_millis(50));
    }
}

/// The metrics listener's scrape, read to the end of the connection.
fn scrape(metrics_addr: SocketAddr) -> String {
    let mut stream = connect(metrics_addr).expect("connect to the metrics listener");
    stream
        .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .unwrap();
    let mut body = String::new();
    stream.read_to_string(&mut body).expect("read the scrape");
    body
}

/// The value of the sample `family` (no labels) in a scrape.
fn sample(body: &str, family: &str) -> Option<f64> {
    body.lines()
        .find_map(|line| line.strip_prefix(family)?.strip_prefix(' '))
        .and_then(|value| value.trim().parse().ok())
}

#[test]
fn a_connection_beyond_the_bound_is_closed_without_a_response_and_counted() {
    let _serial = serial();
    let dir = tempfile::tempdir().expect("a temp dir");
    let [port, metrics_port] = free_ports();
    let mut args = base_args(port, metrics_port);
    // JSON logs, so the bound is read as a field, not through the colours of
    // the human format.
    args.extend(["--max-connections", "2", "--log-json"].map(str::to_string));
    let addr = SocketAddr::from((IP, port));
    let metrics_addr = SocketAddr::from((IP, metrics_port));
    let gateway = Gateway::start(&args, addr, &dir.path().join("gateway.log"));
    let window = Duration::from_secs(10);

    // Two connections within the bound, served and held open.
    let mut first = served_connection(addr, window);
    let mut second = served_connection(addr, window);

    // The third is closed without a byte of response.
    let mut third = connect(addr).expect("the kernel accepts the third connection");
    assert_eq!(
        request(&mut third, "/health"),
        None,
        "the connection beyond the bound was answered"
    );
    // The two within the bound are untouched.
    assert!(ok(request(&mut first, "/health")));
    assert!(ok(request(&mut second, "/health")));

    // A close returns its slot: a new connection is served.
    drop(first);
    let mut fourth = served_connection(addr, window);
    assert!(ok(request(&mut fourth, "/health")));
    assert!(ok(request(&mut second, "/health")));

    // The metrics listener is outside the bound and counts the refusal.
    let rejected = sample(&scrape(metrics_addr), "smg_http_connections_rejected_total")
        .expect("the rejected-connections counter is on the scrape");
    assert!(rejected >= 1.0, "rejected connections: {rejected}");

    let log = gateway.log();
    assert!(
        log.contains("\"max_connections\":2"),
        "the start does not report the bound\n{log}"
    );
    drop(gateway);
}
