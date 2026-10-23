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

use std::{
    fs::{self, File},
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use gateway_launch::{base_args, free_ports, log_with_line, serial, EXIT_TIMEOUT, IP};

use crate::gateway_launch;

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
            .env_remove("RUST_LOG")
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

/// What a keep-alive GET for `path` got back: the response (the head and the
/// `Content-Length` body), the connection closed or reset before a byte of
/// response arrived, or the read timing out with the connection still open
/// (which is not a closure, and fails a case that expects one).
#[derive(Debug)]
enum Reply {
    Answered(String),
    Closed,
    TimedOut,
}

fn request(stream: &mut TcpStream, path: &str) -> Reply {
    let _ = stream.write_all(format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes());
    let mut bytes = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) if bytes.is_empty() => return Reply::Closed,
            Err(e) if bytes.is_empty() => {
                return match e.kind() {
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => {
                        Reply::TimedOut
                    }
                    std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::UnexpectedEof => Reply::Closed,
                    _ => panic!("reading the response: {e}"),
                };
            }
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
    Reply::Answered(String::from_utf8_lossy(&bytes).into_owned())
}

fn ok(reply: Reply) -> bool {
    matches!(reply, Reply::Answered(response) if response.starts_with("HTTP/1.1 200"))
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

    // The third is closed without a byte of response: closed or reset, not
    // left hanging until the read times out.
    let rejected_before =
        sample(&scrape(metrics_addr), "smg_http_connections_rejected_total").unwrap_or(0.0);
    let mut third = connect(addr).expect("the kernel accepts the third connection");
    let reply = request(&mut third, "/health");
    assert!(
        matches!(reply, Reply::Closed),
        "the connection beyond the bound was answered or left hanging: {reply:?}"
    );
    // The refusal is on the counter before the refused stream is dropped,
    // and nothing else has connected to the data port since the scrape
    // before: exactly one more, the third connection's.
    let rejected = sample(&scrape(metrics_addr), "smg_http_connections_rejected_total")
        .expect("the rejected-connections counter is on the scrape");
    assert_eq!(
        rejected,
        rejected_before + 1.0,
        "the third connection's refusal: {rejected_before} -> {rejected}"
    );
    // The two within the bound are untouched.
    assert!(ok(request(&mut first, "/health")));
    assert!(ok(request(&mut second, "/health")));

    // A close returns its slot: a new connection is served.
    drop(first);
    let mut fourth = served_connection(addr, window);
    assert!(ok(request(&mut fourth, "/health")));
    assert!(ok(request(&mut second, "/health")));

    if let Err(log) = log_with_line(&gateway.log, "\"max_connections\":2", EXIT_TIMEOUT) {
        panic!("the start does not report the bound\n{log}");
    }
    drop(gateway);
}
