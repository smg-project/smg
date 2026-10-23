//! A numeric flag outside its bound is a configuration error: the gateway
//! exits non-zero with the field in the error, without a panic, and its port
//! is never bound; values within their bounds serve. Two representative
//! flags of the bounds table: a NaN share (which every comparison lets
//! through) and an infinite multiplier (which a lower bound lets through).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

use std::{
    fs::{self, File},
    io::{Read, Write},
    net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream},
    path::Path,
    process::{Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

const IP: Ipv4Addr = Ipv4Addr::LOCALHOST;
/// How long the gateway gets to refuse or to serve; one that does neither is
/// killed at this deadline.
const EXIT_TIMEOUT: Duration = Duration::from_secs(60);

/// What became of a launched gateway.
struct Launch {
    /// The exit status; `None` when the gateway had to be killed.
    status: Option<ExitStatus>,
    /// Whether the gateway answered `GET /health` on its port at any point.
    listened: bool,
    /// Its stdout and stderr.
    log: String,
}

/// Two distinct free ports: bound at once so the kernel hands out different
/// ones, then released for the gateway to bind moments later.
fn free_ports() -> [u16; 2] {
    let listeners = [
        TcpListener::bind((IP, 0)).expect("a free port"),
        TcpListener::bind((IP, 0)).expect("a free port"),
    ];
    listeners.map(|listener| listener.local_addr().expect("the bound address").port())
}

/// Runs the gateway with `args` on fresh ports, its output in `log_path`,
/// until it exits, binds its port or `EXIT_TIMEOUT` passes.
fn launch(args: &[&str], log_path: &Path) -> Launch {
    let [port, metrics_port] = free_ports();
    let ip = IP.to_string();
    let log = File::create(log_path).expect("log file");
    let mut child = Command::new(env!("CARGO_BIN_EXE_smg"))
        .args(["--enable-igw", "--host", &ip, "--port", &port.to_string()])
        .args(["--prometheus-host", &ip])
        .args(["--prometheus-port", &metrics_port.to_string()])
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone().expect("log handle")))
        .stderr(Stdio::from(log))
        .spawn()
        .expect("spawn the gateway");
    let deadline = Instant::now() + EXIT_TIMEOUT;
    let address = SocketAddr::from((IP, port));
    let mut listened = false;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll the gateway") {
            break Some(status);
        }
        listened = answers_health(&address);
        if listened || Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        thread::sleep(Duration::from_millis(50));
    };
    Launch {
        status,
        listened,
        log: fs::read_to_string(log_path).unwrap_or_default(),
    }
}

/// Whether the gateway answers `GET /health` with 200 on `address`: the
/// positive proof that the gateway itself serves there, where a bare connect
/// would also pass for another process that took the port in the gap after
/// `free_ports` released it.
fn answers_health(address: &SocketAddr) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(address, Duration::from_millis(200)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    if stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut head = [0u8; 64];
    let read = stream.read(&mut head).unwrap_or(0);
    head[..read].starts_with(b"HTTP/1.1 200")
}

/// `args` are refused before anything listens, with `field` in the error
/// and without a panic.
fn refuses(args: &[&str], field: &str, log_name: &str) {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(args, &dir.path().join(log_name));
    let log = &launch.log;
    assert!(!launch.listened, "the gateway served with {args:?}\n{log}");
    let status = launch.status.unwrap_or_else(|| {
        panic!("the gateway neither exited nor listened within {EXIT_TIMEOUT:?}\n{log}")
    });
    assert!(!status.success(), "the gateway exited successfully\n{log}");
    assert!(
        !log.contains("panicked"),
        "a panic is not a configuration error\n{log}"
    );
    assert!(
        log.contains(field),
        "the error does not name {field}\n{log}"
    );
}

/// A NaN share passes every comparison, so it needs the finiteness check.
#[test]
fn a_nan_warm_up_share_refuses_to_start() {
    refuses(
        &["--worker-warmup-share", "nan"],
        "worker_warmup_share",
        "nan-share.log",
    );
}

/// An infinite multiplier passes a lower bound, so it needs the finiteness
/// check too.
#[test]
fn an_infinite_retry_backoff_multiplier_refuses_to_start() {
    refuses(
        &["--retry-backoff-multiplier", "1e400"],
        "retry.backoff_multiplier",
        "inf-multiplier.log",
    );
}

/// The same flags within their bounds serve.
#[test]
fn values_within_their_bounds_serve() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &[
            "--worker-warmup-share",
            "0.25",
            "--worker-warmup-thin-ratio",
            "1.0",
            "--retry-backoff-multiplier",
            "1.5",
        ],
        &dir.path().join("within-bounds.log"),
    );
    assert!(
        launch.listened,
        "the gateway did not serve with values within their bounds (exit {:?})\n{}",
        launch.status, launch.log
    );
}
