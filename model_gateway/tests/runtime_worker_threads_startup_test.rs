//! `--runtime-worker-threads 0` is a configuration error: the gateway exits
//! non-zero with the field in the error, without the async runtime builder's
//! panic, and its port is never bound; a count of 1 serves. The gateway runs
//! as a process on purpose: the runtime is built in `main`, before `startup`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

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

/// Zero is refused before anything listens, with the field in the error and
/// without a panic.
#[test]
fn a_zero_runtime_worker_thread_count_refuses_to_start() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &["--runtime-worker-threads", "0"],
        &dir.path().join("zero.log"),
    );
    let log = &launch.log;
    assert!(
        !launch.listened,
        "the gateway served with a zero worker-thread count\n{log}"
    );
    let status = launch.status.unwrap_or_else(|| {
        panic!("the gateway neither exited nor listened within {EXIT_TIMEOUT:?}\n{log}")
    });
    assert!(!status.success(), "the gateway exited successfully\n{log}");
    assert!(
        !log.contains("panicked"),
        "a panic is not a configuration error\n{log}"
    );
    assert!(
        log.contains("runtime_worker_threads"),
        "the error does not name the field\n{log}"
    );
}

/// The smallest count the runtime accepts serves.
#[test]
fn a_runtime_worker_thread_count_of_one_serves() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let launch = launch(
        &["--runtime-worker-threads", "1"],
        &dir.path().join("one.log"),
    );
    assert!(
        launch.listened,
        "the gateway did not serve with one worker thread (exit {:?})\n{}",
        launch.status, launch.log
    );
}
