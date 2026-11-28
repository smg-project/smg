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
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use gateway_launch::{
    answered_200, base_args, connect, free_ports, keep_alive_get, log_with_line, serial,
    served_connection, Reply, EXIT_TIMEOUT, IP,
};

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
                if answered_200(keep_alive_get(&mut probe, "/health")) {
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
    let mut first = served_connection(addr, window, &gateway.log);
    let mut second = served_connection(addr, window, &gateway.log);

    // The third is closed without a byte of response: closed or reset, not
    // left hanging until the read times out.
    let rejected_before =
        sample(&scrape(metrics_addr), "smg_http_connections_rejected_total").unwrap_or(0.0);
    let mut third = connect(addr).expect("the kernel accepts the third connection");
    let reply = keep_alive_get(&mut third, "/health");
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
    // The refusal's WARN names this listener and its flag: the accept site
    // passes the listener, and a swapped argument there names the other one.
    let refusal = "the data listener's connection limit (--max-connections) is reached";
    if let Err(log) = log_with_line(&gateway.log, refusal, EXIT_TIMEOUT) {
        panic!("the refusal does not name the data listener and its flag\n{log}");
    }
    // The two within the bound are untouched.
    assert!(answered_200(keep_alive_get(&mut first, "/health")));
    assert!(answered_200(keep_alive_get(&mut second, "/health")));

    // A close returns its slot: a new connection is served.
    drop(first);
    let mut fourth = served_connection(addr, window, &gateway.log);
    assert!(answered_200(keep_alive_get(&mut fourth, "/health")));
    assert!(answered_200(keep_alive_get(&mut second, "/health")));

    if let Err(log) = log_with_line(&gateway.log, "\"max_connections\":2", EXIT_TIMEOUT) {
        panic!("the start does not report the bound\n{log}");
    }
    drop(gateway);
}
