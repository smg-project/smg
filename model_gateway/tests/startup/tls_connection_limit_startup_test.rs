//! With TLS on, the two accept sites wrap the TLS acceptor in the
//! connection-limit acceptor, as the plain-HTTP sites do: a connection beyond
//! the bound is closed at accept, before any handshake, and the refusal's
//! WARN names the listener and its flag. The plain-HTTP sites have their own
//! tests; these run the gateway with TLS on, one listener at a time, so a
//! listener argument swapped at a TLS accept site turns its test red. The
//! gateway runs as a process on purpose: the bound sits in the acceptor the
//! binary wires.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]
use std::{
    fs::{self, File},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use gateway_launch::{
    answered_200, base_args, free_ports, keep_alive_get, log_with_line, serial, Reply,
    EXIT_TIMEOUT, IP,
};
use openssl::ssl::{HandshakeError, SslConnector, SslMethod, SslStream};
use smg_test_harness::test_certs::TestCertificates;

use crate::gateway_launch;

/// Every wait on a client socket.
const IO_TIMEOUT: Duration = Duration::from_secs(5);

/// The listener under test: the name the refusal uses, its bound's flag and
/// whether it is the admin listener.
struct Listener {
    name: &'static str,
    bound_flag: &'static str,
    admin: bool,
}

const DATA: Listener = Listener {
    name: "data",
    bound_flag: "--max-connections",
    admin: false,
};

const ADMIN: Listener = Listener {
    name: "admin",
    bound_flag: "--admin-max-connections",
    admin: true,
};

/// A gateway process with TLS on, its log in a file.
struct Gateway {
    child: Child,
    log: PathBuf,
}

impl Gateway {
    /// Starts the gateway with `args`, its output in `log`, and waits until
    /// the data listener answers `GET /health` over TLS.
    fn start(args: &[String], data: SocketAddr, certs: &TestCertificates, log: &Path) -> Self {
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
            if status_over_tls(data, certs, "/health") == Some(200) {
                return gateway;
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

/// What a TLS connection to `addr` came to: the test CA as the trust root,
/// `localhost` as the verified name (a SAN of the server certificate), every
/// wait bounded by `IO_TIMEOUT`.
enum Handshake {
    /// The handshake completed: the listener serves the connection.
    Done(SslStream<TcpStream>),
    /// The connection ended before the handshake completed: nothing
    /// listening, or a connection closed at accept.
    Refused,
    /// The handshake made no progress within `IO_TIMEOUT`.
    Stalled,
}

fn tls_handshake(addr: SocketAddr, certs: &TestCertificates) -> Handshake {
    // The connector is the harness's: a failure here is the test's, not the
    // listener's.
    let mut builder = SslConnector::builder(SslMethod::tls()).expect("a TLS connector");
    builder
        .set_ca_file(&certs.ca_cert_path)
        .expect("the test CA as the trust root");
    let tcp = match TcpStream::connect_timeout(&addr, IO_TIMEOUT) {
        Ok(tcp) => tcp,
        Err(error) if error.kind() == std::io::ErrorKind::TimedOut => return Handshake::Stalled,
        Err(_) => return Handshake::Refused,
    };
    tcp.set_read_timeout(Some(IO_TIMEOUT))
        .expect("a read timeout");
    tcp.set_write_timeout(Some(IO_TIMEOUT))
        .expect("a write timeout");
    match builder.build().connect("localhost", tcp) {
        Ok(stream) => Handshake::Done(stream),
        Err(HandshakeError::WouldBlock(_)) => Handshake::Stalled,
        Err(_) => Handshake::Refused,
    }
}

/// A served TLS connection to `addr`, or `None` when the handshake did not
/// complete.
fn tls_connect(addr: SocketAddr, certs: &TestCertificates) -> Option<SslStream<TcpStream>> {
    match tls_handshake(addr, certs) {
        Handshake::Done(stream) => Some(stream),
        Handshake::Refused | Handshake::Stalled => None,
    }
}

/// The status of one `GET path` over a fresh TLS connection, closed after.
fn status_over_tls(addr: SocketAddr, certs: &TestCertificates, path: &str) -> Option<u16> {
    let mut stream = tls_connect(addr, certs)?;
    match keep_alive_get(&mut stream, path) {
        Reply::Answered(response) => response.split(' ').nth(1)?.parse().ok(),
        Reply::Closed | Reply::TimedOut => None,
    }
}

/// A TLS connection the gateway serves and keeps: the handshake done and a
/// 200 on `GET /health` within `within`, retried while the listener refuses
/// (a permit on its way back). The read answer proves the connection holds
/// a permit; a bare connect returns before the acceptor has admitted it.
fn served_tls_connection(
    addr: SocketAddr,
    certs: &TestCertificates,
    within: Duration,
) -> SslStream<TcpStream> {
    let deadline = Instant::now() + within;
    loop {
        if let Some(mut stream) = tls_connect(addr, certs) {
            if answered_200(keep_alive_get(&mut stream, "/health")) {
                return stream;
            }
        }
        assert!(
            Instant::now() < deadline,
            "no TLS connection was served within {within:?}"
        );
        thread::sleep(Duration::from_millis(50));
    }
}

/// With TLS on and a bound of one on `listener`: one held connection fills
/// the bound, the next is closed at accept before any handshake, the
/// refusal's WARN names the listener and its flag, and the listener answers
/// again once the held connection closes.
fn a_tls_listener_refuses_beyond_its_bound_and_names_itself(listener: &Listener) {
    let _serial = serial();
    let dir = tempfile::tempdir().expect("a temp dir");
    let certs = TestCertificates::generate().expect("test certificates");
    let [port, metrics_port, admin_port] = free_ports();
    let mut args = base_args(port, metrics_port);
    args.extend(
        [
            "--tls-cert-path",
            certs.server_cert_str(),
            "--tls-key-path",
            certs.server_key_str(),
            "--admin-port",
            &admin_port.to_string(),
            listener.bound_flag,
            "1",
        ]
        .map(str::to_string),
    );
    let data = SocketAddr::from((IP, port));
    let admin = SocketAddr::from((IP, admin_port));
    let under_test = if listener.admin { admin } else { data };
    let gateway = Gateway::start(&args, data, &certs, &dir.path().join("gateway.log"));

    // One served connection, kept open, holds the bound: the start's own
    // probe may hold the permit a moment longer, so this one retries. The
    // next is closed at accept, before the handshake: not served, and not
    // left hanging until the read times out.
    let held = served_tls_connection(under_test, &certs, EXIT_TIMEOUT);
    assert!(
        matches!(tls_handshake(under_test, &certs), Handshake::Refused),
        "the bound-plus-one connection was not closed before its handshake"
    );
    // The refusal's WARN names this listener and its flag: the TLS accept
    // site passes the listener, and a swapped argument there names the other.
    let refusal = format!(
        "the {} listener's connection limit ({}) is reached",
        listener.name, listener.bound_flag
    );
    if let Err(log) = log_with_line(&gateway.log, &refusal, EXIT_TIMEOUT) {
        panic!(
            "the refusal does not name the {} listener and its flag\n{log}",
            listener.name
        );
    }

    // The permit goes with the held connection; the listener answers again.
    drop(held);
    let deadline = Instant::now() + EXIT_TIMEOUT;
    while status_over_tls(under_test, &certs, "/health") != Some(200) {
        assert!(
            Instant::now() < deadline,
            "the {} listener did not answer again after the held connection closed\n{}",
            listener.name,
            gateway.log()
        );
        thread::sleep(Duration::from_millis(50));
    }
    drop(gateway);
}

#[test]
fn the_tls_data_listener_refuses_beyond_its_bound_and_names_itself() {
    a_tls_listener_refuses_beyond_its_bound_and_names_itself(&DATA);
}

#[test]
fn the_tls_admin_listener_refuses_beyond_its_bound_and_names_itself() {
    a_tls_listener_refuses_beyond_its_bound_and_names_itself(&ADMIN);
}
