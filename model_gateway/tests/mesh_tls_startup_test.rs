//! Two gateway processes with the mesh on and mTLS configured form a mesh
//! over the loopback: the mesh listener serves TLS with the node certificate
//! and the peer joins through it.
//!
//! The gateways run as separate processes on purpose. The crypto provider
//! rustls uses is process-level, this test binary installs none, and a
//! gateway that left the choice to its embedder never brought its mesh
//! listener up: the panic surfaced only as "Mesh server failed" in its log
//! while `/health` kept answering.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

mod common;

use std::{
    fs::{self, File},
    io::{self, Read, Write},
    net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use common::test_certs::TestCertificates;
use openssl::ssl::{HandshakeError, SslConnector, SslFiletype, SslMethod};

const IP: Ipv4Addr = Ipv4Addr::LOCALHOST;
/// The bound on every wait on a socket (connect, read, write): a peer that
/// stalls fails the probe instead of hanging the test.
const IO_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a gateway gets to answer `/health`.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(90);
/// How long the mesh gets to come up once both gateways are healthy.
const MESH_TIMEOUT: Duration = Duration::from_secs(30);
/// How many port picks a pair of gateways gets (`spawn_pair`).
const SPAWN_ATTEMPTS: usize = 3;

/// A gateway process; killed when dropped, so a failed assertion leaves none.
struct Gateway {
    name: &'static str,
    child: Child,
    log_path: PathBuf,
    http_port: u16,
    mesh_port: u16,
    metrics_port: u16,
}

impl Gateway {
    fn log(&self) -> String {
        fs::read_to_string(&self.log_path).unwrap_or_default()
    }

    /// Whether the process is gone: a failed bind of the HTTP or the metrics
    /// port ends the start-up.
    fn exited(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(Some(_)))
    }

    /// Whether another process took one of the gateway's ports between the
    /// pick and the gateway's own bind. The HTTP and metrics listeners report
    /// the OS error; the mesh listener's failed bind reaches the log as
    /// tonic's "transport error", the OS error being its unprinted source.
    fn lost_a_port(&self) -> bool {
        let log = self.log();
        log.contains("Address already in use")
            || log.contains("Mesh server failed: transport error")
    }

    /// Waits for `/health`; the reason when the start-up ended otherwise. A
    /// lost port ends the wait too and is the caller's to handle.
    fn start_up(&mut self) -> Option<&'static str> {
        wait_until(STARTUP_TIMEOUT, || {
            healthy(self) || self.lost_a_port() || self.exited()
        });
        if healthy(self) || self.lost_a_port() {
            None
        } else if self.exited() {
            Some("exited before it was healthy")
        } else {
            Some("did not come up")
        }
    }
}

impl Drop for Gateway {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `N` distinct free ports: all bound at once, so the kernel hands out
/// different ones, then released for the gateways to bind moments later. A
/// gateway that lost one in between is respawned on fresh ones
/// (`spawn_pair`).
fn free_ports<const N: usize>() -> [u16; N] {
    let listeners: [TcpListener; N] =
        std::array::from_fn(|_| TcpListener::bind((IP, 0)).expect("a free port"));
    listeners.map(|listener| listener.local_addr().expect("the bound address").port())
}

/// A gateway with the mesh on and this node's identity, on `ports` (HTTP,
/// mesh, metrics), dialing `peer_mesh_port` when given; its output goes to
/// `<name>.log` next to the certificates.
fn spawn_gateway(
    name: &'static str,
    certs: &TestCertificates,
    ports: [u16; 3],
    peer_mesh_port: Option<u16>,
) -> Gateway {
    let (cert, key) = certs.node_identity(name).expect("node identity");
    let [http_port, mesh_port, metrics_port] = ports;
    let ip = IP.to_string();
    let log_path = certs.temp_dir.path().join(format!("{name}.log"));
    let log = File::create(&log_path).expect("log file");
    let mut command = Command::new(env!("CARGO_BIN_EXE_smg"));
    command
        .args(["--host", &ip, "--port", &http_port.to_string()])
        .args(["--prometheus-host", &ip])
        .args(["--prometheus-port", &metrics_port.to_string()])
        .args(["--enable-mesh", "--mesh-server-name", name])
        .args(["--mesh-host", &ip, "--mesh-port", &mesh_port.to_string()])
        .args(["--mesh-tls-ca-cert", certs.ca_cert_str()])
        .arg("--mesh-tls-cert")
        .arg(&cert)
        .arg("--mesh-tls-key")
        .arg(&key)
        .args(["--log-level", "info"])
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone().expect("log handle")))
        .stderr(Stdio::from(log));
    if let Some(port) = peer_mesh_port {
        command.args(["--mesh-peer-urls", &format!("{IP}:{port}")]);
    }
    let child = command.spawn().expect("spawn the gateway");
    Gateway {
        name,
        child,
        log_path,
        http_port,
        mesh_port,
        metrics_port,
    }
}

/// Both gateways healthy, B dialing A's mesh port. The pair is respawned on
/// fresh ports when a gateway lost one of its ports between the pick and its
/// bind, at most `SPAWN_ATTEMPTS` times.
fn spawn_pair(certs: &TestCertificates) -> (Gateway, Gateway) {
    let mut attempts = 0;
    loop {
        attempts += 1;
        let [http_a, mesh_a, metrics_a, http_b, mesh_b, metrics_b] = free_ports::<6>();
        let mut a = spawn_gateway("nodea", certs, [http_a, mesh_a, metrics_a], None);
        let mut b = spawn_gateway("nodeb", certs, [http_b, mesh_b, metrics_b], Some(mesh_a));
        let failures = [a.start_up(), b.start_up()];
        if a.lost_a_port() || b.lost_a_port() {
            assert!(
                attempts < SPAWN_ATTEMPTS,
                "no gateway pair kept its ports in {SPAWN_ATTEMPTS} picks\n{}",
                logs(&[&a, &b])
            );
            // Both are dropped here, so killed; fresh ports next.
            continue;
        }
        for (gateway, failure) in [&a, &b].into_iter().zip(failures) {
            assert!(
                failure.is_none(),
                "{} {}\n{}",
                gateway.name,
                failure.unwrap_or_default(),
                logs(&[&a, &b])
            );
        }
        return (a, b);
    }
}

/// A connection to the loopback `port` with `IO_TIMEOUT` on the connect and
/// on every read and write.
fn connect(port: u16) -> io::Result<TcpStream> {
    let stream = TcpStream::connect_timeout(&SocketAddr::from((IP, port)), IO_TIMEOUT)?;
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))?;
    Ok(stream)
}

/// A plain HTTP/1.1 GET on the loopback; the whole response, headers first.
fn http_get(port: u16, path: &str) -> Option<String> {
    let mut stream = connect(port).ok()?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {IP}\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    Some(response)
}

fn healthy(gateway: &Gateway) -> bool {
    http_get(gateway.http_port, "/health").is_some_and(|r| r.starts_with("HTTP/1.1 200"))
}

/// Whether `gateway` reports a live mesh connection with `peer`
/// (`router_mesh_peer_connections{peer="<peer>"} 1` in its metrics).
fn peer_connected(gateway: &Gateway, peer: &str) -> bool {
    let label = format!("peer=\"{peer}\"");
    http_get(gateway.metrics_port, "/metrics").is_some_and(|metrics| {
        metrics.lines().any(|line| {
            line.starts_with("router_mesh_peer_connections")
                && line.contains(&label)
                && line.trim_end().ends_with(" 1")
        })
    })
}

/// What the mesh listener made of a handshake.
#[derive(Debug)]
enum Handshake {
    /// Negotiated: the protocol version and the ALPN protocol.
    Accepted(String),
    /// Ended by the listener with a TLS alert: OpenSSL's reason for it, such
    /// as "tlsv13 alert certificate required".
    Refused(String),
}

/// The alert in an OpenSSL error, when the error is one the peer sent.
fn alert_reason(error: &openssl::ssl::Error) -> Option<String> {
    error
        .ssl_error()?
        .errors()
        .iter()
        .filter_map(|e| e.reason())
        .find(|reason| reason.contains("alert"))
        .map(str::to_owned)
}

/// A TLS handshake with the mesh listener on `port`, as a peer would open
/// it: the CA as the trust root, `identity` presented as the client
/// certificate when given. Every wait on the socket is bounded by
/// `IO_TIMEOUT`; `Err` is a transport failure (nothing listening, a stalled
/// peer, a connection closed without a verdict), not the listener's answer.
fn mesh_handshake(
    port: u16,
    certs: &TestCertificates,
    identity: Option<&(PathBuf, PathBuf)>,
) -> Result<Handshake, String> {
    let mut builder = SslConnector::builder(SslMethod::tls()).map_err(|e| e.to_string())?;
    builder
        .set_ca_file(&certs.ca_cert_path)
        .map_err(|e| e.to_string())?;
    if let Some((cert, key)) = identity {
        builder
            .set_certificate_chain_file(cert)
            .map_err(|e| e.to_string())?;
        builder
            .set_private_key_file(key, SslFiletype::PEM)
            .map_err(|e| e.to_string())?;
    }
    builder
        .set_alpn_protos(b"\x02h2")
        .map_err(|e| e.to_string())?;
    let tcp = connect(port).map_err(|e| format!("connect: {e}"))?;
    // The node certificates carry `localhost` as a DNS SAN next to the
    // loopback addresses: the socket targets the address, the TLS stack
    // verifies the name.
    let mut tls = match builder.build().connect("localhost", tcp) {
        Ok(tls) => tls,
        Err(HandshakeError::Failure(stream)) => {
            return alert_reason(stream.error())
                .map(Handshake::Refused)
                .ok_or_else(|| format!("handshake: {}", stream.error()));
        }
        Err(HandshakeError::WouldBlock(_)) => {
            return Err(format!(
                "handshake: the listener stalled for {IO_TIMEOUT:?}"
            ));
        }
        Err(HandshakeError::SetupFailure(e)) => return Err(format!("handshake setup: {e}")),
    };
    if identity.is_none() {
        // In TLS 1.3 the client's side of the handshake completes before the
        // listener has judged the client certificate; its verdict is the
        // first thing read.
        return match tls.ssl_read(&mut [0u8; 1]) {
            Err(e) => alert_reason(&e)
                .map(Handshake::Refused)
                .ok_or_else(|| format!("no verdict from the listener: {e}")),
            Ok(n) => Err(format!("{n} bytes from the listener instead of a verdict")),
        };
    }
    let alpn = tls
        .ssl()
        .selected_alpn_protocol()
        .map(|p| String::from_utf8_lossy(p).into_owned());
    Ok(Handshake::Accepted(format!(
        "{} alpn={alpn:?}",
        tls.ssl().version_str()
    )))
}

fn wait_until(timeout: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if condition() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn logs(gateways: &[&Gateway]) -> String {
    gateways
        .iter()
        .map(|g| format!("--- {} ---\n{}", g.name, g.log()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn mesh_with_mtls_forms_between_two_gateway_processes() {
    let certs = TestCertificates::generate().expect("test certificates");
    let (a, b) = spawn_pair(&certs);

    // The mesh listener serves TLS to a peer that presents a certificate
    // from the CA.
    let probe = certs.node_identity("probe").expect("probe identity");
    let mut handshake = Err("not attempted".to_owned());
    assert!(
        wait_until(MESH_TIMEOUT, || {
            handshake = mesh_handshake(a.mesh_port, &certs, Some(&probe));
            handshake.is_ok()
        }),
        "no TLS handshake with {}'s mesh listener: {handshake:?}\n{}",
        a.name,
        logs(&[&a, &b])
    );
    assert!(
        matches!(
            &handshake,
            Ok(Handshake::Accepted(negotiated))
                if negotiated.starts_with("TLSv1.3")
                    && negotiated.contains("alpn=Some(\"h2\")")
        ),
        "unexpected handshake with the mesh listener: {handshake:?}"
    );

    // Without a client certificate the peer is refused, and with the
    // "certificate required" alert: not a plain close, not a stall.
    let refusal = mesh_handshake(a.mesh_port, &certs, None);
    assert!(
        matches!(
            &refusal,
            Ok(Handshake::Refused(reason)) if reason.contains("certificate required")
        ),
        "a peer without a client certificate was not refused by {}'s mesh listener: \
         {refusal:?}\n{}",
        a.name,
        logs(&[&a, &b])
    );

    // B joined A through it: a live peer connection on either side.
    assert!(
        wait_until(MESH_TIMEOUT, || peer_connected(&a, "nodeb")
            || peer_connected(&b, "nodea")),
        "the two gateways did not form a mesh\n{}",
        logs(&[&a, &b])
    );
    for gateway in [&a, &b] {
        assert!(
            !gateway.log().contains("Mesh server failed"),
            "{}'s mesh server failed\n{}",
            gateway.name,
            gateway.log()
        );
    }
}
