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
    collections::BTreeSet,
    fs::{self, File},
    io::{Read, Write},
    net::TcpStream,
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use common::test_certs::TestCertificates;
use openssl::ssl::{SslConnector, SslFiletype, SslMethod};
use portpicker::pick_unused_port;

const IP: &str = "127.0.0.1";

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
}

impl Drop for Gateway {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `n` distinct free ports.
fn free_ports(n: usize) -> Vec<u16> {
    let mut ports = BTreeSet::new();
    while ports.len() < n {
        ports.insert(pick_unused_port().expect("a free port"));
    }
    ports.into_iter().collect()
}

/// A gateway with the mesh on and this node's identity, dialing `peer_mesh_port`
/// when given; its output goes to `<name>.log` next to the certificates.
fn spawn_gateway(
    name: &'static str,
    certs: &TestCertificates,
    peer_mesh_port: Option<u16>,
) -> Gateway {
    let (cert, key) = certs.node_identity(name).expect("node identity");
    let ports = free_ports(3);
    let (http_port, mesh_port, metrics_port) = (ports[0], ports[1], ports[2]);
    let log_path = certs.temp_dir.path().join(format!("{name}.log"));
    let log = File::create(&log_path).expect("log file");
    let mut command = Command::new(env!("CARGO_BIN_EXE_smg"));
    command
        .args(["--host", IP, "--port", &http_port.to_string()])
        .args(["--prometheus-host", IP])
        .args(["--prometheus-port", &metrics_port.to_string()])
        .args(["--enable-mesh", "--mesh-server-name", name])
        .args(["--mesh-host", IP, "--mesh-port", &mesh_port.to_string()])
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

/// A plain HTTP/1.1 GET on the loopback; the whole response, headers first.
fn http_get(port: u16, path: &str) -> Option<String> {
    let mut stream = TcpStream::connect((IP, port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
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

/// A TLS handshake with the mesh listener on `port`, as a peer would open
/// it: the CA as the trust root, `identity` presented as the client
/// certificate. The protocol version and ALPN on success.
fn mesh_handshake(
    port: u16,
    certs: &TestCertificates,
    identity: &(PathBuf, PathBuf),
) -> Result<String, String> {
    let mut builder = SslConnector::builder(SslMethod::tls()).map_err(|e| e.to_string())?;
    builder
        .set_ca_file(&certs.ca_cert_path)
        .map_err(|e| e.to_string())?;
    builder
        .set_certificate_chain_file(&identity.0)
        .map_err(|e| e.to_string())?;
    builder
        .set_private_key_file(&identity.1, SslFiletype::PEM)
        .map_err(|e| e.to_string())?;
    builder
        .set_alpn_protos(b"\x02h2")
        .map_err(|e| e.to_string())?;
    let tcp = TcpStream::connect((IP, port)).map_err(|e| format!("connect: {e}"))?;
    // The node certificates carry `localhost` as a DNS SAN next to the
    // loopback addresses: the socket targets the address, the TLS stack
    // verifies the name.
    let tls = builder
        .build()
        .connect("localhost", tcp)
        .map_err(|e| format!("handshake: {e}"))?;
    let alpn = tls
        .ssl()
        .selected_alpn_protocol()
        .map(|p| String::from_utf8_lossy(p).into_owned());
    Ok(format!("{} alpn={alpn:?}", tls.ssl().version_str()))
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
    let a = spawn_gateway("nodea", &certs, None);
    let b = spawn_gateway("nodeb", &certs, Some(a.mesh_port));
    for gateway in [&a, &b] {
        assert!(
            wait_until(Duration::from_secs(90), || healthy(gateway)),
            "{} did not come up\n{}",
            gateway.name,
            logs(&[&a, &b])
        );
    }

    // The mesh listener serves TLS to a peer that presents a certificate
    // from the CA.
    let probe = certs.node_identity("probe").expect("probe identity");
    assert!(
        wait_until(Duration::from_secs(30), || mesh_handshake(
            a.mesh_port,
            &certs,
            &probe
        )
        .is_ok()),
        "no TLS handshake with {}'s mesh listener: {:?}\n{}",
        a.name,
        mesh_handshake(a.mesh_port, &certs, &probe),
        logs(&[&a, &b])
    );
    let handshake = mesh_handshake(a.mesh_port, &certs, &probe).expect("handshake");
    assert!(
        handshake.starts_with("TLSv1.3") && handshake.contains("alpn=Some(\"h2\")"),
        "unexpected handshake with the mesh listener: {handshake}"
    );

    // B joined A through it: a live peer connection on either side.
    assert!(
        wait_until(Duration::from_secs(30), || peer_connected(&a, "nodeb")
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
