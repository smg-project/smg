//! `--tls-client-auth` in the binary: with `required`, a client with a
//! certificate from the configured CA is served and identified by it, a
//! client without one does not complete the handshake; with `optional` and
//! an API key, the certificate stands in for the key; the metrics listener
//! counts the results; the admin listener (`--admin-port`) asks for no
//! client certificate; a mode without its bundle, and a bundle without a
//! mode, are refused before anything binds. The gateway runs as a process
//! on purpose: the listener's TLS configuration and acceptor chain are built
//! by the binary at start-up.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::allow_attributes
)]

#[path = "common/gateway_launch.rs"]
mod gateway_launch;
#[path = "common/test_certs.rs"]
mod test_certs;

use std::{
    fs::{self, File},
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use gateway_launch::{base_args, free_ports, launch, serial, Launch, Until, EXIT_TIMEOUT, IP};
use openssl::ssl::{SslConnector, SslFiletype, SslMethod, SslVerifyMode};
use test_certs::TestCertificates;

const SPIFFE: &str = "spiffe://cluster.local/ns/default/sa/team-red";
const IO_TIMEOUT: Duration = Duration::from_secs(10);

/// A running gateway, killed when dropped.
struct Gateway {
    child: Child,
    log: PathBuf,
}

impl Gateway {
    /// Starts the gateway with `args`, its output in `log`, and waits until
    /// its health route answers over TLS on `addr` (as `probe` connects).
    fn start(args: &[String], addr: SocketAddr, probe: &SslConnector, log: &Path) -> Self {
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
            if let Ok((200, _)) = https(probe, addr, "GET", "/health", None) {
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

/// A TLS client trusting the test CA, presenting `identity` when given.
fn connector(certs: &TestCertificates, identity: Option<&(PathBuf, PathBuf)>) -> SslConnector {
    let mut builder = SslConnector::builder(SslMethod::tls()).unwrap();
    builder.set_ca_file(&certs.ca_cert_path).unwrap();
    builder.set_verify(SslVerifyMode::PEER);
    if let Some((cert, key)) = identity {
        builder
            .set_certificate_file(cert, SslFiletype::PEM)
            .unwrap();
        builder.set_private_key_file(key, SslFiletype::PEM).unwrap();
    }
    builder.build()
}

/// One HTTPS request over a fresh connection: the status and the raw
/// response, or the handshake's or transport's error.
fn https(
    connector: &SslConnector,
    addr: SocketAddr,
    method: &str,
    path: &str,
    authorization: Option<&str>,
) -> Result<(u16, String), String> {
    let tcp = TcpStream::connect_timeout(&addr, IO_TIMEOUT).map_err(|e| e.to_string())?;
    tcp.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
    tcp.set_write_timeout(Some(IO_TIMEOUT)).unwrap();
    let mut tls = connector
        .connect("localhost", tcp)
        .map_err(|e| format!("handshake: {e}"))?;
    let body = r#"{"model":"any","messages":[{"role":"user","content":"hi"}]}"#;
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
        if method == "POST" { body.len() } else { 0 }
    );
    if let Some(authorization) = authorization {
        head.push_str(&format!("Authorization: {authorization}\r\n"));
    }
    head.push_str("\r\n");
    tls.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    if method == "POST" {
        tls.write_all(body.as_bytes()).map_err(|e| e.to_string())?;
    }
    let mut response = String::new();
    // The gateway closes after the response; a close without close_notify
    // is reported as an error after the bytes arrived.
    let _ = tls.read_to_string(&mut response);
    let status = response
        .strip_prefix("HTTP/1.1 ")
        .and_then(|rest| rest.get(..3))
        .and_then(|status| status.parse().ok())
        .ok_or_else(|| format!("no status line in {response:?}"))?;
    Ok((status, response))
}

/// A plain-HTTP scrape of the metrics listener.
fn scrape(port: u16) -> String {
    let mut stream = TcpStream::connect_timeout(&SocketAddr::from((IP, port)), IO_TIMEOUT).unwrap();
    stream.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
    write!(
        stream,
        "GET /metrics HTTP/1.1\r\nHost: {IP}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut body = String::new();
    stream.read_to_string(&mut body).unwrap();
    body
}

/// The value of the `smg_tls_client_auth_total` sample with `result`.
fn client_auth_count(metrics: &str, result: &str) -> Option<f64> {
    metrics
        .lines()
        .find(|line| {
            line.starts_with("smg_tls_client_auth_total{")
                && line.contains(&format!("result=\"{result}\""))
        })
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
}

/// The listener's flags: TLS with the test CA's server certificate, client
/// authentication under `mode` with the CA as the bundle, plus `extra`.
fn tls_args(
    certs: &TestCertificates,
    port: u16,
    metrics_port: u16,
    mode: &str,
    extra: &[&str],
) -> Vec<String> {
    let mut args = base_args(port, metrics_port);
    args.extend(
        [
            "--tls-cert-path",
            certs.server_cert_str(),
            "--tls-key-path",
            certs.server_key_str(),
            "--tls-client-ca-path",
            certs.ca_cert_str(),
            "--tls-client-auth",
            mode,
            "--log-json",
        ]
        .map(str::to_string),
    );
    args.extend(extra.iter().map(|arg| (*arg).to_string()));
    args
}

#[test]
fn required_client_certificates_identify_the_caller_and_turn_the_rest_away() {
    let _serial = serial();
    let certs = TestCertificates::generate().unwrap();
    let identity = certs.client_identity("team-red", SPIFFE).unwrap();
    let with_identity = connector(&certs, Some(&identity));
    let bare = connector(&certs, None);
    let [port, metrics_port] = free_ports();
    let addr = SocketAddr::from((IP, port));
    let log = certs.temp_dir.path().join("required.log");
    let gateway = Gateway::start(
        &tls_args(&certs, port, metrics_port, "required", &[]),
        addr,
        &with_identity,
        &log,
    );

    // The identity passes the serving routes' authentication (the route then
    // finds no worker for the model, which is not 401).
    let (status, response) = https(&with_identity, addr, "POST", "/v1/chat/completions", None)
        .unwrap_or_else(|e| panic!("{e}\n{}", gateway.log()));
    assert_ne!(status, 401, "{response}");

    // With no API key and no control-plane authentication, the admin and
    // worker-management routes are closed: a certificate identity is a tenant
    // credential, which never reaches them.
    let (status, response) = https(&with_identity, addr, "GET", "/workers", None)
        .unwrap_or_else(|e| panic!("{e}\n{}", gateway.log()));
    assert_eq!(status, 401, "the admin routes are closed\n{response}");

    // Under TLS 1.3 the server's refusal arrives after the client's own
    // Finished: the client sees it as a failed handshake or as a connection
    // closed before any response. Either way no HTTP answer.
    let refused = https(&bare, addr, "GET", "/health", None);
    assert!(
        refused.is_err(),
        "a client without a certificate is not served: {refused:?}"
    );

    let metrics = scrape(metrics_port);
    assert!(
        client_auth_count(&metrics, "identified").is_some_and(|n| n >= 1.0),
        "identified connections are counted\n{metrics}"
    );
    assert!(
        client_auth_count(&metrics, "handshake_failed").is_some_and(|n| n >= 1.0),
        "failed handshakes are counted\n{metrics}"
    );
    assert!(
        gateway
            .log()
            .contains("TLS enabled with client-certificate authentication"),
        "the start names the mode\n{}",
        gateway.log()
    );
}

#[test]
fn an_optional_client_certificate_stands_in_for_the_api_key() {
    let _serial = serial();
    let certs = TestCertificates::generate().unwrap();
    let identity = certs.client_identity("team-red", SPIFFE).unwrap();
    let with_identity = connector(&certs, Some(&identity));
    let bare = connector(&certs, None);
    let [port, metrics_port] = free_ports();
    let addr = SocketAddr::from((IP, port));
    let log = certs.temp_dir.path().join("optional.log");
    let gateway = Gateway::start(
        &tls_args(&certs, port, metrics_port, "optional", &["--api-key", "k"]),
        addr,
        &bare,
        &log,
    );
    let chat = |connector: &SslConnector, authorization: Option<&str>| {
        https(
            connector,
            addr,
            "POST",
            "/v1/chat/completions",
            authorization,
        )
        .unwrap_or_else(|e| panic!("{e}\n{}", gateway.log()))
    };

    let (status, response) = chat(&bare, None);
    assert_eq!(status, 401, "no certificate and no key\n{response}");
    let (status, response) = chat(&bare, Some("Bearer k"));
    assert_ne!(status, 401, "the key still works\n{response}");
    let (status, response) = chat(&with_identity, None);
    assert_ne!(
        status, 401,
        "the certificate stands in for the key\n{response}"
    );

    let metrics = scrape(metrics_port);
    assert!(
        client_auth_count(&metrics, "none").is_some_and(|n| n >= 2.0),
        "connections without a certificate are counted\n{metrics}"
    );
    assert!(
        client_auth_count(&metrics, "identified").is_some_and(|n| n >= 1.0),
        "{metrics}"
    );
}

/// `--admin-port` under `required`: the admin listener serves TLS with the
/// server certificate alone, so a probe without a client certificate reaches
/// its health route while the data listener still turns a bare client away;
/// its connections are not counted as the data listener's; and with no API
/// key and no control-plane authentication its admin routes are closed to
/// the certificate holder as to everyone.
#[test]
fn the_admin_listener_asks_for_no_client_certificate() {
    let _serial = serial();
    let certs = TestCertificates::generate().unwrap();
    let identity = certs.client_identity("team-red", SPIFFE).unwrap();
    let with_identity = connector(&certs, Some(&identity));
    let bare = connector(&certs, None);
    let [port, metrics_port, admin_port] = free_ports();
    let addr = SocketAddr::from((IP, port));
    let admin_addr = SocketAddr::from((IP, admin_port));
    let admin_port_arg = admin_port.to_string();
    let log = certs.temp_dir.path().join("admin.log");
    let gateway = Gateway::start(
        &tls_args(
            &certs,
            port,
            metrics_port,
            "required",
            &["--admin-port", &admin_port_arg],
        ),
        addr,
        &with_identity,
        &log,
    );

    let (status, response) = https(&bare, admin_addr, "GET", "/health", None)
        .unwrap_or_else(|e| panic!("{e}\n{}", gateway.log()));
    assert_eq!(
        status, 200,
        "a probe without a certificate reaches the admin listener\n{response}"
    );
    let refused = https(&bare, addr, "GET", "/health", None);
    assert!(
        refused.is_err(),
        "the data listener still requires a certificate: {refused:?}"
    );
    let (status, response) = https(&with_identity, admin_addr, "GET", "/workers", None)
        .unwrap_or_else(|e| panic!("{e}\n{}", gateway.log()));
    assert_eq!(
        status, 401,
        "no key and no control-plane authentication: the admin routes are closed\n{response}"
    );

    let metrics = scrape(metrics_port);
    assert_eq!(
        client_auth_count(&metrics, "handshake_failed"),
        Some(1.0),
        "only the data listener's refusal is counted\n{metrics}"
    );
    assert_eq!(
        client_auth_count(&metrics, "none"),
        Some(0.0),
        "the admin listener's connections are not counted\n{metrics}"
    );
}

/// A launch that must end in a refusal: no watched port ever answered, a
/// non-zero exit, no panic, and `named` in the error.
fn assert_refused(name: &str, launch: &Launch, named: &str) {
    let log = &launch.log;
    assert!(
        launch.bound.is_none(),
        "{name}: port {:?} answered although the configuration is unusable\n{log}",
        launch.bound
    );
    let status = launch.status.unwrap_or_else(|| {
        panic!("{name}: the gateway neither exited nor listened within {EXIT_TIMEOUT:?}\n{log}")
    });
    assert!(!status.success(), "{name}: exited successfully\n{log}");
    assert!(!log.contains("panicked"), "{name}: a panic\n{log}");
    assert!(
        log.contains(named),
        "{name}: the error does not name {named:?}\n{log}"
    );
}

#[test]
fn a_mode_without_its_bundle_and_a_bundle_without_a_mode_are_refused() {
    let certs = TestCertificates::generate().unwrap();
    let dir = certs.temp_dir.path();
    let refused = launch(
        &[
            "--tls-cert-path",
            certs.server_cert_str(),
            "--tls-key-path",
            certs.server_key_str(),
            "--tls-client-auth",
            "required",
        ],
        Until::Bound,
        &dir.join("no-bundle.log"),
    );
    assert_refused(
        "required without a bundle",
        &refused,
        "--tls-client-ca-path",
    );

    let refused = launch(
        &[
            "--tls-cert-path",
            certs.server_cert_str(),
            "--tls-key-path",
            certs.server_key_str(),
            "--tls-client-ca-path",
            certs.ca_cert_str(),
        ],
        Until::Bound,
        &dir.join("no-mode.log"),
    );
    assert_refused("a bundle without a mode", &refused, "--tls-client-auth");
}
