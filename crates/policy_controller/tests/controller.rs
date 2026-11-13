//! The controller's listeners end to end: ephemeral ports, the health
//! service on every gRPC listener, the admin endpoints, readiness through a
//! shutdown, mutual TLS, and the refusals a bind names. Every test binds
//! `127.0.0.1:0`, so they run beside each other and beside another process.
//! The helpers return `Result`s and the tests expect them: a helper is not a
//! test to the lints.

use std::{
    io::Write,
    net::{Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose,
};
use smg_policy_controller::{
    controller::{
        Config, ConfigError, Controller, Error, Listener, Scope, State, Tls, TlsError, SERVICE_NAME,
    },
    store::{LoadSink, LoadSource, WorkerFacts, WorkersStore},
    view::{LoadRow, Role},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    sync::oneshot,
};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Identity};
use tonic_health::pb::{
    health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
};

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

fn listener(name: &str, scopes: &[Scope]) -> Listener {
    Listener {
        name: name.to_owned(),
        addr: loopback(),
        scopes: scopes.to_vec(),
        tls: None,
    }
}

/// The binary's default shape on ephemeral loopback ports, with short
/// shutdown bounds.
fn config() -> Config {
    Config {
        controller_id: "test-controller".to_owned(),
        listeners: vec![
            listener("gateway", &[Scope::Gateway, Scope::Ingestion]),
            listener("control", &[Scope::Control]),
            listener("replication", &[Scope::Replication]),
        ],
        admin: loopback(),
        shutdown_settle: Duration::from_millis(100),
        shutdown_grace: Duration::from_secs(5),
    }
}

/// One HTTP/1.1 GET over a plain socket: the status code and the body.
async fn http_get(addr: SocketAddr, path: &str) -> Result<(u16, String), String> {
    let mut stream = TcpStream::connect(addr)
        .await
        .map_err(|error| format!("connect {addr}: {error}"))?;
    stream
        .write_all(
            format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n").as_bytes(),
        )
        .await
        .map_err(|error| error.to_string())?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .map_err(|error| error.to_string())?;
    let text = String::from_utf8_lossy(&response).into_owned();
    let status = text
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| format!("no status line in {text:?}"))?;
    let body = text.split("\r\n\r\n").nth(1).unwrap_or_default().to_owned();
    Ok((status, body))
}

/// A plaintext health check of `service` on the listener at `addr`.
async fn health(addr: SocketAddr, service: &str) -> Result<ServingStatus, String> {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .map_err(|error| error.to_string())?
        .connect()
        .await
        .map_err(|error| error.to_string())?;
    check(&mut HealthClient::new(channel), service).await
}

/// A health check of `service` over a TLS channel built from `tls`.
async fn health_over_tls(
    addr: SocketAddr,
    tls: ClientTlsConfig,
    service: &str,
) -> Result<ServingStatus, String> {
    let channel = Channel::from_shared(format!("https://{addr}"))
        .map_err(|error| error.to_string())?
        .tls_config(tls)
        .map_err(|error| error.to_string())?
        .connect()
        .await
        .map_err(|error| error.to_string())?;
    check(&mut HealthClient::new(channel), service).await
}

/// The status of `service`, or the gRPC code's name.
async fn check(client: &mut HealthClient<Channel>, service: &str) -> Result<ServingStatus, String> {
    client
        .check(HealthCheckRequest {
            service: service.to_owned(),
        })
        .await
        .map(|response| response.into_inner().status())
        .map_err(|status| format!("{:?}", status.code()))
}

#[tokio::test]
async fn the_listeners_serve_health_and_the_admin_endpoints_on_ephemeral_ports() {
    let controller = Controller::bind(config()).await.expect("bind");
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();
    assert_eq!(readiness.state(), State::Starting);

    let names: Vec<&str> = addresses
        .listeners
        .iter()
        .map(|l| l.name.as_str())
        .collect();
    assert_eq!(names, ["gateway", "control", "replication"]);
    let mut ports: Vec<u16> = addresses
        .listeners
        .iter()
        .map(|l| l.addr.port())
        .chain([addresses.admin.port()])
        .collect();
    ports.sort_unstable();
    ports.dedup();
    assert_eq!(ports.len(), 4, "four ephemeral ports, each its own");
    assert!(ports.iter().all(|port| *port != 0));
    assert_eq!(addresses.of(Scope::Ingestion), addresses.of(Scope::Gateway));
    assert_ne!(
        addresses.of(Scope::Control),
        addresses.of(Scope::Replication)
    );
    assert!(addresses.listeners.iter().all(|l| !l.tls));

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        assert_eq!(readiness.wait_for(State::Ready).await, Some(State::Ready));
        for listener in &addresses.listeners {
            assert_eq!(
                health(listener.addr, "").await,
                Ok(ServingStatus::Serving),
                "{}: the overall status is the readiness",
                listener.name
            );
            assert_eq!(
                health(listener.addr, SERVICE_NAME).await,
                Ok(ServingStatus::NotServing),
                "{}: the scopes' calls are not served yet",
                listener.name
            );
            assert_eq!(
                health(listener.addr, "smg.policy.v1.Nothing").await,
                Err("NotFound".to_owned()),
                "{}: an unknown service is not found",
                listener.name
            );
        }

        let (status, body) = http_get(addresses.admin, "/readiness").await.expect("GET");
        assert_eq!((status, body.as_str()), (200, r#"{"status":"ready"}"#));
        let (status, body) = http_get(addresses.admin, "/liveness").await.expect("GET");
        assert_eq!((status, body.as_str()), (200, "OK"));
        let (status, body) = http_get(addresses.admin, "/health").await.expect("GET");
        assert_eq!((status, body.as_str()), (200, "OK"));
        let (status, body) = http_get(addresses.admin, "/version").await.expect("GET");
        assert_eq!(status, 200);
        let version: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(version["name"], "smg-policy-controller");
        assert_eq!(version["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(version["protocol"], "smg.policy.v1");
        let (status, body) = http_get(addresses.admin, "/metrics").await.expect("GET");
        assert_eq!(status, 200);
        assert!(
            body.contains("smg_policy_controller_state{state=\"ready\"} 1"),
            "{body}"
        );
        assert!(
            body.contains("smg_policy_controller_state{state=\"starting\"} 0"),
            "{body}"
        );
        assert!(
            body.contains("smg_policy_controller_start_time_seconds "),
            "{body}"
        );
        let (status, _) = http_get(addresses.admin, "/nowhere").await.expect("GET");
        assert_eq!(status, 404);

        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");
    assert_eq!(readiness.state(), State::Draining);
}

#[tokio::test]
async fn readiness_fails_at_the_signal_and_the_listeners_settle_before_they_close() {
    let mut config = config();
    config.shutdown_settle = Duration::from_secs(1);
    config.shutdown_grace = Duration::from_secs(10);
    let controller = Controller::bind(config).await.expect("bind");
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();
    let gateway = addresses.of(Scope::Gateway).expect("the gateway listener");

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        assert_eq!(readiness.wait_for(State::Ready).await, Some(State::Ready));
        let (status, _) = http_get(addresses.admin, "/readiness").await.expect("GET");
        assert_eq!(status, 200);

        let signalled = Instant::now();
        stop.send(()).expect("the serve is listening for the stop");
        assert_eq!(
            readiness.wait_for(State::Draining).await,
            Some(State::Draining)
        );

        // Inside the settle window: not ready, still accepting, still alive.
        let (status, body) = http_get(addresses.admin, "/readiness").await.expect("GET");
        assert_eq!(
            (status, body.as_str()),
            (503, r#"{"status":"not ready","reason":"draining"}"#)
        );
        let (status, body) = http_get(addresses.admin, "/liveness").await.expect("GET");
        assert_eq!((status, body.as_str()), (200, "OK"));
        assert_eq!(health(gateway, "").await, Ok(ServingStatus::NotServing));
        let (status, body) = http_get(addresses.admin, "/metrics").await.expect("GET");
        assert_eq!(status, 200);
        assert!(
            body.contains("smg_policy_controller_state{state=\"draining\"} 1"),
            "{body}"
        );
        signalled
    };
    let (served, signalled) = tokio::join!(serve, probe);
    served.expect("a clean stop");
    let took = signalled.elapsed();
    assert!(
        took >= Duration::from_secs(1),
        "the settle window held: {took:?}"
    );
    assert!(
        took < Duration::from_secs(10),
        "the drain did not wait for the grace: {took:?}"
    );

    assert!(
        TcpStream::connect(gateway).await.is_err(),
        "the gateway listener closed"
    );
    assert!(
        TcpStream::connect(addresses.admin).await.is_err(),
        "the admin listener closed"
    );
}

#[tokio::test]
async fn a_bind_refuses_by_name_before_anything_serves() {
    let mut refused = config();
    refused.listeners[1].scopes.clear();
    assert!(matches!(
        Controller::bind(refused).await,
        Err(Error::Config(ConfigError::ListenerWithoutScope(name))) if name == "control"
    ));

    let taken = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("a port to hold");
    let held = taken.local_addr().expect("its address");
    let mut in_use = config();
    in_use.listeners[2].addr = held;
    assert!(matches!(
        Controller::bind(in_use).await,
        Err(Error::Bind { name, addr, .. }) if name == "replication" && addr == held
    ));
    let mut admin_in_use = config();
    admin_in_use.admin = held;
    assert!(matches!(
        Controller::bind(admin_in_use).await,
        Err(Error::Admin(_))
    ));

    let mut missing = config();
    missing.listeners[0].tls = Some(Tls {
        cert: PathBuf::from("/nonexistent/server.pem"),
        key: PathBuf::from("/nonexistent/server.key"),
        client_ca: None,
    });
    assert!(matches!(
        Controller::bind(missing).await,
        Err(Error::Tls { name, source: TlsError::Read { path, .. } })
            if name == "gateway" && path == Path::new("/nonexistent/server.pem")
    ));

    let dir = tempfile::tempdir().expect("a scratch directory");
    let garbage = write(
        dir.path(),
        "server.pem",
        "-----BEGIN CERTIFICATE-----\nbm90IGEgY2VydGlmaWNhdGU=\n-----END CERTIFICATE-----\n",
    )
    .expect("write");
    let key = write(
        dir.path(),
        "server.key",
        concat!(
            "-----BEGIN ",
            "PRIVATE KEY-----\nbm90IGEga2V5\n-----END PRIVATE KEY-----\n"
        ),
    )
    .expect("write");
    let mut unparsable = config();
    unparsable.listeners[0].tls = Some(Tls {
        cert: garbage,
        key,
        client_ca: None,
    });
    assert!(matches!(
        Controller::bind(unparsable).await,
        Err(Error::TlsConfig { name, .. }) if name == "gateway"
    ));
}

/// A certificate authority minted for one test, and what it issues.
struct Authority {
    issuer: Issuer<'static, KeyPair>,
    pem: String,
}

/// A certificate with its key, PEM.
struct Issued {
    cert: String,
    key: String,
}

impl Authority {
    fn mint(name: &str) -> Result<Self, rcgen::Error> {
        let key = KeyPair::generate()?;
        let mut params = CertificateParams::new(Vec::<String>::new())?;
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.distinguished_name.push(DnType::CommonName, name);
        params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        let pem = params.self_signed(&key)?.pem();
        Ok(Self {
            issuer: Issuer::new(params, key),
            pem,
        })
    }

    /// A certificate for `localhost` and `127.0.0.1` with one purpose.
    fn issue(&self, name: &str, purpose: ExtendedKeyUsagePurpose) -> Result<Issued, rcgen::Error> {
        let key = KeyPair::generate()?;
        let mut params =
            CertificateParams::new(vec!["localhost".to_owned(), "127.0.0.1".to_owned()])?;
        params.distinguished_name.push(DnType::CommonName, name);
        params.extended_key_usages = vec![purpose];
        let cert = params.signed_by(&key, &self.issuer)?;
        Ok(Issued {
            cert: cert.pem(),
            key: key.serialize_pem(),
        })
    }
}

fn write(dir: &Path, name: &str, text: &str) -> std::io::Result<PathBuf> {
    let path = dir.join(name);
    std::fs::write(&path, text)?;
    Ok(path)
}

#[tokio::test]
async fn mutual_tls_admits_the_authoritys_clients_and_refuses_the_rest() {
    let authority = Authority::mint("test authority").expect("an authority");
    let server = authority
        .issue("controller", ExtendedKeyUsagePurpose::ServerAuth)
        .expect("a server certificate");
    let gateway_client = authority
        .issue("gateway-1", ExtendedKeyUsagePurpose::ClientAuth)
        .expect("a client certificate");
    let stranger = Authority::mint("another authority")
        .expect("another authority")
        .issue("stranger", ExtendedKeyUsagePurpose::ClientAuth)
        .expect("a stranger's certificate");

    let dir = tempfile::tempdir().expect("a scratch directory");
    let ca = write(dir.path(), "ca.pem", &authority.pem).expect("write");
    let cert = write(dir.path(), "server.pem", &server.cert).expect("write");
    let key = write(dir.path(), "server.key", &server.key).expect("write");

    let mut config = config();
    config.listeners[0].tls = Some(Tls {
        cert: cert.clone(),
        key: key.clone(),
        client_ca: Some(ca),
    });
    config.listeners[1].tls = Some(Tls {
        cert,
        key,
        client_ca: None,
    });
    let controller = Controller::bind(config).await.expect("bind");
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();
    assert_eq!(
        addresses
            .listeners
            .iter()
            .map(|l| l.tls)
            .collect::<Vec<_>>(),
        [true, true, false]
    );
    let gateway = addresses.of(Scope::Gateway).expect("the gateway listener");
    let control = addresses.of(Scope::Control).expect("the control listener");
    let replication = addresses
        .of(Scope::Replication)
        .expect("the replication listener");

    let trust = || {
        ClientTlsConfig::new()
            .ca_certificate(Certificate::from_pem(authority.pem.clone()))
            .domain_name("localhost")
    };
    let identity_of = |issued: &Issued| Identity::from_pem(issued.cert.clone(), issued.key.clone());

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        assert_eq!(readiness.wait_for(State::Ready).await, Some(State::Ready));

        assert_eq!(
            health_over_tls(gateway, trust().identity(identity_of(&gateway_client)), "").await,
            Ok(ServingStatus::Serving),
            "the authority's client is admitted"
        );
        assert!(
            health_over_tls(gateway, trust(), "").await.is_err(),
            "no client certificate: the handshake is refused"
        );
        assert!(
            health_over_tls(gateway, trust().identity(identity_of(&stranger)), "")
                .await
                .is_err(),
            "another authority's client is refused"
        );
        assert!(
            health(gateway, "").await.is_err(),
            "plaintext on a TLS listener is refused"
        );

        assert_eq!(
            health_over_tls(control, trust(), "").await,
            Ok(ServingStatus::Serving),
            "server-TLS alone asks the client for nothing"
        );
        assert_eq!(
            health(replication, "").await,
            Ok(ServingStatus::Serving),
            "a plaintext listener stays plaintext"
        );
        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");
}

/// `VmRSS` of this process, in KiB; `0` where `/proc` has no answer.
fn vm_rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("VmRSS:")
                    .and_then(|rest| rest.trim().split(' ').next())
                    .and_then(|kib| kib.parse().ok())
            })
        })
        .unwrap_or(0)
}

/// The Numbers table's resident rows: the controller serving idle, then with
/// a thousand workers and their load rows in its store. A measurement, run
/// by name with `--ignored`; it prints and asserts only that the feed landed.
#[tokio::test]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn rss_idle_and_with_a_thousand_workers_in_the_store() {
    let controller = Controller::bind(config()).await.expect("bind");
    let store = controller.store();
    let mut readiness = controller.readiness();
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let feed = async {
        assert_eq!(readiness.wait_for(State::Ready).await, Some(State::Ready));
        tokio::time::sleep(Duration::from_millis(200)).await;
        let idle = vm_rss_kib();
        let now = Instant::now();
        for i in 0..1_000u32 {
            let key = format!("http://worker-{i}.fleet:8000");
            let change = store.upsert(WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                role: Role::Single,
                models: vec!["model-a".to_owned()],
                dp_size: 1,
                max_running: Some(256),
                ..WorkerFacts::default()
            });
            let id = change.id();
            store.record(
                id,
                0,
                LoadRow {
                    running: i % 64,
                    queued: i % 8,
                    max_running: 256,
                    ..LoadRow::default()
                },
                now,
                LoadSource::Stream,
            );
        }
        let (_, rows) = store.snapshot();
        assert_eq!(rows.len(), 1_000);
        let fed = vm_rss_kib();
        let mut out = std::io::stdout();
        writeln!(out, "RSS_IDLE_KIB {idle}").expect("stdout");
        writeln!(out, "RSS_1K_WORKERS_KIB {fed}").expect("stdout");
        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, feed);
    served.expect("a clean stop");
}
