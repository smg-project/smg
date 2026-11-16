//! The controller's listeners end to end: ephemeral ports, the health
//! service on every gRPC listener, the admin endpoints, readiness through a
//! shutdown, mutual TLS, and the refusals a bind names. Every test binds
//! `127.0.0.1:0`, so they run beside each other and beside another process.
//! The helpers return `Result`s and the tests expect them: a helper is not a
//! test to the lints.

#[path = "common/net.rs"]
mod net;

use std::{
    io::Write,
    net::{Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use net::{http_get, loopback, vm_rss_kib};
use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose,
};
use smg_policy_controller::{
    controller::{
        Config, ConfigError, Controller, Election, Error, LeasePlace, Listener, Scope, State, Tls,
        TlsError, SERVICE_NAME,
    },
    elector::{ElectionSettings, FileLease, LeaseBackend},
    lease::{Lease, LeaseId},
    proto::{
        controller_frame, gateway_frame, policy_controller_client::PolicyControllerClient,
        GatewayFrame, GetPolicySpecRequest, Hello, LeaderRequest, WatchWorkersRequest,
    },
    store::{
        GatewayId, LeaseRecord, LeaseState, LeasesStore, LoadSink, LoadSource, WorkerFacts,
        WorkersStore,
    },
    view::{fingerprint, LoadRow, Role, WorkerId},
};
use tokio::{
    net::TcpStream,
    sync::{mpsc, oneshot},
};
use tonic::{
    codegen::tokio_stream::wrappers::ReceiverStream,
    metadata::AsciiMetadataValue,
    transport::{Certificate, Channel, ClientTlsConfig, Identity},
    Code, Request,
};
use tonic_health::pb::{
    health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
};

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
        load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
        election: Election::default(),
    }
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
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
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
        assert_eq!(
            (status, body.as_str()),
            (200, r#"{"status":"ready","role":"leader"}"#)
        );
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
            body.contains("smg_policy_controller_state{state=\"leader\"} 1"),
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

/// `message` as a call that states `name`, or states none.
fn stating<T>(name: Option<&'static str>, message: T) -> Request<T> {
    use smg_policy_controller::controller::PRINCIPAL_ENTRY;

    let mut request = Request::new(message);
    if let Some(name) = name {
        request
            .metadata_mut()
            .insert(PRINCIPAL_ENTRY, AsciiMetadataValue::from_static(name));
    }
    request
}

/// A listener its operator named takes the name a call states, and no
/// step serves the call yet. The other listeners take none: a call of
/// theirs that needs a principal is refused whatever it states. `Leader`
/// asks for no principal on either.
#[tokio::test]
async fn a_listener_named_as_told_takes_the_stated_name_and_the_others_take_none() {
    let mut controller = Controller::bind(config()).await.expect("bind");
    assert_eq!(
        controller.identity_as_told("nowhere"),
        Err(ConfigError::UnknownListener("nowhere".to_owned()))
    );
    controller
        .identity_as_told("control")
        .expect("a listener of the configuration");
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
        let dial = |scope| {
            let addr = addresses.of(scope).expect("every scope is mounted");
            PolicyControllerClient::connect(format!("http://{addr}"))
        };
        let mut control = dial(Scope::Control).await.expect("the listener accepts");
        let mut gateway = dial(Scope::Gateway).await.expect("the listener accepts");

        let read = |name| stating(name, GetPolicySpecRequest::default());
        let told = control
            .get_policy_spec(read(Some("operator-a")))
            .await
            .expect_err("no step serves the call");
        assert_eq!(told.code(), Code::Unimplemented, "{told:?}");
        let silent = control
            .get_policy_spec(read(None))
            .await
            .expect_err("no name is stated");
        assert_eq!(silent.code(), Code::Unauthenticated, "{silent:?}");
        // Two names a hop joined into one value: no name.
        let joined = control
            .get_policy_spec(read(Some("operator-a,operator-b")))
            .await
            .expect_err("a comma is two names");
        assert_eq!(joined.code(), Code::Unauthenticated, "{joined:?}");

        let watch = stating(Some("gateway-a"), WatchWorkersRequest::default());
        let unread = gateway
            .watch_workers(watch)
            .await
            .expect_err("the listener takes no identity as told");
        assert_eq!(unread.code(), Code::Unauthenticated, "{unread:?}");

        for client in [&mut control, &mut gateway] {
            let info = client
                .leader(LeaderRequest::default())
                .await
                .expect("Leader asks for no principal")
                .into_inner();
            assert!(info.is_leader, "{info:?}");
        }

        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");
}

/// A gateway's session through the bound controller. It is welcomed under
/// the leader's term, and the run loop's sweep lets it go a grace period
/// after its stream's end: the lease it held is released then, not before.
#[tokio::test]
async fn a_gateway_gone_past_its_grace_period_is_let_go_by_the_run_loop() {
    const GATEWAY: &str = "gateway-a";
    let mut controller = Controller::bind(config()).await.expect("bind");
    controller
        .identity_as_told("gateway")
        .expect("a listener of the configuration");
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();
    let store = controller.store();
    // The store keys a gateway by its subject's fingerprint.
    let gateway = GatewayId(fingerprint(GATEWAY.as_bytes()));

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
        let addr = addresses
            .of(Scope::Gateway)
            .expect("every scope is mounted");
        let mut client = PolicyControllerClient::connect(format!("http://{addr}"))
            .await
            .expect("the listener accepts");
        let (frames, sent) = mpsc::channel(4);
        let mut answers = client
            .session(stating(Some(GATEWAY), ReceiverStream::new(sent)))
            .await
            .expect("the stream opens")
            .into_inner();
        let hello = Hello {
            gateway_id: GATEWAY.to_owned(),
            protocol_min: 1,
            protocol_max: 1,
            ..Hello::default()
        };
        frames
            .send(GatewayFrame {
                seq: 0,
                term_floor: 0,
                frame: Some(gateway_frame::Frame::Hello(hello)),
            })
            .await
            .expect("the stream takes the Hello");
        let answer = answers
            .message()
            .await
            .expect("a frame, not a status")
            .expect("a frame, not the stream's end");
        assert_eq!(answer.term, 1, "the first take's term: {answer:?}");
        let grace = match answer.frame {
            Some(controller_frame::Frame::Welcome(welcome)) => {
                Duration::from_millis(u64::from(welcome.grace_period_ms))
            }
            other => panic!("a Welcome answers a Hello: {other:?}"),
        };

        // A lease the gateway holds, as a decision records it.
        let now = Instant::now();
        store
            .issue(LeaseRecord {
                lease: Lease {
                    id: LeaseId(1),
                    ..Lease::default()
                },
                gateway,
                request: 1,
                target: WorkerId(7),
                decode: None,
                term: 1,
                issued_at: now,
                dispatch_deadline: now + Duration::from_secs(600),
                state: LeaseState::Issued,
            })
            .expect("a lease under the leader's term");
        assert_eq!(store.live(gateway), 1);

        // The stream ends. The gateway stays present for the grace period,
        // then the sweep releases what it held.
        drop(frames);
        drop(answers);
        let ended = Instant::now();
        while store.live(gateway) > 0 {
            assert!(
                ended.elapsed() < grace + Duration::from_secs(20),
                "the lease stands {:?} after the stream's end",
                ended.elapsed()
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            ended.elapsed() >= grace,
            "let go {:?} after the stream's end, inside the grace period of {grace:?}",
            ended.elapsed()
        );

        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");
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
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
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

/// A controller that cannot read its lease is not ready: its probes answer
/// starting and `Leader` names nobody, however long it runs. Once the lease
/// reads, the next step takes it and the probes turn ready. A rollout stops
/// at a replica that cannot reach its lease.
#[tokio::test]
async fn a_controller_that_cannot_read_its_lease_is_not_ready_until_it_can() {
    let dir = tempfile::tempdir().expect("a temp dir");
    // A directory where the lease file belongs: every look fails.
    let lease = dir.path().join("lease");
    std::fs::create_dir(&lease).expect("a directory in the lease's place");
    let mut config = config();
    // No settle window: the stop's release is all the serve waits for.
    config.shutdown_settle = Duration::ZERO;
    config.election = Election {
        lease: LeasePlace::File(lease.clone()),
        settings: ElectionSettings {
            duration: Duration::from_millis(600),
            renew_period: Duration::from_millis(50),
            renew_deadline: Duration::from_millis(400),
        },
        advertise: None,
    };
    let controller = Controller::bind(config).await.expect("bind");
    let addresses = controller.addresses();
    let gateway = addresses.of(Scope::Gateway).expect("the gateway listener");
    let fence = controller.fence();
    let mut readiness = controller.readiness();

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        // Ten renew periods of failed looks: the listeners answer, and the
        // controller is not ready.
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(readiness.state(), State::Starting);
        let (status, body) = http_get(addresses.admin, "/readiness").await.expect("GET");
        assert_eq!(
            (status, body.as_str()),
            (503, r#"{"status":"not ready","reason":"starting"}"#)
        );
        assert_eq!(health(gateway, "").await, Ok(ServingStatus::NotServing));
        assert_eq!(fence.leader_term(Instant::now()), None);
        let mut client = PolicyControllerClient::connect(format!("http://{gateway}"))
            .await
            .expect("the gateway listener accepts");
        let nobody = client
            .leader(LeaderRequest::default())
            .await
            .expect("Leader is served while starting")
            .into_inner();
        assert!(!nobody.is_leader);
        assert_eq!(
            (nobody.controller_id.as_str(), nobody.address.as_str()),
            ("", "")
        );

        // The lease reads again: the next step takes it.
        std::fs::remove_dir(&lease).expect("the directory goes");
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
        assert_eq!(fence.leader_term(Instant::now()), Some(1));
        assert_eq!(health(gateway, "").await, Ok(ServingStatus::Serving));
        let leader = client
            .leader(LeaderRequest::default())
            .await
            .expect("Leader is served")
            .into_inner();
        assert!(leader.is_leader);
        assert_eq!(leader.controller_id, "test-controller");
        assert_eq!(leader.address, gateway.to_string());
        assert_eq!(leader.term, 1);

        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");

    // The stop released the lease: the file names nobody, under the term.
    let record = FileLease::new(&lease)
        .read()
        .expect("the lease file reads")
        .expect("the lease file holds a record");
    assert!(record.is_released(), "{record:?}");
    assert_eq!(record.term, 1);
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
    let mut controller = Controller::bind(config).await.expect("bind");
    // A listener has one credential rule: the one that verifies its callers
    // against a client CA takes no identity as told.
    assert_eq!(
        controller.identity_as_told("gateway"),
        Err(ConfigError::AsToldBesideClientCa("gateway".to_owned()))
    );
    // The rule is that listener's own: the ones beside it verify no caller
    // and are named.
    for beside in ["control", "replication"] {
        controller
            .identity_as_told(beside)
            .expect("a listener with no client CA");
    }
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
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));

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
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
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
