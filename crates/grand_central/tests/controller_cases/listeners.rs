//! The listeners as they bind: ephemeral ports with the health service and
//! the admin endpoints, a listener's stated name, the refusals a bind names,
//! and mutual TLS.

use std::{
    net::{Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
};

use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose,
};
use smg_grand_central::service::{
    controller::{ConfigError, Controller, Error, Scope, State, Tls, TlsError, SERVICE_NAME},
    proto::{
        grand_central_client::GrandCentralClient, GetPolicySpecRequest, LeaderRequest,
        WatchWorkersRequest,
    },
};
use tokio::sync::oneshot;
use tonic::{
    transport::{Certificate, Channel, ClientTlsConfig, Identity},
    Code,
};
use tonic_health::pb::{health_check_response::ServingStatus, health_client::HealthClient};

use crate::{check, config, health, net::http_get, stating};

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
        assert_eq!(version["name"], "smg-grand-central");
        assert_eq!(version["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(version["protocol"], "smg.policy.v1");
        let (status, body) = http_get(addresses.admin, "/metrics").await.expect("GET");
        assert_eq!(status, 200);
        assert!(
            body.contains("smg_grand_central_state{state=\"leader\"} 1"),
            "{body}"
        );
        assert!(
            body.contains("smg_grand_central_state{state=\"starting\"} 0"),
            "{body}"
        );
        assert!(
            body.contains("smg_grand_central_start_time_seconds "),
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
            GrandCentralClient::connect(format!("http://{addr}"))
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
    missing.listeners[0].tls = Some(Tls::of(
        PathBuf::from("/nonexistent/server.pem"),
        PathBuf::from("/nonexistent/server.key"),
        None,
    ));
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
    unparsable.listeners[0].tls = Some(Tls::of(garbage, key, None));
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
    config.listeners[0].tls = Some(Tls::of(cert.clone(), key.clone(), Some(ca)));
    config.listeners[1].tls = Some(Tls::of(cert, key, None));
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
