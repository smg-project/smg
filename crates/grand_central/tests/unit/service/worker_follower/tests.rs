use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::{
        atomic::{AtomicUsize, Ordering::Relaxed},
        Arc,
    },
    time::Duration,
};

use axum::{extract::Request as HttpRequest, Router};
use rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, Issuer, KeyPair};
use smg_worker::registry::WorkerRegistry;
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot, Notify},
    task::JoinSet,
    time,
};
use tonic::{
    service::Routes,
    transport::{server::TcpIncoming, Certificate, Channel, Identity, Server, ServerTlsConfig},
    Request,
};

use super::{
    named_leader, next_delay, with_sources, ClientIdentity, DialError, Error, PemFile, Tls,
    WorkerFollower, WorkerFollowerSettings, FIRST_RETRY, REDIRECTS,
};
use crate::{
    document::operator::test_support::Logged,
    engine::view::WorkerId,
    service::{
        controller::{
            config::tests::valid, tls::install_crypto_provider, Controller, Scope, State,
            PRINCIPAL_ENTRY,
        },
        proto::{self, grand_central_client::GrandCentralClient, worker_event::Event},
    },
    store::{MemoryStore, WorkerStatus},
    worker_host::{Members, Objects},
};

const WAIT: Duration = Duration::from_secs(5);

fn settings(address: SocketAddr) -> WorkerFollowerSettings {
    WorkerFollowerSettings {
        address: address.to_string(),
        tls: None,
        gateway_id: "test-gateway".to_owned(),
    }
}

/// What a follower's owner found at its calls: the calls that found
/// the worker objects standing, one object for each row the store
/// holds and no other, and the calls that did not.
#[derive(Default)]
struct Calls {
    settled: AtomicUsize,
    missed: AtomicUsize,
}

/// A follower of `settings` over `store` and a holder of its own, and
/// what its owner found at each call. The owner lays nothing, so a
/// call that finds a row without its object is a miss.
fn following(
    settings: WorkerFollowerSettings,
    store: &Arc<MemoryStore>,
) -> (WorkerFollower, Arc<Calls>) {
    let calls = Arc::new(Calls::default());
    let registry = Arc::new(WorkerRegistry::new());
    let objects = Arc::new(Objects::<Members>::new(registry));
    let follower = WorkerFollower::start(settings, Arc::clone(store), Arc::clone(&objects), {
        let (calls, store) = (Arc::clone(&calls), Arc::clone(store));
        move || {
            let (_, rows) = store.all_rows();
            let stand = rows.iter().all(|row| objects.object(row.id).is_some());
            let found = if stand && objects.len() == rows.len() {
                &calls.settled
            } else {
                &calls.missed
            };
            found.fetch_add(1, Relaxed);
        }
    });
    (follower, calls)
}

fn operator<T>(body: T) -> Request<T> {
    let mut request = Request::new(body);
    request
        .metadata_mut()
        .insert(PRINCIPAL_ENTRY, "test-operator".parse().unwrap());
    request
}

async fn register(control: &mut GrandCentralClient<Channel>, port: u16) -> proto::WorkerRow {
    let event = control
        .register_worker(operator(proto::RegisterWorkerRequest {
            spec: Some(proto::WorkerSpec {
                url: format!("http://127.0.0.1:{port}"),
                wire: proto::WireMode::Http.into(),
                role: proto::Role::Single.into(),
                models: vec!["model".to_owned()],
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();
    let Some(Event::Registered(row)) = event.event else {
        panic!("registration answered {event:?}");
    };
    row
}

fn rows(store: &MemoryStore) -> Vec<(WorkerId, String, WorkerStatus)> {
    let (_, rows) = store.all_rows();
    let mut rows: Vec<_> = rows
        .into_iter()
        .map(|row| (row.id, row.facts.key, row.status))
        .collect();
    rows.sort_unstable_by_key(|(id, _, _)| id.0);
    rows
}

async fn until(what: &str, mut reached: impl FnMut() -> bool) {
    time::timeout(WAIT, async {
        while !reached() {
            time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
}

/// A controller that leads, served until it is stopped.
struct Serving {
    address: SocketAddr,
    control: GrandCentralClient<Channel>,
    store: Arc<MemoryStore>,
    stop: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
}

impl Serving {
    async fn start() -> Self {
        time::timeout(WAIT, async {
            let mut controller = Controller::bind(valid()).await.unwrap();
            controller.identity_as_told("gateway").unwrap();
            controller.identity_as_told("control").unwrap();
            let address = controller.addresses().of(Scope::Gateway).unwrap();
            let control = controller.addresses().of(Scope::Control).unwrap();
            let store = controller.store();
            let mut readiness = controller.readiness();
            let (stop, stopped) = oneshot::channel();
            #[expect(
                clippy::disallowed_methods,
                reason = "the fixture stops and joins the controller"
            )]
            let task = tokio::spawn(async move {
                controller
                    .serve(async {
                        let _ = stopped.await;
                    })
                    .await
                    .unwrap();
            });
            assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
            let control = GrandCentralClient::connect(format!("http://{control}"))
                .await
                .unwrap();
            Self {
                address,
                control,
                store,
                stop,
                task,
            }
        })
        .await
        .expect("controller starts within five seconds")
    }

    async fn stop(self) {
        self.stop.send(()).unwrap();
        time::timeout(WAIT, self.task).await.unwrap().unwrap();
    }
}

#[tokio::test]
async fn an_unreachable_controller_leaves_nothing_and_the_follower_stops() {
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket
        .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .unwrap();
    let store = Arc::new(MemoryStore::new());
    let (follower, calls) = following(settings(socket.local_addr().unwrap()), &store);
    let progress = follower.first_snapshot.clone();
    time::sleep(Duration::from_millis(300)).await;
    assert!(!follower.has_first_snapshot());
    assert!(rows(&store).is_empty());
    let called = calls.settled.load(Relaxed) + calls.missed.load(Relaxed);
    assert_eq!(called, 0, "no call at all");
    time::timeout(Duration::from_secs(1), follower.stop())
        .await
        .unwrap();
    assert!(progress.has_changed().is_err());
}

/// The store keeps the fleet of a controller that stopped, and the
/// follower still reports it. The owner was called before the fleet was
/// reported, and found the fleet's worker objects standing: it lays
/// none itself.
#[tokio::test]
async fn a_stopped_controller_leaves_its_last_rows_and_fleet_state() {
    let mut controller = Serving::start().await;
    let row = time::timeout(WAIT, register(&mut controller.control, 31001))
        .await
        .unwrap();
    let expected = rows(&controller.store);
    let store = Arc::new(MemoryStore::new());
    let (follower, calls) = following(settings(controller.address), &store);
    until("the first snapshot", || follower.has_first_snapshot()).await;
    assert!(calls.settled.load(Relaxed) > 0);
    assert!(store.get(WorkerId(row.id)).is_some());

    controller.stop().await;
    time::sleep(Duration::from_millis(300)).await;
    assert!(follower.has_first_snapshot());
    assert_eq!(rows(&store), expected);
    time::timeout(Duration::from_secs(1), follower.stop())
        .await
        .unwrap();
    assert_eq!(calls.missed.load(Relaxed), 0, "each call found them");
}

/// A leader a replica names is dialed at once, and so is the leader that
/// one names. A third in a row waits: replicas that name each other
/// are not dialed in a loop.
#[test]
fn a_named_leader_is_followed_at_once_twice_and_then_waited_for() {
    let named = |leader: &str| DialError::from(Error::NotLeader(leader.to_owned()));
    for followed in 0..REDIRECTS {
        assert_eq!(named_leader(&named("b:30100"), followed), Some("b:30100"));
    }
    assert_eq!(named_leader(&named("b:30100"), REDIRECTS), None);
    assert_eq!(
        named_leader(&named(""), 0),
        None,
        "a replica that names none"
    );
    assert_eq!(named_leader(&DialError::from(Error::NoCredit), 0), None);
}

#[test]
fn the_retry_wait_doubles_from_a_quarter_second_to_five_seconds() {
    let mut delay = FIRST_RETRY;
    for millis in [250, 500, 1000, 2000, 4000, 5000, 5000] {
        assert_eq!(delay, Duration::from_millis(millis));
        delay = next_delay(delay);
    }
}

/// A TLS file that cannot be read fails the dial under the name its user
/// set it by, with the system's reason behind it: the line the retry's
/// warning shows.
#[tokio::test]
async fn a_file_that_cannot_be_read_is_reported_under_its_setting() {
    let missing = tempfile::tempdir().unwrap().path().join("ca.pem");
    let path = missing.to_str().unwrap().to_owned();
    let settings = WorkerFollowerSettings {
        tls: Some(Tls {
            ca_cert: PemFile {
                setting: "--tls-ca-cert",
                path: path.clone(),
            },
            identity: None,
        }),
        ..settings(SocketAddr::from((Ipv4Addr::LOCALHOST, 1)))
    };
    let Err(error) = settings.connect(&settings.address).await else {
        panic!("the dial read a file that is not there");
    };
    let line = with_sources(&error);
    let reason = std::fs::read(&missing).unwrap_err();
    assert_eq!(line, format!("reading --tls-ca-cert {path:?}: {reason}"));
}

/// An authority, with the PEM of its certificate.
struct Authority {
    issuer: Issuer<'static, KeyPair>,
    pem: String,
}

impl Authority {
    fn mint(name: &str) -> Self {
        let key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::new(Vec::<String>::new()).unwrap();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.distinguished_name.push(DnType::CommonName, name);
        let pem = params.self_signed(&key).unwrap().pem();
        Self {
            issuer: Issuer::new(params, key),
            pem,
        }
    }

    /// A certificate for `127.0.0.1` and its key, as PEM.
    fn issue(&self, name: &str) -> (String, String) {
        let key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::new(vec!["127.0.0.1".to_owned()]).unwrap();
        params.distinguished_name.push(DnType::CommonName, name);
        let cert = params.signed_by(&key, &self.issuer).unwrap();
        (cert.pem(), key.serialize_pem())
    }
}

#[tokio::test]
async fn the_follower_authenticates_with_the_configured_tls_files() {
    time::timeout(WAIT, async {
        install_crypto_provider();
        let paths = tempfile::tempdir().unwrap();
        let file = |setting, name, pem: &str| {
            let path = paths.path().join(name);
            std::fs::write(&path, pem).unwrap();
            PemFile {
                setting,
                path: path.to_str().unwrap().to_owned(),
            }
        };
        let authority = Authority::mint("controller-ca");
        let (server, server_key) = authority.issue("controller");
        let (client, client_key) = authority.issue("test-gateway");
        let socket = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let settings = WorkerFollowerSettings {
            tls: Some(Tls {
                ca_cert: file("--tls-ca-cert", "ca.pem", &authority.pem),
                identity: Some(ClientIdentity {
                    cert: file("--tls-cert", "gateway.pem", &client),
                    key: file("--tls-key", "gateway-key.pem", &client_key),
                }),
            }),
            ..settings(socket.local_addr().unwrap())
        };
        let tls = ServerTlsConfig::new()
            .identity(Identity::from_pem(server, server_key))
            .client_ca_root(Certificate::from_pem(&authority.pem));
        let routes = Router::new().fallback(|request: HttpRequest| async move {
            let path = request.uri().path();
            if path.ends_with("/Leader") {
                grpc(
                    [proto::LeaderInfo {
                        term: 1,
                        ..Default::default()
                    }],
                    "0",
                )
            } else if path.ends_with("/Session") {
                grpc([welcome()], "0")
            } else {
                assert!(path.ends_with("/WatchWorkers"));
                grpc([snapshot(1, true)], "")
            }
        });
        let mut serving = JoinSet::new();
        serving.spawn(
            Server::builder()
                .tls_config(tls)
                .unwrap()
                .add_routes(Routes::from(routes))
                .serve_with_incoming(TcpIncoming::from(socket)),
        );
        let (follower, _) = following(settings, &Arc::new(MemoryStore::new()));
        until("the first snapshot over mutual TLS", || {
            follower.has_first_snapshot()
        })
        .await;
        follower.stop().await;
    })
    .await
    .expect("the follower takes a snapshot over mutual TLS within five seconds");
}

fn grpc<T: prost::Message>(
    messages: impl IntoIterator<Item = T>,
    status: &'static str,
) -> axum::response::Response {
    use std::convert::Infallible;

    use axum::body::{Body, Bytes};
    use http::{HeaderMap, HeaderValue};
    use http_body::Frame;
    use http_body_util::StreamBody;
    use tokio_stream::StreamExt as _;

    let mut bytes = Vec::new();
    for message in messages {
        bytes.push(0);
        bytes.extend_from_slice(&u32::try_from(message.encoded_len()).unwrap().to_be_bytes());
        message.encode(&mut bytes).unwrap();
    }
    let data = Ok::<_, Infallible>(Frame::data(Bytes::from(bytes)));
    let body = if status.is_empty() {
        Body::new(StreamBody::new(
            tokio_stream::once(data).chain(tokio_stream::pending()),
        ))
    } else {
        let mut trailers = HeaderMap::new();
        trailers.insert("grpc-status", HeaderValue::from_static(status));
        Body::new(StreamBody::new(tokio_stream::iter([
            data,
            Ok(Frame::trailers(trailers)),
        ])))
    };
    axum::response::Response::builder()
        .header("content-type", "application/grpc")
        .body(body)
        .unwrap()
}

fn snapshot(seq: u64, last: bool) -> proto::WorkerEvent {
    proto::WorkerEvent {
        seq,
        term: 1,
        event: Some(Event::Snapshot(proto::WorkerSnapshot {
            last,
            ..Default::default()
        })),
        ..Default::default()
    }
}

fn welcome() -> proto::ControllerFrame {
    proto::ControllerFrame {
        term: 1,
        frame: Some(proto::controller_frame::Frame::Welcome(
            proto::Welcome::default(),
        )),
        ..Default::default()
    }
}

#[tokio::test]
async fn the_follower_reopens_a_watch_and_redials_after_a_stream_error() {
    use axum::body::to_bytes;
    use prost::Message as _;

    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    // A second subscriber stays registered beside it: with one alone a
    // line is on or off by the thread that reaches it first.
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    time::timeout(WAIT, async {
        let lead = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let leader_address = lead.local_addr().unwrap().to_string();
        let seed = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let seed_address = seed.local_addr().unwrap();
        let seeds = Arc::new(AtomicUsize::new(0));
        let seed_routes = Router::new().fallback({
            let seeds = seeds.clone();
            move |request: HttpRequest| {
                let leader_address = leader_address.clone();
                let seeds = seeds.clone();
                async move {
                    if request.uri().path().ends_with("/Leader") {
                        return grpc(
                            [proto::LeaderInfo {
                                term: 1,
                                ..Default::default()
                            }],
                            "0",
                        );
                    }
                    assert!(request.uri().path().ends_with("/Session"));
                    let attempt = seeds.fetch_add(1, Relaxed);
                    grpc(
                        [proto::ControllerFrame {
                            term: 1,
                            frame: Some(proto::controller_frame::Frame::NotLeader(
                                proto::NotLeader {
                                    leader_address: if matches!(attempt, 0 | 2) {
                                        String::new()
                                    } else {
                                        leader_address
                                    },
                                    ..Default::default()
                                },
                            )),
                            ..Default::default()
                        }],
                        "0",
                    )
                }
            }
        });
        let (told, mut calls) = mpsc::channel(8);
        let next = Arc::new(Notify::new());
        let leader_routes = Router::new().fallback({
            let next = next.clone();
            let sessions = Arc::new(AtomicUsize::new(0));
            let watches = Arc::new(AtomicUsize::new(0));
            move |request: HttpRequest| {
                let (sessions, watches, told, next) = (
                    sessions.clone(),
                    watches.clone(),
                    told.clone(),
                    next.clone(),
                );
                async move {
                    let path = request.uri().path();
                    if path.ends_with("/Leader") {
                        return grpc(
                            [proto::LeaderInfo {
                                term: 1,
                                ..Default::default()
                            }],
                            "0",
                        );
                    }
                    if path.ends_with("/Session") {
                        sessions.fetch_add(1, Relaxed);
                        return grpc([welcome()], "0");
                    }
                    assert!(path.ends_with("/WatchWorkers"));
                    let body = to_bytes(request.into_body(), 4096).await.unwrap();
                    let watch = proto::WatchWorkersRequest::decode(&body[5..]).unwrap();
                    let turn = watches.fetch_add(1, Relaxed);
                    told.send((watch.resume_seq, watch.resume_term, sessions.load(Relaxed)))
                        .await
                        .unwrap();
                    match turn {
                        0 => grpc([snapshot(1, false)], "0"),
                        1 => {
                            next.notified().await;
                            grpc([snapshot(2, true)], "0")
                        }
                        2 => grpc(
                            [
                                proto::WorkerEvent {
                                    event: Some(Event::SnapshotRequired(
                                        proto::SnapshotRequired::default(),
                                    )),
                                    ..Default::default()
                                },
                                snapshot(9, true),
                            ],
                            "",
                        ),
                        3 => {
                            next.notified().await;
                            grpc([snapshot(3, true)], "14")
                        }
                        _ => std::future::pending().await,
                    }
                }
            }
        });
        let mut servers = JoinSet::new();
        for (socket, routes) in [(seed, seed_routes), (lead, leader_routes)] {
            servers.spawn(
                Server::builder()
                    .add_routes(Routes::from(routes))
                    .serve_with_incoming(TcpIncoming::from(socket)),
            );
        }
        let (follower, _) = following(settings(seed_address), &Arc::new(MemoryStore::new()));
        let mut progress = follower.first_snapshot.clone();
        assert_eq!(calls.recv().await, Some((0, 0, 1)));
        assert_eq!(calls.recv().await, Some((0, 0, 1)));
        assert!(!follower.has_first_snapshot());
        next.notify_one();
        assert_eq!(calls.recv().await, Some((2, 1, 1)));
        assert_eq!(calls.recv().await, Some((0, 0, 1)));
        assert!(*progress.borrow_and_update());
        let retried = time::Instant::now();
        next.notify_one();
        assert_eq!(calls.recv().await, Some((3, 1, 2)));
        assert!(!progress.has_changed().unwrap());
        assert!(retried.elapsed() >= Duration::from_millis(750));
        assert_eq!(seeds.load(Relaxed), 4);
        follower.stop().await;
    })
    .await
    .expect("the watch reopens and reconnects within five seconds");

    let kept = lines.0.lock();
    let text = String::from_utf8_lossy(&kept);
    assert!(text.contains("Lost the controller's worker watch position"));
    let retries: Vec<_> = text
        .lines()
        .filter(|line| line.contains("Retrying the controller's worker watch"))
        .collect();
    assert!(
        retries.len() == 3
            && retries[0].contains("delay=250ms")
            && retries[1].contains("delay=250ms")
            && retries[2].contains("delay=500ms")
            && retries.iter().all(|line| !line.contains("address=\"\"")),
        "unexpected retries: {retries:?}"
    );
}
