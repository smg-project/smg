//! A gateway's worker follower: one task that follows the worker list of a
//! standalone controller, over that controller's worker watch, into a
//! [`MemoryStore`] and its owner's worker objects. It is no part of the
//! election, where a follower is a replica that does not lead.
//!
//! The task keeps the store as the watch tells it ([`Watcher`]). A watch may
//! reopen or a connection may fail. The same watcher keeps its place, and
//! the store keeps its last fleet until a new snapshot replaces it.
//!
//! The watcher writes through the holder of the owner's worker objects
//! ([`Objects`]), so the object of each row stands when the event that
//! told it returns. The owner follows nothing behind the watch.
//!
//! After each event the store takes at a place, the task calls the
//! function its owner handed it at the start: the task's word that the
//! fleet stands as of a place, for the owner's readiness. The task
//! reports a fleet only after the first such call.
//!
//! The watch and a decision travel apart. A decision the controller
//! answers can name a row whose event the watch has not told yet. The
//! owner passes that target over and asks again.

use std::{io, sync::Arc, time::Duration};

use smg_worker::registry::Generation;
use tokio::{
    sync::{oneshot, watch},
    task::JoinHandle,
    time::sleep,
};
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint, Identity};
use tracing::warn;

use crate::{
    service::{
        client::{Client, Error, ANSWER_WAIT, KEEPALIVE},
        watcher::{WatchStreamEnd, Watcher},
    },
    store::MemoryStore,
    worker_host::Objects,
};

const FIRST_RETRY: Duration = Duration::from_millis(250);
const MAX_RETRY: Duration = Duration::from_secs(5);

/// A PEM file of the follower's TLS dial.
#[derive(Clone, Debug)]
pub struct PemFile {
    /// The name its user set the file by, as a flag or a configuration key.
    /// A read that fails says this name.
    pub setting: &'static str,
    /// The file's path.
    pub path: String,
}

/// How a follower reaches its controller.
#[derive(Clone, Debug)]
pub struct WorkerFollowerSettings {
    /// The controller's address, as `host:port`.
    pub address: String,
    /// TLS to the controller. Without it the dial is plain.
    pub tls: Option<Tls>,
    /// The name the gateway states to the controller. The controller holds
    /// one session under a name: a newer session replaces the one before
    /// it.
    pub gateway_id: String,
}

/// The TLS files of a follower's dial.
#[derive(Clone, Debug)]
pub struct Tls {
    /// The certificate authority the controller's certificate is checked
    /// against: the one root the dial trusts.
    pub ca_cert: PemFile,
    /// The gateway's own certificate and its key, sent where they are set.
    pub identity: Option<ClientIdentity>,
}

/// A gateway's own certificate and the key of it.
#[derive(Clone, Debug)]
pub struct ClientIdentity {
    /// The certificate.
    pub cert: PemFile,
    /// The certificate's key.
    pub key: PemFile,
}

/// Why a dial of the controller failed.
#[derive(Debug, thiserror::Error)]
enum DialError {
    /// A TLS file could not be read.
    #[error("reading {setting} {path:?}")]
    File {
        setting: &'static str,
        path: String,
        #[source]
        source: io::Error,
    },
    /// The address or the TLS settings make no endpoint.
    #[error(transparent)]
    Endpoint(#[from] tonic::transport::Error),
    /// The client's own error: the dial, the session or the watch.
    #[error(transparent)]
    Client(#[from] Error),
}

/// `error` and each of its sources on one line, a colon between them: the
/// form the retry's warning shows.
fn with_sources(error: &dyn std::error::Error) -> String {
    let mut line = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        line.push_str(": ");
        line.push_str(&cause.to_string());
        source = cause.source();
    }
    line
}

async fn read(file: &PemFile) -> Result<Vec<u8>, DialError> {
    tokio::fs::read(&file.path)
        .await
        .map_err(|source| DialError::File {
            setting: file.setting,
            path: file.path.clone(),
            source,
        })
}

impl WorkerFollowerSettings {
    async fn connect(&self, address: &str) -> Result<Client, DialError> {
        let Some(files) = &self.tls else {
            return Ok(Client::connect(&format!("http://{address}"), &self.gateway_id).await?);
        };

        let authority = Certificate::from_pem(read(&files.ca_cert).await?);
        let mut tls = ClientTlsConfig::new().ca_certificate(authority);
        if let Some(identity) = &files.identity {
            let cert = read(&identity.cert).await?;
            let key = read(&identity.key).await?;
            tls = tls.identity(Identity::from_pem(cert, key));
        }
        // The watch's next event waits with no bound, so the channel is kept
        // alive as the client's own dial keeps it: a controller gone silent
        // ends the wait.
        let endpoint = Endpoint::from_shared(format!("https://{address}"))?
            .http2_keep_alive_interval(KEEPALIVE)
            .keep_alive_timeout(ANSWER_WAIT)
            .keep_alive_while_idle(true);
        if let Some(host) = endpoint.uri().host() {
            tls = tls.domain_name(host.trim_matches(['[', ']']));
        }
        Ok(Client::connect_with(endpoint.tls_config(tls)?, &self.gateway_id).await?)
    }
}

/// The gateway's side of a controller's watch: one task that keeps a store
/// as the controller tells it.
pub struct WorkerFollower {
    task: JoinHandle<()>,
    stop: oneshot::Sender<()>,
    /// Whether a first snapshot was taken and its owner called after it.
    first_snapshot: watch::Receiver<bool>,
}

impl WorkerFollower {
    /// Starts the task. The watch alone writes `store`'s rows from here,
    /// through `objects`, the holder of the owner's worker objects over
    /// that store: the object of a row stands when its event returns. The
    /// task calls `after_event` after each event the store takes at a
    /// place. The objects stand by then, so the call lays nothing: it is
    /// the word the owner's readiness waits for.
    ///
    /// It spawns the task, so it is called inside a Tokio runtime. Dropping
    /// the follower ends the task too, without the wait [`Self::stop`]
    /// makes.
    pub fn start<G: Generation>(
        settings: WorkerFollowerSettings,
        store: Arc<MemoryStore>,
        objects: Arc<Objects<G>>,
        after_event: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        let (stop, stopped) = oneshot::channel();
        let (progress, first_snapshot) = watch::channel(false);
        #[expect(
            clippy::disallowed_methods,
            reason = "shutdown signals and joins the follower task"
        )]
        let task = tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = stopped => {}
                () = run(settings, store, objects, after_event, progress) => {}
            }
        });
        Self {
            task,
            stop,
            first_snapshot,
        }
    }

    /// Whether a first snapshot was taken: the store holds the workers as of
    /// a place, and the owner was called after it. It stays true once it is,
    /// whatever the watch does after.
    pub fn has_first_snapshot(&self) -> bool {
        *self.first_snapshot.borrow()
    }

    /// Ends the task and waits for it.
    pub async fn stop(self) {
        let _ = self.stop.send(());
        if let Err(error) = self.task.await {
            warn!(%error, "Worker follower task ended");
        }
    }
}

fn next_delay(delay: Duration) -> Duration {
    (delay * 2).min(MAX_RETRY)
}

/// The leaders followed at once, one after another, before the watch holds a
/// place again. Two replicas that name each other would else be dialed in a
/// loop with no wait.
const REDIRECTS: u32 = 2;

/// The leader `error` names, where the next dial goes to it at once: a
/// replica that does not lead named one, and fewer than [`REDIRECTS`] were
/// followed since the watch last held a place.
fn named_leader(error: &DialError, followed: u32) -> Option<&str> {
    let DialError::Client(Error::NotLeader(leader)) = error else {
        return None;
    };
    (!leader.is_empty() && followed < REDIRECTS).then_some(leader.as_str())
}

async fn run<G: Generation>(
    settings: WorkerFollowerSettings,
    store: Arc<MemoryStore>,
    objects: Arc<Objects<G>>,
    after_event: impl Fn(),
    progress: watch::Sender<bool>,
) {
    let mut watcher = Watcher::new(store, objects);
    let mut delay = FIRST_RETRY;
    let mut redirect: Option<String> = None;
    // The leaders followed at once since the watch last held a place.
    let mut followed = 0;
    loop {
        let address = redirect.as_deref().unwrap_or(&settings.address);
        let error = match settings.connect(address).await {
            Ok(mut client) => follow(
                &mut client,
                &mut watcher,
                &after_event,
                &progress,
                &mut delay,
                &mut followed,
            )
            .await
            .into(),
            Err(error) => error,
        };
        if let Some(leader) = named_leader(&error, followed) {
            followed += 1;
            redirect = Some(leader.to_owned());
            continue;
        }
        warn!(address, error = %with_sources(&error), ?delay, "Retrying the controller's worker watch");
        sleep(delay).await;
        delay = next_delay(delay);
        redirect = None;
        followed = 0;
    }
}

async fn follow<G: Generation>(
    client: &mut Client,
    watcher: &mut Watcher<G>,
    after_event: &impl Fn(),
    progress: &watch::Sender<bool>,
    delay: &mut Duration,
    followed: &mut u32,
) -> Error {
    loop {
        let mut following = match watcher.open(client).await {
            Ok(following) => following,
            Err(error) => return error,
        };
        // Whether this stream brought the watcher to a place. One that ends
        // before it does is opened again after the wait, so a snapshot that
        // cannot land does not spin.
        let mut placed = false;
        loop {
            match following.next().await {
                Ok(true) => {}
                // The controller ended the stream with no error: the next
                // one goes on after the place.
                Ok(false) => break,
                Err(WatchStreamEnd::Lost(lost)) => {
                    warn!(?lost, "Lost the controller's worker watch position");
                    break;
                }
                Err(WatchStreamEnd::Client(error)) => return error,
            }
            if following.watcher().at().is_some() {
                // The store and the worker objects hold the fleet as of a
                // place. The owner is told, and only then does readiness
                // hear of a fleet.
                after_event();
                placed = true;
                *delay = FIRST_RETRY;
                *followed = 0;
                if !*progress.borrow() {
                    progress.send_replace(true);
                }
            }
        }
        if !placed {
            sleep(*delay).await;
            *delay = next_delay(*delay);
        }
    }
}

#[cfg(test)]
mod tests {
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
            proto::{self, policy_controller_client::PolicyControllerClient, worker_event::Event},
        },
        store::{MemoryStore, WorkerStatus, WorkersStore},
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

    /// A follower of `settings` over `store` and a holder of its own, and
    /// the number of times it called its owner with the worker objects
    /// standing: one object for each row the store holds, and no other.
    /// The owner lays nothing, so a call that finds a row without its
    /// object is not counted.
    fn following(
        settings: WorkerFollowerSettings,
        store: &Arc<MemoryStore>,
    ) -> (WorkerFollower, Arc<AtomicUsize>) {
        let settled = Arc::new(AtomicUsize::new(0));
        let registry = Arc::new(WorkerRegistry::new());
        let objects = Arc::new(Objects::<Members>::new(registry).with_status_followed());
        let follower = WorkerFollower::start(settings, Arc::clone(store), Arc::clone(&objects), {
            let (settled, store) = (Arc::clone(&settled), Arc::clone(store));
            move || {
                let (_, rows) = store.all_rows();
                let stand = rows.iter().all(|row| objects.object(row.id).is_some());
                if stand && objects.len() == rows.len() {
                    settled.fetch_add(1, Relaxed);
                }
            }
        });
        (follower, settled)
    }

    fn operator<T>(body: T) -> Request<T> {
        let mut request = Request::new(body);
        request
            .metadata_mut()
            .insert(PRINCIPAL_ENTRY, "test-operator".parse().unwrap());
        request
    }

    async fn register(
        control: &mut PolicyControllerClient<Channel>,
        port: u16,
    ) -> proto::WorkerRow {
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
        control: PolicyControllerClient<Channel>,
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
                let control = PolicyControllerClient::connect(format!("http://{control}"))
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
        let (follower, settled) = following(settings(socket.local_addr().unwrap()), &store);
        let progress = follower.first_snapshot.clone();
        time::sleep(Duration::from_millis(300)).await;
        assert!(!follower.has_first_snapshot());
        assert!(rows(&store).is_empty());
        assert_eq!(settled.load(Relaxed), 0);
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
        let (follower, settled) = following(settings(controller.address), &store);
        until("the first snapshot", || follower.has_first_snapshot()).await;
        assert!(settled.load(Relaxed) > 0);
        assert!(store.get(WorkerId(row.id)).is_some());

        controller.stop().await;
        time::sleep(Duration::from_millis(300)).await;
        assert!(follower.has_first_snapshot());
        assert_eq!(rows(&store), expected);
        time::timeout(Duration::from_secs(1), follower.stop())
            .await
            .unwrap();
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
}
