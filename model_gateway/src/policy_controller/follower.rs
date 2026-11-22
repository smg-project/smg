//! A single task follows the controller's workers into the policy store.
//! A watch may reopen or a connection may fail. The same watcher keeps its
//! place and the store keeps its last fleet until a new snapshot replaces it.
//!
//! After each event the store takes at a place, the task settles the
//! gateway's feed: the registry gets the rows' worker objects, and the
//! store publishes what its build still owes. The gateway reports a fleet
//! only after the first settle, so its workers stand when it turns ready.
//!
//! The store publishes a snapshot's rows as it lays them, and so it does
//! a replaced incarnation and a status change. Between such an event and
//! its settle a decision can name a row whose worker the registry does
//! not hold yet. The policy engine passes that target over and asks again.

use std::{sync::Arc, time::Duration};

use anyhow::Context as _;
use smg_policy_controller::{
    service::{
        client::{Client, Error, ANSWER_WAIT, KEEPALIVE},
        watcher::{Ended, Watcher},
    },
    store::token,
};
use tokio::{
    sync::{oneshot, watch},
    task::JoinHandle,
    time::sleep,
};
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint, Identity};
use tracing::warn;

use super::feed::Feed;
use crate::config::RouterConfig;

const FIRST_RETRY: Duration = Duration::from_millis(250);
const MAX_RETRY: Duration = Duration::from_secs(5);

/// What the follower has reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Followed {
    /// No snapshot was taken yet.
    Nothing,
    /// A snapshot was taken: the store holds the fleet as of a place, and
    /// the registry holds a worker object for each of its rows.
    Fleet,
}

pub(crate) struct Settings {
    address: String,
    ca_cert: Option<String>,
    cert: Option<String>,
    key: Option<String>,
    gateway_id: String,
}

impl Settings {
    pub(crate) fn new(address: String, config: &RouterConfig) -> Self {
        Self {
            address,
            ca_cert: config.policy_controller_tls_ca_cert.clone(),
            cert: config.policy_controller_tls_cert.clone(),
            key: config.policy_controller_tls_key.clone(),
            gateway_id: gateway_id(std::env::var("HOSTNAME").ok()),
        }
    }

    async fn connect(&self, address: &str) -> anyhow::Result<Client> {
        if self.ca_cert.is_none() && self.cert.is_none() {
            return Ok(Client::connect(&format!("http://{address}"), &self.gateway_id).await?);
        }

        let mut tls = ClientTlsConfig::new();
        if let Some(path) = &self.ca_cert {
            let pem = tokio::fs::read(path)
                .await
                .with_context(|| format!("reading --policy-controller-tls-ca-cert {path:?}"))?;
            tls = tls.ca_certificate(Certificate::from_pem(pem));
        }
        if let (Some(cert), Some(key)) = (&self.cert, &self.key) {
            let cert = tokio::fs::read(cert)
                .await
                .with_context(|| format!("reading --policy-controller-tls-cert {cert:?}"))?;
            let key = tokio::fs::read(key)
                .await
                .with_context(|| format!("reading --policy-controller-tls-key {key:?}"))?;
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

fn gateway_id(hostname: Option<String>) -> String {
    hostname
        .filter(|name| name.len() <= 253 && token(name))
        .unwrap_or_else(|| {
            format!(
                "smg-{}",
                uuid::Builder::from_random_bytes(rand::random()).into_uuid()
            )
        })
}

/// The gateway's side of a controller's watch: one task that keeps the
/// process's policy store as the controller tells it.
pub(crate) struct Follower {
    task: JoinHandle<()>,
    stop: oneshot::Sender<()>,
    followed: watch::Receiver<Followed>,
}

impl Follower {
    /// Starts the task. The watch alone writes the store of `feed` from
    /// here, and the task settles `feed` after each event the store takes.
    pub(crate) fn start(settings: Settings, feed: Feed) -> Self {
        let (stop, stopped) = oneshot::channel();
        let (progress, followed) = watch::channel(Followed::Nothing);
        #[expect(
            clippy::disallowed_methods,
            reason = "shutdown signals and joins the follower task"
        )]
        let task = tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = stopped => {}
                () = run(settings, feed, progress) => {}
            }
        });
        Self {
            task,
            stop,
            followed,
        }
    }

    /// What the follower has reached.
    pub(crate) fn followed(&self) -> Followed {
        *self.followed.borrow()
    }

    /// Ends the task and waits for it.
    pub(crate) async fn stop(self) {
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
fn named_leader(error: &anyhow::Error, followed: u32) -> Option<&str> {
    match error.downcast_ref() {
        Some(Error::NotLeader(leader)) if !leader.is_empty() && followed < REDIRECTS => {
            Some(leader)
        }
        _ => None,
    }
}

async fn run(settings: Settings, feed: Feed, progress: watch::Sender<Followed>) {
    let mut watcher = Watcher::new(Arc::clone(feed.store()));
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
                &feed,
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
        warn!(address, error = %format_args!("{error:#}"), ?delay, "Retrying the controller's worker watch");
        sleep(delay).await;
        delay = next_delay(delay);
        redirect = None;
        followed = 0;
    }
}

async fn follow(
    client: &mut Client,
    watcher: &mut Watcher,
    feed: &Feed,
    progress: &watch::Sender<Followed>,
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
                Err(Ended::Lost(lost)) => {
                    warn!(?lost, "Lost the controller's worker watch position");
                    break;
                }
                Err(Ended::Client(error)) => return error,
            }
            if following.watcher().at().is_some() {
                // The store holds the fleet as of a place. The registry
                // takes the rows' objects and the store publishes what its
                // build still owes. Only then does readiness hear of a
                // fleet.
                feed.settle();
                placed = true;
                *delay = FIRST_RETRY;
                *followed = 0;
                if *progress.borrow() == Followed::Nothing {
                    progress.send_replace(Followed::Fleet);
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
pub(crate) mod tests {
    use std::{
        net::{Ipv4Addr, SocketAddr},
        sync::Arc,
        time::{Duration, Instant},
    };

    use openssl::{
        pkey::{PKey, Private},
        x509::X509,
    };
    use smg_policy_controller::{
        engine::WorkerId,
        service::{
            controller::{Config, Controller, Election, Listener, Scope, State, PRINCIPAL_ENTRY},
            proto::{self, policy_controller_client::PolicyControllerClient, worker_event::Event},
        },
        store::{MemoryStore, WorkerStatus, WorkersStore},
    };
    use tokio::{sync::oneshot, time};
    use tonic::{transport::Channel, Request};

    use super::{
        gateway_id, named_leader, next_delay, Error, Followed, Follower, Settings, FIRST_RETRY,
        REDIRECTS,
    };
    use crate::{
        config::RouterConfig,
        policy_controller::{
            engine::{Ask, PolicyEngine},
            feed::{following, Feed},
            start,
        },
        routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BackendConnector},
    };

    const WAIT: Duration = Duration::from_secs(5);

    fn controller_config() -> Config {
        let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
        Config {
            controller_id: "follower-test".to_owned(),
            listeners: vec![
                Listener {
                    name: "gateway".to_owned(),
                    addr: bind,
                    scopes: vec![Scope::Gateway, Scope::Ingestion],
                    tls: None,
                },
                Listener {
                    name: "control".to_owned(),
                    addr: bind,
                    scopes: vec![Scope::Control],
                    tls: None,
                },
                Listener {
                    name: "replication".to_owned(),
                    addr: bind,
                    scopes: vec![Scope::Replication],
                    tls: None,
                },
            ],
            admin: bind,
            shutdown_settle: Duration::ZERO,
            shutdown_grace: Duration::from_secs(1),
            load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
            decide_shards: 1,
            election: Election::default(),
            discovery: None,
        }
    }

    pub(crate) async fn bind_controller() -> Controller {
        let mut controller = Controller::bind(controller_config()).await.unwrap();
        controller.identity_as_told("gateway").unwrap();
        controller.identity_as_told("control").unwrap();
        controller
    }

    fn settings(address: SocketAddr) -> Settings {
        Settings {
            address: address.to_string(),
            ca_cert: None,
            cert: None,
            key: None,
            gateway_id: "test-gateway".to_owned(),
        }
    }

    fn url(port: u16) -> String {
        format!("http://127.0.0.1:{port}")
    }

    /// The feed of a gateway under an address over `store`, with a registry
    /// of its own.
    fn fed(store: &Arc<MemoryStore>) -> Feed {
        followed(store, &Arc::new(WorkerRegistry::new()))
    }

    /// The feed of a gateway under an address over `store`, for the workers
    /// of `registry`, with the shipped connector.
    fn followed(store: &Arc<MemoryStore>, registry: &Arc<WorkerRegistry>) -> Feed {
        following(store, registry, BackendConnector::shared())
    }

    /// The addresses of the registry's workers, in order.
    fn urls(registry: &WorkerRegistry) -> Vec<String> {
        let mut urls: Vec<_> = registry
            .get_all()
            .iter()
            .map(|worker| worker.url().to_owned())
            .collect();
        urls.sort_unstable();
        urls
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
        checks_disabled: bool,
    ) -> proto::WorkerRow {
        let event = control
            .register_worker(operator(proto::RegisterWorkerRequest {
                spec: Some(proto::WorkerSpec {
                    url: url(port),
                    wire: proto::WireMode::Http.into(),
                    role: proto::Role::Single.into(),
                    models: vec!["model".to_owned()],
                    health_policy: checks_disabled.then_some(proto::HealthPolicy {
                        disabled: true,
                        ..Default::default()
                    }),
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
        let (_, rows) = store.snapshot();
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
                let controller = bind_controller().await;
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
    async fn the_follower_takes_the_snapshot_and_follows_registrations_and_removals() {
        let mut controller = Serving::start().await;
        time::timeout(WAIT, async {
            let first = register(&mut controller.control, 31001, false).await;
            register(&mut controller.control, 31002, true).await;
            assert_eq!(rows(&controller.store).len(), 2);
            let store = Arc::new(MemoryStore::new());
            let registry = Arc::new(WorkerRegistry::new());
            let follower =
                Follower::start(settings(controller.address), followed(&store, &registry));
            until("the first snapshot", || {
                follower.followed() == Followed::Fleet
            })
            .await;
            assert_eq!(rows(&store), rows(&controller.store));
            // The fleet is reported once each of its rows has its worker.
            assert_eq!(urls(&registry), [url(31001), url(31002)]);

            register(&mut controller.control, 31003, false).await;
            until("the third registration", || rows(&store).len() == 3).await;
            assert_eq!(rows(&store), rows(&controller.store));
            assert_eq!(urls(&registry), [url(31001), url(31002), url(31003)]);

            controller
                .control
                .deregister_worker(operator(proto::DeregisterWorkerRequest {
                    worker_id: first.id,
                    revision: first.revision,
                    drain: false,
                }))
                .await
                .unwrap();
            until("the removal", || store.get(WorkerId(first.id)).is_none()).await;
            assert_eq!(rows(&store), rows(&controller.store));
            assert_eq!(urls(&registry), [url(31002), url(31003)]);
            follower.stop().await;
        })
        .await
        .expect("the fleet follows within five seconds");
        controller.stop().await;
    }

    /// A decision names a followed row once the fleet is reported, and the
    /// registry holds the worker to send to.
    #[tokio::test]
    async fn a_request_is_placed_on_a_followed_worker_once_the_fleet_is_reported() {
        let mut controller = Serving::start().await;
        time::timeout(WAIT, async {
            let row = register(&mut controller.control, 31001, true).await;
            let store = start(&RouterConfig {
                policy_controller: true,
                ..RouterConfig::default()
            })
            .unwrap()
            .expect("the controller was picked");
            let registry = Arc::new(WorkerRegistry::new());
            let feed = followed(&store, &registry);
            let policy_engine = PolicyEngine::new(store, Arc::clone(&feed.objects));
            let follower = Follower::start(settings(controller.address), feed);
            until("the first snapshot", || {
                follower.followed() == Followed::Fleet
            })
            .await;

            let placed = policy_engine
                .place(&Ask::of(b"first", "model"), Instant::now())
                .await
                .expect("a decision picks the followed worker");
            assert_eq!(placed.id(), WorkerId(row.id));
            assert_eq!(placed.worker().url(), url(31001));
            follower.stop().await;
        })
        .await
        .expect("the request is placed within five seconds");
        controller.stop().await;
    }

    /// The gateway probes no followed worker: the controller's verdict on a
    /// row is the status of the row's worker here.
    #[tokio::test]
    async fn a_followed_worker_is_sendable_only_while_its_row_is_ready() {
        let mut controller = Serving::start().await;
        time::timeout(WAIT, async {
            let waiting = register(&mut controller.control, 31001, false).await;
            let ready = register(&mut controller.control, 31002, true).await;
            let store = Arc::new(MemoryStore::new());
            let registry = Arc::new(WorkerRegistry::new());
            let follower =
                Follower::start(settings(controller.address), followed(&store, &registry));
            until("the first snapshot", || {
                follower.followed() == Followed::Fleet
            })
            .await;
            let sendable = |port| registry.get_by_url(&url(port)).unwrap().is_healthy();
            assert!(!sendable(31001));
            assert!(sendable(31002));

            // The controller's health write turns the waiting row ready.
            controller
                .store
                .set_status(WorkerId(waiting.id), WorkerStatus::Ready, waiting.revision)
                .expect("the status moves");
            until("the promotion", || sendable(31001)).await;

            // A drain takes the ready row out.
            controller
                .control
                .deregister_worker(operator(proto::DeregisterWorkerRequest {
                    worker_id: ready.id,
                    revision: ready.revision,
                    drain: true,
                }))
                .await
                .unwrap();
            until("the drain", || !sendable(31002)).await;
            follower.stop().await;
        })
        .await
        .expect("the status follows within five seconds");
        controller.stop().await;
    }

    #[tokio::test]
    async fn an_unreachable_controller_leaves_nothing_and_the_follower_stops() {
        let socket = tokio::net::TcpSocket::new_v4().unwrap();
        socket
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .unwrap();
        let store = Arc::new(MemoryStore::new());
        let follower = Follower::start(settings(socket.local_addr().unwrap()), fed(&store));
        let progress = follower.followed.clone();
        time::sleep(Duration::from_millis(300)).await;
        assert_eq!(follower.followed(), Followed::Nothing);
        assert!(rows(&store).is_empty());
        time::timeout(Duration::from_secs(1), follower.stop())
            .await
            .unwrap();
        assert!(progress.has_changed().is_err());
    }

    #[tokio::test]
    async fn a_stopped_controller_leaves_its_last_rows_and_fleet_state() {
        let mut controller = Serving::start().await;
        let row = time::timeout(WAIT, register(&mut controller.control, 31001, false))
            .await
            .unwrap();
        let expected = rows(&controller.store);
        let store = Arc::new(MemoryStore::new());
        let follower = Follower::start(settings(controller.address), fed(&store));
        until("the first snapshot", || {
            follower.followed() == Followed::Fleet
        })
        .await;
        assert!(store.get(WorkerId(row.id)).is_some());

        controller.stop().await;
        time::sleep(Duration::from_millis(300)).await;
        assert_eq!(follower.followed(), Followed::Fleet);
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
        let named = |leader: &str| anyhow::Error::from(Error::NotLeader(leader.to_owned()));
        for followed in 0..REDIRECTS {
            assert_eq!(named_leader(&named("b:30100"), followed), Some("b:30100"));
        }
        assert_eq!(named_leader(&named("b:30100"), REDIRECTS), None);
        assert_eq!(
            named_leader(&named(""), 0),
            None,
            "a replica that names none"
        );
        assert_eq!(named_leader(&anyhow::Error::from(Error::NoCredit), 0), None);
    }

    #[test]
    fn the_retry_wait_doubles_from_a_quarter_second_to_five_seconds() {
        let mut delay = FIRST_RETRY;
        for millis in [250, 500, 1000, 2000, 4000, 5000, 5000] {
            assert_eq!(delay, Duration::from_millis(millis));
            delay = next_delay(delay);
        }
    }

    fn make_certificate(
        name: &str,
        serial: u32,
        issuer: Option<(&X509, &PKey<Private>)>,
    ) -> (X509, PKey<Private>) {
        use openssl::{
            asn1::Asn1Time,
            bn::BigNum,
            ec::{EcGroup, EcKey},
            hash::MessageDigest,
            nid::Nid,
            x509::{
                extension::{BasicConstraints, SubjectAlternativeName},
                X509Name,
            },
        };

        let group = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1).unwrap();
        let key = PKey::from_ec_key(EcKey::generate(&group).unwrap()).unwrap();
        let mut subject = X509Name::builder().unwrap();
        subject.append_entry_by_nid(Nid::COMMONNAME, name).unwrap();
        let subject = subject.build();
        let mut cert = X509::builder().unwrap();
        cert.set_version(2).unwrap();
        cert.set_serial_number(&BigNum::from_u32(serial).unwrap().to_asn1_integer().unwrap())
            .unwrap();
        cert.set_not_before(&Asn1Time::days_from_now(0).unwrap())
            .unwrap();
        cert.set_not_after(&Asn1Time::days_from_now(1).unwrap())
            .unwrap();
        cert.set_subject_name(&subject).unwrap();
        cert.set_pubkey(&key).unwrap();
        if let Some((ca, ca_key)) = issuer {
            cert.set_issuer_name(ca.subject_name()).unwrap();
            let san = SubjectAlternativeName::new()
                .ip("127.0.0.1")
                .build(&cert.x509v3_context(Some(ca), None))
                .unwrap();
            cert.append_extension(san).unwrap();
            cert.sign(ca_key, MessageDigest::sha256()).unwrap();
        } else {
            cert.set_issuer_name(&subject).unwrap();
            cert.append_extension(BasicConstraints::new().critical().ca().build().unwrap())
                .unwrap();
            cert.sign(&key, MessageDigest::sha256()).unwrap();
        }
        (cert.build(), key)
    }

    #[tokio::test]
    async fn the_follower_authenticates_with_the_configured_tls_files() {
        use axum::{extract::Request, Router};
        use tokio::{net::TcpListener, task::JoinSet};
        use tonic::{
            service::Routes,
            transport::{server::TcpIncoming, Certificate, Identity, Server, ServerTlsConfig},
        };

        use crate::config::RouterConfig;

        time::timeout(WAIT, async {
            crate::server::install_crypto_provider();
            let paths = tempfile::tempdir().unwrap();
            let write = |name, pem| {
                let path = paths.path().join(name);
                std::fs::write(&path, pem).unwrap();
                path
            };
            let (ca, key) = make_certificate("controller-ca", 1, None);
            let (server, server_key) = make_certificate("controller", 2, Some((&ca, &key)));
            let (client, client_key) = make_certificate("test-gateway", 3, Some((&ca, &key)));
            let ca = ca.to_pem().unwrap();
            let socket = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
            let settings = Settings::new(
                socket.local_addr().unwrap().to_string(),
                &RouterConfig {
                    policy_controller_tls_ca_cert: Some(
                        write("ca.pem", ca.clone()).to_str().unwrap().to_owned(),
                    ),
                    policy_controller_tls_cert: Some(
                        write("gateway.pem", client.to_pem().unwrap())
                            .to_str()
                            .unwrap()
                            .to_owned(),
                    ),
                    policy_controller_tls_key: Some(
                        write(
                            "gateway-key.pem",
                            client_key.private_key_to_pem_pkcs8().unwrap(),
                        )
                        .to_str()
                        .unwrap()
                        .to_owned(),
                    ),
                    ..Default::default()
                },
            );
            let tls = ServerTlsConfig::new()
                .identity(Identity::from_pem(
                    server.to_pem().unwrap(),
                    server_key.private_key_to_pem_pkcs8().unwrap(),
                ))
                .client_ca_root(Certificate::from_pem(ca));
            let routes = Router::new().fallback(|request: Request| async move {
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
            let follower = Follower::start(settings, fed(&Arc::new(MemoryStore::new())));
            until("the first snapshot over mutual TLS", || {
                follower.followed() == Followed::Fleet
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
    #[tracing_test::traced_test]
    async fn the_follower_reopens_a_watch_and_redials_after_a_stream_error() {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

        use axum::{body::to_bytes, extract::Request, Router};
        use prost::Message as _;
        use tokio::{
            net::TcpListener,
            sync::{mpsc, Notify},
            task::JoinSet,
        };
        use tonic::{
            service::Routes,
            transport::{server::TcpIncoming, Server},
        };

        time::timeout(WAIT, async {
            let lead = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
            let leader_address = lead.local_addr().unwrap().to_string();
            let seed = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
            let seed_address = seed.local_addr().unwrap();
            let seeds = Arc::new(AtomicUsize::new(0));
            let seed_routes = Router::new().fallback({
                let seeds = seeds.clone();
                move |request: Request| {
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
                move |request: Request| {
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
            let follower =
                Follower::start(settings(seed_address), fed(&Arc::new(MemoryStore::new())));
            let mut progress = follower.followed.clone();
            assert_eq!(calls.recv().await, Some((0, 0, 1)));
            assert_eq!(calls.recv().await, Some((0, 0, 1)));
            assert_eq!(follower.followed(), Followed::Nothing);
            next.notify_one();
            assert_eq!(calls.recv().await, Some((2, 1, 1)));
            assert_eq!(calls.recv().await, Some((0, 0, 1)));
            assert_eq!(*progress.borrow_and_update(), Followed::Fleet);
            let retried = time::Instant::now();
            next.notify_one();
            assert_eq!(calls.recv().await, Some((3, 1, 2)));
            assert!(!progress.has_changed().unwrap());
            assert!(retried.elapsed() >= Duration::from_millis(750));
            assert_eq!(seeds.load(Relaxed), 4);
            assert!(logs_contain("Lost the controller's worker watch position"));
            logs_assert(|lines| {
                let retries: Vec<_> = lines
                    .iter()
                    .filter(|line| line.contains("Retrying the controller's worker watch"))
                    .collect();
                if retries.len() == 3
                    && retries[0].contains("delay=250ms")
                    && retries[1].contains("delay=250ms")
                    && retries[2].contains("delay=500ms")
                    && retries.iter().all(|line| !line.contains("address=\"\""))
                {
                    Ok(())
                } else {
                    Err(format!("unexpected retries: {retries:?}"))
                }
            });
            follower.stop().await;
        })
        .await
        .expect("the watch reopens and reconnects within five seconds");
    }

    #[test]
    fn the_gateway_states_a_hostname_or_a_generated_id() {
        assert_eq!(gateway_id(Some("edge-1".to_owned())), "edge-1");
        for hostname in [
            None,
            Some(String::new()),
            Some("two words".to_owned()),
            Some("two,words".to_owned()),
            Some("x".repeat(254)),
        ] {
            let id = gateway_id(hostname);
            let uuid = id
                .strip_prefix("smg-")
                .unwrap()
                .parse::<uuid::Uuid>()
                .unwrap();
            assert_eq!(uuid.get_version_num(), 4);
            assert!(id.bytes().all(|byte| byte.is_ascii_graphic()));
        }
    }
}
