//! A single task follows the controller's workers into the policy store:
//! the policy crate's worker follower ([`WorkerFollower`]), started with
//! the settings of the gateway's configuration.
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

use std::sync::Arc;

use smg_policy_controller::{
    service::worker_follower::{PemFile, WorkerFollower, WorkerFollowerSettings},
    store::token,
};

use super::feed::Feed;
use crate::config::RouterConfig;

/// What the follower has reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Followed {
    /// No snapshot was taken yet.
    Nothing,
    /// A snapshot was taken: the store holds the fleet as of a place, and
    /// the registry holds a worker object for each of its rows.
    Fleet,
}

/// How the gateway's follower reaches its controller, as the
/// configuration states it.
pub(crate) struct Settings(WorkerFollowerSettings);

impl Settings {
    pub(crate) fn new(address: String, config: &RouterConfig) -> Self {
        let file = |setting, path: Option<&String>| {
            let path = path?.clone();
            Some(PemFile { setting, path })
        };
        Self(WorkerFollowerSettings {
            address,
            ca_cert: file(
                "--policy-controller-tls-ca-cert",
                config.policy_controller_tls_ca_cert.as_ref(),
            ),
            cert: file(
                "--policy-controller-tls-cert",
                config.policy_controller_tls_cert.as_ref(),
            ),
            key: file(
                "--policy-controller-tls-key",
                config.policy_controller_tls_key.as_ref(),
            ),
            gateway_id: gateway_id(std::env::var("HOSTNAME").ok()),
        })
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

/// The gateway's side of a controller's watch: the policy crate's follower
/// over the gateway's feed.
pub(crate) struct Follower(WorkerFollower);

impl Follower {
    /// Starts the task. The watch alone writes the store of `feed` from
    /// here, and the task settles `feed` after each event the store takes.
    pub(crate) fn start(settings: Settings, feed: Feed) -> Self {
        let store = Arc::clone(feed.store());
        Self(WorkerFollower::start(settings.0, store, move || {
            feed.settle();
        }))
    }

    /// What the follower has reached.
    pub(crate) fn followed(&self) -> Followed {
        if self.0.has_first_snapshot() {
            Followed::Fleet
        } else {
            Followed::Nothing
        }
    }

    /// Ends the task and waits for it.
    pub(crate) async fn stop(self) {
        self.0.stop().await;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        net::{Ipv4Addr, SocketAddr},
        sync::Arc,
        time::{Duration, Instant},
    };

    use smg_policy_controller::{
        engine::WorkerId,
        service::{
            controller::{Config, Controller, Election, Listener, Scope, State, PRINCIPAL_ENTRY},
            proto::{self, policy_controller_client::PolicyControllerClient, worker_event::Event},
            worker_follower::{PemFile, WorkerFollowerSettings},
        },
        store::{MemoryStore, WorkerStatus, WorkersStore},
    };
    use tokio::{sync::oneshot, time};
    use tonic::{transport::Channel, Request};

    use super::{gateway_id, Followed, Follower, Settings};
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
        Settings(WorkerFollowerSettings {
            address: address.to_string(),
            ca_cert: None,
            cert: None,
            key: None,
            gateway_id: "test-gateway".to_owned(),
        })
    }

    fn url(port: u16) -> String {
        format!("http://127.0.0.1:{port}")
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

    /// Each TLS file of the configuration is named by the flag that sets
    /// it, so a file that cannot be read is reported under that flag.
    #[test]
    fn the_settings_name_each_tls_file_by_its_flag() {
        let config = RouterConfig {
            policy_controller_tls_ca_cert: Some("ca.pem".to_owned()),
            policy_controller_tls_cert: Some("gateway.pem".to_owned()),
            policy_controller_tls_key: Some("gateway-key.pem".to_owned()),
            ..RouterConfig::default()
        };
        let Settings(settings) = Settings::new("controller:30100".to_owned(), &config);
        assert_eq!(settings.address, "controller:30100");
        let named = |file: Option<PemFile>| file.map(|file| (file.setting, file.path));
        assert_eq!(
            [
                named(settings.ca_cert),
                named(settings.cert),
                named(settings.key)
            ],
            [
                Some(("--policy-controller-tls-ca-cert", "ca.pem".to_owned())),
                Some(("--policy-controller-tls-cert", "gateway.pem".to_owned())),
                Some(("--policy-controller-tls-key", "gateway-key.pem".to_owned())),
            ]
        );
        let Settings(plain) =
            Settings::new("controller:30100".to_owned(), &RouterConfig::default());
        assert!(plain.ca_cert.is_none() && plain.cert.is_none() && plain.key.is_none());
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
