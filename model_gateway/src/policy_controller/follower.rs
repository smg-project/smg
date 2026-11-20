//! A single task holds the gateway's link to its controller and follows the
//! controller's workers into the policy store over it.
//!
//! The link is the gateway's one connection: its session, its decide stream
//! and the watch share one channel. A controller keeps one session a
//! gateway, so the task dials nothing beside the link. The host asks over
//! the link the task holds ([`Linked`]). The gateway keeps that one link
//! for its life. Where its session ends the task dials the link again in
//! place, so a request in flight reports on the session that stands and
//! the controller's lease ends with the request. A watch that fails on a
//! session that stands is opened again on it: a new dial would take the
//! session from under the requests in flight. The watcher keeps its place
//! across sessions, and the store keeps its last fleet until a new
//! snapshot replaces it.
//!
//! After each event the store takes at a place, the task settles the
//! gateway's feed: the registry gets the rows' worker objects, and the
//! store publishes what its build still owes. The gateway reports a fleet
//! only after the first settle, so its workers stand when it turns ready.
//!
//! The store publishes a snapshot's rows as it lays them, and so it does
//! a replaced incarnation and a status change. Between such an event and
//! its settle a decision can name a row whose worker the registry does
//! not hold yet. The host passes that target over and asks again.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{anyhow, Context as _};
use arc_swap::ArcSwapOption;
use smg_policy_controller::{
    client::{Error, Link, Unanswered},
    door::{Door, Request},
    handle::Handle,
    store::token,
    watcher::{Ended, Watcher},
};
use tokio::{
    sync::{oneshot, watch},
    task::JoinHandle,
    time::sleep,
};
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint, Identity};
use tracing::{debug, warn};

use super::{feed::Feed, host::NoAnswer};
use crate::config::RouterConfig;

const FIRST_RETRY: Duration = Duration::from_millis(250);
const MAX_RETRY: Duration = Duration::from_secs(5);

/// How long an ask over the link waits for its answer. Each ask states a
/// shorter time for the controller to decide in
/// ([`DECIDE_BUDGET_MICROS`](super::host::DECIDE_BUDGET_MICROS)), so an
/// answer made at that budget's end still arrives inside the wait.
const ASK_WAIT: Duration = Duration::from_millis(100);

/// What the follower has reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Followed {
    /// No snapshot was taken yet.
    Nothing,
    /// A snapshot was taken and settled: the store held the fleet as of a
    /// place, and the registry a worker object for each of its rows. A
    /// later snapshot's rows are published before their objects.
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

    /// Reads each configured TLS file once, so a path that cannot be read
    /// refuses the start by its option's name. Each dial reads the files
    /// again.
    pub(crate) fn readable(&self) -> Result<(), String> {
        for (option, path) in [
            ("--policy-controller-tls-ca-cert", &self.ca_cert),
            ("--policy-controller-tls-cert", &self.cert),
            ("--policy-controller-tls-key", &self.key),
        ] {
            if let Some(path) = path {
                std::fs::read(path)
                    .map_err(|error| format!("reading {option} {path:?}: {error}"))?;
            }
        }
        Ok(())
    }

    /// The endpoint of `address`, with the TLS files where the connection
    /// is configured with them. Each dial reads the files again, so a
    /// rotated file is picked up.
    async fn endpoint(&self, address: &str) -> anyhow::Result<Endpoint> {
        if self.ca_cert.is_none() && self.cert.is_none() {
            return Ok(Endpoint::from_shared(format!("http://{address}"))?);
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
        let endpoint = Endpoint::from_shared(format!("https://{address}"))?;
        if let Some(host) = endpoint.uri().host() {
            tls = tls.domain_name(host.trim_matches(['[', ']']));
        }
        Ok(endpoint.tls_config(tls)?)
    }

    /// The gateway's one connection to the controller at `address`: its
    /// session, its decide stream, and the channel the watch opens on. The
    /// link keeps its channel alive itself, so a controller gone silent
    /// ends it.
    async fn dial(&self, address: &str) -> anyhow::Result<Link> {
        let endpoint = self.endpoint(address).await?;
        Ok(Link::connect(endpoint, &self.gateway_id).await?)
    }

    /// The next session of `link`, opened in place at `address`. The
    /// requests in flight hold the link, so they report on that session.
    async fn redial(&self, link: &Link, address: &str) -> anyhow::Result<()> {
        let endpoint = self.endpoint(address).await?;
        Ok(link.redial(endpoint, &self.gateway_id).await?)
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

/// The link the follower's task holds, as the gateway's host reads it: none
/// before the first dial reaches a controller, and the same link from then
/// on. Between two of its sessions an ask gets no answer, and a report
/// waits for the next one. Beside it stands what the task has reached.
pub(crate) struct Linked {
    link: ArcSwapOption<Link>,
    followed: watch::Receiver<Followed>,
}

#[cfg(test)]
impl Default for Linked {
    /// A holder no task writes: no link, and no fleet.
    fn default() -> Self {
        Self::reading(watch::channel(Followed::Nothing).1)
    }
}

impl Linked {
    fn reading(followed: watch::Receiver<Followed>) -> Self {
        Self {
            link: ArcSwapOption::empty(),
            followed,
        }
    }

    /// Whether the store holds a fleet the controller told: the first whole
    /// snapshot stands. From then on the rows are the last told, and the
    /// door of this process has rows to decide over where the controller
    /// gives an ask no decision.
    pub(crate) fn fleet(&self) -> bool {
        *self.followed.borrow() == Followed::Fleet
    }

    /// One decision of the controller for `request`, asked at `now` and
    /// held ([`Handle::asked`]). It waits up to [`ASK_WAIT`].
    ///
    /// # Errors
    ///
    /// [`NoAnswer`]: no dial has reached a controller yet, or the link gave
    /// the ask no decision. A link between two sessions gives none.
    pub(crate) async fn asked(
        &self,
        request: &Request<'_>,
        now: Instant,
    ) -> Result<Handle<Arc<Door>>, NoAnswer> {
        let Some(link) = self.link.load_full() else {
            return Err(NoAnswer::NoLink);
        };
        Handle::asked(&link, request, now, ASK_WAIT)
            .await
            .map_err(|unanswered| {
                debug!(?unanswered, "The controller gave an ask no decision");
                no_answer(&unanswered)
            })
    }
}

/// The gateway's word for an ask its link left with no decision.
fn no_answer(unanswered: &Unanswered) -> NoAnswer {
    match unanswered {
        Unanswered::NoCredit => NoAnswer::NoCredit,
        Unanswered::Busy(_) => NoAnswer::Busy,
        Unanswered::Late => NoAnswer::Late,
        Unanswered::InFlight => NoAnswer::InFlight,
        Unanswered::Ended => NoAnswer::Ended,
        Unanswered::Unsent(_) => NoAnswer::Unsent,
    }
}

/// The gateway's side of a controller: one task that holds the link and
/// keeps the process's policy store as the controller's watch tells it.
pub(crate) struct Follower {
    task: JoinHandle<()>,
    stop: oneshot::Sender<()>,
    followed: watch::Receiver<Followed>,
    linked: Arc<Linked>,
}

impl Follower {
    /// Starts the task. The watch alone writes the store of `feed` from
    /// here, and the task settles `feed` after each event the store takes.
    pub(crate) fn start(settings: Settings, feed: Feed) -> Self {
        let (stop, stopped) = oneshot::channel();
        let (progress, followed) = watch::channel(Followed::Nothing);
        let linked = Arc::new(Linked::reading(followed.clone()));
        let held = Arc::clone(&linked);
        #[expect(
            clippy::disallowed_methods,
            reason = "shutdown signals and joins the follower task"
        )]
        let task = tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = stopped => {}
                () = run(settings, feed, &held, progress) => {}
            }
            // A stop drops the task where it waits. The host is left no
            // link of a task that ended.
            held.link.store(None);
        });
        Self {
            task,
            stop,
            followed,
            linked,
        }
    }

    /// The link the task holds, for the gateway's host.
    pub(crate) fn linked(&self) -> Arc<Linked> {
        Arc::clone(&self.linked)
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
    // A dial's refusal is the error itself. A link's end is its shared
    // cause.
    let cause = error.downcast_ref::<Error>().or_else(|| {
        error
            .downcast_ref::<Arc<Error>>()
            .map(|cause| cause.as_ref())
    });
    match cause {
        Some(Error::NotLeader(leader)) if !leader.is_empty() && followed < REDIRECTS => {
            Some(leader)
        }
        _ => None,
    }
}

/// Why `link` ended, as the next dial reads it.
fn ended(link: &Link) -> anyhow::Error {
    link.ended().map_or_else(
        || anyhow!("the controller's link closed"),
        anyhow::Error::new,
    )
}

async fn run(settings: Settings, feed: Feed, linked: &Linked, progress: watch::Sender<Followed>) {
    let mut watcher = Watcher::new(Arc::clone(feed.store()));
    let mut delay = FIRST_RETRY;
    let mut redirect: Option<String> = None;
    // The leaders followed at once since the watch last held a place.
    let mut followed = 0;
    loop {
        let address = redirect.as_deref().unwrap_or(&settings.address);
        // One link for the task's life. The first dial makes it, and each
        // later one opens its next session in place.
        let dialed = match linked.link.load_full() {
            Some(link) => settings.redial(&link, address).await.map(|()| link),
            None => settings.dial(address).await.map(|link| {
                let link = Arc::new(link);
                linked.link.store(Some(Arc::clone(&link)));
                link
            }),
        };
        let error = match dialed {
            Ok(link) => {
                follow(
                    &link,
                    &mut watcher,
                    &feed,
                    &progress,
                    &mut delay,
                    &mut followed,
                )
                .await
            }
            Err(error) => error,
        };
        if let Some(leader) = named_leader(&error, followed) {
            followed += 1;
            redirect = Some(leader.to_owned());
            continue;
        }
        warn!(address, error = %format_args!("{error:#}"), ?delay, "Retrying the controller's link");
        sleep(delay).await;
        delay = next_delay(delay);
        redirect = None;
        followed = 0;
    }
}

/// Follows the controller's workers over `link` until its session ends or
/// its replica stops leading, and answers why. A watch that fails on a
/// session that stands is opened again on it, after the wait.
async fn follow(
    link: &Link,
    watcher: &mut Watcher,
    feed: &Feed,
    progress: &watch::Sender<Followed>,
    delay: &mut Duration,
    followed: &mut u32,
) -> anyhow::Error {
    loop {
        let mut following = match watcher.open_on(link).await {
            Ok(following) => Some(following),
            Err(error) => match leaves(link, error) {
                Ok(why) => return why,
                Err(error) => {
                    warn!(%error, delay = ?*delay, "Opening the controller's worker watch again on its link");
                    None
                }
            },
        };
        // Whether this stream brought the watcher to a place. One that ends
        // before it does is opened again after the wait, so a snapshot that
        // cannot land does not spin. One that fails waits too.
        let mut placed = false;
        while let Some(following) = &mut following {
            // A watch can outlive its link. The link's end comes first, so
            // the task never follows a controller it can no longer ask.
            let event = tokio::select! {
                biased;
                () = link.closed() => return ended(link),
                event = following.next() => event,
            };
            match event {
                Ok(true) => {}
                // The controller ended the stream with no error: the next
                // one goes on after the place.
                Ok(false) => break,
                Err(Ended::Lost(lost)) => {
                    warn!(?lost, "Lost the controller's worker watch position");
                    break;
                }
                Err(Ended::Client(error)) => match leaves(link, error) {
                    Ok(why) => return why,
                    Err(error) => {
                        warn!(%error, delay = ?*delay, "Opening the controller's worker watch again on its link");
                        placed = false;
                        break;
                    }
                },
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
            // The wait ends with the link.
            tokio::select! {
                biased;
                () = link.closed() => return ended(link),
                () = sleep(*delay) => {}
            }
            *delay = next_delay(*delay);
        }
    }
}

/// Why a watch that failed with `error` ends the follow on `link`. Where
/// the session has ended its own cause is the answer, so a leader it names
/// is dialed at once. Where the watch's replica does not lead, the next
/// dial goes to the leader the watch names. Any other failure is the
/// watch's own and is handed back: the session stands.
fn leaves(link: &Link, error: Error) -> Result<anyhow::Error, Error> {
    if !link.stands() {
        Ok(ended(link))
    } else if matches!(error, Error::NotLeader(_)) {
        Ok(error.into())
    } else {
        Err(error)
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
        client::Link,
        controller::{Config, Controller, Election, Listener, Scope, State, PRINCIPAL_ENTRY},
        decision::ShedReason,
        lease::{LeaseId, TransportClass},
        proto::{self, policy_controller_client::PolicyControllerClient, worker_event::Event},
        store::{GatewayId, LeaseState, MemoryStore, WorkerStatus, WorkersStore},
        view::{fingerprint, WorkerId},
    };
    use tokio::{sync::oneshot, time};
    use tonic::{
        transport::{Channel, Endpoint},
        Request,
    };

    use super::{
        gateway_id, named_leader, next_delay, Error, Followed, Follower, Settings, FIRST_RETRY,
        REDIRECTS,
    };
    use crate::{
        config::RouterConfig,
        observability::metrics::test_support::render_with_recorder,
        policy_controller::{
            feed::{following, Feed},
            host::{Ask, End, Host, NoAnswer, Unplaced},
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

    /// What the gateway's metrics hold after `case`. One thread runs the
    /// case, its controller and its follower, so the thread's recorder
    /// reads what the host counts.
    fn rendered(case: impl std::future::Future<Output = ()>) -> String {
        render_with_recorder(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(case);
        })
    }

    /// Every one of `lines` stands in `rendered`: a failure names each
    /// line that does not.
    fn assert_counted(rendered: &str, lines: &[&str]) {
        let missing: Vec<_> = lines
            .iter()
            .filter(|line| !rendered.lines().any(|held| held == **line))
            .collect();
        assert!(missing.is_empty(), "missing {missing:?} in:\n{rendered}");
    }

    /// The leases a store holds live for a gateway.
    fn live(store: &MemoryStore, gateway: GatewayId) -> usize {
        smg_policy_controller::store::LeasesStore::live(store, gateway)
    }

    /// Whether the store holds `lease` of `gateway` as dispatched.
    fn dispatched(store: &MemoryStore, gateway: GatewayId, lease: LeaseId) -> bool {
        let record = smg_policy_controller::store::LeasesStore::get(store, gateway, lease);
        matches!(
            record.map(|record| record.state),
            Some(LeaseState::Dispatched { .. })
        )
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

    /// The host of a gateway under an address asks its controller. The
    /// decision and its lease are the controller's, the worker is the
    /// gateway's object of the picked row, and the request's end takes the
    /// lease off the controller's file. The decision is counted under the
    /// remote mode and no profile.
    #[test]
    fn a_linked_host_places_a_request_by_its_controllers_decision() {
        let rendered = rendered(placed_by_the_controller());
        assert_counted(
            &rendered,
            &["smg_policy_decisions_total{model=\"model\",profile=\"\",outcome=\"pick\",mode=\"remote\"} 1"],
        );
    }

    async fn placed_by_the_controller() {
        let mut controller = Serving::start().await;
        time::timeout(WAIT, async {
            let config = RouterConfig {
                policy_controller: true,
                ..RouterConfig::default()
            };
            // The controller decides by a document of its own.
            let document = config.policy_document().unwrap();
            controller
                .store
                .apply(config.hosted_policy(&document).unwrap())
                .unwrap();
            let row = register(&mut controller.control, 31001, true).await;
            let store = start(&config).unwrap().expect("the controller was picked");
            let registry = Arc::new(WorkerRegistry::new());
            let feed = followed(&store, &registry);
            let objects = Arc::clone(&feed.objects);
            let follower = Follower::start(settings(controller.address), feed);
            until("the first snapshot", || {
                follower.followed() == Followed::Fleet
            })
            .await;

            let host = Host::linked(Arc::clone(&store), objects, follower.linked());
            let now = Instant::now();
            let mut placed = host
                .place(&Ask::of(b"first", "model"), now)
                .await
                .expect("the controller picks the followed worker");
            assert_eq!(placed.id(), WorkerId(row.id));
            assert_eq!(placed.worker().url(), url(31001));
            // The lease is the controller's. This process books none.
            let gateway = GatewayId(fingerprint(b"test-gateway"));
            assert_eq!(live(&controller.store, gateway), 1);
            assert_eq!(live(&store, GatewayId::LOCAL), 0);

            placed.dispatched(now);
            placed.completed(
                End {
                    status: 200,
                    class: TransportClass::Ok,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: 0,
                },
                now,
            );
            until("the controller to end the lease", || {
                live(&controller.store, gateway) == 0
            })
            .await;
            follower.stop().await;
        })
        .await
        .expect("the request is placed and ended within five seconds");
        controller.stop().await;
    }

    /// A request can outlast the session it was placed over. The follower
    /// dials its one link again, and what a request reports is said on a
    /// session that stands: the controller's lease ends with the request.
    /// One request is dispatched before its session ends and completes on
    /// the next. The other is sent and ends between the two.
    #[tokio::test]
    async fn a_request_placed_over_one_session_ends_its_lease_on_the_next() {
        let mut controller = Serving::start().await;
        time::timeout(WAIT, async {
            let config = RouterConfig {
                policy_controller: true,
                ..RouterConfig::default()
            };
            let document = config.policy_document().unwrap();
            controller
                .store
                .apply(config.hosted_policy(&document).unwrap())
                .unwrap();
            register(&mut controller.control, 31001, true).await;
            let store = start(&config).unwrap().expect("the controller was picked");
            let registry = Arc::new(WorkerRegistry::new());
            let feed = followed(&store, &registry);
            let objects = Arc::clone(&feed.objects);
            let follower = Follower::start(settings(controller.address), feed);
            until("the first snapshot", || {
                follower.followed() == Followed::Fleet
            })
            .await;
            let linked = follower.linked();
            let host = Host::linked(Arc::clone(&store), objects, Arc::clone(&linked));
            let now = Instant::now();
            let mut sent = host
                .place(&Ask::of(b"first", "model"), now)
                .await
                .expect("the controller picks the followed worker");
            let mut waiting = host
                .place(&Ask::of(b"second", "model"), now)
                .await
                .expect("the controller picks the followed worker");
            let gateway = GatewayId(fingerprint(b"test-gateway"));
            assert_eq!(live(&controller.store, gateway), 2);
            let end = || End {
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            };

            // The first request is sent while its session stands, and the
            // controller books the dispatch.
            sent.dispatched(now);
            let lease = sent.lease().expect("a pick holds a lease");
            until("the controller to book the dispatch", || {
                dispatched(&controller.store, gateway, lease)
            })
            .await;

            // A second dial under the gateway's id takes the session from
            // under both requests.
            let link = linked
                .link
                .load_full()
                .expect("the follower holds its link");
            let endpoint = Endpoint::from_shared(format!("http://{}", controller.address)).unwrap();
            let _second = Link::connect(endpoint, "test-gateway").await.unwrap();
            until("the link's session to end", || !link.stands()).await;

            // The second request is sent and ends with no session standing:
            // its reports wait, and the controller keeps both leases.
            waiting.dispatched(now);
            waiting.completed(end(), now);
            assert_eq!(live(&controller.store, gateway), 2);
            until("the link's next session", || link.stands()).await;

            // The next session says the second request's reports, and its
            // lease ends. The first request's lease stands, dispatched: a
            // new session ends no lease by itself.
            until("the second request's lease to end", || {
                live(&controller.store, gateway) == 1
            })
            .await;
            assert!(
                dispatched(&controller.store, gateway, lease),
                "the dispatched lease stands across the dial"
            );

            // The first request completes on the next session, and no lease
            // is left.
            sent.completed(end(), now);
            until("the controller to end the first request's lease", || {
                live(&controller.store, gateway) == 0
            })
            .await;
            follower.stop().await;
        })
        .await
        .expect("the lease ends on the next session within five seconds");
        controller.stop().await;
    }

    /// A controller that went away leaves the gateway deciding. The door of
    /// this process answers in its place over the last rows it told, and
    /// the lease is this process's own. The ask is counted by the reason it
    /// had no answer, and the decision under the local mode with its
    /// profile's name.
    #[test]
    fn a_lost_controller_leaves_the_decisions_to_this_process() {
        let rendered = rendered(placed_here_behind_a_lost_controller());
        assert_counted(
            &rendered,
            &[
                "smg_policy_unanswered_total{reason=\"ended\"} 1",
                "smg_policy_decisions_total{model=\"model\",profile=\"\",outcome=\"unanswered\",mode=\"remote\"} 1",
                "smg_policy_decisions_total{model=\"model\",profile=\"random\",outcome=\"pick\",mode=\"local\"} 1",
            ],
        );
    }

    async fn placed_here_behind_a_lost_controller() {
        let mut controller = Serving::start().await;
        let row = time::timeout(WAIT, register(&mut controller.control, 31001, true))
            .await
            .unwrap();
        let store = start(&RouterConfig {
            policy_controller: true,
            ..RouterConfig::default()
        })
        .unwrap()
        .expect("the controller was picked");
        let feed = fed(&store);
        let objects = Arc::clone(&feed.objects);
        let follower = Follower::start(settings(controller.address), feed);
        until("the first snapshot", || {
            follower.followed() == Followed::Fleet
        })
        .await;
        let linked = follower.linked();
        let host = Host::linked(Arc::clone(&store), objects, Arc::clone(&linked));

        controller.stop().await;
        // The gateway keeps its link. Its session is gone with the controller.
        until("the link's session to end", || {
            linked
                .link
                .load()
                .as_ref()
                .is_some_and(|link| !link.stands())
        })
        .await;
        let now = Instant::now();
        let mut placed = time::timeout(WAIT, host.place(&Ask::of(b"first", "model"), now))
            .await
            .unwrap()
            .expect("the door of this process picks the followed worker");
        assert_eq!(placed.id(), WorkerId(row.id));
        assert_eq!(placed.worker().url(), url(31001));
        assert_eq!(live(&store, GatewayId::LOCAL), 1);

        placed.dispatched(now);
        placed.completed(
            End {
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            },
            now,
        );
        until("the lease to end", || live(&store, GatewayId::LOCAL) == 0).await;
        time::timeout(Duration::from_secs(1), follower.stop())
            .await
            .unwrap();
    }

    /// An ask the link gives no decision is decided by the door of this
    /// process whatever the reason, and counted under it: a stream that
    /// holds no credit, then a controller that does not answer in time.
    #[test]
    fn an_ask_left_with_no_decision_is_decided_here_and_counted_by_its_reason() {
        for (window, reason) in [(0, "no_credit"), (8, "late")] {
            let rendered = rendered(shed_here_under_a_silent_controller(window));
            let unanswered = format!("smg_policy_unanswered_total{{reason=\"{reason}\"}} 1");
            assert_counted(
                &rendered,
                &[
                    unanswered.as_str(),
                    "smg_policy_decisions_total{model=\"model\",profile=\"random\",outcome=\"shed\",mode=\"local\"} 1",
                ],
            );
        }
    }

    /// One attempt of a linked host under a scripted controller that tells
    /// an empty fleet, grants `window` asks and answers none. The door of
    /// this process decides, and over no row it sheds.
    async fn shed_here_under_a_silent_controller(window: u32) {
        use axum::{extract::Request, Router};
        use tokio::{net::TcpListener, task::JoinSet};
        use tonic::{
            service::Routes,
            transport::{server::TcpIncoming, Server},
        };

        time::timeout(WAIT, async {
            let socket = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
            let address = socket.local_addr().unwrap();
            let routes = Router::new().fallback(move |request: Request| async move {
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
                    grpc([granting(window)], "")
                } else if path.ends_with("/Decide") {
                    grpc::<proto::DecideAnswer>([], "")
                } else {
                    assert!(path.ends_with("/WatchWorkers"));
                    grpc([snapshot(1, true)], "")
                }
            });
            let mut serving = JoinSet::new();
            serving.spawn(
                Server::builder()
                    .add_routes(Routes::from(routes))
                    .serve_with_incoming(TcpIncoming::from(socket)),
            );
            let store = start(&RouterConfig {
                policy_controller: true,
                ..RouterConfig::default()
            })
            .unwrap()
            .expect("the controller was picked");
            let feed = fed(&store);
            let objects = Arc::clone(&feed.objects);
            let follower = Follower::start(settings(address), feed);
            until("the first snapshot", || {
                follower.followed() == Followed::Fleet
            })
            .await;

            let host = Host::linked(store, objects, follower.linked());
            let unplaced = host
                .place(&Ask::of(b"first", "model"), Instant::now())
                .await
                .err();
            assert!(
                matches!(
                    unplaced,
                    Some(Unplaced::Shed {
                        reason: ShedReason::NoEndpoints,
                        ..
                    })
                ),
                "{unplaced:?}"
            );
            follower.stop().await;
        })
        .await
        .expect("the attempt ends within five seconds");
    }

    /// A controller keeps one session a gateway. A second dial under the
    /// gateway's id takes the session, so the session of the follower's
    /// link ends. The follower dials the same link again, and its watch
    /// goes on after its place: the fleet is not told whole again.
    #[tokio::test]
    async fn a_link_the_controller_ends_is_dialed_again_and_the_watch_keeps_its_place() {
        let mut controller = Serving::start().await;
        time::timeout(WAIT, async {
            register(&mut controller.control, 31001, true).await;
            let store = Arc::new(MemoryStore::new());
            let follower = Follower::start(settings(controller.address), fed(&store));
            until("the first snapshot", || {
                follower.followed() == Followed::Fleet
            })
            .await;
            let linked = follower.linked();
            let first = linked
                .link
                .load_full()
                .expect("the follower holds its link");
            let told = store.history();

            let endpoint = Endpoint::from_shared(format!("http://{}", controller.address)).unwrap();
            let _second = Link::connect(endpoint, "test-gateway").await.unwrap();
            until("the link's session to end", || !first.stands()).await;
            until("the link's next session", || first.stands()).await;
            let held = linked
                .link
                .load_full()
                .expect("the follower holds its link");
            assert!(Arc::ptr_eq(&held, &first), "the gateway keeps its one link");

            register(&mut controller.control, 31002, true).await;
            until("the registration told on the next session", || {
                rows(&store).len() == 2
            })
            .await;
            assert_eq!(store.history(), told, "the fleet was told whole again");
            follower.stop().await;
        })
        .await
        .expect("the link is dialed again within five seconds");
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
        let feed = fed(&store);
        let objects = Arc::clone(&feed.objects);
        let follower = Follower::start(settings(socket.local_addr().unwrap()), feed);
        let progress = follower.followed.clone();
        time::sleep(Duration::from_millis(300)).await;
        assert_eq!(follower.followed(), Followed::Nothing);
        assert!(rows(&store).is_empty());
        // No link stands, so an attempt of the gateway's host has no answer.
        let host = Host::linked(Arc::clone(&store), objects, follower.linked());
        let unplaced = host
            .place(&Ask::of(b"first", "model"), Instant::now())
            .await
            .err();
        assert_eq!(unplaced, Some(Unplaced::Unanswered(NoAnswer::NoLink)));
        time::timeout(Duration::from_secs(1), follower.stop())
            .await
            .unwrap();
        assert!(progress.has_changed().is_err());
    }

    /// A link that ended because its replica stopped leading names the
    /// leader in its shared cause, and the next dial goes there at once.
    #[test]
    fn a_leader_named_at_a_links_end_is_followed_at_once() {
        let ended = anyhow::Error::new(Arc::new(Error::NotLeader("b:30100".to_owned())));
        assert_eq!(named_leader(&ended, 0), Some("b:30100"));
        assert_eq!(named_leader(&ended, REDIRECTS), None);
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
                    // The session stays open: a link stands while it does.
                    grpc([welcome()], "")
                } else if path.ends_with("/Decide") {
                    grpc::<proto::DecideAnswer>([], "")
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
        granting(0)
    }

    /// A welcome whose decide stream holds `window` asks.
    fn granting(window: u32) -> proto::ControllerFrame {
        proto::ControllerFrame {
            term: 1,
            frame: Some(proto::controller_frame::Frame::Welcome(proto::Welcome {
                decide_window: window,
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn the_follower_reopens_a_failed_watch_on_the_link_that_stands() {
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
                            // The session stays open: a link stands while it does.
                            return grpc([welcome()], "");
                        }
                        if path.ends_with("/Decide") {
                            return grpc::<proto::DecideAnswer>([], "");
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
            // The fourth watch ends in an error while the session stays
            // open. The next one opens on the same link after the wait:
            // no session more, no dial at the seed, the host's link kept.
            let held = follower.linked().link.load_full().expect("a link stands");
            let retried = time::Instant::now();
            next.notify_one();
            assert_eq!(calls.recv().await, Some((3, 1, 1)));
            assert!(!progress.has_changed().unwrap());
            assert!(retried.elapsed() >= Duration::from_millis(250));
            assert_eq!(seeds.load(Relaxed), 2);
            let now = follower.linked().link.load_full().expect("the link stands");
            assert!(Arc::ptr_eq(&held, &now));
            assert!(now.stands());
            assert!(logs_contain(
                "Opening the controller's worker watch again on its link"
            ));
            assert!(logs_contain("Lost the controller's worker watch position"));
            logs_assert(|lines| {
                let retries: Vec<_> = lines
                    .iter()
                    .filter(|line| line.contains("Retrying the controller's link"))
                    .collect();
                if retries.len() == 1
                    && retries[0].contains("delay=250ms")
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
        .expect("the watch reopens on its link within five seconds");
    }

    /// A replica can welcome a gateway and stop leading at once. Its link
    /// ends with the leader's name. A watch that fails behind that end
    /// answers the link's own cause, so the leader is dialed with no wait
    /// and the replica is not dialed again.
    #[tokio::test]
    async fn a_watch_that_fails_on_an_ended_link_follows_the_leader_the_link_names() {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

        use axum::{body::Body, extract::Request, response::Response, Router};
        use tokio::{net::TcpListener, sync::Notify, task::JoinSet};
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
            let fail = Arc::new(Notify::new());
            let seed_routes = Router::new().fallback({
                let (seeds, fail) = (seeds.clone(), fail.clone());
                move |request: Request| {
                    let leader_address = leader_address.clone();
                    let (seeds, fail) = (seeds.clone(), fail.clone());
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
                            seeds.fetch_add(1, Relaxed);
                            // Welcomed, then told of the leader: the link
                            // stands and its session ends at once.
                            let leader = proto::ControllerFrame {
                                term: 1,
                                frame: Some(proto::controller_frame::Frame::NotLeader(
                                    proto::NotLeader {
                                        leader_address,
                                        ..Default::default()
                                    },
                                )),
                                ..Default::default()
                            };
                            return grpc([welcome(), leader], "0");
                        }
                        if path.ends_with("/Decide") {
                            return grpc::<proto::DecideAnswer>([], "");
                        }
                        assert!(path.ends_with("/WatchWorkers"));
                        // The watch fails once the case saw the session's end.
                        fail.notified().await;
                        Response::builder()
                            .header("content-type", "application/grpc")
                            .header("grpc-status", "14")
                            .body(Body::empty())
                            .unwrap()
                    }
                }
            });
            let leader_routes = Router::new().fallback(|request: Request| async move {
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
                    return grpc([welcome()], "");
                }
                if path.ends_with("/Decide") {
                    return grpc::<proto::DecideAnswer>([], "");
                }
                assert!(path.ends_with("/WatchWorkers"));
                grpc([snapshot(1, true)], "")
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
            let linked = follower.linked();
            until("the replica's welcome", || linked.link.load().is_some()).await;
            let link = linked
                .link
                .load_full()
                .expect("the follower holds its link");
            until("the session's end at the replica", || !link.stands()).await;
            fail.notify_one();
            until("the leader's snapshot", || {
                follower.followed() == Followed::Fleet
            })
            .await;
            assert_eq!(seeds.load(Relaxed), 1, "the replica is dialed once");
            assert!(link.stands());
            follower.stop().await;
        })
        .await
        .expect("the named leader is followed within five seconds");
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
