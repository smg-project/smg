//! The standalone controller's run loop (design 2, 6, 8 and 10): the store,
//! the load monitor's task, the election, the gRPC listeners with their
//! scopes, the service and the health service on each, the admin listener,
//! readiness and the drain. The service answers `Leader`, a gateway's
//! `Session` and its `Decide` streams, whose decisions run on the
//! controller's deciding threads, and, on the leader, the control scope's
//! registration calls and the gateway scope's watch of the worker rows, and
//! the run loop sweeps the gateways that are gone; the scopes' other call
//! bodies arrive with the gateway and control scope PRs. In the
//! `elected` mode the gateway runs this same loop inside its process. A
//! controller given a spec file runs its document, watches the file and
//! answers `GetPolicySpec` and a dry run of `ApplyPolicySpec`
//! ([`Controller::spec_file`]). A controller given a spec store runs the
//! newest generation on file, takes its document through the apply call and
//! rolls back to a generation on file ([`Controller::spec_store`]). A host
//! that embeds the controller hands its document to the store's entry
//! itself ([`Controller::store`], then [`MemoryStore::apply`]): it names no
//! source and serves no spec call. That a controller that took a source is
//! applied through that source alone is the host's rule to keep: the entry
//! refuses a stale number and nothing else.
//!
//! A controller is bound first and served second ([`Controller::bind`], then
//! [`Controller::serve`]): the bind takes every address (port `0` for an
//! ephemeral one), reads and parses the TLS material, opens the lease's
//! place and builds the store, so a start refuses before it serves and a
//! test knows its ports; the serve runs the listeners and the election until
//! the shutdown future resolves, then gives the lease up, settles, drains
//! and returns.
//!
//! Every replica runs the same loop, the load monitor's task included. A
//! follower polls no engine, drops the load records its workers' event
//! streams carry and holds no load report: the monitor is stopped where a
//! lead ends, and its snapshot is published empty. Its first poll goes out
//! at its promotion. The store holds no load row. The controller keeps no
//! ingestion driver of its own: its loads come from this monitor and its
//! cache events from the worker host's feed.
//!
//! Every socket a listener accepts is set to send at once: a small answer
//! never waits for the peer's acknowledgement of the write before it.

pub(crate) mod admin;
pub(crate) mod config;
mod decide;
mod decision_slots;
#[cfg(feature = "discovery")]
mod discovery;
pub(crate) mod election;
mod floor;
mod frame_decision;
mod frame_loads;
mod frame_request;
mod gateway_places;
mod gateways;
mod generations;
#[cfg(test)]
#[path = "../../../tests/unit/service/controller/kv_feed/mod.rs"]
mod kv_feed;
mod lease_words;
mod lifecycle;
mod membership;
pub mod metrics;
mod principal;
mod publications;
pub(crate) mod readiness;
mod replication;
mod run;
mod service;
mod session;
mod snapshot;
mod spec_file;
pub(crate) use crate::document::operator::{spec_store, SpecRejected};
mod standby;
pub(crate) mod tls;
mod watch_workers;
pub(crate) mod wire;
mod worker_objects;
pub(crate) mod worker_wire;

use std::{
    convert::Infallible, future::Future, io, net::SocketAddr, path::PathBuf, pin::pin, sync::Arc,
};

use axum::serve::{ListenerExt as _, TapIo};
use smg_worker::load::state::LoadReceiver;
use tokio::{
    net::{TcpListener, TcpStream},
    sync::watch,
    task::{JoinError, JoinSet},
    time::timeout,
};
use tonic::transport::{server::TcpIncoming, Server};
use tonic_health::{server::HealthReporter, ServingStatus};
use tracing::{error, info, warn};

// The Decide call's two bounds, for the reference client: it holds them at
// its own door and reads no second copy.
#[cfg(feature = "client")]
pub(crate) use self::wire::{MAX_FRAME, MAX_REQUEST_ID};
pub use self::{
    admin::{Version, VERSION},
    config::{Config, ConfigError, DiscoverySettings, Listener, Scope, Tls},
    election::{ElectionConfig, LeaseError, LeasePlace},
    principal::ENTRY as PRINCIPAL_ENTRY,
    readiness::{Readiness, State},
    run::run,
    spec_store::StoreRefused,
    tls::TlsError,
};
use self::{
    election::ElectionObservation,
    principal::Layer,
    readiness::Phase,
    service::{Service, Shared},
    spec_file::SpecFile,
    spec_store::SpecStore,
    standby::{Lag, Standby},
};
use crate::{
    service::elector::{Elector, Fence},
    stages::registry::StageRegistry,
    store::{MemoryStore, WorkersStore},
    worker_host::{KvFeed, KvTables, Members, Objects},
};

/// The protobuf package of the controller's API.
pub const PROTOCOL_PACKAGE: &str = "smg.policy.v1";

/// The newest protocol version this build speaks: the top of the range its
/// `Hello` offers and its `Leader` answers, and the newest its `Welcome`
/// picks (design 5.2). A change a peer must know its controller reads
/// raises it, and every other additive change within the package keeps
/// it. The rule binds from the first shipped release that carries the
/// contract, as the contract's header says. Version 2 is the first whose
/// frames state a request's hard demands (`DecideRequest.demands`).
pub const PROTOCOL_VERSION: u32 = 2;

/// The oldest protocol version this build speaks, the bottom of the same
/// two ranges: a controller speaks its own version and the one before it,
/// and upgrades first.
pub const PROTOCOL_FLOOR: u32 = PROTOCOL_VERSION - 1;

/// The gRPC service every scope is a part of (design 8: one service, four
/// credentialed scopes), as the health service names it. Every listener
/// reports it `NOT_SERVING` from its bind on, and no line reports it
/// `SERVING` yet: `Session` being served moves no listener's word. The
/// design's rule waits for the steps that serve a scope's work: `SERVING`
/// while the listener's replica leads and its scope's work can be done
/// there (on the gateway scope, a decision answered). `Leader` is every
/// scope's and moves no listener's word.
pub const SERVICE_NAME: &str = "smg.policy.v1.GrandCentral";

/// A controller with its listeners bound and nothing served yet.
pub struct Controller {
    config: Config,
    /// The stages a policy document may name, fixed at the bind: the stock
    /// build's, or the host's own.
    catalog: Arc<StageRegistry>,
    phase: watch::Sender<Phase>,
    readiness: Readiness,
    metrics: Arc<metrics::Metrics>,
    /// This replica's candidate: the election thread steps it once served.
    elector: Elector,
    /// The election's sight of the lease, written by its thread.
    standing: watch::Sender<ElectionObservation>,
    /// The listeners' close: one word, said once by the serve's drain.
    /// Every task of the serve hears it, and an open decide stream through
    /// [`Shared::stop`].
    stop: watch::Sender<bool>,
    /// The term this replica leads, told by the serve's one task
    /// ([`tell_role`]), which takes this sender whole: [`Shared::role`] is
    /// its receiver.
    role: watch::Sender<Option<u64>>,
    /// What every listener's instance of the service reads; the store is
    /// its field, the run loop's tasks read the same one.
    shared: Arc<Shared>,
    /// The spec's file source, once [`Controller::spec_file`] has read it:
    /// the serve watches it.
    spec_file: Option<SpecFile>,
    listeners: Vec<Bound>,
    admin: AdminSockets,
    admin_addr: SocketAddr,
    /// The replica's worker objects: one object of the worker crate a row
    /// of the store, followed by one task of the serve.
    workers: Arc<worker_objects::WorkerObjects>,
    kv_feed: KvFeed<Members>,
    /// The health probes of the replica's workers: one task of the serve,
    /// which probes while this replica leads.
    prober: worker_objects::Prober,
    /// The load reports of the replica's workers: one task of the serve,
    /// which polls while this replica leads. The deciding threads hold its
    /// channel from the bind.
    monitor: worker_objects::LoadMonitor,
    /// Worker discovery, where the configuration names a provider: one task
    /// of the serve, which runs the provider while this replica leads.
    #[cfg(feature = "discovery")]
    discovery: Option<Box<discovery::Discovery>>,
}

/// Sets an accepted socket to send at once. An answer is a few small writes
/// in a row. A socket left with the kernel's default holds each write after
/// the first until the peer acknowledges the one before, and a peer with
/// nothing to send delays that acknowledgement by tens of milliseconds: a
/// call answered in microseconds took about 40 ms on the wire.
fn send_at_once(socket: &mut TcpStream) {
    if let Err(error) = socket.set_nodelay(true) {
        warn!(%error, "an accepted socket keeps the kernel's send delay");
    }
}

/// The admin listener as it accepts: each socket set to send at once.
type AdminSockets = TapIo<TcpListener, fn(&mut TcpStream)>;

/// Why the serve left its wait.
enum Leave {
    /// The shutdown future resolved.
    Signal,
    /// A listener's task ended first.
    Task(Option<Result<Result<(), Error>, JoinError>>),
    /// The election thread ended first.
    Election,
}

/// A gRPC listener, bound, with its server configured (TLS parsed). Its
/// accepted sockets are set to send at once by the transport's own
/// [`TcpIncoming::with_nodelay`]; [`send_at_once`] says why.
struct Bound {
    name: String,
    scopes: Vec<Scope>,
    addr: SocketAddr,
    listener: TcpIncoming,
    server: Server,
    tls: bool,
    /// Whether it verifies its callers against a client CA, as the bind
    /// read its TLS material.
    mutual: bool,
    /// How the listener's calls come by a principal.
    layer: Layer,
}

/// Where a bound controller listens.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct Addresses {
    /// The gRPC listeners, in configuration order.
    pub listeners: Vec<ListenerAddress>,
    /// The admin listener.
    pub admin: SocketAddr,
}

/// One bound gRPC listener.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ListenerAddress {
    /// The listener's name.
    pub name: String,
    /// The scopes it serves.
    pub scopes: Vec<Scope>,
    /// Where it listens.
    pub addr: SocketAddr,
    /// Whether it speaks TLS.
    pub tls: bool,
}

impl Addresses {
    /// Where `scope` is served.
    pub fn of(&self, scope: Scope) -> Option<SocketAddr> {
        self.listeners
            .iter()
            .find(|l| l.scopes.contains(&scope))
            .map(|l| l.addr)
    }
}

/// A controller that did not start or did not serve.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The configuration was refused.
    #[error("configuration: {0}")]
    Config(#[from] ConfigError),
    /// A listener's address could not be bound.
    #[error("listener {name} on {addr}: {source}")]
    Bind {
        /// The listener.
        name: String,
        /// Its address.
        addr: SocketAddr,
        /// The bind error.
        #[source]
        source: io::Error,
    },
    /// A listener's TLS files were refused.
    #[error("listener {name}: {source}")]
    Tls {
        /// The listener.
        name: String,
        /// Which file, and why.
        #[source]
        source: TlsError,
    },
    /// A listener's TLS material did not parse.
    #[error("listener {name} tls: {source}")]
    TlsConfig {
        /// The listener.
        name: String,
        /// The transport's error.
        #[source]
        source: tonic::transport::Error,
    },
    /// A listener failed while serving.
    #[error("listener {name}: {source}")]
    Serve {
        /// The listener.
        name: String,
        /// The transport's error.
        #[source]
        source: tonic::transport::Error,
    },
    /// The admin listener could not be bound or failed while serving.
    #[error("admin listener: {0}")]
    Admin(#[source] io::Error),
    /// The spec file could not be read, or its document does not run.
    #[error("the spec file {}: {source}", .path.display())]
    SpecFile {
        /// The file.
        path: PathBuf,
        /// Why its document was not applied.
        #[source]
        source: SpecRejected,
    },
    /// The spec store could not be read, its newest generation does not
    /// run, or this controller cannot take one.
    #[error("the spec store {}: {source}", .path.display())]
    SpecStore {
        /// The directory.
        path: PathBuf,
        /// Why it did not become the spec's source.
        #[source]
        source: StoreRefused,
    },
    /// The lease's place could not be opened, or the elector refused its
    /// identity.
    #[error("the election's lease: {0}")]
    Lease(#[source] LeaseError),
    /// The election thread could not be started.
    #[error("the election thread: {0}")]
    ElectionThread(#[source] io::Error),
    /// A deciding thread could not be started.
    #[error("the deciding threads: {0}")]
    DecisionThreads(#[source] io::Error),
    /// The election thread ended before the shutdown: nothing renews the
    /// lease, so the controller stops.
    #[error("the election thread ended before the shutdown")]
    ElectionEnded,
    /// A deciding thread ended before the shutdown: the models it decided
    /// have no decision slot, so the controller stops and gives the lead up.
    #[error("a deciding thread ended before the shutdown")]
    DecisionThreadEnded,
    /// A listener returned before the shutdown signal.
    #[error("a listener stopped before the shutdown")]
    Stopped,
    /// A listener task ended abnormally.
    #[error("a listener task ended abnormally: {0}")]
    Task(String),
}

impl Controller {
    /// [`Controller::bind_with`] over the stock build's stages.
    pub async fn bind(config: Config) -> Result<Self, Error> {
        Self::bind_with(config, StageRegistry::builtin()).await
    }

    /// Validates the configuration, installs the process crypto provider,
    /// reads and parses every listener's TLS material, binds every address,
    /// opens the lease's place and builds the store. Nothing is served yet
    /// and no lease is taken, so a refusal arrives before a client can
    /// connect. A policy document names the stages `catalog` holds: the
    /// stock build's and the ones the host added. One that needs the
    /// lifecycle's calls ([`crate::stages::Lifecycle::needs_lifecycle`]) is
    /// refused as not in this build: the controller makes the membership call,
    /// and a stage hears a lease's issue, its dispatch, completion and release
    /// at its gateway's word and its end at the dispatch deadline or where its
    /// answer does not leave here, and not yet the end of a lease whose gateway
    /// was let go.
    pub async fn bind_with(config: Config, catalog: StageRegistry) -> Result<Self, Error> {
        config.validate()?;
        tls::install_crypto_provider();
        // The stages a policy document may name in this build, the host's
        // own among them.
        info!(
            controller_id = %config.controller_id,
            filters = ?catalog.filters().collect::<Vec<_>>(),
            scorers = ?catalog.scorers().collect::<Vec<_>>(),
            pickers = ?catalog.pickers().collect::<Vec<_>>(),
            "stages held"
        );
        // The store makes the membership call. A served decision tells a
        // stage of its lease, a gateway's frame of the lease's dispatch,
        // completion or release, and the dispatch deadline or an answer
        // that does not leave of its end; no line tells one of a lease
        // that left with its gateway yet: a stage that needs the calls is
        // refused by name, never run on state half kept.
        let catalog = Arc::new(catalog.without_lifecycle());
        let mut listeners = Vec::with_capacity(config.listeners.len());
        for listener in &config.listeners {
            let name = listener.name.clone();
            let mut server = Server::builder();
            let mut mode = "off";
            let mut mutual = false;
            if let Some(tls) = &listener.tls {
                let material = tls::Material::read(tls)
                    .await
                    .map_err(|source| Error::Tls {
                        name: name.clone(),
                        source,
                    })?;
                server = server
                    .tls_config(material.server_config())
                    .map_err(|source| Error::TlsConfig {
                        name: name.clone(),
                        source,
                    })?;
                mutual = material.mutual();
                mode = if mutual { "mutual" } else { "server" };
            }
            let bind = |source| Error::Bind {
                name: name.clone(),
                addr: listener.addr,
                source,
            };
            let socket = TcpListener::bind(listener.addr).await.map_err(bind)?;
            let addr = socket.local_addr().map_err(bind)?;
            info!(
                controller_id = %config.controller_id,
                listener = %name,
                %addr,
                scopes = ?listener.scopes,
                tls = mode,
                "listener bound"
            );
            listeners.push(Bound {
                name,
                scopes: listener.scopes.clone(),
                addr,
                listener: TcpIncoming::from(socket).with_nodelay(Some(true)),
                server,
                tls: listener.tls.is_some(),
                mutual,
                layer: Layer::Off,
            });
        }
        let admin = TcpListener::bind(config.admin)
            .await
            .map_err(Error::Admin)?;
        let admin_addr = admin.local_addr().map_err(Error::Admin)?;
        let admin = admin.tap_io(send_at_once as fn(&mut TcpStream));
        info!(controller_id = %config.controller_id, addr = %admin_addr, "admin listener bound");
        // The holder identity names the gateway listener as bound: a client
        // dials the leader there, with no Service between them.
        let gateway = listeners
            .iter()
            .find(|bound| bound.scopes.contains(&Scope::Gateway))
            .map(|bound| bound.addr)
            .ok_or(ConfigError::ScopeUnmounted(Scope::Gateway))?;
        let address = config.election.address(gateway)?;
        let identity = election::identity(&config.controller_id, address.as_deref());
        let elector = election::elector(&config.election, identity.clone())
            .await
            .map_err(Error::Lease)?;
        let settings = elector.settings();
        info!(
            controller_id = %config.controller_id,
            identity = %identity,
            lease = %config.election.lease,
            duration = ?settings.duration,
            renew_period = ?settings.renew_period,
            renew_deadline = ?settings.renew_deadline,
            "election"
        );
        let (standing, stood) = watch::channel(ElectionObservation::default());
        let steps = stood.clone();
        let journal = replication::journaled(&config.election.lease, config.decide_shards);
        let workers = Arc::new(worker_objects::WorkerObjects::stating(
            Arc::clone(&journal.0),
            config.worker_overload,
        ));
        let kv_feed = KvFeed::new(workers.holder());
        // The engine reads the index the feed's streams write: the store is
        // lent it once, ahead of the first decision, and its one engine
        // holds it under every policy spec it applies.
        journal.0.lend_cache_indexes(workers.cache_indexes());
        let prober = worker_objects::Prober::new(Arc::clone(&workers), elector.fence());
        // The load monitor is built stopped, ahead of the deciding threads:
        // each takes its channel at its start, and the serve runs its task.
        let monitor = worker_objects::LoadMonitor::new(&workers, config.load_poll_interval);
        // The feed takes the monitor's two parts ahead of its first stream.
        // The load records on the workers' streams go to the monitor while
        // this replica leads. The monitor says whose loads are wanted, so a
        // gRPC worker that is not indexed streams its load records alone.
        // Every replica opens that stream: a take finds it open.
        monitor.fed_by(&kv_feed);
        // The serve's two words, each said by one task of it: `Shared` holds
        // their receivers, as every listener's instance hears them.
        let (stop, stopped) = watch::channel(false);
        let (role, led) = watch::channel(None);
        // The standby's lag, made once from the configured interval: the
        // readiness reads the one `Shared` holds.
        let lag = Arc::new(Lag::new(standby::synced_within(config.load_poll_interval)));
        // Under a lease another replica can take, the document door is
        // built under this replica's fence, with a spec source or with
        // none: its source writes while the replica leads, and its hook
        // settles a worker's status under the replica's own lead.
        let shared = Shared::new(
            config.controller_id.clone(),
            address.unwrap_or_default(),
            &elector,
            (stood, stopped, led, lag),
            (journal.0, journal.1, workers.holder(), monitor.loads()),
            (Arc::clone(&catalog), &config.election.lease),
            config.decide_shards,
        )
        .map_err(Error::DecisionThreads)?;
        let (phase, readiness) = readiness::channel(
            elector.fence(),
            steps,
            Arc::clone(&shared.lag),
            Arc::clone(&shared.feed.log),
        );
        let shared = Arc::new(shared);
        #[cfg(feature = "discovery")]
        let discovery = config
            .discovery
            .as_ref()
            .map(|settings| {
                // The replica's own writes are admitted as a control call's
                // are: under the term its store leads, read at each write.
                let own = Service::new(Vec::new(), Arc::clone(&shared));
                let door = membership::ReplicaMembershipWriter::new(
                    Arc::clone(&shared.door),
                    principal::Principal::replica(&config.controller_id),
                    move || own.leading(std::time::Instant::now()),
                );
                discovery::Discovery::registering(
                    settings,
                    &workers.holder(),
                    Arc::new(door),
                    Arc::clone(&shared.fence),
                    config.worker_overload,
                )
                .map(Box::new)
            })
            .transpose()?;
        Ok(Self {
            config,
            catalog,
            phase,
            readiness,
            metrics: Arc::new(metrics::Metrics::new()),
            elector,
            standing,
            stop,
            role,
            shared,
            spec_file: None,
            listeners,
            admin,
            admin_addr,
            workers,
            kv_feed,
            prober,
            monitor,
            #[cfg(feature = "discovery")]
            discovery,
        })
    }

    /// Where the controller listens; every port is the bound one.
    pub fn addresses(&self) -> Addresses {
        Addresses {
            listeners: self
                .listeners
                .iter()
                .map(|bound| ListenerAddress {
                    name: bound.name.clone(),
                    scopes: bound.scopes.clone(),
                    addr: bound.addr,
                    tls: bound.tls,
                })
                .collect(),
            admin: self.admin_addr,
        }
    }

    /// Where this replica dials the leader it follows, or why it follows
    /// none: read once the listeners' layers stand, at the serve's start.
    fn follows(&self) -> Result<u16, NoFollow> {
        if !self.config.election.lease.is_shared() {
            return Err(NoFollow::Lease);
        }
        following(&self.listeners)
    }

    /// Refuses a replica that follows a leader ([`Controller::follows`])
    /// with its replication listener configured on port `0`: asked first
    /// at the serve's start, before anything is served. Replicas are
    /// configured alike, so a standby dials its leader at the port of its
    /// own replication listener, and a number the host drew for this
    /// replica is no other's. A replica that follows nobody keeps port
    /// `0`.
    fn names_its_port(&self) -> Result<(), ConfigError> {
        match unnamed_port(&self.config.listeners) {
            Some(name) if self.follows().is_ok() => {
                Err(ConfigError::ReplicationPortUnnamed(name.to_owned()))
            }
            _ => Ok(()),
        }
    }

    /// Names `listener` one that takes its callers' identities as told
    /// (the contract's `METHOD_AS_TOLD`): a caller states its name in the
    /// `x-smg-principal` metadata entry of every call and nothing is
    /// verified, so whoever reaches the listener acts as any principal of
    /// its scopes. It is for a listener only one trust domain reaches: a
    /// loopback, or a mesh that verifies in front of the process. No
    /// listener is named by default, and one that is not named refuses
    /// every call that needs a principal.
    ///
    /// Called between [`Controller::bind`] and [`Controller::serve`]. This
    /// method is the setting's one home until the configuration is built
    /// in one place.
    ///
    /// # Errors
    ///
    /// No listener has the name, or the listener verifies its callers
    /// against a client CA: a listener has one credential rule.
    pub fn identity_as_told(&mut self, listener: &str) -> Result<(), ConfigError> {
        let bound = self
            .listeners
            .iter_mut()
            .find(|bound| bound.name == listener)
            .ok_or_else(|| ConfigError::UnknownListener(listener.to_owned()))?;
        // The bind read whether the listener verifies its callers: that one
        // reading is asked here, not the configuration again.
        if bound.mutual {
            return Err(ConfigError::AsToldBesideClientCa(listener.to_owned()));
        }
        bound.layer = Layer::AsTold;
        warn!(
            controller_id = %self.config.controller_id,
            listener = %bound.name,
            addr = %bound.addr,
            scopes = ?bound.scopes,
            "identities are taken as told on this listener: nothing is verified, and whoever reaches it acts as any principal of its scopes"
        );
        Ok(())
    }

    /// The stages a policy document may name in this controller, fixed at
    /// the bind: a spec file's document is validated against them
    /// ([`Controller::spec_file`]).
    ///
    /// A stage the host registered under a name the engine refuses
    /// requests under is among them, and no document may name it.
    pub fn stage_registry(&self) -> &StageRegistry {
        &self.catalog
    }

    /// Takes the controller's policy spec from the file at `path` (design
    /// 4.6). A replica alone over its lease reads and applies the document
    /// now, as generation 1 or under the number its `generation` line
    /// writes, and the control scope answers `GetPolicySpec` with it.
    /// [`Self::serve`] then watches the file: a changed one is applied as
    /// the next generation, or under the number it writes, and one that
    /// does not run is rejected whole, with the running generation kept.
    ///
    /// A replica under a lease another replica can take applies nothing
    /// here: its document is passed, and applied under the replica's first
    /// lead where the replica runs nothing by then, at its first take or by
    /// its watch once that lead opens. It holds no document until it
    /// follows or leads, and `GetPolicySpec` says so. Until its document
    /// lands, a replica that leads has no decision for a request. Until it
    /// leads, a replica that follows a leader runs what its leader's log
    /// gives it, under the leader's number. Its watch looks at the file
    /// all the while and applies nothing: it shows the door what the file
    /// holds, and says nothing of a file it cannot read. Once the replica
    /// leads, the watch reads the file again unless its document stood on
    /// the replica since the file took it: a rollout's new file is
    /// applied, and a mount that lags brings nothing older back. The lease
    /// alone decides which of the two starts this is, whatever the order
    /// the listeners are configured in.
    ///
    /// A file that cannot be read, or a document the validator refuses, is
    /// refused here, before anything serves. The document names the stages
    /// the bind was handed ([`Controller::stage_registry`]), a host's own among
    /// them. A controller that takes no spec source serves no spec call,
    /// watches nothing and renders no spec series, and one that took a spec
    /// store takes no file ([`SpecRejected::SecondSource`]).
    pub async fn spec_file(&mut self, path: impl Into<PathBuf>) -> Result<(), Error> {
        let path = path.into();
        let generations = Arc::clone(&self.shared.generations);
        // Under a shared lease a replica may follow a leader, and then
        // runs its leader's generations: its own file is passed here and
        // applied under its first lead, and its source writes while the
        // replica leads under the term its bytes were read under. The door
        // knows which start this is by the fence the bind built it under.
        let read = {
            let (path, generations) = (path.clone(), Arc::clone(&generations));
            tokio::task::spawn_blocking(move || SpecFile::start(path, generations))
        };
        match read.await {
            Ok(Ok(file)) => self.spec_file = Some(file),
            Ok(Err(source)) => return Err(Error::SpecFile { path, source }),
            Err(join) => return Err(Error::Task(join.to_string())),
        }
        info!(
            controller_id = %self.config.controller_id,
            path = %path.display(),
            generation = generations.generation(),
            at_first_take = generations.starts_deferred(),
            "spec file"
        );
        self.metrics.spec(generations);
        Ok(())
    }

    /// Takes the controller's policy spec through the apply call, persisted
    /// in the directory at `dir` (design 4.6): the newest generation on
    /// file is applied now, under the number its name states, and the
    /// control scope then takes `ApplyPolicySpec` in enforce mode where
    /// this replica leads. A document is persisted in the directory, then
    /// applied as the next generation; a rollback runs a generation on file
    /// again as the next one, and the read call answers a generation on
    /// file by its number. A directory with no generation on file starts
    /// the controller with no document, until the first apply.
    ///
    /// One controller holds the directory, under a lock on a file it keeps
    /// there. A start beside another that holds it waits for that one to
    /// stop, for the grace period at most ([`Config::shutdown_grace`]), and
    /// reads the directory once it holds it: it runs that one's last apply.
    ///
    /// Refused here, before anything serves ([`StoreRefused`]): a path that
    /// leads to no directory, a directory another controller still holds at
    /// the end of that wait, one renamed under that wait, a lock the system
    /// does not give, a directory with no lock file where the controller
    /// may make none, a directory that is not read, two files under one
    /// number, a newest generation that does not run, a spec source taken
    /// before this one, and a lease another replica can take
    /// ([`LeasePlace`]). A standby follows a file's generations alone in
    /// this build: a store's directory is one replica's own, so a promoted
    /// one would hold none of it.
    pub async fn spec_store(&mut self, dir: impl Into<PathBuf>) -> Result<(), Error> {
        let path = dir.into();
        let refused = |source| Error::SpecStore {
            path: path.clone(),
            source,
        };
        if self.config.election.lease.is_shared() {
            return Err(refused(StoreRefused::BesideASharedLease));
        }
        // Asked before the directory is: a first source may hold it, and a
        // second one does not wait on this controller's own lock. The door
        // refuses a second source under its lock all the same.
        if self.shared.generations.bound() {
            return Err(refused(StoreRefused::SecondSource));
        }
        let generations = Arc::clone(&self.shared.generations);
        let metrics = Arc::clone(&self.metrics);
        let controller_id = self.config.controller_id.clone();
        let bound = {
            let path = path.clone();
            // A controller that stops lets the directory go inside its
            // grace period: a start waits that long for one before it.
            let wait = self.config.shutdown_grace;
            // The bind, its line and its series are one step of the task:
            // a caller that drops this call while the start waits leaves a
            // store bound with both.
            tokio::task::spawn_blocking(move || {
                let store = SpecStore::open_within(path, wait)?;
                // The line names the directory held: its path as resolved,
                // the one the line of a wait names.
                let held = store.dir().to_owned();
                generations.bind_store(store)?;
                info!(
                    controller_id = %controller_id,
                    path = %held.display(),
                    generation = generations.generation(),
                    "spec store"
                );
                metrics.spec(generations);
                Ok(())
            })
        };
        match bound.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(source)) => Err(refused(source)),
            Err(join) => Err(Error::Task(join.to_string())),
        }
    }

    /// The store the serve's tasks and the scopes' calls feed and read.
    pub fn store(&self) -> Arc<MemoryStore> {
        Arc::clone(&self.shared.store)
    }

    /// The KV tables of this replica: the ones its holder of worker objects
    /// makes lanes in and its cache-event feed writes. A caller reads a
    /// row's cursor, its indexed blocks, the resync counts and a model's
    /// block size through it, and writes nothing.
    pub fn kv_tables(&self) -> Arc<KvTables> {
        self.workers.kv_tables()
    }

    /// The holder of the replica's worker objects, which the scopes' calls
    /// write a worker through. A host that registers a worker in the
    /// serve's own process writes it here ([`Objects::upsert`]) while the
    /// replica leads ([`Self::readiness`]). No call asks the term itself:
    /// the holder's doc says what a write on a replica that follows comes
    /// to ("A host's write calls"). A row written through the entries of
    /// [`Self::store`] itself has no worker object, and no first copy holds
    /// it, until a cut is laid whole. A standby's applier also lays it at
    /// the next event that tells the row and that the store takes.
    pub fn workers(&self) -> Arc<Objects<Members>> {
        self.workers.holder()
    }

    /// A read handle on the controller's state, valid across [`Self::serve`].
    pub fn readiness(&self) -> Readiness {
        self.readiness.clone()
    }

    /// The controller's one fence (design 5.3), valid across
    /// [`Self::serve`]: the term this replica leads under, or none past its
    /// renew deadline. A shard thread reads it before every decision it
    /// sends; readiness and the `Leader` call read the same one.
    pub fn fence(&self) -> Arc<Fence> {
        Arc::clone(&self.shared.fence)
    }

    /// The load reports of this replica's workers as its decisions are
    /// given them: the channel its load monitor publishes on, valid across
    /// [`Self::serve`]. The monitor publishes a snapshot at each poll while
    /// the replica leads, and an empty one at each lead's end.
    pub fn loads(&self) -> LoadReceiver {
        self.monitor.loads()
    }

    /// Serves until `shutdown` resolves, then drains. The listeners answer
    /// from the start; the load monitor polls the replica's workers while
    /// the replica leads, from each take of the lead; the election thread
    /// takes its first look at the lease, and the state turns
    /// [`State::Leader`] or [`State::Follower`] and the health service's
    /// overall status `SERVING` once that look has landed. A replica that
    /// runs a standby turns [`State::Syncing`] where it does not lead, and
    /// is [`State::Follower`] and `SERVING` only while its standby is
    /// synced with its leader. At the shutdown
    /// the state turns [`State::Draining`] and the overall status
    /// `NOT_SERVING` at once, and a leader closes its fence and releases the
    /// lease before the settle window, so a follower takes it within one
    /// renew period; the listeners keep accepting for the settle window,
    /// stop accepting, and close within the rest of the grace period, their
    /// streams with them. An open `Decide` stream ends `UNAVAILABLE` when
    /// they stop accepting: it does not hold the close to the grace period.
    ///
    /// Every listener's health service reports the overall status (service
    /// `""`) as the readiness and [`SERVICE_NAME`] as `NOT_SERVING`, but a
    /// control listener that takes a principal: `SERVING` while this replica
    /// leads, so a client that checks the service's health by that name
    /// finds the leader's control listener. That word is told at every move
    /// of the run loop, after every step of the election and where the
    /// fence's window opens or ends, so a fence that closes between two
    /// steps is told as it closes; a registration call reads the fence
    /// itself. An open watch of the rows follows the same telling: it ends
    /// `ABORTED` when this replica stops leading, an idle one where the
    /// fence closes. A lead opens between two steps, a moment before its
    /// telling: a watch admitted in that moment is ended by any telling
    /// that finds no leader here, the drain's first.
    ///
    /// A listener, an election thread or a deciding thread that ends before
    /// the shutdown ends the serve with its error, the other listeners
    /// closed.
    ///
    /// A replica that follows a leader with its replication listener
    /// configured on port `0` is refused before anything is served
    /// ([`ConfigError::ReplicationPortUnnamed`]).
    pub async fn serve(self, shutdown: impl Future<Output = ()>) -> Result<(), Error> {
        self.serve_until(shutdown, std::future::pending()).await
    }

    /// [`Controller::serve`] with `fault` as a listener's own end: when it
    /// resolves, the serve leaves by its error as it leaves by a listener
    /// that failed. The serve itself never resolves it; a case does.
    async fn serve_until(
        self,
        shutdown: impl Future<Output = ()>,
        fault: impl Future<Output = Error>,
    ) -> Result<(), Error> {
        self.names_its_port()?;
        let follows = self.follows();
        let Self {
            config,
            phase,
            readiness,
            metrics,
            elector,
            standing,
            stop,
            role,
            shared,
            spec_file,
            listeners,
            admin,
            kv_feed,
            prober,
            monitor,
            #[cfg(feature = "discovery")]
            discovery,
            ..
        } = self;
        let store = Arc::clone(&shared.store);
        // A replica under a shared lease follows its leader where it can,
        // and says once why it cannot: it then behaves as before a standby
        // existed.
        let follows = match follows {
            Ok(port) => Some(port),
            // Nobody else leads under a lease of the replica's own.
            Err(NoFollow::Lease) => None,
            Err(why) => {
                info!(
                    controller_id = %config.controller_id,
                    why = why.as_str(),
                    "this replica follows no leader"
                );
                None
            }
        };
        let mut tasks: JoinSet<Result<(), Error>> = JoinSet::new();
        let mut reporters = Vec::with_capacity(listeners.len());
        // The reporters of the listeners whose door a caller can pass: the
        // control scope's, where a layer places a principal.
        let mut doors = Vec::new();
        for bound in listeners {
            let (reporter, health) = tonic_health::server::health_reporter();
            reporter
                .set_service_status("", ServingStatus::NotServing)
                .await;
            reporter
                .set_service_status(SERVICE_NAME, ServingStatus::NotServing)
                .await;
            if bound.scopes.contains(&Scope::Control) && bound.layer != Layer::Off {
                doors.push(reporter.clone());
            }
            let routes = Service::new(bound.scopes, Arc::clone(&shared)).routes(bound.layer);
            let mut stop = stop.subscribe();
            let name = bound.name;
            let incoming = bound.listener;
            let mut server = bound.server;
            tasks.spawn(async move {
                server
                    .add_routes(routes)
                    .add_service(health)
                    .serve_with_incoming_shutdown(incoming, async move {
                        let _ = stop.wait_for(|stop| *stop).await;
                    })
                    .await
                    .map_err(|source| Error::Serve { name, source })
            });
            reporters.push(reporter);
        }
        let teller = tell_role(readiness.clone(), stop.subscribe(), doors, role);
        tasks.spawn(async move {
            teller.await;
            Ok(())
        });
        // The store's publications are paid by one loop on every replica,
        // whoever wrote the rows: a followed row reaches the published
        // world as a leader's own does.
        let (published, publications_stop) = (Arc::clone(&store), stop.subscribe());
        tasks.spawn(async move {
            publications::run(&published, publications_stop).await;
            Ok(())
        });
        // The load monitor's counts are read where they are rendered: the
        // reader is taken ahead of the monitor's task.
        let loads = Arc::new(monitor.counts());
        if let Some(port) = follows {
            info!(
                controller_id = %config.controller_id,
                port,
                "this replica follows its leader while it does not lead"
            );
            // From here this replica is ready as a follower only while
            // its standby is synced.
            shared.lag.follow();
            let standby = Standby::new(Arc::clone(&shared), port);
            let standby_stop = stop.subscribe();
            tasks.spawn(async move {
                standby.run(standby_stop).await;
                Ok(())
            });
        }
        {
            let kv_stop = stop.subscribe();
            tasks.spawn(async move {
                kv_feed.run(kv_stop).await;
                Ok(())
            });
            // The probes run under the role's word on every replica: one
            // that leads alone hears it from its own election, and one
            // that does not lead dials no engine for its health.
            let (probes_stop, led) = (stop.subscribe(), shared.role.clone());
            tasks.spawn(async move {
                prober.run(probes_stop, led).await;
                Ok(())
            });
            // The load monitor runs under the same word: a replica that
            // does not lead polls no engine for its load either. Its stop
            // publishes the snapshot empty for the deciding threads, but
            // for the one loop `LoadMonitor::run` names.
            let (loads_stop, led) = (stop.subscribe(), shared.role.clone());
            tasks.spawn(async move {
                monitor.run(loads_stop, led).await;
                Ok(())
            });
        }
        #[cfg(feature = "discovery")]
        if let Some(discovery) = discovery {
            let standing = shared.standing.clone();
            let discovery_stop = stop.subscribe();
            tasks.spawn(async move {
                discovery.run(standing, discovery_stop).await;
                Ok(())
            });
        }
        // The presence sweep: what a gateway held is let go once its grace
        // period ended with no new `Hello`. The names that left, the leases
        // past their dispatch deadline and the upkeep of the standing
        // document's engine ride its task's clock (`Gateways::run`).
        let sweep = shared.sweep(stop.subscribe());
        tasks.spawn(async move {
            sweep.await;
            Ok(())
        });
        // The restores' clock: a leader lets go of a snapshot's cut whose
        // standby took no part of it for the bound.
        let restores = shared.replication.clone();
        let restores_stop = stop.subscribe();
        tasks.spawn(async move {
            restores.run(restores_stop).await;
            Ok(())
        });
        if let Some(file) = spec_file {
            let file_stop = stop.subscribe();
            // A replica that follows applies its leader's generations from
            // the log: its own watch asks the door, and looks at the file
            // while its source writes there.
            tasks.spawn(async move {
                file.run(file_stop)
                    .await
                    .map_err(|join| Error::Task(join.to_string()))
            });
        }
        // A deciding thread that unwound decides nothing from then on and
        // nothing starts it again: the serve ends by its error, as it ends
        // by the election thread's.
        let mut unwound = shared.deciders.ended();
        let mut deciders_stop = stop.subscribe();
        tasks.spawn(async move {
            let told = async { unwound.wait_for(|unwound| *unwound).await.is_ok() };
            let stopped = async {
                let _ = deciders_stop.wait_for(|stop| *stop).await;
            };
            tokio::select! {
                true = told => Err(Error::DecisionThreadEnded),
                () = stopped => Ok(()),
            }
        });
        let app = admin::router(admin::Admin {
            readiness: readiness.clone(),
            metrics: Arc::clone(&metrics),
            loads,
            decides: Arc::clone(shared.deciders.counts()),
            store: Arc::clone(&store),
        });
        let mut admin_stop = stop.subscribe();
        tasks.spawn(async move {
            axum::serve(admin, app)
                .with_graceful_shutdown(async move {
                    let _ = admin_stop.wait_for(|stop| *stop).await;
                })
                .await
                .map_err(Error::Admin)
        });
        let mut looks = shared.standing.clone();
        let mut election = shared
            .elect(elector, standing)
            .map_err(Error::ElectionThread)?;

        let mut shutdown = pin!(shutdown);
        let mut fault = pin!(fault);
        // Not ready before the first look at the lease has landed: a replica
        // that cannot reach its lease never passes a probe, so a rollout
        // stops at it.
        let first_look = async { looks.wait_for(|standing| standing.looked).await.is_ok() };
        let starting = tokio::select! {
            looked = first_look => if looked { None } else { Some(Leave::Election) },
            () = &mut shutdown => Some(Leave::Signal),
            joined = tasks.join_next() => Some(Leave::Task(joined)),
            error = &mut fault => Some(Leave::Task(Some(Ok(Err(error))))),
        };
        let leave = match starting {
            Some(leave) => leave,
            None => {
                // A replica that runs no standby is ready from here, and
                // its word is told before the state moves. One that runs a
                // standby is ready only while it is synced: its word is
                // told at each turn.
                let mut told = None;
                if follows.is_none() {
                    for reporter in &reporters {
                        reporter
                            .set_service_status("", ServingStatus::Serving)
                            .await;
                    }
                    told = Some(true);
                }
                phase.send_replace(Phase::Serving);
                let mut word = Box::pin(tell_readiness(
                    readiness.clone(),
                    &reporters,
                    told,
                    follows.is_some(),
                    &config.controller_id,
                    &store,
                ));
                tokio::select! {
                    () = &mut shutdown => Leave::Signal,
                    joined = tasks.join_next() => Leave::Task(joined),
                    () = election.ended() => Leave::Election,
                    error = &mut fault => Leave::Task(Some(Ok(Err(error)))),
                    never = &mut word => match never {},
                }
            }
        };

        phase.send_replace(Phase::Draining);
        for reporter in &reporters {
            reporter
                .set_service_status("", ServingStatus::NotServing)
                .await;
        }
        // The fence closes and the lease is released now, in that order,
        // while the listeners still answer `Leader`.
        election.stop();
        let error = match leave {
            Leave::Signal => None,
            Leave::Task(Some(Ok(Ok(()))) | None) => Some(Error::Stopped),
            Leave::Task(Some(Ok(Err(error)))) => Some(error),
            Leave::Task(Some(Err(join))) => Some(Error::Task(join.to_string())),
            Leave::Election => Some(Error::ElectionEnded),
        };
        // A grace period the clock cannot add to the present bounds nothing:
        // the wait for the election thread below then has no deadline.
        let deadline = tokio::time::Instant::now().checked_add(config.shutdown_grace);
        let result = if let Some(error) = error {
            error!(controller_id = %config.controller_id, %error, "the serve ended before the shutdown");
            tasks.abort_all();
            while tasks.join_next().await.is_some() {}
            Err(error)
        } else {
            info!(
                controller_id = %config.controller_id,
                settle = ?config.shutdown_settle,
                grace = ?config.shutdown_grace,
                "draining"
            );
            tokio::time::sleep(config.shutdown_settle).await;
            stop.send_replace(true);
            let rest = config.shutdown_grace.saturating_sub(config.shutdown_settle);
            let drained = timeout(rest, async {
                while let Some(joined) = tasks.join_next().await {
                    match joined {
                        Ok(Ok(())) => {}
                        Ok(Err(error)) => return Err(error),
                        Err(join) => return Err(Error::Task(join.to_string())),
                    }
                }
                Ok(())
            })
            .await;
            match drained {
                Ok(result) => result,
                Err(_elapsed) => {
                    warn!(
                        controller_id = %config.controller_id,
                        open = tasks.len(),
                        "the grace period ran out; closing the listeners"
                    );
                    tasks.abort_all();
                    while tasks.join_next().await.is_some() {}
                    Ok(())
                }
            }
        };
        // The release began at the leave, a signal's or an error's: one
        // write, tried again a renew period later while it fails. Neither
        // leave ends the process before the release landed or was given
        // up, within the grace period: this one wait serves both.
        match deadline {
            Some(deadline) => {
                if tokio::time::timeout_at(deadline, election.ended())
                    .await
                    .is_err()
                {
                    warn!(
                        controller_id = %config.controller_id,
                        "the election thread outlived the grace period; a lease it holds expires by its duration"
                    );
                }
            }
            None => election.ended().await,
        }
        if result.is_ok() {
            info!(controller_id = %config.controller_id, "stopped");
        }
        result
    }
}

/// Tells the role at each change of it and, at the stop, that this replica
/// left: every door's word for the service's name follows it, and so does
/// every open watch of the rows. The serve's one writer of both.
///
/// A watch is told that this replica does not lead at every move while it
/// does not, where a door is told at the change alone. A take with no
/// holder to wait for leads one renew deadline after its write, between
/// two steps of the election: the fence admits a watch from that instant,
/// and until the telling that instant wakes such a watch stands under the
/// word that nobody leads here. The word told again ends it, the drain's
/// included; told at the change alone, the drain would leave that watch
/// open to the end of the grace period.
///
/// The word is the fence's from the first take on ([`Readiness::led`]). A
/// take lands ahead of the serve's turn to serving, and a move in between
/// tells the term: told once more that nobody leads here, a watch the
/// fence admitted in that moment would end under a lead that stands.
///
/// The word names the term led, and a term is told once. A release and the
/// take behind it can both fall between two reads of this loop: the replica
/// leads at each read, under another term at the second. That term is
/// another word, so a task that runs for one term hears where it ended.
///
/// `told` is the word's one sender and goes with this task. At the task's
/// end, a serve's leave by an error included, the word ends: a watch that
/// waits is woken once more, and ends.
async fn tell_role(
    mut role: Readiness,
    mut stop: watch::Receiver<bool>,
    doors: Vec<HealthReporter>,
    told: watch::Sender<Option<u64>>,
) {
    let (mut last, mut left) = (None, false);
    loop {
        let (led, turn) = role.led();
        let led = led.filter(|_| !left);
        let leads = led.is_some();
        // The watches first: nothing is awaited between the read of the
        // fence and their word.
        if leads {
            told.send_if_modified(|was| std::mem::replace(was, led) != led);
        } else {
            told.send_replace(None);
        }
        if last != Some(leads) {
            let word = if leads {
                ServingStatus::Serving
            } else {
                ServingStatus::NotServing
            };
            for door in &doors {
                door.set_service_status(SERVICE_NAME, word).await;
            }
            last = Some(leads);
        }
        if left {
            return;
        }
        // At the stop the listeners close next: the loop says once more
        // that this replica left, then ends. A serve that is gone reads the
        // same.
        tokio::select! {
            _ = stop.wait_for(|stop| *stop) => left = true,
            moved = role.moved_by(turn) => left = moved.is_none(),
        }
    }
}

/// Why a replica follows no leader.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NoFollow {
    /// The lease is the replica's own: nobody else leads under it.
    Lease,
    /// No listener serves the replication scope.
    Listener,
    /// The replication listener places no principal: a leader's would
    /// refuse every standby.
    Principal,
    /// The replication listener serves TLS: a standby's dial has no
    /// credential yet.
    Tls,
}

impl NoFollow {
    fn as_str(self) -> &'static str {
        match self {
            Self::Lease => "the lease is this replica's own",
            Self::Listener => "no listener serves the replication scope",
            Self::Principal => "the replication listener places no principal",
            Self::Tls => {
                "the replication listener serves TLS, and a standby's dial has no credential yet"
            }
        }
    }
}

/// The listener configured to serve the replication scope on port `0`, by
/// its name.
fn unnamed_port(listeners: &[Listener]) -> Option<&str> {
    listeners
        .iter()
        .find(|listener| listener.scopes.contains(&Scope::Replication) && listener.addr.port() == 0)
        .map(|listener| listener.name.as_str())
}

/// The port a standby dials its leader at, on the leader's own host: the
/// port of this replica's listener that serves the replication scope to a
/// stated principal in the clear, since replicas are configured alike. Or
/// why the replica follows no leader, by its last replication listener.
fn following(listeners: &[Bound]) -> Result<u16, NoFollow> {
    let mut why = NoFollow::Listener;
    for bound in listeners {
        if !bound.scopes.contains(&Scope::Replication) {
            continue;
        }
        if bound.layer == Layer::Off {
            why = NoFollow::Principal;
        } else if bound.tls {
            why = NoFollow::Tls;
        } else {
            return Ok(bound.addr.port());
        }
    }
    Err(why)
}

/// Tells every listener's overall word as the readiness turns and says each
/// turn once in the log: the word's one writer while the controller serves.
/// `told` is the word the listeners hold at the call. A replica that runs
/// no standby (`follows` false) is ready from its first look to its drain,
/// so its word is told once. Never resolves.
async fn tell_readiness(
    mut role: Readiness,
    reporters: &[HealthReporter],
    mut told: Option<bool>,
    follows: bool,
    controller_id: &str,
    store: &MemoryStore,
) -> Infallible {
    let mut said = None;
    loop {
        let (state, turn) = role.read();
        let ready = state.is_ready();
        if told != Some(ready) {
            let word = if ready {
                ServingStatus::Serving
            } else {
                ServingStatus::NotServing
            };
            for reporter in reporters {
                reporter.set_service_status("", word).await;
            }
            told = Some(ready);
        }
        if said != Some(ready) {
            if ready {
                info!(
                    controller_id,
                    role = state.as_str(),
                    membership_epoch = store.epoch(),
                    "ready"
                );
            } else {
                info!(
                    controller_id,
                    role = state.as_str(),
                    "not ready: the standby is not synced with its leader"
                );
            }
            said = Some(ready);
        }
        if !follows || role.moved_by(turn).await.is_none() {
            std::future::pending::<()>().await;
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/tests.rs"]
mod tests;
