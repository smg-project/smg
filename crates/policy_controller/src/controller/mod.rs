//! The standalone controller's run loop (design 2, 6, 8 and 10): the store,
//! the load driver over it, the election, the gRPC listeners with their
//! scopes, the service and the health service on each, the admin listener,
//! readiness and the drain. The service answers `Leader`, a gateway's
//! `Session` and its `Decide` streams, whose decisions run on the
//! controller's deciding threads, and, on the leader, the control scope's
//! registration calls and the gateway scope's watch of the worker rows, and
//! the run loop sweeps the gateways that are gone; the scopes' other call
//! bodies arrive with the gateway, ingestion and control scope PRs, the
//! other ingestion drivers with their move out of the gateway. In the
//! `elected` mode the gateway runs this same loop inside its process. A
//! controller given a spec file runs its document, watches the file and
//! answers `GetPolicySpec` and a dry run of `ApplyPolicySpec`
//! ([`Controller::spec_file`]).
//!
//! A controller is bound first and served second ([`Controller::bind`], then
//! [`Controller::serve`]): the bind takes every address (port `0` for an
//! ephemeral one), reads and parses the TLS material, opens the lease's
//! place and builds the store, so a start refuses before it serves and a
//! test knows its ports; the serve runs the listeners and the election until
//! the shutdown future resolves, then gives the lease up, settles, drains
//! and returns.
//!
//! Every replica runs the same loop, the load driver included: a follower
//! that waited for a leader's log nobody writes yet would be cold at its
//! promotion. The standby turns a follower's polls into that log.
//!
//! Every socket a listener accepts is set to send at once: a small answer
//! never waits for the peer's acknowledgement of the write before it.

pub mod admin;
pub mod config;
mod decide;
mod deciders;
pub mod election;
mod floor;
mod gateways;
mod generations;
mod lifecycle;
mod membership;
pub mod metrics;
mod principal;
pub mod readiness;
mod replication;
mod run;
mod service;
mod session;
mod spec_file;
pub mod tls;
mod watch_workers;
mod wire;
mod worker_wire;

use std::{future::Future, io, net::SocketAddr, path::PathBuf, pin::pin, sync::Arc};

use axum::serve::{ListenerExt as _, TapIo};
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
    config::{Config, ConfigError, Listener, Scope, Tls},
    election::{Election, LeaseError, LeasePlace},
    generations::SpecRejected,
    principal::ENTRY as PRINCIPAL_ENTRY,
    readiness::{Readiness, State},
    run::run,
    tls::TlsError,
};
use self::{
    election::Standing,
    principal::Layer,
    readiness::Phase,
    service::{Service, Shared},
    spec_file::SpecFile,
};
use crate::{
    catalog::Catalog,
    elector::{Elector, Fence},
    ingest::{
        loads::{LoadDriver, LoadDriverConfig},
        probe::EngineProbe,
    },
    store::{MemoryStore, WorkersStore},
};

/// The protobuf package of the controller's API.
pub const PROTOCOL_PACKAGE: &str = "smg.policy.v1";

/// The protocol version this build speaks: `Hello`'s range, `Welcome`'s
/// pick and the range `Leader` answers (design 5.2). A change within the
/// package is additive and keeps it.
pub const PROTOCOL_VERSION: u32 = 1;

/// The gRPC service every scope is a part of (design 8: one service, four
/// credentialed scopes), as the health service names it. Every listener
/// reports it `NOT_SERVING` from its bind on, and no line reports it
/// `SERVING` yet: `Session` being served moves no listener's word. The
/// design's rule waits for the steps that serve a scope's work: `SERVING`
/// while the listener's replica leads and its scope's work can be done
/// there (on the gateway scope, a decision answered). `Leader` is every
/// scope's and moves no listener's word.
pub const SERVICE_NAME: &str = "smg.policy.v1.PolicyController";

/// A controller with its listeners bound and nothing served yet.
pub struct Controller {
    config: Config,
    /// The stages a policy document may name, fixed at the bind: the stock
    /// build's, or the host's own.
    catalog: Arc<Catalog>,
    phase: watch::Sender<Phase>,
    readiness: Readiness,
    metrics: Arc<metrics::Metrics>,
    /// The engines' probe the load driver dials through.
    probe: Arc<EngineProbe>,
    /// This replica's candidate: the election thread steps it once served.
    elector: Elector,
    /// The election's sight of the lease, written by its thread.
    standing: watch::Sender<Standing>,
    /// What every listener's instance of the service reads; the store is
    /// its field, the run loop's drivers read the same one.
    shared: Arc<Shared>,
    /// The spec's file source, once [`Controller::spec_file`] has read it:
    /// the serve watches it.
    spec_file: Option<SpecFile>,
    listeners: Vec<Bound>,
    admin: AdminSockets,
    admin_addr: SocketAddr,
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
pub struct Addresses {
    /// The gRPC listeners, in configuration order.
    pub listeners: Vec<ListenerAddress>,
    /// The admin listener.
    pub admin: SocketAddr,
}

/// One bound gRPC listener.
#[derive(Clone, Debug, Eq, PartialEq)]
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
    /// The engines' HTTP client could not be built.
    #[error("the engines' probe: {0}")]
    Probe(#[source] reqwest::Error),
    /// The spec file could not be read, or its document does not run.
    #[error("the spec file {}: {source}", .path.display())]
    SpecFile {
        /// The file.
        path: PathBuf,
        /// Why its document was not applied.
        #[source]
        source: SpecRejected,
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
    Deciders(#[source] io::Error),
    /// The election thread ended before the shutdown: nothing renews the
    /// lease, so the controller stops.
    #[error("the election thread ended before the shutdown")]
    ElectionEnded,
    /// A deciding thread ended before the shutdown: the models it decided
    /// have no decider, so the controller stops and gives the lead up.
    #[error("a deciding thread ended before the shutdown")]
    DeciderEnded,
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
        Self::bind_with(config, Catalog::builtin()).await
    }

    /// Validates the configuration, installs the process crypto provider,
    /// reads and parses every listener's TLS material, binds every address,
    /// opens the lease's place and builds the store. Nothing is served yet
    /// and no lease is taken, so a refusal arrives before a client can
    /// connect. A policy document names the stages `catalog` holds: the
    /// stock build's and the ones the host added. One that needs the
    /// lifecycle's calls ([`crate::Lifecycle::needs_lifecycle`]) is refused
    /// as not in this build: the controller makes the membership call
    /// alone.
    pub async fn bind_with(config: Config, catalog: Catalog) -> Result<Self, Error> {
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
        // The store makes the membership call and no served line makes
        // another: a stage that needs them is refused by name, never run on
        // state nothing keeps.
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
        let (standing, stood) = watch::channel(Standing::default());
        let (phase, readiness) = readiness::channel(elector.fence(), stood.clone());
        let journal = replication::journaled(&config.election.lease);
        let shared = Shared::new(
            config.controller_id.clone(),
            address.unwrap_or_default(),
            &elector,
            stood,
            journal,
            Arc::clone(&catalog),
            config.decide_shards,
        )
        .map_err(Error::Deciders)?;
        let shared = Arc::new(shared);
        let probe = Arc::new(EngineProbe::new().map_err(Error::Probe)?);
        Ok(Self {
            config,
            catalog,
            phase,
            readiness,
            metrics: Arc::new(metrics::Metrics::new()),
            probe,
            elector,
            standing,
            shared,
            spec_file: None,
            listeners,
            admin,
            admin_addr,
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
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// Takes the controller's policy spec from the file at `path` (design
    /// 4.6): the document is read and applied now, as generation 1 or under
    /// the number its `generation` line writes, and the control scope
    /// answers `GetPolicySpec` with it. [`Self::serve`] then watches the
    /// file: a changed one is applied as the next generation, or under the
    /// number it writes, and one that does not run is rejected whole, with
    /// the running generation kept.
    ///
    /// A file that cannot be read, or a document the validator refuses, is
    /// refused here, before anything serves. The document names the stages
    /// the bind was handed ([`Controller::catalog`]), a host's own among
    /// them. A controller that takes no spec file serves no spec call,
    /// watches nothing and renders no spec series.
    pub async fn spec_file(&mut self, path: impl Into<PathBuf>) -> Result<(), Error> {
        let path = path.into();
        let generations = Arc::clone(&self.shared.generations);
        let read = {
            let path = path.clone();
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
            generation = self.shared.generations.generation(),
            "spec file"
        );
        self.metrics.spec(Arc::clone(&self.shared.generations));
        Ok(())
    }

    /// The store the ingestion drivers and the scopes' calls feed and read.
    pub fn store(&self) -> Arc<MemoryStore> {
        Arc::clone(&self.shared.store)
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

    /// Serves until `shutdown` resolves, then drains. The listeners answer
    /// from the start; the load driver polls the store's engines from the
    /// first tick (idle while the store has no rows); the election thread
    /// takes its first look at the lease, and the state turns
    /// [`State::Leader`] or [`State::Follower`] and the health service's
    /// overall status `SERVING` once that look has landed. At the shutdown
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
    /// of the run loop and after every step of the election, so a fence that
    /// closes between two steps reads `SERVING` until the next; a
    /// registration call reads the fence itself and is refused in that
    /// window. An open watch of the rows follows the same telling: it ends
    /// `ABORTED` when this replica stops leading, an idle one at the next
    /// step of the election. A lead can open between two steps, and a watch
    /// admitted before the next one is ended by any telling that finds no
    /// leader here, the drain's first.
    ///
    /// A listener, an election thread or a deciding thread that ends before
    /// the shutdown ends the serve with its error, the other listeners
    /// closed.
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
        let Self {
            config,
            phase,
            readiness,
            metrics,
            probe,
            elector,
            standing,
            shared,
            spec_file,
            listeners,
            admin,
            ..
        } = self;
        let store = Arc::clone(&shared.store);
        // The listeners' close is one word, and an open decide stream hears
        // it too: the word is `Shared`'s.
        let stop = shared.stop.clone();
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
        let teller = tell_role(
            readiness.clone(),
            stop.subscribe(),
            doors,
            shared.role.clone(),
        );
        tasks.spawn(async move {
            teller.await;
            Ok(())
        });
        let driver = Arc::new(LoadDriver::new(
            Arc::clone(&store),
            probe,
            LoadDriverConfig {
                poll_interval: config.load_poll_interval,
                ..LoadDriverConfig::default()
            },
        ));
        let loads = driver.stats();
        let driver_stop = stop.subscribe();
        tasks.spawn(async move {
            driver.run(driver_stop).await;
            Ok(())
        });
        // The presence sweep: what a gateway held is let go once its grace
        // period ended with no new `Hello`.
        let gateways = Arc::clone(&shared.gateways);
        let sweep_stop = stop.subscribe();
        tasks.spawn(async move {
            gateways.run(sweep_stop).await;
            Ok(())
        });
        if let Some(file) = spec_file {
            let file_stop = stop.subscribe();
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
                true = told => Err(Error::DeciderEnded),
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
        let mut election = election::start(elector, Arc::clone(&store), standing)
            .map_err(Error::ElectionThread)?;
        // A gateway's term floor above this replica's term reaches the
        // election through its door.
        shared.floors.open(election.door());

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
                for reporter in &reporters {
                    reporter
                        .set_service_status("", ServingStatus::Serving)
                        .await;
                }
                phase.send_replace(Phase::Serving);
                info!(
                    controller_id = %config.controller_id,
                    role = readiness.state().as_str(),
                    membership_epoch = store.epoch(),
                    "ready"
                );
                tokio::select! {
                    () = &mut shutdown => Leave::Signal,
                    joined = tasks.join_next() => Leave::Task(joined),
                    () = election.ended() => Leave::Election,
                    error = &mut fault => Leave::Task(Some(Ok(Err(error)))),
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
/// and until the next step such a watch stands under the word that nobody
/// leads here. The word told again ends it, the drain's included; told at
/// the change alone, the drain would leave that watch open to the end of
/// the grace period.
async fn tell_role(
    mut role: Readiness,
    mut stop: watch::Receiver<bool>,
    doors: Vec<HealthReporter>,
    told: watch::Sender<bool>,
) {
    let (mut last, mut left) = (None, false);
    loop {
        let leads = !left && role.state() == State::Leader;
        // The watches first: nothing is awaited between the read of the
        // fence and their word.
        if leads {
            told.send_if_modified(|was| !std::mem::replace(was, true));
        } else {
            told.send_replace(false);
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
            moved = role.moved() => left = moved.is_none(),
        }
    }
}

#[cfg(test)]
mod tests {
    //! What no case outside the module reaches: a serve whose election
    //! thread is gone on its own, one whose deciding thread unwound, and a
    //! serve that leaves by a listener's error.

    use std::{
        future::pending,
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };

    use axum::serve::Listener as _;
    use tokio::{
        net::{TcpListener, TcpStream},
        sync::{oneshot, watch},
        task::JoinSet,
        time::timeout,
    };
    use tonic::codegen::tokio_stream::StreamExt as _;

    use super::{
        config::tests::valid,
        election::Standing,
        readiness::{self, Phase},
        tell_role, Controller, Error,
    };
    use crate::{
        elector::{
            ElectError, ElectionSettings, Elector, LeaseBackend, LeaseRecord, MemoryLease, Tick,
            Written,
        },
        store::Kinds,
    };

    /// A lease that stores `writes` writes: the thread that steps it panics
    /// at the next.
    struct GivesWay {
        lease: MemoryLease,
        writes: u32,
    }

    impl LeaseBackend for GivesWay {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<Written, ElectError> {
            assert!(self.writes > 0, "the lease gave way");
            self.writes -= 1;
            self.lease.write(expected, record)
        }

        fn place(&self) -> String {
            self.lease.place()
        }
    }

    /// Every socket the controller accepts sends at once, read from the
    /// socket itself: one accepted by each gRPC listener as bound, and one
    /// by the admin listener. A socket accepted with no option set does not.
    #[tokio::test]
    async fn every_accepted_socket_sends_at_once() {
        let plain = TcpListener::bind("127.0.0.1:0").await.expect("a port");
        let addr = plain.local_addr().expect("its address");
        let dialed = TcpStream::connect(addr).await.expect("a connection");
        let (socket, _) = plain.accept().await.expect("accepted");
        assert!(
            !socket.nodelay().expect("the option reads"),
            "the kernel's default holds small writes"
        );
        drop(dialed);

        let mut controller = Controller::bind(valid()).await.expect("the listeners bind");
        assert!(!controller.listeners.is_empty());
        for bound in &mut controller.listeners {
            let dialed = TcpStream::connect(bound.addr).await.expect("a connection");
            let socket = bound
                .listener
                .next()
                .await
                .expect("a socket")
                .expect("accepted");
            assert!(
                socket.nodelay().expect("the option reads"),
                "{}",
                bound.name
            );
            drop(dialed);
        }
        let dialed = TcpStream::connect(controller.admin_addr)
            .await
            .expect("a connection");
        let (socket, _) = controller.admin.accept().await;
        assert!(socket.nodelay().expect("the option reads"), "admin");
        drop(dialed);
    }

    /// The deciding threads are the configuration's count: a model's
    /// requests go to one of that many.
    #[tokio::test]
    async fn the_controller_starts_the_deciding_threads_its_configuration_counts() {
        let mut config = valid();
        config.decide_shards = 3;
        let controller = Controller::bind(config).await.expect("the listeners bind");
        let deciders = &controller.shared.deciders;
        let highest = (0..64)
            .map(|index| deciders.of(&format!("model-{index}")))
            .max();
        assert_eq!(highest, Some(2));
    }

    /// The bind draws its log's first place: two binds of one configuration
    /// start apart and under no term, so a place of the run before is no
    /// place of this run.
    #[tokio::test]
    async fn the_bind_draws_its_logs_first_place() {
        let mut starts = Vec::new();
        for _ in 0..2 {
            let controller = Controller::bind(valid()).await.expect("the listeners bind");
            starts.push(controller.shared.feed.log.head(Kinds::Watched));
        }
        assert_ne!(starts[0].seq, starts[1].seq, "two runs start apart");
        assert!(starts.iter().all(|start| start.seq != 0 && start.term == 0));
    }

    /// The thread gone at its take, before the serve read ready, and at its
    /// first renewal, after: the serve ends with the error either way, so
    /// the process exits and its supervisor starts it again.
    #[tokio::test]
    async fn a_serve_whose_election_thread_is_gone_ends_with_an_error() {
        let settings = ElectionSettings {
            duration: Duration::from_millis(300),
            renew_period: Duration::from_millis(20),
            renew_deadline: Duration::from_millis(200),
        };
        for writes in [0, 1] {
            let mut controller = Controller::bind(valid()).await.expect("the listeners bind");
            controller.elector = Elector::new(
                Box::new(GivesWay {
                    lease: MemoryLease::new(),
                    writes,
                }),
                "controller-0@127.0.0.1:1",
                settings,
            )
            .expect("the settings are in order");
            let served = timeout(Duration::from_secs(10), controller.serve(pending()))
                .await
                .expect("the serve ends within ten seconds");
            assert!(
                matches!(served, Err(Error::ElectionEnded)),
                "a lease that gave way after {writes} writes: {served:?}"
            );
        }
    }

    /// A deciding thread that unwound leaves its models with no decider:
    /// the serve ends with the error, told before the serve began, before
    /// it read ready or after, so the process exits and its supervisor
    /// starts it again. A word told before the serve began is one its
    /// receiver was made after: the serve reads it by its value.
    #[tokio::test]
    async fn a_serve_whose_deciding_thread_unwound_ends_with_an_error() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        enum Told {
            BeforeTheServe,
            BeforeReady,
            AfterReady,
        }
        for told in [Told::BeforeTheServe, Told::BeforeReady, Told::AfterReady] {
            let controller = Controller::bind(valid()).await.expect("the listeners bind");
            let shared = Arc::clone(&controller.shared);
            let mut phase = controller.phase.subscribe();
            if told == Told::BeforeTheServe {
                shared.deciders.tell_unwound();
            }
            let unwinds = async move {
                if told == Told::AfterReady {
                    let serving = |phase: &Phase| *phase == Phase::Serving;
                    assert!(phase.wait_for(serving).await.is_ok(), "the serve is ready");
                }
                if told != Told::BeforeTheServe {
                    shared.deciders.tell_unwound();
                }
            };
            let (served, ()) = timeout(Duration::from_secs(20), async {
                tokio::join!(controller.serve(pending()), unwinds)
            })
            .await
            .expect("the serve ends within twenty seconds");
            assert!(
                matches!(served, Err(Error::DeciderEnded)),
                "told {told:?}: {served:?}"
            );
        }
    }

    /// A lease that takes `slow` to store a release.
    struct ReleasesSlowly {
        lease: MemoryLease,
        slow: Duration,
    }

    impl LeaseBackend for ReleasesSlowly {
        fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
            self.lease.read()
        }

        fn write(
            &mut self,
            expected: Option<&LeaseRecord>,
            record: &LeaseRecord,
        ) -> Result<Written, ElectError> {
            if record.holder.is_empty() {
                thread::sleep(self.slow);
            }
            self.lease.write(expected, record)
        }

        fn place(&self) -> String {
            self.lease.place()
        }
    }

    /// An error's leave waits for the release as a signal's does: a serve
    /// that leaves by a listener's error while it leads returns that error
    /// once its release is stored, not before. By order alone: the store is
    /// slow to store a release, and the lease is free when the serve has
    /// returned.
    #[tokio::test]
    async fn a_serve_that_leaves_by_an_error_while_it_leads_has_released_its_lease() {
        let mut lease = MemoryLease::new();
        let mut config = valid();
        // Room for the slow release and a loaded host inside the grace
        // period: the serve waits for the release no longer than that.
        config.shutdown_grace = Duration::from_secs(30);
        let mut controller = Controller::bind(config).await.expect("the listeners bind");
        controller.elector = Elector::new(
            Box::new(ReleasesSlowly {
                lease: lease.clone(),
                slow: Duration::from_millis(300),
            }),
            "controller-0@127.0.0.1:1",
            ElectionSettings {
                duration: Duration::from_millis(300),
                renew_period: Duration::from_millis(20),
                renew_deadline: Duration::from_millis(200),
            },
        )
        .expect("the settings are in order");
        let fence = controller.elector.fence();
        let (fail, failed) = oneshot::channel::<()>();
        let serve = controller.serve_until(pending(), async move {
            let _ = failed.await;
            Error::Task("a listener failed".to_owned())
        });
        let fault = async {
            // The take lands, then the listener fails.
            while fence.term() == 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            fail.send(()).expect("the serve waits for its fault");
        };
        let (served, ()) = timeout(Duration::from_secs(60), async {
            tokio::join!(serve, fault)
        })
        .await
        .expect("the serve ends within its grace period");
        assert!(matches!(served, Err(Error::Task(_))), "{served:?}");
        let record = lease.read().expect("a read").expect("a record");
        assert!(
            record.holder.is_empty(),
            "the lease is released when the serve has returned: {record:?}"
        );
    }

    /// The largest grace period a caller can set is one the clock cannot
    /// add to the present. A serve under it leaves as under any other, by a
    /// signal and by a listener's error: it waits for the release, and the
    /// lease is free when the serve has returned.
    #[tokio::test]
    async fn a_serve_under_a_grace_period_the_clock_cannot_hold_leaves_with_its_lease_released() {
        for by_error in [false, true] {
            let mut lease = MemoryLease::new();
            let mut config = valid();
            config.shutdown_grace = Duration::MAX;
            let mut controller = Controller::bind(config).await.expect("the listeners bind");
            controller.elector = Elector::new(
                Box::new(ReleasesSlowly {
                    lease: lease.clone(),
                    slow: Duration::from_millis(500),
                }),
                "controller-0@127.0.0.1:1",
                ElectionSettings {
                    duration: Duration::from_millis(300),
                    renew_period: Duration::from_millis(20),
                    renew_deadline: Duration::from_millis(200),
                },
            )
            .expect("the settings are in order");
            let fence = controller.elector.fence();
            let (leave, left) = oneshot::channel::<()>();
            let serve = async {
                if by_error {
                    controller
                        .serve_until(pending(), async move {
                            let _ = left.await;
                            Error::Task("a listener failed".to_owned())
                        })
                        .await
                } else {
                    controller
                        .serve(async move {
                            let _ = left.await;
                        })
                        .await
                }
            };
            let told = async {
                // The take lands, then the serve is told to leave.
                while fence.term() == 0 {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                leave.send(()).expect("the serve waits for its leave");
            };
            let (served, ()) =
                timeout(Duration::from_secs(60), async { tokio::join!(serve, told) })
                    .await
                    .expect("the serve ends");
            if by_error {
                assert!(matches!(served, Err(Error::Task(_))), "{served:?}");
            } else {
                assert!(served.is_ok(), "{served:?}");
            }
            let record = lease.read().expect("a read").expect("a record");
            assert!(
                record.holder.is_empty(),
                "the lease is released when the serve has returned (by an error: {by_error}): {record:?}"
            );
        }
    }

    /// The role's one teller over a candidate's own fence: it says that the
    /// replica leads once it does, that it does not at the step after a
    /// loss, and at the stop that it left. The candidate is alone over its
    /// lease, so each take leads at once.
    #[tokio::test]
    async fn the_role_is_told_at_each_change_and_as_left_at_the_stop() {
        let mut elector = Elector::alone("controller-0@127.0.0.1:1", ElectionSettings::default())
            .expect("the default settings are in order");
        let (steps, stepped) = watch::channel(Standing::default());
        let (phase, readiness) = readiness::channel(elector.fence(), stepped);
        let (stop, stopped) = watch::channel(false);
        let (told, mut role) = watch::channel(false);
        let mut teller = JoinSet::new();
        teller.spawn(tell_role(readiness, stopped, Vec::new(), told));
        let mut told_as = async |leads: bool| {
            let reached =
                timeout(Duration::from_secs(5), role.wait_for(|told| *told == leads)).await;
            assert!(
                matches!(reached, Ok(Ok(_))),
                "the role was not told as {leads}"
            );
        };

        // Serving, and the lease taken: the step after it tells the role.
        phase.send_replace(Phase::Serving);
        assert!(matches!(
            elector.tick(Instant::now()),
            Ok(Tick::Acquired { .. })
        ));
        steps.send_replace(Standing::default());
        told_as(true).await;
        // The lease let go with no stop: told at the next step.
        assert!(elector.release().expect("a release"));
        steps.send_replace(Standing::default());
        told_as(false).await;
        // Taken again, then the stop: the replica left, and the teller ends.
        assert!(matches!(
            elector.tick(Instant::now()),
            Ok(Tick::Acquired { .. })
        ));
        steps.send_replace(Standing::default());
        told_as(true).await;
        stop.send_replace(true);
        told_as(false).await;
        assert!(
            teller.join_next().await.is_some(),
            "the teller ends at the stop"
        );
    }

    /// A step that finds the leader standing tells the watches nothing: the
    /// word of leading is sent at its change alone, so no watch's task wakes
    /// once a renew period.
    #[tokio::test]
    async fn a_step_under_a_standing_leader_tells_the_watches_nothing() {
        async fn told_as(role: &mut watch::Receiver<bool>, leads: bool) {
            let reached =
                timeout(Duration::from_secs(5), role.wait_for(|told| *told == leads)).await;
            assert!(
                matches!(reached, Ok(Ok(_))),
                "the role was not told as {leads}"
            );
        }
        let mut elector = Elector::alone("controller-0@127.0.0.1:1", ElectionSettings::default())
            .expect("the default settings are in order");
        let (steps, stepped) = watch::channel(Standing::default());
        let (phase, readiness) = readiness::channel(elector.fence(), stepped);
        let (_stop, stopped) = watch::channel(false);
        let (told, mut role) = watch::channel(false);
        let mut teller = JoinSet::new();
        teller.spawn(tell_role(readiness, stopped, Vec::new(), told));

        phase.send_replace(Phase::Serving);
        assert!(matches!(
            elector.tick(Instant::now()),
            Ok(Tick::Acquired { .. })
        ));
        steps.send_replace(Standing::default());
        told_as(&mut role, true).await;
        // Three steps under the standing leader, each taken by the teller:
        // turns of the runtime and no clock.
        for _ in 0..3 {
            steps.send_replace(Standing::default());
            for _ in 0..100 {
                tokio::task::yield_now().await;
            }
        }
        assert!(
            !role.has_changed().expect("the teller holds the word"),
            "a step under a standing leader told the word again"
        );
        // The teller took those steps: the one after the lead's loss is told.
        assert!(elector.release().expect("a release"));
        steps.send_replace(Standing::default());
        told_as(&mut role, false).await;
    }

    /// A serve whose lead opened with no step of the election since: the
    /// take of a lease with no holder to wait for leads one renew deadline
    /// after its write, between two steps. The fence admits a watch from
    /// that instant, and the word such a watch finds still says that nobody
    /// leads here.
    struct Unstepped {
        elector: Elector,
        steps: watch::Sender<Standing>,
        phase: watch::Sender<Phase>,
        /// The role's word, as a watch admitted now holds it.
        watch: watch::Receiver<bool>,
        /// The serve's stop, never said, and the teller's task.
        _serve: (watch::Sender<bool>, JoinSet<()>),
    }

    impl Unstepped {
        async fn new() -> Self {
            let settings = ElectionSettings {
                duration: Duration::from_millis(1500),
                renew_period: Duration::from_millis(100),
                renew_deadline: Duration::from_millis(500),
            };
            let mut elector = Elector::new(
                Box::new(MemoryLease::new()),
                "controller-0@127.0.0.1:1",
                settings,
            )
            .expect("the settings are in order");
            let (steps, stepped) = watch::channel(Standing::default());
            let (phase, readiness) = readiness::channel(elector.fence(), stepped);
            let (stop, stopped) = watch::channel(false);
            let (told, role) = watch::channel(false);
            let mut teller = JoinSet::new();
            teller.spawn(tell_role(readiness, stopped, Vec::new(), told));

            // The take, and a chain of renewals ahead of the clock, each
            // just short of the deadline the one before it stamped: the
            // lead opens one renew deadline after the take's write whatever
            // was renewed since, and the fence stays open seconds past that
            // instant, so the read below needs no narrow window. The step's
            // telling finds a follower.
            phase.send_replace(Phase::Serving);
            let taken = Instant::now();
            assert_eq!(
                elector.tick(taken).expect("the take"),
                Tick::Acquired { term: 1 }
            );
            let mut renewed = taken;
            for _ in 0..10 {
                renewed += settings.renew_deadline - Duration::from_millis(10);
                assert_eq!(
                    elector.tick(renewed).expect("the renewal"),
                    Tick::Renewed { term: 1 }
                );
            }
            steps.send_replace(Standing::default());
            // The lead opens by itself and the fence admits a watch. No step
            // came since: the word the watch finds says that nobody leads.
            tokio::time::sleep(settings.renew_deadline).await;
            assert_eq!(elector.fence().leader_term(Instant::now()), Some(1));
            let mut watch = role.clone();
            assert!(!*watch.borrow_and_update());
            Self {
                elector,
                steps,
                phase,
                watch,
                _serve: (stop, teller),
            }
        }

        /// The watch is told again that nobody leads here.
        async fn hears(&mut self, what: &str) {
            let heard = timeout(Duration::from_secs(5), self.watch.changed()).await;
            assert!(
                matches!(heard, Ok(Ok(()))),
                "{what} was not told to a watch that stood under the word: {heard:?}"
            );
            assert!(!*self.watch.borrow());
        }
    }

    /// A watch admitted before any step told the role as leading stands
    /// under the word that this replica does not lead. The drain tells that
    /// word again, and the watch hears it.
    #[tokio::test]
    async fn a_watch_admitted_before_the_role_was_told_as_leading_hears_the_drain() {
        let mut serve = Unstepped::new().await;
        serve.phase.send_replace(Phase::Draining);
        serve.hears("the drain").await;
    }

    /// So does the step that finds the lead gone, with no stop.
    #[tokio::test]
    async fn a_watch_admitted_before_the_role_was_told_as_leading_hears_the_loss_of_its_lead() {
        let mut serve = Unstepped::new().await;
        assert!(serve.elector.release().expect("a release"));
        serve.steps.send_replace(Standing::default());
        serve.hears("the lead's loss").await;
    }
}
