//! The standalone controller's run loop (design 2, 6, 8 and 10): the store,
//! the load driver over it, the election, the gRPC listeners with their
//! scopes, the service and the health service on each, the admin listener,
//! readiness and the drain. The service answers `Leader` and a gateway's
//! `Session`, and the run loop sweeps the gateways that are gone; the
//! scopes' other call bodies arrive with the gateway, ingestion and control
//! scope PRs, the other ingestion drivers with their move out of the
//! gateway. In the
//! `elected` mode the gateway runs this same loop inside its process.
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

pub mod admin;
pub mod config;
pub mod election;
mod gateways;
pub mod metrics;
mod principal;
pub mod readiness;
mod service;
mod session;
pub mod tls;

use std::{future::Future, io, net::SocketAddr, pin::pin, sync::Arc};

use tokio::{
    net::TcpListener,
    sync::watch,
    task::{JoinError, JoinSet},
    time::timeout,
};
use tonic::transport::{server::TcpIncoming, Server};
use tonic_health::ServingStatus;
use tracing::{error, info, warn};

pub use self::{
    admin::{Version, VERSION},
    config::{Config, ConfigError, Listener, Scope, Tls},
    election::{Election, LeaseError, LeasePlace},
    readiness::{Readiness, State},
    tls::TlsError,
};
use self::{
    election::Standing,
    principal::Layer,
    readiness::Phase,
    service::{Service, Shared},
};
use crate::{
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
/// credentialed scopes), as the health service names it. A listener reports
/// it `SERVING` while its replica leads, its door is open and its scope's
/// work can be done there: on the control scope a registration is taken, on
/// the gateway scope a decision is answered, on the ingestion scope a push is
/// taken, on the replication scope a standby can follow. Any other listener
/// reports it `NOT_SERVING`; `Leader` is every scope's and moves no
/// listener's word.
pub const SERVICE_NAME: &str = "smg.policy.v1.PolicyController";

/// A controller with its listeners bound and nothing served yet.
pub struct Controller {
    config: Config,
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
    listeners: Vec<Bound>,
    admin: TcpListener,
    admin_addr: SocketAddr,
}

/// Why the serve left its wait.
enum Leave {
    /// The shutdown future resolved.
    Signal,
    /// A listener's task ended first.
    Task(Option<Result<Result<(), Error>, JoinError>>),
    /// The election thread ended first.
    Election,
}

/// A gRPC listener, bound, with its server configured (TLS parsed).
struct Bound {
    name: String,
    scopes: Vec<Scope>,
    addr: SocketAddr,
    listener: TcpListener,
    server: Server,
    tls: bool,
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
    /// The lease's place could not be opened, or the elector refused its
    /// identity.
    #[error("the election's lease: {0}")]
    Lease(#[source] LeaseError),
    /// The election thread could not be started.
    #[error("the election thread: {0}")]
    ElectionThread(#[source] io::Error),
    /// The election thread ended before the shutdown: nothing renews the
    /// lease, so the controller stops.
    #[error("the election thread ended before the shutdown")]
    ElectionEnded,
    /// A listener returned before the shutdown signal.
    #[error("a listener stopped before the shutdown")]
    Stopped,
    /// A listener task ended abnormally.
    #[error("a listener task ended abnormally: {0}")]
    Task(String),
}

impl Controller {
    /// Validates the configuration, installs the process crypto provider,
    /// reads and parses every listener's TLS material, binds every address,
    /// opens the lease's place and builds the store. Nothing is served yet
    /// and no lease is taken, so a refusal arrives before a client can
    /// connect.
    pub async fn bind(config: Config) -> Result<Self, Error> {
        config.validate()?;
        tls::install_crypto_provider();
        let mut listeners = Vec::with_capacity(config.listeners.len());
        for listener in &config.listeners {
            let name = listener.name.clone();
            let mut server = Server::builder();
            let mut mode = "off";
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
                mode = if material.mutual() {
                    "mutual"
                } else {
                    "server"
                };
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
                listener: socket,
                server,
                tls: listener.tls.is_some(),
                layer: Layer::Off,
            });
        }
        let admin = TcpListener::bind(config.admin)
            .await
            .map_err(Error::Admin)?;
        let admin_addr = admin.local_addr().map_err(Error::Admin)?;
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
        let shared = Arc::new(Shared::new(
            config.controller_id.clone(),
            address.unwrap_or_default(),
            &elector,
            stood,
            Arc::new(MemoryStore::new()),
        ));
        let probe = Arc::new(EngineProbe::new().map_err(Error::Probe)?);
        Ok(Self {
            config,
            phase,
            readiness,
            metrics: Arc::new(metrics::Metrics::new()),
            probe,
            elector,
            standing,
            shared,
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
        let verifies = self.config.listeners.iter().any(|l| {
            l.name == listener && l.tls.as_ref().is_some_and(|tls| tls.client_ca.is_some())
        });
        if verifies {
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
    /// streams with them.
    ///
    /// Every listener's health service reports the overall status (service
    /// `""`) as the readiness and [`SERVICE_NAME`] as `NOT_SERVING`. A
    /// listener or an election thread that ends before the shutdown ends
    /// the serve with its error, the other listeners closed.
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
            listeners,
            admin,
            ..
        } = self;
        let store = Arc::clone(&shared.store);
        let (stop, _held) = watch::channel(false);
        let mut tasks: JoinSet<Result<(), Error>> = JoinSet::new();
        let mut reporters = Vec::with_capacity(listeners.len());
        for bound in listeners {
            let (reporter, health) = tonic_health::server::health_reporter();
            reporter
                .set_service_status("", ServingStatus::NotServing)
                .await;
            reporter
                .set_service_status(SERVICE_NAME, ServingStatus::NotServing)
                .await;
            let routes = Service::new(bound.scopes, Arc::clone(&shared)).routes(bound.layer);
            let mut stop = stop.subscribe();
            let name = bound.name;
            let incoming = TcpIncoming::from(bound.listener);
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
        let app = admin::router(admin::Admin {
            readiness: readiness.clone(),
            metrics: Arc::clone(&metrics),
            loads,
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
        let deadline = tokio::time::Instant::now() + config.shutdown_grace;
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
        if tokio::time::timeout_at(deadline, election.ended())
            .await
            .is_err()
        {
            warn!(
                controller_id = %config.controller_id,
                "the election thread outlived the grace period; a lease it holds expires by its duration"
            );
        }
        if result.is_ok() {
            info!(controller_id = %config.controller_id, "stopped");
        }
        result
    }
}

#[cfg(test)]
mod tests {
    //! What no case outside the module reaches: a serve whose election
    //! thread is gone on its own, and a serve that leaves by a listener's
    //! error.

    use std::{future::pending, thread, time::Duration};

    use tokio::{sync::oneshot, time::timeout};

    use super::{config::tests::valid, Controller, Error};
    use crate::elector::{
        ElectError, ElectionSettings, Elector, LeaseBackend, LeaseRecord, MemoryLease, Written,
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
}
