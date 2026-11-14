//! The standalone controller's run loop (design 2, 6, 8 and 10): the store,
//! the load driver over it, the gRPC listeners with their scopes and the
//! health service on each, the admin listener, readiness and the drain. The
//! scopes' call bodies arrive with the gateway, ingestion and control scope
//! PRs; the other ingestion drivers with their move out of the gateway; the
//! election with the electors. In the `elected` mode the gateway runs this
//! same loop inside its process.
//!
//! A controller is bound first and served second ([`Controller::bind`], then
//! [`Controller::serve`]): the bind takes every address (port `0` for an
//! ephemeral one), reads and parses the TLS material and builds the store,
//! so a start refuses before it serves and a test knows its ports; the serve
//! runs the listeners until the shutdown future resolves, then settles,
//! drains and returns.

pub mod admin;
pub mod config;
pub mod metrics;
pub mod readiness;
pub mod tls;

use std::{future::Future, io, net::SocketAddr, pin::pin, sync::Arc};

use tokio::{net::TcpListener, sync::watch, task::JoinSet, time::timeout};
use tonic::transport::{server::TcpIncoming, Server};
use tonic_health::ServingStatus;
use tracing::{error, info, warn};

pub use self::{
    admin::{Version, VERSION},
    config::{Config, ConfigError, Listener, Scope, Tls},
    readiness::{Readiness, State},
    tls::TlsError,
};
use crate::{
    ingest::{
        loads::{LoadDriver, LoadDriverConfig},
        probe::EngineProbe,
    },
    store::{MemoryStore, WorkersStore},
};

/// The protobuf package of the controller's API.
pub const PROTOCOL_PACKAGE: &str = "smg.policy.v1";

/// The gRPC service every scope is a part of (design 8: one service, four
/// credentialed scopes), as the health service names it. Every listener
/// reports it `NOT_SERVING` until the scopes' calls exist.
pub const SERVICE_NAME: &str = "smg.policy.v1.PolicyController";

/// A controller with its listeners bound and nothing served yet.
pub struct Controller {
    config: Config,
    store: Arc<MemoryStore>,
    state: watch::Sender<State>,
    readiness: Readiness,
    metrics: Arc<metrics::Metrics>,
    /// The engines' probe the load driver dials through.
    probe: Arc<EngineProbe>,
    listeners: Vec<Bound>,
    admin: TcpListener,
    admin_addr: SocketAddr,
}

/// A gRPC listener, bound, with its server configured (TLS parsed).
struct Bound {
    name: String,
    scopes: Vec<Scope>,
    addr: SocketAddr,
    listener: TcpListener,
    server: Server,
    tls: bool,
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
    /// A listener returned before the shutdown signal.
    #[error("a listener stopped before the shutdown")]
    Stopped,
    /// A listener task ended abnormally.
    #[error("a listener task ended abnormally: {0}")]
    Task(String),
}

impl Controller {
    /// Validates the configuration, installs the process crypto provider,
    /// reads and parses every listener's TLS material, binds every address
    /// and builds the store. Nothing is served yet, so a refusal arrives
    /// before a client can connect.
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
            });
        }
        let admin = TcpListener::bind(config.admin)
            .await
            .map_err(Error::Admin)?;
        let admin_addr = admin.local_addr().map_err(Error::Admin)?;
        info!(controller_id = %config.controller_id, addr = %admin_addr, "admin listener bound");
        let (state, readiness) = readiness::channel();
        let probe = Arc::new(EngineProbe::new().map_err(Error::Probe)?);
        Ok(Self {
            config,
            store: Arc::new(MemoryStore::new()),
            state,
            readiness,
            metrics: Arc::new(metrics::Metrics::new()),
            probe,
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

    /// The store the ingestion drivers and the scopes' calls feed and read.
    pub fn store(&self) -> Arc<MemoryStore> {
        Arc::clone(&self.store)
    }

    /// A read handle on the controller's state, valid across [`Self::serve`].
    pub fn readiness(&self) -> Readiness {
        self.readiness.clone()
    }

    /// Serves until `shutdown` resolves, then drains: the load driver polls
    /// the store's engines from the first tick (idle while the store has no
    /// rows); the state turns
    /// [`State::Draining`] and the health service's overall status
    /// `NOT_SERVING` at once; the listeners keep accepting for the settle
    /// window, stop accepting, and close within the rest of the grace
    /// period, their streams with them.
    ///
    /// Every listener's health service reports the overall status (service
    /// `""`) as the readiness and [`SERVICE_NAME`] as `NOT_SERVING`. A
    /// listener that fails before the shutdown ends the serve with its
    /// error, the other listeners closed.
    pub async fn serve(self, shutdown: impl Future<Output = ()>) -> Result<(), Error> {
        let Self {
            config,
            store,
            state,
            readiness,
            metrics,
            probe,
            listeners,
            admin,
            ..
        } = self;
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
            let mut stop = stop.subscribe();
            let name = bound.name;
            let incoming = TcpIncoming::from(bound.listener);
            let mut server = bound.server;
            tasks.spawn(async move {
                server
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

        for reporter in &reporters {
            reporter
                .set_service_status("", ServingStatus::Serving)
                .await;
        }
        metrics.set_state(State::Ready);
        state.send_replace(State::Ready);
        info!(controller_id = %config.controller_id, membership_epoch = store.epoch(), "ready");

        let mut shutdown = pin!(shutdown);
        let early = tokio::select! {
            () = &mut shutdown => None,
            joined = tasks.join_next() => Some(joined),
        };

        metrics.set_state(State::Draining);
        state.send_replace(State::Draining);
        for reporter in &reporters {
            reporter
                .set_service_status("", ServingStatus::NotServing)
                .await;
        }
        if let Some(joined) = early {
            let error = match joined {
                Some(Ok(Ok(()))) | None => Error::Stopped,
                Some(Ok(Err(error))) => error,
                Some(Err(join)) => Error::Task(join.to_string()),
            };
            error!(controller_id = %config.controller_id, %error, "a listener left before the shutdown");
            tasks.abort_all();
            while tasks.join_next().await.is_some() {}
            return Err(error);
        }

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
            Ok(result) => {
                info!(controller_id = %config.controller_id, "stopped");
                result
            }
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
    }
}
