//! What the steps of a worker workflow ask of the process that runs them.

use std::{fmt, sync::Arc};

use wfaas::{WorkflowError, WorkflowResult};

use crate::{
    connection::{engine::EngineConnector, http_client::WorkerHttpClientCache},
    registry::{Generation, WorkerRegistry},
    settings::RegistrationSettings,
};

/// The host of the worker workflows: the process that keeps the worker
/// registry. The steps of this crate read their host through this trait
/// alone.
pub trait WorkflowHost: Send + Sync + 'static {
    /// What the host's registry keeps of a membership generation.
    type Generation: Generation;

    /// What the host's own steps keep on a registration between them. No step
    /// of this crate reads or writes it, and it is not saved with the
    /// registration.
    type RegistrationNotes: Clone + fmt::Debug + Default + Send + Sync + 'static;

    /// The registry the workflows register workers in and remove them from.
    fn worker_registry(&self) -> &Arc<WorkerRegistry<Self::Generation>>;

    /// The clients a registration reaches its worker with.
    fn worker_client_cache(&self) -> &WorkerHttpClientCache;

    /// What the host fixes for every registration it runs.
    fn registration_settings(&self) -> &RegistrationSettings;

    /// The connector for gRPC probes and the workers a registration builds.
    fn engine_connector(&self) -> Arc<dyn EngineConnector>;
}

/// The host a workflow's data was given, or the workflow's error for a run
/// that was started without one.
pub fn require_host<H>(host: Option<&Arc<H>>) -> WorkflowResult<&Arc<H>> {
    host.ok_or_else(|| WorkflowError::ContextValueNotFound("host".to_string()))
}

/// The hosts of the steps' own cases.
#[cfg(test)]
pub(crate) mod testing {
    use std::{
        net::Ipv4Addr,
        sync::{Arc, LazyLock, OnceLock},
    };

    use tokio::net::TcpListener;

    use super::WorkflowHost;
    use crate::{
        connection::{
            engine::EngineConnector,
            http_client::{WorkerHttpClientCache, WorkerHttpClientSettings},
        },
        settings::{RegistrationSettings, DEFAULT_ZMQ_STARTUP_TIMEOUT_SECS},
        test_support::{CountingConnector, Members, ScriptedConnector, WorkerRegistry},
        worker::HttpPoolConfig,
    };

    /// Settings shared by the test hosts.
    pub(crate) fn registration_settings() -> RegistrationSettings {
        RegistrationSettings {
            upstream_http2: false,
            health_check_timeout_secs: 5,
            startup_timeout_secs: 1,
            startup_delay_secs: 0,
            startup_check_interval_secs: 1,
            model_path: None,
            model_aliases: Default::default(),
            retry: Default::default(),
            retries_enabled: true,
            circuit_breaker: Default::default(),
            circuit_breaker_enabled: true,
            overload_defaults: Default::default(),
            health: Default::default(),
            health_endpoint: "/health".to_string(),
            zmq_startup_timeout_secs: DEFAULT_ZMQ_STARTUP_TIMEOUT_SECS,
            request_timeout_secs: 30,
        }
    }

    /// A host that keeps a registry and nothing else.
    #[derive(Debug)]
    pub(crate) struct RegistryHost {
        pub(crate) worker_registry: Arc<WorkerRegistry>,
    }

    impl WorkflowHost for RegistryHost {
        type Generation = Members;
        type RegistrationNotes = ();

        fn worker_registry(&self) -> &Arc<WorkerRegistry> {
            &self.worker_registry
        }

        fn worker_client_cache(&self) -> &WorkerHttpClientCache {
            // One cache serves every case: this host keeps a registry alone.
            static CACHE: OnceLock<WorkerHttpClientCache> = OnceLock::new();
            CACHE.get_or_init(|| {
                WorkerHttpClientCache::new(WorkerHttpClientSettings {
                    client_identity: None,
                    ca_certificates: Vec::new(),
                    upstream_http2: false,
                    request_timeout_secs: 30,
                    pool_idle_timeout_secs: 0,
                })
            })
        }

        fn registration_settings(&self) -> &RegistrationSettings {
            static SETTINGS: LazyLock<RegistrationSettings> = LazyLock::new(registration_settings);
            &SETTINGS
        }

        fn engine_connector(&self) -> Arc<dyn EngineConnector> {
            Arc::new(CountingConnector::default())
        }
    }

    /// A host for a registration's cases: its own worker clients and
    /// settings (`upstream_http2` decides both), and the connector a case
    /// scripts.
    pub(crate) struct ProbingHost {
        worker_registry: Arc<WorkerRegistry>,
        clients: WorkerHttpClientCache,
        settings: RegistrationSettings,
        connector: Arc<ScriptedConnector>,
    }

    impl ProbingHost {
        pub(crate) fn new(upstream_http2: bool, connector: ScriptedConnector) -> Arc<Self> {
            Self::with_settings(
                RegistrationSettings {
                    upstream_http2,
                    ..registration_settings()
                },
                connector,
            )
        }

        pub(crate) fn with_settings(
            settings: RegistrationSettings,
            connector: ScriptedConnector,
        ) -> Arc<Self> {
            Arc::new(Self {
                worker_registry: Arc::new(WorkerRegistry::new()),
                clients: WorkerHttpClientCache::new(WorkerHttpClientSettings {
                    client_identity: None,
                    ca_certificates: Vec::new(),
                    upstream_http2: settings.upstream_http2,
                    request_timeout_secs: settings.request_timeout_secs,
                    pool_idle_timeout_secs: 0,
                }),
                settings,
                connector: Arc::new(connector),
            })
        }

        /// How many connects the host's connector was asked for.
        pub(crate) fn connects(&self) -> usize {
            self.connector.connects()
        }
    }

    impl WorkflowHost for ProbingHost {
        type Generation = Members;
        type RegistrationNotes = ();

        fn worker_registry(&self) -> &Arc<WorkerRegistry> {
            &self.worker_registry
        }

        fn worker_client_cache(&self) -> &WorkerHttpClientCache {
            &self.clients
        }

        fn registration_settings(&self) -> &RegistrationSettings {
            &self.settings
        }

        fn engine_connector(&self) -> Arc<dyn EngineConnector> {
            let connector: Arc<ScriptedConnector> = Arc::clone(&self.connector);
            connector
        }
    }

    /// A worker-directed client of the crate's own cache, as a step hands one
    /// to a probe.
    pub(crate) fn worker_client() -> Arc<reqwest::Client> {
        ProbingHost::new(false, ScriptedConnector::default())
            .worker_client_cache()
            .get(&HttpPoolConfig::default(), false)
            .unwrap()
    }

    /// An engine's HTTP metadata endpoints as a test serves them: each route
    /// answers every GET with its status and JSON body, any other path a 404.
    /// Returns the base URL; the stub serves for as long as the runtime that
    /// spawned it runs.
    pub(crate) async fn serve_json_routes(
        routes: Vec<(&'static str, http::StatusCode, serde_json::Value)>,
    ) -> String {
        let mut router = axum::Router::new();
        for (path, status, body) in routes {
            router = router.route(
                path,
                axum::routing::get(move || {
                    let body = body.clone();
                    async move { (status, axum::Json(body)) }
                }),
            );
        }
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("a loopback port of the system's choice");
        let addr = listener.local_addr().expect("the bound address");
        #[expect(
            clippy::disallowed_methods,
            reason = "a test fixture: the stub lives as long as the test's runtime"
        )]
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        format!("http://{addr}")
    }
}
