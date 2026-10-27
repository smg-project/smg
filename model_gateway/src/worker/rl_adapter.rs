//! Glue between the RL control plane crate and the gateway registry. This
//! file is the whole of coupling touchpoint (a); see `crates/rl/COUPLING.md`.

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use async_trait::async_trait;
use openai_protocol::worker::{ConnectionMode, HttpPoolConfig, RuntimeType};
use smg_rl::{
    resolve_control_url, CallFailure, RankReply, RlControlCalls, RlState, RlWorkerInfo,
    RlWorkerView,
};
use tracing::warn;

use crate::{
    config::RouterConfig,
    routers::grpc::backend_client::BackendClient,
    worker::{registry::WorkerId, Worker, WorkerHttpClientCache, WorkerRegistry},
};

/// Label an engine (or an operator) sets to name the worker's RL control app.
pub const CONTROL_URL_LABEL: &str = "rl.control_url";

/// Read-only registry view for the RL crate.
pub struct RegistryRlView {
    registry: Arc<WorkerRegistry>,
    client_cache: Arc<WorkerHttpClientCache>,
    /// Control clients for gRPC and ZMQ workers, by pool config. The gateway
    /// cache holds weak entries that live only while some HTTP worker uses the
    /// same config, and these workers hold no client of their own, so without
    /// a strong handle here every control call would build a fresh client. A
    /// failed build is kept as well: logged once, reported on each 422.
    /// Pruned in [`RlWorkerView::list`] to the configs still registered.
    control_clients: Mutex<HashMap<HttpPoolConfig, Result<Arc<reqwest::Client>, String>>>,
}

impl RegistryRlView {
    pub fn new(registry: Arc<WorkerRegistry>, client_cache: Arc<WorkerHttpClientCache>) -> Self {
        Self {
            registry,
            client_cache,
            control_clients: Mutex::new(HashMap::new()),
        }
    }

    /// The shared control client for `pool`: same TLS identity and roots as
    /// every upstream client, HTTP/1.1 because the control apps are uvicorn.
    fn control_client(&self, pool: &HttpPoolConfig) -> Result<Arc<reqwest::Client>, String> {
        let mut clients = self
            .control_clients
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(client) = clients.get(pool) {
            return client.clone();
        }
        let client = self.client_cache.get(pool, false);
        if let Err(e) = &client {
            warn!(
                error = %e, ?pool,
                "no HTTP client for RL control endpoints with this pool config"
            );
        }
        clients.insert(pool.clone(), client.clone());
        client
    }

    /// Drop the handles no registered gRPC or ZMQ worker needs anymore.
    fn prune_control_clients(&self, workers: &[Arc<dyn Worker>]) {
        let mut clients = self
            .control_clients
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        clients.retain(|pool, _| {
            workers.iter().any(|w| {
                *w.connection_mode() != ConnectionMode::Http && w.metadata().spec.http_pool == *pool
            })
        });
    }

    /// Where control calls for `worker` go and which client carries them.
    ///
    /// HTTP workers are controlled through themselves, with the client the
    /// gateway negotiated for them. Other transports need an advertised or
    /// operator-supplied `rl.control_url`, resolved against the worker's own
    /// host, and use the shared control client for their pool config.
    fn control_endpoint(
        &self,
        worker: &Arc<dyn Worker>,
    ) -> (Option<String>, Result<Arc<reqwest::Client>, String>) {
        let spec = &worker.metadata().spec;
        if *worker.connection_mode() == ConnectionMode::Http {
            let client = worker
                .http_client_handle_if_initialized()
                .unwrap_or_else(|| Arc::new(worker.http_client().clone()));
            return (Some(worker.base_url().to_string()), Ok(client));
        }
        let url = spec
            .labels
            .get(CONTROL_URL_LABEL)
            .map(String::as_str)
            .filter(|v| !v.trim().is_empty())
            .map(|advertised| resolve_control_url(advertised, worker.base_url()));
        (url, self.control_client(&spec.http_pool))
    }

    fn info(&self, worker: &Arc<dyn Worker>) -> Option<RlWorkerInfo> {
        let id = self.registry.get_id_by_url(worker.url())?;
        let spec = &worker.metadata().spec;
        let (control_url, control_client) = self.control_endpoint(worker);
        // A TokenSpeed engine on the direct ZMQ lane takes the RL routes as
        // control calls on the worker's own connection.
        let control_calls = (*worker.connection_mode() == ConnectionMode::Zmq
            && spec.runtime_type == RuntimeType::TokenSpeed)
            .then(|| {
                Arc::new(EngineConnectionControl {
                    worker: Arc::clone(worker),
                }) as Arc<dyn RlControlCalls>
            });
        Some(RlWorkerInfo {
            id: id.as_str().to_string(),
            url: worker.url().to_string(),
            base_url: worker.base_url().to_string(),
            api_key: worker.api_key().cloned(),
            model_id: worker.model_id().to_string(),
            runtime: spec.runtime_type,
            worker_type: *worker.worker_type(),
            connection_mode: *worker.connection_mode(),
            status: worker.status(),
            is_dp_aware: worker.is_dp_aware(),
            dp_size: worker.dp_size(),
            labels: spec.labels.clone(),
            control_url,
            control_client,
            control_calls,
        })
    }
}

/// The RL routes as control calls on a direct-ZMQ worker's own engine
/// connection, reached through the worker when a call is made (it may still
/// be handshaking when the registry is read).
struct EngineConnectionControl {
    worker: Arc<dyn Worker>,
}

impl fmt::Debug for EngineConnectionControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EngineConnectionControl")
            .field("worker", &self.worker.url())
            .finish()
    }
}

#[async_trait]
impl RlControlCalls for EngineConnectionControl {
    async fn call(
        &self,
        method: &str,
        body: Option<serde_json::Value>,
        timeout: Duration,
    ) -> Result<Vec<RankReply>, CallFailure> {
        let client = self
            .worker
            .get_backend_client()
            .await
            .map_err(|e| CallFailure::Failed(format!("the engine is not connected: {e}")))?
            .ok_or_else(|| {
                CallFailure::Failed("the worker has no engine connection".to_string())
            })?;
        let BackendClient::Zmq(client) = client.as_ref() else {
            return Err(CallFailure::Failed(
                "the worker is not a direct-ZMQ worker".to_string(),
            ));
        };
        match client.rl_control_call(method, body, timeout).await {
            Ok(replies) => Ok(replies
                .into_iter()
                .map(|(success, message)| RankReply { success, message })
                .collect()),
            Err(status) if status.code() == tonic::Code::DeadlineExceeded => {
                Err(CallFailure::Timeout)
            }
            Err(status) => Err(CallFailure::Failed(status.message().to_string())),
        }
    }
}

impl RlWorkerView for RegistryRlView {
    fn list(&self) -> Vec<RlWorkerInfo> {
        let workers = self.registry.get_all();
        let infos = workers.iter().filter_map(|w| self.info(w)).collect();
        self.prune_control_clients(&workers);
        infos
    }

    fn get(&self, id: &str) -> Option<RlWorkerInfo> {
        let worker = self.registry.get(&WorkerId::from_string(id.to_string()))?;
        self.info(&worker)
    }
}

/// Build the RL state when `config.rl.enabled`; `None` otherwise, so the
/// disabled path constructs nothing.
pub fn build_rl_state(
    registry: &Arc<WorkerRegistry>,
    client_cache: &Arc<WorkerHttpClientCache>,
    config: &RouterConfig,
) -> Option<Arc<RlState>> {
    if !config.rl.enabled {
        return None;
    }
    let view = Arc::new(RegistryRlView::new(
        Arc::clone(registry),
        Arc::clone(client_cache),
    ));
    Some(Arc::new(RlState::new(view, config.rl.clone())))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::ConnectionMode;

    use super::*;
    use crate::{config::RouterConfig, worker::BasicWorkerBuilder};

    fn view_with(config: &RouterConfig, workers: Vec<Arc<dyn Worker>>) -> RegistryRlView {
        let registry = Arc::new(WorkerRegistry::new());
        for w in workers {
            registry.register(w);
        }
        let cache = Arc::new(WorkerHttpClientCache::new(config));
        RegistryRlView::new(registry, cache)
    }

    fn view_over(workers: Vec<Arc<dyn Worker>>) -> RegistryRlView {
        view_with(&RouterConfig::default(), workers)
    }

    /// A gRPC worker whose engine advertised a wildcard-bound control app.
    fn grpc_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .connection_mode(ConnectionMode::Grpc)
                .label("rl.control_url", "http://0.0.0.0:40100")
                .build(),
        )
    }

    /// Control calls to an HTTP worker use the client the gateway negotiated
    /// for it (HTTP version, TLS identity, pool tuning), not a second one.
    #[test]
    fn http_worker_controls_itself_with_its_negotiated_client() {
        let client = Arc::new(crate::clients::http_client());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://engine:30000")
                .connection_mode(ConnectionMode::Http)
                .http_client(Arc::clone(&client))
                .build(),
        );
        let info = view_over(vec![worker]).list().pop().expect("one worker");
        assert_eq!(info.control_url.as_deref(), Some("http://engine:30000"));
        assert!(Arc::ptr_eq(&info.control_client.expect("client"), &client));
    }

    #[test]
    fn grpc_worker_with_a_label_gets_a_control_endpoint_and_a_client() {
        let info = view_over(vec![grpc_worker("grpc://10.0.0.5:30000")])
            .list()
            .pop()
            .expect("one worker");
        assert_eq!(
            info.control_url.as_deref(),
            Some("http://10.0.0.5:40100"),
            "wildcard bind host resolves to the worker host"
        );
        assert!(info.control_client.is_ok());
    }

    #[test]
    fn grpc_worker_without_a_label_has_no_control_endpoint() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://engine:30000")
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        );
        let info = view_over(vec![worker]).list().pop().expect("one worker");
        assert!(info.control_url.is_none());
    }

    #[test]
    fn grpc_worker_with_a_blank_label_has_no_control_endpoint() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://engine:30000")
                .connection_mode(ConnectionMode::Grpc)
                .label("rl.control_url", "  ")
                .build(),
        );
        let info = view_over(vec![worker]).list().pop().expect("one worker");
        assert!(info.control_url.is_none());
    }

    /// A TokenSpeed engine on the direct ZMQ lane is controlled over its own
    /// connection; another runtime's ZMQ worker and a gRPC worker are not.
    #[test]
    fn a_tokenspeed_zmq_worker_is_controlled_over_its_engine_connection() {
        let zmq = |url: &str, runtime: RuntimeType| -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new(url)
                    .connection_mode(ConnectionMode::Zmq)
                    .runtime_type(runtime)
                    .build(),
            )
        };
        let infos = view_over(vec![
            zmq("ipc:///tmp/smg-zmq/ts", RuntimeType::TokenSpeed),
            zmq("ipc:///tmp/smg-zmq/vllm", RuntimeType::Vllm),
            grpc_worker("grpc://a:1"),
        ])
        .list();
        let calls = |url: &str| {
            infos
                .iter()
                .find(|info| info.url == url)
                .expect("registered")
                .control_calls
                .is_some()
        };
        assert!(calls("ipc:///tmp/smg-zmq/ts"));
        assert!(!calls("ipc:///tmp/smg-zmq/vllm"));
        assert!(!calls("grpc://a:1"));
    }

    #[test]
    fn control_clients_are_shared_across_workers_with_the_same_pool_config() {
        let infos = view_over(vec![grpc_worker("grpc://a:1"), grpc_worker("grpc://b:1")]).list();
        let a = infos[0].control_client.as_ref().expect("a");
        let b = infos[1].control_client.as_ref().expect("b");
        assert!(Arc::ptr_eq(a, b));
    }

    /// The gateway cache holds only weak handles and a gRPC worker holds
    /// none, so the view keeps the strong one: a client is built once, not
    /// on every discovery or control call.
    #[test]
    fn the_view_holds_the_control_client_between_calls() {
        let view = view_over(vec![grpc_worker("grpc://a:1")]);
        let client = view
            .list()
            .pop()
            .expect("one worker")
            .control_client
            .expect("client");
        assert_eq!(
            Arc::strong_count(&client),
            2,
            "the view holds the other handle"
        );
        view.registry.remove_by_url("grpc://a:1");
        assert!(view.list().is_empty());
        assert_eq!(
            Arc::strong_count(&client),
            1,
            "no registered worker needs the handle anymore"
        );
    }

    /// A gateway whose TLS settings cannot produce a client is the worker's
    /// problem to report, not a silent `None`: the error rides on the info
    /// and ends up in the 422. (An unparsable client identity fails the
    /// build eagerly; CA bundles are only read when a connection is made.)
    #[test]
    fn a_failed_client_build_is_reported_on_the_worker() {
        let config = RouterConfig {
            client_identity: Some(b"not a pem".to_vec()),
            ..RouterConfig::default()
        };
        let info = view_with(&config, vec![grpc_worker("grpc://a:1")])
            .list()
            .pop()
            .expect("one worker");
        assert_eq!(info.control_url.as_deref(), Some("http://a:40100"));
        let err = info.control_client.expect_err("no client");
        assert!(err.contains("client identity"), "{err}");
    }
}
