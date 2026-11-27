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
use smg_worker::{
    connection::http_client::WorkerHttpClientCache, identity::basic::Worker, registry::WorkerId,
};
use tracing::warn;

use crate::{
    config::RouterConfig,
    routers::{
        common::snapshot::WorkerRegistry,
        grpc::backend_client::{connected_backend_client, get_backend_client, BackendClient},
    },
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
        // control calls on the worker's own connection, when its handshake
        // declared the control reply: a connected engine that declared none
        // has no call to take (its wire carries generate and abort only), so
        // discovery reports it without and a call answers as for a worker
        // without a control endpoint. An engine still handshaking is taken
        // at its word until it lands.
        let control_calls = (*worker.connection_mode() == ConnectionMode::Zmq
            && spec.runtime_type == RuntimeType::TokenSpeed
            && connected_backend_client(worker.as_ref()).is_none_or(|client| {
                matches!(client.as_ref(), BackendClient::Zmq(client) if client.carries_control_calls())
            }))
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
        let client = get_backend_client(self.worker.as_ref())
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
            // The wire carries no control calls (declared at the handshake):
            // not a failure of a reachable engine, the route's 422.
            Err(status) if status.code() == tonic::Code::Unimplemented => {
                Err(CallFailure::Unsupported(status.message().to_string()))
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
#[path = "../tests/unit/rl_adapter/tests.rs"]
mod tests;
