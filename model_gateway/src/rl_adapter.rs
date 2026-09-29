//! Glue between the RL control plane crate and the gateway registry. This
//! file is the whole of coupling touchpoint (a); see `crates/rl/COUPLING.md`.

use std::sync::Arc;

use openai_protocol::worker::ConnectionMode;
use smg_rl::{resolve_control_url, RlState, RlWorkerInfo, RlWorkerView};
use tracing::warn;

use crate::{
    config::RouterConfig,
    worker::{registry::WorkerId, Worker, WorkerHttpClientCache, WorkerRegistry},
};

/// Label an engine (or an operator) sets to name the worker's RL control app.
pub const CONTROL_URL_LABEL: &str = "rl.control_url";

/// Read-only registry view for the RL crate.
pub struct RegistryRlView {
    registry: Arc<WorkerRegistry>,
    client_cache: Arc<WorkerHttpClientCache>,
}

impl RegistryRlView {
    pub fn new(registry: Arc<WorkerRegistry>, client_cache: Arc<WorkerHttpClientCache>) -> Self {
        Self {
            registry,
            client_cache,
        }
    }

    /// Where control calls for `worker` go and which client carries them.
    ///
    /// HTTP workers are controlled through themselves, with the client the
    /// gateway negotiated for them. Other transports need an advertised or
    /// operator-supplied `rl.control_url`; the client comes from the shared
    /// cache (same TLS identity and roots as every upstream client, HTTP/1.1
    /// because the control apps are uvicorn).
    fn control_endpoint(
        &self,
        worker: &Arc<dyn Worker>,
    ) -> (Option<String>, Option<Arc<reqwest::Client>>) {
        let spec = &worker.metadata().spec;
        if *worker.connection_mode() == ConnectionMode::Http {
            let client = worker
                .http_client_handle_if_initialized()
                .unwrap_or_else(|| Arc::new(worker.http_client().clone()));
            return (Some(worker.base_url().to_string()), Some(client));
        }
        let Some(advertised) = spec
            .labels
            .get(CONTROL_URL_LABEL)
            .map(String::as_str)
            .filter(|v| !v.trim().is_empty())
        else {
            return (None, None);
        };
        let url = resolve_control_url(advertised, worker.url());
        match self.client_cache.get(&spec.http_pool, false) {
            Ok(client) => (Some(url), Some(client)),
            Err(e) => {
                warn!(
                    worker = %worker.url(), error = %e,
                    "no HTTP client for the RL control endpoint"
                );
                (Some(url), None)
            }
        }
    }

    fn info(&self, worker: &Arc<dyn Worker>) -> Option<RlWorkerInfo> {
        let id = self.registry.get_id_by_url(worker.url())?;
        let spec = &worker.metadata().spec;
        let (control_url, control_client) = self.control_endpoint(worker);
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
        })
    }
}

impl RlWorkerView for RegistryRlView {
    fn list(&self) -> Vec<RlWorkerInfo> {
        self.registry
            .get_all()
            .iter()
            .filter_map(|w| self.info(w))
            .collect()
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

    fn view_over(workers: Vec<Arc<dyn Worker>>) -> RegistryRlView {
        let registry = Arc::new(WorkerRegistry::new());
        for w in workers {
            registry.register(w);
        }
        let cache = Arc::new(WorkerHttpClientCache::new(&RouterConfig::default()));
        RegistryRlView::new(registry, cache)
    }

    /// Control calls to an HTTP worker use the client the gateway negotiated
    /// for it (HTTP version, TLS identity, pool tuning), not a second one.
    #[test]
    fn http_worker_controls_itself_with_its_negotiated_client() {
        let client = Arc::new(reqwest::Client::new());
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
    fn grpc_worker_with_a_label_gets_a_control_endpoint_and_a_cached_client() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://10.0.0.5:30000")
                .connection_mode(ConnectionMode::Grpc)
                .label("rl.control_url", "http://0.0.0.0:40100")
                .build(),
        );
        let info = view_over(vec![worker]).list().pop().expect("one worker");
        assert_eq!(
            info.control_url.as_deref(),
            Some("http://10.0.0.5:40100"),
            "wildcard bind host resolves to the worker host"
        );
        assert!(info.control_client.is_some());
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
        assert!(info.control_client.is_none());
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
        assert!(info.control_client.is_none());
    }

    #[test]
    fn cached_control_clients_are_shared_across_workers_with_the_same_pool_config() {
        let mk = |url: &str| -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new(url)
                    .connection_mode(ConnectionMode::Grpc)
                    .label("rl.control_url", "http://0.0.0.0:40100")
                    .build(),
            )
        };
        let infos = view_over(vec![mk("grpc://a:1"), mk("grpc://b:1")]).list();
        let a = infos[0].control_client.as_ref().expect("a");
        let b = infos[1].control_client.as_ref().expect("b");
        assert!(Arc::ptr_eq(a, b));
    }
}
