//! Glue between the RL control plane crate and the gateway registry. This
//! file is the whole of coupling touchpoint (a); see `crates/rl/COUPLING.md`.

use std::sync::Arc;

use openai_protocol::worker::ConnectionMode;
use smg_rl::{RlState, RlWorkerInfo, RlWorkerView};

use crate::{
    config::RouterConfig,
    worker::{registry::WorkerId, Worker, WorkerRegistry},
};

/// Read-only registry view for the RL crate.
pub struct RegistryRlView {
    registry: Arc<WorkerRegistry>,
}

impl RegistryRlView {
    pub fn new(registry: Arc<WorkerRegistry>) -> Self {
        Self { registry }
    }

    fn info(&self, worker: &Arc<dyn Worker>) -> Option<RlWorkerInfo> {
        let id = self.registry.get_id_by_url(worker.url())?;
        let spec = &worker.metadata().spec;
        // Borrow the client the gateway negotiated for this worker, as the
        // admin ops do; a worker without one is not spoken to over HTTP.
        let http_client = (*worker.connection_mode() == ConnectionMode::Http).then(|| {
            worker
                .http_client_handle_if_initialized()
                .unwrap_or_else(|| Arc::new(worker.http_client().clone()))
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
            http_client,
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
    config: &RouterConfig,
) -> Option<Arc<RlState>> {
    if !config.rl.enabled {
        return None;
    }
    let view = Arc::new(RegistryRlView::new(Arc::clone(registry)));
    Some(Arc::new(RlState::new(view, config.rl.clone())))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::ConnectionMode;

    use super::*;
    use crate::worker::BasicWorkerBuilder;

    fn registry_with(worker: Arc<dyn Worker>) -> Arc<WorkerRegistry> {
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(worker);
        registry
    }

    /// Control calls must use the client the gateway negotiated for the
    /// worker (HTTP version, TLS identity, pool tuning), not a second one.
    #[test]
    fn view_hands_out_the_worker_negotiated_http_client() {
        let client = Arc::new(reqwest::Client::new());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://engine:30000")
                .connection_mode(ConnectionMode::Http)
                .http_client(Arc::clone(&client))
                .build(),
        );
        let view = RegistryRlView::new(registry_with(worker));

        let info = view.list().pop().expect("one worker");
        let handed = info.http_client.expect("HTTP worker carries a client");
        assert!(Arc::ptr_eq(&handed, &client));
    }

    /// A worker the gateway does not speak HTTP to has no client to hand out.
    #[test]
    fn view_gives_no_http_client_for_grpc_workers() {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://engine:30000")
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        );
        let view = RegistryRlView::new(registry_with(worker));

        let info = view.list().pop().expect("one worker");
        assert!(info.http_client.is_none());
    }
}
