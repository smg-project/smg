//! Glue between the RL control plane crate and the gateway registry. This
//! file is the whole of coupling touchpoint (a); see `crates/rl/COUPLING.md`.

use std::sync::Arc;

use smg_rl::{RlConfig, RlState, RlWorkerInfo, RlWorkerView};

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
) -> Result<Option<Arc<RlState>>, String> {
    if !config.rl.enabled {
        return Ok(None);
    }
    let rl: RlConfig = config.rl.clone();
    let view = Arc::new(RegistryRlView::new(Arc::clone(registry)));
    RlState::new(view, rl, config.upstream_http2)
        .map(|s| Some(Arc::new(s)))
        .map_err(|e| e.to_string())
}
