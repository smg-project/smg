//! Step to remove workers from worker registry.

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::workflow::{
    data::WorkerRemovalWorkflowData,
    steps::shared::{pool_group, refresh_worker_pool_sizes, PoolGroups},
};

/// Step to remove workers from the worker registry.
///
/// Removes each worker by URL from the central worker registry.
pub struct RemoveFromWorkerRegistryStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for RemoveFromWorkerRegistryStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let worker_urls = &context.data.worker_urls;

        debug!(
            "Removing {} worker(s) from worker registry",
            worker_urls.len()
        );

        // Snapshot the unique `(worker_type, connection_mode, model_id)`
        // groups for the workers we are about to remove. We capture this
        // before the removal so the pool-size metric update below can
        // recompute the per-group count after the workers have been
        // pulled from the registry. The cache eviction the old
        // `LoadMonitor` path used to do here is now handled by
        // `WorkerMonitor` reacting to `WorkerEvent::Removed`.
        let mut unique_configs = PoolGroups::new();
        for url in worker_urls {
            if let Some(w) = app_context.worker_registry.get_by_url(url) {
                unique_configs.insert(pool_group(w.as_ref()));
            }
        }

        let mut removed_count = 0;
        for worker_url in worker_urls {
            if app_context
                .worker_registry
                .remove_by_url(worker_url)
                .is_some()
            {
                removed_count += 1;
            }
        }

        // Log if some workers were already removed (e.g., by another process)
        if removed_count == worker_urls.len() {
            debug!("Removed {} worker(s) from registry", removed_count);
        } else {
            warn!(
                "Removed {} of {} workers (some may have been removed by another process)",
                removed_count,
                worker_urls.len()
            );
        }

        // Update Layer 3 worker pool size metrics for unique
        // configurations. WorkerMonitor subscribes to registry events
        // directly and reconciles per-group polling state on its own,
        // so this step no longer needs to push group-removed
        // notifications.
        refresh_worker_pool_sizes(&app_context.worker_registry, &unique_configs);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
