//! Step to remove workers from policy registry.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use tokio::sync::Semaphore;
use tracing::{debug, error};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    policies::PolicyRegistry, worker::WorkerRegistry, workflow::data::WorkerRemovalWorkflowData,
};

/// Bound CPU-heavy radix-tree purges during a fleet-wide scale-down. The
/// worker has already left the routing registry before any task waits here.
const MAX_CONCURRENT_CACHE_REMOVALS: usize = 4;
static CACHE_REMOVAL_PERMITS: Semaphore = Semaphore::const_new(MAX_CONCURRENT_CACHE_REMOVALS);

/// Step to remove workers from the policy registry.
///
/// Removes each worker from cache-aware policies and notifies
/// the policy registry of worker removal.
pub struct RemoveFromPolicyRegistryStep;

impl RemoveFromPolicyRegistryStep {
    async fn remove_cache_state(
        policy_registry: Arc<PolicyRegistry>,
        removals: HashMap<String, HashSet<String>>,
    ) {
        if removals.is_empty() {
            return;
        }
        let Ok(permit) = CACHE_REMOVAL_PERMITS.acquire().await else {
            error!("Cache-removal semaphore closed");
            return;
        };
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            for (model_id, worker_urls) in removals {
                policy_registry.remove_workers_from_cache_aware(&model_id, &worker_urls);
            }
        })
        .await;

        if let Err(error) = result {
            error!(%error, "Cache-aware worker cleanup task failed");
        }
    }
}

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for RemoveFromPolicyRegistryStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let workers_to_remove = context
            .data
            .actual_workers_to_remove
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_remove".to_string()))?;

        debug!(
            "Removing {} worker(s) from policy registry",
            workers_to_remove.len()
        );

        for worker in workers_to_remove {
            // Remove from KV event monitor (stops subscription, removes from indexer)
            if let Some(ref monitor) = app_context.kv_event_monitor {
                monitor.on_worker_removed(worker.url()).await;
            }

            // Drop the worker's cached load report from load-aware policies
            // (power_of_two, least_load) so their caches don't leak under churn.
            app_context
                .policy_registry
                .remove_worker_from_load_aware(worker.url());
        }

        let mut cache_removals: HashMap<String, HashSet<String>> = HashMap::new();
        for worker in workers_to_remove {
            for model_id in WorkerRegistry::worker_model_ids(worker) {
                cache_removals
                    .entry(model_id)
                    .or_default()
                    .insert(worker.url().to_string());
            }
        }
        Self::remove_cache_state(Arc::clone(&app_context.policy_registry), cache_removals).await;

        for worker in workers_to_remove {
            app_context
                .policy_registry
                .on_worker_removed(worker.model_id());
        }

        debug!(
            "Removed {} worker(s) from policy registry",
            workers_to_remove.len()
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
