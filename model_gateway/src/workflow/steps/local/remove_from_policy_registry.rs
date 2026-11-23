//! Step to remove workers from policy registry.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use futures::future::join_all;
use smg_grpc_client::tokenspeed_encoder::TokenSpeedEncoderClient;
use tokio::sync::Semaphore;
use tracing::{debug, error};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::forget_cache_events::forget_cache_events;
use crate::{
    policies::PolicyRegistry,
    routers::common::snapshot::WorkerRegistry,
    workflow::data::{require_app_context, WorkerRemovalWorkflowData},
};

/// Bound CPU-heavy radix-tree purges during a fleet-wide scale-down. The
/// worker has already left the routing registry before any task waits here.
const MAX_CONCURRENT_CACHE_REMOVALS: usize = 4;
static CACHE_REMOVAL_PERMITS: Semaphore = Semaphore::const_new(MAX_CONCURRENT_CACHE_REMOVALS);

/// Step to remove workers from the policy registry.
///
/// Removes each worker from cache-aware policies and drops the policy of a
/// model that no worker of this node serves any more.
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
        let app_context = require_app_context(context.data.host.as_ref())?;
        let workers_to_remove = context
            .data
            .actual_workers_to_remove
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_remove".to_string()))?;

        debug!(
            "Removing {} worker(s) from policy registry",
            workers_to_remove.len()
        );

        // The batch's KV cleanups start together: the monitor's own bound on
        // index purges decides how many run at once.
        join_all(
            workers_to_remove
                .iter()
                .map(|worker| forget_cache_events(app_context, worker.url())),
        )
        .await;

        for worker in workers_to_remove {
            // Drop the state the policies keep on the worker (a load-aware
            // policy's dispatch tally) so it does not leak under churn.
            app_context
                .policy_registry
                .remove_worker_from_policies(worker.url());
            // Its pooled encode connections go too: the pool is keyed by
            // this URL and outlived the worker otherwise.
            TokenSpeedEncoderClient::forget_cached(worker.url());
        }

        // The removed workers' URLs leave their models' trees too, those of
        // the models they were registered with as well as the ones they list
        // now: a same-URL replacement during the drain is the one removed,
        // and its models can differ from the found worker's.
        let mut cache_removals: HashMap<String, HashSet<String>> = HashMap::new();
        for worker in workers_to_remove
            .iter()
            .chain(&context.data.removed_workers)
        {
            for model_id in WorkerRegistry::policy_model_ids(worker) {
                cache_removals
                    .entry(model_id)
                    .or_default()
                    .insert(worker.url().to_string());
            }
        }
        Self::remove_cache_state(Arc::clone(&app_context.policy_registry), cache_removals).await;

        // The workers have left the worker registry: a model none of this
        // node's workers serves any more loses its policy. Asked of the
        // registry rather than counted, so a worker registered again, or a
        // mesh import removed here, cannot keep a policy alive or end it
        // early. Every model a worker was registered with or lists now is
        // asked after, not its primary alone: a worker listing several
        // models can be the last to serve one whose own workers left first.
        // The workers the registry step removed count as well as the ones
        // found, since a same-URL replacement during the drain is the one
        // removed.
        let models: HashSet<String> = workers_to_remove
            .iter()
            .chain(&context.data.removed_workers)
            .flat_map(WorkerRegistry::policy_model_ids)
            .collect();
        for model_id in &models {
            if !app_context.worker_registry.serves_model_locally(model_id) {
                app_context.policy_registry.on_model_unserved(model_id);
            }
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
