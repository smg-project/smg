//! Unified policy update step.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{
    StepExecutor, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::{
    worker::{ConnectionMode, Worker},
    workflow::data::WorkerRegistrationData,
};

/// Unified step to update policy registry for registered workers.
///
/// Handles both local workers (same model, possibly DP-aware) and
/// external workers (different models per worker).
pub struct UpdatePoliciesStep;

impl UpdatePoliciesStep {
    /// Say so when cache_aware routing runs degraded.
    ///
    /// The raw ZMQ wire carries no KV-event stream, so a ZMQ worker under a
    /// cache_aware policy is routed by the approximate radix tree alone. That
    /// is the intended behavior, but silently degrading a policy the operator
    /// asked for deserves a line in the log.
    fn warn_on_cache_aware_without_kv_events(
        model_id: &str,
        worker: &Arc<dyn Worker>,
        cache_aware: bool,
    ) {
        if cache_aware && *worker.connection_mode() == ConnectionMode::Zmq {
            warn!(
                "cache_aware policy for model {model_id} includes ZMQ worker {}: the ZMQ \
                 transport has no KV-event stream, so cache tracking for it falls back to the \
                 approximate prefix tree",
                worker.url()
            );
        }
    }

    /// Check for conflicts between prefill and decode worker configurations for a model.
    fn check_worker_conflicts(model_id: &str, workers: &[Arc<dyn Worker>]) {
        let prefill_workers: Vec<_> = workers
            .iter()
            .filter(|w| {
                w.metadata()
                    .spec
                    .labels
                    .get("disaggregation_mode")
                    .map(|s| s.as_str())
                    == Some("prefill")
            })
            .collect();

        let decode_workers: Vec<_> = workers
            .iter()
            .filter(|w| {
                w.metadata()
                    .spec
                    .labels
                    .get("disaggregation_mode")
                    .map(|s| s.as_str())
                    == Some("decode")
            })
            .collect();

        if prefill_workers.is_empty() || decode_workers.is_empty() {
            return;
        }

        // Compare configurations of prefill vs decode workers
        if let (Some(pw), Some(dw)) = (prefill_workers.first(), decode_workers.first()) {
            let pl = &pw.metadata().spec.labels;
            let dl = &dw.metadata().spec.labels;

            // Define keys to check for equality
            let keys_to_check = ["tp_size", "dp_size", "load_balance_method"];

            for key in keys_to_check {
                let p_val = pl.get(key);
                let d_val = dl.get(key);
                if p_val != d_val {
                    warn!(
                        "Model {} has conflicting {}: prefill={:?}, decode={:?}",
                        model_id, key, p_val, d_val
                    );
                }
            }

            // Specific check for Data-Parallel consistency
            if let Some(dp_size) = pl.get("dp_size").and_then(|s| s.parse::<usize>().ok()) {
                if dp_size > 1 {
                    let plb = pl.get("load_balance_method").map(|s| s.as_str());
                    if plb != Some("follow_bootstrap_room") {
                        warn!(
                            "Model {} has dp_size > 1 but load_balance_method is not 'follow_bootstrap_room' on prefill workers. This may cause rank mismatch in disaggregated mode.",
                            model_id
                        );
                    }
                }
            }
        }
    }
}

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for UpdatePoliciesStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .get_app_context()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();

        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        let labels = context
            .data
            .get_labels()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("labels".to_string()))?;

        let policy_hint = labels.get("policy").map(|s| s.as_str());

        // Track unique model IDs we've updated policies for
        let mut updated_models = Vec::new();

        for worker in workers {
            let model_id = worker.model_id().to_string();

            // Notify policy registry
            app_context
                .policy_registry
                .on_worker_added(&model_id, policy_hint);

            // Initialize cache-aware policy if configured
            let all_workers = app_context.worker_registry.get_by_model(&model_id);

            // Check for configuration conflicts between prefill and decode
            Self::check_worker_conflicts(&model_id, &all_workers);
            let cache_aware = app_context
                .policy_registry
                .get_policy(&model_id)
                .is_some_and(|policy| policy.name() == "cache_aware");
            let cache_aware_length = app_context
                .policy_registry
                .get_policy(&model_id)
                .is_some_and(|policy| policy.name() == "cache_aware_length");
            if cache_aware || cache_aware_length {
                app_context
                    .policy_registry
                    .init_cache_aware_policy(&model_id, &all_workers);
            }

            // Start KV event subscription for gRPC workers with cache_aware or
            // cache_aware_length policy (both are KV-event-capable).
            if cache_aware || cache_aware_length {
                if let Some(ref monitor) = app_context.kv_event_monitor {
                    if *worker.connection_mode() == ConnectionMode::Grpc {
                        monitor.on_worker_added(worker).await;
                    }
                }
            }

            Self::warn_on_cache_aware_without_kv_events(
                &model_id,
                worker,
                cache_aware || cache_aware_length,
            );

            if !updated_models.contains(&model_id) {
                updated_models.push(model_id);
            }
        }

        // Initialize PD policies for prefill/decode workers (local workers only)
        let prefill_workers = app_context.worker_registry.get_prefill_workers();
        let decode_workers = app_context.worker_registry.get_decode_workers();
        app_context
            .policy_registry
            .init_pd_cache_aware_policies(&prefill_workers, &decode_workers);
        if !prefill_workers.is_empty() {
            let policy = app_context.policy_registry.get_prefill_policy();
            if policy.name() == "bucket" {
                app_context
                    .policy_registry
                    .init_pd_bucket_policies(&prefill_workers);
            }
        }

        debug!(
            "Updated policies for {} workers across {} models",
            workers.len(),
            updated_models.len()
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use tracing_test::traced_test;

    use super::*;
    use crate::worker::BasicWorkerBuilder;

    fn worker(url: &str, connection_mode: ConnectionMode) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .connection_mode(connection_mode)
                .build(),
        )
    }

    #[traced_test]
    #[test]
    fn cache_aware_over_zmq_warns_that_kv_events_are_missing() {
        // The ZMQ wire carries no KV-event stream, so this worker's cache
        // tracking is the approximate prefix tree alone. Degrading a policy the
        // operator asked for is fine; doing it silently is not.
        UpdatePoliciesStep::warn_on_cache_aware_without_kv_events(
            "glm-5.2",
            &worker("ipc:///tmp/smg-zmq/a.ipc", ConnectionMode::Zmq),
            true,
        );

        assert!(logs_contain("has no KV-event stream"));
        assert!(logs_contain("glm-5.2"));
        assert!(logs_contain("ipc:///tmp/smg-zmq/a.ipc"));
    }

    #[traced_test]
    #[test]
    fn kv_event_capable_and_non_cache_aware_workers_stay_quiet() {
        // gRPC has the KV-event stream, so cache_aware is served in full...
        UpdatePoliciesStep::warn_on_cache_aware_without_kv_events(
            "glm-5.2",
            &worker("grpc://worker:8080", ConnectionMode::Grpc),
            true,
        );
        // ...and a ZMQ worker under any other policy loses nothing to warn about.
        UpdatePoliciesStep::warn_on_cache_aware_without_kv_events(
            "glm-5.2",
            &worker("ipc:///tmp/smg-zmq/a.ipc", ConnectionMode::Zmq),
            false,
        );

        assert!(!logs_contain("has no KV-event stream"));
    }
}
