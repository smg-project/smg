//! Unified policy update step.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{
    StepExecutor, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::{
    worker::{ConnectionMode, SubscriptionKind, Worker},
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
                worker.redacted_url()
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
            // Policies learn of the request's end through the worker's load
            // guard; the registry routes it to the policy that placed it.
            worker.set_completion_sink(Some(app_context.policy_registry.completion_sink()));

            // Initialize cache-aware policy if configured
            let all_workers = app_context.worker_registry.get_by_model(&model_id);

            // Check for configuration conflicts between prefill and decode
            Self::check_worker_conflicts(&model_id, &all_workers);
            // The registry tests the policy's type itself and does nothing
            // for a policy that is not cache-aware.
            app_context
                .policy_registry
                .init_cache_aware_policy(&model_id, &all_workers);

            // The worker's stream follows the policy that places it, the
            // role's policy for a prefill, decode or encode worker rather than
            // the model's: its KV events into the model's index under a policy
            // that reads the index, its load records alone otherwise, when
            // the worker monitor wants loads at all (always by default; see
            // `WorkerMonitor::loads_wanted`). Only gRPC workers stream.
            let needs_kv_events = app_context
                .policy_registry
                .policy_for_worker(worker.as_ref())
                .needs_kv_events();
            if *worker.connection_mode() == ConnectionMode::Grpc {
                if let Some(ref monitor) = app_context.kv_event_monitor {
                    let kind = if needs_kv_events {
                        Some(SubscriptionKind::Events)
                    } else {
                        app_context
                            .worker_monitor
                            .as_ref()
                            .is_some_and(|loads| loads.loads_wanted(std::slice::from_ref(worker)))
                            .then_some(SubscriptionKind::Loads)
                    };
                    if let Some(kind) = kind {
                        monitor.on_worker_added(worker, kind).await;
                    }
                }
            }

            Self::warn_on_cache_aware_without_kv_events(&model_id, worker, needs_kv_events);

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
    use openai_protocol::worker::{HealthCheckConfig, RuntimeType, WorkerSpec, WorkerType};
    use tracing_test::traced_test;
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::{testing::app_context_for_tests, AppContext},
        config::{PolicyConfig, RouterConfig},
        policies::{as_cache_aware, PolicyFactory},
        worker::{BasicWorkerBuilder, KvEventMonitor, ModelCard, WorkerRegistry},
        workflow::{data::WorkerRegistrationMode, steps::create_worker_workflow_data},
    };

    fn worker(url: &str, connection_mode: ConnectionMode) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .connection_mode(connection_mode)
                .build(),
        )
    }

    const MODEL: &str = "qwen3";

    /// A gRPC worker serving [`MODEL`] as `worker_type`, at a port that
    /// refuses connections at once: a stream it is given reconnects in the
    /// background with no engine involved.
    fn grpc_worker(worker_type: WorkerType) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:1")
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .model(ModelCard::new(MODEL))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    fn cache_aware() -> PolicyConfig {
        PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096)
    }

    /// An app context over `policy` with a KV event monitor and no worker
    /// monitor, so a worker gets a stream only for its KV events, and its
    /// model an index with the first.
    fn context_over(policy: PolicyConfig) -> (Arc<AppContext>, Arc<KvEventMonitor>) {
        let config = RouterConfig::builder().policy(policy).build_unchecked();
        let mut context = AppContext::clone(&app_context_for_tests(
            config,
            Arc::new(WorkerRegistry::new()),
            None,
        ));
        let monitor = Arc::new(KvEventMonitor::new());
        context
            .policy_registry
            .set_kv_event_monitor(Some(Arc::clone(&monitor)));
        context.kv_event_monitor = Some(Arc::clone(&monitor));
        (Arc::new(context), monitor)
    }

    /// Register `worker` with `labels` and run the step over it.
    async fn register(
        app_context: &Arc<AppContext>,
        worker: Arc<dyn Worker>,
        labels: &[(&str, &str)],
    ) {
        app_context
            .worker_registry
            .register(Arc::clone(&worker))
            .unwrap();
        let mut data = create_worker_workflow_data(
            WorkerSpec::new(worker.url()),
            WorkerRegistrationMode::Upsert,
            Arc::clone(app_context),
        );
        data.final_labels = labels
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        data.actual_workers = Some(vec![worker]);
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        UpdatePoliciesStep.execute(&mut context).await.unwrap();
    }

    #[tokio::test]
    async fn a_prefill_worker_placed_by_cache_aware_streams_its_kv_events() {
        // `--policy round_robin --prefill-policy cache_aware`: the model's
        // policy is round_robin, but the prefill pool is placed by
        // cache_aware, which routes on the model's index.
        let (app_context, monitor) = context_over(PolicyConfig::RoundRobin);
        app_context
            .policy_registry
            .set_prefill_policy(PolicyFactory::create_from_config(&cache_aware()));

        register(&app_context, grpc_worker(WorkerType::Prefill), &[]).await;

        assert!(monitor.get_indexer(MODEL).is_some());
        monitor.stop().await;
    }

    #[tokio::test]
    async fn a_policy_label_leaves_a_cache_aware_role_its_kv_events() {
        // A `policy` label names the model's policy, which places no decode
        // worker: the decode pool keeps the default cache_aware.
        let (app_context, monitor) = context_over(cache_aware());

        register(
            &app_context,
            grpc_worker(WorkerType::Decode),
            &[("policy", "round_robin")],
        )
        .await;

        assert!(monitor.get_indexer(MODEL).is_some());
        monitor.stop().await;
    }

    #[tokio::test]
    async fn a_prefill_worker_placed_by_round_robin_streams_no_kv_events() {
        // A `policy=cache_aware` label makes the model's policy cache_aware,
        // but the prefill pool is placed by the default round_robin, which
        // reads no index: the worker's events would fill one nothing reads.
        let (app_context, monitor) = context_over(PolicyConfig::RoundRobin);

        register(
            &app_context,
            grpc_worker(WorkerType::Prefill),
            &[("policy", "cache_aware")],
        )
        .await;

        assert!(monitor.get_indexer(MODEL).is_none());
        monitor.stop().await;
    }

    #[tokio::test]
    async fn a_regular_worker_streams_what_its_models_policy_reads() {
        let (app_context, monitor) = context_over(PolicyConfig::RoundRobin);

        register(
            &app_context,
            grpc_worker(WorkerType::Regular),
            &[("policy", "cache_aware")],
        )
        .await;

        assert!(monitor.get_indexer(MODEL).is_some());
        monitor.stop().await;
    }

    /// A worker registered under a cache-aware model policy is routed
    /// through the model's tree by this step.
    #[tokio::test]
    async fn a_worker_under_a_cache_aware_policy_is_routed_through_the_models_tree() {
        let (app_context, monitor) = context_over(cache_aware());

        register(&app_context, grpc_worker(WorkerType::Regular), &[]).await;

        let policy = app_context
            .policy_registry
            .get_policy(MODEL)
            .expect("the model's policy");
        let tenants = as_cache_aware(policy.as_ref())
            .expect("a cache-aware policy")
            .tree_tenants_for_test(MODEL);
        assert_eq!(tenants, ["grpc://127.0.0.1:1"]);
        monitor.stop().await;
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
