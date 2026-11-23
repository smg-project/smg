//! Unified policy update step.

use std::sync::Arc;

use async_trait::async_trait;
use smg_worker::{
    identity::basic::{ConnectionMode, RuntimeType, Worker},
    kv_events::SubscriptionKind,
};
use tracing::{debug, warn};
use wfaas::{
    StepExecutor, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::{
    app_context::AppContext,
    workflow::data::{require_app_context, WorkerRegistrationData},
};

/// Start the worker's stream on the KV event monitor, or restart it for the
/// object given when the registry replaced the worker in place. The stream
/// follows the policy that places the worker, the role's policy for a
/// prefill, decode or encode worker rather than the model's: its KV events
/// into the model's index under a policy that reads the index
/// (`needs_kv_events`), its load records alone otherwise, when the worker
/// monitor wants loads at all (always by default; see
/// `WorkerMonitor::loads_wanted`). Only gRPC workers stream. A registration
/// and an update that rebuilt the worker both come through here, so the two
/// cannot choose differently. A worker that streams nothing tells the
/// monitor as well: it ends the stream a former object has under the URL,
/// which nothing else on the replace path stops. The monitor reads the
/// registry under its slot lock before it stops or starts anything, so a
/// worker removed or replaced again while the add waited for an old task to
/// end gets no stream.
pub(crate) async fn start_kv_stream(
    app_context: &AppContext,
    worker: &Arc<dyn Worker>,
    needs_kv_events: bool,
) {
    if *worker.connection_mode() != ConnectionMode::Grpc {
        return;
    }
    let Some(monitor) = app_context.kv_event_monitor.as_ref() else {
        return;
    };
    let kind = if needs_kv_events {
        Some(SubscriptionKind::Events)
    } else {
        app_context
            .worker_monitor
            .as_ref()
            .is_some_and(|loads| loads.loads_wanted(std::slice::from_ref(worker)))
            .then_some(SubscriptionKind::Loads)
    };
    match kind {
        Some(kind) => monitor.on_worker_added(worker, kind).await,
        None => monitor.on_worker_added_without_stream(worker).await,
    }
}

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

            // SGLang's prefill side follows the room under the engine's default
            // method (`auto`) as under `follow_bootstrap_room`. The decode side
            // is then free in its width and in its method: each request's room
            // names the prefill rank that takes it, and the decode side places
            // the request by itself.
            let method = pl.get("load_balance_method").map(String::as_str);
            let sglang = pw.metadata().spec.runtime_type == RuntimeType::Sglang;
            let follows_room =
                method == Some("follow_bootstrap_room") || (sglang && method == Some("auto"));
            let decode_side_is_free = sglang && follows_room;

            // Define keys to check for equality
            let keys_to_check = ["tp_size", "dp_size", "load_balance_method"];

            for key in keys_to_check {
                if decode_side_is_free && matches!(key, "dp_size" | "load_balance_method") {
                    continue;
                }
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
            let prefill_ranks = pl.get("dp_size").and_then(|s| s.parse::<usize>().ok());
            if prefill_ranks.is_some_and(|ranks| ranks > 1) && !follows_room {
                warn!(
                    "Model {} has dp_size > 1 but load_balance_method is not 'follow_bootstrap_room' on prefill workers. This may cause rank mismatch in disaggregated mode.",
                    model_id
                );
            }
        }
    }
}

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for UpdatePoliciesStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let app_context = require_app_context(context.data.get_app_context())?.clone();

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
            // Under the policy engine the worker is given no sink.
            worker.set_completion_sink(app_context.policy_registry.worker_completion_sink());

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
            // the model's (see `start_kv_stream`).
            let needs_kv_events = app_context
                .policy_registry
                .worker_needs_kv_events(worker.as_ref());
            start_kv_stream(&app_context, worker, needs_kv_events).await;

            Self::warn_on_cache_aware_without_kv_events(&model_id, worker, needs_kv_events);

            if !updated_models.contains(&model_id) {
                updated_models.push(model_id);
            }
        }

        // Under the policy engine a `policy` label names no policy. Each
        // registration that carries one says so, behind the step's last
        // wait, so a step tried again writes no second line. The label's
        // value is free text and is not written.
        if policy_hint.is_some() && !app_context.policy_registry.keeps_classic_policies() {
            for model_id in &updated_models {
                let long = model_id.chars().nth(64).is_some();
                let cut = if long { "..." } else { "" };
                warn!(
                    "Worker label policy of model {model_id:.64}{cut} names no policy: the \
                     policy document in force places the model"
                );
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
    use openai_protocol::{
        model_card::ModelCard,
        worker::{HealthCheckConfig, RuntimeType, WorkerSpec, WorkerType},
    };
    use smg_policy_controller::{
        document::{AppliedSpec, PolicySpec},
        engine::PolicyEngine,
    };
    use smg_worker::jobs::WorkerJobAsker;
    use tracing_test::traced_test;
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::{testing::app_context_for_tests, AppContext},
        config::{PolicyConfig, RouterConfig},
        policies::{
            as_cache_aware, kv_index::event_monitor::KvEventMonitor, PolicyFactory, PolicyRegistry,
        },
        routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder},
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
        grpc_worker_at("grpc://127.0.0.1:1", worker_type)
    }

    /// [`grpc_worker`] at `url`.
    fn grpc_worker_at(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
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
        // As the app wires the pair: the add reads the registry.
        monitor.set_worker_registry(&context.worker_registry);
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
            WorkerJobAsker::Api,
        );
        data.final_labels = labels
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        data.actual_workers = Some(vec![worker]);
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        UpdatePoliciesStep.execute(&mut context).await.unwrap();
    }

    /// Under the policy engine the step makes no classic policy for a
    /// worker of any role, whatever policy its label names, and gives the
    /// worker no completion sink. Under the classic policies the same
    /// workers make the labelled model's policy beside the default, and
    /// each is given the registry's sink. The engine's step names the
    /// label at each registration that carries it, whichever worker heads
    /// the model's list, and the classic step, which follows it, at none.
    #[traced_test]
    #[tokio::test]
    async fn under_the_policy_engine_a_worker_of_any_role_gets_no_classic_policy() {
        let roles = [
            ("grpc://127.0.0.1:1", WorkerType::Regular),
            ("grpc://127.0.0.1:2", WorkerType::Prefill),
            ("grpc://127.0.0.1:3", WorkerType::Decode),
            ("grpc://127.0.0.1:4", WorkerType::Encode),
        ];
        for (policy_controller, made) in [(true, 0), (false, 2)] {
            let mut config = RouterConfig::builder()
                .policy(PolicyConfig::RoundRobin)
                .build_unchecked();
            config.policy_controller = policy_controller;
            let mut context = AppContext::clone(&app_context_for_tests(
                config.clone(),
                Arc::new(WorkerRegistry::new()),
                None,
            ));
            context.policy_registry = Arc::new(PolicyRegistry::of_config(&config));
            let app_context = Arc::new(context);
            for (nth, (url, role)) in roles.into_iter().enumerate() {
                let worker = grpc_worker_at(url, role);
                register(
                    &app_context,
                    Arc::clone(&worker),
                    &[("policy", "power_of_two")],
                )
                .await;
                assert_eq!(worker.completion_sink().is_some(), !policy_controller);
                // Each of the engine's registrations writes the line once,
                // and the classic ones behind them add none.
                let written = if policy_controller {
                    nth + 1
                } else {
                    roles.len()
                };
                let wanted = format!("Worker label policy of model {MODEL} names no policy");
                logs_assert(|lines: &[&str]| {
                    let named = lines.iter().filter(|line| line.contains(&wanted));
                    match named.filter(|line| line.contains("WARN")).count() {
                        count if count == written => Ok(()),
                        count => Err(format!("the lines that name the label: {count}")),
                    }
                });
            }
            assert_eq!(app_context.policy_registry.classic_made_or_asked(), made);
        }
    }

    /// Under the policy engine a worker's stream follows the document in
    /// force, not the classic settings. The settings name `round_robin`
    /// both times. Under a document that reads cache events the step
    /// starts a regular worker's events stream, so its model gets an
    /// index. Under a document that reads none it starts no such stream.
    #[tokio::test]
    async fn under_the_policy_engine_a_workers_events_stream_follows_the_document() {
        for (preset, streams) in [("cache_aware", true), ("round_robin", false)] {
            let mut config = RouterConfig::builder()
                .policy(PolicyConfig::RoundRobin)
                .build_unchecked();
            config.policy_controller = true;
            let mut context = AppContext::clone(&app_context_for_tests(
                config.clone(),
                Arc::new(WorkerRegistry::new()),
                None,
            ));
            let policies = PolicyRegistry::of_config(&config);
            let document = PolicySpec::preset(preset).expect("a stock preset");
            let spec = AppliedSpec::validate(&document).expect("a valid document");
            policies.set_engine(Box::leak(Box::new(PolicyEngine::new(spec))));
            context.policy_registry = Arc::new(policies);
            let monitor = Arc::new(KvEventMonitor::new());
            context.kv_event_monitor = Some(Arc::clone(&monitor));
            // As the app wires the pair: the add reads the registry.
            monitor.set_worker_registry(&context.worker_registry);
            let app_context = Arc::new(context);

            register(&app_context, grpc_worker(WorkerType::Regular), &[]).await;

            assert_eq!(monitor.get_indexer(MODEL).is_some(), streams, "{preset}");
            assert_eq!(app_context.policy_registry.classic_made_or_asked(), 0);
            monitor.stop().await;
        }
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

    /// One value per side of a pair: the prefill side's, then the decode
    /// side's.
    type Sides = (&'static str, &'static str);

    /// The pair check over one prefill and one decode worker of `runtime`,
    /// at the two sides' widths and methods; its lines name `model`.
    fn check_a_pair(model: &str, runtime: RuntimeType, widths: Sides, methods: Sides) {
        let side = |role: &str, width: &str, method: &str| -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new(format!("grpc://{role}:30000"))
                    .runtime_type(runtime)
                    .label("disaggregation_mode", role)
                    .label("dp_size", width)
                    .label("load_balance_method", method)
                    .build(),
            )
        };
        UpdatePoliciesStep::check_worker_conflicts(
            model,
            &[
                side("prefill", widths.0, methods.0),
                side("decode", widths.1, methods.1),
            ],
        );
    }

    /// The pairs the check is quiet about: an SGLang prefill side that
    /// follows the room, by the engine's default method or by name, beside a
    /// decode side of another width (the prefill side's replicas, or the
    /// decode side's) or of another method (the default, where the prefill
    /// side alone names one).
    const SERVED: [(&str, Sides, Sides); 3] = [
        ("replicas-by-default", ("2", "1"), ("auto", "auto")),
        (
            "replicas-by-name",
            ("2", "1"),
            ("follow_bootstrap_room", "auto"),
        ),
        ("decode-replicas", ("1", "2"), ("auto", "auto")),
    ];

    /// The pairs that keep every warning, two prefill ranks beside one decode
    /// rank of another method: a prefill side off the room, and an engine
    /// that is not known to follow the room under `auto`.
    const WARNED: [(&str, RuntimeType, Sides); 2] = [
        ("off-the-room", RuntimeType::Sglang, ("round_robin", "auto")),
        (
            "another-engine",
            RuntimeType::TokenSpeed,
            ("auto", "round_robin"),
        ),
    ];

    /// Runs the check over every pair and reads `line` (the words after the
    /// model's name) through the test's own `logged`: no served pair logs
    /// it, each warned pair does.
    fn only_the_warned_pairs_log(line: &str, logged: impl Fn(&str) -> bool) {
        for (model, widths, methods) in SERVED {
            check_a_pair(model, RuntimeType::Sglang, widths, methods);
            assert!(
                !logged(&format!("Model {model} {line}")),
                "the served pair {model} logs `{line}`"
            );
        }
        for (model, runtime, methods) in WARNED {
            check_a_pair(model, runtime, ("2", "1"), methods);
            assert!(
                logged(&format!("Model {model} {line}")),
                "the pair {model} does not log `{line}`"
            );
        }
    }

    #[traced_test]
    #[test]
    fn a_width_of_its_own_is_a_conflict_only_off_the_room() {
        only_the_warned_pairs_log("has conflicting dp_size", logs_contain);
    }

    #[traced_test]
    #[test]
    fn a_method_of_its_own_is_a_conflict_only_off_the_room() {
        only_the_warned_pairs_log("has conflicting load_balance_method", logs_contain);
    }

    #[traced_test]
    #[test]
    fn several_prefill_ranks_are_warned_about_only_off_the_room() {
        only_the_warned_pairs_log("has dp_size > 1", logs_contain);
    }
}
