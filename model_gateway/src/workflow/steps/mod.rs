pub mod classify;
pub mod external;
pub mod local;
pub mod shared;
#[cfg(test)]
pub(crate) mod util;

use std::{sync::Arc, time::Duration};

pub use classify::ClassifyWorkerTypeStep;
pub use external::{
    group_models_into_cards, infer_model_type_from_id, CreateExternalWorkersStep,
    DiscoverModelsStep, ModelInfo, ModelsResponse,
};
pub use local::{
    create_worker_removal_workflow, create_worker_removal_workflow_data,
    create_worker_update_workflow, create_worker_update_workflow_data, FindWorkerToUpdateStep,
    ReleaseTokenizerStep, RemoveFromPolicyRegistryStep, UpdatePoliciesForWorkerStep,
    UpdateRemainingPoliciesStep, UpdateWorkerPropertiesStep,
};
use local::{
    CheckLocalWorkerStep, DiscoverZmqLabelsStep, EnsureHarmonyEncodingStep, SubmitTokenizerJobStep,
    CHECK_LOCAL_WORKER_STEP, ZMQ_LABELS_STEP,
};
use openai_protocol::worker::WorkerSpec;
pub use shared::{
    registered_entries, registered_in, ActivateWorkersStep, RegisterWorkersStep,
    UpdatePoliciesStep, WorkerList,
};
use shared::{ACTIVATE_WORKERS_STEP, REGISTER_WORKERS_STEP};
use smg_worker::{
    jobs::WorkerJobAsker,
    workflow::{
        startup_attempts, worker_registration_workflow, RegistrationGates,
        CREATE_LOCAL_WORKER_STEP, DISCOVER_DP_STEP,
    },
};
use wfaas::{BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, WorkflowDefinition};

use crate::{
    app_context::AppContext,
    config::RouterConfig,
    workflow::data::{WorkerRegistrationMode, WorkerWorkflowData},
};

/// Create the unified worker registration workflow definition.
///
/// DAG structure:
/// ```text
///            classify_worker_type
///                    |
///       +------------+------------------+
///       |  (LOCAL branch)               |  (EXTERNAL branch)
///       v                               v
///  detect_connection_mode         discover_models
///       |                               |
///  detect_backend                       |
///       |                               |
///  discover_metadata                    |
///       |                               |
///  discover_dp_info                     |
///       |                               |
///  discover_zmq_labels                  |
///  (ZMQ only)                           |
///       |                               |
///  create_local_worker           create_external_workers
///       |                               |
///  check_local_worker                   |
///       |                               |
///  ensure_harmony_encoding              |
///  (gRPC gpt-oss only)                  |
///       +---------------+---------------+
///                        |
///                 register_workers  (shared)
///                        |
///           +------------+------------+
///           |            |            |
///      update_policies  submit_tok  activate_workers
///                       (local only)
/// ```
pub fn create_worker_registration_workflow(
    router_config: &RouterConfig,
) -> WorkflowDefinition<WorkerWorkflowData> {
    let settings = router_config.registration_settings();
    let startup_delay = Duration::from_secs(settings.startup_delay_secs);
    // The steps that poll a starting engine share one attempt budget, which
    // the worker crate derives from the startup timings.
    let max_attempts = startup_attempts(&settings);

    // Step 0: Classify worker type (Local vs External). This is the entry step,
    // so it carries the one-time startup grace period: leave the engine alone
    // for `worker_startup_delay_secs` before the first probe, then let the
    // downstream steps poll at the startup cadence.
    let mut classify_worker_type = StepDefinition::new(
        "classify_worker_type",
        "Classify Worker Type",
        Arc::new(ClassifyWorkerTypeStep),
    )
    .with_timeout(Duration::from_secs(10))
    .with_failure_action(FailureAction::FailWorkflow);
    if !startup_delay.is_zero() {
        classify_worker_type = classify_worker_type.with_delay(startup_delay);
    }

    // The worker crate detects and builds local workers. The ZMQ label read
    // stands between the detection and the creation, and the check and the
    // encoding steps behind the creation, one after another: the engine
    // writes a step's whole copy of the run's data back, so two steps that
    // wait on one step would each drop what the other wrote.
    let check_local_worker = StepDefinition::new(
        CHECK_LOCAL_WORKER_STEP,
        "Check Local Worker",
        Arc::new(CheckLocalWorkerStep),
    )
    .with_timeout(Duration::from_secs(5))
    .with_failure_action(FailureAction::FailWorkflow)
    .depends_on(&[CREATE_LOCAL_WORKER_STEP]);

    worker_registration_workflow::<AppContext>(
        &settings,
        RegistrationGates {
            detection: Some("classify_worker_type"),
            creation: Some(ZMQ_LABELS_STEP),
        },
    )
    // Step 0: Classify worker type (Local vs External)
    .add_step(classify_worker_type)
    // === LOCAL BRANCH ===
    // Step 2c (ZMQ): what the gateway knows of a ZMQ worker ahead of its
    // handshake. It waits on the crate's DP read, the last step of the
    // crate's detection, so the two write the run's data one after the
    // other. The crate's metadata read reads nothing for such a worker.
    .add_step(
        StepDefinition::new(
            ZMQ_LABELS_STEP,
            "Discover ZMQ Labels",
            Arc::new(DiscoverZmqLabelsStep),
        )
        .with_retry(RetryPolicy {
            max_attempts: 3,
            backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
        })
        .with_timeout(Duration::from_secs(10))
        .with_failure_action(FailureAction::ContinueNextStep)
        .depends_on(&[DISCOVER_DP_STEP]),
    )
    .add_step(check_local_worker)
    // Step 3b (local): gRPC gpt-oss workers need the Harmony encoding; load
    // it before registration so an unavailable vocab fails this worker
    // instead of panicking the gateway or erroring at first request.
    .add_step(
        StepDefinition::new(
            "ensure_harmony_encoding",
            "Ensure Harmony Encoding",
            Arc::new(EnsureHarmonyEncodingStep),
        )
        .with_retry(RetryPolicy {
            max_attempts,
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_secs(1),
                max: Duration::from_secs(10),
            },
        })
        .with_timeout(Duration::from_secs(30))
        .with_failure_action(FailureAction::FailWorkflow)
        .depends_on(&[CHECK_LOCAL_WORKER_STEP]),
    )
    // === EXTERNAL BRANCH ===
    // Step 1 (external): Discover models from /v1/models
    .add_step(
        StepDefinition::new(
            "discover_models",
            "Discover Models",
            Arc::new(DiscoverModelsStep),
        )
        .with_retry(RetryPolicy {
            max_attempts: 3,
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_secs(1),
                max: Duration::from_secs(10),
            },
        })
        .with_timeout(Duration::from_secs(30))
        .with_failure_action(FailureAction::FailWorkflow)
        .depends_on(&["classify_worker_type"]),
    )
    // Step 2 (external): Create external workers
    .add_step(
        StepDefinition::new(
            "create_external_workers",
            "Create External Workers",
            Arc::new(CreateExternalWorkersStep),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::FailWorkflow)
        .depends_on(&["discover_models"]),
    )
    // === SHARED (both branches converge) ===
    // Step 4: Register workers
    .add_step(
        StepDefinition::new(
            REGISTER_WORKERS_STEP,
            "Register Workers",
            Arc::new(RegisterWorkersStep),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::FailWorkflow)
        .depends_on(&[
            CHECK_LOCAL_WORKER_STEP,
            "ensure_harmony_encoding",
            "create_external_workers",
        ]),
    )
    // Step 5a: Submit tokenizer job (local only)
    .add_step(
        StepDefinition::new(
            "submit_tokenizer_job",
            "Submit Tokenizer Job",
            Arc::new(SubmitTokenizerJobStep),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::ContinueNextStep)
        .depends_on(&[REGISTER_WORKERS_STEP]),
    )
    // Step 5b: Update policies
    .add_step(
        StepDefinition::new(
            "update_policies",
            "Update Policies",
            Arc::new(UpdatePoliciesStep),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::ContinueNextStep)
        .depends_on(&[REGISTER_WORKERS_STEP]),
    )
    // Step 5c: Activate workers
    .add_step(
        StepDefinition::new(
            ACTIVATE_WORKERS_STEP,
            "Activate Workers",
            Arc::new(ActivateWorkersStep),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::FailWorkflow)
        .depends_on(&[REGISTER_WORKERS_STEP]),
    )
}

/// Create initial workflow data for the unified worker registration workflow.
pub fn create_worker_workflow_data(
    config: WorkerSpec,
    registration_mode: WorkerRegistrationMode,
    app_context: Arc<AppContext>,
    asker: WorkerJobAsker,
) -> WorkerWorkflowData {
    WorkerWorkflowData::new(config, registration_mode, app_context, asker)
}

#[cfg(test)]
mod policy_lifecycle_tests;

#[cfg(test)]
mod tests {
    use smg_worker::workflow::{
        DETECT_CONNECTION_MODE_STEP, DETECT_RUNTIME_STEP, DISCOVER_METADATA_STEP,
    };
    use wfaas::StepId;

    use super::*;

    fn config_with(timeout_secs: u64, check_interval_secs: u64) -> RouterConfig {
        RouterConfig::builder()
            .regular_mode(vec!["http://worker:8000".to_string()])
            .random_policy()
            .worker_startup_timeout_secs(timeout_secs)
            .worker_startup_check_interval_secs(check_interval_secs)
            .build_unchecked()
    }

    fn detect_step(config: &RouterConfig) -> RetryPolicy {
        create_worker_registration_workflow(config)
            .steps
            .iter()
            .find(|s| s.id.to_string() == "detect_connection_mode")
            .expect("detect_connection_mode step present")
            .retry_policy
            .clone()
            .expect("detect_connection_mode has a retry policy")
    }

    fn startup_delay(config: &RouterConfig) -> Option<Duration> {
        create_worker_registration_workflow(config)
            .steps
            .iter()
            .find(|s| s.id.to_string() == "classify_worker_type")
            .expect("classify_worker_type step present")
            .delay
    }

    #[test]
    fn startup_check_interval_drives_detect_poll_cadence() {
        let retry = detect_step(&config_with(1800, 30));
        // The dead knob now drives the retry cadence: poll every 30s.
        match retry.backoff {
            BackoffStrategy::Fixed(d) => assert_eq!(d, Duration::from_secs(30)),
            other => panic!("expected Fixed(check_interval), got {other:?}"),
        }
        // Attempts span the timeout budget at that cadence:
        // ceil(1800 * 0.9 / 30) = 54.
        assert_eq!(retry.max_attempts, 54);
    }

    #[test]
    fn startup_attempts_scale_with_check_interval() {
        // Halving the interval doubles the attempt budget for the same timeout.
        let coarse = detect_step(&config_with(1800, 60));
        let fine = detect_step(&config_with(1800, 30));
        assert_eq!(coarse.max_attempts, 27);
        assert_eq!(fine.max_attempts, 54);
    }

    #[test]
    fn startup_attempts_have_a_floor_for_tiny_timeouts() {
        // A 1s timeout at a 30s cadence still retries MIN_ATTEMPTS (3) times
        // instead of giving up after a single probe.
        let retry = detect_step(&config_with(1, 30));
        assert_eq!(retry.max_attempts, 3);
    }

    #[test]
    fn startup_delay_defers_first_probe() {
        // The delay knob becomes a one-time wait on the entry step, before any
        // engine probing begins.
        let mut config = config_with(1800, 30);
        config.worker_startup_delay_secs = 120;
        assert_eq!(startup_delay(&config), Some(Duration::from_secs(120)));
        // The detection, the worker crate's step, waits on that entry step
        // and carries no delay of its own: the worker is left alone once.
        let workflow = create_worker_registration_workflow(&config);
        let detection = workflow
            .steps
            .iter()
            .find(|s| s.id.to_string() == DETECT_CONNECTION_MODE_STEP)
            .expect("detect_connection_mode step present");
        assert_eq!(detection.delay, None);
        let waits_on: Vec<String> = detection
            .depends_on
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(waits_on, ["classify_worker_type"]);
    }

    #[test]
    fn zero_startup_delay_leaves_probe_immediate() {
        // Default (0) preserves the immediate-check behavior: no delay is set.
        let config = config_with(1800, 30);
        assert_eq!(config.worker_startup_delay_secs, 0);
        assert_eq!(startup_delay(&config), None);
    }

    fn step<'a>(
        workflow: &'a WorkflowDefinition<WorkerWorkflowData>,
        id: &str,
    ) -> &'a StepDefinition<WorkerWorkflowData> {
        let id = StepId::new(id);
        workflow.steps.iter().find(|step| step.id == id).unwrap()
    }

    /// Each local step writes the run's data before the next one reads it.
    #[test]
    fn a_local_workers_steps_write_the_data_one_after_another() {
        let workflow = create_worker_registration_workflow(&config_with(1800, 30));
        let line = [
            "classify_worker_type",
            DETECT_CONNECTION_MODE_STEP,
            DETECT_RUNTIME_STEP,
            DISCOVER_METADATA_STEP,
            DISCOVER_DP_STEP,
            ZMQ_LABELS_STEP,
            CREATE_LOCAL_WORKER_STEP,
            CHECK_LOCAL_WORKER_STEP,
        ];
        for pair in line.windows(2) {
            let next = step(&workflow, pair[1]);
            assert_eq!(
                next.depends_on,
                [StepId::new(pair[0])],
                "{} waits on {} alone",
                pair[1],
                pair[0]
            );
            assert!(next.depends_on_any.is_empty(), "{}", pair[1]);
        }
        let dependents = |id: &str| {
            let id = StepId::new(id);
            let mut dependents: Vec<_> = workflow
                .steps
                .iter()
                .filter(|step| {
                    step.depends_on
                        .iter()
                        .chain(&step.depends_on_any)
                        .any(|dep| dep == &id)
                })
                .map(|step| step.id.to_string())
                .collect();
            dependents.sort_unstable();
            dependents
        };
        // Classification also starts the external branch.
        for pair in line[1..].windows(2) {
            assert_eq!(dependents(pair[0]), [pair[1]], "{}", pair[0]);
        }
        assert_eq!(
            dependents(CHECK_LOCAL_WORKER_STEP),
            ["ensure_harmony_encoding", REGISTER_WORKERS_STEP]
        );
        let register = step(&workflow, REGISTER_WORKERS_STEP);
        assert_eq!(
            register.depends_on,
            [
                CHECK_LOCAL_WORKER_STEP,
                "ensure_harmony_encoding",
                "create_external_workers"
            ]
            .map(StepId::new)
        );
        assert!(register.depends_on_any.is_empty());
    }

    #[test]
    fn the_harmony_encoding_waits_on_the_check_step_alone() {
        let workflow = create_worker_registration_workflow(&config_with(1800, 30));
        let check = step(&workflow, CHECK_LOCAL_WORKER_STEP);
        assert_eq!(check.id, StepId::new("check_local_worker"));
        assert_eq!(check.name, "Check Local Worker");
        assert_eq!(check.timeout, Some(Duration::from_secs(5)));
        assert!(check.retry_policy.is_none());
        assert!(matches!(check.on_failure, FailureAction::FailWorkflow));

        let harmony = step(&workflow, "ensure_harmony_encoding");
        assert_eq!(harmony.depends_on, [StepId::new(CHECK_LOCAL_WORKER_STEP)]);
        assert!(harmony.depends_on_any.is_empty());
    }

    #[tokio::test]
    async fn an_unknown_parser_override_fails_the_registration_and_writes_no_worker() {
        use openai_protocol::worker::{ConnectionMode, RuntimeType};
        use smg_worker::workflow::WORKER_REGISTRATION_WORKFLOW;
        use wfaas::{StepStatus, WorkflowEngine, WorkflowId, WorkflowStatus};

        use crate::{
            app_context::testing::app_context_for_tests, config::ParserChoice,
            routers::common::snapshot::WorkerRegistry,
        };

        let config = RouterConfig {
            parser: ParserChoice::Legacy,
            ..config_with(1, 1)
        };
        let registry = Arc::new(WorkerRegistry::new());
        let host = app_context_for_tests(
            config.clone(),
            Arc::clone(&registry),
            Some(tool_parser::ParserFactory::default()),
        );
        let engine = WorkflowEngine::new();
        engine
            .register_workflow(create_worker_registration_workflow(&config))
            .unwrap();

        let url = "ipc:///tmp/worker-unknown-parser.ipc";
        let mut spec = WorkerSpec::new(url);
        spec.connection_mode = ConnectionMode::Zmq;
        spec.runtime_type = RuntimeType::Vllm;
        spec.labels.insert("model_id".into(), "m".into());
        spec.labels
            .insert("tool_parser".into(), "no-such-parser".into());
        let id = engine
            .start_workflow(
                WorkflowId::new(WORKER_REGISTRATION_WORKFLOW),
                create_worker_workflow_data(
                    spec,
                    WorkerRegistrationMode::Upsert,
                    host,
                    WorkerJobAsker::Api,
                ),
            )
            .await
            .unwrap();
        let state = engine
            .wait_for_terminal_state(id, url, Duration::from_secs(5))
            .await
            .unwrap();

        assert!(registry.is_empty());
        assert_eq!(state.status, WorkflowStatus::Failed);
        let check = &state.step_states[&StepId::new(CHECK_LOCAL_WORKER_STEP)];
        assert_eq!(check.status, StepStatus::Failed);
        assert_eq!(check.attempt, 1);
        assert_eq!(
            check.last_error.as_deref(),
            Some(
                "Step failed: create_worker - worker ipc:///tmp/worker-unknown-parser.ipc \
                 declares a parser override for model 'm' the gateway cannot read with: \
                 unknown tool-call parser 'no-such-parser'"
            )
        );
    }
}
