pub mod classify;
pub mod external;
pub mod local;
pub mod shared;
#[cfg(test)]
pub(crate) mod util;

use std::{sync::Arc, time::Duration};

pub(crate) use classify::stated_kind;
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
pub(crate) use local::{worker_removal_line, worker_update_line};
use local::{
    CheckLocalWorkerStep, DiscoverZmqLabelsStep, EnsureHarmonyEncodingStep, SubmitTokenizerJobStep,
    CHECK_LOCAL_WORKER_STEP, ZMQ_LABELS_STEP,
};
use openai_protocol::worker::WorkerSpec;
pub use shared::{
    registered_entries, registered_in, ActivateWorkersStep, RegisterWorkersStep,
    UpdatePoliciesStep, WorkerList,
};
use shared::{RegisterBesideRowsStep, REGISTER_WORKERS_STEP};
use smg_policy_controller::worker_host::{registration_at_door, WRITE_WORKERS_AT_DOOR_STEP};
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
    policy_controller::feed::Feed,
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
    worker_registration_line(router_config, None)
}

/// The registration line. With no feed it is
/// [`create_worker_registration_workflow`]'s.
///
/// With a feed the gateway's local workers are rows of the policy
/// controller's store, and the line differs in four things:
/// ```text
///            classify_worker_type
///                    |
///       +------------+------------------+
///       |  (LOCAL branch)               |  (EXTERNAL branch)
///       v                               v
///  (the local steps above)        discover_models
///       |                               |
///  ensure_harmony_encoding       create_external_workers
///       |                               |
///  write_workers_at_door                |
///       +---------------+---------------+
///                        |
///                 register_workers
///                        |
///                 +------+------+
///                 |             |
///            submit_tok   activate_workers
/// ```
/// - A local registration that states an API key registers as any other: a
///   row holds no credential, and the row's object holds the key.
/// - The store crate's write at the feed's door stands behind the encoding
///   step, and no other step waits on that step. It writes a local worker's
///   row, and the holder lays the row's object in the registry.
/// - The register step waits on that write and on the external branch. It
///   writes a provider's workers alone ([`RegisterBesideRowsStep`]).
/// - No step writes the classic policies: they decide nothing there.
pub(crate) fn worker_registration_line(
    router_config: &RouterConfig,
    feed: Option<&Feed>,
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

    // Step 4: Register workers, where both branches converge. Under a feed
    // the local branch's last step is the write at the door.
    let register_workers = match feed {
        None => StepDefinition::new(
            REGISTER_WORKERS_STEP,
            "Register Workers",
            Arc::new(RegisterWorkersStep),
        )
        .depends_on(&[
            CHECK_LOCAL_WORKER_STEP,
            "ensure_harmony_encoding",
            "create_external_workers",
        ]),
        Some(feed) => StepDefinition::new(
            REGISTER_WORKERS_STEP,
            "Register Workers",
            Arc::new(RegisterBesideRowsStep::new(Arc::clone(&feed.objects))),
        )
        .depends_on(&[WRITE_WORKERS_AT_DOOR_STEP, "create_external_workers"]),
    }
    .with_timeout(Duration::from_secs(5))
    .with_failure_action(FailureAction::FailWorkflow);

    // The crate's line. Under a feed it ends with the write at the feed's
    // door.
    let gates = |detection| RegistrationGates {
        detection: Some(detection),
        creation: Some(ZMQ_LABELS_STEP),
    };
    let workflow = match feed {
        None => {
            worker_registration_workflow::<AppContext>(&settings, gates("classify_worker_type"))
        }
        Some(feed) => registration_at_door::<AppContext>(
            &settings,
            gates("classify_worker_type"),
            Arc::clone(&feed.door),
            Arc::clone(&feed.objects),
            &["ensure_harmony_encoding"],
        ),
    }
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
    .add_step(register_workers)
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
    );
    // Step 5b: Update policies, where the classic policies decide
    let workflow = if feed.is_some() {
        workflow
    } else {
        workflow.add_step(
            StepDefinition::new(
                "update_policies",
                "Update Policies",
                Arc::new(UpdatePoliciesStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&[REGISTER_WORKERS_STEP]),
        )
    };
    // Step 5c: Activate workers
    workflow.add_step(
        StepDefinition::new(
            "activate_workers",
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
    use std::{
        collections::{BTreeMap, BTreeSet, HashSet},
        time::Instant,
    };

    use http::StatusCode;
    use openai_protocol::worker::WorkerStatus;
    use serde_json::{json, Value};
    use smg_policy_controller::store::{
        MemoryStore, Registrant, WorkerStatus as RowStatus, WorkersStore,
    };
    use smg_worker::{
        identity::basic::{BasicWorker, Worker},
        workflow::{
            DETECT_CONNECTION_MODE_STEP, DETECT_RUNTIME_STEP, DISCOVER_METADATA_STEP,
            WORKER_REGISTRATION_WORKFLOW,
        },
    };
    use wfaas::{StepId, WorkflowEngine, WorkflowId, WorkflowState, WorkflowStatus};

    use super::{local::test_support::serve_json_routes, *};
    use crate::{
        app_context::testing::app_context_for_tests,
        policy_controller::{
            engine::{Ask, PolicyEngine},
            feed::{feed, following},
            start,
        },
        routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BackendConnector},
    };

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

    /// A line's steps and the steps each waits on, in the line's order.
    fn edges(workflow: &WorkflowDefinition<WorkerWorkflowData>) -> Vec<(String, Vec<String>)> {
        workflow
            .steps
            .iter()
            .map(|step| {
                assert!(step.depends_on_any.is_empty(), "{}", step.id);
                let waits_on = step.depends_on.iter().map(ToString::to_string).collect();
                (step.id.to_string(), waits_on)
            })
            .collect()
    }

    fn written(table: &[(&str, &[&str])]) -> Vec<(String, Vec<String>)> {
        table
            .iter()
            .map(|(id, waits_on)| {
                let waits_on = waits_on.iter().map(ToString::to_string).collect();
                ((*id).to_string(), waits_on)
            })
            .collect()
    }

    #[test]
    fn with_no_feed_the_registration_line_is_todays() {
        let config = config_with(1800, 30);
        let todays = written(&[
            ("detect_connection_mode", &["classify_worker_type"]),
            ("detect_backend", &["detect_connection_mode"]),
            ("discover_metadata", &["detect_backend"]),
            ("discover_dp_info", &["discover_metadata"]),
            ("create_local_worker", &["discover_zmq_labels"]),
            ("classify_worker_type", &[]),
            ("discover_zmq_labels", &["discover_dp_info"]),
            ("check_local_worker", &["create_local_worker"]),
            ("ensure_harmony_encoding", &["check_local_worker"]),
            ("discover_models", &["classify_worker_type"]),
            ("create_external_workers", &["discover_models"]),
            (
                "register_workers",
                &[
                    "check_local_worker",
                    "ensure_harmony_encoding",
                    "create_external_workers",
                ],
            ),
            ("submit_tokenizer_job", &["register_workers"]),
            ("update_policies", &["register_workers"]),
            ("activate_workers", &["register_workers"]),
        ]);
        assert_eq!(edges(&create_worker_registration_workflow(&config)), todays);
        assert_eq!(edges(&worker_registration_line(&config, None)), todays);
    }

    /// The feed of a gateway that started under `config`, over a registry of
    /// its own: the store, the registry and the feed. `None` where no store
    /// started.
    fn started(config: &RouterConfig) -> Option<(Arc<MemoryStore>, Arc<WorkerRegistry>, Feed)> {
        let store = start(config).ok()??;
        let registry = Arc::new(WorkerRegistry::new());
        let feed = feed(&store, &registry, BackendConnector::shared());
        Some((store, registry, feed))
    }

    fn picked(mut config: RouterConfig) -> RouterConfig {
        config.policy_controller = true;
        config.health_check.disable_health_check = true;
        config
    }

    /// Under the feed a local worker's branch ends with the write at the
    /// door, behind the encoding step alone and ahead of the register step.
    /// No step writes the classic policies.
    #[test]
    fn under_the_feed_the_local_line_writes_at_the_door() {
        let config = picked(config_with(1800, 30));
        let (_, _, feed) = started(&config).expect("the store starts");
        let line = written(&[
            ("detect_connection_mode", &["classify_worker_type"]),
            ("detect_backend", &["detect_connection_mode"]),
            ("discover_metadata", &["detect_backend"]),
            ("discover_dp_info", &["discover_metadata"]),
            ("create_local_worker", &["discover_zmq_labels"]),
            ("write_workers_at_door", &["ensure_harmony_encoding"]),
            ("classify_worker_type", &[]),
            ("discover_zmq_labels", &["discover_dp_info"]),
            ("check_local_worker", &["create_local_worker"]),
            ("ensure_harmony_encoding", &["check_local_worker"]),
            ("discover_models", &["classify_worker_type"]),
            ("create_external_workers", &["discover_models"]),
            (
                "register_workers",
                &["write_workers_at_door", "create_external_workers"],
            ),
            ("submit_tokenizer_job", &["register_workers"]),
            ("activate_workers", &["register_workers"]),
        ]);
        assert_eq!(edges(&worker_registration_line(&config, Some(&feed))), line);
    }

    /// Two steps behind one step each write the run's data back whole, and
    /// one drops what the other wrote. Under the feed two steps stand
    /// behind one step in two places alone: at the classification, where
    /// one branch of the two skips, and behind the register step, as with
    /// no feed. One step alone waits on two: the register step.
    #[test]
    fn under_the_feed_every_step_waits_on_one_step_alone_but_the_join() {
        use std::collections::BTreeMap;

        let config = picked(config_with(1800, 30));
        let (_, _, feed) = started(&config).expect("the store starts");
        let mut followers: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut joins = Vec::new();
        for (id, waits_on) in edges(&worker_registration_line(&config, Some(&feed))) {
            if waits_on.len() > 1 {
                joins.push(id.clone());
            }
            for step in waits_on {
                followers.entry(step).or_default().push(id.clone());
            }
        }
        followers.retain(|_, steps| steps.len() > 1);
        assert_eq!(joins, ["register_workers"]);
        assert_eq!(
            followers.into_iter().collect::<Vec<_>>(),
            [
                (
                    "classify_worker_type".to_string(),
                    vec![
                        "detect_connection_mode".to_string(),
                        "discover_models".to_string()
                    ]
                ),
                (
                    "register_workers".to_string(),
                    vec![
                        "submit_tokenizer_job".to_string(),
                        "activate_workers".to_string()
                    ]
                ),
            ]
        );
    }

    /// A registration under the feed, run through the engine to its end.
    async fn registered_under_the_feed(
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        feed: &Feed,
        spec: WorkerSpec,
    ) -> Result<WorkflowState<WorkerWorkflowData>, String> {
        registered_on(config, registry, Some(feed), spec).await
    }

    /// A registration on the line of `feed`, the classic line with none,
    /// run through the engine to its end.
    async fn registered_on(
        config: &RouterConfig,
        registry: &Arc<WorkerRegistry>,
        feed: Option<&Feed>,
        spec: WorkerSpec,
    ) -> Result<WorkflowState<WorkerWorkflowData>, String> {
        let host = app_context_for_tests(config.clone(), Arc::clone(registry), None);
        let engine = WorkflowEngine::new();
        engine
            .register_workflow(worker_registration_line(config, feed))
            .map_err(|invalid| invalid.to_string())?;
        let url = spec.url.clone();
        let id = engine
            .start_workflow(
                WorkflowId::new(WORKER_REGISTRATION_WORKFLOW),
                create_worker_workflow_data(
                    spec,
                    WorkerRegistrationMode::Upsert,
                    host,
                    WorkerJobAsker::Startup,
                ),
            )
            .await
            .map_err(|error| error.to_string())?;
        engine
            .wait_for_terminal_state(id, &url, Duration::from_secs(20))
            .await
    }

    /// What a worker object holds, by field: its spec as it serializes,
    /// then what its builder was handed beside the spec.
    fn fields(worker: &dyn Worker) -> BTreeMap<String, Value> {
        let meta = worker.metadata();
        let spec = serde_json::to_value(&*meta.spec).expect("a spec serializes");
        let mut fields: BTreeMap<String, Value> = spec
            .as_object()
            .expect("a spec is an object")
            .iter()
            .map(|(name, value)| (format!("spec.{name}"), value.clone()))
            .collect();
        let resilience = worker.resilience();
        let sorted = |codes: &HashSet<u16>| format!("{:?}", codes.iter().collect::<BTreeSet<_>>());
        let breaker = worker
            .as_any()
            .downcast_ref::<BasicWorker>()
            .map(|basic| format!("{:?}", basic.circuit_breaker.load().config()));
        let handed = worker.http_client_handle_if_initialized().is_some();
        for (name, value) in [
            ("cards", format!("{:?}", worker.models())),
            ("health_config", format!("{:?}", meta.health_config)),
            ("health_endpoint", meta.health_endpoint.clone()),
            (
                "zmq_startup_timeout",
                format!("{:?}", meta.zmq_startup_timeout),
            ),
            (
                "zmq_request_timeout",
                format!("{:?}", meta.zmq_request_timeout),
            ),
            ("retry", format!("{:?}", resilience.retry)),
            ("retry_enabled", resilience.retry_enabled.to_string()),
            (
                "breaker_enabled",
                resilience.circuit_breaker_enabled.to_string(),
            ),
            (
                "retryable_codes",
                sorted(&resilience.retryable_status_codes),
            ),
            ("capacity_codes", sorted(&resilience.capacity_status_codes)),
            ("breaker", format!("{breaker:?}")),
            ("http2", worker.http2().to_string()),
            ("http_client_handed", handed.to_string()),
            (
                "engine_connector",
                worker.engine_connector().is_some().to_string(),
            ),
            (
                "max_running",
                format!("{:?}", worker.max_running_requests()),
            ),
        ] {
            fields.insert(name.to_owned(), Value::String(value));
        }
        fields
    }

    /// The names of the fields two worker objects hold differently.
    fn differing(one: &dyn Worker, other: &dyn Worker) -> Vec<String> {
        let (one, other) = (fields(one), fields(other));
        let names: BTreeSet<&String> = one.keys().chain(other.keys()).collect();
        names
            .into_iter()
            .filter(|name| one.get(*name) != other.get(*name))
            .cloned()
            .collect()
    }

    /// A worker's routes as a registration probes them.
    async fn serving_worker() -> String {
        serve_json_routes(vec![
            ("/health", StatusCode::OK, json!({"status": "ok"})),
            (
                "/model_info",
                StatusCode::OK,
                json!({"model_path": "m", "is_generation": true}),
            ),
            ("/server_info", StatusCode::OK, json!({"version": "test"})),
        ])
        .await
    }

    /// A job that states what no row holds: a priority, a cost, the two
    /// transfer names and the two parser labels, under a gateway whose
    /// breaker and retry settings are not the defaults.
    fn stated_beside_the_row(url: &str) -> (RouterConfig, WorkerSpec) {
        let mut config = picked(config_with(5, 1));
        config.circuit_breaker.failure_threshold = 7;
        config.circuit_breaker.success_threshold = 5;
        config.circuit_breaker.timeout_duration_secs = 11;
        config.circuit_breaker.window_duration_secs = 13;
        config.retry.max_retries = 9;
        config.retry.initial_backoff_ms = 17;
        let mut spec = WorkerSpec::new(url);
        spec.priority = 7;
        spec.cost = 2.5;
        spec.bootstrap_port = Some(8998);
        spec.kv_connector = Some("connector-a".into());
        spec.kv_role = Some("role-a".into());
        spec.labels.insert("tool_parser".into(), "parser-a".into());
        spec.labels
            .insert("reasoning_parser".into(), "parser-b".into());
        (config, spec)
    }

    /// The object a registration leaves in the registry, field by field,
    /// beside the object the classic line leaves for the same job.
    ///
    /// In process the two read the same: the holder keeps the object the
    /// job built. Under a controller's address no job runs, so the holder
    /// builds the object from the row, and the fields no row states read
    /// as the defaults. Their names are pinned here.
    #[tokio::test]
    async fn a_rows_object_reads_as_the_classic_lines_object_field_by_field() {
        let url = serving_worker().await;
        let (config, spec) = stated_beside_the_row(&url);

        let mut classic_config = config.clone();
        classic_config.policy_controller = false;
        let classic_registry = Arc::new(WorkerRegistry::new());
        let state = registered_on(&classic_config, &classic_registry, None, spec.clone())
            .await
            .expect("the classic run ends");
        assert_eq!(state.status, WorkflowStatus::Completed);
        let classic = classic_registry.get_by_url(&url).expect("the object");

        let (store, registry, feed) = started(&config).expect("the store starts");
        let state = registered_on(&config, &registry, Some(&feed), spec)
            .await
            .expect("the run ends");
        assert_eq!(state.status, WorkflowStatus::Completed);
        let hosted = registry.get_by_url(&url).expect("the row's object");
        let built = state.context.data.actual_workers.as_deref();
        assert!(matches!(built, Some([built]) if Arc::ptr_eq(built, &hosted)));
        assert_eq!(
            differing(classic.as_ref(), hosted.as_ref()),
            Vec::<String>::new()
        );
        assert_eq!(feed.objects.unhanded(), 0);

        let row = store.by_key(&url).expect("the worker's row");
        let served = start(&config).ok().flatten().expect("a second store");
        served.upsert_by(row.facts, row.maker);
        let followed = Arc::new(WorkerRegistry::new());
        following(&served, &followed, BackendConnector::shared()).settle();
        let followed = followed.get_by_url(&url).expect("the followed object");
        assert_eq!(
            differing(classic.as_ref(), followed.as_ref()),
            [
                "breaker",
                "cards",
                "http_client_handed",
                "retry",
                "spec.cost",
                "spec.kv_connector",
                "spec.kv_role",
                "spec.models",
                "spec.priority",
                "zmq_request_timeout",
            ]
        );
    }

    /// The registry's object is the one the job built, by pointer, once
    /// the gateway's own line ran. It is still that object after a status
    /// change and a publication, and its entry names the row. A second
    /// job that moves no fact leaves its own object there.
    #[tokio::test]
    async fn the_jobs_object_stands_until_the_next_job_hands_its_own() {
        let url = serving_worker().await;
        let (config, spec) = stated_beside_the_row(&url);
        let (store, registry, feed) = started(&config).expect("the store starts");
        let stands = |state: &WorkflowState<WorkerWorkflowData>| {
            let built = state.context.data.actual_workers.as_deref();
            let held = registry.get_by_url(&url).expect("the row's object");
            matches!(built, Some([built]) if Arc::ptr_eq(built, &held))
        };

        let first = registered_on(&config, &registry, Some(&feed), spec.clone())
            .await
            .expect("the first run ends");
        assert!(stands(&first));

        let row = store.by_key(&url).expect("the worker's row");
        let moved = store.set_status(row.id, RowStatus::NotReady, row.revision);
        assert!(moved.is_some());
        feed.settle();
        assert!(stands(&first));
        let id = registry.get_id_by_url(&url).expect("the object's id");
        let entry = feed.objects.standing(&id).expect("the object's entry");
        assert_eq!((entry.row, entry.revision), (row.id, row.revision));

        let second = registered_on(&config, &registry, Some(&feed), spec)
            .await
            .expect("the second run ends");
        assert_eq!(second.status, WorkflowStatus::Completed);
        assert!(stands(&second));
        assert!(!stands(&first));
        assert_eq!(
            store.by_key(&url).map(|row| row.revision),
            Some(row.revision)
        );
        assert_eq!(feed.objects.unhanded(), 0);
    }

    /// Two jobs for one URL at once, on two threads. Each hands the door
    /// the object it built in its own turn: one row stands, the registry
    /// holds the object of one of the two jobs, and no lay missed its
    /// object.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[expect(
        clippy::disallowed_methods,
        reason = "the case joins both tasks before it ends"
    )]
    async fn two_jobs_for_one_url_at_once_each_hand_their_object_in() {
        let url = serving_worker().await;
        let (config, spec) = stated_beside_the_row(&url);
        let (store, registry, feed) = started(&config).expect("the store starts");

        let jobs = [spec.clone(), spec].map(|spec| {
            let (config, registry, feed) = (config.clone(), Arc::clone(&registry), feed.clone());
            tokio::spawn(async move { registered_on(&config, &registry, Some(&feed), spec).await })
        });
        let mut built = Vec::new();
        for job in jobs {
            let state = job.await.expect("the job's task").expect("the run ends");
            assert_eq!(state.status, WorkflowStatus::Completed);
            built.extend(state.context.data.actual_workers.iter().flatten().cloned());
        }

        assert_eq!(built.len(), 2);
        let object = registry.get_by_url(&url).expect("the row's object");
        assert!(built.iter().any(|built| Arc::ptr_eq(built, &object)));
        assert_eq!(store.all_rows().1.len(), 1);
        assert_eq!(feed.objects.unhanded(), 0);
    }

    /// A start-up job for a plain HTTP worker, with the health checks off:
    /// the store holds one Ready row at the worker's URL, the registry holds
    /// the object the job built, which the holder keeps as the row's, and
    /// the policy engine's next decision picks the row.
    #[tokio::test]
    async fn a_local_workers_job_ends_as_a_ready_row_and_the_holders_object() {
        let url = serve_json_routes(vec![
            ("/health", StatusCode::OK, json!({"status": "ok"})),
            (
                "/model_info",
                StatusCode::OK,
                json!({"model_path": "m", "is_generation": true}),
            ),
            ("/server_info", StatusCode::OK, json!({"version": "test"})),
        ])
        .await;
        let config = picked(config_with(5, 1));
        let (store, registry, feed) = started(&config).expect("the store starts");

        let state = registered_under_the_feed(&config, &registry, &feed, WorkerSpec::new(&url))
            .await
            .expect("the run ends");
        let errors: Vec<_> = state
            .step_states
            .iter()
            .filter_map(|(id, step)| Some((id.to_string(), step.last_error.clone()?)))
            .collect();
        assert_eq!(state.status, WorkflowStatus::Completed, "{errors:?}");
        assert!(registered_in(&state));

        let row = store.by_key(&url).expect("the worker's row");
        assert_eq!(row.status, RowStatus::Ready);
        assert_eq!(row.maker, Registrant::Control);

        let object = registry.get_by_url(&url).expect("the row's object");
        assert_eq!(feed.objects.object(row.id), registry.get_id_by_url(&url));
        assert_eq!(object.status(), WorkerStatus::Ready);
        let data = &state.context.data;
        let built = data.actual_workers.as_deref().expect("the built workers");
        assert!(matches!(built, [built] if Arc::ptr_eq(built, &object)));
        assert_eq!(feed.objects.unhanded(), 0);
        let written = data.workers.as_ref().expect("the written workers");
        assert_eq!(written.worker_urls, [url]);

        let model = object.model_id().to_string();
        let placed = PolicyEngine::new(Arc::clone(&store), Arc::clone(&feed.objects))
            .place(&Ask::of(b"first", &model), Instant::now())
            .await
            .expect("the next decision picks the worker");
        assert_eq!(placed.id(), row.id);
    }

    /// A local worker that states an API key registers as any other. A row
    /// holds no credential, and the row's object in the registry holds the
    /// key.
    #[tokio::test]
    async fn a_keyed_local_workers_job_ends_as_a_row_whose_object_holds_the_key() {
        let url = serve_json_routes(vec![
            ("/health", StatusCode::OK, json!({"status": "ok"})),
            (
                "/model_info",
                StatusCode::OK,
                json!({"model_path": "m", "is_generation": true}),
            ),
            ("/server_info", StatusCode::OK, json!({"version": "test"})),
        ])
        .await;
        let config = picked(config_with(5, 1));
        let (store, registry, feed) = started(&config).expect("the store starts");
        let mut spec = WorkerSpec::new(&url);
        spec.api_key = Some("secret-key".into());

        let state = registered_under_the_feed(&config, &registry, &feed, spec)
            .await
            .expect("the run ends");
        assert_eq!(state.status, WorkflowStatus::Completed);
        let row = store.by_key(&url).expect("the worker's row");
        assert_eq!(row.status, RowStatus::Ready);
        let object = registry.get_by_url(&url).expect("the row's object");
        assert_eq!(object.api_key().map(String::as_str), Some("secret-key"));
    }
}
