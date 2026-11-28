pub mod classify;
pub mod external;
pub mod local;
pub mod shared;
#[cfg(test)]
#[path = "../../../tests/unit/workflow/steps/util.rs"]
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
#[path = "../../../tests/unit/workflow/steps/policy_lifecycle_tests/mod.rs"]
mod policy_lifecycle_tests;

#[cfg(test)]
#[path = "../../../tests/unit/workflow/steps/tests.rs"]
mod tests;
