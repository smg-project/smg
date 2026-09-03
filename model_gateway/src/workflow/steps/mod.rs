pub mod classify;
pub mod external;
pub mod local;
pub mod shared;
pub(crate) mod util;

use std::{sync::Arc, time::Duration};

pub use classify::ClassifyWorkerTypeStep;
pub use external::{
    group_models_into_cards, infer_model_type_from_id, CreateExternalWorkersStep,
    DiscoverModelsStep, ModelInfo, ModelsResponse,
};
pub use local::{
    create_worker_removal_workflow, create_worker_removal_workflow_data,
    create_worker_update_workflow, create_worker_update_workflow_data, CreateLocalWorkerStep,
    DetectConnectionModeStep, DiscoverDPInfoStep, DiscoverMetadataStep, DpInfo,
    FindWorkerToUpdateStep, FindWorkersToRemoveStep, RemoveFromPolicyRegistryStep,
    RemoveFromWorkerRegistryStep, UpdatePoliciesForWorkerStep, UpdateRemainingPoliciesStep,
    UpdateWorkerPropertiesStep, WorkerRemovalRequest,
};
use local::{
    DetectBackendStep, DiscoverDPInfoStep as DPStep, EnsureHarmonyEncodingStep,
    SubmitTokenizerJobStep,
};
use openai_protocol::worker::WorkerSpec;
pub use shared::{ActivateWorkersStep, RegisterWorkersStep, UpdatePoliciesStep, WorkerList};
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
///  create_local_worker           create_external_workers
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
    let detect_timeout = Duration::from_secs(router_config.worker_startup_timeout_secs);
    let startup_delay = Duration::from_secs(router_config.worker_startup_delay_secs);
    let check_interval = Duration::from_secs(router_config.worker_startup_check_interval_secs);

    // Startup detection polls the starting engine every
    // `worker_startup_check_interval_secs` until `worker_startup_timeout_secs`
    // elapses. Derive the retry attempt budget from that cadence (reserving
    // 10% of the timeout for workflow overhead like step transitions) so the
    // two knobs stay consistent: total wait ≈ max_attempts × check_interval ≈
    // worker_startup_timeout_secs. A floor keeps a very short timeout retrying
    // a few times.
    const EFFECTIVE_TIMEOUT_FACTOR: f64 = 0.9;
    const MIN_ATTEMPTS: u32 = 3;

    let interval_secs = (check_interval.as_secs() as f64).max(1.0);
    let effective_timeout = detect_timeout.as_secs() as f64 * EFFECTIVE_TIMEOUT_FACTOR;
    let max_attempts = ((effective_timeout / interval_secs).ceil() as u32).max(MIN_ATTEMPTS);

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

    WorkflowDefinition::new("worker_registration", "Worker Registration")
        // Step 0: Classify worker type (Local vs External)
        .add_step(classify_worker_type)
        // === LOCAL BRANCH ===
        // Step 1: Detect connection mode (HTTP vs gRPC)
        .add_step(
            StepDefinition::new(
                "detect_connection_mode",
                "Detect Connection Mode",
                Arc::new(DetectConnectionModeStep),
            )
            .with_retry(RetryPolicy {
                max_attempts,
                // Poll the starting engine at the configured startup cadence.
                backoff: BackoffStrategy::Fixed(check_interval),
            })
            .with_timeout(detect_timeout)
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["classify_worker_type"]),
        )
        // Step 1.5: Detect backend runtime (sglang, vllm, trtllm)
        .add_step(
            StepDefinition::new(
                "detect_backend",
                "Detect Backend",
                Arc::new(DetectBackendStep),
            )
            .with_retry(RetryPolicy {
                max_attempts,
                backoff: BackoffStrategy::Linear {
                    increment: Duration::from_secs(1),
                    max: Duration::from_secs(5),
                },
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["detect_connection_mode"]),
        )
        // Step 2a: Discover metadata
        .add_step(
            StepDefinition::new(
                "discover_metadata",
                "Discover Metadata",
                Arc::new(DiscoverMetadataStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["detect_backend"]),
        )
        // Step 2b: Discover DP info (after metadata)
        .add_step(
            StepDefinition::new("discover_dp_info", "Discover DP Info", Arc::new(DPStep))
                .with_retry(RetryPolicy {
                    max_attempts: 3,
                    backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
                })
                .with_timeout(Duration::from_secs(10))
                .with_failure_action(FailureAction::FailWorkflow)
                .depends_on(&["discover_metadata"]),
        )
        // Step 3 (local): Create local worker(s)
        .add_step(
            StepDefinition::new(
                "create_local_worker",
                "Create Local Worker",
                Arc::new(CreateLocalWorkerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["discover_dp_info"]),
        )
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
            .depends_on(&["create_local_worker"]),
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
                "register_workers",
                "Register Workers",
                Arc::new(RegisterWorkersStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&[
                "create_local_worker",
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
            .depends_on(&["register_workers"]),
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
            .depends_on(&["register_workers"]),
        )
        // Step 5c: Activate workers
        .add_step(
            StepDefinition::new(
                "activate_workers",
                "Activate Workers",
                Arc::new(ActivateWorkersStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["register_workers"]),
        )
}

/// Create initial workflow data for the unified worker registration workflow.
pub fn create_worker_workflow_data(
    config: WorkerSpec,
    registration_mode: WorkerRegistrationMode,
    app_context: Arc<AppContext>,
) -> WorkerWorkflowData {
    WorkerWorkflowData {
        config,
        registration_mode,
        worker_kind: None,
        connection_mode: None,
        http2: None,
        http_client_handle: None,
        detected_runtime_type: None,
        discovered_labels: std::collections::HashMap::new(),
        dp_info: None,
        model_cards: Vec::new(),
        workers: None,
        final_labels: std::collections::HashMap::new(),
        app_context: Some(app_context),
        actual_workers: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RouterConfig;

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
    }

    #[test]
    fn zero_startup_delay_leaves_probe_immediate() {
        // Default (0) preserves the immediate-check behavior: no delay is set.
        let config = config_with(1800, 30);
        assert_eq!(config.worker_startup_delay_secs, 0);
        assert_eq!(startup_delay(&config), None);
    }
}
