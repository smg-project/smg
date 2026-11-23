mod check_local_worker;
mod ensure_harmony_encoding;
mod find_worker_to_update;
mod forget_cache_events;
mod rediscover_labels;
mod release_tokenizer;
mod remove_from_policy_registry;
pub(crate) mod submit_tokenizer_job;
#[cfg(test)]
pub(crate) mod test_support;
mod update_policies_for_worker;
mod update_remaining_policies;
mod update_worker_properties;
mod zmq_labels;

use std::{sync::Arc, time::Duration};

pub(crate) use check_local_worker::{CheckLocalWorkerStep, CHECK_LOCAL_WORKER_STEP};
pub use ensure_harmony_encoding::EnsureHarmonyEncodingStep;
pub use find_worker_to_update::FindWorkerToUpdateStep;
use openai_protocol::worker::WorkerUpdateRequest;
pub use rediscover_labels::RediscoverLabelsStep;
pub use release_tokenizer::ReleaseTokenizerStep;
pub use remove_from_policy_registry::RemoveFromPolicyRegistryStep;
use smg_worker::{
    jobs::WorkerJobAsker,
    workflow::{
        single_attempt, worker_removal_workflow, RemovalOrder, WorkerRemovalRequest,
        REMOVE_FROM_WORKER_REGISTRY_STEP,
    },
};
pub use submit_tokenizer_job::SubmitTokenizerJobStep;
pub use update_policies_for_worker::UpdatePoliciesForWorkerStep;
pub use update_remaining_policies::UpdateRemainingPoliciesStep;
pub use update_worker_properties::UpdateWorkerPropertiesStep;
use wfaas::{BackoffStrategy, RetryPolicy, StepDefinition, WorkflowDefinition};
pub(crate) use zmq_labels::{DiscoverZmqLabelsStep, ZMQ_LABELS_STEP};

use crate::{
    app_context::AppContext,
    workflow::data::{WorkerRemovalWorkflowData, WorkerUpdateWorkflowData},
};

/// Create a worker removal workflow definition: the worker crate's removal
/// ([`worker_removal_workflow`]) with this gateway's two steps after it.
///
/// DAG structure:
/// ```text
///     find_workers_to_remove
///              │
///         drain_workers
///              │
///     remove_from_worker_registry
///                  │
///       +----------+----------+
///       │                     │
///       v                     v
///  remove_from_policy_registry   release_tokenizer
/// ```
/// Cache cleanup removes only departed tenants, so surviving workers do not
/// need the previous full-policy reinitialization pass. The tokenizer release
/// waits on the registry step alone: it reads the worker registry and the
/// record of worker-loaded entries, both settled there, and a policy cleanup
/// that fails or times out must not keep a departed model's tokenizer
/// resident (a failed step's dependents are never scheduled).
pub fn create_worker_removal_workflow() -> WorkflowDefinition<WorkerRemovalWorkflowData> {
    worker_removal_workflow(RemovalOrder::Registry)
        .add_step(
            StepDefinition::new(
                "remove_from_policy_registry",
                "Remove workers from policy registry",
                Arc::new(RemoveFromPolicyRegistryStep),
            )
            // The old 10s override was too short for large cache cleanup.
            // Inherit the workflow default after the worker is deregistered.
            .with_retry(single_attempt())
            .depends_on(&[REMOVE_FROM_WORKER_REGISTRY_STEP]),
        )
        .add_step(
            StepDefinition::new(
                "release_tokenizer",
                "Release the tokenizers of the models no worker serves any more",
                Arc::new(ReleaseTokenizerStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(single_attempt())
            // On the registry step, not the policy cleanup: the release reads
            // what that step settled, and a cleanup that fails or times out
            // must not hold it back (a failed step's dependents never run).
            .depends_on(&[REMOVE_FROM_WORKER_REGISTRY_STEP]),
        )
}

/// Create a worker update workflow definition.
///
/// DAG structure:
/// ```text
///     find_worker_to_update
///              │
///     update_worker_properties
///              │
///     update_policies_for_worker
/// ```
pub fn create_worker_update_workflow() -> WorkflowDefinition<WorkerUpdateWorkflowData> {
    WorkflowDefinition::new("worker_update", "Update worker properties")
        .add_step(
            StepDefinition::new(
                "find_worker_to_update",
                "Find worker to update",
                Arc::new(FindWorkerToUpdateStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(single_attempt()),
        )
        .add_step(
            StepDefinition::new(
                "rediscover_labels",
                "Re-read the worker's labels from its engine (label refresh only)",
                Arc::new(RediscoverLabelsStep),
            )
            .with_timeout(Duration::from_secs(10))
            // An engine that just came back may still be warming up: one more
            // try before the refresh is left to the next transition.
            .with_retry(RetryPolicy {
                max_attempts: 2,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(3)),
            })
            .depends_on(&["find_worker_to_update"]),
        )
        .add_step(
            StepDefinition::new(
                "update_worker_properties",
                "Update worker properties",
                Arc::new(UpdateWorkerPropertiesStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(single_attempt())
            .depends_on(&["rediscover_labels"]),
        )
        .add_step(
            StepDefinition::new(
                "update_policies_for_worker",
                "Update policies for updated worker",
                Arc::new(UpdatePoliciesForWorkerStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(single_attempt())
            .depends_on(&["update_worker_properties"]),
        )
}

/// Helper to create initial workflow data for worker removal
pub fn create_worker_removal_workflow_data(
    url: String,
    expected_revisions: Option<std::collections::HashMap<String, Option<u64>>>,
    app_context: Arc<AppContext>,
    asker: WorkerJobAsker,
) -> WorkerRemovalWorkflowData {
    WorkerRemovalWorkflowData {
        config: WorkerRemovalRequest {
            url,
            expected_revisions,
        },
        asker,
        workers_to_remove: None,
        worker_urls: Vec::new(),
        affected_models: std::collections::HashSet::new(),
        host: Some(app_context),
        actual_workers_to_remove: None,
        drained_workers: Vec::new(),
        removed_workers: Vec::new(),
    }
}

/// Helper to create initial workflow data for worker update
pub fn create_worker_update_workflow_data(
    worker_url: String,
    update_config: WorkerUpdateRequest,
    app_context: Arc<AppContext>,
    asker: WorkerJobAsker,
) -> WorkerUpdateWorkflowData {
    WorkerUpdateWorkflowData {
        config: update_config,
        asker,
        worker_url,
        app_context: Some(app_context),
        workers_to_update: None,
        updated_workers: None,
        refresh_discovered_labels: false,
        fresh_engine_labels: None,
        snapshot: None,
    }
}

#[cfg(test)]
mod tests {
    use llm_tokenizer::TokenizerRegistry;
    use openai_protocol::model_card::ModelCard;
    use smg_worker::identity::basic::{ConnectionMode, Worker};
    use wfaas::{
        StepExecutor, StepId, StepResult, StepStatus, WorkflowContext, WorkflowEngine,
        WorkflowError, WorkflowId, WorkflowInstanceId, WorkflowResult, WorkflowState,
        WorkflowStatus,
    };

    use super::*;
    use crate::{
        app_context::testing::app_context_for_tests,
        config::RouterConfig,
        routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder},
        workflow::tokenizer_registration::{
            create_tokenizer_workflow_data, test_support::tokenizer_dir, LoadTokenizerStep,
            TokenizerConfigRequest,
        },
    };

    /// The removal workflow as a whole, its steps run in the definition's
    /// order, on a tokenizer the registration step loaded for a worker: it
    /// stays while a worker of the model remains and leaves the registry
    /// with the model's last worker.
    #[tokio::test(start_paused = true)]
    async fn removing_a_models_last_worker_releases_its_tokenizer() {
        let app_context = app_context_for_tests(
            RouterConfig::builder()
                .round_robin_policy()
                .build_unchecked(),
            Arc::new(WorkerRegistry::new()),
            None,
        );
        for url in ["http://w1:8000", "http://w2:8000"] {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new("m"))
                    .label("tokenizer_path", "/tok/m")
                    .connection_mode(ConnectionMode::Grpc)
                    .build(),
            );
            app_context.worker_registry.register(worker);
        }
        let dir = tokenizer_dir();
        let mut registration = WorkflowContext::new(
            WorkflowInstanceId::new(),
            create_tokenizer_workflow_data(
                TokenizerConfigRequest {
                    id: TokenizerRegistry::generate_id(),
                    name: "m".to_string(),
                    source: dir.path().to_str().unwrap().to_string(),
                    chat_template_path: None,
                    cache_config: None,
                    fail_on_duplicate: false,
                    for_worker: true,
                },
                Arc::clone(&app_context),
            ),
        );
        LoadTokenizerStep.execute(&mut registration).await.unwrap();

        for (url, remaining) in [("http://w1:8000", true), ("http://w2:8000", false)] {
            let workflow = create_worker_removal_workflow();
            let mut context = WorkflowContext::new(
                WorkflowInstanceId::new(),
                create_worker_removal_workflow_data(
                    url.to_string(),
                    None,
                    Arc::clone(&app_context),
                    WorkerJobAsker::Api,
                ),
            );
            for step in &workflow.steps {
                step.executor.execute(&mut context).await.unwrap();
            }
            assert_eq!(
                app_context.worker_registry.contains_model("m"),
                remaining,
                "a worker of the model after removing {url}"
            );
            assert_eq!(
                app_context.tokenizer_registry.contains("m"),
                remaining,
                "the model's tokenizer after removing {url}"
            );
        }
    }

    #[test]
    fn worker_removal_deregisters_before_policy_cleanup() {
        let workflow = create_worker_removal_workflow();
        let registry_step = workflow
            .steps
            .iter()
            .find(|step| step.id == StepId::new("remove_from_worker_registry"))
            .unwrap();
        let policy_step = workflow
            .steps
            .iter()
            .find(|step| step.id == StepId::new("remove_from_policy_registry"))
            .unwrap();

        assert_eq!(registry_step.depends_on, vec![StepId::new("drain_workers")]);
        assert_eq!(
            policy_step.depends_on,
            vec![StepId::new("remove_from_worker_registry")]
        );
        assert_eq!(policy_step.timeout, None);
        assert!(workflow
            .steps
            .iter()
            .all(|step| step.id != StepId::new("update_remaining_policies")));
    }

    #[test]
    fn the_tokenizer_release_waits_on_the_registry_step_alone() {
        let workflow = create_worker_removal_workflow();
        let release_step = workflow
            .steps
            .iter()
            .find(|step| step.id == StepId::new("release_tokenizer"))
            .unwrap();

        assert_eq!(
            release_step.depends_on,
            vec![StepId::new("remove_from_worker_registry")]
        );
    }

    /// A policy cleanup that fails, in the step's place.
    struct FailingPolicyCleanup;

    #[async_trait::async_trait]
    impl StepExecutor<WorkerRemovalWorkflowData> for FailingPolicyCleanup {
        async fn execute(
            &self,
            _context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
        ) -> WorkflowResult<StepResult> {
            Err(WorkflowError::StepFailed {
                step_id: StepId::new("remove_from_policy_registry"),
                message: "the cleanup timed out".to_string(),
            })
        }

        fn is_retryable(&self, _error: &WorkflowError) -> bool {
            false
        }
    }

    /// The state once the engine has no step pending or running, read
    /// without the engine's own waiter (which cleans a terminal state up).
    async fn settled(
        engine: &WorkflowEngine<WorkerRemovalWorkflowData>,
        instance_id: WorkflowInstanceId,
    ) -> WorkflowState<WorkerRemovalWorkflowData> {
        let mut state = engine.get_status(instance_id).await.unwrap();
        for _ in 0..600 {
            if state.status != WorkflowStatus::Running && state.status != WorkflowStatus::Pending {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            state = engine.get_status(instance_id).await.unwrap();
        }
        state
    }

    /// The removal workflow through the engine with a policy cleanup that
    /// fails: the workflow ends failed, and the departed model's tokenizer
    /// leaves the registry all the same, since the release depends on the
    /// registry step and not on the cleanup (a failed step's dependents are
    /// never scheduled).
    #[tokio::test(start_paused = true)]
    async fn a_failed_policy_cleanup_does_not_keep_the_departed_models_tokenizer() {
        let app_context = app_context_for_tests(
            RouterConfig::builder()
                .round_robin_policy()
                .build_unchecked(),
            Arc::new(WorkerRegistry::new()),
            None,
        );
        for url in ["http://w1:8000", "http://w2:8000"] {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new("m"))
                    .label("tokenizer_path", "/tok/m")
                    .connection_mode(ConnectionMode::Grpc)
                    .build(),
            );
            app_context.worker_registry.register(worker);
        }
        let dir = tokenizer_dir();
        let mut registration = WorkflowContext::new(
            WorkflowInstanceId::new(),
            create_tokenizer_workflow_data(
                TokenizerConfigRequest {
                    id: TokenizerRegistry::generate_id(),
                    name: "m".to_string(),
                    source: dir.path().to_str().unwrap().to_string(),
                    chat_template_path: None,
                    cache_config: None,
                    fail_on_duplicate: false,
                    for_worker: true,
                },
                Arc::clone(&app_context),
            ),
        );
        LoadTokenizerStep.execute(&mut registration).await.unwrap();

        let mut workflow = create_worker_removal_workflow();
        workflow
            .steps
            .iter_mut()
            .find(|step| step.id == StepId::new("remove_from_policy_registry"))
            .unwrap()
            .executor = Arc::new(FailingPolicyCleanup);
        let engine: WorkflowEngine<WorkerRemovalWorkflowData> = WorkflowEngine::new();
        engine.register_workflow(workflow).unwrap();

        for (url, remaining) in [("http://w1:8000", true), ("http://w2:8000", false)] {
            let instance_id = engine
                .start_workflow(
                    WorkflowId::new("worker_removal"),
                    create_worker_removal_workflow_data(
                        url.to_string(),
                        None,
                        Arc::clone(&app_context),
                        WorkerJobAsker::Api,
                    ),
                )
                .await
                .unwrap();
            let state = settled(&engine, instance_id).await;
            assert_eq!(
                state.status,
                WorkflowStatus::Failed,
                "the workflow after removing {url}"
            );
            assert_eq!(
                state.step_states[&StepId::new("remove_from_policy_registry")].status,
                StepStatus::Failed,
                "the policy cleanup after removing {url}"
            );
            assert_eq!(
                state.step_states[&StepId::new("release_tokenizer")].status,
                StepStatus::Succeeded,
                "the release after removing {url}"
            );
            assert_eq!(
                app_context.worker_registry.contains_model("m"),
                remaining,
                "a worker of the model after removing {url}"
            );
            assert_eq!(
                app_context.tokenizer_registry.contains("m"),
                remaining,
                "the model's tokenizer after removing {url}"
            );
        }
    }
}
