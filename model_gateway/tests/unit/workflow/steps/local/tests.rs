use llm_tokenizer::TokenizerRegistry;
use openai_protocol::model_card::ModelCard;
use smg_worker::identity::basic::{ConnectionMode, Worker};
use wfaas::{
    StepExecutor, StepId, StepResult, StepStatus, WorkflowContext, WorkflowEngine, WorkflowError,
    WorkflowId, WorkflowInstanceId, WorkflowResult, WorkflowState, WorkflowStatus,
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
