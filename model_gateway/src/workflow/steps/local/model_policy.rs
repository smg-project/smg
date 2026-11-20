//! Steps that tell the classic registry which policy a model has, under
//! the policy controller's feed.
//!
//! The document in force binds a model by the registry's word on it
//! ([`labels`]). The registration's first step runs ahead of the write of
//! a worker's row at the feed's door, since a row states what the
//! generation in force reads of it. Its second step runs behind the
//! registration, where the classic line's policy step stands: it tells
//! anew a model whose policy a crossing leave took out, and tells of the
//! models the replaced workers left. The removal's step runs behind the
//! removal of the rows, where the registry knows which models no worker
//! serves. A registration that does not finish tells of its models at the
//! job's end ([`forget_unfinished`]).

use std::sync::Arc;

use async_trait::async_trait;
use smg_worker::workflow::WorkerKind;
use tracing::warn;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::remove_from_policy_registry::removed_models;
use crate::{
    app_context::AppContext,
    policy_controller::labels,
    workflow::{
        data::{
            require_app_context, WorkerRegistrationData, WorkerRemovalWorkflowData,
            WorkerWorkflowData,
        },
        steps::shared::models_left,
    },
};

/// The registration's first step, in the line that writes at the store's
/// door.
pub(crate) const BIND_MODEL_POLICY_STEP: &str = "bind_model_policy";

/// The registration's second step, behind the registration.
pub(crate) const CONFIRM_MODEL_POLICY_STEP: &str = "confirm_model_policy";

/// The removal's step, in the line that removes at the store's door.
pub(crate) const UNBIND_MODEL_POLICY_STEP: &str = "unbind_model_policy";

/// Runs `tell` on the blocking pool: it may validate a document. A telling
/// that does not end leaves the standing document, and the line goes on.
async fn on_blocking_pool(tell: impl FnOnce() + Send + 'static) {
    if let Err(error) = tokio::task::spawn_blocking(tell).await {
        warn!("A model's policy was not told to the policy registry: {error}");
    }
}

/// What a local registration tells the registry of.
struct Told {
    context: Arc<AppContext>,
    /// The workers' models, each once.
    models: Vec<String>,
    /// The registration's `policy` label.
    label: Option<String>,
}

/// What the registration of `data` tells the registry of. `None` for a
/// provider's registration and ahead of the workers' making.
fn told_of(data: &WorkerWorkflowData) -> WorkflowResult<Option<Told>> {
    if data.worker_kind != Some(WorkerKind::Local) {
        return Ok(None);
    }
    let Some(workers) = data.actual_workers.as_ref() else {
        return Ok(None);
    };
    let context = Arc::clone(require_app_context(data.get_app_context())?);
    let mut models: Vec<String> = Vec::new();
    for worker in workers {
        if !models.iter().any(|model| model == worker.model_id()) {
            models.push(worker.model_id().to_owned());
        }
    }
    let label = data
        .get_labels()
        .and_then(|labels| labels.get("policy"))
        .cloned();
    Ok(Some(Told {
        context,
        models,
        label,
    }))
}

/// Step to tell the registry of the workers' models and their `policy`
/// label, as the classic line's policy step does, once a model. It notes
/// the models the workers this registration replaces serve, for the step
/// behind the registration: the door writes the registry ahead of that
/// step, and only the registry as it is now knows them.
pub(crate) struct BindModelPolicyStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for BindModelPolicyStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let Some(told) = told_of(&context.data)? else {
            return Ok(StepResult::Skip);
        };
        let replaced = context
            .data
            .actual_workers
            .as_deref()
            .map(|workers| models_left(&told.context.worker_registry, workers))
            .unwrap_or_default();
        context.data.host_notes.policy_models_left = replaced;
        on_blocking_pool(move || {
            labels::registered(&told.context, &told.models, told.label.as_deref());
        })
        .await;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// Step to tell the registry again behind the registration, where the
/// classic line's policy step stands. A model whose policy a crossing
/// leave took out is told anew, and each model the replaced workers left
/// is told as at a removal.
pub(crate) struct ConfirmModelPolicyStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for ConfirmModelPolicyStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let Some(told) = told_of(&context.data)? else {
            return Ok(StepResult::Skip);
        };
        let mut replaced: Vec<String> = context
            .data
            .host_notes
            .policy_models_left
            .iter()
            .cloned()
            .collect();
        replaced.sort_unstable();
        on_blocking_pool(move || {
            labels::confirmed(
                &told.context,
                &told.models,
                told.label.as_deref(),
                &replaced,
            );
        })
        .await;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// Step to tell the registry of the models no worker of this node serves
/// any more, as the classic line's removal does.
pub(crate) struct UnbindModelPolicyStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for UnbindModelPolicyStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = Arc::clone(require_app_context(context.data.host.as_ref())?);
        let mut models: Vec<String> = removed_models(&context.data).into_iter().collect();
        models.sort_unstable();
        on_blocking_pool(move || labels::left(&app_context, &models)).await;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// A local registration that did not finish told the registry of its
/// models ahead of the write at the door. Each of them no worker of this
/// node serves is told as at a last leave, so the model's next first
/// worker names its policy anew.
pub(crate) async fn forget_unfinished(data: &WorkerWorkflowData) {
    let Ok(Some(mut told)) = told_of(data) else {
        return;
    };
    told.models.sort_unstable();
    on_blocking_pool(move || labels::left(&told.context, &told.models)).await;
}

#[cfg(test)]
mod tests {
    use openai_protocol::{model_card::ModelCard, worker::WorkerSpec};
    use smg_policy_controller::view::Phase;
    use smg_worker::{identity::basic::Worker, jobs::WorkerJobAsker};
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::AppContext,
        config::{PolicyConfig, RouterConfig},
        routers::grpc::backend_client::BasicWorkerBuilder,
        workflow::{
            data::WorkerRegistrationMode,
            steps::{create_worker_removal_workflow_data, create_worker_workflow_data},
        },
    };

    const MODEL: &str = "qwen3";

    /// A context under the controller whose main policy is round robin.
    async fn context() -> Arc<AppContext> {
        let mut config = RouterConfig {
            policy: PolicyConfig::RoundRobin,
            ..RouterConfig::default()
        };
        config.policy_controller = true;
        config.health_check.disable_health_check = true;
        let context = AppContext::from_config(config, 30, None, None)
            .await
            .expect("a context under the controller");
        Arc::new(context)
    }

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(MODEL))
                .build(),
        )
    }

    /// The generation the store runs, and the profile that decides the
    /// model's single requests under it.
    fn decided_by(context: &AppContext) -> (u64, String) {
        let store = context
            .policy_store
            .as_ref()
            .expect("the controller's store");
        store.with_world(|world| {
            let spec = world.spec();
            let (_, profile) = spec
                .engine()
                .profile_for(MODEL, Phase::Single)
                .expect("a profile decides the model");
            (spec.generation(), profile.name.clone())
        })
    }

    /// A registration's run over `workers` of `kind`, registered under the
    /// `policy` label.
    fn registration(
        context: &Arc<AppContext>,
        kind: WorkerKind,
        workers: Vec<Arc<dyn Worker>>,
        label: Option<&str>,
    ) -> WorkflowContext<WorkerWorkflowData> {
        let mut data = create_worker_workflow_data(
            WorkerSpec::new("http://127.0.0.1:1"),
            WorkerRegistrationMode::Upsert,
            Arc::clone(context),
            WorkerJobAsker::Api,
        );
        data.worker_kind = Some(kind);
        data.final_labels = label
            .into_iter()
            .map(|label| ("policy".to_owned(), label.to_owned()))
            .collect();
        data.actual_workers = Some(workers);
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    /// Runs the registration's first step over `workers` of `kind`,
    /// registered under the `policy` label.
    async fn registered(
        context: &Arc<AppContext>,
        kind: WorkerKind,
        workers: Vec<Arc<dyn Worker>>,
        label: Option<&str>,
    ) -> StepResult {
        let mut run = registration(context, kind, workers, label);
        BindModelPolicyStep
            .execute(&mut run)
            .await
            .expect("the step ends")
    }

    /// Runs the removal's step behind the removal of `worker`.
    async fn removed(context: &Arc<AppContext>, worker: &Arc<dyn Worker>) {
        let mut data = create_worker_removal_workflow_data(
            worker.url().to_owned(),
            None,
            Arc::clone(context),
            WorkerJobAsker::Api,
        );
        data.removed_workers = vec![Arc::clone(worker)];
        let mut run = WorkflowContext::new(WorkflowInstanceId::new(), data);
        UnbindModelPolicyStep
            .execute(&mut run)
            .await
            .expect("the step ends");
    }

    /// The registration's step tells the registry of the workers' model
    /// and label, and the model is decided by the label's policy under one
    /// generation more, however many workers the registration holds.
    #[tokio::test]
    async fn the_registrations_step_binds_a_labelled_model_once_for_its_workers() {
        let context = context().await;
        let two = vec![worker("http://127.0.0.1:1"), worker("http://127.0.0.1:2")];
        let ended = registered(&context, WorkerKind::Local, two, Some("consistent_hashing")).await;
        assert!(matches!(ended, StepResult::Success));
        assert_eq!(decided_by(&context), (2, "consistent_hashing".to_owned()));
    }

    /// A provider's registration is no local worker's: the step tells the
    /// registry nothing of it.
    #[tokio::test]
    async fn the_registrations_step_leaves_a_providers_registration_alone() {
        let context = context().await;
        let one = vec![worker("http://127.0.0.1:1")];
        let ended = registered(
            &context,
            WorkerKind::External,
            one,
            Some("consistent_hashing"),
        )
        .await;
        assert!(matches!(ended, StepResult::Skip));
        assert!(context.policy_registry.get_policy(MODEL).is_none());
        assert_eq!(decided_by(&context), (1, "round_robin".to_owned()));
    }

    /// The removal's step asks the registry which models no worker serves
    /// any more. A model another worker still serves keeps its policy, and
    /// the model's last leave takes its binding out.
    #[tokio::test]
    async fn the_removals_step_unbinds_a_model_at_its_last_leave() {
        let context = context().await;
        let first = worker("http://127.0.0.1:1");
        let second = worker("http://127.0.0.1:2");
        let both = vec![Arc::clone(&first), Arc::clone(&second)];
        registered(
            &context,
            WorkerKind::Local,
            both,
            Some("consistent_hashing"),
        )
        .await;
        context
            .worker_registry
            .register(Arc::clone(&second))
            .expect("the second worker registers");
        removed(&context, &first).await;
        assert_eq!(decided_by(&context), (2, "consistent_hashing".to_owned()));
        context
            .worker_registry
            .remove_by_url(second.url())
            .expect("the second worker leaves");
        removed(&context, &second).await;
        assert_eq!(decided_by(&context), (3, "round_robin".to_owned()));
        assert!(context.policy_registry.get_policy(MODEL).is_none());
    }

    /// A model's last leave that crosses a registration of the same model
    /// takes the policy out between the registration's two steps: the first
    /// found it standing, and the new worker stood in no registry yet. The
    /// second step tells the model anew, and a leave that finds the new
    /// worker in the registry moves nothing.
    #[tokio::test]
    async fn the_second_step_tells_anew_a_model_a_crossing_leave_took_out() {
        let context = context().await;
        let old = worker("http://127.0.0.1:1");
        let one = vec![Arc::clone(&old)];
        registered(&context, WorkerKind::Local, one, Some("consistent_hashing")).await;
        let new = worker("http://127.0.0.1:2");
        let mut run = registration(
            &context,
            WorkerKind::Local,
            vec![Arc::clone(&new)],
            Some("consistent_hashing"),
        );
        BindModelPolicyStep
            .execute(&mut run)
            .await
            .expect("the first step ends");
        assert_eq!(decided_by(&context), (2, "consistent_hashing".to_owned()));

        // The old worker's leave ends ahead of the new worker's write.
        removed(&context, &old).await;
        assert_eq!(decided_by(&context), (3, "round_robin".to_owned()));
        context
            .worker_registry
            .register(Arc::clone(&new))
            .expect("the new worker registers");
        let ended = ConfirmModelPolicyStep
            .execute(&mut run)
            .await
            .expect("the second step ends");
        assert!(matches!(ended, StepResult::Success));
        assert_eq!(decided_by(&context), (4, "consistent_hashing".to_owned()));

        removed(&context, &old).await;
        assert_eq!(decided_by(&context), (4, "consistent_hashing".to_owned()));
    }

    /// A registration that replaces a model's last worker with a worker of
    /// another model tells the registry of the model it left, as the
    /// classic line's register step does. The first step notes the models
    /// the replaced workers serve, and the second tells of each one no
    /// worker serves any more.
    #[tokio::test]
    async fn a_replacement_tells_of_the_model_its_worker_left() {
        const URL: &str = "http://127.0.0.1:1";
        let context = context().await;
        let first = worker(URL);
        let one = vec![Arc::clone(&first)];
        registered(&context, WorkerKind::Local, one, Some("consistent_hashing")).await;
        context
            .worker_registry
            .register(Arc::clone(&first))
            .expect("the first worker registers");

        let second: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(URL)
                .model(ModelCard::new("llama3"))
                .build(),
        );
        let mut run = registration(&context, WorkerKind::Local, vec![Arc::clone(&second)], None);
        BindModelPolicyStep
            .execute(&mut run)
            .await
            .expect("the first step ends");
        assert!(run.data.host_notes.policy_models_left.contains(MODEL));
        assert_eq!(decided_by(&context), (2, "consistent_hashing".to_owned()));

        // The door's write lays the new object in the old one's place.
        context
            .worker_registry
            .remove_by_url(URL)
            .expect("the first worker is replaced");
        context
            .worker_registry
            .register(second)
            .expect("the second worker registers");
        ConfirmModelPolicyStep
            .execute(&mut run)
            .await
            .expect("the second step ends");
        assert!(context.policy_registry.get_policy(MODEL).is_none());
        assert_eq!(decided_by(&context), (3, "round_robin".to_owned()));
    }

    /// A registration that did not finish told the registry of its model
    /// ahead of the write. At the job's end a model no worker serves loses
    /// its policy, and a model a worker serves keeps it.
    #[tokio::test]
    async fn an_unfinished_registration_tells_of_the_model_no_worker_serves() {
        let context = context().await;
        let failed = worker("http://127.0.0.1:1");
        let mut run = registration(
            &context,
            WorkerKind::Local,
            vec![failed],
            Some("consistent_hashing"),
        );
        BindModelPolicyStep
            .execute(&mut run)
            .await
            .expect("the first step ends");
        assert_eq!(decided_by(&context), (2, "consistent_hashing".to_owned()));
        forget_unfinished(&run.data).await;
        assert!(context.policy_registry.get_policy(MODEL).is_none());
        assert_eq!(decided_by(&context), (3, "round_robin".to_owned()));

        BindModelPolicyStep
            .execute(&mut run)
            .await
            .expect("the first step ends");
        context
            .worker_registry
            .register(worker("http://127.0.0.1:2"))
            .expect("a worker of the model registers");
        forget_unfinished(&run.data).await;
        assert_eq!(decided_by(&context), (4, "consistent_hashing".to_owned()));
    }
}
