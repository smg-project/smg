//! A model's policy across the worker workflows: built from the `policy`
//! label of the model's first worker, kept through registrations, updates
//! and replacements of its workers, and dropped with the last worker this
//! node owns, so the model's next registration builds the policy its own
//! labels name.

use std::sync::Arc;

use openai_protocol::worker::{WorkerSpec, WorkerUpdateRequest};
use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

use super::{
    create_worker_removal_workflow_data, create_worker_update_workflow_data,
    create_worker_workflow_data, RegisterWorkersStep, RemoveFromPolicyRegistryStep,
    RemoveFromWorkerRegistryStep, UpdatePoliciesForWorkerStep, UpdatePoliciesStep,
};
use crate::{
    app_context::{testing::app_context_for_tests, AppContext},
    config::RouterConfig,
    worker::{BasicWorkerBuilder, ModelCard, Worker, WorkerRegistry},
    workflow::data::WorkerRegistrationMode,
};

fn app_context() -> Arc<AppContext> {
    app_context_for_tests(
        RouterConfig::builder()
            .round_robin_policy()
            .build_unchecked(),
        Arc::new(WorkerRegistry::new()),
        None,
    )
}

fn worker(url: &str, model: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model))
            .build(),
    )
}

/// A worker serving several models, as an external endpoint's worker lists
/// every model its provider serves; the first is its primary model.
fn endpoint(url: &str, models: &[&str]) -> Arc<dyn Worker> {
    let cards: Vec<ModelCard> = models.iter().map(|model| ModelCard::new(*model)).collect();
    Arc::new(BasicWorkerBuilder::new(url).models(cards).build())
}

/// The model's own policy entry, `None` once it is dropped: routing would
/// fall back to the default policy, this does not.
fn policy_of(app_context: &AppContext, model: &str) -> Option<&'static str> {
    app_context
        .policy_registry
        .get_policy(model)
        .map(|policy| policy.name())
}

/// Register `worker` with `policy` as its `policy` label the way the
/// registration workflow does: the registry write under `mode`, then the
/// policies.
async fn register_as(
    app_context: &Arc<AppContext>,
    worker: Arc<dyn Worker>,
    policy: &str,
    mode: WorkerRegistrationMode,
) {
    let mut data =
        create_worker_workflow_data(WorkerSpec::new(worker.url()), mode, Arc::clone(app_context));
    data.final_labels
        .insert("policy".to_string(), policy.to_string());
    data.actual_workers = Some(vec![worker]);
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    RegisterWorkersStep.execute(&mut context).await.unwrap();
    UpdatePoliciesStep.execute(&mut context).await.unwrap();
}

/// `POST /workers`, a startup or discovery registration of a known URL.
async fn register(app_context: &Arc<AppContext>, worker: Arc<dyn Worker>, policy: &str) {
    register_as(app_context, worker, policy, WorkerRegistrationMode::Upsert).await;
}

/// A `PATCH` or a label refresh: both end in the update workflow's
/// policies step.
async fn update(app_context: &Arc<AppContext>, worker: Arc<dyn Worker>) {
    let request = WorkerUpdateRequest {
        priority: None,
        cost: None,
        labels: None,
        api_key: None,
        health: None,
    };
    let mut data = create_worker_update_workflow_data(
        worker.url().to_string(),
        request,
        Arc::clone(app_context),
    );
    data.updated_workers = Some(vec![worker]);
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    UpdatePoliciesForWorkerStep
        .execute(&mut context)
        .await
        .unwrap();
}

/// `DELETE /workers/{id}`: out of the worker registry, then the policies.
async fn remove(app_context: &Arc<AppContext>, worker: Arc<dyn Worker>) {
    let url = worker.url().to_string();
    let mut data = create_worker_removal_workflow_data(url.clone(), None, Arc::clone(app_context));
    data.worker_urls = vec![url];
    data.actual_workers_to_remove = Some(vec![worker]);
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    RemoveFromWorkerRegistryStep
        .execute(&mut context)
        .await
        .unwrap();
    RemoveFromPolicyRegistryStep
        .execute(&mut context)
        .await
        .unwrap();
}

#[tokio::test]
async fn the_policy_leaves_with_the_last_worker_however_often_it_was_updated() {
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    update(&app_context, Arc::clone(&llama)).await;
    update(&app_context, Arc::clone(&llama)).await;

    remove(&app_context, llama).await;
    assert_eq!(policy_of(&app_context, "llama"), None);

    // The model's next first worker names its own policy.
    register(&app_context, worker("http://w1:8000", "llama"), "random").await;
    assert_eq!(policy_of(&app_context, "llama"), Some("random"));
}

#[tokio::test]
async fn a_registration_of_a_known_url_is_the_same_worker() {
    let app_context = app_context();
    register(
        &app_context,
        worker("http://w1:8000", "llama"),
        "cache_aware",
    )
    .await;
    let again = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&again), "cache_aware").await;

    remove(&app_context, again).await;
    assert_eq!(policy_of(&app_context, "llama"), None);
}

#[tokio::test]
async fn the_policy_stays_while_another_worker_serves_the_model() {
    let app_context = app_context();
    let first = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&first), "cache_aware").await;
    let second = worker("http://w2:8000", "llama");
    register(&app_context, Arc::clone(&second), "cache_aware").await;

    remove(&app_context, first).await;
    assert_eq!(policy_of(&app_context, "llama"), Some("cache_aware"));
    remove(&app_context, second).await;
    assert_eq!(policy_of(&app_context, "llama"), None);
}

#[tokio::test]
async fn a_replacement_on_another_model_takes_the_old_models_policy() {
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;

    // `PUT /workers/{id}` keeps the URL; this body serves another model.
    let replaces = WorkerRegistrationMode::ReplaceById {
        worker_id: app_context
            .worker_registry
            .get_id_by_url(llama.url())
            .expect("the worker is registered")
            .as_str()
            .to_string(),
        expected_revision: llama.revision(),
    };
    register_as(
        &app_context,
        worker("http://w1:8000", "qwen"),
        "random",
        replaces,
    )
    .await;

    assert_eq!(policy_of(&app_context, "llama"), None);
    assert_eq!(policy_of(&app_context, "qwen"), Some("random"));
}

#[tokio::test]
async fn the_policy_leaves_with_a_last_worker_that_serves_the_model_among_others() {
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    // Its registration builds qwen's policy; it serves llama besides.
    let both = endpoint("http://w2:8000", &["qwen", "llama"]);
    register(&app_context, Arc::clone(&both), "random").await;

    remove(&app_context, llama).await;
    assert_eq!(policy_of(&app_context, "llama"), Some("cache_aware"));
    remove(&app_context, both).await;
    assert_eq!(policy_of(&app_context, "llama"), None);
}

#[tokio::test]
async fn a_replacement_that_lists_a_model_fewer_takes_its_policy() {
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    register(
        &app_context,
        endpoint("http://w2:8000", &["qwen", "llama"]),
        "random",
    )
    .await;
    remove(&app_context, llama).await;

    // The endpoint registered again, its provider serving qwen alone: no
    // worker of this node serves llama any more.
    register(
        &app_context,
        endpoint("http://w2:8000", &["qwen"]),
        "random",
    )
    .await;
    assert_eq!(policy_of(&app_context, "llama"), None);
    assert_eq!(policy_of(&app_context, "qwen"), Some("random"));
}

#[tokio::test]
async fn a_removal_takes_the_policy_of_the_worker_a_replacement_put_in_its_place() {
    // The removal found llama's worker, then a registration replaced it at
    // its URL during the drain: the registry step removes the replacement,
    // and qwen's policy goes with qwen's last worker.
    let app_context = app_context();
    let llama = worker("http://w1:8000", "llama");
    register(&app_context, Arc::clone(&llama), "cache_aware").await;
    register(&app_context, worker("http://w1:8000", "qwen"), "random").await;

    remove(&app_context, llama).await;
    assert!(app_context
        .worker_registry
        .get_by_url("http://w1:8000")
        .is_none());
    assert_eq!(policy_of(&app_context, "qwen"), None);
}

#[tokio::test]
async fn a_mesh_import_removed_here_leaves_the_local_workers_policy() {
    // The import never added to the model's workers here, so its removal
    // must not end a policy a local worker still routes by.
    let app_context = app_context();
    register(
        &app_context,
        worker("http://w1:8000", "llama"),
        "cache_aware",
    )
    .await;
    app_context
        .worker_registry
        .on_remote_worker_state(&smg_mesh::WorkerState {
            worker_id: "peer-w1".to_string(),
            model_id: "llama".to_string(),
            url: "http://remote:8080".to_string(),
            health: true,
            load: 0.0,
            version: 1,
            spec: vec![],
            owner: "peer".to_string(),
        });
    let import = app_context
        .worker_registry
        .get_by_url("http://remote:8080")
        .expect("the import");

    remove(&app_context, import).await;
    assert_eq!(policy_of(&app_context, "llama"), Some("cache_aware"));
}
