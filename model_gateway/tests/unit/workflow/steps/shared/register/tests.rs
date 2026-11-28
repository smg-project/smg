use openai_protocol::model_card::ModelCard;
use smg_worker::jobs::WorkerJobAsker;

use super::*;
use crate::routers::grpc::backend_client::BasicWorkerBuilder;

fn worker_at(url: &str, model_id: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model_id))
            .build(),
    )
}

fn worker(model_id: &str) -> Arc<dyn Worker> {
    worker_at("http://worker:8080", model_id)
}

/// The registry names what a registration left in it: its own workers
/// while they are the entries under their URLs, by id and revision, and
/// nothing once a worker under the URL is another one.
#[test]
fn the_registry_names_what_a_registration_left_in_it() {
    use openai_protocol::worker::WorkerSpec;
    use wfaas::{WorkflowId, WorkflowInstanceId};

    use crate::{
        app_context::testing::app_context_for_tests, config::RouterConfig,
        routers::common::snapshot::WorkerRegistry, workflow::steps::create_worker_workflow_data,
    };
    let url = "http://worker:8080";
    let registry = Arc::new(WorkerRegistry::new());
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new(url).build());
    let id = registry
        .register(Arc::clone(&worker))
        .expect("the worker registers");
    let mut data = create_worker_workflow_data(
        WorkerSpec::new(url),
        WorkerRegistrationMode::CreateOnly,
        app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None),
        WorkerJobAsker::Api,
    );
    data.actual_workers = Some(vec![Arc::clone(&worker)]);
    let state = WorkflowState::new(
        WorkflowInstanceId::new(),
        WorkflowId::new("worker_registration"),
        data,
    );
    assert_eq!(
        registered_entries(&registry, &state),
        HashMap::from([(id.as_str().to_string(), worker.revision())]),
        "the registration's worker is the entry under its URL"
    );
    registry.remove(&id);
    let other: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new(url).build());
    registry.register(other).expect("another worker registers");
    assert!(
        registered_entries(&registry, &state).is_empty(),
        "the worker under the URL is not this registration's"
    );
}

/// A terminal state says whether its registry step landed: the step's
/// record reads `Succeeded` once the workers are in the registry, whatever
/// the workflow's own status became after it; no record, or a step that
/// did not run through, means nothing landed.
#[test]
fn a_terminal_state_says_whether_its_registry_step_landed() {
    use openai_protocol::worker::WorkerSpec;
    use wfaas::{StepState, WorkflowId, WorkflowInstanceId, WorkflowStatus};

    use crate::{
        app_context::testing::app_context_for_tests, config::RouterConfig,
        routers::common::snapshot::WorkerRegistry, workflow::steps::create_worker_workflow_data,
    };
    let data = create_worker_workflow_data(
        WorkerSpec::new("http://worker:8080"),
        WorkerRegistrationMode::CreateOnly,
        app_context_for_tests(
            RouterConfig::default(),
            Arc::new(WorkerRegistry::new()),
            None,
        ),
        WorkerJobAsker::Api,
    );
    let mut state = WorkflowState::new(
        WorkflowInstanceId::new(),
        WorkflowId::new("worker_registration"),
        data,
    );
    state.status = WorkflowStatus::Cancelled;
    assert!(
        !registered_in(&state),
        "no record of the step, nothing landed"
    );
    let step = StepId::new(REGISTER_WORKERS_STEP);
    for status in [
        StepStatus::Pending,
        StepStatus::Running,
        StepStatus::Failed,
        StepStatus::Skipped,
    ] {
        state.step_states.insert(
            step.clone(),
            StepState {
                status,
                ..StepState::default()
            },
        );
        assert!(
            !registered_in(&state),
            "a {status:?} registry step landed nothing"
        );
    }
    state.step_states.insert(
        step,
        StepState {
            status: StepStatus::Succeeded,
            ..StepState::default()
        },
    );
    assert!(
        registered_in(&state),
        "a cancelled workflow whose registry step succeeded left its workers"
    );
}

fn replace_by_id(worker_id: &WorkerId, expected_revision: u64) -> WorkerRegistrationMode {
    WorkerRegistrationMode::ReplaceById {
        worker_id: worker_id.as_str().to_string(),
        expected_revision,
    }
}

#[test]
fn create_only_rejects_duplicate_url_and_keeps_existing_worker() {
    let registry = WorkerRegistry::new();
    let original_id = registry.register(worker("original-model")).unwrap();

    let result = register_worker(
        &registry,
        worker("replacement-model"),
        &WorkerRegistrationMode::CreateOnly,
    );

    assert!(matches!(result, Err(WorkflowError::StepFailed { .. })));
    assert_eq!(
        registry.get_id_by_url("http://worker:8080"),
        Some(original_id)
    );
    assert_eq!(registry.get_by_model("original-model").len(), 1);
    assert!(registry.get_by_model("replacement-model").is_empty());
}

#[test]
fn upsert_replaces_the_worker_at_the_same_url() {
    let registry = WorkerRegistry::new();
    let original_id = registry.register(worker("original-model")).unwrap();

    let new_id = register_worker(
        &registry,
        worker("replacement-model"),
        &WorkerRegistrationMode::Upsert,
    )
    .unwrap();

    // Replacing at the same URL keeps the worker ID stable, so callers
    // holding the ID from the original registration stay valid.
    assert_eq!(new_id, original_id);
    assert_eq!(registry.get_id_by_url("http://worker:8080"), Some(new_id));
    assert_eq!(registry.get_by_model("replacement-model").len(), 1);
    assert!(registry.get_by_model("original-model").is_empty());
}

#[test]
fn create_only_rejects_the_batch_before_registering_any_worker() {
    let registry = WorkerRegistry::new();
    registry
        .register(worker_at("http://rank1:8080", "taken"))
        .unwrap();

    // A data-parallel spec expands to several workers. Rank 0 is free and
    // rank 1 is taken, so the whole batch must be refused untouched.
    let batch = vec![
        worker_at("http://rank0:8080", "new-model"),
        worker_at("http://rank1:8080", "new-model"),
    ];
    let result = check_batch(&registry, &batch, &WorkerRegistrationMode::CreateOnly);

    assert!(matches!(result, Err(WorkflowError::StepFailed { .. })));
    assert!(registry.get_by_url("http://rank0:8080").is_none());
    assert_eq!(registry.get_by_model("taken").len(), 1);
    assert!(registry.get_by_model("new-model").is_empty());
}

#[test]
fn replace_by_id_refuses_a_multi_worker_batch() {
    let registry = WorkerRegistry::new();
    let original_id = registry.register(worker("original-model")).unwrap();

    let batch = vec![
        worker_at("http://worker:8080", "new-model"),
        worker_at("http://worker:8081", "new-model"),
    ];
    let result = check_batch(&registry, &batch, &replace_by_id(&original_id, 0));

    assert!(matches!(result, Err(WorkflowError::StepFailed { .. })));
}

#[test]
fn replace_by_id_applies_the_new_spec() {
    let registry = WorkerRegistry::new();
    let original_id = registry.register(worker("original-model")).unwrap();

    let new_id = register_worker(
        &registry,
        worker("replacement-model"),
        &replace_by_id(&original_id, 0),
    )
    .unwrap();

    assert_eq!(new_id, original_id);
    assert_eq!(registry.get_by_model("replacement-model").len(), 1);
    assert!(registry.get_by_model("original-model").is_empty());
}

#[test]
fn replace_by_id_does_not_restore_an_older_spec_over_a_newer_one() {
    let registry = WorkerRegistry::new();
    let original_id = registry.register(worker("original-model")).unwrap();

    // Two PUTs are accepted while the worker sits at revision 0, and the
    // newer one finishes first. Replacing bumps the revision.
    register_worker(
        &registry,
        worker("newer-model"),
        &replace_by_id(&original_id, 0),
    )
    .unwrap();

    let result = register_worker(
        &registry,
        worker("older-model"),
        &replace_by_id(&original_id, 0),
    );

    assert!(matches!(result, Err(WorkflowError::StepFailed { .. })));
    assert_eq!(registry.get_by_model("newer-model").len(), 1);
    assert!(registry.get_by_model("older-model").is_empty());
}

#[test]
fn create_only_rejects_a_batch_that_repeats_a_url() {
    let registry = WorkerRegistry::new();

    let batch = vec![
        worker_at("http://rank0:8080", "new-model"),
        worker_at("http://rank0:8080", "new-model"),
    ];
    let result = check_batch(&registry, &batch, &WorkerRegistrationMode::CreateOnly);

    assert!(matches!(result, Err(WorkflowError::StepFailed { .. })));
    assert!(registry.get_by_url("http://rank0:8080").is_none());
}

#[test]
fn replace_by_id_does_not_resurrect_a_removed_worker() {
    let registry = WorkerRegistry::new();
    let original_id = registry.register(worker("original-model")).unwrap();
    // Stands in for a DELETE that lands while the replacement workflow runs.
    assert!(registry.remove(&original_id).is_some());

    let result = register_worker(
        &registry,
        worker("replacement-model"),
        &replace_by_id(&original_id, 0),
    );

    assert!(matches!(result, Err(WorkflowError::StepFailed { .. })));
    assert!(registry.get_by_url("http://worker:8080").is_none());
}

#[test]
fn replace_by_id_does_not_overwrite_a_recreated_worker() {
    let registry = WorkerRegistry::new();
    let original_id = registry.register(worker("original-model")).unwrap();
    // Stands in for DELETE + POST on the same URL while the replacement
    // workflow runs: same URL, different worker, new ID.
    assert!(registry.remove(&original_id).is_some());
    let recreated_id = registry.register(worker("recreated-model")).unwrap();
    assert_ne!(recreated_id, original_id);

    let result = register_worker(
        &registry,
        worker("replacement-model"),
        &replace_by_id(&original_id, 0),
    );

    assert!(matches!(result, Err(WorkflowError::StepFailed { .. })));
    assert_eq!(
        registry.get_id_by_url("http://worker:8080"),
        Some(recreated_id)
    );
    assert_eq!(registry.get_by_model("recreated-model").len(), 1);
    assert!(registry.get_by_model("replacement-model").is_empty());
}

/// A worker of `model_id` at `url` whose spec overrides the retry count,
/// resolved against the gateway's retry settings as a registration
/// resolves it.
fn worker_overriding_retries(
    url: &str,
    model_id: &str,
    gateway: &smg_worker::settings::RetryConfig,
    max_retries: u32,
) -> Arc<dyn Worker> {
    use openai_protocol::worker::{ResilienceUpdate, WorkerSpec};
    use smg_worker::health::{
        circuit_breaker::CircuitBreakerConfig, resilience::resolve_resilience,
    };

    let overrides = ResilienceUpdate {
        max_retries: Some(max_retries),
        ..Default::default()
    };
    let (resolved, _breaker) = resolve_resilience(
        gateway,
        &CircuitBreakerConfig::default(),
        true,
        true,
        &overrides,
    );
    let mut spec = WorkerSpec::new(url);
    spec.resilience = overrides;
    Arc::new(
        BasicWorkerBuilder::from_spec(spec)
            .model(ModelCard::new(model_id))
            .resilience(resolved)
            .build(),
    )
}

/// A model reads the gateway's own retry settings again once the worker
/// that carried a retry override is replaced by a spec without one and
/// no other worker of the model carries one.
#[tokio::test]
async fn a_model_reads_the_gateways_retries_once_no_worker_overrides_them() {
    use crate::{app_context::testing::app_context_for_tests, config::RouterConfig};

    let url = "http://worker:8080";
    let registry = Arc::new(WorkerRegistry::new());
    // A gateway whose retry settings are not the default ones: the model
    // goes back to these, not to a default.
    let mut router_config = RouterConfig::default();
    router_config.retry.max_retries += 2;
    let app_context = app_context_for_tests(router_config, Arc::clone(&registry), None);
    let gateway = app_context.router_config.effective_retry_config();
    let overridden = gateway.max_retries + 3;
    let first = worker_overriding_retries(url, "model", &gateway, overridden);
    write_batch(
        &app_context,
        &[Arc::clone(&first)],
        &WorkerRegistrationMode::CreateOnly,
        None,
    )
    .expect("the worker with the override registers");
    assert_eq!(
        registry
            .get_retry_config("model")
            .map(|retry| retry.max_retries),
        Some(overridden),
        "the model reads its worker's override"
    );

    let id = registry.get_id_by_url(url).expect("the worker has an id");
    write_batch(
        &app_context,
        &[worker_at(url, "model")],
        &replace_by_id(&id, first.revision()),
        None,
    )
    .expect("a spec without an override replaces it");
    assert_eq!(
        registry
            .get_retry_config("model")
            .unwrap_or_else(|| gateway.clone()),
        gateway,
        "the override went with the spec that carried it"
    );
}

/// The override stays while another worker of the model still carries
/// one.
#[tokio::test]
async fn a_model_keeps_the_retries_another_of_its_workers_still_overrides() {
    use crate::{app_context::testing::app_context_for_tests, config::RouterConfig};

    let url = "http://worker-a:8080";
    let registry = Arc::new(WorkerRegistry::new());
    let app_context = app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None);
    let gateway = app_context.router_config.effective_retry_config();
    let overridden = gateway.max_retries + 1;
    let first = worker_overriding_retries(url, "model", &gateway, gateway.max_retries + 3);
    let second = worker_overriding_retries("http://worker-b:8080", "model", &gateway, overridden);
    for worker in [Arc::clone(&first), second] {
        write_batch(
            &app_context,
            &[worker],
            &WorkerRegistrationMode::CreateOnly,
            None,
        )
        .expect("a worker with the override registers");
    }

    let id = registry.get_id_by_url(url).expect("the worker has an id");
    write_batch(
        &app_context,
        &[worker_at(url, "model")],
        &replace_by_id(&id, first.revision()),
        None,
    )
    .expect("a spec without an override replaces the first worker");
    assert_eq!(
        registry
            .get_retry_config("model")
            .map(|retry| retry.max_retries),
        Some(overridden),
        "the second worker's override still stands"
    );
}

/// A model whose overriding worker was removed while another of its
/// workers stayed reads the gateway's own retry settings again at the
/// removal.
#[tokio::test]
async fn a_model_reads_the_gateways_retries_once_its_overriding_worker_left() {
    use crate::{app_context::testing::app_context_for_tests, config::RouterConfig};

    let registry = Arc::new(WorkerRegistry::new());
    let app_context = app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None);
    let gateway = app_context.router_config.effective_retry_config();
    let overriding = worker_overriding_retries(
        "http://worker-a:8080",
        "model",
        &gateway,
        gateway.max_retries + 3,
    );
    for worker in [overriding, worker_at("http://worker-b:8080", "model")] {
        write_batch(
            &app_context,
            &[worker],
            &WorkerRegistrationMode::CreateOnly,
            None,
        )
        .expect("a worker of the model registers");
    }
    let id = registry
        .get_id_by_url("http://worker-a:8080")
        .expect("the overriding worker has an id");
    assert!(registry.remove(&id).is_some());
    assert_eq!(
        registry
            .get_retry_config("model")
            .unwrap_or_else(|| gateway.clone()),
        gateway,
        "no worker of the model carries the override any more"
    );
}
