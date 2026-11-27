//! Unified worker registration step.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use smg_worker::{
    health::bootstrap_probe,
    identity::basic::Worker,
    registry::{worker_model_ids, WorkerId},
};
use tracing::debug;
use wfaas::{
    StepExecutor, StepId, StepResult, StepStatus, WorkflowContext, WorkflowData, WorkflowError,
    WorkflowResult, WorkflowState,
};

use crate::{
    app_context::AppContext,
    routers::{
        common::{pd_admission, snapshot::WorkerRegistry},
        grpc::regular::symphony::telemetry,
    },
    workflow::data::{require_app_context, WorkerRegistrationData, WorkerRegistrationMode},
};

/// Unified step to register workers in the registry.
///
/// Works with both single workers and batches. Always expects `workers` key
/// in context containing `Vec<Arc<dyn Worker>>`.
/// Works with any workflow data type that implements `WorkerRegistrationData`.
pub struct RegisterWorkersStep;

/// The step's id in the registration DAGs.
pub(crate) const REGISTER_WORKERS_STEP: &str = "register_workers";

/// Whether the registry step of `state` completed: the registration's workers
/// are in the registry, whatever the workflow did after it (a cancellation or
/// a later step's failure leaves them there).
pub fn registered_in<D: WorkflowData>(state: &WorkflowState<D>) -> bool {
    state
        .step_states
        .get(&StepId::new(REGISTER_WORKERS_STEP))
        .is_some_and(|step| step.status == StepStatus::Succeeded)
}

/// The registration's workers that are the registry's entries under their
/// URLs now, by id and revision: what a removal queued for this registration
/// may take. A worker registered or replaced since is not among them, and a
/// registration whose registry step the state does not record as succeeded
/// (a cancellation written while the step ran) is still read from the
/// registry itself.
pub fn registered_entries<D: WorkerRegistrationData>(
    registry: &WorkerRegistry,
    state: &WorkflowState<D>,
) -> HashMap<String, u64> {
    state
        .context
        .data
        .get_actual_workers()
        .into_iter()
        .flatten()
        .filter_map(|worker| {
            let id = registry.get_id_by_url(worker.url())?;
            let entry = registry.get(&id)?;
            Arc::ptr_eq(&entry, worker).then(|| (id.as_str().to_string(), entry.revision()))
        })
        .collect()
}

fn step_failed(message: String) -> WorkflowError {
    WorkflowError::StepFailed {
        step_id: StepId::new(REGISTER_WORKERS_STEP),
        message,
    }
}

/// Reject a batch before any of it is written.
///
/// A data-parallel spec expands to one worker per rank, so a mode that can
/// reject a single worker can also leave earlier ranks registered. Checking
/// the whole batch first keeps that outcome out of the registry.
fn check_batch(
    registry: &WorkerRegistry,
    workers: &[Arc<dyn Worker>],
    registration_mode: &WorkerRegistrationMode,
) -> WorkflowResult<()> {
    match registration_mode {
        WorkerRegistrationMode::CreateOnly => {
            let mut batch_urls = HashSet::with_capacity(workers.len());
            for worker in workers {
                if registry.get_by_url(worker.url()).is_some() {
                    return Err(step_failed(format!(
                        "Worker {} already exists",
                        worker.redacted_url()
                    )));
                }
                // The registry is keyed by URL, so a batch that repeats one
                // would register the first and reject the rest.
                if !batch_urls.insert(worker.url()) {
                    return Err(step_failed(format!(
                        "Worker {} appears twice in the same registration",
                        worker.redacted_url()
                    )));
                }
            }
            Ok(())
        }
        // One ID replaces one worker. `PUT` is refused up front for a rank
        // worker. A plain worker whose engine reports several ranks by now
        // is the one other spec that turns into several workers: the
        // creation step refuses its `PUT` first, since the ranks would
        // stand beside the plain worker, and this check stands behind it,
        // before any of them is written.
        WorkerRegistrationMode::ReplaceById { worker_id, .. } if workers.len() != 1 => {
            Err(step_failed(format!(
                "Replacing worker {worker_id} needs exactly one worker, got {}",
                workers.len()
            )))
        }
        _ => Ok(()),
    }
}

fn register_worker(
    registry: &WorkerRegistry,
    worker: Arc<dyn Worker>,
    registration_mode: &WorkerRegistrationMode,
) -> WorkflowResult<WorkerId> {
    match registration_mode {
        WorkerRegistrationMode::CreateOnly => registry
            .register(worker.clone())
            .ok_or_else(|| step_failed(format!("Worker {} already exists", worker.redacted_url()))),
        WorkerRegistrationMode::Upsert => Ok(registry.register_or_replace(worker)),
        WorkerRegistrationMode::ReplaceById {
            worker_id,
            expected_revision,
        } => {
            let worker_id = WorkerId::from_string(worker_id.clone());
            let address = worker.redacted_url().to_string();
            // Fails when the ID is gone or the worker moved on, which covers
            // the concurrent DELETE, DELETE + POST, and second-PUT cases, and
            // when a removal is draining the worker.
            if registry.replace_claiming_local_unless_draining(
                &worker_id,
                worker,
                *expected_revision,
            ) {
                Ok(worker_id)
            } else {
                Err(step_failed(format!(
                    "Worker {} at {address} is no longer at revision {expected_revision} or \
                     is being removed, so the replacement was dropped",
                    worker_id.as_str()
                )))
            }
        }
    }
}

/// The models a registration of `workers` may move workers off: every model
/// whose policy a worker it replaces at the same URL keeps
/// ([`WorkerRegistry::policy_model_ids`]). Whether a worker of this node, the
/// replacement included, still serves each one is asked once the
/// registration is written.
fn models_left(registry: &WorkerRegistry, workers: &[Arc<dyn Worker>]) -> HashSet<String> {
    workers
        .iter()
        .filter_map(|worker| registry.get_by_url(worker.url()))
        .flat_map(|replaced| WorkerRegistry::policy_model_ids(&replaced))
        .collect()
}

/// Write a batch to the registry, whole or not at all, and do what follows
/// the write ([`after_write`]).
fn write_batch(
    app_context: &AppContext,
    workers: &[Arc<dyn Worker>],
    registration_mode: &WorkerRegistrationMode,
    symphony_table: Option<&'static str>,
) -> WorkflowResult<()> {
    let registry = &app_context.worker_registry;
    check_batch(registry, workers, registration_mode)?;

    let mut worker_ids = Vec::with_capacity(workers.len());
    for worker in workers {
        let worker_id = register_worker(registry, Arc::clone(worker), registration_mode)?;
        debug!(
            "Registered worker {} (model: {}) with ID {:?}",
            worker.redacted_url(),
            worker.model_id(),
            worker_id
        );
        // A decode worker's running window is the PD admission bound; one
        // that reports none leaves the gate off for it, which is said here
        // rather than inferred later from absent series.
        pd_admission::announce_decode_window(worker.as_ref());
        worker_ids.push(worker_id);
    }

    after_write(app_context, workers, symphony_table);

    // WorkerMonitor subscribes to registry events directly (see
    // `smg_worker::load::monitor::WorkerMonitor::start_event_loop`), so this
    // step no longer has to push group-add notifications. The
    // monitor's event loop reconciles the impacted groups as soon as the
    // registry emits `WorkerEvent::Registered` or `WorkerEvent::Replaced`.

    // Note: worker_ids are stored for potential future use but not persisted
    // as they are internal registry identifiers
    debug!(
        "Registered {} workers with IDs: {:?}",
        worker_ids.len(),
        worker_ids
    );
    Ok(())
}

/// What a registration does beside the registry's write, once every worker
/// of the batch stands in the registry: the parser series of
/// `symphony_table`, the port probe and the retry settings of each model.
/// The registry's own write moved the pool sizes.
fn after_write(
    app_context: &AppContext,
    workers: &[Arc<dyn Worker>],
    symphony_table: Option<&'static str>,
) {
    // Under Symphony the failure classes of the table that reads these
    // workers' model exist at zero from here, once every worker of the
    // batch is in the registry: a registration refused before leaves no
    // series behind. The table was resolved where the model card and the
    // names are (none for an HTTP worker, the legacy parsers or a model
    // no table reads).
    if let Some(table) = symphony_table {
        telemetry::init_table_series(table);
    }

    // The rendezvous port a disaggregated worker advertises is probed once
    // the worker is in the registry (DP ranks share the port), off the
    // registration path: a registration refused before, or a workflow
    // that failed before this step, is never reported as a registered
    // worker with a dead port.
    if let Some(worker) = workers.first() {
        bootstrap_probe::spawn_bootstrap_port_probe(worker.as_ref());
    }

    // Update per-model retry config (last write wins).
    // Only update if the worker has non-empty retry overrides in its spec.
    for worker in workers {
        let resilience_spec = &worker.metadata().spec.resilience;
        let has_retry_overrides = resilience_spec.max_retries.is_some()
            || resilience_spec.initial_backoff_ms.is_some()
            || resilience_spec.max_backoff_ms.is_some()
            || resilience_spec.backoff_multiplier.is_some()
            || resilience_spec.jitter_factor.is_some()
            || resilience_spec.disable_retry.is_some();

        if has_retry_overrides {
            let resolved = worker.resilience();
            let retry_config = resolved.retry.clone();
            for model_id in worker_model_ids(worker) {
                app_context.worker_registry.set_model_retry_config(
                    &model_id,
                    retry_config.clone(),
                    resolved.retry_enabled,
                );
            }
        }
    }
}

/// A test's hold on the registry step, between the engine's cancel read for
/// the step and the step's registry write: the window a `DELETE`'s cancel
/// lands in while the step runs, whose worker the job's end then reads from
/// the registry. Placed by URL for its next registration, taken once.
#[cfg(test)]
pub(crate) mod registration_hold {
    use std::sync::{Arc, LazyLock};

    use dashmap::DashMap;
    use tokio::sync::Notify;

    /// The hold on one URL's registry step: `reached` fires when the step is
    /// in its window, `release` lets it write.
    pub(crate) struct RegistrationHold {
        pub(crate) reached: Notify,
        pub(crate) release: Notify,
    }

    static HOLDS: LazyLock<DashMap<String, Arc<RegistrationHold>>> = LazyLock::new(DashMap::new);

    /// Places a hold on the next registry step for `url`.
    pub(crate) fn place(url: &str) -> Arc<RegistrationHold> {
        let hold = Arc::new(RegistrationHold {
            reached: Notify::new(),
            release: Notify::new(),
        });
        HOLDS.insert(url.to_string(), Arc::clone(&hold));
        hold
    }

    /// The step's side: under a hold placed for `url`, says the window is
    /// reached and waits for the release; without one, returns at once.
    pub(crate) async fn honour(url: &str) {
        let Some((_, hold)) = HOLDS.remove(url) else {
            return;
        };
        hold.reached.notify_one();
        hold.release.notified().await;
    }
}

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for RegisterWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let app_context = require_app_context(context.data.get_app_context())?.clone();

        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        // A test's hold on this step's window (`registration_hold`).
        #[cfg(test)]
        {
            if let Some(worker) = workers.first() {
                registration_hold::honour(worker.url()).await;
            }
        }

        let registration_mode = context.data.registration_mode();

        // Read before the registry is written: only the registry as it is
        // now knows the workers this registration replaces.
        let left_models = models_left(&app_context.worker_registry, workers);

        write_batch(
            &app_context,
            workers,
            &registration_mode,
            context.data.symphony_table(),
        )?;

        // A replacement that moved a worker to another model, or that lists
        // a model fewer, takes the policy of each model no worker of this
        // node serves any more, as the worker's removal would.
        for model_id in &left_models {
            if !app_context.worker_registry.serves_model_locally(model_id) {
                app_context.policy_registry.on_model_unserved(model_id);
            }
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
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
            routers::common::snapshot::WorkerRegistry,
            workflow::steps::create_worker_workflow_data,
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
            routers::common::snapshot::WorkerRegistry,
            workflow::steps::create_worker_workflow_data,
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
}
