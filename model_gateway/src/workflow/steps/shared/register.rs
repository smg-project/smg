//! Unified worker registration step.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use tracing::debug;
use wfaas::{
    StepExecutor, StepId, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::{
    observability::metrics::Metrics,
    routers::common::pd_admission,
    worker::{
        registry::WorkerId,
        worker::{ConnectionModeExt, WorkerTypeExt},
        Worker, WorkerRegistry,
    },
    workflow::data::{WorkerRegistrationData, WorkerRegistrationMode},
};

/// Unified step to register workers in the registry.
///
/// Works with both single workers and batches. Always expects `workers` key
/// in context containing `Vec<Arc<dyn Worker>>`.
/// Works with any workflow data type that implements `WorkerRegistrationData`.
pub struct RegisterWorkersStep;

fn step_failed(message: String) -> WorkflowError {
    WorkflowError::StepFailed {
        step_id: StepId::new("register_workers"),
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
                        worker.url()
                    )));
                }
                // The registry is keyed by URL, so a batch that repeats one
                // would register the first and reject the rest.
                if !batch_urls.insert(worker.url()) {
                    return Err(step_failed(format!(
                        "Worker {} appears twice in the same registration",
                        worker.url()
                    )));
                }
            }
            Ok(())
        }
        // One ID replaces one worker. `PUT` is refused up front when the
        // router runs data-parallel, which is the only expansion that turns
        // one spec into several workers; the check keeps a future caller from
        // writing a partial batch.
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
            .ok_or_else(|| step_failed(format!("Worker {} already exists", worker.url()))),
        WorkerRegistrationMode::Upsert => Ok(registry.register_or_replace(worker)),
        WorkerRegistrationMode::ReplaceById {
            worker_id,
            expected_revision,
        } => {
            let worker_id = WorkerId::from_string(worker_id.clone());
            let url = worker.url().to_string();
            // Fails when the ID is gone or the worker moved on, which covers
            // the concurrent DELETE, DELETE + POST, and second-PUT cases.
            if registry.replace_claiming_local(&worker_id, worker, *expected_revision) {
                Ok(worker_id)
            } else {
                Err(step_failed(format!(
                    "Worker {} at {url} is no longer at revision {expected_revision}, \
                     so the replacement was dropped",
                    worker_id.as_str()
                )))
            }
        }
    }
}

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for RegisterWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .get_app_context()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();

        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        let mut worker_ids = Vec::with_capacity(workers.len());
        let registration_mode = context.data.registration_mode();

        check_batch(&app_context.worker_registry, workers, &registration_mode)?;

        for worker in workers {
            let worker_id = register_worker(
                &app_context.worker_registry,
                Arc::clone(worker),
                &registration_mode,
            )?;
            debug!(
                "Registered worker {} (model: {}) with ID {:?}",
                worker.url(),
                worker.model_id(),
                worker_id
            );
            // A decode worker's running window is the PD admission bound; one
            // that reports none leaves the gate off for it, which is said here
            // rather than inferred later from absent series.
            pd_admission::announce_decode_window(worker.as_ref());
            worker_ids.push(worker_id);
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
                for model_id in WorkerRegistry::worker_model_ids(worker) {
                    app_context.worker_registry.set_model_retry_config(
                        &model_id,
                        retry_config.clone(),
                        resolved.retry_enabled,
                    );
                }
            }
        }

        // Collect unique worker configurations to avoid redundant metric updates
        let unique_configs: HashSet<_> = workers
            .iter()
            .map(|w| {
                let meta = w.metadata();
                (
                    meta.spec.worker_type,
                    meta.spec.connection_mode,
                    w.model_id().to_string(),
                )
            })
            .collect();

        // Update Layer 3 worker pool size metrics per unique
        // `(worker_type, connection_mode, model_id)`.
        for (worker_type, connection_mode, model_id) in &unique_configs {
            // Get labels before moving values into get_workers_filtered
            let worker_type_label = worker_type.as_metric_label();
            let connection_mode_label = connection_mode.as_metric_label();

            let pool_size = app_context
                .worker_registry
                .get_workers_filtered(
                    Some(model_id),
                    Some(*worker_type),
                    Some(*connection_mode),
                    None,
                    false,
                )
                .len();

            Metrics::set_worker_pool_size(
                worker_type_label,
                connection_mode_label,
                model_id,
                pool_size,
            );
        }

        // WorkerMonitor subscribes to registry events directly (see
        // `worker::monitor::WorkerMonitor::start_event_loop`), so this
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

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::model_card::ModelCard;

    use super::*;
    use crate::worker::BasicWorkerBuilder;

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
