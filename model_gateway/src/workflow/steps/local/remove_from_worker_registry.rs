//! Step to remove workers from worker registry.

use async_trait::async_trait;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    worker::endpoint::Endpoint,
    workflow::{
        data::WorkerRemovalWorkflowData,
        steps::shared::{pool_group, refresh_worker_pool_sizes, PoolGroups},
    },
};

/// Step to remove workers from the worker registry.
///
/// A pinned removal (`WorkerRemovalRequest::expected_revisions`) removes the
/// workers it selected, by id; an unguarded one removes whatever stands
/// under each URL.
pub struct RemoveFromWorkerRegistryStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for RemoveFromWorkerRegistryStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let worker_urls = &context.data.worker_urls;

        debug!(
            "Removing {} worker(s) from worker registry",
            worker_urls.len()
        );

        // Snapshot the unique `(worker_type, connection_mode, model_id)`
        // groups for the workers we are about to remove. We capture this
        // before the removal so the pool-size metric update below can
        // recompute the per-group count after the workers have been
        // pulled from the registry. The cache eviction the old
        // `LoadMonitor` path used to do here is now handled by
        // `WorkerMonitor` reacting to `WorkerEvent::Removed`.
        let mut unique_configs = PoolGroups::new();
        for url in worker_urls {
            if let Some(w) = app_context.worker_registry.get_by_url(url) {
                unique_configs.insert(pool_group(w.as_ref()));
            }
        }

        // A pinned removal takes the worker it selected, by id: a worker
        // registered under the URL since the selection is not in its map and
        // stays, while a worker rebuilt under its id during the drain
        // inherits the drained runtime and goes with it. An unguarded removal
        // takes whatever stands under the URL. Either way the worker removed
        // can differ from the worker found; the policy step asks after the
        // models of both.
        let registry = &app_context.worker_registry;
        let pins = context.data.config.expected_revisions.as_ref();
        let removed: Vec<_> = worker_urls
            .iter()
            .filter_map(|worker_url| match pins {
                None => registry.remove_by_url(worker_url),
                Some(pins) => {
                    let worker_id = registry.get_id_by_url(worker_url)?;
                    if pins.contains_key(worker_id.as_str()) {
                        registry.remove(&worker_id)
                    } else {
                        debug!(
                            "Worker {} was registered under the URL after the removal's selection, left alone",
                            Endpoint::redacted_str(worker_url)
                        );
                        None
                    }
                }
            })
            .collect();
        let removed_count = removed.len();

        // Log if some workers were already removed (e.g., by another process)
        if removed_count == worker_urls.len() {
            debug!("Removed {} worker(s) from registry", removed_count);
        } else {
            warn!(
                "Removed {} of {} workers (some may have been removed by another process)",
                removed_count,
                worker_urls.len()
            );
        }

        // Update Layer 3 worker pool size metrics for unique
        // configurations. WorkerMonitor subscribes to registry events
        // directly and reconciles per-group polling state on its own,
        // so this step no longer needs to push group-removed
        // notifications.
        refresh_worker_pool_sizes(&app_context.worker_registry, &unique_configs);
        context.data.removed_workers = removed;

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::{testing::app_context_for_tests, AppContext},
        config::RouterConfig,
        worker::{BasicWorkerBuilder, Worker, WorkerRegistry, WorkerType},
        workflow::{data::WorkerList, steps::local::WorkerRemovalRequest},
    };

    fn build_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("served"))
                .build(),
        )
    }

    fn make_app_context(registry: Arc<WorkerRegistry>) -> Arc<AppContext> {
        let router_config = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .build_unchecked();
        app_context_for_tests(router_config, registry, None)
    }

    /// The data the selection leaves for the step: the workers found under
    /// `url`, and the request's pins, if any.
    fn make_context(
        app_context: Arc<AppContext>,
        url: &str,
        workers: Vec<Arc<dyn Worker>>,
        expected_revisions: Option<HashMap<String, Option<u64>>>,
    ) -> WorkflowContext<WorkerRemovalWorkflowData> {
        let data = WorkerRemovalWorkflowData {
            config: WorkerRemovalRequest {
                url: url.to_string(),
                expected_revisions,
            },
            workers_to_remove: Some(WorkerList::from_workers(&workers)),
            worker_urls: workers.iter().map(|w| w.url().to_string()).collect(),
            affected_models: workers.iter().map(|w| w.model_id().to_string()).collect(),
            app_context: Some(app_context),
            actual_workers_to_remove: Some(workers),
            removed_workers: Vec::new(),
        };
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    /// A pinned removal takes the worker it selected and nothing else: the
    /// worker found left the registry after the selection and a registration
    /// landed under the URL since, which the step leaves standing. Removed by
    /// URL, the step took the new worker for the old one's removal.
    #[tokio::test]
    async fn a_pinned_removal_leaves_a_worker_registered_under_the_url_since_its_selection() {
        let url = "http://w1:8080";
        let registry = Arc::new(WorkerRegistry::new());
        let found = build_worker(url);
        let found_id = registry
            .register(Arc::clone(&found))
            .expect("the worker registers");
        let pins = HashMap::from([(found_id.as_str().to_string(), Some(found.revision()))]);
        let mut ctx = make_context(
            make_app_context(Arc::clone(&registry)),
            url,
            vec![Arc::clone(&found)],
            Some(pins),
        );
        assert!(
            registry.remove(&found_id).is_some(),
            "the worker found leaves"
        );
        let fresh = build_worker(url);
        let fresh_id = registry
            .register(Arc::clone(&fresh))
            .expect("the fresh registration lands");
        assert_ne!(
            fresh_id.as_str(),
            found_id.as_str(),
            "the registration has a fresh id"
        );

        let result = RemoveFromWorkerRegistryStep
            .execute(&mut ctx)
            .await
            .unwrap();

        assert_eq!(result, StepResult::Success);
        let standing = registry
            .get_by_url(url)
            .expect("the fresh worker stays under the URL");
        assert!(
            Arc::ptr_eq(&standing, &fresh),
            "the worker under the URL is the fresh one"
        );
        assert!(
            ctx.data.removed_workers.is_empty(),
            "the step removed nothing"
        );
    }

    /// A pinned removal removes the worker it selected by id: a worker rebuilt
    /// under that id after the selection (a `PUT`, a label refresh) inherits
    /// the drained worker's runtime, `Draining` included, and goes with the
    /// removal, whether the pin named a revision or none.
    #[tokio::test]
    async fn a_pinned_removal_removes_the_worker_rebuilt_under_its_id() {
        for pin_at_revision in [true, false] {
            let url = "http://w1:8080";
            let registry = Arc::new(WorkerRegistry::new());
            let found = build_worker(url);
            let found_id = registry
                .register(Arc::clone(&found))
                .expect("the worker registers");
            let pinned = found.revision();
            let pin = pin_at_revision.then_some(pinned);
            let mut ctx = make_context(
                make_app_context(Arc::clone(&registry)),
                url,
                vec![Arc::clone(&found)],
                Some(HashMap::from([(found_id.as_str().to_string(), pin)])),
            );
            // The drain wrote `Draining`; the rebuild under the id inherits it.
            found.set_status(WorkerStatus::Draining);
            let rebuilt = build_worker(url);
            assert!(
                registry.replace(&found_id, Arc::clone(&rebuilt)),
                "the rebuild lands"
            );
            assert_eq!(
                rebuilt.status(),
                WorkerStatus::Draining,
                "the rebuild inherits the drained runtime"
            );
            assert_ne!(rebuilt.revision(), pinned, "the rebuild moved the revision");

            let result = RemoveFromWorkerRegistryStep
                .execute(&mut ctx)
                .await
                .unwrap();

            assert_eq!(result, StepResult::Success);
            assert!(
                registry.get_by_url(url).is_none(),
                "the rebuild under the pinned id is removed (pinned at a revision: {pin_at_revision})"
            );
            assert_eq!(ctx.data.removed_workers.len(), 1, "one worker removed");
            assert!(
                Arc::ptr_eq(&ctx.data.removed_workers[0], &rebuilt),
                "the worker removed is the rebuild"
            );
        }
    }

    /// An unguarded removal (no pins) takes whatever stands under the URL, as
    /// before.
    #[tokio::test]
    async fn an_unguarded_removal_takes_the_worker_under_the_url() {
        let url = "http://w1:8080";
        let registry = Arc::new(WorkerRegistry::new());
        let found = build_worker(url);
        let found_id = registry
            .register(Arc::clone(&found))
            .expect("the worker registers");
        let mut ctx = make_context(
            make_app_context(Arc::clone(&registry)),
            url,
            vec![Arc::clone(&found)],
            None,
        );
        assert!(
            registry.remove(&found_id).is_some(),
            "the worker found leaves"
        );
        let fresh = build_worker(url);
        registry
            .register(Arc::clone(&fresh))
            .expect("the fresh registration lands");

        let result = RemoveFromWorkerRegistryStep
            .execute(&mut ctx)
            .await
            .unwrap();

        assert_eq!(result, StepResult::Success);
        assert!(
            registry.get_by_url(url).is_none(),
            "the unguarded removal takes the worker under the URL"
        );
        assert_eq!(ctx.data.removed_workers.len(), 1, "one worker removed");
        assert!(
            Arc::ptr_eq(&ctx.data.removed_workers[0], &fresh),
            "the worker removed is the one under the URL"
        );
    }
}
