//! Step to remove workers from worker registry.

use std::sync::Arc;

use async_trait::async_trait;
use openai_protocol::worker::WorkerStatus;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    worker::{endpoint::Endpoint, Worker},
    workflow::{
        data::WorkerRemovalWorkflowData,
        steps::shared::{pool_group, refresh_worker_pool_sizes, PoolGroups},
    },
};

/// Step to remove workers from the worker registry.
///
/// A pinned removal (`WorkerRemovalRequest::expected_revisions`) removes the
/// incarnation it selected: an id pinned at a revision while the worker is
/// at that revision or still reads `Draining` (a removal's own mark, this
/// job's or an earlier one's that never completed, which a rebuild under the
/// id inherits and nothing but an activation reverses),
/// an id pinned at no revision (a `DELETE`'s) as it stands. A replacement
/// that stands on its own (activated to `Ready`, or never drained) and a
/// registration under the URL since the selection are left standing and
/// leave the job, so the policy step touches nothing of them. An unguarded
/// removal removes whatever stands under each URL.
pub struct RemoveFromWorkerRegistryStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for RemoveFromWorkerRegistryStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context =
            Arc::clone(
                context.data.app_context.as_ref().ok_or_else(|| {
                    WorkflowError::ContextValueNotFound("app_context".to_string())
                })?,
            );
        let registry = &app_context.worker_registry;
        let worker_urls = context.data.worker_urls.clone();
        let pins = context.data.config.expected_revisions.clone();

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
        for url in &worker_urls {
            if let Some(w) = registry.get_by_url(url) {
                unique_configs.insert(pool_group(w.as_ref()));
            }
        }

        // The worker removed can differ from the worker found (a rebuild
        // under its id during the drain); the policy step asks after the
        // models of both. A URL whose worker is left standing leaves the job
        // below, so that step touches nothing of it.
        let mut removed: Vec<Arc<dyn Worker>> = Vec::with_capacity(worker_urls.len());
        let mut standing: Vec<&str> = Vec::new();
        for worker_url in &worker_urls {
            let outcome = match &pins {
                // An unguarded removal takes whatever stands under the URL.
                None => registry.remove_by_url(worker_url),
                Some(pins) => {
                    let Some(worker_id) = registry.get_id_by_url(worker_url) else {
                        continue;
                    };
                    match pins.get(worker_id.as_str()) {
                        // A registration under the URL since the selection.
                        None => {
                            debug!(
                                "Worker {} was registered under the URL after the removal's selection, left alone",
                                Endpoint::redacted_str(worker_url)
                            );
                            standing.push(worker_url);
                            None
                        }
                        // The `DELETE`'s pin: the id as it stands.
                        Some(None) => registry.remove(&worker_id),
                        // The incarnation selected: at its pinned revision, or
                        // a rebuild of it that still reads `Draining`. That
                        // state is a removal's own mark, this job's drain's or
                        // an earlier removal's that never completed; a rebuild
                        // inherits it and only an activation reverses it, so a
                        // worker still reading it is the drained worker
                        // continued, and this removal's to take. A worker the
                        // drain never marked (it was not `Ready`) that was
                        // replaced since, and a replacement activated since,
                        // read neither the revision nor the mark: they stand
                        // on their own and keep their place.
                        Some(Some(pinned)) => {
                            let pinned = *pinned;
                            let removed = registry.remove_if(&worker_id, |worker| {
                                worker.revision() == pinned
                                    || worker.status() == WorkerStatus::Draining
                            });
                            if removed.is_none() && registry.get(&worker_id).is_some() {
                                debug!(
                                    worker_url = %Endpoint::redacted_str(worker_url),
                                    pinned_revision = pinned,
                                    "Worker was replaced after the removal's selection and stands on its own, left alone"
                                );
                                standing.push(worker_url);
                            }
                            removed
                        }
                    }
                }
            };
            // A pinned removal that took nothing: a worker standing under the
            // URL now (the pinned id gone and a registration under the URL
            // since, inside the window between the lookup above and the
            // removal) is not this job's, and the URL leaves the job with it.
            if outcome.is_none()
                && pins.is_some()
                && standing.last().copied() != Some(worker_url.as_str())
                && registry.get_by_url(worker_url).is_some()
            {
                standing.push(worker_url);
            }
            removed.extend(outcome);
        }
        let removed_count = removed.len();
        let expected = worker_urls.len() - standing.len();

        // Log if some workers were already removed (e.g., by another process)
        if removed_count == expected {
            debug!(
                "Removed {} worker(s) from registry, {} left standing",
                removed_count,
                standing.len()
            );
        } else {
            warn!(
                "Removed {} of {} workers (some may have been removed by another process), {} left standing",
                removed_count,
                expected,
                standing.len()
            );
        }

        // Update Layer 3 worker pool size metrics for unique
        // configurations. WorkerMonitor subscribes to registry events
        // directly and reconciles per-group polling state on its own,
        // so this step no longer needs to push group-removed
        // notifications.
        refresh_worker_pool_sizes(registry, &unique_configs);
        if !standing.is_empty() {
            let kept: Vec<Arc<dyn Worker>> = context
                .data
                .actual_workers_to_remove
                .iter()
                .flatten()
                .filter(|worker| !standing.contains(&worker.url()))
                .map(Arc::clone)
                .collect();
            context.data.keep_only(kept);
        }
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
        // The job carries nothing of the URL any more: the policy step after
        // this one leaves the fresh worker's state alone.
        assert!(
            ctx.data.worker_urls.is_empty(),
            "the fresh worker's URL is out of the job"
        );
        assert!(
            ctx.data
                .actual_workers_to_remove
                .as_ref()
                .is_some_and(Vec::is_empty),
            "the old snapshot is out of the job"
        );
        assert!(
            ctx.data.affected_models.is_empty(),
            "the fresh worker's model is out of the job"
        );
    }

    /// A rebuild under a revision-pinned id that still reads `Draining` is the
    /// drained worker continued (the rebuild inherited the drain's mark and
    /// nothing re-activated it): it goes with the removal, by id.
    #[tokio::test]
    async fn a_rebuild_that_still_drains_goes_with_a_revision_pinned_removal() {
        let url = "http://w1:8080";
        let registry = Arc::new(WorkerRegistry::new());
        let found = build_worker(url);
        let found_id = registry
            .register(Arc::clone(&found))
            .expect("the worker registers");
        let pinned = found.revision();
        let mut ctx = make_context(
            make_app_context(Arc::clone(&registry)),
            url,
            vec![Arc::clone(&found)],
            Some(HashMap::from([(
                found_id.as_str().to_string(),
                Some(pinned),
            )])),
        );
        // The drain wrote `Draining`; the rebuild under the id inherits it.
        found.set_status(WorkerStatus::Draining);
        let rebuilt = build_worker(url);
        assert!(
            registry.replace(&found_id, Arc::clone(&rebuilt)),
            "the rebuild lands"
        );
        assert_ne!(rebuilt.revision(), pinned, "the rebuild moved the revision");
        assert_eq!(
            rebuilt.status(),
            WorkerStatus::Draining,
            "the rebuild inherits the drained runtime"
        );

        let result = RemoveFromWorkerRegistryStep
            .execute(&mut ctx)
            .await
            .unwrap();

        assert_eq!(result, StepResult::Success);
        assert!(
            registry.get_by_url(url).is_none(),
            "the rebuild that still drains is removed"
        );
        assert_eq!(ctx.data.removed_workers.len(), 1, "one worker removed");
        assert!(
            Arc::ptr_eq(&ctx.data.removed_workers[0], &rebuilt),
            "the worker removed is the rebuild"
        );
        assert_eq!(
            ctx.data.worker_urls,
            vec![url.to_string()],
            "the job keeps the URL it removed"
        );
    }

    /// A replacement under a revision-pinned id that stands on its own is left
    /// standing and leaves the job: one a `PUT` activated during the drain
    /// wait (`Ready` over the inherited `Draining`), and one of a worker the
    /// drain never marked (it was not `Ready`, so nothing was written). Removed
    /// by id at any revision, the step took a successful replacement for the
    /// old worker's removal.
    #[tokio::test]
    async fn a_replacement_that_stands_on_its_own_survives_a_revision_pinned_removal() {
        for (drained, after) in [
            (WorkerStatus::Draining, WorkerStatus::Ready),
            (WorkerStatus::NotReady, WorkerStatus::NotReady),
        ] {
            let url = "http://w1:8080";
            let registry = Arc::new(WorkerRegistry::new());
            let found = build_worker(url);
            let found_id = registry
                .register(Arc::clone(&found))
                .expect("the worker registers");
            let pinned = found.revision();
            let mut ctx = make_context(
                make_app_context(Arc::clone(&registry)),
                url,
                vec![Arc::clone(&found)],
                Some(HashMap::from([(
                    found_id.as_str().to_string(),
                    Some(pinned),
                )])),
            );
            found.set_status(drained);
            let replacement = build_worker(url);
            assert!(
                registry.replace(&found_id, Arc::clone(&replacement)),
                "the replacement lands"
            );
            assert_ne!(
                replacement.revision(),
                pinned,
                "the replacement moved the revision"
            );
            // The replacement's own workflow activates it (a PUT's), or it
            // was never drained: either way it stands on its own.
            replacement.set_status(after);

            let result = RemoveFromWorkerRegistryStep
                .execute(&mut ctx)
                .await
                .unwrap();

            assert_eq!(result, StepResult::Success);
            let standing = registry.get_by_url(url).unwrap_or_else(|| {
                panic!("the replacement stays under the URL ({drained:?} -> {after:?})")
            });
            assert!(
                Arc::ptr_eq(&standing, &replacement),
                "the worker under the URL is the replacement"
            );
            assert!(
                ctx.data.removed_workers.is_empty(),
                "the step removed nothing ({drained:?} -> {after:?})"
            );
            assert!(
                ctx.data.worker_urls.is_empty()
                    && ctx
                        .data
                        .actual_workers_to_remove
                        .as_ref()
                        .is_some_and(Vec::is_empty)
                    && ctx.data.affected_models.is_empty(),
                "the replacement is out of the job ({drained:?} -> {after:?})"
            );
        }
    }

    /// An id pinned at no revision (a `DELETE`'s pin) is removed as it
    /// stands: a rebuild under the id goes whether it still drains or its
    /// own workflow activated it, since the `DELETE` named the id.
    #[tokio::test]
    async fn a_no_revision_pin_removes_the_rebuild_under_its_id_as_it_stands() {
        for after in [WorkerStatus::Draining, WorkerStatus::Ready] {
            let url = "http://w1:8080";
            let registry = Arc::new(WorkerRegistry::new());
            let found = build_worker(url);
            let found_id = registry
                .register(Arc::clone(&found))
                .expect("the worker registers");
            let pinned = found.revision();
            let mut ctx = make_context(
                make_app_context(Arc::clone(&registry)),
                url,
                vec![Arc::clone(&found)],
                Some(HashMap::from([(found_id.as_str().to_string(), None)])),
            );
            found.set_status(WorkerStatus::Draining);
            let rebuilt = build_worker(url);
            assert!(
                registry.replace(&found_id, Arc::clone(&rebuilt)),
                "the rebuild lands"
            );
            assert_ne!(rebuilt.revision(), pinned, "the rebuild moved the revision");
            rebuilt.set_status(after);

            let result = RemoveFromWorkerRegistryStep
                .execute(&mut ctx)
                .await
                .unwrap();

            assert_eq!(result, StepResult::Success);
            assert!(
                registry.get_by_url(url).is_none(),
                "the rebuild under the id the DELETE named is removed ({after:?})"
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
