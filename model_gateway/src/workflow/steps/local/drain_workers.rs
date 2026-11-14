//! Step to drain workers before removal.
//!
//! Marks each Ready worker as `Draining` so policies stop selecting it,
//! then sleeps the maximum `drain_settle_secs` across the workers being
//! removed. Non-Ready workers (Pending, NotReady, Failed) are not
//! drained — sleeping for them would just delay the cleanup of broken
//! workers without protecting any in-flight traffic.
//!
//! A pinned removal (`WorkerRemovalRequest::expected_revisions`) reads its
//! pins again here: a worker whose pin moved since the selection is a
//! different incarnation and leaves the job, so the steps after this one
//! touch nothing of it.

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use openai_protocol::worker::WorkerStatus;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    worker::{endpoint::Endpoint, Worker},
    workflow::data::{WorkerList, WorkerRemovalWorkflowData},
};

/// Step that transitions Ready workers to `Draining` and waits for the
/// resolved drain settle window before downstream steps remove them.
pub struct DrainWorkersStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for DrainWorkersStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let workers_to_remove = context
            .data
            .actual_workers_to_remove
            .clone()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_remove".to_string()))?;

        if workers_to_remove.is_empty() {
            return Ok(StepResult::Success);
        }

        let registry = &app_context.worker_registry;
        let pins = context.data.config.expected_revisions.as_ref();
        let mut max_drain_secs: u64 = 0;
        let mut transitioned = 0usize;
        // The workers that stay the job's. A worker whose pin moved since the
        // selection (replaced under its id, or a registration under its URL)
        // is left out, and the job's lists are rewritten without it below.
        let mut kept: Vec<Arc<dyn Worker>> = Vec::with_capacity(workers_to_remove.len());

        for snapshot in &workers_to_remove {
            // Resolve the worker against the live registry: a worker that
            // became `Ready` after `find_workers_to_remove` ran would
            // otherwise bypass the drain and start serving traffic up until
            // the remove step runs.
            let url = snapshot.url();
            let Some(worker_id) = registry.get_id_by_url(url) else {
                debug!(
                    "Worker {} not in registry, skipping drain transition",
                    Endpoint::redacted_str(url)
                );
                kept.push(Arc::clone(snapshot));
                continue;
            };
            // The pin the selection passed for this id: absent from the map,
            // the id is a registration under the URL since the selection;
            // pinned at no revision (a `DELETE`'s), the id as it stands;
            // pinned at a revision, that build of it.
            let pinned_revision = match pins.map(|pins| pins.get(worker_id.as_str()).copied()) {
                Some(None) => {
                    debug!(
                        "Worker {} was registered under the URL after the removal's selection, left alone",
                        Endpoint::redacted_str(url)
                    );
                    continue;
                }
                Some(Some(pinned)) => pinned,
                None => None,
            };
            let revision = match pinned_revision {
                Some(revision) => revision,
                None => {
                    let Some(current) = registry.get(&worker_id) else {
                        debug!(
                            "Worker {} disappeared from registry, skipping drain transition",
                            Endpoint::redacted_str(url)
                        );
                        kept.push(Arc::clone(snapshot));
                        continue;
                    };
                    current.revision()
                }
            };
            // The Ready check is the closure's: it runs under the worker's
            // mutation lock, in one step with the write, so a probe's
            // demotion that lands after a check made outside the lock is
            // not written over with `Draining`. The drain window is read
            // there too, from the worker the write lands on.
            let drained = registry.apply_if_revision(&worker_id, revision, |worker| {
                let ready = worker.status() == WorkerStatus::Ready;
                (
                    worker.metadata().health_config.drain_settle_secs,
                    ready.then_some(WorkerStatus::Draining),
                )
            });
            match drained {
                Some((drain_secs, transition)) => {
                    if transition.is_some() {
                        transitioned += 1;
                        max_drain_secs = max_drain_secs.max(drain_secs);
                    }
                    kept.push(Arc::clone(snapshot));
                }
                // A revision pin that no longer holds: the worker under the
                // id is a replacement, which the job leaves alone (the next
                // pass re-evaluates it). A worker gone altogether stays the
                // job's, as a worker gone under any other pin does: the
                // registry step finds nothing and the policy step prunes it.
                None => match (pinned_revision, registry.get(&worker_id)) {
                    (Some(pinned), Some(current)) => debug!(
                        worker_url = %Endpoint::redacted_str(url),
                        pinned_revision = pinned,
                        current_revision = current.revision(),
                        "Worker was replaced after the removal's selection, left alone"
                    ),
                    _ => {
                        debug!(
                            "Worker {} disappeared from registry, skipping drain transition",
                            Endpoint::redacted_str(url)
                        );
                        kept.push(Arc::clone(snapshot));
                    }
                },
            }
        }

        if kept.len() != workers_to_remove.len() {
            context.data.worker_urls = kept.iter().map(|w| w.url().to_string()).collect();
            context.data.affected_models = kept.iter().map(|w| w.model_id().to_string()).collect();
            context.data.workers_to_remove = Some(WorkerList::from_workers(&kept));
            context.data.actual_workers_to_remove = Some(kept);
        }

        if transitioned == 0 {
            debug!("Drain step: no Ready workers in batch, skipping drain");
            return Ok(StepResult::Success);
        }

        info!(
            "Draining {} worker(s) for {}s before removal",
            transitioned, max_drain_secs
        );
        tokio::time::sleep(Duration::from_secs(max_drain_secs)).await;

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::AppContext,
        worker::{BasicWorkerBuilder, Worker, WorkerType},
        workflow::data::{WorkerList, WorkerRemovalWorkflowData},
    };

    fn build_worker(url: &str, drain_secs: u64, status: WorkerStatus) -> Arc<dyn Worker> {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    drain_settle_secs: drain_secs,
                    ..Default::default()
                })
                .build(),
        );
        worker.set_status(status);
        worker
    }

    fn make_app_context(workers: &[Arc<dyn Worker>]) -> Arc<AppContext> {
        use crate::{
            app_context::testing::app_context_for_tests, config::RouterConfig,
            worker::WorkerRegistry,
        };

        let router_config = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .build_unchecked();
        let registry = Arc::new(WorkerRegistry::new());
        for w in workers {
            registry.register(Arc::clone(w)).unwrap();
        }
        app_context_for_tests(router_config, registry, None)
    }

    fn make_context(
        app_context: Arc<AppContext>,
        workers: Vec<Arc<dyn Worker>>,
    ) -> WorkflowContext<WorkerRemovalWorkflowData> {
        let worker_urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();
        let data = WorkerRemovalWorkflowData {
            config: super::super::find_workers_to_remove::WorkerRemovalRequest {
                url: worker_urls.first().cloned().unwrap_or_default(),
                expected_revisions: None,
            },
            workers_to_remove: Some(WorkerList::from_workers(&workers)),
            worker_urls,
            affected_models: std::collections::HashSet::new(),
            app_context: Some(app_context),
            actual_workers_to_remove: Some(workers),
            removed_workers: Vec::new(),
        };
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    #[tokio::test(start_paused = true)]
    async fn test_drain_workers_step_skips_when_workers_list_empty() {
        let app_ctx = make_app_context(&[]);
        let mut ctx = make_context(app_ctx, vec![]);
        let result = DrainWorkersStep.execute(&mut ctx).await.unwrap();
        assert_eq!(result, StepResult::Success);
    }

    #[tokio::test(start_paused = true)]
    async fn test_drain_workers_step_skips_when_no_ready_workers() {
        // Failed worker: should not transition or sleep.
        let worker = build_worker("http://w1:8080", 30, WorkerStatus::Failed);
        let app_ctx = make_app_context(&[Arc::clone(&worker)]);
        let mut ctx = make_context(Arc::clone(&app_ctx), vec![Arc::clone(&worker)]);

        let start = tokio::time::Instant::now();
        let result = DrainWorkersStep.execute(&mut ctx).await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result, StepResult::Success);
        assert!(
            elapsed < Duration::from_millis(1),
            "non-Ready workers should not trigger sleep, elapsed={elapsed:?}"
        );
        // Status unchanged: still Failed.
        assert_eq!(worker.status(), WorkerStatus::Failed);
    }

    #[tokio::test(start_paused = true)]
    async fn test_drain_workers_step_transitions_ready_to_draining() {
        let worker = build_worker("http://w1:8080", 5, WorkerStatus::Ready);
        let app_ctx = make_app_context(&[Arc::clone(&worker)]);
        let mut ctx = make_context(Arc::clone(&app_ctx), vec![Arc::clone(&worker)]);

        #[expect(
            clippy::disallowed_methods,
            reason = "test-only spawn awaited via the JoinHandle below"
        )]
        let step_handle = tokio::spawn(async move { DrainWorkersStep.execute(&mut ctx).await });

        // Yield so the step starts and reaches the sleep.
        tokio::task::yield_now().await;
        // Transition is immediate.
        assert_eq!(worker.status(), WorkerStatus::Draining);

        // Step should still be sleeping.
        assert!(!step_handle.is_finished(), "step should still be in sleep");

        tokio::time::advance(Duration::from_secs(5)).await;
        let result = step_handle.await.unwrap().unwrap();
        assert_eq!(result, StepResult::Success);
    }

    #[tokio::test(start_paused = true)]
    async fn test_drain_workers_step_uses_max_drain_settle() {
        let w1 = build_worker("http://w1:8080", 3, WorkerStatus::Ready);
        let w2 = build_worker("http://w2:8080", 7, WorkerStatus::Ready);
        let app_ctx = make_app_context(&[Arc::clone(&w1), Arc::clone(&w2)]);
        let mut ctx = make_context(Arc::clone(&app_ctx), vec![Arc::clone(&w1), Arc::clone(&w2)]);

        #[expect(
            clippy::disallowed_methods,
            reason = "test-only spawn awaited via the JoinHandle below"
        )]
        let step_handle = tokio::spawn(async move { DrainWorkersStep.execute(&mut ctx).await });

        tokio::task::yield_now().await;
        // Both transitioned.
        assert_eq!(w1.status(), WorkerStatus::Draining);
        assert_eq!(w2.status(), WorkerStatus::Draining);

        // After 3s: still sleeping (max=7s).
        tokio::time::advance(Duration::from_secs(3)).await;
        tokio::task::yield_now().await;
        assert!(!step_handle.is_finished(), "step should still be in sleep");

        // After total 7s: done.
        tokio::time::advance(Duration::from_secs(4)).await;
        let result = step_handle.await.unwrap().unwrap();
        assert_eq!(result, StepResult::Success);
    }

    /// Regression: status/revision must come from the live registry, not
    /// the snapshot taken in `find_workers_to_remove`. A worker that was
    /// non-Ready at snapshot time but became Ready before this step runs
    /// would otherwise bypass the drain and start serving traffic right
    /// up until removal.
    #[tokio::test(start_paused = true)]
    async fn test_drain_workers_step_uses_live_registry_status_not_snapshot() {
        let worker = build_worker("http://w-live:8080", 5, WorkerStatus::Pending);
        let app_ctx = make_app_context(&[Arc::clone(&worker)]);
        // Snapshot was taken while Pending — the step must NOT trust this.
        let snapshot = vec![Arc::clone(&worker)];
        // Live state flips to Ready before the step runs (e.g. a probe
        // succeeded between find_workers_to_remove and drain_workers).
        worker.set_status(WorkerStatus::Ready);

        let mut ctx = make_context(Arc::clone(&app_ctx), snapshot);

        #[expect(
            clippy::disallowed_methods,
            reason = "test-only spawn awaited via the JoinHandle below"
        )]
        let step_handle = tokio::spawn(async move { DrainWorkersStep.execute(&mut ctx).await });

        tokio::task::yield_now().await;
        // Live status Ready → step did transition us to Draining.
        assert_eq!(worker.status(), WorkerStatus::Draining);

        tokio::time::advance(Duration::from_secs(5)).await;
        let result = step_handle.await.unwrap().unwrap();
        assert_eq!(result, StepResult::Success);
    }

    #[tokio::test(start_paused = true)]
    async fn test_drain_workers_step_drain_secs_zero_skips_sleep() {
        // drain_settle_secs == 0 means "no drain window" — even Ready
        // workers transition to Draining (so policies stop selecting
        // them) but the step does not sleep.
        let worker = build_worker("http://w1:8080", 0, WorkerStatus::Ready);
        let app_ctx = make_app_context(&[Arc::clone(&worker)]);
        let mut ctx = make_context(Arc::clone(&app_ctx), vec![Arc::clone(&worker)]);

        let start = tokio::time::Instant::now();
        let result = DrainWorkersStep.execute(&mut ctx).await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result, StepResult::Success);
        assert!(
            elapsed < Duration::from_millis(1),
            "drain_secs=0 should not sleep, elapsed={elapsed:?}"
        );
        // Worker was still transitioned to Draining (so the policies stop
        // selecting it during the brief window before remove_from_*).
        assert_eq!(worker.status(), WorkerStatus::Draining);
    }

    /// The step's Ready check and its write are one step under the worker's
    /// mutation lock, so a probe that demotes the worker after the step
    /// looked at it and before it writes is not written over: the worker
    /// stays NotReady, nothing is drained and no `Draining` is published.
    /// Checked outside the lock, the step wrote `NotReady -> Draining` here.
    #[tokio::test]
    async fn test_drain_workers_step_keeps_a_demotion_written_between_its_read_and_its_write() {
        use crate::worker::event::WorkerEvent;

        let worker = build_worker("http://w1:8080", 0, WorkerStatus::Ready);
        let app_ctx = make_app_context(&[Arc::clone(&worker)]);
        let registry = &app_ctx.worker_registry;
        let worker_id = registry
            .get_id_by_url("http://w1:8080")
            .expect("registered");
        let revision = worker.revision();
        let mut events = registry.subscribe_events();
        let mut ctx = make_context(Arc::clone(&app_ctx), vec![Arc::clone(&worker)]);

        // The step runs on its own thread and runtime so that, paused at the
        // seam, it blocks neither this test nor the write placed in between.
        let result = registry.with_another_write_before_the_lock(
            |_| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("a runtime for the step")
                    .block_on(DrainWorkersStep.execute(&mut ctx))
            },
            |registry| {
                let ((), transition) = registry
                    .apply_if_revision(&worker_id, revision, |_| ((), Some(WorkerStatus::NotReady)))
                    .expect("the worker is there and current");
                assert_eq!(
                    transition,
                    Some((WorkerStatus::Ready, WorkerStatus::NotReady))
                );
            },
        );

        assert_eq!(result.expect("the step succeeds"), StepResult::Success);
        assert_eq!(
            worker.status(),
            WorkerStatus::NotReady,
            "the demotion written in between stands"
        );
        match events.try_recv() {
            Ok(WorkerEvent::StatusChanged {
                old_status,
                new_status,
                ..
            }) => assert_eq!(
                (old_status, new_status),
                (WorkerStatus::Ready, WorkerStatus::NotReady),
                "the demotion's event"
            ),
            other => panic!("expected the demotion's StatusChanged, got {other:?}"),
        }
        assert!(
            events.try_recv().is_err(),
            "the step published nothing: no Draining"
        );
    }

    /// The context of a pinned removal: the request's pins beside the workers
    /// the selection found.
    fn make_pinned_context(
        app_context: Arc<AppContext>,
        workers: Vec<Arc<dyn Worker>>,
        expected_revisions: HashMap<String, Option<u64>>,
    ) -> WorkflowContext<WorkerRemovalWorkflowData> {
        let mut ctx = make_context(app_context, workers);
        ctx.data.config.expected_revisions = Some(expected_revisions);
        ctx
    }

    /// The pin the selection passed is read again at the drain: a worker
    /// replaced under its id since (its revision moved) is a different
    /// incarnation, which the step neither drains nor keeps in the job, so
    /// the steps after it leave the replacement standing. Before, the step
    /// resolved the live worker and drained it at its live revision.
    #[tokio::test(start_paused = true)]
    async fn a_worker_replaced_after_the_selection_is_left_out_of_a_pinned_removal() {
        let url = "http://w1:8080";
        let found = build_worker(url, 5, WorkerStatus::Ready);
        let app_ctx = make_app_context(&[Arc::clone(&found)]);
        let id = app_ctx
            .worker_registry
            .get_id_by_url(url)
            .expect("registered");
        let pinned = found.revision();
        let mut ctx = make_pinned_context(
            Arc::clone(&app_ctx),
            vec![Arc::clone(&found)],
            HashMap::from([(id.as_str().to_string(), Some(pinned))]),
        );
        // The replacement lands between the selection and the drain.
        let replacement = build_worker(url, 5, WorkerStatus::Ready);
        assert!(
            app_ctx
                .worker_registry
                .replace(&id, Arc::clone(&replacement)),
            "the replacement lands"
        );
        assert_ne!(
            replacement.revision(),
            pinned,
            "the replacement moved the revision"
        );

        let start = tokio::time::Instant::now();
        let result = DrainWorkersStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success);
        assert!(
            start.elapsed() < Duration::from_millis(1),
            "nothing of the job drains, so the step does not sleep"
        );
        assert_eq!(
            replacement.status(),
            WorkerStatus::Ready,
            "the replacement is not drained"
        );
        assert!(
            ctx.data.worker_urls.is_empty(),
            "the replacement is out of the job's URLs"
        );
        assert!(
            ctx.data
                .actual_workers_to_remove
                .as_ref()
                .is_some_and(Vec::is_empty),
            "the replacement is out of the job's workers"
        );
    }

    /// A replacement written between the drain's read of its pin and its
    /// write lands first under the worker's lock (the registry's seam hook,
    /// as the demotion test uses it): the pin no longer holds, the step
    /// writes nothing and leaves the replacement out of the job.
    #[tokio::test]
    async fn a_replacement_written_between_the_drains_read_and_its_write_is_left_out() {
        let url = "http://w1:8080";
        let found = build_worker(url, 0, WorkerStatus::Ready);
        let app_ctx = make_app_context(&[Arc::clone(&found)]);
        let registry = &app_ctx.worker_registry;
        let id = registry.get_id_by_url(url).expect("registered");
        let pinned = found.revision();
        let replacement = build_worker(url, 0, WorkerStatus::Ready);
        let mut ctx = make_pinned_context(
            Arc::clone(&app_ctx),
            vec![Arc::clone(&found)],
            HashMap::from([(id.as_str().to_string(), Some(pinned))]),
        );

        let result = registry.with_another_write_before_the_lock(
            |_| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("a runtime for the step")
                    .block_on(DrainWorkersStep.execute(&mut ctx))
            },
            |registry| {
                assert!(
                    registry.replace(&id, Arc::clone(&replacement)),
                    "the replacement lands between the read and the write"
                );
            },
        );

        assert_eq!(result.expect("the step succeeds"), StepResult::Success);
        assert_eq!(
            replacement.status(),
            WorkerStatus::Ready,
            "the replacement written in between is not drained"
        );
        assert!(
            ctx.data.worker_urls.is_empty(),
            "the replacement is out of the job"
        );
    }

    /// A registration under the URL since the selection (the worker found
    /// left the registry in between) is not in a pinned removal's map: the
    /// step leaves it alone and out of the job.
    #[tokio::test(start_paused = true)]
    async fn a_registration_under_the_url_since_the_selection_is_left_out_of_a_pinned_removal() {
        let url = "http://w1:8080";
        let found = build_worker(url, 5, WorkerStatus::Ready);
        let app_ctx = make_app_context(&[Arc::clone(&found)]);
        let registry = &app_ctx.worker_registry;
        let id = registry.get_id_by_url(url).expect("registered");
        let mut ctx = make_pinned_context(
            Arc::clone(&app_ctx),
            vec![Arc::clone(&found)],
            HashMap::from([(id.as_str().to_string(), Some(found.revision()))]),
        );
        assert!(registry.remove(&id).is_some(), "the worker found leaves");
        let fresh = build_worker(url, 5, WorkerStatus::Ready);
        let fresh_id = registry
            .register(Arc::clone(&fresh))
            .expect("the fresh registration lands");
        assert_ne!(
            fresh_id.as_str(),
            id.as_str(),
            "the registration has a fresh id"
        );

        let result = DrainWorkersStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success);
        assert_eq!(
            fresh.status(),
            WorkerStatus::Ready,
            "the fresh worker is not drained"
        );
        assert!(
            ctx.data.worker_urls.is_empty(),
            "the fresh worker is out of the job"
        );
    }

    /// An id pinned at no revision (a `DELETE`'s pin) is drained as it
    /// stands: a worker rebuilt under the id since the selection is the one
    /// the `DELETE` named, so it is drained and stays in the job.
    #[tokio::test(start_paused = true)]
    async fn a_worker_pinned_at_no_revision_is_drained_as_it_stands() {
        let url = "http://w1:8080";
        let found = build_worker(url, 5, WorkerStatus::Ready);
        let app_ctx = make_app_context(&[Arc::clone(&found)]);
        let id = app_ctx
            .worker_registry
            .get_id_by_url(url)
            .expect("registered");
        let mut ctx = make_pinned_context(
            Arc::clone(&app_ctx),
            vec![Arc::clone(&found)],
            HashMap::from([(id.as_str().to_string(), None)]),
        );
        let replacement = build_worker(url, 5, WorkerStatus::Ready);
        assert!(
            app_ctx
                .worker_registry
                .replace(&id, Arc::clone(&replacement)),
            "the replacement lands"
        );

        #[expect(
            clippy::disallowed_methods,
            reason = "test-only spawn awaited via the JoinHandle below"
        )]
        let step_handle = tokio::spawn(async move {
            let result = DrainWorkersStep.execute(&mut ctx).await;
            (result, ctx)
        });

        tokio::task::yield_now().await;
        assert_eq!(
            replacement.status(),
            WorkerStatus::Draining,
            "the rebuild under the pinned id drains"
        );
        assert!(!step_handle.is_finished(), "step should still be in sleep");

        tokio::time::advance(Duration::from_secs(5)).await;
        let (result, ctx) = step_handle.await.unwrap();
        assert_eq!(result.unwrap(), StepResult::Success);
        assert_eq!(
            ctx.data.worker_urls,
            vec![url.to_string()],
            "the worker stays in the job"
        );
    }
}
