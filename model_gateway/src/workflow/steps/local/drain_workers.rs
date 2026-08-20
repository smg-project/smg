//! Step to drain workers before removal.
//!
//! Marks each Ready worker as `Draining` so policies stop selecting it,
//! then sleeps the maximum `drain_settle_secs` across the workers being
//! removed. Non-Ready workers (Pending, NotReady, Failed) are not
//! drained — sleeping for them would just delay the cleanup of broken
//! workers without protecting any in-flight traffic.

use std::time::Duration;

use async_trait::async_trait;
use openai_protocol::worker::WorkerStatus;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::workflow::data::WorkerRemovalWorkflowData;

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
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_remove".to_string()))?;

        if workers_to_remove.is_empty() {
            return Ok(StepResult::Success);
        }

        let mut max_drain_secs: u64 = 0;
        let mut transitioned = 0usize;

        for snapshot in workers_to_remove {
            // Resolve the worker against the live registry: a worker that
            // became `Ready` (or had its revision bumped via a same-URL
            // replace) after `find_workers_to_remove` ran would otherwise
            // bypass the drain and start serving traffic up until the
            // remove step runs.
            let url = snapshot.url();
            let Some(worker_id) = app_context.worker_registry.get_id_by_url(url) else {
                debug!("Worker {} not in registry, skipping drain transition", url);
                continue;
            };
            let Some(current) = app_context.worker_registry.get(&worker_id) else {
                debug!(
                    "Worker {} disappeared from registry, skipping drain transition",
                    url
                );
                continue;
            };
            if current.status() != WorkerStatus::Ready {
                continue;
            }
            let revision = current.revision();
            let drain_secs = current.metadata().health_config.drain_settle_secs;
            if app_context
                .worker_registry
                .transition_status_if_revision(&worker_id, revision, WorkerStatus::Draining)
                .is_some()
            {
                transitioned += 1;
                max_drain_secs = max_drain_secs.max(drain_secs);
            }
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
    use std::sync::Arc;

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
            config::RouterConfig,
            middleware::TokenBucket,
            observability::inflight_tracker::InFlightRequestTracker,
            routers::{
                common::{openai_bridge, realtime::RealtimeRegistry},
                grpc::multimodal::MultimodalConfigRegistry,
            },
            worker::{WorkerRegistry, WorkerService},
        };

        let router_config = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .build_unchecked();
        let registry = Arc::new(WorkerRegistry::new());
        for w in workers {
            registry.register(Arc::clone(w)).unwrap();
        }
        let job_queue = Arc::new(std::sync::OnceLock::new());

        Arc::new(AppContext {
            client: reqwest::Client::new(),
            router_config: router_config.clone(),
            rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
            rate_limit_manager: None,
            worker_registry: Arc::clone(&registry),
            policy_registry: Arc::new(crate::policies::PolicyRegistry::new(
                router_config.policy.clone(),
            )),
            reasoning_parser_factory: None,
            tool_parser_factory: None,
            router_manager: None,
            response_storage: Arc::new(smg_data_connector::MemoryResponseStorage::new()),
            conversation_storage: Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            conversation_item_storage: Arc::new(
                smg_data_connector::MemoryConversationItemStorage::new(),
            ),
            worker_monitor: None,
            configured_reasoning_parser: None,
            configured_tool_parser: None,
            worker_job_queue: Arc::clone(&job_queue),
            workflow_engines: Arc::new(std::sync::OnceLock::new()),
            mcp_orchestrator: Arc::new(std::sync::OnceLock::new()),
            mcp_format_registry: openai_bridge::FormatRegistry::new(),
            tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
            multimodal_config_registry: Arc::new(MultimodalConfigRegistry::new()),
            wasm_manager: None,
            worker_client_cache: Arc::new(crate::worker::WorkerHttpClientCache::new(
                &router_config,
            )),
            worker_service: Arc::new(WorkerService::new(registry, job_queue, router_config)),
            inflight_tracker: InFlightRequestTracker::new(),
            kv_event_monitor: None,
            realtime_registry: Arc::new(RealtimeRegistry::new()),
            webrtc_bind_addr: None,
            webrtc_stun_server: None,
        })
    }

    fn make_context(
        app_context: Arc<AppContext>,
        workers: Vec<Arc<dyn Worker>>,
    ) -> WorkflowContext<WorkerRemovalWorkflowData> {
        let worker_urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();
        let data = WorkerRemovalWorkflowData {
            config: super::super::find_workers_to_remove::WorkerRemovalRequest {
                url: worker_urls.first().cloned().unwrap_or_default(),
                dp_aware: false,
                expected_revision: None,
            },
            workers_to_remove: Some(WorkerList::from_workers(&workers)),
            worker_urls,
            affected_models: std::collections::HashSet::new(),
            app_context: Some(app_context),
            actual_workers_to_remove: Some(workers),
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
}
