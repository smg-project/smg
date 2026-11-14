//! Step to update policies for updated workers.

use std::collections::HashSet;

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    policies::as_cache_aware,
    workflow::{data::WorkerUpdateWorkflowData, steps::shared::start_kv_stream},
};

/// Step to update policies for updated workers.
///
/// After workers are updated, this step re-initializes cache-aware policies
/// for the affected models to reflect any priority or label changes.
pub struct UpdatePoliciesForWorkerStep;

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for UpdatePoliciesForWorkerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let updated_workers =
            context.data.updated_workers.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("updated_workers".to_string())
            })?;

        // Collect affected models
        let affected_models: HashSet<String> = updated_workers
            .iter()
            .map(|w| w.model_id().to_string())
            .collect();

        debug!(
            "Updating policies for {} affected model(s) after worker update",
            affected_models.len()
        );

        for model_id in &affected_models {
            let workers = app_context.worker_registry.get_by_model(model_id);

            if let Some(policy) = app_context.policy_registry.get_policy(model_id) {
                if as_cache_aware(policy.as_ref()).is_some() && !workers.is_empty() {
                    // Re-initialize cache-aware policy with updated workers
                    app_context
                        .policy_registry
                        .init_cache_aware_policy(model_id, &workers);

                    debug!(
                        "Updated cache-aware policy for model {} ({} workers)",
                        model_id,
                        workers.len()
                    );
                }
            }

            // Only a registration builds a model's policy: an update that
            // lands after the model's last worker was removed must not leave
            // the model one for its next registration to inherit.
            for worker in workers.iter() {
                worker.set_completion_sink(Some(app_context.policy_registry.completion_sink()));
            }
        }

        let prefill_workers = app_context.worker_registry.get_prefill_workers();
        let decode_workers = app_context.worker_registry.get_decode_workers();
        app_context
            .policy_registry
            .init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

        // An update that rebuilt a worker holds a new object under its URL
        // (a label refresh is the common case: an engine that came into
        // service with labels its registration did not have). The stream
        // started for the object it replaced must follow it: the load
        // snapshot fences the pushed loads of any other object out, and its
        // receipts suppress the URL's poll. The same object again changes
        // nothing on the monitor. Last in the step, after the policy work:
        // a restart waits for the old task to end. An object the registry no
        // longer holds (removed, or replaced again since the previous step
        // swapped it in, or while the restart waited) gets no stream: the
        // monitor reads the registry under its slot lock before it stops or
        // starts anything, since a stream for such an object would run until
        // the URL registers again.
        for worker in updated_workers {
            let needs_kv_events = app_context
                .policy_registry
                .worker_needs_kv_events(worker.as_ref());
            start_kv_stream(app_context, worker, needs_kv_events).await;
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::{
        HealthCheckConfig, RuntimeType, WorkerType, WorkerUpdateRequest,
    };
    use tracing_test::traced_test;
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        app_context::{testing::app_context_for_tests, AppContext},
        config::{PolicyConfig, RouterConfig},
        worker::{
            test_support::ready_worker, BasicWorkerBuilder, ConnectionMode, KvEventMonitor,
            ModelCard, SubscriptionKind, Worker, WorkerRegistry,
        },
        workflow::steps::local::create_worker_update_workflow_data,
    };

    const MODEL: &str = "llama-3";

    /// An app context whose policy for [`MODEL`] is cache-aware with an empty
    /// tree: the registry made the policy, nothing routed a worker through it.
    fn context_with_cache_aware_model() -> Arc<AppContext> {
        let config = RouterConfig::builder()
            .policy(PolicyConfig::RoundRobin)
            .build_unchecked();
        let app_context = app_context_for_tests(config, Arc::new(WorkerRegistry::new()), None);
        app_context
            .policy_registry
            .on_worker_added(MODEL, Some("cache_aware"));
        app_context
    }

    /// The worker URLs the model's cache-aware tree routes to.
    fn tree(app_context: &AppContext) -> Vec<String> {
        let policy = app_context
            .policy_registry
            .get_policy(MODEL)
            .expect("the model's policy");
        as_cache_aware(policy.as_ref())
            .expect("a cache-aware policy")
            .tree_tenants_for_test(MODEL)
    }

    /// The step routes the updated workers' model through its cache-aware
    /// tree: a tree that held nothing holds the model's workers after it.
    #[tokio::test]
    async fn the_step_routes_the_updated_workers_through_the_cache_aware_tree() {
        let app_context = context_with_cache_aware_model();
        let workers = vec![
            ready_worker("http://w1:8000", MODEL),
            ready_worker("http://w2:8000", MODEL),
        ];
        for worker in &workers {
            app_context
                .worker_registry
                .register(Arc::clone(worker))
                .unwrap();
        }
        assert!(
            tree(&app_context).is_empty(),
            "nothing routed before the step"
        );

        let update = WorkerUpdateRequest {
            priority: None,
            cost: None,
            labels: None,
            api_key: None,
            health: None,
        };
        let mut data = create_worker_update_workflow_data(
            "http://w1:8000".to_string(),
            update,
            Arc::clone(&app_context),
        );
        data.updated_workers = Some(workers);
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        UpdatePoliciesForWorkerStep
            .execute(&mut context)
            .await
            .unwrap();

        assert_eq!(tree(&app_context), ["http://w1:8000", "http://w2:8000"]);
    }

    /// A gRPC worker serving [`MODEL`] at a port that refuses connections at
    /// once: a stream it is given reconnects in the background with no
    /// engine involved.
    fn grpc_worker() -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:1")
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .model(ModelCard::new(MODEL))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    /// An app context under a cache-aware policy (its workers stream their
    /// KV events) with a KV event monitor and no worker monitor.
    fn context_under_cache_aware() -> (Arc<AppContext>, Arc<KvEventMonitor>) {
        context_under(PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096))
    }

    /// An app context under `policy` with a KV event monitor and no worker
    /// monitor: a worker whose policy reads no index streams nothing here.
    fn context_under(policy: PolicyConfig) -> (Arc<AppContext>, Arc<KvEventMonitor>) {
        let config = RouterConfig::builder().policy(policy).build_unchecked();
        let mut context = AppContext::clone(&app_context_for_tests(
            config,
            Arc::new(WorkerRegistry::new()),
            None,
        ));
        let monitor = Arc::new(KvEventMonitor::new());
        context
            .policy_registry
            .set_kv_event_monitor(Some(Arc::clone(&monitor)));
        context.kv_event_monitor = Some(Arc::clone(&monitor));
        // As the app wires the pair: the add reads the registry.
        monitor.set_worker_registry(&context.worker_registry);
        (Arc::new(context), monitor)
    }

    /// An update that rebuilt the worker (a label refresh, a `PATCH`) swapped
    /// a new object in under its URL. The stream its registration started
    /// must follow it: the step restarts the stream for the object the
    /// registry holds, and the old task releases the one it streamed for.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_rebuilt_worker_gets_the_stream_of_the_object_it_replaced() {
        let (app_context, monitor) = context_under_cache_aware();
        let first = grpc_worker();
        let id = app_context
            .worker_registry
            .register(Arc::clone(&first))
            .expect("the worker registers");
        // The registration's policy step started the stream for `first`.
        monitor
            .on_worker_added(&first, SubscriptionKind::Events)
            .await;

        let second = grpc_worker();
        assert!(
            app_context.worker_registry.replace_claiming_local(
                &id,
                Arc::clone(&second),
                first.revision()
            ),
            "the rebuilt worker replaces the one the update found"
        );
        let old = Arc::downgrade(&first);
        drop(first);

        let data = WorkerUpdateWorkflowData {
            config: WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: None,
                api_key: None,
                health: None,
            },
            worker_url: second.url().to_string(),
            app_context: Some(Arc::clone(&app_context)),
            workers_to_update: None,
            updated_workers: Some(vec![Arc::clone(&second)]),
            refresh_discovered_labels: true,
            fresh_engine_labels: None,
            snapshot: None,
        };
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        UpdatePoliciesForWorkerStep
            .execute(&mut context)
            .await
            .expect("the step succeeds");

        assert!(logs_contain(
            "Restarting KV event subscription for a re-registered worker"
        ));
        assert!(
            old.upgrade().is_none(),
            "the old stream's task has ended and released the object it streamed for"
        );
        monitor.stop().await;
    }

    /// The rebuilt worker streams nothing: its model's policy reads no index
    /// and no worker monitor wants its loads. The stream its registration
    /// started for the object it replaced, under the policy of that time,
    /// must not outlive the swap: the step ends it and starts none.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_rebuilt_worker_that_streams_nothing_ends_the_stream_of_the_object_it_replaced() {
        let (app_context, monitor) = context_under(PolicyConfig::Random);
        let first = grpc_worker();
        let id = app_context
            .worker_registry
            .register(Arc::clone(&first))
            .expect("the worker registers");
        monitor
            .on_worker_added(&first, SubscriptionKind::Events)
            .await;

        let second = grpc_worker();
        assert!(
            app_context.worker_registry.replace_claiming_local(
                &id,
                Arc::clone(&second),
                first.revision()
            ),
            "the rebuilt worker replaces the one the update found"
        );
        let old = Arc::downgrade(&first);
        drop(first);

        let data = WorkerUpdateWorkflowData {
            config: WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: None,
                api_key: None,
                health: None,
            },
            worker_url: second.url().to_string(),
            app_context: Some(Arc::clone(&app_context)),
            workers_to_update: None,
            updated_workers: Some(vec![Arc::clone(&second)]),
            refresh_discovered_labels: true,
            fresh_engine_labels: None,
            snapshot: None,
        };
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        UpdatePoliciesForWorkerStep
            .execute(&mut context)
            .await
            .expect("the step succeeds");

        assert!(logs_contain(
            "Ending the KV event subscription of a replaced worker"
        ));
        assert!(
            old.upgrade().is_none(),
            "the old stream's task has ended and released the object it streamed for"
        );
        logs_assert(|lines: &[&str]| {
            match lines
                .iter()
                .filter(|line| line.contains("Starting KV event subscription"))
                .count()
            {
                1 => Ok(()),
                n => Err(format!("{n} starts: the rebuilt worker streams nothing")),
            }
        });
        monitor.stop().await;
    }

    /// Between the swap and this step the worker left the registry (a
    /// removal workflow ran, or another registration replaced it again). The
    /// object in `updated_workers` is nobody's now: no stream starts for it,
    /// and the stream the URL had is left to whatever holds the URL.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_rebuilt_worker_the_registry_no_longer_holds_gets_no_stream() {
        let (app_context, monitor) = context_under_cache_aware();
        let first = grpc_worker();
        let id = app_context
            .worker_registry
            .register(Arc::clone(&first))
            .expect("the worker registers");
        monitor
            .on_worker_added(&first, SubscriptionKind::Events)
            .await;
        let second = grpc_worker();
        assert!(app_context.worker_registry.replace_claiming_local(
            &id,
            Arc::clone(&second),
            first.revision()
        ));
        assert!(
            app_context
                .worker_registry
                .remove_by_url(second.url())
                .is_some(),
            "the removal lands between the two steps"
        );

        let data = WorkerUpdateWorkflowData {
            config: WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: None,
                api_key: None,
                health: None,
            },
            worker_url: second.url().to_string(),
            app_context: Some(Arc::clone(&app_context)),
            workers_to_update: None,
            updated_workers: Some(vec![Arc::clone(&second)]),
            refresh_discovered_labels: true,
            fresh_engine_labels: None,
            snapshot: None,
        };
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        UpdatePoliciesForWorkerStep
            .execute(&mut context)
            .await
            .expect("the step succeeds");

        assert!(
            !logs_contain("Restarting KV event subscription"),
            "no restart for an object the registry no longer holds"
        );
        logs_assert(|lines: &[&str]| {
            match lines
                .iter()
                .filter(|line| line.contains("Starting KV event subscription"))
                .count()
            {
                1 => Ok(()),
                n => Err(format!("{n} starts; only the first worker's expected")),
            }
        });
        monitor.stop().await;
    }

    /// A removal that lands while the step's restart waits for the old
    /// task to end: the removal workflow takes the worker out of the
    /// registry and tells the monitor, which finds the URL's slot already in
    /// removal and does nothing. Once the old task is gone the add reads the
    /// registry again under the slot lock and starts nothing for an object
    /// the registry no longer holds: the model's index went with its last
    /// stream, and no stream is left for `stop` to end.
    #[traced_test]
    #[tokio::test]
    async fn a_worker_removed_during_the_restarts_wait_gets_no_stream() {
        let (app_context, monitor) = context_under_cache_aware();
        let first = grpc_worker();
        let id = app_context
            .worker_registry
            .register(Arc::clone(&first))
            .expect("the worker registers");
        monitor
            .on_worker_added(&first, SubscriptionKind::Events)
            .await;
        let second = grpc_worker();
        assert!(app_context.worker_registry.replace_claiming_local(
            &id,
            Arc::clone(&second),
            first.revision()
        ));
        let url = second.url().to_string();

        let data = WorkerUpdateWorkflowData {
            config: WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: None,
                api_key: None,
                health: None,
            },
            worker_url: url.clone(),
            app_context: Some(Arc::clone(&app_context)),
            workers_to_update: None,
            updated_workers: Some(vec![Arc::clone(&second)]),
            refresh_discovered_labels: true,
            fresh_engine_labels: None,
            snapshot: None,
        };
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        let mut step = UpdatePoliciesForWorkerStep.execute(&mut context);
        {
            // One poll: the policy work runs, the add finds the stream
            // started for the first object, stops it and waits for its end.
            let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
            assert!(
                std::future::Future::poll(step.as_mut(), &mut cx).is_pending(),
                "the restart waits for the old task to end"
            );
        }
        assert!(logs_contain(
            "Restarting KV event subscription for a re-registered worker"
        ));
        // The removal workflow, inside that wait: its registry step, then
        // its policy step's call on the monitor.
        assert!(
            app_context.worker_registry.remove_by_url(&url).is_some(),
            "the removal lands while the restart waits"
        );
        monitor.on_worker_removed(&url).await;

        step.await.expect("the step succeeds");
        assert!(
            monitor.get_indexer(MODEL).is_none(),
            "the model's index went with its last stream; nothing rebuilt it"
        );
        logs_assert(|lines: &[&str]| {
            match lines
                .iter()
                .filter(|line| line.contains("Starting KV event subscription"))
                .count()
            {
                1 => Ok(()),
                n => Err(format!("{n} starts; only the first worker's expected")),
            }
        });
        monitor.stop().await;
        assert!(
            !logs_contain("Stopping all KV event subscriptions"),
            "no stream survived the removal"
        );
    }

    /// Between the swap and this step another registration replaced the
    /// worker again: the registry holds a third object under the URL, and
    /// the second is nobody's. The read compares objects, not URLs: no
    /// restart for the second, and the stream the URL has stays with the
    /// object it was started for.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_rebuilt_worker_replaced_again_before_the_step_keeps_the_urls_stream() {
        let (app_context, monitor) = context_under_cache_aware();
        let first = grpc_worker();
        let id = app_context
            .worker_registry
            .register(Arc::clone(&first))
            .expect("the worker registers");
        monitor
            .on_worker_added(&first, SubscriptionKind::Events)
            .await;
        let second = grpc_worker();
        assert!(app_context.worker_registry.replace_claiming_local(
            &id,
            Arc::clone(&second),
            first.revision()
        ));
        let third = grpc_worker();
        assert!(
            app_context.worker_registry.replace_claiming_local(
                &id,
                Arc::clone(&third),
                second.revision()
            ),
            "a later registration replaces the worker again before the step"
        );

        let data = WorkerUpdateWorkflowData {
            config: WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: None,
                api_key: None,
                health: None,
            },
            worker_url: second.url().to_string(),
            app_context: Some(Arc::clone(&app_context)),
            workers_to_update: None,
            updated_workers: Some(vec![Arc::clone(&second)]),
            refresh_discovered_labels: true,
            fresh_engine_labels: None,
            snapshot: None,
        };
        let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
        UpdatePoliciesForWorkerStep
            .execute(&mut context)
            .await
            .expect("the step succeeds");

        assert!(
            !logs_contain("Restarting KV event subscription"),
            "no restart for an object the registry no longer holds"
        );
        logs_assert(|lines: &[&str]| {
            match lines
                .iter()
                .filter(|line| line.contains("Starting KV event subscription"))
                .count()
            {
                1 => Ok(()),
                n => Err(format!("{n} starts; only the first worker's expected")),
            }
        });
        monitor.stop().await;
    }
}
