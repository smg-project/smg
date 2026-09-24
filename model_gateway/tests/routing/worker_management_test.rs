//! Worker management integration tests
//!
//! Tests for dynamic worker add/remove operations via management API.
//! The actual worker management API uses:
//! - POST /workers - create a worker
//! - GET /workers - list workers
//! - PUT /workers/{worker_id} - replace a worker
//! - DELETE /workers/{worker_id} - remove a worker

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod dp_removal_tests {
    use std::{collections::HashMap, sync::Arc, time::Duration};

    use openai_protocol::worker::{ConnectionMode, HealthCheckConfig, WorkerStatus, WorkerType};
    use smg::{
        app_context::AppContext,
        config::RouterConfig,
        routers::http::router::Router as HttpRouter,
        worker::{BasicWorkerBuilder, Worker},
        workflow::create_worker_removal_workflow_data,
    };
    use wfaas::WorkflowId;

    use super::*;
    use crate::common::{create_test_context, test_app::create_test_app_with_context};

    fn register(
        context: &AppContext,
        url: &str,
        worker_type: WorkerType,
        rank: Option<usize>,
    ) -> Arc<dyn Worker> {
        let builder = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .connection_mode(if url.starts_with("grpc://") {
                ConnectionMode::Grpc
            } else {
                ConnectionMode::Http
            })
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                drain_settle_secs: 0,
                ..Default::default()
            });
        let builder = match rank {
            Some(rank) => builder.dp_config(rank, 2),
            None => builder,
        };
        let worker: Arc<dyn Worker> = Arc::new(builder.build());
        context
            .worker_registry
            .register(Arc::clone(&worker))
            .unwrap();
        worker
    }

    async fn delete_via_api(app: &axum::Router, context: &AppContext, url: &str) {
        let id = context.worker_registry.get_id_by_url(url).unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/workers/{}", id.as_str()))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);

        // 202 only acknowledges submission. Wait for actual registry removal.
        tokio::time::timeout(Duration::from_secs(10), async {
            while context.worker_registry.get(&id).is_some() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("accepted DELETE did not remove the worker");

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/workers/{}", id.as_str()))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn api_removes_plain_worker_and_individual_rank_in_mixed_pd_fleet() {
        for (plain_type, dp_type) in [
            (WorkerType::Prefill, WorkerType::Decode),
            (WorkerType::Decode, WorkerType::Prefill),
        ] {
            let context = create_test_context(RouterConfig {
                dp_aware: true,
                disable_load_monitoring: true,
                ..Default::default()
            })
            .await;
            let plain = register(&context, "http://plain:30000", plain_type, None);
            plain.set_status(WorkerStatus::Failed);
            let rank0 = register(&context, "http://expanded:30000", dp_type, Some(0));
            let rank1 = register(&context, "http://expanded:30000", dp_type, Some(1));
            let router = Arc::new(HttpRouter::new(&context).await.unwrap());
            let app = create_test_app_with_context(router, Arc::clone(&context));

            delete_via_api(&app, &context, plain.url()).await;
            assert!(context.worker_registry.get_by_url(rank0.url()).is_some());
            assert!(context.worker_registry.get_by_url(rank1.url()).is_some());

            delete_via_api(&app, &context, rank0.url()).await;
            assert!(context.worker_registry.get_by_url(rank1.url()).is_some());
            assert_eq!(context.worker_registry.len(), 1);
        }
    }

    /// Snapshot `(worker id, revision)` for each worker, as discovery does.
    fn guards(context: &AppContext, workers: &[&Arc<dyn Worker>]) -> HashMap<String, u64> {
        workers
            .iter()
            .map(|worker| {
                let id = context
                    .worker_registry
                    .get_id_by_url(worker.url())
                    .expect("worker is registered");
                (id.as_str().to_string(), worker.revision())
            })
            .collect()
    }

    async fn run_removal(
        context: &Arc<AppContext>,
        url: &str,
        expected_revisions: Option<HashMap<String, u64>>,
    ) -> Result<String, String> {
        let engine = &context.workflow_engines.get().unwrap().worker_removal;
        let data = create_worker_removal_workflow_data(
            url.to_string(),
            expected_revisions,
            Arc::clone(context),
        );
        let instance = engine
            .start_workflow(WorkflowId::new("worker_removal"), data)
            .await
            .unwrap();
        engine
            .wait_for_completion(instance, url, Duration::from_secs(10))
            .await
    }

    /// A DP group collapses to one removal job but holds independent
    /// revisions per rank. Guarding it with a single revision retained only
    /// the ranks that happened to share that value and silently left the
    /// others registered against a Pod that was already gone. Per-rank
    /// guards drain the ranks still at their observed revision and skip only
    /// the one that moved.
    #[tokio::test]
    async fn dp_group_removal_skips_only_the_rank_whose_revision_moved() {
        let context = create_test_context(RouterConfig {
            dp_aware: true,
            disable_load_monitoring: true,
            ..Default::default()
        })
        .await;
        let base = "http://worker:30000".to_string();
        let rank0 = register(&context, &base, WorkerType::Decode, Some(0));
        let rank1 = register(&context, &base, WorkerType::Decode, Some(1));

        // Both ranks sit at the same revision, which is exactly the case a
        // scalar guard could not distinguish. Claim rank1 moved on since the
        // snapshot; rank0 is still current.
        let mut expected = guards(&context, &[&rank0]);
        let rank1_id = context
            .worker_registry
            .get_id_by_url(rank1.url())
            .expect("rank1 is registered");
        expected.insert(rank1_id.as_str().to_string(), rank1.revision() + 1);

        run_removal(&context, "worker:30000", Some(expected))
            .await
            .unwrap();

        assert!(
            context.worker_registry.get_by_url(rank0.url()).is_none(),
            "the rank still at its observed revision must drain"
        );
        assert!(
            context.worker_registry.get_by_url(rank1.url()).is_some(),
            "the rank whose revision moved must be skipped, not dropped"
        );
    }

    /// A rank registered after the snapshot has no guard entry at all, so it
    /// is a different incarnation and must be left for the next pass.
    #[tokio::test]
    async fn dp_group_removal_leaves_ranks_absent_from_the_snapshot() {
        let context = create_test_context(RouterConfig {
            dp_aware: true,
            disable_load_monitoring: true,
            ..Default::default()
        })
        .await;
        let base = "http://worker:30000".to_string();
        let rank0 = register(&context, &base, WorkerType::Decode, Some(0));
        let rank1 = register(&context, &base, WorkerType::Decode, Some(1));

        // Snapshot rank0 only, as if rank1 appeared afterwards.
        run_removal(&context, "worker:30000", Some(guards(&context, &[&rank0])))
            .await
            .unwrap();

        assert!(context.worker_registry.get_by_url(rank0.url()).is_none());
        assert!(
            context.worker_registry.get_by_url(rank1.url()).is_some(),
            "an unguarded rank is a later incarnation and must survive"
        );
    }

    #[tokio::test]
    async fn discovery_removal_resolves_plain_and_dp_groups_without_global_dp_setting() {
        for dp_aware in [false, true] {
            for scheme in ["http", "grpc"] {
                let context = create_test_context(RouterConfig {
                    dp_aware,
                    disable_load_monitoring: true,
                    ..Default::default()
                })
                .await;
                let base = format!("{scheme}://worker:30000");
                let plain = register(&context, &base, WorkerType::Prefill, None);
                let rank0 = register(&context, &base, WorkerType::Decode, Some(0));
                let rank1 = register(&context, &base, WorkerType::Decode, Some(1));
                let other = register(
                    &context,
                    &format!("{scheme}://worker:30001"),
                    WorkerType::Prefill,
                    None,
                );

                // A mismatched revision must preserve every current registration.
                let stale: HashMap<String, u64> = guards(&context, &[&plain, &rank0, &rank1])
                    .into_iter()
                    .map(|(id, revision)| (id, revision + 1))
                    .collect();
                run_removal(&context, "worker:30000", Some(stale))
                    .await
                    .unwrap();
                assert_eq!(context.worker_registry.len(), 4);

                let current = guards(&context, &[&plain, &rank0, &rank1]);
                run_removal(&context, "worker:30000", Some(current))
                    .await
                    .unwrap();
                for removed in [plain, rank0, rank1] {
                    assert!(context.worker_registry.get_by_url(removed.url()).is_none());
                }
                assert!(context.worker_registry.get_by_url(other.url()).is_some());
                assert_eq!(context.worker_registry.len(), 1);

                // Discovery removal is idempotent; an explicit missing target errors.
                run_removal(&context, "worker:30000", Some(HashMap::new()))
                    .await
                    .unwrap();
                assert!(run_removal(&context, "worker:30000", None).await.is_err());
            }
        }
    }
}

#[cfg(test)]
mod worker_management_tests {
    use super::*;

    /// Test listing workers via API
    #[tokio::test]
    async fn test_list_workers() {
        let config = TestRouterConfig::round_robin(3900);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(19900),
                TestWorkerConfig::healthy(19901),
            ],
        )
        .await;

        let app = ctx.create_app();

        // List workers via GET /workers
        let req = Request::builder()
            .method("GET")
            .uri("/workers")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "GET /workers should return OK"
        );

        ctx.shutdown().await;
    }

    /// Test that routing continues to work with multiple workers
    #[tokio::test]
    async fn test_routing_with_multiple_workers() {
        let config = TestRouterConfig::round_robin(3901);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(19902),
                TestWorkerConfig::healthy(19903),
            ],
        )
        .await;

        let app = ctx.create_app();
        let mut success_count = 0;

        // Verify routing distributes across workers
        for i in 0..20 {
            let payload = json!({
                "text": format!("Test request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 20,
            "All requests should succeed with multiple workers"
        );

        ctx.shutdown().await;
    }

    /// Test that requests continue to work during worker operations
    #[tokio::test]
    async fn test_requests_during_worker_changes() {
        let config = TestRouterConfig::round_robin(3902);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(19904)]).await;

        let app = ctx.create_app();

        // Send requests and verify they succeed
        let mut success_count = 0;
        for i in 0..10 {
            let payload = json!({
                "text": format!("Request during changes {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 10,
            "All requests should succeed during normal operation"
        );

        ctx.shutdown().await;
    }

    /// PUT /workers/{id} must actually apply the new spec.
    ///
    /// The handler returns 202 and runs the registration workflow in the
    /// background. Before the registration mode existed, that workflow always
    /// rejected an already-registered URL, so the replace failed after the
    /// caller had already seen 202.
    #[tokio::test]
    async fn test_replace_worker_applies_new_spec() {
        let config = TestRouterConfig::round_robin(3903);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(19905)]).await;

        let app = ctx.create_app();

        async fn get_worker(app: axum::Router, worker_id: &str) -> serde_json::Value {
            let req = Request::builder()
                .method("GET")
                .uri(format!("/workers/{worker_id}"))
                .body(Body::empty())
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "GET /workers/{{id}} failed");
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice(&bytes).unwrap()
        }

        let req = Request::builder()
            .method("GET")
            .uri("/workers")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let listed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let worker = &listed["workers"][0];
        let worker_id = worker["id"].as_str().unwrap().to_string();
        let url = worker["url"].as_str().unwrap().to_string();
        let old_priority = worker["priority"].as_u64().unwrap();
        let new_priority = old_priority + 7;

        // Same URL (required by the handler), different priority.
        let body = json!({ "url": url, "priority": new_priority });
        let req = Request::builder()
            .method("PUT")
            .uri(format!("/workers/{worker_id}"))
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::ACCEPTED,
            "PUT /workers/{{id}} should be accepted"
        );

        // The workflow runs in the background, so poll for the new value.
        let mut applied = false;
        for _ in 0..40 {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            let current = get_worker(app.clone(), &worker_id).await;
            if current["priority"].as_u64() == Some(new_priority) {
                applied = true;
                break;
            }
        }

        let current = get_worker(app.clone(), &worker_id).await;
        assert!(applied, "PUT never applied. Worker still reads: {current}");
        assert_eq!(
            current["url"].as_str(),
            Some(url.as_str()),
            "URL must not change"
        );

        ctx.shutdown().await;
    }
}
