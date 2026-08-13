//! Prefix-hash policy integration tests for the regular HTTP router.
//!
//! Only pre-tokenized `/generate` requests carry token IDs. Chat and
//! completions requests reach the policy with routing text alone, so these
//! tests drive the router the way a client does and assert that text-only
//! traffic is both served and hashed onto a stable worker.

use std::{collections::HashSet, sync::Arc};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{set_request_recorder, RequestRecorder},
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};

#[cfg(test)]
mod prefix_hash_tests {
    use super::*;

    const WORKER_COUNT: u16 = 3;

    /// Start `WORKER_COUNT` recorded workers behind a prefix-hash router.
    async fn setup(
        router_port: u16,
        worker_port: u16,
    ) -> (AppTestContext, Vec<Arc<RequestRecorder>>, String) {
        let recorders: Vec<Arc<RequestRecorder>> = (0..WORKER_COUNT)
            .map(|i| {
                let recorder = RequestRecorder::new();
                set_request_recorder(worker_port + i, Arc::clone(&recorder));
                recorder
            })
            .collect();

        let config = TestRouterConfig::prefix_hash(router_port, 16);
        let ctx = AppTestContext::new_with_config(
            config,
            TestWorkerConfig::healthy_workers(worker_port, WORKER_COUNT),
        )
        .await;

        let registry = &ctx.app_context.worker_registry;
        let model_id = registry
            .get_all()
            .first()
            .expect("workers are registered")
            .model_id()
            .to_string();
        assert!(
            registry.get_hash_ring(&model_id).is_some(),
            "the ring for {model_id} has to exist for the policy to hash onto it"
        );

        (ctx, recorders, model_id)
    }

    fn chat_request(model: &str, prompt: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": false
                })
                .to_string(),
            ))
            .unwrap()
    }

    #[tokio::test]
    async fn test_repeated_prompt_pins_to_one_worker() {
        let (ctx, recorders, model_id) = setup(3160, 19430).await;
        let app = ctx.create_app();

        for _ in 0..6 {
            let response = app
                .clone()
                .oneshot(chat_request(&model_id, "explain consistent hashing"))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let served: Vec<usize> = recorders.iter().map(|r| r.bodies().len()).collect();
        assert_eq!(
            served.iter().filter(|count| **count > 0).count(),
            1,
            "one prompt must hash to a single worker, got {served:?}"
        );
        assert_eq!(served.iter().sum::<usize>(), 6);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_distinct_prompts_spread_across_workers() {
        let (ctx, recorders, model_id) = setup(3161, 19433).await;
        let app = ctx.create_app();

        for i in 0..12 {
            let response = app
                .clone()
                .oneshot(chat_request(&model_id, &format!("prompt number {i}")))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let served: Vec<usize> = recorders.iter().map(|r| r.bodies().len()).collect();
        assert!(
            served.iter().filter(|count| **count > 0).count() > 1,
            "distinct prompts must not pile onto one worker, got {served:?}"
        );
        assert_eq!(served.iter().sum::<usize>(), 12);

        // Every prompt reached a worker, and no prompt was split across them.
        let prompts: HashSet<String> = recorders
            .iter()
            .flat_map(|r| r.bodies())
            .map(|body| body["messages"][0]["content"].to_string())
            .collect();
        assert_eq!(prompts.len(), 12);

        ctx.shutdown().await;
    }
}
