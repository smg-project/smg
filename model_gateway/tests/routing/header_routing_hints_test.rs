//! Header routing-hint integration tests.
//!
//! `x-smg-routing-tokens` and `x-smg-routing-key` let text-needing policies
//! route without the body content: a valid token hint wins over body-derived
//! tokens/text, a valid key hint wins over token/text keying, and anything
//! malformed or over-cap is ignored so the request routes exactly as it
//! would without the header.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::{
    policies::{CacheAwareConfig, CacheAwarePolicy, LoadBalancingPolicy, SelectWorkerInfo},
    routers::common::header_utils::parse_routing_tokens_hint,
    worker::{BasicWorkerBuilder, Worker, WorkerType},
};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{set_request_recorder, RequestRecorder},
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};

#[cfg(test)]
mod header_routing_hints_tests {
    use super::*;

    const WORKER_COUNT: u16 = 3;

    /// Start `WORKER_COUNT` recorded workers behind a router with `config`.
    async fn setup(
        config: smg::config::RouterConfig,
        worker_port: u16,
    ) -> (AppTestContext, Vec<Arc<RequestRecorder>>, String) {
        let recorders: Vec<Arc<RequestRecorder>> = (0..WORKER_COUNT)
            .map(|i| {
                let recorder = RequestRecorder::new();
                set_request_recorder(worker_port + i, Arc::clone(&recorder));
                recorder
            })
            .collect();

        let ctx = AppTestContext::new_with_config(
            config,
            TestWorkerConfig::healthy_workers(worker_port, WORKER_COUNT),
        )
        .await;

        let model_id = ctx
            .app_context
            .worker_registry
            .get_all()
            .first()
            .expect("workers are registered")
            .model_id()
            .to_string();

        (ctx, recorders, model_id)
    }

    fn chat_request(model: &str, prompt: &str, hints: &[(&str, &str)]) -> Request<Body> {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json");
        for (name, value) in hints {
            builder = builder.header(*name, *value);
        }
        builder
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

    /// 32 comma-separated ids: two full token-tree pages (PAGE_SIZE = 16).
    fn token_hint() -> String {
        (100u32..132)
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    #[tokio::test]
    async fn test_token_hint_pins_cache_aware_worker_across_bodies() {
        let (ctx, recorders, model_id) = setup(TestRouterConfig::cache_aware(3170), 19450).await;
        let app = ctx.create_app();
        let hint = token_hint();

        for i in 0..6 {
            let response = app
                .clone()
                .oneshot(chat_request(
                    &model_id,
                    &format!("entirely different prompt {i}"),
                    &[("x-smg-routing-tokens", &hint)],
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let served: Vec<usize> = recorders.iter().map(|r| r.bodies().len()).collect();
        assert_eq!(
            served.iter().filter(|count| **count > 0).count(),
            1,
            "one hinted prefix must pin to a single worker, got {served:?}"
        );
        assert_eq!(served.iter().sum::<usize>(), 6);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_no_hint_distinct_bodies_spread_cache_aware_workers() {
        let (ctx, recorders, model_id) = setup(TestRouterConfig::cache_aware(3171), 19453).await;
        let app = ctx.create_app();

        // Prompts share no prefix, so the string tree never produces a cache hit.
        for prompt in [
            "alpha wolves hunt at dusk",
            "borrow a cup of sugar",
            "cedar boxes hold old maps",
            "delta flights leave early",
            "every engine hums a note",
            "frozen lakes crack loudly",
        ] {
            let response = app
                .clone()
                .oneshot(chat_request(&model_id, prompt, &[]))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let served: Vec<usize> = recorders.iter().map(|r| r.bodies().len()).collect();
        assert!(
            served.iter().filter(|count| **count > 0).count() > 1,
            "without a hint, distinct prompts must not pile onto one worker, got {served:?}"
        );
        assert_eq!(served.iter().sum::<usize>(), 6);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_malformed_hints_are_ignored_and_requests_route() {
        let (ctx, recorders, model_id) = setup(TestRouterConfig::cache_aware(3172), 19456).await;
        let app = ctx.create_app();

        let over_cap_tokens = vec!["7"; 513].join(",");
        let over_cap_key = "k".repeat(129);
        let malformed: Vec<(&str, &str)> = vec![
            ("x-smg-routing-tokens", "1,not-a-number,3"),
            ("x-smg-routing-tokens", "-5,3"),
            ("x-smg-routing-tokens", over_cap_tokens.as_str()),
            ("x-smg-routing-key", over_cap_key.as_str()),
        ];

        for (name, value) in &malformed {
            let response = app
                .clone()
                .oneshot(chat_request(
                    &model_id,
                    "same prompt every time",
                    &[(name, value)],
                ))
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "a bad hint must never fail the request ({name}: {value:.32})"
            );
        }

        let served: usize = recorders.iter().map(|r| r.bodies().len()).sum();
        assert_eq!(served, malformed.len(), "every request must reach a worker");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_routing_key_hint_pins_prefix_hash_worker_across_prompts() {
        let (ctx, recorders, model_id) =
            setup(TestRouterConfig::prefix_hash(3173, 16), 19459).await;
        let app = ctx.create_app();

        for i in 0..6 {
            let response = app
                .clone()
                .oneshot(chat_request(
                    &model_id,
                    &format!("entirely different prompt {i}"),
                    &[("x-smg-routing-key", "session-abc")],
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let pinned: Vec<usize> = recorders.iter().map(|r| r.bodies().len()).collect();
        assert_eq!(
            pinned.iter().filter(|count| **count > 0).count(),
            1,
            "one routing key must pin to a single worker, got {pinned:?}"
        );

        // Distinct keys with the same prompt must spread.
        for i in 0..12 {
            let response = app
                .clone()
                .oneshot(chat_request(
                    &model_id,
                    "same prompt every time",
                    &[("x-smg-routing-key", &format!("session-{i}"))],
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let served: Vec<usize> = recorders.iter().map(|r| r.bodies().len()).collect();
        let spread: Vec<usize> = served
            .iter()
            .zip(&pinned)
            .map(|(total, before)| total - before)
            .collect();
        assert!(
            spread.iter().filter(|count| **count > 0).count() > 1,
            "distinct keys must not pile onto one worker, got {spread:?}"
        );
        assert_eq!(served.iter().sum::<usize>(), 18);

        ctx.shutdown().await;
    }

    /// Policy-layer check: hinted tokens flow through cache_aware's token-tree
    /// selection with no request text at all, exactly as the router builds
    /// `SelectWorkerInfo` from a parsed `x-smg-routing-tokens` header.
    #[test]
    fn test_cache_aware_selects_same_worker_for_same_hinted_prefix() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });

        let workers: Vec<Arc<dyn Worker>> = ["http://w1:8000", "http://w2:8000", "http://w3:8000"]
            .iter()
            .map(|url| {
                let worker = BasicWorkerBuilder::new(*url)
                    .worker_type(WorkerType::Regular)
                    .health_config(openai_protocol::worker::HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build();
                policy.add_worker(&worker);
                Arc::new(worker) as Arc<dyn Worker>
            })
            .collect();

        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-tokens", token_hint().parse().unwrap());
        let hinted = parse_routing_tokens_hint(Some(&headers)).expect("hint is valid");
        let info = SelectWorkerInfo {
            tokens: Some(&hinted),
            request_text: None,
            ..Default::default()
        };

        let first = policy
            .select_worker(&workers, &info)
            .expect("hinted request must route");
        for _ in 0..5 {
            assert_eq!(
                policy.select_worker(&workers, &info),
                Some(first),
                "the same hinted prefix must keep its worker"
            );
        }

        // A different hinted prefix is free to learn its own worker.
        let other: Vec<u32> = (500u32..532).collect();
        let other_info = SelectWorkerInfo {
            tokens: Some(&other),
            request_text: None,
            ..Default::default()
        };
        let other_first = policy
            .select_worker(&workers, &other_info)
            .expect("hinted request must route");
        assert_ne!(other_first, first, "an unseen prefix goes to min-load");
    }
}
