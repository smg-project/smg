//! Weighted traffic split integration tests on the regular HTTP router
//! (`--traffic-split`): three mock workers, two in the group `stable` at
//! weight 90 and one in `canary` at weight 10, behind round robin.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::{
    policies::traffic_split::{GROUP_LABEL, WEIGHT_LABEL},
    worker::BasicWorkerBuilder,
};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{set_request_recorder, RequestRecorder},
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};

#[cfg(test)]
mod traffic_split_tests {
    use super::*;

    /// Give the started worker named `port` the split labels, by replacing it
    /// with one carrying them (as a `PATCH /workers/{id}` with labels does).
    fn label(ctx: &AppTestContext, port: u16, group: &str, weight: &str) {
        let registry = &ctx.app_context.worker_registry;
        let worker_id = registry.get_id_by_url(ctx.worker_url_for(port)).unwrap();
        let worker = registry.get(&worker_id).unwrap();
        let mut spec = worker.metadata().spec.as_ref().clone();
        spec.labels
            .insert(GROUP_LABEL.to_string(), group.to_string());
        spec.labels
            .insert(WEIGHT_LABEL.to_string(), weight.to_string());
        let replacement = BasicWorkerBuilder::from_spec(spec)
            .health_config(worker.metadata().health_config.clone())
            .health_endpoint(&worker.metadata().health_endpoint)
            .build();
        assert!(registry.replace(&worker_id, Arc::new(replacement)));
    }

    /// A chat completion for the model the mock workers advertise; the mock
    /// records the bodies of this route per worker, which is how the hits are
    /// counted.
    fn chat_request(text: &str, routing_key: Option<&str>) -> Request<Body> {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json");
        if let Some(key) = routing_key {
            builder = builder.header("x-smg-routing-key", key);
        }
        builder
            .body(Body::from(
                json!({
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": text}],
                    "stream": false
                })
                .to_string(),
            ))
            .unwrap()
    }

    struct Bed {
        ctx: AppTestContext,
        app: axum::Router,
        stable_a: Arc<RequestRecorder>,
        stable_b: Arc<RequestRecorder>,
        canary: Arc<RequestRecorder>,
    }

    impl Bed {
        /// Three healthy workers named `base`, `base + 1` (stable, weight 90)
        /// and `base + 2` (canary, weight `canary_weight`), split on.
        async fn start(gateway_port: u16, base: u16, canary_weight: &str) -> Self {
            let recorders: Vec<Arc<RequestRecorder>> = (0..3)
                .map(|offset| {
                    let recorder = RequestRecorder::new();
                    set_request_recorder(base + offset, Arc::clone(&recorder));
                    recorder
                })
                .collect();
            let mut config = TestRouterConfig::round_robin(gateway_port);
            config.traffic_split = true;
            let ctx = AppTestContext::new_with_config(
                config,
                vec![
                    TestWorkerConfig::healthy(base),
                    TestWorkerConfig::healthy(base + 1),
                    TestWorkerConfig::healthy(base + 2),
                ],
            )
            .await;
            label(&ctx, base, "stable", "90");
            label(&ctx, base + 1, "stable", "90");
            label(&ctx, base + 2, "canary", canary_weight);
            let app = ctx.create_app();
            let mut recorders = recorders.into_iter();
            Self {
                ctx,
                app,
                stable_a: recorders.next().unwrap(),
                stable_b: recorders.next().unwrap(),
                canary: recorders.next().unwrap(),
            }
        }

        async fn send(&self, n: usize, routing_key: Option<&str>) {
            for i in 0..n {
                let response = self
                    .app
                    .clone()
                    .oneshot(chat_request(&format!("request {i}"), routing_key))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
            }
        }

        fn counts(&self) -> (usize, usize, usize) {
            (
                self.stable_a.bodies().len(),
                self.stable_b.bodies().len(),
                self.canary.bodies().len(),
            )
        }
    }

    /// Keyless requests follow the weights: the canary takes about a tenth,
    /// round robin alternates inside the stable group; relabelling the canary
    /// to weight 0 drains it while it stays registered.
    #[tokio::test]
    async fn a_90_10_split_sends_a_tenth_to_the_canary_and_a_zero_weight_drains_it() {
        let bed = Bed::start(3290, 19930, "10").await;
        let n = 600;
        bed.send(n, None).await;
        let (a, b, canary) = bed.counts();
        assert_eq!(a + b + canary, n);
        // Mean 60, standard deviation 7.3: a 4-sigma band.
        assert!(
            (30..=90).contains(&canary),
            "canary took {canary} of {n} (stable {a} / {b})"
        );
        assert!(
            a.abs_diff(b) <= 2,
            "round robin inside the stable group: {a} / {b}"
        );

        label(&bed.ctx, 19932, "canary", "0");
        let app = bed.ctx.create_app();
        for i in 0..100 {
            let response = app
                .clone()
                .oneshot(chat_request(&format!("drained {i}"), None))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let (a2, b2, canary2) = bed.counts();
        assert_eq!(canary2, canary, "a drained canary takes nothing");
        assert_eq!(a2 + b2, a + b + 100);

        bed.ctx.shutdown().await;
    }

    /// Every request of a keyed session lands in one group.
    #[tokio::test]
    async fn a_routing_key_keeps_a_session_in_one_group() {
        let bed = Bed::start(3291, 19933, "50").await;
        let mut in_canary = 0;
        let mut in_stable = 0;
        for session in 0..12 {
            let before = bed.counts();
            bed.send(10, Some(&format!("session-{session}"))).await;
            let after = bed.counts();
            let (stable, canary) = (
                (after.0 + after.1) - (before.0 + before.1),
                after.2 - before.2,
            );
            assert!(
                (stable, canary) == (10, 0) || (stable, canary) == (0, 10),
                "session-{session} split across groups: stable {stable}, canary {canary}"
            );
            if canary == 10 {
                in_canary += 1;
            } else {
                in_stable += 1;
            }
        }
        // Weight 50 against 90: both groups see sessions over twelve keys in
        // all but about one run in a hundred thousand.
        assert!(
            in_canary > 0 && in_stable > 0,
            "canary {in_canary}, stable {in_stable}"
        );

        bed.ctx.shutdown().await;
    }
}
