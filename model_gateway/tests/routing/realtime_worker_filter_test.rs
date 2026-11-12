//! The realtime REST routes select among the regular HTTP workers that carry
//! the `realtime` label, so a pool that mixes capable workers with plain ones
//! never sends realtime traffic to a worker that cannot serve it, and a pool
//! without a capable worker refuses rather than falling back to a plain one.

use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::{json, Value};
use smg::worker::BasicWorkerBuilder;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{set_request_recorder, RequestRecorder},
    AppTestContext, TestWorkerConfig,
};

#[cfg(test)]
mod realtime_worker_filter_tests {
    use super::*;

    const CAPABLE_PORT: u16 = 19880;
    const PLAIN_PORT: u16 = 19881;

    /// Re-register the started worker with the `realtime` label. Mock workers
    /// advertise no labels, so the label is installed after startup.
    fn label_realtime(ctx: &AppTestContext, url: &str) {
        let registry = &ctx.app_context.worker_registry;
        let worker_id = registry.get_id_by_url(url).unwrap();
        let worker = registry.get(&worker_id).unwrap();
        let mut spec = worker.metadata().spec.as_ref().clone();
        spec.labels
            .insert("realtime".to_string(), "true".to_string());
        let replacement = BasicWorkerBuilder::from_spec(spec)
            .health_config(worker.metadata().health_config.clone())
            .health_endpoint(&worker.metadata().health_endpoint)
            .build();
        assert!(registry.replace(&worker_id, Arc::new(replacement)));
    }

    fn session_request() -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/realtime/sessions")
            .header(CONTENT_TYPE, "application/json")
            .header("Authorization", "Bearer test-key")
            .body(Body::from(
                json!({"type": "realtime", "model": "mock-model"}).to_string(),
            ))
            .unwrap()
    }

    /// Eight realtime session requests over a labelled and a plain worker all
    /// reach the labelled one. With the labelled worker gone, the route
    /// refuses the model rather than handing the request to the plain worker.
    #[tokio::test]
    async fn realtime_rest_requests_reach_the_labelled_worker_alone() {
        let capable = RequestRecorder::new();
        set_request_recorder(CAPABLE_PORT, Arc::clone(&capable));
        let plain = RequestRecorder::new();
        set_request_recorder(PLAIN_PORT, Arc::clone(&plain));

        let ctx = AppTestContext::new(vec![
            TestWorkerConfig::healthy(CAPABLE_PORT),
            TestWorkerConfig::healthy(PLAIN_PORT),
        ])
        .await;
        let app = ctx.create_app();
        label_realtime(&ctx, ctx.worker_url_for(CAPABLE_PORT));

        for attempt in 0..8 {
            let response = app.clone().oneshot(session_request()).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "request {attempt}");
        }
        assert_eq!(
            capable.bodies().len(),
            8,
            "every realtime request reaches the labelled worker"
        );
        assert!(
            plain.bodies().is_empty(),
            "the plain worker takes no realtime request"
        );

        let registry = &ctx.app_context.worker_registry;
        let capable_id = registry
            .get_id_by_url(ctx.worker_url_for(CAPABLE_PORT))
            .unwrap();
        assert!(registry.remove(&capable_id).is_some());

        let response = app.clone().oneshot(session_request()).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "no capable worker is left for the model"
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "model_not_found", "{body}");
        assert!(
            plain.bodies().is_empty(),
            "the plain worker is not the fallback"
        );

        ctx.shutdown().await;
    }
}
