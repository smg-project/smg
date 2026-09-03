//! `--upstream-http2` resolves the HTTP version per worker at registration:
//! a worker that answers HTTP/2 prior knowledge is spoken to over h2c, one
//! that does not (or is pinned via `http_pool.http2`) stays on HTTP/1.1.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode, Version},
};
use serde_json::json;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{set_request_recorder, MockWorker, RequestRecorder},
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};

#[cfg(test)]
mod upstream_http2_tests {
    use super::*;

    async fn list_workers(app: axum::Router) -> serde_json::Value {
        let req = Request::builder()
            .method("GET")
            .uri("/workers")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn chat(app: axum::Router) -> StatusCode {
        let body = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        app.oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn upstream_http2_negotiates_h2c_with_a_dual_protocol_worker() {
        let mut config = TestRouterConfig::round_robin(3913);
        config.upstream_http2 = true;
        let recorder = RequestRecorder::new();
        set_request_recorder(19913, recorder.clone());

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(19913)]).await;
        let app = ctx.create_app();

        let listed = list_workers(app.clone()).await;
        assert_eq!(listed["workers"][0]["http2"], json!(true));

        assert_eq!(chat(app).await, StatusCode::OK);
        assert_eq!(recorder.versions(), vec![Version::HTTP_2]);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn upstream_http2_off_keeps_workers_on_http1() {
        let config = TestRouterConfig::round_robin(3914);
        let recorder = RequestRecorder::new();
        set_request_recorder(19914, recorder.clone());

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(19914)]).await;
        let app = ctx.create_app();

        let listed = list_workers(app.clone()).await;
        assert_eq!(listed["workers"][0]["http2"], json!(false));

        assert_eq!(chat(app).await, StatusCode::OK);
        assert_eq!(recorder.versions(), vec![Version::HTTP_11]);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn declared_http_pool_http2_false_pins_http1_under_upstream_http2() {
        let mut config = TestRouterConfig::round_robin(3915);
        config.upstream_http2 = true;
        let recorder = RequestRecorder::new();
        set_request_recorder(19915, recorder.clone());

        let ctx = AppTestContext::new_with_config(config, vec![]).await;
        let app = ctx.create_app();

        let mut worker = MockWorker::new(TestWorkerConfig::healthy(19915));
        let url = worker.start().await.unwrap();

        let body = json!({ "url": url, "http_pool": { "http2": false } });
        let req = Request::builder()
            .method("POST")
            .uri("/workers")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::ACCEPTED);

        // Registration runs in the background; wait for the worker to be routable.
        let mut listed = list_workers(app.clone()).await;
        for _ in 0..50 {
            if listed["workers"][0]["is_healthy"] == json!(true) {
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            listed = list_workers(app.clone()).await;
        }
        assert_eq!(listed["workers"][0]["is_healthy"], json!(true), "{listed}");
        assert_eq!(listed["workers"][0]["http2"], json!(false));

        assert_eq!(chat(app).await, StatusCode::OK);
        assert_eq!(recorder.versions(), vec![Version::HTTP_11]);

        worker.stop().await;
        ctx.shutdown().await;
    }
}
