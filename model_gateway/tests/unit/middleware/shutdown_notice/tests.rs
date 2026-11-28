use std::time::Duration;

use axum::{
    body::Body,
    http::{Request as HttpRequest, StatusCode},
    routing::get,
    Router,
};
use tower::ServiceExt;

use super::*;

/// Two routes behind the layer alone: one whose handler never answers, one
/// whose handler answers at its first poll.
fn routes(tracker: &Arc<InFlightRequestTracker>) -> Router {
    Router::new()
        .route("/never", get(std::future::pending::<StatusCode>))
        .route("/at-once", get(|| async { StatusCode::IM_A_TEAPOT }))
        .layer(ShutdownNoticeLayer::new(Arc::clone(tracker)))
}

async fn answer_of(app: Router, path: &str) -> Response {
    let request = HttpRequest::builder()
        .uri(path)
        .body(Body::empty())
        .unwrap();
    tokio::time::timeout(Duration::from_secs(1), app.oneshot(request))
        .await
        .expect("the request is answered without a wake")
        .unwrap()
}

/// A request that parks after the drain gave up gets no wake: the flag it
/// reads behind its registration answers it at that poll.
#[tokio::test(start_paused = true)]
async fn a_request_that_parks_after_the_giving_up_is_answered_503() {
    let tracker = InFlightRequestTracker::new();
    tracker.give_up_drain();

    let response = answer_of(routes(&tracker), "/never").await;

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["error"]["code"], SERVER_SHUTTING_DOWN_CODE, "{body}");
}

/// The handler is polled first: an answer it has leaves as it is, the drain
/// given up or not.
#[tokio::test(start_paused = true)]
async fn a_handlers_answer_wins_over_the_giving_up() {
    let tracker = InFlightRequestTracker::new();
    tracker.give_up_drain();

    let response = answer_of(routes(&tracker), "/at-once").await;

    assert_eq!(response.status(), StatusCode::IM_A_TEAPOT);
}
