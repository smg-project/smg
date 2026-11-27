use axum::{
    body::Body,
    http::{Request, StatusCode},
    routing::get,
    Router,
};
use openai_protocol::worker::WorkerErrorResponse;
use serde::Deserialize;
use tower::ServiceExt;

use super::*;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Origin {
    Local,
    Discovery,
}

#[derive(Debug, Deserialize)]
struct Filter {
    origin: Option<Origin>,
}

fn app() -> Router {
    Router::new()
        .route(
            "/workers",
            get(|AdminQuery(filter): AdminQuery<Filter>| async move {
                format!("{:?}", filter.origin)
            }),
        )
        .route(
            "/workers/{id}",
            get(|AdminPath(id): AdminPath<String>| async move { id }),
        )
        // a handler that reads a path parameter its route does not declare
        .route(
            "/undeclared",
            get(|AdminPath(id): AdminPath<String>| async move { id }),
        )
}

async fn send(uri: &'static str) -> (StatusCode, String) {
    let response = app()
        .oneshot(
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// The envelope of a refusal: `code` and `error`, the error naming what
/// was wrong.
fn envelope(uri: &str, text: &str) -> WorkerErrorResponse {
    serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("{uri}: not the control plane's envelope ({e}): {text}"))
}

/// A query string the type takes and a path segment that reads reach
/// the handler; a value outside the type's set and a segment that is not
/// UTF-8 are a 400 in the control plane's envelope with the framework's
/// reason.
#[tokio::test]
async fn a_query_string_or_path_segment_the_type_does_not_take_is_a_400_in_the_envelope() {
    let (status, text) = send("/workers?origin=local").await;
    assert_eq!((status, text.as_str()), (StatusCode::OK, "Some(Local)"));
    let (status, text) = send("/workers/w1").await;
    assert_eq!((status, text.as_str()), (StatusCode::OK, "w1"));
    for (uri, detail) in [
        ("/workers?origin=bogus", "unknown variant"),
        ("/workers/%ff", "UTF-8"),
    ] {
        let (status, text) = send(uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {text}");
        let refusal = envelope(uri, &text);
        assert_eq!(refusal.code, "BAD_REQUEST", "{uri}");
        assert!(refusal.error.contains(detail), "{uri}: {}", refusal.error);
    }
}

/// A rejection of the route's own making (a path parameter the route
/// does not declare) keeps the framework's answer, a 500 that is not the
/// envelope, never a 400 laid on the client.
#[tokio::test]
async fn a_rejection_of_the_routes_own_making_keeps_the_frameworks_answer() {
    let (status, text) = send("/undeclared").await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{text}");
    assert!(
        serde_json::from_str::<WorkerErrorResponse>(&text).is_err(),
        "the framework's answer, not the envelope: {text}"
    );
}
