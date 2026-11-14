//! The query-string and path-segment extractors of the control-plane
//! management endpoints (the worker and tokenizer routes): a value the type
//! does not take is refused in the control plane's own error envelope
//! (`{"error": ..., "code": ...}`, the shape [`super::admin_json::AdminJson`]
//! answers for a body), not with the framework's plain-text 400, so a client
//! reads one shape from these endpoints whatever went wrong.

use axum::{
    extract::{
        rejection::{PathRejection, QueryRejection},
        FromRequestParts, Path, Query,
    },
    http::{request::Parts, StatusCode},
    response::{IntoResponse, Response},
};
use serde::de::DeserializeOwned;

use super::admin_json::refusal;

/// A query string of the control-plane management endpoints.
pub(crate) struct AdminQuery<T>(pub T);

impl<S, T> FromRequestParts<S> for AdminQuery<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        match Query::<T>::from_request_parts(parts, state).await {
            Ok(Query(value)) => Ok(Self(value)),
            Err(err) => Err(refuse_query(&err)),
        }
    }
}

/// A path segment of the control-plane management endpoints.
pub(crate) struct AdminPath<T>(pub T);

impl<S, T> FromRequestParts<S> for AdminPath<T>
where
    T: DeserializeOwned + Send,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        match Path::<T>::from_request_parts(parts, state).await {
            Ok(Path(value)) => Ok(Self(value)),
            Err(err) => Err(refuse_path(err)),
        }
    }
}

/// A query string the type does not take: 400 in the envelope with the
/// framework's reason.
fn refuse_query(err: &QueryRejection) -> Response {
    refusal(StatusCode::BAD_REQUEST, "BAD_REQUEST", err.body_text())
}

/// A path segment that is not UTF-8 or does not parse as the type: 400 in
/// the envelope with the framework's reason. A rejection the route itself
/// causes (no such parameter, a shape the extractor cannot fill) keeps the
/// framework's answer: a 500 of the route's making, never a client's.
fn refuse_path(err: PathRejection) -> Response {
    if err.status() == StatusCode::BAD_REQUEST {
        refusal(StatusCode::BAD_REQUEST, "BAD_REQUEST", err.body_text())
    } else {
        err.into_response()
    }
}

#[cfg(test)]
mod tests {
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
}
