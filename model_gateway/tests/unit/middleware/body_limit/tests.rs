use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use axum::{
    body::{to_bytes, Body},
    http::{
        header::{CONTENT_LENGTH, CONTENT_TYPE},
        StatusCode,
    },
    response::IntoResponse,
    routing::post,
    Router,
};
use bytes::Bytes;
use futures_util::stream;
use tower::ServiceExt;

use super::*;
use crate::{
    middleware::{
        body_timeout::{crossed_payload_limit, REQUEST_BODY_TOO_LARGE},
        test_support::{json_of, text_of},
    },
    routers::error::extract_error_code_from_response,
};

/// The layer over two handlers: `/read` reads its body and answers its
/// length, 413 `limit` when the limit cut the read and 400 for any
/// other failed read, counting its runs in `reads`; `/text-413` answers
/// a plain-text 413 of its own (an upstream's page relayed as it is).
fn app(limit: usize, reads: Arc<AtomicUsize>) -> Router {
    Router::new()
        .route(
            "/read",
            post(move |request: Request| {
                let reads = reads.clone();
                async move {
                    reads.fetch_add(1, Ordering::SeqCst);
                    match to_bytes(request.into_body(), usize::MAX).await {
                        Ok(bytes) => bytes.len().to_string().into_response(),
                        Err(err) if crossed_payload_limit(&err) => {
                            (StatusCode::PAYLOAD_TOO_LARGE, "limit").into_response()
                        }
                        Err(_) => StatusCode::BAD_REQUEST.into_response(),
                    }
                }
            }),
        )
        .route(
            "/text-413",
            post(|| async {
                (StatusCode::PAYLOAD_TOO_LARGE, "Request Entity Too Large").into_response()
            }),
        )
        .layer(RequestBodyLimitLayer::new(limit))
}

fn declared(uri: &str, length: usize) -> Request {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .header(CONTENT_LENGTH, length)
        .body(Body::from("x".repeat(length)))
        .unwrap()
}

fn chunked(uri: &str, frames: &[&'static [u8]]) -> Request {
    let frames: Vec<Result<Bytes, Infallible>> = frames
        .iter()
        .map(|frame| Ok(Bytes::from_static(frame)))
        .collect();
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from_stream(stream::iter(frames)))
        .unwrap()
}

/// A declared length over the limit is refused with the gateway's
/// envelope and code before the handler runs.
#[tokio::test]
async fn a_declared_length_over_the_limit_is_refused_before_the_handler() {
    let reads = Arc::new(AtomicUsize::new(0));
    let response = app(16, reads.clone())
        .oneshot(declared("/read", 100))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        extract_error_code_from_response(&response),
        REQUEST_BODY_TOO_LARGE
    );
    let json = json_of(response).await;
    assert_eq!(json["error"]["code"], REQUEST_BODY_TOO_LARGE, "{json}");
    assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
    assert_eq!(reads.load(Ordering::SeqCst), 0, "the handler never ran");
}

/// A body that crosses the limit without declaring its length fails the
/// handler's read with the limit's error, where the extractors and the
/// relays look for it.
#[tokio::test]
async fn a_body_that_crosses_the_limit_fails_the_read_with_the_limit_error() {
    let reads = Arc::new(AtomicUsize::new(0));
    let response = app(16, reads.clone())
        .oneshot(chunked("/read", &[b"0123456789", b"0123456789"]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(text_of(response).await, "limit");
    assert_eq!(reads.load(Ordering::SeqCst), 1);
}

/// A body within the limit, declared or chunked, reaches the handler
/// whole.
#[tokio::test]
async fn a_body_within_the_limit_reaches_the_handler_whole() {
    let reads = Arc::new(AtomicUsize::new(0));
    let app = app(16, reads.clone());
    let response = app.clone().oneshot(declared("/read", 16)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(text_of(response).await, "16");

    let response = app
        .oneshot(chunked("/read", &[b"0123", b"4567"]))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(text_of(response).await, "8");
    assert_eq!(reads.load(Ordering::SeqCst), 2);
}

/// The response side is not the layer's business: a handler's own 413,
/// plain text and all, leaves as the handler made it.
#[tokio::test]
async fn the_response_leaves_as_the_handler_made_it() {
    let response = app(16, Arc::default())
        .oneshot(declared("/text-413", 4))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(extract_error_code_from_response(&response), "");
    assert_eq!(text_of(response).await, "Request Entity Too Large");
}
