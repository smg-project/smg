//! What the middleware tests put under their layers: the echo route, which
//! answers a request with its own body, the request that posts a body to
//! it, and the readers of a response's whole body, as JSON or as text, which
//! the crate's HTTP tests share.

use axum::{
    body::Body,
    extract::Request,
    response::Response,
    routing::{post, MethodRouter},
};
use bytes::Bytes;

/// The path the middleware tests mount the echo route on and post to.
pub(crate) const ECHO_PATH: &str = "/echo";

/// The echo handler: the request's body, back.
pub(crate) fn echo_route() -> MethodRouter {
    post(|body: Bytes| async move { body })
}

/// A POST of `body` to [`ECHO_PATH`].
pub(crate) fn echo_request(body: Body) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(ECHO_PATH)
        .body(body)
        .unwrap()
}

/// The whole body of `response`, read as JSON.
pub(crate) async fn json_of(response: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    serde_json::from_slice(&bytes).expect("a JSON body")
}

/// The whole body of `response`, read as text.
pub(crate) async fn text_of(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    String::from_utf8(bytes.to_vec()).expect("a UTF-8 body")
}
