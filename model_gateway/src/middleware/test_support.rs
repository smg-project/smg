//! What the middleware tests put under their layers: the echo route, which
//! answers a request with its own body, and the request that posts a body
//! to it.

use axum::{
    body::Body,
    extract::Request,
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
