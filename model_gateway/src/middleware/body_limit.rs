//! The payload limit of the HTTP edge: a cap on the request body, and
//! nothing on the response.
//!
//! A declared `Content-Length` over `--max-payload-size` is refused before
//! any layer inside runs, with the gateway's 413 envelope and the code
//! `request_body_too_large`. A body without a declared length, or declared
//! within the limit, is capped on its way in by `http_body_util::Limited`,
//! whose `LengthLimitError` the JSON extractor and the streaming relays turn
//! into the same 413 the moment a chunked upload crosses the limit; no
//! frame past the limit is pulled from the client.
//!
//! The layer is the gateway's own. `tower_http`'s request-body limit types
//! every response it relays as a body of its own, a boxed `poll_frame` more
//! on every streamed frame, to carry a refusal the response side never
//! needs; this one hands the inner response back as it is.

use std::task::{Context, Poll};

use axum::{body::Body, extract::Request, http::header::CONTENT_LENGTH, response::Response};
use futures_util::future::{ready, Either, Ready};
use http_body_util::Limited;
use tower::{Layer, Service};

use super::body_timeout::request_body_too_large_response;

/// Caps every request body at a number of bytes: see the module docs.
#[derive(Clone, Copy, Debug)]
pub struct RequestBodyLimitLayer {
    limit: usize,
}

impl RequestBodyLimitLayer {
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl<S> Layer<S> for RequestBodyLimitLayer {
    type Service = RequestBodyLimit<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestBodyLimit {
            inner,
            limit: self.limit,
        }
    }
}

/// The service [`RequestBodyLimitLayer`] wraps `S` in.
#[derive(Clone, Debug)]
pub struct RequestBodyLimit<S> {
    inner: S,
    limit: usize,
}

impl<S> Service<Request> for RequestBodyLimit<S>
where
    S: Service<Request, Response = Response>,
{
    type Response = Response;
    type Error = S::Error;
    type Future = Either<Ready<Result<Response, S::Error>>, S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request) -> Self::Future {
        let declared = request
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok()?.parse::<usize>().ok());
        let cap = match declared {
            Some(length) if length > self.limit => {
                return Either::Left(ready(Ok(request_body_too_large_response())));
            }
            Some(length) => length.min(self.limit),
            None => self.limit,
        };
        let request = request.map(|body| Body::new(Limited::new(body, cap)));
        Either::Right(self.inner.call(request))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/middleware/body_limit/tests.rs"]
mod tests;
