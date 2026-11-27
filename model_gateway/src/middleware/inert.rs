//! A stack member that is a pass-through when its layer is not configured.
//!
//! The protected routes' middleware is one `tower::ServiceBuilder` stack
//! applied in one `route_layer` call, whatever the configuration: axum boxes
//! the routes once per `route_layer` call and a request clones that boxed
//! chain at every level, so every call is a level and every level costs its
//! depth in boxes per request. A layer that is installed only with a serving
//! key, with wasm, or with an admission mode cannot be its own call then; it
//! is a member of the one stack that passes the request straight through when
//! it has nothing to do. `tower::util::option_layer` would do the same, but
//! its `Either` service widens the error type to `BoxError`, past the
//! `Into<Infallible>` axum's router requires; this enum keeps both arms'
//! `Infallible`. An inert member costs one match per request; it is not a
//! level.

use std::{
    convert::Infallible,
    task::{Context, Poll},
};

use axum::{extract::Request, response::Response};
use futures_util::future::Either;
use tower::{Layer, Service};

/// `layer` as a member of a stack: its service around the inner one when
/// there is a layer, the inner service untouched when there is none.
#[derive(Clone)]
pub struct InertLayer<L>(pub Option<L>);

impl<S, L> Layer<S> for InertLayer<L>
where
    L: Layer<S>,
{
    type Service = InertService<S, L::Service>;

    fn layer(&self, inner: S) -> Self::Service {
        match &self.0 {
            Some(layer) => InertService::Active(layer.layer(inner)),
            None => InertService::Through(inner),
        }
    }
}

/// The service of an [`InertLayer`]: the inner service as it is, or the
/// configured layer's service around it.
#[derive(Clone)]
pub enum InertService<S, A> {
    /// No layer configured: the inner service, called directly.
    Through(S),
    /// The configured layer's service around the inner one.
    Active(A),
}

impl<S, A> Service<Request> for InertService<S, A>
where
    S: Service<Request, Response = Response, Error = Infallible>,
    A: Service<Request, Response = Response, Error = Infallible>,
{
    type Response = Response;
    type Error = Infallible;
    type Future = Either<S::Future, A::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        match self {
            Self::Through(inner) => inner.poll_ready(cx),
            Self::Active(active) => active.poll_ready(cx),
        }
    }

    fn call(&mut self, request: Request) -> Self::Future {
        match self {
            Self::Through(inner) => Either::Left(inner.call(request)),
            Self::Active(active) => Either::Right(active.call(request)),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/middleware/inert/tests.rs"]
mod tests;
