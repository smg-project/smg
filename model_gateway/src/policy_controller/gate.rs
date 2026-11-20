//! The gate ahead of every router of a gateway that follows a controller.
//!
//! Such a gateway takes every worker from its controller, so it holds none
//! until the controller's first whole snapshot. A router asked before that
//! finds no worker for any model and answers that the model is not served,
//! which a client reads as final. The gate answers such a request as the
//! controller's absence, with a retry hint ([`host::awaiting_fleet`]).
//!
//! A stored response or conversation is read and changed with no worker,
//! so those routes pass the gate. Once a fleet was told the gate is one
//! load of a flag: the store keeps its last rows from there on.

use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
};

use axum::{extract::Request, response::Response};
use futures_util::future::{ready, Either, Ready};
use tower::{Layer, Service};

use super::host::{self, Host};

/// The gate of one gateway, as a layer of its route stacks.
#[derive(Clone)]
pub(crate) struct FleetGate(Arc<Gate>);

struct Gate {
    host: Arc<Host>,
    /// Set once the controller told a fleet.
    told: AtomicBool,
}

impl FleetGate {
    /// The gate of `host`, where the host follows a controller. None
    /// elsewhere: a gateway that decides in its own process finds its
    /// workers itself.
    pub(crate) fn of(host: Option<&Arc<Host>>) -> Option<Self> {
        let host = host?;
        host.fleet_told()?;
        Some(Self(Arc::new(Gate {
            host: Arc::clone(host),
            told: AtomicBool::new(false),
        })))
    }

    /// Whether the controller has told a fleet. A fleet once told stands,
    /// so the flag is never taken back.
    fn told(&self) -> bool {
        if self.0.told.load(Ordering::Relaxed) {
            return true;
        }
        let told = self.0.host.fleet_told().unwrap_or(true);
        if told {
            self.0.told.store(true, Ordering::Relaxed);
        }
        told
    }
}

/// Whether a request of `path` places nothing on a worker: a stored
/// response or conversation is read, changed or deleted by its id.
fn stored(path: &str) -> bool {
    path.starts_with("/v1/conversations") || path.starts_with("/v1/responses/")
}

impl<S> Layer<S> for FleetGate {
    type Service = Gated<S>;

    fn layer(&self, inner: S) -> Self::Service {
        Gated {
            inner,
            gate: self.clone(),
        }
    }
}

/// The service of a [`FleetGate`] around a route.
#[derive(Clone)]
pub(crate) struct Gated<S> {
    inner: S,
    gate: FleetGate,
}

impl<S> Service<Request> for Gated<S>
where
    S: Service<Request, Response = Response, Error = Infallible>,
{
    type Response = Response;
    type Error = Infallible;
    type Future = Either<S::Future, Ready<Result<Response, Infallible>>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request) -> Self::Future {
        if self.gate.told() || stored(request.uri().path()) {
            return Either::Left(self.inner.call(request));
        }
        Either::Right(ready(Ok(host::awaiting_fleet())))
    }
}
