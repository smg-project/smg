//! Handler-side preemption integration.
//!
//! `priority_admission_middleware` inserts the admission cancel token into
//! request extensions. Long-running handlers opt into preemption by taking a
//! `PreemptionGuard` extractor and running their upstream work through
//! [`PreemptionGuard::guard`], which races the work against the token: if the
//! request is preempted before the response is produced, the work future is
//! dropped (cancelling the upstream call) and a `Preempted` response is
//! returned instead.
//!
//! Once a streaming response has been produced, preemption is handled by the
//! response body wrapper (see `body.rs`), not here — so `guard` only needs to
//! cover the pre-response window, which is exactly the pre-TTFT window a
//! preemption victim sits in.

use std::{convert::Infallible, future::Future};

use axum::{
    extract::FromRequestParts,
    http::{request::Parts, Extensions},
    response::{IntoResponse, Response},
};
use tokio_util::sync::CancellationToken;

use super::SchedulerError;

/// Infallible extractor for the per-request preemption token. In priority
/// mode the admission middleware inserts the token; in legacy mode it is
/// absent and the guard holds none, so wrapped handlers run their work as
/// they would unguarded. (A fresh token in its place was one allocation per
/// request that nothing could ever cancel.)
#[derive(Clone, Default)]
pub struct PreemptionGuard(Option<CancellationToken>);

impl<S> FromRequestParts<S> for PreemptionGuard
where
    S: Send + Sync,
{
    type Rejection = Infallible;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Infallible> {
        Ok(Self::from_extensions(&parts.extensions))
    }
}

impl PreemptionGuard {
    /// The request's token where the admission layer put one; none otherwise
    /// (legacy admission). What the extractor reads, for a layer that holds
    /// the request itself.
    pub fn from_extensions(extensions: &Extensions) -> Self {
        Self(extensions.get::<CancellationToken>().cloned())
    }

    /// Run `fut`, returning its response — unless the request is preempted
    /// first, in which case `fut` is dropped (cancelling the upstream call)
    /// and a `Preempted` response is returned.
    pub async fn guard<F>(self, fut: F) -> Response
    where
        F: Future<Output = Response>,
    {
        let Some(token) = self.0 else {
            return fut.await;
        };
        tokio::select! {
            resp = fut => resp,
            () = token.cancelled() => SchedulerError::Preempted.into_response(),
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/middleware/scheduler/extract/tests.rs"]
mod tests;
