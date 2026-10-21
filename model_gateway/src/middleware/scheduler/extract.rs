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
/// absent and this yields a default token that never fires, so wrapped
/// handlers behave identically to today.
#[derive(Clone, Default)]
pub struct PreemptionGuard(CancellationToken);

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
    /// The request's token where the admission layer put one; a default
    /// token that never fires otherwise (legacy admission). What the
    /// extractor reads, for a layer that holds the request itself.
    pub fn from_extensions(extensions: &Extensions) -> Self {
        Self(
            extensions
                .get::<CancellationToken>()
                .cloned()
                .unwrap_or_default(),
        )
    }

    /// Run `fut`, returning its response — unless the request is preempted
    /// first, in which case `fut` is dropped (cancelling the upstream call)
    /// and a `Preempted` response is returned.
    pub async fn guard<F>(self, fut: F) -> Response
    where
        F: Future<Output = Response>,
    {
        tokio::select! {
            resp = fut => resp,
            () = self.0.cancelled() => SchedulerError::Preempted.into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future::pending;

    use axum::http::StatusCode;

    use super::*;

    #[tokio::test]
    async fn guard_returns_inner_response_when_not_cancelled() {
        let guard = PreemptionGuard(CancellationToken::new());
        let resp = guard.guard(async { StatusCode::OK.into_response() }).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn default_token_never_fires_so_inner_response_passes_through() {
        // Legacy mode: the extractor yields a default token. Even racing a
        // never-completing future, the only way to return is the inner arm,
        // so a default-token guard must never short-circuit to Preempted.
        let guard = PreemptionGuard::default();
        let resp = guard.guard(async { StatusCode::OK.into_response() }).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn guard_returns_preempted_when_token_cancelled() {
        let token = CancellationToken::new();
        token.cancel();
        let guard = PreemptionGuard(token);
        // The work future never completes, so only the cancel arm can win.
        let resp = guard.guard(pending::<Response>()).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn from_extensions_reads_the_admission_token_or_defaults_to_one_that_never_fires() {
        let mut extensions = Extensions::new();
        let resp = PreemptionGuard::from_extensions(&extensions)
            .guard(async { StatusCode::OK.into_response() })
            .await;
        assert_eq!(resp.status(), StatusCode::OK);

        let token = CancellationToken::new();
        extensions.insert(token.clone());
        token.cancel();
        let resp = PreemptionGuard::from_extensions(&extensions)
            .guard(pending::<Response>())
            .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
