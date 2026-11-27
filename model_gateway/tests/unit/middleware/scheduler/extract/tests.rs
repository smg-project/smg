use std::future::pending;

use axum::http::StatusCode;

use super::*;

#[tokio::test]
async fn guard_returns_inner_response_when_not_cancelled() {
    let guard = PreemptionGuard(Some(CancellationToken::new()));
    let resp = guard.guard(async { StatusCode::OK.into_response() }).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn without_an_admission_token_the_inner_response_passes_through() {
    // Legacy mode: the extractor yields no token, and the guard runs the
    // work as it is; it must never short-circuit to Preempted.
    let guard = PreemptionGuard::default();
    let resp = guard.guard(async { StatusCode::OK.into_response() }).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn guard_returns_preempted_when_token_cancelled() {
    let token = CancellationToken::new();
    token.cancel();
    let guard = PreemptionGuard(Some(token));
    // The work future never completes, so only the cancel arm can win.
    let resp = guard.guard(pending::<Response>()).await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn from_extensions_reads_the_admission_token_or_holds_none() {
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
