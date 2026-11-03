//! Integration tests for tenant rate limiting on the disaggregated HTTP
//! router (`routers/http/pd_router.rs`'s `execute_dual_dispatch`), against
//! a prefill and a decode mock HTTP worker. The router reserves once per
//! logical request before its first dispatch and resolves the reservation
//! once on the answer its dispatch paths converge on, the way the regular
//! HTTP router's typed route does (`tenant_rate_limiting_http_test.rs`,
//! whose worker, limiter and request helpers these tests share).

#![expect(
    clippy::unwrap_used,
    reason = "test helpers - panicking on failure is intentional"
)]

use std::sync::Arc;

use axum::{
    body::to_bytes,
    http::{header::RETRY_AFTER, HeaderMap, StatusCode},
    response::Response,
};
use openai_protocol::{
    completion::CompletionRequest, rerank::RerankRequest, worker::HealthCheckConfig,
};
use serde_json::{json, Value};
use smg::{
    config::PolicyConfig,
    routers::{http::pd_router::PDRouter, RouterTrait},
    worker::{BasicWorkerBuilder, ModelCard, WorkerType},
};

use super::tenant_rate_limiting_http_test::{
    body_of, limited_context_with_policy, mock_worker, request, router, tenant, usage, Endpoint,
    Limiter, MockWorker, RecordingBackend, ESTIMATED_INPUT_TOKENS, MODEL, PROMPT_TEXT,
};

/// The endpoints the disaggregated router serves and reserves on.
const ENDPOINTS: [Endpoint; 5] = [
    Endpoint::Generate,
    Endpoint::Chat,
    Endpoint::Completion,
    Endpoint::Messages,
    Endpoint::Responses,
];

/// The port a loopback URL names.
fn port_of(url: &str) -> u16 {
    url.rsplit(':').next().unwrap().parse().unwrap()
}

/// The disaggregated router over a prefill and a decode mock worker, routing
/// at random (see [`limited_context_with_policy`]).
async fn pd_router(
    limiter: Limiter,
    prefill: &MockWorker,
    decode: &MockWorker,
    retries: bool,
) -> (PDRouter, Option<tempfile::TempDir>) {
    pd_router_with_policy(limiter, prefill, decode, retries, PolicyConfig::Random).await
}

/// [`pd_router`] routing by `policy`.
async fn pd_router_with_policy(
    limiter: Limiter,
    prefill: &MockWorker,
    decode: &MockWorker,
    retries: bool,
    policy: PolicyConfig,
) -> (PDRouter, Option<tempfile::TempDir>) {
    let (ctx, dir) = limited_context_with_policy(limiter, retries, policy).await;
    let health = HealthCheckConfig {
        disable_health_check: true,
        ..Default::default()
    };
    ctx.worker_registry
        .register(Arc::new(
            BasicWorkerBuilder::new(&prefill.url)
                .worker_type(WorkerType::Prefill)
                .bootstrap_port(Some(port_of(&prefill.url)))
                .model(ModelCard::new(MODEL))
                .health_config(health.clone())
                .build(),
        ))
        .unwrap();
    ctx.worker_registry
        .register(Arc::new(
            BasicWorkerBuilder::new(&decode.url)
                .worker_type(WorkerType::Decode)
                .model(ModelCard::new(MODEL))
                .health_config(health)
                .build(),
        ))
        .unwrap();
    (PDRouter::new(&ctx).await.unwrap(), dir)
}

/// A refusal as the client reads it: its status, every header, its bytes.
pub(super) async fn refusal(response: Response) -> (StatusCode, HeaderMap, Vec<u8>) {
    let (parts, body) = response.into_parts();
    let bytes = to_bytes(body, 1 << 20).await.unwrap();
    (parts.status, parts.headers, bytes.to_vec())
}

// ---------------------------------------------------------------------------
// Admission.
// ---------------------------------------------------------------------------

/// A denial the tenant can wait out carries `Retry-After`, and reaches
/// neither side. The first request reserves 3 of 6 and settles the decode
/// side's 1 + 4, leaving 1, short of the second request's 3.
#[tokio::test]
async fn a_denial_the_tenant_can_wait_out_carries_retry_after() {
    let (prefill, decode) = (mock_worker().await, mock_worker().await);
    prefill.answer(Endpoint::Generate.usage_body(1, 4));
    decode.answer(Endpoint::Generate.usage_body(1, 4));
    let (router, _dir) = pd_router(Limiter::Tokens(6), &prefill, &decode, true).await;

    let first = Endpoint::Generate.call(&router, false).await;
    assert_eq!(first.status(), StatusCode::OK);

    let second = Endpoint::Generate.call(&router, false).await;
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        second.headers().get(RETRY_AFTER).is_some(),
        "a finite-wait denial carries Retry-After"
    );
    assert_eq!(
        (prefill.requests(), decode.requests()),
        (1, 1),
        "a denied request is dispatched to neither side"
    );
}

/// A request whose estimate exceeds the tenant's whole budget is denied
/// without `Retry-After`, whichever path dispatch takes.
#[tokio::test]
async fn an_impossible_request_is_denied_without_retry_after() {
    for retries in [false, true] {
        let (prefill, decode) = (mock_worker().await, mock_worker().await);
        let (router, _dir) = pd_router(Limiter::Tokens(1), &prefill, &decode, retries).await;

        let response = Endpoint::Chat.call(&router, false).await;
        assert_eq!(
            response.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "retries={retries}"
        );
        assert!(
            response.headers().get(RETRY_AFTER).is_none(),
            "retries={retries}: an impossible request's denial carries no Retry-After"
        );
        assert_eq!(
            (prefill.requests(), decode.requests()),
            (0, 0),
            "retries={retries}"
        );
    }
}

/// The disaggregated router's refusal is the regular router's, byte for
/// byte: the same status, headers and body for the same tenant in the same
/// state.
#[tokio::test]
async fn the_refusal_is_the_regular_routers_answer() {
    let worker = mock_worker().await;
    worker.answer(Endpoint::Chat.usage_body(1, 4));
    let (regular, _dir) = router(Limiter::Tokens(6), &worker, true).await;
    let (prefill, decode) = (mock_worker().await, mock_worker().await);
    prefill.answer(Endpoint::Chat.usage_body(1, 4));
    decode.answer(Endpoint::Chat.usage_body(1, 4));
    let (disaggregated, _dir) = pd_router(Limiter::Tokens(6), &prefill, &decode, true).await;

    assert_eq!(
        Endpoint::Chat.call(&regular, false).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        Endpoint::Chat.call(&disaggregated, false).await.status(),
        StatusCode::OK
    );
    let expected = refusal(Endpoint::Chat.call(&regular, false).await).await;
    let refused = refusal(Endpoint::Chat.call(&disaggregated, false).await).await;
    assert_eq!(expected.0, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(refused, expected);
}

/// The estimate reads the whole request, whatever the routing policy reads:
/// a text-reading policy routes a completions batch on its first prompt
/// alone, and the tenant is charged for every prompt.
#[tokio::test]
async fn a_completions_batch_reserves_every_prompt_under_a_text_reading_policy() {
    let (prefill, decode) = (mock_worker().await, mock_worker().await);
    prefill.answer(Endpoint::Completion.usage_body(1, 1));
    decode.answer(Endpoint::Completion.usage_body(1, 1));
    let backend = Arc::new(RecordingBackend::default());
    let (router, _dir) = pd_router_with_policy(
        Limiter::Recording(backend.clone()),
        &prefill,
        &decode,
        true,
        PolicyConfig::cache_aware(0.5, 32, 1.0001, 60, 1 << 20),
    )
    .await;

    let batch = request::<CompletionRequest>(json!({
        "model": MODEL, "prompt": ["one two", "three four"],
    }));
    let response = router.route_completion(None, &tenant(), batch, MODEL).await;
    assert_eq!(response.status(), StatusCode::OK);
    // Four words over two prompts: ceil(4 * 1.3), not the first prompt's 3.
    assert_eq!(backend.estimates(), vec![6]);
}

// ---------------------------------------------------------------------------
// Settlement.
// ---------------------------------------------------------------------------

/// A buffered success settles the counts the decode side's body reports,
/// once, over the request's one reservation of its estimate; the client
/// receives the decode side's body unchanged. On both dispatch paths.
#[tokio::test]
async fn a_buffered_success_settles_the_usage_its_body_reports() {
    for retries in [false, true] {
        for endpoint in ENDPOINTS {
            let (prefill, decode) = (mock_worker().await, mock_worker().await);
            let answer = endpoint.usage_body(11, 7);
            prefill.answer(answer.clone());
            decode.answer(answer.clone());
            let backend = Arc::new(RecordingBackend::default());
            let (router, _dir) = pd_router(
                Limiter::Recording(backend.clone()),
                &prefill,
                &decode,
                retries,
            )
            .await;

            let response = endpoint.call(&router, false).await;
            let case = format!("{endpoint:?} retries={retries}");
            assert_eq!(response.status(), StatusCode::OK, "{case}");
            assert_eq!(
                serde_json::from_str::<Value>(&body_of(response).await).unwrap(),
                answer,
                "{case}: the client gets the decode side's body"
            );
            assert_eq!(backend.estimates(), vec![ESTIMATED_INPUT_TOKENS], "{case}");
            assert_eq!(
                backend.settlements(),
                vec![endpoint.settlement(11, 7)],
                "{case}"
            );
            assert_eq!((backend.closes(), backend.abandons()), (0, 0), "{case}");
        }
    }
}

/// A streamed answer is relayed as the decode side sent it and never
/// parsed: it settles nothing, and its reservation resolves when the body
/// drops, keeping its estimate.
#[tokio::test]
async fn a_streamed_answer_resolves_when_its_body_drops() {
    for endpoint in ENDPOINTS {
        let (prefill, decode) = (mock_worker().await, mock_worker().await);
        prefill.answer(endpoint.usage_body(1000, 1000));
        decode.answer(endpoint.usage_body(1000, 1000));
        let backend = Arc::new(RecordingBackend::default());
        let (router, _dir) =
            pd_router(Limiter::Recording(backend.clone()), &prefill, &decode, true).await;

        let response = endpoint.call(&router, true).await;
        assert_eq!(response.status(), StatusCode::OK, "{endpoint:?}");
        assert_eq!(
            (backend.closes(), backend.abandons()),
            (0, 0),
            "{endpoint:?}: the body holds the reservation while it lives"
        );

        let body = body_of(response).await;
        assert!(
            body.contains("1000") && body.ends_with("data: [DONE]\n\n"),
            "{endpoint:?}: the stream is relayed whole: {body}"
        );
        backend.await_abandons(1).await;
        assert_eq!(
            backend.estimates(),
            vec![ESTIMATED_INPUT_TOKENS],
            "{endpoint:?}"
        );
        assert!(backend.settlements().is_empty(), "{endpoint:?}");
        assert_eq!(backend.closes(), 0, "{endpoint:?}");
    }
}

// ---------------------------------------------------------------------------
// Retries and failures.
// ---------------------------------------------------------------------------

/// A request the router retries past a side's refusal reserves once,
/// before its first attempt, and its final answer settles that one
/// reservation.
#[tokio::test]
async fn a_retried_request_reserves_once_and_settles_its_final_answer() {
    let (prefill, decode) = (mock_worker().await, mock_worker().await);
    prefill.answer(Endpoint::Chat.usage_body(11, 7));
    decode.answer(Endpoint::Chat.usage_body(11, 7));
    prefill.fail_next(1);
    let backend = Arc::new(RecordingBackend::default());
    let (router, _dir) =
        pd_router(Limiter::Recording(backend.clone()), &prefill, &decode, true).await;

    let response = Endpoint::Chat.call(&router, false).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(prefill.requests(), 2, "the refused attempt was retried");
    assert_eq!(backend.estimates(), vec![ESTIMATED_INPUT_TOKENS]);
    assert_eq!(backend.settlements(), vec![usage(11, 7)]);
    assert_eq!((backend.closes(), backend.abandons()), (0, 0));
}

/// A request whose every attempt fails closes its one reservation at the
/// estimate, on both dispatch paths.
#[tokio::test]
async fn a_failed_request_closes_its_reservation_at_the_estimate() {
    for retries in [false, true] {
        let (prefill, decode) = (mock_worker().await, mock_worker().await);
        prefill.fail_next(usize::MAX);
        let backend = Arc::new(RecordingBackend::default());
        let (router, _dir) = pd_router(
            Limiter::Recording(backend.clone()),
            &prefill,
            &decode,
            retries,
        )
        .await;

        let response = Endpoint::Generate.call(&router, false).await;
        assert!(
            !response.status().is_success(),
            "retries={retries}: {}",
            response.status()
        );
        tokio::task::yield_now().await;
        assert_eq!(
            backend.estimates(),
            vec![ESTIMATED_INPUT_TOKENS],
            "retries={retries}"
        );
        assert!(backend.settlements().is_empty(), "retries={retries}");
        assert_eq!(
            (backend.closes(), backend.abandons()),
            (1, 0),
            "retries={retries}"
        );
    }
}

// ---------------------------------------------------------------------------
// Scope.
// ---------------------------------------------------------------------------

/// Without a tenant policy every request is served as before, streamed or
/// not.
#[tokio::test]
async fn without_a_tenant_policy_requests_are_served_as_before() {
    let (prefill, decode) = (mock_worker().await, mock_worker().await);
    prefill.answer(Endpoint::Chat.usage_body(1000, 1000));
    decode.answer(Endpoint::Chat.usage_body(1000, 1000));
    let (router, _dir) = pd_router(Limiter::Off, &prefill, &decode, true).await;

    for stream in [false, true] {
        for _ in 0..3 {
            let response = Endpoint::Chat.call(&router, stream).await;
            assert_eq!(response.status(), StatusCode::OK, "stream={stream}");
            body_of(response).await;
        }
    }
}

/// The rerank route dispatches without reserving, as on the regular router.
#[tokio::test]
async fn an_opted_out_endpoint_never_reserves() {
    let (prefill, decode) = (mock_worker().await, mock_worker().await);
    let answer = json!([{"score": 0.5, "document": "d", "index": 0}]);
    prefill.answer(answer.clone());
    decode.answer(answer);
    let backend = Arc::new(RecordingBackend::default());
    let (router, _dir) =
        pd_router(Limiter::Recording(backend.clone()), &prefill, &decode, true).await;

    let rerank = request::<RerankRequest>(json!({
        "model": MODEL, "query": PROMPT_TEXT, "documents": ["d"],
    }));
    let response = router.route_rerank(None, &tenant(), rerank, MODEL).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        (prefill.requests(), decode.requests()),
        (1, 1),
        "the request was dispatched"
    );
    assert!(backend.estimates().is_empty(), "nothing was reserved");
}
