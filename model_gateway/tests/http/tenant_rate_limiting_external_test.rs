//! Integration tests for tenant rate limiting on the external routers'
//! gateway side (`routers/external.rs`'s `ExternalRouterAdapter`), against
//! a mock HTTP worker registered as the provider. The adapter reserves once
//! per logical request before the provider router dispatches and resolves
//! the reservation once on the router's answer, the way the regular HTTP
//! router's typed route does (`tenant_rate_limiting_http_test.rs`, whose
//! worker, limiter and request helpers these tests share).

#![expect(
    clippy::unwrap_used,
    reason = "test helpers - panicking on failure is intentional"
)]

use std::sync::Arc;

use axum::http::{header::RETRY_AFTER, StatusCode};
use common::test_app::{register_external_worker, register_external_worker_with_card};
use openai_protocol::{
    chat::ChatCompletionRequest, interactions::InteractionsRequest, messages::CreateMessageRequest,
};
use serde_json::{json, Value};
use smg::{
    routers::{external::ExternalRouterAdapter, RouterTrait},
    worker::ModelCard,
};
use smg_external_router::{known, ExternalRouterSpec};

use super::{
    tenant_rate_limiting_http_test::{
        body_of, limited_context, mock_worker, request, router, tenant, usage, Endpoint, Limiter,
        MockWorker, RecordingBackend, ESTIMATED_INPUT_TOKENS, MODEL, PROMPT_TEXT,
    },
    tenant_rate_limiting_pd_test::refusal,
};
use crate::common;

/// The router `spec` describes, over `worker` registered as its provider
/// (see [`limited_context`]).
async fn external_router(
    spec: ExternalRouterSpec,
    limiter: Limiter,
    worker: &MockWorker,
    retries: bool,
) -> (Box<dyn RouterTrait>, Option<tempfile::TempDir>) {
    let (ctx, dir) = limited_context(limiter, retries).await;
    register_external_worker(&ctx, &worker.url, Some(vec![MODEL]));
    (ExternalRouterAdapter::mount(spec, &ctx).await.unwrap(), dir)
}

/// An alias of [`MODEL`] on the provider's model card.
const ALIAS: &str = "the-model-by-another-name";

/// [`external_router`] with the worker's model carrying [`ALIAS`], retries
/// off.
async fn aliased_external_router(
    spec: ExternalRouterSpec,
    limiter: Limiter,
    worker: &MockWorker,
) -> (Box<dyn RouterTrait>, Option<tempfile::TempDir>) {
    let (ctx, dir) = limited_context(limiter, false).await;
    register_external_worker_with_card(&ctx, &worker.url, ModelCard::new(MODEL).with_alias(ALIAS));
    (ExternalRouterAdapter::mount(spec, &ctx).await.unwrap(), dir)
}

/// A buffered chat request naming `model`.
async fn chat_as(router: &dyn RouterTrait, model: &str) -> axum::response::Response {
    let body = request::<ChatCompletionRequest>(json!({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TEXT}],
    }));
    router.route_chat(None, &tenant(), body, model).await
}

// ---------------------------------------------------------------------------
// Admission.
// ---------------------------------------------------------------------------

/// A denial the tenant can wait out carries `Retry-After`, and never
/// reaches the provider. The first request reserves 3 of 6 and settles the
/// provider's 1 + 4, leaving 1, short of the second request's 3.
#[tokio::test]
async fn a_denial_the_tenant_can_wait_out_carries_retry_after() {
    let worker = mock_worker().await;
    worker.answer(Endpoint::Chat.usage_body(1, 4));
    let (router, _dir) = external_router(known::OPENAI, Limiter::Tokens(6), &worker, true).await;

    let first = Endpoint::Chat.call(router.as_ref(), false).await;
    assert_eq!(first.status(), StatusCode::OK);

    let second = Endpoint::Chat.call(router.as_ref(), false).await;
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        second.headers().get(RETRY_AFTER).is_some(),
        "a finite-wait denial carries Retry-After"
    );
    assert_eq!(worker.requests(), 1, "a denied request is not dispatched");
}

/// A request whose estimate exceeds the tenant's whole budget is denied
/// without `Retry-After` on every provider route that charges a tenant:
/// chat and Responses on the OpenAI-compatible router, Messages on the
/// Anthropic router, Interactions on the Gemini router.
#[tokio::test]
async fn an_impossible_request_is_denied_on_every_provider_route() {
    let tenant = tenant();
    for retries in [false, true] {
        let worker = mock_worker().await;
        let (openai, _dir) =
            external_router(known::OPENAI, Limiter::Tokens(1), &worker, retries).await;
        let (anthropic, _dir) =
            external_router(known::ANTHROPIC, Limiter::Tokens(1), &worker, retries).await;
        let (gemini, _dir) =
            external_router(known::GEMINI, Limiter::Tokens(1), &worker, retries).await;

        let messages = request::<CreateMessageRequest>(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT_TEXT}],
            "max_tokens": 32,
        }));
        let interactions = request::<InteractionsRequest>(json!({
            "model": MODEL, "input": PROMPT_TEXT, "store": false,
        }));
        let refusals = [
            ("chat", Endpoint::Chat.call(openai.as_ref(), false).await),
            (
                "responses",
                Endpoint::Responses.call(openai.as_ref(), false).await,
            ),
            (
                "messages",
                anthropic
                    .route_messages(None, &tenant, messages, MODEL)
                    .await,
            ),
            (
                "interactions",
                gemini
                    .route_interactions(None, &tenant, interactions, Some(MODEL))
                    .await,
            ),
        ];
        for (route, response) in refusals {
            let case = format!("{route} retries={retries}");
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS, "{case}");
            assert!(
                response.headers().get(RETRY_AFTER).is_none(),
                "{case}: an impossible request's denial carries no Retry-After"
            );
        }
        assert_eq!(worker.requests(), 0, "retries={retries}");
    }
}

/// The external router's refusal is the regular router's, byte for byte:
/// the same status, headers and body for the same tenant in the same state.
#[tokio::test]
async fn the_refusal_is_the_regular_routers_answer() {
    let worker = mock_worker().await;
    worker.answer(Endpoint::Chat.usage_body(1, 4));
    let (regular, _dir) = router(Limiter::Tokens(6), &worker, true).await;
    let provider = mock_worker().await;
    provider.answer(Endpoint::Chat.usage_body(1, 4));
    let (external, _dir) =
        external_router(known::OPENAI, Limiter::Tokens(6), &provider, true).await;

    assert_eq!(
        Endpoint::Chat.call(&regular, false).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        Endpoint::Chat.call(external.as_ref(), false).await.status(),
        StatusCode::OK
    );
    let expected = refusal(Endpoint::Chat.call(&regular, false).await).await;
    let refused = refusal(Endpoint::Chat.call(external.as_ref(), false).await).await;
    assert_eq!(expected.0, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(refused, expected);
}

// ---------------------------------------------------------------------------
// Settlement.
// ---------------------------------------------------------------------------

/// A buffered success settles the counts the provider's body reports,
/// once, over the request's one reservation of its estimate; the client
/// receives the provider's body unchanged. On both dispatch paths.
#[tokio::test]
async fn a_buffered_success_settles_the_usage_its_body_reports() {
    for retries in [false, true] {
        let worker = mock_worker().await;
        let answer = Endpoint::Chat.usage_body(11, 7);
        worker.answer(answer.clone());
        let backend = Arc::new(RecordingBackend::default());
        let (router, _dir) = external_router(
            known::OPENAI,
            Limiter::Recording(backend.clone()),
            &worker,
            retries,
        )
        .await;

        let response = Endpoint::Chat.call(router.as_ref(), false).await;
        let case = format!("retries={retries}");
        assert_eq!(response.status(), StatusCode::OK, "{case}");
        assert_eq!(
            serde_json::from_str::<Value>(&body_of(response).await).unwrap(),
            answer,
            "{case}: the client gets the provider's body"
        );
        assert_eq!(backend.estimates(), vec![ESTIMATED_INPUT_TOKENS], "{case}");
        assert_eq!(backend.settlements(), vec![usage(11, 7)], "{case}");
        assert_eq!((backend.closes(), backend.abandons()), (0, 0), "{case}");
    }
}

/// A streamed answer is relayed as the provider sent it and never parsed:
/// it settles nothing, and its reservation resolves when the body drops,
/// keeping its estimate.
#[tokio::test]
async fn a_streamed_answer_resolves_when_its_body_drops() {
    let worker = mock_worker().await;
    worker.answer(Endpoint::Chat.usage_body(1000, 1000));
    let backend = Arc::new(RecordingBackend::default());
    let (router, _dir) = external_router(
        known::OPENAI,
        Limiter::Recording(backend.clone()),
        &worker,
        true,
    )
    .await;

    let response = Endpoint::Chat.call(router.as_ref(), true).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        (backend.closes(), backend.abandons()),
        (0, 0),
        "the body holds the reservation while it lives"
    );

    let body = body_of(response).await;
    assert!(
        body.contains("1000") && body.ends_with("data: [DONE]\n\n"),
        "the stream is relayed whole: {body}"
    );
    backend.await_abandons(1).await;
    assert_eq!(backend.estimates(), vec![ESTIMATED_INPUT_TOKENS]);
    assert!(backend.settlements().is_empty());
    assert_eq!(backend.closes(), 0);
}

// ---------------------------------------------------------------------------
// Retries and failures.
// ---------------------------------------------------------------------------

/// A request the provider router retries reserves once, before its first
/// attempt, and its final answer settles that one reservation.
#[tokio::test]
async fn a_retried_request_reserves_once_and_settles_its_final_answer() {
    let worker = mock_worker().await;
    worker.answer(Endpoint::Chat.usage_body(11, 7));
    worker.fail_next(1);
    let backend = Arc::new(RecordingBackend::default());
    let (router, _dir) = external_router(
        known::OPENAI,
        Limiter::Recording(backend.clone()),
        &worker,
        true,
    )
    .await;

    let response = Endpoint::Chat.call(router.as_ref(), false).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(worker.requests(), 2, "the 503 was retried");
    assert_eq!(backend.estimates(), vec![ESTIMATED_INPUT_TOKENS]);
    assert_eq!(backend.settlements(), vec![usage(11, 7)]);
    assert_eq!((backend.closes(), backend.abandons()), (0, 0));
}

/// A request whose every attempt fails closes its one reservation at the
/// estimate, on both dispatch paths.
#[tokio::test]
async fn a_failed_request_closes_its_reservation_at_the_estimate() {
    for retries in [false, true] {
        let worker = mock_worker().await;
        worker.fail_next(usize::MAX);
        let backend = Arc::new(RecordingBackend::default());
        let (router, _dir) = external_router(
            known::OPENAI,
            Limiter::Recording(backend.clone()),
            &worker,
            retries,
        )
        .await;

        let response = Endpoint::Chat.call(router.as_ref(), false).await;
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "retries={retries}"
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
    let worker = mock_worker().await;
    worker.answer(Endpoint::Chat.usage_body(1000, 1000));
    let (router, _dir) = external_router(known::OPENAI, Limiter::Off, &worker, true).await;

    for stream in [false, true] {
        for _ in 0..3 {
            let response = Endpoint::Chat.call(router.as_ref(), stream).await;
            assert_eq!(response.status(), StatusCode::OK, "stream={stream}");
            body_of(response).await;
        }
    }
}

/// A request naming an alias of the provider's model is charged under the
/// model's own name, as the regular router charges its aliases: the
/// reservation carries the canonical name, so the limiter's rule for that
/// model holds the alias too, and the refusal is counted under it.
#[tokio::test]
async fn an_alias_request_is_charged_under_its_canonical_model() {
    let backend = Arc::new(RecordingBackend::default());
    let worker = mock_worker().await;
    worker.answer(Endpoint::Chat.usage_body(1, 4));
    let (router, _dir) =
        aliased_external_router(known::OPENAI, Limiter::Recording(backend.clone()), &worker).await;
    let response = chat_as(router.as_ref(), ALIAS).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(backend.models(), vec![Some(MODEL.to_owned())]);

    let worker = mock_worker().await;
    let (router, _dir) =
        aliased_external_router(known::OPENAI, Limiter::ModelRule(MODEL, 1), &worker).await;
    let response = chat_as(router.as_ref(), ALIAS).await;
    assert_eq!(
        response.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "the canonical model's rule holds its alias"
    );
    assert_eq!(worker.requests(), 0, "a denied request is not dispatched");
}
