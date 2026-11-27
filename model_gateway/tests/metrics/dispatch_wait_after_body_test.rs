//! The dispatch wait without the upload of the request body, through the
//! whole gateway app: a chat request whose body comes in two parts with a
//! pause between them has the pause in `smg_router_dispatch_wait_seconds`,
//! which counts from the request's entry, and not in
//! `smg_router_dispatch_wait_after_body_seconds`, which is that wait less
//! what the read of the body waited for the client. A wait for an admission
//! permit is in both, whoever reads the body first: the route, behind that
//! wait, or the WASM request hook, ahead of it, where a module is attached.
//!
//! Each case answers chat requests, and the clean `200` row of that route in
//! the response counter carries no label a test could own. The stream cut
//! test reads exact differences of the row, so the cases here run under its
//! `serial` key, never beside it.

use std::{convert::Infallible, str::FromStr, sync::Arc, time::Duration};

use axum::{
    body::{Body, Bytes},
    http::{header::CONTENT_TYPE, Request},
};
use futures::{stream, StreamExt};
use http_body_util::BodyExt;
use metrics_exporter_prometheus::PrometheusHandle;
use serde_json::json;
use serial_test::serial;
use smg::{
    app_context::AppContext,
    wasm::module::{MiddlewareAttachPoint, WasmModule, WasmModuleAttachPoint, WasmModuleMeta},
};
use smg_test_harness::grpc_harness::grpc_router_config;
use tower::ServiceExt;
use uuid::Uuid;
use wasm_encoder::{Component, Module, ModuleSection, TypeSection};

use super::stream_cut_response_test::{gateway_app, gateway_app_over, MockEngine};
use crate::metrics::recorder;

/// A model of the slow upload's own: no other test of the binary records a
/// wait under it.
const MODEL: &str = "dispatch-wait-after-body-test-model";
/// The models of the request that waits for a permit: with a module at the
/// request hook, and without one.
const HOOKED_MODEL: &str = "dispatch-wait-with-a-request-hook-model";
const UNHOOKED_MODEL: &str = "dispatch-wait-without-a-request-hook-model";
/// How long the client takes between the two parts of the body: long against
/// what the gateway does for the request once it has the body.
const PAUSE: Duration = Duration::from_millis(1500);
/// How long the test keeps the gateway's one admission permit from a request
/// that waits for it.
const HELD: Duration = Duration::from_millis(1000);
const DISPATCH_WAIT: &str = "smg_router_dispatch_wait_seconds";
const DISPATCH_WAIT_AFTER_BODY: &str = "smg_router_dispatch_wait_after_body_seconds";
const ADMISSION_QUEUE_DEPTH: &str = "smg_admission_queue_depth";

/// The `field` sample (`sum` or `count`) of `family` under `model`.
fn sample<T: FromStr>(rendered: &str, family: &str, field: &str, model: &str) -> Option<T> {
    let prefix = format!("{family}_{field}{{");
    rendered
        .lines()
        .find(|line| line.starts_with(&prefix) && line.contains(model))
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
}

/// The seconds and the observations `family` holds under `model`.
#[expect(
    clippy::expect_used,
    reason = "test helper: a family with no series for the answer fails the test"
)]
fn waited(rendered: &str, family: &str, model: &str) -> (f64, u64) {
    let seconds = sample(rendered, family, "sum", model).expect("the family's sum for the answer");
    let observations =
        sample(rendered, family, "count", model).expect("the family's count for the answer");
    (seconds, observations)
}

/// The requests that wait in the admission queue, as `rendered` reads the
/// gauge: no test of this binary but the one of the held permit queues one.
fn queued(rendered: &str) -> Option<f64> {
    rendered
        .lines()
        .find_map(|line| line.strip_prefix(ADMISSION_QUEUE_DEPTH)?.strip_prefix(' '))
        .and_then(|value| value.parse().ok())
}

/// The body of a chat request under `model`.
fn chat_of(model: &str) -> Bytes {
    Bytes::from(
        json!({
            "model": model,
            "messages": [{"role": "user", "content": "count to twenty"}],
            "max_tokens": 16
        })
        .to_string(),
    )
}

/// `body` as one collected chat request through `app`, read to its end: the
/// status.
#[expect(clippy::expect_used, reason = "test helper")]
async fn answered(app: axum::Router, body: Body) -> u16 {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    let status = response.status().as_u16();
    tokio::time::timeout(Duration::from_secs(30), response.into_body().collect())
        .await
        .expect("the answer ends within its budget")
        .expect("the body");
    status
}

/// One collected chat request through `app`, its body in two parts with
/// `PAUSE` between them, read to its end: the status.
async fn chat_with_a_slow_body(app: &axum::Router) -> u16 {
    let mut head = chat_of(MODEL);
    let tail = head.split_off(head.len() / 2);
    let parts =
        stream::iter([(Duration::ZERO, head), (PAUSE, tail)]).then(|(pause, part)| async move {
            tokio::time::sleep(pause).await;
            Ok::<_, Infallible>(part)
        });
    answered(app.clone(), Body::from_stream(parts)).await
}

/// Attaches to the module manager of `context` one module at the request
/// hook whose component is valid but exports nothing: the hook walks its
/// whole path, the read of the body with it, and changes nothing of the
/// request.
#[expect(clippy::expect_used, reason = "test helper")]
fn attach_an_inert_module(context: &AppContext) {
    let mut module = Module::new();
    module.section(&TypeSection::new());
    let mut component = Component::new();
    component.section(&ModuleSection(&module));
    let bytes = component.as_slice().to_vec();
    context
        .wasm_manager
        .as_ref()
        .expect("a configuration with wasm enabled gives the context a module manager")
        .register_module_internal(WasmModule {
            module_uuid: Uuid::now_v7(),
            module_meta: WasmModuleMeta {
                name: "inert".to_string(),
                file_path: String::new(),
                sha256_hash: [0; 32],
                size_bytes: bytes.len() as u64,
                created_at: 0,
                last_accessed_at: 0,
                access_count: 0,
                attach_points: vec![WasmModuleAttachPoint::Middleware(
                    MiddlewareAttachPoint::OnRequest,
                )],
                wasm_bytes: Arc::new(bytes),
            },
        })
        .expect("the inert module registers");
}

/// One chat request under `model` through an app with ONE admission permit,
/// which the test keeps for [`HELD`] from where the request waits for it: the
/// seconds the answer has in the dispatch wait and in the wait without the
/// upload. With `hooked`, an inert module stands at the request
/// hook; the wasm layer is in the app either way.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper: the request's task is awaited before the helper returns"
)]
async fn waits_behind_a_held_permit(
    handle: &PrometheusHandle,
    model: &str,
    hooked: bool,
) -> (f64, f64) {
    let engine = MockEngine::start(model).await;
    let config = grpc_router_config()
        .enable_wasm(true)
        .max_concurrent_requests(1);
    let (app, context) = gateway_app_over(config, engine.port, model).await;
    if hooked {
        attach_an_inert_module(&context);
    }
    let permits = context
        .rate_limiter
        .clone()
        .expect("a bound on the requests in flight gives the context its permits");
    permits
        .try_acquire(1.0)
        .expect("the one permit is free ahead of the request");

    let answer = tokio::spawn(answered(app, Body::from(chat_of(model))));
    // The request waits for the permit where the queue's gauge reads it; an
    // answer ahead of that is one no admission held back.
    let asked = tokio::time::Instant::now();
    while !queued(&handle.render()).is_some_and(|depth| depth >= 1.0) {
        assert!(
            !answer.is_finished(),
            "{model}: the request was answered without a wait for the permit"
        );
        assert!(
            asked.elapsed() < Duration::from_secs(30),
            "{model}: the request neither waits for the permit nor is answered"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    tokio::time::sleep(HELD).await;
    permits.return_tokens(1.0);
    assert_eq!(answer.await.expect("the request's task"), 200);

    let rendered = handle.render();
    let (wait, waits) = waited(&rendered, DISPATCH_WAIT, model);
    let (after_body, after_bodies) = waited(&rendered, DISPATCH_WAIT_AFTER_BODY, model);
    assert_eq!(
        (waits, after_bodies),
        (1, 1),
        "{model}: the one answer is observed once in each family"
    );
    engine.stop().await;
    (wait, after_body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial(chat_response_row)]
async fn a_slow_upload_is_in_the_dispatch_wait_and_not_in_the_wait_after_the_body() {
    // The recorder first: the test may be the first of its process to ask for
    // it, and what a request observes ahead of the install is kept nowhere.
    let handle = recorder();
    let engine = MockEngine::start(MODEL).await;
    let app = gateway_app(engine.port, MODEL).await;

    assert_eq!(chat_with_a_slow_body(&app).await, 200);

    let rendered = handle.render();
    let (wait, waits) = waited(&rendered, DISPATCH_WAIT, MODEL);
    let (after_body, after_bodies) = waited(&rendered, DISPATCH_WAIT_AFTER_BODY, MODEL);
    assert_eq!(
        (waits, after_bodies),
        (1, 1),
        "the one answer is observed once in each family"
    );
    let pause = PAUSE.as_secs_f64();
    assert!(
        wait >= pause,
        "the dispatch wait ({wait} s) counts from the request's entry and holds the upload's pause"
    );
    assert!(
        wait - after_body >= pause,
        "the wait without the upload ({after_body} s) leaves the upload's pause to \
         the dispatch wait ({wait} s)"
    );

    engine.stop().await;
}

/// The upload alone is taken off, whoever reads the body first, through the
/// real stack: the request's body is whole at its entry, so its read waits
/// for no client, and the request then waits [`HELD`] for the one admission
/// permit. That wait is in both families.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial(chat_response_row)]
async fn a_wait_for_a_permit_is_in_both_waits_with_no_module_and_behind_a_request_hook() {
    let handle = recorder();
    let held = HELD.as_secs_f64();

    // No module: the route reads the body, behind the wait for the permit.
    let (wait, after_body) = waits_behind_a_held_permit(&handle, UNHOOKED_MODEL, false).await;
    assert!(
        after_body >= held,
        "without a module the route reads the body behind admission, and the wait for the \
         permit is in the wait without the upload ({after_body} s) as it is in the dispatch \
         wait ({wait} s)"
    );

    // A module at the request hook: the hook reads the body ahead of the
    // layers inside it, the wait for the permit among them.
    let (wait, after_body) = waits_behind_a_held_permit(&handle, HOOKED_MODEL, true).await;
    assert!(
        after_body >= held,
        "with a module the request hook reads the body ahead of admission, and the wait for \
         the permit is in the wait without the upload ({after_body} s) as it is in the \
         dispatch wait ({wait} s)"
    );
}
