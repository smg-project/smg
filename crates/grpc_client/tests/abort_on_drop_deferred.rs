//! Behavioral tests for `AbortOnDropStream`'s deferred-abort mode, driven
//! through the real `VllmEngineClient` against an in-process mock engine.
//!
//! The deferral contract: with `defer_abort_until_first_item`, dropping the
//! stream before the backend produced any response holds the abort until the
//! first response arrives (the disaggregated decode side uses this so a client
//! disconnect can never tear down a request mid-KV-handoff); dropping after
//! the first response — or without the deferral — aborts promptly, and
//! `mark_completed` suppresses the abort entirely. The dropped stream itself
//! is released by the backend's own end of it after the abort, within a
//! bound, not by an immediate reset.

use std::time::Duration;

use futures::StreamExt;
use smg_grpc_client::{vllm_proto as proto, VllmEngineClient};

mod common;
use common::{generate_request, spawn_mock, wait_for_abort};

fn chunk() -> proto::GenerateResponse {
    proto::GenerateResponse::default()
}

#[tokio::test]
async fn default_mode_aborts_immediately_on_drop() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();

    let stream = client.generate(generate_request("imm-1")).await.unwrap();
    drop(stream);

    assert_eq!(wait_for_abort(&state).await, ["imm-1"]);
}

#[tokio::test]
async fn deferred_mode_holds_abort_until_first_item() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();

    let stream = client
        .generate(generate_request("def-1"))
        .await
        .unwrap()
        .defer_abort_until_first_item();
    drop(stream);

    // No abort while the backend hasn't produced anything.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        state.aborted_ids().is_empty(),
        "abort must be held until the first response"
    );

    // The backend produces its first response (handoff complete) — the
    // deferred abort now fires.
    let tx = state.lock_tx().clone().unwrap();
    tx.send(Ok(chunk())).await.unwrap();
    assert_eq!(wait_for_abort(&state).await, ["def-1"]);
}

#[tokio::test]
async fn deferred_mode_aborts_immediately_after_first_item_was_consumed() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();

    let mut stream = client
        .generate(generate_request("def-2"))
        .await
        .unwrap()
        .defer_abort_until_first_item();

    let tx = state.lock_tx().clone().unwrap();
    tx.send(Ok(chunk())).await.unwrap();
    let first = stream.next().await;
    assert!(matches!(first, Some(Ok(_))));

    // The protected window is over — dropping now aborts promptly.
    drop(stream);
    assert_eq!(wait_for_abort(&state).await, ["def-2"]);
}

#[tokio::test]
async fn mark_completed_suppresses_abort_in_deferred_mode() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();

    let stream = client
        .generate(generate_request("done-1"))
        .await
        .unwrap()
        .defer_abort_until_first_item();
    stream.mark_completed();
    drop(stream);

    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(state.aborted_ids().is_empty());
}

#[tokio::test]
async fn flush_cache_forwards_timeout_and_result() {
    let (endpoint, _) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let result = client.flush_cache(1.5).await.unwrap();
    assert!(result.success);
    assert_eq!(result.message, "reset result");
    assert!(!client.flush_cache(0.0).await.unwrap().success);
}

/// A dropped stream is released by the backend's own end of it, not by a
/// reset: after the abort it stays open on the wire until the backend ends
/// it (a chunk the backend sends meanwhile still goes through; a reset would
/// have closed the response body's channel on the backend's side).
#[tokio::test]
async fn a_dropped_stream_is_released_by_the_backends_own_end_not_a_reset() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();

    let stream = client.generate(generate_request("rel-1")).await.unwrap();
    drop(stream);
    assert_eq!(wait_for_abort(&state).await, ["rel-1"]);

    // Time for a reset, had one been sent, to reach the backend and close
    // its response body's channel.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let tx = state.lock_tx().clone().unwrap();
    tx.send(Ok(chunk()))
        .await
        .expect("the stream stays open until the backend ends it");
    // The backend's end of the stream.
    drop(tx);
}

/// A backend that never ends a dropped stream holds it only for the release
/// bound; past it the stream is released with a reset after all.
#[tokio::test]
async fn a_backend_that_never_ends_a_dropped_stream_is_reset_after_the_bound() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();

    let stream = client
        .generate(generate_request("rel-2"))
        .await
        .unwrap()
        .with_release_drain_bound(Duration::from_millis(300));
    drop(stream);
    assert_eq!(wait_for_abort(&state).await, ["rel-2"]);

    let tx = state.lock_tx().clone().unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    tx.send(Ok(chunk()))
        .await
        .expect("still open inside the bound");
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert!(
        tx.send(Ok(chunk())).await.is_err(),
        "past the bound the stream is released with a reset"
    );
}
