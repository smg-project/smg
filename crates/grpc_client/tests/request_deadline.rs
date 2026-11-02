//! Behavioral tests for `AbortOnDropStream::with_deadline`, driven through
//! the real `VllmEngineClient` against an in-process mock engine.
//!
//! The deadline's contract: a stream polled at or past its deadline ends with
//! the deadline's status (and nothing after it), the abort goes to the
//! backend at once rather than when the reader lets go of the stream, and a
//! stream that ends before the deadline is unchanged by it.

use std::time::{Duration, Instant};

use futures::StreamExt;
use smg_grpc_client::{is_request_timeout, vllm_proto as proto, RequestDeadline, VllmEngineClient};

mod common;
use common::{generate_request, spawn_mock, wait_for_abort, GenerateTx, MockState};

/// The sender of the mock's open generate stream, taken out of the mock: the
/// test alone holds the engine's side of the stream.
fn take_tx(state: &MockState) -> Option<GenerateTx> {
    state.lock_tx().take()
}

/// The stream's first message is the one message a 300 ms deadline cuts in
/// an hour of silence; the message's shape is the deadline's own; nothing
/// follows it.
const DEADLINE: Duration = Duration::from_millis(300);

#[tokio::test]
async fn a_silent_stream_ends_at_the_deadline_with_its_status_and_the_abort_goes_out_at_once() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let started = Instant::now();
    let mut stream = client
        .generate(generate_request("silent-1"))
        .await
        .unwrap()
        .with_deadline(RequestDeadline::after(DEADLINE));
    // The engine holds the stream open and says nothing.
    let _held_open = take_tx(&state).expect("the mock opened a stream");

    let first = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("the deadline ends the wait");
    let elapsed = started.elapsed();
    let status = first
        .expect("a message, not the end")
        .expect_err("the deadline's status");
    assert!(
        elapsed >= DEADLINE && elapsed < DEADLINE + Duration::from_secs(5),
        "ended after {elapsed:?}"
    );
    assert!(is_request_timeout(&status), "{status:?}");
    assert!(
        status.message().contains("request timed out after 300ms"),
        "{status:?}"
    );

    // The abort reaches the engine while the reader still holds the stream.
    assert_eq!(wait_for_abort(&state).await, ["silent-1"]);
    // Nothing follows the deadline's status.
    assert!(stream.next().await.is_none());
    drop(stream);
    // The drop sends no second abort.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(state.aborted_ids(), ["silent-1"]);
}

#[tokio::test]
async fn a_stream_that_goes_silent_after_its_first_message_is_cut_at_the_deadline_too() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let mut stream = client
        .generate(generate_request("after-first-1"))
        .await
        .unwrap()
        .with_deadline(RequestDeadline::after(DEADLINE));
    let tx = take_tx(&state).expect("the mock opened a stream");
    tx.send(Ok(proto::GenerateResponse::default()))
        .await
        .unwrap();

    // The message that came in time passes; the silence after it ends at
    // the deadline.
    assert!(stream.next().await.unwrap().is_ok());
    let status = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("the deadline ends the wait")
        .expect("a message, not the end")
        .expect_err("the deadline's status");
    assert!(is_request_timeout(&status), "{status:?}");
    assert_eq!(wait_for_abort(&state).await, ["after-first-1"]);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn a_stream_that_ends_before_the_deadline_is_unchanged_by_it() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let mut stream = client
        .generate(generate_request("in-time-1"))
        .await
        .unwrap()
        .with_deadline(RequestDeadline::after(Duration::from_secs(30)));
    let tx = take_tx(&state).expect("the mock opened a stream");
    for _ in 0..3 {
        tx.send(Ok(proto::GenerateResponse::default()))
            .await
            .unwrap();
    }
    drop(tx);

    let mut messages = 0;
    while let Some(item) = stream.next().await {
        item.expect("a message of the engine's");
        messages += 1;
    }
    assert_eq!(messages, 3);
    stream.mark_completed();
    drop(stream);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        state.aborted_ids().is_empty(),
        "a completed stream sends no abort"
    );
}

#[tokio::test]
async fn a_completed_stream_meets_its_end_not_the_deadline() {
    let (endpoint, state) = spawn_mock().await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let mut stream = client
        .generate(generate_request("completed-1"))
        .await
        .unwrap()
        .with_deadline(RequestDeadline::after(Duration::from_millis(300)));
    let tx = take_tx(&state).expect("the mock opened a stream");
    tx.send(Ok(proto::GenerateResponse::default()))
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().is_ok());

    // The consumer took the final message and marked the request complete,
    // the worker ended its stream, and the deadline passes before the
    // consumer's next poll: that poll meets the end, not the timeout.
    stream.mark_completed();
    drop(tx);
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(
        stream.next().await.is_none(),
        "a completed stream met the deadline instead of its end"
    );
    assert!(
        state.aborted_ids().is_empty(),
        "a completed stream sends no abort"
    );
}
