//! A client that leaves a Harmony Responses stream ends the iteration at
//! once: the reader of the engine stream races every read against the
//! client's leaving, so a generation that stalled mid-stream is not waited
//! out for a client that is gone, and the iteration ends with the error it
//! ends with when it meets the dead client at a send, not as a completed
//! iteration over partial output.

use std::time::{Duration, Instant};

use bytes::Bytes;
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};
use tokio::sync::oneshot;

use super::{usage_tests::harmony_tokens, HarmonyStreamingProcessor};
use crate::routers::{
    common::sse::{sse_channel, SseReceiver},
    grpc::{
        common::responses::streaming::ResponseStreamEventEmitter,
        context,
        regular::streaming::eof_tests::{complete, gated_stream, scripted_stream},
    },
};

/// One chunk frame carrying `token_ids`.
fn chunk(token_ids: &[u32]) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
            index: 0,
            completion_tokens: token_ids.len() as u32,
            token_ids: token_ids.to_vec(),
            ..Default::default()
        })),
    }
}

/// The Harmony tokens of a final-channel answer `text`, without the
/// `<|return|>` that ends the turn: a generation still under way.
fn open_answer(text: &str) -> Vec<u32> {
    let mut tokens = harmony_tokens(None, text);
    tokens.pop();
    tokens
}

fn emitter() -> ResponseStreamEventEmitter {
    ResponseStreamEventEmitter::new(
        "resp_harmony-leave".to_string(),
        "harmony-test".to_string(),
        1,
    )
}

/// Whether the SSE `frame` carries a `response.output_text.delta` event.
fn is_text_delta(frame: &Bytes) -> bool {
    std::str::from_utf8(frame).is_ok_and(|frame| frame.contains("response.output_text.delta"))
}

/// A client that reads the stream for `stay`, then leaves; whether a text
/// delta reached it.
async fn read_then_leave(mut rx: SseReceiver, stay: Duration) -> bool {
    let mut saw_text = false;
    let leave = tokio::time::sleep(stay);
    tokio::pin!(leave);
    loop {
        tokio::select! {
            frame = rx.recv() => match frame {
                Some(Ok(frame)) => saw_text |= is_text_delta(&frame),
                Some(Err(_)) | None => break,
            },
            () = &mut leave => break,
        }
    }
    drop(rx);
    saw_text
}

/// The client reads the answer's first words, the engine then stalls (the
/// gate never opens), and the client leaves: the iteration ends at once with
/// the client's leaving, not with the engine's next frame.
#[tokio::test]
async fn a_client_that_leaves_while_the_engine_is_parked_ends_the_iteration_at_once() {
    let (_gate, gate_rx) = oneshot::channel();
    let (stream, server) = gated_stream(
        vec![chunk(&open_answer("Hello there"))],
        gate_rx,
        vec![complete(0, "stop")],
        "0",
    )
    .await;
    let mut emitter = emitter();
    let (tx, rx) = sse_channel();
    let started = Instant::now();
    let read = HarmonyStreamingProcessor::process_responses_iteration_stream(
        context::ExecutionResult::Single { stream },
        &mut emitter,
        &tx,
        None,
        None,
    );
    let (outcome, saw_text) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(read, read_then_leave(rx, Duration::from_millis(100)))
    })
    .await
    .expect("the gate never opens; the client's leaving ends the read");
    let Err(error) = outcome else {
        panic!("the iteration ends with the client's leaving, not as completed");
    };
    assert_eq!(error, "Client disconnected");
    assert!(saw_text, "the answer was under way when the client left");
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "the read ended with the client's leaving (100 ms in), not with the engine"
    );
    server.abort();
}

/// A client that left before the first read reads nothing: the frames the
/// engine has ready are not read for it, and the iteration ends the same way.
#[tokio::test]
async fn a_client_that_left_before_the_first_read_reads_nothing() {
    let (stream, server) = scripted_stream(
        vec![chunk(&open_answer("Hello there")), complete(0, "stop")],
        "0",
    )
    .await;
    let mut emitter = emitter();
    let (tx, rx) = sse_channel();
    drop(rx);
    let outcome = HarmonyStreamingProcessor::process_responses_iteration_stream(
        context::ExecutionResult::Single { stream },
        &mut emitter,
        &tx,
        None,
        None,
    )
    .await;
    let Err(error) = outcome else {
        panic!("nothing is read for a client that left");
    };
    assert_eq!(error, "Client disconnected");
    server.abort();
}

/// A client that stays reads the whole answer, and the iteration completes
/// as it always has.
#[tokio::test]
async fn a_client_that_stays_reads_the_answer_to_its_end() {
    let (gate, gate_rx) = oneshot::channel();
    let (stream, server) = gated_stream(
        vec![chunk(&open_answer("Hello there"))],
        gate_rx,
        vec![complete(0, "stop")],
        "0",
    )
    .await;
    let mut emitter = emitter();
    let (tx, mut rx) = sse_channel();
    let drain = async {
        let mut frames = 0;
        while rx.recv().await.is_some() {
            frames += 1;
        }
        frames
    };
    let read = async {
        let outcome = HarmonyStreamingProcessor::process_responses_iteration_stream(
            context::ExecutionResult::Single { stream },
            &mut emitter,
            &tx,
            None,
            None,
        )
        .await;
        drop(tx);
        outcome
    };
    let open = async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let _ = gate.send(());
    };
    let (outcome, frames, ()) = tokio::join!(read, drain, open);
    assert!(
        matches!(
            outcome,
            Ok((super::ResponsesIterationResult::Completed { .. }, _))
        ),
        "the iteration completes"
    );
    assert!(frames > 0, "the client read the answer's events");
    server.abort();
}
