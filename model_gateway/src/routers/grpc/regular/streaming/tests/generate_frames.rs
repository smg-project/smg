//! The generate stream's frames, on the plain loop and on the loop that
//! carries logprobs (the prefill-decode loop delegates to the second): the
//! first token's frame reaches the connection before the reading runs on
//! into a burst, with the one yield of its hand-off (the stream's first
//! frame is the first token's, so the body's own yield after it makes the
//! two); the chunks ready behind the first token share one frame; and a
//! chunk the tokenizer cannot decode fails the stream after the chunks
//! written ahead of it have left; and a burst past the flush bound leaves
//! its full frame through the bound's flush and its remainder through the
//! one poll's flush with the channel full, before the engine's next message
//! is read. The events are the ones the loops wrote when every event had a
//! frame of its own: `generate_bytes` pins their bytes.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::Poll,
};

use futures::StreamExt as _;
use llm_tokenizer::traits::Tokenizer;
use serde_json::Value;
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};
use tokio::{sync::oneshot, task::JoinHandle};

use super::{
    super::*,
    eof::{chunk, complete, gated_stream, pull, scripted_stream, CharacterTokenizer},
};

const MODEL: &str = "generate-frames-test";

fn ctx(return_logprob: bool) -> GenerateStreamContext {
    GenerateStreamContext {
        request_id: "gen-frames".to_string(),
        weight_version: "default".to_string(),
        return_logprob,
        backend_type: "test",
        model: MODEL.to_string(),
        expected_choices: 1,
    }
}

/// A chunk of raw token ids: an id that is no character (a surrogate) is one
/// the character tokenizer cannot decode.
fn chunk_of_ids(index: u32, token_ids: Vec<u32>) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
            index,
            completion_tokens: token_ids.len() as u32,
            token_ids,
            ..Default::default()
        })),
    }
}

/// The `text` of each generate event in `frame`, in order: the choice's
/// text so far, as a generate event carries it.
fn texts_of(frame: &[u8]) -> Vec<String> {
    std::str::from_utf8(frame)
        .expect("UTF-8 SSE")
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str::<Value>(data).expect("SSE JSON"))
        .filter_map(|event| event["text"].as_str().map(str::to_owned))
        .collect()
}

/// The body of the generate loop (`with_logprobs`: the one that carries
/// logprobs) reading `responses`, ready at once: the frames it hands out by
/// polling the reading in place, as a response's body does; the flag turns
/// true when the loop returned a failure.
async fn driven_body(
    with_logprobs: bool,
    responses: Vec<proto::GenerateResponse>,
) -> (DrivenBody, Arc<AtomicBool>, JoinHandle<()>) {
    let (stream, server) = scripted_stream(responses, "0").await;
    driven_body_of(with_logprobs, stream, server)
}

/// [`driven_body`] over `stream` (the mock server behind it in `server`).
fn driven_body_of(
    with_logprobs: bool,
    stream: ProtoStream,
    server: JoinHandle<()>,
) -> (DrivenBody, Arc<AtomicBool>, JoinHandle<()>) {
    let (tx, rx) = driven_channel();
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let ctx = ctx(with_logprobs);
    let failed = Arc::new(AtomicBool::new(false));
    let failure_flag = Arc::clone(&failed);
    let reading = async move {
        let result = if with_logprobs {
            StreamingProcessor::process_generate_streaming_with_input_logprobs(
                tokenizer, stream, ctx, None, &tx, None, None,
            )
            .await
        } else {
            StreamingProcessor::process_generate_streaming(tokenizer, stream, ctx, &tx, None).await
        };
        failure_flag.store(result.is_err(), Ordering::SeqCst);
        ReadingEnd::of(&result)
    };
    (DrivenBody::new(Box::pin(reading), rx), failed, server)
}

/// The first token's frame leaves alone and is pending to the connection
/// before the burst behind it is read; the burst shares one frame (the
/// finish among it), in the events' order.
#[tokio::test]
async fn the_first_tokens_frame_is_pending_to_the_connection_before_the_burst_is_read_on_the_generate_stream(
) {
    for with_logprobs in [false, true] {
        let (mut body, failed, server) = driven_body(
            with_logprobs,
            vec![
                chunk(0, "a"),
                chunk(0, "b"),
                chunk(0, "c"),
                chunk(0, "d"),
                complete(0, "stop"),
            ],
        )
        .await;
        let first = body
            .next()
            .await
            .expect("the body ends only after the first token")
            .expect("a write");
        assert_eq!(
            texts_of(&first),
            ["a"],
            "the body's first frame is the first token's, alone (logprobs {with_logprobs})"
        );
        // The pull after it: pending (the body's own yield after its first
        // frame), so the connection writes the first token's frame before
        // the reading runs on.
        let next = pull(&mut body).await;
        assert!(
            next.is_pending(),
            "the pull after the first token's frame is pending, not a frame (logprobs {with_logprobs}): {next:?}"
        );
        // The pull after that: the burst's frame, the reading having yielded
        // once at the hand-off. Pending here would be a second yield.
        let next = pull(&mut body).await;
        let Poll::Ready(Some(Ok(burst))) = next else {
            panic!("the pull after the pending one is the burst's frame (logprobs {with_logprobs}): {next:?}");
        };
        assert_eq!(
            texts_of(&burst),
            ["ab", "abc", "abcd", "abcd"],
            "the chunks ready behind the first token and the finish share a frame (logprobs {with_logprobs})"
        );
        let rest: Vec<_> = body.collect().await;
        assert!(
            rest.is_empty(),
            "nothing follows the finish's frame (logprobs {with_logprobs}): {rest:?}"
        );
        assert!(
            !failed.load(Ordering::SeqCst),
            "the stream ends without a failure (logprobs {with_logprobs})"
        );
        server.abort();
    }
}

/// A chunk the tokenizer cannot decode after the first frame fails the
/// stream, and the chunk written ahead of it reaches the client first: it was
/// buffered behind the first token's frame because the failing chunk was
/// ready in the same poll, and the loop flushes it before it returns.
#[tokio::test]
async fn a_chunk_that_cannot_be_decoded_fails_the_generate_stream_after_the_chunks_ahead_of_it_have_left(
) {
    for with_logprobs in [false, true] {
        let (mut body, failed, server) = driven_body(
            with_logprobs,
            vec![
                chunk(0, "a"),
                chunk(0, "b"),
                chunk_of_ids(0, vec![0xD800]),
                complete(0, "stop"),
            ],
        )
        .await;
        let first = body
            .next()
            .await
            .expect("the first token's frame")
            .expect("a write");
        assert_eq!(
            texts_of(&first),
            ["a"],
            "the first token alone (logprobs {with_logprobs})"
        );
        assert!(
            pull(&mut body).await.is_pending(),
            "the yield after the first frame"
        );
        let frames: Vec<Bytes> = body.map(|frame| frame.expect("a write")).collect().await;
        let texts: Vec<Vec<String>> = frames.iter().map(|frame| texts_of(frame)).collect();
        assert_eq!(
            texts,
            [["ab"]],
            "the chunk ahead of the failure leaves, then nothing (logprobs {with_logprobs})"
        );
        assert!(
            failed.load(Ordering::SeqCst),
            "the undecodable chunk fails the stream (logprobs {with_logprobs})"
        );
        server.abort();
    }
}

/// A burst past the flush bound behind the first token leaves its full frame
/// in the channel's one slot through the bound's flush; the remainder behind
/// it finds the slot still full when the engine goes quiet and leaves through
/// the awaited send after the one poll (the full-channel hand-back of
/// `flush_in_poll`), before the engine's next message is read; the frames
/// joined are the events' bytes in order.
#[tokio::test]
async fn the_remainder_behind_a_full_frame_leaves_before_the_next_engine_message_on_the_generate_stream(
) {
    for with_logprobs in [false, true] {
        // Every event carries the text so far, one character more per chunk:
        // the burst's frames pass CHUNK_FLUSH_BYTES well before its end.
        let burst: Vec<_> = (0..200).map(|_| chunk(0, "a")).collect();
        let (open, gate) = oneshot::channel();
        let (stream, server) = gated_stream(burst, gate, vec![complete(0, "stop")], "0").await;
        let (mut body, failed, server) = driven_body_of(with_logprobs, stream, server);
        // The frames until the body is pending twice in a row: the reading
        // parked on the quiet engine, every frame it had written out.
        let mut frames: Vec<Bytes> = Vec::new();
        let mut pending_in_a_row = 0;
        while pending_in_a_row < 2 {
            match pull(&mut body).await {
                Poll::Ready(Some(Ok(frame))) => {
                    pending_in_a_row = 0;
                    frames.push(frame);
                }
                Poll::Pending => pending_in_a_row += 1,
                other @ Poll::Ready(_) => panic!(
                    "a frame or pending while the engine is quiet (logprobs {with_logprobs}): {other:?}"
                ),
            }
        }
        let texts: Vec<String> = frames.iter().flat_map(|frame| texts_of(frame)).collect();
        let expected: Vec<String> = (1..=200).map(|n| "a".repeat(n)).collect();
        assert_eq!(
            texts, expected,
            "the frames joined are the burst's events in order (logprobs {with_logprobs})"
        );
        assert!(
            frames.iter().any(|frame| frame.len() >= CHUNK_FLUSH_BYTES),
            "a full frame left at the bound (logprobs {with_logprobs})"
        );
        let remainder = frames.last().expect("the remainder's frame");
        assert!(
            remainder.len() < CHUNK_FLUSH_BYTES,
            "the remainder leaves as a partial frame before the engine's next message (logprobs {with_logprobs}): {} bytes",
            remainder.len()
        );
        // The engine's next message: only now.
        open.send(()).expect("the gate");
        let rest: Vec<Vec<String>> = body
            .map(|frame| texts_of(&frame.expect("a write")))
            .collect()
            .await;
        assert_eq!(
            rest,
            [["a".repeat(200)]],
            "the finish's frame follows alone (logprobs {with_logprobs})"
        );
        assert!(
            !failed.load(Ordering::SeqCst),
            "the stream ends without a failure (logprobs {with_logprobs})"
        );
        server.abort();
    }
}
