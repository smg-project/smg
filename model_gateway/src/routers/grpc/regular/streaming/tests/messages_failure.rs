//! A Messages stream whose later chunk cannot be decoded fails as the chat
//! stream does, after every event written ahead of the failure has left: the
//! loop flushes what it wrote since its last frame before it returns the
//! failure, as it does ahead of an upstream failure.

use serde_json::Value;
use tokio::sync::oneshot;

use super::{
    super::*,
    eof::{
        chunk, chunk_of_ids, complete, dispatch, messages_spec, processor, staged_stream,
        CharacterTokenizer,
    },
};
use crate::routers::common::sse::sse_channel;

/// The events of the SSE frames, in order.
fn events_of(frames: &[Bytes]) -> Vec<Value> {
    frames
        .iter()
        .flat_map(|frame| {
            std::str::from_utf8(frame)
                .expect("UTF-8 SSE")
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .map(|data| serde_json::from_str(data).expect("SSE JSON"))
                .collect::<Vec<Value>>()
        })
        .collect()
}

/// The text deltas of the Messages events, in order.
fn text_of(events: &[Value]) -> String {
    events
        .iter()
        .filter(|event| event["type"] == "content_block_delta")
        .filter_map(|event| event["delta"]["text"].as_str())
        .collect()
}

/// A chunk the tokenizer cannot decode after the first token fails the
/// stream, and every event written ahead of it reaches the client first: the
/// second chunk's delta, buffered behind the first token's frame because the
/// failing chunk was ready in the same poll, is flushed before the loop
/// returns.
#[tokio::test]
async fn a_chunk_that_fails_to_decode_after_the_first_token_flushes_the_events_written_ahead_of_it()
{
    // The first token on its own; then, once its frame left, a second chunk
    // and a chunk no tokenizer decodes, ready together.
    let (gate, gated) = oneshot::channel();
    let (stream, server) = staged_stream(
        vec![
            (None, vec![chunk(0, "hi")]),
            (
                Some(gated),
                vec![
                    chunk(0, " there"),
                    chunk_of_ids(0, vec![0xD800]),
                    complete(0, "stop"),
                ],
            ),
        ],
        "0",
    )
    .await;
    let (tx, mut rx) = sse_channel();
    let processor = processor(false);
    let reading = async {
        let result = processor
            .process_messages_streaming_chunks(
                stream,
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, true),
                messages_spec(false),
                &tx,
                None,
            )
            .await;
        drop(tx);
        result
    };
    let reader = async {
        let mut frames = Vec::new();
        // `message_start`, then the first token's frame: the reader opens the
        // gate only after both left.
        for _ in 0..2 {
            frames.push(rx.recv().await.expect("a frame").expect("a write"));
        }
        gate.send(()).expect("the reader waits at the gate");
        while let Some(frame) = rx.recv().await {
            frames.push(frame.expect("a write"));
        }
        frames
    };
    let (result, frames) = tokio::join!(reading, reader);
    server.abort();
    let Err(failure) = result else {
        panic!("the chunk the tokenizer cannot decode fails the stream");
    };
    assert!(
        format!("{failure:?}").contains("decode"),
        "the failure names the decode: {failure:?}"
    );
    let events = events_of(&frames);
    assert_eq!(
        events.first().and_then(|event| event["type"].as_str()),
        Some("message_start"),
        "the stream opened before the failure: {events:?}"
    );
    assert_eq!(
        text_of(&events),
        "hi there",
        "every text delta written ahead of the failure reached the client: {events:?}"
    );
}
