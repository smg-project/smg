//! The message item a Harmony Responses stream closes reads back from the
//! stored object of the turn: the stream closes it with a status, as the
//! regular stream closes its own, so the typed read behind
//! `GET /v1/responses/{id}` keeps it, and the terminal event and the stored
//! object show the same item.

use serde_json::Value;
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};

use super::{super::HarmonyStreamingProcessor, usage::harmony_tokens};
use crate::routers::{
    common::sse::sse_channel,
    grpc::{
        common::responses::streaming::ResponseStreamEventEmitter, context,
        harmony::try_harmony_encoding, regular::streaming::tests::eof::scripted_stream,
    },
};

/// A final-channel answer the token limit cut: no `<|return|>`.
const CUT_ANSWER: &str = "<|channel|>final<|message|>Hello wor";

/// The frames of a text turn: `ids` in chunks of three, then the terminal
/// frame with `finish_reason` and the whole output.
fn frames(ids: &[u32], finish_reason: &str) -> Vec<proto::GenerateResponse> {
    let mut frames: Vec<proto::GenerateResponse> = ids
        .chunks(3)
        .map(|chunk| proto::GenerateResponse {
            response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
                index: 0,
                completion_tokens: chunk.len() as u32,
                token_ids: chunk.to_vec(),
                ..Default::default()
            })),
        })
        .collect();
    frames.push(proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            index: 0,
            finish_reason: finish_reason.to_string(),
            prompt_tokens: 3,
            completion_tokens: ids.len() as u32,
            output_ids: ids.to_vec(),
            ..Default::default()
        })),
    });
    frames
}

/// The streamed iteration over `frames`: the stored object of the turn and
/// the terminal event the stream ends with.
async fn streamed(frames: Vec<proto::GenerateResponse>) -> (Value, Value) {
    let (stream, server) = scripted_stream(frames, "0").await;
    let mut emitter = ResponseStreamEventEmitter::new(
        "resp_harmony-closed-message".to_string(),
        "harmony-test".to_string(),
        1,
    );
    let (tx, mut rx) = sse_channel();
    // The events are read as they come, so the processor never waits on the
    // channel.
    let drain = async { while rx.recv().await.is_some() {} };
    let iteration = async {
        let result = HarmonyStreamingProcessor::process_responses_iteration_stream(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: std::time::Instant::now(),
            },
            &mut emitter,
            &tx,
            None,
            None,
            &[],
        )
        .await;
        drop(tx);
        result
    };
    let (streamed, ()) = tokio::join!(iteration, drain);
    server.abort();
    streamed.expect("the streamed iteration");
    // The stored object before the terminal, which drains the items, as the
    // serving path reads them.
    let stored = serde_json::to_value(emitter.finalize(None)).expect("a JSON object");
    let terminal = emitter.emit_completed(None);
    (stored, terminal)
}

/// The one item of a response object's `output`.
fn only_item<'a>(object: &'a Value, context: &str) -> &'a Value {
    let items = object["output"]
        .as_array()
        .unwrap_or_else(|| panic!("{context}: an output array in {object}"));
    assert_eq!(items.len(), 1, "{context}: {object}");
    &items[0]
}

#[tokio::test]
async fn the_stored_object_of_a_streamed_text_turn_carries_its_message_item() {
    let closed = harmony_tokens(None, "Hello there!");
    let cut = try_harmony_encoding()
        .expect("harmony encoding must load")
        .tokenizer()
        .encode_with_special_tokens(CUT_ANSWER);
    for (name, ids, finish_reason, text) in [
        ("a turn that stopped", &closed, "stop", "Hello there!"),
        ("a turn the token limit cut", &cut, "length", "Hello wor"),
    ] {
        let (stored, terminal) = streamed(frames(ids, finish_reason)).await;

        let item = only_item(&stored, &format!("{name}, the stored object"));
        assert_eq!(item["type"], "message", "{name}: {item}");
        assert_eq!(item["status"], "completed", "{name}: {item}");
        assert_eq!(item["content"][0]["text"], text, "{name}: {item}");

        let shown = only_item(
            &terminal["response"],
            &format!("{name}, the terminal event"),
        );
        for field in ["id", "type", "status"] {
            assert_eq!(
                shown[field], item[field],
                "{name}: {field} of {shown} vs {item}"
            );
        }
        assert_eq!(shown["content"][0]["text"], text, "{name}: {shown}");
    }
}

#[tokio::test]
async fn the_message_item_of_a_turn_the_engine_failed_reads_in_progress() {
    // The regular stream leaves a message item `in_progress` when the generation failed:
    // closing the event sequence does not say it succeeded. The Harmony stream closes its
    // message item under the same finish, the one its terminal frame named, not the finish an
    // earlier turn of the request left on the emitter. (An `error` finish never reaches the
    // decode as a terminal frame: the stream reader ends the iteration with an error first.)
    let ids = harmony_tokens(None, "Hello there!");
    let (stored, terminal) = streamed(frames(&ids, "failed")).await;
    let name = "a turn the engine ended with failed";

    assert_eq!(stored["status"], "failed", "{name}: {stored}");
    let item = only_item(&stored, &format!("{name}, the stored object"));
    assert_eq!(item["type"], "message", "{name}: {item}");
    assert_eq!(item["status"], "in_progress", "{name}: {item}");

    assert_eq!(
        terminal["response"]["status"], "failed",
        "{name}: {terminal}"
    );
    let shown = only_item(
        &terminal["response"],
        &format!("{name}, the terminal event"),
    );
    assert_eq!(shown["status"], item["status"], "{name}: {shown} vs {item}");
}
