//! A generate stream whose tokenizer cannot decode a chunk fails as the chat
//! stream does, on the plain loop as on the one that carries logprobs: the
//! loop returns the failure, the task that owns the response ends it with
//! the error event and `[DONE]`, the client reads no chunk with empty text,
//! and the gateway logs the failure with the request id.

use tracing_test::traced_test;

use super::{
    super::*,
    eof::{chunk, chunk_of_ids, complete, scripted_stream},
    FailingTokenizer,
};
use crate::routers::{
    common::sse::SseReceiver,
    grpc::utils::{StreamFailure, StreamSource},
};

const REQUEST_ID: &str = "gen-decode-failure";
const MODEL: &str = "decode-failure-test";

/// The SSE frames the client read, as text, in order.
async fn collect_frames(mut rx: SseReceiver) -> Vec<String> {
    let mut frames = Vec::new();
    while let Some(frame) = rx.recv().await {
        let frame = frame.expect("successful SSE write");
        frames.push(String::from_utf8(frame.to_vec()).expect("UTF-8 SSE"));
    }
    frames
}

fn byte_level_tokenizer() -> Arc<dyn Tokenizer> {
    let definition = serde_json::json!({
        "version": "1.0",
        "added_tokens": [],
        "model": {
            "type": "WordLevel",
            "unk_token": "[UNK]",
            "vocab": {
                "[UNK]": 0,
                "ð": 240,
                "Ł": 159,
                "ĺ": 152,
                "Ģ": 128
            }
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": true
        }
    });
    let file = tempfile::NamedTempFile::new().expect("temporary tokenizer file");
    std::fs::write(
        file.path(),
        serde_json::to_vec(&definition).expect("tokenizer JSON serializes"),
    )
    .expect("tokenizer JSON is written");
    Arc::new(
        llm_tokenizer::HuggingFaceTokenizer::from_file(
            file.path().to_str().expect("UTF-8 temporary path"),
        )
        .expect("byte-level tokenizer loads"),
    )
}

async fn generated_texts(
    chunks: &[&[u32]],
    with_logprobs: bool,
    send_complete: bool,
) -> Vec<String> {
    let mut responses: Vec<_> = chunks
        .iter()
        .map(|ids| chunk_of_ids(0, ids.to_vec()))
        .collect();
    if send_complete {
        responses.push(complete(0, "stop"));
    }
    let (stream, server) = scripted_stream(responses, "0").await;
    let ctx = GenerateStreamContext {
        request_id: "gen-byte-level".to_string(),
        weight_version: "default".to_string(),
        return_logprob: with_logprobs,
        backend_type: "test",
        model: "byte-level-test".to_string(),
        expected_choices: 1,
        dispatch_wait: None,
    };
    let (tx, rx) = sse_channel();
    let result = if with_logprobs {
        StreamingProcessor::process_generate_streaming_with_input_logprobs(
            byte_level_tokenizer(),
            stream,
            ctx,
            None,
            &tx,
            None,
            None,
        )
        .await
    } else {
        StreamingProcessor::process_generate_streaming(
            byte_level_tokenizer(),
            stream,
            ctx,
            &tx,
            None,
        )
        .await
    };
    drop(tx);
    let frames = collect_frames(rx).await;
    server.abort();
    assert!(result.is_ok(), "logprobs {with_logprobs}: {result:?}");
    frames
        .iter()
        .flat_map(|frame| frame.lines())
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|body| {
            let event: Value = serde_json::from_str(body).expect("generate event is JSON");
            event["text"]
                .as_str()
                .expect("generate event carries text")
                .to_string()
        })
        .collect()
}

#[tokio::test]
async fn generate_byte_level_decode_is_independent_of_engine_chunk_boundaries() {
    let whole: &[u32] = &[240, 159, 152, 128];
    let split: &[&[u32]] = &[&whole[..2], &whole[2..]];

    for with_logprobs in [false, true] {
        let grouped = generated_texts(&[whole], with_logprobs, true).await;
        let divided = generated_texts(split, with_logprobs, true).await;

        assert_eq!(grouped, ["😀", "😀"], "logprobs {with_logprobs}");
        assert_eq!(divided, ["", "😀", "😀"], "logprobs {with_logprobs}");
        assert!(
            divided.iter().all(|text| !text.contains('\u{fffd}')),
            "logprobs {with_logprobs}: {divided:?}"
        );
        assert_eq!(divided.last(), grouped.last(), "logprobs {with_logprobs}");
    }
}

#[tokio::test]
async fn generate_byte_level_decode_drops_incomplete_character_at_stream_end() {
    for with_logprobs in [false, true] {
        for send_complete in [false, true] {
            let texts = generated_texts(&[&[240, 159]], with_logprobs, send_complete).await;
            let expected: &[&str] = if send_complete { &["", ""] } else { &[""] };
            assert_eq!(
                texts, expected,
                "logprobs {with_logprobs}, complete {send_complete}"
            );
        }
    }
}

#[tokio::test]
async fn generate_byte_level_decode_releases_held_text_at_stream_end() {
    for with_logprobs in [false, true] {
        for send_complete in [false, true] {
            let texts = generated_texts(&[&[128]], with_logprobs, send_complete).await;
            assert_eq!(
                texts,
                ["", "\u{fffd}"],
                "logprobs {with_logprobs}, complete {send_complete}"
            );
        }
    }
}

/// A one-choice generate stream of one chunk, read over the plain loop or,
/// with `with_logprobs`, the loop that carries logprobs, by a tokenizer whose
/// decode fails, and ended as the streaming task ends a response (the
/// failure's event when the loop failed, then `[DONE]`): the loop's result
/// and the frames the client read.
async fn stream_with_failing_decode(
    with_logprobs: bool,
) -> (Result<(), StreamFailure>, Vec<String>) {
    let (stream, server) = scripted_stream(vec![chunk(0, "hi"), complete(0, "stop")], "0").await;
    let ctx = GenerateStreamContext {
        request_id: REQUEST_ID.to_string(),
        weight_version: "default".to_string(),
        return_logprob: with_logprobs,
        backend_type: "test",
        model: MODEL.to_string(),
        expected_choices: 1,
        dispatch_wait: None,
    };
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(FailingTokenizer {
        special_tokens: llm_tokenizer::SpecialTokens::default(),
    });
    let (tx, rx) = sse_channel();
    let result = if with_logprobs {
        StreamingProcessor::process_generate_streaming_with_input_logprobs(
            tokenizer, stream, ctx, None, &tx, None, None,
        )
        .await
    } else {
        StreamingProcessor::process_generate_streaming(tokenizer, stream, ctx, &tx, None).await
    };
    if let Err(failure) = &result {
        let dispatch = context::DispatchMetadata {
            request_id: REQUEST_ID.to_string(),
            model: MODEL.to_string(),
            created: 1,
            weight_version: None,
        };
        let source = StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_GENERATE);
        utils::send_stream_failure(&tx, failure, &source).await;
    }
    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
    drop(tx);
    let frames = collect_frames(rx).await;
    server.abort();
    (result, frames)
}

#[tokio::test]
#[traced_test]
async fn an_undecodable_chunk_fails_the_generate_stream_instead_of_sending_empty_text() {
    for with_logprobs in [false, true] {
        let (result, frames) = stream_with_failing_decode(with_logprobs).await;

        let failure = result.expect_err("the decode failure ends the stream");
        assert!(
            matches!(&failure, StreamFailure::Internal(message) if message.contains("decode")),
            "the gateway's own failure, naming the decode: {failure:?} (logprobs {with_logprobs})"
        );

        assert_eq!(
            frames.len(),
            2,
            "the error event and [DONE], no chunk before them: {frames:?} (logprobs {with_logprobs})"
        );
        let event: Value = serde_json::from_str(
            frames[0]
                .strip_prefix("data: ")
                .and_then(|body| body.strip_suffix("\n\n"))
                .expect("one SSE data event"),
        )
        .expect("the error event is JSON");
        assert_eq!(
            event["error"]["type"], "internal_error",
            "a failure of the gateway's own processing (logprobs {with_logprobs})"
        );
        assert!(
            event["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("Failed to decode generate chunk")),
            "the event names the decode: {event} (logprobs {with_logprobs})"
        );
        assert!(
            event.get("text").is_none(),
            "no text reaches the client as the model's: {event} (logprobs {with_logprobs})"
        );
        assert_eq!(frames[1], "data: [DONE]\n\n", "logprobs {with_logprobs}");
    }
    assert!(
        logs_contain("Streaming response failed in the gateway"),
        "the failure is logged"
    );
    assert!(logs_contain(REQUEST_ID), "the line names the request");
}
