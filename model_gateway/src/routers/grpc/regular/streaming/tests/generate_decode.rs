//! A generate stream whose tokenizer cannot decode a chunk fails as the chat
//! stream does, on the plain loop as on the one that carries logprobs: the
//! loop returns the failure, the task that owns the response ends it with
//! the error event and `[DONE]`, the client reads no chunk with empty text,
//! and the gateway logs the failure with the request id.

use tracing_test::traced_test;

use super::{
    super::*,
    eof::{chunk, complete, scripted_stream},
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
