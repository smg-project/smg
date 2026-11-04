//! The finish reason of a streamed generate answer is the typed object of
//! the non-streamed answer (`{"type":"stop"}`, `{"type":"length","length":N}`)
//! on a plain stream as on one that returns logprobs.

use smg_grpc_client::vllm_engine::proto::generate_response::Response as GenerationEvent;

use super::{
    super::*,
    eof::{chunk, collect_events, complete, scripted_stream, CharacterTokenizer},
};

/// The finish event of a one-choice generate stream whose engine ends it on
/// `reason` after `completion_tokens` tokens, driven over the plain loop or,
/// with `with_logprobs`, the loop that carries logprobs.
async fn finish_event(reason: &str, completion_tokens: u32, with_logprobs: bool) -> Value {
    let mut terminal = complete(0, reason);
    if let Some(GenerationEvent::Complete(complete)) = &mut terminal.response {
        complete.completion_tokens = completion_tokens;
    }
    let (stream, server) = scripted_stream(vec![chunk(0, "hi"), terminal], "0").await;
    let ctx = GenerateStreamContext {
        request_id: "gen-finish".to_string(),
        weight_version: "default".to_string(),
        return_logprob: with_logprobs,
        backend_type: "test",
        model: "eof-test".to_string(),
        expected_choices: 1,
    };
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let (tx, rx) = sse_channel();
    let result = if with_logprobs {
        StreamingProcessor::process_generate_streaming_with_input_logprobs(
            tokenizer, stream, ctx, None, &tx, None, None,
        )
        .await
    } else {
        StreamingProcessor::process_generate_streaming(tokenizer, stream, ctx, &tx, None).await
    };
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    events
        .into_iter()
        .rfind(|event| !event["meta_info"]["finish_reason"].is_null())
        .expect("a finish event")
}

#[tokio::test]
async fn a_plain_stream_writes_the_typed_finish_reason_of_the_logprobs_stream() {
    for (reason, expected) in [
        ("length", serde_json::json!({"type": "length", "length": 3})),
        ("stop", serde_json::json!({"type": "stop"})),
    ] {
        for with_logprobs in [false, true] {
            let event = finish_event(reason, 3, with_logprobs).await;
            assert_eq!(
                event["meta_info"]["finish_reason"], expected,
                "{reason}, logprobs {with_logprobs}"
            );
        }
    }
}
