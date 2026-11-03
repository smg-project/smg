mod delta_slicing;
pub(crate) mod eof;
mod failure;
mod generate_finish;
mod isolation;
mod local_stop;
mod pd_anchor;
mod pd_usage;
mod sse_bytes;
mod symphony_chat;
mod thinking_display;
mod usage;
mod usage_chunk;

use super::*;

/// The frames write exactly the bytes of the serialized whole chunk
/// structs, so a field added to a struct turns up here, not on the wire
/// of the per-token chunks alone.
#[test]
fn chunk_frames_write_the_bytes_of_the_whole_structs() {
    fn event(frame: &BytesMut) -> Vec<u8> {
        frame
            .strip_prefix(b"data: ")
            .and_then(|body| body.strip_suffix(b"\n\n"))
            .expect("one SSE data event")
            .to_vec()
    }
    for fingerprint in [None, Some("v7")] {
        let choice = CompletionStreamChoice {
            text: "he said \"hi\"\n".to_string(),
            index: 1,
            logprobs: None,
            finish_reason: Some("stop".to_string()),
        };
        let frame = CompletionChunkFrame::new("cmpl-1", "model-x", 1_700_000_000, fingerprint);
        let mut buffer = BytesMut::new();
        frame.write_chunk(&mut buffer, &choice);
        let whole = CompletionStreamResponse {
            id: "cmpl-1".to_string(),
            object: "text_completion".to_string(),
            created: 1_700_000_000,
            choices: vec![choice],
            model: "model-x".to_string(),
            system_fingerprint: fingerprint.map(str::to_string),
            usage: None,
        };
        assert_eq!(
            event(&buffer),
            serde_json::to_vec(&whole).unwrap(),
            "completion, fingerprint {fingerprint:?}"
        );

        let choice = assistant_choice(0, Some("tok".to_string()), None, None);
        let frame = ChatChunkFrame::new("chatcmpl-1", "model-x", 1_700_000_000, fingerprint);
        for (usage, emit_usage_null) in [
            (None, false),
            (None, true),
            (Some(Usage::from_counts(3, 2)), false),
        ] {
            let mut buffer = BytesMut::new();
            frame.write_chunk(&mut buffer, &choice, usage.as_ref(), emit_usage_null);
            let whole = ChatCompletionStreamResponse {
                id: "chatcmpl-1".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 1_700_000_000,
                model: "model-x".to_string(),
                system_fingerprint: fingerprint.map(str::to_string),
                choices: vec![choice.clone()],
                usage: usage.clone(),
            };
            assert_eq!(
                event(&buffer),
                serde_json::to_vec(&ChatChunkWithUsage::new(&whole, emit_usage_null)).unwrap(),
                "chat, fingerprint {fingerprint:?}, usage {}, placeholder {emit_usage_null}",
                usage.is_some()
            );
        }
    }
}

#[test]
fn continuous_chat_usage_tracks_cumulative_counts_and_shared_prompt() {
    use smg_grpc_client::sglang_proto as proto;

    let mut tracker = ChatStreamUsage::default();
    for (index, completion_tokens, reasoning_tokens) in [(0, 2, 1), (0, 5, 3), (1, 4, 2)] {
        tracker.record_chunk(&ProtoGenerateStreamChunk::Sglang(
            proto::GenerateStreamChunk {
                index,
                prompt_tokens: 10,
                cached_tokens: 8,
                completion_tokens,
                reasoning_tokens,
                token_ids: vec![1], // Incremental IDs must not replace cumulative counts.
                ..Default::default()
            },
        ));
    }
    let usage = serde_json::to_value(tracker.snapshot(0)).unwrap();
    assert_eq!(usage["prompt_tokens"], 10);
    assert_eq!(usage["completion_tokens"], 9);
    assert_eq!(usage["total_tokens"], 19);
    assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 8);
    assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 5);

    tracker.record_complete(&ProtoGenerateComplete::Sglang(proto::GenerateComplete {
        index: 0,
        prompt_tokens: 10,
        cached_tokens: 8,
        completion_tokens: 6,
        reasoning_tokens: 3,
        spec_accepted_tokens: 2,
        spec_draft_tokens: 3,
        ..Default::default()
    }));
    let usage = tracker.snapshot(0);
    assert_eq!(usage.completion_tokens, 10);
    let details = usage.completion_tokens_details.unwrap();
    assert_eq!(details.accepted_prediction_tokens, Some(2));
    assert_eq!(details.rejected_prediction_tokens, Some(1));
}

#[test]
fn continuous_chat_usage_accumulates_delta_ids_without_double_counting_complete() {
    use smg_grpc_client::vllm_proto as proto;

    let mut tracker = ChatStreamUsage::default();
    for ids in [vec![1, 2], vec![3]] {
        tracker.record_chunk(&ProtoGenerateStreamChunk::Vllm(
            proto::GenerateStreamChunk {
                prompt_tokens: 10,
                token_ids: ids,
                ..Default::default()
            },
        ));
    }
    assert_eq!(tracker.snapshot(0).completion_tokens, 3);
    tracker.record_complete(&ProtoGenerateComplete::Vllm(Box::new(
        proto::GenerateComplete {
            prompt_tokens: 10,
            completion_tokens: 3,
            ..Default::default()
        },
    )));
    assert_eq!(tracker.snapshot(0).completion_tokens, 3);
}

/// Wire contract: cache counters serialize as integer zeros, never null,
/// in both the message_start skeleton and the final message_delta usage.
#[test]
fn messages_usage_cache_counters_serialize_as_integer_zeros() {
    let start = serde_json::to_value(StreamingProcessor::initial_messages_usage()).unwrap();
    assert_eq!(start["input_tokens"], 0);
    assert_eq!(start["output_tokens"], 0);
    assert_eq!(start["cache_creation_input_tokens"], 0);
    assert_eq!(start["cache_read_input_tokens"], 0);

    let delta = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
        15,
        Some(25),
        None,
    ))
    .unwrap();
    assert_eq!(delta["output_tokens"], 15);
    assert_eq!(delta["input_tokens"], 25);
    assert_eq!(delta["cache_creation_input_tokens"], 0);
    assert_eq!(delta["cache_read_input_tokens"], 0);
}

/// The `message_start` skeleton carries every key the public API writes
/// on a message's usage; the final delta carries the thinking tokens of a
/// turn that ran with thinking and no such key otherwise.
#[test]
fn messages_usage_always_present_keys_and_thinking_tokens() {
    let start = serde_json::to_value(StreamingProcessor::initial_messages_usage()).unwrap();
    assert_eq!(
        start["cache_creation"],
        serde_json::json!({"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0})
    );
    assert_eq!(start["service_tier"], "standard");
    assert_eq!(start["inference_geo"], "not_available");

    let plain = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
        15,
        Some(25),
        None,
    ))
    .unwrap();
    assert!(plain.get("output_tokens_details").is_none());
    let thinking = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
        15,
        Some(25),
        Some(9),
    ))
    .unwrap();
    assert_eq!(thinking["output_tokens_details"]["thinking_tokens"], 9);
}

/// A clean EOF without a `Complete` message has no authoritative prompt
/// count: `input_tokens` must serialize as null, not a fabricated zero.
#[test]
fn message_delta_input_tokens_null_without_authoritative_usage() {
    let delta = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
        15, None, None,
    ))
    .unwrap();
    assert!(delta.get("input_tokens").is_some_and(Value::is_null));
    assert_eq!(delta["cache_creation_input_tokens"], 0);
}

/// Tokenizer whose decode always fails, simulating a broken deployment
/// (corrupt or mismatched tokenizer files).
struct FailingTokenizer {
    special_tokens: llm_tokenizer::SpecialTokens,
}

impl llm_tokenizer::Encoder for FailingTokenizer {
    fn encode(
        &self,
        _input: &str,
        _add_special_tokens: bool,
    ) -> anyhow::Result<llm_tokenizer::Encoding> {
        Err(anyhow::anyhow!("encode is not supported"))
    }

    fn encode_batch(
        &self,
        _inputs: &[&str],
        _add_special_tokens: bool,
    ) -> anyhow::Result<Vec<llm_tokenizer::Encoding>> {
        Err(anyhow::anyhow!("encode_batch is not supported"))
    }
}

impl llm_tokenizer::Decoder for FailingTokenizer {
    fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
        Err(anyhow::anyhow!("tokenizer decode failed"))
    }

    fn finish_decode(
        &self,
        _ids: &mut Vec<u32>,
        _prefix: &mut String,
        _prefix_index: &mut usize,
        _skip_special_tokens: bool,
    ) -> anyhow::Result<llm_tokenizer::DecodeStep> {
        Err(anyhow::anyhow!("pending token decode failed"))
    }
}

impl Tokenizer for FailingTokenizer {
    fn vocab_size(&self) -> usize {
        0
    }

    fn get_special_tokens(&self) -> &llm_tokenizer::SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }

    fn id_to_token(&self, _id: u32) -> Option<String> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn process_chunk_tokens_propagates_decode_errors() {
    // A decode error must surface instead of being swallowed as `Held`,
    // which would drop the text and let a configured stop silently miss.
    let tokenizer = Arc::new(FailingTokenizer {
        special_tokens: llm_tokenizer::SpecialTokens::default(),
    });
    let config = llm_tokenizer::StopSequenceConfig::default().with_stop_sequence("STOP");
    let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

    let result = StreamingProcessor::process_chunk_tokens(&mut decoder, &[1, 2]);

    let err = result.expect_err("decode failure must propagate");
    assert!(err.contains("Stop decoder failed to process token"));
}

#[test]
fn finish_preserves_pending_decode_errors() {
    let tokenizer = Arc::new(FailingTokenizer {
        special_tokens: llm_tokenizer::SpecialTokens::default(),
    });
    let mut decoder = StopSequenceDecoder::new(
        tokenizer,
        llm_tokenizer::StopSequenceConfig::default(),
        false,
    );
    let error = decoder
        .finish()
        .expect_err("EOF decode error must propagate");
    assert!(error.to_string().contains("pending token decode failed"));
}
