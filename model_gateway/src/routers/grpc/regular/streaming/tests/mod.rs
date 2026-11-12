mod completion_failure;
mod delta_slicing;
pub(crate) mod eof;
mod failure;
mod generate_bytes;
mod generate_decode;
mod generate_finish;
mod generate_frames;
mod generate_tokens;
mod isolation;
mod local_stop;
mod messages_bytes;
mod messages_failure;
mod messages_truncation;
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

/// A bare text delta written in one pass is byte for byte the serialized
/// choice: over both text fields, the indices, texts that need escaping and
/// the usage tail's three shapes; and the shapes the one-pass writer does
/// not take are not read as text deltas.
#[test]
fn text_delta_frames_write_the_bytes_of_the_serialized_choice() {
    use std::borrow::Cow;

    use openai_protocol::chat::{ChatMessageDeltaRef, ChatStreamChoiceRef};

    fn bare(index: u32, field: TextDelta, text: &str) -> ChatStreamChoiceRef<'_> {
        ChatStreamChoiceRef {
            index,
            delta: ChatMessageDeltaRef {
                role: None,
                content: (field == TextDelta::Content).then_some(Cow::Borrowed(text)),
                tool_calls: None,
                reasoning_content: (field == TextDelta::Reasoning).then_some(Cow::Borrowed(text)),
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        }
    }
    let frame = ChatChunkFrame::new("chatcmpl-1", "model-x", 1_700_000_000, Some("v7"));
    for field in [TextDelta::Content, TextDelta::Reasoning] {
        for index in [0u32, 1, 7, 4_000_000_000] {
            for text in [
                "a",
                " ",
                "hello world",
                "he said \"hi\" \\ done",
                "tab\tnewline\n",
                "\u{1F642} ünïcödé",
                "<|im_end|>",
                "{\"json\":[1,2]}",
                "\u{0}\u{1f}\u{7f}",
            ] {
                let choice = bare(index, field, text);
                assert_eq!(text_delta(&choice), Some((field, text)));
                for (usage, emit_usage_null) in [
                    (None, false),
                    (None, true),
                    (Some(Usage::from_counts(3, 2)), false),
                ] {
                    let mut expected = BytesMut::new();
                    frame.write_chunk(&mut expected, &choice, usage.as_ref(), emit_usage_null);
                    let mut got = BytesMut::new();
                    frame.write_text_delta(
                        &mut got,
                        index,
                        field,
                        text,
                        usage.as_ref(),
                        emit_usage_null,
                    );
                    assert_eq!(
                        got,
                        expected,
                        "{field:?} {index} {text:?} usage {} placeholder {emit_usage_null}",
                        usage.is_some()
                    );
                }
            }
        }
    }

    let mut finished = bare(0, TextDelta::Content, "x");
    finished.finish_reason = Some(Cow::Borrowed("stop"));
    assert_eq!(text_delta(&finished), None, "a finished choice");
    let mut stopped = bare(0, TextDelta::Content, "x");
    stopped.matched_stop = Some(serde_json::json!(7));
    assert_eq!(text_delta(&stopped), None, "a matched stop");
    let mut with_role = bare(0, TextDelta::Content, "x");
    with_role.delta.role = Some(Cow::Borrowed("assistant"));
    assert_eq!(text_delta(&with_role), None, "a role");
    let mut with_call = bare(0, TextDelta::Content, "x");
    with_call.delta.tool_calls = Some(vec![]);
    assert_eq!(text_delta(&with_call), None, "tool calls");
    let mut both = bare(0, TextDelta::Content, "x");
    both.delta.reasoning_content = Some(Cow::Borrowed("y"));
    assert_eq!(text_delta(&both), None, "two texts");
    let mut neither = bare(0, TextDelta::Content, "x");
    neither.delta.content = None;
    assert_eq!(text_delta(&neither), None, "no text");
}

/// A generate event written from the choice's escaped text is byte for byte
/// the serialized event with the raw text: the chunks (quotes, a backslash,
/// control bytes, a tab, a newline, a carriage return, characters of two,
/// three and four bytes, a line separator, an empty chunk, a JSON-looking
/// text) pushed one at a time, each step written as a chunk event and as a
/// finish event, with and without logprobs, over several indices, with and
/// without the frame's room owed.
#[test]
fn generate_event_frames_write_the_bytes_of_the_serialized_event() {
    /// The event as serde serializes it from the raw text: the oracle.
    #[derive(Serialize)]
    struct SerializedEvent<'a> {
        text: &'a str,
        output_ids: &'a [u32],
        meta_info: &'a GenerateMetaInfo<'a>,
        index: u32,
    }
    let chunks = [
        "Hello",
        " \"quoted\"",
        " back\\slash",
        "\ttab\nnewline\rreturn",
        "\u{0}\u{1f}\u{7f}",
        "",
        " caf\u{e9}",
        " \u{65e5}\u{672c}\u{8a9e}",
        " \u{1f600}",
        " \u{2028}",
        " {\"json\":[1,2]}",
    ];
    let input_logprobs = [vec![None], vec![Some(-0.25)]];
    let output_logprobs = [vec![Some(-0.5)], vec![None, Some(-1.0)]];
    for index in [0u32, 1, 7, u32::MAX] {
        let mut escaped = EscapedText::default();
        let mut raw = String::new();
        for (step, chunk) in chunks.iter().enumerate() {
            escaped.push_str(chunk);
            raw.push_str(chunk);
            let ids: Vec<u32> = chunk.chars().map(u32::from).collect();
            for with_logprobs in [false, true] {
                for (what, finish_reason, e2e_latency) in [
                    ("chunk", None, None),
                    (
                        "finish",
                        Some(utils::parse_finish_reason("length", 7)),
                        Some(0.0123),
                    ),
                ] {
                    let logprobs = with_logprobs.then_some(GenerateLogprobs {
                        input_token_logprobs: Some(input_logprobs.as_slice()),
                        output_token_logprobs: Some(output_logprobs.as_slice()),
                    });
                    let event = GenerateEvent {
                        text: &escaped,
                        output_ids: &ids,
                        meta_info: GenerateMetaInfo {
                            id: "gen-frames-0",
                            finish_reason,
                            prompt_tokens: 10,
                            weight_version: "v7",
                            logprobs,
                            completion_tokens: step as u32 + 1,
                            cached_tokens: 3,
                            reasoning_tokens: 1,
                            e2e_latency,
                        },
                        index,
                    };
                    let mut expected = b"data: ".to_vec();
                    serde_json::to_writer(
                        &mut expected,
                        &SerializedEvent {
                            text: &raw,
                            output_ids: &ids,
                            meta_info: &event.meta_info,
                            index,
                        },
                    )
                    .expect("the event serializes");
                    expected.extend_from_slice(b"\n\n");
                    for room_owed in [0, FRAME_ROOM] {
                        let mut got = BytesMut::new();
                        let mut owed = room_owed;
                        StreamingProcessor::write_generate_event(&mut got, &mut owed, &event, what)
                            .expect("the event writes");
                        assert_eq!(
                            got.as_ref(),
                            expected.as_slice(),
                            "index {index}, step {step} {chunk:?}, logprobs {with_logprobs}, {what}, room owed {room_owed}"
                        );
                        assert_eq!(owed, 0, "the write takes the room owed");
                    }
                }
            }
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
    // Choice 1 has only chunks, so its draft count is not yet known.
    assert_eq!(details.rejected_prediction_tokens, None);
    tracker.record_complete(&ProtoGenerateComplete::Sglang(proto::GenerateComplete {
        index: 1,
        prompt_tokens: 10,
        completion_tokens: 4,
        spec_accepted_tokens: 1,
        spec_draft_tokens: 1,
        ..Default::default()
    }));
    let details = tracker.snapshot(0).completion_tokens_details.unwrap();
    assert_eq!(details.accepted_prediction_tokens, Some(3));
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
    let start = serde_json::to_value(StreamingProcessor::initial_messages_usage(0)).unwrap();
    assert_eq!(start["input_tokens"], 0);
    assert_eq!(start["output_tokens"], 0);
    assert_eq!(start["cache_creation_input_tokens"], 0);
    assert_eq!(start["cache_read_input_tokens"], 0);

    let delta = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
        15,
        Some(25),
        0,
        None,
    ))
    .unwrap();
    assert_eq!(delta["output_tokens"], 15);
    assert_eq!(delta["input_tokens"], 25);
    assert_eq!(delta["cache_creation_input_tokens"], 0);
    assert_eq!(delta["cache_read_input_tokens"], 0);
}

/// The final message_delta splits the engine's prompt count as the wire
/// does: the input outside the cache beside the hit as
/// `cache_read_input_tokens`, both integers, their sum the prompt; the
/// creation counter stays zero.
#[test]
fn message_delta_cache_read_input_tokens_is_the_engines_hit() {
    let delta = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
        15,
        Some(945),
        896,
        None,
    ))
    .unwrap();
    assert_eq!(delta["input_tokens"], 49);
    assert_eq!(delta["cache_read_input_tokens"], 896);
    assert_eq!(delta["cache_creation_input_tokens"], 0);
}

/// The `message_start` skeleton carries every key the public API writes
/// on a message's usage; the final delta carries the thinking tokens of a
/// turn that ran with thinking and no such key otherwise.
#[test]
fn messages_usage_always_present_keys_and_thinking_tokens() {
    let start = serde_json::to_value(StreamingProcessor::initial_messages_usage(25)).unwrap();
    // The skeleton reports the prompt's count, as the public API's reference
    // stream does, and no output yet.
    assert_eq!(start["input_tokens"], 25);
    assert_eq!(start["output_tokens"], 0);
    assert_eq!(
        start["cache_creation"],
        serde_json::json!({"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0})
    );
    assert_eq!(start["service_tier"], "standard");
    assert_eq!(start["inference_geo"], "not_available");

    let plain = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
        15,
        Some(25),
        0,
        None,
    ))
    .unwrap();
    assert!(plain.get("output_tokens_details").is_none());
    let thinking = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
        15,
        Some(25),
        0,
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
        15, None, 0, None,
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

/// The SSE frame of a Messages event written the plain way: the event's type line, then the
/// whole event through serde.
fn serialized_messages_event(event: &MessageStreamEvent) -> Vec<u8> {
    let mut bytes = format!(
        "event: {}\ndata: ",
        StreamingProcessor::message_event_type_name(event)
    )
    .into_bytes();
    serde_json::to_writer(&mut bytes, event).expect("the event serializes");
    bytes.extend_from_slice(b"\n\n");
    bytes
}

/// The one-pass `content_block_delta` frames write exactly the bytes of the
/// serialized event, so the literal head can never drift from serde's walk of
/// the two tagged enums: every string delta variant, several block indices and
/// texts that need escaping go through both and compare; the citations delta
/// and the other events keep the serialized whole.
#[test]
fn messages_delta_frames_write_the_bytes_of_the_serialized_event() {
    fn formatted(event: &MessageStreamEvent) -> Vec<u8> {
        let mut buffer = BytesMut::with_capacity(64);
        StreamingProcessor::format_messages_sse_into(&mut buffer, event)
            .expect("the event formats");
        buffer.to_vec()
    }
    let texts = [
        "",
        "Hello",
        "say \"hi\"",
        "back\\slash",
        "tab\tnewline\ncarriage\r",
        "\u{1f}control\u{7f}",
        "caf\u{e9} \u{1f600} \u{2028}",
        "{\"partial\": [1, 2",
    ];
    for index in [0u32, 1, 7, 1234, u32::MAX] {
        for text in texts {
            let deltas = [
                ContentBlockDelta::TextDelta {
                    text: text.to_string(),
                },
                ContentBlockDelta::InputJsonDelta {
                    partial_json: text.to_string(),
                },
                ContentBlockDelta::ThinkingDelta {
                    thinking: text.to_string(),
                },
                ContentBlockDelta::SignatureDelta {
                    signature: text.to_string(),
                },
            ];
            for delta in deltas {
                let event = MessageStreamEvent::ContentBlockDelta { index, delta };
                assert_eq!(
                    formatted(&event),
                    serialized_messages_event(&event),
                    "index {index}, text {text:?}"
                );
            }
        }
    }
    let citation = MessageStreamEvent::ContentBlockDelta {
        index: 2,
        delta: ContentBlockDelta::CitationsDelta {
            citation: messages::Citation::CharLocation(messages::CharLocationCitation {
                cited_text: "cited".to_string(),
                document_index: 0,
                document_title: Some("title".to_string()),
                start_char_index: 3,
                end_char_index: 8,
                file_id: None,
            }),
        },
    };
    assert_eq!(formatted(&citation), serialized_messages_event(&citation));
    let stop = MessageStreamEvent::ContentBlockStop { index: 3 };
    assert_eq!(formatted(&stop), serialized_messages_event(&stop));
    assert_eq!(
        formatted(&stop),
        b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":3}\n\n"
    );
}
