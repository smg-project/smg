use openai_protocol::common::{ToolCallDelta, ToolChoiceValue};
use symphony::{formats::Table, CallIds, Declared, Engine};

use super::*;

const SPLIT_TOOL_PREFIX: &str = r#"{"name":"get_weather","arguments":{"city":"#;

/// The call ids of a test reader count from this nonce, so a recorded
/// delta compares whole.
const NONCE: &str = "0123456789abcdef01234567";

/// A reader over the `json` table (what the converter's older generic
/// JSON parser read: bare call objects), its call ids counting from
/// `NONCE`.
fn json_reader(tools: &[Tool]) -> Reader {
    let ids = CallIds::from_nonce(NONCE).unwrap();
    Reader::over(
        Engine::with_call_ids(Table::Json.shared(), Declared::of(tools), ids),
        false,
        false,
        "",
    )
    .unwrap()
}

fn byte_fallback_tokenizer() -> Arc<dyn Tokenizer> {
    let path =
        std::env::temp_dir().join(format!("smg-byte-fallback-{}.json", uuid::Uuid::now_v7()));
    let mut vocab = serde_json::Map::new();
    vocab.insert("[UNK]".into(), serde_json::json!(0));
    for byte in 0_u32..=255 {
        vocab.insert(format!("<0x{byte:02X}>"), serde_json::json!(byte + 1));
    }
    // A regular piece releases a tool prefix before a byte-only final run.
    vocab.insert(SPLIT_TOOL_PREFIX.into(), serde_json::json!(257));
    let definition = serde_json::json!({
        "version":"1.0", "added_tokens":[],
        "model":{"type":"WordLevel", "unk_token":"[UNK]", "vocab":vocab},
        "decoder":{"type":"Sequence", "decoders":[
            {"type":"ByteFallback"}, {"type":"Fuse"}
        ]}
    });
    std::fs::write(&path, serde_json::to_vec(&definition).unwrap()).unwrap();
    let tokenizer = llm_tokenizer::HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(path).unwrap();
    Arc::new(tokenizer)
}

#[test]
/// Flush pending byte fallback at completion while enforcing the configured stop boundary.
fn byte_fallback_complete_releases_tail_and_enforces_stop() {
    let tokenizer = byte_fallback_tokenizer();
    for stop in ["NEVER", "é"] {
        let mut handle = GrpcResponseConverterHandle {
            tokenizer: tokenizer.clone(),
            readers: HashMap::new(),
            stop: Some(StringOrArray::Array(vec![stop.into()])),
            stop_token_ids: None,
            model: "byte-fallback-test".into(),
            request_id: "test".into(),
            created: 0,
            system_fingerprint: None,
            reading: Reading::of("byte-fallback-test", None, None, true),
            prompt: String::new(),
            stream_state: StreamStateManager::new(),
            initial_prompt_tokens: None,
            skip_special_tokens: true,
        };
        let chunk = proto::GenerateResponse {
            media_identity: None,
            prepared_input: Vec::new(),
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Chunk(
                proto::GenerateStreamChunk {
                    token_ids: vec![0xC3 + 1, 0xA9 + 1],
                    completion_tokens: 2,
                    ..Default::default()
                },
            )),
        };
        // The character is complete with its second byte: it streams at
        // once, or the stop on it fires at once (nothing to send then).
        let streamed = convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer).unwrap();
        if stop == "é" {
            assert!(streamed.is_none(), "{streamed:?}");
        } else {
            let streamed = streamed.unwrap();
            assert_eq!(streamed.choices[0].delta.content.as_deref(), Some("é"));
            assert!(streamed.choices[0].finish_reason.is_none());
        }
        let complete = proto::GenerateResponse {
            media_identity: None,
            prepared_input: Vec::new(),
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Complete(
                proto::GenerateComplete {
                    completion_tokens: 2,
                    finish_reason: "length".into(),
                    ..Default::default()
                },
            )),
        };
        let response = convert_proto_chunk_to_openai(complete, &mut handle, &tokenizer)
            .unwrap()
            .unwrap();
        let choice = &response.choices[0];
        if stop == "é" {
            assert!(choice.delta.content.is_none());
            assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
            assert_eq!(choice.matched_stop, Some(Value::String("é".into())));
        } else {
            assert!(choice.delta.content.is_none(), "{response:?}");
            assert_eq!(choice.finish_reason.as_deref(), Some("length"));
        }
    }
}
/// The first chunk of a choice that carries text goes out as one delta
/// with the role and that text, on the plain decode path and on the
/// stop-decoder path alike; a first chunk a table reads carries the role
/// with what the reader released. Nothing the decoder released is
/// dropped.
#[test]
fn the_first_chunk_of_a_choice_carries_the_role_and_its_text() {
    let tokenizer = byte_fallback_tokenizer();
    for stop in [None, Some(StringOrArray::Array(vec!["NEVER".into()]))] {
        let mut handle = GrpcResponseConverterHandle {
            tokenizer: tokenizer.clone(),
            readers: HashMap::new(),
            stop,
            stop_token_ids: None,
            model: "byte-fallback-test".into(),
            request_id: "first-delta".into(),
            created: 0,
            system_fingerprint: None,
            reading: Reading::of("byte-fallback-test", None, None, true),
            prompt: String::new(),
            stream_state: StreamStateManager::new(),
            initial_prompt_tokens: None,
            skip_special_tokens: true,
        };
        let chunk = proto::GenerateResponse {
            media_identity: None,
            prepared_input: Vec::new(),
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Chunk(
                proto::GenerateStreamChunk {
                    token_ids: vec![0xC3 + 1, 0xA9 + 1],
                    completion_tokens: 2,
                    ..Default::default()
                },
            )),
        };
        let first = convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer)
            .unwrap()
            .expect("the first chunk's text is sent");
        let delta = &first.choices[0].delta;
        assert_eq!(delta.role.as_deref(), Some("assistant"), "{first:?}");
        assert_eq!(delta.content.as_deref(), Some("é"), "{first:?}");
    }
    // A first chunk a table reads: under `none` on a model the registry
    // does not know, `plain` gives the text as content; with the json
    // reader, what it released goes out with the role, as content or as
    // calls, never nothing.
    let tool_text = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let token_ids: Vec<u32> = tool_text.bytes().map(|byte| u32::from(byte) + 1).collect();
    for tool_choice in [Some(ToolChoice::Value(ToolChoiceValue::None)), None] {
        let disabled = tool_choice.is_some();
        let mut handle = tool_test_handle(tokenizer.clone(), tool_choice);
        let chunk = proto::GenerateResponse {
            media_identity: None,
            prepared_input: Vec::new(),
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Chunk(
                proto::GenerateStreamChunk {
                    token_ids: token_ids.clone(),
                    completion_tokens: token_ids.len() as u32,
                    ..Default::default()
                },
            )),
        };
        let first = convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer)
            .unwrap()
            .expect("the first chunk is sent");
        let delta = &first.choices[0].delta;
        assert_eq!(delta.role.as_deref(), Some("assistant"), "{first:?}");
        if disabled {
            assert_eq!(delta.content.as_deref(), Some(tool_text), "{first:?}");
            assert!(delta.tool_calls.is_none(), "{first:?}");
        } else {
            // A complete call in one chunk: the reader releases it as tool
            // calls and nothing as content.
            assert!(delta.content.is_none(), "{first:?}");
            assert!(delta.tool_calls.is_some(), "{first:?}");
        }
    }
}

/// A chunk carries the fingerprint the stream set from the engine's
/// weight version: the version itself, or the gateway's `default`.
#[test]
fn a_chunk_carries_the_engines_weight_version_as_its_fingerprint() {
    let tokenizer = byte_fallback_tokenizer();
    for fingerprint in [Some("default".to_string()), Some("v7".to_string())] {
        let mut handle = tool_test_handle(tokenizer.clone(), None);
        handle.system_fingerprint = fingerprint.clone();
        let chunk = proto::GenerateResponse {
            media_identity: None,
            prepared_input: Vec::new(),
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Chunk(
                proto::GenerateStreamChunk {
                    token_ids: vec![0xC3 + 1, 0xA9 + 1],
                    completion_tokens: 2,
                    ..Default::default()
                },
            )),
        };
        let first = convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer)
            .unwrap()
            .expect("the first chunk is sent");
        assert_eq!(first.system_fingerprint, fingerprint, "{first:?}");
    }
}

fn tool_test_handle(
    tokenizer: Arc<dyn Tokenizer>,
    tool_choice: Option<ToolChoice>,
) -> GrpcResponseConverterHandle {
    let tools: Vec<Tool> = serde_json::from_value(serde_json::json!([{
        "type":"function", "function":{
            "name":"get_weather", "parameters":{
                "type":"object", "properties":{"city":{"type":"string"}}
            }
        }
    }]))
    .unwrap();
    // The json reader stands for a model whose table reads these calls;
    // under `none` the handle's own reading reads the choice (`plain`:
    // the registry does not know this model), as in production.
    let readers = match tool_choice {
        Some(ToolChoice::Value(ToolChoiceValue::None)) => HashMap::new(),
        _ => HashMap::from([(0, json_reader(&tools))]),
    };
    let reading = Reading::of(
        "byte-fallback-test",
        Some(&tools),
        tool_choice.as_ref(),
        true,
    );
    GrpcResponseConverterHandle {
        tokenizer,
        readers,
        stop: Some(StringOrArray::Array(vec!["NEVER".into()])),
        stop_token_ids: None,
        model: "byte-fallback-test".into(),
        request_id: "tool-tail".into(),
        created: 0,
        system_fingerprint: None,
        reading,
        prompt: String::new(),
        stream_state: StreamStateManager::new(),
        initial_prompt_tokens: None,
        skip_special_tokens: true,
    }
}

#[test]
/// Process the final tail through the reader while respecting disabled tool choice.
fn byte_fallback_complete_parses_tool_tail_and_respects_none() {
    let tokenizer = byte_fallback_tokenizer();
    let tool_text = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let token_ids: Vec<u32> = tool_text.bytes().map(|byte| u32::from(byte) + 1).collect();
    let completion_tokens = token_ids.len() as u32;
    for tool_choice in [None, Some(ToolChoice::Value(ToolChoiceValue::None))] {
        let disabled = tool_choice.is_some();
        let mut handle = tool_test_handle(tokenizer.clone(), tool_choice);
        let chunk = proto::GenerateResponse {
            media_identity: None,
            prepared_input: Vec::new(),
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Chunk(
                proto::GenerateStreamChunk {
                    token_ids: token_ids.clone(),
                    prompt_tokens: 7,
                    completion_tokens,
                    ..Default::default()
                },
            )),
        };
        // Every byte of the JSON is a complete character: the text streams
        // with its chunk and goes through the tool parser there.
        let streamed = convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer)
            .unwrap()
            .unwrap();
        let streamed_choice = &streamed.choices[0];
        assert!(streamed_choice.finish_reason.is_none());
        let complete = proto::GenerateResponse {
            media_identity: None,
            prepared_input: Vec::new(),
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Complete(
                proto::GenerateComplete {
                    output_ids: token_ids.clone(),
                    prompt_tokens: 7,
                    completion_tokens,
                    cached_tokens: 3,
                    finish_reason: "stop".into(),
                    matched_stop: Some(MatchedStop::MatchedTokenId(99)),
                    ..Default::default()
                },
            )),
        };
        let response = convert_proto_chunk_to_openai(complete, &mut handle, &tokenizer)
            .unwrap()
            .unwrap();
        let choice = &response.choices[0];
        assert_eq!(choice.matched_stop, Some(serde_json::json!(99)));
        let wire = serde_json::to_value(&response).unwrap();
        assert_eq!(wire["usage"]["prompt_tokens"], 7);
        assert_eq!(wire["usage"]["completion_tokens"], completion_tokens);
        assert_eq!(wire["usage"]["prompt_tokens_details"]["cached_tokens"], 3);
        if disabled {
            assert_eq!(streamed_choice.delta.content.as_deref(), Some(tool_text));
            assert!(streamed_choice.delta.tool_calls.is_none());
            assert!(choice.delta.content.is_none(), "{response:?}");
            assert!(choice.delta.tool_calls.is_none());
            assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
        } else {
            assert!(streamed_choice.delta.content.is_none());
            assert!(choice.delta.content.is_none());
            assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
            let mut calls = streamed_choice.delta.tool_calls.clone().unwrap_or_default();
            calls.extend(choice.delta.tool_calls.clone().unwrap_or_default());
            // The existing parser sends the name and arguments as two deltas
            // for one index; preserve that conversion instead of coalescing it.
            assert_eq!(calls.len(), 2);
            assert!(calls.iter().all(|call| call.index == 0));
            let name_delta = &calls[0];
            assert!(name_delta.id.as_deref().unwrap().starts_with("call_"));
            assert_eq!(name_delta.tool_type.as_deref(), Some("function"));
            assert_eq!(
                name_delta.function.as_ref().unwrap().name.as_deref(),
                Some("get_weather")
            );
            let arguments: String = calls
                .iter()
                .filter_map(|call| {
                    call.function
                        .as_ref()
                        .and_then(|function| function.arguments.as_deref())
                })
                .collect();
            let arguments: Value = serde_json::from_str(&arguments).unwrap();
            assert_eq!(arguments, serde_json::json!({"city":"Paris"}));
        }
    }
}
#[test]
/// Recognize a tool control sequence split across chunks before completion.
fn byte_fallback_complete_closed_split_tool_control() {
    let tokenizer = byte_fallback_tokenizer();
    let tail = r#""Paris"}}"#;
    let mut handle = tool_test_handle(tokenizer.clone(), None);
    let prefix_chunk = proto::GenerateResponse {
        media_identity: None,
        prepared_input: Vec::new(),
        request_id: String::new(),
        response: Some(proto::generate_response::Response::Chunk(
            proto::GenerateStreamChunk {
                token_ids: vec![257],
                prompt_tokens: 7,
                completion_tokens: 1,
                ..Default::default()
            },
        )),
    };
    let prefix_response = convert_proto_chunk_to_openai(prefix_chunk, &mut handle, &tokenizer)
        .unwrap()
        .unwrap();
    let prefix_calls = prefix_response.choices[0]
        .delta
        .tool_calls
        .as_ref()
        .unwrap();
    assert!(prefix_calls.iter().any(|call| call
        .function
        .as_ref()
        .and_then(|function| function.name.as_deref())
        == Some("get_weather")));
    let mut arguments: String = prefix_calls
        .iter()
        .filter_map(|call| {
            call.function
                .as_ref()
                .and_then(|function| function.arguments.as_deref())
        })
        .collect();
    let tail_ids: Vec<u32> = tail.bytes().map(|byte| u32::from(byte) + 1).collect();
    let completion_tokens = 1 + tail_ids.len() as u32;
    let tail_chunk = proto::GenerateResponse {
        media_identity: None,
        prepared_input: Vec::new(),
        request_id: String::new(),
        response: Some(proto::generate_response::Response::Chunk(
            proto::GenerateStreamChunk {
                token_ids: tail_ids.clone(),
                prompt_tokens: 7,
                completion_tokens,
                ..Default::default()
            },
        )),
    };
    // The tail's bytes are complete characters: they stream with their
    // chunk and the parser releases the arguments there.
    if let Some(tail_response) =
        convert_proto_chunk_to_openai(tail_chunk, &mut handle, &tokenizer).unwrap()
    {
        let tail_choice = &tail_response.choices[0];
        assert!(tail_choice.delta.content.is_none(), "{tail_response:?}");
        for call in tail_choice.delta.tool_calls.iter().flatten() {
            assert_eq!(call.index, 0);
            if let Some(fragment) = call
                .function
                .as_ref()
                .and_then(|function| function.arguments.as_deref())
            {
                arguments.push_str(fragment);
            }
        }
    }
    let mut output_ids = vec![257];
    output_ids.extend(tail_ids);
    let complete = proto::GenerateResponse {
        media_identity: None,
        prepared_input: Vec::new(),
        request_id: String::new(),
        response: Some(proto::generate_response::Response::Complete(
            proto::GenerateComplete {
                output_ids,
                prompt_tokens: 7,
                completion_tokens,
                cached_tokens: 3,
                finish_reason: "stop".into(),
                matched_stop: Some(MatchedStop::MatchedTokenId(99)),
                ..Default::default()
            },
        )),
    };
    let final_response = convert_proto_chunk_to_openai(complete, &mut handle, &tokenizer)
        .unwrap()
        .unwrap();
    let choice = &final_response.choices[0];
    assert!(choice.delta.content.is_none(), "{final_response:?}");
    for call in choice.delta.tool_calls.iter().flatten() {
        assert_eq!(call.index, 0);
        if let Some(fragment) = call
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_deref())
        {
            arguments.push_str(fragment);
        }
    }
    let parsed: Value = serde_json::from_str(&arguments).unwrap();
    assert_eq!(parsed, serde_json::json!({"city":"Paris"}));
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(choice.matched_stop, Some(serde_json::json!(99)));
    let wire = serde_json::to_value(&final_response).unwrap();
    assert_eq!(wire["usage"]["prompt_tokens"], 7);
    assert_eq!(wire["usage"]["completion_tokens"], completion_tokens);
    assert_eq!(wire["usage"]["prompt_tokens_details"]["cached_tokens"], 3);
}
#[test]
fn byte_fallback_complete_does_not_repeat_emitted_content() {
    let tokenizer = byte_fallback_tokenizer();
    let mut handle = tool_test_handle(
        tokenizer.clone(),
        Some(ToolChoice::Value(ToolChoiceValue::None)),
    );
    let mut text = String::new();
    for ids in [vec![257], vec![0xC3 + 1, 0xA9 + 1]] {
        let chunk = proto::GenerateResponse {
            response: Some(proto::generate_response::Response::Chunk(
                proto::GenerateStreamChunk {
                    token_ids: ids,
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        if let Some(response) =
            convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer).unwrap()
        {
            text.push_str(response.choices[0].delta.content.as_deref().unwrap());
        }
    }
    let complete = proto::GenerateResponse {
        response: Some(proto::generate_response::Response::Complete(
            proto::GenerateComplete {
                finish_reason: "stop".into(),
                ..Default::default()
            },
        )),
        ..Default::default()
    };
    let final_response = convert_proto_chunk_to_openai(complete, &mut handle, &tokenizer)
        .unwrap()
        .unwrap();
    // "é" streamed with its chunk; Complete repeats nothing.
    let tail = final_response.choices[0]
        .delta
        .content
        .as_deref()
        .unwrap_or("");
    assert_eq!(tail, "");
    text.push_str(tail);
    assert_eq!(
        text,
        tokenizer.decode(&[257, 0xC3 + 1, 0xA9 + 1], true).unwrap()
    );
}

#[test]
fn byte_fallback_complete_keeps_interleaved_choices_separate() {
    let tokenizer = byte_fallback_tokenizer();
    let mut handle = tool_test_handle(
        tokenizer.clone(),
        Some(ToolChoice::Value(ToolChoiceValue::None)),
    );
    for (index, byte, expected) in [(0, b't', "t"), (1, b's', "s")] {
        let chunk = proto::GenerateResponse {
            response: Some(proto::generate_response::Response::Chunk(
                proto::GenerateStreamChunk {
                    index,
                    token_ids: vec![u32::from(byte) + 1],
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let response = convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer)
            .unwrap()
            .unwrap();
        assert_eq!(response.choices[0].index, index);
        assert_eq!(response.choices[0].delta.content.as_deref(), Some(expected));
    }
    for index in [0, 1] {
        let complete = proto::GenerateResponse {
            response: Some(proto::generate_response::Response::Complete(
                proto::GenerateComplete {
                    index,
                    finish_reason: "stop".into(),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let response = convert_proto_chunk_to_openai(complete, &mut handle, &tokenizer)
            .unwrap()
            .unwrap();
        assert_eq!(response.choices[0].index, index);
        assert!(response.choices[0].delta.content.is_none(), "{response:?}");
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
    }
}

#[test]
fn a_chunk_tool_call_delta_carries_the_text_released_with_it() {
    let tokenizer = byte_fallback_tokenizer();
    // One chunk holds a value that is not a declared call and the declared
    // call after it: the table releases the first as text and the second
    // as deltas in one step, and the client gets both together; the
    // whitespace between the two objects is the syntax's separator and
    // is dropped.
    let released = r#"{"note":"checking"} "#;
    let call = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let token_ids: Vec<u32> = format!("{released}{call}")
        .bytes()
        .map(|byte| u32::from(byte) + 1)
        .collect();
    let completion_tokens = token_ids.len() as u32;
    let mut handle = tool_test_handle(tokenizer.clone(), None);
    let chunk = proto::GenerateResponse {
        response: Some(proto::generate_response::Response::Chunk(
            proto::GenerateStreamChunk {
                token_ids: token_ids.clone(),
                prompt_tokens: 7,
                completion_tokens,
                ..Default::default()
            },
        )),
        ..Default::default()
    };
    let streamed = convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer)
        .unwrap()
        .unwrap();
    let delta = &streamed.choices[0].delta;
    assert_eq!(delta.role.as_deref(), Some("assistant"));
    assert_eq!(
        delta.content.as_deref(),
        Some(released.trim_end()),
        "{streamed:?}"
    );
    let calls = delta.tool_calls.clone().unwrap_or_default();
    assert!(
        !calls.is_empty() && calls.iter().all(|call| call.index == 0),
        "{calls:?}"
    );
    assert_eq!(
        calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );
    let arguments: String = calls
        .iter()
        .filter_map(|call| {
            call.function
                .as_ref()
                .and_then(|function| function.arguments.as_deref())
        })
        .collect();
    let arguments: Value = serde_json::from_str(&arguments).unwrap();
    assert_eq!(arguments, serde_json::json!({"city":"Paris"}));

    // Complete repeats neither the text nor the call.
    let complete = proto::GenerateResponse {
        response: Some(proto::generate_response::Response::Complete(
            proto::GenerateComplete {
                output_ids: token_ids,
                prompt_tokens: 7,
                completion_tokens,
                finish_reason: "stop".into(),
                ..Default::default()
            },
        )),
        ..Default::default()
    };
    let response = convert_proto_chunk_to_openai(complete, &mut handle, &tokenizer)
        .unwrap()
        .unwrap();
    let choice = &response.choices[0];
    assert!(choice.delta.content.is_none(), "{response:?}");
    assert!(choice.delta.tool_calls.is_none(), "{response:?}");
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
}

fn byte_ids(text: &str) -> Vec<u32> {
    text.bytes().map(|byte| u32::from(byte) + 1).collect()
}

fn chunk_of(token_ids: Vec<u32>, completion_tokens: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(proto::generate_response::Response::Chunk(
            proto::GenerateStreamChunk {
                token_ids,
                prompt_tokens: 7,
                completion_tokens,
                ..Default::default()
            },
        )),
        ..Default::default()
    }
}

fn complete_of(output_ids: Vec<u32>) -> proto::GenerateResponse {
    let completion_tokens = output_ids.len() as u32;
    proto::GenerateResponse {
        response: Some(proto::generate_response::Response::Complete(
            proto::GenerateComplete {
                output_ids,
                prompt_tokens: 7,
                completion_tokens,
                finish_reason: "stop".into(),
                ..Default::default()
            },
        )),
        ..Default::default()
    }
}

fn arguments_of(calls: &[ToolCallDelta]) -> Value {
    let arguments: String = calls
        .iter()
        .filter_map(|call| {
            call.function
                .as_ref()
                .and_then(|function| function.arguments.as_deref())
        })
        .collect();
    serde_json::from_str(&arguments).unwrap()
}

#[test]
fn a_calls_opening_split_over_chunks_does_not_leak_into_the_content() {
    let tokenizer = byte_fallback_tokenizer();
    let call = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let mut handle = tool_test_handle(tokenizer.clone(), None);
    let mut calls = Vec::new();
    // One byte per chunk: the parser holds the opening until the name is
    // complete, and nothing of it is content.
    for (sent, byte) in call.bytes().enumerate() {
        let chunk = chunk_of(vec![u32::from(byte) + 1], sent as u32 + 1);
        if let Some(response) =
            convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer).unwrap()
        {
            let delta = &response.choices[0].delta;
            assert!(delta.content.is_none(), "{response:?}");
            calls.extend(delta.tool_calls.clone().unwrap_or_default());
        }
    }
    let response =
        convert_proto_chunk_to_openai(complete_of(byte_ids(call)), &mut handle, &tokenizer)
            .unwrap()
            .unwrap();
    let choice = &response.choices[0];
    assert!(choice.delta.content.is_none(), "{response:?}");
    calls.extend(choice.delta.tool_calls.clone().unwrap_or_default());
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    assert!(calls.iter().all(|call| call.index == 0), "{calls:?}");
    assert_eq!(
        calls[0].function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );
    // The id travels with the name; the deltas continuing the arguments
    // carry none, so a client keeping the latest id keeps the call's.
    assert!(calls[0].id.is_some(), "{calls:?}");
    assert!(calls[1..].iter().all(|call| call.id.is_none()), "{calls:?}");
    assert_eq!(arguments_of(&calls), serde_json::json!({"city":"Paris"}));
}

#[test]
fn text_the_parser_releases_over_two_chunks_reaches_the_client_once() {
    let tokenizer = byte_fallback_tokenizer();
    let first = r#"{"note":"check"#;
    let second = r#"ing"} {"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let mut handle = tool_test_handle(tokenizer.clone(), None);
    // The parser holds a value that may still become a call: nothing is
    // sent, so the value does not come twice once it is released.
    let held = convert_proto_chunk_to_openai(
        chunk_of(byte_ids(first), first.len() as u32),
        &mut handle,
        &tokenizer,
    )
    .unwrap();
    assert!(held.is_none(), "{held:?}");
    let response = convert_proto_chunk_to_openai(
        chunk_of(byte_ids(second), (first.len() + second.len()) as u32),
        &mut handle,
        &tokenizer,
    )
    .unwrap()
    .unwrap();
    let delta = &response.choices[0].delta;
    // The whitespace between the two objects is the syntax's separator,
    // dropped with the value's release.
    assert_eq!(
        delta.content.as_deref(),
        Some(r#"{"note":"checking"}"#),
        "{response:?}"
    );
    let calls = delta.tool_calls.clone().unwrap_or_default();
    assert_eq!(
        calls
            .first()
            .and_then(|call| call.function.as_ref()?.name.as_deref()),
        Some("get_weather"),
        "{calls:?}"
    );
    assert_eq!(arguments_of(&calls), serde_json::json!({"city":"Paris"}));
    let response = convert_proto_chunk_to_openai(
        complete_of(byte_ids(&format!("{first}{second}"))),
        &mut handle,
        &tokenizer,
    )
    .unwrap()
    .unwrap();
    let choice = &response.choices[0];
    assert!(choice.delta.content.is_none(), "{response:?}");
    assert!(choice.delta.tool_calls.is_none(), "{response:?}");
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
}

#[test]
fn text_the_parser_still_holds_at_the_end_is_content_at_complete() {
    let tokenizer = byte_fallback_tokenizer();
    let mut handle = tool_test_handle(tokenizer.clone(), None);
    // A bare brace that never becomes a call is held, then flushed as
    // content when the stream ends.
    let held =
        convert_proto_chunk_to_openai(chunk_of(byte_ids("{"), 1), &mut handle, &tokenizer).unwrap();
    assert!(held.is_none(), "{held:?}");
    let response =
        convert_proto_chunk_to_openai(complete_of(byte_ids("{")), &mut handle, &tokenizer)
            .unwrap()
            .unwrap();
    let choice = &response.choices[0];
    assert_eq!(choice.delta.content.as_deref(), Some("{"), "{response:?}");
    assert!(choice.delta.tool_calls.is_none(), "{response:?}");
    assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
}

/// The converter the chat path builds from the request's own values is
/// the one the C entry point builds from the request's JSON strings: the
/// tools (request-level and message-declared), the tool choice and the
/// stops.
#[test]
fn the_chat_converter_matches_the_c_entry_point() {
    use crate::tool_constraint::effective_tools_json;

    let tokenizer = byte_fallback_tokenizer();
    let requests = [
        serde_json::json!({
            "model": "kimi-k3",
            "messages": [
                {"role": "system", "content": "", "tools": [{"type": "function", "function":
                    {"name": "get_time", "parameters": {"type": "object"}}}]},
                {"role": "user", "content": "hi"}
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters":
                {"type": "object", "properties": {"city": {"type": "string"}}}}}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "stop": ["\n", "END"],
            "stop_token_ids": [1, 2],
            "skip_special_tokens": false
        }),
        serde_json::json!({
            "model": "plain",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "none",
            "stop": "x"
        }),
    ];
    for request in requests {
        let request: ChatCompletionRequest = serde_json::from_value(request).unwrap();
        let typed = GrpcResponseConverterHandle::for_chat(
            tokenizer.clone(),
            &request,
            "chatcmpl-1".into(),
            "",
        );
        // The strings the entry points used to hand the C entry point.
        let json_arg = |value: Option<String>| value.map(|s| CString::new(s).unwrap());
        let tools = json_arg(effective_tools_json(&request));
        let tool_choice = json_arg(
            request
                .tool_choice
                .as_ref()
                .map(|choice| serde_json::to_string(choice).unwrap()),
        );
        let stop = json_arg(
            request
                .stop
                .as_ref()
                .map(|s| serde_json::to_string(s).unwrap()),
        );
        let stop_token_ids = json_arg(
            request
                .stop_token_ids
                .as_ref()
                .map(|ids| serde_json::to_string(ids).unwrap()),
        );
        let as_ptr = |arg: &Option<CString>| arg.as_ref().map_or(ptr::null(), |s| s.as_ptr());
        let mut tokenizer_handle = TokenizerHandle {
            tokenizer: tokenizer.clone(),
        };
        let model = CString::new(request.model.clone()).unwrap();
        let request_id = CString::new("chatcmpl-1").unwrap();
        // SAFETY: every pointer is a live local or null where the entry
        // point allows null; the handle it returns is taken back once.
        let through_c = unsafe {
            let raw = sgl_grpc_response_converter_create(
                &mut tokenizer_handle,
                model.as_ptr(),
                request_id.as_ptr(),
                as_ptr(&tools),
                as_ptr(&tool_choice),
                as_ptr(&stop),
                as_ptr(&stop_token_ids),
                c_int::from(request.skip_special_tokens),
                0,
                ptr::null(),
                ptr::null_mut(),
            );
            assert!(!raw.is_null());
            *Box::from_raw(raw)
        };
        let view = |handle: &GrpcResponseConverterHandle| {
            serde_json::json!({
                "table": handle.reading.table(),
                "prompt": handle.prompt,
                "stop": handle.stop,
                "stop_token_ids": handle.stop_token_ids,
                "model": handle.model,
                "request_id": handle.request_id,
                "skip_special_tokens": handle.skip_special_tokens,
                "initial_prompt_tokens": handle.initial_prompt_tokens,
                "readers": handle.readers.len(),
            })
        };
        assert_eq!(view(&typed), view(&through_c), "{}", request.model);
        assert_eq!(typed.reading, through_c.reading, "{}", request.model);
        assert_eq!(
            typed.reading.table(),
            if request.model == "kimi-k3" {
                "forced"
            } else {
                "plain"
            },
            "{}",
            request.model
        );
    }
}

// ---- table over the chunk shapes (expected values recorded on main) ----

/// A handle over the byte-fallback tokenizer with the given stop, no
/// tools, and the request's prompt count when known.
fn plain_handle(
    tokenizer: Arc<dyn Tokenizer>,
    stop: Option<&str>,
    initial_prompt_tokens: Option<u32>,
) -> GrpcResponseConverterHandle {
    GrpcResponseConverterHandle {
        tokenizer,
        readers: HashMap::new(),
        stop: stop.map(|stop| StringOrArray::Array(vec![stop.into()])),
        stop_token_ids: None,
        model: "byte-fallback-test".into(),
        request_id: "table".into(),
        created: 0,
        system_fingerprint: None,
        reading: Reading::of("byte-fallback-test", None, None, true),
        prompt: String::new(),
        stream_state: StreamStateManager::new(),
        initial_prompt_tokens,
        skip_special_tokens: true,
    }
}

/// The tool handle, whose reader's call ids count from `NONCE`, so a
/// recorded delta compares whole.
fn recorded_tool_handle(
    tokenizer: Arc<dyn Tokenizer>,
    tool_choice: Option<ToolChoice>,
) -> GrpcResponseConverterHandle {
    let mut handle = tool_test_handle(tokenizer, tool_choice);
    handle.model = "recorded-model".into();
    handle.request_id = "table".into();
    handle
}

fn chunk_at(
    index: u32,
    token_ids: Vec<u32>,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(proto::generate_response::Response::Chunk(
            proto::GenerateStreamChunk {
                index,
                token_ids,
                prompt_tokens,
                completion_tokens,
                ..Default::default()
            },
        )),
        ..Default::default()
    }
}

fn complete_at(
    index: u32,
    finish_reason: &str,
    complete: proto::GenerateComplete,
) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(proto::generate_response::Response::Complete(
            proto::GenerateComplete {
                index,
                finish_reason: finish_reason.into(),
                ..complete
            },
        )),
        ..Default::default()
    }
}

/// Every message of a case through the converter, as the JSON the client
/// reads (`null` for a message that produced no chunk).
fn converted(
    handle: &mut GrpcResponseConverterHandle,
    messages: Vec<proto::GenerateResponse>,
) -> Vec<Value> {
    let tokenizer = Arc::clone(&handle.tokenizer);
    let mut out = Vec::new();
    for message in messages {
        let response = convert_proto_chunk_to_openai(message, handle, &tokenizer).unwrap();
        out.push(response.map_or(Value::Null, |r| serde_json::to_value(r).unwrap()));
    }
    out
}

fn table_cases(
    tokenizer: &Arc<dyn Tokenizer>,
) -> Vec<(
    &'static str,
    GrpcResponseConverterHandle,
    Vec<proto::GenerateResponse>,
)> {
    let call = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let e_acute = vec![0xC3 + 1, 0xA9 + 1];
    vec![
        (
            "a character held until its bytes are complete, then the choice ends",
            plain_handle(tokenizer.clone(), None, None),
            vec![
                chunk_at(0, vec![0xC3 + 1], 7, 1),
                chunk_at(0, vec![0xA9 + 1], 7, 2),
                complete_at(0, "stop", proto::GenerateComplete { output_ids: e_acute.clone(), prompt_tokens: 7, completion_tokens: 2, ..Default::default() }),
            ],
        ),
        (
            "an empty chunk keeps its counts; the engine's empty finish reason is stop and the output ids count",
            plain_handle(tokenizer.clone(), None, None),
            vec![
                chunk_at(0, vec![], 7, 0),
                complete_at(0, "", proto::GenerateComplete { output_ids: vec![1, 2, 3], ..Default::default() }),
            ],
        ),
        (
            "a choice that never streamed: the request's prompt count, a blank finish reason",
            plain_handle(tokenizer.clone(), None, Some(11)),
            vec![complete_at(0, "   ", proto::GenerateComplete::default())],
        ),
        (
            "a stop decoder that finds no stop: the engine's finish reason, its matched token and the usage details",
            plain_handle(tokenizer.clone(), Some("NEVER"), None),
            vec![
                chunk_at(0, byte_ids("t"), 7, 1),
                complete_at(0, "length", proto::GenerateComplete { output_ids: byte_ids("t"), prompt_tokens: 7, completion_tokens: 1, cached_tokens: 3, reasoning_tokens: 4, matched_stop: Some(MatchedStop::MatchedTokenId(99)), ..Default::default() }),
            ],
        ),
        (
            "a stop the decoder finds beats the engine's finish reason and matched stop",
            plain_handle(tokenizer.clone(), Some("é"), None),
            vec![
                chunk_at(0, e_acute.clone(), 7, 2),
                complete_at(0, "length", proto::GenerateComplete { output_ids: e_acute.clone(), prompt_tokens: 7, completion_tokens: 2, matched_stop: Some(MatchedStop::MatchedTokenId(99)), ..Default::default() }),
            ],
        ),
        (
            "the engine's matched stop string",
            plain_handle(tokenizer.clone(), Some("NEVER"), None),
            vec![complete_at(0, "stop", proto::GenerateComplete { matched_stop: Some(MatchedStop::MatchedStopStr("END".into())), ..Default::default() })],
        ),
        (
            "a call in one chunk: deltas with the call, the choice finishes on tool_calls",
            recorded_tool_handle(tokenizer.clone(), None),
            vec![
                chunk_at(0, byte_ids(call), 7, call.len() as u32),
                complete_at(0, "stop", proto::GenerateComplete { output_ids: byte_ids(call), prompt_tokens: 7, completion_tokens: call.len() as u32, ..Default::default() }),
            ],
        ),
        (
            "a call under a length finish keeps the engine's reason",
            recorded_tool_handle(tokenizer.clone(), None),
            vec![
                chunk_at(0, byte_ids(call), 7, call.len() as u32),
                complete_at(0, "length", proto::GenerateComplete { output_ids: byte_ids(call), prompt_tokens: 7, completion_tokens: call.len() as u32, ..Default::default() }),
            ],
        ),
        (
            "a brace the parser holds to the end is content at Complete",
            recorded_tool_handle(tokenizer.clone(), None),
            vec![
                chunk_at(0, byte_ids("{"), 7, 1),
                complete_at(0, "stop", proto::GenerateComplete { output_ids: byte_ids("{"), prompt_tokens: 7, completion_tokens: 1, ..Default::default() }),
            ],
        ),
        (
            "tool choice none: the call's text is content",
            recorded_tool_handle(tokenizer.clone(), Some(ToolChoice::Value(ToolChoiceValue::None))),
            vec![
                chunk_at(0, byte_ids(call), 7, call.len() as u32),
                complete_at(0, "stop", proto::GenerateComplete { output_ids: byte_ids(call), prompt_tokens: 7, completion_tokens: call.len() as u32, ..Default::default() }),
            ],
        ),
        (
            "a second choice has its own index and state",
            plain_handle(tokenizer.clone(), None, None),
            vec![
                chunk_at(1, byte_ids("s"), 7, 1),
                complete_at(1, "stop", proto::GenerateComplete { output_ids: byte_ids("s"), prompt_tokens: 7, completion_tokens: 1, ..Default::default() }),
            ],
        ),
        (
            "a message without a body",
            plain_handle(tokenizer.clone(), None, None),
            vec![proto::GenerateResponse::default()],
        ),
    ]
}

/// The chunks each case produced, recorded on the converter before its split
/// (`null` where a message produced no chunk), in the order of `table_cases`.
/// Re-recorded when the Symphony tables took over the reading: the model
/// name and the call ids (the table's, counting from `NONCE`) are the only
/// changes; every delta's shape stood.
fn recorded_chunks() -> Vec<Vec<Value>> {
    vec![
        // a character held until its bytes are complete, then the choice ends
        vec![
            serde_json::json!(null),
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"byte-fallback-test","choices":[{"index":0,"delta":{"role":"assistant","content":"é"},"logprobs":null,"finish_reason":null}]}),
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"byte-fallback-test","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9,"prompt_tokens_details":{"cached_tokens":0}}}),
        ],
        // an empty chunk keeps its counts; the engine's empty finish reason is stop and the output ids count
        vec![
            serde_json::json!(null),
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"byte-fallback-test","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10,"prompt_tokens_details":{"cached_tokens":0}}}),
        ],
        // a choice that never streamed: the request's prompt count, a blank finish reason
        vec![
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"byte-fallback-test","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":0,"total_tokens":11,"prompt_tokens_details":{"cached_tokens":0}}}),
        ],
        // a stop decoder that finds no stop: the engine's finish reason, its matched token and the usage details
        vec![
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"byte-fallback-test","choices":[{"index":0,"delta":{"role":"assistant","content":"t"},"logprobs":null,"finish_reason":null}]}),
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"byte-fallback-test","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":"length","matched_stop":99}],"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8,"prompt_tokens_details":{"cached_tokens":3},"completion_tokens_details":{"reasoning_tokens":4}}}),
        ],
        // a stop the decoder finds beats the engine's finish reason and matched stop
        vec![
            serde_json::json!(null),
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"byte-fallback-test","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":"stop","matched_stop":"é"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9,"prompt_tokens_details":{"cached_tokens":0}}}),
        ],
        // the engine's matched stop string
        vec![
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"byte-fallback-test","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":"stop","matched_stop":"END"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"prompt_tokens_details":{"cached_tokens":0}}}),
        ],
        // a call in one chunk: deltas with the call, the choice finishes on tool_calls
        vec![
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"recorded-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_0123456789abcdef01234567","type":"function","function":{"name":"get_weather"}},{"index":0,"function":{"arguments":"{\"city\":\"Paris\"}"}}]},"logprobs":null,"finish_reason":null}]}),
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"recorded-model","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":7,"completion_tokens":51,"total_tokens":58,"prompt_tokens_details":{"cached_tokens":0}}}),
        ],
        // a call under a length finish keeps the engine's reason
        vec![
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"recorded-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_0123456789abcdef01234567","type":"function","function":{"name":"get_weather"}},{"index":0,"function":{"arguments":"{\"city\":\"Paris\"}"}}]},"logprobs":null,"finish_reason":null}]}),
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"recorded-model","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":7,"completion_tokens":51,"total_tokens":58,"prompt_tokens_details":{"cached_tokens":0}}}),
        ],
        // a brace the parser holds to the end is content at Complete
        vec![
            serde_json::json!(null),
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"recorded-model","choices":[{"index":0,"delta":{"role":"assistant","content":"{"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8,"prompt_tokens_details":{"cached_tokens":0}}}),
        ],
        // tool choice none: the call's text is content
        vec![
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"recorded-model","choices":[{"index":0,"delta":{"role":"assistant","content":"{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}"},"logprobs":null,"finish_reason":null}]}),
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"recorded-model","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":51,"total_tokens":58,"prompt_tokens_details":{"cached_tokens":0}}}),
        ],
        // a second choice has its own index and state
        vec![
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"byte-fallback-test","choices":[{"index":1,"delta":{"role":"assistant","content":"s"},"logprobs":null,"finish_reason":null}]}),
            serde_json::json!({"id":"table","object":"chat.completion.chunk","created":0,"model":"byte-fallback-test","choices":[{"index":1,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8,"prompt_tokens_details":{"cached_tokens":0}}}),
        ],
        // a message without a body
        vec![serde_json::json!(null)],
    ]
}

/// Every shape a stream's messages take, through the converter and
/// compared whole, as the JSON the client reads, with what the converter
/// produced before its split: a character held across chunks, counts and
/// their fallbacks, the finish reasons, a stop found locally against the
/// engine's, a matched token id or string, a call released in one chunk,
/// a brace held to the end, tool choice none, a second choice, an empty
/// message.
#[test]
fn the_chunk_shapes_convert_as_recorded() {
    let tokenizer = byte_fallback_tokenizer();
    let cases = table_cases(&tokenizer);
    let recorded = recorded_chunks();
    assert_eq!(cases.len(), recorded.len());
    let mut differ = Vec::new();
    for ((name, mut handle, messages), expected) in cases.into_iter().zip(recorded) {
        let produced = converted(&mut handle, messages);
        if produced != expected {
            differ.push(format!("{name}: {produced:?} != {expected:?}"));
        }
    }
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}
