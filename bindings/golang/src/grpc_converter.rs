//! gRPC response converter FFI functions

use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int},
    ptr,
    sync::Arc,
};

use llm_tokenizer::{stop::SequenceDecoderOutput, stream::DecodeStream, traits::Tokenizer};
use openai_protocol::common::{
    FunctionCallDelta, StringOrArray, Tool, ToolCallDelta, ToolChoice, ToolChoiceValue, Usage,
};
use serde_json::Value;
use smg::routers::grpc::utils::create_stop_decoder;
use smg_grpc_client::sglang_proto as proto;
use tokio::sync::Mutex as TokioMutex;
use tool_parser::ToolParser;

use super::{
    error::{clear_error_message, set_error_message, SglErrorCode},
    proto_parse::parse_proto_response,
    runtime::{PARSER_FACTORY, RUNTIME},
    stream_state::StreamStateManager,
    tokenizer::TokenizerHandle,
    utils::generate_tool_call_id,
};

/// Handle for gRPC response converter (maintains state for streaming)
#[repr(C)]
pub struct GrpcResponseConverterHandle {
    pub(crate) tokenizer: Arc<dyn Tokenizer>,
    pub(crate) tool_parser: Option<Arc<TokioMutex<Box<dyn ToolParser>>>>,
    pub(crate) stop: Option<StringOrArray>,
    pub(crate) stop_token_ids: Option<Vec<u32>>,
    pub(crate) model: String,
    pub(crate) request_id: String,
    pub(crate) created: u64,
    pub(crate) system_fingerprint: Option<String>,
    pub(crate) tools: Option<Vec<Tool>>,
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) history_tool_calls_count: usize,
    pub(crate) stream_state: StreamStateManager,
    pub(crate) initial_prompt_tokens: Option<u32>,
    pub(crate) skip_special_tokens: bool,
}

/// Create a gRPC response converter handle
///
/// # Arguments
/// * `tokenizer_handle` - Tokenizer handle (must be valid)
/// * `model` - Model name
/// * `request_id` - Request ID
/// * `tools_json` - Optional JSON array of tools
/// * `tool_choice_json` - Optional JSON object for tool_choice
/// * `stop` - Optional stop sequences (JSON array)
/// * `stop_token_ids` - Optional stop token IDs (JSON array)
/// * `skip_special_tokens` - Whether to skip special tokens
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * Pointer to GrpcResponseConverterHandle on success, null on failure
///
/// # Safety
/// - `tokenizer_handle` must be a valid pointer returned by `sgl_tokenizer_create`
/// - `model` and `request_id` must be valid null-terminated C strings
/// - `tools_json`, `tool_choice_json`, `stop`, `stop_token_ids` may be null
/// - `error_out` may be null; if non-null, must point to writable memory
/// - Caller owns the returned handle and must free it with `sgl_grpc_response_converter_free`
#[no_mangle]
pub unsafe extern "C" fn sgl_grpc_response_converter_create(
    tokenizer_handle: *mut TokenizerHandle,
    model: *const c_char,
    request_id: *const c_char,
    tools_json: *const c_char,
    tool_choice_json: *const c_char,
    stop: *const c_char,
    stop_token_ids: *const c_char,
    skip_special_tokens: c_int,
    error_out: *mut *mut c_char,
) -> *mut GrpcResponseConverterHandle {
    if tokenizer_handle.is_null() || model.is_null() || request_id.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return ptr::null_mut();
    }

    let model_str = match CStr::from_ptr(model).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in model");
            return ptr::null_mut();
        }
    };

    let request_id_str = match CStr::from_ptr(request_id).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in request_id");
            return ptr::null_mut();
        }
    };

    let handle_ref = &*tokenizer_handle;
    let tokenizer = Arc::clone(&handle_ref.tokenizer);

    // Parse tools if provided
    let tools: Option<Vec<Tool>> = if tools_json.is_null() {
        None
    } else {
        match CStr::from_ptr(tools_json).to_str() {
            Ok(s) => serde_json::from_str::<Vec<Tool>>(s).ok(),
            Err(_) => None,
        }
    };

    // Parse tool_choice if provided
    let tool_choice: Option<ToolChoice> = if tool_choice_json.is_null() {
        None
    } else {
        match CStr::from_ptr(tool_choice_json).to_str() {
            Ok(s) => serde_json::from_str::<ToolChoice>(s).ok(),
            Err(_) => None,
        }
    };

    // Parse stop sequences
    let stop: Option<StringOrArray> = if stop.is_null() {
        None
    } else {
        let stop_str = match CStr::from_ptr(stop).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };
        serde_json::from_str::<StringOrArray>(stop_str).ok()
    };

    // Parse stop token IDs
    let stop_token_ids: Option<Vec<u32>> = if stop_token_ids.is_null() {
        None
    } else {
        let ids_str = match CStr::from_ptr(stop_token_ids).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };
        serde_json::from_str::<Vec<u32>>(ids_str).ok()
    };

    // Create tool parser if tools are provided
    let tool_parser = if tools.is_some() {
        PARSER_FACTORY
            .registry()
            .create_for_model(model_str)
            .map(|p| Arc::new(TokioMutex::new(p)))
    } else {
        None
    };

    // Get system fingerprint from model (simplified)
    let system_fingerprint = Some("fp_placeholder".to_string()); // TODO: Get actual fingerprint

    Box::into_raw(Box::new(GrpcResponseConverterHandle {
        tokenizer,
        tool_parser,
        stop,
        stop_token_ids,
        model: model_str.to_string(),
        request_id: request_id_str.to_string(),
        // unwrap_or_default is acceptable here: if the clock is before UNIX epoch,
        // the `created` field in the API response will be 0, which is cosmetic
        // and does not cause data corruption or silent data loss.
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        system_fingerprint,
        tools,
        tool_choice,
        history_tool_calls_count: 0,
        stream_state: StreamStateManager::new(),
        initial_prompt_tokens: None,
        skip_special_tokens: skip_special_tokens != 0,
    }))
}

/// Convert a gRPC GenerateResponse chunk to OpenAI format
///
/// # Arguments
/// * `handle` - Converter handle
/// * `response_json` - JSON string of proto.GenerateResponse
/// * `result_json_out` - Pointer to receive OpenAI format JSON (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_grpc_response_converter_create`
/// - `response_json` must be a valid null-terminated C string containing valid JSON
/// - `result_json_out` must be a valid pointer to writable memory
/// - `error_out` may be null; if non-null, must point to writable memory
/// - Caller must free the string written to `result_json_out` using `sgl_free_string`
#[no_mangle]
pub unsafe extern "C" fn sgl_grpc_response_converter_convert_chunk(
    handle: *mut GrpcResponseConverterHandle,
    response_json: *const c_char,
    result_json_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || response_json.is_null() || result_json_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let response_str = match CStr::from_ptr(response_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in response_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse proto.GenerateResponse from JSON using shared module
    let json_value: Value = match serde_json::from_str(response_str) {
        Ok(v) => v,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse response JSON: {e}"));
            return SglErrorCode::ParsingError;
        }
    };

    let proto_response = match parse_proto_response(&json_value) {
        Ok(r) => r,
        Err(e) => {
            set_error_message(error_out, e);
            return SglErrorCode::ParsingError;
        }
    };

    let handle_ref = &mut *handle;
    let tokenizer = Arc::clone(&handle_ref.tokenizer);

    // Use tokio runtime to run async code
    let result = RUNTIME.block_on(async {
        convert_proto_chunk_to_openai(proto_response, handle_ref, &tokenizer).await
    });

    match result {
        Ok(Some(openai_response)) => {
            // Serialize to JSON
            let result_str = match serde_json::to_string(&openai_response) {
                Ok(s) => s,
                Err(e) => {
                    set_error_message(error_out, &format!("Failed to serialize response: {e}"));
                    return SglErrorCode::ParsingError;
                }
            };

            let result_cstr = match CString::new(result_str) {
                Ok(s) => s,
                Err(e) => {
                    set_error_message(error_out, &format!("Failed to create result string: {e}"));
                    return SglErrorCode::MemoryError;
                }
            };

            *result_json_out = result_cstr.into_raw();
            clear_error_message(error_out);
            SglErrorCode::Success
        }
        Ok(None) => {
            // No response to send (e.g., empty chunk)
            let empty = CString::default();
            *result_json_out = empty.into_raw();
            clear_error_message(error_out);
            SglErrorCode::Success
        }
        Err(e) => {
            set_error_message(error_out, &format!("Conversion error: {e}"));
            SglErrorCode::ParsingError
        }
    }
}

/// Parse new text with the same tool-call and raw-text fallback rules at Chunk and Complete.
async fn parse_tool_text(
    handle: &GrpcResponseConverterHandle,
    text: &str,
) -> Option<(String, Vec<ToolCallDelta>)> {
    if text.is_empty()
        || matches!(
            handle.tool_choice.as_ref(),
            Some(ToolChoice::Value(ToolChoiceValue::None))
        )
    {
        return None;
    }
    let (Some(tools), Some(tool_parser)) = (handle.tools.as_ref(), handle.tool_parser.as_ref())
    else {
        return None;
    };
    let mut parser = tool_parser.lock().await;
    let result = match parser.parse_incremental(text, tools).await {
        Ok(result) => result,
        Err(error) => {
            // Preserve the converter's existing fallback to regular content.
            tracing::warn!("Tool parser error: {}", error);
            return None;
        }
    };
    if result.calls.is_empty() {
        return None;
    }
    Some((
        result.normal_text,
        result
            .calls
            .into_iter()
            .map(|item| {
                let id = if let Some(ref name) = item.name {
                    generate_tool_call_id(
                        &handle.model,
                        name,
                        item.tool_index,
                        handle.history_tool_calls_count,
                    )
                } else {
                    format!("call_{}", item.tool_index)
                };
                ToolCallDelta {
                    index: item.tool_index as u32,
                    id: Some(id),
                    tool_type: item.name.as_ref().map(|_| "function".to_string()),
                    function: Some(FunctionCallDelta {
                        name: item.name,
                        arguments: (!item.parameters.is_empty()).then_some(item.parameters),
                    }),
                }
            })
            .collect(),
    ))
}

/// Helper function to convert proto chunk to OpenAI format
pub(crate) async fn convert_proto_chunk_to_openai(
    proto_response: proto::GenerateResponse,
    handle: &mut GrpcResponseConverterHandle,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Result<Option<openai_protocol::chat::ChatCompletionStreamResponse>, String> {
    use openai_protocol::chat::{ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice};
    use smg_grpc_client::sglang_proto::generate_response::Response::*;

    match proto_response.response {
        Some(Chunk(chunk)) => {
            let index = chunk.index;
            let state = handle.stream_state.get_or_create(index);

            // Mark as not first chunk if we've seen this index before
            let first_chunk = state.is_first_chunk;
            state.is_first_chunk = false;

            // Track token counts (cumulative values from proto)
            if chunk.prompt_tokens > 0 {
                state.prompt_tokens = chunk.prompt_tokens;
            } else if state.prompt_tokens == 0 {
                if let Some(initial_prompt) = handle.initial_prompt_tokens {
                    state.prompt_tokens = initial_prompt;
                }
            }
            state.completion_tokens = chunk.completion_tokens;

            // Process tokens through stop decoder if available, otherwise use incremental decoder
            if state.stop_decoder.is_none()
                && (handle.stop.is_some() || handle.stop_token_ids.is_some())
            {
                state.stop_decoder = Some(create_stop_decoder(
                    tokenizer,
                    handle.stop.as_ref(),
                    handle.stop_token_ids.as_ref(),
                    handle.skip_special_tokens,
                    false,
                    false,
                ));
            }
            let chunk_text = if let Some(decoder_guard) = &mut state.stop_decoder {
                let mut text = String::new();
                for &token_id in &chunk.token_ids {
                    match decoder_guard
                        .process_token(token_id)
                        .unwrap_or(SequenceDecoderOutput::Held)
                    {
                        SequenceDecoderOutput::Text(t) => {
                            text.push_str(&t);
                        }
                        SequenceDecoderOutput::StoppedWithText(t) => {
                            text.push_str(&t);
                            break;
                        }
                        SequenceDecoderOutput::Stopped => {
                            break;
                        }
                        SequenceDecoderOutput::Held => {}
                    }
                }
                text
            } else {
                // Use incremental decoder to handle multi-byte character boundaries
                let skip_special = handle.skip_special_tokens;
                let state = handle.stream_state.get_or_create(index);
                if state.decode_stream.is_none() {
                    state.decode_stream =
                        Some(DecodeStream::new(tokenizer.clone(), &[], skip_special));
                }

                // Process tokens incrementally
                let mut text_parts = Vec::new();
                if let Some(ref mut decode_stream) = state.decode_stream {
                    for &token_id in &chunk.token_ids {
                        if let Ok(Some(text)) = decode_stream.step(token_id) {
                            text_parts.push(text);
                        }
                    }
                }
                text_parts.join("")
            };

            if chunk_text.is_empty() {
                return Ok(None);
            }

            // Send first chunk with role
            if first_chunk {
                let first_response = ChatCompletionStreamResponse {
                    id: handle.request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: handle.created,
                    model: handle.model.clone(),
                    system_fingerprint: handle.system_fingerprint.clone(),
                    choices: vec![ChatStreamChoice {
                        index,
                        delta: ChatMessageDelta {
                            role: Some("assistant".to_string()),
                            content: None,
                            tool_calls: None,
                            reasoning_content: None,
                        },
                        logprobs: None,
                        finish_reason: None,
                        matched_stop: None,
                    }],
                    usage: None,
                };
                return Ok(Some(first_response));
            }

            if let Some((_, tool_call_deltas)) = parse_tool_text(handle, &chunk_text).await {
                handle.stream_state.get_or_create(index).has_tool_calls = true;
                let tool_response = ChatCompletionStreamResponse {
                    id: handle.request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: handle.created,
                    model: handle.model.clone(),
                    system_fingerprint: handle.system_fingerprint.clone(),
                    choices: vec![ChatStreamChoice {
                        index,
                        delta: ChatMessageDelta {
                            role: Some("assistant".to_string()),
                            content: None,
                            tool_calls: Some(tool_call_deltas),
                            reasoning_content: None,
                        },
                        logprobs: None,
                        finish_reason: None,
                        matched_stop: None,
                    }],
                    usage: None,
                };
                return Ok(Some(tool_response));
            }

            // Regular content emission
            let content_response = ChatCompletionStreamResponse {
                id: handle.request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: handle.created,
                model: handle.model.clone(),
                system_fingerprint: handle.system_fingerprint.clone(),
                choices: vec![ChatStreamChoice {
                    index,
                    delta: ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: Some(chunk_text),
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                    finish_reason: None,
                    matched_stop: None,
                }],
                usage: None,
            };

            Ok(Some(content_response))
        }
        Some(Complete(complete)) => {
            let index = complete.index;
            let mut removed_state = handle.stream_state.remove(index);
            let (mut pending_text, stopped_at_finish, local_matched_stop) = if let Some(decoder) =
                removed_state
                    .as_mut()
                    .and_then(|state| state.stop_decoder.as_mut())
            {
                let tail = decoder
                    .finish()
                    .map_err(|error| format!("Failed to finish token decoding: {error}"))?;
                if decoder.prefix_mismatches() > 0 {
                    tracing::warn!(
                        request_id = %handle.request_id,
                        index,
                        recoveries = decoder.prefix_mismatches(),
                        "the decoder rewrote text already streamed; the stream kept its text and sent the remainder"
                    );
                }
                // A stop the decoder found at an earlier chunk (text streams as soon
                // as its characters are complete) counts the same as one in the tail.
                let stopped = decoder.is_stopped();
                let matched = stopped
                    .then(|| {
                        decoder
                            .matched_stop()
                            .map(|stop| Value::String(stop.to_string()))
                    })
                    .flatten();
                let text = match tail {
                    SequenceDecoderOutput::Text(text)
                    | SequenceDecoderOutput::StoppedWithText(text) => text,
                    _ => String::new(),
                };
                (text, stopped, matched)
            } else {
                (String::new(), false, None)
            };

            let (has_tool_calls, state_prompt_tokens, state_completion_tokens) =
                if let Some(mut state) = removed_state {
                    // Flush decode stream
                    if let Some(ref mut decode_stream) = state.decode_stream {
                        if let Ok(Some(remaining)) = decode_stream.flush() {
                            pending_text.push_str(&remaining);
                        }
                    }
                    (
                        state.has_tool_calls,
                        state.prompt_tokens,
                        state.completion_tokens,
                    )
                } else {
                    (false, 0, 0)
                };

            // Only newly decoded text reaches the parser or the client here.
            let (final_text, eof_tool_calls) = match parse_tool_text(handle, &pending_text).await {
                Some((normal_text, calls)) => (normal_text, Some(calls)),
                None => (pending_text, None),
            };
            let has_tool_calls = has_tool_calls || eof_tool_calls.is_some();

            // Determine finish reason
            let engine_finish_reason = if stopped_at_finish {
                "stop"
            } else {
                complete.finish_reason.as_str()
            };
            let finish_reason = if has_tool_calls
                && (engine_finish_reason == "stop" || engine_finish_reason.is_empty())
            {
                "tool_calls".to_string()
            } else if engine_finish_reason.is_empty() || engine_finish_reason.trim().is_empty() {
                "stop".to_string()
            } else {
                engine_finish_reason.to_string()
            };

            // Extract matched_stop
            let matched_stop = local_matched_stop.or_else(|| match &complete.matched_stop {
                Some(proto::generate_complete::MatchedStop::MatchedTokenId(token_id)) => {
                    Some(Value::Number(serde_json::Number::from(*token_id)))
                }
                Some(proto::generate_complete::MatchedStop::MatchedStopStr(stop_str)) => {
                    Some(Value::String(stop_str.clone()))
                }
                None => None,
            });

            // Build usage from state or complete message
            let mut prompt_tokens = if state_prompt_tokens > 0 {
                state_prompt_tokens
            } else if complete.prompt_tokens > 0 {
                complete.prompt_tokens
            } else {
                handle.initial_prompt_tokens.unwrap_or(0)
            };

            let completion_tokens = if state_completion_tokens > 0 {
                state_completion_tokens
            } else if complete.completion_tokens > 0 {
                complete.completion_tokens
            } else if !complete.output_ids.is_empty() {
                complete.output_ids.len() as u32
            } else {
                0
            };

            // Final fallback for prompt_tokens
            if prompt_tokens == 0 {
                if let Some(initial_prompt) = handle.initial_prompt_tokens {
                    prompt_tokens = initial_prompt;
                }
            }

            // Always create usage, even if values are 0 (defensive)
            let usage = Some(
                Usage::from_counts(prompt_tokens, completion_tokens)
                    .with_cached_tokens(complete.cached_tokens)
                    .with_reasoning_tokens(complete.reasoning_tokens),
            );

            let finish_response = ChatCompletionStreamResponse {
                id: handle.request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: handle.created,
                model: handle.model.clone(),
                system_fingerprint: handle.system_fingerprint.clone(),
                choices: vec![ChatStreamChoice {
                    index,
                    delta: ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: if final_text.is_empty() {
                            None
                        } else {
                            Some(final_text)
                        },
                        tool_calls: eof_tool_calls,
                        reasoning_content: None,
                    },
                    logprobs: None,
                    finish_reason: Some(finish_reason),
                    matched_stop,
                }],
                usage,
            };

            Ok(Some(finish_response))
        }
        None => Ok(None),
    }
}

/// Free a gRPC response converter handle
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_grpc_response_converter_create`, or null
/// - `handle` must not be used after this call
/// - This function must not be called more than once for the same handle
#[no_mangle]
pub unsafe extern "C" fn sgl_grpc_response_converter_free(
    handle: *mut GrpcResponseConverterHandle,
) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPLIT_TOOL_PREFIX: &str = r#"{"name":"get_weather","arguments":{"city":"#;

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
        let tokenizer =
            llm_tokenizer::HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(path).unwrap();
        Arc::new(tokenizer)
    }

    #[tokio::test]
    async fn byte_fallback_complete_releases_tail_and_enforces_stop() {
        let tokenizer = byte_fallback_tokenizer();
        for stop in ["NEVER", "é"] {
            let mut handle = GrpcResponseConverterHandle {
                tokenizer: tokenizer.clone(),
                tool_parser: None,
                stop: Some(StringOrArray::Array(vec![stop.into()])),
                stop_token_ids: None,
                model: "byte-fallback-test".into(),
                request_id: "test".into(),
                created: 0,
                system_fingerprint: None,
                tools: None,
                tool_choice: None,
                history_tool_calls_count: 0,
                stream_state: StreamStateManager::new(),
                initial_prompt_tokens: None,
                skip_special_tokens: true,
            };
            handle.stream_state.get_or_create(0).is_first_chunk = false;
            let chunk = proto::GenerateResponse {
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
            let streamed = convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer)
                .await
                .unwrap();
            if stop == "é" {
                assert!(streamed.is_none(), "{streamed:?}");
            } else {
                let streamed = streamed.unwrap();
                assert_eq!(streamed.choices[0].delta.content.as_deref(), Some("é"));
                assert!(streamed.choices[0].finish_reason.is_none());
            }
            let complete = proto::GenerateResponse {
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
                .await
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
    fn tool_test_handle(
        tokenizer: Arc<dyn Tokenizer>,
        tool_choice: Option<ToolChoice>,
    ) -> GrpcResponseConverterHandle {
        let tools = serde_json::from_value(serde_json::json!([{
            "type":"function", "function":{
                "name":"get_weather", "parameters":{
                    "type":"object", "properties":{"city":{"type":"string"}}
                }
            }
        }]))
        .unwrap();
        GrpcResponseConverterHandle {
            tokenizer,
            tool_parser: Some(Arc::new(TokioMutex::new(
                Box::new(tool_parser::JsonParser::new()) as Box<dyn ToolParser>,
            ))),
            stop: Some(StringOrArray::Array(vec!["NEVER".into()])),
            stop_token_ids: None,
            model: "byte-fallback-test".into(),
            request_id: "tool-tail".into(),
            created: 0,
            system_fingerprint: None,
            tools: Some(tools),
            tool_choice,
            history_tool_calls_count: 0,
            stream_state: StreamStateManager::new(),
            initial_prompt_tokens: None,
            skip_special_tokens: true,
        }
    }

    #[tokio::test]
    async fn byte_fallback_complete_parses_tool_tail_and_respects_none() {
        let tokenizer = byte_fallback_tokenizer();
        let tool_text = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
        let token_ids: Vec<u32> = tool_text.bytes().map(|byte| u32::from(byte) + 1).collect();
        let completion_tokens = token_ids.len() as u32;
        for tool_choice in [None, Some(ToolChoice::Value(ToolChoiceValue::None))] {
            let disabled = tool_choice.is_some();
            let mut handle = tool_test_handle(tokenizer.clone(), tool_choice);
            handle.stream_state.get_or_create(0).is_first_chunk = false;
            let chunk = proto::GenerateResponse {
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
                .await
                .unwrap()
                .unwrap();
            let streamed_choice = &streamed.choices[0];
            assert!(streamed_choice.finish_reason.is_none());
            let complete = proto::GenerateResponse {
                request_id: String::new(),
                response: Some(proto::generate_response::Response::Complete(
                    proto::GenerateComplete {
                        output_ids: token_ids.clone(),
                        prompt_tokens: 7,
                        completion_tokens,
                        cached_tokens: 3,
                        finish_reason: "stop".into(),
                        matched_stop: Some(proto::generate_complete::MatchedStop::MatchedTokenId(
                            99,
                        )),
                        ..Default::default()
                    },
                )),
            };
            let response = convert_proto_chunk_to_openai(complete, &mut handle, &tokenizer)
                .await
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
    #[tokio::test]
    async fn byte_fallback_complete_closed_split_tool_control() {
        let tokenizer = byte_fallback_tokenizer();
        let tail = r#""Paris"}}"#;
        let mut handle = tool_test_handle(tokenizer.clone(), None);
        // Prime the role state directly so the known first-Chunk behavior
        // does not discard any part of the tool prefix used by this fixture.
        handle.stream_state.get_or_create(0).is_first_chunk = false;
        let prefix_chunk = proto::GenerateResponse {
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
            .await
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
            convert_proto_chunk_to_openai(tail_chunk, &mut handle, &tokenizer)
                .await
                .unwrap()
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
            request_id: String::new(),
            response: Some(proto::generate_response::Response::Complete(
                proto::GenerateComplete {
                    output_ids,
                    prompt_tokens: 7,
                    completion_tokens,
                    cached_tokens: 3,
                    finish_reason: "stop".into(),
                    matched_stop: Some(proto::generate_complete::MatchedStop::MatchedTokenId(99)),
                    ..Default::default()
                },
            )),
        };
        let final_response = convert_proto_chunk_to_openai(complete, &mut handle, &tokenizer)
            .await
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
    #[tokio::test]
    async fn byte_fallback_complete_does_not_repeat_emitted_content() {
        let tokenizer = byte_fallback_tokenizer();
        let mut handle = tool_test_handle(
            tokenizer.clone(),
            Some(ToolChoice::Value(ToolChoiceValue::None)),
        );
        handle.stream_state.get_or_create(0).is_first_chunk = false;
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
            if let Some(response) = convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer)
                .await
                .unwrap()
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
            .await
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

    #[tokio::test]
    async fn byte_fallback_complete_keeps_interleaved_choices_separate() {
        let tokenizer = byte_fallback_tokenizer();
        let mut handle = tool_test_handle(
            tokenizer.clone(),
            Some(ToolChoice::Value(ToolChoiceValue::None)),
        );
        for index in [0, 1] {
            handle.stream_state.get_or_create(index).is_first_chunk = false;
        }
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
                .await
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
                .await
                .unwrap()
                .unwrap();
            assert_eq!(response.choices[0].index, index);
            assert!(response.choices[0].delta.content.is_none(), "{response:?}");
            assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
        }
    }
}
