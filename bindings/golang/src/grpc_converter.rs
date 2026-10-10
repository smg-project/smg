//! gRPC response converter FFI functions

use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int},
    ptr,
    sync::Arc,
};

use llm_tokenizer::{
    stop::{SequenceDecoderOutput, StopSequenceDecoder},
    stream::DecodeStream,
    traits::Tokenizer,
};
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
    pub(crate) stop_decoder: Option<Arc<TokioMutex<StopSequenceDecoder>>>,
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

    // Create stop decoder if needed
    let stop_decoder = if stop.is_some() || stop_token_ids.is_some() {
        Some(Arc::new(TokioMutex::new(create_stop_decoder(
            &tokenizer,
            stop.as_ref(),
            stop_token_ids.as_ref(),
            skip_special_tokens != 0,
            false, // no_stop_trim
            false, // ignore_eos
        ))))
    } else {
        None
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
        stop_decoder,
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
            let chunk_text = if let Some(ref stop_decoder) = handle.stop_decoder {
                let mut decoder_guard = stop_decoder.lock().await;
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

            // Update stream buffer
            handle
                .stream_state
                .get_or_create(index)
                .text_buffer
                .push_str(&chunk_text);

            // Handle tool calls if tools are provided
            if let (Some(tools), Some(tool_parser)) =
                (handle.tools.as_ref(), handle.tool_parser.as_ref())
            {
                let tool_choice_enabled = !matches!(
                    handle.tool_choice,
                    Some(ToolChoice::Value(ToolChoiceValue::None))
                );

                if tool_choice_enabled {
                    let mut parser_guard = tool_parser.lock().await;
                    match parser_guard.parse_incremental(&chunk_text, tools).await {
                        Ok(streaming_result) => {
                            if !streaming_result.calls.is_empty() {
                                handle.stream_state.get_or_create(index).has_tool_calls = true;
                                // Convert tool call items to OpenAI format
                                let tool_call_deltas: Vec<_> = streaming_result
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
                                            tool_type: if item.name.is_some() {
                                                Some("function".to_string())
                                            } else {
                                                None
                                            },
                                            function: Some(FunctionCallDelta {
                                                name: item.name,
                                                arguments: if item.parameters.is_empty() {
                                                    None
                                                } else {
                                                    Some(item.parameters)
                                                },
                                            }),
                                        }
                                    })
                                    .collect();

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
                        }
                        Err(e) => {
                            // Log error but continue with regular content
                            tracing::warn!("Tool parser error: {}", e);
                        }
                    }
                }
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
            let (pending_text, stopped_at_finish, local_matched_stop) =
                if let Some(decoder) = &handle.stop_decoder {
                    let mut decoder = decoder.lock().await;
                    let tail = decoder
                        .finish()
                        .map_err(|error| format!("Failed to finish token decoding: {error}"))?;
                    let stopped = matches!(
                        tail,
                        SequenceDecoderOutput::Stopped | SequenceDecoderOutput::StoppedWithText(_)
                    );
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

            // Get and remove state for this index
            let removed_state = handle.stream_state.remove(index);
            let (mut final_text, has_tool_calls, state_prompt_tokens, state_completion_tokens) =
                if let Some(mut state) = removed_state {
                    // Flush decode stream
                    let mut text = std::mem::take(&mut state.text_buffer);
                    if let Some(ref mut decode_stream) = state.decode_stream {
                        if let Ok(Some(remaining)) = decode_stream.flush() {
                            text.push_str(&remaining);
                        }
                    }
                    (
                        text,
                        state.has_tool_calls,
                        state.prompt_tokens,
                        state.completion_tokens,
                    )
                } else {
                    (String::new(), false, 0, 0)
                };

            final_text.push_str(&pending_text);

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
                        tool_calls: None,
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

    fn byte_fallback_tokenizer() -> Arc<dyn Tokenizer> {
        let path =
            std::env::temp_dir().join(format!("smg-byte-fallback-{}.json", uuid::Uuid::now_v7()));
        let definition = serde_json::json!({
            "version":"1.0", "added_tokens":[],
            "model":{"type":"WordLevel", "unk_token":"[UNK]", "vocab":{
                "[UNK]":0, "<0xC3>":1, "<0xA9>":2
            }},
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
            let decoder = StopSequenceDecoder::new(
                tokenizer.clone(),
                llm_tokenizer::StopSequenceConfig::default().with_stop_sequence(stop),
                true,
            );
            let mut handle = GrpcResponseConverterHandle {
                tokenizer: tokenizer.clone(),
                tool_parser: None,
                stop_decoder: Some(Arc::new(TokioMutex::new(decoder))),
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
            let chunk = proto::GenerateResponse {
                response: Some(proto::generate_response::Response::Chunk(
                    proto::GenerateStreamChunk {
                        token_ids: vec![1, 2],
                        completion_tokens: 2,
                        ..Default::default()
                    },
                )),
            };
            assert!(
                convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer)
                    .await
                    .unwrap()
                    .is_none()
            );
            let complete = proto::GenerateResponse {
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
                assert_eq!(choice.delta.content.as_deref(), Some("é"));
                assert_eq!(choice.finish_reason.as_deref(), Some("length"));
            }
        }
    }
}
