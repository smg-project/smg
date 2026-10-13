//! gRPC response converter FFI functions

use std::{
    collections::HashMap,
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
use tool_parser::{types::ToolCallItem, ToolParser};

use super::{
    error::{clear_error_message, set_error_message, SglErrorCode},
    forced_tool_choice::tool_parser_for,
    proto_parse::parse_proto_response,
    runtime::RUNTIME,
    stream_state::StreamStateManager,
    tokenizer::TokenizerHandle,
    utils::generate_tool_call_id,
};

/// Handle for gRPC response converter (maintains state for streaming)
#[repr(C)]
pub struct GrpcResponseConverterHandle {
    pub(crate) tokenizer: Arc<dyn Tokenizer>,
    /// The tool parser of each choice, by choice index (see `tool_parser_output`).
    pub(crate) tool_parsers: HashMap<u32, Option<Box<dyn ToolParser>>>,
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

    // Get system fingerprint from model (simplified)
    let system_fingerprint = Some("fp_placeholder".to_string()); // TODO: Get actual fingerprint

    Box::into_raw(Box::new(GrpcResponseConverterHandle {
        tokenizer,
        tool_parsers: HashMap::new(),
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

/// Whether a tool parser reads this stream: tools declared and a tool choice
/// that allows calls.
fn tool_parsing_on(handle: &GrpcResponseConverterHandle) -> bool {
    handle.tools.is_some()
        && !matches!(
            handle.tool_choice.as_ref(),
            Some(ToolChoice::Value(ToolChoiceValue::None))
        )
}

/// What the choice's parser released for `text`: its text (a value that is
/// not a declared call, a marker prefix) and its calls, both empty while it
/// holds the text as a prospective call. `None` when no parser reads the
/// stream: the text is content as it is. A parser error keeps the converter's
/// fallback: the text is content.
///
/// Every choice of an `n > 1` request has its own parser, created on the
/// choice's first text from the request's tool choice (the model's parser, or
/// the generic reader of a forced choice, see `tool_parser_for`) and owned by
/// the handle, as the gateway's streaming path owns its parsers: a parser's
/// state, its buffer and whether it announced a call's name, belongs to one
/// choice, so the choices never share one.
async fn tool_parser_output(
    handle: &mut GrpcResponseConverterHandle,
    index: u32,
    text: &str,
) -> Option<(String, Vec<ToolCallDelta>)> {
    if !tool_parsing_on(handle) {
        return None;
    }
    if text.is_empty() {
        return Some((String::new(), Vec::new()));
    }
    let GrpcResponseConverterHandle {
        tool_parsers,
        tools,
        model,
        tool_choice,
        history_tool_calls_count,
        ..
    } = handle;
    let parser = tool_parsers
        .entry(index)
        .or_insert_with(|| tool_parser_for(model, tool_choice.as_ref()))
        .as_mut()?;
    let tools = tools.as_deref()?;
    match parser.parse_incremental(text, tools).await {
        Ok(result) => Some((
            result.normal_text,
            result
                .calls
                .into_iter()
                .map(|item| tool_call_delta(model, *history_tool_calls_count, item))
                .collect(),
        )),
        Err(error) => {
            tracing::warn!("Tool parser error: {}", error);
            Some((text.to_string(), Vec::new()))
        }
    }
}

/// What a choice's parser still holds once the choice has ended: text that
/// never became a call (content, as the gateway's own streaming path sends
/// it) and arguments it parsed but did not stream. The parser goes with the
/// choice. `None` when no parser read the choice.
fn unstreamed_tool_output(
    handle: &mut GrpcResponseConverterHandle,
    index: u32,
) -> Option<(String, Vec<ToolCallDelta>)> {
    let mut parser = handle.tool_parsers.remove(&index).flatten()?;
    let text = parser.take_unstreamed_normal_text();
    let calls = parser
        .get_unstreamed_tool_args()
        .unwrap_or_default()
        .into_iter()
        .map(|item| ToolCallDelta {
            index: item.tool_index as u32,
            id: None,
            tool_type: None,
            function: Some(FunctionCallDelta {
                name: None,
                arguments: (!item.parameters.is_empty()).then_some(item.parameters),
            }),
        })
        .collect();
    Some((text, calls))
}

/// A parsed call, or a piece of its arguments, as the client's delta: the
/// call's id travels with its name, in the first delta of the call; a delta
/// that continues the arguments carries none, as the gateway's own streaming
/// path and the public API send it (a client keeping the latest id it sees
/// would otherwise end up with a different one than the call's).
fn tool_call_delta(
    model: &str,
    history_tool_calls_count: usize,
    item: ToolCallItem,
) -> ToolCallDelta {
    let id = item
        .name
        .as_ref()
        .map(|name| generate_tool_call_id(model, name, item.tool_index, history_tool_calls_count));
    ToolCallDelta {
        index: item.tool_index as u32,
        id,
        tool_type: item.name.as_ref().map(|_| "function".to_string()),
        function: Some(FunctionCallDelta {
            name: item.name,
            arguments: (!item.parameters.is_empty()).then_some(item.parameters),
        }),
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

            // When the request's tool parser reads the stream, what it
            // releases is what the client gets, as on the gateway's own
            // streaming path: its text (a value that is not a declared call,
            // a marker prefix) beside its calls, in one delta; text it still
            // holds as a prospective call is not sent raw, so a call's
            // opening does not leak into the content and a value released
            // later does not come twice. What it still holds when the stream
            // ends is flushed at Complete. The first delta of a choice
            // carries the role with the first text (or calls) the decoder
            // released, as every delta does: a role-only delta dropped that
            // text, which the decoder had already advanced past.
            let (content, tool_calls) = match tool_parser_output(handle, index, &chunk_text).await {
                Some(released) => released,
                None => (chunk_text, Vec::new()),
            };
            if content.is_empty() && tool_calls.is_empty() {
                return Ok(None);
            }
            if !tool_calls.is_empty() {
                handle.stream_state.get_or_create(index).has_tool_calls = true;
            }
            Ok(Some(ChatCompletionStreamResponse {
                id: handle.request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: handle.created,
                model: handle.model.clone(),
                system_fingerprint: handle.system_fingerprint.clone(),
                choices: vec![ChatStreamChoice {
                    index,
                    delta: ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: (!content.is_empty()).then_some(content),
                        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                        reasoning_content: None,
                    },
                    logprobs: None,
                    finish_reason: None,
                    matched_stop: None,
                }],
                usage: None,
            }))
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

            // Only newly decoded text reaches the parser or the client here,
            // then what the parser still holds: text that never became a
            // call, arguments parsed but not streamed yet.
            let (mut final_text, mut eof_tool_calls) =
                match tool_parser_output(handle, index, &pending_text).await {
                    Some(released) => released,
                    None => (pending_text, Vec::new()),
                };
            if let Some((held_text, held_calls)) = unstreamed_tool_output(handle, index) {
                final_text.push_str(&held_text);
                eof_tool_calls.extend(held_calls);
            }
            let has_tool_calls = has_tool_calls || !eof_tool_calls.is_empty();

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
                        tool_calls: (!eof_tool_calls.is_empty()).then_some(eof_tool_calls),
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
                tool_parsers: HashMap::new(),
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
    /// The first chunk of a choice that carries text goes out as one delta
    /// with the role and that text, on the plain decode path and on the
    /// stop-decoder path alike; a first chunk the tool parser reads carries
    /// the role with what the parser released. Nothing the decoder released
    /// is dropped.
    #[tokio::test]
    async fn the_first_chunk_of_a_choice_carries_the_role_and_its_text() {
        let tokenizer = byte_fallback_tokenizer();
        for stop in [None, Some(StringOrArray::Array(vec!["NEVER".into()]))] {
            let mut handle = GrpcResponseConverterHandle {
                tokenizer: tokenizer.clone(),
                tool_parsers: HashMap::new(),
                stop,
                stop_token_ids: None,
                model: "byte-fallback-test".into(),
                request_id: "first-delta".into(),
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
                .await
                .unwrap()
                .expect("the first chunk's text is sent");
            let delta = &first.choices[0].delta;
            assert_eq!(delta.role.as_deref(), Some("assistant"), "{first:?}");
            assert_eq!(delta.content.as_deref(), Some("é"), "{first:?}");
        }
        // A first chunk the tool parser reads: with tools off, the text goes
        // out as content; with tools on, what the parser released goes out
        // with the role, as content or as calls, never nothing.
        let tool_text = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
        let token_ids: Vec<u32> = tool_text.bytes().map(|byte| u32::from(byte) + 1).collect();
        for tool_choice in [Some(ToolChoice::Value(ToolChoiceValue::None)), None] {
            let disabled = tool_choice.is_some();
            let mut handle = tool_test_handle(tokenizer.clone(), tool_choice);
            let chunk = proto::GenerateResponse {
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
                .await
                .unwrap()
                .expect("the first chunk is sent");
            let delta = &first.choices[0].delta;
            assert_eq!(delta.role.as_deref(), Some("assistant"), "{first:?}");
            if disabled {
                assert_eq!(delta.content.as_deref(), Some(tool_text), "{first:?}");
                assert!(delta.tool_calls.is_none(), "{first:?}");
            } else {
                // A complete call in one chunk: the parser releases it as tool
                // calls and nothing as content.
                assert!(delta.content.is_none(), "{first:?}");
                assert!(delta.tool_calls.is_some(), "{first:?}");
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
            tool_parsers: HashMap::from([(
                0,
                Some(Box::new(tool_parser::JsonParser::new()) as Box<dyn ToolParser>),
            )]),
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

    #[tokio::test]
    async fn a_chunk_tool_call_delta_carries_the_text_released_with_it() {
        let tokenizer = byte_fallback_tokenizer();
        // One chunk holds a value that is not a declared call and the declared
        // call after it: the parser releases the first as text and the second
        // as deltas in one result, and the client gets both together.
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
            .await
            .unwrap()
            .unwrap();
        let delta = &streamed.choices[0].delta;
        assert_eq!(delta.role.as_deref(), Some("assistant"));
        assert_eq!(delta.content.as_deref(), Some(released), "{streamed:?}");
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
            .await
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

    #[tokio::test]
    async fn a_calls_opening_split_over_chunks_does_not_leak_into_the_content() {
        let tokenizer = byte_fallback_tokenizer();
        let call = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
        let mut handle = tool_test_handle(tokenizer.clone(), None);
        let mut calls = Vec::new();
        // One byte per chunk: the parser holds the opening until the name is
        // complete, and nothing of it is content.
        for (sent, byte) in call.bytes().enumerate() {
            let chunk = chunk_of(vec![u32::from(byte) + 1], sent as u32 + 1);
            if let Some(response) = convert_proto_chunk_to_openai(chunk, &mut handle, &tokenizer)
                .await
                .unwrap()
            {
                let delta = &response.choices[0].delta;
                assert!(delta.content.is_none(), "{response:?}");
                calls.extend(delta.tool_calls.clone().unwrap_or_default());
            }
        }
        let response =
            convert_proto_chunk_to_openai(complete_of(byte_ids(call)), &mut handle, &tokenizer)
                .await
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

    #[tokio::test]
    async fn text_the_parser_releases_over_two_chunks_reaches_the_client_once() {
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
        .await
        .unwrap();
        assert!(held.is_none(), "{held:?}");
        let response = convert_proto_chunk_to_openai(
            chunk_of(byte_ids(second), (first.len() + second.len()) as u32),
            &mut handle,
            &tokenizer,
        )
        .await
        .unwrap()
        .unwrap();
        let delta = &response.choices[0].delta;
        assert_eq!(
            delta.content.as_deref(),
            Some(r#"{"note":"checking"} "#),
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
        .await
        .unwrap()
        .unwrap();
        let choice = &response.choices[0];
        assert!(choice.delta.content.is_none(), "{response:?}");
        assert!(choice.delta.tool_calls.is_none(), "{response:?}");
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[tokio::test]
    async fn text_the_parser_still_holds_at_the_end_is_content_at_complete() {
        let tokenizer = byte_fallback_tokenizer();
        let mut handle = tool_test_handle(tokenizer.clone(), None);
        // A bare brace that never becomes a call is held, then flushed as
        // content when the stream ends.
        let held =
            convert_proto_chunk_to_openai(chunk_of(byte_ids("{"), 1), &mut handle, &tokenizer)
                .await
                .unwrap();
        assert!(held.is_none(), "{held:?}");
        let response =
            convert_proto_chunk_to_openai(complete_of(byte_ids("{")), &mut handle, &tokenizer)
                .await
                .unwrap()
                .unwrap();
        let choice = &response.choices[0];
        assert_eq!(choice.delta.content.as_deref(), Some("{"), "{response:?}");
        assert!(choice.delta.tool_calls.is_none(), "{response:?}");
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));
    }
}
