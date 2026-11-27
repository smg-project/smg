//! gRPC response converter FFI functions

use std::{
    collections::{hash_map::Entry, HashMap},
    ffi::{CStr, CString},
    os::raw::{c_char, c_int},
    ptr,
    sync::Arc,
};

use llm_tokenizer::{
    stop::SequenceDecoderOutput, stream::DecodeStream, traits::Tokenizer, StopSequenceDecoder,
};
use openai_protocol::{
    chat::{
        ChatCompletionRequest, ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice,
    },
    common::{StringOrArray, Tool, ToolChoice, Usage},
};
use serde_json::Value;
use smg::routers::grpc::utils::create_stop_decoder;
use smg_grpc_client::sglang_proto::{self as proto, generate_complete::MatchedStop};

use super::{
    chat_stream::fingerprint_of,
    error::{clear_error_message, set_error_message, SglErrorCode},
    proto_parse::parse_proto_response,
    reader::{engine_finish, Reader, Reading},
    stream_state::{StreamIndexState, StreamStateManager},
    tokenizer::TokenizerHandle,
};

/// Handle for gRPC response converter (maintains state for streaming)
#[repr(C)]
pub struct GrpcResponseConverterHandle {
    pub(crate) tokenizer: Arc<dyn Tokenizer>,
    /// The reader of each choice, by choice index (see `reader_of`).
    pub(crate) readers: HashMap<u32, Reader>,
    pub(crate) stop: Option<StringOrArray>,
    pub(crate) stop_token_ids: Option<Vec<u32>>,
    pub(crate) model: String,
    pub(crate) request_id: String,
    pub(crate) created: u64,
    pub(crate) system_fingerprint: Option<String>,
    /// What the request's output is read with (`reader::Reading`), shared by
    /// the readers of its choices.
    pub(crate) reading: Reading,
    /// The part of the rendered prompt the readers replay
    /// (`reader::Reading::replay`): from the table's last turn opener, the
    /// prompt itself for a table that names none; empty from the C entry
    /// point, which does not receive the prompt.
    pub(crate) prompt: String,
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
/// * `initial_prompt_tokens` - The prompt's token count from preprocessing
///   (the usage fallback when a chunk reports none); 0 for none
/// * `weight_version` - The engine's weight version (its model information),
///   the chunks' `system_fingerprint`; may be null or blank for the gateway's
///   `default`
/// * `error_out` - Optional pointer to receive error message
///
/// Neither the request's `separate_reasoning` nor its rendered prompt is
/// among the arguments: the converter reads a thought apart from the
/// content, the API's default, and its readers start at the table's first
/// state.
///
/// The parameter order is the Go prototype's (`internal/ffi/grpc_converter.go`);
/// the two must change together.
///
/// # Returns
/// * Pointer to GrpcResponseConverterHandle on success, null on failure
///
/// # Safety
/// - `tokenizer_handle` must be a valid pointer returned by `sgl_tokenizer_create_from_file`
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
    initial_prompt_tokens: c_int,
    weight_version: *const c_char,
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

    let weight_version = if weight_version.is_null() {
        None
    } else {
        CStr::from_ptr(weight_version).to_str().ok()
    };
    let mut handle = GrpcResponseConverterHandle::new(
        tokenizer,
        model_str.to_string(),
        request_id_str.to_string(),
        skip_special_tokens != 0,
        tools,
        tool_choice,
        true,
        "",
        stop,
        stop_token_ids,
    );
    handle.initial_prompt_tokens = u32::try_from(initial_prompt_tokens).ok().filter(|n| *n > 0);
    handle.system_fingerprint = Some(fingerprint_of(weight_version));
    Box::into_raw(Box::new(handle))
}

impl GrpcResponseConverterHandle {
    /// A converter over one request's stream: what its readers read the
    /// output with (the request's `tools`, `None` when it declares none, its
    /// `tool_choice` and whether it separates the reasoning from the content:
    /// `reader::Reading`), the part of the rendered `prompt` they replay, and
    /// its stops;
    /// the clock stamps `created`.
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new(
        tokenizer: Arc<dyn Tokenizer>,
        model: String,
        request_id: String,
        skip_special_tokens: bool,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        separate_reasoning: bool,
        prompt: &str,
        stop: Option<StringOrArray>,
        stop_token_ids: Option<Vec<u32>>,
    ) -> Self {
        let reading = Reading::of(
            &model,
            tools.as_deref(),
            tool_choice.as_ref(),
            separate_reasoning,
        );
        let prompt = reading.replay(prompt).to_owned();
        Self {
            tokenizer,
            readers: HashMap::new(),
            stop,
            stop_token_ids,
            model,
            request_id,
            // unwrap_or_default is acceptable here: if the clock is before UNIX epoch,
            // the `created` field in the API response will be 0, which is cosmetic
            // and does not cause data corruption or silent data loss.
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            // The engine's weight version, set by the stream or the FFI
            // constructor that knows it.
            system_fingerprint: None,
            reading,
            prompt,
            stream_state: StreamStateManager::new(),
            initial_prompt_tokens: None,
            skip_special_tokens,
        }
    }

    /// The converter of a chat request's stream: its effective tools
    /// (request-level and message-declared, the set its constraint was built
    /// from; `None` when it declares none, so the output is scanned for calls
    /// exactly when the gateway scans it), its tool choice, whether it
    /// separates the reasoning, the part of the rendered `prompt` its readers replay,
    /// and its stops.
    pub(crate) fn for_chat(
        tokenizer: Arc<dyn Tokenizer>,
        request: &ChatCompletionRequest,
        request_id: String,
        prompt: &str,
    ) -> Self {
        let tools: Vec<Tool> = request.effective_tools().cloned().collect();
        Self::new(
            tokenizer,
            request.model.clone(),
            request_id,
            request.skip_special_tokens,
            request.declares_tools().then_some(tools),
            request.tool_choice.clone(),
            request.separate_reasoning,
            prompt,
            request.stop.clone(),
            request.stop_token_ids.clone(),
        )
    }
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

    match convert_proto_chunk_to_openai(proto_response, handle_ref, &tokenizer) {
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

/// The reader of the choice at `index`, created on the choice's first step
/// from the request's reading (`reader::Reading`: the model's table, or the
/// table of a forced choice) and owned by the handle, as the gateway's
/// streaming path owns its readers: a reader's state, what it holds as a
/// prospective call and the calls it started, belongs to one choice, so the
/// choices of an `n > 1` request never share one.
fn reader_of(handle: &mut GrpcResponseConverterHandle, index: u32) -> Result<&mut Reader, String> {
    let GrpcResponseConverterHandle {
        readers,
        reading,
        prompt,
        ..
    } = handle;
    Ok(match readers.entry(index) {
        Entry::Occupied(reader) => reader.into_mut(),
        Entry::Vacant(slot) => slot.insert(reading.reader(prompt)?),
    })
}

/// One engine message of a stream as the client's chunk: a `Chunk` as the
/// delta its text (and calls) make, a `Complete` as the choice's last chunk
/// with its finish reason, matched stop and usage. `None` when the message
/// adds nothing the client gets.
pub(crate) fn convert_proto_chunk_to_openai(
    proto_response: proto::GenerateResponse,
    handle: &mut GrpcResponseConverterHandle,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Result<Option<ChatCompletionStreamResponse>, String> {
    use proto::generate_response::Response;
    match proto_response.response {
        Some(Response::Chunk(chunk)) => chunk_delta(handle, tokenizer, chunk),
        Some(Response::Complete(complete)) => complete_choice(handle, complete).map(Some),
        None => Ok(None),
    }
}

/// The text a chunk's tokens add to its choice: through the request's stop
/// decoder when it has stops (the text up to a stop the decoder finds), else
/// through the incremental decoder, which holds a character until its bytes
/// are complete. Either is created on the choice's first chunk.
fn decode_chunk_text(
    handle: &mut GrpcResponseConverterHandle,
    tokenizer: &Arc<dyn Tokenizer>,
    index: u32,
    token_ids: &[u32],
) -> String {
    let GrpcResponseConverterHandle {
        stream_state,
        stop,
        stop_token_ids,
        skip_special_tokens,
        ..
    } = handle;
    let state = stream_state.get_or_create(index);
    if stop.is_some() || stop_token_ids.is_some() {
        let decoder = state.stop_decoder.get_or_insert_with(|| {
            create_stop_decoder(
                tokenizer,
                stop.as_ref(),
                stop_token_ids.as_ref(),
                *skip_special_tokens,
                false,
                false,
            )
        });
        stop_decoded_text(decoder, token_ids)
    } else {
        let decoder = state
            .decode_stream
            .get_or_insert_with(|| DecodeStream::new(tokenizer.clone(), &[], *skip_special_tokens));
        token_ids
            .iter()
            .filter_map(|&token_id| decoder.step(token_id).ok().flatten())
            .collect()
    }
}

/// The text the stop decoder releases for `token_ids`, up to the stop it
/// finds; a token it cannot decode is held.
fn stop_decoded_text(decoder: &mut StopSequenceDecoder, token_ids: &[u32]) -> String {
    let mut text = String::new();
    for &token_id in token_ids {
        let before = text.len();
        match decoder.process_token_into(token_id, &mut text) {
            Ok(step) if step.stopped() => break,
            Ok(_) => {}
            Err(_) => text.truncate(before),
        }
    }
    text
}

/// A chunk of a choice as the client's delta. The chunk's counts are kept
/// (cumulative in the proto; the prompt count falls back to the request's).
/// When a table reads the stream, what it releases is what the client gets,
/// as on the gateway's own streaming path: its content (a value that is not
/// a declared call, a marker the table shows as text) and its reasoning
/// beside its calls, in one delta; text it still holds as a prospective call
/// is not sent raw, so a call's opening does not leak into the content and a
/// value released later does not come twice. What it still holds when the
/// stream ends is flushed at Complete. The first delta of a choice carries
/// the role with the first text (or calls) the decoder released, as every
/// delta does: a role-only delta dropped that text, which the decoder had
/// already advanced past. `None` when nothing is released.
fn chunk_delta(
    handle: &mut GrpcResponseConverterHandle,
    tokenizer: &Arc<dyn Tokenizer>,
    chunk: proto::GenerateStreamChunk,
) -> Result<Option<ChatCompletionStreamResponse>, String> {
    let index = chunk.index;
    let state = handle.stream_state.get_or_create(index);
    if chunk.prompt_tokens > 0 {
        state.prompt_tokens = chunk.prompt_tokens;
    } else if state.prompt_tokens == 0 {
        if let Some(initial_prompt) = handle.initial_prompt_tokens {
            state.prompt_tokens = initial_prompt;
        }
    }
    state.completion_tokens = chunk.completion_tokens;

    let chunk_text = decode_chunk_text(handle, tokenizer, index, &chunk.token_ids);
    if chunk_text.is_empty() {
        return Ok(None);
    }
    let step = reader_of(handle, index)?.read(index, &chunk_text)?;
    if step.is_empty() {
        return Ok(None);
    }
    Ok(Some(handle.stream_chunk(
        index,
        step.into_delta(),
        None,
        None,
        None,
    )))
}

/// The choice's last chunk: what its decoders still hold goes through the
/// reader, then the reader ends the output as the engine said (what it still
/// held goes out: text that never became a call, a call the end closes) and
/// decides the finish reason; the matched stop and the usage close the
/// choice. The reader goes with the choice.
fn complete_choice(
    handle: &mut GrpcResponseConverterHandle,
    complete: proto::GenerateComplete,
) -> Result<ChatCompletionStreamResponse, String> {
    let index = complete.index;
    let (pending_text, stopped_at_finish, local_matched_stop, counts) =
        match handle.stream_state.remove(index) {
            Some(mut state) => {
                let (text, stopped, matched) =
                    decoders_tail(&mut state, &handle.request_id, index)?;
                let counts = (state.prompt_tokens, state.completion_tokens);
                (text, stopped, matched, counts)
            }
            None => (String::new(), false, None, (0, 0)),
        };
    let (state_prompt_tokens, state_completion_tokens) = counts;
    let engine_finish_reason = if stopped_at_finish {
        "stop"
    } else {
        complete.finish_reason.as_str()
    };

    let reader = reader_of(handle, index)?;
    let mut step = reader
        .read(index, &pending_text)?
        .then(reader.end(index, engine_finish(engine_finish_reason))?);
    handle.readers.remove(&index);
    let finish_reason = step
        .finish_reason
        .take()
        .unwrap_or_else(|| finish_reason_of(engine_finish_reason));
    let delta = step.into_delta();
    let matched_stop =
        local_matched_stop.or_else(|| engine_matched_stop(complete.matched_stop.as_ref()));
    let usage = usage_of(
        &complete,
        state_prompt_tokens,
        state_completion_tokens,
        handle.initial_prompt_tokens,
    );
    Ok(handle.stream_chunk(index, delta, Some(finish_reason), matched_stop, Some(usage)))
}

/// What a choice's decoders still hold when it ends: the stop decoder's
/// tail, with whether it stopped and on which stop, and the incremental
/// decoder's remainder.
fn decoders_tail(
    state: &mut StreamIndexState,
    request_id: &str,
    index: u32,
) -> Result<(String, bool, Option<Value>), String> {
    let (mut text, stopped, matched) = match state.stop_decoder.as_mut() {
        Some(decoder) => {
            let tail = decoder
                .finish()
                .map_err(|error| format!("Failed to finish token decoding: {error}"))?;
            if decoder.prefix_mismatches() > 0 {
                tracing::warn!(
                    request_id = %request_id,
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
        }
        None => (String::new(), false, None),
    };
    if let Some(decode_stream) = state.decode_stream.as_mut() {
        if let Ok(Some(remaining)) = decode_stream.flush() {
            text.push_str(&remaining);
        }
    }
    Ok((text, stopped, matched))
}

/// The client's finish reason where the reader's end named none: `stop`
/// when the engine said nothing, else the engine's own. The reader decides
/// it for every choice it ends (`tool_calls` after a call); this stands in
/// for an end the table left without one.
fn finish_reason_of(engine_finish_reason: &str) -> String {
    if engine_finish_reason.trim().is_empty() {
        "stop".to_string()
    } else {
        engine_finish_reason.to_string()
    }
}

/// The stop the engine matched, as the client's `matched_stop`.
fn engine_matched_stop(matched: Option<&MatchedStop>) -> Option<Value> {
    match matched {
        Some(MatchedStop::MatchedTokenId(token_id)) => Some(Value::Number((*token_id).into())),
        Some(MatchedStop::MatchedStopStr(stop_str)) => Some(Value::String(stop_str.clone())),
        None => None,
    }
}

/// The choice's usage: the counts the chunks carried, else the completion
/// message's, else the request's prompt count and the output ids' length.
fn usage_of(
    complete: &proto::GenerateComplete,
    state_prompt_tokens: u32,
    state_completion_tokens: u32,
    initial_prompt_tokens: Option<u32>,
) -> Usage {
    let prompt_tokens = if state_prompt_tokens > 0 {
        state_prompt_tokens
    } else if complete.prompt_tokens > 0 {
        complete.prompt_tokens
    } else {
        initial_prompt_tokens.unwrap_or(0)
    };
    let completion_tokens = if state_completion_tokens > 0 {
        state_completion_tokens
    } else if complete.completion_tokens > 0 {
        complete.completion_tokens
    } else {
        complete.output_ids.len() as u32
    };
    Usage::from_counts(prompt_tokens, completion_tokens)
        .with_cached_tokens(complete.cached_tokens)
        .with_reasoning_tokens(complete.reasoning_tokens)
}

impl GrpcResponseConverterHandle {
    /// One chunk of the request's stream, carrying one choice.
    fn stream_chunk(
        &self,
        index: u32,
        delta: ChatMessageDelta,
        finish_reason: Option<String>,
        matched_stop: Option<Value>,
        usage: Option<Usage>,
    ) -> ChatCompletionStreamResponse {
        ChatCompletionStreamResponse {
            id: self.request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            system_fingerprint: self.system_fingerprint.clone(),
            prompt_text: None,
            choices: vec![ChatStreamChoice {
                index,
                delta,
                logprobs: None,
                finish_reason,
                matched_stop,
            }],
            usage,
        }
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
#[path = "../tests/unit/grpc_converter/tests.rs"]
mod tests;
