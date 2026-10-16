//! Preprocessing FFI functions for chat requests
//!
//! This module provides C-compatible functions for preprocessing chat completion requests:
//! - Apply chat_template to messages
//! - Tokenize the processed text
//! - Generate tool constraints
//!
//! These functions are designed to be called once per request, reducing FFI overhead.

use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int, c_uint},
    ptr,
};

use llm_tokenizer::{create_tokenizer_from_file, traits::Tokenizer};
use openai_protocol::chat::ChatCompletionRequest;
use smg::routers::grpc::utils::process_chat_messages;

use super::{
    error::{set_error_message, SglErrorCode},
    memory::{sgl_free_string, sgl_free_token_ids},
    tokenizer::TokenizerHandle,
    tool_constraint::{effective_tools_json, tool_constraint_for, tools_keep_special_tokens},
    utils::chat_requires_reasoning,
};

/// Result of preprocessing a chat request
struct PreprocessResult {
    prompt_text: CString,
    token_ids: Vec<u32>,
    tool_constraints: Option<CString>,
    tools_json: Option<CString>,
    prompt_tokens: i32,
    /// Whether the engine and the response converter skip special tokens:
    /// the request's own setting, unless the response parser needs the
    /// engine's markers kept (see `tools_keep_special_tokens`).
    skip_special_tokens: bool,
}

/// Internal helper to preprocess a chat request with a given tokenizer
fn preprocess_impl(
    chat_request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> Result<PreprocessResult, (SglErrorCode, String)> {
    // Process chat messages (apply chat_template)
    let processed_messages = process_chat_messages(chat_request, tokenizer, None).map_err(|e| {
        (
            SglErrorCode::ParsingError,
            format!("Failed to process chat messages: {e}"),
        )
    })?;

    // Tokenize the processed text
    let encoding = tokenizer
        .encode(&processed_messages.text, false)
        .map_err(|e| {
            (
                SglErrorCode::TokenizationError,
                format!("Tokenization failed: {e}"),
            )
        })?;

    let token_ids: Vec<u32> = encoding.token_ids().to_vec();
    let prompt_tokens = token_ids.len() as i32;

    // Generate tool constraints if tools are present
    let tool_constraint = match tool_constraint_for(chat_request) {
        Ok(constraint) => constraint.map(|c| c.to_tuple()),
        Err(e) => {
            return Err((
                SglErrorCode::ParsingError,
                format!("Failed to generate tool constraints: {e}"),
            ))
        }
    };

    // The engine keeps its special tokens when a parser will read them: tools
    // at either level, a choice that allows calls, no JSON schema. Otherwise
    // the request's own setting stands. The in-library request paths derive
    // the flag the same way; the caller sets the engine's sampling parameter
    // and its response converter from it.
    let skip_special_tokens = chat_request.skip_special_tokens
        && !tools_keep_special_tokens(chat_request, tool_constraint.as_ref());

    let tool_constraints = match tool_constraint {
        Some(constraint) => {
            let json_str = serde_json::to_string(&constraint).map_err(|e| {
                (
                    SglErrorCode::ParsingError,
                    format!("Failed to serialize tool constraints: {e}"),
                )
            })?;
            Some(CString::new(json_str).map_err(|e| {
                (
                    SglErrorCode::MemoryError,
                    format!("Failed to create C string: {e}"),
                )
            })?)
        }
        None => None,
    };

    // The tools the response parser resolves calls against (request-level and
    // message-declared), for the caller's converter.
    let tools_json = match effective_tools_json(chat_request) {
        Some(json) => Some(CString::new(json).map_err(|e| {
            (
                SglErrorCode::MemoryError,
                format!("Failed to create C string: {e}"),
            )
        })?),
        None => None,
    };

    // Create prompt text CString
    let prompt_text = CString::new(processed_messages.text).map_err(|e| {
        (
            SglErrorCode::MemoryError,
            format!("Failed to create C string: {e}"),
        )
    })?;

    Ok(PreprocessResult {
        prompt_text,
        token_ids,
        tool_constraints,
        tools_json,
        prompt_tokens,
        skip_special_tokens,
    })
}

/// The out-parameters of a preprocess entry point, as the caller passed them.
struct PreprocessOutputs {
    prompt_text: *mut *mut c_char,
    token_ids: *mut *mut c_uint,
    token_ids_len: *mut usize,
    /// May be null.
    tool_constraints_json: *mut *mut c_char,
    /// May be null.
    tools_json: *mut *mut c_char,
    /// May be null.
    skip_special_tokens: *mut c_int,
    prompt_tokens: *mut c_int,
}

/// Write preprocess results to FFI output pointers
///
/// # Safety
/// Every output pointer must be valid and writable, except those documented
/// as nullable, which may be null.
unsafe fn write_preprocess_outputs(result: PreprocessResult, outputs: PreprocessOutputs) {
    *outputs.prompt_text = result.prompt_text.into_raw();
    *outputs.token_ids_len = result.token_ids.len();
    *outputs.prompt_tokens = result.prompt_tokens;

    if !outputs.skip_special_tokens.is_null() {
        *outputs.skip_special_tokens = c_int::from(result.skip_special_tokens);
    }

    *outputs.token_ids = if result.token_ids.is_empty() {
        ptr::null_mut()
    } else {
        let boxed = result.token_ids.into_boxed_slice();
        Box::into_raw(boxed) as *mut c_uint
    };

    if !outputs.tool_constraints_json.is_null() {
        *outputs.tool_constraints_json = result
            .tool_constraints
            .map(|c| c.into_raw())
            .unwrap_or(ptr::null_mut());
    }

    if !outputs.tools_json.is_null() {
        *outputs.tools_json = result
            .tools_json
            .map(|t| t.into_raw())
            .unwrap_or(ptr::null_mut());
    }
}

/// Preprocess a chat completion request
///
/// This function:
/// 1. Applies chat_template to messages
/// 2. Tokenizes the processed text
/// 3. Generates tool constraints (if tools are present)
///
/// # Arguments
/// * `request_json` - OpenAI ChatCompletionRequest as JSON string
/// * `tokenizer_path` - Path to tokenizer directory
/// * `prompt_text_out` - Pointer to receive prompt text (C string, must be freed with sgl_free_string)
/// * `token_ids_out` - Pointer to receive token IDs array (must be freed with sgl_free_token_ids)
/// * `token_ids_len_out` - Pointer to receive token IDs array length
/// * `tool_constraints_json_out` - Optional pointer to receive tool constraints JSON (must be freed with sgl_free_string)
/// * `tools_json_out` - Optional pointer to receive the tools the response parser resolves calls against, request-level and message-declared, as JSON (freed by sgl_preprocessed_request_free)
/// * `skip_special_tokens_out` - Optional pointer to receive whether the engine and the response converter skip special tokens (1) or keep them for the response parser (0): the request's setting, unless tools are declared under a choice that allows calls and no JSON schema constrains the output
/// * `prompt_tokens_out` - Pointer to receive prompt token count
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// - `request_json` and `tokenizer_path` must be valid null-terminated C strings
/// - `prompt_text_out`, `token_ids_out`, `token_ids_len_out`, `prompt_tokens_out` must be valid pointers
/// - `tool_constraints_json_out` may be null; if non-null, must point to writable memory
/// - `tools_json_out` may be null; if non-null, must point to writable memory
/// - `skip_special_tokens_out` may be null; if non-null, must point to writable memory
/// - `error_out` may be null; if non-null, must point to writable memory
/// - Caller must free allocated strings with `sgl_free_string` and token IDs with `sgl_free_token_ids`
#[no_mangle]
pub unsafe extern "C" fn sgl_preprocess_chat_request(
    request_json: *const c_char,
    tokenizer_path: *const c_char,
    prompt_text_out: *mut *mut c_char,
    token_ids_out: *mut *mut c_uint,
    token_ids_len_out: *mut usize,
    tool_constraints_json_out: *mut *mut c_char,
    tools_json_out: *mut *mut c_char,
    skip_special_tokens_out: *mut c_int,
    prompt_tokens_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if request_json.is_null()
        || tokenizer_path.is_null()
        || prompt_text_out.is_null()
        || token_ids_out.is_null()
        || token_ids_len_out.is_null()
        || prompt_tokens_out.is_null()
    {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let request_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in request_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    let tokenizer_path_str = match CStr::from_ptr(tokenizer_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in tokenizer_path");
            return SglErrorCode::InvalidArgument;
        }
    };

    let chat_request: ChatCompletionRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse request JSON: {e}"));
            return SglErrorCode::ParsingError;
        }
    };

    let tokenizer = match create_tokenizer_from_file(tokenizer_path_str) {
        Ok(t) => t,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to create tokenizer: {e}"));
            return SglErrorCode::TokenizationError;
        }
    };

    match preprocess_impl(&chat_request, tokenizer.as_ref()) {
        Ok(result) => {
            write_preprocess_outputs(
                result,
                PreprocessOutputs {
                    prompt_text: prompt_text_out,
                    token_ids: token_ids_out,
                    token_ids_len: token_ids_len_out,
                    tool_constraints_json: tool_constraints_json_out,
                    tools_json: tools_json_out,
                    skip_special_tokens: skip_special_tokens_out,
                    prompt_tokens: prompt_tokens_out,
                },
            );
            SglErrorCode::Success
        }
        Err((code, msg)) => {
            set_error_message(error_out, &msg);
            code
        }
    }
}

/// Preprocess a chat completion request using an existing tokenizer handle
///
/// This function is similar to sgl_preprocess_chat_request, but accepts a TokenizerHandle
/// instead of creating a new tokenizer. This allows reusing a cached tokenizer instance,
/// significantly reducing initialization overhead in concurrent scenarios.
///
/// # Arguments
/// * `request_json` - OpenAI ChatCompletionRequest as JSON string
/// * `tokenizer_handle` - Existing tokenizer handle (must be valid)
/// * `prompt_text_out` - Pointer to receive prompt text (C string, must be freed with sgl_free_string)
/// * `token_ids_out` - Pointer to receive token IDs array (must be freed with sgl_free_token_ids)
/// * `token_ids_len_out` - Pointer to receive token IDs array length
/// * `tool_constraints_json_out` - Optional pointer to receive tool constraints JSON (must be freed with sgl_free_string)
/// * `tools_json_out` - Optional pointer to receive the tools the response parser resolves calls against, request-level and message-declared, as JSON (freed by sgl_preprocessed_request_free)
/// * `skip_special_tokens_out` - Optional pointer to receive whether the engine and the response converter skip special tokens (1) or keep them for the response parser (0): the request's setting, unless tools are declared under a choice that allows calls and no JSON schema constrains the output
/// * `prompt_tokens_out` - Pointer to receive prompt token count
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// - `request_json` must be a valid null-terminated C string
/// - `tokenizer_handle` must be a valid pointer returned by `sgl_tokenizer_create`
/// - `prompt_text_out`, `token_ids_out`, `token_ids_len_out`, `prompt_tokens_out` must be valid pointers
/// - `tool_constraints_json_out` may be null; if non-null, must point to writable memory
/// - `tools_json_out` may be null; if non-null, must point to writable memory
/// - `skip_special_tokens_out` may be null; if non-null, must point to writable memory
/// - `error_out` may be null; if non-null, must point to writable memory
/// - Caller must free allocated strings with `sgl_free_string` and token IDs with `sgl_free_token_ids`
#[no_mangle]
pub unsafe extern "C" fn sgl_preprocess_chat_request_with_tokenizer(
    request_json: *const c_char,
    tokenizer_handle: *mut TokenizerHandle,
    prompt_text_out: *mut *mut c_char,
    token_ids_out: *mut *mut c_uint,
    token_ids_len_out: *mut usize,
    tool_constraints_json_out: *mut *mut c_char,
    tools_json_out: *mut *mut c_char,
    skip_special_tokens_out: *mut c_int,
    prompt_tokens_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if request_json.is_null()
        || tokenizer_handle.is_null()
        || prompt_text_out.is_null()
        || token_ids_out.is_null()
        || token_ids_len_out.is_null()
        || prompt_tokens_out.is_null()
    {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let request_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in request_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    let chat_request: ChatCompletionRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse request JSON: {e}"));
            return SglErrorCode::ParsingError;
        }
    };

    let handle_ref = &*tokenizer_handle;

    match preprocess_impl(&chat_request, handle_ref.tokenizer.as_ref()) {
        Ok(result) => {
            write_preprocess_outputs(
                result,
                PreprocessOutputs {
                    prompt_text: prompt_text_out,
                    token_ids: token_ids_out,
                    token_ids_len: token_ids_len_out,
                    tool_constraints_json: tool_constraints_json_out,
                    tools_json: tools_json_out,
                    skip_special_tokens: skip_special_tokens_out,
                    prompt_tokens: prompt_tokens_out,
                },
            );
            SglErrorCode::Success
        }
        Err((code, msg)) => {
            set_error_message(error_out, &msg);
            code
        }
    }
}

/// Free a preprocessed request handle (cleanup function)
///
/// This function frees the memory allocated by sgl_preprocess_chat_request.
/// It should be called after the preprocessed data is no longer needed.
///
/// # Safety
/// - `prompt_text` must be a valid pointer allocated by `sgl_preprocess_chat_request`, or null
/// - `token_ids` must be a valid pointer allocated by `sgl_preprocess_chat_request`, or null
/// - `token_ids_len` must match the actual length of the `token_ids` array
/// - `tool_constraints_json` must be a valid pointer allocated by `sgl_preprocess_chat_request`, or null
/// - `tools_json` must be a valid pointer allocated by `sgl_preprocess_chat_request`, or null
/// - These pointers must not be used after this call
#[no_mangle]
pub unsafe extern "C" fn sgl_preprocessed_request_free(
    prompt_text: *mut c_char,
    token_ids: *mut c_uint,
    token_ids_len: usize,
    tool_constraints_json: *mut c_char,
    tools_json: *mut c_char,
) {
    if !prompt_text.is_null() {
        sgl_free_string(prompt_text);
    }

    if !token_ids.is_null() && token_ids_len > 0 {
        sgl_free_token_ids(token_ids, token_ids_len);
    }

    if !tool_constraints_json.is_null() {
        sgl_free_string(tool_constraints_json);
    }

    if !tools_json.is_null() {
        sgl_free_string(tools_json);
    }
}

/// Determine whether a chat request should ask SGLang to count reasoning tokens.
///
/// # Safety
/// - `request_json` must be a valid null-terminated C string.
/// - `tokenizer_handle` must be a valid pointer returned by `sgl_tokenizer_create`.
/// - `require_reasoning_out` must point to writable memory.
/// - `error_out` may be null; if non-null, must point to writable memory.
#[no_mangle]
pub unsafe extern "C" fn sgl_chat_requires_reasoning_with_tokenizer(
    request_json: *const c_char,
    tokenizer_handle: *mut TokenizerHandle,
    require_reasoning_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if request_json.is_null() || tokenizer_handle.is_null() || require_reasoning_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let request_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in request_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    let chat_request: ChatCompletionRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse request JSON: {e}"));
            return SglErrorCode::ParsingError;
        }
    };

    let handle_ref = &*tokenizer_handle;
    *require_reasoning_out = i32::from(chat_requires_reasoning(
        &chat_request,
        handle_ref.tokenizer.as_ref(),
    ));

    SglErrorCode::Success
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::{json, Value};

    use super::*;

    /// The test tokenizer: the renderer that writes one `role: content` line
    /// per message, with no checkpoint behind it.
    fn tokenizer() -> Arc<dyn Tokenizer> {
        create_tokenizer_from_file("mock").unwrap()
    }

    fn weather_tool() -> Value {
        json!([{"type": "function", "function": {"name": "get_weather", "parameters":
            {"type": "object", "properties": {"city": {"type": "string"}}}}}])
    }

    /// A request for an unmapped model; `tools`, `tool_choice` and
    /// `skip_special_tokens` are left out when null.
    fn request_json(tools: Value, tool_choice: Value, skip_special_tokens: Value) -> Value {
        let mut request = json!({
            "model": "unmapped-model",
            "messages": [{"role": "user", "content": "hi"}],
        });
        for (key, value) in [
            ("tools", tools),
            ("tool_choice", tool_choice),
            ("skip_special_tokens", skip_special_tokens),
        ] {
            if !value.is_null() {
                request[key] = value;
            }
        }
        request
    }

    fn preprocess(request: Value) -> PreprocessResult {
        let request: ChatCompletionRequest = serde_json::from_value(request).unwrap();
        preprocess_impl(&request, tokenizer().as_ref()).unwrap()
    }

    /// Tools under a choice that allows calls keep the engine's special tokens
    /// for the response parser; a choice that forces none, a choice that
    /// constrains the output to a JSON schema, or no tools at all leave the
    /// request's setting, which skips them.
    #[test]
    fn special_tokens_are_kept_when_a_parser_reads_the_tools() {
        for (tool_choice, skip) in [
            (Value::Null, false),
            (json!("auto"), false),
            (json!("none"), true),
            (json!("required"), true),
            (
                json!({"type": "function", "function": {"name": "get_weather"}}),
                true,
            ),
        ] {
            let result = preprocess(request_json(
                weather_tool(),
                tool_choice.clone(),
                Value::Null,
            ));
            assert_eq!(
                result.skip_special_tokens, skip,
                "tool_choice {tool_choice}"
            );
            assert!(result.tools_json.is_some());
        }
        for tools in [Value::Null, json!([])] {
            let result = preprocess(request_json(tools.clone(), Value::Null, Value::Null));
            assert!(result.skip_special_tokens, "tools {tools}");
            assert!(result.tools_json.is_none());
        }
    }

    /// A request that asks for its special tokens keeps them whatever its
    /// tools say.
    #[test]
    fn a_request_that_keeps_its_special_tokens_is_honoured() {
        let plain = preprocess(request_json(Value::Null, Value::Null, json!(false)));
        assert!(!plain.skip_special_tokens);
        let none_choice = preprocess(request_json(weather_tool(), json!("none"), json!(false)));
        assert!(!none_choice.skip_special_tokens);
    }

    /// A tool declared on a message (a provider profile that defines them)
    /// counts as the request-level ones do.
    #[test]
    fn a_message_declared_tool_keeps_the_special_tokens() {
        let result = preprocess(json!({
            "model": "kimi-k3",
            "messages": [
                {"role": "system", "content": "", "tools": weather_tool()},
                {"role": "user", "content": "hi"}
            ]
        }));
        assert!(!result.skip_special_tokens);
        assert!(result.tools_json.is_some());
    }

    /// The FFI entry point hands the flag out through its out-parameter, next
    /// to the parser's tool list, and tolerates a caller that passes none.
    #[test]
    fn the_ffi_entry_point_returns_the_flag() {
        let request =
            CString::new(request_json(weather_tool(), Value::Null, Value::Null).to_string())
                .unwrap();
        let tokenizer_path = CString::new("mock").unwrap();
        for read_flag in [true, false] {
            let mut prompt_text: *mut c_char = ptr::null_mut();
            let mut token_ids: *mut c_uint = ptr::null_mut();
            let mut token_ids_len: usize = 0;
            let mut tool_constraints: *mut c_char = ptr::null_mut();
            let mut tools_json: *mut c_char = ptr::null_mut();
            let mut skip_special_tokens: c_int = -1;
            let mut prompt_tokens: c_int = -1;
            let mut error: *mut c_char = ptr::null_mut();
            // SAFETY: every pointer is a live local or null where the
            // function allows null; the strings are null-terminated.
            let code = unsafe {
                sgl_preprocess_chat_request(
                    request.as_ptr(),
                    tokenizer_path.as_ptr(),
                    &mut prompt_text,
                    &mut token_ids,
                    &mut token_ids_len,
                    &mut tool_constraints,
                    &mut tools_json,
                    if read_flag {
                        &mut skip_special_tokens
                    } else {
                        ptr::null_mut()
                    },
                    &mut prompt_tokens,
                    &mut error,
                )
            };
            assert_eq!(code, SglErrorCode::Success);
            assert_eq!(skip_special_tokens, if read_flag { 0 } else { -1 });
            assert!(prompt_tokens >= 0);
            assert!(tool_constraints.is_null());
            assert!(!tools_json.is_null());
            // SAFETY: the pointers were written by the call above and are
            // freed once.
            unsafe {
                sgl_preprocessed_request_free(
                    prompt_text,
                    token_ids,
                    token_ids_len,
                    tool_constraints,
                    tools_json,
                );
            }
        }
    }
}
