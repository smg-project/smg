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
    utils::chat_requires_reasoning,
};

/// Result of preprocessing a chat request
struct PreprocessResult {
    prompt_text: CString,
    token_ids: Vec<u32>,
    tool_constraints: Option<CString>,
    prompt_tokens: i32,
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
    let registry = super::runtime::PARSER_FACTORY.registry();
    let tool_constraints = if let (Some(tools), Some(tool_choice)) = (
        chat_request.tools.as_ref(),
        chat_request.tool_choice.as_ref(),
    ) {
        match registry.generate_tool_constraint(
            None,
            tools,
            tool_choice,
            chat_request.parallel_tool_calls,
        ) {
            Ok(Some(c)) => {
                let json_str = serde_json::to_string(&c.to_tuple()).map_err(|e| {
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
            Ok(None) => None,
            Err(e) => {
                return Err((
                    SglErrorCode::ParsingError,
                    format!("Failed to generate tool constraints: {e}"),
                ))
            }
        }
    } else {
        None
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
        prompt_tokens,
    })
}

/// Write preprocess results to FFI output pointers
///
/// # Safety
/// All output pointers must be valid and writable
unsafe fn write_preprocess_outputs(
    result: PreprocessResult,
    prompt_text_out: *mut *mut c_char,
    token_ids_out: *mut *mut c_uint,
    token_ids_len_out: *mut usize,
    tool_constraints_json_out: *mut *mut c_char,
    prompt_tokens_out: *mut c_int,
) {
    *prompt_text_out = result.prompt_text.into_raw();
    *token_ids_len_out = result.token_ids.len();
    *prompt_tokens_out = result.prompt_tokens;

    *token_ids_out = if result.token_ids.is_empty() {
        ptr::null_mut()
    } else {
        let boxed = result.token_ids.into_boxed_slice();
        Box::into_raw(boxed) as *mut c_uint
    };

    if !tool_constraints_json_out.is_null() {
        *tool_constraints_json_out = result
            .tool_constraints
            .map(|c| c.into_raw())
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
                prompt_text_out,
                token_ids_out,
                token_ids_len_out,
                tool_constraints_json_out,
                prompt_tokens_out,
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
                prompt_text_out,
                token_ids_out,
                token_ids_len_out,
                tool_constraints_json_out,
                prompt_tokens_out,
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
/// - These pointers must not be used after this call
#[no_mangle]
pub unsafe extern "C" fn sgl_preprocessed_request_free(
    prompt_text: *mut c_char,
    token_ids: *mut c_uint,
    token_ids_len: usize,
    tool_constraints_json: *mut c_char,
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
