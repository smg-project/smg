//! Client SDK FFI functions

use std::{ffi::CStr, os::raw::c_char, ptr, sync::Arc};

use llm_tokenizer::{create_tokenizer_from_file, traits::Tokenizer};
use smg_grpc_client::sglang_scheduler::SglangSchedulerClient;

use super::{
    chat_stream::{prepare_chat, start_chat_stream},
    error::{set_error_message, SglErrorCode},
    runtime::RUNTIME,
    stream::SglangStreamHandle,
};

/// Handle for complete client SDK (gRPC client + tokenizer)
/// This handle manages the connection to sglang and provides a complete SDK interface
pub struct SglangClientHandle {
    pub(crate) client: Arc<SglangSchedulerClient>,
    pub(crate) tokenizer: Arc<dyn Tokenizer>,
}

/// Create a new SGLang client handle
///
/// # Arguments
/// * `endpoint` - gRPC endpoint (e.g., "grpc://localhost:20000")
/// * `tokenizer_path` - Path to tokenizer directory
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * Pointer to SglangClientHandle on success, null on failure
///
/// # Safety
/// - `endpoint` and `tokenizer_path` must be valid null-terminated C strings
/// - `error_out` may be null; if non-null, must point to writable memory
/// - Caller owns the returned handle and must free it with `sgl_client_free`
#[no_mangle]
pub unsafe extern "C" fn sgl_client_create(
    endpoint: *const c_char,
    tokenizer_path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SglangClientHandle {
    if endpoint.is_null() || tokenizer_path.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return ptr::null_mut();
    }

    let endpoint_str = match CStr::from_ptr(endpoint).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in endpoint");
            return ptr::null_mut();
        }
    };

    let tokenizer_path_str = match CStr::from_ptr(tokenizer_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in tokenizer_path");
            return ptr::null_mut();
        }
    };

    // Create tokenizer
    let tokenizer = match create_tokenizer_from_file(tokenizer_path_str) {
        Ok(t) => t,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to create tokenizer: {e}"));
            return ptr::null_mut();
        }
    };

    // Create gRPC client
    let client =
        match RUNTIME.block_on(async { SglangSchedulerClient::connect(endpoint_str).await }) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                set_error_message(error_out, &format!("Failed to connect to endpoint: {e}"));
                return ptr::null_mut();
            }
        };

    Box::into_raw(Box::new(SglangClientHandle { client, tokenizer }))
}

/// Free a client handle
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_client_create`, or null
/// - `handle` must not be used after this call
/// - This function must not be called more than once for the same handle
#[no_mangle]
pub unsafe extern "C" fn sgl_client_free(handle: *mut SglangClientHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Send a chat completion request and start streaming
///
/// # Arguments
/// * `client_handle` - Client handle
/// * `request_json` - OpenAI ChatCompletionRequest as JSON string
/// * `stream_handle_out` - Pointer to receive stream handle
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// - `client_handle` must be a valid pointer returned by `sgl_client_create`
/// - `request_json` must be a valid null-terminated C string containing valid JSON
/// - `stream_handle_out` must be a valid pointer to writable memory
/// - `error_out` may be null; if non-null, must point to writable memory
/// - Caller owns the stream handle written to `stream_handle_out` and must free it
#[no_mangle]
pub unsafe extern "C" fn sgl_client_chat_completion_stream(
    client_handle: *mut SglangClientHandle,
    request_json: *const c_char,
    stream_handle_out: *mut *mut SglangStreamHandle,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if client_handle.is_null() || request_json.is_null() || stream_handle_out.is_null() {
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

    let client_ref = &*client_handle;
    let prepared = match prepare_chat(request_str, client_ref.tokenizer.as_ref()) {
        Ok(prepared) => prepared,
        Err((code, message)) => {
            set_error_message(error_out, &message);
            return code;
        }
    };
    // A single client's stream holds no worker load.
    match start_chat_stream(
        &client_ref.client,
        &client_ref.tokenizer,
        prepared,
        None,
        error_out,
    ) {
        Ok(handle) => {
            *stream_handle_out = Box::into_raw(Box::new(handle));
            SglErrorCode::Success
        }
        Err(code) => code,
    }
}
