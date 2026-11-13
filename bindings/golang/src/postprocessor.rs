//! Postprocessing FFI functions for gRPC stream chunks

use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_int},
    ptr,
    sync::Arc,
};

use serde_json::Value;

use super::{
    error::{set_error_message, SglErrorCode},
    grpc_converter::GrpcResponseConverterHandle,
    proto_parse::{is_terminal_response, parse_proto_response},
};

/// Postprocess a gRPC stream chunk to OpenAI format
///
/// This function:
/// 1. Parses the proto chunk from JSON
/// 2. Converts it to OpenAI format using the converter handle
/// 3. Returns the OpenAI format JSON
///
/// # Arguments
/// * `converter_handle` - Converter handle (created with sgl_grpc_response_converter_create)
/// * `proto_chunk_json` - JSON string of proto.GenerateResponse
/// * `openai_json_out` - Pointer to receive OpenAI format JSON (must be freed with sgl_free_string)
/// * `is_done_out` - Pointer to receive is_done flag (1 if stream is complete, 0 otherwise)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// - `converter_handle` must be a valid pointer returned by `sgl_grpc_response_converter_create`
/// - `proto_chunk_json` must be a valid null-terminated C string containing valid JSON
/// - `openai_json_out` and `is_done_out` must be valid pointers to writable memory
/// - `error_out` may be null; if non-null, must point to writable memory
/// - Caller must free the string written to `openai_json_out` using `sgl_free_string`
#[no_mangle]
pub unsafe extern "C" fn sgl_postprocess_stream_chunk(
    converter_handle: *mut GrpcResponseConverterHandle,
    proto_chunk_json: *const c_char,
    openai_json_out: *mut *mut c_char,
    is_done_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if converter_handle.is_null()
        || proto_chunk_json.is_null()
        || openai_json_out.is_null()
        || is_done_out.is_null()
    {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let proto_chunk_str = match CStr::from_ptr(proto_chunk_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in proto_chunk_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse JSON to check if terminal
    let json_value: Value = match serde_json::from_str(proto_chunk_str) {
        Ok(v) => v,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse proto chunk JSON: {e}"));
            return SglErrorCode::ParsingError;
        }
    };

    // Check if stream is done (complete or error)
    let is_done = is_terminal_response(&json_value);

    // Create C string for converter
    let proto_chunk_json_cstr = match CString::new(proto_chunk_str) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to create C string: {e}"));
            return SglErrorCode::MemoryError;
        }
    };

    // Use the existing converter API
    let mut openai_json_ptr: *mut c_char = ptr::null_mut();
    let result = super::grpc_converter::sgl_grpc_response_converter_convert_chunk(
        converter_handle,
        proto_chunk_json_cstr.as_ptr(),
        &mut openai_json_ptr,
        error_out,
    );

    if result == SglErrorCode::Success {
        *openai_json_out = openai_json_ptr;
        *is_done_out = if is_done { 1 } else { 0 };
        SglErrorCode::Success
    } else {
        *openai_json_out = ptr::null_mut();
        *is_done_out = if is_done { 1 } else { 0 };
        result
    }
}

/// Postprocess multiple gRPC stream chunks in batch (reduces FFI overhead)
///
/// This function processes multiple chunks in a single FFI call, significantly reducing
/// FFI overhead in streaming scenarios.
///
/// # Arguments
/// * `converter_handle` - Converter handle (created with sgl_grpc_response_converter_create)
/// * `proto_chunks_json_array` - JSON array string of proto.GenerateResponse chunks
/// * `max_chunks` - Maximum number of chunks to process (for safety)
/// * `openai_chunks_json_array_out` - Pointer to receive JSON array of OpenAI format chunks (must be freed with sgl_free_string)
/// * `chunks_count_out` - Pointer to receive number of processed chunks
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// - `converter_handle` must be a valid pointer returned by `sgl_grpc_response_converter_create`
/// - `proto_chunks_json_array` must be a valid null-terminated C string containing valid JSON array
/// - `openai_chunks_json_array_out` and `chunks_count_out` must be valid pointers to writable memory
/// - `error_out` may be null; if non-null, must point to writable memory
/// - Caller must free the string written to `openai_chunks_json_array_out` using `sgl_free_string`
#[no_mangle]
pub unsafe extern "C" fn sgl_postprocess_stream_chunks_batch(
    converter_handle: *mut GrpcResponseConverterHandle,
    proto_chunks_json_array: *const c_char,
    max_chunks: c_int,
    openai_chunks_json_array_out: *mut *mut c_char,
    chunks_count_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if converter_handle.is_null()
        || proto_chunks_json_array.is_null()
        || openai_chunks_json_array_out.is_null()
        || chunks_count_out.is_null()
    {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let chunks_array_str = match CStr::from_ptr(proto_chunks_json_array).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in proto_chunks_json_array");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse JSON array of chunks
    let chunks_array: Vec<Value> = match serde_json::from_str(chunks_array_str) {
        Ok(arr) => arr,
        Err(e) => {
            set_error_message(
                error_out,
                &format!("Failed to parse chunks JSON array: {e}"),
            );
            return SglErrorCode::ParsingError;
        }
    };

    // Limit batch size for safety
    let max_chunks_usize = max_chunks as usize;
    let chunks_to_process = if chunks_array.len() > max_chunks_usize {
        &chunks_array[..max_chunks_usize]
    } else {
        &chunks_array
    };

    let handle_ref = &mut *converter_handle;
    let tokenizer = Arc::clone(&handle_ref.tokenizer);

    // Process chunks in batch
    let mut results = Vec::new();
    let mut has_error = false;
    let mut error_msg = String::new();

    for chunk_json in chunks_to_process {
        // Parse proto.GenerateResponse using shared function
        let proto_response = match parse_proto_response(chunk_json) {
            Ok(r) => r,
            Err(e) => {
                error_msg = format!("{e}: {chunk_json}");
                has_error = true;
                break;
            }
        };

        // Convert proto chunk to OpenAI format
        let result = super::grpc_converter::convert_proto_chunk_to_openai(
            proto_response,
            handle_ref,
            &tokenizer,
        );

        match result {
            Ok(Some(openai_response)) => {
                results.push(openai_response);
            }
            Ok(None) => {
                // Empty response, skip
            }
            Err(e) => {
                error_msg = format!("Postprocessing failed for chunk: {e}");
                has_error = true;
                break;
            }
        }
    }

    if has_error {
        set_error_message(error_out, &error_msg);
        return SglErrorCode::ParsingError;
    }

    // Serialize results to JSON array
    let results_json = match serde_json::to_string(&results) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(
                error_out,
                &format!("Failed to serialize results JSON array: {e}"),
            );
            return SglErrorCode::ParsingError;
        }
    };

    let results_cstr = match CString::new(results_json) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to create C string: {e}"));
            return SglErrorCode::MemoryError;
        }
    };

    *openai_chunks_json_array_out = results_cstr.into_raw();
    *chunks_count_out = results.len() as c_int;

    SglErrorCode::Success
}
