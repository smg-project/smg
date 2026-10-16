//! Stream handling FFI functions

use std::{
    ffi::CString,
    os::raw::{c_char, c_int},
    ptr,
    sync::Arc,
};

use futures_util::StreamExt;
use smg::worker::Worker;
use smg_grpc_client::{
    sglang_proto as proto,
    sglang_scheduler::{AbortOnDropStream, SglangSchedulerClient},
};

use super::{
    error::{set_error_message, SglErrorCode},
    grpc_converter::{convert_proto_chunk_to_openai, GrpcResponseConverterHandle},
    policy::HeldLoad,
    runtime::RUNTIME,
};

/// Handle for an active streaming request.
///
/// This struct manages the stream and response converter for a single request.
/// It is wrapped in Arc and Mutex for thread-safe concurrent access.
///
/// # Fields
///
/// * `stream` - The gRPC stream wrapped in AbortOnDropStream for automatic cleanup
/// * `converter` - Response converter that transforms proto messages to OpenAI format
/// * `client` - The underlying gRPC client connection
/// * `prompt_tokens` - Number of prompt tokens from the original request
pub struct SglangStreamHandle {
    pub(crate) stream: Arc<tokio::sync::Mutex<AbortOnDropStream>>,
    pub(crate) converter: Arc<tokio::sync::Mutex<GrpcResponseConverterHandle>>,
    #[expect(dead_code)]
    pub(crate) client: Arc<SglangSchedulerClient>,
    #[expect(dead_code)]
    pub(crate) prompt_tokens: u32, // Number of prompt tokens for this request
    /// The load this stream holds on the worker a policy selected, given
    /// back when the handle is freed. None for single-client streams.
    pub(crate) load: Option<HeldLoad>,
}

/// Read next chunk from stream and convert to OpenAI format.
///
/// This function reads the next chunk from the gRPC stream, converts it from the
/// internal protocol format to OpenAI-compatible JSON format, and returns it via
/// the output parameters.
///
/// # Arguments
///
/// * `stream_handle` - Mutable pointer to the stream handle
/// * `response_json_out` - Pointer to receive OpenAI format JSON string
///   - Caller must free this with `sgl_free_string`
///   - May be NULL if no data available
/// * `is_done_out` - Pointer to receive completion status
///   - 0 = stream has more data
///   - 1 = stream is complete
/// * `error_out` - Optional pointer to receive error message
///   - Only set if function returns an error code
///   - Must be freed with `sgl_free_string` if not NULL
///
/// # Returns
///
/// * `SglErrorCode::Success` - Successfully read a chunk or reached end of stream
/// * Other error codes - See `SglErrorCode` for details
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - `stream_handle` must point to a valid `SglangStreamHandle`
/// - Output pointers must be writable
///
/// # Notes
///
/// - Complete messages are identified by the presence of `proto::GenerateResponse::Complete`
/// - When is_done=1, this may be the last readable chunk or the stream may be ending
/// - Subsequent calls after is_done=1 will mark the stream as complete internally
#[no_mangle]
pub unsafe extern "C" fn sgl_stream_read_next(
    stream_handle: *mut SglangStreamHandle,
    response_json_out: *mut *mut c_char,
    is_done_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if stream_handle.is_null() || response_json_out.is_null() || is_done_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let handle_ref = &*stream_handle;
    let stream = Arc::clone(&handle_ref.stream);
    let converter = Arc::clone(&handle_ref.converter);

    // Read next chunk from stream
    let chunk_result = RUNTIME.block_on(async {
        let mut stream_guard = stream.lock().await;
        stream_guard.next().await
    });

    match chunk_result {
        Some(Ok(proto_response)) => {
            // Convert proto response to OpenAI format
            // We need to get the converter lock first
            let conversion_result = RUNTIME.block_on(async {
                let mut converter_guard = converter.lock().await;

                let tokenizer = Arc::clone(&converter_guard.tokenizer);
                convert_proto_chunk_to_openai(
                    proto_response.clone(),
                    &mut converter_guard,
                    &tokenizer,
                )
                .await
            });

            match conversion_result {
                Ok(Some(openai_response)) => {
                    // Serialize to JSON
                    let result_str = match serde_json::to_string(&openai_response) {
                        Ok(s) => s,
                        Err(e) => {
                            set_error_message(
                                error_out,
                                &format!("Failed to serialize response: {e}"),
                            );
                            return SglErrorCode::ParsingError;
                        }
                    };

                    let result_cstr = match CString::new(result_str) {
                        Ok(s) => s,
                        Err(e) => {
                            set_error_message(
                                error_out,
                                &format!("Failed to create result string: {e}"),
                            );
                            return SglErrorCode::MemoryError;
                        }
                    };

                    // Check if this is a complete response (stream done)
                    let is_complete = matches!(
                        proto_response.response,
                        Some(proto::generate_response::Response::Complete(_))
                    );

                    *response_json_out = result_cstr.into_raw();
                    *is_done_out = if is_complete { 1 } else { 0 };

                    if is_complete {
                        // Mark stream as completed to prevent abort on drop
                        RUNTIME.block_on(async {
                            stream.lock().await.mark_completed();
                        });
                    }

                    SglErrorCode::Success
                }
                Ok(None) => {
                    // No response to send (e.g., empty chunk)
                    // Don't mark as completed - stream might continue
                    // Just return null and let caller read more
                    *response_json_out = ptr::null_mut();
                    *is_done_out = 0; // Keep stream open, not done yet
                    SglErrorCode::Success
                }
                Err(e) => {
                    // Conversion error - don't mark as completed
                    // Let the stream end naturally or return error without stopping stream
                    set_error_message(error_out, &format!("Conversion error: {e}"));
                    *response_json_out = ptr::null_mut();
                    *is_done_out = 0; // Don't mark as done - let caller decide
                    SglErrorCode::ParsingError
                }
            }
        }
        Some(Err(e)) => {
            // Stream error - mark as completed to prevent abort
            RUNTIME.block_on(async {
                stream.lock().await.mark_completed();
            });

            set_error_message(error_out, &format!("Stream error: {e}"));
            *is_done_out = 1;
            SglErrorCode::UnknownError
        }
        None => {
            // Stream ended naturally - mark as completed to prevent abort
            RUNTIME.block_on(async {
                stream.lock().await.mark_completed();
            });

            *response_json_out = ptr::null_mut();
            *is_done_out = 1;
            SglErrorCode::Success
        }
    }
}

/// Free a stream handle and release all associated resources.
///
/// This function must be called exactly once for each stream handle returned by
/// `sgl_client_chat_completion_stream`. It marks the stream as completed internally
/// to prevent abort signals from being sent when resources are cleaned up.
///
/// # Arguments
///
/// * `handle` - Mutable pointer to the stream handle to free
///   - If NULL, this function does nothing
///
/// # Safety
///
/// - Must be called only once per handle
/// - Handle must not be used after calling this function
/// - After this call, the stream is no longer valid
///
/// # Notes
///
/// - This function calls `mark_completed()` before freeing to ensure
///   the stream cleanup doesn't trigger an abort RPC to the server
#[no_mangle]
pub unsafe extern "C" fn sgl_stream_free(handle: *mut SglangStreamHandle) {
    if !handle.is_null() {
        let mut handle_ref = Box::from_raw(handle);

        // The worker's bookkeeping for a stream the caller is done with:
        // its load given back, the request counted as processed.
        if let Some(load) = handle_ref.load.take() {
            load.release().increment_processed();
        }

        // Mark stream as completed to prevent abort on drop
        // (should already be marked by ReadNext, but ensure it for safety)
        RUNTIME.block_on(async {
            handle_ref.stream.lock().await.mark_completed();
        });

        // Drop handle - mark_completed() ensures no abort signal is sent
        drop(handle_ref);
    }
}
