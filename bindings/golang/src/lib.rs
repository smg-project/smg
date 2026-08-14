// FFI bindings require unsafe for extern "C" functions, raw pointer handling, and #[no_mangle].
#![allow(unsafe_code)]

//! FFI module for exposing Shepherd Model Gateway preprocessing and postprocessing functions
//! to C-compatible languages (e.g., Golang via cgo)
//!
//! This module provides C-compatible function signatures for:
//! - Tokenizer operations (encode, decode, chat template)
//! - Tool parser operations (parse tool calls)
//! - gRPC client SDK (complete request-response flow)
//!
//! # Safety
//! All functions marked with `#[no_mangle]` and `extern "C"` must be called
//! with valid pointers and follow the documented memory management rules.

// Jemalloc for all Rust-side allocations in the cdylib. Prefixed symbols
// leave the host process's allocator untouched; disable_initial_exec_tls is
// required for a dlopen'd shared library.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// Re-export error types
// Re-export client stream function (defined in client.rs but used by stream)
pub use client::sgl_client_chat_completion_stream;
// Re-export client SDK functions
pub use client::{sgl_client_create, sgl_client_free, SglangClientHandle};
// Re-export multi-worker client with load balancing
pub use error::{clear_error_message, set_error_message, set_error_message_fmt, SglErrorCode};
// Re-export gRPC converter functions
pub use grpc_converter::{
    sgl_grpc_response_converter_convert_chunk, sgl_grpc_response_converter_create,
    sgl_grpc_response_converter_free, GrpcResponseConverterHandle,
};
// Re-export memory management functions
pub use memory::{sgl_free_string, sgl_free_token_ids};
pub use policy::{
    sgl_multi_client_chat_completion_stream, sgl_multi_client_create, sgl_multi_client_free,
    sgl_multi_client_healthy_count, sgl_multi_client_policy_name,
    sgl_multi_client_set_worker_health, sgl_multi_client_tokenizer_path,
    sgl_multi_client_worker_count, MultiWorkerClientHandle,
};
// Re-export postprocessor functions
pub use postprocessor::{sgl_postprocess_stream_chunk, sgl_postprocess_stream_chunks_batch};
// Re-export preprocessor functions
pub use preprocessor::{
    sgl_preprocess_chat_request, sgl_preprocess_chat_request_with_tokenizer,
    sgl_preprocessed_request_free,
};
// Re-export stream functions
pub use stream::{sgl_stream_free, sgl_stream_read_next, SglangStreamHandle};
// Re-export tokenizer functions
pub use tokenizer::{
    sgl_tokenizer_apply_chat_template, sgl_tokenizer_apply_chat_template_with_tools,
    sgl_tokenizer_create_from_file, sgl_tokenizer_decode, sgl_tokenizer_encode, sgl_tokenizer_free,
    TokenizerHandle,
};
// Re-export tool parser functions
pub use tool_parser::{
    sgl_tool_parser_create, sgl_tool_parser_free, sgl_tool_parser_parse_complete,
    sgl_tool_parser_parse_incremental, sgl_tool_parser_reset, ToolParserHandle,
};

// Sub-modules
mod client;
mod error;
mod grpc_converter;
mod memory;
mod policy;
mod postprocessor;
mod preprocessor;
mod proto_parse;
mod runtime;
mod stream;
mod stream_state;
mod tokenizer;
mod tool_parser;
mod utils;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(SglErrorCode::Success as i32, 0);
        assert_eq!(SglErrorCode::InvalidArgument as i32, 1);
    }
}
