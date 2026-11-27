// FFI bindings require unsafe for extern "C" functions, raw pointer handling, and #[no_mangle].
#![allow(unsafe_code)]

//! FFI module for exposing Shepherd Model Gateway preprocessing and postprocessing functions
//! to C-compatible languages (e.g., Golang via cgo)
//!
//! This module provides C-compatible function signatures for:
//! - Tokenizer operations (encode, chat template)
//! - Response conversion (the engine's chunks as the client's deltas, with
//!   tool calls and reasoning read through the Symphony crate's tables)
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

// The gateway's jemalloc options (`SERVER_MALLOC_CONF`). The library is a
// final artifact of its own, so it exports them as jemalloc's
// application-provided `malloc_conf` like the `smg` executable and the Python
// extension do; without this the library ran jemalloc's stock decay inside
// the host process, where a burst's freed pages stayed resident for as long
// as the thread that freed them allocated nothing more, on one arena per
// thread.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
use smg::observability::metrics::SERVER_MALLOC_CONF;
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 100] = SERVER_MALLOC_CONF;

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
    sgl_tokenizer_create_from_file, sgl_tokenizer_encode, sgl_tokenizer_free, TokenizerHandle,
};

// Sub-modules
mod args;
mod chat_stream;
mod client;
mod error;
mod grpc_converter;
mod memory;
mod policy;
mod postprocessor;
mod preprocessor;
mod proto_parse;
mod reader;
mod runtime;
mod stream;
mod stream_state;
mod tokenizer;
mod tool_constraint;
mod utils;

#[cfg(all(test, not(target_env = "msvc"), not(target_env = "musl")))]
#[path = "../tests/unit/allocator_tests.rs"]
mod allocator_tests;

#[cfg(test)]
#[path = "../tests/unit/tests.rs"]
mod tests;
