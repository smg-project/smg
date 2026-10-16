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
pub static MALLOC_CONF: &[u8; 72] = SERVER_MALLOC_CONF;

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
mod args;
mod chat_stream;
mod client;
mod error;
mod forced_tool_choice;
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
mod tool_constraint;
mod tool_parser;
mod utils;

#[cfg(all(test, not(target_env = "msvc"), not(target_env = "musl")))]
mod allocator_tests {
    use tikv_jemalloc_ctl::{opt, raw};

    #[test]
    fn rust_allocations_use_global_jemalloc() {
        let allocated =
            tikv_jemalloc_ctl::thread::allocatedp::read().expect("thread allocation counter");
        let before = allocated.get();
        let allocation = std::hint::black_box(vec![0_u8; 1024 * 1024]);
        let after = allocated.get();
        assert!(after > before, "Rust allocation bypassed jemalloc");
        std::hint::black_box(&allocation);
    }

    /// The options jemalloc runs with, read back through its control
    /// interface: the background purging thread, the dirty and muzzy decay
    /// times in milliseconds and the arena count.
    fn options_in_effect() -> Result<(bool, isize, isize, u32), tikv_jemalloc_ctl::Error> {
        let background_thread = opt::background_thread::read()?;
        let narenas = opt::narenas::read()?;
        // SAFETY: NUL-terminated option names read into the types jemalloc
        // documents for them (both `ssize_t`).
        let (dirty_decay_ms, muzzy_decay_ms) = unsafe {
            (
                raw::read::<isize>(b"opt.dirty_decay_ms\0")?,
                raw::read::<isize>(b"opt.muzzy_decay_ms\0")?,
            )
        };
        Ok((background_thread, dirty_decay_ms, muzzy_decay_ms, narenas))
    }

    #[test]
    fn the_library_applies_the_server_malloc_conf() {
        // The crate's `_rjem_malloc_conf` symbol: purge on a background
        // thread, dirty pages back to the OS after 10 s, muzzy pages at once,
        // and 16 arenas instead of four per CPU.
        #[cfg_attr(target_vendor = "apple", expect(unused_variables))]
        let (background_thread, dirty_decay_ms, muzzy_decay_ms, narenas) =
            options_in_effect().expect("the options in effect");
        // jemalloc does not compile background-thread support for Mach-O.
        #[cfg(not(target_vendor = "apple"))]
        assert!(
            background_thread,
            "jemalloc did not take the library's malloc_conf: no background thread"
        );
        assert_eq!(
            (dirty_decay_ms, muzzy_decay_ms, narenas),
            (10_000, 0, 16),
            "jemalloc did not take the library's malloc_conf (dirty_decay_ms, muzzy_decay_ms, narenas)"
        );
    }

    #[test]
    fn environment_overrides_the_server_malloc_conf_entry_by_entry() {
        // jemalloc reads `_RJEM_MALLOC_CONF` after the exported options, entry
        // by entry; a child process of this test binary runs the probe below
        // under an override of two of them.
        let output = std::process::Command::new(std::env::current_exe().expect("this test binary"))
            .args([
                "--exact",
                "allocator_tests::print_the_options_in_effect",
                "--include-ignored",
                "--nocapture",
            ])
            .env(
                "_RJEM_MALLOC_CONF",
                "background_thread:false,dirty_decay_ms:2000",
            )
            .output()
            .expect("run the probe");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "the probe failed: {stdout}{stderr}"
        );
        assert!(
            stdout.contains(
                "options in effect: background_thread=false dirty_decay_ms=2000 muzzy_decay_ms=0 narenas=16"
            ),
            "_RJEM_MALLOC_CONF did not override the library's malloc_conf entry by entry: {stdout}"
        );
    }

    /// Run by `environment_overrides_the_server_malloc_conf_entry_by_entry`
    /// in a child process; ignored so that a normal run prints nothing.
    #[test]
    #[ignore = "a probe that environment_overrides_the_server_malloc_conf_entry_by_entry runs in a child process"]
    #[expect(
        clippy::print_stdout,
        reason = "the parent test reads this line from the child's stdout"
    )]
    fn print_the_options_in_effect() {
        let (background_thread, dirty_decay_ms, muzzy_decay_ms, narenas) =
            options_in_effect().expect("the options in effect");
        println!(
            "options in effect: background_thread={background_thread} dirty_decay_ms={dirty_decay_ms} muzzy_decay_ms={muzzy_decay_ms} narenas={narenas}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(SglErrorCode::Success as i32, 0);
        assert_eq!(SglErrorCode::InvalidArgument as i32, 1);
    }
}
