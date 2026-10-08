//! Rust engine servicers: the per-engine gRPC contract that the Python
//! `smg_grpc_servicer` package serves today, implemented in Rust over the
//! same-host ZMQ engine adapter (`engine-zmq-adapter`).
//!
//! The Router keeps speaking the engine's own proto (`vllm_engine.proto` for
//! vLLM, `sglang_scheduler.proto` for SGLang, and so on per engine), so it cannot
//! tell this servicer from the Python one. What changes is the node:
//! the Python frontend leaves the request path, and the engine is reached the
//! way the direct-ZMQ lane already reaches it. Python keeps the lifecycle only
//! — it launches the headless engine and drives the server through the PyO3
//! binding — which is why the server runs on its own thread and reports back
//! through plain flags instead of a Python-visible runtime.
//!
//! The engines' ZMQ KV-cache event publishers are relayed by [`kv_events`]
//! after [`kv_wire`] normalizes them; that module documents the per-engine
//! hash folding rule and the one-for-one forwarding of stores and removals.
//! [`kv_state`] keeps the engine's live blocks from that stream, the state
//! snapshot a subscriber receives once the relay's history has rolled.

pub mod engine_hash;
mod engine_link;
mod error;
mod health;
pub mod kv_events;
pub mod kv_history;
pub mod kv_state;
pub mod kv_wire;
mod load_tracker;
mod proto_json;
mod requests;
mod server;
pub mod sglang;
mod stop_match;
#[cfg(test)]
mod testing;
mod tokenizer_bundle;
pub mod tokenspeed;
pub mod vllm;

use std::{pin::Pin, time::Duration};

pub use error::ServicerError;
use futures::Stream;
pub use server::init_tracing;
pub(crate) use server::{lock, record_error, ServerThread, SharedError, Shutdown};
pub use sglang::{SglangModelInfo, SglangServicerConfig, SglangServicerServer};
pub use tokenspeed::{TokenSpeedModelInfo, TokenSpeedServicerConfig, TokenSpeedServicerServer};
use tonic::Status;
pub use vllm::{
    BoxFuture, MediaError, MediaFeatures, MediaProcessor, MediaRefItem, MediaRequest,
    ProcessedMedia, VllmModelInfo, VllmServicerConfig, VllmServicerServer,
};

/// A boxed response stream, the shape tonic's generated traits take.
pub(crate) type BoxStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

/// How long a servicer waits for its engine to complete the ZMQ handshake
/// before it reports the link failed. An engine's start includes model load,
/// kernel JIT and graph capture; a cold kernel cache has taken over ten
/// minutes. A dead engine never waits this long: the lifecycle owner polls the
/// engine process and stops the servicer when it exits.
pub const DEFAULT_ENGINE_STARTUP_TIMEOUT: Duration = Duration::from_secs(30 * 60);
