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
//!
//! Every servicer exposes the same Prometheus families
//! ([`ServicerConfig::metrics_bind_address`]; the `metrics` module): the
//! requests, tokens and latencies observed at the points the request path
//! already passes, on a listener of their own.

mod admin;
pub mod engine_hash;
mod engine_link;
mod error;
mod health;
pub mod kv_events;
pub mod kv_history;
pub mod kv_state;
pub mod kv_wire;
mod load_report;
mod load_tracker;
mod metrics;
mod proto_json;
mod requests;
mod server;
mod servicer;
mod servicer_config;
pub mod sglang;
mod stop_match;
mod stream;
#[cfg(test)]
mod testing;
mod tokenizer_bundle;
pub mod tokenspeed;
mod trace_context;
pub mod vllm;

use std::{pin::Pin, time::Duration};

pub use engine_zmq_adapter::{
    EncodeDispatchErrorSink, EncodeDispatchOutcome, EncodeDispatchRequest, EncodeFuture,
    EncodeRelease, EncoderDispatchErrorReq, SglangEncodeDispatcher,
};
pub use error::ServicerError;
use futures::Stream;
pub use server::init_tracing;
pub(crate) use server::{lock, record_error, ServerThread, SharedError, Shutdown};
pub use servicer_config::ServicerConfig;
pub use sglang::{SglangModelInfo, SglangServicerConfig, SglangServicerServer};
pub use tokenspeed::{TokenSpeedModelInfo, TokenSpeedServicerConfig, TokenSpeedServicerServer};
use tonic::Status;
pub use vllm::{
    BoxFuture, MediaError, MediaFeatures, MediaProcessor, MediaRefItem, MediaRequest,
    ProcessedMedia, VllmModelInfo, VllmServicerConfig, VllmServicerServer, ALL_MODALITIES,
    PIXEL_MODALITIES,
};

/// A boxed response stream, the shape tonic's generated traits take.
pub(crate) type BoxStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;

/// How long a servicer waits for a sign of life from its engine during the
/// ZMQ handshake before it reports the link failed. An engine's start includes
/// model load, kernel JIT and graph capture, and nothing crosses the wire
/// between its first handshake message and its last; the lifecycle owner
/// reports the engine process alive while it polls it
/// ([`VllmServicerServer::note_engine_alive`],
/// [`SglangServicerServer::note_engine_alive`],
/// [`TokenSpeedServicerServer::note_engine_alive`]), so a healthy start that
/// outlasts this bound still completes, under
/// [`DEFAULT_ENGINE_STARTUP_CEILING`]. A dead engine never waits this long:
/// the lifecycle owner stops the servicer when the process exits.
pub const DEFAULT_ENGINE_STARTUP_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// The most a servicer's engine start may take however alive the engine is: a
/// bound on an engine that stays alive without ever completing its
/// handshake. Large checkpoints have taken over half an hour to load on a
/// cold start; four hours leaves room for the storage path, with the knob
/// above it for anything slower.
pub const DEFAULT_ENGINE_STARTUP_CEILING: Duration = Duration::from_secs(4 * 60 * 60);

/// How long a connected engine may hold requests in flight, in total since its
/// last output, with nothing pending prefill, before the servicer treats it as
/// dead, fails the requests and closes the link (the engine's client then
/// reports it dead and the lifecycle owner restarts the pair). A hard engine
/// death sends nothing over the wire, so silence is the only sign of it; the
/// prompt tokens pending prefill extend the bound at
/// [`DEFAULT_ENGINE_PREFILL_TOKENS_PER_SEC`], so a batch of long prompts on a
/// wire that says nothing until a prompt's first token is not read as a dead
/// engine. The engine client's own default
/// ([`engine_zmq_client::ENGINE_SILENCE_DEATH_TIMEOUT`]).
pub const DEFAULT_ENGINE_SILENCE_TIMEOUT: Duration =
    engine_zmq_client::ENGINE_SILENCE_DEATH_TIMEOUT;

/// The prefill rate the engine-silence bound allows the engine for the prompt
/// tokens pending prefill: a bound on how slowly a healthy engine may prefill
/// (500 s per million pending prompt tokens on top of the window), not an
/// estimate of how fast it does. The engine client's own default
/// ([`engine_zmq_client::ENGINE_PREFILL_TOKENS_PER_SEC`]).
pub const DEFAULT_ENGINE_PREFILL_TOKENS_PER_SEC: u64 =
    engine_zmq_client::ENGINE_PREFILL_TOKENS_PER_SEC;
