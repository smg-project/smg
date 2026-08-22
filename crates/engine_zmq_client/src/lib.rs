//! Engine-agnostic ZMQ transport + per-engine protocol modules for same-host
//! SMG backend connections.
//!
//! When SMG and an inference engine share a host, the gRPC path
//! (gateway → gRPC → Python servicer → ZMQ → scheduler) collapses to a direct
//! ZMQ connection over `ipc://`, dropping a process, a serialization
//! round-trip, and a context switch per message. This crate owns that wire.
//!
//! # Layout
//!
//! - [`codec`] — engine-agnostic wire primitives: msgpack positional-tuple
//!   serde, numpy dtype handling, and the zero-copy tensor aux-frame codec.
//! - `protocol` — per-engine protocol modules (vLLM EngineCore first).
//! - `transport` — the ZMQ socket topology (SMG binds; engines connect in).
//! - `connector` — request submission, streaming output, and DP wave handling.
//!
//! # Provenance
//!
//! The vLLM EngineCore protocol module is a clean-room port of vLLM's
//! Apache-2.0 `vllm-engine-core-client` crate
//! (vllm-project/vllm, `rust/src/engine-core-client`). vLLM's client is the
//! reference for the protocol layer, not a dependency — see issue #2001.

pub mod codec;
pub mod connector;
mod error;
pub mod protocol;
pub mod transport;

/// Engine-side ZMQ driver for a mock vLLM EngineCore. Compiled for the crate's
/// own loopback tests and, behind the `mock-engine` feature, for out-of-crate
/// consumers such as `mock-worker`.
#[cfg(any(test, feature = "mock-engine"))]
pub mod mock_engine;

// Crate-root shortcuts for what consumers actually build against; everything
// else stays reachable through its own module path.
pub use connector::{
    Client, EngineCoreClient, EngineCoreStream, RequestStream, TokenSpeedClient, TokenSpeedStream,
};
pub use error::{Error, Result};
pub use transport::{connect_handshake, ConnectedEngine, EngineId, ENGINE_CORE_DEAD_SENTINEL};
