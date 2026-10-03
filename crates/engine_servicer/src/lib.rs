//! Rust engine servicers: the per-engine gRPC contract that the Python
//! `smg_grpc_servicer` package serves today, implemented in Rust over the
//! same-host ZMQ engine adapter (`engine-zmq-adapter`).
//!
//! The Router keeps speaking the engine's own proto (`vllm_engine.proto` for
//! vLLM; each further ZMQ engine's servicer serves its own), so it cannot
//! tell this servicer from the Python one. What changes is the node:
//! the Python frontend leaves the request path, and the engine is reached the
//! way the direct-ZMQ lane already reaches it. Python keeps the lifecycle only
//! — it launches the headless engine and drives the server through the PyO3
//! binding — which is why the server runs on its own thread and reports back
//! through plain flags instead of a Python-visible runtime.

mod error;
mod health;
mod server;
pub mod vllm;

use std::pin::Pin;

pub use error::ServicerError;
use futures::Stream;
pub use server::init_tracing;
pub(crate) use server::{lock, record_error, ServerThread, SharedError, Shutdown};
use tonic::Status;
pub use vllm::{
    BoxFuture, MediaError, MediaFeatures, MediaProcessor, MediaRefItem, MediaRequest,
    ProcessedMedia, VllmModelInfo, VllmServicerConfig, VllmServicerServer,
};

/// A boxed response stream, the shape tonic's generated traits take.
pub(crate) type BoxStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;
