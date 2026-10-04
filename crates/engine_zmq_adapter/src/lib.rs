//! Each engine's own gRPC-proto surface over the same-host ZMQ engine wire.
//!
//! [`ZmqEngineClient`] presents each engine's own gRPC proto request/response
//! types (the contract `smg-grpc-client` speaks to a Python servicer) but talks
//! ZMQ directly to the colocated engine via `engine-zmq-client`, bypassing the
//! Python frontend: vLLM's EngineCore, TokenSpeed's scheduler and SGLang's
//! scheduler (behind the SMG plugin that performs the handshake inside it),
//! each with its protocol module and translation here. It consumes the exact proto requests the builders
//! produce (`vllm::GenerateRequest` for vLLM) and emits the matching proto
//! responses built from the engine's outputs, so a consumer's proto
//! request-execution path is reused unchanged.
//!
//! Two consumers share it: the gateway's direct-ZMQ worker lane, and the Rust
//! engine servicer (`engine-servicer`), which serves the same proto over gRPC
//! on the engine's node. Neither owns the translation; this crate does.
//!
//! Beyond the wire translation, this crate also owns the frontend duties the
//! tokenizer-less EngineCore cannot perform: EOS stop ids
//! ([`fold_tokenizer_eos_backstop`]), the `max_tokens` default, the `n > 1`
//! fan-out, the string-stop resolution helpers in [`stops`], and the pooling
//! params an `Embed` request carries ([`translate_embed_request`]).

mod client;
mod embed;
mod eos;
mod fanout;
pub mod multimodal;
mod sglang;
mod sockets;
pub mod stops;
mod stream;
mod tokenspeed;
mod vllm;

pub use client::{
    connect_for_worker, connect_with_eos, kv_transfer_rejection_params, ZmqDialect,
    ZmqEngineClient, ZmqModelInfo, ZmqServerInfo,
};
pub use embed::translate_embed_request;
pub use engine_zmq_client::protocol::vllm::pooling::{PoolerDefaults, PoolingParams};
pub use eos::{fold_tokenizer_eos_backstop, EosTokenIds};
pub use sglang::SglangGenerateStream;
pub use sockets::zmq_handshake_address;
pub use stream::ZmqGenerateStream;
pub use tokenspeed::{to_tokenspeed_response, TokenSpeedGenerateStream};
pub use vllm::{
    structured_outputs_backend_from_config, ProcessedMedia, StructuredOutputsBackendConfig,
    VllmGenerateStream,
};
