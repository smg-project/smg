//! The vLLM/TokenSpeed gRPC-proto surface over the same-host ZMQ engine wire.
//!
//! [`ZmqEngineClient`] presents each engine's own gRPC proto request/response
//! types (the contract `smg-grpc-client` speaks to a Python servicer) but talks
//! ZMQ directly to a colocated vLLM EngineCore or TokenSpeed scheduler via
//! `engine-zmq-client`, bypassing the Python frontend. It consumes the exact
//! `vllm::GenerateRequest` the proto builders produce and emits
//! `vllm::GenerateResponse` built from `EngineCoreOutput`, so a consumer's
//! proto request-execution path is reused unchanged.
//!
//! Two consumers share it: the gateway's direct-ZMQ worker lane, and the Rust
//! engine servicer (`engine-servicer`), which serves the same proto over gRPC
//! on the engine's node. Neither owns the translation; this crate does.
//!
//! Beyond the wire translation, this crate also owns the frontend duties the
//! tokenizer-less EngineCore cannot perform: EOS stop ids
//! ([`fold_tokenizer_eos_backstop`]), the `max_tokens` default, the `n > 1`
//! fan-out, and the string-stop resolution helpers in [`stops`].

mod client;
mod eos;
mod fanout;
pub mod multimodal;
mod sockets;
pub mod stops;
mod stream;
mod tokenspeed;
mod vllm;

pub use client::{
    connect_for_worker, connect_with_eos, kv_transfer_rejection_params, ZmqDialect,
    ZmqEngineClient, ZmqModelInfo, ZmqServerInfo,
};
pub use eos::{fold_tokenizer_eos_backstop, EosTokenIds};
pub use sockets::zmq_handshake_address;
pub use stream::ZmqGenerateStream;
pub use tokenspeed::TokenSpeedGenerateStream;
pub use vllm::VllmGenerateStream;
