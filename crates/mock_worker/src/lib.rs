//! Library surface for `mock-worker`'s HTTP/gRPC simulators, so both the
//! standalone binary and in-process integration tests can drive them.

pub mod config;
pub mod engine;
pub mod grpc;
pub mod http;
pub mod zmq;
