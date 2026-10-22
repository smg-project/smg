//! gRPC router integration tests

#[path = "common/mod.rs"]
pub mod common;
#[path = "common/scripted_tokenizer.rs"]
#[expect(
    dead_code,
    reason = "a shared fixture; the tests of this binary use part of it"
)]
mod scripted_tokenizer;
#[path = "common/scripted_worker.rs"]
mod scripted_worker;

mod grpc;
