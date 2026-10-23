use super::*;

#[expect(
    dead_code,
    reason = "the shared peer direct client is used by its grpc_client tests"
)]
#[path = "../../../../../../../crates/grpc_client/tests/common/sglang_encoder.rs"]
mod encoder_peer;

mod media;
mod stage;
mod transport;
