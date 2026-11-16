//! Every engine's `generate` stream yields its response message in a box: the
//! decoder builds the message once on the heap and every reader above moves a
//! pointer. On the by-value shape (the message itself as the stream item, boxed
//! by the first reader and moved whole by each one after it) the type pins below
//! do not compile, and the decode pins show the boxed decode is the by-value
//! decode, byte for byte, on each engine's wire.

use futures::Stream;
use prost::Message;
use smg_grpc_client::{
    mlx_engine, mlx_engine::proto::generate_response::Response as MlxResponse, sglang_scheduler,
    tokenspeed_scheduler, trtllm_service, vllm_engine,
    vllm_engine::proto::generate_response::Response as VllmResponse,
};

/// The item of an engine stream is `Result<Box<M>, Status>`: a type pin, not a
/// size one, so it reads the same on every engine's message size.
fn yields_boxed<S, M>()
where
    S: Stream<Item = Result<Box<M>, tonic::Status>>,
{
}

#[test]
fn every_engine_stream_yields_its_message_in_a_box() {
    yields_boxed::<sglang_scheduler::AbortOnDropStream, _>();
    yields_boxed::<vllm_engine::AbortOnDropStream, _>();
    yields_boxed::<trtllm_service::AbortOnDropStream, _>();
    yields_boxed::<mlx_engine::AbortOnDropStream, _>();
    yields_boxed::<tokenspeed_scheduler::AbortOnDropStream, _>();
}

/// The boxed decode of a response is the by-value decode of the same bytes,
/// and encodes back to them: prost's `Message` for `Box<M>` on each engine's
/// response type, one field set per wire (the request id where the wire has
/// one, a token chunk where it has not).
#[expect(
    clippy::expect_used,
    reason = "a decode of the bytes the same message just encoded cannot fail; a failure is the test's finding"
)]
fn boxed_decode_is_the_by_value_decode<M>(message: M)
where
    M: Message + Default + PartialEq + std::fmt::Debug,
{
    let bytes = message.encode_to_vec();
    let by_value = M::decode(bytes.as_slice()).expect("the by-value decode");
    let boxed = Box::<M>::decode(bytes.as_slice()).expect("the boxed decode");
    assert_eq!(*boxed, by_value);
    assert_eq!(by_value, message);
    assert_eq!(boxed.encode_to_vec(), bytes);
    assert_eq!(boxed.encoded_len(), bytes.len());
}

#[test]
fn the_boxed_decode_matches_on_every_engine_wire() {
    boxed_decode_is_the_by_value_decode(sglang_scheduler::proto::GenerateResponse {
        request_id: "req-sglang-1".to_owned(),
        ..Default::default()
    });
    boxed_decode_is_the_by_value_decode(vllm_engine::proto::GenerateResponse {
        response: Some(VllmResponse::Chunk(
            vllm_engine::proto::GenerateStreamChunk {
                token_ids: vec![7, 11],
                ..Default::default()
            },
        )),
    });
    boxed_decode_is_the_by_value_decode(trtllm_service::proto::GenerateResponse {
        request_id: "req-trtllm-1".to_owned(),
        ..Default::default()
    });
    boxed_decode_is_the_by_value_decode(mlx_engine::proto::GenerateResponse {
        response: Some(MlxResponse::Chunk(mlx_engine::proto::GenerateStreamChunk {
            token_ids: vec![7, 11],
            ..Default::default()
        })),
    });
    boxed_decode_is_the_by_value_decode(tokenspeed_scheduler::tokenspeed_proto::GenerateResponse {
        request_id: "req-tokenspeed-1".to_owned(),
        ..Default::default()
    });
}
