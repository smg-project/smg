//! The framed generate call: the request's large inline tensors ride as
//! body frames of their own and the engine receives the request it was,
//! driven through the real `VllmEngineClient` against the in-process mock
//! engine, whose decoder's message limit is raised above tonic's 4 MiB
//! default as the engine servicers raise theirs. A failing engine reads
//! back as the status it named: a trailers-only answer is the error, not a
//! stream.

use std::collections::HashMap;

use bytes::Bytes;
use futures::StreamExt;
use smg_grpc_client::{vllm_proto as proto, VllmEngineClient};

mod common;
use common::spawn_mock_decoding_up_to;

/// The 1080p float32 class: 25 MB, above tonic's default message limit.
const CLASS_1080P: usize = 1920 * 1080 * 3 * 4;
/// The limit the engine servicers and the mock worker decode up to.
const DECODE_LIMIT: usize = 64 << 20;

fn inline_tensor(fill: u8, len: usize) -> proto::TensorData {
    proto::TensorData {
        shape: vec![1, (len / 4) as u32],
        dtype: "float32".into(),
        payload: Some(proto::tensor_data::Payload::Inline(Bytes::from(vec![
            fill;
            len
        ]))),
    }
}

/// A request with a 1080p-class tensor in `pixel_values`, a small one in
/// the map and a frame-sized one in an extra media input.
fn framed_request() -> proto::GenerateRequest {
    proto::GenerateRequest {
        request_id: "framed-1".to_string(),
        mm_inputs: Some(proto::MultimodalInputs {
            pixel_values: Some(inline_tensor(1, CLASS_1080P)),
            model_specific_tensors: HashMap::from([(
                "image_grid_thw".to_string(),
                inline_tensor(2, 24),
            )]),
            ..Default::default()
        }),
        extra_mm_inputs: vec![proto::MultimodalInputs {
            pixel_values: Some(inline_tensor(3, 64 * 1024)),
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[tokio::test]
async fn a_framed_request_reaches_the_engine_as_the_request_it_was() {
    let (endpoint, state) = spawn_mock_decoding_up_to(DECODE_LIMIT).await.unwrap();
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let request = framed_request();
    let mut stream = client
        .generate_framed(request.clone())
        .await
        .expect("the framed call is accepted");
    // The engine's side of the stream, let go: the stream ends clean.
    drop(state.lock_tx().take());
    assert!(stream.next().await.is_none(), "no message, a clean end");
    assert_eq!(
        state.last_generated().as_ref(),
        Some(&request),
        "the engine decoded the request the client framed"
    );
}

#[tokio::test]
async fn a_failing_engine_reads_back_as_its_status() {
    let (endpoint, state) = spawn_mock_decoding_up_to(DECODE_LIMIT).await.unwrap();
    state.fail_generate_with(tonic::Code::Unavailable);
    let client = VllmEngineClient::connect(&endpoint).await.unwrap();
    let error = client
        .generate_framed(framed_request())
        .await
        .err()
        .expect("the trailers-only status");
    assert_eq!(error.code(), tonic::Code::Unavailable);
}
