//! An engine that finishes a request with an error ends the stream with the `Internal` status of
//! `ENGINE_ERROR_MESSAGE`, on every wire.

use super::super::*;

/// One `Complete` per backend finishing with `reason`.
fn completes(reason: &str) -> Vec<ProtoGenerateResponse> {
    let reason = reason.to_string();
    vec![
        ProtoGenerateResponse::Sglang(Box::new(sglang::GenerateResponse {
            response: Some(sglang::generate_response::Response::Complete(
                sglang::GenerateComplete {
                    finish_reason: reason.clone(),
                    ..Default::default()
                },
            )),
            ..Default::default()
        })),
        ProtoGenerateResponse::Vllm(Box::new(vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Complete(
                vllm::GenerateComplete {
                    finish_reason: reason.clone(),
                    ..Default::default()
                },
            )),
        })),
        ProtoGenerateResponse::Trtllm(Box::new(trtllm::GenerateResponse {
            response: Some(trtllm::generate_response::Response::Complete(
                trtllm::GenerateComplete {
                    finish_reason: reason.clone(),
                    ..Default::default()
                },
            )),
            ..Default::default()
        })),
        ProtoGenerateResponse::Mlx(Box::new(mlx::GenerateResponse {
            response: Some(mlx::generate_response::Response::Complete(
                mlx::GenerateComplete {
                    finish_reason: reason.clone(),
                    ..Default::default()
                },
            )),
        })),
        ProtoGenerateResponse::TokenSpeed(Box::new(tokenspeed::GenerateResponse {
            response: Some(tokenspeed::generate_response::Response::Complete(
                tokenspeed::GenerateComplete {
                    finish_reason: reason,
                    ..Default::default()
                },
            )),
            ..Default::default()
        })),
    ]
}

#[test]
fn engine_error_is_true_only_for_complete_with_error_finish_reason() {
    assert!(completes("error")
        .iter()
        .all(ProtoGenerateResponse::engine_error));
    for reason in ["stop", "length", "abort", ""] {
        assert!(!completes(reason)
            .iter()
            .any(ProtoGenerateResponse::engine_error));
    }
    let chunk = ProtoGenerateResponse::Vllm(Box::new(vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Chunk(
            vllm::GenerateStreamChunk::default(),
        )),
    }));
    assert!(!chunk.engine_error());
    let empty = ProtoGenerateResponse::Vllm(Box::default());
    assert!(!empty.engine_error());
}

#[test]
fn reject_engine_error_turns_error_complete_into_internal_status() {
    for response in completes("error") {
        let Err(status) = reject_engine_error(Ok(response)) else {
            panic!("engine error must not pass through as a completion");
        };
        assert_eq!(status.code(), tonic::Code::Internal);
        assert_eq!(
            status.message(),
            "engine finished the request with an error (see engine logs)"
        );
    }
    for response in completes("stop") {
        assert!(reject_engine_error(Ok(response)).is_ok());
    }
    let passed = reject_engine_error(Err(tonic::Status::unavailable("side died")));
    assert!(matches!(passed, Err(status) if status.code() == tonic::Code::Unavailable));
}
