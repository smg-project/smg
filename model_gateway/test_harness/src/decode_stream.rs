//! The decode stream a scripted engine answers in the prefill-decode tests:
//! one token per chunk, then the completion that names them all.

use std::time::Duration;

use proto::generate_response::Response as GenResp;
use smg_grpc_client::vllm_proto as proto;
use tokio::sync::mpsc;
use tonic::Status;

/// The sending half of one scripted generate stream.
pub type GenerateTx = mpsc::Sender<Result<proto::GenerateResponse, Status>>;

/// One decode chunk carrying the token `id`.
pub fn chunk(id: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenResp::Chunk(proto::GenerateStreamChunk {
            token_ids: vec![id],
            prompt_tokens: 1,
            completion_tokens: 1,
            index: 0,
            ..Default::default()
        })),
    }
}

/// The completion after `tokens` tokens, with `finish_reason`.
pub fn complete(finish_reason: &str, tokens: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenResp::Complete(proto::GenerateComplete {
            output_ids: (1..=tokens).collect(),
            finish_reason: finish_reason.to_string(),
            prompt_tokens: 1,
            completion_tokens: tokens,
            index: 0,
            ..Default::default()
        })),
    }
}

/// Feed one generate stream: `tokens` chunks after `delay`, then a `stop` completion.
/// A `delay` the test never reaches holds the sender and writes nothing.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
pub fn produce_after(tx: GenerateTx, delay: Duration, tokens: u32) {
    tokio::spawn(async move {
        tokio::time::sleep(delay).await;
        for id in 1..=tokens {
            if tx.send(Ok(chunk(id))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(complete("stop", tokens))).await;
    });
}
