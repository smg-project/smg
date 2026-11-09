//! The decode stream a scripted engine answers in the prefill-decode and
//! latency tests: one token per chunk, then the completion that names them
//! all; and the engine that answers it with its first token in a chosen shape.

use std::time::Duration;

use proto::generate_response::Response as GenResp;
use smg_grpc_client::vllm_proto as proto;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::engine_answers::{VllmAnswers, VllmGenerateStream};

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
pub fn produce_after(tx: GenerateTx, delay: Duration, tokens: u32) {
    produce_from(tx, delay, 1, tokens);
}

/// Feed one generate stream from the chunk `first` on: the chunks
/// `first..=tokens` after `delay`, then the `stop` completion naming all
/// `tokens` (the chunks before `first` were sent by the caller).
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
pub fn produce_from(tx: GenerateTx, delay: Duration, first: u32, tokens: u32) {
    tokio::spawn(async move {
        tokio::time::sleep(delay).await;
        for id in first..=tokens {
            if tx.send(Ok(chunk(id))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(complete("stop", tokens))).await;
    });
}

/// A scripted engine on the vLLM wire whose first token arrives in a chosen
/// shape. `hold` is the wait before the generate call returns its stream: an
/// engine server that flushes its response headers with the first message
/// holds them that long. With `first_chunk_buffered` the first chunk is in the
/// stream when the call returns. `first_token_after` is the stream's own wait
/// before its remaining tokens, `output_tokens` the count of them all.
#[derive(Clone, Copy, Debug)]
pub struct HeldHeadersEngine {
    pub hold: Duration,
    pub first_chunk_buffered: bool,
    pub first_token_after: Duration,
    pub output_tokens: u32,
}

impl HeldHeadersEngine {
    /// An engine that returns its stream at once and its tokens after `delay`.
    pub fn eager(delay: Duration, output_tokens: u32) -> Self {
        Self {
            hold: Duration::ZERO,
            first_chunk_buffered: false,
            first_token_after: delay,
            output_tokens,
        }
    }

    /// An engine that holds its response headers for `hold`, then streams its
    /// tokens at once.
    pub fn holding(hold: Duration, output_tokens: u32) -> Self {
        Self {
            hold,
            first_chunk_buffered: false,
            first_token_after: Duration::ZERO,
            output_tokens,
        }
    }
}

#[tonic::async_trait]
impl VllmAnswers for HeldHeadersEngine {
    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        let (tx, rx) = mpsc::channel(8);
        // The RPC await absorbs the hold, like a Python servicer holding headers.
        if !self.hold.is_zero() {
            tokio::time::sleep(self.hold).await;
        }
        let mut first = 1;
        if self.first_chunk_buffered {
            tx.send(Ok(chunk(1)))
                .await
                .map_err(|_| Status::cancelled("stream closed"))?;
            first = 2;
        }
        produce_from(tx, self.first_token_after, first, self.output_tokens);
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}
