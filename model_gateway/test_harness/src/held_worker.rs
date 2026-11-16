//! A gRPC worker that answers its stream at once and its first token after a
//! delay: the shape of a long prefill, where the gateway has sent the SSE
//! headers and then nothing until the engine speaks.
use std::time::Duration;

use futures::{stream, StreamExt};
use smg_grpc_client::tokenspeed_scheduler::tokenspeed_proto as ts;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};
use ts::generate_response::Response as GenResp;

use crate::engine_answers::{TokenSpeedAnswers, TokenSpeedGenerateStream};

pub struct HeldWorker {
    /// Silence between the stream's opening and its first token.
    pub first_token_after: Duration,
    pub output_tokens: u32,
}

impl HeldWorker {
    #[expect(
        clippy::expect_used,
        reason = "test worker startup or serving failure must fail the fixture"
    )]
    pub async fn serve(self, listener: TcpListener) {
        Server::builder()
            .add_service(self.into_server())
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("held gRPC worker failed");
    }
}

#[tonic::async_trait]
impl TokenSpeedAnswers for HeldWorker {
    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<TokenSpeedGenerateStream>, Status> {
        let request_id = request.into_inner().request_id;
        // Generated token ids begin at 100, as the scripted tokenizer expects.
        let ids: Vec<u32> = (0..self.output_tokens).map(|i| 100 + i).collect();
        let mut items: Vec<Result<ts::GenerateResponse, Status>> = Vec::new();
        for id in &ids {
            items.push(Ok(ts::GenerateResponse {
                request_id: request_id.clone(),
                response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                    token_ids: vec![*id],
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    cached_tokens: 0,
                    output_logprobs: None,
                    weight_version: None,
                    index: 0,
                })),
            }));
        }
        items.push(Ok(ts::GenerateResponse {
            request_id,
            response: Some(GenResp::Complete(ts::GenerateComplete {
                output_ids: ids,
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: self.output_tokens,
                cached_tokens: 0,
                output_logprobs: None,
                matched_stop: None,
                index: 0,
                ..Default::default()
            })),
        }));
        let delay = self.first_token_after;
        // The stream opens now; its first item comes after the delay.
        let held =
            stream::once(tokio::time::sleep(delay)).flat_map(move |()| stream::iter(items.clone()));
        Ok(Response::new(Box::pin(held)))
    }

    async fn abort(
        &self,
        _: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        Ok(Response::new(ts::AbortResponse::default()))
    }
}
