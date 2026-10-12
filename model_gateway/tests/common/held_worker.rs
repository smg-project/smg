//! A gRPC worker that answers its stream at once and its first token after a
//! delay: the shape of a long prefill, where the gateway has sent the SSE
//! headers and then nothing until the engine speaks.
use std::{pin::Pin, time::Duration};

use futures::{stream, Stream, StreamExt};
use smg_grpc_client::{common_proto as common, tokenspeed_scheduler::tokenspeed_proto as ts};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};
use ts::{
    generate_response::Response as GenResp,
    token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
};

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
            .add_service(TokenSpeedSchedulerServer::new(self))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("held gRPC worker failed");
    }
}

type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;
type KvStream = Pin<Box<dyn Stream<Item = Result<common::KvEventBatch, Status>> + Send>>;
type TokenizerStream =
    Pin<Box<dyn Stream<Item = Result<common::GetTokenizerChunk, Status>> + Send>>;

#[tonic::async_trait]
impl TokenSpeedScheduler for HeldWorker {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = KvStream;
    type GetTokenizerStream = TokenizerStream;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
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

    async fn health_check(
        &self,
        _: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        Err(Status::unimplemented("held worker"))
    }

    async fn abort(
        &self,
        _: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        Ok(Response::new(ts::AbortResponse::default()))
    }

    async fn get_model_info(
        &self,
        _: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("held worker"))
    }

    async fn get_server_info(
        &self,
        _: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("held worker"))
    }

    async fn get_loads(
        &self,
        _: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("held worker"))
    }

    async fn subscribe_kv_events(
        &self,
        _: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("held worker"))
    }

    async fn flush_cache(
        &self,
        _: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("held worker"))
    }

    async fn start_profile(
        &self,
        _: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("held worker"))
    }

    async fn stop_profile(
        &self,
        _: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("held worker"))
    }

    async fn get_tokenizer(
        &self,
        _: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("held worker"))
    }
}
