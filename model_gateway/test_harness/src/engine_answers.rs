//! The answers a test's mock engine gives to the calls it does not script.
//!
//! A mock engine owns one or two RPCs of its proto's service and used to
//! spell the rest by hand, eleven or twelve methods per mock. Each proto has
//! one answer trait here with a default per RPC and one service wrapper whose
//! single implementation of the generated service trait hands every call to
//! the answers, so a mock writes only the methods its test scripts. The
//! defaults are what every mock answered before it shared them: on the
//! TokenSpeed proto a mock answers nothing it does not script; on the vLLM
//! proto it is healthy, aborts, flushes and reports default info, and answers
//! nothing else.

use std::pin::Pin;

use futures::Stream;
use smg_grpc_client::{
    common_proto as common, tokenspeed_scheduler::tokenspeed_proto as ts, vllm_proto as vllm,
};
use tonic::{Request, Response, Status};
use ts::token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer};
use vllm::vllm_engine_server::{VllmEngine, VllmEngineServer};

/// A mock's generate stream on the TokenSpeed proto.
pub type TokenSpeedGenerateStream =
    Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;
/// A mock's generate stream on the vLLM proto.
pub type VllmGenerateStream =
    Pin<Box<dyn Stream<Item = Result<vllm::GenerateResponse, Status>> + Send>>;
/// A mock's KV-event stream, the same on every proto.
pub type KvEventStream = Pin<Box<dyn Stream<Item = Result<common::KvEventBatch, Status>> + Send>>;
/// A mock's tokenizer stream, the same on every proto.
pub type TokenizerStream =
    Pin<Box<dyn Stream<Item = Result<common::GetTokenizerChunk, Status>> + Send>>;

/// The status of a call the mock does not script.
fn unscripted(rpc: &str) -> Status {
    Status::unimplemented(format!("{rpc} is not scripted by this mock engine"))
}

/// A mock engine's answers on the TokenSpeed scheduler proto. Every method
/// has a default that answers `UNIMPLEMENTED`, so a mock implements the calls
/// its test scripts; [`TokenSpeedService`] serves the answers.
#[tonic::async_trait]
pub trait TokenSpeedAnswers: Send + Sync + 'static {
    async fn generate(
        &self,
        _request: Request<ts::GenerateRequest>,
    ) -> Result<Response<TokenSpeedGenerateStream>, Status> {
        Err(unscripted("generate"))
    }

    async fn health_check(
        &self,
        _request: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        Err(unscripted("health_check"))
    }

    async fn abort(
        &self,
        _request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        Err(unscripted("abort"))
    }

    async fn get_model_info(
        &self,
        _request: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Err(unscripted("get_model_info"))
    }

    async fn get_server_info(
        &self,
        _request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Err(unscripted("get_server_info"))
    }

    async fn get_loads(
        &self,
        _request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        Err(unscripted("get_loads"))
    }

    async fn subscribe_kv_events(
        &self,
        _request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<KvEventStream>, Status> {
        Err(unscripted("subscribe_kv_events"))
    }

    async fn flush_cache(
        &self,
        _request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(unscripted("flush_cache"))
    }

    async fn start_profile(
        &self,
        _request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(unscripted("start_profile"))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(unscripted("stop_profile"))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<TokenizerStream>, Status> {
        Err(unscripted("get_tokenizer"))
    }

    /// The tonic server over these answers.
    fn into_server(self) -> TokenSpeedSchedulerServer<TokenSpeedService<Self>>
    where
        Self: Sized,
    {
        TokenSpeedSchedulerServer::new(TokenSpeedService(self))
    }
}

/// The TokenSpeed scheduler service over a mock's answers.
pub struct TokenSpeedService<A>(A);

#[tonic::async_trait]
impl<A: TokenSpeedAnswers> TokenSpeedScheduler for TokenSpeedService<A> {
    type GenerateStream = TokenSpeedGenerateStream;
    type SubscribeKvEventsStream = KvEventStream;
    type GetTokenizerStream = TokenizerStream;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.0.generate(request).await
    }

    async fn health_check(
        &self,
        request: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        self.0.health_check(request).await
    }

    async fn abort(
        &self,
        request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        self.0.abort(request).await
    }

    async fn get_model_info(
        &self,
        request: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        self.0.get_model_info(request).await
    }

    async fn get_server_info(
        &self,
        request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        self.0.get_server_info(request).await
    }

    async fn get_loads(
        &self,
        request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        self.0.get_loads(request).await
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        self.0.subscribe_kv_events(request).await
    }

    async fn flush_cache(
        &self,
        request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        self.0.flush_cache(request).await
    }

    async fn start_profile(
        &self,
        request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        self.0.start_profile(request).await
    }

    async fn stop_profile(
        &self,
        request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        self.0.stop_profile(request).await
    }

    async fn get_tokenizer(
        &self,
        request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        self.0.get_tokenizer(request).await
    }
}

/// A mock engine's answers on the vLLM engine proto. The defaults are a
/// healthy engine that aborts, flushes and reports default info, and answers
/// `UNIMPLEMENTED` to everything else; [`VllmService`] serves the answers.
#[tonic::async_trait]
pub trait VllmAnswers: Send + Sync + 'static {
    async fn generate(
        &self,
        _request: Request<vllm::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        Err(unscripted("generate"))
    }

    async fn embed(
        &self,
        _request: Request<vllm::EmbedRequest>,
    ) -> Result<Response<vllm::EmbedResponse>, Status> {
        Err(unscripted("embed"))
    }

    async fn health_check(
        &self,
        _request: Request<vllm::HealthCheckRequest>,
    ) -> Result<Response<vllm::HealthCheckResponse>, Status> {
        Ok(Response::new(vllm::HealthCheckResponse {
            healthy: true,
            ..Default::default()
        }))
    }

    async fn flush_cache(
        &self,
        _request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Ok(Response::new(common::FlushCacheResponse::default()))
    }

    async fn start_profile(
        &self,
        _request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(unscripted("start_profile"))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(unscripted("stop_profile"))
    }

    async fn abort(
        &self,
        _request: Request<vllm::AbortRequest>,
    ) -> Result<Response<vllm::AbortResponse>, Status> {
        Ok(Response::new(vllm::AbortResponse::default()))
    }

    async fn get_model_info(
        &self,
        _request: Request<vllm::GetModelInfoRequest>,
    ) -> Result<Response<vllm::GetModelInfoResponse>, Status> {
        Ok(Response::new(vllm::GetModelInfoResponse::default()))
    }

    async fn get_server_info(
        &self,
        _request: Request<vllm::GetServerInfoRequest>,
    ) -> Result<Response<vllm::GetServerInfoResponse>, Status> {
        Ok(Response::new(vllm::GetServerInfoResponse::default()))
    }

    async fn get_loads(
        &self,
        _request: Request<vllm::GetLoadsRequest>,
    ) -> Result<Response<vllm::GetLoadsResponse>, Status> {
        Ok(Response::new(vllm::GetLoadsResponse::default()))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<TokenizerStream>, Status> {
        Err(unscripted("get_tokenizer"))
    }

    async fn subscribe_kv_events(
        &self,
        _request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<KvEventStream>, Status> {
        Err(unscripted("subscribe_kv_events"))
    }

    /// The tonic server over these answers.
    fn into_server(self) -> VllmEngineServer<VllmService<Self>>
    where
        Self: Sized,
    {
        VllmEngineServer::new(VllmService(self))
    }
}

/// The vLLM engine service over a mock's answers.
pub struct VllmService<A>(A);

#[tonic::async_trait]
impl<A: VllmAnswers> VllmEngine for VllmService<A> {
    type GenerateStream = VllmGenerateStream;
    type GetTokenizerStream = TokenizerStream;
    type SubscribeKvEventsStream = KvEventStream;

    async fn generate(
        &self,
        request: Request<vllm::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.0.generate(request).await
    }

    async fn embed(
        &self,
        request: Request<vllm::EmbedRequest>,
    ) -> Result<Response<vllm::EmbedResponse>, Status> {
        self.0.embed(request).await
    }

    async fn health_check(
        &self,
        request: Request<vllm::HealthCheckRequest>,
    ) -> Result<Response<vllm::HealthCheckResponse>, Status> {
        self.0.health_check(request).await
    }

    async fn flush_cache(
        &self,
        request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        self.0.flush_cache(request).await
    }

    async fn start_profile(
        &self,
        request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        self.0.start_profile(request).await
    }

    async fn stop_profile(
        &self,
        request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        self.0.stop_profile(request).await
    }

    async fn abort(
        &self,
        request: Request<vllm::AbortRequest>,
    ) -> Result<Response<vllm::AbortResponse>, Status> {
        self.0.abort(request).await
    }

    async fn get_model_info(
        &self,
        request: Request<vllm::GetModelInfoRequest>,
    ) -> Result<Response<vllm::GetModelInfoResponse>, Status> {
        self.0.get_model_info(request).await
    }

    async fn get_server_info(
        &self,
        request: Request<vllm::GetServerInfoRequest>,
    ) -> Result<Response<vllm::GetServerInfoResponse>, Status> {
        self.0.get_server_info(request).await
    }

    async fn get_loads(
        &self,
        request: Request<vllm::GetLoadsRequest>,
    ) -> Result<Response<vllm::GetLoadsResponse>, Status> {
        self.0.get_loads(request).await
    }

    async fn get_tokenizer(
        &self,
        request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        self.0.get_tokenizer(request).await
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        self.0.subscribe_kv_events(request).await
    }
}
