//! `vllm.grpc.engine.VllmEngine` over the shared state: each RPC delegates to
//! its handler module; RPCs the Rust path does not serve yet answer
//! UNIMPLEMENTED so the Python servicer stays the place for them.

use std::sync::Arc;

use smg_grpc_client::{
    common_proto as common,
    vllm_proto::{self as vllm, vllm_engine_server::VllmEngine},
};
use tonic::{Request, Response, Status};

use super::{admin, embed, generate, info, State};
use crate::{kv_events, tokenizer_bundle, BoxStream};

/// The `VllmEngine` service over the shared state.
#[derive(Clone)]
pub(super) struct VllmEngineService {
    pub(super) state: Arc<State>,
}

#[tonic::async_trait]
impl VllmEngine for VllmEngineService {
    type GenerateStream = BoxStream<vllm::GenerateResponse>;
    type GetTokenizerStream = BoxStream<common::GetTokenizerChunk>;
    type SubscribeKvEventsStream = BoxStream<common::KvEventBatch>;

    async fn generate(
        &self,
        request: Request<vllm::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let trace_headers = generate::trace_headers(request.metadata());
        generate::generate(&self.state, request.into_inner(), trace_headers)
            .await
            .map(Response::new)
    }

    async fn embed(
        &self,
        request: Request<vllm::EmbedRequest>,
    ) -> Result<Response<vllm::EmbedResponse>, Status> {
        embed::embed(&self.state, request.into_inner())
            .await
            .map(Response::new)
    }

    async fn health_check(
        &self,
        _request: Request<vllm::HealthCheckRequest>,
    ) -> Result<Response<vllm::HealthCheckResponse>, Status> {
        Ok(Response::new(info::health_check(&self.state)))
    }

    async fn flush_cache(
        &self,
        request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        admin::flush_cache(&self.state, request)
            .await
            .map(Response::new)
    }

    async fn abort(
        &self,
        request: Request<vllm::AbortRequest>,
    ) -> Result<Response<vllm::AbortResponse>, Status> {
        self.state
            .registry
            .abort(&request.into_inner().request_ids)?;
        Ok(Response::new(vllm::AbortResponse::default()))
    }

    async fn get_model_info(
        &self,
        _request: Request<vllm::GetModelInfoRequest>,
    ) -> Result<Response<vllm::GetModelInfoResponse>, Status> {
        Ok(Response::new(info::model_info(&self.state)))
    }

    async fn get_server_info(
        &self,
        _request: Request<vllm::GetServerInfoRequest>,
    ) -> Result<Response<vllm::GetServerInfoResponse>, Status> {
        Ok(Response::new(info::server_info(&self.state).await))
    }

    async fn get_loads(
        &self,
        _request: Request<vllm::GetLoadsRequest>,
    ) -> Result<Response<vllm::GetLoadsResponse>, Status> {
        info::loads(&self.state).map(Response::new)
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        tokenizer_bundle::get_tokenizer(self.state.tokenizer_dir.clone())
            .await
            .map(Response::new)
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        let Some(relay) = &self.state.kv_relay else {
            return Err(Status::unimplemented(kv_events::VLLM_DISABLED_MESSAGE));
        };
        relay.subscribe(request.into_inner()).map(Response::new)
    }
}
