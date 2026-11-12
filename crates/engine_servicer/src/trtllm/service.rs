//! `trtllm.TrtllmService` over the shared state: each RPC delegates to its
//! handler module. `Embed`, which the engine's Python servicer does not serve
//! either, answers UNIMPLEMENTED here with the reason. `SubscribeKvEvents`
//! streams the servicer's own load records for a `loads_only` request; any
//! other is served by the KV-event relay when the launcher configured the
//! engine's publisher endpoint, and answers UNIMPLEMENTED with the reason
//! otherwise (the TensorRT-LLM launcher configures none: the wire has no
//! publisher yet).

use std::sync::Arc;

use smg_grpc_client::{
    common_proto as common,
    trtllm_proto::{self as trt, trtllm_service_server::TrtllmService},
};
use tonic::{Request, Response, Status};

use super::{generate, info, State};
use crate::{kv_events, rpc, BoxStream};

/// What neither servicer serves: the engine's Python servicer does not
/// implement `Embed`, and the wire carries no embedding request.
const NO_EMBED: &str = "Embed is not implemented by the Rust TensorRT-LLM servicer: the engine's \
                        wire carries generate and abort only (the Python servicer does not serve \
                        it either)";

#[derive(Clone)]
pub(super) struct TrtllmServicer {
    pub(super) state: Arc<State>,
}

#[tonic::async_trait]
impl TrtllmService for TrtllmServicer {
    type GenerateStream = BoxStream<trt::GenerateResponse>;
    type GetTokenizerStream = BoxStream<common::GetTokenizerChunk>;
    type SubscribeKvEventsStream = BoxStream<common::KvEventBatch>;

    async fn generate(
        &self,
        request: Request<trt::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let trace_headers = crate::trace_context::trace_headers(request.metadata());
        let stream = generate::generate(&self.state, request.into_inner(), trace_headers).await;
        rpc::generated(&self.state, stream)
    }

    async fn embed(
        &self,
        _request: Request<trt::EmbedRequest>,
    ) -> Result<Response<trt::EmbedResponse>, Status> {
        Err(Status::unimplemented(NO_EMBED))
    }

    async fn health_check(
        &self,
        _request: Request<trt::HealthCheckRequest>,
    ) -> Result<Response<trt::HealthCheckResponse>, Status> {
        Ok(Response::new(info::health_check(&self.state)))
    }

    async fn abort(
        &self,
        request: Request<trt::AbortRequest>,
    ) -> Result<Response<trt::AbortResponse>, Status> {
        let request = request.into_inner();
        // The choices of an `n > 1` request live under the one registration,
        // so aborting the parent id ends every choice. An unknown or finished
        // id is reported as the Python servicer reports it.
        let found = self
            .state
            .shared
            .registry
            .abort(std::slice::from_ref(&request.request_id))?;
        Ok(Response::new(trt::AbortResponse {
            success: found > 0,
            message: format!(
                "Request {} {}",
                request.request_id,
                if found > 0 { "aborted" } else { "not found" }
            ),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<trt::GetModelInfoRequest>,
    ) -> Result<Response<trt::GetModelInfoResponse>, Status> {
        Ok(Response::new(info::model_info(&self.state)))
    }

    async fn get_server_info(
        &self,
        _request: Request<trt::GetServerInfoRequest>,
    ) -> Result<Response<trt::GetServerInfoResponse>, Status> {
        Ok(Response::new(info::server_info(&self.state)))
    }

    async fn get_loads(
        &self,
        request: Request<trt::GetLoadsRequest>,
    ) -> Result<Response<trt::GetLoadsResponse>, Status> {
        Ok(Response::new(info::loads(
            &self.state,
            request.into_inner().dp_rank,
        )?))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        rpc::get_tokenizer(&self.state).await
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        rpc::subscribe_kv_events(&self.state, request, kv_events::TRTLLM_DISABLED_MESSAGE)
    }
}
