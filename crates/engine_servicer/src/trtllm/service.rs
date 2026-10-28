//! `trtllm.TrtllmService` over the shared state: each RPC delegates to its
//! handler module. What the engine's Python servicer does not serve either
//! (`Embed`, `SubscribeKvEvents`) answers UNIMPLEMENTED here, with the reason.

use std::sync::Arc;

use smg_grpc_client::{
    common_proto as common,
    trtllm_proto::{self as trt, trtllm_service_server::TrtllmService},
};
use tonic::{Request, Response, Status};

use super::{generate, info, State};
use crate::{kv_events, tokenizer_bundle, BoxStream};

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
        generate::generate(&self.state, request.into_inner(), trace_headers)
            .await
            .inspect_err(|status| self.state.shared.metrics.request_not_started(status))
            .map(Response::new)
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

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        tokenizer_bundle::get_tokenizer(self.state.shared.tokenizer_dir.clone())
            .await
            .map(Response::new)
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        let Some(relay) = &self.state.shared.kv_relay else {
            return Err(Status::unimplemented(kv_events::TRTLLM_DISABLED_MESSAGE));
        };
        relay.subscribe(request.into_inner()).map(Response::new)
    }
}
