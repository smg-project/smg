//! `sglang.grpc.scheduler.SglangScheduler` over the shared state: each RPC
//! delegates to its handler module. What the Python servicer does not serve
//! either (LoRA loading) answers UNIMPLEMENTED here; the scheduler's KV-event
//! publisher is relayed as for the other engines.

use std::sync::Arc;

use smg_grpc_client::{
    common_proto as common,
    sglang_proto::{self as sg, sglang_scheduler_server::SglangScheduler},
};
use tonic::{Request, Response, Status};

use super::{admin, embed, generate, info, State};
use crate::{kv_events, tokenizer_bundle, BoxStream};

/// What neither servicer serves: the msgpack wire has no adapter-loading
/// message, and the Python servicer does not implement these RPCs either.
const NO_CONTROL_WIRE: &str = "is not available through the Rust SGLang servicer: the \
                               scheduler's msgpack wire carries no LoRA loading message \
                               (the Python servicer does not serve it either)";

#[derive(Clone)]
pub(super) struct SglangService {
    pub(super) state: Arc<State>,
}

fn no_control_wire(rpc: &str) -> Status {
    Status::unimplemented(format!("{rpc} {NO_CONTROL_WIRE}"))
}

#[tonic::async_trait]
impl SglangScheduler for SglangService {
    type GenerateStream = BoxStream<sg::GenerateResponse>;
    type GetTokenizerStream = BoxStream<common::GetTokenizerChunk>;
    type SubscribeKvEventsStream = BoxStream<common::KvEventBatch>;

    async fn generate(
        &self,
        request: Request<sg::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        generate::generate(&self.state, request.into_inner())
            .await
            .map(Response::new)
    }

    async fn embed(
        &self,
        request: Request<sg::EmbedRequest>,
    ) -> Result<Response<sg::EmbedResponse>, Status> {
        embed::embed(&self.state, request.into_inner())
            .await
            .map(Response::new)
    }

    async fn health_check(
        &self,
        _request: Request<sg::HealthCheckRequest>,
    ) -> Result<Response<sg::HealthCheckResponse>, Status> {
        Ok(Response::new(info::health_check(&self.state)))
    }

    async fn abort(
        &self,
        request: Request<sg::AbortRequest>,
    ) -> Result<Response<sg::AbortResponse>, Status> {
        let request = request.into_inner();
        // The choices of an `n > 1` request live under the one registration,
        // so aborting the parent id ends every choice. An unknown or finished
        // id is reported as the Python servicer reports it.
        let found = self
            .state
            .registry
            .abort(std::slice::from_ref(&request.request_id))?;
        Ok(Response::new(if found > 0 {
            sg::AbortResponse {
                success: true,
                message: format!("aborted {}", request.request_id),
            }
        } else {
            sg::AbortResponse {
                success: false,
                message: format!("Request {} not found", request.request_id),
            }
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<sg::GetModelInfoRequest>,
    ) -> Result<Response<sg::GetModelInfoResponse>, Status> {
        Ok(Response::new(info::model_info(&self.state)))
    }

    async fn get_server_info(
        &self,
        _request: Request<sg::GetServerInfoRequest>,
    ) -> Result<Response<sg::GetServerInfoResponse>, Status> {
        Ok(Response::new(info::server_info(&self.state)))
    }

    async fn get_loads(
        &self,
        request: Request<sg::GetLoadsRequest>,
    ) -> Result<Response<sg::GetLoadsResponse>, Status> {
        info::loads(&self.state, request.into_inner().dp_rank).map(Response::new)
    }

    async fn flush_cache(
        &self,
        request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        admin::flush_cache(&self.state, request)
            .await
            .map(Response::new)
    }

    async fn start_profile(
        &self,
        request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        admin::start_profile(&self.state, request.into_inner())
            .await
            .map(Response::new)
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        admin::stop_profile(&self.state).await.map(Response::new)
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
            return Err(Status::unimplemented(kv_events::SGLANG_DISABLED_MESSAGE));
        };
        relay.subscribe(request.into_inner()).map(Response::new)
    }

    async fn load_lo_ra_adapter(
        &self,
        _request: Request<sg::LoadLoRaAdapterRequest>,
    ) -> Result<Response<sg::LoadLoRaAdapterResponse>, Status> {
        Err(no_control_wire("LoadLoRAAdapter"))
    }

    async fn unload_lo_ra_adapter(
        &self,
        _request: Request<sg::UnloadLoRaAdapterRequest>,
    ) -> Result<Response<sg::UnloadLoRaAdapterResponse>, Status> {
        Err(no_control_wire("UnloadLoRAAdapter"))
    }

    async fn list_loaded_lo_ra_adapters(
        &self,
        _request: Request<sg::ListLoadedLoRaAdaptersRequest>,
    ) -> Result<Response<sg::ListLoadedLoRaAdaptersResponse>, Status> {
        Err(no_control_wire("ListLoadedLoRAAdapters"))
    }
}
