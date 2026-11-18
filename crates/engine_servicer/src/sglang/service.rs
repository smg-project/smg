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

use super::{embed, generate, info, State};
use crate::{admin, kv_events, rpc, servicer::RequestOrigin, trace_context, BoxStream};

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

// Ordinary text uses the existing Generate/Abort lifecycle even in an EPD pool.
// Media and the private control envelope still require the fenced control RPC.
/// Separate native text generation from envelopes that require prepared-input ownership.
fn requires_managed_control(request: &sg::GenerateRequest) -> bool {
    !request.prepared_input.is_empty()
        || request.mm_inputs.is_some()
        || !request.extra_mm_inputs.is_empty()
        || !request.input_embeds.is_empty()
        || request.media_identity.is_some()
        || request.sampling_params.as_ref().is_some_and(|params| {
            params
                .custom_params
                .as_ref()
                .is_some_and(|custom| custom.fields.contains_key("__smg_managed_epd_v1"))
        })
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
        if !request.get_ref().prepared_control.is_empty() {
            return super::prepared::generate(&self.state, request.into_inner()).map(Response::new);
        }
        if self.state.managed && requires_managed_control(request.get_ref()) {
            return Err(Status::failed_precondition(
                "prepared-input metadata is required",
            ));
        }
        let trace_headers = trace_context::trace_headers(request.metadata());
        let stream = generate::generate(
            &self.state,
            request.into_inner(),
            trace_headers,
            RequestOrigin::Client,
        )
        .await;
        rpc::generated(&self.state, stream)
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
        Ok(Response::new(info::health_check(&self.state).await))
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
            .shared
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
        admin::flush_cache(&self.state.shared, request)
            .await
            .map(Response::new)
    }

    async fn start_profile(
        &self,
        request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        admin::start_profile(&self.state.shared, request.into_inner())
            .await
            .map(Response::new)
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        admin::stop_profile(&self.state.shared)
            .await
            .map(Response::new)
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
        rpc::subscribe_kv_events(&self.state, request, kv_events::SGLANG_DISABLED_MESSAGE)
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
        self.state.shared.engine()?;
        Ok(Response::new(sg::ListLoadedLoRaAdaptersResponse {
            loaded_adapters: self.state.loras.adapters.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Only ordinary text can bypass the prepared-control stream on a prepared worker.
    fn managed_pool_accepts_ordinary_text_but_not_media_or_control_envelopes() {
        let mut request = sg::GenerateRequest::default();
        assert!(!requires_managed_control(&request));
        request.mm_inputs = Some(Default::default());
        assert!(requires_managed_control(&request));
        request.mm_inputs = None;
        request.extra_mm_inputs.push(Default::default());
        assert!(requires_managed_control(&request));
        request.extra_mm_inputs.clear();
        request.input_embeds.push(1.0);
        assert!(requires_managed_control(&request));
        request.input_embeds.clear();
        request.media_identity = Some(Default::default());
        assert!(requires_managed_control(&request));
        request.media_identity = None;
        request.sampling_params = Some(sg::SamplingParams {
            custom_params: Some(prost_types::Struct {
                fields: [("__smg_managed_epd_v1".to_owned(), Default::default())].into(),
            }),
            ..Default::default()
        });
        assert!(requires_managed_control(&request));
    }
}
