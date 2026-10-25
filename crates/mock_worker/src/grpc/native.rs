//! Native vLLM control discovery and cancellation for the same mock engine.
//! Inference and native KV discovery are deliberately unavailable at this stage.

use vllm_proto::{
    self as v,
    control_server::{Control, ControlServer},
};

use super::*;

pub(super) type NativeServer = ControlServer<MockNative>;

#[derive(Clone)]
pub(super) struct MockNative(pub(super) MockScheduler);

#[tonic::async_trait]
impl Control for MockNative {
    async fn get_server_info(
        &self,
        _request: Request<v::GetServerInfoRequest>,
    ) -> Result<Response<v::ServerInfo>, Status> {
        let cfg = &self.0.cfg;
        let engine = self.0.engine.as_ref().map(|_| &cfg.engine);
        Ok(Response::new(v::ServerInfo {
            engine_version: concat!("mock-worker/", env!("CARGO_PKG_VERSION")).into(),
            api_version: "mock-control".into(),
            instance_id: format!("{}:{}", std::process::id(), self.0.name),
            parallelism: Some(v::ParallelismInfo {
                tensor_parallel_size: 1,
                pipeline_parallel_size: 1,
                data_parallel_size: 1,
                data_parallel_rank: 0,
                decode_context_parallel_size: 1,
                world_size: 1,
                data_parallel_size_local: 1,
            }),
            max_model_len: cfg.context_length.min(i32::MAX as u32),
            kv_block_size: engine.map_or(0, |p| p.block_size.max(1)),
            total_kv_blocks: engine.map_or(0, |p| p.capacity_blocks()),
            max_running_requests: engine.map_or(0, |p| p.max_running as u64),
            max_batched_tokens: engine.map_or(0, |p| u64::from(p.max_batched_tokens)),
            proto_version: v::VERSION.into(),
            ..Default::default()
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<v::GetModelInfoRequest>,
    ) -> Result<Response<v::ModelInfo>, Status> {
        Ok(Response::new(v::ModelInfo {
            model_id: self.0.cfg.model_id.clone(),
            served_model_name: self.0.cfg.model_id.clone(),
            // No native Inference service yet: all input/capability flags stay false.
            ..Default::default()
        }))
    }

    async fn abort(
        &self,
        request: Request<v::AbortRequest>,
    ) -> Result<Response<v::AbortResponse>, Status> {
        let engine =
            self.0.engine.as_ref().ok_or_else(|| {
                Status::failed_precondition("native abort requires --realistic true")
            })?;
        let ids = request.into_inner().request_ids;
        if ids.len() > 1024 || ids.iter().any(|id| id.is_empty() || id.len() > 256) {
            return Err(Status::invalid_argument(
                "abort accepts at most 1024 request IDs, each 1..=256 bytes",
            ));
        }
        // Validate the entire batch before changing the shared engine. Unknown
        // and repeated IDs are harmless, matching the native cancellation API.
        for id in ids {
            engine.abort(id);
        }
        Ok(Response::new(v::AbortResponse {}))
    }

    async fn shutdown(
        &self,
        _request: Request<v::ShutdownRequest>,
    ) -> Result<Response<v::ShutdownResponse>, Status> {
        Err(Status::failed_precondition(
            "this mock does not manage an engine process",
        ))
    }

    async fn load_lora(
        &self,
        _request: Request<v::LoadLoraRequest>,
    ) -> Result<Response<v::LoadLoraResponse>, Status> {
        Err(Status::unimplemented("native LoadLora is not implemented"))
    }

    async fn unload_lora(
        &self,
        _request: Request<v::UnloadLoraRequest>,
    ) -> Result<Response<v::UnloadLoraResponse>, Status> {
        Err(Status::unimplemented(
            "native UnloadLora is not implemented",
        ))
    }

    async fn list_loras(
        &self,
        _request: Request<v::ListLorasRequest>,
    ) -> Result<Response<v::ListLorasResponse>, Status> {
        Err(Status::unimplemented("native ListLoras is not implemented"))
    }

    async fn get_kv_event_sources(
        &self,
        _request: Request<v::GetKvEventSourcesRequest>,
    ) -> Result<Response<v::GetKvEventSourcesResponse>, Status> {
        Err(Status::unimplemented(
            "native GetKvEventSources is not implemented",
        ))
    }

    async fn pause_generation(
        &self,
        _request: Request<v::PauseGenerationRequest>,
    ) -> Result<Response<v::PauseGenerationResponse>, Status> {
        Err(Status::unimplemented(
            "native PauseGeneration is not implemented",
        ))
    }

    async fn resume_generation(
        &self,
        _request: Request<v::ResumeGenerationRequest>,
    ) -> Result<Response<v::ResumeGenerationResponse>, Status> {
        Err(Status::unimplemented(
            "native ResumeGeneration is not implemented",
        ))
    }

    async fn is_paused(
        &self,
        _request: Request<v::IsPausedRequest>,
    ) -> Result<Response<v::IsPausedResponse>, Status> {
        Err(Status::unimplemented("native IsPaused is not implemented"))
    }

    async fn sleep(
        &self,
        _request: Request<v::SleepRequest>,
    ) -> Result<Response<v::SleepResponse>, Status> {
        Err(Status::unimplemented("native Sleep is not implemented"))
    }

    async fn wake_up(
        &self,
        _request: Request<v::WakeUpRequest>,
    ) -> Result<Response<v::WakeUpResponse>, Status> {
        Err(Status::unimplemented("native WakeUp is not implemented"))
    }

    async fn is_sleeping(
        &self,
        _request: Request<v::IsSleepingRequest>,
    ) -> Result<Response<v::IsSleepingResponse>, Status> {
        Err(Status::unimplemented(
            "native IsSleeping is not implemented",
        ))
    }

    async fn init_weight_transfer_engine(
        &self,
        _request: Request<v::InitWeightTransferEngineRequest>,
    ) -> Result<Response<v::InitWeightTransferEngineResponse>, Status> {
        Err(Status::unimplemented(
            "native InitWeightTransferEngine is not implemented",
        ))
    }

    async fn start_weight_update(
        &self,
        _request: Request<v::StartWeightUpdateRequest>,
    ) -> Result<Response<v::StartWeightUpdateResponse>, Status> {
        Err(Status::unimplemented(
            "native StartWeightUpdate is not implemented",
        ))
    }

    async fn start_draft_weight_update(
        &self,
        _request: Request<v::StartDraftWeightUpdateRequest>,
    ) -> Result<Response<v::StartDraftWeightUpdateResponse>, Status> {
        Err(Status::unimplemented(
            "native StartDraftWeightUpdate is not implemented",
        ))
    }

    async fn update_weights(
        &self,
        _request: Request<v::UpdateWeightsRequest>,
    ) -> Result<Response<v::UpdateWeightsResponse>, Status> {
        Err(Status::unimplemented(
            "native UpdateWeights is not implemented",
        ))
    }

    async fn finish_weight_update(
        &self,
        _request: Request<v::FinishWeightUpdateRequest>,
    ) -> Result<Response<v::FinishWeightUpdateResponse>, Status> {
        Err(Status::unimplemented(
            "native FinishWeightUpdate is not implemented",
        ))
    }

    async fn update_weight_version(
        &self,
        _request: Request<v::UpdateWeightVersionRequest>,
    ) -> Result<Response<v::UpdateWeightVersionResponse>, Status> {
        Err(Status::unimplemented(
            "native UpdateWeightVersion is not implemented",
        ))
    }

    async fn get_weight_version(
        &self,
        _request: Request<v::GetWeightVersionRequest>,
    ) -> Result<Response<v::GetWeightVersionResponse>, Status> {
        Err(Status::unimplemented(
            "native GetWeightVersion is not implemented",
        ))
    }
}
