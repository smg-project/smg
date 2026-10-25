//! SMG vLLM control plane backed by the listener's existing simulated worker.
//! Generation remains explicitly unsupported until its semantic adapter lands.

use smg_grpc_client::vllm_proto as v;
use v::vllm_engine_server::{VllmEngine, VllmEngineServer};

use super::*;

pub(super) type VllmServer = VllmEngineServer<MockVllm>;

#[derive(Clone)]
pub(super) struct MockVllm(MockScheduler);

impl MockVllm {
    pub(super) fn new(mut worker: MockScheduler) -> Self {
        // Only the reporting profile changes. Engine, cancellation, cache,
        // prefill accounting, overrides and shutdown remain shared.
        let mut cfg = (*worker.cfg).clone();
        cfg.loads_like = LoadsLike::Vllm;
        worker.cfg = Arc::new(cfg);
        Self(worker)
    }
}

#[tonic::async_trait]
impl VllmEngine for MockVllm {
    type GenerateStream = Pin<Box<dyn Stream<Item = Result<v::GenerateResponse, Status>> + Send>>;
    type SubscribeKvEventsStream = KvEventStream;
    type GetTokenizerStream = TokenizerStream;

    async fn generate(
        &self,
        _request: Request<v::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented(
            "mock-worker vLLM generation is not implemented",
        ))
    }

    async fn embed(
        &self,
        _request: Request<v::EmbedRequest>,
    ) -> Result<Response<v::EmbedResponse>, Status> {
        Err(Status::unimplemented(
            "mock-worker embeddings are not implemented",
        ))
    }

    async fn health_check(
        &self,
        _request: Request<v::HealthCheckRequest>,
    ) -> Result<Response<v::HealthCheckResponse>, Status> {
        let health = self
            .0
            .health_check(Request::new(ts::HealthCheckRequest {}))
            .await?
            .into_inner();
        Ok(Response::new(v::HealthCheckResponse {
            healthy: health.healthy,
            message: health.message,
        }))
    }

    async fn abort(
        &self,
        request: Request<v::AbortRequest>,
    ) -> Result<Response<v::AbortResponse>, Status> {
        if let Some(engine) = &self.0.engine {
            for id in request.into_inner().request_ids {
                engine.abort(id);
            }
        }
        Ok(Response::new(v::AbortResponse {}))
    }

    async fn get_model_info(
        &self,
        _request: Request<v::GetModelInfoRequest>,
    ) -> Result<Response<v::GetModelInfoResponse>, Status> {
        let info = self
            .0
            .get_model_info(Request::new(ts::GetModelInfoRequest {}))
            .await?
            .into_inner();
        Ok(Response::new(v::GetModelInfoResponse {
            model_path: info.model_path,
            is_generation: true,
            max_context_length: info.max_context_length as u32,
            vocab_size: info.vocab_size as u32,
            supports_vision: info.supports_vision,
            served_model_name: info.served_model_name,
            tokenizer_path: info.tokenizer_path,
            model_type: info.model_type,
            architectures: info.architectures,
            eos_token_ids: info.eos_token_ids,
            pad_token_id: info.pad_token_id,
            bos_token_id: info.bos_token_id,
            max_req_input_len: info.max_req_input_len,
            default_sampling_params_json: info.default_sampling_params_json,
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<v::GetServerInfoRequest>,
    ) -> Result<Response<v::GetServerInfoResponse>, Status> {
        let active_requests = self.0.engine.as_ref().map_or(0, |engine| {
            let load = engine.load();
            load.num_running_reqs
                .saturating_add(load.num_waiting_reqs)
                .max(0) as u32
        });
        Ok(Response::new(v::GetServerInfoResponse {
            active_requests,
            server_type: "vllm-grpc".into(),
            data_parallel_size: 1,
            block_size: if self.0.engine.is_some() {
                self.0.cfg.engine.block_size.min(i32::MAX as u32) as i32
            } else {
                0
            },
            max_num_seqs: if self.0.engine.is_some() {
                self.0.cfg.engine.max_running.min(i32::MAX as usize) as i32
            } else {
                0
            },
            ..Default::default()
        }))
    }

    async fn get_loads(
        &self,
        request: Request<v::GetLoadsRequest>,
    ) -> Result<Response<v::GetLoadsResponse>, Status> {
        let request = request.into_inner();
        if request.dp_rank.is_some_and(|rank| rank != 0) {
            return Err(Status::invalid_argument("mock-worker has only DP rank 0"));
        }
        // Like the real servicer, return available core data and omit optional
        // sections. The gateway always requests core/disagg/queues/memory.
        let report = self
            .0
            .get_loads(Request::new(ts::GetLoadsRequest::default()))
            .await?
            .into_inner();
        let loads = report
            .loads
            .into_iter()
            .map(|load| v::SchedulerLoad {
                dp_rank: load.dp_rank,
                num_running_reqs: load.num_running_reqs,
                num_waiting_reqs: load.num_waiting_reqs,
                num_total_reqs: load.num_total_reqs,
                num_used_tokens: load.num_used_tokens,
                max_total_num_tokens: load.max_total_num_tokens,
                token_usage: load.token_usage,
                gen_throughput: load.gen_throughput,
                cache_hit_rate: load.cache_hit_rate,
                utilization: load.utilization,
                max_running_requests: load.max_running_requests,
                num_waiting_uncached_tokens: load.num_waiting_uncached_tokens,
            })
            .collect();
        Ok(Response::new(v::GetLoadsResponse {
            timestamp: report.timestamp,
            version: report.version,
            dp_rank_count: report.dp_rank_count,
            loads,
        }))
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
