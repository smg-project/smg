//! The TokenSpeed scheduler service of the mock gRPC worker: this proto's
//! `Generate` read into the core's input and the core's frames shaped into
//! its chunks and completions, with the worker's `weight_version` stamped on
//! each as the engine stamps them; its info, loads and KV-event subscription.
//! `--capture` records this wire's requests (the capture file's keys are its
//! field names).

use std::pin::Pin;

use futures::{Stream, StreamExt};
use smg_grpc_client::{common_proto as common, tokenspeed_scheduler::tokenspeed_proto as ts};
use tonic::{Request, Response, Status};
use ts::{
    generate_response::Response as GenResp,
    token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
};

use super::{Frame, GenerateInput, LogProbs, MockScheduler, TokenizerStream};
use crate::{engine::KvEventStream, script::MatchedStop};

pub(super) type Server = TokenSpeedSchedulerServer<MockScheduler>;
type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;

#[tonic::async_trait]
impl TokenSpeedScheduler for MockScheduler {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = KvEventStream;
    type GetTokenizerStream = TokenizerStream;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        // Record before anything else: the line is in the file before the
        // stream is returned, so before the gateway can read any frame.
        if let Some(capture) = &self.capture {
            if let Err(e) = capture.record(&req) {
                tracing::error!("capturing request {} failed: {e}", req.request_id);
                return Err(Status::internal(format!("mock-worker capture failed: {e}")));
            }
        }
        let tokenized = req.tokenized.unwrap_or_default();
        let params = req.sampling_params.unwrap_or_default();
        let frames = self
            .generate_frames(GenerateInput {
                request_id: req.request_id.clone(),
                prompt_ids: tokenized.input_ids,
                rendered: tokenized.original_text,
                choices: usize::try_from(params.n).unwrap_or(usize::MAX),
                max_new: params.max_new_tokens,
                stream: req.stream,
                logprobs: req
                    .return_logprob
                    .then(|| usize::try_from(req.top_logprobs_num).unwrap_or(0)),
                pin: req.data_parallel_rank,
            })
            .await?;
        let request_id = req.request_id;
        let weight_version = self.cfg.weight_version.clone();
        Ok(Response::new(Box::pin(frames.map(move |item| {
            item.map(|frame| response(&request_id, weight_version.clone(), frame))
        }))))
    }

    async fn health_check(
        &self,
        _request: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        let (healthy, message) = self.health();
        Ok(Response::new(ts::HealthCheckResponse {
            healthy,
            message: message.to_string(),
        }))
    }

    async fn abort(
        &self,
        request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        self.abort_everywhere(&request.into_inner().request_id);
        Ok(Response::new(ts::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Ok(Response::new(ts::GetModelInfoResponse {
            model_path: self.cfg.model_id.clone(),
            tokenizer_path: self.cfg.tokenizer_path.clone(),
            served_model_name: self.cfg.model_id.clone(),
            model_type: "mock".to_string(),
            architectures: vec!["MockForCausalLM".to_string()],
            max_context_length: self.cfg.context_length.min(i32::MAX as u32) as i32,
            max_req_input_len: self.cfg.context_length.min(i32::MAX as u32) as i32,
            vocab_size: 32000,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            bos_token_id: 1,
            weight_version: "mock".to_string(),
            default_sampling_params_json: self.cfg.default_sampling_params_json.clone(),
            supports_vision: self.overrides.info_bool("supports_vision").unwrap_or(false),
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Ok(Response::new(ts::GetServerInfoResponse {
            server_args: self.server_args(),
            scheduler_info: None,
            active_requests: 0,
            is_paused: false,
            uptime_seconds: 0.0,
            max_total_num_tokens: 1_000_000,
            tokenspeed_version: "mock".to_string(),
            start_time: None,
            constrained_decoding: String::new(),
        }))
    }

    async fn get_loads(
        &self,
        request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        let loads: Vec<ts::SchedulerLoad> = self
            .load_rows(request.into_inner().dp_rank)
            .into_iter()
            .map(|(dp_rank, s)| ts::SchedulerLoad {
                active_token_usage: None,
                dp_rank,
                num_running_reqs: s.num_running_reqs,
                num_waiting_reqs: s.num_waiting_reqs,
                num_waiting_uncached_tokens: s.num_waiting_uncached_tokens,
                num_total_reqs: s.num_running_reqs + s.num_waiting_reqs,
                num_used_tokens: s.num_used_tokens,
                max_total_num_tokens: s.max_total_num_tokens,
                max_running_requests: s.max_running_requests,
                token_usage: s.token_usage,
                gen_throughput: s.gen_throughput,
                prefill_throughput: None,
                cache_hit_rate: s.cache_hit_rate,
                utilization: s.token_usage,
                memory: None,
                queues: None,
            })
            .collect();
        Ok(Response::new(ts::GetLoadsResponse {
            timestamp: String::new(),
            version: "mock".to_string(),
            dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
            loads,
            aggregate: None,
        }))
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Ok(Response::new(self.kv_events(request.into_inner())?))
    }

    async fn flush_cache(
        &self,
        _request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn start_profile(
        &self,
        _request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        // The mock has no tokenizer artifacts to serve; the gateway's
        // remote-tokenizer fallback treats Unimplemented as "not supported".
        Err(Status::unimplemented("mock-worker"))
    }
}

/// A frame in this wire's shape, under the request's id, the worker's
/// `weight_version` on it (`None` leaves the field unset, like an engine
/// that predates it).
fn response(
    request_id: &str,
    weight_version: Option<String>,
    frame: Frame,
) -> ts::GenerateResponse {
    let response = match frame {
        Frame::Chunk(chunk) => GenResp::Chunk(ts::GenerateStreamChunk {
            token_ids: chunk.token_ids,
            prompt_tokens: chunk.prompt_tokens,
            completion_tokens: chunk.completion_tokens,
            cached_tokens: chunk.cached_tokens,
            output_logprobs: chunk.logprobs.map(logprobs),
            index: chunk.index,
            weight_version,
        }),
        Frame::Complete(complete) => GenResp::Complete(ts::GenerateComplete {
            output_ids: complete.output_ids,
            finish_reason: complete.finish_reason,
            prompt_tokens: complete.prompt_tokens,
            completion_tokens: complete.completion_tokens,
            cached_tokens: complete.cached_tokens,
            output_logprobs: complete.logprobs.map(logprobs),
            matched_stop: complete.matched_stop.map(|stop| match stop {
                MatchedStop::Token(id) => ts::generate_complete::MatchedStop::MatchedTokenId(id),
                MatchedStop::Text(text) => ts::generate_complete::MatchedStop::MatchedStopStr(text),
            }),
            index: complete.index,
            weight_version,
            ..Default::default()
        }),
    };
    ts::GenerateResponse {
        request_id: request_id.to_string(),
        response: Some(response),
    }
}

/// The frame's logprobs in this wire's shape, one top entry per position.
fn logprobs(logprobs: LogProbs) -> ts::OutputLogProbs {
    ts::OutputLogProbs {
        token_logprobs: logprobs.token_logprobs,
        token_ids: logprobs.token_ids,
        top_logprobs: logprobs
            .top
            .into_iter()
            .map(|position| ts::TopLogProbs {
                values: position.values,
                token_ids: position.ids,
            })
            .collect(),
    }
}
