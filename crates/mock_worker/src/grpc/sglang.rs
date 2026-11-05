//! The SGLang scheduler service of the mock gRPC worker: this proto's
//! `Generate` read into the core's input and the core's frames shaped into
//! its chunks and completions (under the request's id, as this wire repeats
//! it); its info (a struct of server args, as the TokenSpeed wire's: the
//! width `dp_size`, a role `disaggregation_mode` with the prefill side's
//! bootstrap port), loads and KV-event subscription. Its embedding and LoRA
//! RPCs are unimplemented, as the mock serves generation alone.

use std::pin::Pin;

use futures::{Stream, StreamExt};
use sg::{
    generate_response::Response as GenResp,
    sglang_scheduler_server::{SglangScheduler, SglangSchedulerServer},
};
use smg_grpc_client::{common_proto as common, sglang_proto as sg};
use tonic::{Request, Response, Status};

use super::{Frame, GenerateInput, LogProbs, MockScheduler, TokenizerStream};
use crate::{engine::KvEventStream, script::MatchedStop};

pub(super) type Server = SglangSchedulerServer<MockScheduler>;
type GenStream = Pin<Box<dyn Stream<Item = Result<sg::GenerateResponse, Status>> + Send>>;

#[tonic::async_trait]
impl SglangScheduler for MockScheduler {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = KvEventStream;
    type GetTokenizerStream = TokenizerStream;

    async fn generate(
        &self,
        request: Request<sg::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
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
        Ok(Response::new(Box::pin(frames.map(move |item| {
            item.map(|frame| response(&request_id, frame))
        }))))
    }

    async fn embed(
        &self,
        _request: Request<sg::EmbedRequest>,
    ) -> Result<Response<sg::EmbedResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn health_check(
        &self,
        _request: Request<sg::HealthCheckRequest>,
    ) -> Result<Response<sg::HealthCheckResponse>, Status> {
        let (healthy, message) = self.health();
        Ok(Response::new(sg::HealthCheckResponse {
            healthy,
            message: message.to_string(),
        }))
    }

    async fn abort(
        &self,
        request: Request<sg::AbortRequest>,
    ) -> Result<Response<sg::AbortResponse>, Status> {
        self.abort_everywhere(&request.into_inner().request_id);
        Ok(Response::new(sg::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<sg::GetModelInfoRequest>,
    ) -> Result<Response<sg::GetModelInfoResponse>, Status> {
        let context_length = self.cfg.context_length.min(i32::MAX as u32) as i32;
        Ok(Response::new(sg::GetModelInfoResponse {
            model_path: self.cfg.model_id.clone(),
            tokenizer_path: self.cfg.tokenizer_path.clone(),
            is_generation: true,
            preferred_sampling_params: String::new(),
            weight_version: "mock".to_string(),
            served_model_name: self.cfg.model_id.clone(),
            max_context_length: context_length,
            vocab_size: 32000,
            supports_vision: self.overrides.info_bool("supports_vision").unwrap_or(false),
            model_type: "mock".to_string(),
            eos_token_ids: vec![2],
            pad_token_id: 0,
            bos_token_id: 1,
            max_req_input_len: context_length,
            architectures: vec!["MockForCausalLM".to_string()],
            id2label_json: String::new(),
            num_labels: 0,
            default_sampling_params_json: self.cfg.default_sampling_params_json.clone(),
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<sg::GetServerInfoRequest>,
    ) -> Result<Response<sg::GetServerInfoResponse>, Status> {
        Ok(Response::new(sg::GetServerInfoResponse {
            server_args: self.server_args(),
            scheduler_info: None,
            active_requests: 0,
            is_paused: false,
            last_receive_timestamp: 0.0,
            uptime_seconds: 0.0,
            sglang_version: "mock".to_string(),
            server_type: "grpc".to_string(),
            start_time: None,
            max_total_num_tokens: 1_000_000,
        }))
    }

    async fn get_loads(
        &self,
        request: Request<sg::GetLoadsRequest>,
    ) -> Result<Response<sg::GetLoadsResponse>, Status> {
        let loads: Vec<sg::SchedulerLoad> = self
            .load_rows(request.into_inner().dp_rank)
            .into_iter()
            .map(|(dp_rank, s)| sg::SchedulerLoad {
                dp_rank,
                num_running_reqs: s.num_running_reqs,
                num_waiting_reqs: s.num_waiting_reqs,
                num_total_reqs: s.num_running_reqs + s.num_waiting_reqs,
                num_used_tokens: s.num_used_tokens,
                max_total_num_tokens: s.max_total_num_tokens,
                token_usage: s.token_usage,
                gen_throughput: s.gen_throughput,
                prefill_throughput: None,
                cache_hit_rate: s.cache_hit_rate,
                utilization: s.token_usage,
                max_running_requests: s.max_running_requests,
                num_waiting_uncached_tokens: s.num_waiting_uncached_tokens,
                memory: None,
                speculative: None,
                lora: None,
                disaggregation: None,
                queues: None,
            })
            .collect();
        Ok(Response::new(sg::GetLoadsResponse {
            timestamp: String::new(),
            version: "mock".to_string(),
            dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
            loads,
            aggregate: None,
        }))
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
        Err(Status::unimplemented("mock-worker"))
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Ok(Response::new(self.kv_events(request.into_inner())?))
    }

    async fn load_lo_ra_adapter(
        &self,
        _request: Request<sg::LoadLoRaAdapterRequest>,
    ) -> Result<Response<sg::LoadLoRaAdapterResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn unload_lo_ra_adapter(
        &self,
        _request: Request<sg::UnloadLoRaAdapterRequest>,
    ) -> Result<Response<sg::UnloadLoRaAdapterResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn list_loaded_lo_ra_adapters(
        &self,
        _request: Request<sg::ListLoadedLoRaAdaptersRequest>,
    ) -> Result<Response<sg::ListLoadedLoRaAdaptersResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }
}

/// A frame in this wire's shape, under the request's id.
fn response(request_id: &str, frame: Frame) -> sg::GenerateResponse {
    let response = match frame {
        Frame::Chunk(chunk) => GenResp::Chunk(sg::GenerateStreamChunk {
            token_ids: chunk.token_ids,
            prompt_tokens: chunk.prompt_tokens,
            completion_tokens: chunk.completion_tokens,
            cached_tokens: chunk.cached_tokens,
            output_logprobs: chunk.logprobs.map(logprobs),
            index: chunk.index,
            ..Default::default()
        }),
        Frame::Complete(complete) => GenResp::Complete(sg::GenerateComplete {
            output_ids: complete.output_ids,
            finish_reason: complete.finish_reason,
            prompt_tokens: complete.prompt_tokens,
            completion_tokens: complete.completion_tokens,
            cached_tokens: complete.cached_tokens,
            output_logprobs: complete.logprobs.map(logprobs),
            matched_stop: complete.matched_stop.map(|stop| match stop {
                MatchedStop::Token(id) => sg::generate_complete::MatchedStop::MatchedTokenId(id),
                MatchedStop::Text(text) => sg::generate_complete::MatchedStop::MatchedStopStr(text),
            }),
            index: complete.index,
            ..Default::default()
        }),
    };
    sg::GenerateResponse {
        request_id: request_id.to_string(),
        media_identity: None,
        response: Some(response),
    }
}

/// The frame's logprobs in this wire's shape, one top entry per position.
fn logprobs(logprobs: LogProbs) -> sg::OutputLogProbs {
    sg::OutputLogProbs {
        token_logprobs: logprobs.token_logprobs,
        token_ids: logprobs.token_ids,
        top_logprobs: logprobs
            .top
            .into_iter()
            .map(|position| sg::TopLogProbs {
                values: position.values,
                token_ids: position.ids,
            })
            .collect(),
        token_ids_logprobs: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use sg::sglang_scheduler_client::SglangSchedulerClient;
    use smg_grpc_client::tokenspeed_proto as ts;
    use tonic::{transport::Channel, Code};
    use tonic_health::pb::{
        health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
    };

    use super::*;
    use crate::{
        config::{Config, Role, SchedulerWire},
        engine,
        script::{self, Answer},
    };

    fn request(id: &str, input_ids: &[u32], stream: bool) -> sg::GenerateRequest {
        sg::GenerateRequest {
            request_id: id.to_string(),
            tokenized: Some(sg::TokenizedInput {
                original_text: String::new(),
                input_ids: input_ids.to_vec(),
            }),
            sampling_params: Some(sg::SamplingParams::default()),
            stream,
            ..Default::default()
        }
    }

    async fn read_frames(
        service: &MockScheduler,
        req: sg::GenerateRequest,
    ) -> Vec<sg::GenerateResponse> {
        let mut stream = service
            .generate(Request::new(req))
            .await
            .expect("accepted")
            .into_inner();
        let mut frames = Vec::new();
        while let Some(item) = stream.next().await {
            frames.push(item.expect("a frame"));
        }
        frames
    }

    /// The server-args entry `key`, as a string.
    fn server_arg(info: &sg::GetServerInfoResponse, key: &str) -> Option<String> {
        let value = info.server_args.as_ref()?.fields.get(key)?;
        match value.kind.as_ref()? {
            prost_types::value::Kind::StringValue(text) => Some(text.clone()),
            prost_types::value::Kind::NumberValue(number) => Some(number.to_string()),
            other => Some(format!("{other:?}")),
        }
    }

    /// A scripted answer streams as this wire's chunks under the request's
    /// id, with the logprobs the request's own fields ask for, and ends with
    /// its completion; a request for two choices takes one answer per choice.
    #[tokio::test]
    async fn a_scripted_answer_takes_this_wires_shape() {
        let (service, _stop) = MockScheduler::for_tests(Config::default(), 65_301);
        script::enqueue(
            &service.name,
            vec![
                Answer::parse(&json!({
                    "output_ids": [7, 8, 9], "logprobs": [-0.1, -0.2, -0.3], "matched_stop": 2
                }))
                .expect("an answer"),
                Answer::parse(&json!({"output_ids": [11]})).expect("an answer"),
                Answer::parse(&json!({"output_ids": [12, 13]})).expect("an answer"),
            ],
        )
        .expect("a registered worker");
        let mut req = request("sw-script", &[1, 2, 3, 4], true);
        req.return_logprob = true;
        req.top_logprobs_num = 2;
        let frames = read_frames(&service, req).await;
        assert_eq!(frames.len(), 4, "three chunks and the completion");
        assert!(frames.iter().all(|frame| frame.request_id == "sw-script"));
        match frames[1].response.as_ref() {
            Some(GenResp::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![8]);
                assert_eq!((chunk.prompt_tokens, chunk.completion_tokens), (4, 2));
                let logprobs = chunk.output_logprobs.as_ref().expect("logprobs");
                assert_eq!(
                    logprobs.top_logprobs[0].token_ids.len(),
                    2,
                    "two alternatives"
                );
            }
            other => panic!("not a chunk: {other:?}"),
        }
        match frames[3].response.as_ref() {
            Some(GenResp::Complete(complete)) => {
                assert_eq!(complete.output_ids, vec![7, 8, 9]);
                assert_eq!((complete.prompt_tokens, complete.completion_tokens), (4, 3));
                assert_eq!(
                    complete.matched_stop,
                    Some(sg::generate_complete::MatchedStop::MatchedTokenId(2))
                );
            }
            other => panic!("not a completion: {other:?}"),
        }
        let mut two = request("sw-choices", &[5], false);
        two.sampling_params = Some(sg::SamplingParams {
            n: 2,
            ..Default::default()
        });
        let completions: Vec<(u32, Vec<u32>)> = read_frames(&service, two)
            .await
            .into_iter()
            .map(|frame| match frame.response {
                Some(GenResp::Complete(complete)) => (complete.index, complete.output_ids),
                other => panic!("a non-streaming request reads completions only: {other:?}"),
            })
            .collect();
        assert_eq!(completions, vec![(0, vec![11]), (1, vec![12, 13])]);
    }

    /// A pinned request runs on its rank's engine, a pin past the ranks is
    /// refused, the loads list the ranks and the server args carry the width.
    #[tokio::test]
    async fn a_pinned_request_runs_on_its_rank_and_the_server_args_carry_the_width() {
        let (service, _stop) = MockScheduler::for_tests(
            Config {
                realistic: true,
                grpc_dp_size: 3,
                engine: engine::EngineParams {
                    timing: engine::TimingModel::linear(),
                    ..Default::default()
                },
                ..Config::default()
            },
            65_302,
        );
        let mut req = request("sw-pin-1", &[1, 2, 3], true);
        req.data_parallel_rank = Some(1);
        req.sampling_params = Some(sg::SamplingParams {
            max_new_tokens: Some(2),
            ..Default::default()
        });
        let frames = read_frames(&service, req).await;
        assert!(matches!(
            frames.last().and_then(|frame| frame.response.as_ref()),
            Some(GenResp::Complete(_))
        ));
        let record = engine::records_since(0, usize::MAX)
            .into_iter()
            .find(|record| record.request_id == "sw-pin-1")
            .expect("the request was admitted");
        assert_eq!(record.worker, "grpc:65302:r1");
        let mut past = request("sw-pin-3", &[1], true);
        past.data_parallel_rank = Some(3);
        let refused = match service.generate(Request::new(past)).await {
            Err(status) => status,
            Ok(_) => panic!("a pin past the ranks was accepted"),
        };
        assert_eq!(refused.code(), Code::InvalidArgument);
        let loads = service
            .get_loads(Request::new(sg::GetLoadsRequest {
                dp_rank: Some(2),
                include: Vec::new(),
            }))
            .await
            .expect("loads")
            .into_inner();
        assert_eq!((loads.dp_rank_count, loads.loads[0].dp_rank), (1, 2));
        let info = service
            .get_server_info(Request::new(sg::GetServerInfoRequest {}))
            .await
            .expect("info")
            .into_inner();
        assert_eq!(server_arg(&info, "dp_size").as_deref(), Some("3"));
        assert_eq!(info.server_type, "grpc");
    }

    /// A prefill worker answers the prompt side's completion alone and
    /// advertises its role and bootstrap port in its server args, as the
    /// TokenSpeed wire does.
    #[tokio::test]
    async fn a_prefill_worker_advertises_its_role_and_port_in_the_server_args() {
        let (prefill, _stop) = MockScheduler::for_tests(
            Config {
                role: Some(Role::Prefill),
                bootstrap_base_port: Some(8998),
                ..Config::default()
            },
            65_303,
        );
        let frames = read_frames(&prefill, request("sw-prefill", &[1, 2, 3], false)).await;
        match frames.as_slice() {
            [frame] => match frame.response.as_ref() {
                Some(GenResp::Complete(complete)) => {
                    assert!(complete.output_ids.is_empty());
                    assert_eq!((complete.prompt_tokens, complete.completion_tokens), (3, 0));
                }
                other => panic!("not a completion: {other:?}"),
            },
            other => panic!("not the prompt side's completion alone: {other:?}"),
        }
        let info = prefill
            .get_server_info(Request::new(sg::GetServerInfoRequest {}))
            .await
            .expect("info")
            .into_inner();
        assert_eq!(
            server_arg(&info, "disaggregation_mode").as_deref(),
            Some("prefill")
        );
        assert_eq!(
            server_arg(&info, "disaggregation_bootstrap_port").as_deref(),
            Some("8998")
        );
        assert_eq!(
            server_arg(&info, "disaggregation_transfer_backend").as_deref(),
            Some(super::super::KV_TRANSPORT),
            "the transport the gateway pairs this wire's workers on"
        );
    }

    /// The served worker answers this wire's health check and names its
    /// service on the standard health service; another wire's check is
    /// unimplemented there, which is how the gateway tells the runtime.
    #[tokio::test]
    async fn the_served_worker_answers_this_wires_health_check_alone() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local addr");
        #[expect(
            clippy::disallowed_methods,
            reason = "the server serves until the test process exits"
        )]
        let _server = tokio::spawn(super::super::serve_with_listener(
            Arc::new(Config {
                grpc_wire: SchedulerWire::Sglang,
                ..Config::default()
            }),
            listener,
        ));
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
        let answer = SglangSchedulerClient::new(channel.clone())
            .health_check(sg::HealthCheckRequest {})
            .await
            .expect("this wire's health check")
            .into_inner();
        assert!(answer.healthy);
        let other =
            ts::token_speed_scheduler_client::TokenSpeedSchedulerClient::new(channel.clone())
                .health_check(ts::HealthCheckRequest {})
                .await
                .expect_err("another wire's health check");
        assert_eq!(other.code(), Code::Unimplemented);
        let mut health = HealthClient::new(channel);
        let served = health
            .check(HealthCheckRequest {
                service: super::super::service_name(SchedulerWire::Sglang).to_string(),
            })
            .await
            .expect("the served service")
            .into_inner();
        assert_eq!(served.status(), ServingStatus::Serving);
        let unknown = health
            .check(HealthCheckRequest {
                service: super::super::service_name(SchedulerWire::Vllm).to_string(),
            })
            .await
            .expect_err("a service the worker does not serve");
        assert_eq!(unknown.code(), Code::NotFound);
    }
}
