//! The vLLM engine service of the mock gRPC worker: this proto's `Generate`
//! read into the core's input (its logprobs ride in the sampling params, its
//! prompt in a `oneof` the gateway always fills with token ids) and the
//! core's frames shaped into its chunks and completions; its info (a struct
//! of named fields, not a bag of server args: the width is
//! `data_parallel_size`, a role is `kv_role` behind `kv_connector`), loads
//! and KV-event subscription.

use std::pin::Pin;

use futures::{Stream, StreamExt};
use serde_json::Value;
use smg_grpc_client::{common_proto as common, vllm_proto as vllm};
use tonic::{Request, Response, Status};
use vllm::{
    generate_response::Response as GenResp,
    vllm_engine_server::{VllmEngine, VllmEngineServer},
};

use super::{Frame, GenerateInput, LogProbs, MockScheduler, TokenizerStream};
use crate::{config::Role, engine::KvEventStream, script::MatchedStop};

pub(super) type Server = VllmEngineServer<MockScheduler>;
type GenStream = Pin<Box<dyn Stream<Item = Result<vllm::GenerateResponse, Status>> + Send>>;

/// What this wire's servicer calls itself.
const SERVER_TYPE: &str = "vllm-grpc";
/// The connector a disaggregated mock worker advertises on this wire: the one
/// that carries the mock's transport (`super::KV_TRANSPORT`), which the
/// gateway reads the transport off.
const KV_CONNECTOR: &str = "MooncakeConnector";

#[tonic::async_trait]
impl VllmEngine for MockScheduler {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = KvEventStream;
    type GetTokenizerStream = TokenizerStream;

    async fn generate(
        &self,
        request: Request<vllm::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = request.into_inner();
        let (prompt_ids, rendered) = match req.input {
            Some(vllm::generate_request::Input::Tokenized(tokenized)) => {
                (tokenized.input_ids, tokenized.original_text)
            }
            // The gateway tokenizes every prompt itself; the mock has no
            // tokenizer on its request path.
            Some(vllm::generate_request::Input::Text(_)) => {
                return Err(Status::invalid_argument(
                    "mock-worker: this wire takes token ids, not a text prompt",
                ));
            }
            None => (Vec::new(), String::new()),
        };
        let params = req.sampling_params.unwrap_or_default();
        let frames = self
            .generate_frames(GenerateInput {
                request_id: req.request_id,
                prompt_ids,
                rendered,
                choices: usize::try_from(params.n).unwrap_or(usize::MAX),
                max_new: params.max_tokens,
                stream: req.stream,
                // The alternatives per token; -1 (all) reads as none beyond
                // the sampled token's own.
                logprobs: params.logprobs.map(|top| usize::try_from(top).unwrap_or(0)),
                pin: req.data_parallel_rank,
            })
            .await?;
        Ok(Response::new(Box::pin(
            frames.map(|item| item.map(response)),
        )))
    }

    async fn embed(
        &self,
        _request: Request<vllm::EmbedRequest>,
    ) -> Result<Response<vllm::EmbedResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn health_check(
        &self,
        _request: Request<vllm::HealthCheckRequest>,
    ) -> Result<Response<vllm::HealthCheckResponse>, Status> {
        let (healthy, message) = self.health();
        Ok(Response::new(vllm::HealthCheckResponse {
            healthy,
            message: message.to_string(),
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

    async fn abort(
        &self,
        request: Request<vllm::AbortRequest>,
    ) -> Result<Response<vllm::AbortResponse>, Status> {
        for request_id in &request.into_inner().request_ids {
            self.abort_everywhere(request_id);
        }
        Ok(Response::new(vllm::AbortResponse {}))
    }

    async fn get_model_info(
        &self,
        _request: Request<vllm::GetModelInfoRequest>,
    ) -> Result<Response<vllm::GetModelInfoResponse>, Status> {
        Ok(Response::new(vllm::GetModelInfoResponse {
            model_path: self.cfg.model_id.clone(),
            is_generation: true,
            max_context_length: self.cfg.context_length,
            vocab_size: 32000,
            // On this wire the field is the servicer's own verdict, so the
            // gateway reads a `false` as one: a text-only mock, unless the
            // admin API says otherwise.
            supports_vision: self.overrides.info_bool("supports_vision").unwrap_or(false),
            served_model_name: self.cfg.model_id.clone(),
            tokenizer_path: self.cfg.tokenizer_path.clone(),
            model_type: "mock".to_string(),
            architectures: vec!["MockForCausalLM".to_string()],
            eos_token_ids: vec![2],
            pad_token_id: 0,
            bos_token_id: 1,
            max_req_input_len: self.cfg.context_length.min(i32::MAX as u32) as i32,
            default_sampling_params_json: self.cfg.default_sampling_params_json.clone(),
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<vllm::GetServerInfoRequest>,
    ) -> Result<Response<vllm::GetServerInfoResponse>, Status> {
        let mut info = vllm::GetServerInfoResponse {
            server_type: SERVER_TYPE.to_string(),
            data_parallel_size: i32::try_from(self.dp_size()).unwrap_or(i32::MAX),
            pairing_protocol: self.cfg.pairing_protocol.clone().unwrap_or_default(),
            ..Default::default()
        };
        // A disaggregated worker advertises its role as this wire's servicer
        // does: the connector of the KV handoff and its side of it. The
        // rendezvous port is the operator's to tell the gateway, as for the
        // engine.
        if let Some(role) = self.cfg.role {
            info.kv_connector = KV_CONNECTOR.to_string();
            info.kv_role = match role {
                Role::Prefill => "kv_producer",
                Role::Decode => "kv_consumer",
            }
            .to_string();
        }
        // The configured server args and the admin API's info patch land in
        // the fields they name.
        for (key, value) in &self.cfg.server_args {
            set_field(&mut info, key, &Value::String(value.clone()));
        }
        for (key, value) in &self.overrides.info() {
            set_field(&mut info, key, value);
        }
        Ok(Response::new(info))
    }

    async fn get_loads(
        &self,
        request: Request<vllm::GetLoadsRequest>,
    ) -> Result<Response<vllm::GetLoadsResponse>, Status> {
        let loads: Vec<vllm::SchedulerLoad> = self
            .load_rows(request.into_inner().dp_rank)
            .into_iter()
            .map(|(dp_rank, s)| vllm::SchedulerLoad {
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
            })
            .collect();
        Ok(Response::new(vllm::GetLoadsResponse {
            timestamp: String::new(),
            version: "mock".to_string(),
            dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
            loads,
        }))
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
}

/// A frame in this wire's shape (its responses carry no request id).
fn response(frame: Frame) -> vllm::GenerateResponse {
    let response = match frame {
        Frame::Chunk(chunk) => GenResp::Chunk(vllm::GenerateStreamChunk {
            token_ids: chunk.token_ids,
            prompt_tokens: chunk.prompt_tokens,
            completion_tokens: chunk.completion_tokens,
            cached_tokens: chunk.cached_tokens,
            output_logprobs: chunk.logprobs.map(logprobs),
            input_logprobs: None,
            index: chunk.index,
        }),
        Frame::Complete(complete) => GenResp::Complete(vllm::GenerateComplete {
            output_ids: complete.output_ids,
            finish_reason: complete.finish_reason,
            prompt_tokens: complete.prompt_tokens,
            completion_tokens: complete.completion_tokens,
            cached_tokens: complete.cached_tokens,
            output_logprobs: complete.logprobs.map(logprobs),
            matched_stop: complete.matched_stop.map(|stop| match stop {
                MatchedStop::Token(id) => vllm::generate_complete::MatchedStop::MatchedTokenId(id),
                MatchedStop::Text(text) => {
                    vllm::generate_complete::MatchedStop::MatchedStopStr(text)
                }
            }),
            index: complete.index,
            ..Default::default()
        }),
    };
    vllm::GenerateResponse {
        response: Some(response),
    }
}

/// The frame's logprobs in this wire's shape, one top entry per position.
fn logprobs(logprobs: LogProbs) -> vllm::OutputLogProbs {
    vllm::OutputLogProbs {
        token_logprobs: logprobs.token_logprobs,
        token_ids: logprobs.token_ids,
        top_logprobs: logprobs
            .top
            .into_iter()
            .map(|position| vllm::TopLogProbs {
                values: position.values,
                token_ids: position.ids,
            })
            .collect(),
    }
}

/// Set the field of `info` that `key` names, from a configured server arg
/// or an admin info patch: the keys the gateway reads off this wire's info,
/// with the other wires' spellings of the width and the running window
/// taken too. A key naming no field of this info is dropped with a warning
/// (the wires whose info is a struct of server args carry any key).
fn set_field(info: &mut vllm::GetServerInfoResponse, key: &str, value: &Value) {
    let text = || match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    let number = || {
        value
            .as_i64()
            .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
            .and_then(|number| i32::try_from(number).ok())
    };
    match key {
        "kv_connector" => info.kv_connector = text(),
        "kv_role" => info.kv_role = text(),
        "kv_engine_id" => info.kv_engine_id = text(),
        "shm_namespace_id" => info.shm_namespace_id = text(),
        "kv_cache_dtype" => info.kv_cache_dtype = text(),
        "attention_backend" => info.attention_backend = text(),
        "model_dtype" => info.model_dtype = text(),
        "pairing_protocol" => info.pairing_protocol = text(),
        "mm_processor" => info.mm_processor = text(),
        "mm_media_ref_schemes" => info.mm_media_ref_schemes = text(),
        "mm_processor_source" => info.mm_processor_source = text(),
        "mm_item_limits" => info.mm_item_limits = text(),
        "multimodal_encoder_dtype" => info.multimodal_encoder_dtype = text(),
        "engine_reasoning_parser" => info.engine_reasoning_parser = text(),
        "constrained_decoding_start" => info.constrained_decoding_start = text(),
        "structural_tag_vocabulary" => info.structural_tag_vocabulary = text(),
        "data_parallel_size" | "dp_size" => {
            if let Some(width) = number() {
                info.data_parallel_size = width;
            }
        }
        "block_size" => {
            if let Some(size) = number() {
                info.block_size = size;
            }
        }
        "max_num_seqs" | "max_running_requests" => {
            if let Some(window) = number() {
                info.max_num_seqs = window;
            }
        }
        "mm_encoder_budget" => {
            if let Some(budget) = number() {
                info.mm_encoder_budget = budget;
            }
        }
        "mm_device_do_normalize" => {
            info.mm_device_do_normalize = value
                .as_bool()
                .or_else(|| value.as_str().map(|text| text == "true"))
                .unwrap_or(false);
        }
        // The model info's, read there.
        "supports_vision" => {}
        other => tracing::warn!(
            "server info on the vllm wire has no field {other:?}; the entry is dropped"
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use smg_grpc_client::tokenspeed_proto as ts;
    use tonic::{transport::Channel, Code};
    use tonic_health::pb::{
        health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
    };
    use vllm::vllm_engine_client::VllmEngineClient;

    use super::*;
    use crate::{
        config::{Config, SchedulerWire},
        engine,
        script::{self, Answer},
    };

    fn request(id: &str, input_ids: &[u32], stream: bool) -> vllm::GenerateRequest {
        vllm::GenerateRequest {
            request_id: id.to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: input_ids.to_vec(),
                },
            )),
            sampling_params: Some(vllm::SamplingParams::default()),
            stream,
            ..Default::default()
        }
    }

    async fn read_frames(service: &MockScheduler, req: vllm::GenerateRequest) -> Vec<GenResp> {
        let mut stream = service
            .generate(Request::new(req))
            .await
            .expect("accepted")
            .into_inner();
        let mut frames = Vec::new();
        while let Some(item) = stream.next().await {
            frames.push(item.expect("a frame").response.expect("a response"));
        }
        frames
    }

    /// A scripted answer streams as this wire's chunks (one per token, the
    /// running count on each) and ends with its completion; the logprobs
    /// ride along with the alternatives the sampling params ask for, and
    /// the matched stop text comes back in the proto's `oneof`.
    #[tokio::test]
    async fn a_scripted_answer_takes_this_wires_shape() {
        let (service, _stop) = MockScheduler::for_tests(Config::default(), 65_201);
        script::enqueue(
            &service.name,
            vec![Answer::parse(&json!({
                "output_ids": [7, 8, 9], "logprobs": [-0.1, -0.2, -0.3],
                "finish_reason": "stop", "matched_stop": "</s>", "cached_tokens": 2
            }))
            .expect("an answer")],
        )
        .expect("a registered worker");
        let mut req = request("vw-script", &[1, 2, 3, 4], true);
        req.sampling_params = Some(vllm::SamplingParams {
            logprobs: Some(2),
            ..Default::default()
        });
        let frames = read_frames(&service, req).await;
        assert_eq!(frames.len(), 4, "three chunks and the completion");
        let chunk = match &frames[1] {
            GenResp::Chunk(chunk) => chunk,
            other => panic!("not a chunk: {other:?}"),
        };
        assert_eq!(chunk.token_ids, vec![8]);
        assert_eq!((chunk.prompt_tokens, chunk.completion_tokens), (4, 2));
        let logprobs = chunk.output_logprobs.as_ref().expect("logprobs");
        assert_eq!(logprobs.token_ids, vec![8]);
        assert_eq!(
            logprobs.top_logprobs[0].token_ids.len(),
            2,
            "two alternatives"
        );
        match &frames[3] {
            GenResp::Complete(complete) => {
                assert_eq!(complete.output_ids, vec![7, 8, 9]);
                assert_eq!(complete.finish_reason, "stop");
                assert_eq!(
                    (
                        complete.prompt_tokens,
                        complete.completion_tokens,
                        complete.cached_tokens
                    ),
                    (4, 3, 2)
                );
                assert_eq!(
                    complete.matched_stop,
                    Some(vllm::generate_complete::MatchedStop::MatchedStopStr(
                        "</s>".to_string()
                    ))
                );
                assert_eq!(
                    complete
                        .output_logprobs
                        .as_ref()
                        .expect("logprobs")
                        .token_logprobs
                        .len(),
                    3
                );
            }
            other => panic!("not a completion: {other:?}"),
        }
    }

    /// A request for two choices takes one scripted answer per choice, each
    /// completion under its index, as a non-streaming request reads them.
    #[tokio::test]
    async fn a_request_for_several_choices_completes_each_under_its_index() {
        let (service, _stop) = MockScheduler::for_tests(Config::default(), 65_202);
        script::enqueue(
            &service.name,
            vec![
                Answer::parse(&json!({"output_ids": [11]})).expect("an answer"),
                Answer::parse(&json!({"output_ids": [12, 13]})).expect("an answer"),
            ],
        )
        .expect("a registered worker");
        let mut req = request("vw-choices", &[5], false);
        req.sampling_params = Some(vllm::SamplingParams {
            n: 2,
            ..Default::default()
        });
        let frames = read_frames(&service, req).await;
        let completions: Vec<(u32, Vec<u32>)> = frames
            .iter()
            .map(|frame| match frame {
                GenResp::Complete(complete) => (complete.index, complete.output_ids.clone()),
                other => panic!("a non-streaming request reads completions only: {other:?}"),
            })
            .collect();
        assert_eq!(completions, vec![(0, vec![11]), (1, vec![12, 13])]);
    }

    /// The canned ids come as they do on every wire; a text prompt is the
    /// one input this wire carries that the mock does not serve.
    #[tokio::test]
    async fn the_canned_answer_streams_and_a_text_prompt_is_refused() {
        let (service, _stop) = MockScheduler::for_tests(
            Config {
                output_tokens: 2,
                ..Config::default()
            },
            65_203,
        );
        let frames = read_frames(&service, request("vw-canned", &[1], true)).await;
        assert!(matches!(&frames[0], GenResp::Chunk(chunk) if chunk.token_ids == vec![100]));
        assert!(matches!(&frames[2], GenResp::Complete(c) if c.output_ids == vec![100, 101]));
        let mut text = request("vw-text", &[], false);
        text.input = Some(vllm::generate_request::Input::Text("hello".to_string()));
        let refused = match service.generate(Request::new(text)).await {
            Err(status) => status,
            Ok(_) => panic!("a text prompt was served"),
        };
        assert_eq!(refused.code(), Code::InvalidArgument);
    }

    /// A pinned request runs on its rank's engine, a pin past the ranks is
    /// refused as the servicers refuse it, and the width is this wire's
    /// `data_parallel_size`.
    #[tokio::test]
    async fn a_pinned_request_runs_on_its_rank_and_the_info_carries_the_width() {
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
            65_204,
        );
        let mut req = request("vw-pin-2", &[1, 2, 3], true);
        req.data_parallel_rank = Some(2);
        req.sampling_params = Some(vllm::SamplingParams {
            max_tokens: Some(2),
            ..Default::default()
        });
        let frames = read_frames(&service, req).await;
        assert!(matches!(frames.last(), Some(GenResp::Complete(_))));
        let record = engine::records_since(0, usize::MAX)
            .into_iter()
            .find(|record| record.request_id == "vw-pin-2")
            .expect("the request was admitted");
        assert_eq!(record.worker, "grpc:65204:r2");
        let mut past = request("vw-pin-3", &[1], true);
        past.data_parallel_rank = Some(3);
        let refused = match service.generate(Request::new(past)).await {
            Err(status) => status,
            Ok(_) => panic!("a pin past the ranks was accepted"),
        };
        assert_eq!(refused.code(), Code::InvalidArgument);
        let info = service
            .get_server_info(Request::new(vllm::GetServerInfoRequest {}))
            .await
            .expect("info")
            .into_inner();
        assert_eq!(info.data_parallel_size, 3);
        assert_eq!(info.server_type, SERVER_TYPE);
        let loads = service
            .get_loads(Request::new(vllm::GetLoadsRequest {
                dp_rank: None,
                include: Vec::new(),
            }))
            .await
            .expect("loads")
            .into_inner();
        assert_eq!(loads.dp_rank_count, 3);
        assert_eq!(
            loads
                .loads
                .iter()
                .map(|load| load.dp_rank)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let one = service
            .get_loads(Request::new(vllm::GetLoadsRequest {
                dp_rank: Some(1),
                include: Vec::new(),
            }))
            .await
            .expect("loads")
            .into_inner();
        assert_eq!((one.dp_rank_count, one.loads[0].dp_rank), (1, 1));
    }

    /// A prefill worker answers the prompt side's completion alone and
    /// advertises its role as this wire's servicer does: the KV connector and
    /// the producer side of the handoff; a decode worker the consumer side; a
    /// regular worker neither.
    #[tokio::test]
    async fn the_roles_are_advertised_as_kv_role_behind_the_connector() {
        let (prefill, _stop) = MockScheduler::for_tests(
            Config {
                role: Some(Role::Prefill),
                ..Config::default()
            },
            65_205,
        );
        let frames = read_frames(&prefill, request("vw-prefill", &[1, 2, 3, 4, 5], true)).await;
        match frames.as_slice() {
            [GenResp::Complete(complete)] => {
                assert!(complete.output_ids.is_empty());
                assert_eq!((complete.prompt_tokens, complete.completion_tokens), (5, 0));
            }
            other => panic!("not the prompt side's completion alone: {other:?}"),
        }
        let info = prefill
            .get_server_info(Request::new(vllm::GetServerInfoRequest {}))
            .await
            .expect("info")
            .into_inner();
        assert_eq!(
            (info.kv_connector.as_str(), info.kv_role.as_str()),
            (KV_CONNECTOR, "kv_producer")
        );
        let (decode, _stop) = MockScheduler::for_tests(
            Config {
                role: Some(Role::Decode),
                ..Config::default()
            },
            65_206,
        );
        let info = decode
            .get_server_info(Request::new(vllm::GetServerInfoRequest {}))
            .await
            .expect("info")
            .into_inner();
        assert_eq!(info.kv_role, "kv_consumer");
        let (regular, _stop) = MockScheduler::for_tests(Config::default(), 65_207);
        let info = regular
            .get_server_info(Request::new(vllm::GetServerInfoRequest {}))
            .await
            .expect("info")
            .into_inner();
        assert!(info.kv_connector.is_empty() && info.kv_role.is_empty());
    }

    /// A configured server arg or an admin info patch lands in the field it
    /// names, the other wires' spellings included; an unknown key is dropped.
    #[test]
    fn server_args_and_info_patches_set_the_fields_they_name() {
        let mut info = vllm::GetServerInfoResponse::default();
        set_field(&mut info, "max_running_requests", &json!(8));
        set_field(&mut info, "dp_size", &json!("4"));
        set_field(&mut info, "kv_engine_id", &json!("engine-a"));
        set_field(&mut info, "mm_device_do_normalize", &json!(true));
        set_field(&mut info, "rl.control_url", &json!("http://127.0.0.1:1"));
        assert_eq!(
            (
                info.max_num_seqs,
                info.data_parallel_size,
                info.kv_engine_id.as_str()
            ),
            (8, 4, "engine-a")
        );
        assert!(info.mm_device_do_normalize);
    }

    /// The served worker answers this wire's health check and names its
    /// service on the standard health service; the other wires' checks are
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
                grpc_wire: SchedulerWire::Vllm,
                ..Config::default()
            }),
            listener,
        ));
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
        let answer = VllmEngineClient::new(channel.clone())
            .health_check(vllm::HealthCheckRequest {})
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
        let status = |service: &str| HealthCheckRequest {
            service: service.to_string(),
        };
        let served = health
            .check(status(super::super::service_name(SchedulerWire::Vllm)))
            .await
            .expect("the served service")
            .into_inner();
        assert_eq!(served.status(), ServingStatus::Serving);
        let unknown = health
            .check(status(super::super::service_name(
                SchedulerWire::TokenSpeed,
            )))
            .await
            .expect_err("a service the worker does not serve");
        assert_eq!(unknown.code(), Code::NotFound);
    }
}
