//! The SGLang scheduler service of the mock gRPC worker: this proto's
//! `Generate` read into the core's input and the core's frames shaped into
//! its chunks and completions (under the request's id, as this wire repeats
//! it); its info (a struct of server args, as the TokenSpeed wire's: the
//! width `dp_size`, a role `disaggregation_mode` with the prefill side's
//! bootstrap port), loads and KV-event subscription. Its embedding and LoRA
//! RPCs are unimplemented, as the mock serves generation alone.
//! The admin fan-out's flush and profile RPCs are answered by the core, the
//! same on every wire.

use std::{collections::BTreeMap, pin::Pin};

use engine_zmq_client::protocol::sglang::multimodal::SglangWireTensor;
use futures::{Stream, StreamExt};
use sg::{
    generate_response::Response as GenResp,
    sglang_scheduler_server::{SglangScheduler, SglangSchedulerServer},
};
use smg_grpc_client::{common_proto as common, sglang_proto as sg};
use tonic::{Request, Response, Status};

use super::{Frame, GenerateInput, LogProbs, MockScheduler, Rendezvous, TokenizerStream};
use crate::{engine::KvEventStream, script::MatchedStop};

pub(super) type Server = SglangSchedulerServer<MockScheduler>;
type GenStream = Pin<Box<dyn Stream<Item = Result<sg::GenerateResponse, Status>> + Send>>;

/// A request's media batches (`mm_inputs` and `extra_mm_inputs`, one per
/// modality) against the shape the engine's adapter requires before it
/// places their embeddings; the gateway builds them, so a wrong batch is a
/// gateway regression the canned answer would otherwise hide. The reference
/// form (media URLs without a pixel tensor: what the gateway hands a worker
/// that processes media itself, resolved by the worker's processor or an
/// encoder) is the primary batch's alone and travels with no extra batch; it
/// is accepted as it stands, as the mock has neither. Each tensor batch: one
/// modality, image or video (an empty list is an image batch), placed once
/// across the request, a video's placeholder token id, its pixel tensor read
/// as the wire reads it (or a feature hash standing in for a validated
/// identity) and its model-specific tensors read the same way, placeholder
/// spans inside the prompt and disjoint across the request's batches, and,
/// when the batch names its placeholder token id, that the prompt's tokens
/// of that id are exactly the spans'. A batch without spans is left to the
/// engine, which finds the placeholder tokens itself.
fn check_mm_inputs(req: &sg::GenerateRequest) -> Result<(), Status> {
    let Some(primary) = req.mm_inputs.as_deref() else {
        if req.extra_mm_inputs.is_empty() {
            return Ok(());
        }
        return Err(Status::invalid_argument(format!(
            "mock-worker: {} extra media batch(es) without a primary mm_inputs batch",
            req.extra_mm_inputs.len()
        )));
    };
    if names_references(primary) {
        if !req.extra_mm_inputs.is_empty() {
            return Err(Status::invalid_argument(format!(
                "mock-worker: media batch 0 names media references and {} extra tensor \
                 batch(es) ride beside it; the reference form travels alone",
                req.extra_mm_inputs.len()
            )));
        }
        return Ok(());
    }
    let prompt_ids = req
        .tokenized
        .as_ref()
        .map_or(&[][..], |tokenized| tokenized.input_ids.as_slice());
    let mut covered: BTreeMap<u32, usize> = BTreeMap::new();
    let mut claimed = Vec::new();
    let mut placed: Vec<&str> = Vec::new();
    let batches = std::iter::once(primary).chain(req.extra_mm_inputs.iter());
    for (index, mm) in batches.enumerate() {
        let what = format!("media batch {index}");
        let modality = if mm.modalities.iter().all(|name| name == "image") {
            "image"
        } else if mm.modalities.iter().all(|name| name == "video") {
            "video"
        } else {
            return Err(Status::invalid_argument(format!(
                "mock-worker: {what} names modalities {:?}; a tensor batch is image or video, \
                 one modality per batch (an empty list is an image batch)",
                mm.modalities
            )));
        };
        if modality == "video" && mm.im_token_id.is_none() {
            return Err(Status::invalid_argument(format!(
                "mock-worker: {what} is a video batch without its placeholder token id"
            )));
        }
        if names_references(mm) {
            return Err(Status::invalid_argument(format!(
                "mock-worker: {what} carries media references but no pixel_values; the \
                 reference form is the primary batch's alone"
            )));
        }
        match &mm.pixel_values {
            Some(tensor) => check_tensor(&what, "pixel_values", tensor)?,
            None if mm.feature_hash.is_some() => {}
            None => {
                return Err(Status::invalid_argument(format!(
                    "mock-worker: {what} carries no pixel_values and no feature hash"
                )));
            }
        }
        for (key, tensor) in &mm.model_specific_tensors {
            check_tensor(&what, key, tensor)?;
        }
        if !mm.mm_placeholders.is_empty() {
            let spans = mm
                .mm_placeholders
                .iter()
                .map(|placeholder| (placeholder.offset, placeholder.length));
            let length = super::check_placeholder_spans(
                Some(prompt_ids),
                spans,
                mm.im_token_id,
                &mut claimed,
                &what,
            )?;
            if let Some(id) = mm.im_token_id {
                *covered.entry(id).or_default() += length;
            }
        }
        // The adapter's last check of an extra batch, after the batch itself
        // translated: a modality placed once across the request.
        if placed.contains(&modality) {
            return Err(Status::invalid_argument(format!(
                "mock-worker: {what} is a second {modality} batch; one batch per modality"
            )));
        }
        placed.push(modality);
    }
    for (id, length) in covered {
        super::check_placeholder_tokens(prompt_ids, id, length)?;
    }
    Ok(())
}

/// Whether a batch takes the reference form: media URLs and no pixel tensor
/// (the adapter's own test of it; a batch carrying a tensor takes the tensor
/// path whatever references ride beside it).
fn names_references(mm: &sg::MultimodalInputs) -> bool {
    mm.pixel_values.is_none()
        && !(mm.image_urls.is_empty() && mm.video_urls.is_empty() && mm.audio_urls.is_empty())
}

/// A batch's tensor read as this wire's reader reads it (a dtype the wire
/// carries, bytes matching the shape at that width), so the mock refuses the
/// tensors the engine's adapter refuses; `name` is the field in the refusal.
fn check_tensor(what: &str, name: &str, tensor: &sg::TensorData) -> Result<(), Status> {
    SglangWireTensor::new(
        tensor.dtype.clone(),
        tensor.shape.iter().map(|&dim| u64::from(dim)).collect(),
        tensor.data.clone(),
    )
    .map_err(|error| Status::invalid_argument(format!("mock-worker: {what}'s {name}: {error}")))?;
    Ok(())
}

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
        check_mm_inputs(&req)?;
        let tokenized = req.tokenized.unwrap_or_default();
        let params = req.sampling_params.unwrap_or_default();
        let rendezvous = req
            .disaggregated_params
            .as_ref()
            .map_or(Rendezvous::None, |info| Rendezvous::Named {
                host: info.bootstrap_host.clone(),
                port: info.bootstrap_port,
                room: i64::from(info.bootstrap_room),
            });
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
                score_ids: req.token_ids_logprob.clone(),
                pin: req.data_parallel_rank,
                rendezvous,
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
        self.overrides.model_info().refusal("model info")?;
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
        self.overrides.server_info().refusal("server info")?;
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
                prefill_throughput: s.prefill_throughput,
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
        Ok(Response::new(self.flush_cache_response()))
    }

    async fn start_profile(
        &self,
        request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Ok(Response::new(
            self.start_profile_response(request.into_inner()),
        ))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Ok(Response::new(self.stop_profile_response()))
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

/// The frame's logprobs in this wire's shape: one top entry per position, and
/// the scores of the token ids the request asked for as one row, the shape a
/// prefill-only scoring request reads off the completion.
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
        token_ids_logprobs: if logprobs.scores.is_empty() {
            Vec::new()
        } else {
            let (token_ids, values) = logprobs.scores.into_iter().unzip();
            vec![sg::TopLogProbs { values, token_ids }]
        },
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

    /// The request with the rendezvous the gateway mints for a disaggregated
    /// pair on this wire: the prefill side's bootstrap server and a room.
    fn disaggregated(mut req: sg::GenerateRequest, port: i32) -> sg::GenerateRequest {
        req.disaggregated_params = Some(sg::DisaggregatedParams {
            bootstrap_host: "127.0.0.1".to_string(),
            bootstrap_port: port,
            bootstrap_room: 7,
        });
        req
    }

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

    /// A prefill-only scoring request (no new tokens, `token_ids_logprob` set)
    /// reads a scripted answer's token-id logprobs as the one scores row of
    /// its completion, in the request's order; an answer that scores another
    /// count of ids is refused, so is one for a request that asks for no
    /// logprobs, and so is one that scores with no number.
    #[tokio::test]
    async fn a_scoring_answer_carries_the_requested_ids_logprobs() {
        let (service, _stop) = MockScheduler::for_tests(Config::default(), 65_304);
        script::enqueue(
            &service.name,
            vec![
                Answer::parse(&json!({"output_text": "", "token_id_logprobs": [-0.1, -2.3]}))
                    .expect("an answer"),
                Answer::parse(&json!({"output_text": "", "token_id_logprobs": [-0.1, -2.3, -4.0]}))
                    .expect("an answer"),
                Answer::parse(&json!({"output_text": "", "token_id_logprobs": [-0.1, -2.3]}))
                    .expect("an answer"),
            ],
        )
        .expect("a registered worker");
        let score = |id: &str| {
            let mut req = request(id, &[1, 2, 3, 4], false);
            req.sampling_params = Some(sg::SamplingParams {
                max_new_tokens: Some(0),
                ..Default::default()
            });
            req.return_logprob = true;
            req.token_ids_logprob = vec![42, 5];
            req
        };
        let frames = read_frames(&service, score("sw-score")).await;
        let [frame] = frames.as_slice() else {
            panic!("one completion, got {frames:?}");
        };
        match frame.response.as_ref() {
            Some(GenResp::Complete(complete)) => {
                assert!(complete.output_ids.is_empty());
                assert_eq!((complete.prompt_tokens, complete.completion_tokens), (4, 0));
                assert_eq!(complete.finish_reason, "stop");
                let logprobs = complete.output_logprobs.as_ref().expect("logprobs");
                assert_eq!(
                    logprobs.token_ids_logprobs,
                    vec![sg::TopLogProbs {
                        values: vec![-0.1, -2.3],
                        token_ids: vec![42, 5],
                    }]
                );
                assert!(logprobs.token_logprobs.is_empty() && logprobs.top_logprobs.is_empty());
            }
            other => panic!("not a completion: {other:?}"),
        }
        let mut stream = service
            .generate(Request::new(score("sw-score-three")))
            .await
            .expect("accepted")
            .into_inner();
        let status = stream.next().await.expect("an item").expect_err("refused");
        assert_eq!(status.code(), Code::FailedPrecondition);
        assert!(status
            .message()
            .contains("scores 3 token ids, the request asks for 2"));
        let mut silent = score("sw-score-silent");
        silent.return_logprob = false;
        let mut stream = service
            .generate(Request::new(silent))
            .await
            .expect("accepted")
            .into_inner();
        let status = stream.next().await.expect("an item").expect_err("refused");
        assert_eq!(status.code(), Code::FailedPrecondition);
        assert!(status
            .message()
            .contains("scores 2 token ids, the request asks for no logprobs"));
        assert!(Answer::parse(&json!({"output_text": "", "token_id_logprobs": ["x"]})).is_err());
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

    /// A prefill or decode worker on this wire refuses a request without its
    /// `disaggregated_params`, as the HTTP wire refuses a request without its
    /// `bootstrap_room`, and a prefill worker one naming a port other than the
    /// one it bound; with the rendezvous the gateway mints it answers.
    #[tokio::test]
    async fn a_disaggregated_worker_refuses_a_request_without_its_rendezvous() {
        let (prefill, _stop) = MockScheduler::for_tests(
            Config {
                role: Some(Role::Prefill),
                grpc_wire: SchedulerWire::Sglang,
                bootstrap_base_port: Some(8998),
                ..Config::default()
            },
            65_315,
        );
        let refused = prefill
            .generate(Request::new(request("sw-r1", &[1, 2, 3], false)))
            .await
            .err()
            .expect("no rendezvous, not a stream");
        assert_eq!(refused.code(), Code::InvalidArgument, "{refused:?}");
        assert!(
            refused.message().contains("disaggregated_params"),
            "{refused:?}"
        );
        let elsewhere = prefill
            .generate(Request::new(disaggregated(
                request("sw-r2", &[1, 2, 3], false),
                8999,
            )))
            .await
            .err()
            .expect("another port, not a stream");
        assert_eq!(elsewhere.code(), Code::InvalidArgument, "{elsewhere:?}");
        let (decode, _stop) = MockScheduler::for_tests(
            Config {
                role: Some(Role::Decode),
                grpc_wire: SchedulerWire::Sglang,
                ..Config::default()
            },
            65_316,
        );
        let refused = decode
            .generate(Request::new(request("sw-r3", &[1, 2, 3], false)))
            .await
            .err()
            .expect("no rendezvous, not a stream");
        assert_eq!(refused.code(), Code::InvalidArgument, "{refused:?}");
        let frames = read_frames(
            &decode,
            disaggregated(request("sw-r4", &[1, 2, 3], true), 8998),
        )
        .await;
        assert!(
            !frames.is_empty(),
            "the decode side generates with the rendezvous"
        );
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
        let frames = read_frames(
            &prefill,
            disaggregated(request("sw-prefill", &[1, 2, 3], false), 8998),
        )
        .await;
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

    /// The admin fan-out's three RPCs on this wire call the core's answers:
    /// the flush succeeds, a profiling run starts and stops with one trace
    /// file in the requested directory, and a second stop is refused.
    #[tokio::test]
    async fn the_admin_fan_out_rpcs_answer_on_this_wire() {
        let (service, _stop) = MockScheduler::for_tests(Config::default(), 65_308);
        let flushed = SglangScheduler::flush_cache(
            &service,
            Request::new(common::FlushCacheRequest { timeout_s: 0.0 }),
        )
        .await
        .expect("flush")
        .into_inner();
        assert!(flushed.success, "{}", flushed.message);
        assert!(
            flushed.message.ends_with("cache cleared on 0 engine(s)"),
            "{}",
            flushed.message
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let traces = dir.path().join("traces");
        let started = SglangScheduler::start_profile(
            &service,
            Request::new(common::StartProfileRequest {
                output_dir: Some(traces.display().to_string()),
                num_steps: Some(2),
                ..Default::default()
            }),
        )
        .await
        .expect("start")
        .into_inner();
        assert!(started.success, "{}", started.message);
        let stopped =
            SglangScheduler::stop_profile(&service, Request::new(common::StopProfileRequest {}))
                .await
                .expect("stop")
                .into_inner();
        assert!(stopped.success, "{}", stopped.message);
        let files = std::fs::read_dir(&traces)
            .expect("the output directory")
            .count();
        assert_eq!(files, 1, "{}", stopped.message);
        let again =
            SglangScheduler::stop_profile(&service, Request::new(common::StopProfileRequest {}))
                .await
                .expect("stop")
                .into_inner();
        assert!(!again.success, "a stop with no run in progress is refused");
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

    /// An image batch as the gateway builds it: one span over the placeholder
    /// token 9, a pixel tensor, the token id named.
    fn image_batch(offset: u32, length: u32) -> sg::MultimodalInputs {
        sg::MultimodalInputs {
            modalities: vec!["image".to_string()],
            pixel_values: Some(sg::TensorData {
                data: vec![0; 16].into(),
                shape: vec![1, 4],
                dtype: "float32".to_string(),
            }),
            im_token_id: Some(9),
            mm_placeholders: vec![sg::PlaceholderRange { offset, length }],
            ..Default::default()
        }
    }

    fn with_batch(
        mut req: sg::GenerateRequest,
        batch: sg::MultimodalInputs,
    ) -> sg::GenerateRequest {
        req.mm_inputs = Some(Box::new(batch));
        req
    }

    async fn refused_batch(service: &MockScheduler, req: sg::GenerateRequest) -> Status {
        match service.generate(Request::new(req)).await {
            Err(status) => status,
            Ok(_) => panic!("the malformed media batch was accepted"),
        }
    }

    fn media_service(port: u16) -> (MockScheduler, tokio::sync::watch::Sender<bool>) {
        MockScheduler::for_tests(
            Config {
                grpc_wire: SchedulerWire::Sglang,
                ..Config::default()
            },
            port,
        )
    }

    #[tokio::test]
    async fn a_well_formed_media_batch_is_accepted() {
        let (service, _stop) = media_service(65_350);
        let frames = read_frames(
            &service,
            with_batch(request("sg-mm-ok", &[1, 9, 9, 2], true), image_batch(1, 2)),
        )
        .await;
        assert!(!frames.is_empty(), "the answer streams");
    }

    #[tokio::test]
    async fn a_reference_batch_is_accepted_as_the_worker_processed_form() {
        // The gateway's media references for a worker that processes media
        // itself: the URLs alone, no modality, no tensor, no feature hash
        // and no spans.
        let (service, _stop) = media_service(65_358);
        let image = sg::MultimodalInputs {
            image_urls: vec!["https://example.test/a.png".to_string()],
            ..Default::default()
        };
        let video = sg::MultimodalInputs {
            video_urls: vec!["https://example.test/a.mp4".to_string()],
            ..Default::default()
        };
        let audio = sg::MultimodalInputs {
            audio_urls: vec!["https://example.test/a.wav".to_string()],
            ..Default::default()
        };
        for (id, batch) in [
            ("sg-mm-ref-image", image),
            ("sg-mm-ref-video", video),
            ("sg-mm-ref-audio", audio),
        ] {
            let frames = read_frames(&service, with_batch(request(id, &[1, 2], true), batch)).await;
            assert!(!frames.is_empty(), "{id}: the answer streams");
        }
    }

    #[tokio::test]
    async fn overlapping_placeholders_across_batches_are_refused_on_this_wire() {
        let (service, _stop) = media_service(65_359);
        let mut req = with_batch(
            request("sg-mm-overlap", &[1, 9, 9, 2], true),
            image_batch(1, 1),
        );
        req.extra_mm_inputs.push(image_batch(1, 1));
        let status = refused_batch(&service, req).await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("media batch 1")
                && status
                    .message()
                    .contains("overlaps an earlier placeholder at token 1"),
            "{status:?}"
        );
    }

    #[tokio::test]
    async fn a_media_batch_without_pixels_or_a_feature_hash_is_refused() {
        let (service, _stop) = media_service(65_351);
        let mut batch = image_batch(1, 2);
        batch.pixel_values = None;
        let status = refused_batch(
            &service,
            with_batch(request("sg-mm-px", &[1, 9, 9, 2], true), batch),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("media batch 0") && status.message().contains("pixel_values"),
            "{status:?}"
        );
        let mut identity = image_batch(1, 2);
        identity.pixel_values = None;
        identity.feature_hash = Some(42);
        service
            .generate(Request::new(with_batch(
                request("sg-mm-id", &[1, 9, 9, 2], true),
                identity,
            )))
            .await
            .expect("a feature hash stands in for the tensor");
    }

    #[tokio::test]
    async fn a_media_batch_with_an_unknown_modality_is_refused() {
        let (service, _stop) = media_service(65_352);
        let mut batch = image_batch(1, 2);
        batch.modalities = vec!["point_cloud".to_string()];
        let status = refused_batch(
            &service,
            with_batch(request("sg-mm-mod", &[1, 9, 9, 2], true), batch),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(status.message().contains("image or video"), "{status:?}");
    }

    #[tokio::test]
    async fn a_video_batch_without_its_placeholder_token_id_is_refused() {
        let (service, _stop) = media_service(65_353);
        let mut batch = image_batch(1, 2);
        batch.modalities = vec!["video".to_string()];
        batch.im_token_id = None;
        let status = refused_batch(
            &service,
            with_batch(request("sg-mm-vid", &[1, 9, 9, 2], true), batch),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("placeholder token id"),
            "{status:?}"
        );
    }

    #[tokio::test]
    async fn a_placeholder_outside_the_prompt_is_refused_on_this_wire() {
        let (service, _stop) = media_service(65_354);
        let status = refused_batch(
            &service,
            with_batch(request("sg-mm-out", &[1, 9, 9, 2], true), image_batch(3, 2)),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("outside the prompt of 4 tokens"),
            "{status:?}"
        );
    }

    #[tokio::test]
    async fn a_placeholder_that_does_not_cover_its_token_is_refused_on_this_wire() {
        let (service, _stop) = media_service(65_355);
        let status = refused_batch(
            &service,
            with_batch(
                request("sg-mm-cover", &[1, 9, 9, 2], true),
                image_batch(0, 2),
            ),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("not its placeholder token 9"),
            "{status:?}"
        );
        let status = refused_batch(
            &service,
            with_batch(
                request("sg-mm-stray", &[1, 9, 9, 9], true),
                image_batch(1, 2),
            ),
        )
        .await;
        assert!(
            status.message().contains("holds 3 placeholder token(s) 9")
                && status.message().contains("cover 2"),
            "{status:?}"
        );
    }

    /// The engine's adapter takes the extra batches beside a primary one.
    #[tokio::test]
    async fn extra_batches_without_a_primary_batch_are_refused() {
        let (service, _stop) = media_service(65_360);
        let mut req = request("sg-mm-extra", &[1, 9, 9, 2], true);
        req.extra_mm_inputs.push(image_batch(1, 2));
        let status = refused_batch(&service, req).await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(status.message().contains("without a primary"), "{status:?}");
    }

    /// One batch per modality: two image batches over disjoint spans pass
    /// the overlap rule and the engine still takes one of them; an image
    /// batch beside a video batch is the engine's shape.
    #[tokio::test]
    async fn a_second_batch_of_a_placed_modality_is_refused() {
        let (service, _stop) = media_service(65_361);
        let prompt = [1, 9, 9, 2, 9];
        let mut req = with_batch(request("sg-mm-twice", &prompt, true), image_batch(1, 2));
        req.extra_mm_inputs.push(image_batch(4, 1));
        let status = refused_batch(&service, req).await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("media batch 1")
                && status.message().contains("second image batch"),
            "{status:?}"
        );
        let mut video = image_batch(4, 1);
        video.modalities = vec!["video".to_string()];
        let mut req = with_batch(request("sg-mm-pair", &prompt, true), image_batch(1, 2));
        req.extra_mm_inputs.push(video);
        let frames = read_frames(&service, req).await;
        assert!(!frames.is_empty(), "the answer streams");
    }

    /// The adapter reads an empty `modalities` list as an image batch and
    /// takes image and video tensors alone, one modality per batch.
    #[tokio::test]
    async fn an_unnamed_tensor_batch_is_an_image_batch_and_other_modalities_are_refused() {
        let (service, _stop) = media_service(65_362);
        let mut unnamed = image_batch(1, 2);
        unnamed.modalities.clear();
        let frames = read_frames(
            &service,
            with_batch(request("sg-mm-unnamed", &[1, 9, 9, 2], true), unnamed),
        )
        .await;
        assert!(
            !frames.is_empty(),
            "an unnamed tensor batch is an image batch"
        );
        for (id, names) in [
            ("sg-mm-audio", &["audio"][..]),
            ("sg-mm-mixed", &["image", "video"]),
        ] {
            let mut batch = image_batch(1, 2);
            batch.modalities = names.iter().map(|name| (*name).to_string()).collect();
            let status = refused_batch(
                &service,
                with_batch(request(id, &[1, 9, 9, 2], true), batch),
            )
            .await;
            assert_eq!(status.code(), Code::InvalidArgument, "{id}: {status:?}");
            assert!(
                status.message().contains("image or video"),
                "{id}: {status:?}"
            );
        }
    }

    /// A tensor is read as the wire's reader reads it: the bytes match the
    /// shape at the dtype's width, the dtype is one the wire carries, the
    /// model-specific tensors the same.
    #[tokio::test]
    async fn a_tensor_is_read_as_the_wire_reads_it() {
        let (service, _stop) = media_service(65_363);
        let mut short = image_batch(1, 2);
        short.pixel_values.as_mut().expect("the tensor").data = vec![0; 15].into();
        let status = refused_batch(
            &service,
            with_batch(request("sg-mm-bytes", &[1, 9, 9, 2], true), short),
        )
        .await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("media batch 0's pixel_values")
                && status.message().contains("expected 16, got 15"),
            "{status:?}"
        );
        let mut wide = image_batch(1, 2);
        wide.pixel_values.as_mut().expect("the tensor").dtype = "float64".to_string();
        let status = refused_batch(
            &service,
            with_batch(request("sg-mm-dtype", &[1, 9, 9, 2], true), wide),
        )
        .await;
        assert!(
            status.message().contains("unsupported tensor dtype"),
            "{status:?}"
        );
        let mut grid = image_batch(1, 2);
        grid.model_specific_tensors.insert(
            "image_grid_thw".to_string(),
            sg::TensorData {
                data: vec![0; 8].into(),
                shape: vec![1, 3],
                dtype: "int64".to_string(),
            },
        );
        let status = refused_batch(
            &service,
            with_batch(request("sg-mm-grid", &[1, 9, 9, 2], true), grid),
        )
        .await;
        assert!(
            status.message().contains("image_grid_thw")
                && status.message().contains("expected 24, got 8"),
            "{status:?}"
        );
    }

    /// The reference form is the primary batch's alone, whichever of its
    /// three URL kinds names it: the adapter refuses extra batches beside it,
    /// and a reference batch on the tensor path.
    #[tokio::test]
    async fn the_reference_form_travels_alone_as_the_primary_batch() {
        let (service, _stop) = media_service(65_364);
        let reference = sg::MultimodalInputs {
            image_urls: vec!["https://example.test/a.png".to_string()],
            ..Default::default()
        };
        let video = sg::MultimodalInputs {
            video_urls: vec!["https://example.test/a.mp4".to_string()],
            ..Default::default()
        };
        let audio = sg::MultimodalInputs {
            audio_urls: vec!["https://example.test/a.wav".to_string()],
            ..Default::default()
        };
        for (id, primary) in [
            ("sg-mm-ref-extra", reference.clone()),
            ("sg-mm-ref-video-extra", video),
            ("sg-mm-ref-audio-extra", audio),
        ] {
            let mut req = with_batch(request(id, &[1, 9, 9, 2], true), primary);
            req.extra_mm_inputs.push(image_batch(1, 2));
            let status = refused_batch(&service, req).await;
            assert_eq!(status.code(), Code::InvalidArgument, "{id}: {status:?}");
            assert!(
                status.message().contains("media references")
                    && status.message().contains("1 extra tensor batch"),
                "{id}: {status:?}"
            );
        }
        let mut req = with_batch(
            request("sg-mm-extra-ref", &[1, 9, 9, 2], true),
            image_batch(1, 2),
        );
        req.extra_mm_inputs.push(reference);
        let status = refused_batch(&service, req).await;
        assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
        assert!(
            status.message().contains("media batch 1")
                && status.message().contains("references but no pixel_values"),
            "{status:?}"
        );
    }
}
