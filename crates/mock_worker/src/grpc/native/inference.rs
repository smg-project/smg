//! Restricted token-ID generation on the shared scheduler; no text decoder.
use vllm_proto::inference_server::{Inference, InferenceServer};

use super::{v, MockNative};
use crate::grpc::*;

pub(in crate::grpc) type NativeInferenceServer = InferenceServer<MockNative>;
type NativeStream = Pin<Box<dyn Stream<Item = Result<v::GenerateResponse, Status>> + Send>>;

#[tonic::async_trait]
impl Inference for MockNative {
    type GenerateStreamStream = NativeStream;

    async fn generate(
        &self,
        request: Request<v::GenerateRequest>,
    ) -> Result<Response<v::GenerateResponse>, Status> {
        let mut output = self.generate_native(request, false).await?;
        let response = output
            .next()
            .await
            .ok_or_else(|| Status::internal("missing native completion"))??;
        Ok(Response::new(response))
    }

    async fn generate_stream(
        &self,
        request: Request<v::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        Ok(Response::new(self.generate_native(request, true).await?))
    }
}

impl MockNative {
    async fn generate_native(
        &self,
        request: Request<v::GenerateRequest>,
        streaming: bool,
    ) -> Result<NativeStream, Status> {
        if self.0.engine.is_none() || self.0.cfg.replay.script.is_some() {
            return Err(Status::failed_precondition(
                "native generation requires --engine realistic without --script",
            ));
        }
        let ranks = request.metadata().get_all("x-data-parallel-rank");
        let mut ranks = ranks.iter();
        if let Some(rank) = ranks.next() {
            if rank
                .to_str()
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                != Some(0)
                || ranks.next().is_some()
            {
                return Err(Status::invalid_argument(
                    "native mock accepts one data-parallel-rank header, with rank 0",
                ));
            }
        }
        // Read the same model metadata as TokenSpeed/SMG; future configured vocabularies
        // must not acquire a second, native-only default here.
        let model = self
            .0
            .get_model_info(Request::new(ts::GetModelInfoRequest {}))
            .await?
            .into_inner();
        let (request, mapper) = normalize(request.into_inner(), streaming, &model)?;
        let source = self.0.generate(Request::new(request)).await?.into_inner();
        Ok(adapt(source, mapper))
    }
}

fn adapt(source: GenStream, mapper: Mapper) -> NativeStream {
    Box::pin(stream::unfold(Some((source, mapper)), |state| async move {
        match state {
            Some((mut source, mut mapper)) => {
                let item = match source.next().await {
                    Some(frame) => frame.and_then(|frame| mapper.response(frame)),
                    None => Err(Status::internal(
                        "mock engine ended without a terminal response",
                    )),
                };
                let next = if item.is_err() || mapper.finished {
                    None
                } else {
                    Some((source, mapper))
                };
                Some((item, next))
            }
            None => None,
        }
    }))
}

struct Mapper {
    prompt: Vec<u32>,
    include_prompt: bool,
    streaming: bool,
    max_tokens: u32,
    vocab: u32,
    sent: Vec<u32>,
    started: bool,
    finished: bool,
}

fn normalize(
    request: v::GenerateRequest,
    streaming: bool,
    model_info: &ts::GetModelInfoResponse,
) -> Result<(ts::GenerateRequest, Mapper), Status> {
    let v::GenerateRequest {
        request_id,
        model,
        prompt,
        temperature,
        sampling,
        decoding,
        stopping,
        response,
        kv,
        truncate_prompt_tokens,
        priority,
        session_id,
        media,
        lora_name,
        watermarking,
        kv_hints,
        engine_reasoning_gate,
    } = request;
    if request_id.is_empty() || request_id.len() > 256 {
        return Err(Status::invalid_argument(
            "request_id must contain 1..=256 bytes",
        ));
    }
    if !model.is_empty() && model != model_info.served_model_name {
        return Err(Status::invalid_argument(
            "requested model is not served by this mock",
        ));
    }
    let Some(v::generate_request::Prompt::TokenIds(prompt)) = prompt else {
        return Err(Status::invalid_argument(
            "native mock requires token-ID input",
        ));
    };
    let vocab = model_info.vocab_size as u32;
    if prompt.ids.is_empty() || prompt.ids.iter().any(|id| *id >= vocab) {
        return Err(Status::invalid_argument(
            "prompt tokens must be inside the advertised vocabulary",
        ));
    }
    if temperature.is_some_and(|value| value != 0.0)
        || sampling.as_ref().is_some_and(|p| {
            p.num_sequences > 1
                || p.top_k.is_some_and(|x| x != 0)
                || p.top_p.is_some_and(|x| x != 1.0)
                || p.min_p.is_some_and(|x| x != 0.0)
                || p.seed.is_some()
        })
        || decoding.as_ref().is_some_and(|p| {
            p.presence_penalty != 0.0
                || p.frequency_penalty != 0.0
                || (p.repetition_penalty != 0.0 && p.repetition_penalty != 1.0)
                || !p.logit_bias.is_empty()
                || !p.allowed_token_ids.is_empty()
                || p.structured_output.is_some()
                || !p.bad_words_token_ids.is_empty()
        })
        || kv.as_ref().is_some_and(|p| {
            p.bypass_prefix_cache
                || !p.cache_salt.is_empty()
                || p.kv_transfer_params.is_some()
                || p.ec_transfer_params.is_some()
        })
        || truncate_prompt_tokens != 0
        || priority != 0
        || session_id.is_some()
        || !media.is_empty()
        || !lora_name.is_empty()
        || watermarking == Some(true)
        || kv_hints.is_some()
        || engine_reasoning_gate.is_some()
    {
        return Err(Status::unimplemented("native mock supports one greedy sequence with neutral sampling and no constraints, sessions, cache controls, watermarking, LoRA or media"));
    }
    let stopping = stopping
        .ok_or_else(|| Status::invalid_argument("explicit stopping.max_new_tokens is required"))?;
    if stopping.max_new_tokens == 0
        || prompt.ids.len() as u64 + u64::from(stopping.max_new_tokens)
            > model_info.max_context_length as u64
    {
        return Err(Status::invalid_argument("positive max_new_tokens and prompt plus output within the advertised context are required"));
    }
    if stopping.min_new_tokens != 0
        || !stopping.stop_token_ids.is_empty()
        || !stopping.stop_strings.is_empty()
        || stopping.include_stop_strings
        || stopping
            .thinking_token_budget
            .is_some_and(|value| value != -1)
    {
        return Err(Status::unimplemented(
            "native mock does not implement stop, minimum-length or reasoning-budget controls",
        ));
    }
    let response = response.unwrap_or_default();
    if response.output_text != Some(false)
        || !response.output_token_ids
        || response.prompt_logprobs
        || response.output_logprobs
        || response.prompt_candidates.is_some()
        || response.output_candidates.is_some()
    {
        return Err(Status::unimplemented("native token-ID profile requires output_text=false, output_token_ids=true and no logprobs or candidates"));
    }
    let mapper = Mapper {
        prompt: prompt.ids.clone(),
        include_prompt: response.prompt_token_ids,
        streaming,
        max_tokens: stopping.max_new_tokens,
        vocab,
        sent: Vec::new(),
        started: false,
        finished: false,
    };
    Ok((
        ts::GenerateRequest {
            request_id,
            stream: streaming,
            tokenized: Some(ts::TokenizedInput {
                input_ids: prompt.ids,
                ..Default::default()
            }),
            sampling_params: Some(ts::SamplingParams {
                max_new_tokens: Some(stopping.max_new_tokens),
                temperature: Some(0.0),
                n: 1,
                ignore_eos: stopping.ignore_eos,
                ..Default::default()
            }),
            ..Default::default()
        },
        mapper,
    ))
}

impl Mapper {
    fn response(&mut self, frame: ts::GenerateResponse) -> Result<v::GenerateResponse, Status> {
        let invalid =
            || Status::internal("mock engine response violates the native token-ID profile");
        if self.finished {
            return Err(invalid());
        }
        let (ids, prompt_tokens, cached_tokens, finish_info) = match frame.response {
            Some(GenResp::Chunk(chunk)) => {
                if !self.streaming
                    || chunk.index != 0
                    || chunk.output_logprobs.is_some()
                    || chunk.token_ids.is_empty()
                    || chunk.completion_tokens as usize != self.sent.len() + chunk.token_ids.len()
                {
                    return Err(invalid());
                }
                (
                    chunk.token_ids,
                    chunk.prompt_tokens,
                    chunk.cached_tokens,
                    None,
                )
            }
            Some(GenResp::Complete(done)) => {
                if done.index != 0
                    || done.output_logprobs.is_some()
                    || done.matched_stop.is_some()
                    || done.completion_tokens as usize != done.output_ids.len()
                    || !done.output_ids.starts_with(&self.sent)
                {
                    return Err(invalid());
                }
                let reason = match done.finish_reason.as_str() {
                    "length" if done.completion_tokens == self.max_tokens => {
                        v::finish_info::FinishReason::Length
                    }
                    "abort" => v::finish_info::FinishReason::Aborted,
                    "error" => {
                        return Err(Status::internal("mock engine reported generation failure"))
                    }
                    _ => return Err(invalid()),
                };
                // TokenSpeed's Complete repeats all output IDs. Native outputs are deltas.
                let ids = done.output_ids[self.sent.len()..].to_vec();
                self.finished = true;
                (
                    ids,
                    done.prompt_tokens,
                    done.cached_tokens,
                    Some(v::FinishInfo {
                        num_output_tokens: done.completion_tokens,
                        finish_reason: reason as i32,
                        num_cached_tokens: Some(done.cached_tokens),
                        ..Default::default()
                    }),
                )
            }
            None => return Err(invalid()),
        };
        if self.sent.len() + ids.len() > self.max_tokens as usize
            || ids.iter().any(|id| *id >= self.vocab)
            || prompt_tokens as usize != self.prompt.len()
            || cached_tokens > prompt_tokens
        {
            return Err(invalid());
        }
        self.sent.extend_from_slice(&ids);
        let prompt_info = (!self.started).then(|| v::PromptInfo {
            num_prompt_tokens: prompt_tokens,
            token_ids: if self.include_prompt {
                self.prompt.clone()
            } else {
                Vec::new()
            },
            ..Default::default()
        });
        self.started = true;
        Ok(v::GenerateResponse {
            prompt_info,
            outputs: Some(v::SequenceOutput {
                num_tokens: ids.len() as u32,
                token_ids: ids,
                finish_info,
                ..Default::default()
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use tonic::Code;
    use vllm_proto::control_server::Control;

    use super::*;

    fn request() -> v::GenerateRequest {
        v::GenerateRequest {
            request_id: "native-unit".into(),
            prompt: Some(v::generate_request::Prompt::TokenIds(v::TokenIds {
                ids: vec![100; 16],
            })),
            stopping: Some(v::StoppingCriteria {
                max_new_tokens: 2,
                ..Default::default()
            }),
            response: Some(v::ResponseOptions {
                output_text: Some(false),
                output_token_ids: true,
                prompt_token_ids: true,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn model() -> ts::GetModelInfoResponse {
        ts::GetModelInfoResponse {
            served_model_name: "native-model".into(),
            vocab_size: 32000,
            max_context_length: 4096,
            ..Default::default()
        }
    }

    fn chunk(ids: Vec<u32>, total: u32) -> ts::GenerateResponse {
        ts::GenerateResponse {
            response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                token_ids: ids,
                prompt_tokens: 16,
                completion_tokens: total,
                cached_tokens: 8,
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    fn complete(ids: Vec<u32>, reason: &str) -> ts::GenerateResponse {
        ts::GenerateResponse {
            response: Some(GenResp::Complete(ts::GenerateComplete {
                completion_tokens: ids.len() as u32,
                output_ids: ids,
                prompt_tokens: 16,
                cached_tokens: 8,
                finish_reason: reason.into(),
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    #[test]
    fn neutral_defaults_and_declared_input_bounds_are_explicit() {
        let (normalized, _) = normalize(request(), true, &model()).unwrap();
        assert_eq!(normalized.sampling_params.unwrap().temperature, Some(0.0));
        let edits: &[fn(&mut v::GenerateRequest)] = &[
            |r| r.request_id.clear(),
            |r| r.request_id = "x".repeat(257),
            |r| r.model = "another-model".into(),
            |r| r.prompt = None,
            |r| r.prompt = Some(v::generate_request::Prompt::Text("x".into())),
            |r| {
                r.prompt = Some(v::generate_request::Prompt::TokenIds(v::TokenIds {
                    ids: vec![],
                }))
            },
            |r| {
                r.prompt = Some(v::generate_request::Prompt::TokenIds(v::TokenIds {
                    ids: vec![32000],
                }))
            },
            |r| r.stopping = None,
            |r| r.stopping.as_mut().unwrap().max_new_tokens = 0,
            |r| r.stopping.as_mut().unwrap().max_new_tokens = 4081,
        ];
        for (index, edit) in edits.iter().enumerate() {
            let mut r = request();
            edit(&mut r);
            assert_eq!(
                normalize(r, true, &model()).err().unwrap().code(),
                Code::InvalidArgument,
                "case {index}"
            );
        }
        let mut boundary = request();
        boundary.stopping.as_mut().unwrap().max_new_tokens = 4080;
        assert!(normalize(boundary, false, &model()).is_ok());
    }

    #[test]
    fn unsupported_native_options_are_not_silently_dropped() {
        let edits: &[fn(&mut v::GenerateRequest)] = &[
            |r| r.response = None,
            |r| r.response.as_mut().unwrap().output_text = None,
            |r| r.response.as_mut().unwrap().output_text = Some(true),
            |r| r.response.as_mut().unwrap().output_token_ids = false,
            |r| r.response.as_mut().unwrap().prompt_logprobs = true,
            |r| r.response.as_mut().unwrap().output_logprobs = true,
            |r| {
                r.response.as_mut().unwrap().prompt_candidates = Some(v::CandidateTokens::default())
            },
            |r| {
                r.response.as_mut().unwrap().output_candidates = Some(v::CandidateTokens::default())
            },
            |r| r.temperature = Some(f32::NAN),
            |r| {
                r.sampling = Some(v::RandomSampling {
                    num_sequences: 2,
                    ..Default::default()
                })
            },
            |r| {
                r.sampling = Some(v::RandomSampling {
                    seed: Some(0),
                    ..Default::default()
                })
            },
            |r| {
                r.sampling = Some(v::RandomSampling {
                    top_k: Some(1),
                    ..Default::default()
                })
            },
            |r| {
                r.sampling = Some(v::RandomSampling {
                    top_p: Some(0.9),
                    ..Default::default()
                })
            },
            |r| {
                r.sampling = Some(v::RandomSampling {
                    min_p: Some(0.1),
                    ..Default::default()
                })
            },
            |r| {
                r.decoding = Some(v::DecodingParameters {
                    presence_penalty: 1.0,
                    ..Default::default()
                })
            },
            |r| {
                r.decoding = Some(v::DecodingParameters {
                    frequency_penalty: 1.0,
                    ..Default::default()
                })
            },
            |r| {
                r.decoding = Some(v::DecodingParameters {
                    repetition_penalty: 2.0,
                    ..Default::default()
                })
            },
            |r| {
                r.decoding = Some(v::DecodingParameters {
                    allowed_token_ids: vec![100],
                    ..Default::default()
                })
            },
            |r| {
                r.decoding = Some(v::DecodingParameters {
                    bad_words_token_ids: vec![v::TokenIds { ids: vec![100] }],
                    ..Default::default()
                })
            },
            |r| {
                r.decoding = Some(v::DecodingParameters {
                    logit_bias: [(100, 1.0)].into(),
                    ..Default::default()
                })
            },
            |r| {
                r.decoding = Some(v::DecodingParameters {
                    structured_output: Some(v::decoding_parameters::StructuredOutput::JsonObject(
                        true,
                    )),
                    ..Default::default()
                })
            },
            |r| r.stopping.as_mut().unwrap().stop_token_ids.push(100),
            |r| {
                r.stopping
                    .as_mut()
                    .unwrap()
                    .stop_strings
                    .push("stop".into())
            },
            |r| r.stopping.as_mut().unwrap().include_stop_strings = true,
            |r| r.stopping.as_mut().unwrap().min_new_tokens = 1,
            |r| r.stopping.as_mut().unwrap().thinking_token_budget = Some(0),
            |r| {
                r.kv = Some(v::KvCacheParameters {
                    bypass_prefix_cache: true,
                    ..Default::default()
                })
            },
            |r| {
                r.kv = Some(v::KvCacheParameters {
                    cache_salt: "namespace".into(),
                    ..Default::default()
                })
            },
            |r| {
                r.kv = Some(v::KvCacheParameters {
                    kv_transfer_params: Some(Default::default()),
                    ..Default::default()
                })
            },
            |r| {
                r.kv = Some(v::KvCacheParameters {
                    ec_transfer_params: Some(Default::default()),
                    ..Default::default()
                })
            },
            |r| r.truncate_prompt_tokens = 1,
            |r| r.priority = 1,
            |r| r.session_id = Some(String::new()),
            |r| r.lora_name = "adapter".into(),
            |r| r.media.push(v::MediaItem::default()),
            |r| r.watermarking = Some(true),
            |r| r.kv_hints = Some(v::KvHintsEnvelope::default()),
            |r| r.engine_reasoning_gate = Some(v::EngineReasoningGate::default()),
        ];
        for (index, edit) in edits.iter().enumerate() {
            let mut r = request();
            edit(&mut r);
            assert_eq!(
                normalize(r, true, &model()).err().unwrap().code(),
                Code::Unimplemented,
                "case {index}"
            );
        }
    }

    #[test]
    fn native_terminal_counts_are_totals_and_output_ids_are_deltas() {
        let (_, mut mapper) = normalize(request(), true, &model()).unwrap();
        let first = mapper.response(chunk(vec![100, 101], 2)).unwrap();
        assert_eq!(first.prompt_info.unwrap().token_ids, vec![100; 16]);
        assert_eq!(first.outputs.unwrap().num_tokens, 2);
        let last = mapper.response(complete(vec![100, 101], "length")).unwrap();
        assert!(last.prompt_info.is_none());
        let output = last.outputs.unwrap();
        assert_eq!(output.num_tokens, 0);
        assert!(output.token_ids.is_empty());
        let done = output.finish_info.unwrap();
        assert_eq!(
            (done.num_output_tokens, done.num_cached_tokens),
            (2, Some(8))
        );
        assert_eq!(
            done.finish_reason,
            v::finish_info::FinishReason::Length as i32
        );
        assert!(mapper.response(complete(vec![100, 101], "length")).is_err());
        let (_, mut unary) = normalize(request(), false, &model()).unwrap();
        let output = unary
            .response(complete(vec![100, 101], "length"))
            .unwrap()
            .outputs
            .unwrap();
        assert_eq!((output.num_tokens, output.token_ids), (2, vec![100, 101]));
    }

    #[test]
    fn malformed_and_failed_engine_results_cannot_be_successful_native_results() {
        for frame in [
            chunk(vec![100], 2),
            chunk(vec![32000], 1),
            complete(vec![100], "length"),
            complete(vec![100], "error"),
            complete(vec![100], "stop"),
        ] {
            let (_, mut mapper) = normalize(request(), true, &model()).unwrap();
            assert_eq!(mapper.response(frame).unwrap_err().code(), Code::Internal);
        }
        let (_, mut mapper) = normalize(request(), true, &model()).unwrap();
        mapper.response(chunk(vec![100], 1)).unwrap();
        assert_eq!(
            mapper
                .response(complete(vec![101, 102], "length"))
                .unwrap_err()
                .code(),
            Code::Internal
        );
        let (_, mut aborted) = normalize(request(), true, &model()).unwrap();
        let output = aborted
            .response(complete(vec![], "abort"))
            .unwrap()
            .outputs
            .unwrap();
        assert_eq!(
            output.finish_info.unwrap().finish_reason,
            v::finish_info::FinishReason::Aborted as i32
        );
    }

    #[tokio::test]
    async fn premature_eof_and_transport_errors_remain_failures() {
        for status in [None, Some(Status::unavailable("injected"))] {
            let (_, mapper) = normalize(request(), true, &model()).unwrap();
            let mut frames = vec![Ok(chunk(vec![100], 1))];
            if let Some(status) = status.clone() {
                frames.push(Err(status));
            }
            let mut output = adapt(Box::pin(stream::iter(frames)), mapper);
            assert_eq!(
                output
                    .next()
                    .await
                    .unwrap()
                    .unwrap()
                    .outputs
                    .unwrap()
                    .num_tokens,
                1
            );
            assert_eq!(
                output.next().await.unwrap().unwrap_err().code(),
                status.map_or(Code::Internal, |s| s.code())
            );
            assert!(output.next().await.is_none());
        }
    }

    #[tokio::test]
    async fn native_abort_cancels_admission_for_unary_and_streaming() {
        tokio::time::timeout(Duration::from_secs(5), async {
            for streaming in [false, true] {
                let engine = Engine::spawn(engine::EngineParams::default());
                engine.fault_admit_delay_ms(60_000);
                let (_stop, stopped) = watch::channel(false);
                let service = MockNative(MockScheduler {
                    cfg: Arc::new(Config {
                        realistic: true,
                        ..Default::default()
                    }),
                    name: "native-generation-admission".into(),
                    engine: Some(engine.clone()),
                    capture: None,
                    kv_source: None,
                    stopped,
                    prefill_pending: Arc::new(AtomicU64::new(0)),
                    admissions: admission::Admissions::default(),
                    overrides: crate::overrides::for_port(0),
                });
                let mut pending =
                    Box::pin(service.generate_native(Request::new(request()), streaming));
                assert!(futures::poll!(&mut pending).is_pending());
                service
                    .abort(Request::new(v::AbortRequest {
                        request_ids: vec!["native-unit".into()],
                    }))
                    .await
                    .unwrap();
                assert_eq!(pending.await.err().unwrap().code(), Code::Cancelled);
                assert_eq!(service.0.prefill_pending.load(Ordering::Relaxed), 0);
                assert_eq!(
                    engine.load().num_running_reqs + engine.load().num_waiting_reqs,
                    0
                );
                engine.fault_admit_delay_ms(0);
                let mut output = service
                    .generate_native(Request::new(request()), streaming)
                    .await
                    .unwrap();
                let mut total = 0;
                let mut finished = false;
                while let Some(response) = output.next().await {
                    let out = response.unwrap().outputs.unwrap();
                    total += out.num_tokens;
                    if let Some(info) = out.finish_info {
                        assert_eq!(info.num_output_tokens, 2);
                        finished = true;
                    }
                }
                assert!(finished);
                assert_eq!(total, 2);
            }
        })
        .await
        .expect("admission cancellation completes within bound");
    }
}
