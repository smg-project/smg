//! TensorRT-LLM dialect: proto request → shim request translation and slim
//! output → TensorRT-LLM-proto response mapping.
//!
//! The engine side is the shim beside TensorRT-LLM's LLM API (it performs the
//! handshake, builds the engine's own `SamplingParams` from the request's
//! named fields, and answers with the slim records this crate's SGLang
//! mapping already decodes). What the wire carries is mapped here; what the
//! engine's Python servicer dropped silently is refused with a message.

use engine_zmq_client::{
    connector::TrtllmStream,
    protocol::{
        sglang::output::SglangOutput,
        trtllm::request::{TrtllmDisaggregated, TrtllmGenerateReqInput, TrtllmSamplingParams},
    },
};
use futures::Stream;
use smg_grpc_client::{trtllm_proto as trtllm, vllm_proto as vllm};

use crate::{
    abort_status::engine_abort_status,
    context_window::admit_prompt,
    fanout::fan_out_n,
    sglang::normalize_finish_reason,
    stream::{map_slim_output, matched_stop_of, poll_mapped, MappedGenerateStream, StreamState},
    structured_outputs::ConstraintRef,
};

/// Streaming generate output for one TensorRT-LLM sub-request, mapping each
/// slim record to a vLLM-proto `GenerateResponse` tagged with this sub's
/// choice `index`.
pub struct TrtllmGenerateStream {
    inner: TrtllmStream,
    state: StreamState,
    /// Choice index stamped on every chunk/complete (0 for n=1; the fan-out
    /// position for n>1): the proto's `sequence_index`.
    index: u32,
    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens: the tick's delta goes out as a `Chunk` first.
    pending: Option<vllm::GenerateResponse>,
    /// Whether the prompt logprobs went out on a chunk already (the proto
    /// carries them in the first token-bearing chunk only).
    input_logprobs_emitted: bool,
}

impl TrtllmGenerateStream {
    pub(crate) fn new(inner: TrtllmStream, index: u32) -> Self {
        Self {
            inner,
            state: StreamState::default(),
            index,
            pending: None,
            input_logprobs_emitted: false,
        }
    }

    /// The engine-side id of this choice's request.
    pub fn request_id(&self) -> &str {
        self.inner.request_id()
    }

    /// The engine index this choice's request went to; `None` for an engine
    /// whose identity names no index.
    pub fn engine_index(&self) -> Option<u32> {
        self.inner.engine_index()
    }

    /// Whether the engine already finished this choice and its terminal
    /// `Complete` is parked behind the chunk just yielded (the finish tick
    /// carried new tokens).
    pub fn has_parked_complete(&self) -> bool {
        self.pending.is_some()
    }

    /// End this choice as aborted: the engine's own parked `Complete` when it
    /// already finished, else a terminal `Complete` with `finish_reason =
    /// "abort"` and the output so far. The engine-side request is aborted
    /// when the stream is dropped, so the caller drops it next.
    pub fn complete_aborted(&mut self) -> vllm::GenerateResponse {
        if let Some(parked) = self.pending.take() {
            return parked;
        }
        let mut pending = None;
        // No new tokens on a frontend finish, so `emit_tick` yields the
        // `Complete` directly and parks nothing.
        let mut response = self.state.emit_tick(
            self.index,
            Vec::new(),
            None,
            Some(("abort".to_string(), None)),
            &mut pending,
        );
        self.state.attach_input_logprobs(
            &mut self.pending,
            &mut self.input_logprobs_emitted,
            &mut response,
        );
        response
    }
}

impl MappedGenerateStream for TrtllmGenerateStream {
    type Output = SglangOutput;
    type Inner = TrtllmStream;

    fn inner(&mut self) -> &mut Self::Inner {
        &mut self.inner
    }

    fn pending(&mut self) -> &mut Option<vllm::GenerateResponse> {
        &mut self.pending
    }

    fn map_output(
        &mut self,
        output: SglangOutput,
    ) -> Result<vllm::GenerateResponse, tonic::Status> {
        // The shim has no separate error finish: a request the engine could
        // not serve (refused while its parameters were built, failed inside
        // the engine, timed out there) ends as `abort` with the reason and an
        // HTTP-style status, mapped by the table every slim wire shares. The
        // frontend never asked for it (its own aborts drop the stream first),
        // so it is an error to the caller, not an empty success.
        if output.finish_reason.as_deref() == Some("abort") {
            let message = format!(
                "the engine aborted the request: {}",
                output
                    .finish_message
                    .as_deref()
                    .unwrap_or("no reason reported")
            );
            return Err(engine_abort_status(output.finish_status, message));
        }
        let finish = output.finish_reason.clone().map(|reason| {
            (
                normalize_finish_reason(&reason).to_string(),
                output.matched_stop.clone().map(matched_stop_of),
            )
        });
        Ok(map_slim_output(
            &mut self.state,
            self.index,
            output,
            finish,
            &mut self.pending,
            &mut self.input_logprobs_emitted,
        ))
    }
}

impl Stream for TrtllmGenerateStream {
    type Item = Result<vllm::GenerateResponse, tonic::Status>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        poll_mapped(self.get_mut(), cx)
    }
}

/// A vLLM-proto response of a TensorRT-LLM stream as the proto the
/// `TrtllmService` answers with, under `request_id` (the vLLM proto carries
/// none). The choice index is the proto's `sequence_index`; the logprob
/// columns become one `TokenLogprob` per token with its ranked candidates;
/// the prompt logprobs ride the `Complete`, as the engine's Python servicer
/// reports them. What neither path fills (perf metrics, logits, speculative
/// counters) stays at the proto defaults.
pub fn to_trtllm_response(
    request_id: &str,
    response: vllm::GenerateResponse,
) -> trtllm::GenerateResponse {
    use trtllm::generate_response::Response as TrtResponse;
    use vllm::generate_response::Response;
    let mapped = response.response.map(|inner| match inner {
        Response::Chunk(chunk) => TrtResponse::Chunk(trtllm::GenerateStreamChunk {
            token_ids: chunk.token_ids,
            sequence_index: chunk.index,
            prompt_tokens: chunk.prompt_tokens,
            completion_tokens: chunk.completion_tokens,
            cached_tokens: chunk.cached_tokens,
            logprobs: chunk
                .output_logprobs
                .map(output_token_logprobs)
                .unwrap_or_default(),
        }),
        Response::Complete(complete) => {
            use trtllm::generate_complete::MatchedStop as TrtMatchedStop;
            use vllm::generate_complete::MatchedStop;
            TrtResponse::Complete(trtllm::GenerateComplete {
                output_token_ids: complete.output_ids,
                sequence_index: complete.index,
                finish_reason: complete.finish_reason,
                matched_stop: complete.matched_stop.map(|matched| match matched {
                    MatchedStop::MatchedTokenId(id) => TrtMatchedStop::MatchedTokenId(id),
                    MatchedStop::MatchedStopStr(text) => TrtMatchedStop::MatchedStopStr(text),
                }),
                prompt_tokens: complete.prompt_tokens,
                completion_tokens: complete.completion_tokens,
                cached_tokens: complete.cached_tokens,
                logprobs: complete
                    .output_logprobs
                    .map(output_token_logprobs)
                    .unwrap_or_default(),
                prompt_logprobs: complete
                    .input_logprobs
                    .map(prompt_token_logprobs)
                    .unwrap_or_default(),
                ..Default::default()
            })
        }
    });
    trtllm::GenerateResponse {
        request_id: request_id.to_string(),
        response: mapped,
    }
}

/// The sampled-token columns as the proto's per-token records: the token,
/// its logprob, and the ranked candidates of its position when the request
/// asked for them.
fn output_token_logprobs(logprobs: vllm::OutputLogProbs) -> Vec<trtllm::TokenLogprob> {
    let mut top = logprobs.top_logprobs.into_iter();
    logprobs
        .token_ids
        .into_iter()
        .zip(logprobs.token_logprobs)
        .map(|(token_id, logprob)| trtllm::TokenLogprob {
            token_id,
            logprob,
            top_logprobs: top.next().map(top_logprobs).unwrap_or_default(),
        })
        .collect()
}

/// The prompt columns as the proto's per-token records: a position the
/// engine scored (the first prompt token has no logprob) carries its value,
/// the others `0.0`, as the engine's Python servicer reports a missing one.
fn prompt_token_logprobs(logprobs: vllm::InputLogProbs) -> Vec<trtllm::TokenLogprob> {
    let mut top = logprobs.top_logprobs.into_iter();
    logprobs
        .token_ids
        .into_iter()
        .zip(logprobs.token_logprobs)
        .map(|(token_id, logprob)| trtllm::TokenLogprob {
            token_id,
            logprob: logprob.value.unwrap_or(0.0),
            top_logprobs: top.next().map(top_logprobs).unwrap_or_default(),
        })
        .collect()
}

fn top_logprobs(top: vllm::TopLogProbs) -> Vec<trtllm::TopLogprob> {
    top.token_ids
        .into_iter()
        .zip(top.values)
        .map(|(token_id, logprob)| trtllm::TopLogprob { token_id, logprob })
        .collect()
}

/// Split an `n > 1` (`num_return_sequences`) proto request into `n`
/// single-sample sub-requests: the wire carries one sequence per request id.
/// The `{rid}-{i}` suffix is the shared fan-out convention; a seeded request
/// gives each sub its own seed (`seed + i`), so the samples differ as the
/// engine's own `n` makes them differ.
pub(crate) fn fan_out_trtllm_requests(
    req: trtllm::GenerateRequest,
) -> Vec<trtllm::GenerateRequest> {
    let n = req
        .sampling_config
        .as_ref()
        .map_or(1, |config| config.num_return_sequences.max(1));
    fan_out_n(req, n, |sub, i| {
        sub.request_id = format!("{}-{i}", sub.request_id);
        if let Some(config) = sub.sampling_config.as_mut() {
            config.num_return_sequences = 1;
            if let Some(seed) = config.seed {
                config.seed = Some(seed.wrapping_add(u64::from(i)));
            }
        }
    })
}

/// The guide kinds of the proto, by the LLM API's own `GuidedDecodingParams`
/// field names (what the shim builds from them).
fn guide_kind(guide_type: i32) -> Option<&'static str> {
    use trtllm::guided_decoding_params::GuideType;
    Some(match GuideType::try_from(guide_type).ok()? {
        GuideType::Json => "json_object",
        GuideType::JsonSchema => "json_schema",
        GuideType::Regex => "regex",
        GuideType::EbnfGrammar => "grammar",
        GuideType::StructuralTag => "structural_tag",
        GuideType::Unspecified => return None,
    })
}

/// The request's constraint as the shared refusal reads it
/// ([`crate::structured_outputs::refuse_unenforceable_constraint`]): the
/// proto's guide by kind, with its text.
pub(crate) fn constraint_ref(req: &trtllm::GenerateRequest) -> Option<ConstraintRef<'_>> {
    let guided = req.guided_decoding.as_ref()?;
    Some(match guide_kind(guided.guide_type)? {
        "json_object" => ConstraintRef::JsonObject,
        "json_schema" => ConstraintRef::JsonSchema(&guided.guide),
        "regex" => ConstraintRef::Regex(&guided.guide),
        "grammar" => ConstraintRef::Ebnf(&guided.guide),
        _ => ConstraintRef::StructuralTag(&guided.guide),
    })
}

/// A field the wire does not carry, refused with the way out rather than
/// dropped (the engine's Python servicer dropped these silently).
fn refuse(field: &str, why: &str) -> String {
    format!("{field} is not supported by the Rust servicer: {why}")
}

/// Translate a `TrtllmService` `GenerateRequest` into the shim's request.
/// ZMQ mode requires pre-tokenized input (SMG tokenizes upstream). The
/// sampling block carries the LLM API's own keywords; the engine's Python
/// servicer's safety defaults (temperature, top-p and repetition penalty at
/// 1.0 when unset, so the engine samples rather than decodes greedily) are
/// kept. `max_model_len` is the context window the engine reported at the
/// handshake: an empty prompt, or one that fills the window, is refused
/// before it reaches the engine ([`admit_prompt`]). Beam search and the
/// tensor-returning flags are refused: the slim record is one sequence per
/// request id and carries no logits.
pub(crate) fn translate_request_trtllm(
    req: trtllm::GenerateRequest,
    max_model_len: u64,
) -> Result<TrtllmGenerateReqInput, String> {
    let tokenized = req
        .tokenized
        .ok_or_else(|| "ZMQ mode requires pre-tokenized input; no input provided".to_string())?;
    let input_ids = tokenized.input_token_ids;
    admit_prompt(input_ids.len(), max_model_len)?;
    if !tokenized.query_token_ids.is_empty() {
        return Err(refuse(
            "tokenized.query_token_ids",
            "the wire carries one prompt",
        ));
    }
    if req.max_tokens == 0 {
        return Err("max_tokens is required and must be greater than 0".to_string());
    }
    let sampling = req.sampling_config.unwrap_or_default();
    let output = req.output_config.unwrap_or_default();
    if req
        .disaggregated_params
        .as_ref()
        .and_then(|params| params.context_phase_params.as_ref())
        .is_some_and(|phase| !phase.kv_cache_blocks.is_empty())
    {
        return Err(refuse(
            "disaggregated_params.context_phase_params.kv_cache_blocks",
            "the context phase's opaque state has no slot on this wire; the generation \
             phase takes the first token only",
        ));
    }
    if sampling.beam_width > 1 || !sampling.beam_width_array.is_empty() {
        return Err(refuse(
            "beam search (sampling_config.beam_width > 1)",
            "the wire carries one sampled sequence per request; use num_return_sequences for \
             independent samples",
        ));
    }
    if output.return_context_logits {
        return Err(refuse(
            "output_config.return_context_logits",
            "the wire carries no logits",
        ));
    }
    if output.return_generation_logits {
        return Err(refuse(
            "output_config.return_generation_logits",
            "the wire carries no logits",
        ));
    }
    if output.return_encoder_output {
        return Err(refuse(
            "output_config.return_encoder_output",
            "the wire carries no encoder output",
        ));
    }
    if req.kv_cache_retention.is_some() {
        return Err(refuse(
            "kv_cache_retention",
            "the wire carries no retention policy",
        ));
    }
    if req.prompt_tuning_config.is_some() {
        return Err(refuse(
            "prompt_tuning_config",
            "the wire carries no embedding table",
        ));
    }
    if req.lookahead_config.is_some() {
        return Err(refuse(
            "lookahead_config",
            "the wire carries no per-request decoding config",
        ));
    }
    let guided_decoding = match req.guided_decoding {
        None => None,
        Some(guided) => {
            let kind = guide_kind(guided.guide_type)
                .ok_or_else(|| "guided_decoding.guide_type must name a guide kind".to_string())?;
            if kind != "json_object" && guided.guide.is_empty() {
                return Err(format!(
                    "guided_decoding.guide is empty for guide type {kind}"
                ));
            }
            Some((kind.to_string(), guided.guide))
        }
    };
    let images: Vec<bytes::Bytes> = req
        .multimodal_input
        .map(|input| {
            input
                .image_data
                .into_iter()
                .map(bytes::Bytes::from)
                .collect()
        })
        .unwrap_or_default();
    Ok(TrtllmGenerateReqInput {
        rid: req.request_id,
        input_ids,
        sampling: TrtllmSamplingParams {
            max_tokens: req.max_tokens,
            // The engine's Python servicer's safety defaults: unset, these
            // would leave the engine's own (greedy) decoding.
            temperature: Some(sampling.temperature.unwrap_or(1.0)),
            top_p: Some(sampling.top_p.unwrap_or(1.0)),
            top_k: sampling.top_k,
            min_p: sampling.min_p,
            top_p_min: sampling.top_p_min,
            top_p_reset_ids: sampling.top_p_reset_ids,
            top_p_decay: sampling.top_p_decay,
            seed: sampling.seed,
            min_tokens: sampling.min_tokens,
            repetition_penalty: Some(sampling.repetition_penalty.unwrap_or(1.0)),
            presence_penalty: sampling.presence_penalty,
            frequency_penalty: sampling.frequency_penalty,
            length_penalty: sampling.length_penalty,
            early_stopping: sampling.early_stopping,
            no_repeat_ngram_size: sampling.no_repeat_ngram_size,
            beam_search_diversity_rate: sampling.beam_search_diversity_rate,
            prompt_ignore_length: sampling.prompt_ignore_length,
            stop: req.stop,
            stop_token_ids: req.stop_token_ids,
            // The proto's token flag feeds the LLM API's string flag, as the
            // engine's own servicer maps it (its request manager sets
            // `include_stop_str_in_output` from this field).
            include_stop_str_in_output: req.include_stop_token_in_output,
            bad: req.bad,
            bad_token_ids: req.bad_token_ids,
            ignore_eos: req.ignore_eos,
            // The shim checks the length against the vocabulary.
            embedding_bias: (!req.embedding_bias.is_empty()).then_some(req.embedding_bias),
            logprobs: output.logprobs,
            prompt_logprobs: output.prompt_logprobs,
            // Always true on this wire: the slim output record carries the
            // generated ids only, so the prompt never rides back (a caller
            // has its prompt). The proto's flag is a plain bool whose
            // default, `false`, is also "unset" (the engine's own default is
            // true), so a `false` cannot be told from an absent flag and is
            // not refused; the engine's own servicer forwards a `true` only
            // and lets a `false` fall to the engine's default, true, as well.
            exclude_input_from_output: true,
            // Not forwarded on either path: the proto's `perf_metrics` is
            // not filled by the engine's Python servicer either.
            return_perf_metrics: false,
            guided_decoding,
        },
        // One engine path for both kinds of client: the shim streams every
        // step, and a non-streaming client gets the terminal response alone.
        streaming: true,
        // `weights` and `config_json` are not forwarded on either path: the
        // adapter is resolved by its task id on the engine.
        lora_task_id: req
            .lora_config
            .filter(|config| config.task_id != 0)
            .map(|config| config.task_id),
        disaggregated: req.disaggregated_params.map(|params| {
            use trtllm::disaggregated_params::RequestType;
            let request_type = match RequestType::try_from(params.request_type) {
                Ok(RequestType::ContextOnly) => "context_only",
                Ok(RequestType::GenerationOnly) => "generation_only",
                _ => "context_and_generation",
            };
            TrtllmDisaggregated {
                request_type: request_type.to_string(),
                ctx_request_id: (!params.ctx_request_id.is_empty())
                    .then_some(params.ctx_request_id),
                // The first token crosses as sent: the message's presence
                // says whether there is one (id 0 is a token like any other).
                first_gen_token_id: params
                    .context_phase_params
                    .as_ref()
                    .map(|phase| phase.first_gen_token_id),
            }
        }),
        image_frames: u32::try_from(images.len()).map_err(|_| "too many images".to_string())?,
        cache_salt: req.cache_salt_id.map(|id| id.to_string()),
        trace_headers: None,
        arrival_time: req.arrival_time,
        images,
    })
}

#[cfg(test)]
mod tests {
    use engine_zmq_client::{
        codec::{decode_msgpack, encode_msgpack},
        mock_engine::default_ready_response,
        protocol::{
            sglang::output::{BatchTokenIDSlimOutput, MatchedStop},
            trtllm::request::TrtllmRequestType,
        },
    };
    use openai_protocol::worker::RuntimeType;

    use super::*;
    use crate::test_support::connected_pair;

    fn request(id: &str) -> trtllm::GenerateRequest {
        trtllm::GenerateRequest {
            request_id: id.to_string(),
            tokenized: Some(trtllm::TokenizedInput {
                original_text: String::new(),
                input_token_ids: vec![1, 2, 3],
                query_token_ids: vec![],
            }),
            sampling_config: Some(trtllm::SamplingConfig {
                beam_width: 1,
                num_return_sequences: 1,
                top_k: Some(40),
                temperature: Some(0.5),
                seed: Some(7),
                ..Default::default()
            }),
            output_config: Some(trtllm::OutputConfig {
                logprobs: Some(2),
                exclude_input_from_output: true,
                ..Default::default()
            }),
            max_tokens: 8,
            streaming: true,
            stop: vec!["\n\n".to_string()],
            stop_token_ids: vec![2],
            ..Default::default()
        }
    }

    /// The translation under the mock engine's context window.
    fn translate(req: trtllm::GenerateRequest) -> Result<TrtllmGenerateReqInput, String> {
        translate_request_trtllm(req, default_ready_response().max_model_len)
    }

    /// Every proto field with a slot on the wire lands in the LLM API's own
    /// keyword; the safety defaults fill what the proto left unset; images
    /// become auxiliary frames; the engine is always asked to stream.
    #[test]
    fn the_request_maps_onto_the_llm_apis_keywords() {
        let mut req = request("r1");
        req.guided_decoding = Some(trtllm::GuidedDecodingParams {
            guide_type: trtllm::guided_decoding_params::GuideType::JsonSchema as i32,
            guide: "{}".to_string(),
        });
        req.lora_config = Some(trtllm::LoraConfig {
            task_id: 3,
            weights: None,
            config_json: None,
        });
        req.multimodal_input = Some(trtllm::MultimodalInput {
            image_data: vec![b"img".to_vec()],
        });
        req.cache_salt_id = Some(99);
        req.disaggregated_params = Some(trtllm::DisaggregatedParams {
            request_type: trtllm::disaggregated_params::RequestType::GenerationOnly as i32,
            ctx_request_id: "ctx".to_string(),
            context_phase_params: Some(trtllm::ContextPhaseParams {
                first_gen_token_id: 11,
                kv_cache_blocks: vec![],
            }),
        });
        req.streaming = false;
        let translated = translate(req).unwrap();
        assert_eq!(translated.rid, "r1");
        assert_eq!(translated.input_ids, vec![1, 2, 3]);
        assert!(translated.streaming, "one engine path: always streamed");
        let sampling = &translated.sampling;
        assert_eq!(sampling.max_tokens, 8);
        assert_eq!(sampling.temperature, Some(0.5));
        assert_eq!(sampling.top_p, Some(1.0), "the safety default");
        assert_eq!(sampling.repetition_penalty, Some(1.0), "the safety default");
        assert_eq!(sampling.top_k, Some(40));
        assert_eq!(sampling.seed, Some(7));
        assert_eq!(sampling.stop, vec!["\n\n".to_string()]);
        assert_eq!(sampling.stop_token_ids, vec![2]);
        assert_eq!(sampling.logprobs, Some(2));
        assert!(sampling.exclude_input_from_output);
        assert_eq!(
            sampling.guided_decoding,
            Some(("json_schema".to_string(), "{}".to_string()))
        );
        assert_eq!(translated.lora_task_id, Some(3));
        assert_eq!(translated.cache_salt.as_deref(), Some("99"));
        assert_eq!(translated.image_frames, 1);
        assert_eq!(translated.images, vec![bytes::Bytes::from_static(b"img")]);
        assert_eq!(
            translated.disaggregated,
            Some(TrtllmDisaggregated {
                request_type: "generation_only".to_string(),
                ctx_request_id: Some("ctx".to_string()),
                first_gen_token_id: Some(11),
            })
        );
    }

    /// A generation phase's first token crosses as sent, id 0 included (the
    /// message's presence decides, not the id's value); the context phase's
    /// opaque state has no slot on this wire and is refused by name, never
    /// dropped.
    #[test]
    fn a_first_token_of_id_zero_crosses_and_an_opaque_state_is_refused() {
        let generation_only = |phase: Option<trtllm::ContextPhaseParams>| {
            let mut req = request("g");
            req.disaggregated_params = Some(trtllm::DisaggregatedParams {
                request_type: trtllm::disaggregated_params::RequestType::GenerationOnly as i32,
                ctx_request_id: "ctx".to_string(),
                context_phase_params: phase,
            });
            req
        };
        let first_token = |req| {
            translate(req)
                .unwrap()
                .disaggregated
                .and_then(|disaggregated| disaggregated.first_gen_token_id)
        };
        assert_eq!(
            first_token(generation_only(Some(trtllm::ContextPhaseParams {
                first_gen_token_id: 0,
                kv_cache_blocks: vec![],
            }))),
            Some(0)
        );
        assert_eq!(first_token(generation_only(None)), None);

        let refused = translate(generation_only(Some(trtllm::ContextPhaseParams {
            first_gen_token_id: 4,
            kv_cache_blocks: vec![1, 2, 3],
        })))
        .unwrap_err();
        assert!(refused.contains("kv_cache_blocks"), "{refused}");
    }

    /// What the wire does not carry is refused with the field named, where
    /// the engine's Python servicer dropped it silently; a missing prompt and
    /// a zero budget are refused too.
    #[test]
    fn the_unsupported_fields_are_refused_by_name() {
        let refused = |edit: fn(&mut trtllm::GenerateRequest)| {
            let mut req = request("r");
            edit(&mut req);
            translate(req).unwrap_err()
        };
        assert!(refused(|req| req.tokenized = None).contains("pre-tokenized"));
        assert!(refused(|req| req.max_tokens = 0).contains("max_tokens"));
        assert!(refused(|req| {
            req.sampling_config.as_mut().unwrap().beam_width = 4;
        })
        .contains("beam search"));
        assert!(refused(|req| {
            req.output_config.as_mut().unwrap().return_context_logits = true;
        })
        .contains("return_context_logits"));
        assert!(refused(|req| {
            req.output_config.as_mut().unwrap().return_generation_logits = true;
        })
        .contains("return_generation_logits"));
        assert!(refused(|req| {
            req.output_config.as_mut().unwrap().return_encoder_output = true;
        })
        .contains("return_encoder_output"));
        assert!(refused(|req| {
            req.kv_cache_retention = Some(trtllm::KvCacheRetentionConfig::default());
        })
        .contains("kv_cache_retention"));
        assert!(refused(|req| {
            req.prompt_tuning_config = Some(trtllm::PromptTuningConfig::default());
        })
        .contains("prompt_tuning_config"));
        assert!(refused(|req| {
            req.lookahead_config = Some(trtllm::LookaheadConfig::default());
        })
        .contains("lookahead_config"));
        assert!(refused(|req| {
            req.tokenized.as_mut().unwrap().query_token_ids = vec![1];
        })
        .contains("query_token_ids"));
        assert!(refused(|req| {
            req.guided_decoding = Some(trtllm::GuidedDecodingParams {
                guide_type: trtllm::guided_decoding_params::GuideType::Regex as i32,
                guide: String::new(),
            });
        })
        .contains("guide is empty"));
        assert!(refused(|req| {
            req.tokenized.as_mut().unwrap().input_token_ids = vec![];
        })
        .contains("empty"));
    }

    /// `n > 1` fans out to one sub per sample under the parent's id with
    /// derived seeds; the constraint reads by kind.
    #[test]
    fn fan_out_derives_seeds_and_the_constraint_reads_by_kind() {
        let mut req = request("p");
        req.sampling_config.as_mut().unwrap().num_return_sequences = 3;
        let subs = fan_out_trtllm_requests(req);
        assert_eq!(subs.len(), 3);
        for (i, sub) in subs.iter().enumerate() {
            assert_eq!(sub.request_id, format!("p-{i}"));
            let config = sub.sampling_config.as_ref().unwrap();
            assert_eq!(config.num_return_sequences, 1);
            assert_eq!(config.seed, Some(7 + i as u64));
        }
        let mut req = request("c");
        assert!(constraint_ref(&req).is_none());
        req.guided_decoding = Some(trtllm::GuidedDecodingParams {
            guide_type: trtllm::guided_decoding_params::GuideType::Json as i32,
            guide: String::new(),
        });
        assert!(matches!(
            constraint_ref(&req),
            Some(ConstraintRef::JsonObject)
        ));
        req.guided_decoding = Some(trtllm::GuidedDecodingParams {
            guide_type: trtllm::guided_decoding_params::GuideType::EbnfGrammar as i32,
            guide: "root ::= \"a\"".to_string(),
        });
        assert!(matches!(
            constraint_ref(&req),
            Some(ConstraintRef::Ebnf("root ::= \"a\""))
        ));
    }

    /// The proto response: the choice index as `sequence_index`, one
    /// `TokenLogprob` per token with its candidates, the matched stop, the
    /// prompt logprobs on the `Complete`.
    #[test]
    fn responses_carry_the_protos_logprob_records_and_the_matched_stop() {
        let chunk = vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Chunk(
                vllm::GenerateStreamChunk {
                    token_ids: vec![10, 11],
                    prompt_tokens: 3,
                    completion_tokens: 2,
                    cached_tokens: 1,
                    output_logprobs: Some(vllm::OutputLogProbs {
                        token_logprobs: vec![-0.5, -0.25],
                        token_ids: vec![10, 11],
                        top_logprobs: vec![vllm::TopLogProbs {
                            values: vec![-0.5, -1.0],
                            token_ids: vec![10, 12],
                        }],
                    }),
                    index: 2,
                    ..Default::default()
                },
            )),
        };
        let mapped = to_trtllm_response("r", chunk);
        assert_eq!(mapped.request_id, "r");
        let Some(trtllm::generate_response::Response::Chunk(chunk)) = mapped.response else {
            panic!("expected a chunk");
        };
        assert_eq!(chunk.sequence_index, 2);
        assert_eq!(chunk.token_ids, vec![10, 11]);
        assert_eq!(chunk.cached_tokens, 1);
        assert_eq!(chunk.logprobs.len(), 2);
        assert_eq!(chunk.logprobs[0].token_id, 10);
        assert_eq!(chunk.logprobs[0].logprob, -0.5);
        assert_eq!(chunk.logprobs[0].top_logprobs.len(), 2);
        assert_eq!(chunk.logprobs[0].top_logprobs[1].token_id, 12);
        assert!(chunk.logprobs[1].top_logprobs.is_empty());

        let complete = vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Complete(
                vllm::GenerateComplete {
                    output_ids: vec![10, 11],
                    finish_reason: "stop".to_string(),
                    prompt_tokens: 3,
                    completion_tokens: 2,
                    matched_stop: Some(vllm::generate_complete::MatchedStop::MatchedStopStr(
                        "\n\n".to_string(),
                    )),
                    input_logprobs: Some(vllm::InputLogProbs {
                        token_logprobs: vec![
                            vllm::InputTokenLogProb { value: None },
                            vllm::InputTokenLogProb { value: Some(-2.0) },
                        ],
                        token_ids: vec![1, 2],
                        top_logprobs: vec![],
                    }),
                    index: 0,
                    ..Default::default()
                },
            )),
        };
        let Some(trtllm::generate_response::Response::Complete(complete)) =
            to_trtllm_response("r", complete).response
        else {
            panic!("expected a complete");
        };
        assert_eq!(complete.output_token_ids, vec![10, 11]);
        assert_eq!(complete.finish_reason, "stop");
        assert_eq!(
            complete.matched_stop,
            Some(trtllm::generate_complete::MatchedStop::MatchedStopStr(
                "\n\n".to_string()
            ))
        );
        assert_eq!(complete.prompt_logprobs.len(), 2);
        assert_eq!(complete.prompt_logprobs[0].logprob, 0.0);
        assert_eq!(complete.prompt_logprobs[1].token_id, 2);
        assert_eq!(complete.prompt_logprobs[1].logprob, -2.0);
    }

    /// End-to-end over ipc:// for a TensorRT-LLM backend: the adapter frames a
    /// tagged `TrtllmGenerateReqInput` with the image behind it, and maps the
    /// slim batches back to vLLM-proto responses. The mock engine speaks the
    /// shared transport with raw frames, as the shim does.
    #[tokio::test]
    async fn generate_e2e_translates_and_streams_trtllm() {
        let (client, engine, _dir) = connected_pair(RuntimeType::Trtllm).await;

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let frames = input.recv_frames().await.unwrap();
            assert_eq!(
                TrtllmRequestType::from_frame(frames[0].as_ref()),
                Some(TrtllmRequestType::Add)
            );
            let request: TrtllmGenerateReqInput = decode_msgpack(frames[1].as_ref()).unwrap();
            assert_eq!(request.rid, "r1");
            assert_eq!(request.input_ids, vec![1, 2, 3]);
            assert_eq!(request.sampling.max_tokens, 8);
            assert_eq!(request.image_frames, 1);
            assert_eq!(frames.len(), 3, "the payload and one image frame");
            assert_eq!(frames[2].as_ref(), b"img");

            let step = |ids: Vec<u32>, completion: u32, finish: &str| BatchTokenIDSlimOutput {
                rids: vec!["r1".into()],
                output_ids: vec![ids],
                finished_reasons: vec![finish.to_string()],
                finished_messages: vec![None],
                finished_matched: vec![(finish == "stop").then_some(MatchedStop::TokenId(2))],
                prompt_tokens: vec![3],
                completion_tokens: vec![completion],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-0.5]],
                output_token_logprobs_idx: vec![vec![10]],
                engine_index: 0,
                num_running: 1,
                num_waiting: 0,
                kv_used_tokens: 10,
                kv_total_tokens: 100,
                finished_status: vec![None],
                ..Default::default()
            };
            for batch in [step(vec![10], 1, ""), step(vec![2], 2, "stop")] {
                output
                    .send_frames(vec![bytes::Bytes::from(encode_msgpack(&batch).unwrap())])
                    .await
                    .unwrap();
            }
            // Keep the peer alive until the frontend has consumed the replies.
            (input, output)
        });

        let mut req = request("r1");
        req.multimodal_input = Some(trtllm::MultimodalInput {
            image_data: vec![b"img".to_vec()],
        });
        let mut stream = client.generate_trtllm(req).await.unwrap();
        let first = stream.next().await.unwrap().unwrap();
        let Some(vllm::generate_response::Response::Chunk(chunk)) = first.response else {
            panic!("expected a chunk");
        };
        assert_eq!(chunk.token_ids, vec![10]);
        assert_eq!(chunk.output_logprobs.unwrap().token_logprobs, vec![-0.5]);
        let second = stream.next().await.unwrap().unwrap();
        let Some(vllm::generate_response::Response::Chunk(chunk)) = second.response else {
            panic!("expected the finish tick's delta as a chunk");
        };
        assert_eq!(chunk.token_ids, vec![2]);
        let third = stream.next().await.unwrap().unwrap();
        let Some(vllm::generate_response::Response::Complete(complete)) = third.response else {
            panic!("expected a complete");
        };
        assert_eq!(complete.output_ids, vec![10, 2]);
        assert_eq!(complete.finish_reason, "stop");
        assert_eq!(
            complete.matched_stop,
            Some(vllm::generate_complete::MatchedStop::MatchedTokenId(2))
        );
        assert_eq!(complete.completion_tokens, 2);
        assert!(stream.next().await.is_none());
        engine_task.await.unwrap();
    }

    /// An `abort` from the engine side is the caller's error with the status
    /// its HTTP code means, not an empty success.
    #[tokio::test]
    async fn an_engine_abort_is_an_error_to_the_caller() {
        let (client, engine, _dir) = connected_pair(RuntimeType::Trtllm).await;
        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let _ = input.recv_frames().await.unwrap();
            let batch = BatchTokenIDSlimOutput {
                rids: vec!["r2".into()],
                output_ids: vec![vec![]],
                finished_reasons: vec!["abort".to_string()],
                finished_messages: vec![Some(
                    "embedding_bias length (3) must match vocab_size (10)".into(),
                )],
                finished_matched: vec![None],
                prompt_tokens: vec![3],
                completion_tokens: vec![0],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![]],
                output_token_logprobs_idx: vec![vec![]],
                finished_status: vec![Some(400)],
                ..Default::default()
            };
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&batch).unwrap())])
                .await
                .unwrap();
            // Keep the peer alive until the frontend has read the abort.
            (input, output)
        });
        let mut stream = client.generate_trtllm(request("r2")).await.unwrap();
        let error = stream.next().await.unwrap().unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(error.message().contains("embedding_bias"), "{error}");
        engine_task.await.unwrap();
    }
}
