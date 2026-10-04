//! SGLang dialect: proto request → scheduler request translation and
//! scheduler output → vLLM-proto response mapping.
//!
//! The scheduler side is SGLang's own, unchanged; it joins SMG's transport
//! through the SGLang plugin in `smg_grpc_servicer.sglang` (which performs
//! the handshake inside the scheduler process, normalizes and verifies the
//! sampling params SMG sends, and slices each step's output to the slim
//! batch decoded here).

use engine_zmq_client::{
    codec::OpaqueValue,
    connector::SglangStream,
    protocol::sglang::{
        output::{MatchedStop, SglangOutput},
        request::{TokenizedEmbeddingReqInput, TokenizedGenerateReqInput},
        sampling::SamplingParams as SglangSamplingParams,
    },
};
use futures::Stream;
use smg_grpc_client::{sglang_proto, vllm_proto as vllm};
use tracing::warn;

use crate::{
    fanout::fan_out_n,
    stream::{poll_mapped, MappedGenerateStream, StreamState},
};

/// Streaming generate output for one SGLang sub-request, mapping each
/// [`SglangOutput`] to a vLLM-proto `GenerateResponse` tagged with this sub's
/// choice `index`.
pub struct SglangGenerateStream {
    inner: SglangStream,
    state: StreamState,
    /// Choice index stamped on every chunk/complete (0 for n=1; the fan-out
    /// position for n>1).
    index: u32,
    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens: the tick's delta goes out as a `Chunk` first.
    pending: Option<vllm::GenerateResponse>,
    /// The scheduler's running count of reasoning tokens for this choice
    /// (hybrid-reasoning models; 0 otherwise). The vLLM-proto intermediate
    /// has no slot for it, so the SGLang-proto mapping reads it from here.
    reasoning_tokens: u32,
    /// Whether the prompt logprobs went out on a chunk already (the proto
    /// carries them in the first token-bearing chunk only).
    input_logprobs_emitted: bool,
}

impl SglangGenerateStream {
    pub(crate) fn new(inner: SglangStream, index: u32) -> Self {
        Self {
            inner,
            state: StreamState::default(),
            index,
            pending: None,
            reasoning_tokens: 0,
            input_logprobs_emitted: false,
        }
    }

    /// Reasoning tokens the scheduler has counted for this choice so far.
    pub fn reasoning_tokens(&self) -> u32 {
        self.reasoning_tokens
    }

    /// Attach the accumulated prompt logprobs: once on the first token-bearing
    /// chunk and on every `Complete`, including one parked in `pending`. The
    /// scheduler reports them with the prefill tick, before any sampled token.
    fn attach_input_logprobs(&mut self, response: &mut vllm::GenerateResponse) {
        if self.state.prompt_logprobs.is_empty() {
            return;
        }
        let state = &self.state;
        let build = || vllm::InputLogProbs {
            token_logprobs: state.prompt_logprobs.clone(),
            token_ids: state.prompt_token_ids.clone(),
            top_logprobs: state.prompt_top_logprobs.clone(),
        };
        if let Some(vllm::generate_response::Response::Complete(parked)) = self
            .pending
            .as_mut()
            .and_then(|pending| pending.response.as_mut())
        {
            parked.input_logprobs = Some(build());
        }
        match response.response.as_mut() {
            Some(vllm::generate_response::Response::Chunk(chunk))
                if !self.input_logprobs_emitted && !chunk.token_ids.is_empty() =>
            {
                chunk.input_logprobs = Some(build());
                self.input_logprobs_emitted = true;
            }
            Some(vllm::generate_response::Response::Chunk(_)) => {}
            Some(vllm::generate_response::Response::Complete(complete)) => {
                complete.input_logprobs = Some(build());
            }
            None => {}
        }
    }

    /// End this choice as aborted: the engine's own parked `Complete` when it
    /// already finished, else the terminal `Complete` the Python servicer
    /// yields after an `Abort` RPC (`finish_reason = "abort"`). The
    /// engine-side request is aborted when the stream is dropped, so the
    /// caller drops it next.
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
        self.attach_input_logprobs(&mut response);
        response
    }
}

/// A vLLM-proto response of an SGLang stream as the SGLang proto the
/// `SglangScheduler` service answers with, under `request_id` (the vLLM proto
/// carries none) and the stream's `reasoning_tokens` (which it has no slot
/// for). The two share the chunk and completion fields this wire fills; what
/// it does not carry (hidden states, speculative counters) stays at the proto
/// defaults, as it does on the Python servicer.
pub fn to_sglang_response(
    request_id: &str,
    response: vllm::GenerateResponse,
    reasoning_tokens: u32,
) -> sglang_proto::GenerateResponse {
    use sglang_proto::generate_response::Response as SgResponse;
    use vllm::generate_response::Response;
    let top = |top: Vec<vllm::TopLogProbs>| -> Vec<sglang_proto::TopLogProbs> {
        top.into_iter()
            .map(|top| sglang_proto::TopLogProbs {
                values: top.values,
                token_ids: top.token_ids,
            })
            .collect()
    };
    let logprobs = |logprobs: Option<vllm::OutputLogProbs>| {
        logprobs.map(|lp| sglang_proto::OutputLogProbs {
            token_logprobs: lp.token_logprobs,
            token_ids: lp.token_ids,
            top_logprobs: top(lp.top_logprobs),
        })
    };
    let input_logprobs = |logprobs: Option<vllm::InputLogProbs>| {
        logprobs.map(|lp| sglang_proto::InputLogProbs {
            token_logprobs: lp
                .token_logprobs
                .into_iter()
                .map(|entry| sglang_proto::InputTokenLogProb { value: entry.value })
                .collect(),
            token_ids: lp.token_ids,
            top_logprobs: top(lp.top_logprobs),
        })
    };
    let mapped = response.response.map(|inner| match inner {
        Response::Chunk(chunk) => SgResponse::Chunk(sglang_proto::GenerateStreamChunk {
            token_ids: chunk.token_ids,
            prompt_tokens: chunk.prompt_tokens,
            completion_tokens: chunk.completion_tokens,
            cached_tokens: chunk.cached_tokens,
            reasoning_tokens,
            output_logprobs: logprobs(chunk.output_logprobs),
            input_logprobs: input_logprobs(chunk.input_logprobs),
            index: chunk.index,
            ..Default::default()
        }),
        Response::Complete(complete) => {
            use sglang_proto::generate_complete::MatchedStop as SgMatchedStop;
            use vllm::generate_complete::MatchedStop;
            SgResponse::Complete(sglang_proto::GenerateComplete {
                output_ids: complete.output_ids,
                finish_reason: complete.finish_reason,
                prompt_tokens: complete.prompt_tokens,
                completion_tokens: complete.completion_tokens,
                cached_tokens: complete.cached_tokens,
                reasoning_tokens,
                output_logprobs: logprobs(complete.output_logprobs),
                input_logprobs: input_logprobs(complete.input_logprobs),
                matched_stop: complete.matched_stop.map(|matched| match matched {
                    MatchedStop::MatchedTokenId(id) => SgMatchedStop::MatchedTokenId(id),
                    MatchedStop::MatchedStopStr(text) => SgMatchedStop::MatchedStopStr(text),
                }),
                index: complete.index,
                ..Default::default()
            })
        }
    });
    sglang_proto::GenerateResponse {
        request_id: request_id.to_string(),
        response: mapped,
    }
}

impl MappedGenerateStream for SglangGenerateStream {
    type Output = SglangOutput;
    type Inner = SglangStream;

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
        let state = &mut self.state;
        // The scheduler reports per-request counts directly (cumulative for
        // completions).
        if output.prompt_tokens > 0 {
            state.prompt_tokens = output.prompt_tokens;
        }
        if output.cached_tokens > 0 {
            state.cached_tokens = output.cached_tokens;
        }
        state.completion_tokens = output.completion_tokens;
        state.output_ids.extend(output.output_ids.iter().copied());
        self.reasoning_tokens = output.reasoning_tokens;
        // Prompt logprobs ride the prefill tick (once per request; chunked
        // prefill may split them), in SGLang's shape: no value for the first
        // prompt token, ranked candidates per position when asked.
        if !output.input_logprobs_val.is_empty() {
            state
                .prompt_logprobs
                .extend(
                    output
                        .input_logprobs_val
                        .iter()
                        .map(|lp| vllm::InputTokenLogProb {
                            value: lp.map(|lp| lp as f32),
                        }),
                );
            state
                .prompt_token_ids
                .extend(output.input_logprobs_idx.iter().copied());
            state.prompt_top_logprobs.extend(
                output
                    .input_top_logprobs_val
                    .iter()
                    .zip(&output.input_top_logprobs_idx)
                    .map(|(values, token_ids)| vllm::TopLogProbs {
                        values: values.iter().map(|&lp| lp as f32).collect(),
                        token_ids: token_ids.clone(),
                    }),
            );
        }

        // Ranked candidates per position, as the scheduler reports them (the
        // gRPC servicer forwards the same lists unchanged).
        let tick_top_logprobs: Vec<vllm::TopLogProbs> = output
            .output_top_logprobs_val
            .iter()
            .zip(&output.output_top_logprobs_idx)
            .map(|(values, token_ids)| vllm::TopLogProbs {
                values: values.iter().map(|&lp| lp as f32).collect(),
                token_ids: token_ids.clone(),
            })
            .collect();
        let chunk_logprobs =
            (!output.output_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
                token_logprobs: output
                    .output_logprobs_val
                    .iter()
                    .map(|&lp| lp as f32)
                    .collect(),
                token_ids: output.output_logprobs_idx.clone(),
                top_logprobs: tick_top_logprobs.clone(),
            });
        state
            .output_logprobs_val
            .extend(output.output_logprobs_val.iter().map(|&lp| lp as f32));
        state
            .output_logprobs_idx
            .extend(output.output_logprobs_idx.iter().copied());
        state.output_top_logprobs.extend(tick_top_logprobs);

        // SGLang has no separate error finish: a request it could not serve
        // (rejected by the plugin's validation, refused at admission, dropped
        // from a full queue) ends as `abort` with the reason and, when the
        // request is at fault, HTTP 400. The frontend never asked for it (its
        // own aborts drop the stream first), so it is an error to the caller,
        // not an empty success, with the status mapped as SGLang's gRPC
        // servicer maps it.
        if output.finish_reason.as_deref() == Some("abort") {
            let message = format!(
                "the scheduler aborted the request: {}",
                output
                    .finish_message
                    .as_deref()
                    .unwrap_or("no reason reported")
            );
            return Err(match output.finish_status {
                Some(400) => tonic::Status::invalid_argument(message),
                Some(503) => tonic::Status::unavailable(message),
                _ => tonic::Status::internal(message),
            });
        }
        let finish = output.finish_reason.map(|reason| {
            (
                normalize_finish_reason(&reason).to_string(),
                output.matched_stop.map(|matched| match matched {
                    MatchedStop::TokenId(id) => {
                        vllm::generate_complete::MatchedStop::MatchedTokenId(id)
                    }
                    MatchedStop::Text(text) => {
                        vllm::generate_complete::MatchedStop::MatchedStopStr(text)
                    }
                }),
            )
        });
        let mut response = state.emit_tick(
            self.index,
            output.output_ids,
            chunk_logprobs,
            finish,
            &mut self.pending,
        );
        self.attach_input_logprobs(&mut response);
        Ok(response)
    }
}

impl Stream for SglangGenerateStream {
    type Item = Result<vllm::GenerateResponse, tonic::Status>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        poll_mapped(self.get_mut(), cx)
    }
}

/// Split an `n > 1` SGLang proto request into `n` single-sample sub-requests
/// (the plugin refuses `n != 1`: the frontend owns the fan-out). The SGLang
/// proto carries no seed, so the samples are independent by construction.
/// The `{rid}-{i}` suffix is the shared fan-out convention; note the scheduler
/// aborts by rid *prefix*, so aborting one sub also aborts the subs whose
/// index extends it (`r-1` covers `r-10..`). The gateway only ever drops a
/// whole fan-out, so that over-match is harmless today.
pub(crate) fn fan_out_sglang_requests(
    req: sglang_proto::GenerateRequest,
) -> Vec<sglang_proto::GenerateRequest> {
    let n = req.sampling_params.as_ref().map_or(1, |sp| sp.n.max(1));
    fan_out_n(req, n, |sub, i| {
        sub.request_id = format!("{}-{i}", sub.request_id);
        if let Some(sp) = sub.sampling_params.as_mut() {
            sp.n = 1;
        }
    })
}

/// Translate an SGLang proto `GenerateRequest` into the scheduler's
/// `TokenizedGenerateReqInput`. ZMQ mode requires pre-tokenized input (SMG
/// tokenizes upstream). The slim output carries sampled and prompt logprobs
/// with their ranked candidates; what it has no columns for (hidden states,
/// per-token-id logprobs, which the Python servicer drops silently) and what
/// the wire does not carry (multimodal payloads, PD bootstrap fields) is
/// refused rather than silently dropped.
pub(crate) fn translate_request_sglang(
    req: sglang_proto::GenerateRequest,
) -> Result<TokenizedGenerateReqInput, String> {
    let input_ids = req
        .tokenized
        .map(|tokenized| tokenized.input_ids)
        .ok_or_else(|| "ZMQ mode requires pre-tokenized input; no input provided".to_string())?;
    if req.mm_inputs.is_some() {
        return Err(
            "multimodal inputs are not supported over the SGLang ZMQ backend yet".to_string(),
        );
    }
    if !req.token_ids_logprob.is_empty() {
        return Err("token_ids_logprob is not supported over the SGLang ZMQ backend".to_string());
    }
    if req.return_hidden_states {
        return Err("hidden states are not supported over the SGLang ZMQ backend".to_string());
    }
    // The bootstrap slots stay at the scheduler's defaults on this wire;
    // refuse rather than run without the PD bootstrap the request asked for.
    if req.disaggregated_params.is_some() {
        return Err(
            "PD disaggregation (disaggregated_params) is not supported over the SGLang ZMQ \
             backend"
                .to_string(),
        );
    }
    Ok(TokenizedGenerateReqInput {
        rid: req.request_id,
        input_ids,
        sampling_params: req
            .sampling_params
            .map(translate_sampling_sglang)
            .unwrap_or_default(),
        return_logprob: req.return_logprob,
        logprob_start_len: req.logprob_start_len,
        top_logprobs_num: u32::try_from(req.top_logprobs_num).unwrap_or(0),
        stream: req.stream,
        // Forwarded as the Python servicer forwards them: the scheduler
        // resolves the adapter id and runs the processor it was started with.
        lora_id: (!req.lora_id.is_empty()).then_some(req.lora_id),
        custom_logit_processor: (!req.custom_logit_processor.is_empty())
            .then_some(req.custom_logit_processor),
        require_reasoning: req.require_reasoning,
        ..TokenizedGenerateReqInput::default()
    })
}

/// Translate an SGLang proto `EmbedRequest` into the scheduler's
/// `TokenizedEmbeddingReqInput`, as the Python servicer does: the tokenized
/// prompt with `max_new_tokens = 0` (the scheduler pools, nothing samples),
/// token type ids for cross-encoder inputs. Multimodal payloads are refused.
pub(crate) fn translate_embed_request_sglang(
    req: sglang_proto::EmbedRequest,
) -> Result<TokenizedEmbeddingReqInput, String> {
    if req.request_id.is_empty() {
        return Err("request_id is required".to_string());
    }
    let Some(tokenized) = req.tokenized else {
        return Err("EmbedRequest requires tokenized input".to_string());
    };
    if tokenized.input_ids.is_empty() {
        return Err("the prompt cannot be empty".to_string());
    }
    if req.mm_inputs.is_some() {
        return Err(
            "multimodal inputs are not supported over the SGLang ZMQ backend yet".to_string(),
        );
    }
    let mut sampling_params = req
        .sampling_params
        .map(translate_sampling_sglang)
        .unwrap_or_default();
    sampling_params.max_new_tokens = Some(0);
    Ok(TokenizedEmbeddingReqInput {
        rid: req.request_id,
        input_text: (!tokenized.original_text.is_empty()).then_some(tokenized.original_text),
        input_ids: tokenized.input_ids,
        sampling_params,
        token_type_ids: req.token_type_ids,
        dimensions: None,
    })
}

/// A `StartProfile` request for the scheduler, as the Python servicer builds
/// its `ProfileReq`; `None`s keep the scheduler side's defaults (its
/// `SGLANG_PROFILE_*` environment).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SglangProfileStart {
    pub output_dir: Option<String>,
    pub start_step: Option<i64>,
    pub num_steps: Option<i64>,
    pub activities: Vec<String>,
    pub with_stack: Option<bool>,
    pub record_shapes: Option<bool>,
    pub profile_by_stage: bool,
    pub profile_id: String,
}

impl SglangProfileStart {
    /// The call's one argument: a map of the set options.
    pub(crate) fn into_opaque(self) -> OpaqueValue {
        let mut entries: Vec<(OpaqueValue, OpaqueValue)> = Vec::new();
        let mut put = |key: &str, value: OpaqueValue| {
            entries.push((OpaqueValue::from(key), value));
        };
        if let Some(dir) = self.output_dir {
            put("output_dir", OpaqueValue::from(dir));
        }
        if let Some(step) = self.start_step {
            put("start_step", OpaqueValue::from(step));
        }
        if let Some(steps) = self.num_steps {
            put("num_steps", OpaqueValue::from(steps));
        }
        if !self.activities.is_empty() {
            put(
                "activities",
                OpaqueValue::Array(self.activities.into_iter().map(OpaqueValue::from).collect()),
            );
        }
        if let Some(with_stack) = self.with_stack {
            put("with_stack", OpaqueValue::Boolean(with_stack));
        }
        if let Some(record_shapes) = self.record_shapes {
            put("record_shapes", OpaqueValue::Boolean(record_shapes));
        }
        put(
            "profile_by_stage",
            OpaqueValue::Boolean(self.profile_by_stage),
        );
        put("profile_id", OpaqueValue::from(self.profile_id));
        OpaqueValue::Map(entries)
    }
}

/// A control reply's `success` and `message` (the plugin's
/// `ControlReplySlim` as a utility outcome).
pub(crate) fn control_outcome(value: &OpaqueValue) -> (bool, String) {
    let mut success = false;
    let mut message = String::new();
    for (key, value) in value.as_map().into_iter().flatten() {
        match key.as_str() {
            Some("success") => success = value.as_bool().unwrap_or(false),
            Some("message") => message = value.as_str().unwrap_or_default().to_string(),
            _ => {}
        }
    }
    (success, message)
}

/// Per-rank control replies as one answer, as the Python servicer aggregates
/// its communicator results: every rank must succeed, failures report their
/// messages joined.
pub(crate) fn aggregate_control(results: &[(bool, String)], ok_message: &str) -> (bool, String) {
    if results.is_empty() {
        return (false, "No response from scheduler".to_string());
    }
    let failures: Vec<&str> = results
        .iter()
        .filter(|(success, _)| !success)
        .map(|(_, message)| {
            if message.is_empty() {
                "failed"
            } else {
                message.as_str()
            }
        })
        .collect();
    if failures.is_empty() {
        (true, ok_message.to_string())
    } else {
        (false, failures.join(" | "))
    }
}

/// A proto `Struct` (SGLang's `custom_params`) as the msgpack value the
/// scheduler's `SamplingParams.custom_params` dict decodes from.
fn struct_to_opaque(value: prost_types::Struct) -> OpaqueValue {
    OpaqueValue::Map(
        value
            .fields
            .into_iter()
            .map(|(key, value)| (OpaqueValue::from(key), value_to_opaque(value)))
            .collect(),
    )
}

fn value_to_opaque(value: prost_types::Value) -> OpaqueValue {
    use prost_types::value::Kind;
    match value.kind {
        None | Some(Kind::NullValue(_)) => OpaqueValue::Nil,
        Some(Kind::NumberValue(number)) => OpaqueValue::F64(number),
        Some(Kind::StringValue(text)) => OpaqueValue::from(text),
        Some(Kind::BoolValue(flag)) => OpaqueValue::Boolean(flag),
        Some(Kind::StructValue(fields)) => struct_to_opaque(fields),
        Some(Kind::ListValue(list)) => {
            OpaqueValue::Array(list.values.into_iter().map(value_to_opaque).collect())
        }
    }
}

/// Map the proto sampling params onto SGLang's own struct, in its API-input
/// form: the plugin runs the scheduler's `normalize()`/`verify()` on receipt.
/// String `stop` sequences ride along: the headless scheduler keeps its
/// tokenizer and matches them itself (the gateway's own lane strips them
/// upstream and sends none).
pub(crate) fn translate_sampling_sglang(sp: sglang_proto::SamplingParams) -> SglangSamplingParams {
    use sglang_proto::sampling_params::Constraint;

    let mut params = SglangSamplingParams {
        max_new_tokens: sp.max_new_tokens,
        stop: (!sp.stop.is_empty()).then_some(sp.stop),
        stop_token_ids: (!sp.stop_token_ids.is_empty()).then_some(sp.stop_token_ids),
        temperature: f64::from(sp.temperature),
        top_p: f64::from(sp.top_p),
        top_k: sp.top_k,
        min_p: f64::from(sp.min_p),
        frequency_penalty: f64::from(sp.frequency_penalty),
        presence_penalty: f64::from(sp.presence_penalty),
        // proto3 cannot tell an unset float from 0.0; a zero penalty factor
        // is never meant and would zero every logit.
        repetition_penalty: if sp.repetition_penalty == 0.0 {
            1.0
        } else {
            f64::from(sp.repetition_penalty)
        },
        min_new_tokens: sp.min_new_tokens,
        n: sp.n.max(1),
        ignore_eos: sp.ignore_eos,
        skip_special_tokens: sp.skip_special_tokens,
        spaces_between_special_tokens: sp.spaces_between_special_tokens,
        no_stop_trim: sp.no_stop_trim,
        stream_interval: sp.stream_interval.and_then(|v| u32::try_from(v).ok()),
        logit_bias: (!sp.logit_bias.is_empty()).then(|| {
            sp.logit_bias
                .into_iter()
                .map(|(token, bias)| (token, f64::from(bias)))
                .collect()
        }),
        custom_params: sp.custom_params.map(struct_to_opaque),
        ..SglangSamplingParams::default()
    };
    match sp.constraint {
        Some(Constraint::JsonSchema(schema)) => params.json_schema = Some(schema),
        Some(Constraint::Regex(regex)) => params.regex = Some(regex),
        Some(Constraint::EbnfGrammar(grammar)) => params.ebnf = Some(grammar),
        Some(Constraint::StructuralTag(tag)) => params.structural_tag = Some(tag),
        None => {}
    }
    params
}

/// Normalize an SGLang finish reason into the canonical set the gateway's
/// response layer exact-matches (`stop`, `length`, `abort`).
pub(crate) fn normalize_finish_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "stop",
        "length" => "length",
        "abort" => "abort",
        other => {
            warn!(
                finish_reason = other,
                "unknown SGLang finish_reason; defaulting to \"stop\""
            );
            "stop"
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use engine_zmq_client::{
        codec::{decode_msgpack, encode_msgpack},
        mock_engine::{connect_to_frontend, default_ready_response},
        protocol::sglang::{
            output::BatchTokenIDSlimOutput,
            request::{SglangRequestType, TokenizedGenerateReqInput as WireRequest},
        },
        EngineId,
    };
    use openai_protocol::worker::RuntimeType;

    use super::*;
    use crate::{client::ZmqEngineClient, eos::EosTokenIds};

    #[test]
    fn sglang_sampling_translation_keeps_the_api_form_and_maps_the_constraint() {
        let params = translate_sampling_sglang(sglang_proto::SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: -1,
            repetition_penalty: 0.0, // proto3 default: never a real factor
            max_new_tokens: Some(16),
            stop: vec!["###".into()],
            stop_token_ids: vec![2],
            n: 0,
            constraint: Some(sglang_proto::sampling_params::Constraint::JsonSchema(
                "{}".into(),
            )),
            logit_bias: [("5".to_string(), 1.5f32)].into_iter().collect(),
            ..Default::default()
        });
        assert_eq!(params.max_new_tokens, Some(16));
        assert_eq!(
            params.stop.as_deref(),
            Some(&["###".to_string()][..]),
            "string stops reach the scheduler, which keeps its tokenizer"
        );
        assert_eq!(params.stop_token_ids, Some(vec![2]));
        assert_eq!(params.top_k, -1, "the scheduler resolves the API sentinel");
        assert_eq!(params.repetition_penalty, 1.0);
        assert_eq!(params.n, 1);
        assert_eq!(params.json_schema.as_deref(), Some("{}"));
        assert_eq!(params.logit_bias.unwrap()["5"], 1.5);
        assert!(!params.is_normalized);
    }

    #[test]
    fn sglang_request_translation_refuses_what_the_slim_wire_cannot_carry() {
        let base = sglang_proto::GenerateRequest {
            request_id: "r".into(),
            tokenized: Some(sglang_proto::TokenizedInput {
                input_ids: vec![1, 2],
                original_text: String::new(),
            }),
            ..Default::default()
        };
        assert!(translate_request_sglang(sglang_proto::GenerateRequest {
            tokenized: None,
            ..base.clone()
        })
        .is_err());
        assert_eq!(
            translate_request_sglang(sglang_proto::GenerateRequest {
                top_logprobs_num: 3,
                ..base.clone()
            })
            .unwrap()
            .top_logprobs_num,
            3
        );
        assert!(translate_request_sglang(sglang_proto::GenerateRequest {
            return_hidden_states: true,
            ..base.clone()
        })
        .is_err());
        assert!(translate_request_sglang(sglang_proto::GenerateRequest {
            token_ids_logprob: vec![5],
            ..base.clone()
        })
        .is_err());
        assert!(translate_request_sglang(sglang_proto::GenerateRequest {
            disaggregated_params: Some(sglang_proto::DisaggregatedParams::default()),
            ..base.clone()
        })
        .is_err());
        // Forwarded as the Python servicer forwards them.
        let forwarded = translate_request_sglang(sglang_proto::GenerateRequest {
            lora_id: "adapter".into(),
            custom_logit_processor: "proc".into(),
            sampling_params: Some(sglang_proto::SamplingParams {
                custom_params: Some(prost_types::Struct {
                    fields: [(
                        "k".to_string(),
                        prost_types::Value {
                            kind: Some(prost_types::value::Kind::NumberValue(2.0)),
                        },
                    )]
                    .into_iter()
                    .collect(),
                }),
                ..Default::default()
            }),
            ..base.clone()
        })
        .unwrap();
        assert_eq!(forwarded.lora_id.as_deref(), Some("adapter"));
        assert_eq!(forwarded.custom_logit_processor.as_deref(), Some("proc"));
        assert_eq!(
            forwarded.sampling_params.custom_params,
            Some(OpaqueValue::Map(vec![(
                OpaqueValue::from("k"),
                OpaqueValue::F64(2.0)
            )]))
        );
        // Prompt logprobs ride the wire now.
        let prompt = translate_request_sglang(sglang_proto::GenerateRequest {
            return_logprob: true,
            logprob_start_len: 0,
            top_logprobs_num: 2,
            ..base.clone()
        })
        .unwrap();
        assert!(prompt.return_logprob && prompt.logprob_start_len == 0);
        // Output logprobs only: `-1` is the builders' "no prompt logprobs"
        // (the proto default of 0 would ask for them from the first token).
        let ok = translate_request_sglang(sglang_proto::GenerateRequest {
            return_logprob: true,
            logprob_start_len: -1,
            stream: true,
            require_reasoning: true,
            ..base
        })
        .unwrap();
        assert_eq!(ok.rid, "r");
        assert!(ok.return_logprob && ok.stream && ok.require_reasoning);
    }

    #[test]
    fn sglang_fan_out_suffixes_rids_and_pins_n() {
        let subs = fan_out_sglang_requests(sglang_proto::GenerateRequest {
            request_id: "r".into(),
            sampling_params: Some(sglang_proto::SamplingParams {
                n: 3,
                ..Default::default()
            }),
            ..Default::default()
        });
        assert_eq!(
            subs.iter()
                .map(|s| s.request_id.as_str())
                .collect::<Vec<_>>(),
            ["r-0", "r-1", "r-2"]
        );
        assert!(subs
            .iter()
            .all(|s| s.sampling_params.as_ref().unwrap().n == 1));
    }

    /// End-to-end over ipc:// for an SGLang backend: the adapter frames the
    /// scheduler's own `TokenizedGenerateReqInput`, and maps slim batches
    /// back to vLLM-proto responses, matched stop included.
    #[tokio::test]
    async fn generate_e2e_translates_and_streams_sglang() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Sglang,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let engine = engine.expect("mock engine");

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let frames = input.recv_frames().await.unwrap();
            assert_eq!(
                SglangRequestType::from_frame(frames[0].as_ref()),
                Some(SglangRequestType::Add)
            );
            let request: WireRequest = decode_msgpack(frames[1].as_ref()).unwrap();
            assert_eq!(request.rid, "r1");
            assert_eq!(request.input_ids, vec![1, 2, 3]);
            assert_eq!(request.sampling_params.max_new_tokens, Some(2));
            assert!(request.return_logprob);

            let chunk = BatchTokenIDSlimOutput {
                rids: vec!["r1".into()],
                output_ids: vec![vec![10]],
                finished_reasons: vec![String::new()],
                finished_messages: vec![None],
                finished_matched: vec![None],
                prompt_tokens: vec![3],
                completion_tokens: vec![1],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-0.5]],
                output_token_logprobs_idx: vec![vec![10]],
                ..Default::default()
            };
            let done = BatchTokenIDSlimOutput {
                rids: vec!["r1".into()],
                output_ids: vec![vec![11]],
                finished_reasons: vec!["stop".into()],
                finished_messages: vec![None],
                finished_matched: vec![Some(MatchedStop::TokenId(11))],
                prompt_tokens: vec![3],
                completion_tokens: vec![2],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-1.25]],
                output_token_logprobs_idx: vec![vec![11]],
                engine_index: 0,
                num_running: 1,
                num_waiting: 0,
                kv_used_tokens: 5,
                kv_total_tokens: 100,
                finished_status: vec![None],
                output_top_logprobs_val: vec![vec![vec![-1.25, -2.0]]],
                output_top_logprobs_idx: vec![vec![vec![11, 12]]],
                ..Default::default()
            };
            for batch in [chunk, done] {
                output
                    .send_frames(vec![bytes::Bytes::from(encode_msgpack(&batch).unwrap())])
                    .await
                    .unwrap();
            }
        });

        let req = sglang_proto::GenerateRequest {
            request_id: "r1".to_string(),
            tokenized: Some(sglang_proto::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(sglang_proto::SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            return_logprob: true,
            logprob_start_len: -1,
            stream: true,
            ..Default::default()
        };
        let mut stream = client.generate_sglang(req).await.expect("generate");

        let first = stream.next().await.expect("chunk item").expect("chunk ok");
        match first.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![10]);
                assert_eq!(chunk.prompt_tokens, 3);
                assert_eq!(chunk.output_logprobs.unwrap().token_logprobs, vec![-0.5]);
            }
            other => panic!("expected a chunk, got {other:?}"),
        }
        let mut complete = None;
        while let Some(item) = stream.next().await {
            let response = item.expect("stream ok");
            if let Some(vllm::generate_response::Response::Complete(done)) = response.response {
                complete = Some(done);
            }
        }
        let done = complete.expect("a terminal Complete");
        assert_eq!(done.finish_reason, "stop");
        assert_eq!(done.output_ids, vec![10, 11]);
        assert_eq!(done.completion_tokens, 2);
        assert_eq!(
            done.matched_stop,
            Some(vllm::generate_complete::MatchedStop::MatchedTokenId(11))
        );
        let done_logprobs = done.output_logprobs.unwrap();
        assert_eq!(done_logprobs.token_logprobs, vec![-0.5, -1.25]);
        // Ranked candidates: only the second token asked for them, and the
        // terminal Complete carries the cumulative list.
        assert_eq!(done_logprobs.top_logprobs.len(), 1);
        assert_eq!(done_logprobs.top_logprobs[0].token_ids, vec![11, 12]);
        // The piggybacked load reached the connector: the KV term as reported,
        // the queue counts zeroed once the rank's last request retired (the
        // batch that carried them was sampled before that finish committed).
        let loads = client.get_loads();
        assert_eq!(loads.loads.len(), 1);
        assert_eq!(loads.loads[0].num_running_reqs, 0);
        assert_eq!(loads.loads[0].token_usage, 0.05);
        engine_task.await.unwrap();
    }

    #[test]
    fn sglang_finish_reason_normalizes() {
        assert_eq!(normalize_finish_reason("stop"), "stop");
        assert_eq!(normalize_finish_reason("length"), "length");
        assert_eq!(normalize_finish_reason("abort"), "abort");
        assert_eq!(normalize_finish_reason("mystery"), "stop");
    }

    /// `n = 2` over the SGLang wire: two single-sample sub-requests with
    /// suffixed rids, both finished in one slim batch, demuxed back into two
    /// indexed `Complete`s that each report the shared prompt in full.
    #[tokio::test]
    async fn generate_e2e_fans_out_n2_sglang() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Sglang,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let (client, engine) = (client.unwrap(), engine.unwrap());
        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let mut rids = Vec::new();
            for _ in 0..2 {
                let frames = input.recv_frames().await.unwrap();
                let request: WireRequest = decode_msgpack(frames[1].as_ref()).unwrap();
                assert_eq!(request.sampling_params.n, 1);
                rids.push(request.rid);
            }
            rids.sort();
            assert_eq!(rids, vec!["r1-0".to_string(), "r1-1".to_string()]);
            let done = BatchTokenIDSlimOutput {
                rids: vec!["r1-0".into(), "r1-1".into()],
                output_ids: vec![vec![10], vec![11]],
                finished_reasons: vec!["stop".into(), "stop".into()],
                finished_messages: vec![None, None],
                finished_matched: vec![None, None],
                prompt_tokens: vec![3, 3],
                completion_tokens: vec![1, 1],
                cached_tokens: vec![0, 0],
                output_token_logprobs_val: vec![vec![], vec![]],
                output_token_logprobs_idx: vec![vec![], vec![]],
                ..Default::default()
            };
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&done).unwrap())])
                .await
                .unwrap();
        });
        let req = sglang_proto::GenerateRequest {
            request_id: "r1".to_string(),
            tokenized: Some(sglang_proto::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(sglang_proto::SamplingParams {
                n: 2,
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut stream = client.generate_sglang(req).await.expect("generate");
        let mut completes = Vec::new();
        while let Some(item) = stream.next().await {
            if let Some(vllm::generate_response::Response::Complete(complete)) =
                item.expect("stream item").response
            {
                completes.push(complete);
            }
        }
        completes.sort_by_key(|complete| complete.index);
        assert_eq!(completes.len(), 2, "one Complete per fanned-out sub");
        assert_eq!(
            (completes[0].index, &completes[0].output_ids),
            (0, &vec![10])
        );
        assert_eq!(
            (completes[1].index, &completes[1].output_ids),
            (1, &vec![11])
        );
        assert!(completes.iter().all(|complete| complete.prompt_tokens == 3));
        engine_task.await.unwrap();
    }

    /// An `abort` finish is the scheduler telling the frontend it could not
    /// serve the request; it surfaces as an error with the reason, and the
    /// scheduler's 400 as the caller's fault.
    #[tokio::test]
    async fn sglang_abort_finish_is_an_error_with_the_scheduler_status() {
        let dir = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", dir.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));
        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Sglang,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let (client, engine) = (client.unwrap(), engine.unwrap());
        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            for (rid, status) in [("r2", Some(400)), ("r3", None)] {
                let _ = input.recv_frames().await.unwrap();
                let rejection = BatchTokenIDSlimOutput {
                    rids: vec![rid.into()],
                    output_ids: vec![vec![]],
                    finished_reasons: vec!["abort".into()],
                    finished_messages: vec![Some("input_ids are required on this wire".into())],
                    finished_matched: vec![None],
                    prompt_tokens: vec![0],
                    completion_tokens: vec![0],
                    cached_tokens: vec![0],
                    output_token_logprobs_val: vec![vec![]],
                    output_token_logprobs_idx: vec![vec![]],
                    finished_status: status.into_iter().map(Some).collect(),
                    ..Default::default()
                };
                output
                    .send_frames(vec![bytes::Bytes::from(
                        encode_msgpack(&rejection).unwrap(),
                    )])
                    .await
                    .unwrap();
            }
        });
        for (rid, code) in [
            ("r2", tonic::Code::InvalidArgument),
            ("r3", tonic::Code::Internal),
        ] {
            let req = sglang_proto::GenerateRequest {
                request_id: rid.to_string(),
                tokenized: Some(sglang_proto::TokenizedInput {
                    input_ids: vec![1],
                    original_text: String::new(),
                }),
                ..Default::default()
            };
            let mut stream = client.generate_sglang(req).await.expect("generate");
            let error = stream.next().await.expect("an item").expect_err("an error");
            assert_eq!(error.code(), code, "{rid}");
            assert!(
                error.message().contains("input_ids are required"),
                "{error}"
            );
        }
        engine_task.await.unwrap();
    }
}
