//! SGLang dialect: proto request → scheduler request translation and
//! scheduler output → vLLM-proto response mapping.
//!
//! The scheduler side is SGLang's own, unchanged; it joins SMG's transport
//! through the SGLang plugin in `smg_grpc_servicer.sglang` (which performs
//! the handshake inside the scheduler process, normalizes and verifies the
//! sampling params SMG sends, and slices each step's output to the slim
//! batch decoded here).

use std::collections::BTreeMap;

use engine_zmq_client::{
    codec::OpaqueValue,
    connector::SglangStream,
    protocol::sglang::{
        multimodal::{SglangWireMmInputs, SglangWireMmItem, SglangWireModality, SglangWireTensor},
        output::{MatchedStop, SglangOutput},
        request::{TokenizedEmbeddingReqInput, TokenizedGenerateReqInput},
        sampling::SamplingParams as SglangSamplingParams,
    },
};
use futures::Stream;
use smg_grpc_client::{constraint_failed_status, sglang_proto, vllm_proto as vllm};
use tracing::warn;

use crate::{
    context_window::admit_prompt,
    fanout::fan_out_n,
    stream::{poll_mapped, MappedGenerateStream, StreamState},
    structured_outputs::ConstraintRef,
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
    /// Selected candidate rows have no slot in the vLLM-proto intermediate.
    /// Retain both views for the SGLang response: deltas on chunks, cumulative
    /// rows on completion (including prefill-only scores with no output ids).
    selected_logprobs: Vec<sglang_proto::TopLogProbs>,
    selected_logprobs_delta: Vec<sglang_proto::TopLogProbs>,
    /// The final hidden layer's vectors the scheduler reported for this
    /// choice, when the request asked for them: the rows accumulated from
    /// what each tick added (the plugin sends only the new rows: the prompt
    /// positions the prefill computed, then a decode step's rows), and how
    /// many of them the chunks before the current tick carried. The
    /// vLLM-proto intermediate has no slot for them either: the chunk carries
    /// the tick's rows, the completion the whole list with absolute positions.
    hidden_states: Vec<Vec<f32>>,
    hidden_states_sent: usize,
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
            selected_logprobs: Vec::new(),
            selected_logprobs_delta: Vec::new(),
            hidden_states: Vec::new(),
            hidden_states_sent: 0,
        }
    }

    /// Reasoning tokens the scheduler has counted for this choice so far.
    pub fn reasoning_tokens(&self) -> u32 {
        self.reasoning_tokens
    }

    /// The engine-side id of this choice's request.
    pub fn request_id(&self) -> &str {
        self.inner.request_id()
    }

    /// The data-parallel rank this choice's request went to; `None` for an
    /// engine whose identity names no index.
    pub fn engine_index(&self) -> Option<u32> {
        self.inner.engine_index()
    }

    /// Convert a stream item without discarding the SGLang-only rows: the
    /// selected scores and the hidden states.
    pub fn to_sglang_response(
        &self,
        request_id: &str,
        response: vllm::GenerateResponse,
    ) -> sglang_proto::GenerateResponse {
        use sglang_proto::generate_response::Response;
        let mut response = to_sglang_response(request_id, response, self.reasoning_tokens);
        let (logprobs, selected) = match response.response.as_mut() {
            Some(Response::Chunk(chunk)) => {
                (&mut chunk.output_logprobs, &self.selected_logprobs_delta)
            }
            Some(Response::Complete(complete)) => {
                (&mut complete.output_logprobs, &self.selected_logprobs)
            }
            None => return response,
        };
        if !selected.is_empty() {
            logprobs
                .get_or_insert_with(Default::default)
                .token_ids_logprobs
                .clone_from(selected);
        }
        match response.response.as_mut() {
            Some(Response::Chunk(chunk)) => {
                chunk.hidden_states =
                    flat_hidden_states(&self.hidden_states[self.hidden_states_sent..]);
            }
            Some(Response::Complete(complete)) => {
                complete.all_hidden_states = all_hidden_states(
                    &self.hidden_states,
                    first_hidden_position(
                        self.state.prompt_tokens,
                        self.state.completion_tokens,
                        self.hidden_states.len(),
                    ),
                );
            }
            None => {}
        }
        response
    }

    /// Whether the engine already finished this choice and its terminal
    /// `Complete` is parked behind the chunk just yielded (the finish tick
    /// carried new tokens).
    pub fn has_parked_complete(&self) -> bool {
        self.pending.is_some()
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
        self.state.attach_input_logprobs(
            &mut self.pending,
            &mut self.input_logprobs_emitted,
            &mut response,
        );
        response
    }
}

/// A vLLM-proto response of an SGLang stream as the SGLang proto the
/// `SglangScheduler` service answers with, under `request_id` (the vLLM proto
/// carries none) and the stream's `reasoning_tokens` (which it has no slot
/// for). The two share the chunk and completion fields this wire fills; the
/// hidden states, which the intermediate has no slot for either, are the
/// stream's own (the method above); what the wire does not carry (speculative
/// counters) stays at the proto defaults, as it does on the Python servicer.
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
            token_ids_logprobs: Vec::new(),
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

/// The layer `GenerateComplete.all_hidden_states` names: the model's final
/// hidden layer, the only one SGLang returns.
const FINAL_HIDDEN_LAYER: i32 = -1;

/// `GenerateStreamChunk.hidden_states`: the rows a tick added, flattened (the
/// client splits them by the hidden size).
fn flat_hidden_states(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

/// The absolute position of the first row a request carries, from the
/// scheduler's own counters: the last row is always the last fed-back token
/// (position `prompt + generated - 2`), whatever prefix the radix cache
/// served without rows, so the first is that position less the rows after
/// it.
fn first_hidden_position(prompt_tokens: u32, completion_tokens: u32, rows: usize) -> i32 {
    let first = i64::from(prompt_tokens) + i64::from(completion_tokens) - 1 - rows as i64;
    i32::try_from(first.max(0)).unwrap_or(i32::MAX)
}

/// `GenerateComplete.all_hidden_states`: one entry per position a forward
/// ran on (the prompt positions the prefill computed, as the engine returns
/// them, then each generated token fed back for the next; the last sampled
/// token has no row), the final hidden layer, the positions absolute from
/// `first_position` ([`first_hidden_position`]).
fn all_hidden_states(rows: &[Vec<f32>], first_position: i32) -> Vec<sglang_proto::HiddenStates> {
    rows.iter()
        .enumerate()
        .map(|(position, values)| sglang_proto::HiddenStates {
            values: values.clone(),
            layer: FINAL_HIDDEN_LAYER,
            position: first_position.saturating_add(i32::try_from(position).unwrap_or(i32::MAX)),
        })
        .collect()
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
        if let Some(error) = output.selected_logprobs_error {
            return Err(tonic::Status::internal(error));
        }
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
        // The plugin sends the rows the tick added (the prompt positions the
        // prefill computed, then a decode step's rows): the chunk carries
        // them, the completion the whole list.
        self.hidden_states_sent = self.hidden_states.len();
        self.hidden_states.extend(output.output_hidden_states);
        self.selected_logprobs_delta = output
            .output_token_ids_logprobs_val
            .iter()
            .zip(&output.output_token_ids_logprobs_idx)
            .map(|(values, token_ids)| sglang_proto::TopLogProbs {
                values: values.iter().map(|&value| value as f32).collect(),
                token_ids: token_ids.clone(),
            })
            .collect();
        self.selected_logprobs
            .extend(self.selected_logprobs_delta.iter().cloned());
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
        // from a full queue) ends as `abort` with the reason and an HTTP-style
        // status (400 when the request is at fault). The frontend never asked
        // for it (its own aborts drop the stream first), so it is an error to
        // the caller, not an empty success, with the status mapped by the one
        // table the engine's front ends share (`scheduler_abort_status`). The
        // one abort the scheduler itself attributes to the request's grammar
        // (the plugin's `err_type` stamp on the scheduler's own grammar
        // failure, see `GRAMMAR_REJECTED_ERR_TYPE`) is the constraint failing
        // the request: the constraint-failure status every wire ends such a
        // request with, which a router answers for the constraint and never
        // replays. Every other abort, a status-less one included (the
        // scheduler aborts every running request the same bare way on an
        // abort-all), keeps the table's status: the wire does not say more.
        if output.finish_reason.as_deref() == Some("abort") {
            let message = format!(
                "the scheduler aborted the request: {}",
                output
                    .finish_message
                    .as_deref()
                    .unwrap_or("no reason reported")
            );
            if output.finish_err_type.as_deref() == Some(GRAMMAR_REJECTED_ERR_TYPE) {
                return Err(constraint_failed_status(message));
            }
            return Err(scheduler_abort_status(output.finish_status, message));
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
        self.state.attach_input_logprobs(
            &mut self.pending,
            &mut self.input_logprobs_emitted,
            &mut response,
        );
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
/// with their ranked and selected candidates, and the hidden states a request
/// asks for. Multimodal inputs become the scheduler's processor output
/// ([`translate_sglang_multimodal`]), and a disaggregated request's
/// bootstrap fields ride the scheduler's own slots, as the Python servicer
/// forwards them. `max_model_len` is the context window the scheduler
/// reported at the handshake: an empty prompt, or one that fills the window,
/// is refused before it reaches the scheduler ([`admit_prompt`]). `ranks` is
/// how many scheduler ranks are connected: the request's data-parallel pin is
/// placed among them as the engine's own frontend places it
/// ([`routed_dp_rank`]).
pub(crate) fn translate_request_sglang(
    req: sglang_proto::GenerateRequest,
    max_model_len: u64,
    ranks: usize,
) -> Result<TokenizedGenerateReqInput, String> {
    let tokenized = req
        .tokenized
        .ok_or_else(|| "ZMQ mode requires pre-tokenized input; no input provided".to_string())?;
    let input_ids = tokenized.input_ids;
    // The scheduler's own bound on the prompt applies only after admission
    // (an abort with its reason; a cut down to the bound for a scheduler
    // started with `--allow-auto-truncate`, whose servicer then passes no
    // window here) and nothing of its own refuses an empty prompt. An unset
    // `max_new_tokens` stays unset: the scheduler caps it at the remaining
    // context itself.
    admit_prompt(input_ids.len(), max_model_len)?;
    // Media by reference (the encoders' input on a language-side worker)
    // leaves the processor-output slot empty and forwards the prompt text:
    // the scheduler's receiver rebuilds the prompt from it once the
    // embeddings arrive, as behind the engine's own frontend, which sends the
    // text it rendered. Preprocessed tensors take the processor-output path.
    let (mm_inputs, input_text) = match req.mm_inputs {
        Some(mm) if encode_media_refs(&mm)?.is_some() => (
            None,
            (!tokenized.original_text.is_empty()).then_some(tokenized.original_text),
        ),
        Some(mm) => (
            Some(translate_sglang_multimodal(mm, input_ids.len())?),
            None,
        ),
        None => (None, None),
    };
    // The prefill/decode rendezvous (the prefill side's bootstrap server and
    // the room both sides meet in), forwarded as the Python servicer forwards
    // it: an empty host and a zero port are unset, the room is kept as given
    // (0 is a room: the fake-transfer probe's) whenever the message is
    // present, and an absent message leaves the scheduler's defaults.
    let (bootstrap_host, bootstrap_port, bootstrap_room) = match req.disaggregated_params {
        Some(params) => (
            (!params.bootstrap_host.is_empty()).then_some(params.bootstrap_host),
            (params.bootstrap_port != 0).then_some(params.bootstrap_port),
            Some(i64::from(params.bootstrap_room)),
        ),
        None => (None, None, None),
    };
    let routed_dp_rank = routed_dp_rank(req.data_parallel_rank, ranks)?;
    Ok(TokenizedGenerateReqInput {
        rid: req.request_id,
        input_text,
        input_ids,
        mm_inputs,
        sampling_params: req
            .sampling_params
            .map(translate_sampling_sglang)
            .unwrap_or_default(),
        return_logprob: req.return_logprob,
        logprob_start_len: req.logprob_start_len,
        top_logprobs_num: u32::try_from(req.top_logprobs_num).unwrap_or(0),
        token_ids_logprob: (!req.token_ids_logprob.is_empty()).then_some(req.token_ids_logprob),
        stream: req.stream,
        // Full capture, the one mode the contract's bool can ask for; the
        // servicer keeps the scheduler's launch-time cap ahead of this.
        return_hidden_states: req.return_hidden_states,
        // Forwarded as the Python servicer forwards them: the scheduler
        // resolves the adapter id and runs the processor it was started with.
        lora_id: (!req.lora_id.is_empty()).then_some(req.lora_id),
        custom_logit_processor: (!req.custom_logit_processor.is_empty())
            .then_some(req.custom_logit_processor),
        bootstrap_host,
        bootstrap_port,
        bootstrap_room,
        routed_dp_rank,
        require_reasoning: req.require_reasoning,
        ..TokenizedGenerateReqInput::default()
    })
}

/// The image references a payload names for the encoders of an
/// encode/prefill-disaggregated deployment, in prompt order: `Some` for a
/// payload of references without preprocessed tensors (the encoders' input
/// on a language-side worker, see [`crate::sglang_encode`]), `None` for one
/// carrying tensors (the processor-output path, whatever references ride
/// beside them: the engine's frontend processes such a request locally) or
/// nothing at all. Video and audio references are refused by name: the
/// encoders' gRPC service takes images only.
pub(crate) fn encode_media_refs(
    mm: &sglang_proto::MultimodalInputs,
) -> Result<Option<Vec<String>>, String> {
    if mm.pixel_values.is_some() {
        return Ok(None);
    }
    if !mm.video_urls.is_empty() || !mm.audio_urls.is_empty() {
        return Err(
            "multimodal payload carries video or audio references; the encoders of an \
             encode/prefill-disaggregated deployment take image references only"
                .to_string(),
        );
    }
    Ok((!mm.image_urls.is_empty()).then(|| mm.image_urls.clone()))
}

/// The request's multimodal payload as the scheduler's processor output, by
/// the Python servicer's rules (`_parse_mm_inputs`): one image item whose
/// feature is `pixel_values`, the model-specific tensors beside it, the
/// placeholder ranges as inclusive `[start, end]` offsets (none at all leaves
/// the slot unset, with the same warning), and the image token id. Nothing
/// is derived here: the scheduler hashes the feature, pads the prompt with
/// its model's pattern and computes the position tensors it needs, as it
/// does behind the Python servicer. What that servicer would mis-tag or
/// silently drop is refused instead: a payload naming another modality
/// (every item is tagged an image), one of media references without tensors
/// (the encoders' input, which [`encode_media_refs`] takes before this and
/// the dispatch serves; refused here on a worker that runs none), a tensor
/// of a dtype the wire does not carry or of the wrong length, and a
/// placeholder outside the prompt or of zero length.
pub(crate) fn translate_sglang_multimodal(
    mm: sglang_proto::MultimodalInputs,
    prompt_len: usize,
) -> Result<SglangWireMmInputs, String> {
    if let Some(other) = mm.modalities.iter().find(|modality| *modality != "image") {
        return Err(format!(
            "multimodal modality {other:?} is not carried over the SGLang ZMQ backend; the \
             scheduler takes preprocessed image tensors on this wire"
        ));
    }
    let Some(pixel_values) = mm.pixel_values else {
        return Err(
            if mm.image_urls.is_empty() && mm.video_urls.is_empty() && mm.audio_urls.is_empty() {
                "multimodal payload carries no pixel_values".to_string()
            } else {
                "multimodal payload carries media references but no preprocessed tensors; the \
                 scheduler's encode path is not served over the SGLang ZMQ backend"
                    .to_string()
            },
        );
    };
    let feature = wire_tensor_sglang("pixel_values", pixel_values)?;
    let mut model_specific_data = BTreeMap::new();
    for (key, tensor) in mm.model_specific_tensors {
        let tensor = wire_tensor_sglang(&key, tensor)?;
        model_specific_data.insert(key, tensor);
    }
    let mut offsets = Vec::with_capacity(mm.mm_placeholders.len());
    for range in &mm.mm_placeholders {
        let (start, length) = (u64::from(range.offset), u64::from(range.length));
        if length == 0 {
            return Err(format!(
                "multimodal placeholder at offset {start} has zero length"
            ));
        }
        let end = start + length - 1;
        if end >= prompt_len as u64 {
            return Err(format!(
                "multimodal placeholder {start}..={end} lies outside the prompt of {prompt_len} \
                 tokens"
            ));
        }
        offsets.push((start, end));
    }
    let offsets = if offsets.is_empty() {
        // The Python servicer's warning, word for word in substance.
        warn!(
            "multimodal payload carries no mm_placeholders: the prompt's placeholder expansion \
             may have missed the image token; check that the placeholder token id matches the \
             tokenized image token"
        );
        None
    } else {
        Some(offsets)
    };
    Ok(SglangWireMmInputs {
        mm_items: vec![SglangWireMmItem {
            modality: SglangWireModality::Image,
            offsets,
            feature: Some(feature),
            model_specific_data,
        }],
        im_token_id: mm.im_token_id,
    })
}

/// A proto tensor as the wire carries it: the bytes are viewed, not copied;
/// the dtype name and the length are checked as the Python servicer's reader
/// checks them.
fn wire_tensor_sglang(
    name: &str,
    tensor: sglang_proto::TensorData,
) -> Result<SglangWireTensor, String> {
    SglangWireTensor::new(
        tensor.dtype,
        tensor.shape.into_iter().map(u64::from).collect(),
        tensor.data,
    )
    .map_err(|error| format!("multimodal tensor {name:?}: {error}"))
}

/// The scheduler rank a request's data-parallel pin routes it to, by the
/// rule of SGLang's own frontend (`TokenizerManager`, where the API's pin
/// becomes the request's `routed_dp_rank`): no pin leaves the placement to
/// the dispatcher (here the connector's least-loaded selection); on an
/// engine of one rank any non-negative pin is ignored, as the frontend logs
/// and ignores it; a pin among several ranks must name one of them, else it
/// is refused with the frontend's 400 wording. Each scheduler rank dials the connector with its
/// rank as its engine index, so the rank is the engine the request goes to.
fn routed_dp_rank(pin: Option<i32>, ranks: usize) -> Result<Option<u32>, String> {
    let Some(pin) = pin else {
        return Ok(None);
    };
    let out_of_range = || format!("routed_dp_rank={pin} out of range [0, {ranks})");
    let rank = u32::try_from(pin).map_err(|_| out_of_range())?;
    if ranks <= 1 {
        return Ok(None);
    }
    if usize::try_from(rank).is_ok_and(|rank| rank < ranks) {
        Ok(Some(rank))
    } else {
        Err(out_of_range())
    }
}

/// Translate an SGLang proto `EmbedRequest` into the scheduler's
/// `TokenizedEmbeddingReqInput`, as the Python servicer does: the tokenized
/// prompt with `max_new_tokens = 0` (the scheduler pools, nothing samples),
/// token type ids for cross-encoder inputs. A multimodal payload is refused
/// (the Python servicer drops it without a word, which no embedding model it
/// serves depends on).
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
            "multimodal inputs on an embedding request are not served over the SGLang ZMQ \
             backend"
                .to_string(),
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

/// The request's constraint as the shared refusal reads it
/// ([`crate::structured_outputs::refuse_unenforceable_constraint`]): the
/// proto `constraint` one-of by kind, with its text.
pub(crate) fn constraint_ref(sp: &sglang_proto::SamplingParams) -> Option<ConstraintRef<'_>> {
    use sglang_proto::sampling_params::Constraint;
    Some(match sp.constraint.as_ref()? {
        Constraint::JsonSchema(schema) => ConstraintRef::JsonSchema(schema),
        Constraint::Regex(regex) => ConstraintRef::Regex(regex),
        Constraint::EbnfGrammar(grammar) => ConstraintRef::Ebnf(grammar),
        Constraint::StructuralTag(tag) => ConstraintRef::StructuralTag(tag),
    })
}

/// The `err_type` the plugin stamps on the scheduler's abort of a request
/// whose grammar rejected a sampled token (`req.grammar.accept_token` raised
/// in the scheduler's own grammar paths; the scheduler leaves that finish a
/// bare `FINISH_ABORT()` that nothing on the wire tells from an abort-all).
/// One name on both sides of the wire: the plugin's `GRAMMAR_REJECTED_ERR_TYPE`
/// (`smg_grpc_servicer.sglang`) stamps it, this wire reads it.
pub(crate) const GRAMMAR_REJECTED_ERR_TYPE: &str = "GrammarRejectedToken";

/// The gRPC status for a request the scheduler aborted with an HTTP-style
/// status: the table the engine's Python servicer uses for the same status,
/// so the two front ends of one engine answer alike. 400 is the caller's
/// fault, 404 a missing resource, 408 and 504 a deadline, 429 a full queue,
/// 503 an unavailable engine; anything else, a missing status included, is
/// internal.
pub(crate) fn scheduler_abort_status(status: Option<u16>, message: String) -> tonic::Status {
    match status {
        Some(400) => tonic::Status::invalid_argument(message),
        Some(404) => tonic::Status::not_found(message),
        Some(408 | 504) => tonic::Status::deadline_exceeded(message),
        Some(429) => tonic::Status::resource_exhausted(message),
        Some(503) => tonic::Status::unavailable(message),
        _ => tonic::Status::internal(message),
    }
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
    use engine_zmq_client::{
        codec::{decode_msgpack, encode_msgpack},
        mock_engine::default_ready_response,
        protocol::sglang::{
            output::BatchTokenIDSlimOutput,
            request::{SglangRequestType, TokenizedGenerateReqInput as WireRequest},
        },
    };
    use openai_protocol::worker::RuntimeType;

    use super::*;
    use crate::test_support::connected_pair;

    /// The translation under the mock scheduler's context window, one rank.
    fn translate(req: sglang_proto::GenerateRequest) -> Result<WireRequest, String> {
        translate_request_sglang(req, default_ready_response().max_model_len, 1)
    }

    fn tensor(dtype: &str, shape: Vec<u32>, data: Vec<u8>) -> sglang_proto::TensorData {
        sglang_proto::TensorData {
            data: data.into(),
            shape,
            dtype: dtype.to_string(),
        }
    }

    /// The payload the gateway assembles for this wire: the pixel tensor,
    /// one grid tensor, the placeholder ranges and the image token.
    fn media_payload() -> sglang_proto::MultimodalInputs {
        let pixels: Vec<u8> = (0..8)
            .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
            .collect();
        let grid: Vec<u8> = [1i64, 2, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        sglang_proto::MultimodalInputs {
            modalities: vec!["image".to_string()],
            pixel_values: Some(tensor("float32", vec![2, 4], pixels)),
            model_specific_tensors: [(
                "image_grid_thw".to_string(),
                tensor("int64", vec![1, 3], grid),
            )]
            .into_iter()
            .collect(),
            im_token_id: Some(7),
            mm_placeholders: vec![sglang_proto::PlaceholderRange {
                offset: 1,
                length: 3,
            }],
            ..Default::default()
        }
    }

    fn media_request(mm: sglang_proto::MultimodalInputs) -> sglang_proto::GenerateRequest {
        sglang_proto::GenerateRequest {
            request_id: "mm".into(),
            tokenized: Some(sglang_proto::TokenizedInput {
                input_ids: vec![1, 7, 7, 7, 2],
                original_text: String::new(),
            }),
            mm_inputs: Some(mm),
            ..Default::default()
        }
    }

    /// The gateway's preprocessed media becomes the scheduler's processor
    /// output by the Python servicer's rules: one image item, the pixel
    /// tensor as its feature (bytes viewed, not copied), the grid beside it,
    /// inclusive offsets, the image token id; the hash and pad slots are left
    /// to the scheduler.
    #[test]
    fn multimodal_inputs_become_the_schedulers_processor_output() {
        let payload = media_payload();
        let pixel_ptr = payload.pixel_values.as_ref().unwrap().data.as_ptr();
        let request = translate(media_request(payload)).unwrap();
        let media = request.mm_inputs.expect("the processor output");
        assert_eq!(media.im_token_id, Some(7));
        assert_eq!(media.mm_items.len(), 1);
        let item = &media.mm_items[0];
        assert_eq!(item.modality, SglangWireModality::Image);
        assert_eq!(item.offsets, Some(vec![(1, 3)]));
        let feature = item.feature.as_ref().unwrap();
        assert_eq!(feature.dtype, "float32");
        assert_eq!(feature.shape, vec![2, 4]);
        assert_eq!(
            feature.data.as_ptr(),
            pixel_ptr,
            "the proto bytes are viewed"
        );
        let grid = &item.model_specific_data["image_grid_thw"];
        assert_eq!(
            (grid.dtype.as_str(), grid.shape.as_slice()),
            ("int64", &[1, 3][..])
        );
        // The prompt is handed over as it came: the scheduler pads it.
        assert_eq!(request.input_ids, vec![1, 7, 7, 7, 2]);
        // No placeholders: the slot is left unset, as the Python servicer
        // leaves it (with its warning).
        let mut bare = media_payload();
        bare.mm_placeholders.clear();
        let request = translate(media_request(bare)).unwrap();
        assert_eq!(request.mm_inputs.unwrap().mm_items[0].offsets, None);
        // A text request carries none.
        let mut text = media_request(media_payload());
        text.mm_inputs = None;
        assert_eq!(translate(text).unwrap().mm_inputs, None);
    }

    /// What the Python servicer would mis-tag or drop is refused by name:
    /// another modality, references without tensors, a dtype the wire does
    /// not carry, a wrong byte length, a placeholder outside the prompt or
    /// of zero length; and a media payload on an embedding request.
    #[test]
    fn malformed_multimodal_inputs_are_refused_before_the_scheduler() {
        let refused = |mm: sglang_proto::MultimodalInputs, needle: &str| {
            let error = translate(media_request(mm)).unwrap_err();
            assert!(error.contains(needle), "{error}");
        };
        let mut video = media_payload();
        video.modalities = vec!["video".to_string()];
        refused(video, "modality \"video\" is not carried");
        // References without tensors are the encoders' input, taken before
        // this translation by `encode_media_refs`; handed to the
        // processor-output path anyway, they are refused by name.
        let mut references = media_payload();
        references.pixel_values = None;
        references.image_urls = vec!["http://example.test/a.png".to_string()];
        let error = translate_sglang_multimodal(references, 5).unwrap_err();
        assert!(
            error.contains("media references but no preprocessed tensors"),
            "{error}"
        );
        let mut empty = media_payload();
        empty.pixel_values = None;
        refused(empty, "carries no pixel_values");
        let mut wide = media_payload();
        wide.pixel_values = Some(tensor("float64", vec![1], vec![0; 8]));
        refused(
            wide,
            "multimodal tensor \"pixel_values\": unsupported tensor dtype \"float64\"",
        );
        let mut short = media_payload();
        short.model_specific_tensors.insert(
            "image_grid_thw".to_string(),
            tensor("int64", vec![1, 3], vec![0; 8]),
        );
        refused(
            short,
            "multimodal tensor \"image_grid_thw\": TensorData byte length mismatch",
        );
        let mut outside = media_payload();
        outside.mm_placeholders = vec![sglang_proto::PlaceholderRange {
            offset: 3,
            length: 3,
        }];
        refused(
            outside,
            "placeholder 3..=5 lies outside the prompt of 5 tokens",
        );
        let mut zero = media_payload();
        zero.mm_placeholders = vec![sglang_proto::PlaceholderRange {
            offset: 1,
            length: 0,
        }];
        refused(zero, "placeholder at offset 1 has zero length");
        let embed = sglang_proto::EmbedRequest {
            request_id: "e".into(),
            tokenized: Some(sglang_proto::TokenizedInput {
                input_ids: vec![1, 2],
                original_text: String::new(),
            }),
            mm_inputs: Some(media_payload()),
            ..Default::default()
        };
        let error = translate_embed_request_sglang(embed).unwrap_err();
        assert!(error.contains("embedding request"), "{error}");
    }

    /// References without tensors are the encoders' input on a language-side
    /// worker, not the scheduler's processor output: the translation leaves
    /// the media slot empty, carries the unexpanded prompt for the
    /// scheduler's rebuild once the embeddings land, and sets none of the
    /// dispatch's slots (the dispatch does, once it ran). Tensors beside
    /// references take the processor-output path; video and audio references
    /// are refused: the encoders take images only.
    #[test]
    fn reference_only_media_is_the_encoders_input_not_the_schedulers() {
        let mut references = media_payload();
        references.pixel_values = None;
        references.image_urls = vec!["http://example.test/a.png".to_string()];
        assert_eq!(
            encode_media_refs(&references).unwrap(),
            Some(vec!["http://example.test/a.png".to_string()])
        );
        let mut request = media_request(references.clone());
        request.tokenized.as_mut().unwrap().original_text =
            "<|vision_start|><|image_pad|><|vision_end|>What is this?".to_string();
        let translated = translate(request).unwrap();
        assert_eq!(translated.mm_inputs, None);
        assert_eq!(
            translated.input_text.as_deref(),
            Some("<|vision_start|><|image_pad|><|vision_end|>What is this?")
        );
        assert_eq!(translated.input_ids, vec![1, 7, 7, 7, 2]);
        assert_eq!(
            (
                translated.need_wait_for_mm_inputs,
                translated.num_items_assigned.clone(),
                translated.encoder_urls.clone(),
            ),
            (None, None, None)
        );
        // Without the prompt text the slot stays empty: nothing to rebuild from.
        assert_eq!(
            translate(media_request(references)).unwrap().input_text,
            None
        );
        // Tensors win: references beside them are the frontend's own record.
        let mut both = media_payload();
        both.image_urls = vec!["http://example.test/a.png".to_string()];
        assert_eq!(encode_media_refs(&both).unwrap(), None);
        let translated = translate(media_request(both)).unwrap();
        assert!(translated.mm_inputs.is_some() && translated.input_text.is_none());
        // Nothing at all is nothing to dispatch.
        let mut empty = media_payload();
        empty.pixel_values = None;
        assert_eq!(encode_media_refs(&empty).unwrap(), None);
        let mut video = media_payload();
        video.pixel_values = None;
        video.video_urls = vec!["http://example.test/a.mp4".to_string()];
        let mut audio = media_payload();
        audio.pixel_values = None;
        audio.audio_urls = vec!["http://example.test/a.wav".to_string()];
        for refs in [video, audio] {
            let error = encode_media_refs(&refs).unwrap_err();
            assert!(error.contains("image references only"), "{error}");
            let error = translate(media_request(refs)).unwrap_err();
            assert!(error.contains("image references only"), "{error}");
        }
    }

    #[test]
    fn a_scheduler_abort_carries_the_status_its_http_code_means() {
        let rows = [
            (Some(400), tonic::Code::InvalidArgument),
            (Some(404), tonic::Code::NotFound),
            (Some(408), tonic::Code::DeadlineExceeded),
            (Some(429), tonic::Code::ResourceExhausted),
            (Some(500), tonic::Code::Internal),
            (Some(503), tonic::Code::Unavailable),
            (Some(504), tonic::Code::DeadlineExceeded),
            (Some(418), tonic::Code::Internal),
            (None, tonic::Code::Internal),
        ];
        for (status, code) in rows {
            let got = scheduler_abort_status(status, "why".to_string());
            assert_eq!(got.code(), code, "{status:?}");
            assert_eq!(got.message(), "why");
        }
    }

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
        assert!(params.stop_regex.is_none() && params.custom_params.is_none());
    }

    /// The rows become the proto's shapes: a tick's rows flattened on the
    /// chunk, one `HiddenStates` per position on the completion, the final
    /// layer, the positions absolute from the anchor the scheduler's counters
    /// give (a prefix the radix cache served has no rows).
    #[test]
    fn hidden_state_rows_map_to_the_chunk_and_completion_shapes() {
        let rows = vec![vec![0.5, -0.5], vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(flat_hidden_states(&rows[1..]), vec![1.0, 2.0, 3.0, 4.0]);
        assert!(flat_hidden_states(&[]).is_empty());
        let all = all_hidden_states(&rows, 0);
        assert_eq!(all.len(), 3);
        assert_eq!(all[2].values, vec![3.0, 4.0]);
        assert!(all
            .iter()
            .enumerate()
            .all(|(i, hs)| hs.layer == -1 && hs.position == i32::try_from(i).unwrap()));
        let anchored = all_hidden_states(&rows, 3);
        assert_eq!(
            anchored.iter().map(|hs| hs.position).collect::<Vec<_>>(),
            vec![3, 4, 5]
        );
        // Three prompt rows and one fed-back token: the first row is position
        // 0; a prompt of five with three served from the cache: position 3.
        assert_eq!(first_hidden_position(3, 2, 4), 0);
        assert_eq!(first_hidden_position(5, 2, 3), 3);
        assert_eq!(first_hidden_position(0, 0, 0), 0);
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
        assert!(translate(sglang_proto::GenerateRequest {
            tokenized: None,
            ..base.clone()
        })
        .is_err());
        assert_eq!(
            translate(sglang_proto::GenerateRequest {
                top_logprobs_num: 3,
                ..base.clone()
            })
            .unwrap()
            .top_logprobs_num,
            3
        );
        assert!(
            translate(sglang_proto::GenerateRequest {
                return_hidden_states: true,
                ..base.clone()
            })
            .unwrap()
            .return_hidden_states,
            "hidden states ride the wire"
        );
        // Forwarded as the Python servicer forwards them.
        let forwarded = translate(sglang_proto::GenerateRequest {
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
        let prompt = translate(sglang_proto::GenerateRequest {
            return_logprob: true,
            logprob_start_len: 0,
            top_logprobs_num: 2,
            ..base.clone()
        })
        .unwrap();
        assert!(prompt.return_logprob && prompt.logprob_start_len == 0);
        // Output logprobs only: `-1` is the builders' "no prompt logprobs"
        // (the proto default of 0 would ask for them from the first token).
        let ok = translate(sglang_proto::GenerateRequest {
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

    /// A disaggregated request's bootstrap fields reach the scheduler's own
    /// slots with the Python servicer's rules: an empty host and a zero port
    /// are unset, the room is kept as given (0 included) whenever the message
    /// is present, and no message leaves all three unset.
    #[test]
    fn sglang_translation_forwards_the_pd_bootstrap_as_the_python_servicer_does() {
        let base = sglang_proto::GenerateRequest {
            request_id: "pd".into(),
            tokenized: Some(sglang_proto::TokenizedInput {
                input_ids: vec![1, 2],
                original_text: String::new(),
            }),
            ..Default::default()
        };
        let rendezvous = translate(sglang_proto::GenerateRequest {
            disaggregated_params: Some(sglang_proto::DisaggregatedParams {
                bootstrap_host: "10.0.0.7".into(),
                bootstrap_port: 8998,
                bootstrap_room: 41,
            }),
            ..base.clone()
        })
        .unwrap();
        assert_eq!(rendezvous.bootstrap_host.as_deref(), Some("10.0.0.7"));
        assert_eq!(rendezvous.bootstrap_port, Some(8998));
        assert_eq!(rendezvous.bootstrap_room, Some(41));
        // The fake-transfer probe: a host, room 0, no port.
        let probe = translate(sglang_proto::GenerateRequest {
            disaggregated_params: Some(sglang_proto::DisaggregatedParams {
                bootstrap_host: "2.2.2.2".into(),
                bootstrap_port: 0,
                bootstrap_room: 0,
            }),
            ..base.clone()
        })
        .unwrap();
        assert_eq!(probe.bootstrap_host.as_deref(), Some("2.2.2.2"));
        assert_eq!(probe.bootstrap_port, None);
        assert_eq!(probe.bootstrap_room, Some(0));
        // An empty message is present: the room is 0, the rest unset.
        let empty = translate(sglang_proto::GenerateRequest {
            disaggregated_params: Some(sglang_proto::DisaggregatedParams::default()),
            ..base.clone()
        })
        .unwrap();
        assert_eq!(
            (
                empty.bootstrap_host,
                empty.bootstrap_port,
                empty.bootstrap_room
            ),
            (None, None, Some(0))
        );
        let plain = translate(base).unwrap();
        assert_eq!(
            (
                plain.bootstrap_host,
                plain.bootstrap_port,
                plain.bootstrap_room
            ),
            (None, None, None)
        );
    }

    /// The data-parallel pin is placed as the engine's own frontend places
    /// it: no pin stays unrouted, a pin names a connected rank, a pin on an
    /// engine of one rank is ignored, and a pin past the connected ranks (or
    /// a negative one) is refused with the frontend's 400 wording.
    #[test]
    fn sglang_translation_routes_the_data_parallel_pin_as_the_engine_frontend_does() {
        let pinned = |pin: Option<i32>| sglang_proto::GenerateRequest {
            request_id: "dp".into(),
            tokenized: Some(sglang_proto::TokenizedInput {
                original_text: String::new(),
                input_ids: vec![1, 2, 3],
            }),
            data_parallel_rank: pin,
            ..Default::default()
        };
        let translate_among = |pin: Option<i32>, ranks: usize| {
            translate_request_sglang(pinned(pin), default_ready_response().max_model_len, ranks)
        };
        assert_eq!(translate_among(None, 2).unwrap().routed_dp_rank, None);
        assert_eq!(translate_among(Some(1), 2).unwrap().routed_dp_rank, Some(1));
        assert_eq!(translate_among(Some(0), 2).unwrap().routed_dp_rank, Some(0));
        assert_eq!(
            translate_among(Some(0), 1).unwrap().routed_dp_rank,
            None,
            "a pin on the only rank is ignored, as the frontend ignores it"
        );
        assert_eq!(
            translate_among(Some(2), 2).unwrap_err(),
            "routed_dp_rank=2 out of range [0, 2)"
        );
        assert_eq!(
            translate_among(Some(1), 1).unwrap().routed_dp_rank,
            None,
            "any non-negative pin on the only rank is ignored"
        );
        assert_eq!(
            translate_among(Some(-1), 2).unwrap_err(),
            "routed_dp_rank=-1 out of range [0, 2)"
        );
        // The routed rank survives the wire (its slot is pinned by the wire's
        // own tests) and is what the protocol's rank hook reads.
        let wire = translate_among(Some(1), 2).unwrap();
        let back: WireRequest = decode_msgpack(&encode_msgpack(&wire).unwrap()).unwrap();
        assert_eq!(back.routed_dp_rank, Some(1));
    }

    /// Prompt bounds apply before the scheduler, with or without an output
    /// limit, under the window the handshake reported; the request's own
    /// limit is kept as it is (an unset one stays the scheduler's to cap).
    #[test]
    fn sglang_validates_prompt_length_even_with_explicit_max_new_tokens() {
        let request = |prompt_len: usize, sampling: Option<sglang_proto::SamplingParams>| {
            sglang_proto::GenerateRequest {
                request_id: "r".into(),
                tokenized: Some(sglang_proto::TokenizedInput {
                    input_ids: vec![1; prompt_len],
                    original_text: String::new(),
                }),
                sampling_params: sampling,
                ..Default::default()
            }
        };
        for sampling in [
            None,
            Some(sglang_proto::SamplingParams::default()),
            Some(sglang_proto::SamplingParams {
                max_new_tokens: Some(16_384),
                ..Default::default()
            }),
        ] {
            for (prompt_len, max_model_len, accepted) in [
                (0, 4, false),
                (3, 4, true),
                (4, 4, false),
                (5, 4, false),
                (1_049_692, 1_048_576, false),
                // A scheduler that reported no window bounds nothing but
                // emptiness.
                (1_049_692, 0, true),
                (0, 0, false),
            ] {
                let result = translate_request_sglang(
                    request(prompt_len, sampling.clone()),
                    max_model_len,
                    1,
                );
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "prompt_len={prompt_len}, max_model_len={max_model_len}, max_new_tokens={:?}",
                    sampling.as_ref().and_then(|sp| sp.max_new_tokens)
                );
                match result {
                    Err(error) => assert!(error.contains("prompt"), "{error}"),
                    Ok(wire) => {
                        if let Some(sp) = sampling.as_ref() {
                            assert_eq!(wire.sampling_params.max_new_tokens, sp.max_new_tokens);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sglang_scoring_request_keeps_selected_ids_without_generating_tokens() {
        let request = translate(sglang_proto::GenerateRequest {
            request_id: "score".into(),
            tokenized: Some(sglang_proto::TokenizedInput {
                input_ids: vec![1, 2],
                original_text: String::new(),
            }),
            sampling_params: Some(sglang_proto::SamplingParams {
                max_new_tokens: Some(0),
                ..Default::default()
            }),
            return_logprob: true,
            logprob_start_len: -1,
            token_ids_logprob: vec![5, 42],
            ..Default::default()
        })
        .expect("selected-token scoring is supported");
        assert_eq!(request.token_ids_logprob, Some(vec![5, 42]));
        assert_eq!(request.sampling_params.max_new_tokens, Some(0));
        assert!(request.return_logprob);
        assert_eq!(request.logprob_start_len, -1);
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
        let (client, engine, _dir) = connected_pair(RuntimeType::Sglang).await;

        let unsupported = client
            .generate_sglang(sglang_proto::GenerateRequest {
                token_ids_logprob: vec![5],
                ..Default::default()
            })
            .await
            .err()
            .expect("the merged direct ZMQ path cannot carry selected scores");
        assert_eq!(unsupported.code(), tonic::Code::Unimplemented);
        assert!(unsupported.message().contains("SGLang gRPC servicer"));

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
            // Keep the peer alive until the frontend has consumed the replies.
            (input, output)
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
        let (client, engine, _dir) = connected_pair(RuntimeType::Sglang).await;
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
            // Keep the peer alive until the frontend has consumed the replies.
            (input, output)
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
        let (client, engine, _dir) = connected_pair(RuntimeType::Sglang).await;
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
            // Keep the peer alive until the frontend has consumed the replies.
            (input, output)
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

    /// The scheduler's own attribution decides what an abort of a constrained
    /// request is: the abort the plugin stamped as the grammar's rejection of
    /// a sampled token (`GRAMMAR_REJECTED_ERR_TYPE`, bare as the scheduler's
    /// main grammar paths leave it, or with the 500 its PD decode path adds)
    /// is the constraint-failure status every wire ends such a request with,
    /// the scheduler's words kept. The same bare abort without the stamp (an
    /// abort-all, a session's end, a plugin that predates the column) stays
    /// the internal error it was, and an abort with a status and another name
    /// keeps the status table's entry.
    #[tokio::test]
    async fn only_the_abort_the_scheduler_attributes_to_the_grammar_is_the_constraint_failing() {
        use smg_grpc_client::is_constraint_failure;

        let (client, engine, _dir) = connected_pair(RuntimeType::Sglang).await;
        // (rid, the plugin's err_type column, the status, the message, the code)
        let cases = [
            (
                "r4",
                Some(Some(GRAMMAR_REJECTED_ERR_TYPE)),
                None,
                "the grammar rejected a sampled token",
                tonic::Code::FailedPrecondition,
            ),
            (
                "r5",
                Some(Some(GRAMMAR_REJECTED_ERR_TYPE)),
                Some(500),
                "Grammar accept_token failed for req r5 with token 7: Tokens not accepted: 7",
                tonic::Code::FailedPrecondition,
            ),
            // The bare abort of an abort-all, from a plugin without the column.
            ("r6", None, None, "Aborted", tonic::Code::Internal),
            // The same bare abort, the column present and empty for it.
            ("r7", Some(None), None, "Aborted", tonic::Code::Internal),
            (
                "r8",
                Some(Some("BadRequestError")),
                Some(400),
                "invalid stop_regex",
                tonic::Code::InvalidArgument,
            ),
        ];
        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            for (rid, err_type, status, message, _) in cases {
                let _ = input.recv_frames().await.unwrap();
                let aborted = BatchTokenIDSlimOutput {
                    rids: vec![rid.into()],
                    output_ids: vec![vec![]],
                    finished_reasons: vec!["abort".into()],
                    finished_messages: vec![Some(message.into())],
                    finished_matched: vec![None],
                    prompt_tokens: vec![3],
                    completion_tokens: vec![0],
                    cached_tokens: vec![0],
                    output_token_logprobs_val: vec![vec![]],
                    output_token_logprobs_idx: vec![vec![]],
                    finished_status: vec![status],
                    finished_err_types: err_type
                        .map(|name| vec![name.map(str::to_string)])
                        .unwrap_or_default(),
                    ..Default::default()
                };
                output
                    .send_frames(vec![bytes::Bytes::from(encode_msgpack(&aborted).unwrap())])
                    .await
                    .unwrap();
            }
            // Keep the peer alive until the frontend has consumed the replies.
            (input, output)
        });
        for (rid, _, _, message, code) in cases {
            let req = sglang_proto::GenerateRequest {
                request_id: rid.to_string(),
                tokenized: Some(sglang_proto::TokenizedInput {
                    input_ids: vec![1, 2, 3],
                    original_text: String::new(),
                }),
                sampling_params: Some(sglang_proto::SamplingParams {
                    max_new_tokens: Some(8),
                    constraint: Some(sglang_proto::sampling_params::Constraint::JsonSchema(
                        r#"{"type":"object"}"#.to_string(),
                    )),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let mut stream = client.generate_sglang(req).await.expect("generate");
            let error = stream.next().await.expect("an item").expect_err("an error");
            assert_eq!(error.code(), code, "{rid}");
            let constraint = code == tonic::Code::FailedPrecondition;
            assert_eq!(is_constraint_failure(&error), constraint, "{rid}: {error}");
            assert!(
                error
                    .message()
                    .ends_with(&format!("the scheduler aborted the request: {message}")),
                "{rid}: {error}"
            );
            assert_eq!(
                error.message().starts_with(
                    "the engine could not continue the request under its constraint: "
                ),
                constraint,
                "{rid}: {error}"
            );
        }
        engine_task.await.unwrap();
    }
}
