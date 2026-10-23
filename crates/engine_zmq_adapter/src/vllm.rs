//! vLLM dialect: proto request → `EngineCoreRequest` translation (with the
//! frontend's sampling defaults and checks) and `EngineCoreOutput` → proto
//! response mapping.

use std::{
    collections::{BTreeSet, HashMap},
    time::{SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use engine_zmq_client::{
    codec::{decode_value, dtype::ModelDtype},
    connector::EngineCoreStream,
    protocol::vllm::{
        logprobs::TokenLogprob,
        output::{EngineCoreFinishReason, EngineCoreOutput, SpecDecodeMetrics, StopReason},
        request::{EngineCoreRequest, MmFeaturesPayload},
        sampling::EngineCoreSamplingParams,
        structured_outputs::{StructuredOutputBackend, StructuredOutputsParams},
    },
};
use futures::Stream;
use smg_grpc_client::vllm_proto as vllm;

use crate::{
    context_window::admit_prompt,
    eos::EosTokenIds,
    fanout::fan_out_n,
    multimodal,
    stream::{poll_mapped, MappedGenerateStream, StreamState},
    structured_outputs::{
        refuse_unenforceable_constraint, ConstraintRef, StructuredOutputsBackendConfig,
        StructuredOutputsSupport,
    },
};

/// Ranked candidates to emit per position for a requested logprob count:
/// unset/`0` means the sampled (or prompt) token's own logprob only, and `-1`
/// means every candidate the engine returned.
pub(crate) fn ranked_candidate_count(requested: Option<i32>) -> usize {
    match requested {
        Some(n) if n < 0 => usize::MAX,
        Some(n) => n as usize,
        None => 0,
    }
}

/// Shape one position's wire entries (sampled first, then the engine's ranked
/// candidates) into `top_logprobs`: the sampled entry leads, then ranked
/// candidates fill up to `top_k` entries. The engine leaves the sampled token
/// in its ranked columns, so the ranked entry repeating it is skipped —
/// otherwise the list would carry it twice and drop the last candidate.
pub(crate) fn shape_top_logprobs(entries: &[TokenLogprob], top_k: usize) -> vllm::TopLogProbs {
    let mut top = vllm::TopLogProbs::default();
    let Some((sampled, ranked)) = entries.split_first() else {
        return top;
    };
    if top_k == 0 {
        return top;
    }
    top.values.push(sampled.logprob);
    top.token_ids.push(sampled.token_id);
    for entry in ranked {
        if top.token_ids.len() >= top_k {
            break;
        }
        if entry.token_id == sampled.token_id {
            continue;
        }
        top.values.push(entry.logprob);
        top.token_ids.push(entry.token_id);
    }
    top
}

/// Streaming generate output for one vLLM EngineCore sub-request, mapping each
/// `EngineCoreOutput` to a vLLM-proto `GenerateResponse` (chunks until the
/// terminal output, then a complete), tagged with this sub's choice `index`.
pub struct VllmGenerateStream {
    inner: EngineCoreStream,
    state: StreamState,
    /// Choice index stamped on every chunk/complete (0 for n=1; the fan-out
    /// position for n>1) — the proto field the pipeline demuxes choices by.
    index: u32,
    /// Number of ranked candidates the client requested per position; `0` when
    /// only the sampled logprob (or nothing) was asked for, in which case no
    /// `top_logprobs` are emitted.
    top_logprobs: usize,
    /// Ranked candidates per PROMPT position (`prompt_logprobs`); `0` off.
    prompt_top_logprobs: usize,
    /// First prompt token id; reported with a `null` logprob per the API
    /// contract (nothing precedes it to condition on).
    first_prompt_token: Option<u32>,
    /// Prompt logprobs are attached to the first emitted chunk exactly once.
    input_logprobs_emitted: bool,
    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens: streaming frontends decode text/logprobs from chunks only, so
    /// the tick's delta goes out as a `Chunk` first.
    pending: Option<vllm::GenerateResponse>,
}

impl VllmGenerateStream {
    pub(crate) fn new(
        inner: EngineCoreStream,
        index: u32,
        top_logprobs: usize,
        prompt_top_logprobs: usize,
        first_prompt_token: Option<u32>,
    ) -> Self {
        Self {
            inner,
            state: StreamState::default(),
            index,
            top_logprobs,
            prompt_top_logprobs,
            first_prompt_token,
            input_logprobs_emitted: false,
            pending: None,
        }
    }

    /// The choice `index` stamped on this stream's responses.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// The engine-side id of this choice's request.
    pub fn request_id(&self) -> &str {
        self.inner.request_id()
    }

    /// The data-parallel rank this choice's engine request went to; `None`
    /// for an engine whose identity names no index.
    pub fn engine_index(&self) -> Option<u32> {
        self.inner.engine_index()
    }

    /// Whether the engine already finished this choice and its terminal
    /// `Complete` is parked behind the chunk just yielded (the finish tick
    /// carried new tokens). A frontend that matched a stop on that chunk must
    /// not synthesize its own `Complete`: the accumulated state has already
    /// been drained into the parked one.
    pub fn has_parked_complete(&self) -> bool {
        self.pending.is_some()
    }

    /// End this choice now on a frontend-matched string stop: the terminal
    /// `Complete` built from the accumulated state, exactly as if the engine
    /// had reported `stop` with that string. The engine-side request is
    /// aborted when the stream is dropped, so the caller drops it next.
    pub fn complete_with_matched_stop(&mut self, matched: String) -> vllm::GenerateResponse {
        self.complete_with_finish(
            "stop".to_string(),
            Some(vllm::generate_complete::MatchedStop::MatchedStopStr(
                matched,
            )),
        )
    }

    /// End this choice as aborted: the engine's own parked `Complete` when it
    /// already finished, else the terminal `Complete` the Python servicer
    /// yields after an `Abort` RPC (`finish_reason = "abort"`).
    pub fn complete_aborted(&mut self) -> vllm::GenerateResponse {
        if let Some(parked) = self.pending.take() {
            return parked;
        }
        self.complete_with_finish("abort".to_string(), None)
    }

    fn complete_with_finish(
        &mut self,
        finish_reason: String,
        matched: Option<vllm::generate_complete::MatchedStop>,
    ) -> vllm::GenerateResponse {
        let mut pending = None;
        // No new tokens on a frontend finish, so `emit_tick` yields the
        // `Complete` directly and parks nothing.
        let mut response = self.state.emit_tick(
            self.index,
            Vec::new(),
            None,
            Some((finish_reason, matched)),
            &mut pending,
        );
        self.attach_input_logprobs(&mut response);
        response
    }

    /// Attach the accumulated prompt logprobs: once on the first token-bearing
    /// chunk (the proto puts them in the first chunk only) and on every
    /// `Complete`, including one parked in `pending`. Prefill precedes the
    /// first sampled token, so the set is whole by the time a chunk carries
    /// tokens.
    fn attach_input_logprobs(&mut self, response: &mut vllm::GenerateResponse) {
        if self.state.prompt_logprobs.is_empty() {
            return;
        }
        // Built per attachment site rather than up front: with
        // `prompt_logprobs` requested this runs on every decode tick, and the
        // common tick (a later chunk) attaches nothing.
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
            // Later chunks never repeat them (the proto carries them in the
            // first chunk only), and neither do token-less prefill chunks.
            Some(vllm::generate_response::Response::Chunk(_)) => {}
            Some(vllm::generate_response::Response::Complete(complete)) => {
                complete.input_logprobs = Some(build());
            }
            None => {}
        }
    }
}

impl MappedGenerateStream for VllmGenerateStream {
    type Output = EngineCoreOutput;
    type Inner = EngineCoreStream;

    fn inner(&mut self) -> &mut Self::Inner {
        &mut self.inner
    }

    fn pending(&mut self) -> &mut Option<vllm::GenerateResponse> {
        &mut self.pending
    }

    fn map_output(
        &mut self,
        output: EngineCoreOutput,
    ) -> Result<vllm::GenerateResponse, tonic::Status> {
        let top_k = self.top_logprobs;
        let state = &mut self.state;
        if let Some(stats) = &output.prefill_stats {
            state.prompt_tokens = stats.num_prompt_tokens;
            state.cached_tokens = stats.num_cached_tokens;
        }
        let token_ids = output.new_token_ids;
        state.completion_tokens += token_ids.len() as u32;
        state.output_ids.extend(token_ids.iter().copied());

        // Sampled-token logprobs (entry 0 per position) plus the requested
        // ranked candidates (`top_logprobs`). Chunks carry this tick's
        // increment; the terminal `Complete` carries the cumulative set, so
        // accumulate into `state` and drain it on finish.
        let mut tick_logprobs_val = Vec::new();
        let mut tick_logprobs_idx = Vec::new();
        let mut tick_top_logprobs = Vec::new();
        if let Some(decoded) = &output.new_logprobs {
            for position in &decoded.positions {
                let Some(sampled) = position.entries.first() else {
                    continue;
                };
                tick_logprobs_val.push(sampled.logprob);
                tick_logprobs_idx.push(sampled.token_id);
                // The entries arrive sampled-first then rank-ordered; shape the
                // requested count so one ranked list lands per sampled token.
                if top_k > 0 {
                    tick_top_logprobs.push(shape_top_logprobs(&position.entries, top_k));
                }
            }
        }
        let chunk_logprobs = (!tick_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
            token_logprobs: tick_logprobs_val.clone(),
            token_ids: tick_logprobs_idx.clone(),
            top_logprobs: tick_top_logprobs.clone(),
        });
        state.output_logprobs_val.extend(tick_logprobs_val);
        state.output_logprobs_idx.extend(tick_logprobs_idx);
        state.output_top_logprobs.extend(tick_top_logprobs);

        // Prompt logprobs accumulate the same way (chunked prefill delivers
        // them incrementally); entry 0 per position is the actual prompt
        // token. The API contract reports the first prompt token with a null
        // logprob, so seed it once before the first scored position.
        if let Some(decoded) = &output.new_prompt_logprobs_tensors {
            if state.prompt_logprobs.is_empty() && !decoded.positions.is_empty() {
                if let Some(first) = self.first_prompt_token {
                    state
                        .prompt_logprobs
                        .push(vllm::InputTokenLogProb::default());
                    state.prompt_token_ids.push(first);
                    if self.prompt_top_logprobs > 0 {
                        state.prompt_top_logprobs.push(vllm::TopLogProbs::default());
                    }
                }
            }
            for position in &decoded.positions {
                let Some(selected) = position.entries.first() else {
                    continue;
                };
                state.prompt_logprobs.push(vllm::InputTokenLogProb {
                    value: Some(selected.logprob),
                });
                state.prompt_token_ids.push(selected.token_id);
                if self.prompt_top_logprobs > 0 {
                    state.prompt_top_logprobs.push(shape_top_logprobs(
                        &position.entries,
                        self.prompt_top_logprobs,
                    ));
                }
            }
        }

        // An engine-side request failure (e.g. grammar compilation) must
        // surface as an error, not as a normal completion with empty output —
        // that would produce a 200 with no content. The engine names no cause
        // on this wire: the same `Error` ends a request whose grammar rejected
        // a token, a decode side whose KV load failed, an encoder-cache miss
        // and a pre-processing fault, and an `Abort` the frontend did not ask
        // for is a pause or a shutdown aborting every request; so a
        // constrained request's `Error` stays the engine's own report (the
        // router retries, charges and probes it as such) and its `Abort` keeps
        // its shape. The constraint-failure status is raised only by a wire
        // whose engine attributes the failure (see the SGLang stream).
        if matches!(output.finish_reason, Some(EngineCoreFinishReason::Error)) {
            return Err(tonic::Status::internal(
                "engine finished the request with an error (see engine logs)",
            ));
        }
        let finish = output.finish_reason.map(|reason| {
            (
                finish_reason_str(reason).to_string(),
                output.stop_reason.map(map_matched_stop),
            )
        });
        let mut response = state.emit_tick(
            self.index,
            token_ids,
            chunk_logprobs,
            finish,
            &mut self.pending,
        );
        self.attach_input_logprobs(&mut response);
        attach_kv_transfer_params(&mut response, &mut self.pending, output.kv_transfer_params);
        attach_spec_decode_counts(&mut response, &mut self.pending, output.spec_decode_metrics);
        Ok(response)
    }
}

/// The terminal `Complete` of this tick, direct or parked behind its chunk.
fn complete_of<'a>(
    response: &'a mut vllm::GenerateResponse,
    pending: &'a mut Option<vllm::GenerateResponse>,
) -> Option<&'a mut vllm::GenerateComplete> {
    match response.response.as_mut() {
        Some(vllm::generate_response::Response::Complete(complete)) => Some(complete),
        _ => match pending.as_mut().and_then(|parked| parked.response.as_mut()) {
            Some(vllm::generate_response::Response::Complete(complete)) => Some(complete),
            _ => None,
        },
    }
}

/// Per-request speculative-decoding counts on the `Complete`, as the Python
/// servicer reports them (accepted = histogram-weighted sum; drafted total).
fn attach_spec_decode_counts(
    response: &mut vllm::GenerateResponse,
    pending: &mut Option<vllm::GenerateResponse>,
    metrics: Option<SpecDecodeMetrics>,
) {
    let Some(metrics) = metrics else {
        return;
    };
    if let Some(complete) = complete_of(response, pending) {
        complete.spec_accepted_tokens =
            u32::try_from(metrics.accepted_tokens()).unwrap_or(u32::MAX);
        complete.spec_draft_tokens = u32::try_from(metrics.num_draft_tokens).unwrap_or(u32::MAX);
    }
}

/// Put the connector's returned KV-transfer params (a PD prefill's handoff
/// details) on the terminal `Complete`, direct or parked behind this tick's
/// chunk: the JSON field verbatim, plus the legacy typed mirror when they
/// carry a valid host/port, as the Python servicer reports them.
fn attach_kv_transfer_params(
    response: &mut vllm::GenerateResponse,
    pending: &mut Option<vllm::GenerateResponse>,
    params: Option<serde_json::Value>,
) {
    let Some(params) = params else {
        return;
    };
    // vLLM sets them on the finished output only; nothing to carry otherwise.
    let Some(complete) = complete_of(response, pending) else {
        return;
    };
    complete.kv_transfer_params = params
        .get("remote_host")
        .and_then(serde_json::Value::as_str)
        .filter(|host| !host.is_empty())
        .zip(
            params
                .get("remote_port")
                .and_then(serde_json::Value::as_u64)
                .filter(|port| (1..=65535).contains(port))
                .and_then(|port| u32::try_from(port).ok()),
        )
        .map(|(remote_host, remote_port)| vllm::KvTransferParams {
            remote_host: remote_host.to_string(),
            remote_port,
        });
    complete.kv_transfer_params_json = Some(params.to_string());
}

impl Stream for VllmGenerateStream {
    type Item = Result<vllm::GenerateResponse, tonic::Status>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        poll_mapped(self.get_mut(), cx)
    }
}

/// Split an `n > 1` generate request into `n` independent single-sample wire
/// requests (an `n <= 1` request passes through untouched).
///
/// - Rids: sub `i` is `"{request_id}-{i}"` — engine-side rids must be unique,
///   and the pipeline's request id is already unique per request, so the
///   suffixed forms are too. Dropping the merged stream aborts every sub.
/// - Seeds: with no explicit seed each sub keeps `None` — the engine seeds each
///   rid independently, so the samples differ. An explicit seed becomes
///   `seed + i` per sub: a fixed seed with identical params would otherwise
///   make all n samples identical, and deriving distinct per-sample seeds from
///   the request seed is the established engine convention (each of the n
///   sequences gets its own sampler state for exactly this reason) while
///   staying deterministic for repeat runs.
/// - Usage: every sub reports the full `prompt_tokens` on its `Complete` (the
///   subs share one prompt), matching the gRPC engines' n>1 contract — the
///   pipeline de-duplicates (max per prompt), so nothing is counted n times.
pub(crate) fn fan_out_requests(req: vllm::GenerateRequest) -> Vec<vllm::GenerateRequest> {
    let n = req.sampling_params.as_ref().map_or(1, |sp| sp.n.max(1));
    fan_out_n(req, n, |sub, i| {
        sub.request_id = format!("{}-{i}", sub.request_id);
        if let Some(sp) = sub.sampling_params.as_mut() {
            sp.n = 1;
            // An explicit seed must still yield distinct samples per sub.
            sp.seed = sp.seed.map(|seed| seed.wrapping_add(i as i32));
        }
    })
}

/// Translate a vLLM-proto generate request into an `EngineCoreRequest`. ZMQ mode
/// requires pre-tokenized input (SMG tokenizes upstream).
#[cfg(test)]
pub(crate) fn translate_request(
    req: vllm::GenerateRequest,
    max_model_len: u64,
    model_dtype: ModelDtype,
    eos: &EosTokenIds,
) -> Result<EngineCoreRequest, String> {
    translate_request_with_backend(
        req,
        max_model_len,
        model_dtype,
        eos,
        StructuredOutputsBackendConfig::default(),
    )
}

/// [`StructuredOutputsBackendConfig`] from the configured backend name.
/// vLLM accepts `xgrammar:...`/`guidance:...` option spellings and reads the
/// backend off the prefix; an unknown name is `auto`.
pub fn structured_outputs_backend_from_config(name: &str) -> StructuredOutputsBackendConfig {
    let name = name.trim();
    let backend = if name.starts_with("xgrammar") {
        StructuredOutputBackend::Xgrammar
    } else if name.starts_with("guidance") {
        StructuredOutputBackend::Guidance
    } else if name == "outlines" {
        StructuredOutputBackend::Outlines
    } else if name == "lm-format-enforcer" {
        StructuredOutputBackend::LmFormatEnforcer
    } else {
        return StructuredOutputsBackendConfig::Auto;
    };
    StructuredOutputsBackendConfig::Pinned(backend)
}

/// A request's media translated once, for every choice of the request: the
/// per-item features (built here, or relayed from a worker-side processor
/// with the aux frames its encoder split off), or the identity salt of a
/// tensor-less payload. Cloning shares the tensor bytes.
#[derive(Debug, Clone, Default)]
pub(crate) struct TranslatedMedia {
    pub(crate) mm_features: Option<MmFeaturesPayload>,
    pub(crate) aux_frames: Vec<Bytes>,
    pub(crate) cache_salt: Option<String>,
}

/// Media processed worker-side (the servicer's `media_refs` path), in the
/// shape the processor's pipeline makes it.
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessedMedia {
    /// vLLM's own input processor ran: the request's `mm_features` as vLLM's
    /// `MsgpackEncoder` wrote them, relayed to the engine as is.
    Encoded {
        /// The encoder's primary buffer; `None` when the processed request
        /// carries no features.
        mm_features: Option<Bytes>,
        /// The encoder's aux buffers in order: tensors over the zero-copy
        /// threshold, referenced from the primary buffer by index (1-based).
        aux_frames: Vec<Bytes>,
        cache_salt: Option<String>,
    },
    /// smg's own pipeline ran: the batches a Router request would carry,
    /// translated for the engine the same way.
    Batches(Vec<vllm::MultimodalInputs>),
}

impl Default for ProcessedMedia {
    fn default() -> Self {
        Self::Encoded {
            mm_features: None,
            aux_frames: Vec::new(),
            cache_salt: None,
        }
    }
}

impl ProcessedMedia {
    /// Whether translating this means casting and splitting tensors, work
    /// for a blocking thread.
    pub fn is_batches(&self) -> bool {
        matches!(self, Self::Batches(_))
    }

    fn into_translated(
        self,
        req: &vllm::GenerateRequest,
        model_dtype: ModelDtype,
    ) -> Result<TranslatedMedia, String> {
        match self {
            Self::Encoded {
                mm_features,
                aux_frames,
                cache_salt,
            } => {
                let mm_features = mm_features
                    .map(|bytes| decode_value(&bytes).map(MmFeaturesPayload::Raw))
                    .transpose()
                    .map_err(|error| format!("processed mm_features are not msgpack: {error}"))?;
                Ok(TranslatedMedia {
                    mm_features,
                    aux_frames,
                    cache_salt,
                })
            }
            Self::Batches(batches) => translate_batches_for(req, batches, model_dtype),
        }
    }
}

/// The per-item split and dtype cast of Router-shaped batches, for a request
/// whose prompt already carries the expanded placeholders.
fn translate_batches_for(
    req: &vllm::GenerateRequest,
    batches: Vec<vllm::MultimodalInputs>,
    model_dtype: ModelDtype,
) -> Result<TranslatedMedia, String> {
    if batches.is_empty() {
        return Ok(TranslatedMedia::default());
    }
    let has_kv_transfer = kv_transfer_params(req)?.is_some();
    let (mm_features, cache_salt) = multimodal::translate_batches(
        batches,
        prompt_token_ids(req)?,
        model_dtype,
        has_kv_transfer,
    )?;
    Ok(TranslatedMedia {
        mm_features: mm_features.map(MmFeaturesPayload::Typed),
        aux_frames: Vec::new(),
        cache_salt,
    })
}

/// Whether the request carries multimodal batches to translate.
pub(crate) fn has_media(req: &vllm::GenerateRequest) -> bool {
    req.mm_inputs.is_some() || !req.extra_mm_inputs.is_empty()
}

fn prompt_token_ids(req: &vllm::GenerateRequest) -> Result<&[u32], String> {
    match &req.input {
        Some(vllm::generate_request::Input::Tokenized(tokenized)) => Ok(&tokenized.input_ids),
        Some(vllm::generate_request::Input::Text(_)) => {
            Err("ZMQ mode requires pre-tokenized input (TokenizedInput)".to_string())
        }
        None => Err("ZMQ mode requires pre-tokenized input; no input provided".to_string()),
    }
}

/// Take the request's multimodal batches (`mm_inputs` plus `extra_mm_inputs`)
/// and translate them: the per-item split the Python servicer performs before
/// the engine happens here instead (the ZMQ path bypasses it). This reads
/// `/dev/shm` payloads (once; the file is unlinked) and casts tensors, so a
/// caller on an async runtime runs it off the runtime for requests that carry
/// media ([`has_media`]).
pub(crate) fn translate_media(
    req: &mut vllm::GenerateRequest,
    model_dtype: ModelDtype,
) -> Result<TranslatedMedia, String> {
    let batches: Vec<vllm::MultimodalInputs> = req
        .mm_inputs
        .take()
        .into_iter()
        .chain(std::mem::take(&mut req.extra_mm_inputs))
        .collect();
    if batches.is_empty() {
        return Ok(TranslatedMedia::default());
    }
    translate_batches_for(req, batches, model_dtype)
}

/// The media of a request whose `media_refs` a worker-side processor already
/// turned into engine features, in place of translating its batches.
pub(crate) fn translated_from_processed(
    req: &vllm::GenerateRequest,
    processed: ProcessedMedia,
    model_dtype: ModelDtype,
) -> Result<TranslatedMedia, String> {
    if has_media(req) {
        return Err(
            "a request with worker-processed media cannot also carry preprocessed \
                    multimodal inputs"
                .to_string(),
        );
    }
    processed.into_translated(req, model_dtype)
}

/// [`translate_request`] with the grammar backend configured for the engine;
/// the request's media is translated inline.
#[cfg(test)]
pub(crate) fn translate_request_with_backend(
    mut req: vllm::GenerateRequest,
    max_model_len: u64,
    model_dtype: ModelDtype,
    eos: &EosTokenIds,
    structured_backend: StructuredOutputsBackendConfig,
) -> Result<EngineCoreRequest, String> {
    let media = translate_media(&mut req, model_dtype)?;
    translate_request_with_media(req, media, max_model_len, eos, structured_backend.into())
}

/// Translate a request whose media was already translated
/// ([`translate_media`]), so the choices of one request share the work.
/// An empty prompt, or one that fills the context window, is refused before
/// it can reach the engine ([`admit_prompt`]).
pub(crate) fn translate_request_with_media(
    req: vllm::GenerateRequest,
    media: TranslatedMedia,
    max_model_len: u64,
    eos: &EosTokenIds,
    support: StructuredOutputsSupport,
) -> Result<EngineCoreRequest, String> {
    // Connector params ride `SamplingParams.extra_args`, where vLLM's own
    // frontend puts them, so a request carrying them always gets params.
    let kv_transfer_params = kv_transfer_params(&req)?;
    // vLLM's frontend defaults an unset `max_tokens` to the remaining context
    // (`max_model_len - prompt_len`).
    let default_max_tokens = admit_prompt(prompt_token_ids(&req)?.len(), max_model_len)?;
    let prompt_token_ids = match req.input {
        Some(vllm::generate_request::Input::Tokenized(tokenized)) => Some(tokenized.input_ids),
        _ => None,
    };
    let TranslatedMedia {
        mm_features,
        aux_frames: _,
        cache_salt,
    } = media;
    let data_parallel_rank = req
        .data_parallel_rank
        .map(|rank| u32::try_from(rank).map_err(|_| format!("invalid data_parallel_rank: {rank}")))
        .transpose()?;
    if let Some(sp) = req.sampling_params.as_ref() {
        validate_sampling(sp, sp.max_tokens.unwrap_or(default_max_tokens))?;
        refuse_unenforceable_constraint(constraint_ref(sp), support)?;
    }
    // A grammar the gateway led with the model's thought for a worker whose
    // grammar starts at the first token starts there whatever reasoning parser
    // the engine runs: `reasoning_ended = true` is the engine's own switch for
    // the gate behind its parser being passed. A request that constrains
    // nothing, or whose grammar the gateway did not shape, says nothing of it,
    // and the engine starts the grammar where it does.
    let grammar_from_first_token = req
        .sampling_params
        .as_ref()
        .is_some_and(|sp| sp.grammar_from_first_token && sp.constraint.is_some());
    let sampling_params = match (req.sampling_params, kv_transfer_params.is_some()) {
        (None, false) => None,
        (sp, _) => Some(translate_sampling(
            sp.unwrap_or_default(),
            default_max_tokens,
            eos,
            kv_transfer_params,
            support.backend,
        )),
    };
    Ok(EngineCoreRequest {
        request_id: req.request_id,
        prompt_token_ids,
        mm_features,
        sampling_params,
        arrival_time: now_secs(),
        data_parallel_rank,
        cache_salt,
        reasoning_ended: grammar_from_first_token.then_some(true),
        ..EngineCoreRequest::default()
    })
}

/// Connector KV-transfer params for PD disaggregation, read as the Python
/// servicer reads them: the JSON field verbatim (preferred), else the legacy
/// typed host/port pair. The engine's connector interprets them; here they
/// only have to be a JSON object.
pub(crate) fn kv_transfer_params(
    req: &vllm::GenerateRequest,
) -> Result<Option<serde_json::Value>, String> {
    if let Some(json) = req.kv_transfer_params_json.as_deref() {
        let params: serde_json::Value = serde_json::from_str(json)
            .map_err(|error| format!("invalid kv_transfer_params_json: {error}"))?;
        if !params.is_object() {
            return Err("kv_transfer_params_json must be a JSON object".to_string());
        }
        return Ok(Some(params));
    }
    if let Some(legacy) = req.kv_transfer_params.as_ref() {
        if legacy.remote_host.is_empty() || !(1..=65535).contains(&legacy.remote_port) {
            return Err(
                "invalid kv_transfer_params: remote_host must be set and remote_port \
                        must be in [1, 65535]"
                    .to_string(),
            );
        }
        return Ok(Some(serde_json::json!({
            "remote_host": legacy.remote_host,
            "remote_port": legacy.remote_port,
        })));
    }
    Ok(None)
}

/// The guard a request's KV handoff passes before anything acts on it: a
/// handoff that names another request (`remote_request_id`, the prefill side's
/// id under the pairing connector, which the router gives both sides of one
/// request) is refused, so no request consumes, or releases, the blocks a
/// prefill pinned for a different one. Params that name no request (the
/// legacy host/port pair, a store's handoff, a connector with a vocabulary of
/// its own) pass, as do params the caller's own check refuses as malformed
/// and a request that carries no id of its own (nothing to compare against;
/// the id is required elsewhere); the engine's connector judges the rest.
pub fn refuse_foreign_kv_handoff(req: &vllm::GenerateRequest) -> Result<(), String> {
    let Some(params) = kv_transfer_params(req).ok().flatten() else {
        return Ok(());
    };
    match params
        .get("remote_request_id")
        .and_then(serde_json::Value::as_str)
    {
        Some(named)
            if !named.is_empty() && !req.request_id.is_empty() && named != req.request_id =>
        {
            Err(format!(
                "kv_transfer_params name another request: remote_request_id {named:?} is not this \
                 request's id {:?}",
                req.request_id
            ))
        }
        _ => Ok(()),
    }
}

pub(crate) fn translate_sampling(
    sp: vllm::SamplingParams,
    default_max_tokens: u32,
    eos: &EosTokenIds,
    kv_transfer_params: Option<serde_json::Value>,
    structured_backend: StructuredOutputsBackendConfig,
) -> EngineCoreSamplingParams {
    // Stopping at EOS is the frontend's duty here: the primary id rides
    // `_eos_token_id`, extra ids merge into `stop_token_ids`, and the union
    // feeds `_all_stop_token_ids` (engine-side `min_tokens` masking, built
    // regardless of `ignore_eos`).
    let mut stop_token_ids = sp.stop_token_ids;
    if !sp.ignore_eos {
        for id in &eos.extra {
            if !stop_token_ids.contains(id) {
                stop_token_ids.push(*id);
            }
        }
    }
    let mut all_stop_token_ids: BTreeSet<u32> = stop_token_ids.iter().copied().collect();
    all_stop_token_ids.extend(eos.primary);
    all_stop_token_ids.extend(eos.extra.iter().copied());
    let logit_bias = if sp.logit_bias.is_empty() {
        None
    } else {
        Some(
            sp.logit_bias
                .into_iter()
                .filter_map(|(token, bias)| match u32::try_from(token) {
                    Ok(t) => Some((t, bias)),
                    Err(_) => {
                        // Don't fold negatives onto key 0 (which would silently
                        // drop all but the last); skip them with a warning.
                        tracing::warn!("dropping negative logit_bias token id {token}");
                        None
                    }
                })
                .collect::<HashMap<_, _>>(),
        )
    };
    // Proto defaults the Python servicer also normalizes before vLLM sees
    // them: an unset `top_p`/`repetition_penalty` arrives as 0.0, which the
    // engine rejects, and means "disabled" (1.0).
    let normalize_one = |value: f32| if value == 0.0 { 1.0 } else { value };
    EngineCoreSamplingParams {
        temperature: sp.temperature.unwrap_or(1.0),
        top_p: normalize_one(sp.top_p),
        top_k: sp.top_k,
        min_p: sp.min_p,
        frequency_penalty: sp.frequency_penalty,
        presence_penalty: sp.presence_penalty,
        repetition_penalty: normalize_one(sp.repetition_penalty),
        max_tokens: sp.max_tokens.unwrap_or(default_max_tokens),
        min_tokens: sp.min_tokens,
        stop_token_ids,
        eos_token_id: (!sp.ignore_eos).then_some(eos.primary).flatten(),
        all_stop_token_ids,
        seed: sp.seed.map(i64::from),
        logprobs: sp.logprobs,
        prompt_logprobs: sp.prompt_logprobs,
        logit_bias,
        structured_outputs: sp
            .constraint
            .and_then(translate_constraint)
            .map(|params| params.with_backend(structured_backend.backend())),
        extra_args: kv_transfer_params
            .map(|params| HashMap::from([("kv_transfer_params".to_string(), params)])),
        ..EngineCoreSamplingParams::default()
    }
}

/// vLLM's own `SamplingParams` checks, applied before a request reaches the
/// engine. The Python servicer gets them for free (vLLM validates when it
/// builds the params in the frontend); over ZMQ the engine deserializes the
/// request on its input thread, where a validation error kills that thread
/// and silently stops the engine from reading any further input. So the
/// checks run here, and a bad request is an `invalid_argument` to its caller
/// rather than an outage. Zero `top_p`/`repetition_penalty` are the proto's
/// "unset" and are normalized to 1.0 by the translation, so they pass.
/// vLLM's `_verify_greedy_sampling`: greedy decoding (temperature below its
/// sampling epsilon) cannot yield distinct choices. Checked on the request's
/// own `n`, before the fan-out hands each choice `n = 1`.
pub(crate) fn refuse_greedy_choices(sp: &vllm::SamplingParams) -> Result<(), String> {
    if sp.temperature.is_some_and(|temperature| temperature < 1e-5) && sp.n > 1 {
        return Err(format!(
            "n must be 1 when using greedy sampling, got {}.",
            sp.n
        ));
    }
    Ok(())
}

pub(crate) fn validate_sampling(sp: &vllm::SamplingParams, max_tokens: u32) -> Result<(), String> {
    if let Some(temperature) = sp.temperature {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(format!(
                "temperature must be a finite non-negative number, got {temperature}"
            ));
        }
        refuse_greedy_choices(sp)?;
    }
    if sp.top_p != 0.0 && !(sp.top_p > 0.0 && sp.top_p <= 1.0) {
        return Err(format!("top_p must be in (0, 1], got {}", sp.top_p));
    }
    if !(0.0..=1.0).contains(&sp.min_p) {
        return Err(format!("min_p must be in [0, 1], got {}", sp.min_p));
    }
    if !(-2.0..=2.0).contains(&sp.presence_penalty) {
        return Err(format!(
            "presence_penalty must be in [-2, 2], got {}",
            sp.presence_penalty
        ));
    }
    if !(-2.0..=2.0).contains(&sp.frequency_penalty) {
        return Err(format!(
            "frequency_penalty must be in [-2, 2], got {}",
            sp.frequency_penalty
        ));
    }
    if !sp.repetition_penalty.is_finite() || sp.repetition_penalty < 0.0 {
        return Err(format!(
            "repetition_penalty must be a finite positive number, got {}",
            sp.repetition_penalty
        ));
    }
    if max_tokens == 0 {
        return Err("max_tokens must be at least 1, got 0".to_string());
    }
    if sp.min_tokens > max_tokens {
        return Err(format!(
            "min_tokens must be less than or equal to max_tokens ({max_tokens}), got {}",
            sp.min_tokens
        ));
    }
    for (name, value) in [
        ("logprobs", sp.logprobs),
        ("prompt_logprobs", sp.prompt_logprobs),
    ] {
        if let Some(value) = value {
            if value < -1 {
                return Err(format!("{name} must be non-negative or -1, got {value}"));
            }
        }
    }
    Ok(())
}

/// The request's constraint as the shared refusal reads it
/// ([`refuse_unenforceable_constraint`]): the proto `constraint` one-of by
/// kind, with its text. `json_object = false` is an opt-out, no constraint.
pub(crate) fn constraint_ref(sp: &vllm::SamplingParams) -> Option<ConstraintRef<'_>> {
    use vllm::sampling_params::Constraint;
    match sp.constraint.as_ref()? {
        Constraint::JsonSchema(schema) => Some(ConstraintRef::JsonSchema(schema)),
        Constraint::Regex(regex) => Some(ConstraintRef::Regex(regex)),
        Constraint::Grammar(grammar) => Some(ConstraintRef::Ebnf(grammar)),
        Constraint::StructuralTag(tag) => Some(ConstraintRef::StructuralTag(tag)),
        Constraint::JsonObject(true) => Some(ConstraintRef::JsonObject),
        Constraint::JsonObject(false) => None,
        Constraint::Choice(_) => Some(ConstraintRef::Choice),
    }
}

/// Map the proto `constraint` oneof onto typed structured-output params (the
/// backend is pinned by the caller, [`translate_sampling`]);
/// `json_object=false` selects no constraint (the caller opted out), so it
/// maps to `None`.
pub(crate) fn translate_constraint(
    constraint: vllm::sampling_params::Constraint,
) -> Option<StructuredOutputsParams> {
    use vllm::sampling_params::Constraint;
    match constraint {
        Constraint::JsonSchema(schema) => Some(StructuredOutputsParams::json(
            // The engine accepts a JSON schema object or a schema string; parse
            // to preserve object shape, falling back to the raw string.
            serde_json::from_str(&schema).unwrap_or(serde_json::Value::String(schema)),
        )),
        Constraint::Regex(regex) => Some(StructuredOutputsParams::regex(regex)),
        Constraint::Grammar(grammar) => Some(StructuredOutputsParams::grammar(grammar)),
        Constraint::StructuralTag(tag) => Some(StructuredOutputsParams::structural_tag(tag)),
        Constraint::JsonObject(true) => Some(StructuredOutputsParams::json_object()),
        Constraint::JsonObject(false) => None,
        Constraint::Choice(choice) => Some(StructuredOutputsParams::choice(choice.choices)),
    }
}

pub(crate) fn map_matched_stop(reason: StopReason) -> vllm::generate_complete::MatchedStop {
    match reason {
        StopReason::TokenId(id) => vllm::generate_complete::MatchedStop::MatchedTokenId(id),
        StopReason::Text(text) => vllm::generate_complete::MatchedStop::MatchedStopStr(text),
    }
}

pub(crate) fn finish_reason_str(reason: EngineCoreFinishReason) -> &'static str {
    match reason {
        EngineCoreFinishReason::Stop | EngineCoreFinishReason::Repetition => "stop",
        EngineCoreFinishReason::Length => "length",
        EngineCoreFinishReason::Abort => "abort",
        EngineCoreFinishReason::Error => "error",
    }
}

pub(crate) fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests;
