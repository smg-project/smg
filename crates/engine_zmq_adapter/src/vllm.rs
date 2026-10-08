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
    eos::EosTokenIds,
    fanout::fan_out_n,
    multimodal,
    stream::{poll_mapped, MappedGenerateStream, StreamState},
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
        // that would produce a 200 with no content.
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
    translate_request_with_backend(req, max_model_len, model_dtype, eos, None)
}

/// The engine's `--structured-outputs-config.backend`, as it bears on the
/// `_backend` this translation stamps on structured requests: `auto` (vLLM's
/// default) resolves per constraint the way vLLM's frontend does
/// ([`StructuredOutputsParams::with_auto_backend`]); an explicit backend is
/// pinned on every request. The engine keeps the backend of its first
/// structured request, as it does behind vLLM's own frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredOutputsBackendConfig {
    Auto,
    Pinned(StructuredOutputBackend),
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

/// [`translate_request`] with the grammar backend configured for the engine
/// (`None` keeps the per-constraint default of the translation); the
/// request's media is translated inline.
#[cfg(test)]
pub(crate) fn translate_request_with_backend(
    mut req: vllm::GenerateRequest,
    max_model_len: u64,
    model_dtype: ModelDtype,
    eos: &EosTokenIds,
    structured_backend: Option<StructuredOutputsBackendConfig>,
) -> Result<EngineCoreRequest, String> {
    let media = translate_media(&mut req, model_dtype)?;
    translate_request_with_media(req, media, max_model_len, eos, structured_backend)
}

/// Translate a request whose media was already translated
/// ([`translate_media`]), so the choices of one request share the work.
pub(crate) fn translate_request_with_media(
    req: vllm::GenerateRequest,
    media: TranslatedMedia,
    max_model_len: u64,
    eos: &EosTokenIds,
    structured_backend: Option<StructuredOutputsBackendConfig>,
) -> Result<EngineCoreRequest, String> {
    // Connector params ride `SamplingParams.extra_args`, where vLLM's own
    // frontend puts them, so a request carrying them always gets params.
    let kv_transfer_params = kv_transfer_params(&req)?;
    prompt_token_ids(&req)?;
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
    // vLLM's frontend defaults an unset `max_tokens` to the remaining context
    // (`max_model_len - prompt_len`).
    let prompt_len = prompt_token_ids.as_ref().map_or(0, |ids| ids.len()) as u64;
    let default_max_tokens =
        u32::try_from(max_model_len.saturating_sub(prompt_len)).unwrap_or(u32::MAX);
    if let Some(sp) = req.sampling_params.as_ref() {
        validate_sampling(sp, sp.max_tokens.unwrap_or(default_max_tokens))?;
    }
    let sampling_params = match (req.sampling_params, kv_transfer_params.is_some()) {
        (None, false) => None,
        (sp, _) => Some(translate_sampling(
            sp.unwrap_or_default(),
            default_max_tokens,
            eos,
            kv_transfer_params,
            structured_backend,
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

pub(crate) fn translate_sampling(
    sp: vllm::SamplingParams,
    default_max_tokens: u32,
    eos: &EosTokenIds,
    kv_transfer_params: Option<serde_json::Value>,
    structured_backend: Option<StructuredOutputsBackendConfig>,
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
        structured_outputs: sp.constraint.and_then(translate_constraint).map(|params| {
            match structured_backend {
                None => params,
                Some(StructuredOutputsBackendConfig::Auto) => params.with_auto_backend(),
                Some(StructuredOutputsBackendConfig::Pinned(backend)) => {
                    params.with_backend(backend)
                }
            }
        }),
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

/// Map the proto `constraint` oneof onto typed structured-output params. The
/// backend defaults to guidance engine-side; `json_object=false` selects no
/// constraint (the caller opted out), so it maps to `None`.
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
mod tests {
    use std::time::Duration;

    use engine_zmq_client::{
        mock_engine::{connect_to_frontend, default_ready_response, EngineInbound},
        protocol::vllm::{
            logprobs::{Logprobs, PositionLogprobs, TokenLogprob},
            output::{EngineCoreOutputs, RequestBatchOutputs},
        },
        EngineId,
    };
    use openai_protocol::worker::RuntimeType;

    use super::*;
    use crate::{client::ZmqEngineClient, eos::EosTokenIds};

    fn batch(
        request_id: &str,
        token: u32,
        logprob: Option<f32>,
        finish: Option<EngineCoreFinishReason>,
    ) -> EngineCoreOutputs {
        let finished = finish.map(|_| BTreeSet::from([request_id.to_string()]));
        let new_logprobs = logprob.map(|lp| Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![TokenLogprob {
                    token_id: token,
                    logprob: lp,
                    rank: 1,
                }],
            }],
        });
        EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: request_id.to_string(),
                new_token_ids: vec![token],
                new_logprobs,
                finish_reason: finish,
                ..Default::default()
            }],
            finished_requests: finished,
            ..Default::default()
        })
    }

    /// End-to-end over ipc://: the adapter translates a vLLM-proto request to
    /// EngineCore, and maps the engine's outputs back to vLLM-proto responses.
    #[tokio::test]
    async fn generate_e2e_translates_and_streams_vllm_proto() {
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
                RuntimeType::Vllm,
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
            let inbound = input.recv().await.unwrap();
            let request = match inbound {
                EngineInbound::Add(request) => request,
                other => panic!("expected Add, got {other:?}"),
            };
            assert_eq!(request.request_id, "r1");
            assert_eq!(request.prompt_token_ids, Some(vec![1, 2, 3]));
            assert_eq!(request.sampling_params.as_ref().unwrap().max_tokens, 2);
            output
                .send_outputs(&batch("r1", 10, Some(-0.5), None))
                .await
                .unwrap();
            output
                .send_outputs(&batch(
                    "r1",
                    11,
                    Some(-1.25),
                    Some(EngineCoreFinishReason::Length),
                ))
                .await
                .unwrap();
        });

        let req = vllm::GenerateRequest {
            request_id: "r1".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(2),
                logprobs: Some(1),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = client.generate_vllm(req).await.expect("generate");

        let first = stream.next().await.expect("chunk item").expect("chunk ok");
        match first.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![10]);
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5]);
                assert_eq!(logprobs.token_ids, vec![10]);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        // The finish tick carried a new token, so its delta is emitted as a
        // chunk before the (cumulative) terminal complete.
        let second = stream.next().await.expect("chunk item").expect("chunk ok");
        match second.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![11]);
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-1.25]);
                assert_eq!(logprobs.token_ids, vec![11]);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        let third = stream
            .next()
            .await
            .expect("complete item")
            .expect("complete ok");
        match third.response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                assert_eq!(complete.output_ids, vec![10, 11]);
                assert_eq!(complete.finish_reason, "length");
                assert_eq!(complete.completion_tokens, 2);
                let logprobs = complete.output_logprobs.expect("complete logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5, -1.25]);
                assert_eq!(logprobs.token_ids, vec![10, 11]);
            }
            other => panic!("expected complete, got {other:?}"),
        }
        assert!(stream.next().await.is_none());

        engine_task.await.unwrap();
    }

    /// The sampled token also ranks first — the common case under greedy or
    /// low-temperature decoding. The engine repeats it in the ranked columns,
    /// so the shaped list must carry it once and still return `k` candidates.
    #[test]
    fn shape_top_logprobs_dedups_sampled_token_at_rank_one() {
        let entries = vec![
            TokenLogprob {
                token_id: 10,
                logprob: -0.1,
                rank: 1,
            },
            TokenLogprob {
                token_id: 10,
                logprob: -0.1,
                rank: 1,
            },
            TokenLogprob {
                token_id: 20,
                logprob: -0.3,
                rank: 2,
            },
        ];
        assert_eq!(
            shape_top_logprobs(&entries, 2),
            vllm::TopLogProbs {
                values: vec![-0.1, -0.3],
                token_ids: vec![10, 20],
            }
        );
        assert_eq!(
            shape_top_logprobs(&entries, 1),
            vllm::TopLogProbs {
                values: vec![-0.1],
                token_ids: vec![10],
            }
        );
    }

    /// A sampled token outside the top-k leads the list and the ranked
    /// candidates follow in order, truncated to the requested count.
    #[test]
    fn shape_top_logprobs_keeps_sampled_token_outside_top_k() {
        let entries = vec![
            TokenLogprob {
                token_id: 10,
                logprob: -0.5,
                rank: 5,
            },
            TokenLogprob {
                token_id: 20,
                logprob: -0.1,
                rank: 1,
            },
            TokenLogprob {
                token_id: 30,
                logprob: -0.3,
                rank: 2,
            },
        ];
        assert_eq!(
            shape_top_logprobs(&entries, 2),
            vllm::TopLogProbs {
                values: vec![-0.5, -0.1],
                token_ids: vec![10, 20],
            }
        );
    }

    /// With `logprobs=k`, each position's ranked candidates are shaped into
    /// `top_logprobs`, taking the sampled entry plus the leading candidates up
    /// to the requested count (matching the gRPC servicer's `islice` behaviour).
    #[tokio::test]
    async fn generate_shapes_top_logprobs_to_requested_count() {
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
                RuntimeType::Vllm,
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

        // One position with the sampled token (actual vocab rank) first, then
        // the engine's ranked candidates. The wire carries `k + 1` entries.
        let position = PositionLogprobs {
            entries: vec![
                TokenLogprob {
                    token_id: 10,
                    logprob: -0.5,
                    rank: 5,
                },
                TokenLogprob {
                    token_id: 20,
                    logprob: -0.1,
                    rank: 1,
                },
                TokenLogprob {
                    token_id: 30,
                    logprob: -0.3,
                    rank: 2,
                },
            ],
        };
        let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: "r1".to_string(),
                new_token_ids: vec![10],
                new_logprobs: Some(Logprobs {
                    positions: vec![position],
                }),
                finish_reason: Some(EngineCoreFinishReason::Length),
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from(["r1".to_string()])),
            ..Default::default()
        });

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let inbound = input.recv().await.unwrap();
            let request = match inbound {
                EngineInbound::Add(request) => request,
                other => panic!("expected Add, got {other:?}"),
            };
            assert_eq!(request.sampling_params.as_ref().unwrap().logprobs, Some(2));
            output.send_outputs(&outputs).await.unwrap();
        });

        let req = vllm::GenerateRequest {
            request_id: "r1".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(1),
                logprobs: Some(2),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = client.generate_vllm(req).await.expect("generate");

        // The requested count is 2, so `top_logprobs` keeps the sampled entry
        // plus the leading candidate (the third entry is dropped).
        let expected_top = vec![vllm::TopLogProbs {
            values: vec![-0.5, -0.1],
            token_ids: vec![10, 20],
        }];

        // The finish tick carried a token, so the delta streams as a chunk.
        let chunk = stream.next().await.expect("chunk item").expect("chunk ok");
        match chunk.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                let logprobs = chunk.output_logprobs.expect("chunk logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5]);
                assert_eq!(logprobs.token_ids, vec![10]);
                assert_eq!(logprobs.top_logprobs, expected_top);
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        let complete = stream
            .next()
            .await
            .expect("complete item")
            .expect("complete ok");
        match complete.response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                let logprobs = complete.output_logprobs.expect("complete logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5]);
                assert_eq!(logprobs.token_ids, vec![10]);
                assert_eq!(logprobs.top_logprobs, expected_top);
            }
            other => panic!("expected complete, got {other:?}"),
        }
        assert!(stream.next().await.is_none());

        engine_task.await.unwrap();
    }

    /// Prompt logprobs end to end: the request carries `prompt_logprobs` to the
    /// engine, and the engine's prompt tensors come back as `input_logprobs` on
    /// the first token-bearing chunk (once) and on the terminal `Complete`.
    #[tokio::test]
    async fn generate_streams_prompt_logprobs() {
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
                RuntimeType::Vllm,
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

        // Prompt position for input token 2 (the engine scores every prompt
        // token but the first), with one ranked candidate behind it.
        let prompt_tensors = Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 2,
                        logprob: -0.5,
                        rank: 3,
                    },
                    TokenLogprob {
                        token_id: 20,
                        logprob: -0.1,
                        rank: 1,
                    },
                ],
            }],
        };
        let prefill = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: "r1".to_string(),
                new_token_ids: vec![10],
                new_prompt_logprobs_tensors: Some(prompt_tensors),
                ..Default::default()
            }],
            ..Default::default()
        });
        let decode = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: "r1".to_string(),
                new_token_ids: vec![11],
                finish_reason: Some(EngineCoreFinishReason::Length),
                ..Default::default()
            }],
            finished_requests: Some(BTreeSet::from(["r1".to_string()])),
            ..Default::default()
        });

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let inbound = input.recv().await.unwrap();
            let request = match inbound {
                EngineInbound::Add(request) => request,
                other => panic!("expected Add, got {other:?}"),
            };
            assert_eq!(
                request.sampling_params.as_ref().unwrap().prompt_logprobs,
                Some(1),
                "prompt_logprobs reaches the engine"
            );
            output.send_outputs(&prefill).await.unwrap();
            output.send_outputs(&decode).await.unwrap();
        });

        let req = vllm::GenerateRequest {
            request_id: "r1".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(2),
                prompt_logprobs: Some(1),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        let mut stream = client.generate_vllm(req).await.expect("generate");

        // Prompt token 1 leads with a null logprob (nothing precedes it); the
        // requested count of 1 keeps the prompt token's own ranked entry.
        let expect_input_logprobs = |logprobs: Option<vllm::InputLogProbs>, whose: &str| {
            let logprobs = logprobs.unwrap_or_else(|| panic!("{whose} input logprobs"));
            assert_eq!(logprobs.token_ids, vec![1, 2]);
            assert_eq!(
                logprobs.token_logprobs,
                vec![
                    vllm::InputTokenLogProb { value: None },
                    vllm::InputTokenLogProb { value: Some(-0.5) },
                ]
            );
            assert_eq!(
                logprobs.top_logprobs,
                vec![
                    vllm::TopLogProbs::default(),
                    vllm::TopLogProbs {
                        values: vec![-0.5],
                        token_ids: vec![2],
                    },
                ]
            );
        };

        let first = stream.next().await.expect("chunk item").expect("chunk ok");
        match first.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                expect_input_logprobs(chunk.input_logprobs, "first chunk");
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        // The finish tick carried a token, so its delta streams as a chunk
        // first — without repeating the prompt logprobs.
        let second = stream.next().await.expect("chunk item").expect("chunk ok");
        match second.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert!(
                    chunk.input_logprobs.is_none(),
                    "prompt logprobs ride the first chunk only"
                );
            }
            other => panic!("expected chunk, got {other:?}"),
        }
        let complete = stream
            .next()
            .await
            .expect("complete item")
            .expect("complete ok");
        match complete.response {
            Some(vllm::generate_response::Response::Complete(complete)) => {
                expect_input_logprobs(complete.input_logprobs, "complete");
            }
            other => panic!("expected complete, got {other:?}"),
        }

        engine_task.await.unwrap();
    }

    #[test]
    fn finish_reasons_map_to_vllm_strings() {
        assert_eq!(finish_reason_str(EngineCoreFinishReason::Length), "length");
        assert_eq!(
            finish_reason_str(EngineCoreFinishReason::Repetition),
            "stop"
        );
        assert_eq!(finish_reason_str(EngineCoreFinishReason::Abort), "abort");
    }

    fn tokenized_req(sampling: vllm::SamplingParams) -> vllm::GenerateRequest {
        vllm::GenerateRequest {
            request_id: "r1".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(sampling),
            stream: true,
            ..Default::default()
        }
    }

    #[test]
    fn vllm_forwards_prompt_logprobs() {
        // Prompt logprobs ride the wire sampling params verbatim.
        let request = translate_request(
            tokenized_req(vllm::SamplingParams {
                prompt_logprobs: Some(2),
                ..Default::default()
            }),
            4096,
            ModelDtype::BFloat16,
            &EosTokenIds::default(),
        )
        .expect("translated");
        assert_eq!(
            request.sampling_params.as_ref().unwrap().prompt_logprobs,
            Some(2)
        );
    }

    #[test]
    fn kv_transfer_params_prefer_json_then_the_legacy_pair() {
        let mut req = tokenized_req(vllm::SamplingParams::default());
        assert_eq!(kv_transfer_params(&req).unwrap(), None);

        req.kv_transfer_params = Some(vllm::KvTransferParams {
            remote_host: "10.0.0.1".to_string(),
            remote_port: 5600,
        });
        assert_eq!(
            kv_transfer_params(&req).unwrap(),
            Some(serde_json::json!({"remote_host": "10.0.0.1", "remote_port": 5600}))
        );
        // The JSON field wins over the legacy pair.
        req.kv_transfer_params_json = Some(r#"{"do_remote_decode":true}"#.to_string());
        assert_eq!(
            kv_transfer_params(&req).unwrap(),
            Some(serde_json::json!({"do_remote_decode": true}))
        );
        // Malformed params are the caller's error, as on the Python servicer.
        req.kv_transfer_params_json = Some("[1]".to_string());
        assert!(kv_transfer_params(&req)
            .unwrap_err()
            .contains("JSON object"));
        req.kv_transfer_params_json = Some("{".to_string());
        assert!(kv_transfer_params(&req).unwrap_err().contains("invalid"));
        req.kv_transfer_params_json = None;
        req.kv_transfer_params = Some(vllm::KvTransferParams {
            remote_host: String::new(),
            remote_port: 5600,
        });
        assert!(kv_transfer_params(&req)
            .unwrap_err()
            .contains("remote_host"));
    }

    #[test]
    fn kv_transfer_params_ride_the_sampling_extra_args() {
        let mut req = tokenized_req(vllm::SamplingParams::default());
        req.kv_transfer_params_json = Some(r#"{"do_remote_decode":true}"#.to_string());
        let request = translate_request(req, 4096, ModelDtype::BFloat16, &EosTokenIds::default())
            .expect("translated");
        let extra = request
            .sampling_params
            .as_ref()
            .and_then(|sp| sp.extra_args.as_ref())
            .expect("extra_args");
        assert_eq!(
            extra["kv_transfer_params"],
            serde_json::json!({"do_remote_decode": true})
        );
        // A request without sampling params still carries them.
        let mut req = tokenized_req(vllm::SamplingParams::default());
        req.sampling_params = None;
        req.kv_transfer_params_json = Some(r#"{"do_remote_decode":true}"#.to_string());
        let request = translate_request(req, 4096, ModelDtype::BFloat16, &EosTokenIds::default())
            .expect("translated");
        assert!(request
            .sampling_params
            .and_then(|sp| sp.extra_args)
            .is_some_and(|extra| extra.contains_key("kv_transfer_params")));
    }

    #[test]
    fn returned_kv_transfer_params_land_on_the_complete() {
        let params = serde_json::json!({
            "do_remote_prefill": true,
            "remote_block_ids": [1, 2],
            "remote_host": "10.0.0.1",
            "remote_port": 5600,
        });
        // Direct Complete.
        let mut response = vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Complete(
                vllm::GenerateComplete::default(),
            )),
        };
        let mut pending = None;
        attach_kv_transfer_params(&mut response, &mut pending, Some(params.clone()));
        let Some(vllm::generate_response::Response::Complete(complete)) = response.response else {
            panic!("complete");
        };
        let json: serde_json::Value =
            serde_json::from_str(complete.kv_transfer_params_json.as_deref().unwrap()).unwrap();
        assert_eq!(json, params);
        let legacy = complete.kv_transfer_params.unwrap();
        assert_eq!(
            (legacy.remote_host.as_str(), legacy.remote_port),
            ("10.0.0.1", 5600)
        );

        // Complete parked behind the finish tick's chunk.
        let mut response = vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Chunk(
                vllm::GenerateStreamChunk::default(),
            )),
        };
        let mut pending = Some(vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Complete(
                vllm::GenerateComplete::default(),
            )),
        });
        attach_kv_transfer_params(
            &mut response,
            &mut pending,
            Some(serde_json::json!({"remote_engine_id": "eng-a"})),
        );
        let Some(vllm::generate_response::Response::Complete(parked)) = pending.unwrap().response
        else {
            panic!("parked complete");
        };
        assert_eq!(
            parked.kv_transfer_params_json.as_deref(),
            Some(r#"{"remote_engine_id":"eng-a"}"#)
        );
        // No host/port: no legacy mirror.
        assert!(parked.kv_transfer_params.is_none());

        // A non-finish chunk carries nothing.
        let mut response = vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Chunk(
                vllm::GenerateStreamChunk::default(),
            )),
        };
        let mut pending = None;
        attach_kv_transfer_params(&mut response, &mut pending, Some(params));
        assert!(pending.is_none());
    }

    #[test]
    fn structured_output_backend_follows_the_engine_config() {
        use engine_zmq_client::protocol::vllm::structured_outputs::{
            StructuredOutputBackend, StructuredOutputConstraint,
        };
        use StructuredOutputsBackendConfig::{Auto, Pinned};

        assert_eq!(
            structured_outputs_backend_from_config("guidance"),
            Pinned(StructuredOutputBackend::Guidance)
        );
        assert_eq!(
            structured_outputs_backend_from_config(" outlines "),
            Pinned(StructuredOutputBackend::Outlines)
        );
        assert_eq!(
            structured_outputs_backend_from_config("lm-format-enforcer"),
            Pinned(StructuredOutputBackend::LmFormatEnforcer)
        );
        // vLLM reads the backend off an option-carrying spelling's prefix.
        assert_eq!(
            structured_outputs_backend_from_config("xgrammar:disable-any-whitespace"),
            Pinned(StructuredOutputBackend::Xgrammar)
        );
        // `auto` (and anything unknown) resolves per constraint.
        assert_eq!(structured_outputs_backend_from_config("auto"), Auto);
        assert_eq!(structured_outputs_backend_from_config(""), Auto);

        let constrained = || {
            tokenized_req(vllm::SamplingParams {
                constraint: Some(vllm::sampling_params::Constraint::JsonObject(true)),
                ..Default::default()
            })
        };
        let backend_of = |request: EngineCoreRequest| {
            request
                .sampling_params
                .and_then(|sp| sp.structured_outputs)
                .map(|so| so.backend)
                .expect("structured outputs")
        };
        // Pinned by the caller: every constraint carries the engine's backend.
        let request = translate_request_with_backend(
            constrained(),
            4096,
            ModelDtype::BFloat16,
            &EosTokenIds::default(),
            Some(Pinned(StructuredOutputBackend::Guidance)),
        )
        .expect("translated");
        assert_eq!(backend_of(request), StructuredOutputBackend::Guidance);
        // Unpinned: the translation's own per-constraint default.
        let request = translate_request(
            constrained(),
            4096,
            ModelDtype::BFloat16,
            &EosTokenIds::default(),
        )
        .expect("translated");
        assert_eq!(backend_of(request), StructuredOutputBackend::default());

        // `auto`: vLLM's per-constraint resolution. A Lark grammar goes to
        // xgrammar unchanged (vLLM 0.30 parses Lark there); a schema with a
        // feature xgrammar lacks goes to guidance; a choice headed for
        // xgrammar is lowered to the grammar vLLM's frontend substitutes.
        let auto = |constraint| {
            translate_request_with_backend(
                tokenized_req(vllm::SamplingParams {
                    constraint: Some(constraint),
                    ..Default::default()
                }),
                4096,
                ModelDtype::BFloat16,
                &EosTokenIds::default(),
                Some(Auto),
            )
            .expect("translated")
            .sampling_params
            .and_then(|sp| sp.structured_outputs)
            .expect("structured outputs")
        };
        let lark = "start: \"yes\" | \"no\"";
        let params = auto(vllm::sampling_params::Constraint::Grammar(lark.to_string()));
        assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
        assert_eq!(
            params.constraint,
            StructuredOutputConstraint::Grammar(lark.to_string())
        );
        let params = auto(vllm::sampling_params::Constraint::JsonSchema(
            r#"{"type":"integer","multipleOf":3}"#.to_string(),
        ));
        assert_eq!(params.backend, StructuredOutputBackend::Guidance);
        let params = auto(vllm::sampling_params::Constraint::Choice(
            vllm::ChoiceConstraint {
                choices: vec!["a".to_string(), "b".to_string()],
            },
        ));
        assert_eq!(params.backend, StructuredOutputBackend::Xgrammar);
        assert_eq!(
            params.constraint,
            StructuredOutputConstraint::Grammar("root ::= \"a\" | \"b\"".to_string())
        );
    }

    #[test]
    fn spec_decode_counts_land_on_the_complete() {
        let metrics = SpecDecodeMetrics {
            num_spec_tokens: 3,
            histogram: vec![1, 2, 0, 1],
            num_draft_tokens: 9,
            ..Default::default()
        };
        // 0*1 + 1*2 + 2*0 + 3*1 accepted draft tokens, as the Python servicer sums.
        assert_eq!(metrics.accepted_tokens(), 5);
        let mut response = vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Complete(
                vllm::GenerateComplete::default(),
            )),
        };
        let mut pending = None;
        attach_spec_decode_counts(&mut response, &mut pending, Some(metrics));
        let Some(vllm::generate_response::Response::Complete(complete)) = response.response else {
            panic!("complete");
        };
        assert_eq!(complete.spec_accepted_tokens, 5);
        assert_eq!(complete.spec_draft_tokens, 9);
    }

    #[test]
    fn ranked_candidate_count_maps_the_sentinels() {
        assert_eq!(ranked_candidate_count(None), 0);
        assert_eq!(ranked_candidate_count(Some(0)), 0);
        assert_eq!(ranked_candidate_count(Some(3)), 3);
        // -1 asks for every candidate the engine returned.
        assert_eq!(ranked_candidate_count(Some(-1)), usize::MAX);
    }

    #[test]
    fn vllm_defaults_unset_max_tokens_to_remaining_context() {
        let max_tokens = |sampling, max_model_len| {
            translate_request(
                tokenized_req(sampling),
                max_model_len,
                ModelDtype::BFloat16,
                &EosTokenIds::default(),
            )
            .expect("request translated")
            .sampling_params
            .expect("sampling params present")
            .max_tokens
        };

        // Unset max_tokens defaults to `max_model_len - prompt_len` (prompt is
        // 3 tokens), mirroring vLLM's bypassed OpenAI frontend.
        assert_eq!(max_tokens(vllm::SamplingParams::default(), 100), 97);
        // An explicit value is always honored.
        assert_eq!(
            max_tokens(
                vllm::SamplingParams {
                    max_tokens: Some(8),
                    ..Default::default()
                },
                100,
            ),
            8,
        );
    }

    #[test]
    fn vllm_attaches_eos_stop_ids() {
        let eos = EosTokenIds::new(Some(5), vec![7]);
        let sampling = |sp| {
            translate_request(tokenized_req(sp), 4096, ModelDtype::BFloat16, &eos)
                .expect("request translated")
                .sampling_params
                .expect("sampling params present")
        };

        // Primary rides `_eos_token_id`, extras merge into `stop_token_ids`
        // without duplicating, and the union lands in `_all_stop_token_ids`.
        let sp = sampling(vllm::SamplingParams {
            stop_token_ids: vec![7, 9],
            ..Default::default()
        });
        assert_eq!(sp.eos_token_id, Some(5));
        assert_eq!(sp.stop_token_ids, vec![7, 9]);
        assert_eq!(sp.all_stop_token_ids, BTreeSet::from([5, 7, 9]));

        // ignore_eos drops the EOS stops from the wire but keeps the
        // bookkeeping set (mirrors the reference frontend).
        let sp = sampling(vllm::SamplingParams {
            stop_token_ids: vec![9],
            ignore_eos: true,
            ..Default::default()
        });
        assert_eq!(sp.eos_token_id, None);
        assert_eq!(sp.stop_token_ids, vec![9]);
        assert_eq!(sp.all_stop_token_ids, BTreeSet::from([5, 7, 9]));
    }

    #[test]
    fn vllm_translates_structured_output_constraints() {
        use engine_zmq_client::protocol::vllm::structured_outputs::{
            StructuredOutputBackend, StructuredOutputConstraint,
        };

        let translate = |constraint| {
            translate_request(
                tokenized_req(vllm::SamplingParams {
                    constraint: Some(constraint),
                    ..Default::default()
                }),
                4096,
                ModelDtype::BFloat16,
                &EosTokenIds::default(),
            )
            .expect("constraint translated")
            .sampling_params
            .expect("sampling params present")
            .structured_outputs
        };

        // Each constraint mode maps onto its typed counterpart, always lowering
        // to the guidance backend engine-side.
        let json_object = translate(vllm::sampling_params::Constraint::JsonObject(true))
            .expect("json_object translated");
        assert_eq!(
            json_object.constraint,
            StructuredOutputConstraint::JsonObject
        );
        assert_eq!(json_object.backend, StructuredOutputBackend::Guidance);

        let regex = translate(vllm::sampling_params::Constraint::Regex("a.*".to_string()))
            .expect("regex translated");
        assert_eq!(
            regex.constraint,
            StructuredOutputConstraint::Regex("a.*".to_string())
        );

        let choice = translate(vllm::sampling_params::Constraint::Choice(
            vllm::ChoiceConstraint {
                choices: vec!["yes".to_string(), "no".to_string()],
            },
        ))
        .expect("choice translated");
        assert_eq!(
            choice.constraint,
            StructuredOutputConstraint::Choice(vec!["yes".to_string(), "no".to_string()])
        );

        // A JSON schema string is parsed to preserve object shape.
        let json = translate(vllm::sampling_params::Constraint::JsonSchema(
            r#"{"type":"object"}"#.to_string(),
        ))
        .expect("json schema translated");
        assert_eq!(
            json.constraint,
            StructuredOutputConstraint::Json(serde_json::json!({"type": "object"}))
        );

        // json_object=false means the caller opted out: no constraint.
        assert!(translate(vllm::sampling_params::Constraint::JsonObject(false)).is_none());
    }

    /// n=3 fans out into 3 single-sample wire requests with unique sub-rids.
    /// An explicit seed derives per-sub seeds (`seed + i`) so the samples
    /// differ deterministically; no seed stays `None` per sub (the engine
    /// seeds each rid independently).
    #[test]
    fn fan_out_splits_n_into_single_sample_requests() {
        let mut req = tokenized_req(vllm::SamplingParams {
            n: 3,
            seed: Some(7),
            ..Default::default()
        });
        req.request_id = "r1".to_string();

        let subs = fan_out_requests(req);
        assert_eq!(subs.len(), 3);
        let rids: Vec<&str> = subs.iter().map(|sub| sub.request_id.as_str()).collect();
        assert_eq!(rids, vec!["r1-0", "r1-1", "r1-2"]);
        for (i, sub) in subs.iter().enumerate() {
            let sp = sub.sampling_params.as_ref().unwrap();
            assert_eq!(sp.n, 1);
            assert_eq!(sp.seed, Some(7 + i as i32));
            // Everything else is shared verbatim.
            assert_eq!(
                sub.input,
                Some(vllm::generate_request::Input::Tokenized(
                    vllm::TokenizedInput {
                        original_text: String::new(),
                        input_ids: vec![1, 2, 3],
                    }
                ))
            );
        }

        // No explicit seed: every sub keeps None.
        let subs = fan_out_requests(tokenized_req(vllm::SamplingParams {
            n: 2,
            ..Default::default()
        }));
        assert!(subs
            .iter()
            .all(|sub| sub.sampling_params.as_ref().unwrap().seed.is_none()));

        // n<=1 passes through untouched (rid keeps its original form).
        let single = fan_out_requests(tokenized_req(vllm::SamplingParams::default()));
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].request_id, "r1");
    }

    /// n=2 over a vLLM EngineCore: two engine-side requests with distinct
    /// sub-rids, and the merged stream yields two `Complete`s tagged with the
    /// proto `index` (0 and 1) the pipeline demuxes choices by.
    #[tokio::test]
    async fn generate_e2e_fans_out_n2_vllm() {
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
                RuntimeType::Vllm,
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
            let mut rids = Vec::new();
            for _ in 0..2 {
                let request = match input.recv().await.unwrap() {
                    EngineInbound::Add(request) => request,
                    other => panic!("expected Add, got {other:?}"),
                };
                // Each sub is a single-sample request with a derived seed.
                let sp = request.sampling_params.as_ref().unwrap();
                assert_eq!(sp.max_tokens, 4);
                rids.push((request.request_id.clone(), sp.seed));
            }
            assert_eq!(
                rids,
                vec![("r1-0".to_string(), Some(5)), ("r1-1".to_string(), Some(6))],
                "sub-rids must be unique and seeds derived per sub"
            );
            output
                .send_outputs(&batch("r1-0", 10, None, Some(EngineCoreFinishReason::Stop)))
                .await
                .unwrap();
            output
                .send_outputs(&batch("r1-1", 11, None, Some(EngineCoreFinishReason::Stop)))
                .await
                .unwrap();
        });

        let mut req = tokenized_req(vllm::SamplingParams {
            n: 2,
            seed: Some(5),
            max_tokens: Some(4),
            ..Default::default()
        });
        req.request_id = "r1".to_string();
        let mut stream = client.generate_vllm(req).await.expect("generate");

        let mut completes = Vec::new();
        while let Some(item) = stream.next().await {
            match item.expect("stream item").response {
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    completes.push(complete);
                }
                Some(vllm::generate_response::Response::Chunk(_)) | None => {}
            }
        }
        completes.sort_by_key(|complete| complete.index);
        assert_eq!(completes.len(), 2, "one Complete per fanned-out sub");
        assert_eq!(completes[0].index, 0);
        assert_eq!(completes[0].output_ids, vec![10]);
        assert_eq!(completes[1].index, 1);
        assert_eq!(completes[1].output_ids, vec![11]);

        engine_task.await.unwrap();
    }

    /// Dropping the merged stream before completion aborts EVERY fanned-out
    /// engine-side sub-request, not just one.
    #[tokio::test]
    async fn dropping_fanned_out_stream_aborts_all_subs() {
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
                RuntimeType::Vllm,
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
        let (mut engine_input, _engine_output) = engine.split();

        let stream = client
            .generate_vllm(tokenized_req(vllm::SamplingParams {
                n: 2,
                ..Default::default()
            }))
            .await
            .expect("generate");

        // Consume both Adds first so the drop-triggered aborts are the next
        // inbound messages.
        for _ in 0..2 {
            match engine_input.recv().await.unwrap() {
                EngineInbound::Add(_) => {}
                other => panic!("expected Add, got {other:?}"),
            }
        }

        drop(stream); // unfinished -> every sub auto-aborts

        let mut aborted = BTreeSet::new();
        while aborted.len() < 2 {
            match engine_input.recv().await.unwrap() {
                EngineInbound::Abort(rids) => aborted.extend(rids),
                other => panic!("expected Abort, got {other:?}"),
            }
        }
        assert_eq!(
            aborted,
            BTreeSet::from(["r1-0".to_string(), "r1-1".to_string()])
        );
    }

    fn sampling_request(sampling: vllm::SamplingParams) -> vllm::GenerateRequest {
        vllm::GenerateRequest {
            request_id: "s".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 3],
                },
            )),
            sampling_params: Some(sampling),
            ..Default::default()
        }
    }

    /// Proto "unset" zeros for `top_p`/`repetition_penalty` mean disabled, as
    /// the Python servicer translates them; the engine rejects literal zeros.
    #[test]
    fn translate_normalizes_unset_top_p_and_repetition_penalty() {
        let request = translate_request(
            sampling_request(vllm::SamplingParams::default()),
            1024,
            ModelDtype::Float32,
            &EosTokenIds::default(),
        )
        .expect("defaults translate");
        let sampling = request.sampling_params.expect("sampling params");
        assert_eq!(sampling.top_p, 1.0);
        assert_eq!(sampling.repetition_penalty, 1.0);
        assert_eq!(sampling.temperature, 1.0);
        // Unset max_tokens defaults to the remaining context.
        assert_eq!(sampling.max_tokens, 1024 - 3);
    }

    /// vLLM's parameter checks run before submit: a request the engine would
    /// reject on its input thread (killing it) is an invalid-argument error.
    #[test]
    fn translate_rejects_sampling_the_engine_would_refuse() {
        let cases: Vec<(&str, vllm::SamplingParams)> = vec![
            (
                "temperature",
                vllm::SamplingParams {
                    temperature: Some(-0.5),
                    ..Default::default()
                },
            ),
            (
                "top_p",
                vllm::SamplingParams {
                    top_p: 1.5,
                    ..Default::default()
                },
            ),
            (
                "min_p",
                vllm::SamplingParams {
                    min_p: 2.0,
                    ..Default::default()
                },
            ),
            (
                "presence_penalty",
                vllm::SamplingParams {
                    presence_penalty: 3.0,
                    ..Default::default()
                },
            ),
            (
                "repetition_penalty",
                vllm::SamplingParams {
                    repetition_penalty: -1.0,
                    ..Default::default()
                },
            ),
            (
                "max_tokens",
                vllm::SamplingParams {
                    max_tokens: Some(0),
                    ..Default::default()
                },
            ),
            (
                "min_tokens",
                vllm::SamplingParams {
                    max_tokens: Some(4),
                    min_tokens: 8,
                    ..Default::default()
                },
            ),
            (
                "logprobs",
                vllm::SamplingParams {
                    logprobs: Some(-2),
                    ..Default::default()
                },
            ),
            // vLLM's `_verify_greedy_sampling`: no distinct choices at temperature 0.
            (
                "n must be 1 when using greedy sampling, got 2.",
                vllm::SamplingParams {
                    temperature: Some(0.0),
                    n: 2,
                    ..Default::default()
                },
            ),
        ];
        for (field, sampling) in cases {
            let error = translate_request(
                sampling_request(sampling),
                1024,
                ModelDtype::Float32,
                &EosTokenIds::default(),
            )
            .expect_err(field);
            assert!(error.contains(field), "{field}: {error}");
        }
    }
}
