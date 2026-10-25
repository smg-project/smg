//! TokenSpeed dialect: proto request → scheduler request translation and
//! scheduler output → vLLM-proto response mapping.

use std::collections::BTreeMap;

use engine_zmq_client::{
    codec::tensor::{checked_numel, WireTensor},
    connector::TokenSpeedStream,
    protocol::tokenspeed::{
        multimodal::{
            mm_pad_value, EncodeHandshake, TokenSpeedWireMmInputs, TokenSpeedWireMmItem,
            TokenSpeedWireModality,
        },
        output::TokenSpeedOutput,
        request::TokenizedGenerateReqInput,
        sampling::SamplingParams as TokenSpeedSamplingParams,
    },
    ConnectedEngine,
};
use futures::Stream;
use smg_grpc_client::{common_proto as common, tokenspeed_proto, vllm_proto as vllm};
use tracing::warn;

use crate::{
    context_window::admit_prompt,
    fanout::fan_out_n,
    stream::{poll_mapped, MappedGenerateStream, StreamState},
    structured_outputs::ConstraintRef,
};

mod mrope;

use mrope::{grid_rows, GridItem, GridKind};
pub use mrope::{qwen35_lineage, MropeLayout};

/// Streaming generate output for one TokenSpeed sub-request, mapping each
/// `TokenSpeedOutput` to a vLLM-proto `GenerateResponse`, tagged with this
/// sub's choice `index`.
pub struct TokenSpeedGenerateStream {
    inner: TokenSpeedStream,
    state: StreamState,
    /// Choice index stamped on every chunk/complete (0 for n=1; the fan-out
    /// position for n>1) — the proto field the pipeline demuxes choices by.
    index: u32,
    /// Terminal `Complete` held back when the finish tick also carried new
    /// tokens: streaming frontends decode text/logprobs from chunks only, so
    /// the tick's delta goes out as a `Chunk` first.
    pending: Option<vllm::GenerateResponse>,
}

impl TokenSpeedGenerateStream {
    pub(crate) fn new(inner: TokenSpeedStream, index: u32) -> Self {
        Self {
            inner,
            state: StreamState::default(),
            index,
            pending: None,
        }
    }

    /// The choice index this stream stamps on its responses.
    pub fn index(&self) -> u32 {
        self.index
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
        self.state.emit_tick(
            self.index,
            Vec::new(),
            None,
            Some((finish_reason, matched)),
            &mut pending,
        )
    }
}

/// A vLLM-proto response of a TokenSpeed stream as the TokenSpeed proto the
/// `TokenSpeedScheduler` service answers with, under `request_id` (the
/// vLLM proto carries none): the two carry the same chunk and completion
/// fields (TokenSpeed's `OutputLogProbs` has no prompt side, and the ZMQ wire
/// reports no speculative counters).
pub fn to_tokenspeed_response(
    request_id: &str,
    response: vllm::GenerateResponse,
) -> tokenspeed_proto::GenerateResponse {
    use tokenspeed_proto::generate_response::Response as TsResponse;
    use vllm::generate_response::Response;
    let logprobs = |logprobs: Option<vllm::OutputLogProbs>| {
        logprobs.map(|lp| tokenspeed_proto::OutputLogProbs {
            token_logprobs: lp.token_logprobs,
            token_ids: lp.token_ids,
            top_logprobs: lp
                .top_logprobs
                .into_iter()
                .map(|top| tokenspeed_proto::TopLogProbs {
                    values: top.values,
                    token_ids: top.token_ids,
                })
                .collect(),
        })
    };
    let mapped = response.response.map(|inner| match inner {
        Response::Chunk(chunk) => TsResponse::Chunk(tokenspeed_proto::GenerateStreamChunk {
            token_ids: chunk.token_ids,
            prompt_tokens: chunk.prompt_tokens,
            completion_tokens: chunk.completion_tokens,
            cached_tokens: chunk.cached_tokens,
            output_logprobs: logprobs(chunk.output_logprobs),
            weight_version: None,
            index: chunk.index,
        }),
        Response::Complete(complete) => {
            use tokenspeed_proto::generate_complete::MatchedStop as TsMatchedStop;
            use vllm::generate_complete::MatchedStop;
            TsResponse::Complete(tokenspeed_proto::GenerateComplete {
                output_ids: complete.output_ids,
                finish_reason: complete.finish_reason,
                prompt_tokens: complete.prompt_tokens,
                completion_tokens: complete.completion_tokens,
                cached_tokens: complete.cached_tokens,
                output_logprobs: logprobs(complete.output_logprobs),
                matched_stop: complete.matched_stop.map(|matched| match matched {
                    MatchedStop::MatchedTokenId(id) => TsMatchedStop::MatchedTokenId(id),
                    MatchedStop::MatchedStopStr(text) => TsMatchedStop::MatchedStopStr(text),
                }),
                index: complete.index,
                ..Default::default()
            })
        }
    });
    tokenspeed_proto::GenerateResponse {
        request_id: request_id.to_string(),
        response: mapped,
    }
}

impl MappedGenerateStream for TokenSpeedGenerateStream {
    type Output = TokenSpeedOutput;
    type Inner = TokenSpeedStream;

    fn inner(&mut self) -> &mut Self::Inner {
        &mut self.inner
    }

    fn pending(&mut self) -> &mut Option<vllm::GenerateResponse> {
        &mut self.pending
    }

    fn map_output(
        &mut self,
        output: TokenSpeedOutput,
    ) -> Result<vllm::GenerateResponse, tonic::Status> {
        let state = &mut self.state;
        // TokenSpeed reports per-request token counts directly (cumulative for
        // completions), rather than vLLM's per-output prefill-stats deltas.
        if output.prompt_tokens > 0 {
            state.prompt_tokens = output.prompt_tokens;
        }
        if output.cached_tokens > 0 {
            state.cached_tokens = output.cached_tokens;
        }
        state.completion_tokens = output.completion_tokens;
        state.output_ids.extend(output.output_ids.iter().copied());

        // Sampled-token logprobs, if requested. The proto column is `float`, so
        // downcast the wire's `f64` values. Chunks carry this tick's increment;
        // the terminal `Complete` carries the cumulative set, so accumulate
        // into `state` and drain it on finish.
        let chunk_logprobs =
            (!output.output_logprobs_val.is_empty()).then(|| vllm::OutputLogProbs {
                token_logprobs: output
                    .output_logprobs_val
                    .iter()
                    .map(|&lp| lp as f32)
                    .collect(),
                token_ids: output.output_logprobs_idx.clone(),
                ..Default::default()
            });
        state
            .output_logprobs_val
            .extend(output.output_logprobs_val.iter().map(|&lp| lp as f32));
        state
            .output_logprobs_idx
            .extend(output.output_logprobs_idx.iter().copied());

        // An engine-side failure must surface as an error, not a normal
        // completion with empty output — mirroring the vLLM stream's guard.
        // The slim wire carries the finish type alone (no message, status or
        // abort code), and the engine has no abort code for its grammar
        // anyway: an `abort` the frontend did not ask for is a pause or a
        // capacity abort, an `error` a transfer or numerical failure. So a
        // constrained request's finishes keep the shapes every request has;
        // the constraint-failure status is raised only by a wire whose engine
        // attributes the failure (see the SGLang stream).
        if output.finish_reason.as_deref() == Some("error") {
            return Err(tonic::Status::internal(
                "engine finished the request with an error (see engine logs)",
            ));
        }
        // No matched_stop on this wire: TokenSpeed reports the finish reason
        // only, and the router-side stop machinery owns string matching.
        let finish = output
            .finish_reason
            .map(|reason| (normalize_finish_reason(&reason).to_string(), None));
        Ok(state.emit_tick(
            self.index,
            output.output_ids,
            chunk_logprobs,
            finish,
            &mut self.pending,
        ))
    }
}

impl Stream for TokenSpeedGenerateStream {
    type Item = Result<vllm::GenerateResponse, tonic::Status>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        poll_mapped(self.get_mut(), cx)
    }
}

/// Split an `n > 1` TokenSpeed proto request into `n` single-sample
/// sub-requests, the TokenSpeed analogue of [`fan_out_requests`] (the wire has
/// no per-sample demux, so `generate` fans out here). An `n <= 1` request passes
/// through untouched. Seed handling matches [`fan_out_requests`]: omission
/// delegates independent seed assignment to the engine, while explicit seed
/// `s` derives deterministic per-sample seeds `s + i`.
///
/// [`fan_out_requests`]: crate::vllm::fan_out_requests
pub(crate) fn fan_out_tokenspeed_requests(
    req: tokenspeed_proto::GenerateRequest,
) -> Vec<tokenspeed_proto::GenerateRequest> {
    let n = req.sampling_params.as_ref().map_or(1, |sp| sp.n.max(1));
    fan_out_n(req, n, |sub, i| {
        sub.request_id = format!("{}-{i}", sub.request_id);
        if let Some(sp) = sub.sampling_params.as_mut() {
            sp.n = 1;
            sp.sampling_seed = sp.sampling_seed.map(|seed| seed.wrapping_add(u64::from(i)));
        }
    })
}

/// What a multimodal item's `image_grid_thw` / `video_grid_thw` tensors are
/// for the model served: the inputs of an MRoPE position table this
/// translation computes with the engine's own rule, the inputs of one it has
/// no rule for, or the layout of the patches a vision tower reads while the
/// language model keeps 1-D positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridTensors {
    /// The grid feeds the Qwen3.5 lineage's MRoPE positions, which this
    /// translation computes as the engine's input processor does on the path
    /// this wire bypasses (the `mrope` module) and ships in the request's position
    /// slots.
    Qwen35Positions(MropeLayout),
    /// A Qwen3.5-lineage checkpoint whose launcher shipped no layout facts:
    /// the rule cannot run, and a request of this lineage must never reach
    /// the engine with empty position slots (its executor would take 1-D
    /// positions without a word), so every multimodal item is refused.
    Qwen35PositionsUnavailable,
    /// The grid feeds MRoPE positions by a rule this wire does not have
    /// (another MRoPE family, or a model not known either way): an item that
    /// carries one is refused, since a 1-D fallback would silently degrade
    /// grounding.
    Positions,
    /// The grid only describes the patch layout: the language model's rotary
    /// positions are 1-D (Muse Spark's `rope_type: default`), so the item
    /// passes through as the engine expects it.
    PatchLayout,
}

impl GridTensors {
    /// From the launcher's model facts: the checkpoint's `architectures` and
    /// `model_type` (the engine's own dispatch picks the Qwen3.5 rule by
    /// them, [`qwen35_lineage`]), its `mrope_positions` reading of the text
    /// rope config (`Some(false)` is a patch layout; `Some(true)`, and `None`
    /// for a launcher that does not say, keep the conservative reading), and
    /// the `layout` facts the rule needs (without them a lineage checkpoint's
    /// multimodal items are refused outright). The lineage comes first: a
    /// lineage checkpoint takes its rule whatever `mrope_positions` says (a
    /// `Some(false)` there would otherwise ship empty position slots the
    /// engine fills), and the reading decides only the families without a
    /// rule.
    pub fn for_model(
        architectures: &[String],
        model_type: &str,
        mrope_positions: Option<bool>,
        layout: Option<MropeLayout>,
    ) -> Self {
        if qwen35_lineage(architectures, model_type) {
            return layout.map_or(Self::Qwen35PositionsUnavailable, Self::Qwen35Positions);
        }
        match mrope_positions {
            Some(false) => Self::PatchLayout,
            Some(true) | None => Self::Positions,
        }
    }
}

/// Translate a TokenSpeed proto `GenerateRequest` into the wire
/// `TokenizedGenerateReqInput`. ZMQ mode requires pre-tokenized input (SMG
/// tokenizes upstream). `grid_tensors` says what a multimodal item's grid
/// tensors are: the inputs of a position table computed here, of one this
/// wire cannot derive (refused), or a patch layout that passes
/// ([`GridTensors`]). `max_model_len` is the context window the
/// scheduler reported at the handshake: an empty prompt, or one that fills
/// the window, is refused before it reaches the scheduler
/// ([`admit_prompt`]). `ranks` is how the connected attention data-parallel
/// ranks are reached ([`RankIndices`]), which places the request as the
/// engine's own dispatcher would ([`rendezvous_rank`]).
///
/// The disaggregation fields ride as the Python servicer forwards them: the
/// prefill/decode rendezvous (`kv_bootstrap_info`) onto the request's
/// bootstrap slots, an EPD request's per-item encode rendezvous
/// (`encode_bootstrap_info`) onto each named item, and an item without an
/// `encoder_input` keeps its metadata with no feature when its embedding
/// arrives out of band (an encode worker's, or the prefill side's KV).
pub(crate) fn translate_request_tokenspeed_for(
    req: tokenspeed_proto::GenerateRequest,
    grid_tensors: GridTensors,
    max_model_len: u64,
    ranks: &RankIndices,
) -> Result<TokenizedGenerateReqInput, String> {
    let mut input_ids = match req.tokenized {
        Some(tokenized) => tokenized.input_ids,
        None => {
            return Err("ZMQ mode requires pre-tokenized input; no input provided".to_string());
        }
    };
    // PD: the gateway sends the same rendezvous to both sides; the engine
    // keys the transfer on the room and reaches the prefill's bootstrap
    // server at host:port. The Python servicer's rule: an empty host is
    // None, the port and the room ride as given.
    let (bootstrap_host, bootstrap_port, bootstrap_room) = match req.kv_bootstrap_info {
        Some(info) => (
            (!info.bootstrap_host.is_empty()).then_some(info.bootstrap_host),
            Some(info.bootstrap_port),
            Some(info.bootstrap_room),
        ),
        None => (None, None, None),
    };
    let data_parallel_rank = match rendezvous_rank(
        bootstrap_room,
        req.data_parallel_rank,
        ranks.len(),
        &req.request_id,
    )? {
        Some(rank) => Some(ranks.index_of(rank).ok_or_else(|| {
            format!("no connected attention data-parallel rank {rank} to place the request on")
        })?),
        None => None,
    };
    let (input_ids_unpadded, mut multimodal_inputs) = match req.mm_inputs {
        Some(mm) => {
            let (unpadded, inputs) =
                translate_tokenspeed_multimodal(mm, &mut input_ids, grid_tensors)?;
            (Some(unpadded), Some(inputs))
        }
        None => (None, None),
    };
    // EPD: per-item encode->prefill rendezvous, attached onto its item so it
    // rides with the item through the engine (the gateway splits the media
    // one item per encode request, so `item_index` indexes the items 1:1).
    if let Some(encode) = req.encode_bootstrap_info {
        let Some(mm) = multimodal_inputs.as_mut() else {
            return Err(
                "GenerateRequest.encode_bootstrap_info present but mm_inputs is missing; \
                        bootstrap info describes items that must be in mm_inputs"
                    .to_string(),
            );
        };
        let n_items = mm.mm_items.len();
        for entry in encode.items {
            let Some(item) = usize::try_from(entry.item_index)
                .ok()
                .and_then(|index| mm.mm_items.get_mut(index))
            else {
                return Err(format!(
                    "EPD encode bootstrap item_index {} out of range for {n_items} mm item(s)",
                    entry.item_index
                ));
            };
            item.encode_handshake = Some(EncodeHandshake {
                bootstrap_host: entry.bootstrap_host,
                bootstrap_port: entry.bootstrap_port,
                bootstrap_room: entry.bootstrap_room,
            });
        }
    }
    // An item without a feature is one whose embedding this worker never
    // computes: an EPD prefill's item (the encode worker named by its
    // handshake ships it) or a decode side's item (the KV comes from the
    // prefill, the item's metadata keeps the prompt the same on both
    // sides). The translator cannot tell the prefill side apart (the
    // rendezvous rides on both legs), so a disaggregated request passes the
    // item on either leg, as the Python servicer does; the gateway sends the
    // pixel-free clone to the decode side only. An aggregated request has
    // nowhere else to get it from.
    if let Some(mm) = multimodal_inputs.as_ref() {
        for (index, item) in mm.mm_items.iter().enumerate() {
            if item.feature.is_none() && item.encode_handshake.is_none() && bootstrap_room.is_none()
            {
                return Err(format!(
                    "multimodal item {index} carried no encoder_input: an item without one \
                     needs an encode bootstrap entry (its embedding arrives from an encode \
                     worker) or a request with kv_bootstrap_info (its KV arrives from the \
                     prefill side)"
                ));
            }
        }
    }
    // The prompt as the scheduler will hold it (the multimodal translation
    // keeps its length). The engine's own bound on the prompt lives in the
    // frontend this wire bypasses; its scheduler admits any length and clamps
    // the output budget, so the window is judged here as on every wire. An
    // unset `max_new_tokens` stays unset: the scheduler caps it at the
    // remaining context itself.
    admit_prompt(input_ids.len(), max_model_len)?;
    // Over the ZMQ wire TokenSpeed returns only the single sampled-token logprob
    // per token: no top-k candidates (`top_logprobs_num > 1`) and no prompt
    // logprobs (`token_ids_logprob`). Reject both rather than silently return
    // fewer than asked. A bare `logprobs: true` (count 0/1) is the plain
    // sampled-token logprob and is wired end-to-end via `return_logprob`.
    if req.top_logprobs_num > 1 {
        return Err("top_logprobs are not supported over the TokenSpeed ZMQ backend".to_string());
    }
    if !req.token_ids_logprob.is_empty() {
        return Err(
            "prompt logprobs are not supported over the TokenSpeed ZMQ backend".to_string(),
        );
    }
    Ok(TokenizedGenerateReqInput {
        rid: req.request_id,
        input_ids,
        sampling_params: req
            .sampling_params
            .map(translate_sampling_tokenspeed)
            .unwrap_or_else(|| {
                let mut params = TokenSpeedSamplingParams::default();
                params.normalize();
                params
            }),
        return_logprob: req.return_logprob,
        stream: req.stream,
        input_ids_unpadded,
        multimodal_inputs,
        bootstrap_host,
        bootstrap_port,
        bootstrap_room,
        data_parallel_rank,
        // Every other field keeps its neutral default (text requests do not
        // even emit the fields after `stream`; the engine fills them from
        // defaults).
        ..TokenizedGenerateReqInput::default()
    })
}

/// How the attention data-parallel ranks are reached over the direct wire:
/// the engine index (the ZMQ identity the connector pins a request by) of
/// each rank, in rank order, read off the ranks' own handshakes rather than
/// the order they connected in. The decode side's receiver looks the KV up
/// on the prefill rank of the room's residue, so a request placed by
/// residue must reach the rank the engine itself numbers that way; a
/// mismatch would be a silent transfer miss. A group whose handshakes do
/// not declare distinct ranks (an engine that sends none reads 0 on every
/// one) is reached by its indices in order, which the headless launch
/// numbers 0..n in rank order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RankIndices(Vec<u32>);

impl RankIndices {
    /// The ranks behind `engines`, each read at its handshake.
    pub(crate) fn of(engines: &[ConnectedEngine]) -> Self {
        let mut declared: Vec<(u32, u32)> = engines
            .iter()
            .filter_map(|engine| {
                Some((
                    engine.ready_response.data_parallel_rank,
                    engine.engine_id.engine_index()?,
                ))
            })
            .collect();
        declared.sort_unstable();
        let permutation = declared.len() == engines.len()
            && declared
                .iter()
                .enumerate()
                .all(|(rank, (declared_rank, _))| u32::try_from(rank) == Ok(*declared_rank));
        if permutation {
            return Self(declared.into_iter().map(|(_, index)| index).collect());
        }
        let mut indices: Vec<u32> = engines
            .iter()
            .filter_map(|engine| engine.engine_id.engine_index())
            .collect();
        indices.sort_unstable();
        Self(indices)
    }

    /// The indices in rank order, as given.
    #[cfg(test)]
    pub(crate) fn new(indices_by_rank: Vec<u32>) -> Self {
        Self(indices_by_rank)
    }

    /// How many ranks there are: the engine's `dp_size` for the residue.
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// The engine index that is rank `rank`, `None` past the last rank.
    pub(crate) fn index_of(&self, rank: u32) -> Option<u32> {
        self.0.get(usize::try_from(rank).ok()?).copied()
    }
}

/// The attention data-parallel rank a request goes to, the engine's own
/// dispatch rule over `ranks` connected ranks (its data-parallel controller
/// and input processor, which the direct wire bypasses): a disaggregated
/// request (one with a `bootstrap_room`) goes to `room % ranks` on the
/// prefill side as on the decode side, since the decode side's receiver
/// locates the KV on the prefill group of that residue; a pin that disagrees
/// is logged and ignored, never refused. Outside disaggregation an in-range
/// pin is honoured and an out-of-range one refused as the engine refuses it.
/// A single rank takes everything, pin or not, and `None` leaves an unpinned
/// request to the least-loaded selection.
pub(crate) fn rendezvous_rank(
    bootstrap_room: Option<i64>,
    pin: Option<i32>,
    ranks: usize,
    request_id: &str,
) -> Result<Option<u32>, String> {
    let Ok(ranks) = u32::try_from(ranks) else {
        return Ok(None);
    };
    if ranks <= 1 {
        return Ok(None);
    }
    if let Some(room) = bootstrap_room {
        let rank = u32::try_from(room.rem_euclid(i64::from(ranks))).unwrap_or(0);
        if let Some(pin) = pin {
            if u32::try_from(pin).ok() != Some(rank) {
                warn!(
                    request_id,
                    data_parallel_rank = pin,
                    bootstrap_room = room,
                    dp_size = ranks,
                    "data_parallel_rank conflicts with the bootstrap_room residue; the room \
                     stays authoritative on a disaggregated request, the pin is ignored"
                );
            }
        }
        return Ok(Some(rank));
    }
    match pin {
        None => Ok(None),
        Some(pin) => match u32::try_from(pin) {
            Ok(rank) if rank < ranks => Ok(Some(rank)),
            _ => Err(format!(
                "data_parallel_rank must be in [0, {ranks}), got {pin}"
            )),
        },
    }
}

/// Map the proto multimodal payload onto the TokenSpeed wire structs, doing
/// the work the engine's `InputProcessor` does on other transports but the
/// msgpack path bypasses: derive each item's pad value from its content hash
/// and substitute it into the placeholder ranges of `input_ids`, and, for the
/// Qwen3.5 lineage, compute the MRoPE position table from the original ids
/// and the items' grids ([`mrope`]). Returns the original ids (for
/// `input_ids_unpadded`, which detokenization reads) and the wire payload.
pub(crate) fn translate_tokenspeed_multimodal(
    mm: tokenspeed_proto::MultimodalInputs,
    input_ids: &mut [u32],
    grid_tensors: GridTensors,
) -> Result<(Vec<u32>, TokenSpeedWireMmInputs), String> {
    if mm.items.is_empty() {
        return Err("multimodal payload carried no items".to_string());
    }
    if grid_tensors == GridTensors::Qwen35PositionsUnavailable {
        return Err(
            "multimodal items of this model need its MRoPE position table, which the launcher's \
             model facts (spatial_merge_size, image_token_id, video_token_id) did not provide; \
             the engine would take 1-D positions silently, so the request is refused"
                .to_string(),
        );
    }
    let unpadded = input_ids.to_vec();
    let mut im_token_id = None;
    let mut video_token_id = None;
    let mut mm_items = Vec::with_capacity(mm.items.len());
    let mut grids = Vec::with_capacity(mm.items.len());
    for item in mm.items {
        // The proto and engine modality enums disagree (proto: AUDIO=2,
        // VIDEO=3; engine: VIDEO=2, AUDIO=3) — translate, never pass through.
        // Unknown values are rejected, mirroring the gRPC servicer's
        // `_modality_from_proto`.
        let modality = match item.modality() {
            common::Modality::Image => TokenSpeedWireModality::Image,
            common::Modality::Video => TokenSpeedWireModality::Video,
            common::Modality::Audio => TokenSpeedWireModality::Audio,
            common::Modality::Unspecified => {
                return Err("multimodal item carried an unspecified modality".to_string());
            }
        };
        if item.content_hash.is_empty() {
            return Err(
                "multimodal item carried no content hash; the engine pad value derives from it"
                    .to_string(),
            );
        }
        // u64 little-endian fold of the leading hash bytes — the same fold the
        // gRPC servicer applies (`int.from_bytes(content_hash[:8], "little")`),
        // so an item hashes identically on both transports.
        let mut hash_bytes = [0u8; 8];
        for (dst, src) in hash_bytes.iter_mut().zip(item.content_hash.iter()) {
            *dst = *src;
        }
        let hash = u64::from_le_bytes(hash_bytes);
        let pad_value = mm_pad_value(modality, hash);

        if item.placeholders.is_empty() {
            return Err("multimodal item carried no placeholders".to_string());
        }
        let mut offsets = Vec::with_capacity(item.placeholders.len());
        for placeholder in &item.placeholders {
            if placeholder.length == 0 {
                return Err("multimodal placeholder length must be > 0".to_string());
            }
            let start = placeholder.offset as usize;
            let end = start + placeholder.length as usize - 1;
            if end >= input_ids.len() {
                return Err(format!(
                    "multimodal placeholder [{start}, {end}] exceeds the {} prompt tokens",
                    input_ids.len()
                ));
            }
            for id in &mut input_ids[start..=end] {
                *id = pad_value;
            }
            offsets.push((start as u64, end as u64));
        }
        match modality {
            TokenSpeedWireModality::Image => {
                im_token_id = im_token_id.or(item.placeholder_token_id);
            }
            TokenSpeedWireModality::Video => {
                video_token_id = video_token_id.or(item.placeholder_token_id);
            }
            TokenSpeedWireModality::Audio => {}
        }

        // No `encoder_input`: the embedding arrives out of band (the caller
        // checks that it does, once the request's rendezvous fields are read).
        let feature = item.encoder_input.map(wire_tensor_tokenspeed).transpose()?;
        // The position rule reads an item's grid rows off the same tensor the
        // engine reads them from, before it joins the wire payload.
        let grid = match grid_tensors {
            GridTensors::Qwen35Positions(_) => GridKind::KEYS
                .iter()
                .find_map(|(key, kind)| {
                    item.model_specific_tensors
                        .get(*key)
                        .map(|tensor| grid_rows(key, tensor).map(|rows| (*kind, rows)))
                })
                .transpose()?,
            GridTensors::Qwen35PositionsUnavailable
            | GridTensors::Positions
            | GridTensors::PatchLayout => None,
        };
        grids.push(grid);
        let mut model_specific_data = BTreeMap::new();
        for (name, tensor) in item.model_specific_tensors {
            model_specific_data.insert(name, wire_tensor_tokenspeed(tensor)?);
        }
        // An MRoPE family this translation has no position rule for (the
        // engine's InputProcessor, bypassed here, is the only other place
        // that computes one), where shipping nil silently degrades image
        // grounding to 1-D positions: the grid tensors are the tell; fail
        // loudly rather than succeed wrong. A model whose grid only lays out
        // the patches (`GridTensors::PatchLayout`) keeps 1-D positions, and
        // its items pass.
        for key in ["image_grid_thw", "video_grid_thw"] {
            if grid_tensors == GridTensors::Positions && model_specific_data.contains_key(key) {
                return Err(format!(
                    "multimodal item carries {key:?}: MRoPE position tensors of this model are \
                     not derivable over the TokenSpeed ZMQ wire (the Qwen3.5 lineage's are); \
                     use the gRPC transport for this model"
                ));
            }
        }
        mm_items.push(TokenSpeedWireMmItem {
            modality,
            hash,
            pad_value,
            offsets,
            feature,
            model_specific_data,
            encode_handshake: None,
        });
    }
    let (mrope_positions, mrope_position_delta, mrope_position_delta_scalar) = match grid_tensors {
        GridTensors::Qwen35Positions(layout) => {
            let items: Vec<GridItem<'_>> = grids
                .into_iter()
                .zip(&mm_items)
                .map(|(grid, item)| GridItem {
                    grid,
                    offsets: &item.offsets,
                })
                .collect();
            let table = mrope::qwen35_positions(layout, &unpadded, &items)?;
            (Some(table.positions), Some(table.delta), Some(table.scalar))
        }
        GridTensors::Qwen35PositionsUnavailable
        | GridTensors::Positions
        | GridTensors::PatchLayout => {
            // Surface the position-tensor gap in the gateway's own logs (once
            // per process): a model outside the computed lineage receives no
            // mrope_positions on this wire.
            static MROPE_POSITIONS_NOTE: std::sync::Once = std::sync::Once::new();
            MROPE_POSITIONS_NOTE.call_once(|| {
                warn!(
                    "TokenSpeed ZMQ multimodal requests of this model carry no mrope_positions; \
                     a model that requires them falls back to 1-D positions engine-side"
                );
            });
            (None, None, None)
        }
    };
    Ok((
        unpadded,
        TokenSpeedWireMmInputs {
            mm_items,
            im_token_id,
            video_token_id,
            mrope_positions,
            mrope_position_delta,
            mrope_position_delta_scalar,
        },
    ))
}

/// Convert an inline proto tensor to the wire `(dtype, shape, data)` tuple,
/// viewing the request's bytes rather than copying them: a tensor of the
/// auxiliary size then travels to the engine as a frame sharing that view.
/// SHM/remote payloads never reach this translate: ZMQ assembly runs with SHM
/// disabled and RDMA staging off.
pub(crate) fn wire_tensor_tokenspeed(
    tensor: tokenspeed_proto::TensorData,
) -> Result<WireTensor, String> {
    let bytes = match tensor.payload {
        Some(tokenspeed_proto::tensor_data::Payload::Inline(bytes)) => bytes,
        Some(_) => {
            return Err(
                "ZMQ multimodal requires inline tensor payloads; got a SHM/remote handle"
                    .to_string(),
            );
        }
        None => return Err("multimodal tensor carried no payload".to_string()),
    };
    let shape: Vec<usize> = tensor.shape.iter().map(|&d| d as usize).collect();
    // The engine views raw bytes as the named dtype, and a malformed frame is
    // silently dropped engine-side (no terminal frame back) — so a byte-count
    // mismatch must fail here, where the client still gets an error.
    if let Some(width) = tensor_dtype_width(&tensor.dtype) {
        let numel =
            checked_numel(&shape).ok_or_else(|| "multimodal tensor shape overflows".to_string())?;
        if numel * width != bytes.len() {
            return Err(format!(
                "multimodal tensor byte length {} does not match dtype {:?} shape {:?}",
                bytes.len(),
                tensor.dtype,
                tensor.shape
            ));
        }
    }
    Ok(WireTensor::from_raw_bytes(tensor.dtype, shape, bytes))
}

/// Byte width of the wire dtypes the gateway emits; `None` for dtypes we do
/// not recognize (passed through unvalidated rather than rejected).
pub(crate) fn tensor_dtype_width(dtype: &str) -> Option<usize> {
    match dtype {
        "float64" | "int64" | "uint64" => Some(8),
        "float32" | "int32" | "uint32" => Some(4),
        "bfloat16" | "float16" | "int16" | "uint16" => Some(2),
        "uint8" | "int8" | "bool" => Some(1),
        _ => None,
    }
}

/// Map TokenSpeed proto sampling params onto the wire `SamplingParams`, in the
/// normalized form: the engine skips its decode-time re-derivation once
/// `is_normalized` is set, so [`TokenSpeedSamplingParams::normalize`] resolves
/// the derived fields (top_k sentinel, greedy collapse) before encoding.
///
/// String `stop` sequences are not forwarded — the token-only engine cannot
/// match them; the router-side stop decoder trims them from the text instead.
pub(crate) fn translate_sampling_tokenspeed(
    sp: tokenspeed_proto::SamplingParams,
) -> TokenSpeedSamplingParams {
    let mut params = TokenSpeedSamplingParams {
        max_new_tokens: sp.max_new_tokens,
        stop_token_ids: (!sp.stop_token_ids.is_empty()).then_some(sp.stop_token_ids),
        temperature: f64::from(sp.temperature.unwrap_or(1.0)),
        top_p: f64::from(sp.top_p.unwrap_or(1.0)),
        // The proto keeps the API convention `-1` = "all tokens" (and unset);
        // `normalize` resolves it to the engine's disabled sentinel.
        top_k: sp.top_k.unwrap_or(-1),
        min_p: f64::from(sp.min_p.unwrap_or(0.0)),
        frequency_penalty: f64::from(sp.frequency_penalty.unwrap_or(0.0)),
        presence_penalty: f64::from(sp.presence_penalty.unwrap_or(0.0)),
        repetition_penalty: f64::from(sp.repetition_penalty.unwrap_or(1.0)),
        min_new_tokens: sp.min_new_tokens,
        ignore_eos: sp.ignore_eos,
        skip_special_tokens: sp.skip_special_tokens,
        spaces_between_special_tokens: sp.spaces_between_special_tokens,
        no_stop_trim: sp.no_stop_trim,
        seed: sp.sampling_seed,
        // Proto `0` means unspecified; TokenSpeed expects at least one sample.
        // n>1 is fanned out before translation, so this is always 1 on the wire.
        n: sp.n.max(1),
        ..TokenSpeedSamplingParams::default()
    };
    apply_tokenspeed_constraint(&mut params, sp.constraint);
    params.normalize();
    params
}

/// Map the proto structured-output `constraint` oneof onto the wire's dedicated
/// fields. The oneof is single-valued, so at most one field is set; the rest
/// stay `None`.
pub(crate) fn apply_tokenspeed_constraint(
    params: &mut TokenSpeedSamplingParams,
    constraint: Option<tokenspeed_proto::sampling_params::Constraint>,
) {
    use tokenspeed_proto::sampling_params::Constraint;
    match constraint {
        Some(Constraint::JsonSchema(schema)) => params.json_schema = Some(schema),
        Some(Constraint::Regex(regex)) => params.regex = Some(regex),
        Some(Constraint::EbnfGrammar(grammar)) => params.ebnf = Some(grammar),
        Some(Constraint::StructuralTag(tag)) => params.structural_tag = Some(tag),
        None => {}
    }
}

/// The request's constraint as the shared refusal reads it
/// ([`crate::structured_outputs::refuse_unenforceable_constraint`]): the
/// proto `constraint` one-of by kind, with its text.
pub(crate) fn constraint_ref(sp: &tokenspeed_proto::SamplingParams) -> Option<ConstraintRef<'_>> {
    use tokenspeed_proto::sampling_params::Constraint;
    Some(match sp.constraint.as_ref()? {
        Constraint::JsonSchema(schema) => ConstraintRef::JsonSchema(schema),
        Constraint::Regex(regex) => ConstraintRef::Regex(regex),
        Constraint::EbnfGrammar(grammar) => ConstraintRef::Ebnf(grammar),
        Constraint::StructuralTag(tag) => ConstraintRef::StructuralTag(tag),
    })
}

/// Normalize a TokenSpeed wire finish-reason string into the canonical set the
/// gateway's response layer exact-matches (`stop`, `length`, `abort`, `error`) —
/// the same set the vLLM path emits via
/// [`finish_reason_str`](crate::vllm::finish_reason_str). TokenSpeed emits
/// `stop`/`length`/`abort`; an unknown value falls back to `stop` with a warning
/// so a non-canonical string never mis-renders downstream.
pub(crate) fn normalize_finish_reason(reason: &str) -> &'static str {
    match reason {
        "stop" => "stop",
        "length" => "length",
        "abort" => "abort",
        "error" => "error",
        other => {
            tracing::warn!(
                finish_reason = other,
                "unknown TokenSpeed finish_reason; defaulting to \"stop\""
            );
            "stop"
        }
    }
}

#[cfg(test)]
mod tests {
    use engine_zmq_client::mock_engine::default_ready_response;
    use openai_protocol::worker::RuntimeType;

    use super::*;
    use crate::test_support::connected_pair;

    /// The translation under the mock scheduler's context window, one rank.
    fn translate_for(
        req: tokenspeed_proto::GenerateRequest,
        grid_tensors: GridTensors,
    ) -> Result<TokenizedGenerateReqInput, String> {
        translate_request_tokenspeed_for(
            req,
            grid_tensors,
            default_ready_response().max_model_len,
            &RankIndices::new(vec![0]),
        )
    }

    /// End-to-end over ipc:// for a TokenSpeed backend: the adapter frames a
    /// tagged `TokenizedGenerateReqInput`, and maps `BatchTokenIDOutSlim`
    /// batches back to vLLM-proto responses. The mock engine speaks the shared
    /// transport with raw frames (it decodes/encodes the TokenSpeed structs
    /// directly).
    #[tokio::test]
    async fn generate_e2e_translates_and_streams_tokenspeed() {
        use engine_zmq_client::{
            codec::{decode_msgpack, encode_msgpack},
            protocol::tokenspeed::{
                output::BatchTokenIDOutSlim,
                request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
            },
        };

        let (client, engine, _dir) = connected_pair(RuntimeType::TokenSpeed).await;

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let frames = input.recv_frames().await.unwrap();
            assert_eq!(
                TokenSpeedRequestType::from_frame(frames[0].as_ref()),
                Some(TokenSpeedRequestType::Add)
            );
            let request: TokenizedGenerateReqInput = decode_msgpack(frames[1].as_ref()).unwrap();
            assert_eq!(request.rid, "r1");
            assert_eq!(request.input_ids, vec![1, 2, 3]);
            assert_eq!(request.sampling_params.max_new_tokens, Some(2));
            // The adapter always emits the normalized sampling form.
            assert!(request.sampling_params.is_normalized);
            // A plain sampled-token logprob request (logprobs=1) sets the flag.
            assert!(request.return_logprob);

            let chunk = BatchTokenIDOutSlim {
                rids: vec!["r1".into()],
                output_ids: vec![vec![10]],
                finished_reasons: vec![String::new()],
                prompt_tokens: vec![3],
                completion_tokens: vec![1],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-0.5]],
                output_token_logprobs_idx: vec![vec![10]],
                ..Default::default()
            };
            let done = BatchTokenIDOutSlim {
                rids: vec!["r1".into()],
                output_ids: vec![vec![11]],
                finished_reasons: vec!["length".into()],
                prompt_tokens: vec![3],
                completion_tokens: vec![2],
                cached_tokens: vec![0],
                output_token_logprobs_val: vec![vec![-1.25]],
                output_token_logprobs_idx: vec![vec![11]],
                ..Default::default()
            };
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&chunk).unwrap())])
                .await
                .unwrap();
            output
                .send_frames(vec![bytes::Bytes::from(encode_msgpack(&done).unwrap())])
                .await
                .unwrap();
            // Keep the peer alive until the frontend has consumed the replies.
            (input, output)
        });

        let req = tokenspeed_proto::GenerateRequest {
            request_id: "r1".to_string(),
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(tokenspeed_proto::SamplingParams {
                max_new_tokens: Some(2),
                ..Default::default()
            }),
            // Plain sampled-token logprob; must be wired through.
            return_logprob: true,
            stream: true,
            ..Default::default()
        };
        let mut stream = client.generate_tokenspeed(req).await.expect("generate");

        let first = stream.next().await.expect("chunk item").expect("chunk ok");
        match first.response {
            Some(vllm::generate_response::Response::Chunk(chunk)) => {
                assert_eq!(chunk.token_ids, vec![10]);
                assert_eq!(chunk.prompt_tokens, 3);
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
                // Chunks carry the tick's incremental logprobs; the terminal
                // `Complete` carries the cumulative set, parallel to `output_ids`
                // (the non-streaming renderer reads only the `Complete`).
                let logprobs = complete.output_logprobs.expect("complete logprobs");
                assert_eq!(logprobs.token_logprobs, vec![-0.5, -1.25]);
                assert_eq!(logprobs.token_ids, vec![10, 11]);
            }
            other => panic!("expected complete, got {other:?}"),
        }
        assert!(stream.next().await.is_none());

        engine_task.await.unwrap();
    }

    #[test]
    fn tokenspeed_sampling_maps_top_k_sentinel_and_floors_n() {
        use engine_zmq_client::protocol::tokenspeed::sampling::TOP_K_DISABLED;

        // Unset top_k rides the API convention `-1` ("all tokens") and normalizes
        // to the engine's disabled sentinel; n=0 floors to 1; max_new_tokens
        // forwards.
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            top_k: None,
            sampling_seed: Some(1_234),
            n: 0,
            max_new_tokens: Some(8),
            ..Default::default()
        });
        assert_eq!(mapped.top_k, TOP_K_DISABLED);
        assert_eq!(mapped.seed, Some(1_234));
        assert_eq!(mapped.n, 1);
        assert_eq!(mapped.max_new_tokens, Some(8));
        // The wire form is always normalized (the engine skips re-derivation).
        assert!(mapped.is_normalized);

        // An explicit top_k passes through unchanged.
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            top_k: Some(40),
            ..Default::default()
        });
        assert_eq!(mapped.top_k, 40);

        // A near-zero temperature collapses to greedy on the wire, with the
        // seed pinned to 0 as the engine pins it (a request seed included).
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            temperature: Some(0.0),
            sampling_seed: Some(1_234),
            ..Default::default()
        });
        assert_eq!(mapped.temperature, 1.0);
        assert_eq!(mapped.top_k, 1);
        assert_eq!(mapped.seed, Some(0));

        // Empty stop_token_ids ride as None (the normalized encoding).
        let mapped = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams::default());
        assert_eq!(mapped.stop_token_ids, None);
    }

    fn ts_tokenized_req(
        sampling: tokenspeed_proto::SamplingParams,
    ) -> tokenspeed_proto::GenerateRequest {
        tokenspeed_proto::GenerateRequest {
            request_id: "r1".to_string(),
            tokenized: Some(tokenspeed_proto::TokenizedInput {
                input_ids: vec![1, 2, 3],
                original_text: String::new(),
            }),
            sampling_params: Some(sampling),
            stream: true,
            ..Default::default()
        }
    }

    #[test]
    fn tokenspeed_return_logprob_flag_passes_through() {
        // The request-level `return_logprob` drives the plain sampled-token
        // logprob (count 0/1 in `top_logprobs_num` is the same case).
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.return_logprob = true;
        let wire = translate_for(req, GridTensors::Positions).expect("return_logprob accepted");
        assert!(wire.return_logprob);

        // Unset -> the flag stays false.
        let wire = translate_for(
            ts_tokenized_req(tokenspeed_proto::SamplingParams::default()),
            GridTensors::Positions,
        )
        .expect("no logprobs accepted");
        assert!(!wire.return_logprob);
    }

    #[test]
    fn tokenspeed_rejects_top_logprobs_and_prompt_logprobs() {
        // Top-k logprobs (count > 1) cannot be honored over the wire.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.top_logprobs_num = 5;
        assert!(translate_for(req, GridTensors::Positions).is_err());

        // Prompt (input) logprobs cannot be produced.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.token_ids_logprob = vec![1, 2];
        assert!(translate_for(req, GridTensors::Positions).is_err());

        // A bare count of 0/1 is the plain sampled-token case: accepted.
        for count in [0, 1] {
            let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
            req.top_logprobs_num = count;
            assert!(translate_for(req, GridTensors::Positions).is_ok());
        }
    }

    fn kv_bootstrap(host: &str, port: i32, room: i64) -> tokenspeed_proto::KvBootstrapInfo {
        tokenspeed_proto::KvBootstrapInfo {
            bootstrap_host: host.to_string(),
            bootstrap_port: port,
            bootstrap_room: room,
        }
    }

    /// The prefill/decode rendezvous rides onto the request's bootstrap slots
    /// with the Python servicer's rule (an empty host is None, the port and
    /// the room as given, 0 kept); an aggregated request carries none.
    #[test]
    fn tokenspeed_carries_the_kv_bootstrap_rendezvous_as_the_python_servicer_does() {
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.kv_bootstrap_info = Some(kv_bootstrap("10.0.0.7", 8998, 41));
        let wire = translate_for(req, GridTensors::Positions).expect("a disaggregated request");
        assert_eq!(wire.bootstrap_host.as_deref(), Some("10.0.0.7"));
        assert_eq!(wire.bootstrap_port, Some(8998));
        assert_eq!(wire.bootstrap_room, Some(41));

        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.kv_bootstrap_info = Some(kv_bootstrap("", 0, 0));
        let wire = translate_for(req, GridTensors::Positions).expect("zeroes ride as given");
        assert_eq!(
            (
                wire.bootstrap_host,
                wire.bootstrap_port,
                wire.bootstrap_room
            ),
            (None, Some(0), Some(0))
        );

        let wire = translate_for(
            ts_tokenized_req(tokenspeed_proto::SamplingParams::default()),
            GridTensors::Positions,
        )
        .expect("an aggregated request");
        assert_eq!(
            (
                wire.bootstrap_host,
                wire.bootstrap_port,
                wire.bootstrap_room
            ),
            (None, None, None)
        );
        assert_eq!(wire.data_parallel_rank, None);
    }

    /// Placement is the engine's own dispatch rule: a disaggregated request
    /// goes to its room's residue on either side (a disagreeing pin is
    /// ignored), an aggregated one to its pin when in range (out of range is
    /// the engine's refusal), a single rank takes everything, and an
    /// unpinned aggregated request is left to the least-loaded selection.
    #[test]
    fn tokenspeed_places_a_request_as_the_engines_dispatcher_would() {
        assert_eq!(rendezvous_rank(Some(7), None, 2, "r").unwrap(), Some(1));
        assert_eq!(rendezvous_rank(Some(8), None, 2, "r").unwrap(), Some(0));
        assert_eq!(
            rendezvous_rank(Some(7), Some(0), 2, "r").unwrap(),
            Some(1),
            "the room stays authoritative over the pin"
        );
        assert_eq!(
            rendezvous_rank(Some(-3), None, 4, "r").unwrap(),
            Some(1),
            "a negative room folds as Python's % does"
        );
        assert_eq!(rendezvous_rank(Some(7), Some(1), 1, "r").unwrap(), None);
        assert_eq!(rendezvous_rank(None, Some(1), 2, "r").unwrap(), Some(1));
        assert_eq!(rendezvous_rank(None, None, 2, "r").unwrap(), None);
        assert_eq!(
            rendezvous_rank(None, Some(5), 1, "r").unwrap(),
            None,
            "without attention DP the pin is meaningless and dropped"
        );
        assert_eq!(
            rendezvous_rank(None, Some(2), 2, "r").unwrap_err(),
            "data_parallel_rank must be in [0, 2), got 2"
        );
        assert!(rendezvous_rank(None, Some(-1), 2, "r").is_err());

        // Through the translation, over two ranks whose engine indices are
        // the reverse of their ranks (rank 0 dialled in as index 1): the
        // request is pinned to the INDEX of the rank it belongs on.
        let window = default_ready_response().max_model_len;
        let reversed = RankIndices::new(vec![1, 0]);
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.data_parallel_rank = Some(1);
        let wire = translate_request_tokenspeed_for(req, GridTensors::Positions, window, &reversed)
            .expect("an in-range pin");
        assert_eq!(wire.data_parallel_rank, Some(0), "rank 1 is index 0");
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.data_parallel_rank = Some(1);
        req.kv_bootstrap_info = Some(kv_bootstrap("10.0.0.7", 8998, 4));
        let wire = translate_request_tokenspeed_for(req, GridTensors::Positions, window, &reversed)
            .expect("a disaggregated request");
        assert_eq!(
            wire.data_parallel_rank,
            Some(1),
            "room 4 is rank 0 on two ranks, which is index 1"
        );
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.data_parallel_rank = Some(2);
        let error =
            translate_request_tokenspeed_for(req, GridTensors::Positions, window, &reversed)
                .expect_err("an out-of-range pin");
        assert!(
            error.contains("data_parallel_rank must be in [0, 2)"),
            "{error}"
        );
    }

    /// The ranks are read off the engines' own handshakes: an engine that
    /// dialled in as index 0 but declares rank 1 IS rank 1 (the decode
    /// receiver numbers the prefill ranks as the engine does); a group that
    /// declares no ranks (every handshake reads 0) is reached by its indices
    /// in order; an engine with no index is not a rank.
    #[test]
    fn rank_indices_follow_the_engines_declared_ranks_not_the_connection_order() {
        use engine_zmq_client::{protocol::handshake::EngineCoreReadyResponse, EngineId};

        let engine = |index: u32, declared_rank: u32| ConnectedEngine {
            engine_id: EngineId::from_engine_index(index),
            ready_response: EngineCoreReadyResponse {
                data_parallel_rank: declared_rank,
                ..default_ready_response()
            },
        };
        let swapped = RankIndices::of(&[engine(0, 1), engine(1, 0)]);
        assert_eq!(swapped, RankIndices::new(vec![1, 0]));
        assert_eq!(
            (swapped.len(), swapped.index_of(1), swapped.index_of(2)),
            (2, Some(0), None)
        );
        assert_eq!(
            RankIndices::of(&[engine(1, 1), engine(0, 0)]),
            RankIndices::new(vec![0, 1]),
            "connection order does not matter"
        );
        assert_eq!(
            RankIndices::of(&[engine(0, 0), engine(1, 0)]),
            RankIndices::new(vec![0, 1]),
            "no declared ranks: the indices in order"
        );
        assert_eq!(
            RankIndices::of(&[engine(4, 0), engine(3, 1)]),
            RankIndices::new(vec![4, 3]),
            "declared ranks over arbitrary indices"
        );
        assert_eq!(RankIndices::of(&[]).len(), 0);
    }

    /// An EPD prefill request: each named item carries its encode rendezvous
    /// and no feature (the encode worker ships the embedding); an item not
    /// named keeps its own; the prompt is padded for both as always.
    #[test]
    fn tokenspeed_attaches_each_items_encode_rendezvous() {
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.tokenized.as_mut().unwrap().input_ids = vec![10, 20, 30, 40, 50, 60];
        let mut first = ts_mm_item(&1u64.to_le_bytes(), 1, 2);
        first.encoder_input = None;
        let second = ts_mm_item(&2u64.to_le_bytes(), 3, 2);
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![first, second],
        });
        req.encode_bootstrap_info = Some(tokenspeed_proto::EncodeBootstrapInfo {
            items: vec![tokenspeed_proto::EncodeItemBootstrapInfo {
                item_index: 0,
                bootstrap_host: "10.0.0.9".to_string(),
                bootstrap_port: 8998,
                bootstrap_room: 41,
            }],
        });
        let wire = translate_for(req, GridTensors::Positions).expect("an EPD prefill request");
        let pad1 = mm_pad_value(TokenSpeedWireModality::Image, 1);
        let pad2 = mm_pad_value(TokenSpeedWireModality::Image, 2);
        assert_eq!(wire.input_ids, vec![10, pad1, pad1, pad2, pad2, 60]);
        let mm = wire.multimodal_inputs.expect("mm payload");
        assert_eq!(
            mm.mm_items[0].encode_handshake,
            Some(EncodeHandshake {
                bootstrap_host: "10.0.0.9".to_string(),
                bootstrap_port: 8998,
                bootstrap_room: 41,
            })
        );
        assert!(mm.mm_items[0].feature.is_none());
        assert_eq!(
            mm.mm_items[0].model_specific_data["vit_grid"].dtype,
            "uint32"
        );
        assert_eq!(mm.mm_items[1].encode_handshake, None);
        assert!(mm.mm_items[1].feature.is_some());
    }

    /// The two refusals the Python servicer raises, in its words: an encode
    /// rendezvous on a request without items, and one naming an item index
    /// outside the items.
    #[test]
    fn tokenspeed_refuses_an_encode_rendezvous_that_names_no_item() {
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.encode_bootstrap_info = Some(tokenspeed_proto::EncodeBootstrapInfo { items: vec![] });
        let error = translate_for(req, GridTensors::Positions).unwrap_err();
        assert!(
            error.contains("encode_bootstrap_info present but mm_inputs is missing"),
            "{error}"
        );

        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.tokenized.as_mut().unwrap().input_ids = vec![10, 20, 30, 40, 50];
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![ts_mm_item(&1u64.to_le_bytes(), 1, 3)],
        });
        req.encode_bootstrap_info = Some(tokenspeed_proto::EncodeBootstrapInfo {
            items: vec![tokenspeed_proto::EncodeItemBootstrapInfo {
                item_index: 1,
                bootstrap_host: "10.0.0.9".to_string(),
                bootstrap_port: 8998,
                bootstrap_room: 41,
            }],
        });
        let error = translate_for(req, GridTensors::Positions).unwrap_err();
        assert_eq!(
            error,
            "EPD encode bootstrap item_index 1 out of range for 1 mm item(s)"
        );
    }

    /// An item without an encoder input passes on a disaggregated side (the
    /// pixel-less decode clone: its KV comes from the prefill) with its
    /// metadata intact, and is refused on an aggregated request, which has
    /// nowhere else to get the embedding from.
    #[test]
    fn tokenspeed_accepts_a_feature_less_item_only_where_its_embedding_arrives_out_of_band() {
        let feature_less = || {
            let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
            req.tokenized.as_mut().unwrap().input_ids = vec![10, 20, 30, 40, 50];
            let mut item = ts_mm_item(&1u64.to_le_bytes(), 1, 3);
            item.encoder_input = None;
            req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs { items: vec![item] });
            req
        };
        let error = translate_for(feature_less(), GridTensors::Positions).unwrap_err();
        assert!(error.contains("carried no encoder_input"), "{error}");

        let mut req = feature_less();
        req.kv_bootstrap_info = Some(kv_bootstrap("10.0.0.7", 8998, 41));
        let wire = translate_for(req, GridTensors::Positions).expect("the decode side's clone");
        let pad = mm_pad_value(TokenSpeedWireModality::Image, 1);
        assert_eq!(wire.input_ids, vec![10, pad, pad, pad, 50]);
        let mm = wire.multimodal_inputs.expect("mm payload");
        let item = &mm.mm_items[0];
        assert!(item.feature.is_none() && item.encode_handshake.is_none());
        assert_eq!(item.offsets, vec![(1, 3)]);
        assert_eq!(item.model_specific_data["vit_grid"].dtype, "uint32");
    }

    #[test]
    fn tokenspeed_maps_structured_output_constraints() {
        // The `constraint` oneof maps 1:1 onto the wire's dedicated fields; the
        // oneof is single-valued, so the other three stay unset.
        use tokenspeed_proto::sampling_params::Constraint;

        let json = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::JsonSchema("{\"type\":\"object\"}".into())),
            ..Default::default()
        });
        assert_eq!(json.json_schema.as_deref(), Some("{\"type\":\"object\"}"));
        assert_eq!(json.regex, None);
        assert_eq!(json.ebnf, None);
        assert_eq!(json.structural_tag, None);

        let regex = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::Regex("[0-9]+".into())),
            ..Default::default()
        });
        assert_eq!(regex.regex.as_deref(), Some("[0-9]+"));
        assert_eq!(regex.json_schema, None);

        let ebnf = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::EbnfGrammar("root ::= \"a\"".into())),
            ..Default::default()
        });
        assert_eq!(ebnf.ebnf.as_deref(), Some("root ::= \"a\""));

        let tag = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams {
            constraint: Some(Constraint::StructuralTag("<tag>".into())),
            ..Default::default()
        });
        assert_eq!(tag.structural_tag.as_deref(), Some("<tag>"));

        // No constraint leaves all four structured-output fields unset.
        let none = translate_sampling_tokenspeed(tokenspeed_proto::SamplingParams::default());
        assert_eq!(none.json_schema, None);
        assert_eq!(none.regex, None);
        assert_eq!(none.ebnf, None);
        assert_eq!(none.structural_tag, None);
    }

    #[test]
    fn tokenspeed_forwards_stop_token_ids_and_drops_stop_strings() {
        // String stops are resolved upstream; any that reach here are dropped
        // (the token-only engine cannot match them) while stop token ids ride
        // through and the router-side decoder trims residual text.
        let req = translate_for(
            ts_tokenized_req(tokenspeed_proto::SamplingParams {
                stop: vec!["</s>".to_string()],
                stop_token_ids: vec![13],
                ..Default::default()
            }),
            GridTensors::Positions,
        )
        .expect("residual stop strings must not be rejected");
        assert_eq!(req.sampling_params.stop_token_ids, Some(vec![13]));
        assert_eq!(req.sampling_params.stop, None);
    }

    fn ts_mm_item(hash: &[u8], offset: u32, length: u32) -> tokenspeed_proto::MultimodalItem {
        tokenspeed_proto::MultimodalItem {
            modality: common::Modality::Image as i32,
            content_hash: hash.to_vec(),
            encoder_input: Some(tokenspeed_proto::TensorData {
                shape: vec![2, 4],
                dtype: "bfloat16".to_string(),
                payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(
                    vec![0u8; 16].into(),
                )),
            }),
            model_specific_tensors: [(
                "vit_grid".to_string(),
                tokenspeed_proto::TensorData {
                    shape: vec![1, 3],
                    dtype: "uint32".to_string(),
                    payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(
                        vec![0u8; 12].into(),
                    )),
                },
            )]
            .into(),
            placeholders: vec![tokenspeed_proto::PlaceholderRange { offset, length }],
            placeholder_token_id: Some(9),
        }
    }

    #[test]
    fn tokenspeed_translates_multimodal_inputs() {
        // The translate does the engine InputProcessor's job (bypassed on the
        // msgpack path): pad-value substitution into input_ids, unpadded ids
        // preserved, offsets converted to inclusive [start, end] pairs.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.tokenized.as_mut().unwrap().input_ids = vec![10, 20, 30, 40, 50];
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![ts_mm_item(&0xDEAD_BEEFu64.to_le_bytes(), 1, 3)],
        });
        let wire = translate_for(req, GridTensors::Positions).expect("translated");

        let expected_pad = mm_pad_value(TokenSpeedWireModality::Image, 0xDEAD_BEEF);
        assert_eq!(
            wire.input_ids,
            vec![10, expected_pad, expected_pad, expected_pad, 50]
        );
        assert_eq!(wire.input_ids_unpadded, Some(vec![10, 20, 30, 40, 50]));

        let mm = wire.multimodal_inputs.expect("mm payload");
        assert_eq!(mm.im_token_id, Some(9));
        assert_eq!(mm.video_token_id, None);
        assert_eq!(mm.mm_items.len(), 1);
        let item = &mm.mm_items[0];
        assert_eq!(item.modality, TokenSpeedWireModality::Image);
        assert_eq!(item.hash, 0xDEAD_BEEF);
        assert_eq!(item.pad_value, expected_pad);
        assert_eq!(item.offsets, vec![(1, 3)]);
        assert_eq!(item.model_specific_data["vit_grid"].dtype, "uint32");
    }

    /// Prompt bounds apply before the scheduler, with or without an output
    /// limit, under the window the handshake reported; the request's own
    /// limit is kept as it is (an unset one stays the scheduler's to cap).
    #[test]
    fn tokenspeed_validates_prompt_length_even_with_explicit_max_new_tokens() {
        for sampling in [
            None,
            Some(tokenspeed_proto::SamplingParams::default()),
            Some(tokenspeed_proto::SamplingParams {
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
                let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
                req.sampling_params = sampling.clone();
                req.tokenized.as_mut().unwrap().input_ids = vec![1; prompt_len];
                let result = translate_request_tokenspeed_for(
                    req,
                    GridTensors::Positions,
                    max_model_len,
                    &RankIndices::new(vec![0]),
                );
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "prompt_len={prompt_len}, max_model_len={max_model_len}, max_new_tokens={:?}",
                    sampling.as_ref().and_then(|sp| sp.max_new_tokens)
                );
                match result {
                    Err(error) => assert!(error.contains("prompt"), "{error}"),
                    Ok(wire) => assert_eq!(
                        wire.sampling_params.max_new_tokens,
                        sampling.as_ref().and_then(|sp| sp.max_new_tokens)
                    ),
                }
            }
        }
    }

    /// A multimodal prompt is judged as the scheduler will hold it, after the
    /// pad substitution (which keeps its length): one that fills the window
    /// is refused with its items in order, one position short of it passes.
    #[test]
    fn tokenspeed_judges_a_multimodal_prompt_after_its_translation() {
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.tokenized.as_mut().unwrap().input_ids = vec![10, 20, 30, 40, 50];
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![ts_mm_item(&0xDEAD_BEEFu64.to_le_bytes(), 1, 3)],
        });
        let error = translate_request_tokenspeed_for(
            req.clone(),
            GridTensors::Positions,
            5,
            &RankIndices::new(vec![0]),
        )
        .expect_err("the prompt fills the window");
        assert!(error.contains("length 5"), "{error}");
        assert!(error.contains("model length of 5"), "{error}");
        let wire = translate_request_tokenspeed_for(
            req,
            GridTensors::Positions,
            6,
            &RankIndices::new(vec![0]),
        )
        .expect("one position left for output");
        assert_eq!(wire.input_ids.len(), 5);
        assert!(wire.multimodal_inputs.is_some());
    }

    #[test]
    fn tokenspeed_translates_audio_items_as_the_engine_reads_them() {
        // An audio item: the waveform rows travel as they were written
        // (bfloat16 [samples, 1]) with the per-clip row count beside them;
        // the pad value carries the audio tag; no token id is collected.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.tokenized.as_mut().unwrap().input_ids = vec![1, 7, 100, 100, 100, 8, 2];
        let samples: u32 = 1_920;
        let item = tokenspeed_proto::MultimodalItem {
            modality: common::Modality::Audio as i32,
            content_hash: 0x0123_4567_89AB_CDEFu64.to_le_bytes().to_vec(),
            encoder_input: Some(tokenspeed_proto::TensorData {
                shape: vec![samples, 1],
                dtype: "bfloat16".to_string(),
                payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(
                    vec![0x3Fu8; samples as usize * 2].into(),
                )),
            }),
            model_specific_tensors: [(
                "samples_per_audio".to_string(),
                tokenspeed_proto::TensorData {
                    shape: vec![1],
                    dtype: "int64".to_string(),
                    payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(
                        (samples as i64).to_le_bytes().to_vec().into(),
                    )),
                },
            )]
            .into(),
            placeholders: vec![tokenspeed_proto::PlaceholderRange {
                offset: 2,
                length: 3,
            }],
            placeholder_token_id: Some(100),
        };
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs { items: vec![item] });
        let wire = translate_for(req, GridTensors::Positions).expect("translated");

        let expected_pad = mm_pad_value(TokenSpeedWireModality::Audio, 0x0123_4567_89AB_CDEF);
        assert_ne!(
            expected_pad,
            mm_pad_value(TokenSpeedWireModality::Image, 0x0123_4567_89AB_CDEF)
        );
        assert_eq!(
            wire.input_ids,
            vec![1, 7, expected_pad, expected_pad, expected_pad, 8, 2]
        );
        let mm = wire.multimodal_inputs.expect("mm payload");
        assert_eq!(mm.im_token_id, None);
        assert_eq!(mm.video_token_id, None);
        let item = &mm.mm_items[0];
        assert_eq!(item.modality, TokenSpeedWireModality::Audio);
        assert_eq!(item.pad_value, expected_pad);
        assert_eq!(item.offsets, vec![(2, 4)]);
        let feature = item
            .feature
            .as_ref()
            .expect("an audio item carries its samples");
        assert_eq!(feature.dtype, "bfloat16");
        assert_eq!(feature.shape, vec![samples as usize, 1]);
        let rows = &item.model_specific_data["samples_per_audio"];
        assert_eq!(rows.dtype, "int64");
        assert_eq!(rows.shape, vec![1]);
    }

    #[test]
    fn tokenspeed_multimodal_rejects_bad_items() {
        // Out-of-bounds placeholder: prompt has 3 tokens, range needs 4.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![ts_mm_item(b"12345678", 1, 3)],
        });
        assert!(translate_for(req, GridTensors::Positions)
            .unwrap_err()
            .contains("exceeds"));

        // Missing content hash: pad value cannot be derived.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![ts_mm_item(b"", 0, 1)],
        });
        assert!(translate_for(req, GridTensors::Positions)
            .unwrap_err()
            .contains("content hash"));

        // Byte-length mismatch would be silently dropped engine-side; the
        // translate must catch it while the client can still see an error.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        let mut item = ts_mm_item(b"12345678", 0, 1);
        item.encoder_input.as_mut().unwrap().shape = vec![3, 4];
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs { items: vec![item] });
        assert!(translate_for(req, GridTensors::Positions)
            .unwrap_err()
            .contains("byte length"));

        // Unspecified modality is rejected, mirroring the gRPC servicer.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        let mut item = ts_mm_item(b"12345678", 0, 1);
        item.modality = common::Modality::Unspecified as i32;
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs { items: vec![item] });
        assert!(translate_for(req, GridTensors::Positions)
            .unwrap_err()
            .contains("modality"));

        // Known MRoPE families fail loudly: nothing derives mrope_positions
        // on this wire, and 1-D fallback would silently degrade grounding.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        let mut item = ts_mm_item(b"12345678", 0, 1);
        item.model_specific_tensors.insert(
            "image_grid_thw".to_string(),
            tokenspeed_proto::TensorData {
                shape: vec![1, 3],
                dtype: "uint32".to_string(),
                payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(
                    vec![0u8; 12].into(),
                )),
            },
        );
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs { items: vec![item] });
        assert!(translate_for(req, GridTensors::Positions)
            .unwrap_err()
            .contains("MRoPE"));
    }

    #[test]
    fn a_patch_layout_grid_passes_for_a_model_whose_positions_are_one_dimensional() {
        let grid = tokenspeed_proto::TensorData {
            shape: vec![1, 3],
            dtype: "uint32".to_string(),
            payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(
                vec![0u8; 12].into(),
            )),
        };
        let with_grid = || {
            let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
            let mut item = ts_mm_item(b"12345678", 0, 1);
            item.model_specific_tensors
                .insert("image_grid_thw".to_string(), grid.clone());
            req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs { items: vec![item] });
            req
        };
        // The model's rope facts decide: a model whose grid only lays out the
        // patches (1-D positions) keeps the tensor on the wire; one that
        // needs MRoPE positions by a rule this wire lacks, or a launcher that
        // does not say, is refused as before.
        let outside = vec!["MuseSparkForConditionalGeneration".to_string()];
        assert_eq!(
            GridTensors::for_model(&outside, "muse_spark", Some(false), None),
            GridTensors::PatchLayout
        );
        assert_eq!(
            GridTensors::for_model(&outside, "muse_spark", Some(true), None),
            GridTensors::Positions
        );
        assert_eq!(
            GridTensors::for_model(&outside, "muse_spark", None, None),
            GridTensors::Positions
        );
        let wire = translate_for(with_grid(), GridTensors::PatchLayout)
            .expect("a patch-layout grid passes");
        let items = &wire
            .multimodal_inputs
            .expect("multimodal inputs on the wire")
            .mm_items;
        assert!(items[0].model_specific_data.contains_key("image_grid_thw"));
        assert!(translate_for(with_grid(), GridTensors::Positions)
            .unwrap_err()
            .contains("MRoPE"));
    }

    /// The engine's own test checkpoint facts (`test/runtime/test_multimodal_mrope.py`).
    const QWEN35: MropeLayout = MropeLayout {
        spatial_merge_size: 2,
        image_token_id: 151655,
        video_token_id: 151656,
    };

    fn qwen35_grid_item(
        rows: &[[i64; 3]],
        offset: u32,
        length: u32,
    ) -> tokenspeed_proto::MultimodalItem {
        let mut item = ts_mm_item(b"12345678", offset, length);
        item.placeholder_token_id = Some(QWEN35.image_token_id);
        item.model_specific_tensors.insert(
            "image_grid_thw".to_string(),
            tokenspeed_proto::TensorData {
                shape: vec![rows.len() as u32, 3],
                dtype: "int64".to_string(),
                payload: Some(tokenspeed_proto::tensor_data::Payload::Inline(
                    rows.iter()
                        .flatten()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>()
                        .into(),
                )),
            },
        );
        item
    }

    fn i64_values(tensor: &WireTensor) -> Vec<i64> {
        tensor
            .data
            .as_raw_view()
            .expect("inline")
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect()
    }

    /// A Qwen3.5-lineage checkpoint's grid feeds the position table this
    /// translation computes from the ORIGINAL ids (the engine's rule checks
    /// the placeholder spans before the pad values replace them): the
    /// engine's own image case, its table in slot 3, the delta in slot 4 and
    /// as the scalar in slot 5, while the ids on the wire carry the pad value.
    #[test]
    fn a_qwen35_grid_feeds_the_position_table_the_engine_would_compute() {
        let lineage = vec!["Qwen3_5ForConditionalGeneration".to_string()];
        let rule = GridTensors::for_model(&lineage, "qwen3_5", Some(true), Some(QWEN35));
        assert_eq!(rule, GridTensors::Qwen35Positions(QWEN35));
        // Without the layout facts the lineage's items are refused outright:
        // with or without a grid, never empty position slots for this lineage.
        assert_eq!(
            GridTensors::for_model(&lineage, "qwen3_5", Some(true), None),
            GridTensors::Qwen35PositionsUnavailable
        );
        for item in [
            ts_mm_item(b"12345678", 1, 3),
            qwen35_grid_item(&[[1, 4, 6]], 2, 6),
        ] {
            let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
            req.tokenized.as_mut().unwrap().input_ids = vec![QWEN35.image_token_id; 9];
            req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs { items: vec![item] });
            let err = translate_for(req, GridTensors::Qwen35PositionsUnavailable).unwrap_err();
            assert!(err.contains("did not provide"), "{err}");
        }

        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        let mut ids = vec![10, 151652];
        ids.extend([QWEN35.image_token_id; 6]);
        ids.push(11);
        req.tokenized.as_mut().unwrap().input_ids = ids.clone();
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![qwen35_grid_item(&[[1, 4, 6]], 2, 6)],
        });
        let wire = translate_for(req, rule).expect("the lineage's grid is served");
        let pad = mm_pad_value(
            TokenSpeedWireModality::Image,
            u64::from_le_bytes(*b"12345678"),
        );
        assert_eq!(&wire.input_ids[2..8], &[pad; 6]);
        assert_eq!(wire.input_ids_unpadded, Some(ids));
        let mm = wire.multimodal_inputs.expect("mm payload");
        assert!(mm.mm_items[0]
            .model_specific_data
            .contains_key("image_grid_thw"));
        let positions = mm.mrope_positions.expect("the table");
        assert_eq!(positions.shape, vec![3, 9]);
        assert_eq!(positions.dtype, "int64");
        assert_eq!(
            i64_values(&positions),
            vec![
                0, 1, 2, 2, 2, 2, 2, 2, 5, // time
                0, 1, 2, 2, 2, 3, 3, 3, 5, // height
                0, 1, 2, 3, 4, 2, 3, 4, 5, // width
            ]
        );
        let delta = mm.mrope_position_delta.expect("the delta");
        assert_eq!(delta.shape, vec![1, 1]);
        assert_eq!(i64_values(&delta), vec![-3]);
        assert_eq!(mm.mrope_position_delta_scalar, Some(-3));

        // A span the engine's rule would not accept (its tokens are not the
        // placeholder id) is a refusal naming it, not a guessed table.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.tokenized.as_mut().unwrap().input_ids = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![qwen35_grid_item(&[[1, 4, 6]], 2, 6)],
        });
        let err = translate_for(req, rule).unwrap_err();
        assert!(err.contains("other than the placeholder id"), "{err}");

        // Outside the lineage the same grid is refused as before.
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.tokenized.as_mut().unwrap().input_ids = vec![QWEN35.image_token_id; 9];
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![qwen35_grid_item(&[[1, 4, 6]], 2, 6)],
        });
        assert!(translate_for(req, GridTensors::Positions)
            .unwrap_err()
            .contains("MRoPE"));
    }

    /// A multimodal request of the lineage whose item carries no grid gets
    /// what the engine gives such a prompt: linear positions, zero delta.
    #[test]
    fn a_qwen35_item_without_a_grid_gets_linear_positions() {
        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams::default());
        req.tokenized.as_mut().unwrap().input_ids = vec![10, 20, 30, 40, 50];
        req.mm_inputs = Some(tokenspeed_proto::MultimodalInputs {
            items: vec![ts_mm_item(b"12345678", 1, 3)],
        });
        let wire = translate_for(req, GridTensors::Qwen35Positions(QWEN35)).expect("translated");
        let mm = wire.multimodal_inputs.expect("mm payload");
        assert_eq!(
            i64_values(&mm.mrope_positions.expect("the table")),
            vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        );
        assert_eq!(mm.mrope_position_delta_scalar, Some(0));
    }

    #[test]
    fn tokenspeed_fan_out_derives_explicit_sampling_seeds() {
        let subs =
            fan_out_tokenspeed_requests(ts_tokenized_req(tokenspeed_proto::SamplingParams {
                n: 3,
                sampling_seed: Some(7),
                ..Default::default()
            }));

        assert_eq!(subs.len(), 3);
        for (i, sub) in (0_u64..).zip(&subs) {
            let sampling = sub.sampling_params.as_ref().expect("sampling params");
            assert_eq!(sub.request_id, format!("r1-{i}"));
            assert_eq!(sampling.n, 1);
            assert_eq!(sampling.sampling_seed, Some(7 + i));
        }

        let subs =
            fan_out_tokenspeed_requests(ts_tokenized_req(tokenspeed_proto::SamplingParams {
                n: 2,
                ..Default::default()
            }));
        assert!(subs.iter().all(|sub| sub
            .sampling_params
            .as_ref()
            .expect("sampling params")
            .sampling_seed
            .is_none()));
    }

    /// n=2 over TokenSpeed: two engine-side requests with distinct sub-rids
    /// (delivered in one wire batch), two indexed `Complete`s, and the shared
    /// prompt reported in full on each (the pipeline de-duplicates via max, so
    /// prompt tokens are not counted n times).
    #[tokio::test]
    async fn generate_e2e_fans_out_n2_tokenspeed() {
        use engine_zmq_client::{
            codec::{decode_msgpack, encode_msgpack},
            protocol::tokenspeed::{
                output::BatchTokenIDOutSlim,
                request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
            },
        };

        let (client, engine, _dir) = connected_pair(RuntimeType::TokenSpeed).await;

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            let mut rids = Vec::new();
            for _ in 0..2 {
                let frames = input.recv_frames().await.unwrap();
                assert_eq!(
                    TokenSpeedRequestType::from_frame(frames[0].as_ref()),
                    Some(TokenSpeedRequestType::Add)
                );
                let request: TokenizedGenerateReqInput =
                    decode_msgpack(frames[1].as_ref()).unwrap();
                assert_eq!(request.sampling_params.n, 1);
                // TokenSpeed has no seed on the wire; the engine derives one from
                // the (unique) rid so all TP/DP ranks agree.
                assert_eq!(request.sampling_params.seed, None);
                rids.push(request.rid.clone());
            }
            assert_eq!(
                rids,
                vec!["r1-0".to_string(), "r1-1".to_string()],
                "sub-rids must be unique per sub"
            );
            // Both subs finish in one wire batch (the batch demux fans them
            // back out to their sub-streams).
            let done = BatchTokenIDOutSlim {
                rids: vec!["r1-0".into(), "r1-1".into()],
                output_ids: vec![vec![10], vec![11]],
                finished_reasons: vec!["stop".into(), "stop".into()],
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

        let mut req = ts_tokenized_req(tokenspeed_proto::SamplingParams {
            n: 2,
            ..Default::default()
        });
        req.request_id = "r1".to_string();
        let mut stream = client.generate_tokenspeed(req).await.expect("generate");

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
        // Each sub reports the full shared prompt; the pipeline maxes, so
        // usage is not double-counted.
        assert!(completes.iter().all(|complete| complete.prompt_tokens == 3));

        engine_task.await.unwrap();
    }

    #[test]
    fn tokenspeed_finish_reason_normalizes() {
        assert_eq!(normalize_finish_reason("stop"), "stop");
        assert_eq!(normalize_finish_reason("length"), "length");
        assert_eq!(normalize_finish_reason("abort"), "abort");
        assert_eq!(normalize_finish_reason("error"), "error");
        // An unknown wire value falls back to "stop".
        assert_eq!(normalize_finish_reason("garbage"), "stop");
    }

    /// This wire carries the finish type alone and its engine has no abort
    /// code for its grammar: an `abort` on a constrained request is as often
    /// a pause aborting every request in flight (the finish an RL driver
    /// resubmits on) as anything else, an `error` a transfer or numerical
    /// failure. Both keep the shape every request has, the `abort` a
    /// `Complete` with finish reason `abort` and the `error` the engine's own
    /// report; neither is the constraint-failure status, which only an
    /// engine's own attribution raises.
    #[tokio::test]
    async fn an_abort_or_error_finish_of_a_constrained_request_keeps_the_engines_own_shape() {
        use engine_zmq_client::{
            codec::encode_msgpack, protocol::tokenspeed::output::BatchTokenIDOutSlim,
        };
        use smg_grpc_client::is_constraint_failure;

        let (client, engine, _dir) = connected_pair(RuntimeType::TokenSpeed).await;
        let cases = [
            ("r1", "abort", true),
            ("r2", "error", true),
            ("r3", "abort", false),
            ("r4", "error", false),
        ];

        #[expect(
            clippy::disallowed_methods,
            reason = "engine task ends after responding"
        )]
        let engine_task = tokio::spawn(async move {
            let (mut input, mut output) = engine.split();
            for (rid, finish, _) in cases {
                let _ = input.recv_frames().await.unwrap();
                let done = BatchTokenIDOutSlim {
                    rids: vec![rid.into()],
                    output_ids: vec![vec![]],
                    finished_reasons: vec![finish.into()],
                    prompt_tokens: vec![3],
                    completion_tokens: vec![0],
                    cached_tokens: vec![0],
                    output_token_logprobs_val: vec![vec![]],
                    output_token_logprobs_idx: vec![vec![]],
                    ..Default::default()
                };
                output
                    .send_frames(vec![bytes::Bytes::from(encode_msgpack(&done).unwrap())])
                    .await
                    .unwrap();
            }
            // Keep the peer alive until the frontend has consumed the replies.
            (input, output)
        });

        for (rid, finish, constrained) in cases {
            let req = tokenspeed_proto::GenerateRequest {
                request_id: rid.to_string(),
                tokenized: Some(tokenspeed_proto::TokenizedInput {
                    input_ids: vec![1, 2, 3],
                    original_text: String::new(),
                }),
                sampling_params: Some(tokenspeed_proto::SamplingParams {
                    max_new_tokens: Some(8),
                    constraint: constrained.then(|| {
                        tokenspeed_proto::sampling_params::Constraint::JsonSchema(
                            r#"{"type":"object"}"#.to_string(),
                        )
                    }),
                    ..Default::default()
                }),
                stream: true,
                ..Default::default()
            };
            let mut stream = client.generate_tokenspeed(req).await.expect("generate");
            let item = stream.next().await.expect("an item");
            if finish == "error" {
                let error = item.expect_err("an error");
                assert_eq!(error.code(), tonic::Code::Internal, "{rid}");
                assert_eq!(
                    error.message(),
                    "engine finished the request with an error (see engine logs)",
                    "{rid}: the engine's own report, word for word"
                );
                assert!(!is_constraint_failure(&error), "{rid}: {error}");
            } else {
                match item.expect("a response").response {
                    Some(vllm::generate_response::Response::Complete(complete)) => {
                        assert_eq!(
                            complete.finish_reason, "abort",
                            "{rid} (constrained: {constrained})"
                        );
                    }
                    other => panic!("{rid}: expected a Complete, got {other:?}"),
                }
            }
        }
        engine_task.await.unwrap();
    }
}
