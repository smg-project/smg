//! The request wrappers: one variant per wire for a request (`ProtoRequest`)
//! and for a generate request (`ProtoGenerateRequest`, with the per-wire
//! accessors and setters the routers shape it with), the answer constraint
//! and the vLLM M-RoPE grid keys. Moved out of `proto_wrapper.rs` as is;
//! the four names the code takes from sibling modules come through the
//! parent's imports.

use smg_grpc_client::{
    mlx_proto as mlx, sglang_proto as sglang, tokenspeed_proto as tokenspeed,
    trtllm_proto as trtllm, vllm_proto as vllm,
};

use super::{vllm_mm_hash_identity, vllm_mm_identity, EncodeItemBootstrapInfo, ProtoEmbedRequest};

/// Unified ProtoRequest
#[derive(Clone)]
pub enum ProtoRequest {
    Generate(ProtoGenerateRequest),
    Embed(ProtoEmbedRequest),
}

impl ProtoRequest {
    /// Serialized wire size, for the release metric.
    pub fn wire_len(&self) -> usize {
        match self {
            Self::Generate(request) => request.wire_len(),
            Self::Embed(request) => request.wire_len(),
        }
    }

    /// Get request ID from either variant
    pub fn request_id(&self) -> &str {
        match self {
            Self::Generate(req) => req.request_id(),
            Self::Embed(req) => req.request_id(),
        }
    }
}

/// Unified GenerateRequest that works with all backends
#[derive(Clone)]
pub enum ProtoGenerateRequest {
    Sglang(Box<sglang::GenerateRequest>),
    Vllm(Box<vllm::GenerateRequest>),
    Trtllm(Box<trtllm::GenerateRequest>),
    Mlx(Box<mlx::GenerateRequest>),
    TokenSpeed(Box<tokenspeed::GenerateRequest>),
}

/// Per-item grid metadata the engine reads to compute M-RoPE positions
/// (`_get_mrope_input_positions` in vLLM); a few ints per item. The video
/// timing has two spellings: `second_per_grid_ts` from the Qwen-VL
/// processors, `video_second_per_grid` from the Qwen-Omni family and this
/// gateway's own processors. The servicer's `GRID_KEYS` mirrors this list.
pub(crate) const VLLM_MROPE_GRID_KEYS: [&str; 4] = [
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
    "video_second_per_grid",
];

/// A JSON answer's constraint as the worker of an attempt takes it: the table's
/// structural tag, or the request's own schema where the table gives that
/// worker none (`regular::symphony::worker_constraint`); across attempts the
/// one stands in for the other.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnswerConstraint {
    StructuralTag(String),
    JsonSchema(String),
}

impl ProtoGenerateRequest {
    /// Append stop token ids to the request's sampling params (TRT-LLM keeps
    /// them on the request itself). Requests without sampling params are left
    /// unchanged, matching the per-engine injection this replaces.
    pub fn extend_stop_token_ids(&mut self, ids: &[u32]) {
        match self {
            Self::Sglang(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.stop_token_ids.extend_from_slice(ids);
                }
            }
            Self::Vllm(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.stop_token_ids.extend_from_slice(ids);
                }
            }
            Self::Mlx(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.stop_token_ids.extend_from_slice(ids);
                }
            }
            Self::TokenSpeed(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.stop_token_ids.extend_from_slice(ids);
                }
            }
            Self::Trtllm(req) => req.stop_token_ids.extend_from_slice(ids),
        }
    }

    /// Get SGLang variant (panics if not SGLang)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang(&self) -> &sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            _ => panic!("Expected SGLang GenerateRequest"),
        }
    }

    /// Get mutable SGLang variant (panics if not SGLang)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            _ => panic!("Expected SGLang GenerateRequest"),
        }
    }

    /// Get vLLM variant (panics if not vLLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_vllm() check"
    )]
    pub fn as_vllm(&self) -> &vllm::GenerateRequest {
        match self {
            Self::Vllm(req) => req,
            _ => panic!("Expected vLLM GenerateRequest"),
        }
    }

    /// Get TensorRT-LLM variant (panics if not TensorRT-LLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_trtllm() check"
    )]
    pub fn as_trtllm(&self) -> &trtllm::GenerateRequest {
        match self {
            Self::Trtllm(req) => req,
            _ => panic!("Expected TensorRT-LLM GenerateRequest"),
        }
    }

    /// Get TokenSpeed variant (panics if not TokenSpeed)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_tokenspeed() check"
    )]
    pub fn as_tokenspeed(&self) -> &tokenspeed::GenerateRequest {
        match self {
            Self::TokenSpeed(req) => req,
            _ => panic!("Expected TokenSpeed GenerateRequest"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Check if this is TensorRT-LLM
    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    /// Check if this is TokenSpeed
    pub fn is_tokenspeed(&self) -> bool {
        matches!(self, Self::TokenSpeed(_))
    }

    /// Sanitize sampling params for the prefill-only side (vLLM PD mode).
    /// max_tokens=1 computes KV without generating; min_tokens is cleared so the
    /// engine accepts it; n=1 so the prefill returns a single kv_transfer_params dict.
    /// Stop criteria are cleared and EOS ignored so the side always finishes
    /// length-capped — vLLM < 0.20 returns the NIXL handoff only for that status.
    pub fn sanitize_sampling_for_prefill(&mut self, max_tokens: u32) {
        match self {
            Self::Vllm(req) => {
                let params = req.sampling_params.get_or_insert_with(Default::default);
                params.max_tokens = Some(max_tokens);
                params.min_tokens = 0;
                params.n = 1;
                params.stop.clear();
                params.stop_token_ids.clear();
                params.ignore_eos = true;
            }
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                tracing::warn!(
                    "sanitize_sampling_for_prefill called on non-vLLM request, ignoring"
                );
            }
        }
    }

    /// Set stream mode on the request.
    pub fn set_stream(&mut self, stream: bool) {
        match self {
            Self::Vllm(req) => req.stream = stream,
            Self::Sglang(req) => req.stream = stream,
            Self::Trtllm(req) => req.streaming = stream,
            Self::Mlx(req) => req.stream = stream,
            Self::TokenSpeed(req) => req.stream = stream,
        }
    }

    /// Whether the engine streams this request's responses as it produces
    /// them, or answers once at the end.
    pub fn stream(&self) -> bool {
        match self {
            Self::Vllm(req) => req.stream,
            Self::Sglang(req) => req.stream,
            Self::Trtllm(req) => req.streaming,
            Self::Mlx(req) => req.stream,
            Self::TokenSpeed(req) => req.stream,
        }
    }

    /// Serialized wire size, for the release metric.
    pub fn wire_len(&self) -> usize {
        use prost::Message;
        match self {
            Self::Sglang(req) => req.encoded_len(),
            Self::Vllm(req) => req.encoded_len(),
            Self::Trtllm(req) => req.encoded_len(),
            Self::Mlx(req) => req.encoded_len(),
            Self::TokenSpeed(req) => req.encoded_len(),
        }
    }

    /// Clone for a PD side that carries no multimodal pixels (see
    /// `clear_mm_pixel_values`), without ever duplicating them: detach,
    /// clone, reattach to `self`. vLLM keeps `mm_hashes`/placeholders and the
    /// M-RoPE grid tensors in the clone: the servicer rebuilds pixel-less
    /// mm features from them, so the decode worker computes grid-aware
    /// positions and per-image block hashes instead of treating the prompt
    /// as text (which mis-rotates every generated token on Qwen-VL models
    /// and lets different images alias in the decode prefix cache).
    pub fn clone_without_mm_pixels(&mut self) -> Self {
        match self {
            Self::Sglang(req) => {
                let mm = req.mm_inputs.take();
                let clone = Self::Sglang(req.clone());
                req.mm_inputs = mm;
                clone
            }
            Self::Vllm(req) => {
                let mm = req.mm_inputs.take();
                let extra = std::mem::take(&mut req.extra_mm_inputs);
                let mut clone = Self::Vllm(req.clone());
                if let Self::Vllm(clone_req) = &mut clone {
                    clone_req.mm_inputs = mm.as_ref().and_then(vllm_mm_identity);
                    clone_req.extra_mm_inputs = extra.iter().filter_map(vllm_mm_identity).collect();
                }
                req.mm_inputs = mm;
                req.extra_mm_inputs = extra;
                clone
            }
            Self::TokenSpeed(req) => {
                let detached: Vec<_> = req
                    .mm_inputs
                    .as_mut()
                    .map(|mm| {
                        mm.items
                            .iter_mut()
                            .map(|item| item.encoder_input.take())
                            .collect()
                    })
                    .unwrap_or_default();
                let clone = Self::TokenSpeed(req.clone());
                if let Some(mm) = req.mm_inputs.as_mut() {
                    for (item, encoder_input) in mm.items.iter_mut().zip(detached) {
                        item.encoder_input = encoder_input;
                    }
                }
                clone
            }
            Self::Trtllm(req) => Self::Trtllm(req.clone()),
            Self::Mlx(req) => Self::Mlx(req.clone()),
        }
    }

    /// Clone for a PD decode side on a language-model-only decode worker (no
    /// vision encoder, encoder-cache budget 0 — the production vLLM P/D
    /// shape). The prefill-expanded `input_ids` and the KV handoff stay, but
    /// every multimodal payload beyond the content hashes goes: pixels,
    /// placeholders, grid tensors and media references. The decode engine
    /// then sees a pure-text TokensPrompt and never touches its (zero-budget)
    /// encoder cache, as the language-model-only P/D contract requires; the kept hashes ride
    /// into `cache_salt` servicer-side so different images cannot alias in
    /// the decode prefix cache. Non-vLLM backends have no language-model-only
    /// mode, so they take the pixel-stripping clone.
    pub fn clone_without_mm(&mut self) -> Self {
        match self {
            Self::Vllm(req) => {
                let mm = req.mm_inputs.take();
                let extra = std::mem::take(&mut req.extra_mm_inputs);
                let refs = req.media_refs.take();
                let mut clone = Self::Vllm(req.clone());
                if let Self::Vllm(clone_req) = &mut clone {
                    clone_req.mm_inputs = mm.as_ref().and_then(vllm_mm_hash_identity);
                    clone_req.extra_mm_inputs =
                        extra.iter().filter_map(vllm_mm_hash_identity).collect();
                }
                req.mm_inputs = mm;
                req.extra_mm_inputs = extra;
                req.media_refs = refs;
                clone
            }
            _ => self.clone_without_mm_pixels(),
        }
    }

    /// Whether any vLLM multimodal batch carries M-RoPE grid tensors. The
    /// decode side of a grid-dependent model (Qwen-VL family) derives its
    /// positions from them, so they cannot be stripped for a
    /// language-model-only decode worker.
    pub fn has_vllm_mrope_grids(&self) -> bool {
        match self {
            Self::Vllm(req) => req.mm_inputs.iter().chain(&req.extra_mm_inputs).any(|mm| {
                mm.model_specific_tensors
                    .keys()
                    .any(|key| VLLM_MROPE_GRID_KEYS.contains(&key.as_str()))
            }),
            _ => false,
        }
    }

    /// Drop raw multimodal encoder tensors while keeping item metadata.
    ///
    /// Used by the EPD prefill side: multimodal embeddings arrive from encode workers,
    /// but prefill still needs placeholders/model-specific metadata to slot them.
    pub fn clear_mm_pixel_values(&mut self) {
        match self {
            Self::Sglang(req) => req.mm_inputs = None,
            Self::Vllm(req) => {
                req.mm_inputs = None;
                req.extra_mm_inputs.clear();
            }
            Self::TokenSpeed(req) => {
                if let Some(mm) = req.mm_inputs.as_mut() {
                    for item in &mut mm.items {
                        item.encoder_input = None;
                    }
                }
            }
            Self::Trtllm(_) | Self::Mlx(_) => {}
        }
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
            Self::Vllm(req) => &req.request_id,
            Self::Trtllm(req) => &req.request_id,
            Self::Mlx(req) => &req.request_id,
            Self::TokenSpeed(req) => &req.request_id,
        }
    }

    /// Set request ID (retry attempts re-mint per-execution engine ids on
    /// the retained plan).
    pub fn set_request_id(&mut self, request_id: String) {
        match self {
            Self::Sglang(req) => req.request_id = request_id,
            Self::Vllm(req) => req.request_id = request_id,
            Self::Trtllm(req) => req.request_id = request_id,
            Self::Mlx(req) => req.request_id = request_id,
            Self::TokenSpeed(req) => req.request_id = request_id,
        }
    }

    /// Set KV transfer parameters for Mooncake PD disaggregation (vLLM only).
    /// These parameters tell the decode worker where to fetch KV cache from the prefill worker.
    pub fn set_kv_transfer_params(&mut self, remote_host: String, remote_port: u32) {
        match self {
            Self::Vllm(req) => {
                req.kv_transfer_params = Some(vllm::KvTransferParams {
                    remote_host,
                    remote_port,
                });
            }
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                tracing::warn!("set_kv_transfer_params called on non-vLLM request, ignoring");
            }
        }
    }

    /// Pin the request to a data-parallel rank (engines without the field ignore it).
    pub fn set_data_parallel_rank(&mut self, rank: i32) {
        match self {
            Self::Vllm(req) => req.data_parallel_rank = Some(rank),
            Self::Sglang(req) => req.data_parallel_rank = rank,
            Self::TokenSpeed(req) => req.data_parallel_rank = Some(rank),
            Self::Trtllm(_) | Self::Mlx(_) => {}
        }
    }

    /// Prompt tokens the engine has to prefill for this request: the
    /// tokenized input's length (zero for a text input, which the engine
    /// tokenizes itself).
    pub fn prompt_len(&self) -> usize {
        match self {
            Self::Sglang(req) => req
                .tokenized
                .as_ref()
                .map_or(0, |input| input.input_ids.len()),
            Self::Vllm(req) => match &req.input {
                Some(vllm::generate_request::Input::Tokenized(input)) => input.input_ids.len(),
                _ => 0,
            },
            Self::Trtllm(req) => req
                .tokenized
                .as_ref()
                .map_or(0, |input| input.input_token_ids.len()),
            Self::Mlx(req) => match &req.input {
                Some(mlx::generate_request::Input::Tokenized(input)) => input.input_ids.len(),
                _ => 0,
            },
            Self::TokenSpeed(req) => req
                .tokenized
                .as_ref()
                .map_or(0, |input| input.input_ids.len()),
        }
    }

    /// Attach media references for worker-side multimodal processing (vLLM only).
    pub fn set_vllm_media_refs(&mut self, refs: vllm::MediaRefs) -> Result<(), String> {
        match self {
            Self::Vllm(req) => {
                req.media_refs = Some(refs);
                Ok(())
            }
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                Err("media refs require a vLLM gRPC worker".to_string())
            }
        }
    }

    /// Whether this vLLM request carries media references.
    pub fn has_vllm_media_refs(&self) -> bool {
        matches!(self, Self::Vllm(req) if req.media_refs.is_some())
    }

    /// Replace this vLLM request's media references with the identity the
    /// prefill side produced: the expanded prompt ids and the pixel-less
    /// per-modality inputs, in the shape a router-preprocessed decode side has.
    /// Applied whole or not at all: an identity without ids, or a side whose
    /// input is not tokenized, leaves the side as it is (`false`), so it
    /// reprocesses the references rather than run an empty prompt.
    pub fn apply_media_identity(&mut self, identity: &vllm::MediaIdentity) -> bool {
        let Self::Vllm(req) = self else {
            return false;
        };
        let Some(vllm::generate_request::Input::Tokenized(tokenized)) = req.input.as_mut() else {
            return false;
        };
        if identity.prompt_token_ids.is_empty() {
            return false;
        }
        tokenized.input_ids.clone_from(&identity.prompt_token_ids);
        req.mm_inputs.clone_from(&identity.mm_inputs);
        req.extra_mm_inputs.clone_from(&identity.extra_mm_inputs);
        req.media_refs = None;
        true
    }

    /// Number of parallel samples requested (1 when unset). vLLM, SGLang
    /// and TokenSpeed carry it in their sampling params; the others have no
    /// per-request sample count.
    pub fn sampling_n(&self) -> u32 {
        let n = match self {
            Self::Vllm(req) => req.sampling_params.as_ref().map(|p| p.n),
            Self::Sglang(req) => req.sampling_params.as_ref().map(|p| p.n),
            Self::TokenSpeed(req) => req.sampling_params.as_ref().map(|p| p.n),
            Self::Trtllm(_) | Self::Mlx(_) => None,
        };
        n.filter(|&n| n > 0).unwrap_or(1)
    }

    /// Set the number of parallel samples (engines without the field ignore it).
    pub fn set_sampling_n(&mut self, n: u32) {
        match self {
            Self::Vllm(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.n = n;
                }
            }
            Self::Sglang(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.n = n;
                }
            }
            Self::TokenSpeed(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.n = n;
                }
            }
            Self::Trtllm(_) | Self::Mlx(_) => {}
        }
    }

    /// The structural tag the request's constraint carries, when that is its
    /// constraint (TRT-LLM: a structural-tag guide; MLX: never, its builder
    /// refuses constraints).
    pub fn structural_tag(&self) -> Option<&str> {
        match self {
            Self::Sglang(req) => match req.sampling_params.as_ref()?.constraint.as_ref()? {
                sglang::sampling_params::Constraint::StructuralTag(tag) => Some(tag),
                _ => None,
            },
            Self::Vllm(req) => match req.sampling_params.as_ref()?.constraint.as_ref()? {
                vllm::sampling_params::Constraint::StructuralTag(tag) => Some(tag),
                _ => None,
            },
            Self::TokenSpeed(req) => match req.sampling_params.as_ref()?.constraint.as_ref()? {
                tokenspeed::sampling_params::Constraint::StructuralTag(tag) => Some(tag),
                _ => None,
            },
            Self::Trtllm(req) => {
                let guided = req.guided_decoding.as_ref()?;
                (guided.guide_type
                    == trtllm::guided_decoding_params::GuideType::StructuralTag as i32)
                    .then_some(guided.guide.as_str())
            }
            Self::Mlx(_) => None,
        }
    }

    /// The JSON schema the request's constraint carries, when that is its
    /// constraint (TRT-LLM: a JSON-schema guide; MLX: never, its builder
    /// refuses constraints).
    pub fn json_schema(&self) -> Option<&str> {
        match self {
            Self::Sglang(req) => match req.sampling_params.as_ref()?.constraint.as_ref()? {
                sglang::sampling_params::Constraint::JsonSchema(schema) => Some(schema),
                _ => None,
            },
            Self::Vllm(req) => match req.sampling_params.as_ref()?.constraint.as_ref()? {
                vllm::sampling_params::Constraint::JsonSchema(schema) => Some(schema),
                _ => None,
            },
            Self::TokenSpeed(req) => match req.sampling_params.as_ref()?.constraint.as_ref()? {
                tokenspeed::sampling_params::Constraint::JsonSchema(schema) => Some(schema),
                _ => None,
            },
            Self::Trtllm(req) => {
                let guided = req.guided_decoding.as_ref()?;
                (guided.guide_type == trtllm::guided_decoding_params::GuideType::JsonSchema as i32)
                    .then_some(guided.guide.as_str())
            }
            Self::Mlx(_) => None,
        }
    }

    /// Say whether the request's grammar starts at the first generated token
    /// whatever reasoning parser the engine runs: the gateway led it with the
    /// model's thought and its close for a worker whose grammar starts there
    /// (`SamplingParams.grammar_from_first_token`). Only the dialect with a
    /// request-side switch for the gate behind its parser carries it; the
    /// others start every grammar where their engine does.
    pub(crate) fn set_grammar_from_first_token(&mut self, first_token: bool) {
        if let Self::Vllm(req) = self {
            if let Some(params) = req.sampling_params.as_mut() {
                params.grammar_from_first_token = first_token;
            }
        }
    }

    /// Whether the request says its grammar starts at the first generated
    /// token ([`Self::set_grammar_from_first_token`]); what the tests read.
    #[cfg(test)]
    pub(crate) fn grammar_from_first_token(&self) -> bool {
        match self {
            Self::Vllm(req) => req
                .sampling_params
                .as_ref()
                .is_some_and(|params| params.grammar_from_first_token),
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => false,
        }
    }

    /// Replace the structural tag the request's constraint carries; a request
    /// whose constraint is anything else, or none, is left as it is.
    pub fn set_structural_tag(&mut self, tag: String) {
        match self {
            Self::Sglang(req) => {
                if let Some(sglang::sampling_params::Constraint::StructuralTag(current)) = req
                    .sampling_params
                    .as_mut()
                    .and_then(|params| params.constraint.as_mut())
                {
                    *current = tag;
                }
            }
            Self::Vllm(req) => {
                if let Some(vllm::sampling_params::Constraint::StructuralTag(current)) = req
                    .sampling_params
                    .as_mut()
                    .and_then(|params| params.constraint.as_mut())
                {
                    *current = tag;
                }
            }
            Self::TokenSpeed(req) => {
                if let Some(tokenspeed::sampling_params::Constraint::StructuralTag(current)) = req
                    .sampling_params
                    .as_mut()
                    .and_then(|params| params.constraint.as_mut())
                {
                    *current = tag;
                }
            }
            Self::Trtllm(req) => {
                if let Some(guided) = req.guided_decoding.as_mut() {
                    if guided.guide_type
                        == trtllm::guided_decoding_params::GuideType::StructuralTag as i32
                    {
                        guided.guide = tag;
                    }
                }
            }
            Self::Mlx(_) => {}
        }
    }

    /// Set `tag` as the request's constraint where the request carries a
    /// structural tag or none (the tag it was built with, which a retry
    /// cleared: [`Self::clear_structural_tag`]); a request whose constraint
    /// is anything else is left as it is, like one without sampling params.
    pub fn restore_structural_tag(&mut self, tag: String) {
        match self {
            Self::Sglang(req) => {
                use sglang::sampling_params::Constraint;
                if let Some(params) = req.sampling_params.as_mut() {
                    if matches!(params.constraint, None | Some(Constraint::StructuralTag(_))) {
                        params.constraint = Some(Constraint::StructuralTag(tag));
                    }
                }
            }
            Self::Vllm(req) => {
                use vllm::sampling_params::Constraint;
                if let Some(params) = req.sampling_params.as_mut() {
                    if matches!(params.constraint, None | Some(Constraint::StructuralTag(_))) {
                        params.constraint = Some(Constraint::StructuralTag(tag));
                    }
                }
            }
            Self::TokenSpeed(req) => {
                use tokenspeed::sampling_params::Constraint;
                if let Some(params) = req.sampling_params.as_mut() {
                    if matches!(params.constraint, None | Some(Constraint::StructuralTag(_))) {
                        params.constraint = Some(Constraint::StructuralTag(tag));
                    }
                }
            }
            Self::Trtllm(req) => {
                let structural_tag =
                    trtllm::guided_decoding_params::GuideType::StructuralTag as i32;
                match req.guided_decoding.as_mut() {
                    None => {
                        req.guided_decoding = Some(trtllm::GuidedDecodingParams {
                            guide_type: structural_tag,
                            guide: tag,
                        });
                    }
                    Some(guided) if guided.guide_type == structural_tag => guided.guide = tag,
                    Some(_) => {}
                }
            }
            Self::Mlx(_) => {}
        }
    }

    /// Drop the structural tag the request's constraint carries, leaving the
    /// request unconstrained; a request whose constraint is anything else, or
    /// none, is left as it is.
    pub fn clear_structural_tag(&mut self) {
        match self {
            Self::Sglang(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    if let Some(sglang::sampling_params::Constraint::StructuralTag(_)) =
                        params.constraint
                    {
                        params.constraint = None;
                    }
                }
            }
            Self::Vllm(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    if let Some(vllm::sampling_params::Constraint::StructuralTag(_)) =
                        params.constraint
                    {
                        params.constraint = None;
                    }
                }
            }
            Self::TokenSpeed(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    if let Some(tokenspeed::sampling_params::Constraint::StructuralTag(_)) =
                        params.constraint
                    {
                        params.constraint = None;
                    }
                }
            }
            Self::Trtllm(req) => {
                if req.guided_decoding.as_ref().is_some_and(|guided| {
                    guided.guide_type
                        == trtllm::guided_decoding_params::GuideType::StructuralTag as i32
                }) {
                    req.guided_decoding = None;
                }
            }
            Self::Mlx(_) => {}
        }
    }

    /// Set `constraint` as the request's constraint where the request carries
    /// a structural tag, a JSON schema or none, the forms a JSON answer's
    /// constraint takes on the attempts before ([`AnswerConstraint`]); a
    /// request whose constraint is anything else (a regex, a grammar) is left
    /// as it is, like one without sampling params.
    pub fn restore_answer_constraint(&mut self, constraint: AnswerConstraint) {
        match self {
            Self::Sglang(req) => {
                use sglang::sampling_params::Constraint;
                if let Some(params) = req.sampling_params.as_mut() {
                    if matches!(
                        params.constraint,
                        None | Some(Constraint::StructuralTag(_) | Constraint::JsonSchema(_))
                    ) {
                        params.constraint = Some(match constraint {
                            AnswerConstraint::StructuralTag(tag) => Constraint::StructuralTag(tag),
                            AnswerConstraint::JsonSchema(schema) => Constraint::JsonSchema(schema),
                        });
                    }
                }
            }
            Self::Vllm(req) => {
                use vllm::sampling_params::Constraint;
                if let Some(params) = req.sampling_params.as_mut() {
                    if matches!(
                        params.constraint,
                        None | Some(Constraint::StructuralTag(_) | Constraint::JsonSchema(_))
                    ) {
                        params.constraint = Some(match constraint {
                            AnswerConstraint::StructuralTag(tag) => Constraint::StructuralTag(tag),
                            AnswerConstraint::JsonSchema(schema) => Constraint::JsonSchema(schema),
                        });
                    }
                }
            }
            Self::TokenSpeed(req) => {
                use tokenspeed::sampling_params::Constraint;
                if let Some(params) = req.sampling_params.as_mut() {
                    if matches!(
                        params.constraint,
                        None | Some(Constraint::StructuralTag(_) | Constraint::JsonSchema(_))
                    ) {
                        params.constraint = Some(match constraint {
                            AnswerConstraint::StructuralTag(tag) => Constraint::StructuralTag(tag),
                            AnswerConstraint::JsonSchema(schema) => Constraint::JsonSchema(schema),
                        });
                    }
                }
            }
            Self::Trtllm(req) => {
                use trtllm::guided_decoding_params::GuideType;
                let answers = |guided: &trtllm::GuidedDecodingParams| {
                    guided.guide_type == GuideType::StructuralTag as i32
                        || guided.guide_type == GuideType::JsonSchema as i32
                };
                if req.guided_decoding.as_ref().is_none_or(answers) {
                    let (guide_type, guide) = match constraint {
                        AnswerConstraint::StructuralTag(tag) => (GuideType::StructuralTag, tag),
                        AnswerConstraint::JsonSchema(schema) => (GuideType::JsonSchema, schema),
                    };
                    req.guided_decoding = Some(trtllm::GuidedDecodingParams {
                        guide_type: guide_type as i32,
                        guide,
                    });
                }
            }
            Self::Mlx(_) => {}
        }
    }

    /// Offset an explicit sampling seed so fanned-out samples stay distinct
    /// yet deterministic; an unset seed stays unset (the engine then seeds
    /// each request on its own). Engines without a seed field ignore it.
    pub fn offset_sampling_seed(&mut self, offset: u32) {
        match self {
            Self::Vllm(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    let offset = i32::try_from(offset).unwrap_or(i32::MAX);
                    params.seed = params.seed.map(|seed| seed.wrapping_add(offset));
                }
            }
            Self::TokenSpeed(req) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    params.sampling_seed = params
                        .sampling_seed
                        .map(|seed| seed.wrapping_add(u64::from(offset)));
                }
            }
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) => {}
        }
    }

    /// Whether the request carries multimodal inputs.
    pub fn has_mm_inputs(&self) -> bool {
        match self {
            Self::Sglang(req) => req.mm_inputs.is_some(),
            Self::Vllm(req) => req.mm_inputs.is_some() || !req.extra_mm_inputs.is_empty(),
            Self::TokenSpeed(req) => req.mm_inputs.is_some(),
            Self::Trtllm(_) | Self::Mlx(_) => false,
        }
    }

    /// Set opaque connector KV-transfer params as JSON (vLLM only).
    /// Passed verbatim to the engine (NIXL handoff, etc.).
    pub fn set_kv_transfer_params_json(&mut self, json: String) {
        match self {
            Self::Vllm(req) => req.kv_transfer_params_json = Some(json),
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                tracing::warn!("set_kv_transfer_params_json called on non-vLLM request, ignoring");
            }
        }
    }

    /// Whether the request carries a KV handoff from the prefill worker
    /// (either the modern JSON params or the legacy typed host/port).
    pub fn has_kv_transfer_params(&self) -> bool {
        match self {
            Self::Vllm(req) => {
                req.kv_transfer_params_json.is_some() || req.kv_transfer_params.is_some()
            }
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => false,
        }
    }

    /// Set encode->prefill bootstrap info for backends that receive multimodal embeddings
    /// out-of-band from encode workers.
    pub(crate) fn set_encode_bootstrap_info(&mut self, items: Vec<EncodeItemBootstrapInfo>) {
        match self {
            Self::TokenSpeed(req) => {
                let items = items
                    .into_iter()
                    .map(|item| tokenspeed::EncodeItemBootstrapInfo {
                        item_index: item.item_index,
                        bootstrap_host: item.bootstrap_host,
                        bootstrap_port: item.bootstrap_port,
                        bootstrap_room: item.bootstrap_room,
                    })
                    .collect();
                req.encode_bootstrap_info = Some(tokenspeed::EncodeBootstrapInfo { items });
            }
            Self::Sglang(_) | Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) => {
                tracing::warn!(
                    "set_encode_bootstrap_info called on a backend without encode bootstrap info, ignoring"
                );
            }
        }
    }

    /// Clear prefill-only encode bootstrap info from the decode-side request.
    pub(crate) fn clear_encode_bootstrap_info(&mut self) {
        match self {
            Self::TokenSpeed(req) => req.encode_bootstrap_info = None,
            Self::Sglang(_) | Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) => {}
        }
    }

    /// Set the PD prefill->decode KV rendezvous params (TokenSpeed only).
    ///
    /// The gateway sends identical params to both the prefill and decode worker:
    /// the prefill hosts the Mooncake bootstrap server at (`bootstrap_host`,
    /// `bootstrap_port`) and the decode worker discovers it there, keyed by
    /// `bootstrap_room`.
    pub fn set_kv_bootstrap_info(
        &mut self,
        bootstrap_host: String,
        bootstrap_port: i32,
        bootstrap_room: i64,
    ) {
        match self {
            Self::TokenSpeed(req) => {
                req.kv_bootstrap_info = Some(tokenspeed::KvBootstrapInfo {
                    bootstrap_host,
                    bootstrap_port,
                    bootstrap_room,
                });
            }
            Self::Sglang(_) | Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) => {
                tracing::warn!("set_kv_bootstrap_info called on non-TokenSpeed request, ignoring");
            }
        }
    }
}
