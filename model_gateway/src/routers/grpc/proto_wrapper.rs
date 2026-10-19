//! Protocol buffer type wrappers for the supported gRPC backends.
//!
//! This module provides unified enums that wrap proto types from each
//! supported backend, allowing the router to work with any backend
//! transparently.

use std::{collections::HashMap, future::Future, pin::Pin, sync::Arc, time::Instant};

use futures::future::select_all;
use futures_util::StreamExt;
use smg_grpc_client::{
    mlx_engine::AbortOnDropStream as MlxStream,
    mlx_proto::{self as mlx},
    sglang_proto::{self as sglang, generate_complete::MatchedStop as SglangMatchedStop},
    sglang_scheduler::AbortOnDropStream as SglangStream,
    tokenspeed_proto::{
        self as tokenspeed, generate_complete::MatchedStop as TokenSpeedMatchedStop,
    },
    tokenspeed_scheduler::AbortOnDropStream as TokenSpeedStream,
    trtllm_proto::{self as trtllm, generate_complete::MatchedStop as TrtllmMatchedStop},
    trtllm_service::AbortOnDropStream as TrtllmStream,
    vllm_engine::AbortOnDropStream as VllmStream,
    vllm_proto::{self as vllm, generate_complete::MatchedStop as VllmMatchedStop},
    RequestDeadline,
};
use tokio::time::error::Elapsed;

use crate::{
    routers::{
        common::sse::SseSender,
        grpc::{
            multimodal::{log_mm_timing_enabled, mm_rdma_exporter},
            zmq_client::ZmqGenerateStream,
        },
    },
    worker::{liveness, Worker},
};

mod embed;
mod logprobs;
mod multimodal;
mod request;
mod shm;

pub use self::{
    embed::{ProtoEmbedComplete, ProtoEmbedRequest},
    logprobs::{ProtoInputLogProbs, ProtoOutputLogProbs, ProtoTopLogProbs},
    multimodal::{
        ChunkSemantics, MultimodalData, SglangMultimodalData, TensorBytes, TokenSpeedModality,
        TokenSpeedMultimodalData, TokenSpeedMultimodalItem, TokenSpeedTensor,
        TokenSpeedTensorStorage, TrtllmMultimodalData, VllmMultimodalData,
    },
    request::{AnswerConstraint, ProtoGenerateRequest, ProtoRequest},
    shm::{
        cleanup_mm_shm_handles, collect_tokenspeed_generate_request_shm_handles,
        collect_tokenspeed_multimodal_inputs_shm_handles,
        collect_vllm_generate_request_shm_handles, mm_shm_dev_writable, write_tokenspeed_shm_with,
    },
};
use self::{logprobs::convert_trtllm_output_logprobs, shm::vllm_mm_hash_identity};
pub(crate) use self::{
    multimodal::{stage_tokenspeed_tensor_rdma, EncodeItemBootstrapInfo},
    request::VLLM_MROPE_GRID_KEYS,
    shm::{
        cleanup_tokenspeed_items_encoder_shm, finish_tokenspeed_request, finish_vllm_request,
        vllm_mm_identity,
    },
};

/// Helper macro to convert output logprobs from proto types to unified type.
/// Both SGLang and vLLM have identical OutputLogProbs structure.
/// Note: Cloning is necessary as we convert from borrowed proto types to owned unified types.
/// OOM risk is mitigated by capping top_logprobs at 20 in sampling params.
macro_rules! convert_output_logprobs {
    ($lp:expr) => {
        ProtoOutputLogProbs {
            token_logprobs: $lp.token_logprobs.clone(),
            token_ids: $lp.token_ids.clone(),
            top_logprobs: $lp
                .top_logprobs
                .iter()
                .map(|t| ProtoTopLogProbs {
                    values: t.values.clone(),
                    token_ids: t.token_ids.clone(),
                })
                .collect(),
        }
    };
}

/// Helper macro to convert input logprobs from proto types to unified type.
macro_rules! convert_input_logprobs {
    ($lp:expr) => {
        ProtoInputLogProbs {
            token_logprobs: $lp.token_logprobs.iter().map(|t| t.value).collect(),
            token_ids: $lp.token_ids.clone(),
            top_logprobs: $lp
                .top_logprobs
                .iter()
                .map(|t| ProtoTopLogProbs {
                    values: t.values.clone(),
                    token_ids: t.token_ids.clone(),
                })
                .collect(),
        }
    };
}

/// Unified GenerateResponse from stream
pub enum ProtoGenerateResponse {
    Sglang(Box<sglang::GenerateResponse>),
    Vllm(Box<vllm::GenerateResponse>),
    Trtllm(Box<trtllm::GenerateResponse>),
    Mlx(Box<mlx::GenerateResponse>),
    TokenSpeed(Box<tokenspeed::GenerateResponse>),
}

impl ProtoGenerateResponse {
    /// Stamp the choice index on a chunk or complete. A fanned-out n>1
    /// sample reports index 0 from its own engine request; the fan-out
    /// restamps it with the sample's position.
    pub fn set_index(&mut self, index: u32) {
        match self {
            Self::Sglang(resp) => match resp.response.as_mut() {
                Some(sglang::generate_response::Response::Chunk(chunk)) => chunk.index = index,
                Some(sglang::generate_response::Response::Complete(complete)) => {
                    complete.index = index;
                }
                None => {}
            },
            Self::Vllm(resp) => match resp.response.as_mut() {
                Some(vllm::generate_response::Response::Chunk(chunk)) => chunk.index = index,
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    complete.index = index;
                }
                None => {}
            },
            Self::Trtllm(resp) => match resp.response.as_mut() {
                Some(trtllm::generate_response::Response::Chunk(chunk)) => {
                    chunk.sequence_index = index;
                }
                Some(trtllm::generate_response::Response::Complete(complete)) => {
                    complete.sequence_index = index;
                }
                None => {}
            },
            Self::Mlx(resp) => match resp.response.as_mut() {
                Some(mlx::generate_response::Response::Chunk(chunk)) => chunk.index = index,
                Some(mlx::generate_response::Response::Complete(complete)) => {
                    complete.index = index;
                }
                None => {}
            },
            Self::TokenSpeed(resp) => match resp.response.as_mut() {
                Some(tokenspeed::generate_response::Response::Chunk(chunk)) => {
                    chunk.index = index;
                }
                Some(tokenspeed::generate_response::Response::Complete(complete)) => {
                    complete.index = index;
                }
                None => {}
            },
        }
    }

    /// Stamp a disaggregated request's decode-side response with the prefill
    /// side's prefix-cache hit (see [`PrefillCachedTokens`]). The decode side
    /// loads the prefix it was handed and reports it as cached, so its own
    /// number is the transfer's size, not the request's cache hit; a chunk or
    /// complete whose sample the prefill side did not report keeps its own.
    pub fn restamp_cached_tokens(&mut self, prefill: &PrefillCachedTokens) {
        match self {
            Self::Sglang(resp) => match resp.response.as_mut() {
                Some(sglang::generate_response::Response::Chunk(chunk)) => {
                    prefill.restamp(chunk.index, &mut chunk.cached_tokens);
                }
                Some(sglang::generate_response::Response::Complete(complete)) => {
                    prefill.restamp(complete.index, &mut complete.cached_tokens);
                }
                None => {}
            },
            Self::Vllm(resp) => match resp.response.as_mut() {
                Some(vllm::generate_response::Response::Chunk(chunk)) => {
                    prefill.restamp(chunk.index, &mut chunk.cached_tokens);
                }
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    prefill.restamp(complete.index, &mut complete.cached_tokens);
                }
                None => {}
            },
            Self::Trtllm(resp) => match resp.response.as_mut() {
                Some(trtllm::generate_response::Response::Chunk(chunk)) => {
                    prefill.restamp(chunk.sequence_index, &mut chunk.cached_tokens);
                }
                Some(trtllm::generate_response::Response::Complete(complete)) => {
                    prefill.restamp(complete.sequence_index, &mut complete.cached_tokens);
                }
                None => {}
            },
            Self::Mlx(resp) => match resp.response.as_mut() {
                Some(mlx::generate_response::Response::Chunk(chunk)) => {
                    prefill.restamp(chunk.index, &mut chunk.cached_tokens);
                }
                Some(mlx::generate_response::Response::Complete(complete)) => {
                    prefill.restamp(complete.index, &mut complete.cached_tokens);
                }
                None => {}
            },
            Self::TokenSpeed(resp) => match resp.response.as_mut() {
                Some(tokenspeed::generate_response::Response::Chunk(chunk)) => {
                    prefill.restamp(chunk.index, &mut chunk.cached_tokens);
                }
                Some(tokenspeed::generate_response::Response::Complete(complete)) => {
                    prefill.restamp(complete.index, &mut complete.cached_tokens);
                }
                None => {}
            },
        }
    }

    /// Whether this is a `Complete` the engine finished with reason `"error"`.
    pub fn engine_error(&self) -> bool {
        match self {
            Self::Sglang(resp) => matches!(
                &resp.response,
                Some(sglang::generate_response::Response::Complete(c)) if c.finish_reason == "error"
            ),
            Self::Vllm(resp) => matches!(
                &resp.response,
                Some(vllm::generate_response::Response::Complete(c)) if c.finish_reason == "error"
            ),
            Self::Trtllm(resp) => matches!(
                &resp.response,
                Some(trtllm::generate_response::Response::Complete(c)) if c.finish_reason == "error"
            ),
            Self::Mlx(resp) => matches!(
                &resp.response,
                Some(mlx::generate_response::Response::Complete(c)) if c.finish_reason == "error"
            ),
            Self::TokenSpeed(resp) => matches!(
                &resp.response,
                Some(tokenspeed::generate_response::Response::Complete(c)) if c.finish_reason == "error"
            ),
        }
    }

    /// Whether this is the engine's output: a chunk, or a complete the
    /// engine neither aborted nor failed (the Rust servicers end an aborted
    /// choice with a complete, `finish_reason` "abort", in place of a
    /// status). An empty response is not.
    pub fn is_output(&self) -> bool {
        let finished = |reason: &str| !matches!(reason, "abort" | "error");
        match self {
            Self::Sglang(resp) => match &resp.response {
                Some(sglang::generate_response::Response::Chunk(_)) => true,
                Some(sglang::generate_response::Response::Complete(c)) => {
                    finished(&c.finish_reason)
                }
                None => false,
            },
            Self::Vllm(resp) => match &resp.response {
                Some(vllm::generate_response::Response::Chunk(_)) => true,
                Some(vllm::generate_response::Response::Complete(c)) => finished(&c.finish_reason),
                None => false,
            },
            Self::Trtllm(resp) => match &resp.response {
                Some(trtllm::generate_response::Response::Chunk(_)) => true,
                Some(trtllm::generate_response::Response::Complete(c)) => {
                    finished(&c.finish_reason)
                }
                None => false,
            },
            Self::Mlx(resp) => match &resp.response {
                Some(mlx::generate_response::Response::Chunk(_)) => true,
                Some(mlx::generate_response::Response::Complete(c)) => finished(&c.finish_reason),
                None => false,
            },
            Self::TokenSpeed(resp) => match &resp.response {
                Some(tokenspeed::generate_response::Response::Chunk(_)) => true,
                Some(tokenspeed::generate_response::Response::Complete(c)) => {
                    finished(&c.finish_reason)
                }
                None => false,
            },
        }
    }

    /// The prompt tokens the engine reports served from its prefix cache, on
    /// a chunk or a complete; zero when it reports none.
    pub fn cached_tokens(&self) -> u32 {
        match self {
            Self::Sglang(resp) => match &resp.response {
                Some(sglang::generate_response::Response::Chunk(c)) => c.cached_tokens,
                Some(sglang::generate_response::Response::Complete(c)) => c.cached_tokens,
                None => 0,
            },
            Self::Vllm(resp) => match &resp.response {
                Some(vllm::generate_response::Response::Chunk(c)) => c.cached_tokens,
                Some(vllm::generate_response::Response::Complete(c)) => c.cached_tokens,
                None => 0,
            },
            Self::Trtllm(resp) => match &resp.response {
                Some(trtllm::generate_response::Response::Chunk(c)) => c.cached_tokens,
                Some(trtllm::generate_response::Response::Complete(c)) => c.cached_tokens,
                None => 0,
            },
            Self::Mlx(resp) => match &resp.response {
                Some(mlx::generate_response::Response::Chunk(c)) => c.cached_tokens,
                Some(mlx::generate_response::Response::Complete(c)) => c.cached_tokens,
                None => 0,
            },
            Self::TokenSpeed(resp) => match &resp.response {
                Some(tokenspeed::generate_response::Response::Chunk(c)) => c.cached_tokens,
                Some(tokenspeed::generate_response::Response::Complete(c)) => c.cached_tokens,
                None => 0,
            },
        }
    }

    /// Get the response variant (chunk, complete, or error)
    ///
    /// Consumes self to avoid cloning large proto messages in hot streaming path
    pub fn into_response(self) -> ProtoResponseVariant {
        match self {
            Self::Sglang(resp) => match resp.response {
                Some(sglang::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Sglang(chunk))
                }
                Some(sglang::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Sglang(complete))
                }
                None => ProtoResponseVariant::None,
            },
            Self::Vllm(resp) => match resp.response {
                Some(vllm::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Vllm(chunk))
                }
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Vllm(Box::new(complete)))
                }
                None => ProtoResponseVariant::None,
            },
            Self::Trtllm(resp) => match resp.response {
                Some(trtllm::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Trtllm(chunk))
                }
                Some(trtllm::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Trtllm(complete))
                }
                None => ProtoResponseVariant::None,
            },
            Self::Mlx(resp) => match resp.response {
                Some(mlx::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Mlx(chunk))
                }
                Some(mlx::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Mlx(complete))
                }
                None => ProtoResponseVariant::None,
            },
            Self::TokenSpeed(resp) => match resp.response {
                Some(tokenspeed::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::TokenSpeed(chunk))
                }
                Some(tokenspeed::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::TokenSpeed(complete))
                }
                None => ProtoResponseVariant::None,
            },
        }
    }
}

/// Response variant extracted from GenerateResponse
pub enum ProtoResponseVariant {
    Chunk(ProtoGenerateStreamChunk),
    Complete(ProtoGenerateComplete),
    None,
}

/// Unified GenerateStreamChunk
#[derive(Clone)]
pub enum ProtoGenerateStreamChunk {
    Sglang(sglang::GenerateStreamChunk),
    Vllm(vllm::GenerateStreamChunk),
    Trtllm(trtllm::GenerateStreamChunk),
    Mlx(mlx::GenerateStreamChunk),
    TokenSpeed(tokenspeed::GenerateStreamChunk),
}

impl ProtoGenerateStreamChunk {
    /// Get SGLang variant (panics if not SGLang)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang(&self) -> &sglang::GenerateStreamChunk {
        match self {
            Self::Sglang(chunk) => chunk,
            _ => panic!("Expected SGLang GenerateStreamChunk"),
        }
    }

    /// Get vLLM variant (panics if not vLLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_vllm() check"
    )]
    pub fn as_vllm(&self) -> &vllm::GenerateStreamChunk {
        match self {
            Self::Vllm(chunk) => chunk,
            _ => panic!("Expected vLLM GenerateStreamChunk"),
        }
    }

    /// Get TensorRT-LLM variant (panics if not TensorRT-LLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_trtllm() check"
    )]
    pub fn as_trtllm(&self) -> &trtllm::GenerateStreamChunk {
        match self {
            Self::Trtllm(chunk) => chunk,
            _ => panic!("Expected TensorRT-LLM GenerateStreamChunk"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is TensorRT-LLM
    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    /// Check if this is MLX
    pub fn is_mlx(&self) -> bool {
        matches!(self, Self::Mlx(_))
    }

    /// Check if this is TokenSpeed
    pub fn is_tokenspeed(&self) -> bool {
        matches!(self, Self::TokenSpeed(_))
    }

    /// How this chunk's payloads relate to the preceding ones — see
    /// [`ChunkSemantics`].
    pub fn chunk_semantics(&self) -> ChunkSemantics {
        match self {
            Self::Vllm(_) => ChunkSemantics::Delta,
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                ChunkSemantics::Cumulative
            }
        }
    }

    /// Get token IDs from chunk (common field)
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.token_ids,
            Self::Vllm(c) => &c.token_ids,
            Self::Trtllm(c) => &c.token_ids,
            Self::Mlx(c) => &c.token_ids,
            Self::TokenSpeed(c) => &c.token_ids,
        }
    }

    /// Get index (for n>1 support)
    /// Returns the index of this output when n>1 was requested (0-indexed)
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(c) => c.index,
            Self::Trtllm(c) => c.sequence_index,
            Self::Mlx(c) => c.index,
            Self::TokenSpeed(c) => c.index,
        }
    }

    /// Get output logprobs.
    pub fn output_logprobs(&self) -> Option<ProtoOutputLogProbs> {
        match self {
            Self::Sglang(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::Vllm(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::Trtllm(c) => convert_trtllm_output_logprobs(&c.logprobs),
            Self::Mlx(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::TokenSpeed(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
        }
    }

    /// Get input logprobs (SGLang and vLLM only - streaming chunks don't have prompt logprobs)
    pub fn input_logprobs(&self) -> Option<ProtoInputLogProbs> {
        match self {
            Self::Sglang(c) => c
                .input_logprobs
                .as_ref()
                .map(|lp| convert_input_logprobs!(lp)),
            Self::Vllm(c) => c
                .input_logprobs
                .as_ref()
                .map(|lp| convert_input_logprobs!(lp)),
            // TRT-LLM, MLX, and TokenSpeed streaming chunks don't have input_logprobs
            Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => None,
        }
    }

    /// Get prompt tokens (cumulative)
    pub fn prompt_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens,
            Self::Trtllm(c) => c.prompt_tokens,
            Self::Mlx(c) => c.prompt_tokens,
            Self::TokenSpeed(c) => c.prompt_tokens,
        }
    }

    /// Get completion tokens (cumulative)
    pub fn completion_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens,
            Self::Trtllm(c) => c.completion_tokens,
            Self::Mlx(c) => c.completion_tokens,
            Self::TokenSpeed(c) => c.completion_tokens,
        }
    }

    /// Get cached tokens (cumulative)
    pub fn cached_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens,
            Self::Trtllm(c) => c.cached_tokens,
            Self::Mlx(c) => c.cached_tokens,
            Self::TokenSpeed(c) => c.cached_tokens,
        }
    }

    /// Get reasoning tokens (cumulative).
    pub fn reasoning_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.reasoning_tokens,
            Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => 0,
        }
    }

    /// Weight version the engine reported for this chunk. Only TokenSpeed
    /// carries it on the wire; see [`is_reported_version`] for what counts.
    pub fn weight_version(&self) -> Option<&str> {
        match self {
            Self::TokenSpeed(c) => c
                .weight_version
                .as_deref()
                .filter(|v| is_reported_version(v)),
            Self::Sglang(_) | Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) => None,
        }
    }
}

/// Whether an engine-stamped `weight_version` names a real version. TokenSpeed
/// always stamps `server_args.weight_version`, whose default is the literal
/// `"default"`: that is the engine saying it was never given a version, so it
/// must not shadow a registration label, exactly like an empty string.
fn is_reported_version(version: &str) -> bool {
    !version.is_empty() && version != "default"
}

/// Unified GenerateComplete response
#[derive(Clone)]
pub enum ProtoGenerateComplete {
    Sglang(sglang::GenerateComplete),
    // Boxed: the media identity a PD prefill side returns made this variant
    // several times the size of the others, and completions move through
    // the same channel as every streamed chunk.
    Vllm(Box<vllm::GenerateComplete>),
    Trtllm(trtllm::GenerateComplete),
    Mlx(mlx::GenerateComplete),
    TokenSpeed(tokenspeed::GenerateComplete),
}

impl ProtoGenerateComplete {
    /// Get SGLang variant (panics if not SGLang)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang(&self) -> &sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            _ => panic!("Expected SGLang GenerateComplete"),
        }
    }

    /// Get mutable SGLang variant (panics if not SGLang)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_sglang() check"
    )]
    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            _ => panic!("Expected SGLang GenerateComplete"),
        }
    }

    /// Get vLLM variant (panics if not vLLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_vllm() check"
    )]
    pub fn as_vllm(&self) -> &vllm::GenerateComplete {
        match self {
            Self::Vllm(complete) => complete,
            _ => panic!("Expected vLLM GenerateComplete"),
        }
    }

    /// Get TensorRT-LLM variant (panics if not TensorRT-LLM)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via is_trtllm() check"
    )]
    pub fn as_trtllm(&self) -> &trtllm::GenerateComplete {
        match self {
            Self::Trtllm(complete) => complete,
            _ => panic!("Expected TensorRT-LLM GenerateComplete"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is TensorRT-LLM
    pub fn is_trtllm(&self) -> bool {
        matches!(self, Self::Trtllm(_))
    }

    /// Check if this is MLX
    pub fn is_mlx(&self) -> bool {
        matches!(self, Self::Mlx(_))
    }

    /// Check if this is TokenSpeed
    pub fn is_tokenspeed(&self) -> bool {
        matches!(self, Self::TokenSpeed(_))
    }

    /// Chunk semantics of the stream this `Complete` terminates — see
    /// [`ChunkSemantics`]. `Delta` streams already accumulated their counts
    /// chunk by chunk; `Cumulative` streams report them here.
    pub fn chunk_semantics(&self) -> ChunkSemantics {
        match self {
            Self::Vllm(_) => ChunkSemantics::Delta,
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                ChunkSemantics::Cumulative
            }
        }
    }

    /// Get token IDs from either backend (output_ids in proto)
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
            Self::Trtllm(c) => &c.output_token_ids,
            Self::Mlx(c) => &c.output_ids,
            Self::TokenSpeed(c) => &c.output_ids,
        }
    }

    /// Get prompt tokens
    pub fn prompt_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens,
            Self::Trtllm(c) => c.prompt_tokens,
            Self::Mlx(c) => c.prompt_tokens,
            Self::TokenSpeed(c) => c.prompt_tokens,
        }
    }

    /// Get completion tokens
    pub fn completion_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens,
            Self::Trtllm(c) => c.completion_tokens,
            Self::Mlx(c) => c.completion_tokens,
            Self::TokenSpeed(c) => c.completion_tokens,
        }
    }

    /// Get finish reason
    pub fn finish_reason(&self) -> &str {
        match self {
            Self::Sglang(c) => &c.finish_reason,
            Self::Vllm(c) => &c.finish_reason,
            Self::Trtllm(c) => &c.finish_reason,
            Self::Mlx(c) => &c.finish_reason,
            Self::TokenSpeed(c) => &c.finish_reason,
        }
    }

    /// Get index (for n>1 support)
    /// Returns the index of this output when n>1 was requested (0-indexed)
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(c) => c.index,
            Self::Trtllm(c) => c.sequence_index,
            Self::Mlx(c) => c.index,
            Self::TokenSpeed(c) => c.index,
        }
    }

    /// Get matched stop as a JSON value
    ///
    /// Converts the backend-specific `oneof matched_stop` into a `serde_json::Value`:
    /// - MatchedTokenId → Number
    /// - MatchedStopStr → String
    /// - None → None
    pub fn matched_stop_json(&self) -> Option<serde_json::Value> {
        macro_rules! convert {
            ($oneof:expr, $token_id:path, $stop_str:path) => {
                $oneof.as_ref().map(|m| match m {
                    $token_id(id) => serde_json::Value::Number((*id).into()),
                    $stop_str(s) => serde_json::Value::String(s.clone()),
                })
            };
        }
        match self {
            Self::Sglang(c) => convert!(
                &c.matched_stop,
                SglangMatchedStop::MatchedTokenId,
                SglangMatchedStop::MatchedStopStr
            ),
            Self::Vllm(c) => convert!(
                &c.matched_stop,
                VllmMatchedStop::MatchedTokenId,
                VllmMatchedStop::MatchedStopStr
            ),
            Self::Trtllm(c) => convert!(
                &c.matched_stop,
                TrtllmMatchedStop::MatchedTokenId,
                TrtllmMatchedStop::MatchedStopStr
            ),
            Self::Mlx(c) => c
                .matched_stop_token_id
                .map(|id| serde_json::Value::Number(id.into())),
            Self::TokenSpeed(c) => convert!(
                &c.matched_stop,
                TokenSpeedMatchedStop::MatchedTokenId,
                TokenSpeedMatchedStop::MatchedStopStr
            ),
        }
    }

    /// Get output IDs (decode tokens only)
    pub fn output_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
            Self::Trtllm(c) => &c.output_token_ids,
            Self::Mlx(c) => &c.output_ids,
            Self::TokenSpeed(c) => &c.output_ids,
        }
    }

    /// Get cached tokens
    pub fn cached_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens,
            Self::Trtllm(c) => c.cached_tokens,
            Self::Mlx(c) => c.cached_tokens,
            Self::TokenSpeed(c) => c.cached_tokens,
        }
    }

    /// Get reasoning tokens.
    pub fn reasoning_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.reasoning_tokens,
            Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => 0,
        }
    }

    /// Weight version the engine reported for this completion. Only TokenSpeed
    /// carries it on the wire; see [`is_reported_version`] for what counts.
    pub fn weight_version(&self) -> Option<&str> {
        match self {
            Self::TokenSpeed(c) => c
                .weight_version
                .as_deref()
                .filter(|v| is_reported_version(v)),
            Self::Sglang(_) | Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) => None,
        }
    }

    /// Get accepted speculative draft tokens.
    pub fn spec_accepted_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.spec_accepted_tokens,
            Self::Vllm(c) => c.spec_accepted_tokens,
            Self::Trtllm(c) => c.spec_accepted_tokens,
            Self::TokenSpeed(c) => c.spec_accepted_tokens,
            Self::Mlx(_) => 0,
        }
    }

    /// Get proposed speculative draft tokens.
    pub fn spec_draft_tokens(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.spec_draft_tokens,
            Self::Vllm(c) => c.spec_draft_tokens,
            Self::Trtllm(c) => c.spec_draft_tokens,
            Self::TokenSpeed(c) => c.spec_draft_tokens,
            Self::Mlx(_) => 0,
        }
    }

    /// Get input/prompt logprobs (SGLang, vLLM, and TensorRT-LLM)
    pub fn input_logprobs(&self) -> Option<ProtoInputLogProbs> {
        match self {
            Self::Sglang(c) => c
                .input_logprobs
                .as_ref()
                .map(|lp| convert_input_logprobs!(lp)),
            Self::Vllm(c) => c
                .input_logprobs
                .as_ref()
                .map(|lp| convert_input_logprobs!(lp)),
            Self::Trtllm(c) => {
                if c.prompt_logprobs.is_empty() {
                    None
                } else {
                    Some(ProtoInputLogProbs {
                        // First token has None logprob (no prior context)
                        token_logprobs: c
                            .prompt_logprobs
                            .iter()
                            .enumerate()
                            .map(|(i, lp)| if i == 0 { None } else { Some(lp.logprob) })
                            .collect(),
                        token_ids: c.prompt_logprobs.iter().map(|lp| lp.token_id).collect(),
                        top_logprobs: c
                            .prompt_logprobs
                            .iter()
                            .map(|lp| ProtoTopLogProbs {
                                values: lp.top_logprobs.iter().map(|t| t.logprob).collect(),
                                token_ids: lp.top_logprobs.iter().map(|t| t.token_id).collect(),
                            })
                            .collect(),
                    })
                }
            }
            // MLX and TokenSpeed do not have input_logprobs
            Self::Mlx(_) | Self::TokenSpeed(_) => None,
        }
    }

    /// Get output logprobs.
    pub fn output_logprobs(&self) -> Option<ProtoOutputLogProbs> {
        match self {
            Self::Sglang(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::Vllm(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::Trtllm(c) => convert_trtllm_output_logprobs(&c.logprobs),
            Self::Mlx(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
            Self::TokenSpeed(c) => c
                .output_logprobs
                .as_ref()
                .map(|lp| convert_output_logprobs!(lp)),
        }
    }

    /// Get KV transfer parameters from prefill response (vLLM Mooncake PD only).
    /// Returns (remote_host, remote_port) if present.
    pub fn kv_transfer_params(&self) -> Option<(String, u32)> {
        match self {
            Self::Vllm(c) => c
                .kv_transfer_params
                .as_ref()
                .map(|params| (params.remote_host.clone(), params.remote_port)),
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => None,
        }
    }

    /// The media identity a prefill side returned for its processed
    /// references (vLLM only; absent from older servicers).
    pub fn media_identity(&self) -> Option<&vllm::MediaIdentity> {
        match self {
            Self::Vllm(c) => c.media_identity.as_ref(),
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => None,
        }
    }

    /// Get opaque connector KV-transfer params JSON returned by the engine (vLLM only).
    pub fn kv_transfer_params_json(&self) -> Option<&str> {
        match self {
            Self::Vllm(c) => c
                .kv_transfer_params_json
                .as_deref()
                .filter(|s| !s.is_empty()),
            Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => None,
        }
    }
}

/// Unified stream wrapper.
///
/// One variant per backend. Each yields its own native proto response shape;
/// the chunk / complete accessors above match on the corresponding
/// `ProtoGenerateStreamChunk` / `ProtoGenerateComplete` arm.
pub enum ProtoStream {
    Sglang(SglangStream),
    Vllm(VllmStream),
    Trtllm(TrtllmStream),
    Mlx(MlxStream),
    TokenSpeed(TokenSpeedStream),
    /// ZMQ backend: a custom stream yielding vLLM-proto responses built from
    /// EngineCore outputs. Auto-aborts on drop, so `mark_completed` is a no-op.
    Zmq(ZmqGenerateStream),
    /// An n>1 fan-out over rendezvous-room PD pairs: one child per sample,
    /// each response stamped with its sample's index (see [`FanoutStream`]).
    Fanout(FanoutStream),
    /// Any of the above, with the worker it came from: every response is a
    /// sign of life for that worker (see [`crate::worker::liveness`]).
    Tracked(Box<TrackedStream>),
    /// The decode side of a disaggregated request: every response restamped
    /// with the prefill side's prefix-cache hit (see [`PrefillCachedTokens`]).
    DecodeSide(Box<DecodeStream>),
    /// A stream whose first item was pulled ahead of its reader (see
    /// [`ProtoStream::first_item_ahead`]): replayed once, then the stream as
    /// it was.
    Ahead(Box<AheadStream>),
    /// Any of the above, read for a client that can leave: the moment the
    /// client's response body is dropped, the inner stream is dropped (its
    /// abort-on-drop reaches the engine, its tickets go back to the worker)
    /// and the reader meets the end (see [`WatchedStream`]).
    Watched(Box<WatchedStream>),
}

/// A stream with its first item (`None`: its end) pulled ahead of its reader
/// and held, with nothing booked for it: whatever tracks the stream later
/// books the item when its reader meets it.
pub struct AheadStream {
    inner: ProtoStream,
    first: Option<Result<ProtoGenerateResponse, tonic::Status>>,
}

/// A [`ProtoStream`] read on behalf of one client, whose response body is
/// the receiving end of `client`. A reader parked on the engine (a
/// generation that stalled mid-stream, a first token still owed) would
/// otherwise learn that the client left only when the engine next speaks,
/// or when the request timeout ends the attempt: its send fails then, and
/// the stream it drops at its return sends the abort. Here every poll of the
/// stream races the engine against the client's leaving, so the abort
/// reaches the engine the moment the body is dropped: the inner stream goes
/// (its abort-on-drop and its tickets with it) and the reader meets the
/// end, the path it takes today when the engine ends a stream whose client
/// had already left.
pub struct WatchedStream {
    /// `None` once the client left: the inner stream was dropped there.
    inner: Option<ProtoStream>,
    client: SseSender,
    request_id: String,
}

impl WatchedStream {
    /// The client left: drop the inner stream, which sends its abort to the
    /// engine and gives back its tickets.
    fn client_left(&mut self) {
        self.inner = None;
        tracing::debug!(
            request_id = %self.request_id,
            "Client left during the stream; the request was aborted on the engine"
        );
    }
}

/// A disaggregated request's decode side. Its engine loads the prompt's KV
/// from the prefill side and reports what it loaded as cached, so every chunk
/// and complete it yields is restamped with the prefill side's prefix-cache
/// hit before the gateway reads it. Decode sides are not tracked today; if one
/// ever is, the tracking must sit inside this wrapper, so that a
/// [`PrefillTicket`] reads the engine's own `cached_tokens`, not the
/// restamped value.
pub struct DecodeStream {
    inner: ProtoStream,
    prefill: PrefillCachedTokens,
}

/// The prefix-cache hit of a disaggregated request, by sample index, as its
/// prefill side reported it on each `Complete`. The prefill side is the one
/// that computed the prompt against its prefix cache; the decode side receives
/// that prefix over the KV connector and counts it as cached, so read from the
/// decode side every request's `cached_tokens` is the transfer (the prompt
/// less its last token), whatever the cache held. The decode side's responses
/// are restamped with these values (see [`ProtoStream::decode_side`]) so the
/// client's usage, and everything the gateway derives from it, carries the
/// request's real cache hit.
#[derive(Debug, Clone, Default)]
pub struct PrefillCachedTokens {
    by_index: HashMap<u32, u32>,
}

impl PrefillCachedTokens {
    /// Record a prefill `Complete`'s cache hit under its sample index.
    pub fn note(&mut self, complete: &ProtoGenerateComplete) {
        self.by_index
            .insert(complete.index(), complete.cached_tokens());
    }

    /// Whether no prefill completion was recorded (a stream another stage
    /// drained, or a prefill side that ended without one).
    pub fn is_empty(&self) -> bool {
        self.by_index.is_empty()
    }

    /// The prefill side's cache hit for sample `index`: the sample's own
    /// entry, or the one entry of a prefill that completed once for the
    /// whole request.
    pub fn for_index(&self, index: u32) -> Option<u32> {
        match self.by_index.get(&index) {
            Some(cached) => Some(*cached),
            None if self.by_index.len() == 1 => self.by_index.values().next().copied(),
            None => None,
        }
    }

    fn restamp(&self, index: u32, cached: &mut u32) {
        if let Some(value) = self.for_index(index) {
            *cached = value;
        }
    }
}

/// A [`ProtoStream`] paired with the worker serving it, so each response it
/// yields counts as token progress for that worker, and its prompt counts as
/// pending prefill on the worker until the first response.
pub struct TrackedStream {
    inner: ProtoStream,
    worker: Arc<dyn Worker>,
    prefill: Option<PrefillTicket>,
    /// The request's place in the worker's pile; a non-streaming generation
    /// holds none.
    progress: Option<ProgressTicket>,
    /// Held from the stream's first response to its end: one of the streams
    /// the worker is producing for (see [`Worker::output_silence`]).
    output: Option<OutputTicket>,
    /// Whether the worker's circuit-breaker success for this dispatch is
    /// still to be sampled, at the stream's first response (a disaggregated
    /// decode side: see [`ProtoStream::tracked_decode_side`]).
    samples_success: bool,
    /// A response (or the stream's end) pulled ahead of the stream's reader
    /// by [`ProtoStream::first_response_within`], handed to the next `next`
    /// as it was: its bookkeeping ran when it was pulled.
    replay: Option<Option<Result<ProtoGenerateResponse, tonic::Status>>>,
}

/// What [`ProtoStream::first_response_within`] found as the stream's first
/// item, held for the reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FirstItem {
    /// A response: the side answered.
    Response,
    /// A failure (the engine's own report, or the transport's): the reader
    /// meets it next and accounts for it.
    Failure,
    /// The end: the stream closed with nothing on it.
    End,
}

impl FirstItem {
    fn of(item: Option<&Result<ProtoGenerateResponse, tonic::Status>>) -> Self {
        match item {
            Some(Ok(_)) => Self::Response,
            Some(Err(_)) => Self::Failure,
            None => Self::End,
        }
    }
}

/// The request's place in its worker's tracked pile (see
/// [`Worker::tracked_load`]), held for the stream's life.
struct ProgressTicket {
    worker: Arc<dyn Worker>,
}

impl Drop for ProgressTicket {
    fn drop(&mut self) {
        self.worker.note_tracked_ended();
    }
}

/// A stream the worker has produced output for and that has not ended (see
/// [`Worker::output_silence`]): opened at the first response, given back at
/// the stream's end, or when its reader lets go of it.
struct OutputTicket {
    worker: Arc<dyn Worker>,
}

impl OutputTicket {
    fn open(worker: &Arc<dyn Worker>) -> Self {
        worker.note_output_started();
        Self {
            worker: Arc::clone(worker),
        }
    }
}

impl Drop for OutputTicket {
    fn drop(&mut self) {
        self.worker.note_output_ended();
    }
}

/// The prompt tokens a dispatched request adds to its worker's prefill
/// backlog (see [`Worker::prefill_backlog`]), released by the first response
/// (with the prefix-cache hit it reports, so the worker's rate learns the
/// tokens the engine computed) or, failing that, when the stream is dropped.
pub(crate) struct PrefillTicket {
    worker: Arc<dyn Worker>,
    tokens: u64,
    /// The first response is the request's first token (a streaming
    /// generation), so the prompt's prefill is timed for the worker's rate;
    /// a non-streaming generation's one response is its whole answer.
    timed: bool,
    released: bool,
}

impl PrefillTicket {
    /// Book `prompt_tokens` as pending prefill on `worker` from this moment,
    /// timed: the dispatch paths book a side before its call is polled, so the
    /// span the worker's rate learns runs from the dispatch to the first
    /// response whether the servicer sends its headers at once or, as the
    /// gRPC servicers do, with the first message (a booking at the call's
    /// return would then time nothing and teach a rate the worker does not
    /// have, and the sides waiting in dispatch would be missing from the
    /// backlog). `None` for an empty prompt. The ticket is handed to the side's
    /// stream by [`ProtoStream::tracked_prefill_side_booked`]; dropped before
    /// that (the call failed or was retired), it releases the prompt untimed.
    pub(crate) fn book(worker: &Arc<dyn Worker>, prompt_tokens: u64) -> Option<Self> {
        Self::book_as(worker, prompt_tokens, true)
    }

    /// [`Self::book`] with the prompt's prefill `timed` or not: a regular
    /// dispatch times it when the engine streams the generation (the first
    /// response is the first token) and books it untimed otherwise (the one
    /// response of a non-streaming generation is its whole answer).
    pub(crate) fn book_as(
        worker: &Arc<dyn Worker>,
        prompt_tokens: u64,
        timed: bool,
    ) -> Option<Self> {
        (prompt_tokens > 0).then(|| {
            worker.note_prefill_started(prompt_tokens, timed);
            Self {
                worker: Arc::clone(worker),
                tokens: prompt_tokens,
                timed,
                released: false,
            }
        })
    }

    /// The first response came back, reporting `cached` prompt tokens served
    /// from the engine's prefix cache.
    fn first_response(&mut self, cached: u64) {
        if !self.released {
            self.released = true;
            self.worker
                .note_prefill_ended(self.tokens, self.timed, Some(cached));
        }
    }
}

impl Drop for PrefillTicket {
    fn drop(&mut self) {
        if !self.released {
            self.worker
                .note_prefill_ended(self.tokens, self.timed, None);
        }
    }
}

/// What a regular dispatch books on its worker before its call is polled
/// (see [`PrefillTicket::book`]): the prompt as pending prefill, timed when
/// the engine streams the generation, and, for a streaming generation, the
/// request's place in the worker's pile. [`ProtoStream::tracked_booked`]
/// hands both to the call's stream; dropped before that (the call failed),
/// the booking gives both back.
pub(crate) struct DispatchBooking {
    prefill: Option<PrefillTicket>,
    progress: Option<ProgressTicket>,
}

impl DispatchBooking {
    pub(crate) fn book(worker: &Arc<dyn Worker>, prompt_tokens: u64, streaming: bool) -> Self {
        let progress = streaming.then(|| {
            worker.note_tracked_started();
            ProgressTicket {
                worker: Arc::clone(worker),
            }
        });
        Self {
            prefill: PrefillTicket::book_as(worker, prompt_tokens, streaming),
            progress,
        }
    }
}

#[cfg(test)]
mod tracked_tests {
    use std::sync::Arc;

    use super::{
        sglang, tokenspeed, vllm, DispatchBooking, FanoutStream, FirstItem, PrefillTicket,
        ProtoGenerateResponse, ProtoStream,
    };
    use crate::worker::{BasicWorkerBuilder, Worker};

    #[test]
    fn the_cache_hit_a_first_response_reports_is_read_from_a_chunk_or_a_complete() {
        // The prefill ticket hands the worker the hit with the first
        // response, whichever shape it has; a response that carries no
        // count reads as no hit, and the prompt teaches its full length.
        let chunk = ProtoGenerateResponse::Vllm(Box::new(vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Chunk(
                vllm::GenerateStreamChunk {
                    cached_tokens: 1_900,
                    ..Default::default()
                },
            )),
        }));
        assert_eq!(chunk.cached_tokens(), 1_900);
        let complete = ProtoGenerateResponse::Sglang(Box::new(sglang::GenerateResponse {
            response: Some(sglang::generate_response::Response::Complete(
                sglang::GenerateComplete {
                    cached_tokens: 64,
                    ..Default::default()
                },
            )),
            ..Default::default()
        }));
        assert_eq!(complete.cached_tokens(), 64);
        let empty = ProtoGenerateResponse::Vllm(Box::default());
        assert_eq!(empty.cached_tokens(), 0);
    }

    #[test]
    fn a_tracked_stream_is_one_of_its_workers_pile_for_its_life() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w1:8000").build());
        let stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new())).tracked(
            Arc::clone(&worker),
            0,
            true,
        );
        assert_eq!(worker.tracked_load(), 1, "counted at dispatch");
        let stream = stream.defer_abort_until_first_item();
        assert_eq!(worker.tracked_load(), 1, "and across the stream's rewraps");
        drop(stream);
        assert_eq!(worker.tracked_load(), 0, "released with the stream");
    }

    #[tokio::test]
    async fn a_tracked_stream_inside_a_tracked_stream_ends_and_releases_both_piles() {
        // The in-place poll of `next` covers one tracked layer over a
        // transport stream; a second layer takes the recursive path. Both
        // layers' bookkeeping holds either way.
        let outer: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w1:8000").build());
        let inner: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w2:8000").build());
        let mut stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
            .tracked(Arc::clone(&inner), 0, true)
            .tracked(Arc::clone(&outer), 0, true);
        assert_eq!((inner.tracked_load(), outer.tracked_load()), (1, 1));
        assert!(
            stream.next().await.is_none(),
            "an empty fanout ends at once"
        );
        drop(stream);
        assert_eq!((inner.tracked_load(), outer.tracked_load()), (0, 0));
    }

    #[test]
    fn a_non_streaming_generation_joins_no_pile_but_keeps_its_prefill_and_its_progress() {
        // Review finding on the pushed head: left untracked altogether, a
        // non-streaming generation's one answer no longer counted as progress
        // and its prompt left the prefill backlog, so a saturated engine
        // finishing such requests while streaming ones waited was wedged. It
        // is tracked like any other, minus the pile slot.
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w1:8000").build());
        let stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new())).tracked(
            Arc::clone(&worker),
            64,
            false,
        );
        assert_eq!(
            worker.tracked_load(),
            0,
            "nothing to see progress on until it answers"
        );
        match &stream {
            ProtoStream::Tracked(tracked) => {
                assert!(tracked.progress.is_none(), "no pile slot");
                assert!(
                    tracked.prefill.as_ref().is_some_and(|ticket| !ticket.timed),
                    "pending prefill like any other, untimed: its first response is its whole answer"
                );
                // Its one answer goes through the same `Tracked` arm of
                // `next` as a token, which is where progress is recorded.
            }
            _ => panic!("a tracked stream"),
        }
        let stream = stream.defer_abort_until_first_item();
        assert_eq!(worker.tracked_load(), 0);
        drop(stream);
        assert_eq!(worker.tracked_load(), 0);
    }

    /// A disaggregated decode side earns its worker's breaker success at its
    /// first response, never at its dispatch: a side whose stream ends with
    /// nothing on it leaves a failure before it standing, so the next one
    /// opens a breaker set to two.
    #[tokio::test]
    async fn a_decode_side_that_ends_with_nothing_on_it_earns_no_breaker_success() {
        use crate::worker::circuit_breaker::{CircuitBreakerConfig, CircuitState};
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:30000")
                .circuit_breaker_config(CircuitBreakerConfig {
                    failure_threshold: 2,
                    ..Default::default()
                })
                .build(),
        );
        worker.record_outcome(504);
        let mut stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
            .tracked_decode_side(Arc::clone(&worker));
        match &stream {
            ProtoStream::Tracked(tracked) => {
                assert!(tracked.samples_success, "the success waits for a response");
                assert!(tracked.progress.is_none(), "no pile slot");
                assert!(tracked.prefill.is_none(), "no prompt booked");
            }
            _ => panic!("a tracked stream"),
        }
        assert!(
            stream.next().await.is_none(),
            "an empty fanout ends at once"
        );
        drop(stream);
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
        worker.record_outcome(504);
        assert_eq!(
            worker.circuit_breaker_state(),
            CircuitState::Open,
            "two failures with no success earned between them"
        );
    }

    #[test]
    fn a_prefill_side_booked_at_dispatch_is_pending_before_its_stream_exists() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://p1:8000").build());
        assert!(
            PrefillTicket::book(&worker, 0).is_none(),
            "an empty prompt books nothing"
        );
        let ticket = PrefillTicket::book(&worker, 4_096);
        assert_eq!(
            worker.prefill_backlog(),
            std::time::Duration::from_millis(409),
            "pending from the dispatch, before the call has answered"
        );
        let stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
            .tracked_prefill_side_booked(Arc::clone(&worker), ticket);
        assert_eq!(
            worker.prefill_backlog(),
            std::time::Duration::from_millis(409),
            "handed to the stream: booked once, not again"
        );
        assert_eq!(worker.tracked_load(), 0, "a side is never part of a pile");
        match &stream {
            ProtoStream::Tracked(tracked) => {
                assert!(tracked.prefill.as_ref().is_some_and(|ticket| ticket.timed));
            }
            _ => panic!("a tracked stream"),
        }
        drop(stream);
        assert_eq!(
            worker.prefill_backlog(),
            std::time::Duration::ZERO,
            "released with the stream"
        );
    }

    #[test]
    fn a_side_booked_at_dispatch_whose_call_fails_releases_its_prompt() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://p1:8000").build());
        let ticket = PrefillTicket::book(&worker, 1_000);
        assert_eq!(
            worker.prefill_backlog(),
            std::time::Duration::from_millis(100)
        );
        drop(ticket);
        assert_eq!(
            worker.prefill_backlog(),
            std::time::Duration::ZERO,
            "a call that fails or is retired gives the prompt back untimed"
        );
    }

    #[test]
    fn a_prefill_side_books_its_prompt_timed_and_joins_no_pile() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://p1:8000").build());
        let stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
            .tracked_prefill_side(Arc::clone(&worker), 4_096);
        assert_eq!(worker.tracked_load(), 0, "a side is never part of a pile");
        assert_eq!(
            worker.prefill_backlog(),
            std::time::Duration::from_millis(409),
            "its prompt is pending prefill on the prefill worker (cold prior)"
        );
        match &stream {
            ProtoStream::Tracked(tracked) => {
                assert!(tracked.progress.is_none());
                assert!(tracked.prefill.as_ref().is_some_and(|ticket| ticket.timed));
            }
            _ => panic!("a tracked stream"),
        }
        drop(stream);
        assert_eq!(
            worker.prefill_backlog(),
            std::time::Duration::ZERO,
            "released with the stream"
        );
    }

    #[tokio::test]
    async fn the_first_response_pulled_ahead_is_replayed_to_the_reader_once() {
        // An empty fan-out ends at once: the end is what is pulled ahead, and
        // the reader sees it once more, then the stream as it is.
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://p1:8000").build());
        let mut stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
            .tracked_prefill_side(Arc::clone(&worker), 16);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        assert_eq!(
            stream.first_response_within(deadline).await.ok(),
            Some(FirstItem::End),
            "what was pulled ahead is named: the end, not a response"
        );
        assert_eq!(
            stream.first_response_within(deadline).await.ok(),
            Some(FirstItem::End),
            "a second wait does not pull past the held item"
        );
        assert!(stream.next().await.is_none(), "the held end is replayed");
        assert!(stream.next().await.is_none(), "and the stream stays ended");
    }

    /// The parallel decode side's first item, pulled ahead while its prefill
    /// side is awaited, is held unbooked and replayed once to its reader,
    /// then the stream as it was; whatever tracks the stream then books it
    /// at the replay: the decode worker's output, and its success.
    #[tokio::test]
    async fn a_first_item_pulled_ahead_is_replayed_once_and_booked_by_its_reader() {
        use crate::worker::circuit_breaker::{CircuitBreakerConfig, CircuitState};
        type Item = Result<ProtoGenerateResponse, tonic::Status>;
        // Chunks told apart by the count each carries.
        let chunk = |cached_tokens| -> Item {
            let chunk = vllm::GenerateStreamChunk {
                cached_tokens,
                ..Default::default()
            };
            let response = Some(vllm::generate_response::Response::Chunk(chunk));
            Ok(ProtoGenerateResponse::Vllm(Box::new(
                vllm::GenerateResponse { response },
            )))
        };
        let which = |item: Option<&Item>| Some(item?.as_ref().ok()?.cached_tokens());
        // A child that yields chunks 1 and 2 (each replayed by a tracked
        // layer of its own, on an engine of its own) and ends, read through
        // a fan-out: its first item is pulled from it.
        let engine: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://e:30000").build());
        let empty = ProtoStream::Fanout(FanoutStream::new(Vec::new()));
        let child = [2, 1].into_iter().fold(empty, |inner, id| {
            let mut layer = inner.tracked(Arc::clone(&engine), 0, false);
            if let ProtoStream::Tracked(tracked) = &mut layer {
                tracked.replay = Some(Some(chunk(id)));
            }
            layer
        });
        let mut stream = ProtoStream::Fanout(FanoutStream::new(vec![child]));
        assert_eq!(which(stream.first_item_ahead().await), Some(1), "pulled");
        assert_eq!(
            which(stream.first_item_ahead().await),
            Some(1),
            "held, not read again"
        );
        assert!(stream.worker().is_none());
        let stream = stream.defer_abort_until_first_item();
        assert!(stream.holds_item_ahead(), "rewrapped with what it holds");
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:30000")
                .circuit_breaker_config(CircuitBreakerConfig {
                    failure_threshold: 2,
                    ..Default::default()
                })
                .build(),
        );
        worker.record_outcome(504);
        let mut stream = stream.tracked_decode_side(Arc::clone(&worker));
        assert!(worker.output_silence().is_none(), "nothing booked unread");
        assert_eq!(which(stream.next().await.as_ref()), Some(1), "replayed");
        assert!(worker.output_silence().is_some(), "booked at the replay");
        assert_eq!(which(stream.next().await.as_ref()), Some(2), "then on");
        assert!(stream.next().await.is_none(), "to the stream's end");
        worker.record_outcome(504);
        assert_eq!(
            worker.circuit_breaker_state(),
            CircuitState::Closed,
            "a success, booked at the replay, between the two failures"
        );
    }

    /// A decode side's first item is its prefill worker's handoff only when
    /// it is output: a chunk, or a complete that ended neither in an abort
    /// (an aborted choice, as the Rust servicers end one) nor in an error.
    #[test]
    fn output_is_a_chunk_or_a_complete_the_engine_neither_aborted_nor_failed() {
        use sglang::generate_response::Response as Sg;
        use tokenspeed::generate_response::Response as Ts;
        let ts = |response: Ts| {
            let response = Some(response);
            let response = tokenspeed::GenerateResponse {
                response,
                ..Default::default()
            };
            ProtoGenerateResponse::TokenSpeed(Box::new(response)).is_output()
        };
        let sg = |response: Sg| {
            let response = Some(response);
            let response = sglang::GenerateResponse {
                response,
                ..Default::default()
            };
            ProtoGenerateResponse::Sglang(Box::new(response)).is_output()
        };
        assert!(ts(Ts::Chunk(Default::default())) && sg(Sg::Chunk(Default::default())));
        for (reasons, output) in [(["stop", "length"], true), (["abort", "error"], false)] {
            for reason in reasons {
                let finish_reason = reason.to_string();
                let complete = sglang::GenerateComplete {
                    finish_reason,
                    ..Default::default()
                };
                assert_eq!(sg(Sg::Complete(complete)), output, "{reason}");
                let finish_reason = reason.to_string();
                let complete = tokenspeed::GenerateComplete {
                    finish_reason,
                    ..Default::default()
                };
                assert_eq!(ts(Ts::Complete(complete)), output, "{reason}");
            }
        }
        let empty = ProtoGenerateResponse::TokenSpeed(Box::default());
        assert!(!empty.is_output(), "an empty response");
    }

    #[test]
    fn a_dispatch_booking_holds_the_prompt_and_the_pile_until_its_stream_takes_over() {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w1:8000").build());
        let books = |worker: &Arc<dyn Worker>| (worker.prefill_backlog(), worker.tracked_load());
        // 4,096 prompt tokens at the cold prior read as 409 ms of backlog.
        let pending = std::time::Duration::from_millis(409);
        let none = (std::time::Duration::ZERO, 0);
        // A streaming generation: its prompt timed, its place in the pile.
        let booking = DispatchBooking::book(&worker, 4_096, true);
        assert_eq!(
            books(&worker),
            (pending, 1),
            "booked at the dispatch, before the call has answered"
        );
        assert!(booking.prefill.as_ref().is_some_and(|ticket| ticket.timed));
        let stream = ProtoStream::Fanout(FanoutStream::<ProtoStream>::new(Vec::new()))
            .tracked_booked(Arc::clone(&worker), booking);
        assert_eq!(
            books(&worker),
            (pending, 1),
            "handed to the stream: booked once, not again"
        );
        drop(stream);
        assert_eq!(books(&worker), none, "released with the stream");
        // A non-streaming generation: its prompt pending untimed, no pile slot.
        let booking = DispatchBooking::book(&worker, 4_096, false);
        assert_eq!(books(&worker), (pending, 0));
        assert!(booking.prefill.as_ref().is_some_and(|ticket| !ticket.timed));
        assert!(booking.progress.is_none());
        // Dropped before a stream took it over (the call failed): given back.
        drop(booking);
        assert_eq!(books(&worker), none);
    }
}

/// The message of the `Internal` status a stream ends with when the engine
/// itself finished the request with an error. The ZMQ lane's adapters end
/// theirs with the same words, and the stream-failure accounting keys its
/// `engine_error` label on it.
pub(crate) const ENGINE_ERROR_MESSAGE: &str =
    "engine finished the request with an error (see engine logs)";

/// Surface an engine-side failure (`finish_reason == "error"`) as a stream error, like the ZMQ lane.
fn reject_engine_error(
    item: Result<ProtoGenerateResponse, tonic::Status>,
) -> Result<ProtoGenerateResponse, tonic::Status> {
    match item {
        Ok(response) if response.engine_error() => {
            Err(tonic::Status::internal(ENGINE_ERROR_MESSAGE))
        }
        item => item,
    }
}

impl ProtoStream {
    /// Count this stream's responses as progress for `worker` (a response is
    /// progress whether it is a token of a streaming generation or the one
    /// answer of a non-streaming one), its `prompt_tokens` as pending prefill
    /// there until the first response, and, when the engine streams the
    /// generation (`streaming`), the request as one of the worker's tracked
    /// pile while the stream lives, with its prefill timed by its first
    /// token over the tokens the engine computed for it (its prompt less the
    /// prefix-cache hit that first response reports). A non-streaming
    /// generation shows nothing between dispatch and
    /// completion, so it joins no pile and times no prefill (its one answer
    /// comes after its decode): the pile rule never judges a worker by
    /// requests it cannot see progress on, and the prefill rate is learned
    /// from first tokens only.
    #[must_use]
    pub fn tracked(self, worker: Arc<dyn Worker>, prompt_tokens: u64, streaming: bool) -> Self {
        let booking = DispatchBooking::book(&worker, prompt_tokens, streaming);
        self.tracked_booked(worker, booking)
    }

    /// [`Self::tracked`] for a request booked at its dispatch (see
    /// [`DispatchBooking::book`]): the stream takes the booking over, so the
    /// prompt's span runs from before the call was polled to the first
    /// response, and the request has been in the pile since the dispatch.
    #[must_use]
    pub(crate) fn tracked_booked(self, worker: Arc<dyn Worker>, booking: DispatchBooking) -> Self {
        let DispatchBooking { prefill, progress } = booking;
        Self::Tracked(Box::new(TrackedStream {
            inner: self,
            worker,
            prefill,
            progress,
            output: None,
            samples_success: false,
            replay: None,
        }))
    }

    /// Track this stream as the prefill side of a disaggregated dispatch to
    /// `worker`: its `prompt_tokens` are pending prefill there until its
    /// first response, which times the prompt's prefill for the worker's
    /// rate (the side's first response is the end of its prefill, on either
    /// dispatch shape), and every response is progress for the worker. The
    /// side joins no pile: the gateway bounds its silence per request (see
    /// [`crate::worker::liveness::prefill_side_bound`]) instead of judging the
    /// worker by sides it cannot see progress on.
    #[must_use]
    pub fn tracked_prefill_side(self, worker: Arc<dyn Worker>, prompt_tokens: u64) -> Self {
        let prefill = PrefillTicket::book(&worker, prompt_tokens);
        self.tracked_prefill_side_booked(worker, prefill)
    }

    /// [`Self::tracked_prefill_side`] for a side booked at its dispatch (see
    /// [`PrefillTicket::book`]): the stream takes over the ticket, so the
    /// first response ends the span that began before the call was polled.
    #[must_use]
    pub(crate) fn tracked_prefill_side_booked(
        self,
        worker: Arc<dyn Worker>,
        prefill: Option<PrefillTicket>,
    ) -> Self {
        Self::Tracked(Box::new(TrackedStream {
            inner: self,
            worker,
            prefill,
            progress: None,
            output: None,
            samples_success: false,
            replay: None,
        }))
    }

    /// Track this stream as the decode side of a disaggregated dispatch to
    /// `worker`: every response is the worker's token progress, the side
    /// joins no pile and books no prefill (as [`Self::tracked`] with no
    /// prompt and no pile does), and the worker's circuit-breaker success
    /// for the dispatch is sampled at the side's first response rather than
    /// when the call returned its stream. A decode side's first response
    /// waits for the handoff, so a side accepted into silence, or one whose
    /// stream fails at once, is one failure of its worker and not a success
    /// and then a failure, which reset the breaker's streak on every
    /// request; a stream that ends with nothing on it earns no success
    /// either, as a prefill side's does not.
    #[must_use]
    pub(crate) fn tracked_decode_side(self, worker: Arc<dyn Worker>) -> Self {
        let mut stream = self.tracked(worker, 0, false);
        if let Self::Tracked(tracked) = &mut stream {
            tracked.samples_success = true;
        }
        stream
    }

    /// Wait for the stream's first item, a response or its end, until
    /// `deadline`, and say which it was. Within the deadline the stream is
    /// left for its reader as if nothing had been read: the item is replayed
    /// by the next [`Self::next`], its bookkeeping already done. Past the
    /// deadline the stream has produced nothing, and the caller decides its
    /// fate. Only a tracked stream can hold an item back, so any other
    /// returns at once as if a response had come (the dispatch paths wrap a
    /// side before they bound it).
    pub async fn first_response_within(&mut self, deadline: Instant) -> Result<FirstItem, Elapsed> {
        if let Self::Watched(watched) = self {
            return match watched.inner.as_mut() {
                Some(inner) => Box::pin(inner.first_response_within(deadline)).await,
                None => Ok(FirstItem::End),
            };
        }
        let Self::Tracked(tracked) = self else {
            return Ok(FirstItem::Response);
        };
        if let Some(held) = &tracked.replay {
            return Ok(FirstItem::of(held.as_ref()));
        }
        let item = tokio::time::timeout_at(deadline.into(), self.next()).await?;
        let first = FirstItem::of(item.as_ref());
        if let Self::Tracked(tracked) = self {
            tracked.replay = Some(item);
        }
        Ok(first)
    }

    /// The stream's first item (`None`: its end), pulled ahead of its reader
    /// and held for the next [`Self::next`] as it was, unbooked: a stream
    /// tracked later books it when its reader meets it. Only for a stream
    /// nothing tracks yet (the parallel decode side while its prefill side
    /// is awaited). Cancel-safe: dropped before the item comes, it leaves
    /// the stream as it was.
    pub(crate) async fn first_item_ahead(
        &mut self,
    ) -> Option<&Result<ProtoGenerateResponse, tonic::Status>> {
        if !self.holds_item_ahead() {
            let first = self.next_untracked().await;
            let inner = std::mem::replace(self, Self::Fanout(FanoutStream::new(Vec::new())));
            *self = Self::Ahead(Box::new(AheadStream { inner, first }));
        }
        let Self::Ahead(ahead) = self else {
            return None;
        };
        ahead.first.as_ref()
    }

    /// Whether the stream holds an item pulled ahead of its reader (see
    /// [`Self::first_item_ahead`]).
    pub(crate) fn holds_item_ahead(&self) -> bool {
        matches!(self, Self::Ahead(_))
    }

    /// The worker this stream was dispatched to, when it is tracked.
    pub fn worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Tracked(tracked) => Some(&tracked.worker),
            Self::DecodeSide(side) => side.inner.worker(),
            Self::Ahead(ahead) => ahead.inner.worker(),
            Self::Watched(watched) => watched.inner.as_ref().and_then(ProtoStream::worker),
            _ => None,
        }
    }

    /// This stream read for the client at the receiving end of `client`'s
    /// channel (see [`WatchedStream`]): the moment the client's response body
    /// is dropped, the stream is dropped (its abort reaches the engine) and
    /// its reader meets the end. `request_id` names the request for the log
    /// line.
    #[must_use]
    pub fn until_client_gone(self, client: &SseSender, request_id: &str) -> Self {
        Self::Watched(Box::new(WatchedStream {
            inner: Some(self),
            client: client.clone(),
            request_id: request_id.to_string(),
        }))
    }

    /// This stream as a disaggregated request's decode side: its responses
    /// carry `prefill`'s prefix-cache hit instead of their own (see
    /// [`PrefillCachedTokens`]). A prefill side that reported nothing leaves
    /// the stream as it is.
    #[must_use]
    pub fn decode_side(self, prefill: PrefillCachedTokens) -> Self {
        if prefill.is_empty() {
            return self;
        }
        Self::DecodeSide(Box::new(DecodeStream {
            inner: self,
            prefill,
        }))
    }

    /// Whether the end this stream's reader met was the client's leaving
    /// (see [`Self::until_client_gone`]), not the engine's end: the
    /// generation did not complete, and nothing accounts it as one. Read
    /// through a decode-side wrapper as through the watch itself (dual
    /// dispatch wraps the watched stream once the prefill side has drained);
    /// `false` for a stream that is not watched.
    pub fn client_left(&self) -> bool {
        match self {
            Self::Watched(watched) => watched.inner.is_none(),
            Self::DecodeSide(side) => side.inner.client_left(),
            _ => false,
        }
    }

    /// Get next item from stream
    pub async fn next(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
        let Self::Watched(watched) = self else {
            return self.next_unwatched().await;
        };
        let WatchedStream { inner, client, .. } = &mut **watched;
        // Nothing after the client left: the inner stream went with it.
        let stream = inner.as_mut()?;
        // The client's leaving is looked at first, so a stream the engine
        // keeps busy cannot outrun it; the inner stream is polled in place,
        // without a boxed future per message. `None` here is the client's
        // leaving, `Some(item)` the stream's own item or end.
        let item = tokio::select! {
            biased;
            () = client.closed() => None,
            item = stream.next_unwatched() => Some(item),
        };
        match item {
            Some(item) => item,
            None => {
                watched.client_left();
                None
            }
        }
    }

    /// The next item of a stream that is not [`Self::Watched`]; a watched
    /// stream goes through [`Self::next`], which races the client's leaving.
    async fn next_unwatched(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
        if let Self::Watched(_) = self {
            // Reached only with one watched stream inside another.
            return Box::pin(self.next()).await;
        }
        if let Self::DecodeSide(side) = self {
            // A decode side's messages are polled in place as a tracked
            // stream's are, without a boxed future per message; a wrapper
            // inside the wrapper recurses. The inner stream did its own
            // bookkeeping and engine-error mapping; only the prompt-side usage
            // changes here.
            let item = if matches!(side.inner, Self::Tracked(_) | Self::DecodeSide(_)) {
                Box::pin(side.inner.next()).await
            } else {
                side.inner.next_untracked().await
            };
            return item.map(|result| {
                result.map(|mut response| {
                    response.restamp_cached_tokens(&side.prefill);
                    response
                })
            });
        }
        let Self::Tracked(tracked) = self else {
            return self.next_untracked().await;
        };
        if let Some(item) = tracked.replay.take() {
            return item;
        }
        // Every dispatched stream is one tracked layer over a transport
        // stream: its messages are polled in place, without a boxed future
        // per message. A tracked stream inside a tracked stream recurses.
        let item = if matches!(tracked.inner, Self::Tracked(_)) {
            Box::pin(tracked.inner.next()).await
        } else {
            tracked.inner.next_untracked().await
        };
        match &item {
            Some(Ok(response)) => {
                liveness::on_token_progress(&tracked.worker);
                if std::mem::take(&mut tracked.samples_success) {
                    tracked.worker.record_outcome(200);
                }
                if tracked.output.is_none() {
                    tracked.output = Some(OutputTicket::open(&tracked.worker));
                }
                if let Some(ticket) = tracked.prefill.as_mut() {
                    ticket.first_response(u64::from(response.cached_tokens()));
                }
                tracked.prefill = None;
            }
            // The stream ended, or failed: what the worker was producing for
            // it is over, whenever its reader lets go of it.
            Some(Err(_)) | None => tracked.output = None,
        }
        item
    }

    /// The next item of a stream that is not [`Self::Tracked`]; a tracked
    /// stream goes through [`Self::next`], which does its bookkeeping.
    async fn next_untracked(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
        let item = match self {
            Self::Sglang(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Sglang(Box::new(r)))),
            Self::Vllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Vllm(Box::new(r)))),
            Self::Trtllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Trtllm(Box::new(r)))),
            Self::Mlx(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Mlx(Box::new(r)))),
            Self::TokenSpeed(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::TokenSpeed(Box::new(r)))),
            // Every ZMQ engine (including TokenSpeed) emits vllm-shaped
            // responses: the adapter translates wire output into
            // `vllm::GenerateResponse`, so variant checks like
            // `is_tokenspeed()` on a response are unreliable for ZMQ-backed
            // streams. Key response-side engine logic on the worker's
            // `runtime_type()`, never on the response variant; key
            // accumulation on `chunk_semantics()`, which the vLLM shape
            // (delta chunks, cumulative `Complete`) defines for this lane.
            Self::Zmq(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Vllm(Box::new(r)))),
            Self::Fanout(stream) => stream.next().await,
            Self::Tracked(_) | Self::Watched(_) => return Box::pin(self.next()).await,
            // Reached only with one wrapper inside another: [`Self::next`]
            // polls a decode side in place.
            Self::DecodeSide(_) => return Box::pin(self.next()).await,
            // The held item, mapped when it was read, and the layer goes.
            Self::Ahead(ahead) => {
                let first = ahead.first.take();
                let empty = Self::Fanout(FanoutStream::new(Vec::new()));
                *self = std::mem::replace(&mut ahead.inner, empty);
                return first;
            }
        };
        item.map(reject_engine_error)
    }

    /// Mark stream as completed (no abort needed)
    pub fn mark_completed(&mut self) {
        match self {
            Self::Sglang(stream) => stream.mark_completed(),
            Self::Vllm(stream) => stream.mark_completed(),
            Self::Trtllm(stream) => stream.mark_completed(),
            Self::Mlx(stream) => stream.mark_completed(),
            Self::TokenSpeed(stream) => stream.mark_completed(),
            Self::Zmq(stream) => stream.mark_completed(),
            Self::Fanout(stream) => stream.mark_completed(),
            Self::Tracked(stream) => stream.inner.mark_completed(),
            Self::DecodeSide(side) => side.inner.mark_completed(),
            Self::Ahead(ahead) => ahead.inner.mark_completed(),
            Self::Watched(watched) => {
                if let Some(inner) = watched.inner.as_mut() {
                    inner.mark_completed();
                }
            }
        }
    }

    /// Bound the stream by the request's `deadline` (`--request-timeout-secs`,
    /// counted from the dispatch): polled at or past it, the transport stream
    /// ends with the deadline's status and is released as a dropped one is,
    /// so its abort reaches the engine at once (see
    /// [`smg_grpc_client::AbortOnDropStream::with_deadline`]). Every wrapper
    /// hands the deadline down to the transport stream(s) under it; a fan-out
    /// bounds each child. The ZMQ wire's stream has no deadline hook and is
    /// left as it is.
    #[must_use]
    pub fn with_deadline(self, deadline: RequestDeadline) -> Self {
        match self {
            Self::Sglang(stream) => Self::Sglang(stream.with_deadline(deadline)),
            Self::Vllm(stream) => Self::Vllm(stream.with_deadline(deadline)),
            Self::Trtllm(stream) => Self::Trtllm(stream.with_deadline(deadline)),
            Self::Mlx(stream) => Self::Mlx(stream.with_deadline(deadline)),
            Self::TokenSpeed(stream) => Self::TokenSpeed(stream.with_deadline(deadline)),
            Self::Zmq(stream) => Self::Zmq(stream),
            Self::Fanout(stream) => Self::Fanout(stream.with_deadline(deadline)),
            Self::Tracked(stream) => {
                let TrackedStream {
                    inner,
                    worker,
                    prefill,
                    progress,
                    output,
                    samples_success,
                    replay,
                } = *stream;
                Self::Tracked(Box::new(TrackedStream {
                    inner: inner.with_deadline(deadline),
                    worker,
                    prefill,
                    progress,
                    output,
                    samples_success,
                    replay,
                }))
            }
            Self::DecodeSide(side) => {
                let DecodeStream { inner, prefill } = *side;
                Self::DecodeSide(Box::new(DecodeStream {
                    inner: inner.with_deadline(deadline),
                    prefill,
                }))
            }
            // The item pulled ahead is already in hand; the stream under it
            // is bounded.
            Self::Ahead(ahead) => {
                let AheadStream { inner, first } = *ahead;
                Self::Ahead(Box::new(AheadStream {
                    inner: inner.with_deadline(deadline),
                    first,
                }))
            }
            // The watched stream's inner stream is bounded while the client is
            // still there; once it left, there is nothing left to bound.
            Self::Watched(watched) => {
                let WatchedStream {
                    inner,
                    client,
                    request_id,
                } = *watched;
                Self::Watched(Box::new(WatchedStream {
                    inner: inner.map(|inner| inner.with_deadline(deadline)),
                    client,
                    request_id,
                }))
            }
        }
    }

    /// Defer the abort-on-drop until the stream yields its first response.
    ///
    /// Used for the disaggregated decode side: aborting a decode request while
    /// it is still receiving the KV handoff from its prefill peer can tear
    /// down the transfer mid-write; the first response proves the handoff
    /// completed, after which the (deferred) abort is safe. No-op for ZMQ
    /// streams — the ZMQ adapter owns its own lifecycle and the ZMQ lane does
    /// not serve disaggregated requests.
    #[must_use]
    pub fn defer_abort_until_first_item(self) -> Self {
        match self {
            Self::Sglang(stream) => Self::Sglang(stream.defer_abort_until_first_item()),
            Self::Vllm(stream) => Self::Vllm(stream.defer_abort_until_first_item()),
            Self::Trtllm(stream) => Self::Trtllm(stream.defer_abort_until_first_item()),
            Self::Mlx(stream) => Self::Mlx(stream.defer_abort_until_first_item()),
            Self::TokenSpeed(stream) => Self::TokenSpeed(stream.defer_abort_until_first_item()),
            Self::Zmq(stream) => Self::Zmq(stream),
            Self::Fanout(stream) => Self::Fanout(stream.defer_abort_until_first_item()),
            Self::Tracked(stream) => {
                let TrackedStream {
                    inner,
                    worker,
                    prefill,
                    progress,
                    output,
                    samples_success,
                    replay,
                } = *stream;
                Self::Tracked(Box::new(TrackedStream {
                    inner: inner.defer_abort_until_first_item(),
                    worker,
                    prefill,
                    progress,
                    output,
                    samples_success,
                    replay,
                }))
            }
            Self::DecodeSide(side) => {
                let DecodeStream { inner, prefill } = *side;
                Self::DecodeSide(Box::new(DecodeStream {
                    inner: inner.defer_abort_until_first_item(),
                    prefill,
                }))
            }
            Self::Ahead(ahead) => {
                let AheadStream { inner, first } = *ahead;
                Self::Ahead(Box::new(AheadStream {
                    inner: inner.defer_abort_until_first_item(),
                    first,
                }))
            }
            Self::Watched(watched) => {
                let WatchedStream {
                    inner,
                    client,
                    request_id,
                } = *watched;
                Self::Watched(Box::new(WatchedStream {
                    inner: inner.map(ProtoStream::defer_abort_until_first_item),
                    client,
                    request_id,
                }))
            }
        }
    }
}

impl FanoutChild for ProtoStream {
    fn next_item(
        &mut self,
    ) -> impl Future<Output = Option<Result<ProtoGenerateResponse, tonic::Status>>> + Send {
        self.next()
    }

    fn mark_completed(&mut self) {
        Self::mark_completed(self);
    }

    fn defer_abort_until_first_item(self) -> Self {
        Self::defer_abort_until_first_item(self)
    }

    fn with_deadline(self, deadline: RequestDeadline) -> Self {
        Self::with_deadline(self, deadline)
    }

    fn worker(&self) -> Option<&Arc<dyn Worker>> {
        ProtoStream::worker(self)
    }
}

/// What a fan-out needs from a child stream. Implemented by [`ProtoStream`];
/// tests implement it on a scripted child.
pub trait FanoutChild: Send {
    /// The child's next response, or `None` once it has ended.
    fn next_item(
        &mut self,
    ) -> impl Future<Output = Option<Result<ProtoGenerateResponse, tonic::Status>>> + Send;

    /// Mark the child completed so dropping it sends no abort.
    fn mark_completed(&mut self);

    /// Defer the child's abort-on-drop until its first response.
    #[must_use]
    fn defer_abort_until_first_item(self) -> Self;

    /// Bound the child by the request's `deadline` (see
    /// [`ProtoStream::with_deadline`]); a child with no transport stream
    /// under it is left as it is.
    #[must_use]
    fn with_deadline(self, _deadline: RequestDeadline) -> Self
    where
        Self: Sized,
    {
        self
    }

    /// The one worker the child was dispatched to, when it is tracked; a
    /// fan-out over several workers answers `None`.
    fn worker(&self) -> Option<&Arc<dyn Worker>> {
        None
    }
}

/// One child per sample of an n>1 fan-out over rendezvous-room PD pairs.
///
/// Each child is a single-sample dispatch whose responses report choice
/// index 0; the fan-out restamps them with the child's position, so the
/// pipeline demuxes the merged stream exactly like an engine-native n>1
/// stream. Children are polled together, starting from a rotating cursor so
/// a chatty child cannot starve its siblings, and are retired as they end;
/// the fan-out ends with the last child. Dropping it drops every child, so
/// each side keeps its abort-on-drop, and `mark_completed` reaches every
/// child.
pub struct FanoutStream<C = ProtoStream> {
    children: Vec<Option<C>>,
    cursor: usize,
}

#[cfg(test)]
impl<C: FanoutChild> FanoutChild for FanoutStream<C> {
    fn next_item(
        &mut self,
    ) -> impl Future<Output = Option<Result<ProtoGenerateResponse, tonic::Status>>> + Send {
        self.next()
    }

    fn mark_completed(&mut self) {
        Self::mark_completed(self);
    }

    fn defer_abort_until_first_item(self) -> Self {
        Self::defer_abort_until_first_item(self)
    }
}

type FanoutItem = Option<Result<ProtoGenerateResponse, tonic::Status>>;
/// One child's pending `next_item`, tagged with the child's position.
type FanoutPoll<'a> = Pin<Box<dyn Future<Output = (usize, FanoutItem)> + Send + 'a>>;

impl<C: FanoutChild> FanoutStream<C> {
    pub fn new(children: Vec<C>) -> Self {
        Self {
            children: children.into_iter().map(Some).collect(),
            cursor: 0,
        }
    }

    /// Children that have not ended yet.
    pub fn open_children(&self) -> usize {
        self.children.iter().flatten().count()
    }

    /// Next response from any open child, stamped with that child's index.
    pub async fn next(&mut self) -> FanoutItem {
        loop {
            let len = self.children.len();
            if len == 0 {
                return None;
            }
            let start = self.cursor % len;
            self.cursor = (start + 1) % len;
            // Open children before the cursor go to the back of the poll
            // order, so the cursor rotates which child gets the first look.
            let skip = self.children[..start].iter().flatten().count();
            let mut polls: Vec<FanoutPoll<'_>> = self
                .children
                .iter_mut()
                .enumerate()
                .filter_map(|(position, child)| {
                    child.as_mut().map(|child| {
                        let poll: FanoutPoll<'_> =
                            Box::pin(async move { (position, child.next_item().await) });
                        poll
                    })
                })
                .collect();
            if polls.is_empty() {
                return None;
            }
            let rotate = skip % polls.len();
            polls.rotate_left(rotate);
            let ((position, item), _, _) = select_all(polls).await;
            match item {
                Some(Ok(mut response)) => {
                    response.set_index(u32::try_from(position).unwrap_or(u32::MAX));
                    return Some(Ok(response));
                }
                Some(Err(status)) => return Some(Err(status)),
                None => self.children[position] = None,
            }
        }
    }

    /// Mark every open child completed.
    pub fn mark_completed(&mut self) {
        for child in self.children.iter_mut().flatten() {
            child.mark_completed();
        }
    }

    /// Defer every open child's abort-on-drop until its first response.
    #[must_use]
    pub fn defer_abort_until_first_item(self) -> Self {
        Self {
            children: self
                .children
                .into_iter()
                .map(|child| child.map(FanoutChild::defer_abort_until_first_item))
                .collect(),
            cursor: self.cursor,
        }
    }

    /// Bound every open child by the request's `deadline`.
    #[must_use]
    pub fn with_deadline(self, deadline: RequestDeadline) -> Self {
        Self {
            children: self
                .children
                .into_iter()
                .map(|child| child.map(|child| child.with_deadline(deadline)))
                .collect(),
            cursor: self.cursor,
        }
    }
}

#[cfg(test)]
mod fanout_tests {
    use std::{
        collections::VecDeque,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
    };

    use super::*;
    use crate::{
        routers::grpc::common::response_collection::drain_prefill,
        worker::{
            BasicWorkerBuilder, PrefillAdmission, PrefillLoadGuard, Worker, WorkerLoadGuard,
            WorkerType,
        },
    };

    /// A child that yields a script, then ends.
    struct Scripted {
        items: VecDeque<Result<ProtoGenerateResponse, tonic::Status>>,
        completed: Arc<AtomicBool>,
    }

    impl FanoutChild for Scripted {
        fn next_item(
            &mut self,
        ) -> impl Future<Output = Option<Result<ProtoGenerateResponse, tonic::Status>>> + Send
        {
            let item = self.items.pop_front();
            async move { item }
        }

        fn mark_completed(&mut self) {
            self.completed.store(true, Ordering::SeqCst);
        }

        fn defer_abort_until_first_item(self) -> Self {
            self
        }
    }

    fn chunk(text: &str) -> ProtoGenerateResponse {
        ProtoGenerateResponse::Sglang(Box::new(sglang::GenerateResponse {
            response: Some(sglang::generate_response::Response::Chunk(
                sglang::GenerateStreamChunk {
                    token_ids: vec![text.len() as u32],
                    ..Default::default()
                },
            )),
            ..Default::default()
        }))
    }

    fn complete() -> ProtoGenerateResponse {
        ProtoGenerateResponse::Sglang(Box::new(sglang::GenerateResponse {
            response: Some(sglang::generate_response::Response::Complete(
                sglang::GenerateComplete::default(),
            )),
            ..Default::default()
        }))
    }

    fn child(items: Vec<ProtoGenerateResponse>) -> (Scripted, Arc<AtomicBool>) {
        let completed = Arc::new(AtomicBool::new(false));
        let child = Scripted {
            items: items.into_iter().map(Ok).collect(),
            completed: Arc::clone(&completed),
        };
        (child, completed)
    }

    fn index_of(response: ProtoGenerateResponse) -> u32 {
        match response.into_response() {
            ProtoResponseVariant::Chunk(chunk) => chunk.index(),
            ProtoResponseVariant::Complete(complete) => complete.index(),
            ProtoResponseVariant::None => u32::MAX,
        }
    }

    #[tokio::test]
    async fn fanout_stamps_each_childs_position_and_ends_with_the_last_child() {
        let (first, first_done) = child(vec![chunk("a"), chunk("aa"), complete()]);
        let (second, second_done) = child(vec![chunk("b"), complete()]);
        let mut fanout = FanoutStream::new(vec![first, second]);

        let mut indices = Vec::new();
        while let Some(item) = fanout.next().await {
            indices.push(index_of(item.unwrap()));
        }
        assert_eq!(indices.len(), 5);
        assert_eq!(indices.iter().filter(|&&i| i == 0).count(), 3);
        assert_eq!(indices.iter().filter(|&&i| i == 1).count(), 2);
        // The rotating cursor lets the second child in before the first drains.
        assert_ne!(indices[0], indices[1]);
        assert_eq!(fanout.open_children(), 0);
        assert!(fanout.next().await.is_none());

        assert!(!first_done.load(Ordering::SeqCst));
        fanout.mark_completed();
        // Ended children were retired; only open children get marked.
        assert!(!first_done.load(Ordering::SeqCst));
        assert!(!second_done.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn fanout_mark_completed_reaches_every_open_child() {
        let (first, first_done) = child(vec![chunk("a")]);
        let (second, second_done) = child(vec![chunk("b")]);
        let mut fanout = FanoutStream::new(vec![first, second]);
        assert_eq!(fanout.open_children(), 2);
        fanout.mark_completed();
        assert!(first_done.load(Ordering::SeqCst));
        assert!(second_done.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn fanout_passes_a_child_error_through() {
        let (first, _) = child(vec![chunk("a")]);
        let second = Scripted {
            items: VecDeque::from([Err(tonic::Status::internal("side died"))]),
            completed: Arc::new(AtomicBool::new(false)),
        };
        let mut fanout = FanoutStream::new(vec![second, first]);
        let first_item = fanout.next().await.unwrap();
        assert!(matches!(first_item, Err(status) if status.message() == "side died"));
    }

    #[tokio::test]
    async fn prefill_fanout_retains_unfinished_sample_guards() {
        for finish_at_complete in [false, true] {
            for bounded in [false, true] {
                let worker: Arc<dyn Worker> = Arc::new(
                    BasicWorkerBuilder::new("http://prefill-fanout")
                        .worker_type(WorkerType::Prefill)
                        .build(),
                );
                let admission = PrefillAdmission::new(1, 0, std::time::Duration::from_secs(1));
                let guard = if bounded {
                    let admitted = admission
                        .admit(None, |capacity| capacity.select(Arc::clone(&worker), ()))
                        .await
                        .unwrap();
                    PrefillLoadGuard::Admission {
                        _reservation: Arc::new(admitted.reservation),
                    }
                } else {
                    PrefillLoadGuard::Unbounded {
                        _guard: WorkerLoadGuard::new(Arc::clone(&worker), None),
                    }
                };
                let guards = guard.replicate_to(2).into_iter().map(Some).collect();
                assert_eq!(worker.load(), if bounded { 1 } else { 2 });
                // Each n=1 Prefill dispatch returns one terminal response.
                // The second dispatch is unfinished when the first completes.
                let (first, _) = child(vec![complete()]);
                let (second, _) = child(vec![complete()]);
                let mut stream = FanoutStream::new(vec![first, second]);
                let mut indices = Vec::new();
                drain_prefill(&mut stream, guards, finish_at_complete, |complete| {
                    indices.push(complete.index());
                    assert_eq!(worker.load(), if complete.index() == 0 { 1 } else { 0 });
                })
                .await
                .unwrap();
                assert_eq!(indices, vec![0, 1]);
                assert_eq!(worker.load(), 0);
            }
        }
    }

    #[test]
    fn set_index_restamps_chunks_and_completes() {
        let mut response = chunk("x");
        response.set_index(3);
        assert_eq!(index_of(response), 3);
        let mut response = complete();
        response.set_index(2);
        assert_eq!(index_of(response), 2);
    }

    #[test]
    fn sampling_n_reads_and_writes_the_room_based_engines() {
        let mut sglang_req = ProtoGenerateRequest::Sglang(Box::new(sglang::GenerateRequest {
            sampling_params: Some(sglang::SamplingParams {
                n: 4,
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert_eq!(sglang_req.sampling_n(), 4);
        sglang_req.set_sampling_n(1);
        assert_eq!(sglang_req.sampling_n(), 1);
        assert!(!sglang_req.has_mm_inputs());

        let mut tokenspeed_req =
            ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed::GenerateRequest {
                sampling_params: Some(tokenspeed::SamplingParams {
                    n: 0,
                    sampling_seed: Some(5),
                    ..Default::default()
                }),
                mm_inputs: Some(tokenspeed::MultimodalInputs::default()),
                ..Default::default()
            }));
        assert_eq!(tokenspeed_req.sampling_n(), 1, "0 means unset");
        assert!(tokenspeed_req.has_mm_inputs());
        tokenspeed_req.offset_sampling_seed(2);
        let ProtoGenerateRequest::TokenSpeed(req) = &tokenspeed_req else {
            panic!("runtime changed");
        };
        assert_eq!(req.sampling_params.as_ref().unwrap().sampling_seed, Some(7));
    }
}

#[cfg(test)]
mod engine_error_tests {
    use super::*;

    /// One `Complete` per backend finishing with `reason`.
    fn completes(reason: &str) -> Vec<ProtoGenerateResponse> {
        let reason = reason.to_string();
        vec![
            ProtoGenerateResponse::Sglang(Box::new(sglang::GenerateResponse {
                response: Some(sglang::generate_response::Response::Complete(
                    sglang::GenerateComplete {
                        finish_reason: reason.clone(),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            })),
            ProtoGenerateResponse::Vllm(Box::new(vllm::GenerateResponse {
                response: Some(vllm::generate_response::Response::Complete(
                    vllm::GenerateComplete {
                        finish_reason: reason.clone(),
                        ..Default::default()
                    },
                )),
            })),
            ProtoGenerateResponse::Trtllm(Box::new(trtllm::GenerateResponse {
                response: Some(trtllm::generate_response::Response::Complete(
                    trtllm::GenerateComplete {
                        finish_reason: reason.clone(),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            })),
            ProtoGenerateResponse::Mlx(Box::new(mlx::GenerateResponse {
                response: Some(mlx::generate_response::Response::Complete(
                    mlx::GenerateComplete {
                        finish_reason: reason.clone(),
                        ..Default::default()
                    },
                )),
            })),
            ProtoGenerateResponse::TokenSpeed(Box::new(tokenspeed::GenerateResponse {
                response: Some(tokenspeed::generate_response::Response::Complete(
                    tokenspeed::GenerateComplete {
                        finish_reason: reason,
                        ..Default::default()
                    },
                )),
                ..Default::default()
            })),
        ]
    }

    #[test]
    fn engine_error_is_true_only_for_complete_with_error_finish_reason() {
        assert!(completes("error")
            .iter()
            .all(ProtoGenerateResponse::engine_error));
        for reason in ["stop", "length", "abort", ""] {
            assert!(!completes(reason)
                .iter()
                .any(ProtoGenerateResponse::engine_error));
        }
        let chunk = ProtoGenerateResponse::Vllm(Box::new(vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Chunk(
                vllm::GenerateStreamChunk::default(),
            )),
        }));
        assert!(!chunk.engine_error());
        let empty = ProtoGenerateResponse::Vllm(Box::default());
        assert!(!empty.engine_error());
    }

    #[test]
    fn reject_engine_error_turns_error_complete_into_internal_status() {
        for response in completes("error") {
            let Err(status) = reject_engine_error(Ok(response)) else {
                panic!("engine error must not pass through as a completion");
            };
            assert_eq!(status.code(), tonic::Code::Internal);
            assert_eq!(
                status.message(),
                "engine finished the request with an error (see engine logs)"
            );
        }
        for response in completes("stop") {
            assert!(reject_engine_error(Ok(response)).is_ok());
        }
        let passed = reject_engine_error(Err(tonic::Status::unavailable("side died")));
        assert!(matches!(passed, Err(status) if status.code() == tonic::Code::Unavailable));
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use smg_grpc_client::common_proto as common;

    #[test]
    fn inline_bytes_counts_every_inline_tensor_and_raw_image() {
        let trtllm = MultimodalData::Trtllm(TrtllmMultimodalData {
            image_data: vec![vec![0; 3], vec![0; 5]],
        });
        assert_eq!(trtllm.inline_bytes(), 8);

        let sglang = MultimodalData::Sglang(SglangMultimodalData {
            image_data: vec![vec![0; 2]],
            pixel_values: vec![0; 16],
            pixel_values_shape: vec![4, 4],
            pixel_values_dtype: "float32".to_string(),
            model_specific_tensors: HashMap::from([(
                "grid".to_string(),
                TensorBytes {
                    data: vec![0; 24],
                    shape: vec![1, 3],
                    dtype: "int64".to_string(),
                },
            )]),
            im_token_id: None,
            mm_placeholders: vec![(0, 4)],
        });
        assert_eq!(sglang.inline_bytes(), 42);
    }

    #[test]
    fn inline_bytes_counts_every_batch_of_a_mixed_request() {
        let alone = MultimodalData::Vllm(vllm_mm_data(common::Modality::Image));
        assert_eq!(alone.inline_bytes(), 16);

        let mut video = vllm_mm_data(common::Modality::Video);
        video.model_specific_tensors.insert(
            "video_grid_thw".to_string(),
            TensorBytes {
                data: vec![0; 24],
                shape: vec![1, 3],
                dtype: "int64".to_string(),
            },
        );
        let mut image = vllm_mm_data(common::Modality::Image);
        image.extra_batches = vec![video];
        assert_eq!(MultimodalData::Vllm(image).inline_bytes(), 56);
    }

    use super::{multimodal::tensor_bytes_to_tokenspeed, *};

    #[test]
    fn tokenspeed_image_into_proto_uses_itemized_payload() {
        let mut model_specific_tensors = HashMap::new();
        model_specific_tensors.insert(
            "image_grid_thw".to_string(),
            TensorBytes {
                data: vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
                shape: vec![1, 3],
                dtype: "uint32".to_string(),
            },
        );

        let proto = TokenSpeedMultimodalData {
            items: vec![TokenSpeedMultimodalItem {
                modality: TokenSpeedModality::Image,
                encoder_input: TokenSpeedTensor::inline(
                    vec![42; 8],
                    vec![1, 2],
                    "float32".to_string(),
                ),
                model_specific_tensors,
                placeholder_token_id: Some(151655),
                mm_placeholders: vec![(4, 2)],
                content_hash: vec![7; 32],
            }],
            shm_enabled: false,
            shm_min_bytes: 0,
        }
        .into_proto(false);

        assert_eq!(proto.items.len(), 1);
        let item = &proto.items[0];
        assert_eq!(item.modality, common::Modality::Image as i32);
        assert_eq!(item.placeholder_token_id, Some(151655));
        assert_eq!(item.placeholders[0].offset, 4);
        assert_eq!(item.placeholders[0].length, 2);
        assert_eq!(
            inline_tensor_data(item.encoder_input.as_ref().unwrap()),
            &[42; 8]
        );
        assert!(item.model_specific_tensors.contains_key("image_grid_thw"));
    }

    #[test]
    fn tokenspeed_video_into_proto_uses_itemized_payload() {
        let mut model_specific_tensors = HashMap::new();
        model_specific_tensors.insert(
            "video_grid_thw".to_string(),
            TensorBytes {
                data: vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
                shape: vec![1, 3],
                dtype: "uint32".to_string(),
            },
        );

        let proto = TokenSpeedMultimodalData {
            items: vec![TokenSpeedMultimodalItem {
                modality: TokenSpeedModality::Video,
                encoder_input: TokenSpeedTensor::inline(
                    vec![42; 8],
                    vec![1, 2],
                    "float32".to_string(),
                ),
                model_specific_tensors,
                placeholder_token_id: Some(151656),
                mm_placeholders: vec![(4, 2)],
                content_hash: vec![7; 32],
            }],
            shm_enabled: false,
            shm_min_bytes: 0,
        }
        .into_proto(false);

        assert_eq!(proto.items.len(), 1);
        let item = &proto.items[0];
        assert_eq!(item.modality, common::Modality::Video as i32);
        assert_eq!(item.placeholder_token_id, Some(151656));
        assert_eq!(item.placeholders[0].offset, 4);
        assert_eq!(item.placeholders[0].length, 2);
        assert_eq!(
            inline_tensor_data(item.encoder_input.as_ref().unwrap()),
            &[42; 8]
        );
        assert!(item.model_specific_tensors.contains_key("video_grid_thw"));
    }

    #[test]
    fn tokenspeed_tensor_data_uses_clean_payload_tags() {
        let tensor = tensor_bytes_to_tokenspeed(
            TensorBytes {
                data: vec![0xaa, 0xbb],
                shape: vec![2, 3],
                dtype: "uint32".to_string(),
            },
            false,
            0,
        );

        assert_eq!(
            tensor.encode_to_vec(),
            vec![
                0x0a, 0x02, 0x02, 0x03, // shape = 1, packed uint32 [2, 3]
                0x12, 0x06, b'u', b'i', b'n', b't', b'3', b'2', // dtype = 2
                0x1a, 0x02, 0xaa, 0xbb, // inline = 3
            ]
        );
    }

    #[test]
    fn tokenspeed_shm_encoder_input_into_proto_uses_shm_payload() {
        let proto = TokenSpeedMultimodalData {
            items: vec![TokenSpeedMultimodalItem {
                modality: TokenSpeedModality::Image,
                encoder_input: TokenSpeedTensor::shm(
                    common::ShmHandle {
                        name: "smg-test-shm".to_string(),
                        offset: 0,
                        nbytes: 8,
                        owner_id: "smg:test".to_string(),
                    },
                    vec![1, 2],
                    "bfloat16".to_string(),
                ),
                model_specific_tensors: HashMap::new(),
                placeholder_token_id: Some(151655),
                mm_placeholders: vec![(4, 2)],
                content_hash: vec![7; 32],
            }],
            shm_enabled: true,
            shm_min_bytes: 0,
        }
        .into_proto(false);

        let tensor = proto.items[0].encoder_input.as_ref().unwrap();
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(tensor.dtype, "bfloat16");
        match tensor.payload.as_ref() {
            Some(tokenspeed::tensor_data::Payload::Shm(handle)) => {
                assert_eq!(handle.name, "smg-test-shm");
                assert_eq!(handle.nbytes, 8);
            }
            _ => panic!("expected shm TensorData payload"),
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn tokenspeed_shm_writer_exposes_complete_mapped_payload() {
        use super::shm::tokenspeed_shm_path;

        let expected = [0x12, 0x34, 0x56, 0x78];
        let handle = write_tokenspeed_shm_with(expected.len(), |output| {
            output.copy_from_slice(&expected);
            Ok(())
        })
        .unwrap();
        let payload = std::fs::read(tokenspeed_shm_path(&handle.name));
        cleanup_mm_shm_handles(std::slice::from_ref(&handle));

        assert_eq!(payload.unwrap(), expected);
        assert!(!tokenspeed_shm_path(&handle.name).exists());
    }

    #[test]
    fn tokenspeed_remote_encoder_input_into_proto_uses_remote_payload() {
        let proto = TokenSpeedMultimodalData {
            items: vec![TokenSpeedMultimodalItem {
                modality: TokenSpeedModality::Image,
                encoder_input: TokenSpeedTensor::remote(
                    common::RemoteTensorHandle {
                        transport: "nixl".to_string(),
                        descriptor: vec![1, 2, 3],
                        nbytes: 8,
                    },
                    vec![1, 2],
                    "bfloat16".to_string(),
                ),
                model_specific_tensors: HashMap::new(),
                placeholder_token_id: Some(151655),
                mm_placeholders: vec![(4, 2)],
                content_hash: vec![7; 32],
            }],
            shm_enabled: true,
            shm_min_bytes: 0,
        }
        .into_proto(false);

        let tensor = proto.items[0].encoder_input.as_ref().unwrap();
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(tensor.dtype, "bfloat16");
        match tensor.payload.as_ref() {
            Some(tokenspeed::tensor_data::Payload::Remote(handle)) => {
                assert_eq!(handle.transport, "nixl");
                assert_eq!(handle.descriptor, vec![1, 2, 3]);
                assert_eq!(handle.nbytes, 8);
            }
            _ => panic!("expected remote TensorData payload"),
        }
    }

    fn inline_tensor_data(tensor: &tokenspeed::TensorData) -> &[u8] {
        match tensor.payload.as_ref() {
            Some(tokenspeed::tensor_data::Payload::Inline(data)) => data.as_ref(),
            _ => panic!("expected inline TensorData payload"),
        }
    }

    #[test]
    fn set_data_parallel_rank_per_engine() {
        let mut vllm_req = ProtoGenerateRequest::Vllm(Box::default());
        vllm_req.set_data_parallel_rank(2);
        assert!(matches!(
            &vllm_req,
            ProtoGenerateRequest::Vllm(req) if req.data_parallel_rank == Some(2)
        ));

        let mut sglang_req = ProtoGenerateRequest::Sglang(Box::default());
        sglang_req.set_data_parallel_rank(3);
        assert!(matches!(
            &sglang_req,
            ProtoGenerateRequest::Sglang(req) if req.data_parallel_rank == Some(3)
        ));

        let mut ts_req = ProtoGenerateRequest::TokenSpeed(Box::default());
        ts_req.set_data_parallel_rank(5);
        assert!(matches!(
            &ts_req,
            ProtoGenerateRequest::TokenSpeed(req) if req.data_parallel_rank == Some(5)
        ));

        // Engines without the proto field ignore the pin
        let mut mlx_req = ProtoGenerateRequest::Mlx(Box::default());
        mlx_req.set_data_parallel_rank(1);
    }

    fn vllm_mm_data(modality: common::Modality) -> VllmMultimodalData {
        let is_video = modality == common::Modality::Video;
        VllmMultimodalData {
            pixel_values: vec![0u8; 16],
            pixel_values_shape: vec![1, 4],
            pixel_values_dtype: "float32".to_string(),
            model_specific_tensors: HashMap::new(),
            im_token_id: Some(if is_video { 151656 } else { 151655 }),
            mm_placeholders: vec![(3, 4)],
            mm_hashes: vec!["h0".to_string()],
            batched_keys: vec![],
            flat_keys: HashMap::new(),
            keep_on_cpu_keys: vec![],
            encoder_input_key: None,
            modality,
            shm_enabled: false,
            shm_min_bytes: 0,
            rdma_enabled: false,
            extra_batches: Vec::new(),
        }
    }

    #[test]
    fn vllm_modality_round_trips_into_proto() {
        // The video path must set the proto `modality` so the servicer routes to
        // vLLM's video modality; the image path must set image.
        let video = vllm_mm_data(common::Modality::Video).into_proto();
        assert_eq!(video.modality, common::Modality::Video as i32);
        assert_eq!(video.im_token_id, Some(151656));

        let image = vllm_mm_data(common::Modality::Image).into_proto();
        assert_eq!(image.modality, common::Modality::Image as i32);
    }
    #[test]
    fn extend_stop_token_ids_reaches_every_variant() {
        let ids = [7u32, 8];
        let mut req = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams::default()),
            ..Default::default()
        }));
        req.extend_stop_token_ids(&ids);
        match &req {
            ProtoGenerateRequest::Vllm(r) => {
                assert_eq!(r.sampling_params.as_ref().unwrap().stop_token_ids, ids);
            }
            _ => panic!("variant changed"),
        }

        // TRT-LLM keeps ids on the request itself.
        let mut req = ProtoGenerateRequest::Trtllm(Box::default());
        req.extend_stop_token_ids(&ids);
        match &req {
            ProtoGenerateRequest::Trtllm(r) => assert_eq!(r.stop_token_ids, ids),
            _ => panic!("variant changed"),
        }

        // Missing sampling params: untouched, no panic.
        let mut req = ProtoGenerateRequest::TokenSpeed(Box::default());
        req.extend_stop_token_ids(&ids);
    }

    #[test]
    fn chunk_semantics_follow_the_response_shape() {
        // The vLLM shape is the delta contract — the ZMQ lane emits it for
        // every engine it fronts, TokenSpeed included.
        assert!(
            ProtoGenerateStreamChunk::Vllm(vllm::GenerateStreamChunk::default())
                .chunk_semantics()
                .is_delta()
        );
        assert!(ProtoGenerateComplete::Vllm(Box::default())
            .chunk_semantics()
            .is_delta());

        // Every other shape reports running totals.
        for chunk in [
            ProtoGenerateStreamChunk::Sglang(sglang::GenerateStreamChunk::default()),
            ProtoGenerateStreamChunk::Trtllm(trtllm::GenerateStreamChunk::default()),
            ProtoGenerateStreamChunk::Mlx(mlx::GenerateStreamChunk::default()),
            ProtoGenerateStreamChunk::TokenSpeed(tokenspeed::GenerateStreamChunk::default()),
        ] {
            assert_eq!(chunk.chunk_semantics(), ChunkSemantics::Cumulative);
        }
        for complete in [
            ProtoGenerateComplete::Sglang(sglang::GenerateComplete::default()),
            ProtoGenerateComplete::Trtllm(trtllm::GenerateComplete::default()),
            ProtoGenerateComplete::Mlx(mlx::GenerateComplete::default()),
            ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete::default()),
        ] {
            assert_eq!(complete.chunk_semantics(), ChunkSemantics::Cumulative);
        }
    }

    #[test]
    fn tokenspeed_generate_messages_expose_the_engine_weight_version() {
        let complete = ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete {
            weight_version: Some("v7".to_string()),
            ..Default::default()
        });
        assert_eq!(complete.weight_version(), Some("v7"));
        let empty = ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete {
            weight_version: Some(String::new()),
            ..Default::default()
        });
        assert_eq!(empty.weight_version(), None, "empty is the same as unset");
        let placeholder = ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete {
            weight_version: Some("default".to_string()),
            ..Default::default()
        });
        assert_eq!(
            placeholder.weight_version(),
            None,
            "the engine's own placeholder is not a version and must not shadow a label"
        );
        let unset = ProtoGenerateComplete::TokenSpeed(tokenspeed::GenerateComplete::default());
        assert_eq!(unset.weight_version(), None);

        let chunk = ProtoGenerateStreamChunk::TokenSpeed(tokenspeed::GenerateStreamChunk {
            weight_version: Some("v7".to_string()),
            ..Default::default()
        });
        assert_eq!(chunk.weight_version(), Some("v7"));
        let sglang = ProtoGenerateComplete::Sglang(sglang::GenerateComplete::default());
        assert_eq!(
            sglang.weight_version(),
            None,
            "no other proto carries the field"
        );
    }
}

#[cfg(test)]
mod watched_tests {
    use std::{sync::Arc, time::Duration};

    use smg_grpc_client::vllm_proto as vllm;
    use tokio::sync::oneshot;

    use super::{PrefillCachedTokens, ProtoGenerateComplete, ProtoStream};
    use crate::{
        routers::{
            common::sse::{sse_channel, SseReceiver},
            grpc::regular::streaming::eof_tests::{chunk, complete, gated_stream},
        },
        worker::{BasicWorkerBuilder, Worker},
    };

    /// A stream whose `first` frames come at once and whose `rest` wait for
    /// the returned gate, read for the client at the receiving end of the
    /// returned receiver.
    async fn watched(
        first: Vec<vllm::GenerateResponse>,
        rest: Vec<vllm::GenerateResponse>,
    ) -> (ProtoStream, oneshot::Sender<()>, SseReceiver) {
        let (gate_tx, gate_rx) = oneshot::channel();
        let (stream, _server) = gated_stream(first, gate_rx, rest, "0").await;
        let (client, rx) = sse_channel();
        (
            stream.until_client_gone(&client, "req-watched"),
            gate_tx,
            rx,
        )
    }

    /// A client that leaves while the stream waits on the engine ends the
    /// wait at once: the reader meets the end, not the engine's next frame,
    /// and nothing after it.
    #[tokio::test]
    async fn a_client_that_leaves_while_the_stream_is_parked_ends_it_at_once() {
        // A tracked layer under the watch: its worker and its load pile show
        // the inner stream going with the client.
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w1:8000").build());
        let (_gate_tx, gate_rx) = oneshot::channel();
        let (stream, _server) =
            gated_stream(vec![chunk(0, "a")], gate_rx, vec![complete(0, "stop")], "0").await;
        let (client, rx) = sse_channel();
        let mut stream = stream
            .tracked(Arc::clone(&worker), 0, true)
            .until_client_gone(&client, "req-watched");
        assert!(stream.worker().is_some(), "tracked under the watch");
        assert_eq!(worker.tracked_load(), 1);
        assert!(
            matches!(stream.next().await, Some(Ok(_))),
            "the first response is read"
        );
        assert!(!stream.client_left());
        let (item, ()) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(stream.next(), async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                drop(rx);
            })
        })
        .await
        .expect("the gate never opens; the client's leaving ends the wait");
        assert!(item.is_none(), "the client's leaving is the end");
        assert!(stream.client_left(), "the end was the client's leaving");
        assert!(
            stream.next().await.is_none(),
            "nothing after the client left"
        );
        assert!(
            stream.worker().is_none(),
            "the inner stream went with the client"
        );
        assert_eq!(worker.tracked_load(), 0, "its load released with it");
    }

    /// The client's leaving is seen through a decode-side wrapper too: dual
    /// dispatch wraps the watched stream in one once the prefill side has
    /// drained, and the reader asks the outer layer.
    #[tokio::test]
    async fn a_client_that_leaves_is_seen_through_the_decode_side_wrapper() {
        let (stream, _gate, rx) = watched(vec![chunk(0, "a")], vec![complete(0, "stop")]).await;
        let mut prefill = PrefillCachedTokens::default();
        prefill.note(&ProtoGenerateComplete::Vllm(Box::new(
            vllm::GenerateComplete {
                prompt_tokens: 4,
                cached_tokens: 2,
                ..Default::default()
            },
        )));
        let mut stream = stream.decode_side(prefill);
        assert!(matches!(stream, ProtoStream::DecodeSide(_)), "wrapped");
        assert!(
            matches!(stream.next().await, Some(Ok(_))),
            "the first response is read through the wrapper"
        );
        assert!(!stream.client_left());
        let (item, ()) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(stream.next(), async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                drop(rx);
            })
        })
        .await
        .expect("the gate never opens; the client's leaving ends the wait");
        assert!(item.is_none(), "the client's leaving is the end");
        assert!(
            stream.client_left(),
            "the end was the client's leaving, seen through the wrapper"
        );
    }

    /// A client that left before the stream was first polled reads nothing:
    /// the response the engine has ready is not read for it.
    #[tokio::test]
    async fn a_client_that_left_before_the_first_poll_reads_nothing() {
        let (mut stream, _gate, rx) = watched(vec![chunk(0, "a")], vec![]).await;
        drop(rx);
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
    }

    /// A client that stays reads the stream as it is: every response, a gap
    /// inside it waited out, and the end where the engine ends it.
    #[tokio::test]
    async fn a_client_that_stays_reads_the_stream_as_it_is() {
        let (mut stream, gate, _rx) = watched(
            vec![chunk(0, "a")],
            vec![chunk(0, "b"), complete(0, "stop")],
        )
        .await;
        assert!(matches!(stream.next().await, Some(Ok(_))));
        let (second, ()) = tokio::join!(stream.next(), async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = gate.send(());
        });
        assert!(
            matches!(second, Some(Ok(_))),
            "the gap inside the stream is waited out"
        );
        assert!(matches!(stream.next().await, Some(Ok(_))));
        assert!(stream.next().await.is_none(), "the end is the end");
    }
}
