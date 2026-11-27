//! Protocol buffer type wrappers for the supported gRPC backends.
//!
//! This module provides unified enums that wrap proto types from each
//! supported backend, allowing the router to work with any backend
//! transparently.

use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

use futures::future::select_all;
use futures_util::StreamExt;
use smg_grpc_client::{
    abort_on_drop::DEFERRED_ABORT_MAX_WAIT,
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
use smg_worker::{health::liveness, identity::basic::Worker, runtime::WorkerRuntime};
use tokio::time::error::Elapsed;

use crate::routers::{
    common::{
        placement::{RequestEnd, SharedEnd},
        sse::SseSender,
    },
    grpc::{
        multimodal::{log_mm_timing_enabled, mm_rdma_exporter},
        utils::tonic_ext::TonicStatusExt,
        zmq_client::ZmqGenerateStream,
    },
};

mod embed;
mod logprobs;
mod multimodal;
mod request;
mod shm;

pub(crate) use smg_grpc_client::media_identity::identity as vllm_mm_identity;

use self::shm::vllm_mm_hash_identity;
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
        collect_vllm_generate_request_shm_handles, mm_shm_dev_writable, write_epd_shm_with,
        write_tokenspeed_shm_with, EpdShmGuard,
    },
};
pub(crate) use self::{
    multimodal::{stage_tokenspeed_tensor_rdma, EncodeItemBootstrapInfo},
    request::VLLM_MROPE_GRID_KEYS,
    shm::{
        cleanup_generate_request_shm, cleanup_tokenspeed_items_encoder_shm,
        finish_tokenspeed_request, finish_vllm_request,
    },
};

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

/// The wire dispatch of a counter every wire carries under one field name,
/// written once per form. A wire added to the wrappers is one arm in each
/// form, two in all, where every reader spelled its own before. The compiler
/// still refuses a reader that lacks a wire; the macro only moves where the
/// arms are written. The second form names the wire without the field and
/// what it reads instead.
macro_rules! wire_field {
    ($on:expr, $c:ident => $read:expr) => {
        match $on {
            Self::Sglang($c) => $read,
            Self::Vllm($c) => $read,
            Self::Trtllm($c) => $read,
            Self::Mlx($c) => $read,
            Self::TokenSpeed($c) => $read,
        }
    };
    ($on:expr, $c:ident => $read:expr, Mlx => $none:expr) => {
        match $on {
            Self::Sglang($c) => $read,
            Self::Vllm($c) => $read,
            Self::Trtllm($c) => $read,
            Self::TokenSpeed($c) => $read,
            Self::Mlx(_) => $none,
        }
    };
}

/// The accessors every engine-response wrapper answers with the same five-arm
/// match, written once and invoked inside each wrapper's `impl` (as the engine
/// clients invoke `impl_engine_client_basics`): the chunk semantics of the
/// response shape, the index, the four token counters and the reported weight
/// version. What differs per wrapper (the token ids, the logprobs, the finish
/// reason) stays in its own `impl`.
macro_rules! engine_response_basics {
    () => {
        /// How this response's payloads relate to the preceding ones of its
        /// stream — see [`ChunkSemantics`]: a `Delta` stream's counts are
        /// accumulated chunk by chunk, a `Cumulative` stream's `Complete` reports
        /// them.
        pub fn chunk_semantics(&self) -> ChunkSemantics {
            match self {
                Self::Vllm(_) => ChunkSemantics::Delta,
                Self::Sglang(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => {
                    ChunkSemantics::Cumulative
                }
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

        /// Get prompt tokens
        pub fn prompt_tokens(&self) -> u32 {
            wire_field!(self, c => c.prompt_tokens)
        }

        /// Get completion tokens (the running count on a chunk, the total on a
        /// `Complete`)
        pub fn completion_tokens(&self) -> u32 {
            wire_field!(self, c => c.completion_tokens)
        }

        /// Get cached tokens
        pub fn cached_tokens(&self) -> u32 {
            wire_field!(self, c => c.cached_tokens)
        }

        /// Get reasoning tokens: the engine's own count where its wire carries
        /// one, no higher than the completion count of the same frame. A
        /// speculative step that lands past the token limit is counted whole
        /// as reasoning while the completion count stops at the limit; read
        /// here, every sum, settlement and answer takes a count that was
        /// possible for its choice. Zero on a wire that carries none.
        pub fn reasoning_tokens(&self) -> u32 {
            match self {
                Self::Sglang(c) => c.reasoning_tokens.min(c.completion_tokens),
                Self::Vllm(_) | Self::Trtllm(_) | Self::Mlx(_) | Self::TokenSpeed(_) => 0,
            }
        }

        /// Weight version the engine reported for this response. Only TokenSpeed
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
    };
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
    engine_response_basics!();

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

    /// The output's logprobs, read in place on the message; `None` when the
    /// engine sent none ([`ProtoOutputLogProbs::trtllm`] reads that wire's
    /// empty list as none).
    pub fn output_logprobs(&self) -> Option<ProtoOutputLogProbs<'_>> {
        match self {
            Self::Sglang(c) => c.output_logprobs.as_ref().map(ProtoOutputLogProbs::Sglang),
            Self::Vllm(c) => c.output_logprobs.as_ref().map(ProtoOutputLogProbs::Vllm),
            Self::Trtllm(c) => ProtoOutputLogProbs::trtllm(&c.logprobs),
            Self::Mlx(c) => c.output_logprobs.as_ref().map(ProtoOutputLogProbs::Mlx),
            Self::TokenSpeed(c) => c
                .output_logprobs
                .as_ref()
                .map(ProtoOutputLogProbs::TokenSpeed),
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
    engine_response_basics!();

    /// Get SGLang variant (panics if not SGLang)
    #[expect(
        clippy::panic,
        reason = "typed accessor: the caller guarantees the variant"
    )]
    pub fn as_sglang(&self) -> &sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            _ => panic!("Expected SGLang GenerateComplete"),
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

    /// Get accepted speculative draft tokens.
    pub fn spec_accepted_tokens(&self) -> u32 {
        wire_field!(self, c => c.spec_accepted_tokens, Mlx => 0)
    }

    /// Get proposed speculative draft tokens.
    pub fn spec_draft_tokens(&self) -> u32 {
        wire_field!(self, c => c.spec_draft_tokens, Mlx => 0)
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

    /// The output's logprobs, read in place on the message; `None` when the
    /// engine sent none ([`ProtoOutputLogProbs::trtllm`] reads that wire's
    /// empty list as none).
    pub fn output_logprobs(&self) -> Option<ProtoOutputLogProbs<'_>> {
        match self {
            Self::Sglang(c) => c.output_logprobs.as_ref().map(ProtoOutputLogProbs::Sglang),
            Self::Vllm(c) => c.output_logprobs.as_ref().map(ProtoOutputLogProbs::Vllm),
            Self::Trtllm(c) => ProtoOutputLogProbs::trtllm(&c.logprobs),
            Self::Mlx(c) => c.output_logprobs.as_ref().map(ProtoOutputLogProbs::Mlx),
            Self::TokenSpeed(c) => c
                .output_logprobs
                .as_ref()
                .map(ProtoOutputLogProbs::TokenSpeed),
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
    /// sign of life for that worker (see [`smg_worker::health::liveness`]).
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
    /// A stream that tells Grand Central how it ended (see
    /// [`ReportedStream`]). No stream of a request the classic policies
    /// placed is one.
    Reported(Box<ReportedStream>),
}

/// A worker's stream that tells Grand Central its end, through the
/// handle the execution plan asked for it once its call had answered
/// ([`ProtoStream::report_through`]). It lies
/// under the stream's tracked layer, and it is the one reader that tells a
/// worker's end from a client's leaving: the watch above drops it unread.
///
/// A clean end is the worker's 200. A failure is told by the status the
/// circuit breaker samples for it, and one the breaker takes no sample of
/// is not told ([`TonicStatusExt::end_status`]). The end is told once, and
/// the handle goes with it. A reader whose own stop ended the generation
/// tells the same good end where it lets the engine go
/// ([`ProtoStream::abort_past_stop`]). A stream dropped before its end
/// tells nothing: a client that left, a reading the gateway gave up. The
/// engine then hears that end as it did before. A stream that is one of a
/// batch's tells through the end the batch shares ([`SharedEnd`]).
pub struct ReportedStream {
    inner: ProtoStream,
    end: RequestEnd,
    /// The end a batch's streams share, where this is one of them: its own
    /// handle is empty then.
    among: Option<Arc<SharedEnd>>,
}

impl ReportedStream {
    /// The stream met `item`: an end is told, a response is not looked at.
    #[inline]
    fn met(&mut self, item: Option<&Result<ProtoGenerateResponse, tonic::Status>>) {
        let failure = match item {
            Some(Ok(_)) => return,
            Some(Err(status)) => Some(status),
            None => None,
        };
        let mut end = std::mem::take(&mut self.end);
        if let Some(shared) = self.among.take() {
            end = shared.take(failure.is_some());
        }
        match failure {
            Some(status) => end.report_with(|| status.end_status()),
            None => end.report(200, || true),
        }
    }
}

/// A stream with its first item (`None`: its end) pulled ahead of its reader
/// and held, with nothing booked for it: whatever tracks the stream later
/// books the item when its reader meets it.
pub struct AheadStream {
    inner: ProtoStream,
    first: Option<Result<ProtoGenerateResponse, tonic::Status>>,
}

/// A [`ProtoStream`] read on behalf of one client, whose response body is
/// the receiving end of the watched channel. A reader parked on the engine (a
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
    /// The wait for the client's leaving, one for the stream's whole life:
    /// built with the watch and polled only while the reader is parked on
    /// the engine (see [`ClientGone::poll_read`]), so it registers with the
    /// channel once, at the first park, instead of once per response.
    client_gone: ClientGone,
    request_id: String,
}

/// The wait for a client's leaving that one reader keeps for its whole life
/// (a watched stream, the relay of a Responses stream): the client's channel,
/// its `closed()` boxed once, and the waker that wait is registered for.
pub(crate) struct ClientGone {
    client: SseSender,
    wait: Pin<Box<dyn Future<Output = ()> + Send + Sync>>,
    waker: Option<Waker>,
}

impl ClientGone {
    /// The wait for the client at the receiving end of `client`'s channel.
    pub(crate) fn of(client: &SseSender) -> Self {
        let sender = client.clone();
        Self {
            client: client.clone(),
            wait: Box::pin(async move { sender.closed().await }),
            waker: None,
        }
    }

    /// `None` is the client's leaving, `Some` what `source` gave. The closed
    /// flag is read first, so a busy `source` cannot outrun it; the wait is
    /// polled only while `source` is pending, and not again under the waker
    /// it holds: the channel raises the flag before it wakes that waker.
    pub(crate) fn poll_read<T>(
        &mut self,
        cx: &mut Context<'_>,
        source: impl FnOnce(&mut Context<'_>) -> Poll<T>,
    ) -> Poll<Option<T>> {
        if self.client.is_closed() {
            return Poll::Ready(None);
        }
        if let Poll::Ready(item) = source(cx) {
            return Poll::Ready(Some(item));
        }
        if self
            .waker
            .as_ref()
            .is_some_and(|waker| waker.will_wake(cx.waker()))
        {
            return Poll::Pending;
        }
        if self.wait.as_mut().poll(cx).is_ready() {
            return Poll::Ready(None);
        }
        self.waker = Some(cx.waker().clone());
        Poll::Pending
    }
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
/// hit before the gateway reads it. A decode side's tracking sits inside this
/// wrapper (every route puts the wrapper on last), so that a
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
        // One entry answers every index, as the sample's own or as the whole
        // request's: no hash of the index on a decode side's every message.
        if self.by_index.len() == 1 {
            return self.by_index.values().next().copied();
        }
        self.by_index.get(&index).copied()
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
    /// The worker's state, captured at the stream's start (see
    /// [`Worker::runtime_state`]): every response's progress is noted on it
    /// and the output ticket is opened on it. `None` for a worker that keeps
    /// no state: a response then notes no progress and opens a ticket that
    /// counts nothing. It still ends a standing veto and samples the outcome.
    runtime: Option<Arc<WorkerRuntime>>,
    prefill: Option<PrefillTicket>,
    /// The request's place in the worker's pile; a non-streaming generation
    /// holds none.
    progress: Option<ProgressTicket>,
    /// Held from the stream's first response to its end: one of the streams
    /// the worker is producing for (see [`Worker::output_silence`]).
    output: Option<OutputTicket>,
    /// Whether the worker's circuit-breaker success for this dispatch is
    /// still to be sampled, at the stream's first response (a single dispatch
    /// and a disaggregated decode side: see [`ProtoStream::tracked_decode_side`]).
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
/// [`WorkerRuntime::tracked_load`]), held for the stream's life. The ticket
/// holds the state its dispatch took (see [`Worker::runtime_state`]), so its
/// end lands where its start did; a worker that keeps no state has no pile.
struct ProgressTicket {
    runtime: Option<Arc<WorkerRuntime>>,
}

impl ProgressTicket {
    /// The request joins the pile at `now_ms` on the gateway clock.
    fn open(runtime: Option<Arc<WorkerRuntime>>, now_ms: u64) -> Self {
        if let Some(runtime) = &runtime {
            runtime.note_tracked_started_at(now_ms);
        }
        Self { runtime }
    }
}

impl Drop for ProgressTicket {
    fn drop(&mut self) {
        if let Some(runtime) = &self.runtime {
            runtime.note_tracked_ended();
        }
    }
}

/// A stream the worker has produced output for and that has not ended (see
/// [`Worker::output_silence`]): opened at the first response, given back at
/// the stream's end, or when its reader lets go of it. It holds the state
/// its stream captured, as the other tickets hold their dispatch's.
struct OutputTicket {
    runtime: Option<Arc<WorkerRuntime>>,
}

impl OutputTicket {
    fn open(runtime: Option<Arc<WorkerRuntime>>) -> Self {
        if let Some(runtime) = &runtime {
            runtime.note_output_started();
        }
        Self { runtime }
    }
}

impl Drop for OutputTicket {
    fn drop(&mut self) {
        if let Some(runtime) = &self.runtime {
            runtime.note_output_ended();
        }
    }
}

/// The prompt tokens a dispatched request adds to its worker's prefill
/// backlog (see [`WorkerRuntime::prefill_backlog`]), released by the first response
/// (with the prefix-cache hit it reports, so the worker's rate learns the
/// tokens the engine computed) or, failing that, when the stream is dropped.
pub(crate) struct PrefillTicket {
    /// The state the dispatch took (see [`Worker::runtime_state`]): the
    /// prompt's start and its end are noted on it. `None` for a worker that
    /// keeps no state: it has no backlog.
    runtime: Option<Arc<WorkerRuntime>>,
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
        Self::book_at(worker, prompt_tokens, liveness::now_ms())
    }

    /// [`Self::book`] at `now_ms` on the gateway clock (`liveness::now_ms`):
    /// the dispatch's one stamp, shared with its deadline.
    pub(crate) fn book_at(
        worker: &Arc<dyn Worker>,
        prompt_tokens: u64,
        now_ms: u64,
    ) -> Option<Self> {
        (prompt_tokens > 0).then(|| Self::open(worker.runtime_state(), prompt_tokens, true, now_ms))
    }

    /// A booking, `timed` or not, for a prompt that is not empty, on the state
    /// its dispatch took.
    fn open(
        runtime: Option<Arc<WorkerRuntime>>,
        prompt_tokens: u64,
        timed: bool,
        now_ms: u64,
    ) -> Self {
        if let Some(runtime) = &runtime {
            // 0 is the runtime's unset stamp.
            runtime.note_prefill_started_at(prompt_tokens, timed, now_ms.max(1));
        }
        Self {
            runtime,
            tokens: prompt_tokens,
            timed,
            released: false,
        }
    }

    /// The first response came back at `now_ms` on the gateway clock (the
    /// stamp its reader took for the message), reporting `cached` prompt
    /// tokens served from the engine's prefix cache.
    fn first_response_at(&mut self, cached: u64, now_ms: u64) {
        if !self.released {
            self.released = true;
            if let Some(runtime) = &self.runtime {
                // 0 is the runtime's unset stamp.
                runtime.note_prefill_ended_at(self.tokens, self.timed, Some(cached), now_ms.max(1));
            }
        }
    }
}

impl Drop for PrefillTicket {
    fn drop(&mut self) {
        if !self.released {
            if let Some(runtime) = &self.runtime {
                runtime.note_prefill_ended(self.tokens, self.timed, None);
            }
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
        Self::book_at(worker, prompt_tokens, streaming, Instant::now())
    }

    /// [`Self::book`] for a request dispatched at `dispatched_at`: the one
    /// instant the dispatch read, which its deadline runs from too, stamps
    /// the prefill booking and the start of the worker's tracked run.
    pub(crate) fn book_at(
        worker: &Arc<dyn Worker>,
        prompt_tokens: u64,
        streaming: bool,
        dispatched_at: Instant,
    ) -> Self {
        let now_ms = liveness::ms_at(dispatched_at);
        // The worker's state is taken once for both tickets, and only when
        // there is a ticket to open.
        let runtime = (streaming || prompt_tokens > 0)
            .then(|| worker.runtime_state())
            .flatten();
        let progress = streaming.then(|| ProgressTicket::open(runtime.clone(), now_ms));
        let prefill = (prompt_tokens > 0)
            .then(|| PrefillTicket::open(runtime, prompt_tokens, streaming, now_ms));
        Self { prefill, progress }
    }
}

/// The dispatch tickets' benchmark seam, what `benches/dispatch_tickets.rs`
/// drives: one streaming request's tickets on `worker` for their whole life,
/// with no stream behind them. The booking opens the request's place in the
/// pile and its prompt's prefill, the first response ends the prefill and
/// opens the output, and the drop gives all three back.
#[cfg(any(test, feature = "test-util"))]
pub fn dispatch_ticket_life(worker: &Arc<dyn Worker>, prompt_tokens: u64) {
    let DispatchBooking {
        mut prefill,
        progress,
    } = DispatchBooking::book(worker, prompt_tokens, true);
    // What a stream does at its start and at its first response.
    let runtime = worker.runtime_state();
    let output = OutputTicket::open(runtime.clone());
    if let Some(ticket) = prefill.as_mut() {
        ticket.first_response_at(0, liveness::now_ms());
    }
    drop((output, prefill, progress, runtime));
}

#[cfg(test)]
#[path = "../../../tests/unit/routers/grpc/proto_wrapper/tracked_tests.rs"]
mod tracked_tests;

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
        let runtime = worker.runtime_state();
        Self::Tracked(Box::new(TrackedStream {
            inner: self,
            worker,
            runtime,
            prefill,
            progress,
            output: None,
            samples_success: false,
            replay: None,
        }))
    }

    /// [`Self::tracked_booked`] for a single dispatch: the worker's breaker
    /// success is sampled at the first response, not when the call returned
    /// its stream, as a decode side's is (see [`Self::tracked_decode_side`]).
    #[must_use]
    pub(crate) fn tracked_single_booked(
        self,
        worker: Arc<dyn Worker>,
        booking: DispatchBooking,
    ) -> Self {
        let mut stream = self.tracked_booked(worker, booking);
        if let Self::Tracked(tracked) = &mut stream {
            tracked.samples_success = true;
        }
        stream
    }

    /// Track this stream as the prefill side of a disaggregated dispatch to
    /// `worker`, its prompt booked at the dispatch (see [`PrefillTicket::book`]):
    /// the stream takes over the ticket, so the first response ends the span
    /// that began before the call was polled and times the prompt's prefill
    /// for the worker's rate (the side's first response is the end of its
    /// prefill, on either dispatch shape), and every response is progress for
    /// the worker. The side joins no pile: the gateway bounds its silence per
    /// request (see [`smg_worker::health::liveness::prefill_side_bound`]) instead
    /// of judging the worker by sides it cannot see progress on.
    #[must_use]
    pub(crate) fn tracked_prefill_side_booked(
        self,
        worker: Arc<dyn Worker>,
        prefill: Option<PrefillTicket>,
    ) -> Self {
        let runtime = worker.runtime_state();
        Self::Tracked(Box::new(TrackedStream {
            inner: self,
            worker,
            runtime,
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
        tokio::time::timeout_at(deadline.into(), self.first_response()).await
    }

    /// [`Self::first_response_within`] with no deadline: the first item, a
    /// response or the end, however long it takes (the prefill side of an
    /// EPD request with the silence rule off waits for it beside its
    /// request's encode watch).
    pub async fn first_response(&mut self) -> FirstItem {
        if let Self::Watched(watched) = self {
            return match watched.inner.as_mut() {
                Some(inner) => Box::pin(inner.first_response()).await,
                None => FirstItem::End,
            };
        }
        let Self::Tracked(tracked) = self else {
            return FirstItem::Response;
        };
        if let Some(held) = &tracked.replay {
            return FirstItem::of(held.as_ref());
        }
        let item = self.next().await;
        let first = FirstItem::of(item.as_ref());
        if let Self::Tracked(tracked) = self {
            tracked.replay = Some(item);
        }
        first
    }

    /// The stream's first item (`None`: its end), pulled ahead of its reader
    /// and held for the next [`Self::next`] as it was, unbooked: a stream
    /// tracked later books it when its reader meets it. Only for a stream
    /// nothing tracks yet (a parallel side's, decode or prefill, while the
    /// pair is still dispatching). Cancel-safe: dropped before the item
    /// comes, it leaves the stream as it was.
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
        self.watched(request_id, ClientGone::of(client))
    }

    /// [`Self::until_client_gone`] with `client_gone` as the wait for the
    /// client's leaving: the sender's own `closed()`, or a test's view of it.
    fn watched(self, request_id: &str, client_gone: ClientGone) -> Self {
        Self::Watched(Box::new(WatchedStream {
            inner: Some(self),
            client_gone,
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

    /// From here on this stream tells Grand Central its end through
    /// `end` (see [`ReportedStream`]). An empty handle leaves the stream as
    /// it is: a request the classic policies placed, or one under a policy
    /// document no stage of which takes an end. Such a request pays the
    /// handle's test, and its stream is not moved.
    #[inline]
    pub(crate) fn report_through(&mut self, end: RequestEnd) {
        if !end.is_empty() {
            self.put_report_layer(end, None);
        }
    }

    /// [`Self::report_through`] behind its test of the handle: the layer is
    /// made whatever `end` holds. It goes under the stream's tracked layer,
    /// where it has one. A pair's decode side may come stamped
    /// ([`Self::decode_side`]): the layer goes under the stamp too, so it
    /// lies in one place whichever shape sent the pair. The decision bench
    /// counts the layer through this, on an empty handle. One of a batch's
    /// streams takes `among`, the end the batch shares, and tells through
    /// it.
    ///
    /// The stage puts the layer before a reader watches the stream for its
    /// client: over a watch the layer would read a client's leaving as the
    /// worker's end. A debug build stops at a stream watched already.
    ///
    /// The tracked layer may hold the stream's first item already, read
    /// ahead of the layer: a decode sent behind its prefill is waited for
    /// within its bound. No read hands that item down, so the layer meets
    /// it here.
    #[inline(never)]
    pub(crate) fn put_report_layer(&mut self, end: RequestEnd, among: Option<Arc<SharedEnd>>) {
        debug_assert!(
            !matches!(self, Self::Watched(_)),
            "the layer is put before the stream is watched"
        );
        let over = match self {
            Self::DecodeSide(side) => &mut side.inner,
            stream => stream,
        };
        let (over, held) = match over {
            Self::Tracked(tracked) => {
                let TrackedStream { inner, replay, .. } = &mut **tracked;
                (inner, replay.as_ref())
            }
            stream => (stream, None),
        };
        // Nothing unwinds between the take and the write that follows it.
        let empty = Self::Fanout(FanoutStream::new(Vec::new()));
        let inner = std::mem::replace(over, empty);
        let mut layer = ReportedStream { inner, end, among };
        if let Some(held) = held {
            layer.met(held.as_ref());
        }
        *over = Self::Reported(Box::new(layer));
    }

    /// The reader's own stop ended the generation while the engine streamed
    /// on. The answer is whole, so the stream tells Grand Central the
    /// worker's 200 as at a clean end ([`ReportedStream`]). It is then
    /// dropped un-marked, which aborts the engine's tail.
    ///
    /// A streaming reader's stream over a classic request is the watch for
    /// its client over the tracked layer over its worker's own stream, and
    /// a decode side lays its stamp over that. Such a stream holds no layer
    /// to tell through: it is tested for that shape in line and dropped as
    /// before. A stream of any other shape walks its layers out of line.
    #[inline]
    pub(crate) fn abort_past_stop(mut self) {
        let plain = self.has_only_watch_and_tracking()
            || matches!(
                &self,
                Self::DecodeSide(side) if side.inner.has_only_watch_and_tracking()
            );
        if !plain {
            self.tell_stop();
        }
    }

    /// Whether this is the watch for a client over the tracked layer over
    /// a stream with no layer ([`Self::has_no_layer`]), and nothing else.
    #[inline]
    fn has_only_watch_and_tracking(&self) -> bool {
        let Self::Watched(watched) = self else {
            return false;
        };
        matches!(&watched.inner, Some(Self::Tracked(tracked)) if tracked.inner.has_no_layer())
    }

    /// [`Self::abort_past_stop`] for a stream that may hold the layer, out
    /// of line: the layers are walked, and the layer tells the good end.
    #[inline(never)]
    fn tell_stop(&mut self) {
        if let Some(reported) = self.reported() {
            reported.met(None);
        }
    }

    /// Whether this stream holds no layer the walk to the layer enters
    /// ([`Self::reported`]): a worker's own stream, or a fan-out. A fan-out's
    /// children are streams of their own, and the walk does not enter one
    /// either, so naming it here changes no answer. A form added to the
    /// stream counts as layered until it is named here: it walks its layers
    /// at a reader's stop.
    #[inline]
    fn has_no_layer(&self) -> bool {
        matches!(
            self,
            Self::Sglang(_)
                | Self::Vllm(_)
                | Self::Trtllm(_)
                | Self::Mlx(_)
                | Self::TokenSpeed(_)
                | Self::Zmq(_)
                | Self::Fanout(_)
        )
    }

    /// The layer that tells this stream's end, under whatever lies over it.
    fn reported(&mut self) -> Option<&mut ReportedStream> {
        match self {
            Self::Reported(reported) => Some(reported),
            Self::Tracked(tracked) => tracked.inner.reported(),
            Self::DecodeSide(side) => side.inner.reported(),
            Self::Ahead(ahead) => ahead.inner.reported(),
            Self::Watched(watched) => watched.inner.as_mut()?.reported(),
            Self::Sglang(_)
            | Self::Vllm(_)
            | Self::Trtllm(_)
            | Self::Mlx(_)
            | Self::TokenSpeed(_)
            | Self::Zmq(_)
            | Self::Fanout(_) => None,
        }
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

    /// `item` as this stream hands it on: under a decode-side wrapper a
    /// response carries the prefill side's prefix-cache hit instead of its
    /// own, stamped where the item lies. The inner stream did its own
    /// bookkeeping and engine-error mapping; only the prompt-side usage
    /// changes here. The wrapper is looked at when the item is in hand:
    /// nothing of it is carried across a wait, and a stream with no wrapper
    /// pays one test a message.
    #[inline]
    fn restamp(&self, item: &mut Option<Result<ProtoGenerateResponse, tonic::Status>>) {
        if let (Self::DecodeSide(side), Some(Ok(response))) = (self, item) {
            response.restamp_cached_tokens(&side.prefill);
        }
    }

    /// Get next item from stream
    pub async fn next(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
        // A decode side is read through the layers it wraps, in place, and
        // restamped on the way out: read through this function a second
        // time, its messages paid a boxed future and each layer's dispatch
        // twice.
        let stream = match &mut *self {
            Self::DecodeSide(side) => &mut side.inner,
            stream => stream,
        };
        let Self::Watched(watched) = stream else {
            let mut item = stream.next_unwatched().await;
            self.restamp(&mut item);
            return item;
        };
        let WatchedStream {
            inner, client_gone, ..
        } = &mut **watched;
        // Nothing after the client left: the inner stream went with it.
        let stream = inner.as_mut()?;
        // The inner stream is polled in place, without a boxed future per
        // message.
        let item = {
            let mut next = std::pin::pin!(stream.next_unwatched());
            std::future::poll_fn(|cx| client_gone.poll_read(cx, |cx| next.as_mut().poll(cx))).await
        };
        match item {
            Some(mut item) => {
                self.restamp(&mut item);
                item
            }
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
            // Reached with a watch over the wrapper, or with one wrapper
            // inside another: [`Self::next`] reads a decode side in place. A
            // tracked stream or a wrapper under this one is read through
            // `next`, behind a boxed future. The inner stream did its own
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
                // One stamp for the message: the worker's progress note and,
                // on the first response, the end of the prompt's prefill.
                let now_ms = liveness::now_ms();
                match &tracked.runtime {
                    Some(runtime) => {
                        liveness::on_token_progress_on(runtime, &tracked.worker, now_ms);
                    }
                    None => liveness::on_token_progress_at(&tracked.worker, now_ms),
                }
                if std::mem::take(&mut tracked.samples_success) {
                    tracked.worker.record_outcome(200);
                }
                if tracked.output.is_none() {
                    tracked.output = Some(OutputTicket::open(tracked.runtime.clone()));
                }
                // The ticket is taken, not cleared after the fact: a clear
                // of an empty slot still runs the slot's drop glue, on
                // every message of the stream.
                if let Some(mut ticket) = tracked.prefill.take() {
                    ticket.first_response_at(u64::from(response.cached_tokens()), now_ms);
                }
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
                .map(|result| result.map(ProtoGenerateResponse::Sglang)),
            Self::Vllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(ProtoGenerateResponse::Vllm)),
            Self::Trtllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(ProtoGenerateResponse::Trtllm)),
            Self::Mlx(stream) => stream
                .next()
                .await
                .map(|result| result.map(ProtoGenerateResponse::Mlx)),
            Self::TokenSpeed(stream) => stream
                .next()
                .await
                .map(|result| result.map(ProtoGenerateResponse::TokenSpeed)),
            // Every ZMQ engine (including TokenSpeed) emits vllm-shaped
            // responses: the adapter translates wire output into
            // `vllm::GenerateResponse`, so a response's variant does not name
            // the engine behind a ZMQ-backed stream. Key response-side engine
            // logic on the worker's `runtime_type()`, never on the response
            // variant; key accumulation on `chunk_semantics()`, which the vLLM
            // shape (delta chunks, cumulative `Complete`) defines for this lane.
            Self::Zmq(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Vllm(Box::new(r)))),
            Self::Fanout(stream) => stream.next().await,
            Self::Tracked(_) | Self::Watched(_) => return Box::pin(self.next()).await,
            // Reached with a tracked layer over the wrapper, which no route
            // builds: [`Self::next`] reads a decode side in place.
            Self::DecodeSide(_) => return Box::pin(self.next()).await,
            // The held item, mapped when it was read, and the layer goes.
            Self::Ahead(ahead) => {
                let first = ahead.first.take();
                let empty = Self::Fanout(FanoutStream::new(Vec::new()));
                *self = std::mem::replace(&mut ahead.inner, empty);
                return first;
            }
            // Read behind a boxed future, as any layer under the tracked
            // one is: the stream under it mapped the item.
            Self::Reported(reported) => {
                let item = Box::pin(reported.inner.next_untracked()).await;
                reported.met(item.as_ref());
                return item;
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
            Self::Reported(reported) => reported.inner.mark_completed(),
        }
    }

    /// Bound the stream by the request's `deadline`
    /// (`--grpc-request-timeout-secs`, counted from the dispatch): polled at
    /// or past it, the transport stream
    /// ends with the deadline's status and is released as a dropped one is,
    /// so its abort reaches the engine at once (see
    /// [`smg_grpc_client::AbortOnDropStream::set_deadline`]). The wrappers (a
    /// tracked stream, a decode side, a pulled-ahead item, a watched stream, a
    /// fan-out) reach the transport stream(s) under them through their
    /// existing boxes, so a deadline costs no allocation on the request; a
    /// fan-out bounds each child. The ZMQ wire's stream has no deadline hook
    /// and is left as it is.
    pub fn apply_deadline(&mut self, deadline: RequestDeadline) {
        match self {
            Self::Sglang(stream) => stream.set_deadline(deadline),
            Self::Vllm(stream) => stream.set_deadline(deadline),
            Self::Trtllm(stream) => stream.set_deadline(deadline),
            Self::Mlx(stream) => stream.set_deadline(deadline),
            Self::TokenSpeed(stream) => stream.set_deadline(deadline),
            Self::Zmq(_) => {}
            Self::Fanout(stream) => stream.apply_deadline(deadline),
            Self::Tracked(stream) => stream.inner.apply_deadline(deadline),
            Self::DecodeSide(side) => side.inner.apply_deadline(deadline),
            // The item pulled ahead is already in hand; the stream under it
            // is bounded.
            Self::Ahead(ahead) => ahead.inner.apply_deadline(deadline),
            // The watched stream's inner stream is bounded while the client is
            // still there; once it left, there is nothing left to bound.
            Self::Watched(watched) => {
                if let Some(inner) = watched.inner.as_mut() {
                    inner.apply_deadline(deadline);
                }
            }
            Self::Reported(reported) => reported.inner.apply_deadline(deadline),
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
        self.defer_abort_within(DEFERRED_ABORT_MAX_WAIT)
    }

    /// [`Self::defer_abort_until_first_item`] with the wait for the first
    /// response bounded by `bound` where that is shorter than the house cap
    /// ([`DEFERRED_ABORT_MAX_WAIT`]): the bound the caller holds for the
    /// response, a decode side's first-token bound. A side whose first
    /// response is its whole answer (a non-streaming generation) has no
    /// earlier signal, and a drop would otherwise hold its worker for the
    /// answer's length, the cap at most.
    #[must_use]
    pub fn defer_abort_within(self, bound: Duration) -> Self {
        match self {
            Self::Sglang(stream) => Self::Sglang(stream.defer_abort_within(bound)),
            Self::Vllm(stream) => Self::Vllm(stream.defer_abort_within(bound)),
            Self::Trtllm(stream) => Self::Trtllm(stream.defer_abort_within(bound)),
            Self::Mlx(stream) => Self::Mlx(stream.defer_abort_within(bound)),
            Self::TokenSpeed(stream) => Self::TokenSpeed(stream.defer_abort_within(bound)),
            Self::Zmq(stream) => Self::Zmq(stream),
            Self::Fanout(stream) => Self::Fanout(stream.defer_abort_within(bound)),
            Self::Tracked(stream) => {
                let TrackedStream {
                    inner,
                    worker,
                    runtime,
                    prefill,
                    progress,
                    output,
                    samples_success,
                    replay,
                } = *stream;
                Self::Tracked(Box::new(TrackedStream {
                    inner: inner.defer_abort_within(bound),
                    worker,
                    runtime,
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
                    inner: inner.defer_abort_within(bound),
                    prefill,
                }))
            }
            Self::Ahead(ahead) => {
                let AheadStream { inner, first } = *ahead;
                Self::Ahead(Box::new(AheadStream {
                    inner: inner.defer_abort_within(bound),
                    first,
                }))
            }
            Self::Watched(watched) => {
                let WatchedStream {
                    inner,
                    client_gone,
                    request_id,
                } = *watched;
                Self::Watched(Box::new(WatchedStream {
                    inner: inner.map(|inner| inner.defer_abort_within(bound)),
                    client_gone,
                    request_id,
                }))
            }
            Self::Reported(reported) => {
                let ReportedStream { inner, end, among } = *reported;
                let inner = inner.defer_abort_within(bound);
                Self::Reported(Box::new(ReportedStream { inner, end, among }))
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

    fn defer_abort_within(self, bound: Duration) -> Self {
        Self::defer_abort_within(self, bound)
    }

    fn apply_deadline(&mut self, deadline: RequestDeadline) {
        Self::apply_deadline(self, deadline);
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

    /// Defer the child's abort-on-drop until its first response, the wait
    /// bounded by `bound` where that is shorter than the house cap.
    #[must_use]
    fn defer_abort_within(self, bound: Duration) -> Self;

    /// Bound the child by the request's `deadline`, in place (see
    /// [`ProtoStream::apply_deadline`]); a child with no transport stream
    /// under it is left as it is.
    fn apply_deadline(&mut self, _deadline: RequestDeadline) {}

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

    fn defer_abort_within(self, bound: Duration) -> Self {
        Self::defer_abort_within(self, bound)
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
    #[cfg(test)]
    pub(crate) fn open_children(&self) -> usize {
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

    /// Defer every open child's abort-on-drop until its first response,
    /// each child's wait bounded by `bound`.
    #[must_use]
    pub fn defer_abort_within(self, bound: Duration) -> Self {
        Self {
            children: self
                .children
                .into_iter()
                .map(|child| child.map(|child| child.defer_abort_within(bound)))
                .collect(),
            cursor: self.cursor,
        }
    }

    /// Bound every open child by the request's `deadline`, in place.
    pub fn apply_deadline(&mut self, deadline: RequestDeadline) {
        for child in self.children.iter_mut().flatten() {
            child.apply_deadline(deadline);
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/routers/grpc/proto_wrapper/fanout_tests.rs"]
mod fanout_tests;

#[cfg(test)]
#[path = "../../../tests/unit/routers/grpc/proto_wrapper/engine_error_tests.rs"]
mod engine_error_tests;

#[cfg(test)]
#[path = "../../../tests/unit/routers/grpc/proto_wrapper/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../tests/unit/routers/grpc/proto_wrapper/watched_tests.rs"]
mod watched_tests;

#[cfg(test)]
#[path = "../../../tests/unit/routers/grpc/proto_wrapper/output_logprob_tests.rs"]
mod output_logprob_tests;

#[cfg(test)]
#[path = "../../../tests/unit/routers/grpc/proto_wrapper/wire_field_tests.rs"]
mod wire_field_tests;

#[cfg(test)]
#[path = "../../../tests/unit/routers/grpc/proto_wrapper/first_response_tests.rs"]
mod first_response_tests;
