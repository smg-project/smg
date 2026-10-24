//! Streaming response processor for gRPC routers
//!
//! This module contains shared streaming logic for both Regular and PD router.

use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use axum::response::Response;
use bytes::{Bytes, BytesMut};
use futures::future::try_join_all;
use llm_tokenizer::{
    stop::{SequenceDecoderOutput, StopSequenceDecoder},
    traits::Tokenizer,
};
use openai_protocol::{
    chat::{ChatCompletionStreamResponse, ChatStreamChoice},
    common::{
        ChatLogProbs, FunctionCallDelta, StringOrArray, Tool, ToolCallDelta, ToolChoice,
        ToolChoiceValue, Usage,
    },
    completion::{CompletionStreamChoice, CompletionStreamResponse},
    messages::{
        self, ContentBlock, ContentBlockDelta, Message, MessageDelta, MessageDeltaUsage,
        MessageStreamEvent,
    },
    profile::ProviderProfile,
};
use reasoning_parser::{ParserFactory as ReasoningParserFactory, ParserResult, ReasoningParser};
use serde::Serialize;
use serde_json::{json, Value};
use symphony::{adapt, EngineFinish, Event, Events};
use tokio_stream::wrappers::ReceiverStream;
use tool_parser::{ParserFactory as ToolParserFactory, StreamingParseResult, ToolParser};
use tracing::{debug, error, warn};

use super::symphony::{
    engine_finish, finished, reasoning_text_tokens, telemetry::Who, Delta, Reader,
};
use crate::{
    observability::metrics::{metrics_labels, LocalStopAbort, Metrics, StreamingMetricsParams},
    rate_limit::{SharedReservationHandle, UsageSettlement},
    routers::{
        common::{
            sse::{sse_channel, SseEncoder, SseSender},
            sse_rechunk,
        },
        grpc::{
            common::{
                response_collection::drain_prefill,
                response_formatting::{
                    chat_usage, effective_weight_version, CompletionTokenTracker, TokenCounts,
                },
                responses::{
                    body_guards, build_sse_response_from_stream, build_sse_response_guarded,
                },
            },
            context,
            proto_wrapper::{
                ProtoGenerateComplete, ProtoGenerateStreamChunk, ProtoInputLogProbs,
                ProtoResponseVariant, ProtoStream,
            },
            spec::{
                ChatResponseSpec, CompletionResponseSpec, GenerateResponseSpec,
                MessagesResponseSpec,
            },
            utils,
            utils::message_utils,
        },
    },
    worker::{GuardedStream, PrefillLoadGuard},
};

mod completion;
mod frames;

use self::{
    completion::{CompletionStream, CompletionStreamOutcome, CompletionStreamUnit},
    frames::{
        assistant_choice, forces_tool_call, opening_choice, ChatChunkFrame, CompletionChunkFrame,
        CHUNK_FLUSH_BYTES,
    },
};

/// Running usage is separate from settlement state: only Complete messages
/// authorize settlement, even when intermediate chunks already report counts.
#[derive(Default)]
struct ChatStreamUsage {
    choices: HashMap<u32, ChatStreamTokenCounts>,
    /// The gateway's reasoning parser runs for this request and counts the
    /// tokens it routes to reasoning (reported even when zero).
    reasoning_parsed: bool,
}

#[derive(Default)]
struct ChatStreamTokenCounts {
    prompt: u32,
    completion: u32,
    cached: u32,
    reasoning: u32,
    reasoning_parsed: u32,
    spec_accepted: u32,
    spec_drafted: u32,
}

impl ChatStreamUsage {
    fn record_chunk(&mut self, chunk: &ProtoGenerateStreamChunk) {
        let counts = self.choices.entry(chunk.index()).or_default();
        counts.prompt = chunk.prompt_tokens();
        counts.cached = chunk.cached_tokens();
        counts.reasoning = chunk.reasoning_tokens();
        if chunk.chunk_semantics().is_delta() {
            counts.completion += chunk.token_ids().len() as u32;
        } else {
            counts.completion = chunk.completion_tokens();
        }
    }

    fn record_complete(&mut self, complete: &ProtoGenerateComplete) {
        let counts = self.choices.entry(complete.index()).or_default();
        counts.prompt = complete.prompt_tokens();
        counts.cached = complete.cached_tokens();
        counts.reasoning = complete.reasoning_tokens();
        counts.spec_accepted = complete.spec_accepted_tokens();
        counts.spec_drafted = complete.spec_draft_tokens();
        // Match CompletionTokenTracker: delta streams retain their observed
        // token count, including when a local stop suppresses subsequent output.
        if !complete.chunk_semantics().is_delta() {
            counts.completion = complete.completion_tokens();
        }
    }

    /// Tokens of a chunk the reasoning parser attributed to reasoning.
    fn record_reasoning(&mut self, index: u32, tokens: u32) {
        self.choices.entry(index).or_default().reasoning_parsed += tokens;
    }

    /// The usage so far, in the shape of the request's final usage.
    fn snapshot(&self, unbilled_prompt_tokens: u32) -> Usage {
        // Choices share one prompt/cache but each generates its own output.
        let counts = TokenCounts {
            prompt: self.choices.values().map(|c| c.prompt).max().unwrap_or(0),
            completion: self.choices.values().map(|c| c.completion).sum(),
            cached: self.choices.values().map(|c| c.cached).max().unwrap_or(0),
            reasoning: self.choices.values().map(|c| c.reasoning).sum(),
            spec_accepted: self.choices.values().map(|c| c.spec_accepted).sum(),
            spec_drafted: self.choices.values().map(|c| c.spec_drafted).sum(),
        };
        chat_usage(
            counts.usage(),
            self.reasoning_parsed
                .then(|| self.choices.values().map(|c| c.reasoning_parsed).sum()),
            unbilled_prompt_tokens,
        )
    }
}

/// Shared streaming processor for both single and prefill/decode dispatch modes
#[derive(Clone)]
pub(crate) struct StreamingProcessor {
    tool_parser_factory: ToolParserFactory,
    reasoning_parser_factory: ReasoningParserFactory,
    /// Per-request parser-name resolution (model-card override → configured).
    parser_resolver: utils::ParserResolver,
    backend_type: &'static str,
}

/// Context for generate endpoint streaming - groups config params to reduce function arguments
struct GenerateStreamContext {
    request_id: String,
    weight_version: String,
    return_logprob: bool,
    backend_type: &'static str,
    model: String,
    /// Number of choices (`sampling_params.n`) this request expects to
    /// complete -- a clean EOF with fewer `Complete` messages than this has
    /// only partial usage, not authoritative usage for the whole request.
    expected_choices: u32,
}

impl StreamingProcessor {
    pub fn new(
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        parser_resolver: utils::ParserResolver,
        backend_type: &'static str,
    ) -> Self {
        Self {
            tool_parser_factory,
            reasoning_parser_factory,
            parser_resolver,
            backend_type,
        }
    }

    /// Process streaming chat response and return SSE response
    ///
    /// This is the high-level entry point for streaming responses, handling:
    /// - Channel creation
    /// - Background task spawning
    /// - SSE response building
    ///
    /// The `load_guards` (and the reservation's safety net) ride in the
    /// response body, released when it is dropped: at the stream's end or
    /// the client's disconnect.
    #[expect(clippy::too_many_arguments)]
    pub async fn process_streaming_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        chat_request: ChatResponseSpec,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        skip_special_tokens: bool,
        reservation: Option<Arc<SharedReservationHandle>>,
        load_guards: Option<context::LoadGuards>,
        source: utils::StreamSource,
    ) -> Response {
        let source = source.with_backend(self.backend_type);
        let guards = body_guards(load_guards, reservation.as_ref());
        use bytes::Bytes;

        let stop_params = (
            chat_request.stop.clone(),
            chat_request.stop_token_ids.clone(),
            skip_special_tokens,
            chat_request.no_stop_trim,
            chat_request.ignore_eos,
        );

        // MiniMax's stream-QoS contract bounds per-event delta sizes. The HTTP
        // relay re-slices the upstream body; here the gateway encodes the
        // frames itself, so the channel is re-chunked on its way out.
        let rechunk = chat_request.provider == ProviderProfile::Minimax;

        // Create SSE channel
        let (tx, rx) = sse_channel();
        // Read for this client: the moment the response body is dropped, the
        // request is aborted on the engine (see `ProtoStream::until_client_gone`).
        let execution_result = execution_result.until_client_gone(&tx, &dispatch.request_id);

        // Spawn background task based on execution mode
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                let processor = self.clone();
                let dispatch_clone = dispatch.clone();
                let tokenizer_clone = tokenizer.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "streaming task is fire-and-forget; client disconnect terminates it"
                )]
                tokio::spawn(async move {
                    let result = processor
                        .process_streaming_chunks(
                            stream,
                            dispatch_clone,
                            tokenizer_clone,
                            stop_params,
                            chat_request,
                            &tx,
                            reservation,
                        )
                        .await;

                    if let Err(failure) = result {
                        utils::send_stream_failure(&tx, &failure, &source).await;
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
                });
            }
            context::ExecutionResult::PrefillDecode {
                prefill,
                decode,
                prefill_guards,
                pd_timing,
                // Chat streaming does not emit prompt logprobs.
                prefill_input_logprobs: _,
            } => {
                let processor = self.clone();
                let tokenizer_clone = tokenizer.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "streaming task is fire-and-forget; client disconnect terminates it"
                )]
                tokio::spawn(async move {
                    let result = processor
                        .process_prefill_decode_streaming_chunks(
                            prefill,
                            *decode,
                            dispatch,
                            tokenizer_clone,
                            stop_params,
                            chat_request,
                            &tx,
                            pd_timing,
                            prefill_guards,
                            reservation,
                        )
                        .await;

                    if let Err(failure) = result {
                        utils::send_stream_failure(&tx, &failure, &source).await;
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
                });
            }
            context::ExecutionResult::Embedding { .. } => {
                utils::send_error_sse(
                    &tx,
                    "Embeddings not supported in streaming mode",
                    "invalid_request_error",
                )
                .await;
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
            }
            // Batch results exist only on the completions pipeline.
            context::ExecutionResult::Batch { .. } => {
                utils::send_error_sse(
                    &tx,
                    "Batched results not supported in chat streaming",
                    "invalid_request_error",
                )
                .await;
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
            }
        }

        // Return SSE response; its body holds the guards for as long as it lives.
        let stream = GuardedStream::new(ReceiverStream::new(rx), guards);
        if rechunk {
            build_sse_response_from_stream(sse_rechunk::rechunk_stream(
                stream,
                sse_rechunk::NoopObserver,
            ))
        } else {
            build_sse_response_from_stream(stream)
        }
    }

    /// Process streaming chunks from a single stream (Regular mode)
    #[expect(clippy::too_many_arguments)]
    pub async fn process_streaming_chunks(
        &self,
        grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        original_request: ChatResponseSpec,
        tx: &SseSender,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), utils::StreamFailure> {
        self.process_streaming_chunks_inner(
            grpc_stream,
            dispatch,
            tokenizer,
            stop_params,
            original_request,
            tx,
            None,
            reservation,
        )
        .await
    }

    /// Inner implementation shared by single-mode and PD prefill/decode streaming.
    /// `pd_timing` is `Some` only in PD mode and yields honest PD TTFT
    /// (prefill start to first decode token).
    ///
    /// The stream is a [`ChatStream`]: its set-up, the loop over the engine's
    /// messages, the parser flush, the finish chunks, the usage chunk and the
    /// settlement are its methods, run here in that order.
    #[expect(clippy::too_many_arguments)]
    async fn process_streaming_chunks_inner(
        &self,
        mut grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        original_request: ChatResponseSpec,
        tx: &SseSender,
        pd_timing: Option<context::PdTiming>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), utils::StreamFailure> {
        let mut stream = ChatStream::open(
            self,
            &dispatch,
            tokenizer,
            stop_params,
            &original_request,
            tx,
            pd_timing,
        );
        stream.read_engine(&mut grpc_stream).await?;
        // The stream's verdict, before the engine side is released.
        let client_left = grpc_stream.client_left();
        stream.release_engine(grpc_stream, metrics_labels::ENDPOINT_CHAT);
        // Nothing reaches a client that left, so its read ends with `Ok`
        // here; a sender closed between the read and the sends fails them.
        // Either way the reservation is settled with the stream's actual
        // usage first, and a failed terminal send records no completion
        // metric (the caller records the failure).
        let terminal = if client_left {
            Ok(())
        } else {
            async {
                let usage = stream.usage_snapshot();
                stream.flush_parsers(usage.as_ref()).await?;
                let final_usage = stream.final_usage();
                stream
                    .send_finish_chunks(final_usage.as_ref(), usage.as_ref())
                    .await?;
                stream.send_usage_chunk(final_usage).await
            }
            .await
        };

        stream
            .settle_and_record(reservation, client_left || terminal.is_err())
            .await;
        terminal
    }

    /// Process prefill/decode streaming chunks (prefill + decode) - PD mode
    #[expect(clippy::too_many_arguments)]
    pub async fn process_prefill_decode_streaming_chunks(
        &self,
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        original_request: ChatResponseSpec,
        tx: &SseSender,
        pd_timing: context::PdTiming,
        prefill_guards: Vec<Option<PrefillLoadGuard>>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), utils::StreamFailure> {
        // Phase 1.5: Drain prefill stream. Streaming chat does not emit prompt
        // logprobs, but the Prefill reservation must end with this phase, and
        // the client's prompt-side usage is the prefill side's.
        let prefill_cached_tokens =
            drain_prefill(&mut prefill_stream, prefill_guards, true, |_| {})
                .await
                .map_err(utils::StreamFailure::prefill)?;
        let decode_stream = decode_stream.decode_side(prefill_cached_tokens);

        // Phase 2-5: Process decode stream (same as single mode). Pass pd_timing
        // so the first decode token yields honest PD TTFT.
        // Note: decode_stream will be marked completed inside process_streaming_chunks
        let result = self
            .process_streaming_chunks_inner(
                decode_stream,
                dispatch,
                tokenizer,
                stop_params,
                original_request,
                tx,
                Some(pd_timing),
                reservation,
            )
            .await;

        // The prefill stream is marked completed after the decode's read ended
        // well: a decode that failed leaves both streams to abort on drop; a
        // client that left ends the read with `Ok` too, its decode stream
        // aborting the request on drop, the prefill's own request complete.
        if result.is_ok() {
            prefill_stream.mark_completed();
        }

        result
    }

    /// Process streaming generate response and return SSE response
    ///
    /// Simpler than chat - no tool/reasoning parsing, just text accumulation
    ///
    /// The `load_guards` (and the reservation's safety net) ride in the
    /// response body, released when it is dropped: at the stream's end or
    /// the client's disconnect.
    #[expect(clippy::too_many_arguments)]
    pub async fn process_streaming_generate(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        generate_request: GenerateResponseSpec,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        reservation: Option<Arc<SharedReservationHandle>>,
        load_guards: Option<context::LoadGuards>,
        source: utils::StreamSource,
    ) -> Response {
        let source = source.with_backend(self.backend_type);
        let guards = body_guards(load_guards, reservation.as_ref());
        // Create SSE channel
        let (tx, rx) = sse_channel();
        // Read for this client: the moment the response body is dropped, the
        // request is aborted on the engine (see `ProtoStream::until_client_gone`).
        let execution_result = execution_result.until_client_gone(&tx, &dispatch.request_id);

        // Build context once, clone for spawned task
        let ctx = GenerateStreamContext {
            request_id: dispatch.request_id.clone(),
            weight_version: dispatch
                .weight_version
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            return_logprob: generate_request.return_logprob,
            backend_type: self.backend_type,
            model: dispatch.model.clone(),
            expected_choices: generate_request.expected_choices,
        };

        // Spawn background task based on execution mode
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                let tokenizer = tokenizer.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "streaming task is fire-and-forget; client disconnect terminates it"
                )]
                tokio::spawn(async move {
                    let result =
                        Self::process_generate_streaming(tokenizer, stream, ctx, &tx, reservation)
                            .await;

                    if let Err(failure) = result {
                        utils::send_stream_failure(&tx, &failure, &source).await;
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
                });
            }
            context::ExecutionResult::PrefillDecode {
                prefill,
                decode,
                prefill_guards,
                pd_timing,
                prefill_input_logprobs,
            } => {
                // For PD mode, need to handle prefill stream for input_logprobs
                let tokenizer = tokenizer.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "streaming task is fire-and-forget; client disconnect terminates it"
                )]
                tokio::spawn(async move {
                    let result = Self::process_generate_prefill_decode_streaming(
                        tokenizer,
                        prefill,
                        *decode,
                        ctx,
                        &tx,
                        pd_timing,
                        prefill_input_logprobs,
                        prefill_guards,
                        reservation,
                    )
                    .await;

                    if let Err(failure) = result {
                        utils::send_stream_failure(&tx, &failure, &source).await;
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
                });
            }
            context::ExecutionResult::Embedding { .. } => {
                utils::send_error_sse(
                    &tx,
                    "Embeddings not supported in streaming generate",
                    "invalid_request_error",
                )
                .await;
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
            }
            // Batch results exist only on the completions pipeline.
            context::ExecutionResult::Batch { .. } => {
                utils::send_error_sse(
                    &tx,
                    "Batched results not supported in streaming generate",
                    "invalid_request_error",
                )
                .await;
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
            }
        }

        // Return SSE response
        build_sse_response_guarded(rx, guards)
    }

    /// Process streaming chunks for generate endpoint (no tool/reasoning parsing)
    /// TODO: add streaming logprob support
    async fn process_generate_streaming(
        tokenizer: Arc<dyn Tokenizer>,
        mut stream: ProtoStream,
        ctx: GenerateStreamContext,
        tx: &SseSender,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), utils::StreamFailure> {
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Track state per index for n>1 case
        let mut accumulated_texts: HashMap<u32, String> = HashMap::new();
        let mut completion_tokens_map: HashMap<u32, u32> = HashMap::new();
        // Same prompt for every index; the last Complete's value is as good as any.
        let mut prompt_tokens: u32 = 0;
        // The cache hit is a property of that shared prompt too (max, not sum);
        // reasoning is each choice's own, one entry per index.
        let mut cached_tokens: u32 = 0;
        let mut reasoning_tokens_map: HashMap<u32, u32> = HashMap::new();
        // Indices that received a `Complete` message -- the only source of
        // authoritative usage. Chunks alone (tracked in `completion_tokens_map`)
        // don't count: an index can start generating and never finish.
        let mut completed_indices: HashSet<u32> = HashSet::new();
        // Reusable SSE encoder shared across every chunk emitted for this stream.
        let mut sse_encoder = SseEncoder::new();

        while let Some(response) = stream.next().await {
            let gen_response = response.map_err(|e| {
                utils::StreamFailure::upstream(e, completion_tokens_map.values().sum::<u32>())
            })?;

            match gen_response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => {
                    // Track TTFT immediately on first chunk received from backend
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                    }

                    let index = chunk.index();

                    // Both backends send delta token_ids, so accumulate for both
                    let completion_tokens = completion_tokens_map.entry(index).or_insert(0);
                    *completion_tokens += chunk.token_ids().len() as u32;
                    let current_completion_tokens = *completion_tokens;

                    // Decode tokens to text (skip_special_tokens=true to handle newlines correctly)
                    let chunk_text = tokenizer
                        .decode(chunk.token_ids(), true)
                        .unwrap_or_default();

                    // Accumulate text for this index
                    let accumulated_text = accumulated_texts.entry(index).or_default();
                    accumulated_text.push_str(&chunk_text);

                    // Generate unique ID per index
                    let index_id = format!("{}-{}", ctx.request_id, index);

                    // Build streaming response chunk (SGLang format)
                    let chunk_response = serde_json::json!({
                        "text": accumulated_text.clone(),
                        "output_ids": chunk.token_ids(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": null,
                            "prompt_tokens": chunk.prompt_tokens(),
                            "weight_version": effective_weight_version(chunk.weight_version(), Some(ctx.weight_version.as_str())),
                            "completion_tokens": current_completion_tokens,
                            "cached_tokens": chunk.cached_tokens(),
                            "reasoning_tokens": chunk.reasoning_tokens()
                        },
                        "index": index
                    });

                    let sse_data = sse_encoder
                        .encode_data(&chunk_response)
                        .map_err(|e| format!("Failed to serialize generate chunk: {e}"))?;
                    tx.send(Ok(sse_data))
                        .await
                        .map_err(|_| "Failed to send chunk".to_string())?;
                }
                ProtoResponseVariant::Complete(complete) => {
                    let index = complete.index();
                    let accumulated_text =
                        accumulated_texts.get(&index).cloned().unwrap_or_default();
                    let completion_tokens = *completion_tokens_map.get(&index).unwrap_or(&0);
                    let index_id = format!("{}-{}", ctx.request_id, index);
                    let e2e_latency = start_time.elapsed().as_secs_f64();
                    prompt_tokens = complete.prompt_tokens();
                    cached_tokens = cached_tokens.max(complete.cached_tokens());
                    reasoning_tokens_map.insert(index, complete.reasoning_tokens());
                    completed_indices.insert(index);

                    // Send final chunk with finish_reason
                    let finish_response = serde_json::json!({
                        "text": accumulated_text,
                        "output_ids": complete.output_ids()[complete.output_ids().len().saturating_sub(1)..].to_vec(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": complete.finish_reason(),
                            "prompt_tokens": complete.prompt_tokens(),
                            "weight_version": effective_weight_version(complete.weight_version(), Some(ctx.weight_version.as_str())),
                            "completion_tokens": completion_tokens,
                            "cached_tokens": complete.cached_tokens(),
                            "reasoning_tokens": complete.reasoning_tokens(),
                            "e2e_latency": e2e_latency
                        },
                        "index": index
                    });

                    let sse_data = sse_encoder
                        .encode_data(&finish_response)
                        .map_err(|e| format!("Failed to serialize generate finish: {e}"))?;
                    tx.send(Ok(sse_data))
                        .await
                        .map_err(|_| "Failed to send finish chunk".to_string())?;

                    // Continue to process all completions if n>1
                }
                ProtoResponseVariant::None => continue,
            }
        }

        // Mark stream as completed successfully to prevent abort on drop
        stream.mark_completed();

        // Record streaming metrics
        let total_completion: u32 = completion_tokens_map.values().sum();
        if let Some(handle) = reservation {
            if completed_indices.len() as u32 >= ctx.expected_choices {
                handle
                    .settle_success(UsageSettlement {
                        actual_input_tokens: prompt_tokens,
                        completion_tokens: total_completion,
                        cached_input_tokens: cached_tokens,
                        reasoning_tokens: reasoning_tokens_map.values().sum(),
                    })
                    .await;
            } else {
                handle.close_reserved_only().await;
            }
        }
        // The read ended with the client's leaving, not with the engine's end:
        // no terminal router metric for it, as the HTTP router has none for a
        // disconnect (neither a completed request nor a failure).
        if stream.client_left() {
            return Ok(());
        }
        Self::record_generate_metrics(start_time, first_token_time, total_completion, None, &ctx);

        Ok(())
    }

    /// Process prefill/decode streaming for generate endpoint (PD mode with logprobs support)
    #[expect(
        clippy::too_many_arguments,
        reason = "streaming PD generate keeps phase streams, metadata, guards and reservation separate"
    )]
    async fn process_generate_prefill_decode_streaming(
        tokenizer: Arc<dyn Tokenizer>,
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        ctx: GenerateStreamContext,
        tx: &SseSender,
        pd_timing: context::PdTiming,
        prefill_input_logprobs: Option<ProtoInputLogProbs>,
        prefill_guards: Vec<Option<PrefillLoadGuard>>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), utils::StreamFailure> {
        // Drain prefill stream. Collect input_logprobs only when requested.
        // Sequential PD drained the stream in the execution stage, so its
        // logprobs arrive via `prefill_input_logprobs` instead.
        let mut input_token_logprobs = if ctx.return_logprob {
            prefill_input_logprobs
                .as_ref()
                .map(utils::convert_generate_input_logprobs)
        } else {
            None
        };
        let prefill_cached_tokens =
            drain_prefill(&mut prefill_stream, prefill_guards, true, |complete| {
                if ctx.return_logprob && input_token_logprobs.is_none() {
                    input_token_logprobs = complete
                        .input_logprobs()
                        .as_ref()
                        .map(utils::convert_generate_input_logprobs);
                }
            })
            .await
            .map_err(utils::StreamFailure::prefill)?;
        // The client's prompt-side usage is the prefill side's.
        let decode_stream = decode_stream.decode_side(prefill_cached_tokens);

        // Process decode stream with input_logprobs prepended. Pass pd_timing so
        // the first decode token yields honest PD TTFT.
        // Note: decode_stream will be marked completed inside the function
        let result = Self::process_generate_streaming_with_input_logprobs(
            tokenizer,
            decode_stream,
            ctx,
            input_token_logprobs,
            tx,
            Some(pd_timing),
            reservation,
        )
        .await;

        // The prefill stream is marked completed after the decode's read ended
        // well: a decode that failed leaves both streams to abort on drop; a
        // client that left ends the read with `Ok` too, its decode stream
        // aborting the request on drop, the prefill's own request complete.
        if result.is_ok() {
            prefill_stream.mark_completed();
        }

        result
    }

    /// Process generate streaming with optional input_logprobs
    async fn process_generate_streaming_with_input_logprobs(
        tokenizer: Arc<dyn Tokenizer>,
        mut stream: ProtoStream,
        ctx: GenerateStreamContext,
        input_token_logprobs: Option<Vec<Vec<Option<f64>>>>,
        tx: &SseSender,
        pd_timing: Option<context::PdTiming>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), utils::StreamFailure> {
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Track state per index for n>1 case
        let mut accumulated_texts: HashMap<u32, String> = HashMap::new();
        let mut accumulated_output_logprobs: HashMap<u32, Option<Vec<Vec<Option<f64>>>>> =
            HashMap::new();
        let mut completion_tokens_map: HashMap<u32, u32> = HashMap::new();
        // Same prompt for every index; the last Complete's value is as good as any.
        let mut prompt_tokens: u32 = 0;
        // The cache hit is a property of that shared prompt too (max, not sum);
        // reasoning is each choice's own, one entry per index.
        let mut cached_tokens: u32 = 0;
        let mut reasoning_tokens_map: HashMap<u32, u32> = HashMap::new();
        // Indices that received a `Complete` message -- the only source of
        // authoritative usage. Chunks alone (tracked in `completion_tokens_map`)
        // don't count: an index can start generating and never finish.
        let mut completed_indices: HashSet<u32> = HashSet::new();
        // Reusable SSE encoder shared across every chunk emitted for this stream.
        let mut sse_encoder = SseEncoder::new();

        while let Some(response) = stream.next().await {
            let gen_response = response.map_err(|e| {
                utils::StreamFailure::upstream(e, completion_tokens_map.values().sum::<u32>())
            })?;

            match gen_response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => {
                    // Track TTFT immediately on first chunk received from backend
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                        if let Some(timing) = &pd_timing {
                            timing.record_first_decode_token(ctx.backend_type, &ctx.model);
                        }
                    }

                    let index = chunk.index();

                    // Both backends send delta token_ids, so accumulate for both
                    let completion_tokens = completion_tokens_map.entry(index).or_insert(0);
                    *completion_tokens += chunk.token_ids().len() as u32;
                    let current_completion_tokens = *completion_tokens;

                    // Decode tokens to text
                    let chunk_text = tokenizer
                        .decode(chunk.token_ids(), true)
                        .unwrap_or_default();

                    // Accumulate text for this index
                    let accumulated_text = accumulated_texts.entry(index).or_default();
                    accumulated_text.push_str(&chunk_text);

                    // Handle output logprobs by the stream's chunk semantics:
                    // cumulative chunks replace, delta chunks accumulate.
                    if let Some(ref output_logprobs) = chunk.output_logprobs() {
                        let converted = utils::convert_generate_output_logprobs(output_logprobs);
                        if chunk.chunk_semantics().is_delta() {
                            // Delta - extend existing logprobs
                            if let Some(v) = accumulated_output_logprobs
                                .entry(index)
                                .or_insert_with(|| Some(Vec::new()))
                                .as_mut()
                            {
                                v.extend(converted);
                            }
                        } else {
                            // Cumulative - replace
                            accumulated_output_logprobs.insert(index, Some(converted));
                        }
                    }

                    // Generate unique ID per index
                    let index_id = format!("{}-{}", ctx.request_id, index);

                    // Build streaming response chunk with accumulated logprobs
                    let current_output_logprobs = accumulated_output_logprobs
                        .get(&index)
                        .and_then(|o| o.as_ref());

                    let chunk_response = json!({
                        "text": accumulated_text.clone(),
                        "output_ids": chunk.token_ids(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": null,
                            "prompt_tokens": chunk.prompt_tokens(),
                            "weight_version": effective_weight_version(chunk.weight_version(), Some(ctx.weight_version.as_str())),
                            "input_token_logprobs": input_token_logprobs.as_ref(),
                            "output_token_logprobs": current_output_logprobs,
                            "completion_tokens": current_completion_tokens,
                            "cached_tokens": chunk.cached_tokens(),
                            "reasoning_tokens": chunk.reasoning_tokens()
                        },
                        "index": index
                    });

                    let sse_data = sse_encoder
                        .encode_data(&chunk_response)
                        .map_err(|e| format!("Failed to serialize generate chunk: {e}"))?;
                    tx.send(Ok(sse_data))
                        .await
                        .map_err(|_| "Failed to send chunk".to_string())?;
                }
                ProtoResponseVariant::Complete(complete) => {
                    let index = complete.index();
                    let accumulated_text =
                        accumulated_texts.get(&index).cloned().unwrap_or_default();

                    // Use accumulated count (we tracked deltas from both backends)
                    let completion_tokens = *completion_tokens_map.get(&index).unwrap_or(&0);

                    let final_output_logprobs = accumulated_output_logprobs
                        .get(&index)
                        .and_then(|o| o.as_ref());
                    let index_id = format!("{}-{}", ctx.request_id, index);
                    let e2e_latency = start_time.elapsed().as_secs_f64();
                    prompt_tokens = complete.prompt_tokens();
                    cached_tokens = cached_tokens.max(complete.cached_tokens());
                    reasoning_tokens_map.insert(index, complete.reasoning_tokens());
                    completed_indices.insert(index);

                    // Parse finish_reason
                    let finish_reason = utils::parse_finish_reason(
                        complete.finish_reason(),
                        complete.completion_tokens(),
                    );

                    // Send final chunk with finish_reason
                    let finish_response = json!({
                        "text": accumulated_text,
                        "output_ids": complete.output_ids()[complete.output_ids().len().saturating_sub(1)..].to_vec(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": finish_reason,
                            "prompt_tokens": complete.prompt_tokens(),
                            "weight_version": effective_weight_version(complete.weight_version(), Some(ctx.weight_version.as_str())),
                            "input_token_logprobs": input_token_logprobs.as_ref(),
                            "output_token_logprobs": final_output_logprobs,
                            "completion_tokens": completion_tokens,
                            "cached_tokens": complete.cached_tokens(),
                            "reasoning_tokens": complete.reasoning_tokens(),
                            "e2e_latency": e2e_latency
                        },
                        "index": index
                    });

                    let sse_data = sse_encoder
                        .encode_data(&finish_response)
                        .map_err(|e| format!("Failed to serialize generate finish: {e}"))?;
                    tx.send(Ok(sse_data))
                        .await
                        .map_err(|_| "Failed to send finish chunk".to_string())?;

                    // Continue to process all completions if n>1
                }
                ProtoResponseVariant::None => continue,
            }
        }

        // Mark stream as completed successfully to prevent abort on drop
        stream.mark_completed();

        // Record streaming metrics
        let total_completion: u32 = completion_tokens_map.values().sum();
        if let Some(handle) = reservation {
            if completed_indices.len() as u32 >= ctx.expected_choices {
                handle
                    .settle_success(UsageSettlement {
                        actual_input_tokens: prompt_tokens,
                        completion_tokens: total_completion,
                        cached_input_tokens: cached_tokens,
                        reasoning_tokens: reasoning_tokens_map.values().sum(),
                    })
                    .await;
            } else {
                handle.close_reserved_only().await;
            }
        }
        // The read ended with the client's leaving, not with the engine's end:
        // no terminal router metric for it, as the HTTP router has none for a
        // disconnect (neither a completed request nor a failure).
        if stream.client_left() {
            return Ok(());
        }
        Self::record_generate_metrics(
            start_time,
            first_token_time,
            total_completion,
            pd_timing.as_ref(),
            &ctx,
        );

        Ok(())
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Record streaming metrics for generate endpoint
    fn record_generate_metrics(
        start_time: Instant,
        first_token_time: Option<Instant>,
        total_completion: u32,
        pd_timing: Option<&context::PdTiming>,
        ctx: &GenerateStreamContext,
    ) {
        // PD mode: the stream task starts after the prefill side was dispatched
        // (sequential PD: after it completes, with the first decode chunk
        // already buffered), so anchor TTFT at prefill dispatch. Generation
        // duration shares the base or TPOT underflows.
        let stream_base = pd_timing
            .map(|timing| timing.prefill_start)
            .unwrap_or(start_time);
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: ctx.backend_type,
            model_id: &ctx.model,
            endpoint: metrics_labels::ENDPOINT_GENERATE,
            ttft: first_token_time.map(|t| t.duration_since(stream_base)),
            generation_duration: stream_base.elapsed(),
            input_tokens: None, // generate endpoint doesn't expose prompt tokens in streaming
            cached_tokens: None,
            output_tokens: total_completion as u64,
        });
    }

    /// Process a chunk of tokens through the stop decoder
    ///
    /// Decode errors are propagated instead of being treated as `Held`:
    /// swallowing them would drop the affected text while any configured
    /// stop sequence silently stops matching, letting the stream run on
    /// with missing output.
    fn process_chunk_tokens(
        stop_decoder: &mut StopSequenceDecoder,
        token_ids: &[u32],
    ) -> Result<(String, bool), String> {
        let mut chunk_text = String::new();

        for &token_id in token_ids {
            let step = stop_decoder
                .process_token_into(token_id, &mut chunk_text)
                .map_err(|e| format!("Stop decoder failed to process token {token_id}: {e}"))?;
            if step.stopped() {
                return Ok((chunk_text, true));
            }
        }
        Ok((chunk_text, false))
    }

    /// Helper: Process reasoning content in streaming mode
    /// `None` marks EOF and releases the parser's held text.
    /// Returns the normal text, the choice carrying the reasoning delta (if any),
    /// whether the parser is still inside reasoning, and how many of the chunk's
    /// `chunk_tokens` are reasoning tokens.
    #[expect(clippy::too_many_arguments)]
    fn process_reasoning_stream(
        &self,
        delta: Option<&str>,
        chunk_tokens: u32,
        index: u32,
        reasoning_parsers: &mut HashMap<u32, Box<dyn ReasoningParser>>,
        thinking_override: bool,
        think_in_prefill: bool,
        // Resolved once per request by the caller: re-resolving here could
        // disagree with the upfront availability check if the worker registry
        // changed mid-stream, turning the `expect` below into a panic.
        reasoning_parser_name: Option<&str>,
        model: &str,
    ) -> (String, Option<ChatStreamChoice>, bool, u32) {
        // Create fresh parser for this index (not pooled, to avoid state pollution)
        #[expect(
            clippy::expect_used,
            reason = "parser availability is checked upfront before streaming begins"
        )]
        reasoning_parsers.entry(index).or_insert_with(|| {
            let mut parser = utils::create_reasoning_parser(
                &self.reasoning_parser_factory,
                reasoning_parser_name,
                model,
            )
            .expect("Parser should be available - checked upfront");
            if thinking_override {
                parser.mark_reasoning_started();
                if think_in_prefill {
                    parser.mark_think_start_stripped();
                }
            }
            parser
        });

        if let Some(parser) = reasoning_parsers.get_mut(&index) {
            let (parse_result, was_in_reasoning, in_reasoning) = {
                let was_in_reasoning = parser.is_in_reasoning();
                let result = match delta {
                    Some(text) => parser.parse_reasoning_streaming_incremental(text),
                    None => parser.flush(),
                };
                let in_reasoning = parser.is_in_reasoning();
                (result, was_in_reasoning, in_reasoning)
            };

            match parse_result {
                Ok(ParserResult {
                    reasoning_text,
                    normal_text,
                }) => {
                    // The chunk's tokens are reasoning when its text came out
                    // as reasoning or it stayed inside the block (text held
                    // back); the delimiters opening or closing the block are
                    // not, like the engines' own counts. A chunk that carries
                    // both reasoning and content splits by text share.
                    let reasoning_tokens = if reasoning_text.is_empty() {
                        if was_in_reasoning && in_reasoning {
                            chunk_tokens
                        } else {
                            0
                        }
                    } else {
                        // The reasoning text's share of the chunk's text: the
                        // whole chunk when it is all reasoning, less when the
                        // chunk also carries a delimiter or content. An
                        // approximation: the stream has the chunk's token count
                        // and its decoded text, not the offset of each token,
                        // which the non-streaming count attributes by.
                        let chunk_len = delta.map_or(0, str::len).max(1) as f64;
                        let share = reasoning_text.len() as f64 / chunk_len;
                        (f64::from(chunk_tokens) * share.min(1.0)).round() as u32
                    };
                    let choice = if reasoning_text.is_empty() {
                        None
                    } else {
                        Some(assistant_choice(index, None, Some(reasoning_text), None))
                    };
                    return (normal_text, choice, in_reasoning, reasoning_tokens);
                }
                Err(e) => {
                    warn!("Reasoning parsing error: {}", e);
                }
            }
        }

        (delta.unwrap_or_default().to_string(), None, false, 0)
    }

    /// Helper: Process specific function case - emit tool call deltas with arguments
    fn process_specific_function_stream(
        delta: &str,
        index: u32,
        has_tool_calls: &mut HashMap<u32, bool>,
        tool_choice: Option<&ToolChoice>,
        model: &str,
        history_tool_calls_count: usize,
    ) -> Vec<ChatStreamChoice> {
        let mut chunks = Vec::new();

        if let Some(ToolChoice::Function { function, .. }) = tool_choice {
            let is_first_call = !has_tool_calls.contains_key(&index);

            if is_first_call {
                // First chunk: send name and id
                has_tool_calls.insert(index, true);

                let tool_call_id = utils::generate_tool_call_id(
                    model,
                    &function.name,
                    0,
                    history_tool_calls_count,
                );

                chunks.push(assistant_choice(
                    index,
                    None,
                    None,
                    Some(vec![ToolCallDelta {
                        index: 0,
                        id: Some(tool_call_id),
                        tool_type: Some("function".to_string()),
                        function: Some(FunctionCallDelta {
                            name: Some(function.name.clone()),
                            arguments: None,
                        }),
                    }]),
                ));
            }

            // Emit arguments delta
            if !delta.is_empty() {
                chunks.push(assistant_choice(
                    index,
                    None,
                    None,
                    Some(vec![ToolCallDelta {
                        index: 0,
                        id: None,
                        tool_type: None,
                        function: Some(FunctionCallDelta {
                            name: None,
                            arguments: Some(delta.to_string()),
                        }),
                    }]),
                ));
            }
        }

        chunks
    }

    /// Helper: Process tool calls in streaming mode
    #[expect(clippy::too_many_arguments)]
    async fn process_tool_calls_stream(
        &self,
        delta: &str,
        index: u32,
        tool_parsers: &mut HashMap<u32, Box<dyn ToolParser>>,
        has_tool_calls: &mut HashMap<u32, bool>,
        tools: &[Tool],
        // Resolved once per request by the caller (see process_reasoning_stream).
        tool_parser_name: Option<&str>,
        model: &str,
        history_tool_calls_count: usize,
        use_json_parser: bool,
        first_tool_call_only: bool,
    ) -> Vec<ChatStreamChoice> {
        let mut chunks = Vec::new();

        // Create fresh parser for this index (not pooled, to avoid state pollution)
        #[expect(
            clippy::expect_used,
            reason = "parser availability is checked upfront before streaming begins"
        )]
        tool_parsers.entry(index).or_insert_with(|| {
            if use_json_parser {
                utils::create_tool_parser(&self.tool_parser_factory, Some("json"), model)
                    .expect("JSON parser should be available")
            } else {
                utils::create_tool_parser(&self.tool_parser_factory, tool_parser_name, model)
                    .expect("Parser should be available - checked upfront")
            }
        });

        if let Some(parser) = tool_parsers.get_mut(&index) {
            match parser.parse_incremental(delta, tools).await {
                Ok(StreamingParseResult {
                    normal_text,
                    mut calls,
                }) => {
                    if first_tool_call_only {
                        utils::keep_first_tool_call_items(&mut calls);
                    }
                    // Emit normal text if present
                    if !normal_text.is_empty() {
                        chunks.push(assistant_choice(index, Some(normal_text), None, None));
                    }

                    // Emit tool call chunks
                    for tool_call_item in calls {
                        has_tool_calls.insert(index, true);

                        let tool_call_id = if let Some(ref name) = tool_call_item.name {
                            Some(utils::generate_tool_call_id(
                                model,
                                name,
                                tool_call_item.tool_index,
                                history_tool_calls_count,
                            ))
                        } else {
                            None
                        };

                        let tool_call_delta = ToolCallDelta {
                            index: tool_call_item.tool_index as u32,
                            id: tool_call_id,
                            tool_type: if tool_call_item.name.is_some() {
                                Some("function".to_string())
                            } else {
                                None
                            },
                            function: Some(FunctionCallDelta {
                                name: tool_call_item.name,
                                arguments: if tool_call_item.parameters.is_empty() {
                                    None
                                } else {
                                    Some(tool_call_item.parameters)
                                },
                            }),
                        };

                        chunks.push(assistant_choice(
                            index,
                            None,
                            None,
                            Some(vec![tool_call_delta]),
                        ));
                    }

                    return chunks;
                }
                Err(e) => {
                    error!("Tool call parsing error: {}", e);
                }
            }
        }

        chunks
    }

    /// Write one chat chunk carrying `choice` into `buffer`. The buffer goes
    /// to the channel as one frame, without a copy, by [`Self::flush_chunks`]
    /// once the engine has nothing more ready, or here once it holds
    /// [`CHUNK_FLUSH_BYTES`].
    async fn push_chunk<C: Serialize>(
        tx: &SseSender,
        buffer: &mut BytesMut,
        frame: &ChatChunkFrame,
        choice: &C,
        usage: Option<&Usage>,
        emit_usage_null: bool,
    ) -> Result<(), ()> {
        frame.write_chunk(buffer, choice, usage, emit_usage_null);
        if buffer.len() >= CHUNK_FLUSH_BYTES {
            tx.send(Ok(buffer.split().freeze())).await.map_err(|_| ())?;
        }
        Ok(())
    }

    /// Write the chunks Symphony's events make for the choice at `index`: the
    /// delta each event has, the chunk's logprobs on its first text delta. The
    /// `Finish` event writes nothing here: the finish chunk is the loop's, with
    /// the matched stop. The opening chunk carried the role, so the deltas do
    /// not repeat it, as the older path's do not (and are built without it, one
    /// allocation per delta saved). The step's events are consumed
    /// out of the buffer, which keeps its room for the next step: what an event
    /// owns moves into its chunk, and a text the event borrows from the delta
    /// is written from there, with no copy of its own.
    #[expect(clippy::too_many_arguments)]
    async fn push_events(
        tx: &SseSender,
        buffer: &mut BytesMut,
        frame: &ChatChunkFrame,
        index: u32,
        events: &mut Events<'_>,
        mut logprobs: Option<ChatLogProbs>,
        usage: Option<&Usage>,
        emit_usage_null: bool,
    ) -> Result<(), String> {
        for event in events.consume() {
            let Some(mut choice) = adapt::chat::into_delta_ref(index, event) else {
                continue;
            };
            if choice.finish_reason.is_some() {
                continue;
            }
            if choice.delta.content.is_some() || choice.delta.reasoning_content.is_some() {
                choice.logprobs = logprobs.take();
            }
            Self::push_chunk(tx, buffer, frame, &choice, usage, emit_usage_null)
                .await
                .map_err(|()| "Failed to send chunk".to_string())?;
        }
        Ok(())
    }

    /// Hand the buffered chunks, if any, to the channel as one frame.
    async fn flush_chunks(tx: &SseSender, buffer: &mut BytesMut) -> Result<(), String> {
        if buffer.is_empty() {
            return Ok(());
        }
        tx.send(Ok(buffer.split().freeze()))
            .await
            .map_err(|_| "Failed to send chunks".to_string())
    }

    /// The chunk that opens a choice: the assistant role with the public
    /// API's first delta ([`opening_choice`]).
    async fn send_role_chunk(
        tx: &SseSender,
        buffer: &mut BytesMut,
        frame: &ChatChunkFrame,
        index: u32,
        opens_tool_call: bool,
        usage: Option<&Usage>,
        emit_usage_null: bool,
    ) -> Result<(), String> {
        let opening = opening_choice(index, opens_tool_call);
        Self::push_chunk(tx, buffer, frame, &opening, usage, emit_usage_null)
            .await
            .map_err(|()| "Failed to send first chunk".to_string())
    }

    // =========================================================================
    // Messages API streaming support
    // =========================================================================

    /// Map a `MessageStreamEvent` variant to its SSE event type string.
    fn message_event_type_name(event: &MessageStreamEvent) -> &'static str {
        match event {
            MessageStreamEvent::MessageStart { .. } => "message_start",
            MessageStreamEvent::MessageDelta { .. } => "message_delta",
            MessageStreamEvent::MessageStop => "message_stop",
            MessageStreamEvent::ContentBlockStart { .. } => "content_block_start",
            MessageStreamEvent::ContentBlockDelta { .. } => "content_block_delta",
            MessageStreamEvent::ContentBlockStop { .. } => "content_block_stop",
            MessageStreamEvent::Ping => "ping",
            MessageStreamEvent::Error { .. } => "error",
        }
    }

    /// Format a `MessageStreamEvent` as Anthropic SSE into a reusable buffer.
    ///
    /// Writes `event: {type}\ndata: {json}\n\n` into `buffer`, avoiding per-event
    /// allocations by reusing the same buffer across multiple events.
    #[inline]
    pub(crate) fn format_messages_sse_into(
        buffer: &mut Vec<u8>,
        event: &MessageStreamEvent,
    ) -> Result<(), String> {
        buffer.clear();
        let event_type = Self::message_event_type_name(event);
        buffer.extend_from_slice(b"event: ");
        buffer.extend_from_slice(event_type.as_bytes());
        buffer.extend_from_slice(b"\ndata: ");
        serde_json::to_writer(&mut *buffer, event)
            .map_err(|e| format!("Failed to serialize messages event: {e}"))?;
        buffer.extend_from_slice(b"\n\n");
        Ok(())
    }

    /// Ends Symphony's reading of the message: `flushed`, the text the decoder released last,
    /// then the engine's `finish`; the events go through the machine ([`Self::send_symphony_blocks`],
    /// `thinking_text` as there) and the reasoning tokens they cost are added to `thinking_tokens`,
    /// under `who` for the telemetry of the output's end. The stream's `delta` and `spare` buffer
    /// are read as a step's are, so events that wait in the buffer (the prompt's, when the
    /// message ends before any chunk) go out in their order.
    #[expect(clippy::too_many_arguments)]
    async fn end_symphony_message(
        tx: &SseSender,
        buffer: &mut Vec<u8>,
        (reader, machine): (&mut Reader, &mut adapt::messages::Stream),
        (delta, spare): (&mut Delta, &mut Events<'static>),
        flushed: Option<String>,
        finish: EngineFinish,
        who: Who<'_>,
        thinking_text: bool,
        thinking_tokens: &mut u32,
    ) -> Result<Option<messages::StopReason>, utils::StreamFailure> {
        let mut events: Events<'_> = std::mem::take(spare);
        if let Some(text) = flushed {
            reader.tail(text, delta, &mut events)?;
        }
        reader.end(finish, who, &mut events)?;
        *thinking_tokens += reasoning_text_tokens(events.as_slice());
        let stop =
            Self::send_symphony_blocks(tx, buffer, machine, events.as_slice(), thinking_text).await;
        *spare = events.recycle();
        stop
    }

    /// Feeds Symphony's `events` to the content-block `machine` and sends the stream events it
    /// gives, in order, the `thinking_delta`s only when `thinking_text` (a `thinking.display` of
    /// `omitted` keeps the thinking block, its start and its stop, and no text); the stop reason
    /// when a `Finish` was among the events.
    async fn send_symphony_blocks(
        tx: &SseSender,
        buffer: &mut Vec<u8>,
        machine: &mut adapt::messages::Stream,
        events: &[Event<'_>],
        thinking_text: bool,
    ) -> Result<Option<messages::StopReason>, utils::StreamFailure> {
        let mut stop = None;
        let mut blocks = Vec::new();
        for event in events {
            stop = machine.feed(event, &mut blocks).or(stop);
        }
        for block in &blocks {
            let thinking_delta = matches!(
                block,
                MessageStreamEvent::ContentBlockDelta {
                    delta: ContentBlockDelta::ThinkingDelta { .. },
                    ..
                }
            );
            if thinking_delta && !thinking_text {
                continue;
            }
            Self::send_messages_event(tx, buffer, block).await?;
        }
        Ok(stop)
    }

    /// Send a `MessageStreamEvent` through the SSE channel using a reusable buffer.
    async fn send_messages_event(
        tx: &SseSender,
        buffer: &mut Vec<u8>,
        event: &MessageStreamEvent,
    ) -> Result<(), String> {
        Self::format_messages_sse_into(buffer, event)?;
        tx.send(Ok(Bytes::from(buffer.clone())))
            .await
            .map_err(|_| "Client disconnected".to_string())
    }

    /// Stop the open content block, if any, so the next block gets the next
    /// index: reasoning, text and tool calls can alternate.
    async fn stop_open_block(
        tx: &SseSender,
        buffer: &mut Vec<u8>,
        index: &mut u32,
        open: [&mut bool; 3],
    ) -> Result<(), String> {
        if open
            .into_iter()
            .fold(false, |any, open| std::mem::take(open) | any)
        {
            let stop = MessageStreamEvent::ContentBlockStop { index: *index };
            Self::send_messages_event(tx, buffer, &stop).await?;
            *index += 1;
        }
        Ok(())
    }

    /// Send tool call arguments to the open `tool_use` block. Reasoning can
    /// stop that block in the middle of a call, and the arguments after it
    /// have no block to go to, so they are dropped.
    async fn send_tool_arguments(
        tx: &SseSender,
        buffer: &mut Vec<u8>,
        index: u32,
        tool_block_open: bool,
        partial_json: String,
    ) -> Result<(), String> {
        if partial_json.is_empty() {
            return Ok(());
        }
        if !tool_block_open {
            debug!("Dropping tool arguments without an open tool_use block");
            return Ok(());
        }
        let delta = MessageStreamEvent::ContentBlockDelta {
            index,
            delta: ContentBlockDelta::InputJsonDelta { partial_json },
        };
        Self::send_messages_event(tx, buffer, &delta).await
    }

    /// Process reasoning content in Messages streaming mode (n=1 only).
    ///
    /// Returns `(normal_text, reasoning_text, in_reasoning)`.
    /// `None` marks EOF and releases the parser's held text.
    /// Caller handles SSE event emission.
    fn process_messages_reasoning(
        &self,
        delta: Option<&str>,
        reasoning_parser: &mut Option<Box<dyn ReasoningParser>>,
        thinking_override: bool,
        think_in_prefill: bool,
        // Resolved once per request by the caller (see process_reasoning_stream).
        reasoning_parser_name: Option<&str>,
        model: &str,
    ) -> (String, String, bool) {
        // Lazily create parser
        if reasoning_parser.is_none() {
            if let Some(mut parser) = utils::create_reasoning_parser(
                &self.reasoning_parser_factory,
                reasoning_parser_name,
                model,
            ) {
                if thinking_override {
                    parser.mark_reasoning_started();
                    if think_in_prefill {
                        parser.mark_think_start_stripped();
                    }
                }
                *reasoning_parser = Some(parser);
            }
        }

        if let Some(parser) = reasoning_parser {
            let (parse_result, in_reasoning) = {
                let result = match delta {
                    Some(text) => parser.parse_reasoning_streaming_incremental(text),
                    None => parser.flush(),
                };
                let in_reasoning = parser.is_in_reasoning();
                (result, in_reasoning)
            };
            match parse_result {
                Ok(ParserResult {
                    reasoning_text,
                    normal_text,
                }) => {
                    return (normal_text, reasoning_text, in_reasoning);
                }
                Err(e) => {
                    warn!("Reasoning parsing error in messages streaming: {}", e);
                }
            }
        }

        (delta.unwrap_or_default().to_string(), String::new(), false)
    }

    /// Process streaming Messages API response and return SSE response.
    ///
    /// Parallel to [`Self::process_streaming_response`] for chat, but emits
    /// Anthropic SSE format (`event: {type}\ndata: {json}\n\n`). The
    /// `load_guards` (and the reservation's safety net) ride in the response
    /// body, released when it is dropped.
    #[expect(clippy::too_many_arguments)]
    pub async fn process_messages_streaming_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        mut messages_request: MessagesResponseSpec,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        skip_special_tokens: bool,
        reservation: Option<Arc<SharedReservationHandle>>,
        load_guards: Option<context::LoadGuards>,
        source: utils::StreamSource,
    ) -> Response {
        let source = source.with_backend(self.backend_type);
        let guards = body_guards(load_guards, reservation.as_ref());
        // The stream's decoder takes the spec's stop sequences; nothing reads
        // them from the spec after this.
        let stop_params = (
            messages_request.stop_sequences.take(),
            None::<Vec<u32>>, // No stop_token_ids in Messages API
            skip_special_tokens,
            false, // no_stop_trim
            false, // ignore_eos — not available in Messages API
        );

        let (tx, rx) = sse_channel();
        // Read for this client: the moment the response body is dropped, the
        // request is aborted on the engine (see `ProtoStream::until_client_gone`).
        let execution_result = execution_result.until_client_gone(&tx, &dispatch.request_id);

        match execution_result {
            context::ExecutionResult::Single { stream } => {
                let processor = self.clone();
                let dispatch_clone = dispatch.clone();
                let tokenizer_clone = tokenizer.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "streaming task is fire-and-forget; client disconnect terminates it"
                )]
                tokio::spawn(async move {
                    let result = processor
                        .process_messages_streaming_chunks(
                            stream,
                            dispatch_clone,
                            tokenizer_clone,
                            stop_params,
                            messages_request,
                            &tx,
                            reservation,
                        )
                        .await;

                    if let Err(failure) = result {
                        source.account(&failure);
                        let error_event = MessageStreamEvent::Error {
                            error: messages::ErrorResponse {
                                error_type: failure.messages_error_type().to_string(),
                                message: failure.to_string(),
                            },
                        };
                        let mut buf = Vec::with_capacity(256);
                        let _ = Self::send_messages_event(&tx, &mut buf, &error_event).await;
                    }
                    // No data: [DONE] — Anthropic uses message_stop instead
                });
            }
            context::ExecutionResult::PrefillDecode {
                // TODO(#1781 follow-up): thread pd_timing for honest PD TTFT
                prefill,
                decode,
                prefill_guards,
                ..
            } => {
                let processor = self.clone();
                let tokenizer_clone = tokenizer.clone();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "streaming task is fire-and-forget; client disconnect terminates it"
                )]
                tokio::spawn(async move {
                    let result = processor
                        .process_prefill_decode_messages_streaming_chunks(
                            prefill,
                            *decode,
                            dispatch,
                            tokenizer_clone,
                            stop_params,
                            messages_request,
                            &tx,
                            prefill_guards,
                            reservation,
                        )
                        .await;

                    if let Err(failure) = result {
                        // A decode side the engine failed without a reason is read against
                        // the pair's bootstrap port first, as the chat event is (see
                        // `StreamSource::explain`), so the event names a dead rendezvous.
                        let explained = source.explain(&failure).await;
                        let failure = explained.as_ref().unwrap_or(&failure);
                        source.account(failure);
                        let error_event = MessageStreamEvent::Error {
                            error: messages::ErrorResponse {
                                error_type: failure.messages_error_type().to_string(),
                                message: failure.to_string(),
                            },
                        };
                        let mut buf = Vec::with_capacity(256);
                        let _ = Self::send_messages_event(&tx, &mut buf, &error_event).await;
                    }
                });
            }
            context::ExecutionResult::Embedding { .. } => {
                let error_event = MessageStreamEvent::Error {
                    error: messages::ErrorResponse {
                        error_type: "invalid_request_error".to_string(),
                        message: "Embeddings not supported for Messages API".to_string(),
                    },
                };
                let mut buf = Vec::with_capacity(256);
                let _ = Self::send_messages_event(&tx, &mut buf, &error_event).await;
            }
            // Batch results exist only on the completions pipeline.
            context::ExecutionResult::Batch { .. } => {
                let error_event = MessageStreamEvent::Error {
                    error: messages::ErrorResponse {
                        error_type: "invalid_request_error".to_string(),
                        message: "Batched results not supported for Messages API".to_string(),
                    },
                };
                let mut buf = Vec::with_capacity(256);
                let _ = Self::send_messages_event(&tx, &mut buf, &error_event).await;
            }
        }

        build_sse_response_guarded(rx, guards)
    }

    /// Process Messages API streaming chunks from a single stream.
    ///
    /// Implements the Anthropic streaming protocol with content block
    /// state tracking. Always n=1 (no per-index HashMap). The phases are
    /// the methods of [`MessagesStream`], run in order.
    #[expect(clippy::too_many_arguments)]
    pub async fn process_messages_streaming_chunks(
        &self,
        mut grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        original_request: MessagesResponseSpec,
        tx: &SseSender,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), utils::StreamFailure> {
        let mut stream = MessagesStream::open(
            self,
            &dispatch,
            tokenizer,
            stop_params,
            &original_request,
            tx,
        )?;
        stream.send_message_start().await?;
        stream.read_engine(&mut grpc_stream).await?;
        // The stream's verdict, before the engine side is released.
        let client_left = grpc_stream.client_left();
        stream.release_engine(grpc_stream);
        // As for chat: nothing for a client that left, the settlement with
        // the actual usage before a failed send is returned.
        let terminal = if client_left {
            Ok(())
        } else {
            async {
                stream.flush_parsers().await?;
                stream.close_open_blocks().await?;
                stream.send_message_delta().await?;
                stream.send_message_stop().await
            }
            .await
        };

        stream
            .settle_and_record(reservation, client_left || terminal.is_err())
            .await;
        terminal
    }

    /// Process prefill/decode streaming chunks for Messages API (PD mode).
    ///
    /// Consumes prefill stream then delegates to
    /// [`Self::process_messages_streaming_chunks`] with the decode stream.
    #[expect(clippy::too_many_arguments)]
    pub async fn process_prefill_decode_messages_streaming_chunks(
        &self,
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        original_request: MessagesResponseSpec,
        tx: &SseSender,
        prefill_guards: Vec<Option<PrefillLoadGuard>>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), utils::StreamFailure> {
        // Consume prefill stream (Messages API does not expose prompt logprobs);
        // the client's prompt-side usage is the prefill side's.
        let prefill_cached_tokens =
            drain_prefill(&mut prefill_stream, prefill_guards, true, |_| {})
                .await
                .map_err(utils::StreamFailure::prefill)?;
        let decode_stream = decode_stream.decode_side(prefill_cached_tokens);

        let result = self
            .process_messages_streaming_chunks(
                decode_stream,
                dispatch,
                tokenizer,
                stop_params,
                original_request,
                tx,
                reservation,
            )
            .await;

        if result.is_ok() {
            prefill_stream.mark_completed();
        }

        result
    }

    // =========================================================================
    // Completions API streaming support
    // =========================================================================

    /// Entry point for `/v1/completions` streaming.
    ///
    /// Batched requests fan out into one stream unit per prompt; all units
    /// write typed SSE events (with prompt-major index offsets) into one
    /// channel, and the coordinator emits the single aggregated usage chunk,
    /// metrics record, and `[DONE]`. The `load_guards` (and the reservation's
    /// safety net) ride in the response body, released when it is dropped.
    #[expect(clippy::too_many_arguments)]
    pub async fn process_completion_streaming_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        completion_request: CompletionResponseSpec,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        reservation: Option<Arc<SharedReservationHandle>>,
        load_guards: Option<context::LoadGuards>,
        source: utils::StreamSource,
    ) -> Response {
        let source = source.with_backend(self.backend_type);
        let guards = body_guards(load_guards, reservation.as_ref());
        let (tx, rx) = sse_channel();
        // Read for this client: the moment the response body is dropped, the
        // request is aborted on the engine (see `ProtoStream::until_client_gone`).
        let execution_result = execution_result.until_client_gone(&tx, &dispatch.request_id);

        let units = match Self::completion_stream_units(execution_result) {
            Ok(units) => units,
            Err(message) => {
                utils::send_error_sse(&tx, message, "invalid_request_error").await;
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
                return build_sse_response_guarded(rx, guards);
            }
        };

        let processor = self;
        #[expect(
            clippy::disallowed_methods,
            reason = "streaming task is fire-and-forget; client disconnect terminates it"
        )]
        tokio::spawn(async move {
            let start_time = Instant::now();
            let choices_per_prompt = completion_request.choices_per_prompt;
            let echo = completion_request.echo;
            let include_usage = completion_request.include_usage;

            // Fail-fast: the first stream error cancels the remaining units
            // (their streams abort on drop) and fails the whole request.
            let completion_request = &completion_request;
            let outcomes =
                try_join_all(units.into_iter().enumerate().map(|(prompt_index, unit)| {
                    let stop_params = (
                        completion_request.stop.clone(),
                        completion_request.stop_token_ids.clone(),
                        completion_request.skip_special_tokens,
                        completion_request.no_stop_trim,
                        completion_request.ignore_eos,
                    );
                    let dispatch = dispatch.clone();
                    let tokenizer = tokenizer.clone();
                    let prompt_text = if echo {
                        match completion_request.prompt_texts.get(prompt_index) {
                            Some(text) => text.as_str(),
                            None => {
                                warn!(
                                    prompt_index,
                                    prompt_texts_len = completion_request.prompt_texts.len(),
                                    "echo requested but no prompt text for this prompt index"
                                );
                                ""
                            }
                        }
                    } else {
                        ""
                    };
                    let index_offset = prompt_index as u32 * choices_per_prompt;
                    let processor = &processor;
                    let tx = &tx;
                    async move {
                        match unit {
                            CompletionStreamUnit::Single(stream) => {
                                processor
                                    .process_completion_streaming_chunks(
                                        stream,
                                        dispatch,
                                        tokenizer,
                                        stop_params,
                                        completion_request,
                                        prompt_text,
                                        index_offset,
                                        tx,
                                    )
                                    .await
                            }
                            CompletionStreamUnit::PrefillDecode {
                                prefill,
                                decode,
                                prefill_guards,
                            } => {
                                processor
                                    .process_prefill_decode_completion_streaming_chunks(
                                        prefill,
                                        *decode,
                                        dispatch,
                                        tokenizer,
                                        stop_params,
                                        completion_request,
                                        prompt_text,
                                        index_offset,
                                        tx,
                                        prefill_guards,
                                    )
                                    .await
                            }
                        }
                    }
                }))
                .await;

            match outcomes {
                Err(failure) => {
                    utils::send_stream_failure(&tx, &failure, &source).await;
                }
                Ok(outcomes) => {
                    let mut total_prompt = 0u32;
                    let mut total_cached = 0u32;
                    let mut total_reasoning = 0u32;
                    let mut total_spec_accepted = 0u32;
                    let mut total_spec_drafted = 0u32;
                    let mut total_completion = 0u32;
                    let mut first_token_time: Option<Instant> = None;
                    let mut all_saw_complete = true;
                    for outcome in outcomes {
                        total_prompt += outcome.prompt_tokens;
                        total_cached += outcome.cached_tokens;
                        total_reasoning += outcome.reasoning_tokens;
                        total_spec_accepted += outcome.spec_accepted_tokens;
                        total_spec_drafted += outcome.spec_draft_tokens;
                        total_completion += outcome.completion_tokens;
                        all_saw_complete &= outcome.saw_complete;
                        first_token_time = match (first_token_time, outcome.first_token_time) {
                            (Some(current), Some(candidate)) => Some(current.min(candidate)),
                            (current, candidate) => current.or(candidate),
                        };
                    }

                    // All units succeeded (fail-fast `try_join_all` above), so
                    // this is the one settle point for every mode (Single/PD/Batch).
                    // But a unit that never saw a `Complete` message has no
                    // authoritative usage to contribute -- settling the
                    // aggregate anyway would understate total_prompt/total_completion
                    // and incorrectly refund part of the reservation.
                    if let Some(handle) = &reservation {
                        if all_saw_complete {
                            handle
                                .settle_success(UsageSettlement {
                                    actual_input_tokens: total_prompt,
                                    completion_tokens: total_completion,
                                    cached_input_tokens: total_cached,
                                    reasoning_tokens: total_reasoning,
                                })
                                .await;
                        } else {
                            handle.close_reserved_only().await;
                        }
                    }

                    if include_usage {
                        let usage_chunk = CompletionStreamResponse {
                            id: dispatch.request_id.clone(),
                            object: "text_completion".to_string(),
                            created: dispatch.created,
                            choices: vec![],
                            model: dispatch.model.clone(),
                            system_fingerprint: dispatch.weight_version.clone(),
                            // The usage object the non-streamed answer carries.
                            usage: Some(
                                TokenCounts {
                                    prompt: total_prompt,
                                    completion: total_completion,
                                    cached: total_cached,
                                    reasoning: total_reasoning,
                                    spec_accepted: total_spec_accepted,
                                    spec_drafted: total_spec_drafted,
                                }
                                .usage(),
                            ),
                        };
                        let mut sse_buffer = Vec::with_capacity(256);
                        Self::format_completion_sse_into(&mut sse_buffer, &usage_chunk);
                        let _ = tx.send(Ok(Bytes::from(sse_buffer))).await;
                    }
                    // The units' streams are read for this client (the sender
                    // the watch reads): a sender closed by now says the client
                    // left and ended the read, which is no completed
                    // generation and gets no terminal router metric.
                    if tx.is_closed() {
                        return;
                    }
                    Metrics::record_streaming_metrics(StreamingMetricsParams {
                        router_type: metrics_labels::ROUTER_GRPC,
                        backend_type: processor.backend_type,
                        model_id: &dispatch.model,
                        endpoint: metrics_labels::ENDPOINT_COMPLETIONS,
                        ttft: first_token_time.map(|t| t.duration_since(start_time)),
                        generation_duration: start_time.elapsed(),
                        input_tokens: Some(total_prompt as u64),
                        cached_tokens: Some(u64::from(total_cached)),
                        output_tokens: total_completion as u64,
                    });
                }
            }

            let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
        });

        build_sse_response_guarded(rx, guards)
    }

    /// Split an execution result into per-prompt stream units, in prompt order.
    fn completion_stream_units(
        execution_result: context::ExecutionResult,
    ) -> Result<Vec<CompletionStreamUnit>, &'static str> {
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                Ok(vec![CompletionStreamUnit::Single(stream)])
            }
            context::ExecutionResult::PrefillDecode {
                // TODO(#1781 follow-up): thread pd_timing for honest PD TTFT
                prefill,
                decode,
                prefill_guards,
                ..
            } => Ok(vec![CompletionStreamUnit::PrefillDecode {
                prefill,
                decode,
                prefill_guards,
            }]),
            context::ExecutionResult::Batch { results } => results
                .into_iter()
                .map(|result| match result {
                    context::ExecutionResult::Single { stream } => {
                        Ok(CompletionStreamUnit::Single(stream))
                    }
                    context::ExecutionResult::PrefillDecode {
                        prefill,
                        decode,
                        prefill_guards,
                        ..
                    } => Ok(CompletionStreamUnit::PrefillDecode {
                        prefill,
                        decode,
                        prefill_guards,
                    }),
                    _ => Err("Nested batch or embedding result in completion streaming"),
                })
                .collect(),
            context::ExecutionResult::Embedding { .. } => {
                Err("Embeddings not supported in streaming mode")
            }
        }
    }

    /// Process completion streaming chunks from a single stream.
    ///
    /// Decodes tokens through stop decoder, handles `echo` (first chunk) and
    /// `suffix` (after final chunk), and emits `CompletionStreamResponse` SSE
    /// events with `index_offset`-shifted choice indices. Supports n>1 via
    /// per-index tracking. Returns per-stream totals for the coordinator's
    /// usage chunk and metrics record. The phases are the methods of
    /// [`CompletionStream`], run in order.
    #[expect(clippy::too_many_arguments)]
    async fn process_completion_streaming_chunks(
        &self,
        mut grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        completion_request: &CompletionResponseSpec,
        prompt_text: &str,
        index_offset: u32,
        tx: &SseSender,
    ) -> Result<CompletionStreamOutcome, utils::StreamFailure> {
        let mut stream = CompletionStream::open(
            &dispatch,
            tokenizer,
            stop_params,
            completion_request,
            prompt_text,
            index_offset,
            tx,
        );
        stream.read_engine(&mut grpc_stream).await?;
        // The stream's verdict, before the engine side is released: a client
        // that left gets no finish of its open choices (the held tail would go
        // to a closed sender), as on the chat and Messages paths.
        let client_left = grpc_stream.client_left();
        stream.release_engine(grpc_stream, self.backend_type);
        if !client_left {
            stream.finish_open_choices().await?;
        }

        Ok(stream.outcome())
    }

    /// PD prefill/decode variant: consume prefill stream, then delegate decode
    /// stream to [`Self::process_completion_streaming_chunks`].
    #[expect(clippy::too_many_arguments)]
    async fn process_prefill_decode_completion_streaming_chunks(
        &self,
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        original_request: &CompletionResponseSpec,
        prompt_text: &str,
        index_offset: u32,
        tx: &SseSender,
        prefill_guards: Vec<Option<PrefillLoadGuard>>,
    ) -> Result<CompletionStreamOutcome, utils::StreamFailure> {
        // The client's prompt-side usage is the prefill side's.
        let prefill_cached_tokens =
            drain_prefill(&mut prefill_stream, prefill_guards, true, |_| {})
                .await
                .map_err(utils::StreamFailure::prefill)?;
        let decode_stream = decode_stream.decode_side(prefill_cached_tokens);

        let result = self
            .process_completion_streaming_chunks(
                decode_stream,
                dispatch,
                tokenizer,
                stop_params,
                original_request,
                prompt_text,
                index_offset,
                tx,
            )
            .await;

        // The prefill stream is marked completed after the decode's read ended
        // well: a decode that failed leaves both streams to abort on drop; a
        // client that left ends the read with `Ok` too, its decode stream
        // aborting the request on drop, the prefill's own request complete.
        if result.is_ok() {
            prefill_stream.mark_completed();
        }

        result
    }

    /// Format a `CompletionStreamResponse` into the SSE buffer.
    #[inline]
    fn format_completion_sse_into(buffer: &mut Vec<u8>, chunk: &CompletionStreamResponse) {
        buffer.clear();
        buffer.extend_from_slice(b"data: ");
        if let Err(e) = serde_json::to_writer(&mut *buffer, chunk) {
            error!("Failed to serialize completion SSE chunk: {}", e);
            buffer.clear();
            buffer.extend_from_slice(b"data: ");
            let error_msg = json!({
                "error": {
                    "message": format!("Failed to serialize completion chunk: {e}"),
                    "type": "internal_error"
                }
            })
            .to_string();
            buffer.extend_from_slice(error_msg.as_bytes());
        }
        buffer.extend_from_slice(b"\n\n");
    }

    /// Write one completion chunk carrying `choice` and hand it to the
    /// channel as its own frame, without a copy.
    async fn send_completion_chunk(
        tx: &SseSender,
        buffer: &mut BytesMut,
        frame: &CompletionChunkFrame,
        choice: &CompletionStreamChoice,
    ) -> Result<(), String> {
        frame.write_chunk(buffer, choice);
        tx.send(Ok(buffer.split().freeze()))
            .await
            .map_err(|_| "Channel closed".to_string())
    }

    /// Skeleton usage for the `message_start` event. Cache counters are
    /// integer zeros, never null: the Anthropic wire contract has
    /// always-present cache counters and clients do arithmetic on them; the
    /// cache-creation breakdown, service tier and inference geography are
    /// written on the skeleton too, as the public API writes them.
    fn initial_messages_usage() -> messages::Usage {
        messages::Usage::from_counts(0, 0)
    }

    /// Usage for the final `message_delta` event. `authoritative_input` is the
    /// prompt count only when a `Complete` was seen; a clean EOF without one
    /// must serialize `input_tokens: null` rather than claim a zero-token
    /// prompt. Cache counters follow the same integer-not-null contract as
    /// [`Self::initial_messages_usage`]; `thinking_tokens` is reported when
    /// the turn ran with thinking.
    fn final_messages_delta_usage(
        output_tokens: u32,
        authoritative_input: Option<u32>,
        thinking_tokens: Option<u32>,
    ) -> MessageDeltaUsage {
        MessageDeltaUsage {
            output_tokens,
            input_tokens: authoritative_input,
            cache_creation_input_tokens: Some(0),
            cache_read_input_tokens: Some(0),
            server_tool_use: None,
            output_tokens_details: thinking_tokens
                .map(|thinking_tokens| messages::OutputTokensDetails { thinking_tokens }),
        }
    }
}

/// Add the provider's null placeholder without allocating a JSON value per
/// token or changing the shared Chat response type. A populated usage field
/// is serialized only by `chunk`, so there is never a duplicate JSON key.
#[derive(Serialize)]
struct ChatChunkWithUsage<'a> {
    #[serde(flatten)]
    chunk: &'a ChatCompletionStreamResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<()>,
}

impl<'a> ChatChunkWithUsage<'a> {
    fn new(chunk: &'a ChatCompletionStreamResponse, emit_usage_null: bool) -> Self {
        Self {
            chunk,
            usage: (emit_usage_null && chunk.usage.is_none()).then_some(()),
        }
    }
}

/// Text a choice's decoder released for one engine message, on its way to
/// the parsers and the client: the choice index, the text, the logprobs that
/// ride with it, and the tokens it stands for (held ones included).
type PendingText = (u32, String, Option<ChatLogProbs>, u32);

/// What one pass of the loop over the engine's end yields.
enum StreamEnd {
    /// Every choice's decoder and parsers have had their turn; the loop ends.
    Finished,
    /// This pass released nothing to emit.
    Nothing,
    /// Text to emit, and whether it is the parsers' EOF turn for its choice.
    Text(PendingText, bool),
}

/// The failure of a choice's decoder: where Symphony reads the choice and its output is not over,
/// the reader's failed output, counted as such; the error's text either way. A reader whose output
/// ended already has its outcome: a decoder finished a second time (the Messages stream's end after
/// a `Complete`) counts nothing more.
fn decoding_failed(
    reader: Option<&mut Reader>,
    error: impl std::fmt::Display,
) -> utils::StreamFailure {
    match reader.filter(|reader| !reader.ended()) {
        Some(reader) => reader.failed(error.to_string()).into(),
        None => utils::StreamFailure::from(error.to_string()),
    }
}

/// One chat completion's SSE stream being read off the engine: what the
/// spec and the dispatch decide for the whole stream, the per-choice
/// trackers, decoders, parsers and readers the loop keeps, and the frame,
/// encoder and buffer the chunks leave through. Each phase of the stream is
/// a method; `StreamingProcessor::process_streaming_chunks_inner` runs them
/// in order.
struct ChatStream<'a> {
    processor: &'a StreamingProcessor,
    spec: &'a ChatResponseSpec,
    /// Dispatch metadata, for consistent response fields.
    dispatch: &'a context::DispatchMetadata,
    tx: &'a SseSender,
    tokenizer: Arc<dyn Tokenizer>,
    stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
    pd_timing: Option<context::PdTiming>,

    // Metrics timing
    start_time: Instant,
    first_token_time: Option<Instant>,

    /// DeepSeek always reports aggregate usage on the final finish chunk;
    /// `include_usage` controls null placeholders on earlier chunks only.
    deepseek_usage: bool,
    include_usage: bool,
    emit_usage_null: bool,
    continuous_usage: Option<ChatStreamUsage>,

    /// Per-request effective parser names (model-card override -> configured),
    /// and whether each parser is there (checked once upfront, so the warning
    /// is logged once per request).
    reasoning_parser_name: Option<String>,
    tool_parser_name: Option<String>,
    reasoning_parser_available: bool,
    tool_parser_available: bool,
    /// Request building decided whether the rendered prompt ends inside the
    /// model's thinking block; the parsers then start in reasoning mode.
    thinking_override: bool,
    think_in_prefill: bool,
    /// Whether a JSON schema constraint was used (specific function or
    /// required mode), and whether it is the specific-function case (the
    /// model generates the parameters only, no name field; native formats
    /// include framing tokens the parser must handle).
    used_json_schema: bool,
    is_specific_function: bool,
    /// The first delta's shape: a forced tool call opens as the public API
    /// opens a tool-call stream; anything else as a text stream.
    opens_tool_call: bool,

    // Per-choice state (per index, for n>1 support)
    is_firsts: HashMap<u32, bool>,
    finish_reasons: HashMap<u32, String>,
    matched_stops: HashMap<u32, Option<Value>>,
    /// Indices whose local stop decoder fired: their finish reason is pinned
    /// to "stop" and later engine output for the index is ignored.
    stopped_indices: HashSet<u32>,
    /// Indices whose engine `Complete` came. A choice is closed by its
    /// Complete or by its local stop; once every expected choice is closed
    /// while a stopped one streams on without its Complete, the engine has
    /// outlived the client's stream (see [`Self::engine_outlived_now`]).
    completed_indices: HashSet<u32>,
    /// Stopped indices the engine streamed a chunk for after their stop: it
    /// is generating past the stop, which only a stop it cannot see does (an
    /// engine that saw the stop ends the choice on that step and sends its
    /// Complete next).
    tailed_indices: HashSet<u32>,
    /// The engine was streaming past a stop when every choice had closed:
    /// the loop stopped polling it, and the driver aborts it instead of
    /// marking it completed (see [`Self::release_engine`]).
    engine_outlived: bool,
    /// Tokens the engine streamed for choices after their local stop, read
    /// before the abort: the tail generated for nobody.
    tail_tokens: u64,
    prompt_tokens: HashMap<u32, u32>,
    completion_tokens: CompletionTokenTracker,
    cached_tokens: HashMap<u32, u32>,
    reasoning_tokens: HashMap<u32, u32>,
    /// Tokens the gateway's reasoning parser routed to reasoning, and tokens
    /// of chunks that decoded to no text yet (counted with the chunk that
    /// releases their text).
    parsed_reasoning_tokens: HashMap<u32, u32>,
    held_tokens: HashMap<u32, u32>,
    spec_accepted: HashMap<u32, u32>,
    spec_drafted: HashMap<u32, u32>,
    /// Parser state (lazy initialization per index); the parsers are this
    /// stream's own, nothing else reaches them.
    reasoning_parsers: HashMap<u32, Box<dyn ReasoningParser>>,
    tool_parsers: HashMap<u32, Box<dyn ToolParser>>,
    has_tool_calls: HashMap<u32, bool>,
    /// Per-index stop decoders (each index needs its own state for n>1 support).
    stop_decoders: HashMap<u32, StopSequenceDecoder>,
    /// Symphony reads this response when request building assigned it a table
    /// (see `regular::symphony`): one reader per choice in place of both
    /// parsers, seeded with the prompt it replays. The role chunk, the stop
    /// decoder, the finish chunks and the usage are the same; only the
    /// reading of the tokens differs.
    readers: HashMap<u32, Reader>,
    /// The step's delta and the stream's event buffer, when Symphony reads
    /// the response: a step's events fill the buffer and borrow the delta's
    /// text, and the buffer is recycled once the step's chunks are out (the
    /// crate's `docs/engine.md`, the shape of a step's events). Taken out
    /// for the step and put back after it; a buffer lost to an error is a
    /// stream that ended.
    delta: Delta,
    spare: Events<'static>,
    /// After the engine's end: the (index, parser-EOF) turns still to take,
    /// one per pass of the loop.
    final_indices: Option<Vec<(u32, bool)>>,

    /// Reusable SSE encoder for the post-loop flush / tool / finish / usage
    /// chunks.
    sse_encoder: SseEncoder,
    /// The per-chunk SSE frames: the invariant head of every chunk of this
    /// stream is serialized once; per chunk only its choice (and usage) is
    /// serialized into `sse_buffer`, whose written bytes are split off for
    /// the channel instead of being copied.
    chunk_frame: ChatChunkFrame,
    sse_buffer: BytesMut,
}

impl<'a> ChatStream<'a> {
    /// Phase 1: what the spec, the dispatch and the tokenizer decide for the
    /// whole stream, and the empty per-choice state.
    fn open(
        processor: &'a StreamingProcessor,
        dispatch: &'a context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        spec: &'a ChatResponseSpec,
        tx: &'a SseSender,
        pd_timing: Option<context::PdTiming>,
    ) -> Self {
        let start_time = Instant::now();
        let model = &dispatch.model;

        let deepseek_usage = spec.provider == ProviderProfile::DeepSeek;
        let include_usage = spec
            .stream_options
            .as_ref()
            .is_some_and(|opts| opts.include_usage.unwrap_or(false));
        let emit_usage_null = deepseek_usage && include_usage;
        let mut continuous_usage = spec
            .stream_options
            .as_ref()
            .filter(|opts| {
                !deepseek_usage
                    && opts.include_usage.unwrap_or(false)
                    && opts.continuous_usage_stats.unwrap_or(false)
            })
            .map(|_| ChatStreamUsage::default());

        let reasoning_parser_name = processor.parser_resolver.reasoning_parser(model);
        let tool_parser_name = processor.parser_resolver.tool_parser(model);

        // Where a Symphony table reads the response its reader takes the thought, and no
        // legacy parser is built to be asked whether one is available.
        let reasoning_parser_available = spec.separate_reasoning
            && spec.symphony.is_none()
            && utils::check_reasoning_parser_availability(
                &processor.reasoning_parser_factory,
                reasoning_parser_name.as_deref(),
                model,
            );

        if let Some(usage) = &mut continuous_usage {
            usage.reasoning_parsed = reasoning_parser_available || spec.symphony.is_some();
        }

        let native_tool_format = processor
            .tool_parser_factory
            .registry()
            .uses_native_tool_format(tool_parser_name.as_deref());
        let used_json_schema = if native_tool_format {
            false
        } else {
            spec.tool_choice
                .as_ref()
                .is_some_and(ToolChoice::forces_call)
        };
        let is_specific_function =
            used_json_schema && matches!(spec.tool_choice, Some(ToolChoice::Function { .. }));

        let tool_parser_available = spec.tools.is_some()
            && utils::check_tool_parser_availability(
                &processor.tool_parser_factory,
                tool_parser_name.as_deref(),
                model,
            );

        if spec.separate_reasoning && spec.symphony.is_none() && !reasoning_parser_available {
            debug!(
                "No reasoning parser found for model '{}', skipping reasoning parsing",
                model
            );
        }

        if spec.tools.is_some() && !tool_parser_available {
            debug!(
                "No tool parser found for model '{}', skipping tool call parsing",
                model
            );
        }

        let chunk_frame = ChatChunkFrame::new(
            &dispatch.request_id,
            model,
            dispatch.created,
            dispatch.weight_version.as_deref(),
        );

        Self {
            processor,
            spec,
            dispatch,
            tx,
            think_in_prefill: tokenizer.think_in_prefill(),
            tokenizer,
            stop_params,
            pd_timing,
            start_time,
            first_token_time: None,
            deepseek_usage,
            include_usage,
            emit_usage_null,
            continuous_usage,
            reasoning_parser_name,
            tool_parser_name,
            reasoning_parser_available,
            tool_parser_available,
            thinking_override: spec.reasoning_in_prefill,
            used_json_schema,
            is_specific_function,
            opens_tool_call: forces_tool_call(spec),
            is_firsts: HashMap::new(),
            finish_reasons: HashMap::new(),
            matched_stops: HashMap::new(),
            stopped_indices: HashSet::new(),
            completed_indices: HashSet::new(),
            tailed_indices: HashSet::new(),
            engine_outlived: false,
            tail_tokens: 0,
            prompt_tokens: HashMap::new(),
            completion_tokens: CompletionTokenTracker::new(),
            cached_tokens: HashMap::new(),
            reasoning_tokens: HashMap::new(),
            parsed_reasoning_tokens: HashMap::new(),
            held_tokens: HashMap::new(),
            spec_accepted: HashMap::new(),
            spec_drafted: HashMap::new(),
            reasoning_parsers: HashMap::new(),
            tool_parsers: HashMap::new(),
            has_tool_calls: HashMap::new(),
            stop_decoders: HashMap::new(),
            readers: HashMap::new(),
            delta: Delta::default(),
            spare: Events::new(),
            final_indices: None,
            sse_encoder: SseEncoder::new(),
            chunk_frame,
            sse_buffer: BytesMut::with_capacity(4096),
        }
    }

    /// The usage a chunk carries when the client asked for continuous usage.
    fn usage_snapshot(&self) -> Option<Usage> {
        self.continuous_usage
            .as_ref()
            .map(|tracker| tracker.snapshot(self.spec.unbilled_prompt_tokens))
    }

    /// The stream's first engine output, for the TTFT.
    fn note_first_token(&mut self) {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
            if let Some(timing) = &self.pd_timing {
                timing.record_first_decode_token(self.processor.backend_type, &self.dispatch.model);
            }
        }
    }

    /// A choice's first output opens it with the role chunk, as the engine's
    /// own server does, whether or not that output decodes to visible text (a
    /// special token the decoder skips, a partial UTF-8 sequence): a client
    /// timing its first chunk otherwise waits for the next step's text.
    async fn open_choice(&mut self, index: u32) -> Result<(), String> {
        if self.is_firsts.get(&index).copied().unwrap_or(true) {
            let usage = self.usage_snapshot();
            StreamingProcessor::send_role_chunk(
                self.tx,
                &mut self.sse_buffer,
                &self.chunk_frame,
                index,
                self.opens_tool_call,
                usage.as_ref(),
                self.emit_usage_null,
            )
            .await?;
            self.is_firsts.insert(index, false);
        }
        Ok(())
    }

    /// The logprobs of an engine chunk, in the public API's shape.
    fn logprobs_of(&self, chunk: &ProtoGenerateStreamChunk) -> Option<ChatLogProbs> {
        chunk.output_logprobs().map(|ref proto_logprobs| {
            utils::convert_proto_to_openai_logprobs(
                proto_logprobs,
                &self.tokenizer,
                self.spec.top_logprobs,
            )
        })
    }

    /// Phase 2: the main loop. The engine stream is polled once per message
    /// and every message reaches the client as its own frame as soon as it is
    /// processed; the text a message releases goes through the one emission
    /// below, so nothing reaches the client without being parsed. After the
    /// engine's end the loop keeps turning until every choice's decoder and
    /// parsers have flushed.
    async fn read_engine(
        &mut self,
        grpc_stream: &mut ProtoStream,
    ) -> Result<(), utils::StreamFailure> {
        loop {
            // An engine that outlived the client's stream (every choice
            // closed, a locally stopped one streaming on past its stop) is
            // not polled again: the loop ends as it does at the engine's
            // end, and the driver aborts the engine stream.
            if self.final_indices.is_none() && !self.engine_outlived && self.engine_outlived_now() {
                self.engine_outlived = true;
            }
            let response = if self.final_indices.is_none() && !self.engine_outlived {
                // The chunks written so far leave as a frame before the
                // engine is polled, so every engine message reaches the
                // client as its own frame as soon as it is processed; the
                // engine stream is polled once per message.
                StreamingProcessor::flush_chunks(self.tx, &mut self.sse_buffer).await?;
                let item = grpc_stream.next().await;
                if item.is_none() && grpc_stream.client_left() {
                    // The client's departure ends the read here: text the
                    // decoder or a parser still holds can reach nobody, and
                    // the EOF flush would fail on the closed sender before
                    // the caller settles what was read.
                    return Ok(());
                }
                match item.transpose() {
                    Ok(response) => response,
                    Err(e) => {
                        // What was produced before the failure still goes out.
                        StreamingProcessor::flush_chunks(self.tx, &mut self.sse_buffer).await?;
                        return Err(utils::StreamFailure::upstream(
                            e,
                            self.completion_tokens.delivered(),
                        ));
                    }
                }
            } else {
                None
            };
            let mut final_chunk = response.is_none();

            // Text the stop decoder produced for this response, if any. Per-chunk
            // text and the end-of-stream flush both funnel into the shared emission
            // below, so neither can reach the client without being parsed.
            let pending = match response.map(|response| response.into_response()) {
                Some(ProtoResponseVariant::Chunk(chunk)) => self.read_chunk(chunk).await?,
                Some(ProtoResponseVariant::Complete(complete)) => {
                    self.read_complete(complete).await?
                }
                Some(ProtoResponseVariant::None) => continue,
                None => match self.read_end().await? {
                    StreamEnd::Finished => break,
                    StreamEnd::Nothing => continue,
                    StreamEnd::Text(pending, parser_eof) => {
                        final_chunk = parser_eof;
                        Some(pending)
                    }
                },
            };
            let Some(pending) = pending else {
                continue;
            };
            self.emit(pending, final_chunk).await?;
        }

        StreamingProcessor::flush_chunks(self.tx, &mut self.sse_buffer).await?;
        Ok(())
    }

    /// An engine chunk: the choice's tokens through its stop decoder (or
    /// Symphony's reader, which emits its events here) into the text the
    /// emission parses, with the tokens of chunks that decoded to nothing
    /// yet counted with it.
    async fn read_chunk(
        &mut self,
        chunk: ProtoGenerateStreamChunk,
    ) -> Result<Option<PendingText>, utils::StreamFailure> {
        // Track TTFT immediately on first chunk received from backend
        self.note_first_token();

        let index = chunk.index();

        // Once the local stop decoder has fired for an index, any further
        // engine chunk for it is the tail the engine generated for nobody
        // (an engine that saw the stop sends its Complete instead): counted,
        // not read; the request is aborted once every choice has closed,
        // until then another choice may still be streaming.
        if self.stopped_indices.contains(&index) {
            self.tail_tokens += chunk.token_ids().len() as u64;
            self.tailed_indices.insert(index);
            return Ok(None);
        }

        self.completion_tokens.record_chunk(&chunk);
        if let Some(usage) = &mut self.continuous_usage {
            usage.record_chunk(&chunk);
        }

        self.open_choice(index).await?;

        // Get or create stop decoder for this index
        let stop_decoder = self.stop_decoders.entry(index).or_insert_with(|| {
            let (ref stop, ref stop_token_ids, skip_special_tokens, no_stop_trim, ignore_eos) =
                self.stop_params;
            utils::create_stop_decoder(
                &self.tokenizer,
                stop.as_ref(),
                stop_token_ids.as_ref(),
                skip_special_tokens,
                no_stop_trim,
                ignore_eos,
            )
        });

        // Process tokens through stop decoder: Symphony's reader takes
        // them with their spans and gives events, the older path the text.
        // The step's events borrow the delta, so both leave the stream's
        // state for the step and come back once the chunks are out.
        let mut delta = std::mem::take(&mut self.delta);
        let symphony_events = match self.spec.symphony.as_ref() {
            Some(assignment) => {
                let mut events: Events<'_> = std::mem::take(&mut self.spare);
                let reader = match self.readers.entry(index) {
                    Entry::Occupied(reader) => reader.into_mut(),
                    Entry::Vacant(slot) => slot.insert(
                        Reader::new(
                            assignment,
                            self.spec.tools.clone().unwrap_or_default(),
                            &mut events,
                        )?
                        .first_call_only(self.spec.first_tool_call_only),
                    ),
                };
                let stopped =
                    reader.read(stop_decoder, chunk.token_ids(), &mut delta, &mut events)?;
                Some((events, stopped))
            }
            None => None,
        };
        let (chunk_text, should_stop) = match &symphony_events {
            Some((_, stopped)) => (String::new(), *stopped),
            None => StreamingProcessor::process_chunk_tokens(stop_decoder, chunk.token_ids())?,
        };

        if should_stop {
            // Stop-decoder match takes precedence: pin "stop" even if
            // the backend's eventual Complete carries "length" (the
            // local stop sequence fired first). Any pre-stop text in
            // `chunk_text` is still emitted by `emit` before the finish
            // reason is flushed in Phase 4.
            self.finish_reasons
                .entry(index)
                .or_insert_with(|| "stop".to_string());
            self.matched_stops.entry(index).or_insert_with(|| {
                stop_decoder
                    .matched_stop()
                    .map(|s| Value::String(s.to_string()))
            });
            self.stopped_indices.insert(index);
            // The choice closes here, as its Complete would close it: the
            // counts of the chunk the stop matched in are its counts (the
            // engine's own server counts a stopped request through that
            // step as well), and the tracker settles what the stream had
            // reached. What the engine generates past this step is the tail
            // the abort cuts, never the client's usage.
            self.prompt_tokens.insert(index, chunk.prompt_tokens());
            self.cached_tokens.insert(index, chunk.cached_tokens());
            self.reasoning_tokens
                .insert(index, chunk.reasoning_tokens());
            self.completion_tokens.record_local_stop(index);
        }

        if let Some((mut events, _)) = symphony_events {
            // Recorded before the usage is snapshotted, so a continuous
            // usage counts the reasoning text it rides with.
            let chunk_reasoning_tokens = reasoning_text_tokens(events.as_slice());
            if chunk_reasoning_tokens > 0 {
                *self.parsed_reasoning_tokens.entry(index).or_default() += chunk_reasoning_tokens;
                if let Some(tracker) = &mut self.continuous_usage {
                    tracker.record_reasoning(index, chunk_reasoning_tokens);
                }
            }
            let usage = self.usage_snapshot();
            let choice_logprobs = self.logprobs_of(&chunk);
            StreamingProcessor::push_events(
                self.tx,
                &mut self.sse_buffer,
                &self.chunk_frame,
                index,
                &mut events,
                choice_logprobs,
                usage.as_ref(),
                self.emit_usage_null,
            )
            .await?;
            self.spare = events.recycle();
            self.delta = delta;
            // The stop ended the choice's output: the reader ends it now,
            // with the finish "stop", as its Complete would have (the
            // decoder holds no tail past a stop).
            if should_stop {
                self.end_reader(index, None).await?;
            }
            return Ok(None);
        }
        self.delta = delta;

        if chunk_text.is_empty() {
            *self.held_tokens.entry(index).or_default() += chunk.token_ids().len() as u32;
            return Ok(None);
        }

        // Process logprobs if present
        let choice_logprobs = self.logprobs_of(&chunk);

        let chunk_tokens =
            chunk.token_ids().len() as u32 + self.held_tokens.remove(&index).unwrap_or(0);
        Ok(Some((index, chunk_text, choice_logprobs, chunk_tokens)))
    }

    /// A choice's Complete: its decoder finished, its counts and finish
    /// reason recorded (a local stop match keeps precedence), and, for a
    /// choice Symphony reads, its output ended here; otherwise the text the
    /// decoder still held goes to the emission. The Complete of a choice the
    /// engine streamed past its local stop adds only its speculative counts.
    async fn read_complete(
        &mut self,
        complete: ProtoGenerateComplete,
    ) -> Result<Option<PendingText>, utils::StreamFailure> {
        let index = complete.index();
        let request_id = &self.dispatch.request_id;

        // The Complete of a choice the engine generated past its local stop
        // (another choice kept the stream open) is the tail's end: the
        // choice closed at its stop with that step's counts, so only the
        // counts the stop could not know are taken from it.
        if self.tailed_indices.contains(&index) {
            self.completed_indices.insert(index);
            self.spec_accepted
                .insert(index, complete.spec_accepted_tokens());
            self.spec_drafted
                .insert(index, complete.spec_draft_tokens());
            return Ok(None);
        }

        // Complete byte decoding before routing held text through the
        // same reasoning/tool parsers as ordinary chunks.
        let flushed = match self.stop_decoders.get_mut(&index) {
            Some(decoder) => {
                let tail = decoder
                    .finish()
                    .map_err(|e| decoding_failed(self.readers.get_mut(&index), e))?;
                utils::warn_detokenize_recoveries(request_id, index, decoder);
                if decoder.is_stopped() {
                    self.stopped_indices.insert(index);
                    self.finish_reasons.insert(index, "stop".to_string());
                    self.matched_stops.insert(
                        index,
                        decoder.matched_stop().map(|s| Value::String(s.to_string())),
                    );
                }
                match tail {
                    SequenceDecoderOutput::Text(text)
                    | SequenceDecoderOutput::StoppedWithText(text)
                        if !text.is_empty() =>
                    {
                        Some(text)
                    }
                    _ => None,
                }
            }
            None => None,
        };

        // Store metadata
        self.completed_indices.insert(index);
        self.prompt_tokens.insert(index, complete.prompt_tokens());

        self.completion_tokens.record_complete(&complete);
        if let Some(usage) = &mut self.continuous_usage {
            usage.record_complete(&complete);
        }

        // A choice's first engine output may be its Complete (a
        // completion with no tokens: a finish that carries none
        // comes without a Chunk). It opens the choice as a first
        // Chunk does, the role chunk before the finish chunk, and
        // records the stream's first-token time; its usage, as a
        // first Chunk's, counts the output it rides on.
        self.note_first_token();
        self.open_choice(index).await?;

        self.cached_tokens.insert(index, complete.cached_tokens());
        self.reasoning_tokens
            .insert(index, complete.reasoning_tokens());
        self.spec_accepted
            .insert(index, complete.spec_accepted_tokens());
        self.spec_drafted
            .insert(index, complete.spec_draft_tokens());

        // A local stop-decoder match already pinned "stop" for this
        // index; don't let the engine's finish reason overwrite it.
        if !self.stopped_indices.contains(&index) {
            self.finish_reasons
                .insert(index, complete.finish_reason().to_string());
            self.matched_stops
                .insert(index, complete.matched_stop_json());
        }

        // Symphony ends the choice's output here: the decoder's last
        // text, then the engine's finish (a choice the gateway's own stop
        // ended was ended at its stop). The finish chunk is Phase 4's, with
        // the matched stop; the `Finish` event gives it the calls made, and
        // the usage the reasoning tokens. A choice whose first engine output
        // is its Complete has no reader yet: it gets one for its end, so an
        // output of no tokens is recorded as every output is.
        if self.spec.symphony.is_some() {
            if !self
                .readers
                .get(&index)
                .is_some_and(|reader| reader.ended())
            {
                self.end_reader(index, flushed).await?;
            }
            return Ok(None);
        }

        // Don't break - continue reading all Complete messages for n>1
        // Text the decoder held for a partial stop match: the
        // tokens that produced it were counted as held.
        Ok(flushed.map(|text| {
            (
                index,
                text,
                None,
                self.held_tokens.remove(&index).unwrap_or(0),
            )
        }))
    }

    /// A choice Symphony reads ends: the decoder's last text (`flushed`) is
    /// the reader's, then its output ends with the choice's finish reason
    /// (the engine's, or the "stop" a local stop pinned). The `Finish` event
    /// says what the Complete frame would have: the calls made, for the
    /// finish reason, and the reasoning tokens, for the usage. A choice
    /// without a reader yet (its first engine output is its Complete) gets
    /// one for its end, so an output of no tokens is recorded as every
    /// output is.
    async fn end_reader(
        &mut self,
        index: u32,
        flushed: Option<String>,
    ) -> Result<(), utils::StreamFailure> {
        let Some(assignment) = self.spec.symphony.as_ref() else {
            return Ok(());
        };
        let request_id = &self.dispatch.request_id;
        let reason = self
            .finish_reasons
            .get(&index)
            .map_or("stop", String::as_str);
        let mut delta = std::mem::take(&mut self.delta);
        let mut events: Events<'_> = std::mem::take(&mut self.spare);
        let reader = match self.readers.entry(index) {
            Entry::Occupied(reader) => reader.into_mut(),
            Entry::Vacant(slot) => slot.insert(
                Reader::new(
                    assignment,
                    self.spec.tools.clone().unwrap_or_default(),
                    &mut events,
                )?
                .first_call_only(self.spec.first_tool_call_only),
            ),
        };
        if let Some(text) = flushed {
            reader.tail(text, &mut delta, &mut events)?;
        }
        let who = Who {
            request_id,
            model: &self.dispatch.model,
            choice: index,
        };
        reader.end(engine_finish(reason), who, &mut events)?;
        if let Some(finished) = finished(events.as_slice()) {
            self.has_tool_calls.insert(index, finished.tool_calls > 0);
            self.parsed_reasoning_tokens
                .insert(index, finished.reasoning_tokens);
        }
        let usage = self.usage_snapshot();
        StreamingProcessor::push_events(
            self.tx,
            &mut self.sse_buffer,
            &self.chunk_frame,
            index,
            &mut events,
            None,
            usage.as_ref(),
            self.emit_usage_null,
        )
        .await?;
        self.spare = events.recycle();
        self.delta = delta;
        Ok(())
    }

    /// Whether the engine has outlived the client's stream: every expected
    /// choice is closed (its Complete came or its local stop fired) while a
    /// stopped choice without its Complete has streamed on past its stop,
    /// so the engine is generating a tail nobody receives. A stopped choice
    /// whose next message has not come yet is waited for: an engine that
    /// saw the stop (an EOS id, a stop the request also gave it) ends the
    /// choice on that step and its Complete is the next message, so nothing
    /// is aborted and the Complete is read as always; a chunk instead is
    /// the tail. Nothing to look at while no stop has fired.
    fn engine_outlived_now(&self) -> bool {
        if self.stopped_indices.is_empty() {
            return false;
        }
        let mut open_stops = 0u32;
        let mut tailed = false;
        for index in self.stopped_indices.difference(&self.completed_indices) {
            open_stops += 1;
            tailed |= self.tailed_indices.contains(index);
        }
        tailed && self.completed_indices.len() as u32 + open_stops >= self.spec.expected_choices
    }

    /// The engine stream, released before the client's finish is written.
    /// An engine the gateway's own stop outlived is aborted now: dropping
    /// the stream un-marked is the abort path a client disconnect takes (the
    /// abort RPC or frame, and the worker's end of the stream drained off
    /// this task), so the slot goes back to the engine's batch within a step
    /// instead of running to `max_tokens` for nobody. An engine that ended
    /// on its own is marked completed, so that its drop sends no abort.
    fn release_engine(&self, mut grpc_stream: ProtoStream, endpoint: &'static str) {
        if !self.engine_outlived {
            grpc_stream.mark_completed();
            return;
        }
        Metrics::record_local_stop_abort(LocalStopAbort {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.processor.backend_type,
            model_id: &self.dispatch.model,
            endpoint,
            tail_tokens: self.tail_tokens,
        });
        drop(grpc_stream);
    }

    /// The engine's end, one pass per call: first the choices Symphony reads
    /// whose Complete never came are ended as aborts, then every remaining
    /// choice's decoder and parsers get their turn, one (index, EOF) turn
    /// per pass, until none is left.
    async fn read_end(&mut self) -> Result<StreamEnd, utils::StreamFailure> {
        let request_id = &self.dispatch.request_id;

        // A choice Symphony reads whose Complete never came: the decoder's
        // last text is the reader's, then the output ends as an abort so
        // what the engine holds goes out, as the older parsers' flush
        // below sends theirs. Those indices are not the loop's below.
        let mut unended: Vec<u32> = self
            .readers
            .iter()
            .filter(|(_, reader)| !reader.ended())
            .map(|(index, _)| *index)
            .collect();
        unended.sort_unstable();
        for index in unended {
            let Some(reader) = self.readers.get_mut(&index) else {
                continue;
            };
            let mut delta = std::mem::take(&mut self.delta);
            let mut events: Events<'_> = std::mem::take(&mut self.spare);
            if let Some(decoder) = self.stop_decoders.get_mut(&index) {
                let tail = decoder.finish().map_err(|e| reader.failed(e.to_string()))?;
                utils::warn_detokenize_recoveries(request_id, index, decoder);
                if decoder.is_stopped() {
                    self.stopped_indices.insert(index);
                    self.finish_reasons.insert(index, "stop".to_string());
                    self.matched_stops.insert(
                        index,
                        decoder.matched_stop().map(|s| Value::String(s.to_string())),
                    );
                }
                if let SequenceDecoderOutput::Text(text)
                | SequenceDecoderOutput::StoppedWithText(text) = tail
                {
                    reader.tail(text, &mut delta, &mut events)?;
                }
            }
            let who = Who {
                request_id,
                model: &self.dispatch.model,
                choice: index,
            };
            reader.end(EngineFinish::Abort, who, &mut events)?;
            // The `Finish` event says what the Complete frame would have:
            // the calls made, for the finish reason, and the reasoning
            // tokens, for the usage.
            if let Some(finished) = finished(events.as_slice()) {
                self.has_tool_calls.insert(index, finished.tool_calls > 0);
                self.parsed_reasoning_tokens
                    .insert(index, finished.reasoning_tokens);
            }
            let usage = self.usage_snapshot();
            StreamingProcessor::push_events(
                self.tx,
                &mut self.sse_buffer,
                &self.chunk_frame,
                index,
                &mut events,
                None,
                usage.as_ref(),
                self.emit_usage_null,
            )
            .await?;
            self.spare = events.recycle();
            self.delta = delta;
        }

        // Finish decoder text before sending EOF to the parsers. A
        // clean transport EOF does not supply Complete metadata.
        let indices = self.final_indices.get_or_insert_with(|| {
            let mut indices: Vec<_> = self
                .stop_decoders
                .keys()
                .chain(self.reasoning_parsers.keys())
                .copied()
                .filter(|index| !self.readers.contains_key(index))
                .collect();
            indices.sort_unstable();
            indices.dedup();
            indices
                .into_iter()
                .flat_map(|index| [(index, true), (index, false)])
                .collect()
        });
        let Some((index, parser_eof)) = indices.pop() else {
            return Ok(StreamEnd::Finished);
        };
        let text = if parser_eof {
            String::new()
        } else {
            match self.stop_decoders.get_mut(&index) {
                Some(decoder) => {
                    let tail = decoder.finish().map_err(|e| e.to_string())?;
                    utils::warn_detokenize_recoveries(request_id, index, decoder);
                    if decoder.is_stopped() {
                        self.stopped_indices.insert(index);
                        self.finish_reasons.insert(index, "stop".to_string());
                        self.matched_stops.insert(
                            index,
                            decoder.matched_stop().map(|s| Value::String(s.to_string())),
                        );
                    }
                    match tail {
                        SequenceDecoderOutput::Text(text)
                        | SequenceDecoderOutput::StoppedWithText(text) => text,
                        _ => String::new(),
                    }
                }
                None => String::new(),
            }
        };
        if !parser_eof && text.is_empty() {
            return Ok(StreamEnd::Nothing);
        }
        Ok(StreamEnd::Text(
            (
                index,
                text,
                None,
                self.held_tokens.remove(&index).unwrap_or(0),
            ),
            parser_eof,
        ))
    }

    /// The one emission: text a message released goes through the reasoning
    /// parser, then the tool parsers, then out as content, each as the chunks
    /// the public API streams; `final_chunk` is the parsers' EOF turn.
    async fn emit(
        &mut self,
        pending: PendingText,
        final_chunk: bool,
    ) -> Result<(), utils::StreamFailure> {
        let (index, text, mut choice_logprobs, chunk_tokens) = pending;
        let model = &self.dispatch.model;
        let mut usage = self.usage_snapshot();

        // Text that reaches here without an engine chunk before it (a
        // decoder flush on the terminal frame) still opens its choice.
        self.open_choice(index).await?;

        // Calculate delta
        let mut delta = text;

        // Reasoning content handling
        let in_reasoning = if self.spec.separate_reasoning && self.reasoning_parser_available {
            let (normal_text, reasoning_choice, in_reasoning, chunk_reasoning_tokens) =
                self.processor.process_reasoning_stream(
                    (!final_chunk).then_some(delta.as_str()),
                    chunk_tokens,
                    index,
                    &mut self.reasoning_parsers,
                    self.thinking_override,
                    self.think_in_prefill,
                    self.reasoning_parser_name.as_deref(),
                    model,
                );
            // Recorded before this frame's usage is snapshotted again, so
            // a continuous usage counts the reasoning text it rides with,
            // and the content or tool chunks that follow in the frame
            // agree with it.
            if chunk_reasoning_tokens > 0 {
                *self.parsed_reasoning_tokens.entry(index).or_default() += chunk_reasoning_tokens;
                if let Some(tracker) = &mut self.continuous_usage {
                    tracker.record_reasoning(index, chunk_reasoning_tokens);
                    usage = Some(tracker.snapshot(self.spec.unbilled_prompt_tokens));
                }
            }
            if let Some(mut choice) = reasoning_choice {
                // The chunk's token logprobs ride on its reasoning delta when
                // none of its text is content, as on a content delta.
                if normal_text.is_empty() {
                    choice.logprobs = choice_logprobs.take();
                }
                StreamingProcessor::push_chunk(
                    self.tx,
                    &mut self.sse_buffer,
                    &self.chunk_frame,
                    &choice,
                    usage.as_ref(),
                    self.emit_usage_null,
                )
                .await
                .map_err(|()| "Failed to send reasoning chunk".to_string())?;
            }
            delta = normal_text;
            in_reasoning
        } else {
            false
        };

        if final_chunk && delta.is_empty() {
            return Ok(());
        }

        // Tool call handling
        let tool_choice_enabled = !matches!(
            self.spec.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );

        if let Some(tools_ref) = self.spec.tools.as_ref() {
            if !in_reasoning
                && tool_choice_enabled
                && (self.tool_parser_available || self.used_json_schema)
            {
                let tool_choices = if self.is_specific_function {
                    // Handle specific function case - emit tool call deltas with arguments
                    StreamingProcessor::process_specific_function_stream(
                        &delta,
                        index,
                        &mut self.has_tool_calls,
                        self.spec.tool_choice.as_ref(),
                        model,
                        self.spec.history_tool_calls_count,
                    )
                } else {
                    // Use incremental parser for regular/required modes
                    self.processor
                        .process_tool_calls_stream(
                            &delta,
                            index,
                            &mut self.tool_parsers,
                            &mut self.has_tool_calls,
                            tools_ref,
                            self.tool_parser_name.as_deref(),
                            model,
                            self.spec.history_tool_calls_count,
                            self.used_json_schema,
                            self.spec.first_tool_call_only,
                        )
                        .await
                };

                for choice in &tool_choices {
                    StreamingProcessor::push_chunk(
                        self.tx,
                        &mut self.sse_buffer,
                        &self.chunk_frame,
                        choice,
                        usage.as_ref(),
                        self.emit_usage_null,
                    )
                    .await
                    .map_err(|()| "Failed to send tool call chunk".to_string())?;
                }

                // Always skip regular content when tool parsing is active
                // Parser either emitted chunks or buffered content
                return Ok(());
            }
        }

        // Regular content emission
        if !delta.is_empty() {
            let mut choice = assistant_choice(index, Some(delta), None, None);
            choice.logprobs = choice_logprobs;
            StreamingProcessor::push_chunk(
                self.tx,
                &mut self.sse_buffer,
                &self.chunk_frame,
                &choice,
                usage.as_ref(),
                self.emit_usage_null,
            )
            .await
            .map_err(|()| "Failed to send content chunk".to_string())?;
        }
        Ok(())
    }

    /// Phase 3: the end-of-stream parser flush: first any text still buffered
    /// as a prospective tool call that never materialized (dropping it
    /// produced fully-empty streams), then any parsed-but-unstreamed tool
    /// arguments.
    async fn flush_parsers(&mut self, usage: Option<&Usage>) -> Result<(), utils::StreamFailure> {
        let request_id = &self.dispatch.request_id;
        let model = &self.dispatch.model;
        let created = self.dispatch.created;
        let system_fingerprint = self.dispatch.weight_version.as_deref();

        for (index, parser_guard) in &mut self.tool_parsers {
            let leftover_text = parser_guard.take_unstreamed_normal_text();
            if !leftover_text.is_empty() {
                let content_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                    .created(created)
                    .add_choice_content_with_logprobs(*index, leftover_text, None)
                    .maybe_system_fingerprint(system_fingerprint)
                    .maybe_usage(usage.cloned())
                    .build();

                let sse_chunk = self
                    .sse_encoder
                    .encode_data(&ChatChunkWithUsage::new(
                        &content_chunk,
                        self.emit_usage_null,
                    ))
                    .map_err(|e| format!("Failed to serialize content chunk: {e}"))?;
                self.tx
                    .send(Ok(sse_chunk))
                    .await
                    .map_err(|_| "Failed to send flushed content chunk".to_string())?;
            }

            if let Some(mut unstreamed_items) = parser_guard.get_unstreamed_tool_args() {
                if self.spec.first_tool_call_only {
                    utils::keep_first_tool_call_items(&mut unstreamed_items);
                }
                for tool_call_item in unstreamed_items {
                    let tool_call_delta = ToolCallDelta {
                        index: tool_call_item.tool_index as u32,
                        id: None,
                        tool_type: None,
                        function: Some(FunctionCallDelta {
                            name: None,
                            arguments: if tool_call_item.parameters.is_empty() {
                                None
                            } else {
                                Some(tool_call_item.parameters)
                            },
                        }),
                    };

                    let tool_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_tool_call_delta(*index, tool_call_delta)
                        .maybe_system_fingerprint(system_fingerprint)
                        .maybe_usage(usage.cloned())
                        .build();

                    let sse_chunk = self
                        .sse_encoder
                        .encode_data(&ChatChunkWithUsage::new(&tool_chunk, self.emit_usage_null))
                        .map_err(|e| format!("Failed to serialize tool chunk: {e}"))?;
                    self.tx
                        .send(Ok(sse_chunk))
                        .await
                        .map_err(|_| "Failed to send unstreamed tool args".to_string())?;
                }
            }
        }
        Ok(())
    }

    /// The counters once, for the usage chunk, the settlement and the
    /// metrics: every choice shares one prompt, so prompt/cache counts take
    /// max; completion and reasoning counts sum across choices.
    fn counts(&self) -> TokenCounts {
        TokenCounts {
            prompt: self.prompt_tokens.values().copied().max().unwrap_or(0),
            completion: self.completion_tokens.total(),
            cached: self.cached_tokens.values().copied().max().unwrap_or(0),
            reasoning: self.reasoning_tokens.values().sum(),
            spec_accepted: self.spec_accepted.values().sum(),
            spec_drafted: self.spec_drafted.values().sum(),
        }
    }

    /// The stream's usage when the client asked for one, from
    /// [`Self::counts`]. Built by the non-streamed answer's builder: the
    /// usage chunk carries the usage object that answer carries.
    fn final_usage(&self) -> Option<Usage> {
        (self.deepseek_usage || self.include_usage).then(|| {
            chat_usage(
                self.counts().usage(),
                (self.reasoning_parser_available || self.spec.symphony.is_some())
                    .then(|| self.parsed_reasoning_tokens.values().sum()),
                self.spec.unbilled_prompt_tokens,
            )
        })
    }

    /// Phase 4: the finish reason chunks. Partial counters are not advertised
    /// as a final aggregate if the backend omitted a choice's Complete frame.
    async fn send_finish_chunks(
        &mut self,
        final_usage: Option<&Usage>,
        usage: Option<&Usage>,
    ) -> Result<(), utils::StreamFailure> {
        let request_id = &self.dispatch.request_id;
        let model = &self.dispatch.model;
        let created = self.dispatch.created;
        let system_fingerprint = self.dispatch.weight_version.as_deref();

        let complete_usage = self.prompt_tokens.len() as u32 >= self.spec.expected_choices;
        for (position, (index, finish_reason)) in self.finish_reasons.iter().enumerate() {
            let final_finish_reason = if self.has_tool_calls.get(index).copied().unwrap_or(false)
                && finish_reason == "stop"
            {
                "tool_calls".to_string()
            } else {
                finish_reason.clone()
            };

            let matched_stop_value = self.matched_stops.get(index).and_then(|v| v.clone());

            let finish_usage = if self.deepseek_usage
                && complete_usage
                && position + 1 == self.finish_reasons.len()
            {
                final_usage.cloned()
            } else {
                usage.cloned()
            };
            let finish_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                .created(created)
                .add_choice_finish_reason(*index, final_finish_reason, matched_stop_value)
                .maybe_system_fingerprint(system_fingerprint)
                .maybe_usage(finish_usage)
                .build();

            let sse_chunk = self
                .sse_encoder
                .encode_data(&ChatChunkWithUsage::new(
                    &finish_chunk,
                    self.emit_usage_null,
                ))
                .map_err(|e| format!("Failed to serialize finish chunk: {e}"))?;
            self.tx
                .send(Ok(sse_chunk))
                .await
                .map_err(|_| "Failed to send finish chunk".to_string())?;
        }
        Ok(())
    }

    /// Phase 5: the OpenAI dialect keeps its opt-in, usage-only chunk.
    async fn send_usage_chunk(
        &mut self,
        final_usage: Option<Usage>,
    ) -> Result<(), utils::StreamFailure> {
        if !self.deepseek_usage {
            if let Some(final_usage) = final_usage {
                let usage_chunk = ChatCompletionStreamResponse::builder(
                    &self.dispatch.request_id,
                    &self.dispatch.model,
                )
                .created(self.dispatch.created)
                .usage(final_usage)
                .maybe_system_fingerprint(self.dispatch.weight_version.as_deref())
                .build();
                let sse_chunk = self
                    .sse_encoder
                    .encode_data(&usage_chunk)
                    .map_err(|e| format!("Failed to serialize usage chunk: {e}"))?;
                self.tx
                    .send(Ok(sse_chunk))
                    .await
                    .map_err(|_| "Failed to send usage chunk".to_string())?;
            }
        }
        Ok(())
    }

    /// The end: the reservation settled with what the stream used
    /// ([`Self::counts`], the engine's counts as they are), and the
    /// streaming metrics recorded.
    /// `client_left` says the read ended with the client's leaving, not with the engine's end:
    /// the reservation is still settled, but no terminal router metric is recorded for it, as
    /// the HTTP router has none for a disconnect (neither a completed request nor a failure).
    async fn settle_and_record(
        self,
        reservation: Option<Arc<SharedReservationHandle>>,
        client_left: bool,
    ) {
        let model = &self.dispatch.model;
        let counts = self.counts();
        let (total_prompt, total_completion) = (counts.prompt, counts.completion);
        if let Some(handle) = reservation {
            // `prompt_tokens` has one entry per closed index: from its
            // `Complete`, or from the chunk the gateway's own stop matched
            // in. A clean EOF that didn't close every expected `n>1` choice
            // has only partial usage -- settling with that would understate
            // the real cost. Keep the reserved amount as final instead.
            let expected_choices = self.spec.expected_choices;
            if (self.prompt_tokens.len() as u32) < expected_choices {
                handle.close_reserved_only().await;
            } else {
                handle.settle_success(counts.settlement()).await;
            }
        }
        if client_left {
            return;
        }
        // In PD mode this task only starts once the decode stream exists --
        // for sequential PD that is after prefill completes, with the first
        // decode chunk already buffered -- so `start_time` would report a
        // near-zero TTFT. Anchor at the prefill dispatch instant instead;
        // generation duration must share the base or TPOT underflows.
        let stream_base = self
            .pd_timing
            .as_ref()
            .map(|timing| timing.prefill_start)
            .unwrap_or(self.start_time);
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.processor.backend_type,
            model_id: model,
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: self.first_token_time.map(|t| t.duration_since(stream_base)),
            generation_duration: stream_base.elapsed(),
            input_tokens: Some(total_prompt as u64),
            cached_tokens: Some(u64::from(counts.cached)),
            output_tokens: total_completion as u64,
        });
    }
}

/// The Messages content-block machine: the index of the block being
/// written and which kind of block is open (at most one at a time), with
/// the events that open and stop them. Reasoning, text and tool calls can
/// alternate; every stop gives the next block the next index.
#[derive(Default)]
struct ContentBlocks {
    index: u32,
    thinking_open: bool,
    text_open: bool,
    tool_open: bool,
}

impl ContentBlocks {
    /// Stop the open content block, if any, so the next block gets the next
    /// index.
    async fn stop_open(&mut self, tx: &SseSender, buffer: &mut Vec<u8>) -> Result<(), String> {
        StreamingProcessor::stop_open_block(
            tx,
            buffer,
            &mut self.index,
            [
                &mut self.thinking_open,
                &mut self.text_open,
                &mut self.tool_open,
            ],
        )
        .await
    }

    /// Start `content_block` at the current index, after stopping the open
    /// block.
    async fn start(
        &mut self,
        tx: &SseSender,
        buffer: &mut Vec<u8>,
        content_block: ContentBlock,
    ) -> Result<(), String> {
        self.stop_open(tx, buffer).await?;
        StreamingProcessor::send_messages_event(
            tx,
            buffer,
            &MessageStreamEvent::ContentBlockStart {
                index: self.index,
                content_block,
            },
        )
        .await
    }

    /// A thinking block opens.
    async fn open_thinking(&mut self, tx: &SseSender, buffer: &mut Vec<u8>) -> Result<(), String> {
        self.start(
            tx,
            buffer,
            ContentBlock::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
        )
        .await?;
        self.thinking_open = true;
        Ok(())
    }

    /// A `tool_use` block opens for the call `tool_call_id` of `name`.
    async fn open_tool(
        &mut self,
        tx: &SseSender,
        buffer: &mut Vec<u8>,
        tool_call_id: &str,
        name: String,
    ) -> Result<(), String> {
        self.start(
            tx,
            buffer,
            ContentBlock::ToolUse {
                id: message_utils::anthropic_tool_use_id(tool_call_id),
                name,
                input: Value::Object(serde_json::Map::new()),
                caller: messages::ToolUseCaller::Direct,
            },
        )
        .await?;
        self.tool_open = true;
        Ok(())
    }

    /// `text` goes out as a text delta, in the open text block or in one
    /// that opens for it.
    async fn text_delta(
        &mut self,
        tx: &SseSender,
        buffer: &mut Vec<u8>,
        text: String,
    ) -> Result<(), String> {
        if !self.text_open {
            self.start(
                tx,
                buffer,
                ContentBlock::Text {
                    text: String::new(),
                    citations: None,
                },
            )
            .await?;
            self.text_open = true;
        }
        StreamingProcessor::send_messages_event(
            tx,
            buffer,
            &MessageStreamEvent::ContentBlockDelta {
                index: self.index,
                delta: ContentBlockDelta::TextDelta { text },
            },
        )
        .await
    }

    /// Tool call arguments go to the open `tool_use` block (see
    /// [`StreamingProcessor::send_tool_arguments`]).
    async fn tool_arguments(
        &mut self,
        tx: &SseSender,
        buffer: &mut Vec<u8>,
        partial_json: String,
    ) -> Result<(), String> {
        StreamingProcessor::send_tool_arguments(
            tx,
            buffer,
            self.index,
            self.tool_open,
            partial_json,
        )
        .await
    }
}

/// Text an engine message released for the emission, with the tokens it
/// counts for (its own and those of the chunks that decoded to nothing
/// before it).
type MessagesPendingText = (String, u32);

/// One Messages response's SSE stream being read off the engine: what the
/// spec and the dispatch decide for the whole stream, the content-block
/// machine, the stop decoder, the parsers (or Symphony's reader), the token
/// counts, and the buffer the events leave through. Each phase of the
/// stream is a method; `StreamingProcessor::process_messages_streaming_chunks`
/// runs them in order. Messages is always n=1, so this is one choice's
/// state.
struct MessagesStream<'a> {
    processor: &'a StreamingProcessor,
    spec: &'a MessagesResponseSpec,
    /// Dispatch metadata, for consistent response fields.
    dispatch: &'a context::DispatchMetadata,
    tx: &'a SseSender,

    // Metrics timing
    start_time: Instant,
    first_token_time: Option<Instant>,

    /// Reusable SSE formatting buffer to avoid allocations per event
    sse_buffer: Vec<u8>,

    blocks: ContentBlocks,
    has_tool_calls: bool,

    /// Parser state (simple fields: Messages is always n=1); the parsers are
    /// this stream's own, nothing else reaches them. The tool parser is a
    /// fresh streaming parser (not pooled: streaming parsers maintain state).
    reasoning_parser: Option<Box<dyn ReasoningParser>>,
    streaming_tool_parser: Option<Box<dyn ToolParser>>,
    stop_decoder: StopSequenceDecoder,
    /// Symphony's reader and the content-block machine, when a table reads
    /// the message (see `regular::symphony`): the engine's tokens go through
    /// the reader, its events through the machine, and the parsers above are
    /// not run. The `Finish` event gives the stop reason.
    symphony: Option<(Reader, adapt::messages::Stream)>,
    symphony_stop: Option<messages::StopReason>,
    /// The step's delta and the stream's event buffer, as the chat stream
    /// keeps them; the prompt's events wait in the buffer for the first step's.
    delta: Delta,
    spare: Events<'static>,

    // Token tracking
    completion_tokens: CompletionTokenTracker,
    prompt_tokens: u32,
    /// Authoritative usage only ever arrives via a `Complete` message; a
    /// clean EOF without one leaves `prompt_tokens` at its 0 initializer,
    /// which is indistinguishable from a genuinely empty prompt -- tracked
    /// separately so settle can tell "no usage" from "zero tokens".
    saw_complete: bool,
    finish_reason_str: String,
    matched_stop: Option<Value>,
    /// Set once the local stop decoder fires: pins "stop" and ignores later
    /// engine output (the backend has no stop-string detection over ZMQ).
    stopped: bool,
    /// The gateway's own stop closed the message with the counts of the
    /// chunk it matched in: its usage is as settled as a Complete's.
    closed_by_stop: bool,
    /// The engine streamed a chunk after the local stop: it is generating
    /// past the stop, which only a stop it cannot see does (an engine that
    /// saw the stop ends the request on that step and sends its Complete
    /// next).
    tailed: bool,
    /// The engine was streaming past the stop when the message had closed:
    /// the loop stopped polling it, and the driver aborts it instead of
    /// marking it completed (see [`Self::release_engine`]).
    engine_outlived: bool,
    /// Tokens the engine streamed after the local stop, read before the
    /// abort: the tail generated for nobody.
    tail_tokens: u64,
    /// Thinking tokens of the turn: the engine's count when it keeps one,
    /// else the tokens the parser routed to thinking, chunk by chunk.
    parsed_thinking_tokens: u32,
    engine_thinking_tokens: u32,
    /// The engine's cache hit on the prompt, for the settlement: the
    /// Messages usage reports none.
    cached_tokens: u32,
    was_in_reasoning: bool,
    held_tokens: u32,

    /// Per-request effective reasoning parser name (model-card override ->
    /// configured) and whether the parser is there, checked once upfront:
    /// it runs when the user explicitly enabled thinking, or when the
    /// selected parser needs structural special tokens.
    reasoning_parser_name: Option<String>,
    reasoning_parser_available: bool,
    /// Request building decided whether the rendered prompt ends inside the
    /// model's thinking block; the parser then starts in reasoning mode.
    thinking_override: bool,
    think_in_prefill: bool,
    /// Whether the model output is arguments-only for a specific function
    /// (`ToolChoice::Tool`); only when a JSON schema is used, native formats
    /// include framing tokens.
    is_specific_function: bool,

    /// After the engine's end without a `Complete`: the decoder flushed,
    /// the parsers' EOF turn still to take.
    decoder_finished_at_eof: bool,
}

impl<'a> MessagesStream<'a> {
    /// What the spec, the dispatch and the tokenizer decide for the whole
    /// stream, the stop decoder, the parsers (or Symphony's reader), and
    /// the empty counts.
    fn open(
        processor: &'a StreamingProcessor,
        dispatch: &'a context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        spec: &'a MessagesResponseSpec,
        tx: &'a SseSender,
    ) -> Result<Self, utils::StreamFailure> {
        let start_time = Instant::now();
        let model = &dispatch.model;
        let has_tools = spec.has_tools;

        // Stop decoder
        let stop_decoder = {
            let (ref stop, ref stop_token_ids, skip_special_tokens, no_stop_trim, ignore_eos) =
                stop_params;
            utils::create_stop_decoder(
                &tokenizer,
                stop.as_ref(),
                stop_token_ids.as_ref(),
                skip_special_tokens,
                no_stop_trim,
                ignore_eos,
            )
        };

        // Per-request effective parser names (model-card override → configured).
        let reasoning_parser_name = processor.parser_resolver.reasoning_parser(model);
        let tool_parser_name = processor.parser_resolver.tool_parser(model);

        // Check parser availability once upfront. Run parser when the user explicitly
        // enabled thinking, or when the selected parser needs structural special tokens.
        // Where a Symphony table reads the message its reader takes the thought, and no
        // legacy parser is built to be asked.
        let legacy_reads = spec.symphony.is_none();
        let reasoning_requires_special_tokens = legacy_reads
            && utils::reasoning_parser_requires_special_tokens(
                &processor.reasoning_parser_factory,
                reasoning_parser_name.as_deref(),
                model,
            );
        let separate_reasoning = reasoning_requires_special_tokens || spec.thinking_requested;
        let reasoning_parser_available = separate_reasoning
            && legacy_reads
            && utils::check_reasoning_parser_availability(
                &processor.reasoning_parser_factory,
                reasoning_parser_name.as_deref(),
                model,
            );

        let thinking_override = spec.reasoning_in_prefill;
        let think_in_prefill = tokenizer.think_in_prefill();

        let tool_choice_enabled = !matches!(&spec.tool_choice, Some(messages::ToolChoice::None));

        let tool_parser_available = has_tools
            && tool_choice_enabled
            && utils::check_tool_parser_availability(
                &processor.tool_parser_factory,
                tool_parser_name.as_deref(),
                model,
            );

        let native_tool_format = processor
            .tool_parser_factory
            .registry()
            .uses_native_tool_format(tool_parser_name.as_deref());
        let used_json_schema = !native_tool_format
            && matches!(
                &spec.tool_choice,
                Some(messages::ToolChoice::Tool { .. } | messages::ToolChoice::Any { .. })
            );

        // Check if model output is arguments-only for a specific function (ToolChoice::Tool).
        // Only applies when json_schema is used — native formats include framing tokens.
        let is_specific_function = used_json_schema
            && matches!(&spec.tool_choice, Some(messages::ToolChoice::Tool { .. }));

        // Create fresh streaming tool parser (not pooled — streaming parsers maintain state)
        let streaming_tool_parser: Option<Box<dyn ToolParser>> = if has_tools
            && tool_choice_enabled
            && spec.symphony.is_none()
            && (tool_parser_available || used_json_schema)
        {
            let parser_name = if used_json_schema {
                Some("json")
            } else {
                tool_parser_name.as_deref()
            };
            utils::create_tool_parser(&processor.tool_parser_factory, parser_name, model)
        } else {
            None
        };
        let mut spare = Events::new();
        let symphony = spec
            .symphony
            .as_ref()
            .map(|assignment| {
                Reader::new(assignment, Arc::clone(&spec.chat_tools), &mut spare)
                    .map(|reader| reader.first_call_only(spec.first_tool_call_only))
            })
            .transpose()?
            .map(|reader| (reader, adapt::messages::Stream::new()));

        Ok(Self {
            processor,
            spec,
            dispatch,
            tx,
            start_time,
            first_token_time: None,
            sse_buffer: Vec::with_capacity(512),
            blocks: ContentBlocks::default(),
            has_tool_calls: false,
            reasoning_parser: None,
            streaming_tool_parser,
            stop_decoder,
            symphony,
            symphony_stop: None,
            delta: Delta::default(),
            spare,
            completion_tokens: CompletionTokenTracker::new(),
            prompt_tokens: 0,
            saw_complete: false,
            finish_reason_str: String::new(),
            matched_stop: None,
            stopped: false,
            closed_by_stop: false,
            tailed: false,
            engine_outlived: false,
            tail_tokens: 0,
            parsed_thinking_tokens: 0,
            engine_thinking_tokens: 0,
            cached_tokens: 0,
            was_in_reasoning: false,
            held_tokens: 0,
            reasoning_parser_name,
            reasoning_parser_available,
            thinking_override,
            think_in_prefill,
            is_specific_function,
            decoder_finished_at_eof: false,
        })
    }

    /// The local stop decoder fired: "stop" is pinned over the engine's
    /// eventual finish reason (the local stop sequence fired first), and
    /// the matched sequence is kept for the stop reason of the message.
    fn note_local_stop(&mut self) {
        self.stopped = true;
        self.finish_reason_str = "stop".to_string();
        self.matched_stop = self
            .stop_decoder
            .matched_stop()
            .map(|s| Value::String(s.to_string()));
    }

    /// The gateway's own stop closed the message, as its Complete would
    /// close it: the counts of the chunk the stop matched in are the
    /// message's (the engine's own server counts a stopped request through
    /// that step as well), and the tracker settles what the stream had
    /// reached. What the engine generates past this step is the tail the
    /// abort cuts, never the client's usage.
    fn close_on_local_stop(&mut self, chunk: &ProtoGenerateStreamChunk) {
        self.prompt_tokens = chunk.prompt_tokens();
        self.cached_tokens = chunk.cached_tokens();
        self.engine_thinking_tokens = chunk.reasoning_tokens();
        self.completion_tokens.record_local_stop(0);
        self.closed_by_stop = true;
    }

    /// The engine stream, released before the message's end is written.
    /// An engine the gateway's own stop outlived is aborted now: dropping
    /// the stream un-marked is the abort path a client disconnect takes, so
    /// the slot goes back to the engine's batch within a step instead of
    /// running to `max_tokens` for nobody. An engine that ended on its own
    /// is marked completed, so that its drop sends no abort.
    fn release_engine(&self, mut grpc_stream: ProtoStream) {
        if !self.engine_outlived {
            grpc_stream.mark_completed();
            return;
        }
        Metrics::record_local_stop_abort(LocalStopAbort {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.processor.backend_type,
            model_id: &self.dispatch.model,
            endpoint: metrics_labels::ENDPOINT_MESSAGES,
            tail_tokens: self.tail_tokens,
        });
        drop(grpc_stream);
    }

    /// Phase 1: the `message_start` with the skeleton message.
    async fn send_message_start(&mut self) -> Result<(), utils::StreamFailure> {
        let start_message = Message {
            id: self.dispatch.request_id.clone(),
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![],
            model: self.dispatch.model.clone(),
            stop_reason: None,
            stop_sequence: None,
            stop_details: None,
            usage: StreamingProcessor::initial_messages_usage(),
            container: None,
        };
        StreamingProcessor::send_messages_event(
            self.tx,
            &mut self.sse_buffer,
            &MessageStreamEvent::MessageStart {
                message: start_message,
            },
        )
        .await?;
        Ok(())
    }

    /// Phase 2: the main loop. Text the stop decoder produced for a
    /// response, if any, and the end-of-stream flush both funnel into the
    /// one emission, so neither can reach the client without being parsed;
    /// after the engine's end the loop turns once more for the parsers'
    /// EOF turn.
    async fn read_engine(
        &mut self,
        grpc_stream: &mut ProtoStream,
    ) -> Result<(), utils::StreamFailure> {
        let mut final_chunk = false;
        while !final_chunk {
            // An engine that outlived the message (the local stop closed
            // it, and the engine streamed on instead of sending its
            // Complete) is not polled again: the loop ends as it does at
            // the engine's end, and the driver aborts the engine stream.
            if !self.decoder_finished_at_eof && self.stopped && !self.saw_complete && self.tailed {
                self.engine_outlived = true;
            }
            let response = if self.decoder_finished_at_eof || self.engine_outlived {
                None
            } else {
                let next = grpc_stream.next().await;
                if next.is_none() && grpc_stream.client_left() {
                    // As on the chat path: the departure ends the read before
                    // held-back text is flushed to the closed sender.
                    return Ok(());
                }
                next.transpose().map_err(|e| {
                    utils::StreamFailure::upstream(e, self.completion_tokens.delivered())
                })?
            };
            final_chunk = response.is_none() && self.decoder_finished_at_eof;

            let pending = match response.map(|response| response.into_response()) {
                Some(ProtoResponseVariant::Chunk(chunk)) => self.read_chunk(chunk).await?,
                Some(ProtoResponseVariant::Complete(complete)) => {
                    self.read_complete(complete).await?
                }
                Some(ProtoResponseVariant::None) => continue,
                None if !self.decoder_finished_at_eof => self.read_end().await?,
                None if self.reasoning_parser.is_some() => Some((String::new(), 0)),
                None => break,
            };

            let Some((chunk_text, chunk_tokens)) = pending else {
                continue;
            };
            self.emit(chunk_text, chunk_tokens, final_chunk).await?;
        }
        Ok(())
    }

    /// An engine chunk: its tokens through the stop decoder (or Symphony's
    /// reader, which emits its events here) into the text the emission
    /// parses, with the tokens of chunks that decoded to nothing yet
    /// counted with it.
    async fn read_chunk(
        &mut self,
        chunk: ProtoGenerateStreamChunk,
    ) -> Result<Option<MessagesPendingText>, utils::StreamFailure> {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
        }

        // Once the local stop decoder has fired, a further engine chunk is
        // the tail the engine generated for nobody (an engine that saw the
        // stop sends its Complete instead): counted, not read; the request
        // is aborted as soon as the loop turns.
        if self.stopped {
            self.tail_tokens += chunk.token_ids().len() as u64;
            self.tailed = true;
            return Ok(None);
        }

        self.completion_tokens.record_chunk(&chunk);
        let mut chunk_tokens = chunk.token_ids().len() as u32;

        if let Some((reader, machine)) = self.symphony.as_mut() {
            let mut delta = std::mem::take(&mut self.delta);
            let mut events: Events<'_> = std::mem::take(&mut self.spare);
            let should_stop = reader.read(
                &mut self.stop_decoder,
                chunk.token_ids(),
                &mut delta,
                &mut events,
            )?;
            self.parsed_thinking_tokens += reasoning_text_tokens(events.as_slice());
            self.symphony_stop = StreamingProcessor::send_symphony_blocks(
                self.tx,
                &mut self.sse_buffer,
                machine,
                events.as_slice(),
                self.spec.thinking_text_returned,
            )
            .await?
            .or_else(|| self.symphony_stop.take());
            self.spare = events.recycle();
            self.delta = delta;
            if should_stop {
                self.note_local_stop();
                self.close_on_local_stop(&chunk);
            }
            return Ok(None);
        }

        let (chunk_text, should_stop) =
            StreamingProcessor::process_chunk_tokens(&mut self.stop_decoder, chunk.token_ids())?;

        if should_stop {
            // Stop-decoder match takes precedence over the engine's
            // eventual finish reason (the local stop sequence fired
            // first). Pre-stop text in `chunk_text` is still emitted
            // below; Phase 4 derives StopSequence from `matched_stop`.
            self.note_local_stop();
            self.close_on_local_stop(&chunk);
        }

        if chunk_text.is_empty() {
            self.held_tokens += chunk_tokens;
            return Ok(None);
        }

        chunk_tokens += std::mem::take(&mut self.held_tokens);
        Ok(Some((chunk_text, chunk_tokens)))
    }

    /// The engine's `Complete`: the decoder's tail, the authoritative usage
    /// and the engine's finish reason (unless the local stop decoder pinned
    /// "stop"); under Symphony the reader ends the message here.
    async fn read_complete(
        &mut self,
        complete: ProtoGenerateComplete,
    ) -> Result<Option<MessagesPendingText>, utils::StreamFailure> {
        let tail = self
            .stop_decoder
            .finish()
            .map_err(|e| decoding_failed(self.symphony.as_mut().map(|(reader, _)| reader), e))?;
        utils::warn_detokenize_recoveries(&self.dispatch.request_id, 0, &self.stop_decoder);
        if self.stop_decoder.is_stopped() {
            self.note_local_stop();
        }
        let flushed = match tail {
            SequenceDecoderOutput::Text(text) | SequenceDecoderOutput::StoppedWithText(text)
                if !text.is_empty() =>
            {
                Some(text)
            }
            _ => None,
        };

        self.prompt_tokens = complete.prompt_tokens();
        self.saw_complete = true;
        self.completion_tokens.record_complete(&complete);
        self.engine_thinking_tokens = complete.reasoning_tokens();
        self.cached_tokens = complete.cached_tokens();
        // A local stop-decoder match already pinned "stop"; don't let
        // the engine's finish reason overwrite it.
        if !self.stopped {
            self.finish_reason_str = complete.finish_reason().to_string();
            self.matched_stop = complete.matched_stop_json();
        }
        let mut chunk_tokens = 0;
        if flushed.is_some() {
            chunk_tokens = std::mem::take(&mut self.held_tokens);
        }
        if let Some((reader, machine)) = self.symphony.as_mut() {
            self.symphony_stop = StreamingProcessor::end_symphony_message(
                self.tx,
                &mut self.sse_buffer,
                (reader, machine),
                (&mut self.delta, &mut self.spare),
                flushed,
                engine_finish(&self.finish_reason_str),
                Who {
                    request_id: &self.dispatch.request_id,
                    model: &self.dispatch.model,
                    choice: 0,
                },
                self.spec.thinking_text_returned,
                &mut self.parsed_thinking_tokens,
            )
            .await?
            .or_else(|| self.symphony_stop.take());
            return Ok(None);
        }
        Ok(flushed.map(|text| (text, chunk_tokens)))
    }

    /// The engine's end without a `Complete`: the pending byte text goes
    /// out, then the parser flushes on the next pass without pretending the
    /// engine sent `Complete`.
    async fn read_end(&mut self) -> Result<Option<MessagesPendingText>, utils::StreamFailure> {
        self.decoder_finished_at_eof = true;
        let tail = self
            .stop_decoder
            .finish()
            .map_err(|e| decoding_failed(self.symphony.as_mut().map(|(reader, _)| reader), e))?;
        utils::warn_detokenize_recoveries(&self.dispatch.request_id, 0, &self.stop_decoder);
        if self.stop_decoder.is_stopped() {
            self.note_local_stop();
        }
        let mut chunk_tokens = 0;
        let flushed = match tail {
            SequenceDecoderOutput::Text(text) | SequenceDecoderOutput::StoppedWithText(text)
                if !text.is_empty() =>
            {
                chunk_tokens = std::mem::take(&mut self.held_tokens);
                Some(text)
            }
            _ => None,
        };
        if let Some((reader, machine)) = self.symphony.as_mut() {
            // A stream that ends without Complete is cut: the reader ends on an
            // abort, which leaves an open call cut, as the message says.
            if reader.ended() {
                return Ok(None);
            }
            let finish = if self.finish_reason_str.is_empty() {
                EngineFinish::Abort
            } else {
                engine_finish(&self.finish_reason_str)
            };
            self.symphony_stop = StreamingProcessor::end_symphony_message(
                self.tx,
                &mut self.sse_buffer,
                (reader, machine),
                (&mut self.delta, &mut self.spare),
                flushed,
                finish,
                Who {
                    request_id: &self.dispatch.request_id,
                    model: &self.dispatch.model,
                    choice: 0,
                },
                self.spec.thinking_text_returned,
                &mut self.parsed_thinking_tokens,
            )
            .await?
            .or_else(|| self.symphony_stop.take());
            return Ok(None);
        }
        Ok(flushed.map(|text| (text, chunk_tokens)))
    }

    /// The one emission: the text a message released goes through the
    /// reasoning parser, then the tool parser, then out as content-block
    /// events; `final_chunk` is the parsers' EOF turn.
    async fn emit(
        &mut self,
        chunk_text: String,
        chunk_tokens: u32,
        final_chunk: bool,
    ) -> Result<(), utils::StreamFailure> {
        let chunk_text_len = chunk_text.len();

        // Apply reasoning parser
        let (normal_text, reasoning_chunk_text, in_reasoning) = if self.reasoning_parser_available {
            self.processor.process_messages_reasoning(
                (!final_chunk).then_some(chunk_text.as_str()),
                &mut self.reasoning_parser,
                self.thinking_override,
                self.think_in_prefill,
                self.reasoning_parser_name.as_deref(),
                &self.dispatch.model,
            )
        } else {
            (chunk_text, String::new(), false)
        };
        // The chunk's tokens are thinking when its text came out as
        // thinking or it stayed inside the block (text held back); a
        // chunk carrying both thinking and content splits by text share,
        // as the chat stream counts its reasoning tokens.
        if self.reasoning_parser_available {
            self.parsed_thinking_tokens += if !reasoning_chunk_text.is_empty() {
                let share = reasoning_chunk_text.len() as f64 / chunk_text_len.max(1) as f64;
                (f64::from(chunk_tokens) * share.min(1.0)).round() as u32
            } else if self.was_in_reasoning && in_reasoning {
                chunk_tokens
            } else {
                0
            };
            self.was_in_reasoning = in_reasoning;
        }

        // Emit thinking content block deltas
        if !reasoning_chunk_text.is_empty() {
            if !self.blocks.thinking_open {
                self.blocks
                    .open_thinking(self.tx, &mut self.sse_buffer)
                    .await?;
            }
            if self.spec.thinking_text_returned {
                StreamingProcessor::send_messages_event(
                    self.tx,
                    &mut self.sse_buffer,
                    &MessageStreamEvent::ContentBlockDelta {
                        index: self.blocks.index,
                        delta: ContentBlockDelta::ThinkingDelta {
                            thinking: reasoning_chunk_text,
                        },
                    },
                )
                .await?;
            }
        }

        if final_chunk && normal_text.is_empty() {
            return Ok(());
        }

        // Transition: reasoning ended, close thinking block
        if self.blocks.thinking_open && !in_reasoning && !normal_text.is_empty() {
            StreamingProcessor::send_messages_event(
                self.tx,
                &mut self.sse_buffer,
                &MessageStreamEvent::ContentBlockStop {
                    index: self.blocks.index,
                },
            )
            .await?;
            self.blocks.thinking_open = false;
            self.blocks.index += 1;
        }

        // Tool call handling: incremental streaming parser
        if !in_reasoning && self.streaming_tool_parser.is_some() {
            if self.is_specific_function {
                self.emit_specific_function_arguments(normal_text).await?;
            } else {
                self.emit_parsed_tool_calls(normal_text).await?;
            }
            return Ok(());
        }

        // Regular text emission (no tools active)
        if !normal_text.is_empty() {
            self.blocks
                .text_delta(self.tx, &mut self.sse_buffer, normal_text)
                .await?;
        }
        Ok(())
    }

    /// Specific function: the entire output is the arguments of one tool,
    /// whose `tool_use` block opens on the first text.
    async fn emit_specific_function_arguments(
        &mut self,
        arguments: String,
    ) -> Result<(), utils::StreamFailure> {
        if !self.has_tool_calls {
            self.has_tool_calls = true;
            let tool_name = match &self.spec.tool_choice {
                Some(messages::ToolChoice::Tool { name, .. }) => name.clone(),
                _ => String::new(),
            };
            let tool_call_id = utils::generate_tool_call_id(
                &self.dispatch.model,
                &tool_name,
                0,
                self.spec.history_tool_calls_count,
            );
            // Close the open block before starting the tool block.
            self.blocks
                .open_tool(self.tx, &mut self.sse_buffer, &tool_call_id, tool_name)
                .await?;
        }
        // Emit arguments delta, unless reasoning stopped the block
        self.blocks
            .tool_arguments(self.tx, &mut self.sse_buffer, arguments)
            .await?;
        Ok(())
    }

    /// Regular/required tool choice: `text` through the incremental tool
    /// parser; arguments that finish the open call come before the text
    /// after it in the same chunk, then the calls that open.
    async fn emit_parsed_tool_calls(&mut self, text: String) -> Result<(), utils::StreamFailure> {
        let Some(parser) = self.streaming_tool_parser.as_mut() else {
            return Ok(());
        };
        match parser.parse_incremental(&text, &self.spec.chat_tools).await {
            Ok(StreamingParseResult {
                normal_text: text,
                mut calls,
            }) => {
                if self.spec.first_tool_call_only {
                    utils::keep_first_tool_call_items(&mut calls);
                }
                // Arguments that finish the open call come before
                // the text after it in the same chunk.
                let finishing = if self.blocks.tool_open {
                    calls
                        .iter()
                        .position(|call| call.name.is_some())
                        .unwrap_or(calls.len())
                } else {
                    0
                };
                for tool_call_item in calls.drain(..finishing) {
                    self.blocks
                        .tool_arguments(self.tx, &mut self.sse_buffer, tool_call_item.parameters)
                        .await?;
                }

                // Emit normal text from parser as text content blocks
                if !text.is_empty() {
                    self.blocks
                        .text_delta(self.tx, &mut self.sse_buffer, text)
                        .await?;
                }

                // Emit tool call events
                for tool_call_item in calls {
                    self.has_tool_calls = true;

                    if let Some(ref name) = tool_call_item.name {
                        // New tool call: close previous blocks, emit start
                        let tool_call_id = utils::generate_tool_call_id(
                            &self.dispatch.model,
                            name,
                            tool_call_item.tool_index,
                            self.spec.history_tool_calls_count,
                        );
                        self.blocks
                            .open_tool(self.tx, &mut self.sse_buffer, &tool_call_id, name.clone())
                            .await?;
                    }

                    // Emit incremental arguments
                    self.blocks
                        .tool_arguments(self.tx, &mut self.sse_buffer, tool_call_item.parameters)
                        .await?;
                }
            }
            Err(e) => {
                error!("Tool call parsing error in messages streaming: {}", e);
            }
        }
        Ok(())
    }

    /// Phase 3: the end-of-stream parser flush: first any text still
    /// buffered as a prospective tool call that never materialized (dropping
    /// it produced fully-empty streams), then any parsed-but-unstreamed tool
    /// arguments.
    async fn flush_parsers(&mut self) -> Result<(), utils::StreamFailure> {
        if let Some(parser) = self.streaming_tool_parser.as_mut() {
            let leftover_text = parser.take_unstreamed_normal_text();
            if !leftover_text.is_empty() {
                self.blocks
                    .text_delta(self.tx, &mut self.sse_buffer, leftover_text)
                    .await?;
            }
        }

        if let Some(parser) = self.streaming_tool_parser.as_ref() {
            if let Some(mut unstreamed_items) = parser.get_unstreamed_tool_args() {
                if self.spec.first_tool_call_only {
                    utils::keep_first_tool_call_items(&mut unstreamed_items);
                }
                for tool_call_item in unstreamed_items {
                    self.has_tool_calls = true;

                    if let Some(ref name) = tool_call_item.name {
                        // Close the open block before starting the tool block.
                        let tool_call_id = utils::generate_tool_call_id(
                            &self.dispatch.model,
                            name,
                            tool_call_item.tool_index,
                            self.spec.history_tool_calls_count,
                        );
                        self.blocks
                            .open_tool(self.tx, &mut self.sse_buffer, &tool_call_id, name.clone())
                            .await?;
                    }

                    self.blocks
                        .tool_arguments(self.tx, &mut self.sse_buffer, tool_call_item.parameters)
                        .await?;
                }
            }
        }
        Ok(())
    }

    /// Phase 3.5: close any open content blocks.
    async fn close_open_blocks(&mut self) -> Result<(), utils::StreamFailure> {
        if self.blocks.thinking_open {
            StreamingProcessor::send_messages_event(
                self.tx,
                &mut self.sse_buffer,
                &MessageStreamEvent::ContentBlockStop {
                    index: self.blocks.index,
                },
            )
            .await?;
            self.blocks.index += 1;
        }

        if self.blocks.text_open {
            StreamingProcessor::send_messages_event(
                self.tx,
                &mut self.sse_buffer,
                &MessageStreamEvent::ContentBlockStop {
                    index: self.blocks.index,
                },
            )
            .await?;
            self.blocks.index += 1;
        }

        if self.blocks.tool_open {
            StreamingProcessor::send_messages_event(
                self.tx,
                &mut self.sse_buffer,
                &MessageStreamEvent::ContentBlockStop {
                    index: self.blocks.index,
                },
            )
            .await?;
        }
        Ok(())
    }

    /// Phase 4: the `message_delta` with the stop reason and the usage. A
    /// truncation stays max_tokens after a tool call started, as Chat keeps
    /// `length`: the call may be cut short.
    async fn send_message_delta(&mut self) -> Result<(), utils::StreamFailure> {
        let stop_reason = if self.finish_reason_str == "length" {
            Some(messages::StopReason::MaxTokens)
        } else if self.has_tool_calls || self.finish_reason_str == "tool_calls" {
            Some(messages::StopReason::ToolUse)
        } else if self.matched_stop.as_ref().is_some_and(Value::is_string) {
            // A stop sequence of the request's (the matched string) is `stop_sequence`; the
            // engine's own stop token (a matched id, the EOS it stops on) ends the turn, as the
            // whole message says.
            Some(messages::StopReason::StopSequence)
        } else {
            Some(messages::StopReason::EndTurn)
        };
        // Symphony's stop reason is the `Finish` event's; a stop sequence the decoder or the
        // engine matched is reported as such, as the message says.
        let stop_reason = match self.symphony_stop.take() {
            Some(messages::StopReason::EndTurn)
                if self.matched_stop.as_ref().is_some_and(Value::is_string) =>
            {
                Some(messages::StopReason::StopSequence)
            }
            Some(stop) => Some(stop),
            None => stop_reason,
        };

        let stop_sequence = if matches!(stop_reason, Some(messages::StopReason::StopSequence)) {
            self.matched_stop
                .as_ref()
                .and_then(|v| v.as_str().map(String::from))
        } else {
            None
        };

        StreamingProcessor::send_messages_event(
            self.tx,
            &mut self.sse_buffer,
            &MessageStreamEvent::MessageDelta {
                delta: MessageDelta {
                    stop_reason,
                    stop_sequence,
                    stop_details: None,
                    container: None,
                },
                usage: StreamingProcessor::final_messages_delta_usage(
                    self.completion_tokens.total(),
                    // The input count is the engine's Complete's, or the
                    // count of the chunk the gateway's own stop matched in.
                    (self.saw_complete || self.closed_by_stop).then_some(self.prompt_tokens),
                    self.spec
                        .thinking_requested
                        .then_some(self.engine_thinking_tokens.max(self.parsed_thinking_tokens)),
                ),
            },
        )
        .await?;
        Ok(())
    }

    /// Phase 5: the `message_stop`.
    async fn send_message_stop(&mut self) -> Result<(), utils::StreamFailure> {
        StreamingProcessor::send_messages_event(
            self.tx,
            &mut self.sse_buffer,
            &MessageStreamEvent::MessageStop,
        )
        .await?;
        Ok(())
    }

    /// The reservation settled with the usage the engine reported (or
    /// closed without one), and the stream's metrics recorded.
    /// `client_left` says the read ended with the client's leaving, not with the engine's end:
    /// the reservation is still settled, but no terminal router metric is recorded for it, as
    /// the HTTP router has none for a disconnect (neither a completed request nor a failure).
    async fn settle_and_record(
        self,
        reservation: Option<Arc<SharedReservationHandle>>,
        client_left: bool,
    ) {
        if let Some(handle) = reservation {
            // The usage is settled by the engine's Complete, or by the
            // gateway's own stop with the counts of the chunk it matched in.
            if self.saw_complete || self.closed_by_stop {
                handle
                    .settle_success(UsageSettlement {
                        actual_input_tokens: self.prompt_tokens,
                        completion_tokens: self.completion_tokens.total(),
                        cached_input_tokens: self.cached_tokens,
                        reasoning_tokens: self.engine_thinking_tokens,
                    })
                    .await;
            } else {
                handle.close_reserved_only().await;
            }
        }

        if client_left {
            return;
        }
        // Record metrics
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.processor.backend_type,
            model_id: &self.dispatch.model,
            endpoint: metrics_labels::ENDPOINT_MESSAGES,
            ttft: self
                .first_token_time
                .map(|t| t.duration_since(self.start_time)),
            generation_duration: self.start_time.elapsed(),
            input_tokens: Some(u64::from(self.prompt_tokens)),
            cached_tokens: Some(u64::from(self.cached_tokens)),
            output_tokens: u64::from(self.completion_tokens.total()),
        });
    }
}

#[cfg(test)]
pub(crate) mod eof_tests;
#[cfg(test)]
mod failure_tests;
#[cfg(test)]
mod local_stop_tests;
#[cfg(test)]
mod pd_usage_tests;
#[cfg(test)]
mod symphony_tests;
#[cfg(test)]
mod thinking_display_tests;
#[cfg(test)]
mod usage_chunk_tests;

#[cfg(test)]
mod delta_slicing_tests;
#[cfg(test)]
mod sse_bytes_tests;
#[cfg(test)]
mod usage_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// The frames write exactly the bytes of the serialized whole chunk
    /// structs, so a field added to a struct turns up here, not on the wire
    /// of the per-token chunks alone.
    #[test]
    fn chunk_frames_write_the_bytes_of_the_whole_structs() {
        fn event(frame: &BytesMut) -> Vec<u8> {
            frame
                .strip_prefix(b"data: ")
                .and_then(|body| body.strip_suffix(b"\n\n"))
                .expect("one SSE data event")
                .to_vec()
        }
        for fingerprint in [None, Some("v7")] {
            let choice = CompletionStreamChoice {
                text: "he said \"hi\"\n".to_string(),
                index: 1,
                logprobs: None,
                finish_reason: Some("stop".to_string()),
            };
            let frame = CompletionChunkFrame::new("cmpl-1", "model-x", 1_700_000_000, fingerprint);
            let mut buffer = BytesMut::new();
            frame.write_chunk(&mut buffer, &choice);
            let whole = CompletionStreamResponse {
                id: "cmpl-1".to_string(),
                object: "text_completion".to_string(),
                created: 1_700_000_000,
                choices: vec![choice],
                model: "model-x".to_string(),
                system_fingerprint: fingerprint.map(str::to_string),
                usage: None,
            };
            assert_eq!(
                event(&buffer),
                serde_json::to_vec(&whole).unwrap(),
                "completion, fingerprint {fingerprint:?}"
            );

            let choice = assistant_choice(0, Some("tok".to_string()), None, None);
            let frame = ChatChunkFrame::new("chatcmpl-1", "model-x", 1_700_000_000, fingerprint);
            for (usage, emit_usage_null) in [
                (None, false),
                (None, true),
                (Some(Usage::from_counts(3, 2)), false),
            ] {
                let mut buffer = BytesMut::new();
                frame.write_chunk(&mut buffer, &choice, usage.as_ref(), emit_usage_null);
                let whole = ChatCompletionStreamResponse {
                    id: "chatcmpl-1".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 1_700_000_000,
                    model: "model-x".to_string(),
                    system_fingerprint: fingerprint.map(str::to_string),
                    choices: vec![choice.clone()],
                    usage: usage.clone(),
                };
                assert_eq!(
                    event(&buffer),
                    serde_json::to_vec(&ChatChunkWithUsage::new(&whole, emit_usage_null)).unwrap(),
                    "chat, fingerprint {fingerprint:?}, usage {}, placeholder {emit_usage_null}",
                    usage.is_some()
                );
            }
        }
    }

    #[test]
    fn continuous_chat_usage_tracks_cumulative_counts_and_shared_prompt() {
        use smg_grpc_client::sglang_proto as proto;

        let mut tracker = ChatStreamUsage::default();
        for (index, completion_tokens, reasoning_tokens) in [(0, 2, 1), (0, 5, 3), (1, 4, 2)] {
            tracker.record_chunk(&ProtoGenerateStreamChunk::Sglang(
                proto::GenerateStreamChunk {
                    index,
                    prompt_tokens: 10,
                    cached_tokens: 8,
                    completion_tokens,
                    reasoning_tokens,
                    token_ids: vec![1], // Incremental IDs must not replace cumulative counts.
                    ..Default::default()
                },
            ));
        }
        let usage = serde_json::to_value(tracker.snapshot(0)).unwrap();
        assert_eq!(usage["prompt_tokens"], 10);
        assert_eq!(usage["completion_tokens"], 9);
        assert_eq!(usage["total_tokens"], 19);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 8);
        assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 5);

        tracker.record_complete(&ProtoGenerateComplete::Sglang(proto::GenerateComplete {
            index: 0,
            prompt_tokens: 10,
            cached_tokens: 8,
            completion_tokens: 6,
            reasoning_tokens: 3,
            spec_accepted_tokens: 2,
            spec_draft_tokens: 3,
            ..Default::default()
        }));
        let usage = tracker.snapshot(0);
        assert_eq!(usage.completion_tokens, 10);
        let details = usage.completion_tokens_details.unwrap();
        assert_eq!(details.accepted_prediction_tokens, Some(2));
        assert_eq!(details.rejected_prediction_tokens, Some(1));
    }

    #[test]
    fn continuous_chat_usage_accumulates_delta_ids_without_double_counting_complete() {
        use smg_grpc_client::vllm_proto as proto;

        let mut tracker = ChatStreamUsage::default();
        for ids in [vec![1, 2], vec![3]] {
            tracker.record_chunk(&ProtoGenerateStreamChunk::Vllm(
                proto::GenerateStreamChunk {
                    prompt_tokens: 10,
                    token_ids: ids,
                    ..Default::default()
                },
            ));
        }
        assert_eq!(tracker.snapshot(0).completion_tokens, 3);
        tracker.record_complete(&ProtoGenerateComplete::Vllm(Box::new(
            proto::GenerateComplete {
                prompt_tokens: 10,
                completion_tokens: 3,
                ..Default::default()
            },
        )));
        assert_eq!(tracker.snapshot(0).completion_tokens, 3);
    }

    /// Wire contract: cache counters serialize as integer zeros, never null,
    /// in both the message_start skeleton and the final message_delta usage.
    #[test]
    fn messages_usage_cache_counters_serialize_as_integer_zeros() {
        let start = serde_json::to_value(StreamingProcessor::initial_messages_usage()).unwrap();
        assert_eq!(start["input_tokens"], 0);
        assert_eq!(start["output_tokens"], 0);
        assert_eq!(start["cache_creation_input_tokens"], 0);
        assert_eq!(start["cache_read_input_tokens"], 0);

        let delta = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
            15,
            Some(25),
            None,
        ))
        .unwrap();
        assert_eq!(delta["output_tokens"], 15);
        assert_eq!(delta["input_tokens"], 25);
        assert_eq!(delta["cache_creation_input_tokens"], 0);
        assert_eq!(delta["cache_read_input_tokens"], 0);
    }

    /// The `message_start` skeleton carries every key the public API writes
    /// on a message's usage; the final delta carries the thinking tokens of a
    /// turn that ran with thinking and no such key otherwise.
    #[test]
    fn messages_usage_always_present_keys_and_thinking_tokens() {
        let start = serde_json::to_value(StreamingProcessor::initial_messages_usage()).unwrap();
        assert_eq!(
            start["cache_creation"],
            serde_json::json!({"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0})
        );
        assert_eq!(start["service_tier"], "standard");
        assert_eq!(start["inference_geo"], "not_available");

        let plain = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
            15,
            Some(25),
            None,
        ))
        .unwrap();
        assert!(plain.get("output_tokens_details").is_none());
        let thinking = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
            15,
            Some(25),
            Some(9),
        ))
        .unwrap();
        assert_eq!(thinking["output_tokens_details"]["thinking_tokens"], 9);
    }

    /// A clean EOF without a `Complete` message has no authoritative prompt
    /// count: `input_tokens` must serialize as null, not a fabricated zero.
    #[test]
    fn message_delta_input_tokens_null_without_authoritative_usage() {
        let delta = serde_json::to_value(StreamingProcessor::final_messages_delta_usage(
            15, None, None,
        ))
        .unwrap();
        assert!(delta.get("input_tokens").is_some_and(Value::is_null));
        assert_eq!(delta["cache_creation_input_tokens"], 0);
    }

    /// Tokenizer whose decode always fails, simulating a broken deployment
    /// (corrupt or mismatched tokenizer files).
    struct FailingTokenizer {
        special_tokens: llm_tokenizer::SpecialTokens,
    }

    impl llm_tokenizer::Encoder for FailingTokenizer {
        fn encode(
            &self,
            _input: &str,
            _add_special_tokens: bool,
        ) -> anyhow::Result<llm_tokenizer::Encoding> {
            Err(anyhow::anyhow!("encode is not supported"))
        }

        fn encode_batch(
            &self,
            _inputs: &[&str],
            _add_special_tokens: bool,
        ) -> anyhow::Result<Vec<llm_tokenizer::Encoding>> {
            Err(anyhow::anyhow!("encode_batch is not supported"))
        }
    }

    impl llm_tokenizer::Decoder for FailingTokenizer {
        fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
            Err(anyhow::anyhow!("tokenizer decode failed"))
        }

        fn finish_decode(
            &self,
            _ids: &mut Vec<u32>,
            _prefix: &mut String,
            _prefix_index: &mut usize,
            _skip_special_tokens: bool,
        ) -> anyhow::Result<llm_tokenizer::DecodeStep> {
            Err(anyhow::anyhow!("pending token decode failed"))
        }
    }

    impl Tokenizer for FailingTokenizer {
        fn vocab_size(&self) -> usize {
            0
        }

        fn get_special_tokens(&self) -> &llm_tokenizer::SpecialTokens {
            &self.special_tokens
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }

        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn process_chunk_tokens_propagates_decode_errors() {
        // A decode error must surface instead of being swallowed as `Held`,
        // which would drop the text and let a configured stop silently miss.
        let tokenizer = Arc::new(FailingTokenizer {
            special_tokens: llm_tokenizer::SpecialTokens::default(),
        });
        let config = llm_tokenizer::StopSequenceConfig::default().with_stop_sequence("STOP");
        let mut decoder = StopSequenceDecoder::new(tokenizer, config, false);

        let result = StreamingProcessor::process_chunk_tokens(&mut decoder, &[1, 2]);

        let err = result.expect_err("decode failure must propagate");
        assert!(err.contains("Stop decoder failed to process token"));
    }

    #[test]
    fn finish_preserves_pending_decode_errors() {
        let tokenizer = Arc::new(FailingTokenizer {
            special_tokens: llm_tokenizer::SpecialTokens::default(),
        });
        let mut decoder = StopSequenceDecoder::new(
            tokenizer,
            llm_tokenizer::StopSequenceConfig::default(),
            false,
        );
        let error = decoder
            .finish()
            .expect_err("EOF decode error must propagate");
        assert!(error.to_string().contains("pending token decode failed"));
    }
}
