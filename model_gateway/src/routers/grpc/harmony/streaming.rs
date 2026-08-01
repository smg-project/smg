//! Harmony streaming response processor

use std::{
    collections::{hash_map::Entry::Vacant, HashMap, HashSet},
    io,
    sync::Arc,
    time::Instant,
};

use axum::response::Response;
use bytes::Bytes;
use openai_protocol::{
    chat::{
        ChatCompletionRequest, ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice,
    },
    common::{ChatLogProbs, FunctionCallDelta, ToolCall, ToolCallDelta, Usage},
    responses::{
        InputTokensDetails, OutputTokensDetails, ResponseStatus, ResponseUsage, ResponsesResponse,
        ResponsesUsage,
    },
};
use serde_json::json;
use smg_mcp::{McpToolSession, DEFAULT_SERVER_LABEL};
use tokio::sync::mpsc;
use tracing::{debug, error};

use super::{
    builder::{convert_harmony_logprobs, try_harmony_encoding},
    processor::ResponsesIterationResult,
    stop::TextStopScanner,
    types::HarmonyChannelDelta,
    HarmonyParserAdapter,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics, StreamingMetricsParams},
    rate_limit::{SharedReservationHandle, UsageSettlement},
    routers::{
        common::{
            openai_bridge::{self, descriptor, FormatRegistry, ResponseFormat},
            sse::SseEncoder,
        },
        grpc::{
            common::{
                response_formatting::CompletionTokenTracker,
                responses::{
                    build_sse_response,
                    streaming::{
                        attach_mcp_server_label, OutputItemKind, ResponseStreamEventEmitter,
                    },
                },
            },
            context,
            proto_wrapper::{ProtoResponseVariant, ProtoStream},
            utils,
        },
    },
};

/// Whether a tool call of this `ResponseFormat` streams its arguments via
/// `mcp_call.arguments.delta` / `function_call.arguments.delta` events.
///
/// Hosted built-in tools (`web_search_call`, `code_interpreter_call`,
/// `file_search_call`, `image_generation_call`) instead surface their
/// progress through structured events emitted by the shared
/// [`ResponseStreamEventEmitter`] helpers (`emit_tool_call_in_progress`,
/// `emit_tool_call_searching`, `emit_tool_call_completed` — plus
/// `emit_image_generation_partial_image` for the image_generation
/// partial-image frame). Those builtins therefore skip argument
/// streaming here.
///
/// `None` (plain function tools) and `Some(Passthrough)` (MCP `mcp_call`)
/// are the only formats that stream arguments through this router.
fn streams_arguments(response_format: Option<&ResponseFormat>) -> bool {
    response_format
        .map(|f| descriptor(*f).streams_arguments)
        .unwrap_or(true)
}

/// Processor for streaming Harmony responses
///
/// Returns an SSE stream that parses Harmony tokens incrementally and
/// emits ChatCompletionChunk events for streaming responses.
pub(crate) struct HarmonyStreamingProcessor;

impl HarmonyStreamingProcessor {
    /// Create a new Harmony streaming processor
    pub fn new() -> Self {
        Self
    }

    /// Process a streaming Harmony Chat Completion response
    ///
    /// Returns an SSE response with streaming token updates.
    ///
    /// Note: Caller should attach load guards to the returned response using
    /// `WorkerLoadGuard::attach_to_response()` for proper RAII lifecycle management.
    #[expect(
        clippy::unused_self,
        reason = "takes Arc<Self> for API consistency with other streaming processors"
    )]
    #[expect(
        clippy::disallowed_methods,
        reason = "streaming tasks are fire-and-forget by design; client disconnect terminates them"
    )]
    /// `router_stop_strings` is non-empty only when the router must enforce
    /// string `stop` sequences itself (direct-ZMQ backends: the engine sees
    /// token ids only).
    pub fn process_streaming_chat_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: context::DispatchMetadata,
        router_stop_strings: Vec<String>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Response {
        // Create SSE channel
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        // Spawn background task based on execution mode
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                tokio::spawn(async move {
                    let result = Self::process_single_stream(
                        stream,
                        dispatch,
                        chat_request,
                        &tx,
                        router_stop_strings,
                        reservation,
                    )
                    .await;

                    if let Err(e) = result {
                        error!("Harmony streaming error: {}", e);
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(SseEncoder::done()));
                });
            }
            context::ExecutionResult::PrefillDecode {
                // TODO(#1781 follow-up): thread pd_timing for honest PD TTFT
                prefill,
                decode,
                ..
            } => {
                tokio::spawn(async move {
                    let result = Self::process_prefill_decode_stream(
                        prefill,
                        *decode,
                        dispatch,
                        chat_request,
                        &tx,
                        router_stop_strings,
                        reservation,
                    )
                    .await;

                    if let Err(e) = result {
                        error!("Harmony prefill/decode streaming error: {}", e);
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(SseEncoder::done()));
                });
            }
            context::ExecutionResult::Embedding { .. } => {
                error!("Harmony streaming not supported for embeddings");
                utils::send_error_sse(
                    &tx,
                    "Embeddings not supported in Harmony streaming",
                    "invalid_request_error",
                );
                let _ = tx.send(Ok(SseEncoder::done()));
            }
            // Batch results exist only on the completions pipeline.
            context::ExecutionResult::Batch { .. } => {
                error!("Harmony streaming not supported for batched results");
                utils::send_error_sse(
                    &tx,
                    "Batched results not supported in Harmony streaming",
                    "invalid_request_error",
                );
                let _ = tx.send(Ok(SseEncoder::done()));
            }
        }

        // Return SSE response
        build_sse_response(rx)
    }

    /// Process streaming chunks from a single stream
    async fn process_single_stream(
        grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        original_request: Arc<ChatCompletionRequest>,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        router_stop_strings: Vec<String>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), String> {
        let mut prompt_tokens = HashMap::new();
        let mut cached_tokens = HashMap::new();
        Self::process_chat_decode_stream(
            grpc_stream,
            &dispatch,
            &original_request,
            tx,
            &mut prompt_tokens,
            &mut cached_tokens,
            &router_stop_strings,
            reservation,
        )
        .await
    }

    /// Process streaming chunks from prefill/decode streams (prefill + decode)
    async fn process_prefill_decode_stream(
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        original_request: Arc<ChatCompletionRequest>,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        router_stop_strings: Vec<String>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), String> {
        // Phase 1: Process prefill stream (collect metadata)
        let mut prompt_tokens: HashMap<u32, u32> = HashMap::new();
        let mut cached_tokens: HashMap<u32, u32> = HashMap::new();

        while let Some(result) = prefill_stream.next().await {
            let response = result.map_err(|e| format!("Prefill stream error: {}", e.message()))?;

            if let ProtoResponseVariant::Complete(complete_wrapper) = response.into_response() {
                prompt_tokens.insert(complete_wrapper.index(), complete_wrapper.prompt_tokens());
                cached_tokens.insert(complete_wrapper.index(), complete_wrapper.cached_tokens());
            }
        }

        // Phase 2: Decode (shared helper)
        Self::process_chat_decode_stream(
            decode_stream,
            &dispatch,
            &original_request,
            tx,
            &mut prompt_tokens,
            &mut cached_tokens,
            &router_stop_strings,
            reservation,
        )
        .await?;

        // Mark prefill stream completed AFTER decode succeeds
        // This ensures that if client disconnects during decode, BOTH streams send abort
        prefill_stream.mark_completed();
        Ok(())
    }

    /// Process the decode phase of a Chat Completion stream.
    ///
    /// Shared between single-stream and prefill/decode stream modes. The `prompt_tokens`
    /// and `cached_tokens` maps may be pre-populated from a prefill phase
    /// (prefill/decode stream) or empty (single stream). Values from `Complete` messages
    /// are inserted only if not already present.
    async fn process_chat_decode_stream(
        mut decode_stream: ProtoStream,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        prompt_tokens: &mut HashMap<u32, u32>,
        cached_tokens: &mut HashMap<u32, u32>,
        router_stop_strings: &[String],
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), String> {
        // Timing for metrics
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Per-index state management (for n>1 support)
        let mut parsers: HashMap<u32, HarmonyParserAdapter> = HashMap::new();
        let mut is_firsts: HashMap<u32, bool> = HashMap::new();
        let mut matched_stops: HashMap<u32, Option<serde_json::Value>> = HashMap::new();
        // Router-enforced string stops (direct-ZMQ): per-index, per-channel
        // scanners. Once an index stops, its further deltas are swallowed and
        // the engine's own Complete is not re-emitted.
        let mut analysis_scanners: HashMap<u32, TextStopScanner> = HashMap::new();
        let mut final_scanners: HashMap<u32, TextStopScanner> = HashMap::new();
        let mut router_stopped: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut completion_tokens = CompletionTokenTracker::new();
        // Indices that received a *decode* `Complete` -- unlike `prompt_tokens`
        // (which may already be populated from the prefill phase before this
        // loop even starts, in PD mode), this is only ever set from this
        // stream's own Complete messages, so it can't be fooled by prefill
        // data into thinking decode produced authoritative usage it didn't.
        let mut decode_completed_indices: HashSet<u32> = HashSet::new();
        // Reusable SSE encoder shared across every chunk emitted for this stream.
        let mut encoder = SseEncoder::new();

        let stream_options = &original_request.stream_options;

        // Process stream
        while let Some(result) = decode_stream.next().await {
            let response = result.map_err(|e| format!("Stream error: {}", e.message()))?;

            match response.into_response() {
                ProtoResponseVariant::Chunk(chunk_wrapper) => {
                    let index = chunk_wrapper.index();

                    // Track first token time for TTFT metric
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                    }

                    // Initialize parser for this index if needed
                    if let Vacant(e) = parsers.entry(index) {
                        e.insert(
                            HarmonyParserAdapter::new()
                                .map_err(|e| format!("Failed to create parser: {e}"))?,
                        );
                        is_firsts.insert(index, true);
                    }

                    completion_tokens.record_chunk(&chunk_wrapper);

                    // Convert logprobs if present and requested
                    let chunk_logprobs = if original_request.logprobs {
                        let encoding = try_harmony_encoding()?;
                        chunk_wrapper
                            .output_logprobs()
                            .map(|lp| convert_harmony_logprobs(encoding, &lp))
                    } else {
                        None
                    };

                    // Parse chunk via Harmony parser
                    let parser = parsers
                        .get_mut(&index)
                        .ok_or("Parser not found for index")?;

                    let delta_result = parser
                        .parse_chunk(chunk_wrapper.token_ids())
                        .map_err(|e| format!("Parse error: {e}"))?;

                    // Emit SSE event if there's a delta
                    if let Some(mut delta) = delta_result {
                        if router_stopped.contains(&index) {
                            continue;
                        }
                        let mut stop_matched: Option<String> = None;
                        if !router_stop_strings.is_empty() {
                            if let Some(text) = delta.analysis_delta.take() {
                                let scanner = analysis_scanners.entry(index).or_insert_with(|| {
                                    TextStopScanner::new(router_stop_strings.to_vec())
                                });
                                let scan = scanner.push(&text);
                                delta.analysis_delta = (!scan.emit.is_empty()).then_some(scan.emit);
                                if scan.stopped {
                                    stop_matched = scanner.matched().map(str::to_string);
                                    delta.final_delta = None;
                                    delta.commentary_delta = None;
                                }
                            }
                            if stop_matched.is_none() {
                                if let Some(text) = delta.final_delta.take() {
                                    let scanner =
                                        final_scanners.entry(index).or_insert_with(|| {
                                            TextStopScanner::new(router_stop_strings.to_vec())
                                        });
                                    let scan = scanner.push(&text);
                                    delta.final_delta =
                                        (!scan.emit.is_empty()).then_some(scan.emit);
                                    if scan.stopped {
                                        stop_matched = scanner.matched().map(str::to_string);
                                        delta.commentary_delta = None;
                                    }
                                }
                            }
                        }

                        let has_payload = delta.analysis_delta.is_some()
                            || delta.final_delta.is_some()
                            || delta.commentary_delta.is_some();
                        let is_first = is_firsts.get(&index).copied().unwrap_or(false);
                        if has_payload || is_first {
                            Self::emit_chunk_delta(
                                &delta,
                                index,
                                is_first,
                                dispatch,
                                original_request,
                                tx,
                                &mut encoder,
                                chunk_logprobs,
                            )?;

                            if is_first {
                                is_firsts.insert(index, false);
                            }
                        }

                        // A router-side stop fired: emit the final chunk now
                        // and swallow the rest of this index's stream (the
                        // engine keeps generating until its own limits).
                        if let Some(stop) = stop_matched {
                            Self::emit_final_chunk(
                                index,
                                "stop",
                                Some(&serde_json::Value::String(stop)),
                                dispatch,
                                original_request,
                                tx,
                                &mut encoder,
                            )?;
                            router_stopped.insert(index);
                        }
                    }
                }
                ProtoResponseVariant::Complete(complete_wrapper) => {
                    let index = complete_wrapper.index();
                    decode_completed_indices.insert(index);

                    // Store final metadata
                    matched_stops.insert(index, complete_wrapper.matched_stop_json());
                    prompt_tokens
                        .entry(index)
                        .or_insert_with(|| complete_wrapper.prompt_tokens());
                    completion_tokens.record_complete(&complete_wrapper);
                    cached_tokens
                        .entry(index)
                        .or_insert_with(|| complete_wrapper.cached_tokens());

                    // Finalize parser and emit final chunk
                    if let Some(parser) = parsers.get_mut(&index) {
                        let matched_stop = matched_stops.get(&index).and_then(|m| m.clone());

                        let final_output =
                            parser.finalize(complete_wrapper.finish_reason().to_string());

                        // A router-side stop already closed this choice.
                        if router_stopped.contains(&index) {
                            continue;
                        }

                        // Release scanner-held text that never became a match.
                        let flushed = HarmonyChannelDelta {
                            analysis_delta: analysis_scanners
                                .get_mut(&index)
                                .map(TextStopScanner::flush)
                                .filter(|s| !s.is_empty()),
                            commentary_delta: None,
                            final_delta: final_scanners
                                .get_mut(&index)
                                .map(TextStopScanner::flush)
                                .filter(|s| !s.is_empty()),
                            is_final: false,
                        };
                        if flushed.analysis_delta.is_some() || flushed.final_delta.is_some() {
                            Self::emit_chunk_delta(
                                &flushed,
                                index,
                                false,
                                dispatch,
                                original_request,
                                tx,
                                &mut encoder,
                                None,
                            )?;
                        }

                        Self::emit_final_chunk(
                            index,
                            &final_output.finish_reason,
                            matched_stop.as_ref(),
                            dispatch,
                            original_request,
                            tx,
                            &mut encoder,
                        )?;
                    }
                }
                ProtoResponseVariant::None => {}
            }
        }

        // Mark stream as completed successfully to prevent abort on drop
        decode_stream.mark_completed();

        // Compute totals once for both usage chunk and metrics. Every `n>1`
        // choice shares one prompt; each Complete reports that same full
        // length, so max (not sum) is the actual prompt cost. cached_tokens
        // is a property of that same shared prompt, not of the individual
        // completion, so it takes the same treatment.
        let total_prompt: u32 = prompt_tokens.values().copied().max().unwrap_or(0);
        let total_completion: u32 = completion_tokens.total();
        let total_cached: u32 = cached_tokens.values().copied().max().unwrap_or(0);

        if let Some(handle) = reservation {
            // A clean decode EOF with fewer decode `Complete` messages than
            // this request's `n>1` choices has only partial usage -- settling
            // with that would understate the real cost. Deliberately checked
            // against `decode_completed_indices`, not `prompt_tokens`: in PD
            // mode `prompt_tokens` can already be non-empty from the prefill
            // phase alone, which would otherwise mask a decode phase that
            // never actually finished.
            let expected_choices = original_request.n.unwrap_or(1).max(1);
            if (decode_completed_indices.len() as u32) < expected_choices {
                handle.close_reserved_only().await;
            } else {
                handle
                    .settle_success(UsageSettlement {
                        actual_input_tokens: total_prompt,
                        completion_tokens: total_completion,
                    })
                    .await;
            }
        }

        // Emit final usage if requested
        if let Some(true) = stream_options.as_ref().and_then(|so| so.include_usage) {
            Self::emit_usage_chunk(
                total_prompt,
                total_completion,
                total_cached,
                dispatch,
                original_request,
                tx,
                &mut encoder,
            )?;
        }

        // Record streaming metrics
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_HARMONY,
            model_id: &original_request.model,
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: first_token_time.map(|t| t.duration_since(start_time)),
            generation_duration: start_time.elapsed(),
            input_tokens: Some(total_prompt as u64),
            output_tokens: total_completion as u64,
        });

        Ok(())
    }

    /// Emit a chunk delta from Harmony channels
    #[expect(clippy::too_many_arguments)]
    fn emit_chunk_delta(
        delta: &HarmonyChannelDelta,
        index: u32,
        is_first: bool,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        encoder: &mut SseEncoder,
        logprobs: Option<ChatLogProbs>,
    ) -> Result<(), String> {
        // On first chunk, emit role announcement separately
        if is_first {
            let role_chunk = ChatCompletionStreamResponse::builder(
                &dispatch.request_id,
                &original_request.model,
            )
            .created(dispatch.created)
            .add_choice_role(index, "assistant")
            .maybe_system_fingerprint(dispatch.weight_version.as_deref())
            .build();

            let sse_data = encoder
                .encode_data(&role_chunk)
                .map_err(|e| format!("JSON serialization error: {e}"))?;

            tx.send(Ok(sse_data))
                .map_err(|_| "Failed to send role chunk".to_string())?;
        }

        // Emit content delta (role is always None for content chunks)
        let chat_delta = ChatMessageDelta {
            role: None,
            content: delta.final_delta.clone(),
            tool_calls: delta.commentary_delta.as_ref().map(|tc_delta| {
                vec![ToolCallDelta {
                    index: tc_delta.index as u32,
                    id: tc_delta.id.clone(),
                    tool_type: tc_delta.id.as_ref().map(|_| "function".to_string()),
                    function: tc_delta.function.as_ref().map(|f| FunctionCallDelta {
                        name: f.name.clone(),
                        arguments: f.arguments.clone(),
                    }),
                }]
            }),
            reasoning_content: delta.analysis_delta.clone(),
        };

        // Build and emit chunk
        let chunk =
            ChatCompletionStreamResponse::builder(&dispatch.request_id, &original_request.model)
                .created(dispatch.created)
                .add_choice(ChatStreamChoice {
                    index,
                    delta: chat_delta,
                    logprobs,
                    finish_reason: None,
                    matched_stop: None,
                })
                .maybe_system_fingerprint(dispatch.weight_version.as_deref())
                .build();

        let sse_data = encoder
            .encode_data(&chunk)
            .map_err(|e| format!("JSON serialization error: {e}"))?;

        tx.send(Ok(sse_data))
            .map_err(|_| "Failed to send chunk".to_string())?;

        Ok(())
    }

    /// Emit final chunk with finish_reason
    fn emit_final_chunk(
        index: u32,
        finish_reason: &str,
        matched_stop: Option<&serde_json::Value>,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        encoder: &mut SseEncoder,
    ) -> Result<(), String> {
        let chunk =
            ChatCompletionStreamResponse::builder(&dispatch.request_id, &original_request.model)
                .created(dispatch.created)
                .add_choice_finish_reason(index, finish_reason, matched_stop.cloned())
                .maybe_system_fingerprint(dispatch.weight_version.as_deref())
                .build();

        let sse_data = encoder
            .encode_data(&chunk)
            .map_err(|e| format!("JSON serialization error: {e}"))?;

        tx.send(Ok(sse_data))
            .map_err(|_| "Failed to send final chunk".to_string())?;

        Ok(())
    }

    /// Emit usage chunk at the end
    fn emit_usage_chunk(
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_tokens: u32,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        encoder: &mut SseEncoder,
    ) -> Result<(), String> {
        let usage_chunk =
            ChatCompletionStreamResponse::builder(&dispatch.request_id, &original_request.model)
                .created(dispatch.created)
                .usage(
                    Usage::from_counts(prompt_tokens, completion_tokens)
                        .with_cached_tokens(cached_tokens),
                )
                .maybe_system_fingerprint(dispatch.weight_version.as_deref())
                .build();

        let sse_data = encoder
            .encode_data(&usage_chunk)
            .map_err(|e| format!("JSON serialization error: {e}"))?;

        tx.send(Ok(sse_data))
            .map_err(|_| "Failed to send usage chunk".to_string())?;

        Ok(())
    }

    /// Process streaming chunks for Responses API iteration.
    ///
    /// When MCP context is provided (session):
    /// - MCP tools with `ResponseFormat::WebSearchCall` → `web_search_call.*` events
    /// - Other MCP tools → `mcp_call.*` events
    /// - Other tools → `function_call.*` events
    ///
    /// When no MCP context is provided, all tool calls are treated as function calls.
    pub async fn process_responses_iteration_stream(
        execution_result: context::ExecutionResult,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        session: Option<&McpToolSession<'_>>,
        format_registry: Option<&FormatRegistry>,
    ) -> Result<ResponsesIterationResult, String> {
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                debug!("Processing Responses API single stream mode");
                Self::process_decode_stream(stream, emitter, tx, session, format_registry, 0).await
            }
            context::ExecutionResult::PrefillDecode {
                // TODO(#1781 follow-up): thread pd_timing for honest PD TTFT
                prefill,
                decode,
                ..
            } => {
                debug!("Processing Responses API prefill/decode stream mode");
                Self::process_responses_prefill_decode_stream(
                    prefill,
                    *decode,
                    emitter,
                    tx,
                    session,
                    format_registry,
                )
                .await
            }
            context::ExecutionResult::Embedding { .. } => {
                Err("Embeddings not supported in Responses API streaming".to_string())
            }
            // Batch results exist only on the completions pipeline.
            context::ExecutionResult::Batch { .. } => {
                Err("Batched results not supported in Responses API streaming".to_string())
            }
        }
    }

    async fn process_responses_prefill_decode_stream(
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        session: Option<&McpToolSession<'_>>,
        format_registry: Option<&FormatRegistry>,
    ) -> Result<ResponsesIterationResult, String> {
        // Phase 1: Drain prefill stream, collecting cached_tokens from Complete messages
        let mut prefill_cached_tokens_by_index: HashMap<u32, u32> = HashMap::new();
        while let Some(result) = prefill_stream.next().await {
            let response = result.map_err(|e| format!("Prefill stream error: {}", e.message()))?;
            if let ProtoResponseVariant::Complete(complete_wrapper) = response.into_response() {
                prefill_cached_tokens_by_index
                    .insert(complete_wrapper.index(), complete_wrapper.cached_tokens());
            }
        }
        let prefill_cached_tokens: u32 = prefill_cached_tokens_by_index.values().sum();

        // Phase 2: Process decode stream
        let result = Self::process_decode_stream(
            decode_stream,
            emitter,
            tx,
            session,
            format_registry,
            prefill_cached_tokens,
        )
        .await;

        prefill_stream.mark_completed();
        result
    }

    /// Process decode stream for tool call events.
    async fn process_decode_stream(
        mut decode_stream: ProtoStream,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
        session: Option<&McpToolSession<'_>>,
        format_registry: Option<&FormatRegistry>,
        prefill_cached_tokens: u32,
    ) -> Result<ResponsesIterationResult, String> {
        let mut parser =
            HarmonyParserAdapter::new().map_err(|e| format!("Failed to create parser: {e}"))?;

        let mut has_analysis = false;
        let mut accumulated_final_text = String::new();
        let mut accumulated_tool_calls: Option<Vec<ToolCall>> = None;

        let mut has_emitted_reasoning = false;
        let mut message_output_index: Option<usize> = None;
        let mut message_item_id: Option<String> = None;
        let mut has_emitted_content_part_added = false;

        // Tool call tracking: call_index -> (output_index, item_id, response_format)
        let mut tool_call_tracking: HashMap<usize, (usize, String, Option<ResponseFormat>)> =
            HashMap::new();

        // Metadata from Complete message; seed cached_tokens from prefill phase (prefill/decode stream)
        let mut finish_reason: String;
        let mut finalized_analysis: Option<String> = None;
        let mut prompt_tokens: u32 = 0;
        let mut completion_tokens: u32 = 0;
        let mut cached_tokens: u32 = prefill_cached_tokens;
        let mut reasoning_token_count: u32 = 0;

        // Process stream
        let mut chunk_count = 0;
        while let Some(result) = decode_stream.next().await {
            chunk_count += 1;
            let response = result.map_err(|e| format!("Decode stream error: {}", e.message()))?;

            match response.into_response() {
                ProtoResponseVariant::Chunk(chunk_wrapper) => {
                    // Track token counts for vLLM (vLLM sends deltas)
                    // For SGLang, skip (SGLang sends cumulative values in Complete)
                    if chunk_wrapper.is_vllm() {
                        completion_tokens += chunk_wrapper.token_ids().len() as u32;
                    }

                    // Parse chunk via Harmony parser
                    let delta_result = parser
                        .parse_chunk(chunk_wrapper.token_ids())
                        .map_err(|e| format!("Parse error: {e}"))?;

                    // Emit SSE events if there's a delta
                    if let Some(delta) = delta_result {
                        // Analysis channel → Reasoning item (wrapper events only, emitted once)
                        if let Some(_analysis_text) = &delta.analysis_delta {
                            if !has_emitted_reasoning {
                                // Emit reasoning item (added + done in one call)
                                // Note: reasoning_content will be provided at finalize
                                emitter
                                    .emit_reasoning_item(tx, None)
                                    .map_err(|e| format!("Failed to emit reasoning item: {e}"))?;

                                has_emitted_reasoning = true;
                                has_analysis = true;
                            }
                        }

                        // Final channel → Message item (WITH text streaming)
                        if let Some(final_delta) = &delta.final_delta {
                            if !final_delta.is_empty() {
                                // Allocate message item if needed
                                if message_output_index.is_none() {
                                    let (output_index, item_id) =
                                        emitter.allocate_output_index(OutputItemKind::Message);
                                    message_output_index = Some(output_index);
                                    message_item_id = Some(item_id.clone());

                                    // Build message item structure
                                    let item = json!({
                                        "id": item_id,
                                        "type": "message",
                                        "role": "assistant",
                                        "content": []
                                    });

                                    // Emit output_item.added
                                    let event = emitter.emit_output_item_added(output_index, &item);
                                    emitter.send_event_best_effort(&event, tx);
                                }

                                let Some(output_index) = message_output_index else {
                                    continue;
                                };
                                let Some(item_id) = message_item_id.as_ref() else {
                                    continue;
                                };
                                let content_index = 0; // Single content part

                                // Emit content_part.added before first delta
                                if !has_emitted_content_part_added {
                                    let event = emitter.emit_content_part_added(
                                        output_index,
                                        item_id,
                                        content_index,
                                    );
                                    emitter.send_event_best_effort(&event, tx);
                                    has_emitted_content_part_added = true;
                                }

                                // Emit text delta
                                let event = emitter.emit_text_delta(
                                    final_delta,
                                    output_index,
                                    item_id,
                                    content_index,
                                );
                                emitter.send_event_best_effort(&event, tx);

                                accumulated_final_text.push_str(final_delta);
                            }
                        }

                        // Commentary channel → Tool call streaming
                        if let Some(tc_delta) = &delta.commentary_delta {
                            let call_index = tc_delta.index;

                            // New tool call (has id and name)
                            if let Some(call_id) = &tc_delta.id {
                                let tool_name = tc_delta
                                    .function
                                    .as_ref()
                                    .and_then(|f| f.name.as_ref())
                                    .map(|n| n.as_str())
                                    .unwrap_or("");

                                // Determine response_format based on MCP context.
                                let response_format = session.and_then(|s| {
                                    if s.has_exposed_tool(tool_name) {
                                        format_registry.map(|reg| {
                                            openai_bridge::lookup_tool_format(s, reg, tool_name)
                                        })
                                    } else {
                                        None
                                    }
                                });

                                let type_str = ResponseStreamEventEmitter::type_str_for_format(
                                    response_format.as_ref(),
                                );

                                let (output_index, item_id) =
                                    emitter.allocate_output_index_for_format(response_format);

                                tool_call_tracking.insert(
                                    call_index,
                                    (output_index, item_id.clone(), response_format),
                                );

                                // Build output_item.added event
                                let mut item = json!({
                                    "id": item_id,
                                    "type": type_str,
                                    "name": tool_name,
                                    "call_id": call_id,
                                    "arguments": "",
                                    "status": "in_progress"
                                });

                                let label = session
                                    .map(|s| s.resolve_tool_server_label(tool_name))
                                    .unwrap_or_else(|| DEFAULT_SERVER_LABEL.to_string());
                                attach_mcp_server_label(
                                    &mut item,
                                    Some(label.as_str()),
                                    response_format.as_ref(),
                                );

                                let event = emitter.emit_output_item_added(output_index, &item);
                                emitter.send_event_best_effort(&event, tx);

                                // Emit in_progress event for MCP tools
                                if let Some(fmt) = response_format {
                                    let event = emitter.emit_tool_call_in_progress(
                                        output_index,
                                        &item_id,
                                        fmt,
                                    );
                                    emitter.send_event_best_effort(&event, tx);

                                    // Emit searching/interpreting event for builtin tools
                                    if let Some(event) = emitter.emit_tool_call_searching(
                                        output_index,
                                        &item_id,
                                        fmt,
                                    ) {
                                        emitter.send_event_best_effort(&event, tx);
                                    }
                                }

                                // Emit initial arguments delta for mcp_call / function_call
                                // only. Hosted built-in tools (web_search_call,
                                // code_interpreter_call, file_search_call,
                                // image_generation_call) surface progress via
                                // the structured `*.in_progress` /
                                // `*.searching` / `*.generating` events emitted
                                // above instead of streaming their arguments.
                                if streams_arguments(response_format.as_ref()) {
                                    let event = match &response_format {
                                        Some(_) => emitter.emit_mcp_call_arguments_delta(
                                            output_index,
                                            &item_id,
                                            "",
                                        ),
                                        None => emitter.emit_function_call_arguments_delta(
                                            output_index,
                                            &item_id,
                                            "",
                                        ),
                                    };
                                    emitter.send_event_best_effort(&event, tx);
                                }
                            } else {
                                // Continuing tool call: emit arguments delta
                                if let Some((output_index, item_id, response_format)) =
                                    tool_call_tracking.get(&call_index)
                                {
                                    // Only mcp_call / function_call stream
                                    // arguments; hosted built-in tools
                                    // (web_search_call, code_interpreter_call,
                                    // file_search_call, image_generation_call)
                                    // skip argument deltas — their progress
                                    // rides on the structured events emitted
                                    // around MCP dispatch.
                                    if !streams_arguments(response_format.as_ref()) {
                                        continue;
                                    }

                                    if let Some(args) = tc_delta
                                        .function
                                        .as_ref()
                                        .and_then(|f| f.arguments.as_ref())
                                        .filter(|a| !a.is_empty())
                                    {
                                        let event = match response_format {
                                            Some(_) => emitter.emit_mcp_call_arguments_delta(
                                                *output_index,
                                                item_id,
                                                args,
                                            ),
                                            None => emitter.emit_function_call_arguments_delta(
                                                *output_index,
                                                item_id,
                                                args,
                                            ),
                                        };
                                        emitter.send_event_best_effort(&event, tx);
                                    }
                                }
                            }
                        }
                    }
                }
                ProtoResponseVariant::Complete(complete_wrapper) => {
                    // Store final metadata
                    finish_reason = complete_wrapper.finish_reason().to_string();
                    prompt_tokens = complete_wrapper.prompt_tokens();
                    // Combine decode-stream cached_tokens with any prefill cached_tokens
                    cached_tokens = cached_tokens.saturating_add(complete_wrapper.cached_tokens());
                    // For vLLM, use accumulated count (we tracked deltas above)
                    // For SGLang, use complete value (already cumulative)
                    if !complete_wrapper.is_vllm() {
                        completion_tokens = complete_wrapper.completion_tokens();
                    }

                    // Finalize parser and get complete output
                    // Responses API: no user-specified stop sequences
                    let final_output = parser.finalize(finish_reason.clone());

                    // Store finalized output for later use
                    finalized_analysis = final_output.analysis;
                    accumulated_tool_calls = final_output.commentary;
                    reasoning_token_count = final_output.reasoning_token_count;

                    // Complete all tool calls if we have commentary
                    if let Some(ref tool_calls) = accumulated_tool_calls {
                        for (call_idx, tool_call) in tool_calls.iter().enumerate() {
                            if let Some((output_index, item_id, response_format)) =
                                tool_call_tracking.get(&call_idx)
                            {
                                let tool_name = &tool_call.function.name;
                                let args_str =
                                    tool_call.function.arguments.as_deref().unwrap_or("");

                                // Emit arguments.done for mcp_call /
                                // function_call only. Hosted built-in tools
                                // (web_search_call, code_interpreter_call,
                                // file_search_call, image_generation_call)
                                // close out through the `*.completed`
                                // structured event emitted below.
                                if streams_arguments(response_format.as_ref()) {
                                    let event = match response_format {
                                        Some(_) => emitter.emit_mcp_call_arguments_done(
                                            *output_index,
                                            item_id,
                                            args_str,
                                        ),
                                        None => emitter.emit_function_call_arguments_done(
                                            *output_index,
                                            item_id,
                                            args_str,
                                        ),
                                    };
                                    emitter.send_event_best_effort(&event, tx);
                                }

                                // Emit completed event for MCP tools
                                if let Some(fmt) = *response_format {
                                    let event = emitter.emit_tool_call_completed(
                                        *output_index,
                                        item_id,
                                        fmt,
                                    );
                                    emitter.send_event_best_effort(&event, tx);
                                }

                                // Determine type string for JSON
                                let type_str = ResponseStreamEventEmitter::type_str_for_format(
                                    response_format.as_ref(),
                                );

                                let mut item = json!({
                                    "id": item_id,
                                    "type": type_str,
                                    "name": tool_name,
                                    "call_id": &tool_call.id,
                                    "arguments": args_str,
                                    "status": "completed"
                                });

                                let label = session
                                    .map(|s| s.resolve_tool_server_label(tool_name))
                                    .unwrap_or_else(|| DEFAULT_SERVER_LABEL.to_string());
                                attach_mcp_server_label(
                                    &mut item,
                                    Some(label.as_str()),
                                    response_format.as_ref(),
                                );

                                let event = emitter.emit_output_item_done(*output_index, &item);
                                emitter.complete_output_item(*output_index);
                                emitter.send_event_best_effort(&event, tx);
                            }
                        }
                    }

                    // Close message item if we opened one
                    if let (Some(output_index), Some(item_id)) =
                        (message_output_index, message_item_id.as_ref())
                    {
                        let content_index = 0;

                        // Emit text_done
                        let event = emitter.emit_text_done(output_index, item_id, content_index);
                        emitter.send_event_best_effort(&event, tx);

                        // Emit content_part.done
                        let event =
                            emitter.emit_content_part_done(output_index, item_id, content_index);
                        emitter.send_event_best_effort(&event, tx);

                        // Emit output_item.done
                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": accumulated_final_text.clone()
                            }]
                        });
                        let event = emitter.emit_output_item_done(output_index, &item);

                        // Mark as completed before sending (so it's included in final output even if send fails)
                        emitter.complete_output_item(output_index);

                        emitter.send_event_best_effort(&event, tx);
                    }
                }
                ProtoResponseVariant::None => {}
            }
        }

        debug!(
            "Stream loop ended. Total chunks received: {}, has_analysis: {}, tool_calls: {}, final_text_len: {}",
            chunk_count,
            has_analysis,
            accumulated_tool_calls.as_ref().map(|tc| tc.len()).unwrap_or(0),
            accumulated_final_text.len()
        );

        // Extract tool calls from completed messages or incomplete commentary
        if chunk_count > 0 && accumulated_tool_calls.is_none() {
            let messages = parser.get_messages();

            // Try extracting from completed messages first
            let (analysis_opt, commentary_opt, final_text_extracted) =
                HarmonyParserAdapter::parse_messages(&messages);
            accumulated_tool_calls.clone_from(&commentary_opt);

            // If no tool calls found, check for incomplete commentary in parser state
            if accumulated_tool_calls.is_none() {
                accumulated_tool_calls = parser.extract_incomplete_commentary();
            }

            debug!(
                "Tool call extraction: completed_msgs={}, tool_calls={}, has_analysis={}, final_text_len={}",
                messages.len(),
                accumulated_tool_calls.as_ref().map(|tc| tc.len()).unwrap_or(0),
                analysis_opt.is_some(),
                final_text_extracted.len()
            );

            // Complete any pending tool calls with data from completed messages
            if let Some(ref tool_calls) = accumulated_tool_calls {
                for (call_idx, tool_call) in tool_calls.iter().enumerate() {
                    if let Some((output_index, item_id, response_format)) =
                        tool_call_tracking.get(&call_idx)
                    {
                        let tool_name = &tool_call.function.name;
                        let args_str = tool_call.function.arguments.as_deref().unwrap_or("");

                        // Emit arguments.done for mcp_call / function_call
                        // only. Hosted built-in tools (web_search_call,
                        // code_interpreter_call, file_search_call,
                        // image_generation_call) close out through the
                        // `*.completed` structured event emitted below.
                        if streams_arguments(response_format.as_ref()) {
                            let event = match response_format {
                                Some(_) => emitter.emit_mcp_call_arguments_done(
                                    *output_index,
                                    item_id,
                                    args_str,
                                ),
                                None => emitter.emit_function_call_arguments_done(
                                    *output_index,
                                    item_id,
                                    args_str,
                                ),
                            };
                            emitter.send_event_best_effort(&event, tx);
                        }

                        // Emit completed event for MCP tools
                        if let Some(fmt) = *response_format {
                            let event =
                                emitter.emit_tool_call_completed(*output_index, item_id, fmt);
                            emitter.send_event_best_effort(&event, tx);
                        }

                        let type_str = ResponseStreamEventEmitter::type_str_for_format(
                            response_format.as_ref(),
                        );

                        let mut item = json!({
                            "id": item_id,
                            "type": type_str,
                            "name": tool_name,
                            "call_id": &tool_call.id,
                            "arguments": args_str,
                            "status": "completed"
                        });

                        let label = session
                            .map(|s| s.resolve_tool_server_label(tool_name))
                            .unwrap_or_else(|| DEFAULT_SERVER_LABEL.to_string());
                        attach_mcp_server_label(
                            &mut item,
                            Some(label.as_str()),
                            response_format.as_ref(),
                        );

                        let event = emitter.emit_output_item_done(*output_index, &item);
                        emitter.complete_output_item(*output_index);
                        emitter.send_event_best_effort(&event, tx);
                    }
                }
            }
        }

        // Mark stream as completed successfully to prevent abort on drop
        decode_stream.mark_completed();

        // Return result based on whether tool calls were found
        if let Some(tool_calls) = accumulated_tool_calls {
            if !tool_calls.is_empty() {
                let analysis_content = if has_analysis {
                    finalized_analysis
                } else {
                    None
                };

                return Ok(ResponsesIterationResult::ToolCallsFound {
                    tool_calls,
                    analysis: analysis_content,
                    partial_text: accumulated_final_text,
                    usage: Usage::from_counts(prompt_tokens, completion_tokens)
                        .with_cached_tokens(cached_tokens)
                        .with_reasoning_tokens(reasoning_token_count),
                    request_id: emitter.response_id.clone(),
                });
            }
        }

        // For streaming, we don't build the full ResponsesResponse here
        // The caller will build it from the SSE events
        // Return a placeholder Completed result (caller ignores these fields in streaming mode)
        Ok(ResponsesIterationResult::Completed {
            response: Box::new(
                ResponsesResponse::builder(&emitter.response_id, "")
                    .status(ResponseStatus::Completed)
                    .usage(ResponsesUsage::Modern(ResponseUsage {
                        input_tokens: prompt_tokens,
                        output_tokens: completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                        input_tokens_details: if cached_tokens > 0 {
                            Some(InputTokensDetails { cached_tokens })
                        } else {
                            None
                        },
                        output_tokens_details: if reasoning_token_count > 0 {
                            Some(OutputTokensDetails {
                                reasoning_tokens: reasoning_token_count,
                            })
                        } else {
                            None
                        },
                    }))
                    .build(),
            ),
            usage: Usage::from_counts(prompt_tokens, completion_tokens)
                .with_cached_tokens(cached_tokens)
                .with_reasoning_tokens(reasoning_token_count),
        })
    }
}

impl Default for HarmonyStreamingProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time exhaustiveness anchor.
    ///
    /// This helper exists solely to force every [`ResponseFormat`] variant
    /// to flow through a non-wildcard `match`. If a new variant is added to
    /// [`ResponseFormat`] without being classified, this function fails to
    /// compile — which in turn breaks `streams_arguments_explicit_variants`
    /// below, since both helpers iterate the same variant set.
    ///
    /// Intentionally *does not* call [`streams_arguments`] — this mirrors
    /// the production classifier so a drift between the two is a separate
    /// failure (runtime assertion miss) from a missing variant (compile
    /// error here).
    fn expected_streams_arguments(format: ResponseFormat) -> bool {
        match format {
            ResponseFormat::Passthrough => true,
            ResponseFormat::WebSearchCall
            | ResponseFormat::CodeInterpreterCall
            | ResponseFormat::FileSearchCall
            | ResponseFormat::ImageGenerationCall => false,
        }
    }

    // Locks the `streams_arguments` classification so the Harmony router
    // keeps treating hosted built-in tools — including `image_generation`
    // — as structured-event emitters rather than argument streamers.
    //
    // Every `ResponseFormat` variant is named explicitly (no `_` arm, no
    // iteration over a hand-maintained array), so adding a new variant
    // fails to compile in `expected_streams_arguments` above AND in every
    // explicit `let ... = ResponseFormat::X;` binding here — which in
    // turn ensures the production `streams_arguments` classifier must
    // also be updated to compile.
    #[test]
    fn streams_arguments_explicit_variants() {
        // `None` (plain function tool) streams arguments.
        assert!(streams_arguments(None), "function_call should stream args");

        // `Some(Passthrough)` (mcp_call) streams arguments.
        let passthrough = ResponseFormat::Passthrough;
        assert!(
            streams_arguments(Some(&passthrough)),
            "mcp_call (Passthrough) should stream args",
        );
        assert!(expected_streams_arguments(passthrough));

        // Hosted built-ins do *not* stream arguments — they surface
        // progress via structured `*.in_progress` / `*.searching` /
        // `*.generating` / `*.completed` events from the shared emitter.
        let web_search = ResponseFormat::WebSearchCall;
        assert!(!streams_arguments(Some(&web_search)));
        assert!(!expected_streams_arguments(web_search));

        let code_interpreter = ResponseFormat::CodeInterpreterCall;
        assert!(!streams_arguments(Some(&code_interpreter)));
        assert!(!expected_streams_arguments(code_interpreter));

        let file_search = ResponseFormat::FileSearchCall;
        assert!(!streams_arguments(Some(&file_search)));
        assert!(!expected_streams_arguments(file_search));

        let image_generation = ResponseFormat::ImageGenerationCall;
        assert!(
            !streams_arguments(Some(&image_generation)),
            "image_generation_call must ride the structured-event path",
        );
        assert!(!expected_streams_arguments(image_generation));
    }
}
