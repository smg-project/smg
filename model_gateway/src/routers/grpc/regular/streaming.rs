//! Streaming response processor for gRPC routers
//!
//! This module contains shared streaming logic for both Regular and PD router.

use std::{
    collections::{HashMap, HashSet},
    io,
    sync::Arc,
    time::Instant,
};

use axum::response::Response;
use bytes::Bytes;
use futures::future::try_join_all;
use llm_tokenizer::{
    stop::{SequenceDecoderOutput, StopSequenceDecoder},
    traits::Tokenizer,
};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionStreamResponse},
    common::{
        FunctionCallDelta, StringOrArray, Tool, ToolCallDelta, ToolChoice, ToolChoiceValue, Usage,
    },
    completion::{CompletionRequest, CompletionStreamChoice, CompletionStreamResponse},
    generate::GenerateRequest,
    messages::{
        self, ContentBlock, ContentBlockDelta, CreateMessageRequest, Message, MessageDelta,
        MessageDeltaUsage, MessageStreamEvent,
    },
};
use reasoning_parser::{ParserFactory as ReasoningParserFactory, ParserResult, ReasoningParser};
use serde_json::{json, Value};
use tokio::sync::{mpsc, mpsc::UnboundedSender};
use tool_parser::{ParserFactory as ToolParserFactory, StreamingParseResult, ToolParser};
use tracing::{debug, error, warn};

use crate::{
    observability::metrics::{metrics_labels, Metrics, StreamingMetricsParams},
    rate_limit::{SharedReservationHandle, UsageSettlement},
    routers::{
        common::sse::SseEncoder,
        grpc::{
            common::{response_formatting::CompletionTokenTracker, responses::build_sse_response},
            context,
            proto_wrapper::{ProtoResponseVariant, ProtoStream},
            utils,
            utils::message_utils,
        },
    },
};

/// One backend stream of a `/v1/completions` request. Batched requests fan
/// out into several, each remapped by a prompt-major choice-index offset.
enum CompletionStreamUnit {
    Single(ProtoStream),
    PrefillDecode {
        prefill: ProtoStream,
        decode: Box<ProtoStream>,
    },
}

/// Per-stream token/timing totals returned by the completion chunk loop; the
/// coordinator aggregates them into the single usage chunk and metrics record.
struct CompletionStreamOutcome {
    prompt_tokens: u32,
    cached_tokens: u32,
    reasoning_tokens: u32,
    completion_tokens: u32,
    first_token_time: Option<Instant>,
}

/// Shared streaming processor for both single and prefill/decode dispatch modes
#[derive(Clone)]
pub(crate) struct StreamingProcessor {
    tool_parser_factory: ToolParserFactory,
    reasoning_parser_factory: ReasoningParserFactory,
    configured_tool_parser: Option<String>,
    configured_reasoning_parser: Option<String>,
    backend_type: &'static str,
}

/// Context for generate endpoint streaming - groups config params to reduce function arguments
struct GenerateStreamContext {
    request_id: String,
    weight_version: String,
    return_logprob: bool,
    backend_type: &'static str,
    model: String,
}

impl StreamingProcessor {
    pub fn new(
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
        backend_type: &'static str,
    ) -> Self {
        Self {
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
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
    /// Note: Caller should attach load guards to the returned response using
    /// `WorkerLoadGuard::attach_to_response()` for proper RAII lifecycle management.
    pub fn process_streaming_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        skip_special_tokens: bool,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Response {
        use bytes::Bytes;
        use tokio::sync::mpsc;

        let stop_params = (
            chat_request.stop.clone(),
            chat_request.stop_token_ids.clone(),
            skip_special_tokens,
            chat_request.no_stop_trim,
            chat_request.ignore_eos,
        );

        // Create SSE channel
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

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

                    if let Err(e) = result {
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::PrefillDecode {
                prefill,
                decode,
                pd_timing,
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
                            reservation,
                        )
                        .await;

                    if let Err(e) = result {
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::Embedding { .. } => {
                utils::send_error_sse(
                    &tx,
                    "Embeddings not supported in streaming mode",
                    "invalid_request_error",
                );
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
            }
            // Batch results exist only on the completions pipeline.
            context::ExecutionResult::Batch { .. } => {
                utils::send_error_sse(
                    &tx,
                    "Batched results not supported in chat streaming",
                    "invalid_request_error",
                );
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
            }
        }

        // Return SSE response
        build_sse_response(rx)
    }

    /// Process streaming chunks from a single stream (Regular mode)
    #[expect(clippy::too_many_arguments)]
    pub async fn process_streaming_chunks(
        &self,
        grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        original_request: Arc<ChatCompletionRequest>,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), String> {
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
    #[expect(clippy::too_many_arguments)]
    async fn process_streaming_chunks_inner(
        &self,
        mut grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        original_request: Arc<ChatCompletionRequest>,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
        pd_timing: Option<context::PdTiming>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), String> {
        // Metrics timing
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Extract request parameters
        let separate_reasoning = original_request.separate_reasoning;
        let tool_choice = &original_request.tool_choice;
        let tools = &original_request.tools;
        let history_tool_calls_count = utils::get_history_tool_calls_count(&original_request);
        let stream_options = &original_request.stream_options;

        // Phase 1: Initialize state tracking (per-index for n>1 support)
        let mut is_firsts: HashMap<u32, bool> = HashMap::new();
        let mut stream_buffers: HashMap<u32, String> = HashMap::new();
        let mut finish_reasons: HashMap<u32, String> = HashMap::new();
        let mut matched_stops: HashMap<u32, Option<Value>> = HashMap::new();
        // Indices whose local stop decoder fired: their finish reason is pinned
        // to "stop" and later engine output for the index is ignored.
        let mut stopped_indices: HashSet<u32> = HashSet::new();
        let mut prompt_tokens: HashMap<u32, u32> = HashMap::new();
        let mut completion_tokens = CompletionTokenTracker::new();
        let mut cached_tokens: HashMap<u32, u32> = HashMap::new();
        let mut reasoning_tokens: HashMap<u32, u32> = HashMap::new();

        // Parser state (lazy initialization per index)
        type PooledReasoningParser = Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>;
        let mut reasoning_parsers: HashMap<u32, PooledReasoningParser> = HashMap::new();

        type PooledToolParser = Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>;
        let mut tool_parsers: HashMap<u32, PooledToolParser> = HashMap::new();
        let mut has_tool_calls: HashMap<u32, bool> = HashMap::new();

        // Per-index stop decoders (each index needs its own state for n>1 support)
        let mut stop_decoders: HashMap<u32, StopSequenceDecoder> = HashMap::new();

        // Reusable SSE formatting buffer to avoid allocations per chunk
        let mut sse_buffer = Vec::with_capacity(512);
        // Reusable SSE encoder for the post-loop flush / tool / finish / usage
        // chunks, which previously each did `to_string` + `format!`.
        let mut sse_encoder = SseEncoder::new();

        // Use dispatch metadata for consistent response fields
        let request_id = &dispatch.request_id;
        let model = &dispatch.model;
        let created = dispatch.created;
        let system_fingerprint = dispatch.weight_version.as_deref();

        // Check parser availability once upfront (log warning only once per request)
        let reasoning_parser_available = separate_reasoning
            && utils::check_reasoning_parser_availability(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                model,
            );

        // If the template supports a thinking toggle and the user enabled it,
        // the template injected `<think>` in the prefill — parsers should start
        // in reasoning mode.
        let thinking_override = utils::should_mark_reasoning_started(
            utils::resolve_user_thinking(
                original_request.chat_template_kwargs.as_ref(),
                original_request.reasoning_effort.as_deref(),
                tokenizer.as_ref(),
            ),
            tokenizer.as_ref(),
        );
        let think_in_prefill = tokenizer.think_in_prefill();

        // Check if JSON schema constraint was used (specific function or required mode)
        let has_structural_tag = self
            .tool_parser_factory
            .registry()
            .has_structural_tag_for_parser(self.configured_tool_parser.as_deref());
        let used_json_schema = if has_structural_tag {
            false
        } else {
            match tool_choice {
                Some(ToolChoice::Function { .. }) => true,
                Some(ToolChoice::Value(ToolChoiceValue::Required)) => true,
                Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
                _ => false,
            }
        };

        // Check if this is the specific function case (LLM generates parameters only, no name field).
        // Only applies when json_schema is used — structural tags include framing tokens
        // that the parser must handle.
        let is_specific_function =
            used_json_schema && matches!(tool_choice, Some(ToolChoice::Function { .. }));

        let tool_parser_available = tools.is_some()
            && utils::check_tool_parser_availability(
                &self.tool_parser_factory,
                self.configured_tool_parser.as_deref(),
                model,
            );

        if separate_reasoning && !reasoning_parser_available {
            debug!(
                "No reasoning parser found for model '{}', skipping reasoning parsing",
                model
            );
        }

        if tools.is_some() && !tool_parser_available {
            debug!(
                "No tool parser found for model '{}', skipping tool call parsing",
                model
            );
        }

        // Phase 2: Main streaming loop
        while let Some(response) = grpc_stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e.message()))?;

            match gen_response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => {
                    // Track TTFT immediately on first chunk received from backend
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                        if let Some(timing) = &pd_timing {
                            Metrics::record_pd_ttft(
                                self.backend_type,
                                model,
                                timing.runtime,
                                timing.prefill_start.elapsed(),
                            );
                        }
                    }

                    let index = chunk.index();

                    // Once the local stop decoder has fired for an index, ignore
                    // any further engine output the backend emits for it.
                    if stopped_indices.contains(&index) {
                        continue;
                    }

                    completion_tokens.record_chunk(&chunk);

                    // Get or create stop decoder for this index
                    let stop_decoder = stop_decoders.entry(index).or_insert_with(|| {
                        let (
                            ref stop,
                            ref stop_token_ids,
                            skip_special_tokens,
                            no_stop_trim,
                            ignore_eos,
                        ) = stop_params;
                        utils::create_stop_decoder(
                            &tokenizer,
                            stop.as_ref(),
                            stop_token_ids.as_ref(),
                            skip_special_tokens,
                            no_stop_trim,
                            ignore_eos,
                        )
                    });

                    // Process tokens through stop decoder
                    let (chunk_text, should_stop) =
                        Self::process_chunk_tokens(stop_decoder, chunk.token_ids());

                    if should_stop {
                        // Stop-decoder match takes precedence: pin "stop" even if
                        // the backend's eventual Complete carries "length" (the
                        // local stop sequence fired first). Any pre-stop text in
                        // `chunk_text` is still emitted below before the finish
                        // reason is flushed in Phase 4.
                        finish_reasons
                            .entry(index)
                            .or_insert_with(|| "stop".to_string());
                        matched_stops.entry(index).or_insert_with(|| {
                            stop_decoder
                                .matched_stop()
                                .map(|s| Value::String(s.to_string()))
                        });
                        stopped_indices.insert(index);
                    }

                    if chunk_text.is_empty() {
                        continue;
                    }

                    // Process logprobs if present
                    let choice_logprobs = chunk.output_logprobs().map(|ref proto_logprobs| {
                        utils::convert_proto_to_openai_logprobs(proto_logprobs, &tokenizer)
                    });

                    // Initialize stream buffer if first time
                    let stream_buffer = stream_buffers.entry(index).or_default();

                    // Send first chunk with role
                    if is_firsts.get(&index).copied().unwrap_or(true) {
                        let first_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                            .created(created)
                            .add_choice_role(index, "assistant")
                            .maybe_system_fingerprint(system_fingerprint)
                            .build();
                        Self::format_sse_chunk_into(&mut sse_buffer, &first_chunk);
                        tx.send(Ok(Bytes::from(sse_buffer.clone())))
                            .map_err(|_| "Failed to send first chunk".to_string())?;
                        is_firsts.insert(index, false);
                    }

                    // Calculate delta
                    let mut delta = chunk_text;
                    stream_buffer.push_str(&delta);

                    // Reasoning content handling
                    let in_reasoning = if separate_reasoning && reasoning_parser_available {
                        let (normal_text, reasoning_chunk, in_reasoning) = self
                            .process_reasoning_stream(
                                &delta,
                                index,
                                &mut reasoning_parsers,
                                thinking_override,
                                think_in_prefill,
                                request_id,
                                model,
                                created,
                                system_fingerprint,
                            )
                            .await;
                        if let Some(chunk) = reasoning_chunk {
                            Self::format_sse_chunk_into(&mut sse_buffer, &chunk);
                            tx.send(Ok(Bytes::from(sse_buffer.clone())))
                                .map_err(|_| "Failed to send reasoning chunk".to_string())?;
                        }
                        delta = normal_text;
                        in_reasoning
                    } else {
                        false
                    };

                    // Tool call handling
                    let tool_choice_enabled =
                        !matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None)));

                    if let Some(tools_ref) = tools.as_ref() {
                        if !in_reasoning
                            && tool_choice_enabled
                            && (tool_parser_available || used_json_schema)
                        {
                            let tool_chunks = if is_specific_function {
                                // Handle specific function case - emit tool call deltas with arguments
                                Self::process_specific_function_stream(
                                    &delta,
                                    index,
                                    &mut has_tool_calls,
                                    tool_choice.as_ref(),
                                    request_id,
                                    model,
                                    created,
                                    system_fingerprint,
                                    history_tool_calls_count,
                                )
                            } else {
                                // Use incremental parser for regular/required modes
                                self.process_tool_calls_stream(
                                    &delta,
                                    index,
                                    &mut tool_parsers,
                                    &mut has_tool_calls,
                                    tools_ref,
                                    request_id,
                                    model,
                                    created,
                                    system_fingerprint,
                                    history_tool_calls_count,
                                    used_json_schema,
                                )
                                .await
                            };

                            for chunk in tool_chunks {
                                Self::format_sse_chunk_into(&mut sse_buffer, &chunk);
                                tx.send(Ok(Bytes::from(sse_buffer.clone())))
                                    .map_err(|_| "Failed to send tool call chunk".to_string())?;
                            }

                            // Always skip regular content when tool parsing is active
                            // Parser either emitted chunks or buffered content
                            continue;
                        }
                    }

                    // Regular content emission
                    if !delta.is_empty() {
                        let content_chunk =
                            ChatCompletionStreamResponse::builder(request_id, model)
                                .created(created)
                                .add_choice_content_with_logprobs(
                                    index,
                                    "assistant",
                                    delta,
                                    choice_logprobs,
                                )
                                .maybe_system_fingerprint(system_fingerprint)
                                .build();
                        Self::format_sse_chunk_into(&mut sse_buffer, &content_chunk);
                        tx.send(Ok(Bytes::from(sse_buffer.clone())))
                            .map_err(|_| "Failed to send content chunk".to_string())?;
                    }
                }
                ProtoResponseVariant::Complete(complete) => {
                    let index = complete.index();

                    // Flush any remaining text for this index's stop_decoder
                    if let Some(decoder) = stop_decoders.get_mut(&index) {
                        if let SequenceDecoderOutput::Text(text) = decoder.flush() {
                            if !text.is_empty() {
                                let stream_buffer = stream_buffers.entry(index).or_default();
                                stream_buffer.push_str(&text);

                                let content_chunk =
                                    ChatCompletionStreamResponse::builder(request_id, model)
                                        .created(created)
                                        .add_choice_content(index, "assistant", text)
                                        .maybe_system_fingerprint(system_fingerprint)
                                        .build();

                                let sse_chunk =
                                    sse_encoder.encode_data(&content_chunk).map_err(|e| {
                                        format!("Failed to serialize content chunk: {e}")
                                    })?;
                                tx.send(Ok(sse_chunk))
                                    .map_err(|_| "Failed to send flushed content".to_string())?;
                            }
                        }
                    }

                    // Store metadata
                    prompt_tokens.insert(index, complete.prompt_tokens());

                    completion_tokens.record_complete(&complete);

                    cached_tokens.insert(index, complete.cached_tokens());
                    reasoning_tokens.insert(index, complete.reasoning_tokens());

                    // A local stop-decoder match already pinned "stop" for this
                    // index; don't let the engine's finish reason overwrite it.
                    if !stopped_indices.contains(&index) {
                        finish_reasons.insert(index, complete.finish_reason().to_string());
                        matched_stops.insert(index, complete.matched_stop_json());
                    }

                    // Don't break - continue reading all Complete messages for n>1
                }
                ProtoResponseVariant::None => continue,
            }
        }

        // Phase 3: Check unstreamed tool args
        for (index, parser) in &tool_parsers {
            let parser_guard = parser.lock().await;
            if let Some(unstreamed_items) = parser_guard.get_unstreamed_tool_args() {
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
                        .build();

                    let sse_chunk = sse_encoder
                        .encode_data(&tool_chunk)
                        .map_err(|e| format!("Failed to serialize tool chunk: {e}"))?;
                    tx.send(Ok(sse_chunk))
                        .map_err(|_| "Failed to send unstreamed tool args".to_string())?;
                }
            }
        }

        // Phase 4: Finish reason chunks
        for (index, finish_reason) in &finish_reasons {
            let final_finish_reason =
                if has_tool_calls.get(index).copied().unwrap_or(false) && finish_reason == "stop" {
                    "tool_calls".to_string()
                } else {
                    finish_reason.clone()
                };

            let matched_stop_value = matched_stops.get(index).and_then(|v| v.clone());

            let finish_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                .created(created)
                .add_choice_finish_reason(*index, final_finish_reason, matched_stop_value)
                .maybe_system_fingerprint(system_fingerprint)
                .build();

            let sse_chunk = sse_encoder
                .encode_data(&finish_chunk)
                .map_err(|e| format!("Failed to serialize finish chunk: {e}"))?;
            tx.send(Ok(sse_chunk))
                .map_err(|_| "Failed to send finish chunk".to_string())?;
        }

        // Phase 5: Usage chunk
        if let Some(stream_opts) = stream_options {
            if stream_opts.include_usage.unwrap_or(false) {
                let total_prompt: u32 = prompt_tokens.values().sum();
                let total_completion: u32 = completion_tokens.total();
                let total_cached: u32 = cached_tokens.values().sum();
                let total_reasoning: u32 = reasoning_tokens.values().sum();

                let usage_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                    .created(created)
                    .usage(
                        Usage::from_counts(total_prompt, total_completion)
                            .with_cached_tokens(total_cached)
                            .with_reasoning_tokens(total_reasoning),
                    )
                    .maybe_system_fingerprint(system_fingerprint)
                    .build();

                let sse_chunk = sse_encoder
                    .encode_data(&usage_chunk)
                    .map_err(|e| format!("Failed to serialize usage chunk: {e}"))?;
                tx.send(Ok(sse_chunk))
                    .map_err(|_| "Failed to send usage chunk".to_string())?;
            }
        }

        // Mark stream as completed successfully to prevent abort on drop
        grpc_stream.mark_completed();

        // Record streaming metrics
        let total_prompt: u32 = prompt_tokens.values().sum();
        let total_completion: u32 = completion_tokens.total();
        if let Some(handle) = reservation {
            handle
                .settle_success(UsageSettlement {
                    actual_input_tokens: total_prompt,
                    completion_tokens: total_completion,
                })
                .await;
        }
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.backend_type,
            model_id: model,
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: first_token_time.map(|t| t.duration_since(start_time)),
            generation_duration: start_time.elapsed(),
            input_tokens: Some(total_prompt as u64),
            output_tokens: total_completion as u64,
        });

        Ok(())
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
        original_request: Arc<ChatCompletionRequest>,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
        pd_timing: context::PdTiming,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), String> {
        // Phase 1.5: Collect input_logprobs from prefill stream if requested
        if original_request.logprobs {
            while let Some(response) = prefill_stream.next().await {
                let gen_response =
                    response.map_err(|e| format!("Prefill stream error: {}", e.message()))?;
                match gen_response.into_response() {
                    ProtoResponseVariant::Complete(_complete) => {
                        // Input logprobs collected but not yet used in streaming
                        // (OpenAI spec doesn't require prompt logprobs in streaming responses)
                        break;
                    }
                    _ => continue,
                }
            }
        }

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

        // Mark prefill stream as completed AFTER decode completes successfully
        // This ensures that if client disconnects during decode, BOTH streams send abort
        if result.is_ok() {
            prefill_stream.mark_completed();
        }

        result
    }

    /// Process streaming generate response and return SSE response
    ///
    /// Simpler than chat - no tool/reasoning parsing, just text accumulation
    ///
    /// Note: Caller should attach load guards to the returned response using
    /// `WorkerLoadGuard::attach_to_response()` for proper RAII lifecycle management.
    pub fn process_streaming_generate(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        generate_request: Arc<GenerateRequest>,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Response {
        // Create SSE channel
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        // Build context once, clone for spawned task
        let ctx = GenerateStreamContext {
            request_id: dispatch.request_id.clone(),
            weight_version: dispatch
                .weight_version
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            return_logprob: generate_request.return_logprob.unwrap_or(false),
            backend_type: self.backend_type,
            model: dispatch.model.clone(),
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

                    if let Err(e) = result {
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::PrefillDecode {
                prefill,
                decode,
                pd_timing,
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
                        reservation,
                    )
                    .await;

                    if let Err(e) = result {
                        utils::send_error_sse(&tx, &e, "internal_error");
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::Embedding { .. } => {
                utils::send_error_sse(
                    &tx,
                    "Embeddings not supported in streaming generate",
                    "invalid_request_error",
                );
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
            }
            // Batch results exist only on the completions pipeline.
            context::ExecutionResult::Batch { .. } => {
                utils::send_error_sse(
                    &tx,
                    "Batched results not supported in streaming generate",
                    "invalid_request_error",
                );
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
            }
        }

        // Return SSE response
        build_sse_response(rx)
    }

    /// Process streaming chunks for generate endpoint (no tool/reasoning parsing)
    /// TODO: add streaming logprob support
    async fn process_generate_streaming(
        tokenizer: Arc<dyn Tokenizer>,
        mut stream: ProtoStream,
        ctx: GenerateStreamContext,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), String> {
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Track state per index for n>1 case
        let mut accumulated_texts: HashMap<u32, String> = HashMap::new();
        let mut completion_tokens_map: HashMap<u32, u32> = HashMap::new();
        // Same prompt for every index; the last Complete's value is as good as any.
        let mut prompt_tokens: u32 = 0;
        // Reusable SSE encoder shared across every chunk emitted for this stream.
        let mut sse_encoder = SseEncoder::new();

        while let Some(response) = stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e.message()))?;

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
                            "weight_version": &ctx.weight_version,
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

                    // Send final chunk with finish_reason
                    let finish_response = serde_json::json!({
                        "text": accumulated_text,
                        "output_ids": complete.output_ids()[complete.output_ids().len().saturating_sub(1)..].to_vec(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": complete.finish_reason(),
                            "prompt_tokens": complete.prompt_tokens(),
                            "weight_version": &ctx.weight_version,
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
            handle
                .settle_success(UsageSettlement {
                    actual_input_tokens: prompt_tokens,
                    completion_tokens: total_completion,
                })
                .await;
        }
        Self::record_generate_metrics(start_time, first_token_time, total_completion, &ctx);

        Ok(())
    }

    /// Process prefill/decode streaming for generate endpoint (PD mode with logprobs support)
    async fn process_generate_prefill_decode_streaming(
        tokenizer: Arc<dyn Tokenizer>,
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        ctx: GenerateStreamContext,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
        pd_timing: context::PdTiming,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), String> {
        // Collect input_logprobs from prefill stream if requested
        let input_token_logprobs = if ctx.return_logprob {
            let mut input_logprobs = None;
            while let Some(response) = prefill_stream.next().await {
                let gen_response =
                    response.map_err(|e| format!("Prefill stream error: {}", e.message()))?;
                match gen_response.into_response() {
                    ProtoResponseVariant::Complete(complete) => {
                        // Extract input_logprobs from prefill Complete message (convert proto to SGLang format)
                        input_logprobs = complete
                            .input_logprobs()
                            .as_ref()
                            .map(utils::convert_generate_input_logprobs);
                        break;
                    }
                    _ => continue,
                }
            }
            input_logprobs
        } else {
            None
        };

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

        // Mark prefill stream as completed AFTER decode completes successfully
        // This ensures that if client disconnects during decode, BOTH streams send abort
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
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
        pd_timing: Option<context::PdTiming>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), String> {
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Track state per index for n>1 case
        let mut accumulated_texts: HashMap<u32, String> = HashMap::new();
        let mut accumulated_output_logprobs: HashMap<u32, Option<Vec<Vec<Option<f64>>>>> =
            HashMap::new();
        let mut completion_tokens_map: HashMap<u32, u32> = HashMap::new();
        // Same prompt for every index; the last Complete's value is as good as any.
        let mut prompt_tokens: u32 = 0;
        // Reusable SSE encoder shared across every chunk emitted for this stream.
        let mut sse_encoder = SseEncoder::new();

        while let Some(response) = stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e.message()))?;

            match gen_response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => {
                    // Track TTFT immediately on first chunk received from backend
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                        if let Some(timing) = &pd_timing {
                            Metrics::record_pd_ttft(
                                ctx.backend_type,
                                &ctx.model,
                                timing.runtime,
                                timing.prefill_start.elapsed(),
                            );
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

                    // Handle output logprobs based on backend behavior:
                    // - SGLang sends cumulative logprobs (replace is correct)
                    // - vLLM sends delta logprobs (need to extend/accumulate)
                    if let Some(ref output_logprobs) = chunk.output_logprobs() {
                        let converted = utils::convert_generate_output_logprobs(output_logprobs);
                        if chunk.is_vllm() {
                            // vLLM sends delta - extend existing logprobs
                            if let Some(v) = accumulated_output_logprobs
                                .entry(index)
                                .or_insert_with(|| Some(Vec::new()))
                                .as_mut()
                            {
                                v.extend(converted);
                            }
                        } else {
                            // SGLang sends cumulative - replace
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
                            "weight_version": &ctx.weight_version,
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
                            "weight_version": &ctx.weight_version,
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
            handle
                .settle_success(UsageSettlement {
                    actual_input_tokens: prompt_tokens,
                    completion_tokens: total_completion,
                })
                .await;
        }
        Self::record_generate_metrics(start_time, first_token_time, total_completion, &ctx);

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
        ctx: &GenerateStreamContext,
    ) {
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: ctx.backend_type,
            model_id: &ctx.model,
            endpoint: metrics_labels::ENDPOINT_GENERATE,
            ttft: first_token_time.map(|t| t.duration_since(start_time)),
            generation_duration: start_time.elapsed(),
            input_tokens: None, // generate endpoint doesn't expose prompt tokens in streaming
            output_tokens: total_completion as u64,
        });
    }

    /// Process a chunk of tokens through the stop decoder
    fn process_chunk_tokens(
        stop_decoder: &mut StopSequenceDecoder,
        token_ids: &[u32],
    ) -> (String, bool) {
        let mut chunk_text = String::new();

        for &token_id in token_ids {
            match stop_decoder.process_token(token_id).unwrap_or_else(|e| {
                debug!(
                    "Error processing token {}: {}. Treating as Held.",
                    token_id, e
                );
                SequenceDecoderOutput::Held
            }) {
                SequenceDecoderOutput::Text(text) => {
                    chunk_text.push_str(&text);
                }
                SequenceDecoderOutput::StoppedWithText(text) => {
                    chunk_text.push_str(&text);
                    return (chunk_text, true);
                }
                SequenceDecoderOutput::Stopped => {
                    return (chunk_text, true);
                }
                SequenceDecoderOutput::Held => {}
            }
        }
        (chunk_text, false)
    }

    /// Helper: Process reasoning content in streaming mode
    #[expect(clippy::too_many_arguments)]
    async fn process_reasoning_stream(
        &self,
        delta: &str,
        index: u32,
        reasoning_parsers: &mut HashMap<u32, Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>>,
        thinking_override: bool,
        think_in_prefill: bool,
        request_id: &str,
        model: &str,
        created: u64,
        system_fingerprint: Option<&str>,
    ) -> (String, Option<ChatCompletionStreamResponse>, bool) {
        // Create fresh parser for this index (not pooled, to avoid state pollution)
        #[expect(
            clippy::expect_used,
            reason = "parser availability is checked upfront before streaming begins"
        )]
        reasoning_parsers.entry(index).or_insert_with(|| {
            let mut parser = utils::create_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                model,
            )
            .expect("Parser should be available - checked upfront");
            if thinking_override {
                parser.mark_reasoning_started();
                if think_in_prefill {
                    parser.mark_think_start_stripped();
                }
            }
            Arc::new(tokio::sync::Mutex::new(parser))
        });

        if let Some(pooled_parser) = reasoning_parsers.get(&index) {
            let (parse_result, in_reasoning) = {
                let mut parser = pooled_parser.lock().await;
                let result = parser.parse_reasoning_streaming_incremental(delta);
                let in_reasoning = parser.is_in_reasoning();
                (result, in_reasoning)
            };

            match parse_result {
                Ok(ParserResult {
                    reasoning_text,
                    normal_text,
                }) => {
                    let chunk = if reasoning_text.is_empty() {
                        None
                    } else {
                        Some(
                            ChatCompletionStreamResponse::builder(request_id, model)
                                .created(created)
                                .add_choice_reasoning(index, reasoning_text)
                                .maybe_system_fingerprint(system_fingerprint)
                                .build(),
                        )
                    };
                    return (normal_text, chunk, in_reasoning);
                }
                Err(e) => {
                    warn!("Reasoning parsing error: {}", e);
                }
            }
        }

        (delta.to_string(), None, false)
    }

    /// Helper: Process specific function case - emit tool call deltas with arguments
    #[expect(clippy::too_many_arguments)]
    fn process_specific_function_stream(
        delta: &str,
        index: u32,
        has_tool_calls: &mut HashMap<u32, bool>,
        tool_choice: Option<&ToolChoice>,
        request_id: &str,
        model: &str,
        created: u64,
        system_fingerprint: Option<&str>,
        history_tool_calls_count: usize,
    ) -> Vec<ChatCompletionStreamResponse> {
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

                chunks.push(
                    ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_tool_name(index, tool_call_id, function.name.clone())
                        .maybe_system_fingerprint(system_fingerprint)
                        .build(),
                );
            }

            // Emit arguments delta
            if !delta.is_empty() {
                chunks.push(
                    ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_tool_args(index, delta.to_string())
                        .maybe_system_fingerprint(system_fingerprint)
                        .build(),
                );
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
        tool_parsers: &mut HashMap<u32, Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>>,
        has_tool_calls: &mut HashMap<u32, bool>,
        tools: &[Tool],
        request_id: &str,
        model: &str,
        created: u64,
        system_fingerprint: Option<&str>,
        history_tool_calls_count: usize,
        use_json_parser: bool,
    ) -> Vec<ChatCompletionStreamResponse> {
        let mut chunks = Vec::new();

        // Create fresh parser for this index (not pooled, to avoid state pollution)
        #[expect(
            clippy::expect_used,
            reason = "parser availability is checked upfront before streaming begins"
        )]
        tool_parsers.entry(index).or_insert_with(|| {
            let parser = if use_json_parser {
                utils::create_tool_parser(&self.tool_parser_factory, Some("json"), model)
                    .expect("JSON parser should be available")
            } else {
                utils::create_tool_parser(
                    &self.tool_parser_factory,
                    self.configured_tool_parser.as_deref(),
                    model,
                )
                .expect("Parser should be available - checked upfront")
            };
            Arc::new(tokio::sync::Mutex::new(parser))
        });

        if let Some(pooled_parser) = tool_parsers.get(&index) {
            let mut parser = pooled_parser.lock().await;

            match parser.parse_incremental(delta, tools).await {
                Ok(StreamingParseResult { normal_text, calls }) => {
                    // Emit normal text if present
                    if !normal_text.is_empty() {
                        chunks.push(
                            ChatCompletionStreamResponse::builder(request_id, model)
                                .created(created)
                                .add_choice_content(index, "assistant", normal_text)
                                .maybe_system_fingerprint(system_fingerprint)
                                .build(),
                        );
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

                        chunks.push(
                            ChatCompletionStreamResponse::builder(request_id, model)
                                .created(created)
                                .add_choice_tool_call_delta(index, tool_call_delta)
                                .maybe_system_fingerprint(system_fingerprint)
                                .build(),
                        );
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

    /// Format a response as SSE chunk into a reusable buffer
    /// This avoids allocations by reusing the same buffer across multiple chunks
    #[inline]
    fn format_sse_chunk_into(buffer: &mut Vec<u8>, chunk: &ChatCompletionStreamResponse) {
        buffer.clear();
        buffer.extend_from_slice(b"data: ");
        if let Err(e) = serde_json::to_writer(&mut *buffer, chunk) {
            error!("Failed to serialize SSE chunk: {}", e);
            buffer.clear();
            buffer.extend_from_slice(b"data: ");
            let error_msg = json!({"error": "serialization_failed"}).to_string();
            buffer.extend_from_slice(error_msg.as_bytes());
        }
        buffer.extend_from_slice(b"\n\n");
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
    fn format_messages_sse_into(
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

    /// Send a `MessageStreamEvent` through the SSE channel using a reusable buffer.
    fn send_messages_event(
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
        buffer: &mut Vec<u8>,
        event: &MessageStreamEvent,
    ) -> Result<(), String> {
        Self::format_messages_sse_into(buffer, event)?;
        tx.send(Ok(Bytes::from(buffer.clone())))
            .map_err(|_| "Client disconnected".to_string())
    }

    /// Process reasoning content in Messages streaming mode (n=1 only).
    ///
    /// Returns `(normal_text, reasoning_text, in_reasoning)`.
    /// Caller handles SSE event emission.
    async fn process_messages_reasoning(
        &self,
        delta: &str,
        reasoning_parser: &mut Option<Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>>,
        thinking_override: bool,
        think_in_prefill: bool,
        model: &str,
    ) -> (String, String, bool) {
        // Lazily create parser
        if reasoning_parser.is_none() {
            if let Some(mut parser) = utils::create_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                model,
            ) {
                if thinking_override {
                    parser.mark_reasoning_started();
                    if think_in_prefill {
                        parser.mark_think_start_stripped();
                    }
                }
                *reasoning_parser = Some(Arc::new(tokio::sync::Mutex::new(parser)));
            }
        }

        if let Some(ref parser_arc) = reasoning_parser {
            let (parse_result, in_reasoning) = {
                let mut parser = parser_arc.lock().await;
                let result = parser.parse_reasoning_streaming_incremental(delta);
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

        (delta.to_string(), String::new(), false)
    }

    /// Process streaming Messages API response and return SSE response.
    ///
    /// Parallel to [`Self::process_streaming_response`] for chat, but emits
    /// Anthropic SSE format (`event: {type}\ndata: {json}\n\n`).
    pub fn process_messages_streaming_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        messages_request: Arc<CreateMessageRequest>,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        skip_special_tokens: bool,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Response {
        let stop_params = (
            messages_request
                .stop_sequences
                .clone()
                .map(StringOrArray::Array),
            None::<Vec<u32>>, // No stop_token_ids in Messages API
            skip_special_tokens,
            false, // no_stop_trim
            false, // ignore_eos — not available in Messages API
        );

        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

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

                    if let Err(e) = result {
                        let error_event = MessageStreamEvent::Error {
                            error: messages::ErrorResponse {
                                error_type: "api_error".to_string(),
                                message: e,
                            },
                        };
                        let mut buf = Vec::with_capacity(256);
                        let _ = Self::send_messages_event(&tx, &mut buf, &error_event);
                    }
                    // No data: [DONE] — Anthropic uses message_stop instead
                });
            }
            context::ExecutionResult::PrefillDecode {
                // TODO(#1781 follow-up): thread pd_timing for honest PD TTFT
                prefill,
                decode,
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
                            reservation,
                        )
                        .await;

                    if let Err(e) = result {
                        let error_event = MessageStreamEvent::Error {
                            error: messages::ErrorResponse {
                                error_type: "api_error".to_string(),
                                message: e,
                            },
                        };
                        let mut buf = Vec::with_capacity(256);
                        let _ = Self::send_messages_event(&tx, &mut buf, &error_event);
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
                let _ = Self::send_messages_event(&tx, &mut buf, &error_event);
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
                let _ = Self::send_messages_event(&tx, &mut buf, &error_event);
            }
        }

        build_sse_response(rx)
    }

    /// Process Messages API streaming chunks from a single stream.
    ///
    /// Implements the Anthropic streaming protocol with content block
    /// state tracking. Always n=1 (no per-index HashMap).
    #[expect(clippy::too_many_arguments)]
    pub async fn process_messages_streaming_chunks(
        &self,
        mut grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        original_request: Arc<CreateMessageRequest>,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), String> {
        let start_time = Instant::now();
        let mut first_token_time: Option<Instant> = None;

        // Reusable SSE formatting buffer to avoid allocations per event
        let mut sse_buffer = Vec::with_capacity(512);

        let request_id = &dispatch.request_id;
        let model = &dispatch.model;

        let has_tools = original_request.tools.is_some();

        // Content block state machine
        let mut current_block_index: u32 = 0;
        let mut thinking_block_open = false;
        let mut text_block_open = false;
        let mut tool_block_open = false;
        let mut has_tool_calls = false;

        // Parser state (simple variables — Messages is always n=1)
        let mut reasoning_parser: Option<Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>> = None;

        // Stop decoder
        let mut stop_decoder = {
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

        // Token tracking
        let mut completion_tokens = CompletionTokenTracker::new();
        let mut prompt_tokens: u32 = 0;
        let mut finish_reason_str = String::new();
        let mut matched_stop: Option<Value> = None;
        // Set once the local stop decoder fires: pins "stop" and ignores later
        // engine output (the backend has no stop-string detection over ZMQ).
        let mut stopped = false;

        // Check parser availability once upfront. Run parser when the user explicitly
        // enabled thinking, or when the selected parser needs structural special tokens.
        let reasoning_requires_special_tokens = utils::reasoning_parser_requires_special_tokens(
            &self.reasoning_parser_factory,
            self.configured_reasoning_parser.as_deref(),
            model,
        );
        let separate_reasoning = reasoning_requires_special_tokens
            || matches!(
                &original_request.thinking,
                Some(
                    messages::ThinkingConfig::Enabled { .. }
                        | messages::ThinkingConfig::Adaptive { .. }
                )
            );
        let reasoning_parser_available = separate_reasoning
            && utils::check_reasoning_parser_availability(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                model,
            );

        // Determine if thinking is effectively ON (for mark_reasoning_started).
        let user_thinking = match &original_request.thinking {
            Some(
                messages::ThinkingConfig::Enabled { .. }
                | messages::ThinkingConfig::Adaptive { .. },
            ) => Some(true),
            Some(messages::ThinkingConfig::Disabled) => Some(false),
            None => None,
        };
        let thinking_override =
            utils::should_mark_reasoning_started(user_thinking, tokenizer.as_ref());
        let think_in_prefill = tokenizer.think_in_prefill();

        let tool_choice_enabled = !matches!(
            &original_request.tool_choice,
            Some(messages::ToolChoice::None)
        );

        let tool_parser_available = has_tools
            && tool_choice_enabled
            && utils::check_tool_parser_availability(
                &self.tool_parser_factory,
                self.configured_tool_parser.as_deref(),
                model,
            );

        let has_structural_tag = self
            .tool_parser_factory
            .registry()
            .has_structural_tag_for_parser(self.configured_tool_parser.as_deref());
        let used_json_schema = !has_structural_tag
            && matches!(
                &original_request.tool_choice,
                Some(messages::ToolChoice::Tool { .. } | messages::ToolChoice::Any { .. })
            );

        // Check if model output is arguments-only for a specific function (ToolChoice::Tool).
        // Only applies when json_schema is used — structural tags include framing tokens.
        let is_specific_function = used_json_schema
            && matches!(
                &original_request.tool_choice,
                Some(messages::ToolChoice::Tool { .. })
            );

        let history_tool_calls_count =
            message_utils::get_history_tool_calls_count_messages(&original_request);

        // Pre-convert Messages tools to Chat tools for parser reuse (done once upfront)
        let chat_tools: Vec<Tool> = original_request
            .tools
            .as_deref()
            .map(message_utils::extract_chat_tools)
            .unwrap_or_default();

        // Create fresh streaming tool parser (not pooled — streaming parsers maintain state)
        let mut streaming_tool_parser: Option<Box<dyn ToolParser>> =
            if has_tools && tool_choice_enabled && (tool_parser_available || used_json_schema) {
                let parser_name = if used_json_schema {
                    Some("json")
                } else {
                    self.configured_tool_parser.as_deref()
                };
                utils::create_tool_parser(&self.tool_parser_factory, parser_name, model)
            } else {
                None
            };

        // Phase 1: Emit message_start with skeleton Message
        let start_message = Message {
            id: request_id.clone(),
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![],
            model: model.clone(),
            stop_reason: None,
            stop_sequence: None,
            usage: messages::Usage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                cache_creation: None,
                server_tool_use: None,
                service_tier: None,
            },
        };
        Self::send_messages_event(
            tx,
            &mut sse_buffer,
            &MessageStreamEvent::MessageStart {
                message: start_message,
            },
        )?;

        // Phase 2: Main streaming loop
        while let Some(response) = grpc_stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e.message()))?;

            match gen_response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => {
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                    }

                    // Once the local stop decoder has fired, ignore further
                    // engine output for this (single-choice) request.
                    if stopped {
                        continue;
                    }

                    completion_tokens.record_chunk(&chunk);

                    let (chunk_text, should_stop) =
                        Self::process_chunk_tokens(&mut stop_decoder, chunk.token_ids());

                    if should_stop {
                        // Stop-decoder match takes precedence over the engine's
                        // eventual finish reason (the local stop sequence fired
                        // first). Pre-stop text in `chunk_text` is still emitted
                        // below; Phase 4 derives StopSequence from `matched_stop`.
                        stopped = true;
                        finish_reason_str = "stop".to_string();
                        matched_stop = stop_decoder
                            .matched_stop()
                            .map(|s| Value::String(s.to_string()));
                    }

                    if chunk_text.is_empty() {
                        continue;
                    }

                    // Apply reasoning parser
                    let (normal_text, reasoning_chunk_text, in_reasoning) =
                        if reasoning_parser_available {
                            self.process_messages_reasoning(
                                &chunk_text,
                                &mut reasoning_parser,
                                thinking_override,
                                think_in_prefill,
                                model,
                            )
                            .await
                        } else {
                            (chunk_text, String::new(), false)
                        };

                    // Emit thinking content block deltas
                    if !reasoning_chunk_text.is_empty() {
                        if !thinking_block_open {
                            Self::send_messages_event(
                                tx,
                                &mut sse_buffer,
                                &MessageStreamEvent::ContentBlockStart {
                                    index: current_block_index,
                                    content_block: ContentBlock::Thinking {
                                        thinking: String::new(),
                                        signature: String::new(),
                                    },
                                },
                            )?;
                            thinking_block_open = true;
                        }
                        Self::send_messages_event(
                            tx,
                            &mut sse_buffer,
                            &MessageStreamEvent::ContentBlockDelta {
                                index: current_block_index,
                                delta: ContentBlockDelta::ThinkingDelta {
                                    thinking: reasoning_chunk_text,
                                },
                            },
                        )?;
                    }

                    // Transition: reasoning ended, close thinking block
                    if thinking_block_open && !in_reasoning && !normal_text.is_empty() {
                        Self::send_messages_event(
                            tx,
                            &mut sse_buffer,
                            &MessageStreamEvent::ContentBlockStop {
                                index: current_block_index,
                            },
                        )?;
                        thinking_block_open = false;
                        current_block_index += 1;
                    }

                    // Tool call handling: incremental streaming parser
                    if !in_reasoning && streaming_tool_parser.is_some() {
                        if is_specific_function {
                            // Specific function: entire output is arguments for one tool
                            if !has_tool_calls {
                                has_tool_calls = true;
                                // Close text block if open before starting tool block
                                if text_block_open {
                                    Self::send_messages_event(
                                        tx,
                                        &mut sse_buffer,
                                        &MessageStreamEvent::ContentBlockStop {
                                            index: current_block_index,
                                        },
                                    )?;
                                    text_block_open = false;
                                    current_block_index += 1;
                                }
                                // Emit content_block_start for the tool_use
                                let tool_name = match &original_request.tool_choice {
                                    Some(messages::ToolChoice::Tool { name, .. }) => name.clone(),
                                    _ => String::new(),
                                };
                                let tool_call_id = utils::generate_tool_call_id(
                                    model,
                                    &tool_name,
                                    0,
                                    history_tool_calls_count,
                                );
                                Self::send_messages_event(
                                    tx,
                                    &mut sse_buffer,
                                    &MessageStreamEvent::ContentBlockStart {
                                        index: current_block_index,
                                        content_block: ContentBlock::ToolUse {
                                            id: tool_call_id,
                                            name: tool_name,
                                            input: Value::Object(serde_json::Map::new()),
                                        },
                                    },
                                )?;
                                tool_block_open = true;
                            }
                            // Emit arguments delta
                            if !normal_text.is_empty() {
                                Self::send_messages_event(
                                    tx,
                                    &mut sse_buffer,
                                    &MessageStreamEvent::ContentBlockDelta {
                                        index: current_block_index,
                                        delta: ContentBlockDelta::InputJsonDelta {
                                            partial_json: normal_text,
                                        },
                                    },
                                )?;
                            }
                        } else if let Some(ref mut parser) = streaming_tool_parser {
                            // Regular/required tool choice: use incremental parser
                            match parser.parse_incremental(&normal_text, &chat_tools).await {
                                Ok(StreamingParseResult {
                                    normal_text: text,
                                    calls,
                                }) => {
                                    // Emit normal text from parser as text content blocks
                                    if !text.is_empty() {
                                        if !text_block_open {
                                            Self::send_messages_event(
                                                tx,
                                                &mut sse_buffer,
                                                &MessageStreamEvent::ContentBlockStart {
                                                    index: current_block_index,
                                                    content_block: ContentBlock::Text {
                                                        text: String::new(),
                                                        citations: None,
                                                    },
                                                },
                                            )?;
                                            text_block_open = true;
                                        }
                                        Self::send_messages_event(
                                            tx,
                                            &mut sse_buffer,
                                            &MessageStreamEvent::ContentBlockDelta {
                                                index: current_block_index,
                                                delta: ContentBlockDelta::TextDelta { text },
                                            },
                                        )?;
                                    }

                                    // Emit tool call events
                                    for tool_call_item in calls {
                                        has_tool_calls = true;

                                        if let Some(ref name) = tool_call_item.name {
                                            // New tool call: close previous blocks, emit start
                                            if text_block_open {
                                                Self::send_messages_event(
                                                    tx,
                                                    &mut sse_buffer,
                                                    &MessageStreamEvent::ContentBlockStop {
                                                        index: current_block_index,
                                                    },
                                                )?;
                                                text_block_open = false;
                                                current_block_index += 1;
                                            }
                                            if tool_block_open {
                                                Self::send_messages_event(
                                                    tx,
                                                    &mut sse_buffer,
                                                    &MessageStreamEvent::ContentBlockStop {
                                                        index: current_block_index,
                                                    },
                                                )?;
                                                current_block_index += 1;
                                            }

                                            let tool_call_id = utils::generate_tool_call_id(
                                                model,
                                                name,
                                                tool_call_item.tool_index,
                                                history_tool_calls_count,
                                            );
                                            Self::send_messages_event(
                                                tx,
                                                &mut sse_buffer,
                                                &MessageStreamEvent::ContentBlockStart {
                                                    index: current_block_index,
                                                    content_block: ContentBlock::ToolUse {
                                                        id: tool_call_id,
                                                        name: name.clone(),
                                                        input: Value::Object(serde_json::Map::new()),
                                                    },
                                                },
                                            )?;
                                            tool_block_open = true;
                                        }

                                        // Emit incremental arguments
                                        if !tool_call_item.parameters.is_empty() {
                                            Self::send_messages_event(
                                                tx,
                                                &mut sse_buffer,
                                                &MessageStreamEvent::ContentBlockDelta {
                                                    index: current_block_index,
                                                    delta: ContentBlockDelta::InputJsonDelta {
                                                        partial_json: tool_call_item.parameters,
                                                    },
                                                },
                                            )?;
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Tool call parsing error in messages streaming: {}", e);
                                }
                            }
                        }
                        continue;
                    }

                    // Regular text emission (no tools active)
                    if !normal_text.is_empty() {
                        if !text_block_open {
                            Self::send_messages_event(
                                tx,
                                &mut sse_buffer,
                                &MessageStreamEvent::ContentBlockStart {
                                    index: current_block_index,
                                    content_block: ContentBlock::Text {
                                        text: String::new(),
                                        citations: None,
                                    },
                                },
                            )?;
                            text_block_open = true;
                        }
                        Self::send_messages_event(
                            tx,
                            &mut sse_buffer,
                            &MessageStreamEvent::ContentBlockDelta {
                                index: current_block_index,
                                delta: ContentBlockDelta::TextDelta { text: normal_text },
                            },
                        )?;
                    }
                }
                ProtoResponseVariant::Complete(complete) => {
                    // Flush stop decoder
                    if let SequenceDecoderOutput::Text(text) = stop_decoder.flush() {
                        if !text.is_empty() {
                            if !text_block_open {
                                Self::send_messages_event(
                                    tx,
                                    &mut sse_buffer,
                                    &MessageStreamEvent::ContentBlockStart {
                                        index: current_block_index,
                                        content_block: ContentBlock::Text {
                                            text: String::new(),
                                            citations: None,
                                        },
                                    },
                                )?;
                                text_block_open = true;
                            }
                            Self::send_messages_event(
                                tx,
                                &mut sse_buffer,
                                &MessageStreamEvent::ContentBlockDelta {
                                    index: current_block_index,
                                    delta: ContentBlockDelta::TextDelta { text },
                                },
                            )?;
                        }
                    }

                    prompt_tokens = complete.prompt_tokens();
                    completion_tokens.record_complete(&complete);
                    // A local stop-decoder match already pinned "stop"; don't let
                    // the engine's finish reason overwrite it.
                    if !stopped {
                        finish_reason_str = complete.finish_reason().to_string();
                        matched_stop = complete.matched_stop_json();
                    }
                }
                ProtoResponseVariant::None => continue,
            }
        }

        // Phase 3: Flush unstreamed tool args from the incremental parser
        if let Some(ref parser) = streaming_tool_parser {
            if let Some(unstreamed_items) = parser.get_unstreamed_tool_args() {
                for tool_call_item in unstreamed_items {
                    has_tool_calls = true;

                    if let Some(ref name) = tool_call_item.name {
                        // Close text block if open before starting tool block
                        if text_block_open {
                            Self::send_messages_event(
                                tx,
                                &mut sse_buffer,
                                &MessageStreamEvent::ContentBlockStop {
                                    index: current_block_index,
                                },
                            )?;
                            text_block_open = false;
                            current_block_index += 1;
                        }
                        if tool_block_open {
                            Self::send_messages_event(
                                tx,
                                &mut sse_buffer,
                                &MessageStreamEvent::ContentBlockStop {
                                    index: current_block_index,
                                },
                            )?;
                            current_block_index += 1;
                        }

                        let tool_call_id = utils::generate_tool_call_id(
                            model,
                            name,
                            tool_call_item.tool_index,
                            history_tool_calls_count,
                        );
                        Self::send_messages_event(
                            tx,
                            &mut sse_buffer,
                            &MessageStreamEvent::ContentBlockStart {
                                index: current_block_index,
                                content_block: ContentBlock::ToolUse {
                                    id: tool_call_id,
                                    name: name.clone(),
                                    input: Value::Object(serde_json::Map::new()),
                                },
                            },
                        )?;
                        tool_block_open = true;
                    }

                    if !tool_call_item.parameters.is_empty() {
                        Self::send_messages_event(
                            tx,
                            &mut sse_buffer,
                            &MessageStreamEvent::ContentBlockDelta {
                                index: current_block_index,
                                delta: ContentBlockDelta::InputJsonDelta {
                                    partial_json: tool_call_item.parameters,
                                },
                            },
                        )?;
                    }
                }
            }
        }

        // Phase 3.5: Close any open content blocks
        if thinking_block_open {
            Self::send_messages_event(
                tx,
                &mut sse_buffer,
                &MessageStreamEvent::ContentBlockStop {
                    index: current_block_index,
                },
            )?;
            current_block_index += 1;
        }

        if text_block_open {
            Self::send_messages_event(
                tx,
                &mut sse_buffer,
                &MessageStreamEvent::ContentBlockStop {
                    index: current_block_index,
                },
            )?;
            current_block_index += 1;
        }

        if tool_block_open {
            Self::send_messages_event(
                tx,
                &mut sse_buffer,
                &MessageStreamEvent::ContentBlockStop {
                    index: current_block_index,
                },
            )?;
        }

        // Phase 4: Emit message_delta with stop_reason and usage
        let stop_reason = if has_tool_calls || finish_reason_str == "tool_calls" {
            Some(messages::StopReason::ToolUse)
        } else if matched_stop.is_some() {
            Some(messages::StopReason::StopSequence)
        } else if finish_reason_str == "length" {
            Some(messages::StopReason::MaxTokens)
        } else {
            Some(messages::StopReason::EndTurn)
        };

        let stop_sequence = if matches!(stop_reason, Some(messages::StopReason::StopSequence)) {
            matched_stop.and_then(|v| v.as_str().map(String::from))
        } else {
            None
        };

        Self::send_messages_event(
            tx,
            &mut sse_buffer,
            &MessageStreamEvent::MessageDelta {
                delta: MessageDelta {
                    stop_reason,
                    stop_sequence,
                },
                usage: MessageDeltaUsage {
                    output_tokens: completion_tokens.total(),
                    input_tokens: None,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                    server_tool_use: None,
                },
            },
        )?;

        // Phase 5: Emit message_stop
        Self::send_messages_event(tx, &mut sse_buffer, &MessageStreamEvent::MessageStop)?;

        // Mark stream completed
        grpc_stream.mark_completed();

        if let Some(handle) = reservation {
            handle
                .settle_success(UsageSettlement {
                    actual_input_tokens: prompt_tokens,
                    completion_tokens: completion_tokens.total(),
                })
                .await;
        }

        // Record metrics
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: self.backend_type,
            model_id: model,
            endpoint: metrics_labels::ENDPOINT_MESSAGES,
            ttft: first_token_time.map(|t| t.duration_since(start_time)),
            generation_duration: start_time.elapsed(),
            input_tokens: Some(u64::from(prompt_tokens)),
            output_tokens: u64::from(completion_tokens.total()),
        });

        Ok(())
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
        original_request: Arc<CreateMessageRequest>,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), String> {
        // Consume prefill stream (Messages API does not expose prompt logprobs)
        while let Some(response) = prefill_stream.next().await {
            let gen_response =
                response.map_err(|e| format!("Prefill stream error: {}", e.message()))?;
            match gen_response.into_response() {
                ProtoResponseVariant::Complete(_) => break,
                _ => continue,
            }
        }

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
    /// metrics record, and `[DONE]`.
    pub fn process_completion_streaming_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        completion_request: Arc<CompletionRequest>,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Response {
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        let units = match Self::completion_stream_units(execution_result) {
            Ok(units) => units,
            Err(message) => {
                utils::send_error_sse(&tx, message, "invalid_request_error");
                let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                return build_sse_response(rx);
            }
        };

        let processor = self;
        #[expect(
            clippy::disallowed_methods,
            reason = "streaming task is fire-and-forget; client disconnect terminates it"
        )]
        tokio::spawn(async move {
            let start_time = Instant::now();
            let choices_per_prompt = completion_request.n.unwrap_or(1).max(1);
            let echo = completion_request.echo;
            let include_usage = completion_request
                .stream_options
                .as_ref()
                .and_then(|opts| opts.include_usage)
                .unwrap_or(false);
            let prompt_texts: Vec<&str> = match &completion_request.prompt {
                StringOrArray::String(text) => vec![text.as_str()],
                StringOrArray::Array(texts) => texts.iter().map(String::as_str).collect(),
            };

            // Fail-fast: the first stream error cancels the remaining units
            // (their streams abort on drop) and fails the whole request.
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
                    let completion_request = completion_request.clone();
                    let prompt_text = if echo {
                        prompt_texts.get(prompt_index).copied().unwrap_or_default()
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
                            CompletionStreamUnit::PrefillDecode { prefill, decode } => {
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
                                    )
                                    .await
                            }
                        }
                    }
                }))
                .await;

            match outcomes {
                Err(e) => {
                    utils::send_error_sse(&tx, &e, "internal_error");
                }
                Ok(outcomes) => {
                    let mut total_prompt = 0u32;
                    let mut total_cached = 0u32;
                    let mut total_reasoning = 0u32;
                    let mut total_completion = 0u32;
                    let mut first_token_time: Option<Instant> = None;
                    for outcome in outcomes {
                        total_prompt += outcome.prompt_tokens;
                        total_cached += outcome.cached_tokens;
                        total_reasoning += outcome.reasoning_tokens;
                        total_completion += outcome.completion_tokens;
                        first_token_time = match (first_token_time, outcome.first_token_time) {
                            (Some(current), Some(candidate)) => Some(current.min(candidate)),
                            (current, candidate) => current.or(candidate),
                        };
                    }

                    // All units succeeded (fail-fast `try_join_all` above), so
                    // this is the one settle point for every mode (Single/PD/Batch).
                    if let Some(handle) = &reservation {
                        handle
                            .settle_success(UsageSettlement {
                                actual_input_tokens: total_prompt,
                                completion_tokens: total_completion,
                            })
                            .await;
                    }

                    if include_usage {
                        let usage_chunk = CompletionStreamResponse {
                            id: dispatch.request_id.clone(),
                            object: "text_completion".to_string(),
                            created: dispatch.created,
                            choices: vec![],
                            model: dispatch.model.clone(),
                            system_fingerprint: dispatch.weight_version.clone(),
                            usage: Some(Self::build_completion_streaming_usage(
                                total_prompt,
                                total_completion,
                                total_cached,
                                total_reasoning,
                            )),
                        };
                        let mut sse_buffer = Vec::with_capacity(256);
                        Self::format_completion_sse_into(&mut sse_buffer, &usage_chunk);
                        let _ = tx.send(Ok(Bytes::from(sse_buffer)));
                    }
                    Metrics::record_streaming_metrics(StreamingMetricsParams {
                        router_type: metrics_labels::ROUTER_GRPC,
                        backend_type: processor.backend_type,
                        model_id: &dispatch.model,
                        endpoint: metrics_labels::ENDPOINT_COMPLETIONS,
                        ttft: first_token_time.map(|t| t.duration_since(start_time)),
                        generation_duration: start_time.elapsed(),
                        input_tokens: Some(total_prompt as u64),
                        output_tokens: total_completion as u64,
                    });
                }
            }

            let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
        });

        build_sse_response(rx)
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
                ..
            } => Ok(vec![CompletionStreamUnit::PrefillDecode {
                prefill,
                decode,
            }]),
            context::ExecutionResult::Batch { results } => results
                .into_iter()
                .map(|result| match result {
                    context::ExecutionResult::Single { stream } => {
                        Ok(CompletionStreamUnit::Single(stream))
                    }
                    context::ExecutionResult::PrefillDecode {
                        prefill, decode, ..
                    } => Ok(CompletionStreamUnit::PrefillDecode { prefill, decode }),
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
    /// usage chunk and metrics record.
    #[expect(clippy::too_many_arguments)]
    async fn process_completion_streaming_chunks(
        &self,
        mut grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool, bool),
        completion_request: Arc<CompletionRequest>,
        prompt_text: &str,
        index_offset: u32,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<CompletionStreamOutcome, String> {
        let mut first_token_time: Option<Instant> = None;

        let request_id = &dispatch.request_id;
        let model = &dispatch.model;
        let created = dispatch.created;
        let system_fingerprint = dispatch.weight_version.as_deref();

        let echo = completion_request.echo;
        let suffix = completion_request.suffix.as_deref();
        // TODO: wire per-token logprob streaming when backend support is available
        let _request_logprobs = completion_request.logprobs.is_some();

        let mut stop_decoders: HashMap<u32, StopSequenceDecoder> = HashMap::new();
        let mut is_firsts: HashMap<u32, bool> = HashMap::new();
        let mut stopped_indices: HashSet<u32> = HashSet::new();
        let mut sse_buffer = Vec::with_capacity(512);
        let mut chunk_text = String::new();
        // For n>1, each index shares the same prompt — use max across Complete
        // messages rather than summing (same prompt tokenized once, not per-choice).
        let mut total_prompt = 0u32;
        let mut total_cached = 0u32;
        let mut reasoning_tokens: HashMap<u32, u32> = HashMap::new();
        let mut total_completion = CompletionTokenTracker::new();

        while let Some(response) = grpc_stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e.message()))?;

            match gen_response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => {
                    if first_token_time.is_none() {
                        first_token_time = Some(Instant::now());
                    }

                    let index = index_offset + chunk.index();

                    if stopped_indices.contains(&index) {
                        continue;
                    }

                    let is_first = is_firsts.entry(index).or_insert(true);
                    total_completion.record_chunk(&chunk);

                    let stop_decoder = stop_decoders.entry(index).or_insert_with(|| {
                        let (
                            ref stop,
                            ref stop_token_ids,
                            skip_special_tokens,
                            no_stop_trim,
                            ignore_eos,
                        ) = stop_params;
                        utils::create_stop_decoder(
                            &tokenizer,
                            stop.as_ref(),
                            stop_token_ids.as_ref(),
                            skip_special_tokens,
                            no_stop_trim,
                            ignore_eos,
                        )
                    });

                    let (decoded_text, stopped) =
                        Self::process_chunk_tokens(stop_decoder, chunk.token_ids());
                    chunk_text.clear();
                    chunk_text.push_str(&decoded_text);

                    if *is_first {
                        if echo {
                            chunk_text.insert_str(0, prompt_text);
                        }
                        *is_first = false;
                    }

                    if !chunk_text.is_empty() {
                        let stream_resp = CompletionStreamResponse {
                            id: request_id.clone(),
                            object: "text_completion".to_string(),
                            created,
                            choices: vec![CompletionStreamChoice {
                                text: std::mem::take(&mut chunk_text),
                                index,
                                logprobs: None,
                                finish_reason: None,
                            }],
                            model: model.clone(),
                            system_fingerprint: system_fingerprint.map(String::from),
                            usage: None,
                        };

                        Self::format_completion_sse_into(&mut sse_buffer, &stream_resp);
                        tx.send(Ok(Bytes::from(sse_buffer.clone())))
                            .map_err(|_| "Channel closed".to_string())?;
                    }

                    if stopped {
                        // Stop-decoder match takes precedence: emit "stop" even if
                        // the backend's eventual Complete carries "length". This is
                        // intentional — the local stop sequence fired first.
                        stopped_indices.insert(index);

                        if let Some(sfx) = suffix {
                            let suffix_chunk = CompletionStreamResponse {
                                id: request_id.clone(),
                                object: "text_completion".to_string(),
                                created,
                                choices: vec![CompletionStreamChoice {
                                    text: sfx.to_string(),
                                    index,
                                    logprobs: None,
                                    finish_reason: None,
                                }],
                                model: model.clone(),
                                system_fingerprint: system_fingerprint.map(String::from),
                                usage: None,
                            };
                            Self::format_completion_sse_into(&mut sse_buffer, &suffix_chunk);
                            tx.send(Ok(Bytes::from(sse_buffer.clone())))
                                .map_err(|_| "Channel closed".to_string())?;
                        }

                        let final_chunk = CompletionStreamResponse {
                            id: request_id.clone(),
                            object: "text_completion".to_string(),
                            created,
                            choices: vec![CompletionStreamChoice {
                                text: String::new(),
                                index,
                                logprobs: None,
                                finish_reason: Some("stop".to_string()),
                            }],
                            model: model.clone(),
                            system_fingerprint: system_fingerprint.map(String::from),
                            usage: None,
                        };
                        Self::format_completion_sse_into(&mut sse_buffer, &final_chunk);
                        tx.send(Ok(Bytes::from(sse_buffer.clone())))
                            .map_err(|_| "Channel closed".to_string())?;
                    }
                }
                ProtoResponseVariant::Complete(complete) => {
                    let index = index_offset + complete.index();
                    total_prompt = total_prompt.max(complete.prompt_tokens());
                    total_cached = total_cached.max(complete.cached_tokens());
                    reasoning_tokens.insert(index, complete.reasoning_tokens());
                    total_completion.record_complete(&complete);

                    if stopped_indices.contains(&index) {
                        continue;
                    }

                    // Handle echo when Complete arrives without any preceding Chunks
                    // (e.g., max_tokens=0). The Chunk arm normally prepends prompt_text
                    // on the first event, but if no Chunks arrive we must emit it here.
                    let is_first = is_firsts.entry(index).or_insert(true);
                    if *is_first && echo && !prompt_text.is_empty() {
                        let echo_chunk = CompletionStreamResponse {
                            id: request_id.clone(),
                            object: "text_completion".to_string(),
                            created,
                            choices: vec![CompletionStreamChoice {
                                text: prompt_text.to_string(),
                                index,
                                logprobs: None,
                                finish_reason: None,
                            }],
                            model: model.clone(),
                            system_fingerprint: system_fingerprint.map(String::from),
                            usage: None,
                        };
                        Self::format_completion_sse_into(&mut sse_buffer, &echo_chunk);
                        tx.send(Ok(Bytes::from(sse_buffer.clone())))
                            .map_err(|_| "Channel closed".to_string())?;
                        *is_first = false;
                    }

                    if let Some(decoder) = stop_decoders.get_mut(&index) {
                        if let SequenceDecoderOutput::Text(text) = decoder.flush() {
                            if !text.is_empty() {
                                let stream_resp = CompletionStreamResponse {
                                    id: request_id.clone(),
                                    object: "text_completion".to_string(),
                                    created,
                                    choices: vec![CompletionStreamChoice {
                                        text,
                                        index,
                                        logprobs: None,
                                        finish_reason: None,
                                    }],
                                    model: model.clone(),
                                    system_fingerprint: system_fingerprint.map(String::from),
                                    usage: None,
                                };
                                Self::format_completion_sse_into(&mut sse_buffer, &stream_resp);
                                tx.send(Ok(Bytes::from(sse_buffer.clone())))
                                    .map_err(|_| "Channel closed".to_string())?;
                            }
                        }
                    }

                    if let Some(sfx) = suffix {
                        let stream_resp = CompletionStreamResponse {
                            id: request_id.clone(),
                            object: "text_completion".to_string(),
                            created,
                            choices: vec![CompletionStreamChoice {
                                text: sfx.to_string(),
                                index,
                                logprobs: None,
                                finish_reason: None,
                            }],
                            model: model.clone(),
                            system_fingerprint: system_fingerprint.map(String::from),
                            usage: None,
                        };
                        Self::format_completion_sse_into(&mut sse_buffer, &stream_resp);
                        tx.send(Ok(Bytes::from(sse_buffer.clone())))
                            .map_err(|_| "Channel closed".to_string())?;
                    }

                    let finish_reason = {
                        let reason = complete.finish_reason();
                        if reason.is_empty() || reason == "stop" {
                            Some("stop".to_string())
                        } else if reason == "length" || reason == "content_filter" {
                            Some(reason.to_string())
                        } else if let Ok(json) = serde_json::from_str::<Value>(reason) {
                            json.get("type")
                                .and_then(|v| v.as_str())
                                .map(|t| match t {
                                    "stop" | "length" | "content_filter" => t.to_string(),
                                    other => {
                                        warn!(unexpected_finish_reason = other, "Unmapped finish_reason type from backend, defaulting to stop");
                                        "stop".to_string()
                                    }
                                })
                                .or_else(|| Some("stop".to_string()))
                        } else {
                            warn!(
                                unexpected_finish_reason = reason,
                                "Unrecognized finish_reason from backend, defaulting to stop"
                            );
                            Some("stop".to_string())
                        }
                    };

                    let final_chunk = CompletionStreamResponse {
                        id: request_id.clone(),
                        object: "text_completion".to_string(),
                        created,
                        choices: vec![CompletionStreamChoice {
                            text: String::new(),
                            index,
                            logprobs: None,
                            finish_reason,
                        }],
                        model: model.clone(),
                        system_fingerprint: system_fingerprint.map(String::from),
                        usage: None,
                    };
                    Self::format_completion_sse_into(&mut sse_buffer, &final_chunk);
                    tx.send(Ok(Bytes::from(sse_buffer.clone())))
                        .map_err(|_| "Channel closed".to_string())?;
                }
                ProtoResponseVariant::None => continue,
            }
        }

        grpc_stream.mark_completed();

        Ok(CompletionStreamOutcome {
            prompt_tokens: total_prompt,
            cached_tokens: total_cached,
            reasoning_tokens: reasoning_tokens.values().sum(),
            completion_tokens: total_completion.total(),
            first_token_time,
        })
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
        original_request: Arc<CompletionRequest>,
        prompt_text: &str,
        index_offset: u32,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<CompletionStreamOutcome, String> {
        while let Some(response) = prefill_stream.next().await {
            let gen_response =
                response.map_err(|e| format!("Prefill stream error: {}", e.message()))?;

            match gen_response.into_response() {
                ProtoResponseVariant::Complete(_) => break,
                _ => continue,
            }
        }

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

        // Mark prefill stream as completed AFTER decode completes successfully
        // This ensures that if client disconnects during decode, BOTH streams send abort
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

    fn build_completion_streaming_usage(
        total_prompt: u32,
        total_completion: u32,
        total_cached: u32,
        total_reasoning: u32,
    ) -> Usage {
        Usage::from_counts(total_prompt, total_completion)
            .with_cached_tokens(total_cached)
            .with_reasoning_tokens(total_reasoning)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_streaming_usage_includes_reasoning_tokens() {
        let usage = StreamingProcessor::build_completion_streaming_usage(10, 5, 4, 3);

        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        assert_eq!(
            usage
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens),
            Some(4)
        );
        assert_eq!(
            usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens),
            Some(3)
        );
    }
}
