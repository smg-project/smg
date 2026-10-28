//! Harmony streaming response processor

use std::{
    collections::{hash_map::Entry::Vacant, HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use axum::response::Response;
use openai_protocol::{
    chat::{
        ChatCompletionRequest, ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice,
    },
    common::{ChatLogProbs, FunctionCallDelta, ToolCall, ToolCallDelta},
    responses::{
        InputTokensDetails, OutputTokensDetails, ResponseStatus, ResponseUsage, ResponsesResponse,
        ResponsesUsage,
    },
};
use serde_json::json;
use smg_mcp::{McpToolSession, DEFAULT_SERVER_LABEL};
use tracing::{debug, error};

use super::{
    builder::{convert_harmony_logprobs, try_harmony_encoding},
    processor::{chat_usage, iteration_usage, ResponsesIterationResult},
    stop::TextStopScanner,
    types::{HarmonyChannelDelta, ToolCallDelta as HarmonyToolCallDelta},
    HarmonyParserAdapter,
};
use crate::{
    observability::metrics::{metrics_labels, LocalStopAbort, Metrics, StreamingMetricsParams},
    rate_limit::SharedReservationHandle,
    routers::{
        common::{
            openai_bridge::{self, descriptor, FormatRegistry, ResponseFormat},
            sse::{sse_channel, SseEncoder, SseSender},
        },
        grpc::{
            common::{
                response_collection::drain_prefill,
                response_formatting::{CompletionTokenTracker, TokenCounts},
                responses::{
                    body_guards, build_sse_response_guarded,
                    streaming::{
                        attach_mcp_server_label, OutputItemKind, ResponseStreamEventEmitter,
                    },
                },
            },
            context,
            proto_wrapper::{
                ProtoGenerateComplete, ProtoGenerateResponse, ProtoGenerateStreamChunk,
                ProtoResponseVariant, ProtoStream,
            },
            utils,
        },
    },
    worker::PrefillLoadGuard,
};

/// Whether a tool call of this `ResponseFormat` streams its arguments via
/// `mcp_call.arguments.delta` / `function_call.arguments.delta` events.
///
/// Hosted built-in tools (`web_search_call`, `code_interpreter_call`,
/// `file_search_call`, `image_generation_call`) instead surface their
/// progress through structured events emitted by the shared
/// [`ResponseStreamEventEmitter`] helpers (`emit_tool_call_in_progress`,
/// `emit_tool_call_searching`, `emit_tool_call_completed`). Those
/// builtins therefore skip argument streaming here.
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

/// The error a Responses iteration ends with when its client left (see
/// [`next_response`]): its callers end the stream on their error path, the
/// one they take today when the client's leaving is met at a send.
const CLIENT_LEFT: &str = "Client disconnected";

/// One read of the engine stream for the client of a Responses iteration
/// (see [`next_response`]).
enum EngineRead {
    /// A response of the stream, or the error the read met.
    Response(Result<ProtoGenerateResponse, tonic::Status>),
    /// The stream ended.
    End,
    /// The client left before the stream yielded.
    ClientLeft,
}

/// The next response of the engine `stream`, read for the client at the
/// receiving end of `tx`'s channel. A Responses iteration reads the engine
/// stream itself and sends its events best-effort: parked on a generation
/// that stalled mid-stream, it would learn that its client left only when
/// the engine next spoke, and a generation nobody reads would run to its end
/// and be stored as completed. Here every read races the engine against the
/// client's leaving, looked at first so a busy engine cannot outrun it; on
/// [`EngineRead::ClientLeft`] the caller returns, the stream is dropped
/// unmarked with it, and its abort-on-drop ends the generation on the
/// engine. A background response's `tx` is the sender of its relay, which
/// closes with the work and not with the client: its reads are the engine's
/// own, and the work outlives the client as it is meant to.
async fn next_response(stream: &mut ProtoStream, tx: &SseSender) -> EngineRead {
    tokio::select! {
        biased;
        () = tx.closed() => EngineRead::ClientLeft,
        item = stream.next() => item.map_or(EngineRead::End, EngineRead::Response),
    }
}

impl HarmonyStreamingProcessor {
    /// Create a new Harmony streaming processor
    pub fn new() -> Self {
        Self
    }

    /// Process a streaming Harmony Chat Completion response
    ///
    /// Returns an SSE response with streaming token updates.
    ///
    /// The `load_guards` (and the reservation's safety net) ride in the
    /// response body, released when it is dropped: at the stream's end or
    /// the client's disconnect.
    #[expect(
        clippy::disallowed_methods,
        reason = "streaming tasks are fire-and-forget by design; client disconnect terminates them"
    )]
    #[expect(clippy::too_many_arguments)]
    /// `router_stop_strings` is non-empty only when the router must enforce
    /// string `stop` sequences itself (direct-ZMQ backends: the engine sees
    /// token ids only).
    pub async fn process_streaming_chat_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: context::DispatchMetadata,
        router_stop_strings: Vec<String>,
        reservation: Option<Arc<SharedReservationHandle>>,
        load_guards: Option<context::LoadGuards>,
        source: utils::StreamSource,
    ) -> Response {
        let source = source.with_backend(metrics_labels::BACKEND_HARMONY);
        let guards = body_guards(load_guards, reservation.as_ref());
        // Create SSE channel
        let (tx, rx) = sse_channel();
        // Read for this client: the moment the response body is dropped, the
        // request is aborted on the engine (see `ProtoStream::until_client_gone`).
        let execution_result = execution_result.until_client_gone(&tx, &dispatch.request_id);

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

                    if let Err(failure) = result {
                        utils::send_stream_failure(&tx, &failure, &source).await;
                    }

                    let _ = tx.send(Ok(SseEncoder::done())).await;
                });
            }
            context::ExecutionResult::PrefillDecode {
                prefill,
                decode,
                prefill_guards,
                pd_timing,
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
                        pd_timing,
                        prefill_guards,
                        reservation,
                    )
                    .await;

                    if let Err(failure) = result {
                        utils::send_stream_failure(&tx, &failure, &source).await;
                    }

                    let _ = tx.send(Ok(SseEncoder::done())).await;
                });
            }
            context::ExecutionResult::Embedding { .. } => {
                error!("Harmony streaming not supported for embeddings");
                utils::send_error_sse(
                    &tx,
                    "Embeddings not supported in Harmony streaming",
                    "invalid_request_error",
                )
                .await;
                let _ = tx.send(Ok(SseEncoder::done())).await;
            }
            // Batch results exist only on the completions pipeline.
            context::ExecutionResult::Batch { .. } => {
                error!("Harmony streaming not supported for batched results");
                utils::send_error_sse(
                    &tx,
                    "Batched results not supported in Harmony streaming",
                    "invalid_request_error",
                )
                .await;
                let _ = tx.send(Ok(SseEncoder::done())).await;
            }
        }

        // Return SSE response
        build_sse_response_guarded(rx, guards)
    }

    /// Process streaming chunks from a single stream
    async fn process_single_stream(
        grpc_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        original_request: Arc<ChatCompletionRequest>,
        tx: &SseSender,
        router_stop_strings: Vec<String>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), utils::StreamFailure> {
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
            None,
            reservation,
        )
        .await
    }

    /// Process streaming chunks from prefill/decode streams (prefill + decode)
    #[expect(
        clippy::too_many_arguments,
        reason = "streaming PD chat keeps phase streams, metadata, guards and reservation separate"
    )]
    async fn process_prefill_decode_stream(
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        dispatch: context::DispatchMetadata,
        original_request: Arc<ChatCompletionRequest>,
        tx: &SseSender,
        router_stop_strings: Vec<String>,
        pd_timing: context::PdTiming,
        prefill_guards: Vec<Option<PrefillLoadGuard>>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), utils::StreamFailure> {
        // Phase 1: Process prefill stream (collect metadata)
        let mut prompt_tokens: HashMap<u32, u32> = HashMap::new();
        let mut cached_tokens: HashMap<u32, u32> = HashMap::new();

        drain_prefill(&mut prefill_stream, prefill_guards, false, |complete| {
            prompt_tokens.insert(complete.index(), complete.prompt_tokens());
            cached_tokens.insert(complete.index(), complete.cached_tokens());
        })
        .await
        .map_err(utils::StreamFailure::prefill)?;

        // Phase 2: Decode (shared helper). Pass pd_timing so the first decode
        // token yields honest PD TTFT.
        Self::process_chat_decode_stream(
            decode_stream,
            &dispatch,
            &original_request,
            tx,
            &mut prompt_tokens,
            &mut cached_tokens,
            &router_stop_strings,
            Some(pd_timing),
            reservation,
        )
        .await?;

        // The prefill stream is marked completed after the decode's read ended
        // well: a decode that failed leaves both streams to abort on drop; a
        // client that left ends the read with `Ok` too, its decode stream
        // aborting the request on drop, the prefill's own request complete.
        prefill_stream.mark_completed();
        Ok(())
    }

    /// Process the decode phase of a Chat Completion stream.
    ///
    /// Shared between single-stream and prefill/decode stream modes. The `prompt_tokens`
    /// and `cached_tokens` maps may be pre-populated from a prefill phase
    /// (prefill/decode stream) or empty (single stream). Values from `Complete` messages
    /// are inserted only if not already present. `pd_timing` is `Some` only in PD mode and
    /// yields honest PD TTFT (prefill start to first decode token).
    #[expect(clippy::too_many_arguments)]
    async fn process_chat_decode_stream(
        mut decode_stream: ProtoStream,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &SseSender,
        prompt_tokens: &mut HashMap<u32, u32>,
        cached_tokens: &mut HashMap<u32, u32>,
        router_stop_strings: &[String],
        pd_timing: Option<context::PdTiming>,
        reservation: Option<Arc<SharedReservationHandle>>,
    ) -> Result<(), utils::StreamFailure> {
        let mut decode = ChatDecode::new(pd_timing);
        while let Some(result) = decode_stream.next().await {
            let response = result.map_err(|e| {
                utils::StreamFailure::upstream(e, decode.completion_tokens.delivered())
            })?;
            match response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => {
                    decode
                        .chunk(
                            &chunk,
                            dispatch,
                            original_request,
                            tx,
                            router_stop_strings,
                            prompt_tokens,
                            cached_tokens,
                        )
                        .await?;
                }
                ProtoResponseVariant::Complete(complete) => {
                    decode
                        .complete(
                            &complete,
                            dispatch,
                            original_request,
                            tx,
                            prompt_tokens,
                            cached_tokens,
                        )
                        .await?;
                }
                ProtoResponseVariant::None => {}
            }
            // An engine that outlived the client's stream (every choice closed, a router-stopped
            // one streaming on past its stop) is not polled again: the driver aborts the engine
            // stream.
            if decode.engine_outlived_now(original_request.choices()) {
                decode.engine_outlived = true;
                break;
            }
        }

        // The stream's verdict, before the engine side is released.
        decode.client_left = decode_stream.client_left();
        decode.release_engine(decode_stream, &original_request.model);

        decode
            .finish(
                prompt_tokens,
                cached_tokens,
                reservation,
                dispatch,
                original_request,
                tx,
            )
            .await
    }

    /// Emit a chunk delta from Harmony channels
    #[expect(clippy::too_many_arguments)]
    async fn emit_chunk_delta(
        delta: &HarmonyChannelDelta,
        index: u32,
        is_first: bool,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &SseSender,
        encoder: &mut SseEncoder,
        logprobs: Option<ChatLogProbs>,
    ) -> Result<(), String> {
        // On first chunk, emit role announcement separately
        if is_first {
            Self::emit_role_chunk(index, dispatch, original_request, tx, encoder).await?;
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
            .await
            .map_err(|_| "Failed to send chunk".to_string())?;

        Ok(())
    }

    /// The chunk that opens a choice: its role, as the engines' own servers
    /// send it before any content.
    async fn emit_role_chunk(
        index: u32,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &SseSender,
        encoder: &mut SseEncoder,
    ) -> Result<(), String> {
        let role_chunk =
            ChatCompletionStreamResponse::builder(&dispatch.request_id, &original_request.model)
                .created(dispatch.created)
                .add_choice_role(index, "assistant")
                .maybe_system_fingerprint(dispatch.weight_version.as_deref())
                .build();

        let sse_data = encoder
            .encode_data(&role_chunk)
            .map_err(|e| format!("JSON serialization error: {e}"))?;

        tx.send(Ok(sse_data))
            .await
            .map_err(|_| "Failed to send role chunk".to_string())?;

        Ok(())
    }

    /// Emit final chunk with finish_reason
    async fn emit_final_chunk(
        index: u32,
        finish_reason: &str,
        matched_stop: Option<&serde_json::Value>,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &SseSender,
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
            .await
            .map_err(|_| "Failed to send final chunk".to_string())?;

        Ok(())
    }

    /// Emit the usage chunk at the end: the usage object the non-streamed
    /// answer carries, from `counts` and the reasoning tokens the parser
    /// counted.
    async fn emit_usage_chunk(
        counts: TokenCounts,
        parsed_reasoning_tokens: u32,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &SseSender,
        encoder: &mut SseEncoder,
    ) -> Result<(), String> {
        let usage_chunk =
            ChatCompletionStreamResponse::builder(&dispatch.request_id, &original_request.model)
                .created(dispatch.created)
                .usage(chat_usage(counts, parsed_reasoning_tokens))
                .maybe_system_fingerprint(dispatch.weight_version.as_deref())
                .build();

        let sse_data = encoder
            .encode_data(&usage_chunk)
            .map_err(|e| format!("JSON serialization error: {e}"))?;

        tx.send(Ok(sse_data))
            .await
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
    ///
    /// Answers the iteration's result with the engine's counts the terminal
    /// frame brought, for the settlement of the iteration's reservation;
    /// `None` when the stream ended before that frame (a clean EOF): the
    /// usage the result carries is then the chunks' count at most, and the
    /// caller keeps the reserved debit instead of settling with it.
    pub async fn process_responses_iteration_stream(
        execution_result: context::ExecutionResult,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &SseSender,
        session: Option<&McpToolSession<'_>>,
        format_registry: Option<&FormatRegistry>,
    ) -> Result<(ResponsesIterationResult, Option<TokenCounts>), String> {
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                debug!("Processing Responses API single stream mode");
                Self::process_decode_stream(stream, emitter, tx, session, format_registry, None)
                    .await
            }
            context::ExecutionResult::PrefillDecode {
                prefill,
                decode,
                prefill_guards,
                pd_timing,
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
                    pd_timing,
                    prefill_guards,
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

    #[expect(clippy::too_many_arguments)]
    async fn process_responses_prefill_decode_stream(
        mut prefill_stream: ProtoStream,
        decode_stream: ProtoStream,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &SseSender,
        session: Option<&McpToolSession<'_>>,
        format_registry: Option<&FormatRegistry>,
        pd_timing: context::PdTiming,
        prefill_guards: Vec<Option<PrefillLoadGuard>>,
    ) -> Result<(ResponsesIterationResult, Option<TokenCounts>), String> {
        // Phase 1: Drain prefill stream, for the client (see `next_response`):
        // a client that leaves during the prefill ends both sides at once. The
        // client's prompt-side usage is the prefill side's prefix-cache hit:
        // the decode side counts the KV it was handed as cached, so its
        // responses are restamped with the prefill side's number before the
        // decode phase reads them.
        let drained = tokio::select! {
            biased;
            () = tx.closed() => None,
            drained = drain_prefill(&mut prefill_stream, prefill_guards, false, |_| {}) => {
                Some(drained)
            }
        };
        let Some(drained) = drained else {
            // Both sides go with this return, unmarked: their abort-on-drop
            // ends the request on the engines.
            debug!(
                response_id = %emitter.response_id,
                "Client left during the prefill of the Responses stream; the request is aborted on the engines"
            );
            return Err(CLIENT_LEFT.to_string());
        };
        let prefill_cached_tokens =
            drained.map_err(|e| format!("Prefill stream error: {}", e.message()))?;
        let decode_stream = decode_stream.decode_side(prefill_cached_tokens);

        // Phase 2: Process decode stream. Pass pd_timing so the first decode
        // frame records honest PD TTFT.
        let result = Self::process_decode_stream(
            decode_stream,
            emitter,
            tx,
            session,
            format_registry,
            Some(pd_timing),
        )
        .await;

        prefill_stream.mark_completed();
        result
    }

    /// Process decode stream for tool call events. `pd_timing` is `Some` only
    /// in PD mode and yields honest PD TTFT (prefill start to first decode
    /// frame).
    async fn process_decode_stream(
        mut decode_stream: ProtoStream,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &SseSender,
        session: Option<&McpToolSession<'_>>,
        format_registry: Option<&FormatRegistry>,
        pd_timing: Option<context::PdTiming>,
    ) -> Result<(ResponsesIterationResult, Option<TokenCounts>), String> {
        let mut decode = ResponsesDecode::new(pd_timing)?;
        // Each read races the engine against the client's leaving (see `next_response`).
        loop {
            let result = match next_response(&mut decode_stream, tx).await {
                EngineRead::Response(result) => result,
                EngineRead::End => break,
                // The decode stream goes with this return, unmarked: its
                // abort-on-drop ends the generation on the engine. The
                // iteration ends as it does when the client's leaving is met
                // at a send, and nothing of it is stored.
                EngineRead::ClientLeft => {
                    debug!(
                        response_id = %emitter.response_id,
                        "Client left during the Responses stream; the request is aborted on the engine"
                    );
                    return Err(CLIENT_LEFT.to_string());
                }
            };
            decode.frames += 1;
            let response = result.map_err(|e| format!("Decode stream error: {}", e.message()))?;
            match response.into_response() {
                ProtoResponseVariant::Chunk(chunk) => {
                    decode
                        .chunk(&chunk, emitter, tx, session, format_registry)
                        .await?;
                }
                ProtoResponseVariant::Complete(complete) => {
                    decode.complete(&complete, emitter, tx, session).await;
                }
                ProtoResponseVariant::None => {}
            }
        }
        decode.recover_calls(emitter, tx, session).await;

        // Mark stream as completed successfully to prevent abort on drop
        decode_stream.mark_completed();

        let counts = decode.settlement_counts();
        Ok((decode.result(emitter), counts))
    }
}

/// One Responses iteration's decode stream as it is read: the parser, what the output has shown
/// so far, the items opened for it, and the counters its terminal frame brings. Each frame goes
/// through [`Self::chunk`] or [`Self::complete`]; a stream whose terminal frame brought no calls,
/// or that had none, has its calls recovered after the loop ([`Self::recover_calls`]);
/// [`Self::result`] is the iteration's.
struct ResponsesDecode {
    parser: HarmonyParserAdapter,
    /// The frames read, the terminal one included.
    frames: usize,
    /// The analysis channel showed, and its reasoning item was emitted (added and done in one
    /// call; its content comes with the terminal frame).
    has_analysis: bool,
    /// The final channel's text so far.
    final_text: String,
    /// The calls the terminal frame or the recovery found.
    tool_calls: Option<Vec<ToolCall>>,
    /// The message item the first final-channel delta opened: its output index and id.
    message: Option<(usize, String)>,
    content_part_added: bool,
    /// Each call's output index, item id and response format, by the parser's call index.
    calls: HashMap<usize, (usize, String, Option<ResponseFormat>)>,
    finalized_analysis: Option<String>,
    // The terminal frame's counters (a disaggregated request's decode side carries the prefill
    // side's cache hit, see `ProtoStream::decode_side`); completion tokens are counted per chunk on
    // a delta stream and taken from the terminal frame on a cumulative one.
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    /// The reasoning tokens the parser counted.
    reasoning_token_count: u32,
    /// The engine's own reasoning and speculative counters.
    engine_reasoning_tokens: u32,
    spec_accepted: u32,
    spec_drafted: u32,
    /// The PD dispatch instants, in PD mode, until the first decode frame records the PD
    /// first-token pair against them (the iteration records no streaming TTFT of its own).
    pd_timing: Option<context::PdTiming>,
    /// Whether the terminal frame came: the counters above are the engine's only then.
    completed: bool,
}

impl ResponsesDecode {
    fn new(pd_timing: Option<context::PdTiming>) -> Result<Self, String> {
        Ok(Self {
            parser: HarmonyParserAdapter::new()
                .map_err(|e| format!("Failed to create parser: {e}"))?,
            frames: 0,
            has_analysis: false,
            final_text: String::new(),
            tool_calls: None,
            message: None,
            content_part_added: false,
            calls: HashMap::new(),
            finalized_analysis: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            cached_tokens: 0,
            reasoning_token_count: 0,
            engine_reasoning_tokens: 0,
            spec_accepted: 0,
            spec_drafted: 0,
            pd_timing,
            completed: false,
        })
    }

    /// The first decode frame: in PD mode the PD first-token pair, measured from the dispatch
    /// instants, recorded once.
    fn note_first_output(&mut self, model: &str) {
        if let Some(timing) = self.pd_timing.take() {
            timing.record_first_decode_token(metrics_labels::BACKEND_HARMONY, model);
        }
    }

    /// One chunk: its tokens counted on a delta stream, parsed, and each channel's delta emitted
    /// (the analysis channel opens the reasoning item, the final channel the message and its
    /// text, the commentary channel the calls and their arguments).
    async fn chunk(
        &mut self,
        chunk: &ProtoGenerateStreamChunk,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &SseSender,
        session: Option<&McpToolSession<'_>>,
        format_registry: Option<&FormatRegistry>,
    ) -> Result<(), String> {
        self.note_first_output(emitter.model());
        // Track token counts on delta streams; cumulative streams report them in Complete
        // instead.
        if chunk.chunk_semantics().is_delta() {
            self.completion_tokens += chunk.token_ids().len() as u32;
        }
        let Some(delta) = self
            .parser
            .parse_chunk(chunk.token_ids())
            .map_err(|e| format!("Parse error: {e}"))?
        else {
            return Ok(());
        };
        if delta.analysis_delta.is_some() {
            self.reasoning(emitter, tx).await?;
        }
        if let Some(text) = delta.final_delta.as_deref().filter(|text| !text.is_empty()) {
            self.text(text, emitter, tx).await;
        }
        if let Some(call) = &delta.commentary_delta {
            self.commentary(call, emitter, tx, session, format_registry)
                .await;
        }
        Ok(())
    }

    /// The analysis channel's first delta opens the reasoning item (added and done in one call;
    /// its content comes with the terminal frame).
    async fn reasoning(
        &mut self,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &SseSender,
    ) -> Result<(), String> {
        if self.has_analysis {
            return Ok(());
        }
        emitter
            .emit_reasoning_item(tx, None)
            .await
            .map_err(|e| format!("Failed to emit reasoning item: {e}"))?;
        self.has_analysis = true;
        Ok(())
    }

    /// A final-channel delta: the message item opened on the first, its content part before
    /// the first delta, then the delta itself.
    async fn text(&mut self, text: &str, emitter: &mut ResponseStreamEventEmitter, tx: &SseSender) {
        if self.message.is_none() {
            let (output_index, item_id) = emitter.allocate_output_index(OutputItemKind::Message);
            let item = json!({
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": []
            });
            let event = emitter.emit_output_item_added(output_index, &item);
            emitter.send_event_best_effort(&event, tx).await;
            self.message = Some((output_index, item_id));
        }
        let Some((output_index, item_id)) = &self.message else {
            return;
        };
        let content_index = 0; // Single content part
        if !self.content_part_added {
            let event = emitter.emit_content_part_added(*output_index, item_id, content_index);
            emitter.send_event_best_effort(&event, tx).await;
            self.content_part_added = true;
        }
        let event = emitter.emit_text_delta(text, *output_index, item_id, content_index);
        emitter.send_event_best_effort(&event, tx).await;
        self.final_text.push_str(text);
    }

    /// A commentary-channel delta: a call with its id and name opens its item, a continuing
    /// one streams its arguments (the hosted built-in tools skip the deltas; their progress
    /// rides on the structured events around MCP dispatch).
    async fn commentary(
        &mut self,
        call: &HarmonyToolCallDelta,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &SseSender,
        session: Option<&McpToolSession<'_>>,
        format_registry: Option<&FormatRegistry>,
    ) {
        if let Some(call_id) = &call.id {
            self.open_call(call, call_id, emitter, tx, session, format_registry)
                .await;
            return;
        }
        let Some((output_index, item_id, response_format)) = self.calls.get(&call.index) else {
            return;
        };
        if !streams_arguments(response_format.as_ref()) {
            return;
        }
        let Some(args) = call
            .function
            .as_ref()
            .and_then(|f| f.arguments.as_ref())
            .filter(|a| !a.is_empty())
        else {
            return;
        };
        let event = match response_format {
            Some(_) => emitter.emit_mcp_call_arguments_delta(*output_index, item_id, args),
            None => emitter.emit_function_call_arguments_delta(*output_index, item_id, args),
        };
        emitter.send_event_best_effort(&event, tx).await;
    }

    /// A new call: its item allocated for the tool's format (an MCP tool's, where the session
    /// exposes it), `output_item.added`, the MCP progress events, and the first (empty)
    /// arguments delta where the tool streams its arguments.
    async fn open_call(
        &mut self,
        call: &HarmonyToolCallDelta,
        call_id: &str,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &SseSender,
        session: Option<&McpToolSession<'_>>,
        format_registry: Option<&FormatRegistry>,
    ) {
        let tool_name = call
            .function
            .as_ref()
            .and_then(|f| f.name.as_ref())
            .map(|n| n.as_str())
            .unwrap_or("");
        // Determine response_format based on MCP context.
        let response_format = session.and_then(|s| {
            if s.has_exposed_tool(tool_name) {
                format_registry.map(|reg| openai_bridge::lookup_tool_format(s, reg, tool_name))
            } else {
                None
            }
        });
        let type_str = ResponseStreamEventEmitter::type_str_for_format(response_format.as_ref());
        let (output_index, item_id) = emitter.allocate_output_index_for_format(response_format);
        self.calls
            .insert(call.index, (output_index, item_id.clone(), response_format));

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
        attach_mcp_server_label(&mut item, Some(label.as_str()), response_format.as_ref());
        let event = emitter.emit_output_item_added(output_index, &item);
        emitter.send_event_best_effort(&event, tx).await;

        // Emit in_progress event for MCP tools
        if let Some(fmt) = response_format {
            let event = emitter.emit_tool_call_in_progress(output_index, &item_id, fmt);
            emitter.send_event_best_effort(&event, tx).await;
            // Emit searching/interpreting event for builtin tools
            if let Some(event) = emitter.emit_tool_call_searching(output_index, &item_id, fmt) {
                emitter.send_event_best_effort(&event, tx).await;
            }
        }

        // Emit initial arguments delta for mcp_call / function_call only. Hosted built-in
        // tools (web_search_call, code_interpreter_call, file_search_call,
        // image_generation_call) surface progress via the structured `*.in_progress` /
        // `*.searching` / `*.generating` events emitted above instead of streaming their
        // arguments.
        if streams_arguments(response_format.as_ref()) {
            let event = match &response_format {
                Some(_) => emitter.emit_mcp_call_arguments_delta(output_index, &item_id, ""),
                None => emitter.emit_function_call_arguments_delta(output_index, &item_id, ""),
            };
            emitter.send_event_best_effort(&event, tx).await;
        }
    }

    /// The terminal frame: the counters, the parser's final output, then the calls and the
    /// message closed.
    async fn complete(
        &mut self,
        complete: &ProtoGenerateComplete,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &SseSender,
        session: Option<&McpToolSession<'_>>,
    ) {
        self.note_first_output(emitter.model());
        let finish_reason = complete.finish_reason().to_string();
        self.prompt_tokens = complete.prompt_tokens();
        // The prefill side's hit through a prefill/decode pair, the engine's own through a
        // single engine
        self.cached_tokens = complete.cached_tokens();
        // Delta streams keep the count accumulated per chunk; cumulative streams take it from
        // Complete.
        if !complete.chunk_semantics().is_delta() {
            self.completion_tokens = complete.completion_tokens();
        }
        self.engine_reasoning_tokens = complete.reasoning_tokens();
        self.spec_accepted = complete.spec_accepted_tokens();
        self.spec_drafted = complete.spec_draft_tokens();
        self.completed = true;

        // Finalize parser and get complete output
        // Responses API: no user-specified stop sequences
        let final_output = self.parser.finalize(finish_reason);
        self.finalized_analysis = final_output.analysis;
        self.tool_calls = final_output.commentary;
        self.reasoning_token_count = final_output.reasoning_token_count;

        self.close_calls(emitter, tx, session).await;
        self.close_message(emitter, tx).await;
    }

    /// Every call found closes its item: `arguments.done` where the tool streams its arguments
    /// (the hosted built-in tools close out through the `*.completed` structured event instead),
    /// the MCP `completed` event, and `output_item.done` with the call as completed.
    async fn close_calls(
        &self,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &SseSender,
        session: Option<&McpToolSession<'_>>,
    ) {
        let Some(tool_calls) = &self.tool_calls else {
            return;
        };
        for (call_index, tool_call) in tool_calls.iter().enumerate() {
            let Some((output_index, item_id, response_format)) = self.calls.get(&call_index) else {
                continue;
            };
            let tool_name = &tool_call.function.name;
            let args_str = tool_call.function.arguments.as_deref().unwrap_or("");

            if streams_arguments(response_format.as_ref()) {
                let event = match response_format {
                    Some(_) => {
                        emitter.emit_mcp_call_arguments_done(*output_index, item_id, args_str)
                    }
                    None => {
                        emitter.emit_function_call_arguments_done(*output_index, item_id, args_str)
                    }
                };
                emitter.send_event_best_effort(&event, tx).await;
            }

            // Emit completed event for MCP tools
            if let Some(fmt) = *response_format {
                let event = emitter.emit_tool_call_completed(*output_index, item_id, fmt);
                emitter.send_event_best_effort(&event, tx).await;
            }

            let type_str =
                ResponseStreamEventEmitter::type_str_for_format(response_format.as_ref());
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
            attach_mcp_server_label(&mut item, Some(label.as_str()), response_format.as_ref());

            let event = emitter.emit_output_item_done(*output_index, &item);
            emitter.complete_output_item(*output_index);
            emitter.send_event_best_effort(&event, tx).await;
        }
    }

    /// The message item, if one opened, closes: `text_done`, `content_part.done`, and
    /// `output_item.done` with the whole text.
    async fn close_message(&self, emitter: &mut ResponseStreamEventEmitter, tx: &SseSender) {
        let Some((output_index, item_id)) = &self.message else {
            return;
        };
        let content_index = 0;
        let event = emitter.emit_text_done(*output_index, item_id, content_index);
        emitter.send_event_best_effort(&event, tx).await;
        let event = emitter.emit_content_part_done(*output_index, item_id, content_index);
        emitter.send_event_best_effort(&event, tx).await;
        let item = json!({
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": self.final_text.clone()
            }]
        });
        let event = emitter.emit_output_item_done(*output_index, &item);
        // Mark as completed before sending (so it's included in final output even if send fails)
        emitter.complete_output_item(*output_index);
        emitter.send_event_best_effort(&event, tx).await;
    }

    /// After the loop: a stream that brought frames but no calls from a terminal frame has its
    /// calls read from the parser's completed messages, else from an incomplete commentary, and
    /// closed as the terminal frame would have closed them.
    async fn recover_calls(
        &mut self,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &SseSender,
        session: Option<&McpToolSession<'_>>,
    ) {
        debug!(
            "Stream loop ended. Total chunks received: {}, has_analysis: {}, tool_calls: {}, final_text_len: {}",
            self.frames,
            self.has_analysis,
            self.tool_calls.as_ref().map(|tc| tc.len()).unwrap_or(0),
            self.final_text.len()
        );
        if self.frames == 0 || self.tool_calls.is_some() {
            return;
        }
        let messages = self.parser.get_messages();
        // Try extracting from completed messages first
        let (analysis_opt, commentary_opt, final_text_extracted) =
            HarmonyParserAdapter::parse_messages(&messages);
        self.tool_calls.clone_from(&commentary_opt);
        // If no tool calls found, check for incomplete commentary in parser state
        if self.tool_calls.is_none() {
            self.tool_calls = self.parser.extract_incomplete_commentary();
        }
        debug!(
            "Tool call extraction: completed_msgs={}, tool_calls={}, has_analysis={}, final_text_len={}",
            messages.len(),
            self.tool_calls.as_ref().map(|tc| tc.len()).unwrap_or(0),
            analysis_opt.is_some(),
            final_text_extracted.len()
        );
        self.close_calls(emitter, tx, session).await;
    }

    /// The engine's counts the stream brought (the terminal frame's counters).
    fn counts(&self) -> TokenCounts {
        TokenCounts {
            prompt: self.prompt_tokens,
            completion: self.completion_tokens,
            cached: self.cached_tokens,
            reasoning: self.engine_reasoning_tokens,
            spec_accepted: self.spec_accepted,
            spec_drafted: self.spec_drafted,
        }
    }

    /// The counts the iteration's reservation settles with: the terminal frame's, so none when the
    /// stream ended before it (a clean EOF). The caller then keeps the reserved debit, as the chat
    /// streams do, instead of settling the prompt at zero.
    fn settlement_counts(&self) -> Option<TokenCounts> {
        self.completed.then(|| self.counts())
    }

    /// The iteration's result: the calls found, with the analysis and the text so far, or the
    /// completed response; either with the iteration's usage, as the non-streamed iteration
    /// builds it.
    fn result(self, emitter: &ResponseStreamEventEmitter) -> ResponsesIterationResult {
        let usage = iteration_usage(self.counts(), self.reasoning_token_count);
        if let Some(tool_calls) = self.tool_calls.filter(|calls| !calls.is_empty()) {
            return ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis: self
                    .has_analysis
                    .then_some(self.finalized_analysis)
                    .flatten(),
                partial_text: self.final_text,
                usage,
                request_id: emitter.response_id.clone(),
            };
        }
        // For streaming, the caller builds the response from the SSE events; this result's
        // fields are the placeholders it ignores, the usage apart.
        ResponsesIterationResult::Completed {
            response: Box::new(
                ResponsesResponse::builder(&emitter.response_id, "")
                    .status(ResponseStatus::Completed)
                    .usage(ResponsesUsage::Modern(ResponseUsage {
                        input_tokens: self.prompt_tokens,
                        output_tokens: self.completion_tokens,
                        total_tokens: self.prompt_tokens + self.completion_tokens,
                        input_tokens_details: InputTokensDetails {
                            cached_tokens: self.cached_tokens,
                        },
                        output_tokens_details: OutputTokensDetails {
                            reasoning_tokens: self.reasoning_token_count,
                        },
                    }))
                    .build(),
            ),
            usage,
        }
    }
}

/// One Harmony chat decode stream as it is read: a parser and its state per choice index
/// (`n>1`), the router-side stop scanners, the counters the terminal frames bring, and the SSE
/// encoder every chunk goes through. Each frame goes through [`Self::chunk`] or
/// [`Self::complete`]; [`Self::finish`] settles the usage and writes the usage chunk and the
/// metrics once the stream has ended.
struct ChatDecode {
    // Timing for metrics
    start_time: Instant,
    first_token_time: Option<Instant>,
    /// The PD dispatch instants, in PD mode: the first token records the PD pair against
    /// them and the metrics anchor at the prefill dispatch.
    pd_timing: Option<context::PdTiming>,
    // Per-index state management (for n>1 support)
    parsers: HashMap<u32, HarmonyParserAdapter>,
    is_firsts: HashMap<u32, bool>,
    matched_stops: HashMap<u32, Option<serde_json::Value>>,
    // Router-enforced string stops (direct-ZMQ): per-index, per-channel scanners. Once an index
    // stops, its further deltas are swallowed and the engine's own Complete is not re-emitted.
    analysis_scanners: HashMap<u32, TextStopScanner>,
    final_scanners: HashMap<u32, TextStopScanner>,
    router_stopped: HashSet<u32>,
    completion_tokens: CompletionTokenTracker,
    // The engine's own reasoning and speculative counters per decode `Complete`, and the
    // reasoning tokens the Harmony parser counted over every index: the usage chunk is built
    // from the same counters as the non-streamed answer.
    reasoning_tokens: HashMap<u32, u32>,
    spec_accepted: HashMap<u32, u32>,
    spec_drafted: HashMap<u32, u32>,
    parsed_reasoning_tokens: u32,
    // Indices that received a *decode* `Complete` -- unlike `prompt_tokens` (which may already
    // be populated from the prefill phase before the loop even starts, in PD mode), this is
    // only ever set from this stream's own Complete messages, so it can't be fooled by prefill
    // data into thinking decode produced authoritative usage it didn't.
    decode_completed_indices: HashSet<u32>,
    // Router-stopped indices the engine streamed a chunk for after their stop: it is generating
    // past the stop (an engine that ends the choice on that step sends its Complete next).
    tailed: HashSet<u32>,
    // The engine was streaming past a stop when every choice had closed (its Complete came, or
    // the router's stop ended it): the loop stopped polling it, and the driver aborts it instead
    // of marking it completed (see `release_engine`).
    engine_outlived: bool,
    // Tokens the engine streamed for choices after their router-side stop, read before the
    // abort: the tail generated for nobody.
    tail_tokens: u64,
    /// Whether the client left during the read: the stream's verdict, taken before
    /// the finish so that no terminal router metric is recorded for a leaving client.
    client_left: bool,
    // Reusable SSE encoder shared across every chunk emitted for this stream.
    encoder: SseEncoder,
}

impl ChatDecode {
    fn new(pd_timing: Option<context::PdTiming>) -> Self {
        Self {
            start_time: Instant::now(),
            first_token_time: None,
            pd_timing,
            parsers: HashMap::new(),
            is_firsts: HashMap::new(),
            matched_stops: HashMap::new(),
            analysis_scanners: HashMap::new(),
            final_scanners: HashMap::new(),
            router_stopped: HashSet::new(),
            completion_tokens: CompletionTokenTracker::new(),
            reasoning_tokens: HashMap::new(),
            spec_accepted: HashMap::new(),
            spec_drafted: HashMap::new(),
            parsed_reasoning_tokens: 0,
            decode_completed_indices: HashSet::new(),
            tailed: HashSet::new(),
            engine_outlived: false,
            tail_tokens: 0,
            client_left: false,
            encoder: SseEncoder::new(),
        }
    }

    /// Choices closed by their decode `Complete` or by the router's stop.
    fn closed_choices(&self) -> u32 {
        (self.decode_completed_indices.len()
            + self
                .router_stopped
                .difference(&self.decode_completed_indices)
                .count()) as u32
    }

    /// Whether the engine has outlived the client's stream: every expected choice is closed
    /// while a router-stopped choice without its Complete has streamed on past its stop, so the
    /// engine is generating a tail nobody receives. A stopped choice whose next message has not
    /// come yet is waited for: an engine that ends the choice on that step sends its Complete
    /// next, and nothing is aborted; a chunk instead is the tail. Nothing to look at while no
    /// stop has fired.
    fn engine_outlived_now(&self, expected_choices: u32) -> bool {
        if self.router_stopped.is_empty() {
            return false;
        }
        let mut open_stops = 0u32;
        let mut tailed = false;
        for index in self
            .router_stopped
            .difference(&self.decode_completed_indices)
        {
            open_stops += 1;
            tailed |= self.tailed.contains(index);
        }
        tailed && self.decode_completed_indices.len() as u32 + open_stops >= expected_choices
    }

    /// The engine stream, released before the usage chunk is written. An engine the router's
    /// stop outlived is aborted now: dropping the stream un-marked is the abort path a client
    /// disconnect takes, so the slot goes back to the engine's batch within a step instead of
    /// running to `max_tokens` for nobody. An engine that ended on its own is marked completed,
    /// so that its drop sends no abort.
    fn release_engine(&self, mut decode_stream: ProtoStream, model: &str) {
        if !self.engine_outlived {
            decode_stream.mark_completed();
            return;
        }
        Metrics::record_local_stop_abort(LocalStopAbort {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_HARMONY,
            model_id: model,
            endpoint: metrics_labels::ENDPOINT_CHAT,
            tail_tokens: self.tail_tokens,
        });
        drop(decode_stream);
    }

    /// The stream's first engine output, for the TTFT; in PD mode the PD first-token pair is
    /// recorded against the dispatch instants.
    fn note_first_token(&mut self, model: &str) {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
            if let Some(timing) = &self.pd_timing {
                timing.record_first_decode_token(metrics_labels::BACKEND_HARMONY, model);
            }
        }
    }

    /// One chunk of one choice: its parser made on the first, its tokens counted and parsed,
    /// the router's stop strings scanned, and the delta emitted (the choice's role chunk first);
    /// a router-side stop emits the choice's final chunk, closes the choice with the chunk's
    /// counts and swallows the rest of its stream (the request is aborted once every choice has
    /// closed).
    #[expect(clippy::too_many_arguments)]
    async fn chunk(
        &mut self,
        chunk: &ProtoGenerateStreamChunk,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &SseSender,
        router_stop_strings: &[String],
        prompt_tokens: &mut HashMap<u32, u32>,
        cached_tokens: &mut HashMap<u32, u32>,
    ) -> Result<(), utils::StreamFailure> {
        let index = chunk.index();

        self.note_first_token(&original_request.model);

        // A choice's engine chunk after its router-side stop is the tail the engine generated
        // for nobody (an engine that ends the choice on that step sends its Complete instead):
        // counted, not read; the request is aborted once every choice has closed, until then
        // another choice may still be streaming.
        if self.router_stopped.contains(&index) {
            self.tail_tokens += chunk.token_ids().len() as u64;
            self.tailed.insert(index);
            return Ok(());
        }

        // Initialize parser for this index if needed
        if let Vacant(e) = self.parsers.entry(index) {
            e.insert(
                HarmonyParserAdapter::new().map_err(|e| format!("Failed to create parser: {e}"))?,
            );
            self.is_firsts.insert(index, true);
        }

        self.completion_tokens.record_chunk(chunk);

        // Convert logprobs if present and requested
        let chunk_logprobs = if original_request.logprobs {
            let encoding = try_harmony_encoding()?;
            chunk
                .output_logprobs()
                .map(|lp| convert_harmony_logprobs(encoding, &lp))
        } else {
            None
        };

        // Parse chunk via Harmony parser
        let parser = self
            .parsers
            .get_mut(&index)
            .ok_or("Parser not found for index")?;

        let delta_result = parser
            .parse_chunk(chunk.token_ids())
            .map_err(|e| format!("Parse error: {e}"))?;

        // Emit SSE event if there's a delta; a step whose tokens made none (a
        // channel header the parser consumed) still carries its logprobs, on
        // a chunk with an empty delta, as the engine's own server sends it.
        let bare_logprobs = utils::has_entries(chunk_logprobs.as_ref());
        let mut delta = match delta_result {
            Some(delta) => delta,
            None if bare_logprobs => HarmonyChannelDelta {
                analysis_delta: None,
                commentary_delta: None,
                final_delta: None,
            },
            None => return Ok(()),
        };
        let mut stop_matched: Option<String> = None;
        if !router_stop_strings.is_empty() {
            if let Some(text) = delta.analysis_delta.take() {
                let scanner = self
                    .analysis_scanners
                    .entry(index)
                    .or_insert_with(|| TextStopScanner::new(router_stop_strings.to_vec()));
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
                    let scanner = self
                        .final_scanners
                        .entry(index)
                        .or_insert_with(|| TextStopScanner::new(router_stop_strings.to_vec()));
                    let scan = scanner.push(&text);
                    delta.final_delta = (!scan.emit.is_empty()).then_some(scan.emit);
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
        let is_first = self.is_firsts.get(&index).copied().unwrap_or(false);
        if has_payload || is_first || bare_logprobs {
            HarmonyStreamingProcessor::emit_chunk_delta(
                &delta,
                index,
                is_first,
                dispatch,
                original_request,
                tx,
                &mut self.encoder,
                chunk_logprobs,
            )
            .await?;

            if is_first {
                self.is_firsts.insert(index, false);
            }
        }

        // A router-side stop fired: emit the final chunk now and swallow the rest of this
        // index's stream (the request is aborted once every choice has closed).
        if let Some(stop) = stop_matched {
            HarmonyStreamingProcessor::emit_final_chunk(
                index,
                "stop",
                Some(&serde_json::Value::String(stop)),
                dispatch,
                original_request,
                tx,
                &mut self.encoder,
            )
            .await?;
            self.router_stopped.insert(index);
            // The choice closes here, as its Complete would close it: the counts of the chunk
            // the stop matched in are its counts (the engine's own server counts a stopped
            // request through that step as well), the tracker settles what the stream had
            // reached, and the parser is finalized for the reasoning tokens it counted. What the
            // engine generates past this step is the tail the abort cuts, never the client's
            // usage.
            prompt_tokens
                .entry(index)
                .or_insert_with(|| chunk.prompt_tokens());
            cached_tokens
                .entry(index)
                .or_insert_with(|| chunk.cached_tokens());
            self.reasoning_tokens
                .insert(index, chunk.reasoning_tokens());
            self.completion_tokens.record_local_stop(index);
            if let Some(parser) = self.parsers.get_mut(&index) {
                let final_output = parser.finalize("stop".to_string());
                self.parsed_reasoning_tokens += final_output.reasoning_token_count;
            }
        }
        Ok(())
    }

    /// One choice's terminal frame: its metadata and counters stored, the choice opened with
    /// its role chunk if nothing opened it, its parser finalized, the scanners' held text
    /// released, and its final chunk emitted (a choice a router-side stop already closed gets
    /// none, and one the engine streamed past its stop adds only its speculative counts).
    async fn complete(
        &mut self,
        complete: &ProtoGenerateComplete,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &SseSender,
        prompt_tokens: &mut HashMap<u32, u32>,
        cached_tokens: &mut HashMap<u32, u32>,
    ) -> Result<(), utils::StreamFailure> {
        let index = complete.index();
        self.decode_completed_indices.insert(index);
        self.spec_accepted
            .insert(index, complete.spec_accepted_tokens());
        self.spec_drafted
            .insert(index, complete.spec_draft_tokens());

        // The Complete of a choice the engine generated past the router's stop (another choice
        // kept the stream open) is the tail's end: the choice closed at its stop with that
        // step's counts, and only the counts the stop could not know are taken from it.
        if self.tailed.contains(&index) {
            return Ok(());
        }

        // Store final metadata
        self.matched_stops
            .insert(index, complete.matched_stop_json());
        prompt_tokens
            .entry(index)
            .or_insert_with(|| complete.prompt_tokens());
        self.completion_tokens.record_complete(complete);
        cached_tokens
            .entry(index)
            .or_insert_with(|| complete.cached_tokens());
        self.reasoning_tokens
            .insert(index, complete.reasoning_tokens());

        // A router-side stop already closed this choice: its final chunk went out at the stop
        // and its parser was finalized there; the Complete's counters above are all it adds.
        if self.router_stopped.contains(&index) {
            return Ok(());
        }

        // A choice that reaches its Complete unopened (its only engine output is this Complete,
        // or every Chunk before it was header tokens without content) opens here with its role
        // chunk, as a choice with content opened on its first delta, and is the stream's first
        // output when nothing preceded it.
        if self.is_firsts.get(&index).copied().unwrap_or(true) {
            self.note_first_token(&original_request.model);
            HarmonyStreamingProcessor::emit_role_chunk(
                index,
                dispatch,
                original_request,
                tx,
                &mut self.encoder,
            )
            .await?;
            self.is_firsts.insert(index, false);
        }

        // Finalize parser and emit final chunk
        let Some(parser) = self.parsers.get_mut(&index) else {
            // The choice's only engine output is its Complete (a completion with no tokens: a
            // finish that carries none comes without a Chunk, so no parser was made for it); it
            // ends with its finish chunk as a choice with output does.
            HarmonyStreamingProcessor::emit_final_chunk(
                index,
                complete.finish_reason(),
                self.matched_stops.get(&index).and_then(|m| m.as_ref()),
                dispatch,
                original_request,
                tx,
                &mut self.encoder,
            )
            .await?;
            return Ok(());
        };
        let matched_stop = self.matched_stops.get(&index).and_then(|m| m.clone());

        let final_output = parser.finalize(complete.finish_reason().to_string());
        self.parsed_reasoning_tokens += final_output.reasoning_token_count;

        // Release scanner-held text that never became a match.
        let flushed = HarmonyChannelDelta {
            analysis_delta: self
                .analysis_scanners
                .get_mut(&index)
                .map(TextStopScanner::flush)
                .filter(|s| !s.is_empty()),
            commentary_delta: None,
            final_delta: self
                .final_scanners
                .get_mut(&index)
                .map(TextStopScanner::flush)
                .filter(|s| !s.is_empty()),
        };
        if flushed.analysis_delta.is_some() || flushed.final_delta.is_some() {
            HarmonyStreamingProcessor::emit_chunk_delta(
                &flushed,
                index,
                false,
                dispatch,
                original_request,
                tx,
                &mut self.encoder,
                None,
            )
            .await?;
        }

        HarmonyStreamingProcessor::emit_final_chunk(
            index,
            &final_output.finish_reason,
            matched_stop.as_ref(),
            dispatch,
            original_request,
            tx,
            &mut self.encoder,
        )
        .await?;
        Ok(())
    }

    /// The stream has ended: the counters once, for the usage chunk, the settlement and the
    /// metrics.
    async fn finish(
        mut self,
        prompt_tokens: &HashMap<u32, u32>,
        cached_tokens: &HashMap<u32, u32>,
        reservation: Option<Arc<SharedReservationHandle>>,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &SseSender,
    ) -> Result<(), utils::StreamFailure> {
        // Every `n>1` choice shares one prompt; each Complete reports that same full length, so
        // max (not sum) is the actual prompt cost. cached_tokens is a property of that same
        // shared prompt, not of the individual completion, so it takes the same treatment; each
        // choice is its own generation, so the rest sum.
        let counts = TokenCounts {
            prompt: prompt_tokens.values().copied().max().unwrap_or(0),
            completion: self.completion_tokens.total(),
            cached: cached_tokens.values().copied().max().unwrap_or(0),
            reasoning: self.reasoning_tokens.values().sum(),
            spec_accepted: self.spec_accepted.values().sum(),
            spec_drafted: self.spec_drafted.values().sum(),
        };

        if let Some(handle) = reservation {
            // A clean decode EOF with fewer closed choices (a decode `Complete`, or the router's
            // stop with the chunk's counts) than this request's `n>1` choices has only partial
            // usage -- settling with that would understate the real cost. Deliberately checked
            // against the decode stream's own closes, not `prompt_tokens`: in PD mode
            // `prompt_tokens` can already be non-empty from the prefill phase alone, which would
            // otherwise mask a decode phase that never actually finished.
            let expected_choices = original_request.choices();
            if self.closed_choices() < expected_choices {
                handle.close_reserved_only().await;
            } else {
                handle.settle_success(counts.settlement()).await;
            }
        }

        // The read ended with the client's leaving, not with the engine's end:
        // no usage chunk on the sender it closed, and no terminal router metric
        // for it, as the HTTP router has none for a disconnect (neither a
        // completed request nor a failure).
        if self.client_left {
            return Ok(());
        }

        // Emit final usage if requested
        if let Some(true) = original_request
            .stream_options
            .as_ref()
            .and_then(|so| so.include_usage)
        {
            HarmonyStreamingProcessor::emit_usage_chunk(
                counts,
                self.parsed_reasoning_tokens,
                dispatch,
                original_request,
                tx,
                &mut self.encoder,
            )
            .await?;
        }

        // In PD mode this task only starts once the decode stream exists -- for sequential PD
        // that is after prefill completes, with the first decode chunk already buffered -- so
        // `start_time` would report a near-zero TTFT. Anchor at the prefill dispatch instant
        // instead, as the chat stream does; generation duration must share the base or TPOT
        // underflows.
        let stream_base = self
            .pd_timing
            .as_ref()
            .map(|timing| timing.prefill_start)
            .unwrap_or(self.start_time);
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_HARMONY,
            model_id: &original_request.model,
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: self.first_token_time.map(|t| t.duration_since(stream_base)),
            generation_duration: stream_base.elapsed(),
            input_tokens: Some(counts.prompt as u64),
            cached_tokens: Some(u64::from(counts.cached)),
            output_tokens: counts.completion as u64,
        });

        Ok(())
    }
}

impl Default for HarmonyStreamingProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod client_leave_tests;
#[cfg(test)]
mod local_stop_tests;
#[cfg(test)]
mod logprobs_tests;
#[cfg(test)]
mod lone_complete_tests;
#[cfg(test)]
mod usage_tests;

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
