use std::{borrow::Cow, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    decisions::DecisionsRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    messages::{
        CountMessageTokensRequest, CountMessageTokensResponse, CreateMessageRequest, Message,
        MessageDelta, MessageDeltaUsage, MessageStreamEvent, StopReason, Tool, ToolChoice, Usage,
    },
    responses::ResponsesRequest,
    transcription::{AudioFile, TranscriptionRequest},
};
use tracing::{debug, warn};
use uuid::Uuid;

use super::{
    common::{
        responses::{
            background, handlers::cancel_response_impl, streaming::build_sse_response_from_stream,
            utils::validate_worker_availability, BackgroundResponses, BackgroundStream,
            ResponsesContext,
        },
        stages::{
            context_length_exceeded, helpers::resolve_request_id_stamp, smallest_context_length,
            RateLimitCell, RateLimitOutcome,
        },
    },
    context::{RequestType, SharedComponents},
    harmony::{serve_harmony_responses, serve_harmony_responses_stream, HarmonyDetector},
    mode::Mode,
    multimodal::{mm_settings, MultimodalComponents},
    pipeline::{Endpoint, PipelineDeps, RequestPipeline},
    regular::{
        responses, stages::decisions::supports_worker as supports_decisions_worker,
        streaming::StreamingProcessor,
    },
    utils::{message_utils, tokenizer_not_ready, ParserResolver},
};
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    middleware::TenantRequestMeta,
    observability::metrics::metrics_labels,
    routers::{error, RouterTrait},
    worker::{RoutingPool, WorkerRegistry, WorkerType},
};

/// `501 NOT_IMPLEMENTED`, returned by endpoints this router's mode doesn't
/// serve (matching the `RouterTrait` default).
fn not_implemented(message: &'static str) -> Response {
    (StatusCode::NOT_IMPLEMENTED, message).into_response()
}

/// gRPC router implementation for SGLang.
///
/// A single `Mode`-parameterized router serving Regular, PrefillDecode, and
/// EncodePrefillDecode. `mode` selects the disaggregation params baked into
/// every pipeline and drives the per-mode retry-metric labels, `Debug` output,
/// and `router_type`. Optional members are `None` when the mode doesn't serve
/// them and the corresponding endpoints 501: `embedding_pipeline` and
/// `classify_pipeline` are Regular-only; `harmony_pipeline`,
/// `responses_context`, and `harmony_responses_context` exist in every mode
/// but EPD.
#[derive(Clone)]
pub struct GrpcRouter {
    worker_registry: Arc<WorkerRegistry>,
    mode: Mode,
    pipeline: RequestPipeline,
    harmony_pipeline: Option<RequestPipeline>,
    embedding_pipeline: Option<RequestPipeline>,
    classify_pipeline: Option<RequestPipeline>,
    decisions_pipeline: Option<RequestPipeline>,
    transcription_pipeline: Option<RequestPipeline>,
    messages_pipeline: RequestPipeline,
    completion_pipeline: RequestPipeline,
    shared_components: Arc<SharedComponents>,
    responses_context: Option<ResponsesContext>,
    harmony_responses_context: Option<ResponsesContext>,
    retry_config: RetryConfig,
}

impl GrpcRouter {
    /// Regular and PD build the Harmony pipeline and responses contexts and
    /// require the MCP orchestrator; EPD leaves them `None` and 501s those
    /// endpoints. Embedding/classify pipelines are Regular-only.
    pub fn new(ctx: &Arc<AppContext>, mode: Mode) -> Result<Self, String> {
        // Get tokenizer registry (no longer requires pre-loaded tokenizer)
        let tokenizer_registry = ctx.tokenizer_registry.clone();

        let reasoning_parser_factory = ctx
            .reasoning_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires reasoning parser factory".to_string())?
            .clone();
        let tool_parser_factory = ctx
            .tool_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires tool parser factory".to_string())?
            .clone();

        let worker_registry = ctx.worker_registry.clone();
        let policy_registry = ctx.policy_registry.clone();

        // What can fail here is the operator's own setting, so the router
        // stops rather than coming up with media handling quietly switched
        // off and every media request failing later for no stated reason.
        let multimodal = Some(Arc::new(
            MultimodalComponents::new(
                ctx.multimodal_config_registry.clone(),
                ctx.router_config.mm_per_request_image_limit,
                ctx.router_config.multimodal_max_inflight_bytes,
                mm_settings(),
            )
            .map_err(|e| format!("multimodal components: {e:#}"))?,
        ));

        // Create shared components for pipeline
        let shared_components = Arc::new(SharedComponents {
            tokenizer_registry: tokenizer_registry.clone(),
            worker_registry: worker_registry.clone(),
            tool_parser_factory: tool_parser_factory.clone(),
            reasoning_parser_factory: reasoning_parser_factory.clone(),
            parser_resolver: ParserResolver::new(
                worker_registry.clone(),
                ctx.configured_tool_parser.clone(),
                ctx.configured_reasoning_parser.clone(),
            ),
            multimodal,
            tokenizer_autoload: !ctx.router_config.disable_tokenizer_autoload,
        });

        // The admission queue's timeout also bounds how long an accepted
        // request may sit ahead of worker selection.
        let queue_timeout = Duration::from_secs(ctx.router_config.queue_timeout_secs);
        // Deps for the parser-consuming endpoints (chat/messages/harmony).
        let configured_deps = PipelineDeps::new(
            worker_registry.clone(),
            policy_registry.clone(),
            ctx.prefill_admission.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
            ctx.rate_limit_manager.clone(),
            queue_timeout,
        );
        // Deps for the parser-free endpoints (completion/embeddings/classify).
        // Only completion's stage list actually reads `rate_limit_manager`;
        // embeddings/classify never insert `RateLimitReserveStage`.
        let pair_deps = PipelineDeps::pair(
            worker_registry.clone(),
            policy_registry.clone(),
            ctx.prefill_admission.clone(),
            ctx.rate_limit_manager.clone(),
            queue_timeout,
        );

        // Present in every mode: chat/generate, messages, completion.
        let pipeline = RequestPipeline::build(Endpoint::Chat, mode, &configured_deps)
            .ok_or_else(|| format!("gRPC router: no chat pipeline for mode {mode:?}"))?;
        let messages_pipeline = RequestPipeline::build(Endpoint::Messages, mode, &configured_deps)
            .ok_or_else(|| format!("gRPC router: no messages pipeline for mode {mode:?}"))?;
        let completion_pipeline = RequestPipeline::build(Endpoint::Completion, mode, &pair_deps)
            .ok_or_else(|| format!("gRPC router: no completion pipeline for mode {mode:?}"))?;

        // `None` when the (endpoint, mode) combo is unsupported; those endpoints 501.
        let harmony_pipeline = RequestPipeline::build(Endpoint::Harmony, mode, &configured_deps);
        let embedding_pipeline = RequestPipeline::build(Endpoint::Embeddings, mode, &pair_deps);
        let classify_pipeline = RequestPipeline::build(Endpoint::Classify, mode, &pair_deps);
        let decisions_pipeline = RequestPipeline::build(Endpoint::Decisions, mode, &pair_deps);
        let transcription_pipeline =
            RequestPipeline::build(Endpoint::Transcription, mode, &pair_deps);

        // Responses contexts are the sole consumer of the MCP orchestrator; EPD
        // builds neither (it doesn't serve /v1/responses).
        let (responses_context, harmony_responses_context) = if mode == Mode::EncodePrefillDecode {
            (None, None)
        } else {
            let mcp_orchestrator = ctx
                .mcp_orchestrator
                .get()
                .ok_or_else(|| "gRPC router requires MCP manager".to_string())?
                .clone();

            // Capture storage request context from middleware task-local (before any spawn)
            let storage_request_context = smg_data_connector::current_request_context();
            // One registry of background responses for both contexts: one cap,
            // one place a cancel looks a running id up.
            let background_responses = Arc::new(BackgroundResponses::new(
                ctx.router_config.max_background_responses,
            ));

            // Helper closure to create responses context with a given pipeline
            let create_responses_context = |pipeline: &RequestPipeline| {
                ResponsesContext::new(
                    Arc::new(pipeline.clone()),
                    shared_components.clone(),
                    ctx.response_storage.clone(),
                    ctx.conversation_storage.clone(),
                    ctx.conversation_item_storage.clone(),
                    mcp_orchestrator.clone(),
                    ctx.mcp_format_registry.clone(),
                    storage_request_context.clone(),
                    background_responses.clone(),
                )
            };

            let responses_context = create_responses_context(&pipeline);
            let harmony_responses_context = harmony_pipeline
                .as_ref()
                .map(&create_responses_context)
                .ok_or_else(|| {
                    format!("gRPC router: mode {mode:?} must build a harmony pipeline")
                })?;
            (Some(responses_context), Some(harmony_responses_context))
        };

        Ok(GrpcRouter {
            worker_registry,
            mode,
            pipeline,
            harmony_pipeline,
            embedding_pipeline,
            classify_pipeline,
            decisions_pipeline,
            transcription_pipeline,
            messages_pipeline,
            completion_pipeline,
            shared_components,
            responses_context,
            harmony_responses_context,
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    /// The per-model retry override registered by a worker, else the router
    /// default. Applied at every retrying endpoint
    /// (chat/generate/messages/completion) in every mode.
    ///
    /// Retry overrides are keyed by canonical model ID, so `model_id` must
    /// already be canonical (e.g. via [`Self::resolve_canonical_model_id`]) --
    /// every `route_*_impl` resolves that once before the retry loop starts,
    /// so this never needs to resolve an alias itself. Retry overrides are
    /// read before the pipeline canonicalizes in `RequestContext::new`, so an
    /// alias reaching this unresolved would miss the override and silently
    /// fall back to the router default.
    fn resolve_retry_config_for_canonical(&self, canonical_model_id: &str) -> RetryConfig {
        self.worker_registry
            .get_retry_config(canonical_model_id)
            .unwrap_or_else(|| self.retry_config.clone())
    }

    /// Resolve `model_id` to its canonical form once, before the pipeline
    /// runs. Every dispatch attempt for that logical request reuses this
    /// value -- re-resolving mid-request would let an alias repointed during
    /// the retry window dispatch a different model than whatever the tenant
    /// rate-limit reservation was actually made for, bypassing that model's
    /// own policy and settling its usage against the wrong budget.
    fn resolve_canonical_model_id(&self, model_id: &str) -> String {
        self.worker_registry
            .resolve_model_alias(model_id)
            .as_deref()
            .unwrap_or(model_id)
            .to_string()
    }

    /// Close a reservation that a non-2xx final response never got the
    /// chance to settle (prep failure repeated across every attempt,
    /// retries exhausted, a non-retryable dispatch failure). No-op if
    /// nothing was ever reserved. Safe to call unconditionally alongside a
    /// success path's own inline `settle_success`/streaming `settle_success`+
    /// `ReservationAttachment` -- `close_reserved_only` is CAS-guarded, only
    /// the first resolution of a handle wins.
    async fn close_reservation_if_unsettled(cell: &RateLimitCell, status: StatusCode) {
        if status.is_success() {
            return;
        }
        if let Some(RateLimitOutcome::Admitted(handle)) = cell.peek() {
            handle.close_reserved_only().await;
        }
    }

    /// Main route_chat implementation
    async fn route_chat_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        if let Err(response) = super::validate_text_only_output(&body) {
            return *response;
        }

        // EPD has no Harmony pipeline, so its chat requests all use the single
        // chat/generate pipeline.
        let is_harmony = self.harmony_pipeline.is_some()
            && HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &body.model);

        debug!(
            "Processing chat completion request for model: {}, using_harmony={}",
            model_id, is_harmony
        );

        let pipeline = match self.harmony_pipeline.as_ref() {
            Some(harmony_pipeline) if is_harmony => harmony_pipeline,
            _ => &self.pipeline,
        };

        // Canonicalize once, up front, so every dispatch attempt (and the
        // reservation) targets the same model -- see
        // `resolve_canonical_model_id`'s doc comment. The body's own `model`
        // field is rewritten to match: by this point `model_id_cloned` is
        // already canonical, so `RequestContext::new`'s own alias resolve
        // would no-op and otherwise leave the alias sitting in the body for
        // response metadata and parser selection to read.
        let model_id_cloned = self.resolve_canonical_model_id(model_id);
        let mut canonical_body = body;
        canonical_body.model = model_id_cloned.clone();
        let rate_limit_cell = Arc::new(RateLimitCell::new());
        let retry_config = self.resolve_retry_config_for_canonical(&model_id_cloned);

        // The request Arc moves into the pipeline: no handle survives out
        // here, so the parsed request frees at the build boundary.
        let response = pipeline
            .execute_chat(
                Arc::new(canonical_body),
                headers.cloned(),
                model_id_cloned,
                self.shared_components.clone(),
                Some(tenant_meta.clone()),
                Some(rate_limit_cell.clone()),
                Some(&retry_config),
                None,
            )
            .await;

        Self::close_reservation_if_unsettled(&rate_limit_cell, response.status()).await;
        response
    }

    /// Main route_generate implementation
    async fn route_generate_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: GenerateRequest,
        model_id: &str,
    ) -> Response {
        debug!("Processing generate request for model: {}", model_id);

        // Canonicalize once, up front -- see `resolve_canonical_model_id`'s
        // doc comment. Rewrite the body's `model` field to match; see
        // `route_chat_impl`.
        let model_id_cloned = self.resolve_canonical_model_id(model_id);
        let mut canonical_body = body;
        canonical_body.model = model_id_cloned.clone();
        let rate_limit_cell = Arc::new(RateLimitCell::new());
        let retry_config = self.resolve_retry_config_for_canonical(&model_id_cloned);

        let response = self
            .pipeline
            .execute_generate(
                Arc::new(canonical_body),
                headers.cloned(),
                model_id_cloned,
                self.shared_components.clone(),
                Some(tenant_meta.clone()),
                Some(rate_limit_cell.clone()),
                Some(&retry_config),
            )
            .await;

        Self::close_reservation_if_unsettled(&rate_limit_cell, response.status()).await;
        response
    }

    /// Main route_responses implementation
    ///
    /// Routes to either Harmony or regular responses implementation based on model detection
    async fn route_responses_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        let (Some(responses_context), Some(harmony_responses_context)) =
            (&self.responses_context, &self.harmony_responses_context)
        else {
            return not_implemented("Responses endpoint not implemented");
        };
        let Some(harmony_pipeline) = self.harmony_pipeline.as_ref() else {
            return not_implemented("Responses endpoint not implemented");
        };

        // 0. Fast worker validation (fail-fast before expensive operations)
        if let Some(error_response) = validate_worker_availability(&self.worker_registry, model_id)
        {
            return error_response;
        }

        let (body, canonical_model_id) =
            canonicalize_responses_request(&self.worker_registry, &body, model_id);
        let model_id = canonical_model_id.as_ref();

        // Choose implementation based on Harmony model detection (checks worker metadata)
        let is_harmony =
            HarmonyDetector::is_harmony_model_in_registry(&self.worker_registry, &body.model);

        // `background: true` answers the queued object at once and runs the
        // work behind the response id; with the mode switched off such a
        // request runs in the foreground, and its object says so.
        let mut body = body.into_owned();
        if body.background == Some(true) {
            if responses_context.background.enabled() {
                let request_context = smg_data_connector::current_request_context();
                let ctx = if is_harmony {
                    self.harmony_responses_context(
                        harmony_pipeline,
                        harmony_responses_context,
                        request_context,
                    )
                } else {
                    ResponsesContext {
                        request_context,
                        ..responses_context.clone()
                    }
                };
                return self
                    .route_background_responses(
                        &ctx,
                        is_harmony,
                        headers,
                        tenant_meta,
                        body,
                        model_id,
                    )
                    .await;
            }
            body.background = None;
        }

        if is_harmony {
            debug!(
                "Processing Harmony responses request for model: {}, streaming: {}",
                model_id,
                body.stream.unwrap_or(false)
            );
            let harmony_ctx = self.harmony_responses_context(
                harmony_pipeline,
                harmony_responses_context,
                smg_data_connector::current_request_context(),
            );

            if body.stream.unwrap_or(false) {
                serve_harmony_responses_stream(&harmony_ctx, body, tenant_meta.clone(), None).await
            } else {
                match serve_harmony_responses(&harmony_ctx, body, tenant_meta.clone(), None).await {
                    Ok(response) => Json(response).into_response(),
                    Err(error_response) => error_response,
                }
            }
        } else {
            responses::route_responses(
                responses_context,
                Arc::new(body),
                headers.cloned(),
                tenant_meta.clone(),
                model_id.to_string(),
            )
            .await
        }
    }

    /// The per-request context of the Harmony Responses path: the Harmony
    /// pipeline with the shared storage and MCP handles and this request's
    /// storage context.
    fn harmony_responses_context(
        &self,
        harmony_pipeline: &RequestPipeline,
        harmony_responses_context: &ResponsesContext,
        request_context: Option<smg_data_connector::RequestContext>,
    ) -> ResponsesContext {
        ResponsesContext::new(
            Arc::new(harmony_pipeline.clone()),
            self.shared_components.clone(),
            harmony_responses_context.response_storage.clone(),
            harmony_responses_context.conversation_storage.clone(),
            harmony_responses_context.conversation_item_storage.clone(),
            harmony_responses_context.mcp_orchestrator.clone(),
            harmony_responses_context.mcp_format_registry.clone(),
            request_context,
            harmony_responses_context.background.clone(),
        )
    }

    /// `background: true`: the `queued` Response object is stored under a new
    /// id and answered (or streamed) at once; the work runs in a task of its
    /// own behind that id, on the Harmony or the regular path. One beyond
    /// the configured number in flight answers 429.
    async fn route_background_responses(
        &self,
        ctx: &ResponsesContext,
        is_harmony: bool,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        request: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        let id = format!("resp_{}", Uuid::now_v7());
        let Some(slot) = ctx.background.admit(&id) else {
            warn!(
                in_flight = ctx.background.in_flight(),
                max_in_flight = ctx.background.max_in_flight(),
                "Background response refused: the limit is reached"
            );
            return background::limit_reached(ctx.background.max_in_flight());
        };
        let request = Arc::new(request);
        let queued = background::queued_response(&id, &request);
        if let Err(e) = background::store_record(
            &ctx.response_storage,
            &queued,
            &request,
            ctx.request_context.clone(),
        )
        .await
        {
            return error::storage_failure(
                "store_response_failed",
                "Failed to store the queued response",
                &e,
            );
        }
        debug!(response_id = %id, harmony = is_harmony, "Background response queued");

        let tenant_meta = tenant_meta.clone();
        let headers = headers.cloned();
        let model_id = model_id.to_string();
        if request.stream.unwrap_or(false) {
            let stream = BackgroundStream::new(slot);
            let answer = if is_harmony {
                serve_harmony_responses_stream(
                    ctx,
                    Arc::unwrap_or_clone(request),
                    tenant_meta,
                    Some(stream),
                )
                .await
            } else {
                responses::route_responses_background_stream(
                    ctx,
                    request,
                    headers,
                    tenant_meta,
                    model_id,
                    stream,
                )
                .await
            };
            // A plain error before the stream opened closes the record as
            // failed; the stream itself runs behind the id.
            return background::close_before_stream(
                &ctx.response_storage,
                ctx.request_context.clone(),
                &id,
                answer,
            )
            .await;
        }

        if is_harmony {
            background::spawn(
                ctx.clone(),
                slot,
                request,
                Box::new(move |ctx, request| {
                    Box::pin(async move {
                        serve_harmony_responses(
                            ctx,
                            Arc::unwrap_or_clone(request),
                            tenant_meta,
                            Some(id),
                        )
                        .await
                    })
                }),
            );
        } else {
            background::spawn(
                ctx.clone(),
                slot,
                request,
                Box::new(move |ctx, request| {
                    Box::pin(async move {
                        responses::route_responses_background(
                            ctx,
                            request,
                            headers,
                            tenant_meta,
                            model_id,
                            id,
                        )
                        .await
                    })
                }),
            );
        }
        (StatusCode::OK, Json(queued)).into_response()
    }

    /// Main route_embeddings implementation
    async fn route_embeddings_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: EmbeddingRequest,
        model_id: &str,
    ) -> Response {
        let Some(embedding_pipeline) = self.embedding_pipeline.as_ref() else {
            return not_implemented("Embeddings not implemented");
        };
        debug!("Processing embedding request for model: {}", model_id);

        embedding_pipeline
            .execute_embeddings(
                Arc::new(body),
                headers.cloned(),
                model_id.to_string(),
                self.shared_components.clone(),
                Some(tenant_meta.clone()),
            )
            .await
    }

    /// Main route_messages implementation
    async fn route_messages_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        debug!("Processing messages request for model: {}", model_id);

        // Canonicalize once, up front -- see `resolve_canonical_model_id`'s
        // doc comment. Rewrite the body's `model` field to match; see
        // `route_chat_impl`.
        let model_id_cloned = self.resolve_canonical_model_id(model_id);
        let mut canonical_body = body;
        canonical_body.model = model_id_cloned.clone();
        let canonical_body = Arc::new(canonical_body);

        // `max_tokens: 0` is answered as the public API answers it: an empty
        // message that reports the prompt's token count; nothing is generated.
        // No worker takes part, so the answer carries no rate-limit
        // reservation, as count_tokens reserves none; it is counted as a
        // Messages request, refusals included, where the pipeline counts
        // its own, so the traffic stays visible.
        if canonical_body.max_tokens == 0 {
            const ENDPOINT: &str = metrics_labels::ENDPOINT_MESSAGES;
            self.messages_pipeline.record_request(
                ENDPOINT,
                &model_id_cloned,
                canonical_body.stream.unwrap_or(false),
            );
            let response = self
                .empty_message_response(tenant_meta, canonical_body, &model_id_cloned)
                .await;
            if !response.status().is_success() {
                self.messages_pipeline
                    .record_error(Some(ENDPOINT), &model_id_cloned, &response);
            }
            return response;
        }
        let rate_limit_cell = Arc::new(RateLimitCell::new());
        let retry_config = self.resolve_retry_config_for_canonical(&model_id_cloned);

        let response = self
            .messages_pipeline
            .execute_messages(
                canonical_body,
                headers.cloned(),
                model_id_cloned,
                self.shared_components.clone(),
                Some(tenant_meta.clone()),
                Some(rate_limit_cell.clone()),
                Some(&retry_config),
            )
            .await;

        Self::close_reservation_if_unsettled(&rate_limit_cell, response.status()).await;
        response
    }

    /// `/v1/messages/count_tokens`, answered from the gateway's own
    /// tokenizer: the request is rendered through the model's chat template
    /// exactly as `/v1/messages` renders it and the prompt's token count is
    /// returned. No worker takes part, so every worker type is covered.
    async fn route_messages_count_tokens_impl(
        &self,
        body: CountMessageTokensRequest,
        model_id: &str,
    ) -> Response {
        let model_id = self.resolve_canonical_model_id(model_id);
        match self
            .count_prompt_tokens(
                &model_id,
                message_utils::MessagesPrompt::from(&body),
                body.tools.as_deref(),
                body.tool_choice.as_ref(),
            )
            .await
        {
            Ok(input_tokens) => (
                StatusCode::OK,
                Json(CountMessageTokensResponse { input_tokens }),
            )
                .into_response(),
            Err(refused) => refused,
        }
    }

    /// The prompt's token count, as `/v1/messages/count_tokens` reports it:
    /// the registry's tokenizer for the model, the tools the template renders
    /// and the request's media resolved as the preparation stage resolves
    /// them; no worker takes part. The refusals are count_tokens': a model
    /// nobody serves is the client's mistake (404), whatever a removed worker
    /// left in the tokenizer registry; a served model whose tokenizer the
    /// gateway is still loading is not ready yet (503); a served model with
    /// no tokenizer when the gateway does not load tokenizers itself is a
    /// gateway fault (500), as is a prompt that does not encode; a prompt
    /// that does not render is a 400.
    async fn count_prompt_tokens(
        &self,
        model_id: &str,
        prompt: message_utils::MessagesPrompt<'_>,
        tools: Option<&[Tool]>,
        tool_choice: Option<&ToolChoice>,
    ) -> Result<u32, Response> {
        if !self.worker_registry.contains_model(model_id) {
            return Err(error::model_not_found(model_id));
        }
        let Some(tokenizer) = self.shared_components.tokenizer_registry.get(model_id) else {
            // The window between a worker turning healthy and its tokenizer
            // registering is a transient not-ready state while the gateway
            // loads tokenizers itself; without autoload nothing will register
            // one, and the missing tokenizer is the gateway's fault.
            if self.shared_components.tokenizer_autoload {
                return Err(tokenizer_not_ready(
                    "GrpcRouter::count_prompt_tokens",
                    model_id,
                ));
            }
            return Err(error::internal_error(
                "tokenizer_not_found",
                format!("Tokenizer not found for model: {model_id}"),
            ));
        };
        let chat_tools = message_utils::template_tools(tools, tool_choice);
        let media = message_utils::resolve_messages_media(
            &self.shared_components,
            model_id,
            &*tokenizer,
            prompt.messages,
        )
        .await?;
        message_utils::count_input_tokens(
            prompt,
            tokenizer,
            &chat_tools,
            media.placeholders(),
            media.media_order,
        )
        .await
        .map_err(|e| match e {
            message_utils::CountError::Render(e) => {
                debug!(model = %model_id, error = %e, "the request does not render");
                error::bad_request("process_messages_failed", e)
            }
            message_utils::CountError::Encode(e) => {
                tracing::error!(model = %model_id, error = %e, "tokenization failed");
                error::internal_error("tokenization_failed", format!("Tokenization failed: {e}"))
            }
        })
    }

    /// The answer to `max_tokens: 0`, as the public API answers it: a message
    /// with no content, `stop_reason: max_tokens` and the prompt's token count
    /// in `usage`, counted as `/v1/messages/count_tokens` counts it; no worker
    /// takes part, and the message carries the id the pipeline would have
    /// given it. A streaming request gets that message as `message_start`,
    /// then `message_delta` with the stop reason and `message_stop`. A request
    /// with media is counted at the placeholders the template writes for it,
    /// as count_tokens counts it: the media is neither fetched nor decoded, so
    /// the count is the template's, not the expanded prompt's; an image or a
    /// video to a pool whose every worker is language model only is refused
    /// with `multimodal_not_supported` here as on every Messages path. A
    /// prompt longer than the smallest context window the model's workers
    /// advertise is refused as the pipeline refuses it after worker selection
    /// (`context_length_exceeded`), so a 200 here means the prompt fits
    /// whichever worker a real call lands on; a deployment that advertises no
    /// window leaves the check to the engine, as the pipeline does.
    async fn empty_message_response(
        &self,
        tenant_meta: &TenantRequestMeta,
        body: Arc<CreateMessageRequest>,
        model_id: &str,
    ) -> Response {
        let input_tokens = match self
            .count_prompt_tokens(
                model_id,
                message_utils::MessagesPrompt::from(&*body),
                body.tools.as_deref(),
                body.tool_choice.as_ref(),
            )
            .await
        {
            Ok(input_tokens) => input_tokens,
            Err(refused) => return refused,
        };
        if let Some(limit) =
            smallest_context_length(&self.worker_registry.get_by_model(model_id), model_id)
        {
            if input_tokens > limit {
                return context_length_exceeded(input_tokens as usize, limit, model_id);
            }
        }
        let stream = body.stream.unwrap_or(false);
        let (id, _) = resolve_request_id_stamp(
            &RequestType::Messages(body),
            Some(tenant_meta),
            "msg_",
            self.mode != Mode::Regular,
        );
        let message = Message {
            id,
            message_type: "message".to_string(),
            role: "assistant".to_string(),
            content: Vec::new(),
            model: model_id.to_string(),
            stop_reason: Some(StopReason::MaxTokens),
            stop_sequence: None,
            stop_details: None,
            usage: Usage::from_counts(input_tokens, 0),
            container: None,
        };
        if !stream {
            return (StatusCode::OK, Json(message)).into_response();
        }

        let events = [
            MessageStreamEvent::MessageStart {
                message: Message {
                    stop_reason: None,
                    ..message
                },
            },
            MessageStreamEvent::MessageDelta {
                delta: MessageDelta {
                    stop_reason: Some(StopReason::MaxTokens),
                    ..MessageDelta::default()
                },
                usage: MessageDeltaUsage {
                    output_tokens: 0,
                    input_tokens: Some(input_tokens),
                    cache_creation_input_tokens: Some(0),
                    cache_read_input_tokens: Some(0),
                    server_tool_use: None,
                    output_tokens_details: None,
                },
            },
            MessageStreamEvent::MessageStop,
        ];
        let mut frames: Vec<Result<bytes::Bytes, std::io::Error>> =
            Vec::with_capacity(events.len());
        let mut buffer = Vec::new();
        for event in &events {
            if let Err(e) = StreamingProcessor::format_messages_sse_into(&mut buffer, event) {
                tracing::error!(model = %model_id, error = %e, "max_tokens 0: a stream event does not serialize");
                return error::internal_error("messages_event_serialization_failed", e);
            }
            frames.push(Ok(bytes::Bytes::from(std::mem::take(&mut buffer))));
        }
        build_sse_response_from_stream(futures::stream::iter(frames))
    }

    /// Main route_completion implementation
    async fn route_completion_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CompletionRequest,
        model_id: &str,
    ) -> Response {
        debug!("Processing completion request for model: {}", model_id);

        // Canonicalize once, up front -- see `resolve_canonical_model_id`'s
        // doc comment. Rewrite the body's `model` field to match; see
        // `route_chat_impl`.
        let model_id_cloned = self.resolve_canonical_model_id(model_id);
        let mut canonical_body = body;
        canonical_body.model = model_id_cloned.clone();
        let rate_limit_cell = Arc::new(RateLimitCell::new());
        let retry_config = self.resolve_retry_config_for_canonical(&model_id_cloned);

        let response = self
            .completion_pipeline
            .execute_completion(
                Arc::new(canonical_body),
                headers.cloned(),
                model_id_cloned,
                self.shared_components.clone(),
                Some(tenant_meta.clone()),
                Some(rate_limit_cell.clone()),
                Some(&retry_config),
            )
            .await;

        Self::close_reservation_if_unsettled(&rate_limit_cell, response.status()).await;
        response
    }

    /// Score Decisions through the regular SGLang pipeline.
    async fn route_decisions_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        mut body: DecisionsRequest,
        model_id: &str,
    ) -> Response {
        let Some(pipeline) = self.decisions_pipeline.as_ref() else {
            return error::not_implemented(
                "decisions_not_supported",
                "Decisions requires a regular SGLang gRPC router",
            );
        };
        let canonical = self.resolve_canonical_model_id(model_id);
        if canonical == openai_protocol::UNKNOWN_MODEL_ID {
            return error::model_not_found(model_id);
        }
        let Some(snapshot) = self.worker_registry.model_routing_snapshot(&canonical) else {
            return error::model_not_found(model_id);
        };
        if !snapshot
            .pool(RoutingPool::GrpcPipelineRegular)
            .iter()
            .any(|worker| supports_decisions_worker(worker.as_ref()))
        {
            return error::not_implemented(
                "decisions_not_supported",
                "Decisions scoring requires a regular SGLang gRPC worker",
            );
        }
        body.model = canonical.clone();
        let rate_limit_cell = Arc::new(RateLimitCell::new());
        let retry_config = self.resolve_retry_config_for_canonical(&canonical);
        let response = pipeline
            .execute_decisions(
                Arc::new(body),
                headers.cloned(),
                canonical,
                self.shared_components.clone(),
                Some(tenant_meta.clone()),
                Some(rate_limit_cell.clone()),
                Some(&retry_config),
            )
            .await;
        Self::close_reservation_if_unsettled(&rate_limit_cell, response.status()).await;
        response
    }

    /// Main route_classify implementation
    async fn route_classify_impl(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ClassifyRequest,
        model_id: &str,
    ) -> Response {
        let Some(classify_pipeline) = self.classify_pipeline.as_ref() else {
            return not_implemented("Classify not implemented");
        };
        debug!("Processing classify request for model: {}", model_id);

        classify_pipeline
            .execute_classify(
                Arc::new(body),
                headers.cloned(),
                model_id.to_string(),
                self.shared_components.clone(),
                Some(tenant_meta.clone()),
            )
            .await
    }
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.mode {
            Mode::Regular => {
                let stats = self.worker_registry.stats();
                f.debug_struct("GrpcRouter")
                    .field("workers_count", &stats.total_workers)
                    .finish()
            }
            Mode::PrefillDecode | Mode::EncodePrefillDecode => {
                // Count every worker this router can serve (gRPC and ZMQ both
                // ride the gRPC pipeline), not just ConnectionMode::Grpc.
                let count_pipeline_workers = |worker_type| {
                    self.worker_registry
                        .get_workers_filtered(None, Some(worker_type), None, None, false)
                        .iter()
                        .filter(|w| w.connection_mode().uses_grpc_pipeline())
                        .count()
                };
                let prefill_workers = count_pipeline_workers(WorkerType::Prefill);
                let decode_workers = count_pipeline_workers(WorkerType::Decode);
                f.debug_struct("GrpcRouter")
                    .field("prefill_workers_count", &prefill_workers)
                    .field("decode_workers_count", &decode_workers)
                    .finish()
            }
        }
    }
}

#[async_trait]
impl RouterTrait for GrpcRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: GenerateRequest,
        model_id: &str,
    ) -> Response {
        self.route_generate_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ChatCompletionRequest,
        model_id: &str,
    ) -> Response {
        self.route_chat_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ResponsesRequest,
        model_id: &str,
    ) -> Response {
        self.route_responses_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let Some(responses_context) = self.responses_context.as_ref() else {
            return not_implemented("Cancel response not implemented");
        };
        cancel_response_impl(responses_context, response_id).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: EmbeddingRequest,
        model_id: &str,
    ) -> Response {
        self.route_embeddings_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: ClassifyRequest,
        model_id: &str,
    ) -> Response {
        self.route_classify_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_decisions(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: DecisionsRequest,
        model_id: &str,
    ) -> Response {
        self.route_decisions_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_audio_transcriptions(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &TranscriptionRequest,
        audio: AudioFile,
        model_id: &str,
    ) -> Response {
        // Transcription is a first-class pipeline endpoint (Regular-only, so
        // `None` in PD/EPD). Detection, the chat-shaped request synthesis, and
        // output parsing all live inside the pipeline's transcription stages.
        let Some(pipeline) = self.transcription_pipeline.as_ref() else {
            return not_implemented("Audio transcriptions not implemented");
        };
        pipeline
            .execute_transcription(
                Arc::new(body.clone()),
                Arc::new(audio),
                headers.cloned(),
                model_id.to_string(),
                Arc::clone(&self.shared_components),
                Some(tenant_meta.clone()),
            )
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CompletionRequest,
        model_id: &str,
    ) -> Response {
        self.route_completion_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        self.route_messages_impl(headers, tenant_meta, body, model_id)
            .await
    }

    async fn route_messages_count_tokens(
        &self,
        _headers: Option<&HeaderMap>,
        _tenant_meta: &TenantRequestMeta,
        body: CountMessageTokensRequest,
        model_id: &str,
    ) -> Response {
        self.route_messages_count_tokens_impl(body, model_id).await
    }

    fn router_type(&self) -> &'static str {
        self.mode.router_type()
    }
}

/// Resolve a Responses request's model alias, for both the routing decisions
/// and the request the Responses layer will read.
///
/// The Responses layer builds its SSE events and its tool call responses from
/// the request rather than from the pipeline context, so the alias has to be
/// gone before dispatch. Otherwise a plain answer reports the canonical ID
/// while a tool call answer reports the alias.
///
/// Returns both halves together so a caller cannot take the canonical model ID
/// and forget the body, or the reverse. The common path borrows both untouched
/// and allocates nothing.
fn canonicalize_responses_request<'a>(
    worker_registry: &WorkerRegistry,
    body: &'a ResponsesRequest,
    model_id: &'a str,
) -> (Cow<'a, ResponsesRequest>, Cow<'a, str>) {
    let Some(canonical_model) = worker_registry.resolve_model_alias(model_id) else {
        return (Cow::Borrowed(body), Cow::Borrowed(model_id));
    };
    let mut canonical_body = body.clone();
    canonical_body.model.clear();
    canonical_body.model.push_str(&canonical_model);
    (
        Cow::Owned(canonical_body),
        Cow::Owned(canonical_model.to_string()),
    )
}

#[cfg(test)]
mod pd_tests {
    use std::sync::{Arc, OnceLock};

    use llm_tokenizer::registry::TokenizerRegistry;
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
    use serde_json::json;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use smg_mcp::{McpConfig, McpOrchestrator};
    use tool_parser::ParserFactory as ToolParserFactory;

    use super::*;
    use crate::{
        config::{PolicyConfig, RouterConfig, RoutingMode},
        policies::PolicyRegistry,
        tenant::TenantKey,
        worker::{BasicWorkerBuilder, ConnectionMode, WorkerRegistry},
    };

    fn regular_routing_mode() -> RoutingMode {
        RoutingMode::Regular {
            worker_urls: vec![],
        }
    }

    fn pd_routing_mode() -> RoutingMode {
        RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        }
    }

    fn epd_routing_mode() -> RoutingMode {
        RoutingMode::EncodePrefillDecode {
            encode_urls: vec![],
            prefill_urls: vec![],
            decode_urls: vec![],
            encode_policy: None,
            prefill_policy: None,
            decode_policy: None,
        }
    }

    /// Minimal `AppContext` for constructing a disaggregated gRPC router. PD
    /// serves /v1/responses, so the MCP orchestrator must be initialized.
    async fn grpc_ctx(mode: RoutingMode) -> Arc<AppContext> {
        let config = RouterConfig::builder()
            .mode(mode)
            .grpc_connection()
            .policy(PolicyConfig::Random)
            .host("127.0.0.1")
            .port(3001)
            .max_payload_size(1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(10)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let mcp_orchestrator = Arc::new(OnceLock::new());
        mcp_orchestrator
            .set(Arc::new(
                McpOrchestrator::new(McpConfig::default())
                    .await
                    .expect("mcp orchestrator"),
            ))
            .ok();

        Arc::new(
            AppContext::builder()
                .router_config(config.clone())
                .client(crate::http_client::client())
                .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
                .reasoning_parser_factory(Some(ReasoningParserFactory::new()))
                .tool_parser_factory(Some(ToolParserFactory::new()))
                .worker_registry(Arc::new(WorkerRegistry::new()))
                .policy_registry(Arc::new(PolicyRegistry::new(config.policy.clone())))
                .response_storage(Arc::new(MemoryResponseStorage::new()))
                .conversation_storage(Arc::new(MemoryConversationStorage::new()))
                .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
                .worker_job_queue(Arc::new(OnceLock::new()))
                .workflow_engines(Arc::new(OnceLock::new()))
                .mcp_orchestrator(mcp_orchestrator)
                .build()
                .expect("app context"),
        )
    }

    fn responses_request(model: &str) -> ResponsesRequest {
        serde_json::from_value(json!({"model": model, "input": "hi"})).expect("responses request")
    }

    fn count_request(model: &str) -> CountMessageTokensRequest {
        serde_json::from_value(json!({
            "model": model, "messages": [{"role": "user", "content": "hi"}]
        }))
        .expect("count request")
    }

    /// count_tokens is the router's own answer, from the registry's
    /// tokenizer: the rendered prompt's token count for a served model (here
    /// a renderer that encodes itself hands back three ids), the 404
    /// model_not_found for a model nobody serves. No worker takes part.
    #[tokio::test]
    async fn count_tokens_is_answered_from_the_tokenizer() {
        let ctx = grpc_ctx(regular_routing_mode()).await;
        let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(
            llm_tokenizer::mock::MockTokenizer::new().with_deferred_chat_ids(vec![11, 12, 13]),
        );
        ctx.tokenizer_registry
            .load("m", "m", "mock", || async move { Ok(tokenizer) })
            .await
            .expect("register the tokenizer");
        let worker = BasicWorkerBuilder::new("grpc://worker:30000")
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .model(ModelCard::new("m"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        ctx.worker_registry
            .register(Arc::new(worker))
            .expect("register worker");
        let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
        let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

        let response = router
            .route_messages_count_tokens(None, &tenant_meta, count_request("m"), "m")
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&bytes).expect("json"),
            json!({"input_tokens": 3})
        );

        let response = router
            .route_messages_count_tokens(None, &tenant_meta, count_request("nope"), "nope")
            .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "model_not_found"
        );
    }

    /// A tokenizer a removed worker left in the registry does not make its
    /// model served: the count is refused as an unknown model, like every
    /// other route refuses it.
    #[tokio::test]
    async fn count_tokens_refuses_a_model_no_worker_serves_whatever_the_tokenizer_registry_holds() {
        let ctx = grpc_ctx(regular_routing_mode()).await;
        let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(
            llm_tokenizer::mock::MockTokenizer::new().with_deferred_chat_ids(vec![11, 12, 13]),
        );
        ctx.tokenizer_registry
            .load("gone", "gone", "mock", || async move { Ok(tokenizer) })
            .await
            .expect("register the tokenizer");
        let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
        let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

        let response = router
            .route_messages_count_tokens(None, &tenant_meta, count_request("gone"), "gone")
            .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "model_not_found"
        );
    }

    /// A served model whose tokenizer the gateway has not registered yet
    /// (the window after its worker turned healthy) is not ready: a
    /// retryable 503 with a `Retry-After`, not a 500.
    #[tokio::test]
    async fn count_tokens_answers_not_ready_while_a_served_models_tokenizer_loads() {
        let ctx = grpc_ctx(regular_routing_mode()).await;
        let worker = BasicWorkerBuilder::new("grpc://worker:30000")
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .model(ModelCard::new("m"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        ctx.worker_registry
            .register(Arc::new(worker))
            .expect("register worker");
        let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
        let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

        let response = router
            .route_messages_count_tokens(None, &tenant_meta, count_request("m"), "m")
            .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "tokenizer_not_ready"
        );
        assert_eq!(
            response
                .headers()
                .get(http::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("1")
        );
    }

    fn message_request(body: serde_json::Value) -> CreateMessageRequest {
        serde_json::from_value(body).expect("message request")
    }

    fn sse_event(frame: &str, name: &str) -> serde_json::Value {
        let prefix = format!("event: {name}\ndata: ");
        let data = frame
            .strip_prefix(prefix.as_str())
            .unwrap_or_else(|| panic!("expected a {name} frame, got {frame:?}"));
        serde_json::from_str(data).expect("event json")
    }

    /// `max_tokens: 0` is the router's own answer, as the public API's: a
    /// message with no content, `stop_reason: max_tokens` and the prompt's
    /// token count in `usage` (the renderer hands back three ids), under the
    /// id the pipeline would have given it (the client's `rid` when it sends
    /// one); streamed, that message as `message_start`, then `message_delta`
    /// with the stop reason and `message_stop`. No worker takes part; a
    /// model nobody serves is a 404, and a prompt longer than the smallest
    /// context window the model's workers advertise is the pipeline's 400.
    #[tokio::test]
    async fn max_tokens_zero_is_answered_with_an_empty_message() {
        let ctx = grpc_ctx(regular_routing_mode()).await;
        let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(
            llm_tokenizer::mock::MockTokenizer::new().with_deferred_chat_ids(vec![11, 12, 13]),
        );
        ctx.tokenizer_registry
            .load("m", "m", "mock", || async move { Ok(tokenizer) })
            .await
            .expect("register the tokenizer");
        let worker = BasicWorkerBuilder::new("grpc://worker:30000")
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .model(ModelCard::new("m"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        ctx.worker_registry
            .register(Arc::new(worker))
            .expect("register worker");
        let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
        let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));
        let body = json!({
            "model": "m", "max_tokens": 0,
            "messages": [{"role": "user", "content": "hi"}]
        });

        let response = router
            .route_messages(None, &tenant_meta, message_request(body.clone()), "m")
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let message: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(message["type"], "message");
        assert_eq!(message["role"], "assistant");
        assert_eq!(message["content"], json!([]));
        assert_eq!(message["stop_reason"], "max_tokens");
        assert_eq!(message["usage"]["input_tokens"], 3);
        assert_eq!(message["usage"]["output_tokens"], 0);
        assert_eq!(message["usage"]["cache_read_input_tokens"], 0);
        assert!(message["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("msg_")));

        let mut with_rid = body.clone();
        with_rid["rid"] = json!("req-7");
        let response = router
            .route_messages(None, &tenant_meta, message_request(with_rid), "m")
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let message: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(message["id"], "req-7");

        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let response = router
            .route_messages(None, &tenant_meta, message_request(streamed), "m")
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream; charset=utf-8")
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let text = String::from_utf8(bytes.to_vec()).expect("utf-8");
        let frames: Vec<&str> = text.split("\n\n").filter(|f| !f.is_empty()).collect();
        assert_eq!(frames.len(), 3, "{text}");
        let start = sse_event(frames[0], "message_start");
        assert_eq!(start["message"]["content"], json!([]));
        assert_eq!(start["message"]["stop_reason"], serde_json::Value::Null);
        assert_eq!(start["message"]["usage"]["input_tokens"], 3);
        let delta = sse_event(frames[1], "message_delta");
        assert_eq!(delta["delta"]["stop_reason"], "max_tokens");
        assert_eq!(delta["usage"]["output_tokens"], 0);
        assert_eq!(delta["usage"]["input_tokens"], 3);
        assert_eq!(
            sse_event(frames[2], "message_stop"),
            json!({"type": "message_stop"})
        );

        let response = router
            .route_messages(
                None,
                &tenant_meta,
                message_request(json!({
                    "model": "nope", "max_tokens": 0,
                    "messages": [{"role": "user", "content": "hi"}]
                })),
                "nope",
            )
            .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "model_not_found"
        );

        // The window check the pipeline makes after worker selection, here on
        // the smallest window the model's workers advertise: a worker whose
        // card holds the three-token prompt exactly changes nothing; one with
        // a two-token window refuses it as the pipeline would.
        let window_worker = |url: &str, context_length: u32| {
            Arc::new(
                BasicWorkerBuilder::new(url)
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .model(ModelCard::new("m").with_context_length(context_length))
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            )
        };
        ctx.worker_registry
            .register(window_worker("grpc://worker-window-3:30000", 3))
            .expect("register worker");
        let response = router
            .route_messages(None, &tenant_meta, message_request(body.clone()), "m")
            .await;
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "a three-token prompt fits a three-token window"
        );
        ctx.worker_registry
            .register(window_worker("grpc://worker-window-2:30000", 2))
            .expect("register worker");
        let response = router
            .route_messages(None, &tenant_meta, message_request(body), "m")
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "context_length_exceeded"
        );
    }

    /// PD-mode completion must honor a per-model retry override, not the router
    /// default.
    #[tokio::test]
    async fn pd_completion_honors_per_model_retry_override() {
        let ctx = grpc_ctx(pd_routing_mode()).await;
        let router = GrpcRouter::new(&ctx, Mode::PrefillDecode).expect("pd router");

        // Router default differs from the override so the assertion is meaningful.
        let default_retries = router.retry_config.max_retries;
        let override_retries = default_retries + 7;
        let override_config = RetryConfig {
            max_retries: override_retries,
            ..RetryConfig::default()
        };
        ctx.worker_registry
            .set_model_retry_config("model-a", override_config, true);

        let resolved = router.resolve_retry_config_for_canonical("model-a");
        assert_eq!(
            resolved.max_retries, override_retries,
            "PD completion must use the per-model override, not the router default"
        );

        let fallback = router.resolve_retry_config_for_canonical("model-without-override");
        assert_eq!(fallback.max_retries, default_retries);
    }

    /// PD serves /v1/responses: an unknown model is rejected per-request (404),
    /// not gated behind a blanket 501, and cancel reaches storage.
    #[tokio::test]
    async fn pd_debug_counts_include_zmq_workers() {
        let ctx = grpc_ctx(pd_routing_mode()).await;
        for (url, worker_type, mode) in [
            (
                "grpc://prefill:30000",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
            ),
            (
                "ipc:///tmp/smg-test-prefill",
                WorkerType::Prefill,
                ConnectionMode::Zmq,
            ),
            (
                "ipc:///tmp/smg-test-decode",
                WorkerType::Decode,
                ConnectionMode::Zmq,
            ),
        ] {
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .connection_mode(mode)
                .model(ModelCard::new("m"))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build();
            ctx.worker_registry
                .register(Arc::new(worker))
                .expect("register worker");
        }

        let router = GrpcRouter::new(&ctx, Mode::PrefillDecode).expect("pd router");
        let debug = format!("{router:?}");
        assert!(
            debug.contains("prefill_workers_count: 2"),
            "ZMQ prefill worker missing from debug counts: {debug}"
        );
        assert!(
            debug.contains("decode_workers_count: 1"),
            "ZMQ decode worker missing from debug counts: {debug}"
        );
    }

    #[tokio::test]
    async fn pd_router_serves_responses_and_cancel() {
        let ctx = grpc_ctx(pd_routing_mode()).await;
        let router = GrpcRouter::new(&ctx, Mode::PrefillDecode).expect("pd router");
        let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

        let request = responses_request("missing-model");
        let response = router
            .route_responses(None, &tenant_meta, request.clone(), "missing-model")
            .await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);

        let cancel = router.cancel_response(None, "resp_missing").await;
        assert_eq!(cancel.status(), StatusCode::NOT_FOUND);
    }

    /// EPD still 501s /v1/responses and cancel.
    #[tokio::test]
    async fn epd_router_501s_responses_and_cancel() {
        let ctx = grpc_ctx(epd_routing_mode()).await;
        let router = GrpcRouter::new(&ctx, Mode::EncodePrefillDecode).expect("epd router");
        let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

        let request = responses_request("missing-model");
        let response = router
            .route_responses(None, &tenant_meta, request.clone(), "missing-model")
            .await;
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);

        let cancel = router.cancel_response(None, "resp_missing").await;
        assert_eq!(cancel.status(), StatusCode::NOT_IMPLEMENTED);
    }

    /// A Responses request addressed to an alias must reach the Responses
    /// layer under the canonical model ID. That layer builds its SSE events
    /// and tool-call responses from this request rather than from the pipeline
    /// context, so leaving the alias here is what made the streaming and
    /// tool-call paths disagree with the plain-answer path about which model
    /// ran.
    #[tokio::test]
    async fn responses_request_is_canonical_before_the_responses_layer_sees_it() {
        let ctx = grpc_ctx(pd_routing_mode()).await;
        for (url, worker_type) in [
            ("grpc://prefill:30000", WorkerType::Prefill),
            ("grpc://decode:30000", WorkerType::Decode),
        ] {
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .model(ModelCard::new("canonical-model").with_alias("model-alias"))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build();
            ctx.worker_registry
                .register(Arc::new(worker))
                .expect("register worker");
        }

        let request = responses_request("model-alias");
        let (body, model_id) =
            canonicalize_responses_request(&ctx.worker_registry, &request, "model-alias");
        assert_eq!(
            body.model, "canonical-model",
            "the Responses layer must see the canonical model ID"
        );
        assert_eq!(
            model_id, "canonical-model",
            "worker selection must use the canonical model ID"
        );

        // The canonical ID is left alone, and costs no copy.
        let already_canonical = responses_request("canonical-model");
        let (body, model_id) = canonicalize_responses_request(
            &ctx.worker_registry,
            &already_canonical,
            "canonical-model",
        );
        assert!(matches!(body, Cow::Borrowed(_)));
        assert!(matches!(model_id, Cow::Borrowed(_)));
    }

    /// A per-model retry override must survive an alias. The override is keyed
    /// by canonical model ID and is read before the pipeline canonicalizes, so
    /// without its own resolution an alias silently falls back to the router
    /// default.
    #[tokio::test]
    async fn retry_override_survives_a_model_alias() {
        let ctx = grpc_ctx(pd_routing_mode()).await;
        let worker = BasicWorkerBuilder::new("grpc://decode:30000")
            .worker_type(WorkerType::Decode)
            .connection_mode(ConnectionMode::Grpc)
            .model(ModelCard::new("canonical-model").with_alias("model-alias"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        ctx.worker_registry
            .register(Arc::new(worker))
            .expect("register worker");

        let router = GrpcRouter::new(&ctx, Mode::PrefillDecode).expect("pd router");
        let default_retries = router.retry_config.max_retries;
        let override_retries = default_retries + 7;
        ctx.worker_registry.set_model_retry_config(
            "canonical-model",
            RetryConfig {
                max_retries: override_retries,
                ..RetryConfig::default()
            },
            true,
        );

        // Mirrors the real call sequence every route_*_impl uses: resolve the
        // canonical model once, then look up the retry config for it.
        let retry_config_for = |model_id: &str| {
            router.resolve_retry_config_for_canonical(&router.resolve_canonical_model_id(model_id))
        };

        assert_eq!(
            retry_config_for("canonical-model").max_retries,
            override_retries
        );
        assert_eq!(
            retry_config_for("model-alias").max_retries,
            override_retries,
            "an aliased request must get the same retry override as the canonical ID"
        );
        assert_eq!(
            retry_config_for("unrelated-model").max_retries,
            default_retries
        );
    }
}
