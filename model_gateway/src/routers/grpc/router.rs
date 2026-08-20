use std::{borrow::Cow, sync::Arc};

use async_trait::async_trait;
use axum::{
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    classify::ClassifyRequest,
    common::{ContentPart, InputAudio},
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    messages::CreateMessageRequest,
    responses::ResponsesRequest,
    transcription::{AudioFile, TranscriptionRequest},
};
use serde_json::json;
use tracing::debug;

use super::{
    common::{
        responses::{
            handlers::cancel_response_impl, utils::validate_worker_availability, ResponsesContext,
        },
        stages::{RateLimitCell, RateLimitOutcome},
    },
    context::SharedComponents,
    harmony::{serve_harmony_responses, serve_harmony_responses_stream, HarmonyDetector},
    mode::Mode,
    multimodal::MultimodalComponents,
    pipeline::{Endpoint, PipelineDeps, RequestPipeline},
    regular::responses,
    utils::ParserResolver,
};
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    middleware::TenantRequestMeta,
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::retry::{is_retryable_response, RetryExecutor},
        error, RouterTrait,
    },
    worker::{WorkerRegistry, WorkerType},
};

const QWEN3_ASR_LANGUAGES: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("yue", "Cantonese"),
    ("zh", "Chinese"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("fil", "Filipino"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("de", "German"),
    ("el", "Greek"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("mk", "Macedonian"),
    ("ms", "Malay"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("es", "Spanish"),
    ("sv", "Swedish"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("vi", "Vietnamese"),
];

const ASR_TEXT_TAG: &str = "<asr_text>";
const MAX_ASR_PROMPT_BYTES: usize = 4096;
const QWEN3_ASR_LABEL_KEYS: &[&str] = &[
    "model",
    "model_path",
    "model_type",
    "hf_model_type",
    "tokenizer",
    "tokenizer_path",
];

fn strip_chatml_like_tokens(text: &str) -> String {
    let mut remaining = text;
    let mut output = String::with_capacity(text.len());
    while let Some(start) = remaining.find("<|") {
        output.push_str(&remaining[..start]);
        let candidate = &remaining[start + 2..];
        if let Some(end) = candidate.find("|>") {
            let token = &candidate[..end];
            if !token.is_empty() && !token.contains('|') {
                remaining = &candidate[end + 2..];
                continue;
            }
        }
        output.push_str("<|");
        remaining = candidate;
    }
    output.push_str(remaining);
    output
}

fn sanitize_qwen3_asr_prompt(mut text: String) -> Result<String, Box<Response>> {
    if text.len() > MAX_ASR_PROMPT_BYTES {
        return Err(Box::new(error::bad_request(
            "asr_prompt_too_long",
            format!("Qwen3-ASR prompt must not exceed {MAX_ASR_PROMPT_BYTES} bytes"),
        )));
    }

    loop {
        let sanitized = strip_chatml_like_tokens(&text).replace(ASR_TEXT_TAG, "");
        if sanitized == text {
            return Ok(text);
        }
        text = sanitized;
    }
}

fn normalize_qwen3_asr_language(language: Option<&str>) -> Result<Option<String>, Box<Response>> {
    let Some(language) = language.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    QWEN3_ASR_LANGUAGES
        .iter()
        .find(|(code, name)| {
            code.eq_ignore_ascii_case(language) || name.eq_ignore_ascii_case(language)
        })
        .map(|(_, name)| Some((*name).to_string()))
        .ok_or_else(|| {
            Box::new(error::bad_request(
                "unsupported_transcription_language",
                format!("Qwen3-ASR does not support language '{language}'"),
            ))
        })
}

fn is_qwen3_asr_identifier(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    value.contains("qwen3-asr") || value.contains("qwen3_asr")
}

fn is_qwen3_asr_metadata_label(key: &str, value: &str) -> bool {
    QWEN3_ASR_LABEL_KEYS.contains(&key) && is_qwen3_asr_identifier(value)
}

fn is_qwen3_asr_target(worker_registry: &WorkerRegistry, model_id: &str) -> bool {
    if is_qwen3_asr_identifier(model_id) {
        return true;
    }

    worker_registry.get_by_model(model_id).iter().any(|worker| {
        let metadata = worker.metadata();
        is_qwen3_asr_identifier(metadata.model_id())
            || metadata
                .spec
                .labels
                .iter()
                .any(|(key, value)| is_qwen3_asr_metadata_label(key, value))
    })
}

fn audio_format(audio: &AudioFile) -> String {
    let content_type = audio.content_type.as_deref().unwrap_or_default();
    if content_type.eq_ignore_ascii_case("audio/mpeg")
        || content_type.eq_ignore_ascii_case("audio/mp3")
    {
        return "mp3".to_string();
    }
    if content_type.eq_ignore_ascii_case("audio/wav")
        || content_type.eq_ignore_ascii_case("audio/wave")
        || content_type.eq_ignore_ascii_case("audio/x-wav")
    {
        return "wav".to_string();
    }
    audio
        .file_name
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase())
        .filter(|extension| !extension.is_empty())
        .unwrap_or_else(|| "wav".to_string())
}

fn build_qwen3_asr_chat_request(
    body: &TranscriptionRequest,
    audio: &AudioFile,
    language: Option<&str>,
) -> Result<ChatCompletionRequest, Box<Response>> {
    let mut messages = Vec::with_capacity(3);
    if let Some(prompt) = body.prompt.as_deref() {
        let prompt = sanitize_qwen3_asr_prompt(prompt.to_string())?;
        let prompt = prompt.trim();
        if !prompt.is_empty() {
            messages.push(ChatMessage::System {
                content: MessageContent::Text(prompt.to_string()),
                name: None,
            });
        }
    }
    messages.push(ChatMessage::User {
        content: MessageContent::Parts(vec![ContentPart::InputAudio {
            input_audio: InputAudio {
                data: BASE64_STANDARD.encode(&audio.bytes),
                format: audio_format(audio),
            },
        }]),
        name: None,
    });

    let continue_final_message = if let Some(language) = language {
        messages.push(ChatMessage::Assistant {
            content: Some(MessageContent::Text(format!(
                "language {language}<asr_text>"
            ))),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        });
        true
    } else {
        false
    };

    Ok(ChatCompletionRequest {
        messages,
        model: body.model.clone(),
        n: Some(1),
        stream: false,
        temperature: Some(body.temperature.unwrap_or(0.0)),
        continue_final_message,
        skip_special_tokens: true,
        separate_reasoning: false,
        stream_reasoning: false,
        ..Default::default()
    })
}

fn parse_qwen3_asr_output(raw: &str) -> String {
    let cleaned = raw.replace("<|im_end|>", "");
    let cleaned = cleaned.trim();
    cleaned
        .split_once("<asr_text>")
        .map_or(cleaned, |(_, transcription)| transcription)
        .trim()
        .to_string()
}

#[derive(Clone, Copy, Debug)]
enum TranscriptionResponseFormat {
    Json,
    Text,
}

fn parse_transcription_response_format(
    format: Option<&str>,
) -> Result<TranscriptionResponseFormat, Box<Response>> {
    match format.unwrap_or("json").to_ascii_lowercase().as_str() {
        "json" => Ok(TranscriptionResponseFormat::Json),
        "text" => Ok(TranscriptionResponseFormat::Text),
        unsupported => Err(Box::new(error::bad_request(
            "unsupported_transcription_response_format",
            format!(
                "Qwen3-ASR does not provide timestamps required for response format '{unsupported}'"
            ),
        ))),
    }
}

fn transcription_response(format: TranscriptionResponseFormat, text: String) -> Response {
    match format {
        TranscriptionResponseFormat::Json => axum::Json(json!({"text": text})).into_response(),
        TranscriptionResponseFormat::Text => {
            ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response()
        }
    }
}

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

        // Create multimodal components (best-effort; non-fatal if initialization fails)
        let multimodal = match MultimodalComponents::new(ctx.multimodal_config_registry.clone()) {
            Ok(mc) => Some(Arc::new(mc)),
            Err(e) => {
                tracing::warn!("Multimodal components initialization failed (non-fatal): {e}");
                None
            }
        };

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
        });

        // Deps for the parser-consuming endpoints (chat/messages/harmony).
        let configured_deps = PipelineDeps::new(
            worker_registry.clone(),
            policy_registry.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
            ctx.rate_limit_manager.clone(),
        );
        // Deps for the parser-free endpoints (completion/embeddings/classify).
        // Only completion's stage list actually reads `rate_limit_manager`;
        // embeddings/classify never insert `RateLimitReserveStage`.
        let pair_deps = PipelineDeps::pair(
            worker_registry.clone(),
            policy_registry.clone(),
            ctx.rate_limit_manager.clone(),
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

    /// A rate-limit denial must never be retried -- retrying immediately
    /// against the same tenant budget defeats `retry_after_secs`.
    fn reservation_denied(cell: &RateLimitCell) -> bool {
        matches!(cell.peek(), Some(RateLimitOutcome::Denied))
    }

    /// Resolve `model_id` to its canonical form once, before a retry loop
    /// starts. Every dispatch attempt for that logical request must reuse
    /// this same value -- re-resolving the alias fresh per attempt (as
    /// `RequestContext::new` otherwise would) lets an alias repointed
    /// mid-retry dispatch a different model than whatever the tenant
    /// rate-limit reservation was actually made for, bypassing that model's
    /// own policy and settling its usage against the wrong budget.
    fn resolve_canonical_model_id(&self, model_id: &str) -> String {
        self.worker_registry
            .resolve_model_alias(model_id)
            .as_deref()
            .unwrap_or(model_id)
            .to_string()
    }

    /// Retry metrics for one backoff, labeled per mode: Regular emits a single
    /// `regular` worker label; PD/EPD emit `prefill` and `decode` (never
    /// `encode`).
    fn record_retry(&self, endpoint: &'static str) {
        match self.mode {
            Mode::Regular => {
                Metrics::record_worker_retry(metrics_labels::WORKER_REGULAR, endpoint);
            }
            Mode::PrefillDecode | Mode::EncodePrefillDecode => {
                Metrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
            }
        }
    }

    /// Record retry-exhaustion metrics, labeled per mode (see [`Self::record_retry`]).
    fn record_retries_exhausted(&self, endpoint: &'static str) {
        match self.mode {
            Mode::Regular => {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_REGULAR, endpoint);
            }
            Mode::PrefillDecode | Mode::EncodePrefillDecode => {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_DECODE, endpoint);
            }
        }
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

        // Clone values needed for retry closure. Canonicalize once, up
        // front, so every attempt (and the reservation `RateLimitReserveStage`
        // makes on the first one) targets the same model -- see
        // `resolve_canonical_model_id`'s doc comment. The body's own `model`
        // field is rewritten to match: by this point `model_id_cloned` is
        // already canonical, so `RequestContext::new`'s own alias resolve
        // would no-op and otherwise leave the alias sitting in the body for
        // response metadata and parser selection to read.
        let model_id_cloned = self.resolve_canonical_model_id(model_id);
        let mut canonical_body = body;
        canonical_body.model = model_id_cloned.clone();
        let request = Arc::new(canonical_body);
        let headers_cloned = headers.cloned();
        let components = self.shared_components.clone();
        let tenant_meta_cloned = tenant_meta.clone();
        let rate_limit_cell = Arc::new(RateLimitCell::new());

        let retry_config = self.resolve_retry_config_for_canonical(&model_id_cloned);

        let response = RetryExecutor::execute_response_with_retry(
            &retry_config,
            // Operation: execute pipeline (creates fresh context each attempt)
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                let tenant_meta = tenant_meta_cloned.clone();
                let rate_limit_cell = Arc::clone(&rate_limit_cell);
                async move {
                    pipeline
                        .execute_chat(
                            request,
                            headers,
                            model_id,
                            components,
                            Some(tenant_meta),
                            Some(rate_limit_cell),
                        )
                        .await
                }
            },
            // Should retry: a rate-limit denial must never be retried
            // (retrying immediately against the same tenant budget defeats
            // retry_after_secs); otherwise check if status is retryable.
            |res, _attempt| {
                if Self::reservation_denied(&rate_limit_cell) {
                    return false;
                }
                is_retryable_response(res)
            },
            // On backoff: record retry metrics
            |delay, attempt| {
                self.record_retry(metrics_labels::ENDPOINT_CHAT);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            // On exhausted: record exhaustion
            || {
                self.record_retries_exhausted(metrics_labels::ENDPOINT_CHAT);
            },
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

        // Clone values needed for retry closure. Canonicalize once, up
        // front -- see `resolve_canonical_model_id`'s doc comment. Rewrite
        // the body's `model` field to match; see `route_chat_impl`.
        let model_id_cloned = self.resolve_canonical_model_id(model_id);
        let mut canonical_body = body;
        canonical_body.model = model_id_cloned.clone();
        let request = Arc::new(canonical_body);
        let headers_cloned = headers.cloned();
        let components = self.shared_components.clone();
        let tenant_meta_cloned = tenant_meta.clone();
        let pipeline = &self.pipeline;
        let rate_limit_cell = Arc::new(RateLimitCell::new());

        let retry_config = self.resolve_retry_config_for_canonical(&model_id_cloned);

        let response = RetryExecutor::execute_response_with_retry(
            &retry_config,
            // Operation: execute pipeline (creates fresh context each attempt)
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                let tenant_meta = tenant_meta_cloned.clone();
                let rate_limit_cell = Arc::clone(&rate_limit_cell);
                async move {
                    pipeline
                        .execute_generate(
                            request,
                            headers,
                            model_id,
                            components,
                            Some(tenant_meta),
                            Some(rate_limit_cell),
                        )
                        .await
                }
            },
            // Should retry: a rate-limit denial must never be retried; see route_chat_impl.
            |res, _attempt| {
                if Self::reservation_denied(&rate_limit_cell) {
                    return false;
                }
                is_retryable_response(res)
            },
            // On backoff: record retry metrics
            |delay, attempt| {
                self.record_retry(metrics_labels::ENDPOINT_GENERATE);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            // On exhausted: record exhaustion
            || {
                self.record_retries_exhausted(metrics_labels::ENDPOINT_GENERATE);
            },
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

        if is_harmony {
            debug!(
                "Processing Harmony responses request for model: {}, streaming: {}",
                model_id,
                body.stream.unwrap_or(false)
            );
            let harmony_ctx = ResponsesContext::new(
                Arc::new(harmony_pipeline.clone()),
                self.shared_components.clone(),
                harmony_responses_context.response_storage.clone(),
                harmony_responses_context.conversation_storage.clone(),
                harmony_responses_context.conversation_item_storage.clone(),
                harmony_responses_context.mcp_orchestrator.clone(),
                harmony_responses_context.mcp_format_registry.clone(),
                smg_data_connector::current_request_context(),
            );

            if body.stream.unwrap_or(false) {
                serve_harmony_responses_stream(&harmony_ctx, body.into_owned(), tenant_meta.clone())
                    .await
            } else {
                match serve_harmony_responses(&harmony_ctx, body.into_owned(), tenant_meta.clone())
                    .await
                {
                    Ok(response) => axum::Json(response).into_response(),
                    Err(error_response) => error_response,
                }
            }
        } else {
            responses::route_responses(
                responses_context,
                Arc::new(body.into_owned()),
                headers.cloned(),
                tenant_meta.clone(),
                model_id.to_string(),
            )
            .await
        }
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

        // Clone values needed for retry closure. Canonicalize once, up
        // front -- see `resolve_canonical_model_id`'s doc comment. Rewrite
        // the body's `model` field to match; see `route_chat_impl`.
        let model_id_cloned = self.resolve_canonical_model_id(model_id);
        let mut canonical_body = body;
        canonical_body.model = model_id_cloned.clone();
        let request = Arc::new(canonical_body);
        let headers_cloned = headers.cloned();
        let components = self.shared_components.clone();
        let tenant_meta_cloned = tenant_meta.clone();
        let pipeline = &self.messages_pipeline;
        let rate_limit_cell = Arc::new(RateLimitCell::new());

        let retry_config = self.resolve_retry_config_for_canonical(&model_id_cloned);

        let response = RetryExecutor::execute_response_with_retry(
            &retry_config,
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                let tenant_meta = tenant_meta_cloned.clone();
                let rate_limit_cell = Arc::clone(&rate_limit_cell);
                async move {
                    pipeline
                        .execute_messages(
                            request,
                            headers,
                            model_id,
                            components,
                            Some(tenant_meta),
                            Some(rate_limit_cell),
                        )
                        .await
                }
            },
            // Should retry: a rate-limit denial must never be retried; see route_chat_impl.
            |res, _attempt| {
                if Self::reservation_denied(&rate_limit_cell) {
                    return false;
                }
                is_retryable_response(res)
            },
            |delay, attempt| {
                self.record_retry(metrics_labels::ENDPOINT_MESSAGES);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                self.record_retries_exhausted(metrics_labels::ENDPOINT_MESSAGES);
            },
        )
        .await;

        Self::close_reservation_if_unsettled(&rate_limit_cell, response.status()).await;
        response
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
        let request = Arc::new(canonical_body);
        let headers_cloned = headers.cloned();
        let components = self.shared_components.clone();
        let tenant_meta_cloned = tenant_meta.clone();
        let pipeline = &self.completion_pipeline;
        let rate_limit_cell = Arc::new(RateLimitCell::new());

        let retry_config = self.resolve_retry_config_for_canonical(&model_id_cloned);

        let response = RetryExecutor::execute_response_with_retry(
            &retry_config,
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                let tenant_meta = tenant_meta_cloned.clone();
                let rate_limit_cell = Arc::clone(&rate_limit_cell);
                async move {
                    pipeline
                        .execute_completion(
                            request,
                            headers,
                            model_id,
                            components,
                            Some(tenant_meta),
                            Some(rate_limit_cell),
                        )
                        .await
                }
            },
            // Should retry: a rate-limit denial must never be retried; see route_chat_impl.
            |res, _attempt| {
                if Self::reservation_denied(&rate_limit_cell) {
                    return false;
                }
                is_retryable_response(res)
            },
            |delay, attempt| {
                self.record_retry(metrics_labels::ENDPOINT_COMPLETIONS);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                self.record_retries_exhausted(metrics_labels::ENDPOINT_COMPLETIONS);
            },
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

    async fn route_audio_transcriptions(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: &TranscriptionRequest,
        audio: AudioFile,
        model_id: &str,
    ) -> Response {
        // Qwen3-ASR transcription is Regular-only.
        if self.mode != Mode::Regular {
            return not_implemented("Audio transcriptions not implemented");
        }
        if !is_qwen3_asr_target(&self.worker_registry, model_id) {
            return error::bad_request(
                "audio_transcription_model_not_supported",
                "The TokenSpeed gRPC transcription adapter currently supports Qwen3-ASR only",
            );
        }
        let response_format =
            match parse_transcription_response_format(body.response_format.as_deref()) {
                Ok(format) => format,
                Err(response) => return *response,
            };
        if body.stream.unwrap_or(false) {
            return error::bad_request(
                "streaming_transcription_not_supported",
                "TokenSpeed Qwen3-ASR currently supports whole-file transcription only",
            );
        }
        if body
            .timestamp_granularities
            .as_ref()
            .is_some_and(|values| !values.is_empty())
        {
            return error::bad_request(
                "transcription_timestamps_not_supported",
                "Qwen3-ASR timestamps require the forced-aligner model and are not supported",
            );
        }

        let requested_language = match normalize_qwen3_asr_language(body.language.as_deref()) {
            Ok(language) => language,
            Err(response) => return *response,
        };
        let chat_request =
            match build_qwen3_asr_chat_request(body, &audio, requested_language.as_deref()) {
                Ok(request) => request,
                Err(response) => return *response,
            };

        let chat_response = match self
            .pipeline
            .execute_chat_for_responses(
                Arc::new(chat_request),
                headers.cloned(),
                model_id.to_string(),
                Arc::clone(&self.shared_components),
                Some(tenant_meta.clone()),
            )
            .await
        {
            Ok(response) => response,
            Err(response) => return response,
        };
        let Some(content) = chat_response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
        else {
            return error::internal_error(
                "empty_transcription_response",
                "Qwen3-ASR returned no transcription text",
            );
        };
        let text = parse_qwen3_asr_output(content);
        transcription_response(response_format, text)
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
mod tests {
    use axum::http::StatusCode;
    use bytes::Bytes;

    use super::*;

    fn transcription_request() -> TranscriptionRequest {
        TranscriptionRequest {
            model: "Qwen/Qwen3-ASR-1.7B".to_string(),
            ..Default::default()
        }
    }

    fn wav_file() -> AudioFile {
        AudioFile {
            bytes: Bytes::from_static(b"RIFFtest"),
            file_name: "sample.wav".to_string(),
            content_type: Some("audio/wav".to_string()),
        }
    }

    #[test]
    fn normalizes_qwen3_asr_language_code_or_name() {
        assert_eq!(
            normalize_qwen3_asr_language(Some("zh")).unwrap(),
            Some("Chinese".to_string())
        );
        assert_eq!(
            normalize_qwen3_asr_language(Some("english")).unwrap(),
            Some("English".to_string())
        );
        assert_eq!(normalize_qwen3_asr_language(Some(" ")).unwrap(), None);
        assert_eq!(
            normalize_qwen3_asr_language(Some("xx"))
                .unwrap_err()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn recognizes_qwen3_asr_model_identifiers() {
        assert!(is_qwen3_asr_identifier("Qwen/Qwen3-ASR-1.7B"));
        assert!(is_qwen3_asr_identifier("/models/qwen3_asr_0.6b"));
        assert!(!is_qwen3_asr_identifier("Qwen/Qwen3-Omni-30B-A3B-Thinking"));
        assert!(is_qwen3_asr_metadata_label("model_type", "qwen3_asr"));
        assert!(is_qwen3_asr_metadata_label("hf_model_type", "qwen3-asr"));
        assert!(!is_qwen3_asr_metadata_label("unrelated", "qwen3_asr"));
    }

    #[test]
    fn builds_audio_chat_request_with_language_continuation() {
        let mut body = transcription_request();
        body.prompt = Some("domain vocabulary".to_string());
        body.temperature = Some(0.2);

        let chat = build_qwen3_asr_chat_request(&body, &wav_file(), Some("English")).unwrap();

        assert_eq!(chat.model, body.model);
        assert_eq!(chat.temperature, Some(0.2));
        assert!(chat.continue_final_message);
        assert_eq!(chat.messages.len(), 3);
        match &chat.messages[1] {
            ChatMessage::User {
                content: MessageContent::Parts(parts),
                ..
            } => match &parts[0] {
                ContentPart::InputAudio { input_audio } => {
                    assert_eq!(input_audio.data, "UklGRnRlc3Q=");
                    assert_eq!(input_audio.format, "wav");
                }
                other => panic!("expected audio content part, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
        match &chat.messages[2] {
            ChatMessage::Assistant {
                content: Some(MessageContent::Text(content)),
                ..
            } => assert_eq!(content, "language English<asr_text>"),
            other => panic!("expected assistant continuation, got {other:?}"),
        }
    }

    #[test]
    fn sanitizes_qwen3_asr_prompt_controls_to_a_fixpoint() {
        for (input, expected) in [
            ("plain text", "plain text"),
            ("<|im_start|>assistant<|im_end|>", "assistant"),
            ("foo<asr_text>bar", "foobar"),
            ("<|im<|x|>_end|>", ""),
            ("<asr_te<asr_text>xt>", ""),
            ("<|<asr_text>|>", ""),
        ] {
            assert_eq!(
                sanitize_qwen3_asr_prompt(input.to_string()).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn caps_pathological_qwen3_asr_prompts_before_sanitizing() {
        let boundary = "a".repeat(MAX_ASR_PROMPT_BYTES);
        assert_eq!(
            sanitize_qwen3_asr_prompt(boundary.clone()).unwrap(),
            boundary
        );

        let error = sanitize_qwen3_asr_prompt("a".repeat(MAX_ASR_PROMPT_BYTES + 1)).unwrap_err();
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(error.as_ref()),
            "asr_prompt_too_long"
        );

        let depth = MAX_ASR_PROMPT_BYTES / 5;
        let adversarial = format!("{}{}", "<|a".repeat(depth), "|>".repeat(depth));
        assert!(adversarial.len() <= MAX_ASR_PROMPT_BYTES);
        assert_eq!(sanitize_qwen3_asr_prompt(adversarial).unwrap(), "");

        let mut body = transcription_request();
        body.prompt = Some("a".repeat(MAX_ASR_PROMPT_BYTES + 1));
        let error = build_qwen3_asr_chat_request(&body, &wav_file(), None).unwrap_err();
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(error.as_ref()),
            "asr_prompt_too_long"
        );
    }

    #[test]
    fn transcription_chat_defaults_to_greedy_decoding() {
        let chat =
            build_qwen3_asr_chat_request(&transcription_request(), &wav_file(), None).unwrap();

        assert_eq!(chat.temperature, Some(0.0));
        assert!(!chat.continue_final_message);
    }

    #[test]
    fn parses_qwen3_asr_tagged_and_plain_outputs() {
        assert_eq!(
            parse_qwen3_asr_output("language Chinese<asr_text>\u{4f60}\u{597d}<|im_end|>"),
            "\u{4f60}\u{597d}"
        );
        assert_eq!(
            parse_qwen3_asr_output("plain transcript"),
            "plain transcript"
        );
    }

    #[test]
    fn transcription_response_rejects_timestamp_formats() {
        assert_eq!(
            parse_transcription_response_format(Some("verbose_json"))
                .unwrap_err()
                .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            parse_transcription_response_format(Some("srt"))
                .unwrap_err()
                .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            transcription_response(
                parse_transcription_response_format(Some("json")).unwrap(),
                "text".to_string(),
            )
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            transcription_response(
                parse_transcription_response_format(Some("text")).unwrap(),
                "text".to_string(),
            )
            .status(),
            StatusCode::OK
        );
    }
}

#[cfg(test)]
mod pd_tests {
    use std::sync::{Arc, OnceLock};

    use llm_tokenizer::registry::TokenizerRegistry;
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
    use reasoning_parser::ParserFactory as ReasoningParserFactory;
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
                .client(reqwest::Client::new())
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
