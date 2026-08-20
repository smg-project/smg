//! Pipeline orchestrator for gRPC router request processing
//!
//! This module defines the RequestPipeline orchestrator that coordinates
//! the execution of pipeline stages from request preparation to response delivery.

use std::{sync::Arc, time::Instant};

use axum::response::{IntoResponse, Response};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    messages::CreateMessageRequest,
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{debug, error};

// Import embedding-specific, classify-specific, messages-specific, and completion-specific stages
use super::regular::stages::classify::ClassifyResponseProcessingStage;
use super::{
    common::{responses::ResponsesContext, stages::*},
    context::*,
    harmony,
    mode::Mode,
    regular::{
        processor,
        stages::{
            completion::{
                CompletionPreparationStage, CompletionRequestBuildingStage,
                CompletionResponseProcessingStage,
            },
            embedding::{
                preparation::EmbeddingPreparationStage,
                request_building::EmbeddingRequestBuildingStage,
                response_processing::EmbeddingResponseProcessingStage,
            },
            messages::{
                MessagePreparationStage, MessageRequestBuildingStage,
                MessageResponseProcessingStage,
            },
            ChatGeneratePreparationStage, ChatGenerateRequestBuildingStage,
            ChatGenerateResponseProcessingStage,
        },
        streaming,
    },
    utils,
    utils::error_type_from_status,
};
use crate::{
    middleware::TenantRequestMeta,
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    policies::PolicyRegistry,
    rate_limit::{RateLimitManager, UsageSettlement},
    routers::{common::request_lease::ErasedLease, error},
    worker::WorkerRegistry,
};

/// Which endpoint a pipeline serves. Selects the endpoint-specific stage list
/// (preparation / request-building / response-processing); `Mode` then selects
/// the disaggregation params within that list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Endpoint {
    Chat,
    Messages,
    Completion,
    Harmony,
    Embeddings,
    Classify,
}

/// Construction dependencies shared by every endpoint pipeline.
///
/// The parser factories/overrides are consumed only by the chat/messages/harmony
/// processors; completion builds its own default-factory processors and
/// embeddings/classify build none.
#[derive(Clone)]
pub(crate) struct PipelineDeps {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    tool_parser_factory: ToolParserFactory,
    reasoning_parser_factory: ReasoningParserFactory,
    configured_tool_parser: Option<String>,
    configured_reasoning_parser: Option<String>,
    /// `None` when tenant rate limiting is disabled; read only by the
    /// endpoints that insert `RateLimitReserveStage` (chat/messages/completion/harmony).
    rate_limit_manager: Option<Arc<RateLimitManager>>,
}

impl PipelineDeps {
    /// Full deps for the chat/messages/harmony endpoints, which consume the
    /// configured parser factories/overrides.
    pub(crate) fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
        rate_limit_manager: Option<Arc<RateLimitManager>>,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
            rate_limit_manager,
        }
    }

    /// Deps for endpoints (embeddings/classify/completion) with no configured
    /// parsers; the parser fields are placeholders those endpoints never read.
    pub(crate) fn pair(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        rate_limit_manager: Option<Arc<RateLimitManager>>,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            configured_tool_parser: None,
            configured_reasoning_parser: None,
            rate_limit_manager,
        }
    }

    /// Build the chat/messages response processor pair from the configured
    /// parser factories, labeled with `backend`.
    fn configured_processors(
        &self,
        backend: &'static str,
    ) -> (
        processor::ResponseProcessor,
        Arc<streaming::StreamingProcessor>,
    ) {
        let parser_resolver = utils::ParserResolver::new(
            self.worker_registry.clone(),
            self.configured_tool_parser.clone(),
            self.configured_reasoning_parser.clone(),
        );
        let processor = processor::ResponseProcessor::new(
            self.tool_parser_factory.clone(),
            self.reasoning_parser_factory.clone(),
            parser_resolver.clone(),
        );
        let streaming_processor = Arc::new(streaming::StreamingProcessor::new(
            self.tool_parser_factory.clone(),
            self.reasoning_parser_factory.clone(),
            parser_resolver,
            backend,
        ));
        (processor, streaming_processor)
    }

    /// Build the completion response processor pair from default parser
    /// factories (completion does not use configured parsers), labeled `backend`.
    fn default_processors(
        backend: &'static str,
    ) -> (
        processor::ResponseProcessor,
        Arc<streaming::StreamingProcessor>,
    ) {
        let processor = processor::ResponseProcessor::new(
            ToolParserFactory::default(),
            ReasoningParserFactory::default(),
            utils::ParserResolver::disabled(),
        );
        let streaming_processor = Arc::new(streaming::StreamingProcessor::new(
            ToolParserFactory::default(),
            ReasoningParserFactory::default(),
            utils::ParserResolver::disabled(),
            backend,
        ));
        (processor, streaming_processor)
    }

    #[cfg(test)]
    fn test_default() -> Self {
        use crate::config::types::PolicyConfig;
        Self {
            worker_registry: Arc::new(WorkerRegistry::new()),
            policy_registry: Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            configured_tool_parser: None,
            configured_reasoning_parser: None,
            rate_limit_manager: None,
        }
    }
}

/// Generic request pipeline for all request types
///
/// Orchestrates all stages from request preparation to response delivery.
/// Configured differently for regular vs PD mode.
#[derive(Clone)]
pub(crate) struct RequestPipeline {
    stages: Arc<Vec<Box<dyn PipelineStage>>>,
    /// Backend type for metrics labeling
    backend_type: &'static str,
}

impl RequestPipeline {
    fn wrong_response_type(
        &self,
        function: &'static str,
        expected: &'static str,
        response_type: &FinalResponse,
        model: &str,
        endpoint: &'static str,
    ) -> Response {
        error!(
            function = function,
            response_type = %response_type,
            "Wrong response type: expected {expected}, got {response_type}"
        );
        Metrics::record_router_error(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            model,
            endpoint,
            metrics_labels::ERROR_INTERNAL,
        );
        error::internal_error("wrong_response_type", "Internal error: wrong response type")
    }

    /// Settle a non-streaming success's reservation with real usage, if this
    /// logical request reserved one. No-op if the cell is unset (feature
    /// disabled or endpoint opted out) or was never admitted (e.g. prep
    /// failed before the reserve stage ran). Idempotent -- `settle_success`
    /// is a no-op if the handle was already resolved.
    async fn settle_reservation(
        cell: Option<&RateLimitCell>,
        actual_input_tokens: u32,
        completion_tokens: u32,
    ) {
        let Some(RateLimitOutcome::Admitted(handle)) = cell.and_then(RateLimitCell::peek) else {
            return;
        };
        handle
            .settle_success(UsageSettlement {
                actual_input_tokens,
                completion_tokens,
            })
            .await;
    }

    fn no_response_produced(
        &self,
        function: &'static str,
        model: &str,
        endpoint: &'static str,
    ) -> Response {
        error!(function = function, "No response produced by pipeline");
        Metrics::record_router_error(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            model,
            endpoint,
            metrics_labels::ERROR_INTERNAL,
        );
        error::internal_error("no_response_produced", "No response produced")
    }

    /// Build the pipeline for `endpoint` in the given disaggregation `mode`,
    /// mapping `mode` to the per-stage worker-selection, execution-plan, and
    /// PD-injection params. `None` for endpoint/mode combinations that have no
    /// pipeline: Harmony has no EPD variant, and embeddings/classify are
    /// single-worker only.
    pub(crate) fn build(endpoint: Endpoint, mode: Mode, deps: &PipelineDeps) -> Option<Self> {
        // PD and EPD are both served by the "pd" backend metrics bucket; only
        // plain Regular reports as "regular".
        let backend = match mode {
            Mode::Regular => metrics_labels::BACKEND_REGULAR,
            Mode::PrefillDecode | Mode::EncodePrefillDecode => metrics_labels::BACKEND_PD,
        };
        let worker_selection = mode.worker_selection();
        let plan_kind = mode.plan_kind();
        let inject_pd_metadata = mode.inject_pd_metadata();

        let stages: Vec<Box<dyn PipelineStage>> = match endpoint {
            Endpoint::Chat => {
                let (processor, streaming_processor) = deps.configured_processors(backend);
                let mut stages: Vec<Box<dyn PipelineStage>> = vec![
                    Box::new(ChatGeneratePreparationStage::new()),
                    Box::new(RateLimitReserveStage::new(deps.rate_limit_manager.clone())),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                ];
                if matches!(mode, Mode::EncodePrefillDecode) {
                    stages.push(Box::new(EncodeStage::new()));
                }
                stages.extend([
                    Box::new(ChatGenerateRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )) as Box<dyn PipelineStage>,
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(ChatGenerateResponseProcessingStage::new(
                        processor,
                        streaming_processor,
                    )),
                ]);
                stages
            }
            Endpoint::Messages => {
                let (processor, streaming_processor) = deps.configured_processors(backend);
                let mut stages: Vec<Box<dyn PipelineStage>> = vec![
                    Box::new(MessagePreparationStage),
                    Box::new(RateLimitReserveStage::new(deps.rate_limit_manager.clone())),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                ];
                if matches!(mode, Mode::EncodePrefillDecode) {
                    stages.push(Box::new(EncodeStage::new()));
                }
                stages.extend([
                    Box::new(MessageRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )) as Box<dyn PipelineStage>,
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(MessageResponseProcessingStage::new(
                        processor,
                        streaming_processor,
                    )),
                ]);
                stages
            }
            Endpoint::Completion => {
                // Completion uses default parser factories, not the configured ones.
                let (processor, streaming_processor) = PipelineDeps::default_processors(backend);
                let mut stages: Vec<Box<dyn PipelineStage>> = vec![
                    Box::new(CompletionPreparationStage),
                    Box::new(RateLimitReserveStage::new(deps.rate_limit_manager.clone())),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                ];
                if matches!(mode, Mode::EncodePrefillDecode) {
                    stages.push(Box::new(EncodeStage::new()));
                }
                stages.extend([
                    Box::new(CompletionRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )) as Box<dyn PipelineStage>,
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(CompletionResponseProcessingStage::new(
                        processor,
                        streaming_processor,
                    )),
                ]);
                stages
            }
            Endpoint::Harmony => {
                // Harmony has no EPD variant.
                if matches!(mode, Mode::EncodePrefillDecode) {
                    return None;
                }
                vec![
                    Box::new(harmony::stages::HarmonyPreparationStage::new()),
                    Box::new(RateLimitReserveStage::new(deps.rate_limit_manager.clone())),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                    Box::new(harmony::stages::HarmonyRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )),
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(harmony::stages::HarmonyResponseProcessingStage::new()),
                ]
            }
            Endpoint::Embeddings => {
                // Embeddings are single-worker only.
                if !matches!(mode, Mode::Regular) {
                    return None;
                }
                vec![
                    Box::new(EmbeddingPreparationStage::new()),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                    Box::new(EmbeddingRequestBuildingStage::new()),
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(EmbeddingResponseProcessingStage::new()),
                ]
            }
            Endpoint::Classify => {
                // Classify is single-worker only.
                if !matches!(mode, Mode::Regular) {
                    return None;
                }
                vec![
                    Box::new(EmbeddingPreparationStage::new()),
                    Box::new(WorkerSelectionStage::new(
                        deps.worker_registry.clone(),
                        deps.policy_registry.clone(),
                        worker_selection,
                    )),
                    Box::new(ClientAcquisitionStage),
                    Box::new(EmbeddingRequestBuildingStage::new()),
                    Box::new(DispatchMetadataStage),
                    Box::new(RequestExecutionStage::new()),
                    Box::new(ClassifyResponseProcessingStage::new()),
                ]
            }
        };

        Some(Self {
            stages: Arc::new(stages),
            backend_type: backend,
        })
    }

    /// Execute the complete pipeline for a chat request
    pub async fn execute_chat(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
        rate_limit_cell: Option<Arc<RateLimitCell>>,
        request_lease: Option<Arc<dyn ErasedLease>>,
    ) -> Response {
        let start = Instant::now();
        let streaming = request.stream;
        let mut ctx = RequestContext::for_chat(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;
        ctx.input.rate_limit_cell = rate_limit_cell;
        ctx.input.request_lease = request_lease;
        let model = ctx.input.model_id.clone();

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &model,
            metrics_labels::ENDPOINT_CHAT,
            bool_to_static_str(streaming),
        );

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage completed with streaming response - record success and return
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model,
                        metrics_labels::ENDPOINT_CHAT,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model,
                        metrics_labels::ENDPOINT_CHAT,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Chat(response)) => {
                let usage = response.usage.as_ref();
                Self::settle_reservation(
                    ctx.input.rate_limit_cell.as_deref(),
                    usage.map_or(0, |u| u.prompt_tokens),
                    usage.map_or(0, |u| u.completion_tokens),
                )
                .await;
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &model,
                    metrics_labels::ENDPOINT_CHAT,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(
                response_type @ (FinalResponse::Generate(_)
                | FinalResponse::Completion(_)
                | FinalResponse::Embedding(_)
                | FinalResponse::Classify(_)
                | FinalResponse::Messages(_)),
            ) => self.wrong_response_type(
                "execute_chat",
                "Chat",
                &response_type,
                &model,
                metrics_labels::ENDPOINT_CHAT,
            ),
            None => {
                self.no_response_produced("execute_chat", &model, metrics_labels::ENDPOINT_CHAT)
            }
        }
    }

    /// Execute the complete pipeline for a generate request
    pub async fn execute_generate(
        &self,
        request: Arc<GenerateRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
        rate_limit_cell: Option<Arc<RateLimitCell>>,
        request_lease: Option<Arc<dyn ErasedLease>>,
    ) -> Response {
        let start = Instant::now();
        let streaming = request.stream;
        let mut ctx = RequestContext::for_generate(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;
        ctx.input.rate_limit_cell = rate_limit_cell;
        ctx.input.request_lease = request_lease;
        let model_id = ctx.input.model_id.clone();

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &model_id,
            metrics_labels::ENDPOINT_GENERATE,
            bool_to_static_str(streaming),
        );

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_GENERATE,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_GENERATE,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Generate(response)) => {
                let actual_input_tokens = response.first().map_or(0, |r| r.meta_info.prompt_tokens);
                let completion_tokens =
                    response.iter().map(|r| r.meta_info.completion_tokens).sum();
                Self::settle_reservation(
                    ctx.input.rate_limit_cell.as_deref(),
                    actual_input_tokens,
                    completion_tokens,
                )
                .await;
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &model_id,
                    metrics_labels::ENDPOINT_GENERATE,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(
                response_type @ (FinalResponse::Chat(_)
                | FinalResponse::Completion(_)
                | FinalResponse::Embedding(_)
                | FinalResponse::Classify(_)
                | FinalResponse::Messages(_)),
            ) => self.wrong_response_type(
                "execute_generate",
                "Generate",
                &response_type,
                &model_id,
                metrics_labels::ENDPOINT_GENERATE,
            ),
            None => self.no_response_produced(
                "execute_generate",
                &model_id,
                metrics_labels::ENDPOINT_GENERATE,
            ),
        }
    }

    /// Execute the complete pipeline for a completion request
    pub async fn execute_completion(
        &self,
        request: Arc<CompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
        rate_limit_cell: Option<Arc<RateLimitCell>>,
        request_lease: Option<Arc<dyn ErasedLease>>,
    ) -> Response {
        let start = Instant::now();
        let streaming = request.stream;
        let mut ctx = RequestContext::for_completion(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;
        ctx.input.rate_limit_cell = rate_limit_cell;
        ctx.input.request_lease = request_lease;
        let model = ctx.input.model_id.clone();

        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &model,
            metrics_labels::ENDPOINT_COMPLETIONS,
            bool_to_static_str(streaming),
        );

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model,
                        metrics_labels::ENDPOINT_COMPLETIONS,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model,
                        metrics_labels::ENDPOINT_COMPLETIONS,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Completion(response)) => {
                let usage = response.usage.as_ref();
                Self::settle_reservation(
                    ctx.input.rate_limit_cell.as_deref(),
                    usage.map_or(0, |u| u.prompt_tokens),
                    usage.map_or(0, |u| u.completion_tokens),
                )
                .await;
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &model,
                    metrics_labels::ENDPOINT_COMPLETIONS,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(
                response_type @ (FinalResponse::Chat(_)
                | FinalResponse::Generate(_)
                | FinalResponse::Embedding(_)
                | FinalResponse::Classify(_)
                | FinalResponse::Messages(_)),
            ) => self.wrong_response_type(
                "execute_completion",
                "Completion",
                &response_type,
                &model,
                metrics_labels::ENDPOINT_COMPLETIONS,
            ),
            None => self.no_response_produced(
                "execute_completion",
                &model,
                metrics_labels::ENDPOINT_COMPLETIONS,
            ),
        }
    }

    /// Execute the complete pipeline for an embedding request
    pub async fn execute_embeddings(
        &self,
        request: Arc<EmbeddingRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Response {
        let mut ctx = RequestContext::for_embedding(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;
        let model_id = ctx.input.model_id.clone();
        debug!(
            "execute_embeddings: Starting execution for model: {}",
            &model_id
        );
        let start = Instant::now();

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &model_id,
            metrics_labels::ENDPOINT_EMBEDDINGS,
            bool_to_static_str(false),
        );

        for stage in self.stages.iter() {
            debug!("execute_embeddings: Executing stage: {}", stage.name());
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    debug!(
                        "execute_embeddings: Stage {} returned final response.",
                        stage.name()
                    );
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_EMBEDDINGS,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => {
                    debug!(
                        "execute_embeddings: Stage {} completed, continuing to next stage.",
                        stage.name()
                    );
                    continue;
                }
                Err(response) => {
                    error!(
                        "execute_embeddings: Stage {} failed with status {:?}, returning error response.",
                        stage.name(),
                        response.status()
                    );
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_EMBEDDINGS,
                        error_type_from_status(response.status()),
                    );
                    return response;
                }
            }
        }

        debug!(
            "execute_embeddings: Pipeline finished, processing final_response. Current state: {:?}",
            ctx.state.response.final_response
        );
        match ctx.state.response.final_response {
            Some(FinalResponse::Embedding(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &model_id,
                    metrics_labels::ENDPOINT_EMBEDDINGS,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(_) => {
                error!(function = "execute_embeddings", "Wrong response type");
                error::internal_error("wrong_response_type", "Internal error: wrong response type")
            }
            None => {
                error!(
                    function = "execute_embeddings",
                    "No final response produced by pipeline."
                );
                error::internal_error("no_response_produced", "No response produced")
            }
        }
    }

    /// Execute the complete pipeline for a classify request
    pub async fn execute_classify(
        &self,
        request: Arc<ClassifyRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Response {
        let mut ctx = RequestContext::for_classify(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;
        let model_id = ctx.input.model_id.clone();
        debug!(
            "execute_classify: Starting execution for model: {}",
            &model_id
        );
        let start = Instant::now();

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &model_id,
            metrics_labels::ENDPOINT_CLASSIFY,
            bool_to_static_str(false), // Classify is never streaming
        );

        for stage in self.stages.iter() {
            debug!("execute_classify: Executing stage: {}", stage.name());
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    debug!(
                        "execute_classify: Stage {} returned final response.",
                        stage.name()
                    );
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_CLASSIFY,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => {
                    debug!(
                        "execute_classify: Stage {} completed, continuing to next stage.",
                        stage.name()
                    );
                    continue;
                }
                Err(response) => {
                    error!(
                        "execute_classify: Stage {} failed with status {:?}, returning error response.",
                        stage.name(),
                        response.status()
                    );
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model_id,
                        metrics_labels::ENDPOINT_CLASSIFY,
                        error_type_from_status(response.status()),
                    );
                    return response;
                }
            }
        }

        debug!(
            "execute_classify: Pipeline finished, processing final_response. Current state: {:?}",
            ctx.state.response.final_response
        );
        match ctx.state.response.final_response {
            Some(FinalResponse::Classify(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &model_id,
                    metrics_labels::ENDPOINT_CLASSIFY,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(_) => {
                error!(function = "execute_classify", "Wrong response type");
                error::internal_error("wrong_response_type", "Internal error: wrong response type")
            }
            None => {
                error!(
                    function = "execute_classify",
                    "No final response produced by pipeline."
                );
                error::internal_error("no_response_produced", "No response produced")
            }
        }
    }

    /// Execute the complete pipeline for a Messages API request
    pub async fn execute_messages(
        &self,
        request: Arc<CreateMessageRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
        rate_limit_cell: Option<Arc<RateLimitCell>>,
        request_lease: Option<Arc<dyn ErasedLease>>,
    ) -> Response {
        let start = Instant::now();
        let streaming = request.stream.unwrap_or(false);
        let mut ctx = RequestContext::for_messages(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;
        ctx.input.rate_limit_cell = rate_limit_cell;
        ctx.input.request_lease = request_lease;
        let model = ctx.input.model_id.clone();

        // Record request start
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &model,
            metrics_labels::ENDPOINT_MESSAGES,
            bool_to_static_str(streaming),
        );

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage completed with streaming response
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model,
                        metrics_labels::ENDPOINT_MESSAGES,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &model,
                        metrics_labels::ENDPOINT_MESSAGES,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Messages(response)) => {
                Self::settle_reservation(
                    ctx.input.rate_limit_cell.as_deref(),
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )
                .await;
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &model,
                    metrics_labels::ENDPOINT_MESSAGES,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(
                response_type @ (FinalResponse::Chat(_)
                | FinalResponse::Generate(_)
                | FinalResponse::Completion(_)
                | FinalResponse::Embedding(_)
                | FinalResponse::Classify(_)),
            ) => self.wrong_response_type(
                "execute_messages",
                "Messages",
                &response_type,
                &model,
                metrics_labels::ENDPOINT_MESSAGES,
            ),
            None => self.no_response_produced(
                "execute_messages",
                &model,
                metrics_labels::ENDPOINT_MESSAGES,
            ),
        }
    }

    /// Execute chat pipeline for responses endpoint
    ///
    /// Used by ALL non-streaming /v1/responses requests.
    /// Uses the same 7 pipeline stages as execute_chat(), with two differences:
    /// 1. Returns Result<ChatCompletionResponse, Response> for tool_loop composition
    /// 2. Disallows streaming (responses endpoint uses different SSE format)
    pub async fn execute_chat_for_responses(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Result<ChatCompletionResponse, Response> {
        let mut ctx = RequestContext::for_chat(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;

        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(_response)) => {
                    // Streaming not supported for responses sync mode
                    error!(
                        function = "execute_chat_for_responses",
                        "Streaming attempted in responses context"
                    );
                    return Err(error::bad_request(
                        "streaming_not_supported",
                        "Streaming is not supported in this context".to_string(),
                    ));
                }
                Ok(None) => {
                    continue;
                }
                Err(response) => {
                    // Error occurred - return the response as-is to preserve HTTP status codes
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Chat(response)) => Ok(response),
            Some(FinalResponse::Generate(_))
            | Some(FinalResponse::Completion(_))
            | Some(FinalResponse::Embedding(_))
            | Some(FinalResponse::Classify(_))
            | Some(FinalResponse::Messages(_)) => {
                error!(
                    function = "execute_chat_for_responses",
                    "Wrong response type: expected Chat, got Generate/Embedding/Classify/Messages"
                );
                Err(error::internal_error(
                    "wrong_response_type",
                    "Internal error: wrong response type",
                ))
            }
            None => {
                error!(
                    function = "execute_chat_for_responses",
                    "No response produced by pipeline"
                );
                Err(error::internal_error(
                    "no_response_produced",
                    "No response produced",
                ))
            }
        }
    }

    /// Execute Harmony Responses API request through all pipeline stages
    ///
    /// This method runs a single iteration of the Responses API request,
    /// returning either ToolCallsFound (continue serving) or Completed (final response).
    ///
    /// Called by harmony::responses::serve_harmony_responses() for each iteration.
    ///
    /// # Arguments
    ///
    /// * `request` - Responses API request
    /// * `ctx` - Harmony Responses context with MCP manager and components
    ///
    /// # Returns
    ///
    /// ResponsesIterationResult indicating whether to continue iteration or return
    pub async fn execute_harmony_responses(
        &self,
        request: &openai_protocol::responses::ResponsesRequest,
        harmony_ctx: &ResponsesContext,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Result<harmony::ResponsesIterationResult, Response> {
        // Create RequestContext for this Responses request
        let mut ctx = RequestContext::for_responses(
            Arc::new(request.clone()),
            None,                  // No headers needed for internal pipeline execution
            request.model.clone(), // Model ID from request
            harmony_ctx.components.clone(),
        );
        ctx.input.tenant_request_meta = tenant_request_meta;

        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    // Stage returned early response (e.g., streaming) - not expected for Responses iteration
                    error!(
                        "Stage {} ({}) returned unexpected response during Responses iteration",
                        idx + 1,
                        stage.name()
                    );
                    return Err(response);
                }
                Ok(None) => {
                    continue;
                }
                Err(response) => {
                    // Stage failed
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        // Extract ResponsesIterationResult from context
        // This should have been set by HarmonyResponseProcessingStage
        ctx.state
            .response
            .responses_iteration_result
            .take()
            .ok_or_else(|| {
                error!(
                    function = "execute_harmony_responses",
                    "No ResponsesIterationResult produced by pipeline"
                );
                error::internal_error(
                    "no_responses_iteration_result",
                    "No ResponsesIterationResult produced by pipeline",
                )
            })
    }

    /// Execute Harmony Responses pipeline iteration with streaming support
    ///
    /// This version executes the pipeline up to the dispatch stage and returns
    /// the raw ExecutionResult (with stream) and LoadGuards for token-level streaming processing.
    /// The caller is responsible for keeping load_guards alive until stream processing completes.
    pub async fn execute_harmony_responses_streaming(
        &self,
        request: &openai_protocol::responses::ResponsesRequest,
        harmony_ctx: &ResponsesContext,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Result<(ExecutionResult, Option<LoadGuards>), Response> {
        // Create RequestContext for this Responses request
        let mut ctx = RequestContext::for_responses(
            Arc::new(request.clone()),
            None,
            request.model.clone(),
            harmony_ctx.components.clone(),
        );
        ctx.input.tenant_request_meta = tenant_request_meta;

        for (idx, stage) in self.stages.iter().enumerate() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    error!(
                        "Stage {} ({}) returned unexpected response during streaming Responses",
                        idx + 1,
                        stage.name()
                    );
                    return Err(response);
                }
                Ok(None) => continue,
                Err(response) => {
                    error!(
                        "Stage {} ({}) failed with status {}",
                        idx + 1,
                        stage.name(),
                        response.status()
                    );
                    return Err(response);
                }
            }
        }

        // Extract execution_result (the raw stream from workers) and load_guards
        let execution_result = ctx.state.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "execute_harmony_responses_streaming",
                "No ExecutionResult produced by pipeline"
            );
            error::internal_error(
                "no_execution_result_produced",
                "No ExecutionResult produced by pipeline",
            )
        })?;

        let load_guards = ctx.state.load_guards.take();

        Ok((execution_result, load_guards))
    }
}

#[cfg(test)]
mod build_parity_tests {
    use super::*;
    use crate::routers::grpc::mode::Mode;

    fn sigs(p: &RequestPipeline) -> Vec<String> {
        p.stages.iter().map(|s| s.signature()).collect()
    }

    fn v(stages: &[&str]) -> Vec<String> {
        stages.iter().map(|s| (*s).to_string()).collect()
    }

    /// Hand-transcribed expected stage signatures + metrics backend label per
    /// endpoint/mode. Do not regenerate from `build()` output, or this stops
    /// guarding anything: it exists to catch a wrong `build`/`mode.rs` mapping
    /// (a flipped `inject_pd_metadata`, wrong `plan_kind`, or swapped
    /// `WorkerSelectionMode`).
    ///
    /// Signature format: stages with no mode-varying args emit their short type
    /// name; the mode-bearing overrides append their args.
    /// `ChatGenerateRequestBuildingStage` is a composite wrapping the chat and
    /// generate request-building stages, both fed the same mode args.
    fn golden(endpoint: Endpoint, mode: Mode) -> (Vec<String>, &'static str) {
        const REGULAR: &str = metrics_labels::BACKEND_REGULAR;
        const PD: &str = metrics_labels::BACKEND_PD;
        match (endpoint, mode) {
            (Endpoint::Chat, Mode::Regular) => (
                v(&[
                    "ChatGeneratePreparationStage",
                    "RateLimitReserveStage",
                    "WorkerSelectionStage(Regular)",
                    "ClientAcquisitionStage",
                    "ChatGenerateRequestBuildingStage(ChatRequestBuildingStage(inject_pd_metadata=false, Single), GenerateRequestBuildingStage(inject_pd_metadata=false, Single))",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "ChatGenerateResponseProcessingStage",
                ]),
                REGULAR,
            ),
            (Endpoint::Chat, Mode::PrefillDecode) => (
                v(&[
                    "ChatGeneratePreparationStage",
                    "RateLimitReserveStage",
                    "WorkerSelectionStage(PrefillDecode)",
                    "ClientAcquisitionStage",
                    "ChatGenerateRequestBuildingStage(ChatRequestBuildingStage(inject_pd_metadata=true, PrefillDecode), GenerateRequestBuildingStage(inject_pd_metadata=true, PrefillDecode))",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "ChatGenerateResponseProcessingStage",
                ]),
                PD,
            ),
            (Endpoint::Chat, Mode::EncodePrefillDecode) => (
                v(&[
                    "ChatGeneratePreparationStage",
                    "RateLimitReserveStage",
                    "WorkerSelectionStage(EncodePrefillDecode)",
                    "ClientAcquisitionStage",
                    "EncodeStage",
                    "ChatGenerateRequestBuildingStage(ChatRequestBuildingStage(inject_pd_metadata=false, EncodePrefillDecode), GenerateRequestBuildingStage(inject_pd_metadata=false, EncodePrefillDecode))",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "ChatGenerateResponseProcessingStage",
                ]),
                PD,
            ),
            (Endpoint::Messages, Mode::Regular) => (
                v(&[
                    "MessagePreparationStage",
                    "RateLimitReserveStage",
                    "WorkerSelectionStage(Regular)",
                    "ClientAcquisitionStage",
                    "MessageRequestBuildingStage(inject_pd_metadata=false, Single)",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "MessageResponseProcessingStage",
                ]),
                REGULAR,
            ),
            (Endpoint::Messages, Mode::PrefillDecode) => (
                v(&[
                    "MessagePreparationStage",
                    "RateLimitReserveStage",
                    "WorkerSelectionStage(PrefillDecode)",
                    "ClientAcquisitionStage",
                    "MessageRequestBuildingStage(inject_pd_metadata=true, PrefillDecode)",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "MessageResponseProcessingStage",
                ]),
                PD,
            ),
            (Endpoint::Messages, Mode::EncodePrefillDecode) => (
                v(&[
                    "MessagePreparationStage",
                    "RateLimitReserveStage",
                    "WorkerSelectionStage(EncodePrefillDecode)",
                    "ClientAcquisitionStage",
                    "EncodeStage",
                    "MessageRequestBuildingStage(inject_pd_metadata=false, EncodePrefillDecode)",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "MessageResponseProcessingStage",
                ]),
                PD,
            ),
            (Endpoint::Completion, Mode::Regular) => (
                v(&[
                    "CompletionPreparationStage",
                    "RateLimitReserveStage",
                    "WorkerSelectionStage(Regular)",
                    "ClientAcquisitionStage",
                    "CompletionRequestBuildingStage(inject_pd_metadata=false, Single)",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "CompletionResponseProcessingStage",
                ]),
                REGULAR,
            ),
            (Endpoint::Completion, Mode::PrefillDecode) => (
                v(&[
                    "CompletionPreparationStage",
                    "RateLimitReserveStage",
                    "WorkerSelectionStage(PrefillDecode)",
                    "ClientAcquisitionStage",
                    "CompletionRequestBuildingStage(inject_pd_metadata=true, PrefillDecode)",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "CompletionResponseProcessingStage",
                ]),
                PD,
            ),
            (Endpoint::Completion, Mode::EncodePrefillDecode) => (
                v(&[
                    "CompletionPreparationStage",
                    "RateLimitReserveStage",
                    "WorkerSelectionStage(EncodePrefillDecode)",
                    "ClientAcquisitionStage",
                    "EncodeStage",
                    "CompletionRequestBuildingStage(inject_pd_metadata=false, EncodePrefillDecode)",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "CompletionResponseProcessingStage",
                ]),
                PD,
            ),
            (Endpoint::Harmony, Mode::Regular) => (
                v(&[
                    "HarmonyPreparationStage",
                    "RateLimitReserveStage",
                    "WorkerSelectionStage(Regular)",
                    "ClientAcquisitionStage",
                    "HarmonyRequestBuildingStage(inject_pd_metadata=false, Single)",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "HarmonyResponseProcessingStage",
                ]),
                REGULAR,
            ),
            (Endpoint::Harmony, Mode::PrefillDecode) => (
                v(&[
                    "HarmonyPreparationStage",
                    "RateLimitReserveStage",
                    "WorkerSelectionStage(PrefillDecode)",
                    "ClientAcquisitionStage",
                    "HarmonyRequestBuildingStage(inject_pd_metadata=true, PrefillDecode)",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "HarmonyResponseProcessingStage",
                ]),
                PD,
            ),
            // Embeddings and classify share prep + request building; classify
            // only swaps the response processor.
            (Endpoint::Embeddings, Mode::Regular) => (
                v(&[
                    "EmbeddingPreparationStage",
                    "WorkerSelectionStage(Regular)",
                    "ClientAcquisitionStage",
                    "EmbeddingRequestBuildingStage",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "EmbeddingResponseProcessingStage",
                ]),
                REGULAR,
            ),
            (Endpoint::Classify, Mode::Regular) => (
                v(&[
                    "EmbeddingPreparationStage",
                    "WorkerSelectionStage(Regular)",
                    "ClientAcquisitionStage",
                    "EmbeddingRequestBuildingStage",
                    "DispatchMetadataStage",
                    "RequestExecutionStage",
                    "ClassifyResponseProcessingStage",
                ]),
                REGULAR,
            ),
            (endpoint, mode) => panic!("no golden for invalid combo {endpoint:?}/{mode:?}"),
        }
    }

    /// Assert `build(endpoint, mode)` matches the hand-transcribed golden (stage
    /// sequence + mode-bearing args + metrics backend label).
    fn assert_parity(endpoint: Endpoint, mode: Mode, deps: &PipelineDeps) {
        let (expected_sigs, expected_backend) = golden(endpoint, mode);
        let built = RequestPipeline::build(endpoint, mode, deps)
            .unwrap_or_else(|| panic!("build({endpoint:?}, {mode:?}) should be valid"));
        assert_eq!(
            sigs(&built),
            expected_sigs,
            "stage parity for {endpoint:?}/{mode:?}"
        );
        assert_eq!(
            built.backend_type, expected_backend,
            "backend_type parity for {endpoint:?}/{mode:?}"
        );
    }

    #[test]
    fn build_matches_frozen_goldens() {
        let deps = PipelineDeps::test_default();

        for endpoint in [Endpoint::Chat, Endpoint::Messages, Endpoint::Completion] {
            for mode in [
                Mode::Regular,
                Mode::PrefillDecode,
                Mode::EncodePrefillDecode,
            ] {
                assert_parity(endpoint, mode, &deps);
            }
        }

        assert!(
            RequestPipeline::build(Endpoint::Harmony, Mode::EncodePrefillDecode, &deps).is_none(),
            "Harmony EPD must be invalid"
        );
        assert_parity(Endpoint::Harmony, Mode::Regular, &deps);
        assert_parity(Endpoint::Harmony, Mode::PrefillDecode, &deps);

        for endpoint in [Endpoint::Embeddings, Endpoint::Classify] {
            assert!(
                RequestPipeline::build(endpoint, Mode::PrefillDecode, &deps).is_none(),
                "{endpoint:?} PD must be invalid"
            );
            assert!(
                RequestPipeline::build(endpoint, Mode::EncodePrefillDecode, &deps).is_none(),
                "{endpoint:?} EPD must be invalid"
            );
            assert_parity(endpoint, Mode::Regular, &deps);
        }
    }
}

#[cfg(test)]
mod alias_pipeline_tests {
    use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
    use openai_protocol::{
        generate::GenerateRequest, model_card::ModelCard, worker::HealthCheckConfig,
    };
    use serde_json::json;

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
    };

    const CANONICAL_MODEL: &str = "canonical-model";
    const MODEL_ALIAS: &str = "model-alias";

    fn register_pd_worker(registry: &WorkerRegistry, url: &str, worker_type: WorkerType) {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Sglang)
            .model(ModelCard::new(CANONICAL_MODEL).with_alias(MODEL_ALIAS))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        registry.register(Arc::new(worker)).unwrap();
    }

    #[tokio::test]
    async fn pd_generate_alias_is_canonical_before_preparation() {
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_pd_worker(
            &worker_registry,
            "grpc://prefill:30000",
            WorkerType::Prefill,
        );
        register_pd_worker(&worker_registry, "grpc://decode:30000", WorkerType::Decode);

        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
        tokenizer_registry
            .load("tokenizer-id", CANONICAL_MODEL, "test", || async move {
                Ok(tokenizer)
            })
            .await
            .unwrap();

        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let deps = PipelineDeps::pair(worker_registry.clone(), policy_registry, None);
        let pipeline = RequestPipeline::build(Endpoint::Chat, Mode::PrefillDecode, &deps).unwrap();
        let components = Arc::new(SharedComponents {
            tokenizer_registry,
            worker_registry,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: utils::ParserResolver::disabled(),
            multimodal: None,
        });
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": MODEL_ALIAS,
            "text": "Hello"
        }))
        .unwrap();
        let mut ctx = RequestContext::for_generate(
            Arc::new(request),
            None,
            MODEL_ALIAS.to_string(),
            components,
        );

        assert_eq!(ctx.input.model_id, CANONICAL_MODEL);
        assert_eq!(ctx.generate_request_arc().model, CANONICAL_MODEL);

        for stage in pipeline.stages.iter() {
            assert!(stage.execute(&mut ctx).await.unwrap().is_none());
            if ctx.state.workers.is_some() {
                break;
            }
        }

        assert_eq!(ctx.input.model_id, CANONICAL_MODEL);
        assert_eq!(ctx.generate_request_arc().model, CANONICAL_MODEL);
        assert!(ctx.state.tokenizer.is_some());
        match ctx.state.workers.as_ref().unwrap() {
            WorkerSelection::Disaggregated {
                prefill, decode, ..
            } => {
                assert_eq!(prefill.url(), "grpc://prefill:30000");
                assert_eq!(decode.url(), "grpc://decode:30000");
            }
            WorkerSelection::Single { .. } => panic!("expected PD worker selection"),
        }
    }
}

#[cfg(test)]
mod request_release_tests {
    use std::{
        pin::Pin,
        sync::{
            atomic::{AtomicBool, Ordering},
            Weak,
        },
        time::Duration,
    };

    use futures::Stream;
    use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
    use openai_protocol::{
        completion::CompletionRequest, model_card::ModelCard, worker::HealthCheckConfig,
    };
    use portpicker::pick_unused_port;
    use smg_grpc_client::{common_proto as common, tokenspeed_proto as ts};
    use tokio_stream::wrappers::ReceiverStream;
    use tonic::{transport::Server, Request as TonicRequest, Response as TonicResponse, Status};
    use ts::token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer};

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        routers::common::request_lease::{ReleasePoint, RequestLease, RoutingDerivatives},
        worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
    };

    const MODEL: &str = "request-release-test-model";

    type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;
    type KvEventStream = Pin<Box<dyn Stream<Item = Result<common::KvEventBatch, Status>> + Send>>;
    type TokenizerStream =
        Pin<Box<dyn Stream<Item = Result<common::GetTokenizerChunk, Status>> + Send>>;

    /// TokenSpeed stub gated on the parsed request's drop probe: it withholds
    /// its tokens (or, with `gate_rpc`, the generate RPC itself) until the
    /// probe reaches zero strong references or a deadline passes, recording
    /// the outcome in `released`. An ungated stub (no probe) answers
    /// immediately -- used for the PD prefill leg. `fail_first` makes the
    /// first generate call return UNAVAILABLE, for retry-replay tests; every
    /// call's input token ids are recorded in `seen_input_ids`.
    #[derive(Clone, Default)]
    struct GatedScheduler {
        probe: Option<Weak<CompletionRequest>>,
        gate_rpc: bool,
        released: Arc<AtomicBool>,
        fail_first: bool,
        calls: Arc<std::sync::atomic::AtomicUsize>,
        seen_input_ids: Arc<std::sync::Mutex<Vec<Vec<u32>>>>,
    }

    impl GatedScheduler {
        async fn await_probe(probe: &Weak<CompletionRequest>, released: &AtomicBool) {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
            while probe.strong_count() > 0 && tokio::time::Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            released.store(probe.strong_count() == 0, Ordering::SeqCst);
        }
    }

    fn generate_frames(request_id: &str) -> Vec<Result<ts::GenerateResponse, Status>> {
        use ts::generate_response::Response as GenResp;
        vec![
            Ok(ts::GenerateResponse {
                request_id: request_id.to_string(),
                response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                    token_ids: vec![100],
                    prompt_tokens: 2,
                    completion_tokens: 1,
                    cached_tokens: 0,
                    output_logprobs: None,
                    index: 0,
                })),
            }),
            Ok(ts::GenerateResponse {
                request_id: request_id.to_string(),
                response: Some(GenResp::Complete(ts::GenerateComplete {
                    output_ids: vec![100],
                    finish_reason: "stop".to_string(),
                    prompt_tokens: 2,
                    completion_tokens: 1,
                    cached_tokens: 0,
                    output_logprobs: None,
                    matched_stop: None,
                    index: 0,
                })),
            }),
        ]
    }

    #[tonic::async_trait]
    impl TokenSpeedScheduler for GatedScheduler {
        type GenerateStream = GenStream;
        type SubscribeKvEventsStream = KvEventStream;
        type GetTokenizerStream = TokenizerStream;

        #[expect(
            clippy::disallowed_methods,
            reason = "test stub; the gate task ends at its deadline"
        )]
        async fn generate(
            &self,
            request: TonicRequest<ts::GenerateRequest>,
        ) -> Result<TonicResponse<Self::GenerateStream>, Status> {
            let request = request.into_inner();
            self.seen_input_ids
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(request.tokenized.map(|t| t.input_ids).unwrap_or_default());
            if self.fail_first && self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(Status::unavailable("release-test induced failure"));
            }
            if self.gate_rpc {
                if let Some(probe) = &self.probe {
                    Self::await_probe(probe, &self.released).await;
                }
            }
            let request_id = request.request_id;
            let (tx, rx) = tokio::sync::mpsc::channel(8);
            let probe = (!self.gate_rpc).then(|| self.probe.clone()).flatten();
            let released = Arc::clone(&self.released);
            tokio::spawn(async move {
                if let Some(probe) = probe {
                    Self::await_probe(&probe, &released).await;
                }
                for frame in generate_frames(&request_id) {
                    if tx.send(frame).await.is_err() {
                        return;
                    }
                }
            });
            Ok(TonicResponse::new(Box::pin(ReceiverStream::new(rx))))
        }

        async fn health_check(
            &self,
            _request: TonicRequest<ts::HealthCheckRequest>,
        ) -> Result<TonicResponse<ts::HealthCheckResponse>, Status> {
            Ok(TonicResponse::new(ts::HealthCheckResponse {
                healthy: true,
                message: "ok".to_string(),
            }))
        }

        async fn abort(
            &self,
            _request: TonicRequest<ts::AbortRequest>,
        ) -> Result<TonicResponse<ts::AbortResponse>, Status> {
            Ok(TonicResponse::new(ts::AbortResponse {
                success: true,
                message: String::new(),
            }))
        }

        async fn get_model_info(
            &self,
            _request: TonicRequest<ts::GetModelInfoRequest>,
        ) -> Result<TonicResponse<ts::GetModelInfoResponse>, Status> {
            Ok(TonicResponse::new(ts::GetModelInfoResponse {
                model_path: MODEL.to_string(),
                tokenizer_path: MODEL.to_string(),
                served_model_name: MODEL.to_string(),
                model_type: "mock".to_string(),
                architectures: vec!["MockForCausalLM".to_string()],
                max_context_length: 32768,
                max_req_input_len: 32768,
                vocab_size: 32000,
                eos_token_ids: vec![2],
                pad_token_id: 0,
                bos_token_id: 1,
                weight_version: "mock".to_string(),
                default_sampling_params_json: String::new(),
                supports_vision: false,
                ..Default::default()
            }))
        }

        async fn get_server_info(
            &self,
            _request: TonicRequest<ts::GetServerInfoRequest>,
        ) -> Result<TonicResponse<ts::GetServerInfoResponse>, Status> {
            Ok(TonicResponse::new(ts::GetServerInfoResponse {
                max_total_num_tokens: 1_000_000,
                tokenspeed_version: "mock".to_string(),
                ..Default::default()
            }))
        }

        async fn get_loads(
            &self,
            _request: TonicRequest<ts::GetLoadsRequest>,
        ) -> Result<TonicResponse<ts::GetLoadsResponse>, Status> {
            Err(Status::unimplemented("release-test stub"))
        }

        async fn subscribe_kv_events(
            &self,
            _request: TonicRequest<common::SubscribeKvEventsRequest>,
        ) -> Result<TonicResponse<Self::SubscribeKvEventsStream>, Status> {
            Err(Status::unimplemented("release-test stub"))
        }

        async fn flush_cache(
            &self,
            _request: TonicRequest<common::FlushCacheRequest>,
        ) -> Result<TonicResponse<common::FlushCacheResponse>, Status> {
            Err(Status::unimplemented("release-test stub"))
        }

        async fn start_profile(
            &self,
            _request: TonicRequest<common::StartProfileRequest>,
        ) -> Result<TonicResponse<common::ProfileResponse>, Status> {
            Err(Status::unimplemented("release-test stub"))
        }

        async fn stop_profile(
            &self,
            _request: TonicRequest<common::StopProfileRequest>,
        ) -> Result<TonicResponse<common::ProfileResponse>, Status> {
            Err(Status::unimplemented("release-test stub"))
        }

        async fn get_tokenizer(
            &self,
            _request: TonicRequest<common::GetTokenizerRequest>,
        ) -> Result<TonicResponse<Self::GetTokenizerStream>, Status> {
            Err(Status::unimplemented("release-test stub"))
        }
    }

    #[expect(
        clippy::disallowed_methods,
        reason = "test helper; the stub server lives for the test process"
    )]
    async fn spawn_stub(scheduler: GatedScheduler) -> u16 {
        let port = pick_unused_port().expect("no free port for release-test stub");
        let addr = format!("127.0.0.1:{port}").parse().expect("stub addr");
        tokio::spawn(async move {
            Server::builder()
                .add_service(TokenSpeedSchedulerServer::new(scheduler))
                .serve(addr)
                .await
                .expect("release-test stub server");
        });
        for _ in 0..100 {
            if tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .is_ok()
            {
                return port;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("release-test stub on port {port} never came up");
    }

    fn register_worker(registry: &WorkerRegistry, port: u16, worker_type: WorkerType) {
        let worker = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        registry
            .register(Arc::new(worker))
            .expect("register release-test worker");
    }

    async fn components(worker_registry: Arc<WorkerRegistry>) -> Arc<SharedComponents> {
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
        tokenizer_registry
            .load(
                "tokenizer-id",
                MODEL,
                "test",
                || async move { Ok(tokenizer) },
            )
            .await
            .expect("load mock tokenizer");
        Arc::new(SharedComponents {
            tokenizer_registry,
            worker_registry,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: utils::ParserResolver::disabled(),
            multimodal: None,
        })
    }

    fn completion_request(stream: bool) -> Arc<CompletionRequest> {
        Arc::new(
            serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "prompt": "Hello world",
                "stream": stream,
            }))
            .expect("completion request"),
        )
    }

    fn completion_pipeline(worker_registry: &Arc<WorkerRegistry>, mode: Mode) -> RequestPipeline {
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
        );
        RequestPipeline::build(Endpoint::Completion, mode, &deps).expect("completion pipeline")
    }

    async fn run_and_drain(
        pipeline: RequestPipeline,
        components: Arc<SharedComponents>,
        request: Arc<CompletionRequest>,
        lease: Option<Arc<dyn ErasedLease>>,
    ) -> bytes::Bytes {
        let response = pipeline
            .execute_completion(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                lease,
            )
            .await;
        assert_eq!(response.status(), http::StatusCode::OK);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain SSE body")
    }

    /// The stream task must run off an extracted view: the stub refuses to
    /// emit tokens until the parsed request has been freed, so a stream that
    /// still pinned it would stall past the stub's deadline.
    #[tokio::test]
    async fn streaming_releases_parsed_request_before_first_token() {
        let request = completion_request(true);
        let released = Arc::new(AtomicBool::new(false));
        let port = spawn_stub(GatedScheduler {
            probe: Some(Arc::downgrade(&request)),
            released: Arc::clone(&released),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, port, WorkerType::Regular);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        let body = run_and_drain(pipeline, components, request, None).await;

        assert!(
            released.load(Ordering::SeqCst),
            "the parsed request must be freed before the first token"
        );
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("data: [DONE]"), "stream must finish: {body}");
    }

    /// grpc_pd twin: the decode leg's stream task must not pin the parsed
    /// request either (the prefill stub answers immediately).
    #[tokio::test]
    async fn pd_streaming_releases_parsed_request_before_first_token() {
        let request = completion_request(true);
        let released = Arc::new(AtomicBool::new(false));
        let prefill_port = spawn_stub(GatedScheduler::default()).await;
        let decode_port = spawn_stub(GatedScheduler {
            probe: Some(Arc::downgrade(&request)),
            released: Arc::clone(&released),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
        register_worker(&worker_registry, decode_port, WorkerType::Decode);
        let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
        let components = components(worker_registry).await;

        let body = run_and_drain(pipeline, components, request, None).await;

        assert!(
            released.load(Ordering::SeqCst),
            "the parsed request must be freed before the first decode token"
        );
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("data: [DONE]"), "stream must finish: {body}");
    }

    /// Retries disabled: the parsed request must be freed at dispatch. The
    /// stub refuses to answer the generate RPC until the probe frees.
    #[tokio::test]
    async fn disabled_retries_release_parsed_request_before_upstream_responds() {
        let request = completion_request(false);
        let released = Arc::new(AtomicBool::new(false));
        let port = spawn_stub(GatedScheduler {
            probe: Some(Arc::downgrade(&request)),
            gate_rpc: true,
            released: Arc::clone(&released),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, port, WorkerType::Regular);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        let lease = Arc::new(RequestLease::new(
            request,
            RoutingDerivatives::default(),
            ReleasePoint::AfterDispatch,
        ));
        let attempt = lease.with_view(|view| Arc::clone(view.request));
        let response = pipeline
            .execute_completion(
                attempt,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                Some(lease as Arc<dyn ErasedLease>),
            )
            .await;

        assert_eq!(response.status(), http::StatusCode::OK);
        assert!(
            released.load(Ordering::SeqCst),
            "the parsed request must be freed before the upstream answers"
        );
    }

    /// grpc_pd twin of the dispatch-release probe: both legs' RPCs answer
    /// only after the parsed request is freed.
    #[tokio::test]
    async fn pd_disabled_retries_release_parsed_request_before_upstream_responds() {
        let request = completion_request(false);
        let released = Arc::new(AtomicBool::new(false));
        let prefill_port = spawn_stub(GatedScheduler::default()).await;
        let decode_port = spawn_stub(GatedScheduler {
            probe: Some(Arc::downgrade(&request)),
            gate_rpc: true,
            released: Arc::clone(&released),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
        register_worker(&worker_registry, decode_port, WorkerType::Decode);
        let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
        let components = components(worker_registry).await;

        let lease = Arc::new(RequestLease::new(
            request,
            RoutingDerivatives::default(),
            ReleasePoint::AfterDispatch,
        ));
        let attempt = lease.with_view(|view| Arc::clone(view.request));
        let response = pipeline
            .execute_completion(
                attempt,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                Some(lease as Arc<dyn ErasedLease>),
            )
            .await;

        assert_eq!(response.status(), http::StatusCode::OK);
        assert!(
            released.load(Ordering::SeqCst),
            "the parsed request must be freed before the decode leg answers"
        );
    }

    /// Retries enabled (AtRetryClose): a failed first dispatch must leave the
    /// request intact, the second attempt must send identical token ids, and
    /// the retry-window close (lease drop) frees it.
    #[tokio::test]
    async fn enabled_retries_replay_identical_token_ids() {
        let request = completion_request(false);
        let probe = Arc::downgrade(&request);
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            fail_first: true,
            seen_input_ids: Arc::clone(&seen),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, port, WorkerType::Regular);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        let lease = Arc::new(RequestLease::new(
            request,
            RoutingDerivatives::default(),
            ReleasePoint::AtRetryClose,
        ));

        let attempt = lease.with_view(|view| Arc::clone(view.request));
        let response = pipeline
            .execute_completion(
                attempt,
                None,
                MODEL.to_string(),
                components.clone(),
                None,
                None,
                Some(lease.clone() as Arc<dyn ErasedLease>),
            )
            .await;
        assert!(
            !response.status().is_success(),
            "first dispatch is induced to fail"
        );
        assert!(probe.upgrade().is_some(), "request must survive for replay");

        let attempt = lease.with_view(|view| Arc::clone(view.request));
        let response = pipeline
            .execute_completion(
                attempt,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                Some(lease.clone() as Arc<dyn ErasedLease>),
            )
            .await;
        assert_eq!(response.status(), http::StatusCode::OK);

        {
            let seen = seen.lock().unwrap();
            assert_eq!(seen.len(), 2, "503 then 200 must mean two attempts");
            assert_eq!(
                seen[0], seen[1],
                "the retry must replay identical input ids"
            );
            assert!(
                !seen[0].is_empty(),
                "attempts must carry the tokenized prompt"
            );
        }

        drop(lease);
        assert_eq!(
            probe.strong_count(),
            0,
            "retry-window close must free the request"
        );
    }
}

#[cfg(test)]
mod rate_limit_reserve_tests {
    use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
    use openai_protocol::generate::GenerateRequest;
    use serde_json::json;

    use super::*;
    use crate::{
        config::types::RouterConfig,
        rate_limit::{RateLimitManager, Reservation, ReserveRequest},
        tenant::TenantKey,
    };

    const MODEL: &str = "reserve-test-model";

    async fn test_components() -> Arc<SharedComponents> {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
        tokenizer_registry
            .load(
                "tokenizer-id",
                MODEL,
                "test",
                || async move { Ok(tokenizer) },
            )
            .await
            .unwrap();
        Arc::new(SharedComponents {
            tokenizer_registry,
            worker_registry,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: utils::ParserResolver::disabled(),
            multimodal: None,
        })
    }

    /// Real `RateLimitManager` backed by a tiny in-memory budget --
    /// `RateLimitManager::new` is private to the `rate_limit` module, so
    /// `from_config` (with a temp YAML) is the only way to build one from
    /// here. The YAML is read synchronously inside `from_config`, before the
    /// tempdir is dropped at the end of this function, so no leak is needed.
    fn manager_with_tokens_per_minute(tpm: u32) -> Arc<RateLimitManager> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rate_limit.yaml");
        std::fs::write(
            &path,
            format!("default_policy:\n  tokens_per_minute: {tpm}\n  requests_per_minute: 1000\n"),
        )
        .unwrap();
        let rc = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .tenant_rate_limit_enabled(true)
            .tenant_rate_limit_config(Some(path.to_str().unwrap().to_string()))
            .build_unchecked();
        RateLimitManager::from_config(&rc).unwrap().unwrap()
    }

    fn ctx_with_tokens(components: Arc<SharedComponents>, num_tokens: usize) -> RequestContext {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": MODEL,
            "text": "Hello"
        }))
        .unwrap();
        let mut ctx =
            RequestContext::for_generate(Arc::new(request), None, MODEL.to_string(), components);
        ctx.state.preparation = Some(PreparationOutput::Generate {
            original_text: None,
            token_ids: vec![0; num_tokens],
        });
        ctx.input.tenant_request_meta = Some(TenantRequestMeta::new(TenantKey::new("test-tenant")));
        ctx
    }

    #[tokio::test]
    async fn reserves_exactly_once_across_retry_attempts() {
        // Admits one 5-token reservation but not two (5 + 5 > 6).
        let manager = manager_with_tokens_per_minute(6);
        let components = test_components().await;
        let mut ctx = ctx_with_tokens(components, 5);
        ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

        let stage = RateLimitReserveStage::new(Some(manager));

        // Simulate two retry attempts sharing the same cell, as router.rs does.
        assert!(
            stage.execute(&mut ctx).await.unwrap().is_none(),
            "first attempt should reserve and be admitted"
        );
        assert!(
            stage.execute(&mut ctx).await.unwrap().is_none(),
            "second attempt should see the cached Admitted outcome and skip reserving again"
        );

        assert!(matches!(
            ctx.input.rate_limit_cell.as_ref().unwrap().peek(),
            Some(RateLimitOutcome::Admitted(_))
        ));
    }

    #[tokio::test]
    async fn denial_is_cached_for_should_retry_to_read() {
        // Too small for even one 5-token reservation.
        let manager = manager_with_tokens_per_minute(3);
        let components = test_components().await;
        let mut ctx = ctx_with_tokens(components, 5);
        ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

        let stage = RateLimitReserveStage::new(Some(manager));

        let result = stage.execute(&mut ctx).await;
        let response = result.expect_err("should be denied");
        assert_eq!(response.status(), http::StatusCode::TOO_MANY_REQUESTS);

        // This is the signal router.rs's `should_retry` reads to stop the
        // retry loop instead of retrying a rate-limit denial.
        assert!(matches!(
            ctx.input.rate_limit_cell.as_ref().unwrap().peek(),
            Some(RateLimitOutcome::Denied)
        ));
    }

    #[tokio::test]
    async fn disabled_manager_is_a_no_op() {
        let components = test_components().await;
        let mut ctx = ctx_with_tokens(components, 5);
        ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

        let stage = RateLimitReserveStage::new(None);
        assert!(stage.execute(&mut ctx).await.unwrap().is_none());
        assert!(ctx.input.rate_limit_cell.as_ref().unwrap().peek().is_none());
    }

    #[tokio::test]
    async fn endpoint_without_a_cell_is_a_no_op() {
        // Responses/embeddings/classify don't set rate_limit_cell today.
        let manager = manager_with_tokens_per_minute(6);
        let components = test_components().await;
        let mut ctx = ctx_with_tokens(components, 5);

        let stage = RateLimitReserveStage::new(Some(manager));
        assert!(stage.execute(&mut ctx).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn settle_reservation_with_zero_usage_refunds_the_full_estimate() {
        // Budget 6; reserve 5 leaves 1. A response with no `usage` settles
        // with 0/0 (see the `usage.map_or(0, ...)` call sites in
        // execute_chat/execute_generate/etc.) -- that must refund the full
        // 5-token debit back to the tenant, not silently keep it (nothing
        // was actually consumed) and not leave the reservation stuck open
        // (which would leak budget forever).
        let manager = manager_with_tokens_per_minute(6);
        let components = test_components().await;
        let mut ctx = ctx_with_tokens(components, 5);
        ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

        let stage = RateLimitReserveStage::new(Some(manager.clone()));
        assert!(stage.execute(&mut ctx).await.unwrap().is_none());

        RequestPipeline::settle_reservation(ctx.input.rate_limit_cell.as_deref(), 0, 0).await;

        // The full 6-token budget must be available again -- would be
        // denied if settle had kept the original 5-token debit instead of
        // truing it up to the real (zero) usage.
        let check = ReserveRequest {
            request_charge_id: uuid::Uuid::now_v7(),
            tenant_key: TenantKey::new("test-tenant"),
            model_id: Some(MODEL.to_string()),
            estimated_input_tokens: 6,
        };
        assert!(matches!(
            manager.reserve(check).await,
            Reservation::Admitted(_)
        ));
    }
}
