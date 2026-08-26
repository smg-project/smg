//! Two-phase pipeline orchestrator for gRPC router request processing.
//!
//! Ingress phase (owns the parsed request): preparation → rate-limit reserve
//! → worker selection → client acquisition → encode (EPD) → request building.
//! Request building is the last reader: it yields `(ExecutionPlan,
//! ResponseSpec)` and the request drops at [`RequestContext::into_dispatch`].
//!
//! Dispatch phase (no request, by construction): per-attempt worker
//! re-selection + dispatch of the retained plan, then response processing.
//! The plan is held until the retry window closes (first non-retryable
//! response).

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
use tracing::error;

use super::{
    common::{responses::ResponsesContext, stages::*},
    context::*,
    harmony,
    mode::Mode,
    regular::{
        processor,
        stages::{
            classify::ClassifyResponseProcessingStage,
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
    spec::ResponseSpec,
    utils,
    utils::error_type_from_status,
};
use crate::{
    config::types::RetryConfig,
    middleware::TenantRequestMeta,
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    policies::PolicyRegistry,
    rate_limit::{RateLimitManager, UsageSettlement},
    routers::{
        common::retry::{is_retryable_response, BackoffCalculator},
        error,
    },
    worker::WorkerRegistry,
};

/// Which endpoint a pipeline serves. Selects the endpoint-specific stage set
/// (preparation / request-building / response-processing); `Mode` then selects
/// the disaggregation params within that set.
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

/// The two-phase stage set. Phase ordering is encoded by the struct itself:
/// ingress stages operate on the request-owning [`RequestContext`], the
/// process stage on the request-free [`DispatchContext`].
struct PipelineStages {
    preparation: Box<dyn PipelineStage>,
    /// `None` for endpoints that haven't opted into tenant rate limiting
    /// (embeddings/classify).
    rate_limit: Option<RateLimitReserveStage>,
    worker_selection: WorkerSelectionStage,
    /// EPD-only encode planning/staging.
    encode: Option<EncodeStage>,
    request_building: Box<dyn BuildStage>,
    response_processing: Box<dyn ProcessStage>,
}

/// Generic request pipeline for all request types.
#[derive(Clone)]
pub(crate) struct RequestPipeline {
    stages: Arc<PipelineStages>,
    /// Backend type for metrics labeling
    backend_type: &'static str,
    /// Disaggregation mode, for per-leg retry metric labels.
    mode: Mode,
}

/// Outcome of one full pipeline run.
enum RunOutcome {
    /// Early response (streaming SSE) already produced by response processing.
    Early(Response),
    /// Buffered completion: the final response (or responses-iteration state)
    /// sits on the dispatch context; the Instant is the successful attempt's
    /// start, for duration metrics.
    Final(Box<DispatchContext>, Instant),
}

impl RequestPipeline {
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
        let worker_selection = WorkerSelectionStage::new(
            deps.worker_registry.clone(),
            deps.policy_registry.clone(),
            mode.worker_selection(),
        );
        let plan_kind = mode.plan_kind();
        let inject_pd_metadata = mode.inject_pd_metadata();
        let encode = matches!(mode, Mode::EncodePrefillDecode).then(EncodeStage::new);
        let rate_limit = || RateLimitReserveStage::new(deps.rate_limit_manager.clone());

        let stages = match endpoint {
            Endpoint::Chat => {
                let (processor, streaming_processor) = deps.configured_processors(backend);
                PipelineStages {
                    preparation: Box::new(ChatGeneratePreparationStage::new()),
                    rate_limit: Some(rate_limit()),
                    worker_selection,
                    encode,
                    request_building: Box::new(ChatGenerateRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )),
                    response_processing: Box::new(ChatGenerateResponseProcessingStage::new(
                        processor,
                        streaming_processor,
                    )),
                }
            }
            Endpoint::Messages => {
                let (processor, streaming_processor) = deps.configured_processors(backend);
                PipelineStages {
                    preparation: Box::new(MessagePreparationStage),
                    rate_limit: Some(rate_limit()),
                    worker_selection,
                    encode,
                    request_building: Box::new(MessageRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )),
                    response_processing: Box::new(MessageResponseProcessingStage::new(
                        processor,
                        streaming_processor,
                    )),
                }
            }
            Endpoint::Completion => {
                // Completion uses default parser factories, not the configured ones.
                let (processor, streaming_processor) = PipelineDeps::default_processors(backend);
                PipelineStages {
                    preparation: Box::new(CompletionPreparationStage),
                    rate_limit: Some(rate_limit()),
                    worker_selection,
                    encode,
                    request_building: Box::new(CompletionRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )),
                    response_processing: Box::new(CompletionResponseProcessingStage::new(
                        processor,
                        streaming_processor,
                    )),
                }
            }
            Endpoint::Harmony => {
                // Harmony has no EPD variant.
                if matches!(mode, Mode::EncodePrefillDecode) {
                    return None;
                }
                PipelineStages {
                    preparation: Box::new(harmony::stages::HarmonyPreparationStage::new()),
                    rate_limit: Some(rate_limit()),
                    worker_selection,
                    encode: None,
                    request_building: Box::new(harmony::stages::HarmonyRequestBuildingStage::new(
                        inject_pd_metadata,
                        plan_kind,
                    )),
                    response_processing: Box::new(
                        harmony::stages::HarmonyResponseProcessingStage::new(),
                    ),
                }
            }
            Endpoint::Embeddings => {
                // Embeddings are single-worker only.
                if !matches!(mode, Mode::Regular) {
                    return None;
                }
                PipelineStages {
                    preparation: Box::new(EmbeddingPreparationStage::new()),
                    rate_limit: None,
                    worker_selection,
                    encode: None,
                    request_building: Box::new(EmbeddingRequestBuildingStage::new()),
                    response_processing: Box::new(EmbeddingResponseProcessingStage::new()),
                }
            }
            Endpoint::Classify => {
                // Classify is single-worker only.
                if !matches!(mode, Mode::Regular) {
                    return None;
                }
                PipelineStages {
                    preparation: Box::new(EmbeddingPreparationStage::new()),
                    rate_limit: None,
                    worker_selection,
                    encode: None,
                    request_building: Box::new(EmbeddingRequestBuildingStage::new()),
                    response_processing: Box::new(ClassifyResponseProcessingStage::new()),
                }
            }
        };

        Some(Self {
            stages: Arc::new(stages),
            backend_type: backend,
            mode,
        })
    }

    /// Ingress phase: run the request-owning stages through request building.
    async fn run_ingress(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        macro_rules! step {
            ($stage:expr, $result:expr) => {
                $result.inspect_err(|response: &Response| {
                    error!("Stage {} failed with status {}", $stage, response.status());
                })
            };
        }

        let stages = &self.stages;
        step!(
            stages.preparation.name(),
            stages.preparation.execute(ctx).await
        )?;
        if let Some(rate_limit) = &stages.rate_limit {
            step!(rate_limit.name(), rate_limit.execute(ctx).await)?;
        }
        step!(
            stages.worker_selection.name(),
            stages.worker_selection.execute(ctx).await
        )?;
        let workers = ctx.state.workers.as_ref().ok_or_else(|| {
            error!(function = "run_ingress", "Worker selection not completed");
            error::internal_error(
                "worker_selection_not_completed",
                "Worker selection not completed",
            )
        })?;
        ctx.state.clients = Some(step!(
            "ClientAcquisition",
            acquire_clients(workers, &ctx.input.model_id).await
        )?);
        if let Some(encode) = &stages.encode {
            step!(encode.name(), encode.execute(ctx).await)?;
        }
        step!(
            stages.request_building.name(),
            stages.request_building.build(ctx).await
        )
    }

    /// One dispatch attempt: (re-)selection for retries, plan stamping,
    /// dispatch metadata, execution, response processing.
    async fn run_attempt(
        &self,
        dctx: &mut DispatchContext,
        plan: &mut Option<ExecutionPlan>,
        spec: &ResponseSpec,
        stamp: &AttemptStamp,
        attempt: u32,
        last_attempt: bool,
    ) -> Result<Option<Response>, Response> {
        if attempt > 0 {
            // Fresh worker selection per attempt; the retained plan is
            // re-stamped (engine ids, sampling defaults, PD rooms) for the
            // new workers. Buffered decode state from the failed attempt is
            // reset.
            self.stages.worker_selection.reselect(dctx)?;
            let workers = dctx.workers.as_ref().ok_or_else(|| {
                error!(
                    function = "run_attempt",
                    "Worker re-selection not completed"
                );
                error::internal_error(
                    "worker_selection_not_completed",
                    "Worker selection not completed",
                )
            })?;
            dctx.clients = Some(acquire_clients(workers, &dctx.model_id).await?);
            let retained = plan.as_mut().ok_or_else(|| {
                error!(function = "run_attempt", "Execution plan already consumed");
                error::internal_error("execution_plan_consumed", "Execution plan already consumed")
            })?;
            helpers::restamp_plan_for_attempt(retained, stamp, workers)?;
            if let Some(decoder) = dctx.response.stop_decoder.as_mut() {
                decoder.reset();
            }
            dctx.response.execution_result = None;
        }

        // The last allowed attempt moves the plan (no clone); earlier
        // attempts dispatch a clone and retain the original for replay.
        let attempt_plan = if last_attempt {
            plan.take()
        } else {
            plan.clone()
        }
        .ok_or_else(|| {
            error!(function = "run_attempt", "Execution plan already consumed");
            error::internal_error("execution_plan_consumed", "Execution plan already consumed")
        })?;

        dctx.dispatch = Some(prepare_dispatch_metadata(
            &attempt_plan,
            &dctx.dispatch_model,
            dctx.workers.as_ref(),
        ));

        execute_plan(dctx, attempt_plan).await?;
        self.stages
            .response_processing
            .process(dctx, spec.clone())
            .await
            .inspect_err(|response| {
                error!(
                    "Stage {} failed with status {}",
                    self.stages.response_processing.name(),
                    response.status()
                );
            })
    }

    /// Full two-phase run. `metrics_endpoint` enables per-attempt
    /// router-request/error/duration recording (the responses-internal entry
    /// points pass `None`, matching their historical behavior); `retry_config`
    /// enables the dispatch-phase retry loop. Callers `Box::pin` this future:
    /// it holds ingress + per-attempt state across awaits.
    async fn run(
        &self,
        mut ctx: RequestContext,
        metrics_endpoint: Option<&'static str>,
        retry_config: Option<&RetryConfig>,
    ) -> Result<RunOutcome, Response> {
        let mut attempt_start = Instant::now();
        if let Some(endpoint) = metrics_endpoint {
            Metrics::record_router_request(
                metrics_labels::ROUTER_GRPC,
                self.backend_type,
                metrics_labels::CONNECTION_GRPC,
                &ctx.input.model_id,
                endpoint,
                bool_to_static_str(ctx.is_streaming()),
            );
        }

        let build = match self.run_ingress(&mut ctx).await {
            Ok(build) => build,
            Err(response) => {
                self.record_error(metrics_endpoint, &ctx.input.model_id, &response);
                return Err(response);
            }
        };
        let BuildOutput { plan, spec, stamp } = build;

        // Build boundary: the parsed request drops inside into_dispatch —
        // the dispatch phase has no field that could carry it. The built
        // plan's wire size stands in for the released buffers.
        let model_id = ctx.input.model_id.clone();
        let mut dctx = match ctx.into_dispatch() {
            Ok(dctx) => Box::new(dctx),
            Err(response) => {
                self.record_error(metrics_endpoint, &model_id, &response);
                return Err(response);
            }
        };
        Metrics::record_request_buffers_released_early(plan.wire_len());

        let max_attempts = retry_config.map_or(1, |config| config.max_retries.max(1));
        let mut plan = Some(plan);
        let mut attempt: u32 = 0;
        loop {
            if attempt > 0 {
                attempt_start = Instant::now();
                if let Some(endpoint) = metrics_endpoint {
                    Metrics::record_router_request(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &dctx.model_id,
                        endpoint,
                        bool_to_static_str(dctx.streaming),
                    );
                }
            }

            let last_attempt = attempt + 1 >= max_attempts;
            let failure = match self
                .run_attempt(&mut dctx, &mut plan, &spec, &stamp, attempt, last_attempt)
                .await
            {
                Ok(Some(response)) => {
                    if let Some(endpoint) = metrics_endpoint {
                        Metrics::record_router_duration(
                            metrics_labels::ROUTER_GRPC,
                            self.backend_type,
                            metrics_labels::CONNECTION_GRPC,
                            &dctx.model_id,
                            endpoint,
                            attempt_start.elapsed(),
                        );
                    }
                    return Ok(RunOutcome::Early(response));
                }
                Ok(None) => return Ok(RunOutcome::Final(dctx, attempt_start)),
                Err(response) => response,
            };

            self.record_error(metrics_endpoint, &dctx.model_id, &failure);
            error!(
                attempt,
                status = %failure.status(),
                "pipeline attempt failed"
            );
            // The failed attempt's worker load must not stay elevated through
            // the backoff window (a fresh context dropped them here before).
            dctx.load_guards = None;

            let Some(config) = retry_config else {
                return Err(failure);
            };
            if !is_retryable_response(&failure) {
                return Err(failure);
            }
            if last_attempt {
                if let Some(endpoint) = metrics_endpoint {
                    self.record_retries_exhausted(endpoint);
                }
                return Err(failure);
            }

            let next_attempt = attempt + 1;
            let delay = BackoffCalculator::calculate_delay(config, attempt);
            if let Some(endpoint) = metrics_endpoint {
                self.record_retry(endpoint);
                Metrics::record_worker_retry_backoff(next_attempt, delay);
            }
            tokio::time::sleep(delay).await;
            attempt = next_attempt;
        }
    }

    fn record_error(&self, endpoint: Option<&'static str>, model: &str, response: &Response) {
        if let Some(endpoint) = endpoint {
            Metrics::record_router_error(
                metrics_labels::ROUTER_GRPC,
                self.backend_type,
                metrics_labels::CONNECTION_GRPC,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }
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

    fn record_duration(&self, endpoint: &'static str, model: &str, start: Instant) {
        Metrics::record_router_duration(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            model,
            endpoint,
            start.elapsed(),
        );
    }

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

    /// Execute the complete pipeline for a chat request
    #[expect(clippy::too_many_arguments)]
    pub async fn execute_chat(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
        rate_limit_cell: Option<Arc<RateLimitCell>>,
        retry_config: Option<&RetryConfig>,
    ) -> Response {
        let mut ctx = RequestContext::for_chat(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;
        ctx.input.rate_limit_cell = rate_limit_cell;

        const ENDPOINT: &str = metrics_labels::ENDPOINT_CHAT;
        match Box::pin(self.run(ctx, Some(ENDPOINT), retry_config)).await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Chat(response)) => {
                    let usage = response.usage.as_ref();
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        usage.map_or(0, |u| u.prompt_tokens),
                        usage.map_or(0, |u| u.completion_tokens),
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    axum::Json(response).into_response()
                }
                Some(other) => self.wrong_response_type(
                    "execute_chat",
                    "Chat",
                    &other,
                    &dctx.model_id,
                    ENDPOINT,
                ),
                None => self.no_response_produced("execute_chat", &dctx.model_id, ENDPOINT),
            },
            Err(response) => response,
        }
    }

    /// Execute the complete pipeline for a generate request
    #[expect(clippy::too_many_arguments)]
    pub async fn execute_generate(
        &self,
        request: Arc<GenerateRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
        rate_limit_cell: Option<Arc<RateLimitCell>>,
        retry_config: Option<&RetryConfig>,
    ) -> Response {
        let mut ctx = RequestContext::for_generate(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;
        ctx.input.rate_limit_cell = rate_limit_cell;

        const ENDPOINT: &str = metrics_labels::ENDPOINT_GENERATE;
        match Box::pin(self.run(ctx, Some(ENDPOINT), retry_config)).await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Generate(response)) => {
                    let actual_input_tokens =
                        response.first().map_or(0, |r| r.meta_info.prompt_tokens);
                    let completion_tokens =
                        response.iter().map(|r| r.meta_info.completion_tokens).sum();
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        actual_input_tokens,
                        completion_tokens,
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    axum::Json(response).into_response()
                }
                Some(other) => self.wrong_response_type(
                    "execute_generate",
                    "Generate",
                    &other,
                    &dctx.model_id,
                    ENDPOINT,
                ),
                None => self.no_response_produced("execute_generate", &dctx.model_id, ENDPOINT),
            },
            Err(response) => response,
        }
    }

    /// Execute the complete pipeline for a completion request
    #[expect(clippy::too_many_arguments)]
    pub async fn execute_completion(
        &self,
        request: Arc<CompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
        rate_limit_cell: Option<Arc<RateLimitCell>>,
        retry_config: Option<&RetryConfig>,
    ) -> Response {
        let mut ctx = RequestContext::for_completion(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;
        ctx.input.rate_limit_cell = rate_limit_cell;

        const ENDPOINT: &str = metrics_labels::ENDPOINT_COMPLETIONS;
        match Box::pin(self.run(ctx, Some(ENDPOINT), retry_config)).await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Completion(response)) => {
                    let usage = response.usage.as_ref();
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        usage.map_or(0, |u| u.prompt_tokens),
                        usage.map_or(0, |u| u.completion_tokens),
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    axum::Json(response).into_response()
                }
                Some(other) => self.wrong_response_type(
                    "execute_completion",
                    "Completion",
                    &other,
                    &dctx.model_id,
                    ENDPOINT,
                ),
                None => self.no_response_produced("execute_completion", &dctx.model_id, ENDPOINT),
            },
            Err(response) => response,
        }
    }

    /// Execute the complete pipeline for a Messages API request
    #[expect(clippy::too_many_arguments)]
    pub async fn execute_messages(
        &self,
        request: Arc<CreateMessageRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
        rate_limit_cell: Option<Arc<RateLimitCell>>,
        retry_config: Option<&RetryConfig>,
    ) -> Response {
        let mut ctx = RequestContext::for_messages(request, headers, model_id, components);
        ctx.input.tenant_request_meta = tenant_request_meta;
        ctx.input.rate_limit_cell = rate_limit_cell;

        const ENDPOINT: &str = metrics_labels::ENDPOINT_MESSAGES;
        match Box::pin(self.run(ctx, Some(ENDPOINT), retry_config)).await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Messages(response)) => {
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    axum::Json(response).into_response()
                }
                Some(other) => self.wrong_response_type(
                    "execute_messages",
                    "Messages",
                    &other,
                    &dctx.model_id,
                    ENDPOINT,
                ),
                None => self.no_response_produced("execute_messages", &dctx.model_id, ENDPOINT),
            },
            Err(response) => response,
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

        const ENDPOINT: &str = metrics_labels::ENDPOINT_EMBEDDINGS;
        match Box::pin(self.run(ctx, Some(ENDPOINT), None)).await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Embedding(response)) => {
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    axum::Json(response).into_response()
                }
                Some(other) => self.wrong_response_type(
                    "execute_embeddings",
                    "Embedding",
                    &other,
                    &dctx.model_id,
                    ENDPOINT,
                ),
                None => self.no_response_produced("execute_embeddings", &dctx.model_id, ENDPOINT),
            },
            Err(response) => response,
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

        const ENDPOINT: &str = metrics_labels::ENDPOINT_CLASSIFY;
        match Box::pin(self.run(ctx, Some(ENDPOINT), None)).await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Classify(response)) => {
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    axum::Json(response).into_response()
                }
                Some(other) => self.wrong_response_type(
                    "execute_classify",
                    "Classify",
                    &other,
                    &dctx.model_id,
                    ENDPOINT,
                ),
                None => self.no_response_produced("execute_classify", &dctx.model_id, ENDPOINT),
            },
            Err(response) => response,
        }
    }

    /// Execute chat pipeline for responses endpoint
    ///
    /// Used by ALL non-streaming /v1/responses requests.
    /// Same stages as execute_chat(), with two differences:
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

        match Box::pin(self.run(ctx, None, None)).await {
            Ok(RunOutcome::Early(_)) => {
                // Streaming not supported for responses sync mode
                error!(
                    function = "execute_chat_for_responses",
                    "Streaming attempted in responses context"
                );
                Err(error::bad_request(
                    "streaming_not_supported",
                    "Streaming is not supported in this context".to_string(),
                ))
            }
            Ok(RunOutcome::Final(mut dctx, _)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Chat(response)) => Ok(response),
                Some(_) => {
                    error!(
                        function = "execute_chat_for_responses",
                        "Wrong response type: expected Chat"
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
            },
            Err(response) => Err(response),
        }
    }

    /// Execute one Harmony Responses API iteration through the pipeline.
    ///
    /// Returns ToolCallsFound (continue serving) or Completed (final
    /// response); called by harmony::responses::serve_harmony_responses()
    /// per iteration.
    pub async fn execute_harmony_responses(
        &self,
        request: &openai_protocol::responses::ResponsesRequest,
        harmony_ctx: &ResponsesContext,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Result<harmony::ResponsesIterationResult, Response> {
        let mut ctx = RequestContext::for_responses(
            Arc::new(request.clone()),
            None,                  // No headers needed for internal pipeline execution
            request.model.clone(), // Model ID from request
            harmony_ctx.components.clone(),
        );
        ctx.input.tenant_request_meta = tenant_request_meta;

        let mut dctx = match Box::pin(self.run(ctx, None, None)).await {
            Ok(RunOutcome::Early(response)) => {
                error!(
                    function = "execute_harmony_responses",
                    "Unexpected early response during Responses iteration"
                );
                return Err(response);
            }
            Ok(RunOutcome::Final(dctx, _)) => dctx,
            Err(response) => return Err(response),
        };

        dctx.response
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

    /// Execute a Harmony Responses pipeline iteration with streaming support.
    ///
    /// Runs through dispatch and returns the raw ExecutionResult (with
    /// stream) and LoadGuards for token-level streaming processing. The
    /// caller keeps load_guards alive until stream processing completes.
    pub async fn execute_harmony_responses_streaming(
        &self,
        request: &openai_protocol::responses::ResponsesRequest,
        harmony_ctx: &ResponsesContext,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Result<(ExecutionResult, Option<LoadGuards>), Response> {
        let mut ctx = RequestContext::for_responses(
            Arc::new(request.clone()),
            None,
            request.model.clone(),
            harmony_ctx.components.clone(),
        );
        ctx.input.tenant_request_meta = tenant_request_meta;

        let mut dctx = match Box::pin(self.run(ctx, None, None)).await {
            Ok(RunOutcome::Early(response)) => {
                error!(
                    function = "execute_harmony_responses_streaming",
                    "Unexpected early response during streaming Responses"
                );
                return Err(response);
            }
            Ok(RunOutcome::Final(dctx, _)) => dctx,
            Err(response) => return Err(response),
        };

        let execution_result = dctx.response.execution_result.take().ok_or_else(|| {
            error!(
                function = "execute_harmony_responses_streaming",
                "No ExecutionResult produced by pipeline"
            );
            error::internal_error(
                "no_execution_result_produced",
                "No ExecutionResult produced by pipeline",
            )
        })?;

        let load_guards = dctx.load_guards.take();

        Ok((execution_result, load_guards))
    }
}

#[cfg(test)]
impl RequestPipeline {
    /// Mode-bearing construction descriptor for the parity test. Stage
    /// ordering itself is structural (see `PipelineStages`), so only the
    /// per-mode args and optional stages need freezing.
    fn signature(&self) -> String {
        format!(
            "{}|rate_limit={}|encode={}|{}",
            self.stages.worker_selection.signature(),
            self.stages.rate_limit.is_some(),
            self.stages.encode.is_some(),
            self.stages.request_building.signature(),
        )
    }
}

#[cfg(test)]
mod build_parity_tests {
    use super::*;
    use crate::routers::grpc::mode::Mode;

    /// Hand-transcribed expected construction descriptor + metrics backend
    /// label per endpoint/mode. Do not regenerate from `build()` output, or
    /// this stops guarding anything: it exists to catch a wrong
    /// `build`/`mode.rs` mapping (a flipped `inject_pd_metadata`, wrong
    /// `plan_kind`, or swapped `WorkerSelectionMode`). Stage ordering is
    /// structural in `PipelineStages` and needs no golden.
    fn golden(endpoint: Endpoint, mode: Mode) -> (String, &'static str) {
        const REGULAR: &str = metrics_labels::BACKEND_REGULAR;
        const PD: &str = metrics_labels::BACKEND_PD;
        let (selection, inject, plan, backend) = match mode {
            Mode::Regular => ("Regular", "false", "Single", REGULAR),
            Mode::PrefillDecode => ("PrefillDecode", "true", "PrefillDecode", PD),
            Mode::EncodePrefillDecode => {
                ("EncodePrefillDecode", "false", "EncodePrefillDecode", PD)
            }
        };
        let encode = matches!(mode, Mode::EncodePrefillDecode);
        let build = match endpoint {
            Endpoint::Chat => format!(
                "ChatGenerateRequestBuildingStage(ChatRequestBuildingStage(inject_pd_metadata={inject}, {plan}), GenerateRequestBuildingStage(inject_pd_metadata={inject}, {plan}))"
            ),
            Endpoint::Messages => {
                format!("MessageRequestBuildingStage(inject_pd_metadata={inject}, {plan})")
            }
            Endpoint::Completion => {
                format!("CompletionRequestBuildingStage(inject_pd_metadata={inject}, {plan})")
            }
            Endpoint::Harmony => {
                format!("HarmonyRequestBuildingStage(inject_pd_metadata={inject}, {plan})")
            }
            Endpoint::Embeddings | Endpoint::Classify => {
                "EmbeddingRequestBuildingStage".to_string()
            }
        };
        let rate_limit = !matches!(endpoint, Endpoint::Embeddings | Endpoint::Classify);
        // Harmony never carries the encode stage.
        let encode = encode && !matches!(endpoint, Endpoint::Harmony);
        (
            format!(
                "WorkerSelectionStage({selection})|rate_limit={rate_limit}|encode={encode}|{build}"
            ),
            backend,
        )
    }

    fn assert_parity(endpoint: Endpoint, mode: Mode, deps: &PipelineDeps) {
        let (expected_sig, expected_backend) = golden(endpoint, mode);
        let built = RequestPipeline::build(endpoint, mode, deps)
            .unwrap_or_else(|| panic!("build({endpoint:?}, {mode:?}) should be valid"));
        assert_eq!(
            built.signature(),
            expected_sig,
            "construction parity for {endpoint:?}/{mode:?}"
        );
        assert_eq!(
            built.backend_type, expected_backend,
            "backend_type parity for {endpoint:?}/{mode:?}"
        );
        assert_eq!(built.mode, mode, "mode parity for {endpoint:?}/{mode:?}");
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

        // Ingress up to worker selection: the canonical id must already be in
        // place when selection runs.
        pipeline.stages.preparation.execute(&mut ctx).await.unwrap();
        pipeline
            .stages
            .worker_selection
            .execute(&mut ctx)
            .await
            .unwrap();

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
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Mutex, PoisonError, Weak,
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
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use tonic::{transport::Server, Request as TonicRequest, Response as TonicResponse, Status};
    use ts::token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer};

    use super::*;
    use crate::{
        config::types::PolicyConfig,
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
    /// call's input token ids and engine request id are recorded.
    #[derive(Clone, Default)]
    struct GatedScheduler {
        probe: Option<Weak<CompletionRequest>>,
        gate_rpc: bool,
        released: Arc<AtomicBool>,
        fail_first: bool,
        calls: Arc<AtomicUsize>,
        seen_input_ids: Arc<Mutex<Vec<Vec<u32>>>>,
        seen_request_ids: Arc<Mutex<Vec<String>>>,
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
                .unwrap_or_else(PoisonError::into_inner)
                .push(request.tokenized.map(|t| t.input_ids).unwrap_or_default());
            self.seen_request_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(request.request_id.clone());
            if self.fail_first && self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(Status::unavailable("release-test induced failure"));
            }
            if self.gate_rpc {
                if let Some(probe) = &self.probe {
                    Self::await_probe(probe, &self.released).await;
                }
            }
            let request_id = request.request_id;
            let (tx, rx) = mpsc::channel(8);
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

    fn fast_retry_config(max_retries: u32) -> RetryConfig {
        RetryConfig {
            max_retries,
            initial_backoff_ms: 1,
            max_backoff_ms: 2,
            backoff_multiplier: 1.0,
            jitter_factor: 0.0,
        }
    }

    async fn run_and_drain(
        pipeline: RequestPipeline,
        components: Arc<SharedComponents>,
        request: Arc<CompletionRequest>,
    ) -> bytes::Bytes {
        let response = pipeline
            .execute_completion(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
            )
            .await;
        assert_eq!(response.status(), http::StatusCode::OK);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain SSE body")
    }

    /// The stream task must run off the response spec: the stub refuses to
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

        let body = run_and_drain(pipeline, components, request).await;

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

        let body = run_and_drain(pipeline, components, request).await;

        assert!(
            released.load(Ordering::SeqCst),
            "the parsed request must be freed before the first decode token"
        );
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("data: [DONE]"), "stream must finish: {body}");
    }

    /// The parsed request dies at the build boundary even with the retry
    /// window open: only the execution plan is retained for replay. The stub
    /// refuses to answer the generate RPC until the probe frees.
    #[tokio::test]
    async fn buffered_dispatch_releases_parsed_request_even_with_retries_enabled() {
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

        let retry = fast_retry_config(3);
        let response = pipeline
            .execute_completion(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                Some(&retry),
            )
            .await;

        assert_eq!(response.status(), http::StatusCode::OK);
        assert!(
            released.load(Ordering::SeqCst),
            "the parsed request must be freed before the upstream answers"
        );
    }

    /// grpc_pd twin of the dispatch-release probe: the gated decode leg
    /// answers only after the parsed request is freed.
    #[tokio::test]
    async fn pd_buffered_dispatch_releases_parsed_request_before_upstream_responds() {
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

        let response = pipeline
            .execute_completion(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
            )
            .await;

        assert_eq!(response.status(), http::StatusCode::OK);
        assert!(
            released.load(Ordering::SeqCst),
            "the parsed request must be freed before the decode leg answers"
        );
    }

    /// A failed first dispatch retries from the retained plan: the second
    /// attempt must send identical token ids without re-tokenizing, under a
    /// fresh per-attempt engine id, and the parsed request must already be
    /// gone when the first attempt is sent.
    #[tokio::test]
    async fn retry_replays_identical_token_ids_from_the_retained_plan() {
        let request = completion_request(false);
        let probe = Arc::downgrade(&request);
        let seen_ids = Arc::new(Mutex::new(Vec::new()));
        let seen_request_ids = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            fail_first: true,
            seen_input_ids: Arc::clone(&seen_ids),
            seen_request_ids: Arc::clone(&seen_request_ids),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, port, WorkerType::Regular);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        let retry = fast_retry_config(2);
        let response = pipeline
            .execute_completion(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                Some(&retry),
            )
            .await;
        assert_eq!(response.status(), http::StatusCode::OK);
        assert_eq!(
            probe.strong_count(),
            0,
            "request freed at the build boundary"
        );

        {
            let seen = seen_ids.lock().unwrap_or_else(PoisonError::into_inner);
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
        {
            let ids = seen_request_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            assert_eq!(ids.len(), 2);
            assert!(ids[0].starts_with("cmpl_") && ids[1].starts_with("cmpl_"));
            assert_ne!(ids[0], ids[1], "each attempt gets a fresh engine id");
        }
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
    async fn reserve_is_admitted_once_and_cached_on_the_cell() {
        // Admits one 5-token reservation but not two (5 + 5 > 6).
        let manager = manager_with_tokens_per_minute(6);
        let components = test_components().await;
        let mut ctx = ctx_with_tokens(components, 5);
        ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

        let stage = RateLimitReserveStage::new(Some(manager));

        assert!(stage.execute(&mut ctx).await.is_ok());
        // A second pass (defensive; ingress runs once) sees the cached
        // Admitted outcome and does not reserve again.
        assert!(stage.execute(&mut ctx).await.is_ok());

        assert!(matches!(
            ctx.input.rate_limit_cell.as_ref().unwrap().peek(),
            Some(RateLimitOutcome::Admitted(_))
        ));
    }

    #[tokio::test]
    async fn denial_is_cached_on_the_cell() {
        // Too small for even one 5-token reservation.
        let manager = manager_with_tokens_per_minute(3);
        let components = test_components().await;
        let mut ctx = ctx_with_tokens(components, 5);
        ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

        let stage = RateLimitReserveStage::new(Some(manager));

        let response = stage.execute(&mut ctx).await.expect_err("should be denied");
        assert_eq!(response.status(), http::StatusCode::TOO_MANY_REQUESTS);

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
        assert!(stage.execute(&mut ctx).await.is_ok());
        assert!(ctx.input.rate_limit_cell.as_ref().unwrap().peek().is_none());
    }

    #[tokio::test]
    async fn endpoint_without_a_cell_is_a_no_op() {
        // Responses/embeddings/classify don't set rate_limit_cell today.
        let manager = manager_with_tokens_per_minute(6);
        let components = test_components().await;
        let mut ctx = ctx_with_tokens(components, 5);

        let stage = RateLimitReserveStage::new(Some(manager));
        assert!(stage.execute(&mut ctx).await.is_ok());
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
        assert!(stage.execute(&mut ctx).await.is_ok());

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
