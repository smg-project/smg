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

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::response::{IntoResponse, Response};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    classify::ClassifyRequest,
    common::{InputIds, Usage},
    completion::CompletionRequest,
    decisions::{DecisionUsage, DecisionsRequest},
    embedding::EmbeddingRequest,
    generate::{GenerateRequest, GenerateResponse},
    messages::CreateMessageRequest,
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use smg_external_router::header_utils::insert_routed_worker_id;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{debug, error};

use super::{
    common::{
        response_formatting::TokenCounts,
        responses::{ResponsesContext, StreamStartupSender},
        stages::*,
    },
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
            decisions::{
                DecisionsPreparationStage, DecisionsRequestBuildingStage,
                DecisionsResponseProcessingStage,
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
            ChatGenerateResponseProcessingStage, TranscriptionPreparationStage,
            TranscriptionRequestBuildingStage, TranscriptionResponseProcessingStage,
        },
        streaming,
    },
    spec::ResponseSpec,
    utils,
    utils::{error_reason_for_response, error_type_for_response},
};
use crate::{
    config::types::RetryConfig,
    middleware::{ReportedRefusal, TenantRequestMeta},
    observability::{
        cache_trace,
        metrics::{bool_to_static_str, metrics_labels, Metrics},
    },
    policies::PolicyRegistry,
    rate_limit::{RateLimitManager, UsageSettlement},
    routers::{
        common::{
            placement::{self, Candidates, WireConstraint},
            retry::{is_retryable_response, BackoffCalculator},
        },
        error,
    },
    worker::{PrefillAdmission, RoutingPool, WorkerRegistry},
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
    Decisions,
    Transcription,
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
    prefill_admission: Option<Arc<PrefillAdmission>>,
    tool_parser_factory: ToolParserFactory,
    reasoning_parser_factory: ReasoningParserFactory,
    configured_tool_parser: Option<String>,
    configured_reasoning_parser: Option<String>,
    /// `None` when tenant rate limiting is disabled; read by the
    /// `RateLimitReserveStage` of every endpoint.
    rate_limit_manager: Option<Arc<RateLimitManager>>,
    /// `--queue-timeout-secs`: a request still ahead of worker selection this
    /// long after the gateway accepted it is refused instead of dispatched.
    queue_timeout: Duration,
}

impl PipelineDeps {
    /// Full deps for the chat/messages/harmony endpoints, which consume the
    /// configured parser factories/overrides.
    #[expect(
        clippy::too_many_arguments,
        reason = "pipeline construction wires the endpoint dependency bundle"
    )]
    pub(crate) fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        prefill_admission: Option<Arc<PrefillAdmission>>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
        rate_limit_manager: Option<Arc<RateLimitManager>>,
        queue_timeout: Duration,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            prefill_admission,
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
            rate_limit_manager,
            queue_timeout,
        }
    }

    /// Deps for endpoints (embeddings/classify/completion) with no configured
    /// parsers; the parser fields are placeholders those endpoints never read.
    pub(crate) fn pair(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        prefill_admission: Option<Arc<PrefillAdmission>>,
        rate_limit_manager: Option<Arc<RateLimitManager>>,
        queue_timeout: Duration,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            prefill_admission,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            configured_tool_parser: None,
            configured_reasoning_parser: None,
            rate_limit_manager,
            queue_timeout,
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
            &self.tool_parser_factory,
            &self.reasoning_parser_factory,
        );
        let processor = processor::ResponseProcessor::new(
            self.tool_parser_factory.clone(),
            self.reasoning_parser_factory.clone(),
            parser_resolver.clone(),
        )
        .with_backend(backend);
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
        )
        .with_backend(backend);
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
            prefill_admission: None,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            configured_tool_parser: None,
            configured_reasoning_parser: None,
            rate_limit_manager: None,
            queue_timeout: Duration::from_secs(60),
        }
    }
}

/// The two-phase stage set. Phase ordering is encoded by the struct itself:
/// ingress stages operate on the request-owning [`RequestContext`], the
/// process stage on the request-free [`DispatchContext`].
struct PipelineStages {
    preparation: Box<dyn PipelineStage>,
    /// The tenant's reservation, taken once per logical request by every
    /// endpoint (a no-op while tenant rate limiting is off).
    rate_limit: RateLimitReserveStage,
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
    /// Disaggregation mode, for per-side retry metric labels.
    mode: Mode,
    /// The registry, read at dispatch for `--worker-overload-shed`.
    worker_registry: Arc<WorkerRegistry>,
    /// Shapes a forced tool call's constraint for each attempt's worker.
    tool_parser_factory: ToolParserFactory,
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

/// One full pipeline run: the one box of a run, allocated where the request
/// context moves in (see [`RequestPipeline::run`]).
type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<RunOutcome, Response>> + Send + 'a>>;

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
            deps.prefill_admission.clone(),
        )
        .with_queue_timeout(deps.queue_timeout);
        let plan_kind = mode.plan_kind();
        let inject_pd_metadata = mode.inject_pd_metadata();
        let encode = matches!(mode, Mode::EncodePrefillDecode).then(EncodeStage::new);
        let rate_limit = RateLimitReserveStage::new(deps.rate_limit_manager.clone());

        let stages = match endpoint {
            Endpoint::Decisions => {
                if !matches!(mode, Mode::Regular) {
                    return None;
                }
                PipelineStages {
                    preparation: Box::new(DecisionsPreparationStage),
                    rate_limit,
                    worker_selection,
                    encode: None,
                    request_building: Box::new(DecisionsRequestBuildingStage),
                    response_processing: Box::new(DecisionsResponseProcessingStage),
                }
            }
            Endpoint::Chat => {
                let (processor, streaming_processor) = deps.configured_processors(backend);
                PipelineStages {
                    preparation: Box::new(ChatGeneratePreparationStage::new(
                        mode.multimodal_wire_pools(),
                    )),
                    rate_limit,
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
                    preparation: Box::new(MessagePreparationStage::new(
                        mode.multimodal_wire_pools(),
                    )),
                    rate_limit,
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
                    rate_limit,
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
                    rate_limit,
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
                    rate_limit,
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
                    rate_limit,
                    worker_selection,
                    encode: None,
                    request_building: Box::new(EmbeddingRequestBuildingStage::new()),
                    response_processing: Box::new(ClassifyResponseProcessingStage::new()),
                }
            }
            Endpoint::Transcription => {
                // Transcription is Regular-only (whole-file, single worker).
                if !matches!(mode, Mode::Regular) {
                    return None;
                }
                // Plain text decode: no configured parsers needed.
                let (processor, _streaming) = PipelineDeps::default_processors(backend);
                PipelineStages {
                    preparation: Box::new(TranscriptionPreparationStage),
                    rate_limit,
                    worker_selection,
                    encode: None,
                    // Regular-only: no PD metadata, single-plan (see the stage).
                    request_building: Box::new(TranscriptionRequestBuildingStage::new()),
                    response_processing: Box::new(TranscriptionResponseProcessingStage::new(
                        processor,
                    )),
                }
            }
        };

        Some(Self {
            stages: Arc::new(stages),
            backend_type: backend,
            mode,
            worker_registry: deps.worker_registry.clone(),
            tool_parser_factory: deps.tool_parser_factory.clone(),
        })
    }

    /// Ingress phase: run the request-owning stages through request building.
    async fn run_ingress(&self, ctx: &mut RequestContext) -> Result<BuildOutput, Response> {
        macro_rules! step {
            ($stage:expr, $result:expr) => {
                $result.map_err(|response: Response| {
                    // A refusal the failing stage did not report itself (the
                    // preparation stage's 404 for a model nobody serves)
                    // gets its one rate-limited line here; a reported one,
                    // and every other failure, passes through.
                    let response = self.stages.worker_selection.report_refusal(
                        "RequestPipeline::run_ingress",
                        helpers::middleware_request_id(ctx.input.tenant_request_meta.as_ref()),
                        &ctx.input.model_id,
                        response,
                    );
                    if response.extensions().get::<ReportedRefusal>().is_some() {
                        // The one line this refusal gets was written
                        // (rate-limited, with the pool's state).
                        debug!(
                            "Stage {} refused the request with status {}",
                            $stage,
                            response.status()
                        );
                    } else {
                        error!("Stage {} failed with status {}", $stage, response.status());
                    }
                    response
                })
            };
        }

        let stages = &self.stages;
        step!(
            stages.preparation.name(),
            stages.preparation.execute(ctx).await
        )?;
        step!(
            stages.rate_limit.name(),
            stages.rate_limit.execute(ctx).await
        )?;
        step!(
            stages.worker_selection.name(),
            stages.worker_selection.execute(ctx).await
        )?;
        let workers = stage_output(ctx.state.workers.as_ref(), "run_ingress")?;
        // Keep the post-selection guard for paths without pre-filtering
        // (EPD and direct ZMQ), and for a model card updated during selection.
        // No rejected request acquires a client or reaches an engine (#2380).
        let prep = stage_output(ctx.state.preparation.as_ref(), "run_ingress")?;
        step!(
            "ContextLength",
            enforce_context_length(prep, workers, &ctx.input.model_id)
        )?;
        step!(
            "OutputBudget",
            enforce_output_budget(&ctx.input.request_type, workers, &ctx.input.model_id)
        )?;
        ctx.state.clients = Some(step!(
            "ClientAcquisition",
            acquire_clients(
                workers,
                &ctx.input.model_id,
                self.worker_registry.overload_shed_enabled(),
            )
            .await
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
        spec: &mut Option<ResponseSpec>,
        stamp: &AttemptStamp,
        attempt: u32,
        last_attempt: bool,
    ) -> Result<Option<Response>, Response> {
        dctx.attempt = attempt;
        let consumed = || {
            error!(function = "run_attempt", "Execution plan already consumed");
            error::internal_error("execution_plan_consumed", "Execution plan already consumed")
        };
        if attempt > 0 {
            // Fresh worker selection per attempt; the retained plan is
            // re-stamped (engine ids, sampling defaults, the forced tool
            // call's shape, PD rooms) for the new workers. Buffered decode
            // state from the failed attempt is reset.
            self.stages.worker_selection.reselect(dctx).await?;
            let workers = stage_output(dctx.workers.as_ref(), "run_attempt")?;
            dctx.clients = Some(
                acquire_clients(
                    workers,
                    &dctx.model_id,
                    self.worker_registry.overload_shed_enabled(),
                )
                .await?,
            );
            let retained = plan.as_mut().ok_or_else(consumed)?;
            helpers::restamp_plan_for_attempt(retained, stamp, workers, &self.tool_parser_factory)?;
            if let Some(decoder) = dctx.response.stop_decoder.as_mut() {
                decoder.reset();
            }
            dctx.response.execution_result = None;
        }

        if let Some(ResponseSpec::Decisions(spec)) = spec {
            if let Some(workers) = dctx.workers.as_ref() {
                enforce_decisions_context_length(spec.max_input_tokens, workers, &dctx.model_id)?;
            }
        }

        // The last allowed attempt moves the plan and the spec (no clone);
        // earlier attempts dispatch a clone and retain the originals for replay.
        let (attempt_plan, attempt_spec) = if last_attempt {
            plan.take().zip(spec.take())
        } else {
            plan.clone().zip(spec.clone())
        }
        .ok_or_else(consumed)?;

        dctx.dispatch = Some(prepare_dispatch_metadata(
            &attempt_plan,
            &dctx.dispatch_model,
            dctx.workers.as_ref(),
        ));

        execute_plan(dctx, attempt_plan, last_attempt).await?;
        self.stages
            .response_processing
            .process(dctx, attempt_spec)
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
    /// enables the dispatch-phase retry loop.
    ///
    /// The run is boxed once, here, and the context moves straight into that
    /// box: it holds ingress + per-attempt state across awaits, and an
    /// `async fn` hop in between would keep its own copy of the 2.6 KiB
    /// context in every caller's state machine (an argument lives in the
    /// future as the upvar and as the rebound local).
    fn run<'a>(
        &'a self,
        ctx: RequestContext,
        metrics_endpoint: Option<&'static str>,
        retry_config: Option<&'a RetryConfig>,
    ) -> RunFuture<'a> {
        let requested = cache_trace::requested(ctx.input.headers.as_ref());
        if !cache_trace::enabled() && !requested {
            return Box::pin(self.run_inner(ctx, metrics_endpoint, retry_config));
        }
        let root_id = helpers::middleware_request_id(ctx.input.tenant_request_meta.as_ref())
            .map(str::to_owned);
        Box::pin(cache_trace::scope(requested, async move {
            let result = self.run_inner(ctx, metrics_endpoint, retry_config).await;
            if let Err(response) = &result {
                cache_trace::failure(root_id.as_deref(), response.status().as_u16());
            }
            result
        }))
    }

    /// The run itself, as the future [`Self::run`] boxes: an `async move`
    /// block over the context it was handed, so the context has one place in
    /// that future (an `async fn` would rebind the argument into a second).
    #[expect(
        clippy::manual_async_fn,
        reason = "an async fn keeps a by-value argument twice in its state machine (the captured argument and the rebound local); the async move block holds the 2.6 KiB context once"
    )]
    fn run_inner<'a>(
        &'a self,
        mut ctx: RequestContext,
        metrics_endpoint: Option<&'static str>,
        retry_config: Option<&'a RetryConfig>,
    ) -> impl Future<Output = Result<RunOutcome, Response>> + Send + 'a {
        async move {
            let mut attempt_start = Instant::now();
            if let Some(endpoint) = metrics_endpoint {
                self.record_request(endpoint, &ctx.input.model_id, ctx.is_streaming());
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
            let mut spec = Some(spec);
            let mut attempt: u32 = 0;
            loop {
                if attempt > 0 {
                    attempt_start = Instant::now();
                    if let Some(endpoint) = metrics_endpoint {
                        self.record_request(endpoint, &dctx.model_id, dctx.streaming);
                    }
                }

                let last_attempt = attempt + 1 >= max_attempts;
                let mut failure = match self
                    .run_attempt(
                        &mut dctx,
                        &mut plan,
                        &mut spec,
                        &stamp,
                        attempt,
                        last_attempt,
                    )
                    .await
                {
                    Ok(Some(response)) => {
                        if let Some(endpoint) = metrics_endpoint {
                            self.record_duration(endpoint, &dctx.model_id, attempt_start);
                        }
                        return Ok(RunOutcome::Early(Self::routed_response(&dctx, response)));
                    }
                    Ok(None) => return Ok(RunOutcome::Final(dctx, attempt_start)),
                    Err(response) => response,
                };

                self.record_error(metrics_endpoint, &dctx.model_id, &failure);
                if failure.extensions().get::<ReportedRefusal>().is_some() {
                    // The re-selection wrote this refusal's line (rate-limited,
                    // with the pool's state), as for the first selection.
                    debug!(
                        attempt,
                        status = %failure.status(),
                        "pipeline attempt refused"
                    );
                } else {
                    error!(
                        attempt,
                        status = %failure.status(),
                        "pipeline attempt failed"
                    );
                }
                // The failed attempt's worker load must not stay elevated through
                // the backoff window (a fresh context dropped them here before).
                // This releases its PD admission claim too, so the retry is not
                // queued behind its own predecessor's bootstrap rooms.
                dctx.load_guards = None;
                dctx.pd_prefill_guard = None;
                // A worker's definitive answer closes the retry window on that
                // worker: the next attempt goes to another available worker, or
                // the answer stands as the client's response.
                if let Some(workers) = dctx.workers.as_ref() {
                    // The alternative is sought where the retry would re-select,
                    // among the workers that decide the constraint alike too.
                    let pools = self.retry_pools(&dctx.model_id, dctx.wire);
                    let decided_on = dctx.constraint_decided_on;
                    dctx.ledger.settle(
                        &mut failure,
                        workers.workers(),
                        pools
                            .iter()
                            .flat_map(|pool| pool.as_slice().iter())
                            .filter(|worker| {
                                decided_on
                                    .is_none_or(|decoding| decoding.decides_alike(worker.as_ref()))
                                    && dctx
                                        .routing
                                        .context_requirements
                                        .admits(worker.as_ref(), &dctx.model_id)
                            }),
                    );
                }

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

                // The retry re-selects away from the engines this attempt used,
                // while another available worker is left to take the request.
                let engines: Vec<String> = dctx
                    .workers
                    .iter()
                    .flat_map(WorkerSelection::engines)
                    .map(str::to_string)
                    .collect();
                for engine in engines {
                    if !dctx.tried.contains(&engine) {
                        dctx.tried.push(engine);
                    }
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
    }

    /// The pools a retry of `model_id` re-selects from, narrowed to the
    /// retained wire as the re-selection itself is.
    fn retry_pools(&self, model_id: &str, wire: WireConstraint) -> Vec<Candidates> {
        let pools: &[RoutingPool] = match self.mode {
            Mode::Regular => &[RoutingPool::GrpcPipelineRegular],
            Mode::PrefillDecode | Mode::EncodePrefillDecode => {
                &[RoutingPool::GrpcPrefill, RoutingPool::GrpcDecode]
            }
        };
        pools
            .iter()
            .map(|pool| placement::candidates(&self.worker_registry, model_id, *pool, Some(wire)))
            .collect()
    }

    /// Attribute a response to the worker from the successful dispatch attempt.
    fn routed_response(ctx: &DispatchContext, mut response: Response) -> Response {
        if let Some(trace) = ctx.cache_trace.as_ref() {
            if let Ok(value) = http::HeaderValue::from_str(trace) {
                response.headers_mut().insert("x-smg-cache-trace", value);
            }
        }
        // Which parser read a chat completion, so a client comparing the two can tell.
        if let Some(parser) = ctx.parser {
            response.headers_mut().insert(
                "x-smg-parser",
                http::HeaderValue::from_static(parser.name()),
            );
        }
        if let Some(workers) = ctx.workers.as_ref() {
            let worker = match workers {
                WorkerSelection::Single { worker } => worker,
                WorkerSelection::Disaggregated { decode, .. } => decode,
            };
            insert_routed_worker_id(response.headers_mut(), worker.redacted_url());
        }
        response
    }

    /// One router request under this pipeline's labels, the count the
    /// dashboards read: recorded per attempt here, and by the router for an
    /// answer it gives without the pipeline, so that traffic stays visible.
    /// The `model` label is the served model's id, or `other` for a model no
    /// worker serves ([`WorkerRegistry::metric_model_label`]), here and in
    /// the error and duration records below.
    pub(crate) fn record_request(&self, endpoint: &'static str, model: &str, streaming: bool) {
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            self.worker_registry.metric_model_label(model),
            endpoint,
            bool_to_static_str(streaming),
        );
    }

    /// A refused request under this pipeline's labels, by its class (one of
    /// the gateway's own refusals, or the response's status) and the cause
    /// behind it (the worker's status, the refusal's code).
    pub(crate) fn record_error(
        &self,
        endpoint: Option<&'static str>,
        model: &str,
        response: &Response,
    ) {
        if let Some(endpoint) = endpoint {
            Metrics::record_router_error(
                metrics_labels::ROUTER_GRPC,
                self.backend_type,
                metrics_labels::CONNECTION_GRPC,
                self.worker_registry.metric_model_label(model),
                endpoint,
                error_type_for_response(response),
                error_reason_for_response(response),
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
            self.worker_registry.metric_model_label(model),
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
            self.worker_registry.metric_model_label(model),
            endpoint,
            metrics_labels::ERROR_INTERNAL,
            metrics_labels::REASON_INTERNAL,
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
            self.worker_registry.metric_model_label(model),
            endpoint,
            metrics_labels::ERROR_INTERNAL,
            metrics_labels::REASON_INTERNAL,
        );
        error::internal_error("no_response_produced", "No response produced")
    }

    /// Settle a non-streaming success's reservation with real usage, if this
    /// logical request reserved one. No-op if the cell is unset (feature
    /// disabled or endpoint opted out) or was never admitted (e.g. prep
    /// failed before the reserve stage ran). Idempotent -- `settle_success`
    /// is a no-op if the handle was already resolved.
    async fn settle_reservation(cell: Option<&RateLimitCell>, usage: UsageSettlement) {
        let Some(RateLimitOutcome::Admitted(handle)) = cell.and_then(RateLimitCell::peek) else {
            return;
        };
        handle.settle_success(usage).await;
    }

    /// The settlement of a non-streamed chat or messages answer: the engine's
    /// counters the response stage recorded (see `ResponseState::engine_counts`);
    /// all zeros, a full refund, when none were (the stage always records
    /// them with a response).
    fn recorded_settlement(response: &ResponseState) -> UsageSettlement {
        response
            .engine_counts
            .map_or_else(UsageSettlement::default, TokenCounts::settlement)
    }

    /// The settlement of a non-streamed legacy completion: its usage object
    /// is the engine's counters unchanged (no unbilled prompt tokens, no
    /// parser), summed over the prompts of a batch.
    pub(in crate::routers::grpc) fn completion_settlement(
        usage: Option<&Usage>,
    ) -> UsageSettlement {
        usage.map_or_else(UsageSettlement::default, |usage| UsageSettlement {
            actual_input_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cached_input_tokens: usage
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |details| details.cached_tokens),
            reasoning_tokens: usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens)
                .unwrap_or(0),
        })
    }

    /// The settlement of a non-streamed generate answer, one entry per
    /// choice: the choices share one prompt, so its count and its cache hit
    /// are taken once, the greatest any choice reports, as `TokenCounts::of`
    /// and the streamed paths take them (the choice that prefilled the
    /// shared prefix may report fewer hits than its siblings), and each
    /// choice's own completion and reasoning counts sum.
    pub(in crate::routers::grpc) fn generate_settlement(
        responses: &[GenerateResponse],
    ) -> UsageSettlement {
        TokenCounts::of_generated(responses).settlement()
    }

    /// The settlement of an embeddings or classification answer: the prompt
    /// tokens the engine counted, no completion.
    pub(in crate::routers::grpc) fn prompt_only_settlement(prompt_tokens: u32) -> UsageSettlement {
        UsageSettlement {
            actual_input_tokens: prompt_tokens,
            ..UsageSettlement::default()
        }
    }

    /// The settlement of a decision batch: its input tokens, no completion
    /// (label scoring generates none, as today), and the cache and reasoning
    /// counters its usage reports, zero while the backend reports none.
    pub(in crate::routers::grpc) fn decisions_settlement(usage: &DecisionUsage) -> UsageSettlement {
        let count = |tokens: u64| u32::try_from(tokens).unwrap_or(u32::MAX);
        UsageSettlement {
            actual_input_tokens: count(usage.input_tokens),
            completion_tokens: 0,
            cached_input_tokens: count(usage.input_tokens_details.cached_tokens),
            reasoning_tokens: count(usage.output_tokens_details.reasoning_tokens),
        }
    }

    /// [`Self::run`], reporting to a Responses stream's startup barrier
    /// whether the backend request started. A run that fails before then
    /// hands its response to the barrier; the caller gets a placeholder the
    /// Responses stream never forwards.
    fn run_signalling<'a>(
        &'a self,
        ctx: RequestContext,
        metrics_endpoint: Option<&'static str>,
        retry_config: Option<&'a RetryConfig>,
        stream_start: Option<StreamStartupSender>,
    ) -> impl Future<Output = Result<RunOutcome, Response>> + Send + 'a {
        let run = self.run(ctx, metrics_endpoint, retry_config);
        async move {
            let outcome = run.await;
            let Some(startup) = stream_start else {
                return outcome;
            };
            // A dropped receiver means the client left; nothing waits for
            // either message.
            match outcome {
                Ok(outcome) => {
                    let _ = startup.send(Ok(()));
                    Ok(outcome)
                }
                Err(failure) => {
                    let _ = startup.send(Err(failure));
                    Err(error::internal_error(
                        "responses_stream_startup_failed",
                        "Streaming response failed before backend request startup",
                    ))
                }
            }
        }
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
        stream_start: Option<StreamStartupSender>,
    ) -> Response {
        const ENDPOINT: &str = metrics_labels::ENDPOINT_CHAT;
        let run = {
            let mut ctx = RequestContext::for_chat(request, headers, model_id, components);
            ctx.input.tenant_request_meta = tenant_request_meta;
            ctx.input.rate_limit_cell = rate_limit_cell;
            self.run_signalling(ctx, Some(ENDPOINT), retry_config, stream_start)
        };
        match run.await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Chat(response)) => {
                    // The engine's counts: the usage shown excludes the unbilled
                    // stub and may carry the parser's reasoning split.
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        Self::recorded_settlement(&dctx.response),
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    Self::routed_response(&dctx, axum::Json(response).into_response())
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
        // The response needs to know the request's shape (single prompt vs.
        // batch, and `n`) to answer like SGLang, but `into_dispatch` drops
        // the request by construction (see the module doc) -- the dispatch
        // phase runs with no request, on purpose. Read the two scalars off
        // the request now, before it moves into the context, instead of
        // retaining a handle across dispatch.
        //
        // `single_prompt` is false for `InputIds::Batch`, but that shape
        // never reaches here: the preparation stage 400s it first.
        let single_prompt =
            request.text.is_some() || matches!(request.input_ids, Some(InputIds::Single(_)));
        let n = request
            .sampling_params
            .as_ref()
            .and_then(|sp| sp.n)
            .unwrap_or(1);
        const ENDPOINT: &str = metrics_labels::ENDPOINT_GENERATE;
        let run = {
            let mut ctx = RequestContext::for_generate(request, headers, model_id, components);
            ctx.input.tenant_request_meta = tenant_request_meta;
            ctx.input.rate_limit_cell = rate_limit_cell;
            self.run(ctx, Some(ENDPOINT), retry_config)
        };
        match run.await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Generate(response)) => {
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        Self::generate_settlement(&response),
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    let body = generate_response_body(single_prompt, n, response);
                    Self::routed_response(&dctx, axum::Json(body).into_response())
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
        const ENDPOINT: &str = metrics_labels::ENDPOINT_COMPLETIONS;
        let run = {
            let mut ctx = RequestContext::for_completion(request, headers, model_id, components);
            ctx.input.tenant_request_meta = tenant_request_meta;
            ctx.input.rate_limit_cell = rate_limit_cell;
            self.run(ctx, Some(ENDPOINT), retry_config)
        };
        match run.await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Completion(response)) => {
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        Self::completion_settlement(response.usage.as_ref()),
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    Self::routed_response(&dctx, axum::Json(response).into_response())
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
        const ENDPOINT: &str = metrics_labels::ENDPOINT_MESSAGES;
        let run = {
            let mut ctx = RequestContext::for_messages(request, headers, model_id, components);
            ctx.input.tenant_request_meta = tenant_request_meta;
            ctx.input.rate_limit_cell = rate_limit_cell;
            self.run(ctx, Some(ENDPOINT), retry_config)
        };
        match run.await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Messages(response)) => {
                    // The engine's counts: the Messages usage reports no cache
                    // hits and the greater of the engine's and the parser's
                    // thinking count.
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        Self::recorded_settlement(&dctx.response),
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    Self::routed_response(&dctx, axum::Json(response).into_response())
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

    /// Execute a decision batch and settle its single logical reservation.
    #[expect(
        clippy::too_many_arguments,
        reason = "matches the rate-limited endpoint entry contract"
    )]
    pub async fn execute_decisions(
        &self,
        request: Arc<DecisionsRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
        rate_limit_cell: Option<Arc<RateLimitCell>>,
        retry_config: Option<&RetryConfig>,
    ) -> Response {
        const ENDPOINT: &str = metrics_labels::ENDPOINT_DECISIONS;
        let run = {
            let mut ctx = RequestContext::for_decisions(request, headers, model_id, components);
            ctx.input.tenant_request_meta = tenant_request_meta;
            ctx.input.rate_limit_cell = rate_limit_cell;
            self.run(ctx, Some(ENDPOINT), retry_config)
        };
        match run.await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Decisions(response)) => {
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        Self::decisions_settlement(&response.usage),
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    Self::routed_response(&dctx, axum::Json(response).into_response())
                }
                Some(other) => self.wrong_response_type(
                    "execute_decisions",
                    "Decisions",
                    &other,
                    &dctx.model_id,
                    ENDPOINT,
                ),
                None => self.no_response_produced("execute_decisions", &dctx.model_id, ENDPOINT),
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
        rate_limit_cell: Option<Arc<RateLimitCell>>,
    ) -> Response {
        const ENDPOINT: &str = metrics_labels::ENDPOINT_EMBEDDINGS;
        let run = {
            let mut ctx = RequestContext::for_embedding(request, headers, model_id, components);
            ctx.input.tenant_request_meta = tenant_request_meta;
            ctx.input.rate_limit_cell = rate_limit_cell;
            self.run(ctx, Some(ENDPOINT), None)
        };
        match run.await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Embedding(response)) => {
                    // The prompt tokens the engine counted; an embedding has
                    // no completion.
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        Self::prompt_only_settlement(response.usage.prompt_tokens),
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    Self::routed_response(&dctx, axum::Json(response).into_response())
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

    /// Execute the complete pipeline for an audio transcription request.
    #[expect(
        clippy::too_many_arguments,
        reason = "matches the rate-limited endpoint entry contract"
    )]
    pub async fn execute_transcription(
        &self,
        request: Arc<openai_protocol::transcription::TranscriptionRequest>,
        audio: Arc<openai_protocol::transcription::AudioFile>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
        rate_limit_cell: Option<Arc<RateLimitCell>>,
    ) -> Response {
        // Same label the HTTP layer records for /v1/audio/transcriptions, so
        // the endpoint's metrics don't split across backends.
        const ENDPOINT: &str = metrics_labels::ENDPOINT_AUDIO_TRANSCRIPTIONS;
        let run = {
            let mut ctx =
                RequestContext::for_transcription(request, audio, headers, model_id, components);
            ctx.input.tenant_request_meta = tenant_request_meta;
            ctx.input.rate_limit_cell = rate_limit_cell;
            self.run(ctx, Some(ENDPOINT), None)
        };
        match run.await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Transcription { text, format }) => {
                    // The engine's counts of the transcript's generation, as
                    // the response stage recorded them.
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        Self::recorded_settlement(&dctx.response),
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    Self::routed_response(
                        &dctx,
                        super::regular::stages::transcription::render(format, text),
                    )
                }
                Some(other) => self.wrong_response_type(
                    "execute_transcription",
                    "Transcription",
                    &other,
                    &dctx.model_id,
                    ENDPOINT,
                ),
                None => {
                    self.no_response_produced("execute_transcription", &dctx.model_id, ENDPOINT)
                }
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
        rate_limit_cell: Option<Arc<RateLimitCell>>,
    ) -> Response {
        const ENDPOINT: &str = metrics_labels::ENDPOINT_CLASSIFY;
        let run = {
            let mut ctx = RequestContext::for_classify(request, headers, model_id, components);
            ctx.input.tenant_request_meta = tenant_request_meta;
            ctx.input.rate_limit_cell = rate_limit_cell;
            self.run(ctx, Some(ENDPOINT), None)
        };
        match run.await {
            Ok(RunOutcome::Early(response)) => response,
            Ok(RunOutcome::Final(mut dctx, start)) => match dctx.response.final_response.take() {
                Some(FinalResponse::Classify(response)) => {
                    // The prompt tokens the engine counted; a classification
                    // has no completion.
                    Self::settle_reservation(
                        dctx.rate_limit_cell.as_deref(),
                        Self::prompt_only_settlement(response.usage.prompt_tokens),
                    )
                    .await;
                    self.record_duration(ENDPOINT, &dctx.model_id, start);
                    Self::routed_response(&dctx, axum::Json(response).into_response())
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
        let run = {
            let mut ctx = RequestContext::for_chat(request, headers, model_id, components);
            ctx.input.tenant_request_meta = tenant_request_meta;
            self.run(ctx, None, None)
        };
        match run.await {
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
        let run = {
            let mut ctx = RequestContext::for_responses(
                Arc::new(request.clone()),
                None,                  // No headers needed for internal pipeline execution
                request.model.clone(), // Model ID from request
                harmony_ctx.components.clone(),
            );
            ctx.input.tenant_request_meta = tenant_request_meta;
            self.run(ctx, None, None)
        };
        let mut dctx = match run.await {
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
        stream_start: Option<StreamStartupSender>,
    ) -> Result<(ExecutionResult, Option<LoadGuards>), Response> {
        let run = {
            let mut ctx = RequestContext::for_responses(
                Arc::new(request.clone()),
                None,
                request.model.clone(),
                harmony_ctx.components.clone(),
            );
            ctx.input.tenant_request_meta = tenant_request_meta;
            self.run_signalling(ctx, None, None, stream_start)
        };
        let mut dctx = match run.await {
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

/// A `/generate` response body: a bare object for a single prompt, a list
/// otherwise. `#[serde(untagged)]` makes `One` serialize as its inner
/// object with no wrapper, matching SGLang's shape -- see the doc comment
/// on [`GenerateResponse`].
#[derive(serde::Serialize)]
#[serde(untagged)]
enum GenerateBody {
    One(Box<GenerateResponse>),
    Many(Vec<GenerateResponse>),
}

/// SGLang answers a single prompt with one object and a batch (or `n > 1`)
/// with a list. Mirror that shape here so native clients that never batch
/// (slime, verl) can index `meta_info` directly instead of unwrapping a
/// one-element list.
///
/// `single_prompt` and `n` are read off the request in `execute_generate`
/// (the only caller) before it moves into the dispatch context, which is
/// request-free by construction -- see the module doc.
fn generate_response_body(
    single_prompt: bool,
    n: u32,
    responses: Vec<GenerateResponse>,
) -> GenerateBody {
    if single_prompt && n <= 1 && responses.len() == 1 {
        let mut responses = responses;
        return GenerateBody::One(Box::new(responses.remove(0)));
    }
    GenerateBody::Many(responses)
}

#[cfg(test)]
mod generate_response_body_tests {
    use openai_protocol::generate::{GenerateFinishReason, GenerateFinishType, GenerateMetaInfo};

    use super::*;

    fn response(weight_version: &str) -> GenerateResponse {
        GenerateResponse {
            text: "hi".to_string(),
            output_ids: vec![1, 2, 3],
            meta_info: GenerateMetaInfo {
                id: "req-1".to_string(),
                finish_reason: GenerateFinishReason::Stop {
                    finish_type: GenerateFinishType::Stop,
                },
                prompt_tokens: 3,
                weight_version: weight_version.to_string(),
                input_token_logprobs: None,
                output_token_logprobs: None,
                completion_tokens: 3,
                cached_tokens: 0,
                reasoning_tokens: None,
                e2e_latency: 0.1,
                matched_stop: None,
            },
        }
    }

    fn to_value(body: GenerateBody) -> serde_json::Value {
        serde_json::to_value(body).expect("GenerateBody always serializes")
    }

    #[test]
    fn single_prompt_with_one_response_is_an_object() {
        let value = to_value(generate_response_body(true, 1, vec![response("v1")]));
        assert!(value.is_object(), "expected an object, got {value}");
        assert_eq!(value["meta_info"]["weight_version"], "v1");
    }

    #[test]
    fn n_greater_than_one_stays_a_list_even_for_a_single_prompt() {
        let value = to_value(generate_response_body(
            true,
            2,
            vec![response("v1"), response("v1")],
        ));
        assert!(value.is_array(), "expected a list, got {value}");
        assert_eq!(value.as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn a_batch_request_stays_a_list() {
        // `single_prompt = false` here stands in for a batch `input_ids`
        // request. That shape is unreachable over gRPC today --
        // `GeneratePreparationStage::resolve_generate_input` rejects
        // `InputIds::Batch` with 400 before any response exists -- but the
        // helper still needs to answer it correctly if that ever changes.
        let value = to_value(generate_response_body(
            false,
            1,
            vec![response("v1"), response("v1")],
        ));
        assert!(value.is_array(), "expected a list, got {value}");
    }

    #[test]
    fn single_prompt_with_more_than_one_response_stays_a_list() {
        // Shouldn't happen in practice (a single prompt dispatches once),
        // but the helper must not silently drop a response if it did.
        let value = to_value(generate_response_body(
            true,
            1,
            vec![response("v1"), response("v2")],
        ));
        assert!(value.is_array(), "expected a list, got {value}");
        assert_eq!(value.as_array().map(Vec::len), Some(2));
    }
}

#[cfg(test)]
impl RequestPipeline {
    /// Mode-bearing construction descriptor for the parity test. Stage
    /// ordering itself is structural (see `PipelineStages`), so only the
    /// per-mode args and optional stages need freezing.
    fn signature(&self) -> String {
        format!(
            "{}|encode={}|{}",
            self.stages.worker_selection.signature(),
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
            Endpoint::Transcription => "TranscriptionRequestBuildingStage".to_string(),
            Endpoint::Decisions => "DecisionsRequestBuildingStage".to_string(),
        };
        // Harmony never carries the encode stage.
        let encode = encode && !matches!(endpoint, Endpoint::Harmony);
        (
            format!("WorkerSelectionStage({selection})|encode={encode}|{build}"),
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

        for endpoint in [
            Endpoint::Embeddings,
            Endpoint::Classify,
            Endpoint::Transcription,
            Endpoint::Decisions,
        ] {
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
        routers::grpc::regular::symphony::template::Templates,
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
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            policy_registry,
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline = RequestPipeline::build(Endpoint::Chat, Mode::PrefillDecode, &deps).unwrap();
        let components = Arc::new(SharedComponents {
            templates: Arc::new(Templates::new(tokenizer_registry.clone())),
            tokenizer_registry,
            worker_registry,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: utils::ParserResolver::disabled(),
            multimodal: None,
            tokenizer_autoload: true,
            tokenizer_load_failures: Arc::default(),
            parser: crate::config::ParserChoice::Legacy,
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
        path::{Path, PathBuf},
        pin::Pin,
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Mutex, PoisonError, Weak,
        },
        time::{Duration, Instant},
    };

    use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
    use futures::Stream;
    use llm_tokenizer::{
        traits::{Decoder, Tokenizer},
        HuggingFaceTokenizer, MockTokenizer, TokenizerRegistry,
    };
    use metrics_exporter_prometheus::PrometheusBuilder;
    use openai_protocol::{
        completion::CompletionRequest, model_card::ModelCard, worker::HealthCheckConfig,
    };
    use portpicker::pick_unused_port;
    use smg_grpc_client::{
        common_proto as common, constraint_failed_status, tokenspeed_proto as ts, vllm_proto,
    };
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use tonic::{transport::Server, Request as TonicRequest, Response as TonicResponse, Status};
    use tracing::instrument::WithSubscriber;
    use tracing_subscriber::prelude::*;
    use ts::token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer};

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        routers::grpc::{
            client::ModelInfo,
            multimodal::{
                MultimodalComponents, MultimodalConfigRegistry, MultimodalSettings,
                SUPPORTS_VISION_LABEL,
            },
            proto_wrapper::ENGINE_ERROR_MESSAGE,
            regular::symphony::template::Templates,
            utils::tonic_ext::TonicStatusExt,
        },
        worker::{
            circuit_breaker::{CircuitBreakerConfig, CircuitState},
            BasicWorkerBuilder, ConnectionMode, RequestCompletionSink, RuntimeType, Worker,
            WorkerType,
        },
    };

    const MODEL: &str = "request-release-test-model";
    /// A valid 1x1 PNG, as a client sends an inline image.
    const ONE_PIXEL_PNG_DATA_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
    /// The worker label of an engine without a vision encoder (vLLM
    /// `--language-model-only`, or a text-only model).
    const LANGUAGE_MODEL_ONLY: &[(&str, &str)] = &[(SUPPORTS_VISION_LABEL, "false")];

    /// The labels discovery derives from a servicer's `GetModelInfo`, as
    /// `(key, value)` pairs for the worker builders of these tests.
    fn labels_of(info: &ModelInfo) -> Vec<(String, String)> {
        let mut labels: Vec<(String, String)> = info.to_labels().into_iter().collect();
        labels.sort();
        labels
    }

    fn label_pairs(labels: &[(String, String)]) -> Vec<(&str, &str)> {
        labels
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect()
    }

    /// The `(offset, length)` placeholder ranges of one generate call.
    type PlaceholderRanges = Vec<(u32, u32)>;
    type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;
    type KvEventStream = Pin<Box<dyn Stream<Item = Result<common::KvEventBatch, Status>> + Send>>;
    type TokenizerStream =
        Pin<Box<dyn Stream<Item = Result<common::GetTokenizerChunk, Status>> + Send>>;

    /// TokenSpeed stub gated on the parsed request's drop probe: it withholds
    /// its tokens (or, with `gate_rpc`, the generate RPC itself) until the
    /// probe reaches zero strong references or a deadline passes, recording
    /// the outcome in `released`. An ungated stub (no probe) answers
    /// immediately -- used for the PD prefill side. `fail_first` makes the
    /// first generate call return UNAVAILABLE, for retry-replay tests;
    /// `fail_always` fails every call, `fail_internal` answers every call with
    /// INTERNAL (an engine that took the request and failed on it),
    /// `fail_mid_stream` ends the stream with the engine's own error after
    /// its first frame, `fail_constraint` admits the request and ends the
    /// stream with the engine's report that it could not continue the
    /// request under its constraint, `fail_engine_report` admits it and ends
    /// the stream with the engine's own report before any frame, and
    /// `answer_after` stalls the generate RPC, standing in for an engine that
    /// answers only at its own deadline; `frame_gap` paces the response frames,
    /// for a stream that outlasts its dispatch. Every call's input token ids,
    /// prompt text and engine request id are recorded, as is every aborted
    /// request id.
    #[derive(Clone, Default)]
    struct GatedScheduler {
        probe: Option<Weak<CompletionRequest>>,
        gate_rpc: bool,
        released: Arc<AtomicBool>,
        fail_first: bool,
        fail_always: bool,
        fail_internal: bool,
        fail_mid_stream: bool,
        fail_constraint: bool,
        fail_engine_report: bool,
        answer_after: Option<Duration>,
        frame_gap: Option<Duration>,
        calls: Arc<AtomicUsize>,
        seen_input_ids: Arc<Mutex<Vec<Vec<u32>>>>,
        seen_original_texts: Arc<Mutex<Vec<String>>>,
        seen_mm_placeholders: Arc<Mutex<Vec<PlaceholderRanges>>>,
        seen_request_ids: Arc<Mutex<Vec<String>>>,
        aborted_request_ids: Arc<Mutex<Vec<String>>>,
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
                    weight_version: None,
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
                    ..Default::default()
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
            self.seen_mm_placeholders
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(
                    request
                        .mm_inputs
                        .as_ref()
                        .map(|mm| {
                            mm.items
                                .iter()
                                .flat_map(|item| {
                                    item.placeholders.iter().map(|p| (p.offset, p.length))
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                );
            self.seen_original_texts
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(
                    request
                        .tokenized
                        .as_ref()
                        .map(|t| t.original_text.clone())
                        .unwrap_or_default(),
                );
            self.seen_input_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(request.tokenized.map(|t| t.input_ids).unwrap_or_default());
            self.seen_request_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(request.request_id.clone());
            if self.fail_always
                || (self.fail_first && self.calls.fetch_add(1, Ordering::SeqCst) == 0)
            {
                return Err(Status::unavailable("release-test induced failure"));
            }
            if self.fail_internal {
                return Err(Status::internal("release-test induced engine failure"));
            }
            if self.fail_constraint {
                // The engine admitted the request and could not continue it
                // under its constraint: the call succeeds and the stream ends
                // with the report before any frame, as the wires end it.
                let items: Vec<Result<ts::GenerateResponse, Status>> = vec![Err(
                    constraint_failed_status("release-test induced grammar failure"),
                )];
                return Ok(TonicResponse::new(Box::pin(futures::stream::iter(items))));
            }
            if self.fail_engine_report {
                // The engine failed the request and named no cause: the
                // report the ZMQ wires end their streams with for an `error`
                // finish, before any frame.
                let items: Vec<Result<ts::GenerateResponse, Status>> =
                    vec![Err(Status::internal(ENGINE_ERROR_MESSAGE))];
                return Ok(TonicResponse::new(Box::pin(futures::stream::iter(items))));
            }
            if let Some(delay) = self.answer_after {
                tokio::time::sleep(delay).await;
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
            let frame_gap = self.frame_gap;
            let fail_mid_stream = self.fail_mid_stream;
            tokio::spawn(async move {
                if let Some(probe) = probe {
                    Self::await_probe(&probe, &released).await;
                }
                for (index, frame) in generate_frames(&request_id).into_iter().enumerate() {
                    if let Some(gap) = frame_gap {
                        tokio::time::sleep(gap).await;
                    }
                    let frame = if fail_mid_stream && index == 1 {
                        Err(Status::internal(ENGINE_ERROR_MESSAGE))
                    } else {
                        frame
                    };
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
            request: TonicRequest<ts::AbortRequest>,
        ) -> Result<TonicResponse<ts::AbortResponse>, Status> {
            self.aborted_request_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(request.into_inner().request_id);
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

    fn register_worker(
        registry: &WorkerRegistry,
        port: u16,
        worker_type: WorkerType,
    ) -> Arc<dyn Worker> {
        register_worker_with_labels(registry, port, worker_type, &[])
    }

    fn register_worker_with_labels(
        registry: &WorkerRegistry,
        port: u16,
        worker_type: WorkerType,
        labels: &[(&str, &str)],
    ) -> Arc<dyn Worker> {
        register_worker_of_runtime(registry, port, worker_type, RuntimeType::TokenSpeed, labels)
    }

    /// [`register_worker_with_labels`] for a worker typed as another engine:
    /// the capability matrix reads the runtime, and a request refused before
    /// dispatch never speaks to the stub.
    fn register_worker_of_runtime(
        registry: &WorkerRegistry,
        port: u16,
        worker_type: WorkerType,
        runtime: RuntimeType,
        labels: &[(&str, &str)],
    ) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(runtime)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            });
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        let worker: Arc<dyn Worker> = Arc::new(builder.build());
        registry
            .register(Arc::clone(&worker))
            .expect("register release-test worker");
        worker
    }

    /// [`register_worker`] with a circuit breaker opening at `failure_threshold`.
    fn register_worker_with_breaker(
        registry: &WorkerRegistry,
        port: u16,
        failure_threshold: u32,
    ) -> Arc<dyn Worker> {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .model(ModelCard::new(MODEL))
                .circuit_breaker_config(CircuitBreakerConfig {
                    failure_threshold,
                    success_threshold: 1,
                    timeout_duration: Duration::from_secs(60),
                    window_duration: Duration::from_secs(60),
                })
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry
            .register(Arc::clone(&worker))
            .expect("register release-test worker");
        worker
    }

    async fn components(worker_registry: Arc<WorkerRegistry>) -> Arc<SharedComponents> {
        components_with_multimodal(worker_registry, false).await
    }

    /// Components over the mock tokenizer; `with_multimodal` adds the
    /// multimodal bundle (default settings, no model directory) so media
    /// reaches the multimodal path instead of the "not available" refusal.
    async fn components_with_multimodal(
        worker_registry: Arc<WorkerRegistry>,
        with_multimodal: bool,
    ) -> Arc<SharedComponents> {
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
        let multimodal = with_multimodal.then(|| {
            Arc::new(
                MultimodalComponents::new(
                    Arc::new(MultimodalConfigRegistry::new()),
                    None,
                    None,
                    &MultimodalSettings::default(),
                )
                .expect("multimodal components"),
            )
        });
        Arc::new(SharedComponents {
            templates: Arc::new(Templates::new(tokenizer_registry.clone())),
            tokenizer_registry,
            worker_registry,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: utils::ParserResolver::disabled(),
            multimodal,
            tokenizer_autoload: true,
            tokenizer_load_failures: Arc::default(),
            parser: crate::config::ParserChoice::Legacy,
        })
    }

    mod glm53;

    /// A worker whose servicer left `supports_vision` unset (the SGLang and
    /// TokenSpeed protos carry the field; only vLLM's servicer sets it on
    /// purpose) is not a language-model-only worker: its labels carry no
    /// vision verdict, and an image request goes on past the refusal into
    /// multimodal processing (here, to the config lookup the mock tokenizer
    /// cannot satisfy) instead of a `multimodal_not_supported`.
    #[tokio::test]
    async fn image_to_a_worker_without_a_vision_verdict_is_not_refused() {
        let labels = labels_of(&ModelInfo::Sglang(Box::default()));
        assert!(
            labels.iter().all(|(key, _)| key != SUPPORTS_VISION_LABEL),
            "{labels:?}"
        );
        let seen_ids = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            seen_input_ids: Arc::clone(&seen_ids),
            ..Default::default()
        })
        .await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker_with_labels(
            &worker_registry,
            port,
            WorkerType::Regular,
            &label_pairs(&labels),
        );
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline =
            RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).expect("chat pipeline");
        let components = components_with_multimodal(worker_registry, true).await;
        let request: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour is this pixel?"},
                        {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG_DATA_URL}}
                    ]
                }],
                "max_completion_tokens": 32,
            }))
            .expect("chat request"),
        );
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            pipeline.execute_chat(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
                None,
            ),
        )
        .await
        .expect("the mock path answers");
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        assert_ne!(body["error"]["code"], "multimodal_not_supported", "{body}");
        assert_eq!(
            body["error"]["code"], "invalid_multimodal_request",
            "{body}"
        );
    }

    /// An SGLang servicer that carries no media to its engine says so with a
    /// deliberate `supports_vision=false` in its `server_args` (its
    /// `GetModelInfo` bool is no verdict for this runtime): with the labels
    /// discovery merges from both, the worker reads as language model only,
    /// and an image is refused at the gate before anything is decoded or
    /// dispatched, instead of after the media was fetched.
    #[tokio::test]
    async fn image_to_an_sglang_worker_that_carries_no_media_is_refused_before_dispatch() {
        use std::collections::BTreeMap;

        use smg_grpc_client::sglang_proto;

        use crate::routers::grpc::client::ServerInfo;

        let mut merged = ModelInfo::Sglang(Box::default()).to_labels();
        merged.extend(
            ServerInfo::Sglang(Box::new(sglang_proto::GetServerInfoResponse {
                server_args: Some(prost_types::Struct {
                    fields: BTreeMap::from([(
                        "supports_vision".to_string(),
                        prost_types::Value {
                            kind: Some(prost_types::value::Kind::BoolValue(false)),
                        },
                    )]),
                }),
                ..Default::default()
            }))
            .to_labels(),
        );
        let mut labels: Vec<(String, String)> = merged.into_iter().collect();
        labels.sort();
        assert!(
            labels
                .iter()
                .any(|(key, value)| key == SUPPORTS_VISION_LABEL && value == "false"),
            "{labels:?}"
        );
        let seen_ids = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            seen_input_ids: Arc::clone(&seen_ids),
            ..Default::default()
        })
        .await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker_with_labels(
            &worker_registry,
            port,
            WorkerType::Regular,
            &label_pairs(&labels),
        );
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline =
            RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).expect("chat pipeline");
        let components = components_with_multimodal(worker_registry, true).await;
        let request: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour is this pixel?"},
                        {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG_DATA_URL}}
                    ]
                }],
                "max_completion_tokens": 32,
            }))
            .expect("chat request"),
        );
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            pipeline.execute_chat(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
                None,
            ),
        )
        .await
        .expect("the refusal must not wait on anything");
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "multimodal_not_supported", "{body}");
        assert!(
            seen_ids.lock().expect("seen ids").is_empty(),
            "nothing must reach the worker"
        );
    }

    /// An image to a model whose only worker runs without a vision encoder
    /// is refused with a prompt 400 that names the modality and the model,
    /// before any media is decoded and before anything reaches the worker
    /// (an engine without an encoder admits such a request and never
    /// schedules it, which the client saw as a hang).
    #[tokio::test]
    async fn image_to_a_language_model_only_pool_is_refused_before_dispatch() {
        let seen_ids = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            seen_input_ids: Arc::clone(&seen_ids),
            ..Default::default()
        })
        .await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker_with_labels(
            &worker_registry,
            port,
            WorkerType::Regular,
            LANGUAGE_MODEL_ONLY,
        );
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline =
            RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).expect("chat pipeline");
        let components = components_with_multimodal(worker_registry, true).await;
        let request: Arc<ChatCompletionRequest> = Arc::new(
            serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour is this pixel?"},
                        {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG_DATA_URL}}
                    ]
                }],
                "max_completion_tokens": 32,
            }))
            .expect("chat request"),
        );

        let started = Instant::now();
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            pipeline.execute_chat(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
                None,
            ),
        )
        .await
        .expect("the refusal must not wait on anything");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );

        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "multimodal_not_supported", "{body}");
        let message = body["error"]["message"].as_str().expect("message");
        assert!(
            message.starts_with("image input is not supported by model"),
            "{message}"
        );
        assert!(message.contains("language model only"), "{message}");
        assert!(
            seen_ids
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty(),
            "nothing must reach the worker"
        );
    }

    /// The chat pipeline over one stub worker typed as `runtime`, and the
    /// ids the stub saw.
    async fn chat_pipeline_of_runtime(
        runtime: RuntimeType,
    ) -> (
        RequestPipeline,
        Arc<SharedComponents>,
        Arc<Mutex<Vec<Vec<u32>>>>,
    ) {
        let seen_ids = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            seen_input_ids: Arc::clone(&seen_ids),
            ..Default::default()
        })
        .await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker_of_runtime(&worker_registry, port, WorkerType::Regular, runtime, &[]);
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline =
            RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).expect("chat pipeline");
        let components = components_with_multimodal(worker_registry, true).await;
        (pipeline, components, seen_ids)
    }

    /// The error body of a chat request the pipeline answered within its
    /// budget.
    async fn chat_error_body(
        pipeline: &RequestPipeline,
        components: Arc<SharedComponents>,
        request: serde_json::Value,
    ) -> (http::StatusCode, serde_json::Value) {
        let request: Arc<ChatCompletionRequest> =
            Arc::new(serde_json::from_value(request).expect("chat request"));
        let response = tokio::time::timeout(
            Duration::from_secs(10),
            pipeline.execute_chat(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
                None,
            ),
        )
        .await
        .expect("the refusal must not wait on anything");
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain body");
        (
            status,
            serde_json::from_slice(&body).expect("json error body"),
        )
    }

    /// A video to a model whose only worker runs an engine without video in
    /// the capability matrix is refused in preparation, before the model's
    /// media config is read and before any byte of the clip is fetched or
    /// decoded, with the code and the sentence the selection-time check gave
    /// after the decode; with and without a sampling hint on the part (the
    /// hint decides between router-side and worker-side processing, both
    /// after this refusal). The clip is deliberately undecodable: a request
    /// that reached the decoder would be answered differently.
    #[tokio::test]
    async fn video_to_a_pool_whose_engine_takes_no_video_is_refused_before_any_media_is_touched() {
        for video_url in [
            serde_json::json!({"url": "data:video/mp4;base64,invalid", "fps": 0.5}),
            serde_json::json!({"url": "data:video/mp4;base64,invalid"}),
        ] {
            let (pipeline, components, seen_ids) =
                chat_pipeline_of_runtime(RuntimeType::Sglang).await;
            let started = Instant::now();
            let (status, body) = chat_error_body(
                &pipeline,
                components,
                serde_json::json!({
                    "model": MODEL,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this video briefly."},
                            {"type": "video_url", "video_url": video_url}
                        ]
                    }],
                    "max_completion_tokens": 32,
                }),
            )
            .await;
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "{:?}",
                started.elapsed()
            );
            assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(body["error"]["code"], "multimodal_not_supported", "{body}");
            assert_eq!(
                body["error"]["message"], "backend sglang does not support video inputs",
                "{body}"
            );
            assert!(
                seen_ids
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .is_empty(),
                "nothing must reach the worker"
            );
        }
    }

    /// An image to the same worker is not the matrix's refusal: it goes on
    /// into multimodal processing (here, to the config lookup the mock
    /// tokenizer cannot satisfy, as for the worker without a vision verdict).
    #[tokio::test]
    async fn image_to_a_pool_whose_engine_takes_images_goes_on_into_media_processing() {
        let (pipeline, components, _) = chat_pipeline_of_runtime(RuntimeType::Sglang).await;
        let (status, body) = chat_error_body(
            &pipeline,
            components,
            serde_json::json!({
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour is this pixel?"},
                        {"type": "image_url", "image_url": {"url": ONE_PIXEL_PNG_DATA_URL}}
                    ]
                }],
                "max_completion_tokens": 32,
            }),
        )
        .await;
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        assert_ne!(body["error"]["code"], "multimodal_not_supported", "{body}");
        assert_eq!(
            body["error"]["code"], "invalid_multimodal_request",
            "{body}"
        );
    }

    /// The Messages route shares the refusal: an image block to a model
    /// whose only worker runs an engine without image input in the matrix
    /// is refused before any media is touched, with the same sentence; the
    /// same block to an engine that takes images goes on.
    #[tokio::test]
    async fn messages_image_to_a_pool_whose_engine_takes_no_image_is_refused_before_any_media_is_touched(
    ) {
        let (_, png) = ONE_PIXEL_PNG_DATA_URL
            .split_once("base64,")
            .expect("data url");
        let request = serde_json::json!({
            "model": MODEL,
            "max_tokens": 32,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour is this pixel?"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": png
                    }}
                ]
            }],
        });
        for (runtime, refused) in [(RuntimeType::Mlx, true), (RuntimeType::Sglang, false)] {
            let seen_ids = Arc::new(Mutex::new(Vec::new()));
            let port = spawn_stub(GatedScheduler {
                seen_input_ids: Arc::clone(&seen_ids),
                ..Default::default()
            })
            .await;
            let worker_registry = Arc::new(WorkerRegistry::new());
            register_worker_of_runtime(&worker_registry, port, WorkerType::Regular, runtime, &[]);
            let deps = PipelineDeps::pair(
                worker_registry.clone(),
                Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
                None,
                None,
                Duration::from_secs(60),
            );
            let pipeline = RequestPipeline::build(Endpoint::Messages, Mode::Regular, &deps)
                .expect("messages pipeline");
            let components = components_with_multimodal(worker_registry, true).await;
            let request: Arc<CreateMessageRequest> =
                Arc::new(serde_json::from_value(request.clone()).expect("messages request"));
            let response = tokio::time::timeout(
                Duration::from_secs(10),
                pipeline.execute_messages(
                    request,
                    None,
                    MODEL.to_string(),
                    components,
                    None,
                    None,
                    None,
                ),
            )
            .await
            .expect("the refusal must not wait on anything");
            let status = response.status();
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("drain body");
            let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
            assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
            if refused {
                assert_eq!(body["error"]["code"], "multimodal_not_supported", "{body}");
                assert_eq!(
                    body["error"]["message"], "backend mlx does not support image inputs",
                    "{body}"
                );
            } else {
                assert_ne!(body["error"]["code"], "multimodal_not_supported", "{body}");
            }
            assert!(
                seen_ids
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .is_empty(),
                "nothing must reach the worker"
            );
        }
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
        completion_pipeline_with_admission(worker_registry, mode, None)
    }

    fn completion_pipeline_with_admission(
        worker_registry: &Arc<WorkerRegistry>,
        mode: Mode,
        prefill_admission: Option<Arc<PrefillAdmission>>,
    ) -> RequestPipeline {
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            prefill_admission,
            None,
            Duration::from_secs(60),
        );
        RequestPipeline::build(Endpoint::Completion, mode, &deps).expect("completion pipeline")
    }

    fn completion_pipeline_with_policy(
        worker_registry: &Arc<WorkerRegistry>,
        policy: PolicyConfig,
    ) -> RequestPipeline {
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(policy)),
            None,
            None,
            Duration::from_secs(60),
        );
        RequestPipeline::build(Endpoint::Completion, Mode::Regular, &deps)
            .expect("completion pipeline")
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

    /// A request pinned by a routing key whose engine cannot start it
    /// (UNAVAILABLE) is retried on another engine, not replayed on the one
    /// that just refused: one call there, one on the other engine, 200.
    #[tokio::test]
    async fn a_pinned_retry_moves_to_another_engine_after_an_unavailable_start() {
        let seen_refusing = Arc::new(Mutex::new(Vec::new()));
        let seen_serving = Arc::new(Mutex::new(Vec::new()));
        let refusing_port = spawn_stub(GatedScheduler {
            fail_always: true,
            seen_request_ids: Arc::clone(&seen_refusing),
            ..Default::default()
        })
        .await;
        let serving_port = spawn_stub(GatedScheduler {
            seen_request_ids: Arc::clone(&seen_serving),
            ..Default::default()
        })
        .await;
        let calls =
            |seen: &Mutex<Vec<String>>| seen.lock().unwrap_or_else(PoisonError::into_inner).len();

        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, refusing_port, WorkerType::Regular);
        register_worker(&worker_registry, serving_port, WorkerType::Regular);
        let pipeline =
            completion_pipeline_with_policy(&worker_registry, PolicyConfig::ConsistentHashing);
        let components = components(worker_registry).await;
        let retry = fast_retry_config(3);

        // Keys hash to either engine; the first one owned by the refusing
        // engine is the case under test (the others succeed in one call).
        for key in (0..64).map(|i| format!("session-{i}")) {
            let mut headers = http::HeaderMap::new();
            headers.insert("x-smg-routing-key", key.parse().unwrap());
            let refused_before = calls(&seen_refusing);
            let served_before = calls(&seen_serving);
            let response = pipeline
                .execute_completion(
                    completion_request(false),
                    Some(headers),
                    MODEL.to_string(),
                    Arc::clone(&components),
                    None,
                    None,
                    Some(&retry),
                )
                .await;
            assert_eq!(response.status(), http::StatusCode::OK, "key {key}");
            let refused_now = calls(&seen_refusing) - refused_before;
            let served_now = calls(&seen_serving) - served_before;
            if refused_now == 0 {
                assert_eq!(served_now, 1, "a key owned by the serving engine: one call");
                continue;
            }
            assert_eq!(
                refused_now, 1,
                "the refusing engine sees the request once, not once per retry"
            );
            assert_eq!(served_now, 1, "the retry landed on the other engine");
            return;
        }
        panic!("no key hashed to the refusing engine");
    }

    async fn run_and_drain(
        pipeline: RequestPipeline,
        components: Arc<SharedComponents>,
        request: Arc<CompletionRequest>,
        worker_url: &str,
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
        assert_eq!(response.headers()["x-smg-routed-worker-id"], worker_url);
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain SSE body")
    }

    /// The spans the OpenTelemetry layer exported, as a trace backend sees
    /// them: name, start and end.
    #[derive(Clone, Debug, Default)]
    struct ExportedSpans(Arc<Mutex<Vec<opentelemetry_sdk::trace::SpanData>>>);

    impl opentelemetry_sdk::trace::SpanExporter for ExportedSpans {
        fn export(
            &self,
            batch: Vec<opentelemetry_sdk::trace::SpanData>,
        ) -> impl Future<Output = opentelemetry_sdk::error::OTelSdkResult> + Send {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extend(batch);
            std::future::ready(Ok(()))
        }
    }

    /// The exported upstream span covers the response stream, not the
    /// dispatch alone: with the stub pacing its two frames 100 ms apart,
    /// `grpc_execute` ends after the stream's end, not before its first frame.
    #[tokio::test]
    async fn upstream_span_closes_with_the_response_stream() {
        use opentelemetry::trace::TracerProvider as _;
        const FRAME_GAP: Duration = Duration::from_millis(100);
        let request = completion_request(true);
        let port = spawn_stub(GatedScheduler {
            frame_gap: Some(FRAME_GAP),
            ..Default::default()
        })
        .await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, port, WorkerType::Regular);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;
        // `tracing` rebuilds a callsite's cached interest from the current
        // thread's default alone while it knows of a single dispatcher; a
        // parallel test dispatching first, under no subscriber, would then
        // cache "never" for `grpc_execute`. A second live dispatcher keeps
        // every rebuild over all of them.
        let _second_dispatcher = tracing::Dispatch::new(tracing_subscriber::registry());
        let exported = ExportedSpans::default();
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_simple_exporter(exported.clone())
            .build();
        // The gateway's exporter setup: the OpenTelemetry layer over the
        // gateway's own spans alone (see `CustomOtelFilter`).
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")))
            .with(tracing_subscriber::filter::Targets::new().with_target(
                "smg::otel-trace",
                tracing::level_filters::LevelFilter::TRACE,
            ));

        let (dispatched, drained) = async {
            // A callsite's interest cached before this subscriber existed
            // is rebuilt over the live dispatchers, this one included.
            tracing::callsite::rebuild_interest_cache();
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
            let dispatched = Instant::now();
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("drain SSE body");
            (dispatched, Instant::now())
        }
        .with_subscriber(subscriber)
        .await;

        let upstream_spans = || -> Vec<Duration> {
            exported
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .iter()
                .filter(|span| span.name == "grpc_execute")
                .map(|span| {
                    span.end_time
                        .duration_since(span.start_time)
                        .unwrap_or_default()
                })
                .collect()
        };
        // The body's drop ends the span; give a slow thread a moment.
        for _ in 0..50 {
            if !upstream_spans().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let lasted = upstream_spans();
        assert_eq!(lasted.len(), 1, "one upstream span per dispatch");
        assert!(
            drained >= dispatched + FRAME_GAP,
            "the stub paced the stream over {:?}",
            drained - dispatched
        );
        assert!(
            lasted[0] >= FRAME_GAP,
            "the exported upstream span lasted {:?}: it ended at dispatch, not with the stream ({:?} later)",
            lasted[0],
            drained - dispatched
        );
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

        let body = run_and_drain(
            pipeline,
            components,
            request,
            &format!("grpc://127.0.0.1:{port}"),
        )
        .await;

        assert!(
            released.load(Ordering::SeqCst),
            "the parsed request must be freed before the first token"
        );
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("data: [DONE]"), "stream must finish: {body}");
    }

    /// A text-only request puts its token ids on the wire and not a second
    /// copy of the rendered prompt: the worker generates from the ids, and a
    /// million-token prompt's text would triple the Generate.
    #[tokio::test]
    async fn text_only_request_sends_its_ids_without_the_prompt_text() {
        let request = completion_request(false);
        let seen_ids = Arc::new(Mutex::new(Vec::new()));
        let seen_texts = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            seen_input_ids: Arc::clone(&seen_ids),
            seen_original_texts: Arc::clone(&seen_texts),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, port, WorkerType::Regular);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        run_and_drain(
            pipeline,
            components,
            request,
            &format!("grpc://127.0.0.1:{port}"),
        )
        .await;

        let ids = seen_ids.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(ids.len(), 1, "one Generate: {ids:?}");
        assert!(!ids[0].is_empty(), "the prompt's ids are on the wire");
        let texts = seen_texts.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(
            texts.as_slice(),
            [String::new()],
            "a text-only request sends no prompt text next to its ids"
        );
    }

    /// grpc_pd twin: the decode side's stream task must not pin the parsed
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

        let body = run_and_drain(
            pipeline,
            components,
            request,
            &format!("grpc://127.0.0.1:{decode_port}"),
        )
        .await;

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
        assert_eq!(
            response.headers()["x-smg-routed-worker-id"],
            format!("grpc://127.0.0.1:{port}")
        );
        assert!(
            released.load(Ordering::SeqCst),
            "the parsed request must be freed before the upstream answers"
        );
    }

    /// grpc_pd twin of the dispatch-release probe: the gated decode side
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
        assert_eq!(
            response.headers()["x-smg-routed-worker-id"],
            format!("grpc://127.0.0.1:{decode_port}")
        );
        assert!(
            released.load(Ordering::SeqCst),
            "the parsed request must be freed before the decode side answers"
        );
    }

    /// The one `smg_worker_errors_total` sample of the gRPC workers of
    /// `worker_type`, if any.
    fn side_worker_error_line<'a>(rendered: &'a str, worker_type: &str) -> Option<&'a str> {
        let worker_label = format!("worker_type=\"{worker_type}\"");
        rendered.lines().find(|line| {
            line.starts_with("smg_worker_errors_total{")
                && line.contains(&worker_label)
                && line.contains("connection_mode=\"grpc\"")
        })
    }

    /// A decode side's stream that fails after its first frame while a
    /// non-streamed disaggregated answer is collected is the decode worker's
    /// failure, as it is for a streamed answer: one sample of
    /// `smg_worker_errors_total` under the side's labels per attempt
    /// (`engine_error` for the engine's own report), none for the prefill side
    /// that completed.
    #[test]
    fn a_decode_side_that_fails_mid_way_counts_against_its_worker_for_a_non_streamed_answer() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let attempts = metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async {
                let decode_calls = Arc::new(Mutex::new(Vec::new()));
                let prefill_port = spawn_stub(GatedScheduler::default()).await;
                let decode_port = spawn_stub(GatedScheduler {
                    fail_mid_stream: true,
                    seen_request_ids: Arc::clone(&decode_calls),
                    ..Default::default()
                })
                .await;
                let worker_registry = Arc::new(WorkerRegistry::new());
                register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
                register_worker(&worker_registry, decode_port, WorkerType::Decode);
                let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
                let components = components(worker_registry).await;

                let response = run_completion(pipeline, components, &fast_retry_config(3)).await;

                assert_eq!(
                    response.status(),
                    http::StatusCode::INTERNAL_SERVER_ERROR,
                    "the engine's error ends a non-streamed answer with a 500"
                );
                let attempts = calls_seen(&decode_calls);
                assert!(attempts >= 1, "the decode side was dispatched");
                attempts
            })
        });
        let rendered = handle.render();
        let line = side_worker_error_line(&rendered, "decode")
            .unwrap_or_else(|| panic!("the decode worker's failure is counted:\n{rendered}"));
        assert!(line.contains("error_type=\"engine_error\""), "{line}");
        assert!(
            line.ends_with(&format!(" {attempts}")),
            "one sample per attempt ({attempts}): {line}"
        );
        assert!(
            side_worker_error_line(&rendered, "prefill").is_none(),
            "the prefill side completed:\n{rendered}"
        );
    }

    /// The prefill side's stream, drained before the answer, counts the same
    /// way under its own labels when it fails after its first frame; the
    /// decode side, dropped unread, is not charged.
    #[test]
    fn a_prefill_side_that_fails_mid_way_counts_against_its_worker_for_a_non_streamed_answer() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async {
                let prefill_port = spawn_stub(GatedScheduler {
                    fail_mid_stream: true,
                    ..Default::default()
                })
                .await;
                let decode_port = spawn_stub(GatedScheduler::default()).await;
                let worker_registry = Arc::new(WorkerRegistry::new());
                register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
                register_worker(&worker_registry, decode_port, WorkerType::Decode);
                let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
                let components = components(worker_registry).await;

                let response = run_completion(pipeline, components, &fast_retry_config(1)).await;

                assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
            });
        });
        let rendered = handle.render();
        let line = side_worker_error_line(&rendered, "prefill")
            .unwrap_or_else(|| panic!("the prefill worker's failure is counted:\n{rendered}"));
        assert!(line.contains("error_type=\"engine_error\""), "{line}");
        assert!(line.ends_with(" 1"), "{line}");
        assert!(
            side_worker_error_line(&rendered, "decode").is_none(),
            "the decode side did not fail:\n{rendered}"
        );
    }

    /// The streamed answer's accounting is unchanged by the sides' attribution:
    /// a decode side that fails mid-way is one sample, counted where the stream
    /// is written, not a second time where it is read.
    #[test]
    fn a_decode_side_that_fails_mid_way_is_counted_once_for_a_streamed_answer() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async {
                let prefill_port = spawn_stub(GatedScheduler::default()).await;
                let decode_port = spawn_stub(GatedScheduler {
                    fail_mid_stream: true,
                    ..Default::default()
                })
                .await;
                let worker_registry = Arc::new(WorkerRegistry::new());
                register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
                register_worker(&worker_registry, decode_port, WorkerType::Decode);
                let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
                let components = components(worker_registry).await;

                let body = run_and_drain(
                    pipeline,
                    components,
                    completion_request(true),
                    &format!("grpc://127.0.0.1:{decode_port}"),
                )
                .await;

                let body = String::from_utf8_lossy(&body);
                assert!(
                    body.contains("\"error\""),
                    "the stream ends with the error event: {body}"
                );
                assert!(body.contains("data: [DONE]"), "{body}");
            });
        });
        let rendered = handle.render();
        let line = side_worker_error_line(&rendered, "decode")
            .unwrap_or_else(|| panic!("the decode worker's failure is counted:\n{rendered}"));
        assert!(line.contains("error_type=\"engine_error\""), "{line}");
        assert!(line.ends_with(" 1"), "counted once: {line}");
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
            // The header switch alone must yield the header: the capture
            // does not depend on the log switch.
            let trace_header = response.headers().get("x-smg-cache-trace");
            if std::env::var("SMG_CACHE_TRACE_HEADER").is_ok_and(|v| v == "1") {
                let trace: serde_json::Value =
                    serde_json::from_str(trace_header.unwrap().to_str().unwrap()).unwrap();
                assert_eq!(trace["attempt"], 1);
                assert_eq!(trace["engine_ids"], serde_json::json!([ids[1]]));
                assert!(trace["selections"][0].get("candidates").is_none());
            } else {
                assert!(trace_header.is_none());
            }
        }
    }

    /// `x-smg-cache-trace: 1` on a request returns that request's decision
    /// record, with the chosen worker, whatever the process switches say.
    #[tokio::test]
    async fn a_request_header_opts_into_the_decision_record_with_its_worker() {
        let port = spawn_stub(GatedScheduler::default()).await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker = register_worker(&worker_registry, port, WorkerType::Regular);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-cache-trace", http::HeaderValue::from_static("1"));
        let response = pipeline
            .execute_completion(
                completion_request(false),
                Some(headers),
                MODEL.to_string(),
                Arc::clone(&components),
                None,
                None,
                None,
            )
            .await;
        assert_eq!(response.status(), http::StatusCode::OK);
        let trace: serde_json::Value =
            serde_json::from_str(response.headers()["x-smg-cache-trace"].to_str().unwrap())
                .unwrap();
        assert_eq!(trace["attempt"], 0);
        assert_eq!(trace["selections"][0]["worker"], worker.url());
        assert_eq!(trace["selections"][0]["origin"], "policy");
        assert!(trace["selections"][0].get("candidates").is_none());
        assert_eq!(response.headers()["x-smg-routed-worker-id"], worker.url());

        // Without the request header the process switches decide, as before.
        let response = pipeline
            .execute_completion(
                completion_request(false),
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
            )
            .await;
        assert_eq!(response.status(), http::StatusCode::OK);
        if !std::env::var("SMG_CACHE_TRACE_HEADER").is_ok_and(|v| v == "1") {
            assert!(response.headers().get("x-smg-cache-trace").is_none());
        }
    }

    async fn run_completion(
        pipeline: RequestPipeline,
        components: Arc<SharedComponents>,
        retry: &RetryConfig,
    ) -> Response {
        run_completion_of(pipeline, components, retry, completion_request(false)).await
    }

    async fn run_completion_of(
        pipeline: RequestPipeline,
        components: Arc<SharedComponents>,
        retry: &RetryConfig,
        request: Arc<CompletionRequest>,
    ) -> Response {
        pipeline
            .execute_completion(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                Some(retry),
            )
            .await
    }

    /// The completion with a constraint on its output (a JSON schema).
    fn constrained_completion_request() -> Arc<CompletionRequest> {
        Arc::new(
            serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "prompt": "Hello world",
                "stream": false,
                "json_schema": "{\"type\":\"object\"}",
            }))
            .expect("constrained completion request"),
        )
    }

    fn calls_seen(ids: &Mutex<Vec<String>>) -> usize {
        ids.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    /// Three attempts of one request on a worker that cannot serve it now are
    /// one failure for its breaker: the threshold counts failed requests, not
    /// attempts.
    #[tokio::test]
    async fn transient_start_failures_charge_the_breaker_once_per_request() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            fail_always: true,
            seen_request_ids: Arc::clone(&seen),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker = register_worker_with_breaker(&worker_registry, port, 2);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        let response = run_completion(pipeline, components, &fast_retry_config(3)).await;

        assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            calls_seen(&seen),
            3,
            "an UNAVAILABLE start is replayed up to the retry limit"
        );
        assert_eq!(
            worker.circuit_breaker_state(),
            CircuitState::Closed,
            "one failed request is one failure, under the threshold of two"
        );
    }

    /// INTERNAL is the engine's own answer for this payload: it is returned
    /// at once rather than replayed on the worker, and the worker is charged
    /// once.
    #[tokio::test]
    async fn an_internal_engine_error_is_not_replayed_on_its_worker() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            fail_internal: true,
            seen_request_ids: Arc::clone(&seen),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker = register_worker_with_breaker(&worker_registry, port, 2);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        let response = run_completion(pipeline, components, &fast_retry_config(5)).await;

        assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            calls_seen(&seen),
            1,
            "the answer is definitive for this payload"
        );
        assert!(!is_retryable_response(&response));
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    }

    /// A worker's report that the engine could not continue the request under
    /// its constraint is the request's answer, a 422 `constraint_failed`: one
    /// call on the worker and none elsewhere (the same tokens fail the same
    /// way on a replay, each replay a prefill), terminal for the retry layer,
    /// and no breaker sample against a worker that is healthy.
    #[tokio::test]
    async fn a_constraint_failure_is_answered_once_and_never_replayed() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            fail_constraint: true,
            seen_request_ids: Arc::clone(&seen),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker = register_worker_with_breaker(&worker_registry, port, 1);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        let response = run_completion(pipeline, components, &fast_retry_config(5)).await;

        assert_eq!(response.status(), http::StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "constraint_failed"
        );
        assert_eq!(calls_seen(&seen), 1, "the answer is final for this request");
        assert!(!is_retryable_response(&response));
        assert_eq!(
            worker.circuit_breaker_state(),
            CircuitState::Closed,
            "a healthy worker takes no sample"
        );
    }

    /// The engine's own report on a constrained request (an `error` finish
    /// that names no cause: a failed KV handoff or a pause as often as a
    /// grammar rejection) is answered exactly as on a request without a
    /// constraint: the same status and code (the engine's error, a 500, never
    /// the constraint's 422, which only a worker's own attribution produces),
    /// the same calls on the worker (the retry layer replaying it as before)
    /// and the same breaker state.
    #[tokio::test]
    async fn an_engines_unattributed_report_on_a_constrained_request_keeps_its_shape() {
        let mut outcomes = Vec::new();
        for request in [completion_request(false), constrained_completion_request()] {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let port = spawn_stub(GatedScheduler {
                fail_engine_report: true,
                seen_request_ids: Arc::clone(&seen),
                ..Default::default()
            })
            .await;

            let worker_registry = Arc::new(WorkerRegistry::new());
            let worker = register_worker_with_breaker(&worker_registry, port, 1);
            let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
            let components = components(worker_registry).await;

            let response =
                run_completion_of(pipeline, components, &fast_retry_config(2), request).await;
            outcomes.push((
                response.status(),
                error::extract_error_code_from_response(&response).to_string(),
                calls_seen(&seen),
                worker.circuit_breaker_state(),
            ));
        }

        let (plain, constrained) = (&outcomes[0], &outcomes[1]);
        assert_eq!(plain.0, http::StatusCode::INTERNAL_SERVER_ERROR);
        assert_ne!(
            constrained.1, "constraint_failed",
            "no attribution, no constraint failure"
        );
        assert!(
            plain.2 >= 2,
            "the engine's error is replayed as before ({} calls)",
            plain.2
        );
        assert_eq!(
            constrained, plain,
            "a constraint changes nothing about the engine's own report"
        );
    }

    /// A smaller engine is not an alternative to the engine that returned a
    /// definitive error: keep its 500, without a futile retry or a false 400.
    #[tokio::test]
    async fn context_ineligible_worker_does_not_extend_the_retry_window() {
        let seen_small = Arc::new(Mutex::new(Vec::new()));
        let seen_large = Arc::new(Mutex::new(Vec::new()));
        let small_port = spawn_stub(GatedScheduler {
            seen_request_ids: Arc::clone(&seen_small),
            ..Default::default()
        })
        .await;
        let large_port = spawn_stub(GatedScheduler {
            fail_internal: true,
            seen_request_ids: Arc::clone(&seen_large),
            ..Default::default()
        })
        .await;
        let registry = Arc::new(WorkerRegistry::new());
        for (port, window) in [(small_port, 1), (large_port, 64)] {
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                        .worker_type(WorkerType::Regular)
                        .connection_mode(ConnectionMode::Grpc)
                        .runtime_type(RuntimeType::TokenSpeed)
                        .model(ModelCard::new(MODEL).with_context_length(window))
                        .health_config(HealthCheckConfig {
                            disable_health_check: true,
                            ..Default::default()
                        })
                        .build(),
                ))
                .expect("register context-bound worker");
        }
        let pipeline = completion_pipeline(&registry, Mode::Regular);
        let response =
            run_completion(pipeline, components(registry).await, &fast_retry_config(5)).await;
        assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!is_retryable_response(&response));
        assert_eq!(calls_seen(&seen_small), 0);
        assert_eq!(calls_seen(&seen_large), 1);
    }

    /// With two workers the request moves to the other one once; a second
    /// INTERNAL answer stands.
    #[tokio::test]
    async fn an_internal_engine_error_moves_the_request_to_another_worker_once() {
        let seen_a = Arc::new(Mutex::new(Vec::new()));
        let seen_b = Arc::new(Mutex::new(Vec::new()));
        let port_a = spawn_stub(GatedScheduler {
            fail_internal: true,
            seen_request_ids: Arc::clone(&seen_a),
            ..Default::default()
        })
        .await;
        let port_b = spawn_stub(GatedScheduler {
            fail_internal: true,
            seen_request_ids: Arc::clone(&seen_b),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker_a = register_worker_with_breaker(&worker_registry, port_a, 5);
        let worker_b = register_worker_with_breaker(&worker_registry, port_b, 5);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        let response = run_completion(pipeline, components, &fast_retry_config(5)).await;

        assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(calls_seen(&seen_a), 1);
        assert_eq!(calls_seen(&seen_b), 1);
        assert!(!is_retryable_response(&response));
        assert_eq!(worker_a.circuit_breaker_state(), CircuitState::Closed);
        assert_eq!(worker_b.circuit_breaker_state(), CircuitState::Closed);
    }

    /// Once the only worker's breaker opens, the verdict clears at the
    /// breaker timeout, not inside a backoff: the 503 ends the retry window.
    #[tokio::test]
    async fn a_pool_of_open_breakers_ends_the_retry_window_at_once() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            fail_always: true,
            seen_request_ids: Arc::clone(&seen),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker = register_worker_with_breaker(&worker_registry, port, 1);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(worker_registry).await;

        let response = run_completion(pipeline, components, &fast_retry_config(5)).await;

        assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
        assert_eq!(calls_seen(&seen), 1, "the open breaker takes no attempt");
        assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "no_available_workers"
        );
        assert!(!is_retryable_response(&response));
    }

    /// A prefill side that cannot start must answer the client immediately
    /// rather than waiting out the decode side's own deadline, and the decode
    /// room it stranded must be aborted rather than left for the engine to
    /// time out. The decode stub here answers only after `DECODE_DELAY`,
    /// standing in for an engine whose transfer deadline is what the client
    /// used to wait on.
    #[tokio::test]
    async fn a_failed_prefill_answers_now_and_aborts_the_decode_room() {
        const DECODE_DELAY: Duration = Duration::from_secs(2);

        let prefill_port = spawn_stub(GatedScheduler {
            fail_always: true,
            ..Default::default()
        })
        .await;
        let aborted = Arc::new(Mutex::new(Vec::new()));
        let decode_request_ids = Arc::new(Mutex::new(Vec::new()));
        let decode_port = spawn_stub(GatedScheduler {
            answer_after: Some(DECODE_DELAY),
            aborted_request_ids: Arc::clone(&aborted),
            seen_request_ids: Arc::clone(&decode_request_ids),
            ..Default::default()
        })
        .await;

        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
        register_worker(&worker_registry, decode_port, WorkerType::Decode);
        let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
        let components = components(worker_registry).await;

        let started = Instant::now();
        let response = pipeline
            .execute_completion(
                completion_request(false),
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
            )
            .await;
        let answered_in = started.elapsed();

        assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "prefill_worker_failed_to_start"
        );
        assert!(
            answered_in < DECODE_DELAY,
            "the prefill failure must not wait for the decode side, took {answered_in:?}"
        );

        // The decode side is retired off the request path: once its dispatch
        // lands, its stream drops and that is what sends the abort.
        let deadline = Instant::now() + DECODE_DELAY + Duration::from_secs(8);
        loop {
            if !aborted
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_empty()
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the stranded decode room was never aborted"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let aborted = aborted.lock().unwrap_or_else(PoisonError::into_inner);
        let dispatched = decode_request_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        assert_eq!(
            aborted.as_slice(),
            dispatched.as_slice(),
            "the abort must name the decode side's own request id"
        );
    }

    // ------------------------------------------------------------------
    // Request completion on a client cancel. Policies learn that a request
    // ended through the worker's completion sink, which the load guard
    // drives; these pin that a client leaving mid-request reaches it on the
    // gRPC paths, promptly, while the engine still owes its first token.
    // ------------------------------------------------------------------

    /// Records the worker URLs that reported a request completion.
    #[derive(Debug, Default)]
    struct CompletionSpy(Mutex<Vec<String>>);

    impl RequestCompletionSink for CompletionSpy {
        fn request_completed(&self, worker: &dyn Worker) {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(worker.url().to_string());
        }
    }

    impl CompletionSpy {
        fn urls(&self) -> Vec<String> {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }

        /// Wait for `expected` completions, well inside the stub's five-second
        /// gate: a completion that only arrives with the engine's tokens is a
        /// failure, not a late pass.
        async fn wait_for(&self, expected: usize) -> Vec<String> {
            let deadline = Instant::now() + Duration::from_secs(2);
            while self.urls().len() < expected {
                assert!(
                    Instant::now() < deadline,
                    "only {:?} completed in time, expected {expected}",
                    self.urls()
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            self.urls()
        }
    }

    /// Install one spy as the completion sink of every registered worker.
    fn install_spy(registry: &WorkerRegistry) -> Arc<CompletionSpy> {
        let spy = Arc::new(CompletionSpy::default());
        for worker in registry.get_all() {
            worker.set_completion_sink(Some(Arc::clone(&spy) as Arc<dyn RequestCompletionSink>));
        }
        spy
    }

    /// A stub that accepts the generate RPC at once and withholds every token
    /// while `hold` lives: the engine has the request and owes its first
    /// token, the window a client cancel is hardest to account for.
    fn stalled_while(hold: &Arc<CompletionRequest>) -> GatedScheduler {
        GatedScheduler {
            probe: Some(Arc::downgrade(hold)),
            ..Default::default()
        }
    }

    async fn start_stream(
        pipeline: &RequestPipeline,
        components: Arc<SharedComponents>,
    ) -> Response {
        let started = Instant::now();
        let response = pipeline
            .execute_completion(
                completion_request(true),
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
            started.elapsed() < Duration::from_secs(4),
            "the streaming response must open before the engine's first token"
        );
        response
    }

    /// gRPC regular path: the client drops the stream while the engine is
    /// still prefilling. The load guard rides the response body, so the drop
    /// releases the worker's load and reports the completion at once.
    #[tokio::test]
    async fn cancelled_grpc_stream_reports_the_completion_during_prefill() {
        let hold = completion_request(true);
        let port = spawn_stub(stalled_while(&hold)).await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker = register_worker(&worker_registry, port, WorkerType::Regular);
        let spy = install_spy(&worker_registry);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = components(Arc::clone(&worker_registry)).await;

        let response = start_stream(&pipeline, components).await;
        assert_eq!(worker.load(), 1, "the dispatch holds the worker's load");
        assert!(
            spy.urls().is_empty(),
            "nothing completes while the client is connected"
        );

        drop(response);

        assert_eq!(spy.wait_for(1).await, [worker.url().to_string()]);
        assert_eq!(worker.load(), 0);
        drop(hold);
    }

    /// gRPC PD path: the prefill side answers on its own and reports when its
    /// phase ends; the decode side is still owed its first token when the
    /// client leaves, and reports with the dropped body. Both sides end with
    /// their load released, exactly once each.
    #[tokio::test]
    async fn cancelled_pd_stream_reports_the_prefill_and_decode_completions() {
        let hold = completion_request(true);
        let prefill_port = spawn_stub(GatedScheduler::default()).await;
        let decode_port = spawn_stub(stalled_while(&hold)).await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        let prefill = register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
        let decode = register_worker(&worker_registry, decode_port, WorkerType::Decode);
        let spy = install_spy(&worker_registry);
        let pipeline = completion_pipeline(&worker_registry, Mode::PrefillDecode);
        let components = components(Arc::clone(&worker_registry)).await;

        let response = start_stream(&pipeline, components).await;
        assert_eq!(
            decode.load(),
            1,
            "the decode side holds its load until the stream ends"
        );

        drop(response);

        let mut urls = spy.wait_for(2).await;
        urls.sort();
        let mut expected = vec![prefill.url().to_string(), decode.url().to_string()];
        expected.sort();
        assert_eq!(urls, expected, "each side reports exactly once");
        assert_eq!(prefill.load(), 0);
        assert_eq!(decode.load(), 0);
        drop(hold);
    }

    /// gRPC PD path under prefill admission: the prefill side's load is a
    /// reservation in the admission gate rather than a bare guard, and it
    /// still reports the completion when the phase ends, so a cancelled
    /// request frees its slot for the next one.
    #[tokio::test]
    async fn cancelled_pd_stream_with_admitted_prefill_reports_and_frees_the_slot() {
        let hold = completion_request(true);
        let prefill_port = spawn_stub(GatedScheduler::default()).await;
        let decode_port = spawn_stub(stalled_while(&hold)).await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        let prefill = register_worker(&worker_registry, prefill_port, WorkerType::Prefill);
        let decode = register_worker(&worker_registry, decode_port, WorkerType::Decode);
        let spy = install_spy(&worker_registry);
        let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let pipeline = completion_pipeline_with_admission(
            &worker_registry,
            Mode::PrefillDecode,
            Some(Arc::clone(&admission)),
        );
        let components = components(Arc::clone(&worker_registry)).await;

        let response = start_stream(&pipeline, components).await;
        drop(response);

        let mut urls = spy.wait_for(2).await;
        urls.sort();
        let mut expected = vec![prefill.url().to_string(), decode.url().to_string()];
        expected.sort();
        assert_eq!(urls, expected, "each side reports exactly once");
        assert_eq!(
            prefill.load(),
            0,
            "the admitted prefill's reservation is released"
        );
        assert_eq!(decode.load(), 0);
        // The one slot is free again: a new admission does not queue.
        let admitted = tokio::time::timeout(
            Duration::from_millis(500),
            admission.admit(None, |capacity| capacity.select(Arc::clone(&prefill), ())),
        )
        .await
        .expect("the freed slot must admit at once")
        .expect("admitted");
        drop(admitted);
        drop(hold);
    }

    // ------------------------------------------------------------------
    // DeepSeek-V4.1 parity through the pipeline with the real checkpoint
    // tokenizer. Needs `DEEPSEEK_V41_MODEL_DIR` (or the tokenizer crate's
    // download cache); skips otherwise, like the tokenizer crate's parity
    // test.
    // ------------------------------------------------------------------

    const V41_RENDER_FIXTURES: &str = include_str!(
        "../../../../crates/tokenizer/tests/fixtures/deepseek_v41/render_fixtures.json"
    );
    const V41_ID_FIXTURES: &str = include_str!(
        "../../../../crates/tokenizer/tests/fixtures/deepseek_v41/render_ids_fixtures.json"
    );
    const V41_CORN_PNG: &[u8] =
        include_bytes!("../../../../crates/multimodal/tests/fixtures/images/deepseek_v41_corn.png");
    /// `<｜deepseek_image｜>` in the checkpoint's `config.json`.
    const V41_IMAGE_TOKEN_ID: u32 = 129264;
    /// The reference span for `deepseek_v41_corn.png` (450x308), recorded in
    /// the multimodal crate's goldens.
    const V41_CORN_TOKENS: usize = 189;

    fn deepseek_v41_model_dir() -> Option<PathBuf> {
        let dir = std::env::var_os("DEEPSEEK_V41_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../crates/tokenizer/.tokenizer_cache/deepseek_v41")
            });
        (dir.join("tokenizer.json").is_file() && dir.join("config.json").is_file()).then_some(dir)
    }

    #[expect(
        clippy::print_stderr,
        reason = "test diagnostic: says why the parity test did not run"
    )]
    fn skip_no_tokenizer() {
        eprintln!("skipping: no DeepSeek-V4.1 tokenizer (set DEEPSEEK_V41_MODEL_DIR)");
    }

    /// The recorded request (`messages`, kwargs) and reference ids of one
    /// tokenizer-crate fixture case.
    fn v41_fixture(name: &str) -> (serde_json::Value, Vec<u32>) {
        let render: serde_json::Value =
            serde_json::from_str(V41_RENDER_FIXTURES).expect("render fixtures");
        let ids: serde_json::Value = serde_json::from_str(V41_ID_FIXTURES).expect("id fixtures");
        let find = |doc: &serde_json::Value| {
            doc["cases"]
                .as_array()
                .expect("cases")
                .iter()
                .find(|case| case["name"] == name)
                .unwrap_or_else(|| panic!("fixture case {name}"))
                .clone()
        };
        let case = find(&render);
        let ids = find(&ids)["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|id| id.as_u64().expect("token id") as u32)
            .collect();
        (case, ids)
    }

    async fn v41_components(
        dir: &Path,
        worker_registry: Arc<WorkerRegistry>,
        with_multimodal: bool,
    ) -> Arc<SharedComponents> {
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Arc::new(
            HuggingFaceTokenizer::from_file(tokenizer_path.to_str().expect("utf-8 tokenizer path"))
                .expect("load the DeepSeek-V4.1 tokenizer"),
        ) as Arc<dyn Tokenizer>;
        // The tokenizer's source is the checkpoint directory: the multimodal
        // config registry reads `config.json` (image_token_id) from it.
        let source = dir.to_string_lossy().into_owned();
        tokenizer_registry
            .load(
                "deepseek-v41",
                MODEL,
                &source,
                || async move { Ok(tokenizer) },
            )
            .await
            .expect("register the DeepSeek-V4.1 tokenizer");
        let multimodal = with_multimodal.then(|| {
            Arc::new(
                MultimodalComponents::new(
                    Arc::new(MultimodalConfigRegistry::new()),
                    None,
                    None,
                    &MultimodalSettings::default(),
                )
                .expect("multimodal components"),
            )
        });
        Arc::new(SharedComponents {
            templates: Arc::new(Templates::new(tokenizer_registry.clone())),
            tokenizer_registry,
            worker_registry,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: utils::ParserResolver::disabled(),
            multimodal,
            tokenizer_autoload: true,
            tokenizer_load_failures: Arc::default(),
            parser: crate::config::ParserChoice::Legacy,
        })
    }

    /// Status, body, and the input ids and placeholders of every attempt a
    /// chat request produced.
    type ChatRun = (
        http::StatusCode,
        bytes::Bytes,
        Vec<Vec<u32>>,
        Vec<PlaceholderRanges>,
    );

    /// Runs one chat request against a recording TokenSpeed stub.
    async fn v41_run_chat(
        dir: &Path,
        request: serde_json::Value,
        with_multimodal: bool,
    ) -> ChatRun {
        v41_run_chat_on(dir, request, with_multimodal, &[]).await
    }

    /// [`v41_run_chat`] with `worker_labels` on the stub's worker.
    async fn v41_run_chat_on(
        dir: &Path,
        request: serde_json::Value,
        with_multimodal: bool,
        worker_labels: &[(&str, &str)],
    ) -> ChatRun {
        let seen_ids = Arc::new(Mutex::new(Vec::new()));
        let seen_mm = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_stub(GatedScheduler {
            seen_input_ids: Arc::clone(&seen_ids),
            seen_mm_placeholders: Arc::clone(&seen_mm),
            ..Default::default()
        })
        .await;
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker_with_labels(&worker_registry, port, WorkerType::Regular, worker_labels);
        let deps = PipelineDeps::pair(
            worker_registry.clone(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline =
            RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).expect("chat pipeline");
        let components = v41_components(dir, worker_registry, with_multimodal).await;
        let request: Arc<ChatCompletionRequest> =
            Arc::new(serde_json::from_value(request).expect("chat request"));
        let response = pipeline
            .execute_chat(
                request,
                None,
                MODEL.to_string(),
                components,
                None,
                None,
                None,
                None,
            )
            .await;
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain body");
        let ids = seen_ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let mm = seen_mm
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        (status, body, ids, mm)
    }

    /// The prompt ids the worker receives equal the reference encoder's ids
    /// for a recorded multi-turn chat-mode case, and a `continue_final_message`
    /// request keeps its trailing assistant message and renders it the
    /// reference way (no EOS, no generation header) instead of as a prefix.
    /// The thinking-mode case is the one recorded at the checkpoint's current
    /// revision: the tokenizer directory here carries no encoder, so the
    /// renderer writes that revision's effort budget.
    #[tokio::test]
    async fn deepseek_v41_prompt_ids_match_the_reference_through_the_pipeline() {
        let Some(dir) = deepseek_v41_model_dir() else {
            skip_no_tokenizer();
            return;
        };

        let (case, expected) = v41_fixture("hf_2");
        let (status, body, ids, _) = v41_run_chat(
            &dir,
            serde_json::json!({
                "model": MODEL,
                "messages": case["messages"],
                "chat_template_kwargs": {"thinking": false, "drop_thinking": true},
                "max_tokens": 1,
            }),
            false,
        )
        .await;
        assert_eq!(
            status,
            http::StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&body)
        );
        assert_eq!(
            ids,
            vec![expected],
            "hf_2: prompt ids differ from the reference"
        );

        let (case, expected) = v41_fixture("continue_final_message_current");
        let (status, body, ids, _) = v41_run_chat(
            &dir,
            serde_json::json!({
                "model": MODEL,
                "messages": case["messages"],
                "continue_final_message": true,
                "chat_template_kwargs": {"thinking": true, "drop_thinking": true},
                "max_tokens": 1,
            }),
            false,
        )
        .await;
        assert_eq!(
            status,
            http::StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&body)
        );
        assert_eq!(
            ids,
            vec![expected],
            "continue_final_message: prompt ids differ from the reference's no-EOS rendering"
        );
    }

    /// A renderer validation error (an effort level the reference rejects) is
    /// the client's mistake, so it surfaces as 400, not 500.
    #[tokio::test]
    async fn deepseek_v41_invalid_effort_is_a_bad_request() {
        let Some(dir) = deepseek_v41_model_dir() else {
            skip_no_tokenizer();
            return;
        };
        let (status, body, ids, _) = v41_run_chat(
            &dir,
            serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": "turbo",
                "max_tokens": 1,
            }),
            false,
        )
        .await;
        let body = String::from_utf8_lossy(&body);
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        assert!(body.contains("Invalid reasoning effort"), "{body}");
        assert!(ids.is_empty(), "nothing must reach the worker");
    }

    /// `medium` is a level of the checkpoint's own chat template (budget
    /// 62): the top-level field and the template kwarg both render it
    /// instead of refusing the request.
    #[tokio::test]
    async fn deepseek_v41_medium_effort_renders_the_checkpoints_budget() {
        let Some(dir) = deepseek_v41_model_dir() else {
            skip_no_tokenizer();
            return;
        };
        let tokenizer = HuggingFaceTokenizer::from_file(
            dir.join("tokenizer.json")
                .to_str()
                .expect("utf-8 tokenizer path"),
        )
        .expect("load the DeepSeek-V4.1 tokenizer");
        for request in [
            serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": "medium",
                "max_tokens": 1,
            }),
            serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "chat_template_kwargs": {"reasoning_effort": "medium"},
                "max_tokens": 1,
            }),
        ] {
            let (status, body, ids, _) = v41_run_chat(&dir, request.clone(), false).await;
            assert_eq!(
                status,
                http::StatusCode::OK,
                "{request}: {}",
                String::from_utf8_lossy(&body)
            );
            let prompt = tokenizer.decode(&ids[0], false).expect("decode the prompt");
            assert!(
                prompt.contains("Reasoning Effort: 62 (range"),
                "{request}: {prompt}"
            );
        }
    }

    /// One image expands to exactly the reference span: `<｜deepseek_image｜>`
    /// is inlined at the part's position by the renderer and replaced by
    /// `image_token_id` repeated once per span position (189 for the corn
    /// example), and the worker is told that span as the placeholder range.
    #[tokio::test]
    async fn deepseek_v41_image_expands_to_the_reference_span() {
        let Some(dir) = deepseek_v41_model_dir() else {
            skip_no_tokenizer();
            return;
        };
        let data_url = format!("data:image/png;base64,{}", BASE64.encode(V41_CORN_PNG));
        let (status, body, ids, mm) = v41_run_chat(
            &dir,
            serde_json::json!({
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this picture?"},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }],
                "chat_template_kwargs": {"thinking": false},
                "max_tokens": 1,
            }),
            true,
        )
        .await;
        assert_eq!(
            status,
            http::StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&body)
        );
        assert_eq!(ids.len(), 1);
        let prompt = &ids[0];
        let count = prompt
            .iter()
            .filter(|&&id| id == V41_IMAGE_TOKEN_ID)
            .count();
        assert_eq!(count, V41_CORN_TOKENS, "image span length");
        let first = prompt
            .iter()
            .position(|&id| id == V41_IMAGE_TOKEN_ID)
            .expect("the image span is present");
        assert!(
            prompt[first..first + V41_CORN_TOKENS]
                .iter()
                .all(|&id| id == V41_IMAGE_TOKEN_ID),
            "the image span is contiguous"
        );
        assert_eq!(mm, vec![vec![(first as u32, V41_CORN_TOKENS as u32)]]);
    }

    /// A dispatch the engine answered with RESOURCE_EXHAUSTED is counted as
    /// the rate-limit error the client reads in the 429 body, not as an
    /// internal error of the router.
    #[test]
    fn a_dispatch_the_engine_throttled_is_counted_as_a_rate_limit_error() {
        let worker_registry = Arc::new(WorkerRegistry::new());
        // The model is served: its errors are counted under its own id (an
        // unserved name is counted under `other`).
        register_worker(&worker_registry, 1, WorkerType::Regular);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let failure = Status::resource_exhausted("kv cache full")
            .to_http_error("engine_busy", "kv cache full".to_string());
        assert_eq!(failure.status(), http::StatusCode::TOO_MANY_REQUESTS);

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            pipeline.record_error(Some(metrics_labels::ENDPOINT_COMPLETIONS), MODEL, &failure);
        });
        let rendered = handle.render();
        let line = rendered
            .lines()
            .find(|line| {
                line.starts_with("smg_router_request_errors_total{")
                    && line.contains(&format!("model=\"{MODEL}\""))
                    && line.contains(&format!(
                        "endpoint=\"{}\"",
                        metrics_labels::ENDPOINT_COMPLETIONS
                    ))
            })
            .unwrap_or_else(|| panic!("a router error is counted:\n{rendered}"));
        assert!(line.contains("error_type=\"rate_limit_error\""), "{line}");
        assert!(line.contains("reason=\"overloaded\""), "{line}");
        assert!(line.ends_with(" 1"), "{line}");
    }

    /// Two dispatch failures of one class with different causes are two
    /// series of the error family, one per `reason`, that sum to the count
    /// the family had before the label existed; the gateway's own refusal
    /// carries the reason its code names.
    #[test]
    fn a_router_error_carries_the_reason_behind_its_class() {
        use crate::{observability::metrics::test_support::render_with_recorder, routers::error};

        let worker_registry = Arc::new(WorkerRegistry::new());
        register_worker(&worker_registry, 1, WorkerType::Regular);
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let failed =
            |status: Status| status.to_http_error("start_generation_failed", "failed".to_string());
        let rendered = render_with_recorder(|| {
            for failure in [
                failed(Status::unavailable("draining")),
                failed(Status::cancelled("stream reset")),
                failed(Status::cancelled("connection closed")),
                error::service_unavailable(
                    placement::NO_AVAILABLE_WORKERS_CODE,
                    "every worker is down",
                ),
            ] {
                pipeline.record_error(Some(metrics_labels::ENDPOINT_COMPLETIONS), MODEL, &failure);
            }
        });
        let samples: Vec<&str> = rendered
            .lines()
            .filter(|line| {
                line.starts_with("smg_router_request_errors_total{")
                    && line.contains(&format!("model=\"{MODEL}\""))
            })
            .collect();
        let value = |error_type: &str, reason: &str| -> u64 {
            let line = samples
                .iter()
                .find(|line| {
                    line.contains(&format!("error_type=\"{error_type}\""))
                        && line.contains(&format!("reason=\"{reason}\""))
                })
                .unwrap_or_else(|| panic!("a {error_type}/{reason} series:\n{rendered}"));
            line.rsplit(' ').next().unwrap().parse().unwrap()
        };
        assert_eq!(samples.len(), 3, "{rendered}");
        assert_eq!(value("backend_error", "engine_unavailable"), 1);
        assert_eq!(value("backend_error", "transport_disconnect"), 2);
        assert_eq!(value("unavailable", "no_available_workers"), 1);
    }

    /// The same image to the same model served by a worker without a vision
    /// encoder: refused with a 400 in preparation, nothing preprocessed or
    /// dispatched (the engine would admit the request and never schedule it).
    #[tokio::test]
    async fn deepseek_v41_image_to_a_language_model_only_worker_is_refused() {
        let Some(dir) = deepseek_v41_model_dir() else {
            skip_no_tokenizer();
            return;
        };
        let data_url = format!("data:image/png;base64,{}", BASE64.encode(V41_CORN_PNG));
        let (status, body, ids, mm) = v41_run_chat_on(
            &dir,
            serde_json::json!({
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this picture?"},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }],
                "chat_template_kwargs": {"thinking": false},
                "max_tokens": 1,
            }),
            true,
            LANGUAGE_MODEL_ONLY,
        )
        .await;
        let body = String::from_utf8_lossy(&body);
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        assert!(body.contains("multimodal_not_supported"), "{body}");
        assert!(body.contains("language model only"), "{body}");
        assert!(ids.is_empty(), "nothing must reach the worker");
        assert!(mm.is_empty(), "nothing must be preprocessed");
    }

    /// The same image with the worker labels discovery derives from each
    /// servicer's `GetModelInfo`: a servicer that left `supports_vision`
    /// unset (SGLang, TokenSpeed) gives no verdict, so the image is
    /// preprocessed and dispatched as to any vision worker; vLLM's deliberate
    /// `false` refuses it.
    #[tokio::test]
    async fn deepseek_v41_image_follows_the_servicer_vision_verdict() {
        let Some(dir) = deepseek_v41_model_dir() else {
            skip_no_tokenizer();
            return;
        };
        let request = || {
            serde_json::json!({
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this picture?"},
                        {"type": "image_url", "image_url": {"url": format!(
                            "data:image/png;base64,{}", BASE64.encode(V41_CORN_PNG))}}
                    ]
                }],
                "chat_template_kwargs": {"thinking": false},
                "max_tokens": 1,
            })
        };
        for (name, info) in [
            ("sglang", ModelInfo::Sglang(Box::default())),
            ("tokenspeed", ModelInfo::TokenSpeed(Box::default())),
        ] {
            let labels = labels_of(&info);
            let (status, body, ids, mm) =
                v41_run_chat_on(&dir, request(), true, &label_pairs(&labels)).await;
            assert_eq!(
                status,
                http::StatusCode::OK,
                "{name}: {}",
                String::from_utf8_lossy(&body)
            );
            assert_eq!(ids.len(), 1, "{name}: dispatched once");
            assert_eq!(
                mm.iter().map(Vec::len).sum::<usize>(),
                1,
                "{name}: the image span reached the worker"
            );
        }

        let labels = labels_of(&ModelInfo::Vllm(vllm_proto::GetModelInfoResponse {
            supports_vision: false,
            ..Default::default()
        }));
        let (status, body, ids, _) =
            v41_run_chat_on(&dir, request(), true, &label_pairs(&labels)).await;
        let body = String::from_utf8_lossy(&body);
        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        assert!(body.contains("multimodal_not_supported"), "{body}");
        assert!(ids.is_empty(), "nothing must reach the worker");
    }

    /// The one `smg_worker_errors_total` sample of the regular gRPC workers.
    fn regular_worker_error_line(rendered: &str) -> &str {
        rendered
            .lines()
            .find(|line| {
                line.starts_with("smg_worker_errors_total{")
                    && line.contains("worker_type=\"regular\"")
                    && line.contains("connection_mode=\"grpc\"")
            })
            .unwrap_or_else(|| panic!("a worker error is counted:\n{rendered}"))
    }

    /// Every attempt a worker refuses before the first token is one sample of
    /// `smg_worker_errors_total` under the worker's labels, however the
    /// breaker is charged (once per request): three UNAVAILABLE starts of one
    /// request are three worker errors and one breaker failure.
    #[test]
    fn a_refusal_before_the_first_token_counts_against_the_worker_once_per_attempt() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async {
                let seen = Arc::new(Mutex::new(Vec::new()));
                let port = spawn_stub(GatedScheduler {
                    fail_always: true,
                    seen_request_ids: Arc::clone(&seen),
                    ..Default::default()
                })
                .await;
                let worker_registry = Arc::new(WorkerRegistry::new());
                let worker = register_worker_with_breaker(&worker_registry, port, 5);
                let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
                let components = components(worker_registry).await;

                let response = run_completion(pipeline, components, &fast_retry_config(3)).await;

                assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
                assert_eq!(calls_seen(&seen), 3, "three attempts on the one worker");
                assert_eq!(
                    worker.circuit_breaker_state(),
                    CircuitState::Closed,
                    "one failed request is one breaker failure"
                );
            });
        });
        let rendered = handle.render();
        let line = regular_worker_error_line(&rendered);
        assert!(line.contains("error_type=\"backend_error\""), "{line}");
        assert!(line.ends_with(" 3"), "one sample per attempt: {line}");
    }

    /// A worker stream that fails after its first frame, read into a
    /// non-streamed answer, is the worker's failure as it is for a streamed
    /// one: one sample of `smg_worker_errors_total`, `engine_error` for the
    /// engine's own report.
    #[test]
    fn a_worker_stream_that_fails_mid_way_counts_against_the_worker_for_a_non_streamed_answer() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            rt.block_on(async {
                let port = spawn_stub(GatedScheduler {
                    fail_mid_stream: true,
                    ..Default::default()
                })
                .await;
                let worker_registry = Arc::new(WorkerRegistry::new());
                register_worker(&worker_registry, port, WorkerType::Regular);
                let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
                let components = components(worker_registry).await;

                let response = run_completion(pipeline, components, &fast_retry_config(1)).await;

                assert_eq!(
                    response.status(),
                    http::StatusCode::INTERNAL_SERVER_ERROR,
                    "the engine's error ends a non-streamed answer with a 500"
                );
            });
        });
        let rendered = handle.render();
        let line = regular_worker_error_line(&rendered);
        assert!(line.contains("error_type=\"engine_error\""), "{line}");
        assert!(line.ends_with(" 1"), "{line}");
    }

    /// A completion for a model nobody serves and no tokenizer knows is
    /// refused by the preparation stage with the 404, before selection: the
    /// pipeline reports it once (the response marked, so the stage line and
    /// the access log keep quiet) and writes no ERROR event. The one refusal
    /// path a stage-level test cannot reach.
    #[tokio::test]
    async fn a_404_from_the_preparation_stage_is_reported_once_and_marked() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct ErrorCounter(Arc<AtomicUsize>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for ErrorCounter {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if *event.metadata().level() == tracing::Level::ERROR {
                    self.0.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        let errors = Arc::new(AtomicUsize::new(0));
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry().with(ErrorCounter(Arc::clone(&errors))),
        );
        let worker_registry = Arc::new(WorkerRegistry::new());
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let components = Arc::new(SharedComponents {
            templates: Arc::new(Templates::new(Arc::new(TokenizerRegistry::new()))),
            tokenizer_registry: Arc::new(TokenizerRegistry::new()),
            worker_registry,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: utils::ParserResolver::disabled(),
            multimodal: None,
            tokenizer_autoload: true,
            tokenizer_load_failures: Arc::default(),
            parser: crate::config::ParserChoice::Legacy,
        });

        let response = pipeline
            .execute_completion(
                completion_request(false),
                None,
                "nobody-serves-this".to_string(),
                components,
                None,
                None,
                Some(&fast_retry_config(1)),
            )
            .await;

        assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            placement::MODEL_NOT_FOUND_CODE
        );
        assert!(
            response.extensions().get::<ReportedRefusal>().is_some(),
            "the pipeline reported the 404, so the stage line and the access log keep quiet"
        );
        assert_eq!(
            errors.load(Ordering::Relaxed),
            0,
            "a model nobody serves is the client's mistake, not an error of the gateway"
        );
    }

    /// A refusal the gateway answers on its own (every worker of the model
    /// unavailable) is counted as `unavailable`, not as a backend's error: no
    /// worker was contacted.
    #[test]
    fn a_refusal_for_a_drained_pool_is_counted_as_unavailable_not_as_a_backend_error() {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let pipeline = completion_pipeline(&worker_registry, Mode::Regular);
        let refusal = placement::no_available_workers("every worker is down");
        assert_eq!(refusal.status(), http::StatusCode::SERVICE_UNAVAILABLE);

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            pipeline.record_error(Some(metrics_labels::ENDPOINT_COMPLETIONS), MODEL, &refusal);
        });
        let rendered = handle.render();
        let line = rendered
            .lines()
            .find(|line| {
                // The model label is bounded to the served models (an
                // unregistered one reads "other"), so the line is found by
                // its router, as the HTTP and PD tests find theirs.
                line.starts_with("smg_router_request_errors_total{")
                    && line.contains(&format!("router_type=\"{}\"", metrics_labels::ROUTER_GRPC))
            })
            .unwrap_or_else(|| panic!("a router error is counted:\n{rendered}"));
        assert!(line.contains("error_type=\"unavailable\""), "{line}");
        assert!(line.ends_with(" 1"), "{line}");
    }
}

#[cfg(test)]
mod rate_limit_reserve_tests {
    use bytes::Bytes;
    use llm_multimodal::registry::transcription::FAMILIES;
    use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
    use openai_protocol::{
        decisions::{DecisionInputTokensDetails, DecisionOutputTokensDetails},
        generate::{GenerateFinishReason, GenerateFinishType, GenerateMetaInfo, GenerateRequest},
        transcription::{AudioFile, TranscriptionRequest},
    };
    use serde_json::json;

    use super::*;
    use crate::{
        config::types::RouterConfig,
        rate_limit::{RateLimitManager, Reservation, ReserveRequest},
        routers::grpc::{
            regular::symphony::template::Templates, spec::TranscriptionResponseFormat,
            ProcessedMessages,
        },
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
            templates: Arc::new(Templates::new(tokenizer_registry.clone())),
            tokenizer_registry,
            worker_registry,
            tool_parser_factory: ToolParserFactory::default(),
            reasoning_parser_factory: ReasoningParserFactory::default(),
            parser_resolver: utils::ParserResolver::disabled(),
            multimodal: None,
            tokenizer_autoload: true,
            tokenizer_load_failures: Arc::default(),
            parser: crate::config::ParserChoice::Legacy,
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
        // The Responses iterations set no rate_limit_cell.
        let manager = manager_with_tokens_per_minute(6);
        let components = test_components().await;
        let mut ctx = ctx_with_tokens(components, 5);

        let stage = RateLimitReserveStage::new(Some(manager));
        assert!(stage.execute(&mut ctx).await.is_ok());
    }

    #[tokio::test]
    async fn settle_reservation_with_zero_usage_refunds_the_full_estimate() {
        // Budget 6; reserve 5 leaves 1. A response without usage settles
        // with all zeros (see `recorded_settlement`, `completion_settlement`
        // and `generate_settlement`) -- that must refund the full 5-token
        // debit back to the tenant, not silently keep it (nothing was
        // actually consumed) and not leave the reservation stuck open
        // (which would leak budget forever).
        let manager = manager_with_tokens_per_minute(6);
        let components = test_components().await;
        let mut ctx = ctx_with_tokens(components, 5);
        ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

        let stage = RateLimitReserveStage::new(Some(manager.clone()));
        assert!(stage.execute(&mut ctx).await.is_ok());

        RequestPipeline::settle_reservation(
            ctx.input.rate_limit_cell.as_deref(),
            UsageSettlement::default(),
        )
        .await;

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

    /// The cached and reasoning split of a settlement is carried for the
    /// consumers that price or report it; the true-up charges the counts.
    #[tokio::test]
    async fn settle_reservation_charges_the_counts_and_not_their_split() {
        let manager = manager_with_tokens_per_minute(100);
        let components = test_components().await;
        let mut ctx = ctx_with_tokens(components, 5);
        ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));

        let stage = RateLimitReserveStage::new(Some(manager.clone()));
        assert!(stage.execute(&mut ctx).await.is_ok());

        RequestPipeline::settle_reservation(
            ctx.input.rate_limit_cell.as_deref(),
            UsageSettlement {
                actual_input_tokens: 5,
                completion_tokens: 10,
                cached_input_tokens: 4,
                reasoning_tokens: 3,
            },
        )
        .await;

        // 100 - 15 = 85 remain: the split neither charges twice nor refunds.
        let check = |tokens: u32| ReserveRequest {
            request_charge_id: uuid::Uuid::now_v7(),
            tenant_key: TenantKey::new("test-tenant"),
            model_id: Some(MODEL.to_string()),
            estimated_input_tokens: tokens,
        };
        assert!(matches!(
            manager.reserve(check(85)).await,
            Reservation::Admitted(_)
        ));
        assert!(matches!(
            manager.reserve(check(1)).await,
            Reservation::Denied { .. }
        ));
    }

    /// The non-streamed completion settles from its usage object: the
    /// engine's counters, their details included; zeros without one.
    #[test]
    fn completion_settlement_carries_the_usage_details() {
        let usage = Usage::from_counts(7, 5)
            .with_cached_tokens(3)
            .with_reasoning_tokens(2);
        assert_eq!(
            RequestPipeline::completion_settlement(Some(&usage)),
            UsageSettlement {
                actual_input_tokens: 7,
                completion_tokens: 5,
                cached_input_tokens: 3,
                reasoning_tokens: 2,
            }
        );
        assert_eq!(
            RequestPipeline::completion_settlement(Some(&Usage::from_counts(7, 5))),
            UsageSettlement {
                actual_input_tokens: 7,
                completion_tokens: 5,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            }
        );
        assert_eq!(
            RequestPipeline::completion_settlement(None),
            UsageSettlement::default()
        );
    }

    /// The non-streamed generate answer settles its shared prompt and cache
    /// hit once, the greatest any choice reports, and sums each choice's
    /// completion and reasoning counts; an empty answer settles zeros.
    #[test]
    fn generate_settlement_takes_the_shared_prompt_once_and_sums_the_choices() {
        let choice = |completion_tokens: u32, reasoning_tokens: Option<u32>, cached_tokens: u32| {
            GenerateResponse {
                text: String::new(),
                output_ids: Vec::new(),
                meta_info: GenerateMetaInfo {
                    id: "gen".to_string(),
                    finish_reason: GenerateFinishReason::Stop {
                        finish_type: GenerateFinishType::Stop,
                    },
                    prompt_tokens: 9,
                    weight_version: "default".to_string(),
                    input_token_logprobs: None,
                    output_token_logprobs: None,
                    completion_tokens,
                    cached_tokens,
                    reasoning_tokens,
                    e2e_latency: 0.0,
                    matched_stop: None,
                },
            }
        };
        assert_eq!(
            RequestPipeline::generate_settlement(&[choice(3, Some(1), 4), choice(5, None, 4)]),
            UsageSettlement {
                actual_input_tokens: 9,
                completion_tokens: 8,
                cached_input_tokens: 4,
                reasoning_tokens: 1,
            }
        );
        // The choices report different cache hits: the settlement takes the
        // greatest, as the streamed answer settles.
        assert_eq!(
            RequestPipeline::generate_settlement(&[choice(3, Some(1), 2), choice(5, None, 6)])
                .cached_input_tokens,
            6
        );
        assert_eq!(
            RequestPipeline::generate_settlement(&[]),
            UsageSettlement::default()
        );
    }

    /// A decision batch settles its input tokens and the cache and reasoning
    /// counters its usage carries, no completion; a count past `u32` saturates.
    #[test]
    fn decisions_settlement_carries_the_usage_counters() {
        let usage = DecisionUsage {
            input_tokens: 11,
            output_tokens: 0,
            total_tokens: 11,
            input_tokens_details: DecisionInputTokensDetails {
                cached_tokens: 2,
                cache_write_tokens: 0,
            },
            output_tokens_details: DecisionOutputTokensDetails {
                reasoning_tokens: 1,
            },
        };
        assert_eq!(
            RequestPipeline::decisions_settlement(&usage),
            UsageSettlement {
                actual_input_tokens: 11,
                completion_tokens: 0,
                cached_input_tokens: 2,
                reasoning_tokens: 1,
            }
        );
        let beyond = DecisionUsage {
            input_tokens: u64::from(u32::MAX) + 1,
            ..usage
        };
        assert_eq!(
            RequestPipeline::decisions_settlement(&beyond).actual_input_tokens,
            u32::MAX
        );
    }

    fn with_tenant_and_cell(mut ctx: RequestContext) -> RequestContext {
        ctx.input.tenant_request_meta = Some(TenantRequestMeta::new(TenantKey::new("test-tenant")));
        ctx.input.rate_limit_cell = Some(Arc::new(RateLimitCell::new()));
        ctx
    }

    /// An embeddings request after its preparation: `num_tokens` input tokens.
    fn embedding_ctx(components: Arc<SharedComponents>, num_tokens: usize) -> RequestContext {
        let request: EmbeddingRequest =
            serde_json::from_value(json!({"model": MODEL, "input": "Hello"})).unwrap();
        let mut ctx =
            RequestContext::for_embedding(Arc::new(request), None, MODEL.to_string(), components);
        ctx.state.preparation = Some(PreparationOutput::Embedding {
            original_text: "Hello".to_string(),
            token_ids: vec![0; num_tokens],
        });
        with_tenant_and_cell(ctx)
    }

    /// A classification request after its preparation, which it shares with
    /// embeddings: `num_tokens` input tokens.
    fn classify_ctx(components: Arc<SharedComponents>, num_tokens: usize) -> RequestContext {
        let request: ClassifyRequest =
            serde_json::from_value(json!({"model": MODEL, "input": "Hello"})).unwrap();
        let mut ctx =
            RequestContext::for_classify(Arc::new(request), None, MODEL.to_string(), components);
        ctx.state.preparation = Some(PreparationOutput::Embedding {
            original_text: "Hello".to_string(),
            token_ids: vec![0; num_tokens],
        });
        with_tenant_and_cell(ctx)
    }

    /// A transcription request after its preparation: the chat-shaped
    /// request the family's convention made, rendered to `num_tokens` tokens.
    fn transcription_ctx(components: Arc<SharedComponents>, num_tokens: usize) -> RequestContext {
        let request = TranscriptionRequest {
            model: MODEL.to_string(),
            ..Default::default()
        };
        let audio = AudioFile {
            bytes: Bytes::from_static(b"not audio"),
            file_name: "clip.wav".to_string(),
            content_type: None,
        };
        let chat_request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap();
        let mut ctx = RequestContext::for_transcription(
            Arc::new(request),
            Arc::new(audio),
            None,
            MODEL.to_string(),
            components,
        );
        ctx.state.preparation = Some(PreparationOutput::Transcription {
            token_ids: vec![0; num_tokens],
            processed_messages: ProcessedMessages {
                text: String::new(),
                unbilled_prompt_tokens: 0,
                continued_final_message: false,
            },
            chat_request: Arc::new(chat_request),
            format: TranscriptionResponseFormat::Json,
            family: FAMILIES[0],
        });
        with_tenant_and_cell(ctx)
    }

    /// The reservation of `endpoint`, as the chat endpoint takes it: the
    /// estimate is the preparation's input token count; a budget that cannot
    /// hold it refuses with the chat endpoint's 429, a budget that holds it
    /// once admits the first request and refuses the second.
    async fn assert_reserves_like_chat(
        endpoint: &str,
        make: fn(Arc<SharedComponents>, usize) -> RequestContext,
    ) {
        let components = test_components().await;

        // Too small for one 5-token reservation.
        let stage = RateLimitReserveStage::new(Some(manager_with_tokens_per_minute(3)));
        let mut ctx = make(components.clone(), 5);
        let response = stage.execute(&mut ctx).await.expect_err(endpoint);
        assert_eq!(
            response.status(),
            http::StatusCode::TOO_MANY_REQUESTS,
            "{endpoint}"
        );
        assert!(
            matches!(
                ctx.input.rate_limit_cell.as_ref().unwrap().peek(),
                Some(RateLimitOutcome::Denied)
            ),
            "{endpoint}: the denial is cached on the cell"
        );

        // Room for one 5-token reservation, not two (5 + 5 > 6).
        let stage = RateLimitReserveStage::new(Some(manager_with_tokens_per_minute(6)));
        let mut first = make(components.clone(), 5);
        assert!(stage.execute(&mut first).await.is_ok(), "{endpoint}");
        assert!(
            matches!(
                first.input.rate_limit_cell.as_ref().unwrap().peek(),
                Some(RateLimitOutcome::Admitted(_))
            ),
            "{endpoint}: the admission is cached on the cell"
        );
        let mut second = make(components, 5);
        let response = stage.execute(&mut second).await.expect_err(endpoint);
        assert_eq!(
            response.status(),
            http::StatusCode::TOO_MANY_REQUESTS,
            "{endpoint}: the first request's tokens are debited"
        );
    }

    #[tokio::test]
    async fn an_embeddings_request_reserves_its_input_tokens_like_a_chat_request() {
        assert_reserves_like_chat("embeddings", embedding_ctx).await;
    }

    #[tokio::test]
    async fn a_classify_request_reserves_its_input_tokens_like_a_chat_request() {
        assert_reserves_like_chat("classify", classify_ctx).await;
    }

    #[tokio::test]
    async fn a_transcription_request_reserves_its_prompt_tokens_like_a_chat_request() {
        assert_reserves_like_chat("transcription", transcription_ctx).await;
    }

    /// Embeddings and classification settle the prompt tokens the engine
    /// counted and nothing else.
    #[test]
    fn prompt_only_settlement_charges_the_prompt_alone() {
        assert_eq!(
            RequestPipeline::prompt_only_settlement(7),
            UsageSettlement {
                actual_input_tokens: 7,
                completion_tokens: 0,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            }
        );
    }
}

#[cfg(test)]
mod model_label_tests {
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        observability::metrics::test_support::render_with_recorder,
        worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
    };

    /// smg-lab #288: the pipeline's request and error records carry the
    /// served model's id, or `other` for a model no worker serves, so a
    /// client cannot mint series with unknown names.
    #[test]
    fn the_pipeline_counts_an_unserved_model_under_other() {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker = BasicWorkerBuilder::new("grpc://worker:30000")
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Sglang)
            .model(ModelCard::new("m1"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        worker_registry.register(Arc::new(worker)).unwrap();
        let deps = PipelineDeps::pair(
            worker_registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            None,
            None,
            Duration::from_secs(60),
        );
        let pipeline =
            RequestPipeline::build(Endpoint::Chat, Mode::Regular, &deps).expect("chat pipeline");
        let rendered = render_with_recorder(|| {
            pipeline.record_request(metrics_labels::ENDPOINT_CHAT, "ghost-1", false);
            pipeline.record_request(metrics_labels::ENDPOINT_CHAT, "m1", false);
            pipeline.record_error(
                Some(metrics_labels::ENDPOINT_CHAT),
                "ghost-1",
                &error::model_not_found("ghost-1"),
            );
        });
        assert!(!rendered.contains("ghost-"), "{rendered}");
        let has = |family: &str, label: &str| {
            rendered
                .lines()
                .any(|line| line.starts_with(&format!("{family}{{")) && line.contains(label))
        };
        assert!(
            has("smg_router_requests_total", r#"model="other""#),
            "{rendered}"
        );
        assert!(
            has("smg_router_requests_total", r#"model="m1""#),
            "{rendered}"
        );
        assert!(
            has("smg_router_request_errors_total", r#"model="other""#),
            "{rendered}"
        );
    }
}
