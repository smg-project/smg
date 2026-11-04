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
        streaming, symphony,
    },
    spec::ResponseSpec,
    utils,
    utils::{error_reason_for_response, error_type_for_response},
};
use crate::{
    config::{types::RetryConfig, ParserChoice},
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
    /// `--grpc-request-timeout-secs`: the deadline of each attempt's calls to
    /// its workers and of the streams they open, counted from the dispatch;
    /// `None` leaves them unbounded.
    request_timeout: Option<Duration>,
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
            request_timeout: None,
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
            request_timeout: None,
        }
    }

    /// Bound each attempt's calls to its workers, and the streams they open,
    /// by `timeout` from the dispatch (`--grpc-request-timeout-secs`); zero
    /// leaves them unbounded.
    #[must_use]
    pub(crate) fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = (!timeout.is_zero()).then_some(timeout);
        self
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
            request_timeout: None,
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
    /// The deadline of each attempt's dispatch and streams, when set.
    request_timeout: Option<Duration>,
}

/// Outcome of one full pipeline run.
enum RunOutcome {
    /// Early response (streaming SSE) already produced by response processing.
    Early(Response),
    /// Buffered completion: the final response (or responses-iteration state)
    /// sits on the dispatch context; the Instant is the request's start, for
    /// the duration record.
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
            request_timeout: deps.request_timeout,
        })
    }

    /// The deadline each attempt's dispatch and streams run under, when set.
    #[cfg(test)]
    pub(crate) fn request_timeout(&self) -> Option<Duration> {
        self.request_timeout
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
            enforce_output_budget(
                &ctx.input.request_type,
                workers,
                &ctx.input.model_id,
                ctx.state.provider,
            )
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

        execute_plan(dctx, attempt_plan, last_attempt, self.request_timeout).await?;
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

    /// Full two-phase run. `metrics_endpoint` enables the request's
    /// router-request, error and duration records, one each per request as
    /// the HTTP routers record a route (the responses-internal entry points
    /// pass `None`: the Responses route records its own); `retry_config`
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
            let start = Instant::now();
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
                            self.record_duration(endpoint, &dctx.model_id, start);
                        }
                        return Ok(RunOutcome::Early(Self::routed_response(&dctx, response)));
                    }
                    Ok(None) => return Ok(RunOutcome::Final(dctx, start)),
                    Err(response) => response,
                };

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

                // The attempt's failure is the request's answer unless a retry
                // follows: counted once, here, under its class and reason, as
                // the HTTP routers count the answer of a route; the attempts
                // are the retry families' count.
                let retryable = retry_config.filter(|_| is_retryable_response(&failure));
                let Some(config) = retryable.filter(|_| !last_attempt) else {
                    self.record_error(metrics_endpoint, &dctx.model_id, &failure);
                    if retryable.is_some() {
                        if let Some(endpoint) = metrics_endpoint {
                            self.record_retries_exhausted(endpoint);
                        }
                    }
                    return Err(failure);
                };

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
                symphony::HEADER,
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
    /// dashboards read: recorded once per request here, at its entry (the
    /// attempts are the retry families' count), and by the router for an
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
    /// logical request reserved one ([`RateLimitCell::settle`]; no-op when
    /// the caller set no cell).
    async fn settle_reservation(cell: Option<&RateLimitCell>, usage: UsageSettlement) {
        if let Some(cell) = cell {
            cell.settle(usage).await;
        }
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
    /// whether the backend request started, with the parser header the
    /// started stream's answer carries. A run that fails before then hands
    /// its response to the barrier; the caller gets a placeholder the
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
                    // The parser that read the answer, as the Responses stream's
                    // own answer names it: a stream answers early, its header
                    // set by `routed_response`.
                    let parser = match &outcome {
                        RunOutcome::Early(response) => {
                            response.headers().get(symphony::HEADER).cloned()
                        }
                        RunOutcome::Final(dctx, _) => dctx
                            .parser
                            .map(|parser| http::HeaderValue::from_static(parser.name())),
                    };
                    let _ = startup.send(Ok(parser));
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
    /// Used by ALL non-streaming /v1/responses requests, once per iteration
    /// of a tool loop. Same stages as execute_chat(), the tenant rate-limit
    /// reservation included: every iteration is a generation of its own, so
    /// each takes the reservation a chat request takes (reserved at ingress
    /// under a charge id of its own, settled with the engine's counts, closed
    /// unsettled when the iteration fails). Two differences:
    /// 1. Returns the completion with the parser that read it (what the chat
    ///    route answers in `x-smg-parser`), as a Result for tool_loop composition
    /// 2. Disallows streaming (responses endpoint uses different SSE format)
    pub async fn execute_chat_for_responses(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<http::HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Result<(ChatCompletionResponse, Option<ParserChoice>), Response> {
        let rate_limit_cell = Arc::new(RateLimitCell::new());
        let run = {
            let mut ctx = RequestContext::for_chat(request, headers, model_id, components);
            ctx.input.tenant_request_meta = tenant_request_meta.as_ref().map(iteration_charge);
            ctx.input.rate_limit_cell = Some(rate_limit_cell.clone());
            self.run(ctx, None, None)
        };
        let result = match run.await {
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
                Some(FinalResponse::Chat(response)) => {
                    // The engine's counts the response stage recorded, as
                    // `execute_chat` settles them.
                    rate_limit_cell
                        .settle(Self::recorded_settlement(&dctx.response))
                        .await;
                    Ok((response, dctx.parser))
                }
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
        };
        if result.is_err() {
            // A failed iteration keeps its reserved debit, as a failed typed
            // request does (`GrpcRouter::execute_reserved`).
            rate_limit_cell.close_unsettled().await;
        }
        result
    }

    /// Execute one Harmony Responses API iteration through the pipeline.
    ///
    /// Returns ToolCallsFound (continue serving) or Completed (final
    /// response); called by harmony::responses::serve_harmony_responses()
    /// per iteration. Each iteration takes the tenant rate-limit reservation
    /// a chat request takes: reserved at ingress under a charge id of its own,
    /// settled with the engine's counts the response stage recorded, closed
    /// unsettled when the iteration fails.
    pub async fn execute_harmony_responses(
        &self,
        request: &openai_protocol::responses::ResponsesRequest,
        harmony_ctx: &ResponsesContext,
        tenant_request_meta: Option<TenantRequestMeta>,
    ) -> Result<harmony::ResponsesIterationResult, Response> {
        let rate_limit_cell = Arc::new(RateLimitCell::new());
        let run = {
            let mut ctx = RequestContext::for_responses(
                Arc::new(request.clone()),
                None,                  // No headers needed for internal pipeline execution
                request.model.clone(), // Model ID from request
                harmony_ctx.components.clone(),
            );
            ctx.input.tenant_request_meta = tenant_request_meta.as_ref().map(iteration_charge);
            ctx.input.rate_limit_cell = Some(rate_limit_cell.clone());
            self.run(ctx, None, None)
        };
        let result = match run.await {
            Ok(RunOutcome::Early(response)) => {
                error!(
                    function = "execute_harmony_responses",
                    "Unexpected early response during Responses iteration"
                );
                Err(response)
            }
            Ok(RunOutcome::Final(mut dctx, _)) => {
                match dctx.response.responses_iteration_result.take() {
                    Some(iteration_result) => {
                        rate_limit_cell
                            .settle(Self::recorded_settlement(&dctx.response))
                            .await;
                        Ok(iteration_result)
                    }
                    None => {
                        error!(
                            function = "execute_harmony_responses",
                            "No ResponsesIterationResult produced by pipeline"
                        );
                        Err(error::internal_error(
                            "no_responses_iteration_result",
                            "No ResponsesIterationResult produced by pipeline",
                        ))
                    }
                }
            }
            Err(response) => Err(response),
        };
        if result.is_err() {
            rate_limit_cell.close_unsettled().await;
        }
        result
    }

    /// Execute a Harmony Responses pipeline iteration with streaming support.
    ///
    /// Runs through dispatch and returns the raw ExecutionResult (with
    /// stream) and LoadGuards for token-level streaming processing, with the
    /// iteration's tenant rate-limit reservation (taken under a charge id of
    /// its own) and the router's stop obligations (the string `stop`s the
    /// engine cannot match, for the reader to scan for). The caller keeps
    /// load_guards alive until stream processing completes, then settles the
    /// reservation with the counts the stream brought
    /// ([`RateLimitCell::settle`]) or closes it when the stream broke
    /// ([`RateLimitCell::close_unsettled`]); a cell dropped before either
    /// (the client left) abandons the reservation, its debit kept.
    pub async fn execute_harmony_responses_streaming(
        &self,
        request: &openai_protocol::responses::ResponsesRequest,
        harmony_ctx: &ResponsesContext,
        tenant_request_meta: Option<TenantRequestMeta>,
        stream_start: Option<StreamStartupSender>,
    ) -> Result<
        (
            ExecutionResult,
            Option<LoadGuards>,
            Arc<RateLimitCell>,
            Vec<String>,
        ),
        Response,
    > {
        let rate_limit_cell = Arc::new(RateLimitCell::new());
        let run = {
            let mut ctx = RequestContext::for_responses(
                Arc::new(request.clone()),
                None,
                request.model.clone(),
                harmony_ctx.components.clone(),
            );
            ctx.input.tenant_request_meta = tenant_request_meta.as_ref().map(iteration_charge);
            ctx.input.rate_limit_cell = Some(rate_limit_cell.clone());
            self.run_signalling(ctx, None, None, stream_start)
        };
        let result = match run.await {
            Ok(RunOutcome::Early(response)) => {
                error!(
                    function = "execute_harmony_responses_streaming",
                    "Unexpected early response during streaming Responses"
                );
                Err(response)
            }
            Ok(RunOutcome::Final(mut dctx, _)) => match dctx.response.execution_result.take() {
                Some(execution_result) => {
                    let load_guards = dctx.load_guards.take();
                    // The string stops the engine will never match, for the
                    // Harmony reader to scan the channel text for, as the
                    // response stage hands them to a non-streamed iteration.
                    let router_stops = std::mem::take(&mut dctx.response.router_stop_obligations);
                    Ok((
                        execution_result,
                        load_guards,
                        rate_limit_cell.clone(),
                        router_stops,
                    ))
                }
                None => {
                    error!(
                        function = "execute_harmony_responses_streaming",
                        "No ExecutionResult produced by pipeline"
                    );
                    Err(error::internal_error(
                        "no_execution_result_produced",
                        "No ExecutionResult produced by pipeline",
                    ))
                }
            },
            Err(response) => Err(response),
        };
        if result.is_err() {
            rate_limit_cell.close_unsettled().await;
        }
        result
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
            parser: ParserChoice::Legacy,
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
mod tests;

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
            parser: ParserChoice::Legacy,
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
                generation_prompt: true,
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
