//! Worker selection stage: Select appropriate worker(s) based on routing mode

use std::{
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use axum::{
    http::{header::RETRY_AFTER, HeaderMap, HeaderValue},
    response::Response,
};
use tracing::{debug, error, warn};

use super::{helpers, PipelineStage};
use crate::{
    middleware::{tenant_resolution::AcceptedAt, ReportedRefusal, SHED_RETRY_AFTER_SECS},
    observability::{
        log_throttle::KeyedLogThrottle,
        metrics::{metrics_labels, Metrics},
    },
    policies::{CacheNamespace, LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo, WorkerLeg},
    routers::{
        common::{
            placement::{self, CandidateFilter, PairFailure, PlacementFailure, PlacementInputs},
            retry::mark_non_retryable,
        },
        error,
        grpc::{
            context::{
                DispatchContext, EncodeWorkerAssignment, RequestContext, RequestType,
                RoutingSnapshot, WireConstraint, WorkerSelection,
            },
            multimodal,
        },
        prefill_queue_full, prefill_queue_timeout,
    },
    worker::{
        acquire_prefill, ConnectionMode, ConnectionModeExt, HashRing, ModelWorkerSnapshot,
        PdPairIndex, PdWire, PrefillAcquireError, PrefillAdmission, PrefillAdmissionRejection,
        PrefillCandidateError, PrefillLoadGuard, PrefillSelectionContext, RoutingPool, RuntimeType,
        Worker, WorkerRegistry, WorkerType,
    },
};

/// Result type for PD worker pair selection: (prefill, decode, runtime_type)
type PdWorkerPair = (Arc<dyn Worker>, Arc<dyn Worker>, RuntimeType);

/// The candidate predicate of a request carrying media references: only
/// workers that process them themselves.
const ACCEPTS_MEDIA_REFS: CandidateFilter<'static> = &accepts_media_refs;

/// Why an EPD selection produced nothing. Only a full Prefill leg is told
/// apart, for admission; every other failure is judged per leg afterwards by
/// [`WorkerSelectionStage::selection_failure`].
#[derive(Clone, Copy)]
enum EpdSelectionFailure {
    Unavailable,
    PrefillAtCapacity,
}

impl PrefillCandidateError for EpdSelectionFailure {
    fn is_at_capacity(&self) -> bool {
        matches!(self, Self::PrefillAtCapacity)
    }
}

/// Why a prefill/decode selection produced nothing: the placement's own
/// verdict for the leg that failed, or a request carrying pixel media for a
/// model that pairs no prefill worker with a vision encoder.
#[derive(Debug)]
enum PdSelectionFailure {
    Placement(Box<PairFailure>),
    NoVisionCapablePrefill,
}

impl From<Box<PairFailure>> for PdSelectionFailure {
    fn from(failure: Box<PairFailure>) -> Self {
        Self::Placement(failure)
    }
}

impl PrefillCandidateError for PdSelectionFailure {
    fn is_at_capacity(&self) -> bool {
        matches!(self, Self::Placement(failure) if failure.is_at_capacity())
    }
}

/// Result type for EPD worker selection: (encode assignments, prefill, decode, runtime_type).
type EncodePrefillDecodeWorkerSelection = (
    Vec<EncodeWorkerAssignment>,
    Arc<dyn Worker>,
    Arc<dyn Worker>,
    RuntimeType,
);

/// Client-visible, gateway-owned code for a request refused ahead of worker
/// selection because it had already waited longer than the queue timeout
/// inside the gateway.
pub(crate) const SELECTION_QUEUE_TIMEOUT_ERROR_CODE: &str = "selection_queue_timeout";

/// Lines per second and per model the stage writes for refusals of a model
/// whose workers are all unavailable (`no_available_workers`). One line per
/// refusal, with the request id, the model and the pool's state, shows the
/// condition and whom it hits; past this rate the lines are counted and the
/// next one carries the count, while `smg_router_request_errors_total`
/// keeps the volume. A client retrying at once against a drained pool drew
/// about two thousand refusals a second, three ERROR lines each, before.
const NO_AVAILABLE_WORKERS_LINES_PER_SECOND: u32 = 10;

/// Models the refusal throttle tracks at once; past this the model seen
/// least recently is forgotten (a model that is down keeps refusing, so it
/// stays). One storming model never silences another's first line.
const NO_AVAILABLE_WORKERS_LOG_MODELS: usize = 64;

static NO_AVAILABLE_WORKERS_LOG: KeyedLogThrottle = KeyedLogThrottle::new(
    NO_AVAILABLE_WORKERS_LINES_PER_SECOND,
    Duration::from_secs(1),
    NO_AVAILABLE_WORKERS_LOG_MODELS,
);

/// Worker selection stage: Select appropriate worker(s) based on routing mode
pub(crate) struct WorkerSelectionStage {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    mode: WorkerSelectionMode,
    prefill_admission: Option<Arc<PrefillAdmission>>,
    /// The longest a request may have waited inside the gateway when it
    /// reaches selection; `None` leaves the wait unbounded.
    queue_timeout: Option<Duration>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkerSelectionMode {
    /// Regular mode: select single worker
    Regular,
    /// PD mode: select prefill + decode workers
    PrefillDecode,
    /// EPD mode: select encode + prefill + decode workers
    EncodePrefillDecode,
}

impl WorkerSelectionStage {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        mode: WorkerSelectionMode,
        prefill_admission: Option<Arc<PrefillAdmission>>,
    ) -> Self {
        Self {
            worker_registry,
            policy_registry,
            mode,
            prefill_admission,
            queue_timeout: None,
        }
    }

    /// Refuse a request that reaches selection more than `timeout` after the
    /// gateway accepted it (`--queue-timeout-secs`); zero leaves the wait
    /// unbounded.
    pub(crate) fn with_queue_timeout(mut self, timeout: Duration) -> Self {
        self.queue_timeout = (!timeout.is_zero()).then_some(timeout);
        self
    }

    /// The acceptance stamp this selection is bounded by, taken from the
    /// request's meta once: the first selection of a request takes it, a later
    /// pipeline run of the same request (a Responses tool loop iteration, with
    /// the same meta) finds it spent and runs unbounded, as it did not wait
    /// inside the gateway. `None` without a bound, a meta or a fresh stamp.
    fn acceptance_for_selection(&self, ctx: &RequestContext) -> Option<Instant> {
        self.queue_timeout?;
        ctx.input
            .tenant_request_meta
            .as_ref()?
            .extension::<AcceptedAt>()?
            .take_for_selection()
    }

    /// The 503 for a request held past the queue timeout since `accepted`
    /// (the admission grant where there is one, else the gateway's acceptance),
    /// before any worker is read: the admission bound caps how many requests
    /// the gateway holds, not for how long, and a gateway that falls behind
    /// its own ingress (tokenization, selection) would otherwise dispatch
    /// requests that waited minutes to workers that are not overloaded at all,
    /// so neither the overload shed nor the admission queue ever answers.
    /// Checked again after the prefill admission wait of a PD or EPD request,
    /// which can outlast the bound. Terminal for the retry layer, like every
    /// shed, with the same back-off hint as the admission queue's refusals.
    fn refuse_if_expired(&self, ctx: &RequestContext, accepted: Instant) -> Option<Response> {
        let timeout = self.queue_timeout?;
        let waited = waited_past_timeout(accepted, Instant::now(), timeout)?;
        Some(Self::queue_timeout_refusal(ctx, waited, timeout))
    }

    /// The 503 of a request that waited `waited`, past the queue timeout.
    fn queue_timeout_refusal(
        ctx: &RequestContext,
        waited: Duration,
        timeout: Duration,
    ) -> Response {
        Metrics::record_admission_rejected(metrics_labels::ADMISSION_REJECTED_SELECTION_TIMEOUT);
        debug!(
            model_id = %ctx.input.model_id,
            waited_secs = waited.as_secs_f64(),
            timeout_secs = timeout.as_secs_f64(),
            "Refused ahead of worker selection: waited longer than the queue timeout"
        );
        let mut response = error::service_unavailable(
            SELECTION_QUEUE_TIMEOUT_ERROR_CODE,
            format!(
                "The request waited {:.1} s inside the gateway before worker selection, longer \
                 than the queue timeout of {} s",
                waited.as_secs_f64(),
                timeout.as_secs()
            ),
        );
        response
            .headers_mut()
            .insert(RETRY_AFTER, HeaderValue::from(SHED_RETRY_AFTER_SECS));
        mark_non_retryable(&mut response);
        response
    }

    /// `admission`, a prefill admission wait, held to the remaining budget of
    /// the bound: when the budget runs out the waiter is dropped (its queue
    /// ticket goes with it) and the request is refused as one held past the
    /// queue timeout; without a bound the wait is the prefill queue's own.
    async fn admit_within_budget<T, F>(
        &self,
        ctx: &RequestContext,
        accepted: Option<Instant>,
        admission: F,
    ) -> Result<T, Response>
    where
        F: std::future::Future<Output = Result<T, Response>>,
    {
        let (Some(timeout), Some(accepted)) = (self.queue_timeout, accepted) else {
            return admission.await;
        };
        let budget = timeout.saturating_sub(accepted.elapsed());
        match tokio::time::timeout(budget, admission).await {
            Ok(result) => result,
            Err(_) => Err(Self::queue_timeout_refusal(
                ctx,
                accepted.elapsed(),
                timeout,
            )),
        }
    }
}

/// How long a request accepted at `accepted` has waited by `now`, when that
/// exceeds `timeout`.
pub(crate) fn waited_past_timeout(
    accepted: Instant,
    now: Instant,
    timeout: Duration,
) -> Option<Duration> {
    let waited = now.saturating_duration_since(accepted);
    (waited > timeout).then_some(waited)
}

fn disaggregated((prefill, decode, runtime_type): PdWorkerPair) -> WorkerSelection {
    WorkerSelection::Disaggregated {
        encode_assignments: None,
        prefill,
        decode,
        runtime_type,
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        self.select(ctx).await.map_err(|refusal| {
            let request_id = helpers::middleware_request_id(ctx.input.tenant_request_meta.as_ref());
            self.report_refusal(
                "WorkerSelectionStage::execute",
                request_id,
                &ctx.input.model_id,
                refusal,
            )
        })
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }
}

impl WorkerSelectionStage {
    /// The selection behind [`PipelineStage::execute`]; a refusal it returns
    /// is reported there by [`Self::report_refusal`].
    async fn select(&self, ctx: &mut RequestContext) -> Result<(), Response> {
        let accepted = self.acceptance_for_selection(ctx);
        if let Some(refused) = accepted.and_then(|accepted| self.refuse_if_expired(ctx, accepted)) {
            return Err(refused);
        }
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "WorkerSelectionStage::execute",
                "Preparation stage not completed"
            );
            error::internal_error(
                "preparation_stage_not_completed",
                "Preparation stage not completed",
            )
        })?;

        let intermediate = ctx.state.multimodal_intermediate.as_ref();
        // Media references only go to workers that advertise worker-side processing.
        let media_refs = ctx.state.multimodal_refs.is_some();
        // Pixel media (an image or a video) only goes to workers that accept
        // vision input: regular mode's single worker takes the whole request,
        // and the disaggregated legs their shares of it (see `VisionDemand`).
        let needs_vision =
            multimodal::request_needs_vision(intermediate, ctx.state.multimodal_refs.as_ref());
        let demand = VisionDemand::new(
            self.prefill_leg_needs_vision(needs_vision),
            self.decode_leg_needs_vision(&ctx.input.request_type, intermediate, media_refs),
        )
        .with_media_refs(media_refs);

        let text = prep.routing_text();

        // Get tokens for PrefixHash policy support
        let ids = prep.token_ids();
        let tokens = if ids.is_empty() { None } else { Some(ids) };

        let headers = ctx.input.headers.as_ref();
        let rid_key = self
            .policy_registry
            .derive_rid_key(ctx.input.request_type.rid())
            .map(str::to_string);
        ctx.state.sticky_key = self
            .policy_registry
            .sticky_key(headers, rid_key.as_deref())
            .map(str::to_string);

        // Selection inputs that survive the request drop: retry attempts
        // re-select from these. Text is copied only when a configured policy
        // would actually read it (tokens win otherwise).
        let keep_text =
            tokens.is_none() || self.policy_registry.any_policy_needs_request_text(headers);
        // The typed engine wires (gRPC servicers, direct ZMQ) do not carry the
        // request's cache-partition fields yet, so the engine's prefix cache
        // is unpartitioned on this path and the router must not split
        // affinity for it: a namespace here would cost cache hits for every
        // request that sets a salt today. The snapshot carries the field so
        // the forwarding change can derive it (as the HTTP proxy path does via
        // `GenerationRequest::cache_partition`) without touching selection.
        let cache_namespace: Option<CacheNamespace> = None;
        ctx.state.routing_snapshot = Some(RoutingSnapshot {
            routing_text: keep_text.then(|| text.map(str::to_string)).flatten(),
            token_ids: ids.to_vec(),
            rid_key: rid_key.clone(),
            cache_namespace,
            needs_vision,
            decode_needs_vision: demand.decode,
        });
        let rid_key = rid_key.as_deref();
        let inputs = PlacementInputs {
            text,
            tokens,
            headers,
            rid_key,
            cache_namespace,
            candidate_filter: media_refs.then_some(ACCEPTS_MEDIA_REFS),
            tried: &[],
        };
        let sticky_key = ctx.state.sticky_key.as_deref();

        let model_id = ctx.input.model_id.as_str();
        // Decisions needs selected-token scoring, currently supported only by
        // SGLang gRPC. Retries retain the selected wire via DispatchContext.
        let wire =
            matches!(ctx.input.request_type, RequestType::Decisions(_)).then_some(WireConstraint {
                runtime: RuntimeType::Sglang,
                connection: ConnectionMode::Grpc,
                requires_media_refs: false,
            });
        // The single worker takes the whole request, so it must meet every
        // demand of the request's media. The disaggregated legs take the
        // media-refs predicate from here and their vision demands from
        // `VisionDemand` in the pair selection: the prefill leg's for any
        // pixel media, the decode leg's for what a stripped leg cannot serve.
        let media = MediaDemands {
            media_refs,
            vision: needs_vision,
        };
        let accepts_media = |worker: &dyn Worker| media.accepts(worker);
        let media_filter: Option<CandidateFilter<'_>> = media.any().then_some(&accepts_media);
        let workers = match self.mode {
            WorkerSelectionMode::Regular => {
                match self.select_single_worker(
                    model_id,
                    text,
                    tokens,
                    headers,
                    rid_key,
                    cache_namespace,
                    wire,
                    media_filter,
                ) {
                    Some(w) => WorkerSelection::Single { worker: w },
                    None => {
                        return Err(self.selection_failure(
                            model_id,
                            &[WorkerType::Regular],
                            wire,
                            media,
                        ))
                    }
                }
            }
            WorkerSelectionMode::PrefillDecode => {
                let (pair, guard) = self
                    .admit_within_budget(
                        ctx,
                        accepted,
                        self.admit_pd_pair(model_id, inputs, sticky_key, None, media_refs, demand),
                    )
                    .await?;
                // An admission that lands right at the bound still gives its
                // claim back instead of taking a worker.
                if let Some(refused) =
                    accepted.and_then(|accepted| self.refuse_if_expired(ctx, accepted))
                {
                    drop(guard);
                    return Err(refused);
                }
                ctx.state.pd_prefill_guard = Some(guard);
                disaggregated(pair)
            }
            WorkerSelectionMode::EncodePrefillDecode => {
                // Encode workers never process references: refuse strict
                // worker mode here rather than silently bypass the encode fleet.
                if media_refs {
                    let err = multimodal::MmRefsError::EncodeNotSupported;
                    return Err(error::bad_request(err.code(), err.to_string()));
                }
                let encode_item_hashes = match encode_item_hashes(intermediate) {
                    Ok(hashes) => hashes,
                    Err(err) => {
                        error!(
                            function = "WorkerSelectionStage::execute",
                            error = %err,
                            "Failed to derive encode item routing hashes"
                        );
                        return Err(error::internal_error(
                            "encode_routing_hash_failed",
                            format!("Failed to derive encode routing hashes: {err}"),
                        ));
                    }
                };
                let ((encode_assignments, prefill, decode, runtime_type), guard) = self
                    .admit_within_budget(
                        ctx,
                        accepted,
                        self.admit_encode_prefill_decode_workers(
                            model_id,
                            inputs,
                            sticky_key,
                            &encode_item_hashes,
                        ),
                    )
                    .await?;
                if let Some(refused) =
                    accepted.and_then(|accepted| self.refuse_if_expired(ctx, accepted))
                {
                    drop(guard);
                    return Err(refused);
                }
                ctx.state.pd_prefill_guard = Some(guard);
                WorkerSelection::Disaggregated {
                    encode_assignments: if encode_assignments.is_empty() {
                        None
                    } else {
                        Some(encode_assignments)
                    },
                    prefill,
                    decode,
                    runtime_type,
                }
            }
        };

        // Reject an unsupported (backend, modality) combination now that the
        // runtime is known, before request building assembles the payload only
        // to fail deep in assembly. The prefill leg builds the request in
        // disaggregated mode, so its runtime is the one that must support the
        // request's modalities.
        if let Some(intermediate) = intermediate {
            if let Err(err) = multimodal::ensure_backend_supports_modalities(
                selection_runtime(&workers),
                intermediate,
            ) {
                return Err(error::bad_request(
                    "multimodal_not_supported",
                    format!("{err}"),
                ));
            }
        }

        if let Some(plan) = ctx.state.multimodal_refs.as_ref() {
            if let Err(err) = multimodal::ensure_selection_supports_media_refs(&workers, plan) {
                return Err(error::bad_request(err.code(), err.to_string()));
            }
        }

        ctx.state.workers = Some(workers);
        Ok(())
    }
}

impl WorkerSelectionStage {
    #[cfg(test)]
    pub(crate) fn signature(&self) -> String {
        format!("WorkerSelectionStage({:?})", self.mode)
    }

    /// Per-attempt re-selection from the routing snapshot, after the request
    /// dropped at build. Candidates are pinned to the retained plan's wire
    /// (runtime + transport): the plan cannot be rebuilt for another flavor.
    /// EPD re-selects only the prefill/decode pair — the first dispatch
    /// already launched the encode jobs, and the plan carries their
    /// bootstrap rooms. A disaggregated retry takes a fresh Prefill
    /// admission slot for its new prefill worker.
    pub(crate) async fn reselect(&self, ctx: &mut DispatchContext) -> Result<(), Response> {
        self.reselect_workers(ctx).await.map_err(|refusal| {
            self.report_refusal(
                "WorkerSelectionStage::reselect",
                ctx.request_id.as_deref(),
                &ctx.model_id,
                refusal,
            )
        })
    }

    /// The re-selection behind [`Self::reselect`].
    async fn reselect_workers(&self, ctx: &mut DispatchContext) -> Result<(), Response> {
        let text = ctx.routing.routing_text.as_deref();
        let tokens = if ctx.routing.token_ids.is_empty() {
            None
        } else {
            Some(ctx.routing.token_ids.as_slice())
        };
        let rid_key = ctx.routing.rid_key.as_deref();
        let cache_namespace = ctx.routing.cache_namespace;
        let headers = ctx.headers.as_ref();
        let model_id = ctx.model_id.as_str();
        // The retained wire carries the media-refs pin (the helpers derive it)
        // and the routing snapshot the pixel-media fact.
        let wire = Some(ctx.wire);
        let media_refs = ctx.wire.requires_media_refs;
        let media = MediaDemands {
            media_refs: false,
            vision: ctx.routing.needs_vision,
        };
        // A retry skips the workers that already answered this request
        // definitively (see `routers::common::attempt_ledger`).
        let ledger = &ctx.ledger;
        let admits = |worker: &dyn Worker| ledger.admits(worker);

        let workers = match self.mode {
            WorkerSelectionMode::Regular => {
                let admits_media = |worker: &dyn Worker| admits(worker) && media.accepts(worker);
                match self.select_single_worker_for_attempt(
                    model_id,
                    text,
                    tokens,
                    headers,
                    rid_key,
                    cache_namespace,
                    wire,
                    Some(&admits_media),
                    &ctx.tried,
                ) {
                    Some(w) => WorkerSelection::Single { worker: w },
                    None => {
                        return Err(self.selection_failure(
                            model_id,
                            &[WorkerType::Regular],
                            wire,
                            media,
                        ))
                    }
                }
            }
            WorkerSelectionMode::PrefillDecode | WorkerSelectionMode::EncodePrefillDecode => {
                let pinned = |worker: &dyn Worker| {
                    (!media_refs || accepts_media_refs(worker)) && admits(worker)
                };
                let inputs = PlacementInputs {
                    text,
                    tokens,
                    headers,
                    rid_key,
                    cache_namespace,
                    candidate_filter: Some(&pinned),
                    tried: &[],
                };
                let (pair, guard) = self
                    .admit_pd_pair(
                        model_id,
                        inputs,
                        ctx.sticky_key.as_deref(),
                        wire,
                        media_refs,
                        VisionDemand::new(
                            self.prefill_leg_needs_vision(ctx.routing.needs_vision),
                            ctx.routing.decode_needs_vision,
                        )
                        .with_media_refs(media_refs),
                    )
                    .await?;
                ctx.pd_prefill_guard = Some(guard);
                disaggregated(pair)
            }
        };

        ctx.workers = Some(workers);
        Ok(())
    }

    /// Whether this stage's decode leg needs a vision encoder for the
    /// request ([`decode_needs_vision`]), recorded in the routing snapshot
    /// and paired under: in prefill/decode mode only. In
    /// encode/prefill/decode mode the encode workers take the pixels and the
    /// decode leg is dispatched without them, so it never needs the encoder
    /// and a retry must not be held to a worker with one; regular mode has
    /// no decode leg.
    fn decode_leg_needs_vision(
        &self,
        request: &RequestType,
        intermediate: Option<&multimodal::MultimodalIntermediate>,
        media_refs: bool,
    ) -> bool {
        self.mode == WorkerSelectionMode::PrefillDecode
            && decode_needs_vision(request, intermediate, media_refs)
    }

    /// Whether this stage's prefill leg needs a vision encoder for a request
    /// that carries pixel media (`needs_vision`, see
    /// [`multimodal::request_needs_vision`]), paired under in prefill/decode
    /// mode only, where the prefill leg takes the request's media whole. In
    /// encode/prefill/decode mode the encode workers take the pixels; regular
    /// mode holds its single worker to the demand itself ([`MediaDemands`]).
    fn prefill_leg_needs_vision(&self, needs_vision: bool) -> bool {
        self.mode == WorkerSelectionMode::PrefillDecode && needs_vision
    }

    /// Select a PD pair and take its Prefill admission slot. With admission
    /// enabled the selection runs at the head of the admission queue, so a
    /// queued request commits to a worker only once it is admitted.
    async fn admit_pd_pair(
        &self,
        model_id: &str,
        inputs: PlacementInputs<'_>,
        sticky_key: Option<&str>,
        wire: Option<WireConstraint>,
        media_refs: bool,
        demand: VisionDemand,
    ) -> Result<(PdWorkerPair, PrefillLoadGuard), Response> {
        acquire_prefill(
            self.prefill_admission.as_deref(),
            sticky_key,
            |(prefill, _, _): &PdWorkerPair| prefill,
            |capacity| self.select_pd_pair(model_id, inputs, wire, capacity, demand),
        )
        .await
        .map_err(|error| {
            Self::acquire_failure(model_id, error, |failure| {
                self.pd_selection_failure(model_id, failure, media_refs)
            })
        })
    }

    /// EPD counterpart of [`Self::admit_pd_pair`].
    async fn admit_encode_prefill_decode_workers(
        &self,
        model_id: &str,
        inputs: PlacementInputs<'_>,
        sticky_key: Option<&str>,
        encode_item_hashes: &[Vec<u8>],
    ) -> Result<(EncodePrefillDecodeWorkerSelection, PrefillLoadGuard), Response> {
        acquire_prefill(
            self.prefill_admission.as_deref(),
            sticky_key,
            |(_, prefill, _, _): &EncodePrefillDecodeWorkerSelection| prefill,
            |capacity| {
                self.select_encode_prefill_decode_workers(
                    model_id,
                    inputs,
                    encode_item_hashes,
                    capacity,
                )
            },
        )
        .await
        .map_err(|error| {
            Self::acquire_failure(model_id, error, |_| {
                // Encode is a demanded leg only when the request carries
                // encode items; an idle-but-vetoed encode pool must not shed
                // a text-only request.
                let legs: &[WorkerType] = if encode_item_hashes.is_empty() {
                    &[WorkerType::Prefill, WorkerType::Decode]
                } else {
                    &[WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode]
                };
                self.selection_failure(model_id, legs, None, MediaDemands::NONE)
            })
        })
    }

    /// The response for a failed admission: the selection's own verdict when
    /// selection ran, a 429 when the queue refused the request.
    fn acquire_failure<E>(
        model_id: &str,
        error: PrefillAcquireError<E>,
        candidate: impl FnOnce(E) -> Response,
    ) -> Response {
        match error {
            PrefillAcquireError::Candidate(error) => candidate(error),
            PrefillAcquireError::Rejected(PrefillAdmissionRejection::QueueFull) => {
                prefill_queue_full()
            }
            PrefillAcquireError::Rejected(PrefillAdmissionRejection::QueueTimeout) => {
                prefill_queue_timeout()
            }
            PrefillAcquireError::Rejected(PrefillAdmissionRejection::Unavailable) => {
                Self::workers_unavailable(model_id)
            }
        }
    }
}

/// What a refusal line says about the pools it drew from: per leg, the
/// workers registered for the model, those healthy, and those routable
/// (healthy, circuit closed, not vetoed by load or liveness).
struct PoolState {
    legs: Vec<LegState>,
}

struct LegState {
    leg: &'static str,
    workers: usize,
    healthy: usize,
    available: usize,
}

impl PoolState {
    fn workers(&self) -> usize {
        self.legs.iter().map(|leg| leg.workers).sum()
    }

    fn available(&self) -> usize {
        self.legs.iter().map(|leg| leg.available).sum()
    }
}

impl fmt::Display for PoolState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, leg) in self.legs.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(
                f,
                "{} {}/{} available ({} healthy)",
                leg.leg, leg.available, leg.workers, leg.healthy
            )?;
        }
        Ok(())
    }
}

/// Candidate predicate for requests carrying media references.
fn accepts_media_refs(worker: &dyn Worker) -> bool {
    multimodal::worker_accepts_media_refs(worker)
}

/// Candidate predicate for a request, or a disaggregated leg of one, that
/// needs a vision encoder: a worker that advertises none
/// (`supports_vision=false`) admits such a request and never schedules it. A
/// worker without the label reads as capable, as today.
fn vision_capable(worker: &dyn Worker) -> bool {
    !multimodal::worker_language_model_only(worker)
}

/// Whether a request's disaggregated decode leg cannot be served by a decode
/// worker that accepts no vision input: the shapes dispatch refuses with a
/// non-retryable 400 once such a worker is selected (media references, which
/// each leg expands itself; a multimodal request with `n > 1`, whose decode
/// leg recomputes the prompt locally; grid tensors, which the leg derives
/// its positions from). A text request of any `n` and a single-sample image
/// request without grids take such a worker's stripped leg, so they are not
/// steered away from it.
fn decode_needs_vision(
    request: &RequestType,
    intermediate: Option<&multimodal::MultimodalIntermediate>,
    media_refs: bool,
) -> bool {
    media_refs
        || intermediate
            .is_some_and(|intermediate| request.sampling_n() > 1 || intermediate.has_mrope_grids())
}

/// Which legs of a disaggregated request need a worker with a vision
/// encoder. The prefill leg takes the request's media whole (the processed
/// pixel tensors, or the references it expands itself), so any image or
/// video is a demand on it: a prefill worker without the encoder admits
/// such a request and never schedules it, and no dispatch check catches
/// that. The decode leg's demand is [`decode_needs_vision`]'s, and holds
/// on the prefill leg too (neither leg runs those shapes without the
/// encoder).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VisionDemand {
    prefill: bool,
    decode: bool,
    /// The request forwards media references, which only a worker that
    /// processes them can take: the prefill leg's refusal check then wants
    /// a worker with the encoder among those, or the request could never
    /// pair and a retryable 503 would send the client back for ever.
    refs: bool,
}

impl VisionDemand {
    /// The demand of a request that carries pixel media (`pixel_media`, see
    /// [`multimodal::request_needs_vision`]) and whose decode leg needs the encoder
    /// (`decode`, see [`decode_needs_vision`]).
    fn new(pixel_media: bool, decode: bool) -> Self {
        Self {
            prefill: pixel_media || decode,
            decode,
            refs: false,
        }
    }

    /// The same demand for a request that forwards media references (`refs`,
    /// the caller's `ACCEPTS_MEDIA_REFS` filter): the prefill leg's refusal
    /// is judged among the workers that take them.
    fn with_media_refs(self, refs: bool) -> Self {
        Self { refs, ..self }
    }

    /// The demand on the leg `worker` serves.
    fn on(self, worker: &dyn Worker) -> bool {
        match worker.worker_type() {
            WorkerType::Prefill => self.prefill,
            _ => self.decode,
        }
    }
}

/// The predicate of a request whose prefill leg needs a vision encoder, on
/// either leg: a prefill worker with the encoder, any decode worker.
fn prefill_vision_capable(worker: &dyn Worker) -> bool {
    !matches!(worker.worker_type(), WorkerType::Prefill) || vision_capable(worker)
}

/// Whether `pairs` holds a prefill that `accepts` admits with a partner it
/// admits too, both on `wire` when a retry pins one. Read from the index,
/// not the pools, so a worker that pairs with nothing cannot shut the pool.
fn pair_admitted_by(
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    accepts: &dyn Fn(&dyn Worker) -> bool,
) -> bool {
    let admitted = |worker: &Arc<dyn Worker>| {
        WireConstraint::admits(wire, worker.as_ref()) && accepts(worker.as_ref())
    };
    pairs
        .prefill
        .iter()
        .zip(&pairs.partners)
        .filter(|(prefill, _)| admitted(prefill))
        .any(|(prefill, partners)| {
            let runtime = prefill.metadata().spec.runtime_type;
            partners
                .iter()
                .any(|decode| admitted(decode) && decode.metadata().spec.runtime_type == runtime)
        })
}

/// What a request's media demands of a regular-mode candidate, beyond the
/// pool and the wire. Across a retry the wire pin carries the media-refs
/// demand and the routing snapshot the vision one; [`Self::with_wire`] folds
/// the pin in for the verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MediaDemands {
    /// Media references are forwarded: only a worker that processes them
    /// itself ([`accepts_media_refs`]).
    media_refs: bool,
    /// Pixel media is carried: only a worker with a vision encoder
    /// ([`vision_capable`]).
    vision: bool,
}

impl MediaDemands {
    /// No demand: every worker of the pool qualifies.
    const NONE: Self = Self {
        media_refs: false,
        vision: false,
    };

    /// These demands plus the retained plan's media-refs pin on a retry.
    fn with_wire(self, wire: Option<WireConstraint>) -> Self {
        Self {
            media_refs: self.media_refs || wire.is_some_and(|w| w.requires_media_refs),
            vision: self.vision,
        }
    }

    fn any(self) -> bool {
        self.media_refs || self.vision
    }

    /// Whether `worker` meets every demand.
    fn accepts(self, worker: &dyn Worker) -> bool {
        (!self.media_refs || accepts_media_refs(worker)) && (!self.vision || vision_capable(worker))
    }
}

/// Runtime of the leg that builds the generate request: the sole worker in
/// regular mode, the prefill worker in disaggregated (PD/EPD) mode.
fn selection_runtime(workers: &WorkerSelection) -> RuntimeType {
    match workers {
        WorkerSelection::Single { worker } => worker.metadata().spec.runtime_type,
        WorkerSelection::Disaggregated { runtime_type, .. } => *runtime_type,
    }
}

impl WorkerSelectionStage {
    /// Response for a selection that produced nothing: a 503 shed when a leg's
    /// whole candidate pool is vetoed, the existing 404 otherwise.
    ///
    /// `legs` must be exactly the legs this selection demanded — the verdict is
    /// per leg because a whole-model predicate is false exactly when one
    /// saturated leg made the set unselectable, and an undemanded leg (EPD
    /// without encode items) must not be able to shed a request that never
    /// needed it.
    /// `wire` narrows the verdict to the retained plan's runtime/transport on
    /// the retry path: a drained pinned pool must not answer 404 just because
    /// other runtimes still serve the model.
    /// `media` are the request's media demands: a leg with candidates but
    /// none meeting them is a capability verdict ([`Self::media_failure`]).
    fn selection_failure(
        &self,
        model_id: &str,
        legs: &[WorkerType],
        wire: Option<WireConstraint>,
        media: MediaDemands,
    ) -> Response {
        let media = media.with_wire(wire);
        if media.any() {
            return self.media_failure(model_id, legs, wire, media);
        }
        let mut unavailable = false;
        for leg in legs {
            let verdict = match leg {
                // The regular leg is judged from exactly the pool the shared
                // placement drew from.
                WorkerType::Regular => placement::single_failure(
                    &self.worker_registry,
                    model_id,
                    RoutingPool::GrpcPipelineRegular,
                    wire,
                ),
                WorkerType::Prefill => {
                    self.disaggregated_leg_verdict(model_id, RoutingPool::GrpcPrefill, wire)
                }
                WorkerType::Decode => {
                    self.disaggregated_leg_verdict(model_id, RoutingPool::GrpcDecode, wire)
                }
                WorkerType::Encode => {
                    self.disaggregated_leg_verdict(model_id, RoutingPool::GrpcEncode, wire)
                }
            };
            match verdict {
                PlacementFailure::AllOverloaded(shed) => return shed,
                PlacementFailure::Unavailable
                | PlacementFailure::PolicyDeclined(_)
                | PlacementFailure::NoCompatiblePair { .. }
                | PlacementFailure::PrefillAtCapacity => {
                    unavailable = true;
                }
                PlacementFailure::NoCandidates => {}
            }
        }
        if unavailable {
            return Self::workers_unavailable(model_id);
        }
        error!(
            function = "WorkerSelectionStage::execute",
            mode = ?self.mode,
            model_id = %model_id,
            "No worker serves model"
        );
        error::model_not_found(model_id)
    }

    /// Selection failure for a request with media demands, judged from the
    /// capability-filtered pool: an absent model stays 404, an overloaded
    /// capable pool keeps its shed, and only a leg with candidates but no
    /// capable one is a capability verdict. The vision encoder is judged
    /// first: a pool with no worker that can take the request's pixels is
    /// the request's own 400, whichever workers process references.
    fn media_failure(
        &self,
        model_id: &str,
        legs: &[WorkerType],
        wire: Option<WireConstraint>,
        media: MediaDemands,
    ) -> Response {
        let mut unavailable = false;
        for leg in legs {
            let pool = match leg {
                WorkerType::Regular => RoutingPool::GrpcPipelineRegular,
                WorkerType::Prefill => RoutingPool::GrpcPrefill,
                WorkerType::Decode => RoutingPool::GrpcDecode,
                WorkerType::Encode => RoutingPool::GrpcEncode,
            };
            let candidates = placement::candidates(&self.worker_registry, model_id, pool, wire);
            let candidates = candidates.as_slice();
            if candidates.is_empty() {
                continue;
            }
            let vision_capable: Vec<Arc<dyn Worker>> = candidates
                .iter()
                .filter(|w| !media.vision || vision_capable(w.as_ref()))
                .cloned()
                .collect();
            if vision_capable.is_empty() {
                return self.vision_refusal(model_id, "workers");
            }
            let capable: Vec<Arc<dyn Worker>> = vision_capable
                .into_iter()
                .filter(|w| !media.media_refs || accepts_media_refs(w.as_ref()))
                .collect();
            if capable.is_empty() {
                return self.media_refs_shed(model_id);
            }
            match placement::failure_from(
                &capable,
                model_id,
                self.worker_registry.overload_shed_enabled(),
            ) {
                PlacementFailure::AllOverloaded(shed) => return shed,
                PlacementFailure::Unavailable
                | PlacementFailure::PolicyDeclined(_)
                | PlacementFailure::NoCompatiblePair { .. }
                | PlacementFailure::PrefillAtCapacity => unavailable = true,
                PlacementFailure::NoCandidates => {}
            }
        }
        if unavailable {
            return Self::workers_unavailable(model_id);
        }
        error!(
            function = "WorkerSelectionStage::execute",
            mode = ?self.mode,
            model_id = %model_id,
            "No worker serves model"
        );
        error::model_not_found(model_id)
    }

    /// Workers serve the model but none advertises worker-side multimodal
    /// processing. Terminal: capability cannot change within a retry window.
    fn media_refs_shed(&self, model_id: &str) -> Response {
        error!(
            function = "WorkerSelectionStage::execute",
            mode = ?self.mode,
            model_id = %model_id,
            "No worker advertising worker-side multimodal processing"
        );
        let mut response = error::service_unavailable(
            "no_media_ref_capable_worker",
            format!(
                "model {model_id} has vLLM gRPC workers but none advertises mm_processor; set \
                 SMG_VLLM_MM_PROCESSOR on the workers or SMG_MM_PROCESSING=router on the router"
            ),
        );
        mark_non_retryable(&mut response);
        response
    }

    /// Workers serve the model but none can take the request right now
    /// (unhealthy, circuit breaker open, or the policy declined). A 503 with
    /// the same code the HTTP router uses: the model exists, the client should
    /// retry, and nothing about its request is wrong. Answering 404 here told
    /// clients the model was gone while its workers restarted. Terminal for
    /// the pipeline's own retry loop (see `placement::no_available_workers`).
    /// Logged by [`Self::report_refusal`] at the stage's entry points, once
    /// per refusal and rate-limited, not here.
    fn workers_unavailable(model_id: &str) -> Response {
        placement::no_available_workers(format!(
            "All workers for model '{model_id}' are unavailable (unhealthy or circuit breaker open)"
        ))
    }

    /// The log line of a `no_available_workers` refusal, and its marker.
    ///
    /// One WARN per refusal with the refusing function, the request id, the
    /// model and the state of the pools the stage draws from, capped per
    /// model at [`NO_AVAILABLE_WORKERS_LINES_PER_SECOND`]; a line past the
    /// cap is counted and the next admitted line for that model carries the
    /// count as `suppressed`. The response is marked [`ReportedRefusal`], so
    /// the pipeline's lines and the access log's response line drop to DEBUG
    /// for it: a refusal is the gateway's normal answer to "no worker right
    /// now", not a fault of its own, and three ERROR lines per refusal made
    /// a client's retry loop against a drained pool a log storm. The error
    /// counter (`smg_router_request_errors_total`, recorded by the pipeline)
    /// keeps the volume. Every other refusal passes through unchanged.
    fn report_refusal(
        &self,
        function: &'static str,
        request_id: Option<&str>,
        model_id: &str,
        refusal: Response,
    ) -> Response {
        self.report_refusal_with(
            &NO_AVAILABLE_WORKERS_LOG,
            function,
            request_id,
            model_id,
            refusal,
        )
    }

    /// [`Self::report_refusal`] under an explicit throttle. The id is a
    /// field only when the caller has one: the request's span carries it
    /// too, and an empty field would read as an id.
    fn report_refusal_with(
        &self,
        throttle: &KeyedLogThrottle,
        function: &'static str,
        request_id: Option<&str>,
        model_id: &str,
        mut refusal: Response,
    ) -> Response {
        if error::extract_error_code_from_response(&refusal) != placement::NO_AVAILABLE_WORKERS_CODE
        {
            return refusal;
        }
        refusal.extensions_mut().insert(ReportedRefusal);
        if let Some(admitted) = throttle.admit(model_id) {
            let pool = self.pool_state(model_id);
            warn!(
                function,
                request_id = request_id,
                mode = ?self.mode,
                model_id = %model_id,
                workers = pool.workers(),
                available = pool.available(),
                pool = %pool,
                suppressed = admitted.suppressed,
                "No available workers for model"
            );
        }
        refusal
    }

    /// The pools the stage's mode draws from for `model_id`, as the refusal
    /// line reports them.
    fn pool_state(&self, model_id: &str) -> PoolState {
        let legs: &[(&'static str, RoutingPool)] = match self.mode {
            WorkerSelectionMode::Regular => &[("regular", RoutingPool::GrpcPipelineRegular)],
            WorkerSelectionMode::PrefillDecode => &[
                ("prefill", RoutingPool::GrpcPrefill),
                ("decode", RoutingPool::GrpcDecode),
            ],
            WorkerSelectionMode::EncodePrefillDecode => &[
                ("encode", RoutingPool::GrpcEncode),
                ("prefill", RoutingPool::GrpcPrefill),
                ("decode", RoutingPool::GrpcDecode),
            ],
        };
        PoolState {
            legs: legs
                .iter()
                .map(|(leg, pool)| {
                    let workers = self.worker_registry.get_routing_pool(model_id, *pool);
                    LegState {
                        leg,
                        workers: workers.len(),
                        healthy: workers.iter().filter(|w| w.is_healthy()).count(),
                        available: workers.iter().filter(|w| w.is_available()).count(),
                    }
                })
                .collect(),
        }
    }

    /// The response for a failed prefill/decode selection: the placement's
    /// verdict as [`Self::pd_pair_failure`] answers it, or the refusal of a
    /// request carrying pixel media that no vision-capable prefill worker is
    /// paired for.
    fn pd_selection_failure(
        &self,
        model_id: &str,
        failure: PdSelectionFailure,
        media_refs: bool,
    ) -> Response {
        match failure {
            PdSelectionFailure::Placement(failure) => {
                self.pd_pair_failure(model_id, *failure, media_refs)
            }
            PdSelectionFailure::NoVisionCapablePrefill => {
                self.vision_refusal(model_id, "prefill workers")
            }
        }
    }

    /// `workers` (the regular pool's, or the prefill leg's) serve the model
    /// but none has a vision encoder, so none can take the request's images
    /// or video: the 400 a model without vision gives image input, instead
    /// of a dispatch the engine admits and never schedules. A 400 is
    /// terminal for the retry layer by status.
    fn vision_refusal(&self, model_id: &str, workers: &str) -> Response {
        warn!(
            function = "WorkerSelectionStage::execute",
            mode = ?self.mode,
            model_id = %model_id,
            workers,
            "No worker that accepts vision input for a request carrying pixel media"
        );
        error::bad_request(
            "multimodal_not_supported",
            format!(
                "image or video input is not supported by model '{model_id}': none of its \
                 {workers} accept vision input (the engine runs as a language model only, or \
                 its servicer does not carry media to it)"
            ),
        )
    }

    /// The response for a failed pair placement. The verdict was judged from
    /// the leg's own candidates inside the placement, so a shed is answered
    /// as it was built and counted once; a leg nobody serves is a 404, and a
    /// leg whose workers are all unavailable is the 503 the HTTP router gives.
    fn pair_failure(&self, model_id: &str, failure: PairFailure) -> Response {
        match failure.verdict {
            PlacementFailure::AllOverloaded(shed) => shed,
            // Admission turns a capacity verdict into a wait before it gets here.
            PlacementFailure::Unavailable
            | PlacementFailure::PolicyDeclined(_)
            | PlacementFailure::PrefillAtCapacity => Self::workers_unavailable(model_id),
            PlacementFailure::NoCompatiblePair {
                prefill,
                decode,
                mismatches,
            } => {
                error!(
                    function = "WorkerSelectionStage::execute",
                    mode = ?self.mode,
                    model_id = %model_id,
                    ?mismatches,
                    ?prefill,
                    ?decode,
                    "No prefill/decode pair shares a KV transfer protocol"
                );
                error::service_unavailable(
                    "no_compatible_pd_pair",
                    format!(
                        "No prefill/decode pair for model '{model_id}' shares a KV transfer \
                         protocol (mismatch on {mismatches:?}; prefill: {prefill:?}, decode: \
                         {decode:?})"
                    ),
                )
            }
            PlacementFailure::NoCandidates => {
                error!(
                    function = "WorkerSelectionStage::execute",
                    mode = ?self.mode,
                    model_id = %model_id,
                    leg = ?failure.leg,
                    "No worker serves model"
                );
                error::model_not_found(model_id)
            }
        }
    }

    /// The verdict for one disaggregated leg, judged from the pool it
    /// selected over *before* the `is_available()` filter. The legs are
    /// gRPC-only (no KV rendezvous on ZMQ), so a retry pins the runtime alone;
    /// the regular leg is judged by [`placement::single_failure`] instead.
    /// Failure path only.
    fn disaggregated_leg_verdict(
        &self,
        model_id: &str,
        pool: RoutingPool,
        wire: Option<WireConstraint>,
    ) -> PlacementFailure {
        let candidates: Vec<Arc<dyn Worker>> = self
            .worker_registry
            .get_routing_pool(model_id, pool)
            .iter()
            .filter(|w| wire.is_none_or(|c| w.metadata().spec.runtime_type == c.runtime))
            .cloned()
            .collect();
        placement::failure_from(
            &candidates,
            model_id,
            self.worker_registry.overload_shed_enabled(),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "selection threads every routing input the policy consumes"
    )]
    fn select_single_worker(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        rid_key: Option<&str>,
        cache_namespace: Option<CacheNamespace>,
        wire: Option<WireConstraint>,
        candidate_filter: Option<CandidateFilter<'_>>,
    ) -> Option<Arc<dyn Worker>> {
        self.select_single_worker_for_attempt(
            model_id,
            text,
            tokens,
            headers,
            rid_key,
            cache_namespace,
            wire,
            candidate_filter,
            &[],
        )
    }

    /// [`Self::select_single_worker`] for a retry attempt: `tried` are the
    /// engines the request's failed attempts dispatched to, skipped while
    /// another available candidate is left (see `PlacementInputs::tried`).
    #[expect(
        clippy::too_many_arguments,
        reason = "selection threads every routing input the policy consumes"
    )]
    fn select_single_worker_for_attempt(
        &self,
        model_id: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
        rid_key: Option<&str>,
        cache_namespace: Option<CacheNamespace>,
        wire: Option<WireConstraint>,
        candidate_filter: Option<CandidateFilter<'_>>,
        tried: &[String],
    ) -> Option<Arc<dyn Worker>> {
        // A retry pins the retained wire, and with it the media pin, on top of
        // whatever the caller excludes.
        let pinned = |worker: &dyn Worker| {
            accepts_media_refs(worker) && candidate_filter.is_none_or(|accepts| accepts(worker))
        };
        let candidate_filter: Option<CandidateFilter<'_>> =
            if wire.is_some_and(|w| w.requires_media_refs) {
                Some(&pinned)
            } else {
                candidate_filter
            };
        // The gRPC router serves both gRPC and direct-ZMQ workers, so the pool
        // accepts either transport (not HTTP). A retry pins the retained wire.
        placement::select_single(
            &self.worker_registry,
            &self.policy_registry,
            model_id,
            RoutingPool::GrpcPipelineRegular,
            wire,
            PlacementInputs {
                text,
                tokens,
                headers,
                rid_key,
                cache_namespace,
                candidate_filter,
                tried,
            },
        )
    }

    /// Workers from one leg pool of `snapshot` that also pass the live
    /// `is_available()` check (health, circuit breaker, overload veto).
    fn available_workers(
        snapshot: &ModelWorkerSnapshot,
        pool: RoutingPool,
    ) -> Vec<Arc<dyn Worker>> {
        snapshot
            .pool(pool)
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect()
    }

    fn pd_pair_failure(&self, model_id: &str, failure: PairFailure, media_refs: bool) -> Response {
        if media_refs {
            let snapshot = self.worker_registry.get_routing_snapshot(model_id);
            let pairs = snapshot.pd_pairs(PdWire::Grpc, self.policy_registry.pd_pairing_mode());
            let pool = match failure.leg {
                WorkerLeg::Prefill => &pairs.prefill_pool,
                _ => &pairs.decode_pool,
            };
            if !pool.is_empty() && !pool.iter().any(|w| accepts_media_refs(w.as_ref())) {
                return self.media_refs_shed(model_id);
            }
        }
        self.pair_failure(model_id, failure)
    }

    fn select_pd_pair(
        &self,
        model_id: &str,
        inputs: PlacementInputs<'_>,
        wire: Option<WireConstraint>,
        prefill_capacity: Option<&PrefillSelectionContext<'_>>,
        demand: VisionDemand,
    ) -> Result<PdWorkerPair, PdSelectionFailure> {
        // Both legs derive from ONE membership snapshot: separate pool
        // lookups could straddle a concurrent replacement and pair workers
        // that never coexisted. The pools are strictly gRPC (a ZMQ leg would
        // silently drop the PD bootstrap info, see `RoutingPool::GrpcPrefill`;
        // the wildcard model maps to the global snapshot). The legs must
        // share a runtime, the rendezvous being runtime-specific, and a retry
        // pins both to the retained plan's runtime.
        let snapshot = self.worker_registry.get_routing_snapshot(model_id);
        let pairs = snapshot.pd_pairs(PdWire::Grpc, self.policy_registry.pd_pairing_mode());
        // A leg that needs a vision encoder (see `VisionDemand`) is kept off
        // the workers without one. The prefill leg must find one: a request
        // carrying pixel media is refused when the index pairs no prefill
        // worker with the encoder (a leg nobody serves and a fleet without a
        // shared transport keep the placement's own verdicts), since no
        // dispatch check would catch the leg landing on a worker without it.
        // The decode leg is kept off such workers while the index holds a
        // pair with the encoder on both legs that the caller's own predicate
        // admits too; with none the decode pool is used as it is, and
        // dispatch refuses the leg naming the gap, as it does for a pool
        // with no such worker at all. With media references the worker must
        // take them too (`VisionDemand::refs`): a pool whose referencing
        // prefill worker lacks the encoder and whose encoder-bearing one
        // takes no references can never pair the request, and the verdict
        // of the unfiltered pool would be a retryable 503 over healthy
        // workers. A pool with no pair that takes the references at all is
        // not this refusal: the placement's media-references verdict names
        // that gap, and the fix that applies to it.
        let takes_refs = |worker: &dyn Worker| !demand.refs || accepts_media_refs(worker);
        if demand.prefill
            && !pairs.prefill_pool.is_empty()
            && !pairs.decode_pool.is_empty()
            && pairs.refusal.is_none()
            && pair_admitted_by(&pairs, wire, &takes_refs)
            && !pair_admitted_by(&pairs, wire, &|worker: &dyn Worker| {
                prefill_vision_capable(worker) && takes_refs(worker)
            })
        {
            return Err(PdSelectionFailure::NoVisionCapablePrefill);
        }
        let caller = inputs.candidate_filter;
        let admitted = move |worker: &dyn Worker| caller.is_none_or(|accepts| accepts(worker));
        let with_vision = move |worker: &dyn Worker| {
            (!demand.on(worker) || vision_capable(worker)) && admitted(worker)
        };
        let prefill_with_vision =
            move |worker: &dyn Worker| prefill_vision_capable(worker) && admitted(worker);
        let candidate_filter: Option<CandidateFilter<'_>> =
            if demand.decode && pair_admitted_by(&pairs, wire, &with_vision) {
                Some(&with_vision)
            } else if demand.prefill {
                Some(&prefill_with_vision)
            } else {
                inputs.candidate_filter
            };
        let inputs = PlacementInputs {
            candidate_filter,
            ..inputs
        };
        let pair = placement::select_pair(
            &self.worker_registry,
            &self.policy_registry,
            model_id,
            &pairs,
            wire,
            true,
            prefill_capacity,
            inputs,
        )?;
        Ok((pair.prefill, pair.decode, pair.runtime))
    }

    /// Select per-item encode workers + a prefill/decode pair for EPD routing.
    ///
    /// Mirrors `select_pd_pair` but also assigns each multimodal item to an
    /// encode worker. prefill+decode are selected as a normal PD pair. All pools
    /// are filtered to a runtime shared by the selected encode/prefill/decode
    /// legs.
    ///
    /// The error only tells a full Prefill leg apart from every other
    /// failure; those are judged per leg by [`Self::selection_failure`].
    fn select_encode_prefill_decode_workers(
        &self,
        model_id: &str,
        inputs: PlacementInputs<'_>,
        encode_item_hashes: &[Vec<u8>],
        prefill_capacity: Option<&PrefillSelectionContext<'_>>,
    ) -> Result<EncodePrefillDecodeWorkerSelection, EpdSelectionFailure> {
        let PlacementInputs {
            text,
            tokens,
            headers,
            rid_key,
            cache_namespace,
            ..
        } = inputs;
        // All three legs derive from ONE membership snapshot (see
        // select_pd_pair). The pools are strictly gRPC — encode dispatch is
        // a gRPC encoder RPC the direct-ZMQ worker has no path for, and the
        // ZMQ wire carries no KV-transfer rendezvous for the prefill/decode
        // legs; the wildcard model maps to the global snapshot. Availability
        // stays a live per-request check.
        let snapshot = self.worker_registry.get_routing_snapshot(model_id);
        let all_encode = Self::available_workers(&snapshot, RoutingPool::GrpcEncode);
        let all_prefill = Self::available_workers(&snapshot, RoutingPool::GrpcPrefill);
        let all_decode = Self::available_workers(&snapshot, RoutingPool::GrpcDecode);

        let needs_encode = !encode_item_hashes.is_empty();
        if needs_encode && all_encode.is_empty() {
            warn!("No available encode workers");
            return Err(EpdSelectionFailure::Unavailable);
        }
        if all_prefill.is_empty() {
            warn!("No available prefill workers");
            return Err(EpdSelectionFailure::Unavailable);
        }
        if all_decode.is_empty() {
            warn!("No available decode workers");
            return Err(EpdSelectionFailure::Unavailable);
        }

        // Disaggregated legs must share a runtime. Pick a runtime that has at
        // least one available worker in every required EPD pool instead of
        // blindly using the first prefill runtime.
        let Some(target_runtime) = all_prefill
            .iter()
            .map(|w| w.metadata().spec.runtime_type)
            .find(|runtime| {
                // The current EPD multimodal encoder adapter is TokenSpeed-
                // specific. Do not select a shared SGLang/vLLM runtime only to
                // reject it later during request building.
                (!needs_encode || *runtime == RuntimeType::TokenSpeed)
                    && all_decode
                        .iter()
                        .any(|w| w.metadata().spec.runtime_type == *runtime)
                    && (!needs_encode
                        || all_encode
                            .iter()
                            .any(|w| w.metadata().spec.runtime_type == *runtime))
            })
        else {
            warn!("No available encode/prefill/decode worker set with a shared runtime");
            return Err(EpdSelectionFailure::Unavailable);
        };

        let mixed = all_prefill
            .iter()
            .chain(all_decode.iter())
            .any(|w| w.metadata().spec.runtime_type != target_runtime)
            || (needs_encode
                && all_encode
                    .iter()
                    .any(|w| w.metadata().spec.runtime_type != target_runtime));
        if mixed {
            warn!(
                "Mixed runtime types in encode/prefill/decode workers. Using {:?}.",
                target_runtime
            );
        }

        // Filter all three pools to the target runtime
        let available_encode: Vec<_> = all_encode
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();
        let mut available_prefill: Vec<_> = all_prefill
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();
        let available_decode: Vec<_> = all_decode
            .into_iter()
            .filter(|w| w.metadata().spec.runtime_type == target_runtime)
            .collect();

        if (needs_encode && available_encode.is_empty())
            || available_prefill.is_empty()
            || available_decode.is_empty()
        {
            warn!(
                "No available encode/prefill/decode worker set for runtime {:?}",
                target_runtime
            );
            return Err(EpdSelectionFailure::Unavailable);
        }

        // Select encode, prefill, and decode via their per-role policies. Encode
        // defaults to consistent hashing over each item's content hash; prefill
        // and decode fall back to the main policy when unset.
        let encode_policy = self.policy_registry.get_encode_policy();
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self.worker_registry.get_hash_ring(model_id);

        // Under Prefill admission, full prefill workers leave the candidate
        // set before the policy runs (see `placement::select_pair`).
        if let Some(capacity) = prefill_capacity
            .filter(|_| placement::admission_prefilters(prefill_policy.as_ref(), headers))
        {
            available_prefill.retain(|w| capacity.has_capacity(w));
            if available_prefill.is_empty() {
                return Err(EpdSelectionFailure::PrefillAtCapacity);
            }
        }

        let mut info = SelectWorkerInfo {
            request_text: text,
            tokens,
            headers,
            routing_key: self.policy_registry.resolve_routing_key(headers),
            rid_key,
            cache_namespace,
            hash_ring: hash_ring.clone(),
            leg: WorkerLeg::Prefill,
        };
        let prefill_idx = self
            .policy_registry
            .select_worker_for_model(&prefill_policy, model_id, &available_prefill, &info)
            .ok_or(EpdSelectionFailure::Unavailable)?;
        if prefill_capacity
            .is_some_and(|capacity| !capacity.has_capacity(&available_prefill[prefill_idx]))
        {
            return Err(EpdSelectionFailure::PrefillAtCapacity);
        }
        info.leg = WorkerLeg::Decode;
        let decode_idx = self
            .policy_registry
            .select_worker_for_model(&decode_policy, model_id, &available_decode, &info)
            .ok_or(EpdSelectionFailure::Unavailable)?;

        let encode_assignments = assign_encode_workers(
            &available_encode,
            encode_item_hashes,
            model_id,
            encode_policy.as_ref(),
            hash_ring.clone(),
        )
        .ok_or(EpdSelectionFailure::Unavailable)?;

        // Record worker selection metrics for prefill and decode, each tagged
        // with the policy that picked it. Encode item assignment metrics are
        // recorded in assign_encode_workers.
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            available_prefill[prefill_idx]
                .connection_mode()
                .as_metric_label(),
            model_id,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            available_decode[decode_idx]
                .connection_mode()
                .as_metric_label(),
            model_id,
            decode_policy.name(),
        );

        Ok((
            encode_assignments,
            available_prefill[prefill_idx].clone(),
            available_decode[decode_idx].clone(),
            target_runtime,
        ))
    }
}

fn encode_item_hashes(
    intermediate: Option<&multimodal::MultimodalIntermediate>,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let Some(intermediate) = intermediate else {
        return Ok(Vec::new());
    };
    multimodal::encode_routing_hashes(intermediate)
}

fn assign_encode_workers(
    encode_workers: &[Arc<dyn Worker>],
    item_hashes: &[Vec<u8>],
    model_id: &str,
    policy: &dyn LoadBalancingPolicy,
    hash_ring: Option<Arc<HashRing>>,
) -> Option<Vec<EncodeWorkerAssignment>> {
    if item_hashes.is_empty() {
        return Some(Vec::new());
    }

    item_hashes
        .iter()
        .enumerate()
        .map(|(item_index, content_hash)| {
            let routing_headers = encode_routing_headers(content_hash);
            let info = SelectWorkerInfo {
                request_text: None,
                tokens: None,
                headers: Some(&routing_headers),
                routing_key: None,
                // Encode items key by media-content hash; a conversation key
                // here would defeat per-item encode reuse.
                rid_key: None,
                cache_namespace: None,
                hash_ring: hash_ring.clone(),
                leg: WorkerLeg::Single,
            };
            let worker_idx = policy.select_worker(encode_workers, &info)?;
            let worker = encode_workers[worker_idx].clone();
            Metrics::record_worker_selection(
                metrics_labels::WORKER_ENCODE,
                metrics_labels::CONNECTION_GRPC,
                model_id,
                policy.name(),
            );
            Some(EncodeWorkerAssignment { item_index, worker })
        })
        .collect()
}

fn encode_routing_headers(content_hash: &[u8]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let key = hex_encode(content_hash);
    if let Ok(value) = HeaderValue::from_str(&key) {
        headers.insert("x-smg-routing-key", value);
    }
    headers
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use axum::http::{header::RETRY_AFTER, StatusCode};
    use llm_tokenizer::TokenizerRegistry;
    use openai_protocol::{generate::GenerateRequest, worker::HealthCheckConfig};

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        mesh::adapters::tree_sync::RepairEntry,
        middleware::TenantRequestMeta,
        policies::{CacheAwareConfig, CacheAwarePolicy, PolicyFactory, TreeHandle, TreeKind},
        routers::{
            common::{attempt_ledger::AttemptLedger, retry::is_retryable_response},
            grpc::{
                context::{PreparationOutput, SharedComponents},
                utils,
            },
            PD_PREFILL_QUEUE_FULL,
        },
        tenant::TenantKey,
        worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, PrefillReservation},
    };

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn retry_after(response: &Response) -> Option<&str> {
        response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
    }

    fn register_pd_workers(
        registry: &WorkerRegistry,
        model_id: &str,
        n: usize,
    ) -> (Vec<String>, Vec<String>) {
        let mut prefill_urls = Vec::with_capacity(n);
        let mut decode_urls = Vec::with_capacity(n);

        for i in 0..n {
            let url = format!("grpc://127.0.0.1:{}", 8000 + i);
            prefill_urls.push(url.clone());
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .model(ModelCard::new(model_id))
                        .worker_type(WorkerType::Prefill)
                        .connection_mode(ConnectionMode::Grpc)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }

        for i in 0..n {
            let url = format!("grpc://127.0.0.1:{}", 8100 + i);
            decode_urls.push(url.clone());
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .model(ModelCard::new(model_id))
                        .worker_type(WorkerType::Decode)
                        .connection_mode(ConnectionMode::Grpc)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }

        (prefill_urls, decode_urls)
    }

    fn worker(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("test-model"))
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        )
    }

    fn pd_stage(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        prefill_admission: Arc<PrefillAdmission>,
    ) -> WorkerSelectionStage {
        WorkerSelectionStage::new(
            worker_registry,
            policy_registry,
            WorkerSelectionMode::PrefillDecode,
            Some(prefill_admission),
        )
    }

    /// Occupy one of `worker`'s admission slots.
    async fn occupy(admission: &PrefillAdmission, worker: &Arc<dyn Worker>) -> PrefillReservation {
        let worker = Arc::clone(worker);
        admission
            .admit(None, move |capacity| {
                capacity.select(Arc::clone(&worker), ())
            })
            .await
            .map(|admitted| admitted.reservation)
            .unwrap_or_else(|_| panic!("admission of an idle worker should succeed"))
    }

    fn token_tree_has_tenant(policy: &CacheAwarePolicy, tokens: &[u32], worker_url: &str) -> bool {
        policy
            .open_repair_stream("test-model", TreeKind::Token)
            .expect("token tree should be initialized")
            .any(|entry| {
                matches!(
                    entry,
                    RepairEntry::Token {
                        tokens: path,
                        tenants,
                    } if path == tokens
                        && tenants
                            .iter()
                            .any(|(tenant, _)| tenant.as_ref() == worker_url)
                )
            })
    }

    async fn wait_for_queued(admission: &PrefillAdmission, expected: usize) {
        for _ in 0..1_000 {
            if admission.queued_requests() == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("admission queue never reached {expected} requests");
    }

    #[tokio::test]
    async fn admission_filters_full_prefill_before_policy_selection() {
        let model_id = "test-model";
        let registry = Arc::new(WorkerRegistry::new());
        let full = worker("grpc://prefill-full:30000", WorkerType::Prefill);
        let available = worker("grpc://prefill-available:30000", WorkerType::Prefill);
        let decode = worker("grpc://decode:30000", WorkerType::Decode);
        for worker in [&full, &available, &decode] {
            registry.register(Arc::clone(worker)).unwrap();
        }

        let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let occupied = occupy(&admission, &full).await;
        let stage = pd_stage(
            registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            Arc::clone(&admission),
        );

        let ((prefill, selected_decode, _), guard) = stage
            .admit_pd_pair(
                model_id,
                PlacementInputs::default(),
                None,
                None,
                false,
                VisionDemand::default(),
            )
            .await
            .unwrap_or_else(|_| panic!("admission should select the non-full prefill worker"));

        assert_eq!(prefill.url(), available.url());
        assert_eq!(selected_decode.url(), decode.url());
        assert_eq!(full.load(), 1);
        assert_eq!(available.load(), 1);
        assert_eq!(decode.load(), 0);
        drop(guard);
        drop(occupied);
        assert_eq!(full.load(), 0);
        assert_eq!(available.load(), 0);
    }

    #[tokio::test]
    async fn queued_cache_aware_request_commits_only_after_final_worker_selection() {
        let model_id = "test-model";
        let registry = Arc::new(WorkerRegistry::new());
        let first = worker("grpc://prefill-cache-1:30000", WorkerType::Prefill);
        let second = worker("grpc://prefill-cache-2:30000", WorkerType::Prefill);
        let decode = worker("grpc://decode-cache:30000", WorkerType::Decode);
        for worker in [&first, &second, &decode] {
            registry.register(Arc::clone(worker)).unwrap();
        }

        let cache_policy = Arc::new(CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        }));
        cache_policy.init_workers(&[Arc::clone(&first), Arc::clone(&second)]);
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        policy_registry.set_prefill_policy(cache_policy.clone());

        let admission = Arc::new(PrefillAdmission::new(1, 1, Duration::from_secs(5)));
        let occupied_first = occupy(&admission, &first).await;
        let occupied_second = occupy(&admission, &second).await;
        let stage = Arc::new(pd_stage(registry, policy_registry, Arc::clone(&admission)));
        let tokens: Vec<u32> = (1..=16).collect();

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let queued = tokio::spawn({
            let stage = Arc::clone(&stage);
            let tokens = tokens.clone();
            async move {
                let inputs = PlacementInputs {
                    tokens: Some(&tokens),
                    ..PlacementInputs::default()
                };
                stage
                    .admit_pd_pair(model_id, inputs, None, None, false, VisionDemand::default())
                    .await
            }
        });
        wait_for_queued(&admission, 1).await;
        assert_eq!(first.processed_requests(), 0);
        assert_eq!(second.processed_requests(), 0);
        assert!(!token_tree_has_tenant(&cache_policy, &tokens, first.url()));
        assert!(!token_tree_has_tenant(&cache_policy, &tokens, second.url()));

        drop(occupied_second);
        let ((selected_prefill, _, _), guard) = queued
            .await
            .expect("waiter should join")
            .unwrap_or_else(|_| panic!("queued admission should complete after capacity frees"));

        assert_eq!(selected_prefill.url(), second.url());
        assert_eq!(first.processed_requests(), 0);
        assert_eq!(second.processed_requests(), 1);
        assert!(!token_tree_has_tenant(&cache_policy, &tokens, first.url()));
        assert!(token_tree_has_tenant(&cache_policy, &tokens, second.url()));

        drop(guard);
        drop(occupied_first);
    }

    #[tokio::test]
    async fn admission_does_not_reassign_a_full_explicit_target() {
        let model_id = "test-model";
        let registry = Arc::new(WorkerRegistry::new());
        let first = worker("grpc://prefill-1:30000", WorkerType::Prefill);
        let second = worker("grpc://prefill-2:30000", WorkerType::Prefill);
        let decode = worker("grpc://decode:30000", WorkerType::Decode);
        for worker in [&first, &second, &decode] {
            registry.register(Arc::clone(worker)).unwrap();
        }

        // The target header indexes the prefill pool in registry order.
        let ordered_prefill = registry.get_routing_pool(model_id, RoutingPool::GrpcPrefill);
        assert_eq!(ordered_prefill.len(), 2);
        let target = Arc::clone(&ordered_prefill[0]);
        let alternative = Arc::clone(&ordered_prefill[1]);

        let admission = Arc::new(PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let occupied = occupy(&admission, &target).await;
        let stage = pd_stage(
            registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::ConsistentHashing)),
            Arc::clone(&admission),
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-target-worker", "0".parse().unwrap());
        let inputs = PlacementInputs {
            headers: Some(&headers),
            ..PlacementInputs::default()
        };

        let response = stage
            .admit_pd_pair(model_id, inputs, None, None, false, VisionDemand::default())
            .await
            .err()
            .expect("full explicit target must wait or reject, not reassign");

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            PD_PREFILL_QUEUE_FULL
        );
        assert_eq!(target.load(), 1);
        assert_eq!(alternative.load(), 0);
        assert_eq!(decode.load(), 0);
        drop(occupied);
    }

    fn hit_counts_in_order(urls: &[String], hits: &HashMap<String, usize>) -> Vec<usize> {
        urls.iter()
            .map(|url| hits.get(url).copied().unwrap_or(0))
            .collect()
    }

    /// Correctness bar for PD round-robin: every worker in both pools is hit
    /// equally across 40 `select_pd_pair` calls.
    fn assert_even_pd_round_robin_coverage(
        prefill_urls: &[String],
        decode_urls: &[String],
        prefill_hits: &HashMap<String, usize>,
        decode_hits: &HashMap<String, usize>,
    ) {
        assert_eq!(
            hit_counts_in_order(prefill_urls, prefill_hits),
            vec![10, 10, 10, 10],
            "even PD round-robin coverage: every prefill worker should get 10/40"
        );
        assert_eq!(
            hit_counts_in_order(decode_urls, decode_hits),
            vec![10, 10, 10, 10],
            "even PD round-robin coverage: every decode worker should get 10/40"
        );
    }

    /// Drive `select_pd_pair` through the stage (uses `get_prefill_policy` /
    /// `get_decode_policy` internally) and count selections by worker URL.
    fn count_select_pd_pair_hits(
        stage: &WorkerSelectionStage,
        model_id: &str,
        iterations: usize,
    ) -> (HashMap<String, usize>, HashMap<String, usize>) {
        let mut prefill_hits = HashMap::new();
        let mut decode_hits = HashMap::new();
        for _ in 0..iterations {
            let (prefill, decode, _) = stage
                .select_pd_pair(
                    model_id,
                    PlacementInputs::default(),
                    None,
                    None,
                    VisionDemand::default(),
                )
                .expect("select_pd_pair should return a pair");
            *prefill_hits.entry(prefill.url().to_string()).or_default() += 1;
            *decode_hits.entry(decode.url().to_string()).or_default() += 1;
        }
        (prefill_hits, decode_hits)
    }

    /// A saturated prefill leg is a pressure condition, not model absence.
    ///
    /// The model's decode workers stay unflagged, so a whole-model shed
    /// predicate reads "not all overloaded" and the request would fall through
    /// to 404 — the exact answer this feature exists to replace, and worse than
    /// the pre-feature behaviour where the prefill worker still served.
    #[test]
    fn a_fully_vetoed_prefill_leg_sheds_rather_than_404s() {
        let model_id = "test-model-prefill-veto";
        let worker_registry = Arc::new(WorkerRegistry::new());
        worker_registry.set_overload_shed(true);
        let (prefill_urls, _) = register_pd_workers(&worker_registry, model_id, 4);

        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::PrefillDecode,
            None,
        );
        assert!(stage
            .select_pd_pair(
                model_id,
                PlacementInputs::default(),
                None,
                None,
                VisionDemand::default()
            )
            .is_ok());

        for url in &prefill_urls {
            let worker = worker_registry.get_by_url(url).expect("registered");
            worker_registry.set_worker_overloaded(&worker, true);
        }

        assert!(
            stage
                .select_pd_pair(
                    model_id,
                    PlacementInputs::default(),
                    None,
                    None,
                    VisionDemand::default()
                )
                .is_err(),
            "the veto empties the prefill pool"
        );
        let response = stage.selection_failure(
            model_id,
            &[WorkerType::Prefill, WorkerType::Decode],
            None,
            TEXT,
        );
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "worker_overload_protection_shed"
        );
        assert!(
            !is_retryable_response(&response),
            "a shed must be terminal for the retry layer"
        );
    }

    /// An undemanded leg cannot shed: a text-only EPD request that fails for a
    /// non-overload reason must not 503 just because the (unused) encode pool
    /// is saturated.
    #[test]
    fn an_undemanded_encode_leg_cannot_shed() {
        let model_id = "test-model-encode-veto";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let encode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:8460")
                .model(ModelCard::new(model_id))
                .worker_type(WorkerType::Encode)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        );
        worker_registry.register(Arc::clone(&encode)).unwrap();
        worker_registry.set_worker_overloaded(&encode, true);

        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::EncodePrefillDecode,
            None,
        );

        // No prefill/decode workers registered: with encode undemanded this is
        // model absence (404), not pressure.
        let text_only = stage.selection_failure(
            model_id,
            &[WorkerType::Prefill, WorkerType::Decode],
            None,
            TEXT,
        );
        assert_eq!(text_only.status(), StatusCode::NOT_FOUND);

        // With encode demanded, the saturated encode pool is a shed.
        let with_encode = stage.selection_failure(
            model_id,
            &[WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode],
            None,
            TEXT,
        );
        assert_eq!(with_encode.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// A model nobody serves is still a 404 — the shed must not swallow real
    /// misconfiguration.
    #[test]
    fn an_unserved_model_still_reports_not_found() {
        let stage = WorkerSelectionStage::new(
            Arc::new(WorkerRegistry::new()),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::PrefillDecode,
            None,
        );
        assert_eq!(
            stage
                .selection_failure(
                    "nobody",
                    &[WorkerType::Prefill, WorkerType::Decode],
                    None,
                    TEXT
                )
                .status(),
            StatusCode::NOT_FOUND
        );
    }

    #[test]
    fn select_pd_pair_shared_round_robin_keeps_each_leg_even() {
        // Same correctness bar as the independent test. One shared RoundRobin
        // Arc for P/D used to advance a single counter twice per request and
        // pin each leg to half its workers; the rotation is per candidate
        // set now, so even a shared instance covers both legs evenly.
        let model_id = "test-model-shared";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let (prefill_urls, decode_urls) = register_pd_workers(&worker_registry, model_id, 4);

        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let shared = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
        policy_registry.set_prefill_policy(Arc::clone(&shared));
        policy_registry.set_decode_policy(shared);
        assert!(Arc::ptr_eq(
            &policy_registry.get_prefill_policy(),
            &policy_registry.get_decode_policy()
        ));

        let stage = WorkerSelectionStage::new(
            worker_registry,
            policy_registry,
            WorkerSelectionMode::PrefillDecode,
            None,
        );
        let (prefill_hits, decode_hits) = count_select_pd_pair_hits(&stage, model_id, 40);
        assert_even_pd_round_robin_coverage(
            &prefill_urls,
            &decode_urls,
            &prefill_hits,
            &decode_hits,
        );
    }

    #[test]
    fn select_pd_pair_independent_round_robin_passes_even_coverage() {
        // Production PD startup: two independent RoundRobinPolicy instances.
        // Same correctness bar; this configuration must pass.
        let model_id = "test-model-independent";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let (prefill_urls, decode_urls) = register_pd_workers(&worker_registry, model_id, 4);

        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        policy_registry
            .set_prefill_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        policy_registry
            .set_decode_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        assert!(!Arc::ptr_eq(
            &policy_registry.get_prefill_policy(),
            &policy_registry.get_decode_policy()
        ));

        let stage = WorkerSelectionStage::new(
            worker_registry,
            policy_registry,
            WorkerSelectionMode::PrefillDecode,
            None,
        );
        let (prefill_hits, decode_hits) = count_select_pd_pair_hits(&stage, model_id, 40);
        assert_even_pd_round_robin_coverage(
            &prefill_urls,
            &decode_urls,
            &prefill_hits,
            &decode_hits,
        );
    }

    #[test]
    fn select_pd_pair_ignores_zmq_legs() {
        // The ZMQ wire carries no KV-transfer rendezvous, so ZMQ prefill/decode
        // workers must never be paired even if they reach the registry.
        let model_id = "test-model-zmq";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (port, worker_type) in [(9000, WorkerType::Prefill), (9100, WorkerType::Decode)] {
            worker_registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(format!("ipc:///tmp/smg-zmq/{port}.ipc"))
                        .model(ModelCard::new(model_id))
                        .worker_type(worker_type)
                        .connection_mode(ConnectionMode::Zmq)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }

        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        policy_registry
            .set_prefill_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        policy_registry
            .set_decode_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::clone(&policy_registry),
            WorkerSelectionMode::PrefillDecode,
            None,
        );

        assert!(
            stage
                .select_pd_pair(
                    model_id,
                    PlacementInputs::default(),
                    None,
                    None,
                    VisionDemand::default()
                )
                .is_err(),
            "ZMQ-only PD pools must not yield a pair"
        );

        // Adding gRPC legs makes selection succeed, and it never picks the ZMQ ones.
        let (prefill_urls, decode_urls) = register_pd_workers(&worker_registry, model_id, 4);
        let (prefill, decode, _) = stage
            .select_pd_pair(
                model_id,
                PlacementInputs::default(),
                None,
                None,
                VisionDemand::default(),
            )
            .expect("gRPC PD pair should be selected");
        assert!(prefill_urls.contains(&prefill.url().to_string()));
        assert!(decode_urls.contains(&decode.url().to_string()));
    }

    /// gRPC selection honors the conversation header across fresh request IDs.
    #[test]
    fn grpc_selection_pins_by_header_key_under_override() {
        use crate::config::types::{ManualAssignmentMode, RoutingKeyOverrideConfig};

        let model_id = "test-model-rid-sticky";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for i in 0..2 {
            worker_registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{}", 8300 + i))
                        .model(ModelCard::new(model_id))
                        .worker_type(WorkerType::Regular)
                        .connection_mode(ConnectionMode::Grpc)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }
        let policy_registry = Arc::new(PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                assignment_mode: ManualAssignmentMode::Delegate,
                ..Default::default()
            },
        ));
        let stage = WorkerSelectionStage::new(
            worker_registry,
            policy_registry.clone(),
            WorkerSelectionMode::Regular,
            None,
        );

        let rid_key = policy_registry.derive_rid_key(Some("conv7_t1"));
        assert_eq!(rid_key, Some("conv7"));

        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "conversation-42".parse().unwrap());
        let first = stage
            .select_single_worker(
                model_id,
                None,
                None,
                Some(&headers),
                rid_key,
                None,
                None,
                None,
            )
            .unwrap();
        for rid in ["request-2", "request-3", "request-4"] {
            let again = stage
                .select_single_worker(
                    model_id,
                    None,
                    None,
                    Some(&headers),
                    policy_registry.derive_rid_key(Some(rid)),
                    None,
                    None,
                    None,
                )
                .unwrap();
            assert_eq!(again.url(), first.url(), "follow-up must pin by header key");
        }
    }

    /// The gRPC transport must shed an all-overloaded model with the same
    /// distinct overload 503 the HTTP router uses. Its usual empty-pool answer
    /// is a 404, which would both misreport pressure as model absence and skip
    /// the retry path (404 is not retryable).
    #[test]
    fn grpc_all_overloaded_sheds_503_instead_of_404() {
        use crate::routers::error::extract_error_code_from_response;

        let model_id = "test-model-overload-shed";
        let worker_registry = Arc::new(WorkerRegistry::new());
        worker_registry.set_overload_shed(true);
        let mut workers = Vec::new();
        for i in 0..2 {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{}", 8400 + i))
                    .model(ModelCard::new(model_id))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .health_config(no_health_check())
                    .build(),
            );
            worker_registry.register(Arc::clone(&worker)).unwrap();
            workers.push(worker);
        }
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            policy_registry,
            WorkerSelectionMode::Regular,
            None,
        );

        assert!(stage
            .select_single_worker(model_id, None, None, None, None, None, None, None)
            .is_some());

        worker_registry.set_worker_overloaded(&workers[0], true);
        assert!(
            stage
                .select_single_worker(model_id, None, None, None, None, None, None, None)
                .is_some(),
            "one eligible worker left still serves"
        );

        worker_registry.set_worker_overloaded(&workers[1], true);
        assert!(
            stage
                .select_single_worker(model_id, None, None, None, None, None, None, None)
                .is_none(),
            "the veto empties the candidate pool"
        );

        let response = stage.selection_failure(model_id, &[WorkerType::Regular], None, TEXT);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "worker_overload_protection_shed"
        );

        // Recovery re-admits, and the failure response goes back to 404 for a
        // genuinely absent model.
        worker_registry.set_worker_overloaded(&workers[0], false);
        assert!(stage
            .select_single_worker(model_id, None, None, None, None, None, None, None)
            .is_some());
        assert_eq!(
            stage
                .selection_failure("no-such-model", &[WorkerType::Regular], None, TEXT)
                .status(),
            StatusCode::NOT_FOUND
        );
    }

    /// Workers serve the model but none is available: a 503 with the HTTP
    /// router's code, not the 404 that told clients the model was gone while
    /// its workers restarted.
    #[test]
    fn an_unavailable_regular_worker_answers_503_not_404() {
        use openai_protocol::worker::WorkerStatus;

        use crate::routers::error::extract_error_code_from_response;
        let model_id = "test-model-unavailable";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:8470")
                .model(ModelCard::new(model_id))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        );
        worker_registry.register(Arc::clone(&worker)).unwrap();
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::Regular,
            None,
        );
        // Any status but Ready is unavailable to routing.
        worker.set_status(WorkerStatus::NotReady);
        assert!(stage
            .select_single_worker(model_id, None, None, None, None, None, None, None)
            .is_none());

        let response = stage.selection_failure(model_id, &[WorkerType::Regular], None, TEXT);

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "no_available_workers"
        );
        assert_eq!(retry_after(&response), Some("1"));
        // A model nobody serves stays a 404.
        assert_eq!(
            stage
                .selection_failure("no-such-model", &[WorkerType::Regular], None, TEXT)
                .status(),
            StatusCode::NOT_FOUND
        );
    }

    /// A disaggregated leg whose only worker is down is the same 503, both
    /// through the per-leg fallback and through the pair verdict.
    #[test]
    fn an_unavailable_decode_leg_answers_503_not_404() {
        use openai_protocol::worker::WorkerStatus;

        use crate::{policies::WorkerLeg, routers::error::extract_error_code_from_response};
        let model_id = "test-model-decode-down";
        let worker_registry = Arc::new(WorkerRegistry::new());
        register_pd_workers(&worker_registry, model_id, 1);
        let decode = worker_registry
            .get_routing_pool(model_id, RoutingPool::GrpcDecode)
            .first()
            .cloned()
            .expect("one decode worker registered");
        decode.set_status(WorkerStatus::NotReady);
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::PrefillDecode,
            None,
        );

        let fallback = stage.selection_failure(
            model_id,
            &[WorkerType::Prefill, WorkerType::Decode],
            None,
            TEXT,
        );
        let pair = stage.pair_failure(
            model_id,
            PairFailure {
                leg: WorkerLeg::Decode,
                verdict: PlacementFailure::Unavailable,
            },
        );

        for response in [fallback, pair] {
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(
                extract_error_code_from_response(&response),
                "no_available_workers"
            );
            assert_eq!(retry_after(&response), Some("1"));
        }
        let absent = stage.pair_failure(
            model_id,
            PairFailure {
                leg: WorkerLeg::Decode,
                verdict: PlacementFailure::NoCandidates,
            },
        );
        assert_eq!(absent.status(), StatusCode::NOT_FOUND);
    }

    fn dispatch_ctx(model_id: &str, wire: WireConstraint) -> DispatchContext {
        DispatchContext {
            root_request_id: None,
            request_id: None,
            cache_trace: None,
            attempt: 0,
            ledger: AttemptLedger::default(),
            tried: Vec::new(),
            model_id: model_id.to_string(),
            dispatch_model: model_id.to_string(),
            streaming: false,
            headers: None,
            rate_limit_cell: None,
            routing: RoutingSnapshot {
                routing_text: None,
                token_ids: vec![1, 2, 3],
                rid_key: None,
                cache_namespace: None,
                needs_vision: false,
                decode_needs_vision: false,
            },
            wire,
            tokenizer: None,
            workers: None,
            sticky_key: None,
            clients: None,
            encode_outputs: None,
            dispatch: None,
            load_guards: None,
            multimodal_inflight: None,
            pd_prefill_guard: None,
            response: Default::default(),
        }
    }

    /// Retry re-selection must never leave the retained plan's wire: the
    /// runtime AND transport filters both apply, or a retry could pick a
    /// worker the plan's proto flavor cannot be dispatched to.
    #[tokio::test]
    async fn reselect_pins_regular_candidates_to_the_retained_wire() {
        let model_id = "wire-pin-model";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, runtime, connection) in [
            (
                "grpc://127.0.0.1:9300",
                RuntimeType::Sglang,
                ConnectionMode::Grpc,
            ),
            (
                "grpc://127.0.0.1:9301",
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
            ),
            (
                "ipc:///tmp/smg-wire-pin",
                RuntimeType::Vllm,
                ConnectionMode::Zmq,
            ),
        ] {
            worker_registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .model(ModelCard::new(model_id))
                        .worker_type(WorkerType::Regular)
                        .connection_mode(connection)
                        .runtime_type(runtime)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }
        let stage = WorkerSelectionStage::new(
            worker_registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::Regular,
            None,
        );

        let mut ctx = dispatch_ctx(
            model_id,
            WireConstraint {
                runtime: RuntimeType::Vllm,
                connection: ConnectionMode::Grpc,
                requires_media_refs: false,
            },
        );
        for _ in 0..8 {
            stage.reselect(&mut ctx).await.unwrap();
            match ctx.workers.as_ref().unwrap() {
                WorkerSelection::Single { worker } => {
                    assert_eq!(
                        worker.url(),
                        "grpc://127.0.0.1:9301",
                        "reselect must stay on the retained runtime and transport"
                    );
                }
                WorkerSelection::Disaggregated { .. } => panic!("expected single selection"),
            }
        }
    }

    /// A keyed request's retry re-selects away from the engine its failed
    /// attempt used while another worker on the retained wire is left; with
    /// every engine tried the pool is used whole again.
    #[tokio::test]
    async fn reselect_skips_the_engines_the_failed_attempts_used() {
        let model_id = "tried-model";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for url in [
            "grpc://127.0.0.1:9310",
            "grpc://127.0.0.1:9311",
            "grpc://127.0.0.1:9312",
        ] {
            worker_registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .model(ModelCard::new(model_id))
                        .worker_type(WorkerType::Regular)
                        .connection_mode(ConnectionMode::Grpc)
                        .runtime_type(RuntimeType::Vllm)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }
        let stage = WorkerSelectionStage::new(
            worker_registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::ConsistentHashing)),
            WorkerSelectionMode::Regular,
            None,
        );
        let mut ctx = dispatch_ctx(
            model_id,
            WireConstraint {
                runtime: RuntimeType::Vllm,
                connection: ConnectionMode::Grpc,
                requires_media_refs: false,
            },
        );
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "session-42".parse().unwrap());
        ctx.headers = Some(headers);
        let selected = |ctx: &DispatchContext| match ctx.workers.as_ref().unwrap() {
            WorkerSelection::Single { worker } => Arc::clone(worker),
            WorkerSelection::Disaggregated { .. } => panic!("expected single selection"),
        };

        stage.reselect(&mut ctx).await.unwrap();
        let owner = selected(&ctx);
        stage.reselect(&mut ctx).await.unwrap();
        assert_eq!(
            selected(&ctx).url(),
            owner.url(),
            "without a failed attempt the key re-selects its owner"
        );

        ctx.tried.push(owner.base_url().to_string());
        stage.reselect(&mut ctx).await.unwrap();
        let second = selected(&ctx);
        assert_ne!(
            second.url(),
            owner.url(),
            "the retry leaves the tried engine"
        );

        ctx.tried.push(second.base_url().to_string());
        stage.reselect(&mut ctx).await.unwrap();
        let third = selected(&ctx);
        assert!(
            third.url() != owner.url() && third.url() != second.url(),
            "the second retry leaves both tried engines"
        );

        ctx.tried.push(third.base_url().to_string());
        stage.reselect(&mut ctx).await.unwrap();
        assert_eq!(
            selected(&ctx).url(),
            owner.url(),
            "with every engine tried the pool is used whole again"
        );
    }

    #[tokio::test]
    async fn reselect_pins_pd_pair_to_the_retained_runtime() {
        let model_id = "wire-pin-pd-model";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, runtime) in [
            (
                "grpc://127.0.0.1:9310",
                WorkerType::Prefill,
                RuntimeType::Sglang,
            ),
            (
                "grpc://127.0.0.1:9311",
                WorkerType::Decode,
                RuntimeType::Sglang,
            ),
            (
                "grpc://127.0.0.1:9312",
                WorkerType::Prefill,
                RuntimeType::Vllm,
            ),
            (
                "grpc://127.0.0.1:9313",
                WorkerType::Decode,
                RuntimeType::Vllm,
            ),
        ] {
            worker_registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .model(ModelCard::new(model_id))
                        .worker_type(worker_type)
                        .connection_mode(ConnectionMode::Grpc)
                        .runtime_type(runtime)
                        .health_config(no_health_check())
                        .build(),
                ))
                .unwrap();
        }
        let stage = WorkerSelectionStage::new(
            worker_registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::PrefillDecode,
            None,
        );

        let mut ctx = dispatch_ctx(
            model_id,
            WireConstraint {
                runtime: RuntimeType::Vllm,
                connection: ConnectionMode::Grpc,
                requires_media_refs: false,
            },
        );
        for _ in 0..8 {
            stage.reselect(&mut ctx).await.unwrap();
            match ctx.workers.as_ref().unwrap() {
                WorkerSelection::Disaggregated {
                    prefill,
                    decode,
                    runtime_type,
                    ..
                } => {
                    assert_eq!(*runtime_type, RuntimeType::Vllm);
                    assert_eq!(prefill.url(), "grpc://127.0.0.1:9312");
                    assert_eq!(decode.url(), "grpc://127.0.0.1:9313");
                }
                WorkerSelection::Single { .. } => panic!("expected PD selection"),
            }
        }
    }

    /// A drained pinned pool must shed (503), not answer 404 just because
    /// another runtime still serves the model: 404 is non-retryable and would
    /// end the retry loop on a lie.
    #[tokio::test]
    async fn reselect_verdict_reflects_the_pinned_pool() {
        let model_id = "wire-verdict-model";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let mut pinned = None;
        for (url, runtime) in [
            ("grpc://127.0.0.1:9320", RuntimeType::Sglang),
            ("grpc://127.0.0.1:9321", RuntimeType::Vllm),
        ] {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new(model_id))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(runtime)
                    .health_config(no_health_check())
                    .build(),
            );
            worker_registry.register(worker.clone()).unwrap();
            if runtime == RuntimeType::Vllm {
                pinned = Some(worker);
            }
        }
        worker_registry.set_worker_overloaded(&pinned.expect("vllm worker registered"), true);
        // Under shedding; the steering default would route the retry to the
        // vetoed worker instead, which is the other test's subject.
        worker_registry.set_overload_shed(true);

        let stage = WorkerSelectionStage::new(
            worker_registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::Regular,
            None,
        );
        let mut ctx = dispatch_ctx(
            model_id,
            WireConstraint {
                runtime: RuntimeType::Vllm,
                connection: ConnectionMode::Grpc,
                requires_media_refs: false,
            },
        );

        let response = stage
            .reselect(&mut ctx)
            .await
            .expect_err("the pinned pool is fully vetoed");
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "verdict must reflect the pinned pool, not every runtime"
        );
    }

    /// The namespace lives on the routing snapshot so a retry re-selects in
    /// the same cache partition: the retained snapshot, not a re-derivation
    /// from a request that may already be released, decides the key.
    #[tokio::test]
    async fn reselect_keys_affinity_under_the_retained_cache_namespace() {
        use openai_protocol::common::CachePartition;

        let model_id = "namespace-retry-model";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let workers: Vec<Arc<dyn Worker>> = ["grpc://127.0.0.1:9401", "grpc://127.0.0.1:9402"]
            .iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .model(ModelCard::new(model_id))
                        .worker_type(WorkerType::Regular)
                        .connection_mode(ConnectionMode::Grpc)
                        .runtime_type(RuntimeType::Vllm)
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        for worker in &workers {
            worker_registry.register(Arc::clone(worker)).unwrap();
        }
        let stage = WorkerSelectionStage::new(
            worker_registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 0,
                max_tree_size: 4096,
                block_size: 16,
                balance_token_usage_threshold: 1.0,
                overload_token_usage_threshold: 1.0,
                overlap_decay: 0.0,
                selection_temperature: 0.0,
                cache_index: Default::default(),
                cache_ttl_secs: 180,
                cache_boundaries: Vec::new(),
                selection_policy: None,
                selection_accounting_ttl_ms: 0,
            })),
            WorkerSelectionMode::Regular,
            None,
        );
        let wire = WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Grpc,
            requires_media_refs: false,
        };
        let namespace = |salt: &str| {
            CacheNamespace::derive(&CachePartition {
                cache_salt: Some(salt),
                extra_key: None,
                lora_path: None,
            })
        };
        let selected = |ctx: &DispatchContext| match ctx.workers.as_ref().unwrap() {
            WorkerSelection::Single { worker } => worker.url().to_string(),
            WorkerSelection::Disaggregated { .. } => panic!("expected single selection"),
        };
        let prompt: Vec<u32> = (1..33).collect();

        // Worker 1 is busier, so tenant A's first selection lands on worker 2.
        workers[0].increment_load();
        let mut ctx = dispatch_ctx(model_id, wire);
        ctx.routing.token_ids = prompt.clone();
        ctx.routing.cache_namespace = namespace("tenant-a");
        stage.reselect(&mut ctx).await.unwrap();
        assert_eq!(selected(&ctx), "grpc://127.0.0.1:9402");

        // A retry from the retained snapshot stays in tenant A's partition
        // even once worker 2 is the busier one...
        workers[1].increment_load();
        workers[1].increment_load();
        stage.reselect(&mut ctx).await.unwrap();
        assert_eq!(selected(&ctx), "grpc://127.0.0.1:9402");

        // ...while a retained snapshot for tenant B misses and takes the
        // least-loaded worker 1.
        let mut other = dispatch_ctx(model_id, wire);
        other.routing.token_ids = prompt;
        other.routing.cache_namespace = namespace("tenant-b");
        stage.reselect(&mut other).await.unwrap();
        assert_eq!(selected(&other), "grpc://127.0.0.1:9401");
    }

    fn vllm_grpc_worker(
        url: &str,
        model_id: &str,
        worker_type: WorkerType,
        advertise_refs: bool,
    ) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model_id))
            .worker_type(worker_type)
            .runtime_type(RuntimeType::Vllm)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check());
        if advertise_refs {
            builder = builder
                .label("mm_processor", "inprocess")
                .label("mm_media_ref_schemes", "http,https,data");
        }
        Arc::new(builder.build())
    }

    /// Media references only reach workers advertising worker-side processing,
    /// on first selection and on the wire-pinned retry path alike.
    #[test]
    fn media_refs_selection_filters_to_advertising_workers() {
        let model_id = "test-model-media-refs";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let plain_url = "grpc://127.0.0.1:8700";
        let capable_url = "grpc://127.0.0.1:8701";
        for (url, advertise) in [(plain_url, false), (capable_url, true)] {
            worker_registry
                .register(vllm_grpc_worker(
                    url,
                    model_id,
                    WorkerType::Regular,
                    advertise,
                ))
                .unwrap();
        }
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::clone(&policy_registry),
            WorkerSelectionMode::Regular,
            None,
        );

        for _ in 0..4 {
            let worker = stage
                .select_single_worker(
                    model_id,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(ACCEPTS_MEDIA_REFS),
                )
                .expect("advertising worker is selectable");
            assert_eq!(worker.url(), capable_url);
        }

        let wire = WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Grpc,
            requires_media_refs: true,
        };
        let worker = stage
            .select_single_worker(model_id, None, None, None, None, None, Some(wire), None)
            .expect("retry re-selection stays on advertising workers");
        assert_eq!(worker.url(), capable_url);

        let mut seen = HashMap::new();
        for _ in 0..4 {
            let worker = stage
                .select_single_worker(model_id, None, None, None, None, None, None, None)
                .expect("any worker without refs");
            *seen.entry(worker.url().to_string()).or_insert(0) += 1;
        }
        assert_eq!(seen.len(), 2, "without refs both workers are eligible");
    }

    #[test]
    fn media_refs_selection_sheds_when_no_worker_advertises() {
        let model_id = "test-model-media-refs-none";
        let worker_registry = Arc::new(WorkerRegistry::new());
        worker_registry
            .register(vllm_grpc_worker(
                "grpc://127.0.0.1:8710",
                model_id,
                WorkerType::Regular,
                false,
            ))
            .unwrap();
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::clone(&policy_registry),
            WorkerSelectionMode::Regular,
            None,
        );

        assert!(stage
            .select_single_worker(
                model_id,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(ACCEPTS_MEDIA_REFS),
            )
            .is_none());
        let response = stage.selection_failure(model_id, &[WorkerType::Regular], None, REFS);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "no_media_ref_capable_worker"
        );
        assert!(
            !is_retryable_response(&response),
            "capability cannot change inside a retry window"
        );
        // The wire pin alone carries the media-refs fact on retries.
        let wire = WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Grpc,
            requires_media_refs: true,
        };
        assert!(stage
            .select_single_worker(model_id, None, None, None, None, None, Some(wire), None)
            .is_none());
        let response = stage.selection_failure(model_id, &[WorkerType::Regular], Some(wire), TEXT);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "no_media_ref_capable_worker"
        );

        // An absent model stays a 404 even in worker mode.
        let response = stage.selection_failure("no-such-model", &[WorkerType::Regular], None, REFS);
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// The media demands of a text request, of media references, and of
    /// pixel media.
    const TEXT: MediaDemands = MediaDemands::NONE;
    const REFS: MediaDemands = MediaDemands {
        media_refs: true,
        vision: false,
    };
    const VISION: MediaDemands = MediaDemands {
        media_refs: false,
        vision: true,
    };

    /// A regular worker with `labels`, built as [`labeled_worker`] builds every
    /// worker of these tests.
    fn regular_worker(url: &str, model_id: &str, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
        labeled_worker(url, model_id, WorkerType::Regular, labels)
    }

    fn urls_seen(stage: &WorkerSelectionStage, model_id: &str, media: MediaDemands) -> Vec<String> {
        let accepts = |worker: &dyn Worker| media.accepts(worker);
        let filter: Option<CandidateFilter<'_>> = media.any().then_some(&accepts);
        let mut seen = HashMap::new();
        for _ in 0..6 {
            let worker = stage
                .select_single_worker(model_id, None, None, None, None, None, None, filter)
                .expect("a worker is selectable");
            *seen.entry(worker.url().to_string()).or_insert(0) += 1;
        }
        let mut urls: Vec<String> = seen.into_keys().collect();
        urls.sort();
        urls
    }

    /// Pixel media never reaches a worker labeled without a vision encoder:
    /// in a mixed pool every image request lands on a worker with one or
    /// without a verdict, while a text request still uses the whole pool.
    #[test]
    fn pixel_media_selection_skips_language_model_only_workers() {
        let model_id = "test-model-pixel-media-mixed";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, labels) in [
            ("grpc://127.0.0.1:8760", NO_VISION_ENCODER),
            ("grpc://127.0.0.1:8761", VISION_ENCODER),
            ("grpc://127.0.0.1:8762", UNLABELED),
        ] {
            worker_registry
                .register(regular_worker(url, model_id, labels))
                .unwrap();
        }
        let stage = regular_stage(Arc::clone(&worker_registry));

        assert_eq!(
            urls_seen(&stage, model_id, VISION),
            ["grpc://127.0.0.1:8761", "grpc://127.0.0.1:8762"],
            "an image goes to the worker with the encoder or without a verdict"
        );
        assert_eq!(
            urls_seen(&stage, model_id, TEXT),
            [
                "grpc://127.0.0.1:8760",
                "grpc://127.0.0.1:8761",
                "grpc://127.0.0.1:8762"
            ],
            "a text request uses the whole pool"
        );
    }

    /// With no worker with a vision encoder serving the model, an image
    /// request is refused at once with the modality 400 a model without
    /// vision gives, not dispatched to a worker that never schedules it and
    /// not the 503 of an unavailable pool; the same worker still serves text,
    /// and an absent model stays a 404.
    #[test]
    fn pixel_media_selection_refuses_when_no_worker_has_a_vision_encoder() {
        let model_id = "test-model-pixel-media-none";
        let worker_registry = Arc::new(WorkerRegistry::new());
        worker_registry
            .register(regular_worker(
                "grpc://127.0.0.1:8770",
                model_id,
                NO_VISION_ENCODER,
            ))
            .unwrap();
        let stage = regular_stage(Arc::clone(&worker_registry));

        assert!(stage
            .select_single_worker(
                model_id,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(&vision_capable),
            )
            .is_none());
        let response = stage.selection_failure(model_id, &[WorkerType::Regular], None, VISION);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "multimodal_not_supported"
        );
        assert!(
            !is_retryable_response(&response),
            "capability cannot change inside a retry window"
        );

        assert_eq!(
            urls_seen(&stage, model_id, TEXT),
            ["grpc://127.0.0.1:8770"],
            "text still goes to the language-model-only worker"
        );

        let response =
            stage.selection_failure("no-such-model", &[WorkerType::Regular], None, VISION);
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// A vision-capable worker that is merely overloaded keeps the overload
    /// verdict, with the language-model-only worker still out of the
    /// candidates: the refusal is for a pool that cannot take pixels at all.
    #[test]
    fn pixel_media_overloaded_vision_workers_keep_the_overload_shed() {
        let model_id = "test-model-pixel-media-overload";
        let worker_registry = Arc::new(WorkerRegistry::new());
        worker_registry.set_overload_shed(true);
        worker_registry
            .register(regular_worker(
                "grpc://127.0.0.1:8780",
                model_id,
                NO_VISION_ENCODER,
            ))
            .unwrap();
        let capable = regular_worker("grpc://127.0.0.1:8781", model_id, VISION_ENCODER);
        worker_registry.register(Arc::clone(&capable)).unwrap();
        let stage = regular_stage(Arc::clone(&worker_registry));
        worker_registry.set_worker_overloaded(&capable, true);

        assert!(stage
            .select_single_worker(
                model_id,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(&vision_capable),
            )
            .is_none());
        let response = stage.selection_failure(model_id, &[WorkerType::Regular], None, VISION);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_ne!(
            error::extract_error_code_from_response(&response),
            "multimodal_not_supported"
        );
        assert!(response.headers().contains_key(RETRY_AFTER));
    }

    /// Pixel media forwarded as references needs both capabilities; the
    /// verdicts are judged vision first: a pool whose reference-processing
    /// workers all lack the encoder but which has an encoder elsewhere is the
    /// media-refs shed, one with no encoder at all the modality refusal.
    #[test]
    fn pixel_media_as_references_needs_both_capabilities() {
        let both = MediaDemands {
            media_refs: true,
            vision: true,
        };
        let refs_with_vision = [
            ("mm_processor", "inprocess"),
            ("mm_media_ref_schemes", "http,https,data"),
            (multimodal::SUPPORTS_VISION_LABEL, "true"),
        ];
        let refs_without_vision = [
            ("mm_processor", "inprocess"),
            ("mm_media_ref_schemes", "http,https,data"),
            (multimodal::SUPPORTS_VISION_LABEL, "false"),
        ];
        let worker_registry = Arc::new(WorkerRegistry::new());
        let stage = regular_stage(Arc::clone(&worker_registry));

        let mixed = "test-model-pixel-refs-mixed";
        for (url, labels) in [
            ("grpc://127.0.0.1:8790", &refs_with_vision[..]),
            ("grpc://127.0.0.1:8791", &refs_without_vision[..]),
            ("grpc://127.0.0.1:8792", VISION_ENCODER),
        ] {
            worker_registry
                .register(regular_worker(url, mixed, labels))
                .unwrap();
        }
        assert_eq!(urls_seen(&stage, mixed, both), ["grpc://127.0.0.1:8790"]);

        let no_capable_refs = "test-model-pixel-refs-shed";
        for (url, labels) in [
            ("grpc://127.0.0.1:8793", &refs_without_vision[..]),
            ("grpc://127.0.0.1:8794", VISION_ENCODER),
        ] {
            worker_registry
                .register(regular_worker(url, no_capable_refs, labels))
                .unwrap();
        }
        let response = stage.selection_failure(no_capable_refs, &[WorkerType::Regular], None, both);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "no_media_ref_capable_worker"
        );

        let no_vision = "test-model-pixel-refs-refused";
        worker_registry
            .register(regular_worker(
                "grpc://127.0.0.1:8795",
                no_vision,
                &refs_without_vision,
            ))
            .unwrap();
        let response = stage.selection_failure(no_vision, &[WorkerType::Regular], None, both);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "multimodal_not_supported"
        );
    }

    /// The retry path re-selects under the snapshot's pixel-media fact:
    /// a retried image request stays off the language-model-only worker.
    #[tokio::test]
    async fn reselect_keeps_pixel_media_off_language_model_only_workers() {
        let model_id = "test-model-pixel-media-reselect";
        let worker_registry = Arc::new(WorkerRegistry::new());
        worker_registry
            .register(regular_worker(
                "grpc://127.0.0.1:8800",
                model_id,
                NO_VISION_ENCODER,
            ))
            .unwrap();
        worker_registry
            .register(regular_worker(
                "grpc://127.0.0.1:8801",
                model_id,
                VISION_ENCODER,
            ))
            .unwrap();
        let stage = regular_stage(Arc::clone(&worker_registry));
        let wire = WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Grpc,
            requires_media_refs: false,
        };

        let stage = &stage;
        let reselected = |needs_vision: bool| async move {
            let mut seen = HashMap::new();
            for _ in 0..4 {
                let mut ctx = dispatch_ctx(model_id, wire);
                ctx.routing.needs_vision = needs_vision;
                stage
                    .reselect(&mut ctx)
                    .await
                    .expect("re-selection succeeds");
                match ctx.workers.expect("a worker is selected") {
                    WorkerSelection::Single { worker } => {
                        *seen.entry(worker.url().to_string()).or_insert(0) += 1;
                    }
                    WorkerSelection::Disaggregated { .. } => panic!("regular mode selects one"),
                }
            }
            let mut urls: Vec<String> = seen.into_keys().collect();
            urls.sort();
            urls
        };
        assert_eq!(reselected(true).await, ["grpc://127.0.0.1:8801"]);
        assert_eq!(
            reselected(false).await,
            ["grpc://127.0.0.1:8800", "grpc://127.0.0.1:8801"]
        );
    }

    /// The stage reads the demand off the request's media: an image
    /// forwarded as a reference selects a reference-processing worker with a
    /// vision encoder and the routing snapshot carries the fact for retries,
    /// while a text request uses the whole pool.
    #[tokio::test]
    async fn execute_keeps_an_image_request_off_language_model_only_workers() {
        use llm_multimodal::MediaContentPart;

        let registry = Arc::new(WorkerRegistry::new());
        for (url, verdict) in [
            ("grpc://127.0.0.1:8810", "false"),
            ("grpc://127.0.0.1:8811", "true"),
        ] {
            let labels = [
                ("mm_processor", "inprocess"),
                ("mm_media_ref_schemes", "http,https,data"),
                (multimodal::SUPPORTS_VISION_LABEL, verdict),
            ];
            registry
                .register(regular_worker(url, "test-model", &labels))
                .unwrap();
        }
        let stage = regular_stage(Arc::clone(&registry));
        let image = multimodal::MediaPlan::new([MediaContentPart::ImageUrl {
            url: "data:image/png;base64,AAAA".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }]);

        for _ in 0..4 {
            let mut ctx = prepared_ctx(Arc::clone(&registry), None);
            ctx.state.multimodal_refs = Some(image.clone());
            stage
                .execute(&mut ctx)
                .await
                .expect("an image request selects");
            match ctx.state.workers.as_ref().expect("a worker is selected") {
                WorkerSelection::Single { worker } => {
                    assert_eq!(worker.url(), "grpc://127.0.0.1:8811");
                }
                WorkerSelection::Disaggregated { .. } => panic!("regular mode selects one"),
            }
            let snapshot = ctx.state.routing_snapshot.as_ref().expect("snapshot");
            assert!(snapshot.needs_vision);
        }

        let mut seen = HashMap::new();
        for _ in 0..4 {
            let mut ctx = prepared_ctx(Arc::clone(&registry), None);
            stage
                .execute(&mut ctx)
                .await
                .expect("a text request selects");
            if let Some(WorkerSelection::Single { worker }) = ctx.state.workers.as_ref() {
                *seen.entry(worker.url().to_string()).or_insert(0) += 1;
            }
            let snapshot = ctx.state.routing_snapshot.as_ref().expect("snapshot");
            assert!(!snapshot.needs_vision);
        }
        assert_eq!(
            seen.len(),
            2,
            "a text request uses the whole pool: {seen:?}"
        );
    }

    /// The steering default: with every worker over the thresholds the gRPC
    /// path routes to the least-loaded one instead of refusing, on the single
    /// path and on both PD legs.
    #[test]
    fn grpc_all_overloaded_steers_to_the_least_loaded_by_default() {
        let model_id = "test-model-overload-steer";
        let worker_registry = Arc::new(WorkerRegistry::new());
        let mut workers = Vec::new();
        for i in 0..2 {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{}", 8450 + i))
                    .model(ModelCard::new(model_id))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .health_config(no_health_check())
                    .build(),
            );
            worker_registry.register(Arc::clone(&worker)).unwrap();
            workers.push(worker);
        }
        workers[0].increment_load();
        workers[0].increment_load();
        workers[1].increment_load();
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::Regular,
            None,
        );
        for worker in &workers {
            worker_registry.set_worker_overloaded(worker, true);
        }
        let selected = stage
            .select_single_worker(model_id, None, None, None, None, None, None, None)
            .expect("the all-overloaded pool is steered, not refused");
        assert_eq!(
            selected.url(),
            "grpc://127.0.0.1:8451",
            "the least-loaded serves"
        );

        // PD: both legs vetoed still pair up.
        let pd_model = "test-model-overload-steer-pd";
        let pd_registry = Arc::new(WorkerRegistry::new());
        let (prefill_urls, decode_urls) = register_pd_workers(&pd_registry, pd_model, 2);
        let pd_stage = WorkerSelectionStage::new(
            Arc::clone(&pd_registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::PrefillDecode,
            None,
        );
        for url in prefill_urls.iter().chain(decode_urls.iter()) {
            let worker = pd_registry.get_by_url(url).expect("registered");
            pd_registry.set_worker_overloaded(&worker, true);
        }
        let (prefill, decode, _) = pd_stage
            .select_pd_pair(
                pd_model,
                PlacementInputs::default(),
                None,
                None,
                VisionDemand::default(),
            )
            .expect("both legs over the thresholds still pair");
        assert!(prefill_urls.contains(&prefill.url().to_string()));
        assert!(decode_urls.contains(&decode.url().to_string()));
    }

    /// Capable but overloaded workers keep the overload shed: its code,
    /// Retry-After and non-retryable marking survive worker mode.
    #[test]
    fn media_refs_overloaded_capable_workers_keep_the_overload_shed() {
        let model_id = "test-model-media-refs-overload";
        let worker_registry = Arc::new(WorkerRegistry::new());
        worker_registry.set_overload_shed(true);
        let mut workers = Vec::new();
        for i in 0..2 {
            let worker = vllm_grpc_worker(
                &format!("grpc://127.0.0.1:{}", 8740 + i),
                model_id,
                WorkerType::Regular,
                true,
            );
            worker_registry.register(Arc::clone(&worker)).unwrap();
            workers.push(worker);
        }
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::clone(&policy_registry),
            WorkerSelectionMode::Regular,
            None,
        );
        for worker in &workers {
            worker_registry.set_worker_overloaded(worker, true);
        }
        assert!(stage
            .select_single_worker(
                model_id,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(ACCEPTS_MEDIA_REFS),
            )
            .is_none());
        let response = stage.selection_failure(model_id, &[WorkerType::Regular], None, REFS);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_ne!(
            error::extract_error_code_from_response(&response),
            "no_media_ref_capable_worker"
        );
        assert!(response.headers().contains_key("retry-after"));
        assert!(!is_retryable_response(&response));
    }

    /// Both PD legs process the references, so both must advertise.
    /// A leg with no members is "no worker serves the model", not a capability
    /// shed: the PD path judges emptiness before capability, like the single path.
    #[test]
    fn media_refs_pd_selection_empty_leg_keeps_its_own_verdict() {
        let model_id = "test-model-media-refs-pd-empty-leg";
        let worker_registry = Arc::new(WorkerRegistry::new());
        worker_registry
            .register(vllm_grpc_worker(
                "grpc://127.0.0.1:8740",
                model_id,
                WorkerType::Decode,
                true,
            ))
            .unwrap();
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        policy_registry
            .set_prefill_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        policy_registry
            .set_decode_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::clone(&policy_registry),
            WorkerSelectionMode::PrefillDecode,
            None,
        );

        let response = stage
            .select_pd_pair(
                model_id,
                PlacementInputs {
                    candidate_filter: Some(ACCEPTS_MEDIA_REFS),
                    ..Default::default()
                },
                None,
                None,
                VisionDemand::default(),
            )
            .map_err(|failure| stage.pd_selection_failure(model_id, failure, true))
            .expect_err("no prefill worker at all");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_ne!(
            error::extract_error_code_from_response(&response),
            "no_media_ref_capable_worker"
        );
    }

    /// With the demand an image reference builds, a pool with no pair that
    /// takes the references keeps the placement's media-references verdict
    /// (its message names the fix that applies), not the vision refusal:
    /// every worker here may have the encoder.
    #[test]
    fn media_refs_with_no_pair_that_takes_them_keep_the_media_refs_verdict() {
        let model_id = "test-model-media-refs-pd-vision";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, refs) in [
            ("grpc://127.0.0.1:8740", WorkerType::Prefill, false),
            ("grpc://127.0.0.1:8741", WorkerType::Prefill, true),
            ("grpc://127.0.0.1:8750", WorkerType::Decode, false),
        ] {
            worker_registry
                .register(vllm_grpc_worker(url, model_id, worker_type, refs))
                .unwrap();
        }
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        policy_registry
            .set_prefill_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        policy_registry
            .set_decode_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::clone(&policy_registry),
            WorkerSelectionMode::PrefillDecode,
            None,
        );

        let response = stage
            .select_pd_pair(
                model_id,
                PlacementInputs {
                    candidate_filter: Some(ACCEPTS_MEDIA_REFS),
                    ..Default::default()
                },
                None,
                None,
                VisionDemand::new(true, true).with_media_refs(true),
            )
            .map_err(|failure| stage.pd_selection_failure(model_id, failure, true))
            .expect_err("no pair takes the references");
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "no_media_ref_capable_worker",
            "the media-references verdict, not the vision refusal"
        );
    }

    #[test]
    fn media_refs_pd_selection_requires_both_legs() {
        let model_id = "test-model-media-refs-pd";
        let worker_registry = Arc::new(WorkerRegistry::new());
        worker_registry
            .register(vllm_grpc_worker(
                "grpc://127.0.0.1:8720",
                model_id,
                WorkerType::Prefill,
                false,
            ))
            .unwrap();
        worker_registry
            .register(vllm_grpc_worker(
                "grpc://127.0.0.1:8721",
                model_id,
                WorkerType::Prefill,
                true,
            ))
            .unwrap();
        worker_registry
            .register(vllm_grpc_worker(
                "grpc://127.0.0.1:8730",
                model_id,
                WorkerType::Decode,
                false,
            ))
            .unwrap();
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        policy_registry
            .set_prefill_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        policy_registry
            .set_decode_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        let stage = WorkerSelectionStage::new(
            Arc::clone(&worker_registry),
            Arc::clone(&policy_registry),
            WorkerSelectionMode::PrefillDecode,
            None,
        );

        let response = stage
            .select_pd_pair(
                model_id,
                PlacementInputs {
                    candidate_filter: Some(ACCEPTS_MEDIA_REFS),
                    ..Default::default()
                },
                None,
                None,
                VisionDemand::default(),
            )
            .map_err(|failure| stage.pd_selection_failure(model_id, failure, true))
            .expect_err("no advertising decode worker yet");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "no_media_ref_capable_worker"
        );

        worker_registry
            .register(vllm_grpc_worker(
                "grpc://127.0.0.1:8731",
                model_id,
                WorkerType::Decode,
                true,
            ))
            .unwrap();
        for _ in 0..3 {
            let (prefill, decode, _) = stage
                .select_pd_pair(
                    model_id,
                    PlacementInputs {
                        candidate_filter: Some(ACCEPTS_MEDIA_REFS),
                        ..Default::default()
                    },
                    None,
                    None,
                    VisionDemand::default(),
                )
                .map_err(|failure| stage.pd_selection_failure(model_id, failure, true))
                .expect("advertising pair");
            assert_eq!(prefill.url(), "grpc://127.0.0.1:8721");
            assert_eq!(decode.url(), "grpc://127.0.0.1:8731");
        }
    }

    /// The prefill legs `stage` picks for `model_id` over 12 selections,
    /// sorted, under the given demand.
    fn prefill_legs(
        stage: &WorkerSelectionStage,
        model_id: &str,
        demand: VisionDemand,
    ) -> Vec<String> {
        let mut seen = HashMap::new();
        for _ in 0..12 {
            let (prefill, _, _) = stage
                .select_pd_pair(model_id, PlacementInputs::default(), None, None, demand)
                .expect("a pair");
            *seen.entry(prefill.url().to_string()).or_insert(0) += 1;
        }
        let mut urls: Vec<String> = seen.into_keys().collect();
        urls.sort();
        urls
    }

    /// The prefill leg needs the encoder for any image or video, processed
    /// or referenced, and whenever the decode leg does; the decode leg's
    /// demand is `decode_needs_vision`'s alone.
    #[test]
    fn vision_demand_follows_the_request_s_media() {
        use llm_multimodal::MediaContentPart;

        assert!(!multimodal::request_needs_vision(None, None), "text");
        assert!(
            multimodal::request_needs_vision(Some(&image_intermediate(None)), None),
            "a processed image"
        );
        let image_refs = multimodal::MediaPlan::new([MediaContentPart::ImageUrl {
            url: "data:image/png;base64,AAAA".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }]);
        assert!(
            multimodal::request_needs_vision(None, Some(&image_refs)),
            "a referenced image"
        );
        assert!(
            !multimodal::request_needs_vision(None, Some(&multimodal::MediaPlan::default())),
            "an empty plan"
        );

        assert_eq!(VisionDemand::new(false, false), VisionDemand::default());
        assert_eq!(
            VisionDemand::new(true, false),
            VisionDemand {
                prefill: true,
                decode: false,
                refs: false
            },
            "a single-sample image: the prefill leg takes the pixels, the decode leg is stripped"
        );
        assert_eq!(
            VisionDemand::new(false, true),
            VisionDemand {
                prefill: true,
                decode: true,
                refs: false
            },
            "what the decode leg cannot run without the encoder, the prefill leg cannot either"
        );
        let stage = |mode| {
            WorkerSelectionStage::new(
                Arc::new(WorkerRegistry::new()),
                Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
                mode,
                None,
            )
        };
        assert!(stage(WorkerSelectionMode::PrefillDecode).prefill_leg_needs_vision(true));
        assert!(
            !stage(WorkerSelectionMode::EncodePrefillDecode).prefill_leg_needs_vision(true),
            "the encode workers take the pixels"
        );
        assert!(!stage(WorkerSelectionMode::Regular).prefill_leg_needs_vision(true));
        assert!(!stage(WorkerSelectionMode::PrefillDecode).prefill_leg_needs_vision(false));

        let prefill = labeled_worker("grpc://127.0.0.1:9000", "m", WorkerType::Prefill, UNLABELED);
        let decode = labeled_worker("grpc://127.0.0.1:9001", "m", WorkerType::Decode, UNLABELED);
        let image = VisionDemand::new(true, false);
        assert!(image.on(prefill.as_ref()));
        assert!(!image.on(decode.as_ref()));
        assert!(prefill_vision_capable(decode.as_ref()));
        assert!(prefill_vision_capable(prefill.as_ref()));
        assert!(!prefill_vision_capable(
            labeled_worker(
                "grpc://127.0.0.1:9002",
                "m",
                WorkerType::Prefill,
                NO_VISION_ENCODER
            )
            .as_ref()
        ));
    }

    /// An image request's prefill leg pairs with a prefill worker that has
    /// the encoder (or gave no verdict) while the model has one, instead of
    /// being admitted by a worker without it and never scheduled; text
    /// requests still use every prefill worker; a model whose prefill
    /// workers all lack the encoder refuses the request instead of pairing
    /// it; and a leg nobody serves keeps its own verdict.
    #[test]
    fn an_image_request_s_prefill_leg_pairs_with_a_prefill_worker_with_a_vision_encoder() {
        let model_id = "test-model-vision-prefill-leg";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, labels) in [
            ("grpc://127.0.0.1:9010", WorkerType::Prefill, VISION_ENCODER),
            (
                "grpc://127.0.0.1:9011",
                WorkerType::Prefill,
                NO_VISION_ENCODER,
            ),
            ("grpc://127.0.0.1:9012", WorkerType::Prefill, UNLABELED),
            ("grpc://127.0.0.1:9020", WorkerType::Decode, VISION_ENCODER),
        ] {
            worker_registry
                .register(labeled_worker(url, model_id, worker_type, labels))
                .unwrap();
        }
        let stage = round_robin_pd_stage(Arc::clone(&worker_registry));
        let image = VisionDemand::new(true, false);

        assert_eq!(
            prefill_legs(&stage, model_id, image),
            ["grpc://127.0.0.1:9010", "grpc://127.0.0.1:9012"],
            "the leg goes to a prefill worker with the encoder or without a verdict"
        );
        assert_eq!(
            prefill_legs(&stage, model_id, VisionDemand::default()),
            [
                "grpc://127.0.0.1:9010",
                "grpc://127.0.0.1:9011",
                "grpc://127.0.0.1:9012"
            ],
            "a text request uses every prefill worker"
        );

        // No prefill worker with the encoder: the request is refused, as no
        // dispatch check would catch the leg; text still pairs.
        let uniform = "test-model-vision-prefill-leg-none";
        for (url, worker_type, labels) in [
            (
                "grpc://127.0.0.1:9030",
                WorkerType::Prefill,
                NO_VISION_ENCODER,
            ),
            ("grpc://127.0.0.1:9040", WorkerType::Decode, VISION_ENCODER),
        ] {
            worker_registry
                .register(labeled_worker(url, uniform, worker_type, labels))
                .unwrap();
        }
        let failure = stage
            .select_pd_pair(uniform, PlacementInputs::default(), None, None, image)
            .expect_err("no prefill worker with the encoder");
        assert!(
            matches!(failure, PdSelectionFailure::NoVisionCapablePrefill),
            "{failure:?}"
        );
        let response = stage.pd_selection_failure(uniform, failure, false);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "multimodal_not_supported"
        );
        assert_eq!(
            prefill_legs(&stage, uniform, VisionDemand::default()),
            ["grpc://127.0.0.1:9030"]
        );

        // A leg nobody serves is named before the encoder is asked for.
        let halved = "test-model-vision-prefill-leg-no-decode";
        worker_registry
            .register(labeled_worker(
                "grpc://127.0.0.1:9050",
                halved,
                WorkerType::Prefill,
                NO_VISION_ENCODER,
            ))
            .unwrap();
        let failure = stage
            .select_pd_pair(halved, PlacementInputs::default(), None, None, image)
            .expect_err("no decode worker");
        assert!(
            matches!(
                &failure,
                PdSelectionFailure::Placement(failure)
                    if matches!(failure.verdict, PlacementFailure::NoCandidates)
            ),
            "{failure:?}"
        );
    }

    /// Media references forwarded to the workers put the pixels on the
    /// prefill leg as well: among the workers that process references, the
    /// leg pairs with one that has the encoder.
    #[test]
    fn media_references_pair_with_a_vision_capable_prefill_worker() {
        const REFS: [(&str, &str); 2] = [
            ("mm_processor", "inprocess"),
            ("mm_media_ref_schemes", "http,https,data"),
        ];
        let refs_with = |verdict: &'static str| -> Vec<(&'static str, &'static str)> {
            let mut labels = REFS.to_vec();
            labels.push((multimodal::SUPPORTS_VISION_LABEL, verdict));
            labels
        };
        let model_id = "test-model-vision-prefill-refs";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, labels) in [
            (
                "grpc://127.0.0.1:9060",
                WorkerType::Prefill,
                refs_with("true"),
            ),
            (
                "grpc://127.0.0.1:9061",
                WorkerType::Prefill,
                refs_with("false"),
            ),
            (
                "grpc://127.0.0.1:9062",
                WorkerType::Prefill,
                VISION_ENCODER.to_vec(),
            ),
            (
                "grpc://127.0.0.1:9070",
                WorkerType::Decode,
                refs_with("true"),
            ),
        ] {
            worker_registry
                .register(labeled_worker(url, model_id, worker_type, &labels))
                .unwrap();
        }
        let stage = round_robin_pd_stage(Arc::clone(&worker_registry));

        for _ in 0..6 {
            let (prefill, _, _) = stage
                .select_pd_pair(
                    model_id,
                    PlacementInputs {
                        candidate_filter: Some(ACCEPTS_MEDIA_REFS),
                        ..Default::default()
                    },
                    None,
                    None,
                    VisionDemand::new(true, true),
                )
                .expect("a pair that processes references with the encoder");
            assert_eq!(prefill.url(), "grpc://127.0.0.1:9060");
        }
    }

    /// With media references the prefill leg must find a worker that takes
    /// them and has the encoder: a pool whose referencing prefill worker
    /// lacks the encoder and whose encoder-bearing one takes no references
    /// is the terminal refusal (not a retryable 503 over healthy workers);
    /// without references the same pool pairs with the encoder.
    #[test]
    fn media_references_with_no_vision_capable_prefill_worker_that_takes_them_are_refused() {
        const REFS: [(&str, &str); 2] = [
            ("mm_processor", "inprocess"),
            ("mm_media_ref_schemes", "http,https,data"),
        ];
        let refs_with = |verdict: &'static str| -> Vec<(&'static str, &'static str)> {
            let mut labels = REFS.to_vec();
            labels.push((multimodal::SUPPORTS_VISION_LABEL, verdict));
            labels
        };
        let model_id = "test-model-vision-prefill-refs-refusal";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, labels) in [
            (
                "grpc://127.0.0.1:9090",
                WorkerType::Prefill,
                refs_with("false"),
            ),
            (
                "grpc://127.0.0.1:9091",
                WorkerType::Prefill,
                VISION_ENCODER.to_vec(),
            ),
            (
                "grpc://127.0.0.1:9095",
                WorkerType::Decode,
                refs_with("true"),
            ),
        ] {
            worker_registry
                .register(labeled_worker(url, model_id, worker_type, &labels))
                .unwrap();
        }
        let stage = round_robin_pd_stage(Arc::clone(&worker_registry));

        let failure = stage
            .select_pd_pair(
                model_id,
                PlacementInputs {
                    candidate_filter: Some(ACCEPTS_MEDIA_REFS),
                    ..Default::default()
                },
                None,
                None,
                VisionDemand::new(true, true).with_media_refs(true),
            )
            .expect_err("no prefill worker that takes references has the encoder");
        assert!(
            matches!(failure, PdSelectionFailure::NoVisionCapablePrefill),
            "the terminal refusal, not the unfiltered pool's verdict"
        );

        let (prefill, _, _) = stage
            .select_pd_pair(
                model_id,
                PlacementInputs::default(),
                None,
                None,
                VisionDemand::new(true, true),
            )
            .expect("without references the prefill leg pairs with the encoder");
        assert_eq!(prefill.url(), "grpc://127.0.0.1:9091");
    }

    /// A retry re-selects under the demand the first selection recorded:
    /// the prefill leg of an image request stays on a worker with the
    /// encoder.
    #[tokio::test]
    async fn reselect_keeps_an_image_request_s_prefill_leg_on_a_vision_capable_worker() {
        let model_id = "test-model-vision-prefill-reselect";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, labels) in [
            ("grpc://127.0.0.1:9080", WorkerType::Prefill, VISION_ENCODER),
            (
                "grpc://127.0.0.1:9081",
                WorkerType::Prefill,
                NO_VISION_ENCODER,
            ),
            ("grpc://127.0.0.1:9090", WorkerType::Decode, VISION_ENCODER),
        ] {
            worker_registry
                .register(labeled_worker(url, model_id, worker_type, labels))
                .unwrap();
        }
        let stage = round_robin_pd_stage(Arc::clone(&worker_registry));
        let wire = WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Grpc,
            requires_media_refs: false,
        };

        let stage = &stage;
        let prefill_legs = |needs_vision: bool| async move {
            let mut seen = HashMap::new();
            for _ in 0..8 {
                let mut ctx = dispatch_ctx(model_id, wire);
                ctx.routing.needs_vision = needs_vision;
                stage
                    .reselect(&mut ctx)
                    .await
                    .expect("re-selection succeeds");
                match ctx.workers.expect("a pair is selected") {
                    WorkerSelection::Disaggregated { prefill, .. } => {
                        *seen.entry(prefill.url().to_string()).or_insert(0) += 1;
                    }
                    WorkerSelection::Single { .. } => panic!("expected a PD selection"),
                }
            }
            let mut urls: Vec<String> = seen.into_keys().collect();
            urls.sort();
            urls
        };
        assert_eq!(prefill_legs(true).await, ["grpc://127.0.0.1:9080"]);
        assert_eq!(
            prefill_legs(false).await,
            ["grpc://127.0.0.1:9080", "grpc://127.0.0.1:9081"]
        );
    }

    /// The stage reads the demand off the request: a single-sample image
    /// without grid tensors (no decode demand) still selects a prefill
    /// worker with the encoder, and the routing snapshot records it for
    /// retries; a text request uses every prefill worker.
    #[tokio::test]
    async fn execute_keeps_an_image_request_s_prefill_leg_on_a_vision_capable_worker() {
        let registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, labels) in [
            ("grpc://127.0.0.1:9100", WorkerType::Prefill, VISION_ENCODER),
            (
                "grpc://127.0.0.1:9101",
                WorkerType::Prefill,
                NO_VISION_ENCODER,
            ),
            ("grpc://127.0.0.1:9110", WorkerType::Decode, VISION_ENCODER),
        ] {
            registry
                .register(labeled_worker(url, "test-model", worker_type, labels))
                .unwrap();
        }
        let stage = round_robin_pd_stage(Arc::clone(&registry));

        let mut seen = HashMap::new();
        for _ in 0..4 {
            let mut ctx = prepared_ctx(Arc::clone(&registry), None);
            ctx.state.multimodal_intermediate = Some(image_intermediate(None));
            stage
                .execute(&mut ctx)
                .await
                .expect("a single-sample image request selects");
            match ctx.state.workers.as_ref().expect("a pair is selected") {
                WorkerSelection::Disaggregated { prefill, .. } => {
                    *seen.entry(prefill.url().to_string()).or_insert(0) += 1;
                }
                WorkerSelection::Single { .. } => panic!("expected a PD selection"),
            }
            let snapshot = ctx.state.routing_snapshot.as_ref().expect("snapshot");
            assert!(snapshot.needs_vision);
            assert!(!snapshot.decode_needs_vision);
        }
        assert_eq!(
            seen.into_keys().collect::<Vec<_>>(),
            ["grpc://127.0.0.1:9100"]
        );

        let mut seen = HashMap::new();
        for _ in 0..4 {
            let mut ctx = prepared_ctx(Arc::clone(&registry), None);
            stage
                .execute(&mut ctx)
                .await
                .expect("a text request selects");
            if let Some(WorkerSelection::Disaggregated { prefill, .. }) = ctx.state.workers.as_ref()
            {
                *seen.entry(prefill.url().to_string()).or_insert(0) += 1;
            }
            let snapshot = ctx.state.routing_snapshot.as_ref().expect("snapshot");
            assert!(!snapshot.needs_vision);
        }
        assert_eq!(
            seen.len(),
            2,
            "a text request uses the whole prefill pool: {seen:?}"
        );
    }

    /// A model whose prefill workers all run as a language model only
    /// refuses an image request at selection, where the worker would have
    /// admitted it and never scheduled it; a text request still selects.
    #[tokio::test]
    async fn an_image_request_is_refused_when_no_prefill_worker_has_a_vision_encoder() {
        let registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, labels) in [
            (
                "grpc://127.0.0.1:9120",
                WorkerType::Prefill,
                NO_VISION_ENCODER,
            ),
            ("grpc://127.0.0.1:9130", WorkerType::Decode, VISION_ENCODER),
        ] {
            registry
                .register(labeled_worker(url, "test-model", worker_type, labels))
                .unwrap();
        }
        let stage = round_robin_pd_stage(Arc::clone(&registry));

        let mut ctx = prepared_ctx(Arc::clone(&registry), None);
        ctx.state.multimodal_intermediate = Some(image_intermediate(None));
        let response = stage
            .execute(&mut ctx)
            .await
            .expect_err("no prefill worker with the encoder");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "multimodal_not_supported"
        );

        let mut ctx = prepared_ctx(Arc::clone(&registry), None);
        stage
            .execute(&mut ctx)
            .await
            .expect("a text request selects");
        match ctx.state.workers.as_ref().expect("a pair is selected") {
            WorkerSelection::Disaggregated { prefill, .. } => {
                assert_eq!(prefill.url(), "grpc://127.0.0.1:9120");
            }
            WorkerSelection::Single { .. } => panic!("expected a PD selection"),
        }
    }

    /// A decode worker with the encoder over the thresholds next to an idle
    /// one without it: a request whose decode leg needs the encoder is
    /// steered to the vetoed worker by the overload fallback (it is the
    /// leg's only admitted worker) and, under `--worker-overload-shed`, gets
    /// the overload shed, not `no_available_workers`; a text request takes
    /// the idle worker.
    #[test]
    fn an_overloaded_only_vision_capable_decode_worker_is_steered_to_not_refused() {
        let model_id = "test-model-vision-decode-overloaded";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, labels) in [
            ("grpc://127.0.0.1:9200", WorkerType::Prefill, VISION_ENCODER),
            ("grpc://127.0.0.1:9210", WorkerType::Decode, VISION_ENCODER),
            (
                "grpc://127.0.0.1:9211",
                WorkerType::Decode,
                NO_VISION_ENCODER,
            ),
        ] {
            worker_registry
                .register(labeled_worker(url, model_id, worker_type, labels))
                .unwrap();
        }
        let stage = round_robin_pd_stage(Arc::clone(&worker_registry));
        let d1 = worker_registry
            .get_by_url("grpc://127.0.0.1:9210")
            .expect("registered");
        worker_registry.set_worker_overloaded(&d1, true);
        let demand = VisionDemand::new(false, true);

        let (_, decode, _) = stage
            .select_pd_pair(model_id, PlacementInputs::default(), None, None, demand)
            .expect("steered to the only admitted decode worker");
        assert_eq!(decode.url(), "grpc://127.0.0.1:9210");
        let (_, decode, _) = stage
            .select_pd_pair(
                model_id,
                PlacementInputs::default(),
                None,
                None,
                VisionDemand::default(),
            )
            .expect("a text request pairs with the idle decode worker");
        assert_eq!(decode.url(), "grpc://127.0.0.1:9211");

        worker_registry.set_overload_shed(true);
        let failure = stage
            .select_pd_pair(model_id, PlacementInputs::default(), None, None, demand)
            .expect_err("the admitted decode leg is all-overloaded under shedding");
        let response = stage.pd_selection_failure(model_id, failure, false);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "worker_overload_protection_shed"
        );
    }

    // ---- the pre-selection queue timeout ---------------------------------------------------

    fn regular_stage(registry: Arc<WorkerRegistry>) -> WorkerSelectionStage {
        WorkerSelectionStage::new(
            registry,
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            WorkerSelectionMode::Regular,
            None,
        )
    }

    /// A generate request's context, tokenized, accepted by the gateway at
    /// `accepted` (or carrying no stamp at all).
    fn prepared_ctx(registry: Arc<WorkerRegistry>, accepted: Option<Instant>) -> RequestContext {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "text": "Hello"
        }))
        .unwrap();
        let components = Arc::new(SharedComponents {
            tokenizer_registry: Arc::new(TokenizerRegistry::new()),
            worker_registry: registry,
            tool_parser_factory: Default::default(),
            reasoning_parser_factory: Default::default(),
            parser_resolver: utils::ParserResolver::disabled(),
            multimodal: None,
            tokenizer_autoload: true,
            tokenizer_load_failures: Arc::default(),
        });
        let mut ctx = RequestContext::for_generate(
            Arc::new(request),
            None,
            "test-model".to_string(),
            components,
        );
        ctx.state.preparation = Some(PreparationOutput::Generate {
            original_text: None,
            token_ids: vec![1, 2, 3],
        });
        let meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));
        ctx.input.tenant_request_meta = Some(match accepted {
            Some(accepted) => meta.with_extension(AcceptedAt::at(accepted)),
            None => meta,
        });
        ctx
    }

    /// The vision verdict of a worker without the encoder, of one with it,
    /// and of one whose servicer gave none.
    const NO_VISION_ENCODER: &[(&str, &str)] = &[(multimodal::SUPPORTS_VISION_LABEL, "false")];
    const VISION_ENCODER: &[(&str, &str)] = &[(multimodal::SUPPORTS_VISION_LABEL, "true")];
    const UNLABELED: &[(&str, &str)] = &[];

    /// A gRPC worker of `worker_type`, with `labels`.
    fn labeled_worker(
        url: &str,
        model_id: &str,
        worker_type: WorkerType,
        labels: &[(&str, &str)],
    ) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model_id))
            .worker_type(worker_type)
            .runtime_type(RuntimeType::Vllm)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check());
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        Arc::new(builder.build())
    }

    fn round_robin_pd_stage(worker_registry: Arc<WorkerRegistry>) -> WorkerSelectionStage {
        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        policy_registry
            .set_prefill_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        policy_registry
            .set_decode_policy(PolicyFactory::create_from_config(&PolicyConfig::RoundRobin));
        WorkerSelectionStage::new(
            worker_registry,
            policy_registry,
            WorkerSelectionMode::PrefillDecode,
            None,
        )
    }

    /// A router-processed single-image intermediate, carrying under
    /// `grid_key` the grid tensor a decode leg derives its positions from.
    fn image_intermediate(grid_key: Option<&str>) -> multimodal::MultimodalIntermediate {
        use llm_multimodal::{
            EncoderFieldLayouts, ImageDetail, ImageFrame, ImageSource, ModelSpecificValue,
            PlaceholderRange, PreprocessedEncoderInputs,
        };
        use ndarray::{ArrayD, IxDyn};

        use crate::routers::grpc::multimodal::{
            MediaBatch, MultimodalIntermediate, PrecomputedMultimodalIntermediate, PromptBinding,
        };

        let mut model_specific = HashMap::new();
        if let Some(key) = grid_key {
            model_specific.insert(
                key.to_string(),
                ModelSpecificValue::UintTensor {
                    data: vec![1, 2, 2],
                    shape: vec![1, 3],
                },
            );
        }
        MultimodalIntermediate::try_new(vec![PrecomputedMultimodalIntermediate {
            preprocessed: PreprocessedEncoderInputs {
                encoder_input: ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0])
                    .unwrap()
                    .into(),
                feature_token_counts: vec![1],
                item_sizes: vec![(1, 1)],
                model_specific,
            },
            media: MediaBatch::Images(vec![Arc::new(ImageFrame::new(
                image::DynamicImage::new_rgb8(1, 1),
                bytes::Bytes::from_static(b"image"),
                ImageDetail::Auto,
                ImageSource::InlineBytes,
                "image-hash".to_string(),
            ))]),
            bindings: vec![PromptBinding {
                item_index: 0,
                prompt_ordinal: 0,
                structural: PlaceholderRange {
                    offset: 0,
                    length: 1,
                },
                patches: vec![],
            }],
            placeholder_token_id: Some(10),
            field_layouts: EncoderFieldLayouts::default(),
            keep_on_cpu_keys: vec![],
            encoder_input_key: None,
        }])
        .unwrap()
    }

    /// The decode legs `stage` picks for `model_id` over 12 selections,
    /// sorted, under the given demand.
    fn decode_legs(
        stage: &WorkerSelectionStage,
        model_id: &str,
        needs_vision: bool,
    ) -> Vec<String> {
        let mut seen = HashMap::new();
        for _ in 0..12 {
            let (_, decode, _) = stage
                .select_pd_pair(
                    model_id,
                    PlacementInputs::default(),
                    None,
                    None,
                    VisionDemand::new(false, needs_vision),
                )
                .expect("a pair");
            *seen.entry(decode.url().to_string()).or_insert(0) += 1;
        }
        let mut urls: Vec<String> = seen.into_keys().collect();
        urls.sort();
        urls
    }

    /// The decode-leg demand is for what a decode worker without a vision
    /// encoder cannot serve: media references, a multimodal request with
    /// `n > 1`, grid tensors. A text request of any `n` and a single-sample
    /// image request without grids take such a worker's stripped leg.
    #[test]
    fn decode_needs_vision_only_for_what_a_stripped_decode_leg_cannot_serve() {
        use openai_protocol::chat::ChatCompletionRequest;

        let chat = |n: Option<u32>| -> RequestType {
            let mut body = serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}]
            });
            if let Some(n) = n {
                body["n"] = serde_json::json!(n);
            }
            let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
            RequestType::Chat(Arc::new(request))
        };
        let plain = image_intermediate(None);
        let gridded = image_intermediate(Some("image_grid_thw"));

        assert!(!decode_needs_vision(&chat(None), None, false), "text");
        assert!(
            !decode_needs_vision(&chat(Some(2)), None, false),
            "text with n > 1 recomputes a text prompt: no encoder needed"
        );
        assert!(
            decode_needs_vision(&chat(None), None, true),
            "media references are expanded by each leg"
        );
        assert!(
            !decode_needs_vision(&chat(Some(1)), Some(&plain), false),
            "a single-sample image request takes the stripped leg"
        );
        assert!(
            decode_needs_vision(&chat(Some(2)), Some(&plain), false),
            "a multimodal request with n > 1 recomputes the prompt on the decode leg"
        );
        assert!(
            decode_needs_vision(&chat(None), Some(&gridded), false),
            "grid tensors are read by the decode leg"
        );
    }

    /// The demand is recorded in prefill/decode mode only: the decode leg of
    /// encode/prefill/decode mode is dispatched without the pixels (its
    /// encode workers take them), so neither its first selection nor a retry
    /// is held to a decode worker with the encoder; regular mode has no
    /// decode leg.
    #[test]
    fn the_decode_leg_demand_is_recorded_in_prefill_decode_mode_only() {
        use openai_protocol::chat::ChatCompletionRequest;

        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 2
        }))
        .unwrap();
        let request = RequestType::Chat(Arc::new(request));
        let gridded = image_intermediate(Some("image_grid_thw"));
        let stage = |mode| {
            WorkerSelectionStage::new(
                Arc::new(WorkerRegistry::new()),
                Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
                mode,
                None,
            )
        };

        let pd = stage(WorkerSelectionMode::PrefillDecode);
        assert!(pd.decode_leg_needs_vision(&request, Some(&gridded), false));
        assert!(pd.decode_leg_needs_vision(&request, None, true));

        let epd = stage(WorkerSelectionMode::EncodePrefillDecode);
        assert!(
            !epd.decode_leg_needs_vision(&request, Some(&gridded), false),
            "the encode/prefill/decode decode leg takes no pixels"
        );
        assert!(!epd.decode_leg_needs_vision(&request, None, true));

        let regular = stage(WorkerSelectionMode::Regular);
        assert!(!regular.decode_leg_needs_vision(&request, Some(&gridded), false));
    }

    /// A request whose decode leg needs a vision encoder pairs with a decode
    /// worker that has one (or gave no verdict) while the model has such a
    /// worker, instead of landing on one without it and being refused after
    /// selection; text and single-sample image requests still use every
    /// decode worker; and a model whose decode workers all lack the encoder
    /// is paired as it is, so dispatch names the gap.
    #[test]
    fn a_vision_demanding_decode_leg_pairs_with_a_vision_capable_decode_worker() {
        let model_id = "test-model-vision-decode-leg";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, labels) in [
            ("grpc://127.0.0.1:8900", WorkerType::Prefill, VISION_ENCODER),
            ("grpc://127.0.0.1:8910", WorkerType::Decode, VISION_ENCODER),
            (
                "grpc://127.0.0.1:8911",
                WorkerType::Decode,
                NO_VISION_ENCODER,
            ),
            ("grpc://127.0.0.1:8912", WorkerType::Decode, UNLABELED),
        ] {
            worker_registry
                .register(labeled_worker(url, model_id, worker_type, labels))
                .unwrap();
        }
        let stage = round_robin_pd_stage(Arc::clone(&worker_registry));

        assert_eq!(
            decode_legs(&stage, model_id, true),
            ["grpc://127.0.0.1:8910", "grpc://127.0.0.1:8912"],
            "the leg goes to a decode worker with the encoder or without a verdict"
        );
        assert_eq!(
            decode_legs(&stage, model_id, false),
            [
                "grpc://127.0.0.1:8910",
                "grpc://127.0.0.1:8911",
                "grpc://127.0.0.1:8912"
            ],
            "a text or single-sample image request uses every decode worker"
        );

        // No decode worker with the encoder: the pool is used as it is.
        let uniform = "test-model-vision-decode-leg-none";
        for (url, worker_type, labels) in [
            ("grpc://127.0.0.1:8920", WorkerType::Prefill, VISION_ENCODER),
            (
                "grpc://127.0.0.1:8930",
                WorkerType::Decode,
                NO_VISION_ENCODER,
            ),
        ] {
            worker_registry
                .register(labeled_worker(url, uniform, worker_type, labels))
                .unwrap();
        }
        assert_eq!(
            decode_legs(&stage, uniform, true),
            ["grpc://127.0.0.1:8930"],
            "dispatch, not selection, refuses a pool with no such worker"
        );
    }

    /// Media references forwarded to the workers need the encoder on the
    /// decode leg as well: among the workers that process references, the
    /// leg pairs with one that has it.
    #[test]
    fn media_references_pair_with_a_vision_capable_decode_worker() {
        const REFS: [(&str, &str); 2] = [
            ("mm_processor", "inprocess"),
            ("mm_media_ref_schemes", "http,https,data"),
        ];
        let refs_with = |verdict: &'static str| -> Vec<(&'static str, &'static str)> {
            let mut labels = REFS.to_vec();
            labels.push((multimodal::SUPPORTS_VISION_LABEL, verdict));
            labels
        };
        let model_id = "test-model-vision-decode-refs";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, labels) in [
            (
                "grpc://127.0.0.1:8940",
                WorkerType::Prefill,
                refs_with("true"),
            ),
            (
                "grpc://127.0.0.1:8950",
                WorkerType::Decode,
                refs_with("true"),
            ),
            (
                "grpc://127.0.0.1:8951",
                WorkerType::Decode,
                refs_with("false"),
            ),
            (
                "grpc://127.0.0.1:8952",
                WorkerType::Decode,
                VISION_ENCODER.to_vec(),
            ),
        ] {
            worker_registry
                .register(labeled_worker(url, model_id, worker_type, &labels))
                .unwrap();
        }
        let stage = round_robin_pd_stage(Arc::clone(&worker_registry));

        for _ in 0..6 {
            let (_, decode, _) = stage
                .select_pd_pair(
                    model_id,
                    PlacementInputs {
                        candidate_filter: Some(ACCEPTS_MEDIA_REFS),
                        ..Default::default()
                    },
                    None,
                    None,
                    VisionDemand::new(false, true),
                )
                .expect("a pair that processes references with the encoder");
            assert_eq!(decode.url(), "grpc://127.0.0.1:8950");
        }
    }

    /// A retry re-selects under the demand the first selection recorded:
    /// the decode leg stays on a worker with the encoder.
    #[tokio::test]
    async fn reselect_keeps_a_vision_demanding_decode_leg_on_a_vision_capable_worker() {
        let model_id = "test-model-vision-decode-reselect";
        let worker_registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, labels) in [
            ("grpc://127.0.0.1:8960", WorkerType::Prefill, VISION_ENCODER),
            ("grpc://127.0.0.1:8970", WorkerType::Decode, VISION_ENCODER),
            (
                "grpc://127.0.0.1:8971",
                WorkerType::Decode,
                NO_VISION_ENCODER,
            ),
        ] {
            worker_registry
                .register(labeled_worker(url, model_id, worker_type, labels))
                .unwrap();
        }
        let stage = round_robin_pd_stage(Arc::clone(&worker_registry));
        let wire = WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Grpc,
            requires_media_refs: false,
        };

        let stage = &stage;
        let decode_legs = |decode_needs_vision: bool| async move {
            let mut seen = HashMap::new();
            for _ in 0..8 {
                let mut ctx = dispatch_ctx(model_id, wire);
                ctx.routing.decode_needs_vision = decode_needs_vision;
                stage
                    .reselect(&mut ctx)
                    .await
                    .expect("re-selection succeeds");
                match ctx.workers.expect("a pair is selected") {
                    WorkerSelection::Disaggregated { decode, .. } => {
                        *seen.entry(decode.url().to_string()).or_insert(0) += 1;
                    }
                    WorkerSelection::Single { .. } => panic!("expected a PD selection"),
                }
            }
            let mut urls: Vec<String> = seen.into_keys().collect();
            urls.sort();
            urls
        };
        assert_eq!(decode_legs(true).await, ["grpc://127.0.0.1:8970"]);
        assert_eq!(
            decode_legs(false).await,
            ["grpc://127.0.0.1:8970", "grpc://127.0.0.1:8971"]
        );
    }

    /// The stage reads the demand off the request: an image whose decode leg
    /// derives its positions from grid tensors selects a decode worker with
    /// the encoder, and the routing snapshot records the demand for retries;
    /// a single-sample image without grids uses the whole decode pool.
    #[tokio::test]
    async fn execute_keeps_a_vision_demanding_decode_leg_on_a_vision_capable_worker() {
        let registry = Arc::new(WorkerRegistry::new());
        for (url, worker_type, labels) in [
            ("grpc://127.0.0.1:8980", WorkerType::Prefill, VISION_ENCODER),
            ("grpc://127.0.0.1:8990", WorkerType::Decode, VISION_ENCODER),
            (
                "grpc://127.0.0.1:8991",
                WorkerType::Decode,
                NO_VISION_ENCODER,
            ),
        ] {
            registry
                .register(labeled_worker(url, "test-model", worker_type, labels))
                .unwrap();
        }
        let stage = round_robin_pd_stage(Arc::clone(&registry));

        let mut seen = HashMap::new();
        for _ in 0..4 {
            let mut ctx = prepared_ctx(Arc::clone(&registry), None);
            ctx.state.multimodal_intermediate = Some(image_intermediate(Some("image_grid_thw")));
            stage
                .execute(&mut ctx)
                .await
                .expect("an image request with grid tensors selects");
            match ctx.state.workers.as_ref().expect("a pair is selected") {
                WorkerSelection::Disaggregated { decode, .. } => {
                    *seen.entry(decode.url().to_string()).or_insert(0) += 1;
                }
                WorkerSelection::Single { .. } => panic!("expected a PD selection"),
            }
            let snapshot = ctx.state.routing_snapshot.as_ref().expect("snapshot");
            assert!(snapshot.decode_needs_vision);
        }
        assert_eq!(
            seen.into_keys().collect::<Vec<_>>(),
            ["grpc://127.0.0.1:8990"]
        );

        let mut seen = HashMap::new();
        for _ in 0..4 {
            let mut ctx = prepared_ctx(Arc::clone(&registry), None);
            ctx.state.multimodal_intermediate = Some(image_intermediate(None));
            stage
                .execute(&mut ctx)
                .await
                .expect("a single-sample image request selects");
            if let Some(WorkerSelection::Disaggregated { decode, .. }) = ctx.state.workers.as_ref()
            {
                *seen.entry(decode.url().to_string()).or_insert(0) += 1;
            }
            let snapshot = ctx.state.routing_snapshot.as_ref().expect("snapshot");
            assert!(!snapshot.decode_needs_vision);
        }
        assert_eq!(
            seen.len(),
            2,
            "a single-sample image request uses the whole decode pool: {seen:?}"
        );
    }

    #[test]
    fn the_wait_counts_only_once_it_exceeds_the_timeout() {
        let accepted = Instant::now();
        let timeout = Duration::from_secs(60);
        assert_eq!(waited_past_timeout(accepted, accepted, timeout), None);
        assert_eq!(
            waited_past_timeout(accepted, accepted + Duration::from_secs(60), timeout),
            None,
            "exactly the timeout is still inside the bound"
        );
        assert_eq!(
            waited_past_timeout(accepted, accepted + Duration::from_secs(61), timeout),
            Some(Duration::from_secs(61))
        );
        assert_eq!(
            waited_past_timeout(
                accepted + Duration::from_secs(5),
                accepted,
                Duration::from_secs(1)
            ),
            None,
            "a clock that reads before the acceptance is a zero wait, not a refusal"
        );
    }

    /// A request that reaches selection after waiting longer than the queue
    /// timeout inside the gateway is refused with the distinct 503 before any
    /// worker is read, terminal for the retry layer and with a back-off hint.
    #[tokio::test]
    async fn a_request_held_past_the_queue_timeout_is_refused_before_selection() {
        let registry = Arc::new(WorkerRegistry::new());
        registry
            .register(worker("grpc://127.0.0.1:30000", WorkerType::Regular))
            .unwrap();
        let stage =
            regular_stage(Arc::clone(&registry)).with_queue_timeout(Duration::from_millis(1));
        let mut ctx = prepared_ctx(registry, Some(Instant::now()));
        std::thread::sleep(Duration::from_millis(5));

        let refused = stage
            .execute(&mut ctx)
            .await
            .expect_err("a request that waited past the queue timeout is refused");
        assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&refused),
            SELECTION_QUEUE_TIMEOUT_ERROR_CODE
        );
        let retry_after = refused
            .headers()
            .get(RETRY_AFTER)
            .expect("Retry-After present")
            .to_str()
            .unwrap();
        assert!(retry_after.parse::<u64>().is_ok_and(|secs| secs >= 1));
        assert!(!is_retryable_response(&refused));
        assert!(
            ctx.state.workers.is_none(),
            "no worker was selected for a refused request"
        );
    }

    /// Inside the bound selection runs as before; a stage without a timeout,
    /// or a request without an acceptance stamp, is never refused.
    #[tokio::test]
    async fn a_request_inside_the_queue_timeout_is_selected_as_before() {
        let registry = Arc::new(WorkerRegistry::new());
        registry
            .register(worker("grpc://127.0.0.1:30001", WorkerType::Regular))
            .unwrap();

        let stage =
            regular_stage(Arc::clone(&registry)).with_queue_timeout(Duration::from_secs(60));
        let mut ctx = prepared_ctx(Arc::clone(&registry), Some(Instant::now()));
        stage.execute(&mut ctx).await.expect("selected");
        assert!(ctx.state.workers.is_some());

        let unbounded = regular_stage(Arc::clone(&registry));
        let mut ctx = prepared_ctx(Arc::clone(&registry), Some(Instant::now()));
        std::thread::sleep(Duration::from_millis(5));
        unbounded
            .execute(&mut ctx)
            .await
            .expect("no timeout, no refusal");
        assert!(ctx.state.workers.is_some());

        let zero_is_off = regular_stage(Arc::clone(&registry)).with_queue_timeout(Duration::ZERO);
        let mut ctx = prepared_ctx(Arc::clone(&registry), Some(Instant::now()));
        std::thread::sleep(Duration::from_millis(5));
        zero_is_off
            .execute(&mut ctx)
            .await
            .expect("zero disables the bound");

        let bounded =
            regular_stage(Arc::clone(&registry)).with_queue_timeout(Duration::from_millis(1));
        let mut ctx = prepared_ctx(registry, None);
        std::thread::sleep(Duration::from_millis(5));
        bounded
            .execute(&mut ctx)
            .await
            .expect("without an acceptance stamp the wait is unknown and never refused");
        assert!(ctx.state.workers.is_some());
    }

    /// A request's first selection takes the stamp; a later pipeline run of
    /// the same request (a Responses tool loop iteration carries the same
    /// meta) is not refused for the time the request spent generating and
    /// running tools, however long, while a new request that waited is.
    #[tokio::test]
    async fn a_later_run_of_the_same_request_is_not_bounded() {
        let registry = Arc::new(WorkerRegistry::new());
        registry
            .register(worker("grpc://127.0.0.1:30002", WorkerType::Regular))
            .unwrap();
        let stage =
            regular_stage(Arc::clone(&registry)).with_queue_timeout(Duration::from_millis(1));
        let mut first = prepared_ctx(Arc::clone(&registry), Some(Instant::now()));
        stage
            .execute(&mut first)
            .await
            .expect("the first selection");
        let meta = first.input.tenant_request_meta.clone();

        std::thread::sleep(Duration::from_millis(5));
        let mut later = prepared_ctx(Arc::clone(&registry), None);
        later.input.tenant_request_meta = meta;
        stage
            .execute(&mut later)
            .await
            .expect("a later run of the same request is not held to the bound");
        assert!(later.state.workers.is_some());

        let mut fresh = prepared_ctx(registry, Some(Instant::now()));
        std::thread::sleep(Duration::from_millis(5));
        let refused = stage
            .execute(&mut fresh)
            .await
            .expect_err("a new request that waited past the bound is refused");
        assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// A PD request that passes the check and then waits for prefill
    /// admission is held to the remaining budget: it leaves the prefill queue
    /// at the bound, refused, before a slot opens, and holds no claim.
    #[tokio::test]
    async fn the_prefill_admission_wait_is_held_to_the_remaining_budget() {
        let registry = Arc::new(WorkerRegistry::new());
        let prefill = worker("grpc://prefill-bound:30000", WorkerType::Prefill);
        let decode = worker("grpc://decode-bound:30000", WorkerType::Decode);
        for worker in [&prefill, &decode] {
            registry.register(Arc::clone(worker)).unwrap();
        }
        let admission = Arc::new(PrefillAdmission::new(1, 1, Duration::from_secs(5)));
        let occupied = occupy(&admission, &prefill).await;
        let stage = Arc::new(
            pd_stage(
                Arc::clone(&registry),
                Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
                Arc::clone(&admission),
            )
            .with_queue_timeout(Duration::from_millis(50)),
        );

        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter task is joined before the test ends"
        )]
        let queued = tokio::spawn({
            let stage = Arc::clone(&stage);
            let registry = Arc::clone(&registry);
            async move {
                let mut ctx = prepared_ctx(registry, Some(Instant::now()));
                let result = stage.execute(&mut ctx).await;
                (result, ctx.state.pd_prefill_guard.is_some(), Instant::now())
            }
        });
        wait_for_queued(&admission, 1).await;
        tokio::time::sleep(Duration::from_millis(120)).await;
        let freed_at = Instant::now();
        drop(occupied);

        let (result, holds_claim, refused_at) = queued.await.expect("waiter should join");
        assert!(
            refused_at < freed_at,
            "refused at the bound, while still queued, not once a slot opened"
        );
        assert_eq!(
            admission.queued_requests(),
            0,
            "the expired waiter left the queue"
        );
        let refused = result.expect_err("admitted past the bound, the request is refused");
        assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&refused),
            SELECTION_QUEUE_TIMEOUT_ERROR_CODE
        );
        assert!(!holds_claim, "the prefill claim went back");
        assert_eq!(prefill.load(), 0, "no load left on the prefill worker");
    }

    // ---- the refusal line of a drained pool -------------------------------------------------

    /// Counts, on this thread's subscriber, the ERROR events and the stage's
    /// refusal lines (the events carrying a `suppressed` field).
    struct RefusalLineCounter {
        errors: Arc<AtomicUsize>,
        refusal_lines: Arc<AtomicUsize>,
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for RefusalLineCounter {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if *event.metadata().level() == tracing::Level::ERROR {
                self.errors.fetch_add(1, Ordering::Relaxed);
            }
            if event.metadata().fields().field("suppressed").is_some() {
                self.refusal_lines.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// A model whose only worker is down is refused with the
    /// `no_available_workers` 503, marked as reported for the loggers
    /// downstream, and written at WARN exactly up to the per-second
    /// allowance of its own, never at ERROR: three ERROR lines per refusal
    /// made a client's retry loop against a drained pool a log storm, and a
    /// second model down at the same time still gets its own lines.
    #[tokio::test]
    async fn a_refusal_for_a_drained_pool_is_marked_and_written_at_warn_within_the_allowance() {
        use openai_protocol::worker::WorkerStatus;
        use tracing_subscriber::layer::SubscriberExt;

        let errors = Arc::new(AtomicUsize::new(0));
        let refusal_lines = Arc::new(AtomicUsize::new(0));
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::Registry::default().with(RefusalLineCounter {
                errors: Arc::clone(&errors),
                refusal_lines: Arc::clone(&refusal_lines),
            }),
        );
        let registry = Arc::new(WorkerRegistry::new());
        let down = worker("grpc://127.0.0.1:30010", WorkerType::Regular);
        down.set_status(WorkerStatus::NotReady);
        registry.register(down).unwrap();
        let stage = regular_stage(Arc::clone(&registry));
        let allowance = usize::try_from(NO_AVAILABLE_WORKERS_LINES_PER_SECOND).unwrap();

        // The stage's entry point marks and reports the refusal.
        let mut ctx = prepared_ctx(Arc::clone(&registry), None);
        let refused = stage
            .execute(&mut ctx)
            .await
            .expect_err("no worker is available");
        assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&refused),
            placement::NO_AVAILABLE_WORKERS_CODE
        );
        assert!(
            refused.extensions().get::<ReportedRefusal>().is_some(),
            "the refusal is marked as reported, so the pipeline and the access log keep quiet"
        );

        // Under a throttle of its own (the process-wide one is shared with
        // every other test), the lines are exactly the allowance per model.
        let throttle = KeyedLogThrottle::new(
            NO_AVAILABLE_WORKERS_LINES_PER_SECOND,
            Duration::from_secs(60),
            NO_AVAILABLE_WORKERS_LOG_MODELS,
        );
        refusal_lines.store(0, Ordering::Relaxed);
        let refusals = 5 * allowance;
        for _ in 0..refusals {
            let mut ctx = prepared_ctx(Arc::clone(&registry), None);
            let refused = stage
                .select(&mut ctx)
                .await
                .expect_err("no worker is available");
            let reported = stage.report_refusal_with(
                &throttle,
                "WorkerSelectionStage::execute",
                None,
                "test-model",
                refused,
            );
            assert!(reported.extensions().get::<ReportedRefusal>().is_some());
        }
        assert_eq!(
            refusal_lines.load(Ordering::Relaxed),
            allowance,
            "{refusals} refusals of one model inside one window write exactly {allowance} lines"
        );
        for _ in 0..refusals {
            stage.report_refusal_with(
                &throttle,
                "WorkerSelectionStage::reselect",
                Some("req-other"),
                "other-model",
                WorkerSelectionStage::workers_unavailable("other-model"),
            );
        }
        assert_eq!(
            refusal_lines.load(Ordering::Relaxed),
            2 * allowance,
            "a second model down at the same time gets its own {allowance} lines"
        );
        assert_eq!(
            errors.load(Ordering::Relaxed),
            0,
            "a refusal is not an error of the gateway"
        );
    }
}
