//! Request execution: dispatch one attempt of the retained execution plan.

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::response::Response;
use futures::future::{join_all, try_join_all};
use smg_grpc_client::vllm_proto as vllm;
use tracing::{debug, error, info_span, warn, Instrument};

use super::{
    helpers::{maybe_inject_pd_metadata, maybe_inject_pd_rendezvous},
    pd_protocol::{DpPlacement, PdDispatch, PdProtocol},
};
use crate::{
    observability::{
        cache_trace,
        metrics::{metrics_labels, Metrics},
    },
    routers::{
        common::{
            attempt_ledger::{mark_failed_leg, mark_upstream_answer, AttemptLedger},
            kv_transfer::{
                connector_mode_for_worker, is_moriio_worker, mooncake_decode_params,
                mooncake_prefill_params, KvConnectorMode, NIXL_PREFILL_KV_PARAMS,
            },
            pd_admission,
            retry::mark_non_retryable,
        },
        error,
        grpc::{
            common::stages::encode::EncodeDispatchPlan,
            context::{
                ClientSelection, DispatchContext, ExecutionPlan, ExecutionPlanKind,
                ExecutionResult, LoadGuards, PdTiming, WorkerSelection,
            },
            multimodal::worker_language_model_only,
            proto_wrapper::{
                FanoutStream, FirstItem, PrefillCachedTokens, PrefillTicket, ProtoEmbedRequest,
                ProtoGenerateRequest, ProtoRequest, ProtoResponseVariant, ProtoStream,
            },
            utils::{
                account_pd_leg_failure,
                stream_failure::UpstreamClass,
                tonic_ext::{TonicResultExt, TonicStatusExt},
            },
        },
    },
    worker::{liveness, worker::WorkerTypeExt, ConnectionModeExt, PrefillLoadGuard, Worker},
};

type StreamResult = Result<ProtoStream, tonic::Status>;

/// One leg's owned dispatch. Owned (rather than borrowed from the client
/// selection) so the leg still in flight when its partner fails can be moved
/// off the request path — see [`retire_pd_leg`].
type PdLegDispatch = Pin<Box<dyn Future<Output = StreamResult> + Send>>;

/// Which leg of a disaggregated dispatch a result belongs to. The error code
/// and message are part of the client contract, so they live per leg here
/// rather than being reconstructed at each call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PdLeg {
    Prefill,
    Decode,
}

impl PdLeg {
    fn name(self) -> &'static str {
        match self {
            Self::Prefill => metrics_labels::WORKER_PREFILL,
            Self::Decode => metrics_labels::WORKER_DECODE,
        }
    }

    fn error_code(self) -> &'static str {
        match self {
            Self::Prefill => "prefill_worker_failed_to_start",
            Self::Decode => "decode_worker_failed_to_start",
        }
    }

    fn error_message(self) -> &'static str {
        match self {
            Self::Prefill => "Prefill worker failed to start",
            Self::Decode => "Decode worker failed to start",
        }
    }

    fn partner(self) -> Self {
        match self {
            Self::Prefill => Self::Decode,
            Self::Decode => Self::Prefill,
        }
    }
}

/// How the two parallel legs resolved.
#[expect(
    clippy::large_enum_variant,
    reason = "consumed by its caller on the next line; boxing would add an allocation to the success path to save one stack move"
)]
enum PdDispatchOutcome {
    /// Both legs answered; either or both may carry an error.
    Both(StreamResult, StreamResult),
    /// One leg failed while its partner was still dispatching. The partner is
    /// handed back so the caller can retire it instead of waiting it out.
    FailedFirst {
        leg: PdLeg,
        error: tonic::Status,
        partner: PdLegDispatch,
    },
}

/// Backend requests one plan dispatches — one per batched prompt, times the
/// PD fan-out width where a parallel PD request asks for n>1 samples.
///
/// This is both the load-guard scale and the number of PD bootstrap rooms the
/// plan will post, which is why admission and the guards read the same count.
fn plan_sub_requests(plan: &ExecutionPlan, workers: Option<&WorkerSelection>) -> usize {
    let protocol = workers
        .and_then(WorkerSelection::disaggregated_runtime_type)
        .and_then(|runtime| PdProtocol::for_runtime(*runtime));
    let width = |request: &ProtoGenerateRequest| {
        protocol
            .and_then(|protocol| pd_fanout_width(request, protocol))
            .map_or(1, |n| n as usize)
    };
    match plan {
        ExecutionPlan::Batch {
            kind: ExecutionPlanKind::Single,
            requests,
            ..
        } => requests.len(),
        ExecutionPlan::Batch { requests, .. } => requests.iter().map(width).sum(),
        ExecutionPlan::PrefillDecode(request) | ExecutionPlan::EncodePrefillDecode { request } => {
            width(request)
        }
        ExecutionPlan::Single(_) => 1,
    }
}

/// Fan-out width for a parallel PD dispatch: `n` when the request asks for
/// more than one sample of a text-only prompt, else `None`.
///
/// A rendezvous room serves one sample. The engines that rendezvous on a
/// room broadcast the request's single room to every sample of an n>1
/// request: each of the decode's children then pre-allocates against the
/// same room, and the prefill either rejects the repeats (TokenSpeed) or
/// serves one child while the rest wait on KV that never comes. So each
/// sample becomes its own single-sample PD dispatch with its own room, and
/// the merged streams stamp every child's position as the choice index. A
/// multimodal payload travels as one SHM segment that the prefill unlinks on
/// read, so it cannot be handed to n prefills; those requests keep the
/// single dispatch.
fn pd_fanout_width(request: &ProtoGenerateRequest, protocol: PdProtocol) -> Option<u32> {
    if protocol.dispatch != PdDispatch::Parallel
        || request.has_mm_inputs()
        || request.has_vllm_media_refs()
    {
        return None;
    }
    let n = request.sampling_n();
    (n > 1).then_some(n)
}

fn pd_sub_request_id(base_id: &str, index: u32) -> String {
    format!("{base_id}-{index}")
}

fn trace_engine_ids(
    plan: &mut ExecutionPlan,
    workers: Option<&WorkerSelection>,
) -> (Vec<String>, bool) {
    let protocol = if matches!(
        plan,
        ExecutionPlan::Single(_)
            | ExecutionPlan::Batch {
                kind: ExecutionPlanKind::Single,
                ..
            }
    ) {
        None
    } else {
        workers
            .and_then(WorkerSelection::disaggregated_runtime_type)
            .and_then(|runtime| PdProtocol::for_runtime(*runtime))
    };
    let mut ids: Vec<_> = plan
        .generate_requests_mut()
        .flat_map(|request| {
            let width = protocol.and_then(|protocol| pd_fanout_width(request, protocol));
            let base_id = request.request_id().to_owned();
            (0..width.unwrap_or(1)).map(move |index| {
                if width.is_some() {
                    pd_sub_request_id(&base_id, index)
                } else {
                    base_id.clone()
                }
            })
        })
        .take(33)
        .collect();
    let complete = !ids.is_empty() && ids.len() <= 32;
    ids.truncate(32);
    (ids, complete)
}

/// Give the decode leg the media identity the prefill leg produced, so it
/// is served without pixels or references. Only on a leg that will pull its
/// prompt KV from prefill: it must hold a KV handoff (`handed_off`) and be
/// a single-sample request (`relay_kv_params`); with n>1 decode recomputes
/// the prompt and needs the references itself, even where legacy Mooncake
/// still injects its host and port, and a pixel-less leg without KV is
/// refused by the servicer. A prefill that was asked for an identity
/// (`solicited`: its request carried KV params) but returned none is an
/// older servicer; the leg is left as it is, and decode reprocesses the
/// media.
fn apply_prefill_media_identity(
    decode_request: &mut ProtoGenerateRequest,
    handed_off: bool,
    relay_kv_params: bool,
    solicited: bool,
    identity: Option<&vllm::MediaIdentity>,
) {
    if !handed_off || !relay_kv_params || !decode_request.has_vllm_media_refs() {
        return;
    }
    let applied = match identity {
        Some(identity) => decode_request.apply_media_identity(identity),
        None if solicited => {
            warn!(
                request_id = %decode_request.request_id(),
                "prefill worker returned no media identity; decode leg will reprocess media"
            );
            return;
        }
        None => return,
    };
    if !applied {
        warn!(
            request_id = %decode_request.request_id(),
            "prefill worker's media identity is unusable; decode leg will reprocess media"
        );
    }
}

/// Split an n>1 request into `n` single-sample sub-requests. Sub `i` carries
/// engine id `{id}-{i}`, `n = 1`, a seed offset of `i` when the request
/// pinned a seed, and whatever rendezvous `remint` stamps on it.
fn fan_out_pd_request(
    request: &ProtoGenerateRequest,
    n: u32,
    mut remint: impl FnMut(&mut ProtoGenerateRequest),
) -> Vec<ProtoGenerateRequest> {
    let base_id = request.request_id().to_string();
    (0..n)
        .map(|i| {
            let mut sub = request.clone();
            sub.set_request_id(pd_sub_request_id(&base_id, i));
            sub.set_sampling_n(1);
            sub.offset_sampling_seed(i);
            remint(&mut sub);
            sub
        })
        .collect()
}

/// Metric connection labels for the PD legs (a leg can be gRPC or ZMQ).
fn pd_leg_labels(workers: &WorkerSelection) -> (&'static str, &'static str) {
    match workers {
        WorkerSelection::Disaggregated {
            prefill, decode, ..
        } => (
            prefill.connection_mode().as_metric_label(),
            decode.connection_mode().as_metric_label(),
        ),
        WorkerSelection::Single { worker } => {
            let label = worker.connection_mode().as_metric_label();
            (label, label)
        }
    }
}

/// Dispatch one attempt of the retained plan: create the attempt's load
/// guards, fan out encode jobs on the first EPD dispatch, and store the
/// execution result on the context for response processing.
///
/// `last_attempt` says whether a plan is still retained for a replay, which
/// is what decides when the media bytes stop counting against the in-flight
/// budget.
pub(crate) async fn execute_plan(
    ctx: &mut DispatchContext,
    mut execution_plan: ExecutionPlan,
    last_attempt: bool,
) -> Result<(), Response> {
    // One bootstrap room per backend request the plan will post: a batched
    // completion fans out one PD dispatch per sub-request, so admission has
    // to claim for all of them or the siblings walk past a gate that only
    // ever asked about one.
    let sub_requests = plan_sub_requests(&execution_plan, ctx.workers.as_ref());

    // Admission runs before this attempt claims anything else. The decode
    // leg's engine window is what clears the prefill's bootstrap deadline, so
    // a request the decode cannot admit yet waits here rather than in the
    // engine's queue — where it would burn the prefill's deadline and strand
    // both rooms. Waiting ahead of the encode take below also keeps the
    // encode jobs' SHM unclaimed for the duration: under the burst this gate
    // targets, holding it would queue a second scarce resource behind the
    // decode window and throw the finished encode work away on a shed. The
    // gate abstains when the engine reports no window.
    let admission = match ctx
        .workers
        .as_ref()
        .and_then(WorkerSelection::decode_worker)
    {
        Some(decode) => pd_admission::admit_decode(decode, &ctx.model_id, sub_requests).await?,
        None => None,
    };

    // `None` for non-EPD, text-only EPD, or an EPD retry (the first dispatch
    // consumed it). Taking it transfers the encode jobs' SHM Drop guards
    // here: dispatch consumes them, while an early error before dispatch
    // drops them and reclaims the SHM.
    let encode_dispatch = ctx.encode_outputs.take().map(|o| o.dispatch);

    let clients = ctx.clients.as_mut().ok_or_else(|| {
        error!(
            function = "execute_plan",
            "Client acquisition not completed"
        );
        error::internal_error(
            "client_acquisition_not_completed",
            "Client acquisition not completed",
        )
    })?;

    // Create load guards for worker load tracking (increment load when created)
    // They will be automatically dropped (and decrement load) when the
    // dispatch context (or the streaming body they are attached to) drops.
    let workers = ctx.workers.as_ref().ok_or_else(|| {
        error!(function = "execute_plan", "Worker selection not completed");
        error::internal_error(
            "worker_selection_not_completed",
            "Worker selection not completed",
        )
    })?;
    // Every breaker sample of this attempt goes through the request's
    // ledger, which charges a worker once per request however often the
    // request retried on it.
    let ledger = &ctx.ledger;

    ctx.load_guards = Some(LoadGuards::admitted(
        admission,
        LoadGuards::scaled(workers, ctx.sticky_key.as_deref(), sub_requests),
    ));
    // The Prefill admission slot worker selection took for this attempt. It
    // is released when the Prefill phase ends, not with the load guards.
    let prefill_guard = ctx.pd_prefill_guard.take();

    // Extract dispatch metadata for the tracing span and PD metric labels.
    let dispatch = ctx.dispatch.as_ref().ok_or_else(|| {
        error!(function = "execute_plan", "Dispatch metadata not set");
        error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
    })?;
    let request_id = dispatch.request_id.as_str();
    let model = dispatch.model.as_str();
    let request_type = execution_plan.request_type();
    let mode = execution_plan.mode_label();
    if cache_trace::enabled() {
        let (engine_ids, engine_ids_complete) =
            trace_engine_ids(&mut execution_plan, Some(workers));
        ctx.cache_trace = cache_trace::dispatch(
            ctx.root_request_id.as_deref(),
            ctx.attempt,
            engine_ids,
            engine_ids_complete,
            mode,
        );
    }

    // The upstream span: the dispatch, then the response stream (it closes
    // with the load guards, which the response body holds to its last frame;
    // see `LoadGuards::traced`). A failed dispatch sets its status to ERROR;
    // success is left unset, as the RPC conventions have it.
    let span = info_span!(
        target: "smg::otel-trace",
        "grpc_execute",
        request_type,
        request_id = %request_id,
        model = %model,
        mode = %mode,
        otel.status_code = tracing::field::Empty,
    );

    let result = async {
        match execution_plan {
            ExecutionPlan::Single(request) => match request {
                ProtoRequest::Generate(req) => execute_single(req, clients, workers, ledger).await,
                ProtoRequest::Embed(req) => {
                    execute_single_embed(req, clients, workers, ledger).await
                }
            },
            ExecutionPlan::PrefillDecode(req) => {
                let prefill_guard = require_prefill_guard(prefill_guard)?;
                execute_pd_dispatch(req, clients, workers, model, prefill_guard, ledger).await
            }
            ExecutionPlan::EncodePrefillDecode { request } => {
                // Bootstrap info was injected into the prefill request during
                // request building; dispatch the encode jobs with the
                // prefill+decode leg.
                let prefill_guard = require_prefill_guard(prefill_guard)?;
                execute_epd_dispatch(
                    request,
                    clients,
                    workers,
                    model,
                    encode_dispatch,
                    prefill_guard,
                    ledger,
                )
                .await
            }
            ExecutionPlan::Batch { kind, requests, .. } => {
                execute_batch_dispatch(
                    kind,
                    requests,
                    clients,
                    workers,
                    model,
                    prefill_guard,
                    ledger,
                )
                .await
            }
        }
    }
    .instrument(span.clone())
    .await;
    // The engines hold the request bodies now. An earlier attempt keeps its
    // share of the budget: the retained plan still owns the same media, and a
    // replay would send it again.
    if last_attempt {
        ctx.multimodal_inflight.take();
    }
    let result = match result {
        Ok(result) => result,
        Err(response) => {
            span.record("otel.status_code", "ERROR");
            return Err(response);
        }
    };
    if let Some(guards) = ctx.load_guards.take() {
        ctx.load_guards = Some(LoadGuards::traced(span, guards));
    }

    // Store result in context for response processing
    ctx.response.execution_result = Some(result);
    Ok(())
}

/// A disaggregated dispatch without the Prefill admission slot worker
/// selection took for it is a pipeline bug, not a routable request.
fn require_prefill_guard(guard: Option<PrefillLoadGuard>) -> Result<PrefillLoadGuard, Response> {
    guard.ok_or_else(|| {
        error!(
            function = "execute_plan",
            "PD dispatch without a Prefill admission slot"
        );
        error::internal_error(
            "prefill_admission_missing",
            "PD dispatch without a Prefill admission slot",
        )
    })
}

async fn execute_pd_dispatch(
    proto_request: ProtoGenerateRequest,
    clients: &mut ClientSelection,
    workers: &WorkerSelection,
    model: &str,
    prefill_guard: PrefillLoadGuard,
    ledger: &AttemptLedger,
) -> Result<ExecutionResult, Response> {
    let Some(runtime_type) = workers.disaggregated_runtime_type() else {
        error!(
            function = "execute_pd_dispatch",
            "PD mode requires disaggregated worker selection"
        );
        return Err(error::internal_error(
            "pd_mode_requires_disaggregated_workers",
            "PD mode requires disaggregated worker selection",
        ));
    };
    let Some(protocol) = PdProtocol::for_runtime(*runtime_type) else {
        error!(
            function = "execute_pd_dispatch",
            runtime_type = ?runtime_type,
            "Runtime does not support PD disaggregated mode"
        );
        return Err(error::bad_request(
            "runtime_pd_not_supported",
            "This runtime does not support PD disaggregated mode",
        ));
    };
    // Dispatch shape comes from the per-runtime PD protocol table (see
    // `PdProtocol::for_runtime`): sequential legs relay the KV handoff
    // through the router, parallel legs rendezvous on bootstrap info
    // carried in the request.
    let result = match protocol.dispatch {
        PdDispatch::Sequential => {
            execute_sequential_pd(
                proto_request,
                clients,
                workers,
                model,
                prefill_guard,
                ledger,
            )
            .await
        }
        PdDispatch::Parallel => match pd_fanout_width(&proto_request, protocol) {
            Some(n) => {
                execute_fanout_pd(
                    proto_request,
                    n,
                    clients,
                    workers,
                    protocol,
                    prefill_guard,
                    ledger,
                )
                .await
            }
            None => {
                execute_parallel_pd(
                    proto_request,
                    clients,
                    workers,
                    protocol,
                    prefill_guard,
                    ledger,
                )
                .await
            }
        },
    }?;
    Ok(attribute_pd_legs(result, workers))
}

/// Pair each leg's stream of a disaggregated dispatch with the worker it was
/// sent to, so a failure of the stream can be charged to that worker wherever
/// the stream is read: the streamed answer knows its legs from the request's
/// selection, the buffered one reads the streams alone and had no worker to
/// charge. Each response then counts as that worker's token progress for the
/// liveness monitor, as a regular dispatch's does. The legs join no tracked
/// pile and add no prefill backlog: those are the regular dispatch's rules,
/// and a disaggregated request is admitted through its prefill guard. The
/// sequential dispatch's prefill stream is drained already; the wrapper is a
/// no-op on it.
fn attribute_pd_legs(result: ExecutionResult, workers: &WorkerSelection) -> ExecutionResult {
    let ExecutionResult::PrefillDecode {
        prefill,
        decode,
        prefill_guards,
        pd_timing,
        prefill_input_logprobs,
    } = result
    else {
        return result;
    };
    let attribute = |stream: ProtoStream, worker: Option<&Arc<dyn Worker>>| match worker {
        Some(worker) => stream.tracked(Arc::clone(worker), 0, false),
        None => stream,
    };
    ExecutionResult::PrefillDecode {
        prefill: attribute(prefill, workers.prefill_worker()),
        decode: Box::new(attribute(*decode, workers.decode_worker())),
        prefill_guards,
        pd_timing,
        prefill_input_logprobs,
    }
}

/// Dispatch an n>1 request as `n` concurrent single-sample PD pairs, each
/// with its own rendezvous room, and merge their legs into one PD result
/// whose responses carry the sample index. Fail-fast: the first pair that
/// fails to start fails the request, and dropping the others aborts them.
async fn execute_fanout_pd(
    proto_request: ProtoGenerateRequest,
    n: u32,
    clients: &mut ClientSelection,
    workers: &WorkerSelection,
    protocol: PdProtocol,
    prefill_guard: PrefillLoadGuard,
    ledger: &AttemptLedger,
) -> Result<ExecutionResult, Response> {
    let subs = fan_out_pd_request(&proto_request, n, |sub| {
        maybe_inject_pd_metadata(sub, workers);
        maybe_inject_pd_rendezvous(sub, workers);
    });
    debug!(
        request_id = proto_request.request_id(),
        samples = n,
        "PD fan-out: one single-sample pair per sample, each with its own room"
    );
    // The samples share the one admission slot; each pair's Prefill phase
    // ends on its own, so each carries its own handle.
    let guards = prefill_guard.replicate_to(subs.len());
    let dispatches = subs.into_iter().zip(guards).map(|(sub, guard)| {
        let mut clients = clients.clone();
        async move {
            execute_parallel_pd(sub, &mut clients, workers, protocol, guard, ledger).await
        }
    });
    let results = try_join_all(dispatches).await?;

    let mut prefills = Vec::with_capacity(results.len());
    let mut decodes = Vec::with_capacity(results.len());
    let mut prefill_guards = Vec::with_capacity(results.len());
    let mut timing: Option<PdTiming> = None;
    for result in results {
        let ExecutionResult::PrefillDecode {
            prefill,
            decode,
            prefill_guards: guards,
            pd_timing,
            prefill_input_logprobs: _,
        } = result
        else {
            error!(
                function = "execute_fanout_pd",
                "PD fan-out child returned a non-PD result"
            );
            return Err(error::internal_error(
                "pd_fanout_unexpected_result",
                "PD fan-out child returned a non-PD result",
            ));
        };
        prefills.push(prefill);
        decodes.push(*decode);
        prefill_guards.extend(guards);
        // The earliest prefill start anchors the merged request's TTFT.
        timing = Some(match timing {
            Some(earliest) if earliest.prefill_start <= pd_timing.prefill_start => earliest,
            _ => pd_timing,
        });
    }
    let Some(pd_timing) = timing else {
        return Err(error::internal_error(
            "pd_fanout_empty",
            "PD fan-out produced no dispatch",
        ));
    };
    Ok(ExecutionResult::PrefillDecode {
        prefill: ProtoStream::Fanout(FanoutStream::new(prefills)),
        decode: Box::new(ProtoStream::Fanout(FanoutStream::new(decodes))),
        prefill_guards,
        pd_timing,
        // Fan-out children keep live prefill streams; the streaming layer
        // drains them for input logprobs.
        prefill_input_logprobs: None,
    })
}

async fn execute_epd_dispatch(
    mut proto_request: ProtoGenerateRequest,
    clients: &mut ClientSelection,
    workers: &WorkerSelection,
    model: &str,
    encode_dispatch: Option<EncodeDispatchPlan>,
    prefill_guard: PrefillLoadGuard,
    ledger: &AttemptLedger,
) -> Result<ExecutionResult, Response> {
    if let Some(encode_dispatch) = encode_dispatch {
        spawn_encode_dispatch(encode_dispatch);
    }
    proto_request.clear_mm_pixel_values();
    execute_pd_dispatch(
        proto_request,
        clients,
        workers,
        model,
        prefill_guard,
        ledger,
    )
    .await
}

#[expect(
    clippy::disallowed_methods,
    reason = "EPD encode dispatch is intentionally supervised in the background while the prefill leg blocks on embedding receive."
)]
fn spawn_encode_dispatch(encode_dispatch: EncodeDispatchPlan) {
    if encode_dispatch.is_empty() {
        return;
    }

    let num_encode_items = encode_dispatch.len();
    let sends: Vec<_> = encode_dispatch
        .into_jobs()
        .into_iter()
        .map(|job| tokio::spawn(async move { job.dispatch().await }))
        .collect();

    debug!(
        num_encode_items,
        "EPD encode dispatch issued with prefill/decode"
    );

    tokio::spawn(async move {
        for join_res in join_all(sends).await {
            match join_res {
                Ok(Ok(())) => {}
                Ok(Err(message)) => {
                    error!(
                        function = "execute_epd_dispatch",
                        error = %message,
                        "Backend encode dispatch failed after EPD dispatch; embedding-receive timeout will abort the request"
                    );
                }
                Err(join_err) => {
                    error!(
                        function = "execute_epd_dispatch",
                        error = %join_err,
                        "Encode dispatch task panicked after EPD dispatch"
                    );
                }
            }
        }
    });
}

/// Dispatch one backend request per batched prompt concurrently, preserving
/// prompt order. Fail-fast: the first failed dispatch fails the batch and
/// drops the remaining streams (abort-on-drop reclaims them backend-side).
async fn execute_batch_dispatch(
    kind: ExecutionPlanKind,
    requests: Vec<ProtoGenerateRequest>,
    clients: &ClientSelection,
    workers: &WorkerSelection,
    model: &str,
    prefill_guard: Option<PrefillLoadGuard>,
    ledger: &AttemptLedger,
) -> Result<ExecutionResult, Response> {
    // One Prefill handle per PD sub-request, all on the one admission slot.
    let prefill_guards: Vec<Option<PrefillLoadGuard>> = match kind {
        ExecutionPlanKind::Single => requests.iter().map(|_| None).collect(),
        ExecutionPlanKind::PrefillDecode | ExecutionPlanKind::EncodePrefillDecode => {
            require_prefill_guard(prefill_guard)?
                .replicate_to(requests.len())
                .into_iter()
                .map(Some)
                .collect()
        }
    };
    let dispatches = requests
        .into_iter()
        .zip(prefill_guards)
        .map(|(request, prefill_guard)| {
            let mut clients = clients.clone();
            async move {
                match kind {
                    ExecutionPlanKind::Single => {
                        execute_single(request, &mut clients, workers, ledger).await
                    }
                    // Completion EPD carries no encode jobs; sub-requests dispatch as PD.
                    ExecutionPlanKind::PrefillDecode | ExecutionPlanKind::EncodePrefillDecode => {
                        let prefill_guard = require_prefill_guard(prefill_guard)?;
                        execute_pd_dispatch(
                            request,
                            &mut clients,
                            workers,
                            model,
                            prefill_guard,
                            ledger,
                        )
                        .await
                    }
                }
            }
        });

    let results = try_join_all(dispatches).await?;
    Ok(ExecutionResult::Batch { results })
}

async fn execute_single(
    mut proto_request: ProtoGenerateRequest,
    clients: &mut ClientSelection,
    workers: &WorkerSelection,
    ledger: &AttemptLedger,
) -> Result<ExecutionResult, Response> {
    let client = clients.single_mut().ok_or_else(|| {
        error!(
            function = "execute_single",
            "Expected single client but got disaggregated"
        );
        error::internal_error(
            "expected_single_client_got_disaggregated",
            "Expected single client but got disaggregated",
        )
    })?;

    if let Some(rank) = workers.single().and_then(|w| w.dp_rank()) {
        proto_request.set_data_parallel_rank(rank as i32);
    }

    let prompt_tokens = u64::try_from(proto_request.prompt_len()).unwrap_or(u64::MAX);
    let streaming = proto_request.stream();
    if let Some(worker) = workers.single() {
        Metrics::record_worker_request(worker.url(), worker.model_id());
    }
    let result = client.generate(proto_request).await;
    if let Some(status) = result.breaker_status() {
        workers.record_outcome(ledger, status);
    }

    let stream = result.map_err(|e| {
        if let Some(worker) = workers.single() {
            UpstreamClass::account_dispatch_failure(
                worker.worker_type().as_metric_label(),
                worker.connection_mode().as_metric_label(),
                &e,
            );
        }
        start_failure_response(
            &e,
            "execute_single",
            "Failed to start generation",
            "start_generation_failed",
        )
    })?;
    // Every generation to one worker is tracked: each response on it is
    // progress for the worker (liveness), the one answer of a non-streaming
    // generation included, and its prompt is pending prefill there until the
    // first response. Only a streaming generation joins the worker's pile: a
    // non-streaming one shows nothing between dispatch and completion, and
    // the pile rule never judges a worker by requests it cannot see progress
    // on.
    let stream = match workers.single() {
        Some(worker) => stream.tracked(Arc::clone(worker), prompt_tokens, streaming),
        None => stream,
    };

    Ok(ExecutionResult::Single { stream })
}

/// Client answer for a request the worker did not start; an engine rejection (4xx) is not a fault.
fn start_failure_response(
    e: &tonic::Status,
    function: &'static str,
    description: &str,
    code: &str,
) -> Response {
    // The request's media failed, not the worker (see `TonicStatusExt::media_fault`):
    // the answer is for the media, and another worker fetching the same URL
    // would meet the same host, so the request is not replayed.
    if let Some(fault) = e.media_fault() {
        warn!(function = function, error = %e, ?fault, "{}: the request's media failed", description);
        let mut response = e.to_http_error("media_fetch_failed", e.message().to_string());
        mark_non_retryable(&mut response);
        return response;
    }
    if e.http_status().is_client_error() {
        warn!(function = function, error = %e, "{}: engine rejected the request", description);
    } else {
        error!(function = function, error = %e, "{}", description);
    }
    let mut response = e.to_http_error(code, format!("{description}: {}", e.message()));
    // INTERNAL is the servicers' code for a request the engine took and failed
    // on: the worker's own answer for this payload, which the retry layer does
    // not replay on it (see `routers::common::attempt_ledger`).
    if e.code() == tonic::Code::Internal {
        mark_upstream_answer(&mut response);
    }
    // The worker already spent its whole media-sidecar budget on this input;
    // another attempt costs the same again. Other sidecar failures fail fast.
    if e.code() == tonic::Code::Unavailable && e.message().starts_with("sidecar_timeout") {
        mark_non_retryable(&mut response);
    }
    response
}

async fn execute_single_embed(
    proto_request: ProtoEmbedRequest,
    clients: &mut ClientSelection,
    workers: &WorkerSelection,
    ledger: &AttemptLedger,
) -> Result<ExecutionResult, Response> {
    let client = clients.single_mut().ok_or_else(|| {
        error!(
            function = "execute_single_embed",
            "Expected single client but got disaggregated"
        );
        error::internal_error(
            "expected_single_client_got_disaggregated",
            "Expected single client but got disaggregated",
        )
    })?;

    if let Some(worker) = workers.single() {
        Metrics::record_worker_request(worker.url(), worker.model_id());
    }
    let result = client.embed(proto_request).await;
    if let Some(status) = result.breaker_status() {
        workers.record_outcome(ledger, status);
    }

    let complete = result.map_err(|e| {
        if let Some(worker) = workers.single() {
            UpstreamClass::account_dispatch_failure(
                worker.worker_type().as_metric_label(),
                worker.connection_mode().as_metric_label(),
                &e,
            );
        }
        start_failure_response(
            &e,
            "execute_single_embed",
            "Failed to start embedding",
            "start_embedding_failed",
        )
    })?;

    Ok(ExecutionResult::Embedding { response: complete })
}

async fn execute_parallel_pd(
    proto_request: ProtoGenerateRequest,
    clients: &mut ClientSelection,
    workers: &WorkerSelection,
    protocol: PdProtocol,
    prefill_guard: PrefillLoadGuard,
    ledger: &AttemptLedger,
) -> Result<ExecutionResult, Response> {
    let runtime = workers
        .disaggregated_runtime_type()
        .map(|r| r.as_str())
        .unwrap_or("");
    let (prefill_client, decode_client) = clients.disaggregated_mut().ok_or_else(|| {
        error!(
            function = "execute_parallel_pd",
            "Expected disaggregated clients but got single"
        );
        error::internal_error(
            "expected_disaggregated_clients_got_single",
            "Expected disaggregated clients but got single",
        )
    })?;

    // Decode consumes the KV handoff from prefill, but TokenSpeed still
    // needs multimodal metadata to pad placeholders and compute MRoPE in
    // the same way as prefill. Drop raw pixels and prefill-only encode
    // rooms, but keep the per-item metadata. The pixel-free leg is the
    // clone, so pixel tensors are never duplicated.
    let mut prefill_request = proto_request;
    let mut decode_request = prefill_request.clone_without_mm_pixels();
    decode_request.clear_encode_bootstrap_info();
    // Pin each leg's DP rank only when the engine reads placement from
    // the request field; `RoomResidue` engines take it from the bootstrap
    // room minted in maybe_inject_pd_rendezvous, ignore the pin, and spam
    // conflict warnings on a mismatched decode-leg pin.
    if protocol.dp_placement == DpPlacement::PinField {
        if let Some(rank) = workers.prefill_worker().and_then(|w| w.dp_rank()) {
            prefill_request.set_data_parallel_rank(rank as i32);
        }
        if let Some(rank) = workers.decode_worker().and_then(|w| w.dp_rank()) {
            decode_request.set_data_parallel_rank(rank as i32);
        }
    }

    // `generate` only establishes the prefill stream here (SMG does not drain
    // it on this path), so prefill duration cannot be measured — only TTFT,
    // recorded at the first decode token in streaming. prefill_start anchors it.
    let prefill_start = Instant::now();
    let (prefill_label, decode_label) = pd_leg_labels(workers);
    // The prefill leg's silence is bounded from its dispatch: a call that
    // returns no stream, or a stream with nothing on it, within the bound
    // fails the attempt instead of holding the request (and the decode room
    // the admission gate claimed for it) until its own timeout.
    let prefill_worker = workers.prefill_worker();
    let prompt_tokens = u64::try_from(prefill_request.prompt_len()).unwrap_or(u64::MAX);
    let request_id = prefill_request.request_id().to_string();
    let bound =
        prefill_worker.and_then(|worker| PrefillLegBound::at(worker, prompt_tokens, prefill_start));
    // Each leg owns its client handle (a cheap channel clone, the same one
    // batched dispatch takes per sub-request) so the leg still in flight when
    // its partner fails can be moved off the request path.
    let mut prefill_client = prefill_client.clone();
    let mut decode_client = decode_client.clone();
    for worker in [workers.prefill_worker(), workers.decode_worker()]
        .into_iter()
        .flatten()
    {
        Metrics::record_worker_request(worker.url(), worker.model_id());
    }
    // The leg is booked on its worker here, before its call is polled: a
    // servicer that sends its headers with the first message answers the
    // call only once the prefill is done, so a booking at the call's return
    // would time nothing (see `PrefillTicket::book`). The bound above was
    // read before the booking, so the leg's own prompt is not counted twice.
    let prefill_ticket =
        prefill_worker.and_then(|worker| PrefillTicket::book(worker, prompt_tokens));
    let prefill_dispatch: PdLegDispatch = Box::pin(dispatch_prefill_within(
        bound,
        Box::pin(async move { prefill_client.generate(prefill_request).await }),
        prefill_worker.cloned(),
        prefill_label,
        request_id.clone(),
        prompt_tokens,
    ));
    let decode_dispatch: PdLegDispatch =
        Box::pin(async move { decode_client.generate(decode_request).await });

    match dispatch_pd_legs(prefill_dispatch, decode_dispatch).await {
        PdDispatchOutcome::Both(prefill_result, decode_result) => {
            // Record circuit breaker outcomes (client errors don't count as
            // failures). A prefill dispatch that succeeded is sampled once
            // its first response is in (below), as the sequential path
            // samples it after the drain: a leg that was accepted into
            // silence is one failure of its worker, not a success and then a
            // failure, which would reset the breaker's streak every time.
            workers.record_prefill_decode_outcomes(
                ledger,
                prefill_result
                    .breaker_status()
                    .filter(|_| prefill_result.is_err()),
                decode_result.breaker_status(),
            );

            // Both legs are translated before either is propagated: a
            // decode-side failure is logged and counted even when the prefill
            // error is the one the client sees.
            let prefill = prefill_result
                .map_err(|e| pd_leg_error(PdLeg::Prefill, prefill_label, &e, workers));
            let decode =
                decode_result.map_err(|e| pd_leg_error(PdLeg::Decode, decode_label, &e, workers));
            let (prefill_stream, decode_stream) = match (prefill, decode) {
                (Ok(prefill), Ok(decode)) => (prefill, decode),
                // The surviving leg's stream drops here, which aborts its
                // room on the worker.
                (Err(response), _) | (Ok(_), Err(response)) => return Err(response),
            };

            // The prefill leg's prompt has been pending prefill on its worker
            // since its dispatch, until its first response (which teaches the
            // worker's rate); the stream takes the booking over. What is left
            // of the leg's bound is spent waiting for that response here,
            // before the response to the client begins, so a silent prefill
            // fails the attempt where the retry layer can still take the
            // request to another prefill worker.
            let mut prefill_stream = match prefill_worker {
                Some(worker) => {
                    prefill_stream.tracked_prefill_leg_booked(Arc::clone(worker), prefill_ticket)
                }
                None => prefill_stream,
            };
            match first_prefill_response_within(
                &mut prefill_stream,
                bound,
                prefill_worker,
                prefill_label,
                &request_id,
                prompt_tokens,
            )
            .await
            {
                Err(e) => {
                    // Dropping the prefill stream aborts its room. The
                    // decode's abort waits for its first response (30 s at
                    // most): a handoff the silent prefill may yet start must
                    // not be torn down mid-write.
                    drop(prefill_stream);
                    drop(decode_stream.defer_abort_until_first_item());
                    if let Some(status) = e.breaker_status() {
                        workers.record_outcome_prefill(ledger, status);
                    }
                    return Err(pd_leg_error(PdLeg::Prefill, prefill_label, &e, workers));
                }
                // The leg answered: the success its dispatch's sample waited
                // for. A leg whose first item is a failure is charged once,
                // where its reader meets it (the streaming layer's
                // accounting); a stream that ended with nothing on it is no
                // success of the leg either.
                Ok(FirstItem::Response) => workers.record_outcome_prefill(ledger, 200),
                Ok(FirstItem::Failure | FirstItem::End) => {}
            }

            // A client disconnect drops both leg streams, which normally fires an
            // immediate abort to each worker. The decode leg must not be aborted
            // while it is still receiving the KV handoff from prefill — tearing
            // the request down mid-transfer can crash or leak on the engine — so
            // its abort is deferred until the first decode response (the proof
            // the handoff completed). The prefill leg keeps the immediate abort:
            // if prefill is still running there is nothing to hand off yet, and
            // stopping it promptly frees capacity.
            let decode_stream = decode_stream.defer_abort_until_first_item();

            Ok(ExecutionResult::PrefillDecode {
                prefill: prefill_stream,
                decode: Box::new(decode_stream),
                prefill_guards: vec![Some(prefill_guard)],
                // Both legs were dispatched together, so the decode leg's
                // time to first token is the end-to-end one.
                pd_timing: PdTiming {
                    prefill_start,
                    decode_sent: prefill_start,
                    runtime,
                },
                // The streaming layer drains the live prefill stream itself.
                prefill_input_logprobs: None,
            })
        }
        PdDispatchOutcome::FailedFirst {
            leg,
            error,
            partner,
        } => {
            let status = error.breaker_status();
            // Only the leg that answered is recorded: the abandoned one has
            // said nothing about its worker yet, and `retire_pd_leg` records
            // it when it finally does.
            let (label, partner_label, partner_worker) = match leg {
                PdLeg::Prefill => {
                    if let Some(status) = status {
                        workers.record_outcome_prefill(ledger, status);
                    }
                    (prefill_label, decode_label, workers.decode_worker())
                }
                PdLeg::Decode => {
                    if let Some(status) = status {
                        workers.record_outcome_decode(ledger, status);
                    }
                    (decode_label, prefill_label, workers.prefill_worker())
                }
            };
            if let Some(worker) = partner_worker {
                // The partner of a prefill that fell silent is retired with
                // its abort deferred (that prefill may yet start the handoff
                // the decode is waiting for) and uncharged: a late failure
                // of a handoff that never came is not the decode's fault.
                let retirement = if is_prefill_silence(&error) {
                    Retirement::PartnerSilent
                } else {
                    Retirement::PartnerFailed
                };
                retire_pd_leg(
                    leg.partner(),
                    partner_label,
                    Arc::clone(worker),
                    partner,
                    retirement,
                );
            }
            Err(pd_leg_error(leg, label, &error, workers))
        }
    }
}

/// Dispatch both legs together and answer with the first failure.
///
/// The legs rendezvous on one bootstrap room, so a leg that cannot start
/// leaves its partner's room unreachable — and the engine holding that room
/// for the whole of its own deadline. Waiting for the second verdict before
/// reporting the first failure is what turned a 120 s prefill bootstrap
/// timeout into a 300 s decode transfer timeout for the client, with the
/// pair's slot held throughout. A leg that has already answered is never
/// discarded, so the success path is byte-for-byte what a join produced.
async fn dispatch_pd_legs(
    mut prefill: PdLegDispatch,
    mut decode: PdLegDispatch,
) -> PdDispatchOutcome {
    /// One turn of the dispatch loop. The `select!` handlers may only
    /// classify — moving a still-pending leg out happens after the macro's
    /// borrows end.
    enum LegStep {
        /// The prefill leg answered; a failure here was beaten to the finish
        /// by its partner, so both verdicts are in.
        Prefill(StreamResult),
        Decode(StreamResult),
        /// This leg failed while its partner was still dispatching.
        Alone(PdLeg, tonic::Status),
    }

    // At most one of these is `Some` at the top of the loop: the second
    // answer returns rather than being stored, which is also what keeps both
    // `select!` branches from being disabled at once.
    let mut prefill_result: Option<StreamResult> = None;
    let mut decode_result: Option<StreamResult> = None;

    loop {
        let step = tokio::select! {
            result = &mut prefill, if prefill_result.is_none() => match result {
                Err(error) if decode_result.is_none() => LegStep::Alone(PdLeg::Prefill, error),
                result => LegStep::Prefill(result),
            },
            result = &mut decode, if decode_result.is_none() => match result {
                Err(error) if prefill_result.is_none() => LegStep::Alone(PdLeg::Decode, error),
                result => LegStep::Decode(result),
            },
        };

        match step {
            LegStep::Alone(leg, error) => {
                let partner = match leg {
                    PdLeg::Prefill => decode,
                    PdLeg::Decode => prefill,
                };
                return PdDispatchOutcome::FailedFirst {
                    leg,
                    error,
                    partner,
                };
            }
            LegStep::Prefill(result) => match decode_result.take() {
                Some(decode) => return PdDispatchOutcome::Both(result, decode),
                None => prefill_result = Some(result),
            },
            LegStep::Decode(result) => match prefill_result.take() {
                Some(prefill) => return PdDispatchOutcome::Both(prefill, result),
                None => decode_result = Some(result),
            },
        }
    }
}

/// Log, count and translate one leg's dispatch failure into the client answer.
/// A failure that names the rendezvous counts under the rendezvous family too.
/// A prefill leg's silence verdict was logged and counted where its bound
/// passed, and answers under its own code (see [`prefill_silence_response`]).
fn pd_leg_error(
    leg: PdLeg,
    connection: &'static str,
    error: &tonic::Status,
    workers: &WorkerSelection,
) -> Response {
    if leg == PdLeg::Prefill && is_prefill_silence(error) {
        if let Some(prefill) = workers.prefill_worker() {
            return prefill_silence_response(error, prefill);
        }
    }
    UpstreamClass::account_dispatch_failure(leg.name(), connection, error);
    account_pd_leg_failure(leg.name(), error);
    start_failure_response(
        error,
        "execute_parallel_pd",
        leg.error_message(),
        leg.error_code(),
    )
}

/// Why a PD leg still dispatching is retired off the request path (see
/// [`retire_pd_leg`]), and what its late result is worth to its worker's
/// record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Retirement {
    /// Its partner failed to start, and the request with it. The leg's own
    /// late result is still this request's one word on its worker: sampled
    /// on the breaker and counted as a dispatch's is.
    PartnerFailed,
    /// Its partner, the prefill, fell silent: the request charged that
    /// prefill where its bound passed, and this leg's late result is not its
    /// worker's fault (the handoff it waited for never came), so it is
    /// neither sampled nor counted. Its abort is deferred: the prefill may
    /// yet start the handoff, which must not be torn down mid-write.
    PartnerSilent,
    /// Its own silence bound passed. The request charged its worker once,
    /// where the bound passed (the breaker sample through the ledger, the
    /// `timeout` worker error): its late result is neither sampled nor
    /// counted, whatever it is. A late stream is aborted, a late error noted.
    Silent,
}

impl Retirement {
    fn cause(self) -> &'static str {
        match self {
            Self::PartnerFailed => "its partner failed",
            Self::PartnerSilent => "its partner fell silent",
            Self::Silent => "its silence bound passed",
        }
    }

    fn defer_abort(self) -> bool {
        matches!(self, Self::PartnerSilent)
    }

    /// Whether the late result is this request's word on the worker.
    fn charges(self) -> bool {
        matches!(self, Self::PartnerFailed)
    }
}

/// Finish off a leg that was still dispatching when its request gave up on
/// it: its partner failed, or its own silence bound passed
/// ([`Retirement`]).
///
/// The request is already answered; what is left is the engine-side room,
/// which the abandoned leg will allocate the moment its dispatch lands and
/// then hold for its own deadline. Dropping the leg's stream as soon as it
/// arrives is the abort path the router already uses on a client disconnect,
/// and it is the only thing that releases that room early. The abort is
/// deferred only when a KV handoff may still be written into the leg: the
/// decode partner of a prefill that fell silent rather than failed; a peer
/// that failed to start writes nothing. The late result is sampled and
/// counted only when it is the request's one word on the worker; a leg
/// given up on for its silence charged its worker when the bound passed.
#[expect(
    clippy::disallowed_methods,
    reason = "the abandoned PD leg is retired off the request path; its dispatch is bounded by the engine's own deadline"
)]
fn retire_pd_leg(
    leg: PdLeg,
    connection: &'static str,
    worker: Arc<dyn Worker>,
    dispatch: PdLegDispatch,
    retirement: Retirement,
) {
    tokio::spawn(async move {
        let result = dispatch.await;
        if let Some(status) = result.breaker_status().filter(|_| retirement.charges()) {
            worker.record_outcome(status);
        }
        let cause = retirement.cause();
        match result {
            Ok(stream) => {
                debug!(
                    leg = leg.name(),
                    worker = worker.url(),
                    cause,
                    "Aborting the room of an abandoned PD leg"
                );
                if retirement.defer_abort() {
                    drop(stream.defer_abort_until_first_item());
                } else {
                    drop(stream);
                }
            }
            Err(error) if retirement.charges() => {
                UpstreamClass::account_dispatch_failure(leg.name(), connection, &error);
                error!(
                    function = "retire_pd_leg",
                    leg = leg.name(),
                    worker = worker.url(),
                    cause,
                    error = %error,
                    "Abandoned PD leg failed after its request had given up on it"
                );
            }
            Err(error) => {
                debug!(
                    leg = leg.name(),
                    worker = worker.url(),
                    cause,
                    error = %error,
                    "Abandoned PD leg failed late; its request had charged what it gave up on"
                );
            }
        }
    });
}

/// The message a prefill leg's silence verdict starts with (see
/// [`prefill_leg_silent`]); it tells the verdict from an engine's own
/// `DEADLINE_EXCEEDED`.
const PREFILL_SILENCE_MESSAGE: &str = "prefill leg produced nothing within";

/// The bound on one prefill leg's silence, counted from its dispatch (see
/// [`liveness::prefill_leg_bound`]).
#[derive(Clone, Copy, Debug)]
struct PrefillLegBound {
    deadline: Instant,
    bound: Duration,
}

impl PrefillLegBound {
    /// The bound for a leg of `prompt_tokens` dispatched to `worker` at
    /// `dispatched`; `None` when the rule is off.
    fn at(worker: &Arc<dyn Worker>, prompt_tokens: u64, dispatched: Instant) -> Option<Self> {
        let bound = liveness::prefill_leg_bound(worker, prompt_tokens)?;
        // A bound past the clock's range never expires: the leg stays unbounded.
        let deadline = dispatched.checked_add(bound)?;
        Some(Self { deadline, bound })
    }
}

/// The verdict on a prefill leg that produced nothing within its bound, in
/// `phase` (`dispatch`: the call returned no stream; `first_response`: the
/// stream yielded nothing), while its worker answered no other leg within
/// the bound either (see [`liveness::prefill_leg_reprieve`]): one WARN
/// naming the worker, the request and how long ago the worker last
/// answered anything, the worker-error counter under the timeout class, and
/// the status the leg's failure is reported with. Nothing else bounds this
/// wait below the request's own timeout: the engine's health probes keep
/// passing, the queue timeout ended at worker selection, and the admission
/// gate only gates the dispatch. The attempt fails here; the retry layer
/// takes the request to another prefill worker when one is available (see
/// [`prefill_silence_response`]).
fn prefill_leg_silent(
    worker: &Arc<dyn Worker>,
    connection: &'static str,
    request_id: &str,
    prompt_tokens: u64,
    bound: Duration,
    phase: &'static str,
) -> tonic::Status {
    let worker_last_answer = worker
        .prefill_answered_ago()
        .map_or_else(|| "never".to_string(), |ago| format!("{ago:.1?} ago"));
    warn!(
        request_id = %request_id,
        worker = worker.url(),
        worker_type = metrics_labels::WORKER_PREFILL,
        phase,
        prompt_tokens,
        bound_ms = u64::try_from(bound.as_millis()).unwrap_or(u64::MAX),
        worker_last_answer = %worker_last_answer,
        "PD prefill leg produced nothing within its silence bound, and its worker answered no other leg within it; giving up on it"
    );
    Metrics::record_worker_error(
        metrics_labels::WORKER_PREFILL,
        connection,
        metrics_labels::ERROR_TIMEOUT,
    );
    tonic::Status::deadline_exceeded(format!(
        "{PREFILL_SILENCE_MESSAGE} {bound:.1?} of its dispatch ({phase})"
    ))
}

/// Whether `status` is a prefill leg's silence verdict.
fn is_prefill_silence(status: &tonic::Status) -> bool {
    status.code() == tonic::Code::DeadlineExceeded
        && status.message().starts_with(PREFILL_SILENCE_MESSAGE)
}

/// The client answer for a prefill leg's silence: a 504 under its own error
/// code, replayable on another prefill worker. The leg's worker is marked on
/// the response so the retry window closes on it alone (its decode partner
/// stays selectable) and the answer stands when no other prefill worker is
/// available (see `attempt_ledger::FailedLeg`).
fn prefill_silence_response(status: &tonic::Status, prefill: &Arc<dyn Worker>) -> Response {
    let mut response = error::gateway_timeout(
        "prefill_worker_silent",
        format!("Prefill worker silent: {}", status.message()),
    );
    mark_failed_leg(&mut response, prefill.as_ref());
    response
}

/// Await a prefill leg's dispatch within its bound. Past the deadline the
/// leg waits on while its worker keeps answering other legs (a bound past
/// the worker's latest answer, re-read each time: see
/// [`liveness::prefill_leg_reprieve`]); once the worker has answered
/// nothing for a bound, the still-pending call is retired off the request
/// path (its room is aborted when it lands) and the silence verdict takes
/// its place. Without a bound or a worker to charge, the call is awaited as
/// it is.
async fn dispatch_prefill_within(
    bound: Option<PrefillLegBound>,
    mut dispatch: PdLegDispatch,
    worker: Option<Arc<dyn Worker>>,
    connection: &'static str,
    request_id: String,
    prompt_tokens: u64,
) -> StreamResult {
    let (Some(bound), Some(worker)) = (bound, worker) else {
        return dispatch.await;
    };
    let mut deadline = bound.deadline;
    loop {
        if let Ok(result) = tokio::time::timeout_at(deadline.into(), &mut dispatch).await {
            return result;
        }
        let Some(more) = liveness::prefill_leg_reprieve(&worker, bound.bound) else {
            retire_pd_leg(
                PdLeg::Prefill,
                connection,
                Arc::clone(&worker),
                dispatch,
                Retirement::Silent,
            );
            return Err(prefill_leg_silent(
                &worker,
                connection,
                &request_id,
                prompt_tokens,
                bound.bound,
                "dispatch",
            ));
        };
        deadline = Instant::now() + more;
    }
}

/// Wait for a prefill leg's first item within what is left of its bound,
/// with the same reprieves as [`dispatch_prefill_within`], and say what it
/// was. The item stays on the stream for its reader; past the bound, with
/// the worker silent, the silence verdict (the caller drops the leg, which
/// aborts its room). Without a bound nothing is pulled ahead and the leg
/// counts as answered, as its dispatch did.
async fn first_prefill_response_within(
    stream: &mut ProtoStream,
    bound: Option<PrefillLegBound>,
    worker: Option<&Arc<dyn Worker>>,
    connection: &'static str,
    request_id: &str,
    prompt_tokens: u64,
) -> Result<FirstItem, tonic::Status> {
    let (Some(bound), Some(worker)) = (bound, worker) else {
        return Ok(FirstItem::Response);
    };
    let mut deadline = bound.deadline;
    loop {
        if let Ok(first) = stream.first_response_within(deadline).await {
            return Ok(first);
        }
        let Some(more) = liveness::prefill_leg_reprieve(worker, bound.bound) else {
            return Err(prefill_leg_silent(
                worker,
                connection,
                request_id,
                prompt_tokens,
                bound.bound,
                "first_response",
            ));
        };
        deadline = Instant::now() + more;
    }
}

/// The attempt's answer for a prefill leg that failed to start or fell
/// silent on the sequential path: the breaker sample, the worker-error
/// counter (the silence verdict counted itself) and the client response.
fn sequential_prefill_failure(
    e: &tonic::Status,
    workers: &WorkerSelection,
    ledger: &AttemptLedger,
    connection: &'static str,
) -> Response {
    if let Some(status) = e.breaker_status() {
        workers.record_outcome_prefill(ledger, status);
    }
    if is_prefill_silence(e) {
        if let Some(prefill) = workers.prefill_worker() {
            return prefill_silence_response(e, prefill);
        }
    }
    UpstreamClass::account_dispatch_failure(metrics_labels::WORKER_PREFILL, connection, e);
    account_pd_leg_failure(metrics_labels::WORKER_PREFILL, e);
    start_failure_response(
        e,
        "execute_sequential_pd",
        PdLeg::Prefill.error_message(),
        PdLeg::Prefill.error_code(),
    )
}

/// How the vLLM sequential-PD decode leg is built from the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequentialPdDecodeForm {
    /// Full payload: no KV handoff (n>1), decode recomputes the prompt
    /// locally and runs the vision encoder itself.
    Full,
    /// Pixels dropped, per-image identity and M-RoPE grid tensors kept:
    /// decode computes grid-aware positions against the KV handoff.
    IdentityOnly,
    /// Language-model-only decode worker: everything multimodal goes except
    /// the per-image content hashes (folded into cache_salt servicer-side),
    /// so the engine sees a pure-text TokensPrompt and never touches its
    /// zero-budget encoder cache.
    TextPlusHashes,
}

/// Pick the decode-leg form, failing fast when the selected decode worker is
/// language-model-only and the request is one that pairing cannot serve.
/// Worker selection keeps such a request off such workers while the model
/// pairs a decode worker with the vision encoder (`decode_needs_vision`), so
/// a refusal here means none is paired.
fn sequential_pd_decode_form(
    request: &ProtoGenerateRequest,
    decode_language_model_only: bool,
    relay_kv_params: bool,
) -> Result<SequentialPdDecodeForm, Response> {
    if !decode_language_model_only {
        return Ok(if relay_kv_params {
            SequentialPdDecodeForm::IdentityOnly
        } else {
            SequentialPdDecodeForm::Full
        });
    }
    if request.has_vllm_mrope_grids() {
        // Grid-dependent models (Qwen-VL family) derive decode-side positions
        // from the grid tensors: a language-model-only decode worker rejects
        // them and stripping them would mis-rotate every generated token, so
        // this pairing cannot serve the request.
        let mut response = error::bad_request(
            "pd_decode_language_model_only_mrope",
            "no decode worker with a vision encoder is paired for this model, and its \
             decode leg needs the M-RoPE grid tensors a decode worker running with \
             --language-model-only cannot take; add a decode worker with the vision \
             encoder enabled for this model"
                .to_string(),
        );
        mark_non_retryable(&mut response);
        return Err(response);
    }
    if request.has_vllm_media_refs() {
        // Media references reach each leg as unexpanded anchors the leg
        // expands itself; a language-model-only decode worker cannot.
        let mut response = error::bad_request(
            "pd_decode_language_model_only_media_refs",
            "no decode worker with a vision encoder is paired for this model, and a \
             decode worker running with --language-model-only cannot expand media \
             references; add such a decode worker or use router-side multimodal \
             processing (--mm-processing=router) for this pool"
                .to_string(),
        );
        mark_non_retryable(&mut response);
        return Err(response);
    }
    if !relay_kv_params && request.has_mm_inputs() {
        // n>1 has no KV handoff: decode recomputes the prompt locally, which
        // a pixel-less decode worker cannot do.
        let mut response = error::bad_request(
            "pd_decode_language_model_only_n_samples",
            "no decode worker with a vision encoder is paired for this model, and n>1 \
             parallel sampling of a multimodal prompt recomputes it on the decode \
             leg, which a decode worker running with --language-model-only cannot \
             do; add a decode worker with the vision encoder enabled"
                .to_string(),
        );
        mark_non_retryable(&mut response);
        return Err(response);
    }
    Ok(SequentialPdDecodeForm::TextPlusHashes)
}

/// Execute vLLM PD: send to prefill with max_tokens=1 first, wait for completion,
/// then send original request to decode.
///
/// For Mooncake: injects bootstrap_host/port from prefill worker metadata into
/// the decode request. For NIXL: tags the prefill request with do_remote_decode,
/// then relays the kv_transfer_params returned by the prefill engine to decode.
async fn execute_sequential_pd(
    mut proto_request: ProtoGenerateRequest,
    clients: &mut ClientSelection,
    workers: &WorkerSelection,
    model: &str,
    prefill_guard: PrefillLoadGuard,
    ledger: &AttemptLedger,
) -> Result<ExecutionResult, Response> {
    let runtime = workers
        .disaggregated_runtime_type()
        .map(|r| r.as_str())
        .unwrap_or("");
    let (prefill_client, decode_client) = clients.disaggregated_mut().ok_or_else(|| {
        error!(
            function = "execute_sequential_pd",
            "Expected disaggregated clients but got single"
        );
        error::internal_error(
            "expected_disaggregated_clients_got_single",
            "Expected disaggregated clients but got single",
        )
    })?;

    let mode = match workers.decode_worker() {
        // A MoRI-IO decode engine never recomputes the prompt, whatever the
        // prefill leg runs.
        Some(decode) if is_moriio_worker(decode.as_ref()) => KvConnectorMode::MoriIo,
        _ => workers
            .prefill_worker()
            .map(|w| connector_mode_for_worker(w.as_ref()))
            .unwrap_or(KvConnectorMode::Passthrough),
    };

    // Recorded on the success path (after decode established) so failed
    // requests don't pollute success metrics; captured here before use of mode.
    let kv_connector_label = mode.metrics_label();

    match &mode {
            KvConnectorMode::Mooncake {
                host,
                port,
                engine_id,
            } => debug!(
                bootstrap_host = %host,
                bootstrap_port = port,
                engine_id_known = engine_id.is_some(),
                "vLLM PD (Mooncake): will inject kv_transfer_params into decode request"
            ),
            KvConnectorMode::Nixl => debug!(
                "vLLM PD (NIXL): will tag prefill with do_remote_decode and relay returned kv_transfer_params to decode"
            ),
            // Relaying nothing would let the decode engine compute over KV that
            // never arrives, so refuse until this pipeline speaks the protocol.
            KvConnectorMode::MoriIo => {
                return Err(error::not_implemented(
                    "moriio_grpc_pd_unsupported",
                    "MoRIIOConnector PD is supported by the HTTP PD router only",
                ));
            }
            KvConnectorMode::Passthrough => {
                // Warn once: PD without a discovered connector usually means GetServerInfo
                // lacks kv fields or labels.kv_connector is missing in worker config
                static WARN_ONCE: std::sync::Once = std::sync::Once::new();
                WARN_ONCE.call_once(|| {
                    tracing::warn!(
                        "vLLM PD: no kv_connector detected on prefill worker; KV transfer params \
                         will only be relayed if the engine returns them"
                    );
                });
            }
        }

    // The KV handoff is single-consumer: with n>1 each fan-out child on decode
    // would pull, and the first completion frees the prefill blocks under its
    // siblings (same hazard for NIXL and Mooncake)
    let relay_kv_params = proto_request.sampling_n() <= 1;

    // Mooncake is push-based: the engine returns nothing, so the router mints
    // the transfer correlation id and synthesizes decode params from metadata
    let mooncake_transfer_id = match &mode {
        KvConnectorMode::Mooncake {
            engine_id: Some(_), ..
        } if relay_kv_params => Some(format!("xfer-{}", uuid::Uuid::now_v7())),
        _ => None,
    };

    // Decode normally reuses the request minus pixels: it receives KV via
    // the P/D transfer, and prefill reads and unlinks any /dev/shm
    // segments, so a reused ShmHandle would be unreadable. Same request_id
    // on both legs is load-bearing for NIXL P/D correlation on vLLM <
    // 0.13. The pixel-free leg is the clone, so pixel tensors are never
    // duplicated and die with the prefill send; the per-image mm identity
    // and grid tensors survive for decode-side hashing and positions.
    // Media references are resolved by the prefill leg and relayed to
    // decode as that same identity once prefill completes.
    // Without a KV handoff (n>1) decode recomputes the prompt locally and
    // must run the vision encoder, so that leg keeps the full multimodal
    // payload (SHM-backed tensors cannot serve both legs and fail loudly
    // on the decode read).
    //
    // A decode worker started with `--language-model-only` (the production
    // vLLM P/D shape) has no vision encoder and
    // an encoder-cache budget of 0, so even the identity payload fails to
    // schedule there. Its model info reports supports_vision=false; for such
    // a worker the decode leg is stripped down to what such an engine takes: the
    // prefill-expanded input_ids, the KV handoff, and the per-image content
    // hashes that the servicer folds into cache_salt so different images
    // cannot alias in the decode prefix cache.
    let decode_language_model_only = proto_request.is_vllm()
        && workers
            .decode_worker()
            .is_some_and(|worker| worker_language_model_only(worker.as_ref()));
    let decode_form =
        sequential_pd_decode_form(&proto_request, decode_language_model_only, relay_kv_params)?;
    // A stripped multimodal leg has no local-recompute fallback: the image
    // KV exists only on the prefill worker. Text requests strip to a no-op
    // and keep the fallback, so only mm requests need the handoff checked.
    let stripped_mm_decode = matches!(decode_form, SequentialPdDecodeForm::TextPlusHashes)
        && proto_request.has_mm_inputs();
    let mut decode_request = match decode_form {
        SequentialPdDecodeForm::Full => proto_request.clone(),
        SequentialPdDecodeForm::TextPlusHashes => proto_request.clone_without_mm(),
        SequentialPdDecodeForm::IdentityOnly => proto_request.clone_without_mm_pixels(),
    };
    // Sanitize prefill sampling (max_tokens=1, n=1), stream=false.
    let mut prefill_request = proto_request;
    prefill_request.sanitize_sampling_for_prefill(1);
    prefill_request.set_stream(false);
    if let Some(rank) = workers.prefill_worker().and_then(|w| w.dp_rank()) {
        prefill_request.set_data_parallel_rank(rank as i32);
    }
    if mode == KvConnectorMode::Nixl {
        if relay_kv_params {
            prefill_request.set_kv_transfer_params_json(NIXL_PREFILL_KV_PARAMS.to_string());
        } else {
            debug!(
                request_id = %prefill_request.request_id(),
                "vLLM PD (NIXL): n>1 request, skipping kv_transfer_params relay \
                 (decode recomputes the prompt locally)"
            );
        }
    }
    if let Some(ref transfer_id) = mooncake_transfer_id {
        prefill_request.set_kv_transfer_params_json(mooncake_prefill_params(transfer_id));
    }

    debug!(
        request_id = %prefill_request.request_id(),
        "vLLM PD: sending prefill request (max_tokens=1)"
    );

    // Send to prefill, wait for completion. The leg's silence is bounded
    // from its dispatch (see `prefill_leg_silent`): a call that returns no
    // stream (a servicer that holds its headers until the first message in
    // front of a frozen engine), or a stream with nothing on it, within the
    // bound fails the attempt here, where the retry layer can still take
    // the request to another prefill worker.
    let (prefill_label, decode_label) = pd_leg_labels(workers);
    let prefill_worker = workers.prefill_worker();
    let prompt_tokens = u64::try_from(prefill_request.prompt_len()).unwrap_or(u64::MAX);
    let request_id = prefill_request.request_id().to_string();
    let prefill_start = Instant::now();
    let bound =
        prefill_worker.and_then(|worker| PrefillLegBound::at(worker, prompt_tokens, prefill_start));
    if let Some(prefill) = prefill_worker {
        Metrics::record_worker_request(prefill.url(), prefill.model_id());
    }
    let mut prefill_client = prefill_client.clone();
    // Booked before the call is polled, for the servicers that send their
    // headers with the first message (see `PrefillTicket::book`); the bound
    // above was read before the booking.
    let prefill_ticket =
        prefill_worker.and_then(|worker| PrefillTicket::book(worker, prompt_tokens));
    let prefill_stream = dispatch_prefill_within(
        bound,
        Box::pin(async move { prefill_client.generate(prefill_request).await }),
        prefill_worker.cloned(),
        prefill_label,
        request_id.clone(),
        prompt_tokens,
    )
    .await
    .map_err(|e| sequential_prefill_failure(&e, workers, ledger, prefill_label))?;
    // The prompt has been pending prefill on its worker since the dispatch,
    // until the first response, which teaches the worker's rate; the stream
    // takes the booking over.
    let mut prefill_stream = match prefill_worker {
        Some(worker) => {
            prefill_stream.tracked_prefill_leg_booked(Arc::clone(worker), prefill_ticket)
        }
        None => prefill_stream,
    };
    if let Err(e) = first_prefill_response_within(
        &mut prefill_stream,
        bound,
        prefill_worker,
        prefill_label,
        &request_id,
        prompt_tokens,
    )
    .await
    {
        // Dropping the stream aborts the leg's room: nothing was handed off.
        drop(prefill_stream);
        return Err(sequential_prefill_failure(
            &e,
            workers,
            ledger,
            prefill_label,
        ));
    }

    // Drain prefill response, harvesting connector params, the processed
    // media identity, input logprobs and the prefix-cache hit from the
    // Complete frame
    let mut prefill_kv_params: Option<String> = None;
    let mut prefill_media_identity: Option<vllm::MediaIdentity> = None;
    let mut prefill_input_logprobs = None;
    let mut prefill_cached_tokens = PrefillCachedTokens::default();
    while let Some(result) = prefill_stream.next().await {
        match result {
            Ok(response) => {
                if let ProtoResponseVariant::Complete(complete) = response.into_response() {
                    prefill_cached_tokens.note(&complete);
                    if let Some(json) = complete.kv_transfer_params_json() {
                        prefill_kv_params = Some(json.to_owned());
                    }
                    if let Some(identity) = complete.media_identity() {
                        prefill_media_identity = Some(identity.clone());
                    }
                    if let Some(logprobs) = complete.input_logprobs() {
                        prefill_input_logprobs = Some(logprobs);
                    }
                }
            }
            Err(e) => {
                if let Some(status) = e.breaker_status() {
                    workers.record_outcome_prefill(ledger, status);
                }
                UpstreamClass::account_dispatch_failure(
                    metrics_labels::WORKER_PREFILL,
                    prefill_label,
                    &e,
                );
                account_pd_leg_failure(metrics_labels::WORKER_PREFILL, &e);
                error!(function = "execute_sequential_pd", error = %e, "Prefill stream error");
                return Err(e.to_http_error(
                    "prefill_stream_error",
                    format!("Prefill stream error: {}", e.message()),
                ));
            }
        }
    }
    prefill_stream.mark_completed();
    // Prefill is drained: its admission slot is free while decode runs.
    drop(prefill_guard);
    workers.record_outcome_prefill(ledger, 200);
    // The prefill phase ends at the drain; recorded below only once decode is
    // established.
    let prefill_done = Instant::now();

    debug!("vLLM PD: prefill completed, sending decode request");

    if let Some(rank) = workers.decode_worker().and_then(|w| w.dp_rank()) {
        decode_request.set_data_parallel_rank(rank as i32);
    }
    // The prefill request carried KV params, so the servicer was asked for
    // the media identity; and whether decode ends up holding a handoff.
    let identity_solicited =
        mooncake_transfer_id.is_some() || (mode == KvConnectorMode::Nixl && relay_kv_params);
    let mut handed_off = false;
    match (&mode, prefill_kv_params) {
        // Modern Mooncake: synthesized params under the minted transfer_id
        (
            KvConnectorMode::Mooncake {
                host,
                port,
                engine_id: Some(engine_id),
            },
            _,
        ) if mooncake_transfer_id.is_some() => {
            let transfer_id = mooncake_transfer_id.as_deref().unwrap_or_default();
            debug!(
                request_id = %decode_request.request_id(),
                transfer_id = %transfer_id,
                "vLLM PD (Mooncake): injecting minted kv_transfer_params into decode request"
            );
            decode_request.set_kv_transfer_params_json(mooncake_decode_params(
                transfer_id,
                engine_id,
                host,
                *port,
            ));
            handed_off = true;
        }
        // Legacy Mooncake (no engine_id discovered, or n>1): typed host/port injection
        (KvConnectorMode::Mooncake { host, port, .. }, _) => {
            debug!(
                remote_host = %host,
                remote_port = port,
                "vLLM PD: injecting kv_transfer_params into decode request"
            );
            decode_request.set_kv_transfer_params(host.clone(), *port);
            handed_off = true;
        }
        (KvConnectorMode::Nixl | KvConnectorMode::Passthrough, Some(json)) if relay_kv_params => {
            debug!(
                request_id = %decode_request.request_id(),
                params_len = json.len(),
                "vLLM PD: relaying prefill kv_transfer_params to decode request"
            );
            decode_request.set_kv_transfer_params_json(json);
            handed_off = true;
        }
        (KvConnectorMode::Nixl, None) if relay_kv_params => {
            Metrics::record_pd_kv_transfer_failure();
            tracing::warn!(
                request_id = %decode_request.request_id(),
                "vLLM PD (NIXL): prefill returned no kv_transfer_params; decode will \
                 recompute the prompt locally (outdated smg-grpc-servicer or missing \
                 kv-transfer-config?)"
            );
        }
        _ => {}
    }
    apply_prefill_media_identity(
        &mut decode_request,
        handed_off,
        relay_kv_params,
        identity_solicited,
        prefill_media_identity.as_ref(),
    );

    // A stripped multimodal leg cannot fall back to a local recompute: the
    // image KV exists only on the prefill worker, so without a handoff the
    // decode engine would recompute the prompt as pure text and answer
    // image-blind. Fail instead of dispatching.
    if stripped_mm_decode && !decode_request.has_kv_transfer_params() {
        error!(
            function = "execute_sequential_pd",
            request_id = %decode_request.request_id(),
            "stripped multimodal decode leg has no kv_transfer_params to pull from"
        );
        let mut response = error::bad_gateway(
            "pd_decode_missing_kv_transfer_params",
            "the prefill worker returned no kv_transfer_params for this multimodal \
             request, and the language-model-only decode worker cannot recompute the \
             prompt locally (outdated smg-grpc-servicer or missing kv-transfer-config \
             on the prefill worker?)"
                .to_string(),
        );
        mark_non_retryable(&mut response);
        return Err(response);
    }

    // Send request to decode
    if let Some(decode) = workers.decode_worker() {
        Metrics::record_worker_request(decode.url(), decode.model_id());
    }
    // The relay phase ends at the decode dispatch. Everything after this
    // instant, including the engines' KV transfer and the decode scheduling,
    // is the decode leg's time to first token; the instant is taken before the
    // call so a worker that holds its response headers until the first token
    // still lands in the decode phase.
    let decode_sent = Instant::now();
    let decode_stream = decode_client.generate(decode_request).await.map_err(|e| {
        if let Some(status) = e.breaker_status() {
            workers.record_outcome_decode(ledger, status);
        }
        UpstreamClass::account_dispatch_failure(metrics_labels::WORKER_DECODE, decode_label, &e);
        account_pd_leg_failure(metrics_labels::WORKER_DECODE, &e);
        start_failure_response(
            &e,
            "execute_sequential_pd",
            PdLeg::Decode.error_message(),
            PdLeg::Decode.error_code(),
        )
    })?;

    workers.record_outcome_decode(ledger, 200);
    // Decode established: record the success-only PD metrics here.
    Metrics::record_pd_kv_connector_mode(kv_connector_label);
    Metrics::record_pd_prefill_duration(
        metrics_labels::BACKEND_PD,
        model,
        runtime,
        prefill_done.duration_since(prefill_start),
    );
    Metrics::record_pd_kv_transfer_duration(
        metrics_labels::BACKEND_PD,
        model,
        runtime,
        decode_sent.duration_since(prefill_done),
    );

    // Prefill has completed and its KV blocks are held pending decode's
    // transfer; aborting decode mid-transfer on a client disconnect can
    // crash or leak on the engine side. Defer the abort until the first
    // decode response proves the handoff finished (see the parallel PD
    // path for the same invariant). The client's prompt-side usage is the
    // prefill leg's: the decode leg's responses carry the prefix-cache hit
    // the prefill leg reported, not the transfer the decode leg counts as
    // cached.
    let decode_stream = decode_stream
        .defer_abort_until_first_item()
        .decode_leg(prefill_cached_tokens);

    // Surface the PD shape (not `Single`) so the streaming layer anchors TTFT
    // at prefill dispatch. Without this, the decode stream is established only
    // after prefill completes, and — because the servicer flushes response
    // headers lazily with the first message — the first decode chunk is
    // already buffered when the stream task starts, so a task-local timer
    // measures ~0. The prefill stream is fully drained and its load guard
    // already released, so the prefill leg carries no live guard.
    Ok(ExecutionResult::PrefillDecode {
        prefill: prefill_stream,
        decode: Box::new(decode_stream),
        prefill_guards: vec![None],
        pd_timing: PdTiming {
            prefill_start,
            decode_sent,
            runtime,
        },
        prefill_input_logprobs,
    })
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use smg_grpc_client::{sglang_proto as sglang, tokenspeed_proto as ts, vllm_proto as vllm};

    use super::*;
    use crate::{
        routers::common::{attempt_ledger::FailedLeg, retry::is_retryable_response},
        worker::{
            BasicWorkerBuilder, CircuitBreakerConfig, ConnectionMode, RuntimeType, Worker,
            WorkerType,
        },
    };

    /// A decode leg retired because its prefill fell silent is not charged
    /// for failing late: the handoff it waited for never came, and the
    /// request charged the prefill where the bound passed. A partner that
    /// failed to start still charges, as before.
    #[tokio::test]
    async fn the_partner_of_a_silent_prefill_is_not_charged_for_its_late_failure() {
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:30001")
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc)
                .circuit_breaker_config(CircuitBreakerConfig {
                    failure_threshold: 1,
                    ..Default::default()
                })
                .build(),
        );
        let late: PdLegDispatch =
            Box::pin(async { Err(tonic::Status::unavailable("the handoff never came")) });
        retire_pd_leg(
            PdLeg::Decode,
            "grpc",
            Arc::clone(&decode),
            late,
            Retirement::PartnerSilent,
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            decode.circuit_breaker_can_execute(),
            "a late failure after the prefill fell silent is not the decode worker's fault"
        );
        assert!(Retirement::PartnerSilent.defer_abort());
        assert!(!Retirement::PartnerSilent.charges());
        assert!(Retirement::PartnerFailed.charges());
        assert!(!Retirement::PartnerFailed.defer_abort());
        assert!(!Retirement::Silent.charges());
    }

    fn tokenspeed_request(n: u32, seed: Option<u64>) -> ProtoGenerateRequest {
        ProtoGenerateRequest::TokenSpeed(Box::new(ts::GenerateRequest {
            request_id: "req".to_string(),
            sampling_params: Some(ts::SamplingParams {
                n,
                sampling_seed: seed,
                ..Default::default()
            }),
            ..Default::default()
        }))
    }

    fn tokenspeed_pair() -> WorkerSelection {
        let leg = |url: &str, worker_type: WorkerType| -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new(url)
                    .worker_type(worker_type)
                    .runtime_type(RuntimeType::TokenSpeed)
                    .connection_mode(ConnectionMode::Grpc)
                    .build(),
            )
        };
        WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: leg("grpc://prefill:30000", WorkerType::Prefill),
            decode: leg("grpc://decode:30000", WorkerType::Decode),
            runtime_type: RuntimeType::TokenSpeed,
        }
    }

    /// A leg whose dispatch fails with the engine's rendezvous verdict is
    /// counted under the rendezvous family, by leg and phase; any other
    /// dispatch failure of a leg is not.
    #[test]
    fn a_leg_that_fails_its_rendezvous_at_dispatch_is_counted() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let workers = tokenspeed_pair();
        metrics::with_local_recorder(&recorder, || {
            pd_leg_error(
                PdLeg::Prefill,
                metrics_labels::CONNECTION_GRPC,
                &tonic::Status::internal(
                    "Prefill bootstrap failed for request rank=0 req.rid='r1' \
                     req.bootstrap_room=41 with exception KVTransferError(bootstrap_room=41): \
                     Request 41 timed out after 120.0s in KVPoll.Bootstrapping",
                ),
                &workers,
            );
            pd_leg_error(
                PdLeg::Decode,
                metrics_labels::CONNECTION_GRPC,
                &tonic::Status::unavailable("connection reset"),
                &workers,
            );
        });
        let rendered = handle.render();
        let counted: Vec<&str> = rendered
            .lines()
            .filter(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
            .collect();
        assert_eq!(counted.len(), 1, "{rendered}");
        assert!(
            counted[0].contains(r#"leg="prefill""#)
                && counted[0].contains(r#"phase="bootstrap""#)
                && counted[0].ends_with(" 1"),
            "{rendered}"
        );
    }

    #[test]
    fn engine_rejections_at_start_are_4xx_under_the_same_error_code() {
        let rejected = start_failure_response(
            &tonic::Status::invalid_argument("Invalid grammar specification"),
            "execute_single",
            "Failed to start generation",
            "start_generation_failed",
        );
        assert_eq!(rejected.status(), http::StatusCode::BAD_REQUEST);
        assert_eq!(
            rejected
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .unwrap(),
            "start_generation_failed"
        );

        let failed = start_failure_response(
            &tonic::Status::internal("engine died"),
            "execute_single",
            "Failed to start generation",
            "start_generation_failed",
        );
        assert_eq!(failed.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            failed
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .unwrap(),
            "start_generation_failed"
        );
    }

    /// A leg start the worker's side cancelled (its connection closed under
    /// the call) is the worker's failure, not a rejection of the request: a
    /// 502 under the leg's code, which the retry layer replays on another
    /// worker, never a 400 the client would read as its own mistake.
    #[test]
    fn a_leg_start_the_worker_cancelled_is_a_retryable_502_not_a_rejection() {
        use crate::routers::common::retry::is_retryable_response;

        let cancelled = start_failure_response(
            &tonic::Status::cancelled("operation was canceled"),
            "execute_sequential_pd",
            PdLeg::Decode.error_message(),
            PdLeg::Decode.error_code(),
        );
        assert_eq!(cancelled.status(), http::StatusCode::BAD_GATEWAY);
        assert_eq!(
            cancelled
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .unwrap(),
            "decode_worker_failed_to_start"
        );
        assert!(is_retryable_response(&cancelled));
    }

    #[test]
    fn a_spent_sidecar_budget_is_not_retried_but_a_fast_sidecar_failure_is() {
        use crate::routers::common::retry::is_retryable_response;

        let start = |status: tonic::Status| {
            start_failure_response(
                &status,
                "execute_single",
                "Failed to start generation",
                "start_generation_failed",
            )
        };

        let timed_out = start(tonic::Status::unavailable(
            "sidecar_timeout: no result for job ef684cf9 within 300000 ms",
        ));
        assert_eq!(timed_out.status(), http::StatusCode::SERVICE_UNAVAILABLE);
        assert!(!is_retryable_response(&timed_out));

        for message in [
            "sidecar_overloaded: 8 jobs queued (cap 8)",
            "sidecar_unavailable: connection refused",
            "sidecar_protocol: undecodable result",
            "sidecar_push_failed: ConnectionResetError: reset",
            "worker is saturated",
        ] {
            let failed = start(tonic::Status::unavailable(message));
            assert_eq!(failed.status(), http::StatusCode::SERVICE_UNAVAILABLE);
            assert!(is_retryable_response(&failed), "{message}");
        }

        // The prefix is the worker's; the same words under another code mean something else.
        let rejected = start(tonic::Status::invalid_argument(
            "sidecar_timeout mentioned in a client error",
        ));
        assert_eq!(rejected.status(), http::StatusCode::BAD_REQUEST);
        assert!(!is_retryable_response(&rejected));
    }

    #[test]
    fn pd_fanout_width_applies_to_multi_sample_text_requests_on_parallel_pd() {
        let tokenspeed = PdProtocol::for_runtime(RuntimeType::TokenSpeed).unwrap();
        assert_eq!(
            pd_fanout_width(&tokenspeed_request(3, None), tokenspeed),
            Some(3)
        );
        assert_eq!(
            pd_fanout_width(&tokenspeed_request(1, None), tokenspeed),
            None
        );
        assert_eq!(
            pd_fanout_width(&tokenspeed_request(0, None), tokenspeed),
            None
        );

        let sglang = ProtoGenerateRequest::Sglang(Box::new(sglang::GenerateRequest {
            request_id: "req".to_string(),
            sampling_params: Some(sglang::SamplingParams {
                n: 2,
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert_eq!(
            pd_fanout_width(
                &sglang,
                PdProtocol::for_runtime(RuntimeType::Sglang).unwrap()
            ),
            Some(2)
        );

        // A multimodal payload cannot be handed to n prefills.
        let mut multimodal = tokenspeed_request(3, None);
        if let ProtoGenerateRequest::TokenSpeed(req) = &mut multimodal {
            req.mm_inputs = Some(ts::MultimodalInputs::default());
        }
        assert_eq!(pd_fanout_width(&multimodal, tokenspeed), None);

        // The sequential (vLLM) path relays one KV handoff and already
        // skips it for n>1; no fan-out there.
        let vllm = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            request_id: "req".to_string(),
            sampling_params: Some(vllm::SamplingParams {
                n: 3,
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert_eq!(
            pd_fanout_width(&vllm, PdProtocol::for_runtime(RuntimeType::Vllm).unwrap()),
            None
        );
    }

    #[test]
    fn fan_out_pd_request_gives_each_sample_its_own_id_seed_and_room() {
        let request = tokenspeed_request(3, Some(7));
        let mut next_room = 100;
        let subs = fan_out_pd_request(&request, 3, |sub| {
            sub.set_kv_bootstrap_info("prefill".to_string(), 8998, next_room);
            next_room += 1;
        });
        assert_eq!(subs.len(), 3);
        let mut rooms = Vec::new();
        for (i, sub) in subs.iter().enumerate() {
            assert_eq!(sub.request_id(), format!("req-{i}"));
            assert_eq!(sub.sampling_n(), 1);
            let ProtoGenerateRequest::TokenSpeed(req) = sub else {
                panic!("sub-request changed runtime");
            };
            let params = req.sampling_params.as_ref().unwrap();
            assert_eq!(params.sampling_seed, Some(7 + i as u64));
            rooms.push(req.kv_bootstrap_info.as_ref().unwrap().bootstrap_room);
        }
        assert_eq!(rooms, vec![100, 101, 102]);
        // The original still asks for its three samples.
        assert_eq!(request.sampling_n(), 3);
    }

    #[test]
    fn fan_out_pd_request_leaves_an_unset_seed_unset() {
        for sub in &fan_out_pd_request(&tokenspeed_request(2, None), 2, |_| {}) {
            let ProtoGenerateRequest::TokenSpeed(req) = sub else {
                panic!("sub-request changed runtime");
            };
            assert_eq!(req.sampling_params.as_ref().unwrap().sampling_seed, None);
        }
    }

    #[test]
    fn plan_sub_requests_counts_fanned_out_samples() {
        let workers = tokenspeed_pair();
        let fanned = ExecutionPlan::PrefillDecode(tokenspeed_request(4, None));
        assert_eq!(plan_sub_requests(&fanned, Some(&workers)), 4);
        let (ids, complete) = trace_engine_ids(&mut fanned.clone(), Some(&workers));
        let dispatched: Vec<_> = fan_out_pd_request(&tokenspeed_request(4, None), 4, |_| {})
            .iter()
            .map(|request| request.request_id().to_owned())
            .collect();
        assert_eq!(ids, dispatched);
        assert!(complete);
        let (ids, complete) = trace_engine_ids(
            &mut ExecutionPlan::PrefillDecode(tokenspeed_request(33, None)),
            Some(&workers),
        );
        assert_eq!(ids.len(), 32);
        assert!(!complete);
        // Without a disaggregated selection there is no PD protocol to fan out on.
        assert_eq!(plan_sub_requests(&fanned, None), 1);

        let plain = ExecutionPlan::PrefillDecode(tokenspeed_request(1, None));
        assert_eq!(plan_sub_requests(&plain, Some(&workers)), 1);

        let batch = ExecutionPlan::Batch {
            kind: ExecutionPlanKind::PrefillDecode,
            shared_request_id: "cmpl-1".to_string(),
            requests: vec![tokenspeed_request(2, None), tokenspeed_request(1, None)],
        };
        assert_eq!(plan_sub_requests(&batch, Some(&workers)), 3);
    }

    /// A leg that never answers — the decode leg stuck behind the engine's
    /// transfer deadline while the prefill leg has already given up.
    fn never_answers() -> PdLegDispatch {
        Box::pin(std::future::pending())
    }

    fn fails_now(message: &'static str) -> PdLegDispatch {
        Box::pin(async move { Err(tonic::Status::deadline_exceeded(message)) })
    }

    fn answers_after(delay: Duration) -> PdLegDispatch {
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            Err(tonic::Status::unavailable("late leg"))
        })
    }

    /// A prefill that times out must be reported without waiting for the
    /// decode leg, and the decode's dispatch must come back so its room can
    /// be released instead of being left to the engine's own deadline.
    #[tokio::test(start_paused = true)]
    async fn a_failed_prefill_answers_without_waiting_for_the_decode_leg() {
        let started = tokio::time::Instant::now();
        let outcome = dispatch_pd_legs(fails_now("bootstrap timeout"), never_answers()).await;

        assert_eq!(
            started.elapsed(),
            Duration::ZERO,
            "the prefill error must not wait on the decode leg"
        );
        match outcome {
            PdDispatchOutcome::FailedFirst { leg, error, .. } => {
                assert_eq!(leg, PdLeg::Prefill);
                assert_eq!(error.message(), "bootstrap timeout");
            }
            PdDispatchOutcome::Both(..) => panic!("expected a fail-fast outcome"),
        }
    }

    /// The mirror case: a decode leg that cannot start is answered without
    /// waiting out the prefill's bootstrap deadline.
    #[tokio::test(start_paused = true)]
    async fn a_failed_decode_answers_without_waiting_for_the_prefill_leg() {
        let outcome = dispatch_pd_legs(
            answers_after(Duration::from_secs(120)),
            fails_now("decode refused"),
        )
        .await;

        match outcome {
            PdDispatchOutcome::FailedFirst { leg, error, .. } => {
                assert_eq!(leg, PdLeg::Decode);
                assert_eq!(error.message(), "decode refused");
            }
            PdDispatchOutcome::Both(..) => panic!("expected a fail-fast outcome"),
        }
    }

    /// The failure a dispatch ended with (a stream has no `Debug`).
    fn failure(result: StreamResult) -> tonic::Status {
        match result {
            Err(status) => status,
            Ok(_) => panic!("expected the dispatch to fail"),
        }
    }

    fn three_second_bound() -> PrefillLegBound {
        PrefillLegBound {
            deadline: Instant::now() + Duration::from_secs(3),
            bound: Duration::from_secs(3),
        }
    }

    /// A prefill call that has returned no stream by its bound is given up on
    /// at the bound, not at the request's timeout: the verdict names the
    /// silence, and the attempt answers a 504 under its own code that the
    /// retry layer may take to another prefill worker, this one excluded.
    #[tokio::test(start_paused = true)]
    async fn a_prefill_dispatch_past_its_bound_is_given_up_on_at_the_bound() {
        let workers = tokenspeed_pair();
        let prefill = Arc::clone(workers.prefill_worker().expect("a pair"));
        let started = tokio::time::Instant::now();
        let result = dispatch_prefill_within(
            Some(three_second_bound()),
            never_answers(),
            Some(Arc::clone(&prefill)),
            metrics_labels::CONNECTION_GRPC,
            "chatcmpl-silent".to_string(),
            256,
        )
        .await;
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_secs(3) && elapsed < Duration::from_secs(4),
            "given up on at the bound, not later: {elapsed:?}"
        );
        let status = failure(result);
        assert_eq!(status.code(), tonic::Code::DeadlineExceeded);
        assert!(is_prefill_silence(&status), "{status:?}");

        let response = pd_leg_error(
            PdLeg::Prefill,
            metrics_labels::CONNECTION_GRPC,
            &status,
            &workers,
        );
        assert_eq!(response.status(), http::StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            response
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .and_then(|value| value.to_str().ok()),
            Some("prefill_worker_silent")
        );
        assert!(
            response.extensions().get::<FailedLeg>().is_some(),
            "the retry window closes on the prefill worker alone"
        );
        assert!(
            is_retryable_response(&response),
            "another prefill worker may take the request"
        );
    }

    /// A prefill that answers, or fails on its own, within its bound passes
    /// through as it is; an engine's own deadline is not the verdict; without
    /// a bound the call is awaited as before.
    #[tokio::test(start_paused = true)]
    async fn a_prefill_that_answers_within_its_bound_passes_through() {
        let workers = tokenspeed_pair();
        let prefill = Arc::clone(workers.prefill_worker().expect("a pair"));
        let status = failure(
            dispatch_prefill_within(
                Some(three_second_bound()),
                answers_after(Duration::from_secs(1)),
                Some(prefill),
                metrics_labels::CONNECTION_GRPC,
                "chatcmpl-late".to_string(),
                256,
            )
            .await,
        );
        assert_eq!(status.message(), "late leg");
        assert!(!is_prefill_silence(&status));
        assert!(!is_prefill_silence(&tonic::Status::deadline_exceeded(
            "bootstrap timeout"
        )));
        let status = failure(
            dispatch_prefill_within(
                None,
                fails_now("refused"),
                None,
                metrics_labels::CONNECTION_GRPC,
                "chatcmpl-unbounded".to_string(),
                256,
            )
            .await,
        );
        assert_eq!(status.message(), "refused");
    }

    /// A prefill stream that answers (here: ends) within its bound is left
    /// for its reader as it was, and what came is named: the end is not a
    /// response the breaker would sample a success for.
    #[tokio::test]
    async fn a_prefill_stream_that_answers_within_its_bound_is_left_for_its_reader() {
        let workers = tokenspeed_pair();
        let prefill = Arc::clone(workers.prefill_worker().expect("a pair"));
        let mut stream = ProtoStream::Fanout(FanoutStream::new(Vec::new()))
            .tracked_prefill_leg(Arc::clone(&prefill), 256);
        let first = first_prefill_response_within(
            &mut stream,
            Some(three_second_bound()),
            Some(&prefill),
            metrics_labels::CONNECTION_GRPC,
            "chatcmpl-ended",
            256,
        )
        .await
        .expect("ended within the bound");
        assert_eq!(first, FirstItem::End);
        assert!(stream.next().await.is_none(), "the end is replayed");
    }

    /// A prefill call past its bound waits on while its worker answers other
    /// legs: the verdict is on the worker, not the leg. The worker answers
    /// another leg half a second in, so the call is given up on a bound
    /// after that answer, not at its own bound. (Real time: the worker's
    /// stamps are on the gateway clock, which a paused runtime does not
    /// move.)
    #[tokio::test]
    async fn a_prefill_dispatch_past_its_bound_waits_on_while_its_worker_answers() {
        let workers = tokenspeed_pair();
        let prefill = Arc::clone(workers.prefill_worker().expect("a pair"));
        let bound = Duration::from_secs(1);
        let answered_at = Duration::from_millis(500);
        let other_leg = async {
            tokio::time::sleep(answered_at).await;
            prefill.note_prefill_started(256, true);
            prefill.note_prefill_ended(256, true, Some(0));
        };
        let started = Instant::now();
        let (result, ()) = tokio::join!(
            dispatch_prefill_within(
                Some(PrefillLegBound {
                    deadline: started + bound,
                    bound,
                }),
                never_answers(),
                Some(Arc::clone(&prefill)),
                metrics_labels::CONNECTION_GRPC,
                "chatcmpl-queued".to_string(),
                256,
            ),
            other_leg
        );
        let elapsed = started.elapsed();
        let status = failure(result);
        assert!(is_prefill_silence(&status), "{status:?}");
        assert!(
            elapsed >= answered_at + bound && elapsed < answered_at + 3 * bound,
            "given up on a bound after the worker's latest answer, not at its own bound: {elapsed:?}"
        );
    }

    /// Admission and the load guards read the same count, so a batched plan
    /// claims one room per prompt instead of walking past a gate that only
    /// asked about one.
    #[test]
    fn plan_sub_requests_counts_every_batched_prompt() {
        let single = ExecutionPlan::PrefillDecode(ProtoGenerateRequest::Vllm(Box::default()));
        assert_eq!(plan_sub_requests(&single, None), 1);

        let batch = ExecutionPlan::Batch {
            kind: ExecutionPlanKind::PrefillDecode,
            shared_request_id: "cmpl-1".to_string(),
            requests: (0..4)
                .map(|_| ProtoGenerateRequest::Vllm(Box::default()))
                .collect(),
        };
        assert_eq!(plan_sub_requests(&batch, None), 4);
    }

    #[test]
    fn pd_leg_labels_reflect_each_legs_transport() {
        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("ipc:///tmp/smg-test-prefill")
                .worker_type(WorkerType::Prefill)
                .connection_mode(ConnectionMode::Zmq)
                .build(),
        );
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:30000")
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        );
        let selection = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill,
            decode,
            runtime_type: RuntimeType::TokenSpeed,
        };
        assert_eq!(
            pd_leg_labels(&selection),
            (
                metrics_labels::CONNECTION_ZMQ,
                metrics_labels::CONNECTION_GRPC
            ),
            "each PD leg must carry its own transport label"
        );
    }

    #[test]
    fn mooncake_prefill_params_carry_transfer_id() {
        let value: serde_json::Value =
            serde_json::from_str(&mooncake_prefill_params("xfer-abc")).unwrap();
        assert_eq!(value["do_remote_decode"], true);
        assert_eq!(value["do_remote_prefill"], false);
        assert_eq!(value["transfer_id"], "xfer-abc");
        assert_eq!(value.as_object().unwrap().len(), 3);
    }

    #[test]
    fn mooncake_decode_params_synthesize_full_handoff() {
        let value: serde_json::Value = serde_json::from_str(&mooncake_decode_params(
            "xfer-abc", "engine-1", "10.0.0.1", 8998,
        ))
        .unwrap();
        assert_eq!(value["do_remote_decode"], false);
        assert_eq!(value["do_remote_prefill"], true);
        assert_eq!(value["transfer_id"], "xfer-abc");
        assert_eq!(value["remote_engine_id"], "engine-1");
        assert_eq!(value["remote_bootstrap_addr"], "http://10.0.0.1:8998");
        assert_eq!(value.as_object().unwrap().len(), 5);
    }

    #[test]
    fn nixl_prefill_kv_params_is_valid_json() {
        let value: serde_json::Value = serde_json::from_str(NIXL_PREFILL_KV_PARAMS).unwrap();
        assert_eq!(value["do_remote_decode"], true);
        assert_eq!(value["do_remote_prefill"], false);
        assert_eq!(value.as_object().unwrap().len(), 2);
    }

    #[test]
    fn clone_without_mm_pixels_keeps_pixels_on_the_original_only() {
        let mut request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            request_id: "pd-1".to_string(),
            mm_inputs: Some(vllm::MultimodalInputs::default()),
            ..Default::default()
        }));
        let clone = request.clone_without_mm_pixels();
        let ProtoGenerateRequest::Vllm(original) = request else {
            panic!("expected vLLM request");
        };
        let ProtoGenerateRequest::Vllm(decode) = clone else {
            panic!("expected vLLM clone");
        };
        assert!(original.mm_inputs.is_some(), "prefill leg keeps pixels");
        assert!(
            decode.mm_inputs.is_none(),
            "hash-less mm payload is dropped whole on the decode leg"
        );
        assert_eq!(decode.request_id, "pd-1");
    }

    #[test]
    fn clone_without_mm_pixels_keeps_vllm_media_refs() {
        // The clone keeps the references; on the sequential path they are
        // replaced by the prefill leg's identity before decode is sent.
        let refs = vllm::MediaRefs {
            items: vec![vllm::MediaRef {
                modality: smg_grpc_client::common_proto::Modality::Image as i32,
                url: "https://a/1.png".to_string(),
            }],
        };
        let mut request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            request_id: "pd-refs".to_string(),
            ..Default::default()
        }));
        request
            .set_vllm_media_refs(refs.clone())
            .expect("vLLM request accepts media refs");
        assert!(request.has_vllm_media_refs());
        let clone = request.clone_without_mm_pixels();
        let ProtoGenerateRequest::Vllm(decode) = clone else {
            panic!("expected vLLM clone");
        };
        assert_eq!(decode.media_refs, Some(refs));
        assert!(decode.mm_inputs.is_none());
    }

    fn media_refs_request(id: &str, n: u32) -> ProtoGenerateRequest {
        let mut request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            request_id: id.to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: "describe <|image|>".to_string(),
                    input_ids: vec![7, 8, 9],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                n,
                ..Default::default()
            }),
            ..Default::default()
        }));
        request
            .set_vllm_media_refs(vllm::MediaRefs {
                items: vec![vllm::MediaRef {
                    modality: smg_grpc_client::common_proto::Modality::Image as i32,
                    url: "https://a/1.png".to_string(),
                }],
            })
            .expect("vLLM request accepts media refs");
        request
    }

    fn identity() -> vllm::MediaIdentity {
        vllm::MediaIdentity {
            prompt_token_ids: vec![7, 8, 100, 100, 100, 9],
            mm_inputs: Some(vllm::MultimodalInputs {
                mm_hashes: vec!["h1".to_string(), "h2".to_string()],
                mm_placeholders: vec![
                    vllm::PlaceholderRange {
                        offset: 2,
                        length: 2,
                    },
                    vllm::PlaceholderRange {
                        offset: 4,
                        length: 1,
                    },
                ],
                model_specific_tensors: std::collections::HashMap::from([(
                    "image_grid_thw".to_string(),
                    vllm::TensorData {
                        shape: vec![2, 3],
                        dtype: "int64".to_string(),
                        payload: Some(vllm::tensor_data::Payload::Inline(vec![0; 48])),
                    },
                )]),
                batched_keys: vec!["image_grid_thw".to_string()],
                modality: smg_grpc_client::common_proto::Modality::Image as i32,
                ..Default::default()
            }),
            extra_mm_inputs: vec![],
        }
    }

    /// A media_refs request under parallel PD dispatch would otherwise become
    /// n pairs, each processing the media on both legs.
    #[test]
    fn media_refs_request_never_fans_out() {
        let sglang = PdProtocol::for_runtime(RuntimeType::Sglang).unwrap();
        assert_eq!(sglang.dispatch, PdDispatch::Parallel);
        assert_eq!(pd_fanout_width(&media_refs_request("fan", 3), sglang), None);
    }

    /// Without a KV handoff decode recomputes the prompt locally, so it needs
    /// the media itself and the identity must not replace its references:
    /// n>1 never relays, and a NIXL prefill that returned no params does not
    /// hand off either, even when it did return an identity.
    #[test]
    fn n_greater_than_one_keeps_media_refs_on_decode() {
        let mut decode = media_refs_request("n2", 2).clone_without_mm_pixels();
        apply_prefill_media_identity(&mut decode, false, false, false, Some(&identity()));
        assert!(decode.has_vllm_media_refs());
        let ProtoGenerateRequest::Vllm(request) = &decode else {
            panic!("expected vLLM request");
        };
        assert!(request.mm_inputs.is_none());
    }

    /// Legacy Mooncake injects its host and port into every decode leg, n>1
    /// included, so `handed_off` alone does not say that decode will pull
    /// its prompt KV from prefill: an n>1 leg recomputes locally and keeps
    /// its references whatever the prefill returned.
    #[test]
    fn n_greater_than_one_keeps_media_refs_despite_a_legacy_mooncake_handoff() {
        let mut decode = media_refs_request("n2-mooncake", 2).clone_without_mm_pixels();
        apply_prefill_media_identity(&mut decode, true, false, false, Some(&identity()));
        assert!(decode.has_vllm_media_refs());
        let ProtoGenerateRequest::Vllm(request) = &decode else {
            panic!("expected vLLM request");
        };
        assert!(request.mm_inputs.is_none());
    }

    #[test]
    fn an_identity_without_a_kv_handoff_keeps_the_references() {
        let mut decode = media_refs_request("no-kv", 1).clone_without_mm_pixels();
        apply_prefill_media_identity(&mut decode, false, true, true, Some(&identity()));
        assert!(decode.has_vllm_media_refs());
        let ProtoGenerateRequest::Vllm(request) = &decode else {
            panic!("expected vLLM request");
        };
        assert!(request.mm_inputs.is_none());
    }

    #[test]
    fn decode_leg_from_media_identity_has_no_refs_and_no_pixels() {
        let mut decode = media_refs_request("relay", 1).clone_without_mm_pixels();
        apply_prefill_media_identity(&mut decode, true, true, true, Some(&identity()));
        assert!(!decode.has_vllm_media_refs());
        let ProtoGenerateRequest::Vllm(request) = &decode else {
            panic!("expected vLLM request");
        };
        assert!(request.media_refs.is_none());
        let mm = request.mm_inputs.as_ref().expect("identity mm_inputs");
        assert!(mm.pixel_values.is_none());
        assert_eq!(mm.mm_hashes.len(), 2);
        assert_eq!(mm.mm_placeholders.len(), 2);
        assert!(mm.model_specific_tensors.contains_key("image_grid_thw"));
        let Some(vllm::generate_request::Input::Tokenized(tokenized)) = &request.input else {
            panic!("expected tokenized input");
        };
        assert_eq!(tokenized.input_ids, vec![7, 8, 100, 100, 100, 9]);
        assert_eq!(tokenized.original_text, "describe <|image|>");
    }

    /// An identity the leg cannot take, empty ids or a leg that is not
    /// tokenized, leaves the references in place: better a reprocessed
    /// decode than an empty prompt.
    #[test]
    fn an_unusable_identity_leaves_the_decode_leg_untouched() {
        let mut empty = identity();
        empty.prompt_token_ids.clear();
        let mut decode = media_refs_request("empty", 1).clone_without_mm_pixels();
        apply_prefill_media_identity(&mut decode, true, true, true, Some(&empty));
        assert!(decode.has_vllm_media_refs());
        let ProtoGenerateRequest::Vllm(request) = &decode else {
            panic!("expected vLLM request");
        };
        assert!(request.mm_inputs.is_none());

        let mut decode = media_refs_request("text", 1).clone_without_mm_pixels();
        if let ProtoGenerateRequest::Vllm(request) = &mut decode {
            request.input = Some(vllm::generate_request::Input::Text(
                "describe <|image|>".to_string(),
            ));
        }
        apply_prefill_media_identity(&mut decode, true, true, true, Some(&identity()));
        assert!(decode.has_vllm_media_refs());
        let ProtoGenerateRequest::Vllm(request) = &decode else {
            panic!("expected vLLM request");
        };
        assert!(request.mm_inputs.is_none());
        assert!(matches!(
            request.input,
            Some(vllm::generate_request::Input::Text(_))
        ));
    }

    /// An older servicer returns no identity: the decode leg keeps its
    /// references and reprocesses, as before; the same when no identity was
    /// asked for (a prefill request without KV params).
    #[test]
    fn missing_identity_leaves_the_decode_leg_untouched() {
        for solicited in [true, false] {
            let mut decode = media_refs_request("old", 1).clone_without_mm_pixels();
            apply_prefill_media_identity(&mut decode, true, true, solicited, None);
            assert!(decode.has_vllm_media_refs(), "solicited={solicited}");
        }
    }

    #[test]
    fn clone_without_mm_pixels_keeps_vllm_identity_and_grid_tensors() {
        // The decode leg drops the pixel tensors but keeps the per-image
        // identity and the inline M-RoPE grid tensors.
        let grid = vllm::TensorData {
            shape: vec![1, 3],
            dtype: "int64".to_string(),
            payload: Some(vllm::tensor_data::Payload::Inline(vec![0; 24])),
        };
        let mut request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            request_id: "pd-2".to_string(),
            mm_inputs: Some(vllm::MultimodalInputs {
                pixel_values: Some(vllm::TensorData::default()),
                model_specific_tensors: std::collections::HashMap::from([
                    ("image_grid_thw".to_string(), grid.clone()),
                    // Payload-less grids and non-grid tensors are dropped.
                    ("video_grid_thw".to_string(), vllm::TensorData::default()),
                    ("aspect_ratios".to_string(), grid.clone()),
                    // Flat-classified grid keys keep their sizes tensor.
                    ("second_per_grid_ts".to_string(), grid.clone()),
                    ("ts_sizes".to_string(), grid.clone()),
                    // The Omni family's and the router's own spelling of the
                    // video timing.
                    ("video_second_per_grid".to_string(), grid),
                ]),
                flat_keys: std::collections::HashMap::from([(
                    "second_per_grid_ts".to_string(),
                    "ts_sizes".to_string(),
                )]),
                im_token_id: Some(151_655),
                mm_placeholders: vec![vllm::PlaceholderRange {
                    offset: 3,
                    length: 4,
                }],
                mm_hashes: vec!["h1".to_string()],
                batched_keys: vec![
                    "pixel_values".to_string(),
                    "image_grid_thw".to_string(),
                    "aspect_ratios".to_string(),
                ],
                keep_on_cpu_keys: vec!["image_grid_thw".to_string()],
                ..Default::default()
            }),
            ..Default::default()
        }));
        let clone = request.clone_without_mm_pixels();
        let ProtoGenerateRequest::Vllm(original) = request else {
            panic!("expected vLLM request");
        };
        let ProtoGenerateRequest::Vllm(decode) = clone else {
            panic!("expected vLLM clone");
        };
        let original_mm = original.mm_inputs.expect("prefill leg keeps mm_inputs");
        assert!(
            original_mm.pixel_values.is_some(),
            "prefill leg keeps pixels"
        );
        assert_eq!(original_mm.model_specific_tensors.len(), 6);
        assert_eq!(original_mm.batched_keys.len(), 3);
        let decode_mm = decode.mm_inputs.expect("decode leg keeps identity");
        assert!(
            decode_mm.pixel_values.is_none(),
            "decode leg never carries pixels"
        );
        let mut decode_keys: Vec<_> = decode_mm.model_specific_tensors.keys().collect();
        decode_keys.sort();
        assert_eq!(
            decode_keys,
            vec![
                "image_grid_thw",
                "second_per_grid_ts",
                "ts_sizes",
                "video_second_per_grid"
            ]
        );
        assert_eq!(decode_mm.batched_keys, vec!["image_grid_thw".to_string()]);
        assert_eq!(
            decode_mm.keep_on_cpu_keys,
            vec!["image_grid_thw".to_string()]
        );
        assert_eq!(
            decode_mm.flat_keys,
            std::collections::HashMap::from([(
                "second_per_grid_ts".to_string(),
                "ts_sizes".to_string()
            )])
        );
        assert_eq!(decode_mm.mm_hashes, vec!["h1".to_string()]);
        assert_eq!(decode_mm.mm_placeholders.len(), 1);
        assert_eq!(decode_mm.im_token_id, Some(151_655));
    }

    #[test]
    fn clone_without_mm_pixels_keeps_tokenspeed_item_metadata() {
        let item = ts::MultimodalItem {
            encoder_input: Some(ts::TensorData::default()),
            placeholder_token_id: Some(7),
            ..Default::default()
        };
        let mut request = ProtoGenerateRequest::TokenSpeed(Box::new(ts::GenerateRequest {
            mm_inputs: Some(ts::MultimodalInputs { items: vec![item] }),
            ..Default::default()
        }));
        let clone = request.clone_without_mm_pixels();
        let ProtoGenerateRequest::TokenSpeed(original) = request else {
            panic!("expected TokenSpeed request");
        };
        let ProtoGenerateRequest::TokenSpeed(decode) = clone else {
            panic!("expected TokenSpeed clone");
        };
        let original_item = &original.mm_inputs.as_ref().unwrap().items[0];
        assert!(
            original_item.encoder_input.is_some(),
            "prefill leg keeps encoder input"
        );
        let decode_item = &decode.mm_inputs.as_ref().unwrap().items[0];
        assert!(
            decode_item.encoder_input.is_none(),
            "decode leg drops encoder input"
        );
        assert_eq!(
            decode_item.placeholder_token_id,
            Some(7),
            "decode leg keeps per-item metadata"
        );
    }

    /// A full vLLM multimodal request: expanded token ids, pixels, grids,
    /// an extra video batch, media refs and a relayed KV handoff.
    fn vllm_pd_mm_request() -> ProtoGenerateRequest {
        let grid = vllm::TensorData {
            shape: vec![1, 3],
            dtype: "int64".to_string(),
            payload: Some(vllm::tensor_data::Payload::Inline(vec![0; 24])),
        };
        let mut request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            request_id: "pd-lmo".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: String::new(),
                    input_ids: vec![1, 2, 151_655, 151_655, 3],
                },
            )),
            mm_inputs: Some(vllm::MultimodalInputs {
                pixel_values: Some(vllm::TensorData::default()),
                model_specific_tensors: std::collections::HashMap::from([(
                    "image_grid_thw".to_string(),
                    grid,
                )]),
                im_token_id: Some(151_655),
                mm_placeholders: vec![vllm::PlaceholderRange {
                    offset: 2,
                    length: 2,
                }],
                mm_hashes: vec!["img-hash".to_string()],
                ..Default::default()
            }),
            extra_mm_inputs: vec![vllm::MultimodalInputs {
                mm_hashes: vec!["vid-hash".to_string()],
                modality: smg_grpc_client::common_proto::Modality::Video as i32,
                ..Default::default()
            }],
            ..Default::default()
        }));
        request.set_kv_transfer_params_json("{\"do_remote_prefill\":true}".to_string());
        request
            .set_vllm_media_refs(vllm::MediaRefs {
                items: vec![vllm::MediaRef {
                    modality: smg_grpc_client::common_proto::Modality::Image as i32,
                    url: "https://a/1.png".to_string(),
                }],
            })
            .expect("vLLM request accepts media refs");
        request
    }

    #[test]
    fn clone_without_mm_keeps_only_hashes_ids_and_kv_params() {
        let mut request = vllm_pd_mm_request();
        let clone = request.clone_without_mm();
        let ProtoGenerateRequest::Vllm(original) = request else {
            panic!("expected vLLM request");
        };
        let ProtoGenerateRequest::Vllm(decode) = clone else {
            panic!("expected vLLM clone");
        };

        // The original (prefill leg) is untouched.
        let original_mm = original
            .mm_inputs
            .as_ref()
            .expect("prefill keeps mm_inputs");
        assert!(original_mm.pixel_values.is_some());
        assert!(original.media_refs.is_some());

        // The expanded placeholder-token run survives: it names the sequence
        // whose KV the decode worker pulls from prefill.
        let Some(vllm::generate_request::Input::Tokenized(tokenized)) = &decode.input else {
            panic!("decode leg stays tokenized");
        };
        assert_eq!(tokenized.input_ids, vec![1, 2, 151_655, 151_655, 3]);

        // The KV handoff survives verbatim.
        assert_eq!(
            decode.kv_transfer_params_json.as_deref(),
            Some("{\"do_remote_prefill\":true}")
        );

        // Media references never reach a language-model-only decode worker.
        assert!(decode.media_refs.is_none());

        // Each batch keeps exactly its content hashes (the servicer folds
        // them into cache_salt) and modality; pixels, placeholders, grids and
        // the placeholder token id are gone.
        let decode_mm = decode.mm_inputs.expect("hash identity kept");
        assert_eq!(decode_mm.mm_hashes, vec!["img-hash".to_string()]);
        assert!(decode_mm.pixel_values.is_none());
        assert!(decode_mm.model_specific_tensors.is_empty());
        assert!(decode_mm.mm_placeholders.is_empty());
        assert!(decode_mm.im_token_id.is_none());
        assert!(decode_mm.batched_keys.is_empty());
        assert!(decode_mm.flat_keys.is_empty());
        assert_eq!(decode.extra_mm_inputs.len(), 1);
        let extra = &decode.extra_mm_inputs[0];
        assert_eq!(extra.mm_hashes, vec!["vid-hash".to_string()]);
        assert_eq!(
            extra.modality,
            smg_grpc_client::common_proto::Modality::Video as i32
        );
    }

    #[test]
    fn clone_without_mm_drops_hashless_batches_whole() {
        let mut request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            mm_inputs: Some(vllm::MultimodalInputs {
                pixel_values: Some(vllm::TensorData::default()),
                ..Default::default()
            }),
            ..Default::default()
        }));
        let clone = request.clone_without_mm();
        let ProtoGenerateRequest::Vllm(decode) = clone else {
            panic!("expected vLLM clone");
        };
        assert!(
            decode.mm_inputs.is_none(),
            "a hash-less batch carries no identity worth keeping"
        );
    }

    #[test]
    fn has_vllm_mrope_grids_reads_every_batch() {
        let plain = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            mm_inputs: Some(vllm::MultimodalInputs {
                mm_hashes: vec!["h".to_string()],
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert!(!plain.has_vllm_mrope_grids());

        // Grids hiding in a non-primary batch (mixed-modality request) count.
        let mut mixed = vllm_pd_mm_request();
        let ProtoGenerateRequest::Vllm(req) = &mut mixed else {
            panic!("expected vLLM request");
        };
        req.mm_inputs
            .as_mut()
            .unwrap()
            .model_specific_tensors
            .clear();
        req.extra_mm_inputs[0]
            .model_specific_tensors
            .insert("video_grid_thw".to_string(), vllm::TensorData::default());
        assert!(mixed.has_vllm_mrope_grids());

        let text = ProtoGenerateRequest::Vllm(Box::default());
        assert!(!text.has_vllm_mrope_grids());
    }

    #[test]
    fn sequential_pd_decode_form_picks_the_leg_for_the_worker() {
        let mm_request = vllm_pd_mm_request();

        // Full-vision decode worker: today's forms, untouched.
        assert_eq!(
            sequential_pd_decode_form(&mm_request, false, true).expect("vision decode"),
            SequentialPdDecodeForm::IdentityOnly
        );
        assert_eq!(
            sequential_pd_decode_form(&mm_request, false, false).expect("vision decode n>1"),
            SequentialPdDecodeForm::Full
        );

        // Language-model-only decode worker, no grids (MiniMax-style): the
        // fully stripped leg. A text request takes the same leg harmlessly.
        let mut stripable = vllm_pd_mm_request();
        let ProtoGenerateRequest::Vllm(req) = &mut stripable else {
            panic!("expected vLLM request");
        };
        req.media_refs = None;
        req.mm_inputs
            .as_mut()
            .unwrap()
            .model_specific_tensors
            .clear();
        assert_eq!(
            sequential_pd_decode_form(&stripable, true, true).expect("stripped leg"),
            SequentialPdDecodeForm::TextPlusHashes
        );
        let text = ProtoGenerateRequest::Vllm(Box::default());
        assert_eq!(
            sequential_pd_decode_form(&text, true, true).expect("text request"),
            SequentialPdDecodeForm::TextPlusHashes
        );
        assert_eq!(
            sequential_pd_decode_form(&text, true, false).expect("text n>1"),
            SequentialPdDecodeForm::TextPlusHashes
        );
    }

    #[test]
    fn has_kv_transfer_params_reads_both_wire_forms() {
        let mut json = ProtoGenerateRequest::Vllm(Box::default());
        assert!(!json.has_kv_transfer_params());
        json.set_kv_transfer_params_json("{\"do_remote_prefill\":true}".to_string());
        assert!(json.has_kv_transfer_params());

        let mut typed = ProtoGenerateRequest::Vllm(Box::default());
        typed.set_kv_transfer_params("10.0.0.1".to_string(), 8998);
        assert!(typed.has_kv_transfer_params());

        // Non-vLLM backends carry no connector handoff field.
        assert!(!tokenspeed_request(1, None).has_kv_transfer_params());
    }

    /// The refusals of a pool with no vision-capable decode worker (worker
    /// selection pairs one when the model has it): each names its code and
    /// says that none is paired.
    #[tokio::test]
    async fn sequential_pd_decode_form_rejects_incompatible_pairings() {
        async fn refusal(response: Response) -> String {
            assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("body");
            let text = String::from_utf8(body.to_vec()).expect("utf-8 body");
            assert!(
                text.contains("no decode worker with a vision encoder is paired"),
                "{text}"
            );
            text
        }

        // M-RoPE grids + language-model-only decode: cannot strip (positions
        // would be wrong), cannot send (encoder-cache budget 0).
        let gridded = vllm_pd_mm_request();
        let response = sequential_pd_decode_form(&gridded, true, true)
            .expect_err("mRoPE grids need a vision-capable decode worker");
        assert!(refusal(response)
            .await
            .contains("pd_decode_language_model_only_mrope"));

        // Media references + language-model-only decode: the leg would have
        // to expand the anchors itself.
        let mut refs_only = vllm_pd_mm_request();
        let ProtoGenerateRequest::Vllm(req) = &mut refs_only else {
            panic!("expected vLLM request");
        };
        req.mm_inputs = None;
        req.extra_mm_inputs.clear();
        let response = sequential_pd_decode_form(&refs_only, true, true)
            .expect_err("media refs need a processing decode worker");
        assert!(refusal(response)
            .await
            .contains("pd_decode_language_model_only_media_refs"));

        // n>1 + mm + language-model-only decode: no KV handoff, no local
        // recompute either.
        let mut fanout = vllm_pd_mm_request();
        let ProtoGenerateRequest::Vllm(req) = &mut fanout else {
            panic!("expected vLLM request");
        };
        req.media_refs = None;
        req.mm_inputs
            .as_mut()
            .unwrap()
            .model_specific_tensors
            .clear();
        let response = sequential_pd_decode_form(&fanout, true, false)
            .expect_err("n>1 cannot recompute on a pixel-less decode worker");
        assert!(refusal(response)
            .await
            .contains("pd_decode_language_model_only_n_samples"));
    }

    #[test]
    fn sanitize_sampling_for_prefill_forces_length_capped_finish() {
        let mut request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams {
                max_tokens: Some(128),
                min_tokens: 16,
                n: 4,
                stop: vec!["</s>".to_string()],
                stop_token_ids: vec![2],
                ignore_eos: false,
                ..Default::default()
            }),
            ..Default::default()
        }));
        request.sanitize_sampling_for_prefill(1);
        let ProtoGenerateRequest::Vllm(req) = request else {
            panic!("expected vLLM request");
        };
        let params = req.sampling_params.unwrap();
        assert_eq!(params.max_tokens, Some(1));
        assert_eq!(params.min_tokens, 0);
        assert_eq!(params.n, 1);
        assert!(params.stop.is_empty());
        assert!(params.stop_token_ids.is_empty());
        assert!(params.ignore_eos);
    }

    #[test]
    fn sampling_n_defaults_to_one() {
        let unset = ProtoGenerateRequest::Vllm(Box::default());
        assert_eq!(unset.sampling_n(), 1);

        let zero = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams {
                n: 0,
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert_eq!(zero.sampling_n(), 1);

        let four = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            sampling_params: Some(vllm::SamplingParams {
                n: 4,
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert_eq!(four.sampling_n(), 4);
    }

    #[test]
    fn kv_transfer_params_json_request_roundtrip() {
        let mut request = ProtoGenerateRequest::Vllm(Box::default());
        request.set_kv_transfer_params_json(NIXL_PREFILL_KV_PARAMS.to_string());
        let ProtoGenerateRequest::Vllm(req) = request else {
            panic!("expected vLLM request");
        };
        assert_eq!(
            req.kv_transfer_params_json.as_deref(),
            Some(NIXL_PREFILL_KV_PARAMS)
        );
    }

    #[test]
    fn kv_transfer_params_json_complete_accessor_filters_empty() {
        use crate::routers::grpc::proto_wrapper::ProtoGenerateComplete;

        let complete = ProtoGenerateComplete::Vllm(Box::new(vllm::GenerateComplete {
            kv_transfer_params_json: Some(r#"{"do_remote_prefill":true}"#.to_string()),
            ..Default::default()
        }));
        assert_eq!(
            complete.kv_transfer_params_json(),
            Some(r#"{"do_remote_prefill":true}"#)
        );

        let empty = ProtoGenerateComplete::Vllm(Box::new(vllm::GenerateComplete {
            kv_transfer_params_json: Some(String::new()),
            ..Default::default()
        }));
        assert_eq!(empty.kv_transfer_params_json(), None);

        let unset = ProtoGenerateComplete::Vllm(Box::default());
        assert_eq!(unset.kv_transfer_params_json(), None);
    }
}
