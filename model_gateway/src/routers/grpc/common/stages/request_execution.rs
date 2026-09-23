//! Request execution: dispatch one attempt of the retained execution plan.

use std::{future::Future, pin::Pin, sync::Arc, time::Instant};

use axum::response::Response;
use futures::future::{join_all, try_join_all};
use smg_grpc_client::vllm_proto as vllm;
use tracing::{debug, error, info_span, warn, Instrument};

use super::{
    helpers::{maybe_inject_pd_metadata, maybe_inject_pd_rendezvous},
    pd_protocol::{DpPlacement, PdDispatch, PdProtocol},
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            kv_transfer::{
                connector_mode_for_worker, mooncake_decode_params, mooncake_prefill_params,
                KvConnectorMode, NIXL_PREFILL_KV_PARAMS,
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
                FanoutStream, ProtoEmbedRequest, ProtoGenerateRequest, ProtoRequest,
                ProtoResponseVariant, ProtoStream,
            },
            utils::tonic_ext::{TonicResultExt, TonicStatusExt},
        },
    },
    worker::{ConnectionModeExt, Worker},
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
            sub.set_request_id(format!("{base_id}-{i}"));
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
    execution_plan: ExecutionPlan,
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

    ctx.load_guards = Some(LoadGuards::admitted(
        admission,
        LoadGuards::scaled(workers, ctx.sticky_key.as_deref(), sub_requests),
    ));

    // Extract dispatch metadata for the tracing span and PD metric labels.
    let dispatch = ctx.dispatch.as_ref().ok_or_else(|| {
        error!(function = "execute_plan", "Dispatch metadata not set");
        error::internal_error("dispatch_metadata_not_set", "Dispatch metadata not set")
    })?;
    let request_id = dispatch.request_id.as_str();
    let model = dispatch.model.as_str();
    let request_type = execution_plan.request_type();
    let mode = execution_plan.mode_label();

    // Create OTEL span for gRPC request execution
    let span = info_span!(
        target: "smg::otel-trace",
        "grpc_execute",
        request_type,
        request_id = %request_id,
        model = %model,
        mode = %mode,
    );

    let result = async {
        match execution_plan {
            ExecutionPlan::Single(request) => match request {
                ProtoRequest::Generate(req) => execute_single(req, clients, workers).await,
                ProtoRequest::Embed(req) => execute_single_embed(req, clients, workers).await,
            },
            ExecutionPlan::PrefillDecode(req) => {
                execute_pd_dispatch(req, clients, workers, model).await
            }
            ExecutionPlan::EncodePrefillDecode { request } => {
                // Bootstrap info was injected into the prefill request during
                // request building; dispatch the encode jobs with the
                // prefill+decode leg.
                execute_epd_dispatch(request, clients, workers, model, encode_dispatch).await
            }
            ExecutionPlan::Batch { kind, requests, .. } => {
                execute_batch_dispatch(kind, requests, clients, workers, model).await
            }
        }
    }
    .instrument(span)
    .await;
    // The engines hold the request bodies now. An earlier attempt keeps its
    // share of the budget: the retained plan still owns the same media, and a
    // replay would send it again.
    if last_attempt {
        ctx.multimodal_inflight.take();
    }
    let result = result?;

    // Store result in context for response processing
    ctx.response.execution_result = Some(result);
    Ok(())
}

async fn execute_pd_dispatch(
    proto_request: ProtoGenerateRequest,
    clients: &mut ClientSelection,
    workers: &WorkerSelection,
    model: &str,
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
    match protocol.dispatch {
        PdDispatch::Sequential => {
            execute_sequential_pd(proto_request, clients, workers, model).await
        }
        PdDispatch::Parallel => match pd_fanout_width(&proto_request, protocol) {
            Some(n) => execute_fanout_pd(proto_request, n, clients, workers, protocol).await,
            None => execute_parallel_pd(proto_request, clients, workers, protocol).await,
        },
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
    let dispatches = subs.into_iter().map(|sub| {
        let mut clients = clients.clone();
        async move { execute_parallel_pd(sub, &mut clients, workers, protocol).await }
    });
    let results = try_join_all(dispatches).await?;

    let mut prefills = Vec::with_capacity(results.len());
    let mut decodes = Vec::with_capacity(results.len());
    let mut timing: Option<PdTiming> = None;
    for result in results {
        let ExecutionResult::PrefillDecode {
            prefill,
            decode,
            pd_timing,
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
        pd_timing,
    })
}

async fn execute_epd_dispatch(
    mut proto_request: ProtoGenerateRequest,
    clients: &mut ClientSelection,
    workers: &WorkerSelection,
    model: &str,
    encode_dispatch: Option<EncodeDispatchPlan>,
) -> Result<ExecutionResult, Response> {
    if let Some(encode_dispatch) = encode_dispatch {
        spawn_encode_dispatch(encode_dispatch);
    }
    proto_request.clear_mm_pixel_values();
    execute_pd_dispatch(proto_request, clients, workers, model).await
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
) -> Result<ExecutionResult, Response> {
    let dispatches = requests.into_iter().map(|request| {
        let mut clients = clients.clone();
        async move {
            match kind {
                ExecutionPlanKind::Single => execute_single(request, &mut clients, workers).await,
                // Completion EPD carries no encode jobs; sub-requests dispatch as PD.
                ExecutionPlanKind::PrefillDecode | ExecutionPlanKind::EncodePrefillDecode => {
                    execute_pd_dispatch(request, &mut clients, workers, model).await
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

    let result = client.generate(proto_request).await;
    workers.record_outcome(result.cb_status_code());

    let stream = result.map_err(|e| {
        start_failure_response(
            &e,
            "execute_single",
            "Failed to start generation",
            "start_generation_failed",
        )
    })?;

    Ok(ExecutionResult::Single { stream })
}

/// Client answer for a request the worker did not start; an engine rejection (4xx) is not a fault.
fn start_failure_response(
    e: &tonic::Status,
    function: &'static str,
    description: &str,
    code: &str,
) -> Response {
    if e.http_status().is_client_error() {
        warn!(function = function, error = %e, "{}: engine rejected the request", description);
    } else {
        error!(function = function, error = %e, "{}", description);
    }
    let mut response = e.to_http_error(code, format!("{description}: {}", e.message()));
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

    let result = client.embed(proto_request).await;
    workers.record_outcome(result.cb_status_code());

    let complete = result.map_err(|e| {
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
    // Each leg owns its client handle (a cheap channel clone, the same one
    // batched dispatch takes per sub-request) so the leg still in flight when
    // its partner fails can be moved off the request path.
    let mut prefill_client = prefill_client.clone();
    let mut decode_client = decode_client.clone();
    let prefill_dispatch: PdLegDispatch =
        Box::pin(async move { prefill_client.generate(prefill_request).await });
    let decode_dispatch: PdLegDispatch =
        Box::pin(async move { decode_client.generate(decode_request).await });

    match dispatch_pd_legs(prefill_dispatch, decode_dispatch).await {
        PdDispatchOutcome::Both(prefill_result, decode_result) => {
            // Record circuit breaker outcomes (client errors don't count as failures)
            workers.record_prefill_decode_outcomes(
                prefill_result.cb_status_code(),
                decode_result.cb_status_code(),
            );

            // Both legs are translated before either is propagated: a
            // decode-side failure is logged and counted even when the prefill
            // error is the one the client sees.
            let prefill =
                prefill_result.map_err(|e| pd_leg_error(PdLeg::Prefill, prefill_label, &e));
            let decode = decode_result.map_err(|e| pd_leg_error(PdLeg::Decode, decode_label, &e));
            let (prefill_stream, decode_stream) = match (prefill, decode) {
                (Ok(prefill), Ok(decode)) => (prefill, decode),
                // The surviving leg's stream drops here, which aborts its
                // room on the worker.
                (Err(response), _) | (Ok(_), Err(response)) => return Err(response),
            };

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
                pd_timing: PdTiming {
                    prefill_start,
                    runtime,
                },
            })
        }
        PdDispatchOutcome::FailedFirst {
            leg,
            error,
            partner,
        } => {
            let status = error.http_status().as_u16();
            // Only the leg that answered is recorded: the abandoned one has
            // said nothing about its worker yet, and `retire_pd_leg` records
            // it when it finally does.
            let (label, partner_label, partner_worker) = match leg {
                PdLeg::Prefill => {
                    workers.record_outcome_prefill(status);
                    (prefill_label, decode_label, workers.decode_worker())
                }
                PdLeg::Decode => {
                    workers.record_outcome_decode(status);
                    (decode_label, prefill_label, workers.prefill_worker())
                }
            };
            if let Some(worker) = partner_worker {
                retire_pd_leg(leg.partner(), partner_label, Arc::clone(worker), partner);
            }
            Err(pd_leg_error(leg, label, &error))
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
fn pd_leg_error(leg: PdLeg, connection: &'static str, error: &tonic::Status) -> Response {
    Metrics::record_worker_error(leg.name(), connection, metrics_labels::ERROR_BACKEND);
    start_failure_response(
        error,
        "execute_parallel_pd",
        leg.error_message(),
        leg.error_code(),
    )
}

/// Finish off the leg that was still dispatching when its partner failed.
///
/// The request is already answered; what is left is the engine-side room,
/// which the abandoned leg will allocate the moment its dispatch lands and
/// then hold for its own deadline. Dropping the leg's stream as soon as it
/// arrives is the abort path the router already uses on a client disconnect,
/// and it is the only thing that releases that room early. The abort is *not*
/// deferred here: deferral exists to protect an in-flight KV handoff, and the
/// peer that would have written it is the leg that just failed.
#[expect(
    clippy::disallowed_methods,
    reason = "the abandoned PD leg is retired off the request path; its dispatch is bounded by the engine's own deadline"
)]
fn retire_pd_leg(
    leg: PdLeg,
    connection: &'static str,
    worker: Arc<dyn Worker>,
    dispatch: PdLegDispatch,
) {
    tokio::spawn(async move {
        let result = dispatch.await;
        worker.record_outcome(result.cb_status_code());
        match result {
            Ok(stream) => {
                debug!(
                    leg = leg.name(),
                    worker = worker.url(),
                    "Aborting the room of the PD leg abandoned after its partner failed"
                );
                drop(stream);
            }
            Err(error) => {
                Metrics::record_worker_error(leg.name(), connection, metrics_labels::ERROR_BACKEND);
                error!(
                    function = "retire_pd_leg",
                    leg = leg.name(),
                    worker = worker.url(),
                    error = %error,
                    "PD leg failed after its partner had already failed"
                );
            }
        }
    });
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
            "the selected decode worker runs with --language-model-only, but this \
             model's decode leg needs its M-RoPE grid tensors; run the decode pool \
             with the vision encoder enabled for this model"
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
            "the selected decode worker runs with --language-model-only and cannot \
             expand media references; use router-side multimodal processing \
             (--mm-processing=router) for this pool"
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
            "the selected decode worker runs with --language-model-only and cannot \
             recompute a multimodal prompt; n>1 parallel sampling needs decode \
             workers with the vision encoder enabled"
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

    let mode = workers
        .prefill_worker()
        .map(|w| connector_mode_for_worker(w.as_ref()))
        .unwrap_or(KvConnectorMode::Passthrough);

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
    // vLLM P/D shape — Dynamo pairs the same way) has no vision encoder and
    // an encoder-cache budget of 0, so even the identity payload fails to
    // schedule there. Its model info reports supports_vision=false; for such
    // a worker the decode leg is stripped down to the Dynamo contract: the
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

    // Send to prefill, wait for completion
    let (prefill_label, decode_label) = pd_leg_labels(workers);
    let prefill_start = Instant::now();
    let mut prefill_stream = prefill_client
        .generate(prefill_request)
        .await
        .map_err(|e| {
            workers.record_outcome_prefill(e.http_status().as_u16());
            Metrics::record_worker_error(
                metrics_labels::WORKER_PREFILL,
                prefill_label,
                metrics_labels::ERROR_BACKEND,
            );
            start_failure_response(
                &e,
                "execute_sequential_pd",
                PdLeg::Prefill.error_message(),
                PdLeg::Prefill.error_code(),
            )
        })?;

    // Drain prefill response, harvesting connector params and the processed
    // media identity from the Complete frame
    let mut prefill_kv_params: Option<String> = None;
    let mut prefill_media_identity: Option<vllm::MediaIdentity> = None;
    while let Some(result) = prefill_stream.next().await {
        match result {
            Ok(response) => {
                if let ProtoResponseVariant::Complete(complete) = response.into_response() {
                    if let Some(json) = complete.kv_transfer_params_json() {
                        prefill_kv_params = Some(json.to_owned());
                    }
                    if let Some(identity) = complete.media_identity() {
                        prefill_media_identity = Some(identity.clone());
                    }
                }
            }
            Err(e) => {
                workers.record_outcome_prefill(e.http_status().as_u16());
                Metrics::record_worker_error(
                    metrics_labels::WORKER_PREFILL,
                    prefill_label,
                    metrics_labels::ERROR_BACKEND,
                );
                error!(function = "execute_sequential_pd", error = %e, "Prefill stream error");
                return Err(e.to_http_error(
                    "prefill_stream_error",
                    format!("Prefill stream error: {}", e.message()),
                ));
            }
        }
    }
    prefill_stream.mark_completed();
    workers.record_outcome_prefill(200);
    // Captured at drain; recorded below only once decode is established.
    let prefill_duration = prefill_start.elapsed();

    // KV-transfer window: prefill drain complete to decode send complete.
    let kv_window_start = Instant::now();

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
    let decode_stream = decode_client.generate(decode_request).await.map_err(|e| {
        workers.record_outcome_decode(e.http_status().as_u16());
        Metrics::record_worker_error(
            metrics_labels::WORKER_DECODE,
            decode_label,
            metrics_labels::ERROR_BACKEND,
        );
        start_failure_response(
            &e,
            "execute_sequential_pd",
            PdLeg::Decode.error_message(),
            PdLeg::Decode.error_code(),
        )
    })?;

    workers.record_outcome_decode(200);
    // Decode established: record the success-only PD metrics here.
    Metrics::record_pd_kv_connector_mode(kv_connector_label);
    Metrics::record_pd_prefill_duration(
        metrics_labels::BACKEND_PD,
        model,
        runtime,
        prefill_duration,
    );
    Metrics::record_pd_kv_transfer_duration(
        metrics_labels::BACKEND_PD,
        model,
        runtime,
        kv_window_start.elapsed(),
    );

    // Prefill has completed and its KV blocks are held pending decode's
    // transfer; aborting decode mid-transfer on a client disconnect can
    // crash or leak on the engine side. Defer the abort until the first
    // decode response proves the handoff finished (see the parallel PD
    // path for the same invariant).
    let decode_stream = decode_stream.defer_abort_until_first_item();

    Ok(ExecutionResult::Single {
        stream: decode_stream,
    })
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use smg_grpc_client::{sglang_proto as sglang, tokenspeed_proto as ts, vllm_proto as vllm};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType};

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

    #[tokio::test]
    async fn sequential_pd_decode_form_rejects_incompatible_pairings() {
        // M-RoPE grids + language-model-only decode: cannot strip (positions
        // would be wrong), cannot send (encoder-cache budget 0).
        let gridded = vllm_pd_mm_request();
        let response = sequential_pd_decode_form(&gridded, true, true)
            .expect_err("mRoPE grids need a vision-capable decode worker");
        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);

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
        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);

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
        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
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
