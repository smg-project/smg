//! The sequential prefill/decode handoff: the prefill side is sent first,
//! sanitized to one token and tagged for its connector, and the decode side
//! follows once the prefill has answered, carrying the KV transfer parameters
//! the prefill returned or the gateway minted and the media identity the
//! prefill resolved. The dispatch stage picks this shape from the runtime's
//! protocol table (`PdProtocol::for_runtime`) and keeps the side attribution,
//! the retirement and the outcome recording; the connector forms come from
//! the prefill worker's metadata (`KvConnectorMode`).

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::response::Response;
use smg_grpc_client::vllm_proto as vllm;
use tracing::{debug, error, warn};

use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            attempt_ledger::AttemptLedger,
            kv_transfer::{
                connector_mode_for_worker, is_moriio_worker, mooncake_decode_params,
                mooncake_prefill_params, KvConnectorMode, NIXL_PREFILL_KV_PARAMS,
            },
            pd_failure::{PdSide, SidePhase},
            retry::mark_non_retryable,
        },
        error,
        grpc::{
            backend_client::BackendClient,
            common::stages::request_execution::{
                decode_silence_response, dispatch_prefill_within, first_prefill_response_within,
                is_prefill_silence, prefill_silence_response, retire_pd_side,
                start_failure_response, PdSides, PrefillBound, Retirement, SideDispatch,
                StreamResult,
            },
            context::{ClientSelection, ExecutionResult, PdTiming, WorkerSelection},
            multimodal::worker_language_model_only,
            proto_wrapper::{
                FirstItem, PrefillCachedTokens, PrefillTicket, ProtoGenerateRequest,
                ProtoInputLogProbs, ProtoResponseVariant, ProtoStream,
            },
            utils::{
                account_pd_side_failure, mark_error_reason, stream_failure::UpstreamClass,
                tonic_ext::TonicStatusExt,
            },
        },
    },
    worker::{liveness, PrefillLoadGuard, Worker},
};

/// The message a sequential decode side's silence verdict starts with (see
/// [`decode_side_silent`]); it tells the verdict from an engine's own
/// `DEADLINE_EXCEEDED`.
const DECODE_SIDE_SILENCE_MESSAGE: &str = "decode side produced nothing within";

/// Give the decode side the media identity the prefill side produced, so it
/// is served without pixels or references. Only on a side that will pull its
/// prompt KV from prefill: it must hold a KV handoff (`handed_off`) and be
/// a single-sample request (`relay_kv_params`); with n>1 decode recomputes
/// the prompt and needs the references itself, even where legacy Mooncake
/// still injects its host and port, and a pixel-less side without KV is
/// refused by the servicer. A prefill that was asked for an identity
/// (`solicited`: its request carried KV params) but returned none is an
/// older servicer; the side is left as it is, and decode reprocesses the
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
                "prefill worker returned no media identity; decode side will reprocess media"
            );
            return;
        }
        None => return,
    };
    if !applied {
        warn!(
            request_id = %decode_request.request_id(),
            "prefill worker's media identity is unusable; decode side will reprocess media"
        );
    }
}

/// The bound on a sequential decode side's first token, counted from its
/// dispatch (see [`liveness::decode_side_bound`]).
#[derive(Clone, Copy, Debug)]
struct DecodeBound {
    deadline: Instant,
    bound: Duration,
}

impl DecodeBound {
    /// The bound for a side of `prompt_tokens` dispatched at `dispatched`,
    /// sized from `prefills`, the worker whose prefill of the prompt stands
    /// before the side's first token (see [`liveness::decode_side_bound`]);
    /// `None` when the rule is off.
    fn at(prefills: &Arc<dyn Worker>, prompt_tokens: u64, dispatched: Instant) -> Option<Self> {
        let bound = liveness::decode_side_bound(prefills, prompt_tokens)?;
        // A bound past the clock's range never expires: the side stays unbounded.
        let deadline = dispatched.checked_add(bound)?;
        Some(Self { deadline, bound })
    }
}

/// The worker whose prefill of the prompt stands before a sequential decode
/// side's first token, which sizes the side's bound: the prefill worker when
/// the decode side pulls its prompt KV from it, which takes both a handoff
/// and a request the KV parameters were relayed for (`handed_off` alone does
/// not say so: the legacy Mooncake arm hands off a fan-out in name only and
/// the decode engine recomputes the prompt); else the decode worker, which
/// recomputes the prompt itself (a fan-out, or a prefill that returned no
/// transfer parameters). The decode worker's prefill rate is never measured
/// under PD (its decode sides book no prefill), so that bound follows the
/// cold rate with an empty backlog, the conservative side.
fn worker_prefilling_before_decode(
    workers: &WorkerSelection,
    handed_off: bool,
    relay_kv_params: bool,
) -> Option<&Arc<dyn Worker>> {
    if handed_off && relay_kv_params {
        workers.prefill_worker()
    } else {
        workers.decode_worker()
    }
}

/// The verdict on a sequential decode side that produced nothing within its
/// first-token bound, in `phase` (`dispatch`: the call returned no stream;
/// `first_response`: the stream yielded nothing): one WARN naming the
/// worker and the request, the worker-error counter under the timeout
/// class, and the status the side's failure is reported with. The prefill
/// side has answered by then, so its silence bound no longer covers the
/// request, and nothing else bounds this wait below the request's own
/// timeout: the engines' health probes keep passing while the KV handoff
/// waits on a prefill engine that stalled right after answering, and the
/// admission gate only gated the dispatch. The attempt fails here; the
/// retry layer takes the request to another decode worker when one is
/// available (see [`decode_silence_response`]).
fn decode_side_silent(
    worker: &Arc<dyn Worker>,
    connection: &'static str,
    request_id: &str,
    prompt_tokens: u64,
    bound: Duration,
    phase: &'static str,
) -> tonic::Status {
    warn!(
        request_id = %request_id,
        worker = worker.redacted_url(),
        worker_type = metrics_labels::WORKER_DECODE,
        phase,
        prompt_tokens,
        bound_ms = u64::try_from(bound.as_millis()).unwrap_or(u64::MAX),
        "PD decode side produced nothing within its first-token bound after the prefill side had answered (the handoff did not complete, or the decode engine never scheduled it); giving up on it"
    );
    Metrics::record_worker_error(
        metrics_labels::WORKER_DECODE,
        connection,
        metrics_labels::ERROR_TIMEOUT,
    );
    tonic::Status::deadline_exceeded(format!(
        "{DECODE_SIDE_SILENCE_MESSAGE} {bound:.1?} of its dispatch ({phase})"
    ))
}

/// Whether `status` is a sequential decode side's silence verdict. Its
/// client answer is the one a stalled decode partner gets on the parallel
/// shape ([`decode_silence_response`]): a 504 `decode_worker_silent`,
/// replayable on another decode worker, this one excluded.
fn is_decode_side_silence(status: &tonic::Status) -> bool {
    status.code() == tonic::Code::DeadlineExceeded
        && status.message().starts_with(DECODE_SIDE_SILENCE_MESSAGE)
}

/// Await a sequential decode side's dispatch within its bound. Past the
/// deadline the still-pending call is retired off the request path (its
/// room is aborted once the handoff it may be in the middle of has produced
/// a first item, or the deferral's cap passes) and the silence verdict takes
/// its place. Without a bound or a worker to charge, the call is awaited as
/// it is. No reprieve: unlike a prefill side queued on a busy worker, a
/// decode side was admitted to a room before its dispatch, so nothing of the
/// worker's other sides stands between it and its first token.
async fn dispatch_decode_within(
    bound: Option<DecodeBound>,
    mut dispatch: SideDispatch,
    worker: Option<Arc<dyn Worker>>,
    connection: &'static str,
    request_id: String,
    prompt_tokens: u64,
) -> StreamResult {
    let (Some(bound), Some(worker)) = (bound, worker) else {
        return dispatch.await;
    };
    match tokio::time::timeout_at(bound.deadline.into(), &mut dispatch).await {
        Ok(result) => result,
        Err(_) => {
            retire_pd_side(
                PdSide::Decode,
                connection,
                Arc::clone(&worker),
                dispatch,
                Retirement::HandoffSilent,
            );
            Err(decode_side_silent(
                &worker,
                connection,
                &request_id,
                prompt_tokens,
                bound.bound,
                "dispatch",
            ))
        }
    }
}

/// Wait for a sequential decode side's first item within what is left of its
/// bound, and say what it was. The item stays on the stream for its reader;
/// past the bound, the silence verdict (the caller drops the side, whose
/// abort is deferred past the handoff). Without a bound nothing is pulled
/// ahead and the side counts as answered, as its dispatch did.
async fn first_decode_response_within(
    stream: &mut ProtoStream,
    bound: Option<DecodeBound>,
    worker: Option<&Arc<dyn Worker>>,
    connection: &'static str,
    request_id: &str,
    prompt_tokens: u64,
) -> Result<FirstItem, tonic::Status> {
    let (Some(bound), Some(worker)) = (bound, worker) else {
        return Ok(FirstItem::Response);
    };
    match stream.first_response_within(bound.deadline).await {
        Ok(first) => Ok(first),
        Err(_) => Err(decode_side_silent(
            worker,
            connection,
            request_id,
            prompt_tokens,
            bound.bound,
            "first_response",
        )),
    }
}

/// The attempt's answer for a decode side that failed to start or fell
/// silent on the sequential path: the breaker sample, the worker-error
/// counter (the silence verdict counted itself) and the client response.
fn sequential_decode_failure(
    e: &tonic::Status,
    workers: &WorkerSelection,
    ledger: &AttemptLedger,
    connection: &'static str,
) -> Response {
    if let Some(status) = e.breaker_status() {
        workers.record_outcome_decode(ledger, status);
    }
    if is_decode_side_silence(e) {
        if let Some(decode) = workers.decode_worker() {
            return decode_silence_response(e, decode);
        }
    }
    UpstreamClass::account_dispatch_failure(metrics_labels::WORKER_DECODE, connection, e);
    account_pd_side_failure(metrics_labels::WORKER_DECODE, e);
    start_failure_response(
        e,
        "execute_sequential_pd",
        PdSide::Decode.description(SidePhase::Start),
        PdSide::Decode.code(SidePhase::Start),
    )
}

/// The attempt's answer for a prefill side that failed to start or fell
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
    account_pd_side_failure(metrics_labels::WORKER_PREFILL, e);
    start_failure_response(
        e,
        "execute_sequential_pd",
        PdSide::Prefill.description(SidePhase::Start),
        PdSide::Prefill.code(SidePhase::Start),
    )
}

/// How the vLLM sequential-PD decode side is built from the request.
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

/// Pick the decode-side form, failing fast when the selected decode worker is
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
             decode side needs the M-RoPE grid tensors a decode worker running with \
             --language-model-only cannot take; add a decode worker with the vision \
             encoder enabled for this model"
                .to_string(),
        );
        mark_non_retryable(&mut response);
        return Err(response);
    }
    if request.has_vllm_media_refs() {
        // Media references reach each side as unexpanded anchors the side
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
             side, which a decode worker running with --language-model-only cannot \
             do; add a decode worker with the vision encoder enabled"
                .to_string(),
        );
        mark_non_retryable(&mut response);
        return Err(response);
    }
    Ok(SequentialPdDecodeForm::TextPlusHashes)
}

/// What the sequential dispatch decides before either side is sent: the pair's
/// KV connector mode, whether the KV handoff is relayed through the router,
/// the transfer id the router mints for Mooncake, and the decode side's form.
struct SequentialPlan {
    mode: KvConnectorMode,
    /// The KV handoff is single-consumer: with n>1 each fan-out child on decode
    /// would pull, and the first completion frees the prefill blocks under its
    /// siblings (same hazard for NIXL and Mooncake)
    relay_kv_params: bool,
    /// Mooncake is push-based: the engine returns nothing, so the router mints
    /// the transfer correlation id and synthesizes decode params from metadata
    mooncake_transfer_id: Option<String>,
    decode_form: SequentialPdDecodeForm,
    /// A stripped multimodal side has no local-recompute fallback: the image
    /// KV exists only on the prefill worker. Text requests strip to a no-op
    /// and keep the fallback, so only mm requests need the handoff checked.
    stripped_mm_decode: bool,
    /// The request streams: its decode side owes a first token within a bound.
    streaming: bool,
}

impl SequentialPlan {
    /// Decide the plan from the pair and the request; a pairing this pipeline
    /// cannot serve (a MoRI-IO decode, a language-model-only decode the
    /// request's multimodal payload needs more of) is refused here, before
    /// either side is sent.
    fn decide(
        proto_request: &ProtoGenerateRequest,
        workers: &WorkerSelection,
    ) -> Result<Self, Response> {
        let mode = match workers.decode_worker() {
            // A MoRI-IO decode engine never recomputes the prompt, whatever the
            // prefill side runs.
            Some(decode) if is_moriio_worker(decode.as_ref()) => KvConnectorMode::MoriIo,
            _ => workers
                .prefill_worker()
                .map(|w| connector_mode_for_worker(w.as_ref()))
                .unwrap_or(KvConnectorMode::Passthrough),
        };

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
            KvConnectorMode::Store => debug!(
                "vLLM PD (KV store): the prefill writes its blocks to the shared store and the \
                 decode reads them back by prefix hash; no connector params on either side"
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

        let relay_kv_params = proto_request.sampling_n() <= 1;

        let mooncake_transfer_id = match &mode {
            KvConnectorMode::Mooncake {
                engine_id: Some(_), ..
            } if relay_kv_params => Some(format!("xfer-{}", uuid::Uuid::now_v7())),
            _ => None,
        };

        // A decode worker started with `--language-model-only` (the production
        // vLLM P/D shape) has no vision encoder and
        // an encoder-cache budget of 0, so even the identity payload fails to
        // schedule there. Its model info reports supports_vision=false; for such
        // a worker the decode side is stripped down to what such an engine takes: the
        // prefill-expanded input_ids, the KV handoff, and the per-image content
        // hashes that the servicer folds into cache_salt so different images
        // cannot alias in the decode prefix cache.
        let decode_language_model_only = proto_request.is_vllm()
            && workers
                .decode_worker()
                .is_some_and(|worker| worker_language_model_only(worker.as_ref()));
        let decode_form =
            sequential_pd_decode_form(proto_request, decode_language_model_only, relay_kv_params)?;
        let stripped_mm_decode = matches!(decode_form, SequentialPdDecodeForm::TextPlusHashes)
            && proto_request.has_mm_inputs();
        Ok(Self {
            mode,
            relay_kv_params,
            mooncake_transfer_id,
            decode_form,
            stripped_mm_decode,
            streaming: proto_request.stream(),
        })
    }

    /// The prefill request carried KV params, so the servicer was asked for
    /// the media identity.
    fn identity_solicited(&self) -> bool {
        self.mooncake_transfer_id.is_some()
            || (self.mode == KvConnectorMode::Nixl && self.relay_kv_params)
    }

    /// The two sides' requests from the client's: the decode side in the form
    /// the plan decided, the prefill side sanitized to one token, one sample
    /// and no stream, pinned to its worker's DP rank and tagged for the
    /// connector's handoff.
    fn requests(
        &self,
        mut proto_request: ProtoGenerateRequest,
        workers: &WorkerSelection,
    ) -> (ProtoGenerateRequest, ProtoGenerateRequest) {
        // Decode normally reuses the request minus pixels: it receives KV via
        // the P/D transfer, and prefill reads and unlinks any /dev/shm
        // segments, so a reused ShmHandle would be unreadable. Same request_id
        // on both sides is load-bearing for NIXL P/D correlation on vLLM <
        // 0.13. The pixel-free side is the clone, so pixel tensors are never
        // duplicated and die with the prefill send; the per-image mm identity
        // and grid tensors survive for decode-side hashing and positions.
        // Media references are resolved by the prefill side and relayed to
        // decode as that same identity once prefill completes.
        // Without a KV handoff (n>1) decode recomputes the prompt locally and
        // must run the vision encoder, so that side keeps the full multimodal
        // payload (SHM-backed tensors cannot serve both sides and fail loudly
        // on the decode read).
        let mut decode_request = match self.decode_form {
            SequentialPdDecodeForm::Full => proto_request.clone(),
            SequentialPdDecodeForm::TextPlusHashes => proto_request.clone_without_mm(),
            SequentialPdDecodeForm::IdentityOnly => proto_request.clone_without_mm_pixels(),
        };
        if let ProtoGenerateRequest::Vllm(request) = &mut decode_request {
            request.ec_transfer_params_json = None;
        }
        // Sanitize prefill sampling (max_tokens=1, n=1), stream=false.
        let mut prefill_request = proto_request;
        prefill_request.sanitize_sampling_for_prefill(1);
        prefill_request.set_stream(false);
        if let Some(rank) = workers.prefill_worker().and_then(|w| w.dp_rank()) {
            prefill_request.set_data_parallel_rank(rank as i32);
        }
        if self.mode == KvConnectorMode::Nixl {
            if self.relay_kv_params {
                prefill_request.set_kv_transfer_params_json(NIXL_PREFILL_KV_PARAMS.to_string());
            } else {
                debug!(
                    request_id = %prefill_request.request_id(),
                    "vLLM PD (NIXL): n>1 request, skipping kv_transfer_params relay \
                     (decode recomputes the prompt locally)"
                );
            }
        }
        if let Some(ref transfer_id) = self.mooncake_transfer_id {
            prefill_request.set_kv_transfer_params_json(mooncake_prefill_params(transfer_id));
        }
        (prefill_request, decode_request)
    }
}

/// What the drained prefill side hands on: its completed stream, the connector
/// params and the processed media identity its Complete frame carried, the
/// input logprobs and the prefix-cache hit, and the side's start and end.
struct PrefillHarvest {
    stream: ProtoStream,
    kv_params: Option<String>,
    media_identity: Option<vllm::MediaIdentity>,
    input_logprobs: Option<ProtoInputLogProbs>,
    cached_tokens: PrefillCachedTokens,
    started: Instant,
    done: Instant,
}

/// Execute vLLM PD: send to prefill with max_tokens=1 first, wait for completion,
/// then send original request to decode.
///
/// For Mooncake: injects bootstrap_host/port from prefill worker metadata into
/// the decode request. For NIXL: tags the prefill request with do_remote_decode,
/// then relays the kv_transfer_params returned by the prefill engine to decode.
pub(crate) async fn execute_sequential_pd(
    proto_request: ProtoGenerateRequest,
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
    let (prefill_client, decode_client) =
        clients.disaggregated_or_mismatch("execute_sequential_pd")?;

    let plan = SequentialPlan::decide(&proto_request, workers)?;
    // Recorded on the success path (after decode established) so failed
    // requests don't pollute success metrics; captured here before use of mode.
    let kv_connector_label = plan.mode.metrics_label();
    let (prefill_request, mut decode_request) = plan.requests(proto_request, workers);
    let sides = PdSides::of(workers, ledger, &prefill_request);

    let PrefillHarvest {
        stream: prefill_stream,
        kv_params,
        media_identity,
        input_logprobs: prefill_input_logprobs,
        cached_tokens: prefill_cached_tokens,
        started: prefill_start,
        done: prefill_done,
    } = sequential_prefill_side(
        &sides,
        prefill_request,
        prefill_client.clone(),
        prefill_guard,
    )
    .await?;

    debug!("vLLM PD: prefill completed, sending decode request");

    let handed_off = relay_to_decode(
        &plan,
        workers,
        &mut decode_request,
        kv_params,
        media_identity.as_ref(),
    )?;
    let (decode_stream, decode_sent) = sequential_decode_side(
        &sides,
        &plan,
        decode_request,
        decode_client.clone(),
        handed_off,
    )
    .await?;
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

    // The client's prompt-side usage is the prefill side's: the decode side's
    // responses carry the prefix-cache hit the prefill side reported, not the
    // transfer the decode side counts as cached.
    let decode_stream = decode_stream.decode_side(prefill_cached_tokens);

    // Surface the PD shape (not `Single`) so the streaming layer anchors TTFT
    // at prefill dispatch. Without this, the decode stream is established only
    // after prefill completes, and — because the servicer flushes response
    // headers lazily with the first message — the first decode chunk is
    // already buffered when the stream task starts, so a task-local timer
    // measures ~0. The prefill stream is fully drained and its load guard
    // already released, so the prefill side carries no live guard.
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

/// The prefill side of the sequential dispatch: sent and answered within its
/// bound, booked on its worker, then drained for what the decode side needs;
/// its admission slot is released once it is drained. A failure anywhere is
/// the attempt's, where the retry layer can still take the request to another
/// prefill worker.
async fn sequential_prefill_side(
    sides: &PdSides<'_>,
    prefill_request: ProtoGenerateRequest,
    mut prefill_client: BackendClient,
    prefill_guard: PrefillLoadGuard,
) -> Result<PrefillHarvest, Response> {
    let workers = sides.workers;
    debug!(
        request_id = %prefill_request.request_id(),
        "vLLM PD: sending prefill request (max_tokens=1)"
    );

    // Send to prefill, wait for completion. The side's silence is bounded
    // from its dispatch (see `prefill_side_silent`): a call that returns no
    // stream (a servicer that holds its headers until the first message in
    // front of a frozen engine), or a stream with nothing on it, within the
    // bound fails the attempt here, where the retry layer can still take
    // the request to another prefill worker.
    let prefill_worker = workers.prefill_worker();
    let prefill_start = Instant::now();
    let bound = prefill_worker
        .and_then(|worker| PrefillBound::at(worker, sides.prompt_tokens, prefill_start));
    if let Some(prefill) = prefill_worker {
        Metrics::record_worker_request(prefill.url(), prefill.model_id());
    }
    // Booked before the call is polled, for the servicers that send their
    // headers with the first message (see `PrefillTicket::book`); the bound
    // above was read before the booking.
    let prefill_ticket =
        prefill_worker.and_then(|worker| PrefillTicket::book(worker, sides.prompt_tokens));
    let prefill_stream = dispatch_prefill_within(
        bound.clone(),
        Box::pin(async move { prefill_client.generate(prefill_request).await }),
        prefill_worker.cloned(),
        None,
        sides.prefill_label,
        sides.request_id.clone(),
        sides.prompt_tokens,
    )
    .await
    .map_err(|e| sequential_prefill_failure(&e, workers, sides.ledger, sides.prefill_label))?;
    // The prompt has been pending prefill on its worker since the dispatch,
    // until the first response, which teaches the worker's rate; the stream
    // takes the booking over.
    let mut prefill_stream = match prefill_worker {
        Some(worker) => {
            prefill_stream.tracked_prefill_side_booked(Arc::clone(worker), prefill_ticket)
        }
        None => prefill_stream,
    };
    if let Err(e) = first_prefill_response_within(
        &mut prefill_stream,
        bound,
        prefill_worker,
        None,
        sides.prefill_label,
        &sides.request_id,
        sides.prompt_tokens,
    )
    .await
    {
        // Dropping the stream aborts the side's room: nothing was handed off.
        drop(prefill_stream);
        return Err(sequential_prefill_failure(
            &e,
            workers,
            sides.ledger,
            sides.prefill_label,
        ));
    }

    // Drain prefill response, harvesting connector params, the processed
    // media identity, input logprobs and the prefix-cache hit from the
    // Complete frame
    let mut kv_params: Option<String> = None;
    let mut media_identity: Option<vllm::MediaIdentity> = None;
    let mut input_logprobs = None;
    let mut cached_tokens = PrefillCachedTokens::default();
    while let Some(result) = prefill_stream.next().await {
        match result {
            Ok(response) => {
                if let ProtoResponseVariant::Complete(complete) = response.into_response() {
                    cached_tokens.note(&complete);
                    if let Some(json) = complete.kv_transfer_params_json() {
                        kv_params = Some(json.to_owned());
                    }
                    if let Some(identity) = complete.media_identity() {
                        media_identity = Some(identity.clone());
                    }
                    if let Some(logprobs) = complete.input_logprobs() {
                        input_logprobs = Some(logprobs);
                    }
                }
            }
            Err(e) => {
                if let Some(status) = e.breaker_status() {
                    workers.record_outcome_prefill(sides.ledger, status);
                }
                UpstreamClass::account_dispatch_failure(
                    metrics_labels::WORKER_PREFILL,
                    sides.prefill_label,
                    &e,
                );
                account_pd_side_failure(metrics_labels::WORKER_PREFILL, &e);
                error!(function = "execute_sequential_pd", error = %e, "Prefill stream error");
                let mut response =
                    PdSide::Prefill.answer(SidePhase::Stream, e.http_status(), e.message());
                mark_error_reason(&mut response, e.error_reason());
                return Err(response);
            }
        }
    }
    prefill_stream.mark_completed();
    // Prefill is drained: its admission slot is free while decode runs.
    drop(prefill_guard);
    workers.record_outcome_prefill(sides.ledger, 200);
    // The prefill phase ends at the drain; recorded by the executor only once
    // decode is established.
    let done = Instant::now();
    Ok(PrefillHarvest {
        stream: prefill_stream,
        kv_params,
        media_identity,
        input_logprobs,
        cached_tokens,
        started: prefill_start,
        done,
    })
}

/// Complete the decode side's request from the prefill side's harvest: its
/// worker's DP rank, the KV handoff in the connector's form (minted for
/// Mooncake, relayed for the others) and the media identity the prefill
/// resolved. Returns whether a handoff is on the request. A stripped
/// multimodal side without one is refused rather than sent.
fn relay_to_decode(
    plan: &SequentialPlan,
    workers: &WorkerSelection,
    decode_request: &mut ProtoGenerateRequest,
    prefill_kv_params: Option<String>,
    prefill_media_identity: Option<&vllm::MediaIdentity>,
) -> Result<bool, Response> {
    if let Some(rank) = workers.decode_worker().and_then(|w| w.dp_rank()) {
        decode_request.set_data_parallel_rank(rank as i32);
    }
    let relay_kv_params = plan.relay_kv_params;
    let mut handed_off = false;
    match (&plan.mode, prefill_kv_params) {
        // Modern Mooncake: synthesized params under the minted transfer_id
        (
            KvConnectorMode::Mooncake {
                host,
                port,
                engine_id: Some(engine_id),
            },
            _,
        ) if plan.mooncake_transfer_id.is_some() => {
            let transfer_id = plan.mooncake_transfer_id.as_deref().unwrap_or_default();
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
        (
            KvConnectorMode::Nixl | KvConnectorMode::Store | KvConnectorMode::Passthrough,
            Some(json),
        ) if relay_kv_params => {
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
        decode_request,
        handed_off,
        relay_kv_params,
        plan.identity_solicited(),
        prefill_media_identity,
    );

    // A stripped multimodal side cannot fall back to a local recompute: the
    // image KV exists only on the prefill worker, so without a handoff the
    // decode engine would recompute the prompt as pure text and answer
    // image-blind. Fail instead of dispatching.
    if plan.stripped_mm_decode && !decode_request.has_kv_transfer_params() {
        error!(
            function = "execute_sequential_pd",
            request_id = %decode_request.request_id(),
            "stripped multimodal decode side has no kv_transfer_params to pull from"
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
    Ok(handed_off)
}

/// The decode side of the sequential dispatch: sent once the handoff is on its
/// request, bounded to its first token when the request streams, its abort
/// deferred until that first item proves the handoff finished, tracked on its
/// worker. Returns the established stream and the instant of the dispatch,
/// where the relay phase ends and the decode side's time to first token begins.
async fn sequential_decode_side(
    sides: &PdSides<'_>,
    plan: &SequentialPlan,
    decode_request: ProtoGenerateRequest,
    mut decode_client: BackendClient,
    handed_off: bool,
) -> Result<(ProtoStream, Instant), Response> {
    let workers = sides.workers;
    // Send request to decode
    if let Some(decode) = workers.decode_worker() {
        Metrics::record_worker_request(decode.url(), decode.model_id());
    }
    // The relay phase ends at the decode dispatch. Everything after this
    // instant, including the engines' KV transfer and the decode scheduling,
    // is the decode side's time to first token; the instant is taken before the
    // call so a worker that holds its response headers until the first token
    // still lands in the decode phase.
    let decode_sent = Instant::now();
    // The decode side owes its first token within a bound counted from here
    // (see `liveness::decode_side_bound`): the KV handoff from the prefill
    // side and the decode engine's first step. The prefill side has answered,
    // so its silence bound no longer covers the request, and nothing else
    // bounds this wait below the request's own timeout while the engines'
    // health probes keep passing (a prefill engine that stalls right after
    // answering leaves the handoff waiting on it).
    // The bound covers a streaming request only: a non-streaming side's first
    // item is its whole generation (the servicers send nothing of a
    // non-streaming request before it finishes), no first-token signal. Its
    // size follows the worker whose prefill of the prompt stands before the
    // first token (see `worker_prefilling_before_decode`).
    let decode_bound = if plan.streaming {
        worker_prefilling_before_decode(workers, handed_off, plan.relay_kv_params)
            .and_then(|worker| DecodeBound::at(worker, sides.prompt_tokens, decode_sent))
    } else {
        None
    };
    let decode_worker = workers.decode_worker().cloned();
    let decode_stream = dispatch_decode_within(
        decode_bound,
        Box::pin(async move { decode_client.generate(decode_request).await }),
        decode_worker.clone(),
        sides.decode_label,
        sides.request_id.clone(),
        sides.prompt_tokens,
    )
    .await
    .map_err(|e| sequential_decode_failure(&e, workers, sides.ledger, sides.decode_label))?;

    // Prefill has completed and its KV blocks are held pending decode's
    // transfer; aborting decode mid-transfer on a client disconnect can
    // crash or leak on the engine side. Defer the abort until the first
    // decode response proves the handoff finished (see the parallel PD
    // path for the same invariant). The side is tracked on its worker here,
    // before its first item is waited for: its responses are the worker's
    // progress, it joins no pile (its silence is bounded per request), and
    // the worker's breaker success is sampled at the side's first response
    // (`ProtoStream::tracked_decode_side`), never at the dispatch; the
    // attribution stage leaves a side tracked here as it is.
    let mut decode_stream = match &decode_worker {
        Some(worker) => decode_stream
            .defer_abort_until_first_item()
            .tracked_decode_side(Arc::clone(worker)),
        None => decode_stream.defer_abort_until_first_item(),
    };
    // Without a bound nothing is pulled ahead and the side counts as answered,
    // as before; a first item that is a failure, or an end with nothing on
    // the stream, is met by the reader and earns no success either way.
    if let Err(e) = first_decode_response_within(
        &mut decode_stream,
        decode_bound,
        decode_worker.as_ref(),
        sides.decode_label,
        &sides.request_id,
        sides.prompt_tokens,
    )
    .await
    {
        return Err(sequential_decode_failure(
            &e,
            workers,
            sides.ledger,
            sides.decode_label,
        ));
    }
    Ok((decode_stream, decode_sent))
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;
    use crate::{
        middleware::test_support::text_of,
        routers::{
            common::{attempt_ledger::FailedSide, retry::is_retryable_response},
            grpc::{
                common::stages::request_execution::tests::{
                    answers_after, fails_now, failure, media_refs_request, never_answers,
                    tokenspeed_pair, vllm_pd_mm_request,
                },
                proto_wrapper::FanoutStream,
            },
        },
        worker::{
            BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType,
            MOONCAKE_CONNECTOR, NIXL_CONNECTOR,
        },
    };

    fn three_second_decode_bound() -> DecodeBound {
        DecodeBound {
            deadline: Instant::now() + Duration::from_secs(3),
            bound: Duration::from_secs(3),
        }
    }

    /// A sequential decode call that has returned no stream by its bound is
    /// given up on at the bound, not at the request's timeout: the verdict
    /// names the silence, and the attempt answers a 504 under its own code
    /// that the retry layer may take to another decode worker, this one
    /// excluded.
    #[tokio::test(start_paused = true)]
    async fn a_sequential_decode_dispatch_past_its_bound_is_given_up_on_at_the_bound() {
        let workers = tokenspeed_pair();
        let decode = Arc::clone(workers.decode_worker().expect("a pair"));
        let started = tokio::time::Instant::now();
        let result = dispatch_decode_within(
            Some(three_second_decode_bound()),
            never_answers(),
            Some(Arc::clone(&decode)),
            metrics_labels::CONNECTION_GRPC,
            "chatcmpl-decode-silent".to_string(),
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
        assert!(is_decode_side_silence(&status), "{status:?}");
        assert!(!is_prefill_silence(&status), "{status:?}");

        let ledger = AttemptLedger::default();
        let response =
            sequential_decode_failure(&status, &workers, &ledger, metrics_labels::CONNECTION_GRPC);
        assert_eq!(response.status(), http::StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            response
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .and_then(|value| value.to_str().ok()),
            Some("decode_worker_silent")
        );
        assert!(
            response.extensions().get::<FailedSide>().is_some(),
            "the retry window closes on the decode worker alone"
        );
        assert!(
            is_retryable_response(&response),
            "another decode worker may take the request"
        );
    }

    /// A decode call that answers, or fails on its own, within its bound
    /// passes through as it is; an engine's own deadline is not the verdict;
    /// without a bound the call is awaited as before.
    #[tokio::test(start_paused = true)]
    async fn a_decode_that_answers_within_its_bound_passes_through() {
        let workers = tokenspeed_pair();
        let decode = Arc::clone(workers.decode_worker().expect("a pair"));
        let status = failure(
            dispatch_decode_within(
                Some(three_second_decode_bound()),
                answers_after(Duration::from_secs(1)),
                Some(decode),
                metrics_labels::CONNECTION_GRPC,
                "chatcmpl-decode-late".to_string(),
                256,
            )
            .await,
        );
        assert_eq!(status.message(), "late side");
        assert!(!is_decode_side_silence(&status));
        assert!(!is_decode_side_silence(&tonic::Status::deadline_exceeded(
            "bootstrap timeout"
        )));
        let status = failure(
            dispatch_decode_within(
                None,
                fails_now("refused"),
                None,
                metrics_labels::CONNECTION_GRPC,
                "chatcmpl-decode-unbounded".to_string(),
                256,
            )
            .await,
        );
        assert_eq!(status.message(), "refused");
    }

    /// A decode stream that answers (here: ends) within its bound is left
    /// for its reader as it was, and what came is named.
    #[tokio::test]
    async fn a_decode_stream_that_answers_within_its_bound_is_left_for_its_reader() {
        let workers = tokenspeed_pair();
        let decode = Arc::clone(workers.decode_worker().expect("a pair"));
        let mut stream = ProtoStream::Fanout(FanoutStream::new(Vec::new()))
            .tracked_decode_side(Arc::clone(&decode));
        let first = first_decode_response_within(
            &mut stream,
            Some(three_second_decode_bound()),
            Some(&decode),
            metrics_labels::CONNECTION_GRPC,
            "chatcmpl-decode-ended",
            256,
        )
        .await
        .expect("ended within the bound");
        assert_eq!(first, FirstItem::End);
        assert!(stream.next().await.is_none(), "the end is replayed");
    }

    /// The decode side's first-token bound is sized from the prefill worker
    /// only when the decode side pulls its prompt KV from it: a handoff on a
    /// request the KV parameters were relayed for. A fan-out the legacy
    /// Mooncake arm hands off in name only, and any side without a handoff,
    /// recompute the prompt on the decode worker, whose rate sizes the bound.
    #[test]
    fn the_decode_bound_follows_the_worker_that_prefills_the_prompt() {
        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://prefill:30000")
                .worker_type(WorkerType::Prefill)
                .connection_mode(ConnectionMode::Grpc)
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
            runtime_type: RuntimeType::Vllm,
        };
        let prefilling = |handed_off, relay_kv_params| {
            worker_prefilling_before_decode(&selection, handed_off, relay_kv_params)
                .map(|worker| worker.url().to_string())
        };
        assert_eq!(
            prefilling(true, true).as_deref(),
            Some("grpc://prefill:30000")
        );
        assert_eq!(
            prefilling(true, false).as_deref(),
            Some("grpc://decode:30000")
        );
        assert_eq!(
            prefilling(false, true).as_deref(),
            Some("grpc://decode:30000")
        );
        assert_eq!(
            prefilling(false, false).as_deref(),
            Some("grpc://decode:30000")
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
                        payload: Some(vllm::tensor_data::Payload::Inline(vec![0; 48].into())),
                    },
                )]),
                batched_keys: vec!["image_grid_thw".to_string()],
                modality: smg_grpc_client::common_proto::Modality::Image as i32,
                ..Default::default()
            }),
            extra_mm_inputs: vec![],
        }
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

    /// Legacy Mooncake injects its host and port into every decode side, n>1
    /// included, so `handed_off` alone does not say that decode will pull
    /// its prompt KV from prefill: an n>1 side recomputes locally and keeps
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
    fn decode_side_from_media_identity_has_no_refs_and_no_pixels() {
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

    /// An identity the side cannot take, empty ids or a side that is not
    /// tokenized, leaves the references in place: better a reprocessed
    /// decode than an empty prompt.
    #[test]
    fn an_unusable_identity_leaves_the_decode_side_untouched() {
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

    /// An older servicer returns no identity: the decode side keeps its
    /// references and reprocesses, as before; the same when no identity was
    /// asked for (a prefill request without KV params).
    #[test]
    fn missing_identity_leaves_the_decode_side_untouched() {
        for solicited in [true, false] {
            let mut decode = media_refs_request("old", 1).clone_without_mm_pixels();
            apply_prefill_media_identity(&mut decode, true, true, solicited, None);
            assert!(decode.has_vllm_media_refs(), "solicited={solicited}");
        }
    }

    #[test]
    fn sequential_pd_decode_form_picks_the_side_for_the_worker() {
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
        // fully stripped side. A text request takes the same side harmlessly.
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
            sequential_pd_decode_form(&stripable, true, true).expect("stripped side"),
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

    /// The refusals of a pool with no vision-capable decode worker (worker
    /// selection pairs one when the model has it): each names its code and
    /// says that none is paired.
    #[tokio::test]
    async fn sequential_pd_decode_form_rejects_incompatible_pairings() {
        async fn refusal(response: Response) -> String {
            assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
            let text = text_of(response).await;
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

        // Media references + language-model-only decode: the side would have
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

    /// A streaming text request of `n` samples, the client's side of every
    /// recorded pair below.
    fn recorded_request(n: u32) -> ProtoGenerateRequest {
        ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            request_id: "chatcmpl-recorded-pair".to_string(),
            input: Some(vllm::generate_request::Input::Tokenized(
                vllm::TokenizedInput {
                    original_text: "the pair on the wire".to_string(),
                    input_ids: vec![11, 12, 13, 14, 15],
                },
            )),
            sampling_params: Some(vllm::SamplingParams {
                temperature: Some(0.7),
                top_p: 0.9,
                max_tokens: Some(48),
                n,
                seed: Some(7),
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        }))
    }

    /// A prefill/decode pair whose prefill worker reports `connector`, and the
    /// engine id a worker reports when it has one.
    fn connector_pair(connector: &str, engine_id: Option<&str>) -> WorkerSelection {
        let mut prefill = BasicWorkerBuilder::new("grpc://prefill:30000")
            .worker_type(WorkerType::Prefill)
            .runtime_type(RuntimeType::Vllm)
            .connection_mode(ConnectionMode::Grpc)
            .kv_connector(connector)
            .bootstrap_port(Some(8998));
        if let Some(engine_id) = engine_id {
            prefill = prefill.kv_engine_id(engine_id);
        }
        let prefill: Arc<dyn Worker> = Arc::new(prefill.build());
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:30000")
                .worker_type(WorkerType::Decode)
                .runtime_type(RuntimeType::Vllm)
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        );
        WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill,
            decode,
            runtime_type: RuntimeType::Vllm,
        }
    }

    /// The parameters a prefill engine returns under a NIXL handoff.
    const NIXL_RETURNED: &str = r#"{"do_remote_prefill":true,"remote_block_ids":[3,4,5],"remote_engine_id":"prefill-engine","remote_host":"10.0.0.2","remote_port":5600}"#;

    /// The bytes a request is sent as, in hex.
    fn wire(request: &ProtoGenerateRequest) -> String {
        let ProtoGenerateRequest::Vllm(request) = request else {
            panic!("the sequential pair rides one wire");
        };
        prost::Message::encode_to_vec(request.as_ref())
            .into_iter()
            .fold(String::new(), |mut hex, byte| {
                let _ = write!(hex, "{byte:02x}");
                hex
            })
    }

    /// The pair the plan sends for `request`: the prefill side as built, the
    /// decode side once `prefill_kv_params` (what the prefill returned) is
    /// relayed onto it.
    fn sent_pair(
        plan: &SequentialPlan,
        request: ProtoGenerateRequest,
        workers: &WorkerSelection,
        prefill_kv_params: Option<&str>,
    ) -> (String, String) {
        let (prefill, mut decode) = plan.requests(request, workers);
        relay_to_decode(
            plan,
            workers,
            &mut decode,
            prefill_kv_params.map(str::to_owned),
            None,
        )
        .expect("a handoff the plan can send");
        (wire(&prefill), wire(&decode))
    }

    /// The prefill side of a NIXL handoff: one token, no stream, tagged for the
    /// remote decode; its decode side carries what the prefill returned.
    const NIXL_PREFILL: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22130d3333333f156666663f400170017801c0010742337b22646f5f72656d6f74655f6465636f6465223a747275652c22646f5f72656d6f74655f70726566696c6c223a66616c73657d";
    const NIXL_DECODE: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22110d3333333f156666663f40307801c0010728014285017b22646f5f72656d6f74655f70726566696c6c223a747275652c2272656d6f74655f626c6f636b5f696473223a5b332c342c355d2c2272656d6f74655f656e67696e655f6964223a2270726566696c6c2d656e67696e65222c2272656d6f74655f686f7374223a2231302e302e302e32222c2272656d6f74655f706f7274223a353630307d";
    /// A minted Mooncake handoff: the prefill side pinned under the minted id,
    /// the decode side's parameters synthesized from the prefill worker.
    const MOONCAKE_MINTED_PREFILL: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22130d3333333f156666663f400170017801c0010742517b22646f5f72656d6f74655f6465636f6465223a747275652c22646f5f72656d6f74655f70726566696c6c223a66616c73652c227472616e736665725f6964223a22786665722d7265636f72646564227d";
    const MOONCAKE_MINTED_DECODE: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22110d3333333f156666663f40307801c001072801429d017b22646f5f72656d6f74655f6465636f6465223a66616c73652c22646f5f72656d6f74655f70726566696c6c223a747275652c227472616e736665725f6964223a22786665722d7265636f72646564222c2272656d6f74655f656e67696e655f6964223a22656e67696e652d61222c2272656d6f74655f626f6f7473747261705f61646472223a22687474703a2f2f70726566696c6c3a38393938227d";
    /// The legacy Mooncake handoff of a worker without an engine id: the typed
    /// host and port on the decode side, nothing on the prefill side.
    const MOONCAKE_LEGACY_PREFILL: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22130d3333333f156666663f400170017801c00107";
    const MOONCAKE_LEGACY_DECODE: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22110d3333333f156666663f40307801c001072801320c0a0770726566696c6c10a646";
    /// An n>1 request under NIXL: no handoff on either side, the decode side
    /// recomputes the prompt for its samples.
    const N2_PREFILL: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22130d3333333f156666663f400170017801c00107";
    const N2_DECODE: &str = "0a1663686174636d706c2d7265636f726465642d70616972121d0a147468652070616972206f6e20746865207769726512050b0c0d0e0f22110d3333333f156666663f40307802c001072801";

    /// Each connector form's pair goes on the wire byte for byte as the
    /// dispatch stage sent it before this module (the constants above were
    /// recorded there, from these inputs).
    #[test]
    fn a_nixl_pair_is_sent_as_recorded() {
        let workers = connector_pair(NIXL_CONNECTOR, None);
        let plan = SequentialPlan::decide(&recorded_request(1), &workers).expect("a NIXL pair");
        assert_eq!(plan.mode, KvConnectorMode::Nixl);
        assert_eq!(
            sent_pair(&plan, recorded_request(1), &workers, Some(NIXL_RETURNED)),
            (NIXL_PREFILL.to_string(), NIXL_DECODE.to_string())
        );
    }

    #[test]
    fn a_minted_mooncake_pair_is_sent_as_recorded() {
        let workers = connector_pair(MOONCAKE_CONNECTOR, Some("engine-a"));
        let mut plan =
            SequentialPlan::decide(&recorded_request(1), &workers).expect("a Mooncake pair");
        assert!(
            matches!(
                plan.mode,
                KvConnectorMode::Mooncake {
                    engine_id: Some(_),
                    ..
                }
            ),
            "{:?}",
            plan.mode
        );
        assert!(plan.mooncake_transfer_id.is_some(), "a minted id");
        // The id is minted fresh per request; the record pins the form under one.
        plan.mooncake_transfer_id = Some("xfer-recorded".to_string());
        assert_eq!(
            sent_pair(&plan, recorded_request(1), &workers, None),
            (
                MOONCAKE_MINTED_PREFILL.to_string(),
                MOONCAKE_MINTED_DECODE.to_string()
            )
        );
    }

    #[test]
    fn a_legacy_mooncake_pair_is_sent_as_recorded() {
        let workers = connector_pair(MOONCAKE_CONNECTOR, None);
        let plan = SequentialPlan::decide(&recorded_request(1), &workers).expect("a Mooncake pair");
        assert!(
            matches!(
                plan.mode,
                KvConnectorMode::Mooncake {
                    engine_id: None,
                    ..
                }
            ),
            "{:?}",
            plan.mode
        );
        assert_eq!(
            plan.mooncake_transfer_id, None,
            "nothing minted without an engine id"
        );
        assert_eq!(
            sent_pair(&plan, recorded_request(1), &workers, None),
            (
                MOONCAKE_LEGACY_PREFILL.to_string(),
                MOONCAKE_LEGACY_DECODE.to_string()
            )
        );
    }

    #[test]
    fn an_n_greater_than_one_pair_is_sent_as_recorded() {
        let workers = connector_pair(NIXL_CONNECTOR, None);
        let plan = SequentialPlan::decide(&recorded_request(2), &workers).expect("a NIXL pair");
        assert!(!plan.relay_kv_params, "n>1 relays no handoff");
        assert_eq!(
            sent_pair(&plan, recorded_request(2), &workers, None),
            (N2_PREFILL.to_string(), N2_DECODE.to_string())
        );
    }
}
