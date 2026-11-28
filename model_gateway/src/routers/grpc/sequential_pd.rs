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

use axum::{http::HeaderMap, response::Response};
use smg_grpc_client::vllm_proto as vllm;
use smg_worker::{
    health::liveness, identity::basic::Worker, metrics::writers::Metrics as WorkerMetrics,
};
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
            prefill_admission::PrefillLoadGuard,
            retry::mark_non_retryable,
        },
        error,
        grpc::{
            backend_client::BackendClient,
            common::stages::{
                request_execution::{
                    decode_silence_response, dispatch_prefill_within,
                    first_prefill_response_within, is_prefill_silence, owe_decode_end,
                    pd_start_failure_response, prefill_silence_response, retire_pd_side,
                    tell_prefill_failure, PdSides, PrefillBound, PrefillWait, Retirement,
                    SideDispatch, StreamResult,
                },
                MovedDecode, RequestAgain, WorkerSelectionStage,
            },
            context::{
                ClientSelection, ExecutionResult, LoadGuards, PdTiming, RoutingSnapshot,
                WireConstraint, WorkerSelection,
            },
            multimodal::worker_language_model_only,
            proto_wrapper::{
                FirstItem, PrefillCachedTokens, PrefillTicket, ProtoGenerateRequest,
                ProtoInputLogProbs, ProtoResponseVariant, ProtoStream,
            },
            utils::{
                account_pd_side_failure, mark_error_reason,
                stream_failure::UpstreamClass,
                tonic_ext::{is_receive_bound_refusal, is_the_engines_own_report, TonicStatusExt},
            },
        },
    },
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

/// Whether `identity` can stand in for a stripped references side's whole
/// multimodal payload: expanded prompt ids, and per-item content hashes on
/// every batch, which key the decode prefix cache so two requests' media
/// never share a block.
fn identity_serves_a_stripped_side(identity: &vllm::MediaIdentity) -> bool {
    let mut batches = identity
        .mm_inputs
        .iter()
        .chain(&identity.extra_mm_inputs)
        .peekable();
    !identity.prompt_token_ids.is_empty()
        && batches.peek().is_some()
        && batches.all(|batch| !batch.mm_hashes.is_empty())
}

/// Give a stripped references side its whole multimodal payload: the prefill
/// side's media identity (the expanded prompt ids and the per-item content
/// hashes) replacing the references, then stripped to the hashes alone
/// (pixels, placeholders and grid tensors never leave the prefill side). The
/// identity is applied before the strip: it is what the side is stripped
/// down to. A prefill that returned no identity that can stand in for the
/// payload ([`identity_serves_a_stripped_side`]) fails the request:
/// references never reach a language-model-only decode worker, and a side
/// is never sent with media its prefix cache cannot tell apart.
fn apply_refs_media_identity(
    decode_request: &mut ProtoGenerateRequest,
    prefill_media_identity: Option<&vllm::MediaIdentity>,
) -> Result<(), Response> {
    let applied = prefill_media_identity.is_some_and(|identity| {
        identity_serves_a_stripped_side(identity) && decode_request.apply_media_identity(identity)
    });
    if !applied {
        error!(
            request_id = %decode_request.request_id(),
            "prefill worker returned no media identity a language-model-only decode \
             worker can be served with"
        );
        let mut response = error::bad_gateway(
            "pd_decode_missing_media_identity",
            "the prefill worker returned no usable media identity for this request (the \
             expanded prompt ids, and content hashes on every batch), which a decode worker \
             running with --language-model-only needs in place of the media references: \
             the prefill servicer is older than this pairing, or could not describe the \
             request's media"
                .to_string(),
        );
        mark_non_retryable(&mut response);
        return Err(response);
    }
    decode_request.strip_mm_to_hashes();
    Ok(())
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
/// The answer bears the end the decode owes Grand Central, as the breaker
/// sampled it: the plan holds the decode's guard and tells it.
fn sequential_decode_failure(
    e: &tonic::Status,
    workers: &WorkerSelection,
    ledger: &AttemptLedger,
    connection: &'static str,
) -> Response {
    if let Some(status) = e.breaker_status() {
        workers.record_outcome_decode(ledger, status);
    }
    let mut response = sequential_decode_answer(e, workers, connection);
    owe_decode_end(&mut response, workers, || e.end_status());
    response
}

/// The client response of [`sequential_decode_failure`], with its counter.
fn sequential_decode_answer(
    e: &tonic::Status,
    workers: &WorkerSelection,
    connection: &'static str,
) -> Response {
    if is_decode_side_silence(e) {
        if let Some(decode) = workers.decode_worker() {
            return decode_silence_response(e, decode);
        }
    }
    UpstreamClass::account_dispatch_failure(metrics_labels::WORKER_DECODE, connection, e);
    account_pd_side_failure(metrics_labels::WORKER_DECODE, e);
    pd_start_failure_response(
        PdSide::Decode,
        workers.decode_worker(),
        e,
        "execute_sequential_pd",
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
    pd_start_failure_response(
        PdSide::Prefill,
        workers.prefill_worker(),
        e,
        "execute_sequential_pd",
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
    /// Language-model-only decode worker, media references forwarded: the
    /// references ride the decode side only until the prefill side answers,
    /// then its media identity replaces them (the expanded prompt ids, the
    /// per-item content hashes) and the side is stripped to text and hashes
    /// before dispatch. A prefill that returns no usable identity fails the
    /// request: references never reach a worker that cannot expand them.
    RefsThenIdentity,
}

/// Pick the decode-side form, failing fast when the selected decode worker is
/// language-model-only and the request is one that pairing cannot serve.
/// Worker selection keeps such a request off such workers while the model
/// pairs a decode worker with the vision encoder (`decode_needs_vision`), so
/// a refusal here means none is paired. `language_model_reads_grids` is the
/// model's answer to whether its language model reads the grid tensors the
/// request carries (`RoutingSnapshot::language_model_reads_grids`): when it
/// does not, the grids feed only the vision tower, which runs on the side
/// that encodes the media (the prefill side, or the encode workers in
/// encode/prefill/decode mode), and the stripped side serves the request.
/// For media references that also needs the prefill side to be asked for
/// its media identity (`identity_solicited`, asked only of such a request):
/// the identity is what the stripped side is made of.
// Always in line: the plan's decision ran this built in, on every sequential
// request's path, before the check of its refusals called it too. With two
// callers it stood alone, and a request read 32 instructions more.
#[inline(always)]
fn sequential_pd_decode_form(
    request: &ProtoGenerateRequest,
    decode_language_model_only: bool,
    relay_kv_params: bool,
    identity_solicited: impl FnOnce() -> bool,
    language_model_reads_grids: bool,
) -> Result<SequentialPdDecodeForm, Response> {
    if !decode_language_model_only {
        return Ok(if relay_kv_params {
            SequentialPdDecodeForm::IdentityOnly
        } else {
            SequentialPdDecodeForm::Full
        });
    }
    if language_model_reads_grids && request.has_vllm_mrope_grids() {
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
        // expands itself; a language-model-only decode worker cannot. The
        // pairing still serves the request when the decode side needs no
        // expansion of its own: a single sample relays the KV handoff, the
        // prefill side is asked for its media identity (the expanded prompt
        // ids and the per-item content hashes), and the model's language
        // model reads no grid tensors.
        let unserved = if !relay_kv_params {
            Some("the request asks for more than one sample")
        } else if language_model_reads_grids {
            Some("the model is not known to leave the grid tensors unread by its language model")
        } else if !identity_solicited() {
            Some(
                "the pair's KV connector does not return the prefill's media identity \
                 (NIXL does, and Mooncake with a discovered engine id)",
            )
        } else {
            None
        };
        let Some(unserved) = unserved else {
            return Ok(SequentialPdDecodeForm::RefsThenIdentity);
        };
        let mut response = error::bad_request(
            "pd_decode_language_model_only_media_refs",
            format!(
                "no decode worker with a vision encoder is paired for this model, and a \
                 decode worker running with --language-model-only cannot expand media \
                 references. Such a worker takes the prefill's media identity in their \
                 place, but not here: {unserved}. Add a decode worker with the vision \
                 encoder, or use router-side multimodal processing \
                 (--mm-processing=router) for this pool."
            ),
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

/// Whether the prefill request of a pair on the connector `mode` carries KV
/// params, so its servicer is asked for the media identity: a minted
/// Mooncake handoff, or a relayed NIXL one, of a single sample alone. The
/// plan's decision, its refusal check and the plan itself read it here.
fn solicits_identity(mode: &KvConnectorMode, relay_kv_params: bool) -> bool {
    relay_kv_params
        && matches!(
            mode,
            KvConnectorMode::Nixl
                | KvConnectorMode::Mooncake {
                    engine_id: Some(_),
                    ..
                }
        )
}

/// What refuses a pair on the MoRI-IO connector, for the plan's decision and
/// for the check of its refusals alike.
fn moriio_unsupported() -> Response {
    error::not_implemented(
        "moriio_grpc_pd_unsupported",
        "MoRIIOConnector PD is supported by the HTTP PD router only",
    )
}

/// What the sequential dispatch decides before either side is sent: the pair's
/// KV connector mode, whether the KV handoff is relayed through the router,
/// the transfer id the router mints for Mooncake, and the decode side's form.
pub(crate) struct SequentialPlan {
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
    /// cannot serve (a MoRI-IO decode or prefill, a language-model-only
    /// decode the request's multimodal payload needs more of) is refused
    /// here, before either side is sent. The dispatch decides it ahead of
    /// [`execute_sequential_pd`], so a refused request's prefill was never
    /// said to be sent. [`Self::refuses`] makes the same refusals without
    /// the plan: a refusal added here is added there.
    /// `language_model_reads_grids` is the model's answer the routing
    /// snapshot keeps (see [`sequential_pd_decode_form`]).
    pub(crate) fn decide(
        proto_request: &ProtoGenerateRequest,
        workers: &WorkerSelection,
        language_model_reads_grids: bool,
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
            KvConnectorMode::MoriIo => return Err(moriio_unsupported()),
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
        let decode_form = sequential_pd_decode_form(
            proto_request,
            decode_language_model_only,
            relay_kv_params,
            || solicits_identity(&mode, relay_kv_params),
            language_model_reads_grids,
        )?;
        let stripped_mm_decode = match decode_form {
            SequentialPdDecodeForm::TextPlusHashes => proto_request.has_mm_inputs(),
            SequentialPdDecodeForm::RefsThenIdentity => proto_request.has_vllm_media_refs(),
            SequentialPdDecodeForm::Full | SequentialPdDecodeForm::IdentityOnly => false,
        };
        Ok(Self {
            mode,
            relay_kv_params,
            mooncake_transfer_id,
            decode_form,
            stripped_mm_decode,
            streaming: proto_request.stream(),
        })
    }

    /// What [`Self::decide`] refuses of the pair and the request, asked
    /// without the plan: of each side's connector it asks one thing,
    /// whether it is MoRI-IO, and it mints no transfer id. It reads the
    /// prefill's connector mode only where the decode form asks for it (media
    /// references a language-model-only decode could take stripped), and
    /// logs no line but that reader's own warning of an invalid `dp_size`
    /// label. For a caller that asks ahead of the dispatch that decides the
    /// plan.
    /// `decide` makes the same two refusals in its own lines, which every
    /// sequential request runs.
    pub(crate) fn refuses(
        proto_request: &ProtoGenerateRequest,
        workers: &WorkerSelection,
        language_model_reads_grids: bool,
    ) -> Result<(), Response> {
        // The plan's mode is MoRI-IO for a MoRI-IO decode, and else what the
        // prefill's connector says.
        let moriio = [workers.decode_worker(), workers.prefill_worker()]
            .into_iter()
            .flatten()
            .any(|worker| is_moriio_worker(worker.as_ref()));
        if moriio {
            return Err(moriio_unsupported());
        }
        let relay_kv_params = proto_request.sampling_n() <= 1;
        let decode_language_model_only = proto_request.is_vllm()
            && workers
                .decode_worker()
                .is_some_and(|worker| worker_language_model_only(worker.as_ref()));
        sequential_pd_decode_form(
            proto_request,
            decode_language_model_only,
            relay_kv_params,
            || {
                workers.prefill_worker().is_some_and(|prefill| {
                    solicits_identity(
                        &connector_mode_for_worker(prefill.as_ref()),
                        relay_kv_params,
                    )
                })
            },
            language_model_reads_grids,
        )
        .map(drop)
    }

    /// The prefill request carried KV params, so the servicer was asked for
    /// the media identity.
    fn identity_solicited(&self) -> bool {
        solicits_identity(&self.mode, self.relay_kv_params)
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
            // The references stay on the stripped side until the prefill
            // side's media identity replaces them (`relay_to_decode`): the
            // identity's expanded ids and hashes are what the side is
            // stripped down to, so they must land before the strip.
            SequentialPdDecodeForm::RefsThenIdentity | SequentialPdDecodeForm::IdentityOnly => {
                proto_request.clone_without_mm_pixels()
            }
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
pub(crate) struct PrefillHarvest {
    stream: ProtoStream,
    kv_params: Option<String>,
    media_identity: Option<vllm::MediaIdentity>,
    input_logprobs: Option<ProtoInputLogProbs>,
    cached_tokens: PrefillCachedTokens,
    started: Instant,
    done: Instant,
}

/// What the sequential dispatch of one request holds for its decode, which
/// goes out behind its prefill. The decode's load guard was made ahead of
/// its send ([`LoadGuards::behind_prefill`]), so the dispatch owes it two
/// things once the prefill has answered: the decode asked again
/// ([`WorkerSelectionStage::decode_again`]), which may move the request to
/// another decode, and the send said on the decode the request goes to. A
/// decode the classic policies picked is owed neither, and its dispatch
/// holds nothing for it.
///
/// A batch's dispatch is handed the same for its decode guards, which were
/// made ahead of their send too: it asks no decode again, and says the send
/// once, behind what refuses a sub-request. A dispatch that refuses its
/// request drops this as it is, and the guards are dropped unsent.
pub(crate) struct DecodeBehindPrefill<'a> {
    /// The selection stage, which states the request again.
    pub stage: &'a WorkerSelectionStage,
    /// The model the request is routed under.
    pub model_id: &'a str,
    /// What of the request outlived its parse.
    pub routing: &'a RoutingSnapshot,
    pub headers: Option<&'a HeaderMap>,
    /// The retained plan's wire.
    pub wire: WireConstraint,
    /// The engines the request's failed attempts were sent to.
    pub tried: &'a [String],
    /// The attempt's guards, the decode's among them.
    pub guards: &'a mut LoadGuards,
    /// Where the dispatch leaves the selection that names the decode it
    /// sent, where the decode moved: the response's stages and a retry
    /// read it in the first selection's place.
    pub moved: &'a mut Option<WorkerSelection>,
}

/// Execute vLLM PD: send to prefill with max_tokens=1 first, wait for completion,
/// then send original request to decode.
///
/// For Mooncake: injects bootstrap_host/port from prefill worker metadata into
/// the decode request. For NIXL: tags the prefill request with do_remote_decode,
/// then relays the kv_transfer_params returned by the prefill engine to decode.
///
/// `plan` is the pair's plan as the dispatch decided it
/// ([`SequentialPlan::decide`]), and `prefill_guard` the prefill's load with
/// its send said: the dispatch says it behind the plan's refusals.
///
/// `behind` is what one request's dispatch holds for its decode
/// ([`DecodeBehindPrefill`]): once the prefill has answered the decode is
/// asked again, and its send is said where it goes out. `None` for a
/// dispatch that owes its decode neither: a batch's sub-request, whose
/// decode's send the batch or the guard's making said, and a pair the
/// classic policies picked.
#[expect(clippy::too_many_arguments)]
pub(crate) async fn execute_sequential_pd(
    plan: SequentialPlan,
    proto_request: ProtoGenerateRequest,
    clients: &mut ClientSelection,
    workers: &WorkerSelection,
    model: &str,
    prefill_guard: PrefillLoadGuard,
    ledger: &AttemptLedger,
    behind: Option<DecodeBehindPrefill<'_>>,
) -> Result<ExecutionResult, Response> {
    let runtime = workers
        .disaggregated_runtime_type()
        .map(|r| r.as_str())
        .unwrap_or("");
    let (prefill_client, decode_client) =
        clients.disaggregated_or_mismatch("execute_sequential_pd")?;

    // Recorded on the success path (after decode established) so failed
    // requests don't pollute success metrics; captured here before use of mode.
    let kv_connector_label = plan.mode.metrics_label();
    let (prefill_request, mut decode_request) = plan.requests(proto_request, workers);
    let sides = PdSides::of(workers, ledger, &prefill_request, plan.streaming);

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
        Box::pin({
            let mut client = prefill_client.clone();
            async move { client.generate(prefill_request).await }
        }),
        prefill_guard,
    )
    .await?;

    debug!("vLLM PD: prefill completed, sending decode request");

    // The prefill has answered: the decode is asked again, and the request
    // goes to the decode the answer names, with that decode's client and
    // its gate's claim.
    let mut decode_client = decode_client.clone();
    let mut workers = workers;
    let mut guards = None;
    if let Some(behind) = behind {
        let DecodeBehindPrefill {
            stage,
            model_id,
            routing,
            headers,
            wire,
            tried,
            guards: held,
            moved: slot,
        } = behind;
        if let Some(first) = workers.decode() {
            let request = RequestAgain {
                model_id,
                routing,
                headers,
                tokens: decode_request.token_ids(),
                wire,
                tried,
                ledger,
                claimed: held.claimed(),
            };
            let mut moved = None;
            held.map_decode(|ahead| {
                let (decode, guard) = stage.decode_again(&request, first, ahead);
                moved = decode;
                guard
            });
            if let Some(MovedDecode {
                decode,
                client,
                claim,
            }) = moved
            {
                held.claim_moved(claim);
                if let Some((_, held)) = clients.disaggregated_mut() {
                    *held = client.clone();
                }
                decode_client = client;
                *slot = workers.with_decode(decode);
                workers = slot.as_ref().unwrap_or(workers);
            }
        }
        guards = Some(held);
    }
    let sides = PdSides { workers, ..sides };

    let handed_off = relay_to_decode(
        &plan,
        workers,
        &mut decode_request,
        kv_params,
        media_identity.as_ref(),
    )?;
    // The decode goes out: its send is said on the answer the request goes
    // to, with the guard made ahead of it.
    if let (Some(guards), Some(decode)) = (guards, workers.decode()) {
        guards.map_decode(|ahead| decode.sent(ahead));
    }
    let (decode_stream, decode_sent) =
        sequential_decode_side(&sides, &plan, decode_request, decode_client, handed_off).await?;
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
pub(crate) async fn sequential_prefill_side(
    sides: &PdSides<'_>,
    dispatch: SideDispatch,
    prefill_guard: PrefillLoadGuard,
) -> Result<PrefillHarvest, Response> {
    let workers = sides.workers;
    debug!(
        request_id = %sides.request_id,
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
        WorkerMetrics::record_worker_request(prefill.as_ref());
    }
    // Booked before the call is polled, for the servicers that send their
    // headers with the first message (see `PrefillTicket::book`); the bound
    // above was read before the booking.
    let prefill_ticket =
        prefill_worker.and_then(|worker| PrefillTicket::book(worker, sides.prompt_tokens));
    let prefill_stream = dispatch_prefill_within(
        PrefillWait {
            bound: bound.clone(),
            encode: None,
        },
        dispatch,
        prefill_worker.cloned(),
        None,
        sides.prefill_label,
        sides.request_id.clone(),
        sides.prompt_tokens,
    )
    .await
    .map_err(|e| {
        tell_prefill_failure(&prefill_guard, &e);
        sequential_prefill_failure(&e, workers, sides.ledger, sides.prefill_label)
    })?;
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
        // The sequential protocol takes no encode watch: its request's
        // encode, if any, is the parallel shape's rendezvous.
        PrefillWait {
            bound,
            encode: None,
        },
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
        tell_prefill_failure(&prefill_guard, &e);
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
                // A collected answer does not charge the engine's own report,
                // as in the collector; a streamed answer charges it.
                if let Some(status) = e
                    .breaker_status()
                    .filter(|_| sides.streaming || !is_the_engines_own_report(&e))
                {
                    workers.record_outcome_prefill(sides.ledger, status);
                    tell_prefill_failure(&prefill_guard, &e);
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
                // An answer above the receive bound is as large from any
                // prefill worker: final, as the collector's drain makes it.
                if is_receive_bound_refusal(&e) {
                    mark_non_retryable(&mut response);
                }
                mark_error_reason(&mut response, e.error_reason());
                return Err(response);
            }
        }
    }
    prefill_stream.mark_completed();
    // Prefill is drained: it is told its good answer through its load, and
    // its admission slot is free while decode runs.
    prefill_guard.end().report(200, || true);
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
    match plan.decode_form {
        // A stripped references side has no reprocess fallback: its whole
        // multimodal payload is the prefill side's identity, so that identity
        // must land, and be stripped, before the side is sent.
        SequentialPdDecodeForm::RefsThenIdentity => {
            apply_refs_media_identity(decode_request, prefill_media_identity)?;
        }
        SequentialPdDecodeForm::Full
        | SequentialPdDecodeForm::IdentityOnly
        | SequentialPdDecodeForm::TextPlusHashes => apply_prefill_media_identity(
            decode_request,
            handed_off,
            relay_kv_params,
            plan.identity_solicited(),
            prefill_media_identity,
        ),
    }

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
        WorkerMetrics::record_worker_request(decode.as_ref());
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
    // path for the same invariant). A non-streaming request's first decode
    // response is its whole answer, so a client that left would hold the
    // decode engine for the answer's length (the house cap at most); its
    // wait is bounded instead by the pair's decode bound
    // (`liveness::decode_side_bound`, the bound a streaming side's first
    // token is judged by), taken as the estimate of the handoff's length
    // and counted from the drop: a shorter protection window than the cap,
    // by design. A plain side has no silence verdict of its own, and a
    // streaming side keeps the cap. The side is tracked on its worker here,
    // before its first item is waited for: its responses are the worker's
    // progress, it joins no pile (its silence is bounded per request), and
    // the worker's breaker success is sampled at the side's first response
    // (`ProtoStream::tracked_decode_side`), never at the dispatch; the
    // attribution stage leaves a side tracked here as it is.
    let plain_decode_bound = if plan.streaming {
        None
    } else {
        worker_prefilling_before_decode(workers, handed_off, plan.relay_kv_params)
            .and_then(|worker| liveness::decode_side_bound(worker, sides.prompt_tokens))
    };
    let decode_stream = match plain_decode_bound {
        Some(bound) => decode_stream.defer_abort_within(bound),
        None => decode_stream.defer_abort_until_first_item(),
    };
    let mut decode_stream = match &decode_worker {
        Some(worker) => decode_stream.tracked_decode_side(Arc::clone(worker)),
        None => decode_stream,
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

    use smg_worker::identity::basic::{
        ConnectionMode, RuntimeType, Worker, WorkerType, MOONCAKE_CONNECTOR, NIXL_CONNECTOR,
    };

    use super::*;
    use crate::{
        middleware::test_support::text_of,
        routers::{
            common::{attempt_ledger::FailedSide, retry::is_retryable_response},
            grpc::{
                backend_client::BasicWorkerBuilder,
                common::stages::request_execution::tests::{
                    answers_after, fails_now, failure, media_refs_request, never_answers,
                    tokenspeed_pair, vllm_pd_mm_request,
                },
                proto_wrapper::FanoutStream,
            },
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
            decode: decode.into(),
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
            sequential_pd_decode_form(&mm_request, false, true, || true, true)
                .expect("vision decode"),
            SequentialPdDecodeForm::IdentityOnly
        );
        assert_eq!(
            sequential_pd_decode_form(&mm_request, false, false, || true, true)
                .expect("vision decode n>1"),
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
            sequential_pd_decode_form(&stripable, true, true, || true, true)
                .expect("stripped side"),
            SequentialPdDecodeForm::TextPlusHashes
        );
        let text = ProtoGenerateRequest::Vllm(Box::default());
        assert_eq!(
            sequential_pd_decode_form(&text, true, true, || true, true).expect("text request"),
            SequentialPdDecodeForm::TextPlusHashes
        );
        assert_eq!(
            sequential_pd_decode_form(&text, true, false, || true, true).expect("text n>1"),
            SequentialPdDecodeForm::TextPlusHashes
        );
    }

    /// Grid tensors the model's language model does not read (MiniMax M3's
    /// feed its vision tower alone) leave a language-model-only decode
    /// worker the stripped side, as a request without grids does; a
    /// full-vision decode worker keeps its forms.
    #[test]
    fn grids_the_language_model_does_not_read_take_the_stripped_side() {
        let mut gridded = vllm_pd_mm_request();
        let ProtoGenerateRequest::Vllm(req) = &mut gridded else {
            panic!("expected vLLM request");
        };
        req.media_refs = None;
        assert!(gridded.has_vllm_mrope_grids());

        assert_eq!(
            sequential_pd_decode_form(&gridded, true, true, || true, false)
                .expect("plain-RoPE grids"),
            SequentialPdDecodeForm::TextPlusHashes
        );
        assert_eq!(
            sequential_pd_decode_form(&gridded, false, true, || true, false)
                .expect("vision decode"),
            SequentialPdDecodeForm::IdentityOnly
        );
        // n>1 recomputes the prompt on the decode side whatever the grids
        // are: still refused there, for its samples.
        let Err(refused) = sequential_pd_decode_form(&gridded, true, false, || true, false) else {
            panic!("n>1 cannot recompute on a pixel-less decode worker");
        };
        assert_eq!(
            refused
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .unwrap(),
            "pd_decode_language_model_only_n_samples"
        );
    }

    /// A single-sample media-references request of a model whose language
    /// model reads no grids, with the KV handoff relayed and the prefill side
    /// asked for its media identity, is served by a language-model-only
    /// decode worker: the references go to the prefill side alone, the
    /// decode side takes the identity. Without the identity asked for, or
    /// with grids the model reads, it is refused as before.
    #[test]
    fn sequential_pd_decode_form_serves_refs_on_a_language_model_only_decode() {
        let refs_only = media_refs_request("refs", 1);
        assert_eq!(
            sequential_pd_decode_form(&refs_only, true, true, || true, false)
                .expect("a single-sample refs request of a model that reads no grids"),
            SequentialPdDecodeForm::RefsThenIdentity
        );
        // A full-vision decode worker keeps its form.
        assert_eq!(
            sequential_pd_decode_form(&refs_only, false, true, || true, false)
                .expect("vision decode"),
            SequentialPdDecodeForm::IdentityOnly
        );
        for (name, identity_solicited, reads_grids) in [
            ("no identity asked for", false, false),
            ("grids the language model reads", true, true),
        ] {
            let Err(refused) = sequential_pd_decode_form(
                &refs_only,
                true,
                true,
                || identity_solicited,
                reads_grids,
            ) else {
                panic!("{name}: refused");
            };
            assert_eq!(
                refused
                    .headers()
                    .get(error::HEADER_X_SMG_ERROR_CODE)
                    .unwrap(),
                "pd_decode_language_model_only_media_refs",
                "{name}"
            );
        }
    }

    /// A prefill/decode pair as [`connector_pair`] builds it, its decode
    /// worker running as a language model only.
    fn language_model_only_decode_pair(
        connector: &str,
        engine_id: Option<&str>,
    ) -> WorkerSelection {
        let WorkerSelection::Disaggregated { prefill, .. } = connector_pair(connector, engine_id)
        else {
            panic!("a disaggregated pair");
        };
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:30000")
                .worker_type(WorkerType::Decode)
                .runtime_type(RuntimeType::Vllm)
                .connection_mode(ConnectionMode::Grpc)
                .label(smg_worker::labels::SUPPORTS_VISION_LABEL, "false")
                .build(),
        );
        WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill,
            decode: decode.into(),
            runtime_type: RuntimeType::Vllm,
        }
    }

    /// The plan asks the form with what the pair's connector gives: a NIXL
    /// handoff of a single sample, or a minted Mooncake one, asks the prefill
    /// for its media identity, so such a references request on a
    /// language-model-only decode worker is planned as the stripped
    /// references side, with no local recompute to fall back on. The same
    /// request keeps its refusal when the model's language model reads the
    /// grids, when it asks for several samples, and on a connector that asks
    /// for no identity. The check of the plan's refusals answers alike.
    #[tokio::test]
    async fn the_plan_serves_refs_on_a_language_model_only_decode_only_with_a_solicited_identity() {
        let nixl = language_model_only_decode_pair(NIXL_CONNECTOR, None);
        let plan = SequentialPlan::decide(&media_refs_request("refs-plan", 1), &nixl, false)
            .unwrap_or_else(|_| panic!("a single-sample refs request on a NIXL pair"));
        assert_eq!(plan.decode_form, SequentialPdDecodeForm::RefsThenIdentity);
        assert!(plan.identity_solicited());
        assert!(plan.stripped_mm_decode);

        let minted = language_model_only_decode_pair(MOONCAKE_CONNECTOR, Some("engine-a"));
        let plan = SequentialPlan::decide(&media_refs_request("refs-minted", 1), &minted, false)
            .unwrap_or_else(|_| panic!("a minted Mooncake handoff asks for the identity too"));
        assert_eq!(plan.decode_form, SequentialPdDecodeForm::RefsThenIdentity);
        assert!(
            SequentialPlan::refuses(&media_refs_request("refs-minted", 1), &minted, false).is_ok(),
            "the check reads the same connector"
        );

        let legacy = language_model_only_decode_pair(MOONCAKE_CONNECTOR, None);
        for (name, request, workers, reads_grids) in [
            (
                "the language model reads the grids",
                media_refs_request("refs-grids", 1),
                &nixl,
                true,
            ),
            (
                "several samples",
                media_refs_request("refs-n2", 2),
                &nixl,
                false,
            ),
            (
                "a legacy Mooncake handoff asks for no identity",
                media_refs_request("refs-legacy", 1),
                &legacy,
                false,
            ),
        ] {
            assert!(
                SequentialPlan::refuses(&request, workers, reads_grids).is_err(),
                "{name}: the check refuses it"
            );
            let Err(response) = SequentialPlan::decide(&request, workers, reads_grids) else {
                panic!("{name}: the pairing is refused");
            };
            assert_eq!(response.status(), http::StatusCode::BAD_REQUEST, "{name}");
            assert!(
                text_of(response)
                    .await
                    .contains("pd_decode_language_model_only_media_refs"),
                "{name}"
            );
        }
    }

    /// The plan of a references request served stripped on a
    /// language-model-only decode worker: NIXL, single sample, the identity
    /// asked for.
    fn refs_then_identity_plan() -> SequentialPlan {
        SequentialPlan {
            mode: KvConnectorMode::Nixl,
            relay_kv_params: true,
            mooncake_transfer_id: None,
            decode_form: SequentialPdDecodeForm::RefsThenIdentity,
            stripped_mm_decode: true,
            streaming: false,
        }
    }

    /// A language-model-only decode worker, references forwarded, n=1, the
    /// KV handoff relayed. The prefill side gets the references; when it
    /// answers, its media identity gives the decode side the expanded prompt
    /// ids, and the side is stripped to the per-item content hashes: no
    /// references, no pixels, no grids, and no per-request salt while the
    /// hashes key the cache.
    #[test]
    fn refs_then_identity_serves_decode_the_prefill_identity() {
        let plan = refs_then_identity_plan();
        let workers = language_model_only_decode_pair(NIXL_CONNECTOR, None);
        let (prefill, mut decode) = plan.requests(media_refs_request("refs-relay", 1), &workers);
        // The references go to the prefill side; the decode side holds them
        // only until the identity lands.
        assert!(
            prefill.has_vllm_media_refs(),
            "the prefill side expands them"
        );
        assert!(
            decode.has_vllm_media_refs(),
            "the decode side holds them until the identity lands"
        );
        let ProtoGenerateRequest::Vllm(prefill_req) = &prefill else {
            panic!("expected vLLM request");
        };
        assert_eq!(
            prefill_req.kv_transfer_params_json.as_deref(),
            Some(NIXL_PREFILL_KV_PARAMS),
            "the prefill side is tagged for the handoff and the identity"
        );

        let handed_off = relay_to_decode(
            &plan,
            &workers,
            &mut decode,
            Some("{\"do_remote_prefill\":true}".to_string()),
            Some(&identity()),
        )
        .unwrap_or_else(|_| panic!("the identity serves the stripped side"));
        assert!(handed_off);

        let ProtoGenerateRequest::Vllm(decode_req) = &decode else {
            panic!("expected vLLM request");
        };
        let Some(vllm::generate_request::Input::Tokenized(tokenized)) = &decode_req.input else {
            panic!("the decode side stays tokenized");
        };
        assert_eq!(tokenized.input_ids, vec![7, 8, 100, 100, 100, 9]);
        assert!(decode_req.media_refs.is_none());
        assert!(
            decode_req.cache_salt.is_none(),
            "the router sets no salt: the hashes key the decode prefix cache"
        );
        let mm = decode_req.mm_inputs.as_ref().expect("the hashes stay");
        assert_eq!(mm.mm_hashes, vec!["h1".to_string(), "h2".to_string()]);
        assert!(mm.pixel_values.is_none());
        assert!(mm.mm_placeholders.is_empty());
        assert!(mm.model_specific_tensors.is_empty());
    }

    /// A prefill that answers without a usable identity (none, or one
    /// without expanded ids) fails the request with a 502 the retry layer
    /// does not replay; the decode side is never stripped, so its references
    /// can never reach a language-model-only decode worker.
    #[test]
    fn a_refs_side_without_an_identity_fails_before_the_refs_could_be_sent() {
        let plan = refs_then_identity_plan();
        let workers = language_model_only_decode_pair(NIXL_CONNECTOR, None);
        let kv = Some("{\"do_remote_prefill\":true}".to_string());

        let (_, mut decode) = plan.requests(media_refs_request("refs-noid", 1), &workers);
        let Err(response) = relay_to_decode(&plan, &workers, &mut decode, kv.clone(), None) else {
            panic!("no identity: the request fails");
        };
        assert_eq!(response.status(), http::StatusCode::BAD_GATEWAY);
        assert_eq!(
            response
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .unwrap(),
            "pd_decode_missing_media_identity"
        );
        assert!(
            !is_retryable_response(&response),
            "an older prefill servicer would fail every replay the same way"
        );
        assert!(
            decode.has_vllm_media_refs(),
            "the references are never stripped without an identity"
        );

        let mut empty = identity();
        empty.prompt_token_ids.clear();
        let (_, mut decode) = plan.requests(media_refs_request("refs-empty", 1), &workers);
        assert!(
            relay_to_decode(&plan, &workers, &mut decode, kv, Some(&empty)).is_err(),
            "an identity without ids is unusable"
        );
        assert!(decode.has_vllm_media_refs());
    }

    /// The stripped references side has no local recompute to fall back on:
    /// an identity that lands without a KV handoff (a prefill that returned
    /// no transfer parameters) fails the request with a 502 the retry layer
    /// does not replay, after the references are gone from the side, which
    /// is never sent.
    #[test]
    fn a_stripped_refs_side_without_a_kv_handoff_is_not_sent() {
        let plan = refs_then_identity_plan();
        let workers = language_model_only_decode_pair(NIXL_CONNECTOR, None);
        let (_, mut decode) = plan.requests(media_refs_request("refs-nokv", 1), &workers);
        let Err(response) = relay_to_decode(&plan, &workers, &mut decode, None, Some(&identity()))
        else {
            panic!("no handoff to pull the prompt KV from");
        };
        assert_eq!(response.status(), http::StatusCode::BAD_GATEWAY);
        assert_eq!(
            response
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .unwrap(),
            "pd_decode_missing_kv_transfer_params"
        );
        assert!(
            !is_retryable_response(&response),
            "the prompt's KV is on that prefill alone"
        );
        assert!(
            !decode.has_vllm_media_refs(),
            "the references were stripped before the handoff was checked"
        );
    }

    /// An identity whose batches do not all carry content hashes cannot key
    /// the decode prefix cache by what the media was: the request fails with
    /// the missing-identity 502 the retry layer does not replay, whether no
    /// batch carries hashes or one of two does not, and the references are
    /// never stripped.
    #[test]
    fn an_identity_without_hashes_for_every_batch_fails_before_the_side_could_be_sent() {
        let without_hashes = || {
            let mut batch = identity().mm_inputs.expect("an image batch");
            batch.mm_hashes.clear();
            batch
        };
        let mut none_hashed = identity();
        none_hashed.mm_inputs = Some(without_hashes());
        let mut one_unhashed = identity();
        one_unhashed.extra_mm_inputs = vec![without_hashes()];
        for (name, identity) in [
            ("no batch carries hashes", none_hashed),
            ("one batch of two carries none", one_unhashed),
        ] {
            let plan = refs_then_identity_plan();
            let workers = language_model_only_decode_pair(NIXL_CONNECTOR, None);
            let (_, mut decode) = plan.requests(media_refs_request("refs-hashless", 1), &workers);
            let Err(response) = relay_to_decode(
                &plan,
                &workers,
                &mut decode,
                Some("{\"do_remote_prefill\":true}".to_string()),
                Some(&identity),
            ) else {
                panic!("{name}: the request fails");
            };
            assert_eq!(response.status(), http::StatusCode::BAD_GATEWAY, "{name}");
            assert_eq!(
                response
                    .headers()
                    .get(error::HEADER_X_SMG_ERROR_CODE)
                    .unwrap(),
                "pd_decode_missing_media_identity",
                "{name}"
            );
            assert!(!is_retryable_response(&response), "{name}");
            assert!(decode.has_vllm_media_refs(), "{name}: never stripped");
        }
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
        let response = sequential_pd_decode_form(&gridded, true, true, || true, true)
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
        let response = sequential_pd_decode_form(&refs_only, true, true, || true, true)
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
        let response = sequential_pd_decode_form(&fanout, true, false, || true, true)
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
            decode: decode.into(),
            runtime_type: RuntimeType::Vllm,
        }
    }

    /// The check of a plan's refusals answers as the plan's own decision
    /// does: the same refusal, by its code, for every pairing and request
    /// the decision refuses, and none where the decision makes a plan.
    #[test]
    fn the_check_of_a_plans_refusals_answers_as_the_plan_does() {
        use smg_worker::{identity::basic::MORIIO_CONNECTOR, labels::SUPPORTS_VISION_LABEL};

        let side = |url: &str, worker_type, connector: Option<&str>, vision: Option<&str>| {
            let mut worker = BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .runtime_type(RuntimeType::Vllm)
                .connection_mode(ConnectionMode::Grpc);
            if let Some(connector) = connector {
                worker = worker.kv_connector(connector);
            }
            if let Some(vision) = vision {
                worker = worker.label(SUPPORTS_VISION_LABEL, vision);
            }
            let worker: Arc<dyn Worker> = Arc::new(worker.build());
            worker
        };
        let pair = |prefill: Option<&str>, decode: Option<&str>, vision: Option<&str>| {
            WorkerSelection::Disaggregated {
                encode_assignments: None,
                prefill: side("grpc://prefill:30000", WorkerType::Prefill, prefill, None),
                decode: side("grpc://decode:30000", WorkerType::Decode, decode, vision).into(),
                runtime_type: RuntimeType::Vllm,
            }
        };
        let code = |refused: Response| {
            refused
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .and_then(|code| code.to_str().ok())
                .map(str::to_owned)
        };

        // A multimodal request in the three forms a language-model-only
        // decode is refused, as the decode form's own case builds them: grid
        // tensors (refused only while the model's language model reads
        // them), media references, and samples of a multimodal prompt.
        let mut gridded = vllm_pd_mm_request();
        let ProtoGenerateRequest::Vllm(req) = &mut gridded else {
            panic!("expected vLLM request");
        };
        req.media_refs = None;
        let mut refs_only = vllm_pd_mm_request();
        let ProtoGenerateRequest::Vllm(req) = &mut refs_only else {
            panic!("expected vLLM request");
        };
        req.mm_inputs = None;
        req.extra_mm_inputs.clear();
        // Pixels alone in one sample, which such a decode takes: the same
        // pixels in two samples are the third form it is refused.
        let mut pixels = vllm_pd_mm_request();
        let ProtoGenerateRequest::Vllm(req) = &mut pixels else {
            panic!("expected vLLM request");
        };
        req.media_refs = None;
        req.mm_inputs
            .as_mut()
            .unwrap()
            .model_specific_tensors
            .clear();
        req.sampling_params.get_or_insert_with(Default::default).n = 1;
        let mut fanout = pixels.clone();
        let ProtoGenerateRequest::Vllm(req) = &mut fanout else {
            panic!("expected vLLM request");
        };
        req.sampling_params.get_or_insert_with(Default::default).n = 2;
        let text = recorded_request(1);

        let nixl = Some(NIXL_CONNECTOR);
        let moriio = Some(MORIIO_CONNECTOR);
        let mooncake = Some(MOONCAKE_CONNECTOR);
        // Each row's code when the model's language model reads the grids,
        // and when it reads none of them: the grids row and the references
        // row of a pair that asks for the media identity tell the two apart.
        let rows = [
            ("a NIXL pair", pair(nixl, None, None), &text, None, None),
            (
                "a MoRI-IO decode",
                pair(nixl, moriio, None),
                &text,
                Some("moriio_grpc_pd_unsupported"),
                Some("moriio_grpc_pd_unsupported"),
            ),
            (
                "a MoRI-IO prefill",
                pair(moriio, None, None),
                &text,
                Some("moriio_grpc_pd_unsupported"),
                Some("moriio_grpc_pd_unsupported"),
            ),
            (
                "grids for a language-model-only decode",
                pair(nixl, None, Some("false")),
                &gridded,
                Some("pd_decode_language_model_only_mrope"),
                None,
            ),
            (
                "media references for a language-model-only decode",
                pair(nixl, None, Some("false")),
                &refs_only,
                Some("pd_decode_language_model_only_media_refs"),
                None,
            ),
            (
                "media references for a language-model-only decode on a legacy Mooncake pair",
                pair(mooncake, None, Some("false")),
                &refs_only,
                Some("pd_decode_language_model_only_media_refs"),
                Some("pd_decode_language_model_only_media_refs"),
            ),
            (
                "samples of a multimodal prompt for a language-model-only decode",
                pair(nixl, None, Some("false")),
                &fanout,
                Some("pd_decode_language_model_only_n_samples"),
                Some("pd_decode_language_model_only_n_samples"),
            ),
            (
                "one sample of a multimodal prompt for a language-model-only decode",
                pair(nixl, None, Some("false")),
                &pixels,
                None,
                None,
            ),
            (
                "text for a language-model-only decode",
                pair(nixl, None, Some("false")),
                &text,
                None,
                None,
            ),
            (
                "grids for a decode with the vision encoder",
                pair(nixl, None, Some("true")),
                &gridded,
                None,
                None,
            ),
        ];
        for (name, workers, request, if_read, if_not_read) in rows {
            for (reads, refused) in [(true, if_read), (false, if_not_read)] {
                // A refusal is read as one whatever it bears: a refusal with
                // no code is not a row's `None`.
                let refused = refused.map(|code| Some(code.to_owned()));
                let checked = SequentialPlan::refuses(request, &workers, reads)
                    .err()
                    .map(code);
                let decided = SequentialPlan::decide(request, &workers, reads)
                    .err()
                    .map(code);
                assert_eq!(checked, refused, "{name}, grids read {reads}: the check");
                assert_eq!(
                    decided, refused,
                    "{name}, grids read {reads}: the plan's decision"
                );
            }
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
        let plan =
            SequentialPlan::decide(&recorded_request(1), &workers, true).expect("a NIXL pair");
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
            SequentialPlan::decide(&recorded_request(1), &workers, true).expect("a Mooncake pair");
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
        let plan =
            SequentialPlan::decide(&recorded_request(1), &workers, true).expect("a Mooncake pair");
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
        let plan =
            SequentialPlan::decide(&recorded_request(2), &workers, true).expect("a NIXL pair");
        assert!(!plan.relay_kv_params, "n>1 relays no handoff");
        assert_eq!(
            sent_pair(&plan, recorded_request(2), &workers, None),
            (N2_PREFILL.to_string(), N2_DECODE.to_string())
        );
    }
}
