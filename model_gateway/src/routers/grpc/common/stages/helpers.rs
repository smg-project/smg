//! Common helper functions shared across stages

use std::sync::Arc;

use axum::response::Response;
use llm_tokenizer::traits::Tokenizer;
use rand::RngExt;
use smg_grpc_client::{
    mlx_proto,
    sglang_proto::{self, DisaggregatedParams},
    tokenspeed_proto, vllm_proto,
};
use tracing::{debug, error, warn};

use super::pd_protocol::{DpPlacement, PdProtocol, PdRendezvous};
use crate::{
    middleware::{RequestId, TenantRequestMeta},
    rate_limit::{ReservationAttachment, SharedReservationHandle},
    routers::{
        error,
        grpc::{
            context::{AttemptStamp, ExecutionPlan, LoadGuards, RequestType, WorkerSelection},
            proto_wrapper::ProtoGenerateRequest,
        },
    },
    worker::{
        sampling_defaults::SamplingDefaults, AttachedBody, Worker, DEFAULT_BOOTSTRAP_PORT,
        DEFAULT_SAMPLING_PARAMS_LABEL,
    },
};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SamplingDefaultsMask {
    temperature: bool,
    top_p: bool,
    top_k: bool,
    min_p: bool,
    repetition_penalty: bool,
}

impl SamplingDefaultsMask {
    /// Mask for a chat-shaped request: a knob left unset by the client is
    /// filled from model defaults. Shared by the chat endpoint and the
    /// transcription endpoint (whose backend request is chat-shaped).
    pub(crate) fn from_chat_request(
        request: &openai_protocol::chat::ChatCompletionRequest,
    ) -> Self {
        Self {
            temperature: request.temperature.is_none(),
            top_p: request.top_p.is_none(),
            top_k: request.top_k.is_none(),
            min_p: request.min_p.is_none(),
            repetition_penalty: request.repetition_penalty.is_none(),
        }
    }

    pub(crate) fn from_request_type(request_type: &RequestType) -> Option<Self> {
        match request_type {
            RequestType::Chat(request) => Some(Self::from_chat_request(request)),
            RequestType::Completion(request) => Some(Self {
                temperature: request.temperature.is_none(),
                top_p: request.top_p.is_none(),
                top_k: request.top_k.is_none(),
                min_p: request.min_p.is_none(),
                repetition_penalty: request.repetition_penalty.is_none(),
            }),
            RequestType::Generate(request) => {
                let params = request.sampling_params.as_ref();
                Some(Self {
                    temperature: params.and_then(|params| params.temperature).is_none(),
                    top_p: params.and_then(|params| params.top_p).is_none(),
                    top_k: params.and_then(|params| params.top_k).is_none(),
                    min_p: params.and_then(|params| params.min_p).is_none(),
                    repetition_penalty: params
                        .and_then(|params| params.repetition_penalty)
                        .is_none(),
                })
            }
            RequestType::Messages(request) => Some(Self {
                temperature: request.temperature.is_none(),
                top_p: request.top_p.is_none(),
                top_k: request.top_k.is_none(),
                // Messages does not expose these knobs, so model defaults are
                // the only source of request-level values for them.
                min_p: true,
                repetition_penalty: true,
            }),
            // Transcription builds its mask from the synthesized chat request
            // via `from_chat_request`, not this path.
            RequestType::Responses(_)
            | RequestType::Embedding(_)
            | RequestType::Classify(_)
            | RequestType::Transcription { .. } => None,
        }
    }

    fn any(self) -> bool {
        self.temperature || self.top_p || self.top_k || self.min_p || self.repetition_penalty
    }
}

/// Attach load guards and/or a rate-limit reservation to a streaming
/// response body so each survives (and, for the reservation, resolves via
/// `ReservationAttachment`'s `Drop`) exactly as long as the body does,
/// regardless of how the client disconnects. A no-op returning `response`
/// unchanged when both are `None`.
pub(crate) fn attach_response_guards(
    response: Response,
    guards: Option<LoadGuards>,
    reservation: Option<Arc<SharedReservationHandle>>,
) -> Response {
    match (guards, reservation) {
        (Some(guards), Some(handle)) => {
            AttachedBody::wrap_response(response, (guards, ReservationAttachment::new(handle)))
        }
        (Some(guards), None) => AttachedBody::wrap_response(response, guards),
        (None, Some(handle)) => {
            AttachedBody::wrap_response(response, ReservationAttachment::new(handle))
        }
        (None, None) => response,
    }
}

/// The middleware-assigned request id (client-sent via a configured header,
/// else generated; see `middleware/request_id.rs`), carried on the tenant
/// request metadata.
pub(crate) fn middleware_request_id(tenant_meta: Option<&TenantRequestMeta>) -> Option<&str> {
    tenant_meta
        .and_then(|meta| meta.extension::<RequestId>())
        .map(|request_id| request_id.0.as_str())
}

/// How each retry attempt re-mints the engine request id, captured at build
/// so replays reproduce exactly what a fresh build would have minted.
///
/// Engine ids must be unique per dispatch — a NIXL-tagged prefill keeps the
/// id alive until the KV lease expires, and responses tool loops re-execute
/// the pipeline per iteration — so every derived id gets a fresh
/// per-execution component. A bare `rid` outside PD is the one exception:
/// its value is the caller's contract (matching /generate's long-standing
/// behavior) and stays stable across attempts.
pub(crate) enum IdStamp {
    /// Bare client `rid` outside PD: stable across attempts.
    Exact,
    /// `{base}-{uuid}` minted fresh per attempt (rid under PD,
    /// middleware-derived ids).
    Suffixed { base: String },
    /// `{prefix}{uuid}` minted fresh per attempt.
    Minted { prefix: &'static str },
    /// Batched completion fan-out: shared id plus per-sub engine ids.
    Batch(BatchIdStamp),
}

pub(crate) struct BatchIdStamp {
    /// `None`: the shared id is minted fresh per attempt as `{prefix}{uuid}`.
    pub stable_shared: Option<String>,
    pub prefix: &'static str,
    /// Sub ids carry a per-execution uuid (PD, or a shared id that is stable
    /// across executions).
    pub unique_subs: bool,
}

/// Backend request id for any single-request endpoint, plus the stamp retry
/// attempts re-mint it with.
///
/// Priority: the protocol `rid`, else the middleware request id, else a fresh
/// `{prefix}{uuid}` (see [`IdStamp`] for the uniqueness rules).
pub(crate) fn resolve_request_id_stamp(
    request_type: &RequestType,
    tenant_meta: Option<&TenantRequestMeta>,
    prefix: &'static str,
    disaggregated: bool,
) -> (String, IdStamp) {
    if let Some(rid) = request_type.rid() {
        return if disaggregated {
            let stamp = IdStamp::Suffixed {
                base: rid.to_string(),
            };
            (format!("{rid}-{}", uuid::Uuid::now_v7()), stamp)
        } else {
            (rid.to_string(), IdStamp::Exact)
        };
    }
    match middleware_request_id(tenant_meta) {
        Some(request_id) => (
            format!("{request_id}-{}", uuid::Uuid::now_v7()),
            IdStamp::Suffixed {
                base: request_id.to_string(),
            },
        ),
        None => (
            format!("{prefix}{}", uuid::Uuid::now_v7()),
            IdStamp::Minted { prefix },
        ),
    }
}

/// Shared id + stamp for the batched completion fan-out. The shared id
/// (client rid or middleware request id) stays clean for the response;
/// per-sub engine ids get a uniqueness suffix in PD mode and whenever the
/// shared id is stable across executions (rid- or middleware-derived).
pub(crate) fn resolve_batch_id_stamp(
    request_type: &RequestType,
    tenant_meta: Option<&TenantRequestMeta>,
    prefix: &'static str,
    disaggregated: bool,
) -> (String, BatchIdStamp) {
    let (shared, stable_shared, unique_subs) = match request_type.rid() {
        Some(rid) => (rid.to_string(), Some(rid.to_string()), disaggregated),
        None => match middleware_request_id(tenant_meta) {
            Some(request_id) => (request_id.to_string(), Some(request_id.to_string()), true),
            None => (
                format!("{prefix}{}", uuid::Uuid::now_v7()),
                None,
                disaggregated,
            ),
        },
    };
    (
        shared,
        BatchIdStamp {
            stable_shared,
            prefix,
            unique_subs,
        },
    )
}

/// One batched sub-request's engine id.
pub(crate) fn batch_sub_id(shared: &str, index: usize, unique: bool) -> String {
    if unique {
        format!("{shared}-p{index}-{}", uuid::Uuid::now_v7())
    } else {
        format!("{shared}-p{index}")
    }
}

impl IdStamp {
    /// Re-mint the retained plan's engine id(s) for a retry attempt. A plan
    /// shape that doesn't match the stamp is a build-stage wiring bug: fail
    /// rather than re-dispatch the previous attempt's engine ids.
    pub(crate) fn restamp(&self, plan: &mut ExecutionPlan) -> Result<(), Response> {
        match self {
            Self::Exact => Ok(()),
            Self::Suffixed { base } => {
                plan.set_request_id(format!("{base}-{}", uuid::Uuid::now_v7()))
            }
            Self::Minted { prefix } => {
                plan.set_request_id(format!("{prefix}{}", uuid::Uuid::now_v7()))
            }
            Self::Batch(batch) => {
                let ExecutionPlan::Batch {
                    shared_request_id,
                    requests,
                    ..
                } = plan
                else {
                    error!(
                        function = "IdStamp::restamp",
                        "Batch id stamp on a non-batch plan"
                    );
                    return Err(error::internal_error(
                        "id_stamp_plan_mismatch",
                        "Id stamp does not match the plan shape",
                    ));
                };
                let shared = batch
                    .stable_shared
                    .clone()
                    .unwrap_or_else(|| format!("{}{}", batch.prefix, uuid::Uuid::now_v7()));
                for (i, request) in requests.iter_mut().enumerate() {
                    request.set_request_id(batch_sub_id(&shared, i, batch.unique_subs));
                }
                *shared_request_id = shared;
                Ok(())
            }
        }
    }
}

/// Pre-worker-default values of a plan's sampling params, captured at build
/// before the first worker's defaults are applied. Retry re-application
/// restores the masked fields from this baseline first, so attempt N's
/// effective params are exactly (request values) + (worker N's defaults) —
/// never contaminated by a previous attempt's worker.
#[derive(Clone)]
pub(crate) enum SamplingBaseline {
    Sglang(sglang_proto::SamplingParams),
    Vllm(vllm_proto::SamplingParams),
    Mlx(mlx_proto::SamplingParams),
    TokenSpeed(tokenspeed_proto::SamplingParams),
}

impl SamplingBaseline {
    fn capture(request: &ProtoGenerateRequest) -> Option<Self> {
        match request {
            ProtoGenerateRequest::Sglang(req) => req.sampling_params.clone().map(Self::Sglang),
            ProtoGenerateRequest::Vllm(req) => req.sampling_params.clone().map(Self::Vllm),
            ProtoGenerateRequest::Mlx(req) => req.sampling_params.clone().map(Self::Mlx),
            ProtoGenerateRequest::TokenSpeed(req) => {
                req.sampling_params.clone().map(Self::TokenSpeed)
            }
            ProtoGenerateRequest::Trtllm(_) => None,
        }
    }

    /// Reset the masked fields to their pre-default values.
    fn restore_masked(
        &self,
        request: &mut ProtoGenerateRequest,
        mask: SamplingDefaultsMask,
    ) -> Result<(), Response> {
        macro_rules! restore {
            ($params:expr, $baseline:expr) => {{
                let params = $params;
                let baseline = $baseline;
                if mask.temperature {
                    params.temperature = baseline.temperature;
                }
                if mask.top_p {
                    params.top_p = baseline.top_p;
                }
                if mask.top_k {
                    params.top_k = baseline.top_k;
                }
                if mask.min_p {
                    params.min_p = baseline.min_p;
                }
                if mask.repetition_penalty {
                    params.repetition_penalty = baseline.repetition_penalty;
                }
            }};
        }
        match (request, self) {
            (ProtoGenerateRequest::Sglang(req), Self::Sglang(baseline)) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    restore!(params, baseline);
                }
                Ok(())
            }
            (ProtoGenerateRequest::Vllm(req), Self::Vllm(baseline)) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    restore!(params, baseline);
                }
                Ok(())
            }
            (ProtoGenerateRequest::Mlx(req), Self::Mlx(baseline)) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    restore!(params, baseline);
                }
                Ok(())
            }
            (ProtoGenerateRequest::TokenSpeed(req), Self::TokenSpeed(baseline)) => {
                if let Some(params) = req.sampling_params.as_mut() {
                    restore!(params, baseline);
                }
                Ok(())
            }
            _ => {
                error!(
                    function = "SamplingBaseline::restore_masked",
                    "Sampling baseline does not match the plan's backend"
                );
                Err(error::internal_error(
                    "sampling_baseline_plan_mismatch",
                    "Sampling baseline does not match the plan shape",
                ))
            }
        }
    }
}

/// Re-stamp the retained plan for a retry attempt's newly selected workers:
/// fresh engine ids per [`IdStamp`], the new worker's sampling defaults
/// (re-applied over the build-time baseline), and fresh PD
/// bootstrap/rendezvous rooms. EPD encode bootstrap info is deliberately
/// untouched — those rooms are the rendezvous with encode jobs the first
/// dispatch already launched.
pub(crate) fn restamp_plan_for_attempt(
    plan: &mut ExecutionPlan,
    stamp: &AttemptStamp,
    workers: &WorkerSelection,
) -> Result<(), Response> {
    stamp.id.restamp(plan)?;
    for request in plan.generate_requests_mut() {
        if let (Some(mask), Some(baseline)) = (stamp.sampling_mask, &stamp.sampling_baseline) {
            baseline.restore_masked(request, mask)?;
            apply_sampling_defaults_with_mask(request, mask, Some(workers));
        }
        if stamp.inject_pd_metadata {
            maybe_inject_pd_metadata(request, workers);
        }
        maybe_inject_pd_rendezvous(request, workers);
    }
    Ok(())
}

/// Decode selected-worker sampling defaults from labels.
///
/// In PD mode the decode worker is authoritative because it produces visible
/// output tokens. The resolved request is then sent through the existing PD
/// flow unchanged.
pub(crate) fn sampling_defaults_for_request(
    workers: Option<&WorkerSelection>,
) -> Option<SamplingDefaults> {
    let worker = match workers? {
        WorkerSelection::Single { worker } => worker,
        WorkerSelection::Disaggregated { decode, .. } => decode,
    };
    let json = worker
        .metadata()
        .spec
        .labels
        .get(DEFAULT_SAMPLING_PARAMS_LABEL)?;

    match SamplingDefaults::from_json_str(json) {
        Ok(defaults) => defaults,
        Err(e) => {
            warn!(
                worker_url = %worker.url(),
                error = %e,
                "Ignoring invalid default sampling params label"
            );
            None
        }
    }
}

/// Apply the selected worker's sampling defaults at build, capturing the
/// masked fields' pre-default baseline first: retry attempts re-apply the
/// new worker's defaults over this baseline, not over the previous worker's.
pub(crate) fn apply_sampling_defaults(
    request: &mut ProtoGenerateRequest,
    mask: Option<SamplingDefaultsMask>,
    workers: Option<&WorkerSelection>,
) -> Option<SamplingBaseline> {
    let mask = mask?;
    if !mask.any() || matches!(request, ProtoGenerateRequest::Trtllm(_)) {
        return None;
    }
    let baseline = SamplingBaseline::capture(request);
    apply_sampling_defaults_with_mask(request, mask, workers);
    baseline
}

/// Apply model sampling defaults to a built proto request.
///
/// The proto already contains backend fallback values, so `mask` (derived
/// from the request at build) selects only fields the user did not set.
/// Retry attempts reuse the same mask against the newly selected worker.
fn apply_sampling_defaults_with_mask(
    request: &mut ProtoGenerateRequest,
    mask: SamplingDefaultsMask,
    workers: Option<&WorkerSelection>,
) {
    if matches!(request, ProtoGenerateRequest::Trtllm(_)) {
        return;
    }

    if !mask.any() {
        return;
    }

    let Some(defaults) = sampling_defaults_for_request(workers) else {
        return;
    };

    match request {
        ProtoGenerateRequest::Sglang(req) => {
            let Some(params) = req.sampling_params.as_mut() else {
                warn!("Cannot apply sampling defaults to SGLang request without sampling_params");
                return;
            };
            apply_sglang_sampling_defaults(params, defaults, mask);
        }
        ProtoGenerateRequest::Vllm(req) => {
            let Some(params) = req.sampling_params.as_mut() else {
                warn!("Cannot apply sampling defaults to vLLM request without sampling_params");
                return;
            };
            apply_vllm_sampling_defaults(params, defaults, mask);
        }
        ProtoGenerateRequest::Mlx(req) => {
            let Some(params) = req.sampling_params.as_mut() else {
                warn!("Cannot apply sampling defaults to MLX request without sampling_params");
                return;
            };
            apply_mlx_sampling_defaults(params, defaults, mask);
        }
        ProtoGenerateRequest::TokenSpeed(req) => {
            let Some(params) = req.sampling_params.as_mut() else {
                warn!(
                    "Cannot apply sampling defaults to TokenSpeed request without sampling_params"
                );
                return;
            };
            apply_tokenspeed_sampling_defaults(params, defaults, mask);
        }
        ProtoGenerateRequest::Trtllm(_) => {}
    }
}

macro_rules! apply_numeric_default {
    ($params:expr, $defaults:expr, $mask:expr, $field:ident) => {
        if $mask.$field {
            if let Some(value) = $defaults.$field {
                $params.$field = value;
            }
        }
    };
}

macro_rules! apply_unsigned_top_k_default {
    ($params:expr, $defaults:expr, $mask:expr) => {
        if $mask.top_k {
            if let Some(value) = $defaults.top_k {
                $params.top_k = value.max(0) as u32;
            }
        }
    };
}

macro_rules! optional_temperature_sampling_defaults_fn {
    ($fn_name:ident, $params_ty:path) => {
        fn $fn_name(
            params: &mut $params_ty,
            defaults: SamplingDefaults,
            mask: SamplingDefaultsMask,
        ) {
            if mask.temperature {
                if let Some(value) = defaults.temperature {
                    params.temperature = Some(value);
                }
            }
            apply_numeric_default!(params, defaults, mask, top_p);
            apply_unsigned_top_k_default!(params, defaults, mask);
            apply_numeric_default!(params, defaults, mask, min_p);
            apply_numeric_default!(params, defaults, mask, repetition_penalty);
        }
    };
}

fn apply_sglang_sampling_defaults(
    params: &mut sglang_proto::SamplingParams,
    defaults: SamplingDefaults,
    mask: SamplingDefaultsMask,
) {
    apply_numeric_default!(params, defaults, mask, temperature);
    apply_numeric_default!(params, defaults, mask, top_p);
    apply_numeric_default!(params, defaults, mask, top_k);
    apply_numeric_default!(params, defaults, mask, min_p);
    apply_numeric_default!(params, defaults, mask, repetition_penalty);
}

optional_temperature_sampling_defaults_fn!(
    apply_vllm_sampling_defaults,
    vllm_proto::SamplingParams
);
optional_temperature_sampling_defaults_fn!(apply_mlx_sampling_defaults, mlx_proto::SamplingParams);

/// TokenSpeed declares every sampling scalar as `optional` so the servicer
/// can distinguish "client set 0" from "client unset". Apply defaults by
/// writing `Some(value)` rather than the bare value.
fn apply_tokenspeed_sampling_defaults(
    params: &mut tokenspeed_proto::SamplingParams,
    defaults: SamplingDefaults,
    mask: SamplingDefaultsMask,
) {
    macro_rules! apply_opt {
        ($field:ident) => {
            if mask.$field {
                if let Some(value) = defaults.$field {
                    params.$field = Some(value);
                }
            }
        };
    }
    apply_opt!(temperature);
    apply_opt!(top_p);
    apply_opt!(top_k);
    apply_opt!(min_p);
    apply_opt!(repetition_penalty);
}

/// Convert single-token stop strings into `stop_token_ids` entries so the engine
/// can halt generation early for the common case (e.g. `["."]`, `["\n"]`).
///
/// The proto `stop_token_ids` field is a flat list of single token ids, so a
/// multi-token stop string cannot be represented there — pushing its sub-tokens
/// would stop far too eagerly (on any one of them). Multi-token, empty, and
/// unknown stops are therefore left to the router-side `StopSequenceDecoder`,
/// which detokenizes worker output and trims the stop text. Existing
/// `stop_token_ids` are preserved and deduped.
fn encode_single_token_stops(
    stops: Vec<String>,
    stop_token_ids: &mut Vec<u32>,
    tokenizer: Option<&Arc<dyn Tokenizer>>,
) {
    // Without a tokenizer we cannot encode (not expected on paths that resolve
    // one to tokenize the prompt). Safe: the strings are already dropped by the
    // caller, so the router-side decoder remains the source of truth.
    let Some(tokenizer) = tokenizer else {
        if !stops.is_empty() {
            warn!(
                "No tokenizer available to encode string stop sequences; \
                 relying on router-side stop decoder only"
            );
        }
        return;
    };

    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        // add_special_tokens=false: we want the literal token(s) for the stop
        // string, not a BOS/EOS-wrapped encoding.
        match tokenizer.encode(&stop, false) {
            Ok(encoding) => match encoding.token_ids() {
                [id] => {
                    if !stop_token_ids.contains(id) {
                        stop_token_ids.push(*id);
                    }
                }
                ids => debug!(
                    stop = %stop,
                    token_count = ids.len(),
                    "string stop is not single-token; handled by router-side stop decoder"
                ),
            },
            Err(e) => warn!(
                stop = %stop,
                error = %e,
                "Failed to encode string stop sequence; relying on router-side stop decoder"
            ),
        }
    }
}

/// Router-authoritative string-`stop` resolution for backends whose engine
/// cannot match string stops itself.
///
/// vLLM over gRPC detokenizes server-side (`detokenize=bool(stop)`), TRT-LLM
/// tokenizes stop words server-side, and MLX has no string-`stop` field — those
/// keep their strings untouched. Two paths cannot:
///   - SGLang gRPC workers run with `skip_tokenizer_init=True` and reject string
///     stops outright (a 400 for any request carrying `stop`); and
///   - every direct-ZMQ backend (vLLM EngineCore, TokenSpeed) receives token ids
///     only, so the engine never sees — and cannot match — a stop string.
///
/// For both, the router owns the tokenizer and already matches string stops via
/// `StopSequenceDecoder` (it detokenizes worker output and trims), so the worker
/// never needs the raw strings. This drops the string `stop` list and forwards
/// any single-token stop as a `stop_token_ids` entry for early stopping; the
/// router-side decoder handles the rest. This is the single resolution point
/// shared by SGLang gRPC and every ZMQ backend.
///
/// Returns the stop strings that were stripped — the router's residual
/// obligation: the engine will never match these, so response processing must
/// trim them from the output text. Empty when the engine matches server-side.
pub(crate) fn resolve_string_stops(
    request: &mut ProtoGenerateRequest,
    tokenizer: Option<&Arc<dyn Tokenizer>>,
    token_only_wire: bool,
) -> Vec<String> {
    // SGLang always needs it; the vLLM and TokenSpeed protos only when talking
    // to a token-only wire (direct-ZMQ, the sole path either reaches that way).
    match request {
        ProtoGenerateRequest::Sglang(req) => {
            if let Some(params) = req.sampling_params.as_mut() {
                let stops = std::mem::take(&mut params.stop);
                encode_single_token_stops(stops.clone(), &mut params.stop_token_ids, tokenizer);
                return stops;
            }
        }
        ProtoGenerateRequest::Vllm(req) if token_only_wire => {
            // EOS injection for the tokenizer-less EngineCore is the ZMQ
            // client's own policy (zmq_client::fold_tokenizer_eos_backstop),
            // not part of shared stop resolution.
            if let Some(params) = req.sampling_params.as_mut() {
                let stops = std::mem::take(&mut params.stop);
                encode_single_token_stops(stops.clone(), &mut params.stop_token_ids, tokenizer);
                return stops;
            }
        }
        ProtoGenerateRequest::TokenSpeed(req) if token_only_wire => {
            // TokenSpeed over ZMQ receives token ids only, so its wire
            // translation drops raw `stop` strings; without this a single-token
            // user stop would never reach the engine as a `stop_token_ids`
            // entry. Resolve it exactly as the other token-only backends. No
            // EOS fold here: unlike vLLM EngineCore, the TokenSpeed scheduler
            // stops at EOS itself, so its translation carries no frontend ids.
            if let Some(params) = req.sampling_params.as_mut() {
                let stops = std::mem::take(&mut params.stop);
                encode_single_token_stops(stops.clone(), &mut params.stop_token_ids, tokenizer);
                return stops;
            }
        }
        _ => {}
    }
    Vec::new()
}

/// Inject PD bootstrap metadata when the runtime's rendezvous travels as
/// SGLang-style `DisaggregatedParams` (bootstrap host/port/room).
///
/// vLLM kv_transfer_params are handled in the request_execution stage.
pub(crate) fn maybe_inject_pd_metadata(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
) {
    if let WorkerSelection::Disaggregated {
        prefill,
        runtime_type,
        ..
    } = workers
    {
        let rendezvous = PdProtocol::for_runtime(*runtime_type).map(|p| p.rendezvous);
        if rendezvous == Some(PdRendezvous::SglangBootstrap) {
            inject_sglang_bootstrap_metadata(request, prefill);
        }
    }
}

/// Inject bootstrap metadata into a SGLang gRPC request.
fn inject_sglang_bootstrap_metadata(
    request: &mut ProtoGenerateRequest,
    prefill_worker: &Arc<dyn Worker>,
) {
    let metadata = prefill_worker.metadata();
    let hostname = metadata.bootstrap_host();
    let bootstrap_port = metadata.bootstrap_port().unwrap_or(DEFAULT_BOOTSTRAP_PORT);
    let room_id = rand::rng().random_range(0..i32::MAX);

    let disagg_params = DisaggregatedParams {
        bootstrap_host: hostname.to_string(),
        bootstrap_port: bootstrap_port as i32,
        bootstrap_room: room_id,
    };

    // Guarded by the caller's runtime check, but match defensively: a non-SGLang
    // proto here (e.g. a ZMQ backend reporting an unexpected runtime) must not
    // take down the request task via the panicking accessor.
    let ProtoGenerateRequest::Sglang(sglang_request) = request else {
        warn!("PD bootstrap metadata requested for a non-SGLang request; skipping injection");
        return;
    };
    sglang_request.disaggregated_params = Some(disagg_params);

    debug!(
        "Injected bootstrap metadata: host={}, port={}, room={}",
        hostname, bootstrap_port, room_id
    );
}

/// Inject prefill->decode rendezvous params for backends that carry them in the
/// generate request.
///
/// The gateway mints one room per request and sends identical params to both the
/// prefill and decode worker (`execute_parallel_pd` clones the request after
/// this stage). Host/port name the PREFILL worker's Mooncake bootstrap server
/// (the KV data source); the decode worker discovers it there by `bootstrap_room`.
/// This KV leg is independent of any per-item encode->prefill bootstrap info.
pub(crate) fn maybe_inject_pd_rendezvous(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
) {
    // The KV bootstrap leg is identical for plain PD and EPD; EPD just layers
    // encode assignments on the disaggregated worker selection.
    let (prefill, runtime_type) = match workers {
        WorkerSelection::Disaggregated {
            prefill,
            runtime_type,
            ..
        } => (prefill, runtime_type),
        WorkerSelection::Single { .. } => return,
    };
    let Some(protocol) = PdProtocol::for_runtime(*runtime_type) else {
        return;
    };
    if protocol.rendezvous == PdRendezvous::KvBootstrapRoom {
        inject_kv_bootstrap_room(request, prefill, protocol.dp_placement);
    }
}

/// Inject the KV bootstrap host/port/room into a generate request.
fn inject_kv_bootstrap_room(
    request: &mut ProtoGenerateRequest,
    prefill_worker: &Arc<dyn Worker>,
    dp_placement: DpPlacement,
) {
    let metadata = prefill_worker.metadata();
    let hostname = metadata.bootstrap_host();
    let bootstrap_port = metadata.bootstrap_port().unwrap_or(DEFAULT_BOOTSTRAP_PORT);
    // 63-bit room: no dedup, keep the space wide so the birthday collision
    // rate stays negligible. See the proto field doc.
    //
    // Under `DpPlacement::RoomResidue`, mint the room congruent to the
    // dp-aware prefill worker's rank: the engine dispatches by
    // `room % dp_size`, so the residue is the placement carrier. The low
    // bits therefore carry structure — do not shard on them elsewhere. Under
    // round_robin the decode side follows the same residue and deliberately
    // inherits the prefill placement (decode prefix reuse shrinks the KV
    // transfer).
    let room_id = match (
        dp_placement,
        prefill_worker.dp_rank(),
        prefill_worker.dp_size(),
    ) {
        (DpPlacement::RoomResidue, Some(rank), Some(dp)) if dp > 1 && rank < dp => {
            let dp = dp as i64;
            let base = rand::rng().random_range(0..i64::MAX - dp);
            base - (base % dp) + rank as i64
        }
        _ => rand::rng().random_range(0..i64::MAX),
    };

    request.set_kv_bootstrap_info(hostname.to_string(), bootstrap_port as i32, room_id);

    debug!(
        "Injected PD rendezvous: host={}, port={}, room={}",
        hostname, bootstrap_port, room_id
    );
}

#[cfg(test)]
mod request_id_tests {
    use std::sync::Arc;

    use openai_protocol::chat::ChatCompletionRequest;

    use super::*;
    use crate::tenant::TenantKey;

    fn chat_request_type(rid: Option<&str>) -> RequestType {
        RequestType::Chat(Arc::new(ChatCompletionRequest {
            rid: rid.map(str::to_string),
            ..Default::default()
        }))
    }

    fn meta_with_request_id(id: &str) -> TenantRequestMeta {
        TenantRequestMeta::new(TenantKey::new("test-tenant"))
            .with_extension(RequestId(id.to_string()))
    }

    fn single_plan(request_id: &str) -> ExecutionPlan {
        use smg_grpc_client::sglang_proto;

        use crate::routers::grpc::proto_wrapper::ProtoRequest;
        let request = ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
            request_id: request_id.to_string(),
            ..Default::default()
        }));
        ExecutionPlan::Single(ProtoRequest::Generate(request))
    }

    fn batch_plan(shared: &str, sub_ids: &[String]) -> ExecutionPlan {
        use smg_grpc_client::sglang_proto;

        use crate::routers::grpc::context::ExecutionPlanKind;
        ExecutionPlan::Batch {
            kind: ExecutionPlanKind::Single,
            shared_request_id: shared.to_string(),
            requests: sub_ids
                .iter()
                .map(|id| {
                    ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
                        request_id: id.clone(),
                        ..Default::default()
                    }))
                })
                .collect(),
        }
    }

    fn plan_ids(plan: &ExecutionPlan) -> (String, Vec<String>) {
        match plan {
            ExecutionPlan::Batch {
                shared_request_id,
                requests,
                ..
            } => (
                shared_request_id.clone(),
                requests
                    .iter()
                    .map(|r| r.request_id().to_string())
                    .collect(),
            ),
            other => (other.request_id().to_string(), Vec::new()),
        }
    }

    #[test]
    fn rid_is_used_exactly_outside_pd_and_stays_stable_on_restamp() {
        let request_type = chat_request_type(Some("client-rid"));
        let meta = meta_with_request_id("chatcmpl-mw");

        let (id, stamp) = resolve_request_id_stamp(&request_type, Some(&meta), "chatcmpl-", false);
        assert_eq!(id, "client-rid");

        let mut plan = single_plan(&id);
        stamp.restamp(&mut plan).unwrap();
        assert_eq!(
            plan.request_id(),
            "client-rid",
            "rid is the caller's contract"
        );
    }

    #[test]
    fn rid_gets_per_attempt_suffix_in_pd() {
        let request_type = chat_request_type(Some("client-rid"));

        let (id, stamp) = resolve_request_id_stamp(&request_type, None, "chatcmpl-", true);
        assert!(id.starts_with("client-rid-") && id != "client-rid");

        let mut plan = single_plan(&id);
        stamp.restamp(&mut plan).unwrap();
        let restamped = plan.request_id().to_string();
        assert!(restamped.starts_with("client-rid-"));
        assert_ne!(restamped, id, "each attempt gets a fresh engine id");
    }

    #[test]
    fn middleware_id_is_base_with_per_execution_suffix() {
        let request_type = chat_request_type(None);
        let meta = meta_with_request_id("chatcmpl-mw");

        let (first, stamp) =
            resolve_request_id_stamp(&request_type, Some(&meta), "chatcmpl-", false);
        assert!(first.starts_with("chatcmpl-mw-"));

        let mut plan = single_plan(&first);
        stamp.restamp(&mut plan).unwrap();
        let second = plan.request_id().to_string();
        assert!(second.starts_with("chatcmpl-mw-"));
        assert_ne!(first, second);
    }

    #[test]
    fn falls_back_to_prefixed_mint_without_rid_or_middleware_id() {
        let request_type = chat_request_type(None);
        let meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

        let (id, stamp) = resolve_request_id_stamp(&request_type, Some(&meta), "chatcmpl-", false);
        assert!(id.starts_with("chatcmpl-"));

        let mut plan = single_plan(&id);
        stamp.restamp(&mut plan).unwrap();
        assert!(plan.request_id().starts_with("chatcmpl-"));
        assert_ne!(plan.request_id(), id, "minted ids are fresh per attempt");
    }

    #[test]
    fn batch_rid_subs_stay_stable_outside_pd() {
        let request_type = chat_request_type(Some("client-rid"));
        let (shared, stamp) = resolve_batch_id_stamp(&request_type, None, "cmpl_", false);
        assert_eq!(shared, "client-rid");
        assert!(!stamp.unique_subs);

        let subs: Vec<String> = (0..2).map(|i| batch_sub_id(&shared, i, false)).collect();
        assert_eq!(subs, ["client-rid-p0", "client-rid-p1"]);

        let mut plan = batch_plan(&shared, &subs);
        IdStamp::Batch(stamp).restamp(&mut plan).unwrap();
        assert_eq!(plan_ids(&plan), (shared, subs), "rid batch ids are stable");
    }

    #[test]
    fn batch_middleware_subs_are_unique_per_attempt_under_stable_shared() {
        let request_type = chat_request_type(None);
        let meta = meta_with_request_id("mw-id");
        let (shared, stamp) = resolve_batch_id_stamp(&request_type, Some(&meta), "cmpl_", false);
        assert_eq!(shared, "mw-id");
        assert!(stamp.unique_subs);

        let subs: Vec<String> = (0..2)
            .map(|i| batch_sub_id(&shared, i, stamp.unique_subs))
            .collect();
        let mut plan = batch_plan(&shared, &subs);
        IdStamp::Batch(stamp).restamp(&mut plan).unwrap();
        let (restamped_shared, restamped_subs) = plan_ids(&plan);
        assert_eq!(restamped_shared, "mw-id");
        for (i, (before, after)) in subs.iter().zip(&restamped_subs).enumerate() {
            assert!(after.starts_with(&format!("mw-id-p{i}-")));
            assert_ne!(before, after, "sub engine ids must be fresh per attempt");
        }
    }

    #[test]
    fn batch_minted_shared_is_fresh_per_attempt() {
        let request_type = chat_request_type(None);
        let (shared, stamp) = resolve_batch_id_stamp(&request_type, None, "cmpl_", false);
        assert!(shared.starts_with("cmpl_"));
        assert!(stamp.stable_shared.is_none());

        let subs: Vec<String> = (0..2).map(|i| batch_sub_id(&shared, i, false)).collect();
        let mut plan = batch_plan(&shared, &subs);
        IdStamp::Batch(stamp).restamp(&mut plan).unwrap();
        let (restamped_shared, restamped_subs) = plan_ids(&plan);
        assert!(restamped_shared.starts_with("cmpl_"));
        assert_ne!(restamped_shared, shared);
        assert_eq!(
            restamped_subs,
            [
                format!("{restamped_shared}-p0"),
                format!("{restamped_shared}-p1")
            ]
        );
    }
}

#[cfg(test)]
mod stop_resolution_tests {
    use std::sync::Arc;

    use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer};
    use smg_grpc_client::{sglang_proto, tokenspeed_proto, vllm_proto};

    use super::{resolve_string_stops, ProtoGenerateRequest};

    fn mock_tokenizer() -> Arc<dyn Tokenizer> {
        // MockTokenizer vocab: "." => 6, "Hello" => 1, "world" => 2. `encode`
        // splits on whitespace, so "." => [6] (single) and "Hello world" =>
        // [1, 2] (multi); unknown words encode to [].
        Arc::new(MockTokenizer::new())
    }

    fn sglang_request(stop: Vec<&str>, stop_token_ids: Vec<u32>) -> ProtoGenerateRequest {
        ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
            sampling_params: Some(sglang_proto::SamplingParams {
                stop: stop.into_iter().map(str::to_string).collect(),
                stop_token_ids,
                ..Default::default()
            }),
            ..Default::default()
        }))
    }

    fn vllm_request(stop: Vec<&str>, stop_token_ids: Vec<u32>) -> ProtoGenerateRequest {
        ProtoGenerateRequest::Vllm(Box::new(vllm_proto::GenerateRequest {
            sampling_params: Some(vllm_proto::SamplingParams {
                stop: stop.into_iter().map(str::to_string).collect(),
                stop_token_ids,
                ..Default::default()
            }),
            ..Default::default()
        }))
    }

    fn tokenspeed_request(stop: Vec<&str>, stop_token_ids: Vec<u32>) -> ProtoGenerateRequest {
        ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
            sampling_params: Some(tokenspeed_proto::SamplingParams {
                stop: stop.into_iter().map(str::to_string).collect(),
                stop_token_ids,
                ..Default::default()
            }),
            ..Default::default()
        }))
    }

    fn tokenspeed_params(req: &ProtoGenerateRequest) -> &tokenspeed_proto::SamplingParams {
        match req {
            ProtoGenerateRequest::TokenSpeed(r) => r.sampling_params.as_ref().unwrap(),
            _ => panic!("expected TokenSpeed request"),
        }
    }

    fn sglang_params(req: &ProtoGenerateRequest) -> &sglang_proto::SamplingParams {
        match req {
            ProtoGenerateRequest::Sglang(r) => r.sampling_params.as_ref().unwrap(),
            _ => panic!("expected SGLang request"),
        }
    }

    fn vllm_params(req: &ProtoGenerateRequest) -> &vllm_proto::SamplingParams {
        match req {
            ProtoGenerateRequest::Vllm(r) => r.sampling_params.as_ref().unwrap(),
            _ => panic!("expected vLLM request"),
        }
    }

    #[test]
    fn sglang_single_token_becomes_stop_token_id() {
        let mut req = sglang_request(vec!["."], vec![]);
        resolve_string_stops(&mut req, Some(&mock_tokenizer()), false);

        let params = sglang_params(&req);
        assert!(params.stop.is_empty(), "string stop should be cleared");
        assert_eq!(params.stop_token_ids, vec![6]);
    }

    #[test]
    fn sglang_multi_token_relies_on_router_decoder() {
        // "Hello world" => [1, 2]: can't be a flat stop_token_id, so it must not
        // be forwarded (would over-eagerly stop on any subtoken).
        let mut req = sglang_request(vec!["Hello world"], vec![]);
        resolve_string_stops(&mut req, Some(&mock_tokenizer()), false);

        let params = sglang_params(&req);
        assert!(params.stop.is_empty());
        assert!(params.stop_token_ids.is_empty());
    }

    #[test]
    fn sglang_mixed_only_single_token_forwarded_and_dedups() {
        let mut req = sglang_request(vec![".", "Hello world"], vec![6, 42]);
        resolve_string_stops(&mut req, Some(&mock_tokenizer()), false);

        let params = sglang_params(&req);
        assert!(params.stop.is_empty());
        assert_eq!(
            params.stop_token_ids,
            vec![6, 42],
            "existing ids kept, no dup"
        );
    }

    #[test]
    fn sglang_without_tokenizer_still_clears_strings() {
        let mut req = sglang_request(vec!["."], vec![]);
        resolve_string_stops(&mut req, None, false);

        let params = sglang_params(&req);
        assert!(
            params.stop.is_empty(),
            "strings dropped so worker won't 400"
        );
        assert!(params.stop_token_ids.is_empty());
    }

    #[test]
    fn vllm_resolved_only_over_zmq() {
        // gRPC vLLM keeps its strings (the servicer detokenizes engine-side).
        let mut grpc = vllm_request(vec!["."], vec![]);
        resolve_string_stops(&mut grpc, Some(&mock_tokenizer()), false);
        let params = vllm_params(&grpc);
        assert_eq!(
            params.stop,
            vec![".".to_string()],
            "gRPC vLLM stop preserved"
        );
        assert!(params.stop_token_ids.is_empty());

        // ZMQ vLLM (EngineCore sees token ids only) resolves like SGLang.
        // EOS injection is the ZMQ client's own step, not stop resolution's
        // (see zmq_client::fold_tokenizer_eos_backstop tests).
        let mut zmq = vllm_request(vec!["."], vec![]);
        resolve_string_stops(&mut zmq, Some(&mock_tokenizer()), true);
        let params = vllm_params(&zmq);
        assert!(params.stop.is_empty(), "ZMQ vLLM stop cleared");
        assert_eq!(params.stop_token_ids, vec![6]);
    }

    #[test]
    fn noop_when_no_string_stops() {
        let mut req = sglang_request(vec![], vec![7]);
        resolve_string_stops(&mut req, Some(&mock_tokenizer()), false);

        let params = sglang_params(&req);
        assert_eq!(params.stop_token_ids, vec![7], "unrelated ids untouched");
    }

    #[test]
    fn tokenspeed_resolved_only_over_zmq() {
        // A gRPC TokenSpeed request is never produced, but guard the gate: the
        // strings must survive when is_zmq is false.
        let mut grpc = tokenspeed_request(vec!["."], vec![]);
        resolve_string_stops(&mut grpc, Some(&mock_tokenizer()), false);
        let params = tokenspeed_params(&grpc);
        assert_eq!(params.stop, vec![".".to_string()], "non-zmq stop preserved");
        assert!(params.stop_token_ids.is_empty());

        // Over ZMQ the token-only wire drops raw strings, so a single-token stop
        // must ride as a stop_token_ids entry instead.
        let mut zmq = tokenspeed_request(vec!["."], vec![]);
        resolve_string_stops(&mut zmq, Some(&mock_tokenizer()), true);
        let params = tokenspeed_params(&zmq);
        assert!(params.stop.is_empty(), "ZMQ TokenSpeed stop cleared");
        assert_eq!(params.stop_token_ids, vec![6]);
    }

    #[test]
    fn tokenspeed_zmq_does_not_fold_eos() {
        // Unlike vLLM EngineCore, the TokenSpeed scheduler stops at EOS itself,
        // so resolution must not append the tokenizer's EOS ids (999).
        let mut req = tokenspeed_request(vec!["."], vec![]);
        resolve_string_stops(&mut req, Some(&mock_tokenizer()), true);
        assert_eq!(
            tokenspeed_params(&req).stop_token_ids,
            vec![6],
            "only the single-token stop, no EOS fold"
        );
    }

    #[test]
    fn resolution_returns_router_obligations() {
        // Strings stripped for the engine come back as the router's trim duty.
        let mut req = sglang_request(vec![".", "Hello world"], vec![]);
        let obligations = resolve_string_stops(&mut req, Some(&mock_tokenizer()), false);
        assert_eq!(
            obligations,
            vec![".".to_string(), "Hello world".to_string()]
        );

        // gRPC vLLM matches stops server-side: nothing left for the router.
        let mut req = vllm_request(vec!["."], vec![]);
        assert!(resolve_string_stops(&mut req, Some(&mock_tokenizer()), false).is_empty());
    }

    #[test]
    fn pd_bootstrap_injection_skips_non_sglang_requests() {
        use super::{Worker, WorkerSelection};
        use crate::worker::{BasicWorkerBuilder, RuntimeType, WorkerType};

        // An SGLang-runtime worker selection paired with a non-SGLang proto
        // (e.g. a misreporting backend) must skip injection, not panic.
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://prefill:30000")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        let selection = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: worker.clone(),
            decode: worker,
            runtime_type: RuntimeType::Sglang,
        };

        let mut req = vllm_request(vec!["."], vec![7]);
        let before = match &req {
            ProtoGenerateRequest::Vllm(inner) => (**inner).clone(),
            _ => panic!("vllm_request builds a Vllm variant"),
        };
        super::maybe_inject_pd_metadata(&mut req, &selection);
        match &req {
            ProtoGenerateRequest::Vllm(inner) => {
                assert_eq!(**inner, before, "request must be untouched");
            }
            _ => panic!("variant must be unchanged"),
        }
    }

    #[test]
    fn tokenspeed_zmq_multi_token_relies_on_router_decoder() {
        // "Hello world" => [1, 2]: not a flat stop id, so it must not forward.
        let mut req = tokenspeed_request(vec!["Hello world"], vec![42]);
        resolve_string_stops(&mut req, Some(&mock_tokenizer()), true);
        let params = tokenspeed_params(&req);
        assert!(params.stop.is_empty());
        assert_eq!(params.stop_token_ids, vec![42], "existing ids kept");
    }
}

#[cfg(test)]
mod restamp_shape_mismatch_tests {
    use std::sync::Arc;

    use openai_protocol::chat::ChatCompletionRequest;
    use smg_grpc_client::sglang_proto;

    use super::*;
    use crate::routers::{
        error::extract_error_code_from_response,
        grpc::{context::ExecutionPlanKind, proto_wrapper::ProtoRequest},
    };

    fn single_plan() -> ExecutionPlan {
        ExecutionPlan::Single(ProtoRequest::Generate(ProtoGenerateRequest::Sglang(
            Box::default(),
        )))
    }

    fn batch_plan() -> ExecutionPlan {
        ExecutionPlan::Batch {
            kind: ExecutionPlanKind::Single,
            shared_request_id: "cmpl_shared".to_string(),
            requests: vec![ProtoGenerateRequest::Sglang(Box::default())],
        }
    }

    /// A stamp/plan shape mismatch is a build-stage wiring bug; a warn-and-
    /// continue would re-dispatch the previous attempt's engine ids.
    #[test]
    fn id_stamp_shape_mismatch_is_an_error() {
        let batch_stamp = IdStamp::Batch(BatchIdStamp {
            stable_shared: None,
            prefix: "cmpl_",
            unique_subs: false,
        });
        let response = batch_stamp
            .restamp(&mut single_plan())
            .expect_err("batch stamp on a single plan must fail");
        assert_eq!(
            extract_error_code_from_response(&response),
            "id_stamp_plan_mismatch"
        );

        let minted_stamp = IdStamp::Minted { prefix: "cmpl_" };
        let response = minted_stamp
            .restamp(&mut batch_plan())
            .expect_err("single stamp on a batch plan must fail");
        assert_eq!(
            extract_error_code_from_response(&response),
            "id_stamp_plan_mismatch"
        );
    }

    /// A baseline captured from one backend flavor cannot restore another.
    #[test]
    fn sampling_baseline_backend_mismatch_is_an_error() {
        let baseline = SamplingBaseline::Vllm(vllm_proto::SamplingParams::default());
        let mut request = ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
            sampling_params: Some(sglang_proto::SamplingParams::default()),
            ..Default::default()
        }));
        let mask = SamplingDefaultsMask::from_request_type(&RequestType::Chat(Arc::new(
            ChatCompletionRequest::default(),
        )))
        .expect("chat requests always carry a mask");
        let response = baseline
            .restore_masked(&mut request, mask)
            .expect_err("cross-backend baseline must fail");
        assert_eq!(
            extract_error_code_from_response(&response),
            "sampling_baseline_plan_mismatch"
        );
    }
}

#[cfg(test)]
mod sampling_restamp_tests {
    use std::sync::Arc;

    use openai_protocol::chat::ChatCompletionRequest;
    use smg_grpc_client::sglang_proto;

    use super::*;
    use crate::{
        routers::grpc::{context::ExecutionPlanKind, proto_wrapper::ProtoRequest},
        worker::{BasicWorkerBuilder, WorkerType},
    };

    fn worker_with_defaults(url: &str, defaults_json: Option<&str>) -> WorkerSelection {
        let mut builder = BasicWorkerBuilder::new(url).worker_type(WorkerType::Regular);
        if let Some(json) = defaults_json {
            builder = builder.label(DEFAULT_SAMPLING_PARAMS_LABEL, json);
        }
        WorkerSelection::Single {
            worker: Arc::new(builder.build()),
        }
    }

    fn sglang_request() -> ProtoGenerateRequest {
        ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
            request_id: "sampling-restamp".to_string(),
            sampling_params: Some(sglang_proto::SamplingParams {
                temperature: 1.0,
                top_k: 3,
                ..Default::default()
            }),
            ..Default::default()
        }))
    }

    fn sglang_params(plan: &ExecutionPlan) -> &sglang_proto::SamplingParams {
        match plan {
            ExecutionPlan::Single(ProtoRequest::Generate(ProtoGenerateRequest::Sglang(req))) => {
                req.sampling_params.as_ref().unwrap()
            }
            _ => panic!("expected single SGLang plan"),
        }
    }

    /// Attempt N's effective sampling params must be exactly (request values)
    /// + (worker N's defaults). A retry worker that omits a field the first
    /// worker defaulted must fall back to the build-time baseline, never to
    /// the first worker's value.
    #[test]
    fn retry_reapplies_defaults_from_the_baseline_not_the_previous_worker() {
        let mask = SamplingDefaultsMask::from_request_type(&RequestType::Chat(Arc::new(
            ChatCompletionRequest::default(),
        )))
        .expect("chat requests always carry a mask");

        let worker_a = worker_with_defaults(
            "grpc://worker-a:30000",
            Some(r#"{"temperature":0.5,"top_k":7}"#),
        );
        let mut request = sglang_request();
        let baseline = apply_sampling_defaults(&mut request, Some(mask), Some(&worker_a));
        let stamp = AttemptStamp {
            id: IdStamp::Exact,
            sampling_mask: Some(mask),
            sampling_baseline: baseline,
            inject_pd_metadata: false,
        };
        let mut plan = ExecutionPlan::generate(ExecutionPlanKind::Single, request);
        {
            let params = sglang_params(&plan);
            assert_eq!(params.temperature, 0.5, "worker A's default applied");
            assert_eq!(params.top_k, 7);
        }

        // Worker B omits temperature: it must return to the baseline (1.0),
        // not keep worker A's 0.5.
        let worker_b = worker_with_defaults("grpc://worker-b:30000", Some(r#"{"top_k":9}"#));
        restamp_plan_for_attempt(&mut plan, &stamp, &worker_b).unwrap();
        {
            let params = sglang_params(&plan);
            assert_eq!(params.temperature, 1.0, "baseline restored, not worker A's");
            assert_eq!(params.top_k, 9, "worker B's default applied");
        }

        // Worker C has no defaults label at all: everything returns to the
        // baseline.
        let worker_c = worker_with_defaults("grpc://worker-c:30000", None);
        restamp_plan_for_attempt(&mut plan, &stamp, &worker_c).unwrap();
        {
            let params = sglang_params(&plan);
            assert_eq!(params.temperature, 1.0);
            assert_eq!(params.top_k, 3);
        }
    }
}

#[cfg(test)]
mod epd_restamp_tests {
    use std::sync::Arc;

    use smg_grpc_client::tokenspeed_proto;

    use super::*;
    use crate::{
        routers::grpc::{context::ExecutionPlanKind, proto_wrapper::EncodeItemBootstrapInfo},
        worker::{BasicWorkerBuilder, RuntimeType, Worker, WorkerType},
    };

    fn tokenspeed_pd_pair(prefill_url: &str, decode_url: &str) -> WorkerSelection {
        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(prefill_url)
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(decode_url)
                .worker_type(WorkerType::Decode)
                .build(),
        );
        WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill,
            decode,
            runtime_type: RuntimeType::TokenSpeed,
        }
    }

    fn tokenspeed_rendezvous(
        plan: &ExecutionPlan,
    ) -> (
        &tokenspeed_proto::EncodeBootstrapInfo,
        &tokenspeed_proto::KvBootstrapInfo,
    ) {
        match plan {
            ExecutionPlan::EncodePrefillDecode {
                request: ProtoGenerateRequest::TokenSpeed(req),
            } => (
                req.encode_bootstrap_info.as_ref().unwrap(),
                req.kv_bootstrap_info.as_ref().unwrap(),
            ),
            _ => panic!("expected a TokenSpeed EPD plan"),
        }
    }

    /// An EPD retry re-selects only the prefill/decode pair. The PD KV room
    /// must be re-minted for the new prefill, while the encode rooms must
    /// survive untouched — the first attempt's encode jobs are already
    /// parked at those rendezvous points, so re-minting them would strand
    /// the embeddings.
    #[test]
    fn epd_restamp_remints_pd_room_and_keeps_encode_rooms() {
        let mut request = ProtoGenerateRequest::TokenSpeed(Box::default());
        request.set_encode_bootstrap_info(vec![EncodeItemBootstrapInfo {
            item_index: 0,
            bootstrap_host: "encode-worker".to_string(),
            bootstrap_port: 9000,
            bootstrap_room: 41,
        }]);
        request.set_kv_bootstrap_info("attempt1-prefill".to_string(), 8998, 42);
        let mut plan = ExecutionPlan::generate(ExecutionPlanKind::EncodePrefillDecode, request);

        let stamp = AttemptStamp {
            id: IdStamp::Exact,
            sampling_mask: None,
            sampling_baseline: None,
            inject_pd_metadata: false,
        };
        let retry_pair = tokenspeed_pd_pair("grpc://retry-prefill:30000", "grpc://decode:30000");
        restamp_plan_for_attempt(&mut plan, &stamp, &retry_pair).unwrap();

        let (encode, kv) = tokenspeed_rendezvous(&plan);
        assert_eq!(
            (
                encode.items[0].bootstrap_host.as_str(),
                encode.items[0].bootstrap_room
            ),
            ("encode-worker", 41),
            "encode rendezvous must survive the retry"
        );
        assert_ne!(kv.bootstrap_room, 42, "PD KV room is re-minted per attempt");
        let new_prefill_host = match &retry_pair {
            WorkerSelection::Disaggregated { prefill, .. } => {
                prefill.metadata().bootstrap_host().to_string()
            }
            WorkerSelection::Single { .. } => {
                panic!("tokenspeed_pd_pair builds a disaggregated selection")
            }
        };
        assert_eq!(
            kv.bootstrap_host, new_prefill_host,
            "KV rendezvous names the newly selected prefill"
        );
    }
}
