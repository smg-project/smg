//! Common helper functions shared across stages

use std::sync::Arc;

use llm_tokenizer::traits::Tokenizer;
use rand::RngExt;
use smg_grpc_client::{
    mlx_proto,
    sglang_proto::{self, DisaggregatedParams},
    tokenspeed_proto, vllm_proto,
};
use tracing::{debug, warn};

use super::pd_protocol::{DpPlacement, PdProtocol, PdRendezvous};
use crate::{
    middleware::{RequestId, TenantRequestMeta},
    rate_limit::{ReservationAttachment, SharedReservationHandle},
    routers::grpc::{
        context::{LoadGuards, RequestType, WorkerSelection},
        proto_wrapper::ProtoGenerateRequest,
    },
    worker::{
        sampling_defaults::SamplingDefaults, AttachedBody, Worker, DEFAULT_BOOTSTRAP_PORT,
        DEFAULT_SAMPLING_PARAMS_LABEL,
    },
};

#[derive(Clone, Copy, Debug, Default)]
struct SamplingDefaultsMask {
    temperature: bool,
    top_p: bool,
    top_k: bool,
    min_p: bool,
    repetition_penalty: bool,
}

impl SamplingDefaultsMask {
    fn from_request_type(request_type: &RequestType) -> Option<Self> {
        match request_type {
            RequestType::Chat(request) => Some(Self {
                temperature: request.temperature.is_none(),
                top_p: request.top_p.is_none(),
                top_k: request.top_k.is_none(),
                min_p: request.min_p.is_none(),
                repetition_penalty: request.repetition_penalty.is_none(),
            }),
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
            RequestType::Responses(_) | RequestType::Embedding(_) | RequestType::Classify(_) => {
                None
            }
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
    response: axum::response::Response,
    guards: Option<LoadGuards>,
    reservation: Option<Arc<SharedReservationHandle>>,
) -> axum::response::Response {
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

/// Backend request id for any endpoint.
///
/// Priority: the protocol `rid`, else the middleware request id, else a fresh
/// `{prefix}{uuid}`.
///
/// Engine ids must be unique per dispatch — PD retries re-run request
/// building while a NIXL-tagged prefill keeps the id alive until the KV lease
/// expires, and responses tool loops re-execute the pipeline per iteration —
/// so middleware-derived ids always get a per-execution suffix. A bare `rid`
/// outside PD is used exactly: its value is the caller's contract (matching
/// /generate's long-standing behavior).
pub(crate) fn resolve_request_id(
    request_type: &RequestType,
    tenant_meta: Option<&TenantRequestMeta>,
    prefix: &str,
    disaggregated: bool,
) -> String {
    if let Some(rid) = request_type.rid() {
        return if disaggregated {
            format!("{rid}-{}", uuid::Uuid::now_v7())
        } else {
            rid.to_string()
        };
    }
    match middleware_request_id(tenant_meta) {
        Some(request_id) => format!("{request_id}-{}", uuid::Uuid::now_v7()),
        None => format!("{prefix}{}", uuid::Uuid::now_v7()),
    }
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

/// Apply model sampling defaults to a built proto request.
///
/// The proto already contains backend fallback values, so `request_type` is
/// used only as an omission mask: defaults fill fields the user did not set.
pub(crate) fn apply_sampling_defaults_to_generate_request(
    request: &mut ProtoGenerateRequest,
    request_type: &RequestType,
    workers: Option<&WorkerSelection>,
) {
    if matches!(request, ProtoGenerateRequest::Trtllm(_)) {
        return;
    }

    let Some(mask) = SamplingDefaultsMask::from_request_type(request_type) else {
        return;
    };
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

    #[test]
    fn rid_is_used_exactly_outside_pd() {
        let request_type = chat_request_type(Some("client-rid"));
        let meta = meta_with_request_id("chatcmpl-mw");

        let id = resolve_request_id(&request_type, Some(&meta), "chatcmpl-", false);
        assert_eq!(id, "client-rid");
    }

    #[test]
    fn rid_gets_per_attempt_suffix_in_pd() {
        let request_type = chat_request_type(Some("client-rid"));

        let id = resolve_request_id(&request_type, None, "chatcmpl-", true);
        assert!(id.starts_with("client-rid-") && id != "client-rid");
    }

    #[test]
    fn middleware_id_is_base_with_per_execution_suffix() {
        let request_type = chat_request_type(None);
        let meta = meta_with_request_id("chatcmpl-mw");

        let first = resolve_request_id(&request_type, Some(&meta), "chatcmpl-", false);
        let second = resolve_request_id(&request_type, Some(&meta), "chatcmpl-", false);
        assert!(first.starts_with("chatcmpl-mw-"));
        assert!(second.starts_with("chatcmpl-mw-"));
        assert_ne!(first, second);
    }

    #[test]
    fn falls_back_to_prefixed_mint_without_rid_or_middleware_id() {
        let request_type = chat_request_type(None);
        let meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

        let id = resolve_request_id(&request_type, Some(&meta), "chatcmpl-", false);
        assert!(id.starts_with("chatcmpl-"));
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
