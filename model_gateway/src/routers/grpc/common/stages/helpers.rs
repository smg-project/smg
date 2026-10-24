//! Common helper functions shared across stages

use std::{fmt, sync::Arc};

use axum::response::Response;
use engine_zmq_adapter::stops::encode_single_token_stops;
use llm_tokenizer::traits::Tokenizer;
use rand::RngExt;
use smg_grpc_client::{
    mlx_proto,
    sglang_proto::{self, DisaggregatedParams},
    tokenspeed_proto, vllm_proto,
};
use tool_parser::{GrammarStart, ParserFactory as ToolParserFactory};
use tracing::{debug, error, warn};

use super::pd_protocol::{DpPlacement, PdProtocol, PdRendezvous};
use crate::{
    middleware::{RequestId, TenantRequestMeta},
    routers::{
        error,
        grpc::{
            backend_client::BackendClient,
            context::{
                stage_output, AttemptStamp, ClientSelection, EncodeOutputs, ExecutionPlan,
                RequestContext, RequestType, WorkerSelection,
            },
            multimodal::{
                assemble_media_refs, assemble_multimodal_data,
                assemble_multimodal_data_after_encode, assembly_error_response,
                reserve_multimodal_inflight,
            },
            proto_wrapper::{
                AnswerConstraint, MultimodalData, ProtoGenerateRequest, SglangMultimodalData,
            },
            utils,
        },
    },
    worker::{
        sampling_defaults::SamplingDefaults, Worker, DEFAULT_BOOTSTRAP_PORT,
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
            | RequestType::Decisions(_)
            | RequestType::Transcription { .. } => None,
        }
    }

    fn any(self) -> bool {
        self.temperature || self.top_p || self.top_k || self.min_p || self.repetition_penalty
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

/// What a tokenized `Generate` carries as `original_text` next to its ids.
///
/// Workers generate from the ids. The text is read only by the worker-side
/// media processors, which hand it on as the prompt's text next to the ids,
/// so a request that carries media keeps it; a text-only request sends the
/// ids alone. The rendered text of a 1M-token prompt is about 4 MB: next to
/// its ~2 MB of ids it tripled every Generate and stayed resident in the
/// router for the life of each in-flight request.
pub(crate) fn wire_prompt_text(rendered_prompt: String, carries_media: bool) -> String {
    if carries_media {
        rendered_prompt
    } else {
        String::new()
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

/// A structural tag the table derives for where a worker's engine starts the
/// grammar (`regular::symphony::AutoTag`, `regular::symphony::AnswerTag`): the
/// attempt stamp asks it for the worker each retry takes, as request building
/// asked it for the first.
pub(crate) trait TagForStart: fmt::Debug + Send + Sync {
    /// The tag for an engine whose grammar starts at `start`, or `None` where
    /// the table gives that engine no tag.
    fn tag_for(&self, start: GrammarStart) -> Result<Option<String>, String>;

    /// The request's own JSON schema, where the tag is a JSON answer's: the
    /// constraint a worker the table gives no tag takes in its place, the
    /// engine's own, serialized only when a request takes it; `None` where
    /// no tag leaves the request unconstrained (the `auto` tag).
    fn schema(&self) -> Option<&serde_json::Value>;
}

/// A request's tool constraint as each attempt's worker must take it. Request
/// building gives the first attempt's worker the shape its engine needs; a
/// retry gives the worker it selects the same treatment, since a tag already
/// shaped for one engine cannot be shaped for another: the registry's tags
/// from the unshaped form preparation produced, the table's from what derives
/// them for a start.
#[derive(Clone, Debug)]
pub(crate) enum ToolConstraintStamp {
    /// A forced tool call's constraint as preparation built it (the older
    /// registry's, or the table's tag for the start the model's workers name
    /// together), with the name of the tool parser that shapes the registry's
    /// (`utils::tool_constraints_for_worker`).
    Registry {
        /// The `(kind, value)` pair preparation produced, before any worker shaped it.
        constraint: (String, String),
        /// The configured tool parser the shaping reads its reasoning prefix from.
        parser: Option<String>,
        /// The table built the tag for an engine that starts the grammar at
        /// the first token: the shaping leaves such a tag as it is, so every
        /// retry says so with it (`regular::symphony::Prepared::Constraint`).
        first_token: bool,
    },
    /// The table's tag the selected worker took, shaped for where its engine
    /// starts the grammar: the `auto` tag its policy chose, or a JSON
    /// answer's; the retry's worker gets it derived again for its own start.
    /// Where the table gives that engine none, the `auto` tag leaves the
    /// request unconstrained, and a JSON answer's gives way to the request's
    /// own schema, the engine's constraint ([`TagForStart::schema`]), from
    /// which a later retry takes the tag back. An `auto` tag is captured only
    /// where the first worker got a tag: a plan built without one carries no
    /// stamp, so a retry's worker the table would give one goes without it
    /// too.
    Table(Arc<dyn TagForStart>),
}

impl ToolConstraintStamp {
    /// Capture the stamp for a built request that carries the forced call's
    /// structural tag. Nothing is captured for a JSON-schema constraint (no
    /// shape to follow) or when the backend's builder let the request's
    /// `response_format` outrank the tool constraint (the request then
    /// carries a schema, grammar or regex, never a structural tag).
    /// `first_token` says the table built the tag for an engine that starts
    /// the grammar at the first token.
    pub(crate) fn capture(
        unshaped: Option<(String, String)>,
        parser: Option<&str>,
        first_token: bool,
        request: &ProtoGenerateRequest,
    ) -> Option<Self> {
        let constraint = unshaped.filter(|(kind, _)| kind == "structural_tag")?;
        request.structural_tag()?;
        Some(Self::Registry {
            constraint,
            parser: parser.map(str::to_string),
            first_token,
        })
    }

    /// Capture the stamp for a built request that carries the table's tag,
    /// with what derives the tag for a worker's start; for a JSON answer's, a
    /// request that carries the answer's schema instead, the engine's own
    /// where the table gave the first worker no tag ([`TagForStart::schema`]).
    /// Nothing is captured when the backend's builder let the request's
    /// `response_format` outrank an `auto` tag, as for [`Self::capture`].
    pub(crate) fn capture_table(
        tag: Arc<dyn TagForStart>,
        request: &ProtoGenerateRequest,
    ) -> Option<Self> {
        let carried = request.structural_tag().is_some()
            || (request.json_schema().is_some() && tag.schema().is_some());
        if !carried {
            return None;
        }
        Some(Self::Table(tag))
    }

    /// Write the constraint, shaped for `workers`, over the request's
    /// structural tag; a table's tag the retry's worker gets none of leaves
    /// the request unconstrained, as a first attempt on that worker would be.
    fn restamp(
        &self,
        request: &mut ProtoGenerateRequest,
        workers: &WorkerSelection,
        tool_parser_factory: &ToolParserFactory,
    ) -> Result<(), Response> {
        let failed = |e: String| {
            error!(
                function = "restamp_plan_for_attempt",
                error = %e,
                "Failed to shape the tool constraint for the retry's worker"
            );
            error::internal_error(
                "invalid_tool_constraint",
                format!("Invalid tool constraint: {e}"),
            )
        };
        // Whether the shape for this worker leads with the thought for an engine
        // that starts the grammar at the first token, or the table built it so:
        // the request says so where its dialect has a switch for the gate
        // behind a reasoning parser, as the first attempt's build did
        // (`GenerateRequestBuildOptions`).
        let first_token = match self {
            Self::Registry {
                constraint,
                parser,
                first_token,
            } => {
                let (shaped, led) = utils::tool_constraints_for_worker(
                    tool_parser_factory,
                    parser.as_deref(),
                    Some(constraint.clone()),
                    Some(workers),
                )
                .map_err(failed)?;
                if let Some((_, tag)) = shaped {
                    request.set_structural_tag(tag);
                }
                led || *first_token
            }
            Self::Table(tag) => {
                let start = utils::WorkerDecoding::of(Some(workers)).start;
                let shaped = tag.tag_for(start).map_err(failed)?;
                let first_token = shaped.is_some() && start == GrammarStart::FirstToken;
                match (shaped, tag.schema()) {
                    // A JSON answer's: the tag, or the request's own schema
                    // where the table gives this worker none, the one in the
                    // other's place.
                    (shaped, Some(schema)) => {
                        request.restore_answer_constraint(match shaped {
                            Some(tag) => AnswerConstraint::StructuralTag(tag),
                            None => AnswerConstraint::JsonSchema(schema.to_string()),
                        });
                    }
                    (Some(tag), None) => request.restore_structural_tag(tag),
                    (None, None) => request.clear_structural_tag(),
                }
                first_token
            }
        };
        request.set_grammar_from_first_token(first_token);
        Ok(())
    }
}

/// Re-stamp the retained plan for a retry attempt's newly selected workers:
/// fresh engine ids per [`IdStamp`], the new worker's sampling defaults
/// (re-applied over the build-time baseline), the tool constraint in the
/// shape the new worker's engine needs ([`ToolConstraintStamp`]), and fresh
/// PD bootstrap/rendezvous rooms. EPD
/// encode bootstrap info is deliberately untouched — those rooms are the
/// rendezvous with encode jobs the first dispatch already launched.
pub(crate) fn restamp_plan_for_attempt(
    plan: &mut ExecutionPlan,
    stamp: &AttemptStamp,
    workers: &WorkerSelection,
    tool_parser_factory: &ToolParserFactory,
) -> Result<(), Response> {
    stamp.id.restamp(plan)?;
    for request in plan.generate_requests_mut() {
        if let (Some(mask), Some(baseline)) = (stamp.sampling_mask, &stamp.sampling_baseline) {
            baseline.restore_masked(request, mask)?;
            apply_sampling_defaults_with_mask(request, mask, Some(workers));
        }
        if let Some(tool_constraint) = &stamp.tool_constraint {
            tool_constraint.restamp(request, workers, tool_parser_factory)?;
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
/// This KV handoff is independent of any per-item encode->prefill bootstrap info.
pub(crate) fn maybe_inject_pd_rendezvous(
    request: &mut ProtoGenerateRequest,
    workers: &WorkerSelection,
) {
    // The KV bootstrap handoff is identical for plain PD and EPD; EPD just layers
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

/// The client that builds the request, from the clients a stage acquired, or
/// the failure a stage reports when that stage has not run.
fn builder_client<'a>(
    clients: Option<&'a ClientSelection>,
    function: &'static str,
) -> Result<&'a BackendClient, Response> {
    stage_output(clients, function).map(ClientSelection::builder)
}

/// The request's media, assembled for the engine that builds the request now
/// that the worker is known: `None` for a request without media; otherwise
/// the intermediate is taken (the prefill serialization needs it here), the
/// pixel-drop assembly runs when an encode stage did, the plain one else, a
/// failure is answered as the media error it is, and the request's share of
/// the in-flight media budget is reserved for the inline bytes. The builder is
/// resolved here rather than passed in, so a caller's borrow of the clients
/// does not pin the whole context across the call.
pub(crate) async fn assemble_prompt_media(
    ctx: &mut RequestContext,
    function: &'static str,
) -> Result<Option<MultimodalData>, Response> {
    let Some(intermediate) = ctx.state.multimodal_intermediate.take() else {
        return Ok(None);
    };
    let builder = builder_client(ctx.state.clients.as_ref(), function)?;
    // `encode_outputs` set by EncodeStage selects the pixel-drop assembly path.
    let assembled = if let Some(EncodeOutputs::Sglang(plan)) = &ctx.state.encode_outputs {
        Ok(MultimodalData::Sglang(SglangMultimodalData::for_encoder(
            plan.images.to_vec(),
        )))
    } else if ctx.state.encode_outputs.is_some() {
        assemble_multimodal_data_after_encode(intermediate, builder, ctx.state.workers.as_ref())
            .await
    } else {
        assemble_multimodal_data(intermediate, builder, ctx.state.workers.as_ref()).await
    };
    let data = assembled.map_err(|e| {
        error!(function = function, error = %e, "Failed to assemble multimodal request");
        assembly_error_response(&e)
    })?;
    ctx.state.multimodal_inflight = reserve_multimodal_inflight(
        ctx.components
            .multimodal
            .as_ref()
            .and_then(|multimodal| multimodal.inflight.as_deref()),
        data.inline_bytes(),
    )
    .await?;
    Ok(Some(data))
}

/// What a built generate request gets before it is planned, in the order the
/// dispatch needs: the string `stop`s its engine cannot match resolved by the
/// client, which reports the router's residual trim obligations; the PD
/// metadata when the stage injects it; an EPD request's per-item encode
/// bootstrap info (the dispatch plan stays on `encode_outputs` for request
/// execution to take); the prefill->decode KV rendezvous for the backends
/// that carry it in the request; and, for worker-side multimodal processing,
/// the media references now that the wire is known. The last two land before
/// the disaggregated dispatch clones the request, so both sides carry them.
pub(crate) fn finish_generate_request(
    ctx: &mut RequestContext,
    proto_request: &mut ProtoGenerateRequest,
    inject_pd_metadata: bool,
    function: &'static str,
) -> Result<(), Response> {
    let builder = builder_client(ctx.state.clients.as_ref(), function)?;
    ctx.state.response.router_stop_obligations =
        builder.finalize_generate_request(proto_request, ctx.tokenizer_arc().as_ref());

    if inject_pd_metadata {
        if let Some(workers) = ctx.state.workers.as_ref() {
            maybe_inject_pd_metadata(proto_request, workers);
        }
    }

    if let Some(outputs) = ctx.state.encode_outputs.as_mut() {
        match outputs {
            EncodeOutputs::TokenSpeed { bootstrap_info, .. } => proto_request
                .set_encode_bootstrap_info(std::mem::take(bootstrap_info))
                .map_err(|message| error::bad_request("encode_handoff_mismatch", message))?,
            EncodeOutputs::Sglang(_) => {
                if proto_request.sampling_n() != 1 {
                    return Err(error::bad_request(
                        "sglang_encode_n_unsupported",
                        "SGLang external encoding requires n = 1",
                    ));
                }
                let assignments = ctx
                    .state
                    .workers
                    .as_ref()
                    .and_then(WorkerSelection::encode_assignments)
                    .ok_or_else(|| {
                        error::internal_error(
                            "encode_assignments_missing",
                            "SGLang encoder assignments missing",
                        )
                    })?;
                let handoff =
                    super::sglang_encode::handoff(proto_request.request_id(), assignments)
                        .map_err(|e| error::bad_request("encode_handoff_invalid", e.to_string()))?;
                proto_request
                    .set_encode_bootstrap_info(handoff)
                    .map_err(|message| error::bad_request("encode_handoff_mismatch", message))?;
            }
        }
    }

    if let Some(workers) = ctx.state.workers.as_ref() {
        maybe_inject_pd_rendezvous(proto_request, workers);
    }

    if let Some(plan) = ctx.state.multimodal_refs.take() {
        if builder.is_zmq() {
            return Err(error::bad_request(
                "multimodal_not_supported",
                "media references require a gRPC vLLM worker",
            ));
        }
        let refs =
            assemble_media_refs(plan).map_err(|e| error::bad_request(e.code(), e.to_string()))?;
        proto_request
            .set_vllm_media_refs(refs)
            .map_err(|e| error::bad_request("multimodal_not_supported", e))?;
        ctx.state.media_refs_forwarded = true;
    }
    Ok(())
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
            tool_constraint: None,
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
        restamp_plan_for_attempt(&mut plan, &stamp, &worker_b, &ToolParserFactory::new()).unwrap();
        {
            let params = sglang_params(&plan);
            assert_eq!(params.temperature, 1.0, "baseline restored, not worker A's");
            assert_eq!(params.top_k, 9, "worker B's default applied");
        }

        // Worker C has no defaults label at all: everything returns to the
        // baseline.
        let worker_c = worker_with_defaults("grpc://worker-c:30000", None);
        restamp_plan_for_attempt(&mut plan, &stamp, &worker_c, &ToolParserFactory::new()).unwrap();
        {
            let params = sglang_params(&plan);
            assert_eq!(params.temperature, 1.0);
            assert_eq!(params.top_k, 3);
        }
    }
}

#[cfg(test)]
mod tool_constraint_restamp_tests {
    use std::sync::Arc;

    use openai_protocol::common::{Tool, ToolChoice, ToolChoiceValue};
    use serde_json::Value;
    use smg_grpc_client::{sglang_proto, tokenspeed_proto, trtllm_proto, vllm_proto};
    use tool_parser::ToolConstraint;

    use super::*;
    use crate::{
        routers::grpc::{context::ExecutionPlanKind, proto_wrapper::ProtoRequest},
        worker::{BasicWorkerBuilder, WorkerType},
    };

    /// A parser that registers a reasoning prefix, so a forced call on a
    /// thinking prompt takes the shape the worker's engine needs.
    const PARSER: &str = "glm47_moe";
    const BARE_CALLS: &str = "the bare calls (grammar after the reasoning)";
    const THOUGHT_THEN_CALLS: &str =
        "the thought block then the calls (grammar from the first token)";
    const PREFIX_THEN_CALLS: &str = "the free-text prefix then the calls (either engine)";

    fn worker(labels: &[(&str, &str)]) -> WorkerSelection {
        let mut builder =
            BasicWorkerBuilder::new("grpc://engine:50051").worker_type(WorkerType::Regular);
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        WorkerSelection::Single {
            worker: Arc::new(builder.build()),
        }
    }

    fn forced_call_on_a_thinking_prompt(factory: &ToolParserFactory) -> (String, String) {
        let tools: Vec<Tool> = serde_json::from_value(serde_json::json!([{
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
        }]))
        .unwrap();
        let required = ToolChoice::Value(ToolChoiceValue::Required);
        match factory
            .registry()
            .generate_tool_constraint(Some(PARSER), &tools, &required, true)
            .unwrap()
        {
            Some(ToolConstraint::StructuralTag(tag)) => ("structural_tag".to_string(), tag),
            other => panic!("expected a structural tag, got {other:?}"),
        }
    }

    /// Each backend's built request carrying `tag` where its builder places a
    /// forced call's structural tag.
    fn requests_with_tag(tag: &str) -> Vec<(&'static str, ProtoGenerateRequest)> {
        vec![
            (
                "sglang",
                ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
                    sampling_params: Some(sglang_proto::SamplingParams {
                        constraint: Some(sglang_proto::sampling_params::Constraint::StructuralTag(
                            tag.to_string(),
                        )),
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
            ),
            (
                "vllm",
                ProtoGenerateRequest::Vllm(Box::new(vllm_proto::GenerateRequest {
                    sampling_params: Some(vllm_proto::SamplingParams {
                        constraint: Some(vllm_proto::sampling_params::Constraint::StructuralTag(
                            tag.to_string(),
                        )),
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
            ),
            (
                "tokenspeed",
                ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
                    sampling_params: Some(tokenspeed_proto::SamplingParams {
                        constraint: Some(
                            tokenspeed_proto::sampling_params::Constraint::StructuralTag(
                                tag.to_string(),
                            ),
                        ),
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
            ),
            (
                "trtllm",
                ProtoGenerateRequest::Trtllm(Box::new(trtllm_proto::GenerateRequest {
                    guided_decoding: Some(trtllm_proto::GuidedDecodingParams {
                        guide_type: trtllm_proto::guided_decoding_params::GuideType::StructuralTag
                            as i32,
                        guide: tag.to_string(),
                    }),
                    ..Default::default()
                })),
            ),
        ]
    }

    /// Which engine a forced call's tag is shaped for.
    fn shape_of(tag: &str) -> &'static str {
        let tag: Value = serde_json::from_str(tag).unwrap();
        let format = &tag["format"];
        if format["type"] == "triggered_tags" {
            BARE_CALLS
        } else if format["elements"][0]["end"] == "</think>" {
            THOUGHT_THEN_CALLS
        } else {
            PREFIX_THEN_CALLS
        }
    }

    fn plan_tag(plan: &ExecutionPlan) -> String {
        plan_request(plan)
            .structural_tag()
            .expect("the plan keeps a structural tag")
            .to_string()
    }

    /// The one generate request of a single plan.
    fn plan_request(plan: &ExecutionPlan) -> &ProtoGenerateRequest {
        match plan {
            ExecutionPlan::Single(ProtoRequest::Generate(request)) => request,
            _ => panic!("expected a single generate plan"),
        }
    }

    /// A forced call built for an engine that starts the grammar after its
    /// reasoning parser's gate (the bare calls) reaches a retry's engine that
    /// starts it at the first token as the thought block then the calls, and
    /// the reverse; a retry's worker that does not say gets the prefix shape
    /// that fits either. On every backend that carries a structural tag.
    #[test]
    fn retry_reshapes_the_forced_call_for_its_own_workers_engine() {
        let factory = ToolParserFactory::new();
        let unshaped = forced_call_on_a_thinking_prompt(&factory);
        assert_eq!(shape_of(&unshaped.1), PREFIX_THEN_CALLS);
        let after_reasoning = worker(&[("constrained_decoding_start", "after_reasoning")]);
        let first_token = worker(&[("constrained_decoding_start", "first_token")]);
        let silent = worker(&[]);

        let (_, built_tag) = utils::tool_constraints_for_worker(
            &factory,
            Some(PARSER),
            Some(unshaped.clone()),
            Some(&after_reasoning),
        )
        .unwrap()
        .0
        .unwrap();
        assert_eq!(shape_of(&built_tag), BARE_CALLS);

        for (backend, request) in requests_with_tag(&built_tag) {
            let stamp = AttemptStamp {
                id: IdStamp::Exact,
                sampling_mask: None,
                sampling_baseline: None,
                inject_pd_metadata: false,
                tool_constraint: ToolConstraintStamp::capture(
                    Some(unshaped.clone()),
                    Some(PARSER),
                    false,
                    &request,
                ),
            };
            assert!(
                stamp.tool_constraint.is_some(),
                "{backend}: the built request carries the forced call's tag"
            );
            let mut plan = ExecutionPlan::generate(ExecutionPlanKind::Single, request);

            restamp_plan_for_attempt(&mut plan, &stamp, &first_token, &factory).unwrap();
            assert_eq!(
                shape_of(&plan_tag(&plan)),
                THOUGHT_THEN_CALLS,
                "{backend}: a retry on an engine that starts the grammar at the first token"
            );
            // The request says the grammar starts there, on the dialect with a
            // switch for the gate behind a reasoning parser.
            assert_eq!(
                plan_request(&plan).grammar_from_first_token(),
                backend == "vllm",
                "{backend}: the led tag's start rides with it"
            );

            restamp_plan_for_attempt(&mut plan, &stamp, &after_reasoning, &factory).unwrap();
            assert_eq!(
                shape_of(&plan_tag(&plan)),
                BARE_CALLS,
                "{backend}: a retry back on an engine that starts it after the reasoning"
            );
            assert!(
                !plan_request(&plan).grammar_from_first_token(),
                "{backend}: the bare calls start where the engine does"
            );

            restamp_plan_for_attempt(&mut plan, &stamp, &silent, &factory).unwrap();
            assert_eq!(
                plan_tag(&plan),
                unshaped.1,
                "{backend}: a worker that does not say gets the shape that fits either"
            );
            assert!(!plan_request(&plan).grammar_from_first_token(), "{backend}");
        }
    }

    /// The table's forced-call tag, built for a pool whose engines start the
    /// grammar at the first token, is no shape the registry rewrites: a retry
    /// keeps it as built, and keeps saying the grammar starts there on the
    /// dialect with a switch for the gate behind a reasoning parser, whatever
    /// the retry's worker says of its start.
    #[test]
    fn a_table_tag_built_for_the_first_token_keeps_its_start_on_a_retry() {
        let factory = ToolParserFactory::new();
        let tag = r#"{"format":{"type":"triggered_tags","triggers":["<call>"],"tags":[],"at_least_one":true}}"#;
        let first_token = worker(&[("constrained_decoding_start", "first_token")]);
        let after_reasoning = worker(&[("constrained_decoding_start", "after_reasoning")]);
        let silent = worker(&[]);
        for (backend, request) in requests_with_tag(tag) {
            let stamp = AttemptStamp {
                id: IdStamp::Exact,
                sampling_mask: None,
                sampling_baseline: None,
                inject_pd_metadata: false,
                tool_constraint: ToolConstraintStamp::capture(
                    Some(("structural_tag".to_string(), tag.to_string())),
                    Some(PARSER),
                    true,
                    &request,
                ),
            };
            let mut plan = ExecutionPlan::generate(ExecutionPlanKind::Single, request);
            for retry in [&first_token, &after_reasoning, &silent] {
                restamp_plan_for_attempt(&mut plan, &stamp, retry, &factory).unwrap();
                assert_eq!(plan_tag(&plan), tag, "{backend}: the table's tag as built");
                assert_eq!(
                    plan_request(&plan).grammar_from_first_token(),
                    backend == "vllm",
                    "{backend}: the start the table built it for rides with it"
                );
            }
        }
    }

    /// Nothing is stamped, and nothing rewritten, when the request's
    /// constraint is not the forced call's structural tag: a JSON-schema tool
    /// constraint (no shape to follow), or a backend that let the request's
    /// `response_format` outrank the tool constraint.
    #[test]
    fn a_request_without_the_forced_calls_tag_is_left_as_built() {
        let factory = ToolParserFactory::new();
        let schema = r#"{"type":"object"}"#.to_string();
        let mut request = ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
            sampling_params: Some(sglang_proto::SamplingParams {
                constraint: Some(sglang_proto::sampling_params::Constraint::JsonSchema(
                    schema.clone(),
                )),
                ..Default::default()
            }),
            ..Default::default()
        }));
        let schema_constraint = ("json_schema".to_string(), schema.clone());
        assert!(ToolConstraintStamp::capture(
            Some(schema_constraint),
            Some(PARSER),
            false,
            &request
        )
        .is_none());
        // The forced call's tag that the backend dropped for the request's
        // response_format: the request carries the schema, not the tag.
        let unshaped = forced_call_on_a_thinking_prompt(&factory);
        assert!(
            ToolConstraintStamp::capture(Some(unshaped), Some(PARSER), false, &request).is_none()
        );

        request.set_structural_tag("{}".to_string());
        assert_eq!(request.structural_tag(), None);
        request.clear_structural_tag();
        let ProtoGenerateRequest::Sglang(req) = &request else {
            panic!("the request was built for SGLang");
        };
        assert_eq!(
            req.sampling_params.as_ref().unwrap().constraint,
            Some(sglang_proto::sampling_params::Constraint::JsonSchema(
                schema
            ))
        );
    }

    /// A table's `auto` tag as a test derives it per start: the shape for an
    /// engine starting after its reasoning parser, another for one starting
    /// at the first token, and none for an engine that does not say.
    #[derive(Debug)]
    struct TableTag;

    impl TagForStart for TableTag {
        fn tag_for(&self, start: GrammarStart) -> Result<Option<String>, String> {
            Ok(match start {
                GrammarStart::AfterReasoning => Some(BARE_TABLE_TAG.to_string()),
                GrammarStart::FirstToken => Some(THOUGHT_TABLE_TAG.to_string()),
                GrammarStart::Unknown => None,
            })
        }

        fn schema(&self) -> Option<&Value> {
            None
        }
    }

    /// A JSON answer's tag as a test derives it per start: the closed thought
    /// then the schema for an engine starting at the first token, none for
    /// the others, which take the request's own schema (held here) in its
    /// place.
    #[derive(Debug)]
    struct AnswerTag(Value);

    impl TagForStart for AnswerTag {
        fn tag_for(&self, start: GrammarStart) -> Result<Option<String>, String> {
            Ok((start == GrammarStart::FirstToken).then(|| THOUGHT_ANSWER_TAG.to_string()))
        }

        fn schema(&self) -> Option<&Value> {
            Some(&self.0)
        }
    }

    const ANSWER_SCHEMA: &str = r#"{"type":"object"}"#;
    const THOUGHT_ANSWER_TAG: &str = r#"{"format":{"type":"sequence","elements":[{"type":"tag","begin":"","content":{"type":"any_text","excludes":[]},"end":"</think>"},{"type":"json_schema","json_schema":{"type":"object"}}]}}"#;

    /// Each backend's built request carrying `schema` where its builder places
    /// the request's own JSON schema.
    fn requests_with_schema(schema: &str) -> Vec<(&'static str, ProtoGenerateRequest)> {
        vec![
            (
                "sglang",
                ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
                    sampling_params: Some(sglang_proto::SamplingParams {
                        constraint: Some(sglang_proto::sampling_params::Constraint::JsonSchema(
                            schema.to_string(),
                        )),
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
            ),
            (
                "vllm",
                ProtoGenerateRequest::Vllm(Box::new(vllm_proto::GenerateRequest {
                    sampling_params: Some(vllm_proto::SamplingParams {
                        constraint: Some(vllm_proto::sampling_params::Constraint::JsonSchema(
                            schema.to_string(),
                        )),
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
            ),
            (
                "tokenspeed",
                ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
                    sampling_params: Some(tokenspeed_proto::SamplingParams {
                        constraint: Some(
                            tokenspeed_proto::sampling_params::Constraint::JsonSchema(
                                schema.to_string(),
                            ),
                        ),
                        ..Default::default()
                    }),
                    ..Default::default()
                })),
            ),
            (
                "trtllm",
                ProtoGenerateRequest::Trtllm(Box::new(trtllm_proto::GenerateRequest {
                    guided_decoding: Some(trtllm_proto::GuidedDecodingParams {
                        guide_type: trtllm_proto::guided_decoding_params::GuideType::JsonSchema
                            as i32,
                        guide: schema.to_string(),
                    }),
                    ..Default::default()
                })),
            ),
        ]
    }

    /// The plan's constraint: its structural tag and its JSON schema, one of
    /// them at most.
    fn plan_constraint(plan: &ExecutionPlan) -> (Option<String>, Option<String>) {
        match plan {
            ExecutionPlan::Single(ProtoRequest::Generate(request)) => (
                request.structural_tag().map(str::to_string),
                request.json_schema().map(str::to_string),
            ),
            _ => panic!("expected a single generate plan"),
        }
    }

    /// A JSON answer's tag, shaped at build for the selected worker's start,
    /// reaches a retry's worker derived again for that worker's own start, the
    /// request's own schema standing in where the table gives it no tag: from
    /// a first attempt that carried the tag and from one that carried the
    /// schema alike, on every backend that carries a constraint. The `auto`
    /// tag's derivation stands in for nothing, so a request built with the
    /// schema captures none of it, as before.
    #[test]
    fn retry_derives_the_json_answers_tag_for_its_own_workers_start_or_its_schema() {
        let factory = ToolParserFactory::new();
        let after_reasoning = worker(&[("constrained_decoding_start", "after_reasoning")]);
        let first_token = worker(&[("constrained_decoding_start", "first_token")]);
        let silent = worker(&[]);
        let answer: Arc<dyn TagForStart> =
            Arc::new(AnswerTag(serde_json::from_str(ANSWER_SCHEMA).unwrap()));
        let the_tag = (Some(THOUGHT_ANSWER_TAG.to_string()), None);
        let the_schema = (None, Some(ANSWER_SCHEMA.to_string()));

        let built = requests_with_tag(THOUGHT_ANSWER_TAG)
            .into_iter()
            .map(|(backend, request)| (backend, "the tag", request))
            .chain(
                requests_with_schema(ANSWER_SCHEMA)
                    .into_iter()
                    .map(|(backend, request)| (backend, "the schema", request)),
            );
        for (backend, carried, request) in built {
            let stamp = AttemptStamp {
                id: IdStamp::Exact,
                sampling_mask: None,
                sampling_baseline: None,
                inject_pd_metadata: false,
                tool_constraint: ToolConstraintStamp::capture_table(Arc::clone(&answer), &request),
            };
            assert!(
                matches!(stamp.tool_constraint, Some(ToolConstraintStamp::Table(_))),
                "{backend}, built with {carried}: the answer's derivation goes with the plan"
            );
            let mut plan = ExecutionPlan::generate(ExecutionPlanKind::Single, request);

            restamp_plan_for_attempt(&mut plan, &stamp, &after_reasoning, &factory).unwrap();
            assert_eq!(
                plan_constraint(&plan),
                the_schema,
                "{backend}, built with {carried}: after the gate, the request's own schema"
            );

            restamp_plan_for_attempt(&mut plan, &stamp, &first_token, &factory).unwrap();
            assert_eq!(
                plan_constraint(&plan),
                the_tag,
                "{backend}, built with {carried}: from the first token, the closed thought then \
                 the schema"
            );

            restamp_plan_for_attempt(&mut plan, &stamp, &silent, &factory).unwrap();
            assert_eq!(
                plan_constraint(&plan),
                the_schema,
                "{backend}, built with {carried}: a worker that does not say, the schema"
            );
        }

        for (backend, request) in requests_with_schema(ANSWER_SCHEMA) {
            let auto: Arc<dyn TagForStart> = Arc::new(TableTag);
            assert!(
                ToolConstraintStamp::capture_table(auto, &request).is_none(),
                "{backend}: the auto tag's derivation is not captured over the schema"
            );
        }
    }

    const BARE_TABLE_TAG: &str = r#"{"format":{"type":"const_string","value":"<call>"}}"#;
    const THOUGHT_TABLE_TAG: &str = r#"{"format":{"type":"sequence","elements":[{"type":"tag","begin":"","content":{"type":"any_text","excludes":[]},"end":"</think>"},{"type":"const_string","value":"<call>"}]}}"#;

    /// The table's `auto` tag, shaped at build for the selected worker's
    /// start, reaches a retry's worker derived again for that worker's own
    /// start: the other shape on an engine that starts the grammar elsewhere,
    /// and no tag at all, the request left unconstrained, on a worker whose
    /// engine the table gives none. On every backend that carries a
    /// structural tag; a backend whose builder let the request's
    /// `response_format` outrank the tag captures nothing.
    #[test]
    fn retry_derives_the_tables_auto_tag_for_its_own_workers_start() {
        let factory = ToolParserFactory::new();
        let after_reasoning = worker(&[("constrained_decoding_start", "after_reasoning")]);
        let first_token = worker(&[("constrained_decoding_start", "first_token")]);
        let silent = worker(&[]);
        let tag: Arc<dyn TagForStart> = Arc::new(TableTag);

        for (backend, request) in requests_with_tag(BARE_TABLE_TAG) {
            let stamp = AttemptStamp {
                id: IdStamp::Exact,
                sampling_mask: None,
                sampling_baseline: None,
                inject_pd_metadata: false,
                tool_constraint: ToolConstraintStamp::capture_table(Arc::clone(&tag), &request),
            };
            assert!(
                matches!(stamp.tool_constraint, Some(ToolConstraintStamp::Table(_))),
                "{backend}: the built request carries the table's tag"
            );
            let mut plan = ExecutionPlan::generate(ExecutionPlanKind::Single, request);

            restamp_plan_for_attempt(&mut plan, &stamp, &first_token, &factory).unwrap();
            assert_eq!(
                plan_tag(&plan),
                THOUGHT_TABLE_TAG,
                "{backend}: a retry on an engine that starts the grammar at the first token"
            );

            restamp_plan_for_attempt(&mut plan, &stamp, &after_reasoning, &factory).unwrap();
            assert_eq!(
                plan_tag(&plan),
                BARE_TABLE_TAG,
                "{backend}: a retry back on an engine that starts it after the reasoning"
            );

            restamp_plan_for_attempt(&mut plan, &stamp, &silent, &factory).unwrap();
            let ExecutionPlan::Single(ProtoRequest::Generate(request)) = &plan else {
                panic!("expected a single generate plan");
            };
            assert_eq!(
                request.structural_tag(),
                None,
                "{backend}: a worker the table gives no tag goes unconstrained"
            );

            // And back: the tag returns on the next worker the table shapes one for.
            restamp_plan_for_attempt(&mut plan, &stamp, &after_reasoning, &factory).unwrap();
            assert_eq!(
                plan_tag(&plan),
                BARE_TABLE_TAG,
                "{backend}: a retry after an unconstrained attempt takes the tag again"
            );
        }

        let schema = r#"{"type":"object"}"#.to_string();
        let outranked = ProtoGenerateRequest::Sglang(Box::new(sglang_proto::GenerateRequest {
            sampling_params: Some(sglang_proto::SamplingParams {
                constraint: Some(sglang_proto::sampling_params::Constraint::JsonSchema(
                    schema,
                )),
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert!(ToolConstraintStamp::capture_table(tag, &outranked).is_none());
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
        request
            .set_encode_bootstrap_info(vec![EncodeItemBootstrapInfo {
                item_index: 0,
                bootstrap_host: "encode-worker".to_string(),
                bootstrap_port: 9000,
                bootstrap_room: 41,
            }])
            .unwrap();
        request.set_kv_bootstrap_info("attempt1-prefill".to_string(), 8998, 42);
        let mut plan = ExecutionPlan::generate(ExecutionPlanKind::EncodePrefillDecode, request);

        let stamp = AttemptStamp {
            id: IdStamp::Exact,
            sampling_mask: None,
            sampling_baseline: None,
            inject_pd_metadata: crate::routers::grpc::mode::Mode::EncodePrefillDecode
                .inject_pd_metadata(),
            tool_constraint: None,
        };
        let retry_pair = tokenspeed_pd_pair("grpc://retry-prefill:30000", "grpc://decode:30000");
        restamp_plan_for_attempt(&mut plan, &stamp, &retry_pair, &ToolParserFactory::new())
            .unwrap();

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

    #[test]
    fn epd_pd_metadata_does_not_touch_other_runtime_requests() {
        for runtime in [
            RuntimeType::TokenSpeed,
            RuntimeType::Vllm,
            RuntimeType::Trtllm,
            RuntimeType::Mlx,
            RuntimeType::Generic,
            RuntimeType::External,
            RuntimeType::Unspecified,
        ] {
            let mut workers = tokenspeed_pd_pair("grpc://prefill:30000", "grpc://decode:30000");
            if let WorkerSelection::Disaggregated { runtime_type, .. } = &mut workers {
                *runtime_type = runtime;
            }
            // Even an SGLang-shaped request must stay untouched unless the
            // selected runtime actually uses its bootstrap wire protocol.
            let mut request = ProtoGenerateRequest::Sglang(Box::default());
            let before = request.as_sglang().clone();
            super::maybe_inject_pd_metadata(&mut request, &workers);
            assert_eq!(*request.as_sglang(), before, "runtime {runtime:?}");
        }
    }
}

#[cfg(test)]
mod wire_prompt_text_tests {
    use super::wire_prompt_text;

    #[test]
    fn a_text_only_request_sends_no_prompt_text() {
        assert_eq!(wire_prompt_text("<|user|>hello".to_string(), false), "");
    }

    #[test]
    fn a_request_with_media_keeps_its_prompt_text() {
        assert_eq!(
            wire_prompt_text("<|user|><|image|>describe".to_string(), true),
            "<|user|><|image|>describe"
        );
    }
}

#[cfg(test)]
mod shared_pieces_tests {
    use std::sync::Arc;

    use llm_multimodal::MediaContentPart;
    use llm_tokenizer::TokenizerRegistry;
    use openai_protocol::generate::GenerateRequest;
    use smg_grpc_client::VllmEngineClient;
    use tokio::task::JoinHandle;

    use super::*;
    use crate::{
        routers::grpc::{
            client::GrpcClient, context::SharedComponents, multimodal::MediaPlan,
            regular::symphony::template::Templates, test_engine::mock_engine,
        },
        worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, WorkerRegistry, WorkerType},
    };

    /// A request context after worker selection and client acquisition: one
    /// selected worker, a builder client on the mock engine, and the media
    /// references of worker-side processing waiting to be attached.
    async fn context_with_media_refs() -> (RequestContext, JoinHandle<()>) {
        let (url, server) = mock_engine("test-model").await;
        let client = VllmEngineClient::connect(&url)
            .await
            .expect("connect mock worker");
        let components = Arc::new(SharedComponents {
            templates: Arc::new(Templates::new(Arc::new(TokenizerRegistry::new()))),
            tokenizer_registry: Arc::new(TokenizerRegistry::new()),
            worker_registry: Arc::new(WorkerRegistry::new()),
            tool_parser_factory: Default::default(),
            reasoning_parser_factory: Default::default(),
            parser_resolver: utils::ParserResolver::disabled(),
            multimodal: None,
            tokenizer_autoload: true,
            tokenizer_load_failures: Arc::default(),
            parser: crate::config::ParserChoice::Legacy,
        });
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "text": "Hello"
        }))
        .expect("a generate request");
        let mut ctx = RequestContext::for_generate(
            Arc::new(request),
            None,
            "test-model".to_string(),
            components,
        );
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url.replace("http://", "grpc://"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .model(ModelCard::new("test-model"))
                .build(),
        );
        ctx.state.workers = Some(WorkerSelection::Single { worker });
        ctx.state.clients = Some(ClientSelection::Single {
            client: BackendClient::Grpc(GrpcClient::Vllm(client)),
        });
        ctx.state.multimodal_refs = Some(MediaPlan::new([MediaContentPart::ImageUrl {
            url: "data:image/png;base64,AAAA".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }]));
        (ctx, server)
    }

    /// The media references are attached by the finishing every request
    /// builder shares, once: the request built first carries them and the
    /// context records the forwarding; the plan is taken, so a request
    /// finished after it carries none.
    #[tokio::test]
    async fn finishing_attaches_the_media_references_once() {
        let (mut ctx, server) = context_with_media_refs().await;

        let mut request = ProtoGenerateRequest::Vllm(Box::default());
        finish_generate_request(&mut ctx, &mut request, false, "a_builder").expect("finished");
        assert!(
            request.has_vllm_media_refs(),
            "the first request carries the references"
        );
        assert!(ctx.state.media_refs_forwarded);
        assert!(ctx.state.multimodal_refs.is_none(), "the plan is taken");

        let mut again = ProtoGenerateRequest::Vllm(Box::default());
        finish_generate_request(&mut ctx, &mut again, false, "a_builder").expect("finished");
        assert!(!again.has_vllm_media_refs(), "attached once");
        server.abort();
    }

    /// A request whose preparation left no multimodal intermediate assembles
    /// nothing: the helper answers before it needs the builder client or the
    /// budget behind it, so it answers the same with the clients taken away.
    /// (The media references the fixture carries are worker-side work for the
    /// finishing, not an intermediate.)
    #[tokio::test]
    async fn without_an_intermediate_nothing_is_assembled() {
        let (mut ctx, server) = context_with_media_refs().await;
        ctx.state.clients = None;
        let assembled = assemble_prompt_media(&mut ctx, "a_builder")
            .await
            .expect("assembled");
        assert!(assembled.is_none());
        server.abort();
    }

    /// Before the clients are acquired, the finishing answers the one failure
    /// of a missing client selection, a 500 under its code.
    #[tokio::test]
    async fn without_acquired_clients_the_finishing_reports_the_missing_stage() {
        let (mut ctx, server) = context_with_media_refs().await;
        ctx.state.clients = None;
        let mut request = ProtoGenerateRequest::Vllm(Box::default());
        let failure = match finish_generate_request(&mut ctx, &mut request, false, "a_builder") {
            Ok(()) => panic!("finished without clients"),
            Err(failure) => failure,
        };
        assert_eq!(failure.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(failure.into_body(), usize::MAX)
            .await
            .expect("drain body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
        assert_eq!(
            body["error"]["code"], "client_acquisition_not_completed",
            "{body}"
        );
        server.abort();
    }
}
