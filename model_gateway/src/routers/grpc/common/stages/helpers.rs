//! Common helper functions shared across stages

use std::{
    fmt,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

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
                stage_output, AttemptStamp, ClientSelection, ExecutionPlan, RequestContext,
                RequestType, WorkerSelection,
            },
            multimodal::{
                assemble_media_refs, assemble_multimodal_data,
                assemble_multimodal_data_after_encode, assembly_error_response,
                reserve_multimodal_inflight,
            },
            proto_wrapper::{AnswerConstraint, MultimodalData, ProtoGenerateRequest},
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

/// The request's wall-clock time as `(seconds, subsecond nanos)` since the
/// Unix epoch: the timestamp of its charge id, a version-7 uuid the edge
/// minted when it took the request in, which is the request's one wall-clock
/// read (the response's `created`, the engine request id's stamp and the
/// charge id all tell the same second); read from the clock for a request
/// that carries no meta.
pub(crate) fn request_wall_clock(tenant_meta: Option<&TenantRequestMeta>) -> (u64, u32) {
    tenant_meta
        .and_then(|meta| meta.request_charge_id().get_timestamp())
        .map_or_else(
            || {
                let since_epoch = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                (since_epoch.as_secs(), since_epoch.subsec_nanos())
            },
            |stamp| stamp.to_unix(),
        )
}

/// A fresh version-7 uuid at the request's wall-clock time (see
/// [`request_wall_clock`]): unique by its random bits, ordered by the
/// request's time like one minted now, without a clock read of its own.
fn request_stamp_uuid(tenant_meta: Option<&TenantRequestMeta>) -> uuid::Uuid {
    let (seconds, nanos) = request_wall_clock(tenant_meta);
    uuid::Uuid::new_v7(uuid::Timestamp::from_unix(uuid::NoContext, seconds, nanos))
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
/// `{prefix}{uuid}` (see [`IdStamp`] for the uniqueness rules). The uuid is
/// stamped at the request's wall-clock time ([`request_stamp_uuid`]).
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
            (format!("{rid}-{}", request_stamp_uuid(tenant_meta)), stamp)
        } else {
            (rid.to_string(), IdStamp::Exact)
        };
    }
    match middleware_request_id(tenant_meta) {
        Some(request_id) => (
            format!("{request_id}-{}", request_stamp_uuid(tenant_meta)),
            IdStamp::Suffixed {
                base: request_id.to_string(),
            },
        ),
        None => (
            format!("{prefix}{}", request_stamp_uuid(tenant_meta)),
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
    /// The tag for an engine whose grammar starts at `start`, in the
    /// `vocabulary` of structural-tag nodes its compiler takes, or `None`
    /// where the table gives that engine no tag.
    fn tag_for(
        &self,
        start: GrammarStart,
        vocabulary: utils::TagVocabulary,
    ) -> Result<Option<String>, String>;

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
                let decoding = utils::WorkerDecoding::of(Some(workers));
                let shaped = tag
                    .tag_for(decoding.start, decoding.vocabulary)
                    .map_err(failed)?;
                let first_token = shaped.is_some() && decoding.start == GrammarStart::FirstToken;
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
fn sampling_defaults_for_request(workers: Option<&WorkerSelection>) -> Option<SamplingDefaults> {
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
                worker_url = %worker.redacted_url(),
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
    let assembled = if ctx.state.encode_outputs.is_some() {
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
        proto_request.set_encode_bootstrap_info(std::mem::take(&mut outputs.bootstrap_info));
    }

    if let Some(workers) = ctx.state.workers.as_ref() {
        maybe_inject_pd_rendezvous(proto_request, workers);
    }

    if let Some(plan) = ctx.state.multimodal_refs.take() {
        if builder.is_zmq() {
            return Err(error::bad_request(
                "multimodal_not_supported",
                "media references require a supported gRPC worker",
            ));
        }
        let refs =
            assemble_media_refs(plan).map_err(|e| error::bad_request(e.code(), e.to_string()))?;
        proto_request
            .set_media_refs(refs)
            .map_err(|e| error::bad_request("multimodal_not_supported", e))?;
        ctx.state.media_refs_forwarded = true;
    }
    Ok(())
}

/// A disaggregated worker's bootstrap rendezvous `(host, port)` (a prefill
/// worker's KV, an encode worker's embeddings): the operator's
/// (its URL's host, the port given with it) where set, else what its engine
/// declared at the handshake (a direct-ZMQ worker's URL names the gateway's
/// own bind, not the engine's host), else the default port.
pub(crate) fn bootstrap_rendezvous<'a>(
    registered_host: &'a str,
    registered_port: Option<u16>,
    declared: Option<(&'a str, u16)>,
) -> (&'a str, u16) {
    let host = match registered_host {
        "" => declared.map_or("", |(host, _)| host),
        host => host,
    };
    let port = registered_port
        .or_else(|| declared.map(|(_, port)| port))
        .unwrap_or(DEFAULT_BOOTSTRAP_PORT);
    (host, port)
}

/// Inject the KV bootstrap host/port/room into a generate request, at the
/// prefill worker's [`bootstrap_rendezvous`].
fn inject_kv_bootstrap_room(
    request: &mut ProtoGenerateRequest,
    prefill_worker: &Arc<dyn Worker>,
    dp_placement: DpPlacement,
) {
    let metadata = prefill_worker.metadata();
    let backend = prefill_worker.backend_client_if_initialized();
    let (hostname, bootstrap_port) = bootstrap_rendezvous(
        metadata.bootstrap_host(),
        metadata.bootstrap_port(),
        backend
            .as_deref()
            .and_then(BackendClient::declared_bootstrap),
    );
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
mod tests;
