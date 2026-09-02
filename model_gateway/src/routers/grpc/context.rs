//! Request context types for the two-phase gRPC router pipeline.
//!
//! [`RequestContext`] owns the parsed request through the ingress phase;
//! request building is its last reader and [`RequestContext::into_dispatch`]
//! yields the request-free [`DispatchContext`] the dispatch phase runs on.

use std::sync::Arc;

use axum::http::HeaderMap;
use llm_multimodal::registry::qwen3_asr::transcription::TranscriptionFamily;
use llm_tokenizer::{stop::StopSequenceDecoder, traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    classify::{ClassifyRequest, ClassifyResponse},
    completion::{CompletionRequest, CompletionResponse},
    embedding::{EmbeddingRequest, EmbeddingResponse},
    generate::{GenerateRequest, GenerateResponse},
    messages::{CreateMessageRequest, Message},
    responses::ResponsesRequest,
    transcription::{AudioFile, TranscriptionRequest},
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{debug, error};

use super::{
    backend_client::BackendClient,
    common::stages::{
        encode::EncodeDispatchPlan,
        helpers::{IdStamp, SamplingBaseline, SamplingDefaultsMask},
        RateLimitCell,
    },
    multimodal::{MultimodalComponents, MultimodalIntermediate},
    proto_wrapper::{
        EncodeItemBootstrapInfo, ProtoEmbedComplete, ProtoEmbedRequest, ProtoGenerateRequest,
        ProtoRequest, ProtoStream,
    },
    spec::ResponseSpec,
    utils::ParserResolver,
};
use crate::{
    middleware::TenantRequestMeta,
    routers::error::internal_error,
    worker::{ConnectionMode, RuntimeType, Worker, WorkerLoadGuard, WorkerRegistry},
};

/// Ingress-phase request context: owns the parsed request.
///
/// Lives from request entry through request building, which is the terminal
/// consumer — [`RequestContext::into_dispatch`] drops the request and yields
/// the [`DispatchContext`] the post-build phase runs on.
pub(crate) struct RequestContext {
    pub input: RequestInput,
    pub components: Arc<SharedComponents>,
    pub state: ProcessingState,
}

/// Immutable request input
pub(crate) struct RequestInput {
    pub request_type: RequestType,
    pub headers: Option<HeaderMap>,
    /// Canonical model ID used after aliases are resolved at request entry.
    pub model_id: String,
    /// Captured at construction so it survives the request drop at build.
    pub streaming: bool,
    pub tenant_request_meta: Option<TenantRequestMeta>,
    /// Holds the reservation outcome for the whole request (settle on
    /// success, denial check, streaming handoff). `None` for endpoints that
    /// haven't opted into tenant rate limiting yet (Responses, embeddings,
    /// classify).
    pub rate_limit_cell: Option<Arc<RateLimitCell>>,
}

/// Request type variants
/// Using Arc instead of Box to enable cheap cloning for background tasks
pub(crate) enum RequestType {
    Chat(Arc<ChatCompletionRequest>),
    Generate(Arc<GenerateRequest>),
    Completion(Arc<CompletionRequest>),
    Responses(Arc<ResponsesRequest>),
    Embedding(Arc<EmbeddingRequest>),
    Classify(Arc<ClassifyRequest>),
    Messages(Arc<CreateMessageRequest>),
    /// Audio transcription: the request plus its uploaded audio. The
    /// preparation stage turns these into a chat-shaped backend request
    /// inside the pipeline (no chat request is synthesized before entry).
    Transcription {
        request: Arc<TranscriptionRequest>,
        audio: Arc<AudioFile>,
    },
}

impl RequestType {
    /// Overwrite the request's own `model` field.
    ///
    /// `Arc::make_mut` copies the request when another handle is still
    /// alive. That cost is paid only on the alias path —
    /// [`RequestContext::new`] skips this call entirely when the client
    /// already used the canonical model ID.
    fn set_model(&mut self, model_id: &str) {
        fn replace(model: &mut String, model_id: &str) {
            model.clear();
            model.push_str(model_id);
        }

        match self {
            Self::Chat(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Generate(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Completion(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Responses(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Embedding(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Classify(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Messages(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Transcription { request, .. } => {
                replace(&mut Arc::make_mut(request).model, model_id);
            }
        }
    }

    /// Client-supplied backend request id (`rid`), where the protocol carries
    /// one. Responses ids are storage-owned (`resp_*`) and never client-set.
    pub fn rid(&self) -> Option<&str> {
        match self {
            Self::Chat(r) => r.rid.as_deref(),
            Self::Generate(r) => r.rid.as_deref(),
            Self::Completion(r) => r.rid.as_deref(),
            Self::Embedding(r) => r.rid.as_deref(),
            Self::Classify(r) => r.rid.as_deref(),
            Self::Messages(r) => r.rid.as_deref(),
            Self::Responses(_) | Self::Transcription { .. } => None,
        }
    }
}

impl std::fmt::Display for RequestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat(_) => write!(f, "Chat"),
            Self::Generate(_) => write!(f, "Generate"),
            Self::Completion(_) => write!(f, "Completion"),
            Self::Responses(_) => write!(f, "Responses"),
            Self::Embedding(_) => write!(f, "Embedding"),
            Self::Classify(_) => write!(f, "Classify"),
            Self::Messages(_) => write!(f, "Messages"),
            Self::Transcription { .. } => write!(f, "Transcription"),
        }
    }
}

impl std::fmt::Display for FinalResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat(_) => write!(f, "Chat"),
            Self::Generate(_) => write!(f, "Generate"),
            Self::Completion(_) => write!(f, "Completion"),
            Self::Embedding(_) => write!(f, "Embedding"),
            Self::Classify(_) => write!(f, "Classify"),
            Self::Messages(_) => write!(f, "Messages"),
            Self::Transcription { .. } => write!(f, "Transcription"),
        }
    }
}

/// Shared components (injected once at creation)
pub(crate) struct SharedComponents {
    pub tokenizer_registry: Arc<TokenizerRegistry>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub tool_parser_factory: ToolParserFactory,
    pub reasoning_parser_factory: ReasoningParserFactory,
    /// Per-request parser-name resolution (model-card override → configured
    /// CLI `--tool-call-parser`/`--reasoning-parser` names).
    pub parser_resolver: ParserResolver,
    /// Multimodal processing components (initialized at router creation)
    pub multimodal: Option<Arc<MultimodalComponents>>,
}

/// Ingress-phase state (evolves through preparation, worker selection,
/// client acquisition, encode, and request building).
#[derive(Default)]
pub(crate) struct ProcessingState {
    // Stage 1: Preparation outputs
    pub preparation: Option<PreparationOutput>,

    /// Owned here rather than inside `PreparationOutput` so EPD's `EncodeStage`
    /// can borrow it for the with-pixels encode serialization before request
    /// building `take()`s it for the prefill serialization.
    pub multimodal_intermediate: Option<MultimodalIntermediate>,

    /// `Some` iff the request is multimodal EPD and worker selection produced
    /// encode assignments. Request building injects the bootstrap info and drops
    /// prefill pixels; request execution `take()`s the dispatch plan.
    pub encode_outputs: Option<EncodeOutputs>,

    /// Resolved tokenizer (set once in preparation, reused in response processing)
    /// This avoids redundant registry lookups across pipeline stages.
    pub tokenizer: Option<Arc<dyn Tokenizer>>,

    // Stage 2: Worker selection outputs
    pub workers: Option<WorkerSelection>,

    /// Effective sticky key (rid-derived wins, header falls back), recorded by
    /// worker selection so load guards account keyed load identically.
    pub sticky_key: Option<String>,

    /// Selection inputs that survive the request drop, captured by worker
    /// selection for per-attempt re-selection in the dispatch phase.
    pub routing_snapshot: Option<RoutingSnapshot>,

    // Stage 3: Client acquisition outputs
    pub clients: Option<ClientSelection>,

    // Response processing state seeded during ingress (stop decoder, router
    // stop obligations, derived skip_special_tokens).
    pub response: ResponseState,
}

/// Worker-selection inputs that outlive the parsed request.
#[derive(Default)]
pub(crate) struct RoutingSnapshot {
    /// Captured only when a configured policy actually consumes request text.
    pub routing_text: Option<String>,
    /// Routing-affinity token proxy (first prompt for batched completions).
    pub token_ids: Vec<u32>,
    /// rid-derived sticky key, derived once at first selection.
    pub rid_key: Option<String>,
}

/// The wire the retained plan was built for. Retry re-selection filters
/// candidates to this (runtime, transport): the plan's proto flavor and its
/// stop-resolution are wire-specific and cannot be rebuilt post-drop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WireConstraint {
    pub runtime: RuntimeType,
    pub connection: ConnectionMode,
}

impl WireConstraint {
    fn of(workers: &WorkerSelection) -> Self {
        match workers {
            WorkerSelection::Single { worker } => Self {
                runtime: worker.metadata().spec.runtime_type,
                connection: *worker.connection_mode(),
            },
            // Disaggregated legs are gRPC-only.
            WorkerSelection::Disaggregated { runtime_type, .. } => Self {
                runtime: *runtime_type,
                connection: ConnectionMode::Grpc,
            },
        }
    }
}

/// Post-build request context.
///
/// Invariant, by construction: there is no request field here, so the
/// dispatch phase (worker re-selection, dispatch, response processing,
/// streaming) cannot reach the parsed request — it dropped in
/// [`RequestContext::into_dispatch`]. `ResponseSpec` is the only
/// request-derived input past this point.
pub(crate) struct DispatchContext {
    /// Canonical model ID (routing, registries).
    pub model_id: String,
    /// Model the response reports, captured from the request at the build
    /// boundary.
    pub dispatch_model: String,
    pub streaming: bool,
    pub headers: Option<HeaderMap>,
    pub rate_limit_cell: Option<Arc<RateLimitCell>>,
    pub routing: RoutingSnapshot,
    pub wire: WireConstraint,
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    pub workers: Option<WorkerSelection>,
    pub sticky_key: Option<String>,
    pub clients: Option<ClientSelection>,
    /// Consumed by the first dispatch; retries re-dispatch only the
    /// prefill/decode legs against the already-running encode jobs.
    pub encode_outputs: Option<EncodeOutputs>,
    pub dispatch: Option<DispatchMetadata>,
    pub load_guards: Option<LoadGuards>,
    pub response: ResponseState,
}

impl DispatchContext {
    /// Cached tokenizer (cheap Arc clone).
    pub fn tokenizer_arc(&self) -> Option<Arc<dyn Tokenizer>> {
        self.tokenizer.clone()
    }
}

/// Everything request building hands the dispatch phase.
pub(crate) struct BuildOutput {
    pub plan: ExecutionPlan,
    pub spec: ResponseSpec,
    pub stamp: AttemptStamp,
}

/// Per-attempt plan stamping inputs, captured at build so retries reproduce
/// exactly what a fresh build would have minted for the new attempt.
pub(crate) struct AttemptStamp {
    pub id: IdStamp,
    /// Which sampling fields the client left unset; retry attempts re-apply
    /// the newly selected worker's defaults through this mask.
    pub sampling_mask: Option<SamplingDefaultsMask>,
    /// Pre-default values of the masked fields, so re-application never
    /// carries a previous attempt's worker defaults forward.
    pub sampling_baseline: Option<SamplingBaseline>,
    /// Mode::PrefillDecode only (mirrors the build stage's flag).
    pub inject_pd_metadata: bool,
}

/// Per-item bootstrap rendezvous info for prefill, plus the dispatch plan that
/// fans out to encode workers.
///
/// Not `#[derive(Debug)]`: `EncodeDispatchPlan` transitively holds
/// non-`Debug` raw proto payloads (`TokenSpeedMultimodalItem`).
///
/// Owns the encode jobs' SHM/RDMA Drop guards: dropping this before request
/// execution dispatches (early return / cancellation) reclaims the staged
/// `/dev/shm` segments via `PreparedEncodeItem`'s `Drop`.
pub(crate) struct EncodeOutputs {
    pub bootstrap_info: Vec<EncodeItemBootstrapInfo>,
    pub dispatch: EncodeDispatchPlan,
}

/// Execution shape produced by request building. Retained until the retry
/// window closes; each attempt dispatches a clone (the last moves it).
#[derive(Clone)]
pub(crate) enum ExecutionPlan {
    Single(ProtoRequest),
    PrefillDecode(ProtoGenerateRequest),
    EncodePrefillDecode {
        request: ProtoGenerateRequest,
    },
    /// Batched completion fan-out: one backend request per prompt, all
    /// dispatched with the disaggregation shape given by `kind`. Sub-request
    /// ids are `{shared_request_id}-p{i}`; the client-visible response id is
    /// `shared_request_id`.
    Batch {
        kind: ExecutionPlanKind,
        shared_request_id: String,
        requests: Vec<ProtoGenerateRequest>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutionPlanKind {
    Single,
    PrefillDecode,
    EncodePrefillDecode,
}

impl ExecutionPlan {
    pub(crate) fn generate(kind: ExecutionPlanKind, request: ProtoGenerateRequest) -> Self {
        match kind {
            ExecutionPlanKind::Single => Self::Single(ProtoRequest::Generate(request)),
            ExecutionPlanKind::PrefillDecode => Self::PrefillDecode(request),
            ExecutionPlanKind::EncodePrefillDecode => Self::EncodePrefillDecode { request },
        }
    }

    pub(crate) fn embed(request: ProtoEmbedRequest) -> Self {
        Self::Single(ProtoRequest::Embed(request))
    }

    pub(crate) fn request_id(&self) -> &str {
        match self {
            Self::Single(request) => request.request_id(),
            Self::PrefillDecode(request) | Self::EncodePrefillDecode { request, .. } => {
                request.request_id()
            }
            Self::Batch {
                shared_request_id, ..
            } => shared_request_id,
        }
    }

    pub(crate) fn request_type(&self) -> &'static str {
        match self {
            Self::Single(ProtoRequest::Generate(_))
            | Self::PrefillDecode(_)
            | Self::EncodePrefillDecode { .. }
            | Self::Batch { .. } => "generate",
            Self::Single(ProtoRequest::Embed(_)) => "embed",
        }
    }

    pub(crate) fn mode_label(&self) -> &'static str {
        match self {
            Self::Single(_) => "single",
            Self::PrefillDecode(_) => "prefill_decode",
            Self::EncodePrefillDecode { .. } => "encode_prefill_decode",
            Self::Batch { kind, .. } => match kind {
                ExecutionPlanKind::Single => "single",
                ExecutionPlanKind::PrefillDecode => "prefill_decode",
                ExecutionPlanKind::EncodePrefillDecode => "encode_prefill_decode",
            },
        }
    }

    /// Serialized wire size of the built request(s), for the release metric.
    pub(crate) fn wire_len(&self) -> usize {
        match self {
            Self::Single(request) => request.wire_len(),
            Self::PrefillDecode(request) | Self::EncodePrefillDecode { request } => {
                request.wire_len()
            }
            Self::Batch { requests, .. } => {
                requests.iter().map(ProtoGenerateRequest::wire_len).sum()
            }
        }
    }

    /// Set the engine request id on a non-batch plan. A batch plan here is a
    /// build-stage wiring bug (its sub ids are stamped individually): fail
    /// rather than let a retry re-dispatch the previous attempt's ids.
    pub(crate) fn set_request_id(
        &mut self,
        request_id: String,
    ) -> Result<(), axum::response::Response> {
        match self {
            Self::Single(ProtoRequest::Generate(request))
            | Self::PrefillDecode(request)
            | Self::EncodePrefillDecode { request } => {
                request.set_request_id(request_id);
                Ok(())
            }
            Self::Single(ProtoRequest::Embed(request)) => {
                request.set_request_id(request_id);
                Ok(())
            }
            Self::Batch { .. } => {
                error!(
                    function = "ExecutionPlan::set_request_id",
                    "Single id stamp on a batch plan"
                );
                Err(internal_error(
                    "id_stamp_plan_mismatch",
                    "Id stamp does not match the plan shape",
                ))
            }
        }
    }

    /// Every generate request in the plan, for per-attempt re-stamping.
    pub(crate) fn generate_requests_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut ProtoGenerateRequest> {
        match self {
            Self::Single(ProtoRequest::Generate(request))
            | Self::PrefillDecode(request)
            | Self::EncodePrefillDecode { request } => std::slice::from_mut(request).iter_mut(),
            Self::Single(ProtoRequest::Embed(_)) => [].iter_mut(),
            Self::Batch { requests, .. } => requests.iter_mut(),
        }
    }
}

/// Output from preparation stage (Step 1)
///
/// Each request type produces its own variant, eliminating optional fields
/// that are always None for certain pipelines.
pub(crate) enum PreparationOutput {
    Chat {
        token_ids: Vec<u32>,
        processed_messages: super::ProcessedMessages,
        tool_constraints: Option<(String, String)>,
    },
    Messages {
        token_ids: Vec<u32>,
        processed_messages: super::ProcessedMessages,
        tool_constraints: Option<(String, String)>,
    },
    /// Transcription reuses the chat backend request shape. The chat-shaped
    /// request is synthesized here (inside the pipeline) from the family's
    /// prompt convention, so request building reads it in place of a
    /// client-supplied chat request; `format`/`family` flow into the
    /// response spec.
    Transcription {
        token_ids: Vec<u32>,
        processed_messages: super::ProcessedMessages,
        chat_request: Arc<ChatCompletionRequest>,
        format: super::spec::TranscriptionResponseFormat,
        family: &'static dyn TranscriptionFamily,
    },
    Completion {
        /// One entry per prompt; scalar requests carry exactly one.
        items: Vec<CompletionItem>,
        /// `Some` iff multiple prompts: their texts joined for routing,
        /// mirroring the HTTP router's `extract_text_for_routing`.
        joined_routing_text: Option<String>,
    },
    Generate {
        original_text: Option<String>,
        token_ids: Vec<u32>,
    },
    Embedding {
        original_text: String,
        token_ids: Vec<u32>,
    },
    Harmony {
        token_ids: Vec<u32>,
        selection_text: String,
        tool_constraints: Option<(String, String)>,
        /// Request with response_format cleared (when converted to structural tag)
        modified_request: Option<Box<ChatCompletionRequest>>,
        #[expect(dead_code, reason = "stored for future Harmony history tracking")]
        harmony_messages: Vec<super::harmony::HarmonyMessage>,
        harmony_stop_ids: Vec<u32>,
    },
}

/// One tokenized completion prompt.
pub(crate) struct CompletionItem {
    pub text: String,
    pub token_ids: Vec<u32>,
}

impl PreparationOutput {
    /// Token IDs (common to all variants). Batched completions expose the
    /// first prompt's tokens as the routing-affinity proxy.
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Chat { token_ids, .. }
            | Self::Messages { token_ids, .. }
            | Self::Transcription { token_ids, .. }
            | Self::Generate { token_ids, .. }
            | Self::Embedding { token_ids, .. }
            | Self::Harmony { token_ids, .. } => token_ids,
            Self::Completion { items, .. } => {
                items.first().map_or(&[], |item| item.token_ids.as_slice())
            }
        }
    }

    /// Total input token count across every item -- for accounting (rate-limit
    /// reservation), not routing. Unlike `token_ids()`, which exposes only the
    /// first prompt as a routing-affinity proxy, a batched `Completion`
    /// request's real input cost is the sum of every prompt in the batch.
    pub fn total_input_token_count(&self) -> usize {
        match self {
            Self::Completion { items, .. } => items.iter().map(|item| item.token_ids.len()).sum(),
            other => other.token_ids().len(),
        }
    }

    /// Text for worker routing: original_text for regular pipelines, selection_text for Harmony.
    /// Chat/Messages borrow from processed_messages.text to avoid a redundant clone.
    pub fn routing_text(&self) -> Option<&str> {
        match self {
            Self::Chat {
                processed_messages, ..
            }
            | Self::Messages {
                processed_messages, ..
            }
            | Self::Transcription {
                processed_messages, ..
            } => Some(&processed_messages.text),
            Self::Completion {
                items,
                joined_routing_text,
            } => joined_routing_text
                .as_deref()
                .or_else(|| items.first().map(|item| item.text.as_str())),
            Self::Embedding { original_text, .. } => Some(original_text),
            Self::Generate { original_text, .. } => original_text.as_deref(),
            Self::Harmony { selection_text, .. } => Some(selection_text),
        }
    }
}

#[derive(Clone)]
pub(crate) struct EncodeWorkerAssignment {
    pub item_index: usize,
    pub worker: Arc<dyn Worker>,
}

/// Worker selection (Step 2)
pub(crate) enum WorkerSelection {
    Single {
        worker: Arc<dyn Worker>,
    },
    /// Disaggregated prefill/decode selection. EPD layers per-item encode
    /// assignments on top; plain PD leaves `encode_assignments` unset.
    Disaggregated {
        encode_assignments: Option<Vec<EncodeWorkerAssignment>>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        runtime_type: RuntimeType,
    },
}

/// Client selection (Step 3)
#[derive(Clone)]
pub(crate) enum ClientSelection {
    Single {
        client: BackendClient,
    },
    /// Disaggregated prefill/decode scheduler clients. EPD encode workers are
    /// contacted directly from `WorkerSelection::Disaggregated` assignments.
    Disaggregated {
        prefill: BackendClient,
        decode: BackendClient,
    },
}

/// Dispatch metadata (Step 5)
#[derive(Clone)]
pub(crate) struct DispatchMetadata {
    pub request_id: String,
    pub model: String,
    pub created: u64,
    pub weight_version: Option<String>,
}

/// Load guards for worker load tracking
/// Automatically decrements load when dropped
pub(crate) enum LoadGuards {
    Single {
        _guard: WorkerLoadGuard,
    },
    /// Disaggregated guards cover the prefill+decode pair. EPD encode workers are
    /// assigned per item; their fire-and-supervise RPCs do not hold load guards.
    Disaggregated {
        _prefill: WorkerLoadGuard,
        _decode: WorkerLoadGuard,
    },
    /// Batched completion fan-out: one guard set per sub-request so load-aware
    /// policies see the real backend concurrency.
    Batch {
        _guards: Vec<LoadGuards>,
    },
}

impl LoadGuards {
    pub fn new(selection: &WorkerSelection, routing_key: Option<&str>) -> Self {
        match selection {
            WorkerSelection::Single { worker } => LoadGuards::Single {
                _guard: WorkerLoadGuard::with_key(worker.clone(), routing_key),
            },
            WorkerSelection::Disaggregated {
                prefill, decode, ..
            } => LoadGuards::Disaggregated {
                _prefill: WorkerLoadGuard::with_key(prefill.clone(), routing_key),
                _decode: WorkerLoadGuard::with_key(decode.clone(), routing_key),
            },
        }
    }

    /// One guard set per concurrent sub-request.
    pub fn scaled(selection: &WorkerSelection, routing_key: Option<&str>, count: usize) -> Self {
        if count <= 1 {
            Self::new(selection, routing_key)
        } else {
            Self::Batch {
                _guards: (0..count)
                    .map(|_| Self::new(selection, routing_key))
                    .collect(),
            }
        }
    }
}

/// Response processing state (Step 6)
#[derive(Default)]
pub(crate) struct ResponseState {
    /// Stop sequence decoder
    pub stop_decoder: Option<StopSequenceDecoder>,

    /// String stops the engine will never match, reported by
    /// `BackendClient::finalize_generate_request` during request building.
    /// Response processing must trim these from output text; empty when the
    /// engine matches stops server-side.
    pub router_stop_obligations: Vec<String>,

    /// Derived skip_special_tokens for streaming (set in preparation, read in response_processing).
    /// Stored here because PreparationOutput is consumed by request_building before
    /// response_processing runs.
    pub skip_special_tokens: Option<bool>,

    /// Execution result (streams from workers)
    pub execution_result: Option<ExecutionResult>,

    /// Final processed response
    pub final_response: Option<FinalResponse>,

    /// Responses API iteration result (Harmony only, for tool loop orchestration)
    pub responses_iteration_result: Option<super::harmony::ResponsesIterationResult>,
}

impl RequestContext {
    /// Build a context, resolving a model alias to its canonical model ID.
    ///
    /// This is the single place the gRPC pipeline canonicalizes. Both
    /// `input.model_id` and the request's own `model` field are rewritten, so
    /// every stage below — worker selection, tokenizer lookup, parser
    /// selection, tool call ID format — reads the canonical ID without
    /// resolving anything itself.
    ///
    /// One visible consequence: the response reports the canonical model ID,
    /// not the alias the client sent. That matches how the OpenAI API answers
    /// with the model it actually ran.
    fn new(
        mut request_type: RequestType,
        headers: Option<HeaderMap>,
        mut model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        if let Some(canonical_model_id) = components.worker_registry.resolve_model_alias(&model_id)
        {
            model_id.clear();
            model_id.push_str(&canonical_model_id);
            request_type.set_model(&model_id);
        }
        let streaming = match &request_type {
            RequestType::Chat(req) => req.stream,
            RequestType::Generate(req) => req.stream,
            RequestType::Completion(req) => req.stream,
            RequestType::Responses(req) => req.stream.unwrap_or(false),
            RequestType::Messages(req) => req.stream.unwrap_or(false),
            // Transcription is whole-file only; streaming is rejected in
            // preparation by capability check, never handed off here.
            RequestType::Transcription { .. } => false,
            // Embeddings and classification never stream.
            RequestType::Embedding(_) | RequestType::Classify(_) => false,
        };
        Self {
            input: RequestInput {
                request_type,
                headers,
                model_id,
                streaming,
                tenant_request_meta: None,
                rate_limit_cell: None,
            },
            components,
            state: ProcessingState::default(),
        }
    }

    /// Build-boundary conversion. The parsed request is dropped inside this
    /// function — `DispatchContext` has no field to carry it, so post-build
    /// stages cannot read it even by mistake.
    ///
    /// Fails loudly when worker selection's outputs (routing snapshot, wire)
    /// are absent: a retry context with defaulted routing inputs could
    /// re-select a worker the retained plan cannot be dispatched to.
    pub fn into_dispatch(self) -> Result<DispatchContext, axum::response::Response> {
        let RequestContext {
            input,
            components,
            state,
        } = self;
        let RequestInput {
            request_type,
            headers,
            model_id,
            streaming,
            tenant_request_meta: _,
            rate_limit_cell,
        } = input;
        // The model the response reports. `RequestContext::new` already
        // canonicalized both the request's `model` field and `model_id`, so a
        // request that arrived under an alias is answered under the canonical
        // name. Native `/generate` callers may leave the field empty, so
        // prefer the resolved id there.
        let dispatch_model = match &request_type {
            RequestType::Chat(req) => req.model.clone(),
            RequestType::Completion(req) => req.model.clone(),
            RequestType::Generate(_) => model_id.clone(),
            RequestType::Responses(req) => req.model.clone(),
            RequestType::Embedding(req) => req.model.clone(),
            RequestType::Classify(req) => req.model.clone(),
            RequestType::Messages(req) => req.model.clone(),
            RequestType::Transcription { request, .. } => request.model.clone(),
        };
        drop(request_type);
        drop(components);
        let routing = state.routing_snapshot.ok_or_else(|| {
            error!(
                function = "RequestContext::into_dispatch",
                "Routing snapshot not captured by worker selection"
            );
            internal_error(
                "routing_snapshot_not_captured",
                "Routing snapshot not captured",
            )
        })?;
        let wire = state
            .workers
            .as_ref()
            .map(WireConstraint::of)
            .ok_or_else(|| {
                error!(
                    function = "RequestContext::into_dispatch",
                    "Worker selection not completed"
                );
                internal_error(
                    "worker_selection_not_completed",
                    "Worker selection not completed",
                )
            })?;
        Ok(DispatchContext {
            model_id,
            dispatch_model,
            streaming,
            headers,
            rate_limit_cell,
            routing,
            wire,
            tokenizer: state.tokenizer,
            workers: state.workers,
            sticky_key: state.sticky_key,
            clients: state.clients,
            encode_outputs: state.encode_outputs,
            dispatch: None,
            load_guards: None,
            response: state.response,
        })
    }

    /// Create context for chat completion request
    pub fn for_chat(
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(RequestType::Chat(request), headers, model_id, components)
    }

    /// Create context for an audio transcription request.
    pub fn for_transcription(
        request: Arc<TranscriptionRequest>,
        audio: Arc<AudioFile>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Transcription { request, audio },
            headers,
            model_id,
            components,
        )
    }

    /// Create context for generate request
    pub fn for_generate(
        request: Arc<GenerateRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Generate(request),
            headers,
            model_id,
            components,
        )
    }

    /// Create context for completion request
    pub fn for_completion(
        request: Arc<CompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Completion(request),
            headers,
            model_id,
            components,
        )
    }

    /// Create context for Responses API request
    pub fn for_responses(
        request: Arc<ResponsesRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Responses(request),
            headers,
            model_id,
            components,
        )
    }

    /// Create context for embedding request
    pub fn for_embedding(
        request: Arc<EmbeddingRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Embedding(request),
            headers,
            model_id,
            components,
        )
    }

    /// Create context for classify request
    pub fn for_classify(
        request: Arc<ClassifyRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Classify(request),
            headers,
            model_id,
            components,
        )
    }

    /// Create context for messages request
    pub fn for_messages(
        request: Arc<CreateMessageRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Messages(request),
            headers,
            model_id,
            components,
        )
    }

    /// Get Arc clone of chat request (panics if not chat)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn chat_request_arc(&self) -> Arc<ChatCompletionRequest> {
        match &self.input.request_type {
            RequestType::Chat(req) => Arc::clone(req),
            _ => panic!("Expected chat request"),
        }
    }

    /// Get Arc clones of the transcription request and its audio (panics if
    /// not a transcription request).
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn transcription_input_arc(&self) -> (Arc<TranscriptionRequest>, Arc<AudioFile>) {
        match &self.input.request_type {
            RequestType::Transcription { request, audio } => {
                (Arc::clone(request), Arc::clone(audio))
            }
            _ => panic!("Expected transcription request"),
        }
    }

    /// Get Arc clone of generate request (panics if not generate)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn generate_request_arc(&self) -> Arc<GenerateRequest> {
        match &self.input.request_type {
            RequestType::Generate(req) => Arc::clone(req),
            _ => panic!("Expected generate request"),
        }
    }

    /// Get Arc clone of completion request (panics if not completion)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn completion_request_arc(&self) -> Arc<CompletionRequest> {
        match &self.input.request_type {
            RequestType::Completion(req) => Arc::clone(req),
            _ => panic!("Expected completion request"),
        }
    }

    /// Get Arc clone of responses request (panics if not responses)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn responses_request_arc(&self) -> Arc<ResponsesRequest> {
        match &self.input.request_type {
            RequestType::Responses(req) => Arc::clone(req),
            _ => panic!("Expected responses request"),
        }
    }

    /// Get Arc clone of messages request (panics if not messages)
    #[expect(
        clippy::panic,
        reason = "typed accessor: caller guarantees variant via RequestType construction"
    )]
    pub fn messages_request_arc(&self) -> Arc<CreateMessageRequest> {
        match &self.input.request_type {
            RequestType::Messages(req) => Arc::clone(req),
            _ => panic!("Expected messages request"),
        }
    }

    /// Check if request is streaming (captured at construction).
    pub fn is_streaming(&self) -> bool {
        self.input.streaming
    }

    /// Get the cached tokenizer, cloning the Arc (cheap 8-byte clone)
    ///
    /// Returns None if tokenizer hasn't been resolved yet.
    /// The tokenizer is resolved once in the preparation stage and cached for reuse.
    pub fn tokenizer_arc(&self) -> Option<Arc<dyn Tokenizer>> {
        self.state.tokenizer.clone()
    }
}

/// Some methods are kept for API completeness even if currently unused.
#[expect(dead_code)]
impl WorkerSelection {
    pub fn is_disaggregated(&self) -> bool {
        matches!(self, Self::Disaggregated { .. })
    }

    pub fn single(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Single { worker } => Some(worker),
            Self::Disaggregated { .. } => None,
        }
    }

    /// Record circuit breaker outcome for all workers based on HTTP status code.
    pub fn record_outcome(&self, status_code: u16) {
        match self {
            Self::Single { worker } => worker.record_outcome(status_code),
            Self::Disaggregated {
                prefill, decode, ..
            } => {
                // EPD encode dispatch is asynchronous and supervised by
                // RequestExecution; this records only the prefill/decode leg.
                prefill.record_outcome(status_code);
                decode.record_outcome(status_code);
            }
        }
    }

    /// Record circuit breaker outcomes for disaggregated dispatch (individual tracking)
    pub fn record_prefill_decode_outcomes(&self, prefill_status: u16, decode_status: u16) {
        if let Self::Disaggregated {
            prefill, decode, ..
        } = self
        {
            prefill.record_outcome(prefill_status);
            decode.record_outcome(decode_status);
        }
    }

    /// Record circuit breaker outcome for prefill worker only (sequential PD)
    pub fn record_outcome_prefill(&self, status_code: u16) {
        match self {
            Self::Disaggregated { prefill, .. } => {
                prefill.record_outcome(status_code);
            }
            Self::Single { .. } => {
                debug!("record_outcome_prefill called on Single worker selection, ignoring");
            }
        }
    }

    /// Record circuit breaker outcome for decode worker only (sequential PD)
    pub fn record_outcome_decode(&self, status_code: u16) {
        match self {
            Self::Disaggregated { decode, .. } => {
                decode.record_outcome(status_code);
            }
            Self::Single { .. } => {
                debug!("record_outcome_decode called on Single worker selection, ignoring");
            }
        }
    }

    #[expect(clippy::type_complexity)]
    pub fn disaggregated_pair(&self) -> Option<(&Arc<dyn Worker>, &Arc<dyn Worker>)> {
        match self {
            Self::Disaggregated {
                prefill, decode, ..
            } => Some((prefill, decode)),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Disaggregated { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_worker(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Disaggregated { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }

    /// Get the runtime type for disaggregated mode.
    pub fn disaggregated_runtime_type(&self) -> Option<&RuntimeType> {
        match self {
            Self::Disaggregated { runtime_type, .. } => Some(runtime_type),
            Self::Single { .. } => None,
        }
    }

    pub fn encode_assignments(&self) -> Option<&[EncodeWorkerAssignment]> {
        match self {
            Self::Disaggregated {
                encode_assignments, ..
            } => encode_assignments.as_deref(),
            Self::Single { .. } => None,
        }
    }
}

/// Some methods are kept for API completeness even if currently unused.
#[expect(dead_code)]
impl ClientSelection {
    pub fn single(&self) -> Option<&BackendClient> {
        match self {
            Self::Single { client } => Some(client),
            Self::Disaggregated { .. } => None,
        }
    }

    pub fn single_mut(&mut self) -> Option<&mut BackendClient> {
        match self {
            Self::Single { client } => Some(client),
            Self::Disaggregated { .. } => None,
        }
    }

    pub fn disaggregated_mut(&mut self) -> Option<(&mut BackendClient, &mut BackendClient)> {
        match self {
            Self::Disaggregated { prefill, decode } => Some((prefill, decode)),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_client(&self) -> Option<&BackendClient> {
        match self {
            Self::Disaggregated { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn prefill_client_mut(&mut self) -> Option<&mut BackendClient> {
        match self {
            Self::Disaggregated { prefill, .. } => Some(prefill),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_client(&self) -> Option<&BackendClient> {
        match self {
            Self::Disaggregated { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }

    pub fn decode_client_mut(&mut self) -> Option<&mut BackendClient> {
        match self {
            Self::Disaggregated { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }
}

/// Result of request execution (streams from workers)
/// Uses ProtoStream to automatically abort on cancellation
pub(crate) enum ExecutionResult {
    Single {
        stream: ProtoStream,
    },
    PrefillDecode {
        prefill: ProtoStream,
        decode: Box<ProtoStream>,
        /// PD timing context, for honest PD TTFT (prefill start to first decode token).
        pd_timing: PdTiming,
    },
    /// Embedding requests return a single response, not a stream
    Embedding {
        response: ProtoEmbedComplete,
    },
    /// Batched completion fan-out: one result per prompt, in prompt order.
    Batch {
        results: Vec<ExecutionResult>,
    },
}

/// Timing context threaded from PD execution into the streaming layer so the
/// first decode token can be measured against prefill start.
#[derive(Clone)]
pub(crate) struct PdTiming {
    /// Monotonic instant the prefill RPC was dispatched.
    pub prefill_start: std::time::Instant,
    /// Backend runtime label (e.g. "sglang", "vllm") for the PD metric set.
    pub runtime: &'static str,
}

/// Final processed response
#[derive(Debug)]
pub(crate) enum FinalResponse {
    Chat(ChatCompletionResponse),
    /// Generate response is a Vec of GenerateResponse (n=1 returns single item, n>1 returns multiple)
    Generate(Vec<GenerateResponse>),
    /// Completion response (OpenAI /v1/completions format)
    Completion(CompletionResponse),
    /// Embedding response
    Embedding(EmbeddingResponse),
    /// Classification response
    Classify(ClassifyResponse),
    /// Messages API response
    Messages(Message),
    /// Transcription: the decoded transcript plus its wire format.
    Transcription {
        text: String,
        format: super::spec::TranscriptionResponseFormat,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completion_prep(texts: &[&str], joined: Option<&str>) -> PreparationOutput {
        PreparationOutput::Completion {
            items: texts
                .iter()
                .enumerate()
                .map(|(i, text)| CompletionItem {
                    text: (*text).to_string(),
                    token_ids: vec![i as u32],
                })
                .collect(),
            joined_routing_text: joined.map(str::to_string),
        }
    }

    #[test]
    fn completion_preparation_routes_by_first_item_or_joined_text() {
        let scalar = completion_prep(&["hello"], None);
        assert_eq!(scalar.routing_text(), Some("hello"));
        assert_eq!(scalar.token_ids(), &[0]);

        let batch = completion_prep(&["a", "b"], Some("a b"));
        assert_eq!(batch.routing_text(), Some("a b"));
        assert_eq!(batch.token_ids(), &[0]);
    }

    /// `token_ids()` is deliberately a single-item routing-affinity proxy
    /// (see the test above), but a batched Completion's real input cost is
    /// every prompt in the batch -- `total_input_token_count()` must sum
    /// them all, not just echo the first item like `token_ids()` does.
    #[test]
    fn total_input_token_count_sums_every_batched_completion_item() {
        let batch = PreparationOutput::Completion {
            items: vec![
                CompletionItem {
                    text: "short".to_string(),
                    token_ids: vec![1],
                },
                CompletionItem {
                    text: "much longer prompt".to_string(),
                    token_ids: vec![2, 3, 4, 5, 6],
                },
            ],
            joined_routing_text: Some("short much longer prompt".to_string()),
        };

        // The routing proxy only ever sees the first item...
        assert_eq!(batch.token_ids().len(), 1);
        // ...but reservation accounting must see the whole batch's cost.
        assert_eq!(batch.total_input_token_count(), 6);

        let scalar = completion_prep(&["hello"], None);
        assert_eq!(scalar.total_input_token_count(), scalar.token_ids().len());
    }

    #[test]
    fn batch_execution_plan_reports_shared_id_and_kind_label() {
        let plan = ExecutionPlan::Batch {
            kind: ExecutionPlanKind::PrefillDecode,
            shared_request_id: "cmpl_shared".to_string(),
            requests: vec![],
        };
        assert_eq!(plan.request_id(), "cmpl_shared");
        assert_eq!(plan.request_type(), "generate");
        assert_eq!(plan.mode_label(), "prefill_decode");
    }
}
