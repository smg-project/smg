//! Request context types for the two-phase gRPC router pipeline.
//!
//! [`RequestContext`] owns the parsed request through the ingress phase;
//! request building is its last reader and [`RequestContext::into_dispatch`]
//! yields the request-free [`DispatchContext`] the dispatch phase runs on.

use std::{sync::Arc, time::Instant};

use axum::http::HeaderMap;
use llm_multimodal::registry::transcription::TranscriptionFamily;
use llm_tokenizer::{stop::StopSequenceDecoder, traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    classify::{ClassifyRequest, ClassifyResponse},
    completion::{CompletionRequest, CompletionResponse},
    decisions::{DecisionResponse, DecisionsRequest},
    embedding::{EmbeddingRequest, EmbeddingResponse},
    generate::{GenerateRequest, GenerateResponse},
    messages::{CreateMessageRequest, Message},
    profile::ProviderProfile,
    responses::ResponsesRequest,
    transcription::{AudioFile, TranscriptionRequest},
};
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use smg_grpc_client::RequestDeadline;
use smg_worker::identity::basic::{RuntimeType, Worker, WorkerLoadGuard};
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{debug, error};

use super::{
    backend_client::BackendClient,
    common::{
        response_formatting::TokenCounts,
        stages::{
            encode::EncodeDispatchPlan,
            helpers::{IdStamp, SamplingBaseline, SamplingDefaultsMask, ToolConstraintStamp},
            pd_protocol::{EncodeHandoff, PdProtocol},
            request_execution::EncodeWatch,
            RateLimitCell,
        },
    },
    multimodal::{
        ec_push::PushEncodePlan, InflightPermit, MediaPlan, MultimodalComponents,
        MultimodalIntermediate,
    },
    proto_wrapper::{
        EncodeItemBootstrapInfo, ProtoEmbedComplete, ProtoEmbedRequest, ProtoGenerateRequest,
        ProtoInputLogProbs, ProtoRequest, ProtoStream,
    },
    regular::{
        stages::decisions::scoring::{DecisionPrompt, DecisionScoring},
        symphony::{PromptReplay, Reading},
    },
    spec::{MessagesTools, ResponseSpec},
    utils::{Names, ParserResolver, WorkerDecoding},
};
use crate::{
    config::ParserChoice,
    middleware::{RequestStarted, TenantRequestMeta},
    observability::{cache_trace, metrics::Metrics},
    policies::CacheNamespace,
    routers::{
        common::{
            attempt_ledger::AttemptLedger, pd_admission::PdAdmissionGuard, placement::Placed,
            prefill_admission::PrefillLoadGuard, snapshot::WorkerRegistry, sse::SseSender,
        },
        error::internal_error,
        grpc::regular::symphony::{template::Templates, PreparedConstraint},
    },
    workflow::TokenizerLoadFailures,
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
    /// Holds the reservation outcome for the whole logical request (settle on
    /// success, denial check, streaming handoff): one per typed request, one
    /// per iteration of a Responses request. `None` only where no caller set
    /// one (a unit test's context).
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
    Decisions(Arc<DecisionsRequest>),
    Messages(Arc<CreateMessageRequest>),
    /// Audio transcription: the request plus its uploaded audio. The
    /// preparation stage turns these into a chat-shaped backend request
    /// inside the pipeline (no chat request is synthesized before entry).
    Transcription {
        request: Arc<TranscriptionRequest>,
        audio: Arc<AudioFile>,
    },
}

fn openai_response_model(model: &str, adapter: Option<&str>) -> String {
    adapter.map_or_else(|| model.to_string(), |adapter| format!("{model}:{adapter}"))
}

impl RequestType {
    /// Adapter selection forwarded by the SGLang gRPC request builders.
    pub fn lora_adapter(&self) -> Option<&str> {
        match self {
            Self::Chat(request) => request.lora_path.as_deref(),
            Self::Generate(request) => request.lora_path.as_deref().or(request.lora_id.as_deref()),
            Self::Completion(request) => request.lora_path.as_deref(),
            _ => None,
        }
    }

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
            Self::Decisions(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Messages(request) => replace(&mut Arc::make_mut(request).model, model_id),
            Self::Transcription { request, .. } => {
                replace(&mut Arc::make_mut(request).model, model_id);
            }
        }
    }

    /// Parallel samples the request asks for: 1 when unset, or when the
    /// surface has no such parameter. With more than one, a disaggregated
    /// decode side recomputes the prompt locally instead of pulling the
    /// prefill side's KV.
    pub fn sampling_n(&self) -> u32 {
        match self {
            Self::Chat(r) => r.choices(),
            Self::Completion(r) => r.choices(),
            Self::Generate(r) => r.choices(),
            _ => 1,
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
            Self::Responses(_) | Self::Decisions(_) | Self::Transcription { .. } => None,
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
            Self::Decisions(_) => write!(f, "Decisions"),
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
            Self::Decisions(_) => write!(f, "Decisions"),
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
    /// Whether the gateway loads tokenizers itself, at start-up and when a
    /// worker registers. While it does, a served model without a registered
    /// tokenizer is a transient not-ready state (the one `/readiness` reports
    /// as "tokenizer not yet registered"), not a fault of the gateway.
    pub tokenizer_autoload: bool,
    /// The served models whose tokenizer load was skipped or failed for good,
    /// as the registration workflow recorded them: a request for one of them
    /// is answered 503 `tokenizer_load_failed` without a retry hint, instead
    /// of the not-ready 503 of a load in flight.
    pub tokenizer_load_failures: Arc<TokenizerLoadFailures>,
    /// Which parser reads a chat completion's or a message's output unless the request's
    /// `X-SMG-Parser` header says otherwise (see `regular::symphony`).
    pub parser: ParserChoice,
    /// The tables the registered tokenizers' response templates compile into, which the parser
    /// name `hf` selects (see `regular::symphony::template`).
    pub templates: Arc<Templates>,
}

/// Ingress-phase state (evolves through preparation, worker selection,
/// client acquisition, encode, and request building).
#[derive(Default)]
pub(crate) struct ProcessingState {
    // Stage 1: Preparation outputs
    pub preparation: Option<PreparationOutput>,

    /// The parser for this chat completion or message. Preparation sets the request's choice,
    /// from its `X-SMG-Parser` header or the configured default; request building then sets the
    /// parser that reads the response, Symphony when a table takes the request and legacy
    /// otherwise, which the response reports as `x-smg-parser`. `None` on the other endpoints,
    /// which keep the legacy parsers.
    pub parser: Option<ParserChoice>,

    /// The parser names the request's model's cards set (its workers' labels), resolved once
    /// in preparation (`reading::prepare`: one resolution, a walk over the model's workers per
    /// name) and read by request building in its place (`reading::reasoning_decision`,
    /// `reading::decide`, `backed_plan::build`), which hands them to the response spec
    /// (`ChatResponseSpec::parser_names`, `MessagesResponseSpec::parser_names`) for the
    /// response stages' two readers (`processor`, `streaming`): one resolution per request.
    /// `None` on the endpoints whose preparation resolves no names; a read then resolves them
    /// itself.
    pub parser_names: Option<Names>,

    /// The provider profile of the request's model (`ProviderProfile::for_model`), derived once
    /// by the chat and Messages preparations and read by the later stages in place of a
    /// derivation of their own (the constraint's tool-call reading, request building, the
    /// response spec). `None` on the other endpoints; a read then derives it itself.
    pub provider: Option<ProviderProfile>,
    /// The table's replay of the rendered prompt, as preparation read it for the forced call's
    /// tag (`reading::prepare`): request building's reasoning decision takes it in place of a
    /// second replay of the same prompt (`reading::reasoning_decision`). `None` where the
    /// endpoint's preparation reads no prompt; a decision then replays itself.
    pub prompt_thought: Option<PromptReplay>,
    /// The table decision for the request's model under its parser (`regular::symphony::reading`:
    /// the table that reads the response, or why the legacy parsers do), as the chat and Messages
    /// preparations made it once for the replay, the constraint and the decoder's special tokens
    /// (`reading::prepare`): request building's assignment reads it in its place
    /// (`reading::decide`), so those four share one decision. The whole turn's grammar and the
    /// `auto` tag still decide over the same inputs, shaped for the selected worker's start
    /// (`regular::symphony::turn_constraint`, `AutoTag`). Transcription shares the chat
    /// preparation and keeps the legacy decision it makes, which nothing reads. `None` on the
    /// endpoints whose preparation decides nothing, and when the names refused (the constraint
    /// carries the refusal); the assignment then decides itself.
    pub reading: Option<Reading>,

    /// A Messages request's tools, taken out of the request by the route
    /// while it still owned it (`MessagesTools::take_from`) and stored by
    /// `RequestContext::for_messages`: the Messages preparation clones them
    /// (a flag and an `Arc`) into the response spec. `None` only for a
    /// context of another request type.
    pub messages_tools: Option<MessagesTools>,

    /// Owned here rather than inside `PreparationOutput` so EPD's `EncodeStage`
    /// can borrow it for the with-pixels encode serialization before request
    /// building `take()`s it for the prefill serialization.
    pub multimodal_intermediate: Option<MultimodalIntermediate>,

    /// Media references kept for a worker that processes them itself; never
    /// `Some` together with `multimodal_intermediate`. Request building takes it.
    pub multimodal_refs: Option<MediaPlan>,

    /// Resolved image anchor for vLLM's image-only encoder fan-out.
    pub image_anchor: Option<u32>,

    /// Set once request building attached media references, so retry
    /// re-selection stays pinned to workers that accept them.
    pub media_refs_forwarded: bool,

    /// The selected worker's constrained decoding, set when request building
    /// let its policy choose the request's constraint
    /// (`regular::symphony::constraint::WorkerConstraint::by_policy`), so retry
    /// re-selection stays on workers that decide alike
    /// (`utils::WorkerDecoding::decides_alike`).
    pub constraint_decided_on: Option<WorkerDecoding>,

    /// With forwarded references: which of the workers that process them the
    /// plan may use (the gateway's own pipeline alone for a family only it
    /// expands, the engine's own processors alone for audio, either
    /// otherwise). Set in preparation with `multimodal_refs`; worker
    /// selection and the retry pin read it.
    pub media_refs_pipeline: PipelinePin,

    /// `Some` iff the request is multimodal EPD and worker selection produced
    /// encode assignments: the encode stage's plan, by handoff. Request building
    /// injects a rendezvous plan's bootstrap info and drops the prefill pixels;
    /// request execution `take()`s the plan (a push plan's encode call runs before
    /// the prefill/decode dispatch).
    pub encode_outputs: Option<EncodeOutputs>,

    /// Share of the in-flight media budget this request holds until the
    /// engines have its body.
    pub multimodal_inflight: Option<InflightPermit>,

    /// Resolved tokenizer (set once in preparation, reused in response processing)
    /// This avoids redundant registry lookups across pipeline stages.
    pub tokenizer: Option<Arc<dyn Tokenizer>>,

    // Stage 2: Worker selection outputs
    pub workers: Option<WorkerSelection>,

    /// Effective sticky key (header wins, rid-derived falls back), recorded by
    /// worker selection so load guards account keyed load identically.
    pub sticky_key: Option<String>,

    /// Selection inputs that survive the request drop, captured by worker
    /// selection for per-attempt re-selection in the dispatch phase.
    pub routing_snapshot: Option<RoutingSnapshot>,

    // Stage 3: Client acquisition outputs
    pub clients: Option<ClientSelection>,

    /// Prefill admission slot, taken during PD/EPD worker selection and
    /// handed to the dispatch that runs the Prefill side.
    pub pd_prefill_guard: Option<PrefillLoadGuard>,

    // Response processing state seeded during ingress (stop decoder, router
    // stop obligations, derived skip_special_tokens).
    pub response: ResponseState,
}

/// Worker-selection inputs that outlive the parsed request. The prompt's
/// token ids are not kept: a retry reads them off the retained plan
/// ([`ExecutionPlan::routing_token_ids`]).
#[derive(Default)]
pub(crate) struct RoutingSnapshot {
    /// Exact context checks retained for initial selection and retries.
    pub context_requirements: super::common::stages::ContextRequirements,
    /// Captured when the prompt has no token ids, or when a configured policy
    /// reads request text next to them.
    pub routing_text: Option<String>,
    /// The body's `rid`, kept at first selection where a retry reads it: the
    /// sticky key derives from it again. It stands in the derived key's
    /// place, so the dispatch context keeps its size class.
    pub rid: Option<String>,
    /// The request's cache namespace, derived once at first selection.
    pub cache_namespace: Option<CacheNamespace>,
    /// The request carries pixel media (an image or a video), noted at first
    /// selection: a retry keeps the workers that take it (regular mode's
    /// single worker; the prefill side of prefill/decode mode, see worker
    /// selection's `prefill_side_needs_vision`) on ones with a vision encoder,
    /// as the first attempt did.
    pub needs_vision: bool,
    /// The request's disaggregated decode side needs a worker with a vision
    /// encoder (worker selection's `decode_side_needs_vision`: prefill/decode
    /// mode only), noted at first selection so a retry re-selects under the
    /// same demand.
    pub decode_needs_vision: bool,
}

pub(crate) use crate::routers::common::placement::{PipelinePin, WireConstraint};

impl WireConstraint {
    fn of(workers: &WorkerSelection, requires_media_refs: bool, pipeline: PipelinePin) -> Self {
        match workers {
            WorkerSelection::Single { worker } => Self {
                runtime: worker.metadata().spec.runtime_type,
                connection: *worker.connection_mode(),
                requires_media_refs,
                pipeline,
            },
            // The prefill side builds the request, so its transport is the
            // plan's (a gRPC servicer, or a TokenSpeed engine's ZMQ wire).
            WorkerSelection::Disaggregated {
                prefill,
                runtime_type,
                ..
            } => Self {
                runtime: *runtime_type,
                connection: *prefill.connection_mode(),
                requires_media_refs,
                pipeline,
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
    pub root_request_id: Option<String>,
    /// The id the request-id middleware gave the request, for a line written
    /// about it outside its span (a retry's re-selection); always present,
    /// unlike `root_request_id`, which cache tracing alone fills.
    pub request_id: Option<String>,
    pub cache_trace: Option<String>,
    /// The parser that read a chat completion, for the response's `x-smg-parser` header.
    pub parser: Option<ParserChoice>,
    pub attempt: u32,
    /// The request's bookkeeping across attempts: circuit-breaker charges
    /// and the workers whose answer was definitive.
    pub ledger: AttemptLedger,
    /// Engines (base URLs) a failed attempt of this request dispatched to;
    /// a retry's re-selection skips them while another available worker is
    /// left (see `placement::PlacementInputs::tried`).
    pub tried: Vec<String>,
    /// Canonical model ID (routing, registries).
    pub model_id: String,
    /// Model the response reports, captured from the request at the build
    /// boundary.
    pub dispatch_model: String,
    pub streaming: bool,
    /// The request's wall-clock second (the `created` its responses report):
    /// the request's one wall-clock read, taken when its charge id was minted
    /// at the edge (see `helpers::request_wall_clock`).
    pub created: u64,
    pub headers: Option<HeaderMap>,
    pub rate_limit_cell: Option<Arc<RateLimitCell>>,
    pub routing: RoutingSnapshot,
    pub wire: WireConstraint,
    /// The selected worker's constrained decoding, when its policy chose the
    /// plan's constraint; a retry re-selects only workers that decide alike.
    pub constraint_decided_on: Option<WorkerDecoding>,
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    pub workers: Option<WorkerSelection>,
    pub sticky_key: Option<String>,
    pub clients: Option<ClientSelection>,
    /// Consumed by the first dispatch; a rendezvous retry re-dispatches only the
    /// prefill/decode sides against the already-running encode jobs (a push plan
    /// carrying media is not retried).
    pub encode_outputs: Option<EncodeOutputs>,
    /// The watch on a rendezvous plan's encode dispatch, made when the first
    /// attempt launched the encode jobs beside its pair (see
    /// `request_execution::launch_encode_dispatch`); a replay's prefill side
    /// waits at the same rooms on the same jobs, so it races the same watch
    /// and a failed send ends its wait too. `None` for a push plan, a
    /// text-only request, or a plan with nothing to encode.
    pub encode_watch: Option<EncodeWatch>,
    pub multimodal_inflight: Option<InflightPermit>,
    /// Set for each attempt before it dispatches. Response processing is
    /// its last reader and takes it.
    pub dispatch: Option<DispatchMetadata>,
    pub load_guards: Option<LoadGuards>,
    /// Prefill admission slot for the next PD/EPD dispatch; a retry's
    /// reselection refills it.
    pub pd_prefill_guard: Option<PrefillLoadGuard>,
    pub response: ResponseState,
}

impl DispatchContext {
    /// Cached tokenizer (cheap Arc clone).
    pub fn tokenizer_arc(&self) -> Option<Arc<dyn Tokenizer>> {
        self.tokenizer.clone()
    }
}

/// An output one pipeline stage leaves on the context for the stages after
/// it, with the failure those stages report when it is not there.
pub(crate) trait StageOutput {
    /// The code and the message of that failure: one pair per output,
    /// whichever stage notices it missing.
    const NOT_COMPLETED: (&'static str, &'static str);
}

impl StageOutput for PreparationOutput {
    const NOT_COMPLETED: (&'static str, &'static str) =
        ("preparation_not_completed", "Preparation not completed");
}

impl StageOutput for WorkerSelection {
    const NOT_COMPLETED: (&'static str, &'static str) = (
        "worker_selection_not_completed",
        "Worker selection not completed",
    );
}

impl StageOutput for ClientSelection {
    const NOT_COMPLETED: (&'static str, &'static str) = (
        "client_acquisition_not_completed",
        "Client acquisition not completed",
    );
}

impl StageOutput for ExecutionResult {
    const NOT_COMPLETED: (&'static str, &'static str) =
        ("no_execution_result", "No execution result");
}

impl StageOutput for DispatchMetadata {
    const NOT_COMPLETED: (&'static str, &'static str) =
        ("dispatch_metadata_not_set", "Dispatch metadata not set");
}

impl StageOutput for Arc<dyn Tokenizer> {
    const NOT_COMPLETED: (&'static str, &'static str) = (
        "tokenizer_not_cached",
        "Tokenizer not cached in context - preparation stage may have been skipped",
    );
}

impl StageOutput for StopSequenceDecoder {
    const NOT_COMPLETED: (&'static str, &'static str) = (
        "stop_decoder_not_initialized",
        "Stop decoder not initialized",
    );
}

impl<T: StageOutput> StageOutput for &T {
    const NOT_COMPLETED: (&'static str, &'static str) = T::NOT_COMPLETED;
}

impl<T: StageOutput> StageOutput for &mut T {
    const NOT_COMPLETED: (&'static str, &'static str) = T::NOT_COMPLETED;
}

/// `output` as the stage `function` found it on the context, or the failure
/// every stage reports when the earlier stage that produces it has not run:
/// logged once, under the output's one code, whichever stage noticed.
pub(crate) fn stage_output<T: StageOutput>(
    output: Option<T>,
    function: &'static str,
) -> Result<T, axum::response::Response> {
    output.ok_or_else(|| {
        let (code, message) = T::NOT_COMPLETED;
        error!(function = function, "{message}");
        internal_error(code, message)
    })
}

/// The failure a response-processing stage `function` reports when the
/// dispatch handed it another request kind's [`ResponseSpec`]: logged once
/// under the stage that noticed, one code for every stage.
pub(crate) fn wrong_response_spec(function: &'static str) -> axum::response::Response {
    error!(function = function, "Wrong response spec");
    internal_error("wrong_response_spec", "Wrong response spec")
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
    /// The tool constraint as each attempt's worker must take it: a forced
    /// call's before any worker shaped it, with the tool parser that shapes
    /// it, or what derives the table's `auto` tag for a worker's grammar
    /// start; retry attempts re-shape it for the newly selected worker's
    /// engine, as the build did for the first.
    pub tool_constraint: Option<ToolConstraintStamp>,
    /// Mode::PrefillDecode only (mirrors the build stage's flag).
    pub inject_pd_metadata: bool,
}

/// What the encode stage planned for an EPD request, by its handoff.
///
/// Not `#[derive(Debug)]`: `EncodeDispatchPlan` transitively holds
/// non-`Debug` raw proto payloads (`TokenSpeedMultimodalItem`).
pub(crate) enum EncodeOutputs {
    /// Per-item bootstrap rendezvous info for the prefill request, plus the
    /// dispatch plan that fans out to the encode workers beside the
    /// prefill/decode pair. Owns the encode jobs' SHM/RDMA Drop guards:
    /// dropping this before request execution dispatches (early return /
    /// cancellation) reclaims the staged `/dev/shm` segments via
    /// `PreparedEncodeItem`'s `Drop`.
    Rendezvous {
        bootstrap_info: Vec<EncodeItemBootstrapInfo>,
        dispatch: EncodeDispatchPlan,
    },
    /// The encoder the built prompt goes to first, at dispatch (under an image
    /// anchor, the encoders its images fan out to), and the consumer
    /// destination its transfers go to.
    Push(PushEncodePlan),
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
    /// Batched embedding fan-out: one backend request per input, in the
    /// request's order, answered as one list. Sub-request ids are
    /// `{shared_request_id}-p{i}`.
    EmbedBatch {
        shared_request_id: String,
        requests: Vec<ProtoEmbedRequest>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutionPlanKind {
    Single,
    PrefillDecode,
    EncodePrefillDecode,
}

/// The mode's settings for the request a stage builds: which execution plan
/// it becomes, and whether the prefill-decode metadata is injected into it.
/// Read once off the mode by the pipeline and held by every request-building
/// stage as one value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanSettings {
    pub inject_pd_metadata: bool,
    pub plan_kind: ExecutionPlanKind,
}

impl PlanSettings {
    /// The mode-bearing part of a stage's descriptor in the pipeline parity
    /// test: one form for every stage that holds the settings.
    #[cfg(test)]
    pub(crate) fn signature(self) -> String {
        format!(
            "inject_pd_metadata={}, {:?}",
            self.inject_pd_metadata, self.plan_kind
        )
    }
}

impl ExecutionPlan {
    /// Whether a retry of this plan would replay a replica-bound encode
    /// handoff: under a push handoff (`handoff`, the selected runtime's
    /// row) the encoder rewrites an attempt's clone with handles bound to
    /// the prefill replica it reserved on, and the retained pre-encode plan
    /// cannot replay them after any attempt failure. A plan without media
    /// encodes nothing, and a rendezvous handoff keeps its rooms across a
    /// re-selected pair.
    pub(crate) fn replays_replica_bound_handoff(&self, handoff: Option<EncodeHandoff>) -> bool {
        if handoff != Some(EncodeHandoff::Push) {
            return false;
        }
        let carries_media = |request: &ProtoGenerateRequest| {
            request.has_mm_inputs() || request.has_vllm_media_refs()
        };
        match self {
            Self::EncodePrefillDecode { request } => carries_media(request),
            Self::Batch {
                kind: ExecutionPlanKind::EncodePrefillDecode,
                requests,
                ..
            } => requests.iter().any(carries_media),
            _ => false,
        }
    }

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
            }
            | Self::EmbedBatch {
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
            Self::Single(ProtoRequest::Embed(_)) | Self::EmbedBatch { .. } => "embed",
        }
    }

    pub(crate) fn mode_label(&self) -> &'static str {
        match self {
            Self::Single(_) | Self::EmbedBatch { .. } => "single",
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
            Self::EmbedBatch { requests, .. } => {
                requests.iter().map(ProtoEmbedRequest::wire_len).sum()
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
            Self::Batch { .. } | Self::EmbedBatch { .. } => {
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

    /// The prompt's token ids as a retry's routing hint (a batch's first
    /// prompt, as the first selection used): the plan is retained for as long
    /// as a retry can happen, so the routing snapshot keeps no copy of them.
    pub(crate) fn routing_token_ids(&self) -> &[u32] {
        match self {
            Self::Single(request) => request.token_ids(),
            Self::PrefillDecode(request) | Self::EncodePrefillDecode { request } => {
                request.token_ids()
            }
            Self::Batch { requests, .. } => requests.first().map_or(&[], |r| r.token_ids()),
            Self::EmbedBatch { requests, .. } => requests.first().map_or(&[], |r| r.token_ids()),
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
            Self::Single(ProtoRequest::Embed(_)) | Self::EmbedBatch { .. } => [].iter_mut(),
            Self::Batch { requests, .. } => requests.iter_mut(),
        }
    }
}

/// Output from preparation stage (Step 1)
///
/// Each request type produces its own variant, eliminating optional fields
/// that are always None for certain pipelines.
pub(crate) enum PreparationOutput {
    Decisions {
        items: Vec<DecisionPrompt>,
        scoring: DecisionScoring,
    },
    Chat {
        token_ids: Vec<u32>,
        processed_messages: super::ProcessedMessages,
        /// Request building decides on the selected worker
        /// (`regular::symphony::worker_constraint`).
        tool_constraints: PreparedConstraint,
    },
    Messages {
        token_ids: Vec<u32>,
        processed_messages: super::ProcessedMessages,
        tool_constraints: PreparedConstraint,
        /// The response spec, built by preparation: the template rendered
        /// the view of its chat tools that `tool_choice` leaves (see
        /// `MessagesResponseSpec`).
        spec: Box<super::spec::MessagesResponseSpec>,
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
    /// The inputs of an embedding or classification request, one per input
    /// of the batch, in the request's order.
    Embedding { inputs: Vec<EmbeddingInput> },
    Harmony {
        token_ids: Vec<u32>,
        selection_text: String,
        tool_constraints: Option<(String, String)>,
        /// Request with response_format cleared (when converted to structural tag)
        modified_request: Option<Box<ChatCompletionRequest>>,
        harmony_stop_ids: Vec<u32>,
    },
}

/// One tokenized completion prompt.
pub(crate) struct CompletionItem {
    pub text: String,
    pub token_ids: Vec<u32>,
}

/// One input of an embedding or classification request: a text the gateway
/// tokenized, or the token ids the client sent (no text beside them; the
/// engine embeds from the ids).
pub(crate) struct EmbeddingInput {
    pub text: Option<String>,
    pub token_ids: Vec<u32>,
}

impl PreparationOutput {
    /// Token IDs (common to all variants). Batched completions expose the
    /// first prompt's tokens as the routing-affinity proxy.
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Decisions { items, .. } => {
                items.first().map_or(&[], |item| item.token_ids.as_slice())
            }
            Self::Chat { token_ids, .. }
            | Self::Messages { token_ids, .. }
            | Self::Transcription { token_ids, .. }
            | Self::Generate { token_ids, .. }
            | Self::Harmony { token_ids, .. } => token_ids,
            Self::Completion { items, .. } => {
                items.first().map_or(&[], |item| item.token_ids.as_slice())
            }
            Self::Embedding { inputs } => inputs
                .first()
                .map_or(&[], |input| input.token_ids.as_slice()),
        }
    }

    /// Total input token count across every item -- for accounting (rate-limit
    /// reservation), not routing. Unlike `token_ids()`, which exposes only the
    /// first prompt as a routing-affinity proxy, a batched `Completion`
    /// request's real input cost is the sum of every prompt in the batch.
    pub fn total_input_token_count(&self) -> usize {
        match self {
            Self::Decisions { items, .. } => items.iter().map(|item| item.token_ids.len()).sum(),
            Self::Completion { items, .. } => items.iter().map(|item| item.token_ids.len()).sum(),
            Self::Embedding { inputs } => inputs.iter().map(|input| input.token_ids.len()).sum(),
            other => other.token_ids().len(),
        }
    }

    /// Longest single input in tokens -- what the engine's context window
    /// bounds. Every prompt of a batched `Completion` and every input of an
    /// `Embedding` batch is dispatched as its own engine request, so the
    /// window applies per item, not to their sum and not to the first alone.
    pub fn max_input_token_count(&self) -> usize {
        match self {
            Self::Decisions { items, .. } => items
                .iter()
                .map(|item| item.token_ids.len())
                .max()
                .unwrap_or(0),
            Self::Completion { items, .. } => items
                .iter()
                .map(|item| item.token_ids.len())
                .max()
                .unwrap_or(0),
            Self::Embedding { inputs } => inputs
                .iter()
                .map(|input| input.token_ids.len())
                .max()
                .unwrap_or(0),
            other => other.token_ids().len(),
        }
    }

    /// Text for worker routing: original_text for regular pipelines, selection_text for Harmony.
    /// Chat/Messages borrow from processed_messages.text to avoid a redundant clone.
    pub fn routing_text(&self) -> Option<&str> {
        match self {
            Self::Decisions { items, .. } => items.first().map(|item| item.text.as_str()),
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
            Self::Embedding { inputs } => inputs.first().and_then(|input| input.text.as_deref()),
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
    /// The one worker of a single request, as its placement answered it: the
    /// request's load guard is made from the answer ([`LoadGuards::new`]).
    Single { worker: Placed },
    /// Disaggregated prefill/decode selection. EPD layers per-item encode
    /// assignments on top; plain PD leaves `encode_assignments` unset.
    ///
    /// The decode is kept as its placement answered it: its load guard is
    /// made from the answer ([`LoadGuards::new`],
    /// [`LoadGuards::behind_prefill`]). The prefill's guard was made from
    /// its answer where the Prefill admission took its slot.
    Disaggregated {
        encode_assignments: Option<Vec<EncodeWorkerAssignment>>,
        prefill: Arc<dyn Worker>,
        decode: Placed,
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
    /// Disaggregated guards cover the decode side only. The Prefill load is
    /// a `PrefillLoadGuard` that drops when the Prefill phase ends.
    Disaggregated {
        _decode: WorkerLoadGuard,
    },
    /// Batched completion fan-out: one guard set per sub-request so load-aware
    /// policies see the real backend concurrency.
    Batch {
        _guards: Vec<LoadGuards>,
    },
    /// A disaggregated dispatch whose bootstrap rooms the PD admission gate
    /// claimed before it was allowed to send. The claim rides with the guards
    /// so it is released on every path the dispatch can end on — an early
    /// error, a failed side, a client disconnect, a retry, or completion.
    Admitted {
        _admission: PdAdmissionGuard,
        _guards: Box<LoadGuards>,
    },
    /// The dispatch's upstream span rides with the guards, so it ends when
    /// they are released: at the response stream's last frame, the client's
    /// disconnect, an upstream error or a retry, not at dispatch.
    Traced {
        _span: StreamSpan,
        _guards: Box<LoadGuards>,
    },
}

/// A span held open across a response stream. Released, it is entered once
/// more: the OpenTelemetry layer stamps a span's end at its last exit, not
/// at its close, so a span merely held would still end at dispatch.
pub(crate) struct StreamSpan(tracing::Span);

impl Drop for StreamSpan {
    fn drop(&mut self) {
        let _last_exit = self.0.enter();
    }
}

impl LoadGuards {
    /// The guards of an attempt whose every side goes out with the dispatch
    /// that follows: each is made from its placement's answer, which says
    /// the dispatch where the policy engine placed the request.
    // Always in line: on every request's path, and the call cost more than the body.
    #[inline(always)]
    pub fn new(selection: &WorkerSelection, routing_key: Option<&str>) -> Self {
        match selection {
            WorkerSelection::Single { worker } => LoadGuards::Single {
                _guard: worker.guard(routing_key),
            },
            WorkerSelection::Disaggregated { decode, .. } => LoadGuards::Disaggregated {
                _decode: decode.guard(routing_key),
            },
        }
    }

    /// [`Self::new`] for a pair whose decode is owed its ask and its send
    /// behind its prefill ([`Placed::owed_behind_prefill`]): one request of
    /// a sequential dispatch that the policy engine placed. A classic
    /// pair's takes [`Self::new`]. The decode's guard is made
    /// ahead of its send ([`Placed::ahead`]): the decode is counted from
    /// here, and the dispatch says its send where the decode goes out
    /// ([`Self::map_decode`], in `execute_sequential_pd`).
    pub fn behind_prefill(selection: &WorkerSelection, routing_key: Option<&str>) -> Self {
        match selection {
            WorkerSelection::Disaggregated { decode, .. } => LoadGuards::Disaggregated {
                _decode: decode.ahead(routing_key),
            },
            single @ WorkerSelection::Single { .. } => Self::new(single, routing_key),
        }
    }

    /// Hands the decode's guard of a pair to `with` and keeps the guard it
    /// answers in its place: for the decode asked again, which may move the
    /// guard to another worker, and for the decode's send. The first decode
    /// guard of a batch takes the call. Guards that hold no decode's are
    /// left as they are.
    pub(crate) fn map_decode(&mut self, with: impl FnOnce(WorkerLoadGuard) -> WorkerLoadGuard) {
        let Some(held) = self.decode_guards() else {
            return;
        };
        // The decode's guard is moved out for the call and back, where it
        // stands: the placeholder counts nothing and allocates nothing.
        let empty = Self::Batch {
            _guards: Vec::new(),
        };
        *held = match std::mem::replace(held, empty) {
            Self::Disaggregated { _decode } => Self::Disaggregated {
                _decode: with(_decode),
            },
            other => other,
        };
    }

    /// The guards that hold a pair's decode guard, under whatever rides
    /// with them: the first of a batch.
    fn decode_guards(&mut self) -> Option<&mut Self> {
        match self {
            Self::Disaggregated { .. } => Some(self),
            Self::Batch { _guards } => _guards.iter_mut().find_map(Self::decode_guards),
            Self::Admitted { _guards, .. } | Self::Traced { _guards, .. } => {
                _guards.decode_guards()
            }
            Self::Single { .. } => None,
        }
    }

    /// The decode moved to another worker behind its prefill: `claim` is
    /// what that worker's gate seated for the dispatch, `None` where its
    /// engine reports no window. It takes the place of the claim at the
    /// first decode's gate, which is let go here.
    pub(crate) fn claim_moved(&mut self, claim: Option<PdAdmissionGuard>) {
        let guards = std::mem::replace(
            self,
            Self::Batch {
                _guards: Vec::new(),
            },
        );
        let guards = match guards {
            Self::Admitted { _guards, .. } => *_guards,
            guards => guards,
        };
        *self = Self::admitted(claim, guards);
    }

    /// Bind an admission claim to the dispatch's guards, so the rooms it
    /// reserved outlive nothing else. `None` (the engine reports no window)
    /// returns the guards untouched.
    // Always in line: on every request's path, and the call cost more than the body.
    #[inline(always)]
    pub fn admitted(admission: Option<PdAdmissionGuard>, guards: Self) -> Self {
        match admission {
            Some(admission) => Self::Admitted {
                _admission: admission,
                _guards: Box::new(guards),
            },
            None => guards,
        }
    }

    /// Keep `span` open for as long as the guards are held.
    pub fn traced(span: tracing::Span, guards: Self) -> Self {
        Self::Traced {
            _span: StreamSpan(span),
            _guards: Box::new(guards),
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

    /// The engine's counters of a non-streamed chat, messages or
    /// transcription answer, as the response stage recorded them for the
    /// settlement: the client-facing usage drops the rendered prompt tokens
    /// the provider does not bill and may carry the gateway parser's
    /// reasoning split instead of the engine's count, the Messages shape
    /// reports no cache hits at all, and a transcript carries no usage.
    /// `None` for the shapes whose response carries the engine's counters
    /// unchanged (completion, generate, decisions, embeddings, classify).
    pub engine_counts: Option<TokenCounts>,

    /// Responses API iteration result (Harmony only, for tool loop
    /// orchestration). Behind a pointer: the field is empty on every other
    /// path, and the dispatch context is boxed once per request; inline, its
    /// 208 bytes (a usage object among them) had pushed that box over the
    /// allocator's 2,560-byte class edge on every chat, completion and
    /// Messages request.
    pub responses_iteration_result: Option<Box<super::harmony::ResponsesIterationResult>>,
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
            RequestType::Embedding(_) | RequestType::Classify(_) | RequestType::Decisions(_) => {
                false
            }
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
    ///
    /// Both contexts are a few KiB wide, so both travel boxed: this one
    /// comes in its box, and the dispatch context leaves in the box the run
    /// keeps it in.
    pub fn into_dispatch(
        self: Box<Self>,
    ) -> Result<Box<DispatchContext>, axum::response::Response> {
        // Fields leave the box one at a time: a local that bound `state`
        // whole would copy all of it to the stack first.
        let RequestInput {
            request_type,
            headers,
            model_id,
            streaming,
            tenant_request_meta,
            rate_limit_cell,
        } = self.input;
        // The model the response reports. `RequestContext::new` already
        // canonicalized both the request's `model` field and `model_id`, so a
        // request that arrived under an alias is answered under the canonical
        // name. Native `/generate` callers may leave the field empty, so
        // prefer the resolved id there.
        let dispatch_model = match &request_type {
            RequestType::Chat(req) => openai_response_model(&req.model, req.lora_path.as_deref()),
            RequestType::Completion(req) => {
                openai_response_model(&req.model, req.lora_path.as_deref())
            }
            RequestType::Generate(_) => model_id.clone(),
            RequestType::Responses(req) => req.model.clone(),
            RequestType::Embedding(req) => req.model.clone(),
            RequestType::Classify(req) => req.model.clone(),
            RequestType::Decisions(req) => req.model.clone(),
            RequestType::Messages(req) => req.model.clone(),
            RequestType::Transcription { request, .. } => request.model.clone(),
        };
        drop(request_type);
        let parser = self.state.parser;
        drop(self.components);
        let routing = self.state.routing_snapshot.ok_or_else(|| {
            error!(
                function = "RequestContext::into_dispatch",
                "Routing snapshot not captured by worker selection"
            );
            internal_error(
                "routing_snapshot_not_captured",
                "Routing snapshot not captured",
            )
        })?;
        let wire = WireConstraint::of(
            stage_output(self.state.workers.as_ref(), "RequestContext::into_dispatch")?,
            self.state.media_refs_forwarded,
            if self.state.media_refs_forwarded {
                self.state.media_refs_pipeline
            } else {
                PipelinePin::Any
            },
        );
        let request_id =
            super::common::stages::helpers::middleware_request_id(tenant_request_meta.as_ref())
                .map(str::to_owned);
        let (created, _) =
            super::common::stages::helpers::request_wall_clock(tenant_request_meta.as_ref());
        Ok(Box::new(DispatchContext {
            root_request_id: if cache_trace::enabled() {
                request_id.clone()
            } else {
                None
            },
            request_id,
            cache_trace: None,
            parser,
            attempt: 0,
            ledger: AttemptLedger::default(),
            tried: Vec::new(),
            model_id,
            dispatch_model,
            streaming,
            created,
            headers,
            rate_limit_cell,
            routing,
            wire,
            constraint_decided_on: self.state.constraint_decided_on,
            tokenizer: self.state.tokenizer,
            workers: self.state.workers,
            sticky_key: self.state.sticky_key,
            clients: self.state.clients,
            encode_outputs: self.state.encode_outputs,
            encode_watch: None,
            multimodal_inflight: self.state.multimodal_inflight,
            dispatch: None,
            load_guards: None,
            pd_prefill_guard: self.state.pd_prefill_guard,
            response: self.state.response,
        }))
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

    /// Create context for a Decisions scoring request.
    pub fn for_decisions(
        request: Arc<DecisionsRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        Self::new(
            RequestType::Decisions(request),
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

    /// A Messages context: it carries the request's tools beside the request
    /// (`tools`, taken out of the request by the route while it still owned
    /// it), and the preparation builds the response spec from them.
    pub fn for_messages(
        request: Arc<CreateMessageRequest>,
        tools: MessagesTools,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        let mut ctx = Self::new(
            RequestType::Messages(request),
            headers,
            model_id,
            components,
        );
        ctx.state.messages_tools = Some(tools);
        ctx
    }

    /// A Messages context over a request that still carries its tools, as the
    /// tests build theirs: the tools are converted from the request by
    /// reference.
    #[cfg(test)]
    pub fn for_messages_carrying_its_tools(
        request: Arc<CreateMessageRequest>,
        headers: Option<HeaderMap>,
        model_id: String,
        components: Arc<SharedComponents>,
    ) -> Self {
        let tools = MessagesTools::from(&*request);
        Self::for_messages(request, tools, headers, model_id, components)
    }

    /// Get Arc clone of chat request (panics if not chat)
    pub fn chat_request_arc(&self) -> Arc<ChatCompletionRequest> {
        let RequestType::Chat(req) = &self.input.request_type else {
            expected_shape("chat")
        };
        Arc::clone(req)
    }

    /// Get Arc clones of the transcription request and its audio (panics if
    /// not a transcription request).
    pub fn transcription_input_arc(&self) -> (Arc<TranscriptionRequest>, Arc<AudioFile>) {
        let RequestType::Transcription { request, audio } = &self.input.request_type else {
            expected_shape("transcription")
        };
        (Arc::clone(request), Arc::clone(audio))
    }

    /// Get Arc clone of generate request (panics if not generate)
    pub fn generate_request_arc(&self) -> Arc<GenerateRequest> {
        let RequestType::Generate(req) = &self.input.request_type else {
            expected_shape("generate")
        };
        Arc::clone(req)
    }

    /// Get Arc clone of completion request (panics if not completion)
    pub fn completion_request_arc(&self) -> Arc<CompletionRequest> {
        let RequestType::Completion(req) = &self.input.request_type else {
            expected_shape("completion")
        };
        Arc::clone(req)
    }

    /// Get Arc clone of responses request (panics if not responses)
    pub fn responses_request_arc(&self) -> Arc<ResponsesRequest> {
        let RequestType::Responses(req) = &self.input.request_type else {
            expected_shape("responses")
        };
        Arc::clone(req)
    }

    /// Get Arc clone of messages request (panics if not messages)
    pub fn messages_request_arc(&self) -> Arc<CreateMessageRequest> {
        let RequestType::Messages(req) = &self.input.request_type else {
            expected_shape("messages")
        };
        Arc::clone(req)
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

/// The typed accessors' refusal: the caller read a request shape the context
/// does not hold. The shape is fixed at construction, so a wrong read is a
/// programming error, never a request's. The panic names the accessor's line.
#[expect(
    clippy::panic,
    reason = "typed accessor: caller guarantees variant via RequestType construction"
)]
#[track_caller]
fn expected_shape(shape: &str) -> ! {
    panic!("Expected {shape} request")
}

impl WorkerSelection {
    pub fn single(&self) -> Option<&Arc<dyn Worker>> {
        match self {
            Self::Single { worker } => Some(worker.worker()),
            Self::Disaggregated { .. } => None,
        }
    }

    /// The workers this selection dispatches to: the one worker, or the
    /// prefill and decode sides.
    pub fn workers(&self) -> impl Iterator<Item = &Arc<dyn Worker>> {
        let (first, second) = match self {
            Self::Single { worker } => (worker.worker(), None),
            Self::Disaggregated {
                prefill, decode, ..
            } => (prefill, Some(decode.worker())),
        };
        std::iter::once(first).chain(second)
    }

    /// The engines (base URLs) this selection dispatches to: the one worker,
    /// or the prefill and decode sides.
    pub fn engines(&self) -> impl Iterator<Item = &str> {
        self.workers().map(|worker| worker.base_url())
    }

    /// Record an attempt's circuit-breaker outcome on every worker, through
    /// the request's ledger (one failure charge per request per worker).
    pub fn record_outcome(&self, ledger: &AttemptLedger, status_code: u16) {
        match self {
            Self::Single { worker } => ledger.record_outcome(&**worker, status_code),
            Self::Disaggregated {
                prefill, decode, ..
            } => {
                // EPD encode dispatch is asynchronous and supervised by
                // RequestExecution; this records only the prefill/decode side.
                ledger.record_outcome(prefill.as_ref(), status_code);
                ledger.record_outcome(&**decode, status_code);
            }
        }
    }

    /// Record circuit breaker outcomes for disaggregated dispatch (individual tracking)
    pub fn record_prefill_decode_outcomes(
        &self,
        ledger: &AttemptLedger,
        prefill_status: Option<u16>,
        decode_status: Option<u16>,
    ) {
        if let Self::Disaggregated {
            prefill, decode, ..
        } = self
        {
            if let Some(status) = prefill_status {
                ledger.record_outcome(prefill.as_ref(), status);
            }
            if let Some(status) = decode_status {
                ledger.record_outcome(&**decode, status);
            }
        }
    }

    /// Record circuit breaker outcome for prefill worker only (sequential PD)
    pub fn record_outcome_prefill(&self, ledger: &AttemptLedger, status_code: u16) {
        match self {
            Self::Disaggregated { prefill, .. } => {
                ledger.record_outcome(prefill.as_ref(), status_code);
            }
            Self::Single { .. } => {
                debug!("record_outcome_prefill called on Single worker selection, ignoring");
            }
        }
    }

    /// Record circuit breaker outcome for decode worker only (sequential PD)
    pub fn record_outcome_decode(&self, ledger: &AttemptLedger, status_code: u16) {
        match self {
            Self::Disaggregated { decode, .. } => {
                ledger.record_outcome(&**decode, status_code);
            }
            Self::Single { .. } => {
                debug!("record_outcome_decode called on Single worker selection, ignoring");
            }
        }
    }

    /// The prefill and decode workers of a disaggregated selection, for the
    /// tests that pin which side a dispatch addressed.
    #[cfg(test)]
    #[expect(clippy::type_complexity)]
    pub fn disaggregated_pair(&self) -> Option<(&Arc<dyn Worker>, &Arc<dyn Worker>)> {
        match self {
            Self::Disaggregated {
                prefill, decode, ..
            } => Some((prefill, decode.worker())),
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
        self.decode().map(Placed::worker)
    }

    /// A pair's decode as its placement answered it, which its load guard
    /// is made from.
    pub(crate) fn decode(&self) -> Option<&Placed> {
        match self {
            Self::Disaggregated { decode, .. } => Some(decode),
            Self::Single { .. } => None,
        }
    }

    /// This pair with `decode` in its decode's place: the selection of a
    /// dispatch whose decode moved behind its prefill
    /// (`placement::decode_again`). `None` for a single worker's.
    pub(crate) fn with_decode(&self, decode: Placed) -> Option<Self> {
        match self {
            Self::Disaggregated {
                encode_assignments,
                prefill,
                runtime_type,
                ..
            } => Some(Self::Disaggregated {
                encode_assignments: encode_assignments.clone(),
                prefill: Arc::clone(prefill),
                decode,
                runtime_type: *runtime_type,
            }),
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

    /// The encode handoff of the selected runtime's EPD form (its row of
    /// the disaggregation table); `None` for a single worker or a runtime
    /// without an encode role.
    pub fn encode_handoff(&self) -> Option<EncodeHandoff> {
        self.disaggregated_runtime_type()
            .and_then(|runtime| PdProtocol::encode_handoff(*runtime))
    }
}

impl ClientSelection {
    /// The client that builds the request: the single engine's, or the
    /// prefill's when disaggregated.
    pub fn builder(&self) -> &BackendClient {
        match self {
            Self::Single { client } => client,
            Self::Disaggregated { prefill, .. } => prefill,
        }
    }

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

    /// The single engine's client for an executor of single dispatches, or
    /// the client-shape mismatch answer, logged under `function`. Client
    /// acquisition shapes the selection after the worker selection the plan
    /// kind comes from, so no executor meets the mismatch; the answer says
    /// so if one ever does.
    pub fn single_or_mismatch(
        &mut self,
        function: &'static str,
    ) -> Result<&mut BackendClient, axum::response::Response> {
        self.single_mut().ok_or_else(|| {
            error!(function, "Expected single client but got disaggregated");
            internal_error(
                "expected_single_client_got_disaggregated",
                "Expected single client but got disaggregated",
            )
        })
    }

    /// The prefill and decode clients for an executor of disaggregated
    /// dispatches, or the client-shape mismatch answer, logged under
    /// `function` (see [`Self::single_or_mismatch`]).
    pub fn disaggregated_or_mismatch(
        &mut self,
        function: &'static str,
    ) -> Result<(&mut BackendClient, &mut BackendClient), axum::response::Response> {
        self.disaggregated_mut().ok_or_else(|| {
            error!(function, "Expected disaggregated clients but got single");
            internal_error(
                "expected_disaggregated_clients_got_single",
                "Expected disaggregated clients but got single",
            )
        })
    }
}

/// Result of request execution (streams from workers)
/// Uses ProtoStream to automatically abort on cancellation
pub(crate) enum ExecutionResult {
    Single {
        stream: ProtoStream,
        /// Captured before the RPC, whose headers may wait for the first chunk.
        dispatch_start: Instant,
    },
    PrefillDecode {
        prefill: ProtoStream,
        decode: Box<ProtoStream>,
        /// Guards indexed by fan-out sample. Each starts as `Some` and is taken
        /// when that sample completes; siblings retain their guards. A side
        /// whose prefill already ran to completion in the execution stage
        /// (sequential PD) carries `None` — there is no live guard to release.
        prefill_guards: Vec<Option<PrefillLoadGuard>>,
        /// PD timing context, for honest PD TTFT (prefill start to first decode token).
        pd_timing: PdTiming,
        /// Input (prompt) logprobs harvested from the prefill `Complete` frame.
        /// `Some` only for sequential PD, where the execution stage drains the
        /// prefill stream before the streaming layer runs; parallel/fan-out PD
        /// leaves this `None` and the streaming layer drains the stream itself.
        prefill_input_logprobs: Option<ProtoInputLogProbs>,
    },
    /// An embedding request's answers, one per input in the request's
    /// order: each a response in hand, not a stream.
    Embedding { responses: Vec<ProtoEmbedComplete> },
    /// Batched completion fan-out: one result per prompt, in prompt order.
    Batch { results: Vec<ExecutionResult> },
}

impl ExecutionResult {
    /// Every stream of the result read for the client at the receiving end
    /// of `client`'s channel (see [`ProtoStream::until_client_gone`]): the
    /// one worker's stream, both sides of a disaggregated pair (the prefill
    /// side is drained in the streaming task, after the response began), each
    /// result of a batch. An embedding has no stream.
    #[must_use]
    pub(crate) fn until_client_gone(self, client: &SseSender, request_id: &str) -> Self {
        match self {
            Self::Single {
                stream,
                dispatch_start,
            } => Self::Single {
                stream: stream.until_client_gone(client, request_id),
                dispatch_start,
            },
            Self::PrefillDecode {
                prefill,
                decode,
                prefill_guards,
                pd_timing,
                prefill_input_logprobs,
            } => Self::PrefillDecode {
                prefill: prefill.until_client_gone(client, request_id),
                decode: Box::new(decode.until_client_gone(client, request_id)),
                prefill_guards,
                pd_timing,
                prefill_input_logprobs,
            },
            Self::Embedding { .. } => self,
            Self::Batch { results } => Self::Batch {
                results: results
                    .into_iter()
                    .map(|result| result.until_client_gone(client, request_id))
                    .collect(),
            },
        }
    }

    /// Every stream of the result under the request's `deadline` (see
    /// [`ProtoStream::apply_deadline`]), in place: the one worker's, both sides of a
    /// disaggregated pair, each result of a batch. An embedding is one
    /// answer, already in hand.
    pub(crate) fn apply_deadline(&mut self, deadline: RequestDeadline) {
        match self {
            Self::Single { stream, .. } => stream.apply_deadline(deadline),
            Self::PrefillDecode {
                prefill, decode, ..
            } => {
                prefill.apply_deadline(deadline);
                decode.apply_deadline(deadline);
            }
            Self::Embedding { .. } => {}
            Self::Batch { results } => {
                for result in results {
                    result.apply_deadline(deadline);
                }
            }
        }
    }
}

/// A request's wait ahead of the dispatch its answer's clocks start at, as
/// [`dispatch_wait`] reads it; `None` when no entry stamp is in scope.
pub(crate) type DispatchWait = Option<std::time::Duration>;

/// The wait of the request whose answering attempt was dispatched at
/// `dispatched` (the single RPC's dispatch, the prefill dispatch of a pair):
/// the time since its entry at the edge ([`RequestStarted`]). The body read,
/// auth, admission, tokenizing, templating and worker selection are in it,
/// and so are a refused attempt with its backoff and an encode call that
/// runs ahead of the pair.
///
/// The stamp is scoped to the request's future. The streaming entry points
/// run inside it, so they read the wait there, against the dispatch instant
/// their stream's clocks start at, and hand it to the reading they build:
/// the response body drives that reading once the request's future has
/// returned, outside the scope.
///
/// A streamed Responses tool loop dispatches its generations from a task of
/// its own and puts each of them under a stamp there: the first turn's under
/// the request's own, a later turn's under one taken where that turn starts.
/// A later turn's wait is therefore the gateway's time from the turn's start
/// to its dispatch, with none of the generations and tool calls before it.
///
/// `None` when no stamp is in scope. Behind the edge every entry point that
/// reads the wait runs under one. A pipeline a test runs bare has none, on
/// any turn of a tool loop either, and its stream records its time to first
/// token alone.
pub(crate) fn dispatch_wait(dispatched: Instant) -> DispatchWait {
    let entered = RequestStarted::current()?;
    Some(dispatched.saturating_duration_since(entered))
}

/// Timing context threaded from PD execution into the streaming layer so the
/// first decode token can be measured against the prefill and decode dispatch
/// instants.
#[derive(Clone)]
pub(crate) struct PdTiming {
    /// Monotonic instant the prefill RPC was dispatched.
    pub prefill_start: Instant,
    /// Monotonic instant the decode RPC was dispatched. Sequential dispatch
    /// sends decode once prefill is drained and relayed; parallel dispatch
    /// sends both sides together, so this is the prefill instant.
    pub decode_sent: Instant,
    /// Backend runtime label (e.g. "sglang", "vllm") for the PD metric set.
    pub runtime: &'static str,
}

impl PdTiming {
    /// Record the PD time-to-first-token pair at the first decode token: the
    /// end-to-end value from the prefill dispatch (`smg_pd_ttft_seconds`) and
    /// the decode phase from the decode dispatch (`smg_pd_decode_ttft_seconds`),
    /// off one clock reading so that the prefill, relay and decode phases add
    /// up to the end-to-end value exactly.
    pub(crate) fn record_first_decode_token(&self, backend_type: &'static str, model: &str) {
        let now = Instant::now();
        Metrics::record_pd_ttft(
            backend_type,
            model,
            self.runtime,
            now.duration_since(self.prefill_start),
        );
        Metrics::record_pd_decode_ttft(
            backend_type,
            model,
            self.runtime,
            now.duration_since(self.decode_sent),
        );
    }
}

/// Final processed response
#[derive(Debug)]
pub(crate) enum FinalResponse {
    Decisions(DecisionResponse),
    Chat(ChatCompletionResponse),
    /// Generate response is a Vec of GenerateResponse (n=1 returns single item, n>1 returns multiple)
    Generate(Vec<GenerateResponse>),
    /// Completion response (OpenAI /v1/completions format)
    Completion(CompletionResponse),
    /// Embedding response
    Embedding(EmbeddingResponse),
    /// Classification response
    Classify(ClassifyResponse),
    /// Messages API response (boxed: the object outgrew the other variants)
    Messages(Box<Message>),
    /// Transcription: the decoded transcript plus its wire format.
    Transcription {
        text: String,
        format: super::spec::TranscriptionResponseFormat,
    },
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::VllmEngineClient;

    use super::*;
    use crate::routers::{
        common::prefill_admission::test_support::select_worker,
        grpc::{client::GrpcClient, test_engine::mock_engine},
    };

    /// Only multimodal vLLM EPD owns EC state absent from its retained plan.
    /// The plan's kind, its media and the handoff kind each decide it
    /// alone: a text-only plan encoded nothing, a rendezvous handoff keeps
    /// its rooms across a re-selected pair, and no handoff has nothing to
    /// replay.
    #[test]
    fn only_an_epd_plan_with_media_under_a_push_handoff_replays_a_replica_bound_handoff() {
        use smg_grpc_client::{tokenspeed_proto, vllm_proto};
        let push = Some(EncodeHandoff::Push);
        let media = ProtoGenerateRequest::Vllm(Box::new(vllm_proto::GenerateRequest {
            media_refs: Some(vllm_proto::MediaRefs::default()),
            ..Default::default()
        }));
        for kind in [
            ExecutionPlanKind::Single,
            ExecutionPlanKind::PrefillDecode,
            ExecutionPlanKind::EncodePrefillDecode,
        ] {
            assert_eq!(
                ExecutionPlan::generate(kind, media.clone()).replays_replica_bound_handoff(push),
                kind == ExecutionPlanKind::EncodePrefillDecode
            );
        }
        assert!(ExecutionPlan::Batch {
            kind: ExecutionPlanKind::EncodePrefillDecode,
            shared_request_id: "batch".to_string(),
            requests: vec![media],
        }
        .replays_replica_bound_handoff(push));
        assert!(!ExecutionPlan::EncodePrefillDecode {
            request: ProtoGenerateRequest::Vllm(Box::default()),
        }
        .replays_replica_bound_handoff(push));
        let items = ProtoGenerateRequest::TokenSpeed(Box::new(tokenspeed_proto::GenerateRequest {
            mm_inputs: Some(tokenspeed_proto::MultimodalInputs::default()),
            ..Default::default()
        }));
        for handoff in [Some(EncodeHandoff::Rendezvous), None] {
            assert!(!ExecutionPlan::EncodePrefillDecode {
                request: items.clone(),
            }
            .replays_replica_bound_handoff(handoff));
        }
    }

    /// The status of a failure a stage reports and the code of its JSON
    /// error object.
    async fn failure_shape(failure: axum::response::Response) -> (http::StatusCode, String) {
        let status = failure.status();
        let body = axum::body::to_bytes(failure.into_body(), usize::MAX)
            .await
            .expect("drain body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
        let code = body["error"]["code"]
            .as_str()
            .unwrap_or_else(|| panic!("an error code: {body}"))
            .to_owned();
        (status, code)
    }

    /// The failure a read of a missing output reports (the outputs have no
    /// `Debug` to `expect_err` against).
    fn failure_of<T>(read: Result<T, axum::response::Response>) -> axum::response::Response {
        match read {
            Ok(_) => panic!("the output was there"),
            Err(failure) => failure,
        }
    }

    /// A disaggregated selection's wire is the prefill side's, which builds
    /// the request: a direct-ZMQ prefill worker pins the plan to the ZMQ wire
    /// of the pair's runtime whatever the decode side's transport, and a gRPC
    /// prefill worker to gRPC.
    #[test]
    fn a_disaggregated_selections_wire_is_the_prefill_sides() {
        use smg_worker::identity::basic::{ConnectionMode, RuntimeType, Worker, WorkerType};

        use crate::routers::grpc::backend_client::BasicWorkerBuilder;

        let worker = |url: &str, mode: ConnectionMode, role: WorkerType| -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new(url)
                    .connection_mode(mode)
                    .runtime_type(RuntimeType::TokenSpeed)
                    .worker_type(role)
                    .build(),
            )
        };
        let pair =
            |prefill: Arc<dyn Worker>, decode: Arc<dyn Worker>| WorkerSelection::Disaggregated {
                encode_assignments: None,
                prefill,
                decode: decode.into(),
                runtime_type: RuntimeType::TokenSpeed,
            };

        let zmq_prefill = pair(
            worker(
                "ipc:///tmp/prefill.ipc",
                ConnectionMode::Zmq,
                WorkerType::Prefill,
            ),
            worker("grpc://decode:1", ConnectionMode::Grpc, WorkerType::Decode),
        );
        let wire = WireConstraint::of(&zmq_prefill, false, PipelinePin::Any);
        assert_eq!(wire.connection, ConnectionMode::Zmq);
        assert_eq!(wire.runtime, RuntimeType::TokenSpeed);
        assert!(!wire.requires_media_refs);

        let grpc_prefill = pair(
            worker(
                "grpc://prefill:1",
                ConnectionMode::Grpc,
                WorkerType::Prefill,
            ),
            worker(
                "ipc:///tmp/decode.ipc",
                ConnectionMode::Zmq,
                WorkerType::Decode,
            ),
        );
        let wire = WireConstraint::of(&grpc_prefill, true, PipelinePin::Any);
        assert_eq!(wire.connection, ConnectionMode::Grpc);
        assert!(wire.requires_media_refs);
    }

    #[tokio::test]
    async fn a_missing_preparation_output_is_one_failure_whichever_stage_reads_it() {
        let mut slot: Option<PreparationOutput> = None;
        for failure in [
            failure_of(stage_output(slot.as_ref(), "a_stage")),
            failure_of(stage_output(slot.take(), "another_stage")),
        ] {
            assert_eq!(
                failure_shape(failure).await,
                (
                    http::StatusCode::INTERNAL_SERVER_ERROR,
                    "preparation_not_completed".to_owned()
                )
            );
        }

        slot = Some(completion_prep(&["a prompt"], None));
        assert!(matches!(
            stage_output(slot.as_ref(), "a_stage"),
            Ok(PreparationOutput::Completion { .. })
        ));
        assert!(matches!(
            stage_output(slot.take(), "the_last_reader"),
            Ok(PreparationOutput::Completion { .. })
        ));
        assert!(slot.is_none(), "the taking reader consumes the output");
    }

    #[tokio::test]
    async fn a_missing_worker_selection_is_one_failure_whichever_stage_reads_it() {
        use smg_worker::identity::basic::WorkerType;

        use crate::routers::grpc::backend_client::BasicWorkerBuilder;

        let mut slot: Option<WorkerSelection> = None;
        let failure = failure_of(stage_output(slot.as_ref(), "a_stage"));
        assert_eq!(
            failure_shape(failure).await,
            (
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "worker_selection_not_completed".to_owned()
            )
        );

        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://prefill-selected")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode-selected")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        slot = Some(pd_selection(&prefill, &decode));
        assert!(matches!(
            stage_output(slot.as_ref(), "a_stage"),
            Ok(WorkerSelection::Disaggregated { .. })
        ));
    }

    #[tokio::test]
    async fn a_missing_client_selection_is_one_failure_whichever_stage_reads_it() {
        let mut slot: Option<ClientSelection> = None;
        for failure in [
            failure_of(stage_output(slot.as_ref(), "a_stage")),
            failure_of(stage_output(slot.as_mut(), "the_dispatch")),
        ] {
            assert_eq!(
                failure_shape(failure).await,
                (
                    http::StatusCode::INTERNAL_SERVER_ERROR,
                    "client_acquisition_not_completed".to_owned()
                )
            );
        }
    }

    #[test]
    fn the_dispatch_phase_outputs_keep_their_codes() {
        assert_eq!(
            <ExecutionResult as StageOutput>::NOT_COMPLETED.0,
            "no_execution_result"
        );
        assert_eq!(
            <DispatchMetadata as StageOutput>::NOT_COMPLETED.0,
            "dispatch_metadata_not_set"
        );
        assert_eq!(
            <Arc<dyn Tokenizer> as StageOutput>::NOT_COMPLETED.0,
            "tokenizer_not_cached"
        );
        assert_eq!(
            <StopSequenceDecoder as StageOutput>::NOT_COMPLETED.0,
            "stop_decoder_not_initialized"
        );
    }

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

    fn pd_selection(prefill: &Arc<dyn Worker>, decode: &Arc<dyn Worker>) -> WorkerSelection {
        WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: Arc::clone(prefill),
            decode: Arc::clone(decode).into(),
            runtime_type: RuntimeType::Sglang,
        }
    }

    #[test]
    fn disaggregated_load_guards_hold_decode_only() {
        use smg_worker::identity::basic::WorkerType;

        use crate::routers::grpc::backend_client::BasicWorkerBuilder;

        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://prefill-load")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode-load")
                .worker_type(WorkerType::Decode)
                .build(),
        );

        let prefill_guard = PrefillLoadGuard::Unbounded {
            _guard: WorkerLoadGuard::with_key(Arc::clone(&prefill), None),
        };
        assert_eq!(prefill.load(), 1);
        drop(prefill_guard);

        let guards = LoadGuards::new(&pd_selection(&prefill, &decode), None);
        assert_eq!(prefill.load(), 0);
        assert_eq!(decode.load(), 1);
        drop(guards);
        assert_eq!(decode.load(), 0);
    }

    #[tokio::test]
    async fn batch_prompts_preserve_phase_load_semantics() {
        use smg_worker::identity::basic::WorkerType;

        use crate::routers::{
            common::prefill_admission::PrefillAdmission, grpc::backend_client::BasicWorkerBuilder,
        };

        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://prefill:30000")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:30000")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        let routing_key = "batch-key";

        let admission = PrefillAdmission::new(1, 0, std::time::Duration::from_secs(1));
        let admitted = match admission
            .admit(Some(routing_key), select_worker(&prefill))
            .await
        {
            Ok(admitted) => admitted,
            Err(_) => panic!("initial admission should succeed"),
        };
        let prefill_guard = PrefillLoadGuard::Admission {
            _reservation: Arc::new(admitted.reservation),
        };
        let mut prefill_children = vec![prefill_guard.replicate(), prefill_guard.replicate()];
        prefill_children.push(prefill_guard);
        let mut decode_children =
            match LoadGuards::scaled(&pd_selection(&prefill, &decode), Some(routing_key), 3) {
                LoadGuards::Batch { _guards: guards } => guards,
                _ => panic!("count > 1 should create batch guards"),
            };

        assert_eq!(prefill.load(), 1);
        assert_eq!(decode.load(), 3);
        assert_eq!(prefill.routing_key_load(), 1);
        assert_eq!(decode.routing_key_load(), 1);

        drop(prefill_children.pop());
        drop(decode_children.pop());
        assert_eq!(prefill.load(), 1);
        assert_eq!(decode.load(), 2);
        assert_eq!(prefill.routing_key_load(), 1);
        assert_eq!(decode.routing_key_load(), 1);

        drop(prefill_children);
        drop(decode_children);
        assert_eq!(prefill.load(), 0);
        assert_eq!(decode.load(), 0);
        assert_eq!(prefill.routing_key_load(), 0);
        assert_eq!(decode.routing_key_load(), 0);

        let prefill_guard = PrefillLoadGuard::Unbounded {
            _guard: WorkerLoadGuard::with_key(Arc::clone(&prefill), Some(routing_key)),
        };
        let mut prefill_children = vec![prefill_guard.replicate(), prefill_guard.replicate()];
        prefill_children.push(prefill_guard);
        let mut decode_children =
            match LoadGuards::scaled(&pd_selection(&prefill, &decode), Some(routing_key), 3) {
                LoadGuards::Batch { _guards: guards } => guards,
                _ => panic!("count > 1 should create batch guards"),
            };

        assert_eq!(prefill.load(), 3);
        assert_eq!(decode.load(), 3);
        assert_eq!(prefill.routing_key_load(), 1);
        assert_eq!(decode.routing_key_load(), 1);

        drop(prefill_children.pop());
        drop(decode_children.pop());
        assert_eq!(prefill.load(), 2);
        assert_eq!(decode.load(), 2);
        assert_eq!(prefill.routing_key_load(), 1);
        assert_eq!(decode.routing_key_load(), 1);

        drop(prefill_children);
        drop(decode_children);
        assert_eq!(prefill.load(), 0);
        assert_eq!(decode.load(), 0);
        assert_eq!(prefill.routing_key_load(), 0);
        assert_eq!(decode.routing_key_load(), 0);
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

    /// The context window bounds each engine request, and a batched
    /// Completion dispatches one per prompt: the length check must see the
    /// longest item, not the batch total (which would reject a batch of
    /// short prompts) nor the first item (which would miss a long later one).
    #[test]
    fn max_input_token_count_is_the_longest_batched_completion_item() {
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

        assert_eq!(batch.max_input_token_count(), 5);
        assert_ne!(batch.max_input_token_count(), batch.token_ids().len());
        assert_ne!(
            batch.max_input_token_count(),
            batch.total_input_token_count()
        );

        let scalar = completion_prep(&["hello"], None);
        assert_eq!(scalar.max_input_token_count(), scalar.token_ids().len());

        let empty = PreparationOutput::Completion {
            items: vec![],
            joined_routing_text: None,
        };
        assert_eq!(empty.max_input_token_count(), 0);
    }

    /// An embedding batch goes to the engine one input per call, so the
    /// window applies to its longest input: neither the first alone (which
    /// would let a long later input through) nor their sum.
    #[test]
    fn max_input_token_count_is_the_longest_embedding_input() {
        let batch = PreparationOutput::Embedding {
            inputs: vec![
                EmbeddingInput {
                    text: Some("short".to_string()),
                    token_ids: vec![1],
                },
                EmbeddingInput {
                    text: None,
                    token_ids: vec![2, 3, 4, 5, 6],
                },
            ],
        };

        assert_eq!(batch.max_input_token_count(), 5);
        assert_eq!(batch.token_ids().len(), 1);
        assert_eq!(batch.total_input_token_count(), 6);
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

    /// The ids a retry routes by, per plan shape: what the first selection
    /// used (a batch's first prompt), and none for a prompt that went as text.
    #[test]
    fn a_retry_routes_by_the_plans_prompt_ids_whatever_the_plans_shape() {
        use smg_grpc_client::vllm_proto as vllm;
        let tokenized = |ids: &[u32]| {
            ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
                request_id: "r".to_string(),
                input: Some(vllm::generate_request::Input::Tokenized(
                    vllm::TokenizedInput {
                        original_text: String::new(),
                        input_ids: ids.to_vec(),
                    },
                )),
                ..Default::default()
            }))
        };
        let as_text = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
            request_id: "r".to_string(),
            input: Some(vllm::generate_request::Input::Text("as text".to_string())),
            ..Default::default()
        }));
        let embed = ProtoEmbedRequest::Vllm(Box::new(vllm::EmbedRequest {
            tokenized: Some(vllm::TokenizedInput {
                original_text: String::new(),
                input_ids: vec![4, 5],
            }),
            ..Default::default()
        }));
        let batch = |requests: Vec<ProtoGenerateRequest>| ExecutionPlan::Batch {
            kind: ExecutionPlanKind::Single,
            shared_request_id: "b".to_string(),
            requests,
        };
        let cases: [(&str, ExecutionPlan, &[u32]); 7] = [
            (
                "a single request",
                ExecutionPlan::generate(ExecutionPlanKind::Single, tokenized(&[1, 2, 3])),
                &[1, 2, 3],
            ),
            (
                "a single request that went as text",
                ExecutionPlan::generate(ExecutionPlanKind::Single, as_text),
                &[],
            ),
            (
                "prefill-decode",
                ExecutionPlan::generate(ExecutionPlanKind::PrefillDecode, tokenized(&[6])),
                &[6],
            ),
            (
                "encode-prefill-decode",
                ExecutionPlan::generate(ExecutionPlanKind::EncodePrefillDecode, tokenized(&[7])),
                &[7],
            ),
            ("an embed request", ExecutionPlan::embed(embed), &[4, 5]),
            (
                "a batch: its first prompt",
                batch(vec![tokenized(&[8, 9]), tokenized(&[10])]),
                &[8, 9],
            ),
            ("an empty batch", batch(Vec::new()), &[]),
        ];
        for (shape, plan, expected) in cases {
            assert_eq!(plan.routing_token_ids(), expected, "{shape}");
        }
    }

    #[test]
    fn load_guards_report_completion_for_each_side_they_hold() {
        use smg_worker::identity::basic::{RequestCompletionSink, WorkerType};

        use crate::routers::grpc::backend_client::BasicWorkerBuilder;
        #[derive(Debug, Default)]
        struct CompletionSpy(std::sync::Mutex<Vec<String>>);
        impl RequestCompletionSink for CompletionSpy {
            fn request_completed(&self, worker: &dyn Worker) {
                self.0.lock().unwrap().push(worker.url().to_string());
            }
        }

        let spy = Arc::new(CompletionSpy::default());
        let sink: Arc<dyn RequestCompletionSink> = spy.clone();
        let single: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://single")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://prefill")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        for worker in [&single, &prefill, &decode] {
            worker.set_completion_sink(Some(Arc::clone(&sink)));
        }

        // The regular gRPC path: one guard, one completion when the stream ends.
        drop(LoadGuards::new(
            &WorkerSelection::Single {
                worker: Arc::clone(&single).into(),
            },
            None,
        ));
        assert_eq!(spy.0.lock().unwrap().as_slice(), ["grpc://single"]);

        // The disaggregated path: the decode side completes with the guards,
        // the prefill side with its own guard when the prefill phase ends.
        let guards = LoadGuards::new(&pd_selection(&prefill, &decode), None);
        let prefill_guard = PrefillLoadGuard::Unbounded {
            _guard: WorkerLoadGuard::with_key(Arc::clone(&prefill), None),
        };
        drop(prefill_guard);
        assert_eq!(
            spy.0.lock().unwrap().last().map(String::as_str),
            Some("grpc://prefill")
        );
        drop(guards);
        assert_eq!(
            spy.0.lock().unwrap().last().map(String::as_str),
            Some("grpc://decode")
        );

        // A batched fan-out reports once per sub-request.
        let before = spy.0.lock().unwrap().len();
        drop(LoadGuards::scaled(
            &WorkerSelection::Single {
                worker: Arc::clone(&single).into(),
            },
            None,
            3,
        ));
        assert_eq!(spy.0.lock().unwrap().len(), before + 3);
    }

    /// The client-shape mismatch answers an executor takes from the
    /// selection: one status and code per shape, whichever executor asks;
    /// a selection of the asked shape hands its clients out.
    #[tokio::test]
    async fn a_selection_of_the_other_shape_answers_the_mismatch() {
        let (url, server) = mock_engine("test-model").await;
        let client = VllmEngineClient::connect(&url)
            .await
            .expect("connect mock engine");
        let client = BackendClient::Grpc(GrpcClient::Vllm(client));
        let mut single = ClientSelection::Single {
            client: client.clone(),
        };
        assert!(single.single_or_mismatch("an_executor").is_ok());
        assert_eq!(
            failure_shape(failure_of(single.disaggregated_or_mismatch("an_executor"))).await,
            (
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "expected_disaggregated_clients_got_single".to_owned()
            )
        );
        let mut disaggregated = ClientSelection::Disaggregated {
            prefill: client.clone(),
            decode: client,
        };
        assert!(disaggregated
            .disaggregated_or_mismatch("an_executor")
            .is_ok());
        assert_eq!(
            failure_shape(failure_of(disaggregated.single_or_mismatch("an_executor"))).await,
            (
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "expected_single_client_got_disaggregated".to_owned()
            )
        );
        server.abort();
    }

    /// The Harmony iteration result rides behind a pointer, and the dispatch
    /// context, boxed once per request, fits the allocator's 2,560-byte class
    /// on 64-bit targets: inline, the result's 208 bytes had taken the box
    /// from 2,520 to 2,584 bytes, one class up, when the usage object it and
    /// the final response carry grew by 32 bytes each. A field that grows
    /// the context past the class needs its own read.
    #[test]
    fn the_iteration_result_rides_behind_a_pointer_and_the_dispatch_context_fits_its_class() {
        let state = ResponseState::default();
        assert_eq!(
            size_of_val(&state.responses_iteration_result),
            size_of::<usize>()
        );
        assert!(
            size_of::<DispatchContext>() <= 2560,
            "the dispatch context is {} bytes",
            size_of::<DispatchContext>()
        );
    }
}
