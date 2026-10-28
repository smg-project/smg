//! The `sglang.grpc.scheduler.SglangScheduler` servicer in Rust: the contract
//! the Python `smg_grpc_servicer.sglang` package serves, over the msgpack ZMQ
//! wire a headless SGLang scheduler dials (what the SMG plugin inside the
//! scheduler speaks to the Router's direct lane). The Router cannot tell the
//! two implementations apart; Python keeps the lifecycle, launching the
//! headless scheduler and driving this server through the binding.
//!
//! Prefill/decode disaggregation works as on the Python servicer: a request's
//! bootstrap fields (host, port, room) ride the scheduler's own request struct
//! over the wire, the launcher starts the prefill side's bootstrap server
//! before the ranks, and `GetServerInfo` carries the server args the Router
//! pairs on. A request the Router pins to a data-parallel rank goes to that
//! rank's scheduler (the connector sends it on that rank's socket, the
//! dispatch the engine's data-parallel controller does behind its own
//! frontend), with the rank in the request's `routed_dp_rank` slot. On the
//! prefill side of a disaggregated pair a request runs on the rank of its
//! room's residue instead (`bootstrap_room % dp_size`: where SGLang's prefill
//! side requires it and the decode side looks for the KV), the pin yielding
//! to it with a warning; the decode side keeps the pin.
//!
//! The language side of encode/prefill disaggregation (`--language-only`)
//! works as behind the engine's own frontend: a request's image references
//! go to the encoders through the dispatcher the launcher supplies (SGLang's
//! own receiver, in the launcher process) before the request reaches the
//! scheduler, the scheduler's request carries what that dispatch set (its
//! wait flag, the per-encoder assignment, the encoder list) and the prompt
//! text the scheduler rebuilds from, and a dispatch that fails afterwards
//! reaches the scheduler as its own failure notice on the request's rank
//! ([`engine_zmq_adapter::sglang_encode`]).

mod embed;
mod engine;
mod generate;
mod info;
mod media_identity;
mod service;
#[cfg(test)]
mod tests;

use std::{sync::Arc, time::Duration};

use engine_zmq_adapter::{
    ConstraintKinds, SglangEncodeDispatcher, StructuredOutputBackend,
    StructuredOutputsBackendConfig, StructuredOutputsSupport,
};
use smg_grpc_client::sglang_proto::sglang_scheduler_server::SglangSchedulerServer;
use tonic::service::Routes;

use crate::{
    engine_link::EngineLink,
    kv_events::KvEventRelay,
    metrics::{Identity, ServicerMetrics},
    proto_json,
    servicer::{self, Shared, WireState},
    ServicerConfig, ServicerError,
};

/// The gRPC service name, as the health service reports it.
pub const SERVICE_NAME: &str = "sglang.grpc.scheduler.SglangScheduler";
/// The `wire` label of this servicer's metrics.
const WIRE: &str = "sglang";
/// What `HealthCheck`'s message says of a serving engine on this wire.
const HEALTHY: &str = "healthy";

/// What `GetModelInfo` and `GetServerInfo` report: the facts the Python
/// servicer reads off SGLang's `ModelConfig` and `ServerArgs`, which the
/// launcher computes from the same config before the scheduler starts. The
/// handshake supplies the capacity figures once the scheduler is up.
#[derive(Debug, Clone, Default)]
pub struct SglangModelInfo {
    pub model_path: String,
    pub tokenizer_path: String,
    pub served_model_name: String,
    pub is_generation: bool,
    pub model_type: String,
    pub architectures: Vec<String>,
    /// The launcher's context length (`ModelConfig.context_len`); the
    /// handshake's `max_model_len` is the fallback when the launcher sent 0.
    pub max_context_length: i32,
    pub max_req_input_len: i32,
    pub vocab_size: i32,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: i32,
    pub bos_token_id: i32,
    pub weight_version: String,
    /// `--preferred-sampling-params` as JSON text, or empty.
    pub preferred_sampling_params: String,
    /// The model's generation-config sampling defaults merged with the
    /// preferred ones, as JSON text, or empty.
    pub default_sampling_params_json: String,
    /// Whether the model takes multimodal input, as the launcher reports it:
    /// SGLang's own `ModelConfig.is_multimodal` for the checkpoint and the
    /// launch, the decision the scheduler itself runs under. Advertised by
    /// `GetModelInfo`; the gateway's media pipeline preprocesses for such a
    /// worker and the wire carries the tensors to the scheduler.
    pub supports_vision: bool,
    /// Classification heads: `id2label` as JSON text and the label count.
    pub id2label_json: String,
    pub num_labels: i32,
    /// SGLang's `ServerArgs` as a JSON object, reported as
    /// `GetServerInfo.server_args`: the Router reads its labels (`tp_size`,
    /// `dp_size`, `context_length`, `pairing_protocol`, ...) from it.
    pub server_args_json: String,
    /// Launcher-side `scheduler_info` entries as a JSON object; the handshake
    /// supplies the capacity figures.
    pub scheduler_info_json: String,
    pub sglang_version: String,
    /// `--max-running-requests`, reported as `max_running_requests`; the
    /// handshake's figure is the fallback.
    pub max_running_requests: i32,
    /// The data-parallel size the launcher configured; the handshake's figure
    /// wins once the engines are up.
    pub data_parallel_size: i32,
    /// SGLang's `--kv-events-config` ZMQ publisher endpoint as configured
    /// (`tcp://*:5557` style, rank 0's port), or empty when events are off;
    /// `SubscribeKvEvents` relays it through [`crate::kv_wire`]'s rules.
    pub kv_events_endpoint: String,
    /// The same config's `replay_endpoint` (SGLang's replay ROUTER), or
    /// empty when it runs none: what the relay asks for gaps in flight and
    /// for the batches published before its subscription joined.
    pub kv_events_replay_endpoint: String,
    /// The publisher's topic prefix (empty by default).
    pub kv_events_topic: String,
}

/// What the scheduler enforces of a structured request, read off the
/// launcher's `server_args`: the `grammar_backend` the launcher resolved
/// (SGLang's `--grammar-backend`: `xgrammar` unless set, `outlines`,
/// `llguidance` or `none`) and whether the scheduler keeps its tokenizer,
/// without which (`--skip-tokenizer-init`) its grammar manager builds no
/// backend at all. `none` enforces nothing: the scheduler admits such a
/// request and aborts it with a 400. `outlines` compiles JSON schemas and
/// regular expressions and fails the other kinds at compile; `xgrammar` and
/// `llguidance` compile all four, xgrammar with the schema features it lacks
/// dropped. A name this servicer does not know is a plugin backend of the
/// scheduler's: nothing is judged ahead of it. A launcher that wrote no
/// backend (an older one) leaves the engine's default, xgrammar. The adapter
/// refuses a constraint outside this before the scheduler, as on every wire.
pub(super) fn structured_outputs_support(model: &SglangModelInfo) -> StructuredOutputsSupport {
    use StructuredOutputsBackendConfig::{Other, Pinned};

    let args: serde_json::Value = serde_json::from_str(&model.server_args_json).unwrap_or_default();
    if args
        .get("skip_tokenizer_init")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        return StructuredOutputsSupport::DISABLED;
    }
    let backend = args
        .get("grammar_backend")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    match backend {
        "" => StructuredOutputsSupport::default(),
        "xgrammar" => Pinned(StructuredOutputBackend::Xgrammar).into(),
        "llguidance" => Pinned(StructuredOutputBackend::Guidance).into(),
        "outlines" => StructuredOutputsSupport::new(
            Pinned(StructuredOutputBackend::Outlines),
            ConstraintKinds {
                json_schema: true,
                regex: true,
                ebnf: false,
                structural_tag: false,
            },
        ),
        "none" => StructuredOutputsSupport::DISABLED,
        _ => Other.into(),
    }
}

/// Whether the scheduler was started with `--allow-auto-truncate`: it then
/// cuts a prompt longer than its bound down to it instead of aborting the
/// request, and the connection's prompt admission leaves the upper bound to
/// it (an empty prompt is still refused before the scheduler).
pub(super) fn truncates_long_prompts(model: &SglangModelInfo) -> bool {
    let args: serde_json::Value = serde_json::from_str(&model.server_args_json).unwrap_or_default();
    args.get("allow_auto_truncate")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
}

/// Whether the scheduler is the prefill side of a disaggregated pair
/// (`--disaggregation-mode prefill`), read off the launcher's `server_args`:
/// SGLang's prefill side follows the room by default (`load_balance_method`
/// `auto` resolves to `follow_bootstrap_room` there), its KV sender fails a
/// request that runs on another rank than `bootstrap_room % dp_size`, and the
/// decode side looks for the KV on that rank, so the connection places a
/// disaggregated request by its room's residue, the Router's pin yielding to
/// it. The decode side's own placement is free (its `auto` is
/// `round_robin`): it keeps the pin, as an aggregated engine does.
pub(super) fn places_by_rendezvous_room(model: &SglangModelInfo) -> bool {
    let args: serde_json::Value = serde_json::from_str(&model.server_args_json).unwrap_or_default();
    args.get("disaggregation_mode")
        .and_then(serde_json::Value::as_str)
        == Some("prefill")
}

/// SGLang's `CaptureHiddenMode`: how much of the hidden states the scheduler
/// was launched to return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HiddenStatesCap {
    Null,
    Last,
    Full,
}

/// The scheduler's launch-time cap on hidden states, as its own frontend
/// keeps it: `--return-hidden-states-mode {last,full}` (the earlier
/// `--enable-return-hidden-states` is `full`), read off the launcher's
/// `server_args`. `None` when they name neither: an engine release without
/// the cap, which then decides itself.
pub(super) fn hidden_states_cap(model: &SglangModelInfo) -> Option<HiddenStatesCap> {
    let args: serde_json::Value = serde_json::from_str(&model.server_args_json).unwrap_or_default();
    let mode = args
        .get("return_hidden_states_mode")
        .and_then(serde_json::Value::as_str);
    let legacy = args
        .get("enable_return_hidden_states")
        .and_then(serde_json::Value::as_bool);
    // The engine's own order (`get_server_return_hidden_states_mode`): a
    // `last` mode wins over the legacy flag, which the engine sets whenever a
    // mode is given.
    match (mode, legacy) {
        (Some("last"), _) => Some(HiddenStatesCap::Last),
        (Some("full"), _) | (_, Some(true)) => Some(HiddenStatesCap::Full),
        (None, None) => None,
        _ => Some(HiddenStatesCap::Null),
    }
}

/// Native worker-side media preprocessing. Implementations retain the
/// SGLang request and replace only its unexpanded prompt and media inputs.
pub trait SglangMediaProcessor: Send + Sync {
    /// Identical only for processors with compatible pixel-less Decode metadata.
    fn identity_key(&self) -> Option<&str> {
        None
    }

    /// Resolve references before dispatch, preserving the request's generation options.
    fn process(
        &self,
        request: smg_grpc_client::sglang_proto::GenerateRequest,
    ) -> crate::BoxFuture<Result<smg_grpc_client::sglang_proto::GenerateRequest, tonic::Status>>;
}

/// How to bind, where the engines dial in, and what to advertise.
#[derive(Clone)]
pub struct SglangServicerConfig {
    /// The listener, the scheduler's endpoints, the tokenizer directory and
    /// the start-up bounds. `handshake_address` is what the headless
    /// scheduler dials (the launcher's `--zmq-handshake-address`),
    /// `engine_count` the data-parallel size; `tokenizer_dir` is bundled by
    /// `GetTokenizer` and the native media launcher uses it for expansion.
    /// The scheduler keeps its own tokenizer for stop strings.
    pub servicer: ServicerConfig,
    pub model: SglangModelInfo,
    /// The encode dispatch of a language-side worker under encode/prefill
    /// disaggregation (`--language-only`): a request's image references go
    /// to the encoders through it before the request reaches the scheduler,
    /// as the engine's own frontend sends them ([`SglangEncodeDispatcher`]).
    /// `None` on every other worker: a request of references is refused.
    pub encode_dispatcher: Option<Arc<dyn SglangEncodeDispatcher>>,
    /// Optional native image/video processor; mutually exclusive with encode dispatch.
    pub media_processor: Option<Arc<dyn SglangMediaProcessor>>,
}

impl std::fmt::Debug for SglangServicerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SglangServicerConfig")
            .field("servicer", &self.servicer)
            .field("model", &self.model)
            .field("encode_dispatcher", &self.encode_dispatcher.is_some())
            .finish()
    }
}

/// The servicer's `GetLoads` figures, read for the load record the relay
/// attaches to every batch it streams.
impl crate::kv_events::RankLoads for State {
    fn rank_load(&self, dp_rank: Option<i32>) -> Option<smg_grpc_client::common_proto::EngineLoad> {
        let response = info::loads(self, dp_rank).ok()?;
        // The engine reports no queued token-work on this wire yet: the
        // record's `waiting_uncached_tokens` stays unset.
        Some(smg_grpc_client::common_proto::EngineLoad::from(
            response.loads.first()?,
        ))
    }

    fn engine_link(&self) -> &EngineLink {
        &self.shared.engine
    }
}

pub(super) struct State {
    pub(super) model: SglangModelInfo,
    /// What every servicer's state shares: the engine link, the admitted
    /// requests, the KV-event relay, the metrics and the serving switch.
    pub(super) shared: Shared,
    /// The language-side encode dispatch, when this worker runs one.
    pub(super) encode: Option<Arc<dyn SglangEncodeDispatcher>>,
    pub(super) media: Option<Arc<dyn SglangMediaProcessor>>,
    pub(super) media_identity_enabled: bool,
    pub(super) media_identity_key: Option<String>,
    /// Resolved once at startup; missing or malformed roles cannot enable reuse.
    pub(super) media_disaggregation_mode: Option<String>,
    pub(super) mm_hasher: Option<engine_zmq_adapter::SglangCpuMmHasher>,
}

impl WireState for State {
    fn shared(&self) -> &Shared {
        &self.shared
    }
}

/// A running Rust SGLang servicer. `start` returns once the gRPC listener is
/// bound; the engines connect in the background and gate health.
pub struct SglangServicerServer {
    server: servicer::Server,
    state: Arc<State>,
}

impl SglangServicerServer {
    /// Validate, bind, and serve on a dedicated runtime thread.
    pub fn start(config: SglangServicerConfig) -> Result<Self, ServicerError> {
        // Only processors with an engine-tested identity contract auto-enable
        // CPU hashing/reuse. Unknown models retain scheduler-side processing.
        let supported = config
            .media_processor
            .as_ref()
            .and_then(|processor| processor.identity_key())
            .is_some();
        let enabled = media_reuse_enabled(
            std::env::var("SMG_SGLANG_PD_MEDIA_REUSE").ok().as_deref(),
            supported,
        )?;
        Self::start_with_media_options(config, mm_hash_config(supported)?, enabled)
    }

    /// Start with explicit hash/reuse settings after validating the worker configuration.
    fn start_with_media_options(
        config: SglangServicerConfig,
        mm_hasher: Option<engine_zmq_adapter::SglangCpuMmHasher>,
        media_identity_enabled: bool,
    ) -> Result<Self, ServicerError> {
        let invalid = |message: &str| ServicerError::InvalidConfig(message.to_string());
        config.servicer.validate()?;
        if config.encode_dispatcher.is_some() && config.media_processor.is_some() {
            return Err(invalid(
                "native media processing cannot be combined with encode disaggregation",
            ));
        }
        if config.model.model_path.trim().is_empty() {
            return Err(invalid("model_path must not be empty"));
        }
        // The Router reads its labels off these; a text that is not a JSON
        // object would silently become an empty Struct at GetServerInfo.
        for (name, json) in [
            ("server_args_json", &config.model.server_args_json),
            ("scheduler_info_json", &config.model.scheduler_info_json),
        ] {
            if !json.trim().is_empty() && !proto_json::is_json_object(json) {
                return Err(invalid(&format!("{name} must be a JSON object")));
            }
        }
        // One publisher per data-parallel rank: the launcher's width (the
        // handshake's figure, which may differ, is not known yet).
        let kv_relay = KvEventRelay::for_publisher(
            &config.model.kv_events_endpoint,
            Some(&config.model.kv_events_replay_endpoint),
            &config.model.kv_events_topic,
            u32::try_from(config.model.data_parallel_size).unwrap_or(1),
        );
        let metrics = ServicerMetrics::new(
            &Identity {
                model: config.model.served_model_name.clone(),
                wire: WIRE,
                ranks: config.servicer.engine_count,
                max_model_len: u32::try_from(config.model.max_context_length).unwrap_or(0),
                // The scheduler matches string stops itself.
                matches_string_stops: false,
                // Native SGLang stage timing is emitted by its processor.
                processes_media: false,
            },
            config.servicer.metrics_bind_address.is_some(),
        )?;
        let media_identity_key = config
            .media_processor
            .as_ref()
            .and_then(|processor| processor.identity_key())
            .map(|key| format!("v1:{}:{key}", config.model.sglang_version));
        let media_disaggregation_mode =
            serde_json::from_str::<serde_json::Value>(&config.model.server_args_json)
                .ok()
                .and_then(|args| {
                    args.get("disaggregation_mode")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned)
                });
        let state = Arc::new(State {
            media_identity_key,
            media_disaggregation_mode,
            model: config.model,
            shared: Shared::new(kv_relay, metrics, config.servicer.tokenizer_dir.clone()),
            encode: config.encode_dispatcher,
            media: config.media_processor,
            mm_hasher,
            media_identity_enabled,
        });
        let routes = Routes::new(
            SglangSchedulerServer::new(service::SglangService {
                state: Arc::clone(&state),
            })
            .max_decoding_message_size(usize::MAX)
            .max_encoding_message_size(usize::MAX),
        );
        let connect_state = Arc::clone(&state);
        let server = servicer::Server::start(
            "smg-sglang-servicer",
            config.servicer,
            SERVICE_NAME,
            "SGLang",
            &state,
            routes,
            move |inputs| engine::connect_engine(connect_state, inputs),
        )?;
        Ok(Self { server, state })
    }

    /// The bound `ip:port`.
    pub fn address(&self) -> String {
        self.server.address()
    }

    /// The metrics listener's bound `ip:port`; `None` when none was asked for.
    pub fn metrics_address(&self) -> Option<String> {
        self.server.metrics_address()
    }

    pub fn running(&self) -> bool {
        self.server.running()
    }

    /// Whether the engine handshake completed.
    pub fn engine_ready(&self) -> bool {
        self.state.shared.engine_ready()
    }

    /// Report a sign of life from the scheduler process while the handshake
    /// runs: the lifecycle owner that launched it calls this each time it
    /// polls the process and finds it alive. Each report resets the
    /// handshake's silence bound (`engine_startup_timeout`); the ceiling
    /// (`engine_startup_ceiling`) still holds. A no-op once the scheduler is
    /// connected.
    pub fn note_engine_alive(&self) {
        self.state.shared.note_engine_alive();
    }

    pub fn last_error(&self) -> Result<Option<String>, ServicerError> {
        self.server.last_error()
    }

    /// Announce SERVING (`true`) or drain (`false`): health flips at once,
    /// new admission closes while in-flight streams keep running until `stop`.
    pub fn set_serving(&self, serving: bool) {
        self.state.shared.set_serving(serving);
    }

    /// Hold (`false`) or release (`true`) readiness alone: health flips as
    /// with `set_serving`, admission stays open to the lifecycle owner's own
    /// requests (its planned warm-up before SERVING, `smg-warmup-<n>`) and to
    /// nothing else, so the warm-up's steps share the engine with no client
    /// request. `set_serving(false)` is the drain switch and closes admission
    /// as well.
    pub fn set_ready(&self, ready: bool) {
        self.state.shared.set_ready(ready);
    }

    /// Whether the KV-event relay's subscriptions have reached the scheduler's
    /// publishers, or there is none to wait for (events off, a relay told to
    /// start lazily, one that gave up). The lifecycle owner holds its planned
    /// warm-up until then, so the warm-up's batches start the relay's
    /// history and a gateway that subscribes later gets the scheduler's whole
    /// life.
    pub fn kv_relay_subscribed(&self) -> bool {
        self.state.shared.kv_relay_subscribed()
    }

    /// Admitted requests still preprocessing, streaming, or unwinding an abort.
    pub fn active_requests(&self) -> usize {
        self.state.shared.registry.active()
    }

    /// Signal shutdown and wait up to `timeout` for the server thread;
    /// in-flight streams are cancelled with it (which aborts their engine
    /// requests).
    pub fn stop(&self, timeout: Duration) -> Result<(), ServicerError> {
        self.server.stop(&self.state.shared, timeout)
    }
}

/// Resolve the reuse override without enabling unsupported processors by default.
fn media_reuse_enabled(value: Option<&str>, supported: bool) -> Result<bool, ServicerError> {
    match value {
        None => Ok(supported),
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err(ServicerError::InvalidConfig(
            "SMG_SGLANG_PD_MEDIA_REUSE must be 0 or 1".into(),
        )),
    }
}

/// The identity contract is the CPU SHA256 path in supported engines.
/// Never override configured GPU hashing or the engine's skip-hash semantics.
fn mm_hash_config(
    supported: bool,
) -> Result<Option<engine_zmq_adapter::SglangCpuMmHasher>, ServicerError> {
    let threads = std::env::var("SMG_SGLANG_MM_HASH_THREADS").ok();
    let gpu_buffer = std::env::var("SGLANG_MM_BUFFER_SIZE_MB").unwrap_or_else(|_| "0".into());
    let skip = std::env::var("SGLANG_MM_SKIP_COMPUTE_HASH").unwrap_or_else(|_| "0".into());
    let count = mm_hash_threads(threads.as_deref(), supported, &gpu_buffer, &skip)?;
    Ok(std::num::NonZeroUsize::new(count).map(engine_zmq_adapter::SglangCpuMmHasher::new))
}

/// Validate bounded hash concurrency while honoring engine GPU and skip-hash settings.
fn mm_hash_threads(
    threads: Option<&str>,
    supported: bool,
    gpu_buffer: &str,
    skip: &str,
) -> Result<usize, ServicerError> {
    let threads = threads.unwrap_or(if supported { "16" } else { "0" });
    let count = threads
        .parse::<usize>()
        .ok()
        .filter(|n| *n <= 32)
        .ok_or_else(|| {
            ServicerError::InvalidConfig(
                "SMG_SGLANG_MM_HASH_THREADS must be between 0 and 32".into(),
            )
        })?;
    // Unknown values conservatively retain the engine's own hashing path.
    if count == 0 || gpu_buffer.parse::<u64>() != Ok(0) || skip != "0" {
        return Ok(0);
    }
    Ok(count)
}
