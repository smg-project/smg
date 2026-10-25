//! The `tokenspeed.grpc.scheduler.TokenSpeedScheduler` servicer in Rust: the
//! contract the Python `smg_grpc_servicer.tokenspeed` package serves, over
//! the msgpack ZMQ wire a headless TokenSpeed scheduler dials (what
//! `ts serve --headless` speaks to the Router's direct lane). The Router
//! cannot tell the two implementations apart; Python keeps the lifecycle,
//! launching the headless scheduler and driving this server through the
//! binding.

mod engine;
mod generate;
mod info;
mod service;
#[cfg(test)]
mod tests;

use std::{sync::Arc, time::Duration};

use engine_zmq_adapter::{
    GridTensors, MropeLayout, StructuredOutputBackend, StructuredOutputsBackendConfig,
    StructuredOutputsSupport, WireRevision,
};
use llm_tokenizer::traits::Tokenizer;
use smg_grpc_client::tokenspeed_proto::token_speed_scheduler_server::TokenSpeedSchedulerServer;
use tonic::service::Routes;

use crate::{
    engine_link::{EngineLink, TokenizerSlot},
    kv_events::KvEventRelay,
    metrics::{Identity, ServicerMetrics},
    servicer::{self, Shared, WireState},
    ServicerConfig, ServicerError,
};

/// The gRPC service name, as the health service reports it.
pub const SERVICE_NAME: &str = "tokenspeed.grpc.scheduler.TokenSpeedScheduler";
/// The `wire` label of this servicer's metrics.
const WIRE: &str = "tokenspeed";
/// What `HealthCheck`'s message says of a serving engine on this wire.
const HEALTHY: &str = "healthy";

/// What `GetModelInfo` and `GetServerInfo` report: the facts the Python
/// servicer reads off TokenSpeed's `ModelConfig` and `ServerArgs`, which the
/// launcher computes from the same config before the engine starts.
#[derive(Debug, Clone, Default)]
pub struct TokenSpeedModelInfo {
    pub model_path: String,
    pub served_model_name: String,
    pub tokenizer_path: String,
    pub model_type: String,
    pub architectures: Vec<String>,
    pub max_context_length: i32,
    pub max_req_input_len: i32,
    pub vocab_size: i32,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: i32,
    pub bos_token_id: i32,
    pub weight_version: String,
    pub default_sampling_params_json: String,
    pub supports_vision: bool,
    pub supports_multimodal: bool,
    /// `smg.grpc.common.Modality` values.
    pub supported_modalities: Vec<i32>,
    /// Whether the language model's rotary positions are MRoPE (the text rope
    /// config carries an `mrope_section`), which the launcher reads off the
    /// checkpoint: a multimodal item's grid tensors then feed a position
    /// table the adapter computes for the families whose rule it has (from
    /// `architectures` / `model_type` and the three layout facts below) and
    /// refuses for the rest; `Some(false)` lets a patch-layout grid through;
    /// `None` (a launcher that does not say) refuses as before.
    pub mrope_positions: Option<bool>,
    /// The vision tower's spatial merge size (`vision_config.spatial_merge_size`),
    /// one of the position rule's inputs; `None` when the checkpoint has none.
    pub spatial_merge_size: Option<u32>,
    /// The checkpoint's image placeholder token id (`image_token_id`), which
    /// the position rule checks each image span against.
    pub image_token_id: Option<u32>,
    /// The checkpoint's video placeholder token id (`video_token_id`).
    pub video_token_id: Option<u32>,
    pub model_dtype: String,
    pub multimodal_encoder_dtype: String,
    /// TokenSpeed's `ServerArgs` as a JSON object, reported as
    /// `GetServerInfo.server_args`: the Router reads its labels (`dp_size`,
    /// `pairing_protocol`, `max_running_requests`, ...) from it.
    pub server_args_json: String,
    /// Launcher-side `scheduler_info` entries as a JSON object (such as
    /// `shm_namespace_id`); the handshake supplies the capacity figures.
    pub scheduler_info_json: String,
    pub tokenspeed_version: String,
    /// The engine's declared constrained-decoding policy (`always`, `requested`,
    /// or "" when it declares none), reported as
    /// `GetServerInfo.constrained_decoding` for the Router's `constrained_decoding`
    /// label.
    pub constrained_decoding: String,
    /// The scheduler's admission window (`max_num_seqs`), reported as
    /// `max_running_requests`; the handshake's figure is the fallback.
    pub max_running_requests: i32,
    /// The attention data-parallel size the launcher configured; the
    /// handshake's figure wins once the engines are up.
    pub data_parallel_size: i32,
    /// TokenSpeed's ZMQ KV-event publisher endpoint and topic; an empty
    /// endpoint means `SubscribeKvEvents` is UNIMPLEMENTED.
    pub kv_events_endpoint: String,
    /// The same config's `replay_endpoint` (the publisher's replay ROUTER),
    /// or empty when it runs none: what the relay asks for gaps in flight
    /// and for the batches published before its subscription joined.
    pub kv_events_replay_endpoint: String,
    pub kv_events_topic: String,
    /// Which revision of the msgpack wire the installed engine speaks, as
    /// the launcher read it off the engine before any scheduler started
    /// (the one capability read of this wire): a scheduler of the engine's
    /// Rust control plane is spoken to in the engine's published schema, a
    /// scheduler of its Python loop in the framing this servicer has always
    /// spoken. The default is the latter: a launcher that says nothing
    /// changes nothing.
    pub wire_revision: WireRevision,
}

impl TokenSpeedModelInfo {
    /// What a multimodal item's grid tensors are for this model
    /// ([`GridTensors`]): the position rule's layout facts travel together,
    /// so a launcher that shipped only some of them ships none.
    pub(super) fn grid_tensors(&self) -> GridTensors {
        let layout = match (
            self.spatial_merge_size,
            self.image_token_id,
            self.video_token_id,
        ) {
            (Some(spatial_merge_size), Some(image_token_id), Some(video_token_id)) => {
                Some(MropeLayout {
                    spatial_merge_size,
                    image_token_id,
                    video_token_id,
                })
            }
            _ => None,
        };
        GridTensors::for_model(
            &self.architectures,
            &self.model_type,
            self.mrope_positions,
            layout,
        )
    }
}

/// What the scheduler enforces of a structured request, read off the
/// launcher's `server_args`: the `grammar_backend` the launcher resolved
/// (TokenSpeed's `--grammar-backend`, `xgrammar` or `none`: `none` unless
/// set, `xgrammar` for the checkpoints the engine defaults so). `none`
/// enforces nothing: the scheduler refuses such a request at admission with
/// a 400 that reaches the caller as INTERNAL with `BARE_ENGINE_ABORT` from
/// an engine that declares no status columns, or as INVALID_ARGUMENT with
/// the engine's reason from one that does; refusing before the scheduler
/// gives the caller's error on either. `xgrammar` compiles all four kinds, a JSON
/// schema with the features it lacks dropped. A launcher that wrote no
/// backend could not tell (a checkpoint default not resolved yet, or an
/// older launcher), and a name this servicer does not know is not one the
/// engine starts with: nothing is judged ahead of the engine then. The
/// adapter refuses a constraint outside this before the scheduler, as on
/// every wire.
pub(super) fn structured_outputs_support(model: &TokenSpeedModelInfo) -> StructuredOutputsSupport {
    use StructuredOutputsBackendConfig::{Other, Pinned};

    let args: serde_json::Value = serde_json::from_str(&model.server_args_json).unwrap_or_default();
    let backend = args
        .get("grammar_backend")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    match backend {
        "none" => StructuredOutputsSupport::DISABLED,
        "xgrammar" => Pinned(StructuredOutputBackend::Xgrammar).into(),
        _ => Other.into(),
    }
}

/// How to bind, where the engines dial in, and what to advertise.
#[derive(Debug, Clone)]
pub struct TokenSpeedServicerConfig {
    /// The listener, the scheduler's endpoints, the tokenizer directory and
    /// the start-up bounds. `handshake_address` is what the headless
    /// scheduler dials (its `--data-parallel-address`/
    /// `--data-parallel-rpc-port`), `engine_count` the attention
    /// data-parallel size; without `tokenizer_dir`, requests carrying string
    /// stops are refused.
    pub servicer: ServicerConfig,
    pub model: TokenSpeedModelInfo,
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
    pub(super) model: TokenSpeedModelInfo,
    /// What every servicer's state shares: the engine link, the admitted
    /// requests, the KV-event relay, the metrics and the serving switch.
    pub(super) shared: Shared,
    /// Loaded once alongside the engine connect; `Some(None)` records a load
    /// that failed (string stops are then refused).
    pub(super) tokenizer: TokenizerSlot,
}

impl WireState for State {
    fn shared(&self) -> &Shared {
        &self.shared
    }
}

impl State {
    pub(super) fn tokenizer(&self) -> Option<&Arc<dyn Tokenizer>> {
        self.tokenizer.get().and_then(Option::as_ref)
    }
}

/// A running Rust TokenSpeed servicer. `start` returns once the gRPC
/// listener is bound; the engines connect in the background and gate health.
pub struct TokenSpeedServicerServer {
    server: servicer::Server,
    state: Arc<State>,
}

impl TokenSpeedServicerServer {
    /// Validate, bind, and serve on a dedicated runtime thread.
    pub fn start(config: TokenSpeedServicerConfig) -> Result<Self, ServicerError> {
        Self::start_inner(config, None)
    }

    /// [`Self::start`] with the tokenizer supplied instead of loaded from
    /// `tokenizer_dir`, for a lifecycle owner that already loaded it.
    pub fn start_with_tokenizer(
        config: TokenSpeedServicerConfig,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Result<Self, ServicerError> {
        Self::start_inner(config, Some(tokenizer))
    }

    fn start_inner(
        config: TokenSpeedServicerConfig,
        tokenizer: Option<Arc<dyn Tokenizer>>,
    ) -> Result<Self, ServicerError> {
        config.servicer.validate()?;
        if config.model.model_path.trim().is_empty() {
            return Err(ServicerError::InvalidConfig(
                "model_path must not be empty".to_string(),
            ));
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
                matches_string_stops: true,
                // Media reach this wire as the router's preprocessed tensors: the
                // router-side families are their measurement.
                processes_media: false,
            },
            config.servicer.metrics_bind_address.is_some(),
        )?;
        let state = Arc::new(State {
            model: config.model,
            shared: Shared::new(kv_relay, metrics, config.servicer.tokenizer_dir.clone()),
            tokenizer: TokenizerSlot::new(),
        });
        if let Some(tokenizer) = tokenizer {
            let _ = state.tokenizer.set(Some(tokenizer));
        }
        let routes = Routes::new(
            TokenSpeedSchedulerServer::new(service::TokenSpeedService {
                state: Arc::clone(&state),
            })
            .max_decoding_message_size(usize::MAX)
            .max_encoding_message_size(usize::MAX),
        );
        let connect_state = Arc::clone(&state);
        let server = servicer::Server::start(
            "smg-tokenspeed-servicer",
            config.servicer,
            SERVICE_NAME,
            "TokenSpeed",
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
