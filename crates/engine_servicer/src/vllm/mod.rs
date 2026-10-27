//! The vLLM gRPC servicer in Rust: serves `vllm.grpc.engine.VllmEngine` (the
//! proto the Router already speaks to the Python servicer) and
//! `grpc.health.v1` over a same-host vLLM EngineCore reached through the
//! existing ZMQ engine client.
//!
//! What the Python frontend did that EngineCore cannot, this servicer does:
//! it supplies EOS ids, defaults `max_tokens` to the remaining context, fans
//! `n > 1` out into independent engine requests, and matches string stops on
//! the decoded output with the model's tokenizer. Everything else is the ZMQ
//! client's existing translation; the Router sees the vLLM proto either way.
//!
//! PD disaggregation works as on the Python servicer: connector KV-transfer
//! params pass through to the engine and back, and `GetServerInfo` carries
//! the connector, role, engine id and pairing facts the Router matches on.
//!
//! `Embed`, `FlushCache` (a ZMQ utility call), `GetTokenizer` (the tokenizer
//! directory zipped as the Python servicer zips it) and `SubscribeKvEvents`
//! (vLLM's ZMQ KV-event publisher relayed) are served too. Worker-side media
//! processing (`media_refs`) runs through a [`MediaProcessor`] the lifecycle
//! owner supplies: the binding bridges to the Python servicer's processors,
//! which run vLLM's own input processor, and the result reaches the engine as
//! vLLM encoded it.

mod admin;
mod embed;
mod engine;
mod generate;
mod info;
mod media;
mod service;
#[cfg(test)]
mod tests;

use std::{sync::Arc, time::Duration};

use engine::connect_engine;
use llm_tokenizer::traits::Tokenizer;
use media::MediaGate;
pub use media::{
    BoxFuture, MediaError, MediaFeatures, MediaProcessor, MediaRefItem, MediaRequest,
    ProcessedMedia, ALL_MODALITIES, PIXEL_MODALITIES,
};
use service::VllmEngineService;
use smg_grpc_client::vllm_proto::vllm_engine_server::VllmEngineServer;
use tonic::service::Routes;

use crate::{
    engine_link::{EngineLink, TokenizerSlot},
    kv_events::KvEventRelay,
    metrics::{Identity, ServicerMetrics},
    servicer::{self, Shared, WireState},
    ServicerConfig, ServicerError,
};

pub(crate) const SERVICE_NAME: &str = "vllm.grpc.engine.VllmEngine";
/// `GetServerInfo.server_type` for this implementation.
pub(crate) const SERVER_TYPE: &str = "vllm-grpc";
/// The `wire` label of this servicer's metrics.
const WIRE: &str = "vllm";
/// What `HealthCheck`'s message says of a serving engine on this wire.
const HEALTHY: &str = "Health";

/// What `GetModelInfo` reports. The Python servicer reads these off vLLM's
/// `ModelConfig`; the launcher computes them from the same config before it
/// starts the headless engine, so the Router sees identical metadata.
#[derive(Debug, Clone, Default)]
pub struct VllmModelInfo {
    pub model_path: String,
    pub served_model_name: String,
    pub tokenizer_path: String,
    pub is_generation: bool,
    pub max_context_length: u32,
    pub vocab_size: u32,
    pub supports_vision: bool,
    pub model_type: String,
    pub architectures: Vec<String>,
    /// EOS ids in config order (`config.json` first): the first is the
    /// primary id EngineCore stops on, the rest join the stop set.
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: i32,
    pub bos_token_id: i32,
    pub default_sampling_params_json: String,
    pub data_parallel_size: i32,
    /// `GetServerInfo.pairing_protocol` (`SMG_PAIRING_PROTOCOL`).
    pub pairing_protocol: String,
    /// PD disaggregation identity off `kv_transfer_config`: the connector
    /// (`NixlConnector`, ...), this engine's role and its engine id; empty
    /// when the engine runs without a KV connector.
    pub kv_connector: String,
    pub kv_role: String,
    pub kv_engine_id: String,
    /// Whether a Mooncake connector runs in the engine at all, as the
    /// launcher reads it off the raw `kv_transfer_config` (the top-level
    /// connector or any child a wrapper lists, whatever `kv_connector`
    /// projects): a PD decode side's KV-transfer rejection notice is
    /// withheld there (see `generate.rs`).
    pub runs_mooncake: bool,
    /// PD pairing facts the Router matches before a handoff: requested KV
    /// cache dtype and attention backend (empty when auto), and the model
    /// dtype / block size the handshake confirms (these two are fallbacks
    /// until it does).
    pub kv_cache_dtype: String,
    pub attention_backend: String,
    pub model_dtype: String,
    pub block_size: i32,
    /// `--structured-outputs-config.backend` ("auto" when unset): the
    /// grammar backend the engine's structured-output manager must use.
    pub structured_outputs_backend: String,
    /// vLLM's KV-event publisher (`--kv-events-config`, ZMQ publisher only):
    /// the PUB endpoint, the optional replay endpoint, and the topic. Empty
    /// endpoint means events are not enabled and `SubscribeKvEvents` is
    /// UNIMPLEMENTED.
    pub kv_events_endpoint: String,
    pub kv_events_replay_endpoint: String,
    pub kv_events_topic: String,
    /// This host's `/dev/shm` identity (`<boot_id>:<st_dev>`), advertised so
    /// the Router can verify a shared `/dev/shm` before using the SHM tensor
    /// transport under `auto`; empty when unknown.
    pub shm_namespace_id: String,
    /// The model's pooler config (`--pooler-config`, or the model's own):
    /// what vLLM's frontend fills into an `Embed` request's unset
    /// `use_activation` and `dimensions`.
    pub pooler_use_activation: Option<bool>,
    pub pooler_dimensions: Option<u32>,
    /// Whether the engine rescales and normalizes pixels on device (vLLM's
    /// `mm_device_do_normalize`, where the model supports it): advertised so
    /// the Router sends such an engine the pixels' own bytes.
    pub mm_device_do_normalize: bool,
    /// The engine's per-prompt media limits (vLLM's `--limit-mm-per-prompt`)
    /// as the `mm_item_limits` label the Router's media pipeline holds
    /// requests to (`image=8,video=2`); empty when the engine has none.
    pub mm_item_limits: String,
    /// The engine's encoder budget in feature tokens, as its own front end
    /// derives it: the most one media item may carry and still be scheduled,
    /// advertised as `mm_encoder_budget` so the Router's media pipeline
    /// refuses a larger item before the engine; 0 when unknown.
    pub mm_encoder_budget: i32,
    /// The scheduler's running window (`--max-num-seqs`) as the launcher read
    /// it: how many requests the engine runs at once, which `GetServerInfo`
    /// advertises as the bound the Router's PD admission gate claims decode
    /// rooms against. The handshake's figure is the fallback for a launcher
    /// that reported none.
    pub max_num_seqs: i32,
    /// The engine's reasoning parser (`--reasoning-parser`, "glm45", ...),
    /// advertised as the `engine_reasoning_parser` label; empty when it runs
    /// none.
    pub engine_reasoning_parser: String,
    /// Where the engine starts a request's grammar on a thinking prompt, as
    /// the launcher read it off the structured-outputs config and the Router
    /// shapes a forced tool call's grammar by (`constrained_decoding_start`
    /// label): `after_reasoning` when the engine waits for its reasoning
    /// parser to see the thought's end, `first_token` when the grammar runs
    /// from the first generated token; empty when the config does not say.
    pub constrained_decoding_start: String,
    /// Gateway policy declared by the launcher: requested, always, or unset.
    pub constrained_decoding: String,
    /// Whether the engine's scheduler statistics are on (vLLM's
    /// `--disable-log-stats` not given), as the launcher started it: such an
    /// engine's wire reports every scheduler step, and the pushed load record
    /// says so (`engine_reports_steps`) from the first record rather than
    /// from the first stats-only step batch, so an engine that freezes before
    /// its first batch is judged on the step-reporting path too. `None` when
    /// the launcher did not say (an older one): the wire is then classified
    /// by what it shows.
    pub log_stats: Option<bool>,
}

/// How to bind, where the engine dials in, and what to advertise.
#[derive(Clone)]
pub struct VllmServicerConfig {
    /// The listener, the engine's endpoints, the tokenizer directory and the
    /// start-up bounds. `handshake_address` is what the headless engine dials
    /// (its `--data-parallel-address`/`--data-parallel-rpc-port`),
    /// `engine_count` its engine-level data-parallel size; without
    /// `tokenizer_dir`, requests carrying string stops are refused.
    pub servicer: ServicerConfig,
    pub model: VllmModelInfo,
    /// Worker-side media processing for `media_refs`; `None` refuses them, as
    /// the Python servicer does with `--mm-processor off`.
    pub media_processor: Option<Arc<dyn MediaProcessor>>,
}

impl std::fmt::Debug for VllmServicerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VllmServicerConfig")
            .field("servicer", &self.servicer)
            .field("model", &self.model)
            .field(
                "media_processor",
                &self
                    .media_processor
                    .as_ref()
                    .map(|processor| processor.name()),
            )
            .finish()
    }
}

/// The servicer's `GetLoads` figures, read for the load record the relay
/// attaches to every batch it streams.
impl crate::kv_events::RankLoads for State {
    fn rank_load(&self, dp_rank: Option<i32>) -> Option<smg_grpc_client::common_proto::EngineLoad> {
        let response = info::loads(self).ok()?;
        let load = match dp_rank {
            // A rank the report has no entry for (it has not reported a
            // scheduler step while another rank has) gets no record rather
            // than another rank's figures, as under the Python servicer.
            Some(rank) => response.loads.iter().find(|load| load.dp_rank == rank)?,
            None => response
                .loads
                .iter()
                .find(|load| load.dp_rank == 0)
                .or_else(|| response.loads.first())?,
        };
        // The queued token-work is this servicer's own estimate for the
        // rank (see `info::loads`), so every rank's record reports it.
        let mut record = smg_grpc_client::common_proto::EngineLoad::from(load);
        record.waiting_uncached_tokens =
            Some(u32::try_from(load.num_waiting_uncached_tokens).unwrap_or(0));
        Some(record)
    }

    fn engine_link(&self) -> &EngineLink {
        &self.shared.engine
    }
}

pub(super) struct State {
    pub(super) model: VllmModelInfo,
    /// What every servicer's state shares: the engine link, the admitted
    /// requests, the KV-event relay, the metrics and the serving switch.
    pub(super) shared: Shared,
    /// Queued token-work, generation throughput and hit rate from the
    /// requests this servicer forwards, per rank, for `GetLoads`.
    pub(super) loads: crate::load_tracker::LoadTracker,
    /// Loaded once alongside the engine connect; `Some(None)` records a load
    /// that failed (string stops are then refused, EOS still comes from config).
    pub(super) tokenizer: TokenizerSlot,
    /// The media processor behind its in-flight cap; `None` refuses
    /// `media_refs`.
    pub(super) media: Option<MediaGate>,
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

    pub(super) fn active_requests(&self) -> u32 {
        self.shared.registry.len()
    }
}

/// A running Rust vLLM servicer. `start` returns once the gRPC listener is
/// bound; the engine connects in the background and gates health.
pub struct VllmServicerServer {
    server: servicer::Server,
    state: Arc<State>,
}

impl VllmServicerServer {
    /// Validate, bind, and serve on a dedicated runtime thread.
    pub fn start(config: VllmServicerConfig) -> Result<Self, ServicerError> {
        Self::start_inner(config, None)
    }

    /// [`Self::start`] with the tokenizer supplied instead of loaded from
    /// `tokenizer_dir`: for a lifecycle owner that already loaded it (a
    /// media processor that needs it before the engine connects).
    pub fn start_with_tokenizer(
        config: VllmServicerConfig,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Result<Self, ServicerError> {
        Self::start_inner(config, Some(tokenizer))
    }

    fn start_inner(
        config: VllmServicerConfig,
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
                max_model_len: config.model.max_context_length,
                matches_string_stops: true,
                processes_media: true,
            },
            config.servicer.metrics_bind_address.is_some(),
        )?;
        let state = Arc::new(State {
            model: config.model,
            shared: Shared::new(kv_relay, metrics, config.servicer.tokenizer_dir.clone()),
            loads: crate::load_tracker::LoadTracker::default(),
            tokenizer: TokenizerSlot::new(),
            media: config.media_processor.map(MediaGate::new),
        });
        if let Some(tokenizer) = tokenizer {
            let _ = state.tokenizer.set(Some(tokenizer));
        }
        let routes = Routes::new(
            VllmEngineServer::new(VllmEngineService {
                state: Arc::clone(&state),
            })
            .max_decoding_message_size(usize::MAX)
            .max_encoding_message_size(usize::MAX),
        );
        let connect_state = Arc::clone(&state);
        let stats_state = Arc::clone(&state);
        let server = servicer::Server::start(
            "smg-vllm-servicer",
            config.servicer,
            SERVICE_NAME,
            "vLLM",
            &state,
            routes,
            move |inputs| {
                // The stats line runs beside the connect, from the start.
                #[expect(
                    clippy::disallowed_methods,
                    reason = "the stats log is fire-and-forget; the runtime drop ends it"
                )]
                let _stats = tokio::spawn(info::log_engine_stats(stats_state));
                connect_engine(connect_state, inputs)
            },
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

    /// Report a sign of life from the engine process while the handshake
    /// runs: the lifecycle owner that launched it calls this each time it
    /// polls the process and finds it alive. Each report resets the
    /// handshake's silence bound (`engine_startup_timeout`); the ceiling
    /// (`engine_startup_ceiling`) still holds. A no-op once the engine is
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

    /// Whether the KV-event relay's subscriptions have reached the engine's
    /// publishers, or there is none to wait for (events off, a relay told to
    /// start lazily, one that gave up). The lifecycle owner holds its planned
    /// warm-up until then, so the warm-up's batches start the relay's
    /// history and a gateway that subscribes later gets the engine's whole
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
