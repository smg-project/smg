//! The `trtllm.TrtllmService` servicer in Rust: the contract TensorRT-LLM's
//! own Python servicer (`tensorrt_llm.grpc.smg`) serves in-process over its
//! LLM API, served here over the msgpack ZMQ wire the shim beside that API
//! dials (the shim performs the handshake, builds the engine's own sampling
//! parameters from the request's named fields and answers with the slim
//! records the SGLang plugin declares). The Router cannot tell the two
//! implementations apart; Python keeps the lifecycle, launching the shim and
//! driving this server through the binding.

mod engine;
mod generate;
mod info;
mod service;
#[cfg(test)]
mod tests;

use std::{sync::Arc, time::Duration};

use engine_zmq_adapter::{
    StructuredOutputBackend, StructuredOutputsBackendConfig, StructuredOutputsSupport,
};
use smg_grpc_client::{
    common_proto as common, trtllm_proto::trtllm_service_server::TrtllmServiceServer,
};
use tonic::service::Routes;

use crate::{
    engine_link::EngineLink,
    kv_events::KvEventRelay,
    metrics::{Identity, ServicerMetrics},
    servicer::{self, Shared, WireState},
    ServicerConfig, ServicerError,
};

/// The gRPC service name, as the health service reports it.
pub const SERVICE_NAME: &str = "trtllm.TrtllmService";
/// The `wire` label of this servicer's metrics.
const WIRE: &str = "trtllm";
/// What `HealthCheck`'s status says of a serving engine on this wire (the
/// engine's Python servicer's own word).
const HEALTHY: &str = "OK";

/// What `GetModelInfo` and `GetServerInfo` report: the facts the engine's
/// Python servicer reads off the LLM API's arguments and tokenizer, which the
/// launcher computes from the same arguments before the engine starts. The
/// handshake supplies the window and the parallel sizes once the engine is
/// up, for a launcher that reported none.
#[derive(Debug, Clone, Default)]
pub struct TrtllmModelInfo {
    /// The checkpoint the engine loads.
    pub model_path: String,
    /// The name `GetModelInfo` reports as `model_id` (the served name, else
    /// the checkpoint, as the Python servicer reports it).
    pub served_model_name: String,
    /// The engine's input and sequence bounds (`max_input_len`,
    /// `max_seq_len`); the handshake's window is the fallback for either at 0.
    pub max_input_len: i32,
    pub max_seq_len: i32,
    pub max_batch_size: i32,
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub supported_features: Vec<String>,
    pub eos_token_ids: Vec<u32>,
    /// `GetServerInfo`'s `backend` (`pytorch`, `tensorrt`); the Python
    /// servicer's fixed word is the default.
    pub backend: String,
    pub trtllm_version: String,
    /// The parallel sizes; the handshake's figures are the fallback at 0.
    pub tensor_parallel_size: i32,
    pub pipeline_parallel_size: i32,
    pub context_parallel_size: i32,
    /// The shims that dial in (one per data-parallel engine); the
    /// handshake's figure wins once they are up.
    pub data_parallel_size: i32,
    /// The engine's guided-decoding backend as its arguments resolve it
    /// (`xgrammar`, `llguidance`, `none` when unset, "" when the launcher
    /// cannot tell).
    pub guided_decoding_backend: String,
    /// The engine's admission window, reported in the pushed load record;
    /// the handshake's figure is the fallback.
    pub max_running_requests: i32,
    /// A ZMQ KV-event publisher the shim runs, or empty: `SubscribeKvEvents`
    /// is UNIMPLEMENTED without one.
    pub kv_events_endpoint: String,
    pub kv_events_replay_endpoint: String,
    pub kv_events_topic: String,
}

/// What the engine enforces of a structured request, read off the
/// launcher's facts: the `guided_decoding_backend` the engine's arguments
/// resolve to. `none` enforces nothing (the engine refuses a guided request
/// at admission, which the shim reports as a 400 abort); `xgrammar` and
/// `llguidance` compile all four kinds. A launcher that could not tell leaves
/// nothing judged ahead of the engine. The adapter refuses a constraint
/// outside this before the engine, as on every wire.
pub(super) fn structured_outputs_support(model: &TrtllmModelInfo) -> StructuredOutputsSupport {
    use StructuredOutputsBackendConfig::{Other, Pinned};

    match model.guided_decoding_backend.trim() {
        "xgrammar" => Pinned(StructuredOutputBackend::Xgrammar).into(),
        "llguidance" => Pinned(StructuredOutputBackend::Guidance).into(),
        "none" => StructuredOutputsSupport::DISABLED,
        _ => Other.into(),
    }
}

/// How to bind, where the engines dial in, and what to advertise.
#[derive(Debug, Clone)]
pub struct TrtllmServicerConfig {
    /// The listener, the shim's endpoints, the tokenizer directory and the
    /// start-up bounds. `handshake_address` is what the shim dials,
    /// `engine_count` the number of shims (one per data-parallel engine);
    /// `tokenizer_dir` is bundled by `GetTokenizer` only, the engine keeps
    /// its own tokenizer for stop strings.
    pub servicer: ServicerConfig,
    pub model: TrtllmModelInfo,
}

/// The per-rank load the relay attaches to every batch it streams: the
/// engine's proto has no `GetLoads`, so the record is built from the load
/// tail of the shim's batches directly.
impl crate::kv_events::RankLoads for State {
    fn rank_load(&self, dp_rank: Option<i32>) -> Option<common::EngineLoad> {
        let client = self.shared.engine.client.get()?;
        let load = client
            .get_loads()
            .loads
            .into_iter()
            .find(|load| dp_rank.is_none_or(|rank| rank == load.dp_rank))?;
        Some(common::EngineLoad {
            running_requests: u32::try_from(load.num_running_reqs).unwrap_or(0),
            waiting_requests: u32::try_from(load.num_waiting_reqs).unwrap_or(0),
            token_usage: load.token_usage,
            max_running_requests: u32::try_from(
                self.shared.running_window(self.model.max_running_requests),
            )
            .unwrap_or(0),
            ..Default::default()
        })
    }

    fn engine_link(&self) -> &EngineLink {
        &self.shared.engine
    }
}

pub(super) struct State {
    pub(super) model: TrtllmModelInfo,
    /// What every servicer's state shares: the engine link, the admitted
    /// requests, the KV-event relay, the metrics and the serving switch.
    pub(super) shared: Shared,
}

impl WireState for State {
    fn shared(&self) -> &Shared {
        &self.shared
    }
}

/// A running Rust TensorRT-LLM servicer. `start` returns once the gRPC
/// listener is bound; the engines connect in the background and gate health.
pub struct TrtllmServicerServer {
    server: servicer::Server,
    state: Arc<State>,
}

impl TrtllmServicerServer {
    /// Validate, bind, and serve on a dedicated runtime thread.
    pub fn start(config: TrtllmServicerConfig) -> Result<Self, ServicerError> {
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
        let max_model_len = if config.model.max_seq_len > 0 {
            config.model.max_seq_len
        } else {
            config.model.max_input_len
        };
        let metrics = ServicerMetrics::new(
            &Identity {
                model: config.model.served_model_name.clone(),
                wire: WIRE,
                ranks: config.servicer.engine_count,
                max_model_len: u32::try_from(max_model_len).unwrap_or(0),
                // The engine matches string stops itself.
                matches_string_stops: false,
                // Images cross as the caller's bytes; the engine decodes them.
                processes_media: false,
            },
            config.servicer.metrics_bind_address.is_some(),
        )?;
        let state = Arc::new(State {
            model: config.model,
            shared: Shared::new(kv_relay, metrics, config.servicer.tokenizer_dir.clone()),
        });
        let routes = Routes::new(
            TrtllmServiceServer::new(service::TrtllmServicer {
                state: Arc::clone(&state),
            })
            .max_decoding_message_size(usize::MAX)
            .max_encoding_message_size(usize::MAX),
        );
        let connect_state = Arc::clone(&state);
        let server = servicer::Server::start(
            "smg-trtllm-servicer",
            config.servicer,
            SERVICE_NAME,
            "TensorRT-LLM",
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
    /// nothing else. `set_serving(false)` is the drain switch and closes
    /// admission as well.
    pub fn set_ready(&self, ready: bool) {
        self.state.shared.set_ready(ready);
    }

    /// Whether the KV-event relay's subscriptions have reached the engine's
    /// publishers, or there is none to wait for (events off, a relay told to
    /// start lazily, one that gave up).
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
