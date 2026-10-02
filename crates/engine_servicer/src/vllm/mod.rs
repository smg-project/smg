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
mod kv_events;
mod media;
mod requests;
mod service;
#[cfg(test)]
mod tests;
mod tokenizer_bundle;

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::{Duration, Instant},
};

use engine::{connect_engine, EngineLink};
use engine_zmq_adapter::ZmqEngineClient;
use futures::{FutureExt, StreamExt};
use llm_tokenizer::traits::Tokenizer;
use media::MediaGate;
pub use media::{
    BoxFuture, MediaError, MediaProcessor, MediaRefItem, MediaRequest, ProcessedMedia,
};
use requests::Registry;
use service::VllmEngineService;
use smg_grpc_client::vllm_proto::vllm_engine_server::VllmEngineServer;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Status};
use tonic_health::pb::health_server::HealthServer;
use tracing::{info, warn};

use crate::{health::HealthReporter, lock, ServerThread, ServicerError, Shutdown};

pub(crate) const SERVICE_NAME: &str = "vllm.grpc.engine.VllmEngine";
/// `GetServerInfo.server_type` for this implementation.
pub(crate) const SERVER_TYPE: &str = "vllm-grpc";

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
}

/// How to bind, where the engine dials in, and what to advertise.
#[derive(Clone)]
pub struct VllmServicerConfig {
    /// `host:port` for the gRPC listener.
    pub bind_address: String,
    /// `ipc://<path>` base for the data-plane sockets (`<path>-in.sock` and
    /// `<path>-out.sock`), owned by this process's uid.
    pub ipc_base_url: String,
    /// `tcp://host:port` the headless engine dials for the handshake (its
    /// `--data-parallel-address`/`--data-parallel-rpc-port`).
    pub handshake_address: String,
    /// Engines that will dial in (the engine-level data-parallel size).
    pub engine_count: usize,
    /// Local directory holding the model's tokenizer files; without it,
    /// requests carrying string stops are refused.
    pub tokenizer_dir: Option<String>,
    pub model: VllmModelInfo,
    /// Worker-side media processing for `media_refs`; `None` refuses them, as
    /// the Python servicer does with `--mm-processor off`.
    pub media_processor: Option<Arc<dyn MediaProcessor>>,
}

impl std::fmt::Debug for VllmServicerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VllmServicerConfig")
            .field("bind_address", &self.bind_address)
            .field("ipc_base_url", &self.ipc_base_url)
            .field("handshake_address", &self.handshake_address)
            .field("engine_count", &self.engine_count)
            .field("tokenizer_dir", &self.tokenizer_dir)
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

/// Token counters behind the periodic engine stats line.
#[derive(Default)]
pub(super) struct Stats {
    pub(super) prompt_tokens: AtomicU64,
    pub(super) generation_tokens: AtomicU64,
}

pub(super) struct State {
    pub(super) model: VllmModelInfo,
    pub(super) stats: Stats,
    /// The local tokenizer directory the servicer loaded (`GetTokenizer`
    /// bundles it); `None` when none resolved.
    pub(super) tokenizer_dir: Option<String>,
    pub(super) engine: EngineLink,
    /// Loaded once alongside the engine connect; `Some(None)` records a load
    /// that failed (string stops are then refused, EOS still comes from config).
    pub(super) tokenizer: OnceLock<Option<Arc<dyn Tokenizer>>>,
    pub(super) registry: Registry,
    pub(super) generation: AtomicU64,
    /// Cleared by the lifecycle owner to drain: health flips to NOT_SERVING
    /// while in-flight streams finish.
    pub(super) serving: AtomicBool,
    pub(super) started: Instant,
    /// The media processor behind its in-flight cap; `None` refuses
    /// `media_refs`.
    pub(super) media: Option<MediaGate>,
}

impl State {
    pub(super) fn engine(&self) -> Result<&ZmqEngineClient, Status> {
        if let Some(client) = self.engine.client.get() {
            return Ok(client);
        }
        Err(match self.engine.error() {
            Some(error) => Status::unavailable(format!("engine connection failed: {error}")),
            None => Status::unavailable("engine is still connecting"),
        })
    }

    pub(super) fn tokenizer(&self) -> Option<&Arc<dyn Tokenizer>> {
        self.tokenizer.get().and_then(Option::as_ref)
    }

    /// SERVING only once the engine link is up and alive and the lifecycle
    /// owner has not started draining.
    pub(super) fn is_serving(&self) -> bool {
        self.serving.load(Ordering::Acquire)
            && self
                .engine
                .client
                .get()
                .is_some_and(ZmqEngineClient::is_alive)
    }

    pub(super) fn health_message(&self) -> &'static str {
        if !self.serving.load(Ordering::Acquire) {
            return "Draining";
        }
        match (self.engine.client.get(), self.engine.error()) {
            (Some(client), _) if client.is_alive() => "Health",
            (Some(_), _) => "Engine is not alive",
            (None, Some(_)) => "Engine connection failed",
            (None, None) => "Engine is starting",
        }
    }

    pub(super) fn active_requests(&self) -> u32 {
        self.registry
            .lock()
            .map(|registry| u32::try_from(registry.len()).unwrap_or(u32::MAX))
            .unwrap_or(0)
    }
}

/// A running Rust vLLM servicer. `start` returns once the gRPC listener is
/// bound; the engine connects in the background and gates health.
pub struct VllmServicerServer {
    thread: ServerThread,
    state: Arc<State>,
}

impl VllmServicerServer {
    /// Validate, bind, and serve on a dedicated runtime thread.
    pub fn start(config: VllmServicerConfig) -> Result<Self, ServicerError> {
        Self::start_inner(config, None)
    }

    /// [`Self::start`] with the tokenizer supplied instead of loaded from
    /// `tokenizer_dir`.
    #[cfg(test)]
    pub(crate) fn start_with_tokenizer(
        config: VllmServicerConfig,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> Result<Self, ServicerError> {
        Self::start_inner(config, Some(tokenizer))
    }

    fn start_inner(
        config: VllmServicerConfig,
        tokenizer: Option<Arc<dyn Tokenizer>>,
    ) -> Result<Self, ServicerError> {
        let invalid = |message: &str| ServicerError::InvalidConfig(message.to_string());
        if config.bind_address.rsplit_once(':').is_none() {
            return Err(invalid("bind_address must be host:port"));
        }
        if !config.ipc_base_url.starts_with("ipc://") {
            return Err(invalid("ipc_base_url must be ipc://<path>"));
        }
        if !config.handshake_address.starts_with("tcp://") {
            return Err(invalid("handshake_address must be tcp://host:port"));
        }
        if config.engine_count == 0 {
            return Err(invalid("engine_count must be positive"));
        }
        if config.model.model_path.trim().is_empty() {
            return Err(invalid("model_path must not be empty"));
        }
        let state = Arc::new(State {
            tokenizer_dir: config.tokenizer_dir.clone(),
            model: config.model,
            stats: Stats::default(),
            engine: EngineLink::default(),
            tokenizer: OnceLock::new(),
            registry: Mutex::new(HashMap::new()),
            generation: AtomicU64::new(0),
            serving: AtomicBool::new(true),
            started: Instant::now(),
            media: config.media_processor.map(MediaGate::new),
        });
        if let Some(tokenizer) = tokenizer {
            let _ = state.tokenizer.set(Some(tokenizer));
        }
        let service = VllmEngineService {
            state: Arc::clone(&state),
        };
        let health_state = Arc::clone(&state);
        let health = HealthReporter::new(
            &["", SERVICE_NAME],
            Arc::new(move || health_state.is_serving()),
        );
        let connect_state = Arc::clone(&state);
        let stats_state = Arc::clone(&state);
        let VllmServicerConfig {
            bind_address,
            ipc_base_url,
            handshake_address,
            engine_count,
            tokenizer_dir,
            ..
        } = config;
        let thread = ServerThread::start(
            "smg-vllm-servicer",
            &bind_address,
            move |listener: TcpListener, shutdown: Shutdown, last_error| async move {
                // Fire-and-forget on the server runtime: dropping the runtime
                // with the server thread cancels a still-running connect.
                #[expect(
                    clippy::disallowed_methods,
                    reason = "engine connect is fire-and-forget; the runtime drop cancels it"
                )]
                let _connect = tokio::spawn(connect_engine(
                    connect_state,
                    ipc_base_url,
                    handshake_address,
                    engine_count,
                    tokenizer_dir,
                    last_error,
                ));
                #[expect(
                    clippy::disallowed_methods,
                    reason = "the stats log is fire-and-forget; the runtime drop ends it"
                )]
                let _stats = tokio::spawn(info::log_engine_stats(stats_state));
                info!(address = %listener.local_addr().map(|a| a.to_string()).unwrap_or_default(), "vLLM gRPC servicer listening");
                // Graceful first: stop accepting, let open streams finish. A
                // connected client's idle keepalive connection would hold a
                // purely graceful shutdown forever, so the drain grace bounds it
                // and the server future is dropped (connections closed) after.
                let graceful = Server::builder()
                    // Router-preprocessed media arrives as multi-megabyte
                    // inline tensors; hyper's server defaults (1 MiB stream
                    // and connection windows, 16 KiB frames) would stall each
                    // such message on window updates. Size the windows and
                    // frames for them (grpc-core's BDP probing gets the
                    // Python servicer there on its own).
                    .initial_stream_window_size(Some(16 * 1024 * 1024))
                    .initial_connection_window_size(Some(64 * 1024 * 1024))
                    .max_frame_size(Some(1024 * 1024))
                    .add_service(
                        VllmEngineServer::new(service)
                            .max_decoding_message_size(usize::MAX)
                            .max_encoding_message_size(usize::MAX),
                    )
                    .add_service(HealthServer::new(health))
                    .serve_with_incoming_shutdown(
                        // tonic sets TCP_NODELAY only on listeners it owns; on
                        // this one Nagle would coalesce per-token frames into
                        // bursts whenever a connection is lightly loaded.
                        TcpListenerStream::new(listener).map(|conn| {
                            if let Ok(stream) = &conn {
                                if let Err(error) = stream.set_nodelay(true) {
                                    warn!(%error, "could not set TCP_NODELAY on an accepted connection");
                                }
                            }
                            conn
                        }),
                        shutdown.clone().map(|_| ()),
                    );
                let forced = async move {
                    let grace = shutdown.await;
                    tokio::time::sleep(grace).await;
                };
                tokio::select! {
                    result = graceful => result.map_err(|error| error.to_string()),
                    () = forced => {
                        warn!("gRPC servicer closed its remaining connections after the drain grace");
                        Ok(())
                    }
                }
            },
        )?;
        Ok(Self { thread, state })
    }

    /// The bound `ip:port`.
    pub fn address(&self) -> String {
        self.thread.address().to_string()
    }

    pub fn running(&self) -> bool {
        self.thread.running()
    }

    /// Whether the engine handshake completed.
    pub fn engine_ready(&self) -> bool {
        self.state.engine.client.get().is_some()
    }

    pub fn last_error(&self) -> Result<Option<String>, ServicerError> {
        self.thread.last_error()
    }

    /// Announce SERVING (`true`) or drain (`false`): health flips at once,
    /// in-flight streams keep running until `stop`.
    pub fn set_serving(&self, serving: bool) {
        self.state.serving.store(serving, Ordering::Release);
    }

    /// Signal shutdown and wait up to `timeout` for the server thread; in-flight
    /// streams are cancelled with it (which aborts their engine requests).
    pub fn stop(&self, timeout: Duration) -> Result<(), ServicerError> {
        self.set_serving(false);
        // Fire every registered cancellation so streams end before the
        // listener closes, then let the thread wind down.
        {
            let mut registry = lock(&self.state.registry)?;
            for (_, (_, cancel)) in registry.drain() {
                let _ = cancel.send(());
            }
        }
        self.thread.stop(timeout)
    }
}
