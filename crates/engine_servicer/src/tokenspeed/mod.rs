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

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant, SystemTime},
};

use engine_zmq_adapter::ZmqEngineClient;
use futures::FutureExt;
use llm_tokenizer::traits::Tokenizer;
use smg_grpc_client::tokenspeed_proto::token_speed_scheduler_server::TokenSpeedSchedulerServer;
use tokio::net::TcpListener;
use tokio_stream::{wrappers::TcpListenerStream, StreamExt};
use tonic::{transport::Server, Status};
use tonic_health::pb::health_server::HealthServer;
use tracing::{info, warn};

use crate::{
    engine_link::EngineLink, health::HealthReporter, requests::RequestRegistry, ServerThread,
    ServicerError, Shutdown,
};

/// The gRPC service name, as the health service reports it.
pub const SERVICE_NAME: &str = "tokenspeed.grpc.scheduler.TokenSpeedScheduler";

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
    /// The scheduler's admission window (`max_num_seqs`), reported as
    /// `max_running_requests`; the handshake's figure is the fallback.
    pub max_running_requests: i32,
    /// The attention data-parallel size the launcher configured; the
    /// handshake's figure wins once the engines are up.
    pub data_parallel_size: i32,
    /// TokenSpeed's ZMQ KV-event publisher endpoint and topic; an empty
    /// endpoint means `SubscribeKvEvents` is UNIMPLEMENTED.
    pub kv_events_endpoint: String,
    pub kv_events_topic: String,
}

/// How to bind, where the engines dial in, and what to advertise.
#[derive(Debug, Clone)]
pub struct TokenSpeedServicerConfig {
    /// `host:port` for the gRPC listener.
    pub bind_address: String,
    /// `ipc://<path>` base for the data-plane sockets (`<path>-in.sock` and
    /// `<path>-out.sock`), owned by this process's uid.
    pub ipc_base_url: String,
    /// `tcp://host:port` the headless scheduler dials for the handshake (its
    /// `--data-parallel-address`/`--data-parallel-rpc-port`).
    pub handshake_address: String,
    /// Engines that will dial in (the attention data-parallel size).
    pub engine_count: usize,
    /// Local directory holding the model's tokenizer files; without it,
    /// requests carrying string stops are refused.
    pub tokenizer_dir: Option<String>,
    pub model: TokenSpeedModelInfo,
    /// Bound on the scheduler's startup handshake; see
    /// [`crate::DEFAULT_ENGINE_STARTUP_TIMEOUT`].
    pub engine_startup_timeout: Duration,
}

pub(super) struct State {
    pub(super) model: TokenSpeedModelInfo,
    /// The local tokenizer directory the servicer loaded (`GetTokenizer`
    /// bundles it); `None` when none resolved.
    pub(super) tokenizer_dir: Option<String>,
    pub(super) engine: EngineLink,
    /// Loaded once alongside the engine connect; `Some(None)` records a load
    /// that failed (string stops are then refused).
    pub(super) tokenizer: OnceLock<Option<Arc<dyn Tokenizer>>>,
    pub(super) registry: Arc<RequestRegistry>,
    /// Cleared by the lifecycle owner to drain: health flips to NOT_SERVING
    /// while in-flight streams finish.
    pub(super) serving: AtomicBool,
    pub(super) started: Instant,
    pub(super) started_at: SystemTime,
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
            (Some(client), _) if client.is_alive() => "healthy",
            (Some(_), _) => "Engine is not alive",
            (None, Some(_)) => "Engine connection failed",
            (None, None) => "Engine is starting",
        }
    }
}

/// A running Rust TokenSpeed servicer. `start` returns once the gRPC
/// listener is bound; the engines connect in the background and gate health.
pub struct TokenSpeedServicerServer {
    thread: ServerThread,
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
        if config.engine_startup_timeout.is_zero() {
            return Err(invalid("engine_startup_timeout must be positive"));
        }
        if config.model.model_path.trim().is_empty() {
            return Err(invalid("model_path must not be empty"));
        }
        let state = Arc::new(State {
            tokenizer_dir: config.tokenizer_dir.clone(),
            model: config.model,
            engine: EngineLink::default(),
            tokenizer: OnceLock::new(),
            registry: Arc::new(RequestRegistry::default()),
            serving: AtomicBool::new(true),
            started: Instant::now(),
            started_at: SystemTime::now(),
        });
        if let Some(tokenizer) = tokenizer {
            let _ = state.tokenizer.set(Some(tokenizer));
        }
        let service = service::TokenSpeedService {
            state: Arc::clone(&state),
        };
        let health_state = Arc::clone(&state);
        let health = HealthReporter::new(
            &["", SERVICE_NAME],
            Arc::new(move || health_state.is_serving()),
        );
        let connect_state = Arc::clone(&state);
        let TokenSpeedServicerConfig {
            bind_address,
            ipc_base_url,
            handshake_address,
            engine_count,
            tokenizer_dir,
            engine_startup_timeout,
            ..
        } = config;
        let thread = ServerThread::start(
            "smg-tokenspeed-servicer",
            &bind_address,
            move |listener: TcpListener, shutdown: Shutdown, last_error| async move {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "engine connect is fire-and-forget; the runtime drop cancels it"
                )]
                let _connect = tokio::spawn(engine::connect_engine(
                    connect_state,
                    ipc_base_url,
                    handshake_address,
                    engine_count,
                    engine_startup_timeout,
                    tokenizer_dir,
                    last_error,
                ));
                info!(address = %listener.local_addr().map(|a| a.to_string()).unwrap_or_default(), "TokenSpeed gRPC servicer listening");
                // Graceful first: stop accepting, let open streams finish;
                // the drain grace bounds a client holding an idle connection.
                let graceful = Server::builder()
                    // Router-preprocessed media arrives as multi-megabyte
                    // inline tensors; size the HTTP/2 windows for them.
                    .initial_stream_window_size(Some(16 * 1024 * 1024))
                    .initial_connection_window_size(Some(64 * 1024 * 1024))
                    .max_frame_size(Some(1024 * 1024))
                    .add_service(
                        TokenSpeedSchedulerServer::new(service)
                            .max_decoding_message_size(usize::MAX)
                            .max_encoding_message_size(usize::MAX),
                    )
                    .add_service(HealthServer::new(health))
                    .serve_with_incoming_shutdown(
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

    /// Signal shutdown and wait up to `timeout` for the server thread;
    /// in-flight streams are cancelled with it (which aborts their engine
    /// requests).
    pub fn stop(&self, timeout: Duration) -> Result<(), ServicerError> {
        self.set_serving(false);
        self.state.registry.cancel_all()?;
        self.thread.stop(timeout)
    }
}
