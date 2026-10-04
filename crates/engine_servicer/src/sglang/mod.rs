//! The `sglang.grpc.scheduler.SglangScheduler` servicer in Rust: the contract
//! the Python `smg_grpc_servicer.sglang` package serves, over the msgpack ZMQ
//! wire a headless SGLang scheduler dials (what the SMG plugin inside the
//! scheduler speaks to the Router's direct lane). The Router cannot tell the
//! two implementations apart; Python keeps the lifecycle, launching the
//! headless scheduler and driving this server through the binding.

mod admin;
mod embed;
mod engine;
mod generate;
mod info;
mod service;
#[cfg(test)]
mod tests;

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime},
};

use engine_zmq_adapter::ZmqEngineClient;
use futures::FutureExt;
use smg_grpc_client::sglang_proto::sglang_scheduler_server::SglangSchedulerServer;
use tokio::net::TcpListener;
use tokio_stream::{wrappers::TcpListenerStream, StreamExt};
use tonic::{transport::Server, Status};
use tonic_health::pb::health_server::HealthServer;
use tracing::{info, warn};

use crate::{
    engine_link::EngineLink, health::HealthReporter, proto_json, requests::RequestRegistry,
    ServerThread, ServicerError, Shutdown,
};

/// The gRPC service name, as the health service reports it.
pub const SERVICE_NAME: &str = "sglang.grpc.scheduler.SglangScheduler";

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
}

/// How to bind, where the engines dial in, and what to advertise.
#[derive(Debug, Clone)]
pub struct SglangServicerConfig {
    /// `host:port` for the gRPC listener.
    pub bind_address: String,
    /// `ipc://<path>` base for the data-plane sockets (`<path>-in.sock` and
    /// `<path>-out.sock`), owned by this process's uid.
    pub ipc_base_url: String,
    /// `tcp://host:port` the headless scheduler dials for the handshake (the
    /// launcher's `--zmq-handshake-address`).
    pub handshake_address: String,
    /// Engines that will dial in (the data-parallel size).
    pub engine_count: usize,
    /// Local directory holding the model's tokenizer files, bundled by
    /// `GetTokenizer`; the scheduler keeps its own tokenizer for stop strings.
    pub tokenizer_dir: Option<String>,
    pub model: SglangModelInfo,
    /// Bound on the scheduler's startup handshake; see
    /// [`crate::DEFAULT_ENGINE_STARTUP_TIMEOUT`].
    pub engine_startup_timeout: Duration,
}

pub(super) struct State {
    pub(super) model: SglangModelInfo,
    /// The local tokenizer directory `GetTokenizer` bundles; `None` when none
    /// resolved.
    pub(super) tokenizer_dir: Option<String>,
    pub(super) engine: EngineLink,
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

/// A running Rust SGLang servicer. `start` returns once the gRPC listener is
/// bound; the engines connect in the background and gate health.
pub struct SglangServicerServer {
    thread: ServerThread,
    state: Arc<State>,
}

impl SglangServicerServer {
    /// Validate, bind, and serve on a dedicated runtime thread.
    pub fn start(config: SglangServicerConfig) -> Result<Self, ServicerError> {
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
        let state = Arc::new(State {
            tokenizer_dir: config.tokenizer_dir.clone(),
            model: config.model,
            engine: EngineLink::default(),
            registry: Arc::new(RequestRegistry::default()),
            serving: AtomicBool::new(true),
            started: Instant::now(),
            started_at: SystemTime::now(),
        });
        let service = service::SglangService {
            state: Arc::clone(&state),
        };
        let health_state = Arc::clone(&state);
        let health = HealthReporter::new(
            &["", SERVICE_NAME],
            Arc::new(move || health_state.is_serving()),
        );
        let connect_state = Arc::clone(&state);
        let SglangServicerConfig {
            bind_address,
            ipc_base_url,
            handshake_address,
            engine_count,
            engine_startup_timeout,
            ..
        } = config;
        let thread = ServerThread::start(
            "smg-sglang-servicer",
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
                    last_error,
                ));
                info!(address = %listener.local_addr().map(|a| a.to_string()).unwrap_or_default(), "SGLang gRPC servicer listening");
                // Graceful first: stop accepting, let open streams finish;
                // the drain grace bounds a client holding an idle connection.
                let graceful = Server::builder()
                    // Router-preprocessed media arrives as multi-megabyte
                    // inline tensors; size the HTTP/2 windows for them.
                    .initial_stream_window_size(Some(16 * 1024 * 1024))
                    .initial_connection_window_size(Some(64 * 1024 * 1024))
                    .max_frame_size(Some(1024 * 1024))
                    .add_service(
                        SglangSchedulerServer::new(service)
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
