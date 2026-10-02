//! The Worker's control plane: `WorkerControl`, `grpc.health.v1`, and
//! optionally `WorkerInference`, served from one Rust-owned runtime thread.
//!
//! The listener binds before the engine transport connects, so STARTING is
//! observable for the whole engine handshake. Health is the announced lifecycle
//! gated on that link: SERVING only once both the lifecycle owner announced it
//! and the transport is connected.

use std::{
    collections::HashMap,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError},
        Arc, Mutex, OnceLock,
    },
    thread,
    time::{Duration, SystemTime},
};

use futures::Stream;
pub use proto::WorkerHealthState;
use smg_grpc_client::{
    worker_control::{WORKER_CONTROL_API_MAJOR, WORKER_CONTROL_API_MINOR},
    worker_inference::{
        connect_engine_transport, EngineTransport, EngineTransportStream, EngineWorkerInference,
    },
    worker_inference_proto::{self, worker_inference_server::WorkerInferenceServer},
    worker_proto::{
        self as proto,
        worker_control_server::{WorkerControl, WorkerControlServer as TonicWorkerControlServer},
    },
};
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{server::NamedService, transport::Server, Request, Response, Status};
use tonic_health::pb::{
    health_check_response::ServingStatus,
    health_server::{Health, HealthServer},
    HealthCheckRequest, HealthCheckResponse,
};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use super::engine_transport::ZmqWorkerTransport;
use crate::worker::{RuntimeType, TOKEN_ONLY_WIRE_FEATURE};

/// Budget for the server thread to bind its listener.
const BIND_TIMEOUT: Duration = Duration::from_secs(5);
/// Poll cadence of the `grpc.health.v1` `Watch` stream.
const HEALTH_WATCH_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, thiserror::Error)]
pub enum WorkerNodeError {
    #[error("{0}")]
    InvalidConfig(String),
    #[error("{0}")]
    Startup(String),
    #[error("Worker state is poisoned")]
    Poisoned,
    #[error("Worker server did not stop before the timeout")]
    StopTimeout,
    #[error("Worker server thread panicked")]
    ThreadPanicked,
}

/// Everything a Worker advertises and connects to. `engine_type` names the
/// colocated engine; when `inference_enabled` it must be one the Worker can
/// front (`vllm` or `tokenspeed`).
#[derive(Clone, Debug)]
pub struct WorkerNodeConfig {
    /// `host:port`; the host may be a name or an IP literal.
    pub bind_address: String,
    pub worker_id: String,
    pub instance_id: Option<String>,
    pub hostname: Option<String>,
    pub zone: String,
    pub engine_type: String,
    pub engine_version: String,
    pub engine_endpoint: String,
    pub model_ids: Vec<String>,
    pub features: Option<Vec<String>>,
    pub max_concurrent_requests: u32,
    pub inference_enabled: bool,
    pub engine_attributes: HashMap<String, String>,
    /// `grpc` or `zmq`.
    pub engine_transport: String,
    pub zmq_handshake_address: Option<String>,
    pub engine_count: usize,
}

/// Install a stdout `tracing` subscriber for a process whose only Rust
/// component is the Worker. A second call is a no-op.
pub fn init_tracing(level: Option<&str>) -> Result<(), WorkerNodeError> {
    let filter = match level {
        Some(level) => EnvFilter::try_new(level).map_err(|error| {
            WorkerNodeError::InvalidConfig(format!("invalid log level {level:?}: {error}"))
        })?,
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    };
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    Ok(())
}

pub fn parse_health_state(state: &str) -> Result<WorkerHealthState, WorkerNodeError> {
    match state.to_ascii_lowercase().as_str() {
        "starting" => Ok(WorkerHealthState::Starting),
        "serving" => Ok(WorkerHealthState::Serving),
        "degraded" => Ok(WorkerHealthState::Degraded),
        "draining" => Ok(WorkerHealthState::Draining),
        "not_serving" | "not-serving" => Ok(WorkerHealthState::NotServing),
        _ => Err(WorkerNodeError::InvalidConfig(format!(
            "unknown worker health state {state:?}"
        ))),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkerEngine {
    Vllm,
    TokenSpeed,
}

impl WorkerEngine {
    fn parse(engine_type: &str) -> Result<Self, WorkerNodeError> {
        match engine_type.to_ascii_lowercase().as_str() {
            "vllm" => Ok(Self::Vllm),
            "tokenspeed" | "ts" => Ok(Self::TokenSpeed),
            other => Err(WorkerNodeError::InvalidConfig(format!(
                "a Worker can front vllm or tokenspeed, not {other:?}"
            ))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::TokenSpeed => "tokenspeed",
        }
    }

    fn runtime(self) -> RuntimeType {
        match self {
            Self::Vllm => RuntimeType::Vllm,
            Self::TokenSpeed => RuntimeType::TokenSpeed,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkerEngineTransport {
    Grpc,
    Zmq,
}

impl WorkerEngineTransport {
    fn parse(value: &str) -> Result<Self, WorkerNodeError> {
        match value.to_ascii_lowercase().as_str() {
            "grpc" => Ok(Self::Grpc),
            "zmq" => Ok(Self::Zmq),
            _ => Err(WorkerNodeError::InvalidConfig(format!(
                "unknown Worker engine transport {value:?}; expected grpc or zmq"
            ))),
        }
    }

    /// The `engine_transport` attribute the Router reads back off the
    /// registration; must round-trip through [`Self::parse`].
    fn label(self) -> &'static str {
        match self {
            Self::Grpc => "grpc",
            Self::Zmq => "zmq",
        }
    }
}

/// `token_only_wire` and the `engine_transport` attribute follow the transport:
/// a ZMQ engine cannot match string stops, and the Router keeps stop trimming
/// and the EOS backstop on its side when it sees that.
fn advertise_engine_transport(
    transport: WorkerEngineTransport,
    mut features: Vec<String>,
    mut engine_attributes: HashMap<String, String>,
) -> (Vec<String>, HashMap<String, String>) {
    if transport == WorkerEngineTransport::Zmq
        && !features
            .iter()
            .any(|feature| feature == TOKEN_ONLY_WIRE_FEATURE)
    {
        features.push(TOKEN_ONLY_WIRE_FEATURE.to_string());
    }
    engine_attributes
        .entry("engine_transport".to_string())
        .or_insert_with(|| transport.label().to_string());
    (features, engine_attributes)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HealthSnapshot {
    state: WorkerHealthState,
    message: String,
}

/// Readiness of the engine transport, independent of the announced lifecycle.
#[derive(Default)]
struct EngineLink {
    ready: AtomicBool,
    error: Mutex<Option<String>>,
}

impl EngineLink {
    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    fn error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|error| error.clone())
    }

    fn fail(&self, message: String) {
        if let Ok(mut slot) = self.error.lock() {
            *slot = Some(message);
        }
    }
}

/// Engine transport requests can be routed to before it exists: every call is
/// refused with UNAVAILABLE until the background connect installs the real one.
struct LazyEngineTransport {
    inner: OnceLock<Arc<dyn EngineTransport>>,
    link: Arc<EngineLink>,
}

impl LazyEngineTransport {
    fn new(link: Arc<EngineLink>) -> Self {
        Self {
            inner: OnceLock::new(),
            link,
        }
    }

    fn connected(&self) -> Result<&Arc<dyn EngineTransport>, Status> {
        if let Some(transport) = self.inner.get() {
            return Ok(transport);
        }
        Err(match self.link.error() {
            Some(error) => Status::unavailable(format!("Worker engine transport failed: {error}")),
            None => Status::unavailable("Worker engine transport is still connecting"),
        })
    }
}

#[tonic::async_trait]
impl EngineTransport for LazyEngineTransport {
    async fn generate(
        &self,
        request: worker_inference_proto::GenerateRequest,
    ) -> Result<EngineTransportStream, Status> {
        self.connected()?.generate(request).await
    }

    async fn abort(
        &self,
        request: worker_inference_proto::AbortRequest,
    ) -> Result<worker_inference_proto::AbortResponse, Status> {
        self.connected()?.abort(request).await
    }
}

/// The health the Worker reports: the announced lifecycle, gated on the engine
/// transport when there is one.
fn effective_health(announced: HealthSnapshot, engine_link: Option<&EngineLink>) -> HealthSnapshot {
    let Some(link) = engine_link else {
        return announced;
    };
    if let Some(error) = link.error() {
        return HealthSnapshot {
            state: WorkerHealthState::NotServing,
            message: format!("engine transport failed: {error}"),
        };
    }
    if !link.is_ready()
        && matches!(
            announced.state,
            WorkerHealthState::Serving | WorkerHealthState::Degraded
        )
    {
        return HealthSnapshot {
            state: WorkerHealthState::Starting,
            message: "waiting for the engine transport to connect".to_string(),
        };
    }
    announced
}

struct NodeState {
    identity: proto::WorkerIdentity,
    capabilities: proto::WorkerCapabilities,
    topology: proto::WorkerTopology,
    health: Arc<Mutex<HealthSnapshot>>,
    engine_link: Option<Arc<EngineLink>>,
    inference_enabled: bool,
}

/// What the Worker advertises, built once from the validated config.
struct Advertised {
    worker_id: String,
    instance_id: String,
    hostname: String,
    zone: String,
    engine_type: String,
    engine_version: String,
    engine_endpoint: String,
    model_ids: Vec<String>,
    features: Vec<String>,
    max_concurrent_requests: u32,
    engine_attributes: HashMap<String, String>,
}

#[derive(Clone)]
struct NodeControl {
    state: Arc<NodeState>,
}

impl NodeControl {
    fn new(
        advertised: Advertised,
        health: Arc<Mutex<HealthSnapshot>>,
        engine_link: Option<Arc<EngineLink>>,
    ) -> Self {
        let engine = proto::EngineCapability {
            engine_type: advertised.engine_type.clone(),
            engine_version: advertised.engine_version,
            model_ids: advertised.model_ids.clone(),
            features: advertised.features.clone(),
        };
        Self {
            state: Arc::new(NodeState {
                identity: proto::WorkerIdentity {
                    worker_id: advertised.worker_id.clone(),
                    instance_id: advertised.instance_id,
                    hostname: advertised.hostname,
                    zone: advertised.zone,
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    started_at: Some(now()),
                    labels: [("role".to_string(), "smg-worker".to_string())].into(),
                },
                capabilities: proto::WorkerCapabilities {
                    api_major: WORKER_CONTROL_API_MAJOR,
                    api_minor: WORKER_CONTROL_API_MINOR,
                    features: advertised.features,
                    engines: vec![engine],
                    max_concurrent_requests: advertised.max_concurrent_requests,
                    attributes: advertised.engine_attributes.clone(),
                },
                topology: proto::WorkerTopology {
                    worker_id: advertised.worker_id,
                    topology_version: 1,
                    engines: vec![proto::EngineEndpoint {
                        engine_id: "engine-0".to_string(),
                        engine_type: advertised.engine_type,
                        endpoint: advertised.engine_endpoint,
                        model_ids: advertised.model_ids,
                        replica_group: String::new(),
                        data_parallel_rank: None,
                        tensor_parallel_rank: None,
                        pipeline_parallel_rank: None,
                        attributes: advertised.engine_attributes,
                    }],
                    observed_at: Some(now()),
                },
                health,
                inference_enabled: engine_link.is_some(),
                engine_link,
            }),
        }
    }

    fn health_snapshot(&self) -> Result<HealthSnapshot, Status> {
        let announced = self
            .state
            .health
            .lock()
            .map_err(|_| Status::internal("Worker health state is poisoned"))?
            .clone();
        Ok(effective_health(
            announced,
            self.state.engine_link.as_deref(),
        ))
    }

    /// `grpc.health.v1` view: every service this process serves shares one
    /// status; a name it does not serve is SERVICE_UNKNOWN.
    fn serving_status(&self, service: &str) -> Result<ServingStatus, Status> {
        let known = service.is_empty()
            || service == <TonicWorkerControlServer<Self> as NamedService>::NAME
            || service == <HealthServer<Self> as NamedService>::NAME
            || (self.state.inference_enabled
                && service == <WorkerInferenceServer<EngineWorkerInference> as NamedService>::NAME);
        if !known {
            return Ok(ServingStatus::ServiceUnknown);
        }
        Ok(
            if self.health_snapshot()?.state == WorkerHealthState::Serving {
                ServingStatus::Serving
            } else {
                ServingStatus::NotServing
            },
        )
    }
}

type HealthWatchStream = Pin<Box<dyn Stream<Item = Result<HealthCheckResponse, Status>> + Send>>;

#[tonic::async_trait]
impl Health for NodeControl {
    type WatchStream = HealthWatchStream;

    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let service = request.into_inner().service;
        let status = self.serving_status(&service)?;
        if status == ServingStatus::ServiceUnknown {
            return Err(Status::not_found(format!("unknown service {service:?}")));
        }
        Ok(Response::new(HealthCheckResponse {
            status: status as i32,
        }))
    }

    async fn watch(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        let service = request.into_inner().service;
        let stream = futures::stream::unfold(
            (self.clone(), service, None),
            |(control, service, last)| async move {
                loop {
                    let status = match control.serving_status(&service) {
                        Ok(status) => status as i32,
                        Err(status) => return Some((Err(status), (control, service, last))),
                    };
                    if last != Some(status) {
                        return Some((
                            Ok(HealthCheckResponse { status }),
                            (control, service, Some(status)),
                        ));
                    }
                    tokio::time::sleep(HEALTH_WATCH_INTERVAL).await;
                }
            },
        );
        Ok(Response::new(Box::pin(stream)))
    }
}

#[tonic::async_trait]
impl WorkerControl for NodeControl {
    async fn get_identity(
        &self,
        _request: Request<proto::GetIdentityRequest>,
    ) -> Result<Response<proto::GetIdentityResponse>, Status> {
        Ok(Response::new(proto::GetIdentityResponse {
            identity: Some(self.state.identity.clone()),
        }))
    }

    async fn get_capabilities(
        &self,
        _request: Request<proto::GetCapabilitiesRequest>,
    ) -> Result<Response<proto::GetCapabilitiesResponse>, Status> {
        Ok(Response::new(proto::GetCapabilitiesResponse {
            capabilities: Some(self.state.capabilities.clone()),
        }))
    }

    async fn get_health(
        &self,
        request: Request<proto::GetHealthRequest>,
    ) -> Result<Response<proto::GetHealthResponse>, Status> {
        let health = self.health_snapshot()?;
        let components = request
            .into_inner()
            .include_components
            .then(|| proto::ComponentHealth {
                component_id: "engine-0".to_string(),
                state: health.state.into(),
                message: health.message.clone(),
                checked_at: Some(now()),
            })
            .into_iter()
            .collect();
        Ok(Response::new(proto::GetHealthResponse {
            state: health.state.into(),
            message: health.message,
            checked_at: Some(now()),
            components,
        }))
    }

    async fn get_topology(
        &self,
        _request: Request<proto::GetTopologyRequest>,
    ) -> Result<Response<proto::GetTopologyResponse>, Status> {
        Ok(Response::new(proto::GetTopologyResponse {
            topology: Some(self.state.topology.clone()),
        }))
    }
}

fn now() -> prost_types::Timestamp {
    SystemTime::now().into()
}

struct InferenceConfig {
    engine: WorkerEngine,
    engine_endpoint: String,
    model_id: String,
    max_concurrent_requests: u32,
    serving: Arc<AtomicBool>,
    transport: WorkerEngineTransport,
    zmq_handshake_address: Option<String>,
    engine_count: usize,
    link: Arc<EngineLink>,
}

async fn connect_inference(
    config: &InferenceConfig,
) -> Result<Arc<dyn EngineTransport>, Box<dyn std::error::Error + Send + Sync>> {
    match config.transport {
        WorkerEngineTransport::Grpc => {
            connect_engine_transport(config.engine.name(), &config.engine_endpoint).await
        }
        WorkerEngineTransport::Zmq => {
            let transport = ZmqWorkerTransport::connect(
                &config.engine_endpoint,
                config.model_id.clone(),
                config.engine.runtime(),
                config.zmq_handshake_address.as_deref(),
                config.engine_count,
            )
            .await
            .map_err(std::io::Error::other)?;
            Ok(Arc::new(transport) as Arc<dyn EngineTransport>)
        }
    }
}

/// A validated configuration, ready to bind.
struct StartPlan {
    bind_address: String,
    advertised: Advertised,
    inference: Option<InferenceConfig>,
    engine_link: Option<Arc<EngineLink>>,
    health: Arc<Mutex<HealthSnapshot>>,
    serving: Arc<AtomicBool>,
}

impl StartPlan {
    fn from_config(config: WorkerNodeConfig) -> Result<Self, WorkerNodeError> {
        let invalid = |message: &str| WorkerNodeError::InvalidConfig(message.to_string());
        if config.worker_id.trim().is_empty() {
            return Err(invalid("worker_id must not be empty"));
        }
        if config.engine_type.trim().is_empty() {
            return Err(invalid("engine_type must not be empty"));
        }
        if config.engine_endpoint.trim().is_empty() {
            return Err(invalid("engine_endpoint must not be empty"));
        }
        let Some(primary_model) = config.model_ids.first().cloned() else {
            return Err(invalid("model_ids must not be empty"));
        };
        if config.bind_address.rsplit_once(':').is_none() {
            return Err(invalid("bind_address must be host:port"));
        }
        if config.engine_count == 0 {
            return Err(invalid("engine_count must be positive"));
        }
        let transport = WorkerEngineTransport::parse(&config.engine_transport)?;
        let health = Arc::new(Mutex::new(HealthSnapshot {
            state: WorkerHealthState::Starting,
            message: "starting".to_string(),
        }));
        let serving = Arc::new(AtomicBool::new(false));
        let engine_link = config
            .inference_enabled
            .then(|| Arc::new(EngineLink::default()));
        let inference = match &engine_link {
            Some(link) => Some(InferenceConfig {
                engine: WorkerEngine::parse(&config.engine_type)?,
                engine_endpoint: config.engine_endpoint.clone(),
                model_id: primary_model,
                max_concurrent_requests: config.max_concurrent_requests,
                serving: Arc::clone(&serving),
                transport,
                zmq_handshake_address: config.zmq_handshake_address,
                engine_count: config.engine_count,
                link: Arc::clone(link),
            }),
            None => None,
        };
        let (features, engine_attributes) = advertise_engine_transport(
            transport,
            config
                .features
                .unwrap_or_else(|| vec!["generate".to_string()]),
            config.engine_attributes,
        );
        let bind_host = config
            .bind_address
            .rsplit_once(':')
            .map(|(host, _)| host.trim_matches(|c| c == '[' || c == ']').to_string())
            .unwrap_or_default();
        let advertised = Advertised {
            instance_id: config
                .instance_id
                .unwrap_or_else(|| format!("{}-{:016x}", config.worker_id, rand::random::<u64>())),
            hostname: config.hostname.unwrap_or(bind_host),
            worker_id: config.worker_id,
            zone: config.zone,
            engine_type: config.engine_type,
            engine_version: config.engine_version,
            engine_endpoint: config.engine_endpoint,
            model_ids: config.model_ids,
            features,
            max_concurrent_requests: config.max_concurrent_requests,
            engine_attributes,
        };
        Ok(Self {
            bind_address: config.bind_address,
            advertised,
            inference,
            engine_link,
            health,
            serving,
        })
    }
}

/// A running Worker server. `start` returns once the listener is bound; the
/// engine transport connects in the background and gates health.
pub struct WorkerNodeServer {
    address: String,
    health: Arc<Mutex<HealthSnapshot>>,
    serving: Arc<AtomicBool>,
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
    done: Mutex<Option<Receiver<()>>>,
    running: Arc<AtomicBool>,
    last_error: Arc<Mutex<Option<String>>>,
    engine_link: Option<Arc<EngineLink>>,
}

impl WorkerNodeServer {
    /// Validate, bind, and serve on a dedicated runtime thread. Blocks for at
    /// most [`BIND_TIMEOUT`].
    pub fn start(config: WorkerNodeConfig) -> Result<Self, WorkerNodeError> {
        let StartPlan {
            bind_address,
            advertised,
            inference,
            engine_link,
            health,
            serving,
        } = StartPlan::from_config(config)?;
        let service = NodeControl::new(advertised, Arc::clone(&health), engine_link.clone());

        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (done_tx, done_rx) = mpsc::sync_channel(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let running = Arc::new(AtomicBool::new(false));
        let thread_running = Arc::clone(&running);
        let last_error = Arc::new(Mutex::new(None));
        let thread_last_error = Arc::clone(&last_error);
        let thread_bind_address = bind_address.clone();
        let thread = thread::Builder::new()
            .name("smg-worker-control".to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ = started_tx.send(Err(format!("failed to create runtime: {error}")));
                        return;
                    }
                };
                let runtime_running = Arc::clone(&thread_running);
                runtime.block_on(async move {
                    let listener =
                        match tokio::net::TcpListener::bind(thread_bind_address.as_str()).await {
                            Ok(listener) => listener,
                            Err(error) => {
                                let _ = started_tx.send(Err(format!(
                                    "failed to bind {thread_bind_address}: {error}"
                                )));
                                return;
                            }
                        };
                    let address = match listener.local_addr() {
                        Ok(address) => address,
                        Err(error) => {
                            let _ = started_tx
                                .send(Err(format!("failed to read listener address: {error}")));
                            return;
                        }
                    };
                    info!(%address, "Worker control plane listening");

                    let inference = inference.map(|config| {
                        let lazy = Arc::new(LazyEngineTransport::new(Arc::clone(&config.link)));
                        let inference_service = EngineWorkerInference::from_transport(
                            Arc::clone(&lazy) as Arc<dyn EngineTransport>,
                            config.max_concurrent_requests,
                        )
                        .with_serving_flag(Arc::clone(&config.serving));
                        let connect_error = Arc::clone(&thread_last_error);
                        // Fire-and-forget: the runtime is dropped with the
                        // server thread, which cancels a still-running connect.
                        #[expect(
                            clippy::disallowed_methods,
                            reason = "engine connect is fire-and-forget; runtime drop cancels it"
                        )]
                        let _connect = tokio::spawn(async move {
                            match connect_inference(&config).await {
                                Ok(transport) => {
                                    let _ = lazy.inner.set(transport);
                                    config.link.ready.store(true, Ordering::Release);
                                    info!(
                                        engine = config.engine.name(),
                                        endpoint = %config.engine_endpoint,
                                        transport = config.transport.label(),
                                        "Worker engine transport connected"
                                    );
                                }
                                Err(error) => {
                                    let message = format!(
                                        "failed to connect the {} WorkerInference adapter to {}: {error}",
                                        config.engine.name(),
                                        config.engine_endpoint
                                    );
                                    error!(%message, "Worker engine transport failed");
                                    config.link.fail(message.clone());
                                    if let Ok(mut slot) = connect_error.lock() {
                                        *slot = Some(message);
                                    }
                                }
                            }
                        });
                        WorkerInferenceServer::new(inference_service)
                    });

                    runtime_running.store(true, Ordering::Release);
                    if started_tx.send(Ok(address)).is_err() {
                        return;
                    }
                    let incoming = TcpListenerStream::new(listener);
                    if let Err(error) = Server::builder()
                        .add_service(TonicWorkerControlServer::new(service.clone()))
                        .add_service(HealthServer::new(service))
                        .add_optional_service(inference)
                        .serve_with_incoming_shutdown(incoming, async {
                            let _ = shutdown_rx.await;
                        })
                        .await
                    {
                        error!(%error, "Worker control plane exited");
                        if let Ok(mut last_error) = thread_last_error.lock() {
                            *last_error = Some(error.to_string());
                        }
                    }
                });
                thread_running.store(false, Ordering::Release);
                let _ = done_tx.send(());
            })
            .map_err(|error| {
                WorkerNodeError::Startup(format!("failed to start server thread: {error}"))
            })?;

        let address = started_rx
            .recv_timeout(BIND_TIMEOUT)
            .map_err(|error| match error {
                RecvTimeoutError::Timeout => WorkerNodeError::Startup(format!(
                    "Worker control plane did not bind {bind_address} within {BIND_TIMEOUT:?}"
                )),
                RecvTimeoutError::Disconnected => {
                    WorkerNodeError::Startup("Worker control plane exited during startup".into())
                }
            })?
            .map_err(WorkerNodeError::Startup)?;
        Ok(Self {
            address: address.to_string(),
            health,
            serving,
            shutdown: Mutex::new(Some(shutdown_tx)),
            thread: Mutex::new(Some(thread)),
            done: Mutex::new(Some(done_rx)),
            running,
            last_error,
            engine_link,
        })
    }

    /// The bound `ip:port`.
    pub fn address(&self) -> &str {
        &self.address
    }

    pub fn running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Whether the engine transport has connected; health reports STARTING
    /// until it has, whatever lifecycle was announced.
    pub fn engine_ready(&self) -> bool {
        self.engine_link.as_ref().is_none_or(|link| link.is_ready())
    }

    pub fn last_error(&self) -> Result<Option<String>, WorkerNodeError> {
        Ok(lock(&self.last_error)?.clone())
    }

    pub fn set_health(
        &self,
        state: WorkerHealthState,
        message: String,
    ) -> Result<(), WorkerNodeError> {
        self.serving
            .store(state == WorkerHealthState::Serving, Ordering::Release);
        *lock(&self.health)? = HealthSnapshot { state, message };
        Ok(())
    }

    /// Signal shutdown and wait up to `timeout` for the server thread. On
    /// timeout the server keeps stopping and a later call waits again.
    pub fn stop(&self, timeout: Duration) -> Result<(), WorkerNodeError> {
        if let Some(shutdown) = lock(&self.shutdown)?.take() {
            let _ = shutdown.send(());
        }
        let receiver = lock(&self.done)?.take();
        if let Some(receiver) = receiver {
            match receiver.recv_timeout(timeout) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => {}
                Err(RecvTimeoutError::Timeout) => {
                    *lock(&self.done)? = Some(receiver);
                    return Err(WorkerNodeError::StopTimeout);
                }
            }
        }
        if let Some(thread) = lock(&self.thread)?.take() {
            thread.join().map_err(|_| WorkerNodeError::ThreadPanicked)?;
        }
        Ok(())
    }
}

impl Drop for WorkerNodeServer {
    fn drop(&mut self) {
        if let Ok(shutdown) = self.shutdown.get_mut() {
            if let Some(shutdown) = shutdown.take() {
                let _ = shutdown.send(());
            }
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, WorkerNodeError> {
    mutex.lock().map_err(|_| WorkerNodeError::Poisoned)
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use tonic::transport::Channel;
    use tonic_health::pb::health_client::HealthClient;

    use super::*;

    fn announced(state: WorkerHealthState) -> HealthSnapshot {
        HealthSnapshot {
            state,
            message: "announced".to_string(),
        }
    }

    fn config(inference_enabled: bool) -> WorkerNodeConfig {
        WorkerNodeConfig {
            bind_address: "127.0.0.1:0".to_string(),
            worker_id: "worker-a".to_string(),
            instance_id: Some("instance-a".to_string()),
            hostname: None,
            zone: String::new(),
            engine_type: "vllm".to_string(),
            engine_version: String::new(),
            engine_endpoint: "grpc://127.0.0.1:1".to_string(),
            model_ids: vec!["model-a".to_string()],
            features: None,
            max_concurrent_requests: 4,
            inference_enabled,
            engine_attributes: HashMap::new(),
            engine_transport: "grpc".to_string(),
            zmq_handshake_address: None,
            engine_count: 1,
        }
    }

    fn advertised() -> Advertised {
        Advertised {
            worker_id: "worker-a".to_string(),
            instance_id: "instance-a".to_string(),
            hostname: "node-a".to_string(),
            zone: String::new(),
            engine_type: "vllm".to_string(),
            engine_version: String::new(),
            engine_endpoint: "grpc://worker-a:32000".to_string(),
            model_ids: vec!["model-a".to_string()],
            features: vec!["generate".to_string()],
            max_concurrent_requests: 32,
            engine_attributes: HashMap::new(),
        }
    }

    #[test]
    fn health_stays_starting_until_the_engine_transport_connects() {
        let link = EngineLink::default();
        let health = effective_health(announced(WorkerHealthState::Serving), Some(&link));
        assert_eq!(health.state, WorkerHealthState::Starting);

        link.ready.store(true, Ordering::Release);
        let health = effective_health(announced(WorkerHealthState::Serving), Some(&link));
        assert_eq!(health.state, WorkerHealthState::Serving);
        assert_eq!(health.message, "announced");
    }

    #[test]
    fn engine_transport_failure_reports_not_serving_whatever_was_announced() {
        let link = EngineLink::default();
        link.fail("handshake timed out".to_string());
        for state in [
            WorkerHealthState::Starting,
            WorkerHealthState::Serving,
            WorkerHealthState::Draining,
        ] {
            let health = effective_health(announced(state), Some(&link));
            assert_eq!(health.state, WorkerHealthState::NotServing);
            assert!(health.message.contains("handshake timed out"));
        }
    }

    #[test]
    fn lifecycle_states_other_than_serving_pass_through_while_connecting() {
        let link = EngineLink::default();
        for state in [
            WorkerHealthState::Starting,
            WorkerHealthState::Draining,
            WorkerHealthState::NotServing,
        ] {
            assert_eq!(effective_health(announced(state), Some(&link)).state, state);
        }
        assert_eq!(
            effective_health(announced(WorkerHealthState::Serving), None).state,
            WorkerHealthState::Serving
        );
    }

    #[tokio::test]
    async fn lazy_transport_names_the_connect_failure_once_it_happened() {
        let link = Arc::new(EngineLink::default());
        let lazy = LazyEngineTransport::new(Arc::clone(&link));
        let request = worker_inference_proto::AbortRequest::default();

        let status = lazy.abort(request.clone()).await.unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert!(status.message().contains("still connecting"));

        link.fail("dial refused".to_string());
        let status = lazy.abort(request).await.unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert!(status.message().contains("dial refused"));
    }

    #[test]
    fn zmq_transport_advertises_token_only_wire_without_the_caller_asking() {
        let (features, attributes) = advertise_engine_transport(
            WorkerEngineTransport::Zmq,
            vec!["generate".to_string()],
            HashMap::new(),
        );
        assert!(features.iter().any(|f| f == TOKEN_ONLY_WIRE_FEATURE));
        assert_eq!(
            attributes.get("engine_transport").map(String::as_str),
            Some("zmq")
        );
    }

    #[test]
    fn grpc_transport_does_not_advertise_token_only_wire() {
        let (features, attributes) = advertise_engine_transport(
            WorkerEngineTransport::Grpc,
            vec!["generate".to_string()],
            HashMap::new(),
        );
        assert_eq!(features, vec!["generate".to_string()]);
        assert_eq!(
            attributes.get("engine_transport").map(String::as_str),
            Some("grpc")
        );
    }

    #[test]
    fn an_explicit_transport_attribute_is_left_alone() {
        let (features, attributes) = advertise_engine_transport(
            WorkerEngineTransport::Zmq,
            vec!["generate".to_string(), TOKEN_ONLY_WIRE_FEATURE.to_string()],
            HashMap::from([("engine_transport".to_string(), "zmq".to_string())]),
        );
        assert_eq!(
            features
                .iter()
                .filter(|f| *f == TOKEN_ONLY_WIRE_FEATURE)
                .count(),
            1
        );
        assert_eq!(
            attributes.get("engine_transport").map(String::as_str),
            Some("zmq")
        );
    }

    #[test]
    fn parses_supported_health_states() {
        assert_eq!(
            parse_health_state("serving").unwrap(),
            WorkerHealthState::Serving
        );
        assert_eq!(
            parse_health_state("not_serving").unwrap(),
            WorkerHealthState::NotServing
        );
        assert!(parse_health_state("unknown").is_err());
    }

    #[test]
    fn inference_requires_an_engine_the_worker_can_front() {
        let mut sglang = config(true);
        sglang.engine_type = "sglang".to_string();
        let error = StartPlan::from_config(sglang)
            .err()
            .expect("an engine the Worker cannot front is rejected");
        assert!(matches!(error, WorkerNodeError::InvalidConfig(_)));
        assert!(error.to_string().contains("vllm or tokenspeed"));

        // Control plane only: the engine type is identity, not a transport.
        let mut control_only = config(false);
        control_only.engine_type = "sglang".to_string();
        assert!(StartPlan::from_config(control_only).is_ok());

        let mut ts = config(true);
        ts.engine_type = "TS".to_string();
        let plan = StartPlan::from_config(ts).unwrap();
        assert_eq!(plan.inference.unwrap().engine, WorkerEngine::TokenSpeed);
    }

    #[test]
    fn start_plan_rejects_malformed_config() {
        type Mutation = fn(&mut WorkerNodeConfig);
        let cases: [(&str, Mutation); 7] = [
            ("worker_id", |c| {
                c.worker_id = " ".to_string();
            }),
            ("engine_type", |c| {
                c.engine_type = String::new();
            }),
            ("engine_endpoint", |c| {
                c.engine_endpoint = String::new();
            }),
            ("model_ids", |c| {
                c.model_ids.clear();
            }),
            ("bind_address", |c| {
                c.bind_address = "no-port".to_string();
            }),
            ("engine_count", |c| {
                c.engine_count = 0;
            }),
            ("engine transport", |c| {
                c.engine_transport = "ipc".to_string();
            }),
        ];
        for (field, mutate) in cases {
            let mut config = config(true);
            mutate(&mut config);
            let Some(error) = StartPlan::from_config(config).err() else {
                panic!("{field}: malformed config was accepted");
            };
            assert!(
                matches!(error, WorkerNodeError::InvalidConfig(_)),
                "{field}: {error}"
            );
            assert!(error.to_string().contains(field), "{field}: {error}");
        }
    }

    #[test]
    fn hostname_defaults_to_the_bind_host() {
        let mut named = config(false);
        named.bind_address = "[::1]:0".to_string();
        assert_eq!(
            StartPlan::from_config(named).unwrap().advertised.hostname,
            "::1"
        );
        let mut explicit = config(false);
        explicit.hostname = Some("node-a".to_string());
        assert_eq!(
            StartPlan::from_config(explicit)
                .unwrap()
                .advertised
                .hostname,
            "node-a"
        );
    }

    #[tokio::test]
    async fn lifecycle_health_is_served_from_rust_state() {
        let health = Arc::new(Mutex::new(HealthSnapshot {
            state: WorkerHealthState::Starting,
            message: "warming up".to_string(),
        }));
        let control = NodeControl::new(advertised(), Arc::clone(&health), None);

        let starting = control
            .get_health(Request::new(proto::GetHealthRequest {
                include_components: true,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(starting.state(), WorkerHealthState::Starting);
        assert_eq!(starting.components.len(), 1);

        *health.lock().unwrap() = HealthSnapshot {
            state: WorkerHealthState::Serving,
            message: "ready".to_string(),
        };
        let serving = control
            .get_health(Request::new(proto::GetHealthRequest {
                include_components: false,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(serving.state(), WorkerHealthState::Serving);
        assert!(serving.components.is_empty());
    }

    #[tokio::test]
    async fn standard_health_answers_per_service_name() {
        let health = Arc::new(Mutex::new(announced(WorkerHealthState::Serving)));
        let control = NodeControl::new(advertised(), Arc::clone(&health), None);
        let check = |service: &str| {
            Request::new(HealthCheckRequest {
                service: service.to_string(),
            })
        };

        let overall = control.check(check("")).await.unwrap().into_inner();
        assert_eq!(overall.status, ServingStatus::Serving as i32);
        let named = control
            .check(check(
                <TonicWorkerControlServer<NodeControl> as NamedService>::NAME,
            ))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(named.status, ServingStatus::Serving as i32);
        // Inference is not registered on a control-plane-only Worker.
        let inference = control
            .check(check(
                <WorkerInferenceServer<EngineWorkerInference> as NamedService>::NAME,
            ))
            .await
            .unwrap_err();
        assert_eq!(inference.code(), tonic::Code::NotFound);
        let foreign = control.check(check("acme.Nothing")).await.unwrap_err();
        assert_eq!(foreign.code(), tonic::Code::NotFound);

        *health.lock().unwrap() = announced(WorkerHealthState::Draining);
        let overall = control.check(check("")).await.unwrap().into_inner();
        assert_eq!(overall.status, ServingStatus::NotServing as i32);
    }

    #[tokio::test]
    async fn standard_health_watch_emits_on_change() {
        let health = Arc::new(Mutex::new(announced(WorkerHealthState::Starting)));
        let control = NodeControl::new(advertised(), Arc::clone(&health), None);
        let mut watch = control
            .watch(Request::new(HealthCheckRequest::default()))
            .await
            .unwrap()
            .into_inner();

        let first = watch.next().await.unwrap().unwrap();
        assert_eq!(first.status, ServingStatus::NotServing as i32);
        *health.lock().unwrap() = announced(WorkerHealthState::Serving);
        let second = tokio::time::timeout(Duration::from_secs(5), watch.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(second.status, ServingStatus::Serving as i32);
    }

    #[test]
    fn server_binds_by_host_name_and_stops() {
        let mut config = config(false);
        config.bind_address = "localhost:0".to_string();
        let server = WorkerNodeServer::start(config).unwrap();
        assert!(server.running());
        assert!(server.engine_ready());
        let port = server.address().rsplit(':').next().unwrap();
        assert_ne!(port, "0");

        server
            .set_health(WorkerHealthState::Serving, "ready".to_string())
            .unwrap();
        let address = server.address().to_string();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let status = runtime.block_on(async {
            let channel = Channel::from_shared(format!("http://{address}"))
                .unwrap()
                .connect()
                .await
                .unwrap();
            let mut client = HealthClient::new(channel);
            client
                .check(HealthCheckRequest::default())
                .await
                .unwrap()
                .into_inner()
                .status
        });
        assert_eq!(status, ServingStatus::Serving as i32);

        server.stop(Duration::from_secs(5)).unwrap();
        assert!(!server.running());
        assert_eq!(server.last_error().unwrap(), None);
    }
}
