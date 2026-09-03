//! Python lifecycle binding for the Rust WorkerControl gRPC server.
//!
//! Rust owns transport and serves discovery and health from in-memory state.
//! Python only drives coarse lifecycle transitions; request-time health polls
//! never cross the GIL.

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError},
        Arc, Mutex, OnceLock,
    },
    thread,
    time::{Duration, SystemTime},
};

use pyo3::{
    exceptions::{PyRuntimeError, PyTimeoutError, PyValueError},
    prelude::*,
};
use smg::worker::{RuntimeType, ZmqWorkerTransport, TOKEN_ONLY_WIRE_FEATURE};
use smg_grpc_client::{
    worker_inference::{connect_engine_transport, EngineTransport, EngineWorkerInference},
    worker_inference_proto::{self, worker_inference_server::WorkerInferenceServer},
    worker_proto::{
        self as proto,
        worker_control_server::{WorkerControl, WorkerControlServer as TonicWorkerControlServer},
    },
};
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};
use tonic_health::pb::{
    health_check_response::ServingStatus,
    health_server::{Health, HealthServer},
    HealthCheckRequest, HealthCheckResponse,
};

/// How long the server thread may take to bind its listener. Engine
/// connection no longer happens inside this window; see [`EngineLink`].
const BIND_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone)]
struct HealthSnapshot {
    state: proto::WorkerHealthState,
    message: String,
}

/// Rust-owned readiness of the engine transport, independent of the
/// lifecycle Python announces. The control listener binds first so that
/// STARTING is observable over gRPC while the engine handshake (a model load,
/// for ZMQ) completes in the background; GetHealth reports SERVING only once
/// both Python has announced it *and* the transport is connected.
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
}

/// Engine transport that requests can be routed to before it exists. Until
/// the background connect installs the real transport, every call is refused
/// with UNAVAILABLE rather than hanging or panicking.
#[derive(Default)]
struct LazyEngineTransport {
    inner: OnceLock<Arc<dyn EngineTransport>>,
}

impl LazyEngineTransport {
    fn connected(&self) -> Result<&Arc<dyn EngineTransport>, Status> {
        self.inner
            .get()
            .ok_or_else(|| Status::unavailable("Worker engine transport is still connecting"))
    }
}

#[tonic::async_trait]
impl EngineTransport for LazyEngineTransport {
    async fn generate(
        &self,
        request: worker_inference_proto::GenerateRequest,
    ) -> Result<smg_grpc_client::worker_inference::EngineTransportStream, Status> {
        self.connected()?.generate(request).await
    }

    async fn abort(
        &self,
        request: worker_inference_proto::AbortRequest,
    ) -> Result<worker_inference_proto::AbortResponse, Status> {
        self.connected()?.abort(request).await
    }
}

struct BridgeState {
    identity: proto::WorkerIdentity,
    capabilities: proto::WorkerCapabilities,
    topology: proto::WorkerTopology,
    health: Arc<Mutex<HealthSnapshot>>,
    engine_link: Option<Arc<EngineLink>>,
}

/// The health this Worker actually reports: Python's announced lifecycle,
/// gated on the engine transport when there is one.
fn effective_health(announced: HealthSnapshot, engine_link: Option<&EngineLink>) -> HealthSnapshot {
    let Some(link) = engine_link else {
        return announced;
    };
    if let Some(error) = link.error() {
        return HealthSnapshot {
            state: proto::WorkerHealthState::NotServing,
            message: format!("engine transport failed: {error}"),
        };
    }
    if !link.is_ready()
        && matches!(
            announced.state,
            proto::WorkerHealthState::Serving | proto::WorkerHealthState::Degraded
        )
    {
        return HealthSnapshot {
            state: proto::WorkerHealthState::Starting,
            message: "waiting for the engine transport to connect".to_string(),
        };
    }
    announced
}

#[derive(Clone)]
struct PythonWorkerControl {
    state: Arc<BridgeState>,
}

impl PythonWorkerControl {
    fn new(config: BridgeConfig) -> Self {
        let engine = proto::EngineCapability {
            engine_type: config.engine_type.clone(),
            engine_version: config.engine_version,
            model_ids: config.model_ids.clone(),
            features: config.features.clone(),
        };
        let engine_attributes = config.engine_attributes.clone();
        Self {
            state: Arc::new(BridgeState {
                identity: proto::WorkerIdentity {
                    worker_id: config.worker_id.clone(),
                    instance_id: config.instance_id,
                    hostname: config.hostname,
                    zone: config.zone,
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    started_at: Some(now()),
                    labels: [("role".to_string(), "smg-worker".to_string())].into(),
                },
                capabilities: proto::WorkerCapabilities {
                    api_major: 1,
                    api_minor: 0,
                    features: config.features,
                    engines: vec![engine],
                    max_concurrent_requests: config.max_concurrent_requests,
                    attributes: engine_attributes.clone(),
                },
                topology: proto::WorkerTopology {
                    worker_id: config.worker_id,
                    topology_version: 1,
                    engines: vec![proto::EngineEndpoint {
                        engine_id: "python-engine-0".to_string(),
                        engine_type: config.engine_type,
                        endpoint: config.engine_endpoint,
                        model_ids: config.model_ids,
                        replica_group: String::new(),
                        data_parallel_rank: None,
                        tensor_parallel_rank: None,
                        pipeline_parallel_rank: None,
                        attributes: engine_attributes,
                    }],
                    observed_at: Some(now()),
                },
                health: config.health,
                engine_link: config.engine_link,
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
}

/// Standard `grpc.health.v1` view of the same state, so generic probes
/// (`smg serve`, Kubernetes, grpcurl) can wait for readiness without speaking
/// WorkerControl. SERVING iff [`PythonWorkerControl::health_snapshot`] is.
#[tonic::async_trait]
impl Health for PythonWorkerControl {
    type WatchStream = tokio_stream::Once<Result<HealthCheckResponse, Status>>;

    async fn check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let status = if self.health_snapshot()?.state == proto::WorkerHealthState::Serving {
            ServingStatus::Serving
        } else {
            ServingStatus::NotServing
        };
        Ok(Response::new(HealthCheckResponse {
            status: status as i32,
        }))
    }

    async fn watch(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        // One snapshot, then end: callers that need a live feed poll Check.
        let response = self.check(request).await?.into_inner();
        Ok(Response::new(tokio_stream::once(Ok(response))))
    }
}

#[tonic::async_trait]
impl WorkerControl for PythonWorkerControl {
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
                component_id: "python-engine-0".to_string(),
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

fn parse_health_state(state: &str) -> PyResult<proto::WorkerHealthState> {
    match state.to_ascii_lowercase().as_str() {
        "starting" => Ok(proto::WorkerHealthState::Starting),
        "serving" => Ok(proto::WorkerHealthState::Serving),
        "degraded" => Ok(proto::WorkerHealthState::Degraded),
        "draining" => Ok(proto::WorkerHealthState::Draining),
        "not_serving" | "not-serving" => Ok(proto::WorkerHealthState::NotServing),
        _ => Err(PyValueError::new_err(format!(
            "unknown worker health state {state:?}"
        ))),
    }
}

struct BridgeConfig {
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
    health: Arc<Mutex<HealthSnapshot>>,
    engine_link: Option<Arc<EngineLink>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkerEngineTransport {
    Grpc,
    Zmq,
}

fn parse_engine_transport(value: &str) -> PyResult<WorkerEngineTransport> {
    match value.to_ascii_lowercase().as_str() {
        "grpc" => Ok(WorkerEngineTransport::Grpc),
        "zmq" => Ok(WorkerEngineTransport::Zmq),
        _ => Err(PyValueError::new_err(format!(
            "unknown Worker engine transport {value:?}; expected grpc or zmq"
        ))),
    }
}

/// Label the Router reads back off the registration to decide whether the
/// Worker's wire is token-only. Must round-trip through [`parse_engine_transport`].
fn engine_transport_label(transport: WorkerEngineTransport) -> &'static str {
    match transport {
        WorkerEngineTransport::Grpc => "grpc",
        WorkerEngineTransport::Zmq => "zmq",
    }
}

/// Fold the engine transport into what the Worker advertises.
///
/// `token_only_wire` and the `engine_transport` attribute follow the transport,
/// not the caller: a ZMQ engine cannot match string stops, and the Router needs
/// that fact to keep stop trimming and the EOS backstop on its own side.
/// Deriving both at this boundary means every entry point -- `worker_sidecar.py`
/// and `worker_control_lifecycle.py` alike -- advertises them, instead of each
/// one having to remember.
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
        .or_insert_with(|| engine_transport_label(transport).to_string());
    (features, engine_attributes)
}

fn parse_zmq_runtime(engine_type: &str) -> PyResult<RuntimeType> {
    match engine_type.to_ascii_lowercase().as_str() {
        "vllm" => Ok(RuntimeType::Vllm),
        "tokenspeed" | "ts" => Ok(RuntimeType::TokenSpeed),
        other => Err(PyValueError::new_err(format!(
            "Worker ZMQ transport is not implemented for {other}"
        ))),
    }
}

/// Rust-owned WorkerControl server with lifecycle driven from Python.
#[pyclass(name = "WorkerControlServer")]
pub struct PyWorkerControlServer {
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

#[pymethods]
impl PyWorkerControlServer {
    #[new]
    #[pyo3(signature = (
        bind_address,
        worker_id,
        engine_type,
        model_ids,
        engine_endpoint,
        instance_id = None,
        hostname = None,
        zone = String::new(),
        engine_version = String::new(),
        features = None,
        max_concurrent_requests = 0,
        inference_enabled = false,
        engine_attributes = None,
        engine_transport = String::from("grpc"),
        zmq_handshake_address = None,
        engine_count = 1,
    ))]
    #[expect(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        bind_address: String,
        worker_id: String,
        engine_type: String,
        model_ids: Vec<String>,
        engine_endpoint: String,
        instance_id: Option<String>,
        hostname: Option<String>,
        zone: String,
        engine_version: String,
        features: Option<Vec<String>>,
        max_concurrent_requests: u32,
        inference_enabled: bool,
        engine_attributes: Option<HashMap<String, String>>,
        engine_transport: String,
        zmq_handshake_address: Option<String>,
        engine_count: usize,
    ) -> PyResult<Self> {
        if worker_id.trim().is_empty() {
            return Err(PyValueError::new_err("worker_id must not be empty"));
        }
        if engine_type.trim().is_empty() {
            return Err(PyValueError::new_err("engine_type must not be empty"));
        }
        if engine_endpoint.trim().is_empty() {
            return Err(PyValueError::new_err("engine_endpoint must not be empty"));
        }
        if model_ids.is_empty() {
            return Err(PyValueError::new_err("model_ids must not be empty"));
        }
        let bind_address: SocketAddr = bind_address
            .parse()
            .map_err(|error| PyValueError::new_err(format!("invalid bind_address: {error}")))?;
        let health = Arc::new(Mutex::new(HealthSnapshot {
            state: proto::WorkerHealthState::Starting,
            message: "starting".to_string(),
        }));
        let serving = Arc::new(AtomicBool::new(false));
        let engine_transport = parse_engine_transport(&engine_transport)?;
        let engine_link = inference_enabled.then(|| Arc::new(EngineLink::default()));
        if engine_count == 0 {
            return Err(PyValueError::new_err("engine_count must be positive"));
        }
        let zmq_runtime = (engine_transport == WorkerEngineTransport::Zmq)
            .then(|| parse_zmq_runtime(&engine_type))
            .transpose()?;
        let inference = inference_enabled.then(|| InferenceConfig {
            engine_type: engine_type.clone(),
            engine_endpoint: engine_endpoint.clone(),
            model_id: model_ids[0].clone(),
            max_concurrent_requests,
            serving: Arc::clone(&serving),
            engine_transport,
            zmq_runtime,
            zmq_handshake_address,
            engine_count,
        });
        let (features, engine_attributes) = advertise_engine_transport(
            engine_transport,
            features.unwrap_or_else(|| vec!["generate".to_string()]),
            engine_attributes.unwrap_or_default(),
        );

        let config = BridgeConfig {
            worker_id: worker_id.clone(),
            instance_id: instance_id
                .unwrap_or_else(|| format!("{worker_id}-{:016x}", rand::random::<u64>())),
            hostname: hostname.unwrap_or_else(|| bind_address.ip().to_string()),
            zone,
            engine_type,
            engine_version,
            engine_endpoint,
            model_ids,
            features,
            max_concurrent_requests,
            engine_attributes,
            health: Arc::clone(&health),
            engine_link: engine_link.clone(),
        };
        py.detach(|| {
            start_server(
                bind_address,
                PythonWorkerControl::new(config),
                health,
                serving,
                inference,
                engine_link,
            )
        })
    }

    /// Whether the engine transport has connected. Health reports STARTING
    /// until it has, whatever lifecycle Python announced.
    #[getter]
    fn engine_ready(&self) -> bool {
        self.engine_link.as_ref().is_none_or(|link| link.is_ready())
    }

    #[getter]
    fn address(&self) -> &str {
        &self.address
    }

    #[getter]
    fn running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    #[getter]
    fn last_error(&self) -> PyResult<Option<String>> {
        Ok(lock(&self.last_error)?.clone())
    }

    #[pyo3(signature = (state, message = String::new()))]
    fn set_health(&self, state: &str, message: String) -> PyResult<()> {
        let state = parse_health_state(state)?;
        self.serving.store(
            state == proto::WorkerHealthState::Serving,
            Ordering::Release,
        );
        *lock(&self.health)? = HealthSnapshot { state, message };
        Ok(())
    }

    #[pyo3(signature = (timeout_secs = 5.0))]
    fn stop(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<()> {
        if !timeout_secs.is_finite() || timeout_secs <= 0.0 {
            return Err(PyValueError::new_err(
                "timeout_secs must be finite and positive",
            ));
        }
        if let Some(shutdown) = lock(&self.shutdown)?.take() {
            let _ = shutdown.send(());
        }

        let receiver = lock(&self.done)?.take();
        if let Some(receiver) = receiver {
            let timeout = Duration::from_secs_f64(timeout_secs);
            let (result, receiver) = py.detach(move || {
                let result = receiver.recv_timeout(timeout);
                (result, receiver)
            });
            match result {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => {}
                Err(RecvTimeoutError::Timeout) => {
                    *lock(&self.done)? = Some(receiver);
                    return Err(PyTimeoutError::new_err(
                        "WorkerControl server did not stop before timeout",
                    ));
                }
            }
        }
        if let Some(thread) = lock(&self.thread)?.take() {
            thread
                .join()
                .map_err(|_| PyRuntimeError::new_err("WorkerControl server thread panicked"))?;
        }
        Ok(())
    }
}

impl Drop for PyWorkerControlServer {
    fn drop(&mut self) {
        if let Ok(shutdown) = self.shutdown.get_mut() {
            if let Some(shutdown) = shutdown.take() {
                let _ = shutdown.send(());
            }
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> PyResult<std::sync::MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| PyRuntimeError::new_err("WorkerControl server state is poisoned"))
}

fn start_server(
    bind_address: SocketAddr,
    service: PythonWorkerControl,
    health: Arc<Mutex<HealthSnapshot>>,
    serving: Arc<AtomicBool>,
    inference: Option<InferenceConfig>,
    engine_link: Option<Arc<EngineLink>>,
) -> PyResult<PyWorkerControlServer> {
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (done_tx, done_rx) = mpsc::sync_channel(1);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let running = Arc::new(AtomicBool::new(false));
    let thread_running = Arc::clone(&running);
    let last_error = Arc::new(Mutex::new(None));
    let thread_last_error = Arc::clone(&last_error);
    let thread_link = engine_link.clone();
    let thread = thread::Builder::new()
        .name("smg-python-worker-control".to_string())
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
                // Bind before touching the engine: the control plane must be
                // reachable -- and report STARTING -- for the whole time the
                // engine handshake takes, which for ZMQ is the model load.
                let listener = match tokio::net::TcpListener::bind(bind_address).await {
                    Ok(listener) => listener,
                    Err(error) => {
                        let _ =
                            started_tx.send(Err(format!("failed to bind {bind_address}: {error}")));
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

                // The inference service is registered now, bound to a transport
                // that refuses requests with UNAVAILABLE until the background
                // connect installs the real one.
                let inference = inference.map(|config| {
                    let lazy = Arc::new(LazyEngineTransport::default());
                    let inference_service = EngineWorkerInference::from_transport(
                        Arc::clone(&lazy) as Arc<dyn EngineTransport>,
                        config.max_concurrent_requests,
                    )
                    .with_serving_flag(Arc::clone(&config.serving));
                    let link = thread_link
                        .clone()
                        .unwrap_or_else(|| Arc::new(EngineLink::default()));
                    let connect_error = Arc::clone(&thread_last_error);
                    // Nothing waits on this at shutdown by design: the runtime
                    // is dropped with the server thread, which cancels a
                    // still-running connect, and a connect that lands after
                    // shutdown only fills state nobody reads.
                    #[expect(
                        clippy::disallowed_methods,
                        reason = "engine connect is fire-and-forget; runtime drop cancels it"
                    )]
                    let _connect = tokio::spawn(async move {
                        match connect_inference(&config).await {
                            Ok(transport) => {
                                // Only this task ever sets the cell.
                                let _ = lazy.inner.set(transport);
                                link.ready.store(true, Ordering::Release);
                            }
                            Err(error) => {
                                let message = format!(
                                    "failed to connect {} WorkerInference adapter to {}: {error}",
                                    config.engine_type, config.engine_endpoint
                                );
                                if let Ok(mut slot) = link.error.lock() {
                                    *slot = Some(message.clone());
                                }
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
                    if let Ok(mut last_error) = thread_last_error.lock() {
                        *last_error = Some(error.to_string());
                    }
                }
            });
            thread_running.store(false, Ordering::Release);
            let _ = done_tx.send(());
        })
        .map_err(|error| {
            PyRuntimeError::new_err(format!("failed to start server thread: {error}"))
        })?;

    let address = started_rx
        .recv_timeout(BIND_TIMEOUT)
        .map_err(|error| match error {
            RecvTimeoutError::Timeout => PyRuntimeError::new_err(format!(
                "WorkerControl server did not bind {bind_address} within {BIND_TIMEOUT:?}"
            )),
            RecvTimeoutError::Disconnected => {
                PyRuntimeError::new_err("WorkerControl server exited during startup")
            }
        })?
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyWorkerControlServer {
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

struct InferenceConfig {
    engine_type: String,
    engine_endpoint: String,
    model_id: String,
    max_concurrent_requests: u32,
    serving: Arc<AtomicBool>,
    engine_transport: WorkerEngineTransport,
    zmq_runtime: Option<RuntimeType>,
    zmq_handshake_address: Option<String>,
    engine_count: usize,
}

async fn connect_inference(
    config: &InferenceConfig,
) -> Result<Arc<dyn EngineTransport>, Box<dyn std::error::Error + Send + Sync>> {
    match config.engine_transport {
        WorkerEngineTransport::Grpc => {
            connect_engine_transport(&config.engine_type, &config.engine_endpoint).await
        }
        WorkerEngineTransport::Zmq => {
            let runtime = config
                .zmq_runtime
                .ok_or_else(|| std::io::Error::other("missing Worker ZMQ runtime"))?;
            let transport = ZmqWorkerTransport::connect(
                &config.engine_endpoint,
                config.model_id.clone(),
                runtime,
                config.zmq_handshake_address.as_deref(),
                config.engine_count,
            )
            .await
            .map_err(std::io::Error::other)?;
            Ok(Arc::new(transport) as Arc<dyn EngineTransport>)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn announced(state: proto::WorkerHealthState) -> HealthSnapshot {
        HealthSnapshot {
            state,
            message: "announced".to_string(),
        }
    }

    #[test]
    fn health_stays_starting_until_the_engine_transport_connects() {
        // Python announces SERVING as soon as the constructor returns; the
        // listener is up by then but the engine handshake may still be running.
        let link = EngineLink::default();
        let health = effective_health(announced(proto::WorkerHealthState::Serving), Some(&link));
        assert_eq!(health.state, proto::WorkerHealthState::Starting);

        link.ready.store(true, Ordering::Release);
        let health = effective_health(announced(proto::WorkerHealthState::Serving), Some(&link));
        assert_eq!(health.state, proto::WorkerHealthState::Serving);
        assert_eq!(health.message, "announced");
    }

    #[test]
    fn engine_transport_failure_reports_not_serving_whatever_python_announced() {
        let link = EngineLink::default();
        *link.error.lock().unwrap() = Some("handshake timed out".to_string());
        for state in [
            proto::WorkerHealthState::Starting,
            proto::WorkerHealthState::Serving,
            proto::WorkerHealthState::Draining,
        ] {
            let health = effective_health(announced(state), Some(&link));
            assert_eq!(health.state, proto::WorkerHealthState::NotServing);
            assert!(health.message.contains("handshake timed out"));
        }
    }

    #[test]
    fn lifecycle_states_other_than_serving_pass_through_while_connecting() {
        // Draining or NotServing announced during the connect window are the
        // truth already; only an optimistic SERVING gets held back.
        let link = EngineLink::default();
        for state in [
            proto::WorkerHealthState::Starting,
            proto::WorkerHealthState::Draining,
            proto::WorkerHealthState::NotServing,
        ] {
            assert_eq!(effective_health(announced(state), Some(&link)).state, state);
        }
        // No inference service at all: Python's word is final.
        assert_eq!(
            effective_health(announced(proto::WorkerHealthState::Serving), None).state,
            proto::WorkerHealthState::Serving
        );
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
        // A caller that already set the attribute -- the sidecar did, before
        // this moved to the boundary -- must not have it silently rewritten.
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
            proto::WorkerHealthState::Serving
        );
        assert_eq!(
            parse_health_state("not_serving").unwrap(),
            proto::WorkerHealthState::NotServing
        );
        assert!(parse_health_state("unknown").is_err());
    }

    #[tokio::test]
    async fn lifecycle_health_is_served_from_rust_state() {
        let health = Arc::new(Mutex::new(HealthSnapshot {
            state: proto::WorkerHealthState::Starting,
            message: "warming up".to_string(),
        }));
        let control = PythonWorkerControl::new(BridgeConfig {
            worker_id: "worker-a".to_string(),
            instance_id: "instance-a".to_string(),
            hostname: "node-a".to_string(),
            zone: String::new(),
            engine_type: "sglang".to_string(),
            engine_version: String::new(),
            engine_endpoint: "grpc://worker-a:32000".to_string(),
            model_ids: vec!["model-a".to_string()],
            features: vec!["generate".to_string()],
            max_concurrent_requests: 32,
            engine_attributes: HashMap::new(),
            health: Arc::clone(&health),
            engine_link: None,
        });

        let starting = control
            .get_health(Request::new(proto::GetHealthRequest {
                include_components: true,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(starting.state(), proto::WorkerHealthState::Starting);
        assert_eq!(starting.components.len(), 1);

        *health.lock().unwrap() = HealthSnapshot {
            state: proto::WorkerHealthState::Serving,
            message: "ready".to_string(),
        };
        let serving = control
            .get_health(Request::new(proto::GetHealthRequest {
                include_components: false,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(serving.state(), proto::WorkerHealthState::Serving);
        assert!(serving.components.is_empty());
    }
}
