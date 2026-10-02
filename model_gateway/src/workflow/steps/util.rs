//! Shared URL normalization and network probe utilities for worker steps.

use std::{collections::BTreeSet, fmt, future::Future, time::Duration};

use futures::stream::{FuturesUnordered, StreamExt};
use reqwest::Client;
use smg_grpc_client::{
    connect_channel_with_timeout,
    worker_control::WORKER_CONTROL_API_MAJOR,
    worker_proto::{
        worker_control_client::WorkerControlClient, GetCapabilitiesRequest, GetHealthRequest,
        GetIdentityRequest, GetTopologyRequest, WorkerHealthState,
    },
};
use wfaas::{StepId, WorkflowError};

use crate::{
    routers::grpc::client::GrpcClient,
    worker::{
        worker::{smg_worker_engine_types, SMG_WORKER_ENGINE_TYPES},
        RuntimeType,
    },
    workflow::data::{SmgEngineDiscovery, SmgWorkerDiscovery},
};

fn strip_scheme<'a>(url: &'a str, scheme: &str) -> Option<&'a str> {
    url.get(..scheme.len())
        .filter(|prefix| prefix.eq_ignore_ascii_case(scheme))
        .map(|_| &url[scheme.len()..])
}

fn url_scheme(url: &str) -> Option<String> {
    url.split_once("://")
        .map(|(scheme, _)| scheme.to_ascii_lowercase())
}

/// Strip protocol prefix (http://, https://, grpc://, grpcs://) from URL.
pub(crate) fn strip_protocol(url: &str) -> String {
    for scheme in ["http://", "https://", "grpc://", "grpcs://"] {
        if let Some(rest) = strip_scheme(url, scheme) {
            return rest.to_string();
        }
    }
    url.to_string()
}

/// Ensure URL has an HTTP(S) scheme — handles bare `host:port` and gRPC inputs.
pub(crate) fn http_base_url(url: &str) -> String {
    if strip_scheme(url, "http://").is_some() || strip_scheme(url, "https://").is_some() {
        url.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", strip_protocol(url).trim_end_matches('/'))
    }
}

/// Ensure URL has a gRPC scheme — handles bare `host:port` and HTTP(S) inputs.
pub(crate) fn grpc_base_url(url: &str) -> String {
    if strip_scheme(url, "grpc://").is_some() || strip_scheme(url, "grpcs://").is_some() {
        url.trim_end_matches('/').to_string()
    } else {
        format!("grpc://{}", strip_protocol(url).trim_end_matches('/'))
    }
}

fn http_health_url(url: &str) -> Result<String, String> {
    match url_scheme(url).as_deref() {
        Some("http") | Some("https") => Ok(format!("{}/health", url.trim_end_matches('/'))),
        Some("grpc") | Some("grpcs") => Err(format!(
            "HTTP health check does not accept gRPC URL scheme: {url}"
        )),
        Some(scheme) => Err(format!(
            "HTTP health check does not accept URL scheme '{scheme}': {url}"
        )),
        None => Ok(format!("http://{}/health", url.trim_end_matches('/'))),
    }
}

/// The gRPC URL a probe dials for `url`: `grpc(s)://` kept, a bare
/// `host:port` given `grpc://`, HTTP(S) refused.
pub(crate) fn grpc_reachable_url(url: &str) -> Result<String, String> {
    match url_scheme(url).as_deref() {
        Some("grpc") | Some("grpcs") => Ok(url.trim_end_matches('/').to_string()),
        Some("http") | Some("https") => Err(format!(
            "gRPC health check does not accept HTTP URL scheme: {url}"
        )),
        Some(scheme) => Err(format!(
            "gRPC health check does not accept URL scheme '{scheme}': {url}"
        )),
        None => Ok(format!("grpc://{}", url.trim_end_matches('/'))),
    }
}

/// Try HTTP health check (2xx response required).
pub(crate) async fn try_http_reachable(
    url: &str,
    timeout_secs: u64,
    client: &Client,
) -> Result<(), String> {
    let health_url = http_health_url(url)?;

    client
        .get(&health_url)
        .timeout(Duration::from_secs(timeout_secs))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("Health check failed: {e}"))?;

    Ok(())
}

/// Perform a single gRPC health check with a specific runtime type.
///
/// Also used by `DetectBackendStep` for runtime identification.
pub(crate) async fn do_grpc_health_check(
    grpc_url: &str,
    timeout_secs: u64,
    runtime_type: &str,
) -> Result<(), String> {
    let connect_future = GrpcClient::connect(grpc_url, runtime_type);
    let client = tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {e}"))?;

    let health_future = client.health_check();
    tokio::time::timeout(Duration::from_secs(timeout_secs), health_future)
        .await
        .map_err(|_| "gRPC health check timeout".to_string())?
        .map_err(|e| format!("gRPC health check failed: {e}"))?;

    Ok(())
}

/// Dial the endpoint once to decide whether it is worth probing per-runtime.
///
/// All five runtime clients dial the same authority, so an endpoint that
/// cannot be reached at the transport level fails five identical connects.
/// One dial answers that question, which matters during a mass worker
/// registration: an unready pod that black-holes SYN would otherwise hold
/// five in-flight sockets per attempt instead of one.
async fn grpc_transport_reachable(grpc_url: &str, timeout_secs: u64) -> Result<(), String> {
    let timeout = Duration::from_secs(timeout_secs);
    let connect_future = connect_channel_with_timeout(grpc_url, timeout);

    // tonic's connect timeout is applied to the connector. Keep an outer
    // deadline as a safety net for the remaining Channel::connect setup.
    tokio::time::timeout(timeout, connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {e}"))?;
    Ok(())
}

/// Leads the message of a step failure no retry can change, so a step's
/// `is_retryable` can tell it from a probe that has not succeeded yet.
const CONTRACT_VIOLATION: &str = "contract violation";

/// A step failure the workflow must not retry.
pub(crate) fn contract_violation(step_id: &str, message: impl fmt::Display) -> WorkflowError {
    WorkflowError::StepFailed {
        step_id: StepId::new(step_id),
        message: format!("{CONTRACT_VIOLATION}: {message}"),
    }
}

pub(crate) fn is_contract_violation(error: &WorkflowError) -> bool {
    matches!(
        error,
        WorkflowError::StepFailed { message, .. } if message.starts_with(CONTRACT_VIOLATION)
    )
}

/// Why an SMG Worker handshake failed, split by whether a retry can change it.
#[derive(Debug)]
pub(crate) enum SmgHandshakeError {
    /// Connect or RPC failure, or a Worker not yet SERVING.
    Unavailable(String),
    /// The Worker answered, and its answer can never satisfy the Router.
    Contract(String),
}

/// Verify that an endpoint implements the SMG Worker control-plane contract:
/// identity keeps an arbitrary tonic service from registering as a Worker,
/// capabilities enforce the API major, health gates on a SERVING engine, and
/// topology must name only engines a Worker can front, each on one wire.
pub(crate) async fn try_smg_worker_reachable(
    url: &str,
    timeout_secs: u64,
) -> Result<SmgWorkerDiscovery, SmgHandshakeError> {
    use SmgHandshakeError::{Contract, Unavailable};

    let grpc_url = grpc_reachable_url(url).map_err(Contract)?;
    let timeout = Duration::from_secs(timeout_secs);
    // `timeout` bounds connect plus four sequential RPCs; the connect gets
    // half so a slow accept cannot consume the RPC budget.
    let connect_timeout = (timeout / 2).max(Duration::from_millis(500)).min(timeout);
    let handshake = async {
        let channel = connect_channel_with_timeout(&grpc_url, connect_timeout)
            .await
            .map_err(|error| Unavailable(format!("SMG Worker gRPC connection failed: {error}")))?;
        let mut client = WorkerControlClient::new(channel);

        let identity = client
            .get_identity(GetIdentityRequest {})
            .await
            .map_err(|error| Unavailable(format!("SMG Worker GetIdentity failed: {error}")))?
            .into_inner()
            .identity
            .ok_or_else(|| Contract("SMG Worker GetIdentity returned no identity".to_string()))?;
        if identity.worker_id.trim().is_empty() || identity.instance_id.trim().is_empty() {
            return Err(Contract(
                "SMG Worker identity must include non-empty worker_id and instance_id".to_string(),
            ));
        }

        let capabilities = client
            .get_capabilities(GetCapabilitiesRequest {})
            .await
            .map_err(|error| Unavailable(format!("SMG Worker GetCapabilities failed: {error}")))?
            .into_inner()
            .capabilities
            .ok_or_else(|| {
                Contract("SMG Worker GetCapabilities returned no capabilities".to_string())
            })?;
        if capabilities.api_major != WORKER_CONTROL_API_MAJOR {
            return Err(Contract(format!(
                "unsupported SMG Worker control API {}.{}; Router supports major {}",
                capabilities.api_major, capabilities.api_minor, WORKER_CONTROL_API_MAJOR
            )));
        }

        let health = client
            .get_health(GetHealthRequest {
                include_components: false,
            })
            .await
            .map_err(|error| Unavailable(format!("SMG Worker GetHealth failed: {error}")))?
            .into_inner();
        let state =
            WorkerHealthState::try_from(health.state).unwrap_or(WorkerHealthState::Unspecified);
        if state != WorkerHealthState::Serving {
            return Err(Unavailable(format!(
                "SMG Worker is not ready: state={}, message={}",
                state.as_str_name(),
                health.message
            )));
        }

        let topology = client
            .get_topology(GetTopologyRequest {})
            .await
            .map_err(|error| Unavailable(format!("SMG Worker GetTopology failed: {error}")))?
            .into_inner()
            .topology
            .ok_or_else(|| Contract("SMG Worker GetTopology returned no topology".to_string()))?;
        if topology.worker_id != identity.worker_id {
            return Err(Contract(format!(
                "SMG Worker topology worker_id {:?} does not match identity {:?}",
                topology.worker_id, identity.worker_id
            )));
        }
        if topology.engines.is_empty() {
            return Err(Contract(
                "SMG Worker topology advertises no engines".to_string(),
            ));
        }

        // The string-stop decision is per Worker, so all engines must share
        // one wire.
        let mut transports = BTreeSet::new();
        let mut engines = Vec::with_capacity(topology.engines.len());
        for engine in topology.engines {
            let engine_type = engine
                .engine_type
                .parse::<RuntimeType>()
                .ok()
                .filter(|runtime| SMG_WORKER_ENGINE_TYPES.contains(runtime))
                .ok_or_else(|| {
                    Contract(format!(
                        "SMG Worker engine {:?} runs unsupported engine type {:?}; an SMG \
                         Worker fronts {}",
                        engine.engine_id,
                        engine.engine_type,
                        smg_worker_engine_types()
                    ))
                })?;
            let engine_transport = match engine
                .attributes
                .get("engine_transport")
                .map(|value| value.to_ascii_lowercase())
            {
                Some(transport) if transport == "grpc" || transport == "zmq" => transport,
                Some(other) => {
                    return Err(Contract(format!(
                        "SMG Worker engine {:?} advertises unknown engine_transport {other:?}; \
                         expected grpc or zmq",
                        engine.engine_id
                    )))
                }
                None => {
                    return Err(Contract(format!(
                        "SMG Worker engine {:?} does not advertise its engine_transport \
                         attribute, so the Router cannot tell whether string stops reach the \
                         engine",
                        engine.engine_id
                    )))
                }
            };
            transports.insert(engine_transport.clone());
            let capability = capabilities.engines.iter().find(|candidate| {
                candidate
                    .engine_type
                    .eq_ignore_ascii_case(&engine.engine_type)
            });
            engines.push(SmgEngineDiscovery {
                engine_id: engine.engine_id,
                engine_type: engine_type.as_str().to_string(),
                engine_transport,
                engine_version: capability
                    .map(|value| value.engine_version.clone())
                    .unwrap_or_default(),
                endpoint: engine.endpoint,
                model_ids: if engine.model_ids.is_empty() {
                    capability
                        .map(|value| value.model_ids.clone())
                        .unwrap_or_default()
                } else {
                    engine.model_ids
                },
                features: capability
                    .map(|value| value.features.clone())
                    .unwrap_or_default(),
                attributes: engine.attributes,
            });
        }
        if transports.len() > 1 {
            return Err(Contract(format!(
                "SMG Worker engines disagree on engine_transport ({}); one Worker must front a \
                 single wire because the Router's string-stop decision is per Worker",
                transports.into_iter().collect::<Vec<_>>().join(", ")
            )));
        }

        Ok(SmgWorkerDiscovery {
            worker_id: identity.worker_id,
            instance_id: identity.instance_id,
            hostname: identity.hostname,
            zone: identity.zone,
            version: identity.version,
            identity_labels: identity.labels,
            api_major: capabilities.api_major,
            api_minor: capabilities.api_minor,
            features: capabilities.features,
            max_concurrent_requests: capabilities.max_concurrent_requests,
            capability_attributes: capabilities.attributes,
            topology_version: topology.topology_version,
            engines,
        })
    };

    tokio::time::timeout(timeout, handshake)
        .await
        .map_err(|_| Unavailable("SMG Worker control-plane handshake timeout".to_string()))?
}

const GRPC_RUNTIME_TYPES: [&str; 5] = ["sglang", "vllm", "trtllm", "mlx", "tokenspeed"];

async fn first_success_or_all_errors<F>(
    mut checks: FuturesUnordered<F>,
    runtimes: &[&str],
) -> Result<(), String>
where
    F: Future<Output = (usize, Result<(), String>)>,
{
    let mut errors = vec![None; runtimes.len()];

    while let Some((index, result)) = checks.next().await {
        match result {
            Ok(()) => return Ok(()),
            Err(error) => {
                if let Some(slot) = errors.get_mut(index) {
                    *slot = Some(error);
                }
            }
        }
    }

    let details = runtimes
        .iter()
        .enumerate()
        .map(|(index, runtime)| match errors[index].as_deref() {
            Some(error) => format!("{runtime}={error}"),
            None => format!("{runtime}=health check did not complete"),
        })
        .collect::<Vec<_>>()
        .join(", ");

    Err(format!(
        "gRPC not reachable (tried {}): {details}",
        runtimes.join(", ")
    ))
}

/// Check if gRPC is reachable by trying all known runtime types in parallel.
///
/// We don't care which runtime it is here — that's `DetectBackendStep`'s job.
/// We just need to know: does this endpoint speak gRPC at all?
///
/// The per-runtime fan-out is gated behind a single transport probe so an
/// unreachable endpoint costs one connect rather than five.
/// The remaining runtime probes are cancelled after the first success.
/// `timeout_secs` bounds the gate and fan-out together, not each phase.
pub(crate) async fn try_grpc_reachable(url: &str, timeout_secs: u64) -> Result<(), String> {
    let grpc_url = grpc_reachable_url(url)?;
    let timeout = Duration::from_secs(timeout_secs);

    let reachability = Box::pin(async {
        grpc_transport_reachable(&grpc_url, timeout_secs).await?;

        let checks = FuturesUnordered::new();
        for (index, runtime) in GRPC_RUNTIME_TYPES.iter().copied().enumerate() {
            let grpc_url = &grpc_url;
            checks.push(async move {
                (
                    index,
                    do_grpc_health_check(grpc_url, timeout_secs, runtime).await,
                )
            });
        }

        first_success_or_all_errors(checks, &GRPC_RUNTIME_TYPES).await
    });

    tokio::time::timeout(timeout, reachability)
        .await
        .map_err(|_| "gRPC reachability timeout".to_string())?
}

/// In-process WorkerControl services for registration tests.
#[cfg(test)]
pub(crate) mod smg_control_fixture {
    use std::collections::HashMap;

    use smg_grpc_client::worker_proto::{
        self,
        worker_control_server::{WorkerControl as WorkerControlService, WorkerControlServer},
        WorkerHealthState,
    };
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::{transport::Server, Request, Response, Status};

    /// Serve `control` on an ephemeral loopback port until the returned
    /// handle is aborted; returns the `grpc://` URL to dial.
    pub(crate) async fn serve_control<S: WorkerControlService>(
        control: S,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback listener");
        serve_control_on(listener, control)
    }

    /// Serve `control` on an already bound listener, for services that
    /// advertise their own bound address.
    pub(crate) fn serve_control_on<S: WorkerControlService>(
        listener: tokio::net::TcpListener,
        control: S,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let address = listener.local_addr().expect("listener address");
        #[expect(
            clippy::disallowed_methods,
            reason = "test-only tonic server is aborted before the test returns"
        )]
        let server = tokio::spawn(async move {
            let _ = Server::builder()
                .add_service(WorkerControlServer::new(control))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await;
        });
        (format!("grpc://{address}"), server)
    }

    /// A WorkerControl service answering from fixed responses, so each
    /// handshake rule can be violated one at a time.
    #[derive(Clone)]
    pub(crate) struct ScriptedWorkerControl {
        identity: Option<worker_proto::WorkerIdentity>,
        capabilities: Option<worker_proto::WorkerCapabilities>,
        health: WorkerHealthState,
        topology: Option<worker_proto::WorkerTopology>,
    }

    pub(crate) fn engine(
        id: &str,
        engine_type: &str,
        transport: Option<&str>,
    ) -> worker_proto::EngineEndpoint {
        worker_proto::EngineEndpoint {
            engine_id: id.to_string(),
            engine_type: engine_type.to_string(),
            endpoint: format!("grpc://127.0.0.1:1/{id}"),
            model_ids: vec![],
            attributes: transport
                .map(|transport| {
                    HashMap::from([("engine_transport".to_string(), transport.to_string())])
                })
                .unwrap_or_default(),
            ..Default::default()
        }
    }

    impl ScriptedWorkerControl {
        /// A conforming Worker: one vLLM engine over ZMQ, serving `model-a`.
        pub(crate) fn serving() -> Self {
            Self {
                identity: Some(worker_proto::WorkerIdentity {
                    worker_id: "worker-a".to_string(),
                    instance_id: "instance-a".to_string(),
                    ..Default::default()
                }),
                capabilities: Some(worker_proto::WorkerCapabilities {
                    api_major: 1,
                    api_minor: 3,
                    features: vec!["generate".to_string()],
                    engines: vec![worker_proto::EngineCapability {
                        engine_type: "vllm".to_string(),
                        engine_version: "0.11".to_string(),
                        model_ids: vec!["model-a".to_string()],
                        features: vec!["generate".to_string()],
                    }],
                    max_concurrent_requests: 8,
                    ..Default::default()
                }),
                health: WorkerHealthState::Serving,
                topology: Some(worker_proto::WorkerTopology {
                    worker_id: "worker-a".to_string(),
                    topology_version: 1,
                    engines: vec![engine("engine-0", "vllm", Some("ZMQ"))],
                    ..Default::default()
                }),
            }
        }

        pub(crate) fn with_engines(mut self, engines: Vec<worker_proto::EngineEndpoint>) -> Self {
            if let Some(topology) = self.topology.as_mut() {
                topology.engines = engines;
            }
            self
        }

        pub(crate) fn with_health(mut self, health: WorkerHealthState) -> Self {
            self.health = health;
            self
        }

        pub(crate) fn with_api_major(mut self, api_major: u32) -> Self {
            if let Some(capabilities) = self.capabilities.as_mut() {
                capabilities.api_major = api_major;
            }
            self
        }

        pub(crate) fn with_identity(
            mut self,
            identity: Option<worker_proto::WorkerIdentity>,
        ) -> Self {
            self.identity = identity;
            self
        }
    }

    #[tonic::async_trait]
    impl WorkerControlService for ScriptedWorkerControl {
        async fn get_identity(
            &self,
            _request: Request<worker_proto::GetIdentityRequest>,
        ) -> Result<Response<worker_proto::GetIdentityResponse>, Status> {
            Ok(Response::new(worker_proto::GetIdentityResponse {
                identity: self.identity.clone(),
            }))
        }

        async fn get_capabilities(
            &self,
            _request: Request<worker_proto::GetCapabilitiesRequest>,
        ) -> Result<Response<worker_proto::GetCapabilitiesResponse>, Status> {
            Ok(Response::new(worker_proto::GetCapabilitiesResponse {
                capabilities: self.capabilities.clone(),
            }))
        }

        async fn get_health(
            &self,
            _request: Request<worker_proto::GetHealthRequest>,
        ) -> Result<Response<worker_proto::GetHealthResponse>, Status> {
            Ok(Response::new(worker_proto::GetHealthResponse {
                state: self.health.into(),
                message: "scripted".to_string(),
                ..Default::default()
            }))
        }

        async fn get_topology(
            &self,
            _request: Request<worker_proto::GetTopologyRequest>,
        ) -> Result<Response<worker_proto::GetTopologyResponse>, Status> {
            Ok(Response::new(worker_proto::GetTopologyResponse {
                topology: self.topology.clone(),
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::pending,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
    };

    use mock_worker::{
        config::Config as MockWorkerConfig, control::MockWorkerControl, engine::EngineParams,
    };
    use smg_grpc_client::worker_proto;

    use super::{
        smg_control_fixture::{engine, serve_control, serve_control_on, ScriptedWorkerControl},
        *,
    };

    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn http_health_url_accepts_http_https_and_bare_urls() {
        assert_eq!(
            http_health_url("http://localhost:30000").unwrap(),
            "http://localhost:30000/health"
        );
        assert_eq!(
            http_health_url("https://example.com/").unwrap(),
            "https://example.com/health"
        );
        assert_eq!(
            http_health_url("localhost:30000").unwrap(),
            "http://localhost:30000/health"
        );
    }

    #[test]
    fn http_health_url_rejects_grpc_schemes() {
        assert!(http_health_url("grpc://localhost:30001").is_err());
        assert!(http_health_url("grpcs://localhost:30001").is_err());
    }

    #[test]
    fn grpc_reachable_url_accepts_grpc_grpcs_and_bare_urls() {
        assert_eq!(
            grpc_reachable_url("grpc://localhost:30001").unwrap(),
            "grpc://localhost:30001"
        );
        assert_eq!(
            grpc_reachable_url("grpcs://localhost:30001/").unwrap(),
            "grpcs://localhost:30001"
        );
        assert_eq!(
            grpc_reachable_url("localhost:30001").unwrap(),
            "grpc://localhost:30001"
        );
    }

    #[test]
    fn grpc_reachable_url_rejects_http_schemes() {
        assert!(grpc_reachable_url("http://localhost:30000").is_err());
        assert!(grpc_reachable_url("https://localhost:30000").is_err());
    }

    #[tokio::test]
    async fn runtime_fanout_returns_on_first_success_and_cancels_stalled_checks() {
        let stalled_started = Arc::new(AtomicBool::new(false));
        let stalled_cancelled = Arc::new(AtomicBool::new(false));
        let checks = FuturesUnordered::new();

        for (index, succeeds) in [false, true].into_iter().enumerate() {
            let stalled_started = Arc::clone(&stalled_started);
            let stalled_cancelled = Arc::clone(&stalled_cancelled);
            checks.push(async move {
                let result = if succeeds {
                    tokio::task::yield_now().await;
                    Ok(())
                } else {
                    stalled_started.store(true, Ordering::SeqCst);
                    let _drop_flag = DropFlag(stalled_cancelled);
                    pending::<Result<(), String>>().await
                };
                (index, result)
            });
        }

        let result = tokio::time::timeout(
            Duration::from_secs(1),
            first_success_or_all_errors(checks, &["stalled", "healthy"]),
        )
        .await;

        assert!(matches!(result, Ok(Ok(()))));
        assert!(stalled_started.load(Ordering::SeqCst));
        assert!(stalled_cancelled.load(Ordering::SeqCst));
    }

    /// An endpoint that cannot be reached at the transport level must
    /// short-circuit before the per-runtime fan-out, so one unreachable worker
    /// costs one dial rather than five.
    ///
    /// TCP port 0 cannot be owned by a listener: binding it asks the OS for an
    /// ephemeral nonzero port. This makes the local transport failure
    /// deterministic without a bind-then-drop port-reuse race.
    /// The fan-out aggregate can only be constructed after all runtime probes
    /// run, so the transport error proves the gate returned first.
    #[tokio::test]
    async fn unreachable_transport_short_circuits_the_runtime_fanout() {
        let err = try_grpc_reachable("grpc://127.0.0.1:0", 1)
            .await
            .expect_err("closed local endpoint should not be reachable");

        assert!(
            err.starts_with("gRPC connection"),
            "expected the transport-gate error, got: {err}"
        );
        assert!(
            !err.contains("gRPC not reachable"),
            "fan-out aggregate was produced, so the gate did not short-circuit: {err}"
        );
        for runtime in ["sglang", "vllm", "trtllm", "mlx", "tokenspeed"] {
            assert!(
                !err.contains(runtime),
                "error names {runtime}, so a per-runtime probe ran: {err}"
            );
        }
    }

    async fn handshake(
        control: ScriptedWorkerControl,
    ) -> Result<SmgWorkerDiscovery, SmgHandshakeError> {
        let (url, server) = serve_control(control).await;
        let result = try_smg_worker_reachable(&url, 2).await;
        server.abort();
        result
    }

    fn contract_reason(result: Result<SmgWorkerDiscovery, SmgHandshakeError>) -> String {
        match result {
            Err(SmgHandshakeError::Contract(reason)) => reason,
            other => panic!("expected a contract violation, got {other:?}"),
        }
    }

    fn unavailable_reason(result: Result<SmgWorkerDiscovery, SmgHandshakeError>) -> String {
        match result {
            Err(SmgHandshakeError::Unavailable(reason)) => reason,
            other => panic!("expected an unavailable Worker, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn smg_worker_handshake_accepts_the_rust_mock_control_plane() {
        let config = MockWorkerConfig {
            host: "127.0.0.1".to_string(),
            http_base_port: 0,
            http_count: 0,
            grpc_base_port: 0,
            grpc_count: 1,
            zmq_handshake: None,
            zmq_count: 0,
            zmq_start_index: 0,
            model_id: "mock-model".to_string(),
            tokenizer_path: "mock-model".to_string(),
            gen_delay: Duration::ZERO,
            output_tokens: 8,
            realistic: false,
            engine: EngineParams::default(),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback listener");
        let bound = listener.local_addr().expect("listener address");
        let (url, server) = serve_control_on(listener, MockWorkerControl::new(&config, bound));

        let discovery = try_smg_worker_reachable(&url, 2)
            .await
            .expect("mock Worker control plane conforms");
        server.abort();

        assert_eq!(discovery.worker_id, format!("mock-worker-{}", bound.port()));
        assert_eq!(discovery.engines.len(), 1);
        assert_eq!(discovery.engines[0].engine_type, "tokenspeed");
        assert_eq!(discovery.engines[0].engine_transport, "grpc");
        assert_eq!(discovery.engines[0].model_ids, ["mock-model"]);
    }

    /// Engine type and transport are stored in canonical spelling, and the
    /// per-engine model list falls back to the capability entry.
    #[tokio::test]
    async fn smg_worker_handshake_normalizes_engine_type_and_transport() {
        let discovery = handshake(ScriptedWorkerControl::serving().with_engines(vec![engine(
            "engine-0",
            "VLLM",
            Some("ZMQ"),
        )]))
        .await
        .expect("conforming Worker");

        assert_eq!(discovery.worker_id, "worker-a");
        assert_eq!(discovery.instance_id, "instance-a");
        assert_eq!((discovery.api_major, discovery.api_minor), (1, 3));
        let engine = &discovery.engines[0];
        assert_eq!(engine.engine_type, "vllm");
        assert_eq!(engine.engine_transport, "zmq");
        assert_eq!(engine.engine_version, "0.11");
        assert_eq!(engine.model_ids, ["model-a"]);
    }

    #[tokio::test]
    async fn smg_worker_handshake_rejects_an_unsupported_api_major() {
        let reason =
            contract_reason(handshake(ScriptedWorkerControl::serving().with_api_major(2)).await);
        assert!(reason.contains("control API 2.3"), "{reason}");
        assert!(reason.contains("major 1"), "{reason}");
    }

    #[tokio::test]
    async fn smg_worker_handshake_rejects_a_missing_engine_transport() {
        let reason = contract_reason(
            handshake(
                ScriptedWorkerControl::serving()
                    .with_engines(vec![engine("engine-0", "vllm", None)]),
            )
            .await,
        );
        assert!(
            reason.contains("does not advertise its engine_transport"),
            "{reason}"
        );
    }

    #[tokio::test]
    async fn smg_worker_handshake_rejects_mixed_engine_transports() {
        let reason = contract_reason(
            handshake(ScriptedWorkerControl::serving().with_engines(vec![
                engine("engine-0", "vllm", Some("grpc")),
                engine("engine-1", "vllm", Some("zmq")),
            ]))
            .await,
        );
        assert!(
            reason.contains("disagree on engine_transport (grpc, zmq)"),
            "{reason}"
        );
    }

    #[tokio::test]
    async fn smg_worker_handshake_rejects_an_unknown_engine_transport() {
        let reason = contract_reason(
            handshake(ScriptedWorkerControl::serving().with_engines(vec![engine(
                "engine-0",
                "vllm",
                Some("carrier-pigeon"),
            )]))
            .await,
        );
        assert!(
            reason.contains("unknown engine_transport \"carrier-pigeon\""),
            "{reason}"
        );
    }

    #[tokio::test]
    async fn smg_worker_handshake_rejects_engines_a_worker_cannot_front() {
        for engine_type in ["sglang", "trtllm", "mlx", ""] {
            let reason = contract_reason(
                handshake(ScriptedWorkerControl::serving().with_engines(vec![engine(
                    "engine-0",
                    engine_type,
                    Some("grpc"),
                )]))
                .await,
            );
            assert!(
                reason.contains("unsupported engine type") && reason.contains("vllm or tokenspeed"),
                "{engine_type:?}: {reason}"
            );
        }
    }

    /// A Worker that answers but is not SERVING is retried at the startup
    /// cadence rather than rejected.
    #[tokio::test]
    async fn smg_worker_handshake_treats_a_non_serving_worker_as_unavailable() {
        for state in [
            WorkerHealthState::Starting,
            WorkerHealthState::NotServing,
            WorkerHealthState::Draining,
            WorkerHealthState::Unspecified,
        ] {
            let reason = unavailable_reason(
                handshake(ScriptedWorkerControl::serving().with_health(state)).await,
            );
            assert!(
                reason.contains(&format!("state={}", state.as_str_name())),
                "{reason}"
            );
        }
    }

    #[tokio::test]
    async fn smg_worker_handshake_rejects_a_blank_or_missing_identity() {
        let blank = worker_proto::WorkerIdentity {
            worker_id: "worker-a".to_string(),
            instance_id: "   ".to_string(),
            ..Default::default()
        };
        let reason = contract_reason(
            handshake(ScriptedWorkerControl::serving().with_identity(Some(blank))).await,
        );
        assert!(
            reason.contains("non-empty worker_id and instance_id"),
            "{reason}"
        );

        let reason =
            contract_reason(handshake(ScriptedWorkerControl::serving().with_identity(None)).await);
        assert!(reason.contains("returned no identity"), "{reason}");
    }

    /// Port 0 cannot be listened on, so the dial fails deterministically:
    /// that is an unavailable Worker, not a contract violation.
    #[tokio::test]
    async fn smg_worker_handshake_treats_a_refused_connection_as_unavailable() {
        let reason = unavailable_reason(try_smg_worker_reachable("grpc://127.0.0.1:0", 1).await);
        assert!(
            reason.starts_with("SMG Worker gRPC connection failed"),
            "{reason}"
        );
    }

    #[tokio::test]
    async fn smg_worker_handshake_rejects_an_http_url_before_dialing() {
        let reason = contract_reason(try_smg_worker_reachable("http://127.0.0.1:0", 1).await);
        assert!(
            reason.contains("does not accept HTTP URL scheme"),
            "{reason}"
        );
    }

    #[test]
    fn contract_violations_are_recognized_by_message_and_nothing_else_is() {
        let violation = contract_violation("detect_connection_mode", "api major 2 is unsupported");
        assert!(is_contract_violation(&violation));
        assert_eq!(
            violation.to_string(),
            "Step failed: detect_connection_mode - contract violation: api major 2 is unsupported"
        );

        let transient = WorkflowError::StepFailed {
            step_id: StepId::new("detect_connection_mode"),
            message: "SMG Worker gRPC connection failed".to_string(),
        };
        assert!(!is_contract_violation(&transient));
        assert!(!is_contract_violation(
            &WorkflowError::ContextValueNotFound("app_context".to_string())
        ));
    }
}
