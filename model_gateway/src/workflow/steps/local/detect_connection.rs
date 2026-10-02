//! Connection mode detection step.
//!
//! Determines whether a worker communicates via HTTP or gRPC, and for HTTP
//! which version the router speaks to it.
//! This step only answers "HTTP or gRPC?" — backend runtime detection
//! (sglang vs vllm vs trtllm) is handled by the separate DetectBackendStep.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    app_context::AppContext,
    worker::{ConnectionMode, WorkerMode},
    workflow::{
        data::{WorkerKind, WorkerWorkflowData},
        steps::util::{
            contract_violation, is_contract_violation, try_grpc_reachable, try_http_reachable,
            try_smg_worker_reachable, SmgHandshakeError,
        },
    },
};

const STEP_ID: &str = "detect_connection_mode";

/// Step 1: Detect connection mode (HTTP vs gRPC).
///
/// Explicit URL schemes are honored. For bare host:port URLs, probes both
/// protocols in parallel and HTTP takes priority if both succeed. An HTTP
/// worker also gets its HTTP version resolved here (see [`HttpVersionPolicy`]).
/// Does NOT detect backend runtime — that's handled by DetectBackendStep.
pub struct DetectConnectionModeStep;

/// How the HTTP probe picks the version the router speaks to a worker.
enum HttpVersionPolicy {
    /// `http_pool.http2` was declared, or negotiation does not apply.
    Fixed(bool),
    /// `upstream_http2` on a cleartext URL: probe HTTP/2 prior knowledge and
    /// HTTP/1.1 in parallel and prefer HTTP/2.
    Negotiate,
}

impl HttpVersionPolicy {
    fn resolve(url: &str, declared: Option<bool>, upstream_http2: bool) -> Self {
        match declared {
            Some(http2) => Self::Fixed(http2),
            // TLS negotiates the version via ALPN; prior knowledge would pin it.
            None if upstream_http2 && !url.starts_with("https://") => Self::Negotiate,
            None => Self::Fixed(false),
        }
    }
}

/// A reachable HTTP worker: the version it answered and the client to keep.
struct HttpProbe {
    http2: bool,
    client: Arc<reqwest::Client>,
}

async fn probe_http(
    app_context: &AppContext,
    data: &WorkerWorkflowData,
    timeout: u64,
) -> Result<HttpProbe, String> {
    let config = &data.config;
    let cache = &app_context.worker_client_cache;
    let policy = HttpVersionPolicy::resolve(
        &config.url,
        config.http_pool.http2,
        app_context.router_config.upstream_http2,
    );
    match policy {
        HttpVersionPolicy::Fixed(http2) => {
            let client = cache.get(&config.http_pool, http2)?;
            try_http_reachable(&config.url, timeout, &client).await?;
            Ok(HttpProbe { http2, client })
        }
        HttpVersionPolicy::Negotiate => {
            let h2 = cache.get(&config.http_pool, true)?;
            let h1 = cache.get(&config.http_pool, false)?;
            let (h2_result, h1_result) = tokio::join!(
                try_http_reachable(&config.url, timeout, &h2),
                try_http_reachable(&config.url, timeout, &h1)
            );
            match (h2_result, h1_result) {
                (Ok(()), _) => Ok(HttpProbe {
                    http2: true,
                    client: h2,
                }),
                (Err(h2_err), Ok(())) => {
                    debug!(
                        worker = %config.url,
                        error = %h2_err,
                        "HTTP/2 prior-knowledge probe failed; staying on HTTP/1.1"
                    );
                    Ok(HttpProbe {
                        http2: false,
                        client: h1,
                    })
                }
                (Err(h2_err), Err(h1_err)) => Err(format!("HTTP/2: {h2_err}; HTTP/1.1: {h1_err}")),
            }
        }
    }
}

fn record_http(context: &mut WorkflowContext<WorkerWorkflowData>, probe: HttpProbe) {
    context.data.http2 = Some(probe.http2);
    context.data.http_client_handle = Some(probe.client);
}

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DetectConnectionModeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();
        let config = &context.data.config;

        debug!(
            "Detecting connection mode for {} (timeout: {:?}s, max_attempts: {})",
            config.url, config.health.timeout_secs, config.max_connection_attempts
        );

        let url = config.url.clone();
        let timeout = config
            .health
            .timeout_secs
            .unwrap_or(app_context.router_config.health_check.timeout_secs);

        // An SMG Worker is identified by the WorkerControl handshake, never
        // inferred from engine probes. A Worker that is unreachable or not yet
        // SERVING is retried at the startup cadence; one whose answer can
        // never satisfy the Router fails the registration now.
        if config.worker_mode == WorkerMode::Smg {
            if let Some(mode) =
                ConnectionMode::from_url(&url).filter(|mode| *mode != ConnectionMode::Grpc)
            {
                return Err(contract_violation(
                    STEP_ID,
                    format!(
                        "SMG Worker {} must use a grpc:// or grpcs:// URL, got {mode}",
                        config.url
                    ),
                ));
            }
            let control_url = config.control_url.as_deref().unwrap_or(&url);
            if let Some(mode) =
                ConnectionMode::from_url(control_url).filter(|mode| *mode != ConnectionMode::Grpc)
            {
                return Err(contract_violation(
                    STEP_ID,
                    format!(
                        "SMG Worker control endpoint {control_url} must use grpc:// or grpcs://, \
                         got {mode}"
                    ),
                ));
            }
            let discovery = try_smg_worker_reachable(control_url, timeout)
                .await
                .map_err(|error| match error {
                    SmgHandshakeError::Unavailable(reason) => WorkflowError::StepFailed {
                        step_id: StepId::new(STEP_ID),
                        message: format!(
                            "SMG Worker control-plane handshake failed for {control_url}: {reason}"
                        ),
                    },
                    SmgHandshakeError::Contract(reason) => contract_violation(
                        STEP_ID,
                        format!("SMG Worker control-plane handshake with {control_url}: {reason}"),
                    ),
                })?;
            if config.runtime_type.is_specified()
                && !discovery.engines.iter().any(|engine| {
                    engine
                        .engine_type
                        .eq_ignore_ascii_case(config.runtime_type.as_str())
                })
            {
                return Err(contract_violation(
                    STEP_ID,
                    format!(
                        "SMG Worker {} does not advertise configured runtime {}; engines={:?}",
                        discovery.worker_id,
                        config.runtime_type,
                        discovery
                            .engines
                            .iter()
                            .map(|engine| engine.engine_type.as_str())
                            .collect::<Vec<_>>()
                    ),
                ));
            }
            debug!(
                worker_url = %config.url,
                control_url,
                worker_mode = %config.worker_mode,
                worker_id = %discovery.worker_id,
                instance_id = %discovery.instance_id,
                "SMG Worker handshake succeeded"
            );
            context.data.connection_mode = Some(ConnectionMode::Grpc);
            context.data.smg_worker_discovery = Some(discovery);
            return Ok(StepResult::Success);
        }

        let connection_mode = if let Some(connection_mode) = ConnectionMode::from_url(&url) {
            let result = match connection_mode {
                ConnectionMode::Http => {
                    match probe_http(&app_context, &context.data, timeout).await {
                        Ok(probe) => {
                            record_http(context, probe);
                            Ok(())
                        }
                        Err(err) => Err(err),
                    }
                }
                ConnectionMode::Grpc => try_grpc_reachable(&url, timeout).await,
                // SMG binds the ZMQ sockets and the engine dials in, so there is
                // no endpoint to probe before binding; an explicit ipc:// URL is
                // taken as reachable.
                ConnectionMode::Zmq => Ok(()),
            };

            if let Err(err) = result {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new("detect_connection_mode"),
                    message: format!(
                        "{connection_mode} health check failed for explicitly configured worker URL {url}: {err}"
                    ),
                });
            }
            debug!("{url} explicitly configured as {connection_mode}");
            connection_mode
        } else {
            let (http_result, grpc_result) = tokio::join!(
                probe_http(&app_context, &context.data, timeout),
                try_grpc_reachable(&url, timeout)
            );

            match (http_result, grpc_result) {
                (Ok(probe), _) => {
                    debug!("{url} detected as HTTP");
                    record_http(context, probe);
                    ConnectionMode::Http
                }
                (_, Ok(())) => {
                    debug!("{url} detected as gRPC");
                    ConnectionMode::Grpc
                }
                (Err(http_err), Err(grpc_err)) => {
                    return Err(WorkflowError::StepFailed {
                        step_id: StepId::new("detect_connection_mode"),
                        message: format!(
                            "Both HTTP and gRPC health checks failed for {url}: HTTP: {http_err}, gRPC: {grpc_err}"
                        ),
                    });
                }
            }
        };

        if let Some(http2) = context
            .data
            .http2
            .filter(|_| app_context.router_config.upstream_http2)
        {
            info!(worker = %url, http2, "resolved worker HTTP version");
        }
        context.data.connection_mode = Some(connection_mode);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, error: &WorkflowError) -> bool {
        !is_contract_violation(error)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use llm_tokenizer::registry::TokenizerRegistry;
    use openai_protocol::worker::{HttpPoolConfig, RuntimeType, WorkerSpec};
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use smg_grpc_client::worker_proto::WorkerHealthState;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        config::RouterConfig,
        policies::PolicyRegistry,
        worker::WorkerRegistry,
        workflow::{
            data::WorkerRegistrationMode,
            steps::{
                create_worker_workflow_data,
                util::smg_control_fixture::{engine, serve_control, ScriptedWorkerControl},
            },
        },
    };

    fn app_context(upstream_http2: bool) -> Arc<AppContext> {
        let router_config = RouterConfig {
            upstream_http2,
            ..RouterConfig::default()
        };
        Arc::new(
            AppContext::builder()
                .client(reqwest::Client::new())
                .rate_limiter(None)
                .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
                .reasoning_parser_factory(None)
                .tool_parser_factory(None)
                .worker_registry(Arc::new(WorkerRegistry::new()))
                .policy_registry(Arc::new(PolicyRegistry::new(router_config.policy.clone())))
                .router_config(router_config)
                .response_storage(Arc::new(MemoryResponseStorage::new()))
                .conversation_storage(Arc::new(MemoryConversationStorage::new()))
                .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
                .worker_monitor(None)
                .worker_job_queue(Arc::new(OnceLock::new()))
                .workflow_engines(Arc::new(OnceLock::new()))
                .mcp_orchestrator(Arc::new(OnceLock::new()))
                .build()
                .expect("app context"),
        )
    }

    fn local_context(
        app_context: Arc<AppContext>,
        spec: WorkerSpec,
    ) -> WorkflowContext<WorkerWorkflowData> {
        let mut data =
            create_worker_workflow_data(spec, WorkerRegistrationMode::Upsert, app_context);
        data.worker_kind = Some(WorkerKind::Local);
        WorkflowContext::new(WorkflowInstanceId::new(), data)
    }

    /// axum::serve answers HTTP/1.1 and prior-knowledge h2c on one listener.
    async fn spawn_dual_protocol_server() -> String {
        let app = axum::Router::new().route("/health", axum::routing::get(|| async { "ok" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        #[expect(
            clippy::disallowed_methods,
            reason = "test server lives for the duration of the test process"
        )]
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve");
        });
        format!("http://{addr}")
    }

    /// HTTP/1.1-only server: drops a connection that opens with the HTTP/2
    /// preface and answers every HTTP/1.1 request 200.
    async fn spawn_http1_only_server() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        #[expect(
            clippy::disallowed_methods,
            reason = "test server lives for the duration of the test process"
        )]
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.expect("accept");
                #[expect(clippy::disallowed_methods, reason = "one task per test connection")]
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    if buf[..n].starts_with(b"PRI * HTTP/2.0") {
                        return;
                    }
                    let _ = stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
                        )
                        .await;
                });
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn negotiates_http2_with_a_dual_protocol_worker() {
        let url = spawn_dual_protocol_server().await;
        let app_context = app_context(true);
        let mut ctx = local_context(Arc::clone(&app_context), WorkerSpec::new(url));

        let result = DetectConnectionModeStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success);
        assert_eq!(ctx.data.connection_mode, Some(ConnectionMode::Http));
        assert_eq!(ctx.data.http2, Some(true));
        let expected = app_context
            .worker_client_cache
            .get(&HttpPoolConfig::default(), true)
            .unwrap();
        assert!(Arc::ptr_eq(
            ctx.data.http_client_handle.as_ref().unwrap(),
            &expected
        ));
    }

    #[tokio::test]
    async fn falls_back_to_http1_when_the_worker_rejects_the_preface() {
        let url = spawn_http1_only_server().await;
        let app_context = app_context(true);
        let mut ctx = local_context(Arc::clone(&app_context), WorkerSpec::new(url));

        let result = DetectConnectionModeStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success);
        assert_eq!(ctx.data.http2, Some(false));
        let expected = app_context
            .worker_client_cache
            .get(&HttpPoolConfig::default(), false)
            .unwrap();
        assert!(Arc::ptr_eq(
            ctx.data.http_client_handle.as_ref().unwrap(),
            &expected
        ));
    }

    #[tokio::test]
    async fn stays_on_http1_without_upstream_http2() {
        let url = spawn_dual_protocol_server().await;
        let mut ctx = local_context(app_context(false), WorkerSpec::new(url));

        let result = DetectConnectionModeStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success);
        assert_eq!(ctx.data.http2, Some(false));
    }

    #[tokio::test]
    async fn declared_http2_false_skips_negotiation() {
        let url = spawn_dual_protocol_server().await;
        let mut spec = WorkerSpec::new(url);
        spec.http_pool.http2 = Some(false);
        let mut ctx = local_context(app_context(true), spec);

        let result = DetectConnectionModeStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success);
        assert_eq!(ctx.data.http2, Some(false));
    }

    #[tokio::test]
    async fn declared_http2_true_fails_against_an_http1_worker() {
        let url = spawn_http1_only_server().await;
        let mut spec = WorkerSpec::new(url);
        spec.http_pool.http2 = Some(true);
        let mut ctx = local_context(app_context(false), spec);

        let err = DetectConnectionModeStep
            .execute(&mut ctx)
            .await
            .unwrap_err();

        assert!(matches!(err, WorkflowError::StepFailed { .. }), "{err:?}");
        assert_eq!(ctx.data.http2, None);
    }

    #[tokio::test]
    async fn bare_host_port_negotiates_too() {
        let url = spawn_dual_protocol_server().await;
        let bare = url.trim_start_matches("http://").to_string();
        let mut ctx = local_context(app_context(true), WorkerSpec::new(bare));

        let result = DetectConnectionModeStep.execute(&mut ctx).await.unwrap();

        assert_eq!(result, StepResult::Success);
        assert_eq!(ctx.data.connection_mode, Some(ConnectionMode::Http));
        assert_eq!(ctx.data.http2, Some(true));
    }

    fn smg_spec(url: &str) -> WorkerSpec {
        let mut spec = WorkerSpec::new(url);
        spec.worker_mode = WorkerMode::Smg;
        spec.connection_mode = ConnectionMode::Grpc;
        spec
    }

    async fn run_smg(
        spec: WorkerSpec,
    ) -> Result<WorkflowContext<WorkerWorkflowData>, WorkflowError> {
        let mut ctx = local_context(app_context(false), spec);
        DetectConnectionModeStep.execute(&mut ctx).await?;
        Ok(ctx)
    }

    /// The control URL is dialed for the handshake; the inference URL is not
    /// probed at all.
    #[tokio::test]
    async fn smg_worker_handshake_records_the_discovery() {
        let (control_url, server) = serve_control(ScriptedWorkerControl::serving()).await;
        let mut spec = smg_spec("grpc://127.0.0.1:1");
        spec.control_url = Some(control_url);

        let ctx = run_smg(spec).await.expect("handshake succeeds");
        server.abort();

        assert_eq!(ctx.data.connection_mode, Some(ConnectionMode::Grpc));
        let discovery = ctx.data.smg_worker_discovery.expect("handshake record");
        assert_eq!(discovery.worker_id, "worker-a");
        assert_eq!(discovery.engines[0].engine_type, "vllm");
        assert_eq!(discovery.engines[0].engine_transport, "zmq");
    }

    /// Answers the Router can never accept fail the registration on the
    /// first attempt instead of being polled at the startup cadence.
    #[tokio::test]
    async fn smg_worker_contract_violations_are_not_retried() {
        let (url, server) = serve_control(ScriptedWorkerControl::serving().with_api_major(2)).await;
        let err = run_smg(smg_spec(&url)).await.unwrap_err();
        server.abort();
        assert!(!DetectConnectionModeStep.is_retryable(&err), "{err}");
        assert!(err.to_string().contains("control API 2.3"), "{err}");

        let (url, server) = serve_control(
            ScriptedWorkerControl::serving().with_engines(vec![engine("engine-0", "vllm", None)]),
        )
        .await;
        let err = run_smg(smg_spec(&url)).await.unwrap_err();
        server.abort();
        assert!(!DetectConnectionModeStep.is_retryable(&err), "{err}");
        assert!(err.to_string().contains("engine_transport"), "{err}");

        let (url, server) = serve_control(ScriptedWorkerControl::serving()).await;
        let mut spec = smg_spec(&url);
        spec.runtime_type = RuntimeType::TokenSpeed;
        let err = run_smg(spec).await.unwrap_err();
        server.abort();
        assert!(!DetectConnectionModeStep.is_retryable(&err), "{err}");
        assert!(
            err.to_string()
                .contains("does not advertise configured runtime tokenspeed"),
            "{err}"
        );

        let err = run_smg(smg_spec("http://127.0.0.1:1")).await.unwrap_err();
        assert!(!DetectConnectionModeStep.is_retryable(&err), "{err}");
        assert!(err.to_string().contains("grpc:// or grpcs://"), "{err}");

        let mut spec = smg_spec("grpc://127.0.0.1:1");
        spec.control_url = Some("http://127.0.0.1:1".to_string());
        let err = run_smg(spec).await.unwrap_err();
        assert!(!DetectConnectionModeStep.is_retryable(&err), "{err}");
        assert!(err.to_string().contains("control endpoint"), "{err}");
    }

    /// A Worker that cannot be reached or is still starting is polled at the
    /// startup cadence, exactly like an engine that is not up yet.
    #[tokio::test]
    async fn smg_worker_unavailability_is_retried() {
        let err = run_smg(smg_spec("grpc://127.0.0.1:0")).await.unwrap_err();
        assert!(DetectConnectionModeStep.is_retryable(&err), "{err}");
        assert!(err.to_string().contains("handshake failed"), "{err}");

        let (url, server) = serve_control(
            ScriptedWorkerControl::serving().with_health(WorkerHealthState::Starting),
        )
        .await;
        let err = run_smg(smg_spec(&url)).await.unwrap_err();
        server.abort();
        assert!(DetectConnectionModeStep.is_retryable(&err), "{err}");
        assert!(err.to_string().contains("not ready"), "{err}");
    }
}
