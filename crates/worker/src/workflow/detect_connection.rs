//! Connection mode detection step.
//!
//! Determines whether a worker communicates via HTTP or gRPC, and for HTTP
//! which version the host speaks to it.
//! This step only answers "HTTP or gRPC?"; which runtime the worker runs is
//! the runtime detection's question.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, info};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{
    data::{WorkerKind, WorkerRegistrationWorkflowData},
    host::{require_host, WorkflowHost},
    reach::{try_grpc_reachable, try_http_reachable},
    registration::DETECT_CONNECTION_MODE_STEP,
};
use crate::identity::{basic::ConnectionMode, endpoint::Endpoint};

/// Detects the connection mode (HTTP or gRPC).
///
/// Explicit URL schemes are honored. For bare host:port URLs, probes both
/// protocols in parallel and HTTP takes priority if both succeed. An HTTP
/// worker also gets its HTTP version resolved here (see `HttpVersionPolicy`).
/// It does not detect the worker's runtime.
pub struct DetectConnectionModeStep;

/// How the HTTP probe picks the version the host speaks to a worker.
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

async fn probe_http<H: WorkflowHost>(
    host: &H,
    data: &WorkerRegistrationWorkflowData<H>,
    timeout: u64,
) -> Result<HttpProbe, String> {
    let config = &data.config;
    let cache = host.worker_client_cache();
    let policy = HttpVersionPolicy::resolve(
        &config.url,
        config.http_pool.http2,
        host.registration_settings().upstream_http2,
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
                        worker = %Endpoint::redacted_str(&config.url),
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

fn record_http<H: WorkflowHost>(
    context: &mut WorkflowContext<WorkerRegistrationWorkflowData<H>>,
    probe: HttpProbe,
) {
    context.data.http2 = Some(probe.http2);
    context.data.http_client_handle = Some(probe.client);
}

#[async_trait]
impl<H: WorkflowHost> StepExecutor<WorkerRegistrationWorkflowData<H>> for DetectConnectionModeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRegistrationWorkflowData<H>>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let host = require_host(context.data.host.as_ref())?.clone();
        let connector = host.engine_connector();
        let config = &context.data.config;

        let url = config.url.clone();
        // The address as the step's lines and messages write it: a secret in
        // its userinfo masked.
        let shown = Endpoint::redacted_str(&url);
        debug!(
            "Detecting connection mode for {shown} (timeout: {:?}s, max_attempts: {})",
            config.health.timeout_secs, config.max_connection_attempts
        );

        let timeout = config
            .health
            .timeout_secs
            .unwrap_or_else(|| host.registration_settings().health_check_timeout_secs);

        let connection_mode = if let Some(connection_mode) = ConnectionMode::from_url(&url) {
            let result = match connection_mode {
                ConnectionMode::Http => {
                    match probe_http(host.as_ref(), &context.data, timeout).await {
                        Ok(probe) => {
                            record_http(context, probe);
                            Ok(())
                        }
                        Err(err) => Err(err),
                    }
                }
                ConnectionMode::Grpc => try_grpc_reachable(connector.as_ref(), &url, timeout).await,
                // The host binds the ZMQ sockets and the engine dials in, so there is
                // no endpoint to probe before binding; an explicit ipc:// URL is
                // taken as reachable.
                ConnectionMode::Zmq => Ok(()),
            };

            if let Err(err) = result {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new(DETECT_CONNECTION_MODE_STEP),
                    message: format!(
                        "{connection_mode} health check failed for explicitly configured worker URL {shown}: {err}"
                    ),
                });
            }
            debug!("{shown} explicitly configured as {connection_mode}");
            connection_mode
        } else {
            let (http_result, grpc_result) = tokio::join!(
                probe_http(host.as_ref(), &context.data, timeout),
                try_grpc_reachable(connector.as_ref(), &url, timeout)
            );

            match (http_result, grpc_result) {
                (Ok(probe), _) => {
                    debug!("{shown} detected as HTTP");
                    record_http(context, probe);
                    ConnectionMode::Http
                }
                (_, Ok(())) => {
                    debug!("{shown} detected as gRPC");
                    ConnectionMode::Grpc
                }
                (Err(http_err), Err(grpc_err)) => {
                    return Err(WorkflowError::StepFailed {
                        step_id: StepId::new(DETECT_CONNECTION_MODE_STEP),
                        message: format!(
                            "Both HTTP and gRPC health checks failed for {shown}: HTTP: {http_err}, gRPC: {grpc_err}"
                        ),
                    });
                }
            }
        };

        if let Some(http2) = context
            .data
            .http2
            .filter(|_| host.registration_settings().upstream_http2)
        {
            info!(worker = %shown, http2, "resolved worker HTTP version");
        }
        context.data.connection_mode = Some(connection_mode);
        Ok(StepResult::Success)
    }
}

#[cfg(test)]
mod tests;
