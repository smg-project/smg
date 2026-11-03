//! The administrative calls a worker forwards to its engine (flush the KV
//! cache, start and stop a profile): the HTTP form with its timeouts, the
//! backend-client form with its refusals, and the mapping of either outcome
//! to a [`WorkerResult`]. The `Worker` methods of `BasicWorker` call these.

use std::{sync::Arc, time::Duration};

use super::{WorkerError, WorkerResult};
use crate::routers::grpc::backend_client::BackendClient;

/// Timeout for worker HTTP `flush_cache` requests. Matches the gRPC
/// client's local flush deadline.
pub(super) const FLUSH_HTTP_TIMEOUT: Duration = Duration::from_secs(45);

/// Timeout for worker HTTP profile requests. Stopping a profile can take
/// a long time while the backend serializes large traces. Matches the
/// gRPC client's profile deadline.
pub(super) const PROFILE_HTTP_TIMEOUT: Duration = Duration::from_secs(630);

/// POST an admin endpoint on an HTTP worker and map the outcome to a
/// [`WorkerResult`].
pub(super) async fn admin_http_post(
    client: &reqwest::Client,
    url: String,
    api_key: Option<&String>,
    body: Option<serde_json::Value>,
    operation: &str,
    timeout: Duration,
) -> WorkerResult<()> {
    let mut req = client.post(&url).timeout(timeout);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    if let Some(body) = body {
        req = req.json(&body);
    }
    match req.send().await {
        Ok(resp) if resp.status().is_success() => Ok(()),
        Ok(resp) => Err(WorkerError::OperationFailed {
            url,
            operation: operation.to_string(),
            reason: format!("HTTP {}", resp.status()),
        }),
        Err(e) => Err(WorkerError::OperationFailed {
            url,
            operation: operation.to_string(),
            reason: e.to_string(),
        }),
    }
}

/// Unwrap the optional gRPC client for an admin op, erroring when absent.
fn require_backend_client(
    url: &str,
    operation: &str,
    client: Option<Arc<BackendClient>>,
) -> WorkerResult<Arc<BackendClient>> {
    client.ok_or_else(|| WorkerError::OperationFailed {
        url: url.to_string(),
        operation: operation.to_string(),
        reason: "no backend client available".to_string(),
    })
}

/// A ZMQ engine whose wire carries no control calls (vLLM's EngineCore, a
/// TokenSpeed scheduler that declares none) has no admin RPC: refused before
/// anything is sent.
fn zmq_admin_unsupported(url: &str, operation: &str) -> WorkerError {
    WorkerError::OperationFailed {
        url: url.to_string(),
        operation: operation.to_string(),
        reason: format!("{operation} is not supported over this engine's ZMQ wire"),
    }
}

/// The worker's backend client for an admin op, refused when the backend
/// answers none ([`zmq_admin_unsupported`]).
pub(super) fn admin_backend_client(
    url: &str,
    operation: &str,
    client: Option<Arc<BackendClient>>,
) -> WorkerResult<Arc<BackendClient>> {
    let client = require_backend_client(url, operation, client)?;
    if !client.carries_admin_calls() {
        return Err(zmq_admin_unsupported(url, operation));
    }
    Ok(client)
}

/// Map a gRPC admin-op outcome (`success` flag plus message) to a
/// [`WorkerResult`].
pub(super) fn admin_grpc_result(
    url: &str,
    operation: &str,
    result: Result<(bool, String), tonic::Status>,
) -> WorkerResult<()> {
    match result {
        Ok((true, _)) => Ok(()),
        Ok((false, message)) => Err(WorkerError::OperationFailed {
            url: url.to_string(),
            operation: operation.to_string(),
            reason: if message.is_empty() {
                "backend reported failure".to_string()
            } else {
                message
            },
        }),
        Err(status) => Err(WorkerError::OperationFailed {
            url: url.to_string(),
            operation: operation.to_string(),
            reason: status.to_string(),
        }),
    }
}
