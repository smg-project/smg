//! Runtime detection step.
//!
//! Tells which runtime a worker runs, over HTTP or gRPC.
//! - HTTP: the owner its model list names at `/v1/models`, then the paths
//!   that tell the runtimes apart, one a row of the runtime table. A live
//!   OpenAI-compatible server that matches no row is registered as the
//!   generic runtime rather than rejected: a gateway that stands in front of
//!   an engine answers so.
//! - gRPC: each runtime's health service in turn, in the table's order.

use std::time::Duration;

use async_trait::async_trait;
use futures::future::join_all;
use reqwest::Client;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{
    data::{WorkerKind, WorkerRegistrationWorkflowData},
    host::{require_host, WorkflowHost},
    model_list::ModelsResponse,
    reach::{grpc_health_answer, http_base_url},
    registration::DETECT_RUNTIME_STEP,
    runtimes::{listed, runtime_of_owner, HttpRuntime, GRPC_DETECTION_ORDER, HTTP_RUNTIMES},
};
use crate::{
    connection::engine::EngineConnector,
    identity::{
        basic::{ConnectionMode, RuntimeType},
        endpoint::{grpc_base_url, Endpoint},
    },
};

// ─── gRPC runtime detection ────────────────────────────────────────────────

/// Detect a gRPC worker's runtime by asking each runtime's health service in
/// turn, through `connector`, in the order of [`GRPC_DETECTION_ORDER`].
///
/// A runtime that answers its health RPC identifies itself whatever the
/// answer says about the engine: an engine still loading, or draining, is
/// that runtime all the same. Whether the engine is healthy enough to admit
/// is the connection-mode detection's question, asked before this step.
async fn detect_grpc_runtime(
    connector: &dyn EngineConnector,
    url: &str,
    timeout_secs: u64,
) -> Result<RuntimeType, String> {
    let grpc_url = grpc_base_url(url);
    let mut errors = Vec::new();

    for runtime in GRPC_DETECTION_ORDER {
        match grpc_health_answer(connector, &grpc_url, timeout_secs, runtime).await {
            Ok(_) => return Ok(runtime),
            Err(error) => errors.push(format!("{runtime}: {error}")),
        }
    }

    Err(format!(
        "gRPC backend detection failed for {} (tried {}): {}",
        Endpoint::redacted_str(url),
        listed(&GRPC_DETECTION_ORDER),
        errors.join("; ")
    ))
}

// ─── HTTP runtime detection ────────────────────────────────────────────────

/// Outcome of probing `/v1/models`.
enum ModelsProbe {
    /// `owned_by` named a runtime of the table.
    Detected(RuntimeType),
    /// The endpoint answered like an OpenAI server, but `owned_by` names no
    /// runtime of the table (a gateway that stands in front of an engine
    /// reports an owner of its own, say).
    UnrecognizedOwnedBy(Option<String>),
    /// The endpoint could not be probed: unreachable, non-2xx, bad JSON, or
    /// no models loaded yet.
    Failed(String),
}

/// Probe `/v1/models` and classify the runtime from its `owned_by` field.
async fn probe_models_endpoint(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> ModelsProbe {
    let models_url = format!("{}/v1/models", http_base_url(url));
    // The address as the probe's failures name it: a userinfo in it masked
    // (they end in the step's failure message and its lines).
    let shown = Endpoint::redacted_str(&models_url);

    let mut req = client
        .get(&models_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = match req.send().await {
        Ok(response) => response,
        Err(e) => {
            return ModelsProbe::Failed(format!("failed to reach {shown}: {}", e.without_url()))
        }
    };

    if !response.status().is_success() {
        return ModelsProbe::Failed(format!("{shown} returned status {}", response.status()));
    }

    let models: ModelsResponse = match response.json().await {
        Ok(models) => models,
        Err(e) => {
            return ModelsProbe::Failed(format!(
                "failed to parse {shown} response: {}",
                e.without_url()
            ))
        }
    };

    let Some(first_model) = models.data.first() else {
        return ModelsProbe::Failed(format!("{shown} returned an empty data array"));
    };

    match first_model.owned_by.as_deref().and_then(runtime_of_owner) {
        Some(runtime) => ModelsProbe::Detected(runtime),
        None => ModelsProbe::UnrecognizedOwnedBy(first_model.owned_by.clone()),
    }
}

/// Probe one path of the backend: `Ok` when it is reached and answers with
/// a success status.
async fn probe_path(
    url: &str,
    path: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> Result<(), String> {
    let probe_url = format!("{}{path}", http_base_url(url));

    let mut req = client
        .get(&probe_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req.send().await.map_err(|e| {
        format!(
            "Failed to reach {}: {}",
            Endpoint::redacted_str(&probe_url),
            e.without_url()
        )
    })?;

    if !response.status().is_success() {
        return Err(format!("{path} returned {}", response.status()));
    }

    Ok(())
}

/// Detect HTTP runtime type.
///
/// Strategy:
/// 1. Primary: `GET /v1/models` → the `owned_by` field names a row of the
///    runtime table
/// 2. Fallback: probe the path of every row in parallel; the earliest row
///    whose path answers is the runtime
/// 3. Last resort: a live OpenAI-compatible server matching no row is
///    registered as `generic` rather than rejected
async fn detect_http_runtime(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> Result<RuntimeType, String> {
    // Strategy 1: /v1/models owned_by
    let models_probe = probe_models_endpoint(url, timeout_secs, client, api_key).await;
    match &models_probe {
        ModelsProbe::Detected(runtime) => {
            debug!("Detected HTTP backend via /v1/models owned_by: {}", runtime);
            return Ok(*runtime);
        }
        ModelsProbe::UnrecognizedOwnedBy(owned_by) => {
            debug!(
                "No known engine in /v1/models owned_by ({:?}), trying fallback probes",
                owned_by
            );
        }
        ModelsProbe::Failed(e) => {
            debug!(
                "Could not detect backend via /v1/models, trying fallback: {}",
                e
            );
        }
    }

    // Strategy 2: probe every row's path in parallel. A later row's path is
    // not its runtime's alone: an earlier row's runtime may serve it too. So
    // the earliest row whose path answers decides, whatever the later rows
    // answer.
    let probes = join_all(
        HTTP_RUNTIMES
            .iter()
            .map(|row| probe_path(url, row.path, timeout_secs, client, api_key)),
    )
    .await;
    let answered: Vec<&HttpRuntime> = HTTP_RUNTIMES
        .iter()
        .zip(&probes)
        .filter(|(_, probe)| probe.is_ok())
        .map(|(row, _)| row)
        .collect();
    if let Some(row) = answered.first() {
        if let Some(later) = answered.get(1) {
            debug!(
                "Both {} and {} succeeded for {}; {} is {}-specific, detecting as {}",
                row.path,
                later.path,
                Endpoint::redacted_str(url),
                row.path,
                row.runtime,
                row.runtime
            );
        } else if row.path != HTTP_RUNTIMES[0].path {
            debug!(
                "Detected HTTP backend via {} (no {}): {}",
                row.path, HTTP_RUNTIMES[0].path, row.runtime
            );
        }
        return Ok(row.runtime);
    }

    match models_probe {
        // Strategy 3: the server speaks OpenAI (a live /v1/models with at least
        // one model) but matches no row: a gateway that stands in front of an
        // engine reports an owner of its own and serves no row's path.
        // Register it as a generic OpenAI-compatible backend instead of
        // rejecting a healthy worker.
        ModelsProbe::UnrecognizedOwnedBy(owned_by) => {
            let probed = HTTP_RUNTIMES
                .iter()
                .map(|row| format!("{} ({})", row.path, row.runtime))
                .collect::<Vec<_>>()
                .join(" nor ");
            warn!(
                worker = %Endpoint::redacted_str(url),
                owned_by = ?owned_by,
                "Could not identify backend engine: /v1/models responded but its \
                 owned_by is not a known engine, and neither {probed} is exposed; \
                 registering as `generic` OpenAI-compatible. Set runtime_type on \
                 the worker to override."
            );
            Ok(RuntimeType::Generic)
        }
        ModelsProbe::Failed(models_failure) => {
            let paths: String = HTTP_RUNTIMES
                .iter()
                .zip(probes)
                .map(|(row, probe)| format!("; {}: {}", row.path, probe.err().unwrap_or_default()))
                .collect();
            Err(format!(
                "Could not detect HTTP backend for {}: /v1/models: {models_failure}{paths}",
                Endpoint::redacted_str(url)
            ))
        }
        // Detected returned early above.
        ModelsProbe::Detected(runtime) => Ok(runtime),
    }
}

// ─── Step implementation ───────────────────────────────────────────────────

/// Detects the runtime a local worker runs, or takes the one its spec
/// declares.
///
/// Runs after the connection detection, and sets `detected_runtime_type` in
/// the workflow data for the steps behind it. A ZMQ link is shared by the
/// runtimes, so nothing on it tells one apart: a ZMQ worker whose spec
/// declares no runtime keeps none here, and its host settles it.
pub struct DetectRuntimeStep;

#[async_trait]
impl<H: WorkflowHost> StepExecutor<WorkerRegistrationWorkflowData<H>> for DetectRuntimeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRegistrationWorkflowData<H>>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let config = &context.data.config;
        let connection_mode =
            context.data.connection_mode.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("connection_mode".to_string())
            })?;
        let host = require_host(context.data.host.as_ref())?;

        let timeout = config
            .health
            .timeout_secs
            .unwrap_or_else(|| host.registration_settings().health_check_timeout_secs);

        // If runtime_type is explicitly configured, use it and skip detection
        let config_runtime = config.runtime_type;
        if config_runtime.is_specified() {
            debug!(
                "Using explicitly configured runtime type: {} for {}",
                config_runtime,
                Endpoint::redacted_str(&config.url)
            );
            context.data.detected_runtime_type = Some(config_runtime.to_string());
            return Ok(StepResult::Success);
        }

        debug!(
            "Detecting backend for {} ({:?})",
            Endpoint::redacted_str(&config.url),
            connection_mode
        );

        let detected = match connection_mode {
            ConnectionMode::Http => {
                let client = context.data.http_client(DETECT_RUNTIME_STEP)?;
                detect_http_runtime(&config.url, timeout, &client, config.api_key.as_deref())
                    .await
                    .map_err(|e| WorkflowError::StepFailed {
                        step_id: StepId::new(DETECT_RUNTIME_STEP),
                        message: format!(
                            "HTTP backend detection failed for {}: {}",
                            Endpoint::redacted_str(&config.url),
                            e
                        ),
                    })?
            }
            ConnectionMode::Grpc => {
                let connector = host.engine_connector();
                detect_grpc_runtime(connector.as_ref(), &config.url, timeout)
                    .await
                    .map_err(|e| WorkflowError::StepFailed {
                        step_id: StepId::new(DETECT_RUNTIME_STEP),
                        message: format!(
                            "gRPC backend detection failed for {}: {}",
                            Endpoint::redacted_str(&config.url),
                            e
                        ),
                    })?
            }
            // The ZMQ handshake is shared across the runtimes, so the runtime
            // cannot be probed here. A declared `runtime_type` is honored by
            // the early return above; a worker that left it unspecified keeps
            // none, and its host settles it.
            ConnectionMode::Zmq => {
                debug!(
                    "No runtime is told apart over ZMQ for {}",
                    Endpoint::redacted_str(&config.url)
                );
                return Ok(StepResult::Success);
            }
        };

        debug!(
            "Detected backend: {} for {} ({:?})",
            detected,
            Endpoint::redacted_str(&config.url),
            connection_mode
        );
        context.data.detected_runtime_type = Some(detected.to_string());
        Ok(StepResult::Success)
    }
}

#[cfg(test)]
mod tests;
