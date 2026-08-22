//! Backend runtime detection step.
//!
//! Detects the runtime type for both HTTP and gRPC workers.
//! - HTTP: probes `/v1/models` (owned_by field), falls back to unique endpoints
//!   (`/version` → vllm, `/server_info` → sglang). A live OpenAI-compatible
//!   server matching neither fingerprint is registered as `generic` rather than
//!   rejected — e.g. the SMG gateway `tokenspeed serve` embeds in front of its
//!   engine, which reports `owned_by: "self_hosted"` (issue #2085).
//! - gRPC: tries sglang → vllm → trtllm → tokenspeed → mlx health checks sequentially.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::discover_metadata::ModelsResponse;
use crate::{
    worker::ConnectionMode,
    workflow::{
        data::{WorkerKind, WorkerWorkflowData},
        steps::util::{do_grpc_health_check, grpc_base_url, http_base_url},
    },
};

// ─── gRPC backend detection ────────────────────────────────────────────────

/// Detect gRPC backend by trying runtime-specific health checks sequentially.
///
/// If `runtime_hint` is provided (from explicit config), tries that first.
/// Otherwise tries sglang → vllm → trtllm → mlx.
async fn detect_grpc_backend(
    url: &str,
    timeout_secs: u64,
    runtime_hint: Option<&str>,
) -> Result<String, String> {
    let grpc_url = grpc_base_url(url);

    // If we have a hint, try it first (fast path)
    if let Some(hint) = runtime_hint {
        if do_grpc_health_check(&grpc_url, timeout_secs, hint)
            .await
            .is_ok()
        {
            return Ok(hint.to_string());
        }
    }

    // Try each runtime sequentially (most common first), skipping the hint we already tried
    for runtime in &["sglang", "vllm", "trtllm", "tokenspeed", "mlx"] {
        if Some(*runtime) == runtime_hint {
            continue;
        }
        if do_grpc_health_check(&grpc_url, timeout_secs, runtime)
            .await
            .is_ok()
        {
            return Ok((*runtime).to_string());
        }
    }

    Err(format!(
        "gRPC backend detection failed for {url} (tried sglang, vllm, trtllm, tokenspeed, mlx)"
    ))
}

// ─── HTTP backend detection ────────────────────────────────────────────────

/// Outcome of probing `/v1/models`.
enum ModelsProbe {
    /// `owned_by` matched a known engine.
    Detected(String),
    /// The endpoint answered like an OpenAI server, but `owned_by` does not
    /// name a known engine (e.g. a nested SMG gateway reports "self_hosted").
    UnrecognizedOwnedBy(Option<String>),
    /// The endpoint could not be probed: unreachable, non-2xx, bad JSON, or
    /// no models loaded yet.
    Failed(String),
}

/// Probe `/v1/models` and classify the backend from its `owned_by` field.
async fn probe_models_endpoint(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> ModelsProbe {
    let models_url = format!("{}/v1/models", http_base_url(url));

    let mut req = client
        .get(&models_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = match req.send().await {
        Ok(response) => response,
        Err(e) => return ModelsProbe::Failed(format!("failed to reach {models_url}: {e}")),
    };

    if !response.status().is_success() {
        return ModelsProbe::Failed(format!(
            "{models_url} returned status {}",
            response.status()
        ));
    }

    let models: ModelsResponse = match response.json().await {
        Ok(models) => models,
        Err(e) => {
            return ModelsProbe::Failed(format!("failed to parse {models_url} response: {e}"))
        }
    };

    let Some(first_model) = models.data.first() else {
        return ModelsProbe::Failed(format!("{models_url} returned an empty data array"));
    };

    match first_model.owned_by.as_deref() {
        Some("sglang" | "nvidia") => ModelsProbe::Detected("sglang".to_string()),
        Some("vllm") => ModelsProbe::Detected("vllm".to_string()),
        _ => ModelsProbe::UnrecognizedOwnedBy(first_model.owned_by.clone()),
    }
}

/// Probe vLLM's `/version` endpoint.
async fn try_vllm_version(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> Result<(), String> {
    let version_url = format!("{}/version", http_base_url(url));

    let mut req = client
        .get(&version_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to reach {version_url}: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("/version returned {}", response.status()));
    }

    Ok(())
}

/// Probe SGLang's `/server_info` endpoint.
async fn try_sglang_server_info(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> Result<(), String> {
    let info_url = format!("{}/server_info", http_base_url(url));

    let mut req = client
        .get(&info_url)
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to reach {info_url}: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("/server_info returned {}", response.status()));
    }

    Ok(())
}

/// Detect HTTP backend runtime type.
///
/// Strategy:
/// 1. Primary: `GET /v1/models` → check `owned_by` field
/// 2. Fallback: probe `/version` (vLLM) and `/server_info` (SGLang) in parallel
/// 3. Last resort: a live OpenAI-compatible server matching no engine
///    fingerprint is registered as `generic` rather than rejected
async fn detect_http_backend(
    url: &str,
    timeout_secs: u64,
    client: &Client,
    api_key: Option<&str>,
) -> Result<String, String> {
    // Strategy 1: /v1/models owned_by
    let models_probe = probe_models_endpoint(url, timeout_secs, client, api_key).await;
    match &models_probe {
        ModelsProbe::Detected(runtime) => {
            debug!("Detected HTTP backend via /v1/models owned_by: {}", runtime);
            return Ok(runtime.clone());
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

    // Strategy 2: probe unique endpoints in parallel.
    // /version is unique to vLLM. /server_info is NOT unique to SGLang — vLLM can
    // also expose it. So /version takes priority: if it succeeds, it's definitely vLLM
    // regardless of whether /server_info also succeeds. We only conclude SGLang if
    // /server_info succeeds and /version does not.
    let (vllm_result, sglang_result) = tokio::join!(
        try_vllm_version(url, timeout_secs, client, api_key),
        try_sglang_server_info(url, timeout_secs, client, api_key),
    );

    let vllm_failure = match vllm_result {
        Ok(()) => {
            if sglang_result.is_ok() {
                debug!(
                    "Both /version and /server_info succeeded for {}; /version is vLLM-specific, detecting as vllm",
                    url
                );
            }
            return Ok("vllm".to_string());
        }
        Err(e) => e,
    };
    let sglang_failure = match sglang_result {
        Ok(()) => {
            debug!("Detected HTTP backend via /server_info (no /version): sglang");
            return Ok("sglang".to_string());
        }
        Err(e) => e,
    };

    match models_probe {
        // Strategy 3: the server speaks OpenAI (a live /v1/models with at least
        // one model) but matches no engine fingerprint — e.g. a nested SMG
        // gateway such as the one `tokenspeed serve` embeds, which reports
        // owned_by "self_hosted" and exposes neither /version nor /server_info
        // (issue #2085). Register it as a generic OpenAI-compatible backend
        // instead of rejecting a healthy worker.
        ModelsProbe::UnrecognizedOwnedBy(owned_by) => {
            warn!(
                worker = %url,
                owned_by = ?owned_by,
                "Could not identify backend engine: /v1/models responded but its \
                 owned_by is not a known engine, and neither /version (vllm) nor \
                 /server_info (sglang) is exposed; registering as `generic` \
                 OpenAI-compatible. Set runtime_type on the worker to override."
            );
            Ok("generic".to_string())
        }
        ModelsProbe::Failed(models_failure) => Err(format!(
            "Could not detect HTTP backend for {url}: /v1/models: {models_failure}; \
             /version: {vllm_failure}; /server_info: {sglang_failure}"
        )),
        // Detected returned early above.
        ModelsProbe::Detected(runtime) => Ok(runtime),
    }
}

// ─── Step implementation ───────────────────────────────────────────────────

/// Step 2: Detect backend runtime type (sglang, vllm, trtllm, tokenspeed, mlx,
/// or `generic` for an unidentified OpenAI-compatible HTTP backend).
///
/// Runs after `detect_connection_mode` and before `discover_metadata`.
/// Sets `detected_runtime_type` in workflow data for all downstream steps.
pub struct DetectBackendStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DetectBackendStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        if context.data.worker_kind != Some(WorkerKind::Local) {
            return Ok(StepResult::Skip);
        }

        let config = &context.data.config;
        let connection_mode =
            context.data.connection_mode.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("connection_mode".to_string())
            })?;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        let timeout = config
            .health
            .timeout_secs
            .unwrap_or(app_context.router_config.health_check.timeout_secs);

        // If runtime_type is explicitly configured, use it and skip detection
        let config_runtime = config.runtime_type;
        if config_runtime.is_specified() {
            debug!(
                "Using explicitly configured runtime type: {} for {}",
                config_runtime, config.url
            );
            context.data.detected_runtime_type = Some(config_runtime.to_string());
            return Ok(StepResult::Success);
        }

        debug!(
            "Detecting backend for {} ({:?})",
            config.url, connection_mode
        );

        let detected = match connection_mode {
            ConnectionMode::Http => {
                let client = &app_context.client;
                detect_http_backend(&config.url, timeout, client, config.api_key.as_deref())
                    .await
                    .map_err(|e| WorkflowError::StepFailed {
                        step_id: wfaas::StepId::new("detect_backend"),
                        message: format!("HTTP backend detection failed for {}: {}", config.url, e),
                    })?
            }
            ConnectionMode::Grpc => detect_grpc_backend(&config.url, timeout, None)
                .await
                .map_err(|e| WorkflowError::StepFailed {
                    step_id: wfaas::StepId::new("detect_backend"),
                    message: format!("gRPC backend detection failed for {}: {}", config.url, e),
                })?,
            // A ZMQ EngineCore handshake is shared across engine runtimes, so the
            // runtime cannot be probed here. An explicit `runtime_type` (vllm,
            // sglang, tokenspeed, ...) is honored by the early return above; only a
            // worker that left it unspecified reaches this default of vLLM.
            ConnectionMode::Zmq => {
                warn!(
                    worker = %config.url,
                    "runtime_type unspecified for ZMQ worker; defaulting to vLLM \
                     EngineCore. A TokenSpeed engine must declare its runtime to \
                     speak the correct wire protocol: `--backend tokenspeed` for \
                     startup --worker-urls workers, or an explicit runtime_type \
                     on the worker API spec / YAML worker config."
                );
                "vllm".to_string()
            }
        };

        debug!(
            "Detected backend: {} for {} ({:?})",
            detected, config.url, connection_mode
        );
        context.data.detected_runtime_type = Some(detected);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use axum::{routing::get, Json, Router};
    use reqwest::Client;
    use serde_json::json;
    use tokio::net::TcpListener;

    use super::detect_http_backend;

    async fn serve_router(router: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        #[expect(
            clippy::disallowed_methods,
            reason = "test-only mock backend server; handle is aborted at test end"
        )]
        let handle = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (format!("http://{addr}"), handle)
    }

    fn models_route(payload: serde_json::Value) -> Router {
        Router::new().route(
            "/v1/models",
            get(move || {
                let payload = payload.clone();
                async move { Json(payload) }
            }),
        )
    }

    #[tokio::test]
    async fn http_detection_falls_back_to_generic_for_unrecognized_owned_by() {
        // A nested SMG gateway (e.g. started by `tokenspeed serve`) reports
        // owned_by "self_hosted" and exposes neither /version nor /server_info.
        let (url, server) = serve_router(models_route(json!({
            "object": "list",
            "data": [{"id": "kimi-k3-256k", "object": "model", "created": 0, "owned_by": "self_hosted"}]
        })))
        .await;

        let runtime = detect_http_backend(&url, 5, &Client::new(), None)
            .await
            .unwrap();
        server.abort();

        assert_eq!(runtime, "generic");
    }

    #[tokio::test]
    async fn http_detection_falls_back_to_generic_when_owned_by_missing() {
        let (url, server) = serve_router(models_route(json!({
            "object": "list",
            "data": [{"id": "some-model", "object": "model"}]
        })))
        .await;

        let runtime = detect_http_backend(&url, 5, &Client::new(), None)
            .await
            .unwrap();
        server.abort();

        assert_eq!(runtime, "generic");
    }

    #[tokio::test]
    async fn http_detection_prefers_version_probe_over_generic_fallback() {
        // Unrecognized owned_by but a live /version endpoint → vllm, not generic.
        let router = models_route(json!({
            "object": "list",
            "data": [{"id": "m", "object": "model", "owned_by": "self_hosted"}]
        }))
        .route(
            "/version",
            get(|| async { Json(json!({"version": "0.9.0"})) }),
        );
        let (url, server) = serve_router(router).await;

        let runtime = detect_http_backend(&url, 5, &Client::new(), None)
            .await
            .unwrap();
        server.abort();

        assert_eq!(runtime, "vllm");
    }

    #[tokio::test]
    async fn http_detection_fails_when_models_endpoint_unreachable() {
        // Bind then drop to get a port that refuses connections: a backend that
        // is not up must stay an error so the retryable step keeps polling.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let err = detect_http_backend(&format!("http://{addr}"), 1, &Client::new(), None)
            .await
            .unwrap_err();

        assert!(err.contains("Could not detect HTTP backend"), "got: {err}");
    }

    #[tokio::test]
    async fn http_detection_fails_when_models_data_empty() {
        // Server is up but no model is loaded yet — keep failing so the step
        // retries instead of registering a backend with nothing to serve.
        let (url, server) = serve_router(models_route(json!({"object": "list", "data": []}))).await;

        let result = detect_http_backend(&url, 5, &Client::new(), None).await;
        server.abort();

        result.unwrap_err();
    }

    #[tokio::test]
    async fn http_detection_maps_nvidia_owned_by_to_sglang() {
        let (url, server) = serve_router(models_route(json!({
            "object": "list",
            "data": [{"id": "test-model", "object": "model", "owned_by": "nvidia"}]
        })))
        .await;

        let runtime = detect_http_backend(&url, 5, &Client::new(), None)
            .await
            .unwrap();
        server.abort();

        assert_eq!(runtime, "sglang");
    }
}
