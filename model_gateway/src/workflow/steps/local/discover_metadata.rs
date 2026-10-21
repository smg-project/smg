//! Metadata discovery step for local workers.

use std::{collections::HashMap, time::Duration};

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    routers::grpc::client::{flat_labels, GrpcClient},
    worker::{
        sampling_defaults::SamplingDefaults, ConnectionMode, DEFAULT_SAMPLING_PARAMS_LABEL,
        UNKNOWN_MODEL_ID,
    },
    workflow::{
        data::{WorkerKind, WorkerWorkflowData},
        steps::{
            local::create_worker::resolve_model_id,
            util::{grpc_base_url, http_base_url},
        },
    },
};

/// Per-request deadline for metadata fetches.
const METADATA_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// HTTP response structs (sglang /server_info, /model_info; vllm /v1/models)
// ---------------------------------------------------------------------------

/// SGLang `/server_info` response — curated subset of the full response (~800 fields).
/// Uses `deny_unknown_fields = false` (the default) so extra fields are silently ignored.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerInfo {
    #[serde(alias = "model")]
    pub model_id: Option<String>,
    pub model_path: Option<String>,
    pub served_model_name: Option<String>,
    pub tp_size: Option<usize>,
    pub dp_size: Option<usize>,
    pub attn_dp_size: Option<usize>,
    pub pp_size: Option<usize>,
    pub load_balance_method: Option<String>,
    pub disaggregation_mode: Option<String>,
    pub version: Option<String>,
    pub is_embedding: Option<bool>,
    pub context_length: Option<usize>,
    pub max_total_tokens: Option<usize>,
    /// Per-instance concurrency cap. CLI flag `--max-running-requests` on SGLang.
    /// Already extracted by the SGLang gRPC label pipeline; surfacing it here
    /// closes the HTTP-only path so capacity-aware consumers (e.g. WorkerCapacity)
    /// see the same label regardless of transport.
    pub max_running_requests: Option<usize>,
    /// Whether the engine splits `max_running_requests` across its
    /// attention-DP workers (`attn_dp_size` above carries that width as its
    /// newer releases resolve it): under `--dp-aware` each rank worker then
    /// gets its share of the window (`create_worker::labels_for_rank`), on
    /// this path as on the gRPC one.
    pub enable_dp_attention: Option<bool>,
    pub weight_version: Option<String>,
}

/// SGLang `/model_info` response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelInfo {
    pub model_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub is_generation: Option<bool>,
    pub has_image_understanding: Option<bool>,
    pub has_audio_understanding: Option<bool>,
    pub model_type: Option<String>,
    pub architectures: Option<Vec<String>>,
}

/// Single entry from `/v1/models` (shared by sglang and vllm).
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ModelsResponseEntry {
    pub owned_by: Option<String>,
    pub id: Option<String>,
    pub root: Option<String>,
    pub max_model_len: Option<usize>,
}

/// `/v1/models` response wrapper.
#[derive(Debug, Clone, Deserialize)]
pub(super) struct ModelsResponse {
    pub data: Vec<ModelsResponseEntry>,
}

/// vLLM `/version` response.
#[derive(Debug, Deserialize)]
struct VersionResponse {
    version: String,
}

// ---------------------------------------------------------------------------
// HTTP fetchers
// ---------------------------------------------------------------------------

/// GET JSON with optional bearer auth, with 404 fallback to `/get_<endpoint>`.
/// Why a metadata endpoint gave no answer: the engine does not offer it (a
/// 404 on every spelling), or the read failed (transport, timeout, a non-2xx
/// other than 404, an unparsable body). Registration tolerates both; a strict
/// refresh tolerates only the first.
#[derive(Debug)]
enum MetadataError {
    NotOffered,
    Failed(String),
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotOffered => f.write_str("not offered (404)"),
            Self::Failed(reason) => f.write_str(reason),
        }
    }
}

/// GET JSON with optional bearer auth, with 404 fallback to `/get_<endpoint>`.
async fn get_json_with_fallback<T: serde::de::DeserializeOwned>(
    client: &Client,
    base_url: &str,
    endpoint: &str,
    api_key: Option<&str>,
) -> Result<T, String> {
    get_json_with_fallback_classified(client, base_url, endpoint, api_key)
        .await
        .map_err(|e| e.to_string())
}

async fn get_json_with_fallback_classified<T: serde::de::DeserializeOwned>(
    client: &Client,
    base_url: &str,
    endpoint: &str,
    api_key: Option<&str>,
) -> Result<T, MetadataError> {
    let url = format!("{base_url}/{endpoint}");
    let mut req = client.get(&url).timeout(METADATA_TIMEOUT);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| MetadataError::Failed(format!("Failed to connect to {url}: {e}")))?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        // Fallback to deprecated /get_<endpoint> prefix
        warn!("'/{endpoint}' returned 404, falling back to deprecated '/get_{endpoint}'");
        let old_url = format!("{base_url}/get_{endpoint}");
        let mut req = client.get(&old_url).timeout(METADATA_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| MetadataError::Failed(format!("Failed to connect to {old_url}: {e}")))?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(MetadataError::NotOffered);
        }
        if !resp.status().is_success() {
            return Err(MetadataError::Failed(format!(
                "status {} from {}",
                resp.status(),
                old_url
            )));
        }
        return resp
            .json::<T>()
            .await
            .map_err(|e| MetadataError::Failed(format!("Failed to parse {old_url}: {e}")));
    }

    if !response.status().is_success() {
        return Err(MetadataError::Failed(format!(
            "status {} from {}",
            response.status(),
            url
        )));
    }

    response
        .json::<T>()
        .await
        .map_err(|e| MetadataError::Failed(format!("Failed to parse {url}: {e}")))
}

async fn http_get_json_classified<T: serde::de::DeserializeOwned>(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> Result<T, MetadataError> {
    let mut req = client.get(url).timeout(METADATA_TIMEOUT);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| MetadataError::Failed(format!("Failed to connect to {url}: {e}")))?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(MetadataError::NotOffered);
    }
    if !resp.status().is_success() {
        return Err(MetadataError::Failed(format!(
            "status {} from {}",
            resp.status(),
            url
        )));
    }
    resp.json::<T>()
        .await
        .map_err(|e| MetadataError::Failed(format!("Failed to parse {url}: {e}")))
}

pub async fn get_server_info(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> Result<ServerInfo, String> {
    get_json_with_fallback(client, &http_base_url(url), "server_info", api_key).await
}

/// The labels an engine reports about itself, read the way registration
/// reads them: an HTTP engine's metadata endpoints (by runtime), or a gRPC
/// engine's model and server info. Callable again for a worker that is
/// already registered, so its labels can follow the engine when it comes
/// back into service or finishes loading. Returns the labels and, for a gRPC
/// engine, the runtime the probe confirmed. Registration reads tolerantly
/// (`strict == false`: a failed endpoint leaves a gap); a refresh reads
/// strictly, because a gap would read as labels the engine dropped: a read
/// that fails (transport, timeout, a non-2xx other than 404, an unparsable
/// body, an unknown gRPC error) is an error, while an endpoint the engine does
/// not offer (404 on every spelling, gRPC `Unimplemented`) is tolerated in
/// both modes.
pub(crate) async fn discover_engine_labels(
    client: &Client,
    url: &str,
    connection_mode: ConnectionMode,
    runtime_type: &str,
    api_key: Option<&str>,
    strict: bool,
) -> Result<(HashMap<String, String>, Option<String>), String> {
    match connection_mode {
        ConnectionMode::Http => {
            let labels = match runtime_type {
                "vllm" => fetch_vllm_http_metadata_checked(client, url, api_key, strict).await?,
                _ => fetch_sglang_http_metadata_checked(client, url, api_key, strict).await?,
            };
            Ok((labels, None))
        }
        ConnectionMode::Grpc => fetch_grpc_metadata(url, runtime_type, strict)
            .await
            .map(|(labels, runtime)| (labels, Some(runtime))),
        ConnectionMode::Zmq => Err(
            "a ZMQ engine reports its metadata at the handshake, not at an endpoint".to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Per-backend metadata fetchers
// ---------------------------------------------------------------------------

/// The tolerant read as a plain map, kept for the tests of the registration shape.
#[cfg(test)]
async fn fetch_sglang_http_metadata(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> HashMap<String, String> {
    fetch_sglang_http_metadata_checked(client, url, api_key, false)
        .await
        .unwrap_or_default()
}

/// The SGLang HTTP metadata. Registration tolerates an endpoint that fails
/// (`strict == false`: the gap is left); a refresh must not mistake a failed
/// read for an engine that reports less, so with `strict` every endpoint has
/// to answer or the read is an error.
async fn fetch_sglang_http_metadata_checked(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
    strict: bool,
) -> Result<HashMap<String, String>, String> {
    let base = http_base_url(url);
    let mut labels = HashMap::new();

    match get_json_with_fallback_classified::<ServerInfo>(client, &base, "server_info", api_key)
        .await
    {
        Ok(info) => labels.extend(flat_labels(&info)),
        Err(MetadataError::Failed(e)) if strict => return Err(format!("server_info: {e}")),
        Err(_) => {}
    }
    match get_json_with_fallback_classified::<ModelInfo>(client, &base, "model_info", api_key).await
    {
        Ok(info) => labels.extend(flat_labels(&info)),
        Err(MetadataError::Failed(e)) if strict => return Err(format!("model_info: {e}")),
        Err(_) => {}
    }

    // /v1/models gives us model identity and max_model_len when a compatible
    // local frontend does not expose the SGLang-specific metadata endpoints.
    match http_get_json_classified::<ModelsResponse>(client, &format!("{base}/v1/models"), api_key)
        .await
    {
        Ok(models) => {
            if let Some(m) = models.data.first() {
                if let Some(id) = m.id.as_ref().filter(|id| !id.is_empty()) {
                    labels
                        .entry("served_model_name".to_string())
                        .or_insert_with(|| id.clone());
                }
                if let Some(root) = m.root.as_ref().filter(|root| !root.is_empty()) {
                    labels
                        .entry("model_path".to_string())
                        .or_insert_with(|| root.clone());
                }
                if let Some(len) = m.max_model_len.filter(|&n| n > 0) {
                    labels
                        .entry("max_model_len".to_string())
                        .or_insert_with(|| len.to_string());
                }
            }
        }
        Err(MetadataError::Failed(e)) if strict => return Err(format!("/v1/models: {e}")),
        Err(_) => {}
    }

    Ok(labels)
}

/// The tolerant read as a plain map, kept for the tests of the registration shape.
#[cfg(test)]
async fn fetch_vllm_http_metadata(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> HashMap<String, String> {
    fetch_vllm_http_metadata_checked(client, url, api_key, false)
        .await
        .unwrap_or_default()
}

/// The vLLM HTTP metadata; `strict` as for the SGLang fetcher.
async fn fetch_vllm_http_metadata_checked(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
    strict: bool,
) -> Result<HashMap<String, String>, String> {
    let base = http_base_url(url);
    let mut labels = HashMap::new();

    // /v1/models — vLLM uses `root` as model_path, `id` as served_model_name
    match http_get_json_classified::<ModelsResponse>(client, &format!("{base}/v1/models"), api_key)
        .await
    {
        Ok(models) => {
            if let Some(m) = models.data.first() {
                if let Some(ref root) = m.root {
                    labels.insert("model_path".to_string(), root.clone());
                }
                if let Some(ref id) = m.id {
                    labels.insert("served_model_name".to_string(), id.clone());
                }
                if let Some(len) = m.max_model_len.filter(|&n| n > 0) {
                    labels.insert("max_model_len".to_string(), len.to_string());
                }
            }
        }
        Err(MetadataError::Failed(e)) if strict => return Err(format!("/v1/models: {e}")),
        Err(_) => {}
    }

    // /version
    match http_get_json_classified::<VersionResponse>(client, &format!("{base}/version"), api_key)
        .await
    {
        Ok(v) => {
            if !v.version.is_empty() {
                labels.insert("version".to_string(), v.version);
            }
        }
        Err(MetadataError::Failed(e)) if strict => return Err(format!("/version: {e}")),
        Err(_) => {}
    }

    Ok(labels)
}

/// Re-read the KV transfer engine id a gRPC engine reports, for a PD worker
/// that came back on the same address (#2491): a restarted engine process
/// carries a new id, and a handoff minted for the old one strands the decode.
///
/// Unlike registration discovery, a server-info failure is an error here
/// rather than a tolerated gap: the caller must tell a read that did not
/// complete (retry later) from an engine that reports no id (nothing to
/// retry).
pub(crate) async fn discover_grpc_kv_engine_id(
    url: &str,
    runtime_type: &str,
) -> Result<Option<String>, String> {
    let client = GrpcClient::connect(&grpc_base_url(url), runtime_type)
        .await
        .map_err(|e| format!("Failed to connect to gRPC: {e}"))?;
    let mut labels = client
        .get_server_info()
        .await
        .map_err(|e| format!("Failed to fetch gRPC server info: {e}"))?
        .to_labels();
    normalize_grpc_keys(&mut labels);
    Ok(labels.remove("kv_engine_id").filter(|id| !id.is_empty()))
}

/// The gRPC model and server info as labels. Registration tolerates a
/// server-info failure (`strict == false`: the model info alone is kept); a
/// refresh must not mistake it for an engine that reports less, so with
/// `strict` it is an error.
async fn fetch_grpc_metadata(
    url: &str,
    runtime_type: &str,
    strict: bool,
) -> Result<(HashMap<String, String>, String), String> {
    let grpc_url = grpc_base_url(url);

    let client = GrpcClient::connect(&grpc_url, runtime_type)
        .await
        .map_err(|e| format!("Failed to connect to gRPC: {e}"))?;

    let mut labels = client
        .get_model_info()
        .await
        .map_err(|e| format!("Failed to fetch gRPC model info: {e}"))?
        .to_labels();

    match client.get_server_info().await {
        Ok(info) => labels.extend(info.to_labels()),
        // An engine that does not implement the call offers nothing to read;
        // any other failure fails a strict read.
        Err(e) if strict && e.code() != tonic::Code::Unimplemented => {
            return Err(format!("Failed to fetch gRPC server info: {e}"));
        }
        Err(e) => warn!("Failed to fetch gRPC server info: {}", e),
    }

    normalize_grpc_keys(&mut labels);
    Ok((labels, runtime_type.to_string()))
}

/// The gateway's model (a directory or a Hub id), when this worker serves it: a
/// worker with no identity of its own takes the gateway's (as registration
/// does), one declaring a different model keeps its own defaults.
fn zmq_model(router_model: Option<String>, declared_model: &str) -> Option<String> {
    router_model.filter(|path| declared_model == UNKNOWN_MODEL_ID || declared_model == path)
}

/// Labels for a ZMQ worker from the gateway's model: the sampling defaults in
/// its `generation_config.json`. A local directory is read in place; a Hub
/// repo id is served from the HF cache the gateway's tokenizer loads from,
/// fetched into it when absent, within a bound that keeps registration short
/// on an offline gateway. A local file, a path that does not exist on this
/// host, or a model without the file yields none.
async fn zmq_model_labels(model: Option<&str>) -> HashMap<String, String> {
    let Some(model) = model else {
        return HashMap::new();
    };
    let local = std::path::Path::new(model);
    let path = if local.is_dir() {
        local.join("generation_config.json")
    } else if local.exists() || !is_hub_repo_id(model) {
        return HashMap::new();
    } else {
        let fetch = llm_tokenizer::hub::fetch_file(model, "generation_config.json");
        match tokio::time::timeout(HUB_FETCH_TIMEOUT, fetch).await {
            Ok(Ok(path)) => path,
            Ok(Err(e)) => {
                debug!("No generation_config.json for {model} via the Hub cache: {e}");
                return HashMap::new();
            }
            Err(_) => {
                warn!("Hub lookup of generation_config.json for {model} timed out; no sampling defaults");
                return HashMap::new();
            }
        }
    };
    sampling_defaults_label(&path).await
}

/// Bound on fetching `generation_config.json` from the Hub, inside the step's
/// own timeout so an unreachable Hub costs one wait, not the step's retries.
const HUB_FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// `owner/name`, or a bare legacy `name`: no leading `/`, `.` or `~` and at
/// most one separator, so a path meant for another host is never sent to the
/// Hub as a repo id.
fn is_hub_repo_id(model: &str) -> bool {
    !model.starts_with(['/', '.', '~'])
        && model.matches('/').count() <= 1
        && model.split('/').all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
}

/// The sampling-defaults label from one `generation_config.json`, or none.
async fn sampling_defaults_label(path: &std::path::Path) -> HashMap<String, String> {
    let mut labels = HashMap::new();
    match tokio::fs::read_to_string(path).await {
        Ok(raw) => match SamplingDefaults::canonical_json_from_str(&raw) {
            Ok(Some(canonical)) => {
                labels.insert(DEFAULT_SAMPLING_PARAMS_LABEL.to_string(), canonical);
            }
            Ok(None) => {}
            Err(e) => warn!("Ignoring {}: {}", path.display(), e),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => warn!("Could not read {}: {}", path.display(), e),
    }
    labels
}

/// Rename gRPC-specific keys to canonical names and strip transient state.
fn normalize_grpc_keys(labels: &mut HashMap<String, String>) {
    for &(from, to) in &[
        ("tensor_parallel_size", "tp_size"),
        // TokenSpeed reports no `tp_size`: `attn_tp_size` is its attention
        // TP width, the full width for a dense model and narrower than
        // `world_size` only under attention DP or CP. A `tp_size` the engine
        // does report wins, as for every canonical label below.
        ("attn_tp_size", "tp_size"),
        ("pipeline_parallel_size", "pp_size"),
        ("context_parallel_size", "cp_size"),
        ("data_parallel_size", "dp_size"),
        // vLLM's name for the KV page granularity SGLang calls `page_size`.
        ("block_size", "page_size"),
    ] {
        if let Some(val) = labels.remove(from) {
            labels.entry(to.to_string()).or_insert(val);
        }
    }
    for key in [
        "active_requests",
        "is_paused",
        "last_receive_timestamp",
        "uptime_seconds",
        "server_type",
    ] {
        labels.remove(key);
    }
    normalize_default_sampling_params_label(labels);
}

fn normalize_default_sampling_params_label(labels: &mut HashMap<String, String>) {
    let Some(raw) = labels.get(DEFAULT_SAMPLING_PARAMS_LABEL).cloned() else {
        return;
    };

    match SamplingDefaults::canonical_json_from_str(&raw) {
        Ok(Some(canonical)) => {
            labels.insert(DEFAULT_SAMPLING_PARAMS_LABEL.to_string(), canonical);
        }
        Ok(None) => {
            labels.remove(DEFAULT_SAMPLING_PARAMS_LABEL);
        }
        Err(e) => {
            warn!(
                error = %e,
                "Ignoring invalid default sampling params label"
            );
            labels.remove(DEFAULT_SAMPLING_PARAMS_LABEL);
        }
    }
}

// ---------------------------------------------------------------------------
// Step executor
// ---------------------------------------------------------------------------

pub struct DiscoverMetadataStep;

#[async_trait]
impl StepExecutor<WorkerWorkflowData> for DiscoverMetadataStep {
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

        debug!(
            "Discovering metadata for {} ({:?})",
            config.url, connection_mode
        );

        let (discovered_labels, detected_runtime) = match connection_mode {
            ConnectionMode::Http => {
                let runtime = context
                    .data
                    .detected_runtime_type
                    .as_deref()
                    .unwrap_or_else(|| {
                        warn!(
                            "No detected_runtime_type for {}, defaulting to sglang",
                            config.url
                        );
                        "sglang"
                    });
                let client = context.data.http_client("discover_metadata")?;
                discover_engine_labels(
                    &client,
                    &config.url,
                    ConnectionMode::Http,
                    runtime,
                    config.api_key.as_deref(),
                    false,
                )
                .await
            }
            ConnectionMode::Grpc => {
                let config_runtime = config.runtime_type.to_string();
                let runtime_type = context
                    .data
                    .detected_runtime_type
                    .as_deref()
                    .unwrap_or(&config_runtime);
                fetch_grpc_metadata(&config.url, runtime_type, false)
                    .await
                    .map(|(labels, rt)| (labels, Some(rt)))
            }
            // A ZMQ engine reports its geometry at the handshake, which runs
            // after this step; the one fact worth having earlier is the
            // model's own sampling defaults, read from the model the gateway
            // was started with (a local directory, or a Hub id through the
            // HF cache), for a worker that serves that model (one registered
            // for another model, by spec or label, must not inherit them).
            // The runtime stays `None`: the handshake is shared across
            // engines, so it cannot be probed here and the configured or
            // detected runtime is preserved.
            ConnectionMode::Zmq => {
                let router_model = context
                    .data
                    .app_context
                    .as_ref()
                    .and_then(|app| app.router_config.model_path.clone());
                let model = zmq_model(router_model, resolve_model_id(config, &config.labels));
                Ok((zmq_model_labels(model.as_deref()).await, None))
            }
        }
        .unwrap_or_else(|e| {
            warn!("Failed to fetch metadata for {}: {}", config.url, e);
            (HashMap::new(), None)
        });

        debug!(
            "Discovered {} labels for {}",
            discovered_labels.len(),
            config.url
        );
        context.data.discovered_labels = discovered_labels;
        if let Some(runtime) = detected_runtime {
            context.data.detected_runtime_type = Some(runtime);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn zmq_labels_carry_the_local_generation_config_sampling_defaults() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05, "max_new_tokens": 2048, "do_sample": true}"#,
        )
        .unwrap();
        let labels = zmq_model_labels(Some(dir.path().to_str().unwrap())).await;
        let defaults: serde_json::Value =
            serde_json::from_str(&labels[DEFAULT_SAMPLING_PARAMS_LABEL]).unwrap();
        assert_eq!(defaults["top_k"], 20);
        assert_eq!(defaults["temperature"], 0.7);
        assert!(
            defaults.get("max_new_tokens").is_none(),
            "length limits are not sampling defaults"
        );
        // Only a worker serving the gateway's model takes its defaults.
        assert_eq!(
            zmq_model(Some("/m/a".into()), UNKNOWN_MODEL_ID).as_deref(),
            Some("/m/a")
        );
        assert_eq!(
            zmq_model(Some("/m/a".into()), "/m/a").as_deref(),
            Some("/m/a")
        );
        assert_eq!(zmq_model(Some("/m/a".into()), "org/other"), None);
        assert_eq!(zmq_model(None, UNKNOWN_MODEL_ID), None);
        // No model, or a directory without the file: no label.
        assert!(zmq_model_labels(None).await.is_empty());
        let bare = tempfile::tempdir().unwrap();
        assert!(zmq_model_labels(Some(bare.path().to_str().unwrap()))
            .await
            .is_empty());
        // A local file, or a path that is no repo id and does not exist here
        // (an engine-side mount), never goes to the Hub: no label.
        let file = tempfile::NamedTempFile::new().unwrap();
        assert!(zmq_model_labels(Some(file.path().to_str().unwrap()))
            .await
            .is_empty());
        assert!(zmq_model_labels(Some("/models/not/mounted/here"))
            .await
            .is_empty());
        assert!(is_hub_repo_id("Qwen/Qwen3-0.6B"));
        assert!(is_hub_repo_id("gpt2"));
        for not_id in [
            "/models/qwen",
            "./qwen",
            "~/qwen",
            "a/b/c",
            "models--Qwen--Qwen3-0.6B/snapshots/x",
            "",
            "org/",
        ] {
            assert!(!is_hub_repo_id(not_id), "{not_id}");
        }
    }

    use super::*;

    /// Every engine's spelling of the parallelism widths folds into the
    /// canonical labels, and an existing canonical label is never overwritten.
    #[test]
    fn normalize_grpc_keys_canonicalises_every_parallelism_spelling() {
        let mut labels: HashMap<String, String> = [
            ("attn_tp_size", "2"),
            ("pipeline_parallel_size", "1"),
            ("data_parallel_size", "4"),
            ("context_parallel_size", "1"),
            ("block_size", "16"),
            ("uptime_seconds", "12.5"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        normalize_grpc_keys(&mut labels);
        assert_eq!(labels.get("tp_size").map(String::as_str), Some("2"));
        assert_eq!(labels.get("pp_size").map(String::as_str), Some("1"));
        assert_eq!(labels.get("dp_size").map(String::as_str), Some("4"));
        assert_eq!(labels.get("cp_size").map(String::as_str), Some("1"));
        assert!(!labels.contains_key("attn_tp_size"));
        assert_eq!(labels.get("page_size").map(String::as_str), Some("16"));
        assert!(!labels.contains_key("block_size"));
        assert!(!labels.contains_key("uptime_seconds"));

        // An engine that reports both keeps its own `tp_size`.
        let mut attn: HashMap<String, String> = [("tp_size", "8"), ("attn_tp_size", "2")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        normalize_grpc_keys(&mut attn);
        assert_eq!(attn.get("tp_size").map(String::as_str), Some("8"));
        assert!(!attn.contains_key("attn_tp_size"));

        let mut both: HashMap<String, String> = [("tp_size", "8"), ("tensor_parallel_size", "2")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        normalize_grpc_keys(&mut both);
        assert_eq!(both.get("tp_size").map(String::as_str), Some("8"));
    }

    #[expect(clippy::print_stderr)]
    fn dump_labels(title: &str, labels: &HashMap<String, String>) {
        eprintln!("\n=== {title} ({} labels) ===", labels.len());
        let mut keys: Vec<_> = labels.keys().collect();
        keys.sort();
        for key in keys {
            eprintln!("  {key}: {}", labels[key]);
        }
    }

    #[tokio::test]
    #[ignore = "needs a live engine on a local port; run with --ignored"]
    async fn test_sglang_http_metadata() {
        let labels = fetch_sglang_http_metadata(
            &crate::clients::http_client(),
            "http://0.0.0.0:30000",
            None,
        )
        .await;
        dump_labels("SGLang HTTP combined", &labels);
        assert!(labels.contains_key("model_path"));
        assert!(labels.contains_key("tokenizer_path"));
    }

    #[tokio::test]
    #[ignore = "needs a live engine on a local port; run with --ignored"]
    async fn test_vllm_http_metadata() {
        let labels =
            fetch_vllm_http_metadata(&crate::clients::http_client(), "http://0.0.0.0:20000", None)
                .await;
        dump_labels("vLLM HTTP", &labels);
        assert!(labels.contains_key("model_path"));
        assert!(labels.contains_key("version"));
    }

    #[tokio::test]
    #[ignore = "needs a live engine on a local port; run with --ignored"]
    async fn test_sglang_grpc_metadata() {
        let (labels, _) = fetch_grpc_metadata("grpc://0.0.0.0:30001", "sglang", false)
            .await
            .expect("grpc metadata");
        dump_labels("SGLang gRPC", &labels);
        assert!(labels.contains_key("model_path"));
    }

    #[tokio::test]
    #[ignore = "needs a live engine on a local port; run with --ignored"]
    async fn test_vllm_grpc_metadata() {
        let (labels, _) = fetch_grpc_metadata("grpc://0.0.0.0:20001", "vllm", false)
            .await
            .expect("grpc metadata");
        dump_labels("vLLM gRPC", &labels);
        assert!(!labels.is_empty());
    }

    #[test]
    fn test_sglang_server_info_surfaces_max_running_requests_label() {
        // Subset of an actual SGLang /server_info response. The full payload has
        // many more fields; `serde(deny_unknown_fields)` is off by default so
        // they're silently ignored.
        for (dp_size, attn_dp_size) in [(4, None), (1, Some(4))] {
            let body = serde_json::json!({
                "model_path": "Qwen/Qwen3-8B",
                "tp_size": 4,
                "dp_size": dp_size,
                "attn_dp_size": attn_dp_size,
                "max_running_requests": 256,
                "context_length": 32768,
            });
            let info: ServerInfo = serde_json::from_value(body).expect("deserialize ServerInfo");
            assert_eq!(info.max_running_requests, Some(256));
            assert_eq!(info.attn_dp_size, attn_dp_size);

            let labels = flat_labels(&info);
            assert_eq!(
                labels.get("max_running_requests").map(String::as_str),
                Some("256")
            );
            let dp_info = super::super::discover_dp::dp_info_from_labels(&labels).unwrap();
            assert_eq!(dp_info.dp_size, 4);
        }
    }

    /// The attention-DP split reaches the labels on this path too, so a
    /// dp-aware registration over HTTP gives each rank worker its share of
    /// the window as the gRPC path does.
    #[test]
    fn test_sglang_server_info_surfaces_the_attention_dp_split() {
        let body = serde_json::json!({
            "model_path": "Qwen/Qwen3-8B",
            "tp_size": 16,
            "dp_size": 8,
            "enable_dp_attention": true,
            "attn_dp_size": 8,
            "max_running_requests": 16384,
        });
        let info: ServerInfo = serde_json::from_value(body).expect("deserialize ServerInfo");
        let labels = flat_labels(&info);
        assert_eq!(
            labels.get("enable_dp_attention").map(String::as_str),
            Some("true")
        );
        assert_eq!(labels.get("attn_dp_size").map(String::as_str), Some("8"));

        // Absent fields make no label, so a rank worker of an engine that
        // reports neither keeps its window as it is.
        let plain: ServerInfo = serde_json::from_value(serde_json::json!({
            "model_path": "Qwen/Qwen3-8B",
            "tp_size": 1,
            "dp_size": 1,
            "max_running_requests": 256,
        }))
        .expect("deserialize ServerInfo");
        let labels = flat_labels(&plain);
        assert!(
            !labels.contains_key("enable_dp_attention") && !labels.contains_key("attn_dp_size"),
            "{labels:?}"
        );
    }

    #[test]
    fn test_sglang_server_info_max_running_requests_optional() {
        // Older SGLang versions or special configurations may omit the field.
        let body = serde_json::json!({
            "model_path": "Qwen/Qwen3-8B",
            "tp_size": 1,
        });
        let info: ServerInfo = serde_json::from_value(body).expect("deserialize ServerInfo");
        assert_eq!(info.max_running_requests, None);

        let labels = flat_labels(&info);
        assert!(!labels.contains_key("max_running_requests"));
    }

    #[tokio::test]
    async fn test_sglang_http_metadata_uses_models_identity() {
        use axum::{routing::get, Json, Router};
        use serde_json::json;
        use tokio::net::TcpListener;

        async fn models() -> Json<serde_json::Value> {
            Json(json!({
                "data": [{
                    "id": "test-model",
                    "max_model_len": 4096,
                    "object": "model",
                    "owned_by": "nvidia",
                    "root": "test-root"
                }],
                "object": "list"
            }))
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        #[expect(
            clippy::disallowed_methods,
            reason = "test-only mock /v1/models server; handle is aborted at test end"
        )]
        let server = tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/v1/models", get(models)))
                .await
                .unwrap();
        });

        let labels = fetch_sglang_http_metadata(
            &crate::clients::http_client(),
            &format!("http://{addr}"),
            None,
        )
        .await;
        server.abort();

        assert_eq!(
            labels.get("served_model_name").map(String::as_str),
            Some("test-model")
        );
        assert_eq!(
            labels.get("model_path").map(String::as_str),
            Some("test-root")
        );
        assert_eq!(
            labels.get("max_model_len").map(String::as_str),
            Some("4096")
        );
    }
}
