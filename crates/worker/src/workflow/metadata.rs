//! What a worker's engine reports about itself, read as labels: the metadata
//! routes of an HTTP worker, by the runtime table's row, and the model and
//! server info of a gRPC worker.

use std::{collections::HashMap, time::Duration};

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::{
    model_list::ModelsResponse,
    reach::http_base_url,
    runtimes::{http_runtime, MetadataRead},
};
use crate::{
    connection::engine::EngineConnector,
    identity::{
        basic::{ConnectionMode, RuntimeType},
        endpoint::{grpc_base_url, Endpoint},
        sampling_defaults::{SamplingDefaults, DEFAULT_SAMPLING_PARAMS_LABEL},
    },
    labels::flat_labels,
};

/// Per-request deadline for metadata fetches.
const METADATA_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// HTTP response structs (/server_info, /model_info, /version)
// ---------------------------------------------------------------------------

/// `/server_info` response — curated subset of the full response (~800 fields).
/// Uses `deny_unknown_fields = false` (the default) so extra fields are silently ignored.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct ServerInfo {
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
    /// Per-instance concurrency cap. The gRPC label read carries it too;
    /// surfacing it here closes the HTTP-only path, so capacity-aware
    /// consumers see the same label regardless of transport.
    pub max_running_requests: Option<usize>,
    /// Whether the engine splits `max_running_requests` across its
    /// attention-DP workers (`attn_dp_size` above carries that width as its
    /// newer releases resolve it): a host that registers one worker a rank
    /// then gives each its share of the window, on this path as on the
    /// gRPC one.
    pub enable_dp_attention: Option<bool>,
    pub weight_version: Option<String>,
}

/// `/model_info` response.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct ModelInfo {
    pub model_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub is_generation: Option<bool>,
    pub has_image_understanding: Option<bool>,
    pub has_audio_understanding: Option<bool>,
    pub model_type: Option<String>,
    pub architectures: Option<Vec<String>>,
}

// The optional SMG extension belongs only to the vLLM model-list read;
// info-route/SGLang model-list parsing keeps its existing schema.
#[derive(Debug, Deserialize)]
struct VllmModelsResponse {
    #[serde(flatten)]
    models: ModelsResponse,
    pairing_protocol: Option<String>,
}

/// `/version` response.
#[derive(Debug, Deserialize)]
struct VersionResponse {
    version: String,
}

// ---------------------------------------------------------------------------
// HTTP fetchers
// ---------------------------------------------------------------------------

/// Why a metadata endpoint gave no answer: the engine does not offer it (a
/// 404 on every spelling), or the read failed (transport, timeout, a non-2xx
/// other than 404, an unparsable body). Registration tolerates optional
/// failures; an explicitly invalid SMG extension fails in both modes.
#[derive(Debug)]
enum MetadataError {
    NotOffered,
    Failed(String),
    InvalidExtension(String),
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotOffered => f.write_str("not offered (404)"),
            Self::Failed(reason) | Self::InvalidExtension(reason) => f.write_str(reason),
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
    match http_get_json_classified(client, &format!("{base_url}/{endpoint}"), api_key).await {
        Err(MetadataError::NotOffered) => {
            // Fallback to deprecated /get_<endpoint> prefix
            warn!("'/{endpoint}' returned 404, falling back to deprecated '/get_{endpoint}'");
            http_get_json_classified(client, &format!("{base_url}/get_{endpoint}"), api_key).await
        }
        read => read,
    }
}

/// GET JSON with optional bearer auth under the metadata timeout: a 404 is
/// `NotOffered`, any other failure `Failed`, each text naming the address masked.
async fn http_get_response_classified(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> Result<reqwest::Response, MetadataError> {
    let shown = Endpoint::redacted_str(url);
    let mut req = client.get(url).timeout(METADATA_TIMEOUT);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.map_err(|e| {
        MetadataError::Failed(format!("Failed to connect to {shown}: {}", e.without_url()))
    })?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(MetadataError::NotOffered);
    }
    Ok(resp)
}

async fn http_get_json_classified<T: serde::de::DeserializeOwned>(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> Result<T, MetadataError> {
    let shown = Endpoint::redacted_str(url);
    let resp = http_get_response_classified(client, url, api_key).await?;
    if !resp.status().is_success() {
        return Err(MetadataError::Failed(format!(
            "status {} from {shown}",
            resp.status()
        )));
    }
    resp.json::<T>()
        .await
        .map_err(|e| MetadataError::Failed(format!("Failed to parse {shown}: {}", e.without_url())))
}

async fn http_get_vllm_models_classified(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> Result<VllmModelsResponse, MetadataError> {
    let shown = Endpoint::redacted_str(url);
    let resp = http_get_response_classified(client, url, api_key).await?;
    let status = resp.status();
    if !status.is_success() {
        if status == reqwest::StatusCode::BAD_GATEWAY
            && resp
                .json::<serde_json::Value>()
                .await
                .ok()
                .is_some_and(|body| {
                    body.pointer("/error/code")
                        .and_then(serde_json::Value::as_str)
                        == Some("smg_http_metadata_invalid")
                })
        {
            return Err(MetadataError::InvalidExtension(
                "smg_http_metadata_invalid response".to_string(),
            ));
        }
        return Err(MetadataError::Failed(format!(
            "status {status} from {shown}"
        )));
    }
    let bytes = resp.bytes().await.map_err(|e| {
        MetadataError::Failed(format!("Failed to read {shown}: {}", e.without_url()))
    })?;
    let body: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| MetadataError::Failed(format!("Failed to parse {shown}: {e}")))?;
    #[derive(Deserialize)]
    struct PairingExtension {
        pairing_protocol: Option<String>,
    }
    // Parse the field from the original wire so duplicate declarations cannot
    // be hidden by a JSON map's last-value-wins normalization.
    let extension = if body.get("pairing_protocol").is_some() {
        serde_json::from_slice::<PairingExtension>(&bytes).map_err(|e| {
            MetadataError::InvalidExtension(format!("Invalid pairing_protocol: {e}"))
        })?
    } else {
        PairingExtension {
            pairing_protocol: None,
        }
    };
    let declares_protocol = extension
        .pairing_protocol
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());
    serde_json::from_slice(&bytes).map_err(|e| {
        if declares_protocol {
            MetadataError::InvalidExtension(format!(
                "Invalid model list with pairing_protocol: {e}"
            ))
        } else {
            MetadataError::Failed(format!("Failed to parse {shown}: {e}"))
        }
    })
}

pub(super) async fn get_server_info(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> Result<ServerInfo, String> {
    get_json_with_fallback(client, &http_base_url(url), "server_info", api_key).await
}

/// The labels an engine reports about itself, read the way registration
/// reads them: an HTTP engine's metadata endpoints (by runtime), or a gRPC
/// engine's model and server info through `connector`. Callable again for a
/// worker that is
/// already registered, so its labels can follow the engine when it comes
/// back into service or finishes loading. Returns the labels and, for a gRPC
/// engine, the runtime the probe confirmed. Registration reads tolerantly
/// (`strict == false`: a failed endpoint leaves a gap); a refresh reads
/// strictly, because a gap would read as labels the engine dropped. An invalid
/// explicit SMG extension fails in either mode. A read
/// that fails (transport, timeout, a non-2xx other than 404, an unparsable
/// body, an unknown gRPC error) is an error, while an endpoint the engine does
/// not offer (404 on every spelling, gRPC `Unimplemented`) is tolerated in
/// both modes.
pub async fn discover_engine_labels(
    client: &Client,
    connector: &dyn EngineConnector,
    url: &str,
    connection_mode: ConnectionMode,
    runtime_type: &str,
    api_key: Option<&str>,
    strict: bool,
) -> Result<(HashMap<String, String>, Option<String>), String> {
    match connection_mode {
        ConnectionMode::Http => {
            // The row of the runtime says how its metadata is read; a runtime
            // with no row is read by the info routes.
            let labels = match http_runtime(runtime_type) {
                Some(row) if row.metadata == MetadataRead::ModelListAndPath => {
                    read_model_list_and_path(client, url, api_key, strict, row.path).await?
                }
                _ => read_info_routes(client, url, api_key, strict).await?,
            };
            Ok((labels, None))
        }
        ConnectionMode::Grpc => fetch_grpc_metadata(connector, url, runtime_type, strict)
            .await
            .map(|(labels, runtime)| (labels, Some(runtime))),
        ConnectionMode::Zmq => Err(
            "a ZMQ engine reports its metadata at the handshake, not at an endpoint".to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// The metadata reads
// ---------------------------------------------------------------------------

/// The metadata of the info routes: the server's, the model's, then the
/// model list for what they left open. Registration tolerates an endpoint
/// that fails
/// (`strict == false`: the gap is left); a refresh must not mistake a failed
/// read for an engine that reports less, so with `strict` every endpoint has
/// to answer or the read is an error.
async fn read_info_routes(
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
    // local frontend does not expose the info routes.
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

/// The metadata of the model list and of `path`, the row's own path, which
/// answers a version; `strict` as for the info routes.
async fn read_model_list_and_path(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
    strict: bool,
    path: &str,
) -> Result<HashMap<String, String>, String> {
    let mut labels = HashMap::new();
    read_model_list_and_path_into(client, url, api_key, strict, path, &mut labels).await?;
    Ok(labels)
}

// Keep validated model/protocol labels outside the optional-version future.
// This is the model-list metadata row (vLLM); the table still supplies its path.
pub(super) async fn read_model_list_and_path_into(
    client: &Client,
    url: &str,
    api_key: Option<&str>,
    strict: bool,
    path: &str,
    labels: &mut HashMap<String, String>,
) -> Result<(), String> {
    let base = http_base_url(url);

    // /v1/models: `root` is the model path, `id` the served model name
    match http_get_vllm_models_classified(client, &format!("{base}/v1/models"), api_key).await {
        Ok(models) => {
            if let Some(protocol) = models
                .pairing_protocol
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                labels.insert("pairing_protocol".to_string(), protocol.to_string());
            }
            if let Some(m) = models.models.data.first() {
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
        Err(MetadataError::InvalidExtension(e)) => return Err(format!("/v1/models: {e}")),
        Err(MetadataError::Failed(e)) if strict => return Err(format!("/v1/models: {e}")),
        Err(_) => {}
    }

    // The row's path
    match http_get_json_classified::<VersionResponse>(client, &format!("{base}{path}"), api_key)
        .await
    {
        Ok(v) => {
            if !v.version.is_empty() {
                labels.insert("version".to_string(), v.version);
            }
        }
        Err(MetadataError::Failed(e)) if strict => return Err(format!("{path}: {e}")),
        Err(_) => {}
    }

    Ok(())
}

/// The gRPC model and server info as labels. Registration tolerates a
/// server-info failure (`strict == false`: the model info alone is kept); a
/// refresh must not mistake it for an engine that reports less, so with
/// `strict` it is an error. The engine is reached through `connector`, as the
/// runtime `runtime_type` names.
pub(super) async fn fetch_grpc_metadata(
    connector: &dyn EngineConnector,
    url: &str,
    runtime_type: &str,
    strict: bool,
) -> Result<(HashMap<String, String>, String), String> {
    let grpc_url = grpc_base_url(url);

    let runtime: RuntimeType = runtime_type
        .parse()
        .map_err(|_| format!("Failed to connect to gRPC: Unknown runtime type: {runtime_type}"))?;
    let connection = connector
        .connect_grpc(&grpc_url, runtime)
        .await
        .map_err(|e| format!("Failed to connect to gRPC: {e}"))?;

    let mut labels = connection
        .model_labels()
        .await
        .map_err(|e| format!("Failed to fetch gRPC model info: {e}"))?;

    match connection.server_labels().await {
        Ok(server) => labels.extend(server),
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

/// Rename gRPC-specific keys to canonical names and strip transient state.
fn normalize_grpc_keys(labels: &mut HashMap<String, String>) {
    for &(from, to) in &[
        ("tensor_parallel_size", "tp_size"),
        // One runtime reports no `tp_size`: `attn_tp_size` is its attention
        // TP width, the full width for a dense model and narrower than
        // `world_size` only under attention DP or CP. A `tp_size` the engine
        // does report wins, as for every canonical label below.
        ("attn_tp_size", "tp_size"),
        ("pipeline_parallel_size", "pp_size"),
        ("context_parallel_size", "cp_size"),
        ("data_parallel_size", "dp_size"),
        // Another name for the KV page granularity, `page_size`.
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

#[cfg(test)]
mod tests;
