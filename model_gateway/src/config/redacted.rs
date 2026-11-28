//! The effective configuration with every secret redacted: what
//! `GET /config` answers.
//!
//! Every serialized field of [`RouterConfig`] is classified here, once. A
//! plain field is copied as the configuration serializes it. A secret-bearing
//! field is rewritten: a key becomes its fingerprint, a worker or store URL
//! loses its credentials, the CA bundle its bytes. A field this module does
//! not name is left out of the view, and the test
//! `every_serialized_field_is_classified` fails on it, so a new field cannot
//! reach the view by omission. The TLS keys and the MCP configuration are
//! not serialized at all (`#[serde(skip)]` on [`RouterConfig`]).

use serde_json::{json, Map, Value};
use smg_data_connector::redacted_url;

use super::{RouterConfig, TenantApiKeyEntry};
use crate::{
    middleware::key_fingerprint,
    server::posture::{SecretSource, SecretSources},
    version,
};

/// The fields copied as the configuration serializes them: none carries a
/// credential.
const PLAIN: &[&str] = &[
    "connection_mode",
    "startup_worker_runtime_type",
    "zmq_engine_count",
    "policy",
    "cache_boundaries",
    "routing_key_override",
    "grouped_dp_sticky",
    "pd_pairing_mode",
    "host",
    "port",
    "health_check_port",
    "pd_readiness",
    "admin_port",
    "admin_host",
    "runtime_worker_threads",
    "max_payload_size",
    "max_buffered_request_bytes",
    "stream_body_stall_timeout_secs",
    "header_read_timeout_secs",
    "idle_connection_timeout_secs",
    "max_connections",
    "enable_responses_websocket",
    "admin_max_connections",
    "request_timeout_secs",
    "grpc_request_timeout_secs",
    "sse_keepalive_secs",
    "upstream_pool_idle_timeout_secs",
    "worker_startup_timeout_secs",
    "worker_startup_delay_secs",
    "worker_startup_check_interval_secs",
    "zmq_startup_timeout_secs",
    "job_queue_capacity",
    "job_queue_concurrency",
    "load_monitor_interval_secs",
    "load_monitor_interval_ms",
    "worker_stall_secs",
    "worker_wedge_secs",
    "worker_stale_secs",
    "worker_warmup_secs",
    "worker_warmup_share",
    "worker_warmup_blocks",
    "worker_warmup_thin_ratio",
    "worker_warmup_divert_every",
    "pd_admission_wait_secs",
    "disable_load_monitoring",
    "worker_overload_protection",
    "worker_overload_waiting_requests",
    "worker_overload_token_usage",
    "worker_overload_shed",
    "engine_metrics",
    "multimodal_tensor_transport",
    "multimodal_shm_min_bytes",
    "multimodal_max_inflight_bytes",
    "mm_per_request_image_limit",
    "mm_processing",
    "mm_pixel_cache_mb",
    "mm_pixel_rdma",
    "mm_allowed_media_domains",
    "mm_allow_private_media_hosts",
    "mm_turbojpeg",
    "mm_video_fetch_timeout_ms",
    "mm_fetch_timeout_ms",
    "rdma_listen_ip",
    "rdma_slot_ttl_s",
    "log_mm_timing",
    "dp_aware",
    "dp_minimum_tokens_scheduler",
    "discovery",
    "metrics",
    "trace_config",
    "log_dir",
    "log_level",
    "access_log_fields",
    "jemalloc_prof_dir",
    "request_id_headers",
    "storage_context_headers",
    "tenant_resolution",
    "max_concurrent_requests",
    "queue_size",
    "queue_timeout_secs",
    "prefill_max_inflight_requests_per_worker",
    "prefill_queue_size",
    "prefill_queue_timeout_secs",
    "rate_limit_tokens_per_second",
    "max_background_responses",
    "priority_scheduler_enabled",
    "priority_scheduler_default_max_class",
    "priority_scheduler_config",
    "priority_scheduler_tenant_metric_top_n",
    "tenant_rate_limit_enabled",
    "tenant_rate_limit_config",
    "cors_allowed_origins",
    "retry",
    "circuit_breaker",
    "disable_retries",
    "disable_circuit_breaker",
    "health_check",
    "enable_igw",
    "rl",
    "model_path",
    "tokenizer_path",
    "chat_template",
    "model_aliases",
    "disable_tokenizer_autoload",
    "parser",
    "history_backend",
    "memory",
    "reasoning_parser",
    "tool_call_parser",
    "tokenizer_cache",
    "upstream_http2",
    "enable_wasm",
    "storage_hook_wasm_path",
    "grand_central",
    "policy_spec_path",
];

/// The fields `redacted` rewrites before they leave, as the classification
/// test reads them: every serialized field is in one of the two tables.
#[cfg(test)]
const REDACTED: &[&str] = &[
    "api_key",
    "tenant_api_keys",
    "mode",
    "postgres",
    "redis",
    "oracle",
    "ca_certificates",
];

/// The keys in force, as the view names them: the shared key by its
/// fingerprint, each tenant key by its tenant and fingerprint, and how many
/// control-plane entries are in force. The start's configuration gives the
/// first record ([`KeysInForce::of`]); a reload of the key files gives each
/// later one ([`crate::secrets::SecretReloader::keys_in_force`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct KeysInForce {
    /// The shared serving key's fingerprint, if one is in force.
    pub api_key: Option<String>,
    /// Each tenant key in force: the tenant and the key's fingerprint.
    pub tenant_api_keys: Vec<(String, String)>,
    /// The control-plane entries in force, `None` when no set is held.
    pub control_plane_entries: Option<usize>,
}

impl KeysInForce {
    /// The record of `api_key`, `tenant_api_keys` and a control-plane set
    /// of `control_plane_entries` entries.
    pub(crate) fn of(
        api_key: Option<&str>,
        tenant_api_keys: &[TenantApiKeyEntry],
        control_plane_entries: Option<usize>,
    ) -> Self {
        Self {
            api_key: api_key.map(key_fingerprint),
            tenant_api_keys: tenant_api_keys
                .iter()
                .map(|entry| (entry.tenant_id.clone(), key_fingerprint(&entry.key)))
                .collect(),
            control_plane_entries,
        }
    }
}

/// The view: the build identity, where this start read its secrets with the
/// entries in force, and the configuration with its secrets redacted and
/// its keys named by the record in force.
pub(crate) fn effective(
    config: &RouterConfig,
    sources: &SecretSources,
    keys: &KeysInForce,
) -> serde_json::Result<Value> {
    Ok(json!({
        "build": build_identity(),
        "secret_sources": secret_sources(sources, keys),
        "config": redacted(config, keys)?,
    }))
}

/// The binary that answers: its version and how it was built.
fn build_identity() -> Value {
    json!({
        "version": version::VERSION,
        "git_commit": version::GIT_COMMIT,
        "git_branch": version::GIT_BRANCH,
        "git_status": version::GIT_STATUS,
        "build_time": version::BUILD_TIME,
        "build_mode": version::BUILD_MODE,
        "target": version::TARGET_TRIPLE,
        "rustc": version::RUSTC_VERSION,
    })
}

/// The sources record in the words of the start-up line: `flag`, `env`,
/// `file` or `launcher` per secret, `null` for a secret that was not given.
/// The entry counts are the sets in force, which a reload of the key files
/// moves.
fn secret_sources(sources: &SecretSources, keys: &KeysInForce) -> Value {
    json!({
        "api_key": sources.api_key.map(SecretSource::as_str),
        "tenant_api_keys": sources.tenant_api_keys.map(|(source, _)| {
            json!({ "source": source.as_str(), "entries": keys.tenant_api_keys.len() })
        }),
        "control_plane_api_keys": sources.control_plane_api_keys.map(|(source, entries)| {
            let entries = keys.control_plane_entries.unwrap_or(entries);
            json!({ "source": source.as_str(), "entries": entries })
        }),
        "postgres_db_url": sources.postgres_db_url.map(SecretSource::as_str),
        "redis_url": sources.redis_url.map(SecretSource::as_str),
    })
}

/// The configuration's serialized fields: the plain ones as they are, the
/// secret-bearing ones rewritten (the keys by the record in force), an
/// unclassified one left out.
pub(crate) fn redacted(config: &RouterConfig, keys: &KeysInForce) -> serde_json::Result<Value> {
    let Value::Object(fields) = serde_json::to_value(config)? else {
        // A struct serializes to an object; nothing else carries a field.
        return Ok(Value::Null);
    };
    let mut view = Map::with_capacity(fields.len());
    for (field, value) in fields {
        let value = match field.as_str() {
            "api_key" => json!(keys.api_key),
            "tenant_api_keys" => Value::Array(
                keys.tenant_api_keys
                    .iter()
                    .map(|(tenant_id, fingerprint)| {
                        json!({ "tenant_id": tenant_id, "fingerprint": fingerprint })
                    })
                    .collect(),
            ),
            "mode" => serde_json::to_value(config.mode.redacted())?,
            "postgres" => with_masked_url(value, "db_url"),
            "redis" => with_masked_url(value, "url"),
            "oracle" => without(value, "password"),
            "ca_certificates" => json!(config.ca_certificates.len()),
            plain if PLAIN.contains(&plain) => value,
            _ => continue,
        };
        view.insert(field, value);
    }
    Ok(Value::Object(view))
}

/// `store` with its `url_field` masked the way the data connector masks it in
/// its own log lines: the userinfo password and a `password` query pair.
fn with_masked_url(mut store: Value, url_field: &str) -> Value {
    if let Some(url) = store.get_mut(url_field) {
        if let Some(plain) = url.as_str() {
            *url = Value::String(redacted_url(plain));
        }
    }
    store
}

/// `store` without `field`: the Oracle password, which the store's own
/// `Debug` leaves out the same way.
fn without(mut store: Value, field: &str) -> Value {
    if let Some(fields) = store.as_object_mut() {
        fields.remove(field);
    }
    store
}

#[cfg(test)]
#[path = "../../tests/unit/config/redacted/tests.rs"]
mod tests;
