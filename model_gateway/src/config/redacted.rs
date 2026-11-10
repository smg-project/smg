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

use super::RouterConfig;
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

/// The view: the build identity, where this start read its secrets, and the
/// configuration with its secrets redacted.
pub(crate) fn effective(
    config: &RouterConfig,
    sources: &SecretSources,
) -> serde_json::Result<Value> {
    Ok(json!({
        "build": build_identity(),
        "secret_sources": secret_sources(sources),
        "config": redacted(config)?,
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

/// The sources record in the words of the start-up line: `flag`, `env` or
/// `file` per secret, `null` for a secret that was not given.
fn secret_sources(sources: &SecretSources) -> Value {
    json!({
        "api_key": sources.api_key.map(SecretSource::as_str),
        "tenant_api_keys": sources
            .tenant_api_keys
            .map(|(source, entries)| json!({ "source": source.as_str(), "entries": entries })),
        "control_plane_api_keys": sources
            .control_plane_api_keys
            .map(|(source, entries)| json!({ "source": source.as_str(), "entries": entries })),
        "postgres_db_url": sources.postgres_db_url.map(SecretSource::as_str),
        "redis_url": sources.redis_url.map(SecretSource::as_str),
    })
}

/// The configuration's serialized fields: the plain ones as they are, the
/// secret-bearing ones rewritten, an unclassified one left out.
pub(crate) fn redacted(config: &RouterConfig) -> serde_json::Result<Value> {
    let Value::Object(fields) = serde_json::to_value(config)? else {
        // A struct serializes to an object; nothing else carries a field.
        return Ok(Value::Null);
    };
    let mut view = Map::with_capacity(fields.len());
    for (field, value) in fields {
        let value = match field.as_str() {
            "api_key" => json!(config.api_key.as_deref().map(key_fingerprint)),
            "tenant_api_keys" => Value::Array(
                config
                    .tenant_api_keys
                    .iter()
                    .map(|entry| {
                        json!({
                            "tenant_id": entry.tenant_id,
                            "fingerprint": key_fingerprint(&entry.key),
                        })
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
mod tests {
    use smg_data_connector::{OracleConfig, PostgresConfig, RedisConfig};

    use super::*;
    use crate::config::{RoutingMode, TenantApiKeyEntry};

    /// Every serialized field of `RouterConfig`, read from its definition:
    /// the `pub` fields of the struct that serde does not skip.
    fn serialized_fields() -> Vec<String> {
        let source = include_str!("types.rs");
        let start = source
            .find("pub struct RouterConfig {")
            .expect("types.rs defines RouterConfig");
        let body = &source[start..];
        let end = body.find("\n}\n").expect("the struct ends");
        let mut fields = Vec::new();
        let mut skipped = false;
        for line in body[..end].lines() {
            let line = line.trim();
            if line.starts_with("#[serde(") {
                skipped |= line.contains("skip)") || line.contains("skip,");
            }
            if let Some((name, _)) = line.strip_prefix("pub ").and_then(|f| f.split_once(':')) {
                if !skipped {
                    fields.push(name.to_string());
                }
                skipped = false;
            }
        }
        fields
    }

    #[test]
    fn every_serialized_field_is_classified() {
        let fields = serialized_fields();
        assert!(fields.len() > 100, "the struct was read: {fields:?}");

        let unclassified: Vec<_> = fields
            .iter()
            .filter(|f| !PLAIN.contains(&f.as_str()) && !REDACTED.contains(&f.as_str()))
            .collect();
        assert!(
            unclassified.is_empty(),
            "classify these fields of RouterConfig in config/redacted.rs, plain or redacted: {unclassified:?}"
        );
        let gone: Vec<_> = PLAIN
            .iter()
            .chain(REDACTED)
            .filter(|f| !fields.iter().any(|g| g == *f))
            .collect();
        assert!(
            gone.is_empty(),
            "these classified fields no longer exist on RouterConfig: {gone:?}"
        );
        let both: Vec<_> = PLAIN.iter().filter(|f| REDACTED.contains(f)).collect();
        assert!(
            both.is_empty(),
            "a field is plain or redacted, not both: {both:?}"
        );
    }

    /// A configuration with a secret in every place one can be.
    fn configuration_with_every_secret() -> RouterConfig {
        let mut config = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: vec![(
                    "http://prefill-user:prefill-secret@prefill:8000".to_string(),
                    Some(8998),
                )],
                decode_urls: vec!["http://decode-user:decode-secret@decode:8001".to_string()],
                prefill_policy: None,
                decode_policy: None,
            },
            ..RouterConfig::default()
        };
        config.api_key = Some("shared-secret-key".to_string());
        config.tenant_api_keys = vec![TenantApiKeyEntry {
            tenant_id: "team-red".to_string(),
            key: "red-tenant-secret".to_string(),
        }];
        config.postgres = Some(
            serde_json::from_value::<PostgresConfig>(json!({
                "db_url": "postgres://pg-user:pg-secret@db:5432/smg?password=pg-query-secret",
                "pool_max": 4,
            }))
            .expect("a postgres config"),
        );
        config.redis = Some(
            serde_json::from_value::<RedisConfig>(
                json!({ "url": "redis://:redis-secret@cache:6379/0" }),
            )
            .expect("a redis config"),
        );
        config.oracle = Some(
            serde_json::from_value::<OracleConfig>(json!({
                "connect_descriptor": "tcps://db:1522/svc",
                "username": "ora-user",
                "password": "oracle-secret",
            }))
            .expect("an oracle config"),
        );
        config.ca_certificates = vec![b"-----BEGIN CERTIFICATE-----".to_vec()];
        config
    }

    #[test]
    fn the_view_carries_none_of_the_secrets_and_names_each_by_its_mask() {
        let config = configuration_with_every_secret();

        let view = redacted(&config).expect("the view serializes");
        let text = view.to_string();

        for secret in [
            "prefill-secret",
            "decode-secret",
            "shared-secret-key",
            "red-tenant-secret",
            "pg-secret",
            "pg-query-secret",
            "redis-secret",
            "oracle-secret",
            "BEGIN CERTIFICATE",
        ] {
            assert!(!text.contains(secret), "{secret} reached the view: {text}");
        }
        assert_eq!(view["api_key"], json!(key_fingerprint("shared-secret-key")));
        assert_eq!(
            view["tenant_api_keys"],
            json!([{ "tenant_id": "team-red", "fingerprint": key_fingerprint("red-tenant-secret") }])
        );
        assert_eq!(view["mode"]["type"], json!("prefill_decode"));
        assert_eq!(
            view["mode"]["prefill_urls"],
            json!([["http://***@prefill:8000", 8998]])
        );
        assert_eq!(
            view["mode"]["decode_urls"],
            json!(["http://***@decode:8001"])
        );
        assert_eq!(
            view["postgres"]["db_url"],
            json!("postgres://pg-user:****@db:5432/smg?password=****")
        );
        assert_eq!(view["postgres"]["pool_max"], json!(4));
        assert_eq!(view["redis"]["url"], json!("redis://:****@cache:6379/0"));
        assert_eq!(view["oracle"]["username"], json!("ora-user"));
        assert!(
            view["oracle"].get("password").is_none(),
            "the oracle password is left out: {}",
            view["oracle"]
        );
        assert_eq!(view["ca_certificates"], json!(1));
    }

    #[test]
    fn plain_fields_are_copied_as_the_configuration_serializes_them() {
        let config = RouterConfig {
            host: "0.0.0.0".to_string(),
            port: 30123,
            cors_allowed_origins: vec!["https://example.test".to_string()],
            ..RouterConfig::default()
        };

        let view = redacted(&config).expect("the view serializes");

        assert_eq!(view["host"], json!("0.0.0.0"));
        assert_eq!(view["port"], json!(30123));
        assert_eq!(
            view["cors_allowed_origins"],
            json!(["https://example.test"])
        );
        assert_eq!(
            view["policy"],
            serde_json::to_value(&config.policy).expect("the policy serializes")
        );
        assert_eq!(view["api_key"], Value::Null, "no key, no fingerprint");
        assert_eq!(view["tenant_api_keys"], json!([]));
    }

    #[test]
    fn the_sources_record_uses_the_start_up_words() {
        let sources = SecretSources {
            api_key: Some(SecretSource::File),
            tenant_api_keys: Some((SecretSource::Environment, 2)),
            control_plane_api_keys: Some((SecretSource::File, 1)),
            postgres_db_url: None,
            redis_url: Some(SecretSource::Flag),
        };

        assert_eq!(
            secret_sources(&sources),
            json!({
                "api_key": "file",
                "tenant_api_keys": { "source": "env", "entries": 2 },
                "control_plane_api_keys": { "source": "file", "entries": 1 },
                "postgres_db_url": null,
                "redis_url": "flag",
            })
        );
    }

    #[test]
    fn the_effective_view_has_its_three_parts() {
        let view = effective(&RouterConfig::default(), &SecretSources::default())
            .expect("the view serializes");

        assert_eq!(view["build"]["version"], json!(version::VERSION));
        assert!(view["secret_sources"].is_object(), "{view}");
        assert_eq!(view["config"]["port"], json!(RouterConfig::default().port));
    }
}
