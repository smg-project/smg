use std::{
    net::{IpAddr, SocketAddr},
    path::Path,
};

use axum::http::{HeaderName, HeaderValue, Uri};
use sha2::{Digest, Sha256};

use super::*;
use crate::{middleware::scheduler, policies::cost as selection_cost};

/// Each listener needs its own port. `listeners` are `(field, flag, port)`
/// in configuration order; the first two on one port are refused under the
/// later listener's field, naming both flags.
fn validate_distinct_listener_ports(listeners: &[(&str, &str, u16)]) -> ConfigResult<()> {
    for (i, (field, flag, port)) in listeners.iter().enumerate() {
        if let Some((_, other_flag, _)) = listeners[..i].iter().find(|(_, _, p)| p == port) {
            return Err(ConfigError::InvalidValue {
                field: (*field).to_string(),
                value: port.to_string(),
                reason: format!(
                    "{flag} and {other_flag} are both {port}; each listener needs its own port"
                ),
            });
        }
    }
    Ok(())
}

/// The mesh listener's port against the HTTP, metrics and probe ports. The
/// mesh configuration lives beside the router configuration, out of
/// [`ConfigValidator`]'s sight, so `server::startup` runs this check: it is
/// the one place every way of building the server configuration (the CLI,
/// the Python binding, a library caller) passes through.
pub fn validate_mesh_listener_port(
    mesh_port: u16,
    port: u16,
    metrics_port: Option<u16>,
    health_check_port: Option<u16>,
) -> ConfigResult<()> {
    let listeners = [
        Some(("port", "--port", port)),
        metrics_port.map(|port| ("metrics.port", "--prometheus-port", port)),
        health_check_port.map(|port| ("health_check_port", "--health-check-port", port)),
        Some(("mesh_port", "--mesh-port", mesh_port)),
    ];
    let listeners: Vec<_> = listeners.into_iter().flatten().collect();
    validate_distinct_listener_ports(&listeners)
}

/// Whether a tokenizer source names a path on this host rather than a hub
/// id: absolute or explicitly relative, more than one separator (a hub id is
/// `owner/name`), or simply present on disk.
fn names_a_local_path(source: &str) -> bool {
    source.starts_with('/')
        || source.starts_with("./")
        || source.starts_with("../")
        || source.starts_with('~')
        || source.matches('/').count() > 1
        || Path::new(source).exists()
}

/// Whether a gRPC worker can still bring the tokenizer when the local source
/// is missing: the connection mode is gRPC; IGW mode takes registrations at
/// run time (`POST /workers` with `grpc://` URLs); a discovery provider adds
/// workers later (bare `host:port` URLs whose mode is probed when they
/// register); or a start-up URL names a gRPC worker itself. The connection
/// mode alone misses the middle two: it follows the start-up URLs only.
fn a_grpc_worker_may_register(config: &RouterConfig) -> bool {
    if config.connection_mode == crate::worker::ConnectionMode::Grpc
        || config.enable_igw
        || config.discovery.is_some()
    {
        return true;
    }
    let grpc = |url: &str| url.starts_with("grpc://") || url.starts_with("grpcs://");
    match &config.mode {
        RoutingMode::Regular { worker_urls } => worker_urls.iter().any(|url| grpc(url)),
        RoutingMode::PrefillDecode {
            prefill_urls,
            decode_urls,
            ..
        }
        | RoutingMode::EncodePrefillDecode {
            prefill_urls,
            decode_urls,
            ..
        } => {
            prefill_urls.iter().any(|(url, _)| grpc(url)) || decode_urls.iter().any(|url| grpc(url))
        }
        // External providers: no worker of ours registers.
        RoutingMode::OpenAI { .. } | RoutingMode::Anthropic { .. } | RoutingMode::Gemini { .. } => {
            false
        }
    }
}

/// Why a local tokenizer source cannot serve.
#[derive(Debug, PartialEq, Eq)]
enum SourceProblem {
    /// Nothing at the path on this host.
    Missing,
    /// The path is there and cannot serve, for the reason given.
    Unusable(String),
}

impl SourceProblem {
    fn reason(self) -> String {
        match self {
            Self::Missing => "does not exist".to_string(),
            Self::Unusable(reason) => reason,
        }
    }
}

/// Why `path` cannot serve as a local tokenizer source, if it cannot: it
/// must exist and be readable; a directory must hold one of the layouts the
/// tokenizer loader recognizes, a file must not be empty.
fn local_tokenizer_source_error(path: &Path) -> Result<(), SourceProblem> {
    let unusable = |reason: String| SourceProblem::Unusable(reason);
    let metadata = std::fs::metadata(path).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => SourceProblem::Missing,
        _ => unusable(format!("cannot be read: {error}")),
    })?;
    if !metadata.is_dir() {
        std::fs::File::open(path).map_err(|error| unusable(format!("cannot be read: {error}")))?;
        if metadata.len() == 0 {
            return Err(unusable("is empty".to_string()));
        }
        return Ok(());
    }
    let mut entries =
        std::fs::read_dir(path).map_err(|error| unusable(format!("cannot be read: {error}")))?;
    // The loader's own rule: one of the single files, or the vocab.json +
    // merges.txt pair (vocab.json alone is not a layout it loads).
    let holds_a_tokenizer = entries.any(|entry| {
        entry.is_ok_and(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name == "tokenizer.json" || name == "tiktoken.model" || name.ends_with(".tiktoken")
        })
    }) || (path.join("vocab.json").is_file()
        && path.join("merges.txt").is_file());
    if holds_a_tokenizer {
        Ok(())
    } else {
        Err(unusable(
            "contains no tokenizer file (tokenizer.json, tiktoken.model, *.tiktoken, or vocab.json with merges.txt)"
                .to_string(),
        ))
    }
}

/// Why `path` is not a readable, non-empty text file, if it is not.
fn text_file_error(path: &Path) -> Result<(), String> {
    let metadata = std::fs::metadata(path).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => "does not exist".to_string(),
        _ => format!("cannot be read: {error}"),
    })?;
    if metadata.is_dir() {
        return Err("is a directory, not a file".to_string());
    }
    let content =
        std::fs::read_to_string(path).map_err(|error| format!("cannot be read: {error}"))?;
    if content.trim().is_empty() {
        return Err("is empty".to_string());
    }
    Ok(())
}

/// Validate a user-supplied mesh server name. The name keys rate-limit
/// shards as `rl:{counter}:{name}`, so an empty name or one containing the
/// separator would corrupt shard keys; rejecting at config time avoids a
/// panic at mesh adapter construction during startup.
pub fn validate_mesh_server_name(name: &str) -> ConfigResult<()> {
    if name.is_empty() || name.contains(':') {
        return Err(ConfigError::InvalidValue {
            field: "mesh_server_name".to_string(),
            value: name.to_string(),
            reason: "must be non-empty and must not contain ':'".to_string(),
        });
    }
    Ok(())
}

/// Validate a single worker URL: non-empty, an allowed scheme, and a
/// parseable host. Shared by [`ConfigValidator::validate_urls`] (startup
/// config) and the worker-management API so both reject schemeless or
/// unparsable URLs identically. Rejecting at the API boundary prevents
/// the orphaned `url_to_id` reservation from #1533: the AddWorker
/// workflow rewrites schemeless input via `normalize_url`, so a
/// reservation keyed on the raw URL would never match the registered
/// worker.
pub fn validate_worker_url(url: &str) -> ConfigResult<()> {
    if url.is_empty() {
        return Err(ConfigError::InvalidValue {
            field: "worker_url".to_string(),
            value: url.to_string(),
            reason: "URL cannot be empty".to_string(),
        });
    }

    // Exact (lowercase) scheme allow-list. Case-insensitive matching is
    // tempting but wrong here: the AddWorker workflow's normalize_url
    // matches schemes case-sensitively, so a mixed-case scheme would be
    // rewritten downstream and diverge from the reservation key — the
    // same orphan failure as a schemeless URL.
    const ALLOWED_SCHEMES: &[&str] = &["http", "https", "grpc", "grpcs", "ipc"];
    let scheme = url.split_once("://").map_or("", |(s, _)| s);
    if !ALLOWED_SCHEMES.contains(&scheme) {
        return Err(ConfigError::InvalidValue {
            field: "worker_url".to_string(),
            value: url.to_string(),
            reason: "URL must start with a lowercase http://, https://, grpc://, grpcs://, or ipc:// scheme"
                .to_string(),
        });
    }

    // ipc:// worker URLs are same-host ZMQ unix-socket paths (no host); validate
    // the path is present rather than requiring a host below.
    if scheme == "ipc" {
        let path = url.strip_prefix("ipc://").unwrap_or("");
        if path.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "worker_url".to_string(),
                value: url.to_string(),
                reason: "ipc:// worker URL must include a socket path".to_string(),
            });
        }
        return Ok(());
    }

    match ::url::Url::parse(url) {
        Ok(parsed) => {
            if parsed.host_str().is_none() {
                return Err(ConfigError::InvalidValue {
                    field: "worker_url".to_string(),
                    value: url.to_string(),
                    reason: "URL must have a valid host".to_string(),
                });
            }
        }
        Err(e) => {
            return Err(ConfigError::InvalidValue {
                field: "worker_url".to_string(),
                value: url.to_string(),
                reason: format!("Invalid URL format: {e}"),
            });
        }
    }
    Ok(())
}

/// Parse a listener host as operators write it: an IP literal, with or
/// without the brackets an IPv6 address carries inside a `host:port` string
/// (`0.0.0.0`, `::`, `[::]`, `::1`, `[fd00::1]`). Every bind flag goes
/// through this one rule, so `--host`, `--prometheus-host`, the probe
/// listener and the mesh listener accept the same spellings. A hostname is
/// not a bind address and is rejected, like anything else that is not an IP
/// address.
pub fn parse_bind_host(host: &str) -> Result<IpAddr, String> {
    let bare = host
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(host);
    bare.parse::<IpAddr>().map_err(|_| {
        format!(
            "invalid bind host '{host}': expected an IP address such as \
             0.0.0.0, ::, [::] or [fd00::1]"
        )
    })
}

/// The socket address a listener binds for `host` (any spelling
/// [`parse_bind_host`] accepts) and `port`.
pub fn bind_socket_addr(host: &str, port: u16) -> Result<SocketAddr, String> {
    parse_bind_host(host).map(|ip| SocketAddr::new(ip, port))
}

/// Configuration validator
pub(crate) struct ConfigValidator;
impl ConfigValidator {
    pub(crate) fn validate(config: &RouterConfig) -> ConfigResult<()> {
        Self::validate_mode(&config.mode)?;
        bounds::validate(config)?;
        Self::validate_policy(&config.policy)?;
        Self::validate_cache_boundaries(&config.cache_boundaries)?;
        Self::validate_server_settings(config)?;
        Self::validate_cors_allowed_origins(config)?;
        Self::validate_value_shapes(config)?;
        Self::validate_log_dir(config)?;
        Self::validate_storage_context_headers(config)?;
        Self::validate_routing_key_headers(config)?;
        Self::validate_tenant_resolution(config)?;
        Self::validate_tenant_api_keys(config)?;
        Self::validate_memory_history(config)?;
        Self::validate_model_aliases(config)?;
        Self::validate_rl(config)?;
        Self::validate_kv_index(config)?;
        Self::validate_prefill_admission(config)?;
        Self::validate_priority_scheduler(config)?;
        if let Some(discovery) = &config.discovery {
            Self::validate_discovery(discovery, &config.mode)?;
        }

        if let Some(metrics) = &config.metrics {
            Self::validate_metrics(metrics)?;
        }

        if let Some(trace_config) = &config.trace_config {
            Self::validate_trace(trace_config)?;
        }

        Self::validate_compatibility(config)?;

        let retry_cfg = config.effective_retry_config();
        let cb_cfg = config.effective_circuit_breaker_config();
        Self::validate_retry(&retry_cfg)?;
        Self::validate_circuit_breaker(&cb_cfg)?;
        Self::validate_health_check(&config.health_check)?;

        if config.history_backend == HistoryBackend::Oracle {
            if config.oracle.is_none() {
                return Err(ConfigError::MissingRequired {
                    field: "oracle".to_string(),
                });
            }
            if let Some(oracle) = &config.oracle {
                Self::validate_oracle(oracle)?;
            }
        }
        // The Redis and Postgres configurations carry their own numeric rules
        // (the pool size, the call bound, a retention of 0 days that would
        // expire every record as it is written); the command line ran them
        // while building its config, the bindings and a directly built
        // RouterConfig did not, so they run here for every start-up. The
        // connection string's shape is the driver's: a socket path or a
        // key=value string the driver understands is not refused here (the
        // command line's own URL check stays with the command line).
        if config.history_backend == HistoryBackend::Redis {
            if let Some(redis) = &config.redis {
                redis
                    .validate_bounds()
                    .map_err(|reason| ConfigError::ValidationFailed { reason })?;
            }
        }
        if config.history_backend == HistoryBackend::Postgres {
            if let Some(postgres) = &config.postgres {
                postgres
                    .validate_bounds()
                    .map_err(|reason| ConfigError::ValidationFailed { reason })?;
            }
        }

        Self::validate_tokenizer_cache(&config.tokenizer_cache)?;
        Self::validate_tokenizer_sources(config)?;

        Ok(())
    }

    fn validate_prefill_admission(config: &RouterConfig) -> ConfigResult<()> {
        if config.prefill_max_inflight_requests_per_worker <= 0 {
            if config.prefill_queue_size.is_some() || config.prefill_queue_timeout_secs.is_some() {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "Prefill queue options require --prefill-max-inflight-requests-per-worker to be positive".to_string(),
                });
            }
            return Ok(());
        }

        if !matches!(
            config.mode,
            RoutingMode::PrefillDecode { .. } | RoutingMode::EncodePrefillDecode { .. }
        ) {
            return Err(ConfigError::IncompatibleConfig {
                reason: "Prefill admission is only supported in PD or EPD mode".to_string(),
            });
        }

        if config.priority_scheduler_enabled {
            return Err(ConfigError::IncompatibleConfig {
                reason: "Prefill admission cannot be combined with --priority-scheduler-enabled"
                    .to_string(),
            });
        }

        if config.effective_prefill_queue_size() > 0
            && config.effective_prefill_queue_timeout_secs() == 0
        {
            return Err(ConfigError::InvalidValue {
                field: "prefill_queue_timeout_secs".to_string(),
                value: "0".to_string(),
                reason: "Must be > 0 when the Prefill queue is enabled".to_string(),
            });
        }

        Ok(())
    }

    /// With the priority scheduler enabled, the optional YAML file must be
    /// readable, parsable (no key the scheduler does not know) and valid, and
    /// the default max class must name one of the four classes: an operator
    /// who asked for tenant priorities gets a start-up error naming the field
    /// instead of a gateway that serves with the legacy admission or with a
    /// policy they did not write (the contract of `tenant_rate_limit_config`).
    /// Without the flag neither is read.
    fn validate_priority_scheduler(config: &RouterConfig) -> ConfigResult<()> {
        if !config.priority_scheduler_enabled {
            return Ok(());
        }
        scheduler::settings_from_config(config)
            .map(drop)
            .map_err(|e| ConfigError::InvalidValue {
                field: e.field.to_string(),
                value: e.value,
                reason: e.reason,
            })
    }

    fn validate_model_aliases(config: &RouterConfig) -> ConfigResult<()> {
        for (alias, canonical) in &config.model_aliases {
            if alias.is_empty() || canonical.is_empty() {
                return Err(ConfigError::InvalidValue {
                    field: "model_aliases".to_string(),
                    value: format!("{alias}={canonical}"),
                    reason: "Alias and canonical model ID must be non-empty".to_string(),
                });
            }
            if alias == canonical {
                return Err(ConfigError::InvalidValue {
                    field: "model_aliases".to_string(),
                    value: alias.clone(),
                    reason: "Alias must differ from the canonical model ID".to_string(),
                });
            }
        }

        Ok(())
    }

    /// The fields whose value has a required shape, refused by name when the
    /// shape is wrong, on every path that builds a `RouterConfig` (the
    /// command line, a configuration file, the language binding): checked on
    /// the command line alone, a wrong shape from a binding still matched no
    /// request's id header or switched authentication on with an empty
    /// credential, in silence. The CORS origins and the health-check endpoint
    /// have rules of their own (`validate_cors_allowed_origins`,
    /// `validate_health_check`); the command line keeps the shapes of what
    /// never reaches `RouterConfig`.
    fn validate_value_shapes(config: &RouterConfig) -> ConfigResult<()> {
        let invalid = |field: &str, value: &str, reason: &str| ConfigError::InvalidValue {
            field: field.to_string(),
            value: value.to_string(),
            reason: reason.to_string(),
        };
        for name in config.request_id_headers.iter().flatten() {
            if HeaderName::try_from(name.as_str()).is_err() {
                return Err(invalid(
                    "request_id_headers",
                    name,
                    "not a valid HTTP header name: it could never match a request",
                ));
            }
        }
        if let Some(key) = &config.api_key {
            if key.trim().is_empty() {
                return Err(invalid(
                    "api_key",
                    "",
                    "an empty credential would switch authentication on with a key nothing can present",
                ));
            }
        }
        Ok(())
    }

    fn validate_storage_context_headers(config: &RouterConfig) -> ConfigResult<()> {
        let mut seen_context_keys = std::collections::HashSet::new();

        for (header_name, context_key) in &config.storage_context_headers {
            let header_name = header_name.trim();
            let context_key = context_key.trim();

            if header_name.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: "storage_context_headers must not contain empty header names"
                        .to_string(),
                });
            }

            if HeaderName::try_from(header_name).is_err() {
                return Err(ConfigError::InvalidValue {
                    field: "storage_context_headers".to_string(),
                    value: header_name.to_string(),
                    reason: "not a valid HTTP header name: it could never match a request"
                        .to_string(),
                });
            }

            if context_key.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: "storage_context_headers must not contain empty context keys"
                        .to_string(),
                });
            }

            if !seen_context_keys.insert(context_key.to_string()) {
                return Err(ConfigError::ValidationFailed {
                    reason: format!(
                        "storage_context_headers must not map multiple headers to the same context key: '{context_key}'"
                    ),
                });
            }
        }

        Ok(())
    }

    fn validate_routing_key_headers(config: &RouterConfig) -> ConfigResult<()> {
        for name in &config.routing_key_override.headers {
            HeaderName::try_from(name.as_str()).map_err(|e| ConfigError::ValidationFailed {
                reason: format!(
                    "routing_key_override.headers contains an invalid HTTP header name '{name}': {e}"
                ),
            })?;
        }
        Ok(())
    }

    /// The memory history backend's bounds: at least one record of each kind.
    fn validate_memory_history(config: &RouterConfig) -> ConfigResult<()> {
        if let Some(memory) = &config.memory {
            memory
                .validate()
                .map_err(|reason| ConfigError::ValidationFailed { reason })?;
        }
        Ok(())
    }

    fn validate_tenant_resolution(config: &RouterConfig) -> ConfigResult<()> {
        let header_name = config.tenant_resolution.tenant_header_name.trim();
        if header_name.is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: "tenant_resolution.tenant_header_name must not be empty".to_string(),
            });
        }

        HeaderName::try_from(header_name).map_err(|e| ConfigError::ValidationFailed {
            reason: format!(
                "tenant_resolution.tenant_header_name must be a valid HTTP header name: {e}"
            ),
        })?;

        Ok(())
    }

    /// Validates `tenant_api_keys`: non-empty `tenant_id`/`key`, and no two
    /// credentials (including the shared `api_key`) sharing a secret value —
    /// duplicates would silently attribute one tenant's traffic to another.
    /// Runs regardless of construction path, since `TenantApiKeyEntry` is a
    /// public deserializable struct. Compares hashes only; errors never
    /// include a raw key value.
    fn validate_tenant_api_keys(config: &RouterConfig) -> ConfigResult<()> {
        fn hash(key: &str) -> [u8; 32] {
            Sha256::digest(key.as_bytes()).into()
        }

        let mut seen: std::collections::HashMap<[u8; 32], String> =
            std::collections::HashMap::new();

        if let Some(api_key) = &config.api_key {
            seen.insert(hash(api_key), "the shared api_key".to_string());
        }

        for entry in &config.tenant_api_keys {
            let trimmed_tenant_id = entry.tenant_id.trim();
            if trimmed_tenant_id.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: "tenant_api_keys entries must have a non-empty tenant_id".to_string(),
                });
            }
            // The CLI parser already trims tenant_id, but config-file/binding
            // entries bypass it — reject padding here instead of silently
            // normalizing, since `auth:<tenant_id>` embeds it verbatim and a
            // padded id would resolve to a different, likely-unintended
            // tenant identity than the canonical one.
            if trimmed_tenant_id != entry.tenant_id {
                return Err(ConfigError::ValidationFailed {
                    reason: format!(
                        "tenant_api_keys tenant_id '{}' must not have surrounding whitespace",
                        entry.tenant_id
                    ),
                });
            }
            // The canonical key is stored with every response and
            // conversation, in a 255-byte column on the SQL backends.
            if entry.tenant_id.len() > crate::tenant::MAX_TENANT_ID_LEN {
                return Err(ConfigError::ValidationFailed {
                    reason: format!(
                        "tenant_api_keys tenant_id '{}' is longer than {} bytes",
                        entry.tenant_id,
                        crate::tenant::MAX_TENANT_ID_LEN
                    ),
                });
            }
            let trimmed_key = entry.key.trim();
            if trimmed_key.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: format!(
                        "tenant_api_keys entry for tenant_id '{}' must have a non-empty key",
                        entry.tenant_id
                    ),
                });
            }
            // Same asymmetry as tenant_id: the CLI trims the key, but
            // config-file/binding entries don't go through it. A padded key
            // would hash differently than the operator likely intended,
            // silently defeating the duplicate-value check above for that
            // entry.
            if trimmed_key != entry.key {
                return Err(ConfigError::ValidationFailed {
                    reason: format!(
                        "tenant_api_keys entry for tenant_id '{}' key must not have surrounding whitespace",
                        entry.tenant_id
                    ),
                });
            }

            let label = format!("tenant_id '{}'", entry.tenant_id);
            if let Some(existing) = seen.insert(hash(&entry.key), label.clone()) {
                return Err(ConfigError::ValidationFailed {
                    reason: format!(
                        "duplicate API key value: {label} uses the same credential as {existing}. Each credential must be unique, or requests authenticate as whichever entry is checked last."
                    ),
                });
            }
        }

        Ok(())
    }

    fn validate_oracle(oracle: &OracleConfig) -> ConfigResult<()> {
        if oracle.external_auth {
            if !oracle.username.is_empty() || !oracle.password.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: "oracle.username and oracle.password must be empty when oracle.external_auth is true"
                        .to_string(),
                });
            }
        } else {
            if oracle.username.is_empty() {
                return Err(ConfigError::MissingRequired {
                    field: "oracle.username".to_string(),
                });
            }

            if oracle.password.is_empty() {
                return Err(ConfigError::MissingRequired {
                    field: "oracle.password".to_string(),
                });
            }
        }

        if oracle.connect_descriptor.is_empty() {
            return Err(ConfigError::MissingRequired {
                field: "oracle_dsn or oracle_tns_alias".to_string(),
            });
        }

        if oracle.pool_min < 1 {
            return Err(ConfigError::InvalidValue {
                field: "oracle.pool_min".to_string(),
                value: oracle.pool_min.to_string(),
                reason: "Must be at least 1".to_string(),
            });
        }

        if oracle.pool_max < oracle.pool_min {
            return Err(ConfigError::InvalidValue {
                field: "oracle.pool_max".to_string(),
                value: oracle.pool_max.to_string(),
                reason: "Must be >= oracle.pool_min".to_string(),
            });
        }

        if oracle.pool_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "oracle.pool_timeout_secs".to_string(),
                value: oracle.pool_timeout_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        Ok(())
    }

    fn validate_mode(mode: &RoutingMode) -> ConfigResult<()> {
        match mode {
            RoutingMode::Regular { worker_urls } => {
                if !worker_urls.is_empty() {
                    Self::validate_urls(worker_urls)?;
                }
                // Allow empty URLs without service discovery to match legacy behavior
            }
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy,
                decode_policy,
            } => {
                // Allow empty URLs even without service discovery to support dynamic worker addition
                // URLs will be validated if provided
                if !prefill_urls.is_empty() {
                    let prefill_url_strings: Vec<String> =
                        prefill_urls.iter().map(|(url, _)| url.clone()).collect();
                    Self::validate_urls(&prefill_url_strings)?;
                }
                if !decode_urls.is_empty() {
                    Self::validate_urls(decode_urls)?;
                }

                for (_url, port) in prefill_urls {
                    if let Some(port) = port {
                        if *port == 0 {
                            return Err(ConfigError::InvalidValue {
                                field: "bootstrap_port".to_string(),
                                value: port.to_string(),
                                reason: "Port must be between 1 and 65535".to_string(),
                            });
                        }
                    }
                }

                if let Some(p_policy) = prefill_policy {
                    Self::validate_policy(p_policy)?;
                }
                if let Some(d_policy) = decode_policy {
                    Self::validate_policy(d_policy)?;
                }
            }
            RoutingMode::EncodePrefillDecode {
                encode_urls,
                prefill_urls,
                decode_urls,
                encode_policy,
                prefill_policy,
                decode_policy,
            } => {
                for urls in [encode_urls, prefill_urls] {
                    if !urls.is_empty() {
                        let url_strings: Vec<String> =
                            urls.iter().map(|(url, _)| url.clone()).collect();
                        Self::validate_urls(&url_strings)?;
                    }
                    for (_url, port) in urls {
                        if let Some(port) = port {
                            if *port == 0 {
                                return Err(ConfigError::InvalidValue {
                                    field: "bootstrap_port".to_string(),
                                    value: port.to_string(),
                                    reason: "Port must be between 1 and 65535".to_string(),
                                });
                            }
                        }
                    }
                }
                if !decode_urls.is_empty() {
                    Self::validate_urls(decode_urls)?;
                }
                if let Some(policy) = encode_policy {
                    Self::validate_policy(policy)?;
                    Self::validate_encode_policy(policy)?;
                }
                if let Some(policy) = prefill_policy {
                    Self::validate_policy(policy)?;
                }
                if let Some(policy) = decode_policy {
                    Self::validate_policy(policy)?;
                }
            }
            RoutingMode::OpenAI { worker_urls } => {
                // Allow empty URLs to support dynamic worker addition
                // URLs will be validated if provided
                if !worker_urls.is_empty() {
                    Self::validate_urls(worker_urls)?;
                }
            }
            RoutingMode::Anthropic { worker_urls } => {
                // Allow empty URLs to support dynamic worker addition
                // URLs will be validated if provided
                if !worker_urls.is_empty() {
                    Self::validate_urls(worker_urls)?;
                }
            }
            RoutingMode::Gemini { worker_urls } => {
                // Allow empty URLs to support dynamic worker addition
                if !worker_urls.is_empty() {
                    Self::validate_urls(worker_urls)?;
                }
            }
        }
        Ok(())
    }

    fn validate_policy(policy: &PolicyConfig) -> ConfigResult<()> {
        match policy {
            PolicyConfig::Random
            | PolicyConfig::RoundRobin
            | PolicyConfig::Passthrough
            | PolicyConfig::Manual { .. }
            | PolicyConfig::ConsistentHashing => {}
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold: _,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
                block_size,
                balance_token_usage_threshold,
                overload_token_usage_threshold,
                overlap_decay,
                selection_temperature,
                cache_index,
                cache_ttl_secs,
                cache_boundaries,
                selection_policy,
                selection_accounting_ttl_ms: _,
            } => {
                Self::validate_cache_boundaries(cache_boundaries)?;

                // Build the selection policy once here so a bad name or
                // parameter fails configuration instead of routing.
                let selection_policy_name = selection_policy
                    .as_deref()
                    .unwrap_or(selection_cost::DEFAULT_POLICY);
                if let Err(err) =
                    selection_cost::build(selection_policy_name, *selection_temperature)
                {
                    return Err(ConfigError::InvalidValue {
                        field: "selection_policy".to_string(),
                        value: selection_policy_name.to_string(),
                        reason: err.to_string(),
                    });
                }

                if *cache_ttl_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "cache_ttl_secs".to_string(),
                        value: cache_ttl_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                if *cache_index == CacheIndexKind::Hash && cache_boundaries.is_empty() {
                    return Err(ConfigError::InvalidValue {
                        field: "cache_index".to_string(),
                        value: "hash".to_string(),
                        reason: "cache_index=hash requires non-empty cache_boundaries".to_string(),
                    });
                }

                if !overlap_decay.is_finite() || *overlap_decay < 0.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "overlap_decay".to_string(),
                        value: overlap_decay.to_string(),
                        reason: "Must be finite and >= 0.0 (0.0 disables)".to_string(),
                    });
                }

                if !selection_temperature.is_finite() || *selection_temperature < 0.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "selection_temperature".to_string(),
                        value: selection_temperature.to_string(),
                        reason: "Must be finite and >= 0.0 (0.0 is argmax)".to_string(),
                    });
                }

                if *block_size == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "block_size".to_string(),
                        value: block_size.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                // NaN passes every comparison and an infinite threshold is a typo
                // for "disabled": both are refused with the sign errors.
                if !balance_token_usage_threshold.is_finite()
                    || *balance_token_usage_threshold <= 0.0
                {
                    return Err(ConfigError::InvalidValue {
                        field: "balance_token_usage_threshold".to_string(),
                        value: balance_token_usage_threshold.to_string(),
                        reason: "Must be finite and > 0.0 (use >= 1.0 to disable)".to_string(),
                    });
                }

                if !overload_token_usage_threshold.is_finite()
                    || *overload_token_usage_threshold <= 0.0
                {
                    return Err(ConfigError::InvalidValue {
                        field: "overload_token_usage_threshold".to_string(),
                        value: overload_token_usage_threshold.to_string(),
                        reason: "Must be finite and > 0.0 (use >= 1.0 to disable)".to_string(),
                    });
                }

                if !(0.0..=1.0).contains(cache_threshold) {
                    return Err(ConfigError::InvalidValue {
                        field: "cache_threshold".to_string(),
                        value: cache_threshold.to_string(),
                        reason: "Must be between 0.0 and 1.0".to_string(),
                    });
                }

                if !balance_rel_threshold.is_finite() || *balance_rel_threshold < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "balance_rel_threshold".to_string(),
                        value: balance_rel_threshold.to_string(),
                        reason: "Must be finite and >= 1.0".to_string(),
                    });
                }

                if *eviction_interval_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "eviction_interval_secs".to_string(),
                        value: eviction_interval_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                if *max_tree_size == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "max_tree_size".to_string(),
                        value: max_tree_size.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
                ..
            } => {
                if *load_check_interval_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "load_check_interval_secs".to_string(),
                        value: load_check_interval_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::LeastLoad {
                load_check_interval_secs,
                kv_pressure_weight,
                mean_prefill_tokens,
                default_throughput,
                max_waiting_requests: _,
            } => {
                if *load_check_interval_secs == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "load_check_interval_secs".to_string(),
                        value: load_check_interval_secs.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                if !kv_pressure_weight.is_finite() || *kv_pressure_weight < 0.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "kv_pressure_weight".to_string(),
                        value: kv_pressure_weight.to_string(),
                        reason: "Must be finite and >= 0.0".to_string(),
                    });
                }

                if *mean_prefill_tokens == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "mean_prefill_tokens".to_string(),
                        value: mean_prefill_tokens.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                if !default_throughput.is_finite() || *default_throughput <= 0.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "default_throughput".to_string(),
                        value: default_throughput.to_string(),
                        reason: "Must be finite and > 0.0".to_string(),
                    });
                }
            }
            PolicyConfig::Bucket {
                balance_abs_threshold: _,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                if !balance_rel_threshold.is_finite() || *balance_rel_threshold < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "balance_rel_threshold".to_string(),
                        value: balance_rel_threshold.to_string(),
                        reason: "Must be finite and >= 1.0".to_string(),
                    });
                }

                if *bucket_adjust_interval_secs < 1 {
                    return Err(ConfigError::InvalidValue {
                        field: "bucket_adjust_interval_secs".to_string(),
                        value: bucket_adjust_interval_secs.to_string(),
                        reason: "Must be >= 1s".to_string(),
                    });
                }
                if *bucket_adjust_interval_secs >= 4294967296 {
                    return Err(ConfigError::InvalidValue {
                        field: "bucket_adjust_interval_secs".to_string(),
                        value: bucket_adjust_interval_secs.to_string(),
                        reason: "Must be < 4294967296s".to_string(),
                    });
                }
            }
            PolicyConfig::PrefixHash {
                prefix_token_count,
                load_factor,
                balance_abs_threshold: _,
                cache_boundaries,
            } => {
                if *prefix_token_count == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "prefix_token_count".to_string(),
                        value: prefix_token_count.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }

                if !load_factor.is_finite() || *load_factor < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "load_factor".to_string(),
                        value: load_factor.to_string(),
                        reason: "Must be finite and >= 1.0".to_string(),
                    });
                }

                Self::validate_cache_boundaries(cache_boundaries)?;
            }
        }
        Ok(())
    }

    fn validate_cache_boundaries(boundaries: &[usize]) -> ConfigResult<()> {
        if boundaries.first() == Some(&0) {
            return Err(ConfigError::InvalidValue {
                field: "cache_boundaries".to_string(),
                value: "0".to_string(),
                reason: "Boundaries must be > 0".to_string(),
            });
        }
        if boundaries.windows(2).any(|w| w[0] >= w[1]) {
            return Err(ConfigError::InvalidValue {
                field: "cache_boundaries".to_string(),
                value: format!("{boundaries:?}"),
                reason: "Boundaries must be strictly ascending".to_string(),
            });
        }
        Ok(())
    }

    fn validate_encode_policy(policy: &PolicyConfig) -> ConfigResult<()> {
        match policy {
            PolicyConfig::Random | PolicyConfig::RoundRobin | PolicyConfig::ConsistentHashing => {
                Ok(())
            }
            _ => Err(ConfigError::IncompatibleConfig {
                reason: "Encode policy supports random, round_robin, or consistent_hashing"
                    .to_string(),
            }),
        }
    }

    fn validate_server_settings(config: &RouterConfig) -> ConfigResult<()> {
        if let Err(reason) = parse_bind_host(&config.host) {
            return Err(ConfigError::InvalidValue {
                field: "host".to_string(),
                value: config.host.clone(),
                reason,
            });
        }

        if config.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "port".to_string(),
                value: config.port.to_string(),
                reason: "Port must be > 0".to_string(),
            });
        }

        // Reject a configured dedicated probe port of 0: 0 means "OS-assigned
        // ephemeral port", which breaks the fail-fast contract that probes
        // live on a stable operator-configured port. (`start_probe_listener`
        // itself still accepts 0 so the ephemeral-port unit tests can bind;
        // only the config-sourced value is rejected here.)
        if config.health_check_port == Some(0) {
            return Err(ConfigError::InvalidValue {
                field: "health_check_port".to_string(),
                value: "0".to_string(),
                reason: "Port must be > 0 (0 would request an unstable OS-ephemeral port)"
                    .to_string(),
            });
        }

        // Each listener needs its own port. Two of them on one port used to
        // bind the first, answer, and die at the second bind with a raw
        // AddrInUse that named no flag: a partial start an operator's
        // monitoring saw come up and vanish.
        let listeners = [
            Some(("port", "--port", config.port)),
            config
                .metrics
                .as_ref()
                .map(|metrics| ("metrics.port", "--prometheus-port", metrics.port)),
            config
                .health_check_port
                .map(|port| ("health_check_port", "--health-check-port", port)),
        ];
        let listeners: Vec<_> = listeners.into_iter().flatten().collect();
        validate_distinct_listener_ports(&listeners)?;

        if config.max_payload_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "max_payload_size".to_string(),
                value: config.max_payload_size.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        if config.mm_per_request_image_limit == Some(0) {
            return Err(ConfigError::InvalidValue {
                field: "mm_per_request_image_limit".to_string(),
                value: "0".to_string(),
                reason: "Must be at least 1".to_string(),
            });
        }

        // The async runtime builder panics on zero worker threads; refuse it
        // here like every other count that must be positive.
        if config.runtime_worker_threads == Some(0) {
            return Err(ConfigError::InvalidValue {
                field: "runtime_worker_threads".to_string(),
                value: "0".to_string(),
                reason: "Must be >= 1".to_string(),
            });
        }

        // A zero-capacity job channel panics at construction and a
        // zero-permit dispatcher never dequeues; reject both here so a
        // config-file value fails as early as the CLI parsers do.
        // Mirror the CLI parser bounds so config-file and bindings paths get
        // the same guarantees (zero panics at channel construction; unbounded
        // values are allocation hazards).
        if !(1..=1_000_000).contains(&config.job_queue_capacity) {
            return Err(ConfigError::InvalidValue {
                field: "job_queue_capacity".to_string(),
                value: config.job_queue_capacity.to_string(),
                reason: "Must be in 1..=1000000".to_string(),
            });
        }
        if !(1..=100_000).contains(&config.job_queue_concurrency) {
            return Err(ConfigError::InvalidValue {
                field: "job_queue_concurrency".to_string(),
                value: config.job_queue_concurrency.to_string(),
                reason: "Must be in 1..=100000".to_string(),
            });
        }

        // Overload thresholds are `>=` comparisons, so the excluded ends of
        // these ranges are exactly the values that would veto every worker
        // unconditionally. Mirrors the CLI parsers for the config-file and
        // bindings paths.
        if config.worker_overload_waiting_requests == Some(0) {
            return Err(ConfigError::InvalidValue {
                field: "worker_overload_waiting_requests".to_string(),
                value: "0".to_string(),
                reason: "Must be >= 1 (0 would mark every worker overloaded)".to_string(),
            });
        }
        if let Some(threshold) = config.worker_overload_token_usage {
            if !threshold.is_finite() || threshold <= 0.0 || threshold > 1.0 {
                return Err(ConfigError::InvalidValue {
                    field: "worker_overload_token_usage".to_string(),
                    value: threshold.to_string(),
                    reason: "Must be a fraction in (0.0, 1.0]".to_string(),
                });
            }
        }

        if config.request_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "request_timeout_secs".to_string(),
                value: config.request_timeout_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        if config.queue_size > 0 && config.queue_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "queue_timeout_secs".to_string(),
                value: config.queue_timeout_secs.to_string(),
                reason: "Must be > 0 when queue_size > 0".to_string(),
            });
        }

        if let Some(tokens_per_second) = config.rate_limit_tokens_per_second {
            // Allow 0 for pure concurrency limiting (semaphore behavior)
            if tokens_per_second < 0 {
                return Err(ConfigError::InvalidValue {
                    field: "rate_limit_tokens_per_second".to_string(),
                    value: tokens_per_second.to_string(),
                    reason: "Must be >= 0 when specified".to_string(),
                });
            }
        }

        if config.worker_startup_timeout_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "worker_startup_timeout_secs".to_string(),
                value: config.worker_startup_timeout_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        if config.worker_startup_check_interval_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "worker_startup_check_interval_secs".to_string(),
                value: config.worker_startup_check_interval_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        if config.load_monitor_interval_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "load_monitor_interval_secs".to_string(),
                value: config.load_monitor_interval_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        Ok(())
    }

    /// `cors_allowed_origins`: exact origins, or a lone `*` for the open
    /// policy. This is the one place the rule is checked, so the Python
    /// launcher and library callers (`RouterConfig::builder().build()`)
    /// refuse the same values by name as `smg launch` does.
    fn validate_cors_allowed_origins(config: &RouterConfig) -> ConfigResult<()> {
        let invalid = |origin: &str, reason: &str| ConfigError::InvalidValue {
            field: "cors_allowed_origins".to_string(),
            value: origin.to_string(),
            reason: reason.to_string(),
        };
        for origin in &config.cors_allowed_origins {
            // A lone `*` is the open policy; next to an origin it would be
            // either a typo or a list the layer cannot express.
            if origin == "*" {
                if config.cors_allowed_origins.len() == 1 {
                    continue;
                }
                return Err(invalid(
                    origin,
                    "'*' stands alone: either any origin or a list of exact origins",
                ));
            }
            // Exactly what a browser sends as `Origin`: the scheme and the host
            // in lower case, no userinfo, no default port, nothing after the
            // authority (not even `/`); anything else never matches the exact
            // comparison of the CORS layer.
            let default_port = |scheme: &str| match scheme {
                "http" | "ws" => Some(80),
                "https" | "wss" => Some(443),
                _ => None,
            };
            let parsed = Uri::try_from(origin.as_str()).ok().filter(|uri| {
                matches!((uri.scheme_str(), uri.authority()), (Some(scheme), Some(authority))
                    if origin.as_str() == format!("{scheme}://{authority}")
                        && authority.as_str() == authority.as_str().to_ascii_lowercase()
                        && !authority.as_str().contains('@')
                        && authority
                            .port_u16()
                            .is_none_or(|port| Some(port) != default_port(scheme)))
            });
            if parsed.is_none() || HeaderValue::from_str(origin).is_err() {
                return Err(invalid(
                    origin,
                    "not exactly an origin (lower-case scheme://host[:port], no userinfo, no default port, nothing after it); the policy would silently narrow to the origins that match",
                ));
            }
        }
        Ok(())
    }

    /// `log_dir` names the directory the file log appender writes to, so it
    /// is checked before anything binds: a path that exists must be a
    /// directory the gateway can write; one that does not exist must sit
    /// under a directory the gateway can write (its nearest existing
    /// ancestor), where logging start-up creates it, as it always did.
    /// Anything else is a configuration error naming the path and the
    /// reason, where the appender used to panic (an existing file) or the
    /// gateway started without any log output at all (a path it could not
    /// create). Nothing is created here: `validate()` is also run by
    /// callers that build or check a configuration without starting the
    /// gateway, and leaves no directory behind on their host.
    fn validate_log_dir(config: &RouterConfig) -> ConfigResult<()> {
        let Some(log_dir) = config.log_dir.as_deref() else {
            return Ok(());
        };
        let invalid = |reason: String| ConfigError::InvalidValue {
            field: "log_dir".to_string(),
            value: log_dir.to_string(),
            reason,
        };
        if log_dir.trim().is_empty() {
            return Err(invalid("must name a directory".to_string()));
        }
        let path = Path::new(log_dir);
        let (probed, missing) = match std::fs::metadata(path) {
            Ok(metadata) if metadata.is_dir() => (path, false),
            Ok(_) => return Err(invalid("exists and is not a directory".to_string())),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                ) =>
            {
                // Logging start-up creates the missing part under the nearest
                // existing ancestor, which must therefore be a directory the
                // gateway can write.
                (
                    Self::nearest_existing_ancestor(path).map_err(&invalid)?,
                    true,
                )
            }
            Err(error) => return Err(invalid(format!("cannot be read: {error}"))),
        };
        // The appender opens a file for writing in this directory (or creates
        // the directory under the ancestor); an unnamed temporary file is the
        // same permission check with nothing left behind.
        tempfile::tempfile_in(probed).map(drop).map_err(|error| {
            if missing {
                invalid(format!(
                    "cannot be created: {} is not writable: {error}",
                    probed.display()
                ))
            } else {
                invalid(format!("is not writable: {error}"))
            }
        })
    }

    /// The nearest existing ancestor of a path that does not exist: the
    /// directory the missing part would be created under. A relative path's
    /// last ancestor is "", which is the current directory but does not
    /// stat, so it reads as ".". The error is the reason of the refusal when
    /// there is no such ancestor or it is not a directory: a file in the way
    /// is named, which explains the failure better than the OS error for the
    /// leaf.
    fn nearest_existing_ancestor(path: &Path) -> Result<&Path, String> {
        let ancestor = path
            .ancestors()
            .skip(1)
            .map(|ancestor| {
                if ancestor.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    ancestor
                }
            })
            .find(|ancestor| ancestor.exists())
            .ok_or_else(|| "cannot be created: no existing ancestor".to_string())?;
        if !ancestor.is_dir() {
            return Err(format!(
                "cannot be created: {} is not a directory",
                ancestor.display()
            ));
        }
        Ok(ancestor)
    }

    /// The deprecated `kv_index` field: the chain index is the only KV index,
    /// so `chain` (in any case) is the one value accepted; `positional` names
    /// the index that was removed and is refused with the reason, as is any
    /// other value.
    fn validate_kv_index(config: &RouterConfig) -> ConfigResult<()> {
        let Some(value) = config.kv_index.as_deref() else {
            return Ok(());
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "chain" => Ok(()),
            "positional" => Err(ConfigError::InvalidValue {
                field: "kv_index".to_string(),
                value: value.to_string(),
                reason: "the positional KV index was removed (kv-index 2.0.0): the chain index \
                         is the only KV index, drop kv_index"
                    .to_string(),
            }),
            _ => Err(ConfigError::InvalidValue {
                field: "kv_index".to_string(),
                value: value.to_string(),
                reason: "the chain index is the only KV index, so kv_index takes chain only \
                         (and is deprecated)"
                    .to_string(),
            }),
        }
    }

    fn validate_rl(config: &RouterConfig) -> ConfigResult<()> {
        config
            .rl
            .validate()
            .map_err(|reason| ConfigError::InvalidValue {
                field: "rl".to_string(),
                value: format!(
                    "control_timeout_secs={} fanout_concurrency={}",
                    config.rl.control_timeout_secs, config.rl.fanout_concurrency
                ),
                reason,
            })
    }

    fn validate_discovery(discovery: &DiscoveryConfig, mode: &RoutingMode) -> ConfigResult<()> {
        match discovery {
            DiscoveryConfig::Kubernetes(kubernetes) => {
                Self::validate_kubernetes_discovery(kubernetes, mode)
            }
            DiscoveryConfig::File(file) => Self::validate_file_discovery(file, mode),
        }
    }

    /// The manifest's own contents (worker types against the routing mode
    /// included) are checked each time it is read, not here: it may not exist
    /// yet, and it changes while the router runs.
    fn validate_file_discovery(
        discovery: &FileDiscoveryConfig,
        mode: &RoutingMode,
    ) -> ConfigResult<()> {
        if discovery.path.as_os_str().is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "discovery.path".to_string(),
                value: String::new(),
                reason: "File discovery needs a manifest path".to_string(),
            });
        }

        if discovery.check_interval_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "discovery.check_interval_secs".to_string(),
                value: discovery.check_interval_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        let unsupported = match mode {
            RoutingMode::Regular { .. }
            | RoutingMode::PrefillDecode { .. }
            | RoutingMode::EncodePrefillDecode { .. } => return Ok(()),
            RoutingMode::OpenAI { .. } => "OpenAI",
            RoutingMode::Anthropic { .. } => "Anthropic",
            RoutingMode::Gemini { .. } => "Gemini",
        };
        Err(ConfigError::ValidationFailed {
            reason: format!("{unsupported} mode does not support service discovery"),
        })
    }

    fn validate_kubernetes_discovery(
        discovery: &KubernetesDiscoveryConfig,
        mode: &RoutingMode,
    ) -> ConfigResult<()> {
        if discovery.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "discovery.port".to_string(),
                value: discovery.port.to_string(),
                reason: "Port must be > 0".to_string(),
            });
        }

        if discovery.check_interval_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "discovery.check_interval_secs".to_string(),
                value: discovery.check_interval_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }

        for (field, value) in [
            (
                "discovery.kv_connector_annotation",
                &discovery.kv_connector_annotation,
            ),
            (
                "discovery.kv_engine_id_annotation",
                &discovery.kv_engine_id_annotation,
            ),
        ] {
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed != value {
                return Err(ConfigError::InvalidValue {
                    field: field.to_string(),
                    value: value.clone(),
                    reason: "Annotation name must not be empty or padded with whitespace"
                        .to_string(),
                });
            }
        }

        match mode {
            RoutingMode::Regular { .. } => {
                if discovery.selector.is_empty() {
                    return Err(ConfigError::ValidationFailed {
                        reason: "Regular mode with service discovery requires a non-empty selector"
                            .to_string(),
                    });
                }
            }
            RoutingMode::PrefillDecode { .. } => {
                if discovery.prefill_selector.is_empty() && discovery.decode_selector.is_empty() {
                    return Err(ConfigError::ValidationFailed {
                        reason: "PD mode with service discovery requires at least one non-empty selector (prefill or decode)".to_string(),
                    });
                }
            }
            RoutingMode::EncodePrefillDecode { .. } => {
                if discovery.encode_selector.is_empty()
                    || discovery.prefill_selector.is_empty()
                    || discovery.decode_selector.is_empty()
                {
                    return Err(ConfigError::ValidationFailed {
                        reason: "EPD mode with service discovery requires non-empty encode_selector, prefill_selector, and decode_selector".to_string(),
                    });
                }
            }
            RoutingMode::OpenAI { .. } => {
                return Err(ConfigError::ValidationFailed {
                    reason: "OpenAI mode does not support service discovery".to_string(),
                });
            }
            RoutingMode::Anthropic { .. } => {
                return Err(ConfigError::ValidationFailed {
                    reason: "Anthropic mode does not support service discovery".to_string(),
                });
            }
            RoutingMode::Gemini { .. } => {
                return Err(ConfigError::ValidationFailed {
                    reason: "Gemini mode does not support service discovery".to_string(),
                });
            }
        }

        Ok(())
    }

    fn validate_metrics(metrics: &MetricsConfig) -> ConfigResult<()> {
        if metrics.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "metrics.port".to_string(),
                value: metrics.port.to_string(),
                reason: "Port must be > 0 (0 would request an unstable OS-ephemeral port)"
                    .to_string(),
            });
        }

        if metrics.host.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "metrics.host".to_string(),
                value: metrics.host.clone(),
                reason: "Host cannot be empty".to_string(),
            });
        }

        if let Err(reason) = parse_bind_host(&metrics.host) {
            return Err(ConfigError::InvalidValue {
                field: "metrics.host".to_string(),
                value: metrics.host.clone(),
                reason,
            });
        }

        Ok(())
    }

    fn validate_trace(trace_config: &TraceConfig) -> ConfigResult<()> {
        if !trace_config.enable_trace {
            return Ok(());
        }

        let endpoint = &trace_config.otlp_traces_endpoint;
        let invalid = |reason: String| ConfigError::InvalidValue {
            field: "trace_config.otlp_traces_endpoint".to_string(),
            value: endpoint.clone(),
            reason,
        };
        const EXPECTED: &str = "expected <host>:<port> or http(s)://<host>:<port>, with an \
                                IPv6 literal in brackets, e.g. otel-collector:4317, \
                                127.0.0.1:4317 or [::1]:4317";

        // Parse exactly what the exporter dials (it prefixes `http://` to a
        // scheme-less endpoint), so a value that validates also builds a URI.
        // A last-colon split used to let an unbracketed IPv6 literal
        // (`::1:4317`) or a port-less one (`fd00::1`, host `fd00:` port `1`)
        // through, and startup then aborted in the exporter with an opaque
        // "invalid URI".
        let with_scheme = if endpoint.contains("://") {
            endpoint.clone()
        } else {
            format!("http://{endpoint}")
        };
        let url =
            ::url::Url::parse(&with_scheme).map_err(|e| invalid(format!("{EXPECTED}: {e}")))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(invalid(format!(
                "{EXPECTED}: unsupported scheme '{}'",
                url.scheme()
            )));
        }
        if url.host_str().is_none_or(str::is_empty) {
            return Err(invalid(format!("{EXPECTED}: host part cannot be empty")));
        }
        // A missing port and port 0 are refused; a scheme-default port that
        // was written passes although `Url::port()` drops it.
        if !matches!(Self::written_port(&with_scheme, &url), Some(1..)) {
            return Err(invalid(format!(
                "{EXPECTED}: port must be a number between 1 and 65535"
            )));
        }

        Ok(())
    }

    /// The port of `with_scheme` as written: `Url::port()` drops a
    /// scheme-default port (`:80`, `:443`), so one that was written is read
    /// back off the authority; `None` when the endpoint names no port.
    fn written_port(with_scheme: &str, url: &::url::Url) -> Option<u16> {
        url.port().or_else(|| {
            let authority = with_scheme
                .split_once("://")
                .map_or("", |(_, rest)| rest)
                .split(['/', '?', '#'])
                .next()
                .unwrap_or("");
            url.port_or_known_default()
                .filter(|default| authority.ends_with(&format!(":{default}")))
        })
    }

    fn validate_retry(retry: &RetryConfig) -> ConfigResult<()> {
        if retry.max_retries < 1 {
            return Err(ConfigError::InvalidValue {
                field: "retry.max_retries".to_string(),
                value: retry.max_retries.to_string(),
                reason: "Must be >= 1 (set to 1 to effectively disable retries)".to_string(),
            });
        }
        if retry.initial_backoff_ms == 0 {
            return Err(ConfigError::InvalidValue {
                field: "retry.initial_backoff_ms".to_string(),
                value: retry.initial_backoff_ms.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if retry.max_backoff_ms < retry.initial_backoff_ms {
            return Err(ConfigError::InvalidValue {
                field: "retry.max_backoff_ms".to_string(),
                value: retry.max_backoff_ms.to_string(),
                reason: "Must be >= initial_backoff_ms".to_string(),
            });
        }
        if !retry.backoff_multiplier.is_finite() || retry.backoff_multiplier < 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "retry.backoff_multiplier".to_string(),
                value: retry.backoff_multiplier.to_string(),
                reason: "Must be finite and >= 1.0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&retry.jitter_factor) {
            return Err(ConfigError::InvalidValue {
                field: "retry.jitter_factor".to_string(),
                value: retry.jitter_factor.to_string(),
                reason: "Must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(())
    }

    fn validate_circuit_breaker(cb: &CircuitBreakerConfig) -> ConfigResult<()> {
        if cb.failure_threshold < 1 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.failure_threshold".to_string(),
                value: cb.failure_threshold.to_string(),
                reason: "Must be >= 1 (set to u32::MAX to effectively disable CB)".to_string(),
            });
        }
        if cb.success_threshold < 1 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.success_threshold".to_string(),
                value: cb.success_threshold.to_string(),
                reason: "Must be >= 1".to_string(),
            });
        }
        if cb.timeout_duration_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.timeout_duration_secs".to_string(),
                value: cb.timeout_duration_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        if cb.window_duration_secs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "circuit_breaker.window_duration_secs".to_string(),
                value: cb.window_duration_secs.to_string(),
                reason: "Must be > 0".to_string(),
            });
        }
        Ok(())
    }

    /// The worker health checker's knobs: the interval and the timeout are
    /// periods in whole seconds, the thresholds counts of consecutive
    /// probes, and none of them means anything at 0. An interval of 0 made
    /// the checker a busy loop (thousands of probes per second against every
    /// worker, one core at 100 %); a timeout of 0 times every probe out at
    /// once; a threshold of 0 flips a worker's state without a probe. The
    /// interval is refused with probing disabled too: the checker's loop
    /// still sleeps on it between connect signals.
    fn validate_health_check(health_check: &HealthCheckConfig) -> ConfigResult<()> {
        if !health_check.endpoint.starts_with('/') {
            return Err(ConfigError::InvalidValue {
                field: "health_check.endpoint".to_string(),
                value: health_check.endpoint.clone(),
                reason: "must start with '/': it is appended to every HTTP worker's URL"
                    .to_string(),
            });
        }
        let knobs = [
            (
                "health_check.check_interval_secs",
                health_check.check_interval_secs,
                "--health-check-interval-secs 0 would probe every worker in a busy loop; \
                 --disable-health-check turns probing off",
            ),
            (
                "health_check.timeout_secs",
                health_check.timeout_secs,
                "--health-check-timeout-secs 0 would time every probe out at once",
            ),
            (
                "health_check.failure_threshold",
                u64::from(health_check.failure_threshold),
                "--health-failure-threshold 0 would fail a worker without a probe",
            ),
            (
                "health_check.success_threshold",
                u64::from(health_check.success_threshold),
                "--health-success-threshold 0 would pass a worker without a probe",
            ),
        ];
        for (field, value, consequence) in knobs {
            if value == 0 {
                return Err(ConfigError::InvalidValue {
                    field: field.to_string(),
                    value: value.to_string(),
                    reason: format!("Must be >= 1 ({consequence})"),
                });
            }
        }
        Ok(())
    }

    fn validate_tokenizer_cache(cache: &TokenizerCacheConfig) -> ConfigResult<()> {
        if cache.enable_l0 && cache.l0_max_entries == 0 {
            return Err(ConfigError::InvalidValue {
                field: "tokenizer_cache.l0_max_entries".to_string(),
                value: cache.l0_max_entries.to_string(),
                reason: "Must be > 0 when L0 cache is enabled".to_string(),
            });
        }

        if cache.enable_l0 && cache.l0_max_memory == 0 {
            return Err(ConfigError::InvalidValue {
                field: "tokenizer_cache.l0_max_memory".to_string(),
                value: cache.l0_max_memory.to_string(),
                reason: "Must be > 0 when L0 cache is enabled".to_string(),
            });
        }

        if cache.enable_l1 && cache.l1_max_memory == 0 {
            return Err(ConfigError::InvalidValue {
                field: "tokenizer_cache.l1_max_memory".to_string(),
                value: cache.l1_max_memory.to_string(),
                reason: "Must be > 0 when L1 cache is enabled".to_string(),
            });
        }

        Ok(())
    }

    /// The tokenizer source the gateway will load is checked before anything
    /// binds: `tokenizer_path`, else `model_path` (the source start-up and
    /// worker registration read, so a `model_path` beside an explicit
    /// tokenizer path is never read as one and not checked), when it names a
    /// path on this host (absolute, explicitly relative, more than one
    /// separator, or present on disk; anything else is a hub id the loader
    /// fetches lazily, as before), and `chat_template` always. A path that
    /// exists must be readable, a directory must hold a layout the loader
    /// knows, a file must not be empty. A path that does not exist is left
    /// alone when a gRPC worker can still register (the gateway then loads
    /// the tokenizer from the first one, and a model path that exists on the
    /// worker hosts only is a supported layout; see
    /// [`a_grpc_worker_may_register`]) and is a configuration error when
    /// none can. The load itself stays where it was, a background job at
    /// start-up and at worker registration: this is the part of it that can
    /// be known before serving. With autoload disabled the paths are never
    /// read and are not checked.
    fn validate_tokenizer_sources(config: &RouterConfig) -> ConfigResult<()> {
        if config.disable_tokenizer_autoload {
            return Ok(());
        }
        let (field, source) = match config.tokenizer_path.as_deref() {
            Some(tokenizer_path) => ("tokenizer_path", Some(tokenizer_path)),
            None => ("model_path", config.model_path.as_deref()),
        };
        if let Some(source) = source.filter(|source| names_a_local_path(source)) {
            match local_tokenizer_source_error(Path::new(source)) {
                Ok(()) => {}
                Err(SourceProblem::Missing) if a_grpc_worker_may_register(config) => {}
                Err(problem) => {
                    return Err(ConfigError::InvalidValue {
                        field: field.to_string(),
                        value: source.to_string(),
                        reason: problem.reason(),
                    });
                }
            }
        }
        if let Some(template) = config.chat_template.as_deref() {
            text_file_error(Path::new(template)).map_err(|reason| ConfigError::InvalidValue {
                field: "chat_template".to_string(),
                value: template.to_string(),
                reason,
            })?;
        }
        Ok(())
    }

    fn validate_mtls(config: &RouterConfig) -> ConfigResult<()> {
        if let Some(identity) = &config.client_identity {
            if identity.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: "Client identity cannot be empty".to_string(),
                });
            }
        }

        for (idx, ca_cert) in config.ca_certificates.iter().enumerate() {
            if ca_cert.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: format!("CA certificate at index {idx} cannot be empty"),
                });
            }
        }

        Ok(())
    }

    fn validate_compatibility(config: &RouterConfig) -> ConfigResult<()> {
        let has_service_discovery = config.discovery.is_some();
        let invalid_decode_policy = match &config.mode {
            RoutingMode::PrefillDecode { decode_policy, .. } if !config.enable_igw => {
                !has_service_discovery && matches!(decode_policy, Some(PolicyConfig::Bucket { .. }))
            }
            RoutingMode::PrefillDecode { .. } | RoutingMode::EncodePrefillDecode { .. } => {
                matches!(
                    config.mode.get_decode_policy(&config.policy),
                    PolicyConfig::Bucket { .. }
                )
            }
            _ => false,
        };
        if invalid_decode_policy {
            return Err(ConfigError::IncompatibleConfig {
                reason: "Decode policy should not be allowed to be bucket".to_string(),
            });
        }

        // IGW may receive workers dynamically, so worker-count validation cannot
        // reason about its eventual topology. Mode-level compatibility still applies.
        if config.enable_igw {
            return Ok(());
        }

        Self::validate_mtls(config)?;

        // Under dp_aware each URL registers one worker per DP rank.
        if !has_service_discovery && !config.dp_aware {
            if let PolicyConfig::PowerOfTwo { .. } = &config.policy {
                let worker_count = config.mode.worker_count();
                if worker_count < 2 {
                    return Err(ConfigError::IncompatibleConfig {
                        reason: "Power-of-two policy requires at least 2 workers".to_string(),
                    });
                }
            }

            if let RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy,
                decode_policy,
            } = &config.mode
            {
                if let Some(PolicyConfig::PowerOfTwo { .. }) = prefill_policy {
                    if prefill_urls.len() < 2 {
                        return Err(ConfigError::IncompatibleConfig {
                            reason: "Power-of-two policy for prefill requires at least 2 prefill workers".to_string(),
                        });
                    }
                }

                if let Some(PolicyConfig::PowerOfTwo { .. }) = decode_policy {
                    if decode_urls.len() < 2 {
                        return Err(ConfigError::IncompatibleConfig {
                            reason:
                                "Power-of-two policy for decode requires at least 2 decode workers"
                                    .to_string(),
                        });
                    }
                }
            }

            if let RoutingMode::EncodePrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy,
                decode_policy,
                ..
            } = &config.mode
            {
                let effective_prefill_policy = prefill_policy.as_ref().unwrap_or(&config.policy);
                let effective_decode_policy = decode_policy.as_ref().unwrap_or(&config.policy);

                if matches!(effective_prefill_policy, PolicyConfig::PowerOfTwo { .. })
                    && prefill_urls.len() < 2
                {
                    return Err(ConfigError::IncompatibleConfig {
                        reason:
                            "Power-of-two policy for prefill requires at least 2 prefill workers"
                                .to_string(),
                    });
                }

                if matches!(effective_decode_policy, PolicyConfig::PowerOfTwo { .. })
                    && decode_urls.len() < 2
                {
                    return Err(ConfigError::IncompatibleConfig {
                        reason: "Power-of-two policy for decode requires at least 2 decode workers"
                            .to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    fn validate_urls(urls: &[String]) -> ConfigResult<()> {
        for url in urls {
            validate_worker_url(url)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::ConnectionMode;

    /// The OTLP endpoint is validated as the URL the exporter dials, so an
    /// unbracketed IPv6 literal is a configuration error with the bracketed
    /// form in the message instead of an "invalid URI" abort at startup.
    #[test]
    fn otlp_endpoint_must_be_a_url_the_exporter_can_dial() {
        let trace = |endpoint: &str| TraceConfig {
            enable_trace: true,
            otlp_traces_endpoint: endpoint.to_string(),
        };
        for ok in [
            "otel-collector:4317",
            "localhost:4317",
            "127.0.0.1:4317",
            "[::1]:4317",
            "[fd00::1]:4317",
            "http://[::1]:4317",
            "http://127.0.0.1:4317",
            "https://otel-collector.example:4317",
            "http://otel-collector:80",
        ] {
            ConfigValidator::validate_trace(&trace(ok))
                .unwrap_or_else(|e| panic!("{ok} should validate: {e}"));
        }
        for bad in [
            "::1:4317",
            "fd00::1",
            "http://::1:4317",
            "[::1]",
            "otel-collector",
            "otel-collector:0",
            "otel-collector:70000",
            "grpc://otel-collector:4317",
            ":4317",
            "",
        ] {
            let err = ConfigValidator::validate_trace(&trace(bad)).unwrap_err();
            assert!(
                matches!(&err, ConfigError::InvalidValue { field, .. }
                    if field == "trace_config.otlp_traces_endpoint"),
                "{bad:?}: {err}"
            );
            assert!(err.to_string().contains("[::1]:4317"), "{bad:?}: {err}");
        }
        // Tracing off: the endpoint is not looked at.
        ConfigValidator::validate_trace(&TraceConfig {
            enable_trace: false,
            otlp_traces_endpoint: "::1:4317".to_string(),
        })
        .unwrap();
    }

    #[test]
    fn bind_hosts_accept_ipv6_with_or_without_brackets() {
        for (host, expected) in [
            ("::", "[::]:30000"),
            ("[::]", "[::]:30000"),
            ("::1", "[::1]:30000"),
            ("[fd00::1]", "[fd00::1]:30000"),
            ("0.0.0.0", "0.0.0.0:30000"),
            ("127.0.0.1", "127.0.0.1:30000"),
        ] {
            assert_eq!(
                bind_socket_addr(host, 30000).unwrap().to_string(),
                expected,
                "host {host:?}"
            );
        }
        for bad in [
            "",
            "localhost",
            "[::",
            "::]",
            "[]",
            "0.0.0.0:30000",
            "[::1]:30000",
        ] {
            let err = parse_bind_host(bad).unwrap_err();
            assert!(err.contains("invalid bind host"), "host {bad:?}: {err}");
        }
    }

    #[test]
    fn server_and_metrics_hosts_are_validated_as_bind_hosts() {
        let config = |host: &str, metrics_host: &str| RouterConfig {
            host: host.to_string(),
            metrics: Some(MetricsConfig {
                port: 29000,
                host: metrics_host.to_string(),
            }),
            mode: RoutingMode::Regular {
                worker_urls: vec!["http://worker:8000".to_string()],
            },
            ..Default::default()
        };
        for (host, metrics_host) in [("[::]", "::"), ("::", "[::]"), ("0.0.0.0", "0.0.0.0")] {
            config(host, metrics_host)
                .validate()
                .unwrap_or_else(|e| panic!("host {host:?} metrics {metrics_host:?}: {e}"));
        }
        match config("localhost", "::").validate() {
            Err(ConfigError::InvalidValue { field, .. }) => assert_eq!(field, "host"),
            other => panic!("expected an invalid host, got {other:?}"),
        }
        match config("[::]", "metrics").validate() {
            Err(ConfigError::InvalidValue { field, .. }) => assert_eq!(field, "metrics.host"),
            other => panic!("expected an invalid metrics host, got {other:?}"),
        }
    }

    #[test]
    fn igw_disaggregated_modes_reject_bucket_decode_policy() {
        let bucket = || PolicyConfig::Bucket {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            bucket_adjust_interval_secs: 5,
        };
        let mut config = RouterConfig {
            discovery: Some(KubernetesDiscoveryConfig::default().into()),
            mode: RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: Some(bucket()),
            },
            ..Default::default()
        };

        assert!(ConfigValidator::validate_compatibility(&config).is_ok());
        config.enable_igw = true;
        assert!(ConfigValidator::validate_compatibility(&config).is_err());

        if let RoutingMode::PrefillDecode { decode_policy, .. } = &mut config.mode {
            *decode_policy = None;
        }
        config.policy = bucket();
        assert!(ConfigValidator::validate_compatibility(&config).is_err());

        config.policy = PolicyConfig::Random;
        config.mode = RoutingMode::EncodePrefillDecode {
            encode_urls: vec![],
            prefill_urls: vec![],
            decode_urls: vec![],
            encode_policy: None,
            prefill_policy: None,
            decode_policy: Some(bucket()),
        };
        assert!(ConfigValidator::validate_compatibility(&config).is_err());
    }

    #[test]
    fn prefix_hash_policy_cache_boundaries_are_validated() {
        let config = RouterConfig {
            policy: PolicyConfig::PrefixHash {
                prefix_token_count: 256,
                load_factor: 1.25,
                balance_abs_threshold: 10,
                cache_boundaries: vec![8192, 2048],
            },
            ..Default::default()
        };
        assert!(matches!(
            ConfigValidator::validate(&config),
            Err(ConfigError::InvalidValue { ref field, .. }) if field == "cache_boundaries"
        ));
    }

    #[test]
    fn zero_mm_per_request_image_limit_is_rejected() {
        let config = RouterConfig {
            mm_per_request_image_limit: Some(0),
            ..Default::default()
        };
        assert!(matches!(
            ConfigValidator::validate(&config),
            Err(ConfigError::InvalidValue { ref field, .. }) if field == "mm_per_request_image_limit"
        ));

        let config = RouterConfig {
            mm_per_request_image_limit: Some(128),
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    /// The deprecated `kv_index` field, as a config file or a binding sets
    /// it: `chain` in any case is accepted, the removed `positional` index
    /// is refused with the removal in the reason, any other value is refused.
    #[test]
    fn kv_index_takes_chain_only() {
        for value in ["chain", "Chain", " CHAIN "] {
            let config = RouterConfig {
                kv_index: Some(value.to_string()),
                ..Default::default()
            };
            assert!(ConfigValidator::validate(&config).is_ok(), "{value:?}");
        }

        let config = RouterConfig {
            kv_index: Some("positional".to_string()),
            ..Default::default()
        };
        assert!(matches!(
            ConfigValidator::validate(&config),
            Err(ConfigError::InvalidValue { ref field, ref reason, .. })
                if field == "kv_index" && reason.contains("positional KV index was removed")
        ));

        let config = RouterConfig {
            kv_index: Some("run".to_string()),
            ..Default::default()
        };
        assert!(matches!(
            ConfigValidator::validate(&config),
            Err(ConfigError::InvalidValue { ref field, .. }) if field == "kv_index"
        ));
    }

    /// A `RouterConfig` built directly, as a language binding or a library
    /// caller builds one: the validator refuses each malformed shape by its
    /// field, and accepts the well-formed values.
    #[test]
    fn malformed_value_shapes_are_refused_by_name_on_a_directly_built_router_config() {
        let defaults = RouterConfig::default;
        let cases: Vec<(&str, RouterConfig)> = vec![
            (
                "health_check.endpoint",
                RouterConfig {
                    health_check: HealthCheckConfig {
                        endpoint: "health".to_owned(),
                        ..HealthCheckConfig::default()
                    },
                    ..defaults()
                },
            ),
            (
                "request_id_headers",
                RouterConfig {
                    request_id_headers: Some(vec!["bad header".to_owned()]),
                    ..defaults()
                },
            ),
            (
                "storage_context_headers",
                RouterConfig {
                    storage_context_headers: [("bad header".to_owned(), "tenant".to_owned())]
                        .into_iter()
                        .collect(),
                    ..defaults()
                },
            ),
            (
                "api_key",
                RouterConfig {
                    api_key: Some(" ".to_owned()),
                    ..defaults()
                },
            ),
        ];
        for (field, config) in cases {
            match ConfigValidator::validate(&config) {
                Err(ConfigError::InvalidValue { field: named, .. }) => {
                    assert_eq!(named, field);
                }
                other => panic!("{field}: expected a refusal by name, got {other:?}"),
            }
        }
        let accepted = RouterConfig {
            health_check: HealthCheckConfig {
                endpoint: "/health".to_owned(),
                ..HealthCheckConfig::default()
            },
            request_id_headers: Some(vec!["x-request-id".to_owned()]),
            storage_context_headers: [("x-tenant".to_owned(), "tenant".to_owned())]
                .into_iter()
                .collect(),
            api_key: Some("secret".to_owned()),
            ..defaults()
        };
        assert!(ConfigValidator::validate(&accepted).is_ok());
    }

    fn scheduler_config(path: Option<&str>) -> RouterConfig {
        RouterConfig {
            priority_scheduler_enabled: true,
            priority_scheduler_config: path.map(str::to_owned),
            ..Default::default()
        }
    }

    /// A class entry with every field, `queue_timeout_secs` set to `timeout`.
    fn class_yaml(class: &str, timeout: u64) -> String {
        format!(
            "classes:\n  {class}:\n    reserved_floor: 1\n    queue_size: 16\n    \
             queue_timeout_secs: {timeout}\n    starvation_threshold_secs: 10\n    \
             can_preempt: false\n"
        )
    }

    /// The `(value, reason)` of the `priority_scheduler_config` error a
    /// configuration is refused with.
    fn refused_scheduler_config(config: &RouterConfig) -> (String, String) {
        match ConfigValidator::validate(config) {
            Err(ConfigError::InvalidValue {
                field,
                value,
                reason,
            }) if field == "priority_scheduler_config" => (value, reason),
            other => panic!("expected a priority_scheduler_config error, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_priority_scheduler_config_is_a_configuration_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.yaml");
        let path = path.to_str().unwrap();
        let (value, reason) = refused_scheduler_config(&scheduler_config(Some(path)));
        assert_eq!(value, path);
        assert!(
            reason.starts_with("reading ") && reason.contains(path),
            "the reason names the file and the read error: {reason}"
        );
    }

    #[test]
    fn an_unparsable_priority_scheduler_config_is_a_configuration_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage.yaml");
        std::fs::write(&path, "not: [valid: yaml\n").unwrap();
        let path = path.to_str().unwrap();
        let (value, reason) = refused_scheduler_config(&scheduler_config(Some(path)));
        assert_eq!(value, path);
        assert!(
            reason.starts_with("parsing ") && reason.contains(path),
            "the reason names the file and the parse error: {reason}"
        );
    }

    #[test]
    fn an_invalid_priority_scheduler_config_is_a_configuration_error() {
        let dir = tempfile::tempdir().unwrap();
        // A class the scheduler does not know.
        let unknown_class = dir.path().join("unknown-class.yaml");
        std::fs::write(&unknown_class, class_yaml("gold", 1)).unwrap();
        let (value, reason) = refused_scheduler_config(&scheduler_config(unknown_class.to_str()));
        assert_eq!(value, unknown_class.to_str().unwrap());
        assert!(
            reason.contains("gold"),
            "the reason names the unknown class: {reason}"
        );
        // A value the scheduler's own validation refuses.
        let zero_timeout = dir.path().join("zero-timeout.yaml");
        std::fs::write(&zero_timeout, class_yaml("bulk", 0)).unwrap();
        let (value, reason) = refused_scheduler_config(&scheduler_config(zero_timeout.to_str()));
        assert_eq!(value, zero_timeout.to_str().unwrap());
        assert!(
            reason.contains("queue_timeout_secs must be > 0"),
            "the reason is the scheduler's own: {reason}"
        );
    }

    #[test]
    fn an_unknown_priority_scheduler_default_max_class_is_a_configuration_error() {
        for unknown in ["gold", "urgent"] {
            let config = RouterConfig {
                priority_scheduler_default_max_class: unknown.to_owned(),
                ..scheduler_config(None)
            };
            match ConfigValidator::validate(&config) {
                Err(ConfigError::InvalidValue {
                    field,
                    value,
                    reason,
                }) if field == "priority_scheduler_default_max_class" => {
                    assert_eq!(value, unknown);
                    assert!(
                        ["system", "interactive", "default", "bulk"]
                            .iter()
                            .all(|class| reason.contains(class)),
                        "the reason names the four classes: {reason}"
                    );
                }
                other => panic!(
                    "`{unknown}` was not refused as a priority_scheduler_default_max_class error: {other:?}"
                ),
            }
        }
        // The four names in any case are accepted.
        for known in ["system", "Interactive", " default ", "BULK"] {
            let config = RouterConfig {
                priority_scheduler_default_max_class: known.to_owned(),
                ..scheduler_config(None)
            };
            assert!(ConfigValidator::validate(&config).is_ok(), "{known}");
        }
        // Not enabled: the flag is not read, like the path.
        let config = RouterConfig {
            priority_scheduler_default_max_class: "gold".to_owned(),
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn a_priority_scheduler_config_with_an_unknown_key_is_a_configuration_error() {
        let dir = tempfile::tempdir().unwrap();
        let unknown_key = dir.path().join("unknown-key.yaml");
        std::fs::write(
            &unknown_key,
            "classes: {}\ntenant_policies: {}\nextra_field: 1\n",
        )
        .unwrap();
        let (value, reason) = refused_scheduler_config(&scheduler_config(unknown_key.to_str()));
        assert_eq!(value, unknown_key.to_str().unwrap());
        assert!(
            reason.starts_with("parsing ") && reason.contains("extra_field"),
            "the reason names the file and the unknown key: {reason}"
        );
        // A misspelt section is an unknown key, not an empty section.
        let misspelt = dir.path().join("misspelt.yaml");
        std::fs::write(
            &misspelt,
            "tenant_polices:\n  \"header:a\":\n    max_class: bulk\n",
        )
        .unwrap();
        let (_, reason) = refused_scheduler_config(&scheduler_config(misspelt.to_str()));
        assert!(
            reason.contains("tenant_polices"),
            "the reason names the misspelt section: {reason}"
        );
    }

    #[test]
    fn a_valid_or_absent_priority_scheduler_config_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let valid = dir.path().join("scheduler.yaml");
        let tenant_policies = "tenant_policies:\n  \"header:a\":\n    max_class: bulk\n";
        std::fs::write(
            &valid,
            format!("{}{tenant_policies}", class_yaml("interactive", 5)),
        )
        .unwrap();
        assert!(ConfigValidator::validate(&scheduler_config(valid.to_str())).is_ok());
        // Enabled without a file: the built-in class defaults.
        assert!(ConfigValidator::validate(&scheduler_config(None)).is_ok());
        // Not enabled: the path is not read, as for the tenant rate limiter.
        let missing = dir.path().join("missing.yaml");
        let config = RouterConfig {
            priority_scheduler_config: missing.to_str().map(str::to_owned),
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn zero_runtime_worker_threads_is_rejected() {
        let config = RouterConfig {
            runtime_worker_threads: Some(0),
            ..Default::default()
        };
        assert!(matches!(
            ConfigValidator::validate(&config),
            Err(ConfigError::InvalidValue { ref field, .. }) if field == "runtime_worker_threads"
        ));

        let config = RouterConfig {
            runtime_worker_threads: Some(4),
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    fn log_dir_config(log_dir: &str) -> RouterConfig {
        RouterConfig {
            log_dir: Some(log_dir.to_string()),
            ..Default::default()
        }
    }

    /// The `(value, reason)` of the `log_dir` error a configuration is
    /// refused with.
    fn refused_log_dir(config: &RouterConfig) -> (String, String) {
        match ConfigValidator::validate(config) {
            Err(ConfigError::InvalidValue {
                field,
                value,
                reason,
            }) if field == "log_dir" => (value, reason),
            other => panic!("expected a log_dir error, got {other:?}"),
        }
    }

    #[test]
    fn a_log_dir_that_is_a_file_is_a_configuration_error() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a-file");
        std::fs::write(&file, "not a directory").unwrap();
        let file = file.to_str().unwrap();
        let (value, reason) = refused_log_dir(&log_dir_config(file));
        assert_eq!(value, file);
        assert!(reason.contains("not a directory"), "{reason}");
    }

    #[test]
    fn a_log_dir_that_cannot_be_created_is_a_configuration_error() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a-file");
        std::fs::write(&file, "not a directory").unwrap();
        let under_a_file = file.join("logs");
        let file = file.to_str().unwrap();
        let under = under_a_file.to_str().unwrap();
        let (value, reason) = refused_log_dir(&log_dir_config(under));
        assert_eq!(value, under);
        assert!(
            reason.contains("cannot be created") && reason.contains(file),
            "the reason names the file in the way: {reason}"
        );
        assert!(!under_a_file.exists());
    }

    #[cfg(unix)]
    #[test]
    fn a_log_dir_the_gateway_cannot_write_is_a_configuration_error() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let locked = dir.path().join("locked");
        std::fs::create_dir(&locked).unwrap();
        let writable = std::fs::metadata(&locked).unwrap().permissions();
        let mut read_only = writable.clone();
        read_only.set_mode(0o555);
        std::fs::set_permissions(&locked, read_only).unwrap();
        // A privileged process writes regardless of the mode: nothing to test.
        let privileged = std::fs::File::create(locked.join("probe")).is_ok();
        let verdict = ConfigValidator::validate(&log_dir_config(locked.to_str().unwrap()));
        std::fs::set_permissions(&locked, writable).unwrap();
        if privileged {
            return;
        }
        match verdict {
            Err(ConfigError::InvalidValue { field, reason, .. }) if field == "log_dir" => {
                assert!(reason.contains("not writable"), "{reason}");
            }
            other => panic!("expected a log_dir error, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_log_dir_under_a_writable_directory_is_accepted_and_not_created() {
        let dir = tempfile::tempdir().unwrap();
        let fresh = dir.path().join("created").join("logs");
        assert!(ConfigValidator::validate(&log_dir_config(fresh.to_str().unwrap())).is_ok());
        assert!(
            !fresh.exists() && !dir.path().join("created").exists(),
            "validation leaves no directory behind; logging start-up creates it"
        );
        assert!(ConfigValidator::validate(&log_dir_config(dir.path().to_str().unwrap())).is_ok());
        assert!(ConfigValidator::validate(&RouterConfig::default()).is_ok());
    }

    /// A relative `--log-dir` (a systemd `WorkingDirectory=`, a container
    /// `WORKDIR`, a local run) that does not exist yet is accepted: its
    /// nearest existing ancestor is the working directory, which the walk
    /// reaches as "" and must read as ".". Nothing is created.
    #[test]
    fn a_missing_relative_log_dir_under_the_working_directory_is_accepted() {
        let leaf = format!(
            "log-dir-validation-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        );
        for relative in [leaf.clone(), format!("{leaf}/run/logs")] {
            assert!(
                ConfigValidator::validate(&log_dir_config(&relative)).is_ok(),
                "{relative}"
            );
            assert!(
                !Path::new(&leaf).exists(),
                "{relative}: created by the validation"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_missing_log_dir_under_a_directory_the_gateway_cannot_write_is_a_configuration_error() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let locked = dir.path().join("locked");
        std::fs::create_dir(&locked).unwrap();
        let writable = std::fs::metadata(&locked).unwrap().permissions();
        let mut read_only = writable.clone();
        read_only.set_mode(0o555);
        std::fs::set_permissions(&locked, read_only).unwrap();
        let privileged = std::fs::File::create(locked.join("probe")).is_ok();
        let under = locked.join("gateway").join("logs");
        let verdict = ConfigValidator::validate(&log_dir_config(under.to_str().unwrap()));
        std::fs::set_permissions(&locked, writable).unwrap();
        if privileged {
            return;
        }
        match verdict {
            Err(ConfigError::InvalidValue { field, reason, .. }) if field == "log_dir" => {
                assert!(
                    reason.contains("cannot be created")
                        && reason.contains(locked.to_str().unwrap())
                        && reason.contains("not writable"),
                    "the reason names the ancestor: {reason}"
                );
            }
            other => panic!("expected a log_dir error, got {other:?}"),
        }
        assert!(!under.exists());
    }

    /// The Redis configuration's own rules run through the shared validation
    /// too, so the bindings and a directly built configuration refuse a
    /// retention of 0 days (every record expired as it is written) as the
    /// command line does; unset, the retention is persistent and passes.
    #[test]
    fn a_redis_retention_of_zero_days_is_refused_through_the_shared_validation() {
        let redis = |retention_days| RedisConfig {
            url: "redis://localhost:6379".to_string(),
            pool_max: 4,
            op_timeout_secs: RedisConfig::default_op_timeout_secs(),
            schema: None,
            retention_days,
        };
        let config = RouterConfig {
            history_backend: HistoryBackend::Redis,
            redis: Some(redis(Some(0))),
            ..Default::default()
        };
        match ConfigValidator::validate(&config) {
            Err(ConfigError::ValidationFailed { reason }) => {
                assert!(
                    reason.contains("retention_days must be at least 1"),
                    "{reason}"
                );
            }
            other => panic!("expected the Redis rule, got {other:?}"),
        }
        let config = RouterConfig {
            history_backend: HistoryBackend::Redis,
            redis: Some(redis(None)),
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_ok());
        // Another backend leaves the Redis block unread.
        let config = RouterConfig {
            redis: Some(redis(Some(0))),
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    /// The shared validation holds the stores' numeric rules and leaves the
    /// connection string to the driver: a Redis socket path and a Postgres
    /// key=value string, which the drivers accept and the command line's URL
    /// check does not, pass it; a zero bound in the same configuration does
    /// not.
    #[test]
    fn the_shared_validation_leaves_the_store_connection_string_to_the_driver() {
        let redis = |pool_max| RedisConfig {
            url: "unix:///run/redis.sock".to_string(),
            pool_max,
            op_timeout_secs: RedisConfig::default_op_timeout_secs(),
            schema: None,
            retention_days: Some(7),
        };
        let config = RouterConfig {
            history_backend: HistoryBackend::Redis,
            redis: Some(redis(4)),
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_ok());
        let config = RouterConfig {
            history_backend: HistoryBackend::Redis,
            redis: Some(redis(0)),
            ..Default::default()
        };
        match ConfigValidator::validate(&config) {
            Err(ConfigError::ValidationFailed { reason }) => {
                assert!(reason.contains("pool_max"), "{reason}");
            }
            other => panic!("expected the Redis bound, got {other:?}"),
        }

        let postgres = |op_timeout_secs| PostgresConfig {
            db_url: "host=/var/run/postgresql dbname=smg user=smg".to_string(),
            pool_max: 4,
            op_timeout_secs,
            schema: None,
        };
        let config = RouterConfig {
            history_backend: HistoryBackend::Postgres,
            postgres: Some(postgres(5)),
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_ok());
        let config = RouterConfig {
            history_backend: HistoryBackend::Postgres,
            postgres: Some(postgres(0)),
            ..Default::default()
        };
        match ConfigValidator::validate(&config) {
            Err(ConfigError::ValidationFailed { reason }) => {
                assert!(reason.contains("op_timeout_secs"), "{reason}");
            }
            other => panic!("expected the Postgres bound, got {other:?}"),
        }
    }

    #[test]
    fn mesh_server_name_with_colon_is_rejected() {
        assert!(matches!(
            validate_mesh_server_name("node:a"),
            Err(ConfigError::InvalidValue { ref field, .. }) if field == "mesh_server_name"
        ));
    }

    #[test]
    fn empty_mesh_server_name_is_rejected() {
        assert!(matches!(
            validate_mesh_server_name(""),
            Err(ConfigError::InvalidValue { ref field, .. }) if field == "mesh_server_name"
        ));
    }

    #[test]
    fn valid_mesh_server_name_is_accepted() {
        assert!(validate_mesh_server_name("node-a").is_ok());
    }

    #[test]
    fn worker_url_accepts_allowed_schemes() {
        for url in [
            "http://10.0.0.5:8000",
            "https://worker.example.com",
            "grpc://10.0.0.5:50051",
            "grpcs://worker.example.com:443",
        ] {
            assert!(validate_worker_url(url).is_ok(), "expected {url} to pass");
        }
    }

    #[test]
    fn worker_url_rejects_non_lowercase_scheme() {
        // normalize_url in the AddWorker workflow matches schemes
        // case-sensitively, so `HTTP://…` would be mangled into
        // `http://HTTP://…` and orphan the reservation just like a
        // schemeless URL would.
        assert!(validate_worker_url("HTTP://10.0.0.5:8000").is_err());
        assert!(validate_worker_url("Grpc://10.0.0.5:50051").is_err());
    }

    #[test]
    fn worker_url_rejects_empty() {
        assert!(matches!(
            validate_worker_url(""),
            Err(ConfigError::InvalidValue { ref field, .. }) if field == "worker_url"
        ));
    }

    #[test]
    fn worker_url_rejects_schemeless_host_port() {
        // The #1533 case: bare host:port input would be rewritten by the
        // AddWorker workflow, orphaning the API-layer ID reservation.
        assert!(matches!(
            validate_worker_url("10.0.0.5:8000"),
            Err(ConfigError::InvalidValue { ref field, .. }) if field == "worker_url"
        ));
    }

    #[test]
    fn worker_url_rejects_disallowed_scheme() {
        assert!(validate_worker_url("ftp://example.com:21").is_err());
    }

    #[test]
    fn worker_url_rejects_unparsable() {
        assert!(validate_worker_url("http://").is_err());
    }

    #[test]
    fn test_validate_regular_mode() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker:8000".to_string()],
            },
            PolicyConfig::Random,
        );

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_model_aliases() {
        let mut config = regular_mode_config();
        config.model_aliases = std::collections::HashMap::from([
            ("GLM-5.2-Coding".to_string(), "GLM-5.2".to_string()),
            ("glm-5.2".to_string(), "GLM-5.2".to_string()),
        ]);
        assert!(ConfigValidator::validate(&config).is_ok());

        for (alias, canonical) in [
            ("", "GLM-5.2"),
            ("GLM-5.2-Coding", ""),
            ("GLM-5.2", "GLM-5.2"),
        ] {
            config.model_aliases =
                std::collections::HashMap::from([(alias.to_string(), canonical.to_string())]);
            assert!(matches!(
                ConfigValidator::validate(&config),
                Err(ConfigError::InvalidValue { ref field, .. }) if field == "model_aliases"
            ));
        }
    }

    fn regular_mode_config() -> RouterConfig {
        RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker:8000".to_string()],
            },
            PolicyConfig::Random,
        )
    }

    #[test]
    fn routing_key_headers_validated_as_header_names() {
        let mut config = regular_mode_config();
        config.routing_key_override.headers =
            vec!["x-routing-key".to_string(), "x-smg-routing-key".to_string()];
        assert!(ConfigValidator::validate(&config).is_ok());

        for bad in ["", "has space", "bad\nname"] {
            config.routing_key_override.headers = vec![bad.to_string()];
            assert!(matches!(
                ConfigValidator::validate(&config),
                Err(ConfigError::ValidationFailed { ref reason })
                    if reason.contains("routing_key_override.headers")
            ));
        }
    }

    #[test]
    fn test_validate_distinct_tenant_api_keys_accepted() {
        let mut config = regular_mode_config();
        config.tenant_api_keys = vec![
            TenantApiKeyEntry {
                tenant_id: "team-a".to_string(),
                key: "secret-a".to_string(),
            },
            TenantApiKeyEntry {
                tenant_id: "team-b".to_string(),
                key: "secret-b".to_string(),
            },
        ];

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_duplicate_tenant_api_keys_rejected() {
        let mut config = regular_mode_config();
        config.tenant_api_keys = vec![
            TenantApiKeyEntry {
                tenant_id: "team-a".to_string(),
                key: "shared-secret".to_string(),
            },
            TenantApiKeyEntry {
                tenant_id: "team-b".to_string(),
                key: "shared-secret".to_string(),
            },
        ];

        let err = ConfigValidator::validate(&config).unwrap_err();
        assert!(matches!(err, ConfigError::ValidationFailed { .. }));
        let message = err.to_string();
        assert!(message.contains("team-a"));
        assert!(message.contains("team-b"));
        assert!(
            !message.contains("shared-secret"),
            "error must not leak the credential value: {message}"
        );
    }

    #[test]
    fn test_validate_tenant_api_key_matching_shared_api_key_rejected() {
        let mut config = regular_mode_config();
        config.api_key = Some("shared-secret".to_string());
        config.tenant_api_keys = vec![TenantApiKeyEntry {
            tenant_id: "team-a".to_string(),
            key: "shared-secret".to_string(),
        }];

        let err = ConfigValidator::validate(&config).unwrap_err();
        assert!(matches!(err, ConfigError::ValidationFailed { .. }));
        let message = err.to_string();
        assert!(message.contains("team-a"));
        assert!(message.contains("shared api_key"));
        assert!(
            !message.contains("shared-secret"),
            "error must not leak the credential value: {message}"
        );
    }

    #[test]
    fn test_validate_empty_tenant_id_rejected() {
        let mut config = regular_mode_config();
        config.tenant_api_keys = vec![TenantApiKeyEntry {
            tenant_id: String::new(),
            key: "some-secret".to_string(),
        }];

        let err = ConfigValidator::validate(&config).unwrap_err();
        assert!(matches!(err, ConfigError::ValidationFailed { .. }));
        assert!(err.to_string().contains("tenant_id"));
    }

    #[test]
    fn test_validate_whitespace_only_tenant_id_rejected() {
        let mut config = regular_mode_config();
        config.tenant_api_keys = vec![TenantApiKeyEntry {
            tenant_id: "   ".to_string(),
            key: "some-secret".to_string(),
        }];

        assert!(ConfigValidator::validate(&config).is_err());
    }

    /// A padded-but-otherwise-valid tenant_id (e.g. supplied via a config
    /// file or binding, bypassing the CLI parser's own trim) must be
    /// rejected rather than silently resolving to a different `auth:` key
    /// than the canonical, unpadded tenant_id.
    #[test]
    fn test_validate_padded_tenant_id_rejected() {
        let mut config = regular_mode_config();
        config.tenant_api_keys = vec![TenantApiKeyEntry {
            tenant_id: " team-a ".to_string(),
            key: "some-secret".to_string(),
        }];

        let err = ConfigValidator::validate(&config).unwrap_err();
        assert!(matches!(err, ConfigError::ValidationFailed { .. }));
        assert!(err.to_string().contains("whitespace"));
    }

    /// Same asymmetry as the padded-tenant_id case, but for `key`: the CLI
    /// trims it, config-file/binding entries don't, so an untrimmed key
    /// would hash differently than intended and silently evade the
    /// duplicate-value check.
    #[test]
    fn test_validate_padded_key_rejected() {
        let mut config = regular_mode_config();
        config.tenant_api_keys = vec![TenantApiKeyEntry {
            tenant_id: "team-a".to_string(),
            key: " some-secret ".to_string(),
        }];

        let err = ConfigValidator::validate(&config).unwrap_err();
        assert!(matches!(err, ConfigError::ValidationFailed { .. }));
        let message = err.to_string();
        assert!(message.contains("team-a"));
        assert!(message.contains("whitespace"));
        assert!(
            !message.contains("some-secret"),
            "error must not leak the credential value: {message}"
        );
    }

    #[test]
    fn test_validate_empty_key_rejected() {
        let mut config = regular_mode_config();
        config.tenant_api_keys = vec![TenantApiKeyEntry {
            tenant_id: "team-a".to_string(),
            key: String::new(),
        }];

        // An empty key would make `Authorization: Bearer ` (empty token) a
        // valid credential for this tenant.
        let err = ConfigValidator::validate(&config).unwrap_err();
        assert!(matches!(err, ConfigError::ValidationFailed { .. }));
        let message = err.to_string();
        assert!(message.contains("team-a"));
        assert!(message.contains("key"));
    }

    #[test]
    fn test_validate_whitespace_only_key_rejected() {
        let mut config = regular_mode_config();
        config.tenant_api_keys = vec![TenantApiKeyEntry {
            tenant_id: "team-a".to_string(),
            key: "   ".to_string(),
        }];

        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_empty_worker_urls() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            PolicyConfig::Random,
        );

        // Empty worker URLs are now allowed to match legacy behavior
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_empty_worker_urls_with_service_discovery() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            PolicyConfig::Random,
        );

        // Enable service discovery
        config.discovery = Some(
            KubernetesDiscoveryConfig {
                selector: vec![("app".to_string(), "test".to_string())]
                    .into_iter()
                    .collect(),
                ..Default::default()
            }
            .into(),
        );

        // Should pass validation since service discovery is enabled
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_file_discovery() {
        let file = |path: &str, check_interval_secs| {
            Some(DiscoveryConfig::File(FileDiscoveryConfig {
                path: path.into(),
                check_interval_secs,
            }))
        };

        let mut config = regular_mode_config();
        config.discovery = file("/run/smg/workers.json", 30);
        assert!(ConfigValidator::validate(&config).is_ok());

        for (discovery, field) in [
            (file("", 30), "discovery.path"),
            (
                file("/run/smg/workers.json", 0),
                "discovery.check_interval_secs",
            ),
        ] {
            config.discovery = discovery;
            assert!(matches!(
                ConfigValidator::validate(&config),
                Err(ConfigError::InvalidValue { field: ref actual, .. }) if actual == field
            ));
        }

        let mut openai = RouterConfig {
            mode: RoutingMode::OpenAI {
                worker_urls: vec![],
            },
            ..Default::default()
        };
        openai.discovery = file("/run/smg/workers.json", 30);
        assert!(matches!(
            ConfigValidator::validate(&openai),
            Err(ConfigError::ValidationFailed { ref reason })
                if reason.contains("does not support service discovery")
        ));
    }

    #[test]
    fn test_validate_invalid_kv_annotation_names() {
        for (field, kv_connector_annotation, kv_engine_id_annotation) in [
            (
                "discovery.kv_connector_annotation",
                " ",
                "smg.ai/kv-engine-id",
            ),
            (
                "discovery.kv_engine_id_annotation",
                "smg.ai/kv-connector",
                " smg.ai/kv-engine-id",
            ),
        ] {
            let mut config = regular_mode_config();
            config.discovery = Some(
                KubernetesDiscoveryConfig {
                    selector: [("app".to_string(), "worker".to_string())].into(),
                    kv_connector_annotation: kv_connector_annotation.to_string(),
                    kv_engine_id_annotation: kv_engine_id_annotation.to_string(),
                    ..Default::default()
                }
                .into(),
            );

            assert!(matches!(
                ConfigValidator::validate(&config),
                Err(ConfigError::InvalidValue { field: ref actual, .. }) if actual == field
            ));
        }
    }

    #[test]
    fn test_validate_invalid_urls() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["invalid-url".to_string()],
            },
            PolicyConfig::Random,
        );

        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_cache_aware_thresholds() {
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![
                    "http://worker1:8000".to_string(),
                    "http://worker2:8000".to_string(),
                ],
            },
            PolicyConfig::CacheAware {
                cache_threshold: 1.5, // Invalid: > 1.0
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
                block_size: 16,
                balance_token_usage_threshold: 1.0,
                overload_token_usage_threshold: 1.0,
                overlap_decay: 0.0,
                selection_temperature: 0.0,
                cache_index: Default::default(),
                cache_ttl_secs: 180,
                cache_boundaries: Vec::new(),
                selection_policy: None,
                selection_accounting_ttl_ms: 0,
            },
        );

        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_cache_aware_pressure_knobs() {
        let make = |overlap_decay: f32, selection_temperature: f32| {
            RouterConfig::new(
                RoutingMode::Regular {
                    worker_urls: vec![
                        "http://worker1:8000".to_string(),
                        "http://worker2:8000".to_string(),
                    ],
                },
                PolicyConfig::CacheAware {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    eviction_interval_secs: 60,
                    max_tree_size: 1000,
                    block_size: 16,
                    balance_token_usage_threshold: 1.0,
                    overload_token_usage_threshold: 1.0,
                    overlap_decay,
                    selection_temperature,
                    cache_index: Default::default(),
                    cache_ttl_secs: 180,
                    cache_boundaries: Vec::new(),
                    selection_policy: None,
                    selection_accounting_ttl_ms: 0,
                },
            )
        };

        // Off (defaults) and positive values are valid.
        assert!(ConfigValidator::validate(&make(0.0, 0.0)).is_ok());
        assert!(ConfigValidator::validate(&make(4.0, 0.7)).is_ok());
        // Negative or non-finite values are rejected for both knobs.
        assert!(ConfigValidator::validate(&make(-0.1, 0.0)).is_err());
        assert!(ConfigValidator::validate(&make(0.0, -1.0)).is_err());
        assert!(ConfigValidator::validate(&make(f32::NAN, 0.0)).is_err());
        assert!(ConfigValidator::validate(&make(0.0, f32::INFINITY)).is_err());
    }

    /// NaN passes every `<` and `<=` comparison and an infinity passes the
    /// ones written as a lower bound, so the thresholds a policy compares
    /// that way are checked for finiteness first: a spill gate, token-usage
    /// threshold or load factor that is NaN never fires, which reads as a
    /// quiet routing change instead of the typo it is.
    #[test]
    fn non_finite_policy_thresholds_are_rejected() {
        fn refused_field(config: &RouterConfig) -> String {
            match ConfigValidator::validate(config) {
                Err(ConfigError::InvalidValue { field, .. }) => field,
                other => panic!("accepted, or refused for another reason: {other:?}"),
            }
        }
        let two_workers = || RoutingMode::Regular {
            worker_urls: vec![
                "http://worker1:8000".to_string(),
                "http://worker2:8000".to_string(),
            ],
        };
        let cache_aware = |balance_rel_threshold: f32,
                           balance_token_usage_threshold: f32,
                           overload_token_usage_threshold: f32| {
            RouterConfig::new(
                two_workers(),
                PolicyConfig::CacheAware {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 32,
                    balance_rel_threshold,
                    eviction_interval_secs: 60,
                    max_tree_size: 1000,
                    block_size: 16,
                    balance_token_usage_threshold,
                    overload_token_usage_threshold,
                    overlap_decay: 0.0,
                    selection_temperature: 0.0,
                    cache_index: Default::default(),
                    cache_ttl_secs: 180,
                    cache_boundaries: Vec::new(),
                    selection_policy: None,
                    selection_accounting_ttl_ms: 0,
                },
            )
        };
        let bucket = |balance_rel_threshold: f32| {
            RouterConfig::new(
                two_workers(),
                PolicyConfig::Bucket {
                    balance_abs_threshold: 32,
                    balance_rel_threshold,
                    bucket_adjust_interval_secs: 60,
                },
            )
        };
        let prefix_hash = |load_factor: f64| {
            RouterConfig::new(
                two_workers(),
                PolicyConfig::PrefixHash {
                    prefix_token_count: 256,
                    load_factor,
                    balance_abs_threshold: 10,
                    cache_boundaries: Vec::new(),
                },
            )
        };
        assert!(ConfigValidator::validate(&cache_aware(1.5, 1.0, 1.0)).is_ok());
        assert!(ConfigValidator::validate(&bucket(1.5)).is_ok());
        assert!(ConfigValidator::validate(&prefix_hash(1.25)).is_ok());
        for bad in [f32::NAN, f32::INFINITY] {
            assert_eq!(
                refused_field(&cache_aware(bad, 1.0, 1.0)),
                "balance_rel_threshold",
                "{bad}"
            );
            assert_eq!(
                refused_field(&cache_aware(1.5, bad, 1.0)),
                "balance_token_usage_threshold",
                "{bad}"
            );
            assert_eq!(
                refused_field(&cache_aware(1.5, 1.0, bad)),
                "overload_token_usage_threshold",
                "{bad}"
            );
            assert_eq!(
                refused_field(&bucket(bad)),
                "balance_rel_threshold",
                "{bad}"
            );
            assert_eq!(
                refused_field(&prefix_hash(f64::from(bad))),
                "load_factor",
                "{bad}"
            );
        }
    }

    /// The retry backoff multiplier is a lower bound too.
    #[test]
    fn a_non_finite_retry_backoff_multiplier_is_rejected() {
        let with_multiplier = |backoff_multiplier: f32| {
            let mut config = regular_mode_config();
            config.retry = RetryConfig {
                backoff_multiplier,
                ..Default::default()
            };
            config
        };
        assert!(ConfigValidator::validate(&with_multiplier(1.5)).is_ok());
        for bad in [f32::NAN, f32::INFINITY, 0.5] {
            assert!(
                matches!(
                    ConfigValidator::validate(&with_multiplier(bad)),
                    Err(ConfigError::InvalidValue { ref field, .. })
                        if field == "retry.backoff_multiplier"
                ),
                "{bad}"
            );
        }
    }

    /// The metrics port is a stable, operator-configured port like the other
    /// listeners' ports: 0 (an OS-assigned ephemeral port) is refused naming
    /// the field, as the flag help says.
    #[test]
    fn a_metrics_port_of_zero_is_rejected() {
        let mut config = regular_mode_config();
        config.metrics = Some(MetricsConfig {
            port: 0,
            host: "127.0.0.1".to_string(),
        });
        match ConfigValidator::validate(&config) {
            Err(ConfigError::InvalidValue {
                field,
                value,
                reason,
            }) => {
                assert_eq!(field, "metrics.port");
                assert_eq!(value, "0");
                assert!(reason.contains("ephemeral"), "{reason}");
            }
            other => panic!("a metrics port of 0 was accepted: {other:?}"),
        }
        config.metrics = Some(MetricsConfig {
            port: 29000,
            host: "127.0.0.1".to_string(),
        });
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_cache_index_fields() {
        let make = |cache_index: CacheIndexKind, cache_ttl_secs: u64, boundaries: Vec<usize>| {
            RouterConfig::new(
                RoutingMode::Regular {
                    worker_urls: vec!["http://worker1:8000".to_string()],
                },
                PolicyConfig::CacheAware {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    eviction_interval_secs: 60,
                    max_tree_size: 1000,
                    block_size: 16,
                    balance_token_usage_threshold: 1.0,
                    overload_token_usage_threshold: 1.0,
                    overlap_decay: 0.0,
                    selection_temperature: 0.0,
                    cache_index,
                    cache_ttl_secs,
                    cache_boundaries: boundaries,
                    selection_policy: None,
                    selection_accounting_ttl_ms: 0,
                },
            )
        };

        assert!(ConfigValidator::validate(&make(CacheIndexKind::Tree, 180, vec![])).is_ok());
        assert!(ConfigValidator::validate(&make(CacheIndexKind::Hash, 180, vec![16, 64])).is_ok());
        // TTL must be positive.
        assert!(ConfigValidator::validate(&make(CacheIndexKind::Tree, 0, vec![])).is_err());
        // Hash mode without boundaries has nothing to key on.
        assert!(ConfigValidator::validate(&make(CacheIndexKind::Hash, 180, vec![])).is_err());
        // Boundaries must be strictly ascending and non-zero.
        assert!(ConfigValidator::validate(&make(CacheIndexKind::Hash, 180, vec![64, 16])).is_err());
        assert!(ConfigValidator::validate(&make(CacheIndexKind::Hash, 180, vec![16, 16])).is_err());
        assert!(ConfigValidator::validate(&make(CacheIndexKind::Hash, 180, vec![0, 16])).is_err());
    }

    #[test]
    fn test_validate_shared_cache_boundaries_field() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker1:8000".to_string()],
            },
            PolicyConfig::Random,
        );
        config.cache_boundaries = vec![16, 64];
        assert!(ConfigValidator::validate(&config).is_ok());

        config.cache_boundaries = vec![64, 16];
        assert!(ConfigValidator::validate(&config).is_err());
    }

    #[test]
    fn test_validate_cache_aware_single_worker() {
        // Cache-aware with single worker should be allowed (even if not optimal)
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker1:8000".to_string()],
            },
            PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
                block_size: 16,
                balance_token_usage_threshold: 1.0,
                overload_token_usage_threshold: 1.0,
                overlap_decay: 0.0,
                selection_temperature: 0.0,
                cache_index: Default::default(),
                cache_ttl_secs: 180,
                cache_boundaries: Vec::new(),
                selection_policy: None,
                selection_accounting_ttl_ms: 0,
            },
        );

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_pd_mode() {
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), Some(8081))],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::Random,
        );

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    fn pd_mode_config() -> RouterConfig {
        RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::Random,
        )
    }

    #[test]
    fn prefill_queue_options_require_admission() {
        let mut config = pd_mode_config();
        config.prefill_queue_size = Some(1);

        assert!(matches!(
            ConfigValidator::validate(&config),
            Err(ConfigError::IncompatibleConfig { .. })
        ));
    }

    #[test]
    fn prefill_admission_rejects_priority_scheduler() {
        let mut config = pd_mode_config();
        config.prefill_max_inflight_requests_per_worker = 1;
        config.priority_scheduler_enabled = true;

        assert!(matches!(
            ConfigValidator::validate(&config),
            Err(ConfigError::IncompatibleConfig { .. })
        ));
    }

    #[test]
    fn enabled_prefill_queue_requires_nonzero_timeout() {
        let mut config = pd_mode_config();
        config.prefill_max_inflight_requests_per_worker = 1;
        config.prefill_queue_size = Some(1);
        config.prefill_queue_timeout_secs = Some(0);

        assert!(matches!(
            ConfigValidator::validate(&config),
            Err(ConfigError::InvalidValue { ref field, .. })
                if field == "prefill_queue_timeout_secs"
        ));
    }

    #[test]
    fn zero_size_prefill_queue_does_not_require_timeout() {
        let mut config = pd_mode_config();
        config.prefill_max_inflight_requests_per_worker = 1;
        config.prefill_queue_size = Some(0);
        config.prefill_queue_timeout_secs = Some(0);

        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_roundrobin_with_pd_mode() {
        // RoundRobin with PD mode is now supported
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::RoundRobin,
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_cache_aware_with_pd_mode() {
        // CacheAware with PD mode is now supported
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
                block_size: 16,
                balance_token_usage_threshold: 1.0,
                overload_token_usage_threshold: 1.0,
                overlap_decay: 0.0,
                selection_temperature: 0.0,
                cache_index: Default::default(),
                cache_ttl_secs: 180,
                cache_boundaries: Vec::new(),
                selection_policy: None,
                selection_accounting_ttl_ms: 0,
            },
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_power_of_two_with_regular_mode() {
        // PowerOfTwo with Regular mode is now supported
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![
                    "http://worker1:8000".to_string(),
                    "http://worker2:8000".to_string(),
                ],
            },
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
                load_metric: Default::default(),
            },
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_pd_mode_with_separate_policies() {
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![
                    ("http://prefill1:8000".to_string(), None),
                    ("http://prefill2:8000".to_string(), None),
                ],
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: Some(PolicyConfig::CacheAware {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    eviction_interval_secs: 60,
                    max_tree_size: 1000,
                    block_size: 16,
                    balance_token_usage_threshold: 1.0,
                    overload_token_usage_threshold: 1.0,
                    overlap_decay: 0.0,
                    selection_temperature: 0.0,
                    cache_index: Default::default(),
                    cache_ttl_secs: 180,
                    cache_boundaries: Vec::new(),
                    selection_policy: None,
                    selection_accounting_ttl_ms: 0,
                }),
                decode_policy: Some(PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 60,
                    load_metric: Default::default(),
                }),
            },
            PolicyConfig::Random, // Main policy as fallback
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_pd_mode_power_of_two_insufficient_workers() {
        let mut config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill1:8000".to_string(), None)], // Only 1 prefill
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: Some(PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 60,
                    load_metric: Default::default(),
                }), // Requires 2+ workers
                decode_policy: None,
            },
            PolicyConfig::Random,
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("prefill requires at least 2"));
        }

        config.dp_aware = true;
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_pd_mode_bucket_policy_restrictions() {
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![
                    ("http://prefill1:8000".to_string(), None),
                    ("http://prefill2:8000".to_string(), None),
                ],
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: Some(PolicyConfig::Bucket {
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    bucket_adjust_interval_secs: 5,
                }),
                decode_policy: Some(PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 60,
                    load_metric: Default::default(),
                }),
            },
            PolicyConfig::Random, // Main policy as fallback
        );

        let result = ConfigValidator::validate(&config);
        assert!(
            result.is_ok(),
            "Prefill policy should be allowed to be bucket"
        );

        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![
                    ("http://prefill1:8000".to_string(), None),
                    ("http://prefill2:8000".to_string(), None),
                ],
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: Some(PolicyConfig::Bucket {
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    bucket_adjust_interval_secs: 5,
                }),
                decode_policy: Some(PolicyConfig::Bucket {
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    bucket_adjust_interval_secs: 5,
                }),
            },
            PolicyConfig::Random, // Main policy as fallback
        );

        let result = ConfigValidator::validate(&config);
        assert!(
            result.is_err(),
            "Decode policy should not be allowed to be bucket"
        );
    }

    #[test]
    fn test_validate_epd_mode_encode_policy_restrictions() {
        let valid = RouterConfig::new(
            RoutingMode::EncodePrefillDecode {
                encode_urls: vec![("http://encode:8000".to_string(), None)],
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                encode_policy: Some(PolicyConfig::ConsistentHashing),
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::Random,
        );
        assert!(ConfigValidator::validate(&valid).is_ok());

        let invalid_cache_aware = RouterConfig::new(
            RoutingMode::EncodePrefillDecode {
                encode_urls: vec![("http://encode:8000".to_string(), None)],
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                encode_policy: Some(PolicyConfig::CacheAware {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
                    eviction_interval_secs: 60,
                    max_tree_size: 1000,
                    block_size: 16,
                    balance_token_usage_threshold: 1.0,
                    overload_token_usage_threshold: 1.0,
                    overlap_decay: 0.0,
                    selection_temperature: 0.0,
                    cache_index: Default::default(),
                    cache_ttl_secs: 180,
                    cache_boundaries: Vec::new(),
                    selection_policy: None,
                    selection_accounting_ttl_ms: 0,
                }),
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::Random,
        );
        assert!(ConfigValidator::validate(&invalid_cache_aware).is_err());

        let invalid_least_load = RouterConfig::new(
            RoutingMode::EncodePrefillDecode {
                encode_urls: vec![("http://encode:8000".to_string(), None)],
                prefill_urls: vec![("http://prefill:8000".to_string(), None)],
                decode_urls: vec!["http://decode:8000".to_string()],
                encode_policy: Some(PolicyConfig::LeastLoad {
                    load_check_interval_secs: 5,
                    kv_pressure_weight: 0.15,
                    mean_prefill_tokens: 1024,
                    default_throughput: 2000.0,
                    max_waiting_requests: 0,
                }),
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::Random,
        );
        assert!(ConfigValidator::validate(&invalid_least_load).is_err());
    }

    #[test]
    fn test_validate_empty_urls_allowed_without_service_discovery() {
        // Test that empty URLs are now allowed in PD mode
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: None,
            },
            PolicyConfig::Random,
        );

        // Should pass validation even with empty URLs
        assert!(ConfigValidator::validate(&config).is_ok());

        // Test that empty URLs are allowed in Regular mode
        let config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            PolicyConfig::Random,
        );

        // Should pass validation even with empty URLs
        assert!(ConfigValidator::validate(&config).is_ok());

        // Test that empty URLs are allowed in OpenAI mode
        let config = RouterConfig::new(
            RoutingMode::OpenAI {
                worker_urls: vec![],
            },
            PolicyConfig::Random,
        );

        // Should pass validation even with empty URLs
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn test_validate_grpc_with_model_path() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["grpc://worker:50051".to_string()],
            },
            PolicyConfig::Random,
        );

        config.connection_mode = ConnectionMode::Grpc;
        config.model_path = Some("meta-llama/Llama-3-8B".to_string());

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_grpcs_worker_url() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["grpcs://worker:50051".to_string()],
            },
            PolicyConfig::Random,
        );

        config.connection_mode = ConnectionMode::Grpc;
        config.model_path = Some("meta-llama/Llama-3-8B".to_string());

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_grpc_with_tokenizer_path() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["grpc://worker:50051".to_string()],
            },
            PolicyConfig::Random,
        );

        config.connection_mode = ConnectionMode::Grpc;
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = dir.path().join("tokenizer.json");
        std::fs::write(&tokenizer, "{}").unwrap();
        config.tokenizer_path = Some(tokenizer.to_string_lossy().into_owned());

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    /// A directory with the one file the tokenizer loader looks for first.
    fn tokenizer_dir(dir: &tempfile::TempDir) -> String {
        let tok = dir.path().join("tok");
        std::fs::create_dir(&tok).unwrap();
        std::fs::write(tok.join("tokenizer.json"), "{}").unwrap();
        tok.to_str().unwrap().to_string()
    }

    /// The `(value, reason)` of the `expected_field` error a configuration
    /// is refused with.
    fn refused_source(config: &RouterConfig, expected_field: &str) -> (String, String) {
        match ConfigValidator::validate(config) {
            Err(ConfigError::InvalidValue {
                field,
                value,
                reason,
            }) if field == expected_field => (value, reason),
            other => panic!("expected a {expected_field} error, got {other:?}"),
        }
    }

    #[test]
    fn a_tokenizer_or_model_path_that_cannot_be_read_is_a_configuration_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing");
        let empty_dir = dir.path().join("empty-dir");
        std::fs::create_dir(&empty_dir).unwrap();
        let empty_file = dir.path().join("empty.json");
        std::fs::write(&empty_file, "").unwrap();
        let cases = [
            (missing.to_str().unwrap(), "does not exist"),
            (empty_dir.to_str().unwrap(), "contains no tokenizer file"),
            (empty_file.to_str().unwrap(), "is empty"),
        ];
        for (path, expected) in cases {
            let as_tokenizer = RouterConfig {
                tokenizer_path: Some(path.to_string()),
                ..Default::default()
            };
            let as_model = RouterConfig {
                model_path: Some(path.to_string()),
                ..Default::default()
            };
            for (field, config) in [("tokenizer_path", as_tokenizer), ("model_path", as_model)] {
                let (value, reason) = refused_source(&config, field);
                assert_eq!(value, path);
                assert!(reason.contains(expected), "{field} {path}: {reason}");
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_tokenizer_path_is_a_configuration_error() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let unreadable = dir.path().join("unreadable.json");
        std::fs::write(&unreadable, "{}").unwrap();
        std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o000)).unwrap();
        // A privileged process reads regardless of the mode: nothing to test.
        let privileged = std::fs::File::open(&unreadable).is_ok();
        let config = RouterConfig {
            tokenizer_path: Some(unreadable.to_str().unwrap().to_string()),
            ..Default::default()
        };
        let verdict = ConfigValidator::validate(&config);
        std::fs::set_permissions(&unreadable, std::fs::Permissions::from_mode(0o600)).unwrap();
        if privileged {
            return;
        }
        match verdict {
            Err(ConfigError::InvalidValue { field, reason, .. }) if field == "tokenizer_path" => {
                assert!(reason.contains("cannot be read"), "{reason}");
            }
            other => panic!("expected a tokenizer_path error, got {other:?}"),
        }
    }

    /// Only the source the gateway loads is checked: a `model_path` beside an
    /// explicit tokenizer path is never read as a tokenizer source.
    #[test]
    fn a_model_path_beside_an_explicit_tokenizer_path_is_not_checked() {
        let dir = tempfile::tempdir().unwrap();
        let tok = tokenizer_dir(&dir);
        let config = RouterConfig {
            tokenizer_path: Some(tok),
            model_path: Some(format!("{}/missing", dir.path().to_str().unwrap())),
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    /// A source that does not exist on this host is left to the worker
    /// fallback wherever a gRPC worker can still register: gRPC start-up
    /// URLs, IGW mode, a discovery provider; it is refused where none can
    /// (http URLs only, no IGW, no discovery), and one that exists and cannot
    /// serve is refused everywhere.
    #[test]
    fn a_missing_source_is_left_to_the_worker_fallback_where_a_grpc_worker_may_register() {
        let dir = tempfile::tempdir().unwrap();
        let missing = format!("{}/models/on-the-worker", dir.path().to_str().unwrap());
        let regular = |urls: &[&str]| RoutingMode::Regular {
            worker_urls: urls.iter().map(|u| u.to_string()).collect(),
        };
        let with =
            |mode: RoutingMode, igw: bool, discovery: Option<DiscoveryConfig>| RouterConfig {
                mode,
                enable_igw: igw,
                discovery,
                model_path: Some(missing.clone()),
                ..Default::default()
            };
        let discovery = DiscoveryConfig::File(FileDiscoveryConfig {
            path: dir.path().join("workers.json"),
            check_interval_secs: 5,
        });
        for (name, config) in [
            ("igw", with(regular(&[]), true, None)),
            ("discovery", with(regular(&[]), false, Some(discovery))),
            (
                "a grpcs url",
                with(regular(&["grpcs://b:50051"]), false, None),
            ),
            (
                "a grpc url beside an http one",
                with(regular(&["http://a:8000", "grpc://b:50051"]), false, None),
            ),
        ] {
            assert!(ConfigValidator::validate(&config).is_ok(), "{name}");
        }
        for (name, config) in [
            (
                "explicit http urls only",
                with(regular(&["http://a:8000", "https://b:8443"]), false, None),
            ),
            (
                "no urls, no igw, no discovery",
                with(regular(&[]), false, None),
            ),
        ] {
            let (value, reason) = refused_source(&config, "model_path");
            assert_eq!(value, missing, "{name}");
            assert!(reason.contains("does not exist"), "{name}: {reason}");
        }
    }

    #[test]
    fn a_missing_source_is_left_to_the_worker_fallback_in_grpc_mode() {
        let dir = tempfile::tempdir().unwrap();
        let missing = format!("{}/models/on-the-worker", dir.path().to_str().unwrap());
        let empty_dir = dir.path().join("empty-dir");
        std::fs::create_dir(&empty_dir).unwrap();
        let grpc = |model_path: &str| {
            let mut config = RouterConfig::new(
                RoutingMode::Regular {
                    worker_urls: vec!["grpc://worker:50051".to_string()],
                },
                PolicyConfig::Random,
            );
            config.connection_mode = ConnectionMode::Grpc;
            config.model_path = Some(model_path.to_string());
            config
        };
        assert!(ConfigValidator::validate(&grpc(&missing)).is_ok());
        let (value, reason) = refused_source(&grpc(empty_dir.to_str().unwrap()), "model_path");
        assert_eq!(value, empty_dir.to_str().unwrap());
        assert!(reason.contains("contains no tokenizer file"), "{reason}");
        let http = RouterConfig {
            model_path: Some(missing.clone()),
            ..Default::default()
        };
        let (value, reason) = refused_source(&http, "model_path");
        assert_eq!(value, missing);
        assert!(reason.contains("does not exist"), "{reason}");
    }

    /// The directory rule is the loader's: `vocab.json` counts with
    /// `merges.txt` beside it, not alone.
    #[test]
    fn a_directory_with_vocab_json_alone_is_not_a_tokenizer_source() {
        let dir = tempfile::tempdir().unwrap();
        let bpe = dir.path().join("bpe");
        std::fs::create_dir(&bpe).unwrap();
        std::fs::write(bpe.join("vocab.json"), "{}").unwrap();
        let config = RouterConfig {
            tokenizer_path: Some(bpe.to_str().unwrap().to_string()),
            ..Default::default()
        };
        let (_, reason) = refused_source(&config, "tokenizer_path");
        assert!(reason.contains("vocab.json with merges.txt"), "{reason}");
        std::fs::write(bpe.join("merges.txt"), "#version: 0.2\n").unwrap();
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn a_hub_id_or_a_present_local_tokenizer_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let tok = tokenizer_dir(&dir);
        let sources = [
            tok.clone(),
            format!("{tok}/tokenizer.json"),
            "some-org/some-model".to_string(),
        ];
        for source in sources {
            let config = RouterConfig {
                tokenizer_path: Some(source.clone()),
                ..Default::default()
            };
            assert!(ConfigValidator::validate(&config).is_ok(), "{source}");
            let config = RouterConfig {
                model_path: Some(source.clone()),
                ..Default::default()
            };
            assert!(ConfigValidator::validate(&config).is_ok(), "{source}");
        }
        // With autoload disabled the paths are never read, so not checked.
        let config = RouterConfig {
            tokenizer_path: Some(format!("{}/missing", dir.path().to_str().unwrap())),
            disable_tokenizer_autoload: true,
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn a_chat_template_that_cannot_be_used_is_a_configuration_error() {
        let dir = tempfile::tempdir().unwrap();
        let tok = tokenizer_dir(&dir);
        let missing = dir.path().join("missing.jinja");
        let blank = dir.path().join("blank.jinja");
        std::fs::write(&blank, " \n").unwrap();
        let template = dir.path().join("template.jinja");
        std::fs::write(&template, "{{ messages }}").unwrap();
        let with_template = |path: &str| RouterConfig {
            tokenizer_path: Some(tok.clone()),
            chat_template: Some(path.to_string()),
            ..Default::default()
        };
        let cases = [
            (missing.to_str().unwrap(), "does not exist"),
            (blank.to_str().unwrap(), "is empty"),
            (dir.path().to_str().unwrap(), "is a directory"),
        ];
        for (path, expected) in cases {
            let (value, reason) = refused_source(&with_template(path), "chat_template");
            assert_eq!(value, path);
            assert!(reason.contains(expected), "{path}: {reason}");
        }
        assert!(ConfigValidator::validate(&with_template(template.to_str().unwrap())).is_ok());
    }

    #[test]
    fn test_reject_duplicate_storage_context_keys() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker1:8000".to_string()],
            },
            PolicyConfig::Random,
        );

        config.storage_context_headers = std::collections::HashMap::from([
            ("x-tenant-id".to_string(), "tenant_id".to_string()),
            ("x-org-id".to_string(), "tenant_id".to_string()),
        ]);

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_empty_storage_context_key() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker1:8000".to_string()],
            },
            PolicyConfig::Random,
        );

        config.storage_context_headers =
            std::collections::HashMap::from([("x-tenant-id".to_string(), " ".to_string())]);

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_empty_storage_context_header_name() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker1:8000".to_string()],
            },
            PolicyConfig::Random,
        );

        config.storage_context_headers =
            std::collections::HashMap::from([(" ".to_string(), "tenant_id".to_string())]);

        let result = ConfigValidator::validate(&config);
        assert!(result.is_err());
    }

    /// Two listeners on one port are refused naming both flags; distinct
    /// ports pass.
    #[test]
    fn equal_listener_ports_are_rejected() {
        let listeners = |port: u16, metrics_port: u16, probe_port: Option<u16>| RouterConfig {
            port,
            metrics: Some(MetricsConfig {
                port: metrics_port,
                host: "127.0.0.1".to_string(),
            }),
            health_check_port: probe_port,
            ..Default::default()
        };
        let refused = |config: &RouterConfig| match ConfigValidator::validate(config) {
            Err(ConfigError::InvalidValue {
                field,
                value,
                reason,
            }) => (field, value, reason),
            other => panic!("expected an InvalidValue, got {other:?}"),
        };

        let (field, value, reason) = refused(&listeners(30001, 30001, None));
        assert_eq!((field.as_str(), value.as_str()), ("metrics.port", "30001"));
        assert!(
            reason.contains("--prometheus-port") && reason.contains("--port"),
            "{reason}"
        );

        let (field, _, reason) = refused(&listeners(30001, 30002, Some(30001)));
        assert_eq!(field, "health_check_port");
        assert!(
            reason.contains("--health-check-port") && reason.contains("--port"),
            "{reason}"
        );

        let (field, _, reason) = refused(&listeners(30001, 30002, Some(30002)));
        assert_eq!(field, "health_check_port");
        assert!(
            reason.contains("--health-check-port") && reason.contains("--prometheus-port"),
            "{reason}"
        );

        assert!(ConfigValidator::validate(&listeners(30001, 30002, Some(30003))).is_ok());
    }

    /// The mesh listener cannot share a port with the HTTP, metrics or probe
    /// listener: refused under the mesh port's field, naming both flags. The
    /// check runs in `server::startup`, so it holds for every way the server
    /// configuration is built.
    #[test]
    fn mesh_port_equal_to_another_listener_is_rejected() {
        for (flag, taken) in [
            ("--port", 30001),
            ("--prometheus-port", 30002),
            ("--health-check-port", 30003),
        ] {
            match validate_mesh_listener_port(taken, 30001, Some(30002), Some(30003)) {
                Err(ConfigError::InvalidValue {
                    field,
                    value,
                    reason,
                }) => {
                    assert_eq!(field, "mesh_port");
                    assert_eq!(value, taken.to_string());
                    assert!(
                        reason.contains("--mesh-port") && reason.contains(flag),
                        "{reason}"
                    );
                }
                other => panic!("--mesh-port equal to {flag} must be rejected, got {other:?}"),
            }
        }
        assert!(validate_mesh_listener_port(30004, 30001, Some(30002), Some(30003)).is_ok());
        assert!(validate_mesh_listener_port(30002, 30001, None, None).is_ok());
    }

    #[test]
    fn test_reject_health_check_port_zero() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker1:8000".to_string()],
            },
            PolicyConfig::Random,
        );

        // 0 = OS-ephemeral; breaks the stable-probe-port contract.
        config.health_check_port = Some(0);
        assert!(matches!(
            ConfigValidator::validate(&config),
            Err(ConfigError::InvalidValue { ref field, .. }) if field == "health_check_port"
        ));

        // A real port and the unset (None) default both validate.
        config.health_check_port = Some(8081);
        assert!(ConfigValidator::validate(&config).is_ok());
        config.health_check_port = None;
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    /// Every health-check knob at 0 is refused naming the field; the smallest
    /// values pass; a zero interval is refused with probing disabled too.
    #[test]
    fn zero_health_check_knobs_are_rejected() {
        let zeroed = [
            (
                "health_check.check_interval_secs",
                HealthCheckConfig {
                    check_interval_secs: 0,
                    ..Default::default()
                },
            ),
            (
                "health_check.timeout_secs",
                HealthCheckConfig {
                    timeout_secs: 0,
                    ..Default::default()
                },
            ),
            (
                "health_check.failure_threshold",
                HealthCheckConfig {
                    failure_threshold: 0,
                    ..Default::default()
                },
            ),
            (
                "health_check.success_threshold",
                HealthCheckConfig {
                    success_threshold: 0,
                    ..Default::default()
                },
            ),
        ];
        for (name, health_check) in zeroed {
            let config = RouterConfig {
                health_check,
                ..Default::default()
            };
            match ConfigValidator::validate(&config) {
                Err(ConfigError::InvalidValue {
                    field,
                    value,
                    reason,
                }) => {
                    assert_eq!(field, name);
                    assert_eq!(value, "0");
                    assert!(reason.starts_with("Must be >= 1"), "{name}: {reason}");
                }
                other => panic!("{name}: expected an InvalidValue, got {other:?}"),
            }
        }

        let smallest = RouterConfig {
            health_check: HealthCheckConfig {
                check_interval_secs: 1,
                timeout_secs: 1,
                failure_threshold: 1,
                success_threshold: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&smallest).is_ok());
        assert!(ConfigValidator::validate(&RouterConfig::default()).is_ok());

        let disabled_but_zero = RouterConfig {
            health_check: HealthCheckConfig {
                check_interval_secs: 0,
                disable_health_check: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(
            matches!(
                ConfigValidator::validate(&disabled_but_zero),
                Err(ConfigError::InvalidValue { ref field, .. }) if field == "health_check.check_interval_secs"
            ),
            "the checker's loop sleeps on the interval with probing disabled too"
        );
    }

    /// The CLI parsers reject these, but TOML/JSON and the Python bindings
    /// reach `RouterConfig` without going through clap.
    #[test]
    fn test_reject_degenerate_worker_overload_thresholds() {
        let mut config = RouterConfig::new(
            RoutingMode::Regular {
                worker_urls: vec!["http://worker1:8000".to_string()],
            },
            PolicyConfig::Random,
        );

        // Unset is the default and always valid.
        assert!(ConfigValidator::validate(&config).is_ok());

        config.worker_overload_waiting_requests = Some(0);
        assert!(matches!(
            ConfigValidator::validate(&config),
            Err(ConfigError::InvalidValue { ref field, .. })
                if field == "worker_overload_waiting_requests"
        ));
        config.worker_overload_waiting_requests = Some(1);
        assert!(ConfigValidator::validate(&config).is_ok());

        for bad in [0.0, -0.1, 1.5, f64::NAN] {
            config.worker_overload_token_usage = Some(bad);
            assert!(
                matches!(
                    ConfigValidator::validate(&config),
                    Err(ConfigError::InvalidValue { ref field, .. })
                        if field == "worker_overload_token_usage"
                ),
                "{bad} must be rejected"
            );
        }
        config.worker_overload_token_usage = Some(1.0);
        assert!(ConfigValidator::validate(&config).is_ok());
    }

    #[test]
    fn rl_config_defaults_to_disabled_and_validates_when_enabled() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .build_unchecked();
        assert!(!config.rl.enabled);
        assert_eq!(config.rl.control_timeout_secs, 600);
        assert_eq!(config.rl.fanout_concurrency, 32);
        assert!(ConfigValidator::validate(&config).is_ok());

        let bad = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .rl(smg_rl::RlConfig {
                enabled: true,
                control_timeout_secs: 0,
                fanout_concurrency: 32,
            })
            .build_unchecked();
        let err = ConfigValidator::validate(&bad).unwrap_err().to_string();
        assert!(err.contains("control_timeout_secs"), "{err}");

        let ok = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .rl(smg_rl::RlConfig {
                enabled: true,
                ..smg_rl::RlConfig::default()
            })
            .build_unchecked();
        assert!(ConfigValidator::validate(&ok).is_ok());
    }

    /// The CORS origin rule holds on every construction path, not only in
    /// the CLI: a lone `*` is the open policy, `*` next to an origin and a
    /// value that is not exactly an origin are refused by name.
    #[test]
    fn cors_allowed_origins_are_exact_origins_or_a_lone_wildcard_on_every_path() {
        let config = |origins: &[&str]| {
            RouterConfig::builder()
                .regular_mode(vec![])
                .round_robin_policy()
                .cors_allowed_origins(origins.iter().map(|origin| (*origin).to_string()).collect())
                .build_unchecked()
        };
        for accepted in [
            &[][..],
            &["*"][..],
            &["https://app.example"][..],
            &["https://app.example", "http://localhost:3000"][..],
        ] {
            assert!(
                ConfigValidator::validate(&config(accepted)).is_ok(),
                "{accepted:?} is accepted"
            );
        }
        for (refused, named) in [
            (&["*", "https://app.example"][..], "*"),
            (&["https://app.example", "*"][..], "*"),
            (&["https://Example.com"][..], "https://Example.com"),
            (&["example.com"][..], "example.com"),
            (&["https://example.com/"][..], "https://example.com/"),
        ] {
            match ConfigValidator::validate(&config(refused)) {
                Err(ConfigError::InvalidValue { field, value, .. }) => {
                    assert_eq!(
                        (field.as_str(), value.as_str()),
                        ("cors_allowed_origins", named)
                    );
                }
                other => panic!("{refused:?}: {other:?}"),
            }
        }
        // The launcher's path: the builder's validating `build()`.
        let built = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .cors_allowed_origins(vec!["*".to_string(), "https://app.example".to_string()])
            .build();
        assert!(
            matches!(built, Err(ConfigError::InvalidValue { ref field, .. }) if field == "cors_allowed_origins"),
            "{built:?}"
        );
    }
}
