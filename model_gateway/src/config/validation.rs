use axum::http::HeaderName;
use sha2::{Digest, Sha256};

use super::*;

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

/// Configuration validator
pub(crate) struct ConfigValidator;
impl ConfigValidator {
    pub(crate) fn validate(config: &RouterConfig) -> ConfigResult<()> {
        Self::validate_mode(&config.mode)?;
        Self::validate_policy(&config.policy)?;
        Self::validate_cache_boundaries(&config.cache_boundaries)?;
        Self::validate_long_prefill_indices(config)?;
        Self::validate_server_settings(config)?;
        Self::validate_storage_context_headers(config)?;
        Self::validate_routing_key_headers(config)?;
        Self::validate_tenant_resolution(config)?;
        Self::validate_tenant_api_keys(config)?;
        Self::validate_model_aliases(config)?;
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

        Self::validate_tokenizer_cache(&config.tokenizer_cache)?;

        Ok(())
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

    fn validate_long_prefill_indices(config: &RouterConfig) -> ConfigResult<()> {
        let indices = &config.long_prefill_indices;
        if indices.is_empty() {
            return Ok(());
        }
        let mut seen = std::collections::HashSet::new();
        for &i in indices {
            if !seen.insert(i) {
                return Err(ConfigError::InvalidValue {
                    field: "long_prefill_indices".to_string(),
                    value: i.to_string(),
                    reason: "must not contain duplicate values".to_string(),
                });
            }
        }
        let prefill_count = match &config.mode {
            RoutingMode::PrefillDecode { prefill_urls, .. }
            | RoutingMode::EncodePrefillDecode { prefill_urls, .. } => prefill_urls.len(),
            _ => 0,
        };
        if let Some(&max) = indices.iter().max() {
            if max >= prefill_count {
                return Err(ConfigError::InvalidValue {
                    field: "long_prefill_indices".to_string(),
                    value: max.to_string(),
                    reason: format!(
                        "out of range for {prefill_count} configured prefill workers"
                    ),
                });
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
            } => {
                Self::validate_cache_aware_shared(
                    cache_threshold,
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
                )?;
            }
            PolicyConfig::CacheAwareLength {
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
                chars_per_token,
                long_prefill_threshold,
                long_pool_max_load,
                short_pool_max_load,
            } => {
                Self::validate_cache_aware_shared(
                    cache_threshold,
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
                )?;

                // ---- Length-specific checks ----
                if *chars_per_token == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "chars_per_token".to_string(),
                        value: chars_per_token.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
                if *long_prefill_threshold == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "long_prefill_threshold".to_string(),
                        value: long_prefill_threshold.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
                if *long_pool_max_load == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "long_pool_max_load".to_string(),
                        value: long_pool_max_load.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
                if *short_pool_max_load == 0 {
                    return Err(ConfigError::InvalidValue {
                        field: "short_pool_max_load".to_string(),
                        value: short_pool_max_load.to_string(),
                        reason: "Must be > 0".to_string(),
                    });
                }
            }
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
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
                if *balance_rel_threshold < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "balance_rel_threshold".to_string(),
                        value: balance_rel_threshold.to_string(),
                        reason: "Must be >= 1.0".to_string(),
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

                if *load_factor < 1.0 {
                    return Err(ConfigError::InvalidValue {
                        field: "load_factor".to_string(),
                        value: load_factor.to_string(),
                        reason: "Must be >= 1.0".to_string(),
                    });
                }

                Self::validate_cache_boundaries(cache_boundaries)?;
            }
        }
        Ok(())
    }

    /// Shared validation for the cache-aware fields common to both
    /// `CacheAware` and `CacheAwareLength` policy variants.
    #[expect(clippy::too_many_arguments, reason = "mirrors the PolicyConfig fields")]
    fn validate_cache_aware_shared(
        cache_threshold: &f32,
        balance_rel_threshold: &f32,
        eviction_interval_secs: &u64,
        max_tree_size: &usize,
        block_size: &usize,
        balance_token_usage_threshold: &f32,
        overload_token_usage_threshold: &f32,
        overlap_decay: &f32,
        selection_temperature: &f32,
        cache_index: &CacheIndexKind,
        cache_ttl_secs: &u64,
        cache_boundaries: &[usize],
    ) -> ConfigResult<()> {
        Self::validate_cache_boundaries(cache_boundaries)?;

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

        if config.max_payload_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "max_payload_size".to_string(),
                value: config.max_payload_size.to_string(),
                reason: "Must be > 0".to_string(),
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

        // The body-limit layer rejects payloads above max_payload_size before
        // the streaming threshold is consulted, so a threshold at or above it
        // could never activate.
        if config.stream_request_bodies_over != 0
            && config.stream_request_bodies_over >= config.max_payload_size as u64
        {
            return Err(ConfigError::InvalidValue {
                field: "stream_request_bodies_over".to_string(),
                value: config.stream_request_bodies_over.to_string(),
                reason: format!("Must be < max_payload_size ({})", config.max_payload_size),
            });
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

    fn validate_discovery(discovery: &DiscoveryConfig, mode: &RoutingMode) -> ConfigResult<()> {
        if !discovery.enabled {
            return Ok(());
        }

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
                reason: "Port must be > 0".to_string(),
            });
        }

        if metrics.host.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "metrics.host".to_string(),
                value: metrics.host.clone(),
                reason: "Host cannot be empty".to_string(),
            });
        }

        Ok(())
    }

    fn validate_trace(trace_config: &TraceConfig) -> ConfigResult<()> {
        if !trace_config.enable_trace {
            return Ok(());
        }

        let endpoint = &trace_config.otlp_traces_endpoint;

        let Some((host, port_str)) = endpoint.rsplit_once(':') else {
            return Err(ConfigError::InvalidValue {
                field: "trace_config.otlp_traces_endpoint".to_string(),
                value: endpoint.clone(),
                reason:
                    "expected format <host>:<port>, e.g., otel-collector:4317 or 127.0.0.1:4317"
                        .to_string(),
            });
        };

        if host.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "trace_config.otlp_traces_endpoint".to_string(),
                value: endpoint.clone(),
                reason: "host part cannot be empty".to_string(),
            });
        }

        // check port: must be 1~65535
        match port_str.parse::<u16>() {
            Ok(p) if p > 0 => (), // valid port
            _ => {
                return Err(ConfigError::InvalidValue {
                    field: "trace_config.otlp_traces_endpoint".to_string(),
                    value: endpoint.clone(),
                    reason: "port must be a number between 1 and 65535".to_string(),
                });
            }
        };

        Ok(())
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
        if retry.backoff_multiplier < 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "retry.backoff_multiplier".to_string(),
                value: retry.backoff_multiplier.to_string(),
                reason: "Must be >= 1.0".to_string(),
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

    fn validate_tokenizer_cache(cache: &TokenizerCacheConfig) -> ConfigResult<()> {
        if cache.enable_l0 && cache.l0_max_entries == 0 {
            return Err(ConfigError::InvalidValue {
                field: "tokenizer_cache.l0_max_entries".to_string(),
                value: cache.l0_max_entries.to_string(),
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
        if config.enable_igw {
            return Ok(());
        }

        Self::validate_mtls(config)?;

        let has_service_discovery = config.discovery.as_ref().is_some_and(|d| d.enabled);

        if let RoutingMode::EncodePrefillDecode { decode_policy, .. } = &config.mode {
            let effective_decode_policy = decode_policy.as_ref().unwrap_or(&config.policy);
            if matches!(effective_decode_policy, PolicyConfig::Bucket { .. }) {
                return Err(ConfigError::IncompatibleConfig {
                    reason: "Decode policy should not be allowed to be bucket".to_string(),
                });
            }
        }

        if !has_service_discovery {
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

                // Check bucket for decode
                if let Some(PolicyConfig::Bucket { .. }) = decode_policy {
                    return Err(ConfigError::IncompatibleConfig {
                        reason: "Decode policy should not be allowed to be bucket".to_string(),
                    });
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

    #[test]
    fn stream_threshold_at_or_above_payload_cap_is_rejected() {
        let below = RouterConfig {
            stream_request_bodies_over: 1024,
            ..Default::default()
        };
        assert!(ConfigValidator::validate(&below).is_ok());

        let at_cap = RouterConfig {
            stream_request_bodies_over: RouterConfig::default().max_payload_size as u64,
            ..Default::default()
        };
        assert!(matches!(
            ConfigValidator::validate(&at_cap),
            Err(ConfigError::InvalidValue { ref field, .. })
                if field == "stream_request_bodies_over"
        ));
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
        config.discovery = Some(DiscoveryConfig {
            enabled: true,
            selector: vec![("app".to_string(), "test".to_string())]
                .into_iter()
                .collect(),
            ..Default::default()
        });

        // Should pass validation since service discovery is enabled
        assert!(ConfigValidator::validate(&config).is_ok());
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

    #[test]
    fn test_validate_cache_aware_length_rejects_nan_thresholds() {
        let make = |balance_token_usage_threshold: f32,
                    overload_token_usage_threshold: f32| {
            RouterConfig::new(
                RoutingMode::Regular {
                    worker_urls: vec![
                        "http://worker1:8000".to_string(),
                        "http://worker2:8000".to_string(),
                    ],
                },
                PolicyConfig::CacheAwareLength {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
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
                    chars_per_token: 4,
                    long_prefill_threshold: 100_000,
                    long_pool_max_load: 4,
                    short_pool_max_load: 32,
                },
            )
        };

        // Valid defaults pass.
        assert!(ConfigValidator::validate(&make(1.0, 1.0)).is_ok());
        // NaN rejected for both fields.
        assert!(ConfigValidator::validate(&make(f32::NAN, 1.0)).is_err());
        assert!(ConfigValidator::validate(&make(1.0, f32::NAN)).is_err());
        // Infinity rejected for both fields.
        assert!(ConfigValidator::validate(&make(f32::INFINITY, 1.0)).is_err());
        assert!(ConfigValidator::validate(&make(1.0, f32::INFINITY)).is_err());
        // Zero rejected.
        assert!(ConfigValidator::validate(&make(0.0, 1.0)).is_err());
        assert!(ConfigValidator::validate(&make(1.0, 0.0)).is_err());
    }

    #[test]
    fn test_validate_cache_aware_length_rejects_non_finite_balance_rel_threshold() {
        let make = |balance_rel_threshold: f32| {
            RouterConfig::new(
                RoutingMode::Regular {
                    worker_urls: vec![
                        "http://worker1:8000".to_string(),
                        "http://worker2:8000".to_string(),
                    ],
                },
                PolicyConfig::CacheAwareLength {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 32,
                    balance_rel_threshold,
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
                    chars_per_token: 4,
                    long_prefill_threshold: 100_000,
                    long_pool_max_load: 4,
                    short_pool_max_load: 32,
                },
            )
        };

        // Valid value passes.
        assert!(ConfigValidator::validate(&make(1.1)).is_ok());
        // NaN passes the < 1.0 check (NaN < 1.0 is false) but must be rejected.
        assert!(ConfigValidator::validate(&make(f32::NAN)).is_err());
        // Infinity must be rejected.
        assert!(ConfigValidator::validate(&make(f32::INFINITY)).is_err());
        // Below 1.0 rejected.
        assert!(ConfigValidator::validate(&make(0.9)).is_err());
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
                }),
                decode_policy: Some(PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 60,
                }),
            },
            PolicyConfig::Random, // Main policy as fallback
        );

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_pd_mode_power_of_two_insufficient_workers() {
        let config = RouterConfig::new(
            RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill1:8000".to_string(), None)], // Only 1 prefill
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: Some(PolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 60,
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
        config.tokenizer_path = Some("/path/to/tokenizer.json".to_string());

        let result = ConfigValidator::validate(&config);
        assert!(result.is_ok());
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
}
