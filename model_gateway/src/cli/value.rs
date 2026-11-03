//! The value parsers of the command line: each turns one flag's text into
//! its typed value, or a refusal that names the value and the shape expected
//! (the API-key parsers name only the shape: their value is the credential).

use openai_protocol::worker::{MmProcessingMode, TransportMode};
use smg::{
    config::{
        AccessLogField, ConfigError, ConfigResult, PowerOfTwoLoadMetric, TenantApiKeyEntry,
        TurboJpegPolicy,
    },
    service_discovery::ModelIdSource,
};
use smg_auth::{ApiKeyEntry, Role};

/// Parse the `--multimodal-tensor-transport` value into a `TransportMode`.
pub(crate) fn parse_transport_mode(value: &str) -> Result<TransportMode, String> {
    TransportMode::parse(value)
        .ok_or_else(|| format!("invalid value '{value}'; expected inline, shm, auto, or rdma"))
}

/// Parse the `--power-of-two-load-metric` value into a `PowerOfTwoLoadMetric`.
pub(crate) fn parse_power_of_two_load_metric(value: &str) -> Result<PowerOfTwoLoadMetric, String> {
    PowerOfTwoLoadMetric::parse(value).ok_or_else(|| {
        format!("invalid value '{value}'; expected expected_wait, requests, or least_requests")
    })
}

/// Parse one `--access-log-fields` name into an `AccessLogField`.
pub(crate) fn parse_access_log_field(value: &str) -> Result<AccessLogField, String> {
    value.parse()
}

/// Parse the `--mm-processing` value into an `MmProcessingMode`.
pub(crate) fn parse_mm_processing(value: &str) -> Result<MmProcessingMode, String> {
    MmProcessingMode::parse(value)
        .ok_or_else(|| format!("invalid value '{value}'; expected auto, router, or worker"))
}

/// Parse the `--mm-turbojpeg` value into a `TurboJpegPolicy`.
pub(crate) fn parse_mm_turbojpeg(value: &str) -> Result<TurboJpegPolicy, String> {
    TurboJpegPolicy::parse(value)
        .ok_or_else(|| format!("invalid value '{value}'; expected prefer or require"))
}

pub(crate) fn parse_positive_usize(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        Ok(_) => Err(format!("invalid value '{value}'; expected a count >= 1")),
        Err(err) => Err(format!("invalid value '{value}': {err}")),
    }
}

fn parse_usize_in_range(value: &str, max: usize) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if (1..=max).contains(&n) => Ok(n),
        Ok(n) => Err(format!("invalid value '{n}'; expected 1..={max}")),
        Err(err) => Err(format!("invalid value '{value}': {err}")),
    }
}

/// Parse a ratio in `(0.0, 1.0]`. Zero is excluded because a `>=` threshold of
/// 0.0 would mark every worker overloaded unconditionally.
pub(crate) fn parse_unit_fraction(value: &str) -> Result<f64, String> {
    match value.parse::<f64>() {
        Ok(v) if v > 0.0 && v <= 1.0 => Ok(v),
        Ok(_) => Err(format!(
            "invalid value '{value}'; expected a fraction in (0.0, 1.0]"
        )),
        Err(err) => Err(format!("invalid value '{value}': {err}")),
    }
}

pub(crate) fn parse_job_queue_capacity(value: &str) -> Result<usize, String> {
    parse_usize_in_range(value, 1_000_000)
}

pub(crate) fn parse_job_queue_concurrency(value: &str) -> Result<usize, String> {
    parse_usize_in_range(value, 100_000)
}

/// Validate `--model-id-from` value at CLI parse time.
pub(crate) fn parse_model_id_from(s: &str) -> Result<String, String> {
    ModelIdSource::parse(s)?;
    Ok(s.to_string())
}

/// Validate `--routing-key-headers` values at CLI parse time; names are
/// normalized to lowercase.
pub(crate) fn parse_routing_key_header(s: &str) -> Result<String, String> {
    http::header::HeaderName::try_from(s)
        .map(|name| name.as_str().to_string())
        .map_err(|e| format!("Invalid header name '{s}': {e}"))
}

/// Validate `--model-alias` value at CLI parse time (format: alias=canonical).
pub(crate) fn parse_model_alias(s: &str) -> Result<String, String> {
    let Some((alias, canonical)) = s.split_once('=') else {
        return Err(format!(
            "Invalid model-alias value '{s}'. Expected: <alias>=<canonical>"
        ));
    };
    if alias.is_empty() || canonical.is_empty() {
        return Err(format!(
            "Invalid model-alias value '{s}'. Alias and canonical model ID must be non-empty"
        ));
    }
    if alias == canonical {
        return Err(format!(
            "Invalid model-alias value '{s}'. Alias must differ from the canonical model ID"
        ));
    }
    Ok(s.to_string())
}

/// Parse a JWT role mapping from CLI format "idp_role=gateway_role". A
/// malformed entry is a configuration error, never a dropped mapping: with
/// the mapping dropped the map can end up empty, and an empty map grants the
/// gateway's admin role to any token whose roles claim says `admin`.
pub(crate) fn parse_role_mapping(mapping: &str) -> ConfigResult<(String, Role)> {
    let parts: Vec<&str> = mapping.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(ConfigError::InvalidValue {
            field: "jwt-role-mapping".to_string(),
            value: mapping.to_string(),
            reason: "expected 'idp_role=gateway_role'".to_string(),
        });
    }
    let idp_role = parts[0].to_string();
    let gateway_role = match parts[1].to_lowercase().as_str() {
        "admin" => Role::Admin,
        "user" => Role::User,
        _ => {
            return Err(ConfigError::InvalidValue {
                field: "jwt-role-mapping".to_string(),
                value: mapping.to_string(),
                reason: "the gateway role must be 'admin' or 'user'".to_string(),
            });
        }
    };
    Ok((idp_role, gateway_role))
}

/// Parse a control-plane API key from CLI format "id:name:role:key". A
/// malformed entry is a configuration error, never a dropped key: a dropped
/// key would leave the control plane open (or lock an operator out) with
/// nothing but a stderr line to show for it. The entry is not echoed, since
/// it carries the credential.
pub(crate) fn parse_control_plane_api_key(key_str: &str) -> ConfigResult<ApiKeyEntry> {
    let parts: Vec<&str> = key_str.splitn(4, ':').collect();
    if parts.len() != 4 {
        return Err(ConfigError::InvalidValue {
            field: "control-plane-api-keys".to_string(),
            value: "<redacted>".to_string(),
            reason: "expected 'id:name:role:key'".to_string(),
        });
    }
    let id = parts[0];
    let name = parts[1];
    let role_str = parts[2];
    let key = parts[3];

    let role = match role_str.to_lowercase().as_str() {
        "admin" => Role::Admin,
        "user" => Role::User,
        // The field is not named either: with the fields out of order it is the credential.
        _ => {
            return Err(ConfigError::InvalidValue {
                field: "control-plane-api-keys".to_string(),
                value: "<redacted>".to_string(),
                reason: "the role must be 'admin' or 'user'".to_string(),
            });
        }
    };

    Ok(ApiKeyEntry::new(id, name, key, role))
}

/// Parse a tenant-scoped data-plane API key from CLI format "tenant_id:key".
/// Only checks for the ':' separator; non-empty/duplicate checks live in
/// `ConfigValidator::validate_tenant_api_keys` so they also cover
/// `TenantApiKeyEntry` values from a config file or language binding. Fails
/// hard (an empty `AuthConfig` disables auth entirely, not just narrows it)
/// and never echoes `key_str`, which may be the plaintext credential.
pub(crate) fn parse_tenant_api_key(key_str: &str) -> ConfigResult<TenantApiKeyEntry> {
    let Some((tenant_id, key)) = key_str.split_once(':') else {
        return Err(ConfigError::InvalidValue {
            field: "tenant-api-key".to_string(),
            value: "<redacted>".to_string(),
            reason: "expected 'tenant_id:key' (missing ':' separator)".to_string(),
        });
    };

    Ok(TenantApiKeyEntry {
        tenant_id: tenant_id.trim().to_string(),
        key: key.trim().to_string(),
    })
}
