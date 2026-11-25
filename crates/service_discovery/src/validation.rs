//! Checks a provider's startup settings.

use std::{collections::HashMap, fmt::Display};

use super::{
    config::{
        ConsulDiscoveryConfig, DiscoveryConfig, DiscoveryConfigError, DiscoveryConfigResult,
        FileDiscoveryConfig, KubernetesDiscoveryConfig, SlurmDiscoveryConfig,
    },
    provider::Roles,
};

impl DiscoveryConfig {
    /// Check the selected provider's settings for the given worker roles.
    ///
    /// This check uses only the supplied settings and works with any feature set.
    pub fn validate(&self, roles: Roles) -> DiscoveryConfigResult<()> {
        self.validate_for_roles(Some(roles))
    }

    // With no roles, only the settings that apply to any roles are checked.
    pub(crate) fn validate_for_roles(&self, roles: Option<Roles>) -> DiscoveryConfigResult<()> {
        match self {
            Self::Kubernetes(config) => validate_kubernetes_discovery(config, roles),
            Self::File(config) => validate_file_discovery(config),
            Self::Slurm(config) => validate_slurm_discovery(config, roles),
            Self::Consul(config) => validate_consul_discovery(config),
        }
    }
}

fn invalid_value(field: &str, value: impl Display, reason: &str) -> DiscoveryConfigError {
    DiscoveryConfigError::InvalidValue {
        field: field.to_string(),
        value: value.to_string(),
        reason: reason.to_string(),
    }
}

fn validate_check_interval(check_interval_secs: u64) -> DiscoveryConfigResult<()> {
    if check_interval_secs == 0 {
        return Err(invalid_value(
            "discovery.check_interval_secs",
            check_interval_secs,
            "Must be > 0",
        ));
    }
    Ok(())
}

fn validate_slurm_discovery(
    discovery: &SlurmDiscoveryConfig,
    roles: Option<Roles>,
) -> DiscoveryConfigResult<()> {
    if discovery.port == 0 {
        return Err(invalid_value("discovery.port", 0, "Port must be > 0"));
    }
    if discovery.bootstrap_port == Some(0) {
        return Err(invalid_value(
            "discovery.bootstrap_port",
            0,
            "Port must be > 0",
        ));
    }
    validate_check_interval(discovery.check_interval_secs)?;

    let jobs = [
        ("discovery.job_names", "regular", &discovery.job_names),
        (
            "discovery.encode_job_names",
            "encode",
            &discovery.encode_job_names,
        ),
        (
            "discovery.prefill_job_names",
            "prefill",
            &discovery.prefill_job_names,
        ),
        (
            "discovery.decode_job_names",
            "decode",
            &discovery.decode_job_names,
        ),
    ];
    // A comma would split one filter into two in the list passed to squeue.
    let scopes = [
        ("discovery.user", &discovery.user),
        ("discovery.account", &discovery.account),
        ("discovery.partition", &discovery.partition),
    ];
    let values = jobs
        .iter()
        .flat_map(|(field, _, names)| names.iter().map(move |name| (*field, name)))
        .chain(
            scopes
                .iter()
                .filter_map(|(field, value)| value.as_ref().map(|value| (*field, value))),
        );
    for (field, value) in values {
        if value.is_empty() || value.contains(',') {
            return Err(invalid_value(
                field,
                value,
                "Must be non-empty and contain no comma",
            ));
        }
    }

    let mut role_of = HashMap::<&str, &str>::new();
    for (_, role, names) in &jobs {
        for name in *names {
            if let Some(other) = role_of.insert(name, role) {
                if other != *role {
                    return Err(DiscoveryConfigError::ValidationFailed {
                        reason: format!(
                            "Slurm job name '{name}' maps to both {other} and {role} workers"
                        ),
                    });
                }
            }
        }
    }

    let [regular, encode, prefill, decode] = jobs.map(|(_, _, names)| !names.is_empty());
    let (needed, unused) = match roles {
        Some(Roles::Regular) => (
            (!regular).then_some("Regular mode with Slurm discovery requires a regular job name"),
            (encode || prefill || decode).then_some(
                "Regular mode has no encode, prefill or decode workers to map Slurm jobs to",
            ),
        ),
        Some(Roles::PrefillDecode) => (
            (!(prefill || decode))
                .then_some("PD mode with Slurm discovery requires a prefill or decode job name"),
            (regular || encode)
                .then_some("PD mode has no regular or encode workers to map Slurm jobs to"),
        ),
        Some(Roles::EncodePrefillDecode) => (
            (!(encode && prefill && decode)).then_some(
                "EPD mode with Slurm discovery requires encode, prefill and decode job names",
            ),
            regular.then_some("EPD mode has no regular workers to map Slurm jobs to"),
        ),
        None => (None, None),
    };
    if let Some(reason) = needed.or(unused) {
        return Err(DiscoveryConfigError::ValidationFailed {
            reason: reason.to_string(),
        });
    }
    if discovery.bootstrap_port.is_some() && !(encode || prefill) {
        return Err(DiscoveryConfigError::ValidationFailed {
            reason: "A Slurm bootstrap port applies only to encode and prefill workers".to_string(),
        });
    }
    Ok(())
}

fn validate_file_discovery(discovery: &FileDiscoveryConfig) -> DiscoveryConfigResult<()> {
    if discovery.path.as_os_str().is_empty() {
        return Err(invalid_value(
            "discovery.path",
            "",
            "File discovery needs a manifest path",
        ));
    }
    validate_check_interval(discovery.check_interval_secs)
}

fn validate_consul_discovery(discovery: &ConsulDiscoveryConfig) -> DiscoveryConfigResult<()> {
    if discovery.service.is_empty() {
        return Err(invalid_value(
            "discovery.service",
            "",
            "Consul discovery needs a service name",
        ));
    }
    for (field, value) in [
        ("discovery.address", &discovery.address),
        ("discovery.datacenter", &discovery.datacenter),
    ] {
        if value.as_deref() == Some("") {
            return Err(invalid_value(field, "", "Must not be empty when given"));
        }
    }
    validate_check_interval(discovery.check_interval_secs)
}

fn validate_kubernetes_discovery(
    discovery: &KubernetesDiscoveryConfig,
    roles: Option<Roles>,
) -> DiscoveryConfigResult<()> {
    if discovery.port == 0 {
        return Err(invalid_value(
            "discovery.port",
            discovery.port,
            "Port must be > 0",
        ));
    }

    validate_check_interval(discovery.check_interval_secs)?;

    for (field, value, reason) in [
        (
            "discovery.kv_connector_annotation",
            &discovery.kv_connector_annotation,
            "Annotation name must not be empty or padded with whitespace",
        ),
        (
            "discovery.kv_engine_id_annotation",
            &discovery.kv_engine_id_annotation,
            "Annotation name must not be empty or padded with whitespace",
        ),
        (
            "discovery.pairing_protocol_label",
            &discovery.pairing_protocol_label,
            "Label name must not be empty or padded with whitespace",
        ),
    ] {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed != value {
            return Err(invalid_value(field, value, reason));
        }
    }

    let reason = match roles {
        Some(Roles::Regular) if discovery.selector.is_empty() => {
            "Regular mode with service discovery requires a non-empty selector"
        }
        Some(Roles::PrefillDecode)
            if discovery.prefill_selector.is_empty() && discovery.decode_selector.is_empty() =>
        {
            "PD mode with service discovery requires at least one non-empty selector (prefill or decode)"
        }
        Some(Roles::EncodePrefillDecode)
            if discovery.encode_selector.is_empty()
                || discovery.prefill_selector.is_empty()
                || discovery.decode_selector.is_empty() =>
        {
            "EPD mode with service discovery requires non-empty encode_selector, prefill_selector, and decode_selector"
        }
        Some(Roles::Regular | Roles::PrefillDecode | Roles::EncodePrefillDecode) | None => {
            return Ok(());
        }
    };
    Err(DiscoveryConfigError::ValidationFailed {
        reason: reason.to_string(),
    })
}

#[cfg(test)]
mod tests;
