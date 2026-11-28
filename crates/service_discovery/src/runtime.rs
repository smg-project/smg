//! Worker discovery as a running host uses it, and the one conversion
//! that produces it from configuration.

#[cfg(any(feature = "file", feature = "kubernetes"))]
use std::time::Duration;
use std::{error::Error, sync::Arc};

use tokio::task;

#[cfg(feature = "consul")]
use super::consul::{self, ConsulProviderConfig};
#[cfg(feature = "file")]
use super::file::{self, FileProviderConfig};
#[cfg(feature = "kubernetes")]
use super::kubernetes::{self, ModelIdSource, ServiceDiscoveryConfig};
#[cfg(feature = "slurm")]
use super::slurm::{self, SlurmProviderConfig};
use super::{
    config::{
        ConsulDiscoveryConfig, DiscoveryConfig, DiscoveryConfigError, DiscoveryConfigResult,
        FileDiscoveryConfig, KubernetesDiscoveryConfig, SlurmDiscoveryConfig,
    },
    host::DiscoveryHost,
    provider::Roles,
};

/// The selected worker-discovery provider, resolved for the running host.
///
/// `Option<RuntimeDiscoveryConfig>` is on versus off; a variant carries no
/// `enabled` flag of its own. A provider is a variant only in a build that
/// holds it: each provider is a feature of the crate, and a build with none
/// has no variant.
///
/// The Kubernetes and the Consul settings are boxed. Each is several times
/// the size of the file or Slurm settings, and unboxed the enum would carry
/// that size wherever a build holds one of them.
///
/// The enum is open to more variants. A build holds the providers of every
/// host in it, so a host matches the enum with an arm for the rest.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RuntimeDiscoveryConfig {
    #[cfg(feature = "kubernetes")]
    Kubernetes(Box<ServiceDiscoveryConfig>),
    #[cfg(feature = "file")]
    File(FileProviderConfig),
    #[cfg(feature = "slurm")]
    Slurm(SlurmProviderConfig),
    #[cfg(feature = "consul")]
    Consul(Box<ConsulProviderConfig>),
}

impl RuntimeDiscoveryConfig {
    /// Resolve configuration into its runtime form.
    ///
    /// Every input surface of a host (a command line, a language binding)
    /// converts through here, so a setting cannot reach the runtime through one
    /// surface and be dropped by another. Values arrive as the surface built
    /// them, defaults included: a command line's port 80 and
    /// `KubernetesDiscoveryConfig`'s port 8000 are each kept, not reconciled.
    ///
    /// Settings for a provider this build does not hold are refused, in words
    /// that name the provider and the feature that builds it. For a built
    /// provider, settings are checked before resolving runtime inputs. When
    /// roles are absent, settings that apply to any roles are still checked.
    pub fn from_config(
        discovery: &DiscoveryConfig,
        roles: Option<Roles>,
    ) -> DiscoveryConfigResult<Self> {
        let built = match discovery {
            DiscoveryConfig::Kubernetes(_) => cfg!(feature = "kubernetes"),
            DiscoveryConfig::File(_) => cfg!(feature = "file"),
            DiscoveryConfig::Slurm(_) => cfg!(feature = "slurm"),
            DiscoveryConfig::Consul(_) => cfg!(feature = "consul"),
        };
        if built {
            discovery.validate_for_roles(roles)?;
        }
        match discovery {
            DiscoveryConfig::Kubernetes(kubernetes) => kubernetes_runtime(kubernetes, roles),
            DiscoveryConfig::File(file) => file_runtime(file, roles),
            DiscoveryConfig::Slurm(slurm) => slurm_runtime(slurm, roles),
            DiscoveryConfig::Consul(consul) => consul_runtime(consul, roles),
        }
    }
}

/// The refusal of settings for a provider this build does not hold.
#[cfg(not(all(
    feature = "consul",
    feature = "file",
    feature = "kubernetes",
    feature = "slurm"
)))]
fn not_built(provider: &str, feature: &str) -> DiscoveryConfigError {
    DiscoveryConfigError::ValidationFailed {
        reason: format!(
            "this build holds no {provider} discovery provider: build `smg-service-discovery` with its `{feature}` feature"
        ),
    }
}

#[cfg(feature = "kubernetes")]
fn kubernetes_runtime(
    config: &KubernetesDiscoveryConfig,
    roles: Option<Roles>,
) -> DiscoveryConfigResult<RuntimeDiscoveryConfig> {
    let model_id_source = config
        .model_id_source
        .as_deref()
        .map(|source| {
            ModelIdSource::parse(source).map_err(|reason| DiscoveryConfigError::InvalidValue {
                field: "discovery.model_id_source".to_string(),
                value: source.to_string(),
                reason,
            })
        })
        .transpose()?;

    let settings = ServiceDiscoveryConfig {
        selector: config.selector.clone(),
        check_interval: Duration::from_secs(config.check_interval_secs),
        port: config.port,
        namespace: config.namespace.clone(),
        disaggregated_mode: matches!(
            roles,
            Some(Roles::PrefillDecode | Roles::EncodePrefillDecode)
        ),
        encode_selector: config.encode_selector.clone(),
        prefill_selector: config.prefill_selector.clone(),
        decode_selector: config.decode_selector.clone(),
        bootstrap_port_annotation: config.bootstrap_port_annotation.clone(),
        worker_ports_annotation: config.worker_ports_annotation.clone(),
        kv_connector_annotation: config.kv_connector_annotation.clone(),
        kv_engine_id_annotation: config.kv_engine_id_annotation.clone(),
        pairing_protocol_label: config.pairing_protocol_label.clone(),
        model_id_source,
    };
    Ok(RuntimeDiscoveryConfig::Kubernetes(Box::new(settings)))
}

#[cfg(not(feature = "kubernetes"))]
fn kubernetes_runtime(
    _: &KubernetesDiscoveryConfig,
    _: Option<Roles>,
) -> DiscoveryConfigResult<RuntimeDiscoveryConfig> {
    Err(not_built("Kubernetes", "kubernetes"))
}

#[cfg(feature = "file")]
fn file_runtime(
    config: &FileDiscoveryConfig,
    roles: Option<Roles>,
) -> DiscoveryConfigResult<RuntimeDiscoveryConfig> {
    Ok(RuntimeDiscoveryConfig::File(FileProviderConfig::new(
        config.path.clone(),
        Duration::from_secs(config.check_interval_secs),
        roles,
    )?))
}

#[cfg(not(feature = "file"))]
fn file_runtime(
    _: &FileDiscoveryConfig,
    _: Option<Roles>,
) -> DiscoveryConfigResult<RuntimeDiscoveryConfig> {
    Err(not_built("file", "file"))
}

#[cfg(feature = "slurm")]
fn slurm_runtime(
    config: &SlurmDiscoveryConfig,
    roles: Option<Roles>,
) -> DiscoveryConfigResult<RuntimeDiscoveryConfig> {
    Ok(RuntimeDiscoveryConfig::Slurm(SlurmProviderConfig::new(
        config, roles,
    )?))
}

#[cfg(not(feature = "slurm"))]
fn slurm_runtime(
    _: &SlurmDiscoveryConfig,
    _: Option<Roles>,
) -> DiscoveryConfigResult<RuntimeDiscoveryConfig> {
    Err(not_built("Slurm", "slurm"))
}

#[cfg(feature = "consul")]
fn consul_runtime(
    config: &ConsulDiscoveryConfig,
    roles: Option<Roles>,
) -> DiscoveryConfigResult<RuntimeDiscoveryConfig> {
    let settings = ConsulProviderConfig::new(config, roles)?;
    Ok(RuntimeDiscoveryConfig::Consul(Box::new(settings)))
}

#[cfg(not(feature = "consul"))]
fn consul_runtime(
    _: &ConsulDiscoveryConfig,
    _: Option<Roles>,
) -> DiscoveryConfigResult<RuntimeDiscoveryConfig> {
    Err(not_built("Consul", "consul"))
}

/// Start the selected provider's discovery task.
pub async fn start_service_discovery(
    config: RuntimeDiscoveryConfig,
    host: Arc<dyn DiscoveryHost>,
) -> Result<task::JoinHandle<()>, Box<dyn Error + Send + Sync>> {
    // A build with no provider has no variant to start, and nothing takes the host.
    #[cfg(not(any(
        feature = "consul",
        feature = "file",
        feature = "kubernetes",
        feature = "slurm"
    )))]
    let _ = &host;
    match config {
        #[cfg(feature = "kubernetes")]
        RuntimeDiscoveryConfig::Kubernetes(config) => {
            Ok(kubernetes::start_kubernetes_discovery(*config, host).await?)
        }
        #[cfg(feature = "file")]
        RuntimeDiscoveryConfig::File(config) => Ok(file::start_file_discovery(config, host)),
        #[cfg(feature = "slurm")]
        RuntimeDiscoveryConfig::Slurm(config) => Ok(slurm::start_slurm_discovery(config, host)?),
        #[cfg(feature = "consul")]
        RuntimeDiscoveryConfig::Consul(config) => {
            Ok(consul::start_consul_discovery(*config, host)?)
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/runtime/tests.rs"]
mod tests;
