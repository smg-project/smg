//! Worker discovery as the running gateway uses it, and the one conversion
//! that produces it from configuration.

use std::{error::Error, sync::Arc, time::Duration};

use tokio::task;

use super::kubernetes::{self, ModelIdSource, ServiceDiscoveryConfig};
use crate::{
    app_context::AppContext,
    config::{ConfigError, ConfigResult, DiscoveryConfig, KubernetesDiscoveryConfig, RoutingMode},
};

/// The selected worker-discovery provider, resolved for the running gateway.
///
/// `Option<RuntimeDiscoveryConfig>` is on versus off; a variant carries no
/// `enabled` flag of its own.
#[derive(Debug, Clone)]
pub enum RuntimeDiscoveryConfig {
    Kubernetes(ServiceDiscoveryConfig),
}

impl RuntimeDiscoveryConfig {
    /// Resolve configuration into its runtime form.
    ///
    /// Every input surface — the CLI and the Python binding — converts
    /// through here, so a setting cannot reach the runtime through one surface
    /// and be dropped by another. Values arrive as the surface built them,
    /// defaults included: the CLI's port 80 and `KubernetesDiscoveryConfig`'s
    /// port 8000 are each kept, not reconciled.
    pub fn from_config(discovery: &DiscoveryConfig, mode: &RoutingMode) -> ConfigResult<Self> {
        match discovery {
            DiscoveryConfig::Kubernetes(kubernetes) => {
                Ok(Self::Kubernetes(kubernetes_runtime(kubernetes, mode)?))
            }
        }
    }
}

fn kubernetes_runtime(
    config: &KubernetesDiscoveryConfig,
    mode: &RoutingMode,
) -> ConfigResult<ServiceDiscoveryConfig> {
    let model_id_source = config
        .model_id_source
        .as_deref()
        .map(|source| {
            ModelIdSource::parse(source).map_err(|reason| ConfigError::InvalidValue {
                field: "discovery.model_id_source".to_string(),
                value: source.to_string(),
                reason,
            })
        })
        .transpose()?;

    Ok(ServiceDiscoveryConfig {
        selector: config.selector.clone(),
        check_interval: Duration::from_secs(config.check_interval_secs),
        port: config.port,
        namespace: config.namespace.clone(),
        disaggregated_mode: matches!(
            mode,
            RoutingMode::PrefillDecode { .. } | RoutingMode::EncodePrefillDecode { .. }
        ),
        encode_selector: config.encode_selector.clone(),
        prefill_selector: config.prefill_selector.clone(),
        decode_selector: config.decode_selector.clone(),
        bootstrap_port_annotation: config.bootstrap_port_annotation.clone(),
        worker_ports_annotation: config.worker_ports_annotation.clone(),
        kv_connector_annotation: config.kv_connector_annotation.clone(),
        kv_engine_id_annotation: config.kv_engine_id_annotation.clone(),
        model_id_source,
    })
}

/// Start the selected provider's discovery task.
pub async fn start_service_discovery(
    config: RuntimeDiscoveryConfig,
    app_context: Arc<AppContext>,
) -> Result<task::JoinHandle<()>, Box<dyn Error + Send + Sync>> {
    match config {
        RuntimeDiscoveryConfig::Kubernetes(config) => {
            Ok(kubernetes::start_kubernetes_discovery(config, app_context).await?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regular() -> RoutingMode {
        RoutingMode::Regular {
            worker_urls: vec![],
        }
    }

    fn kubernetes(config: &DiscoveryConfig, mode: &RoutingMode) -> ServiceDiscoveryConfig {
        let RuntimeDiscoveryConfig::Kubernetes(runtime) =
            RuntimeDiscoveryConfig::from_config(config, mode).unwrap();
        runtime
    }

    /// Every Kubernetes setting reaches the runtime, the KV annotation names
    /// included: those once flowed through only one of two hand-written
    /// conversions.
    #[test]
    fn kubernetes_settings_reach_the_runtime() {
        let config = DiscoveryConfig::Kubernetes(KubernetesDiscoveryConfig {
            namespace: Some("prod".to_string()),
            port: 9000,
            check_interval_secs: 45,
            selector: [("app".to_string(), "worker".to_string())].into(),
            bootstrap_port_annotation: "example.com/bootstrap".to_string(),
            worker_ports_annotation: "example.com/ports".to_string(),
            kv_connector_annotation: "example.com/connector".to_string(),
            kv_engine_id_annotation: "example.com/engine-id".to_string(),
            model_id_source: Some("namespace".to_string()),
            ..Default::default()
        });

        let runtime = kubernetes(&config, &regular());

        assert_eq!(runtime.namespace.as_deref(), Some("prod"));
        assert_eq!(runtime.port, 9000);
        assert_eq!(runtime.check_interval, Duration::from_secs(45));
        assert_eq!(
            runtime.selector.get("app").map(String::as_str),
            Some("worker")
        );
        assert_eq!(runtime.bootstrap_port_annotation, "example.com/bootstrap");
        assert_eq!(runtime.worker_ports_annotation, "example.com/ports");
        assert_eq!(runtime.kv_connector_annotation, "example.com/connector");
        assert_eq!(runtime.kv_engine_id_annotation, "example.com/engine-id");
        assert!(matches!(
            runtime.model_id_source,
            Some(ModelIdSource::Namespace)
        ));
        assert!(!runtime.disaggregated_mode);
    }

    #[test]
    fn disaggregated_modes_select_role_selectors() {
        let config = DiscoveryConfig::Kubernetes(KubernetesDiscoveryConfig::default());
        for mode in [
            RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: None,
            },
            RoutingMode::EncodePrefillDecode {
                encode_urls: vec![],
                prefill_urls: vec![],
                decode_urls: vec![],
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            },
        ] {
            assert!(kubernetes(&config, &mode).disaggregated_mode, "{mode:?}");
        }
    }

    #[test]
    fn invalid_model_id_source_is_a_config_error() {
        let config = DiscoveryConfig::Kubernetes(KubernetesDiscoveryConfig {
            model_id_source: Some("pod-name".to_string()),
            ..Default::default()
        });
        let err = RuntimeDiscoveryConfig::from_config(&config, &regular()).unwrap_err();
        assert!(
            matches!(err, ConfigError::InvalidValue { ref field, .. } if field == "discovery.model_id_source"),
            "{err}"
        );
    }
}
