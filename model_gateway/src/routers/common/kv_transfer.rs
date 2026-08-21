//! vLLM PD KV-transfer connector handling, shared by both transports.
//!
//! vLLM disaggregation is sequential: the prefill leg is tagged with
//! connector params, the engine returns (or the router mints) the handoff
//! params, and the decode leg carries them. The gRPC pipeline and the HTTP
//! PD router both implement that flow; the connector vocabulary lives here
//! so the two stay in lockstep.

use tracing::warn;

use crate::{
    observability::metrics::metrics_labels,
    worker::{Worker, DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR, NIXL_CONNECTOR},
};

/// KV-transfer params tagged onto the NIXL prefill leg so the engine pins its
/// KV blocks and returns the handoff params for the decode worker.
pub(crate) const NIXL_PREFILL_KV_PARAMS: &str =
    r#"{"do_remote_decode":true,"do_remote_prefill":false}"#;

/// PD KV-transfer behavior derived from prefill worker metadata.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum KvConnectorMode {
    /// MooncakeConnector: mint a transfer_id, tag both legs, synthesize decode
    /// params from worker metadata; legacy host/port injection when the
    /// servicer predates kv_engine_id reporting (or DP runs without a pinned rank).
    Mooncake {
        host: String,
        port: u32,
        engine_id: Option<String>,
    },
    /// NixlConnector: tag prefill with do_remote_decode, relay returned params to decode.
    Nixl,
    /// Unknown/absent connector: relay returned params opportunistically.
    Passthrough,
}

impl KvConnectorMode {
    pub(crate) fn metrics_label(&self) -> &'static str {
        match self {
            Self::Mooncake { .. } => metrics_labels::KV_CONNECTOR_MOONCAKE,
            Self::Nixl => metrics_labels::KV_CONNECTOR_NIXL,
            Self::Passthrough => metrics_labels::KV_CONNECTOR_PASSTHROUGH,
        }
    }
}

pub(crate) fn kv_connector_mode(
    kv_connector: Option<&str>,
    bootstrap_host: &str,
    bootstrap_port: Option<u16>,
    kv_engine_id: Option<&str>,
) -> KvConnectorMode {
    match kv_connector {
        Some(MOONCAKE_CONNECTOR) => KvConnectorMode::Mooncake {
            host: bootstrap_host.to_string(),
            port: u32::from(bootstrap_port.unwrap_or(DEFAULT_BOOTSTRAP_PORT)),
            // Empty means unknown (forces the legacy fallback)
            engine_id: kv_engine_id.filter(|s| !s.is_empty()).map(str::to_string),
        },
        Some(NIXL_CONNECTOR) => KvConnectorMode::Nixl,
        _ => KvConnectorMode::Passthrough,
    }
}

/// Connector id of the engine core serving the prefill leg. With DP the cores
/// suffix the configured id as `{base}_dp{rank}`, so minting needs a pinned
/// rank; unpinned DP>1 yields None (no mint — decode recomputes locally).
pub(crate) fn effective_kv_engine_id(
    base: Option<&str>,
    dp_size: Option<usize>,
    dp_rank: Option<usize>,
) -> Option<String> {
    let base = base.filter(|s| !s.is_empty())?;
    if dp_size.unwrap_or(1) > 1 {
        dp_rank.map(|rank| format!("{base}_dp{rank}"))
    } else {
        Some(base.to_string())
    }
}

/// The connector mode for a PD pair, read off the prefill worker's metadata.
/// Discovered dp_size matters even without `--dp-aware` expansion: a DP>1
/// engine behind an unexpanded worker must not be minted for.
pub(crate) fn connector_mode_for_worker(worker: &dyn Worker) -> KvConnectorMode {
    let meta = worker.metadata();
    let dp_label = meta.spec.labels.get("dp_size");
    let label_dp = dp_label.and_then(|s| s.parse::<usize>().ok().filter(|v| *v > 0));
    let engine_id = if worker.dp_size().is_none() && dp_label.is_some() && label_dp.is_none() {
        // A dp_size label that does not parse as a positive integer means the
        // DP topology is unknown, not absent. Minting an unsuffixed engine id
        // could target the wrong engine core, so fail closed: no mint, decode
        // recomputes the prompt.
        warn!(
            worker = %worker.url(),
            dp_size = ?dp_label,
            "invalid dp_size label; treating DP topology as unknown and \
             skipping KV engine-id minting"
        );
        None
    } else {
        let dp_size = worker.dp_size().or(label_dp);
        effective_kv_engine_id(meta.spec.kv_engine_id.as_deref(), dp_size, worker.dp_rank())
    };
    kv_connector_mode(
        meta.spec.kv_connector.as_deref(),
        &meta.spec.bootstrap_host,
        meta.spec.bootstrap_port,
        engine_id.as_deref(),
    )
}

/// Prefill-leg params for Mooncake: the engine pins blocks under the minted id.
pub(crate) fn mooncake_prefill_params(transfer_id: &str) -> String {
    serde_json::json!({
        "do_remote_decode": true,
        "do_remote_prefill": false,
        "transfer_id": transfer_id,
    })
    .to_string()
}

/// Decode-leg params for Mooncake, synthesized from prefill worker metadata
/// (the engine returns nothing to relay; the connector is push-based).
pub(crate) fn mooncake_decode_params(
    transfer_id: &str,
    engine_id: &str,
    host: &str,
    port: u32,
) -> String {
    serde_json::json!({
        "do_remote_decode": false,
        "do_remote_prefill": true,
        "transfer_id": transfer_id,
        "remote_engine_id": engine_id,
        "remote_bootstrap_addr": format!("http://{host}:{port}"),
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_connector_mode_mooncake_uses_bootstrap_metadata() {
        let mode = kv_connector_mode(
            Some(MOONCAKE_CONNECTOR),
            "prefill-host",
            Some(9090),
            Some("engine-1"),
        );
        assert_eq!(
            mode,
            KvConnectorMode::Mooncake {
                host: "prefill-host".to_string(),
                port: 9090,
                engine_id: Some("engine-1".to_string()),
            }
        );
    }

    #[test]
    fn kv_connector_mode_mooncake_defaults_port_and_tolerates_missing_engine_id() {
        let mode = kv_connector_mode(Some(MOONCAKE_CONNECTOR), "prefill-host", None, None);
        assert_eq!(
            mode,
            KvConnectorMode::Mooncake {
                host: "prefill-host".to_string(),
                port: u32::from(DEFAULT_BOOTSTRAP_PORT),
                engine_id: None,
            }
        );
    }

    #[test]
    fn kv_connector_mode_mooncake_empty_engine_id_means_legacy() {
        let mode = kv_connector_mode(Some(MOONCAKE_CONNECTOR), "host", Some(9090), Some(""));
        assert_eq!(
            mode,
            KvConnectorMode::Mooncake {
                host: "host".to_string(),
                port: 9090,
                engine_id: None,
            }
        );
    }

    #[test]
    fn kv_connector_mode_nixl() {
        assert_eq!(
            kv_connector_mode(Some(NIXL_CONNECTOR), "ignored", Some(9090), None),
            KvConnectorMode::Nixl
        );
    }

    #[test]
    fn kv_connector_mode_unknown_or_missing_is_passthrough() {
        assert_eq!(
            kv_connector_mode(Some("LMCacheConnector"), "host", None, None),
            KvConnectorMode::Passthrough
        );
        assert_eq!(
            kv_connector_mode(None, "host", None, None),
            KvConnectorMode::Passthrough
        );
    }

    #[test]
    fn invalid_dp_size_label_fails_closed_on_minting() {
        use crate::worker::{BasicWorkerBuilder, WorkerType};

        let worker = BasicWorkerBuilder::new("http://prefill:8000")
            .worker_type(WorkerType::Prefill)
            .kv_connector(MOONCAKE_CONNECTOR)
            .kv_engine_id("eng")
            .label("dp_size", "not-a-number")
            .build();
        let mode = connector_mode_for_worker(&worker);
        // Unknown DP topology must not mint an unsuffixed engine id.
        assert!(matches!(
            mode,
            KvConnectorMode::Mooncake {
                engine_id: None,
                ..
            }
        ));

        let worker = BasicWorkerBuilder::new("http://prefill:8000")
            .worker_type(WorkerType::Prefill)
            .kv_connector(MOONCAKE_CONNECTOR)
            .kv_engine_id("eng")
            .label("dp_size", "1")
            .build();
        let mode = connector_mode_for_worker(&worker);
        assert!(matches!(
            mode,
            KvConnectorMode::Mooncake {
                engine_id: Some(ref id),
                ..
            } if id == "eng"
        ));
    }

    #[test]
    fn effective_engine_id_requires_pinned_rank_under_dp() {
        assert_eq!(
            effective_kv_engine_id(Some("eng"), Some(2), Some(1)),
            Some("eng_dp1".to_string())
        );
        assert_eq!(effective_kv_engine_id(Some("eng"), Some(2), None), None);
        assert_eq!(
            effective_kv_engine_id(Some("eng"), None, None),
            Some("eng".to_string())
        );
        assert_eq!(effective_kv_engine_id(Some(""), None, None), None);
        assert_eq!(effective_kv_engine_id(None, Some(2), Some(0)), None);
    }
}
