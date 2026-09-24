//! Wire types of the RL control plane (`/v1/rl/*`): worker discovery rows,
//! per-call outcomes, and fan-out envelopes.
//!
//! The logic that produces them lives in the `smg-rl` crate; this module is
//! the contract that clients and the OpenAPI generator depend on, in the same
//! way [`super::worker::WorkerInfo`] is the contract of `GET /workers`.

use std::collections::{BTreeMap, HashMap};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Version of the `/v1/rl` wire contract, reported by `GET /v1/rl/workers`.
/// Bumped only for incompatible changes; additive fields keep the version.
pub const RL_PROTOCOL_VERSION: u32 = 1;

/// Where a worker's capability row came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum RlCapabilitySource {
    /// The built-in per-engine table.
    Static,
    /// At least one `rl.*` registration label overrode the table.
    Label,
}

/// What an engine can do for RL control, as reported by discovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RlCapabilities {
    pub source: RlCapabilitySource,
    /// Accepted `mode` values of the engine's pause route.
    pub pause_modes: Vec<String>,
    /// Weight sources the engine can refit from (`disk`, `tensor`, ...).
    pub update_from: Vec<String>,
    pub abort: bool,
    pub flush_cache: bool,
    pub sleep_wake: bool,
    /// Whether the engine reports a weight version after a refit.
    pub reports_weight_version: bool,
}

/// One row of `GET /v1/rl/workers`. DP-aware ranks that share a `base_url`
/// collapse into one row; `dp_ranks` counts them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RlWorkerEntry {
    /// Registry UUID; the `{id}` of the per-worker routes.
    pub id: String,
    /// Registry URL; carries an `@<rank>` suffix for DP-aware workers.
    pub url: String,
    /// The address control calls are sent to.
    pub base_url: String,
    /// Engine name (`sglang`, `vllm`, ...); `unknown` when undetected.
    pub engine: String,
    pub engine_version: Option<String>,
    pub model_id: String,
    pub worker_type: String,
    pub connection_mode: String,
    pub tp_size: Option<u64>,
    pub dp_size: Option<u64>,
    pub pp_size: Option<u64>,
    /// DP ranks collapsed into this row; 1 for a non-DP worker.
    pub dp_ranks: usize,
    /// The `role` registration label, when set.
    pub role: Option<String>,
    /// Registry status (`ready`, `not_ready`, ...).
    pub health: String,
    pub weight_version: Option<String>,
    /// Registration labels, discovered metadata merged with caller labels.
    pub labels: HashMap<String, String>,
    pub capabilities: RlCapabilities,
}

/// Body of `GET /v1/rl/workers`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RlWorkersResponse {
    /// [`RL_PROTOCOL_VERSION`] of the gateway that answered.
    pub protocol_version: u32,
    pub workers: Vec<RlWorkerEntry>,
    pub total: usize,
}

/// One completed engine call, whatever its HTTP status. Also the body of the
/// per-worker proxy route, whose HTTP status mirrors `status`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RlCallOutcome {
    pub worker_id: String,
    pub url: String,
    /// The engine's HTTP status.
    pub status: u16,
    pub latency_ms: u64,
    /// The engine's body: parsed JSON when it was JSON, otherwise text.
    pub body: Value,
    /// Set when the body was cut at the gateway's size cap.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub body_truncated: bool,
}

impl RlCallOutcome {
    /// Whether the engine answered 2xx.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// One fan-out target that did not succeed: a transport failure (no
/// outcome) or an engine error status (also present in `results`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RlFailedCall {
    pub worker_id: String,
    pub url: String,
    /// `upstream_error`, `upstream_unreachable`, `upstream_timeout`, or
    /// `unsupported_connection_mode`.
    pub error: String,
    pub message: String,
    /// The engine's HTTP status, for `upstream_error`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// The worker's connection mode, for `unsupported_connection_mode`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection_mode: Option<String>,
}

/// Body of a fan-out: HTTP 200 when `failed` is empty, 207 otherwise.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RlFanoutResponse {
    /// Every completed call keyed by worker id, whatever its status.
    pub results: BTreeMap<String, RlCallOutcome>,
    pub failed: Vec<RlFailedCall>,
    pub total: usize,
    pub succeeded: usize,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use schemars::schema_for;
    use serde_json::{json, Value};

    use super::*;

    fn sample_entry() -> RlWorkerEntry {
        RlWorkerEntry {
            id: "w1".to_string(),
            url: "http://a:1".to_string(),
            base_url: "http://a:1".to_string(),
            engine: "sglang".to_string(),
            engine_version: None,
            model_id: "m".to_string(),
            worker_type: "regular".to_string(),
            connection_mode: "http".to_string(),
            tp_size: Some(1),
            dp_size: None,
            pp_size: None,
            dp_ranks: 1,
            role: None,
            health: "ready".to_string(),
            weight_version: Some("default".to_string()),
            labels: HashMap::new(),
            capabilities: RlCapabilities {
                source: RlCapabilitySource::Static,
                pause_modes: vec!["abort".to_string()],
                update_from: vec![],
                abort: true,
                flush_cache: true,
                sleep_wake: false,
                reports_weight_version: true,
            },
        }
    }

    #[test]
    fn capability_source_serializes_lowercase() {
        assert_eq!(
            serde_json::to_value(RlCapabilitySource::Static).unwrap(),
            json!("static")
        );
        assert_eq!(
            serde_json::to_value(RlCapabilitySource::Label).unwrap(),
            json!("label")
        );
    }

    #[test]
    fn workers_response_round_trips_and_reports_the_protocol_version() {
        let resp = RlWorkersResponse {
            protocol_version: RL_PROTOCOL_VERSION,
            workers: vec![sample_entry()],
            total: 1,
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["protocol_version"], 1);
        assert_eq!(
            v["workers"][0]["engine_version"],
            Value::Null,
            "absent optionals stay explicit nulls on the wire"
        );
        let back: RlWorkersResponse = serde_json::from_value(v).unwrap();
        assert_eq!(back, resp);
    }

    #[test]
    fn call_outcome_omits_body_truncated_unless_set() {
        let mut outcome = RlCallOutcome {
            worker_id: "w".to_string(),
            url: "http://w".to_string(),
            status: 200,
            latency_ms: 3,
            body: json!({"ok": true}),
            body_truncated: false,
        };
        assert!(outcome.is_success());
        assert!(serde_json::to_value(&outcome)
            .unwrap()
            .get("body_truncated")
            .is_none());
        outcome.body_truncated = true;
        assert_eq!(
            serde_json::to_value(&outcome).unwrap()["body_truncated"],
            true
        );

        let parsed: RlCallOutcome = serde_json::from_value(json!({
            "worker_id": "w", "url": "u", "status": 500, "latency_ms": 1, "body": "x"
        }))
        .unwrap();
        assert!(!parsed.body_truncated);
        assert!(!parsed.is_success());
    }

    #[test]
    fn failed_call_omits_absent_context() {
        let f = RlFailedCall {
            worker_id: "w".to_string(),
            url: "u".to_string(),
            error: "upstream_unreachable".to_string(),
            message: "m".to_string(),
            status: None,
            connection_mode: None,
        };
        let v = serde_json::to_value(&f).unwrap();
        assert!(v.get("status").is_none());
        assert!(v.get("connection_mode").is_none());
        let empty: RlFanoutResponse = serde_json::from_value(json!({
            "results": {}, "failed": [], "total": 0, "succeeded": 0
        }))
        .unwrap();
        assert_eq!(empty, RlFanoutResponse::default());
    }

    #[test]
    fn every_wire_type_has_a_json_schema() {
        for schema in [
            schema_for!(RlWorkersResponse),
            schema_for!(RlWorkerEntry),
            schema_for!(RlCallOutcome),
            schema_for!(RlFanoutResponse),
        ] {
            assert!(schema.to_value().get("title").is_some());
        }
    }
}
