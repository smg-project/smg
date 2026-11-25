//! The settings a host reads for worker discovery, one type a provider, and
//! the error a provider answers settings it cannot honor with.

use std::{collections::HashMap, path::PathBuf};

use serde::{de, Deserialize, Deserializer, Serialize};

/// Why a provider refuses its settings.
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryConfigError {
    /// The settings break a rule that no one field carries: `reason` says
    /// which.
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },

    /// One field's value is refused: `field` names it, `value` shows it and
    /// `reason` says why.
    #[error("Invalid value for field '{field}': {value} - {reason}")]
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },
}

/// What resolving a provider's settings answers.
pub type DiscoveryConfigResult<T> = Result<T, DiscoveryConfigError>;

/// Worker discovery: the one provider that finds this host's workers, and
/// that provider's settings. Presence means discovery is on; there is no
/// `enabled` flag.
///
/// Serialized as a `provider` tag beside the provider's own flat fields:
///
/// ```yaml
/// discovery:
///   provider: kubernetes
///   selector: { app: worker }
///   port: 8000
/// ```
///
/// A host's `discovery` field also reads the flat Kubernetes object that
/// predates the tag; see [`deserialize_discovery`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
#[expect(
    clippy::large_enum_variant,
    reason = "one per router, built at startup; boxing the Kubernetes variant would complicate every construction and match to save a few hundred bytes"
)]
pub enum DiscoveryConfig {
    Kubernetes(KubernetesDiscoveryConfig),
    File(FileDiscoveryConfig),
    Slurm(SlurmDiscoveryConfig),
    Consul(ConsulDiscoveryConfig),
}

impl From<KubernetesDiscoveryConfig> for DiscoveryConfig {
    fn from(config: KubernetesDiscoveryConfig) -> Self {
        Self::Kubernetes(config)
    }
}

impl From<FileDiscoveryConfig> for DiscoveryConfig {
    fn from(config: FileDiscoveryConfig) -> Self {
        Self::File(config)
    }
}

impl From<SlurmDiscoveryConfig> for DiscoveryConfig {
    fn from(config: SlurmDiscoveryConfig) -> Self {
        Self::Slurm(config)
    }
}

impl From<ConsulDiscoveryConfig> for DiscoveryConfig {
    fn from(config: ConsulDiscoveryConfig) -> Self {
        Self::Consul(config)
    }
}

/// Deserialize a host's `discovery` field from either shape it has had.
///
/// The flat Kubernetes object that predates the `provider` tag has no tag and
/// carries an `enabled` flag. A missing tag means Kubernetes, the only
/// provider that object could describe. `enabled: false` becomes `None` here,
/// at deserialization, because canonical output has no `enabled` to carry the
/// `false`: a disabled config that is read and written back must stay
/// disabled.
///
/// Fields are read only after the object is buffered to find its tag, so a
/// YAML scalar keeps the type YAML gave it: a label value that looks like a
/// number or boolean must be quoted (`version: "1"`), as in a Kubernetes
/// manifest. The flat struct once read such values as strings.
///
/// Public so every input surface — the Python binding's `discovery` mapping
/// included — reads discovery by the same rules.
pub fn deserialize_discovery<'de, D>(deserializer: D) -> Result<Option<DiscoveryConfig>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(mut value) = Option::<serde_json::Value>::deserialize(deserializer)? else {
        return Ok(None);
    };
    if let Some(fields) = value.as_object_mut() {
        match fields.remove("enabled") {
            None | Some(serde_json::Value::Bool(true)) => {}
            Some(serde_json::Value::Bool(false)) => return Ok(None),
            Some(other) => {
                return Err(de::Error::custom(format!(
                    "discovery.enabled must be a boolean, got {other}"
                )));
            }
        }
        fields
            .entry("provider")
            .or_insert_with(|| serde_json::Value::from("kubernetes"));
    }
    DiscoveryConfig::deserialize(value)
        .map(Some)
        .map_err(de::Error::custom)
}

/// Kubernetes worker discovery: Pods matching a selector become workers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KubernetesDiscoveryConfig {
    /// None = all namespaces
    pub namespace: Option<String>,
    pub port: u16,
    pub check_interval_secs: u64,
    /// Regular mode
    pub selector: HashMap<String, String>,
    /// EPD mode encode
    #[serde(default)]
    pub encode_selector: HashMap<String, String>,
    /// PD mode prefill
    pub prefill_selector: HashMap<String, String>,
    /// PD mode decode
    pub decode_selector: HashMap<String, String>,
    pub bootstrap_port_annotation: String,
    /// Annotation listing a pod's worker data ports (comma-separated).
    /// Absent on a pod = single worker at `port`.
    #[serde(default = "default_worker_ports_annotation")]
    pub worker_ports_annotation: String,
    #[serde(default = "default_kv_connector_annotation")]
    pub kv_connector_annotation: String,
    #[serde(default = "default_kv_engine_id_annotation")]
    pub kv_engine_id_annotation: String,
    /// Pod label holding the worker's prefill/decode pairing protocol: a
    /// prefill and a decode pair only when their protocols are equal. The
    /// value is trimmed, and an absent or empty label states none.
    #[serde(default = "default_pairing_protocol_label")]
    pub pairing_protocol_label: String,
    /// Router node discovery for HA (Kubernetes label selector)
    #[serde(default)]
    pub router_selector: HashMap<String, String>,
    /// Annotation key to read mesh port from Router Pods
    #[serde(default = "default_router_mesh_port_annotation")]
    pub router_mesh_port_annotation: String,
    /// Source for per-worker model_id override: `namespace`, `label:<key>`, or `annotation:<key>`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id_source: Option<String>,
}

fn default_router_mesh_port_annotation() -> String {
    "sglang.ai/mesh-port".to_string()
}

fn default_worker_ports_annotation() -> String {
    "smg.ai/worker-ports".to_string()
}

fn default_kv_connector_annotation() -> String {
    "smg.ai/kv-connector".to_string()
}

fn default_kv_engine_id_annotation() -> String {
    "smg.ai/kv-engine-id".to_string()
}

fn default_pairing_protocol_label() -> String {
    "smg.ai/pairing-protocol".to_string()
}

impl Default for KubernetesDiscoveryConfig {
    fn default() -> Self {
        Self {
            namespace: None,
            port: 8000,
            check_interval_secs: 120,
            selector: HashMap::new(),
            encode_selector: HashMap::new(),
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            worker_ports_annotation: default_worker_ports_annotation(),
            kv_connector_annotation: default_kv_connector_annotation(),
            kv_engine_id_annotation: default_kv_engine_id_annotation(),
            pairing_protocol_label: default_pairing_protocol_label(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: default_router_mesh_port_annotation(),
            model_id_source: None,
        }
    }
}

/// The interval of a discovery provider's periodic pass when none is given,
/// the same for every provider. Kubernetes, file and Consul discovery apply a
/// change as it happens and use the pass to retry what earlier ones left
/// undone; Slurm discovery polls on it. ([`KubernetesDiscoveryConfig`] keeps its own
/// default, 120, for programmatic and serialized configuration.)
pub const DEFAULT_DISCOVERY_CHECK_INTERVAL_SECS: u64 = 30;

fn default_discovery_check_interval_secs() -> u64 {
    DEFAULT_DISCOVERY_CHECK_INTERVAL_SECS
}

/// File worker discovery: the workers listed in a JSON manifest on disk.
///
/// The manifest is reread in full whenever it changes, and every
/// `check_interval_secs` in case a change went unreported. It may be absent at
/// startup: discovery reads it once a writer creates it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileDiscoveryConfig {
    pub path: PathBuf,
    #[serde(default = "default_discovery_check_interval_secs")]
    pub check_interval_secs: u64,
}

/// Slurm worker discovery: one worker on every node of each running job whose
/// name maps to a worker type.
///
/// Every `check_interval_secs`, `squeue` lists the running jobs with a mapped
/// name, scoped to `user` (by default, the user the host runs as) and, when
/// given, to `account` and `partition`. `scontrol` resolves each job's nodes,
/// and every node serves one inference server on `port`.
///
/// Names map to worker types by the field that lists them: `job_names` are
/// regular workers, the others the pools of disaggregated routing. A name maps
/// to one type only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlurmDiscoveryConfig {
    #[serde(default)]
    pub job_names: Vec<String>,
    #[serde(default)]
    pub encode_job_names: Vec<String>,
    #[serde(default)]
    pub prefill_job_names: Vec<String>,
    #[serde(default)]
    pub decode_job_names: Vec<String>,
    /// The port of every discovered server.
    pub port: u16,
    /// The bootstrap port of the encode and prefill workers.
    #[serde(default)]
    pub bootstrap_port: Option<u16>,
    /// The user whose jobs are discovered; by default, the user `squeue`
    /// runs as.
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub account: Option<String>,
    #[serde(default)]
    pub partition: Option<String>,
    #[serde(default = "default_discovery_check_interval_secs")]
    pub check_interval_secs: u64,
}

/// Consul worker discovery: the instances of one Consul service whose health
/// checks all pass, each configured by its `smg_*` service metadata.
///
/// Consul is read with blocking queries, so a change applies as soon as
/// Consul reports it; every `check_interval_secs` the last good list is
/// reconciled again. `address` is Consul's HTTP API, by default
/// `CONSUL_HTTP_ADDR` or `127.0.0.1:8500`; `datacenter` is by default
/// the agent's own. The ACL token and the TLS settings come from Consul's own
/// environment variables, never from this configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsulDiscoveryConfig {
    pub service: String,
    #[serde(default)]
    pub address: Option<String>,
    #[serde(default)]
    pub datacenter: Option<String>,
    #[serde(default = "default_discovery_check_interval_secs")]
    pub check_interval_secs: u64,
}
