//! Managed EPD state is separate from the legacy servicer registry and fallback.

pub mod decode;
pub mod ledger;
pub mod native;
pub mod service;

use smg_grpc_client::sglang_managed::proto::Capabilities;

/// Explicit managed-only launch configuration; legacy launch has no such value.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagedConfig {
    pub worker_id: String,
    pub generation: String,
    pub role: String,
    pub metadata_endpoint: Option<String>,
    pub max_records: usize,
    pub max_active: usize,
    pub max_work: u64,
    pub max_context_bytes: u64,
    pub retention_ms: u64,
    pub max_duration_ms: u64,
    pub max_clock_error_ms: u64,
    pub contract: ManagedContract,
}

/// Deployment contracts must identify fixed weights and the validated processor/transport.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManagedContract {
    pub model_revision: String,
    pub processor_contract: String,
    pub embedding_dtype: String,
    pub kv_backend: String,
    pub embedding_backend: String,
    pub topology_contract: String,
    pub bootstrap_host: String,
    pub bootstrap_port: u32,
    pub tp_size: u32,
}

impl ManagedConfig {
    /// Validate before binding any listener or launching a scheduler.
    pub fn validate(&self) -> Result<(), tonic::Status> {
        use smg_grpc_client::sglang_managed::{proto::Instance, validate_instance};
        validate_instance(&Instance {
            worker_id: self.worker_id.clone(),
            generation: self.generation.clone(),
        })?;
        if self.max_clock_error_ms == 0
            || self.contract.model_revision.is_empty()
            || self.contract.processor_contract.is_empty()
            || self.contract.topology_contract.is_empty()
            || self.contract.tp_size == 0
            || !matches!(
                self.contract.embedding_dtype.as_str(),
                "bfloat16" | "float16" | "float32"
            )
            || !matches!(
                self.contract.embedding_backend.as_str(),
                "mooncake" | "zmq_to_scheduler"
            )
            || self.contract.kv_backend.is_empty()
            || (self.role == "prefill"
                && (self.contract.bootstrap_host.is_empty()
                    || self.contract.bootstrap_port == 0
                    || self.contract.bootstrap_port > u32::from(u16::MAX)))
        {
            return Err(tonic::Status::invalid_argument(
                "incomplete managed capability contract",
            ));
        }
        if !matches!(self.role.as_str(), "prefill" | "decode")
            || (self.role == "prefill"
                && !self
                    .metadata_endpoint
                    .as_ref()
                    .is_some_and(|s| s.starts_with("ipc://")))
            || (self.role == "decode" && self.metadata_endpoint.is_some())
        {
            return Err(tonic::Status::invalid_argument(
                "invalid managed role/metadata endpoint",
            ));
        }
        Ok(())
    }

    /// Advertise only the managed contract; native readiness is checked at each query.
    pub fn capabilities(&self) -> Capabilities {
        use smg_grpc_client::sglang_managed::{proto::Instance, LAYOUT, VERSION};
        Capabilities {
            version: VERSION,
            instance: Some(Instance {
                worker_id: self.worker_id.clone(),
                generation: self.generation.clone(),
            }),
            role: self.role.clone(),
            model_revision: self.contract.model_revision.clone(),
            processor_contract: self.contract.processor_contract.clone(),
            layout_contract: LAYOUT.into(),
            embedding_dtype: self.contract.embedding_dtype.clone(),
            kv_backend: self.contract.kv_backend.clone(),
            embedding_backend: self.contract.embedding_backend.clone(),
            topology_contract: self.contract.topology_contract.clone(),
            bootstrap_host: self.contract.bootstrap_host.clone(),
            bootstrap_port: self.contract.bootstrap_port,
            capacity: self.max_active as u64,
            tp_size: self.contract.tp_size,
            ..Default::default()
        }
    }
}
