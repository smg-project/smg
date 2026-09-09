//! The registry view the gateway hands to the RL crate. This trait is the
//! entire read-side coupling to `model_gateway` (touchpoint (a) in COUPLING.md).

use std::{collections::HashMap, sync::Arc};

use openai_protocol::worker::{ConnectionMode, RuntimeType, WorkerStatus, WorkerType};

/// A snapshot of one registered worker, read from the gateway registry.
#[derive(Debug, Clone)]
pub struct RlWorkerInfo {
    /// Registry UUID.
    pub id: String,
    /// `Worker::url()`; carries an `@<rank>` suffix for DP-aware workers.
    pub url: String,
    /// `Worker::base_url()`; the address control calls are sent to.
    pub base_url: String,
    pub api_key: Option<String>,
    pub model_id: String,
    pub runtime: RuntimeType,
    pub worker_type: WorkerType,
    pub connection_mode: ConnectionMode,
    pub status: WorkerStatus,
    pub is_dp_aware: bool,
    pub dp_size: Option<usize>,
    /// `WorkerSpec.labels`: discovered metadata merged with caller labels.
    pub labels: HashMap<String, String>,
    /// The client the gateway negotiated for this worker (HTTP version,
    /// TLS identity and roots, pool tuning), shared with its data-plane and
    /// admin calls. `None` when the gateway does not speak HTTP to it.
    pub http_client: Option<Arc<reqwest::Client>>,
}

/// Read-only access to the worker registry.
pub trait RlWorkerView: Send + Sync {
    /// Every registered worker, in registry order.
    fn list(&self) -> Vec<RlWorkerInfo>;
    /// One worker by registry UUID.
    fn get(&self, id: &str) -> Option<RlWorkerInfo>;
}
