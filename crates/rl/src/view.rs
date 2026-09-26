//! The registry view the gateway hands to the RL crate. This trait is the
//! entire read-side coupling to `model_gateway` (touchpoint (a) in COUPLING.md).

use std::{collections::HashMap, fmt, sync::Arc};

use openai_protocol::worker::{ConnectionMode, RuntimeType, WorkerStatus, WorkerType};

/// A snapshot of one registered worker, read from the gateway registry.
/// `Debug` reports whether `api_key` is set, never its value.
#[derive(Clone)]
pub struct RlWorkerInfo {
    /// Registry UUID.
    pub id: String,
    /// `Worker::url()`; carries an `@<rank>` suffix for DP-aware workers.
    pub url: String,
    /// `Worker::base_url()`; see `control_url` for where control calls go.
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
    /// Base URL of the worker's RL control routes: the worker itself for an
    /// HTTP worker, the engine-advertised `rl.control_url` (wildcard host
    /// resolved) for a gRPC or ZMQ worker, `None` when it has neither.
    pub control_url: Option<String>,
    /// The client used for control calls. For an HTTP worker it is the
    /// client the gateway negotiated for the worker (HTTP version, TLS
    /// identity and roots, pool tuning); for other transports it is a cached
    /// client with the same TLS settings on HTTP/1.1. `None` when the worker
    /// has no control endpoint.
    pub control_client: Option<Arc<reqwest::Client>>,
}

impl fmt::Debug for RlWorkerInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RlWorkerInfo")
            .field("id", &self.id)
            .field("url", &self.url)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("model_id", &self.model_id)
            .field("runtime", &self.runtime)
            .field("worker_type", &self.worker_type)
            .field("connection_mode", &self.connection_mode)
            .field("status", &self.status)
            .field("is_dp_aware", &self.is_dp_aware)
            .field("dp_size", &self.dp_size)
            .field("labels", &self.labels)
            .field("control_url", &self.control_url)
            .field(
                "control_client",
                &self.control_client.as_ref().map(|_| ".."),
            )
            .finish()
    }
}

/// Read-only access to the worker registry.
pub trait RlWorkerView: Send + Sync {
    /// Every registered worker, in registry order.
    fn list(&self) -> Vec<RlWorkerInfo>;
    /// One worker by registry UUID.
    fn get(&self, id: &str) -> Option<RlWorkerInfo>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RlWorkerInfo {
        RlWorkerInfo {
            id: "w".to_string(),
            url: "http://a:1".to_string(),
            base_url: "http://a:1".to_string(),
            api_key: Some("hunter2-secret".to_string()),
            model_id: "m".to_string(),
            runtime: RuntimeType::Sglang,
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
            status: WorkerStatus::Ready,
            is_dp_aware: false,
            dp_size: None,
            labels: HashMap::new(),
            control_url: None,
            control_client: None,
        }
    }

    #[test]
    fn debug_output_redacts_the_api_key() {
        let info = sample();
        let dbg = format!("{info:?}");
        assert!(!dbg.contains("hunter2-secret"), "{dbg}");
        assert!(dbg.contains("api_key: Some(\"<redacted>\")"), "{dbg}");
        assert!(dbg.contains("id: \"w\""), "{dbg}");
    }

    #[test]
    fn debug_output_shows_the_control_url() {
        let mut info = sample();
        info.control_url = Some("http://ctl:1".to_string());
        assert!(format!("{info:?}").contains("control_url: Some(\"http://ctl:1\")"));
    }
}
