//! Worker domain — identity, registry, health, resilience, monitoring, service.

pub mod builder;
pub mod capacity;
pub mod circuit_breaker;
pub mod error;
pub mod event;
pub mod hash_ring;
pub mod http_client;
pub mod kv_event_monitor;
pub mod manager;
pub mod metrics_aggregator;
pub mod monitor;
pub mod overload;
pub mod registry;
pub mod resilience;
pub mod sampling_defaults;
pub mod service;
// FIXME: worker.rs is a 1800-line monolith containing the Worker trait,
// BasicWorker impl, HealthChecker, WorkerType, ConnectionMode, and more.
// Break it apart into focused modules (e.g. health_checker.rs, types.rs).
#[expect(
    clippy::module_inception,
    reason = "FIXME: worker.rs needs to be broken apart into focused modules"
)]
pub mod worker;

// Re-export commonly used types for convenience
pub use builder::BasicWorkerBuilder;
pub use capacity::{CapacitySource, CapacityTrackerSettings, WorkerCapacity};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
pub use error::{WorkerError, WorkerResult};
pub use hash_ring::HashRing;
pub use http_client::WorkerHttpClientCache;
pub use kv_event_monitor::KvEventMonitor;
pub use manager::WorkerManager;
pub use monitor::{WorkerLoadManager, WorkerMonitor};
// Re-export UNKNOWN_MODEL_ID from protocols
pub use openai_protocol::UNKNOWN_MODEL_ID;
pub use openai_protocol::{
    model_card::ModelCard,
    model_type::{Endpoint, ModelType},
    worker::{ProviderType, WorkerGroupKey},
};
pub use overload::OverloadThresholds;
pub use registry::{WorkerOrigin, WorkerRegistry};
pub use resilience::{resolve_resilience, ResolvedResilience, DEFAULT_RETRYABLE_STATUS_CODES};
pub use sampling_defaults::DEFAULT_SAMPLING_PARAMS_LABEL;
pub use service::WorkerService;
pub(crate) use worker::ConnectionModeExt;
pub use worker::{
    AttachedBody, BasicWorker, ConnectionMode, RuntimeType, Worker, WorkerLoadGuard, WorkerType,
    DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR, NIXL_CONNECTOR,
};
