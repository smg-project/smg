//! Worker domain — identity, registry, health, resilience, monitoring, service.

pub(crate) mod admin_json;
pub mod basic;
pub(crate) mod bootstrap_probe;
pub mod builder;
pub mod capacity;
pub mod circuit_breaker;
pub mod endpoint;
pub mod error;
pub mod event;
pub mod expected_wait;
pub mod hash_ring;
pub mod http_client;
pub mod kv_event_monitor;
mod kv_event_recovery;
pub mod kv_index_backend;
pub(crate) mod liveness;
pub mod load;
pub mod load_state;
pub mod manager;
pub mod metadata;
pub mod metrics_aggregator;
pub mod monitor;
pub mod overload;
pub mod pd_pair_index;
pub mod pd_pairing;
pub mod prefill_admission;
pub mod registry;
pub mod resilience;
pub mod runtime;
pub mod sampling_defaults;
pub mod service;
pub mod stream;
#[cfg(test)]
pub(crate) mod test_support;

// Re-export commonly used types for convenience
pub(crate) use basic::ConnectionModeExt;
pub use basic::{
    BasicWorker, ConnectionMode, RequestCompletionSink, RuntimeType, Worker, WorkerLoadGuard,
    WorkerType, DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR, MORIIO_CONNECTOR, MORIIO_MODE_LABEL,
    NIXL_CONNECTOR,
};
pub use builder::BasicWorkerBuilder;
pub use capacity::{CapacitySource, CapacityTrackerSettings, WorkerCapacity};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
// `endpoint::Endpoint` is deliberately not re-exported here: `model_type::Endpoint`
// already occupies that name at this level and means an API route, not an address.
pub use endpoint::{endpoint_key, EndpointError, EndpointKey};
pub use error::{WorkerError, WorkerResult};
pub use hash_ring::HashRing;
pub use http_client::WorkerHttpClientCache;
pub use kv_event_monitor::KvEventMonitor;
pub use kv_index_backend::{KvIndex, WorkerBlocks};
pub use liveness::{budgets as liveness_budgets, Budgets as LivenessBudgets};
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
pub(crate) use pd_pair_index::{PdPairIndex, PdWire};
pub use pd_pairing::{PairingMismatch, PdPairing, PdPairingMode};
pub use prefill_admission::{
    acquire_prefill, AdmittedPrefill, PrefillAcquireError, PrefillAdmission,
    PrefillAdmissionAttempt, PrefillAdmissionRejection, PrefillCandidateError, PrefillLoadGuard,
    PrefillReservation, PrefillSelection, PrefillSelectionContext,
};
pub(crate) use registry::{ModelWorkerSnapshot, RoutingPool};
pub use registry::{WorkerOrigin, WorkerRegistry};
pub use resilience::{resolve_resilience, ResolvedResilience, DEFAULT_RETRYABLE_STATUS_CODES};
pub use sampling_defaults::DEFAULT_SAMPLING_PARAMS_LABEL;
pub use service::WorkerService;
pub use stream::GuardedStream;
