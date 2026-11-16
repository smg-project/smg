//! Worker domain — identity, registry, health, resilience, monitoring, service.

pub mod connection;
pub mod health;
pub mod identity;
pub mod kv;
pub mod load;
pub mod pd;
pub(crate) mod periodic_task;
pub mod registry;
pub mod runtime;
#[cfg(any(test, feature = "test-util"))]
pub mod test_support;

// Re-export commonly used types for convenience
pub use connection::http_client::WorkerHttpClientCache;
/// The liveness budgets' setter for a host that builds the context without
/// the builder, the test harness today; the first call in a process wins.
pub use health::liveness::configure_from as configure_liveness;
pub use health::{
    circuit_breaker::{CircuitBreaker, CircuitBreakerConfig},
    liveness::{budgets as liveness_budgets, Budgets as LivenessBudgets},
    manager::WorkerManager,
    overload::OverloadThresholds,
    resilience::{resolve_resilience, ResolvedResilience, DEFAULT_RETRYABLE_STATUS_CODES},
};
pub(crate) use identity::basic::ConnectionModeExt;
// `endpoint::Endpoint` is deliberately not re-exported here: `model_type::Endpoint`
// already occupies that name at this level and means an API route, not an address.
pub use identity::endpoint::{endpoint_key, EndpointError, EndpointKey};
pub use identity::{
    basic::{
        BasicWorker, ConnectionMode, RequestCompletionSink, RuntimeType, Worker, WorkerLoadGuard,
        WorkerType, DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR, MORIIO_CONNECTOR,
        MORIIO_MODE_LABEL, NIXL_CONNECTOR,
    },
    builder::BasicWorkerBuilder,
    error::{WorkerError, WorkerResult},
    metadata::DecodingLabels,
    sampling_defaults::DEFAULT_SAMPLING_PARAMS_LABEL,
};
pub use kv::{
    event_monitor::{KvEventMonitor, SubscriptionKind},
    index_backend::{KvIndex, WorkerBlocks},
};
pub use load::{
    capacity::{CapacitySource, CapacityTrackerSettings, WorkerCapacity},
    monitor::{WorkerLoadManager, WorkerMonitor},
};
// Re-export UNKNOWN_MODEL_ID from protocols
pub use openai_protocol::UNKNOWN_MODEL_ID;
pub use openai_protocol::{
    model_card::ModelCard,
    model_type::{Endpoint, ModelType},
    worker::{ProviderType, WorkerGroupKey},
};
pub(crate) use pd::pair_index::{PdPairIndex, PdWire};
pub use pd::pairing::{PairingMismatch, PdPairing, PdPairingMode};
pub use registry::{service::WorkerService, WorkerOrigin, WorkerRegistry};
pub(crate) use registry::{ModelWorkerSnapshot, RoutingPool, WindowFloor};
