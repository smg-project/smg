//! Worker domain — identity, registry, health, resilience, monitoring, service.

pub mod connection;
pub mod health;
pub mod identity;
pub mod kv;
pub mod load;
pub mod metrics;
pub mod pd;
// The worker crate's, at the path it had here.
pub(crate) use smg_worker::periodic_task;
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
    manager::{WorkerJobs, WorkerManager},
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
    monitor::{LoadConsumer, WorkerLoadManager},
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
pub(crate) use registry::{ModelLabels, ModelWorkerSnapshot, RoutingPool, WindowFloor};
pub use registry::{RemoteWorkerState, WorkerOrigin};

// The gateway's connector, for the gateway's name for the builder below.
use crate::routers::grpc::backend_client::BackendConnector;

/// The worker builder as the gateway names it: a worker it builds connects
/// through the gateway's client unless the builder is handed another
/// connector. The two lines are the gateway's, and stay with it when the
/// runtime moves into the worker crate.
pub type BasicWorkerBuilder = identity::builder::BasicWorkerBuilder<BackendConnector>;

/// The registry as the gateway names it: its value of a membership
/// generation is the model snapshot that keeps the routers' projections.
/// The load monitor holds that registry. The two lines are the gateway's,
/// and stay with it when the registry moves into the worker crate.
pub type WorkerRegistry = registry::WorkerRegistry<ModelWorkerSnapshot>;
pub type WorkerMonitor = load::monitor::WorkerMonitor<ModelWorkerSnapshot>;
