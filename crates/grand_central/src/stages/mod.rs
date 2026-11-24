//! Stages, the building blocks of a policy: filters, scorers and pickers, and
//! the registry that finds one by its name.
//!
//! The fence of retired workers is here too: the stages of
//! `consistent_hashing` and `bucket` each hold one.
//!
//! A stateful scorer or picker hears what became of its decisions through
//! [`Lifecycle`], which the engine's `Scorer` and `Picker` traits extend.

pub(crate) mod cost_terms;
pub(crate) mod filters;
pub(crate) mod lifecycle;
pub(crate) mod pickers;
pub(crate) mod registry;
pub(crate) mod retired_workers;
pub(crate) mod scorers;

pub use filters::{HasCapacity, HasRoom, Healthy, NotOverloaded, NotTried};
pub use lifecycle::Lifecycle;
pub use pickers::{Best, CountFirst, SampledBest};
pub use registry::{StageBuild, StageError, StageKind, StageRegistry, StageRegistryError};
pub use retired_workers::RETIRED_WORKER_TTL;
pub use scorers::{
    score_costs, FewestSessions, LeastCost, LeastRequests, ResidentWork, SessionAffinity,
    DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT,
};
