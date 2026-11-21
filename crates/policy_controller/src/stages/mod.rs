//! Stages, the building blocks of a policy: filters, scorers and pickers, and
//! the registry that finds one by its name.

pub(crate) mod builtins;
pub(crate) mod catalog;

pub use builtins::{
    score_costs, Best, CountFirst, FewestSessions, HasCapacity, HasRoom, Healthy, LeastCost,
    LeastLoad, LeastRequests, Manual, NotOverloaded, NotTried, Passthrough, PowerOfTwo,
    PowerOfTwoCountFirst, Random, ResidentWork, RoundRobin, SampledBest, SessionAffinity,
    DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT,
};
pub use catalog::{Catalog, CatalogError, StageBuild, StageError, StageKind};
