//! Stages, the building blocks of a policy: filters, scorers and pickers, and
//! the registry that finds one by its name.
//!
//! The fence of retired workers is here too: the stages of
//! `consistent_hashing` and `bucket` each hold one.

pub(crate) mod catalog;
pub(crate) mod fence;
pub(crate) mod filters;
pub(crate) mod pickers;
pub(crate) mod scorers;

pub use catalog::{Catalog, CatalogError, StageBuild, StageError, StageKind};
pub use fence::FENCE;
pub use filters::{HasCapacity, HasRoom, Healthy, NotOverloaded, NotTried};
pub use pickers::{Best, CountFirst, SampledBest};
pub use scorers::{
    score_costs, FewestSessions, LeastCost, LeastRequests, ResidentWork, SessionAffinity,
    DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT,
};
